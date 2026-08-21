# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thread-safety tests for the native disaggregated KV-transfer DEALER sockets.

pyzmq sockets are not thread-safe. Sender worker threads must send only on
their own threading.local DEALER; the shared ``_dealers`` dict is reached from
both the listener and the executor thread and must be locked.
"""

from __future__ import annotations

import queue
import threading
import time
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import numpy as np

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _AGENT_RESULT_BY_CODE,
    _KV_RESULT_PREFIX,
    AgentResult,
    MessageType,
    Receiver,
    Sender,
    TaskStatus,
    WriteMeta,
)

_ENDPOINT = "tcp://127.0.0.1:15555"


class _FakeDealer:
    """Stand-in for ZMQMessenger(mode="DEALER"): records who built and used it."""

    def __init__(self, mode: str, endpoint: Optional[str] = None) -> None:
        self.mode = mode
        self.endpoint = endpoint
        self.created_by = threading.current_thread().name
        self.sent: list[list[bytes]] = []
        self.sent_by: list[str] = []
        self.stopped = False

    def send(self, messages: list[bytes], recipient: Optional[bytes] = None) -> None:
        self.sent.append(list(messages))
        self.sent_by.append(threading.current_thread().name)

    def stop(self) -> None:
        self.stopped = True


class _DealerFactory:
    """Replaces transfer.ZMQMessenger so no real socket is ever opened."""

    def __init__(self, construct_delay_s: float = 0.0) -> None:
        self.created: list[_FakeDealer] = []
        self._delay_s = construct_delay_s
        self._lock = threading.Lock()

    def __call__(self, mode: str, endpoint: Optional[str] = None) -> _FakeDealer:
        # Widen the check-then-create window so an unsynchronized
        # get-or-connect reliably double-constructs.
        if self._delay_s:
            time.sleep(self._delay_s)
        dealer = _FakeDealer(mode, endpoint)
        with self._lock:
            self.created.append(dealer)
        return dealer


class _FakeSendTask:
    def __init__(self) -> None:
        self.status = TaskStatus.INIT
        self._perf_timer = None
        self.lock = threading.Lock()
        self.transferred_count = 0
        self.failed_with: Optional[Exception] = None
        self._done = False

    def fail(self, exc: Exception) -> None:
        self.failed_with = exc
        self.status = TaskStatus.ERROR
        self._done = True

    def complete(self) -> None:
        self.status = TaskStatus.TRANSFERRED
        self._done = True

    @property
    def is_done(self) -> bool:
        return self._done

    def print_perf_info(self, *_args) -> None:
        pass


class _FakeTxSession:
    def __init__(self, status: SessionStatus, kv_tasks: list[_FakeSendTask]) -> None:
        self.status = status
        self.kv_tasks = kv_tasks
        self.lock = threading.Lock()
        self.exception_reason: Optional[str] = None

    def set_exception(self, reason: str = "") -> None:
        self.exception_reason = reason


def _run_threads(targets: list[tuple[str, Callable[[], object]]]) -> list[object]:
    """Run each (thread_name, fn) concurrently; re-raise the first failure here."""
    results: dict[int, object] = {}
    errors: list[BaseException] = []

    def wrap(idx: int, fn: Callable[[], object]) -> Callable[[], None]:
        def run() -> None:
            try:
                results[idx] = fn()
            except BaseException as exc:  # surfaced on the test thread below
                errors.append(exc)

        return run

    threads = [
        threading.Thread(target=wrap(i, fn), name=name) for i, (name, fn) in enumerate(targets)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "a worker thread did not finish"
    if errors:
        raise errors[0]
    return [results.get(i) for i in range(len(targets))]


def _make_sender(num_threads: int = 1) -> Sender:
    sender = object.__new__(Sender)
    sender._dealers = {}
    # Present even on an unpatched Sender: harmless there, so this test file
    # runs identically against main and only the assertions differ.
    sender._dealers_lock = threading.Lock()
    sender._thread_local = threading.local()
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._instance_rank = 3
    sender._device_id = 0
    sender._bounce = None
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=3)
    )
    sender._num_threads = num_threads
    sender._send_task_queues = [queue.Queue() for _ in range(num_threads)]
    return sender


def _make_receiver() -> Receiver:
    receiver = object.__new__(Receiver)
    receiver._dealers = {}
    receiver._dealers_lock = threading.Lock()
    return receiver


def _make_write_meta(task: _FakeSendTask, *, unique_rid: int = 7, slice_id: int = 0) -> WriteMeta:
    ptrs = np.array([4096], dtype=np.int64)
    return WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen0",
        peer_rank=0,
        peer_endpoint=_ENDPOINT,
        unique_rid=unique_rid,
        src_ptrs=ptrs,
        dst_ptrs=ptrs,
        sizes=np.array([16], dtype=np.int64),
        dst_device_id=0,
        slice_id=slice_id,
        is_last_slice=True,
    )


def _assert_failed_kv_result(dealer: _FakeDealer, *, unique_rid: int, slice_id: int) -> None:
    assert len(dealer.sent) == 1
    message = dealer.sent[0]
    assert message[0] == MessageType.KV_AGENT_RESULT
    rank, rid, sent_slice_id, is_last, status_code, _size = _KV_RESULT_PREFIX.unpack(message[1])
    assert (rank, rid, sent_slice_id, is_last) == (3, unique_rid, slice_id, True)
    assert _AGENT_RESULT_BY_CODE[status_code] == AgentResult.FAILED


def test_cancelled_session_abort_sends_on_the_worker_thread_local_dealer(monkeypatch) -> None:
    # Abort path 1: session already CANCELLED. Runs on a worker thread, so it
    # must not touch the shared _dealers dict that the listener also sends on.
    factory = _DealerFactory()
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    sender = _make_sender()
    task = _FakeSendTask()
    session = _FakeTxSession(SessionStatus.CANCELLED, [task])
    sender._sessions[7] = weakref.ref(session)
    write_meta = _make_write_meta(task)

    def abort_on_worker() -> dict:
        sender._deliver_kv_to_agent(write_meta)
        # threading.local is only readable from its owning thread.
        return dict(getattr(sender._thread_local, "dealers", {}))

    (thread_dealers,) = _run_threads([("kv-worker-0", abort_on_worker)])

    assert sender._dealers == {}, "worker thread must not populate the shared dealer dict"
    assert len(factory.created) == 1
    dealer = factory.created[0]
    assert thread_dealers == {_ENDPOINT: dealer}
    assert dealer.created_by == "kv-worker-0"
    assert dealer.sent_by == ["kv-worker-0"]
    _assert_failed_kv_result(dealer, unique_rid=7, slice_id=0)
    assert task.status == TaskStatus.ERROR


def test_build_send_request_failure_abort_uses_the_worker_thread_local_dealer(
    monkeypatch,
) -> None:
    # Abort path 2: build_send_request raised. Same worker thread, same rule.
    from tensorrt_llm._torch.disaggregation.native import bounce as bounce_mod

    factory = _DealerFactory()
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    monkeypatch.setattr(
        bounce_mod, "build_send_request", Mock(side_effect=RuntimeError("gather fault"))
    )
    sender = _make_sender()
    task = _FakeSendTask()
    session = _FakeTxSession(SessionStatus.TRANSFERRING, [task])
    sender._sessions[7] = weakref.ref(session)
    write_meta = _make_write_meta(task)

    def abort_on_worker() -> dict:
        sender._deliver_kv_to_agent(write_meta)
        return dict(getattr(sender._thread_local, "dealers", {}))

    (thread_dealers,) = _run_threads([("kv-worker-0", abort_on_worker)])

    assert sender._dealers == {}, "worker thread must not populate the shared dealer dict"
    assert len(factory.created) == 1
    dealer = factory.created[0]
    assert thread_dealers == {_ENDPOINT: dealer}
    assert dealer.sent_by == ["kv-worker-0"]
    _assert_failed_kv_result(dealer, unique_rid=7, slice_id=0)
    assert task.status == TaskStatus.ERROR
    assert "build_send_request failed" in str(task.failed_with)


def test_concurrent_worker_threads_each_abort_on_their_own_dealer(monkeypatch) -> None:
    # The N-worker case that raising TRTLLM_KV_TRANSFER_NUM_THREADS creates:
    # three workers aborting the same peer endpoint must use three sockets.
    factory = _DealerFactory()
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    sender = _make_sender(num_threads=3)
    tasks = [_FakeSendTask() for _ in range(3)]
    session = _FakeTxSession(SessionStatus.CANCELLED, tasks)
    sender._sessions[7] = weakref.ref(session)
    metas = [_make_write_meta(tasks[i], slice_id=i) for i in range(3)]

    def abort(idx: int) -> Callable[[], None]:
        return lambda: sender._deliver_kv_to_agent(metas[idx])

    _run_threads([(f"kv-worker-{i}", abort(i)) for i in range(3)])

    assert sender._dealers == {}
    assert len(factory.created) == 3, "each worker thread needs its own DEALER socket"
    assert {d.created_by for d in factory.created} == {
        "kv-worker-0",
        "kv-worker-1",
        "kv-worker-2",
    }
    # Every socket was used only by the thread that owns it.
    assert all(d.sent_by == [d.created_by] for d in factory.created)


def test_worker_teardown_closes_the_dealer_the_abort_path_used(monkeypatch) -> None:
    # Socket lifetime: the abort send now lands on the thread-local socket, so
    # _process_task_queue's finally block must be what closes it.
    factory = _DealerFactory()
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    monkeypatch.setattr(transfer_mod, "CUASSERT", lambda *_a, **_k: None)
    monkeypatch.setattr(transfer_mod, "cudart", Mock())
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    sender = _make_sender()
    task = _FakeSendTask()
    session = _FakeTxSession(SessionStatus.CANCELLED, [task])
    sender._sessions[7] = weakref.ref(session)
    sender._send_task_queues[0].put(_make_write_meta(task))
    sender._send_task_queues[0].put(None)  # shutdown sentinel

    worker = threading.Thread(target=sender._process_task_queue, args=(0,), name="kv-worker-0")
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert sender._dealers == {}, "the abort must not leak into the shared dict"
    assert len(factory.created) == 1
    dealer = factory.created[0]
    assert dealer.sent_by == ["kv-worker-0"]
    assert dealer.stopped, "worker finally must close the socket its abort path used"


def test_sender_shared_dealer_connect_is_serialized(monkeypatch) -> None:
    # The listener and executor threads genuinely share _dealers (listener:
    # CANCEL_SESSION; executor: cancel_request -> TxSession.cancel), so the
    # get-or-connect must be atomic or one socket is built and then dropped.
    factory = _DealerFactory(construct_delay_s=0.05)
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    sender = _make_sender()
    barrier = threading.Barrier(2)

    def connect() -> object:
        barrier.wait(timeout=10)
        return sender._get_or_connect_dealer(_ENDPOINT)

    first, second = _run_threads([("listener", connect), ("executor", connect)])

    assert len(factory.created) == 1, "double-connect leaks a socket and loses one"
    assert first is second
    assert sender._dealers == {_ENDPOINT: factory.created[0]}


def test_sender_shutdown_detaches_dealers_under_the_lock(monkeypatch) -> None:
    factory = _DealerFactory()
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    sender = _make_sender()
    dealer = sender._get_or_connect_dealer(_ENDPOINT)
    sender._shutdown = False
    sender._messenger = Mock()
    sender._worker_threads = []
    sender._loaded_remote_agents = set()
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()

    sender.shutdown()

    assert dealer.stopped
    assert sender._dealers == {}
    # A late cancel after shutdown reconnects rather than reusing a dead socket.
    assert sender._get_or_connect_dealer(_ENDPOINT) is not dealer


def test_receiver_executor_and_listener_share_exactly_one_dealer(monkeypatch) -> None:
    # Receiver side of the same defect class: dispatch_task runs on the executor
    # thread, send_cancel_to_senders runs on the listener thread (CANCEL_SESSION).
    factory = _DealerFactory(construct_delay_s=0.05)
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", factory)
    receiver = _make_receiver()
    barrier = threading.Barrier(2)

    def executor() -> None:
        barrier.wait(timeout=10)
        receiver._request_sender_data(_ENDPOINT, b"payload")

    def listener() -> None:
        barrier.wait(timeout=10)
        receiver.send_cancel_to_senders(9, {_ENDPOINT})

    _run_threads([("executor", executor), ("listener", listener)])

    assert len(factory.created) == 1, "double-connect leaks a socket and loses one"
    dealer = factory.created[0]
    assert sorted(dealer.sent_by) == ["executor", "listener"]
    assert {message[0] for message in dealer.sent} == {
        MessageType.REQUEST_DATA,
        MessageType.CANCEL_SESSION,
    }


class _ReentrancyDetectingSocket:
    """Records the peak number of threads simultaneously inside send_multipart."""

    def __init__(self, hold_s: float = 0.05) -> None:
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.sent: list[list[bytes]] = []

    def send_multipart(self, frames: list[bytes]) -> None:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        time.sleep(self._hold_s)
        with self._lock:
            self.sent.append(list(frames))
            self._in_flight -= 1


def test_zmq_messenger_send_serializes_concurrent_callers() -> None:
    # _send_lock mirrors what __init__ installs (object.__new__ skips __init__,
    # so no real socket is opened). A send() that ignores it lets all four
    # threads into send_multipart at once and interleaves their frames.
    messenger = object.__new__(ZMQMessenger)
    messenger._send_lock = threading.Lock()
    socket = _ReentrancyDetectingSocket()
    messenger._socket = socket

    def send(idx: int) -> Callable[[], None]:
        return lambda: messenger.send([b"FRAME_A", str(idx).encode("ascii")])

    _run_threads([(f"sender-{i}", send(i)) for i in range(4)])

    assert socket.max_in_flight == 1, "two threads were inside send_multipart at once"
    assert len(socket.sent) == 4
    assert all(frames[0] == b"FRAME_A" for frames in socket.sent)

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
"""Bounded polling tests for KvCacheTransceiverV2 Tx sessions and the sender's agent wait."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.agent import AgentWaitState
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _AGENT_POLL_SLICE_MS,
    _KV_RESULT_PREFIX,
    _MIN_AGENT_WAIT_S,
    AgentResult,
    AuxSendTask,
    KVSendTask,
    MessageType,
    Sender,
    TaskStatus,
    TransferWorker,
    TransferWorkerConfig,
    TxSession,
    WriteMeta,
    WriteMetaType,
    _agent_wait_deadline_s,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState


@dataclass
class _FakeRequest:
    state: Optional[LlmRequestState] = None


class _FakeTransferWorker:
    def __init__(self) -> None:
        self.sweep_count = 0

    def sweep_stale_req_infos(self) -> None:
        self.sweep_count += 1


class _FakeSession:
    def __init__(
        self,
        rid: int,
        wait_result: Optional[WaitResult],
        *,
        status: SessionStatus = SessionStatus.READY,
        is_completed: bool = False,
        has_failed: bool = False,
    ) -> None:
        self._rid = rid
        self._wait_result = wait_result
        self._status = status
        self._is_completed = is_completed
        self._has_failed = has_failed
        self.blocking_calls: list[bool] = []
        self.closed = False
        self.aux_slot: Optional[int] = 0

    @property
    def disagg_request_id(self) -> int:
        return self._rid

    @property
    def status(self) -> SessionStatus:
        return self._status

    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]:
        self.blocking_calls.append(blocking)
        return self._wait_result

    def is_completed(self) -> bool:
        return self._is_completed

    def has_failed(self) -> bool:
        return self._has_failed

    def close(self) -> None:
        self.closed = True
        self.aux_slot = None


class _FakeTask:
    def __init__(
        self,
        status: TaskStatus,
        wait_result: bool | list[bool] = True,
        on_wait: Optional[Callable[[Optional[float]], None]] = None,
    ) -> None:
        self.status = status
        self._wait_results = list(wait_result) if isinstance(wait_result, list) else [wait_result]
        self._on_wait = on_wait
        self.wait_calls: list[Optional[float]] = []

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.wait_calls.append(timeout)
        if self._on_wait is not None:
            self._on_wait(timeout)
        result = self._wait_results.pop(0) if len(self._wait_results) > 1 else self._wait_results[0]
        if result and self.status != TaskStatus.ERROR:
            self.status = TaskStatus.TRANSFERRED
        return result


class _FakeClock:
    def __init__(self, now_s: float = 0.0) -> None:
        self.now_s = now_s

    def monotonic(self) -> float:
        return self.now_s

    def advance(self, elapsed_s: Optional[float]) -> None:
        assert elapsed_s is not None
        self.now_s += elapsed_s


def _make_transceiver(
    sessions: dict[int, _FakeSession],
    reqs: Optional[dict[int, _FakeRequest]] = None,
) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_sessions = sessions
    transceiver._send_reqs = reqs or {rid: _FakeRequest() for rid in sessions}
    transceiver._sender_future_timeout_ms = 123
    transceiver.kv_transfer_timeout_ms = 60_000
    # Attributes read by check_context_transfer_status before it processes sessions.
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = lambda local_ids: list(local_ids)
    transceiver._ctx_consensus_outcome = lambda _to_process, cancelled, failed, completed: (
        cancelled,
        failed,
        completed,
    )
    return transceiver


def _make_tx_session(
    kv_tasks: list[_FakeTask],
    *,
    need_aux: bool = False,
    aux_task: Optional[_FakeTask] = None,
    timeout_s: Optional[float] = 0.25,
    deadline_monotonic_s: Optional[float] = None,
) -> TxSession:
    session = object.__new__(TxSession)
    session._timeout_s = timeout_s
    session._overall_timeout_s = None
    session._deadline_monotonic_s = deadline_monotonic_s
    session._need_aux = need_aux
    session._terminal_status = None
    session._exception = None
    session.receiver_ready = True
    session.kv_tasks = kv_tasks
    session.aux_task = aux_task
    session.lock = threading.Lock()
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = None
    return session


def test_context_transfer_status_bounded_poll_keeps_not_ready_session_queued(
    monkeypatch,
) -> None:
    session = _FakeSession(rid=11, wait_result=None)
    transceiver = _make_transceiver({11: session})
    monotonic = Mock(side_effect=[0.0, 0.0, 0.123])
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.monotonic",
        monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.sleep",
        Mock(),
    )

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=1)

    assert completed == []
    assert failed == []
    assert session.blocking_calls == [False]
    assert not session.closed
    assert 11 in transceiver._send_sessions
    assert 11 in transceiver._send_reqs
    assert transceiver._transfer_worker.sweep_count == 1


def test_context_transfer_status_bounded_poll_reaps_completion(monkeypatch) -> None:
    session = _FakeSession(rid=14, wait_result=WaitResult.COMPLETED)
    req = _FakeRequest()
    transceiver = _make_transceiver({14: session}, {14: req})

    def complete_on_poll(blocking: bool = True) -> WaitResult:
        session.blocking_calls.append(blocking)
        session._is_completed = True
        return WaitResult.COMPLETED

    session.wait_complete = complete_on_poll
    sleep = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.monotonic",
        Mock(return_value=0.0),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.sleep",
        sleep,
    )

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=1)

    assert completed == [14]
    assert failed == []
    assert session.blocking_calls == [False, False]
    sleep.assert_called_once_with(0.001)
    assert session.closed
    assert 14 not in transceiver._send_sessions
    assert 14 not in transceiver._send_reqs


def test_context_transfer_status_block_all_uses_blocking_wait() -> None:
    session = _FakeSession(rid=12, wait_result=WaitResult.COMPLETED)
    req = _FakeRequest()
    transceiver = _make_transceiver({12: session}, {12: req})

    completed, failed = transceiver.check_context_transfer_status(
        at_least_request_num=None,
        mark_complete=True,
    )

    assert completed == [12]
    assert failed == []
    assert session.blocking_calls == [True]
    assert session.closed
    assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert 12 not in transceiver._send_sessions
    assert 12 not in transceiver._send_reqs


def test_context_transfer_status_timeout_retains_session_and_request(monkeypatch) -> None:
    session = _FakeSession(rid=16, wait_result=WaitResult.TIMEOUT)
    req = _FakeRequest()
    transceiver = _make_transceiver({16: session}, {16: req})
    transceiver.kv_transfer_timeout_ms = 60_000
    warning = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.logger.warning",
        warning,
    )

    completed, failed = transceiver.check_context_transfer_status(None)

    completed_again, failed_again = transceiver.check_context_transfer_status(None)

    assert completed == []
    assert failed == []
    assert completed_again == []
    assert failed_again == []
    assert session.blocking_calls == [True, True]
    assert not session.closed
    assert session.aux_slot == 0
    assert transceiver._send_sessions == {16: session}
    assert transceiver._send_reqs == {16: req}
    assert warning.call_count == 2
    messages = [args[0] for args, _kwargs in warning.call_args_list]
    assert all("rid=16" in message for message in messages)
    assert all("kv_transfer_timeout_ms=60000ms" in message for message in messages)
    assert all("keeping it in progress" in message for message in messages)


def test_context_transfer_status_zero_budget_processes_task_level_failure() -> None:
    session = _FakeSession(
        rid=13,
        wait_result=WaitResult.FAILED,
        has_failed=True,
    )
    req = _FakeRequest()
    transceiver = _make_transceiver({13: session}, {13: req})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == [13]
    assert session.blocking_calls == [False]
    assert session.closed
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert 13 not in transceiver._send_sessions
    assert 13 not in transceiver._send_reqs


def _patch_poll_clock(monkeypatch) -> tuple[_FakeClock, list[float]]:
    clock = _FakeClock()
    sleeps: list[float] = []

    def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.monotonic", clock.monotonic
    )
    monkeypatch.setattr("tensorrt_llm._torch.disaggregation.transceiver.time.sleep", record_sleep)
    return clock, sleeps


def test_poll_sessions_for_interval_returns_when_target_is_unreachable(monkeypatch) -> None:
    # A rank owning fewer sessions than the wait target can never reach it, so it must return
    # instead of burning the whole poll interval and stalling its peers at the next collective.
    _clock, sleeps = _patch_poll_clock(monkeypatch)
    transceiver = _make_transceiver({})

    transceiver._poll_sessions_for_interval({}, {}, wait_num=1, poll_interval_ms=5000)

    assert sleeps == []


def test_poll_sessions_for_interval_still_waits_out_pending_session(monkeypatch) -> None:
    _clock, sleeps = _patch_poll_clock(monkeypatch)
    session = _FakeSession(rid=21, wait_result=None)
    transceiver = _make_transceiver({21: session})

    transceiver._poll_sessions_for_interval(
        transceiver._send_sessions, transceiver._send_reqs, wait_num=1, poll_interval_ms=5
    )

    assert sleeps
    assert session.blocking_calls == [False] * len(sleeps)


def test_poll_sessions_for_interval_clamps_target_to_owned_sessions(monkeypatch) -> None:
    # An over-large target is clamped rather than abandoned: waiting for every session this rank
    # owns is the most progress it can make, so it still polls instead of returning immediately.
    _clock, sleeps = _patch_poll_clock(monkeypatch)
    session = _FakeSession(rid=22, wait_result=None)
    transceiver = _make_transceiver({22: session})

    transceiver._poll_sessions_for_interval(
        transceiver._send_sessions, transceiver._send_reqs, wait_num=3, poll_interval_ms=5
    )

    assert sleeps
    assert session.blocking_calls == [False] * len(sleeps)


def test_context_transfer_status_skips_consensus_when_never_sent() -> None:
    # A worker that never sends skips the ctx consensus even when TP sync would need it, but still
    # sweeps so nothing leaks.
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    transceiver._ctx_consensus.assert_not_called()
    assert transceiver._transfer_worker.sweep_count == 1


def test_context_transfer_status_never_sent_no_sync_is_a_noop() -> None:
    # With no tp/pp sync (e.g. attention_dp), a never-sent worker skips the consensus and the sweep,
    # unchanged from before -- a true no-op, so the fix can't slow attention_dp workers.
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    assert transceiver.check_context_transfer_status(at_least_request_num=0) == ([], [])
    transceiver._ctx_consensus.assert_not_called()
    assert transceiver._transfer_worker.sweep_count == 0  # matches the original early-out exactly


def test_gen_transfer_status_enters_consensus_when_sync_required() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = False
    transceiver._gen_need_sync = True
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._gen_consensus = Mock(return_value=[])
    transceiver._build_to_process = Mock(return_value=[])
    transceiver._gen_consensus_outcome = Mock(return_value=([], [], []))
    transceiver._close_failed_sessions = Mock()

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == []
    transceiver._gen_consensus.assert_called_once_with([])


def test_consensus_outcome_uses_single_batched_allgather() -> None:
    # The cancelled/failed/completed id lists are exchanged with ONE allgather
    # (packed as a list-of-lists) instead of three; verify a single call and that
    # union (cancelled/failed) + intersection (completed) semantics are preserved.
    transceiver = object.__new__(KvCacheTransceiverV2)
    calls: list = []

    def fake_allgather(payload):
        calls.append(payload)
        # rank0 = this rank's [cancelled, failed, completed]; rank1 = a peer rank.
        return [payload, [[], [99], [7, 8]]]

    to_process = [1, 2, 7, 8, 99]
    new_cancelled, new_failed, new_completed = transceiver._consensus_outcome(
        to_process, [1], [2], [7], fake_allgather, True
    )

    assert len(calls) == 1  # batched: a single allgather, not three
    assert calls[0] == [[1], [2], [7]]
    assert new_cancelled == [1]  # union of cancelled across ranks
    assert new_failed == [2, 99]  # union of failed across ranks
    assert new_completed == [7]  # intersection only (8 is completed on the peer only)


def test_ctx_tp_consensus_does_not_complete_when_peer_times_out() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._dist = SimpleNamespace(
        tp_allgather=lambda payload: [payload, [[], [], []]],
    )

    cancelled, failed, completed = transceiver._ctx_consensus_outcome([21], [], [], [21])

    assert cancelled == []
    assert failed == []
    assert completed == []


def test_ctx_pp_consensus_does_not_complete_when_peer_times_out() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = True
    transceiver._dist = SimpleNamespace(
        tp_allgather=Mock(side_effect=AssertionError("TP allgather must be skipped")),
        pp_allgather=lambda payload: [payload, [[], [], []]],
    )

    cancelled, failed, completed = transceiver._ctx_consensus_outcome([22], [], [], [22])

    assert cancelled == []
    assert failed == []
    assert completed == []
    transceiver._dist.tp_allgather.assert_not_called()


@pytest.mark.skip(
    reason="ctx idle fast-path was dropped from this branch. TODO: when the "
    "fast-path is reintroduced, its terminal-count reduction must mirror "
    "_ctx_consensus()'s communicator scope (TP group, then PP group; TP "
    "skipped under attention DP) — a WORLD-scoped allreduce hangs under "
    "ADP+PP because independent attention-DP lanes poll on their own "
    "schedules. Re-enable this test and add scoped mock coverage for the "
    "TP+PP and ADP+PP configurations plus real-collective MP tests."
)
def test_ctx_consensus_fastpath_skips_when_idle(monkeypatch) -> None:
    # With the fast-path enabled, an all-zero terminal count (one fixed-size
    # allreduce) makes every rank skip the variable-length consensus; a non-zero
    # count falls through to the normal consensus path.
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver._CTX_CONSENSUS_FASTPATH", True
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._dist = Mock()
    transceiver._dist.allreduce = Mock(return_value=0)
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._build_to_process = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], []))
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._close_failed_sessions = Mock()

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == [] and failed == []
    transceiver._dist.allreduce.assert_called_once()
    transceiver._ctx_consensus.assert_not_called()  # idle fast-path skipped the consensus

    # Non-zero global terminal count => fast-path does not skip; consensus runs.
    transceiver._dist.allreduce = Mock(return_value=2)
    transceiver.check_context_transfer_status(at_least_request_num=0)
    transceiver._ctx_consensus.assert_called_once()


def test_tx_session_blocking_wait_retries_wait_slices_until_complete() -> None:
    task = _FakeTask(TaskStatus.INIT, wait_result=[False, True])
    session = _make_tx_session([task])

    assert session.wait_complete() == WaitResult.COMPLETED
    assert task.wait_calls == [0.25, 0.25]


def test_tx_session_blocking_wait_times_out_stalled_task_and_does_not_reset_deadline(
    monkeypatch,
) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False, on_wait=clock.advance)
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.6,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert task.wait_calls == pytest.approx([0.25, 0.25, 0.1])
    assert not session._closed
    assert not session.has_failed()

    wait_call_count = len(task.wait_calls)
    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert len(task.wait_calls) == wait_call_count


def test_tx_session_blocking_wait_uses_finite_overall_fallback_when_unset(
    monkeypatch,
) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False, on_wait=clock.advance)
    session = _make_tx_session([task], timeout_s=0.25)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer._FALLBACK_TX_OVERALL_TIMEOUT_S",
        0.6,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert task.wait_calls == pytest.approx([0.25, 0.25, 0.1])
    assert session._deadline_monotonic_s == pytest.approx(0.6)

    wait_call_count = len(task.wait_calls)
    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert len(task.wait_calls) == wait_call_count


def test_tx_session_completion_observed_at_deadline_wins_over_timeout(monkeypatch) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)

    def complete_at_deadline(timeout_s: Optional[float]) -> None:
        clock.advance(timeout_s)
        task.status = TaskStatus.TRANSFERRED

    task._on_wait = complete_at_deadline
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.COMPLETED
    assert task.wait_calls == [0.25]


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        ("completed", WaitResult.COMPLETED),
        ("failed", WaitResult.FAILED),
        ("cancelled", WaitResult.FAILED),
    ],
)
def test_tx_session_terminal_transition_during_deadline_read_wins_over_timeout(
    monkeypatch,
    terminal: str,
    expected: WaitResult,
) -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )

    def expire_after_terminal_transition() -> float:
        if terminal == "completed":
            task.status = TaskStatus.TRANSFERRED
        elif terminal == "failed":
            task.status = TaskStatus.ERROR
        else:
            session._terminal_status = SessionStatus.CANCELLED
        return 0.25

    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        expire_after_terminal_transition,
    )

    assert session.wait_complete(blocking=True) == expected
    assert task.wait_calls == []


def test_tx_session_failure_observed_at_deadline_wins_over_timeout(monkeypatch) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)

    def fail_at_deadline(timeout_s: Optional[float]) -> None:
        clock.advance(timeout_s)
        task.status = TaskStatus.ERROR

    task._on_wait = fail_at_deadline
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [0.25]


def test_tx_session_cancellation_observed_at_deadline_wins_over_timeout(monkeypatch) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )

    def cancel_at_deadline(timeout_s: Optional[float]) -> None:
        clock.advance(timeout_s)
        session._terminal_status = SessionStatus.CANCELLED

    task._on_wait = cancel_at_deadline
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [0.25]


def test_tx_session_tasks_and_aux_share_one_deadline(monkeypatch) -> None:
    clock = _FakeClock()
    first_wait_durations = iter([0.5, 0.1])

    def advance_first_task(_timeout_s: Optional[float]) -> None:
        clock.advance(next(first_wait_durations))

    first_task = _FakeTask(
        TaskStatus.TRANSFERRING,
        wait_result=[False, True],
        on_wait=advance_first_task,
    )
    second_task = _FakeTask(
        TaskStatus.TRANSFERRING,
        wait_result=True,
        on_wait=lambda _timeout_s: clock.advance(0.2),
    )
    aux_task = _FakeTask(
        TaskStatus.TRANSFERRING,
        wait_result=False,
        on_wait=clock.advance,
    )
    session = _make_tx_session(
        [first_task, second_task],
        need_aux=True,
        aux_task=aux_task,
        timeout_s=0.5,
        deadline_monotonic_s=1.0,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert first_task.wait_calls == pytest.approx([0.5, 0.5])
    assert second_task.wait_calls == pytest.approx([0.4])
    assert aux_task.wait_calls == pytest.approx([0.2])


def test_tx_session_first_send_anchors_deadline_once(monkeypatch) -> None:
    clock = _FakeClock(now_s=10.0)
    sender = Mock()
    sender._get_req_info.return_value = {}
    params = SimpleNamespace(
        schedule_style="CONTEXT_FIRST",
        disagg_request_id=31,
        ctx_request_id=None,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.tensorrt_llm.bindings.global_steady_clock_now",
        Mock(return_value=123),
    )
    session = TxSession(
        request_id=31,
        params=params,
        sender=sender,
        timeout_s=0.25,
        overall_timeout_s=2.0,
    )

    assert session._deadline_monotonic_s is None
    session.send(Mock())
    assert session._deadline_monotonic_s == 12.0

    clock.advance(0.5)
    session.send(Mock())
    assert session._deadline_monotonic_s == 12.0
    assert sender.dispatch_task.call_count == 2
    session.close()


@pytest.mark.parametrize(
    ("transfer_timeout_ms", "sender_wait_ms", "expected_timeout_s", "expected_slice_s"),
    [
        (60_000, 1_000, 60.0, 1.0),
        (60_000, None, 60.0, None),
    ],
)
def test_transceiver_wires_separate_sender_slice_and_overall_timeout(
    monkeypatch,
    transfer_timeout_ms: Optional[int],
    sender_wait_ms: Optional[int],
    expected_timeout_s: Optional[float],
    expected_slice_s: Optional[float],
) -> None:
    worker = SimpleNamespace(page_table=None)
    worker_constructor = Mock(return_value=worker)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.TransferWorker",
        worker_constructor,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.create_cache_reuse_adapter",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.bounce_config_from_size",
        Mock(return_value=None),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.torch.cuda.current_device",
        Mock(return_value=0),
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_broadcast_instance_name",
        lambda _self: "ctx",
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_broadcast_context_endpoint",
        lambda _self: "endpoint",
    )
    monkeypatch.setattr(KvCacheTransceiverV2, "_init_sync_policy", lambda _self: None)
    monkeypatch.setattr(KvCacheTransceiverV2, "_exchange_rank_info", lambda _self: None)
    mapping = SimpleNamespace(
        cp_size=1,
        tp_rank=0,
        tp_size=1,
        enable_attention_dp=False,
    )
    cache_config = SimpleNamespace(
        kv_transfer_timeout_ms=transfer_timeout_ms,
        kv_transfer_poll_interval_ms=5_000,
        kv_transfer_sender_future_timeout_ms=sender_wait_ms,
        kv_cache_bounce_size_mb=0,
    )

    KvCacheTransceiverV2(
        mapping=mapping,
        dist=Mock(),
        kv_cache_manager=SimpleNamespace(max_batch_size=4),
        cache_transceiver_config=cache_config,
    )

    worker_config = worker_constructor.call_args.args[0]
    assert isinstance(worker_config, TransferWorkerConfig)
    assert worker_config.tx_timeout_s == expected_slice_s
    assert worker_config.tx_overall_timeout_s == expected_timeout_s
    assert worker_config.rx_timeout_s == expected_timeout_s


def test_transceiver_rejects_unset_transfer_timeout() -> None:
    cache_config = SimpleNamespace(
        kv_transfer_timeout_ms=None,
        kv_transfer_poll_interval_ms=5_000,
        kv_transfer_sender_future_timeout_ms=1_000,
    )

    with pytest.raises(
        ValueError,
        match="KvCacheTransceiverV2 requires a finite kv_transfer_timeout_ms",
    ):
        KvCacheTransceiverV2(
            mapping=Mock(),
            dist=Mock(),
            kv_cache_manager=Mock(),
            cache_transceiver_config=cache_config,
        )


def test_transfer_worker_passes_overall_timeout_to_tx_session(monkeypatch) -> None:
    session_constructor = Mock(return_value=Mock())
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.TxSession",
        session_constructor,
    )
    worker = object.__new__(TransferWorker)
    worker._config = TransferWorkerConfig(
        kv_cache_manager=Mock(),
        device_id=0,
        instance_name="ctx",
        tx_timeout_s=0.25,
        tx_overall_timeout_s=60.0,
    )
    worker._sender = Mock()
    worker._aux_buffer = Mock()
    request = SimpleNamespace(
        py_disaggregated_params=Mock(),
        py_request_id=41,
        prompt_len=128,
        py_beam_width=1,
    )

    worker.create_tx_session(request)

    session_constructor.assert_called_once_with(
        request_id=41,
        params=request.py_disaggregated_params,
        sender=worker._sender,
        aux_buffer=worker._aux_buffer,
        timeout_s=0.25,
        prompt_len=128,
        beam_width=1,
        overall_timeout_s=60.0,
    )


def test_context_transfer_status_block_all_drains_wait_slices_before_close() -> None:
    task = _FakeTask(TaskStatus.INIT, wait_result=[False, True])
    session = _make_tx_session([task])
    transceiver = _make_transceiver({15: session}, {15: _FakeRequest()})

    completed, failed = transceiver.check_context_transfer_status(None)

    assert completed == [15]
    assert failed == []
    assert task.wait_calls == [0.25, 0.25]
    assert session._closed
    assert 15 not in transceiver._send_sessions


def test_tx_session_blocking_wait_treats_cancelled_session_as_terminal() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([task])
    session._terminal_status = SessionStatus.CANCELLED

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == []


def test_tx_session_blocking_wait_observes_cancellation_between_slices() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([task])
    wait = task.wait

    def cancel_during_wait(timeout: Optional[float] = None) -> bool:
        result = wait(timeout)
        session._terminal_status = SessionStatus.CANCELLED
        return result

    task.wait = cancel_during_wait

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [0.25]


@pytest.mark.parametrize("timeout_s", [None, 0.0, -1.0])
def test_tx_session_blocking_wait_uses_fallback_without_positive_timeout(
    timeout_s: Optional[float],
) -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([task], timeout_s=timeout_s)
    wait = task.wait

    def cancel_during_wait(timeout: Optional[float] = None) -> bool:
        result = wait(timeout)
        session._terminal_status = SessionStatus.CANCELLED
        return result

    task.wait = cancel_during_wait

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [1.0]


def test_tx_session_blocking_wait_treats_task_failure_as_terminal() -> None:
    failed_task = _FakeTask(TaskStatus.ERROR)
    pending_task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=[False, True])
    session = _make_tx_session([failed_task, pending_task])

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert failed_task.wait_calls == []
    # A failed task event does not prove sibling physical writers quiesced, so
    # precheck callers retain the wave instead of treating failure as drained.
    assert pending_task.wait_calls == []


def test_tx_session_blocking_wait_detects_failed_sibling_behind_pending_task() -> None:
    pending_task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    failed_task = _FakeTask(TaskStatus.TRANSFERRING)
    session = _make_tx_session([pending_task, failed_task])

    def fail_sibling(_timeout: Optional[float]) -> None:
        failed_task.status = TaskStatus.ERROR

    pending_task._on_wait = fail_sibling

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert pending_task.wait_calls == [0.25]
    assert failed_task.wait_calls == []


def test_tx_session_blocking_wait_retries_aux_wait_slices() -> None:
    kv_task = _FakeTask(TaskStatus.TRANSFERRED)
    aux_task = _FakeTask(TaskStatus.INIT, wait_result=[False, True])
    session = _make_tx_session([kv_task], need_aux=True, aux_task=aux_task)

    assert session.wait_complete(blocking=True) == WaitResult.COMPLETED
    assert kv_task.wait_calls == []
    assert aux_task.wait_calls == [0.25, 0.25]


def test_tx_session_blocking_aux_wait_observes_cancellation_between_slices() -> None:
    kv_task = _FakeTask(TaskStatus.TRANSFERRED)
    aux_task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([kv_task], need_aux=True, aux_task=aux_task)
    wait = aux_task.wait

    def cancel_during_wait(timeout: Optional[float] = None) -> bool:
        result = wait(timeout)
        session._terminal_status = SessionStatus.CANCELLED
        return result

    aux_task.wait = cancel_during_wait

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert kv_task.wait_calls == []
    assert aux_task.wait_calls == [0.25]


def test_tx_session_blocking_wait_fails_missing_required_aux() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRED)
    session = _make_tx_session([task], need_aux=True)

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert session.status == SessionStatus.ERROR
    assert isinstance(session.exception, RuntimeError)
    assert task.wait_calls == []


def test_generation_first_tx_session_nonblocking_missing_aux_stays_pending() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRED)
    session = _make_tx_session([task], need_aux=True)

    assert session.wait_complete(blocking=False) is None
    assert session.status == SessionStatus.KV_TRANSFERRED
    assert session.exception is None
    assert task.wait_calls == []


def test_tx_session_wait_complete_nonblocking_returns_none_without_waiting() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING)
    session = _make_tx_session([task])

    assert session.wait_complete(blocking=False) is None
    assert task.wait_calls == []


def test_tx_session_wait_complete_nonblocking_reports_later_task_error() -> None:
    pending_task = _FakeTask(TaskStatus.TRANSFERRING)
    failed_task = _FakeTask(TaskStatus.ERROR)
    session = _make_tx_session([pending_task, failed_task])

    assert session.wait_complete(blocking=False) == WaitResult.FAILED
    assert pending_task.wait_calls == []
    assert failed_task.wait_calls == []


def test_tx_session_has_failed_reports_task_error() -> None:
    task = _FakeTask(TaskStatus.ERROR)
    session = _make_tx_session([task])

    assert session.has_failed()


def test_check_context_runs_consensus_after_a_send() -> None:
    # Once the worker has sent, the ctx consensus runs as usual.
    transceiver = _make_transceiver({})
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], []))

    transceiver.check_context_transfer_status(0)
    transceiver._ctx_consensus.assert_called_once()


def test_prepare_context_requests_skips_consensus_when_nothing_waiting() -> None:
    # With nothing waiting on any rank, prepare_context_requests returns before the consensus; the
    # waiting set is the same on every rank.
    transceiver = _make_transceiver({})
    transceiver._wait_reqs = {}
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    transceiver.prepare_context_requests([])
    transceiver._ctx_consensus.assert_not_called()


# --------------------------------------------------------------------------- #
# Sender-side bounded agent wait (nvbugs/6312828).
#
# The sender used to call status.wait() with no timeout, pinning its only worker
# thread (and its bounce send slot) on one transfer forever. It now polls in
# bounded slices and, on giving up, releases the backend transfer request
# explicitly -- the same primitive the C++ in-flight cancel uses
# (agent_utils/connection.cpp). A successful release does NOT prove the NIC
# quiesced (executor/transferAgent.h), which is why the send region is still
# quarantined rather than released, and why no AgentResult.FAILED is sent (that
# would tell the receiver this writer had drained).
#
# What the give-up does NOT do is keep the request's KV pages allocated: the
# executor's CANCELLED reap frees them regardless of the task state. On the
# bounced path the abandoned write reads the quarantined slot instead of those
# pages; with kv_cache_bounce_size_mb=0 (the default) it reads them directly and
# nothing here holds them. Both facts are pinned below.
# --------------------------------------------------------------------------- #

# _deliver_kv_to_agent imports the bounce package, which pulls CUDA bindings at
# import time; skip gracefully on a CPU-only env without them.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as _bounce_impl  # noqa: F401

    _HAVE_BOUNCE = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_BOUNCE = False

_needs_bounce = pytest.mark.skipif(not _HAVE_BOUNCE, reason="bounce import needs the CUDA bindings")


# _wait_for_agent recomputes its budget from time.monotonic() every pass, so a stub that returns
# IN_PROGRESS without advancing the injected clock spins forever. Cap the polls so that mistake
# fails instantly instead of burning the pytest timeout and being reported as an ordinary failure.
_MAX_FAKE_AGENT_POLLS = 1000


class _FakeAgentStatus:
    """Agent status stub exposing the tri-state wait and release(), recording each poll budget."""

    def __init__(
        self,
        states: list[AgentWaitState],
        clock: Optional[_FakeClock] = None,
        on_poll: Optional[Callable[[], None]] = None,
        release_result: bool = True,
        journal: Optional[list[str]] = None,
    ) -> None:
        self._states = list(states)
        self._clock = clock
        self._on_poll = on_poll
        self._release_result = release_result
        self.journal = journal if journal is not None else []
        self.poll_budgets_ms: list[Optional[int]] = []
        self.release_calls = 0

    def wait(self, timeout_ms: Optional[int] = None) -> AgentWaitState:
        assert len(self.poll_budgets_ms) < _MAX_FAKE_AGENT_POLLS, (
            f"the caller polled this stub {_MAX_FAKE_AGENT_POLLS} times without terminating; it "
            f"is almost certainly returning IN_PROGRESS against a frozen clock "
            f"(clock={self._clock!r}). Pass clock=<_FakeClock> so the deadline can arrive."
        )
        self.poll_budgets_ms.append(timeout_ms)
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        if self._clock is not None and state == AgentWaitState.IN_PROGRESS:
            self._clock.advance((timeout_ms or 0) / 1000.0)
        if self._on_poll is not None:
            self._on_poll()
        return state

    def release(self) -> bool:
        self.release_calls += 1
        self.journal.append("release")
        return self._release_result

    def last_status_str(self) -> str:
        return "NIXL_ERR_BACKEND"


class _LegacyAgentStatus:
    """Status from an extension built before release() was bound.

    Delegates every other attribute to a real _FakeAgentStatus so the two stubs can only differ in
    whether release() exists; a hand-written copy loses the clock advance and hangs _wait_for_agent.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._inner = _FakeAgentStatus(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in ("release", "_inner"):
            raise AttributeError(name)  # hidden: getattr(status, "release", None) must be None
        return getattr(self._inner, name)


class _FakeAgent:
    name = "ctx3"

    def __init__(self, status) -> None:
        self.status = status
        self.submitted: list = []

    def submit_transfer_requests(self, request):
        self.submitted.append(request)
        return self.status


class _FakeBounce:
    """Bounce stub recording which send-region verb the sender chose."""

    def __init__(self, slot_id: Optional[int] = 7, journal: Optional[list[str]] = None) -> None:
        self._slot_id = slot_id
        self.released: list[int] = []
        self.quarantined: list[int] = []
        self.journal = journal if journal is not None else []

    def build_request(self, write_meta):
        if self._slot_id is None:
            return None
        return SimpleNamespace(op="WRITE", remote_name="gen1"), self._slot_id

    def release_send(self, slot_id) -> None:
        self.released.append(slot_id)
        self.journal.append("release_send")

    def quarantine_send(self, slot_id, grace_s=None) -> None:
        self.quarantined.append(slot_id)
        self.journal.append("quarantine_send")


class _RecordingDealer:
    def __init__(self) -> None:
        self.sent: list[list] = []

    def send(self, message) -> None:
        self.sent.append(message)


def _make_sender_session(rid: int = 77, deadline_monotonic_s: Optional[float] = None) -> TxSession:
    session = object.__new__(TxSession)
    params = SimpleNamespace(
        disagg_request_id=rid, ctx_request_id=None, schedule_style="CONTEXT_FIRST"
    )
    session._base_args = SimpleNamespace(params=params, prompt_len=8, beam_width=1)
    session._timeout_s = 0.25
    session._overall_timeout_s = None
    session._deadline_monotonic_s = deadline_monotonic_s
    session._need_aux = False
    session._terminal_status = None
    session._exception = None
    session._closed = False
    session.receiver_ready = True
    session.aux_task = None
    session.aux_slot = None
    session._aux_buffer = None
    session._sender = None
    session.lock = threading.Lock()
    session.transfer_start_time = None
    session.transfer_end_time = None
    task = KVSendTask(SimpleNamespace(), params, 0)
    task._perf_timer = None
    task._unique_rid = rid
    session.kv_tasks = [task]
    return session


def _make_sender(agent, bounce, session: TxSession, dealer: _RecordingDealer) -> Sender:
    sender = object.__new__(Sender)
    sender._device_id = 0
    sender._agent = agent
    sender._bounce = bounce
    sender._instance_rank = 3
    sender._sessions = {session.disagg_request_id: lambda: session}
    sender._sessions_lock = threading.Lock()
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=3)
    )
    sender._get_or_connect_thread_dealer = lambda _endpoint: dealer
    return sender


def _make_kv_write_meta(
    session: TxSession, *, bounce_dst_base: Optional[int] = 0x1000
) -> WriteMeta:
    return WriteMeta(
        task=session.kv_tasks[0],
        expected_transfers=1,
        peer_name="gen1",
        peer_rank=1,
        peer_endpoint="tcp://127.0.0.1:5555",
        unique_rid=session.disagg_request_id,
        src_ptrs=np.array([0x100], dtype=np.int64),
        dst_ptrs=np.array([0x200], dtype=np.int64),
        sizes=np.array([64], dtype=np.int64),
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        meta_type=WriteMetaType.KV,
        bounce_dst_base=bounce_dst_base,
    )


def _kv_results(dealer: _RecordingDealer) -> list[tuple]:
    return [
        _KV_RESULT_PREFIX.unpack(message[1])
        for message in dealer.sent
        if message[0] == MessageType.KV_AGENT_RESULT
    ]


def test_agent_wait_deadline_tracks_the_session_deadline(monkeypatch) -> None:
    clock = _FakeClock(now_s=100.0)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    session = _make_sender_session(deadline_monotonic_s=160.0)

    assert _agent_wait_deadline_s(session) == pytest.approx(160.0)


def test_agent_wait_deadline_floors_an_already_expired_deadline(monkeypatch) -> None:
    # A transfer dispatched at or past the deadline still gets one real attempt.
    clock = _FakeClock(now_s=100.0)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    session = _make_sender_session(deadline_monotonic_s=40.0)

    assert _agent_wait_deadline_s(session) == pytest.approx(100.0 + _MIN_AGENT_WAIT_S)


def test_agent_wait_deadline_falls_back_when_session_is_unanchored(monkeypatch) -> None:
    clock = _FakeClock(now_s=100.0)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer._FALLBACK_TX_OVERALL_TIMEOUT_S",
        30.0,
    )
    session = _make_sender_session(deadline_monotonic_s=None)

    assert _agent_wait_deadline_s(session) == pytest.approx(130.0)


def test_wait_for_agent_polls_in_bounded_slices_until_success(monkeypatch) -> None:
    clock = _FakeClock()
    status = _FakeAgentStatus(
        [AgentWaitState.IN_PROGRESS, AgentWaitState.IN_PROGRESS, AgentWaitState.SUCCESS],
        clock=clock,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = object.__new__(Sender)
    session = SimpleNamespace(status=SessionStatus.TRANSFERRING)

    state = sender._wait_for_agent(status, session, deadline_s=60.0)

    assert state == AgentWaitState.SUCCESS
    assert status.poll_budgets_ms == [_AGENT_POLL_SLICE_MS] * 3


def test_wait_for_agent_returns_failure_without_further_polling(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    status = _FakeAgentStatus([AgentWaitState.FAILURE])
    sender = object.__new__(Sender)
    session = SimpleNamespace(status=SessionStatus.TRANSFERRING)

    assert sender._wait_for_agent(status, session, deadline_s=60.0) == AgentWaitState.FAILURE
    assert status.poll_budgets_ms == [_AGENT_POLL_SLICE_MS]


def test_wait_for_agent_clamps_the_last_slice_to_the_remaining_budget(monkeypatch) -> None:
    # A budget that is not a whole number of slices must still end AT the deadline: the last real
    # slice is clamped to what is left (50ms here), never rounded up to a full slice.
    clock = _FakeClock()
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = object.__new__(Sender)
    session = SimpleNamespace(status=SessionStatus.TRANSFERRING)

    state = sender._wait_for_agent(status, session, deadline_s=0.25)

    assert state == AgentWaitState.IN_PROGRESS
    # 100 + 100 + 50 exhausts the 250ms budget; the trailing 0 is the non-blocking last look.
    assert status.poll_budgets_ms == [_AGENT_POLL_SLICE_MS, _AGENT_POLL_SLICE_MS, 50, 0]
    assert clock.now_s == pytest.approx(0.25)  # ended at the deadline, not past it


def test_wait_for_agent_completes_a_transfer_that_lands_on_the_final_look(monkeypatch) -> None:
    # The zero-timeout look after the budget is spent is not decoration: a transfer that landed
    # during the last slice must be completed, not abandoned (abandoning would quarantine its send
    # region for the grace period and strand the task in TRANSFERRING).
    clock = _FakeClock()
    status = _FakeAgentStatus(
        [
            AgentWaitState.IN_PROGRESS,
            AgentWaitState.IN_PROGRESS,
            AgentWaitState.IN_PROGRESS,
            AgentWaitState.SUCCESS,
        ],
        clock=clock,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = object.__new__(Sender)
    session = SimpleNamespace(status=SessionStatus.TRANSFERRING)

    state = sender._wait_for_agent(status, session, deadline_s=0.25)

    assert state == AgentWaitState.SUCCESS
    assert status.poll_budgets_ms[-1] == 0


def test_wait_for_agent_never_passes_a_negative_timeout(monkeypatch) -> None:
    # timeout_ms < 0 means "spin forever" to the C++ agent, so a deadline already in the past must
    # degrade to a single non-blocking check, not to an unbounded wait.
    clock = _FakeClock(now_s=100.0)
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = object.__new__(Sender)
    session = SimpleNamespace(status=SessionStatus.TRANSFERRING)

    state = sender._wait_for_agent(status, session, deadline_s=40.0)

    assert state == AgentWaitState.IN_PROGRESS
    assert status.poll_budgets_ms == [0]
    assert all(budget >= 0 for budget in status.poll_budgets_ms)


def test_expired_session_deadline_still_gets_one_real_slice(monkeypatch) -> None:
    # _MIN_AGENT_WAIT_S floors the deadline, and clamping must compose with that floor: a transfer
    # dispatched at or past the session deadline still gets a real attempt, not a 0ms no-op.
    clock = _FakeClock(now_s=100.0)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    session = _make_sender_session(deadline_monotonic_s=40.0)  # long past
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    sender = object.__new__(Sender)

    state = sender._wait_for_agent(
        status,
        SimpleNamespace(status=SessionStatus.TRANSFERRING),
        _agent_wait_deadline_s(session),
    )

    assert state == AgentWaitState.IN_PROGRESS
    assert status.poll_budgets_ms[0] == _AGENT_POLL_SLICE_MS  # a real attempt, not skipped
    assert clock.now_s == pytest.approx(100.0 + _MIN_AGENT_WAIT_S)


def test_wait_for_agent_gives_up_when_the_session_is_cancelled(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    session = SimpleNamespace(status=SessionStatus.TRANSFERRING)

    def cancel_after_first_poll() -> None:
        session.status = SessionStatus.CANCELLED

    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], on_poll=cancel_after_first_poll)
    sender = object.__new__(Sender)

    assert sender._wait_for_agent(status, session, deadline_s=1e9) == AgentWaitState.IN_PROGRESS
    assert len(status.poll_budgets_ms) == 1


@_needs_bounce
def test_kv_transfer_past_deadline_releases_the_request_then_quarantines_the_slot(
    monkeypatch,
) -> None:
    # FINDING 1's pin: the give-up is an EXPLICIT in-flight cancel, not a destructor side effect.
    # release() must be called once, before the send region is quarantined, so the grace period
    # starts after the backend has been told to tear the request down. A merely-slow transfer is
    # still not turned into a failure: FAILED would tell the receiver this writer had drained.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    error = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        error,
    )
    journal: list[str] = []
    session = _make_sender_session(deadline_monotonic_s=0.5)
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock, journal=journal)
    bounce = _FakeBounce(slot_id=7, journal=journal)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), bounce, session, dealer)
    task = session.kv_tasks[0]

    sender._deliver_kv_to_agent(_make_kv_write_meta(session))

    assert status.release_calls == 1  # deliberate cancel, not release-on-destruction
    assert journal == ["release", "quarantine_send"]  # release first, then start the grace period
    assert task.status == TaskStatus.TRANSFERRING  # cancel_request() still reports "not drained"
    assert not task.is_done
    assert session.has_transferring_tasks()
    assert bounce.quarantined == [7]  # release does not prove the NIC stopped reading it
    assert bounce.released == []
    assert dealer.sent == []  # silence, NOT AgentResult.FAILED ("this writer drained")
    assert error.call_count == 1
    message = error.call_args.args[0]
    assert "still in flight at its deadline" in message
    assert "released the backend request" in message
    assert "release_accepted=True" in message
    assert "src=bounce_slot:7 (quarantined for the grace period)" in message


@_needs_bounce
def test_kv_transfer_past_deadline_reports_a_refused_release_as_a_leaked_handle(
    monkeypatch,
) -> None:
    # A release the backend refuses leaves one backend transfer handle active for the agent's
    # lifetime. That must be logged as such, and must not change the give-up: the region is still
    # quarantined and the receiver is still told nothing.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    error = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        error,
    )
    session = _make_sender_session(deadline_monotonic_s=0.5)
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock, release_result=False)
    bounce = _FakeBounce(slot_id=7)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), bounce, session, dealer)

    sender._deliver_kv_to_agent(_make_kv_write_meta(session))

    messages = [call.args[0] for call in error.call_args_list]
    assert any("refused to release" in message for message in messages)
    assert any("transfer handle stays active" in message for message in messages)
    assert any("release_accepted=False" in message for message in messages)
    assert bounce.quarantined == [7]
    assert dealer.sent == []


@_needs_bounce
def test_kv_transfer_past_deadline_says_so_when_release_is_not_available(monkeypatch) -> None:
    # Against an extension built before release() was bound, the give-up cannot cancel anything.
    # It must say so rather than look like a successful in-flight cancel.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        Mock(),
    )
    warning_once = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.warning_once",
        warning_once,
    )
    session = _make_sender_session(deadline_monotonic_s=0.5)
    bounce = _FakeBounce(slot_id=7)
    dealer = _RecordingDealer()
    status = _LegacyAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    sender = _make_sender(_FakeAgent(status), bounce, session, dealer)

    sender._deliver_kv_to_agent(_make_kv_write_meta(session))

    assert getattr(status, "release", None) is None  # the whole point: no release() to call
    keys = [call.kwargs.get("key") for call in warning_once.call_args_list]
    assert "agent-status-no-release" in keys
    assert bounce.quarantined == [7]  # the rest of the give-up is unchanged
    assert dealer.sent == []


@_needs_bounce
@pytest.mark.parametrize(
    ("state", "expected_task_status"),
    [
        (AgentWaitState.SUCCESS, TaskStatus.TRANSFERRED),
        (AgentWaitState.FAILURE, TaskStatus.ERROR),
    ],
)
def test_terminal_kv_transfer_does_not_release_explicitly(
    monkeypatch,
    state: AgentWaitState,
    expected_task_status: TaskStatus,
) -> None:
    # The explicit release is scoped to the give-up path. A drained transfer is torn down by the
    # status destructor as before, so a terminal wait must not add a second, redundant cancel.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        Mock(),
    )
    session = _make_sender_session(deadline_monotonic_s=60.0)
    status = _FakeAgentStatus([state], clock=clock)
    bounce = _FakeBounce(slot_id=7)
    sender = _make_sender(_FakeAgent(status), bounce, session, _RecordingDealer())

    sender._deliver_kv_to_agent(_make_kv_write_meta(session))

    assert status.release_calls == 0
    assert session.kv_tasks[0].status == expected_task_status
    assert bounce.released == [7]
    assert bounce.quarantined == []


@_needs_bounce
def test_unbounced_kv_give_up_reports_that_nothing_holds_the_kv_pages(monkeypatch) -> None:
    # FINDING 2's pin, part 1: with kv_cache_bounce_size_mb=0 (the default) there is no send
    # region, so the abandoned write reads the request's KV pages directly and this module holds
    # nothing. The log must say that instead of claiming the pages are pinned.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        Sender,
        "_make_agent_request",
        staticmethod(lambda write_meta, device_id: SimpleNamespace(op="WRITE")),
    )
    error = Mock()
    warning_once = Mock()
    monkeypatch.setattr("tensorrt_llm._torch.disaggregation.native.transfer.logger.error", error)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.warning_once", warning_once
    )
    session = _make_sender_session(deadline_monotonic_s=0.5)
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    bounce = _FakeBounce(slot_id=None)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), bounce, session, dealer)

    sender._deliver_kv_to_agent(_make_kv_write_meta(session, bounce_dst_base=None))

    assert status.release_calls == 1  # the request is still released explicitly
    assert bounce.quarantined == []  # there is no send region to hold
    assert bounce.released == []
    assert dealer.sent == []
    message = error.call_args.args[0]
    assert "src=kv_pages" in message
    assert "this module does not hold them" in message
    assert "reaped as CANCELLED" in message
    assert "KV pages are not freed" not in message  # the claim this test exists to keep out
    assert "kv-abandon-unbounced" in [
        call.kwargs.get("key") for call in warning_once.call_args_list
    ]


@_needs_bounce
def test_abandoned_task_does_not_keep_the_kv_pages_allocated(monkeypatch) -> None:
    # FINDING 2's pin, part 2: leaving the task TRANSFERRING only makes cancel_request() report
    # "not drained" for one iteration. check_context_transfer_status then sees a CANCELLED session
    # and closes it regardless of the mid-write task, after which cancel_request() returns True and
    # the executor frees the KV pages. Nothing here protects them on the unbounced path.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        Sender,
        "_make_agent_request",
        staticmethod(lambda write_meta, device_id: SimpleNamespace(op="WRITE")),
    )
    monkeypatch.setattr("tensorrt_llm._torch.disaggregation.native.transfer.logger.error", Mock())
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.warning_once", Mock()
    )
    session = _make_sender_session(rid=77, deadline_monotonic_s=0.5)
    session._sender = Mock()
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    sender = _make_sender(
        _FakeAgent(status), _FakeBounce(slot_id=None), session, _RecordingDealer()
    )

    sender._deliver_kv_to_agent(_make_kv_write_meta(session, bounce_dst_base=None))
    task = session.kv_tasks[0]
    assert task.status == TaskStatus.TRANSFERRING

    req = SimpleNamespace(py_disaggregated_params=None, request_id=77)
    transceiver = _make_transceiver({77: session}, {77: req})
    transceiver._wait_reqs = {}
    transceiver._recv_sessions = {}

    # First cancel: the session is still registered and mid-write, so the executor is told to retry.
    assert transceiver.cancel_request(req) is False
    assert session.status == SessionStatus.CANCELLED

    # The CANCELLED reap closes and drops the session even though the task is still TRANSFERRING.
    transceiver.check_context_transfer_status(None)
    assert 77 not in transceiver._send_sessions
    assert task.status == TaskStatus.TRANSFERRING

    # With no session left, the next cancel succeeds and the request (and its KV pages) is freed.
    assert transceiver.cancel_request(req) is True


@_needs_bounce
def test_kv_transfer_agent_error_reports_failed_and_releases_the_send_slot(monkeypatch) -> None:
    # A genuine backend error HAS drained, so the drain-before-release contract holds: fail the
    # task, release the send region, and tell the receiver.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    error = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        error,
    )
    session = _make_sender_session(deadline_monotonic_s=60.0)
    status = _FakeAgentStatus([AgentWaitState.FAILURE], clock=clock)
    bounce = _FakeBounce(slot_id=7)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), bounce, session, dealer)
    task = session.kv_tasks[0]

    sender._deliver_kv_to_agent(_make_kv_write_meta(session))

    assert task.status == TaskStatus.ERROR
    assert bounce.released == [7]
    assert bounce.quarantined == []
    results = _kv_results(dealer)
    assert len(results) == 1
    peer_rank, unique_rid, slice_id, is_last, status_code, _size = results[0]
    assert (peer_rank, unique_rid, slice_id, is_last) == (3, 77, 0, True)
    assert status_code == 1  # AgentResult.FAILED
    assert len(dealer.sent[0]) == 2  # no bounce tail on a failure
    assert "agent reported error" in error.call_args.args[0]
    assert "nixl_status=NIXL_ERR_BACKEND" in error.call_args.args[0]


@_needs_bounce
def test_kv_transfer_success_releases_the_send_slot_and_reports_success(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    session = _make_sender_session(deadline_monotonic_s=60.0)
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS, AgentWaitState.SUCCESS], clock=clock)
    bounce = _FakeBounce(slot_id=7)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), bounce, session, dealer)
    task = session.kv_tasks[0]

    sender._deliver_kv_to_agent(_make_kv_write_meta(session))

    assert task.status == TaskStatus.TRANSFERRED
    assert task.is_done
    assert bounce.released == [7]
    assert bounce.quarantined == []
    results = _kv_results(dealer)
    assert len(results) == 1
    assert results[0][4] == 0  # AgentResult.SUCCESS
    assert len(dealer.sent[0]) == 5  # KV result plus the 3-frame bounce tail


@_needs_bounce
def test_kv_transfer_worker_is_released_within_the_deadline(monkeypatch) -> None:
    # The pin is now bounded by kv_transfer_timeout_ms instead of being indefinite: the worker
    # returns and can serve the next queued transfer. (This bounds head-of-line blocking; it does
    # not remove it -- one stuck transfer still holds the thread for the whole budget.)
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        Mock(),
    )
    stuck_session = _make_sender_session(rid=77, deadline_monotonic_s=1.0)
    stuck_status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(stuck_status), _FakeBounce(slot_id=7), stuck_session, dealer)

    sender._deliver_kv_to_agent(_make_kv_write_meta(stuck_session))

    assert clock.now_s == pytest.approx(1.0)  # gave up exactly at the session deadline
    # Ten full slices spend the 1.0s budget; the trailing 0 is the non-blocking last look.
    assert stuck_status.poll_budgets_ms[:-1] == [_AGENT_POLL_SLICE_MS] * 10
    assert stuck_status.poll_budgets_ms[-1] == 0
    assert sum(stuck_status.poll_budgets_ms) <= 1000  # the wait never runs past the budget

    # The same worker now serves an unrelated request, which completes normally.
    next_session = _make_sender_session(rid=88, deadline_monotonic_s=clock.now_s + 60.0)
    sender._sessions[88] = lambda: next_session
    sender._agent = _FakeAgent(_FakeAgentStatus([AgentWaitState.SUCCESS], clock=clock))

    sender._deliver_kv_to_agent(_make_kv_write_meta(next_session))

    assert next_session.kv_tasks[0].status == TaskStatus.TRANSFERRED


@_needs_bounce
def test_aux_transfer_past_deadline_releases_the_request_and_is_not_reported_failed(
    monkeypatch,
) -> None:
    # Aux mirrors KV: release the request explicitly, then stay silent. set_exception() would
    # resolve every task and let the session close, freeing the aux slot -- and the release does
    # not prove the NIC stopped reading it.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    error = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        error,
    )
    monkeypatch.setattr(
        Sender,
        "_make_agent_request",
        staticmethod(lambda write_meta, device_id: SimpleNamespace(op="WRITE")),
    )
    session = _make_sender_session(deadline_monotonic_s=0.5)
    params = session._base_args.params
    aux_task = AuxSendTask(params, None)
    aux_task._perf_timer = None
    session.aux_task = aux_task
    status = _FakeAgentStatus([AgentWaitState.IN_PROGRESS], clock=clock)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), _FakeBounce(slot_id=None), session, dealer)

    write_meta = _make_kv_write_meta(session, bounce_dst_base=None)
    write_meta.meta_type = WriteMetaType.AUX
    write_meta.task = aux_task
    sender._deliver_aux_to_agent(write_meta)

    assert status.release_calls == 1  # the give-up is an explicit cancel here too
    assert dealer.sent == []  # no AUX_AGENT_RESULT: that would mean "this writer drained"
    assert session.exception is None
    assert session.status != SessionStatus.ERROR
    assert aux_task._transfer_count == 0
    assert "released the backend request" in error.call_args.args[0]
    assert "release_accepted=True" in error.call_args.args[0]


@_needs_bounce
def test_aux_transfer_agent_error_reports_failed(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.logger.error",
        Mock(),
    )
    monkeypatch.setattr(
        Sender,
        "_make_agent_request",
        staticmethod(lambda write_meta, device_id: SimpleNamespace(op="WRITE")),
    )
    session = _make_sender_session(deadline_monotonic_s=60.0)
    params = session._base_args.params
    aux_task = AuxSendTask(params, None)
    aux_task._perf_timer = None
    session.aux_task = aux_task
    status = _FakeAgentStatus([AgentWaitState.FAILURE], clock=clock)
    dealer = _RecordingDealer()
    sender = _make_sender(_FakeAgent(status), _FakeBounce(slot_id=None), session, dealer)

    write_meta = _make_kv_write_meta(session, bounce_dst_base=None)
    write_meta.meta_type = WriteMetaType.AUX
    write_meta.task = aux_task
    sender._deliver_aux_to_agent(write_meta)

    assert len(dealer.sent) == 1
    assert dealer.sent[0][0] == MessageType.AUX_AGENT_RESULT
    assert dealer.sent[0][3] == AgentResult.FAILED.value.encode("ascii")
    assert session.status == SessionStatus.ERROR

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
"""Retired-session tombstone tests for the disaggregated native Sender.

A REQUEST_DATA that lands after its TxSession was retired used to be dropped in
silence, stranding the generation-side KVRecvTask in TRANSFERRING forever and
pinning its transfer-block slot (nvbugs/6480621). These tests pin the reply
matrix, the success/failure distinction, and the tombstone bound.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _AGENT_RESULT_BY_CODE,
    _KV_RESULT_PREFIX,
    AgentResult,
    MessageType,
    RecvReqInfo,
    RetiredSession,
    Sender,
    TaskStatus,
    TxSession,
)

_PEER_INSTANCE = "gen"
_SELF_RANK = 0


class _FakeDealer:
    """Stands in for the DEALER ZMQMessenger, recording what the Sender sends."""

    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []

    def send(self, messages: list[bytes]) -> None:
        self.sent.append(messages)


class _FakeRegistrar:
    """Minimal PeerRegistrar: every peer rank resolves to one endpoint and one overlap."""

    def __init__(self, endpoint: str, overlap_ranks: list[int]) -> None:
        self._endpoint = endpoint
        self._overlap_ranks = overlap_ranks

    def get_peer_rank_info(self, instance_name: str, instance_rank: int) -> SimpleNamespace:
        return SimpleNamespace(
            instance_name=instance_name,
            instance_rank=instance_rank,
            dp_rank=0,
            device_id=0,
            self_endpoint=self._endpoint,
        )

    def get_peer_overlap(self, _peer_ri, _dp_rank) -> SimpleNamespace:
        return SimpleNamespace(ranks=list(self._overlap_ranks))


class _FakeClock:
    def __init__(self, now_s: float = 0.0) -> None:
        self.now_s = now_s

    def monotonic(self) -> float:
        return self.now_s

    def advance(self, elapsed_s: float) -> None:
        self.now_s += elapsed_s


class _FakeTask:
    def __init__(self, status: TaskStatus) -> None:
        self.status = status


class _FakeTxSession:
    """Weakref-able stand-in for a live TxSession; SimpleNamespace cannot be weakref'd."""

    def __init__(self, rid: int, status: SessionStatus = SessionStatus.READY) -> None:
        self.disagg_request_id = rid
        self.lock = threading.Lock()
        self.receiver_ready = False
        self.kv_tasks: list = []
        self.status = status


def _make_sender(*, overlap_ranks: Optional[list[int]] = None) -> Sender:
    """Build a Sender without its __init__ (no ZMQ sockets, no worker threads)."""
    sender = object.__new__(Sender)
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._pre_cancelled_rids = set()
    sender._retired_sessions = OrderedDict()
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    sender._peer_requests_lock = threading.Lock()
    sender._instance_rank = _SELF_RANK
    sender._registrar = _FakeRegistrar("tcp://peer:1", overlap_ranks or [3])
    # Pre-seeding _dealers makes _get_or_connect_dealer return the fake instead of
    # opening a real socket; the lock it takes is created in __init__, which this
    # stub bypasses, so seed it too or the send is swallowed as an AttributeError.
    sender._dealers = {"tcp://peer:1": _FakeDealer()}
    sender._dealers_lock = threading.Lock()
    return sender


def _dealer(sender: Sender) -> _FakeDealer:
    return sender._dealers["tcp://peer:1"]


def _make_req_info(rid: int, instance_rank: int = 3, slice_id: int = 0) -> RecvReqInfo:
    return RecvReqInfo(
        sender_req_id=rid,
        instance_name=_PEER_INSTANCE,
        instance_rank=instance_rank,
        block_ids_per_layer_groups=[np.array([1, 2, 3], dtype=np.int64)],
        unique_rid=rid,
        slice_id=slice_id,
    )


def _request_data(sender: Sender, info: RecvReqInfo) -> None:
    """Deliver a REQUEST_DATA exactly as the listener thread would."""
    sender._respond_with_kv(b"peer-id", [MessageType.REQUEST_DATA, info.to_bytes()])


def _decode_kv_result(message: list[bytes]) -> SimpleNamespace:
    assert message[0] == MessageType.KV_AGENT_RESULT
    peer_rank, unique_rid, slice_id, is_last_slice, status_code, transfer_size = (
        _KV_RESULT_PREFIX.unpack(message[1])
    )
    return SimpleNamespace(
        peer_rank=peer_rank,
        unique_rid=unique_rid,
        slice_id=slice_id,
        is_last_slice=is_last_slice,
        status=_AGENT_RESULT_BY_CODE[status_code],
        transfer_size=transfer_size,
    )


def test_request_data_after_failed_retirement_gets_a_failed_reply() -> None:
    # The 6480621 case: the ctx session was cancelled/timed out and retired before this
    # peer's REQUEST_DATA landed. Silence here strands the receiver forever.
    sender = _make_sender()
    sender.clear_session(7, completed=False)

    _request_data(sender, _make_req_info(7, instance_rank=3, slice_id=2))

    assert len(_dealer(sender).sent) == 1
    result = _decode_kv_result(_dealer(sender).sent[0])
    assert result.status == AgentResult.FAILED
    assert result.unique_rid == 7
    # is_last_slice must be set or the receiver never resolves the task future, and the
    # slice_id must be the receiver's own so its assert on _kv_tasks holds.
    assert result.is_last_slice is True
    assert result.slice_id == 2
    assert result.peer_rank == _SELF_RANK
    # Answered rather than stashed: no orphan RecvReqInfo is left for the TTL sweep.
    assert sender._peer_requests == {}


def test_request_data_after_successful_retirement_sends_nothing() -> None:
    # clear_session runs on normal successful completion too. A naive tombstone would
    # answer FAILED here and flip a transfer that actually landed back to ERROR.
    sender = _make_sender()
    sender.clear_session(7, completed=True)

    _request_data(sender, _make_req_info(7))

    assert _dealer(sender).sent == []


def test_request_data_from_an_already_served_rank_is_never_failed() -> None:
    # MEMORY SAFETY: this rank's RecvReqInfo was accepted before retirement, so a
    # WriteMeta went out for it and a one-sided write may still be landing.
    # AgentResult.FAILED means "this writer has drained" and would let the receiver
    # release its bounce region with no quarantine, so it must not be sent here.
    sender = _make_sender()
    sender._peer_requests[7] = {3: _make_req_info(7, instance_rank=3)}
    sender._peer_requests_timestamps[7] = 0.0
    sender.clear_session(7, completed=False)
    assert sender._retired_sessions[7].served_ranks == frozenset({3})

    _request_data(sender, _make_req_info(7, instance_rank=3))

    assert _dealer(sender).sent == []


def test_request_data_from_an_unserved_rank_is_failed_even_when_a_sibling_was_served() -> None:
    # Rank 5 never had a RecvReqInfo accepted, so no WriteMeta was ever built for it and
    # nothing can be in flight to it; FAILED is drained-accurate for rank 5 alone.
    sender = _make_sender(overlap_ranks=[3, 5])
    sender._peer_requests[7] = {3: _make_req_info(7, instance_rank=3)}
    sender._peer_requests_timestamps[7] = 0.0
    sender.clear_session(7, completed=False)

    _request_data(sender, _make_req_info(7, instance_rank=5))

    assert len(_dealer(sender).sent) == 1
    assert _decode_kv_result(_dealer(sender).sent[0]).status == AgentResult.FAILED


def test_request_data_without_a_tombstone_still_stashes_req_info() -> None:
    # REQUEST_DATA that arrives before setup_session must keep its pre-existing
    # behaviour: stash the RecvReqInfo and stay silent, never answer FAILED.
    sender = _make_sender()

    _request_data(sender, _make_req_info(7))

    assert _dealer(sender).sent == []
    assert list(sender._peer_requests[7]) == [3]


def test_setup_session_drops_a_stale_tombstone() -> None:
    sender = _make_sender()
    sender.clear_session(7, completed=False)
    session = _FakeTxSession(7)

    sender.setup_session(session)

    assert 7 not in sender._retired_sessions
    assert 7 in sender._sessions


def test_a_live_session_takes_precedence_over_its_tombstone() -> None:
    # A rid reused by a new session must be served, not answered from the old tombstone.
    sender = _make_sender()
    sender.clear_session(7, completed=False)
    session = _FakeTxSession(7)
    sender.setup_session(session)

    _request_data(sender, _make_req_info(7))

    assert _dealer(sender).sent == []
    assert list(sender._peer_requests[7]) == [3]


def test_tombstones_are_capacity_bounded() -> None:
    # A long-running server retires unboundedly many sessions; the newest N survive.
    sender = _make_sender()
    sender._MAX_RETIRED_SESSIONS = 4

    for rid in range(20):
        sender.clear_session(rid, completed=False)

    assert len(sender._retired_sessions) == 4
    assert list(sender._retired_sessions) == [16, 17, 18, 19]


def test_re_retiring_a_rid_keeps_the_map_ordered_by_retirement_time(monkeypatch) -> None:
    # _sweep_retired_sessions stops at the first fresh head, so insertion order must stay
    # sorted by retired_at even when a rid is retired twice.
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = _make_sender()
    sender.clear_session(1, completed=False)
    clock.advance(1.0)
    sender.clear_session(2, completed=False)
    clock.advance(1.0)
    sender.clear_session(1, completed=True)

    assert list(sender._retired_sessions) == [2, 1]
    retired_at = [entry.retired_at for entry in sender._retired_sessions.values()]
    assert retired_at == sorted(retired_at)


def test_sweep_stale_req_infos_evicts_expired_tombstones(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = _make_sender()
    sender.clear_session(1, completed=False)
    clock.advance(sender._RETIRED_SESSION_TTL_S + 1.0)
    sender.clear_session(2, completed=False)

    # The existing periodic sweep is the single entry point, so tombstones cannot leak
    # even on a server that never hits the capacity bound.
    sender.sweep_stale_req_infos()

    assert list(sender._retired_sessions) == [2]


def test_expired_tombstone_stops_answering_and_falls_back_to_stashing(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    sender = _make_sender()
    sender.clear_session(7, completed=False)
    clock.advance(sender._RETIRED_SESSION_TTL_S + 1.0)
    sender.sweep_stale_req_infos()

    _request_data(sender, _make_req_info(7))

    assert _dealer(sender).sent == []
    assert list(sender._peer_requests[7]) == [3]


def _make_closing_tx_session(
    rid: int,
    kv_tasks: list[_FakeTask],
    terminal_status: Optional[SessionStatus] = None,
) -> TxSession:
    """A real TxSession with only the fields close()/is_completed() touch."""
    session = object.__new__(TxSession)
    session._base_args = SimpleNamespace(
        params=SimpleNamespace(disagg_request_id=rid, ctx_request_id=None)
    )
    session.request_id = rid
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._need_aux = False
    session._terminal_status = terminal_status
    session.receiver_ready = True
    session.kv_tasks = kv_tasks
    session.aux_task = None
    return session


@pytest.mark.parametrize(
    ("kv_tasks", "terminal_status", "expected_completed"),
    [
        ([_FakeTask(TaskStatus.TRANSFERRED)], None, True),
        ([_FakeTask(TaskStatus.TRANSFERRING)], None, False),
        ([_FakeTask(TaskStatus.TRANSFERRED)], SessionStatus.CANCELLED, False),
        ([_FakeTask(TaskStatus.ERROR)], SessionStatus.ERROR, False),
        ([], None, False),
    ],
)
def test_tx_session_close_reports_its_own_success_verdict(
    kv_tasks: list[_FakeTask],
    terminal_status: Optional[SessionStatus],
    expected_completed: bool,
) -> None:
    # is_completed() is the positive success test, so an abandoned or cancelled session
    # tombstones as a failure and only a genuinely finished one as a success.
    sender = _make_sender()
    session = _make_closing_tx_session(9, kv_tasks, terminal_status=terminal_status)
    session._sender = sender

    session.close()

    assert sender._retired_sessions[9].completed is expected_completed


def test_close_after_success_then_late_request_data_is_silent() -> None:
    # End to end through the real TxSession.close(): a completed session leaves a
    # success tombstone, and a late REQUEST_DATA gets no spurious failure.
    sender = _make_sender()
    session = _make_closing_tx_session(9, [_FakeTask(TaskStatus.TRANSFERRED)])
    session._sender = sender
    session.close()

    _request_data(sender, _make_req_info(9))

    assert isinstance(sender._retired_sessions[9], RetiredSession)
    assert _dealer(sender).sent == []


def test_close_after_cancel_then_late_request_data_fails_the_receiver() -> None:
    sender = _make_sender()
    session = _make_closing_tx_session(
        9, [_FakeTask(TaskStatus.INIT)], terminal_status=SessionStatus.CANCELLED
    )
    session._sender = sender
    session.close()

    _request_data(sender, _make_req_info(9))

    assert len(_dealer(sender).sent) == 1
    assert _decode_kv_result(_dealer(sender).sent[0]).status == AgentResult.FAILED

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for AgentHandle / _RWLock; pure-Python, no GPU, no NIXL."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.disaggregation.base.agent import AgentClosedError, AgentHandle, _RWLock


def test_use_yields_underlying_agent():
    agent = MagicMock()
    handle = AgentHandle(agent)
    with handle.use() as a:
        assert a is agent
    assert not handle.is_closed


def test_close_with_no_active_use_succeeds_and_calls_shutdown():
    agent = MagicMock()
    handle = AgentHandle(agent)
    handle.close(timeout=1.0)
    agent.shutdown.assert_called_once()
    assert handle.is_closed


def test_use_after_close_raises_agent_closed_error():
    handle = AgentHandle(MagicMock())
    handle.close(timeout=1.0)
    with pytest.raises(AgentClosedError):
        with handle.use():
            pass


def test_double_close_is_idempotent():
    agent = MagicMock()
    handle = AgentHandle(agent)
    handle.close(timeout=1.0)
    handle.close(timeout=1.0)
    agent.shutdown.assert_called_once()


def test_close_blocks_until_active_reader_releases():
    agent = MagicMock()
    handle = AgentHandle(agent)

    reader_entered = threading.Event()
    reader_release = threading.Event()
    reader_done = threading.Event()

    def reader():
        with handle.use():
            reader_entered.set()
            reader_release.wait(timeout=5.0)
        reader_done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert reader_entered.wait(timeout=2.0)

    closer_done = threading.Event()

    def closer():
        handle.close(timeout=5.0)
        closer_done.set()

    c = threading.Thread(target=closer, daemon=True)
    c.start()

    assert not closer_done.wait(timeout=0.2)
    agent.shutdown.assert_not_called()

    reader_release.set()
    assert reader_done.wait(timeout=2.0)
    assert closer_done.wait(timeout=2.0)
    agent.shutdown.assert_called_once()


def test_close_timeout_raises_when_reader_does_not_release():
    agent = MagicMock()
    handle = AgentHandle(agent)

    reader_entered = threading.Event()
    reader_release = threading.Event()

    def reader():
        with handle.use():
            reader_entered.set()
            reader_release.wait(timeout=5.0)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert reader_entered.wait(timeout=2.0)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        handle.close(timeout=0.2)
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed <= 1.5
    agent.shutdown.assert_not_called()

    reader_release.set()
    t.join(timeout=2.0)


def test_waiting_writer_blocks_new_readers():
    """Writer priority: once close() is waiting, new use() calls must not enter."""
    agent = MagicMock()
    handle = AgentHandle(agent)

    r1_entered = threading.Event()
    r1_release = threading.Event()

    def reader_one():
        with handle.use():
            r1_entered.set()
            r1_release.wait(timeout=5.0)

    t1 = threading.Thread(target=reader_one, daemon=True)
    t1.start()
    assert r1_entered.wait(timeout=2.0)

    closer_done = threading.Event()

    def closer():
        handle.close(timeout=5.0)
        closer_done.set()

    c = threading.Thread(target=closer, daemon=True)
    c.start()

    time.sleep(0.05)

    r2_entered = threading.Event()
    r2_outcome = {}

    def reader_two():
        try:
            with handle.use():
                r2_entered.set()
        except AgentClosedError:
            r2_outcome["closed"] = True

    t2 = threading.Thread(target=reader_two, daemon=True)
    t2.start()

    assert not r2_entered.wait(timeout=0.2)

    r1_release.set()
    assert closer_done.wait(timeout=2.0)
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert r2_outcome.get("closed") is True


def test_register_memory_delegates_and_records():
    agent = MagicMock()
    handle = AgentHandle(agent)
    handle.register_memory("desc1")
    handle.register_memory("desc2")
    agent.register_memory.assert_any_call("desc1")
    agent.register_memory.assert_any_call("desc2")
    assert handle._registered_mem == ["desc1", "desc2"]


def test_register_memory_failure_does_not_record():
    agent = MagicMock()
    agent.register_memory.side_effect = RuntimeError("boom")
    handle = AgentHandle(agent)
    with pytest.raises(RuntimeError):
        handle.register_memory("desc1")
    assert handle._registered_mem == []
    handle.close(timeout=1.0)
    agent.deregister_memory.assert_not_called()


def test_close_deregisters_before_shutdown_in_order():
    agent = MagicMock()
    call_order = []
    agent.deregister_memory.side_effect = lambda d: call_order.append(("dereg", d))
    agent.shutdown.side_effect = lambda: call_order.append(("shutdown",))

    handle = AgentHandle(agent)
    handle.register_memory("a")
    handle.register_memory("b")
    handle.register_memory("c")
    handle.close(timeout=1.0)

    assert call_order == [("dereg", "a"), ("dereg", "b"), ("dereg", "c"), ("shutdown",)]


def test_close_continues_when_a_deregister_raises():
    agent = MagicMock()
    call_order = []

    def dereg(d):
        call_order.append(("dereg", d))
        if d == "b":
            raise RuntimeError("dereg b failed")

    agent.deregister_memory.side_effect = dereg
    agent.shutdown.side_effect = lambda: call_order.append(("shutdown",))

    handle = AgentHandle(agent)
    handle.register_memory("a")
    handle.register_memory("b")
    handle.register_memory("c")
    handle.close(timeout=1.0)

    # All dereg attempted in order; shutdown still called despite the middle failure.
    assert call_order == [("dereg", "a"), ("dereg", "b"), ("dereg", "c"), ("shutdown",)]
    assert handle.is_closed


def test_register_memory_after_close_raises():
    handle = AgentHandle(MagicMock())
    handle.close(timeout=1.0)
    with pytest.raises(AgentClosedError):
        handle.register_memory("desc")


def test_bindings_shutdown_propagates_errors():
    try:
        from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent
    except ImportError:
        pytest.skip("C++ transfer agent binding not available")

    cpp_agent = MagicMock()
    cpp_agent.shutdown.side_effect = RuntimeError("boom")
    # __new__ bypasses __init__ so the test doesn't need a real C++ CppNixlTransferAgent.
    agent = BindingsNixlTransferAgent.__new__(BindingsNixlTransferAgent)
    agent._cpp_agent = cpp_agent
    agent.name = "test"

    with pytest.raises(RuntimeError, match="boom"):
        agent.shutdown()
    # Wrapper field nulled before the raised call, so the next shutdown is a no-op.
    assert agent._cpp_agent is None
    cpp_agent.shutdown.assert_called_once()


def test_bindings_shutdown_idempotent():
    try:
        from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent
    except ImportError:
        pytest.skip("C++ transfer agent binding not available")

    cpp_agent = MagicMock()
    # __new__ bypasses __init__ so the test doesn't need a real C++ CppNixlTransferAgent.
    agent = BindingsNixlTransferAgent.__new__(BindingsNixlTransferAgent)
    agent._cpp_agent = cpp_agent
    agent.name = "test"

    agent.shutdown()
    agent.shutdown()
    cpp_agent.shutdown.assert_called_once()


def test_nested_use_on_same_thread_raises():
    handle = AgentHandle(MagicMock())
    with handle.use():
        with pytest.raises(RuntimeError, match="nested"):
            with handle.use():
                pass
    # Outer slot released cleanly; close still works.
    handle.close(timeout=1.0)


def test_nested_use_via_register_memory_raises():
    handle = AgentHandle(MagicMock())
    with handle.use():
        with pytest.raises(RuntimeError, match="nested"):
            handle.register_memory("d")


def test_nested_use_raises_even_with_writer_waiting():
    """The footgun this guard actually exists to prevent."""
    handle = AgentHandle(MagicMock())
    closer_started = threading.Event()

    def closer():
        closer_started.set()
        handle.close(timeout=5.0)

    with handle.use():
        c = threading.Thread(target=closer, daemon=True)
        c.start()
        assert closer_started.wait(timeout=1.0)
        # Poll for the closer to enter the writer_waiting state, instead of a fixed sleep.
        deadline = time.monotonic() + 1.0
        while not handle._lock._writer_waiting:
            if time.monotonic() > deadline:
                raise AssertionError("closer never entered writer_waiting state")
            time.sleep(0.001)
        with pytest.raises(RuntimeError, match="nested"):
            with handle.use():
                pass
    c.join(timeout=2.0)
    assert not c.is_alive()


def test_close_consistent_state_when_agent_shutdown_raises():
    agent = MagicMock()
    agent.shutdown.side_effect = RuntimeError("shutdown failed")
    handle = AgentHandle(agent)
    handle.register_memory("d")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        handle.close(timeout=1.0)

    # State must remain closed even though agent.shutdown raised.
    assert handle.is_closed
    # Dereg ran before shutdown; counted once.
    agent.deregister_memory.assert_called_once_with("d")
    agent.shutdown.assert_called_once()
    # Second close is a no-op; no extra calls.
    handle.close(timeout=1.0)
    agent.shutdown.assert_called_once()


def test_rwlock_allows_concurrent_readers():
    lock = _RWLock()
    barrier = threading.Barrier(parties=4)
    seen_in_parallel = []
    in_parallel_lock = threading.Lock()

    def reader():
        with lock.read():
            barrier.wait(timeout=2.0)
            with in_parallel_lock:
                seen_in_parallel.append(threading.current_thread().name)

    threads = [threading.Thread(target=reader, name=f"r{i}", daemon=True) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)
    assert len(seen_in_parallel) == 4

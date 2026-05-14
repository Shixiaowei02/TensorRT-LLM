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
"""Reader-writer lock used by AgentHandle; intentionally lightweight, no third-party dep."""

import contextlib
import threading
from typing import Optional


class _RWLock:
    """Reader-writer lock with writer priority; readers parallel, writer exclusive."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer_waiting = False
        self._writer_active = False
        self._reader_owners: set[int] = set()

    @contextlib.contextmanager
    def read(self):
        tid = threading.get_ident()
        with self._cond:
            # Check nested-read FIRST: if this thread already holds a slot, a concurrent
            # waiting writer would otherwise deadlock us inside the wait loop below.
            if tid in self._reader_owners:
                raise RuntimeError(
                    "nested AgentHandle.use()/register_memory() on the same thread "
                    "is not supported (would deadlock with close)"
                )
            while self._writer_waiting or self._writer_active:
                self._cond.wait()
            self._reader_owners.add(tid)
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._reader_owners.discard(tid)
                if self._readers == 0:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def write(self, timeout: Optional[float] = None):
        with self._cond:
            self._writer_waiting = True
            try:
                ok = self._cond.wait_for(
                    lambda: self._readers == 0 and not self._writer_active,
                    timeout,
                )
                if not ok:
                    raise TimeoutError(
                        f"_RWLock.write: timeout after {timeout}s "
                        f"with {self._readers} reader(s) still active"
                    )
                self._writer_active = True
            finally:
                self._writer_waiting = False
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()

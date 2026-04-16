# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for CacheReuseAdapter and _align_kv_blocks prefix-offset logic."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import Sender

# ---------------------------------------------------------------------------
# _align_kv_blocks tests (the core block-alignment helper)
# ---------------------------------------------------------------------------


class TestAlignKvBlocks:
    """Verify Sender._align_kv_blocks handles prefix-cache offsets correctly."""

    TPB = 64  # tokens_per_block

    def _align(self, src, dst, src_start=0, dst_start=0):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=self.TPB,
        )

    def test_no_prefix_cache(self):
        """Neither side has prefix cache — identity."""
        src, dst = self._align([10, 11, 12], [20, 21, 22])
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_gen_has_prefix_cache(self):
        """Gen cached 2 blocks — only transfer suffix."""
        # ctx sends blocks for tokens [0, 320), gen needs only [128, 320)
        src, dst = self._align(
            [10, 11, 12, 13, 14],  # 5 blocks, full prompt
            [20, 21, 22],  # 3 blocks, suffix only
            src_start=0,
            dst_start=2 * self.TPB,  # gen cached 2 blocks
        )
        # src should skip first 2 blocks, transfer 3
        np.testing.assert_array_equal(src, [12, 13, 14])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_ctx_has_prefix_cache(self):
        """Ctx cached 1 block — ctx starts from block 1."""
        src, dst = self._align(
            [10, 11, 12],  # ctx: 3 blocks starting from token 64
            [20, 21, 22, 23],  # gen: 4 blocks, full prompt
            src_start=1 * self.TPB,
            dst_start=0,
        )
        # dst should skip first block, transfer 3
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [21, 22, 23])

    def test_both_have_prefix_cache(self):
        """Both sides cached — transfer only the overlap."""
        src, dst = self._align(
            [10, 11, 12],  # ctx: 3 blocks from token 64
            [20, 21],  # gen: 2 blocks from token 128
            src_start=1 * self.TPB,
            dst_start=2 * self.TPB,
        )
        # overlap starts at token 128 → src_skip=1, dst_skip=0, n=2
        np.testing.assert_array_equal(src, [11, 12])
        np.testing.assert_array_equal(dst, [20, 21])

    def test_gen_full_cache_hit(self):
        """Gen has entire prompt cached — nothing to transfer."""
        src, dst = self._align(
            [10, 11, 12],  # ctx: full prompt
            [20, 21, 22],  # gen: full prompt (all cached)
            src_start=0,
            dst_start=3 * self.TPB,  # gen cached all 3 blocks
        )
        assert src.size == 0
        assert dst.size == 0

    def test_gen_prefix_with_draft_block(self):
        """Gen has prefix cache + 1 extra draft block — transfer suffix only."""
        src, dst = self._align(
            [10, 11, 12, 13],  # ctx: 4 blocks
            [20, 21, 22],  # gen: 2 suffix + 1 draft
            src_start=0,
            dst_start=2 * self.TPB,
        )
        # transfer min(4-2, 3-0) = 2 blocks
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [20, 21])


# ---------------------------------------------------------------------------
# KVSlice token_range with prefix offset
# ---------------------------------------------------------------------------


class TestTokenRangeWithPrefix:
    """Verify TokenRange is correctly set with prefix offsets."""

    def test_no_cache(self):
        from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange

        tr = TokenRange(start=0, end=256)
        assert tr.start == 0
        assert tr.end == 256

    def test_partial_cache(self):
        from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange

        tr = TokenRange(start=128, end=256)
        assert tr.start == 128
        assert tr.end == 256

    def test_full_cache_hit_raises(self):
        """When cached_tokens == prompt_len, start == end should raise."""
        from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange

        with pytest.raises(ValueError):
            TokenRange(start=256, end=256)

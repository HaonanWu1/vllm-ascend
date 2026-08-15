#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310
from vllm_ascend.attention.attention_v1 import AscendAttentionState


class TestAscendAttentionMetadataBuilder310Causal(TestBase):
    def test_full_dflash_capture_and_runtime_share_proposer_buffers(self):
        builder = object.__new__(AscendAttentionMetadataBuilder310)
        builder.device = torch.device("cpu")
        builder._query_lens_cpu_buffer = torch.zeros(8, dtype=torch.int32)
        owner = SimpleNamespace(
            num_speculative_tokens=15,
            slot_mapping_group=[
                torch.full((64,), -1, dtype=torch.int32)
            ],
            seq_lens_group=[torch.zeros(8, dtype=torch.int32)],
            query_start_loc_group=[torch.zeros(9, dtype=torch.int32)],
        )
        builder._dflash_full_graph_owner_310 = owner

        capture_common = MagicMock()
        capture_common.num_reqs = 2
        capture_common.causal = False
        capture_common.max_query_len = 16
        capture_common.num_actual_tokens = 32
        capture_common.query_start_loc = torch.tensor(
            [0, 16, 32], dtype=torch.int32
        )
        capture_common.query_start_loc_cpu = capture_common.query_start_loc
        capture_common.seq_lens = torch.tensor([20, 24], dtype=torch.int32)
        capture_common.slot_mapping = torch.arange(32, dtype=torch.int32)
        capture_common._seq_lens_cpu = torch.tensor(
            [20, 24], dtype=torch.int32
        )
        capture_common.attn_state = AscendAttentionState.ChunkedPrefill

        base_metadata = MagicMock()
        base_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        with (
            patch.object(
                AscendAttentionMetadataBuilder310.__bases__[0],
                "build",
                return_value=base_metadata,
            ) as base_build,
            patch(
                "vllm_ascend._310p.attention.metadata_builder."
                "is_compressed_mask_supported",
                return_value=False,
            ),
        ):
            result = builder.build(0, capture_common)

        common_seen = base_build.call_args.args[1]
        self.assertEqual(
            common_seen.query_start_loc.data_ptr(),
            owner.query_start_loc_group[0].data_ptr(),
        )
        self.assertEqual(
            common_seen.seq_lens.data_ptr(),
            owner.seq_lens_group[0].data_ptr(),
        )
        self.assertEqual(
            common_seen.slot_mapping.data_ptr(),
            owner.slot_mapping_group[0].data_ptr(),
        )
        self.assertEqual(
            result.query_start_loc.data_ptr(),
            owner.query_start_loc_group[0].data_ptr(),
        )
        self.assertEqual(
            result.seq_lens.data_ptr(),
            owner.seq_lens_group[0].data_ptr(),
        )
        torch.testing.assert_close(
            owner.seq_lens_group[0][:2],
            torch.tensor([36, 40], dtype=torch.int32),
        )
        self.assertFalse(
            hasattr(builder, "_dflash_full_graph_owner_310")
        )

    def test_build_non_causal_uses_zero_compressed_mask(self):
        builder = object.__new__(AscendAttentionMetadataBuilder310)
        builder.device = torch.device("cpu")
        builder._query_lens_cpu_buffer = torch.zeros(8, dtype=torch.int32, device="cpu")

        from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310

        builder.attn_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), 4096)

        common = MagicMock()
        common.num_reqs = 2
        common.causal = False
        common.query_start_loc = torch.tensor([0, 1, 3])
        common.query_start_loc_cpu = torch.tensor([0, 1, 3])
        common.seq_lens = torch.tensor([4, 6])
        common.attn_state = AscendAttentionState.ChunkedPrefill

        base_metadata = MagicMock()
        base_metadata.attn_state = AscendAttentionState.ChunkedPrefill

        with patch.object(
            AscendAttentionMetadataBuilder310.__bases__[0], "build", return_value=base_metadata
        ), patch(
            "vllm_ascend._310p.attention.metadata_builder.is_compressed_mask_supported",
            return_value=True,
        ), patch(
            "vllm_ascend._310p.attention.metadata_builder.AttentionMaskBuilder310.get_compressed_non_causal_splitfuse_mask",
            return_value=torch.zeros(2048, 2048),
        ) as mock_non_causal:
            result = builder.build(0, common)
            mock_non_causal.assert_called_once_with(builder.device)
            self.assertIs(result.attn_mask, mock_non_causal.return_value)

    def test_build_attaches_host_seq_lens_cpu_for_prefill(self):
        # PrefillNoCache (non-splitfuse) returns early, but the host seq_lens must
        # still be attached so ATB flash attention gets host data even when the base
        # builder left seq_lens on device (parallel-drafting path).
        builder = object.__new__(AscendAttentionMetadataBuilder310)
        builder.device = torch.device("cpu")
        builder._query_lens_cpu_buffer = torch.zeros(8, dtype=torch.int32, device="cpu")

        from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310

        builder.attn_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), 4096)

        common = MagicMock()
        common.num_reqs = 2
        common._seq_lens_cpu = torch.tensor([4, 6, 99], dtype=torch.int32)
        common.attn_state = AscendAttentionState.PrefillNoCache

        base_metadata = MagicMock()
        base_metadata.attn_state = AscendAttentionState.PrefillNoCache

        with patch.object(AscendAttentionMetadataBuilder310.__bases__[0], "build", return_value=base_metadata):
            result = builder.build(0, common)

        torch.testing.assert_close(result.seq_lens_cpu, torch.tensor([4, 6], dtype=torch.int32))

    def test_build_decode_binds_device_seq_lens_view(self):
        builder = object.__new__(AscendAttentionMetadataBuilder310)
        builder.device = torch.device("cpu")
        builder._query_lens_cpu_buffer = torch.zeros(
            8,
            dtype=torch.int32,
            device="cpu",
        )

        host_seq_lens = torch.tensor([4, 6, 99], dtype=torch.int32)
        device_seq_lens = torch.tensor([4, 6, 99], dtype=torch.int32)
        common = MagicMock()
        common.num_reqs = 2
        common._seq_lens_cpu = host_seq_lens
        common.seq_lens_cpu = None
        common.seq_lens = device_seq_lens
        common.attn_state = AscendAttentionState.DecodeOnly

        base_metadata = MagicMock()
        base_metadata.attn_state = AscendAttentionState.DecodeOnly
        base_metadata.seq_lens = host_seq_lens[:2]

        with patch.object(
            AscendAttentionMetadataBuilder310.__bases__[0],
            "build",
            return_value=base_metadata,
        ):
            result = builder.build(0, common)

        self.assertEqual(
            result.seq_lens.data_ptr(),
            device_seq_lens.data_ptr(),
        )
        host_seq_lens[0] = 100
        torch.testing.assert_close(
            result.seq_lens,
            torch.tensor([4, 6], dtype=torch.int32),
        )
        device_seq_lens[1] = 7
        torch.testing.assert_close(
            result.seq_lens,
            torch.tensor([4, 7], dtype=torch.int32),
        )

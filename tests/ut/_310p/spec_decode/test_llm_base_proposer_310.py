#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, writing
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend._310p.ops import rotary_embedding as rotary_310
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendRotaryEmbedding310,
    _build_draft_cos_sin_slice,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class TestAscendSpecDecodeBaseProposer310(TestBase):
    @staticmethod
    def _reset_draft_rope_buffers():
        rotary_310._draft_cos = None
        rotary_310._draft_sin = None
        rotary_310._draft_rope_dim = None

    @staticmethod
    def _draft_rope_growth_addresses() -> list[int]:
        cos_sin_cache = torch.randn(256, 128, dtype=torch.float32)
        return [
            _build_draft_cos_sin_slice(
                cos_sin_cache,
                torch.arange(num_tokens, dtype=torch.long),
            )[0].data_ptr()
            for num_tokens in (128, 141, 128)
        ]

    @staticmethod
    def _seed_draft_rope_buffer(num_tokens: int = 128) -> None:
        _build_draft_cos_sin_slice(
            torch.randn(256, 128, dtype=torch.float32),
            torch.arange(num_tokens, dtype=torch.long),
        )

    def test_run_merged_draft_sets_rope_flag_before_call(self):
        flag_states = []

        def mock_original(
            self,
            num_input_tokens,
            batch_size,
            token_indices_to_sample,
            target_positions,
            inputs_embeds,
            multi_steps_attn_metadata,
            num_tokens,
            is_prefill=None,
        ):
            flag_states.append(AscendRotaryEmbedding310._is_drafting_update_enabled)
            return torch.zeros(num_tokens, dtype=torch.long)

        with (
            patch.object(AscendSpecDecodeBaseProposer, "_run_merged_draft", mock_original),
            patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft", mock_original),
        ):
            proposer = object.__new__(AscendSpecDecodeBaseProposer310)
            # object.__new__ bypasses the base proposer initializer, which
            # always supplies this discriminator in production.
            proposer.method = "mtp"
            proposer._run_merged_draft(
                num_input_tokens=4,
                batch_size=2,
                token_indices_to_sample=torch.tensor([0, 1]),
                target_positions=torch.tensor([0, 1, 2, 3]),
                inputs_embeds=torch.zeros(4, 128),
                multi_steps_attn_metadata=None,
                num_tokens=4,
            )

        self.assertEqual(len(flag_states), 1)
        self.assertTrue(flag_states[0])
        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_run_merged_draft_restores_rope_flag_after_exception(self):
        def mock_original(*args, **kwargs):
            raise RuntimeError("Test exception")

        with (
            patch.object(AscendSpecDecodeBaseProposer, "_run_merged_draft", mock_original),
            patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft", mock_original),
        ):
            proposer = object.__new__(AscendSpecDecodeBaseProposer310)
            proposer.method = "mtp"
            with self.assertRaises(RuntimeError):
                proposer._run_merged_draft(
                    num_input_tokens=4,
                    batch_size=2,
                    token_indices_to_sample=torch.tensor([0, 1]),
                    target_positions=torch.tensor([0, 1, 2, 3]),
                    inputs_embeds=torch.zeros(4, 128),
                    multi_steps_attn_metadata=None,
                    num_tokens=4,
                )

        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_run_merged_dflash_reserves_context_rope_capacity(self):
        addresses = []

        def mock_original(*args, **kwargs):
            addresses.extend(self._draft_rope_growth_addresses())
            return torch.zeros(128, dtype=torch.long)

        proposer = object.__new__(AscendSpecDecodeBaseProposer310)
        proposer.method = "dflash"
        proposer.max_num_tokens = 256
        self._reset_draft_rope_buffers()
        try:
            self._seed_draft_rope_buffer()
            with (
                patch(
                    "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
                    "_original_run_merged_draft",
                    side_effect=mock_original,
                ),
                patch(
                    "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
                    "dflash_diagnostic_enabled",
                    return_value=False,
                ),
            ):
                proposer._run_merged_draft(
                    num_input_tokens=141,
                    batch_size=8,
                    token_indices_to_sample=torch.arange(120),
                    target_positions=torch.arange(128),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=None,
                    num_tokens=128,
                )

            self.assertEqual(len(set(addresses)), 1)
        finally:
            self._reset_draft_rope_buffers()

    def test_dummy_run_wrapper_enables_and_restores_flag(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import wrap_dummy_run_with_draft_flag

        AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
        seen = []

        def original(self, *args, **kwargs):
            seen.append(AscendRotaryEmbedding310._is_drafting_update_enabled)
            return "ok"

        wrapped = wrap_dummy_run_with_draft_flag(original)
        result = wrapped(object())

        self.assertEqual(result, "ok")
        self.assertEqual(seen, [True])
        # Restored to the prior value after the call.
        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_dummy_run_wrapper_restores_flag_on_exception(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import wrap_dummy_run_with_draft_flag

        AscendRotaryEmbedding310.set_rope_position_flag_310p(False)

        def original(self, *args, **kwargs):
            raise RuntimeError("boom")

        wrapped = wrap_dummy_run_with_draft_flag(original)
        with self.assertRaises(RuntimeError):
            wrapped(object())

        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_dummy_run_wrapper_prepares_only_dflash_full_capture(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
            wrap_dummy_run_with_draft_flag,
        )

        def original(self, *args, **kwargs):
            return "ok"

        wrapped = wrap_dummy_run_with_draft_flag(original)
        builder = SimpleNamespace()
        attention_impl = SimpleNamespace()
        dflash = SimpleNamespace(
            method="dflash",
            max_num_tokens=0,
            max_query_tokens=128,
            model=SimpleNamespace(
                model=SimpleNamespace(
                    _attn_layers=[SimpleNamespace(impl=attention_impl)]
                )
            ),
            draft_attn_groups=[
                SimpleNamespace(get_metadata_builder=lambda: builder)
            ],
        )
        dspark = SimpleNamespace(
            method="dspark",
            max_num_tokens=0,
            max_query_tokens=128,
        )
        with patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310."
            "_prepare_dflash_full_graph_capture_310",
            create=True,
            side_effect=lambda owner, **kwargs: (
                setattr(builder, "_dflash_full_graph_owner_310", owner),
                setattr(
                    owner,
                    "_dflash_context_slot_mapping_by_layer_310",
                    [object()],
                ),
                setattr(
                    owner,
                    "_dflash_query_slot_mapping_by_layer_310",
                    [object()],
                ),
                setattr(
                    owner,
                    "_dflash_block_table_by_layer_310",
                    [object()],
                ),
                setattr(
                    attention_impl,
                    "_dflash_query_slot_mapping_310",
                    object(),
                ),
                setattr(
                    attention_impl,
                    "_dflash_block_table_310",
                    object(),
                ),
            ),
        ) as prepare:
            self.assertEqual(
                wrapped(
                    dflash,
                    num_tokens=32,
                    num_reqs=2,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                ),
                "ok",
            )
            prepare.assert_called_once_with(
                dflash,
                num_context=32,
                num_reqs=2,
                metadata_builders=[builder],
            )
            self.assertFalse(
                hasattr(builder, "_dflash_full_graph_owner_310")
            )
            for attr_name in (
                "_dflash_context_slot_mapping_by_layer_310",
                "_dflash_query_slot_mapping_by_layer_310",
                "_dflash_block_table_by_layer_310",
            ):
                self.assertFalse(hasattr(dflash, attr_name))
            self.assertFalse(
                hasattr(
                    attention_impl,
                    "_dflash_query_slot_mapping_310",
                )
            )
            self.assertFalse(
                hasattr(attention_impl, "_dflash_block_table_310")
            )

            prepare.reset_mock()
            wrapped(
                dflash,
                num_tokens=32,
                num_reqs=2,
                aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            )
            wrapped(
                dspark,
                num_tokens=32,
                num_reqs=2,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )
            prepare.assert_not_called()

    def test_dflash_dummy_run_reserves_context_rope_capacity(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
            wrap_dummy_run_with_draft_flag,
        )

        addresses = []

        def original(self, *args, **kwargs):
            addresses.extend(
                TestAscendSpecDecodeBaseProposer310._draft_rope_growth_addresses()
            )
            return "ok"

        proposer = SimpleNamespace(method="dflash", max_num_tokens=256)
        self._reset_draft_rope_buffers()
        try:
            self._seed_draft_rope_buffer()
            self.assertEqual(
                wrap_dummy_run_with_draft_flag(original)(proposer),
                "ok",
            )
            self.assertEqual(len(set(addresses)), 1)
        finally:
            self._reset_draft_rope_buffers()

    def test_dummy_run_wrapper_cleans_partial_full_preparation_on_error(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
            wrap_dummy_run_with_draft_flag,
        )

        builder = SimpleNamespace()
        attention_impl = SimpleNamespace()
        proposer = SimpleNamespace(
            method="dflash",
            max_num_tokens=0,
            max_query_tokens=128,
            model=SimpleNamespace(
                model=SimpleNamespace(
                    _attn_layers=[SimpleNamespace(impl=attention_impl)]
                )
            ),
            draft_attn_groups=[
                SimpleNamespace(get_metadata_builder=lambda: builder)
            ],
        )

        def fail_after_partial_preparation(owner, **kwargs):
            owner._dflash_query_slot_mapping_by_layer_310 = [object()]
            attention_impl._dflash_query_slot_mapping_310 = object()
            builder._dflash_full_graph_owner_310 = owner
            raise RuntimeError("prepare failed")

        with patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310."
            "_prepare_dflash_full_graph_capture_310",
            side_effect=fail_after_partial_preparation,
        ):
            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                wrap_dummy_run_with_draft_flag(lambda self: None)(
                    proposer,
                    num_tokens=32,
                    num_reqs=2,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                )

        self.assertFalse(
            hasattr(
                proposer,
                "_dflash_query_slot_mapping_by_layer_310",
            )
        )
        self.assertFalse(
            hasattr(attention_impl, "_dflash_query_slot_mapping_310")
        )
        self.assertFalse(
            hasattr(builder, "_dflash_full_graph_owner_310")
        )

    def test_dummy_run_wrapper_preserves_builder_lookup_error(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
            wrap_dummy_run_with_draft_flag,
        )

        builder = SimpleNamespace()
        attention_impl = SimpleNamespace()
        lookup_attempts = 0

        def failing_builder_lookup():
            nonlocal lookup_attempts
            lookup_attempts += 1
            if lookup_attempts == 1:
                raise RuntimeError("prepare builder lookup failed")
            raise RuntimeError("cleanup builder lookup failed")

        proposer = SimpleNamespace(
            method="dflash",
            max_num_tokens=0,
            max_query_tokens=128,
            model=SimpleNamespace(
                model=SimpleNamespace(
                    _attn_layers=[SimpleNamespace(impl=attention_impl)]
                )
            ),
            draft_attn_groups=[
                SimpleNamespace(get_metadata_builder=lambda: builder),
                SimpleNamespace(
                    get_metadata_builder=failing_builder_lookup
                ),
            ],
        )

        def fail_during_builder_lookup(owner, **kwargs):
            owner._dflash_query_slot_mapping_by_layer_310 = [object()]
            attention_impl._dflash_query_slot_mapping_310 = object()
            first_builder = owner.draft_attn_groups[0].get_metadata_builder()
            first_builder._dflash_full_graph_owner_310 = owner
            owner.draft_attn_groups[1].get_metadata_builder()

        with patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310."
            "_prepare_dflash_full_graph_capture_310",
            side_effect=fail_during_builder_lookup,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "prepare builder lookup failed",
            ):
                wrap_dummy_run_with_draft_flag(lambda self: None)(
                    proposer,
                    num_tokens=32,
                    num_reqs=2,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                )

        self.assertFalse(
            hasattr(
                proposer,
                "_dflash_query_slot_mapping_by_layer_310",
            )
        )
        self.assertFalse(
            hasattr(attention_impl, "_dflash_query_slot_mapping_310")
        )
        self.assertFalse(
            hasattr(builder, "_dflash_full_graph_owner_310")
        )

    def test_dspark_dummy_run_does_not_reserve_dflash_rope_capacity(self):
        from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
            wrap_dummy_run_with_draft_flag,
        )

        addresses = []

        def original(self, *args, **kwargs):
            addresses.extend(
                TestAscendSpecDecodeBaseProposer310._draft_rope_growth_addresses()
            )
            return "ok"

        proposer = SimpleNamespace(method="dspark", max_num_tokens=256)
        self._reset_draft_rope_buffers()
        try:
            self.assertEqual(
                wrap_dummy_run_with_draft_flag(original)(proposer),
                "ok",
            )
            self.assertNotEqual(addresses[0], addresses[1])
            self.assertEqual(addresses[1], addresses[2])
        finally:
            self._reset_draft_rope_buffers()

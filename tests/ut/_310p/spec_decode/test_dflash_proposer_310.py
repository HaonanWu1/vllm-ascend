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
from torch.overrides import TorchFunctionMode

from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    AscendDflashProposer310,
    _copy_and_expand_inputs_ascendc,
)


class TestCopyAndExpandInputsAscendC(TestBase):
    def _make_self(self, num_query_total, num_context):
        return SimpleNamespace(
            method="dflash",
            device=torch.device("cpu"),
            parallel_drafting_token_id=999,
            kernel_block_size=128,
            _kernel_block_size_fixed_310=True,
            num_speculative_tokens=3,
            input_ids=torch.zeros(num_query_total, dtype=torch.int32),
            positions=torch.zeros(num_query_total, dtype=torch.int32),
            _slot_mapping_buffer=torch.zeros(num_query_total, dtype=torch.int32),
            _context_positions_buffer=torch.zeros(num_context, dtype=torch.int32),
            _context_slot_mapping_buffer=torch.zeros(num_context, dtype=torch.int32),
        )

    def _run(self, fake_self, target_positions, num_context, batch_size, num_query_per_req, captured):
        num_query_total = batch_size * num_query_per_req

        cad = SimpleNamespace(
            slot_mapping=torch.zeros(num_context, dtype=torch.int32),
            query_start_loc=torch.tensor([0, num_context], dtype=torch.int32),
            seq_lens=torch.tensor([num_context], dtype=torch.int32),
            block_table_tensor=torch.zeros(batch_size, 8, dtype=torch.int32),
        )

        def fake_op(next_token_ids, tpos, *args, **kwargs):
            captured["tpos"] = tpos
            block_size = int(args[6])
            captured.setdefault("block_sizes", []).append(block_size)
            n = tpos.shape[0]
            context_slots_by_block = captured.get(
                "op_context_slots_by_block", {}
            )
            context_slots = context_slots_by_block.get(
                block_size,
                captured.get(
                    "op_context_slots",
                    torch.zeros(n, dtype=torch.int32),
                ),
            ).clone()
            query_slots = torch.full(
                (num_query_total,),
                block_size + 1000,
                dtype=torch.int32,
            )
            return (
                torch.zeros(num_query_total, dtype=torch.int32),
                torch.zeros(num_query_total, dtype=torch.int32),
                query_slots,
                torch.arange(n, dtype=torch.int32),
                context_slots,
                torch.zeros(batch_size * 3, dtype=torch.int32),
            )

        mock_ascend = MagicMock()
        mock_ascend.npu_copy_and_expand_dflash_inputs.side_effect = fake_op
        mock_ops = MagicMock()
        mock_ops._C_ascend = mock_ascend

        with patch.object(torch, "ops", mock_ops):
            _copy_and_expand_inputs_ascendc(
                fake_self,
                next_token_ids=torch.tensor([5], dtype=torch.int32),
                target_positions=target_positions,
                cad=cad,
                num_rejected_tokens_gpu=None,
                num_query_per_req=num_query_per_req,
                batch_size=batch_size,
                num_context=num_context,
                sample_from_anchor=False,
            )

    def test_mrope_positions_reduced_to_row0(self):
        # MRoPE models feed positions as [3, num_context]; the op must receive a
        # flat [num_context] vector (row 0) so the context outputs are sized by the
        # token count, not the mrope dim (which would size them as 3).
        num_context = 17
        target_positions = torch.stack(
            [
                torch.arange(num_context, dtype=torch.int32),
                torch.arange(num_context, dtype=torch.int32) + 100,
                torch.arange(num_context, dtype=torch.int32) + 200,
            ]
        )
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        captured = {}

        self._run(fake_self, target_positions, num_context, batch_size=1, num_query_per_req=4, captured=captured)

        self.assertEqual(captured["tpos"].dim(), 1)
        self.assertEqual(captured["tpos"].shape[0], num_context)
        torch.testing.assert_close(captured["tpos"], torch.arange(num_context, dtype=torch.int32))

    def test_1d_positions_passthrough(self):
        # Regular RoPE already provides a 1D [num_context] positions vector.
        num_context = 12
        target_positions = torch.arange(num_context, dtype=torch.int32)
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        captured = {}

        self._run(fake_self, target_positions, num_context, batch_size=1, num_query_per_req=4, captured=captured)

        self.assertEqual(captured["tpos"].dim(), 1)
        self.assertEqual(captured["tpos"].shape[0], num_context)

    def test_custom_op_context_slots_are_authoritative(self):
        num_context = 12
        target_positions = torch.arange(num_context, dtype=torch.int32)
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        expected_slots = torch.arange(500, 500 + num_context, dtype=torch.int32)
        captured = {"op_context_slots": expected_slots}

        self._run(fake_self, target_positions, num_context, batch_size=1, num_query_per_req=4, captured=captured)

        torch.testing.assert_close(fake_self._context_slot_mapping_buffer, expected_slots)

    def test_copy_path_does_not_subtract_query_start_locations(self):
        class OpRecorder(TorchFunctionMode):
            def __init__(self):
                self.operations = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.operations.append(func.__name__)
                return func(*args, **(kwargs or {}))

        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.kernel_block_size = 64
        fake_self.attn_layer_names = ["layer.0", "layer.1"]
        fake_self._dflash_layer_block_sizes_310 = {
            "layer.0": 64,
            "layer.1": 128,
        }
        recorder = OpRecorder()

        with recorder:
            self._run(
                fake_self,
                torch.arange(num_context, dtype=torch.int32),
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured={},
            )

        self.assertNotIn("sub", recorder.operations)

    def test_uniform_cache_layout_calls_custom_op_once(self):
        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.kernel_block_size = 64
        fake_self.attn_layer_names = ["layer.0", "layer.1", "layer.2"]
        fake_self._dflash_layer_block_sizes_310 = {
            "layer.0": 64,
            "layer.1": 64,
            "layer.2": 64,
        }
        captured = {}

        self._run(
            fake_self,
            torch.arange(num_context, dtype=torch.int32),
            num_context,
            batch_size=1,
            num_query_per_req=4,
            captured=captured,
        )

        self.assertEqual(captured["block_sizes"], [64])
        self.assertEqual(
            len(fake_self._dflash_context_slot_mapping_by_layer_310),
            3,
        )
        self.assertEqual(
            len(fake_self._dflash_query_slot_mapping_by_layer_310),
            3,
        )

    def test_heterogeneous_cache_layouts_keep_per_layer_slots(self):
        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.kernel_block_size = 64
        fake_self.attn_layer_names = ["layer.0", "layer.1", "layer.2"]
        fake_self._dflash_layer_block_sizes_310 = {
            "layer.0": 64,
            "layer.1": 64,
            "layer.2": 128,
        }
        context_64 = torch.arange(64, 64 + num_context, dtype=torch.int32)
        context_128 = torch.arange(128, 128 + num_context, dtype=torch.int32)
        captured = {
            "op_context_slots_by_block": {
                64: context_64,
                128: context_128,
            }
        }

        self._run(
            fake_self,
            torch.arange(num_context, dtype=torch.int32),
            num_context,
            batch_size=1,
            num_query_per_req=4,
            captured=captured,
        )

        self.assertEqual(captured["block_sizes"], [64, 128])
        context_by_layer = fake_self._dflash_context_slot_mapping_by_layer_310
        query_by_layer = fake_self._dflash_query_slot_mapping_by_layer_310
        torch.testing.assert_close(context_by_layer[0], context_64)
        torch.testing.assert_close(context_by_layer[1], context_64)
        torch.testing.assert_close(context_by_layer[2], context_128)
        torch.testing.assert_close(
            query_by_layer[0],
            torch.full((4,), 1064, dtype=torch.int32),
        )
        torch.testing.assert_close(query_by_layer[1], query_by_layer[0])
        torch.testing.assert_close(
            query_by_layer[2],
            torch.full((4,), 1128, dtype=torch.int32),
        )

    def test_qwen_cache_shapes_are_discovered_and_bound_per_layer(self):
        num_context = 12
        layer_names = [
            f"model.layers.{layer_index}.self_attn.attn"
            for layer_index in range(24, 29)
        ]
        physical_shapes = [
            (22840, 32, 64, 16),
            (22840, 32, 64, 16),
            (22840, 32, 64, 16),
            (11420, 32, 128, 16),
            (11420, 32, 128, 16),
        ]
        layers = {
            layer_name: SimpleNamespace(
                kv_cache=[
                    SimpleNamespace(
                        shape=shape,
                        dim=lambda shape=shape: len(shape),
                    )
                ]
            )
            for layer_name, shape in zip(layer_names, physical_shapes)
        }
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self._kernel_block_size_fixed_310 = False
        fake_self.vllm_config = object()
        fake_self.attn_layer_names = layer_names
        context_64 = torch.arange(64, 64 + num_context, dtype=torch.int32)
        context_128 = torch.arange(128, 128 + num_context, dtype=torch.int32)
        captured = {
            "op_context_slots_by_block": {
                64: context_64,
                128: context_128,
            }
        }

        with patch(
            "vllm.config.get_layers_from_vllm_config",
            return_value=layers,
        ):
            self._run(
                fake_self,
                torch.arange(num_context, dtype=torch.int32),
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured=captured,
            )

        self.assertEqual(fake_self.kernel_block_size, 64)
        self.assertEqual(
            fake_self._dflash_layer_block_sizes_310,
            dict(zip(layer_names, [64, 64, 64, 128, 128])),
        )
        self.assertEqual(captured["block_sizes"], [64, 128])
        context_by_layer = fake_self._dflash_context_slot_mapping_by_layer_310
        query_by_layer = fake_self._dflash_query_slot_mapping_by_layer_310
        self.assertEqual(len(context_by_layer), 5)
        self.assertEqual(len(query_by_layer), 5)
        for layer_index in range(3):
            torch.testing.assert_close(context_by_layer[layer_index], context_64)
            torch.testing.assert_close(
                query_by_layer[layer_index],
                torch.full((4,), 1064, dtype=torch.int32),
            )
        for layer_index in range(3, 5):
            torch.testing.assert_close(context_by_layer[layer_index], context_128)
            torch.testing.assert_close(
                query_by_layer[layer_index],
                torch.full((4,), 1128, dtype=torch.int32),
            )

        attention_layers = [
            SimpleNamespace(impl=SimpleNamespace()) for _ in layer_names
        ]
        model = MagicMock()
        model.model = SimpleNamespace(_attn_layers=attention_layers)
        fake_self._dflash_num_context = num_context
        fake_self._dflash_hidden_states = torch.randn(num_context, 8)
        fake_self.model = model
        AscendDflashProposer310.build_model_inputs_first_pass(
            fake_self,
            num_input_tokens=4,
        )

        precompute_call = model.precompute_and_store_context_kv.call_args
        self.assertIs(precompute_call.args[2], context_by_layer)
        for layer_index, attention in enumerate(attention_layers):
            self.assertIs(
                attention.impl._dflash_query_slot_mapping_310,
                query_by_layer[layer_index],
            )

    def test_heterogeneous_slot_buffers_are_allocated_once(self):
        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.kernel_block_size = 64
        fake_self.attn_layer_names = ["layer.0", "layer.1"]
        fake_self._dflash_layer_block_sizes_310 = {
            "layer.0": 64,
            "layer.1": 128,
        }

        with patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310.torch.empty_like",
            wraps=torch.empty_like,
        ) as empty_like:
            for _ in range(2):
                self._run(
                    fake_self,
                    torch.arange(num_context, dtype=torch.int32),
                    num_context,
                    batch_size=1,
                    num_query_per_req=4,
                    captured={},
                )

        # One persistent context buffer and one persistent query buffer are
        # needed for the secondary layout; replay must reuse both.
        self.assertEqual(empty_like.call_count, 2)

    def test_dspark_does_not_create_dflash_layout_state(self):
        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.method = "dspark"
        captured = {}

        self._run(
            fake_self,
            torch.arange(num_context, dtype=torch.int32),
            num_context,
            batch_size=1,
            num_query_per_req=4,
            captured=captured,
        )

        self.assertEqual(captured["block_sizes"], [128])
        self.assertFalse(
            hasattr(fake_self, "_dflash_context_slot_buffers_by_size_310")
        )
        self.assertFalse(
            hasattr(fake_self, "_dflash_query_slot_buffers_by_size_310")
        )

    def test_per_layer_slots_are_bound_to_context_and_query_writes(self):
        context_by_layer = [
            torch.tensor([64, 65], dtype=torch.int32),
            torch.tensor([128, 129], dtype=torch.int32),
        ]
        query_by_layer = [
            torch.tensor([66, 67, 68, 69], dtype=torch.int32),
            torch.tensor([130, 131, 132, 133], dtype=torch.int32),
        ]
        attention_layers = [
            SimpleNamespace(impl=SimpleNamespace()),
            SimpleNamespace(impl=SimpleNamespace()),
        ]
        model = MagicMock()
        model.model = SimpleNamespace(_attn_layers=attention_layers)
        fake_self = SimpleNamespace(
            _dflash_num_context=2,
            _dflash_hidden_states=torch.randn(2, 8),
            _context_positions_buffer=torch.tensor([10, 11], dtype=torch.int32),
            _context_slot_mapping_buffer=torch.tensor([1, 2], dtype=torch.int32),
            _dflash_context_slot_mapping_by_layer_310=context_by_layer,
            _dflash_query_slot_mapping_by_layer_310=query_by_layer,
            input_ids=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
            positions=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
            model=model,
        )

        result = AscendDflashProposer310.build_model_inputs_first_pass(
            fake_self,
            num_input_tokens=4,
        )

        model.precompute_and_store_context_kv.assert_called_once()
        call = model.precompute_and_store_context_kv.call_args
        self.assertIs(call.args[2], context_by_layer)
        self.assertIs(
            attention_layers[0].impl._dflash_query_slot_mapping_310,
            query_by_layer[0],
        )
        self.assertIs(
            attention_layers[1].impl._dflash_query_slot_mapping_310,
            query_by_layer[1],
        )
        self.assertEqual(
            result["input_ids"].data_ptr(),
            fake_self.input_ids.data_ptr(),
        )
        self.assertEqual(
            result["positions"].data_ptr(),
            fake_self.positions.data_ptr(),
        )

    def test_copy_path_captures_bounded_diagnostic_inputs(self):
        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)

        with (
            patch(
                "vllm_ascend._310p.spec_decode.dflash_proposer_310.dflash_diagnostic_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.spec_decode.dflash_proposer_310.capture_dflash_diagnostic"
            ) as capture,
        ):
            self._run(
                fake_self,
                torch.arange(num_context, dtype=torch.int32),
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured={},
            )

        capture.assert_called_once()
        (stage,) = capture.call_args.args
        self.assertEqual(stage, "draft_inputs")
        payload = capture.call_args.kwargs["payload_builder"]()
        self.assertEqual(payload["num_context"], num_context)
        self.assertIn("input_ids", payload)
        self.assertIn("positions", payload)
        self.assertIn("query_slots", payload)
        self.assertIn("context_slots", payload)
        self.assertIn("block_table", payload)

    def test_copy_path_does_not_capture_for_dspark(self):
        num_context = 12
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.method = "dspark"

        with (
            patch(
                "vllm_ascend._310p.spec_decode.dflash_proposer_310.dflash_diagnostic_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.spec_decode.dflash_proposer_310.capture_dflash_diagnostic"
            ) as capture,
        ):
            self._run(
                fake_self,
                torch.arange(num_context, dtype=torch.int32),
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured={},
            )

        capture.assert_not_called()

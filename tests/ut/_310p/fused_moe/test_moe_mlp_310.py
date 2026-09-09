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

from unittest.mock import MagicMock, call, patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.fused_moe.moe_comm_method import AllGatherCommImpl310
from vllm_ascend._310p.fused_moe.moe_mlp import _quant_grouped_matmul, quant_apply_mlp, unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType


def build_mlp_compute_input_fixture(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    with_quant: bool,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    group_list_type: int = 1,
) -> MoEMlpComputeInput:
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale),
        quant=MoEQuantParams(quant_type=QuantType.W8A8 if with_quant else QuantType.NONE),
        fusion=False,
        activation="silu",
        need_trans=False,
        dynamic_eplb=False,
    )


class TestUnifiedApplyMLP310(TestBase):
    @patch("vllm_ascend._310p.fused_moe.moe_comm_method.unified_apply_mlp")
    def test_all_gather_apply_mlp_returns_common_tuple_contract(self, mock_unified_apply_mlp):
        mlp_compute_input = MagicMock(spec=MoEMlpComputeInput)
        mlp_output = torch.randn(10, 20, dtype=torch.float16)
        mock_unified_apply_mlp.return_value = mlp_output

        comm_impl = AllGatherCommImpl310.__new__(AllGatherCommImpl310)

        output, before_gmm2_evt = comm_impl._apply_mlp(mlp_compute_input)

        self.assertIs(output, mlp_output)
        self.assertIsNone(before_gmm2_evt)
        mock_unified_apply_mlp.assert_called_once_with(mlp_compute_input=mlp_compute_input)

    @patch("torch_npu.npu_grouped_matmul", create=True)
    @patch("torch_npu.npu_swiglu")
    def test_unified_apply_mlp_without_quantization_310(self, mock_npu_swiglu, mock_npu_grouped_matmul):
        mock_gmm1_out = torch.randn(10, 40, dtype=torch.float16)
        mock_gmm2_out = torch.randn(10, 20, dtype=torch.float16)
        mock_npu_grouped_matmul.side_effect = [[mock_gmm1_out], [mock_gmm2_out]]

        mock_npu_swiglu_output = torch.randn(10, 40, dtype=torch.float16)
        mock_npu_swiglu.return_value = mock_npu_swiglu_output

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=False,
            )
        )

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_grouped_matmul.assert_has_calls(
            [
                call(
                    x=[hidden_states], weight=[w1], split_item=2, group_list_type=1, group_type=0, group_list=group_list
                ),
                call(
                    x=[mock_npu_swiglu_output],
                    weight=[w2],
                    split_item=2,
                    group_list_type=1,
                    group_type=0,
                    group_list=group_list,
                ),
            ],
            any_order=True,
        )
        mock_npu_swiglu.assert_called_once()
        mock_npu_swiglu.assert_called_with(mock_gmm1_out)

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)

    @patch("torch.cumsum")
    @patch("torch_npu.npu_quant_grouped_matmul_dequant", create=True)
    @patch("torch_npu.npu_swiglu")
    def test_unified_apply_mlp_with_quantization_310(
        self, mock_npu_swiglu, mock_npu_quant_grouped_matmul_dequant, mock_cumsum
    ):
        mock_cumsum_out = torch.arange(0, 10, dtype=torch.int64)
        mock_cumsum.return_value = mock_cumsum_out
        mock_gmm1_out = torch.randn(10, 40, dtype=torch.float16)
        mock_gmm2_out = torch.randn(10, 20, dtype=torch.float16)
        mock_npu_quant_grouped_matmul_dequant.side_effect = [mock_gmm1_out, mock_gmm2_out]

        mock_npu_swiglu_output = torch.randn(10, 40, dtype=torch.float16)
        mock_npu_swiglu.return_value = mock_npu_swiglu_output

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w1_scale = torch.rand(5, 40, dtype=torch.float32)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        w2_scale = torch.rand(5, 40, dtype=torch.float32)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        )

        mock_cumsum.assert_called_once()
        self.assertEqual(mock_npu_quant_grouped_matmul_dequant.call_count, 2)
        mock_npu_quant_grouped_matmul_dequant.assert_has_calls(
            [
                call(
                    x=hidden_states,
                    quantized_weight=w1,
                    weight_scale=w1_scale,
                    group_list=mock_cumsum_out,
                    quant_mode="pertoken",
                    x_scale=None,
                ),
                call(
                    x=mock_npu_swiglu_output,
                    quantized_weight=w2,
                    weight_scale=w2_scale,
                    group_list=mock_cumsum_out,
                    quant_mode="pertoken",
                    x_scale=None,
                ),
            ],
            any_order=True,
        )
        mock_npu_swiglu.assert_called_once()
        mock_npu_swiglu.assert_called_with(mock_gmm1_out)

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)


@pytest.fixture
def quant_gmm_mocks():
    with (
        patch("torch_npu.get_npu_format", return_value=29, create=True) as get_format,
        patch("torch_npu.npu_quant_grouped_matmul_dequant", create=True) as gmm,
    ):
        yield get_format, gmm


def make_quant_gmm_inputs(
    rows=640,
    k=256,
    n=2048,
    experts=256,
    x_dtype=torch.float16,
    weight_dtype=torch.int8,
    scale_dtype=torch.float32,
    group_dtype=torch.int64,
):
    x = torch.full((rows, k), 0.5, dtype=x_dtype)
    # Metadata-only weights avoid allocating hundreds of MB in CPU unit tests.
    weight = torch.empty((experts, n, k), dtype=weight_dtype, device="meta")
    scale = torch.empty((experts, n), dtype=scale_dtype, device="meta")
    groups = torch.empty(experts, dtype=group_dtype, device="meta")
    return x, weight, scale, groups


@pytest.mark.parametrize("k,n", [(2048, 512), (256, 2048)])
def test_batch_scale_is_per_row_and_preserves_gmm_arguments(quant_gmm_mocks, k, n):
    get_format, gmm = quant_gmm_mocks
    x, weight, weight_scale, groups = make_quant_gmm_inputs(k=k, n=n)
    # Cover all-zero, negative maximum, subnormal and largest finite FP16 rows.
    x[0] = 0
    x[1, 0] = -2
    x[2] = torch.finfo(torch.float16).smallest_normal / 2
    x[3, 0] = torch.finfo(torch.float16).max
    original = x.clone()

    result = _quant_grouped_matmul(x, weight, weight_scale, groups)

    kwargs = gmm.call_args.kwargs
    assert result is gmm.return_value
    assert kwargs["x"] is x
    assert kwargs["quantized_weight"] is weight
    assert kwargs["weight_scale"] is weight_scale
    assert kwargs["group_list"] is groups
    assert kwargs["quant_mode"] == "pertoken"
    assert set(kwargs) == {"x", "quantized_weight", "weight_scale", "group_list", "quant_mode", "x_scale"}
    expected = original.float().abs().amax(dim=-1) / 127.0
    expected[0] = 1
    torch.testing.assert_close(kwargs["x_scale"], expected, rtol=0, atol=0)
    assert kwargs["x_scale"].shape == (640,)
    assert kwargs["x_scale"].dtype == torch.float32
    assert kwargs["x_scale"].device == x.device
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    get_format.assert_called_once_with(weight)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rows": 0},
        {"rows": 80},
        {"rows": 639},
        {"rows": 641},
        {"rows": 1280},  # K15/C10 must not activate the K7/C10 shape guard.
        {"k": 128},
        {"n": 1024},
        {"experts": 128},
        {"x_dtype": torch.float32},
        {"x_dtype": torch.bfloat16},
        {"weight_dtype": torch.float16},
        {"scale_dtype": torch.float16},
        {"group_dtype": torch.int32},
    ],
)
def test_unvalidated_contract_uses_internal_scale(quant_gmm_mocks, overrides):
    get_format, gmm = quant_gmm_mocks
    inputs = make_quant_gmm_inputs(**overrides)
    _quant_grouped_matmul(*inputs)
    assert gmm.call_args.kwargs["x_scale"] is None
    get_format.assert_not_called()


def test_non_nz_weight_uses_internal_scale(quant_gmm_mocks):
    get_format, gmm = quant_gmm_mocks
    get_format.return_value = 2
    _quant_grouped_matmul(*make_quant_gmm_inputs())
    assert gmm.call_args.kwargs["x_scale"] is None


def test_batch_scale_is_recomputed_for_each_call(quant_gmm_mocks):
    _, gmm = quant_gmm_mocks
    x, weight, scale, groups = make_quant_gmm_inputs()
    _quant_grouped_matmul(x, weight, scale, groups)
    first_scale = gmm.call_args.kwargs["x_scale"]
    x.mul_(2)
    _quant_grouped_matmul(x, weight, scale, groups)
    torch.testing.assert_close(gmm.call_args.kwargs["x_scale"], first_scale * 2, rtol=0, atol=0)


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_quant_mlp_computes_separate_scales_around_swiglu(quant_gmm_mocks, group_list_type):
    _, gmm = quant_gmm_mocks
    x, w1, w1_scale, _ = make_quant_gmm_inputs(k=2048, n=512)
    act_out, w2, w2_scale, _ = make_quant_gmm_inputs()
    act_out.mul_(4)
    counts = torch.zeros(256, dtype=torch.int64)
    counts[-1] = x.shape[0]
    groups = counts if group_list_type == 1 else counts.cumsum(dim=0)
    expected_groups = counts.cumsum(dim=0)
    gmm1_out = torch.empty((640, 512), dtype=torch.float16)
    gmm2_out = torch.empty_like(x)
    gmm.side_effect = [gmm1_out, gmm2_out]

    with patch("torch_npu.npu_swiglu", return_value=act_out) as swiglu:
        result = quant_apply_mlp(x, w1, w1_scale, w2, w2_scale, groups, group_list_type)

    assert result is gmm2_out
    swiglu.assert_called_once_with(gmm1_out)
    assert gmm.call_count == 2
    first, second = [c.kwargs for c in gmm.call_args_list]
    assert first["x"] is x
    assert second["x"] is act_out
    torch.testing.assert_close(first["x_scale"], torch.full((640,), 0.5 / 127.0), rtol=0, atol=0)
    torch.testing.assert_close(second["x_scale"], torch.full((640,), 2.0 / 127.0), rtol=0, atol=0)
    torch.testing.assert_close(first["group_list"], expected_groups)
    torch.testing.assert_close(second["group_list"], expected_groups)

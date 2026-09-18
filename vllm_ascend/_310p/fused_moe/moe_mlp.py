# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.


import torch
import torch_npu

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput

_BATCH_SCALE_GATE_UP_X_SHAPE = (640, 2048)
_BATCH_SCALE_GATE_UP_WEIGHT_SHAPE = (256, 512, 2048)
_BATCH_SCALE_DOWN_X_SHAPE = (640, 256)
_BATCH_SCALE_DOWN_WEIGHT_SHAPE = (256, 2048, 256)
_FRACTAL_NZ_FORMAT = 29
_INT8_QUANT_MAX = 127.0


def _supports_named_quant_gmm(x, weight, weight_scale, group_list) -> bool:
    """Layout guard after the validated shape/dtype guard; metadata only."""
    return (
        x.device.type == "npu"
        and tuple(weight_scale.shape) == tuple(weight.shape[:2])
        and tuple(group_list.shape) == (weight.shape[0],)
        and all(t.device == x.device and t.is_contiguous() for t in (x, weight, weight_scale, group_list))
        and all(torch_npu.get_npu_format(t) in (0, 2) for t in (x, weight_scale, group_list))
    )


def _quant_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    # Only use batch scale calculation for shapes validated on 310P. Keep
    # other workloads on CANN's internal per-token dynamic-quant path.
    supported_shape = (
        tuple(x.shape) == _BATCH_SCALE_GATE_UP_X_SHAPE and tuple(weight.shape) == _BATCH_SCALE_GATE_UP_WEIGHT_SHAPE
    ) or (tuple(x.shape) == _BATCH_SCALE_DOWN_X_SHAPE and tuple(weight.shape) == _BATCH_SCALE_DOWN_WEIGHT_SHAPE)
    x_scale = None
    if (
        supported_shape
        and x.dtype == torch.float16
        and weight.dtype == torch.int8
        and weight_scale.dtype == torch.float32
        and group_list.dtype == torch.int64
        and torch_npu.get_npu_format(weight) == _FRACTAL_NZ_FORMAT
    ):
        # Reduce each row independently before entering the grouped kernel.
        # Keep FP32 scale division and a nonzero scale for all-zero rows.
        absmax = x.abs().amax(dim=-1).float()
        x_scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / _INT8_QUANT_MAX)
        # Custom ops are loaded by worker initialization. An older extension
        # or unsupported tensor layout keeps the original CANN call below.
        # This dispatch uses metadata only, never a device-to-host group read.
        custom_op = getattr(torch.ops._C_ascend, "npu_quant_grouped_matmul_dequant_310", None)
        if custom_op is not None and _supports_named_quant_gmm(x, weight, weight_scale, group_list):
            return custom_op(x, weight, weight_scale, group_list, x_scale)
    return torch_npu.npu_quant_grouped_matmul_dequant(
        x=x,
        quantized_weight=weight,
        weight_scale=weight_scale,
        group_list=group_list,
        quant_mode="pertoken",
        x_scale=x_scale,
    )


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
) -> torch.Tensor:
    if group_list_type == 1:
        # Convert group_list to cumulative sum format if group_list is count format
        group_list = torch.cumsum(group_list, dim=0)

    hidden_states = _quant_grouped_matmul(hidden_states, w1, w1_scale, group_list)
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    hidden_states = _quant_grouped_matmul(hidden_states, w2, w2_scale, group_list)
    return hidden_states


def unquant_apply_mlp(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, group_list: torch.Tensor, group_list_type: int = 1
) -> torch.Tensor:
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    act_out = torch_npu.npu_swiglu(gate_up_out)

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[act_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    hidden_states = mlp_compute_input.hidden_states
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    assert isinstance(w1, torch.Tensor)
    assert isinstance(w2, torch.Tensor)

    if mlp_compute_input.quant.is_quant:
        assert isinstance(w1_scale, torch.Tensor)
        assert isinstance(w2_scale, torch.Tensor)
        assert w1_scale is not None and w2_scale is not None
        return quant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    return unquant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        group_list=group_list,
        group_list_type=group_list_type,
    )

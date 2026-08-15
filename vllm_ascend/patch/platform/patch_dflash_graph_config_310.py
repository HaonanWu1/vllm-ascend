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
# This file is a part of the vllm-ascend project.
#

from typing import Any

from vllm.config import CUDAGraphMode
from vllm.logger import logger

from vllm_ascend.platform import NPUPlatform

DFLASH_REQUESTED_GRAPH_MODE_ATTR_310 = (
    "_dflash_requested_cudagraph_mode_310"
)
DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310 = (
    "_dflash_effective_cudagraph_mode_310"
)
_DFLASH_FULL_ATB_OPERATION_310 = "ATB PagedAttentionOperation"
_DFLASH_FULL_ATB_REASON_310 = (
    "qLensTensor hostData is required to build splitfuse parameters, "
    "but ACL graph replay cannot refresh capture-time host qLens values"
)
_QWEN36_PIECEWISE_OPERATION_310 = "HCCL ACL graph capture events"
_QWEN36_PIECEWISE_REASON_310 = (
    "the required ten-gear TP2 capture exhausts HCCL events with "
    "Insufficient_Event_Resources (EL0008); the three tested two-gear "
    "controls do not meet the DFlash graph performance gate"
)
_QWEN36_FULL_DECODE_OPERATION_310 = (
    "CANN GatherV2 draft FULL decode replay"
)
_QWEN36_FULL_DECODE_REASON_310 = (
    "multi-request 112/128-token replay triggers an MTE DDR address "
    "out-of-range AICore exception (507011)"
)


def _is_qwen36_k15_graph_profile_310(vllm_config: Any) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if (
        getattr(speculative_config, "method", None) != "dflash"
        or getattr(speculative_config, "num_speculative_tokens", None) != 15
        or getattr(
            speculative_config,
            "draft_tensor_parallel_size",
            None,
        )
        != 2
        or getattr(compilation_config, "cudagraph_mode", None)
        in (None, CUDAGraphMode.NONE)
    ):
        return False

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    layer_types = getattr(text_config, "layer_types", ()) or ()
    return (
        getattr(parallel_config, "tensor_parallel_size", None) == 2
        and getattr(hf_config, "model_type", None) == "qwen3_5_moe"
        and getattr(text_config, "model_type", None)
        == "qwen3_5_moe_text"
        and getattr(text_config, "num_hidden_layers", None) == 40
        and getattr(text_config, "num_experts", None) == 256
        and getattr(text_config, "linear_num_value_heads", None) == 32
        and "full_attention" in layer_types
    )


def _resolve_qwen36_quantization_310(vllm_config: Any) -> None:
    if not _is_qwen36_k15_graph_profile_310(vllm_config):
        return
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "quantization", None) is not None:
        return

    from vllm_ascend.quantization.utils import (
        maybe_auto_detect_quantization,
    )

    maybe_auto_detect_quantization(vllm_config)


def _is_qwen36_w8a8_k15_graph_310(vllm_config: Any) -> bool:
    return (
        _is_qwen36_k15_graph_profile_310(vllm_config)
        and getattr(vllm_config.model_config, "quantization", None)
        == "ascend"
    )


def _qwen36_graph_limitation_310(
    requested_mode: CUDAGraphMode,
) -> tuple[str, str]:
    operations = []
    reasons = []
    if requested_mode == CUDAGraphMode.FULL:
        operations.append(_DFLASH_FULL_ATB_OPERATION_310)
        reasons.append(_DFLASH_FULL_ATB_REASON_310)
    if requested_mode in (
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL,
    ):
        operations.append(_QWEN36_PIECEWISE_OPERATION_310)
        reasons.append(_QWEN36_PIECEWISE_REASON_310)
    if requested_mode in (
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL,
    ):
        operations.append(_QWEN36_FULL_DECODE_OPERATION_310)
        reasons.append(_QWEN36_FULL_DECODE_REASON_310)
    return "; ".join(operations), "; ".join(reasons)


def _apply_qwen36_dflash_graph_fallback_310(vllm_config: Any) -> None:
    """Disable graph execution for the exact unsupported Qwen3.6 profile."""
    if not _is_qwen36_w8a8_k15_graph_310(vllm_config):
        return

    compilation_config = vllm_config.compilation_config
    requested_mode = compilation_config.cudagraph_mode
    effective_mode = CUDAGraphMode.NONE
    operations, reason = _qwen36_graph_limitation_310(requested_mode)
    setattr(
        compilation_config,
        DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
        requested_mode,
    )
    setattr(
        compilation_config,
        DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310,
        effective_mode,
    )
    compilation_config.cudagraph_mode = effective_mode
    logger.warning(
        "310P DFlash graph fallback: requested_mode=%s, "
        "effective_mode=%s, backend_operations=%s, reason=%s",
        requested_mode.name,
        effective_mode.name,
        operations,
        reason,
    )


def _requires_atb_splitfuse_310(vllm_config: Any) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if (
        getattr(speculative_config, "method", None) != "dflash"
        or getattr(compilation_config, "cudagraph_mode", None)
        != CUDAGraphMode.FULL
    ):
        return False

    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    layer_types = getattr(text_config, "layer_types", ()) or ()
    return "full_attention" in layer_types


def _apply_dflash_full_atb_fallback_310(vllm_config: Any) -> None:
    """Resolve the host-qLens limitation before compile partitions are set."""
    if not _requires_atb_splitfuse_310(vllm_config):
        return

    compilation_config = vllm_config.compilation_config
    requested_mode = CUDAGraphMode.FULL
    effective_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    setattr(
        compilation_config,
        DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
        requested_mode,
    )
    setattr(
        compilation_config,
        DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310,
        effective_mode,
    )
    compilation_config.cudagraph_mode = effective_mode
    logger.warning(
        "310P DFlash graph fallback: requested_mode=%s, "
        "effective_mode=%s, backend_operation=%s, reason=%s",
        requested_mode.name,
        effective_mode.name,
        _DFLASH_FULL_ATB_OPERATION_310,
        _DFLASH_FULL_ATB_REASON_310,
    )


_original_check_and_update_config_310 = (
    NPUPlatform.check_and_update_config.__func__
)


def check_and_update_config_310(cls, vllm_config) -> None:
    _resolve_qwen36_quantization_310(vllm_config)
    _apply_qwen36_dflash_graph_fallback_310(vllm_config)
    _apply_dflash_full_atb_fallback_310(vllm_config)
    _original_check_and_update_config_310(cls, vllm_config)


NPUPlatform.check_and_update_config = classmethod(
    check_and_update_config_310
)

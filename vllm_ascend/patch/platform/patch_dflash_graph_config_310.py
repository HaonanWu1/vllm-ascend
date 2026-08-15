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
_DFLASH_FULL_ATB_OPERATION_310 = "ATB PagedAttentionOperation"
_DFLASH_FULL_ATB_REASON_310 = (
    "qLensTensor hostData is required to build splitfuse parameters, "
    "but ACL graph replay cannot refresh capture-time host qLens values"
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
    _apply_dflash_full_atb_fallback_310(vllm_config)
    _original_check_and_update_config_310(cls, vllm_config)


NPUPlatform.check_and_update_config = classmethod(
    check_and_update_config_310
)

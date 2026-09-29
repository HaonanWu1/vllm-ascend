# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend import platform, utils


@pytest.mark.parametrize(
    "hardware,mla,expected",
    [
        (True, False, "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310"),
        (False, False, "vllm_ascend.attention.attention_v1.AscendAttentionBackend"),
        (False, True, "vllm_ascend.attention.mla_v1.AscendMLABackend"),
    ],
)
def test_attention_backend_follows_explicit_hardware(hardware, mla, expected):
    """T01：按明确硬件/MLA输入选择对应backend，防止310P行为污染公共路径。

    输入：参数表的310P或非310P、普通attention或公共MLA。
    输出：逐字核对完整backend类路径；调用真实NPUPlatform分发函数。
    替身：只替换硬件身份查询，不替换被测分发；不宣称实际attention kernel正确。
    """
    config = SimpleNamespace(use_compress=False, use_mla=mla, use_sparse=False)
    with patch.object(platform, "is_310p", return_value=hardware):
        assert platform.NPUPlatform.get_attn_backend_cls("ascend", config) == expected


@pytest.mark.parametrize(
    "hardware,mode,dtype,expected",
    [
        (True, 0, torch.float16, True),
        (True, 1, torch.float16, True),
        (True, 2, torch.float16, True),
        (True, 2, torch.float32, False),
        (False, 0, torch.float16, False),
        (False, 1, torch.float16, False),
        (False, 2, torch.float16, True),
    ],
)
def test_nz_policy_with_real_weights(hardware, mode, dtype, expected):
    """T01：真实权重张量上的NZ选择guard与host输出接线。

    输入：2x4具体权重、hardware/mode/dtype参数表；310P FP16始终NZ、FP32不转。
    输出：转换次数、format号与输入tensor identity准确，未转换时返回原tensor且数值不变。
    依据：平台格式合同；仅NPU format cast是外部设备替身，返回clone便于验证选择。
    局限：不能证明NZ storage布局；该项须真实NPU另测。
    """
    weight = torch.arange(8, dtype=dtype).reshape(2, 4)
    with (
        patch.object(utils, "is_310p", return_value=hardware),
        patch.object(utils, "get_ascend_config", return_value=SimpleNamespace(weight_nz_mode=mode)),
        patch.object(utils.torch_npu, "npu_format_cast", side_effect=lambda tensor, _: tensor.clone()) as cast,
    ):
        result = utils.maybe_trans_nz(weight)
    torch.testing.assert_close(result, weight)
    if expected:
        cast.assert_called_once_with(weight, utils.ACL_FORMAT_FRACTAL_NZ)
        assert result.data_ptr() != weight.data_ptr()
    else:
        cast.assert_not_called()
        assert result is weight

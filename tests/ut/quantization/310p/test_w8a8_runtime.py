# SPDX-License-Identifier: Apache-2.0
"""T18: real dynamic quantization and matmul with an exactly quantizable oracle."""

import pytest
import torch
import torch_npu

from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod310


@pytest.mark.parametrize("rows", [1, 1023, 1024, 1025])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("three_dimensional", [False, True])
def test_real_w8a8_linear_crosses_former_chunk_boundary(rows, with_bias, three_dimensional):
    """T18-01：真实W8A8入口跨旧1024分块边界，bias与[N,1,K]还原。

    输入：K128/N64，行数1/1023/1024/1025；X由整数[-127,127]乘精确2^-5构成，
      每行首列127保证动态scale恰为2^-5；int8权重[-3,3]，每channel scale=2^-4；
      bias为设备ABI支持的INT32累加域128，反量化后为.25，另测浮点bias拒绝边界。
    输出：真实NPU结果等于CPU int32 matmul乘2^-9再加bias并转FP16；shape/dtype保留输入合同。
    场景：load后真实NZ变换与转置、动态quant、matmul、3D squeeze/unsqueeze完整链路。
    依据：可精确量化输入使CPU参考无需复用NPU动态quant函数，避免两边共用同一错误。
    替身：无；所有生产算子真实执行。输入/权重原值另留CPU副本作不变性检查。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    rng = torch.Generator().manual_seed(20260922 + rows)
    integer_x = torch.randint(-127, 128, (rows, 128), generator=rng, dtype=torch.int32)
    integer_x[:, 0] = 127
    integer_w = torch.randint(-3, 4, (64, 128), generator=rng, dtype=torch.int8)
    input_cpu = (integer_x.float() / 32).half()
    if three_dimensional:
        input_cpu = input_cpu.unsqueeze(1)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(integer_w.npu(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.full((64, 1), 1 / 16, device="npu", dtype=torch.float32), requires_grad=False
    )
    layer.weight_offset = torch.nn.Parameter(
        torch.zeros((64, 1), device="npu", dtype=torch.float32), requires_grad=False
    )
    method = AscendW8A8DynamicLinearMethod310()
    method.process_weights_after_loading(layer)
    bias = torch.full((64,), 128, device="npu", dtype=torch.int32) if with_bias else None
    x = input_cpu.npu()
    actual = method.apply(layer, x, bias=bias)
    torch.npu.synchronize()
    expected = (integer_x @ integer_w.int().T).float() / 512
    if with_bias:
        expected += 0.25
    if three_dimensional:
        expected = expected.unsqueeze(1)
    assert actual.device.type == "npu" and actual.dtype == torch.float16
    torch.testing.assert_close(actual.cpu(), expected.half(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.cpu(), input_cpu, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.cpu().T, integer_w, rtol=0, atol=0)


def test_w8a8_float_bias_reports_current_device_abi_limit():
    """T18-02：浮点bias的当前310P WeightNz ABI限制单独记录。

    输入：FP16 X[1,128]、int8权重、FP32 scale和FP32 bias[64]。
    输出：真实apply经CANN参数检查抛RuntimeError，明确要求Bias dtype INT32。
    场景：不能把参数被传入就算浮点bias支持；这是当前能力限制，正常数值用例使用INT32。
    替身：无；没有改生产逻辑来绕过检查。此测试通过不代表模型层浮点bias功能通过。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(64, 128, device="npu", dtype=torch.int8), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.ones(64, 1, device="npu", dtype=torch.float32), requires_grad=False)
    layer.weight_offset = torch.nn.Parameter(torch.zeros(64, 1, device="npu", dtype=torch.float32), requires_grad=False)
    method = AscendW8A8DynamicLinearMethod310()
    method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="Bias dtype should be INT32"):
        method.apply(
            layer,
            torch.ones(1, 128, device="npu", dtype=torch.float16),
            bias=torch.full((64,), 0.25, device="npu", dtype=torch.float32),
        )

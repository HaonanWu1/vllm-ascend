# SPDX-License-Identifier: Apache-2.0
"""Real 310P Qwen3.5 target RoPE and W8A8 shape regressions; no mocked operators."""

import pytest
import torch
import torch_npu

from tests.ut._310p.spec_decode.test_qwen35_rope_contracts import (
    TARGET_HEADS,
    independent_rotation,
    rotary_fixture,
)
from vllm_ascend._310p.ops import rotary_embedding as rope
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod310


@pytest.mark.parametrize("model,q_heads,kv_heads", TARGET_HEADS)
def test_qwen35_real_target_mrope_preserves_unrotated_tail(model, q_heads, kv_heads, monkeypatch):
    """输入：五种TP2主模型Q/K头数、FP16 head256/rotary64，非相等T/H/W坐标。

    输出：真实AscendMRotaryEmbedding310.forward_oot和NPU算子结果，与CPU双精度
    复数参考比较；后192维严格保持；长批8→短批3→长批8复用同一buffer地址。
    场景：真实部分RoPE、GQA、缩批后位置刷新；无算子mock，不以仅形状断言代替数值。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    monkeypatch.setattr(rope, "_mrope_cos_slice", None)
    monkeypatch.setattr(rope, "_mrope_sin_slice", None)
    emb = rotary_fixture("npu", torch.float16)
    gen = torch.Generator().manual_seed(928)
    q_cpu = torch.randn(8, q_heads * 256, generator=gen).half()
    k_cpu = torch.randn(8, kv_heads * 256, generator=gen).half()
    pointer = None
    for count, offset in [(8, 0), (3, 31), (8, 63)]:
        positions = torch.stack([torch.arange(count) + offset + axis * 7 for axis in range(3)])
        rope.set_mrope_apply_rotary_slices(
            emb.cos_sin_cache, positions.npu(), mrope_section=[11, 11, 10], mrope_interleaved=True, capacity_tokens=16
        )
        pointer = pointer or rope._mrope_cos_slice.data_ptr()
        assert pointer == rope._mrope_cos_slice.data_ptr()
        actual = rope.AscendMRotaryEmbedding310.forward_oot(
            emb, positions.npu(), q_cpu[:count].npu(), k_cpu[:count].npu()
        )
        torch.npu.synchronize()
        for source, output in zip((q_cpu[:count], k_cpu[:count]), actual):
            torch.testing.assert_close(output.cpu(), independent_rotation(source, positions), rtol=3e-3, atol=3e-3)
            torch.testing.assert_close(
                output.cpu().reshape(count, -1, 256)[..., 64:], source.reshape(count, -1, 256)[..., 64:], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "model,hidden,output_features",
    [("2B", 2048, 2560), ("4B", 2560, 5120), ("9B", 4096, 5120), ("27B", 5120, 7168), ("35B-A3B", 2048, 4608)],
)
@pytest.mark.parametrize("rows", [1, 8, 17])
def test_qwen35_w8a8_hidden_width_matches_integer_matmul(model, hidden, output_features, rows):
    """输入：五组实际TP2 gated-QKV投影尺寸，decode1/K7验证8/非对齐17行。

    权重[N,H]中的N=(2*local_Q_heads+2*local_KV_heads)*256，含Qwen3.5 attention gate；
    来自Qwen3NextAttention的QKVParallelLinear和下载配置，而非缩小成统一输出宽度。
    X由整数乘2^-5构造，各行有127，真实动态量化scale应恰为2^-5；权重scale2^-4。
    输出：真实生产process_weights_after_loading→apply结果等于CPU整数矩阵乘2^-9，
    且shape/dtype正确。场景：不同模型宽度、NZ转换、动态quant与matmul真实组合。
    无mock；合成可精确量化权重验证真实投影尺寸，不冒充模型权重质量或所有MoE专家覆盖。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    gen = torch.Generator().manual_seed(928 + hidden + rows)
    x_int = torch.randint(-127, 128, (rows, hidden), generator=gen, dtype=torch.int32)
    x_int[:, 0] = 127
    w_int = torch.randint(-3, 4, (output_features, hidden), generator=gen, dtype=torch.int8)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_int.npu(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.full((output_features, 1), 1 / 16, device="npu"), requires_grad=False)
    layer.weight_offset = torch.nn.Parameter(torch.zeros(output_features, 1, device="npu"), requires_grad=False)
    method = AscendW8A8DynamicLinearMethod310()
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, (x_int.float() / 32).half().npu())
    torch.npu.synchronize()
    expected = ((x_int @ w_int.int().T).float() / 512).half()
    assert actual.dtype == torch.float16 and actual.shape == (rows, output_features)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)

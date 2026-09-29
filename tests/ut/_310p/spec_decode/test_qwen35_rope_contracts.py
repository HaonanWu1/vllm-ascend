# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 position contracts, using independent trigonometric references.

These are computation UTs, not checkpoint-loading or model-quality tests.
The device suite reuses only the CPU oracle and immutable shape parameters.
"""

import cmath
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.ops import rotary_embedding as rope
from vllm_ascend._310p.spec_decode.dflash_mrope import DFlashMRoPEState310, build_dflash_mrope_positions

# TP2 local Q/KV heads from text_config; head_dim=256. Share identical 4B/9B shapes.
TARGET_HEADS = [("2B", 4, 1), ("4B+9B", 8, 2), ("27B", 12, 2), ("35B-A3B", 8, 1)]


def rotary_fixture(device="cpu", dtype=torch.float32):
    frequencies = torch.arange(32, dtype=torch.float64) / 32
    angles = torch.arange(256, dtype=torch.float64)[:, None] / (10_000_000**frequencies)
    return SimpleNamespace(
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1).to(device=device, dtype=dtype),
        head_size=256,
        rotary_dim=64,
        is_neox_style=True,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )


def independent_rotation(value, positions):
    """Scalar complex multiplication; does not use production gather/rotate/cache."""
    source = value.double().reshape(value.shape[0], -1, 256)
    result = source.clone()
    for token in range(source.shape[0]):
        for frequency in range(32):
            # Qwen3.5 training layout: H at 1,4,...,31; W at 2,5,...,29.
            axis = 1 if frequency % 3 == 1 else (2 if frequency % 3 == 2 else 0)
            phase = int(positions[axis, token]) / (10_000_000 ** (frequency / 32))
            rotation = cmath.exp(1j * phase)
            for head in range(source.shape[1]):
                z = complex(source[token, head, frequency], source[token, head, frequency + 32]) * rotation
                result[token, head, frequency] = z.real
                result[token, head, frequency + 32] = z.imag
    return result.reshape_as(value).to(value.dtype)


@pytest.mark.parametrize("model,q_heads,kv_heads", TARGET_HEADS)
@pytest.mark.parametrize("context", [False, True])
def test_qwen35_partial_rotation_and_gqa_match_scalar_oracle(model, q_heads, kv_heads, context):
    """输入：五个主模型的 TP2 Q/KV 形状，head256/rotary64，不同 T/H/W 坐标。

    输出：Q/K 前64维与独立复数乘法参考一致，后192维逐位不变，输入不被改写。
    场景：GQA不同Q/K head数、部分旋转、draft-local context/query隔离。
    替身：仅轻量配置对象；生产位置选取及旋转运算真实执行，不mock数值函数。
    限制：这是共享mRoPE数学合同测试，不声称这些主模型通过draft-local状态执行。
    """
    gen = torch.Generator().manual_seed(928)
    query = torch.randn(3, q_heads * 256, generator=gen)
    key = torch.randn(3, kv_heads * 256, generator=gen)
    query_before, key_before = query.clone(), key.clone()
    positions = torch.tensor([[0, 17, 127], [3, 18, 61], [7, 19, 31]], dtype=torch.int32)
    state = DFlashMRoPEState310(rotary_fixture(), 8)
    state.refresh(positions if not context else positions + 1, positions if context else positions + 2)
    actual = state.apply(query, key, context=context)
    for source, output in zip((query, key), actual):
        torch.testing.assert_close(output, independent_rotation(source, positions), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            output.reshape(3, -1, 256)[..., 64:], source.reshape(3, -1, 256)[..., 64:], rtol=0, atol=0
        )
    torch.testing.assert_close(query, query_before, rtol=0, atol=0)
    torch.testing.assert_close(key, key_before, rtol=0, atol=0)


@pytest.mark.parametrize("width", [2, 4, 8, 16])
@pytest.mark.parametrize("rejected", [None, [-1, 0, 1], [1, 7, 99]])
def test_qwen35_mixed_prefill_decode_positions_use_logical_cache_offsets(width, rejected):
    """输入：三个请求，已缓存前缀、1/3/2个本次token、INT32负mRoPE delta，K1/3/7/15。

    输出：cache地址固定[16,97,98,99,254,255]；query按接受末尾+delta递增，
    负坐标归零；拒绝数先夹到[0,本次token数]。三轴文本query一致且int32。
    场景：长短混批、未完成图像prefill、全拒绝/过大拒绝数、K边界。
    依据：Python标量计算；输入图像坐标刻意不等于物理cache位置。无mock。
    """
    positions = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 2, 3, 3, 7, 7], [0, 1, 0, 1, 2, 3]])
    counts, lengths, deltas = [1, 3, 2], [17, 100, 256], [-30, -50, -7]
    cache, query = build_dflash_mrope_positions(
        positions,
        torch.tensor([0, 1, 4, 6]),
        torch.tensor(lengths),
        None if rejected is None else torch.tensor(rejected),
        torch.tensor(deltas, dtype=torch.int32),
        6,
        width,
    )
    expected = []
    for i in range(3):
        n = 0 if rejected is None else min(counts[i], max(0, rejected[i]))
        start = max(0, lengths[i] - n + deltas[i])
        expected.extend(range(start, start + width))
    assert cache.tolist() == [16, 97, 98, 99, 254, 255]
    assert query.tolist() == [expected] * 3
    assert cache.dtype == query.dtype == torch.int32


def test_qwen35_target_slices_and_draft_128_slices_remain_independent(monkeypatch):
    """输入：target rotary64三轴坐标与draft rotary128一轴坐标，随后target缩批。

    输出：target与draft的buffer地址分别稳定；更新target不能修改draft的cos/sin。
    场景：Qwen3.5主模型head256/rotary64和Qwen3草稿head128/rotary128共存。
    替身：monkeypatch仅隔离模块级缓存，实际gather/copy/三轴合并均真实执行。
    """
    for name in ("_mrope_cos_slice", "_mrope_sin_slice", "_draft_cos", "_draft_sin", "_draft_rope_dim"):
        monkeypatch.setattr(rope, name, None)
    monkeypatch.setattr(rope, "_draft_min_capacity_tokens", 16)
    target = rotary_fixture()
    positions = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rope.set_mrope_apply_rotary_slices(
        target.cos_sin_cache, positions, mrope_section=[11, 11, 10], mrope_interleaved=True, capacity_tokens=16
    )
    target_ptr = rope._mrope_cos_slice.data_ptr()
    freq = torch.arange(64).float() / 64
    angles = torch.arange(256).float()[:, None] / (10_000_000**freq)
    draft_cache = torch.cat((angles.cos(), angles.sin()), -1)
    cos, sin = rope._build_draft_cos_sin_slice(draft_cache, torch.tensor([11, 12, 13]))
    draft_ptr = cos.data_ptr()
    before = (cos.clone(), sin.clone())
    rope.set_mrope_apply_rotary_slices(
        target.cos_sin_cache,
        positions[:, :1] + 20,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        capacity_tokens=16,
    )
    assert target_ptr == rope._mrope_cos_slice.data_ptr()
    assert draft_ptr == cos.data_ptr() and target_ptr != draft_ptr
    torch.testing.assert_close(cos, before[0], rtol=0, atol=0)
    torch.testing.assert_close(sin, before[1], rtol=0, atol=0)
    assert cos.shape[-1] == 128 and rope._mrope_cos_slice.shape[-1] == 64

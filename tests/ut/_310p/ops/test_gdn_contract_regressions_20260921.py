"""Regression UTs for 310P GDN shape, padding, and WY contracts.

These tests execute the real CPU reference helpers from the production modules.
They do not claim coverage of the AscendC kernels; kernel coverage remains an
NPU functional-test responsibility.
"""

import pytest
import torch

from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as gdn
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import CHUNK_SIZE


def test_gdn_expands_grouped_query_heads_without_copying_semantics():
    """
    测试功能：验证 GDN grouped-query 的 q/k head 扩展保持每个源 head 的值。
    对应修改：92dfbff2a、c6ff47548；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU float32 tensor，shape [2, 2, 3]，Hqk=2、Hv=4；没有设备 kernel 或 mock。
    场景：Hv 是 Hqk 的整数倍，生产函数应按 group_size=2 repeat-interleave。
    输出：shape [2, 4, 3]，第 0/1 个 value head 来自源 head 0，第 2/3 个来自源 head 1。
    预期：输出与 torch.repeat_interleave(x, 2, dim=-2) 完全一致；失败表示 GQA
        的 head topology 会错位，后续 GDN 状态更新可能把不同请求的 head 混合。
    """
    source = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)

    expanded = gdn._expand_qk_to_v_heads(source, num_v_heads=4)

    assert expanded.shape == (2, 4, 3)
    torch.testing.assert_close(expanded, source.repeat_interleave(2, dim=-2))


def test_gdn_rejects_non_divisible_grouped_heads():
    """
    测试功能：验证 GDN 在 Hqk 不能整除 Hv 时拒绝非法 grouped-head topology。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU float16 tensor，shape [1, 2, 4]，Hqk=2、Hv=3，K=7 场景不涉及设备图。
    场景：非整数 group_size，不能猜测复制或截断 head。
    输出：ValueError，错误信息包含 Hqk 和 Hv 的实际值。
    预期：异常类型和关键字段都稳定；失败表示错误 topology 可能静默产生错误
        attention 数值，而不是在进入 kernel 前被拒绝。
    """
    source = torch.ones(1, 2, 4, dtype=torch.float16)

    with pytest.raises(ValueError, match=r"Hqk=2, Hv=3"):
        gdn._expand_qk_to_v_heads(source, num_v_heads=3)


def test_gdn_normalize_rejects_tnd_without_variable_length_metadata():
    """
    测试功能：验证 TND 输入必须同时提供 cu_seqlens，避免把变长 token 当成 batch。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU float16 q/k/v shape [9, 2, 16]/[9, 4, 12]，g/beta shape [9, 4]，无 cu_seqlens。
    场景：TND layout 的 9 个 token 属于多个请求，但调用者遗漏了请求边界。
    输出：ValueError，明确说明 TND 需要 cu_seqlens。
    预期：函数不返回部分归一化张量、不修改输入；失败表示 variable-length
        请求会用错误 state 行计算，造成接受长度和缓存状态错误。
    """
    q = torch.zeros(9, 2, 16, dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.zeros(9, 4, 12, dtype=torch.float16)
    g = torch.zeros(9, 4, dtype=torch.float32)
    beta = torch.zeros_like(g)

    with pytest.raises(ValueError, match="TND inputs require"):
        gdn._normalize_chunk_inputs(q, k, v, g, beta, cu_seqlens=None)


def test_gdn_pads_bthd_to_chunk_without_changing_logical_tokens():
    """
    测试功能：验证 BTHD prefill 在 65 token 时只补到下一个 64-token chunk，且补零不污染真实 token。
    对应修改：655f8a097、c6ff47548；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU float16 q/k/v shape [1, 65, 1, 2]，g/beta shape [1, 65, 1]，chunk_size=64。
    场景：刚跨过 64-token boundary，生产路径需要为 AscendC kernel 形成 128-token physical buffer。
    输出：五个 padded tensor 的 token 维为 128；seq_ranges 仍报告 logical [0, 65)；新增区域全零。
    预期：前 65 个 token 与输入逐元素相同，65:128 为零；失败表示 padding 会参与
        attention 或 state 更新，直接影响 FDO/GDN 的数值边界。
    """
    q = torch.arange(65 * 2, dtype=torch.float16).reshape(1, 65, 1, 2)
    k = q + 1
    v = q + 2
    g = torch.ones(1, 65, 1, dtype=torch.float32)
    beta = torch.full_like(g, 0.5)

    padded = gdn._pad_bthd_to_chunk(q, k, v, g, beta, CHUNK_SIZE)
    q_pad, k_pad, v_pad, g_pad, beta_pad, ranges, cu_kernel = padded

    assert q_pad.shape[1] == 2 * CHUNK_SIZE
    assert ranges == [(0, 0, 65)]
    assert cu_kernel is None
    torch.testing.assert_close(q_pad[:, :65], q)
    torch.testing.assert_close(k_pad[:, :65], k)
    torch.testing.assert_close(v_pad[:, :65], v)
    torch.testing.assert_close(g_pad[:, :65], g)
    torch.testing.assert_close(beta_pad[:, :65], beta)
    assert torch.count_nonzero(q_pad[:, 65:]) == 0
    assert torch.count_nonzero(k_pad[:, 65:]) == 0
    assert torch.count_nonzero(v_pad[:, 65:]) == 0
    assert torch.count_nonzero(g_pad[:, 65:]) == 0
    assert torch.count_nonzero(beta_pad[:, 65:]) == 0


def test_gdn_varlen_chunk_indices_skip_empty_sequence():
    """
    测试功能：验证变长 GDN 的 chunk index 使用 compact sequence id，并跳过空请求。
    对应修改：655f8a097、c6ff47548；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU metadata cu_seqlens=(0, 0, 64, 129)，chunk_size=64；第一个请求长度为 0。
    场景：包含空请求、恰好一 chunk 和 65-token 两 chunk 的混合 prefill。
    输出：chunk index [(0,0),(1,0),(1,1)] 的扁平列表 [0,0,1,0,1,1]。
    预期：空序列没有 kernel chunk，后续 compact id 连续；失败表示 state 索引会偏移，
        将一个请求的 recurrent state 写到另一个请求。
    """
    indices = gdn._prepare_chunk_indices_list((0, 0, 64, 129), CHUNK_SIZE)

    assert indices == [0, 0, 1, 0, 1, 1]


def test_gdn_wy_doubling_matches_triangular_reference():
    """
    测试功能：验证优化的 WY doubling 与独立 triangular forward-substitution reference 一致。
    对应修改：655f8a097、c6ff47548；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU float32 lower-triangular A shape [1,1,64,64]，rhs shape [1,1,64,3]，A 对角为零。
    场景：真实 64-token GDN chunk；reference 用独立逐行 forward substitution，不复用 production helper。
    输出：生产 helper 与独立逐行求解均得到 [1,1,64,3] 解矩阵。
    预期：torch.testing.assert_close；失败表示 WY 优化会改变 recurrent state 或 FwdO 输入，
        即使 shape 测试仍然通过也可能造成输出漂移。
    """
    n = 64
    row = torch.arange(n, dtype=torch.float32)
    a = (0.002 * (row[:, None] - row[None, :])).tril(diagonal=-1).reshape(1, 1, n, n)
    rhs = torch.arange(n * 3, dtype=torch.float32).reshape(1, 1, n, 3) / 100

    optimized = gdn._wy_doubling_apply(a, rhs)
    reference = torch.zeros_like(rhs)
    for i in range(n):
        reference[..., i, :] = rhs[..., i, :] + (a[..., i, :i].unsqueeze(-1) * reference[..., :i, :]).sum(dim=-2)

    torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-6)


def test_gdn_reference_rejects_head_first_and_mismatched_qk():
    """
    测试功能：验证 310P reference 明确拒绝不支持的 head_first 与 q/k shape mismatch。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py。
    输入：CPU float16 q shape [1,2,1,4]、k shape [1,2,1,3]、v/g/beta 为对应小张量。
    场景：先传 head_first=True，再传 head_first=False 但 q/k 最后一维不同。
    输出：分别为 DeprecationWarning 与 ValueError，不能执行到部分状态写入。
    预期：错误契约具体且可复现；失败表示调用者可能把错误布局送入真实 AscendC kernel。
    """
    q = torch.zeros(1, 2, 1, 4, dtype=torch.float16)
    k = torch.zeros(1, 2, 1, 3, dtype=torch.float16)
    v = torch.zeros(1, 2, 1, 4, dtype=torch.float16)
    g = torch.zeros(1, 2, 1, dtype=torch.float32)
    beta = torch.zeros_like(g)

    with pytest.raises(DeprecationWarning, match="head_first"):
        gdn.chunk_gated_delta_rule_pytorch(q, k, v, g, beta, head_first=True)
    with pytest.raises(ValueError, match="q and k shapes must match"):
        gdn.chunk_gated_delta_rule_pytorch(q, k, v, g, beta)

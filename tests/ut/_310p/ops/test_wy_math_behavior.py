# SPDX-License-Identifier: Apache-2.0
"""T16: production WY preparation checked against independent forward substitution."""

import pytest
import torch

from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as production


@pytest.mark.parametrize("tokens,decay", [(64, 0.01), (128, 0.01), (64, 1.0)])
def test_complete_wy_preparation_matches_independent_equations(tokens, decay):
    """T16-01：完整torch/doubling WY前处理与独立逐行前代方程一致。

    输入：B1、Hk2/Hv4、Dk/Dv64，T64/128，FP16 Q/K/V/beta，FP32负衰减；seed固定。
    输出：Q/K保持原值和kernel布局；W/U等于解(I-A)X=R，G等于分块累积衰减。
    场景：grouped heads扩展、多个64-token块、强/弱衰减，检查所有输出与dtype/连续性。
    依据：独立按head、chunk、row构造A/R并前代，未调用生产WY作为预期。
    替身：无。这里只验证真实CPU数学，AscendC数值与路由另由NPU测试证明。
    """
    rng = torch.Generator().manual_seed(20260922)
    q = (torch.randn(1, tokens, 2, 64, generator=rng) * 0.01).half()
    k = (torch.randn(q.shape, generator=rng) * 0.01).half()
    v = (torch.randn(1, tokens, 4, 64, generator=rng) * 0.1).half()
    g = -torch.rand(1, tokens, 4, generator=rng) * decay
    beta = (0.2 + 0.4 * torch.rand(1, tokens, 4, generator=rng)).half()
    expected_u = torch.empty(1, 4, tokens, 64, dtype=torch.float64)
    expected_w = torch.empty_like(expected_u)
    expected_g = torch.empty(1, 4, tokens, dtype=torch.float64)
    for head in range(4):
        for start in range(0, tokens, 64):
            keys = k[0, start : start + 64, head // 2].double()
            values = v[0, start : start + 64, head].double()
            b = beta[0, start : start + 64, head].double()
            cumulative = g[0, start : start + 64, head].double().cumsum(0)
            rows = []
            for row in range(64):
                rhs = torch.cat((b[row] * values[row], b[row] * cumulative[row].exp() * keys[row]))
                for col in range(row):
                    coefficient = -b[row] * torch.dot(keys[row], keys[col]) * (cumulative[row] - cumulative[col]).exp()
                    rhs = rhs + coefficient * rows[col]
                rows.append(rhs)
            solution = torch.stack(rows)
            expected_u[0, head, start : start + 64] = solution[:, :64]
            expected_w[0, head, start : start + 64] = solution[:, 64:]
            expected_g[0, head, start : start + 64] = cumulative
    for prepare in (
        production._compute_kernel_inputs_from_torch_wy,
        production._compute_kernel_inputs_from_doubling_wy,
    ):
        actual_q, actual_k, w, u, cumulative = prepare(q, k, v, g, beta, 64)
        torch.testing.assert_close(actual_q, q.transpose(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(actual_k, k.transpose(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(u, expected_u.half(), rtol=0.002, atol=2e-5)
        torch.testing.assert_close(w, expected_w.half(), rtol=0.002, atol=2e-6)
        torch.testing.assert_close(cumulative, expected_g.float(), rtol=2e-6, atol=2e-6)
        assert all(x.is_contiguous() for x in (actual_q, actual_k, w, u, cumulative))


def test_blocked_and_doubling_solve_nontrivial_triangular_system():
    """T16-02：两种真实WY解法处理带长依赖链的64阶严格下三角系统。

    输入：A的第一下对角线为.25、第二下对角线为-.125；独立先选X，再构造R=(I-A)X。
    输出：blocked和doubling均恢复全部X，A/R输入保持；不是用两个实现互为唯一oracle。
    场景：能检出doubling少迭代、错误符号、遗漏长依赖或原地污染。
    替身：无；FP64给出较严格数值容差，真实torch.linalg和matmul执行。
    """
    a = torch.diag(torch.full((63,), 0.25, dtype=torch.float64), -1)
    a += torch.diag(torch.full((62,), -0.125, dtype=torch.float64), -2)
    x = torch.arange(64 * 3, dtype=torch.float64).reshape(64, 3) / 100
    rhs = x - a @ x
    before_a, before_rhs = a.clone(), rhs.clone()
    for solve in (production._wy_blocked_fs_apply, production._wy_doubling_apply):
        torch.testing.assert_close(solve(a, rhs), x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(a, before_a, rtol=0, atol=0)
        torch.testing.assert_close(rhs, before_rhs, rtol=0, atol=0)

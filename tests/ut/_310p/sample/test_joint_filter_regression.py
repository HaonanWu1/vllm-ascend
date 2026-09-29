# SPDX-License-Identifier: Apache-2.0
"""Independent retained-token oracles for normal and TP candidate filtering."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.sample import sampler
from vllm_ascend.worker import model_runner_v1 as runner_module


@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize(
    "values,k,p,kept",
    [
        ([0.1, 0.2, 0.3, 0.4], 2, 0.5, [3]),
        ([1.0, 1.0, 1.0, 1.0], 4, 0.5, [2, 3]),
        ([0.1, 0.2, 0.3, 0.4], 3, 1.0, [1, 2, 3]),
        ([0.1, 0.2, 0.3, 0.4], 1, 0.001, [3]),
        ([0.1, 0.2, 0.3, 0.4], 4, 1.0, [0, 1, 2, 3]),
    ],
)
def test_joint_filter_retained_candidates(values, k, p, kept, reduce):
    """输入：表中4词概率、k/p；普通路径或TP2候选收集，两个分片每片2词。

    输出：保留token集合严格等于预先手算列表，有限logits原值不变，无全-inf行。
    场景：联合过滤需在top-k后重归一化；均匀top-p只保留排序后的尾部两个；
      TP2的k=3必须跨越本地词表大小2，k=总词表4时不能被截成2。
    替身：配置与TP通信边界；all_gather输入校验且返回独立构造的真实分片候选；
      排序、mask、softmax及输出均为真实torch，通信正确性另由设备/服务覆盖。
    """
    logits = torch.tensor([values], dtype=torch.float64).log()
    top_k, top_p = torch.tensor([k], dtype=torch.int32), torch.tensor([p], dtype=torch.float64)
    gathered_values = torch.cat([part.topk(2).values for part in logits.split(2, dim=1)], dim=1)
    gathered_indices = torch.cat(
        [part.topk(2).indices + rank * 2 for rank, part in enumerate(logits.split(2, dim=1))], dim=1
    )
    # Equal logits retain the last two in the candidate sort order. The TP
    # candidate order from local topk is a real input, not a global-ID tie rule.
    if reduce and len(set(values)) == 1:
        kept = sorted(gathered_indices[0, -2:].tolist())
    calls = []

    def gather(value, dim):
        assert dim == -1
        expected = logits[:, :2].topk(2)
        torch.testing.assert_close(value, expected.values if not calls else expected.indices)
        calls.append(value)
        return (gathered_values if len(calls) == 1 else gathered_indices).clone()

    group = SimpleNamespace(rank_in_group=0, all_gather=gather)
    with (
        patch.object(sampler, "get_ascend_config", return_value=SimpleNamespace(enable_reduce_sample=reduce)),
        patch.object(sampler, "get_tp_group", return_value=group),
    ):
        result = sampler._apply_top_k_top_p_pytorch(
            logits[:, :2].clone() if reduce else logits.clone(), top_k, top_p, 2
        )
    if reduce:
        values_out, indices = result
        result = torch.full_like(logits, float("-inf")).scatter_(1, indices, values_out)
        assert len(calls) == 2
    assert torch.isfinite(result[0]).nonzero().flatten().tolist() == kept
    torch.testing.assert_close(result[0, kept], logits[0, kept], rtol=0, atol=0)


def test_p_only_reduce_ignores_previous_topk_hint():
    """输入：TP1本地4词概率.1/.2/.3/.4，本轮k=None,p=.85，旧轮缓存top_k=1。

    输出：保留概率累计满足p=.85的候选[.2,.3,.4]，不得截成1个候选。
      p=.85远离累计概率边界，避免将浮点舍入误差当成候选截断错误。
    场景：top-p-only请求复用sampler对象，前轮top-k hint不属于本轮参数。
    替身：TP1 all_gather为恒等通信及reduce配置；所有候选选择/概率过滤实际执行。
    """
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64).log()
    group = SimpleNamespace(rank_in_group=0, all_gather=lambda tensor, dim: tensor)
    with (
        patch.object(sampler, "get_ascend_config", return_value=SimpleNamespace(enable_reduce_sample=True)),
        patch.object(sampler, "get_tp_group", return_value=group),
    ):
        values, indices = sampler._apply_top_k_top_p_pytorch(logits, None, torch.tensor([0.85], dtype=torch.float64), 1)
    assert sorted(indices[torch.isfinite(values)].tolist()) == [1, 2, 3]


@pytest.mark.parametrize("speculative", [False, True])
def test_reduce_candidate_hint_includes_active_global_k_only(speculative):
    """输入：两活跃请求top_k=[8,3]，TP2本地词表4，未使用CPU buffer尾部值999。

    输出：普通/投机_sample均向sampler传最大活跃全局k=8，不能按本地V筛成3或读取999。
    场景：混合top-k与不限制k的行，候选收集必须保留后者的完整分布。
    替身：模型采样器只记录prepare_sampling入参、返回哨兵；此项仅验证真实runner配置传递，
      不计作概率数值测试。候选过滤的数值正确性由上面的真实tensor用例独立验证。
    """
    hints = []

    class Consumer:
        def prepare_sampling(self, value):
            hints.append(int(value))

        def __call__(self, *args, **kwargs):
            return "sampled"

    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        top_k_cpu=np.array([8, 3, 999]),
        sampling_metadata=SimpleNamespace(top_k=torch.tensor([8, 3])),
        update_async_output_token_ids=lambda: None,
    )
    runner.sampler = runner.rejection_sampler = Consumer()
    with (
        patch.object(runner_module, "get_ascend_config", return_value=SimpleNamespace(enable_reduce_sample=True)),
        patch.object(runner_module, "lmhead_tp_enable", return_value=False),
        patch.object(runner_module, "get_pp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        assert runner._sample(torch.zeros((2, 4)), SimpleNamespace() if speculative else None) == "sampled"
    assert hints == [8]

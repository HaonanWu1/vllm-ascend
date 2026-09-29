# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch as upstream_filter

from vllm_ascend.sample import rejection_sampler, sampler


@pytest.mark.parametrize(
    "k,p,kept",
    [
        (None, None, [0, 1, 2, 3]),
        (2, None, [2, 3]),
        (None, 0.5, [2, 3]),
        (2, 0.5, [3]),
        (1, 0.9, [3]),
        (4, 1.0, [0, 1, 2, 3]),
        (None, 0.1, [3]),
    ],
    ids=["unfiltered", "top-k", "top-p", "top-k-then-top-p", "k1", "no-op", "small-p"],
)
def test_topk_topp_filter_matches_hand_calculation_and_upstream(k, p, kept):
    """T22-01a：top-k/top-p组合必须按固定vLLM版本语义生效。

    输入：概率[.1,.2,.3,.4]对应logits，k/p与保留ID在参数表预先固定。
    输出：保留集合和重归一化概率与手算及上游独立生产reference一致。
      k=2,p=.5时先保留[.3,.4]，归一化为[3/7,4/7]，top-p仅保留ID3。
    场景：不能对原始分布分别取两个mask交集，忽略top-k后归一化。
    替身：只有enable_reduce_sample=False配置输入，真实排序/softmax与过滤执行。
    回归：若组合参数与vLLM语义不一致，保留FAIL；不按当前错误实现修改expected。
    """
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64).log()
    top_k = None if k is None else torch.tensor([k], dtype=torch.int32)
    top_p = None if p is None else torch.tensor([p], dtype=torch.float64)
    reference = upstream_filter(logits.clone(), top_k, top_p)
    assert torch.isfinite(reference[0]).nonzero().flatten().tolist() == kept
    with patch.object(sampler, "get_ascend_config", return_value=SimpleNamespace(enable_reduce_sample=False)):
        result = sampler._apply_top_k_top_p_pytorch(logits.clone(), top_k, top_p)
    assert torch.isfinite(result[0]).nonzero().flatten().tolist() == kept
    torch.testing.assert_close(result.softmax(-1), reference.softmax(-1), rtol=1e-12, atol=1e-12)


def test_temperature_expands_per_request_without_scaling_greedy_zero(monkeypatch):
    """T22-01b：ragged请求temperature扩展和混批temperature0保护。

    输入：3个logits行[0,1,2,3]，请求draft数[2,1]，temperature[0,.5]，不截断。
    输出：前两行保持原值，第三行[0,2,4,6]；不得除零或把请求0温度串给请求1。
    依据：按请求token区间手算logits/T；调用真实apply_sampling_constraints及expand。
    替身：仅CPU pin分配和关闭reduce配置，未替换扩展/采样逻辑。
    """
    original_tensor = torch.tensor

    def tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor)
    config = SimpleNamespace(enable_reduce_sample=False)
    metadata = SimpleNamespace(all_greedy=False, temperature=torch.tensor([0.0, 0.5]), top_k=None, top_p=None)
    logits = torch.arange(4, dtype=torch.float32).repeat(3, 1)
    with (
        patch.object(sampler, "get_ascend_config", return_value=config),
        patch.object(rejection_sampler, "get_ascend_config", return_value=config),
    ):
        result = rejection_sampler.apply_sampling_constraints(logits, torch.tensor([2, 3]), metadata, None)
    torch.testing.assert_close(result, torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]]))

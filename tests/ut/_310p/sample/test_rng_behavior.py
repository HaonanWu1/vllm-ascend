# SPDX-License-Identifier: Apache-2.0
"""T22: use real CPU tensors and generators, never fake random arithmetic."""

import pytest
import torch

from vllm_ascend._310p.sample import sampler as production


@pytest.fixture(autouse=True)
def isolate_rng_cache():
    previous = production._CPU_GENERATOR_CACHE_310P.copy()
    production._CPU_GENERATOR_CACHE_310P.clear()
    with torch.random.fork_rng(devices=[]):
        yield
    production._CPU_GENERATOR_CACHE_310P.clear()
    production._CPU_GENERATOR_CACHE_310P.update(previous)


def test_seeded_rows_advance_and_new_request_resets_only_own_stream():
    """T22-06a：连续抽样推进RNG，新请求不继承被取消请求状态。

    输入：2x16 FP32 CPU tensor、seed11/22；第二轮将第1行换为seed33的新Generator。
    输出：第0行两轮严格等于独立seed11序列前两段；第1行分别为seed22/33首段。
    依据：独立torch.Generator的真实exponential_调用，比较全部数值及缓存RNG状态。
    替身：无；fixture只隔离生产全局cache与默认RNG，CPU UT不证明NPU传输。
    回归：每轮重置seed、槽位复用串用旧状态或未更新结果必失败。
    """
    sources = {0: torch.Generator().manual_seed(11), 1: torch.Generator().manual_seed(22)}
    oracle0 = torch.Generator().manual_seed(11)
    for second_seed in (22, 33):
        if second_seed == 33:
            sources[1] = torch.Generator().manual_seed(33)
        out = torch.full((2, 16), float("nan"))
        production._fill_cpu_exponential_310p(out, sources)
        expected = torch.stack(
            [
                torch.empty(16).exponential_(generator=oracle0),
                torch.empty(16).exponential_(generator=torch.Generator().manual_seed(second_seed)),
            ]
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert torch.equal(production._CPU_GENERATOR_CACHE_310P[sources[0]].get_state(), oracle0.get_state())


@pytest.mark.parametrize(
    "seeded,mask",
    [({}, None), ({0: 31}, None), ({0: 31, 1: 32}, [True, False]), ({0: 31, 1: 32}, [False, True])],
    ids=["unseeded", "partially-seeded", "first-has-draft", "second-has-draft"],
)
def test_unseeded_and_masked_rows_are_initialized_from_correct_stream(seeded, mask):
    """T22-06b：seed部分覆盖和has_draft_mask不留下未初始化随机数。

    输入：2x12 NaN张量，默认seed99，部分/全部请求seed31/32及参数表mask。
    输出：seed覆盖且有draft的行等于该Generator首段，其余行等于默认RNG首批。
    依据：独立默认RNG状态和逐请求Generator手工组织的期望；逐元素严格比较。
    替身：无；检测漏prefill、mask反转或随机行之间串扰。
    """
    torch.manual_seed(99)
    expected = torch.empty((2, 12)).exponential_()
    for row, seed in seeded.items():
        if mask is None or mask[row]:
            expected[row] = torch.empty(12).exponential_(generator=torch.Generator().manual_seed(seed))
    torch.manual_seed(99)
    output = torch.full((2, 12), float("nan"))
    production._fill_cpu_exponential_310p(
        output,
        {row: torch.Generator().manual_seed(seed) for row, seed in seeded.items()},
        None if mask is None else torch.tensor(mask),
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.isfinite(output).all() and (output > 0).all()


def test_request_reordering_keeps_rng_continuation():
    """T22-06c：请求换batch槽位后应继续其随机序列。

    输入：请求A使用真实seed41，先位于row0，下一轮移动到row1；其Generator对象不变。
    输出：A第二轮必须等于独立seed41的第二段，不得重复首段。
    依据：请求随机状态应随请求延续，batch槽位不是新的请求。
    替身：无；回归旧版按row缓存造成的序列重置，直接比较两段真实随机数。
    """
    source = torch.Generator().manual_seed(41)
    oracle = torch.Generator().manual_seed(41)
    first = torch.empty((2, 16))
    production._fill_cpu_exponential_310p(first, {0: source})
    torch.testing.assert_close(first[0], torch.empty(16).exponential_(generator=oracle), rtol=0, atol=0)
    second = torch.empty((2, 16))
    production._fill_cpu_exponential_310p(second, {1: source})
    torch.testing.assert_close(second[1], torch.empty(16).exponential_(generator=oracle), rtol=0, atol=0)

# SPDX-License-Identifier: Apache-2.0
"""Request-owned RNG: numerical continuity, retirement and discarded samples."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend._310p import model_runner_310p as runner_module
from vllm_ascend._310p.sample import sampler


@pytest.fixture(autouse=True)
def isolated_cache():
    previous = sampler._CPU_GENERATOR_CACHE_310P.copy()
    sampler._CPU_GENERATOR_CACHE_310P.clear()
    yield
    sampler._CPU_GENERATOR_CACHE_310P.clear()
    sampler._CPU_GENERATOR_CACHE_310P.update(previous)


def test_reorder_preempt_and_same_seed_new_request():
    """输入：A/B/C种子11/22/33；批次ABC→CA→BC→新A与旧A（同种子不同对象）。

    输出：每个请求的所有实际抽样逐值等于其独立CPU Generator连续序列；新A从首段开始。
    场景：压缩、换行、插队、暂时缺席/恢复；batch行号和相同seed都不能充当请求身份。
    依据：每请求独立真实exponential_ oracle，旧/新A同时在批次中，无随机算子替身。
    """
    seeds = {"A": 11, "B": 22, "C": 33, "newA": 11}
    sources = {name: torch.Generator().manual_seed(seed) for name, seed in seeds.items()}
    oracles = {name: torch.Generator().manual_seed(seed) for name, seed in seeds.items()}
    for names in [("A", "B", "C"), ("C", "A"), ("B", "C"), ("newA", "A")]:
        result = torch.empty((len(names), 37))
        sampler._fill_cpu_exponential_310p(result, {row: sources[name] for row, name in enumerate(names)})
        for row, name in enumerate(names):
            expected = torch.empty(37).exponential_(generator=oracles[name])
            torch.testing.assert_close(result[row], expected, rtol=0, atol=0)


def test_unseeded_discard_keeps_default_rng_without_creating_request_cache():
    """输入：丢弃row0没有请求Generator，row1有seed777；默认CPU seed44，真实2x11随机填充。

    输出：无seed行使用默认随机流，seed行使用777首段；cache仅有row1的Generator。
    场景：混合有seed/无seed请求时，回退逻辑不得为无seed行创建/借用其他请求状态。
    替身：父_sample边界执行真实生产填充；所有随机数与独立默认/请求Generator逐值比较。
    """
    source = torch.Generator().manual_seed(777)
    runner = runner_module.NPUModelRunner310.__new__(runner_module.NPUModelRunner310)
    runner.input_batch = SimpleNamespace(generators={1: source})
    runner.num_discarded_requests = 1
    runner.discard_request_indices = SimpleNamespace(np=np.array([0], dtype=np.int32))
    output = torch.empty((2, 11))

    def sample(self, logits, metadata):
        sampler._fill_cpu_exponential_310p(output, self.input_batch.generators)
        return output

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        expected = torch.empty((2, 11)).exponential_()
        expected[1] = torch.empty(11).exponential_(generator=torch.Generator().manual_seed(777))
        torch.manual_seed(44)
        with patch.object(runner_module.NPUModelRunner, "_sample", sample):
            assert runner._sample(None, None) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert set(sampler._CPU_GENERATOR_CACHE_310P) == {source}


def test_finished_and_cancelled_requests_release_only_their_streams():
    """输入：真实种子11/22/33生成一段；finished集合包含完成A、取消C和未知ID。

    输出：生产_update_states释放A/C引用，仅保留B；B后续抽样等于自身第二段。
    场景：完成/取消清理、未知ID幂等、存活请求不受影响；防止强引用cache随请求数增长。
    替身：父runner的调度状态删除和NPU同步为外部边界；被测310P方法/cache/RNG均真实。
      父边界会删除请求，以验证必须先读取请求所有权再调用父实现。
    """
    sources = [torch.Generator().manual_seed(seed) for seed in [11, 22, 33]]
    runner = runner_module.NPUModelRunner310.__new__(runner_module.NPUModelRunner310)
    runner.requests = {name: SimpleNamespace(generator=gen) for name, gen in zip("ABC", sources)}
    oracle = torch.Generator().manual_seed(22)
    torch.empty(17).exponential_(generator=oracle)
    sampler._fill_cpu_exponential_310p(torch.empty((3, 17)), dict(enumerate(sources)))
    schedule = SimpleNamespace(finished_req_ids={"A", "C", "unknown"})

    def remove_requests(self, output):
        for req_id in output.finished_req_ids:
            self.requests.pop(req_id, None)
        return "parent-return"

    with (
        patch.object(runner_module.NPUModelRunner, "_update_states", remove_requests),
        patch.object(torch.npu, "current_stream"),
    ):
        assert runner._update_states(schedule) == "parent-return"
        assert runner._update_states(schedule) == "parent-return"
    assert set(sampler._CPU_GENERATOR_CACHE_310P) == {sources[1]}
    result = torch.empty((1, 17))
    sampler._fill_cpu_exponential_310p(result, {0: sources[1]})
    torch.testing.assert_close(result[0], torch.empty(17).exponential_(generator=oracle), rtol=0, atol=0)


@pytest.mark.parametrize("discarded,raises", [([], False), ([0], False), ([0, 1], False), ([1], True)])
def test_discarded_chunked_prefill_restores_cpu_rng_only_for_discarded_rows(discarded, raises):
    """输入：两请求seed41/42，已消耗一段；参数表控制本轮丢弃行及父采样异常。

    输出：丢弃行的下一段与未消耗本轮时一致；有效行继续第三段，返回值/异常原样传递。
    场景：chunked prefill不能消耗用户尚未输出的随机token；每行状态独立回退。
    依据：真实CPU RNG状态/逐元素输出。父_sample边界替身仍调用真实生产随机填充，
      只省略模型logits/设备采样以隔离310P包装层职责，不替换状态或随机数计算。
    """
    sources = {i: torch.Generator().manual_seed(41 + i) for i in range(2)}
    oracles = [torch.Generator().manual_seed(41 + i) for i in range(2)]
    sampler._fill_cpu_exponential_310p(torch.empty((2, 29)), sources)
    for oracle in oracles:
        torch.empty(29).exponential_(generator=oracle)
    runner = runner_module.NPUModelRunner310.__new__(runner_module.NPUModelRunner310)
    runner.input_batch = SimpleNamespace(generators=sources)
    runner.num_discarded_requests = len(discarded)
    runner.discard_request_indices = SimpleNamespace(np=np.array(discarded, dtype=np.int32))
    output = torch.empty((2, 29))

    def sample(self, logits, metadata):
        sampler._fill_cpu_exponential_310p(output, self.input_batch.generators)
        if raises:
            raise ValueError("sampling-failed")
        return output

    with patch.object(runner_module.NPUModelRunner, "_sample", sample):
        if raises:
            with pytest.raises(ValueError, match="sampling-failed"):
                runner._sample(None, None)
        else:
            assert runner._sample(None, None) is output
    next_output = torch.empty((2, 29))
    sampler._fill_cpu_exponential_310p(next_output, sources)
    for row, oracle in enumerate(oracles):
        if row not in discarded:
            torch.empty(29).exponential_(generator=oracle)
        torch.testing.assert_close(next_output[row], torch.empty(29).exponential_(generator=oracle), rtol=0, atol=0)

# SPDX-License-Identifier: Apache-2.0
"""Real 310P numerical regressions for rejection, filtering and request RNG."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend._310p import model_runner_310p as runner_module
from vllm_ascend._310p.sample import sampler as sampler310
from vllm_ascend.sample import rejection_sampler, sampler
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker import model_runner_v1 as common_runner


@pytest.fixture(autouse=True)
def device_setup():
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    before = sampler310._CPU_GENERATOR_CACHE_310P.copy()
    sampler310._CPU_GENERATOR_CACHE_310P.clear()
    yield
    sampler310._CPU_GENERATOR_CACHE_310P.clear()
    sampler310._CPU_GENERATOR_CACHE_310P.update(before)


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("k", [1, 7, 15])
@pytest.mark.parametrize("accept", [False, True])
def test_real_rejection_index_arithmetic(block, k, accept):
    """输入：真实NPU累计draft数[K,K,2K]，K1/7/15，普通/block verify，第三行greedy。

    输出：接受行K个0+bonus1或拒绝行recovered2；零draft行bonus1；greedy行哨兵7保持。
    场景：服务崩溃的索引Add兼容性修复同时覆盖普通/block算法、混批和空draft。
    依据：单点分布或p0=.125<U=.8的独立手算，真实NPU分配/算子/同步，无替身。
    """
    result = torch.full((3, k + 1), -1, dtype=torch.int32, device="npu")
    result[2].fill_(7)
    probs = torch.tensor([[1.0, 0.0, 0.0] if accept else [0.125, 0.375, 0.5]], device="npu").repeat(2 * k, 1)
    function = (
        rejection_sampler.rejection_random_sample_block_verify_pytorch
        if block
        else rejection_sampler.rejection_random_sample_pytorch
    )
    function(
        result,
        torch.tensor([k, k, 2 * k], dtype=torch.int32, device="npu"),
        torch.zeros(2 * k, dtype=torch.int32, device="npu"),
        None,
        probs,
        torch.ones(3, dtype=torch.int32, device="npu"),
        torch.full((2 * k,), 2, dtype=torch.int32, device="npu"),
        torch.full((2 * k,), 0.8, device="npu"),
        torch.tensor([False, False, True], device="npu"),
        k,
        3,
        IS_NGRAM=True,
    )
    torch.npu.synchronize()
    expected = torch.full((3, k + 1), -1, dtype=torch.int32)
    expected[0, 0] = 2
    if accept:
        expected[0, :k], expected[0, k] = 0, 1
    expected[1, 0], expected[2] = 1, 7
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [4, 248320])
def test_real_joint_topk_topp_and_per_row_parameters(monkeypatch, vocab):
    """输入：3行NPU logits，前4词概率.1/.2/.3/.4、其余-inf；k=[2,3,V],p=[.5,1,1]。

    输出：三行保留集合分别[3]、[1,2,3]、[0,1,2,3]，保留logits原值不变。
    场景：真实模型词表大小248320与小向量，逐行参数及top-k后重新归一化。
    替身：只设置关闭reduce的配置，真实NPU sort/softmax/cumsum/scatter执行并同步。
    """
    monkeypatch.setattr(sampler, "get_ascend_config", lambda: SimpleNamespace(enable_reduce_sample=False))
    logits_cpu = torch.full((3, vocab), float("-inf"))
    logits_cpu[:, :4] = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
    result = sampler._apply_top_k_top_p_pytorch(
        logits_cpu.npu(),
        torch.tensor([2, 3, vocab], dtype=torch.int32, device="npu"),
        torch.tensor([0.5, 1.0, 1.0], device="npu"),
    ).cpu()
    for row, kept in enumerate([[3], [1, 2, 3], [0, 1, 2, 3]]):
        assert torch.isfinite(result[row]).nonzero().flatten().tolist() == kept
        torch.testing.assert_close(result[row, kept], logits_cpu[row, kept], rtol=0, atol=0)


def test_real_npu_request_generators_survive_reorder_and_retirement():
    """输入：真实NPU Generator种子101/202；batch AB→B→BA；最后释放A。

    输出：每次NPU消费到的q逐值等于各自独立CPU Generator连续片段；最终cache仅含B。
    场景：NPU状态无法导入CPU时的seed初始化、缩批、换行、恢复、生命周期清理。
    依据：独立CPU随机数oracle；生产CPU填充/H2D/stream同步全部执行，不mock任何设备接口。
    """
    sources = [torch.Generator(device="npu").manual_seed(seed) for seed in [101, 202]]
    oracles = [torch.Generator().manual_seed(seed) for seed in [101, 202]]
    for order in [[0, 1], [1], [1, 0]]:
        result = torch.empty((len(order), 31), device="npu")
        sampler310.fill_exponential_310p(result, {row: sources[index] for row, index in enumerate(order)})
        for row, index in enumerate(order):
            torch.testing.assert_close(
                result[row].cpu(), torch.empty(31).exponential_(generator=oracles[index]), rtol=0, atol=0
            )
    sampler310.release_cpu_generator_310p(sources[0])
    assert set(sampler310._CPU_GENERATOR_CACHE_310P) == {sources[1]}


@pytest.mark.parametrize("device_draws", [0, 7])
def test_discarded_prefill_restores_cpu_and_device_streams_through_bookkeeping(device_draws):
    """输入：seed303真实NPU Generator，丢弃唯一prefill行；采样消耗CPU q与0/7个NPU uniform。

    输出：经过生产_sample与完整_bookkeeping_sync后，两个Generator状态逐字节不变，输出为空。
    场景：普通CPU exponential采样并未推进NPU offset，不能固定减4造成下溢；
      speculative uniform消耗长度也不能假定恒为4。回退必须恢复抽样前完整状态。
    替身：父_sample省略模型，仅执行真实CPU/NPU随机数；_to_list执行真实D2H；
      不需要prompt logprobs故该外部投影返回空。状态快照/回退/输出丢弃逻辑均真实。
    """
    source = torch.Generator(device="npu").manual_seed(303)
    sampler310.fill_exponential_310p(torch.empty((1, 17), device="npu"), {0: source})
    cpu_generator = sampler310._CPU_GENERATOR_CACHE_310P[source]
    cpu_before, device_before = cpu_generator.get_state(), source.get_state()
    runner = runner_module.NPUModelRunner310.__new__(runner_module.NPUModelRunner310)
    runner.input_batch = SimpleNamespace(generators={0: source}, req_ids=["A"], req_id_to_index={"A": 0})
    runner.num_discarded_requests = 1
    runner.discard_request_indices = SimpleNamespace(np=np.array([0], dtype=np.int32))
    runner.use_async_scheduling = runner.routed_experts_initialized = False
    runner._to_list = lambda tensor: tensor.cpu().tolist()
    runner._get_prompt_logprobs_dict = lambda *_: {}

    def parent_sample(self, logits, metadata):
        sampler310.fill_exponential_310p(torch.empty((1, 17), device="npu"), self.input_batch.generators)
        if device_draws:
            torch.rand(device_draws, device="npu", generator=source)
        return SimpleNamespace(
            sampled_token_ids=torch.ones((1, 1), dtype=torch.int32, device="npu"), logprobs_tensors=None
        )

    with patch.object(runner_module.NPUModelRunner, "_sample", parent_sample):
        sampled = runner._sample(None, None)
    result = runner._bookkeeping_sync(
        SimpleNamespace(num_scheduled_tokens={"A": 1}), sampled, None, torch.zeros((1, 1), device="npu"), 1, None
    )
    assert result[1] == [[]]
    assert torch.equal(cpu_generator.get_state(), cpu_before)
    assert torch.equal(source.get_state(), device_before)


def test_other_device_bookkeeping_retains_existing_offset_rollback(monkeypatch):
    """输入：公共runner按A2配置分支，真实NPU Generator offset16，丢弃唯一prefill输出。

    输出：完整bookkeeping后offset12、输出空列表；310P专用改动不取消其他设备原有回退4。
    场景：共享调用方的设备分流兼容性，防止修复310P时误改其他设备行为。
    替身：设备型号配置、真实D2H包装及空prompt-logprobs边界；状态接口用真实310P Generator。
    限定：这是公共逻辑分支回归，不能冒充A2硬件算子、服务或随机分布验证。
    """
    source = torch.Generator(device="npu").manual_seed(404)
    source.set_offset(16)
    runner = common_runner.NPUModelRunner.__new__(common_runner.NPUModelRunner)
    runner.input_batch = SimpleNamespace(generators={0: source}, req_ids=["A"], req_id_to_index={"A": 0})
    runner.num_discarded_requests = 1
    runner.discard_request_indices = SimpleNamespace(np=np.array([0], dtype=np.int32))
    runner.use_async_scheduling = runner.routed_experts_initialized = False
    runner._to_list = lambda tensor: tensor.cpu().tolist()
    runner._get_prompt_logprobs_dict = lambda *_: {}
    monkeypatch.setattr(common_runner, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    sampled = SimpleNamespace(
        sampled_token_ids=torch.ones((1, 1), dtype=torch.int32, device="npu"), logprobs_tensors=None
    )
    result = runner._bookkeeping_sync(
        SimpleNamespace(num_scheduled_tokens={"A": 1}), sampled, None, torch.zeros((1, 1), device="npu"), 1, None
    )
    assert result[1] == [[]]
    assert source.get_offset() == 12

# SPDX-License-Identifier: Apache-2.0
"""T22 real 310P execution. No device mocks, no CPU pin-allocation wrappers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend import ascend_config
from vllm_ascend._310p.sample import sampler as sampling_310
from vllm_ascend._310p.sample.rejection_sampler import AscendRejectionSampler310
from vllm_ascend.sample import rejection_sampler as rejection


@pytest.fixture(scope="module")
def device():
    assert Path(torch_npu.__file__).is_file(), "real torch_npu is required"
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    return torch.device("npu:0")


def test_exponential_transfer_is_visible_to_real_npu_consumer(device):
    """T22-08a：真实CPU RNG→NPU传输后consumer读取本轮数据。

    输入：1x16 FP32 NPU张量，真实CPU Generator seed73，连续两次填充。
    输出：两轮设备结果逐元素等于独立CPU Generator对应两段，device仍为NPU。
    场景：验证生产fill_exponential_310p的复制/同步及cache推进；不模拟stream。
    依据：独立Generator序列；CPU填充后的NPU消费为真实算子，异常即FAIL。
    替身：无；退出时恢复模块cache，避免污染其他用例。
    """
    previous = sampling_310._CPU_GENERATOR_CACHE_310P.copy()
    sampling_310._CPU_GENERATOR_CACHE_310P.clear()
    source = torch.Generator().manual_seed(73)
    oracle = torch.Generator().manual_seed(73)
    try:
        out = torch.empty((1, 16), dtype=torch.float32, device=device)
        for _ in range(2):
            sampling_310.fill_exponential_310p(out, {0: source})
            expected = torch.empty((1, 16)).exponential_(generator=oracle)
            assert out.device == device
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    finally:
        sampling_310._CPU_GENERATOR_CACHE_310P.clear()
        sampling_310._CPU_GENERATOR_CACHE_310P.update(previous)


@pytest.mark.parametrize("batch", [1, 10], ids=["c1", "c10"])
def test_nongreedy_k7_rejection_executes_on_310p(device, batch):
    """T22-02/04/08b：真实NPU上执行K7非贪婪None-proposal拒绝分支。

    输入：C1/C10、每行7个draft0，target=[.125,.375,.5]，U=.8，recovered2，bonus1。
    输出：每行[2,-1,-1,-1,-1,-1,-1,-1]，不允许设备错误、错误接受或尾部污染。
    依据：None路径接受率.125<.8，首token必拒绝；torch_npu与全部生产操作真实执行。
    场景：CPU正确不保证310P小向量Add/索引指令可执行；该用例专门检测设备兼容性。
    替身：无；出现设备异常保持FAIL，不能切CPU或mock算子绕过。
    """
    k = 7
    output = torch.full((batch, k + 1), -1, dtype=torch.int32, device=device)
    cu = torch.arange(1, batch + 1, dtype=torch.int32, device=device) * k
    rejection.rejection_random_sample_pytorch(
        output,
        cu,
        torch.zeros(batch * k, dtype=torch.int32, device=device),
        None,
        torch.tensor([[0.125, 0.375, 0.5]], device=device).repeat(batch * k, 1),
        torch.ones(batch, dtype=torch.int32, device=device),
        torch.full((batch * k,), 2, dtype=torch.int32, device=device),
        torch.full((batch * k,), 0.8, device=device),
        torch.zeros(batch, dtype=torch.bool, device=device),
        k,
        3,
        IS_NGRAM=True,
    )
    torch.npu.synchronize()
    expected = torch.full((batch, k + 1), -1, dtype=torch.int32)
    expected[:, 0] = 2
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("with_logprobs", [False, True], ids=["without-logprobs", "with-logprobs"])
def test_full_310p_nongreedy_forward_and_logprobs(device, monkeypatch, with_logprobs):
    """T22-04/08c：真实310P forward贯通bonus、约束、拒绝/恢复及可选logprobs。

    输入：单请求K7，draft全0，16词表target只有token2概率1，temperature=.7；
      max_num_logprobs=None或3；全部Metadata及Sampler均为真实生产对象。
    输出：采样[2,-1..-1]；要求logprobs时有效token2的logprob=0，不能报设备错误。
    依据：单点分布使结果可精确验证；服务同款jit_compile=False，真实NPU流与RNG。
    替身：仅提供关闭reduce/block/entropy的配置值，无采样/设备替身。
    回归：特别覆盖服务首错涉及的logprobs索引Add；报错保留FAIL并单独进程复现。
    """
    config = SimpleNamespace(
        enable_reduce_sample=False,
        enable_async_exponential=False,
        ascend_compilation_config=SimpleNamespace(),
        eplb_config=SimpleNamespace(),
        rejection_sampler_config=SimpleNamespace(
            enable_block_verify=False, enable_entropy_verify=False, posterior_threshold=0.95, posterior_alpha=0.4
        ),
    )
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", config)
    scalar = torch.tensor([0.0], device=device)
    sampling = SamplingMetadata(
        temperature=torch.tensor([0.7], device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=3 if with_logprobs else None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=scalar,
        presence_penalties=scalar,
        repetition_penalties=torch.ones(1, device=device),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    metadata = SpecDecodeMetadata(
        draft_token_ids=torch.zeros(7, dtype=torch.int32, device=device),
        num_draft_tokens=[7],
        cu_num_draft_tokens=torch.tensor([7], dtype=torch.int32, device=device),
        cu_num_sampled_tokens=torch.tensor([8], dtype=torch.int32, device=device),
        target_logits_indices=torch.arange(7, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor([7], dtype=torch.int32, device=device),
        logits_indices=torch.arange(8, dtype=torch.int32, device=device),
    )
    logits = torch.full((8, 16), float("-inf"), device=device)
    logits[:, 2] = 0
    sampler = AscendRejectionSampler310(sampling_310.AscendSampler310())
    result = sampler(metadata, None, logits, sampling)
    torch.npu.synchronize()
    assert result.sampled_token_ids.cpu().tolist() == [[2, -1, -1, -1, -1, -1, -1, -1]]
    if with_logprobs:
        assert result.logprobs_tensors is not None
        assert result.logprobs_tensors.logprobs.cpu()[0, 0].item() == 0
    else:
        assert result.logprobs_tensors is None

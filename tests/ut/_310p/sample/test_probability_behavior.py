# SPDX-License-Identifier: Apache-2.0
"""T14/T22: independent arithmetic and distribution checks of production sampling.

CPU allocation wrappers only disable pin_memory (no CPU-only pin allocator).
They retain real torch arithmetic. These results do not certify NPU synchronization.
"""

import math

import pytest
import torch

from vllm_ascend.sample import rejection_sampler as production


@pytest.fixture(autouse=True)
def cpu_pin_allocator_boundary(monkeypatch):
    for name in ("tensor", "arange", "full", "ones", "empty", "zeros"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)


@pytest.mark.parametrize("k", range(1, 16), ids=lambda k: f"k{k}")
@pytest.mark.parametrize("none_proposal", [False, True], ids=["explicit-q", "none-q"])
def test_every_rejection_position_ragged_and_greedy_isolation(k, none_proposal):
    """T22-02/04/05：K1～15逐个首次拒绝位置、ragged、零draft与greedy混批。

    输入：4行draft数[K,K//2,0,K]；draft均0，target=[.125,.375,.5]，
      proposal=[.5,.375,.125]或None，U=.1接受/.8拒绝，recovered=2，bonus=1。
    场景：逐一构造拒绝位置0..K；最后一行greedy且预填7，用来检测越行覆盖。
    输出：接受前缀0、首拒绝2、尾部-1；全接受加1；零draft直接bonus；greedy保持7。
    依据：手算接受率.25或.125和首拒绝语义；不用生产函数计算expected。
    替身：只有pin allocator边界；所有张量运算和被测函数真实执行。
    """
    counts = [k, k // 2, 0, k]
    cu = torch.tensor(counts, dtype=torch.int32).cumsum(0).to(torch.int32)
    total = sum(counts)
    draft = torch.zeros(total, dtype=torch.int64)
    p = torch.tensor([[0.125, 0.375, 0.5]]).repeat(total, 1)
    proposal = None if none_proposal else torch.tensor([[0.5, 0.375, 0.125]]).repeat(total, 1)
    for reject_position in range(k + 1):
        output = torch.full((4, k + 1), -1, dtype=torch.int32)
        output[3].fill_(7)
        uniform = torch.full((total,), 0.1)
        if reject_position < k:
            uniform[reject_position] = 0.8
        production.rejection_random_sample_pytorch(
            output,
            cu,
            draft,
            proposal,
            p,
            torch.ones(4, dtype=torch.int32),
            torch.full((total,), 2, dtype=torch.int32),
            uniform,
            torch.tensor([False, False, False, True]),
            k,
            3,
            IS_NGRAM=none_proposal,
        )
        expected = torch.full_like(output, -1)
        if reject_position < k:
            expected[0, :reject_position] = 0
            expected[0, reject_position] = 2
        else:
            expected[0, :k] = 0
            expected[0, k] = 1
        expected[1, : k // 2] = 0
        expected[1, k // 2] = 1
        expected[2, 0] = 1
        expected[3].fill_(7)
        torch.testing.assert_close(output, expected, msg=f"K={k}, first_reject={reject_position}")


@pytest.mark.parametrize("none_proposal,expected", [(False, [2, 2]), (True, [1, 2])])
def test_recovered_tokens_follow_residual_and_request_rng(none_proposal, expected):
    """T22-03/04：同一target在显式q与None路径有不同残差分布。

    输入：两请求各1draft=0，p=[.125,.375,.5]，显式q=[.5,.375,.125]；
      exponential数E分别[1,.1,1]与[1,1,.1]，作为函数合法显式输入。
    输出：显式q残差只有token2，必须[2,2]；None去掉token0后分别选[1,2]。
    依据：逐候选手算残差/E；真实张量结果及输入p/q不被修改。
    替身：仅pin allocator，无被测函数或概率结果替换。
    """
    p = torch.tensor([[0.125, 0.375, 0.5], [0.125, 0.375, 0.5]])
    q = None if none_proposal else torch.tensor([[0.5, 0.375, 0.125], [0.5, 0.375, 0.125]])
    before = p.clone()
    output = torch.full((2,), -1, dtype=torch.int64)
    production.sample_recovered_tokens_pytorch(
        output,
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        q,
        p,
        torch.tensor([[1.0, 0.1, 1.0], [1.0, 1.0, 0.1]]),
        3,
        IS_NGRAM=none_proposal,
    )
    assert output.tolist() == expected
    torch.testing.assert_close(p, before)
    if q is not None:
        torch.testing.assert_close(q, torch.tensor([[0.5, 0.375, 0.125]]).repeat(2, 1))


@pytest.mark.parametrize("none_proposal,expected", [(False, 5), (True, 2)])
def test_reduced_candidates_return_global_vocabulary_ids(none_proposal, expected):
    """T22-07：reduced sampling必须回映射全局词表ID。

    输入：候选全局ID[5,2,0]、概率[.6,.3,.1]、draft5；显式q的ID0/2/5为.6/.3/.1。
    输出：显式q仅ID5有正残差，返回5；None去掉5后返回2，不能返回局部0/1。
    依据：独立手算，E固定全1仅测试确定性选择与映射，不当作真实RNG检验。
    替身：仅CPU pin allocator；分布和TP通信由其他用例验证。
    """
    output = torch.full((1,), -1, dtype=torch.int64)
    production.sample_recovered_tokens_pytorch(
        output,
        torch.tensor([1]),
        torch.tensor([5]),
        None if none_proposal else torch.tensor([[0.6, 0.0, 0.3, 0.0, 0.0, 0.1]]),
        torch.tensor([[0.6, 0.3, 0.1]]),
        torch.ones((1, 3)),
        3,
        IS_NGRAM=none_proposal,
        target_indices=torch.tensor([[5, 2, 0]]),
        enable_reduce_sampling=True,
    )
    assert output.item() == expected


@pytest.mark.parametrize("none_proposal", [False, True], ids=["explicit-q", "none-q"])
@pytest.mark.parametrize(
    "target",
    [[0.1, 0.2, 0.3, 0.4], [0.9, 0.05, 0.05, 0.0], [0.0, 0.5, 0.0, 0.5]],
    ids=["ordinary", "peaked-and-zero", "filtered"],
)
def test_final_sample_distribution_matches_target(target, none_proposal):
    """T22-09：真实随机数驱动拒绝+恢复生产函数，最终分布应等于target。

    输入：词表4，三组明确target；q=[.55,.15,.2,.1]或固定draft0的None路径。
      N=20000，seed=20260922，分块250避免生产token-to-batch矩阵占用过大。
    输出：每token频率距解析target不超过Hoeffding阈值，零概率token零命中。
    依据：M=24个预先固定比较、family alpha=.001，阈值在运行前确定；
      proposal用真实multinomial，U用rand，E用exponential，三者同真实Generator连续推进。
    替身：只去掉CPU pin_memory；不mock RNG、不重试挑seed、不使用另一采样器当oracle。
    回归：接受率/残差公式错误或强制greedy会使计数断言失败；未声称多token/NPU完成。
    """
    sample_count, batch_size = 20000, 250
    epsilon = math.sqrt(math.log(2 * 24 / 0.001) / (2 * sample_count))
    generator = torch.Generator().manual_seed(20260922)
    p = torch.tensor(target).repeat(batch_size, 1)
    q = torch.tensor([0.55, 0.15, 0.2, 0.1]).repeat(batch_size, 1)
    cu = torch.arange(1, batch_size + 1, dtype=torch.int32)
    counts = torch.zeros(4, dtype=torch.int64)
    for _ in range(sample_count // batch_size):
        draft = (
            torch.zeros(batch_size, dtype=torch.int64)
            if none_proposal
            else torch.multinomial(q, 1, generator=generator).flatten()
        )
        uniform = torch.rand(batch_size, generator=generator)
        exponential = torch.empty((batch_size, 4)).exponential_(generator=generator)
        recovered = torch.empty(batch_size, dtype=torch.int64)
        production.sample_recovered_tokens_pytorch(
            recovered,
            cu,
            draft,
            None if none_proposal else q,
            p,
            exponential,
            4,
            IS_NGRAM=none_proposal,
        )
        output = torch.full((batch_size, 2), -1, dtype=torch.int32)
        production.rejection_random_sample_pytorch(
            output,
            cu,
            draft,
            None if none_proposal else q,
            p,
            torch.zeros(batch_size, dtype=torch.int32),
            recovered,
            uniform,
            torch.zeros(batch_size, dtype=torch.bool),
            1,
            4,
            IS_NGRAM=none_proposal,
        )
        assert ((output[:, 0] >= 0) & (output[:, 0] < 4)).all()
        counts += torch.bincount(output[:, 0].long(), minlength=4)
    observed = counts.double() / sample_count
    deviation = (observed - torch.tensor(target, dtype=torch.float64)).abs()
    assert (deviation <= epsilon).all(), (counts.tolist(), epsilon, deviation.tolist())
    assert all(counts[i] == 0 for i, probability in enumerate(target) if probability == 0)
    print({"target": target, "none_proposal": none_proposal, "counts": counts.tolist(), "epsilon": epsilon})

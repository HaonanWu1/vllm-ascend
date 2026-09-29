# SPDX-License-Identifier: Apache-2.0
"""Execute rejection arithmetic and inspect its real dispatched Add operands."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.sample import rejection_sampler as production


class RecordIntegerAdd(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.add.Tensor:
            self.dtypes.extend(value.dtype for value in args if isinstance(value, torch.Tensor))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("block_verify", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("k", [1, 7, 15])
@pytest.mark.parametrize("accept", [False, True])
def test_rejection_preserves_index_width_and_exact_tokens(monkeypatch, block_verify, dtype, k, accept):
    """输入：K1/7/15，draft数[K,0,K]，累计长度int32/int64，第三行greedy。

    输出：接受时第一行K个0再bonus1；拒绝时首位recovered2、余位-1；
      零draft行仅bonus1，greedy行所有7保持不变。普通和block verify均执行。
    场景：无dtype常量不能把int32索引提升到会触发310P故障的int64 Add。
    依据：单点target或p0=.125、U=.8独立手算；记录真实ATen调用并原样执行，
      禁止替换算子或返回值。int32输入不允许出现int64 Add，int64输入仍验证数值。
    替身：仅取消CPU环境的pin_memory；本UT验证数学结果与实际算子输入契约，
      设备执行和原服务调用历史另行回归，不能用本项替代NPU测试。
    """
    for name in ("tensor", "arange", "full", "ones"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    output = torch.full((3, k + 1), -1, dtype=torch.int32)
    output[2].fill_(7)
    probs = torch.tensor([[1.0, 0.0, 0.0] if accept else [0.125, 0.375, 0.5]]).repeat(2 * k, 1)
    fn = (
        production.rejection_random_sample_block_verify_pytorch
        if block_verify
        else production.rejection_random_sample_pytorch
    )
    with RecordIntegerAdd() as calls:
        fn(
            output,
            torch.tensor([k, k, 2 * k], dtype=dtype),
            torch.zeros(2 * k, dtype=torch.int32),
            None,
            probs,
            torch.ones(3, dtype=torch.int32),
            torch.full((2 * k,), 2, dtype=torch.int32),
            torch.full((2 * k,), 0.8),
            torch.tensor([False, False, True]),
            k,
            3,
            IS_NGRAM=True,
        )
    expected = torch.full_like(output, -1)
    if accept:
        expected[0, :k] = 0
        expected[0, k] = 1
    else:
        expected[0, 0] = 2
    expected[1, 0] = 1
    expected[2].fill_(7)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if dtype == torch.int32:
        assert torch.int64 not in calls.dtypes, calls.dtypes

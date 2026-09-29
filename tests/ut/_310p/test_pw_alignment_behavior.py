# SPDX-License-Identifier: Apache-2.0
"""T08: distinguish current validation from the actual uniform padding constraint."""

from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend._310p import dflash_full_and_piecewise as production


def config(piecewise, full=80, mbt=2560):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=7),
        scheduler_config=SimpleNamespace(max_num_seqs=10, max_num_batched_tokens=mbt),
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=None,
            max_cudagraph_capture_size=None,
            compile_sizes=[],
        ),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": piecewise,
                    "full_capture_size": full,
                }
            }
        },
    )


def dispatcher_with_union(value, sizes):
    """Use real dispatcher methods; the explicit union is a separate algorithm input."""
    value.compilation_config.cudagraph_capture_sizes = sizes
    value.compilation_config.max_cudagraph_capture_size = max(sizes)
    dispatcher = CudagraphDispatcher.__new__(CudagraphDispatcher)
    dispatcher.vllm_config = value
    dispatcher.compilation_config = value.compilation_config
    dispatcher.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    dispatcher.uniform_decode_query_len = 8
    dispatcher.specialize_lora_count = False
    dispatcher._compute_bs_to_padded_graph_size()
    return dispatcher


@pytest.mark.parametrize("pw", [1220, [1220], [1220, 2400]])
def test_large_non_aligned_pw_never_intercepts_uniform_decode_up_to_80(pw, monkeypatch):
    """T08-01：PW大于80时，uniform 8..80不会进入1220这个非8对齐桶。

    输入：K7/ms10/MBT2560，PW1220标量或列表[1220]/[1220,2400]，FULL80。
    输出：当前校验只接受标量、拒绝列表；但相同真实padding算法下全部uniform均pad到80。
      Mixed实际token83则可pad到1220，descriptor.uniform=False，不受8整除限制。
    场景：验证用户指出的scalar/list差异与guard必要性，分开报告现状和底层约束。
    替身：仅硬件身份；调用真实padding/descriptor方法。手动union用于算法验证，
      没有绕过服务校验宣称列表可部署；这一测试不能证明完整列表服务已支持。
    """
    monkeypatch.setattr(production, "is_310p", lambda: True)
    value = config(pw)
    if isinstance(pw, list):
        with pytest.raises(ValueError, match="list values must be divisible"):
            production.apply_dflash_full_and_piecewise_capture_config(value)
    else:
        assert production.apply_dflash_full_and_piecewise_capture_config(value)
    sizes = sorted({80, *(pw if isinstance(pw, list) else [pw])})
    dispatcher = dispatcher_with_union(value, sizes)
    for requests in range(1, 11):
        actual = dispatcher._create_padded_batch_descriptor(requests * 8, True, False)
        assert actual.num_tokens == 80 and actual.num_reqs == 10 and actual.uniform
    mixed = dispatcher._create_padded_batch_descriptor(83, False, False)
    assert mixed.num_tokens == 1220 and not mixed.uniform


def test_small_non_aligned_pw_can_break_uniform_descriptor():
    """T08-02：小PW桶确实能触发uniform整除失败，但Mixed本身仍合法。

    输入：共享sizes=[32,40,60,80]，K7，uniform实际48或Mixed实际49。
    输出：uniform48先pad到60再触发AssertionError；Mixed49合法得到60/非uniform。
    依据：真实dispatcher生产padding与descriptor校验；无mock。
    局限：独立算法层反例，配置入口本来拒绝PW列表[32,60]，不当作成功部署用例。
    """
    dispatcher = dispatcher_with_union(config([32, 60], [40, 80]), [32, 40, 60, 80])
    assert dispatcher._bs_to_padded_graph_size[48] == 60
    with pytest.raises(AssertionError):
        dispatcher._create_padded_batch_descriptor(48, True, False)
    mixed = dispatcher._create_padded_batch_descriptor(49, False, False)
    assert mixed.num_tokens == 60 and not mixed.uniform

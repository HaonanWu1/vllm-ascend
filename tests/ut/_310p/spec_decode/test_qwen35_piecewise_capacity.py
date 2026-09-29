# SPDX-License-Identifier: Apache-2.0
"""Regression for the Qwen3.5 pure-PW service crash observed on 2026-09-28.

Valid configurations assert successful numerical behavior; do not xfail or
invert the original pure-PW crash regression.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend._310p import dflash_full_and_piecewise, dflash_full_decode_only, dflash_piecewise
from vllm_ascend._310p.ops import rotary_embedding as rope
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310


@pytest.mark.parametrize(
    "mode,descriptor",
    [
        (CUDAGraphMode.PIECEWISE, 64),
        (CUDAGraphMode.PIECEWISE, 80),
        (CUDAGraphMode.PIECEWISE, 1216),
        (CUDAGraphMode.FULL_AND_PIECEWISE, 1220),
        (CUDAGraphMode.FULL_DECODE_ONLY, 80),
    ],
    ids=["pure-pw-below-capacity", "pure-pw-at-capacity", "pure-pw-1216", "fap-pw-1220-control", "fdo-prefill-control"],
)
def test_large_target_descriptor_preserves_small_draft_query_capacity(monkeypatch, mode, descriptor):
    """输入：MBT1280、MS10、K7，真实80元素query缓冲区，target描述符64/80/1216/1220。

    输出：四次准备均成功；query/context有效位置的cos/sin等于独立标量三角值，
    填充区域为单位旋转；query从8→16→描述符/容量上界→8，context从128→64→128→1，
    每轮刷新后缓冲地址保持不变，缩批不得留下上一轮位置数值。
    场景：target描述符小于/等于/大于draft容量，以及context实际长度超过小描述符；
    FAP是原始1216首请求崩溃边界的成功对照；FDO的NONE回退验证新增条件的假分支，
    数值与缓冲合同保持不变。不能把合法大PW当作异常用例。
    替身：仅310P平台检测和轻量配置/缓存对象；真实_get_positions、容量判断、
    prepare_full_decode_draft_rope_310、缓存填充都执行，不mock数值或被测函数。
    限制：CPU缓冲合同UT，不声称本用例执行ACL图；整模型服务器日志提供设备复现。
    """
    for module in (dflash_full_decode_only, dflash_full_and_piecewise, dflash_piecewise):
        monkeypatch.setattr(module, "is_310p", lambda: True)
    # Isolate production module state; monkeypatch restores every original value.
    for name in (
        "_full_decode_query_cos",
        "_full_decode_query_sin",
        "_full_decode_context_cos",
        "_full_decode_context_sin",
    ):
        monkeypatch.setattr(rope, name, None)
    monkeypatch.setattr(rope, "_full_decode_rope_precomputed", False)
    monkeypatch.setattr(rope, "_full_decode_rope_source", "query")

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 1220,
                    "full_capture_size": 80,
                }
            }
        },
    )
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = config
    proposer.runner = SimpleNamespace(vllm_config=config, max_num_tokens=1280)
    proposer.method = "dflash"
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.max_query_tokens = 80
    proposer.positions = torch.empty(80, dtype=torch.int32)
    proposer._context_positions_buffer = torch.empty(1280, dtype=torch.int32)
    frequency = 1_000_000 ** (-torch.arange(64, dtype=torch.float64) / 64)
    angles = torch.arange(2048, dtype=torch.float64)[:, None] * frequency
    proposer._full_decode_draft_rotary_310 = SimpleNamespace(
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    )

    pointers = None
    for query_count, context_count, offset in [(8, 128, 17), (16, 64, 29), (min(80, descriptor), 128, 37), (8, 1, 41)]:
        proposer.positions.copy_(torch.arange(80, dtype=torch.int32) + offset)
        proposer._context_positions_buffer.copy_(torch.arange(1280, dtype=torch.int32) + offset + 3)
        proposer._dflash_num_context = context_count
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=proposer.positions[:query_count],
            query_actual_tokens=query_count,
            descriptor_tokens=descriptor,
            runtime_mode=CUDAGraphMode.NONE if mode == CUDAGraphMode.FULL_DECODE_ONLY else CUDAGraphMode.PIECEWISE,
        )
        assert prepared is True
        buffers = rope.get_full_decode_draft_rope_buffers_310()
        actual_pointers = tuple(buffer.data_ptr() for buffer in buffers)
        if pointers is not None:
            assert actual_pointers == pointers
        pointers = actual_pointers
        for cos, sin, count, start in (
            (*buffers[:2], query_count, offset),
            (*buffers[2:], context_count, offset + 3),
        ):
            expected_cos = torch.ones(1280, 128)
            expected_sin = torch.zeros(1280, 128)
            for token in range(count):
                for channel in range(128):
                    angle = (start + token) / (1_000_000 ** ((channel % 64) / 64))
                    expected_cos[token, channel] = math.cos(angle)
                    expected_sin[token, channel] = math.sin(angle)
            torch.testing.assert_close(cos[0, :, 0], expected_cos, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(sin[0, :, 0], expected_sin, rtol=1e-6, atol=1e-6)
        proposer._finish_full_decode_draft_rope(prepared)
        assert rope._full_decode_rope_precomputed is False


def test_piecewise_rejects_active_queries_beyond_allocated_capacity(monkeypatch):
    """输入：PW1216、80元素query缓冲，却声明81个有效query，属于真实越界输入。

    输出：准备函数必须在访问/改写位置缓冲前拒绝，错误明确标出actual81/query80。
    场景：合法的大target描述符不能掩盖draft实际越界；这不是把原始合法PW崩溃判为通过。
    替身：仅310P检测与轻量配置；容量判断、异常和输入不变断言均执行真实代码。
    """
    monkeypatch.setattr(dflash_piecewise, "is_310p", lambda: True)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = config
    proposer.method = "dflash"
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.max_query_tokens = 80
    proposer.positions = torch.arange(80, dtype=torch.int32)
    before = proposer.positions.clone()
    with pytest.raises(RuntimeError, match="active query extent exceeds.*actual=81, query_descriptor=80"):
        proposer._prepare_full_decode_draft_rope(
            query_positions=proposer.positions,
            query_actual_tokens=81,
            descriptor_tokens=1216,
            runtime_mode=CUDAGraphMode.PIECEWISE,
        )
    torch.testing.assert_close(proposer.positions, before, rtol=0, atol=0)

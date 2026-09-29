# SPDX-License-Identifier: Apache-2.0
"""T03: device slot mapping must not write outside its logical destination slice."""

import pytest
import torch
import torch_npu

from tests.ut._310p.test_prepare_inputs_behavior import make_runner


@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 11, 15, 16, 17])
def test_device_slot_mapping_preserves_all_adjacent_sentinels(tokens):
    """T03-004：真实NPU slot mapping在32/64-byte尾部边界不越界写。

    输入：block128，物理块5，positions0..N-1；N=1/7/8/9/11/15/16/17，64元素buffer填-999。
    输出：前N个slot=640..640+N-1；其余逐元素保持-999，dtype为int32且留在NPU。
    场景：runner中发现11-token写入改变第11..15位置，独立调用生产BlockTable缩小根因。
    依据：CPU手算物理地址和整块尾哨兵；无mock，不把异常尾部视为合法padding。
    失败保持FAIL，不能通过扩大切片/放宽断言隐藏；仅测试正常合法输入，不注入坏kernel。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    runner = make_runner("npu:0")
    table = runner.input_batch.block_table.block_tables[0]
    runner.input_batch.block_table.commit_block_table(2)
    assert table.slot_mapping.gpu.cpu().tolist() == [-999] * 64
    positions = torch.arange(tokens, dtype=torch.int64, device=runner.device)
    indices = torch.zeros(tokens, dtype=torch.int32, device=runner.device)
    table.compute_slot_mapping_device(indices, positions)
    torch.npu.synchronize()
    assert table.slot_mapping.gpu.dtype == torch.int32
    assert table.slot_mapping.gpu.device == runner.device
    assert table.slot_mapping.gpu.cpu().tolist() == list(range(640, 640 + tokens)) + [-999] * (64 - tokens)

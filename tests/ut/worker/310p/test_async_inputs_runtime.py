# SPDX-License-Identifier: Apache-2.0
"""T02/T03: complete runner input preparation with actual 310P events and tensors."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch_npu
from vllm.config import CUDAGraphMode

from tests.ut._310p.test_prepare_inputs_behavior import make_runner
from vllm_ascend._310p import model_runner_310p as production


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE])
def test_complete_async_prepare_uses_corrected_positions_after_reorder(mode):
    """T02-004/T03：真实异步producer→event→runner，换行后只修正一次。

    输入：上一批[A,B]的device computed=[100,200]、accepted=[2,4]、draft=[7,7]；
      当前[B,new]、CPU乐观computed=[208,50]，query=[8,3]，B sample801/draft802..808。
    输出：device computed=[204,50]、positions=[204..211,50..52]、seq=[212,53,0,0]；
      B映射物理块11得到slot1484..1491，new映射块13得到1714..1716；accepted=[4,1]。
    场景：FDO/FAP的int32 staging、新请求插入、旧请求换行、partial acceptance和mixed输入。
    依据：上一轮200+4及独立block128地址计算；若重复修正将得到208，断言必须失败。
    替身：仅RC/lmhead布尔配置；真实NPU stream/event、H2D/D2H、完整_prepare_inputs，
      update_num_computed、slot、spec metadata均未替换。测试不声称已执行图capture/replay。
    时序：前置哨兵检查在producer提交之前；提交后不主动同步producer，交由runner消费event。
      该检查验证实际事件链路，但没有注入确定性设备延迟，不能单凭通过证明所有竞态均被排除。
    """
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    runner = make_runner("npu:0")
    runner.vllm_config.compilation_config.cudagraph_mode = mode
    runner.use_async_scheduling = runner.use_async_spec_decode = True
    for name in ("_fdo_position_base_i32", "_fdo_query_pos_i32", "_fdo_positions_i32"):
        setattr(runner, name, torch.empty(64, dtype=torch.int32, device=runner.device))
    runner._fdo_position_staging_logged = False
    runner.num_computed_tokens.copy_(torch.tensor([100, 200, 0, 0], device=runner.device, dtype=torch.int32))
    runner.prev_num_draft_tokens.np[:] = [7, 7, 0, 0]
    batch = runner.input_batch
    batch.req_ids, batch.req_id_to_index = ["B", "new"], {"B": 0, "new": 1}
    batch.prev_req_id_to_index = {"A": 0, "B": 1}
    batch.num_computed_tokens_cpu[:] = [208, 50, 0, 0]
    batch.num_prompt_tokens[:] = [100, 53, 0, 0]
    table = batch.block_table.block_tables[0]
    table.block_table.np[:2] = [[7, 11], [13, 17]]
    batch.prev_sampled_token_ids = torch.tensor([[701], [801]], dtype=torch.int32, device=runner.device)
    runner._draft_token_ids = torch.tensor(
        [list(range(702, 709)), list(range(802, 809))], dtype=torch.int32, device=runner.device
    )
    runner.requests = {"B": SimpleNamespace(num_tokens=208), "new": SimpleNamespace(num_tokens=53)}
    runner.valid_sampled_token_count_gpu = torch.empty(4, dtype=torch.int32, device=runner.device)
    assert table.slot_mapping.gpu.cpu().tolist() == [-999] * 64
    producer = torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    event = torch.npu.Event()
    with torch.npu.stream(producer):
        runner.valid_sampled_token_count_gpu.copy_(torch.tensor([2, 4, 1, 1], dtype=torch.int32))
        event.record()
    runner.num_accepted_tokens_event = event
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=11,
        num_scheduled_tokens={"B": 8, "new": 3},
        scheduled_spec_decode_tokens={"B": list(range(802, 809))},
        scheduled_new_reqs=[],
    )
    with (
        patch.object(production, "is_rc_device", return_value=False),
        patch.object(production, "lmhead_tp_enable", return_value=False),
    ):
        logits, metadata, total = runner._prepare_inputs(schedule, np.array([8, 3], dtype=np.int32))
    torch.npu.synchronize()
    assert total == 11 and logits.cpu().tolist() == list(range(8)) + [10]
    assert metadata.num_draft_tokens == [7, 0]
    assert runner.num_computed_tokens[:2].cpu().tolist() == [204, 50]
    assert runner.positions[:11].cpu().tolist() == list(range(204, 212)) + [50, 51, 52]
    assert runner.seq_lens.cpu().tolist() == [212, 53, 0, 0]
    assert runner.num_accepted_tokens.gpu[:2].cpu().tolist() == [4, 1]
    print(
        {
            "mode": mode.name,
            "slot_storage": table.slot_mapping.gpu.cpu().tolist(),
            "slot_slice": table.slot_mapping.gpu[:12].cpu().tolist(),
        }
    )
    assert runner.input_ids.gpu[:11].cpu().tolist() == list(range(801, 809)) + [2050, 2051, 2052]
    assert runner.optimistic_seq_lens_cpu.tolist() == [216, 53, 0, 0]
    assert runner.gdn_query_start_loc.gpu.cpu().tolist() == [0, 8, 11, 11, 11, 11]
    assert table.slot_mapping.gpu.cpu().tolist() == list(range(1484, 1492)) + [1714, 1715, 1716] + [-999] * 53

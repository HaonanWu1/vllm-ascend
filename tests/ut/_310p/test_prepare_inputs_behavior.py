# SPDX-License-Identifier: Apache-2.0
"""T02/T03: execute the complete production input preparation, not source strings."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend._310p import model_runner_310p as production
from vllm_ascend._310p.block_table import BlockTable, MultiGroupBlockTable


def make_runner(device="cpu"):
    """Construct explicit runtime state; all tested methods/buffers remain real."""
    device = torch.device(device)
    runner = production.NPUModelRunner310.__new__(production.NPUModelRunner310)
    runner.device, runner.pin_memory = device, False
    max_reqs, max_tokens = 4, 64

    def buffer(size, dtype=torch.int32):
        return CpuGpuBuffer(size, dtype=dtype, device=device, pin_memory=False)

    for name in (
        "num_accepted_tokens",
        "prev_positions",
        "prev_num_draft_tokens",
        "num_scheduled_tokens",
        "num_decode_draft_tokens",
        "discard_request_indices",
    ):
        setattr(runner, name, buffer(max_reqs))
    for name in ("query_start_loc", "gdn_query_start_loc"):
        setattr(runner, name, buffer(max_reqs + 2))
    for name in ("req_indices", "query_pos", "input_ids"):
        setattr(runner, name, buffer(max_tokens))
    runner.positions = torch.full((max_tokens,), -777, dtype=torch.int64, device=device)
    runner._positions_cpu_buf = torch.zeros(max_tokens, dtype=torch.int64)
    runner._positions_np_buf = runner._positions_cpu_buf.numpy()
    runner.arange_np = np.arange(max_tokens, dtype=np.int64)
    runner._arange_scratch = np.zeros(max_tokens, dtype=np.int32)
    runner.num_computed_tokens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    runner.seq_lens = torch.full((max_reqs,), -777, dtype=torch.int32, device=device)
    runner.optimistic_seq_lens_cpu = torch.zeros(max_reqs, dtype=torch.int32)
    runner.speculative_config = SimpleNamespace(method="dflash")
    runner.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    runner.vllm_config = SimpleNamespace(
        speculative_config=runner.speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    runner.uniform_decode_query_len = 8
    runner.use_async_scheduling = runner.use_async_spec_decode = False
    runner.valid_sampled_token_count_gpu = runner.num_accepted_tokens_event = None
    runner.pcp_size = runner.dcp_size = 1
    runner._has_gdn = True
    runner.enable_prompt_embeds = runner.is_multimodal_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner._needs_seq_lens_cpu_sync = False
    runner.lora_config = None
    runner.is_kv_consumer = False
    runner.num_spec_tokens = 7
    runner._draft_token_ids = None

    block = BlockTable.__new__(BlockTable)
    block.block_size = block.physical_block_size = 128
    block.max_num_blocks_per_req, block.blocks_per_phys_block = 2, 1
    block.pcp_world_size = block.dcp_world_size = 1
    block.pcp_rank = block.dcp_rank = 0
    block.cp_kv_cache_interleave_size = 1
    block.block_table = CpuGpuBuffer(max_reqs, 2, dtype=torch.int32, device=device, pin_memory=False)
    block.block_table.np[:] = [[5, 9], [7, 11], [13, 17], [19, 23]]
    block.slot_mapping = buffer(max_tokens)
    block.slot_mapping.gpu.fill_(-999)
    block.slot_mapping.cpu.fill_(-999)
    tables = MultiGroupBlockTable.__new__(MultiGroupBlockTable)
    tables.block_tables = [block]
    token_ids = torch.stack([torch.arange(256, dtype=torch.int32) + 1000 * (i + 1) for i in range(max_reqs)])
    computed = torch.tensor([127, 12, 0, 0], dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["A", "B"],
        req_id_to_index={"A": 0, "B": 1},
        block_table=tables,
        num_computed_tokens_cpu_tensor=computed,
        num_computed_tokens_cpu=computed.numpy(),
        num_accepted_tokens_cpu=np.ones(max_reqs, dtype=np.int32),
        prev_req_id_to_index=None,
        prev_sampled_token_ids=None,
        token_ids_cpu_tensor=token_ids,
        token_ids_cpu=token_ids.numpy(),
        req_prompt_embeds={},
        num_prompt_tokens=np.array([130, 15, 0, 0], dtype=np.int32),
    )
    runner.requests = {"A": SimpleNamespace(num_tokens=130), "B": SimpleNamespace(num_tokens=15)}
    return runner


@pytest.mark.parametrize("rc", [False, True], ids=["discrete", "rc"])
def test_prepare_inputs_cross_block_and_clears_shrunk_batch(monkeypatch, rc):
    """T02-001/T03：完整_prepare_inputs跨block后缩批，数值与尾部逐项验证。

    输入：A computed127/query2、B computed12/query3；block128，物理块A=[5,9]、B=[7,11]。
    输出：positions=[127,128,12,13,14]，slot=[767,1152,908,909,910]，seq=[129,15]；
      tokens=[1127,1128,2012,2013,2014]，logits=[1,4]；A未完成prefill故discard=[0]。
    第二步：只保留B于row0、computed15/query1；seq尾部归零，query/GDN尾部符合各自合同。
    依据：独立绝对位置与physical_block*128+offset手算，不patch_prepare_inputs及其子逻辑。
    替身：CPU pin_memory分配、RC与lmhead配置边界；真实CpuGpuBuffer/BlockTable计算。
    局限：同步host路径；设备异步依赖须由单独NPU测试证明。
    """
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *args, **kwargs: self)
    runner = make_runner()
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=5,
        num_scheduled_tokens={"A": 2, "B": 3},
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
    )
    with (
        patch.object(production, "is_rc_device", return_value=rc),
        patch.object(production, "lmhead_tp_enable", return_value=False),
    ):
        logits, metadata, total = runner._prepare_inputs(schedule, np.array([2, 3], dtype=np.int32))
        assert total == 5 and metadata is None
        assert logits.tolist() == [1, 4]
        assert runner.positions[:5].tolist() == [127, 128, 12, 13, 14]
        assert runner.input_ids.gpu[:5].tolist() == [1127, 1128, 2012, 2013, 2014]
        assert runner.seq_lens.tolist() == [129, 15, 0, 0]
        assert runner.query_start_loc.gpu.tolist() == [0, 2, 5, -1, -1, -1]
        assert runner.gdn_query_start_loc.gpu.tolist() == [0, 2, 5, 5, 5, 5]
        assert runner.input_batch.block_table.block_tables[0].slot_mapping.gpu[:6].tolist() == [
            767,
            1152,
            908,
            909,
            910,
            -999,
        ]
        assert runner.num_discarded_requests == 1
        assert runner.discard_request_indices.gpu[0].item() == 0

        batch = runner.input_batch
        batch.num_reqs, batch.req_ids, batch.req_id_to_index = 1, ["B"], {"B": 0}
        batch.prev_req_id_to_index = {"A": 0, "B": 1}
        batch.num_computed_tokens_cpu[0] = 15
        batch.token_ids_cpu_tensor[0].copy_(batch.token_ids_cpu_tensor[1])
        batch.block_table.block_tables[0].block_table.cpu[0].copy_(batch.block_table.block_tables[0].block_table.cpu[1])
        schedule = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"B": 1},
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[],
        )
        logits, metadata, total = runner._prepare_inputs(schedule, np.array([1], dtype=np.int32))
        assert total == 1 and metadata is None and logits.tolist() == [0]
        assert runner.positions[0].item() == 15
        assert runner.input_ids.gpu[0].item() == 2015
        assert runner.seq_lens.tolist() == [16, 0, 0, 0]
        assert runner.query_start_loc.gpu.tolist() == [0, 1, -1, -1, -1, -1]
        assert runner.gdn_query_start_loc.gpu.tolist() == [0, 1, 1, 1, 1, 1]
        assert runner.prev_positions.np[0] == 1
        assert batch.block_table.block_tables[0].slot_mapping.gpu[0].item() == 911


def test_prepare_input_ids_reorders_previous_samples_and_drafts():
    """T02-002：异步token回填按请求ID重排，不能按旧batch行直接复制。

    输入：当前[B,new,A]，旧[A,B]；每旧请求有2draft；旧sample=[31,41]，draft=[[32,33],[42,43]]。
    输出：按当前三段[3,2,3]得到[41,42,43,103,104,31,32,33]，第8位哨兵保持。
    场景：旧请求换行并插入新prefill，生产_prepare_input_ids完整scatter两个token来源。
    依据：逐请求sample+draft拼接，新请求保留CPU token；无mock，真实CPU tensor/buffer。
    """
    runner = make_runner()
    runner.num_spec_tokens = 2
    runner.input_batch.req_ids = ["B", "new", "A"]
    runner.input_batch.req_id_to_index = {"B": 0, "new": 1, "A": 2}
    runner.input_batch.prev_req_id_to_index = {"A": 0, "B": 1}
    runner.input_batch.prev_sampled_token_ids = torch.tensor([[31], [41]], dtype=torch.int32)
    runner._draft_token_ids = torch.tensor([[32, 33], [42, 43]], dtype=torch.int32)
    runner.input_ids.cpu[:8] = torch.arange(100, 108, dtype=torch.int32)
    runner.input_ids.gpu.fill_(-999)
    schedule = SimpleNamespace(scheduled_spec_decode_tokens={"B": [42, 43], "A": [32, 33]})
    runner._prepare_input_ids(schedule, 3, 8, np.array([3, 5, 8], dtype=np.int32))
    assert runner.input_ids.gpu[:9].tolist() == [41, 42, 43, 103, 104, 31, 32, 33, -999]


def test_complete_prepare_inputs_builds_ragged_spec_metadata(monkeypatch):
    """T02-003：真实输入准备继续执行公共spec metadata构建，验证ragged索引。

    输入：A query3/draft2已过prompt、B query8/draft7尚在prefill；沿用跨block绝对位置。
    输出：logits0..10，target索引[0,1,3..9]，bonus[2,10]，draft来自各段第2个token起；
      num_decode_draft_tokens=[2,-1,-1,-1]，A/B状态不能混用。
    依据：逐请求query中target/bonus的明确布局与真实输入token ID。
    替身：只禁用CPU pin allocator、提供RC/lmhead配置；metadata和所有数学操作未替换。
    """
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *args, **kwargs: self)
    runner = make_runner()
    runner.input_batch.num_prompt_tokens[0] = 100
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=11,
        num_scheduled_tokens={"A": 3, "B": 8},
        scheduled_spec_decode_tokens={"A": [8, 9], "B": list(range(7))},
        scheduled_new_reqs=[],
    )
    with (
        patch.object(production, "is_rc_device", return_value=False),
        patch.object(production, "lmhead_tp_enable", return_value=False),
    ):
        logits, metadata, total = runner._prepare_inputs(schedule, np.array([3, 8], dtype=np.int32))
    assert total == 11 and logits.tolist() == list(range(11))
    assert metadata.num_draft_tokens == [2, 7]
    assert metadata.cu_num_draft_tokens.tolist() == [2, 9]
    assert metadata.cu_num_sampled_tokens.tolist() == [3, 11]
    assert metadata.target_logits_indices.tolist() == [0, 1, 3, 4, 5, 6, 7, 8, 9]
    assert metadata.bonus_logits_indices.tolist() == [2, 10]
    assert metadata.draft_token_ids.tolist() == [1128, 1129, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    assert runner.num_decode_draft_tokens.gpu.tolist() == [2, -1, -1, -1]

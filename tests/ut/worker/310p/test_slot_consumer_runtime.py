# SPDX-License-Identifier: Apache-2.0
"""Run production metadata padding followed by the real 310P KV writer."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.utils import CpuGpuBuffer

from tests.ut._310p.test_prepare_inputs_behavior import make_runner
from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310, AscendAttentionBackendImpl310
from vllm_ascend.attention.attention_v1 import AscendAttentionState


def slot_runner():
    runner = make_runner("npu:0")
    runner.input_batch.num_prompt_tokens_cpu_tensor = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    runner.input_batch.num_computed_tokens_cpu_tensor.fill_(1)
    runner.optimistic_seq_lens_cpu.fill_(128)
    runner.seq_lens.fill_(128)
    runner.query_start_loc.cpu[:] = torch.tensor([0, 8, 16, 24, 32, 32], dtype=torch.int32)
    runner.query_start_loc.copy_to_gpu()
    for name in ("group_len", "group_key_idx", "group_key_cache_idx"):
        setattr(runner, name, CpuGpuBuffer(4, dtype=torch.int32, device=runner.device, pin_memory=False))
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.float16),
                layer_names=["attention"],
            )
        ]
    )
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False, is_encoder_decoder=False)
    runner.use_compress = runner._has_gdn = False
    runner.enable_hamming_sparse = runner.is_mm_prefix_lm = False
    runner.attn_groups = [[]]
    runner.drafter = SimpleNamespace()
    runner.actual_seq_lengths_q = [8]
    runner.decode_token_per_req = 8
    runner.attn_state = AscendAttentionState.SpecDecoding
    runner.max_model_len = 8192
    table = runner.input_batch.block_table[0]
    table.block_table.cpu[:] = torch.arange(1, 5, dtype=torch.int32)[:, None]
    table.block_table.copy_to_gpu()
    return runner, table


def writer():
    impl = AscendAttentionBackendImpl310.__new__(AscendAttentionBackendImpl310)
    impl.key_cache = impl.value_cache = None
    impl.attn_type = AttentionType.DECODER
    impl.kv_sharing_target_layer_name = None
    impl.is_kv_producer = False
    cache = tuple(
        torch_npu.empty_with_format(
            size=AscendAttentionBackend310.get_kv_cache_shape(24, 128, 1, 128)[1:],
            dtype=torch.float16,
            device="npu",
            acl_format=29,
        ).fill_(-9)
        for _ in range(2)
    )
    key = torch.arange(1, 65, dtype=torch.float16, device="npu")[:, None, None].expand(64, 1, 128).contiguous()
    value = key + 1000
    return impl, cache, key, value


def check_cache(cache, slots):
    for side, base in [(0, 1), (1, 1001)]:
        actual = cache[side].cpu().permute(0, 2, 1, 3).reshape(-1, 128)
        expected = torch.full_like(actual, -9.0)
        expected[slots] = torch.arange(base, base + len(slots), dtype=torch.float16)[:, None]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("count,padded", [(11, 11), (11, 64), (9, 64), (17, 64)])
def test_dirty_slot_tail_is_cleared_or_excluded_before_kv_write(count, padded):
    """输入：真实device slot计算N9/11/17；NONE不补齐或PW补齐64，block128。

    输出：仅slot128..128+N-1的KV被写入，其余3072-N个cache槽位均保持-9；
      padded场景生产metadata builder把N..64清成PAD=-1。
    场景：确认UT发现的尾部污染是否穿过真实metadata清理并到达KV消费。
    依据：独立逐槽KV内容期望；完整_build_attention_metadata及真实310P
      reshape_and_cache都执行，无算子/结果mock。只省略与槽位无关的模型和attention计算。
    边界：此处验证NONE/PW写入；FULL重放使用下一用例的合法8倍数输入。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    runner, table = slot_runner()
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch.num_prompt_tokens_cpu_tensor.fill_(128)
    runner.input_batch.num_computed_tokens_cpu_tensor.zero_()
    runner.optimistic_seq_lens_cpu.fill_(count)
    runner.seq_lens.fill_(count)
    runner.query_start_loc.cpu.fill_(count)
    runner.query_start_loc.cpu[0] = 0
    runner.query_start_loc.copy_to_gpu()
    runner.actual_seq_lengths_q = [count]
    table.compute_slot_mapping_device(
        torch.zeros(count, dtype=torch.int32, device="npu"), torch.arange(count, dtype=torch.int64, device="npu")
    )
    before = table.slot_mapping.gpu.cpu().tolist()
    _, metadata = runner._build_attention_metadata(count, 1, count, num_tokens_padded=padded, num_reqs_padded=1)
    assert metadata is not None
    if padded > count:
        assert table.slot_mapping.gpu[count:padded].cpu().tolist() == [-1] * (padded - count)
    impl, cache, key, value = writer()
    impl.reshape_and_cache(key, key, value, (cache[0], cache[1]), metadata, key)
    torch.npu.synchronize()
    check_cache(cache, list(range(128, 128 + count)))
    print({"count": count, "padded": padded, "before_tail": before[count:24], "kv_tail_preserved": True})


def test_full_graph_replay_uses_padding_guard_and_control_detects_consumption():
    """输入：捕获32-token KV写图，重放K7合法FULL批次N8→24→8；每步生产builder补PAD。

    输出：每轮仅有效槽位写入、其余cache哨兵保持。额外因果对照在builder之后
      将一个padding slot改为合法保留槽位3000，重放必须实测该槽位被改写。
    场景：说明图内padding位置确实可被kernel读取，而生产PAD清理阻止污染消费。
    依据：真实NPUGraph.capture/replay、真实KV kernel及独立全cache期望；不模拟图。
    边界：N11不是K7合法FULL批次，不将它伪装为FULL成功用例。
    """
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    runner, table = slot_runner()
    runner.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    impl, cache, key, value = writer()
    table.slot_mapping.gpu.fill_(-1)
    metadata = SimpleNamespace(slot_mapping=table.slot_mapping.gpu[:32], num_actual_tokens=32)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        impl.reshape_and_cache(key, key, value, (cache[0], cache[1]), metadata, key)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        impl.reshape_and_cache(key, key, value, (cache[0], cache[1]), metadata, key)
    for count in [8, 24, 8]:
        for tensor in cache:
            tensor.fill_(-9)
        # Each request owns a separate physical block; refresh the block table
        # as input preparation does after the previous step padded inactive rows.
        table.block_table.copy_to_gpu()
        runner.input_batch.num_computed_tokens_cpu_tensor.zero_()
        runner.seq_lens.fill_(8)
        runner.optimistic_seq_lens_cpu.fill_(8)
        runner.actual_seq_lengths_q = list(range(8, count + 1, 8))
        runner.query_start_loc.cpu[:] = torch.arange(6, dtype=torch.int32).clamp(max=count // 8) * 8
        runner.query_start_loc.copy_to_gpu()
        indices = torch.arange(count, dtype=torch.int32, device="npu")
        table.compute_slot_mapping_device(indices // 8, (indices % 8).to(torch.int64))
        runner._build_attention_metadata(count, count // 8, 8, num_tokens_padded=32, num_reqs_padded=4)
        graph.replay()
        torch.npu.synchronize()
        check_cache(cache, [(index // 8 + 1) * 128 + index % 8 for index in range(count)])
    # Positive control: modify the actual graph input after the production guard.
    poisoned = table.slot_mapping.gpu.cpu()
    poisoned[8] = 3000
    table.slot_mapping.gpu.copy_(poisoned)
    graph.replay()
    torch.npu.synchronize()
    observed = cache[0].cpu().permute(0, 2, 1, 3).reshape(-1, 128)[3000]
    torch.testing.assert_close(observed, torch.full_like(observed, 9.0), rtol=0, atol=0)

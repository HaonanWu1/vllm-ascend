# SPDX-License-Identifier: Apache-2.0
"""T05: real BlockPool/managers, plus ContextVar lifetime and isolation."""

from contextvars import Context, copy_context

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.patch.platform.dflash_kv_context import dflash_scheduler_init_scope, resolve_kv_use_eagle
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator


def test_common_prefix_hit_uses_real_block_pool_and_trims_every_full_group():
    """T05-01：真实多FullAttention manager共同前缀，cold→hit→partial→miss。

    输入：block128/64两个group，hash64，同一512-token请求分别cache512和256 token。
    输出：cold=0，缓存后共同hit=256且块数[2,4]；第150 token变化后hit128、块数[1,2]；
      首token变化则两组都空。被裁掉的第一组后256 token不能作为公共命中返回。
    依据：真实Request哈希、BlockPool分配/cache、coordinator完整find入口；无mock。
    场景：DFlash不应用EAGLE tail-drop；各层block大小不同不能只裁第一个FullAttention组。
    局限：host KV目录合同，不代表NPU KV内容或Mamba checkpoint数值已经验证。
    """
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"layer{index}"],
                kv_cache_spec=FullAttentionSpec(block_size=size, num_kv_heads=1, head_size=16, dtype=torch.float16),
            )
            for index, size in enumerate([128, 64])
        ],
    )
    coordinator = AscendHybridKVCacheCoordinator(
        config,
        max_model_len=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=64,
        max_num_batched_tokens=512,
        scheduler_block_size=128,
    )

    def request(name, tokens):
        params = SamplingParams(max_tokens=1)
        params.update_from_generation_config({}, eos_token_id=999)
        return Request(
            request_id=name,
            prompt_token_ids=tokens,
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(64, sha256),
        )

    original = request("original", list(range(512)))
    blocks, count = coordinator.find_longest_cache_hit(original.block_hashes, 512)
    assert count == 0 and [len(group) for group in blocks] == [0, 0]
    for manager, count in zip(coordinator.single_type_managers, [512, 256]):
        manager.allocate_new_blocks(original.request_id, num_tokens=count, num_tokens_main_model=count)
        manager.cache_blocks(original, num_tokens=count)
    hit, count = coordinator.find_longest_cache_hit(original.block_hashes, 512)
    assert count == 256 and [len(group) for group in hit] == [2, 4]
    assert coordinator.num_uncached_common_prefix_tokens == 256
    for index, size in [(150, 128), (0, 0)]:
        tokens = list(range(512))
        tokens[index] = 99999
        changed = request(f"changed-{index}", tokens)
        blocks, count = coordinator.find_longest_cache_hit(changed.block_hashes, 512)
        assert count == size and [len(group) for group in blocks] == [size // 128, size // 64]
    # A miss must not corrupt the prior cached request or release its live blocks.
    again, count = coordinator.find_longest_cache_hit(original.block_hashes, 512)
    assert count == 256
    assert [[block.block_id for block in group] for group in again] == [
        [block.block_id for block in group] for group in hit
    ]


def test_dflash_context_nested_exception_and_independent_context():
    """T05-02：DFlash KV策略scope嵌套、异常退出及上下文隔离。

    输入：外层/内层真实ContextVar scope、内层主动抛RuntimeError、新Context与copy_context。
    输出：scope内EAGLE标志关闭；内层异常后仍维持外层关闭；全退出后原EAGLE=True恢复；
      独立Context保持True，复制当前Context保持False，传入False始终False。
    场景：一次DFlash初始化失败不得污染下一次普通EAGLE初始化；无mock、无线程时序依赖。
    """
    assert resolve_kv_use_eagle(True)
    with dflash_scheduler_init_scope():
        assert not resolve_kv_use_eagle(True)
        with pytest.raises(RuntimeError, match="test failure"), dflash_scheduler_init_scope():
            assert not resolve_kv_use_eagle(False)
            raise RuntimeError("test failure")
        assert not resolve_kv_use_eagle(True)
        assert Context().run(resolve_kv_use_eagle, True)
        assert not copy_context().run(resolve_kv_use_eagle, True)
    assert resolve_kv_use_eagle(True)
    assert not resolve_kv_use_eagle(False)

"""Behavior UTs for 310P DFlash proposer validation and cache-layout helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p.spec_decode import dflash_proposer_310 as proposer_module


def test_dflash_speculative_width_accepts_compiled_limit_and_rejects_overflow():
    """
    测试功能：验证 DFlash K 上限校验只拒绝超过 310P recurrent buffer 的 query width。
    对应修改：180cf50dd、c6ff47548；源码 vllm_ascend/_310p/spec_decode/dflash_proposer_310.py。
    输入：K=None、K=15 和 K=16；CPU 配置值，不启动服务、不调用设备 kernel。
    场景：K=15 对应 target verify 的 16 token buffer，K=16 会多出一个 token。
    输出：None/K=15 正常返回；K=16 抛出 ValueError，消息包含最大值和 bonus token 说明。
    预期：校验边界与实现的 16-token recurrent capacity 一致；失败表示合法 K=15
        会被误拒绝，或非法 K 超过 kernel capacity 后才在设备侧崩溃。
    """
    proposer_module._validate_num_spec_tokens_310(None)
    proposer_module._validate_num_spec_tokens_310(15)

    with pytest.raises(ValueError, match=r"at most 15 speculative tokens"):
        proposer_module._validate_num_spec_tokens_310(16)


def test_dflash_index_fill_empty_and_signed_discard_indices_preserve_real_values():
    """
    测试功能：验证 310P DFlash 的无 Add index-fill 对空 discard 与负索引都遵守 Python 索引语义。
    对应修改：180cf50dd、c6ff47548；源码 vllm_ascend/_310p/spec_decode/dflash_proposer_310.py。
    输入：CPU int64 tensor shape [3,4]；空 indices，以及 indices=[0,-1]；K=7 不影响该辅助函数。
    场景：没有 request 被丢弃时保持原 tensor；有索引时只写首尾两行，中间 request 不变。
    输出：空输入返回同一对象；有索引时第 0、-1 行全为 99，第 1 行保持原值。
    预期：不触发动态 int64 Add，不误清除中间请求；失败表示 accepted-token discard 会污染
        其他请求的 sampled token。
    """
    tensor = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    empty_result = proposer_module._index_fill_without_add_310p_dflash(tensor, 0, torch.empty(0, dtype=torch.int64), 99)
    assert empty_result is tensor
    assert tensor.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    output = proposer_module._index_fill_without_add_310p_dflash(
        tensor.clone(), 0, torch.tensor([0, -1], dtype=torch.int64), 99
    )
    assert output.tolist() == [[99, 99, 99, 99], [4, 5, 6, 7], [99, 99, 99, 99]]


@pytest.mark.parametrize(
    ("source", "counts", "physical", "source_size", "target_size", "expected"),
    [
        (np.array([[0, 1, 2, 3]], dtype=np.int32), np.array([4]), 128, 64, 128, np.array([[0, 1]], dtype=np.int32)),
        (np.array([[0, 1]], dtype=np.int32), np.array([2]), 128, 128, 64, np.array([[0, 1, 2, 3]], dtype=np.int32)),
    ],
)
def test_dflash_block_table_conversion_maps_complete_physical_pages(
    source, counts, physical, source_size, target_size, expected
):
    """
    测试功能：验证不同 kernel block size 的合法 physical page 会按 page id 展开/收缩。
    对应修改：d82690319、c6ff47548；源码 dflash_proposer_310.py::_convert_block_table_layout_310。
    输入：CPU NumPy 连续 logical block table，physical=128，source/target block size 为64/128或反向。
    场景：完整物理页跨越 source logical blocks，转换不得重排 page id 或填入零页。
    输出：精确的 target logical table，并保持 int32 二维布局。
    预期：结果等于独立 page-id 展开规则；失败表示 draft cache 会访问错误的 KV page。
    """
    actual = proposer_module._convert_block_table_layout_310(source, counts, physical, source_size, target_size)
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual[:, : expected.shape[1]], expected)


def test_dflash_slot_mapping_int32_address_math_matches_wide_math():
    """
    测试功能：验证 DFlash physical slot mapping 在 310P 安全 int32 算术和宽算术下结果一致。
    对应修改：7f1abe8cd、d82690319、c6ff47548；源码 dflash_proposer_310.py。
    输入：CPU int32 positions=[0,63,64,127]、request_ids=[0,0,1,1]、block_table shape [2,2]、block_size=64。
    场景：跨 64-token page 边界，分别执行 use_int32_math=False/True；目标/草稿 cache ownership 相同。
    输出：两组 int32 slot ids [320,383,576,639]，shape [4] 且 contiguous。
    预期：地址算术优化只改变中间 dtype，不改变物理 slot；失败表示 FDO 图路径可能把
        token 写入错误 KV page。
    """
    positions = torch.tensor([0, 63, 64, 127], dtype=torch.int32)
    request_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    block_table = torch.tensor([[5, 6], [8, 9]], dtype=torch.int32)

    wide = proposer_module._compute_slots_for_block_size_310(
        positions, request_ids, block_table, 64, use_int32_math=False
    )
    narrow = proposer_module._compute_slots_for_block_size_310(
        positions, request_ids, block_table, 64, use_int32_math=True
    )

    assert wide.dtype == narrow.dtype == torch.int32
    assert wide.is_contiguous() and narrow.is_contiguous()
    assert wide.tolist() == narrow.tolist() == [320, 383, 576, 639]


@pytest.mark.parametrize(
    ("source", "counts", "physical", "source_size", "target_size", "message"),
    [
        (np.zeros((2, 4, 1), dtype=np.int32), np.array([4, 4]), 128, 64, 128, "two-dimensional"),
        (np.zeros((1, 4), dtype=np.int32), np.array([4]), 0, 64, 128, "positive"),
        (np.zeros((1, 3), dtype=np.int32), np.array([3]), 128, 64, 128, "whole physical pages"),
        (np.array([[0, 2, 3, 4]], dtype=np.int32), np.array([4]), 128, 64, 128, "contiguous logical blocks"),
    ],
)
def test_dflash_block_table_conversion_rejects_incomplete_physical_layout(
    source, counts, physical, source_size, target_size, message
):
    """
    测试功能：验证不同 kernel block size 的 block-table 转换拒绝不可解释的物理 page 布局。
    对应修改：d82690319、c6ff47548；源码 dflash_proposer_310.py。
    输入：CPU NumPy source、每行 logical block count、physical/source/target block sizes，覆盖维度、
        正数、整页和连续性四类非法输入。
    场景：DFlash 混合 attention group 需要将 source page 重新展开到 target layout，任何 page 破损都不能猜测。
    输出：ValueError，消息包含具体契约字段（{message}）。
    预期：转换在生成新 table 前失败且不返回部分结果；失败表示不同 block size 的 draft cache
        会读到错误 page，接受率下降且难以从 Python 日志定位。
    """
    with pytest.raises(ValueError, match=message):
        proposer_module._convert_block_table_layout_310(source, counts, physical, source_size, target_size)


def test_dflash_cache_block_size_reader_unwraps_nested_per_layer_cache(monkeypatch):
    """
    测试功能：验证 proposer 能从每层 list/tuple 嵌套的 KV cache 读取真实 physical block size。
    对应修改：d82690319、c6ff47548；源码 dflash_proposer_310.py。
    输入：CPU cache tensor shape [1,2,64,8] 与 [1,2,128,8]，attn_layer_names=[layer0,layer1]。
    场景：Qwen 混合 FullAttention group 每层 block size 不同，cache 可能按 virtual engine 和 K/V tuple 嵌套。
    输出：{layer0:64, layer1:128}，首层 helper 返回 64；无 layer names 返回空映射。
    预期：读取的是 cache.shape[-2] 而不是全局默认值；失败表示后续 slot mapping 会访问空 page。
    """
    cache_layers = {
        "layer0": SimpleNamespace(kv_cache=[(torch.empty(1, 2, 64, 8), torch.empty(1, 2, 64, 8))]),
        "layer1": SimpleNamespace(kv_cache=torch.empty(1, 2, 128, 8)),
    }
    proposer = SimpleNamespace(
        attn_layer_names=["layer0", "layer1"],
        vllm_config=SimpleNamespace(),
    )

    import vllm.config

    monkeypatch.setattr(
        vllm.config,
        "get_layers_from_vllm_config",
        lambda _config, _cls: cache_layers,
    )
    sizes = proposer_module._draft_cache_block_sizes_310(proposer)
    first = proposer_module._draft_cache_block_size_310(proposer)

    assert sizes == {"layer0": 64, "layer1": 128}
    assert first == 64
    assert proposer_module._draft_cache_block_sizes_310(SimpleNamespace(attn_layer_names=[])) == {}

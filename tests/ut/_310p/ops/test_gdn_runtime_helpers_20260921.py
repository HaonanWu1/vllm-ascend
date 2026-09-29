"""Behavior UTs for 310P GDN runtime metadata helpers."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.ops.fla import gdn_310


def test_gdn_zero_padded_tokens_keeps_only_logical_prefix():
    """
    测试功能：验证 GDN graph batch 只保留 logical token prefix，物理 padding 被置零。
    对应修改：92dfbff2a、c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py。
    输入：CPU float16 tensor shape [2, 4, 3]，token_dim=1，valid_tokens=2。
    场景：FDO FULL descriptor 有 4 个物理 token，但本轮只有 2 个有效 token。
    输出：shape 不变；第 0、1 个 token 保持原值，第 2、3 个 token 为零。
    预期：不会清除合法前缀，也不会让 padding 进入 recurrent attention；失败表示图重放
        会把旧请求数据当作当前请求的 token。
    """
    tensor = torch.arange(24, dtype=torch.float16).reshape(2, 4, 3)
    expected = tensor.clone()
    expected[:, 2:] = 0

    output = gdn_310._zero_padded_tokens(tensor, torch.tensor(2), token_dim=1)

    torch.testing.assert_close(output, expected)


def test_gdn_clear_states_uses_where_to_remove_nan_stale_rows():
    """
    测试功能：验证没有 initial state 的 recurrent rows 被无条件清零，包括 NaN/Inf stale cache。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/ops/fla/gdn_310.py。
    输入：CPU float32 state shape [2, 2, 3]，第 0 行有效，第 1 行包含 NaN/Inf；mask=[True,False]。
    场景：混合 prefill/spec batch 中，dummy request 没有初始 SSM state。
    输出：第 0 行保持原值，第 1 行全零且没有 NaN/Inf。
    预期：不是乘零（乘零会传播 NaN），而是 torch.where 语义；失败表示 stale cache
        可能污染下一轮 GDN 输出。
    """
    states = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[float("nan"), float("inf"), -3.0], [2.0, 4.0, 8.0]]],
        dtype=torch.float32,
    )

    output = gdn_310._clear_states_without_initial(states, torch.tensor([True, False]))

    torch.testing.assert_close(output[0], states[0])
    assert torch.count_nonzero(output[1]) == 0
    assert torch.isfinite(output).all()


def test_gdn_flatten_uniform_state_indices_preserves_graph_order(monkeypatch):
    """
    测试功能：验证 uniform spec decode 将 [num_reqs, q_per_req] state indices 按 graph 顺序展平。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py。
    输入：CPU int64 indices shape [2, 3]、cu_seqlens=[0,3,6]、total_tokens=6；capture=False。
    场景：K=2 对应 query width=3，两个 request 都完整填满 FULL descriptor。
    输出：contiguous int32 tensor [7,8,9,10,11,12]。
    预期：不走 masked_select，不改变 request 顺序和 token 数；失败表示 state index 与
        accepted-token correction 错配。
    """
    monkeypatch.setattr(gdn_310, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    indices = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int64)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

    output = gdn_310._flatten_state_indices(indices, cu_seqlens, total_tokens=6)

    assert output.dtype == torch.int32
    assert output.is_contiguous()
    assert output.tolist() == [7, 8, 9, 10, 11, 12]


def test_gdn_mask_padded_accepted_tokens_zeroes_dummy_requests():
    """
    测试功能：验证 padded/dummy request 的 accepted token count 被清零并保持 int32 contiguous。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py。
    输入：CPU int64 accepted=[3,2,9]，actual_seq_lengths=[4,0]，只取有效 request 数 2。
    场景：FULL graph 的第二个物理 request 没有实际 sequence，不能沿用旧 accepted count。
    输出：int32 contiguous [3,0]。
    预期：有效请求保留 accepted=3，dummy 请求强制为 0；失败表示状态推进会超出
        当前请求的 logical length。
    """
    accepted = torch.tensor([3, 2, 9], dtype=torch.int64)
    actual_lengths = torch.tensor([4, 0], dtype=torch.int32)

    output = gdn_310._mask_padded_recurrent_accepted_tokens(accepted, actual_lengths)

    assert output.dtype == torch.int32
    assert output.is_contiguous()
    assert output.tolist() == [3, 0]


def test_gdn_merge_spec_and_non_spec_outputs_checks_lengths_and_layout():
    """
    测试功能：验证混合 spec/non-spec GDN 输出按原 token indices 写回，并拒绝长度不一致。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py。
    输入：CPU float32 core output shape [4,2]，spec indices=[0,2]、non-spec=[1,3]，两路输出各 [1,2,2]。
    场景：同一 batch 同时包含 draft verification 与普通 prefill/decode token。
    输出：core 的四行分别为 spec0、non0、spec1、non1；错误长度触发 RuntimeError。
    预期：token layout 完整且无覆盖；失败表示不同阶段输出会写入错误请求。
    """
    core = torch.zeros(4, 2, dtype=torch.float32)
    spec_indices = torch.tensor([0, 2], dtype=torch.long)
    non_spec_indices = torch.tensor([1, 3], dtype=torch.long)
    spec_out = torch.tensor([[[10.0, 11.0], [20.0, 21.0]]])
    non_spec_out = torch.tensor([[[30.0, 31.0], [40.0, 41.0]]])

    gdn_310._merge_spec_and_non_spec_outputs_310(
        core,
        4,
        spec_indices,
        non_spec_indices,
        spec_out,
        non_spec_out,
    )

    torch.testing.assert_close(core, torch.tensor([[10, 11], [30, 31], [20, 21], [40, 41]], dtype=torch.float32))
    with pytest.raises(RuntimeError, match="spec output length"):
        gdn_310._merge_spec_and_non_spec_outputs_310(
            torch.zeros(4, 2),
            4,
            spec_indices,
            non_spec_indices,
            torch.zeros(1, 1, 2),
            non_spec_out,
        )
    with pytest.raises(RuntimeError, match="non-spec output length"):
        gdn_310._merge_spec_and_non_spec_outputs_310(
            torch.zeros(4, 2),
            4,
            spec_indices,
            non_spec_indices,
            spec_out,
            torch.zeros(1, 1, 2),
        )


def test_gdn_flatten_state_indices_uses_capture_width_for_irregular_runtime_lengths(monkeypatch):
    """
    测试功能：验证 ACL graph capture 时按固定 query width 展平，即使本轮运行长度不均匀。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py::_flatten_state_indices。
    输入：CPU int64 indices=[[7,8,9],[10,11,12]]、cu_seqlens=[0,2,5]、total_tokens=5；capturing=True。
    场景：图描述符固定每请求 3 个 query，但运行批次第一请求只有 2 个真实 token。
    输出：按图宽读取并截断为 [7,8,9,10,11]，不进入 eager 的变长 masked_select。
    预期：capture 分支保持固定地址和顺序；失败表示不均匀请求会改变图输入布局。
    """
    monkeypatch.setattr(gdn_310, "_EXTRA_CTX", SimpleNamespace(capturing=True))
    indices = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int64)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    output = gdn_310._flatten_state_indices(indices, cu_seqlens, total_tokens=5)

    assert output.tolist() == [7, 8, 9, 10, 11]


def test_gdn_flatten_state_indices_keeps_one_dimensional_fast_path(monkeypatch):
    """
    测试功能：验证已经压平的一维 state index 只截断、转换 int32，不重复解释 request metadata。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py::_flatten_state_indices。
    输入：CPU int64 indices=[7,8,9,10]、任意 cu_seqlens=[0,2,4]、total_tokens=3；capturing=False。
    场景：上游已经按实际 token 生成一维索引，不应进入二维 uniform/eager 分支。
    输出：contiguous int32 [7,8,9]。
    预期：只发生 dtype/长度规范化；失败表示合法的一维 metadata 会被错误重排。
    """
    monkeypatch.setattr(gdn_310, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    output = gdn_310._flatten_state_indices(
        torch.tensor([7, 8, 9, 10], dtype=torch.int64),
        torch.tensor([0, 2, 4], dtype=torch.int32),
        total_tokens=3,
    )
    assert output.dtype == torch.int32
    assert output.is_contiguous()
    assert output.tolist() == [7, 8, 9]


def test_gdn_flatten_state_indices_compacts_variable_lengths_in_eager(monkeypatch):
    """
    测试功能：验证非捕获的变长 batch 只保留每个请求真实 query length 的 state index。
    对应修改：c6ff47548；源码 vllm_ascend/_310p/ops/fla/gdn_310.py::_flatten_state_indices。
    输入：CPU int64 indices=[[7,8,9],[10,11,12]]、cu_seqlens=[0,2,5]、total_tokens=5；capturing=False。
    场景：第一请求2 token、第二请求3 token，二维 padding 中的 9 不属于第一请求有效输出。
    输出：pinned/contiguous int32 [7,8,10,11,12]。
    预期：eager masked_select 按真实长度压紧顺序；失败表示 padding state 会推进错误请求。
    """
    monkeypatch.setattr(gdn_310, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    output = gdn_310._flatten_state_indices(
        torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int64),
        torch.tensor([0, 2, 5], dtype=torch.int32),
        total_tokens=5,
    )
    assert output.dtype == torch.int32
    assert output.is_contiguous()
    assert output.tolist() == [7, 8, 10, 11, 12]


def test_gdn_rearrange_mixed_qkv_materializes_contiguous_head_views():
    """
    测试功能：验证 310P GDN 将 mixed QKV 拆成真实 contiguous 的 [1,T,H,D] query/key/value。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/ops/fla/gdn_310.py。
    输入：CPU float16 mixed shape [3,12]，key_dim=value_dim=4、tp=1、head_dim=2，来自真实线性投影布局。
    场景：普通/非 spec 读取路径，要求三份 materialization 不依赖第二次 concat。
    输出：三个 tensor shape [1,3,2,2]，数值拼接顺序为 q、k、v，均 contiguous。
    预期：拼接后的 q/k/v 恢复原始列区间；失败表示 head fold 或 value head topology 错位。
    """
    mixed = torch.arange(36, dtype=torch.float16).reshape(3, 12)
    layer = SimpleNamespace(key_dim=4, value_dim=4, tp_size=1, head_k_dim=2, head_v_dim=2)

    query, key, value = gdn_310._rearrange_mixed_qkv_310(layer, mixed)

    assert query.shape == key.shape == value.shape == (1, 3, 2, 2)
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    torch.testing.assert_close(query.reshape(3, 4), mixed[:, :4])
    torch.testing.assert_close(key.reshape(3, 4), mixed[:, 4:8])
    torch.testing.assert_close(value.reshape(3, 4), mixed[:, 8:])


def test_gdn_rearrange_mixed_qkv_handles_tp_and_gqa_head_topology():
    """
    测试功能：验证 mixed QKV 在 TP>1 且 query/key 与 value head 数不同的 GQA 拆分。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/ops/fla/gdn_310.py::_rearrange_mixed_qkv_310。
    输入：CPU float16 mixed shape [3,10]，key_dim=8、value_dim=4、tp=2，head_k_dim=head_v_dim=2。
    场景：TP 分片后 q/k 各4列、v为2列，query/key 有2 heads 而 value 只有1 head。
    输出：q/k shape [1,3,2,2]，v shape [1,3,1,2]，三者 contiguous 且列值不交叉。
    预期：真实 GQA topology 和 TP 分片保持；失败表示 value head 被错误复制或读取错列。
    """
    mixed = torch.arange(30, dtype=torch.float16).reshape(3, 10)
    layer = SimpleNamespace(key_dim=8, value_dim=4, tp_size=2, head_k_dim=2, head_v_dim=2)

    query, key, value = gdn_310._rearrange_mixed_qkv_310(layer, mixed)

    assert query.shape == key.shape == (1, 3, 2, 2)
    assert value.shape == (1, 3, 1, 2)
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    torch.testing.assert_close(query.reshape(3, 4), mixed[:, :4])
    torch.testing.assert_close(key.reshape(3, 4), mixed[:, 4:8])
    torch.testing.assert_close(value.reshape(3, 2), mixed[:, 8:])

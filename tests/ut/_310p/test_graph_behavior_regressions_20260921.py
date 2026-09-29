"""FP-05/07/08/09/10: real graph-contract and runtime metadata behavior."""

from dataclasses import replace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend._310p.attention.dflash_hybrid_draft_graph_safe_attention import (
    build_dflash_hybrid_draft_paged_view_310,
    create_dflash_hybrid_draft_attention_inputs_310,
    update_dflash_hybrid_draft_attention_inputs_310,
)
from vllm_ascend._310p.dflash_full_and_piecewise import (
    build_dflash_draft_forward_contract,
    build_dflash_hybrid_route_observation,
    classify_dflash_hybrid_route,
)
from vllm_ascend._310p.graph_input_contract import (
    GraphInputContractError,
    GraphInputSource,
    capture_graph_input_contracts,
    capture_graph_input_sources,
    validate_graph_input_contracts,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState


@pytest.mark.parametrize(
    "query_lens,token_capacity,request_capacity,reason",
    [
        ([3, 5], 8, 2, None),
        ([3, 5], 7, 2, "draft_token_capacity_exceeded"),
        ([3, 5], 8, 1, "draft_request_capacity_exceeded"),
        ([3, 5], 7, 1, "draft_request_capacity_exceeded"),
    ],
)
def test_draft_contract_preserves_ragged_requests_on_capacity_failure(
    query_lens, token_capacity, request_capacity, reason
):
    """功能 FP-05/08：Draft logical 与 physical 容量独立；对应 3bafe5f、6f40366。
    源码：_310p/dflash_full_and_piecewise.py::build_dflash_draft_forward_contract。
    输入：CPU host 列表 [3,5]，token capacity=7/8，request capacity=1/2；K 不参与此合同。
    场景：刚好容纳、token 超一、request 超一、两者都超；无 mock。
    输出：完整 query lengths/starts、logical counts、fallback reason、padding。
    预期：不截断请求，双超优先报 request；失败表示不安全的图可被标为 eligible。
    """
    before = query_lens.copy()
    result = build_dflash_draft_forward_contract(
        logical_query_lens=query_lens,
        physical_token_capacity=token_capacity,
        physical_request_capacity=request_capacity,
    )
    assert (result.logical_num_reqs, result.logical_num_tokens) == (2, 8)
    assert result.query_lens == (3, 5)
    assert result.query_start_loc == (0, 3, 8)
    assert (result.physical_token_capacity, result.physical_request_capacity) == (token_capacity, request_capacity)
    assert (result.padding_request_count, result.padding_token_count) == (0, 0)
    assert result.graph_eligible == (reason is None)
    assert result.fallback_reason == reason
    assert query_lens == before


@pytest.mark.parametrize(
    "lengths,tokens,requests,message",
    [
        ([], 8, 2, "at least one"),
        ([0, 8], 8, 2, "positive"),
        ([-1, 9], 8, 2, "positive"),
        ([8], 0, 1, "token capacity"),
        ([8], 8, 0, "request capacity"),
    ],
)
def test_draft_contract_rejects_invalid_logical_input(lengths, tokens, requests, message):
    """功能 FP-05：拒绝未定义的 Draft 图输入；对应 3bafe5f，源码 dflash_full_and_piecewise.py。
    输入：CPU 列表，空请求/0或负query_len/0容量，具体非法值列在参数表；无 tensor、无 mock。
    场景：构造合同失败，不能返回半有效对象；K/cache 不适用。
    输出：具体 ValueError 并保持输入不变。
    预期：错误指向缺失请求、长度或容量；失效会允许零容量或负长度进入捕获。
    """
    before = lengths.copy()
    with pytest.raises(ValueError, match=message):
        build_dflash_draft_forward_contract(
            logical_query_lens=lengths, physical_token_capacity=tokens, physical_request_capacity=requests
        )
    assert lengths == before


def test_route_observation_reports_exact_padding_and_full_fallback():
    """功能 FP-08：FULL 候选回落 PW 的实际容量、原因与 padding；对应 3bafe5f、6f40366。
    源码：dflash_full_and_piecewise.py::classify_dflash_hybrid_route/build_dflash_hybrid_route_observation。
    输入：K=7，3请求各8 token，总24；PW descriptor=40、max_reqs=10；CPU host metadata，无 mock。
    场景：运行时缺 FULL，dispatcher 选择 PW；检查合同与可观测字段一致，非日志断言。
    输出：required=24，物理请求10/token40，padding请求7/token16，比率0.4。
    预期：reason=full_descriptor_unavailable，无容量违约；失败会误报路由或 padding。
    """
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.SpecDecoding,
        num_reqs=3,
        num_tokens=24,
        num_scheduled_tokens=[8, 8, 8],
        all_decode=True,
        num_speculative_tokens=7,
    )
    descriptor = BatchDescriptor(num_tokens=40, num_reqs=None, uniform=False)
    observation = build_dflash_hybrid_route_observation(
        configured_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        effective_mode=CUDAGraphMode.PIECEWISE,
        decision=decision,
        descriptor=descriptor,
        max_num_reqs=10,
        max_capture_tokens=40,
    )
    assert observation.candidate_mode is CUDAGraphMode.FULL
    assert observation.selected_mode is observation.effective_mode is CUDAGraphMode.PIECEWISE
    assert observation.descriptor == descriptor
    assert (observation.active_num_reqs, observation.real_num_tokens) == (3, 24)
    assert (observation.num_speculative_tokens, observation.verification_width, observation.required_tokens) == (
        7,
        8,
        24,
    )
    assert (observation.physical_request_capacity, observation.physical_token_capacity) == (10, 40)
    assert (observation.padding_request_count, observation.padding_token_count) == (7, 16)
    assert observation.padding_ratio == 0.4
    assert observation.fallback_reason == "full_descriptor_unavailable"
    assert observation.contract_mismatch_reason is None


def test_graph_contract_handles_cycles_empty_views_and_unusual_keys():
    """功能 FP-09：嵌套容器循环与空 view 不误判；对应 92dfbff2a，源码 graph_input_contract.py。
    输入：CPU float32 base[8]、offset=3 的空 view，字典key='not-valid'并自引用。
    场景：同一容器重复进入 args/kwargs，防止无限遍历；真实 PyTorch storage，无 mock。
    输出：仅一条合同，空字节区间[12,12)，shape=(0,)，stride=(1,)，CPU/float32。
    预期：不丢合法空view，不循环，不复制tensor；失败会破坏图输入inventory或卡住capture。
    """
    base = torch.arange(8, dtype=torch.float32)
    view = base[3:3]
    cyclic = {"not-valid": view}
    cyclic["self"] = cyclic
    contracts = capture_graph_input_contracts((cyclic,), {"alias": cyclic})
    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.path == "args[0]['not-valid']"
    assert (contract.shape, contract.stride, contract.storage_offset) == ((0,), (1,), 3)
    assert (contract.view_start_byte, contract.view_end_byte, contract.storage_nbytes) == (12, 12, 32)
    assert (contract.dtype, contract.device) == ("torch.float32", "cpu")
    assert contract.base_ptr == base.untyped_storage().data_ptr()
    validate_graph_input_contracts(contracts, capture_graph_input_contracts((cyclic,), {"alias": cyclic}))


@pytest.mark.parametrize("reuse_tensor", [False, True])
@pytest.mark.parametrize(
    "field,value,message",
    [
        ("ownership", "", "ownership"),
        ("alignment_source", "", "alignment source"),
        ("required_alignment", 0, "alignment must be positive"),
        ("role", "", "semantic role"),
    ],
)
def test_graph_source_policies_validated_for_first_and_shared_tensor(reuse_tensor, field, value, message):
    """功能 FP-09：首次/复用tensor都验证声明；对应92dfbff2a，源码graph_input_contract.py。
    输入：CPU int32[4]，owner='runner'、alignment=4；参数表将一个声明设为空或0。
    场景：同一tensor第二语义角色不能绕过ownership/alignment校验；无mock，真实合同捕获。
    输出：GraphInputContractError，字段名明确；原tensor仍[0,1,2,3]。
    预期：元数据复用只减少采样，不忽略角色约束；失败会允许错误owner和对齐要求参与replay。
    """
    tensor = torch.arange(4, dtype=torch.int32)
    source = GraphInputSource("first", tensor, "runner", 4, "element-size", True, True)
    invalid = replace(source, role="second", **({field: value} if field != "role" else {}))
    if field == "role":
        invalid = replace(invalid, role=value)
    sources = (source, invalid) if reuse_tensor else (invalid,)
    with pytest.raises(GraphInputContractError, match=message):
        capture_graph_input_sources(sources)
    assert tensor.tolist() == [0, 1, 2, 3]


def test_graph_contract_rejects_real_misaligned_view_even_when_unchanged():
    """功能 FP-09：相等但实际不对齐的view必须拒绝；对应92dfbff2a，源码graph_input_contract.py。
    输入：CPU float32[8]偏移一个元素的view[4]，required_alignment=16，base满足16字节对齐。
    场景：合同对象由真实storage生成，没有伪造data_ptr或alignment_ok，无mock。
    输出：capture alignment_ok=False；validate同一合同仍抛GraphInputContractError。
    预期：不能仅凭dataclass相等放行；失败会把非16字节对齐地址送入NPU kernel。
    """
    base = torch.arange(16, dtype=torch.float32)
    view = next(base[offset : offset + 4] for offset in range(4) if base[offset : offset + 4].data_ptr() % 16)
    source = GraphInputSource("draft.query", view, "draft", 16, "NPU-ABI", True, True)
    contracts = capture_graph_input_sources((source,))
    assert contracts[0].alignment_ok is False
    with pytest.raises(GraphInputContractError, match="not aligned to 16 bytes"):
        validate_graph_input_contracts(contracts, contracts)


def test_graph_source_contract_accepts_valid_owned_roles_and_replays_unchanged():
    """
    测试功能：验证 graph input source 的合法 ownership、role、alignment 元数据可被捕获并复用。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/graph_input_contract.py。
    输入：CPU int32 query=[0,1,2,3] 与独立 block_table=[4,5]，均由 runner 持有、4字节对齐。
    场景：首次 capture 后以同一 tensor 身份和语义角色 replay；不替换被测合同函数。
    输出：两份 contract 的 role/path、shape、stride、ownership、alignment 均相等。
    预期：合法共享 storage 不被误拒绝；失败表示正常 graph replay 会被错误降级到 eager。
    """
    query = torch.arange(4, dtype=torch.int32)
    block_table = torch.tensor([4, 5], dtype=torch.int32)
    sources = (
        GraphInputSource("draft.query", query, "runner", 4, "tensor-element-size", True, True),
        GraphInputSource("draft.block_table", block_table, "runner", 4, "tensor-element-size", True, True),
    )

    expected = capture_graph_input_sources(sources)
    actual = capture_graph_input_sources(sources)
    validate_graph_input_contracts(expected, actual)
    assert [contract.path for contract in expected] == ["draft.query", "draft.block_table"]
    assert all(contract.alignment_ok and contract.ownership == "runner" for contract in expected)


def test_graph_source_contract_rejects_duplicate_semantic_role():
    """
    测试功能：验证两个 provider 不能声明同一个 graph input semantic role。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/graph_input_contract.py::capture_graph_input_sources。
    输入：两个 CPU int32 tensor，均声明 role='draft.query'，owner/alignment 元数据合法。
    场景：不同 storage 也不能用同名语义覆盖首个 provider；不修改任何 tensor 数据。
    输出：GraphInputContractError，消息包含 duplicate semantic role。
    预期：role 冲突在 capture 阶段被拒绝；失败表示 replay 时无法确定实际 query owner。
    """
    sources = (
        GraphInputSource("draft.query", torch.arange(4, dtype=torch.int32), "runner", 4, "abi", True, True),
        GraphInputSource("draft.query", torch.arange(4, dtype=torch.int32), "draft", 4, "abi", True, True),
    )
    with pytest.raises(GraphInputContractError, match="duplicate graph input semantic role"):
        capture_graph_input_sources(sources)


@pytest.mark.parametrize("change", ["pointer", "shape"])
def test_graph_contract_rejects_real_replay_tensor_change(change):
    """
    测试功能：验证真实 replay tensor 的 data_ptr 或 shape 变化会触发具体合同错误。
    对应修改：92dfbff2a；源码 vllm_ascend/_310p/graph_input_contract.py::validate_graph_input_contracts。
    输入：首次 CPU int32 query[4]；pointer 场景使用新 allocation，shape 场景使用同 storage 的 query[:3]。
    场景：保持同一 semantic role/owner，但 graph replay 的 tensor identity 或 view shape 改变。
    输出：GraphInputContractError，分别指出 data_ptr 或 view_end_byte（shape view 的边界变化）；原始 capture 不被修改。
    预期：合同按真实 pointer/shape 拒绝重放；失败表示图会读错内存范围或越界。
    """
    original = torch.arange(4, dtype=torch.int32)
    expected = capture_graph_input_sources(
        (GraphInputSource("draft.query", original, "runner", 4, "tensor-element-size", True, True),)
    )
    changed = torch.arange(4, dtype=torch.int32) if change == "pointer" else original[:3]
    actual = capture_graph_input_sources(
        (GraphInputSource("draft.query", changed, "runner", 4, "tensor-element-size", True, True),)
    )
    message = "data_ptr" if change == "pointer" else "view_end_byte"
    with pytest.raises(GraphInputContractError, match=message):
        validate_graph_input_contracts(expected, actual)


def test_hybrid_metadata_refresh_removes_previous_request_at_stable_addresses():
    """功能 FP-05/07/10：descriptor复用时清旧请求；对应3bafe5f、92dfbff2a。
    源码：attention/dflash_hybrid_draft_graph_safe_attention.py 的create/update/build_paged_view。
    输入：CPU int32，capacity requests=3/tokens=8/blocks=2；先[2,3]请求再[1]请求。
    场景：第二轮batch收缩，旧seq_lens/block_table不能残留，捕获地址保持不变；无mock。
    输出：新valid mask=[True,False*7]、context=[20,1*7]，尾部请求/表清零。
    预期：所有buffer pointer不变、真实请求映射正确；失败表示图读取旧请求cache。
    """
    inputs = create_dflash_hybrid_draft_attention_inputs_310(
        capacity_reqs=3, capacity_tokens=8, max_blocks=2, device=torch.device("cpu")
    )
    pointers = {name: value.data_ptr() for name, value in vars(inputs).items() if isinstance(value, torch.Tensor)}
    update_dflash_hybrid_draft_attention_inputs_310(
        inputs,
        query_lens=torch.tensor([2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([10, 15], dtype=torch.int32),
        block_table=torch.tensor([[4, 5], [8, 9]], dtype=torch.int32),
        valid_num_reqs=2,
        valid_num_tokens=5,
    )
    view = build_dflash_hybrid_draft_paged_view_310(inputs)
    assert view.request_ids.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert view.context_lens.tolist() == [10, 10, 15, 15, 15, 1, 1, 1]
    update_dflash_hybrid_draft_attention_inputs_310(
        inputs,
        query_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([20], dtype=torch.int32),
        block_table=torch.tensor([[12, 13]], dtype=torch.int32),
        valid_num_reqs=1,
        valid_num_tokens=1,
    )
    view = build_dflash_hybrid_draft_paged_view_310(inputs)
    assert inputs.query_lens.tolist() == [1, 0, 0]
    assert inputs.seq_lens.tolist() == [20, 0, 0]
    assert inputs.block_table.tolist() == [[12, 13], [0, 0], [0, 0]]
    assert view.valid_token_mask.tolist() == [True, False, False, False, False, False, False, False]
    assert view.context_lens.tolist() == [20, 1, 1, 1, 1, 1, 1, 1]
    assert pointers == {
        name: value.data_ptr() for name, value in vars(inputs).items() if isinstance(value, torch.Tensor)
    }


@pytest.mark.parametrize("invalid", ["query_dtype", "seq_shape", "block_width", "token_overflow", "request_overflow"])
def test_hybrid_metadata_rejection_is_atomic(invalid):
    """功能 FP-05/10：非法runtime metadata不得部分覆盖已捕获buffer；对应3bafe5f、92dfbff2a。
    源码：attention/dflash_hybrid_draft_graph_safe_attention.py::update_dflash_hybrid_draft_attention_inputs_310。
    输入：CPU int32 descriptor(2请求/8token/2blocks)，以参数表破坏dtype/shape/容量之一。
    场景：先写合法数据再更新非法数据；真实CPU tensor，无mock。
    输出：TypeError或ValueError；所有persistent buffer数值、shape、dtype保持原样。
    预期：拒绝发生在第一次写入前；失败表示一次坏请求会污染下一次replay。
    """
    inputs = create_dflash_hybrid_draft_attention_inputs_310(
        capacity_reqs=2, capacity_tokens=8, max_blocks=2, device=torch.device("cpu")
    )
    arguments = dict(
        query_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        block_table=torch.tensor([[4, 5]], dtype=torch.int32),
        valid_num_reqs=1,
        valid_num_tokens=2,
    )
    update_dflash_hybrid_draft_attention_inputs_310(inputs, **arguments)
    snapshot = {name: value.clone() for name, value in vars(inputs).items() if isinstance(value, torch.Tensor)}
    failures = {
        "query_dtype": ({"query_lens": arguments["query_lens"].long()}, TypeError, "int32"),
        "seq_shape": ({"seq_lens": torch.empty(0, dtype=torch.int32)}, ValueError, "seq_lens"),
        "block_width": ({"block_table": torch.ones(1, 1, dtype=torch.int32)}, ValueError, "block_table"),
        "token_overflow": ({"valid_num_tokens": 9}, ValueError, "capacity"),
        "request_overflow": ({"valid_num_reqs": 3}, ValueError, "capacity"),
    }
    updates, error, message = failures[invalid]
    with pytest.raises(error, match=message):
        update_dflash_hybrid_draft_attention_inputs_310(inputs, **(arguments | updates))
    for name, expected in snapshot.items():
        torch.testing.assert_close(getattr(inputs, name), expected, rtol=0, atol=0)

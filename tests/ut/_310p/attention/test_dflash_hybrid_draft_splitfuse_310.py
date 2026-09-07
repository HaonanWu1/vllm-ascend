# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend._310p.attention import dflash_hybrid_draft_graph_safe_attention as attention
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310
from vllm_ascend._310p.spec_decode import llm_base_proposer_310 as proposer


def _builder(k=15):
    builder = object.__new__(AscendAttentionMetadataBuilder310)
    builder._vllm_config_310 = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=k)
    )
    return builder


def _common(widths, capacity=160, capacity_reqs=10):
    qsl = torch.tensor([0, *torch.tensor(widths).cumsum(0).tolist()], dtype=torch.int32)
    return SimpleNamespace(
        num_actual_tokens=sum(widths),
        decode_token_per_req=max(widths),
        max_query_len=max(widths),
        num_input_tokens=capacity,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.arange(24, 24 + len(widths), dtype=torch.int32),
        block_table_tensor=torch.arange(capacity_reqs * 4, dtype=torch.int32).reshape(capacity_reqs, 4),
    )


@pytest.mark.parametrize(
    "k,widths,capacity,want",
    [
        (15, [16] * 10, 160, [16] * 10),
        (15, [16] * 6, 160, [16] * 10),
        (7, [8] * 3, 80, [8] * 10),
        (3, [4], 40, [4] * 10),
    ],
)
def test_uniform_draft_uses_descriptor_fixed_host_groups(k, widths, capacity, want):
    inputs = _builder(k)._prepare_dflash_hybrid_draft_attention_inputs_310(_common(widths, capacity), draft_step=0)
    # Fails if grouping is absent, derived from active count, or hardcoded to K15.
    assert getattr(inputs, "uniform_query_width", 0) == k + 1
    assert inputs.splitfuse_query_lens_cpu.device.type == "cpu"
    assert inputs.splitfuse_query_lens_cpu.dtype == torch.int32
    assert inputs.splitfuse_query_lens_cpu.tolist() == want


@pytest.mark.parametrize(
    "widths,capacity,capacity_reqs",
    [
        ([8, 16], 160, 10),
        ([16], 162, 10),
        ([16], 160, 4),
        ([1] * 10, 160, 10),
    ],
)
def test_incompatible_topology_retains_generic_attention(widths, capacity, capacity_reqs):
    inputs = _builder()._prepare_dflash_hybrid_draft_attention_inputs_310(
        _common(widths, capacity, capacity_reqs),
        draft_step=0,
        real_num_reqs_override=len(widths),
    )
    assert getattr(inputs, "uniform_query_width", 0) == 0


def test_request_count_shrink_keeps_graph_inputs_and_host_groups_stable():
    builder = _builder()
    first = builder._prepare_dflash_hybrid_draft_attention_inputs_310(_common([16] * 10), draft_step=0)
    ptrs = [first.seq_lens.data_ptr(), first.block_table.data_ptr(), first.query_lens.data_ptr()]
    second = builder._prepare_dflash_hybrid_draft_attention_inputs_310(_common([16] * 3), draft_step=0)
    assert getattr(second, "uniform_query_width", 0) == 16
    assert second is first
    assert ptrs == [second.seq_lens.data_ptr(), second.block_table.data_ptr(), second.query_lens.data_ptr()]
    assert second.splitfuse_query_lens_cpu.tolist() == [16] * 10
    assert second.valid_num_tokens.tolist() == [48]
    assert second.seq_lens.tolist() == [24, 25, 26, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("compressed", [False, True])
def test_uniform_attention_groups_queries_without_expanding_block_tables(monkeypatch, compressed):
    inputs = _builder(3)._prepare_dflash_hybrid_draft_attention_inputs_310(_common([4, 4], 16, 4), draft_step=0)
    query = torch.ones(16, 2, 16)
    output = torch.zeros_like(query)
    calls = []

    def splitfuse(**kwargs):
        calls.append(kwargs)
        # Distinct per-request results, plus poisoned dummy rows, exercise the
        # real adapter's grouping and tail isolation at the device boundary.
        kwargs["out"][:4].fill_(24)
        kwargs["out"][4:8].fill_(25)
        kwargs["out"][8:].fill_(float("nan"))

    monkeypatch.setattr(attention.torch_npu, "_npu_paged_attention_splitfuse_v2", splitfuse, raising=False)
    monkeypatch.setattr(attention.torch_npu, "_npu_paged_attention_splitfuse", splitfuse, raising=False)
    monkeypatch.setattr(attention, "is_compressed_mask_supported", lambda: compressed)
    monkeypatch.setattr(attention.torch_npu, "_npu_flash_attention_v3", lambda: None, raising=False)
    monkeypatch.setattr(attention.torch_npu, "_npu_paged_attention", lambda **kw: kw["out"].fill_(42))
    monkeypatch.setattr(AttentionMaskBuilder310, "compressed_non_causal_splitfuse_mask", torch.zeros(2048, 2048))
    base_mask = torch.zeros(32, 32).masked_fill(torch.ones(32, 32).triu(1).bool(), float("-inf"))
    monkeypatch.setattr(AttentionMaskBuilder310, "chunked_prefill_attn_mask", base_mask)
    monkeypatch.setattr(attention.torch_npu, "npu_format_cast", lambda value, fmt: value)
    result = attention.dflash_hybrid_draft_graph_safe_attention_310(
        query=query,
        key_cache=torch.zeros(1),
        value_cache=torch.zeros(1),
        inputs=inputs,
        num_kv_heads=1,
        num_heads=2,
        scale=0.25,
        output=output,
    )
    assert torch.equal(result[:4], torch.full_like(result[:4], 24))
    assert torch.equal(result[4:8], torch.full_like(result[4:8], 25))
    assert torch.equal(result[8:], torch.zeros_like(result[8:]))
    assert result.data_ptr() == output.data_ptr()
    assert len(calls) == 1
    assert calls[0]["seq_len"].tolist() == [4, 4, 4, 4]
    assert calls[0]["block_table"].shape == (4, 4)
    assert calls[0]["context_lens"].tolist() == [24, 25, 4, 4]
    if not compressed:
        # Undo only the documented NZ reshape; expected non-causal visibility
        # is literal (24 columns, 25 columns, then 4-column dummy groups).
        mask = calls[0]["mask"].permute(0, 2, 1, 3).reshape(16, 32)
        for row, visible in ((0, 24), (3, 24), (4, 25), (7, 25), (8, 4), (15, 4)):
            assert torch.equal(mask[row, :visible], torch.zeros(visible))
            assert torch.isneginf(mask[row, visible:]).all()


def test_copy_rejects_different_captured_host_grouping():
    optimized = _builder()._prepare_dflash_hybrid_draft_attention_inputs_310(_common([16]), draft_step=0)
    generic = _builder()._prepare_dflash_hybrid_draft_attention_inputs_310(_common([1]), draft_step=0)
    before = optimized.valid_num_tokens.clone()
    with pytest.raises(ValueError, match="group"):
        attention.copy_dflash_hybrid_draft_attention_inputs_310(optimized, generic)
    assert torch.equal(optimized.valid_num_tokens, before)


@pytest.mark.parametrize("raise_from_model", [False, True])
def test_changed_host_layout_bypasses_replay_without_mutating_captured_inputs(raise_from_model):
    captured = _builder()._prepare_dflash_hybrid_draft_attention_inputs_310(_common([16]), draft_step=0)
    current = _builder()._prepare_dflash_hybrid_draft_attention_inputs_310(_common([1]), draft_step=0)
    descriptor = "test-full160"
    context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata={"layer.0": SimpleNamespace(private=current)},
    )
    observed_modes = []

    def run_model(**kwargs):
        observed_modes.append(context.cudagraph_runtime_mode)
        if raise_from_model:
            raise ValueError("test model failure")
        return "uncaptured-result"

    wrapper = object.__new__(proposer.DFlashHybridDraftForwardACLGraphWrapper310)
    wrapper.vllm_config = object()
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.runnable = run_model
    wrapper.concrete_aclgraph_entries = {descriptor: SimpleNamespace(aclgraph=object())}
    wrapper._hybrid_draft_staging_by_descriptor_310 = {descriptor: {"layer.0": captured}}
    with (
        patch.object(proposer, "get_forward_context", return_value=context),
        patch("vllm_ascend.compilation.acl_graph.get_forward_context", return_value=context),
        patch.object(proposer, "is_310p_dflash_full_and_piecewise", return_value=True),
        patch.object(proposer, "get_dflash_hybrid_draft_attention_inputs_310", side_effect=lambda m: m.private),
    ):
        if raise_from_model:
            with pytest.raises(ValueError, match="test model failure"):
                wrapper()
        else:
            assert wrapper() == "uncaptured-result"
    assert observed_modes == [CUDAGraphMode.NONE]
    assert context.cudagraph_runtime_mode is CUDAGraphMode.FULL
    assert captured.valid_num_tokens.tolist() == [16]
    assert captured.splitfuse_query_lens_cpu.tolist() == [16] * 10

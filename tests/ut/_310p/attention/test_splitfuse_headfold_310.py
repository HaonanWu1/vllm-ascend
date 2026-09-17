# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for the K7 FDO head-folding gate, layout, and mask lifetime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend._310p.attention.attention_mask as mask_module
import vllm_ascend._310p.attention.attention_v1 as attention_module


def make_config(k=7):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=k),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
    )


@pytest.mark.parametrize(
    "heads,kvheads,dim,expected",
    [(8, 1, 256, 8), (16, 4, 128, 4), (8, 8, 128, 1), (8, 4, 128, 1), (8, 1, 64, 1), (8, 0, 128, 1)],
)
def test_shape_gate(monkeypatch, heads, kvheads, dim, expected):
    monkeypatch.setattr(attention_module, "is_310p_dflash_full_decode_only", lambda cfg: True)
    impl = SimpleNamespace(num_heads=heads, num_kv_heads=kvheads, vllm_config=make_config())
    query = torch.empty((80, heads, dim), dtype=torch.float16)
    factor = attention_module.AscendAttentionBackendImpl310._splitfuse_head_fold_factor
    assert factor(impl, query, torch.empty_like(query), CUDAGraphMode.FULL) == expected


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_other_runtime_modes_preserved(monkeypatch, mode):
    monkeypatch.setattr(attention_module, "is_310p_dflash_full_decode_only", lambda cfg: True)
    impl = SimpleNamespace(num_heads=8, num_kv_heads=1, vllm_config=make_config())
    query = torch.empty((80, 8, 256), dtype=torch.float16)
    factor = attention_module.AscendAttentionBackendImpl310._splitfuse_head_fold_factor
    assert factor(impl, query, query, mode) == 1


@pytest.mark.parametrize("k,tokens", [(15, 80), (7, 8), (7, 40)])
def test_other_speculative_lengths_and_descriptors_preserved(monkeypatch, k, tokens):
    monkeypatch.setattr(attention_module, "is_310p_dflash_full_decode_only", lambda cfg: True)
    impl = SimpleNamespace(num_heads=8, num_kv_heads=1, vllm_config=make_config(k))
    query = torch.empty((tokens, 8, 256), dtype=torch.float16)
    factor = attention_module.AscendAttentionBackendImpl310._splitfuse_head_fold_factor
    assert factor(impl, query, query, CUDAGraphMode.FULL) == 1


def test_other_config_preserved(monkeypatch):
    monkeypatch.setattr(attention_module, "is_310p_dflash_full_decode_only", lambda cfg: False)
    impl = SimpleNamespace(num_heads=8, num_kv_heads=1, vllm_config=make_config())
    query = torch.empty((80, 8, 256), dtype=torch.float16)
    factor = attention_module.AscendAttentionBackendImpl310._splitfuse_head_fold_factor
    assert factor(impl, query, query, CUDAGraphMode.FULL) == 1


@pytest.mark.parametrize("case", ["dtype", "query_stride", "output_stride"])
def test_unsupported_tensor_layout_preserved(monkeypatch, case):
    monkeypatch.setattr(attention_module, "is_310p_dflash_full_decode_only", lambda cfg: True)
    impl = SimpleNamespace(num_heads=8, num_kv_heads=1, vllm_config=make_config())
    query = torch.empty((80, 8, 256), dtype=torch.float16)
    output = torch.empty_like(query)
    if case == "dtype":
        query = query.float()
    elif case == "query_stride":
        query = torch.empty((80, 8, 512), dtype=torch.float16)[..., ::2]
    else:
        output = torch.empty((80, 8, 512), dtype=torch.float16)[..., ::2]
    factor = attention_module.AscendAttentionBackendImpl310._splitfuse_head_fold_factor
    assert factor(impl, query, output, CUDAGraphMode.FULL) == 1


@pytest.mark.parametrize("causal", [True, False])
def test_folded_masks_cache_separately_and_expire(monkeypatch, causal):
    cls = mask_module.AttentionMaskBuilder310
    metadata = SimpleNamespace(num_actual_tokens=2, query_start_loc=torch.tensor([0, 2]), seq_lens=torch.tensor([2]))
    context = SimpleNamespace(vllm_config=make_config(), cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(mask_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(mask_module, "is_310p_dflash_full_decode_only", lambda cfg: True)
    positions = Mock(return_value=torch.tensor([0, 1]))
    monkeypatch.setattr(cls, "_get_query_positions", positions)
    monkeypatch.setattr(mask_module, "nd_to_nz_spec", lambda tensor: tensor)
    monkeypatch.setattr(mask_module.torch_npu, "npu_format_cast", lambda tensor, fmt: tensor)
    monkeypatch.setattr(
        cls, "chunked_prefill_attn_mask", torch.tensor([[0, float("-inf")], [0, 0]], dtype=torch.float16)
    )
    monkeypatch.setattr(cls, "max_seqlen", 2)
    build = cls.get_splitfuse_mask if causal else cls.get_non_causal_splitfuse_mask
    original = build(metadata, torch.device("cpu"))
    folded = build(metadata, torch.device("cpu"), query_head_repeat=8)
    assert torch.equal(folded, original.repeat_interleave(8, dim=0))
    assert build(metadata, torch.device("cpu"), query_head_repeat=8) is folded
    assert positions.call_count == 2
    context = SimpleNamespace(vllm_config=make_config(), cudagraph_runtime_mode=CUDAGraphMode.FULL)
    assert build(metadata, torch.device("cpu"), query_head_repeat=8) is not folded
    assert positions.call_count == 3


@pytest.mark.parametrize("heads,kvheads,dim,causal", [(8, 1, 256, True), (16, 4, 128, False)])
def test_forward_restores_heads_and_clears_unwritten_padding(monkeypatch, heads, kvheads, dim, causal):
    impl = object.__new__(attention_module.AscendAttentionBackendImpl310)
    impl.num_heads, impl.num_kv_heads = heads, kvheads
    impl.scale = 0.125
    impl.support_compressed_mask = False
    impl.vllm_config = make_config()
    impl.key_cache = torch.empty(0)
    impl.value_cache = torch.empty(0)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(attention_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention_module, "is_310p_dflash_full_decode_only", lambda cfg: True)
    monkeypatch.setattr(attention_module, "get_dflash_hybrid_draft_attention_inputs_310", lambda metadata: None)
    qlens = torch.full((10,), 8, dtype=torch.int32)
    monkeypatch.setattr(attention_module, "get_query_lens_cpu", lambda metadata: qlens)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    mask = torch.empty(0)
    builder = Mock(return_value=mask)
    method = "get_splitfuse_mask" if causal else "get_non_causal_splitfuse_mask"
    monkeypatch.setattr(attention_module.AttentionMaskBuilder310, method, builder)
    metadata = SimpleNamespace(
        num_actual_tokens=80,
        seq_lens=torch.tensor([32, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        block_tables=torch.zeros((10, 1), dtype=torch.int32),
        causal=causal,
    )
    query = (torch.arange(80 * heads * dim) % 31).reshape(80, heads, dim).to(torch.float16)
    output = torch.full_like(query, -19)
    fold = heads // kvheads

    def fake_splitfuse(**kwargs):
        assert kwargs["num_heads"] == kvheads
        assert kwargs["num_kv_heads"] == kvheads
        assert kwargs["key_cache"] is impl.key_cache
        assert kwargs["value_cache"] is impl.value_cache
        assert kwargs["context_lens"] is metadata.seq_lens
        assert kwargs["scale_value"] == impl.scale
        assert torch.equal(kwargs["seq_len"], qlens * fold)
        # Simulate a kernel that writes only the first active request.
        kwargs["out"][: 8 * fold].copy_(kwargs["query"][: 8 * fold] * 2)

    monkeypatch.setattr(attention_module.torch_npu, "_npu_paged_attention_splitfuse", fake_splitfuse)
    assert impl.forward_chunked_prefill_310(query, metadata, output) is output
    builder.assert_called_once_with(metadata, query.device, query_head_repeat=fold)
    torch.testing.assert_close(output[:8], query[:8] * 2, rtol=0, atol=0)
    assert torch.count_nonzero(output[8:]) == 0

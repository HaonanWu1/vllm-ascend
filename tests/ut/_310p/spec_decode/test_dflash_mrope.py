# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm_ascend._310p.spec_decode.dflash_mrope import (
    DFlashMRoPEState310,
    build_dflash_mrope_positions,
    gather_mrope_cos_sin,
    mrope_axis_indices,
)


def rotary():
    inv_freq = 1.0 / (5000000 ** (torch.arange(0, 128, 2).float() / 128))
    freqs = torch.arange(1024).float()[:, None] * inv_freq
    return SimpleNamespace(
        cos_sin_cache=torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        head_size=128,
        rotary_dim=128,
        is_neox_style=True,
        mrope_section=[24, 20, 20],
        mrope_interleaved=True,
    )


@pytest.mark.parametrize("rejected", [None, [0, 0], [1, 3], [2, 4]])
def test_image_positions_never_become_cache_offsets(rejected):
    # Two differently sized scheduled suffixes of requests with cached prefixes.
    positions = torch.tensor([[3, 4, 1, 2, 3, 4], [7, 8, 2, 2, 3, 3], [1, 1, 5, 6, 5, 6]])
    rejection = None if rejected is None else torch.tensor(rejected)
    context, query = build_dflash_mrope_positions(
        positions,
        torch.tensor([0, 2, 6]),
        torch.tensor([102, 204]),
        rejection,
        torch.tensor([-50, -180]),
        6,
        8,
    )
    assert context.tolist() == [100, 101, 200, 201, 202, 203]
    rejection = [0, 0] if rejected is None else rejected
    expected = list(range(52 - rejection[0], 60 - rejection[0])) + list(range(24 - rejection[1], 32 - rejection[1]))
    assert query.tolist() == [expected] * 3


def test_short_context_does_not_read_previous_requests_position():
    context, query = build_dflash_mrope_positions(
        torch.zeros(3, 2),
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 17]),
        torch.tensor([1, 1]),
        torch.tensor([0, -5]),
        2,
        8,
    )
    assert context.tolist() == [0, 16]
    assert query[0].tolist() == list(range(8)) + list(range(11, 19))


def test_unfinished_image_prefill_uses_valid_dummy_query_coordinates():
    # The first request has only processed its first image chunk; its delta
    # already describes the complete two-image prompt. The second is decoding.
    context, query = build_dflash_mrope_positions(
        torch.zeros(3, 65),
        torch.tensor([0, 64, 65]),
        torch.tensor([64, 200]),
        None,
        torch.tensor([-112, -56]),
        65,
        8,
    )
    assert context.tolist() == list(range(64)) + [199]
    assert query.tolist() == [list(range(8)) + list(range(144, 152))] * 3


def test_interleaved_mrope_matches_training_frequency_selection():
    emb = rotary()
    # Independent reproduction of the saved training model's frequency rule.
    positions = torch.tensor([[3, 4, 5], [7, 8, 9], [11, 12, 13]])
    inv_freq = 1.0 / (5000000 ** (torch.arange(0, 128, 2).float() / 128))
    freqs = positions.float()[:, :, None] * inv_freq
    reference = freqs[0].clone()
    reference[:, 1:60:3] = freqs[1, :, 1:60:3]
    reference[:, 2:60:3] = freqs[2, :, 2:60:3]
    reference = torch.cat((reference, reference), dim=-1).view(1, 3, 1, 128)
    axes = mrope_axis_indices([24, 20, 20], True, positions.device)
    cos, sin = gather_mrope_cos_sin(emb.cos_sin_cache, positions, axes)
    torch.testing.assert_close(cos, reference.cos())
    torch.testing.assert_close(sin, reference.sin())
    assert axes.bincount().tolist() == [24, 20, 20]


def test_text_mrope_reduces_to_standard_rope():
    emb = rotary()
    state = DFlashMRoPEState310(emb, 8)
    positions = torch.tensor([[17, 18, 19]]).expand(3, -1)
    state.refresh(positions, positions)
    q, k = torch.randn(3, 256), torch.randn(3, 128)
    out_q, out_k = state.apply(q, k)
    cos, sin = emb.cos_sin_cache[positions[0]].chunk(2, -1)
    for value, result in [(q, out_q), (k, out_k)]:
        heads = value.view(3, -1, 128)
        left, right = heads.chunk(2, -1)
        expected = heads * cos.repeat(1, 2)[:, None] + torch.cat((-right, left), -1) * sin.repeat(1, 2)[:, None]
        torch.testing.assert_close(result, expected.reshape_as(value))


def test_context_query_and_target_cache_are_isolated_and_persistent():
    emb = rotary()
    cache = emb.cos_sin_cache.clone()
    first, second = DFlashMRoPEState310(emb, 16), DFlashMRoPEState310(emb, 16)
    ptrs = [x.data_ptr() for x in (first.query_cos, first.query_sin, first.context_cos, first.context_sin)]
    pos = torch.arange(8).expand(3, -1)
    first.refresh(pos + 40, pos)
    q, k = torch.randn(8, 256), torch.randn(8, 128)
    query_k = first.apply(q, k)[1]
    context_k = first.apply(q, k, context=True)[1]
    assert not torch.allclose(query_k, context_k)
    second.refresh(pos + 200, pos + 100)
    torch.testing.assert_close(first.apply(q, k)[1], query_k)
    first.refresh(pos[:, :3], pos[:, :0])
    assert ptrs == [x.data_ptr() for x in (first.query_cos, first.query_sin, first.context_cos, first.context_sin)]
    assert torch.all(first.query_cos[:, 3:] == 1)
    assert torch.all(first.query_sin[:, 3:] == 0)
    assert torch.all(first.context_cos == 1)
    torch.testing.assert_close(emb.cos_sin_cache, cache)
    empty_q, empty_k = first.apply(q[:0], k[:0], context=True)
    assert empty_q.shape == (0, 256)
    assert empty_k.shape == (0, 128)


def test_bad_positions_and_sections_are_rejected():
    emb = rotary()
    axes = mrope_axis_indices([24, 20, 20], True, torch.device("cpu"))
    with pytest.raises(ValueError, match="shape"):
        gather_mrope_cos_sin(emb.cos_sin_cache, torch.arange(3), axes)
    with pytest.raises(ValueError, match="sections"):
        mrope_axis_indices([10, -1, 4], True, torch.device("cpu"))
    with pytest.raises(ValueError, match="capacity"):
        DFlashMRoPEState310(emb, 2).refresh(torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 1, dtype=torch.long))


def test_proposer_preserves_three_axes_and_orders_request_deltas():
    from vllm_ascend._310p.spec_decode.dflash_proposer_310 import AscendDflashProposer310

    delta_cpu = np.zeros(2, dtype=np.int32)
    deltas = SimpleNamespace(np=delta_cpu, gpu=torch.from_numpy(delta_cpu), copy_to_gpu=Mock())
    proposer = SimpleNamespace(
        uses_mrope=True,
        num_speculative_tokens=7,
        _dflash_hidden_states=torch.zeros(6, 4),
        _context_mrope_positions_buffer=torch.ones(3, 20, dtype=torch.int32),
        mrope_positions=torch.ones(3, 20, dtype=torch.int32),
        _dflash_mrope_deltas=deltas,
        _slot_mapping_buffer=torch.zeros(16, dtype=torch.int32),
        arange_dflash=torch.arange(20, dtype=torch.int32),
        token_arange_np=np.arange(20),
        vllm_config=SimpleNamespace(),
        runner=SimpleNamespace(
            uses_mrope=True,
            input_batch=SimpleNamespace(req_ids=["second", "first"]),
            requests={
                "first": SimpleNamespace(mrope_position_delta=-180),
                "second": SimpleNamespace(mrope_position_delta=-50),
            },
        ),
    )
    cad = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 6]),
        seq_lens=torch.tensor([102, 204]),
        max_seq_len=204,
    )
    positions = torch.tensor([[3, 4, 1, 2, 3, 4], [7, 8, 2, 2, 3, 3], [1, 1, 5, 6, 5, 6]])
    with (
        patch("vllm_ascend._310p.spec_decode.dflash_proposer_310.is_310p_dflash_full_decode_only", return_value=False),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310._copy_and_expand_inputs_ascendc",
            return_value=torch.arange(14),
        ) as expand,
    ):
        AscendDflashProposer310.set_inputs_first_pass(
            proposer,
            torch.zeros(6, dtype=torch.int32),
            torch.tensor([1, 2]),
            positions,
            torch.randn(6, 4),
            None,
            cad,
            torch.tensor([1, 3]),
        )
    assert expand.call_args.kwargs["target_positions"].tolist() == [100, 101, 200, 201, 202, 203]
    torch.testing.assert_close(proposer._context_mrope_positions_buffer[:, :6], positions.to(torch.int32))
    assert proposer.mrope_positions[0, :16].tolist() == list(range(51, 59)) + list(range(21, 29))
    assert torch.all(proposer.mrope_positions[:, 16:] == 0)
    assert torch.all(proposer._context_mrope_positions_buffer[:, 6:] == 0)
    assert cad.seq_lens.tolist() == [109, 209]
    assert not cad.causal


def test_mrope_guard_is_only_relaxed_for_dflash():
    from vllm_ascend._310p.spec_decode.dflash_proposer_310 import AscendDflashProposer310

    with patch("vllm_ascend._310p.spec_decode.dflash_proposer_310._original_dflash_raise_if_mrope") as original:
        proposer = SimpleNamespace(method="dflash", draft_model_config=SimpleNamespace(uses_mrope=True))
        AscendDflashProposer310._raise_if_mrope(proposer)
        original.assert_not_called()
        proposer.method = "dspark"
        AscendDflashProposer310._raise_if_mrope(proposer)
        original.assert_called_once_with(proposer)


def test_worker_patch_binds_bootstrap_and_runtime_dflash():
    # The GPU runner briefly constructs an upstream DFlash proposer before the
    # Ascend runner replaces it. Test the real patch import in an isolated
    # process so its other worker bindings cannot affect unrelated unit tests.
    import subprocess
    import sys

    code = """
import runpy
runpy.run_path("tests/ut/conftest.py")
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import AscendDflashProposer310
from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer
dspark_guard = AscendDsparkProposer._raise_if_mrope
import vllm_ascend.patch.worker.patch_idex_310
assert DFlashProposer._raise_if_mrope is AscendDflashProposer310._raise_if_mrope
assert AscendDflashProposer._raise_if_mrope is AscendDflashProposer310._raise_if_mrope
assert AscendDflashProposer.__init__ is AscendDflashProposer310.__init__
assert AscendDsparkProposer._raise_if_mrope is dspark_guard
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend._310p.spec_decode.dflash_vocab import load_dflash_weights_310, maybe_share_dflash_lm_head_310
assert DFlashQwen3ForCausalLM.load_weights is load_dflash_weights_310
assert AscendSpecDecodeBaseProposer._maybe_share_lm_head is maybe_share_dflash_lm_head_310
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_context_mrope_rotation_precedes_each_physical_cache_write():
    from vllm_ascend._310p.spec_decode.dflash_model_310 import precompute_and_store_context_kv_310

    hidden = torch.randn(3, 4)
    weights = torch.randn(2 * 2 * 128, 4)
    state = SimpleNamespace(apply=Mock(side_effect=lambda q, k, context: (q * 2, k * 2)))
    attention = [
        SimpleNamespace(k_norm=torch.nn.Identity(), rotary_emb=Mock(), _dflash_mrope_state=state) for _ in range(2)
    ]
    caches = [SimpleNamespace(kv_cache=object(), impl=SimpleNamespace(do_kv_cache_update=Mock())) for _ in range(2)]
    model = SimpleNamespace(
        _num_attn_layers=2,
        _kv_size=128,
        _head_dim=128,
        _num_kv_heads=1,
        hidden_norm=torch.nn.Identity(),
        _fused_kv_weight=weights,
        _fused_kv_bias=None,
        layers=[SimpleNamespace(self_attn=attn) for attn in attention],
        _attn_layers=caches,
    )
    slots = [torch.tensor([65, 66, 67]), torch.tensor([129, 130, 131])]
    precompute_and_store_context_kv_310(model, hidden, torch.arange(3).expand(3, -1), slots)
    projected = torch.nn.functional.linear(hidden, weights).view(3, 2, 2, 1, 128)
    for i, cache in enumerate(caches):
        args = cache.impl.do_kv_cache_update.call_args.args
        torch.testing.assert_close(args[1], projected[:, i, 0] * 2)
        torch.testing.assert_close(args[2], projected[:, i, 1])
        assert args[4] is slots[i]
        attention[i].rotary_emb.assert_not_called()
    assert state.apply.call_count == 2
    assert all(call.kwargs["context"] for call in state.apply.call_args_list)


@pytest.mark.parametrize("fdo,full", [(True, True), (True, False), (False, True)])
def test_mrope_refresh_preserves_fdo_context_padding_invalidation(fdo, full):
    from vllm.config import CUDAGraphMode

    from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310

    positions = torch.arange(12).reshape(3, 4)
    context = torch.arange(24).reshape(3, 8)
    state = SimpleNamespace(refresh=Mock())
    proposer = SimpleNamespace(
        method="dflash",
        uses_mrope=True,
        vllm_config=SimpleNamespace(),
        _dflash_mrope_state=state,
        _dflash_num_context=3,
        _context_mrope_positions_buffer=context,
        _context_slot_mapping_buffer=torch.arange(12, dtype=torch.int32),
        _get_positions=Mock(return_value=positions),
    )
    with patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only", return_value=fdo):
        prepared = AscendSpecDecodeBaseProposer310._prepare_full_decode_draft_rope(
            proposer,
            query_positions=torch.zeros(3, 8),
            query_actual_tokens=4,
            descriptor_tokens=8,
            runtime_mode=CUDAGraphMode.FULL if full else CUDAGraphMode.NONE,
        )
    assert not prepared  # Draft-owned caches must not clear the target's state.
    assert state.refresh.call_args.args[0] is positions
    torch.testing.assert_close(state.refresh.call_args.args[1], context[:, :3])
    expected = torch.arange(12, dtype=torch.int32)
    if fdo and full:
        expected[3:8] = -1
    torch.testing.assert_close(proposer._context_slot_mapping_buffer, expected)

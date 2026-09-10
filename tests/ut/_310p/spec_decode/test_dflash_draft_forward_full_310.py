# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend._310p.spec_decode.llm_base_proposer_310 as proposer_310
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    AscendDflashProposer310,
)


class _FakeDraftModel:
    def __call__(self, **model_inputs):
        return model_inputs["input_ids"]

    def compute_logits(self, hidden_states):
        return ("logits", hidden_states)


class _FakeACLGraphWrapper:
    def __init__(
        self,
        runnable,
        _vllm_config,
        runtime_mode,
        *,
        use_eagle=False,
        enable_enpu=False,
        component=None,
        retained_input_provider=None,
        proposer=None,
    ):
        self.runnable = runnable
        self.runtime_mode = runtime_mode
        self.use_eagle = use_eagle
        self.enable_enpu = enable_enpu
        self.component = component
        self.retained_input_provider = retained_input_provider
        self.proposer = proposer
        self.concrete_aclgraph_entries = {}

    def unwrap(self):
        return self.runnable

    def __getattr__(self, name):
        return getattr(self.runnable, name)


def _hybrid_config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            num_speculative_tokens=15,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )


def test_hybrid_full_wraps_merged_draft_including_context_and_sampling():
    merged_tail = object()
    draft_model = _FakeDraftModel()
    proposer = SimpleNamespace(
        vllm_config=_hybrid_config(),
        use_cuda_graph=True,
        use_eagle=True,
        enable_enpu=False,
        _runnable=_FakeACLGraphWrapper(
            merged_tail,
            _hybrid_config(),
            CUDAGraphMode.FULL,
            component="draft",
        ),
        model=draft_model,
        _full_decode_draft_retained_inputs=lambda *_: (),
    )

    with (
        patch.object(proposer_310, "ACLGraphWrapper", _FakeACLGraphWrapper),
        patch.object(
            proposer_310,
            "DFlashHybridDraftACLGraphWrapper310",
            _FakeACLGraphWrapper,
        ),
        patch.object(
            proposer_310,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
    ):
        proposer_310.AscendSpecDecodeBaseProposer310._install_hybrid_draft_full_graph_310(
            proposer
        )

    assert proposer.model is draft_model
    assert isinstance(proposer._runnable, _FakeACLGraphWrapper)
    assert proposer._runnable.runnable is merged_tail
    assert proposer._runnable.runtime_mode is CUDAGraphMode.FULL
    assert proposer._runnable.component == "draft"
    assert proposer._runnable.proposer is proposer
    assert proposer.model.compute_logits("hidden") == ("logits", "hidden")


def test_hybrid_first_draft_step_builds_the_same_private_device_contract():
    metadata = SimpleNamespace(causal=False, attn_mask=object())
    builder = Mock()
    builder.build.return_value = metadata
    runner = SimpleNamespace(get_model=lambda: object())
    proposer = SimpleNamespace(
        vllm_config=_hybrid_config(),
        runner=runner,
        attn_layer_names=["layer.0"],
    )
    common = object()

    with patch(
        "vllm_ascend._310p.spec_decode.dflash_proposer_310."
        "is_310p_dflash_full_and_piecewise",
        return_value=True,
    ):
        result = AscendDflashProposer310._build_first_pass_per_layer_attn_metadata(
            proposer,
            builder,
            common,
            SimpleNamespace(causal=False),
            {},
        )

    assert result == {"layer.0": metadata}
    builder.build.assert_called_once()
    assert builder.build.call_args.kwargs == {
        "is_drafting": True,
        "dflash_hybrid_draft_step": 0,
    }


def _private_inputs(value: int):
    return proposer_310.DFlashHybridDraftAttentionInputs310(
        valid_num_reqs=torch.tensor([value], dtype=torch.int32),
        valid_num_tokens=torch.tensor([value], dtype=torch.int32),
        query_lens=torch.full((4,), value, dtype=torch.int32),
        query_starts=torch.full((4,), value, dtype=torch.int32),
        query_ends=torch.full((4,), value, dtype=torch.int32),
        seq_lens=torch.full((4,), value, dtype=torch.int32),
        block_table=torch.full((4, 2), value, dtype=torch.int32),
        token_indices=torch.arange(8, dtype=torch.int32),
    )


def test_hybrid_draft_model_copies_current_substep_into_captured_device_inputs():
    descriptor = Mock(num_tokens=64)
    captured_inputs = _private_inputs(1)
    current_inputs = _private_inputs(2)
    step_zero = {"layer.0": SimpleNamespace(private=captured_inputs)}
    step_one = {"layer.0": SimpleNamespace(private=current_inputs)}
    forward_context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata=step_zero,
    )
    proposer = SimpleNamespace(vllm_config=_hybrid_config())
    wrapper = object.__new__(
        proposer_310.DFlashHybridDraftACLGraphWrapper310
    )
    wrapper.vllm_config = proposer.vllm_config
    wrapper._hybrid_draft_staging_by_descriptor_310 = {
        descriptor: {"layer.0": captured_inputs}
    }
    wrapper.concrete_aclgraph_entries = {
        descriptor: SimpleNamespace(aclgraph=object())
    }

    with (
        patch.object(
            proposer_310,
            "get_forward_context",
            return_value=forward_context,
        ),
        patch.object(
            proposer_310,
            "get_dflash_hybrid_draft_attention_inputs_310",
            side_effect=lambda metadata: metadata.private,
        ),
        patch.object(
            proposer_310.ACLGraphWrapper,
            "__call__",
            return_value="replayed",
        ),
    ):
        forward_context.attn_metadata = step_one
        assert wrapper._call_with_metadata() == "replayed"

    assert captured_inputs.valid_num_reqs.item() == 2
    assert captured_inputs.valid_num_tokens.item() == 2
    assert captured_inputs.query_lens.tolist() == [2, 2, 2, 2]
    assert captured_inputs.seq_lens.tolist() == [2, 2, 2, 2]
    assert captured_inputs.block_table.tolist() == [[2, 2]] * 4


def test_hybrid_draft_model_records_capture_inputs_without_copying_them():
    descriptor = Mock(num_tokens=64)
    captured_inputs = _private_inputs(3)
    forward_context = SimpleNamespace(
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata={"layer.0": SimpleNamespace(private=captured_inputs)},
    )
    proposer = SimpleNamespace(vllm_config=_hybrid_config())
    wrapper = object.__new__(
        proposer_310.DFlashHybridDraftACLGraphWrapper310
    )
    wrapper.vllm_config = proposer.vllm_config
    wrapper._hybrid_draft_staging_by_descriptor_310 = {}
    wrapper.concrete_aclgraph_entries = {}

    def capture(*args, **kwargs):
        wrapper.concrete_aclgraph_entries[descriptor] = SimpleNamespace(
            aclgraph=object()
        )
        return "captured"

    with (
        patch.object(
            proposer_310,
            "get_forward_context",
            return_value=forward_context,
        ),
        patch.object(
            proposer_310,
            "get_dflash_hybrid_draft_attention_inputs_310",
            side_effect=lambda metadata: metadata.private,
        ),
        patch.object(
            proposer_310.ACLGraphWrapper,
            "__call__",
            side_effect=capture,
        ),
    ):
        assert wrapper._call_with_metadata() == "captured"

    assert wrapper._hybrid_draft_staging_by_descriptor_310[descriptor] == {
        "layer.0": captured_inputs
    }


def test_non_hybrid_does_not_change_existing_draft_wrapper_boundary():
    merged_tail = object()
    draft_model = _FakeDraftModel()
    existing_wrapper = _FakeACLGraphWrapper(
        merged_tail,
        _hybrid_config(),
        CUDAGraphMode.FULL,
        component="draft",
    )
    proposer = SimpleNamespace(
        vllm_config=_hybrid_config(),
        use_cuda_graph=True,
        _runnable=existing_wrapper,
        model=draft_model,
    )

    with patch.object(
        proposer_310,
        "is_310p_dflash_full_and_piecewise",
        return_value=False,
    ):
        proposer_310.AscendSpecDecodeBaseProposer310._install_hybrid_draft_full_graph_310(
            proposer
        )

    assert proposer._runnable is existing_wrapper
    assert proposer.model is draft_model


def test_patched_load_model_installs_island_on_public_proposer_instance():
    proposer = SimpleNamespace()
    target_model = object()
    original_load_model = Mock()
    installer = Mock()

    with (
        patch.object(
            proposer_310,
            "_original_load_model",
            original_load_model,
        ),
        patch.object(
            proposer_310.AscendSpecDecodeBaseProposer310,
            "_install_hybrid_draft_full_graph_310",
            installer,
        ),
    ):
        proposer_310.AscendSpecDecodeBaseProposer310.load_model(
            proposer,
            target_model,
        )

    original_load_model.assert_called_once_with(proposer, target_model)
    installer.assert_called_once_with(proposer)


def test_merged_draft_refreshes_fixed_inputs_and_restores_logical_context():
    proposer = SimpleNamespace(
        num_speculative_tokens=2,
        attn_layer_names=["layer.0", "layer.1"],
        token_indices_to_sample=torch.zeros(4, dtype=torch.int64),
        _dflash_num_context=3,
        _context_slot_mapping_buffer=torch.tensor([1, 2, 3, 91, 92, 93]),
        _slot_mapping_buffer=torch.tensor([10, 11, 12, 91, 92, 93]),
        _dflash_context_slot_mapping_by_layer_310=[
            torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22])
        ],
    )
    wrapper = object.__new__(proposer_310.DFlashHybridDraftACLGraphWrapper310)
    wrapper.proposer = proposer
    wrapper._hybrid_context_slots_310 = {}
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    original_slots = proposer._dflash_context_slot_mapping_by_layer_310
    captured = []

    def replay(**kwargs):
        slots = proposer._dflash_context_slot_mapping_by_layer_310
        assert proposer._dflash_num_context == 6
        assert slots[0].tolist() == [10, 11, 12, -1, -1, -1]
        assert slots[1].tolist() == [20, 21, 22, -1, -1, -1]
        captured.append((slots[0].data_ptr(), kwargs["token_indices_to_sample"].data_ptr()))
        return kwargs["token_indices_to_sample"].view(-1, 2).clone()

    wrapper._call_with_metadata = replay
    with patch.object(proposer_310, "get_forward_context", return_value=context):
        for indices in ([1, 2], [2, 0]):
            result = wrapper._call_merged_draft(
                num_input_tokens=6, token_indices_to_sample=torch.tensor(indices)
            )
            assert result.tolist() == [indices]
            assert proposer._dflash_num_context == 3
            assert proposer._dflash_context_slot_mapping_by_layer_310 is original_slots
    assert captured[0] == captured[1]


def test_merged_draft_restores_proposer_after_failure():
    proposer = SimpleNamespace(
        num_speculative_tokens=2, attn_layer_names=["layer.0"],
        token_indices_to_sample=torch.zeros(4, dtype=torch.int64),
        _dflash_num_context=3,
        _context_slot_mapping_buffer=torch.tensor([1, 2, 3, 91, 92, 93]),
        _slot_mapping_buffer=torch.tensor([10, 11, 12, 91, 92, 93]),
    )
    wrapper = object.__new__(proposer_310.DFlashHybridDraftACLGraphWrapper310)
    wrapper.proposer = proposer
    wrapper._hybrid_context_slots_310 = {}
    wrapper._call_with_metadata = Mock(side_effect=RuntimeError("device failure"))
    with pytest.raises(RuntimeError, match="device failure"):
        wrapper._call_merged_draft(num_input_tokens=6, token_indices_to_sample=torch.tensor([1, 2]))
    assert proposer._dflash_num_context == 3
    assert not hasattr(proposer, "_dflash_context_slot_mapping_by_layer_310")


def test_merged_draft_large_context_uses_original_inputs_without_replay():
    proposer = SimpleNamespace(_dflash_num_context=12)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    wrapper = object.__new__(proposer_310.DFlashHybridDraftACLGraphWrapper310)
    wrapper.proposer = proposer
    indices = torch.tensor([1, 2])

    def eager(**kwargs):
        assert context.cudagraph_runtime_mode is CUDAGraphMode.NONE
        assert proposer._dflash_num_context == 12
        assert kwargs["token_indices_to_sample"] is indices
        return "uncaptured"

    wrapper._call_with_metadata = eager
    with patch.object(proposer_310, "get_forward_context", return_value=context):
        assert wrapper._call_merged_draft(num_input_tokens=6, token_indices_to_sample=indices) == "uncaptured"
    assert context.cudagraph_runtime_mode is CUDAGraphMode.FULL


@pytest.mark.parametrize("num_requests", [1, 10])
def test_merged_draft_invalidates_captured_query_padding(num_requests):
    # C10 warmup leaves valid addresses in the buffer captured by attention.
    # Clearing a separate runtime slot group must not hide stale graph slots.
    query_slots = torch.arange(176, dtype=torch.int32) + 1000
    before = query_slots.clone()
    proposer = SimpleNamespace(
        num_speculative_tokens=15,
        attn_layer_names=["layer.0"],
        token_indices_to_sample=torch.zeros(150, dtype=torch.int64),
        _dflash_num_context=16,
        _context_slot_mapping_buffer=torch.arange(160, dtype=torch.int32),
        _slot_mapping_buffer=query_slots,
        slot_mapping_group=[torch.full((160,), -1, dtype=torch.int32)],
    )
    wrapper = object.__new__(proposer_310.DFlashHybridDraftACLGraphWrapper310)
    wrapper.proposer = proposer
    wrapper._hybrid_context_slots_310 = {}
    actual_tokens = {1: 16, 10: 160}[num_requests]
    captured_pointer = query_slots.data_ptr()

    def replay(**kwargs):
        assert proposer._slot_mapping_buffer.data_ptr() == captured_pointer
        assert torch.equal(query_slots[:actual_tokens], before[:actual_tokens])
        assert query_slots[actual_tokens:160].tolist() == [-1] * (160 - actual_tokens)
        assert torch.equal(query_slots[160:], before[160:])
        return torch.zeros((10, 15), dtype=torch.int64)

    wrapper._call_with_metadata = replay
    result = wrapper._call_merged_draft(
        num_input_tokens=160,
        token_indices_to_sample=torch.arange(num_requests * 15),
    )
    assert result.shape == (num_requests, 15)

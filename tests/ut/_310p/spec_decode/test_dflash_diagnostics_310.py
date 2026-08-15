# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend._310p.spec_decode.dflash_diagnostics_310 import (
    _reset_dflash_diagnostics_for_test,
    assert_dflash_graph_tensor_addresses_310,
    capture_dflash_graph_dispatch,
    capture_dflash_diagnostic,
    collect_dflash_graph_tensors_310,
    observe_dflash_acl_graph_call_310,
    remember_dflash_graph_modes,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import (
    AscendSpecDecodeBaseProposer310,
)


def test_disabled_diagnostics_do_not_inspect_or_write(tmp_path: Path):
    output = tmp_path / "disabled.jsonl"
    builder = MagicMock(side_effect=AssertionError("payload builder must stay lazy"))
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        _reset_dflash_diagnostics_for_test()
        capture_dflash_diagnostic("draft_inputs", payload_builder=builder)
    _reset_dflash_diagnostics_for_test()

    builder.assert_not_called()
    assert not output.exists()


def test_enabled_diagnostics_are_bounded_per_stage(tmp_path: Path):
    output = tmp_path / "capture.jsonl"
    env = {
        "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output),
        "ASCEND_DFLASH_DIAGNOSTIC_LIMIT": "2",
    }
    with patch.dict(os.environ, env, clear=False):
        _reset_dflash_diagnostics_for_test()
        for value in range(3):
            capture_dflash_diagnostic(
                "draft_inputs",
                token_ids=torch.tensor([[value, value + 1]], dtype=torch.int32),
            )
        capture_dflash_diagnostic("verify", accepted_counts=torch.tensor([3]))
    _reset_dflash_diagnostics_for_test()

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in records] == [
        "draft_inputs",
        "draft_inputs",
        "verify",
    ]
    assert [record["occurrence"] for record in records] == [0, 1, 0]
    assert records[0]["token_ids"] == {
        "dtype": "torch.int32",
        "shape": [1, 2],
        "values": [[0, 1]],
    }


def test_payload_builder_failure_is_swallowed(tmp_path: Path):
    output = tmp_path / "failed.jsonl"
    env = {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output)}
    builder = MagicMock(side_effect=RuntimeError("diagnostic-only failure"))
    with (
        patch.dict(os.environ, env, clear=False),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310.logger.warning_once"
        ) as warning,
    ):
        _reset_dflash_diagnostics_for_test()
        capture_dflash_diagnostic("verify", payload_builder=builder)
    _reset_dflash_diagnostics_for_test()

    builder.assert_called_once_with()
    warning.assert_called_once()
    assert not output.exists()


def test_empty_payload_does_not_consume_stage_quota(tmp_path: Path):
    output = tmp_path / "capture.jsonl"
    env = {
        "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output),
        "ASCEND_DFLASH_DIAGNOSTIC_LIMIT": "1",
    }
    with patch.dict(os.environ, env, clear=False):
        _reset_dflash_diagnostics_for_test()
        capture_dflash_diagnostic("state_metadata", payload_builder=lambda: None)
        capture_dflash_diagnostic("state_metadata", payload_builder=lambda: None)
        capture_dflash_diagnostic(
            "state_metadata",
            payload_builder=lambda: {"state_indices": torch.tensor([[2, 3]])},
        )
    _reset_dflash_diagnostics_for_test()

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["occurrence"] == 0
    assert records[0]["state_indices"]["values"] == [[2, 3]]


def _dflash_graph_config() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        ),
    )


def _graph_descriptor() -> BatchDescriptor:
    return BatchDescriptor(
        num_tokens=32,
        num_reqs=2,
        uniform=True,
        has_lora=False,
        num_active_loras=0,
    )


def test_graph_dispatch_diagnostic_records_requested_normalized_and_runtime(
    tmp_path: Path,
):
    output = tmp_path / "graph.jsonl"
    config = _dflash_graph_config()
    with patch.dict(
        os.environ,
        {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output)},
        clear=False,
    ):
        _reset_dflash_diagnostics_for_test()
        remember_dflash_graph_modes(
            config,
            requested_mode=CUDAGraphMode.FULL,
            normalized_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        capture_dflash_graph_dispatch(
            config,
            path="target",
            runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=_graph_descriptor(),
            actual_num_tokens=23,
        )
    _reset_dflash_diagnostics_for_test()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record == {
        "stage": "graph_dispatch_target",
        "occurrence": 0,
        "path": "target",
        "requested_mode": "FULL",
        "normalized_mode": "FULL_DECODE_ONLY",
        "runtime_mode": "NONE",
        "actual_num_tokens": 23,
        "capture_descriptor": {
            "num_tokens": 32,
            "num_reqs": 2,
            "uniform": True,
            "has_lora": False,
            "num_active_loras": 0,
        },
        "capture_occurred": False,
        "replay_occurred": False,
    }


def test_graph_mode_refresh_preserves_the_original_request(tmp_path: Path):
    output = tmp_path / "graph-refresh.jsonl"
    config = _dflash_graph_config()
    with patch.dict(
        os.environ,
        {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output)},
        clear=False,
    ):
        _reset_dflash_diagnostics_for_test()
        remember_dflash_graph_modes(
            config,
            requested_mode=CUDAGraphMode.FULL,
            normalized_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        remember_dflash_graph_modes(
            config,
            requested_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            normalized_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        capture_dflash_graph_dispatch(
            config,
            path="target",
            runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=_graph_descriptor(),
        )
    _reset_dflash_diagnostics_for_test()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["requested_mode"] == "FULL"
    assert record["normalized_mode"] == "FULL_DECODE_ONLY"


@pytest.mark.parametrize("path", ["target", "draft"])
@pytest.mark.parametrize("action", ["capture", "replay"])
def test_acl_graph_observer_records_path_and_actual_occurrence(
    tmp_path: Path,
    path: str,
    action: str,
):
    output = tmp_path / f"{path}-{action}.jsonl"
    config = _dflash_graph_config()
    descriptor = _graph_descriptor()
    graph = object()
    entries = (
        {}
        if action == "capture"
        else {descriptor: SimpleNamespace(aclgraph=graph)}
    )
    wrapper = SimpleNamespace(
        vllm_config=config,
        runtime_mode=CUDAGraphMode.FULL,
        concrete_aclgraph_entries=entries,
    )
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=descriptor,
    )

    def run_graph(_wrapper, *_args, **_kwargs):
        if action == "capture":
            entries[descriptor] = SimpleNamespace(aclgraph=graph)
        return "result"

    with (
        patch.dict(
            os.environ,
            {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output)},
            clear=False,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_original_acl_graph_call_310",
            side_effect=run_graph,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_is_draft_graph_path",
            return_value=path == "draft",
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "assert_dflash_graph_tensor_addresses_310"
        ) as assert_addresses,
    ):
        _reset_dflash_diagnostics_for_test()
        remember_dflash_graph_modes(
            config,
            requested_mode=CUDAGraphMode.FULL,
            normalized_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        result = observe_dflash_acl_graph_call_310(wrapper, torch.ones(1))
    _reset_dflash_diagnostics_for_test()

    assert result == "result"
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["stage"] == f"graph_{action}_{path}"
    assert record["path"] == path
    assert record["requested_mode"] == "FULL"
    assert record["normalized_mode"] == "FULL_DECODE_ONLY"
    assert record["runtime_mode"] == "FULL"
    assert record["capture_descriptor"]["num_tokens"] == 32
    assert record["capture_occurred"] is (action == "capture")
    assert record["replay_occurred"] is (action == "replay")
    assert_addresses.assert_called_once()
    assert assert_addresses.call_args.kwargs["path"] == path
    assert assert_addresses.call_args.kwargs["action"] == action


def test_disabled_acl_graph_observer_does_not_inspect_forward_context():
    wrapper = SimpleNamespace(vllm_config=_dflash_graph_config())
    context = MagicMock(side_effect=AssertionError("disabled observer must stay lazy"))
    original = MagicMock(return_value="result")

    with (
        patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_original_acl_graph_call_310",
            original,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "get_forward_context",
            context,
        ),
    ):
        _reset_dflash_diagnostics_for_test()
        result = observe_dflash_acl_graph_call_310(wrapper, "arg", key="value")
    _reset_dflash_diagnostics_for_test()

    assert result == "result"
    original.assert_called_once_with(wrapper, "arg", key="value")
    context.assert_not_called()


def test_enabled_graph_observer_bypasses_dspark_before_context_read(
    tmp_path: Path,
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark"),
    )
    wrapper = SimpleNamespace(vllm_config=config)
    context = MagicMock(
        side_effect=AssertionError("DSpark must not enter DFlash diagnostics")
    )
    original = MagicMock(return_value="result")

    with (
        patch.dict(
            os.environ,
            {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(tmp_path / "dspark.jsonl")},
            clear=False,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_original_acl_graph_call_310",
            original,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "get_forward_context",
            context,
        ),
    ):
        _reset_dflash_diagnostics_for_test()
        result = observe_dflash_acl_graph_call_310(
            wrapper,
            "arg",
            key="value",
        )
    _reset_dflash_diagnostics_for_test()

    assert result == "result"
    original.assert_called_once_with(wrapper, "arg", key="value")
    context.assert_not_called()


def test_graph_replay_asserts_all_persistent_dflash_tensor_addresses(
    tmp_path: Path,
):
    config = _dflash_graph_config()
    descriptor = _graph_descriptor()
    wrapper = object()
    tensors = {
        "input.input_ids": torch.zeros(8),
        "position.positions": torch.zeros(8),
        "slot.query": torch.zeros(8),
        "block_table.layer": torch.zeros((2, 4)),
        "recurrent_state.layer": torch.zeros((2, 4)),
        "convolution_state.layer": torch.zeros((2, 4)),
        "rejection_metadata.accepted": torch.zeros(2),
    }
    with patch.dict(
        os.environ,
        {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(tmp_path / "graph.jsonl")},
        clear=False,
    ):
        _reset_dflash_diagnostics_for_test()
        assert_dflash_graph_tensor_addresses_310(
            config,
            path="target",
            wrapper=wrapper,
            batch_descriptor=descriptor,
            action="capture",
            tensors=tensors,
        )
        assert_dflash_graph_tensor_addresses_310(
            config,
            path="target",
            wrapper=wrapper,
            batch_descriptor=descriptor,
            action="replay",
            tensors=tensors,
        )

        changed = dict(tensors)
        changed["position.positions"] = tensors["position.positions"].clone()
        with pytest.raises(AssertionError, match="position.positions"):
            assert_dflash_graph_tensor_addresses_310(
                config,
                path="target",
                wrapper=wrapper,
                batch_descriptor=descriptor,
                action="replay",
                tensors=changed,
            )
    _reset_dflash_diagnostics_for_test()


def test_graph_observer_rejects_changed_positional_input_without_debug_logging(
    tmp_path: Path,
):
    config = _dflash_graph_config()
    descriptor = _graph_descriptor()
    entries = {}
    wrapper = SimpleNamespace(
        vllm_config=config,
        runtime_mode=CUDAGraphMode.PIECEWISE,
        concrete_aclgraph_entries=entries,
    )
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=descriptor,
        attn_metadata={},
        no_compile_layers={},
    )

    def run_graph(_wrapper, *_args, **_kwargs):
        entries.setdefault(descriptor, SimpleNamespace(aclgraph=object()))
        return "result"

    with (
        patch.dict(
            os.environ,
            {
                "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(tmp_path / "args.jsonl"),
                "VLLM_LOGGING_LEVEL": "INFO",
            },
            clear=False,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_original_acl_graph_call_310",
            side_effect=run_graph,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_is_draft_graph_path",
            return_value=False,
        ),
    ):
        _reset_dflash_diagnostics_for_test()
        try:
            graph_input = torch.zeros(8)
            assert observe_dflash_acl_graph_call_310(wrapper, graph_input) == "result"
            with pytest.raises(AssertionError, match=r"argument\.0"):
                observe_dflash_acl_graph_call_310(wrapper, graph_input.clone())
        finally:
            _reset_dflash_diagnostics_for_test()


def test_piecewise_observer_ignores_metadata_added_after_dummy_capture(
    tmp_path: Path,
):
    """PIECEWISE graphs own their call arguments, not outer metadata."""
    config = _dflash_graph_config()
    descriptor = _graph_descriptor()
    entries = {}
    wrapper = SimpleNamespace(
        vllm_config=config,
        runtime_mode=CUDAGraphMode.PIECEWISE,
        concrete_aclgraph_entries=entries,
    )
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=descriptor,
        attn_metadata={},
        no_compile_layers={},
    )

    def run_graph(_wrapper, *_args, **_kwargs):
        entries.setdefault(descriptor, SimpleNamespace(aclgraph=object()))
        return "result"

    with (
        patch.dict(
            os.environ,
            {
                "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(
                    tmp_path / "piecewise-lifecycle.jsonl"
                )
            },
            clear=False,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_original_acl_graph_call_310",
            side_effect=run_graph,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_is_draft_graph_path",
            return_value=False,
        ),
    ):
        _reset_dflash_diagnostics_for_test()
        try:
            graph_input = torch.zeros(8)
            keyword_input = torch.ones(8)
            assert observe_dflash_acl_graph_call_310(
                wrapper,
                graph_input,
                scale=keyword_input,
            ) == "result"
            forward_context.attn_metadata = {
                "model.layers.0.attn": SimpleNamespace(
                    seq_lens=torch.ones(1, dtype=torch.int32),
                    query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                    slot_mapping=torch.zeros(1, dtype=torch.int32),
                    block_tables=torch.zeros((1, 1), dtype=torch.int32),
                )
            }
            assert observe_dflash_acl_graph_call_310(
                wrapper,
                graph_input,
                scale=keyword_input,
            ) == "result"
            with pytest.raises(AssertionError, match=r"keyword\.scale"):
                observe_dflash_acl_graph_call_310(
                    wrapper,
                    graph_input,
                    scale=keyword_input.clone(),
                )
        finally:
            _reset_dflash_diagnostics_for_test()


@pytest.mark.parametrize(
    ("changed_field", "expected_name"),
    [
        ("per_layer_slot", r"slot\.query_by_layer\.1"),
        ("context_hidden_states", r"input\.context_hidden_states"),
    ],
)
def test_graph_observer_rejects_changed_draft_persistent_input(
    tmp_path: Path,
    changed_field: str,
    expected_name: str,
):
    class DraftOwner:
        method = "dflash"

        def __init__(self):
            self.input_ids = torch.zeros(32, dtype=torch.int32)
            self._dflash_hidden_states = torch.zeros((32, 64))
            self.positions = torch.zeros(32, dtype=torch.int32)
            self._context_positions_buffer = torch.zeros(32, dtype=torch.int32)
            self._slot_mapping_buffer = torch.zeros(32, dtype=torch.int32)
            self._context_slot_mapping_buffer = torch.zeros(32, dtype=torch.int32)
            self._dflash_query_slot_mapping_by_layer_310 = [
                torch.zeros(32, dtype=torch.int32),
                torch.zeros(32, dtype=torch.int32),
            ]
            self._dflash_context_slot_mapping_by_layer_310 = [
                torch.zeros(32, dtype=torch.int32),
                torch.zeros(32, dtype=torch.int32),
            ]
            self._dflash_block_table_by_layer_310 = [
                torch.zeros((2, 4), dtype=torch.int32),
                torch.zeros((2, 8), dtype=torch.int32),
            ]

        def run(self):
            return "result"

    config = _dflash_graph_config()
    descriptor = _graph_descriptor()
    owner = DraftOwner()
    entries = {}
    wrapper = SimpleNamespace(
        vllm_config=config,
        runtime_mode=CUDAGraphMode.FULL,
        concrete_aclgraph_entries=entries,
        runnable=owner.run,
    )
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=descriptor,
        attn_metadata={},
        no_compile_layers={},
    )

    def run_graph(_wrapper, *_args, **_kwargs):
        entries.setdefault(descriptor, SimpleNamespace(aclgraph=object()))
        return "result"

    with (
        patch.dict(
            os.environ,
            {
                "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(
                    tmp_path / "draft-slot.jsonl"
                )
            },
            clear=False,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_original_acl_graph_call_310",
            side_effect=run_graph,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310."
            "_is_draft_graph_path",
            return_value=True,
        ),
    ):
        _reset_dflash_diagnostics_for_test()
        try:
            assert observe_dflash_acl_graph_call_310(wrapper) == "result"
            if changed_field == "per_layer_slot":
                owner._dflash_query_slot_mapping_by_layer_310[1] = (
                    owner._dflash_query_slot_mapping_by_layer_310[1].clone()
                )
            else:
                owner._dflash_hidden_states = owner._dflash_hidden_states.clone()
            with pytest.raises(AssertionError, match=expected_name):
                observe_dflash_acl_graph_call_310(wrapper)
        finally:
            _reset_dflash_diagnostics_for_test()


@pytest.mark.parametrize(
    "changed_field",
    [
        "seq_lens",
        "query_start_loc",
        "attn_mask",
        "has_initial_state",
        "spec_query_start_loc",
        "non_spec_query_start_loc",
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "spec_sequence_masks",
        "spec_token_indx",
        "non_spec_token_indx",
        "num_accepted_tokens",
        "spec_decode_metadata.spec_causal_conv1d.query_start_loc",
        "spec_decode_metadata.spec_causal_conv1d.cache_indices",
        "spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens",
    ],
)
def test_graph_address_collection_rejects_changed_attention_control_buffer(
    changed_field: str,
):
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([32, 48], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 16, 32], dtype=torch.int32),
        attn_mask=torch.zeros((32, 32), dtype=torch.bool),
        has_initial_state=torch.zeros(2, dtype=torch.bool),
        spec_state_indices_tensor=torch.zeros((2, 16), dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(2, dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, 16, 32], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0], dtype=torch.int32),
        spec_sequence_masks=torch.ones((2, 16), dtype=torch.bool),
        spec_token_indx=torch.arange(32, dtype=torch.int32),
        non_spec_token_indx=torch.arange(2, dtype=torch.int32),
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=torch.tensor([0, 16, 32], dtype=torch.int32),
                cache_indices=torch.zeros((2, 16), dtype=torch.int32),
                num_accepted_tokens=torch.ones(2, dtype=torch.int32),
            ),
        ),
    )
    forward_context = SimpleNamespace(
        attn_metadata={"model.layers.0.gdn": metadata},
        no_compile_layers={},
    )
    config = _dflash_graph_config()
    descriptor = _graph_descriptor()
    wrapper = SimpleNamespace(runnable=MagicMock())

    _reset_dflash_diagnostics_for_test()
    try:
        captured = collect_dflash_graph_tensors_310(
            wrapper,
            forward_context=forward_context,
            path="target",
            args=(),
            kwargs={},
        )
        assert_dflash_graph_tensor_addresses_310(
            config,
            path="target",
            wrapper=wrapper,
            batch_descriptor=descriptor,
            action="capture",
            tensors=captured,
        )

        owner = metadata
        field_parts = changed_field.split(".")
        for field_name in field_parts[:-1]:
            owner = getattr(owner, field_name)
        leaf_name = field_parts[-1]
        setattr(owner, leaf_name, getattr(owner, leaf_name).clone())
        replay = collect_dflash_graph_tensors_310(
            wrapper,
            forward_context=forward_context,
            path="target",
            args=(),
            kwargs={},
        )
        with pytest.raises(AssertionError, match=leaf_name):
            assert_dflash_graph_tensor_addresses_310(
                config,
                path="target",
                wrapper=wrapper,
                batch_descriptor=descriptor,
                action="replay",
                tensors=replay,
            )
    finally:
        _reset_dflash_diagnostics_for_test()


def test_graph_collection_excludes_nonconsumed_310p_gdn_nested_buffers():
    metadata = SimpleNamespace(
        spec_state_indices_tensor=torch.zeros((2, 16), dtype=torch.int32),
        spec_decode_metadata=SimpleNamespace(
            actual_seq_lengths=torch.tensor([16, 16], dtype=torch.int32),
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=torch.tensor([0, 16, 32], dtype=torch.int32),
                cache_indices=torch.zeros((2, 16), dtype=torch.int32),
                num_accepted_tokens=torch.ones(2, dtype=torch.int32),
            ),
        ),
        non_spec_decode_metadata=SimpleNamespace(
            actual_seq_lengths=torch.tensor([1, 1], dtype=torch.int32),
        ),
        nums_dict={"unused": torch.ones(1)},
    )
    tensors = collect_dflash_graph_tensors_310(
        SimpleNamespace(runnable=MagicMock()),
        forward_context=SimpleNamespace(
            attn_metadata={"model.layers.0.gdn": metadata},
            no_compile_layers={},
        ),
        path="target",
        args=(),
        kwargs={},
    )

    names = set(tensors)
    assert not any("actual_seq_lengths" in name for name in names)
    assert not any("non_spec_decode_metadata" in name for name in names)
    assert not any("nums_dict" in name for name in names)


@pytest.mark.parametrize("path", ["target", "draft"])
def test_graph_tensor_collection_covers_required_dflash_state(path: str):
    conv_state = torch.zeros((4, 3))
    recurrent_state = torch.zeros((4, 5))
    metadata = SimpleNamespace(
        spec_state_indices_tensor=torch.zeros((2, 16), dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        block_tables=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(32, dtype=torch.int32),
    )
    forward_context = SimpleNamespace(
        attn_metadata={"model.layers.0.attn": metadata},
        no_compile_layers={
            "model.layers.0.attn": SimpleNamespace(
                kv_cache=[conv_state, recurrent_state]
            )
        },
    )

    if path == "draft":
        class DraftOwner:
            method = "dflash"
            input_ids = torch.zeros(32, dtype=torch.int32)
            positions = torch.zeros(32, dtype=torch.int32)
            _context_positions_buffer = torch.zeros(32, dtype=torch.int32)
            _slot_mapping_buffer = torch.zeros(32, dtype=torch.int32)
            _context_slot_mapping_buffer = torch.zeros(32, dtype=torch.int32)
            _dflash_block_table_by_layer_310 = [
                torch.zeros((2, 4), dtype=torch.int32)
            ]

            def run(self):
                return None

        owner = DraftOwner()
        wrapper = SimpleNamespace(runnable=owner.run)
        kwargs = {"token_indices_to_sample": torch.zeros(2, dtype=torch.int32)}
    else:
        wrapper = SimpleNamespace(runnable=MagicMock())
        kwargs = {
            "input_ids": torch.zeros(32, dtype=torch.int32),
            "positions": torch.zeros(32, dtype=torch.int32),
        }

    tensors = collect_dflash_graph_tensors_310(
        wrapper,
        forward_context=forward_context,
        path=path,
        args=(),
        kwargs=kwargs,
    )
    categories = {name.split(".", 1)[0] for name in tensors}
    assert categories == {
        "input",
        "position",
        "slot",
        "block_table",
        "recurrent_state",
        "convolution_state",
        "rejection_metadata",
    }
    assert tensors["slot.model.layers.0.attn"] is metadata.slot_mapping
    if path == "draft":
        assert (
            tensors["position.context_positions"]
            is owner._context_positions_buffer
        )


def test_310p_patch_installs_acl_graph_observer():
    patch_source = (
        Path(__file__).resolve().parents[4]
        / "vllm_ascend"
        / "patch"
        / "worker"
        / "patch_idex_310.py"
    ).read_text(encoding="utf-8")

    guard = patch_source.index("if dflash_diagnostic_enabled():")
    install = patch_source.index(
        "ACLGraphWrapper.__call__ = observe_dflash_acl_graph_call_310"
    )
    assert guard < install


def test_draft_wrapper_captures_returned_tokens_only_for_dflash():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.method = "dflash"
    proposer.vllm_config = _dflash_graph_config()
    result = torch.tensor([[11, 12, 13]], dtype=torch.int32)
    token_indices = torch.tensor([3])

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft",
            return_value=result,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.dflash_diagnostic_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.capture_dflash_diagnostic"
        ) as capture,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "capture_current_dflash_graph_dispatch"
        ) as capture_graph_dispatch,
    ):
        actual = proposer._run_merged_draft(
            4,
            1,
            token_indices,
            torch.arange(4),
            None,
            [SimpleNamespace()],
            4,
        )

    torch.testing.assert_close(actual, result)
    capture_graph_dispatch.assert_called_once_with(
        proposer.vllm_config,
        path="draft",
    )
    capture.assert_called_once()
    assert capture.call_args.args == ("draft_output",)
    payload = capture.call_args.kwargs["payload_builder"]()
    torch.testing.assert_close(payload["draft_token_ids"], result)
    torch.testing.assert_close(payload["token_indices_to_sample"], token_indices)


def test_draft_wrapper_skips_returned_tokens_during_spec_dummy_capture():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.method = "dflash"
    proposer.vllm_config = _dflash_graph_config()
    proposer.runner = SimpleNamespace(_spec_dummy_capture=True)
    result = torch.tensor([[11, 12, 13]], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "_original_run_merged_draft",
            return_value=result,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "dflash_diagnostic_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "capture_dflash_diagnostic"
        ) as capture,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "capture_current_dflash_graph_dispatch"
        ) as capture_graph_dispatch,
    ):
        actual = proposer._run_merged_draft(
            4,
            1,
            torch.tensor([3]),
            torch.arange(4),
            None,
            [SimpleNamespace()],
            4,
        )

    torch.testing.assert_close(actual, result)
    capture_graph_dispatch.assert_called_once_with(
        proposer.vllm_config,
        path="draft",
    )
    capture.assert_not_called()


def test_draft_wrapper_does_not_capture_for_dspark():
    class UnexpectedMarkerRead:
        @property
        def _spec_dummy_capture(self):
            raise AssertionError("DSpark must not inspect DFlash capture state")

    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.method = "dspark"
    proposer.runner = UnexpectedMarkerRead()
    result = torch.tensor([[11, 12, 13]], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft",
            return_value=result,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.dflash_diagnostic_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.capture_dflash_diagnostic"
        ) as capture,
    ):
        actual = proposer._run_merged_draft(
            4,
            1,
            torch.tensor([3]),
            torch.arange(4),
            None,
            [SimpleNamespace()],
            4,
        )

    torch.testing.assert_close(actual, result)
    capture.assert_not_called()

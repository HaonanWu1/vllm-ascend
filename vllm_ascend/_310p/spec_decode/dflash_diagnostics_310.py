# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded diagnostics for 310P DFlash correctness isolation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

_PATH_ENV = "ASCEND_DFLASH_DIAGNOSTIC_PATH"
_LIMIT_ENV = "ASCEND_DFLASH_DIAGNOSTIC_LIMIT"
_MAX_ELEMENTS_ENV = "ASCEND_DFLASH_DIAGNOSTIC_MAX_ELEMENTS"
_DEFAULT_LIMIT = 4
_DEFAULT_MAX_ELEMENTS = 4096

_lock = threading.Lock()
_config: tuple[Path | None, int, int] | None = None
_stage_counts: dict[str, int] = {}
_graph_modes: dict[int, tuple[CUDAGraphMode, CUDAGraphMode]] = {}
_graph_tensor_addresses: dict[
    tuple[int, str, int, tuple[Any, ...]],
    dict[str, int],
] = {}
_original_acl_graph_call_310 = ACLGraphWrapper.__call__


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default))), 0)
    except ValueError:
        return default


def _get_config() -> tuple[Path | None, int, int]:
    global _config
    if _config is None:
        raw_path = os.getenv(_PATH_ENV, "").strip()
        _config = (
            Path(raw_path) if raw_path else None,
            _positive_int_env(_LIMIT_ENV, _DEFAULT_LIMIT),
            _positive_int_env(_MAX_ELEMENTS_ENV, _DEFAULT_MAX_ELEMENTS),
        )
    return _config


def dflash_diagnostic_enabled() -> bool:
    path, limit, _ = _get_config()
    return path is not None and limit > 0


def _json_value(value: Any, max_elements: int) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach()
        shape = list(tensor.shape)
        result = {
            "dtype": str(tensor.dtype),
            "shape": shape,
        }
        if tensor.numel() <= max_elements:
            result["values"] = tensor.cpu().tolist()
        else:
            result["values"] = tensor.flatten()[:max_elements].cpu().tolist()
            result["truncated"] = True
            result["numel"] = tensor.numel()
        return result
    if isinstance(value, dict):
        return {str(key): _json_value(item, max_elements) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, max_elements) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def capture_dflash_diagnostic(
    stage: str,
    payload_builder: Callable[[], dict[str, Any] | None] | None = None,
    **payload: Any,
) -> None:
    """Append at most ``LIMIT`` records per stage when explicitly enabled.

    The disabled path returns before inspecting tensors, so normal execution
    performs no device synchronization or file I/O.
    """
    path, limit, max_elements = _get_config()
    if path is None or limit <= 0:
        return

    with _lock:
        occurrence = _stage_counts.get(stage, 0)
        if occurrence >= limit:
            return
        _stage_counts[stage] = occurrence + 1
        try:
            if payload_builder is not None:
                built_payload = payload_builder()
                if built_payload is None:
                    _stage_counts[stage] = occurrence
                    return
                payload = {**payload, **built_payload}
            record = {
                "stage": stage,
                "occurrence": occurrence,
                **{
                    key: _json_value(value, max_elements)
                    for key, value in payload.items()
                },
            }
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, encoded)
            finally:
                os.close(fd)
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("310P DFlash diagnostic capture failed: %s", exc)


def remember_dflash_graph_modes(
    vllm_config: Any,
    *,
    requested_mode: CUDAGraphMode,
    normalized_mode: CUDAGraphMode,
) -> None:
    """Retain the pre/post-normalization modes for opt-in graph evidence."""
    if not dflash_diagnostic_enabled():
        return
    with _lock:
        config_id = id(vllm_config)
        original_request = _graph_modes.get(
            config_id,
            (requested_mode, normalized_mode),
        )[0]
        _graph_modes[config_id] = (original_request, normalized_mode)


def _mode_name(mode: Any) -> str:
    return getattr(mode, "name", str(mode))


def _graph_mode_names(vllm_config: Any) -> tuple[str, str]:
    modes = _graph_modes.get(id(vllm_config))
    if modes is None:
        normalized_mode = getattr(
            getattr(vllm_config, "compilation_config", None),
            "cudagraph_mode",
            CUDAGraphMode.NONE,
        )
        modes = (normalized_mode, normalized_mode)
    return _mode_name(modes[0]), _mode_name(modes[1])


def _capture_descriptor_payload(
    batch_descriptor: BatchDescriptor | None,
) -> dict[str, Any] | None:
    if batch_descriptor is None:
        return None
    return {
        "num_tokens": batch_descriptor.num_tokens,
        "num_reqs": batch_descriptor.num_reqs,
        "uniform": batch_descriptor.uniform,
        "has_lora": batch_descriptor.has_lora,
        "num_active_loras": batch_descriptor.num_active_loras,
    }


def _graph_event_payload(
    vllm_config: Any,
    *,
    path: str,
    runtime_mode: CUDAGraphMode,
    batch_descriptor: BatchDescriptor | None,
    capture_occurred: bool,
    replay_occurred: bool,
    actual_num_tokens: int | None = None,
) -> dict[str, Any]:
    requested_mode, normalized_mode = _graph_mode_names(vllm_config)
    payload = {
        "path": path,
        "requested_mode": requested_mode,
        "normalized_mode": normalized_mode,
        "runtime_mode": _mode_name(runtime_mode),
        "capture_descriptor": _capture_descriptor_payload(batch_descriptor),
        "capture_occurred": capture_occurred,
        "replay_occurred": replay_occurred,
    }
    if actual_num_tokens is not None:
        payload["actual_num_tokens"] = actual_num_tokens
    return payload


def capture_dflash_graph_dispatch(
    vllm_config: Any,
    *,
    path: str,
    runtime_mode: CUDAGraphMode,
    batch_descriptor: BatchDescriptor | None,
    actual_num_tokens: int | None = None,
) -> None:
    """Record one target/draft dispatch without inspecting data when disabled."""
    capture_dflash_diagnostic(
        f"graph_dispatch_{path}",
        payload_builder=lambda: _graph_event_payload(
            vllm_config,
            path=path,
            runtime_mode=runtime_mode,
            batch_descriptor=batch_descriptor,
            capture_occurred=False,
            replay_occurred=False,
            actual_num_tokens=actual_num_tokens,
        ),
    )


def capture_current_dflash_graph_dispatch(
    vllm_config: Any,
    *,
    path: str,
) -> None:
    """Record the dispatch carried by the active Ascend forward context."""

    def _payload() -> dict[str, Any]:
        forward_context = get_forward_context()
        return _graph_event_payload(
            vllm_config,
            path=path,
            runtime_mode=forward_context.cudagraph_runtime_mode,
            batch_descriptor=forward_context.batch_descriptor,
            capture_occurred=False,
            replay_occurred=False,
        )

    capture_dflash_diagnostic(
        f"graph_dispatch_{path}",
        payload_builder=_payload,
    )


def _is_dflash_config(vllm_config: Any) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return getattr(speculative_config, "method", None) == "dflash"


def _is_draft_graph_path() -> bool:
    return bool(_EXTRA_CTX.is_draft_model)


def _add_graph_tensors(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    value: Any,
) -> None:
    if torch.is_tensor(value):
        tensors[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _add_graph_tensors(tensors, f"{prefix}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _add_graph_tensors(tensors, f"{prefix}.{index}", item)


_ATTENTION_CONTROL_TENSOR_FIELDS = (
    "attn_mask",
    "seq_lens",
    "query_start_loc",
)


def _add_attention_control_tensors(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    metadata: Any,
) -> None:
    """Collect tensor metadata directly consumed by standard attention."""
    for field in _ATTENTION_CONTROL_TENSOR_FIELDS:
        _add_graph_tensors(
            tensors,
            f"{prefix}.{field}",
            getattr(metadata, field, None),
        )


_GDN_CONTROL_TENSOR_FIELDS = (
    "has_initial_state",
    "spec_query_start_loc",
    "non_spec_query_start_loc",
    "spec_state_indices_tensor",
    "non_spec_state_indices_tensor",
    "spec_sequence_masks",
    "spec_token_indx",
    "non_spec_token_indx",
    "num_accepted_tokens",
)
_GDN_SPEC_CAUSAL_CONV_TENSOR_FIELDS = (
    "query_start_loc",
    "cache_indices",
    "num_accepted_tokens",
)


def _add_gdn_control_tensors(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    metadata: Any,
) -> None:
    """Collect only tensor fields read by the 310P GDN forward path."""
    for field in _GDN_CONTROL_TENSOR_FIELDS:
        _add_graph_tensors(
            tensors,
            f"{prefix}.{field}",
            getattr(metadata, field, None),
        )

    spec_decode_metadata = getattr(metadata, "spec_decode_metadata", None)
    if spec_decode_metadata is None:
        return
    spec_causal_conv = getattr(
        spec_decode_metadata,
        "spec_causal_conv1d",
        None,
    )
    if spec_causal_conv is None:
        return
    for field in _GDN_SPEC_CAUSAL_CONV_TENSOR_FIELDS:
        _add_graph_tensors(
            tensors,
            f"{prefix}.spec_decode_metadata.spec_causal_conv1d.{field}",
            getattr(spec_causal_conv, field, None),
        )


def _iter_graph_attention_metadata(
    attn_metadata: Any,
):
    if isinstance(attn_metadata, dict):
        yield from attn_metadata.items()
    elif isinstance(attn_metadata, list):
        for ubatch_index, ubatch_metadata in enumerate(attn_metadata):
            if isinstance(ubatch_metadata, dict):
                for layer_name, metadata in ubatch_metadata.items():
                    yield f"ubatch{ubatch_index}.{layer_name}", metadata


def collect_dflash_graph_tensors_310(
    wrapper: Any,
    *,
    forward_context: Any,
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Collect graph-visible persistent tensors without reading their values."""
    tensors: dict[str, torch.Tensor] = {}
    for index, value in enumerate(args):
        _add_graph_tensors(tensors, f"argument.{index}", value)

    # A PIECEWISE wrapper captures one compiler-produced region. Its complete
    # graph contract is carried by the region's explicit call arguments;
    # forward-context metadata belongs to split-out operations and can be
    # populated only after dummy capture. FULL wraps the model/proposer call
    # and must additionally validate the persistent owner/context buffers.
    if getattr(wrapper, "runtime_mode", None) == CUDAGraphMode.PIECEWISE:
        for name, value in kwargs.items():
            _add_graph_tensors(tensors, f"keyword.{name}", value)
        return tensors

    if path == "target":
        _add_graph_tensors(tensors, "input.input_ids", kwargs.get("input_ids"))
        _add_graph_tensors(tensors, "position.positions", kwargs.get("positions"))
    else:
        owner = getattr(getattr(wrapper, "runnable", None), "__self__", None)
        if getattr(owner, "method", None) == "dflash":
            _add_graph_tensors(tensors, "input.input_ids", getattr(owner, "input_ids", None))
            _add_graph_tensors(
                tensors,
                "input.context_hidden_states",
                getattr(owner, "_dflash_hidden_states", None),
            )
            for name in ("positions", "mrope_positions", "xdrope_positions"):
                value = getattr(owner, name, None)
                if value is not None:
                    _add_graph_tensors(tensors, f"position.{name}", value)
                    break
            _add_graph_tensors(
                tensors,
                "position.context_positions",
                getattr(owner, "_context_positions_buffer", None),
            )
            _add_graph_tensors(
                tensors,
                "slot.query",
                getattr(owner, "_slot_mapping_buffer", None),
            )
            _add_graph_tensors(
                tensors,
                "slot.context",
                getattr(owner, "_context_slot_mapping_buffer", None),
            )
            _add_graph_tensors(
                tensors,
                "slot.query_by_layer",
                getattr(owner, "_dflash_query_slot_mapping_by_layer_310", None),
            )
            _add_graph_tensors(
                tensors,
                "slot.context_by_layer",
                getattr(owner, "_dflash_context_slot_mapping_by_layer_310", None),
            )
            _add_graph_tensors(
                tensors,
                "block_table.draft_layout",
                getattr(owner, "_dflash_block_table_by_layer_310", None),
            )

    _add_graph_tensors(
        tensors,
        "rejection_metadata.token_indices_to_sample",
        kwargs.get("token_indices_to_sample"),
    )

    no_compile_layers = getattr(forward_context, "no_compile_layers", {})
    for layer_name, metadata in _iter_graph_attention_metadata(
        getattr(forward_context, "attn_metadata", None)
    ):
        _add_attention_control_tensors(
            tensors,
            f"input.attention_metadata.{layer_name}",
            metadata,
        )
        _add_graph_tensors(
            tensors,
            f"slot.{layer_name}",
            getattr(metadata, "slot_mapping", None),
        )
        for field in ("block_tables", "block_table_tensor"):
            _add_graph_tensors(
                tensors,
                f"block_table.{layer_name}.{field}",
                getattr(metadata, field, None),
            )
        has_state_metadata = any(
            getattr(metadata, field, None) is not None
            for field in ("spec_state_indices_tensor", "non_spec_state_indices_tensor")
        )
        if has_state_metadata:
            _add_gdn_control_tensors(
                tensors,
                f"rejection_metadata.{layer_name}",
                metadata,
            )
        base_layer_name = layer_name.split(".", 1)[1] if layer_name.startswith("ubatch") else layer_name
        layer = no_compile_layers.get(base_layer_name)
        kv_cache = getattr(layer, "kv_cache", None)
        if has_state_metadata and isinstance(kv_cache, (list, tuple)):
            if len(kv_cache) > 0:
                _add_graph_tensors(
                    tensors,
                    f"convolution_state.{layer_name}",
                    kv_cache[0],
                )
            if len(kv_cache) > 1:
                _add_graph_tensors(
                    tensors,
                    f"recurrent_state.{layer_name}",
                    kv_cache[1],
                )

    return tensors


def _batch_descriptor_address_key(
    batch_descriptor: BatchDescriptor,
) -> tuple[Any, ...]:
    return (
        batch_descriptor.num_tokens,
        batch_descriptor.num_reqs,
        batch_descriptor.uniform,
        batch_descriptor.has_lora,
        batch_descriptor.num_active_loras,
    )


def assert_dflash_graph_tensor_addresses_310(
    vllm_config: Any,
    *,
    path: str,
    wrapper: Any,
    batch_descriptor: BatchDescriptor,
    action: str,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Remember capture addresses and assert exact identity before replay."""
    key = (
        id(vllm_config),
        path,
        id(wrapper),
        _batch_descriptor_address_key(batch_descriptor),
    )
    addresses = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    with _lock:
        if action == "capture":
            _graph_tensor_addresses[key] = addresses
            return
        if action != "replay":
            return
        expected = _graph_tensor_addresses.get(key)

    assert expected is not None, (
        f"310P DFlash {path} graph replay has no captured address baseline"
    )
    changed = {
        name: (expected.get(name), addresses.get(name))
        for name in expected.keys() | addresses.keys()
        if expected.get(name) != addresses.get(name)
    }
    assert not changed, (
        f"310P DFlash {path} graph persistent tensor addresses changed "
        f"during replay: {changed}"
    )


def observe_dflash_acl_graph_call_310(self: ACLGraphWrapper, *args: Any, **kwargs: Any) -> Any:
    """Observe successful 310P DFlash ACL graph captures and replays.

    This function is installed as a 310P-only class patch. The normal path
    returns before reading forward context, and the wrapped implementation
    remains the sole owner of graph behavior.
    """
    if not _is_dflash_config(self.vllm_config) or not dflash_diagnostic_enabled():
        return _original_acl_graph_call_310(self, *args, **kwargs)

    try:
        forward_context = get_forward_context()
        runtime_mode = forward_context.cudagraph_runtime_mode
        batch_descriptor = forward_context.batch_descriptor
        eligible = (
            runtime_mode != CUDAGraphMode.NONE
            and runtime_mode == self.runtime_mode
            and batch_descriptor is not None
        )
        entry = (
            self.concrete_aclgraph_entries.get(batch_descriptor)
            if eligible
            else None
        )
        action = None
        if eligible:
            action = "capture" if entry is None or entry.aclgraph is None else "replay"
    except Exception:  # noqa: BLE001
        # Diagnostics must never make an otherwise valid graph call fail.
        return _original_acl_graph_call_310(self, *args, **kwargs)

    path = "draft" if _is_draft_graph_path() else "target"
    graph_tensors: dict[str, torch.Tensor] | None = None
    if action is not None:
        try:
            graph_tensors = collect_dflash_graph_tensors_310(
                self,
                forward_context=forward_context,
                path=path,
                args=args,
                kwargs=kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning_once(
                "310P DFlash graph address collection failed: %s",
                exc,
            )
        if action == "replay" and graph_tensors is not None:
            assert_dflash_graph_tensor_addresses_310(
                self.vllm_config,
                path=path,
                wrapper=self,
                batch_descriptor=batch_descriptor,
                action=action,
                tensors=graph_tensors,
            )

    result = _original_acl_graph_call_310(self, *args, **kwargs)
    if action is None:
        return result

    if action == "capture":
        captured_entry = self.concrete_aclgraph_entries.get(batch_descriptor)
        if captured_entry is None or captured_entry.aclgraph is None:
            return result
        if graph_tensors is not None:
            assert_dflash_graph_tensor_addresses_310(
                self.vllm_config,
                path=path,
                wrapper=self,
                batch_descriptor=batch_descriptor,
                action=action,
                tensors=graph_tensors,
            )

    capture_dflash_diagnostic(
        f"graph_{action}_{path}",
        payload_builder=lambda: _graph_event_payload(
            self.vllm_config,
            path=path,
            runtime_mode=runtime_mode,
            batch_descriptor=batch_descriptor,
            capture_occurred=action == "capture",
            replay_occurred=action == "replay",
        ),
    )
    return result


def _reset_dflash_diagnostics_for_test() -> None:
    global _config
    with _lock:
        _config = None
        _stage_counts.clear()
        _graph_modes.clear()
        _graph_tensor_addresses.clear()

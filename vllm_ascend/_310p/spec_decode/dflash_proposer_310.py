#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""310P dflash/dspark input construction.

310P has no Triton, so the dflash/dspark ``set_inputs_first_pass`` overrides
here build the draft-model inputs via the AscendC custom op
``npu_copy_and_expand_dflash_inputs`` instead of the Triton
``copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid`` used on other
platforms. These classes are wired onto the shared proposers by
``vllm_ascend.patch.worker.patch_idex_310`` so the generic spec-decode modules
stay free of any 310P coupling.
"""

import functools
import inspect
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend._310p.attention.metadata_builder import (
    dflash_hybrid_draft_capture_scope_310,
)
from vllm_ascend._310p.dflash_full_and_piecewise import (
    get_310p_dflash_graph_capabilities,
    is_310p_dflash_effective_piecewise,
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend._310p.dflash_full_decode_only import (
    is_310p_dflash_full_decode_only,
)
from vllm_ascend._310p.dflash_piecewise import is_310p_dflash_piecewise
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendRotaryEmbedding310,
    configure_draft_rope_capacity_310,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer

# The 310P recurrent GDN kernel reserves buffers for a maximum recurrent query
# length of 16. DFlash verification prepends one target bonus token, leaving
# room for up to 15 draft tokens. Real-weight Qwen3.5-9B runs and isolated
# CausalConv1d/GDN comparisons cover K=6, K=8 and K=15.
MAX_SUPPORTED_NUM_SPEC_TOKENS_310P = 15


def _uses_int32_draft_address_math_310(vllm_config: Any) -> bool:
    """Avoid 310P's unaligned dynamic-int64 address arithmetic in graphs."""
    return is_310p_dflash_full_decode_only(vllm_config) or is_310p_dflash_full_and_piecewise(vllm_config)


def _uses_piecewise_persistent_buffers(vllm_config: Any) -> bool:
    """Use PIECEWISE buffers only for an effective PIECEWISE round.

    Pure PIECEWISE keeps its historical behavior. Hybrid mode must consult the
    current forward context instead of inferring runtime state from the
    configured capability.
    """
    if is_310p_dflash_piecewise(vllm_config):
        return True
    try:
        runtime_mode = get_forward_context().cudagraph_runtime_mode
    except (AssertionError, RuntimeError):
        return False
    return is_310p_dflash_effective_piecewise(vllm_config, runtime_mode)


def _validate_num_spec_tokens_310(num_speculative_tokens: int | None) -> None:
    """Reject only query lengths that exceed the compiled recurrent buffers."""
    if num_speculative_tokens is not None and num_speculative_tokens > MAX_SUPPORTED_NUM_SPEC_TOKENS_310P:
        raise ValueError(
            "dflash/dspark on 310P supports at most "
            f"{MAX_SUPPORTED_NUM_SPEC_TOKENS_310P} speculative tokens, but got "
            f"{num_speculative_tokens}. The target verify query also contains "
            "one bonus token and the recurrent GDN kernel supports 16 tokens."
        )


def _index_fill_without_add_310p_dflash(
    tensor: torch.Tensor,
    dim: int,
    indices: torch.Tensor,
    value: int,
) -> torch.Tensor:
    """Fill DFlash discard indices without launching the faulty 310P Add."""
    if indices.numel() == 0:
        return tensor
    dim_size = tensor.size(dim)
    positive_positions = torch.arange(
        dim_size,
        device=tensor.device,
        dtype=indices.dtype,
    )
    negative_positions = torch.arange(
        -dim_size,
        0,
        device=tensor.device,
        dtype=indices.dtype,
    )
    mask = torch.logical_or(
        torch.eq(
            positive_positions.unsqueeze(1),
            indices.unsqueeze(0),
        ),
        torch.eq(
            negative_positions.unsqueeze(1),
            indices.unsqueeze(0),
        ),
    )
    mask = mask.any(dim=1)
    tensor_index = [slice(None)] * tensor.dim()
    tensor_index[dim] = mask
    tensor[tuple(tensor_index)] = value
    return tensor


def _draft_cache_block_sizes_310(proposer: Any) -> dict[str, int]:
    """Read the physical block size used by every allocated draft KV cache.

    The draft ``kv_cache_spec.block_size`` (e.g. 640) is split into kernel
    sub-blocks when the KV cache is allocated. Hybrid groups can contain more
    than one physical block size: Qwen3.5-9B DFlash uses 64 for its first four
    layers and 128 for its last two layers. A single global block size therefore
    leaves the latter layers reading empty blocks.
    """
    layer_names = getattr(proposer, "attn_layer_names", None)
    if not layer_names:
        return {}

    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(proposer.vllm_config, AttentionLayerBase)
    block_sizes: dict[str, int] = {}
    for layer_name in layer_names:
        cache = getattr(layers[layer_name], "kv_cache", None)
        # kv_cache is list[per-virtual-engine]; each entry may be a (k, v)
        # tuple or a single stacked tensor. Unwrap to the first real tensor.
        while isinstance(cache, (list, tuple)) and len(cache) > 0:
            cache = cache[0]
        if cache is not None and hasattr(cache, "shape") and cache.dim() >= 2:
            block_sizes[layer_name] = int(cache.shape[-2])
    return block_sizes


def _draft_cache_block_size_310(proposer: Any) -> int | None:
    block_sizes = _draft_cache_block_sizes_310(proposer)
    if not block_sizes:
        return None
    return block_sizes.get(proposer.attn_layer_names[0])


def _compute_slots_for_block_size_310(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    *,
    use_int32_math: bool = False,
) -> torch.Tensor:
    """Map absolute token positions to physical cache slots for one layout."""
    if not use_int32_math:
        positions_long = positions.to(device=block_table.device, dtype=torch.long)
        request_ids_long = request_ids.to(
            device=block_table.device,
            dtype=torch.long,
        )
        block_numbers = torch.div(
            positions_long,
            block_size,
            rounding_mode="floor",
        )
        block_ids = block_table[request_ids_long, block_numbers]
        return (block_ids * block_size + positions_long.remainder(block_size)).to(torch.int32)

    positions_i32 = positions.to(device=block_table.device, dtype=torch.int32)
    request_ids_long = request_ids.to(
        device=block_table.device,
        dtype=torch.long,
    )
    block_numbers_long = torch.div(
        positions_i32,
        block_size,
        rounding_mode="floor",
    ).to(torch.long)
    block_ids_i32 = block_table[
        request_ids_long,
        block_numbers_long,
    ].to(torch.int32)
    return block_ids_i32 * block_size + positions_i32.remainder(block_size)


def _convert_block_table_layout_310(
    source: np.ndarray,
    num_source_blocks_per_row: np.ndarray,
    physical_block_size: int,
    source_block_size: int,
    target_block_size: int,
) -> np.ndarray:
    """Re-expand physical page IDs for a different kernel block size."""
    if source.ndim != 2:
        raise ValueError("source block table must be two-dimensional")
    if physical_block_size <= 0 or source_block_size <= 0 or target_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if physical_block_size % source_block_size != 0 or physical_block_size % target_block_size != 0:
        raise ValueError("kernel block sizes must divide the physical block size")

    source_ratio = physical_block_size // source_block_size
    target_ratio = physical_block_size // target_block_size
    if source.shape[1] % source_ratio != 0:
        raise ValueError("source block table width must cover whole physical pages")
    if len(num_source_blocks_per_row) != source.shape[0]:
        raise ValueError("num_source_blocks_per_row must contain one value per row")

    max_physical_blocks = source.shape[1] // source_ratio
    result = np.zeros(
        (source.shape[0], max_physical_blocks * target_ratio),
        dtype=np.int32,
    )
    for row, source_count_value in enumerate(num_source_blocks_per_row.tolist()):
        source_count = int(source_count_value)
        if source_count < 0 or source_count > source.shape[1]:
            raise ValueError("source logical block count is outside the block table")
        if source_count % source_ratio != 0:
            raise ValueError("source logical block count must cover whole physical pages")

        output_offset = 0
        for start in range(0, source_count, source_ratio):
            chunk = source[row, start : start + source_ratio]
            base = int(chunk[0])
            expected = np.arange(
                base,
                base + source_ratio,
                dtype=np.int32,
            )
            if base % source_ratio != 0 or not np.array_equal(
                chunk,
                expected,
            ):
                raise ValueError("source page must contain contiguous logical blocks")

            physical_id = base // source_ratio
            target_base = physical_id * target_ratio
            result[
                row,
                output_offset : output_offset + target_ratio,
            ] = np.arange(
                target_base,
                target_base + target_ratio,
                dtype=np.int32,
            )
            output_offset += target_ratio

    return result


def _prepare_block_tables_by_size_310(
    proposer: Any,
    cad: CommonAttentionMetadata,
    block_sizes: set[int],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Return a block table whose logical layout matches every kernel size."""
    source_table = proposer.runner.input_batch.block_table[proposer.kv_cache_gid]
    source_block_size = int(source_table.block_size)
    physical_block_size = int(source_table.physical_block_size)
    if len(block_sizes) > 1 and source_table.dcp_world_size * source_table.pcp_world_size > 1:
        raise RuntimeError(
            "310P DFlash does not support mixed draft kernel block sizes together with context parallelism"
        )

    buffers_by_size = getattr(
        proposer,
        "_dflash_block_table_buffers_by_size_310",
        None,
    )
    if buffers_by_size is None:
        buffers_by_size = {}
        proposer._dflash_block_table_buffers_by_size_310 = buffers_by_size

    block_tables_by_size: dict[int, torch.Tensor] = {}
    for block_size in sorted(block_sizes):
        if block_size == source_block_size:
            block_tables_by_size[block_size] = cad.block_table_tensor
            continue

        source = source_table.get_numpy_array()[:batch_size]
        source_counts = source_table.num_blocks_per_row[:batch_size]
        converted = _convert_block_table_layout_310(
            source,
            source_counts,
            physical_block_size,
            source_block_size,
            block_size,
        )
        target_shape = (
            source_table.max_num_reqs,
            converted.shape[1],
        )
        buffers = buffers_by_size.get(block_size)
        if buffers is None or tuple(buffers[0].shape) != target_shape:
            host_buffer = torch.zeros(
                target_shape,
                dtype=torch.int32,
                device="cpu",
                pin_memory=bool(proposer.runner.pin_memory),
            )
            device_buffer = torch.zeros(
                target_shape,
                dtype=torch.int32,
                device=proposer.device,
            )
            buffers = (host_buffer, device_buffer)
            buffers_by_size[block_size] = buffers

        host_buffer, device_buffer = buffers
        host_buffer[:batch_size].copy_(torch.from_numpy(converted))
        host_buffer[batch_size:].zero_()
        device_buffer[:batch_size].copy_(
            host_buffer[:batch_size],
            non_blocking=bool(proposer.runner.pin_memory),
        )
        device_buffer[batch_size:].zero_()
        block_tables_by_size[block_size] = device_buffer[:batch_size]

    return block_tables_by_size


def _recompute_context_slots_310(
    out_context_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    kbs: int,
    num_context: int,
    num_reqs: int,
    *,
    use_int32_math: bool = False,
) -> None:
    """Rebuild the context KV-cache slots with block_table + kernel_block_size.

    The AscendC op recomputes QUERY slots from ``block_table[pos // kbs] * kbs +
    pos % kbs`` (kbs = corrected 64), but CONTEXT slots are an identity
    passthrough of ``cad.slot_mapping`` which the model runner built for a
    128-block layout. In the allocated 64-block draft cache those context slots
    (e.g. 15*128=1920) point at an unreachable physical block (30) that is not in
    the request's block_table, so the draft cross-attention never sees the
    context K/V. Recompute context slots with the SAME scheme as the query slots
    so context and query land in the same physical blocks.
    """
    if num_context <= 0 or kbs <= 0:
        return
    dev = out_context_slot_mapping.device
    qsl = query_start_loc[: num_reqs + 1].to(device=dev, dtype=torch.long)
    counts = (qsl[1:] - qsl[:-1]).clamp(min=0)
    req_ids = torch.repeat_interleave(torch.arange(num_reqs, device=dev), counts)[:num_context]
    slot = _compute_slots_for_block_size_310(
        context_positions[:num_context],
        req_ids,
        block_table,
        kbs,
        use_int32_math=use_int32_math,
    ).to(out_context_slot_mapping.dtype)
    out_context_slot_mapping[:num_context].copy_(slot)


def _prepare_per_layer_slot_mappings_310(
    proposer: Any,
    out_query_positions: torch.Tensor,
    out_context_positions: torch.Tensor,
    cad: CommonAttentionMetadata,
    num_query_total: int,
    num_query_per_req: int,
    num_context: int,
    batch_size: int,
) -> None:
    """Retain query/context slots for every physical draft cache layout."""
    block_sizes_by_layer = _draft_cache_block_sizes_310(proposer)
    if not block_sizes_by_layer:
        return

    source_table = proposer.runner.input_batch.block_table[proposer.kv_cache_gid]
    source_block_size = int(source_table.block_size)
    converted_block_sizes = {
        block_size for block_size in block_sizes_by_layer.values() if block_size != source_block_size
    }
    if not converted_block_sizes:
        # The base proposer already applies graph padding and sliding-window
        # cropping to the source layout. Do not retain an earlier view and then
        # overwrite that finalized metadata in the per-layer hook.
        proposer._dflash_query_slot_mapping_by_layer_310 = {}
        proposer._dflash_block_table_by_layer_310 = {}
        proposer._dflash_block_size_by_layer_310 = {}
        if hasattr(proposer, "_dflash_context_slot_mapping_by_layer_310"):
            del proposer._dflash_context_slot_mapping_by_layer_310
        return
    if getattr(proposer, "draft_window_size", None) is not None:
        # A cropped table's start token and effective seq_len depend on its
        # logical block size. Supporting several sizes therefore needs
        # per-layer window metadata, not just per-layer table/slot tensors.
        raise RuntimeError(
            "310P DFlash does not support draft_window_size together with converted or mixed draft kernel block sizes"
        )

    block_tables_by_size = _prepare_block_tables_by_size_310(
        proposer,
        cad,
        set(block_sizes_by_layer.values()),
        batch_size,
    )

    context_counts = (cad.query_start_loc[1 : batch_size + 1] - cad.query_start_loc[:batch_size]).clamp(min=0)
    context_req_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=proposer.device),
        context_counts.to(device=proposer.device, dtype=torch.long),
    )[:num_context]
    query_req_ids = torch.arange(batch_size, device=proposer.device).repeat_interleave(num_query_per_req)
    context_slots_by_size: dict[int, torch.Tensor] = {}
    query_slots_by_size: dict[int, torch.Tensor] = {}
    query_slot_buffers_by_size = getattr(
        proposer,
        "_dflash_query_slot_mapping_buffers_by_size_310",
        None,
    )
    if query_slot_buffers_by_size is None:
        query_slot_buffers_by_size = {}
        proposer._dflash_query_slot_mapping_buffers_by_size_310 = query_slot_buffers_by_size
    use_int32_math = _uses_int32_draft_address_math_310(proposer.vllm_config)
    for block_size in sorted(set(block_sizes_by_layer.values())):
        block_table = block_tables_by_size[block_size]
        context_slots_by_size[block_size] = _compute_slots_for_block_size_310(
            out_context_positions[:num_context],
            context_req_ids,
            block_table,
            block_size,
            use_int32_math=use_int32_math,
        )
        query_slots = _compute_slots_for_block_size_310(
            out_query_positions[:num_query_total],
            query_req_ids,
            block_table,
            block_size,
            use_int32_math=use_int32_math,
        )
        if block_size != source_block_size:
            final_slot_capacity = proposer.slot_mapping_group[0].shape[0]
            slot_buffer = query_slot_buffers_by_size.get(block_size)
            if (
                slot_buffer is None
                or slot_buffer.shape[0] != final_slot_capacity
                or slot_buffer.dtype != query_slots.dtype
            ):
                slot_buffer = torch.full(
                    (final_slot_capacity,),
                    -1,
                    dtype=query_slots.dtype,
                    device=proposer.device,
                )
                query_slot_buffers_by_size[block_size] = slot_buffer
            slot_buffer[:num_query_total].copy_(query_slots)
            slot_buffer[num_query_total:].fill_(-1)
        query_slots_by_size[block_size] = query_slots

    proposer._dflash_context_slot_mapping_by_layer_310 = [
        context_slots_by_size[block_sizes_by_layer[layer_name]] for layer_name in proposer.attn_layer_names
    ]
    proposer._dflash_query_slot_mapping_by_layer_310 = {
        layer_name: query_slots_by_size[block_sizes_by_layer[layer_name]] for layer_name in proposer.attn_layer_names
    }
    proposer._dflash_block_table_by_layer_310 = {
        layer_name: block_tables_by_size[block_sizes_by_layer[layer_name]] for layer_name in proposer.attn_layer_names
    }
    proposer._dflash_source_block_size_310 = source_block_size
    proposer._dflash_block_size_by_layer_310 = block_sizes_by_layer


def _ensure_kernel_block_size_matches_cache_310(proposer: Any) -> None:
    """Align the draft ``kernel_block_size`` with the allocated KV cache.

    The base proposer sets ``kernel_block_size = get_supported_kernel_block_sizes()[0]``
    (128), but on 310P the draft KV cache is allocated with a smaller kernel
    block size (e.g. 64, from splitting a 640-block spec). The mismatch makes the
    AscendC draft input builder compute slot mappings for a 128-block cache while
    the real cache/block_table use 64, so the SplitFuse cross-attention reads
    empty blocks and returns all-zero output (acceptance ~0).
    """
    if getattr(proposer, "_kernel_block_size_fixed_310", False):
        return
    proposer._kernel_block_size_fixed_310 = True

    _validate_num_spec_tokens_310(getattr(proposer, "num_speculative_tokens", None))

    current = getattr(proposer, "kernel_block_size", None)
    try:
        cache_block_size = _draft_cache_block_size_310(proposer)
        if cache_block_size and cache_block_size != current:
            proposer.kernel_block_size = cache_block_size
            logger.info(
                "Aligned dflash draft kernel_block_size %s -> %s to match allocated KV cache",
                current,
                cache_block_size,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash draft kernel_block_size alignment skipped: %s", exc)


def wrap_dummy_run_with_draft_flag(original):
    """Wrap a proposer ``dummy_run`` so the draft-model forward runs with the
    310P drafting RoPE flag enabled.

    ``dummy_run``'s profile branch calls the draft model directly (not through
    ``_run_merged_draft``, which is where the real flow sets the flag). Without
    the flag, ``_rope_forward_oot`` falls back to the main model's global cos/sin
    slice, which is never populated for a VL main model (it uses MRoPE), leaving
    cos/sin as ``None`` and crashing ``npu_apply_rotary_pos_emb`` with
    ``cos != nullptr``. Enabling the flag makes the draft build cos/sin from its
    own ``cos_sin_cache``. The prior flag value is restored so nesting is safe.
    """

    @functools.wraps(original)
    def dummy_run(self, *args, **kwargs):
        num_tokens = kwargs.get("num_tokens", args[0] if args else None)
        num_reqs = kwargs.get("num_reqs")
        runtime_mode = kwargs.get(
            "aclgraph_runtime_mode",
            CUDAGraphMode.NONE,
        )
        is_profile = kwargs.get("is_profile", False)
        try:
            bound = inspect.signature(original).bind_partial(
                self,
                *args,
                **kwargs,
            )
            num_tokens = bound.arguments.get("num_tokens", num_tokens)
            num_reqs = bound.arguments.get("num_reqs", num_reqs)
            runtime_mode = bound.arguments.get(
                "aclgraph_runtime_mode",
                runtime_mode,
            )
            is_profile = bound.arguments.get("is_profile", is_profile)
        except (TypeError, ValueError):
            pass

        vllm_config = getattr(self, "vllm_config", None)
        uses_hybrid_graph = vllm_config is not None and is_310p_dflash_full_and_piecewise(vllm_config)
        rope_num_tokens = num_tokens
        max_query_tokens = getattr(self, "max_query_tokens", None)
        if isinstance(rope_num_tokens, int) and isinstance(max_query_tokens, int):
            rope_num_tokens = min(rope_num_tokens, max_query_tokens)
        if is_profile:
            runtime_mode = CUDAGraphMode.FULL
        rope_prepared = None
        prepare_rope = getattr(self, "_prepare_full_decode_draft_rope", None)
        if callable(prepare_rope) and isinstance(rope_num_tokens, int):
            rope_prepared = prepare_rope(
                query_positions=self._get_positions(rope_num_tokens),
                query_actual_tokens=rope_num_tokens,
                descriptor_tokens=rope_num_tokens,
                runtime_mode=runtime_mode,
            )

        if vllm_config is not None and (
            is_310p_dflash_piecewise(vllm_config) or get_310p_dflash_graph_capabilities(vllm_config).supports_piecewise
        ):
            runner = getattr(self, "runner", None)
            capacity_tokens = getattr(runner, "max_num_tokens", None)
            if isinstance(capacity_tokens, int) and capacity_tokens > 0:
                configure_draft_rope_capacity_310(capacity_tokens)
        prev_flag = AscendRotaryEmbedding310._is_drafting_update_enabled
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            if (
                uses_hybrid_graph
                and runtime_mode is CUDAGraphMode.FULL
                and isinstance(num_tokens, int)
                and num_tokens > 0
                and isinstance(num_reqs, int)
                and num_reqs > 0
            ):
                with dflash_hybrid_draft_capture_scope_310(
                    real_num_reqs=num_reqs,
                    capacity_tokens=num_tokens,
                ):
                    return original(self, *args, **kwargs)
            return original(self, *args, **kwargs)
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(prev_flag)
            finish_rope = getattr(
                self,
                "_finish_full_decode_draft_rope",
                None,
            )
            if callable(finish_rope):
                finish_rope(rope_prepared)

    return dummy_run


def _copy_and_expand_inputs_ascendc(
    self,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    cad: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None,
    num_query_per_req: int,
    batch_size: int,
    num_context: int,
    sample_from_anchor: bool,
) -> torch.Tensor:
    """AscendC (310P) replacement for the Triton dflash/dspark input
    construction kernel.

    Writes the query/context buffers in place (mirroring the Triton path)
    and returns ``token_indices_to_sample``.
    """
    # MRoPE models feed positions as [3, num_context]; the op (like the Triton
    # kernel it replaces) treats target_positions as a flat [num_context] vector,
    # and both its tiling and infershape size the context outputs from
    # target_positions.shape[0]. The Triton kernel reads the first num_context
    # contiguous elements (row 0), so take that row to keep shape[0] equal to the
    # context token count instead of the mrope dim (which would size the context
    # outputs as 3 and mismatch the buffers).
    if target_positions.dim() > 1:
        target_positions = target_positions[0]

    # 310P: the draft KV cache is allocated by splitting the draft spec
    # block_size (e.g. 640) into smaller kernel blocks (e.g. 64), but the base
    # proposer set self.kernel_block_size to get_supported_kernel_block_sizes()[0]
    # (128). Passing 128 here makes the AscendC builder compute draft slot
    # mappings for a 128-block cache while the real cache/block_table use 64, so
    # SplitFuse reads empty blocks and returns all-zero output. Align
    # self.kernel_block_size with the allocated cache once, right before the op.
    _ensure_kernel_block_size_matches_cache_310(self)

    if num_rejected_tokens_gpu is not None:
        num_rejected = num_rejected_tokens_gpu.to(torch.int32)
    elif _uses_piecewise_persistent_buffers(self.vllm_config):
        max_num_reqs = int(self.runner.max_num_reqs)
        zero_buffer = getattr(self, "_zero_num_rejected_buffer_310", None)
        if zero_buffer is None or zero_buffer.shape[0] < max_num_reqs:
            zero_buffer = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )
            self._zero_num_rejected_buffer_310 = zero_buffer
        else:
            zero_buffer.zero_()
        num_rejected = zero_buffer[:batch_size]
    else:
        # The op always consumes a real [batch_size] tensor; when the caller
        # has no rejection info we feed an all-zero one.
        num_rejected = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

    (
        out_input_ids,
        out_query_positions,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_token_indices,
    ) = torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(
        next_token_ids.to(torch.int32),
        target_positions.to(torch.int32),
        cad.slot_mapping.to(torch.int32),
        cad.query_start_loc.to(torch.int32),
        cad.seq_lens.to(torch.int32),
        cad.block_table_tensor.to(torch.int32),
        num_rejected,
        int(self.parallel_drafting_token_id),
        int(self.kernel_block_size),
        int(num_query_per_req),
        int(self.num_speculative_tokens),
        bool(sample_from_anchor),
    )

    num_query_total = batch_size * num_query_per_req

    # The op passes context slots through from cad.slot_mapping (128-block
    # layout) while query slots are recomputed for the 64-block cache. Rebuild
    # context slots with the same block_table + kernel_block_size so context K/V
    # is written where the draft cross-attention actually reads it.
    _recompute_context_slots_310(
        out_context_slot_mapping,
        out_context_positions,
        cad.query_start_loc,
        cad.block_table_tensor,
        int(self.kernel_block_size),
        num_context,
        batch_size,
        use_int32_math=_uses_int32_draft_address_math_310(self.vllm_config),
    )

    if not sample_from_anchor:
        # Only DFlash consumes per-layer mixed-layout metadata. DSpark shares
        # this input builder but keeps its original single-layout path.
        _prepare_per_layer_slot_mappings_310(
            self,
            out_query_positions,
            out_context_positions,
            cad,
            num_query_total,
            num_query_per_req,
            num_context,
            batch_size,
        )

    self.input_ids[:num_query_total].copy_(out_input_ids[:num_query_total])
    self.positions[:num_query_total].copy_(out_query_positions[:num_query_total])
    self._slot_mapping_buffer[:num_query_total].copy_(out_query_slot_mapping[:num_query_total])
    self._context_positions_buffer[:num_context].copy_(out_context_positions[:num_context])
    self._context_slot_mapping_buffer[:num_context].copy_(out_context_slot_mapping[:num_context])
    return out_token_indices


class AscendDflashProposer310(AscendDflashProposer):
    """310P dflash proposer: builds inputs with the AscendC op (no Triton)."""

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare DFlash tokens without the shared 310P index-fill Add."""
        num_reqs = gpu_input_batch.num_reqs
        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i]) for i in range(num_reqs)]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        discard_sampled_tokens_req_indices = discard_request_indices[:num_discarded_requests]
        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu = _index_fill_without_add_310p_dflash(
            valid_sampled_token_ids_gpu,
            0,
            discard_sampled_tokens_req_indices,
            -1,
        )

        valid_mask = (valid_sampled_token_ids_gpu != -1) & (valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu,
            1,
            last_valid_indices_safe.unsqueeze(1),
        ).squeeze(1)

        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )
        return next_token_ids, valid_sampled_tokens_count

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.num_reqs
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        has_num_rejected = num_rejected_tokens_gpu is not None

        token_indices_to_sample = _copy_and_expand_inputs_ascendc(
            self,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            num_query_per_req=num_query_per_req,
            batch_size=batch_size,
            num_context=num_context,
            sample_from_anchor=False,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        # 310P's dynamic int64 Add requires an internal workspace whose
        # address is not guaranteed to be 64-byte aligned. Sequence lengths
        # are bounded by max_model_len and the downstream draft buffers are
        # int32, so keep this arithmetic in int32 as well.
        use_int32_math = _uses_int32_draft_address_math_310(self.vllm_config)
        effective_seq_lens = cad.seq_lens.to(torch.int32) if use_int32_math else cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu.to(effective_seq_lens.dtype)

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        # DFlash draft cross-attention is non-causal: the query tokens (last
        # sampled token + parallel-drafting mask tokens) attend bidirectionally
        # to the full context and to each other. On 310P this maps to the
        # non-causal SplitFuse mask.
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
    ) -> dict[str, Any]:
        """Insert target context K/V with each draft layer's cache layout."""
        num_context = self._dflash_num_context
        context_slot_mapping = getattr(
            self,
            "_dflash_context_slot_mapping_by_layer_310",
            self._context_slot_mapping_buffer[:num_context],
        )
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            context_slot_mapping,
        )
        return dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self.positions[:num_input_tokens],
            inputs_embeds=None,
        )

    def _build_first_pass_per_layer_attn_metadata(
        self,
        builder,
        common_attn_metadata: CommonAttentionMetadata,
        attn_metadata,
        extra_attn_metadata_args: dict,
    ) -> dict[str, Any]:
        """Build metadata with the physical slot mapping of each draft layer."""
        hybrid_full = is_310p_dflash_full_and_piecewise(self.vllm_config)
        logger.debug(
            "[310p-dflash-full-and-piecewise/draft-metadata] "
            "event=first-step-builder builder=%s hybrid=%s per_layer_slots=%s",
            type(builder).__name__,
            hybrid_full,
            bool(
                getattr(
                    self,
                    "_dflash_query_slot_mapping_by_layer_310",
                    None,
                )
            ),
        )

        def build_metadata(
            layer_common_metadata: CommonAttentionMetadata,
        ):
            if not hybrid_full:
                return builder.build(
                    0,
                    layer_common_metadata,
                    self.runner.get_model(),
                    **extra_attn_metadata_args,
                )
            return builder.build(
                0,
                layer_common_metadata,
                self.runner.get_model(),
                is_drafting=True,
                dflash_hybrid_draft_step=0,
                **extra_attn_metadata_args,
            )

        query_slots_by_layer = getattr(self, "_dflash_query_slot_mapping_by_layer_310", None)
        if not query_slots_by_layer:
            if hybrid_full:
                attn_metadata = build_metadata(common_attn_metadata)
                if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
                    attn_metadata.attn_mask = None
            return {layer_name: attn_metadata for layer_name in self.attn_layer_names}

        block_tables_by_layer = getattr(
            self,
            "_dflash_block_table_by_layer_310",
            {},
        )
        block_sizes_by_layer = getattr(
            self,
            "_dflash_block_size_by_layer_310",
            {},
        )
        source_block_size = getattr(
            self,
            "_dflash_source_block_size_310",
            None,
        )
        block_table_buffers_by_size = getattr(
            self,
            "_dflash_block_table_buffers_by_size_310",
            {},
        )
        query_slot_buffers_by_size = getattr(
            self,
            "_dflash_query_slot_mapping_buffers_by_size_310",
            {},
        )
        metadata_by_layout: dict[tuple[int, int], Any] = {}
        per_layer: dict[str, Any] = {}
        for layer_name in self.attn_layer_names:
            block_size = block_sizes_by_layer.get(layer_name)
            if block_size is not None and block_size == source_block_size:
                # The base path has already applied eager/full-graph row
                # adjustment (and, where configured, sliding-window cropping).
                block_table = common_attn_metadata.block_table_tensor
                slot_mapping = common_attn_metadata.slot_mapping
            elif block_size is not None:
                table_buffers = block_table_buffers_by_size.get(block_size)
                slot_buffer = query_slot_buffers_by_size.get(block_size)
                if table_buffers is None or slot_buffer is None:
                    raise RuntimeError(
                        "310P DFlash converted layout buffers are missing for "
                        f"layer={layer_name}, block_size={block_size}"
                    )
                block_table_buffer = table_buffers[1]
                num_rows = common_attn_metadata.block_table_tensor.shape[0]
                num_slots = common_attn_metadata.slot_mapping.shape[0]
                if num_rows > block_table_buffer.shape[0] or num_slots > slot_buffer.shape[0]:
                    raise RuntimeError(
                        "310P DFlash converted layout buffer is too small for "
                        f"layer={layer_name}, block_size={block_size}, "
                        f"rows={num_rows}, slots={num_slots}"
                    )
                block_table = block_table_buffer[:num_rows]
                slot_mapping = slot_buffer[:num_slots]
            else:
                # Compatibility with existing single-layout callers and unit
                # test stubs that do not install the 310P layout descriptors.
                slot_mapping = query_slots_by_layer[layer_name]
                block_table = block_tables_by_layer.get(
                    layer_name,
                    common_attn_metadata.block_table_tensor,
                )
            cache_key = (
                block_table.data_ptr(),
                slot_mapping.data_ptr(),
            )
            if cache_key not in metadata_by_layout:
                layer_common_metadata = replace(
                    common_attn_metadata,
                    block_table_tensor=block_table,
                    slot_mapping=slot_mapping,
                )
                layer_metadata = build_metadata(layer_common_metadata)
                if hasattr(layer_metadata, "causal") and not layer_metadata.causal:
                    layer_metadata.attn_mask = None
                metadata_by_layout[cache_key] = layer_metadata
            per_layer[layer_name] = metadata_by_layout[cache_key]

        return per_layer


class AscendDsparkProposer310(AscendDsparkProposer):
    """310P dspark proposer: builds inputs with the AscendC op (no Triton)."""

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # Dspark cross-attention: context K/V from target hidden states,
        # Q from query embeddings (next token + mask tokens).
        batch_size = cad.num_reqs

        # Query length of a single request and the whole batch
        num_query_per_req = self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Newly added hidden_states, need to convert to KV Cache
        num_context = target_token_ids.shape[0]
        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        if n < self._dspark_seed_buffer.shape[0]:
            self._dspark_seed_buffer[n:].fill_(0)

        has_num_rejected = num_rejected_tokens_gpu is not None

        # Remove the rejected token to avoid polluting cross-attention
        token_indices_to_sample = _copy_and_expand_inputs_ascendc(
            self,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            num_query_per_req=num_query_per_req,
            batch_size=batch_size,
            num_context=num_context,
            sample_from_anchor=True,
        )

        # Build attn_metadata
        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

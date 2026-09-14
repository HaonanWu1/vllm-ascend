# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Draft-local three-axis RoPE and logical cache positions for 310P DFlash."""

import torch


def build_dflash_mrope_positions(
    target_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_rejected_tokens: torch.Tensor | None,
    position_deltas: torch.Tensor,
    num_context: int,
    num_query_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logical context offsets and T/H/W positions for generated text.

    RoPE coordinates are not cache addresses: image coordinates can repeat and
    text coordinates include a per-request delta. Scheduled context spans may
    also start after a cached prefix, and include subsequently rejected tokens.
    """
    num_reqs = seq_lens.numel()
    starts = query_start_loc[: num_reqs + 1].to(torch.int32)
    counts = starts[1:] - starts[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=seq_lens.device), counts.long(), output_size=num_context
    )
    seq_lens = seq_lens.to(torch.int32)
    offsets = torch.arange(num_context, dtype=torch.int32, device=seq_lens.device)
    cache_positions = seq_lens[req_ids] - counts[req_ids] + offsets - starts[req_ids]
    if target_positions.shape[-1] != num_context:
        raise ValueError("DFlash hidden states and RoPE positions have different token counts")
    rejected = torch.zeros_like(seq_lens) if num_rejected_tokens is None else num_rejected_tokens.to(torch.int32)
    rejected = torch.minimum(rejected.clamp_min(0), counts)
    query_offsets = torch.arange(num_query_per_req, dtype=torch.int32, device=seq_lens.device)
    # A padded drafter batch can include unfinished multimodal prefills. Their
    # final prompt delta already includes images in later chunks, so the unused
    # draft queries can otherwise have negative coordinates. Those candidates
    # are discarded by the runner; use valid dummy coordinates until generation
    # starts. Complete prompts always have a nonnegative text start.
    query_starts = (seq_lens - rejected + position_deltas).clamp_min(0)
    text_positions = query_starts[:, None] + query_offsets
    return cache_positions, text_positions.reshape(1, -1).expand(3, -1)


def mrope_axis_indices(section: list[int], interleaved: bool, device: torch.device) -> torch.Tensor:
    if len(section) != 3 or any(n < 0 for n in section):
        raise ValueError("DFlash m-RoPE requires three nonnegative T/H/W sections")
    if interleaved:
        axes = [0] * sum(section)
        for axis in (1, 2):
            indices = range(axis, section[axis] * 3, 3)
            if any(i >= len(axes) for i in indices):
                raise ValueError("DFlash interleaved m-RoPE sections exceed the rotary dimension")
            for i in indices:
                axes[i] = axis
    else:
        axes = [axis for axis, width in enumerate(section) for _ in range(width)]
    return torch.tensor(axes, dtype=torch.long, device=device)


def gather_mrope_cos_sin(
    cache: torch.Tensor, positions: torch.Tensor, axes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("DFlash m-RoPE positions must have shape [3, num_tokens]")
    num_tokens = positions.shape[-1]
    selected = cache.index_select(0, positions.long().reshape(-1)).view(3, num_tokens, cache.shape[-1])
    cos, sin = selected.chunk(2, dim=-1)
    index = axes.view(1, 1, -1).expand(num_tokens, 1, -1)
    cos = cos.permute(1, 0, 2).gather(1, index).squeeze(1)
    sin = sin.permute(1, 0, 2).gather(1, index).squeeze(1)
    return (
        torch.cat((cos, cos), dim=-1).view(1, num_tokens, 1, cache.shape[-1]),
        torch.cat((sin, sin), dim=-1).view(1, num_tokens, 1, cache.shape[-1]),
    )


class DFlashMRoPEState310:
    """Persistent context/query slices, owned by one drafter, never its target.

    Refresh outside the compiled draft forward. Captured forwards read stable
    slices of these tensors, including when FULL dispatch falls back to eager.
    The original rotary module may be shared through vLLM's get_rope cache;
    neither that module nor the target runner's global slices are modified.
    """

    def __init__(self, rotary, capacity: int):
        self.cache = rotary.cos_sin_cache
        self.head_size = rotary.head_size
        self.rotary_dim = rotary.rotary_dim
        self.is_neox_style = rotary.is_neox_style
        self.axes = mrope_axis_indices(list(rotary.mrope_section), rotary.mrope_interleaved, self.cache.device)
        if self.axes.numel() * 2 != self.rotary_dim:
            raise ValueError("DFlash m-RoPE sections do not cover the rotary dimension")
        shape = (1, capacity, 1, self.rotary_dim)
        self.query_cos = torch.ones(shape, dtype=self.cache.dtype, device=self.cache.device)
        self.query_sin = torch.zeros_like(self.query_cos)
        self.context_cos = torch.ones_like(self.query_cos)
        self.context_sin = torch.zeros_like(self.query_cos)

    def refresh(self, query_positions: torch.Tensor, context_positions: torch.Tensor) -> None:
        for positions, out_cos, out_sin in (
            (query_positions, self.query_cos, self.query_sin),
            (context_positions, self.context_cos, self.context_sin),
        ):
            count = positions.shape[-1]
            if count > out_cos.shape[1]:
                raise ValueError("DFlash m-RoPE position count exceeds its preallocated capacity")
            cos, sin = gather_mrope_cos_sin(self.cache, positions, self.axes)
            out_cos[:, :count].copy_(cos)
            out_sin[:, :count].copy_(sin)
            out_cos[:, count:].fill_(1)
            out_sin[:, count:].zero_()

    def apply(self, query: torch.Tensor, key: torch.Tensor, *, context: bool = False):
        num_tokens = query.shape[0]
        if num_tokens == 0:
            return query, key
        cos = (self.context_cos if context else self.query_cos)[:, :num_tokens]
        sin = (self.context_sin if context else self.query_sin)[:, :num_tokens]
        q = query.reshape(1, num_tokens, -1, self.head_size)
        k = key.reshape(1, num_tokens, -1, self.head_size)
        q_rot, k_rot = q[..., : self.rotary_dim].contiguous(), k[..., : self.rotary_dim].contiguous()
        if query.device.type == "npu" and self.rotary_dim in (64, 128):
            import torch_npu

            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
                q_rot, k_rot, cos, sin, rotary_mode="half" if self.is_neox_style else "interleave"
            )
        else:

            def rotate(x):
                if self.is_neox_style:
                    left, right = x.chunk(2, dim=-1)
                    return torch.cat((-right, left), dim=-1)
                pairs = x.reshape(*x.shape[:-1], -1, 2)
                return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)

            q_rot = q_rot * cos + rotate(q_rot) * sin
            k_rot = k_rot * cos + rotate(k_rot) * sin
        if self.rotary_dim < self.head_size:
            q_rot = torch.cat((q_rot, q[..., self.rotary_dim :]), dim=-1)
            k_rot = torch.cat((k_rot, k[..., self.rotary_dim :]), dim=-1)
        return q_rot.reshape(query.shape), k_rot.reshape(key.shape)

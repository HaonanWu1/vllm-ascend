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

from contextlib import contextmanager

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from vllm_ascend._310p.sample.sampler import fill_exponential_310p
from vllm_ascend._310p.spec_decode.dflash_diagnostics_310 import (
    capture_dflash_diagnostic,
    dflash_diagnostic_enabled,
)
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)


@contextmanager
def _force_pytorch_rejection_path(fn):
    """Route the base rejection sampler through its PyTorch fallbacks on 310P.

    310P has no working Triton, so ``HAS_TRITON`` is forced off (otherwise the base
    ``rejection_sample`` hits ``cal_grid_and_block_size`` -> ``get_vectorcore_num``
    and fails with "Device properties not initialized"). The PyTorch
    recovered-token sampler is bound at the same time. Both module globals are
    restored on exit so nothing else is affected.
    """
    original_has_triton = rejection_sampler_module.HAS_TRITON
    original_recovered = rejection_sampler_module.sample_recovered_tokens
    rejection_sampler_module.HAS_TRITON = False
    rejection_sampler_module.sample_recovered_tokens = fn
    try:
        yield
    finally:
        rejection_sampler_module.HAS_TRITON = original_has_triton
        rejection_sampler_module.sample_recovered_tokens = original_recovered


def _build_verify_diagnostic(
    metadata: SpecDecodeMetadata,
    logits: torch.Tensor,
    output: SamplerOutput,
) -> dict:
    raw_target_argmax = logits[metadata.target_logits_indices].argmax(dim=-1)
    sampled_token_ids = output.sampled_token_ids
    accepted_draft_counts = (
        sampled_token_ids.ne(PLACEHOLDER_TOKEN_ID).sum(dim=1) - 1
    ).clamp(min=0, max=metadata.max_spec_len)
    positions = torch.arange(
        metadata.max_spec_len,
        device=accepted_draft_counts.device,
    )
    per_position_accepted = positions.unsqueeze(0) < accepted_draft_counts.unsqueeze(1)
    return {
        "draft_token_ids": metadata.draft_token_ids,
        "num_draft_tokens": metadata.num_draft_tokens,
        "raw_target_argmax": raw_target_argmax,
        "sampled_token_ids": sampled_token_ids,
        "accepted_draft_counts": accepted_draft_counts,
        "per_position_accepted": per_position_accepted,
    }


class AscendRejectionSampler310(AscendRejectionSampler):
    """310P rejection sampler: PyTorch recovered-token path with CPU RNG (no Triton)."""

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        with _force_pytorch_rejection_path(self.sample_recovered_tokens):
            output = super().forward(metadata, draft_probs, logits, sampling_metadata)

        if (
            getattr(self, "_capture_dflash_diagnostics", False)
            and dflash_diagnostic_enabled()
        ):
            capture_dflash_diagnostic(
                "verify",
                payload_builder=lambda: _build_verify_diagnostic(
                    metadata,
                    logits,
                    output,
                ),
            )
        return output

    def sample_recovered_tokens(
        self,
        max_spec_len: int,
        num_draft_tokens: list[int],
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        device: torch.device,
        use_block_verify: bool = False,
        target_indices: torch.Tensor | None = None,
        global_vocab_size: int | None = None,
        enable_reduce_sampling: bool = False,
    ) -> torch.Tensor:
        batch_size = len(num_draft_tokens)
        vocab_size = target_probs.shape[-1]

        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
        has_draft_mask = num_draft_tensor > 0
        fill_exponential_310p(q, sampling_metadata.generators, has_draft_mask)

        recovered_token_ids = torch.empty_like(draft_token_ids)
        if use_block_verify:
            sample_recovered_tokens_blockwise_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        else:
            sample_recovered_tokens_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        return recovered_token_ids

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

from typing import Any

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend._310p.attention.attention_mask import (
    AttentionMaskBuilder310,
    is_compressed_mask_supported,
)
from vllm_ascend._310p.attention.dflash_hybrid_draft_graph_safe_attention import (
    dflash_hybrid_draft_graph_safe_attention_310,
)
from vllm_ascend._310p.attention.metadata_builder import (
    AscendAttentionMetadataBuilder310,
    get_dflash_hybrid_draft_attention_inputs_310,
    get_query_lens_cpu,
)
from vllm_ascend._310p.dflash_full_and_piecewise import (
    is_310p_dflash_effective_full,
    is_310p_dflash_effective_piecewise,
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend._310p.dflash_full_decode_only import is_310p_dflash_full_decode_only
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_forward_context
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)

MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION = 3
MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION = 5
HEAD_FOLD_QUERY_LEN = 8
HEAD_FOLD_TOKEN_COUNT = 80
HEAD_FOLD_GQA_RATIOS = (4, 8)
HEAD_FOLD_HEAD_SIZES = (128, 256)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend310(AscendAttentionBackend):
    def __init__(self, *args, **kwargs):
        """
        Initializes the 310P backend and sets up the device-specific mask builder.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ):
        """
        Determines the shape of the Key-Value (KV) cache tensor.

        The 310P hardware requires specific memory alignment for optimal performance.
        This method defines a 5D tensor shape where the head size dimension is
        split to ensure alignment to multiples of 16.

        Args:
            num_blocks (int): Number of memory blocks.
            block_size (int): Size of each block.
            num_kv_heads (int): Number of KV heads.
            head_size (int): Dimension size of each head.

        Returns:
            tuple: The specific 5D shape required by the hardware
                   (2, num_blocks, hidden_dim_aligned, block_size, 16).
        """
        # Align to a multiple of 16, as required by the 310P device.
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)

    @staticmethod
    def get_impl_cls():
        """
        Returns the implementation class for the attention operations.
        """
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        """
        Returns the metadata builder class specifically for 310P.
        """
        return AscendAttentionMetadataBuilder310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128, 64]


class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    """
    Implementation of attention operations (Prefill, Decode, Chunked Prefill)
    optimized for the Ascend 310P architecture.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support_compressed_mask = is_compressed_mask_supported()

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        seq_len: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.support_compressed_mask:
            torch_npu._npu_flash_attention(
                query=query,
                key=key,
                value=value,
                mask=mask,
                seq_len=seq_len,
                scale_value=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                out=output,
            )
            return output

        torch_npu._npu_flash_attention_v3(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mask_type=MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._flash_attention(
            query,
            key,
            value,
            attn_metadata.attn_mask,
            attn_metadata.seq_lens,
            output,
        )

    def _uses_hybrid_piecewise_prefill(self) -> bool:
        """Return whether this forward uses the hybrid PIECEWISE prefill route."""
        if not is_310p_dflash_full_and_piecewise(self.vllm_config):
            return False

        from vllm_ascend.ascend_forward_context import get_forward_context

        try:
            runtime_mode = get_forward_context().cudagraph_runtime_mode
        except (AssertionError, RuntimeError):
            return False
        return is_310p_dflash_effective_piecewise(
            self.vllm_config,
            runtime_mode,
        )

    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        """
        Executes Paged Attention (typically for the decode phase).

        Ensures that the sequence length metadata is on the correct device
        before invoking the base implementation.

        Args:
            query (Any): The query tensor.
            attn_metadata (AscendMetadata): Metadata associated with the attention request.
            output (Any | None): Optional output tensor.

        Returns:
            Any: The result of the attention operation.
        """
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def forward_prefill_310(self, query, key, value, attn_metadata, output):
        """
        Executes Flash Attention for the prefill phase on 310P.

        This method handles memory alignment padding. If the query shape implies
        padding (aligned_tokens > real_tokens), it adjusts the sequence length
        of the last request to account for the delta, ensuring the NPU operator
        processes the data correctly.

        Args:
            query, key, value: Input tensors.
            attn_metadata (AscendMetadata): Attention metadata containing masks and seq_lens.
            output: Output tensor.

        Returns:
            The output tensor after flash attention.
        """
        # ATB SelfAttention reads seqLen from host data to build its tiling. Prefer
        # the host seq_lens attached by the 310P metadata builder: the base builder
        # leaves attn_metadata.seq_lens on device for the parallel-drafting path
        # (dflash/dspark), which has no hostData and crashes ATB with
        # "tensor.hostData is null". Reading the CPU copy also lets us compute the
        # token count on host, avoiding a device->host sync in this prefill path.
        seq_len = attn_metadata.seq_lens_cpu
        if seq_len is None or seq_len.device.type != "cpu":
            seq_len = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)

        real_tokens = int(seq_len.sum())
        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens

        # FULL_AND_PIECEWISE keeps a fixed physical descriptor while prefill is
        # executed by the PIECEWISE route.  The descriptor tail is capacity,
        # not part of the final logical request.  Run attention only over the
        # real view and keep the returned physical output buffer stable for the
        # surrounding graph islands.
        if delta > 0 and self._uses_hybrid_piecewise_prefill():
            output_slice = output[:real_tokens]
            self._flash_attention(
                query[:real_tokens],
                key[:real_tokens],
                value[:real_tokens],
                attn_metadata.attn_mask,
                seq_len,
                output_slice,
            )
            output[real_tokens:].zero_()
            return output

        # Preserve the established public 310P behavior for every other route.
        # Clone first so the shared host buffer is never mutated in place.
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        return self._flash_attention(query, key, value, mask, seq_len, output)

    def _splitfuse_head_fold_factor(self, query, output, runtime_mode):
        """Limit head folding to the validated K7 FDO decode contract.

        This changes the operator's row/head presentation, not the model's
        token count, KV-cache layout, attention scale, or graph descriptor.
        Other speculative lengths and execution modes retain the original path.
        """
        if (
            runtime_mode != CUDAGraphMode.FULL
            or not is_310p_dflash_full_decode_only(self.vllm_config)
            or self.vllm_config.speculative_config.num_speculative_tokens + 1 != HEAD_FOLD_QUERY_LEN
            or query.ndim != 3
            or query.shape[0] != HEAD_FOLD_TOKEN_COUNT
            or query.shape[1] != self.num_heads
            or query.dtype != torch.float16
            or not query.is_contiguous()
            or not output.is_contiguous()
            or self.num_kv_heads <= 0
            or self.num_heads % self.num_kv_heads
            or query.shape[-1] not in HEAD_FOLD_HEAD_SIZES
        ):
            return 1
        ratio = self.num_heads // self.num_kv_heads
        return ratio if ratio in HEAD_FOLD_GQA_RATIOS else 1

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        """
        Executes SplitFuse (Chunked Prefill) attention on 310P.

        This handles scenarios where the prefill is split into chunks. It prepares
        the necessary metadata (query lengths, block tables) and generates the
        specific splitfuse mask before calling the NPU operator.

        Args:
            query: The query tensor.
            attn_metadata (AscendMetadata): Metadata containing start locations and block tables.
            output: The output tensor.
        """
        private_draft_inputs = get_dflash_hybrid_draft_attention_inputs_310(attn_metadata)
        try:
            runtime_mode = get_forward_context().cudagraph_runtime_mode
        except (AssertionError, RuntimeError):
            runtime_mode = None
        if (
            private_draft_inputs is not None
            and _EXTRA_CTX.is_draft_model
            and not attn_metadata.causal
            and runtime_mode is not None
            and is_310p_dflash_effective_full(
                self.vllm_config,
                runtime_mode,
            )
        ):
            return dflash_hybrid_draft_graph_safe_attention_310(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                inputs=private_draft_inputs,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                output=output,
            )

        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query = query[:num_actual_tokens]
        output_slice = output[:num_actual_tokens]

        # Host qLens filled in AscendAttentionMetadataBuilder310.build(); eager fallback only.
        qlens = get_query_lens_cpu(attn_metadata)
        if qlens is None:
            if _EXTRA_CTX.capturing:
                raise RuntimeError(
                    "310P splitfuse requires attn_metadata.query_lens_cpu during graph capture; "
                    "ensure AscendAttentionMetadataBuilder310.build() ran before forward."
                )
            qsl_cpu = attn_metadata.query_start_loc.cpu()
            qlens = qsl_cpu[1:] - qsl_cpu[:-1]

        block_table = attn_metadata.block_tables

        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        if self.support_compressed_mask:
            # splitfuse_v2 requires fixed ND [2048, 2048]; parent build() may set FRACTAL_NZ mask.
            if attn_metadata.causal:
                mask = AttentionMaskBuilder310.get_compressed_splitfuse_mask(query.device)
            else:
                mask = AttentionMaskBuilder310.get_compressed_non_causal_splitfuse_mask(query.device)
            torch_npu._npu_paged_attention_splitfuse_v2(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                mask=mask,
                block_table=block_table,
                seq_len=qlens,
                context_lens=attn_metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                mask_type=MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
                out=output_slice,
            )
            return output

        # Same-KV query heads are independent rows with identical visibility.
        # Folding fills the small Cube M dimension without copying the KV cache.
        fold = self._splitfuse_head_fold_factor(query, output_slice, runtime_mode)
        original_tokens = query.shape[0]
        folded_heads = self.num_heads // fold
        if fold != 1:
            query = query.view(original_tokens, folded_heads, fold, query.shape[-1])
            query = query.transpose(1, 2).reshape(original_tokens * fold, folded_heads, query.shape[-1]).contiguous()
            # Host qLens is descriptor-constant for uniform FULL decode; this
            # operation is performed at capture, with no device synchronization.
            qlens = (qlens * fold).pin_memory()
            if folded_heads == 1:
                folded_output = output_slice.view(original_tokens * fold, folded_heads, output_slice.shape[-1])
            else:
                folded_output = torch.empty_like(query)
            # ATB's folded layout can skip writing zero-context descriptor
            # slots. Never let stale outputs from an earlier replay escape.
            folded_output.zero_()
        else:
            folded_output = output_slice

        # Generate the specific mask for splitfuse
        if attn_metadata.causal:
            mask = AttentionMaskBuilder310.get_splitfuse_mask(attn_metadata, query.device, query_head_repeat=fold)
        else:
            mask = AttentionMaskBuilder310.get_non_causal_splitfuse_mask(
                attn_metadata, query.device, query_head_repeat=fold
            )
        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=folded_heads,
            scale_value=self.scale,
            out=folded_output,
        )

        if fold != 1 and folded_heads != 1:
            restored = folded_output.view(original_tokens, fold, folded_heads, output_slice.shape[-1])
            output_view = output_slice.view(original_tokens, folded_heads, fold, output_slice.shape[-1])
            output_view.copy_(restored.transpose(1, 2))

        return output

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        """
        Main dispatch method for attention operations.

        Routes the execution to Decode, Prefill, or Chunked Prefill methods
        based on the current attention state found in metadata.

        Args:
            query, key, value: Input tensors (Key/Value usually empty for decode/chunked).
            kv_cache: The KV cache structure.
            attn_metadata: Metadata determining the state (Prefill vs Decode).
            output: Tensor to write results to.

        Returns:
            The output tensor.

        Raises:
            NotImplementedError: If the attention state is not supported on 310P.
        """
        state = attn_metadata.attn_state
        # Condition for PrefillNoCache: No previous tokens have been processed yet
        if state == AscendAttentionState.PrefillNoCache:
            output = self.forward_prefill_310(query, key, value, attn_metadata, output)
        # Condition for DecodeOnly: Pure decoding phase where each request generates one token
        elif state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        # ChunkedPrefill / PrefillCacheHit: chunked prefill or mixed batches.
        # SpecDecoding: MTP uniform spec verify (splitfuse on 310P).
        elif (
            state in [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit]
            or state == AscendAttentionState.SpecDecoding
        ):
            output = self.forward_chunked_prefill_310(query, attn_metadata, output)
        else:
            raise NotImplementedError(f"AscendAttentionState: {state} is not supported for 310P currently.")
        return output

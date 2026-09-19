# SPDX-License-Identifier: Apache-2.0
"""310P DFlash checkpoint and GDN numerical-chunk scheduling.

DFlash keeps its draft-attention blocks smaller than the target/Mamba blocks
to reduce KV-cache waste.  The upstream splitter reads the resulting global
minimum block size, which can make a prefill cross a reusable Mamba checkpoint
without ever writing the recurrent state for that checkpoint.

GDN additionally needs intermediate prefills to end on its internal 64-token
boundary. Otherwise the WY path selector sees a different padded suffix and
recurrent state is rounded at a different position when the token budget changes.
"""

from functools import wraps
from inspect import signature

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend._310p.gdn_constants import GDN_PREFILL_CHUNK_SIZE
from vllm_ascend.patch.platform.dflash_kv_context import (
    dflash_scheduler_init_scope,
)

_original_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split
_original_scheduler_init = Scheduler.__init__
_ORIGINAL_MAMBA_SPLIT_ACCEPTS_COMMON_PREFIX = (
    "num_uncached_common_prefix_tokens" in signature(_original_mamba_block_aligned_split).parameters
)


def _speculative_config_uses_dflash(speculative_config) -> bool:
    if speculative_config is None:
        return False

    use_dflash = getattr(speculative_config, "use_dflash", None)
    if callable(use_dflash):
        return bool(use_dflash())
    return getattr(speculative_config, "method", None) == "dflash"


def _uses_dflash(scheduler: Scheduler) -> bool:
    speculative_config = getattr(scheduler.vllm_config, "speculative_config", None)
    return _speculative_config_uses_dflash(speculative_config)


def _has_gdn_layers(scheduler: Scheduler) -> bool:
    kv_cache_config = getattr(scheduler, "kv_cache_config", None)
    return any(
        isinstance(group.kv_cache_spec, MambaSpec)
        and group.kv_cache_spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
        for group in getattr(kv_cache_config, "kv_cache_groups", ())
    )


@wraps(_original_scheduler_init)
def _dflash_scheduler_init(self, vllm_config, *args, **kwargs):
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if not _speculative_config_uses_dflash(speculative_config):
        return _original_scheduler_init(self, vllm_config, *args, **kwargs)

    with dflash_scheduler_init_scope():
        _original_scheduler_init(self, vllm_config, *args, **kwargs)

    if _has_gdn_layers(self):
        # GDN needs a stable numerical partition even without prefix caching.
        # Reject configurations that would otherwise defer a long prefill forever.
        budget = self.max_num_scheduled_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        if budget < GDN_PREFILL_CHUNK_SIZE or 0 < threshold < GDN_PREFILL_CHUNK_SIZE:
            raise ValueError(
                "310P DFlash GDN requires a scheduled token budget of at least "
                f"{GDN_PREFILL_CHUNK_SIZE}; long_prefill_token_threshold must be "
                f"0 or at least {GDN_PREFILL_CHUNK_SIZE}."
            )
        if self.cache_config.enable_prefix_caching and self.block_size % GDN_PREFILL_CHUNK_SIZE:
            raise ValueError("310P DFlash GDN cache checkpoints must align to the GDN prefill chunk size.")
        self.need_mamba_block_aligned_split = True


def _needs_dflash_mamba_checkpoint_split(scheduler: Scheduler) -> bool:
    cache_config = scheduler.cache_config
    scheduler_block_size = scheduler.block_size
    return cache_config.enable_prefix_caching and _uses_dflash(scheduler) and scheduler_block_size > 0


def _dflash_mamba_block_aligned_split(
    self: Scheduler,
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    num_uncached_common_prefix_tokens: int = 0,
) -> int:
    split_checkpoint = _needs_dflash_mamba_checkpoint_split(self)
    align_gdn = _uses_dflash(self) and _has_gdn_layers(self)
    if not split_checkpoint and not align_gdn:
        original_args = (
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if _ORIGINAL_MAMBA_SPLIT_ACCEPTS_COMMON_PREFIX:
            return _original_mamba_block_aligned_split(
                *original_args,
                num_uncached_common_prefix_tokens,
            )
        return _original_mamba_block_aligned_split(*original_args)

    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if num_computed_tokens >= prefill_end or num_new_tokens <= 0:
        return num_new_tokens

    # Split at the *next absolute* target/Mamba checkpoint.  Using an absolute
    # boundary matters when an earlier token-budget split left the request in
    # the middle of a block.  For example, computed=1000 must schedule 280,
    # not another full 1280 tokens.
    if split_checkpoint:
        block_size = self.block_size
        next_checkpoint = (num_computed_tokens // block_size + 1) * block_size
        scheduled_end = num_computed_tokens + num_new_tokens
        if num_computed_tokens < next_checkpoint < scheduled_end:
            num_new_tokens = next_checkpoint - num_computed_tokens

        # Preserve Marconi admission at target/Mamba checkpoints, not draft pages.
        if num_uncached_common_prefix_tokens >= block_size and num_new_tokens > num_uncached_common_prefix_tokens:
            num_new_tokens = num_uncached_common_prefix_tokens
            num_new_tokens = num_new_tokens // block_size * block_size

    if align_gdn and num_computed_tokens + num_new_tokens < prefill_end:
        # The WY selector and recurrent state rounding operate on 64-token
        # chunks. Splitting a chunk changes its padded suffix and introduces an
        # extra state cast. Keep intermediate boundaries absolute and stable;
        # only the final prefill chunk may be short. Decode is returned above.
        scheduled_end = num_computed_tokens + num_new_tokens
        aligned_end = scheduled_end // GDN_PREFILL_CHUNK_SIZE * GDN_PREFILL_CHUNK_SIZE
        num_new_tokens = max(0, aligned_end - num_computed_tokens)

    return num_new_tokens


Scheduler._mamba_block_aligned_split = _dflash_mamba_block_aligned_split
Scheduler.__init__ = _dflash_scheduler_init

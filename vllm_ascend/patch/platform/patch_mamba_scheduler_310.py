# SPDX-License-Identifier: Apache-2.0
"""310P DFlash Mamba checkpoint scheduling fix.

DFlash keeps its draft-attention blocks smaller than the target/Mamba blocks
to reduce KV-cache waste.  The upstream splitter reads the resulting global
minimum block size, which can make a prefill cross a reusable Mamba checkpoint
without ever writing the recurrent state for that checkpoint.
"""

from functools import wraps
from inspect import signature

from vllm.v1.core.sched.scheduler import Scheduler

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


@wraps(_original_scheduler_init)
def _dflash_scheduler_init(self, vllm_config, *args, **kwargs):
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if not _speculative_config_uses_dflash(speculative_config):
        return _original_scheduler_init(self, vllm_config, *args, **kwargs)

    with dflash_scheduler_init_scope():
        return _original_scheduler_init(self, vllm_config, *args, **kwargs)


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
    if not _needs_dflash_mamba_checkpoint_split(self):
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
    if num_computed_tokens >= prefill_end:
        return num_new_tokens

    # Split at the *next absolute* target/Mamba checkpoint.  Using an absolute
    # boundary matters when an earlier token-budget split left the request in
    # the middle of a block.  For example, computed=1000 must schedule 280,
    # not another full 1280 tokens.
    block_size = self.block_size
    next_checkpoint = (num_computed_tokens // block_size + 1) * block_size
    scheduled_end = num_computed_tokens + num_new_tokens
    if num_computed_tokens < next_checkpoint < scheduled_end:
        num_new_tokens = next_checkpoint - num_computed_tokens

    # Preserve upstream's Marconi admission optimization, but align it to the
    # target/Mamba checkpoint size rather than the smaller draft block size.
    if num_uncached_common_prefix_tokens >= block_size and num_new_tokens > num_uncached_common_prefix_tokens:
        num_new_tokens = num_uncached_common_prefix_tokens
        num_new_tokens = num_new_tokens // block_size * block_size

    return num_new_tokens


Scheduler._mamba_block_aligned_split = _dflash_mamba_block_aligned_split
Scheduler.__init__ = _dflash_scheduler_init

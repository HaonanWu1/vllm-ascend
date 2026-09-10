# SPDX-License-Identifier: Apache-2.0
"""Regressions for GDN prefill changing with non-aligned scheduler cuts."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_ascend.patch.platform  # noqa: F401
import vllm_ascend.patch.platform.patch_mamba_scheduler_310 as scheduler_patch


def _scheduler(*, method="dflash", model_type="qwen3_5_moe_text", prefix_caching=False, has_mamba=True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = 1280
    scheduler.use_eagle = True
    scheduler.has_mamba_layers = has_mamba
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.cache_config = SimpleNamespace(block_size=640, enable_prefix_caching=prefix_caching)
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        cache_config=scheduler.cache_config,
    )
    return scheduler


@pytest.mark.parametrize(
    ("computed", "budget", "prompt", "expected"),
    [
        (0, 864, 4096, 832),
        (0, 464, 4096, 448),
        (0, 1140, 4096, 1088),
        (832, 1108, 4096, 1088),
        (448, 1124, 4096, 1088),
        (4032, 73, 4105, 73),
        (4096, 9, 4105, 9),
        (4096, 4, 4105, 0),
        (64, 63, 4096, 0),
        (0, 4096, 4096, 4096),
        (4000, 80, 4096, 32),
    ],
)
def test_gdn_prefill_never_ends_an_intermediate_chunk_inside_kernel_block(computed, budget, prompt, expected):
    scheduler = _scheduler()
    request = SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt, num_tokens=prompt)
    assert scheduler._mamba_block_aligned_split(request, budget) == expected


def test_gdn_split_accounts_for_local_and_external_computed_prefix():
    scheduler = _scheduler()
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4096, num_tokens=4096)
    assert scheduler._mamba_block_aligned_split(request, 80, 3968, 32) == 32


@pytest.mark.parametrize("model_type", ["qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"])
def test_gdn_decode_keeps_the_full_speculative_token_budget(model_type):
    scheduler = _scheduler(model_type=model_type)
    request = SimpleNamespace(num_computed_tokens=4109, num_prompt_tokens=4096, num_tokens=4110)
    assert scheduler._mamba_block_aligned_split(request, 16) == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "eagle"},
        {"model_type": "mamba"},
        {"model_type": "qwen3_moe"},
        {"has_mamba": False},
    ],
)
def test_other_models_keep_the_original_split(kwargs):
    scheduler = _scheduler(**kwargs)
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=1460, num_tokens=1460)
    expected = scheduler_patch._original_mamba_block_aligned_split(scheduler, request, 1460)
    assert scheduler._mamba_block_aligned_split(request, 1460) == expected


def test_prefix_caching_preserves_existing_dflash_checkpoint_split():
    scheduler = _scheduler(prefix_caching=True)
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=1460, num_tokens=1460)
    assert scheduler._mamba_block_aligned_split(request, 1000) == 1000


@pytest.mark.parametrize(
    ("kwargs", "original_enabled", "expected"),
    [
        ({}, False, True),
        ({}, True, True),
        ({"method": "eagle"}, False, False),
        ({"model_type": "mamba"}, False, False),
        ({"has_mamba": False}, False, False),
        ({"prefix_caching": True}, False, False),
    ],
)
def test_scheduler_activates_alignment_only_for_uncached_dflash_gdn(monkeypatch, kwargs, original_enabled, expected):
    scheduler = _scheduler(**kwargs)

    # Keep the wrapper and its scope checks real; avoid constructing unrelated
    # KV managers in this focused test. Existing constructor tests cover those.
    def original_init(self, config):
        self.need_mamba_block_aligned_split = original_enabled

    monkeypatch.setattr(scheduler_patch, "_original_scheduler_init", original_init)
    scheduler_patch._dflash_scheduler_init(scheduler, scheduler.vllm_config)
    assert scheduler.need_mamba_block_aligned_split is expected


@pytest.mark.parametrize(("budget", "prefill_limit"), [(32, 0), (1140, 32)])
def test_sub_block_limits_do_not_enable_an_unprogressable_split(monkeypatch, budget, prefill_limit):
    scheduler = _scheduler()
    scheduler.max_num_scheduled_tokens = budget
    scheduler.vllm_config.scheduler_config = SimpleNamespace(long_prefill_token_threshold=prefill_limit)

    def original_init(self, config):
        self.need_mamba_block_aligned_split = False

    monkeypatch.setattr(scheduler_patch, "_original_scheduler_init", original_init)
    scheduler_patch._dflash_scheduler_init(scheduler, scheduler.vllm_config)
    assert scheduler.need_mamba_block_aligned_split is False


@pytest.mark.parametrize("original_enabled", [False, True])
def test_enabled_encoder_preserves_original_scheduler_behavior(monkeypatch, original_enabled):
    scheduler = _scheduler()
    scheduler.max_num_encoder_input_tokens = 1280

    def original_init(self, config):
        self.need_mamba_block_aligned_split = original_enabled

    monkeypatch.setattr(scheduler_patch, "_original_scheduler_init", original_init)
    scheduler_patch._dflash_scheduler_init(scheduler, scheduler.vllm_config)
    assert scheduler.need_mamba_block_aligned_split is original_enabled
    assert not scheduler_patch._needs_dflash_gdn_prefill_alignment(scheduler)
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=1460, num_tokens=1460)
    expected = scheduler_patch._original_mamba_block_aligned_split(scheduler, request, 9)
    assert scheduler._mamba_block_aligned_split(request, 9) == expected

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

# Apply the Ascend platform patches to the upstream Scheduler class used by
# these focused tests.
import vllm_ascend.patch.platform  # noqa: F401
import vllm_ascend.patch.platform.patch_mamba_scheduler_310 as scheduler_patch
from vllm_ascend.patch.platform.dflash_kv_context import resolve_kv_use_eagle


class _SpeculativeConfig:
    def __init__(self, method: str) -> None:
        self.method = method

    def use_dflash(self) -> bool:
        return self.method == "dflash"


def _make_scheduler(
    *,
    method: str = "dflash",
    prefix_caching: bool = True,
    scheduler_block_size: int = 1280,
    cache_block_size: int = 640,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = scheduler_block_size
    scheduler.cache_config = SimpleNamespace(
        block_size=cache_block_size,
        enable_prefix_caching=prefix_caching,
    )
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=_SpeculativeConfig(method),
    )
    scheduler.use_eagle = True
    return scheduler


def _make_request(*, num_tokens: int, num_computed_tokens: int = 0):
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
    )


def _make_single_group_kv_cache_config() -> KVCacheConfig:
    full_spec = FullAttentionSpec(
        block_size=1280,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=full_spec.page_size_bytes * 32,
                shared_by=["attn"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["attn"],
                kv_cache_spec=full_spec,
            )
        ],
    )


def _construct_scheduler_for_kv_policy(method: str | None):
    speculative_config = None
    if method is not None:
        speculative_config = SimpleNamespace(
            num_speculative_tokens_per_batch_size=None,
            use_eagle=lambda: True,
            uses_draft_model=lambda: True,
            use_dflash=lambda: method == "dflash",
            method=method,
        )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=10,
            max_num_scheduled_tokens=None,
            max_num_batched_tokens=4096,
            policy="fcfs",
            watermark=0.0,
            scheduler_reserve_full_isl=False,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=32,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        lora_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        observability_config=SimpleNamespace(kv_cache_metrics=False),
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=8192,
            enable_return_routed_experts=False,
        ),
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=speculative_config,
        num_speculative_tokens=15,
        max_concurrent_batches=1,
        use_v2_model_runner=False,
    )
    captured = {}

    def _fake_original_coordinator(**kwargs):
        captured["effective_use_eagle"] = kwargs["use_eagle"]
        return SimpleNamespace(block_pool=MagicMock())

    with (
        patch(
            "vllm.v1.core.sched.scheduler.EventPublisherFactory.create",
            return_value=None,
        ),
        patch(
            "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
            side_effect=_fake_original_coordinator,
        ),
    ):
        scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=_make_single_group_kv_cache_config(),
            structured_output_manager=object(),
            block_size=1280,
            mm_registry=SimpleNamespace(
                supports_multimodal_inputs=lambda _: False,
            ),
        )

    return scheduler, captured


@pytest.mark.parametrize(
    (
        "method",
        "expected_scheduler_use_eagle",
        "expected_lookahead",
        "expected_kv_use_eagle",
    ),
    [
        pytest.param("dflash", True, 16, False, id="dflash"),
        pytest.param("eagle", True, 15, True, id="eagle"),
        pytest.param(None, False, 0, False, id="non-speculative"),
    ],
)
def test_scheduler_separates_dflash_scheduling_from_eagle_kv_policy(
    method: str | None,
    expected_scheduler_use_eagle: bool,
    expected_lookahead: int,
    expected_kv_use_eagle: bool,
) -> None:
    scheduler, captured = _construct_scheduler_for_kv_policy(method)

    assert scheduler.use_eagle is expected_scheduler_use_eagle
    assert scheduler.num_lookahead_tokens == expected_lookahead
    assert scheduler.kv_cache_manager.use_eagle is expected_scheduler_use_eagle
    assert captured["effective_use_eagle"] is expected_kv_use_eagle


def test_dflash_scheduler_init_restores_context_when_original_raises(monkeypatch) -> None:
    def _raise_during_init(*args, **kwargs):
        raise RuntimeError("scheduler init failed")

    monkeypatch.setattr(
        scheduler_patch,
        "_original_scheduler_init",
        _raise_during_init,
    )
    vllm_config = SimpleNamespace(
        speculative_config=_SpeculativeConfig("dflash"),
    )

    with pytest.raises(RuntimeError, match="scheduler init failed"):
        scheduler_patch._dflash_scheduler_init(object(), vllm_config)

    assert resolve_kv_use_eagle(True) is True


@pytest.mark.parametrize(
    ("computed", "requested", "expected"),
    [
        (0, 640, 640),
        (640, 820, 640),
        (1280, 180, 180),
        (1200, 200, 80),
        (1280, 1400, 1280),
        (2560, 100, 100),
    ],
)
@pytest.mark.parametrize("cache_block_size", [640, 1280])
def test_dflash_split_stops_at_absolute_mamba_checkpoint(
    computed: int,
    requested: int,
    expected: int,
    cache_block_size: int,
) -> None:
    scheduler = _make_scheduler(cache_block_size=cache_block_size)
    request = _make_request(
        num_tokens=4000,
        num_computed_tokens=computed,
    )

    assert scheduler._mamba_block_aligned_split(request, requested) == expected


def test_dflash_prefill_materializes_each_mamba_checkpoint() -> None:
    """A long DFlash prefill must not jump over a reusable Mamba state."""
    scheduler = _make_scheduler()
    request = _make_request(num_tokens=3000)

    chunks = []
    while request.num_computed_tokens < request.num_tokens:
        remaining = request.num_tokens - request.num_computed_tokens
        scheduled = scheduler._mamba_block_aligned_split(request, remaining)
        chunks.append(scheduled)
        request.num_computed_tokens += scheduled

    assert chunks == [1280, 1280, 440]


def test_dflash_1460_prefill_uses_two_rounds() -> None:
    """Regression: the broken 640-byte draft granularity produced 640 + 820."""
    scheduler = _make_scheduler()
    request = _make_request(num_tokens=1460)

    first = scheduler._mamba_block_aligned_split(request, 1460)
    request.num_computed_tokens += first
    second = scheduler._mamba_block_aligned_split(request, 1460 - first)

    assert [first, second] == [1280, 180]


def test_dflash_split_uses_absolute_checkpoint_after_budget_limited_chunk() -> None:
    """A prior short chunk must not shift the 1280-token checkpoint."""
    scheduler = _make_scheduler()
    request = _make_request(num_tokens=1460)

    first = scheduler._mamba_block_aligned_split(request, 1000)
    request.num_computed_tokens += first
    second = scheduler._mamba_block_aligned_split(request, 1460 - first)
    request.num_computed_tokens += second
    third = scheduler._mamba_block_aligned_split(
        request,
        1460 - request.num_computed_tokens,
    )

    assert [first, second, third] == [1000, 280, 180]


def test_non_dflash_keeps_upstream_eagle_split() -> None:
    scheduler = _make_scheduler(method="eagle")
    request = _make_request(num_tokens=1460)

    assert scheduler._mamba_block_aligned_split(request, 1460) == 640


def test_dflash_without_prefix_caching_keeps_upstream_split() -> None:
    scheduler = _make_scheduler(prefix_caching=False)
    request = _make_request(num_tokens=1460)

    assert scheduler._mamba_block_aligned_split(request, 1460) == 640


def test_fallback_supports_legacy_mamba_split_signature(monkeypatch) -> None:
    captured = {}

    def _legacy_split(
        scheduler,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        captured["args"] = (
            scheduler,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        return 321

    monkeypatch.setattr(
        scheduler_patch,
        "_original_mamba_block_aligned_split",
        _legacy_split,
    )
    monkeypatch.setattr(
        scheduler_patch,
        "_ORIGINAL_MAMBA_SPLIT_ACCEPTS_COMMON_PREFIX",
        False,
        raising=False,
    )
    scheduler = _make_scheduler(method="eagle")
    request = _make_request(num_tokens=1460)

    result = scheduler._mamba_block_aligned_split(
        request,
        700,
        11,
        13,
        17,
    )

    assert result == 321
    assert captured["args"] == (scheduler, request, 700, 11, 13)

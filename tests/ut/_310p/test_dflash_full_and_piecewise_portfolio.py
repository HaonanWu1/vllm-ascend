# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend._310p.dflash_full_and_piecewise import (
    apply_dflash_full_and_piecewise_capture_config,
    get_310p_dflash_graph_capabilities,
    initialize_dflash_full_and_piecewise_cudagraph_keys,
    is_310p_dflash_effective_full,
    is_310p_dflash_effective_piecewise,
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend.ascend_config import AscendCompilationConfig
from vllm_ascend.patch.worker.patch_cudagraph import _create_padded_batch_descriptor

PORTFOLIO_KEY = "dflash_full_and_piecewise_capture_config"


def _config(
    *,
    mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    method="dflash",
    piecewise=32,
    full=80,
    k=7,
    max_num_seqs=10,
    max_num_batched_tokens=3584,
    capture_sizes=None,
):
    portfolio = {
        "piecewise_capture_size": piecewise,
        "full_capture_size": full,
    }
    return SimpleNamespace(
        speculative_config=(
            SimpleNamespace(method=method, num_speculative_tokens=k)
            if method is not None
            else None
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=(
                max(capture_sizes) if capture_sizes else None
            ),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        additional_config={
            "ascend_compilation_config": {PORTFOLIO_KEY: portfolio}
        },
    )


class _FakeDispatcher:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1 + vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None else 1
        )
        self.cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_keys = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }
        self.keys_initialized = False
        self.specialize_lora_count = False

    def _compute_bs_to_padded_graph_size(self):
        sizes = self.compilation_config.cudagraph_capture_sizes
        max_size = self.compilation_config.max_cudagraph_capture_size
        self._bs_to_padded_graph_size = [0] * (max_size + 1)
        for value in range(max_size + 1):
            self._bs_to_padded_graph_size[value] = next(
                (size for size in sizes if size >= value),
                max_size,
            )

    @staticmethod
    def _get_lora_cases():
        return [0]

    _create_padded_batch_descriptor = _create_padded_batch_descriptor

    def add_cudagraph_key(self, runtime_mode, descriptor):
        self.cudagraph_keys[runtime_mode].add(descriptor)


def _sizes(dispatcher, mode):
    return {desc.num_tokens for desc in dispatcher.cudagraph_keys[mode]}


def test_config_parses_one_piecewise_and_one_full_capacity():
    config = AscendCompilationConfig(
        **{
            PORTFOLIO_KEY: {
                "piecewise_capture_size": 32,
                "full_capture_size": 80,
            }
        }
    )

    assert getattr(config, PORTFOLIO_KEY) == {
        "piecewise_capture_size": 32,
        "full_capture_size": 80,
    }


def test_config_parses_multiple_full_capacities_as_a_json_list():
    config = AscendCompilationConfig(
        **{
            PORTFOLIO_KEY: {
                "piecewise_capture_size": 32,
                "full_capture_size": [40, 80],
            }
        }
    )

    assert getattr(config, PORTFOLIO_KEY) == {
        "piecewise_capture_size": 32,
        "full_capture_size": [40, 80],
    }


def test_absent_config_does_not_add_a_new_ascend_config_field():
    config = AscendCompilationConfig()

    assert not hasattr(config, PORTFOLIO_KEY)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"piecewise_capture_size": 0, "full_capture_size": 80},
         "positive integer"),
        ({"piecewise_capture_size": 32, "full_capture_size": []},
         "non-empty list"),
        ({"piecewise_capture_size": 32, "full_capture_size": [80, 80]},
         "must not contain duplicate"),
        ({"piecewise_capture_size": 32, "full_capture_size": [80, 0]},
         "positive integer"),
        ({"piecewise_capture_size": 32,
          "full_capture_size": [80, True]}, "positive integer"),
        ({"piecewise_capture_size": 32, "full_capture_size": (80, 96)},
         "positive integer or a list"),
        ({"piecewise_capture_size": 32}, "requires exactly"),
        ({"piecewise_capture_size": 32, "full_capture_size": 80, "extra": 1},
         "requires exactly"),
    ],
)
def test_config_rejects_unvalidated_portfolios(raw, message):
    with pytest.raises(ValueError, match=message):
        AscendCompilationConfig(**{PORTFOLIO_KEY: raw})


def test_absent_config_preserves_existing_capture_sizes():
    config = _config(capture_sizes=[16, 32, 80])
    del config.additional_config["ascend_compilation_config"][PORTFOLIO_KEY]

    assert not apply_dflash_full_and_piecewise_capture_config(config)
    assert config.compilation_config.cudagraph_capture_sizes == [16, 32, 80]


def test_absent_config_disables_private_hybrid_capability():
    config = _config()
    del config.additional_config["ascend_compilation_config"][PORTFOLIO_KEY]

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert not is_310p_dflash_full_and_piecewise(config)
        assert not get_310p_dflash_graph_capabilities(config).any
        assert not is_310p_dflash_effective_full(
            config,
            CUDAGraphMode.FULL,
        )
        assert not is_310p_dflash_effective_piecewise(
            config,
            CUDAGraphMode.PIECEWISE,
        )


@pytest.mark.parametrize(
    ("platform_310p", "method", "mode"),
    [
        (False, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, "mtp", CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, None, CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, "dflash", CUDAGraphMode.PIECEWISE),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_explicit_config_does_not_mutate_other_scopes(
    platform_310p,
    method,
    mode,
):
    config = _config(
        method=method,
        mode=mode,
        capture_sizes=[16, 24],
    )

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=platform_310p,
    ):
        assert not apply_dflash_full_and_piecewise_capture_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [16, 24]


def test_platform_planner_forms_descriptor_union_without_ownership():
    config = _config(capture_sizes=None)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert apply_dflash_full_and_piecewise_capture_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [32, 80]
    assert config.compilation_config.max_cudagraph_capture_size == 80


def test_platform_planner_forms_union_for_multiple_full_capacities():
    config = _config(full=[40, 80], capture_sizes=None)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert apply_dflash_full_and_piecewise_capture_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [32, 40, 80]
    assert config.compilation_config.max_cudagraph_capture_size == 80


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"full": 82}, "divisible"),
        ({"full": 88}, "logical deployment bound"),
        ({"full": [40, 82]}, "divisible"),
        ({"full": [40, 88]}, "logical deployment bound"),
        ({"full": [40, 80], "max_num_batched_tokens": 64},
         "max_num_batched_tokens"),
        ({"piecewise": 4096}, "max_num_batched_tokens"),
    ],
)
def test_platform_planner_rejects_unsafe_capacity_contracts(kwargs, message):
    config = _config(**kwargs)

    with (
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        pytest.raises(ValueError, match=message),
    ):
        apply_dflash_full_and_piecewise_capture_config(config)


def test_target_inventory_has_strict_mode_ownership():
    config = _config(capture_sizes=[32, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        handled = initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert handled
    assert _sizes(dispatcher, CUDAGraphMode.PIECEWISE) == {32}
    assert _sizes(dispatcher, CUDAGraphMode.FULL) == {80}


def test_target_inventory_has_each_configured_full_capacity():
    config = _config(full=[40, 80], capture_sizes=[32, 40, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        handled = initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert handled
    assert _sizes(dispatcher, CUDAGraphMode.PIECEWISE) == {32}
    assert _sizes(dispatcher, CUDAGraphMode.FULL) == {40, 80}


def test_target_inventory_builds_full_capacity_lora_product():
    config = _config(full=[40, 80], capture_sizes=[32, 40, 80])
    dispatcher = _FakeDispatcher(config)
    dispatcher._get_lora_cases = lambda: [0, 1, 2]

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert {
        (descriptor.num_tokens, descriptor.has_lora,
         descriptor.num_active_loras)
        for descriptor in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
    } == {
        (40, False, 0),
        (40, True, 1),
        (40, True, 2),
        (80, False, 0),
        (80, True, 1),
        (80, True, 2),
    }
    assert dispatcher.captured_lora_counts == [1, 2]


def test_existing_dispatcher_routes_full_piecewise_and_safe_fallback():
    config = _config(capture_sizes=[32, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    dispatch = CudagraphDispatcher.dispatch.__get__(dispatcher)

    mode, descriptor = dispatch(80, uniform_decode=True)
    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == 80

    mode, descriptor = dispatch(32, uniform_decode=False)
    assert mode == CUDAGraphMode.PIECEWISE
    assert descriptor.num_tokens == 32

    # A uniform workload below the FULL bucket can safely use the configured
    # PIECEWISE bucket without adding FULL32 to the capture inventory.
    mode, descriptor = dispatch(16, uniform_decode=True)
    assert mode == CUDAGraphMode.PIECEWISE
    assert descriptor.num_tokens == 32

    mode, descriptor = dispatch(40, uniform_decode=False)
    assert mode == CUDAGraphMode.NONE
    assert descriptor.num_tokens == 40

    mode, descriptor = dispatch(88, uniform_decode=True)
    assert mode == CUDAGraphMode.NONE
    assert descriptor.num_tokens == 88


def test_existing_dispatcher_routes_across_multiple_full_capacities():
    config = _config(full=[40, 80], capture_sizes=[32, 40, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    dispatch = CudagraphDispatcher.dispatch.__get__(dispatcher)

    mode, descriptor = dispatch(40, uniform_decode=True)
    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == 40

    mode, descriptor = dispatch(48, uniform_decode=True)
    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == 80


def test_draft_outer_dispatcher_keeps_only_piecewise_capacity():
    config = _config(capture_sizes=[32, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        handled = initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert handled
    assert _sizes(dispatcher, CUDAGraphMode.PIECEWISE) == {32}
    assert _sizes(dispatcher, CUDAGraphMode.FULL) == set()


def test_initializer_defers_to_upstream_without_explicit_portfolio():
    config = _config(capture_sizes=[32, 80])
    del config.additional_config["ascend_compilation_config"][PORTFOLIO_KEY]
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert not initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert not dispatcher.keys_initialized


@pytest.mark.parametrize("piecewise, full", [([64, 32], 80), ([32, 64], [40, 80]), ([32], [80])])
def test_multiple_piecewise_config_round_trips_without_changing_scalar_full(piecewise, full):
    config = AscendCompilationConfig(
        **{
            PORTFOLIO_KEY: {
                "piecewise_capture_size": piecewise,
                "full_capture_size": full,
            }
        }
    )
    assert getattr(config, PORTFOLIO_KEY) == {
        "piecewise_capture_size": piecewise,
        "full_capture_size": full,
    }


@pytest.mark.parametrize(
    "piecewise", [[], [32, 32], [32, 0], [32, -1], [32, True], [32, 1.5], [None], (32, 64), {32, 64}]
)
def test_multiple_piecewise_config_rejects_invalid_capacities(piecewise):
    with pytest.raises(ValueError):
        AscendCompilationConfig(
            **{
                PORTFOLIO_KEY: {
                    "piecewise_capture_size": piecewise,
                    "full_capture_size": 80,
                }
            }
        )


def test_multiple_piecewise_planner_sorts_union_and_checks_every_capacity():
    config = _config(piecewise=[64, 32], full=[80, 40], capture_sizes=None)
    with patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True):
        assert apply_dflash_full_and_piecewise_capture_config(config)
    assert config.compilation_config.cudagraph_capture_sizes == [32, 40, 64, 80]
    assert config.compilation_config.max_cudagraph_capture_size == 80
    invalid = _config(piecewise=[4096, 32], full=80)
    with (
        patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True),
        pytest.raises(ValueError, match="max_num_batched_tokens"),
    ):
        apply_dflash_full_and_piecewise_capture_config(invalid)


@pytest.mark.parametrize("runtime_mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.PIECEWISE])
def test_multiple_piecewise_inventory_preserves_target_and_draft_ownership(runtime_mode):
    config = _config(piecewise=[32, 64], full=[40, 64, 80], capture_sizes=[32, 40, 64, 80])
    dispatcher = _FakeDispatcher(config)
    with patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True):
        assert initialize_dflash_full_and_piecewise_cudagraph_keys(dispatcher, runtime_mode, uniform_decode_query_len=8)
    assert _sizes(dispatcher, CUDAGraphMode.PIECEWISE) == {32, 64}
    assert _sizes(dispatcher, CUDAGraphMode.FULL) == (
        {40, 64, 80} if runtime_mode == CUDAGraphMode.FULL_AND_PIECEWISE else set()
    )
    assert all(d.num_reqs is None and not d.uniform for d in dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE])


def test_multiple_piecewise_inventory_preserves_lora_cases_and_capture_order():
    dispatcher = _FakeDispatcher(_config(piecewise=[64, 32], full=[40, 80], capture_sizes=[32, 40, 64, 80]))
    dispatcher._get_lora_cases = lambda: [0, 1, 2]
    with patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True):
        initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher, CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=8
        )
    assert {
        (d.num_tokens, d.has_lora, d.num_active_loras) for d in dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]
    } == {
        (32, False, 0),
        (32, True, 1),
        (32, True, 2),
        (64, False, 0),
        (64, True, 1),
        (64, True, 2),
    }
    assert dispatcher.captured_lora_counts == [1, 2]
    captures = CudagraphDispatcher.get_capture_descs(dispatcher)
    assert [(mode, [d.num_tokens for d in descs]) for mode, descs in captures] == [
        (CUDAGraphMode.PIECEWISE, [64, 64, 64, 32, 32, 32]),
        (CUDAGraphMode.FULL, [80, 80, 80, 40, 40, 40]),
    ]


@pytest.mark.parametrize(
    "tokens, uniform, expected_mode, expected_size",
    [
        (24, False, CUDAGraphMode.PIECEWISE, 32),
        (48, False, CUDAGraphMode.PIECEWISE, 64),
        (40, True, CUDAGraphMode.FULL, 40),
        (72, True, CUDAGraphMode.FULL, 80),
        (33, False, CUDAGraphMode.NONE, 33),
        (48, True, CUDAGraphMode.PIECEWISE, 64),
        (81, False, CUDAGraphMode.NONE, 81),
    ],
)
def test_multiple_piecewise_uses_existing_dispatch_and_safe_fallback(tokens, uniform, expected_mode, expected_size):
    dispatcher = _FakeDispatcher(_config(piecewise=[32, 64], full=[40, 80], capture_sizes=[32, 40, 64, 80]))
    with patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True):
        initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher, CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=8
        )
    mode, descriptor = CudagraphDispatcher.dispatch(dispatcher, tokens, uniform_decode=uniform)
    assert mode == expected_mode
    assert descriptor.num_tokens == expected_size


@pytest.mark.parametrize(
    "platform, method, mode",
    [
        (False, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, "mtp", CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, None, CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, "dflash", CUDAGraphMode.NONE),
        (True, "dflash", CUDAGraphMode.PIECEWISE),
        (True, "dflash", CUDAGraphMode.FULL),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_multiple_piecewise_does_not_change_other_platforms_or_graph_modes(platform, method, mode):
    config = _config(piecewise=[32, 64], full=[40, 80], method=method, mode=mode, capture_sizes=[16, 80])
    dispatcher = _FakeDispatcher(config)
    with patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=platform):
        assert not apply_dflash_full_and_piecewise_capture_config(config)
        assert not initialize_dflash_full_and_piecewise_cudagraph_keys(dispatcher, mode, uniform_decode_query_len=8)
    assert config.compilation_config.cudagraph_capture_sizes == [16, 80]
    assert config.compilation_config.max_cudagraph_capture_size == 80
    assert not dispatcher.keys_initialized


@pytest.mark.parametrize("piecewise", [[60], [32, 60], [32, 41], [32, 83]])
def test_rejects_unaligned_piecewise_list_before_uniform_dispatch(piecewise):
    config = _config(piecewise=piecewise, full=[40, 80])
    with (
        patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True),
        pytest.raises(ValueError, match="piecewise_capture_size.*verification width"),
    ):
        apply_dflash_full_and_piecewise_capture_config(config)
    assert config.compilation_config.cudagraph_capture_sizes is None


def test_unaligned_scalar_piecewise_preserves_historical_behavior():
    config = _config(piecewise=2500, full=[40, 80])
    with patch("vllm_ascend._310p.dflash_full_and_piecewise.is_310p", return_value=True):
        assert apply_dflash_full_and_piecewise_capture_config(config)
        dispatcher = _FakeDispatcher(config)
        assert initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher, CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=8
        )
    assert config.compilation_config.cudagraph_capture_sizes == [40, 80, 2500]
    for tokens in range(8, 81, 8):
        mode, descriptor = CudagraphDispatcher.dispatch(dispatcher, tokens, uniform_decode=True)
        assert mode == CUDAGraphMode.FULL
        assert descriptor.num_tokens % 8 == 0

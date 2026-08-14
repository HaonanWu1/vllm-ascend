# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MambaSpec,
)

from vllm_ascend.patch.platform import patch_mamba_config_310

_pad_dflash_mamba_page_sizes_310 = (
    patch_mamba_config_310._pad_dflash_mamba_page_sizes_310
)


def _mixed_page_specs():
    target_attention = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
    )
    draft_attention = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float16,
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((8,),),
        dtypes=(torch.float16,),
        page_size_padded=target_attention.page_size_bytes,
    )
    return {
        "target.attn": target_attention,
        "target.mamba": mamba,
        "draft.attn": draft_attention,
    }


def test_dflash_pads_mamba_page_without_changing_logical_block() -> None:
    specs = _mixed_page_specs()
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
    )

    padded = _pad_dflash_mamba_page_sizes_310(config, specs)

    assert padded is not specs
    assert padded["target.attn"] is specs["target.attn"]
    assert padded["draft.attn"] is specs["draft.attn"]
    assert padded["target.mamba"] is not specs["target.mamba"]
    assert specs["target.mamba"].page_size_bytes == 32
    assert padded["target.mamba"].page_size_bytes == 64
    assert padded["target.mamba"].page_size_padded == 64
    assert padded["target.mamba"].block_size == 16

    unified = unify_kv_cache_spec_page_size(padded)
    assert {spec.page_size_bytes for spec in unified.values()} == {64}
    assert unified["target.attn"].block_size == 8
    assert unified["target.mamba"].block_size == 16


def test_mamba_page_padding_is_dflash_only() -> None:
    specs = _mixed_page_specs()

    for speculative_config in (
        None,
        SimpleNamespace(method="dspark"),
    ):
        actual = _pad_dflash_mamba_page_sizes_310(
            SimpleNamespace(speculative_config=speculative_config),
            specs,
        )

        assert actual is specs
        assert actual["target.mamba"].page_size_bytes == 32


def test_uniform_dflash_pages_are_not_rewritten() -> None:
    specs = _mixed_page_specs()
    uniform_mamba = MambaSpec(
        block_size=16,
        shapes=((8,),),
        dtypes=(torch.float16,),
        page_size_padded=64,
    )
    specs["target.mamba"] = uniform_mamba

    actual = _pad_dflash_mamba_page_sizes_310(
        SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash"),
        ),
        specs,
    )

    assert actual is specs
    assert actual["target.mamba"] is uniform_mamba


def test_dflash_ignores_larger_hidden_state_page() -> None:
    specs = _mixed_page_specs()
    specs["draft.hidden"] = HiddenStateCacheSpec(
        block_size=4,
        num_kv_heads=4,
        head_size=4,
        dtype=torch.float16,
    )
    assert specs["draft.hidden"].page_size_bytes == 128

    actual = _pad_dflash_mamba_page_sizes_310(
        SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash"),
        ),
        specs,
    )

    assert actual["target.mamba"].page_size_bytes == 64
    assert actual["draft.hidden"] is specs["draft.hidden"]


def test_dflash_mamba_only_pages_are_not_rewritten() -> None:
    specs = {
        "mamba.small": MambaSpec(
            block_size=16,
            shapes=((16,),),
            dtypes=(torch.float16,),
        ),
        "mamba.large": MambaSpec(
            block_size=16,
            shapes=((32,),),
            dtypes=(torch.float16,),
        ),
    }

    actual = _pad_dflash_mamba_page_sizes_310(
        SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash"),
        ),
        specs,
    )

    assert actual is specs


def test_installed_wrapper_passes_padded_specs_to_upstream(monkeypatch) -> None:
    specs = _mixed_page_specs()
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
    )
    expected_groups = [object()]
    captured = {}

    def fake_get_kv_cache_groups(actual_config, actual_specs):
        captured["config"] = actual_config
        captured["specs"] = actual_specs
        return expected_groups

    monkeypatch.setattr(
        patch_mamba_config_310,
        "_ORIGINAL_GET_KV_CACHE_GROUPS",
        fake_get_kv_cache_groups,
    )

    actual_groups = patch_mamba_config_310._get_kv_cache_groups_310(
        config,
        specs,
    )

    assert actual_groups is expected_groups
    assert captured["config"] is config
    assert captured["specs"]["target.mamba"].page_size_bytes == 64
    assert (
        kv_cache_utils.get_kv_cache_groups
        is patch_mamba_config_310._get_kv_cache_groups_310
    )

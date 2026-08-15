from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.patch.platform.patch_dflash_graph_config_310 import (
    DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
    check_and_update_config_310,
)
from vllm_ascend.platform import NPUPlatform


def _config(
    method="dflash",
    mode=CUDAGraphMode.FULL,
    layer_types=("linear_attention", "full_attention"),
):
    return SimpleNamespace(
        speculative_config=(
            SimpleNamespace(method=method)
            if method is not None
            else None
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types),
        ),
    )


def test_full_dflash_hybrid_fallback_precedes_platform_compile_setup():
    config = _config()
    seen_modes = []

    def original(_cls, current_config):
        seen_modes.append(
            current_config.compilation_config.cudagraph_mode
        )

    with (
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310."
            "_original_check_and_update_config_310",
            side_effect=original,
        ),
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310.logger.warning"
        ) as warning,
    ):
        check_and_update_config_310(object, config)

    assert seen_modes == [CUDAGraphMode.FULL_AND_PIECEWISE]
    assert (
        config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert (
        getattr(
            config.compilation_config,
            DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
        )
        == CUDAGraphMode.FULL
    )
    warning.assert_called_once()
    message, requested, effective, operation, reason = warning.call_args.args
    assert "requested_mode=%s" in message
    assert "effective_mode=%s" in message
    assert requested == "FULL"
    assert effective == "FULL_AND_PIECEWISE"
    assert operation == "ATB PagedAttentionOperation"
    assert "qLensTensor" in reason
    assert "hostData" in reason


def test_full_dflash_fallback_is_bound_once_at_public_platform_entry():
    config = _config()
    seen_modes = []

    def original(_cls, current_config):
        seen_modes.append(
            current_config.compilation_config.cudagraph_mode
        )

    with (
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310."
            "_original_check_and_update_config_310",
            side_effect=original,
        ),
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310.logger.warning"
        ) as warning,
    ):
        NPUPlatform.check_and_update_config(config)
        NPUPlatform.check_and_update_config(config)

    assert seen_modes == [
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ]
    warning.assert_called_once()


@pytest.mark.parametrize(
    ("method", "mode", "layer_types"),
    [
        pytest.param(
            "dflash",
            CUDAGraphMode.FULL_AND_PIECEWISE,
            ("linear_attention", "full_attention"),
            id="other-dflash-mode",
        ),
        pytest.param(
            "dspark",
            CUDAGraphMode.FULL,
            ("linear_attention", "full_attention"),
            id="other-speculative-method",
        ),
        pytest.param(
            "dflash",
            CUDAGraphMode.FULL,
            ("linear_attention",),
            id="no-atb-full-attention",
        ),
        pytest.param(
            "dflash",
            CUDAGraphMode.FULL,
            None,
            id="missing-layer-types",
        ),
        pytest.param(
            None,
            CUDAGraphMode.FULL,
            ("linear_attention", "full_attention"),
            id="non-speculative",
        ),
    ],
)
def test_full_dflash_hybrid_fallback_is_narrowly_scoped(
    method,
    mode,
    layer_types,
):
    config = _config(method, mode, layer_types)

    with (
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310."
            "_original_check_and_update_config_310"
        ) as original,
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310.logger.warning"
        ) as warning,
    ):
        check_and_update_config_310(object, config)

    original.assert_called_once_with(object, config)
    assert config.compilation_config.cudagraph_mode == mode
    assert not hasattr(
        config.compilation_config,
        DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
    )
    warning.assert_not_called()


def test_full_dflash_capability_check_does_not_mask_config_errors():
    class BrokenLayerTypes:
        def __contains__(self, _value):
            raise RuntimeError("layer type discovery failed")

    config = _config(layer_types=BrokenLayerTypes())
    with (
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310."
            "_original_check_and_update_config_310"
        ) as original,
        patch(
            "vllm_ascend.patch.platform."
            "patch_dflash_graph_config_310.logger.warning"
        ) as warning,
        pytest.raises(RuntimeError, match="layer type discovery failed"),
    ):
        check_and_update_config_310(object, config)

    original.assert_not_called()
    warning.assert_not_called()

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.patch.platform.patch_dflash_graph_config_310 import (
    DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
    check_and_update_config_310,
)
from vllm_ascend.platform import NPUPlatform


_DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310 = (
    "_dflash_effective_cudagraph_mode_310"
)
_MISSING = object()


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


def _qwen36_config(
    mode=CUDAGraphMode.FULL_DECODE_ONLY,
    *,
    method="dflash",
    num_speculative_tokens=15,
    draft_tensor_parallel_size=2,
    quantization="ascend",
    tensor_parallel_size=2,
    model_type="qwen3_5_moe",
    text_model_type="qwen3_5_moe_text",
    num_hidden_layers=40,
    num_experts=256,
    linear_num_value_heads=32,
    layer_types=("linear_attention", "full_attention"),
):
    text_config_kwargs = {
        "model_type": text_model_type,
        "num_hidden_layers": num_hidden_layers,
        "num_experts": num_experts,
        "linear_num_value_heads": linear_num_value_heads,
    }
    if layer_types is not _MISSING:
        text_config_kwargs["layer_types"] = layer_types
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
            draft_tensor_parallel_size=draft_tensor_parallel_size,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        model_config=SimpleNamespace(
            quantization=quantization,
            hf_config=SimpleNamespace(model_type=model_type),
            hf_text_config=SimpleNamespace(**text_config_kwargs),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
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
    assert (
        getattr(
            config.compilation_config,
            _DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310,
        )
        == CUDAGraphMode.FULL_AND_PIECEWISE
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


@pytest.mark.parametrize(
    ("mode", "expected_operation_fragments", "expected_reason_fragments"),
    [
        pytest.param(
            CUDAGraphMode.PIECEWISE,
            ("HCCL",),
            (
                "Insufficient_Event_Resources",
                "three tested two-gear controls",
            ),
            id="piecewise-hccl-events",
        ),
        pytest.param(
            CUDAGraphMode.FULL_DECODE_ONLY,
            ("CANN GatherV2",),
            ("MTE DDR address",),
            id="full-decode-gather",
        ),
        pytest.param(
            CUDAGraphMode.FULL_AND_PIECEWISE,
            ("HCCL", "CANN GatherV2"),
            (
                "Insufficient_Event_Resources",
                "three tested two-gear controls",
                "MTE DDR address",
            ),
            id="combined-both-boundaries",
        ),
        pytest.param(
            CUDAGraphMode.FULL,
            ("ATB PagedAttentionOperation", "HCCL", "CANN GatherV2"),
            (
                "qLensTensor hostData",
                "Insufficient_Event_Resources",
                "three tested two-gear controls",
                "MTE DDR address",
            ),
            id="full-all-boundaries",
        ),
    ],
)
def test_qwen36_w8a8_k15_dflash_graphs_fallback_before_compile(
    mode,
    expected_operation_fragments,
    expected_reason_fragments,
):
    config = _qwen36_config(mode)
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

    assert seen_modes == [CUDAGraphMode.NONE]
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert (
        getattr(
            config.compilation_config,
            DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
        )
        == mode
    )
    assert (
        getattr(
            config.compilation_config,
            _DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310,
        )
        == CUDAGraphMode.NONE
    )
    warning.assert_called_once()
    message, requested, effective, operations, reason = warning.call_args.args
    assert "requested_mode=%s" in message
    assert "effective_mode=%s" in message
    assert requested == mode.name
    assert effective == CUDAGraphMode.NONE.name
    for fragment in expected_operation_fragments:
        assert fragment in operations
    for fragment in expected_reason_fragments:
        assert fragment in reason


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"method": "dspark"}, id="other-method"),
        pytest.param({"num_speculative_tokens": 7}, id="other-k"),
        pytest.param(
            {"draft_tensor_parallel_size": 1},
            id="other-draft-tp",
        ),
        pytest.param({"mode": CUDAGraphMode.NONE}, id="eager"),
        pytest.param(
            {"quantization": "compressed-tensors"},
            id="other-quantization",
        ),
        pytest.param({"tensor_parallel_size": 1}, id="other-tp"),
        pytest.param({"model_type": "qwen3_5"}, id="other-model"),
        pytest.param(
            {"text_model_type": "qwen3_5_text"},
            id="other-text-model",
        ),
        pytest.param({"num_hidden_layers": 24}, id="other-layer-count"),
        pytest.param({"num_experts": 128}, id="other-expert-count"),
        pytest.param(
            {"linear_num_value_heads": 16},
            id="other-value-head-count",
        ),
        pytest.param({"layer_types": _MISSING}, id="missing-layer-types"),
        pytest.param({"layer_types": None}, id="none-layer-types"),
        pytest.param(
            {"layer_types": ("linear_attention",)},
            id="no-full-attention",
        ),
    ],
)
def test_qwen36_graph_fallback_is_exactly_scoped(overrides):
    config_kwargs = {"mode": CUDAGraphMode.PIECEWISE, **overrides}
    config = _qwen36_config(**config_kwargs)

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
    assert (
        config.compilation_config.cudagraph_mode
        == overrides.get("mode", CUDAGraphMode.PIECEWISE)
    )
    assert not hasattr(
        config.compilation_config,
        DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
    )
    assert not hasattr(
        config.compilation_config,
        _DFLASH_EFFECTIVE_GRAPH_MODE_ATTR_310,
    )
    warning.assert_not_called()


def test_qwen36_auto_detected_ascend_quantization_falls_back():
    config = _qwen36_config(
        CUDAGraphMode.FULL_DECODE_ONLY,
        quantization=None,
    )
    seen_modes = []

    def auto_detect(current_config):
        current_config.model_config.quantization = "ascend"

    def original(_cls, current_config):
        seen_modes.append(
            current_config.compilation_config.cudagraph_mode
        )

    with (
        patch(
            "vllm_ascend.quantization.utils."
            "maybe_auto_detect_quantization",
            side_effect=auto_detect,
        ) as detect,
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

    detect.assert_called_once_with(config)
    assert seen_modes == [CUDAGraphMode.NONE]
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert (
        getattr(
            config.compilation_config,
            DFLASH_REQUESTED_GRAPH_MODE_ATTR_310,
        )
        == CUDAGraphMode.FULL_DECODE_ONLY
    )
    assert warning.call_count == 1


@pytest.mark.parametrize("detected_quantization", [None, "compressed-tensors"])
def test_qwen36_non_ascend_auto_detection_does_not_fallback(
    detected_quantization,
):
    config = _qwen36_config(
        CUDAGraphMode.FULL_DECODE_ONLY,
        quantization=None,
    )

    def auto_detect(current_config):
        current_config.model_config.quantization = detected_quantization

    with (
        patch(
            "vllm_ascend.quantization.utils."
            "maybe_auto_detect_quantization",
            side_effect=auto_detect,
        ) as detect,
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

    detect.assert_called_once_with(config)
    original.assert_called_once_with(object, config)
    assert (
        config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
    )
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

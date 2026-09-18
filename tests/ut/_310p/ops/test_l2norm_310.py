from unittest.mock import patch

import torch

import vllm_ascend._310p.ops.fla.l2norm as l2norm_310


def test_l2norm_weight_cache_uses_device_values_and_stays_bounded():
    cached = l2norm_310._l2norm_unit_weight
    cached.cache_clear()
    try:
        first = cached(128, torch.float16, torch.device("cpu"))
        second = cached(128, torch.float16, torch.device("cpu"))
        assert first is second
        assert cached(128, torch.float32, torch.device("cpu")).dtype == torch.float32
        for dim in range(8, 20):
            value = cached(dim, torch.float16, torch.device("cpu"))
            assert torch.equal(value, torch.full((dim,), dim**-0.5, dtype=torch.float16))
        assert cached.cache_info().currsize == 8
    finally:
        cached.cache_clear()


def test_l2norm_310p_uses_adn_dispatch_for_fp16():
    x = torch.randn(2, 3, 256, dtype=torch.float16)
    expected = torch.randn_like(x)
    fallback = torch.randn_like(x)

    with (
        patch.object(
            l2norm_310,
            "adn_rms_norm_or_fallback",
            return_value=expected.reshape(-1, 256),
        ) as experimental_dispatch,
        patch(
            "torch_npu.npu_rms_norm",
            return_value=(fallback.reshape(-1, 256), None),
        ) as baseline,
    ):
        out = l2norm_310.l2norm_310p(x)

    experimental_dispatch.assert_called_once()
    candidate_x, candidate_weight, candidate_eps = experimental_dispatch.call_args.args
    assert experimental_dispatch.call_args.kwargs == {}
    assert torch.equal(candidate_x, x.reshape(-1, 256))
    assert torch.equal(
        candidate_weight,
        torch.full((256,), 1.0 / (256**0.5), dtype=torch.float16),
    )
    assert candidate_eps == 1e-6 / 256
    baseline.assert_not_called()
    assert torch.equal(out, expected)

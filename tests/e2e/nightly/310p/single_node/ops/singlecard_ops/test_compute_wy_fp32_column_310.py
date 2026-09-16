"""Regression coverage for column-oriented fp32 forward substitution."""

import statistics

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend._310p.ops.fla.chunk_gated_delta_rule as chunk_mod
from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64
FP32_FS_ROW_SUM_THRESHOLD = 2.5
# The vectorized solve measures about 0.31 ms; allow headroom for 310P jitter.
STABLE_PATH_T128_KV128_MAX_MEDIAN_MS = 0.40
PERF_CALLS_PER_TRIAL = 50
PERF_TRIALS = 7


@pytest.mark.parametrize("k_dim,v_dim", [(64, 64), (64, 128), (128, 64), (128, 128)])
@pytest.mark.parametrize("batch,tokens,k_heads,v_heads", [(1, 64, 3, 3), (2, 128, 2, 4)])
def test_fp32_wy_column_updates_shapes_and_graph(k_dim, v_dim, batch, tokens, k_heads, v_heads):
    """Mixed safe/fast heads, multiple chunks/batches, and both supported widths."""
    enable_custom_op()
    torch.manual_seed(20260916)
    q = torch.nn.functional.normalize(torch.randn(batch, tokens, k_heads, k_dim), dim=-1).half()
    base = torch.randn(batch, 1, k_heads, k_dim)
    k = torch.nn.functional.normalize(base + 0.02 * torch.randn(batch, tokens, k_heads, k_dim), dim=-1).half()
    v = (0.1 * torch.randn(batch, tokens, v_heads, v_dim)).half()
    g = torch.full((batch, tokens, v_heads), -0.001, dtype=torch.float32)
    beta = torch.full((batch, tokens, v_heads), 0.8, dtype=torch.float16)
    # Keep one head on the compensated path and the others on the fp32 path.
    beta[..., 0] = 0.001
    a, _, _, _ = chunk_mod._wy_build_A_and_R(k, v, g, beta, CHUNK_SIZE)
    row_max = a.abs().sum(-1).amax(-1)
    assert (row_max < FP32_FS_ROW_SUM_THRESHOLD).any() and (row_max >= FP32_FS_ROW_SUM_THRESHOLD).any()
    reference = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
    inputs = tuple(x.npu() for x in (q, k, v, g, beta))
    call = lambda: torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(*inputs, CHUNK_SIZE)
    eager = call()
    cpu = tuple(x.cpu() for x in eager)
    for i in (0, 1):
        torch.testing.assert_close(cpu[i], reference[i], rtol=0, atol=0)
    torch.testing.assert_close(cpu[4], reference[4], rtol=1e-5, atol=1e-5)
    for i, tolerance in ((2, 6e-5), (3, 2.5e-5)):
        assert torch.isfinite(cpu[i]).all()
        relative = (cpu[i].double() - reference[i].double()).norm() / reference[i].double().norm()
        assert relative <= tolerance, (i, relative.item())
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = call()
    for _ in range(5):
        graph.replay()
        for actual, expected in zip(captured, cpu):
            assert torch.equal(actual.cpu(), expected)


def test_compute_wy_stable_path_t128_kv128_performance():
    """Keep the correlated-input fp32 solve within its 310P graph budget."""
    enable_custom_op()
    torch.manual_seed(20260916)
    batch, tokens, k_heads, v_heads, dim = 1, 128, 8, 16, 128
    q = torch.nn.functional.normalize(torch.randn(batch, tokens, k_heads, dim), dim=-1).half()
    base = torch.randn(batch, 1, k_heads, dim)
    k = torch.nn.functional.normalize(base + 0.02 * torch.randn(batch, tokens, k_heads, dim), dim=-1).half()
    v = (0.1 * torch.randn(batch, tokens, v_heads, dim)).half()
    g = torch.full((batch, tokens, v_heads), -0.001, dtype=torch.float32)
    beta = torch.full((batch, tokens, v_heads), 0.8, dtype=torch.float16)
    a, _, _, _ = chunk_mod._wy_build_A_and_R(k, v, g, beta, CHUNK_SIZE)
    assert (a.abs().sum(-1).amax(-1) >= FP32_FS_ROW_SUM_THRESHOLD).all()
    inputs = tuple(x.npu() for x in (q, k, v, g, beta))
    call = lambda: torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(*inputs, CHUNK_SIZE)
    eager = tuple(x.cpu() for x in call())
    graph = torch.npu.NPUGraph()
    captured = []
    torch.npu.synchronize()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        for _ in range(PERF_CALLS_PER_TRIAL):
            captured.append(call())
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    elapsed_ms = []
    for _ in range(PERF_TRIALS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end) / PERF_CALLS_PER_TRIAL)
    for actual, expected in zip(captured[-1], eager):
        assert torch.equal(actual.cpu(), expected)
    median_ms = statistics.median(elapsed_ms)
    print(f"stable-path T128 K/V128 median: {median_ms:.6f} ms")
    assert median_ms <= STABLE_PATH_T128_KV128_MAX_MEDIAN_MS

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Replay trusted captured inputs to check and benchmark 310P batch x_scale.

Each .pt file contains CPU tensors: x, weight (logical E,N,K), scale and
groups (cumulative endpoints). No weights or request data are bundled here.
Scale calculation is included in candidate timing, including graph replay.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend._310p.fused_moe.moe_mlp import _quant_grouped_matmul


def measure(function, graph_mode, iterations, rounds):
    for _ in range(10):
        function()
    torch.npu.synchronize()
    graph = None
    captured_output = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured_output = function()
        function = graph.replay
        for _ in range(10):
            function()
        torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    # Retain the captured output for the entire replay lifetime.
    del captured_output
    return {"median_us": statistics.median(samples), "rounds_us": samples}


def run_case(path, iterations, rounds):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    x = payload["x"].npu()
    weight = torch_npu.npu_format_cast(payload["weight"].npu(), 29)
    scale = payload["scale"].npu()
    groups = payload["groups"].npu()

    def baseline():
        return torch_npu.npu_quant_grouped_matmul_dequant(
            x=x, quantized_weight=weight, weight_scale=scale, group_list=groups, quant_mode="pertoken"
        )

    def candidate():
        return _quant_grouped_matmul(x, weight, scale, groups)

    # Numerical checks and CPU copies are outside the timed region.
    expected, actual = baseline().cpu(), candidate().cpu()
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise AssertionError(f"Non-finite GMM output: {path.name}")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = {"case": path.name, "x_shape": list(x.shape), "weight_shape": list(weight.shape), "exact_equal": True}
    for graph_mode in (False, True):
        ref = measure(baseline, graph_mode, iterations, rounds)
        opt = measure(candidate, graph_mode, iterations, rounds)
        result["graph" if graph_mode else "eager"] = {
            "baseline": ref,
            "batch_scale": opt,
            "speedup": ref["median_us"] / opt["median_us"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Trusted captured .pt files")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 1:
        parser.error("iterations and rounds must be positive")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    with torch.inference_mode():
        for path in args.inputs:
            print(json.dumps(run_case(path, args.iterations, args.rounds)), flush=True)


if __name__ == "__main__":
    main()

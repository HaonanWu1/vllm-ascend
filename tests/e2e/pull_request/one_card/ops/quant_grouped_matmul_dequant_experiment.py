# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit opt-in regression for the unregistered 310P MoE experiment.

Run in separate processes for the installed kernel and a process-private OPP
overlay. This script does not install an operator or change the toolkit.
Captures must contain x, weight (ND), scale, groups, and layer from real weights.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu


def input_cases(data, routing):
    values, groups = data["x"], data["groups"]
    cases = [("real", values, groups)]
    cases.extend(
        (f"routing_{i}", values, torch.tensor(record["group_endpoints"][data["layer"]], dtype=torch.int64))
        for i, record in enumerate(routing)
    )
    for size in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 32, 64]:
        counts = torch.zeros(256, dtype=torch.int64)
        full = min(255, 640 // size)
        counts[:full] = size
        counts[full] = 640 - full * size
        cases.append((f"group_size_{size}", values, counts.cumsum(0)))
    generator = torch.Generator().manual_seed(20260916)
    for i in range(4):
        cases.append((f"random_{i}", torch.randn(values.shape, generator=generator).half(), groups))
    mixed = values.clone()
    mixed[::2] = 0
    halfway = torch.randint(-100, 101, values.shape, generator=generator).float().add_(0.5).half()
    halfway[:, 0] = 127
    tiny = torch.randn(values.shape, generator=generator).mul_(1e-5).half()
    cases.extend(
        [
            ("zero", torch.zeros_like(values), groups),
            ("mixed_zero", mixed, groups),
            ("halfway", halfway, groups),
            ("tiny", tiny, groups),
        ]
    )
    return cases


def check_output(actual, filename, options):
    assert torch.isfinite(actual).all(), filename
    if options.reference is None:
        torch.save(actual, options.output / filename)
        return {"finite": True, "reference_written": True}
    expected = torch.load(options.reference / filename, weights_only=True, map_location="cpu")
    equal = torch.equal(actual, expected)
    error = (actual.float() - expected.float()).abs().max().item()
    assert equal, f"{filename}: max_abs={error}"
    return {"finite": True, "bitwise_equal": equal, "max_abs": error}


def run(options):
    # Offline checks intentionally synchronize; none of this code is imported
    # into the service or its performance-critical operator path.
    files = sorted(options.capture.glob("layer*.pt"))
    assert files, "No real-weight captures found"
    options.output.mkdir(parents=True, exist_ok=False)
    routing = [json.loads(line) for line in (options.capture / "groups_rank0.jsonl").read_text().splitlines()]
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    results = []
    with torch.inference_mode():
        for path in files:
            data = torch.load(path, weights_only=True, map_location="cpu")
            assert data["x"].shape[0] == 640
            x, groups = data["x"].npu(), data["groups"].npu()
            weight = torch_npu.npu_format_cast(data["weight"].npu(), 29)
            weight_scale = data["scale"].npu()

            def compute(values=x, ends=groups, weights=weight, scales=weight_scale):
                absmax = values.abs().amax(-1).float()
                x_scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127)
                return torch_npu.npu_quant_grouped_matmul_dequant(
                    x=values,
                    quantized_weight=weights,
                    weight_scale=scales,
                    group_list=ends,
                    quant_mode="pertoken",
                    x_scale=x_scale,
                )

            for _ in range(3):
                compute()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = compute()
            for name, values, ends in input_cases(data, routing):
                x.copy_(values)
                groups.copy_(ends)
                graph.replay()
                stats = check_output(output.cpu(), f"{path.stem}_{name}.pt", options)
                results.append({"case": path.stem, "test": name, "graph": True, **stats})
            actual = compute(data["x"][:64].npu(), data["groups"].clamp_max(64).npu()).cpu()
            stats = check_output(actual, f"{path.stem}_fallback64.pt", options)
            results.append({"case": path.stem, "test": "fallback64", "graph": False, **stats})
            (options.output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps({"capture": path.name, "passed": len(results)}), flush=True)
            del graph, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    run(parser.parse_args())

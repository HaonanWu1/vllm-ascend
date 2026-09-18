# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit real-weight regression for the 310P grouped MoE operator.

Select the installed CANN kernel or the independently registered custom op.
This script does not install an operator or change the toolkit.
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
    # Static expert-ID ownership can strand all small experts on one shard.
    # Include a large expert elsewhere to exercise cross-phase synchronization.
    for lane in range(4):
        counts = torch.zeros(256, dtype=torch.int64)
        counts[lane:252:4] = 8
        counts[(lane + 1) % 4] = 640 - int(counts.sum())
        cases.append((f"imbalanced_lane{lane}", values, counts.cumsum(0)))
    for index in [0, 255]:
        counts = torch.zeros(256, dtype=torch.int64)
        counts[index] = 640
        cases.append((f"one_large_{index}", values, counts.cumsum(0)))
    counts = torch.zeros(256, dtype=torch.int64)
    counts[:80] = 8
    cases.append(("all_small8", values, counts.cumsum(0)))
    counts.zero_()
    remaining = 640
    for index in range(256):
        counts[index] = min(remaining, 8 + index % 2)
        remaining -= int(counts[index])
    cases.append(("alternating8_9", values, counts.cumsum(0)))
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
    # Exercise padded-M boundaries and both sides of the 80-row private slot.
    # Keep the names used by the frozen candidate regression references.
    for size in (31, 63, 65, 79, 80, 81, 82, 127, 128, 129):
        counts = torch.zeros(256, dtype=torch.int64)
        full = min(255, 640 // size)
        counts[:full] = size
        counts[full] = 640 - full * size
        cases.append((f"extended_group_size_{size}", values, counts.cumsum(0)))
    counts = torch.zeros(256, dtype=torch.int64)
    sizes = [8, 9, 16, 17, 79, 80, 81]
    sizes.append(640 - sum(sizes))
    for index, size in zip((0, 4, 31, 64, 128, 192, 254, 255), sizes):
        counts[index] = size
    cases.append(("mixed_independent_shared_boundary", values, counts.cumsum(0)))
    return cases


def check_output(actual, filename, options):
    assert actual.dtype == torch.float16, filename
    assert torch.isfinite(actual).all(), filename
    reference = (options.reference or options.output) / filename
    if options.reference is None and not reference.exists():
        torch.save(actual, reference)
        return {"finite": True, "reference_written": True}
    expected = torch.load(reference, weights_only=True, map_location="cpu")
    assert actual.shape == expected.shape and expected.dtype == torch.float16, filename
    # Float equality alone does not distinguish positive and negative zero.
    equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    error = (actual.float() - expected.float()).abs().max().item()
    assert equal, f"{filename}: max_abs={error}"
    return {"finite": True, "bitwise_equal": equal, "max_abs": error}


def run(options):
    # Offline checks intentionally synchronize; none of this code is imported
    # into the service or its performance-critical operator path.
    files = sorted(options.capture.glob("layer*.pt"))
    assert files, "No real-weight captures found"
    options.output.mkdir(parents=True, exist_ok=False)
    route_files = sorted(options.capture.glob("groups_rank*.jsonl"))
    assert route_files, "No routing captures found"
    routing = [json.loads(line) for path in route_files for line in path.read_text().splitlines()]
    assert options.replays > 0, "At least one replay is required"
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    use_custom = getattr(options, "implementation", "cann") == "custom"
    if use_custom:
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op(), "Custom extension must be built for this regression"
        assert hasattr(torch.ops._C_ascend, "npu_quant_grouped_matmul_dequant_310")
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
                if use_custom and values.shape[0] == 640:
                    return torch.ops._C_ascend.npu_quant_grouped_matmul_dequant_310(
                        values, weights, scales, ends, x_scale
                    )
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
                assert ends.shape == (256,) and ends[-1] == 640
                assert bool((torch.diff(ends) >= 0).all()) and ends[0] >= 0
                x.copy_(values)
                groups.copy_(ends)
                for replay in range(options.replays):
                    graph.replay()
                    stats = check_output(output.cpu(), f"{path.stem}_{name}.pt", options)
                    results.append({"case": path.stem, "test": name, "replay": replay, "graph": True, **stats})
            del graph, output
            # M != 640 must retain the old dispatch, including either side of
            # the guard boundary. Check eager execution and repeated graphs.
            for rows in [64, 639, 641]:
                values = data["x"][:rows] if rows <= 640 else torch.cat([data["x"], data["x"][:1]])
                ends = data["groups"].clamp_max(rows).clone()
                ends[-1] = rows
                fallback_x, fallback_groups = values.npu(), ends.npu()
                filename = f"{path.stem}_fallback{rows}.pt"
                actual = compute(fallback_x, fallback_groups).cpu()
                stats = check_output(actual, filename, options)
                results.append({"case": path.stem, "test": f"fallback{rows}", "graph": False, **stats})
                for _ in range(3):
                    compute(fallback_x, fallback_groups)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = compute(fallback_x, fallback_groups)
                for replay in range(options.replays):
                    graph.replay()
                    stats = check_output(output.cpu(), filename, options)
                    results.append(
                        {"case": path.stem, "test": f"fallback{rows}", "replay": replay, "graph": True, **stats}
                    )
                del graph, output
            (options.output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps({"capture": path.name, "passed": len(results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--implementation", choices=("cann", "custom"), default="cann")
    parser.add_argument(
        "--replays", type=int, default=5, help="Graph replays per input; first baseline output is frozen"
    )
    run(parser.parse_args())

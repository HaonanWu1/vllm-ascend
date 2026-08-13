# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed 310P DFlash validation workload.

This module intentionally lives under tests: it observes inference and metrics but
does not patch or otherwise change the production execution path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

DFLASH_K = 15
MAX_MODEL_LEN = 2048
MAX_NUM_BATCHED_TOKENS = 2048
MAX_NUM_SEQS = 8
MAX_OUTPUT_TOKENS = 128
GRAPH_CAPTURE_SIZES = [16, 32, 64, 128]

WARMUP_PROMPTS = [
    "Compute 19 * 23. Explain the arithmetic briefly and give the final answer.",
]

MEASURED_PROMPTS = [
    "Compute 27 * 14. Explain the arithmetic briefly and give the final answer.",
    "A box has 48 red balls and 37 blue balls. How many balls are there? Explain briefly.",
    "Write a short Python function that returns the sum of the integers from 1 through n.",
    "Explain in three concise steps why water freezes when its temperature falls sufficiently.",
    "A train travels 180 kilometers in 3 hours. Calculate its average speed and show the formula.",
    "List the first eight prime numbers, then state how many of them are odd.",
    "Translate 'Reliable systems make failures visible' into Chinese and explain the key noun.",
    "Describe a deterministic procedure for checking whether a positive integer is even.",
]


@dataclass(frozen=True)
class ModelProfile:
    target: str
    draft: str
    tensor_parallel_size: int = 1
    dtype: str = "float16"


@dataclass(frozen=True)
class RunProfile:
    uses_dflash: bool
    enforce_eager: bool
    cudagraph_mode: str | None
    temperature: float = 0


@dataclass(frozen=True)
class WorkloadMetrics:
    draft_rounds: int
    drafted_tokens: int
    accepted_tokens: int


MODEL_PROFILES = {
    "qwen3.5-2b": ModelProfile(
        target="/home/models/Qwen3.5-2B",
        draft="/home/models/Qwen3.5-2B-Dflash",
    ),
    "qwen3.6-35b-a3b-w8a8": ModelProfile(
        target="/home/models/Qwen3.6-35B-A3B-w8a8",
        draft="/home/models/Qwen3.6-35B-A3B-DFlash",
    ),
}

RUN_PROFILES = {
    "target-only": RunProfile(False, True, None),
    "eager": RunProfile(True, True, None),
    "PIECEWISE": RunProfile(True, False, "PIECEWISE"),
    "FULL_DECODE_ONLY": RunProfile(True, False, "FULL_DECODE_ONLY"),
    "FULL_AND_PIECEWISE": RunProfile(True, False, "FULL_AND_PIECEWISE"),
    "FULL": RunProfile(True, False, "FULL"),
}

_SPEC_COUNTER_NAMES = {
    "draft_rounds": "vllm:spec_decode_num_drafts",
    "drafted_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def _counter_values(metrics: Sequence[Any]) -> dict[str, int]:
    values = {field: 0 for field in _SPEC_COUNTER_NAMES}
    names_to_fields = {name: field for field, name in _SPEC_COUNTER_NAMES.items()}
    for metric in metrics:
        field = names_to_fields.get(getattr(metric, "name", None))
        if field is not None:
            values[field] += int(metric.value)
    return values


def collect_speculative_counters(
    before: Sequence[Any],
    after: Sequence[Any],
) -> WorkloadMetrics:
    """Return measured-workload counter deltas, excluding warmup activity."""
    before_values = _counter_values(before)
    after_values = _counter_values(after)
    deltas = {
        field: after_values[field] - before_values[field]
        for field in _SPEC_COUNTER_NAMES
    }
    decreased = [field for field, value in deltas.items() if value < 0]
    if decreased:
        raise ValueError(f"speculative counters decreased during the run: {decreased}")
    return WorkloadMetrics(**deltas)


def summarize_run(
    counters: WorkloadMetrics,
    output_token_ids: Sequence[Sequence[int]],
    elapsed_seconds: float,
) -> dict[str, int | float | None]:
    """Apply the single acceptance and performance formula used by all runs.

    One target token is produced per draft round even when no draft token is
    accepted, so average accepted length is ``1 + accepted / draft_rounds``.
    It is not applicable to target-only runs, which have no draft rounds.
    """
    if min(asdict(counters).values()) < 0:
        raise ValueError("workload counters must be non-negative")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed steady-state time must be positive")

    measured_output_tokens = sum(len(token_ids) for token_ids in output_token_ids)
    average_accepted_length = None
    if counters.draft_rounds:
        average_accepted_length = 1 + (
            counters.accepted_tokens / counters.draft_rounds
        )

    return {
        **asdict(counters),
        "average_accepted_length": average_accepted_length,
        "measured_output_tokens": measured_output_tokens,
        "elapsed_steady_state_seconds": elapsed_seconds,
        "output_tokens_per_second": measured_output_tokens / elapsed_seconds,
    }


def _build_llm(model_profile: ModelProfile, run_profile: RunProfile):
    from vllm import LLM
    from vllm.config import CompilationConfig

    kwargs: dict[str, Any] = {
        "model": model_profile.target,
        "tensor_parallel_size": model_profile.tensor_parallel_size,
        "dtype": model_profile.dtype,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "enforce_eager": run_profile.enforce_eager,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    if run_profile.uses_dflash:
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": model_profile.draft,
            "num_speculative_tokens": DFLASH_K,
        }
    if run_profile.cudagraph_mode is not None:
        kwargs["compilation_config"] = CompilationConfig(
            cudagraph_mode=run_profile.cudagraph_mode,
            cudagraph_capture_sizes=GRAPH_CAPTURE_SIZES,
        )
    return LLM(**kwargs)


def _generate(llm: Any, prompts: Sequence[str], temperature: float):
    from vllm import SamplingParams

    return llm.generate(
        list(prompts),
        SamplingParams(
            temperature=temperature,
            ignore_eos=False,
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
    )


def run_validation(model_name: str, run_name: str) -> dict[str, Any]:
    import torch

    model_profile = MODEL_PROFILES[model_name]
    run_profile = RUN_PROFILES[run_name]
    llm = _build_llm(model_profile, run_profile)

    _generate(llm, WARMUP_PROMPTS, run_profile.temperature)
    metrics_before = llm.get_metrics()
    torch.npu.synchronize()
    start = time.perf_counter()
    outputs = _generate(llm, MEASURED_PROMPTS, run_profile.temperature)
    torch.npu.synchronize()
    elapsed_seconds = time.perf_counter() - start
    metrics_after = llm.get_metrics()

    output_token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    counters = collect_speculative_counters(metrics_before, metrics_after)
    if run_profile.uses_dflash and counters.draft_rounds == 0:
        raise RuntimeError("DFlash run produced no speculative draft rounds")

    return {
        "configuration": {
            "model": model_name,
            "run": run_name,
            "target_model_path": model_profile.target,
            "draft_model_path": model_profile.draft if run_profile.uses_dflash else None,
            "num_speculative_tokens": DFLASH_K if run_profile.uses_dflash else 0,
            "temperature": run_profile.temperature,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "cudagraph_mode": run_profile.cudagraph_mode,
            "graph_capture_sizes": (
                GRAPH_CAPTURE_SIZES if run_profile.cudagraph_mode else []
            ),
        },
        "metrics": summarize_run(counters, output_token_ids, elapsed_seconds),
        "outputs": [
            {
                "prompt": prompt,
                "token_ids": token_ids,
                "text": output.outputs[0].text,
            }
            for prompt, token_ids, output in zip(
                MEASURED_PROMPTS, output_token_ids, outputs
            )
        ],
    }


def write_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_PROFILES, required=True)
    parser.add_argument("--run", choices=RUN_PROFILES, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete JSON result directly from the main process.",
    )
    args = parser.parse_args()
    result = run_validation(args.model, args.run)
    if args.output is not None:
        write_result(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

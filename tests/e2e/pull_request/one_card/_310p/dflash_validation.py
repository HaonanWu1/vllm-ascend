# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed 310P DFlash validation workload.

This module intentionally lives under tests: it observes inference and metrics but
does not patch or otherwise change the production execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

DFLASH_K = 15
AVERAGE_ACCEPTED_LENGTH_FORMULA = (
    "mean(request_output_tokens / request_draft_rounds)"
)
GLOBAL_COUNTER_AVERAGE_ACCEPTED_LENGTH_FORMULA = (
    "1 + accepted_tokens / draft_rounds"
)
COMPLETION_TOKENS_PER_DRAFT_ROUND_FORMULA = (
    "measured_output_tokens / draft_rounds"
)
MAX_MODEL_LEN = 2048
MAX_NUM_BATCHED_TOKENS = 2048
MAX_NUM_SEQS = 8
MAX_OUTPUT_TOKENS = 128
SAMPLING_TOP_P = 1
SAMPLING_TOP_K = 1
GRAPH_CAPTURE_SIZES = [16, 32, 64, 128]
WORKLOAD_NAMES = ("mixed-short", "math500")
MATH500_DATASET_PATH = Path("/home/datasets/math500-hf-6e4ed1a2/test.jsonl")
MATH500_DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_DATASET_SHA256 = (
    "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
)
MATH500_BENCHMARK_REVISION = "94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756"
MATH500_MEASURED_PROMPT_COUNT = 32
MATH500_MAX_OUTPUT_TOKENS = 2048
MATH500_MAX_MODEL_LEN = 4096
MATH500_PROMPT_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

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
class ValidationWorkload:
    name: str
    warmup_prompts: list[str]
    measured_prompts: list[str]
    max_output_tokens: int
    max_model_len: int
    max_num_seqs: int
    sequential: bool
    metadata: dict[str, Any]


def load_math500_prompts(
    dataset_path: Path,
    *,
    expected_sha256: str,
    num_measured: int,
) -> tuple[list[str], list[str]]:
    """Load the official ordered Math500 prompt subset for the eager gate."""
    actual_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Math500 dataset checksum does not match the pinned validation "
            f"revision: expected {expected_sha256}, got {actual_sha256}"
        )
    with dataset_path.open(encoding="utf-8") as dataset_file:
        rows = [json.loads(line) for line in dataset_file]
    required_rows = num_measured + 1
    if len(rows) < required_rows:
        raise ValueError(
            f"Math500 validation requires {required_rows} rows, got {len(rows)}"
        )
    prompts = [f"{row['problem']}\n{MATH500_PROMPT_SUFFIX}" for row in rows]
    return prompts[:1], prompts[1 : num_measured + 1]


def apply_chat_template(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    enable_thinking: bool,
) -> list[str]:
    """Apply the target tokenizer's single-turn user chat template."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in prompts
    ]


def prompt_batches(
    prompts: Sequence[str],
    *,
    sequential: bool,
) -> list[list[str]]:
    """Group prompts according to the validation workload's concurrency."""
    if sequential:
        return [[prompt] for prompt in prompts]
    return [list(prompts)]


def build_math500_workload(
    tokenizer: Any,
    *,
    dataset_path: Path = MATH500_DATASET_PATH,
    expected_sha256: str = MATH500_DATASET_SHA256,
    num_measured: int = MATH500_MEASURED_PROMPT_COUNT,
) -> ValidationWorkload:
    """Build the pinned, concurrency-one Math500 eager-gate workload."""
    warmup_prompts, measured_prompts = load_math500_prompts(
        dataset_path,
        expected_sha256=expected_sha256,
        num_measured=num_measured,
    )
    warmup_prompts = apply_chat_template(
        tokenizer,
        warmup_prompts,
        enable_thinking=True,
    )
    measured_prompts = apply_chat_template(
        tokenizer,
        measured_prompts,
        enable_thinking=True,
    )
    return ValidationWorkload(
        name="math500",
        warmup_prompts=warmup_prompts,
        measured_prompts=measured_prompts,
        max_output_tokens=MATH500_MAX_OUTPUT_TOKENS,
        max_model_len=MATH500_MAX_MODEL_LEN,
        max_num_seqs=1,
        sequential=True,
        metadata={
            "dataset_path": str(dataset_path),
            "dataset_revision": MATH500_DATASET_REVISION,
            "dataset_sha256": expected_sha256,
            "benchmark_revision": MATH500_BENCHMARK_REVISION,
            "measured_dataset_indices": [1, num_measured],
            "prompt_suffix": MATH500_PROMPT_SUFFIX,
            "chat_template": {
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        },
    )


def build_mixed_short_workload() -> ValidationWorkload:
    """Return the original fixed batch used for output/performance checks."""
    return ValidationWorkload(
        name="mixed-short",
        warmup_prompts=WARMUP_PROMPTS,
        measured_prompts=MEASURED_PROMPTS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        sequential=False,
        metadata={
            "prompt_source": "embedded",
            "measured_prompt_count": len(MEASURED_PROMPTS),
            "chat_template": None,
        },
    )


@dataclass(frozen=True)
class ModelProfile:
    target: str
    draft: str
    tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int | None = None
    dtype: str = "float16"
    quantization: str | None = None
    enable_expert_parallel: bool = False
    trust_remote_code: bool = False
    expected_target_revision: str | None = None
    expected_draft_revision: str | None = None


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
        expected_target_revision="c00cc5fd7803c60b7788e053dcce33d0d26b11ef",
        expected_draft_revision="f6f7b57bebf7f208ad92abde37caf095601b48d8",
    ),
    "qwen3.6-35b-a3b-w8a8": ModelProfile(
        target="/home/models/Qwen3.6-35B-A3B-w8a8",
        draft="/home/models/Qwen3.6-35B-A3B-DFlash",
        tensor_parallel_size=2,
        draft_tensor_parallel_size=2,
        quantization="ascend",
        trust_remote_code=True,
        expected_target_revision="1a118d717bcbd59480f4a110fe22181d21711b4d",
        expected_draft_revision="74911aca0cf3156587f4d198b18857e553657cd6",
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
    *,
    request_counters: Sequence[WorkloadMetrics] | None = None,
) -> dict[str, Any]:
    """Apply the pinned benchmark's acceptance and performance formulas.

    The official gate gives every measured request equal weight. The two global
    ratios remain diagnostics because weighting requests by their draft-round
    count can change the benchmark result.
    """
    if min(asdict(counters).values()) < 0:
        raise ValueError("workload counters must be non-negative")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed steady-state time must be positive")

    measured_output_tokens = sum(len(token_ids) for token_ids in output_token_ids)
    global_counter_average_accepted_length = None
    completion_tokens_per_draft_round = None
    if counters.draft_rounds:
        global_counter_average_accepted_length = 1 + (
            counters.accepted_tokens / counters.draft_rounds
        )
        completion_tokens_per_draft_round = (
            measured_output_tokens / counters.draft_rounds
        )

    request_metrics = None
    average_accepted_length = None
    if request_counters is not None:
        if len(request_counters) != len(output_token_ids):
            raise ValueError(
                "request counter count must match measured output count"
            )
        request_counter_sums = WorkloadMetrics(
            draft_rounds=sum(item.draft_rounds for item in request_counters),
            drafted_tokens=sum(item.drafted_tokens for item in request_counters),
            accepted_tokens=sum(item.accepted_tokens for item in request_counters),
        )
        if request_counter_sums != counters:
            raise ValueError(
                "request counters must sum to the workload counters"
            )
        if counters.draft_rounds and any(
            item.draft_rounds == 0 for item in request_counters
        ):
            raise ValueError(
                "every measured request in a speculative run must have draft "
                "rounds"
            )
        request_metrics = []
        request_completion_values = []
        for request_counter, token_ids in zip(request_counters, output_token_ids):
            if min(asdict(request_counter).values()) < 0:
                raise ValueError("request counters must be non-negative")
            request_average_accepted_length = None
            counter_average_accepted_length = None
            request_completion_tokens_per_draft_round = None
            if request_counter.draft_rounds:
                counter_average_accepted_length = 1 + (
                    request_counter.accepted_tokens / request_counter.draft_rounds
                )
                request_completion_tokens_per_draft_round = (
                    len(token_ids) / request_counter.draft_rounds
                )
                request_average_accepted_length = (
                    request_completion_tokens_per_draft_round
                )
                request_completion_values.append(
                    request_completion_tokens_per_draft_round
                )
            request_metrics.append(
                {
                    **asdict(request_counter),
                    "output_tokens": len(token_ids),
                    "average_accepted_length": request_average_accepted_length,
                    "counter_average_accepted_length": (
                        counter_average_accepted_length
                    ),
                    "completion_tokens_per_draft_round": (
                        request_completion_tokens_per_draft_round
                    ),
                }
            )
        if request_completion_values:
            average_accepted_length = sum(
                request_completion_values
            ) / len(request_completion_values)

    return {
        **asdict(counters),
        "average_accepted_length": average_accepted_length,
        "average_accepted_length_formula": AVERAGE_ACCEPTED_LENGTH_FORMULA,
        "global_counter_average_accepted_length": (
            global_counter_average_accepted_length
        ),
        "global_counter_average_accepted_length_formula": (
            GLOBAL_COUNTER_AVERAGE_ACCEPTED_LENGTH_FORMULA
        ),
        "completion_tokens_per_draft_round": (
            completion_tokens_per_draft_round
        ),
        "completion_tokens_per_draft_round_formula": (
            COMPLETION_TOKENS_PER_DRAFT_ROUND_FORMULA
        ),
        "request_metrics": request_metrics,
        "measured_output_tokens": measured_output_tokens,
        "elapsed_steady_state_seconds": elapsed_seconds,
        "output_tokens_per_second": measured_output_tokens / elapsed_seconds,
    }


def _build_llm(
    model_profile: ModelProfile,
    run_profile: RunProfile,
    workload: ValidationWorkload,
):
    from vllm import LLM
    from vllm.config import CompilationConfig

    kwargs: dict[str, Any] = {
        "model": model_profile.target,
        "tensor_parallel_size": model_profile.tensor_parallel_size,
        "dtype": model_profile.dtype,
        "max_model_len": workload.max_model_len,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": workload.max_num_seqs,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "enforce_eager": run_profile.enforce_eager,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    if model_profile.quantization is not None:
        kwargs["quantization"] = model_profile.quantization
    if model_profile.enable_expert_parallel:
        kwargs["enable_expert_parallel"] = True
    if model_profile.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if run_profile.uses_dflash:
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": model_profile.draft,
            "num_speculative_tokens": DFLASH_K,
        }
        if model_profile.draft_tensor_parallel_size is not None:
            kwargs["speculative_config"]["draft_tensor_parallel_size"] = (
                model_profile.draft_tensor_parallel_size
            )
    if run_profile.cudagraph_mode is not None:
        kwargs["compilation_config"] = CompilationConfig(
            cudagraph_mode=run_profile.cudagraph_mode,
            cudagraph_capture_sizes=GRAPH_CAPTURE_SIZES,
        )
    return LLM(**kwargs)


def _model_runtime_configuration(
    model_profile: ModelProfile,
    run_profile: RunProfile,
) -> dict[str, Any]:
    """Return model runtime choices recorded in every validation result."""
    return {
        "tensor_parallel_size": model_profile.tensor_parallel_size,
        "draft_tensor_parallel_size": (
            model_profile.draft_tensor_parallel_size
            if run_profile.uses_dflash
            else None
        ),
        "dtype": model_profile.dtype,
        "quantization": model_profile.quantization,
        "enable_expert_parallel": model_profile.enable_expert_parallel,
        "trust_remote_code": model_profile.trust_remote_code,
    }


def _build_workload(
    workload_name: str,
    model_profile: ModelProfile,
) -> ValidationWorkload:
    if workload_name == "mixed-short":
        return build_mixed_short_workload()
    if workload_name == "math500":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_profile.target,
            trust_remote_code=True,
        )
        return build_math500_workload(tokenizer)
    raise ValueError(f"unknown validation workload: {workload_name}")


def _generate(
    llm: Any,
    prompts: Sequence[str],
    temperature: float,
    *,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    sequential: bool = False,
):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=SAMPLING_TOP_P,
        top_k=SAMPLING_TOP_K,
        ignore_eos=False,
        max_tokens=max_output_tokens,
    )
    outputs = []
    for batch in prompt_batches(prompts, sequential=sequential):
        outputs.extend(llm.generate(batch, sampling_params))
    return outputs


def _generate_sequential_with_counters(
    llm: Any,
    prompts: Sequence[str],
    temperature: float,
    *,
    max_output_tokens: int,
) -> tuple[list[Any], list[WorkloadMetrics]]:
    """Generate one request at a time and retain each counter delta."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=SAMPLING_TOP_P,
        top_k=SAMPLING_TOP_K,
        ignore_eos=False,
        max_tokens=max_output_tokens,
    )
    outputs = []
    request_counters = []
    for prompt in prompts:
        metrics_before = llm.get_metrics()
        outputs.extend(llm.generate([prompt], sampling_params))
        metrics_after = llm.get_metrics()
        request_counters.append(
            collect_speculative_counters(metrics_before, metrics_after)
        )
    return outputs, request_counters


def run_validation(
    model_name: str,
    run_name: str,
    workload_name: str = "mixed-short",
) -> dict[str, Any]:
    import torch

    model_profile = MODEL_PROFILES[model_name]
    run_profile = RUN_PROFILES[run_name]
    workload = _build_workload(workload_name, model_profile)
    llm = _build_llm(model_profile, run_profile, workload)

    _generate(
        llm,
        workload.warmup_prompts,
        run_profile.temperature,
        max_output_tokens=workload.max_output_tokens,
        sequential=workload.sequential,
    )
    metrics_before = llm.get_metrics()
    torch.npu.synchronize()
    start = time.perf_counter()
    request_counters = None
    if workload.sequential:
        outputs, request_counters = _generate_sequential_with_counters(
            llm,
            workload.measured_prompts,
            run_profile.temperature,
            max_output_tokens=workload.max_output_tokens,
        )
    else:
        outputs = _generate(
            llm,
            workload.measured_prompts,
            run_profile.temperature,
            max_output_tokens=workload.max_output_tokens,
            sequential=False,
        )
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
            "workload": workload.name,
            "target_model_path": model_profile.target,
            "expected_target_model_revision": (
                model_profile.expected_target_revision
            ),
            "draft_model_path": model_profile.draft if run_profile.uses_dflash else None,
            "expected_draft_model_revision": (
                model_profile.expected_draft_revision
                if run_profile.uses_dflash
                else None
            ),
            "num_speculative_tokens": DFLASH_K if run_profile.uses_dflash else 0,
            **_model_runtime_configuration(model_profile, run_profile),
            "temperature": run_profile.temperature,
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "max_output_tokens": workload.max_output_tokens,
            "max_model_len": workload.max_model_len,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": workload.max_num_seqs,
            "sequential": workload.sequential,
            "workload_metadata": workload.metadata,
            "cudagraph_mode": run_profile.cudagraph_mode,
            "graph_capture_sizes": (
                GRAPH_CAPTURE_SIZES if run_profile.cudagraph_mode else []
            ),
        },
        "metrics": summarize_run(
            counters,
            output_token_ids,
            elapsed_seconds,
            request_counters=request_counters,
        ),
        "outputs": [
            {
                "prompt": prompt,
                "token_ids": token_ids,
                "text": output.outputs[0].text,
            }
            for prompt, token_ids, output in zip(
                workload.measured_prompts, output_token_ids, outputs
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
        "--workload",
        choices=WORKLOAD_NAMES,
        default="mixed-short",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete JSON result directly from the main process.",
    )
    args = parser.parse_args()
    result = run_validation(args.model, args.run, args.workload)
    if args.output is not None:
        write_result(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

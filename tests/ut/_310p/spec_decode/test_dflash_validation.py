# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.e2e.pull_request.one_card._310p import dflash_validation

from tests.e2e.pull_request.one_card._310p.dflash_validation import (
    DFLASH_K,
    MODEL_PROFILES,
    RUN_PROFILES,
    WorkloadMetrics,
    collect_speculative_counters,
    summarize_run,
    write_result,
)


def _counter(name: str, value: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value)


def test_validation_profiles_fix_workload_and_models() -> None:
    assert DFLASH_K == 15
    assert MODEL_PROFILES["qwen3.5-2b"].target == "/home/models/Qwen3.5-2B"
    assert MODEL_PROFILES["qwen3.5-2b"].draft == "/home/models/Qwen3.5-2B-Dflash"
    assert MODEL_PROFILES["qwen3.5-2b"].expected_target_revision == (
        "c00cc5fd7803c60b7788e053dcce33d0d26b11ef"
    )
    assert MODEL_PROFILES["qwen3.5-2b"].expected_draft_revision == (
        "f6f7b57bebf7f208ad92abde37caf095601b48d8"
    )
    assert MODEL_PROFILES["qwen3.6-35b-a3b-w8a8"].target == (
        "/home/models/Qwen3.6-35B-A3B-w8a8"
    )
    assert MODEL_PROFILES["qwen3.6-35b-a3b-w8a8"].draft == (
        "/home/models/Qwen3.6-35B-A3B-DFlash"
    )
    assert RUN_PROFILES["eager"].enforce_eager is True
    assert RUN_PROFILES["target-only"].uses_dflash is False
    assert all(profile.temperature == 0 for profile in RUN_PROFILES.values())


def test_math500_workload_reserves_warmup_and_formats_official_prompt(
    tmp_path,
) -> None:
    payload = "".join(
        json.dumps({"problem": problem}) + "\n"
        for problem in ("Warmup problem", "Measured one", "Measured two")
    )
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(payload, encoding="utf-8")

    warmup, measured = dflash_validation.load_math500_prompts(
        dataset_path,
        expected_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        num_measured=2,
    )

    assert warmup == [
        "Warmup problem\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.",
    ]
    assert measured == [
        "Measured one\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.",
        "Measured two\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.",
    ]


def test_math500_workload_rejects_unpinned_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"problem": "Changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        dflash_validation.load_math500_prompts(
            dataset_path,
            expected_sha256="0" * 64,
            num_measured=0,
        )


def test_math500_workload_requires_warmup_plus_measured_rows(tmp_path) -> None:
    payload = '{"problem": "Only warmup"}\n'
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="requires 2 rows"):
        dflash_validation.load_math500_prompts(
            dataset_path,
            expected_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            num_measured=1,
        )


def test_math500_chat_template_enables_qwen_thinking() -> None:
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return f"chat:{messages[0]['content']}"

    prompts = dflash_validation.apply_chat_template(
        Tokenizer(),
        ["First", "Second"],
        enable_thinking=True,
    )

    assert prompts == ["chat:First", "chat:Second"]
    assert calls == [
        (
            [{"role": "user", "content": "First"}],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        ),
        (
            [{"role": "user", "content": "Second"}],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        ),
    ]


def test_prompt_batches_keep_math500_at_concurrency_one() -> None:
    assert dflash_validation.prompt_batches(
        ["First", "Second"],
        sequential=True,
    ) == [["First"], ["Second"]]
    assert dflash_validation.prompt_batches(
        ["First", "Second"],
        sequential=False,
    ) == [["First", "Second"]]


def test_math500_workload_uses_published_generation_settings(tmp_path) -> None:
    payload = "".join(
        json.dumps({"problem": problem}) + "\n"
        for problem in ("Warmup", "Measured one", "Measured two")
    )
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(payload, encoding="utf-8")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"chat:{messages[0]['content']}:{kwargs['enable_thinking']}"

    workload = dflash_validation.build_math500_workload(
        Tokenizer(),
        dataset_path=dataset_path,
        expected_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        num_measured=2,
    )

    assert workload.warmup_prompts == [
        "chat:Warmup\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.:True",
    ]
    assert workload.measured_prompts == [
        "chat:Measured one\nPlease reason step by step, and put your final "
        "answer within \\boxed{}.:True",
        "chat:Measured two\nPlease reason step by step, and put your final "
        "answer within \\boxed{}.:True",
    ]
    assert workload.max_output_tokens == 2048
    assert workload.max_model_len == 4096
    assert workload.max_num_seqs == 1
    assert workload.sequential is True


def test_mixed_short_workload_retains_original_batch_contract() -> None:
    workload = dflash_validation.build_mixed_short_workload()

    assert workload.warmup_prompts == dflash_validation.WARMUP_PROMPTS
    assert workload.measured_prompts == dflash_validation.MEASURED_PROMPTS
    assert workload.max_output_tokens == 128
    assert workload.max_model_len == 2048
    assert workload.max_num_seqs == 8
    assert workload.sequential is False


def test_generate_executes_sequential_workload_one_request_at_a_time() -> None:
    calls = []

    class LLM:
        def generate(self, prompts, sampling_params):
            calls.append(
                (
                    prompts,
                    sampling_params.max_tokens,
                    sampling_params.temperature,
                    sampling_params.top_p,
                    sampling_params.top_k,
                    sampling_params.ignore_eos,
                )
            )
            return [f"output:{prompt}" for prompt in prompts]

    outputs = dflash_validation._generate(
        LLM(),
        ["First", "Second"],
        temperature=0,
        max_output_tokens=2048,
        sequential=True,
    )

    assert outputs == ["output:First", "output:Second"]
    # vLLM normalizes every temperature-zero request to its internal greedy
    # representation after validating the caller's requested top-k value.
    assert calls == [
        (["First"], 2048, 0, 1, 0, False),
        (["Second"], 2048, 0, 1, 0, False),
    ]


def test_generate_requests_published_sampling_settings(monkeypatch) -> None:
    import vllm

    requested = {}

    class SamplingParams:
        def __init__(self, **kwargs):
            requested.update(kwargs)

    class LLM:
        def generate(self, prompts, sampling_params):
            return [f"output:{prompt}" for prompt in prompts]

    monkeypatch.setattr(vllm, "SamplingParams", SamplingParams)

    dflash_validation._generate(
        LLM(),
        ["First"],
        temperature=0,
        max_output_tokens=2048,
        sequential=True,
    )

    assert requested == {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "ignore_eos": False,
        "max_tokens": 2048,
    }


def test_generate_sequential_with_counters_records_each_request() -> None:
    counter_values = {
        "vllm:spec_decode_num_drafts": 0,
        "vllm:spec_decode_num_draft_tokens": 0,
        "vllm:spec_decode_num_accepted_tokens": 0,
    }
    increments = {
        "First": (2, 30, 8),
        "Second": (4, 60, 28),
    }
    sampling_params_seen = []

    class LLM:
        def get_metrics(self):
            return [
                _counter(name, value) for name, value in counter_values.items()
            ]

        def generate(self, prompts, sampling_params):
            sampling_params_seen.append(sampling_params)
            drafts, draft_tokens, accepted = increments[prompts[0]]
            counter_values["vllm:spec_decode_num_drafts"] += drafts
            counter_values["vllm:spec_decode_num_draft_tokens"] += draft_tokens
            counter_values["vllm:spec_decode_num_accepted_tokens"] += accepted
            return [f"output:{prompts[0]}"]

    outputs, request_counters = (
        dflash_validation._generate_sequential_with_counters(
            LLM(),
            ["First", "Second"],
            temperature=0,
            max_output_tokens=2048,
        )
    )

    assert outputs == ["output:First", "output:Second"]
    assert request_counters == [
        WorkloadMetrics(2, 30, 8),
        WorkloadMetrics(4, 60, 28),
    ]
    assert sampling_params_seen[0] is sampling_params_seen[1]


def test_collect_speculative_counters_uses_metric_deltas() -> None:
    before = [
        _counter("vllm:spec_decode_num_drafts", 4),
        _counter("vllm:spec_decode_num_draft_tokens", 60),
        _counter("vllm:spec_decode_num_accepted_tokens", 22),
    ]
    after = [
        _counter("vllm:spec_decode_num_drafts", 10),
        _counter("vllm:spec_decode_num_draft_tokens", 150),
        _counter("vllm:spec_decode_num_accepted_tokens", 58),
    ]

    counters = collect_speculative_counters(before, after)

    assert counters == WorkloadMetrics(
        draft_rounds=6,
        drafted_tokens=90,
        accepted_tokens=36,
    )


def test_summarize_run_defines_average_accepted_length() -> None:
    counters = WorkloadMetrics(
        draft_rounds=6,
        drafted_tokens=90,
        accepted_tokens=36,
    )

    summary = summarize_run(
        counters,
        output_token_ids=[[1, 2, 3], [4, 5]],
        elapsed_seconds=2.5,
    )

    assert summary["average_accepted_length"] == pytest.approx(7.0)
    assert summary["average_accepted_length_formula"] == (
        "1 + accepted_tokens / draft_rounds"
    )
    assert summary["completion_tokens_per_draft_round"] == pytest.approx(5 / 6)
    assert summary["mean_request_completion_tokens_per_draft_round"] is None
    assert summary["measured_output_tokens"] == 5
    assert summary["elapsed_steady_state_seconds"] == 2.5
    assert summary["output_tokens_per_second"] == 2.0


def test_summarize_run_keeps_three_acceptance_aggregations_distinct() -> None:
    request_counters = [
        WorkloadMetrics(2, 30, 8),
        WorkloadMetrics(4, 60, 28),
    ]
    output_token_ids = [list(range(12)), list(range(8))]

    summary = summarize_run(
        WorkloadMetrics(6, 90, 36),
        output_token_ids,
        elapsed_seconds=2.0,
        request_counters=request_counters,
    )

    assert summary["average_accepted_length"] == pytest.approx(7.0)
    assert summary["completion_tokens_per_draft_round"] == pytest.approx(20 / 6)
    assert summary["mean_request_completion_tokens_per_draft_round"] == pytest.approx(
        (12 / 2 + 8 / 4) / 2
    )
    assert summary["request_metrics"] == [
        {
            "draft_rounds": 2,
            "drafted_tokens": 30,
            "accepted_tokens": 8,
            "output_tokens": 12,
            "average_accepted_length": 5.0,
            "completion_tokens_per_draft_round": 6.0,
        },
        {
            "draft_rounds": 4,
            "drafted_tokens": 60,
            "accepted_tokens": 28,
            "output_tokens": 8,
            "average_accepted_length": 8.0,
            "completion_tokens_per_draft_round": 2.0,
        },
    ]


def test_summarize_target_only_marks_acceptance_not_applicable() -> None:
    counters = WorkloadMetrics(0, 0, 0)

    summary = summarize_run(counters, [[1, 2]], elapsed_seconds=1.0)

    assert summary["average_accepted_length"] is None


def test_counter_delta_rejects_reset_or_mismatched_metrics() -> None:
    before = [_counter("vllm:spec_decode_num_drafts", 5)]
    after = [_counter("vllm:spec_decode_num_drafts", 4)]

    with pytest.raises(ValueError, match="decreased"):
        collect_speculative_counters(before, after)

    incomplete = replace(WorkloadMetrics(1, 2, 1), drafted_tokens=-1)
    with pytest.raises(ValueError, match="non-negative"):
        summarize_run(incomplete, [[1]], elapsed_seconds=1.0)


def test_write_result_persists_complete_json(tmp_path) -> None:
    output_path = tmp_path / "result.json"
    result = {"metrics": {"draft_rounds": 6}, "outputs": [{"token_ids": [1, 2]}]}

    write_result(result, output_path)

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert '"draft_rounds": 6' in output_path.read_text(encoding="utf-8")

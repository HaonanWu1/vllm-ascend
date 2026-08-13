# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest

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
    assert MODEL_PROFILES["qwen3.6-35b-a3b-w8a8"].target == (
        "/home/models/Qwen3.6-35B-A3B-w8a8"
    )
    assert MODEL_PROFILES["qwen3.6-35b-a3b-w8a8"].draft == (
        "/home/models/Qwen3.6-35B-A3B-DFlash"
    )
    assert RUN_PROFILES["eager"].enforce_eager is True
    assert RUN_PROFILES["target-only"].uses_dflash is False
    assert all(profile.temperature == 0 for profile in RUN_PROFILES.values())


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
    assert summary["measured_output_tokens"] == 5
    assert summary["elapsed_steady_state_seconds"] == 2.5
    assert summary["output_tokens_per_second"] == 2.0


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

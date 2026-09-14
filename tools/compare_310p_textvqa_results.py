# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare aligned Ascend and SGLang first-turn caption evaluation records."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from tools.evaluate_310p_textvqa_dflash import summarize


def audit_metrics(rows, directory, summary):
    """Check counter attribution independently against the run endpoints."""
    before = json.loads((directory / "metrics-before.json").read_text())
    after = json.loads((directory / "metrics-after.json").read_text())
    assert len({row["index"] for row in rows}) == len(rows)
    for key, value in summary["metrics"].items():
        assert math.isclose(value, after.get(key, 0) - before.get(key, 0), abs_tol=1e-8), key
    for row in rows:
        metrics = row["metrics_delta"]
        assert all(value >= 0 for value in metrics.values()), row["index"]
        assert metrics["vllm:request_success_total"] == 1, row["index"]
        drafts = metrics["vllm:spec_decode_num_drafts_total"]
        positions = [metrics[f"vllm:spec_decode_num_accepted_tokens_per_pos_total:{p}"] for p in range(7)]
        assert metrics["vllm:spec_decode_num_draft_tokens_total"] == 7 * drafts, row["index"]
        assert sum(positions) == metrics["vllm:spec_decode_num_accepted_tokens_total"], row["index"]
        assert all(a >= b for a, b in zip([drafts] + positions, positions)), row["index"]
    assert summary["metrics"]["vllm:request_success_total"] == len(rows)
    return {
        "passed": True,
        "unique_request_indices": len(rows),
        "endpoint_deltas_match_per_request_sums": True,
        "seven_candidates_per_draft": True,
        "accepted_position_sum_matches_total": True,
        "accepted_prefix_counts_monotonic": True,
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "warmup_successes_before_formal_run": before["vllm:request_success_total"],
    }


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-requests", type=Path, required=True)
    parser.add_argument("--gpu-requests", type=Path, required=True)
    parser.add_argument("--gpu-summary", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, default=1000)
    args = parser.parse_args()
    ascend = [json.loads(line) for line in args.ascend_requests.read_text().splitlines() if line.strip()]
    gpu = [json.loads(line) for line in args.gpu_requests.read_text().splitlines() if line.strip()]
    gpu_summary = json.loads(args.gpu_summary.read_text())
    assert len(ascend) == len(gpu) == args.expected_requests
    assert all(a["index"] == b["index"] and a["id"] == b["id"] for a, b in zip(ascend, gpu))
    assert all(a["status"] == b["status"] == "success" for a, b in zip(ascend, gpu))
    assert all(a["prompt"] == b["prompt"] for a, b in zip(ascend, gpu))
    summary = summarize(ascend)
    per_request = []
    for a, b in zip(ascend, gpu):
        metrics = a["metrics_delta"]
        proposed = metrics.get("vllm:spec_decode_num_draft_tokens_total", 0)
        accepted = metrics.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        per_request.append(
            {
                "index": a["index"],
                "id": a["id"],
                "prompt_tokens_equal": a["usage"]["prompt_tokens"] == b["usage"]["prompt_tokens"],
                "answer_exact_match": a["answer"] == b["answer"],
                "answer_whitespace_normalized_match": " ".join(a["answer"].split()) == " ".join(b["answer"].split()),
                "ascend_completion_tokens": a["usage"]["completion_tokens"],
                "gpu_completion_tokens": b["usage"]["completion_tokens"],
                "ascend_acceptance_rate": accepted / proposed if proposed else 0,
                "ascend_client_latency_seconds": a["latency"],
                "gpu_client_latency_seconds": b["latency"],
            }
        )
    rates = [row["ascend_acceptance_rate"] for row in per_request]
    latencies = [row["latency"] for row in ascend]
    gpu_latencies = [row["latency"] for row in gpu]
    report = {
        "scope": "First-turn image caption plus OCR prompts; exact answer match is not TextVQA accuracy.",
        "requests": len(ascend),
        "ascend": summary,
        "metrics_audit": audit_metrics(ascend, args.ascend_requests.parent, summary),
        "gpu": gpu_summary,
        "ascend_mean_request_accept_rate": statistics.fmean(rates),
        "ascend_median_request_accept_rate": statistics.median(rates),
        "ascend_p10_request_accept_rate": percentile(rates, 0.1),
        "ascend_p90_request_accept_rate": percentile(rates, 0.9),
        "ascend_p90_client_latency_seconds": percentile(latencies, 0.9),
        "gpu_mean_client_latency_seconds": statistics.fmean(gpu_latencies),
        "gpu_median_client_latency_seconds": statistics.median(gpu_latencies),
        "gpu_p90_client_latency_seconds": percentile(gpu_latencies, 0.9),
        "acceptance_difference_percentage_points": 100
        * (summary["global_spec_accept_rate"] - gpu_summary["global_spec_accept_rate"]),
        "prompt_token_count_matches": sum(row["prompt_tokens_equal"] for row in per_request),
        "answer_exact_matches": sum(row["answer_exact_match"] for row in per_request),
        "answer_whitespace_normalized_matches": sum(row["answer_whitespace_normalized_match"] for row in per_request),
        "answer_mismatch_examples": [
            {"index": a["index"], "id": a["id"], "ascend_answer": a["answer"], "gpu_answer": b["answer"]}
            for a, b in zip(ascend, gpu)
            if a["answer"] != b["answer"]
        ][:20],
        "per_request": per_request,
        "comparison_limits": [
            "Ascend FP16 versus GPU BF16; different hardware and engine execution modes.",
            "GPU latency summary uses server e2e time; client times are calculated separately from request records.",
            "This is an acceptance/caption-output comparison, not an official TextVQA accuracy evaluation.",
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("per_request", "answer_mismatch_examples")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

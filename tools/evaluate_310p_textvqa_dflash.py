# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sequential first-turn image requests with per-request Prometheus deltas.

Uses the same non-streaming OpenAI payload as SpecForge evaluate_vlm_dflash.py.
Only the first user turn is sent; dataset assistant answers are never supplied.
Use a dedicated server with no other clients so counter deltas are attributable.
"""

import argparse
import base64
import json
import mimetypes
import statistics
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def read_metrics(address):
    with urllib.request.urlopen(f"http://{address}/metrics", timeout=30) as response:
        text = response.read().decode()
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.endswith("_created"):
                continue
            if sample.name.startswith("vllm:spec_decode"):
                suffix = ":" + sample.labels["position"] if "position" in sample.labels else ""
                key = sample.name + suffix
            elif sample.name in (
                "vllm:request_success_total",
                "vllm:e2e_request_latency_seconds_sum",
                "vllm:e2e_request_latency_seconds_count",
                "vllm:num_preemptions_total",
            ):
                key = sample.name
            else:
                continue
            values[key] = values.get(key, 0) + sample.value
    return values


def metric_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def summarize(rows):
    successful = [row for row in rows if row["status"] == "success"]
    counters = {}
    for row in successful:
        for key, value in row["metrics_delta"].items():
            counters[key] = counters.get(key, 0) + value
    proposed = counters.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    drafts = counters.get("vllm:spec_decode_num_drafts_total", 0)
    completion = sum(row["usage"]["completion_tokens"] for row in successful)
    latency = [row["latency"] for row in successful]
    return {
        "requests": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "completion_tokens": completion,
        "accepted_drafts": accepted,
        "proposed_drafts": proposed,
        "verify_calls": drafts,
        "global_spec_accept_rate": accepted / proposed if proposed else None,
        "draft_based_accept_length": 1 + accepted / drafts if drafts else None,
        "token_based_accept_length": completion / drafts if drafts else None,
        "mean_client_latency_seconds": statistics.fmean(latency) if latency else None,
        "median_client_latency_seconds": statistics.median(latency) if latency else None,
        "mean_engine_latency_seconds": (
            counters.get("vllm:e2e_request_latency_seconds_sum", 0) / counters["vllm:e2e_request_latency_seconds_count"]
            if counters.get("vllm:e2e_request_latency_seconds_count")
            else None
        ),
        "aggregate_client_output_tokens_per_second": completion / sum(latency) if sum(latency) else None,
        "metrics": counters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-address", default="127.0.0.1:31042")
    parser.add_argument("--model", default="/home/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input_path.read_text().splitlines() if line.strip()]
    rows = rows[args.start_index : args.start_index + args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (args.output_dir / "requests.jsonl").open("x", encoding="utf-8")
    (args.output_dir / "test-config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    before = read_metrics(args.server_address)
    (args.output_dir / "metrics-before.json").write_text(json.dumps(before, indent=2))
    results = []
    with output:
        for index, row in enumerate(rows, args.start_index):
            record = {"index": index, "id": row.get("id"), "conversation_mode": "first-turn"}
            try:
                user = next(
                    message
                    for message in row["conversations"]
                    if message.get("role", message.get("from")) in ("user", "human")
                )
                prompt = user.get("content", user.get("value"))
                image = Path(row["image"])
                if not image.is_absolute():
                    image = args.input_path.parent / image
                mime = mimetypes.guess_type(str(image))[0] or "image/jpeg"
                url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")
                payload = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                }
                request = urllib.request.Request(
                    f"http://{args.server_address}/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    result = json.load(response)
                elapsed = time.perf_counter() - start
                after = read_metrics(args.server_address)
                deadline = time.monotonic() + 5
                while after.get("vllm:request_success_total", 0) < before.get("vllm:request_success_total", 0) + 1:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Finished-request metrics did not arrive within five seconds")
                    time.sleep(0.05)
                    after = read_metrics(args.server_address)
                delta = metric_delta(before, after)
                if delta.get("vllm:request_success_total") != 1:
                    raise RuntimeError("Unexpected concurrent request or metrics reset on the dedicated server")
                choice = result["choices"][0]
                record.update(
                    status="success",
                    image=str(image),
                    prompt=prompt,
                    request_message_count=1,
                    answer=choice["message"]["content"],
                    finish_reason=choice.get("finish_reason"),
                    latency=elapsed,
                    usage=result["usage"],
                    request_id=result["id"],
                    metrics_delta=delta,
                )
                before = after
            except Exception as exc:
                record.update(status="error", error=repr(exc))
            results.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            summary = summarize(results)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(
                f"[{len(results)}/{len(rows)}] {record['status']} "
                f"latency={record.get('latency')} accept_rate={summary['global_spec_accept_rate']}",
                flush=True,
            )
            # Do not let a failed/timed-out request contaminate subsequent deltas.
            if record["status"] != "success":
                raise RuntimeError(record["error"])
    (args.output_dir / "metrics-after.json").write_text(json.dumps(before, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Exercise exact prefix boundaries and concurrent reuse through the API."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


def _post(endpoint: str, payload: dict) -> dict:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:6666/v1/completions")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B-w8a8")
    parser.add_argument("--model-path", default="/home/models/Qwen3.6-35B-A3B-w8a8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pattern = tokenizer.encode(
        " We solve each arithmetic problem carefully and verify every step.",
        add_special_tokens=False,
    )
    warm_tail = tokenizer.encode(" warmup", add_special_tokens=False)
    hit_tail = tokenizer.encode(" alternate", add_special_tokens=False)

    def fill(length: int, source: list[int] = pattern) -> list[int]:
        return (source * (length // len(source) + 1))[:length]

    def complete(
        prompt: list[int],
        *,
        max_tokens: int = 8,
        ignore_eos: bool = False,
        allowed_token_ids: list[int] | None = None,
    ) -> dict:
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": ignore_eos,
        }
        if allowed_token_ids is not None:
            payload["allowed_token_ids"] = allowed_token_ids
        response = _post(args.endpoint, payload)
        if response.get("error") is not None:
            raise RuntimeError(f"completion failed: {response['error']}")
        choices = response.get("choices")
        usage = response.get("usage") or {}
        if not choices or not isinstance(choices[0].get("text"), str):
            raise RuntimeError(f"completion has no text choice: {response}")
        if usage.get("prompt_tokens") != len(prompt):
            raise RuntimeError(f"completion prompt length mismatch: expected={len(prompt)}, usage={usage}")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int) or not (1 <= completion_tokens <= max_tokens):
            raise RuntimeError(
                f"completion token count is outside the requested range: max_tokens={max_tokens}, usage={usage}"
            )
        return response

    results: dict = {"boundaries": {}}
    for length in (639, 640, 1279, 1280, 1281, 1460, 2559, 2560, 2561):
        response = complete(fill(length), max_tokens=1)
        results["boundaries"][str(length)] = {
            "usage": response.get("usage"),
            "text": response["choices"][0]["text"],
        }

    common = fill(1415)
    warmup = common + fill(45, warm_tail)
    hit = common + fill(83, hit_tail)
    warmup_response = complete(warmup, max_tokens=1)
    hit_response = complete(hit)
    results["shared_prefix"] = {
        "common_tokens": len(common),
        "warmup_tokens": len(warmup),
        "hit_tokens": len(hit),
        "warmup": warmup_response,
        "hit": hit_response,
    }

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    eos_prompt = fill(640)
    stop_on_eos = complete(
        eos_prompt,
        max_tokens=4,
        ignore_eos=False,
        allowed_token_ids=[eos_token_id],
    )
    ignore_eos = complete(
        eos_prompt,
        max_tokens=4,
        ignore_eos=True,
        allowed_token_ids=[eos_token_id],
    )
    if stop_on_eos["usage"]["completion_tokens"] != 1 or stop_on_eos["choices"][0]["finish_reason"] != "stop":
        raise RuntimeError(f"forced EOS did not stop immediately: {stop_on_eos}")
    if ignore_eos["usage"]["completion_tokens"] != 4 or ignore_eos["choices"][0]["finish_reason"] != "length":
        raise RuntimeError(f"ignore_eos did not reach max_tokens: {ignore_eos}")
    results["ignore_eos_modes"] = {
        "eos_token_id": eos_token_id,
        "stop_on_eos": stop_on_eos,
        "ignore_eos": ignore_eos,
    }

    reuse_prompt = common + fill(
        64,
        tokenizer.encode(" deterministic reuse", add_special_tokens=False),
    )
    reuse_baseline = complete(reuse_prompt)
    reuse_churn = [
        complete(
            common
            + fill(
                64 + index,
                tokenizer.encode(
                    f" sequential reuse {index}",
                    add_special_tokens=False,
                ),
            )
        )
        for index in range(3)
    ]
    reuse_after_finish = complete(reuse_prompt)
    baseline_choice = reuse_baseline["choices"][0]
    reused_choice = reuse_after_finish["choices"][0]
    if (
        reused_choice["text"] != baseline_choice["text"]
        or reused_choice["finish_reason"] != baseline_choice["finish_reason"]
        or reuse_after_finish["usage"]["completion_tokens"] != reuse_baseline["usage"]["completion_tokens"]
    ):
        raise RuntimeError(
            f"finished-request reuse changed deterministic output: before={reuse_baseline}, after={reuse_after_finish}"
        )
    results["reuse_after_finish"] = {
        "baseline": reuse_baseline,
        "churn": reuse_churn,
        "after": reuse_after_finish,
    }

    concurrent_prompts = [
        fill(1280) + fill(120 + index, tokenizer.encode(f" branch {index}", add_special_tokens=False))
        for index in range(10)
    ]
    with ThreadPoolExecutor(max_workers=10) as executor:
        concurrent = list(executor.map(complete, concurrent_prompts))
    results["concurrent_10"] = concurrent

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "boundary_requests": len(results["boundaries"]),
                "shared_prefix_hit_text": hit_response["choices"][0]["text"],
                "forced_eos_stop_tokens": stop_on_eos["usage"]["completion_tokens"],
                "forced_eos_ignore_tokens": ignore_eos["usage"]["completion_tokens"],
                "reuse_after_finish_match": True,
                "concurrent_requests": len(concurrent),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

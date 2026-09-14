# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run a reproducible DFlash correctness/performance case in a fresh process."""

import argparse
import json
import time
from pathlib import Path


def main():
    """Build one isolated model case, warm it up, and record repeatable outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--mode", choices=("eager", "piecewise", "full", "full-and-piecewise"), default="eager")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--images-per-prompt", type=int, default=1)
    parser.add_argument("--mixed", action="store_true", help="Alternate image and text requests")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--respect-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--capture-path", type=Path, help="Offline first-block numerical snapshot (eager only)")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tp,
        max_model_len=2048,
        max_num_seqs=max(args.batch, 8),
        max_num_batched_tokens=args.batched_tokens,
        block_size=128,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.mode == "eager",
        enable_prefix_caching=args.prefix_cache,
        enable_chunked_prefill=args.chunked_prefill,
        async_scheduling=args.async_scheduling,
        disable_log_stats=False,
        seed=42,
    )
    if args.kv_cache_memory_bytes is not None:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    model_config_path = Path(args.model) / "config.json"
    if model_config_path.is_file() and "vision_config" in json.loads(model_config_path.read_text()):
        kwargs["limit_mm_per_prompt"] = {"image": args.images_per_prompt if args.image else 0, "video": 0}
        if args.image:
            kwargs["mm_processor_kwargs"] = {"max_pixels": 50176}
    if args.mode != "eager":
        kwargs["compilation_config"] = {
            "cudagraph_mode": {
                "piecewise": "PIECEWISE",
                "full": "FULL_DECODE_ONLY",
                "full-and-piecewise": "FULL_AND_PIECEWISE",
            }[args.mode],
            "cudagraph_capture_sizes": [args.num_speculative_tokens + 1, 8 * (args.num_speculative_tokens + 1)],
        }
    if args.mode == "full-and-piecewise":
        # The 310P FAP route requires the existing explicit capture portfolio;
        # selecting the upstream enum alone does not activate that route.
        if not args.draft:
            raise ValueError("The 310P DFlash FULL_AND_PIECEWISE case requires --draft")
        kwargs["additional_config"] = {
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": args.batched_tokens,
                    "full_capture_size": [
                        args.num_speculative_tokens + 1,
                        8 * (args.num_speculative_tokens + 1),
                    ],
                }
            }
        }
    if args.draft:
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.capture_path:
        if args.mode != "eager" or not args.draft or args.batch != 1 or args.tp != 1:
            raise ValueError("Numerical capture requires an eager, TP=1, single-request draft run")
        kwargs["worker_extension_cls"] = "tools.validate_310p_dflash_layers.DFlashValidationWorkerExtension"
    llm = LLM(**kwargs)
    if args.capture_path:
        llm.collective_rpc("install_dflash_capture")
    tokenizer = llm.get_tokenizer()
    questions = [
        "Explain why the sky is blue in three short sentences.",
        "Compute 17 times 23 and explain the calculation.",
        "Write a short story about a sailor who discovers a new island.",
        "List five ways to save water at home and explain each one briefly.",
    ]
    prompts = []
    for i in range(args.batch):
        question = questions[i % len(questions)]
        use_image = args.image and (not args.mixed or i % 2 == 0)
        if use_image:
            from PIL import Image, ImageDraw

            picture = Image.new("RGB", (224, 224), "white")
            ImageDraw.Draw(picture).rectangle((30, 30, 190, 190), fill="red")
            question = "What color is the large square in the image? Describe its shape."
            text = "<|vision_start|><|image_pad|><|vision_end|>" * args.images_per_prompt + question
        else:
            text = question
        text += " Please consider the question carefully." * (args.prompt_repeat - 1)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(
            {"prompt": prompt, "multi_modal_data": {"image": [picture] * args.images_per_prompt}}
            if use_image
            else prompt
        )
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=not args.respect_eos)
    llm.generate(prompts, params, use_tqdm=False)
    if args.capture_path:
        llm.collective_rpc("save_dflash_capture", args=(str(args.capture_path),))
    runs = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        runs.append(
            {
                "seconds": elapsed,
                "tokens_per_second": sum(len(o.outputs[0].token_ids) for o in outputs) / elapsed,
                "token_ids": [list(o.outputs[0].token_ids) for o in outputs],
                "texts": [o.outputs[0].text for o in outputs],
                "finish_reasons": [o.outputs[0].finish_reason for o in outputs],
            }
        )
    metrics = []
    for metric in llm.get_metrics():
        if any(name in metric.name for name in ("spec_decode", "prefix_cache", "num_preemptions")):
            metrics.append(
                {"name": metric.name, "value": getattr(metric, "value", getattr(metric, "values", str(metric)))}
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": vars(args), "runs": runs, "metrics": metrics}, default=str, indent=2))
    print(f"Validation result: {args.output}", flush=True)


if __name__ == "__main__":
    main()

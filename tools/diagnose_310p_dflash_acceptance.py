# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Offline CPU diagnosis of one saved synthetic-image DFlash validation case.

Rebuild target features with Hugging Face, then use the checkpoint's saved
training forward without the Ascend proposer, cache, rotary, or sampler.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def error(actual, expected):
    actual, expected = actual.double(), expected.double()
    return {
        "relative_rms": ((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
    }


def leading_matches(matches):
    return next((i for i, value in enumerate(matches) if not value), len(matches))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    snapshot = torch.load(args.capture, map_location="cpu", weights_only=True)
    generated = json.loads(args.target_result.read_text())["runs"][0]["token_ids"][0][:64]
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=50176)
    picture = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(picture).rectangle((30, 30, 190, 190), fill="red")
    question = "What color is the large square in the image? Describe its shape."
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>" + question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=prompt, images=picture, return_tensors="pt")
    context_length = inputs.input_ids.shape[-1]
    assert context_length == snapshot["target_hidden"].shape[0]
    inputs["input_ids"] = torch.cat((inputs.input_ids, torch.tensor([generated])), dim=-1)
    inputs["attention_mask"] = torch.ones_like(inputs.input_ids)
    print("Loading independent HF target on CPU", flush=True)
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    inputs["mm_token_type_ids"] = (inputs.input_ids == target.config.image_token_id).long()
    inputs["pixel_values"] = inputs.pixel_values.to(torch.float16)
    positions, _ = target.model.get_rope_index(
        inputs.input_ids,
        inputs.mm_token_type_ids,
        image_grid_thw=inputs.image_grid_thw,
        attention_mask=inputs.attention_mask,
    )
    print("Computing independent target hidden states", flush=True)
    outputs = target(**inputs, output_hidden_states=True, use_cache=False)
    layers = [3, 10, 18, 25, 32]
    features = torch.cat([outputs.hidden_states[i + 1] for i in layers], dim=-1)
    hf_target_ids = outputs.logits[0].argmax(-1)
    report = {
        "context_tokens": context_length,
        "positions_match": torch.equal(positions[:, 0, :context_length], snapshot["context_positions"]),
        "target_features": error(snapshot["target_hidden"], features[0, :context_length]),
        "target_features_by_layer": {
            str(layer): error(
                snapshot["target_hidden"][:, i * 2560 : (i + 1) * 2560],
                outputs.hidden_states[layer + 1][0, :context_length],
            )
            for i, layer in enumerate(layers)
        },
        "hf_target_argmax_agreement": (
            hf_target_ids[context_length - 1 : context_length - 1 + len(generated)] == torch.tensor(generated)
        ).tolist(),
    }
    spec = importlib.util.spec_from_file_location("checkpoint_dflash_reference", Path(args.draft) / "dflash.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.Qwen3Config.from_pretrained(args.draft)
    config._attn_implementation = "eager"
    draft = module.DFlashDraftModel(config).to(torch.float16).eval()
    draft.load_state_dict(load_file(str(Path(args.draft) / "model.safetensors")), strict=True)
    embed = target.get_input_embeddings()
    head = target.get_output_embeddings()
    first_noise = embed(torch.tensor([[generated[0]] + [151669] * 7]))
    report["anchor_and_mask_embedding"] = error(snapshot["noise_embedding"], first_noise[0])
    first_positions = positions[:, :, : context_length + 8]
    reference = draft(
        position_ids=first_positions,
        noise_embedding=snapshot["noise_embedding"][None],
        target_hidden=snapshot["target_hidden"][None],
    )[0]
    captured_ids = head(snapshot["output"]).argmax(-1)
    reference_ids = head(reference).argmax(-1)
    report["same_input_draft_top1_equal"] = (captured_ids == reference_ids).tolist()
    report["first_block"] = {
        "target_tokens": generated[1:8],
        "ascend_candidates": captured_ids[1:].tolist(),
        "cpu_same_input_candidates": reference_ids[1:].tolist(),
        "target_text": processor.tokenizer.decode(generated[1:8]),
        "ascend_text": processor.tokenizer.decode(captured_ids[1:]),
        "cpu_same_input_text": processor.tokenizer.decode(reference_ids[1:]),
    }
    print("Comparing independent HF-feature draft blocks", flush=True)
    blocks = []
    for offset in range(len(generated) - 7):
        anchor = context_length + offset
        noise = embed(torch.tensor([[generated[offset]] + [151669] * 7]))
        hidden = draft(
            position_ids=positions[:, :, : anchor + 8],
            noise_embedding=noise,
            target_hidden=features[:, :anchor],
        )[0]
        candidates = head(hidden[1:]).argmax(-1).tolist()
        expected = generated[offset + 1 : offset + 8]
        matches = [a == b for a, b in zip(candidates, expected)]
        blocks.append(
            {
                "offset": offset,
                "candidates": candidates,
                "expected": expected,
                "matches": matches,
                "accepted_prefix": leading_matches(matches),
            }
        )
    report["independent_blocks"] = blocks
    report["independent_summary"] = {
        "blocks": len(blocks),
        "per_position_unconditional_matches": [sum(b["matches"][i] for b in blocks) for i in range(7)],
        "per_position_prefix_matches": [sum(b["accepted_prefix"] > i for b in blocks) for i in range(7)],
        "mean_accepted_prefix": sum(b["accepted_prefix"] for b in blocks) / len(blocks),
        "accepted_token_fraction": sum(b["accepted_prefix"] for b in blocks) / (7 * len(blocks)),
    }
    report["first_block_precision_candidates"] = {"float16": reference_ids[1:].tolist()}
    for dtype in (torch.bfloat16, torch.float32):
        # Recreate buffers from configuration instead of upcasting a previously
        # rounded rotary frequency table from another precision control.
        draft = module.DFlashDraftModel(config).to(dtype).eval()
        draft.load_state_dict(load_file(str(Path(args.draft) / "model.safetensors")), strict=True)
        head.to(dtype)
        hidden = draft(
            position_ids=first_positions,
            noise_embedding=snapshot["noise_embedding"][None].to(dtype),
            target_hidden=snapshot["target_hidden"][None].to(dtype),
        )[0]
        report["first_block_precision_candidates"][str(dtype)] = head(hidden[1:]).argmax(-1).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        json.dumps({k: v for k, v in report.items() if k != "independent_blocks"}, indent=2, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

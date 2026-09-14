# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare real Qwen3-VL image positions and 310P rotary outputs with HF."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from vllm_ascend._310p.spec_decode.dflash_mrope import DFlashMRoPEState310, build_dflash_mrope_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=50176)
    # get_rope_index only needs configuration and helper methods, not weights.
    target = Qwen3VLModel.__new__(Qwen3VLModel)
    torch.nn.Module.__init__(target)
    target.config = config
    hf_rotary = Qwen3VLTextRotaryEmbedding(config.text_config)
    torch.manual_seed(42)
    results = []
    for size in [(224, 224), (448, 112)]:
        image = Image.new("RGB", size, "red")
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is this?"}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        ids = inputs.input_ids
        types = (ids == config.image_token_id).long()
        context_positions, delta = target.get_rope_index(ids, types, image_grid_thw=inputs.image_grid_thw)
        extended = torch.cat((ids, torch.full((1, 8), 151669, dtype=ids.dtype)), dim=-1)
        full_positions, _ = target.get_rope_index(
            extended, (extended == config.image_token_id).long(), image_grid_thw=inputs.image_grid_thw
        )
        context_length = ids.shape[-1]
        prefix = context_length // 2
        logical, query_positions = build_dflash_mrope_positions(
            context_positions[:, 0, prefix:],
            torch.tensor([0, context_length - prefix]),
            torch.tensor([context_length]),
            None,
            delta.flatten(),
            context_length - prefix,
            8,
        )
        torch.testing.assert_close(logical, torch.arange(prefix, context_length, dtype=torch.int32))
        torch.testing.assert_close(query_positions, full_positions[:, 0, -8:], check_dtype=False)
        all_positions = full_positions[:, 0]
        num_tokens = all_positions.shape[-1]
        inv_freq = hf_rotary.inv_freq.detach().float()
        freqs = torch.arange(int(all_positions.max()) + 1).float()[:, None] * inv_freq
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        state = DFlashMRoPEState310(
            SimpleNamespace(
                cos_sin_cache=cache.to(device=args.device, dtype=torch.float16),
                head_size=128,
                rotary_dim=128,
                is_neox_style=True,
                mrope_section=[24, 20, 20],
                mrope_interleaved=True,
            ),
            num_tokens + 8,
        )
        state.refresh(all_positions.to(args.device), all_positions.to(args.device))
        q = torch.randn(1, 32, num_tokens, 128, dtype=torch.float16)
        k = torch.randn(1, 8, num_tokens, 128, dtype=torch.float16)
        cos, sin = hf_rotary(q, full_positions)
        expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
        flat_q = q.transpose(1, 2).reshape(num_tokens, -1)
        flat_k = k.transpose(1, 2).reshape(num_tokens, -1)
        actual_q, actual_k = state.apply(flat_q.to(args.device), flat_k.to(args.device))
        actual_q, actual_k = actual_q.cpu(), actual_k.cpu()
        expected_q = expected_q.transpose(1, 2).reshape_as(actual_q)
        expected_k = expected_k.transpose(1, 2).reshape_as(actual_k)
        torch.testing.assert_close(actual_q, expected_q, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(actual_k, expected_k, atol=3e-3, rtol=3e-3)
        results.append(
            {
                "image_size": size,
                "context_tokens": context_length,
                "prefix_tokens": prefix,
                "mrope_delta": int(delta.item()),
                "position_match": True,
                "q_max_abs_error": float((actual_q - expected_q).abs().max()),
                "k_max_abs_error": float((actual_k - expected_k).abs().max()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": args.device, "results": results}, indent=2))
    print(args.output.read_text(), flush=True)


if __name__ == "__main__":
    main()

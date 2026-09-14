# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Offline first-block comparison against the checkpoint's saved training model.

Capture hooks are installed explicitly by the validation harness after engine
initialization. They are never imported by the production inference path.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


class DFlashValidationWorkerExtension:
    def install_dflash_capture(self):
        return install_capture(self)

    def save_dflash_capture(self, path):
        return save_capture(self, path)


def install_capture(worker):
    proposer = worker.model_runner.drafter
    model = proposer.model.model
    snapshot = {}
    handles = []

    def fc_hook(module, inputs, output):
        if "target_hidden" not in snapshot:
            snapshot["target_hidden"] = inputs[0].detach().clone()

    def input_hook(module, inputs, kwargs):
        if "noise_embedding" in snapshot:
            return
        hidden = kwargs.get("hidden_states", inputs[1] if len(inputs) > 1 else None)
        snapshot["noise_embedding"] = hidden.detach().clone()
        snapshot["context_positions"] = proposer._context_mrope_positions_buffer[
            :, : proposer._dflash_num_context
        ].clone()
        snapshot["query_positions"] = proposer._get_positions(hidden.shape[0]).clone()

    def layer_hook(index):
        def capture(module, inputs, output):
            key = f"layer.{index}"
            if key not in snapshot:
                snapshot[key] = (output[0] + output[1]).detach().clone()

        return capture

    def output_hook(module, inputs, output):
        if "output" not in snapshot:
            snapshot["output"] = output[0].detach().clone()

    handles.append(model.fc.register_forward_hook(fc_hook))
    # The model's compile decorator bypasses Module.__call__; decoder and norm
    # hooks still observe the eager execution used by this diagnostic.
    handles.append(model.layers[0].register_forward_pre_hook(input_hook, with_kwargs=True))
    handles.append(model.norm.register_forward_hook(output_hook))
    for index, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(layer_hook(index)))
    worker._dflash_validation_capture = (snapshot, handles)


def save_capture(worker, path):
    snapshot, handles = worker._dflash_validation_capture
    for handle in handles:
        handle.remove()
    required = {"target_hidden", "noise_embedding", "context_positions", "query_positions", "output"}
    if not required.issubset(snapshot):
        raise RuntimeError(f"Incomplete first-block snapshot: missing {required - snapshot.keys()}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in snapshot.items()}, path)
    del worker._dflash_validation_capture
    return sorted(snapshot)


def error_summary(actual, expected):
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    return {
        "max_abs_error": difference.abs().max().item(),
        "relative_rms_error": (
            difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
        ).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location("checkpoint_dflash_reference", Path(args.draft) / "dflash.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.Qwen3Config.from_pretrained(args.draft)
    config._attn_implementation = "eager"
    model = module.DFlashDraftModel(config).to(torch.float16).eval()
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(Path(args.draft) / "model.safetensors")), strict=True)
    snapshot = torch.load(args.capture, map_location="cpu", weights_only=True)
    reference = {}

    def hook(index):
        def capture(module, inputs, output):
            reference[f"layer.{index}"] = output.squeeze(0).detach()

        return capture

    for index, layer in enumerate(model.layers):
        layer.register_forward_hook(hook(index))
    positions = torch.cat((snapshot["context_positions"], snapshot["query_positions"]), dim=-1)[:, None]
    reference["output"] = model(
        position_ids=positions,
        noise_embedding=snapshot["noise_embedding"][None],
        target_hidden=snapshot["target_hidden"][None],
    ).squeeze(0)
    results = {name: error_summary(snapshot[name], expected) for name, expected in reference.items()}
    report = {
        "reference": "checkpoint saved dflash.py, CPU FP16, identical captured first-block inputs",
        "context_tokens": snapshot["context_positions"].shape[-1],
        "query_tokens": snapshot["query_positions"].shape[-1],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(args.output.read_text(), flush=True)
    # Relative RMS and cosine are robust to isolated large FP16 activations.
    assert all(
        result["relative_rms_error"] < 0.02 and result["cosine_similarity"] > 0.999 for result in results.values()
    ), report


if __name__ == "__main__":
    main()

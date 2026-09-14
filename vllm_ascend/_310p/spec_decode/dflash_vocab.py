# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load frozen target output rows for headless, reduced-vocabulary DFlash."""

import torch
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_original_load_weights = DFlashQwen3ForCausalLM.load_weights
_original_maybe_share_lm_head = AscendSpecDecodeBaseProposer._maybe_share_lm_head


def load_dflash_weights_310(self, weights):
    # Track the actual weight stream: a d2t mapping alone does not mean the
    # checkpoint contains its own trained output head.
    self._dflash_checkpoint_has_lm_head = False

    def tracked_weights():
        for name, weight in weights:
            if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
                self._dflash_checkpoint_has_lm_head = True
            yield name, weight

    return _original_load_weights(self, tracked_weights())


def maybe_share_dflash_lm_head_310(self, model):
    draft = self.model
    mapping = getattr(draft, "draft_id_to_target_id", None)
    if (
        self.method == "dflash"
        and mapping is not None
        and getattr(draft, "_dflash_checkpoint_has_lm_head", None) is False
    ):
        target = (
            model.get_language_model()
            if not hasattr(model, "lm_head") and hasattr(model, "get_language_model")
            else model
        )
        head = getattr(target, "lm_head", None)
        if head is None:
            raise ValueError("Headless reduced-vocabulary DFlash requires a target lm_head")
        weight = head.weight
        if weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError("Headless reduced-vocabulary DFlash requires unquantized target lm_head weights")
        if getattr(head, "num_added_embeddings", 0):
            raise ValueError("Headless reduced-vocabulary DFlash does not support added target vocabulary rows")
        with torch.no_grad():
            # Target and draft use different vocabulary shards. Gather target
            # rows first, then let the draft's weight loader shard/pad them.
            if get_tensor_model_parallel_world_size() > 1:
                weight = tensor_model_parallel_all_gather(weight, dim=0)
            indices = torch.arange(mapping.numel(), device=mapping.device) + mapping
            selected = weight.index_select(0, indices.to(weight.device))
            draft.lm_head.weight_loader(draft.lm_head.weight, selected)
            # The initial model load already built weight_nz from the missing
            # head's empty storage. Rebuild it before any compile/capture.
            draft.lm_head.quant_method.process_weights_after_loading(draft.lm_head)
    # Retain the existing own-head/full-vocabulary policy and graph wrapping.
    return _original_maybe_share_lm_head(self, model)

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import nullcontext
from typing import Any

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.ops.rotary_embedding import (
    AscendRotaryEmbedding310,
    reserve_draft_rope_capacity_310p,
)
from vllm_ascend._310p.spec_decode.dflash_diagnostics_310 import (
    capture_current_dflash_graph_dispatch,
    capture_dflash_diagnostic,
    dflash_diagnostic_enabled,
)
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft


def _describe_dflash_embedding_310(proposer: Any) -> dict[str, Any]:
    """Describe the real shared embedding selected by the 310P draft path."""
    draft_model = getattr(proposer, "model", None)
    language_model = getattr(draft_model, "model", None)
    embedding = getattr(language_model, "embed_tokens", None)
    configured_mode = getattr(
        getattr(
            getattr(proposer, "vllm_config", None),
            "compilation_config",
            None,
        ),
        "cudagraph_mode",
        None,
    )
    if embedding is None:
        return {
            "embedding_type": None,
            "embedding_forward_origin": None,
            "embedding_tp_size": None,
            "embedding_graph_output_eligible": None,
            "embedding_graph_output_eligibility_error": None,
            "embedding_has_persistent_output": False,
            "embedding_persistent_output_data_ptr": None,
            "embedding_persistent_output_alignment_512": None,
            "embedding_weight_data_ptr": None,
            "embedding_quant_method_type": None,
            "configured_graph_mode": getattr(
                configured_mode,
                "name",
                str(configured_mode),
            ),
        }

    embedding_type = type(embedding)
    forward_origin = getattr(embedding, "_forward_origin", None)
    forward_function = getattr(forward_origin, "__func__", forward_origin)
    forward_name = None
    if forward_function is not None:
        forward_name = (
            f"{getattr(forward_function, '__module__', '')}."
            f"{getattr(forward_function, '__qualname__', type(forward_function).__qualname__)}"
        ).lstrip(".")

    eligible = None
    eligibility_error = None
    eligibility_check = getattr(
        embedding,
        "_use_dflash_full_graph_output_310",
        None,
    )
    if callable(eligibility_check):
        try:
            eligible = bool(eligibility_check())
        except Exception as exc:  # noqa: BLE001
            eligibility_error = f"{type(exc).__name__}: {exc}"

    persistent_output = getattr(
        embedding,
        "_dflash_graph_embedding_output_310",
        None,
    )
    persistent_output_data_ptr = (
        persistent_output.data_ptr()
        if isinstance(persistent_output, torch.Tensor)
        else None
    )
    weight = getattr(embedding, "weight", None)
    weight_data_ptr = weight.data_ptr() if isinstance(weight, torch.Tensor) else None
    quant_method = getattr(embedding, "quant_method", None)
    quant_method_type = type(quant_method) if quant_method is not None else None

    return {
        "embedding_type": (
            f"{embedding_type.__module__}.{embedding_type.__qualname__}"
        ),
        "embedding_forward_origin": forward_name,
        "embedding_tp_size": getattr(embedding, "tp_size", None),
        "embedding_graph_output_eligible": eligible,
        "embedding_graph_output_eligibility_error": eligibility_error,
        "embedding_has_persistent_output": persistent_output_data_ptr is not None,
        "embedding_persistent_output_data_ptr": persistent_output_data_ptr,
        "embedding_persistent_output_alignment_512": (
            persistent_output_data_ptr % 512
            if persistent_output_data_ptr is not None
            else None
        ),
        "embedding_weight_data_ptr": weight_data_ptr,
        "embedding_quant_method_type": (
            f"{quant_method_type.__module__}.{quant_method_type.__qualname__}"
            if quant_method_type is not None
            else None
        ),
        "configured_graph_mode": getattr(
            configured_mode,
            "name",
            str(configured_mode),
        ),
    }


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer overrides for NPU-specific spec-decode workarounds."""

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        capture_diagnostics = self.method == "dflash" and dflash_diagnostic_enabled()
        if capture_diagnostics:
            capture_current_dflash_graph_dispatch(
                self.vllm_config,
                path="draft",
            )
        in_spec_dummy_capture = capture_diagnostics and bool(
            getattr(
                getattr(self, "runner", None),
                "_spec_dummy_capture",
                False,
            )
        )
        if capture_diagnostics and not in_spec_dummy_capture:
            embedding_diagnostic = _describe_dflash_embedding_310(self)

            def _runtime_inputs_payload() -> dict[str, Any]:
                input_ids = self.input_ids[:num_input_tokens]
                positions = self._get_positions(num_input_tokens)
                return {
                    "num_input_tokens": num_input_tokens,
                    "num_active_tokens": num_tokens,
                    "input_ids": input_ids,
                    "positions": positions,
                    "input_ids_data_ptr": input_ids.data_ptr(),
                    "positions_data_ptr": positions.data_ptr(),
                    **embedding_diagnostic,
                }

            capture_dflash_diagnostic(
                "draft_runtime_inputs",
                payload_builder=_runtime_inputs_payload,
            )
        rope_capacity = (
            reserve_draft_rope_capacity_310p(
                getattr(self, "max_num_tokens", 0)
            )
            if self.method == "dflash"
            else nullcontext()
        )
        try:
            with rope_capacity:
                result = _original_run_merged_draft(
                    self,
                    num_input_tokens,
                    batch_size,
                    token_indices_to_sample,
                    target_positions,
                    inputs_embeds,
                    multi_steps_attn_metadata,
                    num_tokens,
                    is_prefill,
                )
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
        if capture_diagnostics and not in_spec_dummy_capture:
            capture_dflash_diagnostic(
                "draft_output",
                payload_builder=lambda: {
                    "draft_token_ids": result,
                    "token_indices_to_sample": token_indices_to_sample,
                },
            )
        return result

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # 310P workaround for MTP:
            # The NPU implementation of the slice assign
            #   self.input_ids[:num_tokens-1] = target_token_ids[1:]
            # can corrupt the tail element (index num_tokens-1) of the
            # persistent drafter input_ids buffer. We save/restore it to
            # avoid feeding garbage to the draft model or later GatherV2.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]

            # Protected shift (310P specific)
            tail_save = self.input_ids[num_tokens - 1].clone()
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            self.input_ids[num_tokens - 1] = tail_save

            # Replace the last token with the next token.
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None

            # 310P does not support PCP/DCP, so we skip all PCP handling.
            ori_token_indices_to_sample = None
            query_lens_d = None

            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        return super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )

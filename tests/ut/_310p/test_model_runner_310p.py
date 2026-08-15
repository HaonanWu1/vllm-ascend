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

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.metadata_builder import (
    AscendAttentionMetadataBuilder310,
)
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    wrap_dummy_run_with_draft_flag,
)
from vllm_ascend._310p.spec_decode.dflash_diagnostics_310 import (
    _reset_dflash_diagnostics_for_test,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)


_DFLASH_GRAPH_RUNTIME_CONTRACT = {
    CUDAGraphMode.NONE: {
        "prefill": CUDAGraphMode.NONE,
        "mixed": CUDAGraphMode.NONE,
        "uniform_decode": CUDAGraphMode.NONE,
    },
    CUDAGraphMode.PIECEWISE: {
        "prefill": CUDAGraphMode.PIECEWISE,
        "mixed": CUDAGraphMode.PIECEWISE,
        "uniform_decode": CUDAGraphMode.PIECEWISE,
    },
    CUDAGraphMode.FULL_DECODE_ONLY: {
        "prefill": CUDAGraphMode.NONE,
        "mixed": CUDAGraphMode.NONE,
        "uniform_decode": CUDAGraphMode.FULL,
    },
    CUDAGraphMode.FULL_AND_PIECEWISE: {
        "prefill": CUDAGraphMode.PIECEWISE,
        "mixed": CUDAGraphMode.PIECEWISE,
        "uniform_decode": CUDAGraphMode.FULL,
    },
    CUDAGraphMode.FULL: {
        "prefill": CUDAGraphMode.FULL,
        "mixed": CUDAGraphMode.FULL,
        "uniform_decode": CUDAGraphMode.FULL,
    },
}


def _make_k15_dflash_dispatcher(
    requested_mode: CUDAGraphMode,
) -> CudagraphDispatcher:
    compilation_config = SimpleNamespace(
        cudagraph_mode=requested_mode,
        max_cudagraph_capture_size=128,
        cudagraph_capture_sizes=[16, 32, 64, 128],
        compile_sizes=[],
        cudagraph_specialize_lora=True,
        is_attention_compiled_piecewise=lambda: True,
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation_config,
        num_speculative_tokens=15,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        lora_config=None,
    )
    dispatcher = CudagraphDispatcher(vllm_config)
    dispatcher.initialize_cudagraph_keys(requested_mode, 16)
    return dispatcher


def _make_k15_spec_runner(
    requested_mode: CUDAGraphMode,
    method: str = "dflash",
) -> tuple[NPUModelRunner310, CudagraphDispatcher]:
    dispatcher = _make_k15_dflash_dispatcher(requested_mode)
    vllm_config = dispatcher.vllm_config
    vllm_config.speculative_config = SimpleNamespace(method=method)
    vllm_config.model_config = SimpleNamespace(is_encoder_decoder=False)
    vllm_config.parallel_config = SimpleNamespace(
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    vllm_config.observability_config = SimpleNamespace(cudagraph_metrics=False)
    vllm_config.additional_config = {}

    runner = object.__new__(NPUModelRunner310)
    runner.vllm_config = vllm_config
    runner.speculative_config = vllm_config.speculative_config
    runner.model_config = vllm_config.model_config
    runner.cudagraph_dispatcher = dispatcher
    runner._dflash_requested_cudagraph_mode_310 = requested_mode
    runner.uniform_decode_query_len = 16
    return runner, dispatcher


@pytest.mark.parametrize("execution_path", ["target", "draft"])
@pytest.mark.parametrize(
    ("batch_kind", "num_tokens", "uniform_decode"),
    [
        pytest.param("prefill", 7, False, id="prefill"),
        pytest.param("mixed", 23, False, id="mixed"),
        pytest.param("uniform_decode", 32, True, id="k15-uniform-decode"),
    ],
)
@pytest.mark.parametrize(
    "requested_mode",
    [
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL,
    ],
)
def test_k15_dflash_graph_runtime_contract(
    execution_path: str,
    batch_kind: str,
    num_tokens: int,
    uniform_decode: bool,
    requested_mode: CUDAGraphMode,
) -> None:
    """Target and draft dispatch independently to the required runtime mode."""
    dispatcher = _make_k15_dflash_dispatcher(requested_mode)

    runtime_mode, descriptor = dispatcher.dispatch(
        num_tokens=num_tokens,
        uniform_decode=uniform_decode,
    )

    expected_mode = _DFLASH_GRAPH_RUNTIME_CONTRACT[requested_mode][batch_kind]
    assert runtime_mode == expected_mode, (
        f"{execution_path} {batch_kind}: requested={requested_mode}, "
        f"runtime={runtime_mode}"
    )
    if expected_mode == CUDAGraphMode.NONE:
        assert descriptor.num_tokens == num_tokens
    else:
        assert descriptor.num_tokens >= num_tokens


@pytest.mark.parametrize(
    (
        "batch_kind",
        "attn_state",
        "num_scheduled_tokens",
        "num_computed_tokens",
    ),
    [
        pytest.param(
            "prefill",
            AscendAttentionState.ChunkedPrefill,
            [7],
            [0],
            id="prefill",
        ),
        pytest.param(
            "prefill_cache_hit",
            AscendAttentionState.PrefillCacheHit,
            [7],
            [0],
            id="prefill-cache-hit",
        ),
        pytest.param(
            "mixed",
            AscendAttentionState.ChunkedPrefill,
            [7, 16],
            [0, 32],
            id="mixed",
        ),
        pytest.param(
            "uniform_decode",
            AscendAttentionState.SpecDecoding,
            [16, 16],
            [32, 48],
            id="k15-uniform-decode",
        ),
    ],
)
def test_k15_dflash_piecewise_target_and_draft_dispatch_integration(
    batch_kind: str,
    attn_state: AscendAttentionState,
    num_scheduled_tokens: list[int],
    num_computed_tokens: list[int],
) -> None:
    """Exercise the target owner and the draft's real dispatcher contract."""
    runner, dispatcher = _make_k15_spec_runner(CUDAGraphMode.PIECEWISE)
    runner.attn_state = attn_state
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray(num_computed_tokens, dtype=np.int32),
        lora_id_to_lora_request={},
    )

    scheduled = np.asarray(num_scheduled_tokens, dtype=np.int32)
    with (
        patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=False),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        target_mode, target_descriptor, *_ = (
            runner._determine_batch_execution_and_padding(
                num_tokens=int(scheduled.sum()),
                num_reqs=len(scheduled),
                num_scheduled_tokens_np=scheduled,
                max_num_scheduled_tokens=int(scheduled.max()),
                use_cascade_attn=False,
            )
        )

    # AscendSpecDecodeBaseProposer._propose dispatches its 16-token-per-request
    # DFlash query through this same dispatcher, using the target descriptor's
    # uniform bit rather than reusing the target runtime mode.
    draft_mode, _ = dispatcher.dispatch(
        num_tokens=16 * len(scheduled),
        uniform_decode=target_descriptor.uniform,
        has_lora=False,
    )

    assert (target_mode, draft_mode) == (
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.PIECEWISE,
    ), (
        f"{batch_kind}: target={target_mode.name}, draft={draft_mode.name}"
    )


@pytest.mark.parametrize(
    ("requested_mode", "non_decode_mode"),
    [
        pytest.param(
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.NONE,
            id="full-decode-only",
        ),
        pytest.param(
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.PIECEWISE,
            id="full-and-piecewise",
        ),
    ],
)
def test_k15_dflash_full_decode_target_and_real_draft_path(
    requested_mode: CUDAGraphMode,
    non_decode_mode: CUDAGraphMode,
) -> None:
    """Compose the requested non-decode path with shared FULL decode inputs."""
    runner, dispatcher = _make_k15_spec_runner(requested_mode)
    runner.vllm_config.model_config.use_mla = False
    runner.dynamic_eplb = False
    runner.pcp_manager = None
    synced_draft_tokens = 0

    def sync_draft_tokens(num_tokens, is_draft_model):
        assert is_draft_model
        return synced_draft_tokens or num_tokens, synced_draft_tokens, False

    def pad_query_start_loc(
        query_start_loc,
        num_tokens_padded,
        num_reqs_padded,
        num_reqs,
        *args,
    ):
        tail = torch.arange(
            num_reqs + 1,
            num_reqs_padded + 1,
            dtype=torch.int32,
        )
        tail.mul_(16)
        query_start_loc.gpu[num_reqs + 1 : num_reqs_padded + 1].copy_(
            tail
        )
        query_start_loc.cpu[num_reqs + 1 : num_reqs_padded + 1].copy_(
            tail
        )
        return num_reqs_padded

    runner._sync_metadata_across_dp = sync_draft_tokens
    runner._pad_query_start_loc_for_fia = pad_query_start_loc

    class FakeDFlashModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(_attn_layers=[])

        @staticmethod
        def combine_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    draft_model = FakeDFlashModel()
    runner.get_model = lambda: draft_model

    builder = object.__new__(AscendAttentionMetadataBuilder310)
    builder.device = torch.device("cpu")
    builder._query_lens_cpu_buffer = torch.zeros(8, dtype=torch.int32)

    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "dflash"
    proposer.model = draft_model
    proposer.get_model = lambda: draft_model
    proposer.hidden_size = 4
    proposer.runner = runner
    proposer.vllm_config = runner.vllm_config
    proposer.use_cuda_graph = True
    proposer.use_compress = False
    proposer.pcp_size = 1
    proposer.dcp_size = 1
    proposer.decode_threshold = 16
    proposer.query_start_loc = SimpleNamespace(
        gpu=torch.zeros(9, dtype=torch.int32),
        cpu=torch.zeros(9, dtype=torch.int32),
    )
    proposer.draft_window_size = None
    proposer.supports_mm_inputs = False
    proposer.slot_mapping_group = [
        torch.full((128,), -1, dtype=torch.int32)
    ]
    proposer.seq_lens_group = [torch.zeros(8, dtype=torch.int32)]
    proposer.query_start_loc_group = [
        torch.zeros(9, dtype=torch.int32)
    ]
    proposer.draft_attn_groups = [
        SimpleNamespace(
            get_metadata_builder=lambda: builder,
            kv_cache_spec=SimpleNamespace(block_size=128),
        )
    ]
    proposer.attn_layer_names = ["model.layers.0.self_attn.attn"]
    proposer.uses_mrope = False
    proposer.positions = torch.arange(128, dtype=torch.int32)
    proposer.parallel_drafting = True
    proposer.num_speculative_tokens = 15
    proposer.token_indices_to_sample = torch.zeros(128, dtype=torch.int32)
    proposer.enable_enpu = False
    proposer.max_num_tokens = 128
    proposer.max_query_tokens = 128
    def adjust_tensor(tensor, length):
        if tensor.shape[0] >= length:
            return tensor[:length]
        padded = torch.zeros(
            (length, *tensor.shape[1:]),
            dtype=tensor.dtype,
        )
        padded[: tensor.shape[0]].copy_(tensor)
        return padded

    proposer._adjust_tensor = adjust_tensor
    proposer._update_full_graph_params_if_needed = lambda *args: None
    proposer._runnable = lambda **kwargs: torch.zeros(
        kwargs["batch_size"], dtype=torch.int32
    )

    phase = "capture"
    metadata_pointers: dict[str, dict[str, int]] = {}

    def build_base_metadata(_, common, *args, **kwargs):
        metadata_pointers[phase] = {
            "query_start_loc": common.query_start_loc.data_ptr(),
            "seq_lens": common.seq_lens.data_ptr(),
            "slot_mapping": common.slot_mapping.data_ptr(),
        }
        return SimpleNamespace(
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            num_prefills=0,
            attn_mask=None,
        )

    def make_common_metadata(num_reqs: int) -> SimpleNamespace:
        num_tokens = 16 * num_reqs
        query_start_loc = torch.arange(
            0,
            num_tokens + 1,
            16,
            dtype=torch.int32,
        )
        return SimpleNamespace(
            batch_size=lambda: num_reqs,
            query_start_loc=query_start_loc.clone(),
            query_start_loc_cpu=query_start_loc.clone(),
            seq_lens=torch.arange(
                20,
                20 + num_reqs,
                dtype=torch.int32,
            ),
            seq_lens_cpu=None,
            _seq_lens_cpu=None,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=16,
            block_table_tensor=torch.zeros(
                num_reqs,
                8,
                dtype=torch.int32,
            ),
            slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
            causal=False,
            num_computed_tokens_cpu=None,
        )

    def capture_draft(self, **kwargs):
        return builder.build(0, make_common_metadata(4))

    captured_context = None
    draft_modes: list[CUDAGraphMode] = []

    @contextmanager
    def record_forward_context(*args, **kwargs):
        nonlocal captured_context
        captured_context = SimpleNamespace(
            cudagraph_runtime_mode=kwargs["aclgraph_runtime_mode"],
            moe_layer_index=None,
        )
        draft_modes.append(captured_context.cudagraph_runtime_mode)
        try:
            yield captured_context
        finally:
            captured_context = None

    cases = [
        (
            AscendAttentionState.ChunkedPrefill,
            [7],
            [0],
            non_decode_mode,
            16,
        ),
        (
            AscendAttentionState.ChunkedPrefill,
            [7, 16],
            [0, 32],
            non_decode_mode,
            32,
        ),
        (
            AscendAttentionState.SpecDecoding,
            [16, 16],
            [32, 48],
            CUDAGraphMode.FULL,
            64,
        ),
    ]

    with (
        patch.object(
            AscendAttentionMetadataBuilder310.__bases__[0],
            "build",
            side_effect=build_base_metadata,
        ),
        patch(
            "vllm_ascend._310p.attention.metadata_builder."
            "is_compressed_mask_supported",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310."
            "_prepare_dflash_full_graph_capture_310",
            create=True,
            side_effect=lambda owner, **kwargs: setattr(
                builder,
                "_dflash_full_graph_owner_310",
                owner,
            ),
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer."
            "DFlashQwen3ForCausalLM",
            FakeDFlashModel,
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer."
            "set_ascend_forward_context",
            side_effect=record_forward_context,
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.get_forward_context",
            side_effect=lambda: captured_context,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp",
            return_value=False,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        wrap_dummy_run_with_draft_flag(capture_draft)(
            proposer,
            num_tokens=64,
            num_reqs=4,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
        )

        for case_index, (
            attn_state,
            scheduled_tokens,
            computed_tokens,
            expected_mode,
            expected_synced_tokens,
        ) in enumerate(cases):
            scheduled = np.asarray(scheduled_tokens, dtype=np.int32)
            runner.attn_state = attn_state
            runner.input_batch = SimpleNamespace(
                num_computed_tokens_cpu=np.asarray(
                    computed_tokens,
                    dtype=np.int32,
                ),
                lora_id_to_lora_request={},
            )
            target_mode, target_descriptor, *_ = (
                runner._determine_batch_execution_and_padding(
                    num_tokens=int(scheduled.sum()),
                    num_reqs=len(scheduled),
                    num_scheduled_tokens_np=scheduled,
                    max_num_scheduled_tokens=int(scheduled.max()),
                    use_cascade_attn=False,
                )
            )

            common = make_common_metadata(len(scheduled))
            sample_indices = torch.arange(
                15,
                16 * len(scheduled),
                16,
                dtype=torch.int32,
            )
            proposer.set_inputs_first_pass = lambda **kwargs: (
                16 * len(scheduled),
                sample_indices,
                common,
                None,
            )
            synced_draft_tokens = expected_synced_tokens
            phase = f"runtime-{case_index}"
            with patch.object(
                dispatcher,
                "dispatch",
                wraps=dispatcher.dispatch,
            ) as draft_dispatch:
                proposer._propose(
                    target_token_ids=torch.zeros(1, dtype=torch.int32),
                    target_positions=torch.zeros(1, dtype=torch.int32),
                    target_hidden_states=torch.zeros(1, 4),
                    next_token_ids=torch.zeros(
                        len(scheduled),
                        dtype=torch.int32,
                    ),
                    token_indices_to_sample=sample_indices,
                    common_attn_metadata=common,
                    target_model_batch_desc=target_descriptor,
                    sampling_metadata=SimpleNamespace(),
                )

            assert draft_modes[-1] == expected_mode
            assert [
                call.kwargs["uniform_decode"]
                for call in draft_dispatch.call_args_list
            ] == [target_descriptor.uniform, target_descriptor.uniform]
            assert [
                call.kwargs["num_tokens"]
                for call in draft_dispatch.call_args_list
            ] == [16 * len(scheduled), expected_synced_tokens]
            assert target_mode == expected_mode

    assert metadata_pointers["runtime-2"] == metadata_pointers["capture"]


@pytest.mark.parametrize(
    ("requested_mode", "method", "force_eager"),
    [
        pytest.param(
            CUDAGraphMode.PIECEWISE,
            "dspark",
            False,
            id="other-speculative-method",
        ),
        pytest.param(
            CUDAGraphMode.FULL_DECODE_ONLY,
            "dflash",
            False,
            id="other-graph-mode",
        ),
        pytest.param(
            CUDAGraphMode.PIECEWISE,
            "dflash",
            True,
            id="explicit-force-eager",
        ),
    ],
)
def test_k15_dflash_piecewise_relaxation_is_narrowly_scoped(
    requested_mode: CUDAGraphMode,
    method: str,
    force_eager: bool,
) -> None:
    runner, _ = _make_k15_spec_runner(requested_mode, method)
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
        lora_id_to_lora_request={},
    )
    scheduled = np.asarray([7], dtype=np.int32)

    with (
        patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=False),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runtime_mode, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=7,
            num_reqs=1,
            num_scheduled_tokens_np=scheduled,
            max_num_scheduled_tokens=7,
            use_cascade_attn=False,
            force_eager=force_eager,
        )

    assert runtime_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    "requested_mode",
    [
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL,
    ],
)
def test_k15_dflash_normalized_piecewise_does_not_expand_requested_mode(
    requested_mode: CUDAGraphMode,
) -> None:
    runner, _ = _make_k15_spec_runner(CUDAGraphMode.PIECEWISE)
    runner._dflash_requested_cudagraph_mode_310 = requested_mode
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
        lora_id_to_lora_request={},
    )
    scheduled = np.asarray([7], dtype=np.int32)

    with (
        patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=False),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runtime_mode, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=7,
            num_reqs=1,
            num_scheduled_tokens_np=scheduled,
            max_num_scheduled_tokens=7,
            use_cascade_attn=False,
        )

    assert runtime_mode == CUDAGraphMode.NONE


def test_k15_dflash_profile_before_mode_normalization_stays_eager() -> None:
    runner, _ = _make_k15_spec_runner(CUDAGraphMode.PIECEWISE)
    del runner._dflash_requested_cudagraph_mode_310
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
        lora_id_to_lora_request={},
    )
    scheduled = np.asarray([7], dtype=np.int32)

    with (
        patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=False),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runtime_mode, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=7,
            num_reqs=1,
            num_scheduled_tokens_np=scheduled,
            max_num_scheduled_tokens=7,
            use_cascade_attn=False,
        )

    assert runtime_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize("diagnostic_enabled", [False, True])
def test_target_dispatch_emits_path_labelled_dflash_graph_evidence(
    diagnostic_enabled: bool,
) -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash")
    )
    runner.speculative_config = runner.vllm_config.speculative_config
    runner.cudagraph_dispatcher = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE
    )
    runner._dflash_requested_cudagraph_mode_310 = CUDAGraphMode.NONE
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([3, 5], dtype=np.int32)
    )
    runner.attn_state = object()
    runner.uniform_decode_query_len = 16
    descriptor = SimpleNamespace(num_tokens=32)
    parent_result = (
        CUDAGraphMode.NONE,
        descriptor,
        False,
        None,
        None,
    )

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner."
            "_determine_batch_execution_and_padding",
            return_value=parent_result,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "capture_dflash_graph_dispatch"
        ) as capture,
        patch(
            "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
            return_value=diagnostic_enabled,
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=23,
            num_reqs=2,
            num_scheduled_tokens_np=np.array([7, 16], dtype=np.int32),
            max_num_scheduled_tokens=16,
            use_cascade_attn=False,
        )

    assert result is parent_result
    if diagnostic_enabled:
        capture.assert_called_once_with(
            runner.vllm_config,
            path="target",
            runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=descriptor,
            actual_num_tokens=23,
        )
    else:
        capture.assert_not_called()


@pytest.mark.parametrize("diagnostic_enabled", [False, True])
def test_dflash_graph_mode_normalization_is_remembered(
    diagnostic_enabled: bool,
) -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
    )
    runner = object.__new__(NPUModelRunner310)
    runner.vllm_config = config
    runner.speculative_config = config.speculative_config
    runner.compilation_config = config.compilation_config
    runner.cudagraph_dispatcher = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)

    def normalize(*_args, **_kwargs):
        runner.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        runner.cudagraph_dispatcher.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner."
            "_check_and_update_cudagraph_mode",
            side_effect=normalize,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.remember_dflash_graph_modes"
        ) as remember,
        patch(
            "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
            return_value=diagnostic_enabled,
        ),
    ):
        runner._check_and_update_cudagraph_mode([], [])

    assert runner._dflash_requested_cudagraph_mode_310 == CUDAGraphMode.FULL

    if diagnostic_enabled:
        remember.assert_called_once_with(
            config,
            requested_mode=CUDAGraphMode.FULL,
            normalized_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
    else:
        remember.assert_not_called()


def test_dflash_early_fallback_mode_is_remembered() -> None:
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        _dflash_requested_cudagraph_mode_310=CUDAGraphMode.FULL,
        _dflash_effective_cudagraph_mode_310=(
            CUDAGraphMode.FULL_AND_PIECEWISE
        ),
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=compilation_config,
    )
    runner = object.__new__(NPUModelRunner310)
    runner.vllm_config = config
    runner.speculative_config = config.speculative_config
    runner.compilation_config = compilation_config
    runner.cudagraph_dispatcher = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner."
            "_check_and_update_cudagraph_mode"
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "remember_dflash_graph_modes"
        ) as remember,
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=True,
        ),
    ):
        runner._check_and_update_cudagraph_mode([], [])

    assert runner._dflash_requested_cudagraph_mode_310 == CUDAGraphMode.FULL
    assert (
        runner._dflash_effective_cudagraph_mode_310
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    remember.assert_called_once_with(
        config,
        requested_mode=CUDAGraphMode.FULL,
        normalized_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )


def test_dflash_early_fallback_effective_mode_drives_dispatch() -> None:
    runner, _ = _make_k15_spec_runner(
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    runner._dflash_requested_cudagraph_mode_310 = CUDAGraphMode.FULL
    runner._dflash_effective_cudagraph_mode_310 = (
        CUDAGraphMode.FULL_AND_PIECEWISE
    )
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
        lora_id_to_lora_request={},
    )

    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp",
            return_value=False,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runtime_mode, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=7,
            num_reqs=1,
            num_scheduled_tokens_np=np.asarray([7], dtype=np.int32),
            max_num_scheduled_tokens=7,
            use_cascade_attn=False,
        )

    assert runtime_mode == CUDAGraphMode.PIECEWISE


@pytest.mark.parametrize(
    "requested_mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL,
    ],
)
def test_dflash_explicit_eager_fallback_drives_runner_mode(
    requested_mode: CUDAGraphMode,
) -> None:
    runner, _ = _make_k15_spec_runner(CUDAGraphMode.NONE)
    runner.compilation_config = runner.vllm_config.compilation_config
    setattr(
        runner.compilation_config,
        "_dflash_requested_cudagraph_mode_310",
        requested_mode,
    )
    setattr(
        runner.compilation_config,
        "_dflash_effective_cudagraph_mode_310",
        CUDAGraphMode.NONE,
    )
    del runner._dflash_requested_cudagraph_mode_310

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner."
            "_check_and_update_cudagraph_mode"
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runner._check_and_update_cudagraph_mode([], [])

    assert runner._dflash_requested_cudagraph_mode_310 == requested_mode
    assert (
        runner._dflash_effective_cudagraph_mode_310
        == CUDAGraphMode.NONE
    )

    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
        lora_id_to_lora_request={},
    )
    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp",
            return_value=False,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runtime_mode, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=7,
            num_reqs=1,
            num_scheduled_tokens_np=np.asarray([7], dtype=np.int32),
            max_num_scheduled_tokens=7,
            use_cascade_attn=False,
        )

    assert runtime_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    ("requested_mode", "normalized_mode"),
    [
        pytest.param(
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.PIECEWISE,
            id="combined-to-piecewise",
        ),
        pytest.param(
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            id="full-to-combined",
        ),
    ],
)
def test_dflash_unmarked_backend_normalization_does_not_expand_runtime(
    requested_mode: CUDAGraphMode,
    normalized_mode: CUDAGraphMode,
) -> None:
    runner, _ = _make_k15_spec_runner(requested_mode)
    runner.compilation_config = runner.vllm_config.compilation_config

    def normalize(*_args, **_kwargs):
        runner.compilation_config.cudagraph_mode = normalized_mode
        runner.cudagraph_dispatcher.cudagraph_mode = normalized_mode

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner."
            "_check_and_update_cudagraph_mode",
            side_effect=normalize,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runner._check_and_update_cudagraph_mode([], [])

    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
        lora_id_to_lora_request={},
    )
    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp",
            return_value=False,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.enable_sp_by_pass",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p."
            "dflash_diagnostic_enabled",
            return_value=False,
        ),
    ):
        runtime_mode, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=7,
            num_reqs=1,
            num_scheduled_tokens_np=np.asarray([7], dtype=np.int32),
            max_num_scheduled_tokens=7,
            use_cascade_attn=False,
        )

    assert runtime_mode == CUDAGraphMode.NONE


def _prepare_inputs_source() -> str:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def _prepare_inputs(")
    end = source.index("    @torch.inference_mode()", start)
    return source[start:end]


def test_prepare_inputs_keeps_aclgraph_metadata_on_cpu() -> None:
    source = _prepare_inputs_source()

    assert "block_table.compute_slot_mapping(" in source
    assert "req_indices," in source
    assert "positions_np[:total_num_scheduled_tokens]" in source

    assert "self.input_batch.block_table.compute_slot_mapping(" not in source
    assert "query_start_loc.gpu[: num_reqs + 1]" not in source
    assert "req_indices_gpu" not in source
    assert "self.num_computed_tokens[req_indices_gpu]" not in source

    assert "self.positions[:total_num_scheduled_tokens].copy_(" in source
    assert "self._positions_cpu_buf[:total_num_scheduled_tokens]" in source
    assert "self.seq_lens[:num_reqs].copy_(" in source
    assert "self.optimistic_seq_lens_cpu[:num_reqs]" in source


def test_model_forward_updates_mtp_full_graph_params_before_replay() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.uses_mrope = False
    runner.enable_enpu = False
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.update_stream = MagicMock()
    runner._all_gather_hidden_states_and_aux = MagicMock()

    calls = []

    def fake_update(*args):
        calls.append("update")

    def fake_model(**kwargs):
        calls.append("model")
        return torch.ones(1)

    runner.model = fake_model
    runner._update_full_graph_params_if_needed = fake_update
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=False,
        flash_comm_v1_enabled=False,
    )

    with patch(
        "vllm_ascend._310p.model_runner_310p.get_forward_context",
        return_value=forward_context,
    ):
        hidden_states = runner._model_forward(
            8,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
        )

    assert calls == ["update", "model"]
    torch.testing.assert_close(hidden_states, torch.ones(1))


class TestNPUModelRunner310(TestBase):
    def test_dflash_shared_attention_cache_uses_per_layer_physical_views(self):
        runner = object.__new__(NPUModelRunner310)
        runner.speculative_config = SimpleNamespace(method="dflash")
        runner.runner_only_attn_layers = set()
        runner.device = torch.device("cpu")
        runner._acl_format = 29
        runner.attn_backend = SimpleNamespace(
            get_supported_kernel_block_sizes=lambda: [128, 64],
            get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size: (
                2,
                num_blocks,
                (num_kv_heads * head_size) // 16,
                block_size,
                16,
            ),
        )
        target_spec = AttentionSpec(
            block_size=2560,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.float16,
        )
        draft_spec = AttentionSpec(
            block_size=1280,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.float16,
        )
        self.assertEqual(target_spec.page_size_bytes, draft_spec.page_size_bytes)
        num_blocks = 2
        config = SimpleNamespace(
            num_blocks=num_blocks,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["target.attn"],
                    kv_cache_spec=target_spec,
                ),
                SimpleNamespace(
                    layer_names=["draft.attn"],
                    kv_cache_spec=draft_spec,
                ),
            ],
            kv_cache_tensors=[
                SimpleNamespace(
                    size=target_spec.page_size_bytes * num_blocks,
                    shared_by=["target.attn", "draft.attn"],
                )
            ],
        )

        def allocate():
            with patch(
                "vllm_ascend._310p.model_runner_310p.torch_npu.empty_with_format",
                side_effect=lambda size, dtype, device, acl_format: torch.empty(
                    size,
                    dtype=dtype,
                ),
            ):
                return runner._allocate_kv_cache_tensors(config)

        caches = allocate()

        target_k, target_v = caches["target.attn"]
        draft_k, draft_v = caches["draft.attn"]
        self.assertEqual(target_k.shape, (80, 16, 64, 16))
        self.assertEqual(draft_k.shape, (20, 32, 128, 16))
        self.assertEqual(target_k.data_ptr(), draft_k.data_ptr())
        self.assertEqual(target_v.data_ptr(), draft_v.data_ptr())
        self.assertIsNot(target_k, draft_k)
        self.assertIsNot(target_v, draft_v)

        runner.speculative_config = SimpleNamespace(method="dspark")
        non_dflash_caches = allocate()
        self.assertIs(
            non_dflash_caches["target.attn"][0],
            non_dflash_caches["draft.attn"][0],
        )

        runner.speculative_config = SimpleNamespace(method="dflash")
        config.kv_cache_groups[1].kv_cache_spec = target_spec
        uniform_caches = allocate()
        self.assertIs(
            uniform_caches["target.attn"][0],
            uniform_caches["draft.attn"][0],
        )

        config.kv_cache_groups[1].kv_cache_spec = AttentionSpec(
            block_size=1280,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(RuntimeError, "must use one dtype"):
            allocate()

        config.kv_cache_groups[1].kv_cache_spec = AttentionSpec(
            block_size=640,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.float16,
            page_size_padded=target_spec.page_size_bytes,
        )
        with self.assertRaisesRegex(RuntimeError, "equal storage sizes"):
            allocate()

    def test_dflash_attention_metadata_captures_state_indices(self):
        runner = object.__new__(NPUModelRunner310)
        runner.speculative_config = SimpleNamespace(method="dflash")
        gdn_metadata = SimpleNamespace(
            spec_state_indices_tensor=torch.tensor([[2, 3, 4]]),
            num_accepted_tokens=torch.tensor([2]),
            spec_query_start_loc=torch.tensor([0, 3]),
            spec_sequence_masks=torch.tensor([[True, True, True]]),
        )
        result = ({"layer.gdn": gdn_metadata}, SimpleNamespace())

        with (
            patch(
                "vllm_ascend._310p.model_runner_310p.NPUModelRunner._build_attention_metadata",
                return_value=result,
            ),
            patch(
                "vllm_ascend._310p.model_runner_310p.dflash_diagnostic_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.model_runner_310p.capture_dflash_diagnostic"
            ) as capture,
        ):
            actual = runner._build_attention_metadata()

        self.assertIs(actual, result)
        capture.assert_called_once()
        self.assertEqual(capture.call_args.args, ("state_metadata",))
        payload = capture.call_args.kwargs["payload_builder"]()
        self.assertEqual(payload["layer_name"], "layer.gdn")
        torch.testing.assert_close(
            payload["state_indices"], gdn_metadata.spec_state_indices_tensor
        )
        torch.testing.assert_close(
            payload["num_accepted_tokens"], gdn_metadata.num_accepted_tokens
        )
        torch.testing.assert_close(
            payload["query_start_loc"], gdn_metadata.spec_query_start_loc
        )
        torch.testing.assert_close(
            payload["sequence_masks"], gdn_metadata.spec_sequence_masks
        )

    def test_dflash_state_diagnostic_failure_preserves_attention_metadata(self):
        class RaisingMetadata:
            @property
            def spec_state_indices_tensor(self):
                raise RuntimeError("diagnostic-only failure")

        runner = object.__new__(NPUModelRunner310)
        runner.speculative_config = SimpleNamespace(method="dflash")
        result = ({"layer.gdn": RaisingMetadata()}, SimpleNamespace())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            env = {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(path)}
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "vllm_ascend._310p.model_runner_310p.NPUModelRunner._build_attention_metadata",
                    return_value=result,
                ),
                patch(
                    "vllm_ascend._310p.spec_decode.dflash_diagnostics_310.logger.warning_once"
                ),
            ):
                _reset_dflash_diagnostics_for_test()
                actual = runner._build_attention_metadata()
            _reset_dflash_diagnostics_for_test()

            self.assertIs(actual, result)
            self.assertFalse(path.exists())

    def test_may_reinitialize_input_batch_expands_prefix_mamba_block_table(self):
        runner = object.__new__(NPUModelRunner310)
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.max_encoder_len = 0
        runner.max_num_tokens = 1024
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.model_config = SimpleNamespace(max_model_len=512, get_vocab_size=lambda: 32000)
        runner.cache_config = SimpleNamespace(block_size=128, enable_prefix_caching=True)
        runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=4)
        runner.vllm_config = SimpleNamespace(speculative_config=None)
        runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
        runner.input_batch = SimpleNamespace(logitsprocs=MagicMock())
        attention_backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
        runner.attn_groups = [[SimpleNamespace(backend=attention_backend)]]

        attention_spec = AttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        mamba_spec = MambaSpec(
            block_size=128,
            shapes=((16,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec),
                SimpleNamespace(kv_cache_spec=mamba_spec),
            ]
        )

        with (
            patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch") as mock_input_batch,
            patch("vllm_ascend._310p.model_runner_310p.get_total_cp_world_size", return_value=1),
        ):
            runner.may_reinitialize_input_batch(kv_cache_config)

        kwargs = mock_input_batch.call_args.kwargs
        self.assertEqual(kwargs["block_sizes"], [128, 128])
        self.assertEqual(kwargs["kernel_block_sizes"], [[128, 64], [0]])
        self.assertEqual(kwargs["max_num_blocks_per_req"], [4, 6])
        self.assertIs(kwargs["kv_cache_groups"], kv_cache_config.kv_cache_groups)
        self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 4)

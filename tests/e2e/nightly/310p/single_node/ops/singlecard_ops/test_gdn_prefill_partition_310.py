# SPDX-License-Identifier: Apache-2.0
"""The scheduler must preserve GDN's numerical chunks across token budgets."""

from itertools import cycle
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend._310p.gdn_constants import GDN_PREFILL_CHUNK_SIZE
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
from vllm_ascend.patch.platform.patch_mamba_scheduler_310 import _dflash_mamba_block_aligned_split
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("tokens", [192, 278, 4106])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_scheduled_gdn_prefill_matches_unchunked(tokens, dim, state_dtype):
    enable_custom_op()
    torch.manual_seed(20260919)
    q = torch.randn(1, tokens, 2, dim).half().npu()
    k = torch.randn(1, tokens, 2, dim).half().npu()
    # WY supports 64/128 for K and V, but the complete FwdH/FwdO path
    # requires V to be a multiple of 128.
    v_dim = 128
    v = (0.1 * torch.randn(1, tokens, 4, v_dim)).half().npu()
    g = (-0.01 * torch.rand(1, tokens, 4)).npu()
    beta = (0.2 + 0.6 * torch.rand(1, tokens, 4)).half().npu()
    initial = torch.zeros((1, 4, v_dim, dim), dtype=state_dtype, device="npu")
    spec = MambaSpec(
        block_size=2304,
        shapes=((4, v_dim, dim),),
        dtypes=(state_dtype,),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    scheduler = SimpleNamespace(
        block_size=2304,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method="dflash")),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]),
    )
    request = SimpleNamespace(num_tokens=tokens, num_prompt_tokens=tokens, num_computed_tokens=0)

    def run(start, count, state):
        inputs = [x[:, start : start + count].contiguous() for x in (q, k, v, g, beta)]
        return chunk_gated_delta_rule_310(
            *inputs,
            initial_state=state,
            output_final_state=True,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    expected, expected_state = run(0, tokens, initial)
    state = initial
    pieces = []
    # Include an off-boundary first chunk and insufficient residual budgets.
    for budget in cycle((150, 1952, 23, 512)):
        start = request.num_computed_tokens
        count = _dflash_mamba_block_aligned_split(scheduler, request, min(budget, tokens - start))
        if count == 0:
            continue
        output, state = run(start, count, state.to(state_dtype))
        pieces.append(output)
        request.num_computed_tokens += count
        if request.num_computed_tokens == tokens:
            break
        assert request.num_computed_tokens % GDN_PREFILL_CHUNK_SIZE == 0

    torch.testing.assert_close(torch.cat(pieces, dim=1).cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(state.cpu(), expected_state.cpu(), rtol=0, atol=0)

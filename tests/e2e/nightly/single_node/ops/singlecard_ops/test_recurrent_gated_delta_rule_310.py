import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

torch_npu.npu.set_compile_mode(jit_compile=False)


def npu_recurrent_gated_delta_rule_310(
    query,
    key,
    value,
    beta,
    state,
    actual_seq_lengths,
    ssm_state_indices,
    g=None,
    gk=None,
    num_accepted_tokens=None,
    scale=1.0,
):
    """Call RecurrentGatedDeltaRule."""
    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=query,
        key=key,
        value=value,
        g=g,
        gk=gk,
        beta=beta,
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale_value=scale,
    )
    return out


def golden_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    g,
    gk,
    num_accepted_tokens,
):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()

    beta = torch.ones(T, n_heads_v).to(torch.float32) if beta is None else beta.to(torch.float32)
    o = torch.empty_like(v).to(torch.float32)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q = q * scale

    seq_start = 0
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]]

        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + actual_seq_lengths[i]):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_k)]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_k)]
                v_i = v[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                beta_i = beta[slot_id][head_id]
                S = S * alpha_i
                if gk is not None:
                    S = S * gk[slot_id][head_id].to(torch.float32).exp().unsqueeze(0)
                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S_ = y[:, None] * k_i[None, :]
                S = S + S_
                initial_state[ssm_state_indices[slot_id]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(query.dtype), initial_state.to(query.dtype)


def run_recurrent_case(
    actual_lengths,
    *,
    headnum=(4, 8),
    headdim_k=128,
    headdim_v=128,
    with_g=True,
    with_gk=False,
    accepted_tokens=None,
    state_indices=None,
):
    enable_custom_op()
    torch.manual_seed(20260813 + sum(actual_lengths))
    dtype = torch.float16
    headnum_k, headnum_v = headnum
    actual_seq_lengths = torch.tensor(actual_lengths, dtype=torch.int32)
    total_tokens = int(actual_seq_lengths.sum())
    state = torch.rand(
        (total_tokens, headnum_v, headdim_v, headdim_k),
        dtype=dtype,
    )
    query = torch.nn.functional.normalize(
        torch.rand((total_tokens, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((total_tokens, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((total_tokens, headnum_v, headdim_v), dtype=dtype)
    g = -torch.rand((total_tokens, headnum_v), dtype=torch.float32) if with_g else None
    gk = (
        -torch.rand((total_tokens, headnum_v, headdim_k), dtype=torch.float32)
        if with_gk
        else None
    )
    beta = torch.rand((total_tokens, headnum_v), dtype=dtype)
    ssm_state_indices = (
        torch.arange(total_tokens, dtype=torch.int32)
        if state_indices is None
        else torch.tensor(state_indices, dtype=torch.int32)
    )
    accepted = (
        None
        if accepted_tokens is None
        else torch.tensor(accepted_tokens, dtype=torch.int32)
    )
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        actual_seq_lengths,
        ssm_state_indices,
        g,
        gk,
        accepted,
    )

    state_npu = state.npu()
    out = npu_recurrent_gated_delta_rule_310(
        query.npu(),
        key.npu(),
        value.npu(),
        beta.npu(),
        state_npu,
        actual_seq_lengths.npu(),
        ssm_state_indices.npu(),
        g=None if g is None else g.npu(),
        gk=None if gk is None else gk.npu(),
        num_accepted_tokens=None if accepted is None else accepted.npu(),
        scale=scale,
    )

    atol = 2.5e-2 if max(actual_lengths) > 2 else 1e-2
    torch.testing.assert_close(
        out.to(torch.float32).cpu(),
        out_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=atol,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=atol,
        equal_nan=True,
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
def test_fused_recurrent_gated_delta_rule_310(batch_size, mtp, headnum, headdim_k, headdim_v):
    enable_custom_op()
    dtype = torch.float16
    headnum_k, headnum_v = headnum
    actual_seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(actual_seq_lengths))
    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(dtype)
    query = torch.nn.functional.normalize(torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1).to(dtype)
    key = torch.nn.functional.normalize(torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch_size,), dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        actual_seq_lengths,
        ssm_state_indices,
        g,
        None,
        num_accepted_tokens,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    state_npu = state.npu()
    out = npu_recurrent_gated_delta_rule_310(
        query.npu(),
        key.npu(),
        value.npu(),
        beta.npu(),
        state_npu,
        actual_seq_lengths.npu(),
        ssm_state_indices.npu(),
        g=g.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
        scale=scale,
    )
    out = out.to(torch.float32).cpu()

    torch.testing.assert_close(
        out.to(torch.float32).cpu(),
        out_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_recurrent_multi_v_tile_state_prefetch():
    run_recurrent_case(
        [1],
        headdim_v=512,
        with_g=False,
        accepted_tokens=[1],
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_recurrent_fp16_output_tile_alignment():
    run_recurrent_case(
        [1],
        headdim_v=416,
        with_g=False,
        accepted_tokens=[1],
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_recurrent_repeated_state_writeback():
    run_recurrent_case(
        [8],
        headdim_v=512,
        with_g=False,
        accepted_tokens=[7],
        state_indices=[0] * 8,
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("length", (3, 16))
def test_recurrent_window_matches_persistent_fp16_step_decode(length):
    """A verification window must observe the persisted state precision.

    Target decode writes the FP16 recurrent state after every token and reloads
    it for the next call. A multi-token verification call must produce the same
    outputs and per-token persistent states from the same inputs and initial
    state instead of retaining extra FP32 precision between tokens.
    """
    enable_custom_op()
    torch.manual_seed(20260815)
    num_k_heads = 8
    num_v_heads = 16
    head_dim = 128
    dtype = torch.float16
    scale = head_dim**-0.5

    query = torch.nn.functional.normalize(
        torch.randn(length, num_k_heads, head_dim), dim=-1
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.randn(length, num_k_heads, head_dim), dim=-1
    ).to(dtype)
    value = torch.randn(length, num_v_heads, head_dim, dtype=dtype)
    beta = torch.rand(length, num_v_heads, dtype=dtype)
    g = -torch.rand(length, num_v_heads, dtype=torch.float32)
    initial_state = torch.randn(
        length,
        num_v_heads,
        head_dim,
        head_dim,
        dtype=dtype,
    )
    window_indices = torch.arange(length, dtype=torch.int32)
    one_accepted = torch.ones(1, dtype=torch.int32)

    window_state = initial_state.npu()
    window_output = npu_recurrent_gated_delta_rule_310(
        query.npu(),
        key.npu(),
        value.npu(),
        beta.npu(),
        window_state,
        torch.tensor([length], dtype=torch.int32).npu(),
        window_indices.npu(),
        g=g.npu(),
        num_accepted_tokens=one_accepted.npu(),
        scale=scale,
    )

    step_state = initial_state.npu()
    step_outputs = []
    step_states = []
    step_index = torch.zeros(1, dtype=torch.int32).npu()
    step_length = torch.ones(1, dtype=torch.int32).npu()
    for token_idx in range(length):
        step_outputs.append(
            npu_recurrent_gated_delta_rule_310(
                query[token_idx : token_idx + 1].npu(),
                key[token_idx : token_idx + 1].npu(),
                value[token_idx : token_idx + 1].npu(),
                beta[token_idx : token_idx + 1].npu(),
                step_state,
                step_length,
                step_index,
                g=g[token_idx : token_idx + 1].npu(),
                num_accepted_tokens=one_accepted.npu(),
                scale=scale,
            )
        )
        step_states.append(step_state[0].clone())

    torch.testing.assert_close(
        window_output.cpu(),
        torch.cat(step_outputs).cpu(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        window_state[:length].cpu(),
        torch.stack(step_states).cpu(),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("length", range(1, 17))
def test_recurrent_verification_window_lengths(length):
    run_recurrent_case(
        [length],
        accepted_tokens=[length],
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_recurrent_uneven_k15_batch_and_accepted_state():
    run_recurrent_case(
        [1, 4, 9, 16],
        accepted_tokens=[1, 2, 5, 15],
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("with_g", [False, True])
@pytest.mark.parametrize("with_gk", [False, True])
def test_recurrent_optional_gates(with_g, with_gk):
    run_recurrent_case(
        [3, 16],
        with_g=with_g,
        with_gk=with_gk,
        accepted_tokens=[2, 15],
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("headnum", [(16, 16), (16, 32)])
def test_recurrent_k15_qwen_shapes(headnum):
    run_recurrent_case(
        [16],
        headnum=headnum,
        accepted_tokens=[15],
    )


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_recurrent_k15_repeated_state_slot():
    run_recurrent_case(
        [16],
        with_g=False,
        accepted_tokens=[15],
        state_indices=[0] * 16,
    )

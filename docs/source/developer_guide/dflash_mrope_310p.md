# 310P DFlash m-RoPE implementation and validation

> 本文保留 47dbcb3 基线上的实现及实机记录；最新 f48bd99 分支在指定容器中的部署状态见 [容器安装记录](dflash_mrope_container_install_zh.md)。

See the [Chinese change summary](dflash_mrope_310p_summary_zh.md) and
[regression report](dflash_mrope_310p_regression_zh.md) for the completed results:
149 unit tests, 132 on-device operator tests, 24 ordinary-RoPE baseline comparisons
and 13 m-RoPE greedy target-only comparisons passed. The latter include TP=2
eager and FULL_DECODE_ONLY. Nine-repeat same-device performance follow-ups stayed
within the 5% regression threshold. The new checkpoint has low acceptance on the
tested image prompt, so these results do not establish a DFlash speedup.

## Scope and source versions

This change is confined to vLLM-Ascend. The implementation worktree is
`/home/qzh/vllm-ascend-dflash-mrope`, branch `feat/310p-dflash-mrope`, based on
Ascend `47dbcb3`. Runtime validation uses the read-only vLLM checkout
`/home/qzh/vllm-v0.24.0-clean` at `ee0da84` (v0.24.0).
The pristine Ascend comparison worktree is
`/home/qzh/vllm-ascend-dflash-mrope-baseline`.

The target is `/home/models/Qwen3-VL-4B-Instruct`. The draft is
`/home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750`.
Neither checkpoint nor SpecForge nor upstream vLLM is modified.
The initial scope is text/image input and text output; video is not validated.

## What the checkpoint learned

The training framework is `/home/specforge-sglang/src/SpecForge-pr585`.
The saved checkpoint configuration and saved `dflash.py` are the reference for
this run; subsequent edits to a training framework do not change saved weights.

| Item | Checkpoint / training behavior |
| --- | --- |
| Target | 36 language decoder layers, hidden size 2560 |
| Captured features | Post-layer outputs at zero-based layers `[3,10,18,25,32]` |
| HF tuple indexing | `hidden_states[4,11,19,26,33]`, because index 0 is the embedding output |
| Feature fusion | Concatenate five 2560-wide tensors, then `fc: 12800 -> 2560` and `hidden_norm` |
| Draft | Five decoder layers, 32 Q heads, 8 KV heads, head dimension 128, intermediate size 9728 |
| Context | Each draft layer projects K/V from the same fused target features; features are not assigned one per draft layer |
| Block | One known anchor plus seven masked candidate positions; block size 8 |
| Mask / loss | Bidirectional attention within the query block, context strictly before the anchor; same-position CE on positions 1–7, excluding the anchor |
| Rotary | Full 128 dimensions, theta 5000000, interleaved T/H/W sections `[24,20,20]` |
| Vocabulary | 151936; mask token 151669; target embedding and LM head are shared |
| Saved weights | 537,427,200 parameters, BF16; no separate embedding/head or reduced-vocabulary mapping |

The saved training arguments specify HF target execution with SDPA, three epochs,
step 3750, batch size 1, accumulation 1, learning rate 2e-4, warmup ratio 0.04,
maximum sequence length 4096, maximum anchors 32, gamma 4 and seed 42.
Image pixel limits are 50176–802816. The configured five draft layers override
the command-line default of one layer. Runtime FP16 does not rewrite BF16 weights.

The saved draft's convenience generation routine uses one-dimensional positions;
it is not an image-position reference. Validation uses HF's Qwen3-VL position
construction and interleaved rotary implementation instead.

## Implementation

`patch/worker/patch_idex_310.py` binds methods onto the actual Ascend proposer.
It also overrides the guard on the temporary upstream **DFlash** proposer created
by the GPU runner constructor before the NPU runner replaces it. The shared
proposer guard and other speculative methods retain their existing restrictions.
The branch is selected from the **draft configuration's `uses_mrope`**, not the
target's multimodal capability.

`_310p/spec_decode/dflash_mrope.py` separates two kinds of position:

* Context/query rotary coordinates have shape `[3,N]` in T/H/W order.
* Cache positions have shape `[N]` and count tokens, including image tokens.

Context coordinates are copied in the same order and with the same extent as
the incoming target hidden states. Logical context offsets are reconstructed
from scheduled lengths and cumulative query boundaries. The existing AscendC
input expansion and physical/per-layer slot reconstruction receive these
logical offsets. Repeated image coordinates therefore cannot alias cache slots.
Query positions are `effective_sequence_length + request_mrope_delta + offset`,
repeated on all three axes for generated text. Request deltas are read from CPU
request metadata in the current input-batch order; no new device-to-host read
is needed. Rejected speculative suffixes are excluded from the effective length.
Unfinished multimodal prefills can occur in a padded drafter batch before their
full-prompt delta gives nonnegative query coordinates. Their unused candidates
are discarded by the runner; the query start is clamped to zero to keep these
temporary rotary lookups valid. Complete prompts retain their exact positions.

The proposer owns preallocated three-axis buffers and a `DFlashMRoPEState310`.
This state maintains separate persistent context and query cos/sin tensors.
The base frequency table comes from the **draft** rotary module. It is read-only:
the target's rotary module and target global cos/sin are never overwritten, even
when vLLM's rotary factory returns a shared module. The interleaved rule selects
H at indices `1:60:3`, W at `2:60:3`, and T elsewhere.

Context K follows the existing fused projection, per-head K normalization and
NZ cache update. Only rotary application changes. Query Q/K follows the existing
projection, normalization, attention and output projection, using the independent
query cos/sin. Without draft-local m-RoPE state, the original attention forward
is called. Existing embedding/head sharing, hidden-layer offsets, reduced-vocabulary
mapping, noncausal block attention and sliding cache layouts are retained.

Before draft execution, persistent rotary tensors are refreshed outside the
compiled model. Unused capacity is filled with identity rotation (cos=1, sin=0).
Captured forwards read stable tensor addresses. The same refresh occurs on eager
fallback and profiling paths; no global enable/disable state is introduced for
m-RoPE. Actual graph validation status is recorded in the accompanying report.

## Reproduce a validation case

Run from the repository root, with matching vLLM, torch-npu and compiled Ascend
operators available. Use module execution because this repository's `tools`
directory contains names that can shadow Python standard-library modules.

```bash
python -m tools.run_310p_dflash_mrope_validation \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --draft /home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750 \
  --image --max-tokens 64 --repeats 3 \
  --output artifacts/mrope/vl-image-eager.json

python -m tools.validate_310p_dflash_mrope_reference \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --output artifacts/mrope/hf-npu-reference.json
```

The harness uses TP=1, FP16, block size 128, synchronous scheduling, no prefix
cache and no chunked prefill by default. It performs one warmup and then records
per-request token IDs, text, timing and speculative metrics. Remove `--draft`
for the target-only greedy reference. Add `--mode piecewise` or `--mode full`
for PIECEWISE or FULL_DECODE_ONLY. Other controls include `--tp 2`, `--batch 8`,
`--images-per-prompt 2`, `--mixed`, `--prefix-cache`, `--chunked-prefill`,
`--batched-tokens`, `--prompt-repeat`, `--async-scheduling`, `--respect-eos`
and `--temperature`. Sampling runs are statistical checks, not exact random
sequence comparisons. No new required production configuration is introduced.

The validation tools deliberately synchronize to measure/compare outputs.
These are offline tools; production inference adds no diagnostic synchronization.

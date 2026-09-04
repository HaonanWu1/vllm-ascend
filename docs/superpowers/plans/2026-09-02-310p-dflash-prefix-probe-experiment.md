# 310P DFlash Prefix Cache Probe Experiment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this diagnostic plan task-by-task. Do not implement a product fix during this experiment.

**Goal:** 在不干扰远程服务器上另一项性能任务的前提下，通过 Prefix Cache 开/关 A/B 和环境变量控制的运行时探针，定位 310P DFlash 精度下降的第一个内部状态分歧。

**Architecture:** 探针只在 Eager、关闭 async scheduling 的诊断服务中启用。在 Target→Draft、DFlash context KV、Draft 输出和 rejection sampler 四个边界记录小型元数据及张量摘要；先确定分歧属于 prefix/context、draft、target verification 还是 rejection。只有第一阶段证明 sampler 输出正确但下一轮 target 状态异常时，才增加 GDN rollback 探针。

**Tech Stack:** Python 3.12、PyTorch/torch-npu、vLLM/vllm-ascend、AISBench、Docker、Atlas 310P TP2。

**Spec:** 本文档同时作为诊断实验设计和执行计划，不包含产品修复设计。

## Global Constraints

- 基线提交固定为 `5584725b857f5ed12ded731c501b23ea3f0fce3c`，分支来源为 `dev_v24/feat-310p-dflash`。
- 不修改 `/home/shichuchao/scc_dflash/vllm-ascend`，探针代码只能位于独立 worktree `/home/shichuchao/scc_dflash/prefix_probe_wt`。
- 不执行 `pip install -e`，运行时通过 `PYTHONPATH=/home/shichuchao/scc_dflash/prefix_probe_wt` 加载探针代码。
- 不读取业务请求、不停止、不重启或复用另一个 agent 的 vLLM 服务；不向它的端口发送请求。
- 允许在同一容器内使用确认空闲的 NPU 并行执行诊断。启动前必须同时确认目标 NPU 无计算进程、实验端口未监听、输出目录独立；不得执行任何针对其他任务的 `kill`、`pkill`、`docker stop` 或重启操作。
- 第一阶段使用端口 `6766`；P4 优先复用已释放的 `6766`，若被占用则选择新的空闲端口并记录。AISBench 和探针产物写入独立目录 `/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902`。
- A/B 除 `--enable-prefix-caching` 与 `--no-enable-prefix-caching` 外必须完全一致：Eager、TP2、DFlash K=15、temperature=0、`ignore_eos=false`、async scheduling 关闭。
- 探针默认关闭；关闭时不得触发 NPU 同步、D2H 拷贝、文件写入或额外张量计算。
- 本实验只定位根因，不提交修复，不把探针性能数据用于性能结论。

---

## 1. 已知证据和待验证假设

已完成的 50 题 A/B：

| 运行配置 | AISBench 精度 | 第 4 个 few-shot 尾段串入 |
|---|---:|---:|
| Eager + Prefix Cache 关闭 | 49/50（手工复核 98%） | 0/50 |
| Eager + Prefix Cache 开启 | 27/50（官方 54%） | 17/50，且 17 题全部判错 |

当前代码链路显示：

1. Prefix hit 后，`vllm_ascend/worker/model_runner_v1.py` 只把本轮 scheduled token 及其 hidden states 传给 DFlash。
2. `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py` 只为这些 context token 执行 context KV 写入。
3. Draft attention 随后仍使用完整 `seq_lens` 和完整 `block_table`。

实验按以下优先级检验假设：

- **H1：DFlash prefix context KV 复用错误。** Prefix 命中的 draft KV 块未正确填充、复用或寻址，P1 首先分歧。
- **H2：Target verification 已受 prefix/GDN 状态污染。** P1/P2 后，P3 的 target argmax 相对 Prefix-off 首先分歧。
- **H3：Greedy rejection 错误放行。** Target argmax 正确，但 sampler 输出越过首个 mismatch，或未输出 target EOS。
- **H4：GDN speculative rollback 错误。** P3 输出正确，但下一 decode step 的 target 状态或 logits 首先分歧。

## 2. 探针接口与产物

### 2.1 环境变量

新增以下诊断变量：

```bash
VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE=1
VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_MAX_STEPS=320
```

Prefix-off 使用：

```bash
VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_DIR=/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_off
```

Prefix-on 使用：

```bash
VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_DIR=/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_on
```

探针模块在 import 时校验输出目录不为空，并按 `LOCAL_RANK` 创建 rank 子目录。

### 2.2 Python 接口

新增模块 `vllm_ascend/_310p/dflash_prefix_probe.py`，只提供以下窄接口：

```python
def is_prefix_probe_enabled() -> bool:
    """返回探针是否启用；结果在模块加载时缓存。"""

def record_prefix_probe(
    stage: str,
    *,
    request_keys: tuple[str, ...],
    step: int,
    tensors: dict[str, torch.Tensor | None],
    scalars: dict[str, object],
) -> None:
    """记录一次边界事件；关闭时立即返回。"""
```

每个事件按 `LOCAL_RANK` 写入 `rank0/trace.jsonl` 或 `rank1/trace.jsonl`。小型整型张量保存完整值；浮点张量仅保存：

- shape、dtype、device；
- `all_finite`；
- FP32 sum、L2 norm、max absolute；
- 首尾各最多 8 个 active 元素；
- active row 数和 active row 索引。

请求使用“prompt token 哈希 + 已生成 token 哈希”形成 `request_keys`，不依赖两次服务启动间不稳定的内部 request ID。

### 2.3 四个第一阶段边界

| Stage | 调用点 | 必须记录的数据 |
|---|---|---|
| `P0_TARGET_TO_DRAFT` | `model_runner_v1.py` 调用 `drafter._propose` 前 | prompt/output 哈希、`num_computed_tokens`、`num_scheduled_tokens`、target token IDs、positions、target hidden 摘要、query boundaries |
| `P1_DRAFT_CONTEXT` | `dflash_proposer_310.py` input expansion 后、`precompute_and_store_context_kv` 前后 | context positions、每层 context slot mapping、`seq_lens`、block table active rows、hidden 摘要、写入 KV active slots 摘要 |
| `P2_DRAFT_OUTPUT` | drafter 返回后 | 每请求 draft token IDs、draft 数量、首轮可用 logits 的 top-k/argmax 摘要 |
| `P3_REJECTION` | `AscendRejectionSampler310.forward` 前后 | draft IDs、target argmax、bonus IDs、cu draft counts、首个 mismatch、sampler 输出、有效输出数量 |

第一阶段不修改 GDN kernel，不 dump 全层 hidden/KV，不记录模型权重。

## 3. 判定规则

比较器按 `request_key + step + stage + rank` 对齐 Prefix-off 与 Prefix-on：

| 第一个分歧 | 结论 |
|---|---|
| P0 的 token/position/query metadata | Prefix manager→DFlash 输入契约错误 |
| P0 metadata 相同，P1 slot/block/KV 首先不同 | DFlash prefix context KV 未正确填充、复用或寻址 |
| P1 相同，P2 draft token/logit 首先不同 | Draft attention/RoPE/GDN 内部问题 |
| P2 不同，但 P3 target argmax 相同且 sampler 遵守首 mismatch | Draft 错误被正确拦截，不足以解释最终精度下降 |
| P3 target argmax 相同，但 sampler 输出越过首 mismatch | 310P greedy rejection 路径根因 |
| P3 sampler 输出正确，下一 step 的 P0 target 首先不同 | 进入第二阶段，检查 GDN accepted-token rollback |

Greedy rejection 的硬性判据：对每个请求，从位置 0 开始比较 `draft_token_ids` 与 `target_argmax`；sampler 只能输出首个 mismatch 之前的相同 token，并在 mismatch 位置输出 target token。只有全部 draft token 相同时才允许追加 bonus token。

## 4. 实施任务

### Task 1: 创建隔离 worktree 并验证基线

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes: 当前远程仓库和固定提交 `5584725b857f5ed12ded731c501b23ea3f0fce3c`。
- Produces: 隔离目录 `/home/shichuchao/scc_dflash/prefix_probe_wt`。

- [ ] **Step 1: 只读确认当前 checkout 和 worktree 状态**

```bash
cd /home/shichuchao/scc_dflash/vllm-ascend
git status --short --branch
git rev-parse HEAD
git worktree list
```

Expected: HEAD 为固定提交；现有工作目录不被修改。

- [ ] **Step 2: 创建独立 worktree**

```bash
cd /home/shichuchao/scc_dflash/vllm-ascend
git worktree add /home/shichuchao/scc_dflash/prefix_probe_wt -b diag/310p-dflash-prefix-probe 5584725b857f5ed12ded731c501b23ea3f0fce3c
```

Expected: 新 worktree 位于固定路径，原 checkout 的 status 不变化。如果 branch 或目录已存在，先只读检查其 HEAD；仅在 HEAD 完全相同时复用，不删除现有目录或分支。

- [ ] **Step 3: 运行现有定向纯 Python 基线测试**

```bash
cd /home/shichuchao/scc_dflash/prefix_probe_wt
PYTHONPATH=$PWD pytest -q tests/ut/_310p/spec_decode/test_dflash_proposer_310.py tests/ut/_310p/sample/test_rejection_sampler_310.py
```

Expected: 现有定向测试通过；不得启动 vLLM 服务。

### Task 2: 实现默认关闭的探针模块

**Files:**
- Create: `vllm_ascend/_310p/dflash_prefix_probe.py`
- Create: `tests/ut/_310p/test_dflash_prefix_probe.py`

**Interfaces:**
- Consumes: `torch.Tensor`、`LOCAL_RANK` 和三个探针环境变量。
- Produces: `is_prefix_probe_enabled()`、`record_prefix_probe(...)`、rank 隔离 JSONL 产物。

- [ ] **Step 1: 写默认关闭测试**

验证未设置环境变量时：`record_prefix_probe` 不创建目录、不访问 tensor 内容、不调用 NPU 同步。

- [ ] **Step 2: 写启用后的序列化测试**

使用 CPU tensor 验证：整型小 tensor 完整保存；浮点 tensor 只保存摘要；不同 rank 写入不同路径；超过 `MAX_STEPS` 的事件被忽略。

- [ ] **Step 3: 运行测试并确认先失败**

```bash
cd /home/shichuchao/scc_dflash/prefix_probe_wt
PYTHONPATH=$PWD pytest -q tests/ut/_310p/test_dflash_prefix_probe.py
```

Expected: 因探针模块尚不存在而失败。

- [ ] **Step 4: 实现最小探针模块**

实现上述两个公开接口；文件写入使用进程内锁和 JSON Lines，一次事件一行。关闭路径必须在任何 tensor 运算前返回。

- [ ] **Step 5: 运行单测**

```bash
PYTHONPATH=$PWD pytest -q tests/ut/_310p/test_dflash_prefix_probe.py
```

Expected: PASS。

### Task 3: 接入 P0～P3 边界

**Files:**
- Modify: `vllm_ascend/worker/model_runner_v1.py`
- Modify: `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py`
- Modify: `vllm_ascend/_310p/sample/rejection_sampler.py`
- Test: `tests/ut/_310p/test_dflash_prefix_probe.py`

**Interfaces:**
- Consumes: Task 2 的 `record_prefix_probe(...)`。
- Produces: `P0_TARGET_TO_DRAFT`、`P1_DRAFT_CONTEXT`、`P2_DRAFT_OUTPUT`、`P3_REJECTION` 四类事件。

- [ ] **Step 1: 增加调用点测试**

通过 mock 验证每个调用点传入 stage 名、step、request key 和规定字段；探针关闭时原返回值和原 tensor 地址不变。

- [ ] **Step 2: 运行测试并确认新增断言失败**

```bash
PYTHONPATH=$PWD pytest -q tests/ut/_310p/test_dflash_prefix_probe.py
```

Expected: 缺少 P0～P3 调用而失败。

- [ ] **Step 3: 接入四个边界**

只增加旁路记录，不修改 token、position、slot mapping、logits、accepted count 或返回值。P1 的 KV 摘要只索引 active slots，过滤 `PADDING_SLOT_ID=-1`。

- [ ] **Step 4: 运行定向回归**

```bash
PYTHONPATH=$PWD pytest -q \
  tests/ut/_310p/test_dflash_prefix_probe.py \
  tests/ut/_310p/spec_decode/test_dflash_proposer_310.py \
  tests/ut/_310p/sample/test_rejection_sampler_310.py \
  tests/ut/_310p/test_model_runner_310p.py
```

Expected: 全部通过。

- [ ] **Step 5: 静态检查**

```bash
ruff check \
  vllm_ascend/_310p/dflash_prefix_probe.py \
  vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/_310p/spec_decode/dflash_proposer_310.py \
  vllm_ascend/_310p/sample/rejection_sampler.py \
  tests/ut/_310p/test_dflash_prefix_probe.py
```

Expected: 无 Ruff 错误。

### Task 4: NPU 实验前安全门

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes: 远程容器和 NPU 进程状态。
- Produces: “允许启动”或“等待另一 agent”的二元结果。

- [ ] **Step 1: 检查容器内服务**

```bash
docker top scc_dflash_dev -eo pid,lstart,args | grep -E 'vllm serve|VLLM::EngineCore|VLLM::Worker' || true
```

- [ ] **Step 2: 检查设备进程**

```bash
npu-smi info
```

- [ ] **Step 3: 执行硬门禁**

只要任一检查显示另一个推理/profile 进程，停止本任务并等待。不得执行 `docker stop`、`pkill`、`kill` 或重启容器。

### Task 5: 运行 Prefix-off 对照

**Files:**
- Write artifacts only: `/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_off`

**Interfaces:**
- Consumes: 探针 worktree、模型权重、端口 6766。
- Produces: Prefix-off AISBench 预测和 rank trace。

- [ ] **Step 1: 清理本实验自己的旧输出目录**

如果目录已存在，不递归删除；改用带当前时间戳的新子目录，确保历史证据可恢复。

- [ ] **Step 2: 在独立终端启动诊断服务**

```bash
docker exec -it scc_dflash_dev bash -lc '
  cd /home/shichuchao/scc_dflash/prefix_probe_wt &&
  export PYTHONPATH=$PWD &&
  export VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE=1 &&
  export VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_DIR=/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_off &&
  export VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_MAX_STEPS=320 &&
  vllm serve /home/models/Qwen3.6-35B-A3B-w8a8 \
    --served-model-name Qwen3.6-35B-A3B-w8a8 \
    --host 127.0.0.1 --port 6766 \
    --dtype float16 --quantization ascend \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 10 --max-num-batched-tokens 2048 --max-model-len 8192 \
    --trust-remote-code --no-enable-prefix-caching \
    --enable-chunked-prefill --no-async-scheduling \
    --safetensors-load-strategy eager --enforce-eager \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false,"fuse_norm_quant":false}}' \
    --speculative-config '{"method":"dflash","model":"/home/models/Qwen3.6-35B-A3B-DFlash","draft_tensor_parallel_size":2,"num_speculative_tokens":15}'
'
```

- [ ] **Step 3: 健康检查**

```bash
curl -f http://127.0.0.1:6766/health
```

Expected: HTTP 200。

- [ ] **Step 4: 运行前 12 个 GSM8K 请求**

将 AISBench 模型配置临时指向 `localhost:6766`，保持 `temperature=0`、`enable_thinking=False`、`max_out_len=256`、`ignore_eos=False`，并使用独立 work-dir：

```bash
docker exec scc_aisbench_client bash -lc '
  cd /home/shichuchao/scc_dflash/benchmark/ais_bench/datasets &&
  ais_bench \
    --models vllm_api_general_chat \
    --datasets gsm8k_gen_4_shot_cot_str \
    --num-prompts 12 \
    --work-dir /home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_off/aisbench_outputs
'
```

执行前必须确认该 AISBench 配置实际端口为 6766；不得改写另一个 agent 正在使用的共享配置文件。实现时优先复制配置到实验目录并以实验配置名加载。

- [ ] **Step 5: 用终端 Ctrl-C 停止本实验前台服务**

只停止 Task 5 自己启动的前台会话，等待 EngineCore 和 TP Worker 全部退出后再继续。

### Task 6: 运行 Prefix-on 异常组

**Files:**
- Write artifacts only: `/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_on`

**Interfaces:**
- Consumes: 与 Task 5 完全相同的模型、请求和探针。
- Produces: Prefix-on AISBench 预测和 rank trace。

- [ ] **Step 1: 执行 Prefix-on 安全门**

```bash
docker top scc_dflash_dev -eo pid,lstart,args | grep -E 'vllm serve|VLLM::EngineCore|VLLM::Worker' || true
npu-smi info
```

Expected: 不存在任何其他 vLLM、EngineCore、Worker 或占用目标 NPU 的进程。否则停止本任务并等待，不执行 kill/stop。

- [ ] **Step 2: 启动 Prefix-on 服务**

```bash
docker exec -it scc_dflash_dev bash -lc '
  cd /home/shichuchao/scc_dflash/prefix_probe_wt &&
  export PYTHONPATH=$PWD &&
  export VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE=1 &&
  export VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_DIR=/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_on &&
  export VLLM_ASCEND_310P_DFLASH_PREFIX_PROBE_MAX_STEPS=320 &&
  vllm serve /home/models/Qwen3.6-35B-A3B-w8a8 \
    --served-model-name Qwen3.6-35B-A3B-w8a8 \
    --host 127.0.0.1 --port 6766 \
    --dtype float16 --quantization ascend \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 10 --max-num-batched-tokens 2048 --max-model-len 8192 \
    --trust-remote-code --enable-prefix-caching \
    --enable-chunked-prefill --no-async-scheduling \
    --safetensors-load-strategy eager --enforce-eager \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false,"fuse_norm_quant":false}}' \
    --speculative-config '{"method":"dflash","model":"/home/models/Qwen3.6-35B-A3B-DFlash","draft_tensor_parallel_size":2,"num_speculative_tokens":15}'
'
```

Expected: 命令只包含 `--enable-prefix-caching`，不包含 `--no-enable-prefix-caching`；健康检查 `curl -f http://127.0.0.1:6766/health` 返回成功。

- [ ] **Step 3: 运行相同 12 题**

```bash
docker exec scc_aisbench_client bash -lc '
  cd /home/shichuchao/scc_dflash/benchmark/ais_bench/datasets &&
  ais_bench \
    --models vllm_api_general_chat \
    --datasets gsm8k_gen_4_shot_cot_str \
    --num-prompts 12 \
    --work-dir /home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_on/aisbench_outputs
'
```

执行前必须确认实验 AISBench 模型配置实际端口为 6766，且未改写另一个 agent 使用的共享配置。

- [ ] **Step 4: 确认异常确实复现**

至少满足以下之一才进入比较：

- 精度低于 Prefix-off；
- 任一 prediction 包含 `The number of apples in the fourth basket`；
- 任一 prediction 在正确答案后追加第四个 fruit-basket few-shot 尾段。

如果 12 题未复现，扩大到原始 50 题，但不改变探针和服务配置。

- [ ] **Step 5: 停止本实验服务并确认设备释放**

只通过 Task 6 前台会话 Ctrl-C 停止，随后用 `docker top` 和 `npu-smi info` 验证本实验进程已退出。

### Task 7: 对齐证据并生成结论

**Files:**
- Create: `tools/compare_dflash_prefix_probe.py`
- Write: `/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/comparison.json`
- Write: `/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/comparison.md`

**Interfaces:**
- Consumes: Prefix-off/on 的 rank trace 和 AISBench predictions。
- Produces: 第一个分歧 stage、request、step、rank 及判定结论。

- [ ] **Step 1: 写比较器单测**

构造四组小型 JSONL fixture，覆盖 P0、P1、P3 target 和 P3 sampler 四种首分歧，验证比较器返回正确分类。

- [ ] **Step 2: 实现比较器**

按 `request_key + step + stage + rank` 对齐；先比较整型元数据，再比较浮点摘要。报告第一个结构性差异，不把普通 FP16 微小误差自动分类为根因。

- [ ] **Step 3: 运行真实比较**

```bash
cd /home/shichuchao/scc_dflash/prefix_probe_wt
PYTHONPATH=$PWD python tools/compare_dflash_prefix_probe.py \
  --prefix-off /home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_off \
  --prefix-on /home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/prefix_on \
  --output /home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902
```

- [ ] **Step 4: 输出结论报告**

报告必须包含：

```text
异常是否复现：
首个分歧 request/step/rank：
首个分歧 stage：
Prefix hit token 数：
Context slot/block/KV 是否一致：
Draft token 是否首先分歧：
Target argmax 是否正确：
Sampler 是否遵守首 mismatch：
代码责任范围：
是否需要第二阶段 GDN probe：
```

## 5. 第二阶段 P4：GDN Prefix Boundary State 探针

**执行状态：已完成。** R/U/A 最小实验已将错误定位到“640-token 调度切分导致 1280-token Mamba checkpoint 未物化”。完整证据和修复边界见 [P4 验证结果](./2026-09-02-310p-dflash-prefix-probe-result.md)。不再扩大探针或运行 12/50 题。

第一阶段已经证明：

- Draft prefix context KV 未复用，但 rejection sampler 能严格阻断首个 mismatch；
- Prefix hit 请求在第一次 rejection 之前，部分请求的首个 target argmax 已与 Prefix-off 分歧；
- 因而 P4 不先假设 accepted-token rollback，而是验证更早的契约：**Prefix hit 恢复的 GDN conv/SSM state，是否等于完整计算公共前缀后得到的状态。**

### 5.1 单一主假设

运行时读取 `B = mamba_spec.block_size`。现有实验预计 `B=1280`，但探针和请求生成器不得硬编码。

当前高优先级假设是：310P GDN prefill 跨过可缓存的 Mamba block 边界时，只把整段 prefill 的最终 recurrent state 写到当前 state block，没有把边界状态 `S_B` 物化到公共前缀 block。原 warmup 分块为 `640 + 820`；若 Block 0 保留 `S_640`，后续请求却把它当作 `S_1280` 恢复，就会在第一次 target verify 前产生错误 logits。

### 5.2 最小 R/U/A 请求

由同一 tokenizer 直接生成 token IDs：

```text
P = 恰好 B 个公共前缀 token
W = P + tail_A
H = P + tail_B
tail_A != tail_B
```

所有请求统一 `temperature=0`、`max_tokens=1`、`ignore_eos=false`，服务统一 Eager、async scheduling off、DFlash K=15、TP2。只运行：

| 运行 | Prefix Cache | 请求顺序 | 目的 |
|---|---|---|---|
| R / Reference | 关闭 | `P`、`H` | 生成标准 `S_B`，记录 H 的正确首个 target token |
| U / Unaligned | 开启 | `W`、`H` | 复现跨 B 边界 warmup 后的异常恢复 |
| A / Aligned | 开启 | `P`、`H` | 让 warmup 恰好结束在 B 边界，做因果干预 |

H 必须满足 `num_computed_tokens == B`；否则该运行无效并立即停止分析，不扩大请求数量。

### 5.3 P4 stage 与插点

#### `P4_STATE_PLAN`

在 `vllm_ascend/patch/worker/patch_mamba_utils.py::preprocess_mamba` 中记录：

- request key、step、`num_computed_tokens`、`num_scheduled_tokens`；
- `mamba_spec.block_size`、`num_speculative_blocks`；
- `prev_state_idx`、`curr_state_idx`、`accept_token_bias`；
- 每个 Mamba cache group 的 logical-to-physical block IDs；
- Prefix hit token 数和 copy 是否计划执行。

#### `P4_STATE_WRITE`

在 `vllm_ascend/_310p/ops/fla/gdn_310.py` 的 non-spec prefill 路径中，分别在 conv state 更新和 `ssm_state[...] = last_recurrent_state` 后记录：

- layer name、state kind（`conv` / `ssm`）；
- `has_initial_state`、`non_spec_state_indices_tensor`；
- 实际写入的 physical block；
- state 摘要。

#### `P4_STATE_COPY`

在 `_collect_mamba_copy_meta_torch` 和 `_do_mamba_copy_block_torch` 中记录 `state_copy_func` 最终产生的真实 tensor view，而不是只记录逻辑 Block：

- layer name、state kind；
- source/destination logical block 和 physical block；
- copy element offset、`num_elements`；
- `src_before`、`dst_before`、`dst_after`、`src_after` 摘要。

硬性判据：`dst_after == src_before` 且 `src_after == src_before`。

#### `P4_STATE_CONSUME`

在第一个 GDN forward 消费初始状态前记录：

- `has_initial_state`；
- `non_spec_state_indices_tensor` / `spec_state_indices_tensor`；
- conv cache indices；
- 算子实际读取 physical block 的 conv/SSM 摘要。

要求 copied destination、metadata 指向 block 和算子消费 block 三者一致。

#### `P4_STATE_IMMUTABILITY`

记录 warmup 公共前缀 block 在三个时刻的摘要：prefill 完成后、warmup speculative postprocess 后、H preprocess 开始前。只有状态最初正确但随后变化时，才进入 `_postprocess_mamba_align_gpu_cpu_fallback` 的 source/destination 细探针。

P4 继续复用 P2 的 rejection 前 target argmax，不新增 sampler 逻辑。

### 5.4 State 摘要与资源约束

每个 rank、每个 GDN layer、每种 state 记录：shape、dtype、finite count、FP32 sum、abs sum、square sum、max abs、固定位置采样值和 raw-byte SHA256。物理 Block ID 可以不同，但同一 token 边界的状态内容必须数值等价。

- 探针必须环境变量控制，关闭时在任何 tensor 运算、同步或文件写入前返回。
- 不创建逐层 `torch.npu.Event`；每个 stage 最多统一 stream synchronize 一次。
- 默认只记录两个请求和前两个 engine step；不 dump 模型权重或全量 hidden states。
- TP2 的 rank0/rank1 都必须记录；任一 rank 首先分歧即为有效故障边界。

### 5.5 判定矩阵

| 首个失败的契约 | 根因范围 |
|---|---|
| H 与 warmup 的公共前缀 physical block 不一致 | Prefix cache group / Block table 映射 |
| warmup Prefix Block 在 prefill 后就不等于 Reference `S_B` | GDN Prefix 边界 state 未物化或写错 |
| warmup 后正确、postprocess 后改变 | 共享 Prefix Block 被 decode/postprocess 原地污染 |
| source 正确、copy 后 destination 不一致 | 310P tensor-copy fallback |
| destination 正确、GDN metadata 消费其他 Block | GDN metadata/index 映射 |
| GDN 输入状态正确，但第一层输出首先分歧 | 310P GDN state layout / initial-state 算子语义 |
| A(H) target argmax 等于 R(H)，而 U(H) 不等于 R(H) | 证明跨 Prefix 边界 state 未物化；停止扩大探针 |

### 5.6 停止条件与后续修复边界

一旦 R/U/A 能把首个错误映射到上表单一边界，立即停止 P4，不运行 12/50 题，不修改产品行为。

已确认“边界 state 未物化”。正式 Target 修复必须保证每个未来可复用的 Mamba block 边界都有有效 conv/SSM checkpoint。优先方案是在 scheduler 中按 `scheduler_block_size`/Mamba block 截断跨边界的 prefill，使 GDN 自然在边界写出 checkpoint；备选方案才是在单次跨边界 forward 中显式输出并保存中间状态。仅把一个字段由 640 改成 1280、却仍允许一次 forward 跨过边界，并不能满足该契约。

Draft prefix context KV 未复用已由 P0/P1 独立确认，不在 P4 重复探测；完整产品修复仍需让 DFlash context KV 参与 Prefix Cache，或建立等价的前缀 KV 生命周期和 block 映射契约。

## 6. 停止条件

满足任一条件立即停止，不继续扩大实验：

- 另一个 agent 的服务或 NPU 进程仍在运行；
- A/B 除 prefix flag 外存在其他配置差异；
- 探针关闭时单测发现行为或 tensor 地址变化；
- Prefix-on 未复现原始串入且 50 题仍正常；
- 已找到第一个结构性分歧并能映射到单一代码边界。

最终结论必须区分“已验证根因”“证据支持的候选”和“已排除项”，不得仅凭输出文本推断内部状态。

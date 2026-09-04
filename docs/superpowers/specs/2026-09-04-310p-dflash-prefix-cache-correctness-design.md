# 310P DFlash 前缀缓存正确性修复设计

- 日期：2026-09-04
- 状态：已实现并完成远程端到端验证（2026-09-04）
- 目标分支：`dev_v24/feat-310p-dflash`
- 目标模型：Qwen3.6-35B-A3B-w8a8 + DFlash
- 目标部署：310P、TP2、并发 10、Prefix Cache 开启、Async Scheduling 开启
- 关联分析：`docs/superpowers/plans/2026-09-02-310p-dflash-prefix-probe-result.md`

## 1. 问题陈述

当前 310P DFlash 为降低高并发下的 KV Cache 内存压力，把 draft 模型的逻辑块大小保持为 640 token；target Attention 和 Mamba 状态仍以 1280 token 为完整检查点。开启 Prefix Cache 后，调度器、KV Cache 协调器和 DFlash worker 对“已经计算了多少 token”“每个模型实际拥有多少 KV 块”“同一个物理页如何映射到不同 kernel block size”的理解不完全一致。

已确认的直接错误包括：

1. 长度 1460 的请求在命中 640 token 前缀后，可能被调度成 `640 + 820`，跨过 1280 检查点却没有在 1280 停下，导致 Mamba 状态与 token 位置不一致。
2. DFlash 复用了 EAGLE 的 KV Cache 特例：读命中时丢弃最后一块、写分配时额外窥视下一块。DFlash 的验证算法需要 speculative lookahead，但它的 KV 所有权语义不是 EAGLE；复用该特例会让 target、draft 与 Mamba 对已缓存前缀的认知分叉。
3. 混合 KV 布局最终截断只处理第一个 FullAttention group；存在多个 FullAttention group 时，返回块数与命中 token 数可能不匹配。
4. 当前缺少覆盖不变量检查，错误的块表可以继续流入 worker，最终表现为精度下降，而不是在根因位置快速失败。
5. 对于 draft 各层 kernel block size 混用 64/128 的模型，当前只替换 per-layer `slot_mapping`，没有同步替换 `block_tables`；非连续物理页下读写都会映射到错误位置。当前 Qwen3.6 六层均为 128，不触发该问题，但修复不能继续保留这个已知隐患。
6. Async Scheduling 路径可能在 accepted-token 元数据生产事件完成前消费该元数据。该问题需要用独立 NPU 小实验确认；若成立，使用设备侧 `wait_event` 建立依赖，不使用主机同步。

`ignore_eos` 只决定生成遇到 EOS 后是否停止，不会改变 KV 块命中、检查点切分或 accepted-token 回滚本身，因此不是上述前缀缓存根因。

## 2. 目标与非目标

### 2.1 目标

1. 保留 draft 640 token 逻辑块的内存优化。
2. 保证 DFlash + Prefix Cache 下，target Attention、draft Attention 与 Mamba 状态覆盖同一段已提交 token。
3. 保证调度在所有绝对 1280 检查点停顿，而不是只修复长度 1460 这一组输入。
4. 保证 speculative rejection、EOS、请求结束/释放/复用，以及异步调度下状态提交顺序正确。
5. 保证当前全 128 kernel block size 快速路径不产生额外转换；混合 64/128 时生成相匹配的 block table 与 slot mapping。
6. 用单元测试、小型 NPU 实验和固定 50 题 AISBench 精度对比证明修复有效。

### 2.2 非目标

1. 不改变 DFlash 的 proposer、采样或 rejection 算法。
2. 不取消 Prefix Cache、Async Scheduling 或 draft 640 内存优化。
3. 不以全局 `synchronize()` 作为长期修复。
4. 不调整模型权重、量化配置、随机采样参数来掩盖精度问题。
5. 本轮不新增 Context Parallel 能力；若发现 DFlash 混合映射与 CP 组合尚无正确实现，则在入口显式拒绝该组合，避免静默错误。

## 3. 必须成立的不变量

### 3.1 绝对检查点不变量

设请求开始本轮调度前已经计算 `C` 个 token，本轮计划新增 `N` 个 token，完整 Mamba 检查点间隔为 `M=1280`。

若 `C < kM < C+N`，本轮必须截断到最近的下一个绝对检查点：

```text
N' = kM - C
```

例如请求总长 1460、前缀命中 640 时，第一轮只能调度 640 个新 token 到达 1280，第二轮再调度 180：

```text
1280 + 180
```

不能调度成 `640 + 820`。这里的 1280 是 token 序列中的绝对位置，不是“本轮再取一个 1280 大小的块”。

### 3.2 覆盖不变量

若协调器报告命中长度为 `H`，每个 KV group 返回的逻辑块必须恰好覆盖 `[0, H)`：

```text
len(returned_blocks[group]) == H / effective_block_size[group]
```

命中长度只能落在该 group 可表示的完整块边界上。任何 group 覆盖不足、覆盖过量或不可整除都必须在协调器处报错，不允许继续执行。

### 3.3 所有权与内容不变量

1. Prefix Cache 命中后，命中范围内 target/draft 的物理块身份不应被无故丢弃或重分配。
2. Rejection 只能撤销未提交 speculative token，不能覆盖或释放已提交前缀。
3. EOS、`ignore_eos` 和普通 rejection 最终都必须通过同一“已接受 token 数”更新状态；结束策略不能绕过 KV 状态提交。
4. 请求完成并释放后，复用该物理页的新请求不能观察到旧请求的逻辑映射。

### 3.4 异步顺序不变量

读取 `num_accepted_tokens`、执行 Mamba postprocess 或根据接受数更新 block table 之前，消费 stream 必须等待生产该元数据的 event。依赖通过 NPU stream `wait_event` 建立，避免 CPU 阻塞和全设备同步。

### 3.5 物理页映射不变量

同一个 640-token 物理逻辑页可被不同 kernel block size 展开，但所有 per-layer 元数据必须成对转换：

```text
(block_tables, slot_mapping)
```

不能只替换 `slot_mapping`。对于非连续物理页，例如物理页 `[10, 4]`：

```text
64-token table:  [100..109, 40..49]
128-token table: [50..54, 20..24]
```

token 640 在 128-token 布局中必须落到 block 20、slot 2560，不能把 64-token 表按 128 再解释为 block 105、slot 13440。

## 4. 设计

### 4.1 将“DFlash 是否使用 lookahead”与“是否使用 EAGLE KV 特例”解耦

文件：

- `vllm_ascend/patch/platform/dflash_kv_context.py`
- `vllm_ascend/patch/platform/patch_mamba_scheduler_310.py`
- `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`

不修改 vLLM。310P Scheduler 初始化补丁在构造 DFlash Scheduler 时设置一个仅对当前调用链有效的 `ContextVar`；原始 Scheduler 继续保留 `self.use_eagle=True`，确保 DFlash 仍能执行 K+1 lookahead、proposer 与 rejection verification。Ascend coordinator 工厂读取该上下文，只对 KV 语义做转换：

```text
use_eagle_for_kv_cache = use_eagle and not dflash_scheduler_init_context
```

上下文在 `finally` 中通过 `ContextVar.reset(token)` 恢复，避免初始化异常、并发线程或异步任务污染后续 Scheduler。

也就是说：

- EAGLE：继续使用 KV 命中丢最后一块和写侧额外 peek 的现有规则。
- DFlash：保留 speculative 调度，但使用普通 KV Cache 的命中与分配规则。

该改动解决“为了 speculative lookahead，却连带继承 EAGLE KV 所有权规则”的根本耦合。

### 4.2 按绝对 Mamba 检查点切分调度

文件：`vllm_ascend/patch/platform/patch_mamba_scheduler_310.py`

现有候选逻辑只在 scheduler block size 大于 cache block size 时启用；这会漏掉未来两者均为 1280、但仍需要检查点切分的布局。启用条件改为由能力决定：

1. Prefix Cache 开启；
2. speculative method 为 DFlash；
3. 当前执行路径包含 Mamba 对齐需求；
4. Mamba checkpoint size 为正。

切分公式使用绝对位置：

```text
end = num_computed_tokens + num_new_tokens
next_checkpoint = ceil((num_computed_tokens + 1) / checkpoint_size) * checkpoint_size
if num_computed_tokens < next_checkpoint < end:
    num_new_tokens = next_checkpoint - num_computed_tokens
```

边界恰好等于检查点时不额外生成 0-token 轮次。连续跨越多个检查点时，每轮只到最近的下一个检查点，后续轮次继续推进。

### 4.3 让混合 KV 协调器在固定点上返回一致前缀

文件：`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`

协调过程继续寻找所有 KV groups 都可共同表示的最大前缀，但最终收敛和截断必须满足：

1. 遍历全部 FullAttention groups，而不是只处理第一个 group。
2. 每个 group 按自己的 effective block size 截断到统一命中长度。
3. 截断完成后执行覆盖不变量检查；错误信息包含 request id、group id、命中 token、期望块数和实际块数。
4. 重复实现的两个返回路径共享同一个 helper，避免以后只修一处。
5. 恢复 `num_uncached_common_prefix_tokens` 的准确计算，供 admission/Marconi 使用；它属于调度压力控制，不改变已命中 KV 的正确性。

该层只返回“所有 group 都真实拥有”的共同前缀，不负责 speculative lookahead，也不伪造不存在的 draft block。

### 4.4 为不同 kernel block size 生成配套的 per-layer 元数据

文件：

- `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py`
- `vllm_ascend/_310p/model_runner_310p.py`
- 必要时复用 `vllm_ascend/_310p/block_table.py` 中已有缓冲区能力

采用 310P proposer 层的最小转换方案，避免改写通用多 group BlockTable：

1. 以 scheduler 提供的逻辑表、源 kernel block size 和 640-token 物理逻辑页大小恢复物理 page id。
2. 校验同一物理页在源表中是连续编码；不满足时立即报错。
3. 按每种目标 kernel block size 重新展开物理 page id，并写入持久复用的 device buffer。
4. 构造 per-layer attention metadata 时，同时替换 `block_table_tensor` 和 `slot_mapping`。
5. 相同 block size 的层共享同一份转换结果；当前 Qwen3.6 全 128 布局走零转换快速路径。
6. context KV update 同样消费对应层的 slot mapping，保证读路径和写路径一致。

如果启用 Context Parallel 后无法从本地 rank 的交错表无歧义恢复物理 page id，则入口显式拒绝 DFlash + CP + 混合 kernel block size 组合；不能沿用当前错误映射。

### 4.5 用小型 NPU 实验决定异步依赖补丁

文件：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/_310p/model_runner_310p.py`

先运行不占用现有服务端口、只绑定空闲 NPU 的小实验：在生产 stream 上延迟写 accepted-token 元数据并 record event，在消费 stream 上立即读取，分别比较无等待、`wait_event` 和 host synchronize 三种结果。

决策规则固定如下：

- 若无等待路径可稳定观察到旧值，而 `wait_event` 路径稳定正确：在 310P async fast path 消费 accepted-token 元数据前加入当前消费 stream 的 `wait_event(num_accepted_tokens_event)`。
- 若框架已有等价 stream dependency，探针和 stream trace 证明消费必然晚于 event：不重复加等待，只补回归测试和注释。

无论哪种结果，禁止用 `global_stream.synchronize()`、`event.synchronize()` 或 device synchronize 作为最终实现，因为它会破坏异步调度吞吐。

## 5. 预期代码改动

| 仓库 | 文件 | 改动 |
| --- | --- | --- |
| vLLM Ascend | `vllm_ascend/patch/platform/dflash_kv_context.py` | 以调用上下文携带 DFlash 身份并解析有效 EAGLE KV 策略 |
| vLLM Ascend | `vllm_ascend/patch/platform/patch_mamba_scheduler_310.py` | 标记 DFlash 初始化调用链；绝对 1280 检查点切分，覆盖相等 block size |
| vLLM Ascend | `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py` | 全 FullAttention group 截断、覆盖断言、common-prefix 统计 |
| vLLM Ascend | `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py` | per-size block table 与 slot mapping 成对下发 |
| vLLM Ascend | `vllm_ascend/_310p/model_runner_310p.py` | per-layer 映射消费；按探针结果建立 async event 依赖 |
| vLLM Ascend | 相关 310P 单测 | 检查点、混合 groups、映射、异步状态回归 |

生产代码只做上述局部语义修复，不重写通用 scheduler 分块算法，不改变 draft 640 配置。

## 6. 测试设计

### 6.1 单元测试：先红后绿

#### 调度检查点

至少参数化以下 `(computed, requested_new, expected_new)`：

```text
(0,    640,  640)   # 未跨检查点
(640,  820,  640)   # 1460 场景：先到 1280
(1280, 180,  180)   # 已在检查点上
(1200, 200,   80)   # 跨过 1280
(1280, 1400, 1280)  # 跨过 2560
(2560, 100,  100)   # 新检查点之后继续
```

同一测试在 scheduler/cache block size 为 `1280/640` 和 `1280/1280` 两种布局下均执行。

#### EAGLE 与 DFlash KV 语义

1. Prefix hit=1280 时，DFlash 不丢 target 最后一块，也不在写侧多 peek 一块。
2. EAGLE 原有 drop/peek 行为保持不变。
3. DFlash speculative lookahead 数量保持原值，防止修 KV 语义时误关 speculative decoding。

#### 多 FullAttention groups

构造 `Mamba(1280) + FullAttention-A(640) + FullAttention-B(640)`：

1. 共同命中 1280 时分别返回 `1/2/2` 个块。
2. 从候选 2560 收敛到 1280 时，两个 FullAttention groups 都截断为 2 块。
3. 任意 group 少一块或多一块时，覆盖断言必须失败并打印 group 信息。

#### 混合 64/128 映射

1. 使用非连续物理 page `[10, 4]`，验证 64 与 128 两组 block table 的确切值。
2. 验证 token 0、639、640、1279 的 block/slot 边界。
3. 验证 query attention metadata 同时替换 block table 和 slot mapping。
4. 验证 context KV update 使用相同 per-layer slot mapping。
5. 全 128 布局验证不转换且原 buffer 身份可复用。

#### 生命周期

覆盖 Prefix hit 后的全接受、部分拒绝、全拒绝、EOS、`ignore_eos`、finish/free/reuse，并验证已提交前缀的 block id 与内容摘要不变。

### 6.2 小型 NPU 验证

在远程 `scc_dflash_dev` 容器中先检查 NPU 进程和设备占用，只使用空闲设备、独立端口和独立日志目录，不停止或修改其他 agent 的服务。

1. 执行 async event 可见性探针，按第 4.5 节决策规则落地或排除补丁。
2. 启动 TP2、小 `max-num-seqs` 的临时服务。
3. 连续发送两条共享长前缀、后缀不同的请求，第二条必须产生 Prefix Cache hit。
4. 覆盖长度位于 640、1280 附近及跨多个检查点的请求。
5. 比较 async off/on，检查输出、命中长度、各 group 块数、accepted-token 序列和 Mamba 状态推进位置。

### 6.3 AISBench 端到端精度

固定相同模型、数据集、50 条 prompt、chat template、generation 参数和随机种子，依次运行：

1. 无 DFlash 基线；
2. DFlash + Prefix Cache 关闭的控制组；
3. 修复后 DFlash + Prefix Cache 开启，Async Scheduling 开启的目标组。

命令核心参数：

```text
ais_bench --models vllm_api_general_chat \
  --datasets gsm8k_gen_4_shot_cot_str \
  --num-prompts 50
```

验收同时看聚合精度和逐题差异：

- 目标组精度不得低于无 DFlash 基线，也不得低于 Prefix Cache 关闭控制组。
- 对每个“基线正确、目标组错误”的新增回归逐条保存 prompt、答案和生成输出；存在未解释的新增回归时，不判定修复完成。
- 服务日志中不得出现覆盖断言、NPU 异步错误或 KV 生命周期错误。

## 7. 实施顺序

1. 为绝对检查点、DFlash/EAGLE KV 解耦、多 FullAttention groups 和 64/128 映射编写失败单测。
2. 实现 scheduler 与 KV coordinator 的最小修复，使对应单测转绿。
3. 实现 per-size block table/slot mapping 成对转换，使映射测试转绿。
4. 在空闲 NPU 上执行 async 探针，并按固定规则实现或排除 event 依赖补丁。
5. 运行相关单元测试和静态检查。
6. 将改动同步到远程容器的隔离工作副本，执行两请求 Prefix Cache 小实验。
7. 用独立端口依次运行三组 AISBench，生成逐题对比。
8. 将代码改动、测试证据、精度结果和剩余限制更新到结果文档。

## 8. 风险与控制

1. **误伤 EAGLE**：通过 EAGLE 保持原 drop/peek 的对照单测控制。
2. **降低吞吐**：保留 draft 640；全 128 映射走快速路径；异步依赖只允许 device-side `wait_event`。
3. **只修特例**：检查点和映射测试使用参数化边界及非连续物理页，不依赖 1460 单样例。
4. **多仓库版本不一致**：分别记录 vLLM 与 vLLM Ascend commit，并在远程启动日志打印实际 import 路径。
5. **影响其他任务**：远程只使用空闲 NPU、独立端口、独立日志目录；不 kill、不重启、不覆盖其他 agent 的进程或容器文件。
6. **错误被静默传播**：协调器覆盖、物理页恢复和不支持的 CP 组合均 fail fast，错误信息包含可定位上下文。

## 9. 完成标准

以下条件全部满足后才可声明修复完成：

1. draft 逻辑块仍为 640，目标 `max-num-seqs=10` 配置可启动。
2. W1460 Prefix hit=640 稳定调度为 `1280 + 180`。
3. DFlash 不再继承 EAGLE KV drop/peek，但 speculative lookahead 保持。
4. 所有 FullAttention groups 均满足共同前缀覆盖不变量。
5. 全 128 和混合 64/128 的 block table/slot mapping 均通过边界测试。
6. Async Scheduling 的 accepted-token 消费存在经实验验证的正确 stream 顺序。
7. Prefix hit 下全接受、拒绝、EOS、`ignore_eos`、finish/free/reuse 无状态污染。
8. 相关单元测试、远程 Prefix Cache 小实验全部通过。
9. 50 题 AISBench 目标组无精度下降，且不存在未解释的逐题新增回归。

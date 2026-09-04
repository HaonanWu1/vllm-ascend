# 310P DFlash Prefix Cache 探针验证结果

## 最终结论

P4 已定位并验证了原始精度回归的一条直接根因，不需要继续扩大同一组 P4 探针。但在保留 draft 640 block 并对混合 KV 布局做整体审计后，又确认了 coordinator 的一处独立正确性根因、一处多 Full Attention group 的截断缺陷，以及一处静态上已经缺少跨流依赖、仍需 NPU 动态确认影响的异步风险。此前记录的“draft 只写 tail”是当前故障的真实现象，但代码复核表明它主要是 coordinator 没有复用 draft prefix blocks 的下游症状，尚不能列为独立根因。因而“两轮调度修复”只是必要修复，不是完整修复。

2026-09-03 第二轮遗漏审计没有发现新的、比 I1/I2 更靠前的当前同步路径根因，但补全了两个重要边界：第一，DFlash 被误套 EAGLE 语义不仅影响 cache lookup，还会让 draft manager 在 1280 公共边界之外额外发布一个 640 block；第二，当前候选 scheduler 补丁只在 `scheduler_block_size > cache_config.block_size` 时生效，若以后关闭 640 优化、三个 group 都为 1280，DFlash 仍可能因通用 EAGLE prune 一次跨过 1280 checkpoint。后者不影响当前必须保留 640 的目标配置，但正式修复不能依赖“刚好存在 640”才能正确。

310P DFlash 为降低统一 KV page 浪费，把 draft attention 的逻辑 block 从 1280 缩成 640。EngineCore 建完异构 KV cache groups 后，又把全局 `cache_config.block_size` 改成所有 group 的最小值 640；但 Mamba state block 和 scheduler 的公共对齐粒度仍然是 1280。调度器的 `_mamba_block_aligned_split` 错误使用了前者，最终把首个 warmup prompt 切成 `640 + 820`。

这与 `preprocess_mamba` 的 1280-token recurrent state 语义冲突：

1. 第一段 640 token 把 `S640` 写到 Mamba logical block 0。
2. 第二段从 640 跨过 1280 边界并结束于 1460。`preprocess_mamba` 先把 block 0 的 `S640` 复制到 logical block 1，GDN 再把最终 `S1460` 写到 block 1。
3. logical block 0 没有被更新为本应代表公共前缀的 `S1280`，仍是 `S640`。
4. 后续请求命中 1280-token prefix 后，从 logical block 0 恢复状态，因此把 `S640` 当成 `S1280` 使用。第一个 target token 在 rejection sampler 之前就已错误。

P4 对齐干预把 warmup 改成恰好 1280 token 后，同一 block 0 会经历 `S640 -> S1280` 更新；随后 prefix-hit 请求恢复的状态和输出都恢复到 Prefix-off reference。这构成了“错误 Mamba checkpoint 会直接改变 target 输出”的因果证据，但不能证明其他 KV group 的 Prefix Cache 元数据和内容已经正确。

因此问题不是：

- `ignore_eos`；实验明确为 `false`。
- rejection sampler 放行错误 draft；已检查的 row 全部遵守首 mismatch 规则。
- 310P tensor-copy fallback；P4 的 180 个 conv/SSM copy tensor 全部满足 copy 后 `src == dst`。
- logical/physical block 映射选错；算子实际消费的 physical block 与 copy 目标一致。

## 触发链路

触发这个回归的代码链可以精确到以下四步：

1. 提交 `f62666eb2 fix(310p): reduce DFlash KV cache page waste`：
   - `NPUModelRunner310.get_kv_cache_spec()` 调用 `_resize_dflash_draft_kv_cache_specs()`；
   - DFlash draft attention block 从 1280 改为 640；
   - `patch_kv_cache_utils.py` 允许 `(scheduler_block_size, hash_block_size) = (1280, 640)`。
2. vLLM `EngineCore._initialize_kv_caches()`：
   - 执行 `cache_config.block_size = min(group.block_size)`；
   - 异构 group `[1280, 640, 1280, ...]` 使该字段变成 640。
3. vLLM `Scheduler`：
   - 构造参数 `self.block_size` 是 LCM，即正确的 scheduler alignment 1280；
   - 但 `_mamba_block_aligned_split()` 使用 `self.cache_config.block_size`，实际按 640 切分。
4. Ascend `preprocess_mamba()`：
   - 使用正确的 `mamba_spec.block_size=1280` 计算 `prev_state_idx/curr_state_idx`；
   - 因输入调度已经按 640 切分，跨边界时只复制了 `S640`，无法为 block 0 物化 `S1280`。

这不是单纯的“hash 粒度变成 640”问题，而是三个粒度被混用了：

| 概念 | 运行时值 | 正确用途 |
|---|---:|---|
| Draft attention logical block / hash granularity | 640 | draft KV 页和请求 hash |
| Scheduler common alignment | 1280 | 所有 KV group 的共同可复用边界 |
| Mamba recurrent state block | 1280 | `S_B` 的 checkpoint 语义 |

只有后两者必须保持一致；当前调度路径却用 640 决定了 Mamba prefill 的切分点。

## P4 实验设计

公共边界 `B` 从运行时读取为 1280。构造：

- `P`：目标 prompt 的前 1280 token；
- `W`：完整 warmup prompt，1460 token；
- `H`：另一个 prompt，1498 token；
- `W/H` 实际公共前缀为 1415 token，因此 prefix cache 的共同完整边界为 1280。

只运行三个最小对照，每个请求 `temperature=0`、`max_tokens=1`、`ignore_eos=false`：

| 组 | Prefix cache | 请求序列 | 目的 |
|---|---|---|---|
| R / Reference | 关 | `P -> H` | H 的无缓存参考输出 |
| U / Unaligned | 开 | `W -> H` | 复现跨 1280 边界的 warmup |
| A / Aligned | 开 | `P -> H` | 让 warmup 恰好落在 1280 边界 |

远程配置：

- 主机/容器：`<remote-host>:10005` / `scc_dflash_dev`
- 代码：`/home/shichuchao/scc_dflash/prefix_probe_wt`，HEAD `5584725b857f5ed12ded731c501b23ea3f0fce3c`
- 设备：物理 NPU `2,3`，TP2；未使用另一 agent 占用的设备和服务
- 服务：Eager、DFlash K=15、async scheduling off、端口 `6766`
- 产物：`/home/shichuchao/scc_dflash/tmp/dflash_prefix_probe_20260902/p4_run_20260902_203500/{R,U,A}`

## P4 输出结果

同一个 H 请求的首 token：

| 组 | H 的首 token 文本 | 结论 |
|---|---|---|
| R | `Jan` | Reference |
| U | `Let` | Prefix-on 非对齐 warmup 后分歧 |
| A | `Jan` | 对齐干预恢复 Reference |

即 `A(H) == R(H)` 且 `U(H) != R(H)`。这已经把输出分歧与 warmup 是否正确物化 1280-token state checkpoint 建立了因果联系。

## P4 状态证据

### U：错误路径

`P4_STATE_PLAN`：

| Step | 请求 | computed/scheduled | prev -> curr | Prefix 恢复 | copy |
|---:|---|---:|---:|---|---|
| 0 | W | `0 / 640` | `-1 -> 0` | 否 | 否 |
| 1 | W | `640 / 820` | `0 -> 1` | 否 | 是 |
| 2 | H | `1280 / 218` | `0 -> 1` | 是 | 是 |

30 个 GDN 层、conv/SSM 两种状态均证明：

- W 第一段写出的 `S640` 与 W 第二段消费的状态：`30/30 + 30/30` SHA256 完全一致；
- 同一 `S640` 与 H prefix-hit 后消费的状态：`30/30 + 30/30` 完全一致；
- W 第二段把最终 `S1460` 写到 logical block 1，未回写 logical block 0。

因此 H 标称恢复 1280-token prefix，实际消费的是 warmup 的 `S640`。

### A：正确路径

`P4_STATE_PLAN`：

| Step | 请求 | computed/scheduled | prev -> curr | Prefix 恢复 | copy |
|---:|---|---:|---:|---|---|
| 0 | P | `0 / 640` | `-1 -> 0` | 否 | 否 |
| 1 | P | `640 / 640` | `0 -> 0` | 否 | 否 |
| 2 | H | `1280 / 218` | `0 -> 1` | 是 | 是 |

第二段仍写 logical block 0，因此将其从 `S640` 正确更新为 `S1280`。随后：

- P 最终写出的 `S1280` 与 H 实际消费状态：conv `30/30`、SSM `30/30` SHA256 完全一致；
- A(H) 与 U(H) 的消费状态：conv/SSM 的 SHA256 相同数均为 `0/30`，说明两条路径在全部 GDN 层均已分歧；
- A(H) 输出恢复为 Reference 的 `Jan`。

跨独立服务启动比较最早 `S640` 会受数值执行差异影响，不能用于否定同一运行内的结论；上述决定性比较都在同一服务运行内完成。

### Copy 和索引排除

- U 有 120 次 copy-after 记录，A 有 60 次，共 180 个 tensor；全部满足 `dst_after == src_before` 且 `src_after == src_before`。
- H 消费的 physical block 与 `preprocess_mamba` 计划、copy 目的 block 一致。
- 所以 copy 实现和 physical block 寻址不是首因；被复制的源 state 在语义上已经过期。

## 第一阶段 P0～P3 的补充结论

此前 50 题 A/B 已复现：

| 配置 | GSM8K | 第四个 few-shot 尾段串入 |
|---|---:|---:|
| Eager + Prefix off | 49/50（手工复核 98%） | 0/50 |
| Eager + Prefix on | 27/50（官方 54%） | 17/50 |

P0/P1 还发现：prefix-hit 请求只给 DFlash proposer 传入 tail，draft context KV 也只写 tail；draft attention 却使用完整 `seq_lens/block_table`，同时该请求的 draft prefix physical blocks 没有复用 warmup 对应块。这会污染 draft token、降低 acceptance/performance。进一步代码复核后需要把两件事分开：**只传 tail 在 prefix blocks 已正确复用时是正常优化；真正已经确认的错误是 prefix blocks 没有复用。**

P3 检查结果：

| 模式 | verify row | 存在 draft/target mismatch | sampler 违规 |
|---|---:|---:|---:|
| Prefix-off | 78 | 77 | 0 |
| Prefix-on | 128 | 127 | 0 |

错误 draft 被首 mismatch 规则拦截；它不是已经证明的最终精度首因。P4 确认 target recurrent state 错误可以直接改变 target 输出。draft prefix 未复用则由后文 I2 直接解释；应先修 I2，再做内容级验证，不能在现阶段直接追加“重算完整 draft prefix”这种高开销修复。

## 混合 KV 布局整体审计：当前问题清单（2026-09-03）

以下结论覆盖 target Full Attention 1280、draft Full Attention 640、Mamba state 1280 的当前 Qwen3.6 DFlash 布局。状态严格区分“已确认根因”“已确认代码缺陷/症状”和“待动态验证”，避免把静态风险写成已复现故障。

| 编号 | 状态 | 问题 | 直接后果 |
|---|---|---|---|
| I1 | 已确认根因，已有候选补丁 | Mamba prefill 使用 640 粒度切分，跨过 1280 checkpoint | Prefix hit 把 `S640` 当成 `S1280`，直接改变 target logits/输出 |
| I2 | 已确认独立根因，未修复 | DFlash 被归入 EAGLE，但没有显式 eagle group；coordinator 回退为对所有 groups 执行 EAGLE 最后一块丢弃，并在写侧额外发布 EAGLE peek block。Full Attention 会丢块，Mamba 不丢，Mamba 随后又能把全局 `hit_length` 抬高 | 全局命中长度大于 target/draft block table 的真实覆盖；额外 draft block 还会增加 KV 压力并放大 I4 的共享块重写风险 |
| I3 | 已确认症状，是否有独立残留待复测 | Prefix-hit H 只写 tail，而当前 I2 又使 H 未复用 W 的 draft prefix blocks | 当前 draft attention 读取空的 prefix slots；修 I2 后，tail-only 写法理论上成立，需用 block-id/content 探针确认 |
| I4 | 已确认代码缺陷，并发影响待动态验证 | 固定点求交得到更短最终命中后，只截断第一个 Full Attention spec group，没有截断第二个 draft Full Attention group | draft request table 可保留最终命中范围以外的共享 blocks；recompute 会再次写这些 hash-owned blocks，并发下形成共享 cache 读写风险 |
| I5 | 静态确认依赖缺口，运行时影响待 NPU 验证 | async fast path 下 Mamba accepted-token postprocess 在 global stream，下一批 preprocess/copy 在 default stream；同步分支会 `event.synchronize()`，fast path 明确跳过，当前没有对应的 default-stream `wait_event` | 拒绝、EOS 回滚、跨 1280 边界或 block 复用时，下一批可能看到尚未完成的 Mamba state 更新 |
| I6 | 已确认防御缺口 | coordinator 返回结果后没有校验“每个 group 的 blocks 覆盖范围等于全局 hit” | I2/I4 不会就地报错，而是进入 allocation/forward 后静默污染结果 |
| I7 | 已确认接口契约冲突，当前 310P 场景不可达 | Ascend `find_longest_cache_hit_per_group()` 返回单个 `int`；balance scheduler 按每组长度序列调用 `max(per_group_hits)`，recompute scheduler 又继续把它当单个长度 | 混合模型 + KV connector 路径没有统一可满足的返回契约；310P 当前明确拒绝 KV transfer，不是本次精度回归来源 |
| I8 | 已确认性能遗漏，不是精度根因 | Ascend coordinator 未维护 upstream 的 `num_uncached_common_prefix_tokens` | Marconi 公共前缀 admission 提示恒为 0，可能增加重复 prefill 和 KV 压力，尤其影响并发 10 |
| I9 | 已确认同类缺陷复制，当前 310P 场景不可达 | AscendStore 外部缓存 coordinator 同样在无显式 EAGLE group 时标记全部 groups，最终也只截断第一个 Full Attention group | 如果以后给 310P/该混合布局开放外部 KV cache，I2/I4 会在外部 cache hit 路径再次出现 |
| I10 | 候选补丁作用域缺口，当前 640 配置不触发 | `patch_mamba_scheduler_310.py:26-35` 仅在 scheduler block 大于最小 group block 时接管；若 DFlash+Mamba 各 group 都是 1280，则回退到 upstream，而 upstream 又因 `use_eagle` 把 `last_cache_position` 减一块 | 关闭 640 优化或更换布局后，1460 prompt 仍可能一次跨过 1280，重新制造过期 Mamba checkpoint |

### I1：Mamba checkpoint 被 640 分块跨越

这是 P4 已完成因果验证的问题。候选补丁 `patch_mamba_scheduler_310.py` 让 DFlash Prefix-on prefill 在下一个**绝对 1280 边界**暂停；1460 变成 `1280 + 180`。轻量穷举覆盖 prompt 长度 1～10000、不同 token budget 和任意非对齐起点，共 10035 个 case、213851 次 chunk 调用，没有任何 chunk 跨过内部 1280 checkpoint。

该结果证明调度公式不只适用于 1460；但它只保证 Mamba checkpoint 被物化，不会自动修复 I2～I5。

### I2：全局命中长度与各 group 的实际 block 覆盖不一致

直接代码位置：

- `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:127-131`：`use_eagle=True` 且没有 group 被显式标记时，把全部 group id 放入 `eagle_group_ids`。
- 同文件 `301-345`：每个 EAGLE attention group 用“多看一块、再丢最后一块”的规则找命中；Mamba manager 对 `drop_eagle_block` 没有相同语义。
- Ascend coordinator 没有覆盖 `cache_blocks()`，实际继承 upstream `kv_cache_coordinator.py:594-620`；只要 `manager.use_eagle=True`，写侧还会把 `aligned_num_computed_tokens + manager.block_size` 以内的一块设为可缓存。对 1280/640 布局，若 token budget 使某轮停在 1920，draft 第 3 个 640 block 会被发布，虽然全局下一次可命中边界仍是 2560。

使用真实 `AscendHybridKVCacheCoordinator` 类和当前三种 spec 构造 CPU 反例，得到：

| 可命中前缀 | coordinator 返回的 `hit_length` | target/draft/Mamba 实际返回块数 | 正确块数 |
|---:|---:|---:|---:|
| 1280 | 1280 | `0 / 0 / 1` | `1 / 2 / 1` |
| 2560 | 2560 | `1 / 2 / 2` | `2 / 4 / 2` |
| 3840 | 3840 | `2 / 4 / 3` | `3 / 6 / 3` |

同一构造令“KV cache 读写策略不使用 EAGLE drop/peek”后，三组块数分别恢复为正确的 `1/2/1`、`2/4/2`、`3/6/3`。这里不是关闭 DFlash 推测本身：scheduler 仍需保留 DFlash 的 `K+1` lookahead、proposer 和 speculative verification，只应把“KV prefix 是否丢最后一块、是否额外发布一块”从通用 `use_eagle` 中拆开。该反例证明 host 侧 Prefix Cache 元数据契约已经被击穿：**全局 `hit_length=L` 时，每一个参与求交的 group 都必须提供覆盖 `[0,L)` 的有效 blocks。**

进一步复核 DFlash 算法后可以确认，这个拆分不会让 proposer 缺少“最后一个缓存 token 的 target hidden state”：`vllm/v1/spec_decode/dflash.py:37,95-117` 明确把已计算 target hidden states仅作为 context K/V；`copy_and_expand_dflash_inputs_kernel` 用 `next_token_ids` 构造 bonus query，再追加 mask queries；`qwen3_dflash.py:125-150` 的 draft forward 也只计算这些 query。命中的 prefix 已经有 draft context KV，当前轮只需为 tail hidden states补写 KV。因此 DFlash 不需要沿用 EAGLE 为 hidden-state shift 设计的 last-block recompute。

后续 `allocate_new_computed_blocks` 会按已经增加的 `num_computed_tokens` 继续分配，无法补回被 scheduler 跳过的那段 prefix 内容。实际 two-round trace 也与反例一致：W 的 draft physical block table 前十项为 `25..34`，H 命中 1280 后前十项却变成 `420..424,415..419`，没有复用 W 的前缀块。H 只写 1280 之后的 tail，因此 draft 前缀槽为空。P4 当时只校验了 Mamba state 和一个首 token，所以不能用单样本输出恢复来否定这个独立问题。

### I3：tail-only 不是天然错误，当前错误由 I2 触发

直接代码范围：`vllm_ascend/_310p/spec_decode/dflash_proposer_310.py:592-682`。

Prefix hit 后，target 已跳过公共 prefix，只把 tail hidden states 交给 proposer；precompute 也只会在 tail slot 写 draft KV。draft attention 的 `seq_lens` 和 block table 表示“prefix + tail”的完整序列。这里的正确契约应是：prefix 由缓存块提供，tail 由本轮 proposer 写入。

```text
复用 prefix KV + 写入 tail KV + 按完整长度读取
```

代码链证明 W 的每个 prefill chunk 都会运行 proposer：`model_runner_v1.py:2741-2753` 对 speculative config 调用 proposer；`1898-1915` 在普通 prefill 中把本轮所有 scheduled target hidden states 传进去；`dflash_proposer_310.py:592-682` 将这些位置的 draft context KV 写到对应 slots。因此 two-round W 会先写 `[0,1280)`，再写 `[1280,1460)`。只要 I2 修复后 H 正确复用 W 的 `[0,1280)` draft blocks，H 只写 tail 是完整生命周期，不需要重新计算整个 prefix。

所以 I3 当前应作为 I2 的可观测症状：P0/P1 已观测到 prefix blocks 没复用，127/128 个 verify rows 的 draft 与 target mismatch；P3 证明 sampler 没越过首 mismatch。修 I2 后需增加一个内容级回归：H 的 prefix block ids 必须等于 W，且 H 的 draft context prefix checksum 在 proposer 前后不变。只有该测试仍失败，才继续追独立 proposer bug。

### I4：只截断第一个 Full Attention group

直接代码位置：`patch_kv_cache_coordinator.py:349-361`；`find_longest_cache_hit_per_group` 的 `451-463` 还有一份相同逻辑。

当前实现只执行：

```python
spec, group_ids, _ = self.attention_groups[0]
```

即只截断排在第一个的 target Full Attention group。Qwen3.6 DFlash 同时有 target Full Attention 1280 和 draft Full Attention 640，draft 是第二个不同 spec 的 Full Attention group。当后续 Mamba/SWA 求交把最终命中缩短时，draft 已查到的更长块列表不会被截短。

CPU 反例中，先缓存 3000 token，再让第二个 Mamba checkpoint 不可用：最终命中回退到 1280，但返回块数是 `target=1 / draft=4 / Mamba=1`；按全局命中契约应为 `1 / 2 / 1`。额外两个 draft blocks 的 token hash 仍可能与本请求后续 token 相同，所以串行执行时不能简单称为“内容必错”；真正的问题是 scheduler 从 1280 开始 recompute 时，DFlash proposer 会再次写这些仍带缓存 hash、且可能被其他请求共享读取的 blocks。并发 10 下可能出现同一 cache block 的并发读写，也使 eviction/reference 统计偏离全局命中范围。

修复不能只改一个下标，应遍历 `self.attention_groups`，将**所有** `FullAttentionSpec` groups 截到最终 `hit_length`。同样逻辑在 `find_longest_cache_hit()` 和 Ascend 自定义的 `find_longest_cache_hit_per_group()` 中各有一份；后者更适合直接恢复为 v0.24 的“各组独立查询、返回 `tuple[int, ...]`”接口。

### I5：异步调度不改变边界公式，但可能改变 state 可见时序

已有静态证据和排除项：

- upstream scheduler 在提交 batch 后、设备结果返回前就增加 `request.num_computed_tokens`，并允许多个 batch in flight；
- `allocate_slots()` 会在 forward 执行前发布/cache block 元数据；但 multiprocess worker 按 FIFO 执行 `Execute(W) -> Sample/Propose(W) -> Execute(H)`，普通 target/draft 写又在同一默认 stream，因此“提前发布”本身尚不能判错；
- Mamba manager 的 `cached_blocks_this_step` 会阻止同一个 scheduler step 内的另一请求命中尚未生成的 Mamba checkpoint；
- 310P Mamba accepted-token postprocess 在 `global_stream()` 上执行，而下一批 `preprocess_mamba()`/copy 在 default stream；`model_runner_310p.py:652-709,886-924` 的 async device-metadata fast path 跳过 host `synchronize()`，当前也未找到 default stream 对上一轮 `num_accepted_tokens_event` 的设备侧 `wait_event`。

因此 I5 的准确表述是“静态上确认缺少跨流 happens-before，动态上尚未证明已经产生可见 race”。`model_runner_v1.py:2801-2808` 在 `global_stream()` 上执行 postprocess，随后记录 `num_accepted_tokens_event`；`model_runner_310p.py:652-709,886-924` 的 async device-metadata fast path不等待该事件，而同步路径明确执行 `event.synchronize()`。推荐修复方向是在下一轮 default stream 消费/复制 Mamba state 前，对上一轮 `num_accepted_tokens_event` 做设备侧 `wait_event`，避免恢复 host 全同步；最终位置必须由 NPU 事件探针验证。现有 P4 使用 `--no-async-scheduling`、`max_tokens=1`，没有覆盖这一项。后续最小 NPU 实验应围绕 accepted count 1～16、首 mismatch、EOS/ignore_eos、跨 1280 checkpoint、preemption/block reuse 和并发 10，而不是先重跑大规模精度集再猜。

### I6～I10：防御、接口和补丁作用域遗漏

- `SingleTypeKVCacheManager.add_local_computed_blocks()` 会直接 touch 并追加 coordinator 返回的全部 blocks，再把 `num_cached_block` 设为列表长度；它不检查列表是否只覆盖全局 hit。因此建议在 coordinator 返回前增加通用不变量校验，测试环境强制断言每个参与 group 的 `len(blocks) * effective_block_size == hit_length`。
- Ascend 自定义 `find_longest_cache_hit_per_group()` 当前复制了固定点求交逻辑并返回单个 `int`；`patch_balance_schedule.py:445-459` 明确把第二项当每组长度序列并执行 `max(per_group_hits)`，而 `core/recompute_scheduler.py:538-549` 又继续把它当单个 token 长度。接口与两个消费者互相冲突，不能只把返回值改成 tuple；应先统一接口，再原子更新两个调用方和统计口径。310P 在 `model_runner_310p.py:1251-1254` 拒绝 KV transfer，所以它不会造成当前单机 TP2 精度下降。
- upstream coordinator 用 `longest_hit_length - hit_length` 维护 `num_uncached_common_prefix_tokens`，Ascend 自定义实现没有维护。scheduler 通过 `getattr(..., 0)` 静默降级，因此不是精度 bug，但会失去 Marconi admission 优化；在并发 10 和高 KV 压力下可能增加重复计算与瞬时占用。
- `distributed/kv_transfer/kv_pool/ascend_store/coordinator.py:92-94,215-286` 还有一套独立的外部缓存求交实现：它同样在无显式 group 标记时把全部 groups 当 EAGLE，也同样只截断 `attention_groups[0]`。当前 310P 拒绝 KV transfer，因此这不是本次线上路径；但修 I2/I4 时应同步修复或抽取公共实现，否则一旦开放 AscendStore，外部 cache hit 会重新违反相同覆盖不变量。
- 当前 scheduler 候选补丁在 `_needs_dflash_mamba_checkpoint_split()` 中要求 `scheduler_block_size > cache_config.block_size`。这个条件能覆盖当前 1280/640/1280 布局，却把正确性错误地绑定到了内存优化。正式实现应对所有“DFlash + hybrid Mamba + Prefix Cache”按真实公共 checkpoint 处理；即使 group block 都是 1280，也不能回退到 DFlash 的通用 EAGLE prune。真实 EAGLE/MTP 的原语义仍保持不变。

### 全场景复核矩阵

| 场景 | 结论 |
|---|---|
| prompt 小于 1280，或没有共同完整 1280 prefix | 正常回退为从 0 计算；two-round 不会制造伪命中 |
| prompt 恰好 1280 | `max_cache_hit_length=prompt_len-1=1279`，再按 1280 对齐后命中 0，整段重算；这是为获得最后一个 token logits 的当前设计限制，不是精度 bug |
| prompt 为 1281～2559 | 最多命中 1280；I1 决定 `S1280` 是否真实，I2 决定 target/draft blocks 是否覆盖 1280，两者都必须修 |
| prompt 恰好 2560 | 最大命中上限是 2559，只能命中 1280、重算最后 1280；同样是预期的整块重算性能阶跃 |
| prompt 为 2561～3839 | 最多命中 2560；要求 `S1280/S2560` 均真实，且三组分别覆盖 `2/4/2` blocks |
| prompt 跨多个 1280 checkpoint | two-round 会逐个停在绝对边界；已用 1～10000 token 穷举切分，但每个边界仍受 I2/I4 约束 |
| target/draft/Mamba 某一组被 eviction | 最终 hit 应取各组共同可用长度；后置 Mamba 缩短时会触发 I4 |
| 同 scheduler step 出现相同前缀请求 | Mamba `cached_blocks_this_step` 会推迟消费者，未发现同 step 读取未完成 checkpoint 的遗漏 |
| preemption/resume | request 会把 `num_computed_tokens` 归零并重新走 prefix lookup，因此会重新触发 I1/I2/I4，不是独立算法 |
| async scheduling | 不改变 1280 绝对边界公式；普通 FIFO/default-stream 路径可保序，但 I5 跨流 state 更新仍待验证 |
| TP2 | coordinator 和 block 语义不会乘二；两个 rank 必须看到相同调度/block table，问题数量不是 2 倍 |
| eager/ACLGraph | I1/I2/I4 都在 host 元数据层，与图模式无关；图模式还需覆盖 I5 的事件依赖 |
| `ignore_eos` 开关 | 只影响停止条件；可能改变 EOS 回滚样本覆盖，不会制造上述错误 checkpoint/blocks |

本轮静态复核还确认：`may_reinitialize_input_batch()` 会按各 group 的真实 block sizes 重建多组 block table；310P DFlash proposer 也按每层真实 physical kernel block size 重算 context/query slot mapping。因此“所有 worker 仍把 640 当 1280”不是新的遗漏根因。

### 第二轮审计的排除项与潜在兼容风险

| 项目 | 当前结论 | 何时需要重新评估 |
|---|---|---|
| DFlash 是否需要重算已命中 prefix 的最后一个 target hidden state | 不需要。query 是 `next token + mask tokens`，prefix 只需已有 draft context KV；修 I2 后 tail-only 是正确优化 | 修 I2/I4 后 prefix block id 或 checksum 仍不一致 |
| `num_context == 0` 导致 DFlash 读取 `last_pos[-1]` | 当前 prompt lookup 被 `prompt_len-1` 限制，至少会重算一个粒度块/尾 token；正常 decode 也至少有一个 target token，因此当前不可达 | 将来支持真正的 100% prompt hit 且不重算 logits token |
| rejection/EOS 写入 draft tail | rejected trailing context 虽先写 slot，但 query 从最后 accepted position 开始并覆盖相同后缀 slot，attention 的 `seq_lens` 也减去 rejected 数；未发现独立同步路径错误 | async 下仍需结合 I5 做 event/state checksum 验证 |
| C8/NZ attention 的全局 block-size view | `attention_v1.py` 的 C8 backend 仍有读取全局 `cache_config.block_size` 的混合布局风险；当前命令的 `--quantization ascend` 是权重量化，KV cache 不是 C8，因此不在本次路径 | 开启 C8/量化 KV cache，且不同 group 使用不同逻辑 block |
| 310P KV block zeroer 的单一 logical-page ratio | zeroer 从第一个 Full Attention group 保存一个 ratio，若用于 1280/640 两种 Full Attention cache 会清错行；但 worker 当前只对 `eagle3` 初始化，DFlash 不会进入 | 将 zeroer 扩展到 DFlash，或统一初始化所有 speculative methods |
| `--disable-hybrid-kv-cache-manager` | 当前 Full Attention + Mamba + DFlash 组合无法被 unifier 合并，会在初始化阶段拒绝，不是静默精度问题 | 将来 unifier 支持 Mamba mixed specs |
| 外部 KV connector / AscendStore | 310P 当前明确拒绝该路径；I7/I9 不影响当前单机 TP2 | 后续开放 P/D disaggregation 或 AscendStore |

以上风险中，只有 I1、I2、I4 位于当前 Prefix-on 同步执行主链；I5 位于当前可配置的 async 主链但尚缺 NPU 动态证据；I10 和表中其他项是正式修复必须考虑的边界，不应与当前 50 题精度下降的已证实根因混为一谈。

本轮又做了一个不依赖 NPU 的边界算术复核：prompt 长度 1～3000 乘以 9 种 token budget，共 27000 个切分 case，按绝对 1280 checkpoint 处理后内部边界跨越数为 0；同时确认等 block 的 upstream DFlash 首轮仍会取 1460，而正确值应为 1280；当 budget 让当前长度停在 1920 时，现有 EAGLE 写策略会发布 3 个 draft 640 blocks，拆分后只应发布公共 1280 范围内的 2 个。该复核验证的是边界公式和发布数量，不替代 coordinator 单测或 NPU 内容校验。

### 目前可以确认的修复边界

完整修复至少要同时建立四条不变量：

1. **Checkpoint 不变量**：任何被 Prefix Cache 发布的 1280 边界都有完整有效的 Mamba `S_B`。
2. **覆盖不变量**：全局 `hit_length=L` 时，target、draft、Mamba 每个 group 的返回 blocks 都真实覆盖 `[0,L)`。
3. **内容/所有权不变量**：任何出现在 draft block table 的 prefix slot，都已写入正确 KV 或复用了内容等价的缓存块；被作为 cache hit 共享的块不得被本请求再次写入。
4. **时序不变量**：异步下一批读取/复制 Mamba state 前，上一批 accepted-token 更新已经在设备上完成。

当前 1280+640 两轮 scheduler 补丁只满足目标配置中的第 1 条；I10 表明它还没有覆盖等 block 的 DFlash+Mamba。I2 和 I4 应先用纯 CPU coordinator 回归测试修复；I3 先作为修复后的验证项，不预设额外重算方案；I5 必须在 NPU 上做最小跨流验证后才能定最终改法。

## 修复方案

### 必须修复：Target/Mamba checkpoint 契约

保留 prefix cache 和 draft 640 block 的前提下，正式修复应满足：

1. Mamba prefill 不能跨过一个未来可 prefix-hit 的 1280 边界，却只保存跨越后的最终状态。
2. 调度器对 Mamba 使用 `scheduler_block_size`/Mamba block（这里均为 1280），不能使用被 EngineCore 改成最小 group block 的 `cache_config.block_size=640`。
3. 当 `computed < next_1280_boundary < computed + scheduled` 时，先把本轮截到该边界，确保 GDN 把 `S1280` 写入 logical block 0；下一轮再计算 tail。
4. DFlash 被 vLLM 归类为 Eagle-like method，现有“drop/recompute last block”分支不能绕过 Mamba checkpoint 物化。Eagle hidden-state 需求和 Mamba recurrent checkpoint 应分别处理。
5. prefix manager 只能发布已经完成对应 state write 的 1280 边界；建议增加运行时断言或 validity 标记，禁止把 partial state block 当作完整 prefix checkpoint。

仅把 `_mamba_block_aligned_split` 中的字段从 `cache_config.block_size` 改成 `self.block_size` 还不够：对 1460-token W，Eagle 分支可能一次计算完整 1460，仍不会自然产生中间 `S1280`。修复必须直接保证“跨可复用边界时先停在边界”，或让 GDN 显式输出/保存中间边界状态。前者改动更小，建议先做。

### 必须修复：DFlash 与 EAGLE 的 KV lookup 语义拆分

保留 scheduler 的 DFlash speculative lookahead，但传给 KV cache coordinator 的 EAGLE KV 标志必须对 DFlash 为 false。这个标志同时控制 lookup 的 last-block drop 和 `cache_blocks()` 的额外 peek-block 发布，两边必须一起拆，不能只改查询。不要简单把全局 `self.use_eagle` 改成 false，否则会连带破坏 DFlash 的 `K+1` lookahead、encoder shift 等其他行为。可在 Scheduler 构造 KVCacheManager 时引入单独的 `use_eagle_for_kv_cache`：真实 EAGLE/MTP 沿用现状，DFlash 的 KV cache 读写不执行 drop/peek。

同时把 DFlash 的 Mamba checkpoint 物化条件从“`scheduler_block_size > cache_config.block_size`”扩大为“DFlash + hybrid Mamba + Prefix Cache”。当前 640 优化继续保留，但不再作为正确性开关；这样 1280/640/1280 和未来 1280/1280/1280 两种布局都按绝对公共 checkpoint 切分。

### 必须修复：所有 Full Attention groups 对齐最终 hit

在 Ascend coordinator 的最终截断阶段遍历全部 `FullAttentionSpec` groups，target 1280 和 draft 640 都截到全局 `hit_length`。同时增加覆盖断言，使 I2/I4 在 host 侧立即失败，而不是进入 NPU forward 后静默错算。

### 修复后验证：DFlash draft prefix KV 生命周期

先验证低成本路径：

- W 分两轮写满 draft context `[0,1460)`；
- H 命中 1280 后，draft prefix 的 logical/physical block ids 与 W 相同；
- H proposer 只写 tail，prefix block checksum 不变；
- draft attention 的每个 `[0, seq_len)` slot 都能映射到有效缓存内容。

若以上成立，不需要补算完整 prefix，可保留 640 内存优化和 tail-only 性能优势。只有修 I2/I4 后仍有缺失内容，才考虑补算 prefix；不应继续使用当前“只写 tail + 读取完整序列 + 新分配 prefix blocks”的组合。

### 待验证修复：async Mamba 跨流依赖

在 default stream 的下一轮 `preprocess_mamba()` 读取/copy state 前等待上一轮 `num_accepted_tokens_event`。优先使用 stream `wait_event`，不恢复每轮 host `synchronize()`；先用事件前后 state checksum 探针确认 race，再固化代码位置。

### 最小回归测试

产品修复至少需要以下测试：

1. 异构 groups：target/Mamba block 1280、draft block 640、hash 640、scheduler alignment 1280。
2. 非对齐 W：1460 token 必须调度为 `1280 + 180`（或能证明等价物化 `S1280`），不能是 `640 + 820` 或单次跨过 1280。
3. Prefix-hit H：`num_computed_tokens=1280` 时，恢复 state checksum 必须等于 W 在 token 1280 的 checkpoint。
4. R/U/A：修复后 `U(H) == A(H) == R(H)`。
5. 原始 AISBench：50 个 GSM8K，Prefix-on 精度应回到 Prefix-off 统计波动范围，且不再出现第四个 few-shot 尾段串入。
6. Draft KV：prefix-hit 请求读取的每个 draft prefix block 都必须已写入或复用，不能读取新分配的未初始化块。
7. Coordinator：对 1280/2560/3840 hit，target/draft/Mamba 必须返回 `1/2/1`、`2/4/2`、`3/6/3`；DFlash 不得套 EAGLE drop。
8. Eviction：Mamba hit 从 2560 回退到 1280 时，两个 Full Attention groups 都必须同步截断，且 cache-hit blocks 不得进入本轮写集合。
9. 并发 10：混合长短 prompt、相同前缀、Mamba checkpoint eviction、preemption，检查共享 block 无并发写以及 Prefix-on/Prefix-off 输出一致。
10. Async：accepted count 1～16、首 token reject、EOS、跨 block 边界，检查下一轮消费 state 等于上一轮 postprocess 完成后的 state。
11. 等 block 兼容：把 target/draft/Mamba 都设为 1280，DFlash 1460 prefill 仍必须得到 `1280 + 180`，不能因 `use_eagle` 回退为一次跨越。
12. Prompt 精确边界：1280/2560 prompt 分别只能命中 0/1280，而 1281/2561 分别命中 1280/2560；这些是 `prompt_len-1` 规则的预期行为，输出必须与 Prefix-off 一致。
13. 写侧边界：人为用 token budget 让请求停在 1920；DFlash KV 模式下 draft 第 3 个 640 block 不应在 2560 公共边界前作为可复用 prefix 发布。

临时保护如“310P DFlash 禁用 prefix caching”能规避精度问题，但不符合保留功能的目标；修改 EOS/rejection 参数无效。

## 两轮 DFlash 修复快速验证（2026-09-03）

已按上述优先方案实现最小补丁：仅当运行环境同时满足 310P、DFlash、Prefix Cache 已开启、`scheduler_block_size > cache_config.block_size` 且两者整除时，Mamba prefill 才按下一个绝对 `scheduler_block_size` checkpoint 截断；其他模型、Prefix-off 和 block 粒度一致的路径继续调用 vLLM 原实现。draft attention 的 640-token block 保持不变。

新增回归测试覆盖：

1. 1460-token W 必须是 `1280 + 180` 两轮。
2. 超过两个 checkpoint 的 3000-token prompt 必须逐个物化，调度为 `1280 + 1280 + 440`。
3. 先因 token budget 调度 1000 token 时，后续必须按绝对边界调度 `280 + 180`，不能把边界错误平移。
4. 非 DFlash 路径保持上游 Eagle 的现有分块行为。
5. DFlash Prefix-off 路径保持上游行为。

TDD RED 阶段在未加补丁时得到 `2 failed, 2 passed`，其中 1460 case 明确显示当前错误值 `640 + 820`；补丁后新增测试 `5 passed`。连同 KV block 解析和 310P model runner 相邻回归，最终结果为 `42 passed, 14 warnings`，Ruff 为 `All checks passed!`。

远程最小 P4 使用独立 worktree、端口 `6766` 和当时空闲的设备 `0,1`，只发送 W/H 两个请求。运行时证据为：

- W 的 `P4_STATE_PLAN`：`0/1280`、`1280/180`。
- H 的 `P4_STATE_PLAN`：`1280/218`，`restored_from_prefix=true`。
- W 在 1280 checkpoint 写入的 conv/SSM state 与 H 恢复后实际消费的 state：30/30 层 hash 完全一致。
- H 首 token：修复前 Prefix-on U 为 `Let`，修复后为 `Jan`，与 Prefix-off R 和边界干预 A 的 `Jan` 一致。

这次快速验证确认了 Target/Mamba checkpoint 修复的因果链，但不替代最终 50 题 AISBench，也没有解决 I2/I4；trace 中 H 的 draft prefix block ids 未复用 W，正好提供了 I2 的设备侧症状证据。应先完成 coordinator 修复，再判断 proposer 是否还有独立问题。实验服务已按记录 PID 停止，端口 `6766` 已释放，未操作另一个 agent 的服务。

## 验证与资源清理

- P4 probe 单测先 RED：`2 failed, 3 passed`；补齐 SHA256/abs/square summary 后 GREEN：`5 passed`。
- 最终远端定向回归：`13 passed, 14 warnings in 0.07s`。
- 最终 Ruff：`All checks passed!`。
- P4 R/U/A 均已停止；端口 `6766` 无监听，诊断 worktree 无残留进程。
- 实验使用的物理 NPU `2,3` 已释放；没有停止、重启、请求或修改另一 agent 的服务。
- 一次 U 启动曾因设备 TSD/AICPU 初始化超时 `507033` 失败，未发送请求；改用空闲且健康的设备后完整重跑，不计入实验结果。

本地分析产物：

- `D:/work/dflash_fix/.remote_p4/analyze_p4.py`
- `D:/work/dflash_fix/.remote_p4/analyze_two_round.py`
- `D:/work/dflash_fix/.remote_p4/results/{R,U,A}_trace_rank0.jsonl`
- `D:/work/dflash_fix/.remote_p4/results/{R,U,A}_requests.jsonl`
- `D:/work/dflash_fix/.remote_p4/results/two_round_20260903/{trace_rank0,requests}.jsonl`

## 正式修复与端到端验收（2026-09-04）

### 最终根因与修复范围

本轮没有通过关闭 Prefix Cache、Async Scheduling 或 draft 640-token block 规避问题。正式修复同时处理了以下相互独立的错误：

1. **Mamba checkpoint 被跨过（I1/I10）**：310P DFlash + Prefix Cache 始终按下一个绝对 1280-token checkpoint 截断 prefill。W1460、已命中 640 时先补到 1280，再计算 180，而不是从 640 一次计算 820。
2. **DFlash 错用了 EAGLE KV 所有权规则（I2）**：DFlash 继续保留 K+1 lookahead、proposer 和 rejection verification，但 `KVCacheManager` 不再对 DFlash执行 EAGLE 的 last-block drop/extra-peek。真实 EAGLE 行为不变。
3. **共同命中只截断第一个 FullAttention group（I4/I6/I8）**：两个 coordinator lookup 路径共用 finalizer；所有 FullAttention groups 都截到统一 hit length，并检查实际 blocks 是否恰好覆盖命中范围，同时恢复 `num_uncached_common_prefix_tokens` 统计。
4. **不同 kernel block size 的元数据没有成对转换**：310P proposer 现在按 block size 同时生成并下发 `(block_tables, slot_mapping)`，使用持久 host/device buffer；当前 Qwen3.6 全 128 kernel-block 快速路径继续复用原表。混合 block size 与 Context Parallel 的未支持组合会显式拒绝，不再静默错算。
5. **Async accepted-token 跨 stream 缺少依赖（I5）**：消费 stream 在使用设备侧 accepted-token/Mamba 元数据前执行 `wait_event`，没有新增 host/global/device synchronize。

I3（prefix-hit 请求只计算 tail）不是独立错误：修复 I2/I4 后，prefix KV 由缓存块提供，tail 由本轮 proposer 写入，仍保留原性能优化。`ignore_eos` 也不是根因；它只改变停止策略，不改变 checkpoint、KV ownership 或跨流顺序。

### 代码位置

| 仓库 | 文件 | 作用 |
|---|---|---|
| vLLM Ascend | `vllm_ascend/patch/platform/dflash_kv_context.py` | 以调用上下文将 DFlash 身份传到 KV coordinator，不修改 vLLM |
| vLLM Ascend | `vllm_ascend/patch/platform/patch_mamba_scheduler_310.py` | 标记 DFlash 初始化调用链；绝对 1280 checkpoint 切分 |
| vLLM Ascend | `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py` | DFlash/EAGLE KV 语义解耦；全 FullAttention group 截断、覆盖校验和 common-prefix 统计 |
| vLLM Ascend | `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py` | per-size block table/slot mapping 成对转换和缓存 |
| vLLM Ascend | `vllm_ascend/_310p/model_runner_310p.py` | async accepted-token 设备侧 event 依赖 |
| vLLM Ascend | `tools/probes/probe_310p_async_event_visibility.py` | NPU 跨流可见性实验 |
| vLLM Ascend | `tools/probes/probe_310p_prefix_lifecycle.py` | 640/1280/2560 边界、共享前缀和并发 10 探针 |

### 单元测试与静态检查

在远程隔离 worktree `/home/shichuchao/scc_dflash/worktrees/prefix-fix-20260904` 复验：

- vLLM Ascend 组合测试：`107 passed, 15 warnings in 0.52s`。
- Ascend 改动文件 Ruff：`All checks passed!`。
- `git diff --check`：通过。
- diff-only 检查确认 async 修复没有新增 `.synchronize()`。
- DFlash/EAGLE 构造级集成测试确认：两者的 `Scheduler.use_eagle` 和 K+1 lookahead 均保持不变；只有 DFlash 初始化调用链传给 Ascend KV coordinator 的有效 `use_eagle` 为 False，普通 EAGLE 仍为 True；异常退出后 ContextVar 恢复。
- vLLM overlay 已恢复原始 `use_eagle=self.use_eagle`，没有 `_uses_eagle_kv_cache` 等直接修改；本地 vLLM 仓库保持 clean。

vLLM 更大的 scheduler 测试集合在该容器离线环境中会尝试加载未缓存的 `facebook/opt-125m`，因此没有作为有效证据；离线策略/构造级测试和实际 Qwen3.6 服务启动/推理均通过。

### 独立代码审查补强

独立审查未发现 Critical，提出的 Important 已逐项处理：

- per-layer 混合布局不再用 graph padding 前保存的 source table/slot 覆盖 base 最终 metadata；转换布局使用可覆盖最终 graph shape 的持久 table/slot buffer，并新增 Full Graph padding 回归测试。
- `draft_window_size` 的窗口起点和有效 `seq_len` 会随 kernel block size 改变；当前对 converted/mixed kernel block size 组合明确 fail-fast，避免用一套窗口 metadata 静默算错。全 source block-size 的正常 sliding-window 路径继续交给 base 实现。
- 新增离线 Scheduler 构造级 DFlash/EAGLE 对照：DFlash 仍保持 `use_eagle=True` 的调度属性和 K+1 lookahead，但 `KVCacheManager(use_eagle=False)`；EAGLE 仍传 `use_eagle=True`。
- 生命周期 API 探针新增 response/usage 强断言、`ignore_eos` 两种请求模式、finish 后顺序复用；异步状态修正单测新增 all-rejected 和 all-accepted 两端边界。

### NPU 小型实验

实验只使用空闲物理 NPU 6、7 和独立端口 6666，没有操作另一 agent 使用的物理 NPU 4、5 和端口 3333。

Async event 可见性探针在物理 NPU 6 上运行 50 轮：

```json
{"host_sync_stale": 0, "no_wait_stale": 42, "rounds": 50, "wait_event_stale": 0}
```

这证明原路径确实可能读到旧值，而设备侧 `wait_event` 足以建立正确顺序。

Prefix 生命周期探针在修复后的 DFlash + Prefix On + Async + TP2 服务上全部完成：

- 边界请求：639、640、1279、1280、1281、1460、2559、2560、2561，共 9 个；
- 共享前缀：W=1460、H=1498、公共 token=1415；
- 并发请求：10/10 成功；
- 服务无 coverage mismatch、NPU async 错误、traceback 或 500。

第一次隔离启动没有加载 CANN 自定义算子环境，首个 0-hit 请求在 `aclnnCausalConv1dV310` 前失败；加载主工作副本只读的 `custom_transformer/bin/set_env.bash` 后重启，探针和后续三组 AISBench 均正常。该失败发生在任何 Prefix Cache 命中之前，属于隔离启动环境问题，不计入产品修复结果。

### AISBench 三组端到端精度

三组均固定使用 Qwen3.6-35B-A3B-w8a8、TP2、Eager、Async Scheduling、`max-num-seqs=10`、相同 chat/dataset/generation 配置；DFlash 两组保持 K=15 和 draft 640-token KV 优化。

命令：

```text
ais_bench --models vllm_api_general_chat \
  --datasets gsm8k_gen_4_shot_cot_str \
  --num-prompts 50 --dump-eval-details
```

| 配置 | 成功请求 | 官方 accuracy | 推理耗时 | `details` 错误项 |
|---|---:|---:|---:|---|
| 修复后 DFlash + Prefix On | 50/50 | 94.00（47/50） | 1:31 | 2、17、37 |
| DFlash + Prefix Off | 50/50 | 94.00（47/50） | 2:06 | 2、12、17 |
| 无 DFlash + Prefix On | 50/50 | 94.00（47/50） | 2:32 | 2、17、37 |

验收结论：

- 修复后的 Prefix On 精度与 Prefix Off、无 DFlash 基线完全相同，较旧实现的官方 54% 恢复到 94%。
- 修复后的 Prefix On 与无 DFlash 基线的错误项集合完全一致；不存在 `baseline=correct, fixed=wrong` 的新增回归。
- 三组自然语言推导表述存在非确定性的措辞差异，但最终抽取答案和逐题判分满足上述验收条件。
- 三份服务日志均无执行期 traceback、RuntimeError、coverage mismatch、EngineDead 或 HTTP 500。

远程产物根目录：

```text
/home/shichuchao/scc_dflash/tmp/prefix_fix_20260904_0205
```

关键产物：

- `fixed_prefix_on_lifecycle.json`
- `fixed_prefix_on_attempt4.log`
- `dflash_prefix_off.log`
- `baseline_no_dflash_prefix_on.log`
- `ais_fixed_prefix_on/20260904_024146/`
- `ais_dflash_prefix_off/20260904_024759/`
- `ais_baseline_no_dflash/20260904_025445/`

### 资源清理与剩余边界

- 本轮启动的 API PID 3678765、3681362、3683677 均已按记录精确停止；端口 6666 已释放。
- 另一 agent 的服务和进程未被停止、重启或修改。
- 310P 当前显式拒绝 KV transfer；I7/I9 的外部 KV connector 接口问题不在本次单机 TP2 路径中，后续开放该能力时必须单独修复。
- 当前 Qwen3.6 全 128 kernel-block 走零转换路径；64/128 混合布局已由 CPU 非连续物理页测试覆盖，尚未用当前模型做不可达的在线混合布局实验。

### 独立审查补丁后的最终远程复验

上述临时 HCCL 上下文释放后，在同一隔离 worktree、物理 NPU 6/7、端口 6666 上重新启动了完整目标配置：Qwen3.6-35B-A3B-w8a8、DFlash K=15、draft TP2、Prefix Cache On、Async Scheduling、Eager、`max-num-seqs=10`。独立审查提出的 Full Graph/混合布局和生命周期探针补丁均已包含。

增强后的生命周期探针全部通过：

```json
{"boundary_requests": 9, "concurrent_requests": 10, "forced_eos_ignore_tokens": 4, "forced_eos_stop_tokens": 1, "reuse_after_finish_match": true}
```

- 9 个 640/1280/2560 邻接边界请求均成功；
- 10 个共享前缀请求并发成功，服务日志的 Prefix Cache hit rate 最终为 80.5%；
- 通过 `allowed_token_ids=[eos_token_id]` 确定性强制 EOS：普通模式生成 1 token 后以 `stop` 结束，`ignore_eos=true` 继续到 4 token 并以 `length` 结束；
- 请求 finish 后再次提交相同请求，输出与探针基准一致，未发现残留 KV 污染。

随后重新执行同一条 50 题 AISBench 命令：

| 配置 | 成功请求 | 官方 accuracy | 推理耗时 | `details` 错误项 |
|---|---:|---:|---:|---|
| 独立审查补丁后 DFlash + Prefix On | 50/50 | 94.00（47/50） | 1:26 | 2、12、37 |
| 无 DFlash + Prefix On 基线 | 50/50 | 94.00（47/50） | 2:32 | 2、17、37 |

最终修复版与无 DFlash 基线的汇总精度相同，均为 94%，因此本次要求的“保留 Prefix Cache 和 draft 640 内存优化，同时精度不下降”得到端到端验证。逐题结果不是完全相同：第 12 题由基线正确变为修复版错误，第 17 题由基线错误变为修复版正确，净精度变化为 0；这两题互换应记录为生成路径的样本级波动，不能表述成逐 token 或逐题完全一致。

最终运行目录：

```text
/home/shichuchao/scc_dflash/tmp/prefix_fix_20260904_0205/final_review_20260904/ais_final/20260904_035531
```

最终服务日志没有 traceback、RuntimeError、断言失败、coverage mismatch、EngineDead 或 HTTP 500。API PID 3699501 及其 Engine/TP worker 已按精确 PID 正常退出，端口 6666 已释放；没有停止或修改另一 agent 的 NPU 4/5、端口 3333 服务。

### FULL_DECODE_ONLY 图模式补充精度验证

在同一份最终修复代码上补充了 310P `FULL_DECODE_ONLY` 实机验证。除图模式外，配置与 Eager 验收组保持一致：Qwen3.6-35B-A3B-w8a8、TP2、DFlash K=15、draft TP2、Prefix Cache On、Async Scheduling、`max-num-seqs=10`；去掉 `--enforce-eager`，增加：

```text
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[160,16]}'
```

图模式真实性证据：

- 引擎解析结果为 `enforce_eager=False`、`CUDAGraphMode.FULL_DECODE_ONLY`；
- 日志完成 `Capturing CUDA graphs (decode, FULL): 2/2`；
- rank0、rank1 manifest 均为 `components=target,draft descriptors=[16, 160]`，证明 target 和 draft 都已注册 FULL graph，而非仅配置图模式后静默回退 Eager；
- Prefix 生命周期探针随后实际发起 decode，请求结果为：

```json
{"boundary_requests": 9, "concurrent_requests": 10, "forced_eos_ignore_tokens": 4, "forced_eos_stop_tokens": 1, "reuse_after_finish_match": true}
```

同一服务连续执行两次 GSM8K 50题：

| 配置 | 运行 | 成功请求 | 官方 accuracy | 推理耗时 | `details` 错误项 |
|---|---|---:|---:|---:|---|
| DFlash + Prefix On + Eager | 对照 | 50/50 | 94.00（47/50） | 1:26 | 2、12、37 |
| DFlash + Prefix On + FULL_DECODE_ONLY | 第1次 | 50/50 | 94.00（47/50） | 1:24 | 2、12、17 |
| DFlash + Prefix On + FULL_DECODE_ONLY | 第2次 | 50/50 | 94.00（47/50） | 1:22 | 2、12、17 |

性能数据需要分成两组理解：

| 对照 | AISBench infer elapsed | 输出 token 总数 | 仅供观察的 `输出 token / elapsed` |
|---|---:|---:|---:|
| 修复前问题复现组：DFlash + Prefix On + Eager | 317.43s | 12948 | 40.79 token/s |
| 修复后：DFlash + Prefix On + Eager | 86.21s | 9762 | 113.24 token/s |
| 修复后：DFlash + Prefix On + FULL_DECODE_ONLY，第1次 | 84.85s | 9888 | 116.54 token/s |
| 修复后：DFlash + Prefix On + FULL_DECODE_ONLY，第2次 | 83.03s | 9717 | 117.03 token/s |

- 修复后两次 FULL_DECODE_ONLY 平均 83.94s，比修复后 Eager 的 86.21s 少 2.63%；在这次 50 题小样本中没有观察到图模式性能下降。1.58%～3.69% 的差值较小，应视为“没有明显回退”，不能据此宣称稳定提升。
- 修复后 Eager 的原始耗时比修复前问题复现组少 72.84%（原始时间比为 3.68 倍），因此没有观察到修复导致的整体变慢。但是这不是严格的纯代码性能 A/B：修复前错误路径只得到 54% 精度，生成了 12948 个输出 token，其中 5 题顶到 512 token；修复后生成 9762 个输出 token，只有 2 题顶到 512 token。错误输出更长会直接拉长旧组耗时。
- 四组输入 token 总数均为 74657，AISBench 关键配置均为 `batch_size=10`、`max_out_len=512`、`temperature=0`、`ignore_eos=False`。表中的 token/s 是用整个 AISBench infer elapsed 粗略相除的端到端观察值，混合了 prefill、decode、并发调度和请求长度差异，不等价于正式 benchmark 的 TTFT、TPOT 或稳态吞吐。

因此，现有结果支持的严谨结论是：**未发现本次修复带来的性能下降；FULL_DECODE_ONLY 相对修复后 Eager 也未见下降。** 若要量化修复本身的性能成本，仍需在固定输出长度、相同前缀命中状态和相同服务热态下做多轮 A/B；本轮精度验收不替代另一条独立 profile 性能任务。

结论：在当前目标 workload 上，FULL_DECODE_ONLY 两次复验均为 94%，相对 Eager 没有汇总精度下降，并且两次图模式运行的最终抽取答案完全一致。图模式与 Eager 不是逐题/token级完全一致：第17题由 Eager 正确变为图模式错误，同时第37题由 Eager 错误变为图模式正确，净精度变化为0；两者有4题抽取答案不同、38题自然语言推导文本不同。由于 AISBench 已明确设置 `temperature=0`，这些差异应作为图/Eager 数值路径差异记录，不能表述为 bitwise 等价；但 Eager 的两次历史运行本身也出现过5题抽取答案、33题文本变化，因此现有证据没有显示 FULL_DECODE_ONLY 引入净精度退化。

两次图模式服务日志均未出现 traceback、RuntimeError、断言失败、coverage mismatch、EngineDead、HTTP 500、ACL 507015、HCCL ERR02005、L0C conflict 或 `QuantBatchMatmulV3` 错误；Prefix Cache hit rate 最终为80.5%。结果目录：

```text
/home/shichuchao/scc_dflash/tmp/prefix_fix_20260904_0205/graph_full_decode_only_20260904_1425/
```

本轮 API PID 3721059 及其 Engine/TP worker 已按精确 PID正常退出，端口6666已释放；没有停止或修改其他 agent 的进程。

### Ascend-only ContextVar 最终复验

为满足“只能修改 vLLM Ascend 仓库”，最终实现移除了 vLLM scheduler 的直接代码修改。310P 平台补丁在构造 DFlash Scheduler 的短暂调用作用域内设置 ContextVar；Ascend KV coordinator 在该作用域中把传入的 `use_eagle=True` 解释为 DFlash KV 策略 False，而 Scheduler 本身的 speculative 属性、K+1 lookahead 和普通 EAGLE 行为不变。作用域使用 ContextVar token 在 `finally` 中恢复，避免初始化异常、嵌套调用或同进程后续 Scheduler 构造继承错误状态。

最终代码在 clean vLLM v0.24.0 overlay 上重新验证：

- vLLM Ascend 相关组合单测：`107 passed, 15 warnings in 0.52s`；
- Ruff：`All checks passed!`；
- 生命周期探针：9 个边界请求、10 个并发请求、EOS/`ignore_eos` 和 finish 后复用全部通过；
- 服务真实配置：TP2、`max-num-seqs=10`、Prefix Cache On、Async Scheduling、DFlash K=15、`FULL_DECODE_ONLY`；
- AISBench：50/50 请求成功，GSM8K 官方 accuracy `96.00`，推理耗时 `1:25`；
- 服务运行期没有 traceback、RuntimeError、断言失败、coverage mismatch、EngineDead 或 HTTP 500。日志中的 Triton import ERROR 是环境未安装可选 Triton kernel 的启动提示，不影响 Ascend 服务启动和本轮推理。

本轮 Prefix Cache hit rate 最终达到 80.5%，证明前缀复用功能没有被关闭。结果目录：

```text
/home/shichuchao/scc_dflash/tmp/prefix_fix_20260904_0205/ascend_only_context_20260904_1515/
```

API PID 3725065 及其 Engine/TP worker 已按精确 PID正常退出，端口 6666 已释放；没有停止或修改其他 agent 的进程。最终提交只包含 vLLM Ascend 仓库改动，vLLM 仓库保持 clean。

### 提交前独立审查补强

独立只读审查没有发现 Critical，并确认 ContextVar token 恢复、Scheduler 参数透传、平台补丁导入顺序和 coordinator cached binding 重绑设计成立。审查提出的兼容性项在提交前补齐：

- fallback 根据原始 `_mamba_block_aligned_split` 签名决定是否传入 v0.24 新增的 common-prefix 参数，保留 v0.23 非 DFlash/Prefix-off 路径；
- DSpark 虽与 DFlash 共用 AscendC input builder，但不再进入 DFlash 专用 per-layer 混合布局转换；DFlash 路径仍保留该转换；
- Scheduler 构造测试改为经过真实 `Scheduler -> KVCacheManager -> patched coordinator factory` 调用链，并覆盖 DFlash、EAGLE 和非 speculative 三组；
- 新增 wrapped Scheduler 初始化异常恢复、ContextVar 嵌套恢复、v0.23 fallback 签名和 DSpark 隔离回归测试；
- 将 310P scheduler 新平台补丁登记到 `vllm_ascend/patch/__init__.py` patch catalog。

审查补强后的最终组合回归为 `107 passed, 15 warnings in 0.52s`。随后用最终代码再次启动 TP2、`max-num-seqs=10`、Prefix Cache On、Async Scheduling、DFlash K=15、`FULL_DECODE_ONLY` 服务，并执行 50 题 AISBench：

- 50/50 请求成功；
- GSM8K 官方 accuracy `94.00`，与无 DFlash + Prefix On 基线的 `94.00` 相同；
- infer 阶段约 `1:36`；
- Prefix Cache hit rate 最终达到 84.0%；
- 服务日志没有运行期 traceback、RuntimeError、断言失败、coverage mismatch、EngineDead 或 HTTP 500。

最终结果目录：

```text
/home/shichuchao/scc_dflash/tmp/prefix_fix_20260904_0205/ascend_only_context_post_review_20260904/
```

本轮 API PID 3729816 及其 Engine/TP worker 已按精确 PID正常退出，端口 6666 已释放；没有停止或修改其他 agent 的进程。

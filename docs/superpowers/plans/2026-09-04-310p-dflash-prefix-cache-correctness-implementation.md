# 310P DFlash Prefix Cache Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留 draft 640-token KV 内存优化，并修复 310P DFlash 在 Prefix Cache、混合 KV 布局和 Async Scheduling 下的状态错位与精度下降。

**Status:** 已实现并验证。最终证据见 `docs/superpowers/plans/2026-09-02-310p-dflash-prefix-probe-result.md` 的“正式修复与端到端验收”章节。

**Execution note:** 下方复选框保留为最初执行清单，不再作为完成状态来源。Task 1-7 的代码、定向测试、静态检查、远程探针和 AISBench 验收均已完成；最终实现只修改 vLLM Ascend，计划中的分步 commit 不执行，全部改动合并为一个本地 commit。

**Architecture:** 将 DFlash speculative 调度语义与 EAGLE KV drop/peek 语义解耦；调度器按绝对 1280-token Mamba 检查点切分；Ascend hybrid coordinator 对全部 FullAttention groups 统一截断并验证覆盖；310P proposer 为不同 kernel block size 成对生成 block table 与 slot mapping。异步 accepted-token 跨流依赖先由空闲 NPU 探针确认，再以设备侧 `wait_event` 修复。

**Tech Stack:** Python 3、pytest、PyTorch/torch-npu、vLLM v1 scheduler/KV cache、vLLM Ascend 310P、AISBench/OpenAI-compatible serving。

**Spec:** `docs/superpowers/specs/2026-09-04-310p-dflash-prefix-cache-correctness-design.md`

## Global Constraints

- draft KV 逻辑块保持 640 token；不得改回 1280 规避问题。
- 目标部署保持 TP2、`max-num-seqs=10`、Prefix Cache 与 Async Scheduling 开启。
- DFlash 保留 K+1 lookahead、proposer 和 rejection verification。
- 最终异步修复只允许设备侧 stream `wait_event`；不得加入 host/global/device synchronize。
- 远程只使用空闲 NPU、独立端口与独立结果目录；不得终止、重启或覆盖其他 agent 的进程与文件。
- 当前 Qwen3.6 全 128 kernel-block 布局必须保持零转换快速路径。
- 所有生产改动先有会失败的定向测试，再做最小实现。

---

### Task 1: 统一截断并验证全部 FullAttention groups

**Files:**

- Modify: `tests/ut/patch/platform/test_prefix_cache_cp_patches.py`
- Modify: `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:255-465`

**Interfaces:**

- Consumes: `self.attention_groups`、`self._get_effective_block_size(spec)`、`hit_blocks_by_group` 和最终 `hit_length`。
- Produces: `AscendHybridKVCacheCoordinator._finalize_common_prefix_hit(...) -> tuple[tuple[list[KVCacheBlock], ...], int]`，同时更新 `num_uncached_common_prefix_tokens`。

- [ ] **Step 1: 写出多 FullAttention group 截断的失败测试**

```python
def _full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )

def test_finalize_common_prefix_truncates_every_full_attention_group() -> None:
    coordinator = _make_coordinator_for_effective_block_size(
        dcp_world_size=1, pcp_world_size=1, enable_caching=True
    )
    coordinator.attention_groups = [
        (_full_spec(1280), [0], MagicMock()),
        (_full_spec(640), [1, 2], MagicMock()),
    ]
    blocks = [[object(), object()], [object()] * 4, [object()] * 4]

    finalized, hit_length = coordinator._finalize_common_prefix_hit(
        blocks, hit_length=1280, longest_hit_length=2560
    )

    assert [len(group) for group in finalized] == [1, 2, 2]
    assert hit_length == 1280
    assert coordinator.num_uncached_common_prefix_tokens == 1280
```

- [ ] **Step 2: 写出覆盖不足必须 fail-fast 的失败测试**

```python
def test_finalize_common_prefix_rejects_undercovered_group() -> None:
    coordinator = _make_coordinator_for_effective_block_size(
        dcp_world_size=1, pcp_world_size=1, enable_caching=True
    )
    coordinator.attention_groups = [
        (_full_spec(1280), [0], MagicMock()),
        (_full_spec(640), [1], MagicMock()),
    ]
    with pytest.raises(RuntimeError, match=r"group_id=1.*expected_blocks=2.*actual_blocks=1"):
        coordinator._finalize_common_prefix_hit(
            [[object()], [object()]], hit_length=1280, longest_hit_length=1280
        )
```

- [ ] **Step 3: 运行测试并确认 helper 缺失**

Run: `python -m pytest tests/ut/patch/platform/test_prefix_cache_cp_patches.py -q -k finalize_common_prefix`

Expected: 两个测试都因 `_finalize_common_prefix_hit` 不存在而失败。

- [ ] **Step 4: 实现一个由两个查找路径共享的 finalizer**

```python
def _finalize_common_prefix_hit(
    self,
    hit_blocks_by_group: list[list[KVCacheBlock] | None],
    hit_length: int,
    longest_hit_length: int | None = None,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    for spec, group_ids, _ in self.attention_groups:
        if not isinstance(spec, FullAttentionSpec):
            continue
        effective_block_size = self._get_effective_block_size(spec)
        if hit_length % effective_block_size:
            raise RuntimeError(
                "FullAttention prefix hit is not block aligned: "
                f"hit_length={hit_length}, block_size={effective_block_size}"
            )
        expected_blocks = hit_length // effective_block_size
        for group_id in group_ids:
            blocks = hit_blocks_by_group[group_id]
            if blocks is None:
                blocks = []
                hit_blocks_by_group[group_id] = blocks
            del blocks[expected_blocks:]
            if len(blocks) != expected_blocks:
                raise RuntimeError(
                    "FullAttention prefix coverage mismatch: "
                    f"group_id={group_id}, hit_length={hit_length}, "
                    f"expected_blocks={expected_blocks}, actual_blocks={len(blocks)}"
                )
    if longest_hit_length is not None:
        self.num_uncached_common_prefix_tokens = longest_hit_length - hit_length
    return tuple(blocks or [] for blocks in hit_blocks_by_group), hit_length
```

在 `find_longest_cache_hit` 初始化并更新 `longest_hit_length`，末尾调用 finalizer；`find_longest_cache_hit_per_group` 也调用同一 finalizer，但不更新 common-prefix 统计。

- [ ] **Step 5: 运行 coordinator 测试与既有 CP/DeepSeek 回归**

Run: `python -m pytest tests/ut/patch/platform/test_prefix_cache_cp_patches.py tests/ut/test_compressed_prefix_cache.py -q`

Expected: 全部通过；既有 EAGLE group 传播和 CP effective block size 行为不变。

- [ ] **Step 6: 提交 coordinator 修复**

```bash
git add tests/ut/patch/platform/test_prefix_cache_cp_patches.py \
  vllm_ascend/patch/platform/patch_kv_cache_coordinator.py
git commit -m "fix(prefix-cache): finalize every full attention group"
```

---

### Task 2: 成对转换混合 64/128 draft block table 与 slot mapping

**Files:**

- Modify: `tests/ut/_310p/spec_decode/test_dflash_proposer_310.py`
- Modify: `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py:140-299,428-535,667-755`

**Interfaces:**

- Consumes: source `BlockTable.get_numpy_array()`、`num_blocks_per_row`、`physical_block_size`、`block_size`，以及每个 draft layer 的真实 cache block size。
- Produces: `_convert_block_table_layout_310(...) -> np.ndarray`；`_dflash_block_table_by_layer_310: dict[str, torch.Tensor]`；原有 query/context per-layer slot mappings 使用匹配表计算。

- [ ] **Step 1: 用非连续物理页写出转换失败测试**

```python
def test_convert_block_table_layout_preserves_physical_pages() -> None:
    source = np.array(
        [[*range(100, 110), *range(40, 50), *([0] * 10)]],
        dtype=np.int32,
    )
    converted = dflash_proposer_310._convert_block_table_layout_310(
        source,
        num_source_blocks_per_row=np.array([20], dtype=np.int32),
        physical_block_size=640,
        source_block_size=64,
        target_block_size=128,
    )
    assert converted[0, :10].tolist() == [50, 51, 52, 53, 54, 20, 21, 22, 23, 24]
    assert np.all(converted[0, 10:] == 0)
```

- [ ] **Step 2: 写出无效连续编码与边界 slot 测试**

```python
def test_convert_block_table_layout_rejects_broken_source_chunk() -> None:
    source = np.array([[100, 102, *range(103, 111)]], dtype=np.int32)
    with pytest.raises(ValueError, match="contiguous logical blocks"):
        dflash_proposer_310._convert_block_table_layout_310(
            source, np.array([10], dtype=np.int32), 640, 64, 128
        )

def test_converted_128_table_maps_second_physical_page_correctly() -> None:
    table = torch.tensor([[50, 51, 52, 53, 54, 20, 21, 22, 23, 24]])
    positions = torch.tensor([0, 639, 640, 1279], dtype=torch.int32)
    req_ids = torch.zeros(4, dtype=torch.long)
    slots = _compute_slots_for_block_size_310(positions, req_ids, table, 128)
    assert slots.tolist() == [6400, 7039, 2560, 3199]
```

- [ ] **Step 3: 运行转换测试并确认 helper 缺失**

Run: `python -m pytest tests/ut/_310p/spec_decode/test_dflash_proposer_310.py -q -k 'convert_block_table_layout or converted_128_table'`

Expected: 转换测试因 helper 不存在失败；边界测试独立通过。

- [ ] **Step 4: 实现 CPU 物理页恢复和目标布局展开**

```python
def _convert_block_table_layout_310(
    source: np.ndarray,
    num_source_blocks_per_row: np.ndarray,
    physical_block_size: int,
    source_block_size: int,
    target_block_size: int,
) -> np.ndarray:
    if physical_block_size % source_block_size or physical_block_size % target_block_size:
        raise ValueError("kernel block sizes must divide the physical block size")
    source_ratio = physical_block_size // source_block_size
    target_ratio = physical_block_size // target_block_size
    max_physical_blocks = source.shape[1] // source_ratio
    result = np.zeros((source.shape[0], max_physical_blocks * target_ratio), dtype=np.int32)
    for row, source_count in enumerate(num_source_blocks_per_row.tolist()):
        if source_count % source_ratio:
            raise ValueError("source logical block count must cover whole physical pages")
        output = []
        for start in range(0, source_count, source_ratio):
            chunk = source[row, start : start + source_ratio]
            base = int(chunk[0])
            if base % source_ratio or not np.array_equal(
                chunk, np.arange(base, base + source_ratio, dtype=np.int32)
            ):
                raise ValueError("source page must contain contiguous logical blocks")
            physical_id = base // source_ratio
            output.extend(range(physical_id * target_ratio, (physical_id + 1) * target_ratio))
        result[row, : len(output)] = output
    return result
```

- [ ] **Step 5: 为每种目标 size 维护持久 host/device buffer**

在 `_prepare_per_layer_slot_mappings_310` 中读取 proposer 对应 source `BlockTable`。若目标 size 等于 source size，直接复用 `cad.block_table_tensor`；否则将转换结果写入按 size 缓存的 pinned host tensor，再 non-blocking copy 到持久 device tensor。若 `source_table.dcp_world_size * source_table.pcp_world_size > 1` 且存在多个 target size，抛出明确的 unsupported-combination `RuntimeError`。

缓存结构固定为：

```python
proposer._dflash_block_table_buffers_by_size_310 = {
    block_size: (host_tensor, device_tensor)
}
proposer._dflash_block_table_by_layer_310 = {
    layer_name: block_tables_by_size[block_sizes_by_layer[layer_name]]
}
```

随后计算 query/context slot 时使用 `block_tables_by_size[block_size]`，不再对所有 size 重用 `cad.block_table_tensor`。

- [ ] **Step 6: 写出 metadata 必须成对替换的失败测试**

```python
@dataclass
class _CommonMetadataStub:
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

def test_per_layer_metadata_replaces_table_and_slots_together(monkeypatch) -> None:
    table64 = torch.tensor([[100, 101]])
    table128 = torch.tensor([[50]])
    slots64 = torch.tensor([6400], dtype=torch.int32)
    slots128 = torch.tensor([6400], dtype=torch.int32)
    proposer = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        attn_layer_names=["layer64", "layer128"],
        _dflash_block_table_by_layer_310={"layer64": table64, "layer128": table128},
        _dflash_query_slot_mapping_by_layer_310={"layer64": slots64, "layer128": slots128},
        runner=SimpleNamespace(get_model=lambda: object()),
    )
    builder = MagicMock()
    builder.build.side_effect = lambda _, metadata, __, **kwargs: metadata
    monkeypatch.setattr(dflash_proposer_310, "is_310p_dflash_full_and_piecewise", lambda _: True)

    result = AscendDflashProposer310._build_first_pass_per_layer_attn_metadata(
        proposer, builder, _CommonMetadataStub(table64, slots64), object(), {}
    )
    assert result["layer64"].block_table_tensor is table64
    assert result["layer64"].slot_mapping is slots64
    assert result["layer128"].block_table_tensor is table128
    assert result["layer128"].slot_mapping is slots128
```

- [ ] **Step 7: 修改 metadata cache key，并运行 proposer 全套测试**

将 cache key 从单独的 slot pointer 改为 `(block_table.data_ptr(), slot_mapping.data_ptr())`，`replace(...)` 同时传入两个字段。

Run: `python -m pytest tests/ut/_310p/spec_decode/test_dflash_proposer_310.py -q`

Expected: 新增物理页、边界、成对 metadata 测试与既有 310P DFlash 测试全部通过；source=target=128 复用原表。

- [ ] **Step 8: 提交映射修复**

```bash
git add tests/ut/_310p/spec_decode/test_dflash_proposer_310.py \
  vllm_ascend/_310p/spec_decode/dflash_proposer_310.py
git commit -m "fix(310p): pair dflash block tables with slot mappings"
```

---

### Task 3: 补齐绝对 Mamba 检查点调度边界

**Files:**

- Modify: `tests/ut/_310p/test_dflash_mamba_scheduler_310.py:20-97`
- Modify: `vllm_ascend/patch/platform/patch_mamba_scheduler_310.py:26-35`

**Interfaces:**

- Consumes: `Scheduler.block_size` 作为 target/Mamba checkpoint size，`cache_config.enable_prefix_caching` 和 `speculative_config.use_dflash()` 作为启用条件。
- Produces: `_needs_dflash_mamba_checkpoint_split(scheduler: Scheduler) -> bool`；`Scheduler._mamba_block_aligned_split(...) -> int` 在最近的下一个绝对 checkpoint 截断。

- [ ] **Step 1: 参数化测试构造器，使 scheduler/cache block size 可独立配置**

```python
def _make_scheduler(
    *,
    method: str = "dflash",
    prefix_caching: bool = True,
    scheduler_block_size: int = 1280,
    cache_block_size: int = 640,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = scheduler_block_size
    scheduler.cache_config = SimpleNamespace(
        block_size=cache_block_size,
        enable_prefix_caching=prefix_caching,
    )
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=_SpeculativeConfig(method),
    )
    scheduler.use_eagle = True
    return scheduler
```

- [ ] **Step 2: 写出相等 block size 和检查点边界的失败测试**

```python
@pytest.mark.parametrize(
    ("computed", "requested", "expected"),
    [(0, 640, 640), (640, 820, 640), (1280, 180, 180),
     (1200, 200, 80), (1280, 1400, 1280), (2560, 100, 100)],
)
@pytest.mark.parametrize("cache_block_size", [640, 1280])
def test_dflash_split_stops_at_absolute_mamba_checkpoint(
    computed, requested, expected, cache_block_size
):
    scheduler = _make_scheduler(cache_block_size=cache_block_size)
    request = _make_request(num_tokens=4000, num_computed_tokens=computed)
    assert scheduler._mamba_block_aligned_split(request, requested) == expected
```

- [ ] **Step 3: 运行测试并确认相等 block size 用例为红**

Run: `python -m pytest tests/ut/_310p/test_dflash_mamba_scheduler_310.py -q`

Expected: `cache_block_size=1280` 的跨 checkpoint 用例失败，旧条件返回未截断值。

- [ ] **Step 4: 最小化启用条件，不再依赖 1280/640 的大小差**

```python
def _needs_dflash_mamba_checkpoint_split(scheduler: Scheduler) -> bool:
    return (
        scheduler.cache_config.enable_prefix_caching
        and _uses_dflash(scheduler)
        and scheduler.block_size > 0
    )
```

保留 `_dflash_mamba_block_aligned_split` 中基于 `self.block_size` 的绝对位置公式，以及非 DFlash/Prefix-off 回退上游逻辑。

- [ ] **Step 5: 运行调度测试并确认全绿**

Run: `python -m pytest tests/ut/_310p/test_dflash_mamba_scheduler_310.py -q`

Expected: 全部通过；W1460 仍为 `[1280, 180]`，EAGLE 与 Prefix-off 对照行为不变。

- [ ] **Step 6: 提交该独立修复**

```bash
git add tests/ut/_310p/test_dflash_mamba_scheduler_310.py \
  vllm_ascend/patch/platform/patch_mamba_scheduler_310.py
git commit -m "fix(310p): split dflash prefill at mamba checkpoints"
```

---

### Task 4: 解耦 DFlash speculative 与 EAGLE KV drop/peek

**Files:**

- Create: `vllm_ascend/patch/platform/dflash_kv_context.py`
- Modify: `vllm_ascend/patch/platform/patch_mamba_scheduler_310.py`
- Modify: `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
- Modify: `tests/ut/_310p/test_dflash_mamba_scheduler_310.py`
- Modify: `tests/ut/patch/platform/test_prefix_cache_cp_patches.py`

**Interfaces:**

- Consumes: `SpeculativeConfig.use_eagle()` 与 `SpeculativeConfig.use_dflash()`。
- Produces: `dflash_scheduler_init_scope()` 与 `resolve_kv_use_eagle(use_eagle) -> bool`；`Scheduler.use_eagle` 仍控制 speculative 行为，Ascend coordinator 只在真正的 EAGLE/MTP 语义下启用 KV 特例。

- [ ] **Step 1: 写出 coordinator 调用上下文的失败测试**

```python
with dflash_scheduler_init_scope():
    coordinator = get_kv_cache_coordinator(..., use_eagle=True)

assert captured_use_eagle is False
```

- [ ] **Step 2: 运行测试并确认上下文 API 尚不存在**

Run: `python -m pytest tests/ut/patch/platform/test_prefix_cache_cp_patches.py -q -k dflash_scheduler_context`

Expected: 失败并指出 `dflash_kv_context` 尚不存在。

- [ ] **Step 3: 加入 ContextVar，并在 Ascend coordinator 入口解析有效策略**

```python
_DFLASH_SCHEDULER_INIT = ContextVar("dflash_scheduler_init", default=False)

def resolve_kv_use_eagle(use_eagle: bool) -> bool:
    return use_eagle and not _DFLASH_SCHEDULER_INIT.get()
```

310P Scheduler 初始化包装只在 DFlash 配置下设置该上下文，并在 `finally` 中用 token 恢复。vLLM 原始 `Scheduler.__init__` 完全不修改：

```python
with dflash_scheduler_init_scope():
    return _original_scheduler_init(self, vllm_config, *args, **kwargs)
```

- [ ] **Step 4: 运行定向与异常恢复回归测试**

Run: `python -m pytest tests/ut/_310p/test_dflash_mamba_scheduler_310.py tests/ut/patch/platform/test_prefix_cache_cp_patches.py -q`

Expected: DFlash 的 Scheduler `use_eagle=True`、K+1 lookahead 不变、coordinator 有效策略为 False；普通 EAGLE 仍为 True；初始化异常后上下文恢复。

- [ ] **Step 5: 纳入最终 vLLM Ascend 单仓 commit**

本 Task 不单独提交，与其余修复合并为最终一个 commit；vLLM 仓库保持 clean。

---

### Task 5: 用空闲 NPU 判定并修复 accepted-token 跨流依赖

**Files:**

- Create: `tools/probes/probe_310p_async_event_visibility.py`
- Modify: `tests/ut/_310p/test_model_runner_310p.py`
- Modify when the probe confirms missing ordering: `vllm_ascend/_310p/model_runner_310p.py:652-659`

**Interfaces:**

- Consumes: 上一轮在 `global_stream()` 上 record 的 `num_accepted_tokens_event`。
- Produces: `_wait_for_accepted_tokens_event_310(runner, use_async_device_metadata: bool) -> None`；下一轮 default/current stream 在读取 accepted-token 及 Mamba state 前拥有 happens-before。

- [ ] **Step 1: 创建只使用当前可见单卡的事件探针**

探针固定执行三组各 50 轮：producer stream 先执行矩阵乘制造短延迟，再写递增标记并 record event；current stream 分别无等待读取、`wait_event` 后读取和 host synchronize 后读取。输出 JSON：`no_wait_stale`、`wait_event_stale`、`host_sync_stale`，并在后两者非零时退出失败。

核心执行顺序：

```python
with torch.npu.stream(producer):
    torch.mm(delay_left, delay_right)
    marker.fill_(iteration)
    event.record()
if mode == "wait_event":
    torch.npu.current_stream().wait_event(event)
elif mode == "host_sync":
    event.synchronize()
observed.copy_(marker)
torch.npu.synchronize()
```

- [ ] **Step 2: 远程只读检查并选择空闲 NPU**

Run on host:

```bash
npu-smi info
docker top scc_dflash_dev -eo pid,ppid,stat,etime,args
ss -ltnp
```

Expected: 记录其他 agent 的进程、设备和端口；选择没有计算进程的单卡，不 kill 任何来源不明进程。

- [ ] **Step 3: 在容器独立目录运行探针并按固定规则判定**

Run:

```bash
ASCEND_RT_VISIBLE_DEVICES=<free-device> python3 \
  /home/shichuchao/scc_dflash/tmp/prefix_fix_<timestamp>/probe_310p_async_event_visibility.py
```

Decision:

- `no_wait_stale > 0` 且 `wait_event_stale == 0`：执行 Steps 4-7。
- `no_wait_stale == 0`：检查 stream trace；若没有框架等价依赖，仍执行 Steps 4-7，因为静态 happens-before 缺失。
- `wait_event_stale > 0`：停止异步代码修改，保存探针输出并先修正 torch-npu event 用法。

- [ ] **Step 4: 写出 device-side wait helper 的失败测试**

```python
def test_wait_for_accepted_tokens_event_uses_current_stream(monkeypatch) -> None:
    stream = MagicMock()
    event = object()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    runner = SimpleNamespace(num_accepted_tokens_event=event)
    model_runner_310p._wait_for_accepted_tokens_event_310(runner, True)
    stream.wait_event.assert_called_once_with(event)

def test_wait_for_accepted_tokens_event_skips_non_async_path(monkeypatch) -> None:
    stream = MagicMock()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    runner = SimpleNamespace(num_accepted_tokens_event=object())
    model_runner_310p._wait_for_accepted_tokens_event_310(runner, False)
    stream.wait_event.assert_not_called()
```

- [ ] **Step 5: 实现 helper 并在 `_prepare_inputs` fast-path 判定后调用一次**

```python
def _wait_for_accepted_tokens_event_310(runner, use_async_device_metadata: bool) -> None:
    event = runner.num_accepted_tokens_event
    if use_async_device_metadata and event is not None:
        torch.npu.current_stream().wait_event(event)
```

调用点紧跟 `use_async_device_metadata = (...)`，早于 accepted-token、positions、slot mapping 和 Mamba state 的任何消费。

- [ ] **Step 6: 运行 310P model-runner 测试和静态同步检查**

Run: `python -m pytest tests/ut/_310p/test_model_runner_310p.py -q -k 'accepted_tokens_event or async or dflash'`

Run: `rg -n "synchronize\(" vllm_ascend/_310p/model_runner_310p.py`

Expected: helper 测试通过；没有新增 `.synchronize()`。

- [ ] **Step 7: 提交探针与异步依赖修复**

```bash
git add tools/probes/probe_310p_async_event_visibility.py \
  tests/ut/_310p/test_model_runner_310p.py \
  vllm_ascend/_310p/model_runner_310p.py
git commit -m "fix(310p): order async dflash accepted-token updates"
```

---

### Task 6: 运行组合回归并检查实现不变量

**Files:**

- Modify only when a regression identifies a defect in Tasks 1-5.

**Interfaces:**

- Consumes: Tasks 1-5 的代码与测试。
- Produces: CPU 单测证据、静态不变量证据和可同步到远程的最小文件列表。

- [ ] **Step 1: 运行 vLLM scheduler 定向测试**

Run from vLLM repo: `python -m pytest tests/v1/core/test_scheduler.py -q -k 'eagle_kv_cache_policy or speculative or prefix'`

Expected: PASS。

- [ ] **Step 2: 运行 vLLM Ascend 定向测试集合**

```bash
python -m pytest \
  tests/ut/_310p/test_dflash_mamba_scheduler_310.py \
  tests/ut/patch/platform/test_prefix_cache_cp_patches.py \
  tests/ut/test_compressed_prefix_cache.py \
  tests/ut/_310p/spec_decode/test_dflash_proposer_310.py \
  tests/ut/_310p/test_model_runner_310p.py -q
```

Expected: PASS。

- [ ] **Step 3: 执行静态不变量检查**

```bash
rg -n "synchronize\(" vllm_ascend/_310p/model_runner_310p.py
rg -n "resolve_kv_use_eagle|dflash_scheduler_init_scope" \
  vllm_ascend/patch/platform
rg -n "_finalize_common_prefix_hit" \
  vllm_ascend/patch/platform/patch_kv_cache_coordinator.py
```

Expected: 不新增 async fast-path synchronize；DFlash 上下文只在 310P Scheduler 初始化期间设置；coordinator 两个返回路径共享 finalizer。

- [ ] **Step 4: 记录 vLLM Ascend commit/status，并确认 vLLM 仓库 clean**

```bash
git status --short
git log -5 --oneline
```

Expected: 只列出本方案文件或预先存在、已记录的用户改动；不得将无关文件纳入同步或提交。

---

### Task 7: 远程 Prefix Cache 小实验与 AISBench 精度验收

**Files:**

- Modify: `docs/superpowers/plans/2026-09-02-310p-dflash-prefix-probe-result.md`
- Create under remote result directory: launch logs、request/response JSONL、AISBench outputs、comparison summary。

**Interfaces:**

- Consumes: Tasks 1-6 的已验证代码、`scc_dflash_dev` 容器、Qwen3.6 target/draft 模型和固定 GSM8K 50 prompts。
- Produces: Prefix hit 内容/块覆盖证据、三组精度和逐题差异；最终结果文档。

- [ ] **Step 1: 再次检查远程资源，选择 TP2 空闲设备与独立端口**

```bash
npu-smi info
docker top scc_dflash_dev -eo pid,ppid,stat,etime,args
ss -ltnp
```

Expected: 避开另一 agent 使用的设备和服务；在 `/home/shichuchao/scc_dflash/tmp/prefix_fix_<timestamp>` 新建结果目录。

- [ ] **Step 2: 只同步本方案改动并确认实际 import 路径**

在远程隔离 worktree `/home/shichuchao/scc_dflash/prefix_probe_wt` 应用精确补丁；进入容器后运行：

```bash
python3 - <<'PY'
import inspect
import vllm.v1.core.sched.scheduler as scheduler
import vllm_ascend.patch.platform.patch_kv_cache_coordinator as coordinator
print(inspect.getfile(scheduler))
print(inspect.getfile(coordinator))
PY
```

Expected: import 指向隔离 worktree，不覆盖 `/home/shichuchao/scc_dflash/vllm-ascend` 的其他 agent 工作副本。

- [ ] **Step 3: 运行两请求 Prefix Cache 小实验**

使用 TP2、Eager、DFlash K=15、Prefix Cache on、独立端口启动临时服务。发送共享至少 1280-token 前缀但后缀不同的 W/H 请求，并覆盖 639/640、1279/1280、1460、2559/2560 边界。

Expected:

- W1460 调度事件为 `1280 + 180`。
- H 请求命中前缀后，target/draft/Mamba 的返回块数与 hit length 一致。
- 命中范围 block ids 复用，prefix 内容摘要不变。

- [ ] **Step 4: 开启 Async Scheduling 运行生命周期矩阵**

发送全接受、部分拒绝、全拒绝、EOS、`ignore_eos`、finish/free/reuse 和并发 10 请求。

Expected: 无 accepted-token 旧值、Mamba checkpoint 跳跃、block reuse 污染、NPU event 错误或覆盖断言；Async off/on 的确定性答案一致。

- [ ] **Step 5: 固定输入依次运行三组 50 题 AISBench**

三组服务除 speculative/prefix 开关外保持模型、端口外参数、随机种子、chat template 和 generation 参数一致：

1. no-DFlash baseline；
2. DFlash + Prefix Cache off；
3. fixed DFlash + Prefix Cache on + Async Scheduling on。

每组运行：

```bash
ais_bench --models vllm_api_general_chat \
  --datasets gsm8k_gen_4_shot_cot_str \
  --num-prompts 50
```

- [ ] **Step 6: 生成聚合与逐题差异报告**

Expected:

- fixed Prefix-on 精度不低于 no-DFlash baseline，也不低于 Prefix-off 控制组。
- 对 `baseline=correct, fixed=wrong` 的每题保存 prompt、reference、两组输出；存在未解释新增回归则继续定位，不宣布完成。
- 日志无覆盖断言、NPU async 错误或 KV 生命周期错误。

- [ ] **Step 7: 更新结果文档并停止本次临时服务**

记录代码 commit、设备、端口、启动命令、探针 JSON、单测摘要、三组 AISBench 指标、逐题差异和远程产物路径。只停止本次记录 PID 的临时服务，确认端口与设备释放；不得操作另一 agent 进程。

- [ ] **Step 8: 提交验证文档**

```bash
git add docs/superpowers/plans/2026-09-02-310p-dflash-prefix-probe-result.md
git commit -m "docs: record dflash prefix-cache correctness validation"
```

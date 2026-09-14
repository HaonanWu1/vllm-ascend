# 310P DFlash m-RoPE：最新分支集成与提交说明

本次在 `HaonanWu1/vllm-ascend` 的 `dev_v24/feat-310p-dflash` 最新代码上集成 m-RoPE 及压缩词表支持。上游 vLLM、SpecForge、模型权重未修改。

## 代码基线与隔离工作区

- 远端基线：`5469c180`（2026-09-14 拉取）。
- 原 m-RoPE 本地提交：`9208f4e29bc6c79fe80f4181ffa55068a903a1bd`，原基线 `f48bd99`。
- 新工作区：容器 `qzh_v023_vllm024_dflash` 内 `/vllm-workspace/vllm-ascend-dflash-mrope-submit`。
- 本地分支：`feat/310p-dflash-mrope-submit`；交付目标：上述远端原分支。
- 原工作区及未提交改动完整保留；备份引用：`backup/dflash-mrope-before-submit-20260914`。
- 本次原生算子源码未变，测试复用原工作区已编译的扩展和 CANN 自定义算子；通过 `PYTHONPATH` 加载新工作区的 Python 代码。生成的二进制和链接不纳入提交。

## 修改内容

1. 按草稿配置启用 m-RoPE，通过实际 310P proposer 方法绑定接入；保留普通 RoPE 和相邻投机方法的入口。
2. 分离 context/query 三轴旋转位置与一维 KV 缓存顺序位置，按请求排列、有效长度和 m-RoPE delta 构造文字 query 坐标。
3. 草稿实例持有独立 context/query cos/sin 缓冲，支持交错频率、图预热、捕获与回放刷新；不修改主模型旋转缓存。
4. 对仅含词表映射、未保存 LM head 的压缩词表 checkpoint，从主模型选择对应输出权重，按 TP 规则加载，并重建 310P NZ 权重缓存。已有独立输出头和普通全词表行为保留。
5. 增加位置、旋转、patch 接入、词表映射和缓存回归测试，以及数值对齐、TextVQA、图模式验证工具和中文报告。

## 与远端新提交的兼容处理

保留这三个远端提交：

| 提交 | 内容 |
| --- | --- |
| `61a51f3f` | 捕获完整 FAP DFlash 草稿执行 |
| `289586be` | 使 FAP 草稿 query padding 槽位失效 |
| `5469c180` | 使 FDO 草稿 context padding 槽位失效 |

冲突涉及 `llm_base_proposer_310.py` 的图包装安装方法。采用远端完整草稿图捕获逻辑，没有恢复旧的局部 forward 捕获实现。

另外，m-RoPE 分支会提前返回，必须显式保留 FDO context padding 失效处理，否则回放可能向有效历史槽位写入 padding KV。本次已补齐，并新增三种图运行条件下的回归测试。一个既有普通 RoPE 测试补齐了远端新逻辑所需的槽位缓冲 fixture。

验证工具新增 `--mode full-and-piecewise`；`--mode full` 仍表示 `FULL_DECODE_ONLY`。

## 本次检查

- 18 个 310P DFlash 相关单元测试文件：**439 passed**，14 条既有弃用警告。
- 改动 Python 文件语法检查、Ruff check/format 通过。
- 改动文件 Markdown、codespell、禁止导入、with 条件和长函数检查通过。
- `format.sh ci` 已尝试；准备 actionlint 依赖时长期未完成，停止该次依赖安装，逐项执行与本次文件有关的已安装检查。**未宣称全仓格式流水线通过**。
- 生产代码新增部分未引入临时诊断日志、`.cpu()`、`.item()`、`.tolist()` 或同步探针。
- 原生 NPU 复测结果见下方记录。初次新工作区测试的 CANN 路径和自定义算子链接问题，归档为环境准备失败，不计作通过。

## 本次真实权重复测结果

Qwen3-VL-4B + 230k 压缩词表草稿，FP16、TP=1、K=7：

| 模式 | 两轮输出 token 序列 | 结束原因 | 进程退出 |
| --- | --- | --- | --- |
| eager | 8/8 与历史主模型基线一致 | 一致 | 0 |
| PIECEWISE | 8/8 与本次 eager 一致 | 一致 | 0 |
| FULL_DECODE_ONLY | 8/8 与本次 eager 一致 | 一致 | 0 |
| FULL_AND_PIECEWISE | 8/8 与本次 eager 一致 | 一致 | 0 |

四种模式的累计指标相同：354 次草稿验证，提出 2478 个候选，接受 384 个；这些计数包含一次预热及两次记录运行，不是 TextVQA 指标。每轮输出长度为 `[60,64,60,64]`，运行日志无引擎 ERROR。

PIECEWISE 和 FULL_DECODE_ONLY 日志包含主模型与草稿的捕获/回放记录。FAP 日志包含 FULL 路由、五层草稿的 device-metadata-capture-source 和 device-metadata-refresh；核对了真实回放，而非仅确认命令能启动。

FAP 使用远端已有的显式配置：

```json
{
  "ascend_compilation_config": {
    "dflash_full_and_piecewise_capture_config": {
      "piecewise_capture_size": 2048,
      "full_capture_size": [8, 64]
    }
  }
}
```

工具会在 `--mode full-and-piecewise` 时自动提供该配置。仅指定上游枚举而不提供这项现有配置，会导致专用路径未启用并在 dummy run 发生模式不匹配；首次失败已保留在 `full-and-piecewise-missing-portfolio`，随后补齐配置通过。没有因此更改生产图分发逻辑。

## 启动及测试方式

容器内执行，保留 CANN 设置的 `PYTHONPATH`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm-workspace/vllm-ascend-dflash-mrope-submit:/vllm-workspace/vllm:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=6
cd /vllm-workspace/vllm-ascend-dflash-mrope-submit

python -m tools.run_310p_dflash_mrope_validation \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --draft /home/xj/checkpoints/qwen3-vl-4b-dflash-mixed300000-vocab32000-epoch3-6gpu-bs1-5layer-block8-gamma4-lr1e-4/epoch_5_step_192260 \
  --mode full --num-speculative-tokens 7 \
  --batch 4 --max-tokens 64 --repeats 2 \
  --image --mixed --images-per-prompt 2 --prompt-repeat 4 \
  --batched-tokens 2048 --gpu-memory-utilization 0.45 \
  --kv-cache-memory-bytes 1073741824 --respect-eos \
  --output /tmp/dflash-mrope-full.json
```

同一工具分别使用 `eager`、`piecewise`、`full`、`full-and-piecewise`。它会启动真实模型、预热一次，再运行两轮，保存输出 token IDs、结束原因和指标。输入交替包含文字和两张相同的合成红方块图；不代表不同尺寸真实多图的完整覆盖。精度 FP16、TP=1、temperature=0。

本次日志、精确命令、失败重试和结果目录：

`/home/qzh/vllm-ascend-dflash-mrope/artifacts/mrope-submit-5469-20260914`

TextVQA 服务启动、HTTP 测试方式和完整 1K 结果参见 [230k 图模式回归报告](dflash_230k_textvqa_graph_regression_zh.md)。

### 普通 RoPE 相邻路径补测

Qwen3.5-4B + Qwen3.5-4B-DFlash，FP16、TP=1、K=15、FAP，4 个文字请求、max_tokens=32、两轮记录运行。8 条记录均成功，重复运行 token IDs 完全一致，进程退出码 0。包含预热的指标为 156 次草稿验证、2340 个候选、234 个接受，抢占 0；日志确认草稿使用普通 `AscendRotaryEmbedding310`，主模型仍为多模态模型。

这项仅为相邻路径冒烟，没有新增同配置主模型独立对照，不将重复稳定性等同于完整正确性证明。测试使用物理设备 2、4 GiB KV 预算；最初设备 6 被其他任务占用，随后 1 GiB KV 预算不足的两次启动失败均独立归档，没有混入成功结果。

复现时在上述工具命令中使用：

```bash
--model /home/models/Qwen3.5-4B \
--draft /home/models/Qwen3.5-4B-DFlash \
--mode full-and-piecewise --num-speculative-tokens 15 \
--batch 4 --max-tokens 32 --repeats 2 --prompt-repeat 4 \
--batched-tokens 2048 --gpu-memory-utilization 0.45 \
--kv-cache-memory-bytes 4294967296 --respect-eos \
--output /tmp/dflash-ordinary-k15-fap.json
```

不添加 `--image`、`--mixed` 或 `--images-per-prompt`，设置 `ASCEND_RT_VISIBLE_DEVICES=2`。完整实际命令见 `ordinary-qwen35-k15-fap/command.json` 和 `run.sh`。

## 已知限制与历史结果边界

- 此前 230k TextVQA 1K：平均接受长度 `1 + accepted / verification = 2.39383`，接受率 19.91186%，单次同设备主模型/投机平均时延比约 1.30105×。这些是旧基线测量，不当作新基线性能结果。
- 旧基线 1K 中有 25 条贪心输出与主模型不同；其中三个样本在 eager 中也可复现。根因未最终归属，不能宣称完整贪心正确性验收通过。
- 96k 重传文件仍与此前异常文件 SHA256 相同，包含 17 个 NaN，未作为本次验收模型。
- 本次短回归不能代替全模型、TP=2、采样、视频和多轮数据的完整验收，也没有据此宣称所有旧模型吞吐下降小于 5%。

# 310P DFlash m-RoPE：最新容器验证记录

验证日期：2026-09-10。本材料记录最新代码与重建容器上的复验，历史 `47dbcb3` 回归不计入本轮通过数量。

**结论：在本轮 TP=1、FP16、文字/图像输入及已测执行模式范围内，未发现需要修正的 m-RoPE 生产代码缺陷。** 393 项单元测试、132 项 NPU 算子测试、独立三轴位置与旋转参考、实际草稿首块逐层对齐，以及 6 组生成对照均通过。性能、TP=2 和其他未复测范围见下文，不将功能验证等同于完整性能验收。

## 环境与代码

| 项目 | 本轮实际值 |
| --- | --- |
| 容器 | `qzh_v023_vllm024_dflash`，重建后的 privileged 容器 |
| 代码目录 | `/vllm-workspace/vllm-ascend-dflash-mrope-latest` |
| 分支 | `feat/310p-dflash-mrope-latest`，改动未提交 |
| Ascend 基线 | `f48bd99fd7b9458bbad9c61f36a10b2217eb40b3`，叠加本次 m-RoPE 修改 |
| vLLM | `0.24.0+empty`，源目录 `/vllm-workspace/vllm`；此前核对的 DFlash 源文件与 `ee0da84` 一致 |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0` |
| Transformers | `5.5.4` |
| 硬件、精度与并行 | Ascend 310P3，FP16，TP=1 |
| 主模型 | `/home/models/Qwen3-VL-4B-Instruct` |
| 草稿 checkpoint | `/home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750` |
| 草稿结构 | 5 层，目标特征层 `[3,10,18,25,32]`，交错 m-RoPE `[24,20,20]`，块大小 8、7 个候选 token |

生产文件 SHA256 和实际 Python 导入位置保存在 `runtime-source.json`。本轮未修改上游 vLLM、SpecForge 或模型权重。

## 自动测试与数值参考

| 检查 | 结果 |
| --- | --- |
| DFlash、相邻投机路径、注意力、旋转、采样及图执行单元测试 | 393 项通过 |
| NPU 输入展开与槽位算子测试 | 132 项通过，包含候选长度 7/15、物理 block size 64/128 和拒绝边界 |
| Hugging Face 三轴位置参考 | 两种图像尺寸 `224×224`、`448×112`；位置整数完全一致 |
| NPU Q/K 旋转与 HF 参考 | 两组 Q、K 最大绝对误差均为 0 |
| 语法、Ruff、格式与 Git 差异检查 | 通过；验证脚本最后一次修改后另做检查 |

零接受、部分接受、全部接受的整数位置及槽位边界由单元测试和算子测试覆盖；不将这些结果表述为真实 checkpoint 已自然产生全部接受。

实际图像请求首块包含 88 个 context token 和 8 个 query token。将相同的主模型特征、噪声 embedding 和三轴位置输入 checkpoint 保存的 `dflash.py`，严格加载 checkpoint 权重，以 CPU FP16 计算作为参考：

| 对齐位置 | 最大绝对误差 | 相对 RMS 误差 | 余弦相似度 |
| --- | --- | --- | --- |
| 草稿层 0 | 0.03125 | 0.09793% | 0.99999732 |
| 草稿层 1 | 0.06250 | 0.08325% | 0.99999666 |
| 草稿层 2 | 0.06250 | 0.08090% | 0.99999732 |
| 草稿层 3 | 0.06250 | 0.07858% | 0.99999756 |
| 草稿层 4 | 0.09375 | 0.07657% | 0.99999666 |
| 最终归一化输出 | 0.0078125 | 0.08109% | 0.99999815 |

全部满足验证程序原有阈值：相对 RMS 误差小于 2%、余弦相似度大于 0.999。该结果覆盖实际首块的投影、K 归一化与旋转、缓存写入后注意力、Q/K 旋转及五层计算；不将首块对齐扩大表述为全部请求、全部步数的逐层对齐。

## 生成与调度对照

所有对照使用贪心生成、最大模型长度 2048。每个进程先预热一次，再记录一次输出。比较的是输出 token ID 和结束原因，不仅是显示文本。

| 对照 | 配置 | 结果 |
| --- | --- | --- |
| eager | 3 个图文混合请求，各生成 32 token | token 与结束原因完全一致 |
| FULL_DECODE_ONLY | 同上 | token 与结束原因完全一致 |
| PIECEWISE | 同上 | token 与结束原因完全一致，实际图捕获与回放成功 |
| prefix cache + chunked prefill | 3 个双图请求，各生成 32 token | token 与结束原因完全一致，实际缓存命中 |
| 异步调度 | 3 个图像请求，各生成 32 token | token 与结束原因完全一致 |
| FULL_DECODE_ONLY + EOS | 4 个图文混合请求，最多 256 token | 输出长度均为 `[66,256,66,256]`，结束原因均为 `[stop,length,stop,length]` |

合计 6 组、19 个请求对照，1,124 个记录轮次的输出 token 全部一致。EOS 用例的活动请求数从 4 变为 2；日志确认主模型与草稿均完成图捕获，共 4 个图实例，描述符容量为 `[8,64]`。

其中 eager 与常规 FULL_DECODE_ONLY 的 32-token 对照复用重建容器后、同一 `f48bd99` 加补丁源码上的实测结果；其余用例在本轮重新启动进程执行。验证脚本的 `--mode full` 明确对应 `FULL_DECODE_ONLY`，不代表所有 FULL 类组合。

双图缓存用例使用 3 个请求、每个请求 2 张图片、扩展文本提示和 64-token 分块预算。实际 prefix cache 查询 1,312 token、命中 1,024 token，命中率约 78.05%，抢占次数为 0。

PIECEWISE 图文混合用例与此前同请求的 eager、FULL_DECODE_ONLY 接受指标一致：累计提出 1,092 个候选、接受 36 个，各位置接受数为 `[32,4,0,0,0,0,0]`。这些计数包含预热与记录轮次。约 3.30% 的接受率说明草稿确实参与了推理，同时也说明当前 checkpoint 在该样例上接受率较低；不能据此承诺加速收益。

## 普通 RoPE 兼容性与范围

重建后的同一容器已完成以下普通 RoPE 草稿冒烟测试，均成功生成：

| 组合 | 已测配置 |
| --- | --- |
| Qwen3-8B + Qwen3-8B-DFlash | eager、TP=1、候选 7、batch=2、文字输入、16-token 输出 |
| Qwen3.5-4B + Qwen3.5-4B-DFlash | eager、TP=1、候选 15、batch=2、文字输入、16-token 输出；草稿包含滑窗与全注意力层 |

普通 RoPE 的位置降维、m-RoPE 的三轴保留、实际 proposer 方法绑定、DSpark 分支隔离和实例旋转缓存隔离均包含在单元测试中。生产加载日志确认本 checkpoint 继续复用主模型的 embedding 与 LM head。

本轮没有在最新基线上重新执行 TP=2、普通模型全部候选长度与图模式矩阵、FULL_AND_PIECEWISE 实机组合、视频、完整 TextVQA 数据集、长于 2048 的上下文或采样模式实机统计回归。上述普通模型完整矩阵及 TP=2 的历史记录见 `dflash_mrope_310p_regression_zh.md`，不将历史结果计作本次最新源码验收。

## 环境失败与验证脚本调整

首次模型测试在主模型对照组启动阶段遇到三种环境限制：空闲显存低于 0.75 的预留预算、其他任务释放显存引发内存统计断言，以及共享设备上自动估算出的 KV 缓存空间为负。失败均发生在未启用 DFlash 的对照进程中，原始日志已保留。

后续生成对照改用物理设备 6，并对主模型组与草稿组统一设置 `gpu_memory_utilization=0.45`、`kv_cache_memory_bytes=536870912`（512 MiB）。固定缓存预算沿用框架现有接口，没有修改内存统计实现，也没有停止其他容器或任务。

本轮仅增强 `tools/run_310p_dflash_mrope_validation.py`：增加可选显存预算、可选固定 KV 缓存大小，并保存 prefix cache 与抢占计数。原默认显存比例仍为 0.75，固定缓存大小默认不设置；不新增生产启动必填参数。

本次设备为共享环境，记录的耗时不用于吞吐回退或加速比验收。逐层数值采集仅通过显式诊断参数开启，生产默认路径没有新增诊断日志或 NPU→CPU 同步探针。

## 复现与证据位置

容器和宿主机均可访问本轮原始证据目录：

`/home/qzh/vllm-ascend-dflash-mrope/artifacts/mrope-latest-validation`

- `queue-4.json`：参考数值、132 项算子测试和 393 项单元测试的完整命令。
- `queue-6.json`：最终采用的模型对照和逐层对齐命令，包含完整模型路径、请求参数和缓存预算。
- `hf-reference.json`、`layers.json`、`first-block.pt`：数值参考结果及实际首块输入快照。
- `comparison.json`：逐 token、结束原因及跨图模式接受指标的汇总比较。
- `*-exit.json`、`*.log`：各进程退出码与原始日志。
- `static-checks.json`、`validator-final-static-checks.json`、`runtime-source.json`：静态检查、版本和源码身份。

此前重建容器的 eager/FULL 对照和普通 RoPE 冒烟证据位于相邻的 `artifacts/container-latest` 目录，以 `recreated-` 开头。

执行命令时使用 `python -m tools.<模块名>`，从当前 Ascend 工作树根目录运行，并先加载 CANN 环境。数值参考和层级对齐的完整参数已经记录在队列 JSON 中。

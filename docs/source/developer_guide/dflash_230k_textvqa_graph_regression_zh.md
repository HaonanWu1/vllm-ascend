# 230k DFlash：TextVQA 加速比与图模式回归

日期：2026-09-14。按用户最新要求，本轮聚焦 TextVQA；OCR-VQA、Mixed 单轮和多轮暂不测试。

## 结果

**两组 TextVQA 均完成 1000/1000，零失败；本轮 230k DFlash 客户端耗时加速比为 1.3011×。六项功能回归及抽样图模式检查通过，但完整 1K 贪心回答文本只有 975/1000 完全一致，不能宣称全量贪心一致性通过。**

| 指标 | 主模型单独推理 | 230k DFlash |
| --- | ---: | ---: |
| 成功 / 失败 | 1000 / 0 | 1000 / 0 |
| 平均客户端耗时 | 2.5993 秒 | 1.9979 秒 |
| 平均服务端耗时 | 2.5950 秒 | 1.9938 秒 |
| 客户端输出吞吐 | 13.0072 token/s | 16.9255 token/s |
| 输出 token 总数 | 33810 | 33815 |
| 接受候选 / 提出候选 | — | 19338 / 97118 |
| 候选接受率 | — | 19.9119% |
| 平均接受长度：1 + 接受 / 验证 | — | 2.3938 |
| 输出 token / 验证轮数 | — | 2.4373 |
| 验证轮数 | — | 13874 |
| 抢占次数 | 0 | 0 |

服务端耗时比为 1.3016×，输出吞吐比为 1.3012×。这些数值来自本轮同设备顺序测试，不使用旧 10k 草稿的耗时充当主模型基线。

两组的 1000 个样本 ID、提示词和 prompt token 数全部对齐；都以 `stop` 正常结束，没有达到 max_tokens 上限的截断。计数首尾差值与逐请求加总相同，5 条冒烟没有混入正式指标。每轮接受 0–7 个候选的次数依次为 `[5186, 3747, 2203, 1248, 672, 389, 195, 234]`，实际覆盖了零接受、部分接受和全部接受。

完整回答共有 **25 条差异**，索引为 `[166, 171, 184, 211, 245, 297, 304, 329, 402, 480, 546, 552, 606, 632, 646, 672, 734, 769, 861, 871, 904, 909, 935, 948, 979]`。这是回答文本比较，不是官方 TextVQA 准确率。抽查的三个差异在 eager 下仍复现，详见下文；根因尚未完成逐层归因。

96k 的重新上传路径在本轮两次复查中仍指向同一异常内容。最终一次 SHA256 检查时间为北京时间 **2026-09-14 17:07**，仍为 `3637ea6da1354e53077c2ffa2e49205e367aeabaab5424c8de77c23f641a3c7b`，因此之前确认的 17 个 NaN 和 2881 个超出 FP16 范围的有限值仍然存在，未重跑无效权重。

## 模型、代码和范围

- 主模型：`/home/models/Qwen3-VL-4B-Instruct`。
- 草稿：`/home/xj/checkpoints/qwen3-vl-4b-dflash-mixed300000-vocab32000-epoch3-6gpu-bs1-5layer-block8-gamma4-lr1e-4/epoch_5_step_192260`。
- 容器：`qzh_v023_vllm024_dflash`。
- 工作树：`/vllm-workspace/vllm-ascend-dflash-mrope-latest`，本地提交 `9208f4e`，加上前轮未提交的压缩词表 LM head / NZ 加载修复。
- 本轮复用已有测试工具，没有为测试增加推理热路径日志或 NPU→CPU 探针。功能检查启用已有 DEBUG 日志；正式速度测试使用原日志级别。

“FULL”在本文具体指 `FULL_DECODE_ONLY`，不等同于未测试的所有 FULL 图执行组合。功能回归仅覆盖本报告列出的场景，不能替代 TP=2、视频、采样统计和所有普通 RoPE 模型的完整验收。

## TextVQA 测试方式

复用 `/home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl`，保持原顺序，发送每条样本的第一轮 user。图片在文本之前，不发送标准 assistant 答案。该文件是图片描述加 OCR 提示，本文没有计算官方 TextVQA 问答准确率。

在同一物理设备 **3** 上先跑主模型单独推理，再跑 230k DFlash。两种运行分别先执行 5 条冒烟，然后重新记录计数基线，正式执行 1000 条。参数沿用前轮：FP16、TP=1、并发 1、temperature=0、max_tokens=128、max_model_len=2048、物理 block_size=128、1 GiB KV 缓存、关闭 prefix cache / chunked prefill / async scheduling、图像 min_pixels=65536 / max_pixels=1048576。DFlash 提出 7 个候选；FULL_DECODE_ONLY 捕获尺寸为 `[8]`，主模型单独运行用 `[1]`。

加速比按主模型客户端耗时总和除以 DFlash 客户端耗时总和计算，另报服务端耗时比和输出吞吐比。接受长度同时报告 `1 + accepted / verify` 和 `completion_tokens / verify`，避免混用口径。

同一设备顺序测试减少设备差异，但服务器仍由多个任务共享；功能检查使用设备 6。该结果是一轮实测，不是完全隔离、重复测量的性能统计。截图中 SGLang 的 TextVQA 230k 接受长度为 2.4448、加速比为 1.5152×；截图未列明两项计算公式和 GPU 硬件，作为外部参考保留，不用于证明跨硬件性能一致。

## 功能回归方式

复用 `tools.run_310p_dflash_mrope_validation.py`。每个独立进程预热一次，记录两轮输出 token ID 和结束原因。基础请求为 4 个请求，交替使用合成双图与文本，测试自然 EOS 和长度上限结束。双图为同一张 224×224 红色方块图重复两次，未覆盖不同尺寸多图，图像处理上限为 50176 像素；这部分是功能检查，与真实 TextVQA 性能数据分开。

基础组合覆盖 eager、PIECEWISE、FULL_DECODE_ONLY，并额外检查 15 个候选。高级组合同时开启 prefix cache、chunked prefill 和 async scheduling，使用更长提示和 256 的预填充分块预算；与同配置主模型基线核对。短输出组合检查 batch=3、max_tokens=1。

另对真实 TextVQA 前 20 条执行 eager 与 PIECEWISE 检查，图像和生成参数与正式 1K 测试相同。图模式不仅检查启动成功，也检查已有日志中的捕获与回放记录。

| DFlash 场景 | 记录输出组数 | 与主模型 token / 结束原因对齐 | 图执行证据 |
| --- | ---: | --- | --- |
| eager，batch=4，K=7 | 8 | 全部一致 | eager |
| PIECEWISE，batch=4，K=7 | 8 | 全部一致 | 主模型 / 草稿捕获及回放均有记录 |
| FULL_DECODE_ONLY，batch=4，K=7 | 8 | 全部一致 | 主模型 / 草稿捕获及回放均有记录 |
| FULL_DECODE_ONLY，batch=4，K=15 | 8 | 全部一致 | 主模型 / 草稿捕获及回放均有记录 |
| FULL_DECODE_ONLY，缓存 + 分块 + 异步 | 8 | 全部一致 | 主模型 / 草稿捕获及回放均有记录 |
| FULL_DECODE_ONLY，batch=3，max_tokens=1 | 6 | 全部一致 | 捕获完成；单 token 请求无需进入草稿解码回放 |

共 **46 组 DFlash 输出全部对齐**，候选计数检查通过，无图输入检查错误。两个主模型参考进程另各记录 8 组输出，不计入上述 46 组。

基础场景包含自然 `stop` 和达到上限的 `length`；短输出场景全部在 1 个 token 处以 `length` 结束，未调用草稿验证，符合预期。高级组合的主模型与 DFlash 都记录到 2432 个 prefix cache 命中 token（工具总计数包含预热）；图像提示在展开图像前已有 262 个 token，超过 256 的预填充预算，确保分块场景有实际输入覆盖。图像展开后长度还会增加。

日志中的 `event=replay` 是首次回放记录，表中只据此确认回放发生，不把日志条数当作完整回放次数。功能工具的接受指标包含预热，仅作诊断；正式 TextVQA 接受指标独立计算。

真实 TextVQA 前 20 条的补充检查结果：

| 模式 | 成功 / 失败 | 与主模型回答文本一致 | 接受率 | 1 + 接受 / 验证 |
| --- | ---: | ---: | ---: | ---: |
| eager | 20 / 0 | 20 / 20 | 19.9275% | 2.3949 |
| PIECEWISE | 20 / 0 | 20 / 20 | 19.9275% | 2.3949 |

两组都按首尾计数审计通过，均正常 `stop`。PIECEWISE 服务在完成请求后、收到 SIGTERM 的退出清理过程中出现 `AsyncLLM output_handler failed / EngineDeadError`，并有 31 个 semaphore 的资源清理告警；日志顺序确认在 shutdown 标记之后，测试请求没有失败，服务最终退出。本轮没有修改退出处理流程，不能宣称退出过程完全无告警。PIECEWISE 的主模型和草稿捕获、回放均有记录，无需仅根据启动参数推断使用了图。这里比较 HTTP 返回的回答文本；前面的独立功能工具才是直接 token ID 比较。

## 回答差异的补充检查

完整主模型 / DFlash 对照中出现少量回答文本差异，因此补测源数据索引 166、171、184（从 0 开始），每个样本在独立 eager 主模型、eager DFlash 服务中分别重复两次，保持 FP16、temperature=0 和原图像参数。

| 检查 | 三个样本的结果 |
| --- | --- |
| 主模型 eager 与主模型 FULL_DECODE_ONLY | 全部一致 |
| DFlash eager 与 DFlash FULL_DECODE_ONLY | 全部一致 |
| 同一模式重复两次 | 全部稳定 |
| 主模型 eager 与 DFlash eager | 三条仍均有差异 |

例如索引 166 的时间表达分别为“13:36”和“1:36 PM”；其他样本还出现描述内容差异。**这三个样本的差异并非仅由图回放造成，但本轮没有完成首个分歧位置的 logits 或逐层数值对齐，不能确定归因于 FP16 数值误差，也不能排除投机路径的数值或流程问题。**

因此，前述有限场景的图模式回归通过，不等同于主模型与 DFlash 在全部输入上贪心输出逐字一致。完整 1K 的一致数量及所有差异原文以本报告结果区和 `textvqa-pair-comparison.json` 为准。

补充检查证据保存于 `answer-difference-probes/`，含选出的输入、两组实际启动/测试命令、重复请求结果和 `comparison.json`。该检查使用独立设备 6，不向正式 1K 测试服务插入额外请求。

## 命令和证据目录

宿主机及容器共享目录：

```text
/home/qzh/vllm-ascend-dflash-mrope/artifacts/scenario-matrix-230k-20260914
```

- `textvqa-first1k-target/`、`textvqa-first1k-dflash/`：正式 1K 主模型和 DFlash 对照，包含实际 `launch-server.sh`、`run-tests.sh`、命令 JSON、逐请求回答、计数边界和服务日志。
- `functional/各场景/`：独立功能运行的 `run.sh`、`command.json`、`result.json` 和 DEBUG 日志；输出记录真实 token ID。
- `graph-textvqa/eager/`、`graph-textvqa/piecewise/`：真实 TextVQA 前 20 条图模式检查及完整命令。
- `source-manifest.json`、`source/`：本轮使用的代码版本与关键工具快照。
- `96k-reupload-check.json`、`96k-reupload-host-check.json`：用户重新指定 96k 路径后的只读复查。宿主机与容器 SHA256 均与先前异常文件一致，仍有 17 个 NaN 和 2881 个有限值超出 FP16 范围；未重新运行无效权重。

以下为 230k 正式 1K 使用的实际参数。主模型基线的完整命令保存在 `textvqa-first1k-target/launch-server.sh`；它不带 `--speculative-config`，捕获尺寸为 `[1]`。复跑前确认设备及端口空闲，并更换输出目录；测试工具拒绝覆盖已有结果。

启动 DFlash：

```bash
docker exec -i \
  -e ASCEND_RT_VISIBLE_DEVICES=3 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -w /vllm-workspace/vllm-ascend-dflash-mrope-latest \
  qzh_v023_vllm024_dflash bash <<'BASH'
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 31047 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --block-size 128 \
  --gpu-memory-utilization 0.45 \
  --kv-cache-memory-bytes 1073741824 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --generation-config vllm \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}' \
  --speculative-config '{"method": "dflash", "model": "/home/xj/checkpoints/qwen3-vl-4b-dflash-mixed300000-vocab32000-epoch3-6gpu-bs1-5layer-block8-gamma4-lr1e-4/epoch_5_step_192260", "num_speculative_tokens": 7}' \
  --disable-uvicorn-access-log \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'
BASH
```

服务就绪后，在另一个宿主机终端执行 5 条冒烟和 1000 条正式测试：

```bash
docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/scenario-matrix-230k-20260914/textvqa-first1k-dflash/smoke \
  --server-address 127.0.0.1:31047 \
  --limit 5 \
  --max-tokens 128 \
  --temperature 0 \
  --timeout 120

docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/scenario-matrix-230k-20260914/textvqa-first1k-dflash/formal \
  --server-address 127.0.0.1:31047 \
  --limit 1000 \
  --max-tokens 128 \
  --temperature 0 \
  --timeout 120
```

功能场景的每个 `run.sh` 是可独立执行的完整命令。例如：

```bash
bash /home/qzh/vllm-ascend-dflash-mrope/artifacts/scenario-matrix-230k-20260914/functional/draft-full-advanced/run.sh
```

功能脚本中的输出路径也应在复跑前改为新目录，以保留本轮原始证据。

## 交付状态

本轮推理源码及测试工具 SHA256 与启动时一致；所有测试 API 服务已退出，正式 DFlash EngineCore 也已退出。此前压缩词表加载修复仍为本地未提交改动，本轮没有创建新提交或推送远程。运行日志、图捕获/回放证据、计数审计及全部差异原文保存在上述证据目录。

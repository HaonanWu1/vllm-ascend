# Qwen3-VL-4B DFlash 新混合数据 checkpoint 测试

日期：2026-09-14。沿用 2026-09-10 的 `textvqa-test-1k`、主模型和测试参数，比较用户提供的新草稿与此前 10k 草稿。原始数据任务为图片描述加 OCR 提示；接受率不是官方 TextVQA 问答准确率。

## 结果与结论

**230k / 32k 词表草稿完成 1000 条正式测试：全部成功，接受率 19.9460%。96k 全词表草稿在传输完成后执行了 5 条冒烟，但权重含 NaN 及异常大值，正式 1000 条测试暂缓。**

| 指标 | 原 10k 草稿 | 新 230k / 32k 词表草稿 |
| --- | ---: | ---: |
| 成功 / 失败 | 1000 / 0 | 1000 / 0 |
| 接受候选 / 提出候选 | 14461 / 128807 | 19360 / 97062 |
| 全局候选接受率 | 11.2269% | 19.9460% |
| 验证轮数 | 18401 | 13866 |
| 1 + 接受候选数 / 验证轮数 | 1.7859 | 2.3962 |
| 输出 token / 验证轮数 | 1.8381 | 2.4397 |
| 输出 token 总数 | 33822 | 33829 |
| 平均客户端耗时 / 请求 | 2.4371 秒 | 2.0146 秒 |
| 中位客户端耗时 | 2.3653 秒 | 1.9521 秒 |
| 输出 token / 客户端耗时总和 | 13.8782 token/s | 16.7922 token/s |
| 抢占次数 | 0 | 0 |

接受率提高 **8.7191 个百分点**，为旧草稿的 **1.7766 倍**。客户端平均耗时本轮观测降低 17.34%；由于测试日期、物理设备不同且服务器共享，不将此数值作为严格的同条件性能回归结论。

7 个候选位置的最终接受计数分别为 `[8692,4944,2743,1494,820,431,236]`。指标审计通过：1000 个索引和 ID 唯一、每轮提出 7 个候选、逐位置之和等于接受总数、逐请求之和等于正式测试首尾计数差；正式测试排除了 5 条冒烟。1000 条均为 `stop`，无长度截断，运行日志无错误或异常堆栈。

与旧 10k 草稿的 1000 条结果核对：输入 ID、提示词和 prompt token 数全部一致；回答文本 996/1000 完全一致，差异索引为 245、646、769、861（从 0 开始），原文保存在 `230k-final-comparison.json`。本次另跑的主模型单独推理对照只有前 20 条，20/20 与新草稿回答一致。因此本轮没有证明全部 1000 条都与主模型单独推理逐字一致，也没有通过这些文本比较计算官方任务准确率。

本次初始 0% 接受率有明确的加载流程问题：压缩 checkpoint 缺少输出层时没有从主模型构造对应词表行，随后还需刷新 310P 的 NZ 权重。修复后接受率稳定在约 20%。最终新草稿优于旧草稿的观测成立，但缺少新 checkpoint 的 GPU 对照，不能据此断言剩余拒绝完全由训练质量造成。

### 96k 传输完成后的补测（2026-09-14 15:01–15:08）

文件长度现为完整的 **1074860568 字节**，safetensors 文件结构可解析，58 个张量。源码 SHA256 与 230k 有效运行时一致；使用设备 1、端口 31044，其余生成、图像、调度和图执行参数相同。启动日志确认从主模型共享 embedding 与 LM head，全词表模型不进入压缩词表输出层重建分支。

| 本次 96k 冒烟指标 | 结果 |
| --- | ---: |
| 请求成功 / 失败 | 5 / 0 |
| 接受 / 提出候选 | 0 / 1141 |
| 验证轮数 | 163 |
| 候选接受率 | 0% |
| 平均接受长度（1 + 接受 / 验证） | 1.0000 |
| 实际输出 token / 验证轮数 | 1.0307 |
| 主模型单独推理文本核对 | 5 / 5 一致 |
| 正式 1000 条 | 未启动 |

这组 0% 仅为异常文件的诊断结果，不能作为该训练版本的正常接受率，也不能与另外两组完整 1000 条结果直接比较。

权重检查发现：

- `layers.3.mlp.gate_proj.weight`（从 0 开始的第 3 层，即第 4 层）形状 `[9728,2560]`，BF16，包含 **17 个 NaN**，集中在行 2790–2793。
- 同一张量另有 **2881 个有限值超过 FP16 最大值 65504**；最大有限绝对值约为 **2.51224 × 10^38**。
- 使用 CPU 加载张量与 Python `struct` 直接检查 BF16 原始字节，两条独立路径均确认 17 个 NaN。它们已存在于文件中，发生在 vLLM、Ascend 和 m-RoPE 计算之前。
- 独立 CPU FP32 测试向这一 MLP 输入全 1 的有限向量：gate 输出有 4 个 NaN，down projection 的 2560 个输出全部为 NaN。该实验说明异常权重能够传播并污染输出，但不冒充完整 NPU 推理追踪。
- 旧 10k 草稿所有浮点权重均有限；此前 230k 权重检查也未发现非有限值。

本机 96k `model.safetensors` 的 SHA256（启动前后两次计算一致）：

```text
3637ea6da1354e53077c2ffa2e49205e367aeabaab5424c8de77c23f641a3c7b
```

用户目前无法访问训练机，尚未取得源文件 SHA256。**目前只能确认本机文件异常，不能确定异常来自训练/保存还是传输。** 后续若源 SHA256 不同，需取得可靠副本并重新校验；若相同，则应检查源 checkpoint 数值并重新导出或选用正常 checkpoint。没有替换 NaN、截断异常值或修改原权重。

冒烟指标逐请求与服务端首尾计数核对通过，5 条均为 `stop`，无抢占。测试队列在检测到 0 接受后退出，正式 1000 条没有启动；本次服务已停止。补测未新增生产代码改动或 Git 提交。

原始证据目录：`96k-full-ready/`。其中 `checkpoint-preflight.json` 保存文件长度、配置、SHA256 与张量键；`weight-statistics.json` 保存全张量统计；`raw-weight-nonfinite-audit.json` 保存每个异常值的坐标、BF16 位值和文件偏移；`cpu-mlp-nan-propagation.json` 保存传播实验；`smoke-audit.json` 保存冒烟计数审计。首次记录上传未完整的 `96k-full/` 仍保留作历史诊断。

## 模型与实际文件

主模型：`/home/models/Qwen3-VL-4B-Instruct`。

“96k 全词表”和“230k / 32k 词表”沿用用户的名称。目录名含有不同的数据量和 epoch 字样，本文不根据名称推断实际训练轮数或样本数量。实际选择的唯一 checkpoint 为：

```text
96k 全词表：
/home/xj/checkpoints/qwen3-vl-4b-dflash-mixed96953-1epoch-6gpu-bs1-5layer-block8-gamma4-lr1e-4/epoch_3_step_48477

230k / 32k 词表：
/home/xj/checkpoints/qwen3-vl-4b-dflash-mixed300000-vocab32000-epoch3-6gpu-bs1-5layer-block8-gamma4-lr1e-4/epoch_5_step_192260
```

两者均为 5 层草稿、目标特征 `[3,10,18,25,32]`、交错 m-RoPE `[24,20,20]`、草稿块大小 8。全词表为 151936；压缩模型的 `draft_vocab_size=32000`。

32k checkpoint 的 `d2t` 为偏移量：`target_id = draft_id + d2t[draft_id]`。32000 个实际 target ID 均唯一，范围为 0–151645，与 `t2d` 中的真值索引完全一致；全部浮点权重均为有限值。

首次检查时，96k 的 `model.safetensors` 只有 305430528 字节，而其文件头要求 1074860568 字节；用户确认仍在传输。230k 的模型文件实际与要求的长度均为 1075268656 字节。文件未完整前的启动失败不计为模型测试结果。

## 压缩词表加载修复

本次发现一种现有路径未覆盖的 checkpoint：**保存了词表映射，却没有保存自己的 `lm_head.weight`**。原实现把“存在 d2t”直接当作“草稿自带输出层”，保留未加载权重的输出层。首次 5 条诊断请求接受 0 / 1141 个候选，不能用这个值评价训练质量。

修复仅位于 vLLM-Ascend：

1. 跟踪实际权重加载流，判断 checkpoint 是否真的包含 `lm_head.weight`。
2. 对缺少输出层的压缩词表 DFlash，从主模型输出层按映射选出对应行；TP 场景先收集目标词表分片，再由草稿权重加载器完成切分与 padding。
3. 加载后重新执行输出层的权重后处理。310P 实际矩阵乘使用 `weight_nz`，只修改原始 `weight` 会留下旧的 NZ 缓存，仍然产生无效候选。
4. 已自带输出层的压缩模型、普通全词表 DFlash 和相邻投机方法保持原加载策略。新逻辑在模型加载时执行，没有增加推理热路径同步探针。

涉及的文件：

- `vllm_ascend/_310p/spec_decode/dflash_vocab.py`：实际输出层权重检测、目标行选取及 NZ 重建。
- `vllm_ascend/patch/worker/patch_idex_310.py`：把权重加载与输出层处理绑定到实际模型和 proposer。
- `tests/ut/_310p/spec_decode/test_dflash_vocab.py`：映射、缓存刷新、已有输出层、全词表、相邻方法及 TP 收集顺序回归。
- `tests/ut/_310p/spec_decode/test_dflash_mrope.py`：增加实际 patch 绑定检查。

验证结果：405 项相关单元测试通过，14 条为 PyTorch 废弃提示。四个代码/测试文件的语法、Ruff、格式及 `git diff --check` 均通过，运行源码 SHA256 与保存的修复快照一致。独立 NPU 检查使用实际 FRACTAL_NZ（格式 29），验证非连续目标行 `[1,17,66]`、padding 与矩阵乘：权重和 logits 的最大绝对误差均为 0。单元测试的 TP 检查不等同于 TP=2 实机验收；本次数据集测试使用 TP=1。

新修复基于本地提交 `9208f4e`，未修改上游 vLLM 或 checkpoint，也未自动创建新提交或推送远程。两次修复未完整时的 0% 请求与最终测试分目录保存，正式汇总仅使用正确加载后的运行。

## 测试配置与统计

| 项目 | 本轮配置 |
| --- | --- |
| 容器 | `qzh_v023_vllm024_dflash` |
| 工作树 | `/vllm-workspace/vllm-ascend-dflash-mrope-latest` |
| 主模型精度 / TP | FP16 / 1 |
| 图模式 | FULL_DECODE_ONLY，capture sizes `[8]` |
| 请求 | 顺序、并发 1、非流式；首轮 user，图片在文本前 |
| 生成 | temperature=0，max_tokens=128，timeout=120 秒 |
| 调度 | 同步；关闭 prefix cache、chunked prefill |
| 长度 / 缓存 | max_model_len=2048；物理 block_size=128；KV 缓存固定 1 GiB |
| 图像 | min_pixels=65536，max_pixels=1048576；与上次相同 |
| 数据 | `/home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl`，原顺序 1000 条 |

流程为先执行 5 条冒烟，通过后重新取指标基线执行正式 1000 条；96k 因权重异常在冒烟后中止。成功请求、候选、接受 token 及各候选位置均按 Prometheus 差值记录；检查逐请求加总与首尾差值一致。5 条冒烟不混入正式接受率，但图像处理缓存未重置。

全局接受率为 `accepted_drafts / proposed_drafts`，草稿每轮提出 7 个候选；接受长度同时报告 `1 + accepted / verify` 与 `completion_tokens / verify`。客户端耗时仅测 HTTP 请求，不包括指标采集或服务启动。

服务器为共享环境，新旧测试日期及使用的物理设备不同。耗时是实测观测值，不作为同条件重复性能回归或训练方案优劣的单独证据。新模型没有对应的 GPU SGLang 测试结果；此前 GPU 11.2628% 来自旧 10k checkpoint，不能当作新模型的 GPU 对照。

## 命令与原始证据

宿主机及容器内共享目录：

```text
/home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-new-checkpoints-20260914
```

以下为本次 230k 有效测试实际参数对应的宿主机命令。设备 3、端口 31045；复跑前应确认资源空闲，并更换测试输出目录。原始命令数组和运行脚本均保存在有效运行目录。

启动：

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
  --port 31045 \
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

服务健康检查通过后，另一个终端依次执行 5 条冒烟和 1000 条正式测试：

```bash
docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-new-checkpoints-20260914/230k-vocab32k-nz/smoke \
  --server-address 127.0.0.1:31045 \
  --limit 5 \
  --max-tokens 128 \
  --temperature 0 \
  --timeout 120

docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-new-checkpoints-20260914/230k-vocab32k-nz/formal \
  --server-address 127.0.0.1:31045 \
  --limit 1000 \
  --max-tokens 128 \
  --temperature 0 \
  --timeout 120
```

所有正式测试均使用工作树中的 `python -m tools.evaluate_310p_textvqa_dflash`。复跑应更换输出目录，工具拒绝覆盖已有 `requests.jsonl`。每个有效运行目录包含：

- `server-command.json`、`launch-server.sh`：完整实际启动命令。
- `test-commands.json`、`run-tests.sh`：5 条冒烟和正式测试命令。
- `formal/requests.jsonl`：逐条回答、输入/输出 token 数、结束原因、耗时和接受指标。
- `formal/summary.json`、`formal/metrics-before.json`、`formal/metrics-after.json`：汇总和计数边界。
- `server.log`、`formal-client.log`、`queue-exit.json`：运行日志和退出状态。

根目录另存：`head-nz-fix-unit-tests.log`、`head-nz-npu-check.json`、`head-fix-source.json` 及源码副本。`target-control` 保存本次主模型单独推理的 20 条对照；`230k-target-control-comparison.json` 记录对应文本核对结果。

最终汇总和审计分别见 `230k-final-comparison.json`、`final-audit.log`、`final-runtime-audit.json`、`final-static-checks.json`。本材料也保存于该工作树的 `docs/source/developer_guide/dflash_mrope_mixed_checkpoints_zh.md`。

## 96k 补测启动与测试命令

以下命令已用于本次 5 条冒烟。当前文件数值检查未通过，应取得正常权重后再复跑，并更换输出目录。

```bash
docker exec -i \
  -e ASCEND_RT_VISIBLE_DEVICES=1 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -w /vllm-workspace/vllm-ascend-dflash-mrope-latest \
  qzh_v023_vllm024_dflash bash <<'BASH'
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 31044 \
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
  --speculative-config '{"method": "dflash", "model": "/home/xj/checkpoints/qwen3-vl-4b-dflash-mixed96953-1epoch-6gpu-bs1-5layer-block8-gamma4-lr1e-4/epoch_3_step_48477", "num_speculative_tokens": 7}' \
  --disable-uvicorn-access-log \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'
BASH
```

服务就绪后实际执行的冒烟：

```bash
docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-new-checkpoints-20260914/96k-full-ready/smoke \
  --server-address 127.0.0.1:31044 \
  --limit 5 \
  --max-tokens 128 \
  --temperature 0 \
  --timeout 120
```

`96k-full-ready/test-commands.json` 及 `run-tests.sh` 也保存了原定正式 1000 条命令；**该正式命令本次没有执行**，不能把预先生成的脚本视为已完成测试的证据。

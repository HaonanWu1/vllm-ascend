# Qwen3-VL-4B DFlash：TextVQA 1000 条接受率对照

测试日期：2026-09-10。本文记录用户提供的 `textvqa-test-1k` 在 Ascend 310P 上的实测，并对照 `hn47-reference` 中的 GPU SGLang 结果。

## 结论与完整结果

**1000 / 1000 请求成功，零失败，全部以 `stop` 正常结束。Ascend 接受率为 11.2269%，GPU SGLang 为 11.2628%，Ascend 低 0.0359 个百分点。** 本次真实数据测试没有发现 Ascend 特有的明显接受率损失。

| 指标 | Ascend FP16 | GPU SGLang BF16 |
| --- | ---: | ---: |
| 成功请求数 | 1000 | 1000 |
| 输出 token 总数 | 33822 | 33957 |
| 提出的候选 token 总数 | 128807 | 129178 |
| 接受的候选 token 总数 | 14461 | 14549 |
| 草稿块 / 校验次数 | 18401 | 18454 |
| 全局候选接受率 | 11.2269% | 11.2628% |
| `1 + accepted / verify` | 1.7859 | 1.7884 |
| `completion_tokens / verify` | 1.8381 | 1.8401 |
| 逐请求接受率的简单平均 | 12.3828% | 12.3440% |
| 客户端平均请求耗时 | 2.4371 秒 | 0.4589 秒 |
| 客户端中位请求耗时 | 2.3653 秒 | 0.4373 秒 |
| 客户端 P90 请求耗时 | 3.2295 秒 | 0.6108 秒 |
| 引擎端平均 e2e 耗时 | 2.4334 秒 | 0.4517 秒 |

Ascend 按全部请求 HTTP 耗时之和计算的输出速度为 13.8782 token/s。以上耗时来自不同硬件和精度，且不是同机重复测量；不能作为引擎优化优劣或此前“吞吐下降不超过 5%”的回归验收结论。

输入与结果核验：

- 1000 个唯一 ID、样本顺序和首轮用户文本与 GPU 记录全部对齐；**1000 / 1000 prompt token 数一致**。
- 最长 prompt 为 1287 token，最长输出为 80 token；没有请求触及 128-token 输出上限。抢占计数为 0，正式服务日志没有 ERROR 或 traceback。
- 正式成功请求计数差值为 1000，启动前的 5 条冒烟已排除；逐请求指标加总等于正式首尾指标差值。
- `128807 = 7 × 18401`，各位置接受计数加总为 14461，全部请求均通过计数审计。
- Ascend 开启 DFlash 与 Ascend 主模型单独推理的前 20 条，**回答文本、结束原因、输入及输出 token 数均为 20 / 20 一致**。这是 API 文本与计数对照，没有将其扩大为 1000 条逐 token ID 对照。
- Ascend 与 GPU 的回答文本完全一致为 **395 / 1000**，合并空白后仍为 395 / 1000。这是跨精度、跨后端的开放式描述一致率，**不是准确率**。两边使用 FP16 / BF16，数值差异可能改变贪心生成的早期选词并影响后续回答；仅凭文本不同不能确定差异来源。上述 20 条控制组中，关闭 DFlash 后仍保留了与 GPU 的措辞差异。

### 对此前低接受率问题的判断

此前合成图像用例约 3.3%，本次用户提供的数据上升到约 11.23%，GPU 在同一测试集上也只有约 11.26%。结合此前独立 HF 草稿计算与 Ascend 接受流程重放一致的证据，**目前更符合“草稿在不同输入上的 token 匹配能力有限”，没有证据表明需要通过继续修改 m-RoPE 来修复这组低接受率**。

Ascend 各候选位置的连续前缀接受计数为：

| 候选位置 | 连续通过到该位置的块数 |
| --- | ---: |
| 1 | 8383 |
| 2 | 3760 |
| 3 | 1510 |
| 4 | 535 |
| 5 | 190 |
| 6 | 56 |
| 7 | 27 |

18401 个块中，有 10018 个块的第一个候选即未通过，约占 54.44%；每块平均接受约 0.786 个草稿 token。因此块大小 8 并不等于每轮能输出 8 个有效 token。进一步提高接受率应检查草稿训练数据、监督目标与实际任务的匹配情况；当前仍没有训练 loss/accuracy 日志，不能据此断言欠拟合、过拟合或训练代码存在错误。

本轮新增验证范围为 TP=1、FP16、单图、顺序请求、FULL_DECODE_ONLY。它补充此前 m-RoPE 数值与生成验证，不代替 TP=2、多请求并发、其他图模式及普通 RoPE 完整性能矩阵的验收。正式测试、比较工具均以退出码 0 结束；语法、Ruff 检查和格式检查通过，指标审计还验证了异常计数会被拒绝。

## 测试对象与范围

| 项目 | Ascend 本次测试 | GPU 已有参考 |
| --- | --- | --- |
| 引擎 | vLLM 0.24.0 + vLLM-Ascend | 用户提供的 SGLang 0.5.14 PR18387 路径 |
| Ascend 源码 | `f48bd99fd7b9458bbad9c61f36a10b2217eb40b3` + 未提交 m-RoPE 修改 | 不适用 |
| 硬件 | Ascend 310P3，物理卡 4 | 原始命令仅给出 CUDA 设备 0，具体型号未提供 |
| 精度 / TP | FP16 / 1 | BF16 / 1 |
| 草稿 | 5 层，目标特征 `[3,10,18,25,32]`，交错 m-RoPE | 用户提供的同一草稿配置 |
| 草稿块 / 候选 | 块 8 / 候选 7 | 块 8 / 候选 7 |
| 采样 / 输出上限 | temperature=0 / max_tokens=128 | 相同 |
| 请求方式 | 顺序发送，并发 1，非流式 | 原评估脚本顺序发送，并发 1，非流式 |
| 图执行 | FULL_DECODE_ONLY，capture sizes `[8]` | 原启动命令默认行为，未提供额外图配置 |
| 调度 | 同步；关闭 prefix cache 和 chunked prefill | 原启动命令默认行为 |

容器为 `qzh_v023_vllm024_dflash`，代码及 Git 工作树仍在：

```text
/vllm-workspace/vllm-ascend-dflash-mrope-latest
branch: feat/310p-dflash-mrope-latest
common Git directory: /vllm-workspace/vllm-ascend/.git
```

主模型为 `/home/models/Qwen3-VL-4B-Instruct`；草稿为 `/home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750`。

本轮沿用现有 m-RoPE 生产修改，只增加测试、比较工具和本报告；没有修改上游 vLLM、SpecForge 或模型权重，也没有重新训练。

## 输入与统计方式

1. 读取 `/home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl`，按原顺序执行全部 1000 条。1000 个 ID 及首轮用户文本逐条与 GPU 记录对齐；数据清单中的 1001 个 SHA256 均验证通过。
2. 每个请求只发送一条 user 消息，内容顺序为图片、文本。图片使用原文件的 base64 data URL。没有发送 system 消息，也没有将数据中的 assistant 参考答案送入模型。这与 SpecForge 的 `scripts/evaluate_vlm_dflash.py` 一致。
3. 这份数据实际提示为 **图片描述 + Reference OCR token**，例如“Provide a one-sentence caption…”。本次测量接受率与回答一致性，**不是官方 TextVQA 问答准确率评测**。
4. 图像处理使用 `min_pixels=65536,max_pixels=1048576`。原始图片最大为 1048576 像素；预检使用 HF resize 规则，确认这个上限与模型默认 16777216 上限在全部 1000 条上得到相同的缩放后高宽及图像 token 数，也与 GPU 的 image_tokens 一致。降低上限是为了避免服务启动时按默认 16384 个视觉 token 做过大的预热，不会降低这份测试集的实际输入分辨率。
5. 先执行 5 条冒烟，再执行完整 1000 条。正式运行前保存指标基线，5 条冒烟不计入正式接受率。GPU 汇总中的 `excluded_startup_requests=1` 指引擎启动检查请求，并未删除测试集第一条；双方均比较数据集的完整 1000 条。prefix cache 关闭；视觉处理缓存没有重置，因此正式测试前五张图片可能受处理缓存预热影响。启动图捕获耗时不计入请求耗时。
6. 该端口只供本测试使用，每条请求完成后读取 Prometheus 指标差值，要求成功请求数恰好增加 1。正式结束后核验总差值与逐请求加总一致，以及 `候选数=7×草稿块数`、各位置接受数之和等于接受总数。错误会立即停止，防止超时请求污染后续指标。
7. 客户端耗时只覆盖 HTTP 请求到完整响应，指标读取和本地文件准备不计入。另保存引擎端 e2e 指标；GPU 客户端耗时从原逐请求记录重新统计，不能把 GPU 引擎耗时与 Ascend 客户端耗时混在一起比较。

全局接受率定义为 `接受的草稿 token 总数 / 提出的草稿 token 总数`，不能使用逐请求接受率的简单平均替代。

两个接受长度的口径分别为：

- `1 + accepted_drafts / verify_calls`：每次校验接受的候选数量，加上一个主模型 token。
- `completion_tokens / verify_calls`：实际输出 token 数除以校验次数；首 token、EOS 等边界会使它与前者不同。

各位置接受计数是连续前缀通过数，后续位置需要前面的候选先通过，不是各位置互相独立的预测准确率。

## 复现：启动服务

下面在 **Ascend 宿主机** 执行；物理卡 4 必须有足够空闲内存，端口不能被其他服务占用。服务与正式测试分两个终端执行。原始完整命令也已保存为证据目录的 `launch-server.sh` 和 `server-command.json`。

```bash
docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -w /vllm-workspace/vllm-ascend-dflash-mrope-latest \
  qzh_v023_vllm024_dflash bash -c '
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 --port 31042 \
  --dtype float16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 --block-size 128 \
  --gpu-memory-utilization 0.45 \
  --kv-cache-memory-bytes 1073741824 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --no-async-scheduling \
  --generation-config vllm \
  --limit-mm-per-prompt "{\"image\":1,\"video\":0}" \
  --mm-processor-kwargs "{\"min_pixels\":65536,\"max_pixels\":1048576}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[8]}" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750\",\"num_speculative_tokens\":7}" \
  --disable-uvicorn-access-log
'
```

FP16 是本轮 310P 实测配置。物理 KV block_size=128 与 DFlash 的 8-token 草稿块不是同一个参数。固定 1 GiB KV 缓存用于避免共享服务器上自动显存估算受到其他任务影响。

## 复现：执行与比较

服务健康后，先执行冒烟，再执行正式测试。输出目录不能已有 `requests.jsonl`；工具会拒绝覆盖。复跑时更换为新目录并保留原始结果。

```bash
curl --fail http://127.0.0.1:31042/health

# 5 条冒烟；正式统计会在它结束后重新取指标基线。
docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest \
  qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-1k/smoke \
  --limit 5 --max-tokens 128 --temperature 0 --timeout 120

# 本次实际使用的正式测试命令。
docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest \
  qzh_v023_vllm024_dflash python -m tools.evaluate_310p_textvqa_dflash \
  --input-path /home/xj/data/textvqa-test-1k/textvqa-test-1k.jsonl \
  --output-dir /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-1k/ascend-1000 \
  --limit 1000 --max-tokens 128 --temperature 0 --timeout 120

# 完整结果与 GPU 参考比较，同时核验指标。
docker exec -w /vllm-workspace/vllm-ascend-dflash-mrope-latest \
  qzh_v023_vllm024_dflash python -m tools.compare_310p_textvqa_results \
  --ascend-requests /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-1k/ascend-1000/requests.jsonl \
  --gpu-requests /home/xj/data/hn47-reference/requests-textvqa-first1k.jsonl \
  --gpu-summary /home/xj/data/hn47-reference/summary-textvqa1k.json \
  --output-path /home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-1k/comparison.json \
  --expected-requests 1000
```

测试工具默认地址为 `127.0.0.1:31042`，model 为上述主模型绝对路径，可以通过 `--server-address` 和 `--model` 显式覆盖。它使用 OpenAI 兼容请求以及 vLLM 原有指标，没有在 NPU 热路径插入额外同步探针。

主模型单独推理的 20 条对照使用同一容器、物理卡 3、端口 31043，去掉 `--speculative-config`，图捕获大小改为 `[1]`，其余参数相同；测试工具增加 `--server-address 127.0.0.1:31043 --limit 20` 并使用 `target-20` 输出目录。精确命令分别保存在 `target-server-command.json`、`target-test-command.json`。这个小样本对照用于检查回答正确性，没有做同一设备上的重复性能测量，因此不作为加速比验收。

## 证据与改动

宿主机和容器内共享证据路径：

```text
/home/qzh/vllm-ascend-dflash-mrope/artifacts/textvqa-1k
```

- `ascend-1000/requests.jsonl`：1000 条回答、用量、结束原因、客户端耗时与逐请求接受指标。
- `ascend-1000/summary.json`、`comparison.json`：总指标、与 GPU 的比较、指标审计及逐条一致性。
- `ascend-1000/metrics-before.json`、`metrics-after.json`：正式测量的指标边界。
- `dataset-preflight.json`、`resize-dimensions-preflight.json`：数据清单校验、图像 token 与缩放后高宽预检。
- `target-20-comparison.json`、`target-20/`：主模型单独推理的 20 条对照。
- `server-command.json`、`test-command.json`、`launch-server.sh`、`run-test.sh`：实际启动和测试命令。
- `server.log`、`test-client.log`、`test-exit.json`：运行日志及正式退出状态。
- `runtime-source.json`、`static-checks.json`：源码身份和本轮工具检查。

新增仓库文件：`tools/evaluate_310p_textvqa_dflash.py`、`tools/compare_310p_textvqa_results.py` 和本报告。已有 m-RoPE 实现及此前逐层数值验证见 `dflash_mrope_latest_validation_zh.md`，此前低接受率的独立 HF 排查见 `dflash_mrope_acceptance_diagnosis_zh.md`。

首次启动按模型默认视觉上限做过大的 profile，后改为上述对数据无损的像素上限重新启动；原始日志以 `unbounded-profile-` 前缀保留。正式 1000 条只来自修正启动配置后的服务。

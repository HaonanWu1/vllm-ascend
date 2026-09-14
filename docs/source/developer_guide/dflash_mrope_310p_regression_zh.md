# 310P DFlash m-RoPE 回归报告

> 本文保留 47dbcb3 基线上的实现及实机记录；最新 f48bd99 分支在指定容器中的部署状态见 [容器安装记录](dflash_mrope_container_install_zh.md)。

测试日期：2026-09-10。实现和训练结构说明见 [修改总结](dflash_mrope_310p_summary_zh.md)，英文说明见 [implementation notes](dflash_mrope_310p.md)。

## 环境与方法

Ascend 基线为 `47dbcb3bd7de18de1de910a3bbf708a22b8d22a4`，vLLM 为 `ee0da84ab9e04ac7610e28580af62c365e898389`。隔离修改目录为 `/home/qzh/vllm-ascend-dflash-mrope`，基线目录为 `/home/qzh/vllm-ascend-dflash-mrope-baseline`。源代码、测试及文档改动全部限于 Ascend 仓库。

运行于 `scc_dflash_dev` 的 310P3，使用 FP16、匹配的 Ascend 编译算子、torch 2.10.0、torch-npu 2.10.0.post2、Transformers 5.5.4。两个版本使用同一编译算子，其 SHA256 为 `0b3efa42923e28ed6097a53e92bf7aecfcaadbfd225ab220b0dd62db73208e82`。完整记录及生产源码校验值见 `artifacts/mrope/environment.json`。

模型测试默认 TP=1、物理 block size 128、max model len 2048、max num seqs 8、max batched tokens 2048、GPU memory utilization 0.75、seed 42。每个进程预热一次后测量；普通 RoPE 初始矩阵重复 3 次，专项性能复测重复 9 次。新 m-RoPE 功能用例预热后测量一次，通常生成 32 个 token；EOS 用例上限为 256。

本文 `FULL` 指本基线已支持的 `FULL_DECODE_ONLY`，decode 使用 FULL，prefill 使用既有回退路径。没有将它扩展解读为所有 prefill 也必须 FULL。

## 数值与整数值对齐

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| HF 三轴位置：224×224 和 448×112 两种图像，缓存前缀后的 context 后缀及八个 query | 全部整数值一致 | `hf-npu-reference.json` |
| 同上两种位置的 FP16 Q/K，HF 与 NPU 旋转 | 最大绝对误差均为 0 | `hf-npu-reference.json` |
| 实际单图第一块，五层草稿输出与保存的训练实现 | 相对 RMS 误差 0.0766%–0.0979%，余弦相似度均大于 0.999996 | `layer-reference.json` |
| 第一块最终 norm 输出 | 相对 RMS 误差 0.0811%，最大绝对误差 0.0078125 | `layer-reference.json` |
| AscendC 输入展开与槽位：K=7/15、block=64/128、零/部分/全部候选接受 | 所有六组整数输出与 NumPy 参考完全一致 | `onboard-input-tests.log` |

逐层数值检查使用实机捕获的相同目标特征和 query embedding，context 为 88 个 token、query 为 8 个 token。它比较草稿前向，不独立重算 HF 主模型的每层隐藏状态，也不代表对每个生成步逐层采集。隐藏层选择和 HF 的 +1 偏移另经过源码核对。

| 草稿输出 | 最大绝对误差 | 相对 RMS 误差 |
| --- | ---: | ---: |
| Layer 0 | 0.03125 | 0.097928% |
| Layer 1 | 0.0625 | 0.083253% |
| Layer 2 | 0.0625 | 0.080897% |
| Layer 3 | 0.0625 | 0.078585% |
| Layer 4 | 0.09375 | 0.076565% |
| Final norm | 0.0078125 | 0.081090% |

## 新 checkpoint 的生成与调度回归

主模型为 Qwen3-VL-4B-Instruct，草稿为指定 epoch_3_step_3750，候选数为 7。下列贪心用例与同配置主模型单独推理比较 token ID，采样用例单列。

| 用例 | 配置要点 | 结果 |
| --- | --- | --- |
| 纯文本 | eager，单请求 | 逐 token 一致 |
| 单图 | eager，单请求 | 逐 token 一致 |
| 单图 | PIECEWISE，单请求 | 与 eager 及主模型参考一致 |
| 单图 | FULL，单请求及三个请求 | 逐 token 一致 |
| 图文混合 | eager，四个请求 | 逐 token 一致 |
| 图文混合 | PIECEWISE，三个请求 | 逐 token 一致 |
| 多图与缓存 | 每请求两图、三个请求、prefix cache、chunked prefill、256 token 调度预算、长 prompt | 逐 token 一致 |
| 更小预填充分块 | 每请求两图、三个请求、prefix cache、64 token 调度预算 | 逐 token 一致；覆盖未完成图片 prefill 的临时负 query 起点 |
| 异步调度 | 单图、三个请求、eager | 逐 token 一致 |
| EOS 与动态 batch | 图文混合四请求，FULL，尊重 EOS，上限 256 | 逐 token 一致；长度 `[66,256,66,256]`，结束原因为 `[stop,length,stop,length]`，batch 从 4 降至 2 |
| TP=2 | 图文混合三个请求，eager | 逐 token 一致 |
| TP=2 | 图文混合三个请求，FULL | 逐 token 一致，图捕获及回放完成 |
| 采样 | temperature=0.8、单图三个请求、尊重 EOS、上限 64 | 连续生成成功；共享采样器统计/恢复分布单元回归通过，不要求随机序列与非投机推理相同 |

以上共 13 组贪心对照均通过，汇总为 `artifacts/mrope/mrope-comparison.json`。具体请求、生成内容、token ID、耗时和累计投机指标保存在 `artifacts/mrope/mrope-*.json` 与对应 `target-*.json`。拒绝回退和请求重排另外有单元测试；零、部分、全部接受通过构造的整数位置与槽位参考验证，不声称该低接受率 checkpoint 在上述自然生成样本中出现了全部七个候选接受。

## 普通 RoPE DFlash 回归

使用服务器已有 Qwen3-8B + Qwen3-8B-DFlash，以及 Qwen3.5-4B + Qwen3.5-4B-DFlash。后者的草稿为五层滑窗和一层全注意力，主模型是多模态模型，而草稿仍为普通 RoPE。

两种组合各覆盖候选数 7/15、batch 1/4、eager/PIECEWISE/FULL，共 **24 组改动前后对照**。所有测量重复的 token ID 完全一致。多模态主模型向普通 RoPE 草稿提供一维位置的已有回归保留，另增 m-RoPE 草稿保留三轴的回归。Qwen3.5 的实际图片输入也在下面的专项复测中通过。

初始矩阵吞吐取三次测量中位数。多个设备的模型初始化曾同时进行，这组结果用于筛查异常；不能将其中较大的正向变化直接解释为本次修改带来的加速。完整 24 行数据见报告末尾，原始汇总为 `initial-comparison.json`。

初测中 Qwen3-8B FULL K=15 batch=4 为 -8.34%，Qwen3.5 eager K=15 batch=1 为 -5.84%。随后在同一张物理 4 号卡上，基线和修改版依次独立运行、各测九次，未复现超过 5% 的下降。主机仍有其他用户服务，测试没有停止它们；以下是重复测量结论，不是对初测波动原因的确定归因。

| 九次测量的中位吞吐（token/s） | 基线 | 修改版 | 变化 |
| --- | ---: | ---: | ---: |
| Qwen3-8B，FULL，K=15，batch=4 | 47.567 | 47.409 | -0.33% |
| Qwen3.5-4B，eager，K=15，batch=1 | 11.720 | 11.260 | -3.92% |
| Qwen3.5-4B，eager，K=7，batch=1 | 12.436 | 11.840 | -4.79% |
| Qwen3.5-4B，eager，K=7，batch=2，图片输入 | 31.927 | 31.032 | -2.80% |

这四组的全部九次 token ID 和累计接受指标均相同。记录为 `performance-comparison.json` 及 `baseline-perf-*.json`、`new-perf-*.json`。

## 测试、静态检查与保留逻辑

* **149 项单元测试通过**：310P speculative decode、并行草稿注意力、FULL 图合同和执行、旋转、310P 采样及共享 rejection sampler。日志为 `final-unit-tests.log`。
* **132 项实机算子测试通过**：扩展现有 copy-and-expand DFlash/DSpark 测试到 K=7/15，并增加三轴位置与缓存逻辑位置分离后的完整整数对齐。日志为 `onboard-input-tests.log`。
* 十个修改或新增的 Python 文件通过语法、Ruff check、Ruff format check；`git diff --check` 通过。
* 真实 patch 入口测试确认运行时 Ascend proposer 和临时上游 DFlash proposer 的绑定生效，同时其他投机方法保留原限制。
* 隐藏层偏移、共享 embedding/LM head、缩减词表映射、非因果块注意力均沿用现有逻辑。三个服务器草稿均没有缩减词表，故缩减词表路径的结论限于代码未改动，没有额外声称它的实权重上机验证。
* 未执行整个仓库的所有 nightly 用例；执行范围为上述相关单元测试、已有并扩展的算子测试以及实际服务器模型矩阵。测试中的依赖弃用警告没有被隐藏。

复现单元与算子检查：

```bash
python -m pytest tests/ut/_310p/spec_decode \
  tests/ut/_310p/attention/test_parallel_draft_attention_310p.py \
  tests/ut/_310p/test_dflash_full_decode_acl_graph.py \
  tests/ut/_310p/test_dflash_full_decode_contract.py \
  tests/ut/_310p/test_dflash_full_decode_only.py \
  tests/ut/_310p/ops/test_rotary_embedding_310.py \
  tests/ut/_310p/sample tests/ut/sample/test_rejection_sampler.py -q

ASCEND_RT_VISIBLE_DEVICES=4 python -m pytest \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_copy_and_expand_dflash_inputs.py -q
```

## 接受率、性能和范围限制

首轮单图 eager 测试中，预热及一次测量累计提出 364 个候选，接受 12 个，约 **3.3%**。测得 DFlash 约 **9.95 token/s**，同配置主模型单独推理约 **16.31 token/s**；该组只测一次，不能作为稳定性能基准，但也没有加速证据。PIECEWISE 同一单图样本约为 13.28 token/s。

因此本次交付结论是位置、旋转和推理路径的适配及所列范围内的正确性回归通过，不能将它表述为该 checkpoint 已实现性能收益。接受率的进一步诊断和改善需要单独评估；本次没有重新训练、改权重或通过改变接受规则提高指标。

视频未纳入首轮范围。当前测试是列出的文字、合成图片和短生成样本，不能替代更大真实数据集的长时间稳定性测试。离线数值采集只在专用工具中开启，采集结束即移除钩子；最终性能测试没有启用采集、临时日志或 NPU→CPU 同步探针。

## 初始 24 组吞吐记录

| 用例 | 基线 token/s | 修改版 token/s | 变化 | token ID |
| --- | ---: | ---: | ---: | --- |
| qwen3-8b-eager-k15-b1 | 14.504 | 14.534 | +0.21% | 一致 |
| qwen3-8b-eager-k15-b4 | 41.803 | 41.530 | -0.65% | 一致 |
| qwen3-8b-eager-k7-b4 | 31.081 | 35.189 | +13.22% | 一致 |
| qwen3-8b-full-k15-b1 | 17.531 | 17.815 | +1.62% | 一致 |
| qwen3-8b-full-k15-b4 | 47.365 | 43.413 | -8.34% | 一致 |
| qwen3-8b-full-k7-b1 | 14.410 | 14.363 | -0.33% | 一致 |
| qwen3-8b-full-k7-b4 | 37.020 | 38.412 | +3.76% | 一致 |
| qwen3-8b-piecewise-k15-b1 | 17.798 | 17.729 | -0.39% | 一致 |
| qwen3-8b-piecewise-k15-b4 | 46.688 | 46.569 | -0.25% | 一致 |
| qwen3-8b-piecewise-k7-b1 | 14.517 | 14.472 | -0.31% | 一致 |
| qwen3-8b-piecewise-k7-b4 | 41.291 | 40.911 | -0.92% | 一致 |
| qwen3-k7-b1 | 12.108 | 11.721 | -3.19% | 一致 |
| qwen3.5-4b-eager-k15-b1 | 11.616 | 10.937 | -5.84% | 一致 |
| qwen3.5-4b-eager-k15-b4 | 29.016 | 35.047 | +20.79% | 一致 |
| qwen3.5-4b-eager-k7-b4 | 40.558 | 43.360 | +6.91% | 一致 |
| qwen3.5-4b-full-k15-b1 | 16.505 | 16.131 | -2.27% | 一致 |
| qwen3.5-4b-full-k15-b4 | 38.039 | 38.101 | +0.16% | 一致 |
| qwen3.5-4b-full-k7-b1 | 18.619 | 18.695 | +0.40% | 一致 |
| qwen3.5-4b-full-k7-b4 | 49.839 | 50.074 | +0.47% | 一致 |
| qwen3.5-4b-piecewise-k15-b1 | 14.971 | 15.132 | +1.08% | 一致 |
| qwen3.5-4b-piecewise-k15-b4 | 36.850 | 36.706 | -0.39% | 一致 |
| qwen3.5-4b-piecewise-k7-b1 | 17.706 | 17.417 | -1.63% | 一致 |
| qwen3.5-4b-piecewise-k7-b4 | 49.949 | 49.483 | -0.93% | 一致 |
| qwen35-k7-b1 | 12.915 | 12.614 | -2.33% | 一致 |

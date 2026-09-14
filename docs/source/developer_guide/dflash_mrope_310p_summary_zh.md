# 310P DFlash m-RoPE 修改总结

> 本文保留 47dbcb3 基线上的实现及实机记录；最新 f48bd99 分支在指定容器中的部署状态见 [容器安装记录](dflash_mrope_container_install_zh.md)。

## 范围与版本

本次为 Qwen3-VL-4B 的 DFlash 草稿增加 310P m-RoPE 推理支持。代码、测试、验证工具和文档均位于 vLLM-Ascend 仓库，未修改上游 vLLM、SpecForge 或模型权重，未重新训练或提交远程 PR。

| 项目 | 实际版本或位置 |
| --- | --- |
| Ascend 基线 | `47dbcb3bd7de18de1de910a3bbf708a22b8d22a4` |
| vLLM 基线 | `ee0da84ab9e04ac7610e28580af62c365e898389`，v0.24.0 |
| 修改工作区 | `/home/qzh/vllm-ascend-dflash-mrope` |
| 分支 | `feat/310p-dflash-mrope` |
| 独立基线工作区 | `/home/qzh/vllm-ascend-dflash-mrope-baseline` |
| 主模型 | `/home/models/Qwen3-VL-4B-Instruct` |
| 草稿 | `/home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750` |
| 设备与运行环境 | 310P3；`scc_dflash_dev`；FP16；torch 2.10 / torch-npu 2.10.0.post2 |

实机环境通过 `PYTHONPATH` 指向上述隔离源码。基线和修改版本使用同一份匹配的 Ascend 编译算子；版本、源码和算子校验值保存在 `artifacts/mrope/environment.json`。

## 训练框架与隐藏层说明

训练目录为 `/home/specforge-sglang/src/SpecForge-pr585`，本次以 checkpoint 中保存的配置、训练参数及 `dflash.py` 核对实际模型结构。

主模型有 36 层语言解码器。草稿选取 **零起始层号 `[3,10,18,25,32]` 的层输出**，对应 HF 返回的 `hidden_states[4,11,19,26,33]`；HF 的索引 0 是 embedding 输出，因此有一个索引偏移。这些是已经融合视觉信息的语言层特征，不是视觉编码器的层号。

每层特征宽度为 2560，五组特征拼接为 12800 维，再经 `fc` 投影到 2560 维及 `hidden_norm` 归一化。五个草稿层都从这份融合特征构造 context K/V，不能理解为一个主模型特征层只供一个草稿层使用。

```mermaid
flowchart LR
    A["主模型语言层 3 / 10 / 18 / 25 / 32"] --> B["拼接为 12800 维"]
    B --> C["fc 投影到 2560 维 + hidden_norm"]
    C --> D["五个草稿层各自构造 context K/V"]
    E["anchor + 7 个 MASK 的 embedding"] --> F["五层草稿网络"]
    D --> F
    F --> G["共享 LM head，得到 7 个候选 token"]
```

草稿有 5 层、32 个 Q 头、8 个 KV 头、head dimension 128，全部为 full attention。块大小 8 包含一个已知 anchor 和七个候选位置，mask token 为 151669。块内双向注意，context 位于 anchor 之前；训练损失计算候选位置，不计算 anchor。embedding 和 LM head 由主模型共享，该 checkpoint 没有缩减词表映射。

旋转配置为 theta 5000000、完整 128 维旋转、交错 m-RoPE，T/H/W 分段为 `[24,20,20]`。保存权重为 BF16，实机测试以 FP16 加载。保存训练参数为三轮、step 3750、batch 1、学习率 2e-4、最大长度 4096、最多 32 个 anchor、gamma 4、seed 42。保存配置中的五层设置覆盖了命令行层数默认值。

训练入口为 `scripts/train_dflash.py`，主要流程如下：

1. 数据加载器提供 token、attention mask、loss mask，以及图像像素和网格信息。
2. HF 目标模型在无梯度模式下计算特征和三轴位置；本次保存参数选择 HF backend。
3. `specforge/core/dflash.py` 根据可监督的位置采样 anchor，构造多个互相隔离的候选块。每个块只能访问 anchor 之前的 context 和本块 token。
4. 草稿以训练样本中的后续 token 为标签计算候选位置的交叉熵，反向传播后由 `BF16Optimizer(draft_model, ...)` 更新草稿参数，并保存模型及优化器状态。主模型用于提供特征及共享的 embedding/head。

## 修改内容

| 文件 | 修改及作用 |
| --- | --- |
| `vllm_ascend/patch/worker/patch_idex_310.py` | 将新方法绑定到实际 Ascend DFlash proposer；同时处理 GPU runner 初始化时临时创建的上游 DFlash proposer。没有放开其他投机方法的公共限制。 |
| `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py` | 按草稿的 `uses_mrope` 分支；预分配三轴缓冲；按请求当前排列读取 delta；分别构造旋转坐标和缓存 token 顺序坐标。 |
| `vllm_ascend/_310p/spec_decode/dflash_mrope.py` | 新增三轴位置和交错频率选择；维护草稿实例自己的 context/query cos/sin；刷新有效范围并将 padding 设为单位旋转。 |
| `vllm_ascend/_310p/spec_decode/dflash_model_310.py` | context K 在归一化后使用 context 三轴旋转，再走已有 NZ 缓存写入；query Q/K 使用独立 query 旋转。普通 RoPE 调用原 forward。 |
| `vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py` | 在预热、捕获、回放及 eager 回退前刷新草稿旋转缓冲，使图读取固定地址上的当前数据。 |

旋转坐标是 `[3,N]`，缓存顺序位置是 `[N]`。图片的旋转坐标可以重复，也可能与文本 token 的顺序位置相差一个 delta，不能用它们计算物理缓存槽位。现有 AscendC 算子继续处理一维缓存位置，物理 block size、滑窗及逐层槽位映射仍沿用已有逻辑。

生成文字的 query 三轴坐标为：有效序列长度减去拒绝的候选数，加上该请求的 m-RoPE delta，再加块内偏移。三轴 context 坐标与目标隐藏状态保持同样的排列和有效范围。请求重排时使用当前 input batch 的请求顺序读取 delta。

未完成的图片预填充也可能出现在补齐后的草稿 batch 中。其整段 prompt 的 delta 可能让临时 query 坐标为负，因此对这些随后被 runner 丢弃的候选使用有效占位坐标；完整 prompt 的有效文字位置不变。空 context 旋转直接返回，避免空张量 reshape 和无效算子调用。

主模型的旋转缓存不会被覆盖。草稿自己的频率表来自草稿配置，context 和 query cos/sin 分开管理；即使旋转模块被 vLLM 工厂缓存复用，也不修改该模块的共享状态。生产热路径没有新增 NPU→CPU 数值读取。

普通 RoPE 草稿仍走原来的一维路径，包含多模态主模型向普通草稿提供降维位置的情形。隐藏层偏移、共享 embedding/LM head、词表映射、块内非因果注意力保持现有实现。

## 验证方法与结果

已完成本轮适配和相关回归，详细范围、逐项数据与限制见 [回归报告](dflash_mrope_310p_regression_zh.md)。新增 m-RoPE 的 13 组贪心对照均与主模型单独推理逐 token 一致，覆盖 TP=1/2、eager、PIECEWISE、FULL、多图、混合请求、prefix cache、chunked prefill、异步调度和 EOS。这里 FULL 指基线已有的 FULL_DECODE_ONLY。

已完成的独立数值检查：

* 使用 HF Qwen3-VL 构造两种图像比例下的位置，包含缓存前缀后的 context 后缀和八个 query 位置。三轴位置与整数顺序位置完全一致。
* FP16 Q/K 的 NPU 旋转与 HF 旋转参考最大绝对误差为 0。
* 从实机采集第一块的实际输入，与 checkpoint 保存的训练实现逐层比较。五层相对 RMS 误差为 **0.0766%–0.0979%**，最终输出为 **0.0811%**；余弦相似度均超过 0.999996。

该逐层比较覆盖单图第一块、88 个 context token 和 8 个 query token。它验证草稿前向对相同输入的数值一致性，不等同于对所有请求及所有生成步逐层比对。实机调度正确性另通过主模型单独推理的贪心输出对照检查。

新增和已有相关单元测试共 149 项通过，包含真实绑定入口、三轴保留、缓存隔离、拒绝回退、物理槽位、相邻 DSpark 路径、图执行及共享采样器的随机接受/恢复分布回归。另有 132 项实机算子测试通过，包含 K=7/15、物理 block size 64/128，以及零、部分、全部候选接受时的整数位置和槽位对照；语法及 Ruff 检查通过。

普通 RoPE 的两种服务器模型组合完成 24 组改动前后回归，token ID 均一致。对初测异常及补充图片场景进行了同卡、串行、各九次的性能复测，四组中位吞吐变化为 -0.33%、-3.92%、-4.79%、-2.80%，均未超过 5% 的下降阈值，接受指标一致。

首轮单图测试中接受率较低，因此当前不能声称该 checkpoint 已带来加速。正确性、接受率和吞吐在报告中分别列出。采样模式不要求与主模型单独推理产生同一随机序列。

## 启动与复现

在已配置匹配依赖及 Ascend 算子的仓库根目录执行：

```bash
python -m tools.run_310p_dflash_mrope_validation \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --draft /home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750 \
  --image --max-tokens 64 --repeats 3 \
  --output artifacts/mrope/vl-image-eager.json
```

默认 TP=1、FP16、同步调度、eager、关闭 prefix cache 和 chunked prefill，物理 block size 为 128。这里的物理 block size 与训练的候选块大小 8 是两个概念。

同配置服务启动示例如下。需要在有设备访问权限、并挂载上述工作区的容器中执行；设备编号应选择当时空闲的卡。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/home/qzh/vllm-ascend-dflash-mrope:/home/qzh/vllm-v0.24.0-clean:${PYTHONPATH}"
ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /home/models/Qwen3-VL-4B-Instruct \
  --dtype float16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --block-size 128 --gpu-memory-utilization 0.75 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":50176}' \
  --speculative-config '{"method":"dflash","model":"/home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750","num_speculative_tokens":7}'
```

添加 `--mode piecewise` 或 `--mode full` 可测试图执行；后者对应 310P 已有 `FULL_DECODE_ONLY` 配置，decode 实际走 FULL，prefill 保留相应回退行为。删除 `--draft` 可运行主模型贪心参考。工具还支持多请求、多图、图文混合、prefix cache、chunked prefill、异步调度、TP=2 和 EOS 检查，详见工具帮助。

数值参考工具为 `tools/validate_310p_dflash_mrope_reference.py` 和 `tools/validate_310p_dflash_layers.py`。后者的采集扩展只用于离线诊断，在采集后移除钩子，不属于生产推理路径。性能测试不启用采集。

本轮范围为文字/图像输入、文字输出；视频单独扩展。没有改变请求格式、checkpoint 格式或新增必填 DFlash 参数。

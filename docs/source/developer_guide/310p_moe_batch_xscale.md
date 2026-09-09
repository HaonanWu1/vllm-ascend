# 310P MoE 整批 x_scale 优化

## 实现与边界

本改动继续调用 CANN `npu_quant_grouped_matmul_dequant`，只把每行激活的缩放因子提前整批计算：
`max(abs(x[row])) / 127`，归约后转 FP32 再除法，全零行使用 1。量化、整数矩阵乘和反量化仍由 CANN 完成。
两次 GMM 分别根据当前输入重新计算 scale；不复用 scale，不增加 CPU 同步或诊断日志。

仅在下面两个已验证的 shape 组合及 FP16 X、INT8 NZ 权重、FP32 weight scale、INT64 group list 下启用。
其他输入保持 CANN 内部动态量化路径，不按模型名称分派。

| X shape | 权重逻辑 shape (E,N,K) | 本次验证位置 |
| --- | --- | --- |
| (640, 2048) | (256, 512, 2048) | gate/up GMM |
| (640, 256) | (256, 2048, 256) | down GMM |

这些 shape 对应已测模型 TP2、K7、C10 的满批 verification。
K15/C10 等其他行数不在当前优化范围，不能推断同样收益。没有修改 WY、greedy、GDN、缓存或图路由。
本次只有 Python 路径修改，无需重新编译原生算子；更新代码后需重启推理服务。

## 已有验证结果

2026-09-08，基于 `6e794469` 及两版共同保留的四个本地 GDN 修改，使用真实 Qwen3.6-35B-A3B-w8a8 权重和 DFlash。
该 GDN 差异不属于本提交，下面结果不是对纯净远端工作树重新测得的结果。

- TP2、K7、C10，图开启 `FULL_DECODE_ONLY [8,80]`；正式测速关闭 profiler 和数值探针。
- 模型内同输入对照：2 ranks × 20 个满批步骤 × 40 层 × 2 次 GMM = 3200 次，逐元素一致，最大绝对误差 0。
- 端到端：每版 50 条、约 4K 输入/2048 输出，共 100/100 请求通过校验。
- 相同输入、相同物理卡，两版 KV 容量均为 75473 tokens，抢占计数均为 0。
- 前缀缓存开启，数据包含 50% 重复前缀，但两版实际缓存命中均为 0。

| 指标 | 原版 | 整批 scale 候选 |
| --- | ---: | ---: |
| 平均 TPOT (ms) | 70.9 | 63.4 |
| 输出吞吐 (tokens/s) | 125.7130 | 143.7145 |
| 平均 TTFT (ms) | 7377.1 | 7909.4 |
| 有效并发 | 9.3651 | 9.6687 |
| 投机接受率 | 57.8688% | 60.2608% |

单轮 A/B 观察到 TPOT 降低 10.6%、吞吐提高 14.3%，但 TTFT 均值增加 7.2%。
有效并发和接受率也变化，不能将全部端到端收益归因于 GMM 本体。
这不是标准模型准确率评测，也未验证其他模型、K15、EP 或其他图模式。
整理提交时仅调整辅助函数命名、类型注解、常量及格式，不扩大已测候选的计算逻辑。

## 回归检查与微基准

CPU 单元测试覆盖两次 GMM 的 scale、全零/负值/小值/大值行、逐次重算、参数透传、两种 group list 格式以及未验证 shape/dtype/layout 的原路径：

```bash
python -m pytest -q tests/ut/_310p/fused_moe/test_moe_mlp_310.py
```

在已安装本仓代码的 310P 环境中，可使用可信输入快照复测。每个 `.pt` 包含 CPU tensor：
`x`、`weight`（逻辑 E,N,K）、`scale`、`groups`（累积结束位置）。权重及用户输入不随代码提交。

```bash
python benchmarks/scripts/benchmark_quant_gmm_scale_310p.py \
  /path/to/gate_up.pt /path/to/down.pt --device 0 --iterations 100 --rounds 5
```

脚本先要求候选与原版逐元素一致，再分别输出 eager/图模式的原版、候选每轮耗时和中位数。
候选每次执行都包含 scale 计算，不把预计算 scale 当作收益；比较和 CPU 复制在计时之外。
这是固定输入微基准，不代替端到端压测。共享卡负载会影响结果，应在独占空闲设备上复测。

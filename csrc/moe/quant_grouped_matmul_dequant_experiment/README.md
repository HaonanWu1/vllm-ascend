# 310P MoE 小分组量化实验（未接入默认构建）

这是隔离实验，不是可直接安装的正式自定义算子。没有替换共享 CANN，
也没有修改 vLLM、torch_npu 或默认 vllm-ascend 安装。

## 来源与范围

`op_kernel` 来自当前 CANN 9.1.0-beta.1 自带的
`quant_grouped_matmul_dequant` 源码，保留原版权和 CANN Open Software
License Agreement Version 2.0 声明。正式集成或对外发布前，需要完成
许可证核对、独立算子注册及构建接入；不要将本目录视为原创完整矩阵乘实现。

相比原始学习副本，算法修改仅在 `quant_grouped_matmul_dequant_gemv.h`；
其余文件只可能有末尾换行或空白整理：

- 在 `ProcessX()` 增加受约束的小分组分派。
- 新增 `ProcessXBatchedDecode()`，将逐行搬运及类型转换改为整组处理。
- 逐行 FP32 scale、除数倒数、RINT 舍入、INT8 转换、矩阵乘和反量化顺序不变。
- 保留跨流水依赖；不是简单删除屏障。
- 原 GEMV 阈值仍为 8，没有采用筛选时尝试过的 16 行阈值。

当前只验证以下外部 per-token scale 路径：

| 项目 | gate/up | down |
| --- | --- | --- |
| X 逻辑形状 | 640 × 2048 | 640 × 256 |
| W 逻辑形状 | 256 × 512 × 2048 | 256 × 2048 × 256 |
| 类型/格式 | FP16 X、INT8 NZ W、FP32 weight scale | 相同 |
| 分组 | 累计 INT64 group_list；每专家 1–8 行走新增路径 | 相同 |

其他形状、内部动态量化、smooth scale、INT64 weight scale 或较大分组保留
原路径。分派依据张量契约，不依赖模型名称或 K 的数值。K=15 未测，不能
宣称 K=15 已优化；也不会为命中本路径改变图档位或填充输入。

最大新增工作区布局使用原有 UB 的前 114720 字节：
FP16 16384 个、FP32 16384 个、INT8 16384 个、FP32 scale 8 个。
这些缓冲区互不重叠；矩阵乘和输出阶段按原有协议复用内存。

## 验证与复现

Windows 结果与实验脚本：
`D:/vllm-ascend-0.21.0rc2/vllm-ascend-0.21.0rc1/experiments/moe_decode_20260916`

服务器同名实验目录：
`/home/qzh/vllm024_dflash_debug/moe_decode_20260916`

- `screen.py`：现成量化接口和 scale 表达式筛选，未采用。
- `build_kernel.py`：通过已安装编译入口，仅编译私有副本。
- `kernel_bench.py`：原版、原源码重编、16 行阈值、整组量化微基准。
- `validate_kernel.py`：真实权重及路由、边界输入、图重放、非目标形状回归。
- `run_e2e.py`：真实模型端到端 A/B；固定 K7、C10、FDO `[8,80]`。

实验利用进程专属 OPP 目录验证候选，除了目标 `.o` 和 `.json` 文件，
其余内容链接原工具链。该方法只用于隔离诊断，依赖当前 CANN 版本，
不应作为正式部署方案。

## 已验证的检查点

2026-09-16：6 组真实权重（rank0，第 0/20/39 层、gate/up 和 down），
252 个图重放/边界/非目标形状检查与安装版逐元素一致。
两批 C10、4K 输入/2K 输出的平均 TPOT 为 51.13 → 49.85 ms；
5 轮 decode trace 的 rank0 GMM 累计为 80.62 → 71.47 ms/轮。
样本较少，C10 生成结果及接受率存在波动，不等于完整模型精度验收。

仓内离线回归入口：
`tests/e2e/pull_request/one_card/ops/quant_grouped_matmul_dequant_experiment.py`。
使用同一份真实 capture，在两个独立进程中运行：

```bash
# 已安装的原始算子；输出目录必须不存在。
ASCEND_RT_VISIBLE_DEVICES=4 ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
python tests/e2e/pull_request/one_card/ops/quant_grouped_matmul_dequant_experiment.py \
  --capture /path/to/capture --output /path/to/reference

# 私有候选算子；不更改全局工具链或默认安装。
ASCEND_RT_VISIBLE_DEVICES=4 ASCEND_OPP_PATH=/path/to/private/opp \
python tests/e2e/pull_request/one_card/ops/quant_grouped_matmul_dequant_experiment.py \
  --capture /path/to/capture --reference /path/to/reference --output /path/to/check
```

执行前需确认所选设备空闲；capture 包含 `layer*.pt` 和
`groups_rank0.jsonl`。未注册的原型不会由默认服务自动调用。
这是已有模型的受限算子实验，不新增模型适配、128K/VL/EP 验收声明。

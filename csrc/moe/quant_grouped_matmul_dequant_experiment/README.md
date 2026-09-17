# 310P MoE 小分组量化实验（未接入默认构建）

这是隔离实验，不是可直接安装的正式自定义算子。没有替换共享 CANN，
也没有修改 vLLM、torch_npu 或默认 vllm-ascend 安装。

## 来源与范围

`op_kernel` 来自当前 CANN 9.1.0-beta.1 自带的
`quant_grouped_matmul_dequant` 源码，保留原版权和 CANN Open Software
License Agreement Version 2.0 声明。正式集成或对外发布前，需要完成
许可证核对、独立算子注册及构建接入；不要将本目录视为原创完整矩阵乘实现。

2026-09-16 检查点相比原始学习副本，算法修改在
`quant_grouped_matmul_dequant_gemv.h`：

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
`groups_rank0.jsonl`；若有 `groups_rank1.jsonl` 也会读取。每项默认重放
5 次，对比 FP16 存储位（包含正负零），并检查 M=64/639/641 的原路径。
参考版首次输出会冻结，后续重放也必须与它一致。
未注册的原型不会由默认服务自动调用。
这是已有模型的受限算子实验，不新增模型适配、128K/VL/EP 验收声明。

## 2026-09-17 追加实验：小专家分核

此检查点另修改 `quant_matmul_dequant_grouped.h`：gate/up 仅在
E=256、M=640、K=2048、N=512、8 核及上述外部 per-token scale 条件下，
先用 4 组、每组 2 核处理不同的小专家，再由全核按原顺序处理大专家。
没有改变逐输出的矩阵乘累加和量化取整顺序；本节实验的 down 不变。
当前检查点还包含下节的 down 扩展，非目标形状保留原路径。

同卡 A-B-B-A 的三个真实 gate/up 输入耗时下降约 11%～14%。旧 252 项
回归数值一致；新增 120 组输入各 5 次图重放，FP16 存储位全部一致。
这些不等于全模型精度验收或端到端性能结论。固定模分配可能因路由
分布不均而退化，尚未作为默认路径集成。

本轮完整记录：
`D:/vllm-ascend-0.21.0rc2/vllm-ascend-0.21.0rc1/experiments/moe_stage_20260917/实验说明.md`。
服务器对应目录为 `/home/qzh/vllm024_dflash_debug/moe_stage_20260917`。

## 2026-09-17 追加实验：down 非空小专家轮流分核

本工作树保留已验证的 gate/up 四组分核，并单独扩展 down 的
K=256、N=2048 路径；E=256、M=640、8 核及 scale 限制保持不变。
down 使用四组、每组两核处理不同的小专家，每核负责 N 的一半。
按实际非空小专家的序号轮流分配，而不是专家编号取模，避免编号集中时
某组忙、其他组闲。大专家仍在第二阶段使用原来的全核路径。

只调整任务归属与 N 列划分，没有改变逐输出的量化、INT32 累加、两次
FP32 反量化乘法及 FP16 舍入顺序。使用原有局部缓冲区和同步协议，
没有新增 NPU→CPU 同步、运行时环境变量或全局可变状态。

未采用最初的静态编号取模版：四组版在构造的偏斜路由中最差慢约 84%。
改成非空小专家轮流分配后，该组偏斜样例均改善；这不保证任意路由都加速。
gate/up 的分配方式未更改，它仍有原文注明的静态分配局限。

同卡 A–B–B–A 图重放中，三个真实 down 输入耗时分别约
925→814、645→603、837→748 微秒，gate/up 对照基本不变。
834 次 FP16 存储位检查全部一致，覆盖真实路由、偏斜、8/9 行边界、
零值/随机值/舍入边界，以及 M=64/639/641 的原路径；另有 1539 项
CPU 任务归属和大专家同步顺序模拟检查。这些不代替全模型精度评估。

实验脚本与结果在 D 盘项目的 `experiments/moe_down_20260917`；服务器为
`/home/qzh/vllm024_dflash_debug/moe_down_20260917`。端到端结果以该目录
的 `clean_summary.json` 和中文实验记录为准。K7/C10、4K/2K、FDO `[8,80]`
同卡 A–B–B–A 中，平均 TPOT 48.37→47.41 ms，观测改善 1.98%；同一
输入的全部输出 token、接受率与模型迭代数一致。每版仅两次批次，
不等于任意负载的收益保证。该版本仍未接入默认安装。

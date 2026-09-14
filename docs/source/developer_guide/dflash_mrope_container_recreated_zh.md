# DFlash 容器特权模式重建记录

日期：2026-09-10。本记录更新此前 [诊断材料](dflash_mrope_container_diagnosis_zh.md) 中“原容器尚未替换”的状态。

## 已执行的操作

1. 检查原容器没有运行中的模型任务，保存配置记录。
2. 保存包含代码、Python 安装、编译算子及容器内部文件的镜像快照。
3. 停止原容器，改名保留为备份。
4. 使用快照镜像，以原名称创建特权容器；重新挂载宿主 `/home`、驱动和相关文件。
5. 检查安装路径和编译扩展，运行实际 NPU 运算及模型验证。

| 项目 | 当前值 |
| --- | --- |
| 正在运行的容器 | `qzh_v023_vllm024_dflash` |
| 停止保留的原容器 | `qzh_v023_vllm024_dflash_backup_20260910` |
| 快照镜像 | `qzh-dflash-mrope:f48bd99-20260910` |
| 镜像 ID | `sha256:6f81e59aad532f0f5a93722721f188ee8ed23f8b915691cac67f11fb23b46d3d` |
| 新容器 ID | `4f7f800d669aca64146eb1670d02aeeec3644381a3c45acc628eb2a8561fb544` |
| 代码工作区 | `/vllm-workspace/vllm-ascend-dflash-mrope-latest` |
| 代码基线 | `f48bd99fd7b9458bbad9c61f36a10b2217eb40b3`，叠加本次 m-RoPE 补丁 |
| 安装版本 | `0.1.dev26+gf48bd99fd.d20260910` |

新容器使用 `--privileged --net=host --ipc=host --pid=host`，不逐张配置 `--device`；驱动、npu-smi、dcmi、安装信息、hccn.conf、add-ons 均保持只读挂载，`/home` 绑定宿主目录。容器入口显式设为 `/bin/bash`，工作目录设为新代码工作区。

原 `/vllm-workspace/vllm-ascend` 中四个 GDN 相关未提交改动及其他文件随快照保留；没有清理旧目录、覆盖它们或重新训练模型。宿主 `/home` 和驱动原本就是外部挂载，不包含在镜像快照中，新容器通过挂载继续使用。

## 验证结果

* 新容器 `npu-smi info` 成功，之前的 `-8020 / aclInit 507899` 初始化失败已消除。
* 从 `/tmp` 导入，确认 Python 包来自 `/vllm-workspace/vllm-ascend-dflash-mrope-latest`。
* 编译扩展 SHA256 仍为 `9b35ffbd578f73e8bb0b530725bc9c52500a46fad5d77ae70db7cba504428e10`，与重建前一致。
* 新容器内实际 NPU 加法 `[1,2]+1` 返回 `[2,3]`。
* 下列模型验证全部完成，测试进程正常退出。

| 模型组合 | 本轮配置 | 结果 |
| --- | --- | --- |
| Qwen3-VL-4B-Instruct 单独推理 | FP16、TP=1、eager，三个图文混合请求，每个生成 32 token | 成功，作为贪心参考 |
| Qwen3-VL-4B-Instruct + 指定五层 m-RoPE checkpoint | 同上，7 个候选，eager | 与主模型参考逐 token 一致 |
| 同上 DFlash 组合 | FULL_DECODE_ONLY，三个图文混合请求，每个生成 32 token | 图捕获及回放成功，输出与主模型参考及 eager 一致 |
| Qwen3-8B + Qwen3-8B-DFlash | FP16、TP=1、eager，7 个候选，两个请求各生成 16 token | 成功 |
| Qwen3.5-4B + Qwen3.5-4B-DFlash | FP16、TP=1、eager，15 个候选，两个请求各生成 16 token | 成功 |

FULL 测试记录显示 target/draft 共四个图条目，descriptor 为 `[8,64]`。贪心对照汇总见 `recreated-comparison.json`。普通 RoPE 两组在本轮验证了生成连通性，没有在本次重建任务中重复执行完整性能矩阵。

本轮模型检查属于重建后的推理连通性和输出对照，不代替旧基线完整回归报告，也不重新声称性能收益。设备可见不代表显存空闲；Qwen3-8B 首次在物理 3 号卡启动时仅有 30.57 GiB 空闲，不足所设 32 GiB 预算，因此改用其他测试结束后释放的卡重试，未停止他人任务。

## 使用方式

进入容器：

```bash
docker exec -it qzh_v023_vllm024_dflash /bin/bash
cd /vllm-workspace/vllm-ascend-dflash-mrope-latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

示例验证命令如下，运行前应确认所选物理卡有足够空闲显存。这里的 4 是本轮使用的设备编号，不是永久保留的资源。

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m tools.run_310p_dflash_mrope_validation \
  --model /home/models/Qwen3-VL-4B-Instruct \
  --draft /home/xj/checkpoints/qwen3-vl-4b-dflash-textvqa-10k-epoch3-5layer-block8/epoch_3_step_3750 \
  --image --max-tokens 32 --repeats 1 \
  --output /home/qzh/vllm-ascend-dflash-mrope/artifacts/container-latest/manual-vl-dflash.json
```

默认 FP16、TP=1、eager、7 个候选 token。添加 `--mode full` 测试 FULL_DECODE_ONLY。测试完成后释放模型进程；容器保留运行，未创建常驻模型 API 服务。

## 记录位置

记录均位于 `/home/qzh/vllm-ascend-dflash-mrope/artifacts/container-latest/`：

* `pre-recreate-inspect.json`：原容器配置。
* `snapshot.json`：快照镜像信息。
* `recreated-container.json`：新容器信息和完整创建参数。
* `recreated-runtime.json`：导入路径、版本、扩展校验值、NPU 运算结果。
* `recreated-*.json` 和对应日志：模型验证结果。

原容器与快照镜像均保留，未删除备份。旧容器保留原来的非特权设备映射；直接回到旧容器也会恢复此前的设备访问限制。

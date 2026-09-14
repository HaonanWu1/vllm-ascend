# DFlash m-RoPE 最新分支迁移与容器安装记录

日期：2026-09-10。

## 代码位置和版本

目标容器为 `qzh_v023_vllm024_dflash`。已从 `https://github.com/HaonanWu1/vllm-ascend.git` 拉取 `dev_v24/feat-310p-dflash`，本轮迁移基线为：

```text
f48bd99fd7b9458bbad9c61f36a10b2217eb40b3
fix(310p): correct chunk FwdO causal masking
```

在容器内建立工作区 `/vllm-workspace/vllm-ascend-dflash-mrope-latest`，分支为 `feat/310p-dflash-mrope-latest`。本轮不提交远程 PR。

原目录 `/vllm-workspace/vllm-ascend` 保持在 `perf/310p-moe-batch-xscale` 分支，原有四个 GDN 相关未提交文件及 `export_only_prof_dir/` 保留。新工作区承载本次适配，避免将用户已有改动混入迁移。

## 合入内容

原适配基于 Ascend `47dbcb3`。最新分支较它增加了图执行、prefix cache、SplitFuse 和若干算子修改，因此按补丁合入，保留最新分支的实现。

* 草稿三轴旋转位置、独立的一维缓存位置、context/query 独立 cos/sin 和实际 proposer 绑定均已合入。
* `llm_base_proposer_310.py` 的刷新入口手动合并：m-RoPE 分支使用草稿实例缓冲，普通 RoPE 保留最新 PIECEWISE、FULL_DECODE_ONLY 和 FULL_AND_PIECEWISE 逻辑。
* 原有数值参考工具、模型验证工具和说明材料一并迁移。
* 新基线的三个 CPU 图注意力单元测试依赖 `_npu_paged_attention` 模拟接口，原 `tests/ut/conftest.py` 的 CPU stub 未提供。补齐该模拟接口后，393 项相关单元测试全部通过；没有改动生产算子来规避测试。

旧版 [修改总结](dflash_mrope_310p_summary_zh.md) 和 [回归报告](dflash_mrope_310p_regression_zh.md) 记录的是 `47dbcb3` 基线和 `scc_dflash_dev` 上的历史实机结果，不能将其中的 24/13 组模型对照及性能数字直接当作本容器最新版本的验收结果。

## 安装

在新工作区使用原有 Python、vLLM 0.24.0、torch 2.10.0 和 torch-npu 2.10.0 依赖，重新构建 Ascend 自定义算子及扩展，并进行 editable 安装：

```bash
docker exec -it qzh_v023_vllm024_dflash bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /vllm-workspace/vllm-ascend-dflash-mrope-latest
SOC_VERSION=ascend310p1 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  python -m pip install -e . --no-build-isolation --no-deps
```

安装已成功，版本为 `0.1.dev26+gf48bd99fd.d20260910`。从 `/tmp` 导入时确认源码与扩展均来自新工作区，editable 安装路径正确。扩展中的 `npu_copy_and_expand_dflash_inputs`、`adn_rms_norm`、`npu_rejection_sample_greedy_310` 均已注册。记录为 `installed-environment.json`；扩展 SHA256 为 `9b35ffbd578f73e8bb0b530725bc9c52500a46fad5d77ae70db7cba504428e10`。

运行时应进入上述新目录，避免从旧源码目录启动导致当前目录优先于 editable 安装路径。主模型和 checkpoint 路径沿用原配置；不修改权重或上游 vLLM。

## 检查和当前限制

* 393 项相关单元测试通过，涵盖 speculative decode、310P attention、FULL/PIECEWISE 路由、旋转及 rejection sampler。
* 修改 Python 文件的语法和 Ruff 检查通过。
* 容器内 NPU 初始化仍失败：仅执行 `torch.npu.set_device(0)` 就在 `aclInit` 阶段报 `507899 / Resource_Busy / 0x7020010`，发生在主模型或草稿加载前。普通执行和 `docker exec --privileged` 的探测均复现。
* 容器配置映射了 0–7 号设备；主机还运行着其他用户的 NPU 服务。本轮没有重置设备、停止其他服务或重建容器。设备枚举失败的具体原因仍需进一步核查，当前不能声明该容器内的模型推理已跑通。

昇腾的[容器设备访问故障说明](https://www.hiascend.com/document/detail/zh/canncommercial/601/troublemanagement/troubleshooting/troubleshooting_0055.html)列出了设备被其他容器占用导致枚举失败的情形，可作为后续环境排查参考；本轮诊断依据以实际探测日志为准。

日志位于容器可见路径 `/home/qzh/vllm-ascend-dflash-mrope/artifacts/container-latest/`：`install.log`、`install-resume.log`、`unit-tests-final.log`、`device-probe.log`、`device-privileged-probe.log`。

单元测试在没有可用 NPU 的容器中使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 运行，防止真实 torch-npu 自动加载与 CPU stub 冲突。该变量只用于测试进程，不写入模型启动配置。

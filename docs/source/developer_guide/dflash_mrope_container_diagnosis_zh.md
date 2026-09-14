# qzh DFlash 容器 NPU 初始化失败诊断

诊断日期：2026-09-10。本文补充此前的 [容器安装记录](dflash_mrope_container_install_zh.md)。

## 结论

故障来自 **物理 7 号 NPU 的非特权容器设备分配冲突**。`qzh_v023_vllm024_dflash` 映射全部 0–7 号设备，而 7 号设备已经由运行中的非特权容器 `vllm-ascend-merged-g60108801` 使用。即使该卡没有正在运行的计算进程，容器对设备的分配仍存在。

冲突发生在设备枚举阶段，导致 `npu-smi` 报 `device is used / -8020`，随后 torch-npu 初始化报 `aclInit 507899 / Resource_Busy / 0x7020010`。它早于 vLLM、模型权重或 DFlash 的加载。

## 对照证据

临时诊断容器均使用 qzh 原镜像 `quay.io/ascend/vllm-ascend:v0.23.0rc1-310p`，保持非特权模式，挂载相同宿主驱动和管理工具，只改变 NPU 映射。诊断容器禁用网络并在退出后自动删除。

| 配置 | 检查 | 结果 |
| --- | --- | --- |
| 原 qzh 容器，映射 0–7 | `npu-smi info` | 失败，`-8020` |
| 临时容器，只映射 7 | 设备查询 | 同样失败，`-8020` |
| 临时容器，映射 0–6，排除 7 | 设备查询 | 成功 |
| 临时容器，映射 3、4 | 设备查询 | 成功 |
| 临时容器，映射 3、4、7 | 设备查询 | 失败，`-8020` |
| 临时容器，只映射 4 | 设备查询与实际 NPU 运算 | 成功；设备数 1，`[1,2]+1` 返回 `[2,3]` |
| 临时容器，映射 0、4 | 设备查询 | 成功；说明并非所有正在使用显存的卡都会造成同样的枚举冲突 |
| 已有 `vllm-ascend-merged-g60108801`，映射 7 | 使用其已有 `/usr/local/npu-smi info` | 成功，显示 PCI `0000:83:00.0`，当时没有 NPU 计算进程 |

已有 7 号卡容器在 `2026-09-09T11:55:08Z` 启动，早于本轮 qzh 容器的 `2026-09-10T02:42:39Z`。其容器内逻辑设备 0 对应宿主机物理设备 7，不能混淆两种编号。

qzh 与可用的 `scc_dflash_dev` 挂载的宿主驱动 `libascend_hal.so` 和 `version.info` 的 SHA256 完全一致。qzh 的 device cgroup 已允许所有声明的设备，并非漏加 `/dev/davinci4` 的访问权限。对照中未发现 SELinux AVC 拒绝；无需通过关闭宿主机 SELinux来解释或解决本次问题。

原 qzh 容器即使指定 `ASCEND_RT_VISIBLE_DEVICES=4` 也初始化失败。该运行时选择不能代替容器创建时的设备映射，因为当前故障发生在底层枚举映射设备时。此前 `docker exec --privileged` 也未改变原容器的冲突状态。

## 建议的处理方式

保留当前容器的代码、安装结果及其他文件，再调整容器创建配置，只映射当时可分配的设备。例如，本次已验证可查询的物理 3、4 号卡可作为 TP=2 的后续候选；单卡 4 号已经通过实际运算测试。保留管理设备 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` 和现有驱动挂载。

不应只修改 `ASCEND_RT_VISIBLE_DEVICES` 而继续映射 7 号卡。若必须使用 7 号卡，需要先协调其现有容器释放设备。本轮没有停止该容器、重置 NPU 或修改其配置。

普通 `docker exec` 不能替换已有容器的设备映射；后续应在保留现有文件和可回滚状态后重建配置。此次任务完成的是原因定位，尚未重建 qzh 容器，因此原容器当前的初始化错误仍然存在。临时容器的运算成功不等同于已在 qzh 容器内重新通过模型推理测试。

原始对照结果：`/home/qzh/vllm-ascend-dflash-mrope/artifacts/container-latest/device-mapping-diagnosis.json`。临时诊断容器已全部删除。

昇腾官方 [调用 Device 失败的故障说明](https://www.hiascend.com/document/detail/zh/canncommercial/601/troublemanagement/troubleshooting/troubleshooting_0055.html)也列出了其他容器占用设备导致设备列表获取失败的情形，并说明需要释放占用设备的容器。本次具体定位到物理 7 号卡的结论来自上面的现场对照。

## 特权容器方案补充验证

随后按用户提出的配置，使用相同原镜像建立临时容器：`--privileged --net=host --ipc=host --pid=host`，挂载 `/home`、宿主驱动和管理工具，不逐张声明 `--device`。设备查询正常，指定物理 4 号卡的最小 NPU 加法返回 `[2.0, 3.0]`，进程退出码为 0。临时容器已自动删除，原 qzh 容器未替换。

因此，特权容器是本环境中已验证可行的另一条路径。关键是容器创建时的 `--privileged`；已有正常工作的 `scc_dflash_dev` 同样采用特权模式，同时保留独立的 PID 命名空间，所以 `--pid=host` 不是修复此次枚举冲突的必要条件。现有原容器内单次 `docker exec --privileged` 与从创建时就使用特权模式的结果不同。

该方式解决设备枚举问题，仍需选择当时有足够显存的卡执行模型任务，不能把设备可见等同于资源空闲。

迁移时应先保留当前容器的文件系统和安装结果，再用保存的镜像创建新容器。代码和 editable 安装目标位于容器内部 `/vllm-workspace/vllm-ascend-dflash-mrope-latest`，仅挂载 `/home` 不会自动带出这部分内容；直接使用原始镜像新建将不会包含本轮安装。

实测日志为 `privileged-container-check.log`，对应启动参数为 `privileged-container-check-command.json`。Ascend [RecSDK 官方容器说明](https://github.com/Ascend/RecSDK/blob/develop/docker/OVERVIEW.zh.md)也说明非特权容器的设备独占行为，以及使用特权容器的处理方式。

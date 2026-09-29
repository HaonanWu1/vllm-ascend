# SPDX-License-Identifier: Apache-2.0
"""T01: host memory budget decisions; device readings are explicit external inputs."""

import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def production():
    # CPU conftest lacks optional torch_npu ATB/profiler import modules. These
    # are device-library boundaries, not replacements for either worker class.
    missing = {}
    for name in ("torch_npu.op_plugin.atb._atb_ops", "torch_npu.profiler"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            missing[name] = MagicMock()
    with patch.dict(sys.modules, missing):
        yield importlib.import_module("vllm_ascend._310p.worker_310p")


@pytest.mark.parametrize(
    "rc,after_free,expected",
    [(False, 600, 350), (True, 600, 600), (False, 1000, None), (False, 1100, None)],
    ids=["discrete-memory", "rc-host-memory", "equal-free-reject", "freed-during-profile"],
)
def test_memory_budget_and_concurrent_release_guard(production, rc, after_free, expected):
    """T01-001：profiling后真实预算算术与内存外部释放guard。

    输入：requested1800、初始free1000、total2000、profile时free800、reserved500；
      profile nonKV600/nonTorch200/peak100；RC主机total3000/available2400。
    输出：离散卡(1800-600-(700-200))/2=350；RC(1800-600)/2=600。
      after_free>=1000表示外部释放，必须抛AssertionError且不发布预算。
    场景：workspace预留一半，不能遗漏cache清理前后的nonTorch差额。
    替身：只有设备内存读数、profile context和模型profile_run；预算生产函数完整执行。
    局限：host合同UT，不证明NPU实际显存或OOM行为。
    """
    worker = production.NPUWorker310.__new__(production.NPUWorker310)
    calls = []
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    worker.init_snapshot = SimpleNamespace(free_memory=1000)
    worker.requested_memory = 1800
    worker.model_runner = SimpleNamespace(model_memory_usage=300, profile_run=lambda: calls.append("profile"))
    profile = SimpleNamespace(
        non_torch_increase=200,
        torch_peak_increase=100,
        non_kv_cache_memory=600,
        after_profile=SimpleNamespace(free_memory=after_free),
    )

    @contextmanager
    def profiling(snapshot, weights_memory):
        assert snapshot is worker.init_snapshot and weights_memory == 300
        yield profile

    with (
        patch.object(production, "memory_profiling", profiling),
        patch.object(production, "is_rc_device", return_value=rc),
        patch.object(production.torch.npu, "mem_get_info", return_value=(800, 2000)),
        patch.object(production.torch.npu, "memory_reserved", return_value=500),
        patch.object(production.psutil, "virtual_memory", return_value=SimpleNamespace(total=3000, available=2400)),
    ):
        if expected is None:
            with pytest.raises(AssertionError, match="memory profiling"):
                worker.determine_available_memory()
            assert not hasattr(worker, "available_kv_cache_memory_bytes")
        else:
            budget = worker.determine_available_memory()
            assert budget == expected
            assert worker.available_kv_cache_memory_bytes == budget
            assert worker.non_torch_memory == 200
            assert worker.peak_activation_memory == 100
    assert calls == ["profile"]


@pytest.mark.parametrize(
    "rc,free,dp,visible,exception",
    [
        (False, 1800, 1, 1, None),
        (True, 100, 1, 1, None),
        (False, 500, 1, 1, ValueError),
        (False, 1800, 2, 1, AssertionError),
        (False, 1800, 2, 2, None),
    ],
    ids=["normal-device", "rc-uses-host-free", "insufficient-memory", "too-few-devices", "valid-dp"],
)
def test_device_initialization_validates_budget_before_distributed(production, rc, free, dp, visible, exception):
    """T01-002：设备初始化先验证预算/可见卡数，再初始化分布式和seed。

    输入：total2000、显存比例0.8，free及DP/visible取参数表；RC host available1900。
    输出：通过时requested1600、设备npu:0，并严格按distributed→seed17执行；
      内存不足或DP卡数不足时拒绝，不能执行后续分布式初始化。
    依据：预算乘法、DP本机可见卡合同；检查真实worker字段及调用顺序。
    替身：设备访问、GC、分布式初始化和seed设置均为外部边界，无真实NPU声明。
    """
    worker = production.NPUWorker310.__new__(production.NPUWorker310)
    worker.local_rank = 0
    worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.8)
    worker.parallel_config = SimpleNamespace(
        data_parallel_size=dp,
        data_parallel_size_local=dp,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        local_world_size=dp,
    )
    worker.vllm_config = SimpleNamespace(parallel_config=worker.parallel_config)
    worker.model_config = SimpleNamespace(seed=17)
    calls = []
    worker._init_worker_distributed_environment = lambda: calls.append("distributed")
    snapshot = SimpleNamespace(total_memory=2000, free_memory=free)
    with (
        patch.object(production, "MemorySnapshot", return_value=snapshot),
        patch.object(production, "is_rc_device", return_value=rc),
        patch.object(production.psutil, "virtual_memory", return_value=SimpleNamespace(available=1900)),
        patch.object(production.torch.npu, "set_device"),
        patch.object(production.torch.npu, "empty_cache"),
        patch.object(production.torch.npu, "device_count", return_value=visible),
        patch.object(production.torch.npu, "is_available", return_value=True),
        patch.object(production.gc, "collect"),
        patch.object(production, "set_random_seed", side_effect=lambda seed: calls.append(seed)),
    ):
        if exception:
            with pytest.raises(exception):
                worker._init_device()
            assert calls == []
        else:
            device = worker._init_device()
            assert str(device) == "npu:0"
            assert calls == ["distributed", 17]
            assert worker.requested_memory == 1600
            assert worker.init_snapshot.free_memory == (1900 if rc else free)

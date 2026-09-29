# SPDX-License-Identifier: Apache-2.0
"""T20: opt-in probe validation and actual artifact corruption, not mocked loaders."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend._310p import dflash_fdo_numerical_probe as production


def identity():
    return production.ProbeTraceIdentity(
        mode="NONE",
        component="target",
        tp_rank=0,
        dataset_request=0,
        generated_prefix=(),
        speculative_iteration=0,
        draft_substep=None,
        descriptor=2,
        actual_tokens=2,
        active_rows=(0, 1),
        semantic_role="input_ids",
        shape=(2,),
        dtype="torch.int32",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", ""),
        ("component", ""),
        ("semantic_role", ""),
        ("tp_rank", -1),
        ("dataset_request", -1),
        ("speculative_iteration", -1),
        ("descriptor", -1),
        ("actual_tokens", -1),
        ("draft_substep", -1),
        ("shape", ()),
        ("shape", (-1,)),
        ("active_rows", (0, 0)),
        ("active_rows", (-1,)),
        ("active_rows", (2,)),
    ],
)
def test_probe_identity_rejects_ambiguous_or_out_of_range_coordinates(field, value):
    """T20-01：探针trace不能接受不可对齐或越界的身份字段。

    输入：合法2-token identity，仅修改参数表所列字段为非法值。
    输出：真实dataclass构造抛ArtifactError，原identity仍完整可序列化并往返相等。
    场景：空角色、负计数、非法shape、重复/越界active rows；无mock和文件操作。
    依据：这些坐标用于跨模式比较，非法坐标不能产生貌似有效的比较结果。
    """
    valid = identity()
    with pytest.raises(production.FdoNumericalProbeArtifactError):
        replace(valid, **{field: value})
    assert production.ProbeTraceIdentity.from_manifest_dict(valid.to_manifest_dict()) == valid


@pytest.mark.parametrize(
    "key,value",
    [
        (production.PROBE_MAX_ITERATIONS_ENV, "abc"),
        (production.PROBE_MAX_RECORDS_ENV, "1.5"),
        (production.PROBE_DATASET_REQUEST_ENV, "bad"),
        (production.PROBE_DATASET_REQUEST_ENV, "-1"),
    ],
)
def test_probe_bad_numeric_config_has_no_filesystem_side_effect(tmp_path, monkeypatch, key, value):
    """T20-02：启用探针时非法数字必须在创建输出目录前拒绝。

    输入：DFlash/FDO/310P配置及临时目录，iteration/records含非整数或request<0。
    输出：ConfigError；临时目录内没有trace文件或目录。
    替身：仅is_310p硬件身份，真实from_environ解析和验证；不模拟校验结果。
    """
    monkeypatch.setattr(production, "is_310p", lambda: True)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
    )
    with pytest.raises(production.FdoNumericalProbeConfigError):
        production.FdoNumericalProbeConfig.from_environ(
            config, environ={production.PROBE_DIR_ENV: str(tmp_path / "trace"), key: value}
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "damage", ["missing-field", "unsafe-name", "size", "hash", "missing-file", "incomplete", "json"]
)
def test_real_probe_artifact_corruption_is_never_loaded_as_valid(tmp_path, damage):
    """T20-03：真实tensor落盘→手动损坏→生产loader必须拒绝。

    输入：writer写入int32[11,22]，先校验正常往返，再按参数破坏manifest或tensor文件。
    输出：ArtifactError；不能把缺字段/目录穿越/size或SHA不符/丢失/未完成/坏JSON当有效trace。
    场景：失败或中断后的诊断产物不能误导精度比较；不mock open/torch.save/load/hash。
    依据：固定原tensor与实际磁盘文件，两阶段断言区分正常能力和故障拒绝能力。
    """
    config = production.FdoNumericalProbeConfig(enabled=True, output_dir=tmp_path, mode="NONE")
    writer = production.BoundedProbeWriter(config)
    original = torch.tensor([11, 22], dtype=torch.int32)
    writer.write_tensor(identity(), original)
    loaded = production.load_probe_records(tmp_path)
    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0].tensor, original, rtol=0, atol=0)
    manifest_path = tmp_path / "manifest.jsonl"
    record = json.loads(manifest_path.read_text())
    if damage == "missing-field":
        del record["artifact"]
    elif damage == "unsafe-name":
        record["artifact"] = "../outside.pt"
    elif damage == "size":
        record["artifact_bytes"] += 1
    elif damage == "hash":
        record["sha256"] = "0" * 64
    elif damage == "missing-file":
        (tmp_path / record["artifact"]).unlink()
    elif damage == "incomplete":
        record["complete"] = False
    manifest_path.write_text("not-json\n" if damage == "json" else json.dumps(record) + "\n")
    with pytest.raises(production.FdoNumericalProbeArtifactError):
        production.load_probe_records(tmp_path)

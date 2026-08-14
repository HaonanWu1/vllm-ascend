# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded diagnostics for 310P DFlash correctness isolation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from vllm.logger import logger

_PATH_ENV = "ASCEND_DFLASH_DIAGNOSTIC_PATH"
_LIMIT_ENV = "ASCEND_DFLASH_DIAGNOSTIC_LIMIT"
_MAX_ELEMENTS_ENV = "ASCEND_DFLASH_DIAGNOSTIC_MAX_ELEMENTS"
_DEFAULT_LIMIT = 4
_DEFAULT_MAX_ELEMENTS = 4096

_lock = threading.Lock()
_config: tuple[Path | None, int, int] | None = None
_stage_counts: dict[str, int] = {}


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default))), 0)
    except ValueError:
        return default


def _get_config() -> tuple[Path | None, int, int]:
    global _config
    if _config is None:
        raw_path = os.getenv(_PATH_ENV, "").strip()
        _config = (
            Path(raw_path) if raw_path else None,
            _positive_int_env(_LIMIT_ENV, _DEFAULT_LIMIT),
            _positive_int_env(_MAX_ELEMENTS_ENV, _DEFAULT_MAX_ELEMENTS),
        )
    return _config


def dflash_diagnostic_enabled() -> bool:
    path, limit, _ = _get_config()
    return path is not None and limit > 0


def _json_value(value: Any, max_elements: int) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach()
        shape = list(tensor.shape)
        result = {
            "dtype": str(tensor.dtype),
            "shape": shape,
        }
        if tensor.numel() <= max_elements:
            result["values"] = tensor.cpu().tolist()
        else:
            result["values"] = tensor.flatten()[:max_elements].cpu().tolist()
            result["truncated"] = True
            result["numel"] = tensor.numel()
        return result
    if isinstance(value, dict):
        return {str(key): _json_value(item, max_elements) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, max_elements) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def capture_dflash_diagnostic(
    stage: str,
    payload_builder: Callable[[], dict[str, Any] | None] | None = None,
    **payload: Any,
) -> None:
    """Append at most ``LIMIT`` records per stage when explicitly enabled.

    The disabled path returns before inspecting tensors, so normal execution
    performs no device synchronization or file I/O.
    """
    path, limit, max_elements = _get_config()
    if path is None or limit <= 0:
        return

    with _lock:
        occurrence = _stage_counts.get(stage, 0)
        if occurrence >= limit:
            return
        _stage_counts[stage] = occurrence + 1
        try:
            if payload_builder is not None:
                built_payload = payload_builder()
                if built_payload is None:
                    _stage_counts[stage] = occurrence
                    return
                payload = {**payload, **built_payload}
            record = {
                "stage": stage,
                "occurrence": occurrence,
                **{
                    key: _json_value(value, max_elements)
                    for key, value in payload.items()
                },
            }
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, encoded)
            finally:
                os.close(fd)
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("310P DFlash diagnostic capture failed: %s", exc)


def _reset_dflash_diagnostics_for_test() -> None:
    global _config
    with _lock:
        _config = None
        _stage_counts.clear()

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._310p.spec_decode.dflash_diagnostics_310 import (
    _reset_dflash_diagnostics_for_test,
    capture_dflash_diagnostic,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import (
    AscendSpecDecodeBaseProposer310,
)


def test_disabled_diagnostics_do_not_inspect_or_write(tmp_path: Path):
    output = tmp_path / "disabled.jsonl"
    builder = MagicMock(side_effect=AssertionError("payload builder must stay lazy"))
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        _reset_dflash_diagnostics_for_test()
        capture_dflash_diagnostic("draft_inputs", payload_builder=builder)
    _reset_dflash_diagnostics_for_test()

    builder.assert_not_called()
    assert not output.exists()


def test_enabled_diagnostics_are_bounded_per_stage(tmp_path: Path):
    output = tmp_path / "capture.jsonl"
    env = {
        "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output),
        "ASCEND_DFLASH_DIAGNOSTIC_LIMIT": "2",
    }
    with patch.dict(os.environ, env, clear=False):
        _reset_dflash_diagnostics_for_test()
        for value in range(3):
            capture_dflash_diagnostic(
                "draft_inputs",
                token_ids=torch.tensor([[value, value + 1]], dtype=torch.int32),
            )
        capture_dflash_diagnostic("verify", accepted_counts=torch.tensor([3]))
    _reset_dflash_diagnostics_for_test()

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in records] == [
        "draft_inputs",
        "draft_inputs",
        "verify",
    ]
    assert [record["occurrence"] for record in records] == [0, 1, 0]
    assert records[0]["token_ids"] == {
        "dtype": "torch.int32",
        "shape": [1, 2],
        "values": [[0, 1]],
    }


def test_payload_builder_failure_is_swallowed(tmp_path: Path):
    output = tmp_path / "failed.jsonl"
    env = {"ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output)}
    builder = MagicMock(side_effect=RuntimeError("diagnostic-only failure"))
    with (
        patch.dict(os.environ, env, clear=False),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_diagnostics_310.logger.warning_once"
        ) as warning,
    ):
        _reset_dflash_diagnostics_for_test()
        capture_dflash_diagnostic("verify", payload_builder=builder)
    _reset_dflash_diagnostics_for_test()

    builder.assert_called_once_with()
    warning.assert_called_once()
    assert not output.exists()


def test_empty_payload_does_not_consume_stage_quota(tmp_path: Path):
    output = tmp_path / "capture.jsonl"
    env = {
        "ASCEND_DFLASH_DIAGNOSTIC_PATH": str(output),
        "ASCEND_DFLASH_DIAGNOSTIC_LIMIT": "1",
    }
    with patch.dict(os.environ, env, clear=False):
        _reset_dflash_diagnostics_for_test()
        capture_dflash_diagnostic("state_metadata", payload_builder=lambda: None)
        capture_dflash_diagnostic("state_metadata", payload_builder=lambda: None)
        capture_dflash_diagnostic(
            "state_metadata",
            payload_builder=lambda: {"state_indices": torch.tensor([[2, 3]])},
        )
    _reset_dflash_diagnostics_for_test()

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["occurrence"] == 0
    assert records[0]["state_indices"]["values"] == [[2, 3]]


def test_draft_wrapper_captures_returned_tokens_only_for_dflash():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.method = "dflash"
    result = torch.tensor([[11, 12, 13]], dtype=torch.int32)
    token_indices = torch.tensor([3])

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft",
            return_value=result,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.dflash_diagnostic_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.capture_dflash_diagnostic"
        ) as capture,
    ):
        actual = proposer._run_merged_draft(
            4,
            1,
            token_indices,
            torch.arange(4),
            None,
            [SimpleNamespace()],
            4,
        )

    torch.testing.assert_close(actual, result)
    capture.assert_called_once()
    assert capture.call_args.args == ("draft_output",)
    payload = capture.call_args.kwargs["payload_builder"]()
    torch.testing.assert_close(payload["draft_token_ids"], result)
    torch.testing.assert_close(payload["token_indices_to_sample"], token_indices)


def test_draft_wrapper_does_not_capture_for_dspark():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.method = "dspark"
    result = torch.tensor([[11, 12, 13]], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft",
            return_value=result,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.dflash_diagnostic_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.capture_dflash_diagnostic"
        ) as capture,
    ):
        actual = proposer._run_merged_draft(
            4,
            1,
            torch.tensor([3]),
            torch.arange(4),
            None,
            [SimpleNamespace()],
            4,
        )

    torch.testing.assert_close(actual, result)
    capture.assert_not_called()

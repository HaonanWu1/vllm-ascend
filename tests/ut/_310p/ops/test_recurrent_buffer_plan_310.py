# SPDX-License-Identifier: Apache-2.0
"""Compile the actual device planning methods with minimal CPU stubs."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_short_sequence_buffer_plan(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-side planning check")
    repo = Path(__file__).resolve().parents[4]
    kernel_dir = repo / "csrc/attention/recurrent_gated_delta_rule_v310/op_kernel"
    header = (kernel_dir / "recurrent_gated_delta_rule_v310.h").read_text()
    methods = []
    for name in ("ComputeAvgload", "SelectShortSequenceBuffers"):
        start = header.index("    __aicore__ inline void " + name + "()")
        end = header.index("{", start) + 1
        depth = 1
        while depth:
            depth += (header[end] == "{") - (header[end] == "}")
            end += 1
        methods.append(header[start:end])
    template = Path(__file__).with_name("recurrent_buffer_plan_harness.cpp").read_text()
    marker = "// INSERT_ACTUAL_KERNEL_METHODS"
    assert template.count(marker) == 1
    source = template.replace(marker, "\n".join(methods))
    binary = tmp_path / "buffer_plan"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-sign-compare",
            "-x",
            "c++",
            "-",
            "-o",
            str(binary),
        ],
        input=source,
        text=True,
        check=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "37 CPU checks passed" in result.stdout

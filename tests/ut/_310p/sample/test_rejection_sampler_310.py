#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.outputs import SamplerOutput

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from tests.ut.base import TestBase
from vllm_ascend._310p.sample.rejection_sampler import (
    AscendRejectionSampler310,
    _force_pytorch_rejection_path,
)
from vllm_ascend._310p.spec_decode.dflash_diagnostics_310 import (
    _reset_dflash_diagnostics_for_test,
)


class TestForcePytorchRejectionPath(TestBase):
    def test_disables_triton_and_binds_recovered_then_restores(self):
        orig_triton = rejection_sampler_module.HAS_TRITON
        orig_recovered = rejection_sampler_module.sample_recovered_tokens

        def sentinel(*args, **kwargs):
            return None

        with _force_pytorch_rejection_path(sentinel):
            # 310P has no working Triton; the base sampler must take PyTorch paths.
            self.assertFalse(rejection_sampler_module.HAS_TRITON)
            self.assertIs(rejection_sampler_module.sample_recovered_tokens, sentinel)

        self.assertEqual(rejection_sampler_module.HAS_TRITON, orig_triton)
        self.assertIs(rejection_sampler_module.sample_recovered_tokens, orig_recovered)

    def test_dflash_diagnostics_capture_target_and_per_position_acceptance(self):
        sampler = object.__new__(AscendRejectionSampler310)
        sampler._capture_dflash_diagnostics = True
        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1, 2, 3]),
            draft_token_ids=torch.tensor([4, 3, 2, 6]),
            num_draft_tokens=[1, 3],
            max_spec_len=3,
        )
        logits = torch.zeros(4, 9)
        logits[0, 4] = 9
        logits[1, 3] = 9
        logits[2, 2] = 9
        logits[3, 6] = 9
        output = SamplerOutput(
            sampled_token_ids=torch.tensor(
                [[4, 7, -1, -1], [3, 2, 5, -1]]
            ),
            logprobs_tensors=None,
        )

        with (
            patch.object(rejection_sampler_module.AscendRejectionSampler, "forward", return_value=output),
            patch(
                "vllm_ascend._310p.sample.rejection_sampler.dflash_diagnostic_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.sample.rejection_sampler.capture_dflash_diagnostic"
            ) as capture,
        ):
            actual = sampler.forward(metadata, None, logits, SimpleNamespace())

        self.assertIs(actual, output)
        capture.assert_called_once()
        self.assertEqual(capture.call_args.args, ("verify",))
        payload = capture.call_args.kwargs["payload_builder"]()
        torch.testing.assert_close(
            payload["raw_target_argmax"], torch.tensor([4, 3, 2, 6])
        )
        torch.testing.assert_close(
            payload["accepted_draft_counts"], torch.tensor([1, 2])
        )
        torch.testing.assert_close(
            payload["per_position_accepted"],
            torch.tensor(
                [[True, False, False], [True, True, False]]
            ),
        )

    def test_non_dflash_sampler_does_not_capture(self):
        sampler = object.__new__(AscendRejectionSampler310)
        sampler._capture_dflash_diagnostics = False
        output = SamplerOutput(
            sampled_token_ids=torch.tensor([[4, -1]]),
            logprobs_tensors=None,
        )

        with (
            patch.object(
                rejection_sampler_module.AscendRejectionSampler,
                "forward",
                return_value=output,
            ),
            patch(
                "vllm_ascend._310p.sample.rejection_sampler.dflash_diagnostic_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.sample.rejection_sampler.capture_dflash_diagnostic"
            ) as capture,
        ):
            actual = sampler.forward(
                SimpleNamespace(), None, torch.empty(0), SimpleNamespace()
            )

        self.assertIs(actual, output)
        capture.assert_not_called()

    def test_diagnostic_failure_does_not_change_sampler_output(self):
        sampler = object.__new__(AscendRejectionSampler310)
        sampler._capture_dflash_diagnostics = True
        output = SamplerOutput(
            sampled_token_ids=torch.tensor([[4, -1]]),
            logprobs_tensors=None,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            env = {"VLLM_ASCEND_DFLASH_DIAGNOSTIC_PATH": str(path)}
            with (
                patch.dict(os.environ, env, clear=False),
                patch.object(
                    rejection_sampler_module.AscendRejectionSampler,
                    "forward",
                    return_value=output,
                ),
                patch(
                    "vllm_ascend._310p.sample.rejection_sampler._build_verify_diagnostic",
                    side_effect=RuntimeError("diagnostic-only failure"),
                ),
            ):
                _reset_dflash_diagnostics_for_test()
                actual = sampler.forward(
                    SimpleNamespace(), None, torch.empty(0), SimpleNamespace()
                )
            _reset_dflash_diagnostics_for_test()

            self.assertIs(actual, output)
            self.assertFalse(path.exists())

    def test_restores_on_exception(self):
        orig_triton = rejection_sampler_module.HAS_TRITON
        orig_recovered = rejection_sampler_module.sample_recovered_tokens

        with self.assertRaises(RuntimeError):
            with _force_pytorch_rejection_path(lambda *a, **k: None):
                raise RuntimeError("boom")

        self.assertEqual(rejection_sampler_module.HAS_TRITON, orig_triton)
        self.assertIs(rejection_sampler_module.sample_recovered_tokens, orig_recovered)

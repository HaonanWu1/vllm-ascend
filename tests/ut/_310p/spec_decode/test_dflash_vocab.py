# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend._310p.spec_decode import dflash_vocab as vocab


@pytest.mark.parametrize("has_head", [False, True])
def test_actual_weight_stream_determines_head_presence(monkeypatch, has_head):
    items = [("d2t", torch.tensor([1, 3, 4])), ("layers.0.weight", torch.ones(2))]
    if has_head:
        items.append(("lm_head.weight", torch.ones(3, 4)))
    received = []

    def load(model, weights):
        received.extend(weights)
        return "loaded"

    monkeypatch.setattr(vocab, "_original_load_weights", load)
    model = SimpleNamespace()
    assert vocab.load_dflash_weights_310(model, iter(items)) == "loaded"
    assert model._dflash_checkpoint_has_lm_head is has_head
    assert [name for name, _ in received] == [name for name, _ in items]
    assert all(a is b for (_, a), (_, b) in zip(received, items))


def make_models(has_head=False, mapping=True, method="dflash"):
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    output = torch.full((4, 4), float("nan"))

    def loader(parameter, selected):
        parameter.zero_()
        parameter[: len(selected)].copy_(selected)

    draft = SimpleNamespace(
        draft_id_to_target_id=torch.tensor([1, 3, 4]) if mapping else None,
        _dflash_checkpoint_has_lm_head=has_head,
        lm_head=SimpleNamespace(weight=output, weight_loader=Mock(side_effect=loader)),
    )
    draft.lm_head.weight_nz = output.clone()
    draft.lm_head.quant_method = SimpleNamespace(
        process_weights_after_loading=Mock(side_effect=lambda head: setattr(head, "weight_nz", head.weight.clone()))
    )
    target = SimpleNamespace(lm_head=SimpleNamespace(weight=weight))
    return SimpleNamespace(model=draft, method=method), target


@pytest.mark.parametrize("multimodal", [False, True])
def test_missing_head_matches_target_subset_logits(monkeypatch, multimodal):
    proposer, target = make_models()
    source = target.lm_head.weight.clone()
    wrapper = SimpleNamespace(get_language_model=lambda: target) if multimodal else target
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 1)
    original = Mock(return_value="wrapped")
    monkeypatch.setattr(vocab, "_original_maybe_share_lm_head", original)
    assert vocab.maybe_share_dflash_lm_head_310(proposer, wrapper) == "wrapped"
    hidden = torch.tensor([[0.2, -0.4, 1.0, 2.0]])
    # Mapping stores offsets. Absolute target IDs are [1, 4, 6], not [1, 3, 4].
    expected = F.linear(hidden, source)[:, [1, 4, 6]]
    actual = F.linear(hidden, proposer.model.lm_head.weight_nz[:3])
    torch.testing.assert_close(actual, expected)
    assert torch.all(proposer.model.lm_head.weight[3] == 0)
    torch.testing.assert_close(target.lm_head.weight, source)
    original.assert_called_once_with(proposer, wrapper)


@pytest.mark.parametrize(
    "has_head,mapping,method",
    [(True, True, "dflash"), (False, False, "dflash"), (False, True, "dspark"), (False, True, "eagle")],
)
def test_existing_and_adjacent_paths_keep_original_policy(monkeypatch, has_head, mapping, method):
    proposer, target = make_models(has_head, mapping, method)
    original = Mock()
    monkeypatch.setattr(vocab, "_original_maybe_share_lm_head", original)
    vocab.maybe_share_dflash_lm_head_310(proposer, target)
    proposer.model.lm_head.weight_loader.assert_not_called()
    proposer.model.lm_head.quant_method.process_weights_after_loading.assert_not_called()
    original.assert_called_once_with(proposer, target)


def test_tp_gathers_target_before_draft_loader_shards(monkeypatch):
    proposer, target = make_models()
    full = target.lm_head.weight.clone()
    target.lm_head.weight = full[:4]
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 2)
    gather = Mock(return_value=full)
    monkeypatch.setattr(vocab, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(vocab, "_original_maybe_share_lm_head", Mock())
    vocab.maybe_share_dflash_lm_head_310(proposer, target)
    gather.assert_called_once_with(target.lm_head.weight, dim=0)
    torch.testing.assert_close(proposer.model.lm_head.weight_loader.call_args.args[1], full[[1, 4, 6]])


@pytest.mark.parametrize("failure", ["missing", "quantized", "added"])
def test_unsupported_target_head_fails_before_uninitialized_logits(monkeypatch, failure):
    proposer, target = make_models()
    if failure == "missing":
        del target.lm_head
    elif failure == "quantized":
        target.lm_head.weight = target.lm_head.weight.to(torch.int8)
    else:
        target.lm_head.num_added_embeddings = 4
    monkeypatch.setattr(vocab, "_original_maybe_share_lm_head", Mock())
    with pytest.raises(ValueError):
        vocab.maybe_share_dflash_lm_head_310(proposer, target)
    proposer.model.lm_head.weight_loader.assert_not_called()

# SPDX-License-Identifier: Apache-2.0
"""T04: real page-size calculation; only upstream setup/model lookup are boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.patch.platform import patch_mamba_config_310 as production


def _config(
    *,
    block=None,
    mode="align",
    state_elements=65536,
    draft_heads=None,
    method="dflash",
    tp=1,
    draft_tp=None,
    dtype="auto",
    user_mamba_block=None,
):
    def model(heads):
        return SimpleNamespace(
            dtype=torch.float16,
            use_mla=False,
            architecture="UnitHybridModel",
            get_num_kv_heads=lambda parallel: heads // parallel.tensor_parallel_size,
            get_head_size=lambda: 128,
            get_mamba_chunk_size=lambda: 64,
        )

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block,
            mamba_block_size=user_mamba_block,
            cache_dtype=dtype,
            mamba_cache_mode=mode,
            mamba_page_size_padded=None,
        ),
        model_config=model(4),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=None,
    )
    if draft_heads is not None:
        cfg.speculative_config = SimpleNamespace(
            method=method,
            draft_model_config=model(draft_heads),
            draft_parallel_config=(SimpleNamespace(tensor_parallel_size=draft_tp) if draft_tp is not None else None),
        )
    state_model = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _: ((state_elements,),),
        get_mamba_state_dtype_from_config=lambda _: (torch.float16,),
    )
    return cfg, state_model


def _verify(cfg, model):
    # Upstream default configuration and model loading are separate components;
    # FullAttentionSpec/MambaSpec and the entire production calculation stay real.
    with (
        patch.object(production.MambaModelConfig, "verify_and_update_config"),
        patch.object(production.ModelRegistry, "resolve_model_cls", return_value=(model, None)),
    ):
        production.verify_and_update_config.__func__(object, cfg)


@pytest.mark.parametrize(
    "tp,block,padded", [(1, 128, 262144), (2, 128, None), (4, 256, None)], ids=["tp1-pad", "tp2-exact", "tp4-grow"]
)
def test_target_page_size_matches_real_specs(tp, block, padded):
    """T04-001：TP变化时统一页满足真实Mamba状态大小。

    输入：4个KV heads、head_size128、FP16、65536个FP16状态元素，TP=1/2/4。
    场景：每token页分别2048/1024/512字节，需要128对齐的attention block。
    输出：block=128/128/256；仅TP1填充至262144字节，其余恰好131072无需填充。
    依据：K和V两个矩阵的字节数与独立手算；真实生产函数和Spec执行。
    替身：仅上游默认设置及模型注册查找；不代表真实模型加载/NPU验证。
    回归：错误TP head数、漏乘K/V或向下取整都会失败。
    """
    cfg, model = _config(tp=tp)
    _verify(cfg, model)
    assert cfg.cache_config.block_size == block
    assert cfg.cache_config.mamba_block_size == block
    assert cfg.cache_config.mamba_page_size_padded == padded


@pytest.mark.parametrize(
    "options,block,padded",
    [
        ({"block": 512, "draft_heads": 8}, 512, 1048576),
        ({"block": 128, "draft_heads": 8}, 128, 524288),
        ({"block": 512, "draft_heads": 8, "method": "eagle"}, 512, 2097152),
        ({"block": 128, "draft_heads": 12}, 128, 786432),
        ({"block": 256, "draft_heads": 8, "draft_tp": 2}, 256, 524288),
        ({"block": 256, "draft_heads": 2}, 256, 524288),
        ({"block": 32}, 128, 262144),
        ({"block": 256, "dtype": "float32"}, 256, 1048576),
    ],
    ids=[
        "dflash-resize-draft",
        "unaligned-draft-fallback",
        "eagle-no-resize",
        "nondivisible-draft",
        "draft-own-tp",
        "smaller-draft",
        "small-user-block",
        "fp32-cache",
    ],
)
def test_draft_page_resize_and_fallback(options, block, padded):
    """T04-002：Target/Draft不同KV布局下的页复用与回退。

    输入：参数表列出block、Draft heads/TP、方法及cache dtype；状态为131072字节。
    场景：只有DFlash且Draft逻辑block为正的128倍数时可复用Target页。
    输出：逐项断言参数表中手算的block及padding字节；重复验证不得改变结果。
    依据：Target/Draft的K/V字节数，未从被测输出反推期望。
    替身：同_verify边界；真实Spec与生产分支运行，无设备数值声明。
    回归：错误复用EAGLE、不整除或未对齐Draft页、忽略Draft TP都会失败。
    """
    cfg, model = _config(**options)
    _verify(cfg, model)
    assert cfg.cache_config.block_size == block
    assert cfg.cache_config.mamba_page_size_padded == padded
    _verify(cfg, model)
    assert cfg.cache_config.mamba_page_size_padded == padded


@pytest.mark.parametrize("chunk,expected", [(None, 128), (192, 384), (256, 256)])
def test_all_cache_uses_lcm_of_chunk_and_kernel_alignment(chunk, expected):
    """T04-003：all模式同时满足用户chunk和128 kernel边界。

    输入：chunk默认64/用户192/256，Mamba状态131072字节，Target每token2048字节。
    输出：attention与mamba block分别128/384/256，padding=block*2048。
    依据：独立最小公倍数手算；尤其192不能简单向上取128倍数为256。
    替身：上游设置和模型查找；无被测数学逻辑替换。
    """
    cfg, model = _config(mode="all", user_mamba_block=chunk)
    _verify(cfg, model)
    assert cfg.cache_config.block_size == expected
    assert cfg.cache_config.mamba_block_size == expected
    assert cfg.cache_config.mamba_page_size_padded == expected * 2048


def test_disabled_mamba_keeps_cache_and_mla_rejects():
    """T04-004：零Mamba状态早退与不支持的MLA拒绝。

    输入：状态0、已有block256；另一次将use_mla设True。
    输出：早退保持block256且不填充；MLA抛含MLA的RuntimeError。
    场景：不能给禁用Mamba的模型改缓存，也不能悄悄接受不支持布局。
    替身：仅模型查找/默认设置；两个分支调用真实生产函数。
    """
    cfg, model = _config(block=256, state_elements=0)
    _verify(cfg, model)
    assert cfg.cache_config.block_size == 256
    assert cfg.cache_config.mamba_page_size_padded is None
    cfg.model_config.use_mla = True
    with pytest.raises(RuntimeError, match="MLA"):
        _verify(cfg, model)

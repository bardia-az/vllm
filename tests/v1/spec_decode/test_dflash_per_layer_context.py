# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dnest per-layer target context (context_mode="per_layer") serving tests.

Covers the config plumbing (algos.update_dnest passes context_mode/kv_mode
into dflash_config) and the per-layer context K/V projection math: with the
RMSNorm kernel swapped for an eager reference, the stacked bmm path must
reproduce each layer's own input_layernorm -> k/v_proj -> k_norm computation
-- the exact computation the target model ran for that layer's keys/values.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_dflash
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.transformers_utils.configs.speculators import algos

HIDDEN = 64
NUM_LAYERS = 3
NUM_KV_HEADS = 2
HEAD_DIM = 32
NUM_CTX = 12


def _eager_grouped_rms_norm(out, inp, weight, eps):
    """Eager stand-in for the grouped vLLM RMSNorm kernel: a 1-D weight is
    shared; a stacked [L, D] weight is selected per layer (outermost index).
    """
    x = inp.float()
    var = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    if weight.dim() == 1:
        w = weight.float()
    else:
        shape = [weight.shape[0]] + [1] * (inp.dim() - 2) + [weight.shape[-1]]
        w = weight.float().view(shape)
    out.copy_((normed * w).to(inp.dtype))


@pytest.fixture(autouse=True)
def eager_rms_norm(monkeypatch):
    monkeypatch.setattr(qwen3_dflash.ops, "rms_norm", _eager_grouped_rms_norm)
    # No distributed init in this unit test: single-rank TP.
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)


def _make_attn(kv_mode, seed):
    g = torch.Generator().manual_seed(seed)
    attn = SimpleNamespace(
        total_num_kv_heads=NUM_KV_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_size=4 * HEAD_DIM,
        qkv_proj=SimpleNamespace(
            # q = 4 heads x HEAD_DIM, k = v = NUM_KV_HEADS x HEAD_DIM
            weight=nn.Parameter(torch.randn(8 * HEAD_DIM, HIDDEN, generator=g))
        ),
        k_norm=SimpleNamespace(
            weight=nn.Parameter(torch.rand(HEAD_DIM, generator=g) + 0.5)
        ),
        q_norm=SimpleNamespace(variance_epsilon=1e-6),
    )
    layer = SimpleNamespace(
        self_attn=attn,
        input_layernorm=SimpleNamespace(
            weight=nn.Parameter(torch.rand(HIDDEN, generator=g) + 0.5)
        ),
    )
    side_weights = {}
    if kv_mode == "split":
        side_weights = {
            "input_layernorm": torch.rand(HIDDEN, generator=g) + 0.5,
            "k_proj": torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN, generator=g),
            "v_proj": torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN, generator=g),
            "k_norm": torch.rand(HEAD_DIM, generator=g) + 0.5,
        }
    return layer, attn, side_weights


def _make_model(kv_mode, seed=7):
    layers, attns, side = [], [], {}
    for j in range(NUM_LAYERS):
        layer, attn, side_weights = _make_attn(kv_mode, seed + j)
        if side_weights:
            side[j] = side_weights
        layers.append(layer)
        attns.append(attn)
    model = SimpleNamespace(
        context_mode="per_layer",
        kv_mode=kv_mode,
        config=SimpleNamespace(hidden_size=HIDDEN),
        layers=layers,
        _context_kv_checkpoint_weights=side,
        _rms_norm_eps=1e-6,
    )
    for method_name in (
        "_build_context_kv_buffers",
        "_build_per_layer_context_kv_buffers",
        "_project_context_kv",
        "_project_per_layer_context_kv",
        "_normalize_context_k",
    ):
        setattr(
            model,
            method_name,
            MethodType(getattr(DFlashQwen3Model, method_name), model),
        )
    model._build_context_kv_buffers(attns, has_bias=False)
    return model, layers, attns, side


def _reference_kv(kv_mode, taps, layers, side):
    """Eager per-layer reference: the target layer's own K/V computation."""
    all_k, all_v = [], []
    eps = 1e-6
    for j, layer in enumerate(layers):
        h = taps[j]
        if kv_mode == "split":
            norm_w = side[j]["input_layernorm"]
            k_proj, v_proj, k_norm_w = (
                side[j]["k_proj"],
                side[j]["v_proj"],
                side[j]["k_norm"],
            )
        else:
            norm_w = layer.input_layernorm.weight.data
            kv = layer.self_attn.qkv_proj.weight.data[4 * HEAD_DIM :]
            k_proj, v_proj = kv[: NUM_KV_HEADS * HEAD_DIM], kv[NUM_KV_HEADS * HEAD_DIM :]
            k_norm_w = layer.self_attn.k_norm.weight.data
        normed = h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + eps) * norm_w
        k = torch.nn.functional.linear(normed, k_proj)
        k = k.view(NUM_CTX, NUM_KV_HEADS, HEAD_DIM)
        k = k * torch.rsqrt(
            k.float().pow(2).mean(-1, keepdim=True) + eps
        ) * k_norm_w
        v = torch.nn.functional.linear(normed, v_proj).view(
            NUM_CTX, NUM_KV_HEADS, HEAD_DIM
        )
        all_k.append(k)
        all_v.append(v)
    return torch.stack(all_k), torch.stack(all_v)


@pytest.mark.parametrize("kv_mode", ["frozen_shared", "split", "trainable_shared"])
def test_per_layer_projection_matches_eager(kv_mode):
    """The stacked (grouped-norm + bmm) path reproduces the per-layer eager
    computation exactly, for every kv_mode's weight selection."""
    model, layers, _, _ = _make_model(kv_mode)
    g = torch.Generator().manual_seed(11)
    taps = torch.randn(NUM_LAYERS, NUM_CTX, HIDDEN, generator=g)
    context_states = taps.permute(1, 0, 2).reshape(NUM_CTX, NUM_LAYERS * HIDDEN)

    all_k, all_v = model._project_context_kv(
        context_states, NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
    )
    all_k = model._normalize_context_k(all_k)

    ref_k, ref_v = _reference_kv(kv_mode, taps, layers, model._context_kv_checkpoint_weights)
    torch.testing.assert_close(all_k, ref_k, rtol=0, atol=1e-5)
    torch.testing.assert_close(all_v, ref_v, rtol=0, atol=1e-5)


def test_per_layer_projection_rejects_wrong_width():
    model, _, _, _ = _make_model("frozen_shared")
    with pytest.raises(ValueError, match="per_layer context expects"):
        model._project_context_kv(
            torch.randn(NUM_CTX, HIDDEN), NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
        )


def test_split_requires_side_loaded_context_kv():
    """kv_mode=split without the frozen context_kv tensors fails loudly."""
    model, _, _, _ = _make_model("split")
    model._context_kv_checkpoint_weights = {}
    with pytest.raises(ValueError, match="context key/value tensors"):
        model._build_context_kv_buffers(
            [layer.self_attn for layer in model.layers], has_bias=False
        )


def test_update_dnest_passes_per_layer_fields():
    """algos: a per_layer dnest config carries context_mode/kv_mode into
    dflash_config; a fused one does not."""
    config_dict = {
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "mask_token_id": 151669,
        "context_mode": "per_layer",
        "kv_mode": "split",
    }
    pre_trained_config = {}
    algos.update_dnest(config_dict, pre_trained_config)
    dflash_config = pre_trained_config["dflash_config"]
    assert dflash_config["context_mode"] == "per_layer"
    assert dflash_config["kv_mode"] == "split"

    fused = dict(config_dict, context_mode="fused")
    pre_trained_config = {}
    algos.update_dnest(fused, pre_trained_config)
    assert "context_mode" not in pre_trained_config["dflash_config"]
    assert "kv_mode" not in pre_trained_config["dflash_config"]

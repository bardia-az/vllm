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
    if kv_mode in ("split", "split_trainable"):
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
        if kv_mode in ("split", "split_trainable"):
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


@pytest.mark.parametrize(
    "kv_mode", ["frozen_shared", "split", "split_trainable", "trainable_shared"]
)
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


@pytest.mark.parametrize("kv_mode", ["split", "split_trainable"])
def test_split_requires_side_loaded_context_kv(kv_mode):
    """The side-loading kv_modes without their context_kv tensors fail
    loudly."""
    model, _, _, _ = _make_model(kv_mode)
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
        "kv_mode": "split_trainable",
    }
    pre_trained_config = {}
    algos.update_dnest(config_dict, pre_trained_config)
    dflash_config = pre_trained_config["dflash_config"]
    assert dflash_config["context_mode"] == "per_layer"
    assert dflash_config["kv_mode"] == "split_trainable"

    fused = dict(config_dict, context_mode="fused")
    pre_trained_config = {}
    algos.update_dnest(fused, pre_trained_config)
    assert "context_mode" not in pre_trained_config["dflash_config"]
    assert "kv_mode" not in pre_trained_config["dflash_config"]


# --- ground truth: the serving projection vs a real Qwen3's KV cache ------
#
# The parity test above checks the stacked path against an eager reference
# built from the same stub tensors. This section anchors the whole serving
# convention against ground truth: real Qwen3 layer tensors wired into the
# serving buffers (exactly what load_weights + buffer building produce from
# a per_layer checkpoint) must reproduce the target model's OWN HF KV cache
# at the source layers -- V exactly, K through k_norm then RoPE, with the
# RoPE applied from the target's own cos/sin and the vLLM cos_sin_cache
# layout independently asserted equal to HF's.


def _make_target_and_cache():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    torch.manual_seed(0)
    target = Qwen3ForCausalLM(config).eval()
    torch.manual_seed(5)
    ids = torch.randint(0, 256, (1, NUM_CTX))
    with torch.no_grad():
        out = target(ids, use_cache=True, output_hidden_states=True)
    return target, out


def _make_model_from_target(kv_mode, target, sources):
    """Serving-side stubs wired to the target's real tensors, mirroring what
    load_weights + _build_per_layer_context_kv_buffers see after loading a
    per_layer checkpoint trained from this target (shared modes: the layer's
    own tensors; split: frozen copies of the source layer's exact path)."""
    layers, attns, side = [], [], {}
    for source in sources:
        tl = target.model.layers[source]
        # QKVParallelLinear packs [q; k; v] along dim 0 -- the layout the
        # serving path slices with q_size.
        qkv = torch.cat(
            [
                tl.self_attn.q_proj.weight.data,
                tl.self_attn.k_proj.weight.data,
                tl.self_attn.v_proj.weight.data,
            ],
            dim=0,
        )
        attn = SimpleNamespace(
            total_num_kv_heads=NUM_KV_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            q_size=4 * HEAD_DIM,
            qkv_proj=SimpleNamespace(weight=nn.Parameter(qkv.clone())),
            k_norm=SimpleNamespace(
                weight=nn.Parameter(tl.self_attn.k_norm.weight.data.clone())
            ),
            q_norm=SimpleNamespace(variance_epsilon=1e-6),
        )
        layer = SimpleNamespace(
            self_attn=attn,
            input_layernorm=SimpleNamespace(
                weight=nn.Parameter(tl.input_layernorm.weight.data.clone())
            ),
        )
        if kv_mode in ("split", "split_trainable"):
            side[len(layers)] = {
                "input_layernorm": tl.input_layernorm.weight.data.clone(),
                "k_proj": tl.self_attn.k_proj.weight.data.clone(),
                "v_proj": tl.self_attn.v_proj.weight.data.clone(),
                "k_norm": tl.self_attn.k_norm.weight.data.clone(),
            }
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
    return model


@pytest.mark.parametrize(
    "kv_mode", ["frozen_shared", "split", "split_trainable"]
)
def test_per_layer_projection_matches_target_kv_cache(kv_mode):
    """The serving projection reproduces a real Qwen3's own KV cache."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    target, out = _make_target_and_cache()
    # Arbitrary distinct source layers; tap s is the input of layer s.
    sources = [1, 3, 2]
    model = _make_model_from_target(kv_mode, target, sources)

    taps = torch.stack([out.hidden_states[s][0] for s in sources], dim=0)
    context_states = taps.permute(1, 0, 2).reshape(NUM_CTX, len(sources) * HIDDEN)

    all_k, all_v = model._project_context_kv(
        context_states, NUM_CTX, len(sources), NUM_KV_HEADS, HEAD_DIM
    )
    all_k = model._normalize_context_k(all_k)  # [L, nctx, nkv, hd], pre-RoPE

    # The vLLM cos_sin_cache layout the real RoPE kernel reads must equal
    # HF's cos/sin at every context position (neox: [cos | sin] halves).
    # Built with vLLM's own RotaryEmbedding (the production cache code).
    # It subclasses CustomOp, whose init dispatch reads the compilation
    # config -- mock it the way test_dflash2.py does (noconftest runs
    # cannot use the default_vllm_config fixture from tests/conftest.py).
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    mock_vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        )
    )
    with set_current_vllm_config(mock_vllm_config):
        rope = RotaryEmbedding(
            head_size=HEAD_DIM,
            rotary_dim=HEAD_DIM,
            max_position_embeddings=NUM_CTX,
            base=1000000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )
    positions = torch.arange(NUM_CTX).unsqueeze(0)
    cos_hf, sin_hf = target.model.rotary_emb(out.hidden_states[0], positions)
    half = HEAD_DIM // 2
    torch.testing.assert_close(
        rope.cos_sin_cache[:NUM_CTX, :half], cos_hf[0, :, :half], rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        rope.cos_sin_cache[:NUM_CTX, half:], sin_hf[0, :, :half], rtol=0, atol=1e-5
    )

    for j, source in enumerate(sources):
        k = all_k[j].transpose(0, 1).unsqueeze(0)  # [1, nkv, nctx, hd]
        _, k_rope = apply_rotary_pos_emb(torch.zeros_like(k), k, cos_hf, sin_hf)
        torch.testing.assert_close(
            k_rope, out.past_key_values.layers[source].keys, rtol=0, atol=1e-5
        )
        torch.testing.assert_close(
            all_v[j].transpose(0, 1).unsqueeze(0),
            out.past_key_values.layers[source].values,
            rtol=0,
            atol=1e-5,
        )

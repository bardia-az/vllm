# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``maybe_override_with_speculators`` config merging.

When a model is in speculators format, the function previously dropped any
user-provided ``--speculative-config`` overrides (e.g. ``attention_backend``
added in https://github.com/vllm-project/vllm/pull/39930), silently replacing
the dict with a freshly-extracted one. These tests pin the merge behavior so
runtime fields from the user are preserved while ``model`` stays auto-detected
from the speculators format.
"""

from unittest.mock import patch

from vllm.transformers_utils.config import maybe_override_with_speculators

_SPECULATORS_CONFIG_DICT = {
    "architectures": ["DFlashDraftModel"],
    "speculators_config": {
        "algorithm": "dflash",
        "default_proposal_method": "greedy",
        "proposal_methods": [
            {
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 8,
                "verifier_accept_k": 1,
            }
        ],
        "verifier": {
            "architectures": [],
            "name_or_path": "fake-org/fake-verifier",
        },
    },
    "speculators_model_type": "dflash",
    "transformer_layer_config": {},
}


def _patched_get_config_dict(config_dict):
    """Return a patch context that bypasses HF Hub for get_config_dict."""
    return patch(
        "vllm.transformers_utils.config.PretrainedConfig.get_config_dict",
        return_value=(config_dict, {}),
    )


def test_speculators_extracts_baseline_config():
    """Sanity: extracted dict has method/num_speculative_tokens/model fields."""
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        model, tok, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=None,
        )

    assert model == "fake-org/fake-verifier"
    assert tok == "fake-org/fake-verifier"
    assert spec is not None
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 8
    assert spec["model"] == "fake-org/fake-dflash-spec"


def test_speculators_preserves_user_attention_backend_override():
    """User-supplied attention_backend must survive the speculators path.

    Regression for the case where vllm_speculative_config was silently
    discarded for speculators-format models.
    """
    user_overrides = {"attention_backend": "FLASH_ATTN"}
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["attention_backend"] == "FLASH_ATTN"
    # Auto-detected fields still present
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 8
    assert spec["model"] == "fake-org/fake-dflash-spec"


def test_speculators_user_cannot_override_model_field():
    """``model`` is dictated by the speculators format and must stay locked.

    Even if a user passes a different ``model`` in their override dict, the
    final config must use the speculators model path (so the drafter actually
    loads from the right weights).
    """
    user_overrides = {"model": "fake-org/wrong-model"}
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["model"] == "fake-org/fake-dflash-spec"


def test_speculators_user_cannot_override_method_field():
    user_overrides = {"method": "ngram"}
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["method"] == "dflash"


def test_speculators_user_can_override_num_speculative_tokens():
    """Runtime knobs (num_speculative_tokens, moe_backend, etc.) take user value."""
    user_overrides = {
        "num_speculative_tokens": 4,
        "moe_backend": "fake-backend",
    }
    with _patched_get_config_dict(_SPECULATORS_CONFIG_DICT):
        _, _, spec = maybe_override_with_speculators(
            model="fake-org/fake-dflash-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert spec is not None
    assert spec["num_speculative_tokens"] == 4
    assert spec["moe_backend"] == "fake-backend"


def test_non_speculators_model_passes_user_config_through_unchanged():
    """When the model is NOT a speculators model, user config is returned as-is."""
    plain_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096}
    user_overrides = {"method": "ngram", "num_speculative_tokens": 3}
    with _patched_get_config_dict(plain_config):
        model, tok, spec = maybe_override_with_speculators(
            model="fake-org/plain-llama",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=user_overrides,
        )

    assert model == "fake-org/plain-llama"
    assert tok is None
    assert spec == user_overrides


_DNEST_CONFIG_DICT = {
    "architectures": ["DNestDraftModel"],
    "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
    "block_size": 16,
    "draft_vocab_size": 151936,
    "init_mode": "target_after_extracted",
    "init_source_layers": [2, 10, 18, 26, 34],
    "fusion_init": "random",
    "copied_lr_scale": 1.0,
    "init_final_norm": False,
    "mask_token_id": 151669,
    "sample_from_anchor": False,
    "sliding_window_non_causal": False,
    "speculators_config": _SPECULATORS_CONFIG_DICT["speculators_config"],
    "speculators_model_type": "dnest",
    "transformer_layer_config": {},
}


def test_speculators_dnest_maps_to_dflash_method():
    """A dnest checkpoint (DFlash + target-derived init) serves via the
    DFlash runtime: method rewritten to dflash, dflash_config derived, and
    the architecture forced to the DFlash model loader."""
    with _patched_get_config_dict(_DNEST_CONFIG_DICT):
        model, tok, spec = maybe_override_with_speculators(
            model="fake-org/fake-dnest-spec",
            tokenizer=None,
            trust_remote_code=False,
            vllm_speculative_config=None,
        )

    assert spec is not None
    assert spec["method"] == "dflash"
    assert spec["model"] == "fake-org/fake-dnest-spec"
    assert model == "fake-org/fake-verifier"

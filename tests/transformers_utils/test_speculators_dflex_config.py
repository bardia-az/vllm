# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_dflex_config(sample_from_anchor: bool = True) -> dict:
    """A config dict shaped like a speculators dflex checkpoint's config.json."""
    return {
        "speculators_model_type": "dflex",
        "architectures": ["DFlexSpeculator"],
        "sample_from_anchor": sample_from_anchor,
        "draft_vocab_size": 32,
        "block_size": 15,
        "mask_token_id": 31,
        "aux_hidden_state_layer_ids": [2, 5],
        "sliding_window_non_causal": False,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
        "speculators_config": {
            "algorithm": "dflex",
            "proposal_methods": [
                {"type": "greedy_token_proposal", "speculative_tokens": 15}
            ],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": "Qwen/Qwen3-4B"},
        },
    }


def test_dflex_updater_maps_dflash_runtime_fields() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dflex_config()
    )

    # dflex reuses the DFlash model loader and runtime unchanged
    assert config["architectures"] == ["DFlashDraftModel"]
    assert config["dflash_config"]["sample_from_anchor"] is True
    assert config["dflash_config"]["block_size"] == 15
    assert config["dflash_config"]["mask_token_id"] == 31
    # DFlash layer-indexing convention (aux_id - 1)
    assert config["dflash_config"]["target_layer_ids"] == [1, 4]
    assert config["dflash_config"]["causal"] is True


def test_dflex_updater_rejects_bonus_anchor_checkpoint() -> None:
    with pytest.raises(ValueError, match="sample_from_anchor=True"):
        SpeculatorsConfig.extract_transformers_pre_trained_config(
            _make_dflex_config(sample_from_anchor=False)
        )


def test_dflex_builds_dedicated_vllm_method() -> None:
    spec = SpeculatorsConfig.extract_vllm_speculative_config(_make_dflex_config())

    assert spec["method"] == "dflex"
    assert spec["num_speculative_tokens"] == 15


def test_dflash_updater_unchanged_by_dflex_support() -> None:
    outer = _make_dflex_config()
    outer["speculators_model_type"] = "dflash"
    outer["sample_from_anchor"] = False
    outer["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 14

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer)
    spec = SpeculatorsConfig.extract_vllm_speculative_config(outer)

    assert config["dflash_config"]["sample_from_anchor"] is False
    assert "block_size" not in config["dflash_config"]
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 14

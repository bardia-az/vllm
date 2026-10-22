# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup

TARGET = "Qwen/Qwen3-4B"


def _write_speculators_checkpoint(tmp_path, speculators_model_type: str):
    """Write a minimal speculators-format config.json like the speculators
    trainer saves, with the given model type, and return its path."""
    sample_from_anchor = speculators_model_type == "dflex"
    speculative_tokens = 15 if sample_from_anchor else 14
    config = {
        "speculators_model_type": speculators_model_type,
        "architectures": [f"{speculators_model_type.capitalize()}Speculator"],
        "sample_from_anchor": sample_from_anchor,
        "draft_vocab_size": 32,
        "block_size": 15,
        "mask_token_id": 31,
        "aux_hidden_state_layer_ids": [2, 5],
        "sliding_window_non_causal": False,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 2,
        },
        "speculators_config": {
            "algorithm": speculators_model_type,
            "proposal_methods": [
                {
                    "type": "greedy_token_proposal",
                    "speculative_tokens": speculative_tokens,
                }
            ],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": "Qwen/Qwen3-4B"},
        },
    }
    draft_dir = tmp_path / f"draft_{speculators_model_type}"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(config))
    return draft_dir


def _vllm_config(tmp_path, method: str) -> VllmConfig:
    draft_dir = _write_speculators_checkpoint(tmp_path, method)
    draft_hf_config = SpeculatorsConfig.from_pretrained(draft_dir)
    assert draft_hf_config.architectures == ["DFlashDraftModel"]
    draft_model_config = MagicMock(
        model=str(draft_dir),
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    model_config = ModelConfig(TARGET, max_model_len=128)
    parallel_config = ParallelConfig(distributed_executor_backend="uni")
    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        speculative_config = SpeculativeConfig(
            method=method,
            model=str(draft_dir),
            num_speculative_tokens=15,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )
    return VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
        ),
        parallel_config=parallel_config,
        speculative_config=speculative_config,
    )


def test_dflex_scheduling_uses_anchor_layout(tmp_path) -> None:
    """The dflex method samples from the anchor like DSpark: the anchor query
    is the first prediction, so lookahead is num_speculative_tokens (not the
    +1 in-fill margin DFlash's bonus-anchor layout needs)."""
    cfg = _vllm_config(tmp_path, "dflex")
    spec = cfg.speculative_config

    assert spec.method == "dflex"
    assert spec.use_dflex()
    assert spec.use_eagle()
    assert not spec.use_dflash()
    assert spec.parallel_drafting
    assert cfg.num_lookahead_tokens == 15


def test_dflash_scheduling_unchanged(tmp_path) -> None:
    cfg = _vllm_config(tmp_path, "dflash")

    assert cfg.speculative_config.use_dflash()
    assert not cfg.speculative_config.use_dflex()
    assert cfg.speculative_config.parallel_drafting
    assert cfg.num_lookahead_tokens == 16

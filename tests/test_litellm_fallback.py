import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)

from dependency_stubs import install_dependency_stubs


install_dependency_stubs()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LiteLLMChatWrapper, ModelConfig, ModelType


def _make_wrapper():
    config = ModelConfig(
        type=ModelType.CHAT,
        provider="openai",
        name="primary",
        fallbacks=["fallback", "backup"],
    )
    return LiteLLMChatWrapper(model=config.name, provider=config.provider, model_config=config)


def test_model_sequence_prioritizes_last_successful_and_deprioritizes_failed():
    wrapper = _make_wrapper()

    # Simulate the primary model failing and the fallback succeeding
    wrapper._register_failed_model("primary")
    wrapper._update_successful_model("fallback")

    sequence = wrapper._model_sequence()

    assert sequence[0] == "fallback"
    assert sequence[-1] == "primary"
    assert sequence == ["fallback", "backup", "primary"]


def test_success_resets_failure_tracking():
    wrapper = _make_wrapper()

    wrapper._register_failed_model("fallback")
    assert "fallback" in wrapper._failed_models

    wrapper._update_successful_model("fallback")

    assert "fallback" not in wrapper._failed_models
    assert wrapper._model_sequence()[0] == "fallback"

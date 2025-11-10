import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)

from dependency_stubs import install_dependency_stubs


install_dependency_stubs()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.extensions.system_prompt import _10_system_prompt as system_prompt_module


class DummyAgent:
    def __init__(self, prompts: dict[str, str] | None = None):
        self._prompts = prompts or {}

    def read_prompt(self, name: str) -> str:
        if name not in self._prompts:
            raise FileNotFoundError(name)
        return self._prompts[name]


@pytest.fixture
def reset_settings(monkeypatch):
    original_get_settings = system_prompt_module.get_settings
    data = {
        "system_prompt_main_override": "",
        "system_prompt_user_preferences": "",
    }

    def fake_settings():
        return data

    monkeypatch.setattr(system_prompt_module, "get_settings", fake_settings)
    yield data
    monkeypatch.setattr(system_prompt_module, "get_settings", original_get_settings)


def test_get_main_prompt_uses_default_when_override_missing(reset_settings):
    agent = DummyAgent({"agent.system.main.md": "default prompt"})
    assert system_prompt_module.get_main_prompt(agent) == "default prompt"


def test_get_main_prompt_prefers_override(reset_settings):
    agent = DummyAgent({"agent.system.main.md": "default prompt"})
    reset_settings["system_prompt_main_override"] = "Custom override"
    assert system_prompt_module.get_main_prompt(agent) == "Custom override"


def test_user_preferences_returns_template_when_not_customized(reset_settings):
    agent = DummyAgent({"agent.system.user-preferences.md": "template text"})
    assert system_prompt_module.get_user_preferences_prompt(agent) == "template text"


def test_user_preferences_uses_custom_text_when_provided(reset_settings):
    agent = DummyAgent({"agent.system.user-preferences.md": "template text"})
    reset_settings["system_prompt_user_preferences"] = "User prefers concise replies."
    assert (
        system_prompt_module.get_user_preferences_prompt(agent)
        == "User prefers concise replies."
    )


def test_user_preferences_handles_missing_template(reset_settings):
    agent = DummyAgent()
    assert system_prompt_module.get_user_preferences_prompt(agent) == ""
    reset_settings["system_prompt_user_preferences"] = "Always be kind."
    assert system_prompt_module.get_user_preferences_prompt(agent) == "Always be kind."

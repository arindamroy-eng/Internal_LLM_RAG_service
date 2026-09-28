"""
Validation of the `model` alias on agent create/update.

Before this, `agents.model` was a free-form string: a typo was accepted at
creation, persisted to Postgres, and only surfaced as a 502 on the user's
first chat — far from the mistake, and blamed on the LLM rather than on the
agent config.
"""

import pytest
from pydantic import ValidationError

from app.config import settings
from app.models.schemas import AgentCreate, AgentUpdate


VALID = dict(name="Research Assistant", system_prompt="You are helpful.")


def test_known_aliases_are_accepted():
    for alias in settings.allowed_agent_models:
        agent = AgentCreate(**VALID, model=alias)
        assert agent.model == alias


def test_unknown_alias_is_rejected_at_creation():
    with pytest.raises(ValidationError) as exc:
        AgentCreate(**VALID, model="gpt-4.1")

    message = str(exc.value)
    assert "gpt-4.1" in message
    # The error must name the valid options, or the user cannot self-correct.
    assert settings.chat_model_alias in message


def test_embedding_alias_is_rejected_for_agents():
    """An agent cannot converse with an embedding model."""
    with pytest.raises(ValidationError):
        AgentCreate(**VALID, model=settings.embed_model_alias)


def test_default_model_comes_from_config_not_a_literal():
    assert AgentCreate(**VALID).model == settings.chat_model_alias


def test_code_model_alias_is_a_valid_choice():
    """
    Guards against code_model_alias drifting back into dead config — it had
    no consumer anywhere in the codebase before this validator.
    """
    assert settings.code_model_alias in settings.allowed_agent_models
    assert AgentCreate(**VALID, model=settings.code_model_alias).model == (
        settings.code_model_alias
    )


def test_update_rejects_unknown_alias():
    with pytest.raises(ValidationError):
        AgentUpdate(model="llama-405b")


def test_update_allows_omitting_model():
    """None means 'leave unchanged' and must not trip the validator."""
    assert AgentUpdate(name="renamed").model is None
    assert AgentUpdate(model=None).model is None

"""
Unit tests for LLMClient and deterministic code synthesis in project_marl.workspace.llm_client.
"""

import ast
import os
import pytest
from project_marl.workspace.llm_client import LLMClient, LLMConfig


def test_llm_client_default_fallback() -> None:
    """Verifies that LLMClient defaults to deterministic fallback when API keys are absent."""
    config = LLMConfig(provider="fallback", openai_api_key=None, anthropic_api_key=None)
    client = LLMClient(config)

    assert client.provider == "fallback"
    assert client.is_live() is False


def test_llm_client_deterministic_synthesis_new_file() -> None:
    """Verifies synthesis on a brand new file produces valid AST Python code."""
    client = LLMClient(LLMConfig(provider="fallback"))
    code = client.synthesize_code(
        task_title="Implement User Authentication",
        task_description="Implement JWT verify function and session validator",
        file_path="src/auth.py",
        existing_code="",
    )

    assert code is not None
    assert len(code) > 0
    # Must parse without syntax errors
    tree = ast.parse(code)
    assert tree is not None
    assert "execute_implement_user_authentication" in code


def test_llm_client_deterministic_synthesis_append_existing() -> None:
    """Verifies synthesis appends new functions without destroying existing code."""
    client = LLMClient(LLMConfig(provider="fallback"))
    existing = (
        "def base_health_check() -> bool:\n"
        "    return True\n"
    )

    updated = client.synthesize_code(
        task_title="Add Rate Limiting",
        task_description="Implement token bucket algorithm",
        file_path="src/limiter.py",
        existing_code=existing,
    )

    assert "base_health_check" in updated
    assert "execute_add_rate_limiting" in updated

    # AST validation
    tree = ast.parse(updated)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "base_health_check" in func_names
    assert "execute_add_rate_limiting" in func_names


def test_llm_client_strip_code_fences() -> None:
    """Verifies stripping of markdown code fences."""
    raw = "```python\ndef test():\n    pass\n```"
    stripped = LLMClient._strip_code_fences(raw)
    assert stripped == "def test():\n    pass"

    raw_plain = "def test():\n    pass"
    assert LLMClient._strip_code_fences(raw_plain) == raw_plain


def test_llm_client_provider_resolution() -> None:
    """Tests provider auto-resolution based on config keys."""
    cfg_openai = LLMConfig(provider="auto", openai_api_key="sk-test-key")
    client_openai = LLMClient(cfg_openai)
    assert client_openai.provider == "openai"
    assert client_openai.is_live() is True

    cfg_anthropic = LLMConfig(provider="auto", openai_api_key=None, anthropic_api_key="sk-ant-test")
    client_anthropic = LLMClient(cfg_anthropic)
    assert client_anthropic.provider == "anthropic"
    assert client_anthropic.is_live() is True

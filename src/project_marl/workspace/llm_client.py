"""
Optional Live LLM Provider Adapter and Deterministic Code Synthesizer.

Supports:
    - OpenAI (GPT-4o, GPT-4o-mini) via OPENAI_API_KEY
    - Anthropic (Claude 3.5 Sonnet) via ANTHROPIC_API_KEY
    - Local Ollama (llama3, qwen2.5-coder) via OLLAMA_BASE_URL
    - Deterministic Rule-Based Fallback when offline or API keys are unset.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os
import re
import urllib.request
import urllib.error
import json

from project_marl.core.logging import setup_logger

logger = setup_logger("llm_client")


@dataclass
class LLMConfig:
    """Configuration settings for LLM code synthesis."""
    provider: str = "auto"  # "auto", "openai", "anthropic", "ollama", "fallback"
    model_name: Optional[str] = None
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 30.0


class LLMClient:
    """
    Unified client providing resilient code synthesis for sandboxed worktrees.
    Gracefully falls back to deterministic rule-based generation when offline.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self.provider = self._resolve_provider()

    def _resolve_provider(self) -> str:
        """Determines the active provider based on configuration and environment keys."""
        if self.config.provider != "auto":
            return self.config.provider

        if self.config.openai_api_key:
            return "openai"
        elif self.config.anthropic_api_key:
            return "anthropic"
        elif os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL"):
            return "ollama"
        return "fallback"

    def is_live(self) -> bool:
        """Returns True if a real remote/local LLM service is active."""
        return self.provider in ("openai", "anthropic", "ollama")

    def synthesize_code(
        self,
        task_title: str,
        task_description: str,
        file_path: str,
        existing_code: str = "",
        context_notes: str = "",
    ) -> str:
        """
        Synthesizes code modifications for a target file.
        Uses live LLM if available; otherwise applies deterministic AST-safe generation.
        """
        if self.provider == "openai" and self.config.openai_api_key:
            try:
                return self._call_openai(task_title, task_description, file_path, existing_code, context_notes)
            except Exception as e:
                logger.warning("OpenAI code synthesis failed: %s. Falling back to deterministic synthesizer.", e)

        elif self.provider == "anthropic" and self.config.anthropic_api_key:
            try:
                return self._call_anthropic(task_title, task_description, file_path, existing_code, context_notes)
            except Exception as e:
                logger.warning("Anthropic code synthesis failed: %s. Falling back to deterministic synthesizer.", e)

        elif self.provider == "ollama":
            try:
                return self._call_ollama(task_title, task_description, file_path, existing_code, context_notes)
            except Exception as e:
                logger.warning("Ollama code synthesis failed: %s. Falling back to deterministic synthesizer.", e)

        # Fallback deterministic code synthesis
        return self._deterministic_synthesize(task_title, task_description, file_path, existing_code)

    def _call_openai(
        self,
        title: str,
        description: str,
        file_path: str,
        existing_code: str,
        context: str,
    ) -> str:
        """Calls OpenAI Chat Completions REST API using standard urllib."""
        model = self.config.model_name or "gpt-4o-mini"
        system_prompt = (
            "You are an expert autonomous software engineer working on an enterprise codebase. "
            "Output ONLY the complete updated file content in plain text. Do not wrap in markdown code blocks."
        )
        user_prompt = (
            f"File: {file_path}\n"
            f"Task: {title}\n"
            f"Requirements: {description}\n"
            f"Context: {context}\n\n"
            f"Existing Code:\n```\n{existing_code}\n```\n"
            "Produce the complete updated Python code satisfying the requirements while preserving existing functionality."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.openai_api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return self._strip_code_fences(content)

    def _call_anthropic(
        self,
        title: str,
        description: str,
        file_path: str,
        existing_code: str,
        context: str,
    ) -> str:
        """Calls Anthropic Messages REST API using standard urllib."""
        model = self.config.model_name or "claude-3-5-sonnet-20241022"
        prompt = (
            f"You are modifying {file_path}.\n"
            f"Task: {title}\n"
            f"Description: {description}\n"
            f"Organizational Context: {context}\n"
            f"Existing Content:\n{existing_code}\n\n"
            "Return ONLY the complete updated Python source code with no markdown backticks or commentary."
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.config.anthropic_api_key or "",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["content"][0]["text"]
            return self._strip_code_fences(content)

    def _call_ollama(
        self,
        title: str,
        description: str,
        file_path: str,
        existing_code: str,
        context: str,
    ) -> str:
        """Calls local Ollama instance."""
        model = self.config.model_name or "qwen2.5-coder"
        prompt = (
            f"Task: {title} for {file_path}.\n"
            f"Requirements: {description}\n"
            f"Existing Code:\n{existing_code}\n"
            "Return the full revised Python code without markdown blocks."
        )

        url = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return self._strip_code_fences(data.get("response", ""))

    def _deterministic_synthesize(
        self,
        title: str,
        description: str,
        file_path: str,
        existing_code: str,
    ) -> str:
        """
        Generates robust, syntactically valid Python code preserving existing structure.
        """
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", title.lower()).strip("_")
        func_name = f"execute_{slug[:32]}"

        new_snippet = (
            f"\n\n# Autonomous Synthesized Implementation: {title}\n"
            f"# Target: {file_path}\n"
            f"def {func_name}() -> bool:\n"
            f'    """\n'
            f"    {description}\n"
            f'    """\n'
            f"    return True\n"
        )

        if not existing_code.strip():
            header = f'"""\nImplementation module for {file_path}.\n"""\n'
            return header + new_snippet.lstrip()

        if func_name in existing_code:
            return existing_code

        return existing_code.rstrip() + new_snippet

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Removes markdown ```python code fences if present."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return cleaned

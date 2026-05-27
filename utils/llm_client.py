from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Model catalog ────────────────────────────────────────────────────────────

MODEL_CATALOG: list[dict] = [
    # Anthropic native
    {
        "label": "Claude Sonnet 4   (Anthropic)",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
    },
    {
        "label": "Claude Haiku 3.5  (Anthropic)",
        "provider": "anthropic",
        "model_id": "claude-haiku-3-5-20241022",
    },
    {
        "label": "Claude Opus 4     (Anthropic)",
        "provider": "anthropic",
        "model_id": "claude-opus-4-20250514",
    },
    # Claude via OpenRouter
    {
        "label": "Claude Sonnet 4   (OpenRouter)",
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4",
    },
    {
        "label": "Claude Haiku 3.5  (OpenRouter)",
        "provider": "openrouter",
        "model_id": "anthropic/claude-haiku-3-5",
    },
    # OpenAI via OpenRouter
    {
        "label": "GPT-4o            (OpenRouter)",
        "provider": "openrouter",
        "model_id": "openai/gpt-4o",
    },
    {
        "label": "GPT-4o mini       (OpenRouter)",
        "provider": "openrouter",
        "model_id": "openai/gpt-4o-mini",
    },
    {
        "label": "o3 mini           (OpenRouter)",
        "provider": "openrouter",
        "model_id": "openai/o3-mini",
    },
    # Google via OpenRouter
    {
        "label": "Gemini 2.0 Flash  (OpenRouter)",
        "provider": "openrouter",
        "model_id": "google/gemini-2.0-flash-001",
    },
    {
        "label": "Gemini 1.5 Pro    (OpenRouter)",
        "provider": "openrouter",
        "model_id": "google/gemini-pro-1.5",
    },
    # Meta via OpenRouter
    {
        "label": "Llama 3.3 70B     (OpenRouter)",
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
    },
    {
        "label": "Llama 3.1 405B    (OpenRouter)",
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.1-405b-instruct",
    },
    # DeepSeek via OpenRouter
    {
        "label": "DeepSeek V3       (OpenRouter)",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-chat",
    },
    {
        "label": "DeepSeek R1       (OpenRouter)",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
    },
    # Mistral via OpenRouter
    {
        "label": "Mistral Large     (OpenRouter)",
        "provider": "openrouter",
        "model_id": "mistralai/mistral-large-2411",
    },
    # Qwen via OpenRouter
    {
        "label": "Qwen 2.5 72B      (OpenRouter)",
        "provider": "openrouter",
        "model_id": "qwen/qwen-2.5-72b-instruct",
    },
    # OpenAI native
    {"label": "GPT-4o            (OpenAI)", "provider": "openai", "model_id": "gpt-4o"},
    {
        "label": "GPT-4o mini       (OpenAI)",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
    },
    # Free tier via OpenRouter (rate-limited)
    {
        "label": "DeepSeek V4 Flash (Free)",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v4-flash:free",
    },
    {
        "label": "Gemini Flash 2.0  (Free)",
        "provider": "openrouter",
        "model_id": "google/gemini-2.0-flash-exp:free",
    },
    {
        "label": "Llama 3.3 70B     (Free)",
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct:free",
    },
]

PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openrouter": "anthropic/claude-sonnet-4",
    "openai": "gpt-4o",
}

# ─── Client ───────────────────────────────────────────────────────────────────


class LLMClient:
    def __init__(self, provider: str, api_key: str, model_id: str) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model_id = model_id

        if provider == "anthropic":
            from anthropic import Anthropic  # lazy import

            self._client: Any = Anthropic(api_key=api_key)
        elif provider == "openrouter":
            from openai import OpenAI  # lazy import

            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        elif provider == "openai":
            from openai import OpenAI  # lazy import

            self._client = OpenAI(api_key=api_key)
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                "Expected 'anthropic', 'openrouter', or 'openai'."
            )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2500,
        temperature: float = 0.2,
    ) -> str:
        logger.info("LLM call → provider=%s model=%s", self.provider, self.model_id)

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )

        # openrouter or openai — both use the OpenAI-compatible chat completions API
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.provider == "openrouter":
            kwargs["extra_headers"] = {
                "HTTP-Referer": "https://github.com/InvestorAI",
                "X-Title": "InvestorAI Stock Analyst",
            }
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @property
    def display_name(self) -> str:
        return f"{self.model_id} ({self.provider})"


# ─── Factory ──────────────────────────────────────────────────────────────────


def build_llm_client(
    provider: str | None = None,
    model_id: str | None = None,
) -> LLMClient | None:
    """Resolve provider + model_id from args, env vars, and auto-detection, then return an LLMClient.

    Returns None (with a warning) if no usable API key can be found.
    """
    from config.settings import (  # local import to avoid circular deps at module load
        ANTHROPIC_API_KEY,
        LLM_MODEL,
        LLM_PROVIDER,
        OPENAI_API_KEY,
        OPENROUTER_API_KEY,
    )

    # 1. Resolve provider: explicit arg > LLM_PROVIDER env > auto-detect
    resolved_provider: str = (
        provider
        or LLM_PROVIDER
        or _auto_detect_provider(ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY)
    )
    if not resolved_provider:
        logger.warning(
            "No LLM provider could be determined and no API keys are configured. "
            "Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
        )
        return None

    # 2. Resolve API key for that provider
    api_key = _key_for_provider(
        resolved_provider, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY
    )
    if not api_key:
        logger.warning(
            "No API key found for provider %r. "
            "Set the corresponding environment variable.",
            resolved_provider,
        )
        return None

    # 3. Resolve model_id
    #    - explicit arg always wins
    #    - if provider was explicitly overridden (passed as arg) but no model given,
    #      use that provider's default (don't let an env model from a different provider bleed in)
    #    - otherwise honour LLM_MODEL from env
    if model_id:
        resolved_model: str = model_id
    elif provider:  # explicit provider override → use its default
        resolved_model = PROVIDER_DEFAULTS.get(resolved_provider, "")
    else:  # fully auto / env-driven → honour LLM_MODEL
        resolved_model = _resolve_model_id(LLM_MODEL, resolved_provider)

    return LLMClient(
        provider=resolved_provider, api_key=api_key, model_id=resolved_model
    )


# ── private helpers ───────────────────────────────────────────────────────────


def _auto_detect_provider(
    anthropic_key: str,
    openrouter_key: str,
    openai_key: str,
) -> str:
    """Return the first provider whose API key is non-empty."""
    if anthropic_key:
        return "anthropic"
    if openrouter_key:
        return "openrouter"
    if openai_key:
        return "openai"
    return ""


def _key_for_provider(
    provider: str,
    anthropic_key: str,
    openrouter_key: str,
    openai_key: str,
) -> str:
    return {
        "anthropic": anthropic_key,
        "openrouter": openrouter_key,
        "openai": openai_key,
    }.get(provider, "")


def _resolve_model_id(env_model: str, provider: str) -> str:
    """Look up *env_model* in MODEL_CATALOG (by label or model_id).

    Falls back to treating the value as a raw model ID if not found,
    or to the provider default when *env_model* is empty.
    """
    if env_model:
        for entry in MODEL_CATALOG:
            if entry["label"] == env_model or entry["model_id"] == env_model:
                return entry["model_id"]
        # Not in catalog — treat as a raw model ID passed directly
        return env_model
    return PROVIDER_DEFAULTS.get(provider, "")

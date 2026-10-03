"""Realtime SIP adapters for the built-in voice agent."""

from .realtime import GrokAdapter, OpenAIAdapter, ProviderError
from .webhook import verify_webhook


def create_adapter(binding: dict):
    provider = binding.get("provider")
    if provider == "openai":
        return OpenAIAdapter(binding)
    if provider == "grok":
        return GrokAdapter(binding)
    raise ValueError("unsupported provider")


__all__ = ["create_adapter", "verify_webhook", "ProviderError", "OpenAIAdapter", "GrokAdapter"]

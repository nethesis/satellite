"""Realtime SIP adapters for the built-in voice agent."""

from .realtime import GrokAdapter, OpenAIAdapter, ProviderError
from .webhook import verify_webhook


def uses_live(binding, profile=None):
    return binding.get("provider") == "openai" and str((profile or {}).get("model") or "").startswith("gpt-live-")


def create_adapter(binding: dict, profile=None):
    provider = binding.get("provider")
    if provider == "openai":
        if uses_live(binding, profile):
            from .live import OpenAILiveAdapter
            return OpenAILiveAdapter(binding)
        return OpenAIAdapter(binding)
    if provider == "grok":
        return GrokAdapter(binding)
    raise ValueError("unsupported provider")


__all__ = ["create_adapter", "verify_webhook", "ProviderError", "OpenAIAdapter", "GrokAdapter"]

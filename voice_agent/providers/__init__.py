"""Provider-specific SIP sideband adapters; GPT-Live is a separate future contract."""
from .realtime import OpenAIAdapter, GrokAdapter, ProviderError
from .webhook import verify_webhook

def create_adapter(binding):
    if binding["provider"] == "openai":
        return OpenAIAdapter(binding)
    if binding["provider"] == "grok":
        return GrokAdapter(binding)
    raise ValueError("invalid_provider")

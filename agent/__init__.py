"""Common Satellite Agent contracts; no provider or Asterisk dependencies."""

from .configuration import ConfigurationStore
from .events import EventSink
from .tools import ToolRegistry

__all__ = ["ConfigurationStore", "EventSink", "ToolRegistry"]

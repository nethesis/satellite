"""Purpose-bound authenticated encryption; no secret values in configuration."""

import base64
import json
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .contracts import ApplicationError, canonical


class ContentKey:
    def __init__(self, encoded=None):
        self.key = None
        try:
            raw = base64.b64decode(encoded if encoded is not None else os.getenv("SATELLITE_APPLICATION_CONTENT_KEY", ""), validate=True)
            if len(raw) == 32:
                self.key = AESGCM(raw)
        except (ValueError, TypeError):
            pass

    def encrypt(self, purpose, value):
        if self.key is None:
            raise ApplicationError("content_key_unavailable", 503)
        nonce = os.urandom(12)
        return nonce + self.key.encrypt(nonce, canonical(value).encode(), purpose.encode())

    def decrypt(self, purpose, value):
        if self.key is None:
            raise ApplicationError("content_key_unavailable", 503)
        try:
            raw = bytes(value)
            return json.loads(self.key.decrypt(raw[:12], raw[12:], purpose.encode()))
        except Exception:
            raise ApplicationError("content_unavailable", 503) from None


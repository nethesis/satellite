import json
import logging
import sys
from typing import Any, Dict

import ai
import db

logger = logging.getLogger("call_processor")


def _read_stdin_json() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Expected JSON on stdin")
    return json.loads(raw)


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    try:
        payload = _read_stdin_json()
        transcript_id = int(payload["transcript_id"])
        raw_transcription = str(payload["raw_transcription"])
        summarize = bool(payload.get("summarize", False))

        if db.is_configured():
            db.replace_transcript_embeddings(
                transcript_id=transcript_id,
                raw_transcription=raw_transcription,
            )

            if not summarize:
                sys.stdout.write(json.dumps({"ok": True, "sentiment": None}))
                return 0

            cleaned, summary, sentiment = ai.generate_clean_summary_sentiment(raw_transcription)

            db.update_transcript_ai_fields(
                transcript_id=transcript_id,
                cleaned_transcription=cleaned,
                summary=summary,
                sentiment=sentiment,
            )
        sys.stdout.write(json.dumps({"ok": True, "sentiment": sentiment}))
        return 0
    except Exception:
        logger.exception("Call processing failed")
        sys.stdout.write(json.dumps({"ok": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

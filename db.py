import logging
import os
import re
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import psycopg
from pgvector.psycopg import register_vector

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("db")


_EMBEDDING_MODEL = "text-embedding-3-small"
# OpenAI text-embedding-3-small is 1536 dims.
_EMBEDDING_DIM = 1536

_schema_lock = threading.Lock()
_schema_initialized = False


def is_configured() -> bool:
    required = [
        "PGVECTOR_HOST",
        "PGVECTOR_PORT",
        "PGVECTOR_USER",
        "PGVECTOR_PASSWORD",
        "PGVECTOR_DATABASE",
    ]
    return all(os.getenv(k) for k in required)


def _conninfo() -> str:
    host = os.getenv("PGVECTOR_HOST")
    port = os.getenv("PGVECTOR_PORT")
    user = os.getenv("PGVECTOR_USER")
    password = os.getenv("PGVECTOR_PASSWORD")
    dbname = os.getenv("PGVECTOR_DATABASE")

    if not all([host, port, user, password, dbname]):
        missing = [k for k in ["PGVECTOR_HOST", "PGVECTOR_PORT", "PGVECTOR_USER", "PGVECTOR_PASSWORD", "PGVECTOR_DATABASE"] if not os.getenv(k)]
        raise RuntimeError(f"Postgres vectorstore is not configured; missing env vars: {', '.join(missing)}")

    # psycopg conninfo format
    return f"host={host} port={port} user={user} password={password} dbname={dbname}"


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(_conninfo())
    register_vector(conn)
    return conn


def _ensure_schema() -> None:
    global _schema_initialized
    if _schema_initialized:
        return

    with _schema_lock:
        if _schema_initialized:
            return

        with _connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id BIGSERIAL PRIMARY KEY,
                    uniqueid TEXT NOT NULL UNIQUE,
                    raw_transcription TEXT NOT NULL,
                    cleaned_transcription TEXT,
                    summary TEXT,
                    detected_language TEXT,
                    diarized_transcript TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    transcript_id BIGINT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(%s) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (transcript_id, chunk_index)
                )
                """,
                (_EMBEDDING_DIM,),
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS transcript_chunks_transcript_id_idx ON transcript_chunks (transcript_id)"
            )

            # "Modern" pgvector index: HNSW (if supported by server pgvector version)
            try:
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS transcript_chunks_embedding_hnsw
                    ON transcript_chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """
                )
            except Exception:
                logger.warning("HNSW index creation failed; pgvector may be too old", exc_info=True)

        _schema_initialized = True


_uniqueid_re = re.compile(r"^\d+\.\d+$")


def validate_uniqueid(uniqueid: str) -> None:
    if not uniqueid or not uniqueid.strip():
        raise ValueError("Missing required form field 'uniqueid'")
    if not _uniqueid_re.match(uniqueid.strip()):
        raise ValueError("Invalid 'uniqueid' format; expected like 1234567890.1234")


def upsert_transcript_raw(
    *,
    uniqueid: str,
    raw_transcription: str,
    detected_language: Optional[str] = None,
    diarized_transcript: Optional[str] = None,
) -> int:
    """Insert or update the raw transcript row and return its transcript id."""

    validate_uniqueid(uniqueid)
    _ensure_schema()

    with _connect() as conn:
        row = conn.execute(
            """
            INSERT INTO transcripts (uniqueid, raw_transcription, detected_language, diarized_transcript)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uniqueid)
            DO UPDATE SET
                raw_transcription = EXCLUDED.raw_transcription,
                detected_language = EXCLUDED.detected_language,
                diarized_transcript = EXCLUDED.diarized_transcript,
                updated_at = now()
            RETURNING id
            """,
            (uniqueid, raw_transcription, detected_language, diarized_transcript),
        ).fetchone()

        if row is None:
            raise RuntimeError("Failed to upsert transcript")
        return int(row[0])


def update_transcript_ai_fields(*, transcript_id: int, cleaned_transcription: str, summary: str) -> None:
    _ensure_schema()

    with _connect() as conn:
        conn.execute(
            """
            UPDATE transcripts
            SET cleaned_transcription = %s,
                summary = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (cleaned_transcription, summary, transcript_id),
        )


def _split_text_for_embedding(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text or "")]
    return [c for c in chunks if c]


def replace_transcript_embeddings(*, transcript_id: int, raw_transcription: str) -> int:
    """Replace all chunk embeddings for transcript_id. Returns number of chunks stored."""

    _ensure_schema()

    chunks = _split_text_for_embedding(raw_transcription)
    if not chunks:
        return 0

    embedder = OpenAIEmbeddings(model=_EMBEDDING_MODEL)
    vectors = embedder.embed_documents(chunks)

    with _connect() as conn:
        conn.execute("DELETE FROM transcript_chunks WHERE transcript_id = %s", (transcript_id,))
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            conn.execute(
                """
                INSERT INTO transcript_chunks (transcript_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (transcript_id, idx, chunk, vector),
            )

    return len(chunks)

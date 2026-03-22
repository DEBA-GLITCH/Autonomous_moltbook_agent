from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator


class MemoryStore:
    """SQLite-backed memory for de-duplication, pacing, and short history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    post_id TEXT,
                    author_id TEXT,
                    content_hash TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(action_type, target_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT,
                    title TEXT NOT NULL,
                    topic_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def content_hash(text: str) -> str:
        """Build a stable hash used to detect repeated content."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def has_action(self, action_type: str, target_id: str) -> bool:
        """Check whether we have already performed a specific action on target."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM interactions
                WHERE action_type = ? AND target_id = ?
                LIMIT 1
                """,
                (action_type, target_id),
            ).fetchone()
            return row is not None

    def record_action(
        self,
        action_type: str,
        target_id: str,
        *,
        post_id: str = "",
        author_id: str = "",
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one performed action (reply/post/etc)."""
        now = datetime.now(timezone.utc).isoformat()
        content_hash = self.content_hash(content) if content else ""
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO interactions (
                    action_type, target_id, post_id, author_id, content_hash, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_type,
                    target_id,
                    post_id,
                    author_id,
                    content_hash,
                    metadata_json,
                    now,
                ),
            )
            conn.commit()

    def recently_posted(self, minutes: int) -> bool:
        """True when a create_post action exists in the configured time window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM interactions
                WHERE action_type = 'create_post' AND created_at >= ?
                LIMIT 1
                """,
                (cutoff,),
            ).fetchone()
            return row is not None

    def is_duplicate_content(self, text: str, lookback_days: int = 14) -> bool:
        """Prevent posting/commenting the same thing repeatedly."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        hashed = self.content_hash(text)
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM interactions
                WHERE content_hash = ? AND created_at >= ?
                LIMIT 1
                """,
                (hashed, cutoff),
            ).fetchone()
            return row is not None

    def remember_post_topic(self, post_id: str, title: str) -> None:
        """Store lightweight topic fingerprints to avoid repetitive new posts."""
        now = datetime.now(timezone.utc).isoformat()
        fingerprint = self.content_hash(title)[:16]
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO posts (post_id, title, topic_fingerprint, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (post_id, title, fingerprint, now),
            )
            conn.commit()

    def recent_topics(self, limit: int = 12) -> list[str]:
        """Get recent post titles for prompt conditioning."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT title FROM posts
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [str(row[0]) for row in rows]

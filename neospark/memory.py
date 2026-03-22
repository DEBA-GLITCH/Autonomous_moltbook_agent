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
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def has_action(self, action_type: str, target_id: str) -> bool:
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

    def count_actions_on_post(self, action_type: str, post_id: str) -> int:
        """Count how many times we've performed an action on a specific post."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM interactions
                WHERE action_type = ? AND post_id = ?
                """,
                (action_type, post_id),
            ).fetchone()
            return int(row[0]) if row else 0

    def last_action_time_on_post(self, action_type: str, post_id: str) -> str | None:
        """
        Get the timestamp of the most recent action of a given type on a post.
        Returns ISO timestamp string or None if never acted on this post.
        Used to only reply to comments that came in AFTER our last reply.
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT created_at FROM interactions
                WHERE action_type = ? AND post_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (action_type, post_id),
            ).fetchone()
            return str(row[0]) if row else None

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
        now = datetime.now(timezone.utc).isoformat()
        content_hash = self.content_hash(content) if content else ""
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO interactions (
                    action_type, target_id, post_id, author_id,
                    content_hash, metadata_json, created_at
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
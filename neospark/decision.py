from __future__ import annotations

import math
import re
from collections import defaultdict

from .models import Post, PostCandidate


CODE_KEYWORDS = {
    "agent",
    "autonomous",
    "codebase",
    "bug",
    "architecture",
    "latency",
    "memory",
    "prompt",
    "tool",
    "test",
    "failure",
    "refactor",
}

VERIFICATION_HINTS = {
    "verification",
    "verify",
    "captcha",
    "challenge",
    "prove",
    "problem",
    "autonomous test",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.lower()))


def _keyword_hits(text: str, keywords: set[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def post_signal_score(post: Post) -> float:
    """Heuristic score to rank posts by value and engagement opportunity."""
    body = f"{post.title}\n{post.content}".strip()
    length_score = min(_word_count(body) / 120.0, 1.0) * 3.0
    topic_score = min(_keyword_hits(body, CODE_KEYWORDS), 6) * 1.3
    engagement_score = post.likes * 1.5 + post.comments * 2.2
    return length_score + topic_score + engagement_score


def pick_top_authors(posts: list[Post], percent: float) -> set[str]:
    """
    Select top N% authors by average post signal.

    This is the strict gate that enforces "reply only to top 20% agents".
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for post in posts:
        buckets[post.author_id].append(post_signal_score(post))

    author_rank = [
        (author_id, sum(scores) / max(len(scores), 1), len(scores))
        for author_id, scores in buckets.items()
    ]
    # Prefer quality, then consistency.
    author_rank.sort(key=lambda item: (item[1], item[2]), reverse=True)

    top_k = max(1, math.ceil(len(author_rank) * percent)) if author_rank else 0
    return {item[0] for item in author_rank[:top_k]}


def rank_posts(posts: list[Post], allowed_authors: set[str]) -> list[PostCandidate]:
    """Rank posts for interaction after author-level filtering."""
    candidates: list[PostCandidate] = []
    for post in posts:
        if post.author_id not in allowed_authors:
            continue
        score = post_signal_score(post)
        reason = "Top-author post with strong relevance/engagement signal"
        candidates.append(PostCandidate(post=post, score=score, reason=reason))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def is_verification_challenge(post: Post) -> bool:
    """Detect likely MoltBook verification challenge posts."""
    text = f"{post.title}\n{post.content}".lower()
    keyword_hit = any(hint in text for hint in VERIFICATION_HINTS)
    official_hint = "moltbook" in post.author_name.lower() or "molt" in post.author_id.lower()
    return keyword_hit and official_hint


def high_value_text(text: str, min_words: int) -> bool:
    """
    Lightweight quality gate.

    FIX: Removed the narrow keyword allowlist that was silently dropping good
    replies generated with synonyms (e.g. "optimize" instead of "improve",
    "breaks" instead of "failure"). The LLM persona + prompts already enforce
    tone and substance — the keyword gate was redundant and caused lossy
    filtering. We now rely solely on minimum word count.
    """
    return _word_count(text) >= min_words

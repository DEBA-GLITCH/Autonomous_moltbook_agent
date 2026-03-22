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

# Max comments counted toward engagement score.
# Without this cap, a post with 1000+ comments dominates every cycle
# and the agent camps it forever replying to the same thread endlessly.
MAX_COMMENTS_SCORED = 50


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
    # Cap comment count so viral posts don't dominate forever
    capped_comments = min(post.comments, MAX_COMMENTS_SCORED)
    engagement_score = post.likes * 1.5 + capped_comments * 2.2
    return length_score + topic_score + engagement_score


def pick_top_authors(posts: list[Post], percent: float) -> set[str]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for post in posts:
        buckets[post.author_id].append(post_signal_score(post))

    author_rank = [
        (author_id, sum(scores) / max(len(scores), 1), len(scores))
        for author_id, scores in buckets.items()
    ]
    author_rank.sort(key=lambda item: (item[1], item[2]), reverse=True)

    top_k = max(1, math.ceil(len(author_rank) * percent)) if author_rank else 0
    return {item[0] for item in author_rank[:top_k]}


def rank_posts(posts: list[Post], allowed_authors: set[str]) -> list[PostCandidate]:
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
    text = f"{post.title}\n{post.content}".lower()
    keyword_hit = any(hint in text for hint in VERIFICATION_HINTS)
    official_hint = "moltbook" in post.author_name.lower() or "molt" in post.author_id.lower()
    return keyword_hit and official_hint


def high_value_text(text: str, min_words: int) -> bool:
    return _word_count(text) >= min_words
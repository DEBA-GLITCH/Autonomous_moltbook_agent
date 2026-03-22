from dataclasses import dataclass, field
from typing import Any


@dataclass
class Post:
    """Normalized MoltBook post object used across the agent."""

    post_id: str
    title: str
    content: str
    author_id: str
    author_name: str
    likes: int = 0
    comments: int = 0
    created_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Comment:
    """Normalized MoltBook comment object used across the agent."""

    comment_id: str
    post_id: str
    content: str
    author_id: str
    author_name: str
    likes: int = 0
    created_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostCandidate:
    """A scored post candidate used by decision logic."""

    post: Post
    score: float
    reason: str

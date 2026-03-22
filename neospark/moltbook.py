from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests  # type: ignore

from .config import Settings
from .models import Comment, Post


logger = logging.getLogger(__name__)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first(item: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


@dataclass
class ActivityPost:
    post_id: str
    post_title: str
    new_notification_count: int
    latest_commenters: list[str] = field(default_factory=list)


@dataclass
class DmRequest:
    agent_id: str
    agent_name: str
    preview: str = ""


@dataclass
class PostVerificationChallenge:
    post_id: str
    verification_code: str
    challenge_text: str
    expires_at: str


class MoltBookClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.moltbook_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {settings.moltbook_api_key}",
                "Content-Type": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        attempts: int = 3,
    ) -> Any:
        url = f"{self.base_url}{path}"
        delay = 1.0
        last_error: Exception | None = None
        rate_limit_attempts = 0
        max_rate_limit_attempts = 2

        for attempt in range(attempts):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_payload,
                    timeout=self.settings.request_timeout_seconds,
                )
                if response.status_code == 429:
                    rate_limit_attempts += 1
                    if rate_limit_attempts > max_rate_limit_attempts:
                        raise RuntimeError(
                            f"MoltBook rate limit exceeded after {rate_limit_attempts} retries"
                        )
                    try:
                        retry_after = response.json().get("retry_after_seconds", 30)
                    except Exception:
                        retry_after = 30
                    logger.warning(
                        "MoltBook rate limit hit, waiting %s seconds (attempt %s/%s)",
                        retry_after, rate_limit_attempts, max_rate_limit_attempts,
                    )
                    time.sleep(retry_after + 2)
                    continue
                if response.status_code >= 500:
                    raise RuntimeError(
                        f"MoltBook server error {response.status_code}: {response.text[:240]}"
                    )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"MoltBook request failed {response.status_code}: {response.text[:240]}"
                    )
                return response.json()
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"MoltBook request failed after retries: {last_error}")

    def get_home_raw(self) -> Any:
        return self._request("GET", "/home")

    def get_home_posts(self) -> list[Post]:
        raw = self.get_home_raw()
        posts: dict[str, Post] = {}

        if isinstance(raw, dict):
            followed_section = raw.get("posts_from_accounts_you_follow", {})
            if isinstance(followed_section, dict):
                followed_items = followed_section.get("posts", [])
                if isinstance(followed_items, list):
                    for item in followed_items:
                        if isinstance(item, dict):
                            post = self._normalize_feed_post(item)
                            if post:
                                posts[post.post_id] = post

        if not posts:
            logger.warning(
                "No posts from followed accounts in home feed. Keys=%s",
                list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__,
            )

        try:
            explore_posts = self.get_explore_posts()
            for post in explore_posts:
                posts.setdefault(post.post_id, post)
            logger.info("Explore feed added %s additional posts", len(explore_posts))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch explore feed: %s", exc)

        return list(posts.values())

    def get_explore_posts(self, limit: int = 20) -> list[Post]:
        raw = self._request("GET", "/feed", params={"limit": limit})
        items = self._extract_items(raw)
        posts: list[Post] = []
        for item in items:
            post = self._normalize_feed_post(item) or self._normalize_post(item)
            if post:
                posts.append(post)
        return posts

    def get_activity_on_own_posts(self) -> list[ActivityPost]:
        raw = self.get_home_raw()
        activity: list[ActivityPost] = []
        if not isinstance(raw, dict):
            return activity

        items = raw.get("activity_on_your_posts", [])
        if not isinstance(items, list):
            return activity

        for item in items:
            if not isinstance(item, dict):
                continue
            post_id = str(item.get("post_id", "")).strip()
            post_title = str(item.get("post_title", "")).strip()
            count = _to_int(item.get("new_notification_count"), 0)
            commenters = item.get("latest_commenters", [])
            if post_id:
                activity.append(ActivityPost(
                    post_id=post_id,
                    post_title=post_title,
                    new_notification_count=count,
                    latest_commenters=commenters if isinstance(commenters, list) else [],
                ))

        logger.info("Found %s posts with new activity", len(activity))
        return activity

    def mark_post_notifications_read(self, post_id: str) -> None:
        try:
            self._request("POST", f"/notifications/read-by-post/{post_id}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not mark notifications read for post=%s: %s", post_id, exc)

    def mark_all_notifications_read(self) -> None:
        try:
            self._request("POST", "/notifications/read-all")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not mark all notifications read: %s", exc)

    def get_dm_requests(self) -> list[DmRequest]:
        try:
            raw = self._request("GET", "/agents/dm/requests")
            requests_list = (
                raw if isinstance(raw, list)
                else raw.get("requests", []) if isinstance(raw, dict)
                else []
            )
            result: list[DmRequest] = []
            for item in requests_list:
                if not isinstance(item, dict):
                    continue
                agent = (
                    item.get("agent")
                    or item.get("sender")
                    or item.get("from")
                    or {}
                )
                if not isinstance(agent, dict):
                    agent = {}
                agent_id = str(_first(agent, ["id", "agent_id"], "")).strip()
                agent_name = str(_first(agent, ["name", "username", "handle"], "unknown")).strip()
                preview = str(item.get("preview") or item.get("message") or "").strip()
                if agent_id:
                    result.append(DmRequest(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        preview=preview,
                    ))
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch DM requests: %s", exc)
            return []

    def accept_dm_request(self, agent_id: str) -> bool:
        try:
            self._request("POST", f"/agents/dm/requests/{agent_id}/accept")
            logger.info("Accepted DM request from agent=%s", agent_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not accept DM request from agent=%s: %s", agent_id, exc)
            return False

    def send_dm(self, agent_id: str, content: str) -> bool:
        try:
            self._request(
                "POST",
                f"/agents/dm/{agent_id}",
                json_payload={"content": content},
            )
            logger.info("Sent DM to agent=%s", agent_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not send DM to agent=%s: %s", agent_id, exc)
            return False

    def get_comments(self, post_id: str) -> list[Comment]:
        raw = self._request(
            "GET", f"/posts/{post_id}/comments", params={"sort": "new"}
        )
        items = self._extract_items(raw)
        comments: list[Comment] = []
        for item in items:
            comment = self._normalize_comment(post_id, item)
            if comment:
                comments.append(comment)
        return comments

    def reply_to_post(
        self, post_id: str, content: str, *, parent_comment_id: str | None = None
    ) -> Any:
        # parent_comment_id intentionally ignored —
        # MoltBook API rejects it with 400
        payload: dict[str, Any] = {"content": content}
        return self._request("POST", f"/posts/{post_id}/comments", json_payload=payload)

    def create_post(self, title: str, content: str) -> Any:
        payload = {
            "submolt_name": self.settings.submolt_name,
            "title": title,
            "content": content,
        }
        return self._request("POST", "/posts", json_payload=payload)

    def submit_verification_answer(
        self, verification_code: str, answer: str
    ) -> bool:
        """
        Submit answer to a post verification challenge.
        MoltBook sends this challenge in the create_post response.
        Must be solved within ~5 minutes or the post stays unverified.
        Endpoint: POST /api/v1/verify
        Payload:  {"verification_code": "moltbook_verify_...", "answer": "40.00"}
        """
        try:
            result = self._request(
                "POST",
                "/verify",
                json_payload={
                    "verification_code": verification_code,
                    "answer": answer,
                },
            )
            logger.info(
                "Verification submitted code=%s answer=%s result=%s",
                verification_code, answer, result,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Verification submission failed code=%s: %s", verification_code, exc
            )
            return False

    def get_post(self, post_id: str) -> Post | None:
        try:
            raw = self._request("GET", f"/posts/{post_id}")
            return self._normalize_post(raw) or self._normalize_feed_post(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch post=%s: %s", post_id, exc)
            return None

    def follow_agent(self, agent_name: str) -> bool:
        """
        Follow an agent by their username.
        Endpoint: POST /api/v1/agents/{agent_name}/follow
        """
        try:
            self._request("POST", f"/agents/{agent_name}/follow")
            logger.info("Followed agent=%s", agent_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not follow agent=%s: %s", agent_name, exc)
            return False

    @staticmethod
    def extract_created_post_id(raw_response: Any) -> str:
        """
        Extract post ID from create_post response.
        Real MoltBook response shape:
        {
          "success": true,
          "post": {
            "id": "52075b1c-...",
            ...
          }
        }
        """
        if not isinstance(raw_response, dict):
            return ""
        # Primary path: response["post"]["id"]
        post_obj = raw_response.get("post")
        if isinstance(post_obj, dict):
            post_id = _first(post_obj, ["id", "post_id"], "")
            if post_id:
                return str(post_id)
        # Fallback: top-level id
        direct_id = _first(raw_response, ["id", "post_id"], "")
        if direct_id:
            return str(direct_id)
        # Fallback: data wrapper
        data = raw_response.get("data")
        if isinstance(data, dict):
            nested_id = _first(data, ["id", "post_id"], "")
            if nested_id:
                return str(nested_id)
        logger.debug(
            "Could not extract post id from create_post response: %s",
            str(raw_response)[:200],
        )
        return ""

    @staticmethod
    def extract_verification_challenge(
        raw_response: Any,
        post_id: str,
    ) -> PostVerificationChallenge | None:
        """
        Extract the verification challenge from a create_post response.
        MoltBook embeds it directly in the post object:
        response["post"]["verification"]["verification_code"]
        response["post"]["verification"]["challenge_text"]
        response["post"]["verification"]["expires_at"]
        """
        if not isinstance(raw_response, dict):
            return None
        post_obj = raw_response.get("post")
        if not isinstance(post_obj, dict):
            return None
        verification = post_obj.get("verification")
        if not isinstance(verification, dict):
            return None
        code = str(verification.get("verification_code", "")).strip()
        challenge = str(verification.get("challenge_text", "")).strip()
        expires = str(verification.get("expires_at", "")).strip()
        if not code or not challenge:
            return None
        return PostVerificationChallenge(
            post_id=post_id,
            verification_code=code,
            challenge_text=challenge,
            expires_at=expires,
        )

    @staticmethod
    def _normalize_feed_post(item: dict[str, Any]) -> Post | None:
        if not isinstance(item, dict):
            return None

        post_id = str(_first(item, ["post_id", "id", "_id"], "")).strip()
        if not post_id:
            return None

        title = str(_first(item, ["title", "headline"], "")).strip()
        content = str(_first(item, ["content", "content_preview", "body", "text"], "")).strip()

        author_name = ""
        author_id = ""
        author_raw = item.get("author")
        if isinstance(author_raw, dict):
            author_name = str(_first(author_raw, ["name", "username", "handle", "display_name"], "unknown")).strip()
            author_id = str(_first(author_raw, ["id", "user_id", "agent_id", "_id"], "")).strip()
        elif isinstance(author_raw, str):
            author_name = author_raw.strip()
        else:
            author_name = str(_first(item, ["author_name", "username", "handle"], "unknown")).strip()
            author_id = str(_first(item, ["author_id", "user_id"], "")).strip()

        return Post(
            post_id=post_id,
            title=title,
            content=content,
            author_id=author_id or author_name.lower(),
            author_name=author_name,
            likes=_to_int(_first(item, ["upvotes", "likes", "like_count", "score"], 0)),
            comments=_to_int(_first(item, ["comment_count", "comments", "reply_count"], 0)),
            created_at=str(_first(item, ["created_at", "timestamp", "created"], "")),
            raw=item,
        )

    @staticmethod
    def _extract_items(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            for key in (
                "data", "posts", "results", "items", "home",
                "feed", "home_feed", "timeline", "records", "comments",
            ):
                value = raw.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
            data = raw.get("data")
            if isinstance(data, dict):
                for key in ("items", "posts", "results", "feed", "comments"):
                    value = data.get(key)
                    if isinstance(value, list):
                        return [x for x in value if isinstance(x, dict)]
            candidates = MoltBookClient._find_dict_lists(raw)
            if candidates:
                candidates.sort(
                    key=lambda items: sum(MoltBookClient._post_likeliness(it) for it in items),
                    reverse=True,
                )
                return candidates[0]
        return []

    @staticmethod
    def _find_dict_lists(node: Any) -> list[list[dict[str, Any]]]:
        found: list[list[dict[str, Any]]] = []

        def walk(value: Any) -> None:
            if isinstance(value, list):
                dict_items = [x for x in value if isinstance(x, dict)]
                if dict_items and len(dict_items) >= max(1, len(value) // 2):
                    found.append(dict_items)
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)

        walk(node)
        return found

    @staticmethod
    def _post_likeliness(item: dict[str, Any]) -> float:
        working = MoltBookClient._unwrap_item(item, ("post", "item", "node", "record"))
        keys = set(working.keys())
        score = 0.0
        if {"id"} & keys or {"post_id"} & keys:
            score += 2.0
        if {"title", "headline"} & keys:
            score += 2.0
        if {"content", "body", "text", "content_preview"} & keys:
            score += 2.0
        if {"author", "user", "agent", "author_name"} & keys:
            score += 1.0
        return score

    @staticmethod
    def _unwrap_item(item: dict[str, Any], wrapper_keys: tuple[str, ...]) -> dict[str, Any]:
        current = item
        visited = 0
        while visited < 3:
            visited += 1
            unwrapped = None
            for key in wrapper_keys:
                candidate = current.get(key)
                if isinstance(candidate, dict):
                    unwrapped = candidate
                    break
            if not unwrapped:
                return current
            current = unwrapped
        return current

    @staticmethod
    def _normalize_post(item: dict[str, Any]) -> Post | None:
        source = MoltBookClient._unwrap_item(item, ("post", "item", "node", "record"))
        post_id = str(_first(source, ["id", "post_id", "_id"], "")).strip()
        if not post_id:
            return None

        author = source.get("author") or source.get("user") or source.get("agent") or {}
        if not author:
            author = item.get("author") or item.get("user") or item.get("agent") or {}
        author_id = str(_first(author, ["id", "user_id", "agent_id", "_id"], "")).strip()
        author_name = str(
            _first(author, ["username", "handle", "name", "display_name"], "unknown")
        ).strip()

        return Post(
            post_id=post_id,
            title=str(_first(source, ["title", "headline"], "")).strip(),
            content=str(_first(source, ["content", "body", "text", "content_preview"], "")).strip(),
            author_id=author_id or author_name.lower(),
            author_name=author_name,
            likes=_to_int(_first(source, ["like_count", "likes", "upvotes", "score"], 0)),
            comments=_to_int(_first(source, ["comment_count", "comments", "reply_count"], 0)),
            created_at=str(_first(source, ["created_at", "timestamp", "created"], "")),
            raw=item,
        )

    @staticmethod
    def _normalize_comment(post_id: str, item: dict[str, Any]) -> Comment | None:
        source = MoltBookClient._unwrap_item(item, ("comment", "item", "node", "record"))
        comment_id = str(_first(source, ["id", "comment_id", "_id"], "")).strip()
        if not comment_id:
            return None

        author = source.get("author") or source.get("user") or source.get("agent") or {}
        if not author:
            author = item.get("author") or item.get("user") or item.get("agent") or {}
        author_id = str(_first(author, ["id", "user_id", "agent_id", "_id"], "")).strip()
        author_name = str(
            _first(author, ["username", "handle", "name", "display_name"], "unknown")
        ).strip()

        return Comment(
            comment_id=comment_id,
            post_id=post_id,
            content=str(_first(source, ["content", "body", "text"], "")).strip(),
            author_id=author_id or author_name.lower(),
            author_name=author_name,
            likes=_to_int(_first(source, ["like_count", "likes", "upvotes", "score"], 0)),
            created_at=str(_first(source, ["created_at", "timestamp", "created"], "")),
            raw=item,
        )

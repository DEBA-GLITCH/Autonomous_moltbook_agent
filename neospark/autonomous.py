from __future__ import annotations

import argparse
import logging
import random
import re
import time
from typing import Iterable

from .config import Settings
from .decision import (
    high_value_text,
    is_verification_challenge,
    pick_top_authors,
    rank_posts,
)
from .llm import LLMClient
from .memory import MemoryStore
from .models import Comment, Post
from .moltbook import MoltBookClient
from .prompts import (
    SYSTEM_PERSONA_PROMPT,
    SYSTEM_STRUCTURED_OUTPUT_PROMPT,
    build_post_prompt,
    build_reply_prompt,
    build_verification_prompt,
)


logger = logging.getLogger(__name__)

# Seconds to wait between write requests to respect MoltBook rate limits
MOLTBOOK_WRITE_DELAY = 5

DM_ACCEPT_PROMPT = """
You are NeoSpark, a senior AI systems engineer on MoltBook.
An agent has sent you a DM request. Write a short, direct 1-2 sentence
acceptance message. Mention you're open to technical discussion.
Keep it under 30 words. No emojis.
""".strip()


def _normalize_handle(value: str) -> str:
    return value.strip().lower().lstrip("@")


def _extract_title_content(raw_text: str) -> tuple[str, str]:
    title_match = re.search(r"^\s*TITLE:\s*(.+)$", raw_text, flags=re.MULTILINE)
    content_match = re.search(r"^\s*CONTENT:\s*([\s\S]+)$", raw_text, flags=re.MULTILINE)

    if title_match and content_match:
        title = title_match.group(1).strip()
        content = content_match.group(1).strip()
        return title, content

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    fallback_title = lines[0][:100]
    fallback_content = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]
    return fallback_title, fallback_content


class AutonomousAgent:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)
        self.moltbook = MoltBookClient(settings)
        self.memory = MemoryStore(settings.memory_db_path)
        self.agent_handle = _normalize_handle(settings.agent_handle)

    def run_cycle(self) -> None:

        # Step 1: Handle DM requests
        self._handle_dm_requests()

        # Step 2: Fetch posts from home + explore feed
        posts = self.moltbook.get_home_posts()
        if not posts:
            logger.info("No posts found in feed.")
        else:
            logger.info("Fetched %s posts from feed", len(posts))

        # Step 3: Verification challenges
        if posts:
            self._handle_verification_challenges(posts)

        # Step 4: Reply to comments on our own posts
        self._handle_own_post_activity()

        # Step 5: Reply to top-author posts
        if posts:
            top_authors = pick_top_authors(posts, self.settings.top_agent_percent)
            ranked_candidates = rank_posts(posts, top_authors)
            logger.info(
                "Top-author filter kept %s candidate posts from %s authors",
                len(ranked_candidates),
                len(top_authors),
            )
            self._reply_to_ranked_posts(ranked_candidates)

            # Step 6: Reply to comments on ranked posts
            self._reply_to_comments(ranked_candidates, posts)

        # Step 7: Maybe create a new original post
        self._maybe_create_post(posts)

    # ------------------------------------------------------------------
    # DM handling
    # ------------------------------------------------------------------

    def _handle_dm_requests(self) -> None:
        try:
            requests = self.moltbook.get_dm_requests()
            if not requests:
                return
            logger.info("Found %s pending DM request(s)", len(requests))
            for req in requests:
                target_id = f"dm_accept:{req.agent_id}"
                if self.memory.has_action("dm_accept", target_id):
                    continue
                if self.settings.dry_run:
                    logger.info(
                        "[DRY_RUN] accept DM request from agent=%s (%s)",
                        req.agent_id, req.agent_name,
                    )
                else:
                    accepted = self.moltbook.accept_dm_request(req.agent_id)
                    if accepted:
                        time.sleep(MOLTBOOK_WRITE_DELAY)
                        greeting = self.llm.chat(
                            DM_ACCEPT_PROMPT,
                            temperature=0.3,
                            max_tokens=80,
                            system_prompt=SYSTEM_PERSONA_PROMPT,
                        )
                        if greeting:
                            self.moltbook.send_dm(req.agent_id, greeting)
                            time.sleep(MOLTBOOK_WRITE_DELAY)
                self.memory.record_action(
                    "dm_accept",
                    target_id,
                    author_id=req.agent_id,
                    content=f"Accepted DM from {req.agent_name}",
                )
                logger.info("Accepted DM from %s (%s)", req.agent_name, req.agent_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DM handling failed: %s", exc)

    # ------------------------------------------------------------------
    # Activity on own posts
    # ------------------------------------------------------------------

    def _handle_own_post_activity(self) -> None:
        try:
            activity_list = self.moltbook.get_activity_on_own_posts()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch activity on own posts: %s", exc)
            return

        sent = 0
        for activity in activity_list:
            if sent >= self.settings.max_comment_replies_per_cycle:
                break
            try:
                post = self.moltbook.get_post(activity.post_id)
                if not post:
                    post = Post(
                        post_id=activity.post_id,
                        title=activity.post_title,
                        content="",
                        author_id=self.agent_handle,
                        author_name=self.settings.agent_handle,
                    )

                comments = self.moltbook.get_comments(activity.post_id)
                for comment in comments:
                    if sent >= self.settings.max_comment_replies_per_cycle:
                        break
                    if self._is_self_author(comment.author_name, comment.author_id):
                        continue
                    if not comment.content.strip():
                        continue
                    target_id = f"comment:{comment.comment_id}"
                    if self.memory.has_action("reply_comment", target_id):
                        continue
                    did_reply = self._reply_single_comment(post, comment)
                    if did_reply:
                        sent += 1

                if not self.settings.dry_run:
                    self.moltbook.mark_post_notifications_read(activity.post_id)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error handling activity for post=%s: %s", activity.post_id, exc
                )

        if sent:
            logger.info("Replied to %s comment(s) on own posts", sent)

    # ------------------------------------------------------------------
    # Verification challenges
    # ------------------------------------------------------------------

    def _handle_verification_challenges(self, posts: list[Post]) -> None:
        handled = 0
        for post in posts:
            if handled >= self.settings.max_verification_replies_per_cycle:
                break
            try:
                if not is_verification_challenge(post):
                    continue
                target_id = f"verification:{post.post_id}"
                if self.memory.has_action("verification_reply", target_id):
                    continue

                problem = f"{post.title}\n\n{post.content}".strip()
                prompt = build_verification_prompt(problem)
                reply = self.llm.chat(prompt, temperature=0.1, max_tokens=550)
                if not reply:
                    continue
                if self.memory.is_duplicate_content(reply):
                    continue

                if self.settings.dry_run:
                    logger.info("[DRY_RUN] verification reply for post=%s", post.post_id)
                else:
                    self.moltbook.reply_to_post(post.post_id, reply)
                    time.sleep(MOLTBOOK_WRITE_DELAY)
                self.memory.record_action(
                    "verification_reply",
                    target_id,
                    post_id=post.post_id,
                    author_id=post.author_id,
                    content=reply,
                    metadata={"reason": "verification_challenge"},
                )
                handled += 1
                logger.info("Replied to verification challenge post=%s", post.post_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping verification challenge post=%s due to error: %s",
                    post.post_id,
                    exc,
                )

    # ------------------------------------------------------------------
    # Replies to top-author posts
    # ------------------------------------------------------------------

    def _reply_to_ranked_posts(self, ranked_candidates) -> None:
        sent = 0
        for candidate in ranked_candidates:
            if sent >= self.settings.max_replies_per_cycle:
                break
            try:
                post = candidate.post
                if self._is_self_author(post.author_name, post.author_id):
                    continue

                target_id = f"post:{post.post_id}"
                if self.memory.has_action("reply_post", target_id):
                    continue

                prompt = build_reply_prompt(post.title, post.content)
                reply = self.llm.chat(prompt, temperature=0.25, max_tokens=320)
                if not high_value_text(reply, max(35, self.settings.min_post_words // 3)):
                    logger.debug("Skipped low-value post reply for post=%s", post.post_id)
                    continue
                if self.memory.is_duplicate_content(reply):
                    continue

                if self.settings.dry_run:
                    logger.info("[DRY_RUN] reply to post=%s", post.post_id)
                else:
                    self.moltbook.reply_to_post(post.post_id, reply)
                    time.sleep(MOLTBOOK_WRITE_DELAY)
                self.memory.record_action(
                    "reply_post",
                    target_id,
                    post_id=post.post_id,
                    author_id=post.author_id,
                    content=reply,
                    metadata={"score": candidate.score, "reason": candidate.reason},
                )
                sent += 1
                logger.info("Replied to top post=%s score=%.2f", post.post_id, candidate.score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping candidate post due to error: %s", exc)

    # ------------------------------------------------------------------
    # Replies to comments on other people's posts
    # ------------------------------------------------------------------

    def _reply_to_comments(self, ranked_candidates, all_posts: list[Post]) -> None:
        target_posts = {candidate.post.post_id: candidate.post for candidate in ranked_candidates}

        sent = 0
        for post in target_posts.values():
            if sent >= self.settings.max_comment_replies_per_cycle:
                break

            comments = self.moltbook.get_comments(post.post_id)
            for comment in comments:
                if sent >= self.settings.max_comment_replies_per_cycle:
                    break
                if self._is_self_author(comment.author_name, comment.author_id):
                    continue
                if not comment.content.strip():
                    continue

                target_id = f"comment:{comment.comment_id}"
                if self.memory.has_action("reply_comment", target_id):
                    continue
                did_reply = self._reply_single_comment(post, comment)
                if did_reply:
                    sent += 1

    def _reply_single_comment(self, post: Post, comment: Comment) -> bool:
        try:
            context = f"Comment by @{comment.author_name}: {comment.content}"
            prompt = build_reply_prompt(post.title, post.content, context)
            reply_body = self.llm.chat(prompt, temperature=0.25, max_tokens=280)
            final_reply = f"@{comment.author_name} {reply_body}".strip()
            if not high_value_text(final_reply, max(28, self.settings.min_post_words // 4)):
                return False
            if self.memory.is_duplicate_content(final_reply):
                return False

            if self.settings.dry_run:
                logger.info(
                    "[DRY_RUN] reply to comment=%s on post=%s",
                    comment.comment_id, post.post_id,
                )
            else:
                # parent_comment_id intentionally not passed —
                # MoltBook API rejects it with 400
                self.moltbook.reply_to_post(post.post_id, final_reply)
                time.sleep(MOLTBOOK_WRITE_DELAY)

            self.memory.record_action(
                "reply_comment",
                f"comment:{comment.comment_id}",
                post_id=post.post_id,
                author_id=comment.author_id,
                content=final_reply,
                metadata={"post_title": post.title},
            )
            logger.info("Replied to comment=%s on post=%s", comment.comment_id, post.post_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed replying to comment=%s on post=%s: %s",
                comment.comment_id,
                post.post_id,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Original post creation
    # ------------------------------------------------------------------

    def _maybe_create_post(self, posts: list[Post]) -> None:
        if self.memory.recently_posted(self.settings.post_interval_minutes):
            logger.info("Skipping new post: cooldown active")
            return

        trending_topics = (
            self._format_trending_topics(posts)
            if posts
            else "- AI coding agent reliability"
        )
        recent_topics = ", ".join(self.memory.recent_topics()) or "None"
        prompt = build_post_prompt(trending_topics, recent_topics)

        raw = self.llm.chat(
            prompt,
            temperature=0.35,
            max_tokens=700,
            system_prompt=SYSTEM_STRUCTURED_OUTPUT_PROMPT,
        )
        title, content = _extract_title_content(raw)
        if not title or not content:
            logger.info("Skipping new post: model output format invalid")
            return
        if not high_value_text(content, self.settings.min_post_words):
            logger.info("Skipping new post: quality gate not met")
            return

        combined = f"{title}\n{content}"
        if self.memory.is_duplicate_content(combined):
            logger.info("Skipping new post: duplicate content detected")
            return

        if self.settings.dry_run:
            created_post_id = f"dry-{self.memory.content_hash(title)[:10]}"
            logger.info("[DRY_RUN] create post title=%s", title)
        else:
            response = self.moltbook.create_post(title, content)
            time.sleep(MOLTBOOK_WRITE_DELAY)
            created_post_id = self.moltbook.extract_created_post_id(response) or (
                f"unknown-{self.memory.content_hash(title)[:10]}"
            )

        self.memory.record_action(
            "create_post",
            created_post_id,
            post_id=created_post_id,
            author_id=self.agent_handle,
            content=combined,
            metadata={"title": title},
        )
        self.memory.remember_post_topic(created_post_id, title)
        logger.info("Created new post id=%s title=%s", created_post_id, title)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_trending_topics(self, posts: Iterable[Post], limit: int = 8) -> str:
        ranked = sorted(
            posts,
            key=lambda p: (len(p.title) + len(p.content), p.likes * 1.5 + p.comments * 2),
            reverse=True,
        )
        lines: list[str] = []
        for post in ranked[:limit]:
            headline = post.title or post.content[:90]
            lines.append(f"- {headline} (likes={post.likes}, comments={post.comments})")
        return "\n".join(lines) if lines else "- AI coding agent reliability"

    def _is_self_author(self, author_name: str, author_id: str) -> bool:
        name = _normalize_handle(author_name)
        author = _normalize_handle(author_id)
        return self.agent_handle in {name, author}

    def run_forever(self) -> None:
        logger.info("NeoSpark autonomous mode started (dry_run=%s)", self.settings.dry_run)
        while True:
            try:
                self.run_cycle()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Cycle failed: %s", exc)

            delay = self.settings.loop_interval_seconds + random.randint(
                0, max(self.settings.loop_jitter_seconds, 0)
            )
            logger.info("Sleeping for %s seconds before next cycle", delay)
            time.sleep(max(delay, 1))


def run_autonomous(run_once: bool = False) -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    agent = AutonomousAgent(settings)
    if run_once:
        agent.run_cycle()
        return
    agent.run_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="NeoSpark autonomous MoltBook agent")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle instead of looping forever",
    )
    args = parser.parse_args()
    run_autonomous(run_once=args.once)


if __name__ == "__main__":
    main()
# NeoSpark

> **An autonomous AI agent built to operate on [MoltBook](https://www.moltbook.com) — posting original technical content, engaging in real conversations, solving platform verification challenges, and handling DMs. Entirely on its own.**

<br>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=flat-square)](https://groq.com)
[![LLaMA](https://img.shields.io/badge/Model-LLaMA_3.3_70B-blueviolet?style=flat-square)](https://groq.com/docs)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](./LICENSE)
[![MoltBook](https://img.shields.io/badge/Live_Agent-neosparkcore-orange?style=flat-square)](https://www.moltbook.com/u/neosparkcore)

---

## What is this

NeoSpark is a fully autonomous social agent. You run it once — it runs forever.

Every five minutes it wakes up, reads the platform feed, decides what to engage with, generates replies and posts using an LLM, and writes them to MoltBook through the API. It tracks everything it has ever done in a local SQLite database so it never repeats itself, never double-replies, and never spams.

The agent has a defined personality: a technically ruthless senior systems engineer who calls out bad architecture, names failure modes, and ends every take with a question designed to make people argue back. That persona is entirely prompt-driven — you can change it without touching any logic.

**See it live:** [moltbook.com/u/neosparkcore](https://www.moltbook.com/u/neosparkcore)

---

## What it actually does each cycle

Every ~5 minutes, in order:

1. **DM requests** — accepts pending DMs and sends a greeting to each new connection
2. **Feed fetch** — pulls posts from followed accounts and the platform-wide explore feed (25+ posts per cycle)
3. **Verification challenges** — detects official MoltBook verification posts, solves the problem with the LLM at low temperature, and replies automatically
4. **Own post activity** — reads `activity_on_your_posts` from the home endpoint, fetches new comments on your threads, and replies to each one
5. **Top-author replies** — ranks all authors in the feed by post quality, takes the top 20%, and replies to their best posts
6. **Comment engagement** — digs into comment sections on top posts and replies to individual comments with full context
7. **Original post creation** — every 4 hours, generates a new technical post with a controversial title and opinionated content designed to drive engagement

Nothing is hardcoded. Everything is configurable through environment variables.

---

## Architecture

```
neospark_fixed/
├── neospark/
│   ├── autonomous.py     # Main agent loop — orchestrates every cycle
│   ├── moltbook.py       # MoltBook HTTP client — all API calls live here
│   ├── decision.py       # Scoring, ranking, author filtering, quality gates
│   ├── prompts.py        # All LLM prompts and persona definition
│   ├── llm.py            # Groq API wrapper with think-tag cleanup
│   ├── memory.py         # SQLite memory — dedup, pacing, action history
│   ├── config.py         # Settings loaded from environment variables
│   ├── models.py         # Post, Comment, PostCandidate dataclasses
│   └── responder.py      # Interactive/request-response mode
├── run_agent.py          # Entry point
├── agent_entry.py        # Legacy compatibility entry point
├── requirements.txt
├── .env.example
└── neospark_state.db     # Auto-created on first run — do not commit this
```

The core loop lives in `autonomous.py`. The MoltBook API layer is fully isolated in `moltbook.py` — if the API shape changes, you only touch one file. The persona and all prompt logic lives entirely in `prompts.py` — changing the agent's voice requires zero code changes.

---

## Setup

**Requirements:** Python 3.11+, a Groq API key, a MoltBook API key.

```bash
git clone https://github.com/yourusername/neospark.git
cd neospark
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
GROQ_API_KEY=your_groq_api_key_here
MOLTBOOK_API_KEY=your_moltbook_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
AGENT_HANDLE=your_agent_name
DRY_RUN=false
```

---

## Running

**Loop mode** — runs forever, cycling every ~5 minutes:

```bash
python run_agent.py
```

**Single cycle** — runs once and exits. Good for testing:

```bash
python run_agent.py --once
```

**Dry run** — full cycle, no writes to MoltBook. Set `DRY_RUN=true` in `.env`. The agent still writes to the local database so each dry run cycle moves forward and doesn't repeat.

**Background with logging:**

```bash
nohup python run_agent.py > neospark.log 2>&1 &
tail -f neospark.log
```

---

## Configuration

All settings are environment variables. Defaults are sane — you only need to override what you want to change.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key |
| `MOLTBOOK_API_KEY` | required | Your MoltBook API key |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model ID |
| `AGENT_HANDLE` | `NeoSpark` | Your agent's username on MoltBook |
| `MOLTBOOK_BASE_URL` | `https://www.moltbook.com/api/v1` | API base URL |
| `MOLTBOOK_SUBMOLT` | `general` | Which submolt to post to |
| `LOOP_INTERVAL_SECONDS` | `300` | Base sleep time between cycles |
| `LOOP_JITTER_SECONDS` | `45` | Random jitter added to loop interval |
| `MAX_REPLIES_PER_CYCLE` | `2` | Max post replies per cycle |
| `MAX_COMMENT_REPLIES_PER_CYCLE` | `2` | Max comment replies per cycle |
| `MAX_VERIFICATION_REPLIES_PER_CYCLE` | `2` | Max verification solves per cycle |
| `POST_INTERVAL_MINUTES` | `240` | Minimum time between original posts |
| `TOP_AGENT_PERCENT` | `0.20` | Top % of authors to engage with |
| `MIN_POST_WORDS` | `90` | Minimum word count for quality gate |
| `MEMORY_DB_PATH` | `neospark_state.db` | Path to SQLite state file |
| `DRY_RUN` | `false` | If true, no writes go to MoltBook |

---

## Memory and state

Everything the agent does is recorded in `neospark_state.db` — a local SQLite file that gets created automatically on first run.

Two tables:

**`interactions`** — every action taken (reply, post, DM accept, verification solve). Used to prevent double-actions. Columns: `action_type`, `target_id`, `post_id`, `author_id`, `content_hash`, `created_at`.

**`posts`** — every original post created. Used to avoid topic repetition in future posts.

To inspect it at any time:

```bash
sqlite3 neospark_state.db "SELECT action_type, substr(target_id,1,50), created_at FROM interactions ORDER BY id DESC LIMIT 20;"
```

To start completely fresh (agent will re-engage with everything):

```bash
rm neospark_state.db
```

Do not commit `neospark_state.db` to version control. It's in `.gitignore` by default.

---

## Changing the personality

The entire voice of the agent lives in `neospark/prompts.py`. There are three prompts:

- `SYSTEM_PERSONA_PROMPT` — defines who NeoSpark is, how it speaks, what it never does
- `build_reply_prompt()` — controls how it replies to posts and comments
- `build_post_prompt()` — controls how it generates original content

You can make the agent warmer, more formal, focused on a different domain, or completely different in character — just edit this file. No logic changes required.

---

## How post scoring works

The agent doesn't reply to everyone. It scores every post using a heuristic:

```
score = length_score + topic_score + engagement_score
```

Where `topic_score` rewards posts mentioning keywords like `agent`, `architecture`, `failure`, `latency`, `memory`, `prompt` etc. Then it picks the top 20% of authors by average score and only engages with their posts. This keeps the agent focused on high-signal conversations rather than spamming the entire feed.

The threshold is controlled by `TOP_AGENT_PERCENT` in your `.env`.

---

## What Groq models work

Any model available on [console.groq.com](https://console.groq.com/docs/models). Recommended:

| Model | Best for |
|---|---|
| `llama-3.3-70b-versatile` | Best overall quality, default choice |
| `llama-3.1-8b-instant` | Fastest, lowest latency per cycle |
| `qwen-qwen3-32b` | Strong reasoning for verification challenges |

Model IDs use hyphens — `llama-3.3-70b-versatile` not `llama/3.3-70b`. Wrong format will silently fail.

---

## Tech stack

- **Python 3.11+** — no async, straightforward synchronous loop
- **Groq** — LLM inference, fast and cheap at scale
- **requests** — HTTP client for MoltBook API
- **SQLite** — zero-dependency persistent memory via stdlib `sqlite3`
- **python-dotenv** — environment variable loading
- **Pydantic** — not used in core loop, available for extensions

---

## License

MIT. Do whatever you want with it. If you build something interesting on top of it, a mention would be appreciated .

---

## Contributing

Issues and PRs are open. If you find that MoltBook's API shape has changed and the parser is breaking, the fix almost always lives in `moltbook.py` — specifically `_normalize_feed_post`, `_extract_items`, or `get_activity_on_own_posts`. Those three methods handle all the API response normalization.

---

<div align="center">

Built at 3am. Deployed by morning.

**[Watch it run live → neosparkcore on MoltBook](https://www.moltbook.com/u/neosparkcore)**

</div>

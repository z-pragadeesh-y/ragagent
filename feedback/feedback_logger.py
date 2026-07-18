"""
Plan 2 Step 10 (roadmap) / Plan 2 Step 5 (local numbering): Feedback logging.

Best-effort, fire-and-forget SQLite logging of live production traffic - NOT
the golden-set evaluation from Plan 1 Step 4, this captures what real users
actually asked and got, so live performance can be reviewed later (and a
future thumbs-up/down UI could attach to these rows by id without
re-plumbing).

Deliberately a plain module, not a graph node: this needs to run after
update_history with zero effect on graph state/routing. Called from
api/main.py in a background thread (see main.py's hook) so a logging write
can never add latency to, or block, the actual response returned to the user.

NAMING NOTE: intentionally placed under feedback/, not logging/ (even though
the task spec used "logging/feedback_logger.py" as an example path) - a
top-level package literally named `logging` would shadow Python's own
stdlib logging module, which this entire project already imports everywhere
(`import logging`). That would break every existing `import logging`
statement in the codebase depending on import order. feedback/ avoids the
collision while keeping the same intent.

Uses its own feedback.sqlite file, separate from checkpoints.sqlite (which
is LangGraph's SqliteSaver-owned file for conversation state - this is
unrelated application data and shouldn't be grafted onto it).
"""
import json
import logging
import sqlite3
import time

logger = logging.getLogger("llm_manager")

DB_PATH = "feedback.sqlite"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    thread_id TEXT,
    question TEXT,
    answer TEXT,
    citations TEXT,
    retrieved_sources TEXT,
    route_category TEXT,
    is_injection INTEGER,
    scope_flagged INTEGER
)
"""


def log_feedback(
    thread_id: str,
    question: str,
    answer: str,
    citations: list,
    retrieved_sources: list,
    route_category: str,
    is_injection: bool,
    scope_flagged: bool,
) -> None:
    """
    Best-effort insert of one query/answer/guardrail-flags row. Never raises -
    any failure (DB locked, disk full, malformed data, etc.) is logged as a
    warning and swallowed, since logging must never be able to take down or
    delay a real user-facing response. Call this from a background thread
    (see api/main.py) if you also want to guarantee it adds zero latency to
    the response path, not just zero risk of crashing it.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute(CREATE_TABLE_SQL)
        conn.execute(
            """
            INSERT INTO feedback_log
                (timestamp, thread_id, question, answer, citations,
                 retrieved_sources, route_category, is_injection, scope_flagged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                thread_id,
                question,
                answer,
                json.dumps(citations or []),
                json.dumps(retrieved_sources or []),
                route_category,
                int(bool(is_injection)),
                int(bool(scope_flagged)),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.warning("feedback_logger: failed to log feedback row (non-fatal)", exc_info=True)

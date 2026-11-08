"""Durable dashboard transcript and immutable conversation snapshots."""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from core.embeddings import embed


def _chunks(events):
    text = "\n".join(
        f"{event.get('role')}: {event.get('content', '')}"
        for event in events
        if event.get("role") in {"user", "assistant"} and event.get("source") != "startup"
    )
    return [text[pos:pos + 1200] for pos in range(0, len(text), 1000)]


class ConversationStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS active_events (id INTEGER PRIMARY KEY, event TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active_context (id INTEGER PRIMARY KEY CHECK (id = 1), context TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, created REAL NOT NULL, events TEXT NOT NULL, context TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS search_chunks (snapshot_id TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT, PRIMARY KEY(snapshot_id, ordinal), FOREIGN KEY(snapshot_id) REFERENCES snapshots(id) ON DELETE CASCADE);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    def active(self):
        with self._lock, self._connect() as db:
            events = [json.loads(row[0]) for row in db.execute("SELECT event FROM active_events ORDER BY id")]
            row = db.execute("SELECT context FROM active_context WHERE id = 1").fetchone()
            return events, json.loads(row[0]) if row else []

    def append(self, event, context=None):
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO active_events(event) VALUES (?)", (json.dumps(event),))
            if context is not None:
                db.execute("INSERT OR REPLACE INTO active_context(id, context) VALUES (1, ?)",
                           (json.dumps(context),))

    def clear(self):
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM active_events")
            db.execute("DELETE FROM active_context")

    def _save(self, db, context):
        events = [json.loads(row[0]) for row in db.execute("SELECT event FROM active_events ORDER BY id")]
        if not any(e.get("role") == "user" for e in events):
            raise ValueError("The current conversation is empty")
        identifier, created = str(uuid.uuid4()), time.time()
        db.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                   (identifier, created, json.dumps(events), json.dumps(context)))
        db.executemany("INSERT INTO search_chunks(snapshot_id, ordinal, text) VALUES (?, ?, ?)",
                       [(identifier, i, text) for i, text in enumerate(_chunks(events))])
        return {"id": identifier, "created_at": created}

    def save(self, context):
        with self._lock, self._connect() as db:
            return self._save(db, context)

    def list(self, offset=0, limit=20, query=""):
        if query:
            with self._lock, self._connect() as db:
                pending = db.execute("SELECT snapshot_id, ordinal, text FROM search_chunks WHERE embedding IS NULL").fetchall()
            # Model inference can be slow. Do not hold the transcript write lock
            # while embedding, so active conversations remain responsive.
            for start in range(0, len(pending), 32):
                batch = pending[start:start + 32]
                vectors = embed([row[2] for row in batch])
                with self._lock, self._connect() as db:
                    db.executemany("UPDATE search_chunks SET embedding = ? WHERE snapshot_id = ? AND ordinal = ?",
                                   [(json.dumps(vector.tolist()), row[0], row[1]) for row, vector in zip(batch, vectors, strict=True)])
            vector = embed([query], query=True)[0]
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT id, created, events FROM snapshots ORDER BY created DESC, id DESC").fetchall()
            if query:
                scores = {}
                for identifier, encoded in db.execute("SELECT snapshot_id, embedding FROM search_chunks WHERE embedding IS NOT NULL"):
                    score = float(np.dot(np.asarray(json.loads(encoded), dtype=np.float32), vector))
                    scores[identifier] = max(score, scores.get(identifier, -1.0))
                rows = sorted((row for row in rows if scores.get(row[0], -1) >= 0.25),
                              key=lambda row: (-scores[row[0]], -row[1]))
            total = len(rows)
            items = []
            for identifier, created, raw in rows[offset:offset + limit]:
                events = json.loads(raw)
                title = next((e.get("content", "") for e in events if e.get("role") == "user"), "Conversation")
                items.append({"id": identifier, "created_at": created, "title": str(title)[:100]})
            return {"conversations": items, "next_offset": offset + limit if offset + limit < total else None}

    def load(self, identifier, save_context=None):
        with self._lock, self._connect() as db:
            row = db.execute("SELECT events, context FROM snapshots WHERE id = ?", (identifier,)).fetchone()
            if row is None:
                return None
            events, context = json.loads(row[0]), json.loads(row[1])
            snapshot = self._save(db, save_context) if save_context is not None else None
            db.execute("DELETE FROM active_events")
            db.executemany("INSERT INTO active_events(event) VALUES (?)", [(json.dumps(event),) for event in events])
            db.execute("INSERT OR REPLACE INTO active_context(id, context) VALUES (1, ?)", (json.dumps(context),))
            return events, context, snapshot

    def delete(self, identifier):
        with self._lock, self._connect() as db:
            return db.execute("DELETE FROM snapshots WHERE id = ?", (identifier,)).rowcount > 0

"""Tier 1 Episodic Skill Bank, Client Node Registry, Trajectory Store, and Telemetry Database."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def compute_local_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic fast token/bigram feature hashing embedding (0 MB GPU overhead)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        return vec.tolist()

    features = list(tokens) + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / denom)


@dataclass
class SkillCard:
    id: Optional[int]
    title: str
    domain: str
    trigger_signature: str
    anti_pattern: str
    verified_pattern: str
    tags: list[str] = field(default_factory=list)
    times_retrieved: int = 0
    success_count: int = 0
    failure_count: int = 0
    similarity: float = 0.0
    active: bool = True

    @property
    def utility_score(self) -> float:
        return (self.success_count + 1.0) / (self.success_count + self.failure_count + 2.0)


@dataclass
class TaskTelemetry:
    task_id: str
    prompt: str
    domain: str
    passed_first_try: bool
    iterations_to_green: int
    patch_survival_rate: float
    skills_used: list[int]
    adapter_version: str
    tokens_used: int = 0
    client_node: str = "local-workstation"
    created_at: float = field(default_factory=time.time)


@dataclass
class ClientNode:
    node_id: str
    hostname: str
    ip_address: str
    status: str  # "online", "streaming", "paused"
    requests_count: int
    prompt_tokens: int
    completion_tokens: int
    last_tok_per_sec: float
    active_model: str
    last_seen: float


class MPDCStore:
    """SQLite + Vector store backing Episodic Skills, Client Output Nodes, Trajectories, and Telemetry."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodic_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    trigger_signature TEXT NOT NULL,
                    anti_pattern TEXT NOT NULL,
                    verified_pattern TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    times_retrieved INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    active INTEGER DEFAULT 1,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_telemetry (
                    task_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    passed_first_try INTEGER NOT NULL,
                    iterations_to_green INTEGER NOT NULL,
                    patch_survival_rate REAL NOT NULL,
                    skills_used_json TEXT NOT NULL,
                    adapter_version TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    client_node TEXT DEFAULT 'local-workstation',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS harvested_trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    rejected_response TEXT,
                    chosen_response TEXT NOT NULL,
                    patch_survival_rate REAL NOT NULL,
                    split TEXT NOT NULL,
                    used_in_adapter TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS adapter_registry (
                    version TEXT PRIMARY KEY,
                    base_model TEXT NOT NULL,
                    adapter_path TEXT NOT NULL,
                    replay_pass_rate REAL NOT NULL,
                    holdout_perplexity REAL NOT NULL,
                    sanity_score REAL NOT NULL,
                    promoted INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS client_nodes (
                    node_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'online',
                    requests_count INTEGER DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    last_tok_per_sec REAL DEFAULT 0.0,
                    active_model TEXT DEFAULT 'qwen3-mpdc',
                    last_seen REAL NOT NULL
                );
                """
            )

    def upsert_client_node(
        self,
        node_id: str,
        hostname: Optional[str] = None,
        ip_address: str = "127.0.0.1",
        prompt_tokens_delta: int = 0,
        completion_tokens_delta: int = 0,
        tok_per_sec: Optional[float] = None,
        active_model: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM client_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                final_status = status or "online"
                conn.execute(
                    """
                    INSERT INTO client_nodes (
                        node_id, hostname, ip_address, status, requests_count,
                        prompt_tokens, completion_tokens, last_tok_per_sec, active_model, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        hostname or node_id,
                        ip_address,
                        final_status,
                        1 if (prompt_tokens_delta or completion_tokens_delta) else 0,
                        prompt_tokens_delta,
                        completion_tokens_delta,
                        tok_per_sec or 0.0,
                        active_model or "qwen3-mpdc",
                        now,
                    ),
                )
            else:
                # Preserve 'paused' unless explicitly overridden
                current_status = row["status"]
                new_status = status if status is not None else current_status
                req_inc = 1 if (prompt_tokens_delta or completion_tokens_delta) else 0
                conn.execute(
                    """
                    UPDATE client_nodes
                    SET hostname = ?,
                        ip_address = ?,
                        status = ?,
                        requests_count = requests_count + ?,
                        prompt_tokens = prompt_tokens + ?,
                        completion_tokens = completion_tokens + ?,
                        last_tok_per_sec = ?,
                        active_model = ?,
                        last_seen = ?
                    WHERE node_id = ?
                    """,
                    (
                        hostname or row["hostname"],
                        ip_address or row["ip_address"],
                        new_status,
                        req_inc,
                        prompt_tokens_delta,
                        completion_tokens_delta,
                        tok_per_sec if tok_per_sec is not None else row["last_tok_per_sec"],
                        active_model or row["active_model"],
                        now,
                        node_id,
                    ),
                )
            updated = conn.execute(
                "SELECT * FROM client_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            return dict(updated)

    def get_client_node(self, node_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM client_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_client_nodes(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM client_nodes ORDER BY last_seen DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def toggle_client_node_pause(self, node_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM client_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if not row:
                return None
            new_status = "online" if row["status"] == "paused" else "paused"
            conn.execute(
                "UPDATE client_nodes SET status = ? WHERE node_id = ?",
                (new_status, node_id),
            )
            updated = conn.execute(
                "SELECT * FROM client_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            return dict(updated)

    def add_skill(
        self,
        title: str,
        domain: str,
        trigger_signature: str,
        anti_pattern: str,
        verified_pattern: str,
        tags: Optional[list[str]] = None,
    ) -> int:
        tags_list = tags or []
        embed_text = f"{title} {domain} {trigger_signature} {' '.join(tags_list)} {verified_pattern}"
        embedding = compute_local_embedding(embed_text)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodic_skills (
                    title, domain, trigger_signature, anti_pattern, verified_pattern,
                    tags_json, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    domain,
                    trigger_signature,
                    anti_pattern,
                    verified_pattern,
                    json.dumps(tags_list),
                    json.dumps(embedding),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def list_skills(self, include_inactive: bool = True) -> list[dict[str, Any]]:
        with self._connect() as conn:
            query = (
                "SELECT * FROM episodic_skills ORDER BY id DESC"
                if include_inactive
                else "SELECT * FROM episodic_skills WHERE active = 1 ORDER BY id DESC"
            )
            rows = conn.execute(query).fetchall()
        result = []
        for r in rows:
            s, f = r["success_count"], r["failure_count"]
            utility = round((s + 1.0) / (s + f + 2.0), 3)
            result.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "domain": r["domain"],
                    "trigger_signature": r["trigger_signature"],
                    "anti_pattern": r["anti_pattern"],
                    "verified_pattern": r["verified_pattern"],
                    "times_retrieved": r["times_retrieved"],
                    "success_count": s,
                    "failure_count": f,
                    "utility_score": utility,
                    "active": bool(r["active"]),
                }
            )
        return result

    def toggle_skill(self, skill_id: int) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodic_skills WHERE id = ?", (skill_id,)
            ).fetchone()
            if not row:
                return None
            new_active = 0 if row["active"] else 1
            conn.execute(
                "UPDATE episodic_skills SET active = ? WHERE id = ?",
                (new_active, skill_id),
            )
        skills = [s for s in self.list_skills(include_inactive=True) if s["id"] == skill_id]
        return skills[0] if skills else None

    def retrieve_skills(
        self, query: str, top_k: int = 4, min_similarity: float = 0.20
    ) -> list[SkillCard]:
        query_emb = compute_local_embedding(query)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodic_skills WHERE active = 1"
            ).fetchall()

        scored: list[SkillCard] = []
        for r in rows:
            emb = json.loads(r["embedding_json"])
            sim = cosine_similarity(query_emb, emb)
            if sim >= min_similarity:
                card = SkillCard(
                    id=r["id"],
                    title=r["title"],
                    domain=r["domain"],
                    trigger_signature=r["trigger_signature"],
                    anti_pattern=r["anti_pattern"],
                    verified_pattern=r["verified_pattern"],
                    tags=json.loads(r["tags_json"]),
                    times_retrieved=r["times_retrieved"],
                    success_count=r["success_count"],
                    failure_count=r["failure_count"],
                    similarity=sim,
                    active=bool(r["active"]),
                )
                scored.append(card)

        scored.sort(key=lambda c: c.similarity * c.utility_score, reverse=True)
        selected = scored[:top_k]

        if selected:
            with self._connect() as conn:
                for s in selected:
                    conn.execute(
                        "UPDATE episodic_skills SET times_retrieved = times_retrieved + 1 WHERE id = ?",
                        (s.id,),
                    )
        return selected

    def record_task_telemetry(self, telemetry: TaskTelemetry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_telemetry (
                    task_id, prompt, domain, passed_first_try, iterations_to_green,
                    patch_survival_rate, skills_used_json, adapter_version, tokens_used,
                    client_node, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telemetry.task_id,
                    telemetry.prompt,
                    telemetry.domain,
                    1 if telemetry.passed_first_try else 0,
                    telemetry.iterations_to_green,
                    telemetry.patch_survival_rate,
                    json.dumps(telemetry.skills_used),
                    telemetry.adapter_version,
                    telemetry.tokens_used,
                    telemetry.client_node,
                    telemetry.created_at,
                ),
            )
            succeeded = telemetry.passed_first_try and telemetry.patch_survival_rate >= 0.70
            for skill_id in telemetry.skills_used:
                if succeeded:
                    conn.execute(
                        "UPDATE episodic_skills SET success_count = success_count + 1 WHERE id = ?",
                        (skill_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE episodic_skills SET failure_count = failure_count + 1 WHERE id = ?",
                        (skill_id,),
                    )
                row = conn.execute(
                    "SELECT success_count, failure_count FROM episodic_skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
                if row:
                    s, f = row["success_count"], row["failure_count"]
                    if (s + f) >= 4 and ((s + 1.0) / (s + f + 2.0)) < 0.30:
                        conn.execute(
                            "UPDATE episodic_skills SET active = 0 WHERE id = ?",
                            (skill_id,),
                        )

    def add_trajectory(
        self,
        task_id: str,
        prompt: str,
        chosen_response: str,
        rejected_response: Optional[str] = None,
        patch_survival_rate: float = 1.0,
        train_split_ratio: float = 0.80,
    ) -> str:
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        bucket = (int.from_bytes(digest[:4], "little") % 1000) / 1000.0
        if rejected_response and bucket >= train_split_ratio:
            split = "golden_replay"
        else:
            split = "train"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO harvested_trajectories (
                    task_id, prompt, rejected_response, chosen_response,
                    patch_survival_rate, split, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    prompt,
                    rejected_response,
                    chosen_response,
                    patch_survival_rate,
                    split,
                    time.time(),
                ),
            )
        return split

    def register_adapter(
        self,
        version: str,
        base_model: str,
        adapter_path: str,
        replay_pass_rate: float,
        holdout_perplexity: float,
        sanity_score: float,
        promoted: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO adapter_registry (
                    version, base_model, adapter_path, replay_pass_rate,
                    holdout_perplexity, sanity_score, promoted, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    base_model,
                    adapter_path,
                    replay_pass_rate,
                    holdout_perplexity,
                    sanity_score,
                    1 if promoted else 0,
                    time.time(),
                ),
            )

    def list_adapters(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM adapter_registry ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_promoted_adapter(self) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM adapter_registry
                WHERE promoted = 1
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
            return dict(row) if row else None

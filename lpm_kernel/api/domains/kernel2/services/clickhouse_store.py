"""
ClickHouse store for chat learning events (REQUIRED).

Why:
- SQLite is fine for core config, but reinforcement logs are append-only, high volume.
- ClickHouse fits better for analytics + offline dataset building.

This module does NOT change the primary app DB. It replaces SQLite for RL logs:
- chat_experiences
- chat_preferences

Env vars (REQUIRED):
- CLICKHOUSE_HTTP_URL: e.g. http://localhost:8123
- CLICKHOUSE_DB: e.g. default
- CLICKHOUSE_USER, CLICKHOUSE_PASSWORD (optional)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from lpm_kernel.common.logging import logger


class ClickHouseStore:
    def __init__(self):
        self.http_url = os.getenv("CLICKHOUSE_HTTP_URL", "").rstrip("/")
        self.db = os.getenv("CLICKHOUSE_DB", "").strip()
        self.user = os.getenv("CLICKHOUSE_USER")
        self.password = os.getenv("CLICKHOUSE_PASSWORD")

        if not self.http_url:
            raise ValueError("CLICKHOUSE_HTTP_URL is required (ClickHouse is the only RL database).")
        if not self.db:
            raise ValueError("CLICKHOUSE_DB is required (ClickHouse is the only RL database).")

    def _post(self, sql: str, data: Optional[str] = None) -> None:
        params = {"database": self.db}
        auth = (self.user, self.password) if self.user and self.password else None
        url = f"{self.http_url}/"
        resp = requests.post(url, params=params, data=(sql + ("\n" + data if data else "")), auth=auth, timeout=10)
        resp.raise_for_status()

    def _query_json_each_row(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute query and parse FORMAT JSONEachRow.
        """
        params = {"database": self.db}
        auth = (self.user, self.password) if self.user and self.password else None
        url = f"{self.http_url}/"
        resp = requests.post(url, params=params, data=sql, auth=auth, timeout=10)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.splitlines() if ln.strip()]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out

    def ensure_schema(self) -> None:
        """
        Create tables if missing. Safe to call repeatedly.
        """
        # Using JSONEachRow format and ReplacingMergeTree for "upserts" by inserting new versions.
        self._post(
            """
CREATE TABLE IF NOT EXISTS chat_experiences
(
    id String,
    source String,
    model Nullable(String),
    prompt_messages String,
    temperature Int32,
    max_tokens Nullable(Int32),
    seed Nullable(Int32),
    completion Nullable(String),
    finish_reason Nullable(String),
    meta_data String,
    created_at DateTime,
    updated_at DateTime
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (id)
            """.strip()
        )
        self._post(
            """
CREATE TABLE IF NOT EXISTS chat_preferences
(
    id String,
    source String,
    model Nullable(String),
    prompt_messages String,
    chosen String,
    rejected String,
    meta_data String,
    created_at DateTime
)
ENGINE = MergeTree
ORDER BY (source, created_at, id)
            """.strip()
        )
        self._post(
            """
CREATE TABLE IF NOT EXISTS chat_feedback
(
    id String,
    experience_id String,
    rating Int32,
    comment Nullable(String),
    created_at DateTime
)
ENGINE = MergeTree
ORDER BY (experience_id, created_at, id)
            """.strip()
        )

    def insert_experience(self, row: Dict[str, Any]) -> None:
        self.ensure_schema()
        line = json.dumps(row, ensure_ascii=False)
        self._post("INSERT INTO chat_experiences FORMAT JSONEachRow", data=line)

    def insert_preference(self, row: Dict[str, Any]) -> None:
        self.ensure_schema()
        line = json.dumps(row, ensure_ascii=False)
        self._post("INSERT INTO chat_preferences FORMAT JSONEachRow", data=line)

    def insert_feedback(self, row: Dict[str, Any]) -> None:
        self.ensure_schema()
        line = json.dumps(row, ensure_ascii=False)
        self._post("INSERT INTO chat_feedback FORMAT JSONEachRow", data=line)

    def count_table(self, table: str) -> int:
        rows = self._query_json_each_row(f"SELECT count() AS c FROM {table} FORMAT JSONEachRow")
        if not rows:
            return 0
        return int(rows[0].get("c", 0))

    def find_latest_experience_id_by_completion(self, completion: str) -> Optional[str]:
        completion_escaped = completion.replace("\\", "\\\\").replace("'", "\\'")
        sql = (
            "SELECT id FROM chat_experiences "
            f"WHERE completion = '{completion_escaped}' "
            "ORDER BY updated_at DESC LIMIT 1 "
            "FORMAT JSONEachRow"
        )
        rows = self._query_json_each_row(sql)
        if not rows:
            return None
        return rows[0].get("id")

    def fetch_preferences(self, limit: int = 5000) -> List[Dict[str, Any]]:
        sql = (
            "SELECT prompt_messages, chosen, rejected "
            "FROM chat_preferences ORDER BY created_at DESC "
            f"LIMIT {int(limit)} "
            "FORMAT JSONEachRow"
        )
        return self._query_json_each_row(sql)


clickhouse_store = ClickHouseStore()


from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptkit.events import (
    Decision,
    FeedbackSentiment,
    FeedbackTarget,
    LearningMode,
    ObservationStatus,
    PreferenceEvent,
)
from adaptkit.exceptions import (
    DecisionMismatchError,
    DecisionNotFoundError,
    StorageBusyError,
    UnsupportedSchemaVersionError,
    ValidationError,
)

from .base import (
    PolicySnapshot,
    PolicyUpdateResult,
    StateStore,
    StateUpdater,
    StoredObservation,
)


class SQLiteStore(StateStore):
    """Single-host durable storage with transactional, idempotent updates."""

    def __init__(self, path: str | Path, *, busy_timeout: float = 1.0) -> None:
        if isinstance(busy_timeout, bool) or not isinstance(busy_timeout, (int, float)):
            raise ValidationError("busy_timeout must be a number")
        if not math.isfinite(busy_timeout) or busy_timeout <= 0:
            raise ValidationError("busy_timeout must be finite and positive")
        if str(path) == ":memory:":
            raise ValidationError(
                "SQLiteStore requires a file path; use InMemoryStore for memory-only state"
            )
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValidationError("SQLite path must be a file, not a directory")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout = float(busy_timeout)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {round(self.busy_timeout * 1000)}")
        return connection

    @staticmethod
    def _translate(exc: sqlite3.OperationalError) -> Exception:
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            return StorageBusyError("SQLite write lock timed out")
        return exc

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                metadata_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='adaptkit_metadata'"
                ).fetchone()
                if metadata_exists:
                    row = connection.execute(
                        "SELECT value FROM adaptkit_metadata WHERE key='schema_version'"
                    ).fetchone()
                    version = "missing" if row is None else row["value"]
                    try:
                        supported = row is not None and int(version) == self.schema_version
                    except (TypeError, ValueError):
                        supported = False
                    if not supported:
                        raise UnsupportedSchemaVersionError(
                            f"unsupported AdaptKit schema version: {version}"
                        )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS adaptkit_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS policies (
                        user_id TEXT NOT NULL,
                        context TEXT NOT NULL,
                        version INTEGER NOT NULL CHECK (version >= 0),
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, context)
                    );
                    CREATE TABLE IF NOT EXISTS policy_actions (
                        user_id TEXT NOT NULL,
                        context TEXT NOT NULL,
                        action TEXT NOT NULL,
                        alpha REAL NOT NULL CHECK (alpha > 0),
                        beta REAL NOT NULL CHECK (beta > 0),
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, context, action)
                    );
                    CREATE TABLE IF NOT EXISTS decisions (
                        decision_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        context TEXT NOT NULL,
                        action TEXT NOT NULL,
                        policy_version INTEGER NOT NULL CHECK (policy_version >= 0),
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS decisions_user_idx
                        ON decisions(user_id, context);
                    CREATE TABLE IF NOT EXISTS observations (
                        observation_id TEXT PRIMARY KEY,
                        decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
                        idempotency_key TEXT NOT NULL,
                        target TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                        source TEXT NOT NULL,
                        learning_mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        applied INTEGER NOT NULL CHECK (applied IN (0, 1)),
                        policy_version_before INTEGER NOT NULL,
                        policy_version_after INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (decision_id, idempotency_key)
                    );
                    CREATE TABLE IF NOT EXISTS policy_operations (
                        user_id TEXT NOT NULL,
                        context TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        policy_version_before INTEGER NOT NULL,
                        policy_version_after INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, context, idempotency_key)
                    );
                    """
                )
                connection.execute(
                    "INSERT OR IGNORE INTO adaptkit_metadata(key, value) VALUES('schema_version', ?)",
                    (str(self.schema_version),),
                )
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _decision(row: sqlite3.Row) -> Decision:
        return Decision(
            decision_id=row["decision_id"],
            user_id=row["user_id"],
            context=row["context"],
            action=row["action"],
            policy_version=row["policy_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _observation(row: sqlite3.Row, *, duplicate: bool = False) -> StoredObservation:
        return StoredObservation(
            observation_id=row["observation_id"],
            decision_id=row["decision_id"],
            idempotency_key=row["idempotency_key"],
            event=PreferenceEvent(
                target=FeedbackTarget(row["target"]),
                sentiment=FeedbackSentiment(row["sentiment"]),
                confidence=row["confidence"],
                source=row["source"],
            ),
            learning_mode=LearningMode(row["learning_mode"]),
            status=ObservationStatus(row["status"]),
            applied=bool(row["applied"]),
            policy_version_before=row["policy_version_before"],
            policy_version_after=row["policy_version_after"],
            created_at=datetime.fromisoformat(row["created_at"]),
            duplicate=duplicate,
        )

    def policy_snapshot(
        self,
        user_id: str,
        context: str,
        actions: Sequence[str],
        initial_state: Mapping[str, float],
    ) -> PolicySnapshot:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                version_row = connection.execute(
                    "SELECT version FROM policies WHERE user_id=? AND context=?",
                    (user_id, context),
                ).fetchone()
                rows = connection.execute(
                    "SELECT action, alpha, beta FROM policy_actions "
                    "WHERE user_id=? AND context=?",
                    (user_id, context),
                ).fetchall()
                connection.execute("COMMIT")
            stored = {row["action"]: {"alpha": row["alpha"], "beta": row["beta"]} for row in rows}
            states = {action: dict(stored.get(action, initial_state)) for action in actions}
            return PolicySnapshot(0 if version_row is None else version_row["version"], states)
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def create_decision(self, decision: Decision) -> None:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM decisions WHERE decision_id=?", (decision.decision_id,)
                ).fetchone()
                if existing is not None:
                    if self._decision(existing) != decision:
                        raise DecisionMismatchError(
                            "decision_id already identifies another decision"
                        )
                    connection.execute("COMMIT")
                    return
                connection.execute(
                    "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        decision.user_id,
                        decision.context,
                        decision.action,
                        decision.policy_version,
                        decision.created_at.isoformat(),
                    ),
                )
                connection.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def get_decision(self, decision_id: str) -> Decision | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM decisions WHERE decision_id=?", (decision_id,)
                ).fetchone()
            return None if row is None else self._decision(row)
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def find_observation(
        self, decision_id: str, idempotency_key: str
    ) -> StoredObservation | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM observations WHERE decision_id=? AND idempotency_key=?",
                    (decision_id, idempotency_key),
                ).fetchone()
            return None if row is None else self._observation(row, duplicate=True)
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    @staticmethod
    def _policy_version(connection: sqlite3.Connection, user_id: str, context: str) -> int:
        row = connection.execute(
            "SELECT version FROM policies WHERE user_id=? AND context=?",
            (user_id, context),
        ).fetchone()
        return 0 if row is None else row["version"]

    @staticmethod
    def _update_action(
        connection: sqlite3.Connection,
        user_id: str,
        context: str,
        action: str,
        initial_state: Mapping[str, float],
        updater: StateUpdater,
        timestamp: str,
    ) -> None:
        row = connection.execute(
            "SELECT alpha, beta FROM policy_actions "
            "WHERE user_id=? AND context=? AND action=?",
            (user_id, context, action),
        ).fetchone()
        state = dict(initial_state) if row is None else {"alpha": row["alpha"], "beta": row["beta"]}
        updater(state)
        connection.execute(
            "INSERT INTO policy_actions VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, context, action) DO UPDATE SET "
            "alpha=excluded.alpha, beta=excluded.beta, updated_at=excluded.updated_at",
            (user_id, context, action, state["alpha"], state["beta"], timestamp),
        )

    @staticmethod
    def _set_policy_version(
        connection: sqlite3.Connection,
        user_id: str,
        context: str,
        version: int,
        timestamp: str,
    ) -> None:
        connection.execute(
            "INSERT INTO policies VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, context) DO UPDATE SET "
            "version=excluded.version, updated_at=excluded.updated_at",
            (user_id, context, version, timestamp),
        )

    def record_observation(
        self,
        *,
        decision: Decision,
        observation_id: str,
        idempotency_key: str,
        event: PreferenceEvent,
        learning_mode: LearningMode,
        status: ObservationStatus,
        initial_state: Mapping[str, float],
        updater: StateUpdater | None,
    ) -> StoredObservation:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                stored_row = connection.execute(
                    "SELECT * FROM decisions WHERE decision_id=?", (decision.decision_id,)
                ).fetchone()
                if stored_row is None:
                    raise DecisionNotFoundError(f"unknown decision: {decision.decision_id}")
                if self._decision(stored_row) != decision:
                    raise DecisionMismatchError("decision does not match its persisted record")
                existing = connection.execute(
                    "SELECT * FROM observations WHERE decision_id=? AND idempotency_key=?",
                    (decision.decision_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    connection.execute("COMMIT")
                    return self._observation(existing, duplicate=True)

                timestamp = self._utc_now()
                version_before = self._policy_version(
                    connection, decision.user_id, decision.context
                )
                version_after = version_before
                if updater is not None:
                    self._update_action(
                        connection,
                        decision.user_id,
                        decision.context,
                        decision.action,
                        initial_state,
                        updater,
                        timestamp,
                    )
                    version_after += 1
                    self._set_policy_version(
                        connection,
                        decision.user_id,
                        decision.context,
                        version_after,
                        timestamp,
                    )
                connection.execute(
                    "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        observation_id,
                        decision.decision_id,
                        idempotency_key,
                        event.target.value,
                        event.sentiment.value,
                        event.confidence,
                        event.source,
                        learning_mode.value,
                        status.value,
                        int(updater is not None),
                        version_before,
                        version_after,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM observations WHERE observation_id=?", (observation_id,)
                ).fetchone()
                connection.execute("COMMIT")
                assert row is not None
                return self._observation(row)
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def atomic_policy_update(
        self,
        *,
        user_id: str,
        context: str,
        idempotency_key: str,
        initial_state: Mapping[str, float],
        updaters: Mapping[str, StateUpdater],
    ) -> PolicyUpdateResult:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT policy_version_before, policy_version_after "
                    "FROM policy_operations WHERE user_id=? AND context=? AND idempotency_key=?",
                    (user_id, context, idempotency_key),
                ).fetchone()
                if existing is not None:
                    connection.execute("COMMIT")
                    return PolicyUpdateResult(
                        False,
                        True,
                        existing["policy_version_before"],
                        existing["policy_version_after"],
                    )
                timestamp = self._utc_now()
                version_before = self._policy_version(connection, user_id, context)
                for action, updater in updaters.items():
                    self._update_action(
                        connection,
                        user_id,
                        context,
                        action,
                        initial_state,
                        updater,
                        timestamp,
                    )
                version_after = version_before + bool(updaters)
                if updaters:
                    self._set_policy_version(
                        connection, user_id, context, version_after, timestamp
                    )
                connection.execute(
                    "INSERT INTO policy_operations VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        context,
                        idempotency_key,
                        version_before,
                        version_after,
                        timestamp,
                    ),
                )
                connection.execute("COMMIT")
                return PolicyUpdateResult(
                    bool(updaters), False, version_before, version_after
                )
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def snapshot(self, user_id: str) -> dict[str, dict[str, dict[str, float]]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT context, action, alpha, beta FROM policy_actions "
                    "WHERE user_id=? ORDER BY context, action",
                    (user_id,),
                ).fetchall()
            result: dict[str, dict[str, dict[str, float]]] = {}
            for row in rows:
                result.setdefault(row["context"], {})[row["action"]] = {
                    "alpha": row["alpha"],
                    "beta": row["beta"],
                }
            return result
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def export_user(self, user_id: str) -> dict[str, Any]:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                policies = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT context, version FROM policies WHERE user_id=? ORDER BY context",
                        (user_id,),
                    )
                ]
                decisions = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT decision_id, context, action, policy_version, created_at "
                        "FROM decisions WHERE user_id=? ORDER BY decision_id",
                        (user_id,),
                    )
                ]
                observations = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT o.observation_id, o.decision_id, o.idempotency_key, "
                        "o.target, o.sentiment, o.confidence, o.source, o.learning_mode, "
                        "o.status, o.applied, o.policy_version_before, "
                        "o.policy_version_after, o.created_at "
                        "FROM observations o JOIN decisions d ON d.decision_id=o.decision_id "
                        "WHERE d.user_id=? ORDER BY o.observation_id",
                        (user_id,),
                    )
                ]
                action_rows = connection.execute(
                    "SELECT context, action, alpha, beta FROM policy_actions "
                    "WHERE user_id=? ORDER BY context, action",
                    (user_id,),
                ).fetchall()
                policy_state: dict[str, dict[str, dict[str, float]]] = {}
                for row in action_rows:
                    policy_state.setdefault(row["context"], {})[row["action"]] = {
                        "alpha": row["alpha"],
                        "beta": row["beta"],
                    }
                connection.execute("COMMIT")
            for observation in observations:
                observation["applied"] = bool(observation["applied"])
            return {
                "schema_version": self.schema_version,
                "user_id": user_id,
                "policies": policies,
                "policy_state": policy_state,
                "decisions": decisions,
                "observations": observations,
            }
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

    def delete_user(self, user_id: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM observations WHERE decision_id IN "
                    "(SELECT decision_id FROM decisions WHERE user_id=?)",
                    (user_id,),
                )
                connection.execute("DELETE FROM decisions WHERE user_id=?", (user_id,))
                connection.execute("DELETE FROM policy_actions WHERE user_id=?", (user_id,))
                connection.execute("DELETE FROM policies WHERE user_id=?", (user_id,))
                connection.execute("DELETE FROM policy_operations WHERE user_id=?", (user_id,))
                connection.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc

import multiprocessing
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from adaptkit import (
    ActionSetMismatchError,
    BetaPrior,
    FeedbackSentiment,
    FeedbackTarget,
    IdempotencyConflictError,
    LearningMode,
    ObservationStatus,
    PreferenceEvent,
    Profile,
    SQLiteStore,
    StorageBusyError,
    UnsupportedSchemaVersionError,
    ValidationError,
)


def _process_like(path: str, decision, key: str) -> None:
    store = SQLiteStore(path, busy_timeout=5)
    profile = Profile(user_id=decision.user_id, actions=[decision.action], store=store)
    profile.like(decision, idempotency_key=key)


def _create_schema_one(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE adaptkit_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO adaptkit_metadata VALUES ('schema_version', '1');
        CREATE TABLE policies (
            user_id TEXT NOT NULL, context TEXT NOT NULL, version INTEGER NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY (user_id, context)
        );
        CREATE TABLE policy_actions (
            user_id TEXT NOT NULL, context TEXT NOT NULL, action TEXT NOT NULL,
            alpha REAL NOT NULL, beta REAL NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, context, action)
        );
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, context TEXT NOT NULL,
            action TEXT NOT NULL, policy_version INTEGER NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE observations (
            observation_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL, target TEXT NOT NULL, sentiment TEXT NOT NULL,
            confidence REAL NOT NULL, source TEXT NOT NULL, learning_mode TEXT NOT NULL,
            status TEXT NOT NULL, applied INTEGER NOT NULL,
            policy_version_before INTEGER NOT NULL, policy_version_after INTEGER NOT NULL,
            created_at TEXT NOT NULL, UNIQUE (decision_id, idempotency_key)
        );
        CREATE TABLE policy_operations (
            user_id TEXT NOT NULL, context TEXT NOT NULL, idempotency_key TEXT NOT NULL,
            policy_version_before INTEGER NOT NULL, policy_version_after INTEGER NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY (user_id, context, idempotency_key)
        );
        INSERT INTO policies VALUES ('u', 'ctx', 1, '2026-01-01T00:00:00+00:00');
        INSERT INTO policy_actions VALUES (
            'u', 'ctx', 'a', 2.0, 1.0, '2026-01-01T00:00:00+00:00'
        );
        """
    )
    connection.commit()
    connection.close()


def _process_initialize(path: str, barrier, results) -> None:
    try:
        barrier.wait(timeout=10)
        SQLiteStore(path, busy_timeout=5)
        results.put("ok")
    except Exception as exc:  # pragma: no cover - reported to the parent process
        results.put(type(exc).__name__)


def _assert_schema_two(test: unittest.TestCase, path: Path) -> None:
    connection = sqlite3.connect(path)
    version = connection.execute(
        "SELECT value FROM adaptkit_metadata WHERE key='schema_version'"
    ).fetchone()[0]
    decision_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(decisions)")
    }
    observation_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(observations)")
    }
    operation_identity_tables = connection.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name='policy_operation_identities'"
    ).fetchone()[0]
    connection.close()
    test.assertEqual(version, "2")
    test.assertIn("selection_source", decision_columns)
    test.assertIn("metadata_json", observation_columns)
    test.assertEqual(operation_identity_tables, 1)


class SQLiteStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "adaptkit.db"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_default_busy_timeout_and_restart_persistence(self):
        store = SQLiteStore(self.database)
        self.assertEqual(store.busy_timeout, 1.0)
        profile = Profile(user_id="u", actions=["a"], store=store)
        decision = profile.choose("ctx")
        profile.like(decision, idempotency_key="first")

        reopened = Profile(
            user_id="u", actions=["a"], store=SQLiteStore(self.database)
        )
        self.assertEqual(reopened.state()["ctx"]["a"], {"alpha": 2.0, "beta": 1.0})
        self.assertEqual(reopened.choose("ctx").policy_version, 1)

    def test_memory_pseudopath_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "file path"):
            SQLiteStore(":memory:")

    def test_duplicate_and_distinct_observation_keys(self):
        profile = Profile(
            user_id="u", actions=["a"], store=SQLiteStore(self.database)
        )
        decision = profile.choose("ctx")
        first = profile.like(decision, idempotency_key="explicit")
        duplicate = profile.like(decision, idempotency_key="explicit")
        second = profile.like(decision, idempotency_key="implicit")
        self.assertTrue(first.updated)
        self.assertEqual(duplicate.status, ObservationStatus.DUPLICATE)
        self.assertTrue(second.updated)
        self.assertEqual(profile.policy("ctx")["version"], 2)

    def test_action_set_and_per_action_priors_persist(self):
        store = SQLiteStore(self.database)
        profile = Profile(
            user_id="u",
            actions=["a", "b"],
            action_priors={"a": BetaPrior(4, 1), "b": BetaPrior(1, 3)},
            store=store,
        )
        profile.choose("ctx")
        reopened = Profile(
            user_id="u",
            actions=["a", "b"],
            action_priors={"a": BetaPrior(1, 9), "b": BetaPrior(9, 1)},
            store=SQLiteStore(self.database),
        )
        self.assertEqual(reopened.state()["ctx"]["a"], {"alpha": 4.0, "beta": 1.0})
        self.assertEqual(
            len(reopened.export_user()["policies"][0]["action_set_fingerprint"]), 64
        )
        with self.assertRaises(ActionSetMismatchError):
            Profile(
                user_id="u",
                actions=["a", "c"],
                store=SQLiteStore(self.database),
            ).choose("ctx")

    def test_passive_signal_payload_conflict_is_atomic(self):
        profile = Profile(
            user_id="u", actions=["a"], store=SQLiteStore(self.database)
        )
        decision = profile.choose("ctx")
        profile.signal(
            decision,
            signal_type="ui.regeneration",
            sentiment="negative",
            metadata={"attempt": 1},
            idempotency_key="regen",
        )
        with self.assertRaises(IdempotencyConflictError):
            profile.signal(
                decision,
                signal_type="ui.regeneration",
                sentiment="positive",
                metadata={"attempt": 2},
                idempotency_key="regen",
            )
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 1.0, "beta": 2.0})

    def test_pairwise_update_is_transactional_and_idempotent(self):
        profile = Profile(
            user_id="u", actions=["a", "b"], store=SQLiteStore(self.database)
        )
        result = profile.prefer(
            "ctx", preferred="a", rejected="b", idempotency_key="pair"
        )
        duplicate = profile.prefer(
            "ctx", preferred="a", rejected="b", idempotency_key="pair"
        )
        self.assertEqual(result.policy_version_after, 1)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.policy_version_before, 0)
        self.assertEqual(duplicate.policy_version_after, 1)
        self.assertEqual(profile.state()["ctx"]["a"]["alpha"], 2)
        self.assertEqual(profile.state()["ctx"]["b"]["beta"], 2)

    def test_concurrent_threads_preserve_every_distinct_signal(self):
        profile = Profile(
            user_id="u",
            actions=["a"],
            store=SQLiteStore(self.database, busy_timeout=5),
        )
        decision = profile.choose("ctx")
        barrier = threading.Barrier(5)

        def apply(index: int) -> None:
            barrier.wait(timeout=5)
            profile.like(decision, idempotency_key=f"thread-{index}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(apply, index) for index in range(4)]
            barrier.wait(timeout=5)
            for future in futures:
                future.result(timeout=10)
        self.assertEqual(profile.state()["ctx"]["a"]["alpha"], 5)
        self.assertEqual(profile.policy("ctx")["version"], 4)

    def test_concurrent_processes_preserve_every_distinct_signal(self):
        profile = Profile(
            user_id="u",
            actions=["a"],
            store=SQLiteStore(self.database, busy_timeout=5),
        )
        decision = profile.choose("ctx")
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_process_like,
                args=(str(self.database), decision, f"process-{index}"),
            )
            for index in range(3)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(profile.state()["ctx"]["a"]["alpha"], 4)
        self.assertEqual(profile.policy("ctx")["version"], 3)

    def test_lock_timeout_has_no_partial_update(self):
        store = SQLiteStore(self.database)
        profile = Profile(user_id="u", actions=["a"], store=store)
        decision = profile.choose("ctx")
        blocker = sqlite3.connect(self.database, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            with self.assertRaises(StorageBusyError):
                profile.like(decision, idempotency_key="blocked")
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        self.assertGreaterEqual(time.monotonic() - started, 0.9)
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 1.0, "beta": 1.0})
        self.assertEqual(profile.export_user()["observations"], [])

    def test_exception_rolls_back_observation_and_state(self):
        store = SQLiteStore(self.database)
        profile = Profile(user_id="u", actions=["a"], store=store)
        decision = profile.choose("ctx")

        def broken_updater(state):
            state["alpha"] += 1
            raise RuntimeError("stop")

        with self.assertRaises(RuntimeError):
            store.record_observation(
                decision=decision,
                observation_id="observation",
                idempotency_key="broken",
                event=PreferenceEvent(
                    FeedbackTarget.BEHAVIOR,
                    FeedbackSentiment.POSITIVE,
                    source="explicit",
                ),
                learning_mode=LearningMode.ACTIVE,
                status=ObservationStatus.UPDATED,
                initial_state={"alpha": 1.0, "beta": 1.0},
                updater=broken_updater,
            )
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 1.0, "beta": 1.0})
        self.assertIsNone(store.find_observation(decision.decision_id, "broken"))

    def test_export_excludes_conversation_text_and_delete_is_complete(self):
        sentinel = "PRIVATE-PROMPT-SENTINEL"
        profile = Profile(
            user_id="u", actions=["a"], store=SQLiteStore(self.database)
        )
        profile.like(profile.choose("ctx"), idempotency_key="explicit")
        exported = profile.export_user()
        self.assertNotIn(sentinel, repr(exported))
        self.assertEqual(len(exported["decisions"]), 1)
        self.assertEqual(len(exported["observations"]), 1)
        profile.delete_user()
        self.assertEqual(profile.export_user()["decisions"], [])
        self.assertEqual(profile.state(), {})

    def test_unsupported_schema_is_rejected(self):
        connection = sqlite3.connect(self.database)
        connection.execute(
            "CREATE TABLE adaptkit_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO adaptkit_metadata VALUES ('schema_version', '999')"
        )
        connection.commit()
        connection.close()
        with self.assertRaises(UnsupportedSchemaVersionError):
            SQLiteStore(self.database)

    def test_malformed_schema_version_is_rejected_cleanly(self):
        connection = sqlite3.connect(self.database)
        connection.execute(
            "CREATE TABLE adaptkit_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO adaptkit_metadata VALUES ('schema_version', 'not-an-integer')"
        )
        connection.commit()
        connection.close()
        with self.assertRaises(UnsupportedSchemaVersionError):
            SQLiteStore(self.database)

    def test_missing_schema_version_is_rejected_cleanly(self):
        connection = sqlite3.connect(self.database)
        connection.execute(
            "CREATE TABLE adaptkit_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(UnsupportedSchemaVersionError, "missing"):
            SQLiteStore(self.database)

    def test_schema_one_database_is_migrated_and_bound_on_first_use(self):
        _create_schema_one(self.database)

        store = SQLiteStore(self.database)
        profile = Profile(user_id="u", actions=["a"], store=store)
        self.assertEqual(profile.policy("ctx")["version"], 1)
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 2.0, "beta": 1.0})
        self.assertEqual(profile.export_user()["schema_version"], 2)

        _assert_schema_two(self, self.database)

    def test_concurrent_threads_migrate_schema_once(self):
        _create_schema_one(self.database)
        barrier = threading.Barrier(5)

        def initialize() -> None:
            barrier.wait(timeout=5)
            SQLiteStore(self.database, busy_timeout=5)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(initialize) for _ in range(4)]
            barrier.wait(timeout=5)
            for future in futures:
                future.result(timeout=10)
        _assert_schema_two(self, self.database)

    def test_concurrent_processes_migrate_schema_once(self):
        _create_schema_one(self.database)
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(4)
        results = context.Queue()
        processes = [
            context.Process(
                target=_process_initialize,
                args=(str(self.database), barrier, results),
            )
            for _ in range(3)
        ]
        for process in processes:
            process.start()
        barrier.wait(timeout=10)
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual([results.get(timeout=2) for _ in processes], ["ok"] * 3)
        _assert_schema_two(self, self.database)

    def test_failed_migration_rolls_back_every_schema_change(self):
        _create_schema_one(self.database)
        original = SQLiteStore._migrate_v1_to_v2_locked

        def fail_after_first_column(store, connection):
            connection.execute(
                "ALTER TABLE policies ADD COLUMN action_set_fingerprint TEXT"
            )
            raise RuntimeError("injected migration failure")

        with patch.object(SQLiteStore, "_migrate_v1_to_v2_locked", fail_after_first_column):
            with self.assertRaisesRegex(RuntimeError, "injected migration failure"):
                SQLiteStore(self.database)

        connection = sqlite3.connect(self.database)
        version = connection.execute(
            "SELECT value FROM adaptkit_metadata WHERE key='schema_version'"
        ).fetchone()[0]
        policy_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(policies)")
        }
        connection.close()
        self.assertEqual(version, "1")
        self.assertNotIn("action_set_fingerprint", policy_columns)

        with patch.object(SQLiteStore, "_migrate_v1_to_v2_locked", original):
            SQLiteStore(self.database)
        _assert_schema_two(self, self.database)

    def test_concurrent_initialization_of_new_database(self):
        barrier = threading.Barrier(5)

        def initialize() -> None:
            barrier.wait(timeout=5)
            SQLiteStore(self.database, busy_timeout=5)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(initialize) for _ in range(4)]
            barrier.wait(timeout=5)
            for future in futures:
                future.result(timeout=10)
        _assert_schema_two(self, self.database)


if __name__ == "__main__":
    unittest.main()

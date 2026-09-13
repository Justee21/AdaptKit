import multiprocessing
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adaptkit import (
    FeedbackSentiment,
    FeedbackTarget,
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
        self.assertEqual(profile.state(), {})
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
        self.assertEqual(profile.state(), {})
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


if __name__ == "__main__":
    unittest.main()

import unittest

from app.core.readiness import readiness_state


class HealthReadinessTests(unittest.TestCase):
    def tearDown(self) -> None:
        readiness_state.reset()

    def test_snapshot_is_ready_when_database_and_embedding_are_ready(self):
        readiness_state.begin_startup(
            requires_local_embedding=True,
            embedding_runtime="sentence_transformers",
            requires_flashrank_warmup=True,
            flashrank_model="ms-marco-TinyBERT-L-2-v2",
        )
        readiness_state.mark_database_ready()
        readiness_state.mark_embedding_ready(runtime="sentence_transformers")
        readiness_state.mark_flashrank_ready(model_name="ms-marco-TinyBERT-L-2-v2")

        snapshot = readiness_state.snapshot()

        self.assertEqual(snapshot.status, "ready")
        self.assertTrue(snapshot.database_ready)
        self.assertTrue(snapshot.embedding_ready)
        self.assertEqual(snapshot.embedding_state, "ready")
        self.assertTrue(snapshot.flashrank_ready)
        self.assertEqual(snapshot.flashrank_state, "ready")

    def test_snapshot_stays_starting_while_embedding_is_warming(self):
        readiness_state.begin_startup(
            requires_local_embedding=True,
            embedding_runtime="sentence_transformers",
            requires_flashrank_warmup=False,
            flashrank_model=None,
        )
        readiness_state.mark_database_ready()
        readiness_state.mark_embedding_warming(runtime="sentence_transformers")

        snapshot = readiness_state.snapshot()

        self.assertEqual(snapshot.status, "starting")
        self.assertTrue(snapshot.database_ready)
        self.assertFalse(snapshot.embedding_ready)
        self.assertEqual(snapshot.embedding_state, "warming")

    def test_snapshot_stays_starting_while_flashrank_is_warming(self):
        readiness_state.begin_startup(
            requires_local_embedding=False,
            embedding_runtime=None,
            requires_flashrank_warmup=True,
            flashrank_model="ms-marco-TinyBERT-L-2-v2",
        )
        readiness_state.mark_database_ready()

        snapshot = readiness_state.snapshot()

        self.assertEqual(snapshot.status, "starting")
        self.assertTrue(snapshot.database_ready)
        self.assertTrue(snapshot.embedding_ready)
        self.assertFalse(snapshot.flashrank_ready)
        self.assertEqual(snapshot.flashrank_state, "warming")

    def test_snapshot_reports_ready_when_flashrank_warmup_failed(self):
        readiness_state.begin_startup(
            requires_local_embedding=False,
            embedding_runtime=None,
            requires_flashrank_warmup=True,
            flashrank_model="ms-marco-TinyBERT-L-2-v2",
        )
        readiness_state.mark_database_ready()
        readiness_state.mark_flashrank_failed(
            "Service is ready, but FlashRank warmup failed. Reranking may fall back to fast mode.",
            model_name="ms-marco-TinyBERT-L-2-v2",
        )

        snapshot = readiness_state.snapshot()

        self.assertEqual(snapshot.status, "ready")
        self.assertEqual(snapshot.flashrank_state, "failed")
        self.assertTrue(snapshot.flashrank_ready)


if __name__ == "__main__":
    unittest.main()

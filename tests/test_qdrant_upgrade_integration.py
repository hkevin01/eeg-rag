"""Opt-in live validation of the patched Qdrant upgrade and rollback."""

import os
import uuid
from unittest.mock import patch

import numpy as np
import pytest

from eeg_rag.storage.vector_db import VectorDB

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("EEG_RAG_TEST_QDRANT_URL"),
        reason="Isolated Qdrant URL not configured",
    ),
]


class FixedEncoder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0, 0.0] for text in texts], dtype=np.float32)


@pytest.fixture
def database():
    name = f"eeg_validation_{uuid.uuid4().hex}"
    url = os.environ["EEG_RAG_TEST_QDRANT_URL"]
    with patch(
        "eeg_rag.storage.vector_db.SentenceTransformer", return_value=FixedEncoder()
    ):
        instance = VectorDB(
            qdrant_url=url, collection_name=name, model_revision="fixture-v1"
        )
    try:
        yield instance
    finally:
        if instance.client.collection_exists(name):
            instance.client.delete_collection(name)
        instance.client.close()


def test_native_bm25_dense_fusion_preserves_year_eligibility(database):
    assert (
        database.index_documents(
            [
                {"id": "old", "text": "P300 EEG latency", "metadata": {"year": 2019}},
                {
                    "id": "eligible",
                    "text": "P300 EEG latency",
                    "metadata": {"year": 2025},
                },
            ]
        )
        == 2
    )
    database.configure_bm25()
    results = database.hybrid_search("P300 EEG", filters={"year": {"gte": 2020}})
    assert len(results) == 1
    assert results[0].payload["id"] == "eligible"
    assert results[0].score > 0


def test_snapshot_restore_preserves_identity_and_documents(database):
    database.index_documents([{"id": "paper", "text": "Sleep spindle EEG"}])
    snapshot = database.client.create_snapshot(database.collection_name)
    restored = f"{database.collection_name}_restored"
    try:
        database.client.recover_snapshot(
            restored,
            location=f"http://127.0.0.1:6333/collections/{database.collection_name}/snapshots/{snapshot.name}",
            wait=True,
        )
        assert database.client.count(restored, exact=True).count == 1
        assert (
            database.client.get_collection(restored).config.metadata
            == database.client.get_collection(database.collection_name).config.metadata
        )
    finally:
        if database.client.collection_exists(restored):
            database.client.delete_collection(restored)


def test_turbo8_requires_and_preserves_full_precision_archive(database, tmp_path):
    from qdrant_client.models import Datatype

    database.vector_datatype = Datatype.TURBO8
    archive = tmp_path / "originals.jsonl"
    assert (
        database.index_documents(
            [{"id": "paper", "text": "EEG epilepsy"}], original_vectors_path=archive
        )
        == 1
    )
    assert '"vector": [1.0, 0.0, 0.0]' in archive.read_text()
    records, _ = database.client.scroll(database.collection_name)
    assert "embedding_vector" not in records[0].payload

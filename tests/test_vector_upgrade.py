"""Vector compatibility and native hybrid contracts without model downloads."""

from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from eeg_rag.storage.vector_db import VectorDB


class FixedEncoder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0, 0.0] for text in texts], dtype=np.float32)


@pytest.fixture
def database():
    with patch(
        "eeg_rag.storage.vector_db.SentenceTransformer", return_value=FixedEncoder()
    ):
        instance = VectorDB(model_revision="test-revision")
    instance._client = QdrantClient(":memory:")
    yield instance
    instance.client.close()


def test_backend_configuration_reaches_model_loader():
    with patch(
        "eeg_rag.storage.vector_db.SentenceTransformer", return_value=FixedEncoder()
    ) as loader:
        VectorDB(embedding_backend="onnx", model_revision="pinned", onnx_threads=2)
    options = loader.call_args.kwargs
    assert options["backend"] == "onnx"
    assert options["revision"] == "pinned"
    assert options["model_kwargs"]["session_options"].intra_op_num_threads == 2


def test_collection_records_embedding_identity(database):
    assert database.create_collection()
    identity = database.client.get_collection(database.collection_name).config.metadata[
        "eeg_rag_embedding"
    ]
    assert identity["model_revision"] == "test-revision"
    assert identity["dimensions"] == 3
    assert identity["normalization"] is False


def test_mismatched_encoder_cannot_search_collection(database):
    assert database.create_collection()
    database.model_revision = "other-revision"
    with pytest.raises(ValueError, match="embedding"):
        database.search("EEG")


def test_incremental_indexing_preserves_existing_papers(database):
    first = {"id": "paper-a", "text": "EEG epilepsy", "metadata": {"year": 2020}}
    second = {"id": "paper-b", "text": "Sleep spindles", "metadata": {"year": 2025}}
    assert database.index_documents([first]) == 1
    assert database.index_documents([second]) == 1
    assert database.client.count(database.collection_name, exact=True).count == 2
    assert database.index_documents([first]) == 1
    assert database.client.count(database.collection_name, exact=True).count == 2


def test_qdrant_range_and_metadata_filters_match_local_policy(database):
    database.index_documents(
        [
            {"id": str(year), "text": "EEG", "metadata": {"year": year}}
            for year in (2019, 2020, 2025, 2026)
        ]
    )
    results = database.search(
        "EEG", filter_conditions={"year": {"gte": 2020, "lte": 2025}}
    )
    assert {result.payload["metadata"]["year"] for result in results} == {2020, 2025}


def test_invalid_filter_is_not_silently_empty_search(database):
    with pytest.raises(ValueError, match="range"):
        database.search("EEG", filter_conditions={"year": {"between": [2020, 2025]}})


def test_turbo_datatype_configuration_is_explicit():
    with patch(
        "eeg_rag.storage.vector_db.SentenceTransformer", return_value=FixedEncoder()
    ):
        instance = VectorDB(vector_datatype="turbo8", model_revision="pinned")
    assert instance.vector_datatype == models.Datatype.TURBO8

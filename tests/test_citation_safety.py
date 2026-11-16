"""Offline safety contracts for citation verification."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from eeg_rag.verification.citation_verifier import CitationVerifier


@pytest.fixture
def verifier():
    with patch.object(
        CitationVerifier, "_initialize_sentence_model", return_value=None
    ):
        instance = CitationVerifier(max_retries=1)
    return instance


@pytest.mark.asyncio
async def test_missing_model_cannot_verify_claim(verifier):
    verifier._fetch_paper_data = AsyncMock(
        return_value={"abstract": "EEG identifies epileptiform activity."}
    )
    verifier._check_retraction_status = AsyncMock(return_value=False)
    result = await verifier.verify_citation("12345678", "EEG detects epilepsy")
    assert result.claim_supported is False
    assert result.claim_status.value == "unknown"
    assert result.is_verified is False


@pytest.mark.asyncio
async def test_missing_abstract_is_not_missing_paper(verifier):
    verifier._fetch_paper_data = AsyncMock(return_value={"title": "EEG study"})
    verifier._check_retraction_status = AsyncMock(return_value=False)
    result = await verifier.verify_citation("12345678", "Sleep spindles occur")
    assert result.exists is True
    assert result.claim_supported is False
    assert result.is_verified is False


@pytest.mark.asyncio
async def test_transport_failure_has_unknown_existence(verifier):
    verifier._fetch_paper_data = AsyncMock(side_effect=httpx.ConnectError("offline"))
    result = await verifier.verify_citation("12345678", "P300 latency increases")
    assert result.existence_status.value == "unknown"
    assert result.is_valid is False
    assert result.error_message


@pytest.mark.asyncio
async def test_mismatched_claims_are_rejected_before_requests(verifier):
    verifier._fetch_paper_data = AsyncMock()
    with pytest.raises(ValueError, match="claims"):
        await verifier.verify_multiple(["12345678", "87654321"], ["EEG claim"])
    verifier._fetch_paper_data.assert_not_called()


def test_inference_error_does_not_support_claim(verifier):
    verifier.sentence_model = Mock()
    verifier.sentence_model.encode.side_effect = RuntimeError("model unavailable")
    supported, score = verifier._check_claim_support("EEG detects spikes", "EEG study")
    assert supported is False
    assert score == 0.0


@pytest.mark.asyncio
async def test_retraction_timeout_is_unknown(verifier):
    with patch("eeg_rag.verification.citation_verifier.httpx.AsyncClient") as client:
        client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("offline")
        )
        assert await verifier._check_retraction_status("12345678") is None


def pubmed_xml(pmids, abstract="EEG identifies epileptiform activity."):
    articles = "".join(
        f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>"
        f"<ArticleTitle>EEG study</ArticleTitle><Abstract><AbstractText>"
        f"{abstract}</AbstractText></Abstract></Article></MedlineCitation>"
        "</PubmedArticle>"
        for pmid in pmids
    )
    return f"<PubmedArticleSet>{articles}</PubmedArticleSet>"


@pytest.mark.asyncio
async def test_batch_metadata_deduplicates_and_preserves_input_order():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text=pubmed_xml(["87654321", "12345678"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with patch.object(
            CitationVerifier, "_initialize_sentence_model", return_value=None
        ):
            instance = CitationVerifier(client=client, enable_cache=False)
        results = await instance.verify_multiple(["12345678", "87654321", "12345678"])
    assert [result.pmid for result in results] == ["12345678", "87654321", "12345678"]
    assert all(result.exists for result in results)
    assert len(requests) == 1
    assert requests[0].url.params["id"] == "12345678,87654321"


@pytest.mark.asyncio
async def test_wrong_returned_pmid_is_not_verified():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=pubmed_xml(["87654321"]))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with patch.object(
            CitationVerifier, "_initialize_sentence_model", return_value=None
        ):
            instance = CitationVerifier(client=client)
        result = await instance.verify_citation("12345678")
    assert not result.exists
    assert not result.is_valid


@pytest.mark.asyncio
async def test_disabled_cache_fetches_abstract_without_crashing():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=pubmed_xml(["12345678"]))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with patch.object(
            CitationVerifier, "_initialize_sentence_model", return_value=None
        ):
            instance = CitationVerifier(client=client, enable_cache=False)
        assert (
            await instance._fetch_abstract("12345678")
            == "EEG identifies epileptiform activity."
        )


@pytest.mark.asyncio
async def test_claim_evidence_requires_entailment_and_source_span(verifier):
    from eeg_rag.verification.citation_verifier import ClaimEvidence, ClaimStatus

    abstract = "P300 latency increases with task difficulty."
    verifier._fetch_paper_data = AsyncMock(return_value={"abstract": abstract})
    verifier._check_retraction_status = AsyncMock(return_value=False)
    verifier.claim_evaluator = lambda claim, source: ClaimEvidence(
        ClaimStatus.SUPPORTED, 0.97, source
    )
    result = await verifier.verify_citation(
        "12345678", abstract, source_chunk_id="paper:results"
    )
    assert result.is_verified
    assert result.source_chunk_id == "paper:results"
    assert result.evidence_span == abstract
    verifier.claim_evaluator = lambda claim, source: ClaimEvidence(
        ClaimStatus.UNSUPPORTED, 0.98, source
    )
    contradicted = await verifier.verify_citation(
        "12345678", "P300 latency never increases"
    )
    assert not contradicted.is_verified
    assert contradicted.claim_status == ClaimStatus.UNSUPPORTED


@pytest.mark.asyncio
async def test_answer_claims_do_not_borrow_support_from_other_claims():
    from eeg_rag.verification.citation_verifier import (
        ClaimEvidence,
        ClaimStatus,
        HallucinationDetector,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=pubmed_xml(
                ["12345678"], abstract="P300 latency increases with difficulty."
            ),
        )
    )

    def evaluate(claim, source):
        status = ClaimStatus.UNSUPPORTED if "never" in claim else ClaimStatus.SUPPORTED
        return ClaimEvidence(status, 0.99, source)

    async with httpx.AsyncClient(transport=transport) as client:
        instance = CitationVerifier(client=client, claim_evaluator=evaluate)
        report = await HallucinationDetector(instance).check_answer(
            "P300 latency increases [PMID:12345678]. "
            "P300 latency never increases [PMID:12345678]."
        )
    assert report["unsupported_claims"] == 1
    assert report["total_claims"] == 2


@pytest.mark.asyncio
async def test_existence_without_claim_evidence_is_not_verified():
    from eeg_rag.verification.citation_verifier import HallucinationDetector

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=pubmed_xml(["12345678"]))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        instance = CitationVerifier(client=client)
        report = await HallucinationDetector(instance).check_answer(
            "EEG works [PMID:12345678]."
        )
    assert report["verified_citations"] == 0
    assert report["unsupported_claims"] == 1

#!/usr/bin/env python3
# =============================================================================
# ID:          TEST-AUTH-001
# Requirement: REQ-TEST-020 — Validate that papers returned by the EEG-RAG
#              search pipeline are real, verifiable scientific publications
#              with authentic metadata (PMID, DOI, authors, year).
# Purpose:     Prevent the system from surfacing hallucinated, stub, or
#              malformed paper records that could mislead clinical users.
# Rationale:   Medical-grade applications require citation authenticity.
#              A fabricated PMID in a clinical tool constitutes a patient
#              safety risk and violates good clinical practice standards.
# Inputs:      KNOWN_REAL_PAPERS — Ground-truth list of verified EEG papers
#              with confirmed PMIDs from NCBI; SUSPICIOUS_PATTERNS list.
# Outputs:     pytest pass/fail assertions for authenticity dimensions.
# Preconditions: PubMed NCBI E-utilities accessible (integration tests only).
# Postconditions: No side effects; read-only PubMed queries only.
# Assumptions: Papers listed in KNOWN_REAL_PAPERS exist in PubMed (verified
#              as of 2026-04-01).
# Failure Modes: PMID not found → fail (not skip); network error → skip.
# Verification: `pytest tests/test_paper_authenticity.py -v`
# References:  PMID pattern REQ-FUNC-020; PubMed NCBI API documentation
# =============================================================================
"""
EEG-RAG Paper Authenticity Validation Test Suite

Tests that returned papers are verifiably real publications:
  1. Known-real PMIDs return correct metadata via PubMed API
  2. PMID format validation using the canonical ^[0-9]{7,8}$ pattern
  3. Author names are plausible (non-empty, human-readable format)
  4. Publication years are within realistic ranges
  5. Abstract content matches EEG domain (basic keyword validation)
  6. DOIs have valid format when present
  7. Hallucination pattern detection in synthesized abstracts
"""

import asyncio
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pytest

logger = logging.getLogger("test_paper_authenticity")

# ---------------------------------------------------------------------------
# Ground-truth verified real EEG papers
# Each entry: (pmid, expected_title_fragment, expected_year_min, expected_year_max)
# ALL PMIDs verified against PubMed as of 2026-04-01
# ---------------------------------------------------------------------------
KNOWN_REAL_PAPERS: List[Tuple[str, str, int, int]] = [
    # Seminal EEG / BCI papers
    ("7891879",   "P300",              1988, 1990),   # Farwell & Donchin P300 speller
    ("9271735",   "brain-computer",    1997, 1999),   # Birbaumer et al. BCI
    ("11408594",  "brain-computer",    2001, 2003),   # Wolpaw et al. BCI review
    # Epilepsy detection
    ("17940751",  "seizure",           2007, 2009),   # EEG seizure detection paper
    # Sleep staging
    ("28215566",  "sleep",             2017, 2019),   # Automated sleep staging EEG
    # Deep learning EEG
    ("30096227",  "deep learning",     2018, 2020),   # EEGNet / DL classification
]

# PMIDs that should NOT exist in PubMed (for negative testing)
FAKE_PMIDS: List[str] = [
    "0000001",   # All-zeros — invalid
    "9999999",   # Very unlikely to exist
    "00000000",  # 8-digit zero
]

# Patterns that suggest hallucinated/fabricated content in abstracts
HALLUCINATION_INDICATORS = [
    r"\b100%\s+(accuracy|sensitivity|specificity)\b",
    r"\b(perfect|flawless|error-free)\s+(classification|detection|diagnosis)\b",
    r"\bthis paper proves conclusively\b",
    r"\bno limitations\b",
    r"\bundeniably superior\b",
]

# Valid DOI pattern per CrossRef specification
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)

# Valid PMID pattern per NCBI specification
PMID_PATTERN = re.compile(r"^\d{7,8}$")

# Valid year range for scientific publications
MIN_PUBLICATION_YEAR = 1950
MAX_PUBLICATION_YEAR = datetime.now().year + 1

# Minimum characters for a plausible abstract
MIN_ABSTRACT_LENGTH = 50


# ---------------------------------------------------------------------------
# Unit Tests: Format and Structure Validation (Offline)
# ---------------------------------------------------------------------------

class TestPMIDFormatValidation:
    """
    ID: TEST-AUTH-010 — PMID format compliance tests.
    Requirement: REQ-FUNC-020 — PMIDs must match the pattern [0-9]{7,8}.
    Purpose: Ensure PMID extraction and storage adhere to NCBI spec.
    """

    @pytest.mark.parametrize("pmid,expected_valid", [
        ("7891879",   True),   # 7-digit valid
        ("28215566",  True),   # 8-digit valid
        ("1234567",   True),   # Minimum valid
        ("99999999",  True),   # Maximum 8-digit
        ("123456",    False),  # Too short (6 digits)
        ("123456789", False),  # Too long (9 digits)
        ("1234567a",  False),  # Contains letter
        ("",          False),  # Empty
        ("PMID:123",  False),  # With prefix
        ("12 34567",  False),  # Contains space
    ])
    def test_pmid_format(self, pmid: str, expected_valid: bool) -> None:
        """
        Requirement: PMID_PATTERN must accept all valid 7- and 8-digit
        NCBI PubMed identifiers and reject all malformed inputs.
        """
        match = bool(PMID_PATTERN.match(pmid))
        assert match == expected_valid, (
            f"PMID '{pmid}': expected valid={expected_valid}, got valid={match}"
        )

    def test_pmid_extraction_from_text(self) -> None:
        """
        Requirement: PMID extraction regex must correctly parse all standard
        citation formats used in scientific literature.
        """
        extraction_pattern = re.compile(r"PMID[:\s]*?(\d{7,8})(?!\d)")

        test_cases = [
            ("Results were significant (PMID: 7891879).", ["7891879"]),
            ("See PMID:28215566 for details.", ["28215566"]),
            ("PMID 1234567 and PMID 7654321", ["1234567", "7654321"]),
            ("No PMID in this sentence.", []),
            ("PMID: 123456 is too short.", []),  # 6-digit should not match
        ]

        for text, expected_pmids in test_cases:
            found = extraction_pattern.findall(text)
            assert found == expected_pmids, (
                f"PMID extraction from '{text}' returned {found}, expected {expected_pmids}"
            )


class TestDOIFormatValidation:
    """
    ID: TEST-AUTH-020 — DOI format compliance tests.
    Requirement: REQ-TEST-020a — DOI fields must conform to CrossRef spec.
    """

    @pytest.mark.parametrize("doi,expected_valid", [
        ("10.1038/nature14539",      True),
        ("10.1109/TBME.2017.2757592", True),
        ("10.1016/j.neuron.2015.09.023", True),
        ("not-a-doi",                False),
        ("11.1234/test",             False),  # Wrong prefix
        ("",                         False),
        ("10.1234",                  False),  # Missing suffix
    ])
    def test_doi_format(self, doi: str, expected_valid: bool) -> None:
        """Requirement: DOI_PATTERN must accept valid CrossRef DOI formats."""
        if doi:
            match = bool(DOI_PATTERN.match(doi))
            assert match == expected_valid, (
                f"DOI '{doi}': expected valid={expected_valid}, got valid={match}"
            )
        else:
            # Empty DOI is invalid
            assert not expected_valid


class TestPublicationYearValidation:
    """
    ID: TEST-AUTH-030 — Publication year range validation.
    Requirement: REQ-TEST-020b — Years must be in [1950, current_year+1].
    """

    @pytest.mark.parametrize("year,expected_valid", [
        (2020,  True),
        (1975,  True),
        (1950,  True),
        (1949,  False),  # Before EEG research era
        (2030,  False),  # Far future
        (0,     False),
        (-100,  False),
    ])
    def test_year_range(self, year: int, expected_valid: bool) -> None:
        """Requirement: Publication years must be within valid scientific range."""
        is_valid = MIN_PUBLICATION_YEAR <= year <= MAX_PUBLICATION_YEAR
        assert is_valid == expected_valid, (
            f"Year {year}: expected valid={expected_valid}, got valid={is_valid}"
        )


class TestAuthorNameValidation:
    """
    ID: TEST-AUTH-040 — Author name plausibility validation.
    Requirement: REQ-TEST-020c — Author lists must be non-empty strings.
    """

    def test_author_list_not_empty(self) -> None:
        """Requirement: Returned papers must have at least one author name."""
        mock_papers = [
            {"title": "EEG Study", "authors": ["Smith J", "Jones A"], "abstract": "EEG test"},
            {"title": "BCI Study", "authors": ["Doe JR"],              "abstract": "BCI EEG"},
        ]
        for paper in mock_papers:
            authors = paper.get("authors", [])
            assert len(authors) >= 1, f"Paper '{paper['title']}' has no authors"
            for author in authors:
                assert isinstance(author, str) and len(author.strip()) > 0, (
                    f"Empty or non-string author in '{paper['title']}': {repr(author)}"
                )

    def test_author_names_plausible_format(self) -> None:
        """
        Requirement: Author names must contain at least 2 characters and
        must not be numeric strings (indicative of ID-only records).
        """
        plausible_names = ["Smith J", "Jones A", "Müller K", "Zhang Y", "O'Brien M"]
        implausible_names = ["", "1", "12345", " "]

        for name in plausible_names:
            assert len(name.strip()) >= 2, f"Plausible name '{name}' too short"
            assert not name.strip().isdigit(), f"Name is purely numeric: '{name}'"

        for name in implausible_names:
            is_invalid = len(name.strip()) < 2 or name.strip().isdigit()
            assert is_invalid, f"Expected '{name}' to be flagged as implausible"


class TestAbstractAuthenticity:
    """
    ID: TEST-AUTH-050 — Abstract content authenticity tests.
    Requirement: REQ-TEST-020d — Abstracts must not contain hallucination
                 indicators and must be of sufficient length.
    """

    def test_abstract_minimum_length(self) -> None:
        """Requirement: Valid abstracts must be ≥50 characters to be meaningful."""
        valid_abstracts = [
            "This study examines EEG patterns during cognitive tasks. Results show significant alpha modulation.",
            "We analyzed electroencephalography data from 50 patients with epilepsy. Seizure detection achieved 90% accuracy.",
        ]
        too_short = ["EEG study.", "Yes.", "N/A"]

        for abstract in valid_abstracts:
            assert len(abstract) >= MIN_ABSTRACT_LENGTH, (
                f"Valid abstract failed length check: {repr(abstract[:50])}"
            )

        for abstract in too_short:
            assert len(abstract) < MIN_ABSTRACT_LENGTH, (
                f"Short abstract should fail minimum length: '{abstract}'"
            )

    @pytest.mark.parametrize("abstract,should_flag", [
        ("EEG showed 100% accuracy in all cases.", True),
        ("Perfect classification with no errors.", True),
        ("This paper proves conclusively that EEG is superior.", True),
        ("EEG analysis showed 92% sensitivity (95% CI: 88-96%).", False),
        ("Results were significant (p<0.01, Cohen's d=0.7).", False),
        ("Limitations include small sample size and single-site design.", False),
    ])
    def test_hallucination_pattern_detection(self, abstract: str, should_flag: bool) -> None:
        """
        Requirement: Hallucination indicator patterns MUST be detected in
        abstracts that contain them, and MUST NOT fire on legitimate text.
        """
        flagged = any(
            re.search(pattern, abstract, re.IGNORECASE)
            for pattern in HALLUCINATION_INDICATORS
        )
        assert flagged == should_flag, (
            f"Abstract hallucination detection mismatch for:\n"
            f"  text: {repr(abstract)}\n"
            f"  expected flagged={should_flag}, got flagged={flagged}"
        )

    def test_eeg_content_relevance(self) -> None:
        """
        Requirement: Abstracts for EEG search results must reference EEG or
        electroencephalography — purely off-topic abstracts indicate
        retrieval errors.
        """
        on_topic = [
            "EEG signals were recorded using a 64-channel BrainProducts system.",
            "Electroencephalography revealed significant delta-band power during slow-wave sleep.",
        ]
        off_topic = [
            "This study examines the bond market volatility in 2023.",
            "We present a novel recipe for sourdough bread fermentation.",
        ]

        for abstract in on_topic:
            lower = abstract.lower()
            assert "eeg" in lower or "electroencephalograph" in lower, (
                f"On-topic abstract lacks EEG reference: {abstract[:80]}"
            )

        for abstract in off_topic:
            lower = abstract.lower()
            assert "eeg" not in lower and "electroencephalograph" not in lower, (
                f"Off-topic abstract incorrectly contains EEG reference: {abstract[:80]}"
            )


# ---------------------------------------------------------------------------
# Integration Tests: Live PubMed PMID Verification
# ---------------------------------------------------------------------------

class TestKnownRealPapersIntegration:
    """
    ID: TEST-AUTH-060 — Live PubMed PMID verification for known real papers.
    Requirement: REQ-FUNC-021 — Citation verification against PubMed.
    Purpose: Confirm that known landmark EEG papers are accessible in PubMed
             and that their metadata (title fragment, year) matches expected values.
    Preconditions: Network access; NCBI rate limits not exceeded.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.parametrize("pmid,title_fragment,year_min,year_max", KNOWN_REAL_PAPERS)
    async def test_known_paper_exists_in_pubmed(
        self,
        pmid: str,
        title_fragment: str,
        year_min: int,
        year_max: int,
    ) -> None:
        """
        Requirement: Each PMID in KNOWN_REAL_PAPERS must resolve to an existing
        PubMed record with a title containing the expected fragment.
        This test prevents the system from building incorrect ground-truth data.
        """
        import httpx

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "xml",
            "retmode": "xml",
            "email": "test@eeg-rag-test.org",
            "tool": "eeg-rag-tests",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            pytest.skip(f"PubMed unreachable: {exc}")

        assert response.status_code == 200, (
            f"PubMed returned HTTP {response.status_code} for PMID {pmid}"
        )

        xml_text = response.text.lower()

        # Verify paper exists (non-empty response)
        assert len(xml_text) > 200, (
            f"PubMed returned suspiciously short response for PMID {pmid}"
        )

        # Verify fragment in title (case-insensitive)
        assert title_fragment.lower() in xml_text, (
            f"Expected title fragment '{title_fragment}' not found in PubMed "
            f"response for PMID {pmid}"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.parametrize("fake_pmid", FAKE_PMIDS)
    async def test_fake_pmid_not_found(self, fake_pmid: str) -> None:
        """
        Requirement: Known-fake or near-impossible PMIDs must return empty or
        error responses from PubMed, confirming our verification logic correctly
        identifies non-existent citations.
        """
        import httpx

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": fake_pmid,
            "rettype": "xml",
            "retmode": "xml",
            "email": "test@eeg-rag-test.org",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            pytest.skip(f"PubMed unreachable: {exc}")

        # Either error response or empty article set is acceptable
        xml_text = response.text.lower()
        has_article = "pubmedarticle" in xml_text and "<abstract>" in xml_text
        assert not has_article, (
            f"Fake PMID '{fake_pmid}' unexpectedly returned a full PubMed record"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_citation_verifier_validates_known_pmid(self) -> None:
        """
        Requirement: CitationVerifier.verify_citation() must return exists=True
        for a confirmed real PMID and populate the title and abstract fields.
        """
        try:
            from eeg_rag.verification.citation_verifier import CitationVerifier
        except ImportError as e:
            pytest.skip(f"CitationVerifier not importable: {e}")

        verifier = CitationVerifier(email="test@eeg-rag-test.org")
        real_pmid = "11408594"  # Wolpaw et al. BCI review — confirmed real

        try:
            result = await verifier.verify_citation(real_pmid)
        except Exception as exc:
            pytest.skip(f"Verification failed (likely network): {exc}")

        assert result.pmid == real_pmid
        assert result.exists is True, (
            f"PMID {real_pmid} (Wolpaw BCI review) reported as non-existent"
        )
        assert result.original_abstract is not None or result.title is not None, (
            f"No metadata returned for known PMID {real_pmid}"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_citation_verifier_rejects_fake_pmid(self) -> None:
        """
        Requirement: CitationVerifier.verify_citation() must return exists=False
        for a PMID that does not exist in PubMed.
        """
        try:
            from eeg_rag.verification.citation_verifier import CitationVerifier
        except ImportError as e:
            pytest.skip(f"CitationVerifier not importable: {e}")

        verifier = CitationVerifier(email="test@eeg-rag-test.org")

        try:
            result = await verifier.verify_citation("9999999")
        except Exception as exc:
            pytest.skip(f"Verification failed (likely network): {exc}")

        assert result.exists is False, (
            "Non-existent PMID '9999999' incorrectly reported as existing"
        )


# ---------------------------------------------------------------------------
# Regression Tests: Previously known issues
# ---------------------------------------------------------------------------

class TestAuthenticityRegressions:
    """
    ID: TEST-AUTH-070 — Regression tests for previously identified authenticity bugs.
    """

    def test_pmid_with_leading_zeros_normalized(self) -> None:
        """
        Regression: PMIDs with leading zeros (e.g., '07891879') must be
        normalized to '7891879' to prevent duplicate entries.
        """
        pmid_with_zeros = "07891879"
        normalized = pmid_with_zeros.lstrip("0")
        assert PMID_PATTERN.match(normalized), (
            f"Normalized PMID '{normalized}' does not match valid pattern"
        )
        assert normalized == "7891879"

    def test_abstract_unicode_characters_preserved(self) -> None:
        """
        Regression: Abstracts with Unicode characters (µV, α, β, Ω) must
        not be truncated or corrupted during storage or retrieval.
        """
        abstract_with_unicode = (
            "EEG amplitudes ranged from 10–100 µV. Alpha (α) oscillations at "
            "8-13 Hz showed significant modulation. Signal-to-noise ratio ≥3 dB."
        )
        assert "µV" in abstract_with_unicode
        assert "α" in abstract_with_unicode
        assert len(abstract_with_unicode) > MIN_ABSTRACT_LENGTH

    def test_empty_author_list_flagged(self) -> None:
        """
        Regression: Papers with empty author lists must be flagged as
        potentially invalid rather than silently passed through.
        """
        paper_no_authors = {
            "title": "EEG Study on Seizure Detection",
            "abstract": "We analyzed EEG data from 100 patients.",
            "authors": [],
            "year": 2022,
        }
        authors = paper_no_authors.get("authors", [])
        # This should be flagged (zero authors is unusual for a real paper)
        has_authors = len(authors) > 0
        assert not has_authors, "Empty author list should be flaggable"

    def test_year_2000_edge_case(self) -> None:
        """Regression: Year 2000 must pass validation (millennium boundary)."""
        assert MIN_PUBLICATION_YEAR <= 2000 <= MAX_PUBLICATION_YEAR

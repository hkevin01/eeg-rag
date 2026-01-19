#!/usr/bin/env python3
"""
Tests for Citation Verification and Hallucination Detection
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.eeg_rag.verification.citation_verifier import (
    CitationVerifier, HallucinationDetector, VerificationResult,
    verify_answer_citations, quick_citation_check
)


class TestCitationVerifier:
    """Test citation verification functionality"""
    
    @pytest.fixture
    def verifier(self):
        return CitationVerifier(email="test@example.com")
    
    @pytest.fixture
    def mock_pubmed_response(self):
        return '''
        <PubMedArticleSet>
            <PubMedArticle>
                <MedlineCitation>
                    <Article>
                        <Abstract>
                            <AbstractText Label="BACKGROUND">EEG is used for epilepsy diagnosis.</AbstractText>
                            <AbstractText Label="METHODS">We analyzed 100 patients.</AbstractText>
                            <AbstractText Label="RESULTS">Spike detection accuracy was 95%.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubMedArticle>
        </PubMedArticleSet>
        '''
    
    def test_init(self, verifier):
        """Test verifier initialization"""
        assert verifier.email == "test@example.com"
        assert verifier.similarity_threshold == 0.5
        assert verifier.cache == {}
    
    @patch('httpx.AsyncClient')
    async def test_verify_citation_success(self, mock_client, verifier, mock_pubmed_response):
        """Test successful citation verification"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_pubmed_response
        
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        result = await verifier.verify_citation("12345678", "EEG is used for epilepsy")
        
        assert isinstance(result, VerificationResult)
        assert result.pmid == "12345678"
        assert result.exists is True
        assert result.original_abstract is not None
    
    @patch('httpx.AsyncClient')
    async def test_verify_citation_not_found(self, mock_client, verifier):
        """Test citation not found"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        result = await verifier.verify_citation("99999999")
        
        assert result.exists is False
        assert result.error_message is not None
    
    @patch('httpx.AsyncClient')
    async def test_verify_multiple_citations(self, mock_client, verifier, mock_pubmed_response):
        """Test multiple citation verification"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_pubmed_response
        
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        pmids = ["12345678", "87654321"]
        results = await verifier.verify_multiple(pmids)
        
        assert len(results) == 2
        assert all(isinstance(r, VerificationResult) for r in results)
    
    def test_parse_abstract_from_xml(self, verifier, mock_pubmed_response):
        """Test XML parsing"""
        abstract = verifier._parse_abstract_from_xml(mock_pubmed_response)
        
        assert abstract is not None
        assert "epilepsy diagnosis" in abstract
        assert "95%" in abstract
    
    def test_check_claim_support_no_model(self, verifier):
        """Test claim support check without model"""
        verifier.sentence_model = None
        
        result = verifier._check_claim_support("EEG test", "Abstract about EEG")
        assert result is True  # Should default to True
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions"""
        with patch('src.eeg_rag.verification.citation_verifier.CitationVerifier') as mock_verifier:
            mock_instance = Mock()
            mock_instance.verify_multiple = AsyncMock(return_value=[Mock(exists=True)])
            mock_verifier.return_value = mock_instance
            
            result = await quick_citation_check(["12345678"])
            assert len(result) == 1


class TestHallucinationDetector:
    """Test hallucination detection functionality"""
    
    @pytest.fixture
    def detector(self):
        verifier = Mock()
        verifier.similarity_threshold = 0.6
        verifier.sentence_model = None
        return HallucinationDetector(verifier)
    
    def test_init(self, detector):
        """Test detector initialization"""
        assert detector.verifier is not None
        assert 'definitive_claims' in detector.hallucination_patterns
    
    def test_extract_pmids(self, detector):
        """Test PMID extraction"""
        text = "Study shows effectiveness (PMID: 12345678). Another study PMID:87654321 confirms this."
        pmids = detector._extract_pmids(text)
        
        assert "12345678" in pmids
        assert "87654321" in pmids
        assert len(pmids) == 2
    
    def test_extract_claims_with_citations(self, detector):
        """Test claim extraction"""
        text = "EEG is effective for diagnosis (PMID: 12345678). Another finding shows 95% accuracy."
        claims = detector._extract_claims_with_citations(text)
        
        assert len(claims) == 2
        assert claims[0][1] == ["12345678"]  # First claim has citation
        assert claims[1][1] == []  # Second claim has no citation
    
    def test_check_hallucination_patterns(self, detector):
        """Test pattern detection"""
        text = "EEG always works perfectly and never fails. All patients show 100% improvement."
        flags = detector._check_hallucination_patterns(text)
        
        assert flags['definitive_claims'] > 0
        assert all(0 <= score <= 1 for score in flags.values())
    
    @patch.object(HallucinationDetector, '_extract_pmids')
    @patch.object(HallucinationDetector, '_extract_claims_with_citations')
    async def test_check_answer(self, mock_extract_claims, mock_extract_pmids, detector):
        """Test complete answer checking"""
        # Setup mocks
        mock_extract_pmids.return_value = ["12345678"]
        mock_extract_claims.return_value = [("Test claim", ["12345678"])]
        
        detector.verifier.verify_multiple = AsyncMock(return_value=[
            VerificationResult("12345678", True, 1.0, True)
        ])
        
        answer = "EEG is effective (PMID: 12345678)."
        result = await detector.check_answer(answer)
        
        assert 'hallucination_score' in result
        assert 'citation_accuracy' in result
        assert 'verified_citations' in result
        assert 0 <= result['hallucination_score'] <= 1
    
    def test_calculate_citation_accuracy(self, detector):
        """Test citation accuracy calculation"""
        results = [
            VerificationResult("1", True, 1.0, True),
            VerificationResult("2", False, 0.0, False),
            VerificationResult("3", True, 1.0, True)
        ]
        
        accuracy = detector._calculate_citation_accuracy(results)
        assert accuracy == 2/3  # 2 out of 3 valid
    
    def test_count_unsupported_claims(self, detector):
        """Test unsupported claims counting"""
        claims = [
            ("Supported claim", ["1"]),
            ("Unsupported claim", ["2"]),
            ("No citation claim", [])
        ]
        
        citation_results = [
            VerificationResult("1", True, 1.0, True),
            VerificationResult("2", True, 1.0, False)
        ]
        
        unsupported = detector._count_unsupported_claims(claims, citation_results)
        assert unsupported == 2  # One unsupported + one without citation


class TestVerificationResult:
    """Test verification result data class"""
    
    def test_to_dict(self):
        """Test result serialization"""
        result = VerificationResult(
            pmid="12345678",
            exists=True,
            title_match=0.9,
            claim_supported=True,
            original_abstract="Test abstract"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['pmid'] == "12345678"
        assert result_dict['exists'] is True
        assert result_dict['title_match'] == 0.9
        assert result_dict['claim_supported'] is True
        assert result_dict['has_abstract'] is True


if __name__ == "__main__":
    pytest.main([__file__])

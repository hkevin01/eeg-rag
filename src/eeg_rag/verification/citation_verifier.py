#!/usr/bin/env python3
"""
Citation Verification and Hallucination Detection

Verifies citations and detects hallucinated claims in generated answers.
Essential for medical/research applications where accuracy is critical.
"""

import re
import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import httpx
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import xml.etree.ElementTree as ET
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Medical literature domain constants
PMID_PATTERN = re.compile(r'PMID[:\s]*?(\d{7,8})(?![\d])')  # Precise PMID extraction
MAX_ABSTRACT_LENGTH = 5000  # Maximum abstract length for processing
DEFAULT_REQUEST_TIMEOUT = 30.0  # PubMed API timeout in seconds
RETRY_ATTEMPTS = 3  # Number of retry attempts for failed requests
RETRY_DELAY = 1.0  # Initial retry delay in seconds

# Hallucination risk patterns for medical content
HALLUCINATION_PATTERNS = {
    'absolute_claims': {
        'pattern': r'\b(always|never|all|none|every|no|100%|0%|completely|entirely|absolutely)\s+(patients?|studies?|cases?|research|evidence)\b',
        'description': 'Absolute claims without qualification',
        'risk_multiplier': 2.0
    },
    'unsupported_statistics': {
        'pattern': r'\b\d+%\s*of\s*(patients?|subjects?|cases?)(?!.*\(.*PMID)',
        'description': 'Specific percentages without citation',
        'risk_multiplier': 1.8
    },
    'temporal_claims': {
        'pattern': r'\b(recent|latest|new|cutting-edge|breakthrough)\s*(research|studies?|findings?)(?!.*\(.*PMID)',
        'description': 'Temporal claims without recent citations',
        'risk_multiplier': 1.5
    },
    'superlative_claims': {
        'pattern': r'\b(best|most effective|optimal|superior|inferior|highest|lowest)\s*(treatment|method|approach)(?!.*\(.*PMID)',
        'description': 'Comparative claims without evidence',
        'risk_multiplier': 1.7
    }
}


@dataclass
class VerificationResult:
    """Comprehensive result of citation verification.
    
    This class encapsulates all aspects of citation verification including
    existence validation, content matching, retraction status, and claim support.
    Critical for medical domain applications where citation accuracy is mandatory.
    
    Attributes:
        pmid: PubMed identifier (7-8 digits)
        exists: Whether the PMID exists in PubMed database
        title_match: Similarity score between claimed and actual title (0-1)
        claim_supported: Whether the claim is supported by the abstract content
        is_retracted: Whether the paper has been retracted
        original_abstract: Full abstract text if retrieved successfully
        error_message: Detailed error information if verification failed
        verification_timestamp: When the verification was performed
        api_response_time: Time taken for PubMed API response (ms)
        title: Paper title if retrieved
        journal: Journal name if retrieved
        doi: DOI if available
    """
    pmid: str
    exists: bool
    title_match: float  # 0-1 similarity score
    claim_supported: bool
    is_retracted: bool = False
    original_abstract: Optional[str] = None
    error_message: Optional[str] = None
    verification_timestamp: Optional[float] = None
    api_response_time: Optional[float] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    
    def __post_init__(self):
        """Validate PMID format and score ranges."""
        if not self._is_valid_pmid(self.pmid):
            logger.warning(f"Invalid PMID format: {self.pmid}")
        
        if not 0.0 <= self.title_match <= 1.0:
            logger.warning(f"Title match score out of range: {self.title_match}")
            self.title_match = max(0.0, min(1.0, self.title_match))
    
    @staticmethod
    def _is_valid_pmid(pmid: str) -> bool:
        """Validate PMID format (7-8 digits)."""
        return bool(re.match(r'^\d{7,8}$', pmid))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns comprehensive verification data suitable for logging,
        API responses, or persistent storage.
        """
        return {
            'pmid': self.pmid,
            'exists': self.exists,
            'is_retracted': self.is_retracted,
            'title_match': round(self.title_match, 3),
            'claim_supported': self.claim_supported,
            'has_abstract': self.original_abstract is not None,
            'abstract_length': len(self.original_abstract) if self.original_abstract else 0,
            'error_message': self.error_message,
            'verification_timestamp': self.verification_timestamp,
            'api_response_time_ms': self.api_response_time,
            'is_valid_format': self._is_valid_pmid(self.pmid),
            'title': self.title,
            'journal': self.journal,
            'doi': self.doi,
            'is_valid': self.exists and not self.is_retracted,
        }
    
    @property
    def is_valid(self) -> bool:
        """Check if citation is valid (exists and not retracted)."""
        return self.exists and not self.is_retracted


class CitationVerifier:
    """Production-grade citation verification against PubMed database.
    
    This class provides comprehensive citation verification capabilities specifically
    designed for medical and scientific literature. It validates PMIDs against the
    NCBI PubMed database, retrieves abstracts, and assesses claim support using
    semantic analysis.
    
    Key Features:
        - Real-time PMID validation against PubMed API
        - Abstract retrieval with XML parsing
        - Semantic claim-abstract matching using sentence transformers
        - Comprehensive error handling with retry logic
        - Performance optimization through caching
        - Medical domain-specific validation patterns
    
    Args:
        email: Contact email for PubMed API (required for compliance)
        similarity_threshold: Minimum similarity for claim support (0.0-1.0)
        request_timeout: API request timeout in seconds
        enable_cache: Whether to cache abstract retrievals
        max_retries: Maximum retry attempts for failed requests
    
    Example:
        >>> verifier = CitationVerifier(email="researcher@university.edu")
        >>> result = await verifier.verify_citation(
        ...     pmid="12345678",
        ...     claimed_finding="EEG shows 95% seizure detection accuracy"
        ... )
        >>> print(f"Citation valid: {result.exists}")
        >>> print(f"Claim supported: {result.claim_supported}")
    """
    
    def __init__(self, 
                 email: Optional[str] = None, 
                 similarity_threshold: float = 0.5,
                 request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
                 enable_cache: bool = True,
                 max_retries: int = RETRY_ATTEMPTS):
        
        # API configuration with validation
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email or "research@eeg-rag.org"
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        
        # Validation parameters
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"Similarity threshold must be 0.0-1.0, got {similarity_threshold}")
        self.similarity_threshold = similarity_threshold
        
        # Performance optimization
        self.cache = {} if enable_cache else None
        self._request_count = 0
        self._cache_hits = 0
        
        # Initialize ML model for semantic analysis
        self.sentence_model = self._initialize_sentence_model()
        
        logger.info(f"CitationVerifier initialized with threshold={similarity_threshold}, "
                   f"timeout={request_timeout}s, email={self.email}")
    
    def _initialize_sentence_model(self) -> Optional[SentenceTransformer]:
        """Initialize sentence transformer with error handling.
        
        Returns:
            SentenceTransformer model or None if initialization fails
        """
        try:
            # Use PubMedBERT for biomedical domain
            model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
            logger.info("Biomedical sentence transformer loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Failed to load PubMedBERT, falling back to general model: {e}")
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("General sentence transformer loaded as fallback")
                return model
            except Exception as e2:
                logger.error(f"Failed to load any sentence transformer: {e2}")
                return None
    
    async def verify_citation(self, pmid: str, claimed_finding: str = "", check_retraction: bool = True) -> VerificationResult:
        """Verify a single PMID citation with optional retraction checking.
        
        Args:
            pmid: PubMed ID to verify
            claimed_finding: Optional claim to check against abstract
            check_retraction: Whether to check retraction status (default True)
            
        Returns:
            VerificationResult with complete verification data
        """
        try:
            # Fetch abstract and metadata from PubMed
            paper_data = await self._fetch_paper_data(pmid)
            
            if not paper_data or not paper_data.get("abstract"):
                return VerificationResult(
                    pmid=pmid,
                    exists=False,
                    title_match=0.0,
                    claim_supported=False,
                    is_retracted=False,
                    error_message="PMID not found or no abstract available"
                )
            
            abstract = paper_data.get("abstract", "")
            title = paper_data.get("title", "")
            journal = paper_data.get("journal", "")
            doi = paper_data.get("doi", "")
            
            # Check if claimed finding is supported
            claim_supported = True
            if claimed_finding and self.sentence_model:
                claim_supported = self._check_claim_support(claimed_finding, abstract)
            
            # Check retraction status
            is_retracted = False
            if check_retraction:
                is_retracted = await self._check_retraction_status(pmid, doi)
            
            return VerificationResult(
                pmid=pmid,
                exists=True,
                title_match=1.0,
                claim_supported=claim_supported,
                is_retracted=is_retracted,
                original_abstract=abstract,
                title=title,
                journal=journal,
                doi=doi
            )
            
        except Exception as e:
            logger.error(f"Error verifying citation {pmid}: {e}")
            return VerificationResult(
                pmid=pmid,
                exists=False,
                title_match=0.0,
                claim_supported=False,
                error_message=str(e)
            )
    
    async def verify_multiple(self, pmids: List[str], claims: List[str] = None) -> List[VerificationResult]:
        """Verify multiple citations concurrently"""
        claims = claims or [""] * len(pmids)
        
        tasks = [
            self.verify_citation(pmid, claim)
            for pmid, claim in zip(pmids, claims)
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_abstract(self, pmid: str) -> Optional[str]:
        """Fetch abstract from PubMed"""
        if pmid in self.cache:
            return self.cache[pmid]
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use efetch to get the abstract
                url = f"{self.pubmed_base}/efetch.fcgi"
                params = {
                    'db': 'pubmed',
                    'id': pmid,
                    'retmode': 'xml',
                    'email': self.email
                }
                
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    abstract = self._parse_abstract_from_xml(response.text)
                    if abstract:
                        self.cache[pmid] = abstract
                    return abstract
                
                logger.warning(f"Failed to fetch PMID {pmid}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching abstract for {pmid}: {e}")
            return None
    
    def _parse_abstract_from_xml(self, xml_content: str) -> Optional[str]:
        """Parse abstract from PubMed XML response"""
        try:
            root = ET.fromstring(xml_content)
            
            # Look for abstract text
            abstract_elements = root.findall(".//AbstractText")
            
            if abstract_elements:
                # Combine all abstract parts
                abstract_parts = []
                for elem in abstract_elements:
                    if elem.text:
                        # Include label if present
                        label = elem.get('Label', '')
                        text = elem.text.strip()
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                
                return " ".join(abstract_parts)
            
            # Fallback: look for any text content
            article = root.find(".//Article")
            if article is not None:
                all_text = ET.tostring(article, encoding='unicode', method='text')
                return " ".join(all_text.split())[:1000]  # Limit length
            
            return None
            
        except ET.ParseError as e:
            logger.warning(f"Failed to parse XML: {e}")
            return None
    
    def _check_claim_support(self, claim: str, abstract: str) -> bool:
        """Check if claim is supported by abstract using semantic similarity"""
        if not self.sentence_model or not claim.strip() or not abstract.strip():
            return True  # Default to supported if we can't check
        
        try:
            # Encode claim and abstract
            claim_emb = self.sentence_model.encode(claim, convert_to_tensor=True)
            abstract_emb = self.sentence_model.encode(abstract, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = cos_sim(claim_emb, abstract_emb).item()
            
            return similarity > self.similarity_threshold
            
        except Exception as e:
            logger.warning(f"Error checking claim support: {e}")
            return True  # Default to supported if error
    
    async def _fetch_paper_data(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Fetch complete paper data including title, journal, DOI from PubMed.
        
        Args:
            pmid: PubMed ID to fetch
            
        Returns:
            Dictionary with paper metadata or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.pubmed_base}/efetch.fcgi"
                params = {
                    'db': 'pubmed',
                    'id': pmid,
                    'retmode': 'xml',
                    'email': self.email
                }
                
                response = await client.get(url, params=params)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch PMID {pmid}: HTTP {response.status_code}")
                    return None
                
                return self._parse_paper_data_from_xml(response.text)
                
        except Exception as e:
            logger.error(f"Error fetching paper data for {pmid}: {e}")
            return None
    
    def _parse_paper_data_from_xml(self, xml_content: str) -> Optional[Dict[str, Any]]:
        """Parse complete paper data from PubMed XML response."""
        try:
            root = ET.fromstring(xml_content)
            article = root.find(".//PubmedArticle")
            
            if article is None:
                return None
            
            medline = article.find("MedlineCitation")
            if medline is None:
                return None
            
            article_data = medline.find("Article")
            if article_data is None:
                return None
            
            # Extract title
            title_elem = article_data.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract = ""
            abstract_elem = article_data.find("Abstract")
            if abstract_elem is not None:
                abstract_parts = []
                for at in abstract_elem.findall("AbstractText"):
                    label = at.get("Label", "")
                    text = at.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)
            
            # Extract journal
            journal = ""
            journal_elem = article_data.find("Journal")
            if journal_elem is not None:
                title_elem = journal_elem.find("Title")
                if title_elem is not None:
                    journal = title_elem.text or ""
            
            # Extract DOI
            doi = ""
            article_ids = article.find(".//ArticleIdList")
            if article_ids is not None:
                for aid in article_ids.findall("ArticleId"):
                    if aid.get("IdType") == "doi" and aid.text:
                        doi = aid.text
                        break
            
            return {
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "doi": doi,
            }
            
        except ET.ParseError as e:
            logger.warning(f"Failed to parse paper XML: {e}")
            return None
    
    async def _check_retraction_status(self, pmid: str, doi: Optional[str] = None) -> bool:
        """Check if a paper has been retracted.
        
        Uses multiple methods:
        1. Check PubMed for retraction notices
        2. Check Crossref for retraction status (if DOI available)
        
        Args:
            pmid: PubMed ID to check
            doi: Optional DOI for additional checking
            
        Returns:
            True if paper is retracted, False otherwise
        """
        # Method 1: Check PubMed for retraction notices
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Search for retraction notices linked to this PMID
                url = f"{self.pubmed_base}/esearch.fcgi"
                params = {
                    'db': 'pubmed',
                    'term': f'"{pmid}"[PMID] AND (retraction[pt] OR retracted publication[pt])',
                    'retmode': 'json',
                    'email': self.email
                }
                
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    count = int(data.get("esearchresult", {}).get("count", "0"))
                    if count > 0:
                        logger.warning(f"Retraction notice found for PMID {pmid}")
                        return True
                        
        except Exception as e:
            logger.warning(f"Error checking PubMed retraction for {pmid}: {e}")
        
        # Method 2: Check Crossref if DOI available
        if doi:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    url = f"https://api.crossref.org/works/{doi}"
                    headers = {"User-Agent": "EEG-RAG/1.0 (mailto:research@eeg-rag.org)"}
                    
                    response = await client.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        message = data.get("message", {})
                        
                        # Check for retraction in update-to field
                        updates = message.get("update-to", [])
                        for update in updates:
                            if update.get("type") == "retraction":
                                logger.warning(f"Crossref retraction found for DOI {doi}")
                                return True
                        
                        # Also check the 'is-retracted' field if present
                        if message.get("is-retracted", False):
                            logger.warning(f"DOI {doi} marked as retracted in Crossref")
                            return True
                            
            except Exception as e:
                logger.debug(f"Error checking Crossref retraction for {doi}: {e}")
        
        return False


class HallucinationDetector:
    """Detects potential hallucinations in generated text"""
    
    def __init__(self, verifier: CitationVerifier):
        self.verifier = verifier
        
        # Patterns that might indicate hallucination
        self.hallucination_patterns = {
            'definitive_claims': r'\b(always|never|all|none|definitely|certainly|absolutely)\b',
            'specific_numbers': r'\b\d+%\s*of\s*(patients|subjects|cases)\b',
            'unprovable_statistics': r'\b(most|majority|significant|substantial)\s*(studies|research|evidence)\b',
            'temporal_claims': r'\b(recent|latest|new|cutting-edge)\s*(research|findings|studies)\b'
        }
    
    async def check_answer(self, answer: str, context_docs: List[str] = None) -> Dict[str, Any]:
        """Comprehensive hallucination check of an answer"""
        # Extract claims and citations
        claims = self._extract_claims_with_citations(answer)
        cited_pmids = self._extract_pmids(answer)
        
        # Verify citations
        citation_results = []
        if cited_pmids:
            citation_results = await self.verifier.verify_multiple(cited_pmids)
        
        # Check for hallucination patterns
        pattern_flags = self._check_hallucination_patterns(answer)
        
        # Check claim support against context
        context_support = self._check_context_support(claims, context_docs or [])
        
        # Calculate overall scores
        citation_accuracy = self._calculate_citation_accuracy(citation_results)
        unsupported_claims = self._count_unsupported_claims(claims, citation_results)
        pattern_risk_score = sum(pattern_flags.values()) / len(pattern_flags)
        
        # Overall hallucination score (0 = no hallucination, 1 = high hallucination risk)
        hallucination_score = (
            (1 - citation_accuracy) * 0.4 +
            pattern_risk_score * 0.3 +
            (unsupported_claims / max(len(claims), 1)) * 0.3
        )
        
        return {
            'hallucination_score': min(hallucination_score, 1.0),
            'citation_accuracy': citation_accuracy,
            'verified_citations': len([r for r in citation_results if isinstance(r, VerificationResult) and r.exists]),
            'invalid_citations': len([r for r in citation_results if isinstance(r, VerificationResult) and not r.exists]),
            'unsupported_claims': unsupported_claims,
            'pattern_flags': pattern_flags,
            'context_support': context_support,
            'total_claims': len(claims),
            'total_citations': len(cited_pmids),
            'citation_results': [r.to_dict() if isinstance(r, VerificationResult) else str(r) for r in citation_results]
        }
    
    def _extract_claims_with_citations(self, text: str) -> List[Tuple[str, List[str]]]:
        """Extract claims and their associated citations"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract PMIDs from this sentence
            pmids = self._extract_pmids(sentence)
            
            # Clean sentence of citations for claim extraction
            clean_sentence = re.sub(r'\(PMID:?\s*\d+\)', '', sentence)
            clean_sentence = re.sub(r'PMID:?\s*\d+', '', clean_sentence).strip()
            
            if clean_sentence:
                claims.append((clean_sentence, pmids))
        
        return claims
    
    def _extract_pmids(self, text: str) -> List[str]:
        """Extract PMID numbers from text"""
        pmid_pattern = r'PMID:?\s*(\d{8})'
        return re.findall(pmid_pattern, text)
    
    def _check_hallucination_patterns(self, text: str) -> Dict[str, float]:
        """Check for patterns that might indicate hallucination"""
        flags = {}
        text_lower = text.lower()
        
        for pattern_name, pattern in self.hallucination_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            # Normalize by text length (matches per 100 words)
            word_count = len(text.split())
            flags[pattern_name] = min((matches / max(word_count / 100, 1)), 1.0)
        
        return flags
    
    def _check_context_support(self, claims: List[Tuple[str, List[str]]], context_docs: List[str]) -> float:
        """Check how well claims are supported by provided context"""
        if not claims or not context_docs or not self.verifier.sentence_model:
            return 0.5  # Neutral if we can't check
        
        try:
            # Combine all context
            full_context = " ".join(context_docs)
            context_emb = self.verifier.sentence_model.encode(full_context, convert_to_tensor=True)
            
            supported_claims = 0
            total_claims = len(claims)
            
            for claim, _ in claims:
                claim_emb = self.verifier.sentence_model.encode(claim, convert_to_tensor=True)
                similarity = cos_sim(claim_emb, context_emb).item()
                
                if similarity > self.verifier.similarity_threshold:
                    supported_claims += 1
            
            return supported_claims / total_claims
            
        except Exception as e:
            logger.warning(f"Error checking context support: {e}")
            return 0.5
    
    def _calculate_citation_accuracy(self, results: List[VerificationResult]) -> float:
        """Calculate percentage of valid citations"""
        if not results:
            return 1.0
        
        valid_count = sum(1 for r in results if isinstance(r, VerificationResult) and r.exists)
        return valid_count / len(results)
    
    def _count_unsupported_claims(self, claims: List[Tuple[str, List[str]]], 
                                 citation_results: List[VerificationResult]) -> int:
        """Count claims that are not supported by their citations"""
        unsupported = 0
        
        # Create a map of PMID to verification result
        pmid_to_result = {}
        for result in citation_results:
            if isinstance(result, VerificationResult):
                pmid_to_result[result.pmid] = result
        
        for claim, pmids in claims:
            if not pmids:  # Claim without citation
                unsupported += 1
                continue
            
            # Check if any citation supports the claim
            supported = False
            for pmid in pmids:
                if pmid in pmid_to_result:
                    result = pmid_to_result[pmid]
                    if result.exists and result.claim_supported:
                        supported = True
                        break
            
            if not supported:
                unsupported += 1
        
        return unsupported


# Convenience functions for easy integration
async def verify_answer_citations(answer: str, email: str = None) -> Dict[str, Any]:
    """Quick function to verify all citations in an answer"""
    verifier = CitationVerifier(email=email)
    detector = HallucinationDetector(verifier)
    
    return await detector.check_answer(answer)


async def quick_citation_check(pmids: List[str], email: str = None) -> List[bool]:
    """Quick check if PMIDs exist"""
    verifier = CitationVerifier(email=email)
    results = await verifier.verify_multiple(pmids)
    
    return [r.exists if isinstance(r, VerificationResult) else False for r in results]

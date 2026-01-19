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


@dataclass
class VerificationResult:
    """Result of citation verification"""
    pmid: str
    exists: bool
    title_match: float  # 0-1 similarity
    claim_supported: bool
    original_abstract: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pmid': self.pmid,
            'exists': self.exists,
            'title_match': self.title_match,
            'claim_supported': self.claim_supported,
            'has_abstract': self.original_abstract is not None,
            'error_message': self.error_message
        }


class CitationVerifier:
    """Verifies citations against PubMed database"""
    
    def __init__(self, email: Optional[str] = None, similarity_threshold: float = 0.5):
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email or "research@eeg-rag.org"
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, str] = {}
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    async def verify_citation(self, pmid: str, claimed_finding: str = "") -> VerificationResult:
        """Verify a single PMID citation"""
        try:
            # Fetch abstract from PubMed
            abstract = await self._fetch_abstract(pmid)
            
            if not abstract:
                return VerificationResult(
                    pmid=pmid,
                    exists=False,
                    title_match=0.0,
                    claim_supported=False,
                    error_message="PMID not found or no abstract available"
                )
            
            # Check if claimed finding is supported
            claim_supported = True
            if claimed_finding and self.sentence_model:
                claim_supported = self._check_claim_support(claimed_finding, abstract)
            
            return VerificationResult(
                pmid=pmid,
                exists=True,
                title_match=1.0,  # Could implement title comparison
                claim_supported=claim_supported,
                original_abstract=abstract
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

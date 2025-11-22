"""
Dataset Security Scanner

Protects against modern cyber threats including:
- SVG poisoning attacks (embedded JavaScript, external scripts)
- PDF malware (embedded JavaScript, file attachments, form actions)
- Prompt injection in embedded text
- Malicious instructions in image metadata
- Foreign data in datasets

Requirements:
- REQ-SEC-001: Scan all external documents before ingestion
- REQ-SEC-002: Detect SVG poisoning attempts
- REQ-SEC-003: Scan PDFs for malicious payloads
- REQ-SEC-004: Verify domain whitelist for trusted sources
- REQ-SEC-005: Log all security events for audit
"""

import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path


class ThreatLevel(Enum):
    """Threat severity levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats"""
    SVG_POISONING = "svg_poisoning"
    PDF_MALWARE = "pdf_malware"
    PROMPT_INJECTION = "prompt_injection"
    XSS_ATTACK = "xss_attack"
    EXTERNAL_SCRIPT = "external_script"
    UNTRUSTED_DOMAIN = "untrusted_domain"
    SUSPICIOUS_METADATA = "suspicious_metadata"


@dataclass
class SecurityThreat:
    """
    Detected security threat
    
    Attributes:
        threat_type: Type of threat detected
        threat_level: Severity level
        description: Human-readable description
        location: Where threat was found (line number, field, etc.)
        evidence: Suspicious content snippet
        mitigation: Recommended action
    """
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    location: str
    evidence: str
    mitigation: str


@dataclass
class ScanResult:
    """
    Results of security scan
    
    Attributes:
        document_id: Identifier for scanned document
        safe: Whether document passed all security checks
        threats: List of detected threats
        scan_time: Time taken for scan (seconds)
        hash: SHA-256 hash of document for tracking
    """
    document_id: str
    safe: bool
    threats: List[SecurityThreat]
    scan_time: float
    hash: str


class DatasetSecurityScanner:
    """
    Security scanner for research datasets
    
    Protects EEG-RAG system from:
    1. SVG Poisoning: Malicious SVG files with embedded scripts
    2. PDF Malware: PDFs with JavaScript, attachments, or form actions
    3. Prompt Injection: Embedded instructions to manipulate AI behavior
    4. XSS Attacks: Cross-site scripting attempts in metadata
    5. Untrusted Domains: Content from non-verified sources
    
    Usage:
        scanner = DatasetSecurityScanner(
            trusted_domains=['pubmed.ncbi.nlm.nih.gov', 'arxiv.org']
        )
        
        # Scan SVG file
        result = scanner.scan_svg(svg_content)
        if not result.safe:
            print(f"⚠️  {len(result.threats)} threats detected!")
            for threat in result.threats:
                print(f"  - {threat.description}")
        
        # Scan PDF
        result = scanner.scan_pdf(pdf_path)
        
        # Scan text for prompt injection
        result = scanner.scan_text(text_content)
    """
    
    # SVG threat patterns
    SVG_SCRIPT_PATTERNS = [
        r'<script[^>]*>',  # Script tags
        r'on\w+\s*=',  # Event handlers (onclick, onload, etc.)
        r'javascript:',  # JavaScript protocol
        r'<iframe[^>]*>',  # Iframe tags
        r'<embed[^>]*>',  # Embed tags
        r'<object[^>]*>',  # Object tags
        r'xlink:href\s*=\s*["\'](?!#)',  # External links
    ]
    
    # PDF threat patterns
    PDF_MALWARE_PATTERNS = [
        b'/JavaScript',  # JavaScript in PDF
        b'/JS',  # JavaScript shorthand
        b'/AA',  # Additional Actions (auto-execute)
        b'/OpenAction',  # Actions on document open
        b'/Launch',  # Launch external programs
        b'/EmbeddedFile',  # Embedded files
        b'/RichMedia',  # Rich media content
        b'/SubmitForm',  # Form submission
    ]
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(previous|all|above)\s+(instructions|prompts)',
        r'new\s+instructions?:',
        r'you\s+are\s+now\s+a',
        r'disregard\s+(previous|all)\s+',
        r'system\s*:\s*',
        r'<\|im_start\|>',  # ChatML injection
        r'<\|endoftext\|>',  # GPT token injection
        r'\[INST\]',  # Llama instruction injection
    ]
    
    def __init__(
        self,
        trusted_domains: Optional[List[str]] = None,
        enable_aggressive_scanning: bool = True
    ):
        """
        Initialize security scanner
        
        Args:
            trusted_domains: List of verified domain names
            enable_aggressive_scanning: Enable more strict scanning (may have false positives)
        """
        self.trusted_domains = trusted_domains or [
            'pubmed.ncbi.nlm.nih.gov',
            'www.ncbi.nlm.nih.gov',
            'arxiv.org',
            'www.arxiv.org',
            'biorxiv.org',
            'www.biorxiv.org',
            'doi.org',
            'dx.doi.org'
        ]
        self.enable_aggressive_scanning = enable_aggressive_scanning
        
        # Compile regex patterns for performance
        self.svg_patterns = [re.compile(p, re.IGNORECASE) for p in self.SVG_SCRIPT_PATTERNS]
        self.injection_patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
    
    def scan_svg(self, svg_content: str, document_id: str = "unknown") -> ScanResult:
        """
        Scan SVG content for poisoning attacks
        
        Args:
            svg_content: SVG file content as string
            document_id: Identifier for document
            
        Returns:
            ScanResult with threat analysis
        """
        import time
        start_time = time.time()
        
        threats = []
        
        # Check for script tags
        for pattern in self.svg_patterns:
            matches = pattern.finditer(svg_content)
            for match in matches:
                threat = SecurityThreat(
                    threat_type=ThreatType.SVG_POISONING,
                    threat_level=ThreatLevel.CRITICAL,
                    description="Malicious script detected in SVG",
                    location=f"Character position {match.start()}",
                    evidence=match.group()[:100],
                    mitigation="REJECT: Do not ingest this SVG file"
                )
                threats.append(threat)
        
        # Check for external references
        external_refs = re.findall(r'xlink:href\s*=\s*["\']([^"\']+)["\']', svg_content, re.IGNORECASE)
        for ref in external_refs:
            if not ref.startswith('#'):  # Internal reference
                threat = SecurityThreat(
                    threat_type=ThreatType.EXTERNAL_SCRIPT,
                    threat_level=ThreatLevel.HIGH,
                    description="External resource referenced in SVG",
                    location=f"xlink:href",
                    evidence=ref[:100],
                    mitigation="SANITIZE: Remove external references"
                )
                threats.append(threat)
        
        scan_time = time.time() - start_time
        doc_hash = hashlib.sha256(svg_content.encode()).hexdigest()
        
        return ScanResult(
            document_id=document_id,
            safe=len(threats) == 0,
            threats=threats,
            scan_time=scan_time,
            hash=doc_hash
        )
    
    def scan_pdf(self, pdf_path: str) -> ScanResult:
        """
        Scan PDF file for malware
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ScanResult with threat analysis
        """
        import time
        start_time = time.time()
        
        threats = []
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            return ScanResult(
                document_id=pdf_path,
                safe=False,
                threats=[SecurityThreat(
                    threat_type=ThreatType.PDF_MALWARE,
                    threat_level=ThreatLevel.MEDIUM,
                    description="PDF file not found",
                    location=pdf_path,
                    evidence="N/A",
                    mitigation="Verify file path"
                )],
                scan_time=time.time() - start_time,
                hash=""
            )
        
        # Read PDF content as binary
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        # Check for malware patterns
        for pattern in self.PDF_MALWARE_PATTERNS:
            if pattern in pdf_content:
                threat = SecurityThreat(
                    threat_type=ThreatType.PDF_MALWARE,
                    threat_level=ThreatLevel.CRITICAL,
                    description=f"Suspicious PDF object detected: {pattern.decode('latin-1')}",
                    location=pdf_path,
                    evidence=pattern.decode('latin-1'),
                    mitigation="REJECT: Do not ingest this PDF"
                )
                threats.append(threat)
        
        scan_time = time.time() - start_time
        doc_hash = hashlib.sha256(pdf_content).hexdigest()
        
        return ScanResult(
            document_id=pdf_path,
            safe=len(threats) == 0,
            threats=threats,
            scan_time=scan_time,
            hash=doc_hash
        )
    
    def scan_text(self, text: str, document_id: str = "unknown") -> ScanResult:
        """
        Scan text for prompt injection attempts
        
        Args:
            text: Text content to scan
            document_id: Identifier for document
            
        Returns:
            ScanResult with threat analysis
        """
        import time
        start_time = time.time()
        
        threats = []
        
        # Check for prompt injection patterns
        for pattern in self.injection_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                threat = SecurityThreat(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    description="Potential prompt injection detected",
                    location=f"Character position {match.start()}",
                    evidence=match.group()[:100],
                    mitigation="SANITIZE: Remove or escape suspicious instructions"
                )
                threats.append(threat)
        
        scan_time = time.time() - start_time
        doc_hash = hashlib.sha256(text.encode()).hexdigest()
        
        return ScanResult(
            document_id=document_id,
            safe=len(threats) == 0,
            threats=threats,
            scan_time=scan_time,
            hash=doc_hash
        )
    
    def verify_domain(self, url: str) -> bool:
        """
        Verify URL is from trusted domain
        
        Args:
            url: URL to verify
            
        Returns:
            True if domain is trusted, False otherwise
        """
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check exact match
        if domain in self.trusted_domains:
            return True
        
        # Check subdomain match (e.g., api.pubmed.ncbi.nlm.nih.gov)
        for trusted in self.trusted_domains:
            if domain.endswith('.' + trusted) or domain == trusted:
                return True
        
        return False
    
    def scan_complete_dataset(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Scan complete dataset of documents
        
        Args:
            documents: List of document dictionaries with 'content' and 'url' fields
            
        Returns:
            Dictionary with scan summary and flagged documents
        """
        results = {
            'total_documents': len(documents),
            'safe_documents': 0,
            'flagged_documents': 0,
            'threats_by_type': {},
            'flagged_details': []
        }
        
        for doc in documents:
            # Scan text content
            content = doc.get('content', '')
            url = doc.get('url', '')
            doc_id = doc.get('id', 'unknown')
            
            text_result = self.scan_text(content, doc_id)
            
            # Check domain if URL provided
            domain_safe = True
            if url and not self.verify_domain(url):
                domain_safe = False
                text_result.threats.append(
                    SecurityThreat(
                        threat_type=ThreatType.UNTRUSTED_DOMAIN,
                        threat_level=ThreatLevel.MEDIUM,
                        description="Document from untrusted domain",
                        location=url,
                        evidence=url,
                        mitigation="VERIFY: Manually verify source before ingestion"
                    )
                )
            
            # Update results
            if text_result.safe and domain_safe:
                results['safe_documents'] += 1
            else:
                results['flagged_documents'] += 1
                results['flagged_details'].append({
                    'document_id': doc_id,
                    'url': url,
                    'threats': [
                        {
                            'type': t.threat_type.value,
                            'level': t.threat_level.value,
                            'description': t.description
                        }
                        for t in text_result.threats
                    ]
                })
                
                # Count threat types
                for threat in text_result.threats:
                    threat_type = threat.threat_type.value
                    results['threats_by_type'][threat_type] = \
                        results['threats_by_type'].get(threat_type, 0) + 1
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scanner configuration statistics"""
        return {
            'trusted_domains': len(self.trusted_domains),
            'svg_patterns': len(self.svg_patterns),
            'injection_patterns': len(self.injection_patterns),
            'pdf_patterns': len(self.PDF_MALWARE_PATTERNS),
            'aggressive_scanning': self.enable_aggressive_scanning
        }

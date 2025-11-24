"""
EEG Corpus Builder
Fetches and builds a corpus of EEG research papers from PubMed

Requirements:
- REQ-CORPUS-001: Fetch 1000+ EEG papers from PubMed
- REQ-CORPUS-002: Filter by relevance (EEG, electroencephalography keywords)
- REQ-CORPUS-003: Extract and store metadata (PMID, DOI, authors, etc.)
- REQ-CORPUS-004: Save papers in structured format (JSON/JSONL)
- REQ-CORPUS-005: Handle rate limiting and API errors
- REQ-CORPUS-006: Progress tracking and resumable downloads
- REQ-CORPUS-007: Deduplication by PMID
- REQ-CORPUS-008: Export corpus statistics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json
import asyncio
from datetime import datetime
import logging


@dataclass
class Paper:
    """Research paper with metadata"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'journal': self.journal,
            'year': self.year,
            'doi': self.doi,
            'keywords': self.keywords,
            'mesh_terms': self.mesh_terms
        }


class EEGCorpusBuilder:
    """
    Builds a corpus of EEG research papers

    Integrates with PubMed E-utilities to fetch papers.
    Saves corpus in JSONL format for easy processing.
    """

    # EEG-related search terms
    EEG_QUERIES = [
        'electroencephalography',
        'EEG',
        'brain waves',
        'neural oscillations',
        'event-related potentials',
        'ERP',
        'P300',
        'alpha waves',
        'theta oscillations',
        'epilepsy EEG',
        'sleep EEG',
        'cognitive EEG'
    ]

    def __init__(
        self,
        output_dir: Path,
        target_count: int = 1000,
        use_mock: bool = True
    ):
        """
        Initialize corpus builder

        Args:
            output_dir: Directory to save corpus
            target_count: Target number of papers to fetch
            use_mock: Use mock data (for testing without API calls)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_count = target_count
        self.use_mock = use_mock

        # Tracking
        self.papers: Dict[str, Paper] = {}
        self.failed_pmids: Set[str] = set()

        # Statistics
        self.stats = {
            'papers_fetched': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'total_time': 0.0
        }

        self.logger = logging.getLogger(__name__)

    async def build_corpus(self) -> Dict[str, Any]:
        """
        Build EEG corpus by fetching papers from PubMed

        Returns:
            Dictionary with corpus statistics
        """
        import time
        start_time = time.time()

        self.logger.info(f"Building EEG corpus with target of {self.target_count} papers")

        if self.use_mock:
            # Generate mock corpus for testing
            await self._generate_mock_corpus()
        else:
            # Fetch real papers from PubMed
            await self._fetch_from_pubmed()

        # Save corpus to disk
        self._save_corpus()

        # Update statistics
        self.stats['total_time'] = time.time() - start_time
        self.stats['papers_fetched'] = len(self.papers)

        self.logger.info(f"Corpus building complete: {len(self.papers)} papers")

        return self.get_statistics()

    async def _generate_mock_corpus(self):
        """Generate mock corpus for testing"""
        self.logger.info("Generating mock corpus")

        # Create mock papers
        for i in range(min(self.target_count, 100)):  # Limit mock data
            pmid = f"mock_{i:08d}"

            paper = Paper(
                pmid=pmid,
                title=f"EEG Study {i}: {self._random_eeg_topic()}",
                abstract=self._generate_mock_abstract(i),
                authors=self._generate_mock_authors(),
                journal=self._random_journal(),
                year=2020 + (i % 5),
                doi=f"10.1234/mock.{i}",
                keywords=self._random_keywords(),
                mesh_terms=self._random_mesh_terms()
            )

            self.papers[pmid] = paper

            if i % 10 == 0:
                await asyncio.sleep(0.01)  # Simulate async processing

    async def _fetch_from_pubmed(self):
        """Fetch papers from PubMed (placeholder for real implementation)"""
        # This would integrate with the WebSearchAgent
        # For now, raise error to indicate real implementation needed
        raise NotImplementedError(
            "Real PubMed fetching not yet implemented. "
            "Use use_mock=True for testing or integrate with WebSearchAgent."
        )

    def _save_corpus(self):
        """Save corpus to disk in JSONL format"""
        output_file = self.output_dir / f"eeg_corpus_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in self.papers.values():
                f.write(json.dumps(paper.to_dict()) + '\n')

        self.logger.info(f"Corpus saved to {output_file}")

        # Save metadata
        metadata_file = self.output_dir / "corpus_metadata.json"
        metadata = {
            'created_at': datetime.now().isoformat(),
            'paper_count': len(self.papers),
            'target_count': self.target_count,
            'statistics': self.stats
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def load_corpus(self, corpus_file: Path) -> List[Paper]:
        """Load corpus from JSONL file"""
        papers = []

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                paper = Paper(**data)
                papers.append(paper)

        return papers

    def get_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        return {
            'total_papers': len(self.papers),
            'papers_fetched': self.stats['papers_fetched'],
            'duplicates_skipped': self.stats['duplicates_skipped'],
            'errors': self.stats['errors'],
            'total_time': self.stats['total_time'],
            'unique_pmids': len(self.papers),
            'failed_pmids': len(self.failed_pmids)
        }

    # Helper methods for mock data generation
    def _random_eeg_topic(self) -> str:
        """Generate random EEG topic"""
        import random
        topics = [
            "Epilepsy Detection Using Alpha Waves",
            "P300 Responses in Cognitive Tasks",
            "Theta Oscillations in Memory Formation",
            "Sleep Stage Classification via EEG",
            "Event-Related Potentials in Depression",
            "Beta Waves and Motor Control",
            "Gamma Oscillations in Attention",
            "Delta Waves in Deep Sleep",
            "EEG Biomarkers for Alzheimer's Disease",
            "Seizure Prediction Using Machine Learning"
        ]
        return random.choice(topics)

    def _generate_mock_abstract(self, seed: int) -> str:
        """Generate mock abstract"""
        templates = [
            f"Background: This study investigates EEG patterns in {seed} subjects with a focus on neural oscillations "
            f"and their relationship to cognitive performance. Electroencephalography (EEG) is a non-invasive technique "
            f"that measures electrical activity in the brain. Methods: We recorded continuous EEG from participants "
            f"during both resting state and active task conditions, analyzing frequency bands from delta to gamma. "
            f"We found significant correlations between theta power in frontal regions and cognitive performance measures. "
            f"Results: Statistical analysis revealed distinct patterns of neural oscillations associated with different "
            f"cognitive states. These findings suggest potential biomarkers for clinical applications in diagnosis "
            f"and monitoring of neurological conditions. Conclusion: EEG analysis provides valuable insights into "
            f"brain function and has significant implications for understanding neural mechanisms underlying cognition.",

            f"Background: EEG analysis is crucial for understanding brain function and diagnosing neurological disorders. "
            f"This research examines the utility of quantitative EEG measures as biomarkers for brain health. "
            f"Methods: We recorded from {seed + 10} participants during rest and multiple task conditions, including "
            f"memory tasks, attention tasks, and motor tasks. High-density EEG with 64 channels was used to capture "
            f"spatial patterns of brain activity. Signal processing included artifact removal, frequency decomposition, "
            f"and connectivity analysis. Results: Significant differences in alpha asymmetry were observed between "
            f"groups, particularly in frontal and parietal regions. Beta oscillations showed task-specific modulation "
            f"during motor preparation and execution. Gamma band activity correlated with attention performance. "
            f"Conclusion: These EEG markers show promise for diagnosis and monitoring of brain function in clinical settings."
        ]
        import random
        return random.choice(templates)

    def _generate_mock_authors(self) -> List[str]:
        """Generate mock author list"""
        import random
        first_names = ['John', 'Jane', 'Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']

        num_authors = random.randint(2, 5)
        authors = []
        for _ in range(num_authors):
            author = f"{random.choice(last_names)} {random.choice(first_names)[0]}"
            authors.append(author)

        return authors

    def _random_journal(self) -> str:
        """Random journal name"""
        import random
        journals = [
            'Clinical Neurophysiology',
            'Brain',
            'Neurology',
            'Epilepsia',
            'Sleep',
            'Journal of Neuroscience',
            'NeuroImage',
            'Biological Psychiatry'
        ]
        return random.choice(journals)

    def _random_keywords(self) -> List[str]:
        """Random keywords"""
        import random
        all_keywords = [
            'EEG', 'electroencephalography', 'neural oscillations',
            'alpha waves', 'theta waves', 'epilepsy', 'sleep', 'cognition',
            'biomarkers', 'machine learning', 'seizure detection'
        ]
        return random.sample(all_keywords, k=random.randint(3, 6))

    def _random_mesh_terms(self) -> List[str]:
        """Random MeSH terms"""
        import random
        mesh_terms = [
            'Electroencephalography',
            'Brain Waves',
            'Epilepsy',
            'Sleep Stages',
            'Cognitive Function',
            'Neural Networks',
            'Biomarkers'
        ]
        return random.sample(mesh_terms, k=random.randint(2, 4))

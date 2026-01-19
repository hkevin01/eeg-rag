# Testing Standards for EEG-RAG

## Test Organization
- Unit tests in tests/unit/ mirroring src/ structure
- Integration tests in tests/integration/
- Evaluation benchmarks in tests/evaluation/
- Fixtures in tests/conftest.py

## Test Coverage Requirements
- Minimum 85% coverage for core/ and agents/ directories
- 100% coverage for verification and citation modules
- All public methods must have at least one test

## Mocking Guidelines
- Mock external APIs (PubMed, OpenAI, Neo4j) in unit tests
- Use pytest-asyncio for async test functions
- Create reusable fixtures for common test data

## EEG-Specific Test Data
Always include test cases for:
- EEG terminology (electrode names, frequency bands, ERP components)
- Edge cases (malformed PMIDs, empty abstracts, non-English text)
- Clinical scenarios (seizure detection, sleep staging, BCI queries)

## Example Test Structure
```python
import pytest
from unittest.mock import AsyncMock, patch
from eeg_rag.agents.local_agent import LocalDataAgent
from eeg_rag.models import Query, RetrievalResult

@pytest.fixture
def sample_eeg_documents():
    return [
        {"content": "P300 amplitude decreased in patients with schizophrenia...", 
         "pmid": "12345678"},
        {"content": "Theta oscillations (4-8 Hz) increased during memory encoding...",
         "pmid": "23456789"},
    ]

@pytest.fixture
def local_agent(sample_eeg_documents):
    agent = LocalDataAgent(name="test", agent_id="test-001")
    agent.add_documents(sample_eeg_documents)
    return agent

class TestLocalDataAgent:
    @pytest.mark.asyncio
    async def test_retrieves_relevant_eeg_documents(self, local_agent):
        query = Query(text="P300 in schizophrenia")
        results = await local_agent.execute(query)
        
        assert len(results) > 0
        assert any("P300" in r.content for r in results)
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_handles_electrode_terminology(self, local_agent):
        """Verify agent correctly handles 10-20 system electrode names."""
        query = Query(text="Fp1 Fp2 frontal activity")
        results = await local_agent.execute(query)
        assert results is not None
    
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_results(self, local_agent):
        query = Query(text="")
        results = await local_agent.execute(query)
        assert results == []

    @pytest.mark.parametrize("frequency_band,expected_hz", [
        ("delta", "0.5-4"),
        ("theta", "4-8"),
        ("alpha", "8-13"),
        ("beta", "13-30"),
        ("gamma", "30-100"),
    ])
    @pytest.mark.asyncio
    async def test_frequency_band_queries(self, local_agent, frequency_band, expected_hz):
        query = Query(text=f"{frequency_band} oscillations")
        results = await local_agent.execute(query)
        # Should understand frequency band terminology
        assert results is not None
```
"""
Integration tests for web UI components.

Tests the interaction between different components,
end-to-end workflows, and data flow between modules.
"""

import pytest
import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch


class TestBenchmarkWorkflow:
    """Integration tests for benchmark workflow."""
    
    @pytest.mark.asyncio
    async def test_full_benchmark_workflow(self, sample_ground_truth_csv):
        """Test complete benchmark workflow from CSV to results."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        # Initialize benchmark
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["architecture_type", "domain", "best_accuracy"],
            strict_matching=False
        )
        
        # Run evaluation
        results = await benchmark.evaluate_extraction_accuracy(max_papers=5)
        
        # Verify results structure
        assert results.total_papers == 5
        assert results.papers_evaluated == 5
        assert len(results.paper_results) == 5
        
        # Verify each paper result
        for paper in results.paper_results:
            assert paper.paper_id is not None
            assert paper.title is not None
            assert 0 <= paper.overall_accuracy <= 1.0
            assert len(paper.field_results) > 0
            
    @pytest.mark.asyncio
    async def test_benchmark_to_dataframe_workflow(self, sample_ground_truth_csv):
        """Test benchmark results to DataFrame conversion."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["architecture_type", "domain"]
        )
        
        results = await benchmark.evaluate_extraction_accuracy(max_papers=3)
        df = results.to_dataframe()
        
        # Verify DataFrame structure
        assert len(df) == 3
        assert "paper_id" in df.columns
        assert "accuracy" in df.columns
        
        # Verify data integrity
        for i, paper in enumerate(results.paper_results):
            assert df.iloc[i]["paper_id"] == paper.paper_id
            assert df.iloc[i]["accuracy"] == pytest.approx(paper.overall_accuracy)


class TestQueryWorkflow:
    """Integration tests for query workflow."""
    
    @pytest.mark.asyncio
    async def test_full_query_workflow(self, query_engine):
        """Test complete query workflow."""
        from eeg_rag.web_ui.app import QueryResult
        
        # Execute query
        query = "What are the best CNNs for EEG seizure detection?"
        result = await query_engine.query(query, max_sources=5)
        
        # Verify result structure
        assert isinstance(result, QueryResult)
        assert result.query == query
        assert len(result.response) > 100
        assert len(result.sources) > 0
        assert len(result.citations) > 0
        
    @pytest.mark.asyncio
    async def test_query_and_citation_linking(self, query_engine):
        """Test that query results have linked citations."""
        result = await query_engine.query("seizure detection", max_sources=3)
        
        # All source PMIDs should be in citations
        source_pmids = {s["pmid"] for s in result.sources}
        citation_pmids = {c.replace("PMID:", "") for c in result.citations}
        
        for pmid in source_pmids:
            assert pmid in citation_pmids


class TestDataExtraction:
    """Integration tests for data extraction pipeline."""
    
    def test_architecture_extraction_pipeline(self, benchmark_instance, sample_paper_texts):
        """Test architecture extraction from sample texts."""
        expected_results = {
            "seizure_cnn": "CNN",
            "sleep_lstm": "RNN",
            "bci_hybrid": "CNN+RNN",
            "emotion_ae": "AE",
            "cognitive_dbn": "DBN"
        }
        
        for text_key, expected_arch in expected_results.items():
            text = sample_paper_texts[text_key]
            result = benchmark_instance._extract_architecture(text)
            assert result == expected_arch, f"Failed for {text_key}: got {result}"
            
    def test_domain_extraction_pipeline(self, benchmark_instance, sample_paper_texts):
        """Test domain extraction from sample texts."""
        expected_results = {
            "seizure_cnn": "Epilepsy",
            "sleep_lstm": "Sleep",
            "bci_hybrid": "BCI",
            "emotion_ae": "Emotion",
            "cognitive_dbn": "Cognitive"
        }
        
        for text_key, expected_domain in expected_results.items():
            text = sample_paper_texts[text_key]
            result = benchmark_instance._extract_domain(text)
            assert result == expected_domain, f"Failed for {text_key}: got {result}"
            
    def test_accuracy_extraction_pipeline(self, benchmark_instance, sample_paper_texts):
        """Test accuracy extraction from sample texts."""
        for text_key, text in sample_paper_texts.items():
            result = benchmark_instance._extract_accuracy(text)
            # All sample texts should have accuracy values
            assert result is not None, f"No accuracy found in {text_key}"
            assert 70 <= result <= 100


class TestFieldStatistics:
    """Integration tests for field statistics."""
    
    def test_field_statistics_generation(self, benchmark_instance):
        """Test field statistics generation."""
        stats = benchmark_instance.get_field_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0
        
        # Verify statistics columns
        assert "field" in stats.columns
        assert "non_null_count" in stats.columns
        assert "fill_rate" in stats.columns
        
    def test_field_statistics_accuracy(self, sample_ground_truth_csv):
        """Test field statistics accuracy."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        benchmark = SystematicReviewBenchmark(str(sample_ground_truth_csv))
        stats = benchmark.get_field_statistics()
        
        # All fields should have some fill rate
        for _, row in stats.iterrows():
            assert row["fill_rate"] >= 0
            assert row["non_null_count"] >= 0


class TestErrorRecovery:
    """Integration tests for error recovery."""
    
    @pytest.mark.asyncio
    async def test_benchmark_continues_on_missing_fields(self, sample_ground_truth_csv):
        """Test that benchmark continues when fields are missing."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        # Request non-existent field
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["architecture_type", "nonexistent_field"]
        )
        
        # Should not raise
        results = await benchmark.evaluate_extraction_accuracy(max_papers=2)
        assert results is not None
        
    @pytest.mark.asyncio
    async def test_benchmark_handles_empty_values(self, sample_ground_truth_csv):
        """Test benchmark handling of empty/null values."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        import pandas as pd
        
        # Modify CSV to have some empty values
        df = pd.read_csv(sample_ground_truth_csv)
        df.loc[0, "Architecture (clean)"] = None
        df.to_csv(sample_ground_truth_csv, index=False)
        
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["architecture_type"]
        )
        
        results = await benchmark.evaluate_extraction_accuracy(max_papers=3)
        assert results is not None


class TestPerformance:
    """Integration tests for performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_benchmark_timing(self, benchmark_instance):
        """Test that benchmark tracks timing."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=5)
        
        # Total time should be tracked
        assert results.extraction_time_total_ms > 0
        
        # Individual paper times should sum approximately to total
        individual_sum = sum(p.extraction_time_ms for p in results.paper_results)
        # Allow for some overhead
        assert individual_sum <= results.extraction_time_total_ms * 1.5
        
    @pytest.mark.asyncio
    async def test_query_timing(self, query_engine):
        """Test that query tracks timing."""
        result = await query_engine.query("test query")
        
        assert result.processing_time_ms > 0


class TestDataConsistency:
    """Integration tests for data consistency."""
    
    @pytest.mark.asyncio
    async def test_paper_count_consistency(self, benchmark_instance):
        """Test paper count consistency across results."""
        max_papers = 4
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=max_papers)
        
        assert results.papers_evaluated == max_papers
        assert len(results.paper_results) == max_papers
        
    @pytest.mark.asyncio
    async def test_accuracy_range_consistency(self, benchmark_instance):
        """Test accuracy values are in valid range."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=5)
        
        # Overall accuracy
        assert 0 <= results.overall_accuracy <= 1.0
        
        # Per-field accuracy
        for field, acc in results.per_field_accuracy.items():
            assert 0 <= acc <= 1.0, f"Invalid accuracy for {field}: {acc}"
            
        # Per-paper accuracy
        for paper in results.paper_results:
            assert 0 <= paper.overall_accuracy <= 1.0


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_pipeline(self, sample_ground_truth_csv, query_engine):
        """Test complete analysis from data load to results."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        # Step 1: Load and analyze ground truth
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["architecture_type", "domain", "best_accuracy"]
        )
        
        # Step 2: Run benchmark
        benchmark_results = await benchmark.evaluate_extraction_accuracy(max_papers=5)
        
        # Step 3: Get field statistics
        field_stats = benchmark.get_field_statistics()
        
        # Step 4: Run a query based on findings
        top_domain = benchmark.ground_truth_df["Domain 1"].value_counts().index[0]
        query_result = await query_engine.query(f"deep learning for {top_domain}")
        
        # Verify complete pipeline
        assert benchmark_results.papers_evaluated == 5
        assert len(field_stats) > 0
        assert query_result.response is not None
        
    @pytest.mark.asyncio
    async def test_multiple_benchmark_runs(self, sample_ground_truth_csv):
        """Test multiple benchmark runs produce consistent results."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["architecture_type"]
        )
        
        # Run multiple times
        results1 = await benchmark.evaluate_extraction_accuracy(max_papers=5)
        results2 = await benchmark.evaluate_extraction_accuracy(max_papers=5)
        
        # Results should be consistent
        assert results1.papers_evaluated == results2.papers_evaluated
        assert results1.overall_accuracy == results2.overall_accuracy
        
        for field in results1.per_field_accuracy:
            assert results1.per_field_accuracy[field] == results2.per_field_accuracy[field]


class TestSessionIntegration:
    """Tests for session state integration."""
    
    def test_session_state_persistence(self, mock_streamlit, sample_benchmark_result):
        """Test session state persists across function calls."""
        mock_streamlit.session_state["benchmark_results"] = sample_benchmark_result
        mock_streamlit.session_state["query_history"] = []
        
        # Simulate adding a query result
        from eeg_rag.web_ui.app import QueryResult
        
        result = QueryResult(
            query="test",
            response="response",
            sources=[],
            citations=[],
            confidence=0.8,
            processing_time_ms=100,
            timestamp="2024-01-01T00:00:00"
        )
        
        mock_streamlit.session_state["query_history"].append(result)
        
        # Verify both persisted
        assert mock_streamlit.session_state["benchmark_results"] is not None
        assert len(mock_streamlit.session_state["query_history"]) == 1


class TestCrossComponentValidation:
    """Tests for cross-component data validation."""
    
    def test_extraction_result_to_paper_result(self, sample_extraction_result):
        """Test ExtractionResult integration into PaperExtractionResult."""
        from eeg_rag.web_ui.app import PaperExtractionResult
        
        paper_result = PaperExtractionResult(
            paper_id="test_001",
            title="Test Paper",
            year=2021,
            field_results={"architecture_type": sample_extraction_result},
            overall_accuracy=1.0
        )
        
        assert len(paper_result.field_results) == 1
        assert paper_result.get_correct_fields() == ["architecture_type"]
        assert paper_result.get_incorrect_fields() == []
        
    def test_paper_results_to_benchmark_result(self, sample_paper_extraction_result):
        """Test PaperExtractionResult integration into BenchmarkResult."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmarkResult
        
        benchmark_result = SystematicReviewBenchmarkResult(
            total_papers=1,
            papers_evaluated=1,
            fields_evaluated=["architecture_type", "domain"],
            per_field_accuracy={"architecture_type": 1.0, "domain": 0.0},
            overall_accuracy=0.5,
            paper_results=[sample_paper_extraction_result],
            extraction_time_total_ms=15.3,
            timestamp="2024-01-01"
        )
        
        df = benchmark_result.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["accuracy"] == 0.5

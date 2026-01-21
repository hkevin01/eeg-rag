"""
Tests for systematic review module.
"""

import pytest
from pathlib import Path

from eeg_rag.review.extractor import SystematicReviewExtractor, ExtractionField
from eeg_rag.review.comparator import ReproducibilityScorer


def test_extraction_field():
    """Test ExtractionField dataclass."""
    field = ExtractionField(
        name="test_field",
        description="Test description",
        field_type="string",
        required=True
    )
    assert field.name == "test_field"
    assert field.field_type == "string"
    assert field.required is True


def test_reproducibility_scorer():
    """Test reproducibility scoring."""
    scorer = ReproducibilityScorer()
    
    paper_with_code = {
        "code_available": "GitHub: https://github.com/example/repo",
        "dataset_name": "CHB-MIT"
    }
    
    score, category, justifications = scorer.score_paper(paper_with_code)
    
    assert score >= 15  # GitHub (10) + public dataset (8) = 18
    assert category == "Fully Reproducible"
    assert any("Public code repository" in j for j in justifications)
    assert any("public dataset" in j.lower() for j in justifications)


def test_reproducibility_scorer_no_code():
    """Test reproducibility scoring for paper without code."""
    scorer = ReproducibilityScorer()
    
    paper_no_code = {
        "code_available": "Not available",
        "dataset_name": "Private clinical data"
    }
    
    score, category, justifications = scorer.score_paper(paper_no_code)
    
    assert score < 10
    assert category in ["Not Reproducible", "Limited Reproducibility"]


def test_extractor_initialization():
    """Test extractor initialization with dictionary protocol."""
    protocol = {
        "fields": [
            {
                "name": "architecture_type",
                "description": "DL architecture",
                "type": "string",
                "required": True
            }
        ]
    }
    
    extractor = SystematicReviewExtractor(protocol=protocol)
    assert len(extractor.fields) == 1
    assert extractor.fields[0].name == "architecture_type"


def test_extractor_yaml_schema():
    """Test extractor with YAML schema."""
    schema_path = Path(__file__).parent.parent / "schemas" / "dl_eeg_review_2019_schema.yaml"
    
    if not schema_path.exists():
        pytest.skip("Schema file not found")
    
    extractor = SystematicReviewExtractor(protocol=str(schema_path))
    assert len(extractor.fields) > 0
    assert any(f.name == "architecture_type" for f in extractor.fields)
    assert any(f.name == "task_type" for f in extractor.fields)


def test_rule_based_extraction():
    """Test rule-based extraction."""
    protocol = {
        "fields": [
            {
                "name": "architecture_type",
                "description": "DL architecture",
                "type": "enum",
                "enum_values": ["CNN", "RNN", "Transformer"],
                "required": True
            },
            {
                "name": "dataset_name",
                "description": "Dataset used",
                "type": "string",
                "required": False
            }
        ]
    }
    
    extractor = SystematicReviewExtractor(protocol=protocol)
    
    papers = [
        {
            "paper_id": "test_1",
            "title": "CNN-Based EEG Analysis",
            "authors": "Test et al.",
            "year": 2023,
            "abstract": "We use a convolutional neural network on the CHB-MIT dataset."
        }
    ]
    
    df = extractor.run(papers)
    
    assert len(df) == 1
    assert df.iloc[0]["architecture_type"] == "CNN"
    assert "CHB-MIT" in df.iloc[0]["dataset_name"]
    assert df.iloc[0]["architecture_type_confidence"] > 0.5


def test_export_formats():
    """Test exporting to different formats."""
    import tempfile
    import json
    
    protocol = {
        "fields": [
            {
                "name": "test_field",
                "description": "Test",
                "type": "string",
                "required": False
            }
        ]
    }
    
    extractor = SystematicReviewExtractor(protocol=protocol)
    papers = [
        {
            "paper_id": "test_1",
            "title": "Test Paper",
            "authors": "Author",
            "year": 2023,
            "abstract": "Test abstract"
        }
    ]
    
    df = extractor.run(papers)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CSV export
        csv_path = Path(tmpdir) / "test.csv"
        extractor.export(str(csv_path), format="csv")
        assert csv_path.exists()
        
        # Test JSON export
        json_path = Path(tmpdir) / "test.json"
        extractor.export(str(json_path), format="json")
        assert json_path.exists()
        
        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["title"] == "Test Paper"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

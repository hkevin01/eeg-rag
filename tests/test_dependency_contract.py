"""Installation contracts for the supported Python and retrieval stack."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def test_modern_stack_is_declared_and_python_floor_is_explicit():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    python_versions = SpecifierSet(project["requires-python"])
    assert "3.11" in python_versions
    assert "3.9" not in python_versions
    dependencies = {
        requirement.name: requirement
        for requirement in map(Requirement, project["dependencies"])
    }
    assert "6.3.0" in dependencies["sentence-transformers"].specifier
    assert "2.2.0" not in dependencies["sentence-transformers"].specifier
    assert "5.23.0" in dependencies["transformers"].specifier
    assert "qdrant-client" in dependencies
    assert "aiohttp" in dependencies
    assert {"retrieval", "ingestion", "onnx"} <= set(project["optional-dependencies"])

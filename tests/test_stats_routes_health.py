"""Tests for stats health route generation readiness payload behavior."""

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


def _import_stats_routes_module():
    """Import stats routes module with lightweight FastAPI stubs."""
    if "fastapi" not in sys.modules:
        class _DummyRouter:
            def __init__(self, *args, **kwargs):
                _ = (args, kwargs)

            def get(self, *args, **kwargs):
                _ = (args, kwargs)

                def _decorator(func):
                    return func

                return _decorator

            def post(self, *args, **kwargs):
                _ = (args, kwargs)

                def _decorator(func):
                    return func

                return _decorator

        fastapi_stub = SimpleNamespace(APIRouter=_DummyRouter, HTTPException=Exception)
        sys.modules["fastapi"] = fastapi_stub

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "eeg_rag"
        / "api"
        / "stats_routes.py"
    )
    module_name = "testable_stats_routes"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load stats_routes module for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_health_payload_includes_unavailable_generation_readiness(monkeypatch) -> None:
    """Health route exposes readiness payload shape for unavailable providers."""
    mod = _import_stats_routes_module()

    fake_stats = SimpleNamespace(
        total_papers=123,
        index_health={
            "status": "healthy",
            "issues": [],
        },
    )
    fake_service = SimpleNamespace(get_full_stats=lambda: fake_stats)

    monkeypatch.setattr(mod, "get_stats_service", lambda: fake_service)
    monkeypatch.setattr(
        mod,
        "_get_generation_readiness_report",
        lambda: {
            "status": "unavailable",
            "providers": [],
            "best_provider": None,
            "error": "no provider configured",
        },
    )

    payload = await mod.get_health()
    readiness = payload.generation_readiness

    assert readiness is not None
    assert readiness["status"] == "unavailable"
    assert readiness["providers"] == []
    assert readiness["best_provider"] is None
    assert "error" in readiness

"""Tests for the production LLM provider readiness ranking."""

import importlib


class _FakeProviderBase:
    def __init__(self, config):
        self.config = config
        self.provider_type = None

    async def generate(self, query, context, streaming=True):
        _ = (query, context, streaming)
        if False:
            yield ""


class _FakeOpenAIProvider(_FakeProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.provider_type = self._provider_type


class _FakeAnthropicProvider(_FakeProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.provider_type = self._provider_type


class _FakeOllamaProvider(_FakeProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.provider_type = self._provider_type


def _import_response_generator_module():
    return importlib.import_module("src.eeg_rag.generation.response_generator")


def test_provider_readiness_scores_rank_by_production_fit(monkeypatch) -> None:
    mod = _import_response_generator_module()

    monkeypatch.setattr(mod, "OpenAIProvider", _FakeOpenAIProvider)
    monkeypatch.setattr(mod, "AnthropicProvider", _FakeAnthropicProvider)
    monkeypatch.setattr(mod, "OllamaProvider", _FakeOllamaProvider)

    _FakeOpenAIProvider._provider_type = mod.ProviderType.OPENAI
    _FakeAnthropicProvider._provider_type = mod.ProviderType.ANTHROPIC
    _FakeOllamaProvider._provider_type = mod.ProviderType.OLLAMA

    config = mod.GenerationConfig(
        providers=[
            mod.ProviderType.OLLAMA,
            mod.ProviderType.ANTHROPIC,
            mod.ProviderType.OPENAI,
        ],
        stream=True,
    )
    generator = mod.ResponseGenerator(config)

    readiness = generator.get_provider_readiness()
    scores = [item.readiness_score for item in readiness]

    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert [item.provider for item in readiness] == [
        provider.provider_type for provider in generator.fallback_chain
    ]


def test_provider_readiness_formula_prefers_quality_over_ordering(monkeypatch) -> None:
    mod = _import_response_generator_module()

    monkeypatch.setattr(mod, "OpenAIProvider", _FakeOpenAIProvider)
    monkeypatch.setattr(mod, "AnthropicProvider", _FakeAnthropicProvider)
    monkeypatch.setattr(mod, "OllamaProvider", _FakeOllamaProvider)

    _FakeOpenAIProvider._provider_type = mod.ProviderType.OPENAI
    _FakeAnthropicProvider._provider_type = mod.ProviderType.ANTHROPIC
    _FakeOllamaProvider._provider_type = mod.ProviderType.OLLAMA

    config = mod.GenerationConfig(
        providers=[
            mod.ProviderType.OPENAI,
            mod.ProviderType.ANTHROPIC,
            mod.ProviderType.OLLAMA,
        ],
    )
    generator = mod.ResponseGenerator(config)

    openai_score = generator._score_provider_readiness(
        mod.ProviderType.OPENAI,
        config_rank=0,
        total_providers=3,
    ).readiness_score
    ollama_score = generator._score_provider_readiness(
        mod.ProviderType.OLLAMA,
        config_rank=2,
        total_providers=3,
    ).readiness_score

    assert openai_score > ollama_score
    assert openai_score == max(
        item.readiness_score for item in generator.get_provider_readiness()
    )

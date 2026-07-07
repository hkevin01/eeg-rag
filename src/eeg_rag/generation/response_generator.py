"""
Production-grade LLM Response Generator with multi-provider fallback.

Supports OpenAI, Anthropic, and Ollama with automatic failover, streaming,
and citation integration.
"""

from typing import AsyncGenerator, Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import os
import time
from datetime import datetime
import json
from collections import deque

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import httpx
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.ProviderType
# Requirement  : `ProviderType` class shall be instantiable and expose the documented interface
# Purpose      : Supported LLM providers
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ProviderType with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.ProviderError
# Requirement  : `ProviderError` class shall be instantiable and expose the documented interface
# Purpose      : Base exception for provider errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ProviderError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.AllProvidersFailedError
# Requirement  : `AllProvidersFailedError` class shall be instantiable and expose the documented interface
# Purpose      : All configured providers failed
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate AllProvidersFailedError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class AllProvidersFailedError(Exception):
    """All configured providers failed."""
    pass


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.Document
# Requirement  : `Document` class shall be instantiable and expose the documented interface
# Purpose      : Document with content and metadata
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate Document with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """Document with content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pmid: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.Document.format_citation
    # Requirement  : `format_citation` shall format document as citation
    # Purpose      : Format document as citation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def format_citation(self) -> str:
        """Format document as citation."""
        if self.pmid:
            return f"[PMID:{self.pmid}]"
        return f"[{self.title[:50] if self.title else 'Unknown'}...]"


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.GenerationConfig
# Requirement  : `GenerationConfig` class shall be instantiable and expose the documented interface
# Purpose      : Configuration for response generation
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate GenerationConfig with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    providers: List[ProviderType] = field(default_factory=lambda: [
        ProviderType.OPENAI,
        ProviderType.ANTHROPIC,
        ProviderType.OLLAMA
    ])
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = True
    include_citations: bool = True
    system_prompt: Optional[str] = None
    openai_model: str = "gpt-4"
    anthropic_model: str = "claude-3-sonnet-20240229"
    ollama_model: str = "llama2"
    retry_attempts: int = 3
    timeout_seconds: int = 30


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.ProviderReadiness
# Requirement  : `ProviderReadiness` class shall be instantiable and expose the documented interface
# Purpose      : Ranked provider readiness snapshot
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ProviderReadiness with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderReadiness:
    """Ranked readiness snapshot for a provider."""

    provider: ProviderType
    readiness_score: float
    quality_score: float
    latency_score: float
    privacy_score: float
    ordering_score: float


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.ProviderTelemetry
# Requirement  : `ProviderTelemetry` class shall be instantiable and expose the documented interface
# Purpose      : Telemetry for provider health and tuning
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ProviderTelemetry with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ProviderTelemetry:
    """Observed provider telemetry used for readiness and tuning."""

    provider: ProviderType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_failures: int = 0
    total_latency_ms: float = 0.0
    recent_latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    recent_failures: Deque[str] = field(default_factory=lambda: deque(maxlen=16))
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Return observed success rate."""
        if self.total_requests <= 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Return observed failure rate."""
        if self.total_requests <= 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Return average latency across observed requests."""
        if self.total_requests <= 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def p95_latency_ms(self) -> float:
        """Return approximate p95 latency from recent samples."""
        if not self.recent_latencies_ms:
            return 0.0
        samples = sorted(self.recent_latencies_ms)
        index = max(0, min(len(samples) - 1, int(round((len(samples) - 1) * 0.95))))
        return float(samples[index])

    def record(self, latency_ms: float, success: bool, error: Optional[BaseException] = None) -> None:
        """Record an observed generation outcome."""
        self.total_requests += 1
        self.total_latency_ms += max(0.0, float(latency_ms))
        self.recent_latencies_ms.append(max(0.0, float(latency_ms)))

        if success:
            self.successful_requests += 1
            self.recent_failures.clear()
            return

        self.failed_requests += 1
        error_name = error.__class__.__name__ if error else "GenerationError"
        self.recent_failures.append(error_name)
        self.last_error = f"{error_name}: {error}" if error else error_name
        self.last_error_at = datetime.utcnow().isoformat()
        if "timeout" in error_name.lower():
            self.timeout_failures += 1


_PROVIDER_QUALITY_SCORES: Dict[ProviderType, float] = {
    ProviderType.OPENAI: 1.00,
    ProviderType.ANTHROPIC: 0.93,
    ProviderType.OLLAMA: 0.85,
}

_PROVIDER_LATENCY_SCORES: Dict[ProviderType, float] = {
    ProviderType.OPENAI: 0.95,
    ProviderType.ANTHROPIC: 0.91,
    ProviderType.OLLAMA: 0.83,
}

_PROVIDER_PRIVACY_SCORES: Dict[ProviderType, float] = {
    ProviderType.OPENAI: 0.80,
    ProviderType.ANTHROPIC: 0.90,
    ProviderType.OLLAMA: 1.00,
}


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.BaseProvider
# Requirement  : `BaseProvider` class shall be instantiable and expose the documented interface
# Purpose      : Base class for LLM providers
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate BaseProvider with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class BaseProvider:
    """Base class for LLM providers."""

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.BaseProvider.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: GenerationConfig
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.provider_type: ProviderType = None

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.BaseProvider.generate
    # Requirement  : `generate` shall generate response with optional streaming
    # Purpose      : Generate response with optional streaming
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: List[Document]; streaming: bool (default=True)
    # Outputs      : AsyncGenerator[str, None]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        context: List[Document],
        streaming: bool = True,
        timeout_seconds: Optional[int] = None,
        retry_attempts: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response with optional streaming."""
        raise NotImplementedError

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.BaseProvider._format_context
    # Requirement  : `_format_context` shall format documents into context string
    # Purpose      : Format documents into context string
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : documents: List[Document]
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            citation = doc.format_citation()
            context_parts.append(
                f"\n[Document {i}] {citation}\n{doc.content}\n"
            )
        return "\n".join(context_parts)

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.BaseProvider._build_prompt
    # Requirement  : `_build_prompt` shall build prompt with query and context
    # Purpose      : Build prompt with query and context
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt with query and context."""
        system = self.config.system_prompt or self._default_system_prompt()
        return f"""{system}

# Research Query
{query}

# Retrieved Literature
{context}

# Instructions
Generate a comprehensive, evidence-based response that:
1. Directly answers the research question
2. Synthesizes findings from multiple papers
3. Includes inline citations in [PMID:XXXXXXXX] format
4. Highlights consensus and contradictions
5. Identifies research gaps if applicable

Your response:"""

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.BaseProvider._default_system_prompt
    # Requirement  : `_default_system_prompt` shall default system prompt for EEG research
    # Purpose      : Default system prompt for EEG research
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _default_system_prompt(self) -> str:
        """Default system prompt for EEG research."""
        return """You are an expert neuroscientist specializing in electroencephalography (EEG) research.
Your role is to synthesize literature findings into clear, accurate, and well-cited responses.

Guidelines:
- Prioritize recent, high-quality studies
- Always include specific citations [PMID:XXXXXXXX]
- Highlight areas of scientific consensus
- Note conflicting evidence when present
- Use precise EEG terminology
- Focus on evidence-based conclusions"""


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.OpenAIProvider
# Requirement  : `OpenAIProvider` class shall be instantiable and expose the documented interface
# Purpose      : OpenAI GPT provider
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate OpenAIProvider with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class OpenAIProvider(BaseProvider):
    """OpenAI GPT provider."""

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.OpenAIProvider.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: GenerationConfig
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.provider_type = ProviderType.OPENAI
        if not OPENAI_AVAILABLE:
            raise ProviderError("OpenAI not available - install: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not set")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        logger.info(f"OpenAI provider initialized (model={config.openai_model})")

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.OpenAIProvider.generate
    # Requirement  : `generate` shall generate response using OpenAI
    # Purpose      : Generate response using OpenAI
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: List[Document]; streaming: bool (default=True)
    # Outputs      : AsyncGenerator[str, None]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        context: List[Document],
        streaming: bool = True,
        timeout_seconds: Optional[int] = None,
        retry_attempts: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response using OpenAI."""
        try:
            context_str = self._format_context(context)
            prompt = self._build_prompt(query, context_str)
            effective_timeout = timeout_seconds or self.config.timeout_seconds
            effective_retries = max(1, retry_attempts or self.config.retry_attempts)

            messages = [
                {"role": "system", "content": self._default_system_prompt()},
                {"role": "user", "content": prompt}
            ]

            for attempt in range(effective_retries):
                try:
                    if streaming:
                        stream = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=self.config.openai_model,
                                messages=messages,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                stream=True
                            ),
                            timeout=effective_timeout,
                        )

                        async for chunk in stream:
                            if chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    else:
                        response = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=self.config.openai_model,
                                messages=messages,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                stream=False
                            ),
                            timeout=effective_timeout,
                        )
                        yield response.choices[0].message.content
                    return
                except Exception:
                    if attempt >= effective_retries - 1:
                        raise
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise ProviderError(f"OpenAI error: {e}")


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.AnthropicProvider
# Requirement  : `AnthropicProvider` class shall be instantiable and expose the documented interface
# Purpose      : Anthropic Claude provider
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate AnthropicProvider with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.AnthropicProvider.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: GenerationConfig
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.provider_type = ProviderType.ANTHROPIC
        if not ANTHROPIC_AVAILABLE:
            raise ProviderError("Anthropic not available - install: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info(f"Anthropic provider initialized (model={config.anthropic_model})")

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.AnthropicProvider.generate
    # Requirement  : `generate` shall generate response using Anthropic Claude
    # Purpose      : Generate response using Anthropic Claude
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: List[Document]; streaming: bool (default=True)
    # Outputs      : AsyncGenerator[str, None]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        context: List[Document],
        streaming: bool = True,
        timeout_seconds: Optional[int] = None,
        retry_attempts: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response using Anthropic Claude."""
        try:
            context_str = self._format_context(context)
            prompt = self._build_prompt(query, context_str)
            effective_timeout = timeout_seconds or self.config.timeout_seconds
            effective_retries = max(1, retry_attempts or self.config.retry_attempts)

            for attempt in range(effective_retries):
                try:
                    if streaming:
                        async with asyncio.timeout(effective_timeout):
                            async with self.client.messages.stream(
                                model=self.config.anthropic_model,
                                max_tokens=self.config.max_tokens,
                                temperature=self.config.temperature,
                                system=self._default_system_prompt(),
                                messages=[{"role": "user", "content": prompt}]
                            ) as stream:
                                async for text in stream.text_stream:
                                    yield text
                    else:
                        async with asyncio.timeout(effective_timeout):
                            message = await self.client.messages.create(
                                model=self.config.anthropic_model,
                                max_tokens=self.config.max_tokens,
                                temperature=self.config.temperature,
                                system=self._default_system_prompt(),
                                messages=[{"role": "user", "content": prompt}]
                            )
                        yield message.content[0].text
                    return
                except Exception:
                    if attempt >= effective_retries - 1:
                        raise
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise ProviderError(f"Anthropic error: {e}")


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.OllamaProvider
# Requirement  : `OllamaProvider` class shall be instantiable and expose the documented interface
# Purpose      : Local Ollama provider
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate OllamaProvider with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class OllamaProvider(BaseProvider):
    """Local Ollama provider."""

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.OllamaProvider.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: GenerationConfig
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.provider_type = ProviderType.OLLAMA
        if not OLLAMA_AVAILABLE:
            raise ProviderError("Ollama requires httpx - install: pip install httpx")

        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        logger.info(f"Ollama provider initialized (model={config.ollama_model}, url={self.base_url})")

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.OllamaProvider.generate
    # Requirement  : `generate` shall generate response using Ollama
    # Purpose      : Generate response using Ollama
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: List[Document]; streaming: bool (default=True)
    # Outputs      : AsyncGenerator[str, None]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        context: List[Document],
        streaming: bool = True,
        timeout_seconds: Optional[int] = None,
        retry_attempts: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response using Ollama."""
        try:
            context_str = self._format_context(context)
            prompt = self._build_prompt(query, context_str)
            effective_timeout = timeout_seconds or self.config.timeout_seconds
            effective_retries = max(1, retry_attempts or self.config.retry_attempts)

            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                data = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": streaming,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                }

                for attempt in range(effective_retries):
                    try:
                        if streaming:
                            async with client.stream(
                                "POST",
                                f"{self.base_url}/api/generate",
                                json=data
                            ) as response:
                                response.raise_for_status()
                                async for line in response.aiter_lines():
                                    if line:
                                        chunk = json.loads(line)
                                        if "response" in chunk:
                                            yield chunk["response"]
                        else:
                            response = await client.post(
                                f"{self.base_url}/api/generate",
                                json=data
                            )
                            response.raise_for_status()
                            yield response.json()["response"]
                        return
                    except Exception:
                        if attempt >= effective_retries - 1:
                            raise
                        await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise ProviderError(f"Ollama error: {e}")


# ---------------------------------------------------------------------------
# ID           : generation.response_generator.ResponseGenerator
# Requirement  : `ResponseGenerator` class shall be instantiable and expose the documented interface
# Purpose      : Production-grade response generator with multi-provider fallback
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ResponseGenerator with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ResponseGenerator:
    """
    Production-grade response generator with multi-provider fallback.

    Features:
    - Multiple LLM providers (OpenAI, Anthropic, Ollama)
    - Automatic failover chain
    - Streaming and non-streaming modes
    - Citation integration
    - Configurable retry logic

    Example:
        config = GenerationConfig(
            providers=[ProviderType.OPENAI, ProviderType.OLLAMA],
            temperature=0.7,
            stream=True
        )
        generator = ResponseGenerator(config)

        async for chunk in generator.generate(query, documents):
            print(chunk, end="", flush=True)
    """

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: Optional[GenerationConfig] (default=None)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.provider_telemetry: Dict[ProviderType, ProviderTelemetry] = {
            provider_type: ProviderTelemetry(provider=provider_type)
            for provider_type in self.config.providers
        }
        self.providers = self._init_providers()
        self.fallback_chain = self._build_fallback_chain()
        logger.info(f"ResponseGenerator initialized with {len(self.providers)} providers")

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator._score_provider_readiness
    # Requirement  : `_score_provider_readiness` shall rank providers using a deterministic readiness formula
    # Purpose      : Rank provider readiness for fallback selection
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : provider_type: ProviderType; config_rank: int; total_providers: int
    # Outputs      : ProviderReadiness
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _score_provider_readiness(
        self,
        provider_type: ProviderType,
        config_rank: int,
        total_providers: int,
    ) -> ProviderReadiness:
        """Rank provider readiness using observed telemetry and fallback priors."""
        telemetry = self.provider_telemetry.get(provider_type)
        average_latency_ms = telemetry.average_latency_ms if telemetry else 0.0
        success_rate = telemetry.success_rate if telemetry else 0.0
        failure_rate = telemetry.failure_rate if telemetry else 0.0
        p95_latency_ms = telemetry.p95_latency_ms if telemetry else 0.0

        if telemetry and telemetry.total_requests > 0:
            quality_score = max(0.0, min(1.0, success_rate))
            latency_basis = p95_latency_ms if p95_latency_ms > 0.0 else average_latency_ms
            latency_score = max(0.0, 1.0 - min(1.0, latency_basis / 4000.0))
            privacy_score = 1.0 if provider_type == ProviderType.OLLAMA else 0.75
        else:
            quality_score = 0.5
            latency_score = 0.5
            privacy_score = 1.0 if provider_type == ProviderType.OLLAMA else 0.75

        ordering_score = 1.0 if total_providers <= 1 else 1.0 - (
            config_rank / max(1, total_providers - 1)
        )
        readiness_score = (
            0.58 * quality_score
            + 0.26 * latency_score
            + 0.10 * privacy_score
            + 0.06 * ordering_score
        )
        readiness_score -= min(0.15, failure_rate * 0.15)
        readiness_score = max(0.0, min(1.0, readiness_score))
        return ProviderReadiness(
            provider=provider_type,
            readiness_score=readiness_score,
            quality_score=quality_score,
            latency_score=latency_score,
            privacy_score=privacy_score,
            ordering_score=ordering_score,
        )

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator.get_provider_readiness
    # Requirement  : `get_provider_readiness` shall expose the ranked readiness report
    # Purpose      : Provide a ranked readiness report for available providers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[ProviderReadiness]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_provider_readiness(self) -> List[ProviderReadiness]:
        """Return readiness metrics for each initialized provider, ranked high to low."""
        total_providers = max(1, len(self.config.providers))
        readiness = []

        for config_rank, provider_type in enumerate(self.config.providers):
            if provider_type not in self.providers:
                continue
            readiness.append(
                self._score_provider_readiness(
                    provider_type=provider_type,
                    config_rank=config_rank,
                    total_providers=total_providers,
                )
            )

        readiness.sort(
            key=lambda item: (
                -item.readiness_score,
                item.provider.value,
            )
        )
        return readiness

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator._derive_runtime_settings
    # Requirement  : `_derive_runtime_settings` shall compute timeout and retry settings from telemetry
    # Purpose      : Derive provider-specific runtime settings
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : provider_type: ProviderType
    # Outputs      : Dict[str, int]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _derive_runtime_settings(self, provider_type: ProviderType) -> Dict[str, int]:
        """Derive per-provider timeout and retry settings from observed failures."""
        telemetry = self.provider_telemetry.get(provider_type)
        base_timeout = max(5, int(getattr(self.config, "timeout_seconds", 30)))
        base_retries = max(1, int(getattr(self.config, "retry_attempts", 3)))

        if not telemetry or telemetry.total_requests <= 0:
            return {
                "timeout_seconds": base_timeout,
                "retry_attempts": base_retries,
            }

        failure_rate = telemetry.failure_rate
        timeout_pressure = min(1.5, (telemetry.p95_latency_ms / max(1.0, base_timeout * 1000.0)))
        timeout_seconds = int(
            round(base_timeout * (1.0 + (0.35 * failure_rate) + (0.25 * timeout_pressure)))
        )
        timeout_seconds = max(5, min(120, timeout_seconds))

        retry_attempts = int(
            round(base_retries + (failure_rate * 2.0) + (1.0 if telemetry.timeout_failures > 0 else 0.0))
        )
        retry_attempts = max(1, min(5, retry_attempts))

        return {
            "timeout_seconds": timeout_seconds,
            "retry_attempts": retry_attempts,
        }

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator._record_provider_outcome
    # Requirement  : `_record_provider_outcome` shall persist provider telemetry after generation
    # Purpose      : Persist provider telemetry after generation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : provider_type: ProviderType; latency_ms: float; success: bool; error: Optional[BaseException]
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _record_provider_outcome(
        self,
        provider_type: ProviderType,
        latency_ms: float,
        success: bool,
        error: Optional[BaseException] = None,
    ) -> None:
        """Record telemetry for a provider generation attempt."""
        telemetry = self.provider_telemetry.setdefault(
            provider_type,
            ProviderTelemetry(provider=provider_type),
        )
        telemetry.record(latency_ms=latency_ms, success=success, error=error)

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator.get_generation_readiness_report
    # Requirement  : `get_generation_readiness_report` shall expose provider telemetry and tuning
    # Purpose      : Expose provider telemetry and tuning
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_generation_readiness_report(self) -> Dict[str, Any]:
        """Return provider readiness, telemetry, and runtime tuning settings."""
        report = []
        for item in self.get_provider_readiness():
            runtime = self._derive_runtime_settings(item.provider)
            telemetry = self.provider_telemetry.get(item.provider)
            report.append(
                {
                    "provider": item.provider.value,
                    "readiness_score": item.readiness_score,
                    "success_rate": telemetry.success_rate if telemetry else 0.0,
                    "average_latency_ms": telemetry.average_latency_ms if telemetry else 0.0,
                    "p95_latency_ms": telemetry.p95_latency_ms if telemetry else 0.0,
                    "failure_rate": telemetry.failure_rate if telemetry else 0.0,
                    "timeout_seconds": runtime["timeout_seconds"],
                    "retry_attempts": runtime["retry_attempts"],
                    "total_requests": telemetry.total_requests if telemetry else 0,
                    "successful_requests": telemetry.successful_requests if telemetry else 0,
                    "failed_requests": telemetry.failed_requests if telemetry else 0,
                    "timeout_failures": telemetry.timeout_failures if telemetry else 0,
                    "last_error": telemetry.last_error if telemetry else None,
                    "last_error_at": telemetry.last_error_at if telemetry else None,
                }
            )

        return {
            "status": "available" if report else "unavailable",
            "providers": report,
            "best_provider": report[0]["provider"] if report else None,
        }

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator._init_providers
    # Requirement  : `_init_providers` shall initialize available providers
    # Purpose      : Initialize available providers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[ProviderType, BaseProvider]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _init_providers(self) -> Dict[ProviderType, BaseProvider]:
        """Initialize available providers."""
        providers = {}

        for provider_type in self.config.providers:
            try:
                if provider_type == ProviderType.OPENAI:
                    providers[provider_type] = OpenAIProvider(self.config)
                elif provider_type == ProviderType.ANTHROPIC:
                    providers[provider_type] = AnthropicProvider(self.config)
                elif provider_type == ProviderType.OLLAMA:
                    providers[provider_type] = OllamaProvider(self.config)
            except ProviderError as e:
                logger.warning(f"Failed to initialize {provider_type.value}: {e}")

        if not providers:
            raise AllProvidersFailedError("No LLM providers available")

        return providers

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator._build_fallback_chain
    # Requirement  : `_build_fallback_chain` shall build ordered fallback chain
    # Purpose      : Build ordered fallback chain
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[BaseProvider]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _build_fallback_chain(self) -> List[BaseProvider]:
        """Build ordered fallback chain."""
        return [
            self.providers[item.provider]
            for item in self.get_provider_readiness()
        ]

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator.generate
    # Requirement  : `generate` shall generate response with automatic fallback
    # Purpose      : Generate response with automatic fallback
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: List[Document]; streaming: bool (default=None)
    # Outputs      : AsyncGenerator[str, None]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        context: List[Document],
        streaming: bool = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate response with automatic fallback.

        Args:
            query: User query
            context: Retrieved documents
            streaming: Enable streaming (uses config default if None)

        Yields:
            Response chunks if streaming, full response otherwise

        Raises:
            AllProvidersFailedError: If all providers fail
        """
        streaming = streaming if streaming is not None else self.config.stream

        for i, provider in enumerate(self.fallback_chain):
            settings = self._derive_runtime_settings(provider.provider_type)
            start_time = time.time()
            try:
                logger.info(f"Attempting generation with {provider.provider_type.value}")

                async for chunk in provider.generate(
                    query,
                    context,
                    streaming,
                    timeout_seconds=settings["timeout_seconds"],
                    retry_attempts=settings["retry_attempts"],
                ):
                    yield chunk

                elapsed_ms = (time.time() - start_time) * 1000.0
                self._record_provider_outcome(
                    provider.provider_type,
                    latency_ms=elapsed_ms,
                    success=True,
                )
                logger.info(f"Generation successful with {provider.provider_type.value}")
                return

            except ProviderError as e:
                elapsed_ms = (time.time() - start_time) * 1000.0
                self._record_provider_outcome(
                    provider.provider_type,
                    latency_ms=elapsed_ms,
                    success=False,
                    error=e,
                )
                logger.warning(
                    f"Provider {provider.provider_type.value} failed: {e}. "
                    f"Trying next provider ({i+1}/{len(self.fallback_chain)})"
                )
                continue

        raise AllProvidersFailedError("All configured providers failed")

    # ---------------------------------------------------------------------------
    # ID           : generation.response_generator.ResponseGenerator.generate_with_citations
    # Requirement  : `generate_with_citations` shall generate response and append citation list
    # Purpose      : Generate response and append citation list
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; context: List[Document]; streaming: bool (default=None)
    # Outputs      : AsyncGenerator[str, None]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def generate_with_citations(
        self,
        query: str,
        context: List[Document],
        streaming: bool = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate response and append citation list.

        Args:
            query: User query
            context: Retrieved documents
            streaming: Enable streaming

        Yields:
            Response chunks with appended citations
        """
        # Generate main response
        async for chunk in self.generate(query, context, streaming):
            yield chunk

        # Append citations
        if context:
            yield "\n\n## References\n"
            for i, doc in enumerate(context, 1):
                citation_info = []
                if doc.pmid:
                    citation_info.append(f"PMID:{doc.pmid}")
                if doc.title:
                    citation_info.append(doc.title)
                if doc.authors:
                    citation_info.append(", ".join(doc.authors[:3]))
                if doc.year:
                    citation_info.append(f"({doc.year})")

                yield f"\n{i}. {' - '.join(citation_info)}"

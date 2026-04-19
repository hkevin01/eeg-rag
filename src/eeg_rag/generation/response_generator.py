"""
Production-grade LLM Response Generator with multi-provider fallback.

Supports OpenAI, Anthropic, and Ollama with automatic failover, streaming,
and citation integration.
"""

from typing import AsyncGenerator, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import os
from datetime import datetime
import json

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
        streaming: bool = True
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
        streaming: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response using OpenAI."""
        try:
            context_str = self._format_context(context)
            prompt = self._build_prompt(query, context_str)
            
            messages = [
                {"role": "system", "content": self._default_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            if streaming:
                stream = await self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                response = await self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False
                )
                yield response.choices[0].message.content
                
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
        streaming: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response using Anthropic Claude."""
        try:
            context_str = self._format_context(context)
            prompt = self._build_prompt(query, context_str)
            
            if streaming:
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
                message = await self.client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self._default_system_prompt(),
                    messages=[{"role": "user", "content": prompt}]
                )
                yield message.content[0].text
                
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
        streaming: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response using Ollama."""
        try:
            context_str = self._format_context(context)
            prompt = self._build_prompt(query, context_str)
            
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                data = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": streaming,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                }
                
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
        self.providers = self._init_providers()
        self.fallback_chain = self._build_fallback_chain()
        logger.info(f"ResponseGenerator initialized with {len(self.providers)} providers")
    
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
        return [self.providers[pt] for pt in self.config.providers if pt in self.providers]
    
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
            try:
                logger.info(f"Attempting generation with {provider.provider_type.value}")
                
                async for chunk in provider.generate(query, context, streaming):
                    yield chunk
                
                logger.info(f"Generation successful with {provider.provider_type.value}")
                return
                
            except ProviderError as e:
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

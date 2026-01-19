# GitHub Copilot Prompt Reference for EEG-RAG

This document provides prompt templates and examples for effectively using GitHub Copilot with the EEG-RAG project.

## Quick Start Prompts

### New Component Development
```
Create a new EEG-RAG [component type] that:
- Follows medical-grade quality standards
- Includes comprehensive error handling
- Uses async/await patterns
- Has full type hints and Google-style docstrings
- Includes EEG domain validation
- Has 85%+ test coverage

Component requirements:
[specific requirements]

EEG domain considerations:
[domain-specific needs]
```

### Testing Prompts
```
Generate comprehensive tests for [component] that:
- Test EEG terminology handling (electrodes, frequency bands, ERPs)
- Include edge cases for medical data
- Mock external APIs (PubMed, OpenAI)
- Use pytest-asyncio for async functions
- Follow the existing test structure in tests/
- Include parametrized tests for frequency bands and electrode names
```

### Refactoring Prompts
```
Refactor this [component] to:
- Meet EEG-RAG production standards
- Add comprehensive error handling
- Improve performance for [specific requirement]
- Maintain backward compatibility
- Add proper logging and monitoring
- Ensure medical domain validation
```

## Domain-Specific Prompts

### EEG Data Processing
```
Create an EEG data processor that:
- Handles 10-20 electrode system data
- Processes frequency bands: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-100Hz)
- Validates electrode names and montages
- Preserves clinical metadata
- Uses numpy arrays for efficient computation
- Includes comprehensive input validation
```

### Citation and PMID Handling
```
Implement PMID validation that:
- Extracts PMIDs using regex pattern PMID[:\s]*(\d{7,8})
- Validates against PubMed API
- Handles rate limiting and retries
- Caches validated PMIDs
- Returns structured citation data
- Includes comprehensive error handling for invalid PMIDs
```

### RAG Pipeline Components
```
Develop a RAG [retrieval/generation/evaluation] component that:
- Follows the multi-agent architecture pattern
- Implements hybrid search (BM25 + dense vectors)
- Preserves medical terminology and citations
- Includes performance monitoring
- Has configurable parameters
- Integrates with the existing agent orchestrator
```

## Architecture-Specific Prompts

### Agent Development
```
Create a new RAG agent that inherits from BaseAgent and:
- Implements the required async execute() method
- Handles [specific query type] efficiently
- Includes proper error handling and retries
- Logs performance metrics
- Validates EEG domain inputs
- Returns structured results with confidence scores
```

### Database Integration
```
Implement [database operation] that:
- Uses async database connections
- Handles connection pooling
- Includes comprehensive error handling
- Validates input parameters
- Uses parameterized queries to prevent injection
- Includes proper transaction management
- Logs operations for debugging
```

### API Integration
```
Create an async API client for [service] that:
- Uses aiohttp with proper session management
- Implements exponential backoff retry logic
- Handles rate limiting appropriately
- Validates API responses
- Includes comprehensive error handling
- Logs API calls for monitoring
- Uses environment variables for configuration
```

## Performance-Specific Prompts

### Optimization
```
Optimize this [component] for:
- Processing 10K+ EEG documents efficiently
- Sub-100ms retrieval latency
- Minimal memory footprint
- Concurrent request handling
- Efficient caching strategy
- Batch processing capabilities
```

### Caching
```
Implement caching for [component] that:
- Uses appropriate cache expiry (embeddings: indefinite, responses: 1 hour)
- Implements LRU eviction with configurable size limits
- Handles cache misses gracefully
- Includes cache hit rate monitoring
- Uses async-compatible caching library
- Supports cache invalidation patterns
```

## Security-Specific Prompts

### Input Validation
```
Add input validation that:
- Sanitizes all user inputs
- Validates PMID format (7-8 digits)
- Limits query length to 1000 characters
- Removes control characters
- Prevents injection attacks
- Includes comprehensive logging for security events
```

### API Security
```
Secure this API endpoint with:
- Input validation using Pydantic models
- Rate limiting per user/IP
- Request size limits
- Proper error responses (no sensitive data leakage)
- Comprehensive logging
- Authentication/authorization if needed
```

## Example Prompt Templates

### Complete Feature Implementation
```
I need to implement [feature name] for the EEG-RAG system. This feature should:

**Core Requirements:**
- [list specific requirements]

**EEG Domain Requirements:**
- Handle [specific EEG terminology/concepts]
- Validate [domain-specific data]
- Process [clinical scenarios]

**Technical Requirements:**
- Follow EEG-RAG code standards (Python 3.9+, async, type hints)
- Include comprehensive error handling
- Achieve target performance: [specific metrics]
- Integrate with existing [components]

**Testing Requirements:**
- Unit tests with 85%+ coverage
- Integration tests for [scenarios]
- EEG-specific test cases
- Mock [external dependencies]

**Documentation:**
- Google-style docstrings
- Usage examples
- Architecture notes if complex

Please implement this following the patterns established in the codebase.
```

### Bug Fix Prompt
```
I have a bug in [component] where [description of issue]. 

The issue appears to be related to [suspected cause].

Current behavior: [what happens now]
Expected behavior: [what should happen]

Please:
1. Identify the root cause
2. Fix the issue while maintaining EEG-RAG code standards
3. Add tests to prevent regression
4. Ensure the fix doesn't impact EEG domain functionality
5. Include proper error handling

Relevant code: [paste relevant code sections]
```

### Code Review Prompt
```
Please review this [component] for:

**Code Quality:**
- Adherence to EEG-RAG coding standards
- Type hints and documentation completeness
- Error handling patterns
- Performance considerations

**Domain Accuracy:**
- Correct handling of EEG terminology
- Medical data validation
- Citation accuracy

**Architecture:**
- Integration with existing components
- Async patterns usage
- Security considerations

**Testing:**
- Test coverage adequacy
- EEG-specific test scenarios
- Edge case handling

Suggest specific improvements with examples.
```

## Tips for Effective Copilot Usage

1. **Be Specific**: Include exact requirements, constraints, and domain context
2. **Reference Architecture**: Mention integration with existing components
3. **Include Domain Context**: Always specify EEG-related requirements
4. **Specify Quality Standards**: Reference medical-grade quality expectations
5. **Performance Goals**: Include specific latency and throughput targets
6. **Testing Scope**: Specify test coverage and scenario requirements
7. **Error Handling**: Always request comprehensive error handling
8. **Documentation**: Ask for complete docstrings and examples

## Common Anti-Patterns to Avoid

- Don't ask for generic solutions without EEG domain context
- Don't request code without specifying error handling requirements
- Don't skip performance considerations for production components
- Don't forget to specify async/await requirements
- Don't omit testing requirements
- Don't request code without type hints and documentation
# Security and Input Validation

## Input Sanitization
- Validate all PMIDs match pattern ^\d{7,8}$
- Sanitize text inputs before embedding (remove control characters)
- Limit query length to 1000 characters
- Escape special characters in Neo4j Cypher queries

## API Security
- Never log API keys or tokens
- Rotate credentials regularly
- Implement rate limiting for all external APIs
- Validate webhook signatures if implemented

## Data Validation
```python
import re
from typing import Optional
from pydantic import BaseModel, validator, Field

class QueryInput(BaseModel):
    """Validated user query input."""
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    include_sources: bool = True
    
    @validator('text')
    def sanitize_text(cls, v):
        # Remove control characters
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        # Remove excessive whitespace
        v = ' '.join(v.split())
        return v

class PMIDValidator:
    PMID_PATTERN = re.compile(r'^\d{7,8}$')
    
    @classmethod
    def validate(cls, pmid: str) -> bool:
        return bool(cls.PMID_PATTERN.match(pmid))
    
    @classmethod
    def extract_from_text(cls, text: str) -> list[str]:
        """Extract valid PMIDs from text."""
        candidates = re.findall(r'PMID[:\s]*(\d{7,8})', text, re.IGNORECASE)
        return [p for p in candidates if cls.validate(p)]

class CypherSanitizer:
    """Prevent Cypher injection in Neo4j queries."""
    
    @staticmethod
    def escape_string(value: str) -> str:
        """Escape special characters for Cypher strings."""
        return value.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
    
    @staticmethod
    def validate_label(label: str) -> bool:
        """Validate Neo4j label names."""
        return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label))
```

## Prompt Injection Prevention
- Never directly interpolate user input into system prompts
- Use structured input formats
- Implement output validation for generated responses
- Log suspicious patterns for review
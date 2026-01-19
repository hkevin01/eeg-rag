# RAG Implementation Guidelines

## Retrieval Best Practices
- Always return relevance scores with retrieved documents
- Implement maximum marginal relevance (MMR) for diversity
- Cache embeddings for frequently accessed documents
- Use batch embedding for efficiency (batch size 32-64)

## Context Window Management
- Track token counts for all context passed to LLM
- Implement smart truncation that preserves complete sentences
- Prioritize most relevant chunks when context limit approached
- Always reserve tokens for the answer generation

## Citation Handling
- Extract PMIDs using regex pattern: r'PMID[:\s]*(\d{7,8})'
- Validate all PMIDs against PubMed before including in response
- Track which chunks contributed to which parts of the answer
- Implement citation deduplication

## Prompt Engineering
- Use structured prompts with clear sections
- Include few-shot examples for EEG domain
- Specify output format requirements explicitly
- Add instructions for handling uncertainty

## Example Prompt Template
```python
SYSTEM_PROMPT = """You are an EEG research assistant with expertise in:
- Clinical EEG (epilepsy, sleep disorders, encephalopathy)
- Cognitive neuroscience (ERPs, oscillations, connectivity)
- Brain-computer interfaces and machine learning

Guidelines:
1. Base answers ONLY on provided context
2. Cite sources using [PMID:XXXXXXXX] format
3. If information is insufficient, state this clearly
4. Use precise EEG terminology
5. Distinguish between established findings and emerging research"""

USER_PROMPT_TEMPLATE = """Context from EEG literature:
{context}

Question: {question}

Provide a comprehensive answer with citations. If the context doesn't 
contain sufficient information, acknowledge the limitation."""
```

## Chunk Metadata
Always preserve and index:
- Source PMID and DOI
- Publication year
- Section type (abstract, methods, results, discussion)
- MeSH terms if available
- Author affiliations
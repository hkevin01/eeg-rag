# EEG Domain Knowledge for Code Generation

## Terminology Mapping
When generating code that handles EEG terms, understand these relationships:

### Electrode Systems
- 10-20 system: Fp1, Fp2, F3, F4, F7, F8, Fz, C3, C4, Cz, T3, T4, T5, T6, P3, P4, Pz, O1, O2
- 10-10 system: Extended with intermediate positions
- High-density: 64, 128, 256 channel systems

### Frequency Bands (always include Hz ranges)
- Delta: 0.5-4 Hz (deep sleep, pathology)
- Theta: 4-8 Hz (drowsiness, memory)
- Alpha: 8-13 Hz (relaxed wakefulness)
- Beta: 13-30 Hz (active thinking)
- Gamma: 30-100 Hz (cognitive processing)

### ERP Components (include typical latency)
- P1/P100: ~100ms, early visual
- N170: ~170ms, face processing
- P300/P3: ~300ms, attention/memory
- N400: ~400ms, semantic processing
- P600: ~600ms, syntactic processing

## Clinical Relevance Mapping
Link conditions to their EEG signatures:
- Epilepsy: spikes, sharp waves, spike-wave complexes
- Alzheimer's: slowing, reduced alpha, increased theta
- ADHD: elevated theta/beta ratio
- Depression: alpha asymmetry
- Sleep disorders: K-complexes, sleep spindles, slow waves

## Code Generation Examples
```python
# When handling frequency analysis
FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# When validating electrode names
VALID_10_20_ELECTRODES = {
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2", "A1", "A2"
}

# When extracting ERP information
ERP_PATTERNS = {
    "P300": {"latency_ms": (250, 500), "polarity": "positive", "location": "parietal"},
    "N400": {"latency_ms": (300, 500), "polarity": "negative", "location": "centroparietal"},
    "MMN": {"latency_ms": (150, 250), "polarity": "negative", "location": "frontocentral"},
}
```

## Query Understanding
When processing user queries, recognize these patterns:
- "What is X" → definitional, use local knowledge base
- "Latest/recent/2024" → needs PubMed search for new papers
- "X vs Y" or "compare" → comparative analysis, multiple sources
- "How to" → methodological, check for protocols
- Clinical terms (diagnosis, treatment, prognosis) → prioritize clinical sources
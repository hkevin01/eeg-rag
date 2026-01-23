# Quick Start Guide: Top 5 Improvements

## 🚀 Installation

```bash
# Install new dependencies
pip install pyjwt anthropic networkx

# Or install all dependencies
pip install -r requirements.txt
```

## 1️⃣ LLM Response Generation

### Basic Usage

```python
from eeg_rag.generation import ResponseGenerator, GenerationConfig, Document

# Setup
config = GenerationConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=500
)
generator = ResponseGenerator(config=config)

# Generate response
documents = [
    Document(
        content="Alpha oscillations (8-13 Hz) are associated with relaxation.",
        pmid="12345678",
        title="EEG Alpha Rhythms"
    )
]

response = await generator.generate_response(
    query="What are alpha oscillations?",
    documents=documents
)

print(response["answer"])       # Synthesized answer
print(response["citations"])    # [PMID:12345678]
print(response["provider"])     # "openai"
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OLLAMA_HOST="http://localhost:11434"
```

## 2️⃣ EEG Signal Processing

### Basic Usage

```python
from eeg_rag.signals import EEGCaseMatcher, FeatureExtractor
import numpy as np

# Load your EEG data (channels x samples)
eeg_data = np.load("patient_eeg.npy")  # Shape: (19, 30000)
sampling_rate = 250  # Hz

# Extract features
extractor = FeatureExtractor(sampling_rate=sampling_rate)
features = extractor.extract(eeg_data)

print(f"Alpha power: {features.alpha_power:.2f} μV²")
print(f"Dominant freq: {features.dominant_frequency:.1f} Hz")

# Find similar cases
matcher = EEGCaseMatcher(retriever=your_retriever)
similar_cases = matcher.find_similar_cases(eeg_data, sampling_rate, top_k=5)

for case in similar_cases:
    print(f"PMID: {case['pmid']}, Similarity: {case['similarity']:.1%}")
```

### Supported Features

- Delta power (0.5-4 Hz)
- Theta power (4-8 Hz)
- Alpha power (8-13 Hz)
- Beta power (13-30 Hz)
- Gamma power (30-100 Hz)
- Dominant frequency
- Sample entropy
- Hemispheric asymmetry
- Inter-channel coherence

## 3️⃣ Feedback & Learning

### Basic Usage

```python
from eeg_rag.feedback import FeedbackCollector, Feedback

collector = FeedbackCollector()

# Record user feedback
feedback = Feedback(
    query_id="abc123",
    rating=5,  # 1-5 stars
    clicked_pmids=["12345678"],
    ignored_pmids=["87654321"]
)

collector.record_feedback(
    query="What are alpha oscillations?",
    feedback=feedback
)

# Generate training data
dataset = collector.generate_training_data(min_rating=4)
dataset.export_jsonl("training_data.jsonl")

# Get statistics
stats = collector.get_statistics()
print(f"Avg rating: {stats['avg_rating']:.2f}/5")
print(f"CTR: {stats['click_through_rate']:.1%}")
```

### Storage

Feedback is stored in `data/feedback.jsonl` by default.

## 4️⃣ Citation Network Analysis

### Basic Usage

```python
from eeg_rag.analysis import CitationNetworkAnalyzer

analyzer = CitationNetworkAnalyzer(graph_store=neo4j_client)

# Find emerging research areas
fronts = analyzer.find_research_fronts("seizure detection", years=3)

for front in fronts:
    print(f"Topic: {front.topic}")
    print(f"Papers: {len(front.papers)}")

# Identify seminal papers
seminal = analyzer.find_seminal_papers(
    topic="EEG classification",
    min_citations=50,
    top_k=10
)

# Analyze trends
trend = analyzer.analyze_trend("deep learning EEG", 2020, 2024)
print(f"Growth: {trend['growth_rate']:.1%} per year")
```

### Requirements

Requires Neo4j or similar graph database for production use.

## 5️⃣ Production Deployment

### Local Development

```bash
# Start API with new middleware
uvicorn eeg_rag.api.main:app --reload

# Test with JWT token
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/search \
     -d '{"query": "alpha oscillations"}'
```

### Generate JWT Token

```python
from eeg_rag.api.middleware import create_access_token

token = create_access_token(
    user_id="user123",
    username="researcher",
    roles=["premium"],
    expires_in_hours=24
)
```

### Kubernetes Deployment

```bash
# Setup secrets
kubectl create secret generic eeg-rag-secrets \
  --from-literal=jwt-secret="$(openssl rand -hex 32)" \
  --from-literal=ncbi-api-key="YOUR_KEY" \
  --from-literal=s2-api-key="YOUR_KEY"

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify
kubectl get pods -l app=eeg-rag
kubectl get hpa
```

### Rate Limits

- **Free tier**: 10 requests/min, 100/hour
- **Premium tier**: 100 requests/min, 1000/hour
- **Admin tier**: 1000 requests/min, 10000/hour

## 🎯 Demo Application

Run the comprehensive demo:

```bash
python examples/demo_all_improvements.py
```

Demonstrates all 5 improvements with synthetic data.

## 📚 Documentation

- **Full Documentation**: `docs/TOP_5_IMPROVEMENTS_COMPLETE.md`
- **API Reference**: Start API and visit `/docs`
- **Code Examples**: `examples/` directory

## 🐛 Troubleshooting

### Missing Dependencies

```bash
pip install pyjwt anthropic networkx scipy
```

### Import Errors

Ensure you're in the project root:

```bash
cd /home/kevin/Projects/eeg-rag
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### API Key Errors

Set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Rate Limit Errors

Use authentication token to get higher limits:

```python
headers = {"Authorization": f"Bearer {token}"}
```

## 📊 Monitoring

### Check Telemetry

```bash
curl http://localhost:8000/metrics
```

Returns:
- Total requests
- Error rate
- Average response time
- P95/P99 latency
- Status code distribution

### Check Logs

```bash
# Local
tail -f logs/eeg_rag.log

# Kubernetes
kubectl logs -l app=eeg-rag --tail=100 -f
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."
export JWT_SECRET_KEY="your-secret-key"

# Optional
export ANTHROPIC_API_KEY="sk-ant-..."
export OLLAMA_HOST="http://localhost:11434"
export LOG_LEVEL="INFO"
export CHROMA_HOST="localhost"
export CHROMA_PORT="8000"
```

### Configuration File

Create `.env` in project root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
JWT_SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
```

## ✅ Testing

```bash
# Syntax check
python3 -m py_compile src/eeg_rag/generation/*.py
python3 -m py_compile src/eeg_rag/signals/*.py

# Run tests (create test files)
pytest tests/test_response_generator.py
pytest tests/test_eeg_matcher.py
pytest tests/test_feedback_learning.py
```

## 🚀 Next Steps

1. Set up API keys
2. Run demo application
3. Test locally with `uvicorn`
4. Deploy to Kubernetes
5. Integrate with your application

For detailed information, see `docs/TOP_5_IMPROVEMENTS_COMPLETE.md`

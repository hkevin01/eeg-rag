# EEG-RAG Quick Start Guide 🚀

Get the EEG-RAG system running in **5 minutes**.

---

## Prerequisites

- Python 3.9+ installed
- 4GB RAM minimum (8GB recommended)
- Optional: API keys for PubMed and Semantic Scholar

---

## 1. Installation (2 minutes)

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/yourusername/eeg-rag.git
cd eeg-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Configuration (1 minute)

Create a `.env` file in the project root:

```bash
# Required
RESEARCHER_EMAIL=your.email@example.com

# Optional (but recommended for full functionality)
NCBI_API_KEY=your_ncbi_key_here
S2_API_KEY=your_s2_key_here

# ChromaDB (if using external instance)
CHROMA_HOST=localhost
CHROMA_PORT=8000

# API Server
CORS_ORIGINS=*
```

**Getting API Keys (Free):**
- **PubMed**: https://www.ncbi.nlm.nih.gov/account/register/
- **Semantic Scholar**: https://www.semanticscholar.org/product/api

---

## 3. Start the API Server (30 seconds)

```bash
# Option 1: Quick start (development mode)
uvicorn eeg_rag.api.main:app --reload --host 0.0.0.0 --port 8080

# Option 2: Production mode
gunicorn eeg_rag.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080
```

Server will start at: http://localhost:8080

---

## 4. Test It! (30 seconds)

### Option A: Use the Interactive Docs

Visit http://localhost:8080/docs and try the `/search` endpoint:

```json
{
  "query": "deep learning EEG classification",
  "max_results": 10,
  "synthesize": true
}
```

### Option B: Use curl

```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning EEG classification",
    "max_results": 10,
    "synthesize": true
  }'
```

### Option C: Use Python

```python
import requests

response = requests.post(
    "http://localhost:8080/search",
    json={
        "query": "deep learning EEG classification",
        "max_results": 10,
        "synthesize": True
    }
)

result = response.json()
print(f"Found {result['total_found']} papers!")
```

---

## 5. Check Health (10 seconds)

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-23T12:34:56.789",
  "version": "1.0.0",
  "agents": ["local", "pubmed", "s2", "synthesis"],
  "uptime_seconds": 3600.5
}
```

---

## What's Available

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Agent performance metrics |
| `/search` | POST | Standard search |
| `/search/stream` | POST | Streaming search (SSE) |
| `/paper/details` | POST | Get paper metadata |
| `/paper/citations` | POST | Citation network |
| `/suggest` | GET | Query suggestions |
| `/query-types` | GET | Available query types |

### Interactive Documentation

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Schema**: http://localhost:8080/openapi.json

---

## Common Issues

### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill it
kill -9 <PID>

# Or use a different port
uvicorn eeg_rag.api.main:app --port 8888
```

### Import Errors

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### ChromaDB Connection Issues

```bash
# If ChromaDB is not running, the system will use in-memory storage
# To start ChromaDB separately:
docker run -p 8000:8000 chromadb/chroma
```

---

## Next Steps

1. **Explore the API**: Try different queries and parameters
2. **Check Documentation**: Read [WEB_INTEGRATION_COMPLETE.md](WEB_INTEGRATION_COMPLETE.md)
3. **Add Data**: Import your own EEG papers into the local database
4. **Deploy**: See production deployment guide
5. **Frontend**: Build a React UI (see artifacts)

---

## Example Queries

### Basic Search
```json
{
  "query": "seizure detection algorithms",
  "max_results": 20
}
```

### Advanced Search with Filters
```json
{
  "query": "deep learning motor imagery classification",
  "max_results": 50,
  "sources": ["local", "pubmed", "s2"],
  "date_range": [2020, 2025],
  "synthesize": true
}
```

### Citation Network Analysis
```json
{
  "paper_id": "12345678",
  "direction": "both",
  "source": "s2",
  "max_results": 100
}
```

---

## Performance Tips

1. **Use Streaming** for real-time updates:
   ```javascript
   const eventSource = new EventSource('/search/stream?query=EEG');
   eventSource.onmessage = (e) => console.log(JSON.parse(e.data));
   ```

2. **Cache Results** - The system automatically caches:
   - Embeddings (indefinitely)
   - Search results (1 hour)
   - Paper metadata (24 hours)

3. **Parallel Agents** - The orchestrator runs agents in parallel when possible

4. **Limit Sources** - For faster results, specify specific sources:
   ```json
   {"query": "...", "sources": ["local"]}
   ```

---

## Monitoring

### View Metrics
```bash
curl http://localhost:8080/metrics
```

### Check Logs
```bash
# Logs are written to stdout
# Redirect to file:
uvicorn eeg_rag.api.main:app --log-config logging.conf > app.log 2>&1
```

### Health Checks (for Load Balancers)
```bash
# Returns 200 if healthy, 503 if not
curl -f http://localhost:8080/health || echo "Service down!"
```

---

## Development Mode

```bash
# Auto-reload on code changes
uvicorn eeg_rag.api.main:app --reload

# Debug mode (more logging)
export LOG_LEVEL=DEBUG
uvicorn eeg_rag.api.main:app --log-level debug
```

---

## Production Checklist

- [ ] Set strong API keys in `.env`
- [ ] Configure `CORS_ORIGINS` properly (don't use `*`)
- [ ] Use `gunicorn` with multiple workers
- [ ] Set up reverse proxy (nginx/traefik)
- [ ] Enable HTTPS
- [ ] Configure rate limiting
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up automated backups
- [ ] Test failover scenarios

---

## Help & Support

- **Documentation**: See `/docs` folder
- **API Docs**: http://localhost:8080/docs
- **Issues**: Create a GitHub issue
- **Architecture**: See [agentic-rag-architecture.md](docs/agentic-rag-architecture.md)

---

**You're all set!** 🎉

The EEG-RAG API is now running and ready to process literature search queries with multi-agent coordination, real-time progress updates, and intelligent synthesis.

Try a search and watch the agents work together to find, analyze, and synthesize EEG research literature!

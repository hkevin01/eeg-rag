# EEG-RAG Installation Guide

This guide covers all installation methods for EEG-RAG. Choose the method that best fits your needs.

## 📋 Requirements

| Requirement | Minimum                     | Recommended |
| ----------- | --------------------------- | ----------- |
| Python      | 3.9+                        | 3.11+       |
| RAM         | 8 GB (for Mistral)          | 16 GB       |
| Disk Space  | 6 GB (includes Mistral)     | 15 GB       |
| OS          | Linux, macOS, Windows (WSL) | Linux/macOS |

**Note:** EEG-RAG now includes **Mistral 7B** via Ollama for local AI responses (no API key needed).

**Optional (for advanced features):**
- Docker & Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA (for faster embeddings)
- Neo4j 5.x (for knowledge graph features)
- Redis (for caching)

---

## 🚀 Quick Start (Recommended)

### Option 1: One-Command Setup

```bash
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag
./scripts/setup.sh
```

This script will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ **Install Ollama + Mistral 7B** (local LLM)
- ✅ Set up data directories
- ✅ Create config file template

**After setup:**
```bash
source venv/bin/activate
streamlit run src/eeg_rag/web_ui/app.py
```

### Option 2: Manual pip Install

```bash
# Clone repository
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama for local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull mistral

# Install
pip install -e .

# Set up directories
mkdir -p data/raw data/processed data/embeddings logs
```

---

## 🐳 Docker Installation

Docker provides the most reproducible environment with all dependencies pre-configured.

### Quick Docker Start (Lite)

Minimal setup without Neo4j/Redis:

```bash
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag
make docker-up-lite
```

Or without make:
```bash
docker compose -f docker/docker-compose.lite.yml up -d
```

### Full Docker Stack

Includes Neo4j (knowledge graph) and Redis (caching):

```bash
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag
make docker-up
```

**Access points:**
- EEG-RAG API: http://localhost:8000
- Neo4j Browser: http://localhost:7474
- Redis: localhost:6379

### Docker Commands

```bash
make docker-build    # Build image
make docker-up       # Start all services
make docker-down     # Stop all services
make docker-logs     # View logs
make docker-shell    # Shell into container
```

---

## �️ Web UI

EEG-RAG includes a Streamlit-based web interface for easy interaction.

### Starting the Web UI

```bash
# Using make
make ui

# Or directly
source .venv/bin/activate  # If using venv
streamlit run src/eeg_rag/web_ui/app.py
```

**Access at:** http://localhost:8501

### Web UI Features

- **🔍 Query System** - Ask questions about EEG research literature
- **📊 Systematic Review Benchmark** - Test extraction accuracy against Roy et al. 2019 ground truth
- **📈 Results Dashboard** - View and export benchmark results
- **⚙️ Settings** - Configure corpus, models, and API keys

### Download Benchmark Data

For the systematic review benchmark feature:

```bash
make download-benchmark-data
# Or manually:
curl -o data/systematic_review/roy_et_al_2019_data_items.csv \
  https://raw.githubusercontent.com/hubertjb/dl-eeg-review/master/data/data_items.csv
```

---

## �🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for LLM features
OPENAI_API_KEY=sk-your-api-key-here

# Recommended for PubMed API
PUBMED_EMAIL=your-email@example.com

# Optional: Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Optional: Redis
REDIS_URL=redis://localhost:6379
```

### Getting API Keys

1. **OpenAI API Key** (Required for LLM features)
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key
   - Add to `.env` as `OPENAI_API_KEY`

2. **PubMed Email** (Recommended)
   - Use any valid email address
   - Required for higher API rate limits

---

## 📦 Installation Options

### Basic Installation

```bash
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

Includes: pytest, black, pylint, mypy, jupyter

### With Knowledge Graph Support

```bash
pip install -e ".[knowledge-graph]"
```

Includes: neo4j, redis

### Full Installation

```bash
pip install -e ".[full]"
```

Includes all optional dependencies.

---

## ✅ Verify Installation

### Run Tests

```bash
make test
# Or: python -m pytest tests/ -v
```

### Run Demo

```bash
python examples/demo_all_components.py
```

### Check Version

```python
import eeg_rag
print(eeg_rag.__version__)
```

---

## 🔧 Troubleshooting

### Common Issues

**1. PyTorch/CUDA Issues**
```bash
# Install CPU-only PyTorch if CUDA is not available
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**2. FAISS Installation Fails**
```bash
# Try the CPU version
pip install faiss-cpu
```

**3. Memory Issues with Large Models**
```bash
# Use smaller embedding model
export EEG_RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**4. Docker Permission Denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Platform-Specific Notes

**macOS (Apple Silicon)**
```bash
# Install with Rosetta for x86 compatibility if needed
arch -x86_64 pip install -e .
```

**Windows**
- Use WSL2 for best compatibility
- Or use Docker Desktop

**Linux**
- Works out of the box on most distributions
- Ensure Python 3.9+ is installed

---

## 📁 Project Structure After Installation

```
eeg-rag/
├── data/
│   ├── raw/          # Raw documents
│   ├── processed/    # Processed chunks
│   └── embeddings/   # Vector embeddings
├── logs/             # Application logs
├── .env              # Configuration (create this)
└── venv/             # Virtual environment
```

---

## 🆘 Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: https://github.com/hkevin01/eeg-rag/issues
- **Discussions**: https://github.com/hkevin01/eeg-rag/discussions

---

## 🎉 Next Steps

1. **Add your EEG literature**: Place PDFs in `data/raw/`
2. **Process documents**: Run the indexing pipeline
3. **Start querying**: Use the CLI or API

See the [README](README.md) for usage examples.

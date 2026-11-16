# EEG-RAG Installation Guide

This guide covers all installation methods for EEG-RAG. Choose the method that best fits your needs.

## 📋 Requirements

| Requirement | Minimum                     | Recommended |
| ----------- | --------------------------- | ----------- |
| Python      | 3.11-3.14                   | 3.12        |
| RAM         | 8 GB for local model work   | 16 GB       |
| Disk Space  | Several GB for dependencies | Additional space for corpus/models |
| OS          | Linux, macOS, Windows (WSL) | Linux/macOS |

**Note:** A GPU and a paid LLM key are not required to install or open the browser
interface. Generated answers require a configured provider. Ollama and its models
are optional and are not downloaded by the recommended setup.

**Optional (for advanced features):**
- Docker & Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA (for faster embeddings)
- Neo4j 5.x (for knowledge graph features)
- Redis (for caching)

---

## 🚀 Quick Start (Recommended)

### Option 1: Researcher Setup

Install Python 3.12 from <https://www.python.org/downloads/>. Download this project
as a ZIP using GitHub's **Code > Download ZIP**, extract it, and open a terminal
in the extracted folder. Git users can instead clone the repository:

```bash
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag
python scripts/install_researcher.py
```

Use `python3` on Linux/macOS if `python` is unavailable, or `py -3.12` on Windows.
The script checks Python, installs CPU-only PyTorch and the researcher/browser
dependencies in an environment outside the project, prepares data directories,
and copies the example configuration only when no configuration exists. It uses
uv if available, otherwise Python/pip. It needs internet access and disk space,
but no activation command, compiler, Docker, or administrator access.

**After setup:**
```bash
python scripts/install_researcher.py --launch
```

Open `http://127.0.0.1:8501`. To use another port:

```sh
python scripts/install_researcher.py --launch --port 8502
```

Use `--check` to see installed versions and `--plan` to preview commands. Rerun
the same install command to retry or update the checkout; existing configuration
is preserved. `--environment /path/outside/project` chooses a custom environment.
Researcher setup does not start Qdrant, ingest a corpus, download embedding models,
or configure an answer provider. The Settings page and the guides explain those
next steps. Supply your PubMed contact email in the local configuration before
making sustained literature requests.

### Option 2: Advanced Manual Install

```bash
# Clone repository
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag

# Linux/macOS example: keep the environment outside the project
python3.12 -m venv "$HOME/.local/share/eeg-rag/manual"
source "$HOME/.local/share/eeg-rag/manual/bin/activate"

# Install Python dependencies
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install
python -m pip install -c requirements-researcher.txt -e ".[researcher]"

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
python scripts/install_researcher.py --launch
```

**Access at:** http://localhost:8501

Advanced citation-network analysis uses the optional `bibliometrics` extra.
Its native dependencies may need a compiler; they are intentionally absent from
the default researcher installation. See [today's compatibility notes](docs/releases/2026-11-16.md).

### Setup Troubleshooting

- **Python not found:** install Python 3.12, reopen the terminal, and use `python3`
   (Linux/macOS) or `py -3.12` (Windows).
- **Unsupported Python:** select 3.11-3.14; no system Python is replaced by setup.
- **Download failure:** check connectivity, firewall/proxy rules, and free disk
   space, then rerun the same command. Institution-managed systems may need IT to
   allow PyPI and the official PyTorch download domain.
- **Missing pip/venv support:** use your platform's supported Python installation
   or an installed uv. The script never runs administrator commands for you.
- **Port already used:** launch with `--port 8502`.
- **No generated answer:** installation is separate from provider configuration.
   Configure a local or cloud LLM; keep credentials out of shared files and Git.

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
# Optional: Enables OpenAI provider in readiness ranking
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

### Generation Provider Readiness and Fallback

EEG-RAG ranks generation providers from live telemetry, not static constants.

- Readiness score uses observed success rate, latency, failure patterns, and provider ordering.
- Runtime tuning is provider-specific (timeout and retry settings adjust from recent failures).
- Providers without usable credentials or backend connectivity are excluded from active generation.
- Ollama remains the local path when no API key is present, as long as the Ollama server is running.

Operationally:

- If `OPENAI_API_KEY` is set and valid, OpenAI participates in readiness ranking.
- If `OPENAI_API_KEY` is not set, OpenAI is skipped and local backends can still serve requests.
- Readiness and tuning details are exposed through the existing stats health endpoint.

### Getting API Keys

1. **OpenAI API Key** (Optional, for OpenAI provider)
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
- Ensure Python 3.11-3.14 is installed (3.12 recommended)

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

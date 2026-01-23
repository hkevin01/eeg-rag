# EEG-RAG Agent Dashboard

An enhanced Streamlit interface with **full agent visibility** showing real-time progress, execution timeline, and query metrics.

## Features

### 🤖 Agent Activity Panel
Shows all 9 agents with real-time status:
- **Orchestrator** - Coordinates all agents and routes queries
- **Query Planner** - Decomposes complex queries into sub-tasks
- **Local Data Agent** - FAISS vector search over indexed papers
- **PubMed Agent** - Queries PubMed API for papers
- **Semantic Scholar** - Queries Semantic Scholar API
- **Knowledge Graph** - Traverses Neo4j knowledge graph
- **Citation Validator** - Verifies PMIDs and validates citations
- **Context Aggregator** - Merges and deduplicates results
- **Response Synthesizer** - LLM-based response generation

### 📜 Execution Timeline
Chronological log of what each agent did, with timestamps and color-coded events (info/success/error/warning).

### 📊 Query Metrics
- Total latency (seconds)
- Papers searched (K)
- Citations verified

### 📚 Source Display
Each source shows:
- Title, authors, year, journal
- Domain, architecture, accuracy
- Citation count
- Direct links to PubMed, DOI, and Google Scholar

## Running the Dashboard

```bash
# From project root
cd /home/kevin/Projects/eeg-rag
source .venv/bin/activate

# Run the Agent Dashboard
streamlit run src/eeg_rag/web_ui/app_agent_dashboard.py --server.port 8503

# Or run the original app
streamlit run src/eeg_rag/web_ui/app.py --server.port 8501
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EEG-RAG Agent Dashboard                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │    Main Content Area    │  │     Agent Activity Panel     │  │
│  │                         │  │                              │  │
│  │  ┌───────────────────┐  │  │  🧠 Orchestrator      ✅     │  │
│  │  │   Search Query    │  │  │  📋 Query Planner     ✅     │  │
│  │  └───────────────────┘  │  │  💾 Local Agent       ✅     │  │
│  │                         │  │  🏥 PubMed Agent      ✅     │  │
│  │  ┌───────────────────┐  │  │  🎓 Semantic Scholar  ✅     │  │
│  │  │ Execution Timeline│  │  │  🕸️ Knowledge Graph   ✅     │  │
│  │  │  (expandable)     │  │  │  ✅ Citation Agent    ✅     │  │
│  │  └───────────────────┘  │  │  🔀 Aggregator        ✅     │  │
│  │                         │  │  ✍️ Synthesizer       ✅     │  │
│  │  ┌───────────────────┐  │  │                              │  │
│  │  │   AI Response     │  │  ├──────────────────────────────┤  │
│  │  │   with Citations  │  │  │     Query Metrics            │  │
│  │  └───────────────────┘  │  │  ┌────────┬────────┬────────┐│  │
│  │                         │  │  │ 3.2s   │  125K  │   25   ││  │
│  │  ┌───────────────────┐  │  │  │Latency │Searched│Verified││  │
│  │  │ Verified Sources  │  │  │  └────────┴────────┴────────┘│  │
│  │  │ (expandable)      │  │  └──────────────────────────────┘  │
│  │  └───────────────────┘  │                                    │
│  └─────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with Real Agents

The dashboard currently uses a simulation mode for demo purposes. To connect to the real agent system:

1. Ensure all dependencies are installed
2. Configure API keys (NCBI_API_KEY, S2_API_KEY, OPENAI_API_KEY)
3. The dashboard will automatically detect and use the OrchestratorAgent

The `execute_with_agent_tracking()` function first tries to import real agents, and falls back to simulation if not available.

## Files

- `app_agent_dashboard.py` - The new agent-visible dashboard
- `app.py` - Original Streamlit app (still works)
- `app_multipage.py` - Multi-page app entry point
- `pages/` - Additional Streamlit pages

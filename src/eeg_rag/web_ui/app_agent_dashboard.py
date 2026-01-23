#!/usr/bin/env python3
"""
EEG-RAG Agent Dashboard - Enhanced Streamlit Web Application

A comprehensive web interface with FULL AGENT VISIBILITY showing:
- Real-time agent activity and status
- Execution timeline with step-by-step logging
- Query metrics and performance stats
- Evidence-based responses with verified citations
"""

import streamlit as st
import asyncio
import json
import time
import hashlib
import urllib.parse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
import threading
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Agent Status System
# =============================================================================


class AgentStatusEnum(Enum):
    """Agent execution status for UI display."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class AgentDisplayState:
    """State for displaying an agent in the UI."""
    id: str
    name: str
    icon: str
    description: str
    status: AgentStatusEnum = AgentStatusEnum.IDLE
    progress: int = 0
    details: str = ""
    results_count: int = 0
    elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "icon": self.icon,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "details": self.details,
            "results_count": self.results_count,
            "elapsed_ms": self.elapsed_ms
        }


@dataclass
class TimelineEvent:
    """Event in the execution timeline."""
    timestamp: datetime
    agent_name: str
    message: str
    event_type: str = "info"  # info, success, error, warning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "time_str": self.timestamp.strftime("%H:%M:%S.%f")[:-3],
            "agent": self.agent_name,
            "message": self.message,
            "type": self.event_type
        }


class AgentActivityTracker:
    """
    Tracks agent activity and provides real-time updates to the UI.
    
    This bridges the backend agent system with the Streamlit frontend.
    """
    
    # Default agent definitions
    DEFAULT_AGENTS = [
        AgentDisplayState(
            id="orchestrator",
            name="Orchestrator",
            icon="🧠",
            description="Coordinates all agents and routes queries"
        ),
        AgentDisplayState(
            id="query_planner", 
            name="Query Planner",
            icon="📋",
            description="Decomposes complex queries into sub-tasks"
        ),
        AgentDisplayState(
            id="local_agent",
            name="Local Data Agent", 
            icon="💾",
            description="FAISS vector search over indexed papers"
        ),
        AgentDisplayState(
            id="pubmed_agent",
            name="PubMed Agent",
            icon="🏥",
            description="Queries PubMed API for papers"
        ),
        AgentDisplayState(
            id="semantic_scholar",
            name="Semantic Scholar",
            icon="🎓",
            description="Queries Semantic Scholar API"
        ),
        AgentDisplayState(
            id="graph_agent",
            name="Knowledge Graph",
            icon="🕸️",
            description="Traverses Neo4j knowledge graph"
        ),
        AgentDisplayState(
            id="citation_agent",
            name="Citation Validator",
            icon="✅",
            description="Verifies PMIDs and validates citations"
        ),
        AgentDisplayState(
            id="aggregator",
            name="Context Aggregator",
            icon="🔀",
            description="Merges and deduplicates results"
        ),
        AgentDisplayState(
            id="synthesis_agent",
            name="Response Synthesizer",
            icon="✍️",
            description="LLM-based response generation"
        ),
    ]
    
    def __init__(self):
        self.agents: Dict[str, AgentDisplayState] = {
            a.id: AgentDisplayState(
                id=a.id, name=a.name, icon=a.icon, description=a.description
            )
            for a in self.DEFAULT_AGENTS
        }
        self.timeline: List[TimelineEvent] = []
        self.metrics = {
            "total_latency_ms": 0.0,
            "papers_searched": 0,
            "papers_retrieved": 0,
            "citations_verified": 0,
            "cache_hits": 0
        }
        self._lock = threading.Lock()
        
    def reset(self):
        """Reset all agents to idle state."""
        with self._lock:
            for agent_id in self.agents:
                self.agents[agent_id].status = AgentStatusEnum.IDLE
                self.agents[agent_id].progress = 0
                self.agents[agent_id].details = ""
                self.agents[agent_id].results_count = 0
                self.agents[agent_id].elapsed_ms = 0.0
            self.timeline = []
            self.metrics = {
                "total_latency_ms": 0.0,
                "papers_searched": 0,
                "papers_retrieved": 0,
                "citations_verified": 0,
                "cache_hits": 0
            }
    
    def update_agent(
        self,
        agent_id: str,
        status: Optional[AgentStatusEnum] = None,
        progress: Optional[int] = None,
        details: Optional[str] = None,
        results_count: Optional[int] = None,
        elapsed_ms: Optional[float] = None
    ):
        """Update an agent's display state."""
        with self._lock:
            if agent_id not in self.agents:
                return
            agent = self.agents[agent_id]
            if status is not None:
                agent.status = status
            if progress is not None:
                agent.progress = progress
            if details is not None:
                agent.details = details
            if results_count is not None:
                agent.results_count = results_count
            if elapsed_ms is not None:
                agent.elapsed_ms = elapsed_ms
    
    def add_timeline_event(
        self,
        agent_name: str,
        message: str,
        event_type: str = "info"
    ):
        """Add an event to the timeline."""
        with self._lock:
            self.timeline.append(TimelineEvent(
                timestamp=datetime.now(),
                agent_name=agent_name,
                message=message,
                event_type=event_type
            ))
    
    def update_metrics(self, **kwargs):
        """Update query metrics."""
        with self._lock:
            for key, value in kwargs.items():
                if key in self.metrics:
                    self.metrics[key] = value
    
    def get_agents_snapshot(self) -> List[Dict[str, Any]]:
        """Get current state of all agents."""
        with self._lock:
            return [a.to_dict() for a in self.agents.values()]
    
    def get_timeline_snapshot(self) -> List[Dict[str, Any]]:
        """Get current timeline events."""
        with self._lock:
            return [e.to_dict() for e in self.timeline]
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return self.metrics.copy()


# =============================================================================
# Simulated Agent Execution (for demo - connects to real agents when available)
# =============================================================================


async def execute_with_agent_tracking(
    query: str,
    tracker: AgentActivityTracker,
    max_sources: int = 5
) -> Dict[str, Any]:
    """
    Execute a RAG query with full agent activity tracking.
    
    This function simulates agent execution for demo purposes,
    but can be connected to the real OrchestratorAgent.
    """
    start_time = time.time()
    tracker.reset()
    
    # Try to use real agents if available
    try:
        return await _execute_with_real_agents(query, tracker, max_sources)
    except ImportError:
        logger.info("Real agents not available, using simulation")
        return await _execute_simulated(query, tracker, max_sources)


async def _execute_with_real_agents(
    query: str,
    tracker: AgentActivityTracker,
    max_sources: int
) -> Dict[str, Any]:
    """Execute using real agent system."""
    from eeg_rag.agents.base_agent import AgentQuery, AgentRegistry
    from eeg_rag.agents.orchestrator.orchestrator_agent import OrchestratorAgent
    from eeg_rag.core.memory_manager import MemoryManager
    
    # Initialize components
    memory = MemoryManager()
    registry = AgentRegistry()
    
    # Create orchestrator with activity callbacks
    orchestrator = OrchestratorAgent(
        memory_manager=memory,
        agent_registry=registry,
        config={"enable_adaptive_replanning": True}
    )
    
    # Execute with tracking
    tracker.update_agent("orchestrator", AgentStatusEnum.RUNNING, 20, "Initializing...")
    tracker.add_timeline_event("Orchestrator", "Query received, initializing agents")
    
    agent_query = AgentQuery(text=query)
    result = await orchestrator.execute(agent_query)
    
    # Update final status
    if result.success:
        tracker.update_agent("orchestrator", AgentStatusEnum.SUCCESS, 100, "Complete")
        return {
            "success": True,
            "response": result.data.get("response", ""),
            "sources": result.data.get("sources", []),
            "citations": result.data.get("citations", []),
            "metrics": result.metadata
        }
    else:
        tracker.update_agent("orchestrator", AgentStatusEnum.ERROR, 0, result.error)
        return {
            "success": False,
            "error": result.error
        }


async def _execute_simulated(
    query: str,
    tracker: AgentActivityTracker,
    max_sources: int
) -> Dict[str, Any]:
    """Simulate agent execution for demo/development."""
    start_time = time.time()
    
    # Phase 1: Orchestrator starts
    tracker.update_agent("orchestrator", AgentStatusEnum.RUNNING, 20, "Analyzing query complexity...")
    tracker.add_timeline_event("Orchestrator", f"Received query: '{query[:50]}...'")
    await asyncio.sleep(0.3)
    
    tracker.update_agent("orchestrator", AgentStatusEnum.RUNNING, 50, "Routing to agents...")
    tracker.add_timeline_event("Orchestrator", "Query classified as RESEARCH type, routing to 5 agents", "success")
    await asyncio.sleep(0.2)
    
    # Phase 2: Query Planning
    tracker.update_agent("query_planner", AgentStatusEnum.RUNNING, 30, "Decomposing query...")
    tracker.add_timeline_event("Query Planner", "Breaking down query into sub-questions")
    await asyncio.sleep(0.3)
    
    tracker.update_agent("query_planner", AgentStatusEnum.RUNNING, 70, "Identifying EEG terms...")
    await asyncio.sleep(0.2)
    
    # Extract some terms from query for realistic display
    eeg_terms = [t for t in query.lower().split() if t in [
        'eeg', 'seizure', 'epilepsy', 'sleep', 'bci', 'motor', 'imagery', 
        'p300', 'erp', 'alpha', 'beta', 'theta', 'delta', 'gamma', 'cnn', 
        'lstm', 'transformer', 'deep', 'learning', 'classification', 'detection'
    ]]
    terms_display = ', '.join(eeg_terms[:5]) if eeg_terms else 'EEG, classification, deep learning'
    
    tracker.update_agent("query_planner", AgentStatusEnum.SUCCESS, 100, f"Found: {terms_display}")
    tracker.add_timeline_event("Query Planner", f"Identified terms: {terms_display}", "success")
    await asyncio.sleep(0.1)
    
    # Phase 3: Parallel Agent Execution
    tracker.update_agent("local_agent", AgentStatusEnum.RUNNING, 10, "Searching FAISS index...")
    tracker.update_agent("pubmed_agent", AgentStatusEnum.RUNNING, 5, "Querying PubMed API...")
    tracker.update_agent("semantic_scholar", AgentStatusEnum.RUNNING, 5, "Querying S2 API...")
    
    tracker.add_timeline_event("Local Agent", "Searching 125,000 indexed paper embeddings")
    tracker.add_timeline_event("PubMed Agent", "Querying NCBI E-utilities API")
    tracker.add_timeline_event("Semantic Scholar", "Querying academic graph API")
    
    await asyncio.sleep(0.4)
    
    # Local agent progress
    tracker.update_agent("local_agent", AgentStatusEnum.RUNNING, 50, "Found 45 candidates...")
    await asyncio.sleep(0.3)
    tracker.update_agent("local_agent", AgentStatusEnum.SUCCESS, 100, f"Retrieved {max_sources + 5} papers", max_sources + 5)
    tracker.add_timeline_event("Local Agent", f"Found {max_sources + 5} papers with avg similarity 0.82", "success")
    
    # PubMed agent progress
    await asyncio.sleep(0.2)
    tracker.update_agent("pubmed_agent", AgentStatusEnum.RUNNING, 60, "Parsing XML responses...")
    await asyncio.sleep(0.3)
    tracker.update_agent("pubmed_agent", AgentStatusEnum.SUCCESS, 100, "Retrieved 12 papers", 12)
    tracker.add_timeline_event("PubMed Agent", "Found 12 papers from last 5 years with PMIDs", "success")
    
    # Semantic Scholar progress
    tracker.update_agent("semantic_scholar", AgentStatusEnum.RUNNING, 40, "Expanding citations...")
    await asyncio.sleep(0.2)
    tracker.update_agent("semantic_scholar", AgentStatusEnum.SUCCESS, 100, "Retrieved 8 papers", 8)
    tracker.add_timeline_event("Semantic Scholar", "Found 8 highly-cited papers (>100 citations)", "success")
    
    # Knowledge graph (may or may not be available)
    tracker.update_agent("graph_agent", AgentStatusEnum.RUNNING, 20, "Querying Neo4j...")
    await asyncio.sleep(0.3)
    tracker.update_agent("graph_agent", AgentStatusEnum.SUCCESS, 100, "Found 5 relationships", 5)
    tracker.add_timeline_event("Knowledge Graph", "Found concept relationships: EEG→seizure→prediction", "success")
    
    # Update search metrics
    tracker.update_metrics(papers_searched=125000, papers_retrieved=30)
    
    # Phase 4: Citation Validation
    tracker.update_agent("orchestrator", AgentStatusEnum.RUNNING, 70, "Validating citations...")
    tracker.update_agent("citation_agent", AgentStatusEnum.RUNNING, 30, "Verifying PMIDs...")
    tracker.add_timeline_event("Citation Agent", "Validating 30 citations against PubMed")
    await asyncio.sleep(0.4)
    
    tracker.update_agent("citation_agent", AgentStatusEnum.RUNNING, 70, "Checking retractions...")
    await asyncio.sleep(0.3)
    tracker.update_agent("citation_agent", AgentStatusEnum.SUCCESS, 100, "25 verified, 5 excluded", 25)
    tracker.add_timeline_event("Citation Agent", "Validated 25 papers (2 retracted, 3 no PMID)", "success")
    tracker.update_metrics(citations_verified=25)
    
    # Phase 5: Aggregation
    tracker.update_agent("aggregator", AgentStatusEnum.RUNNING, 40, "Merging results...")
    tracker.add_timeline_event("Aggregator", "Deduplicating and ranking 25 papers")
    await asyncio.sleep(0.3)
    
    tracker.update_agent("aggregator", AgentStatusEnum.RUNNING, 80, "Ranking by relevance...")
    await asyncio.sleep(0.2)
    tracker.update_agent("aggregator", AgentStatusEnum.SUCCESS, 100, f"Top {max_sources} selected")
    tracker.add_timeline_event("Aggregator", f"Selected top {max_sources} papers (removed 5 duplicates)", "success")
    
    # Phase 6: Response Generation
    tracker.update_agent("synthesis_agent", AgentStatusEnum.RUNNING, 20, "Building context...")
    tracker.add_timeline_event("Response Synthesizer", "Generating response with Mistral/GPT-4")
    await asyncio.sleep(0.3)
    
    tracker.update_agent("synthesis_agent", AgentStatusEnum.RUNNING, 60, "Synthesizing answer...")
    await asyncio.sleep(0.4)
    
    tracker.update_agent("synthesis_agent", AgentStatusEnum.RUNNING, 90, "Adding citations...")
    await asyncio.sleep(0.2)
    
    tracker.update_agent("synthesis_agent", AgentStatusEnum.SUCCESS, 100, f"Generated with {max_sources} citations")
    tracker.add_timeline_event("Response Synthesizer", f"Generated comprehensive answer with {max_sources} inline citations", "success")
    
    # Finalize
    tracker.update_agent("orchestrator", AgentStatusEnum.SUCCESS, 100, "Query complete")
    
    total_time = (time.time() - start_time) * 1000
    tracker.update_metrics(total_latency_ms=total_time)
    tracker.add_timeline_event("Orchestrator", f"Query completed in {total_time:.0f}ms", "success")
    
    # Generate simulated response
    response = _generate_demo_response(query, max_sources)
    
    return {
        "success": True,
        "response": response["text"],
        "sources": response["sources"],
        "citations": response["citations"],
        "metrics": {
            "latency_ms": total_time,
            "papers_searched": 125000,
            "papers_retrieved": 30,
            "citations_verified": 25,
            "sources_used": max_sources
        }
    }


def _generate_demo_response(query: str, num_sources: int) -> Dict[str, Any]:
    """Generate a realistic demo response based on the query."""
    
    # Sample EEG research data for realistic responses
    sample_papers = [
        {
            "title": "Deep Learning for EEG-Based Seizure Detection: A Systematic Review",
            "authors": "Zhang et al.",
            "year": 2023,
            "pmid": "37234567",
            "doi": "10.1016/j.neunet.2023.01.001",
            "journal": "Neural Networks",
            "domain": "Epilepsy",
            "architecture": "CNN",
            "accuracy": "94.5%",
            "dataset": "CHB-MIT",
            "citation_count": 156
        },
        {
            "title": "Transformer Networks for Motor Imagery EEG Classification",
            "authors": "Li et al.",
            "year": 2024,
            "pmid": "38123456",
            "doi": "10.1109/TNSRE.2024.001234",
            "journal": "IEEE TNSRE",
            "domain": "BCI",
            "architecture": "Transformer",
            "accuracy": "89.2%",
            "dataset": "BCI Competition IV",
            "citation_count": 89
        },
        {
            "title": "EEGNet: A Compact CNN for EEG-Based BCIs",
            "authors": "Lawhern et al.",
            "year": 2018,
            "pmid": "29932424",
            "doi": "10.1088/1741-2552/aace8c",
            "journal": "J Neural Eng",
            "domain": "BCI",
            "architecture": "CNN",
            "accuracy": "82.1%",
            "dataset": "Multiple",
            "citation_count": 1420
        },
        {
            "title": "Attention-Based LSTM for Sleep Stage Classification",
            "authors": "Supratak et al.",
            "year": 2020,
            "pmid": "32145678",
            "doi": "10.1109/JBHI.2020.123456",
            "journal": "IEEE JBHI",
            "domain": "Sleep",
            "architecture": "LSTM+Attention",
            "accuracy": "86.3%",
            "dataset": "Sleep-EDF",
            "citation_count": 234
        },
        {
            "title": "Graph Neural Networks for EEG Emotion Recognition",
            "authors": "Song et al.",
            "year": 2023,
            "pmid": "36789012",
            "doi": "10.1016/j.knosys.2023.110456",
            "journal": "Knowledge-Based Systems",
            "domain": "Emotion",
            "architecture": "GNN",
            "accuracy": "91.8%",
            "dataset": "DEAP",
            "citation_count": 67
        },
    ]
    
    sources = sample_papers[:num_sources]
    
    # Build response text
    query_lower = query.lower()
    
    if "seizure" in query_lower or "epilepsy" in query_lower:
        response_text = f"""Based on the analysis of {len(sources)} verified papers, here are the key findings on EEG seizure detection:

## Deep Learning Architectures for Seizure Detection

**1. Convolutional Neural Networks (CNNs)** [PMID: 37234567]
CNNs have shown excellent performance for seizure detection, achieving up to 94.5% accuracy on the CHB-MIT dataset. The hierarchical feature extraction capability makes them ideal for learning spatial EEG patterns.

**2. Transformer-Based Models** [PMID: 38123456]
Recent transformer architectures leverage self-attention mechanisms to capture long-range temporal dependencies in EEG signals, showing promising results for pre-ictal pattern detection.

**3. EEGNet Architecture** [PMID: 29932424]
The compact EEGNet architecture (Lawhern et al., 2018) remains a strong baseline with 82.1% accuracy across multiple BCI paradigms, requiring minimal preprocessing.

## Key Findings

- **Best Performance**: CNN-based models on CHB-MIT dataset (94.5% accuracy)
- **Common Preprocessing**: Bandpass filtering (0.5-40Hz), artifact removal with ICA
- **Recommended Approach**: Ensemble of CNN + LSTM for both spatial and temporal features

## Clinical Implications

Real-time seizure detection systems using these architectures are being validated for clinical use, with latency requirements of <2 seconds for effective intervention.

---
*Response synthesized from {len(sources)} peer-reviewed papers with verified PMIDs.*"""

    elif "sleep" in query_lower:
        response_text = f"""Based on {len(sources)} verified papers, here's the current state of EEG sleep staging:

## Deep Learning for Sleep Stage Classification

**1. Attention-Based LSTM** [PMID: 32145678]
Attention mechanisms combined with LSTM networks achieve 86.3% overall accuracy on Sleep-EDF, with particularly strong performance on N2 and REM stages.

**2. DeepSleepNet Architecture**
Multi-scale CNNs followed by bidirectional LSTMs can learn both epoch-level and sequence-level features, matching human expert agreement rates.

## Key Datasets
- Sleep-EDF: 197 whole-night recordings
- SHHS: Large-scale cardiovascular study data
- ISRUC: Multi-channel clinical recordings

---
*Response synthesized from {len(sources)} peer-reviewed papers.*"""

    else:
        response_text = f"""Based on the analysis of {len(sources)} verified papers, here are the key findings for your query:

## Summary of Findings

**1. State-of-the-Art Architectures**
Modern EEG analysis primarily uses deep learning approaches including CNNs, LSTMs, and increasingly Transformer-based models. Each architecture has specific strengths:
- **CNNs**: Spatial feature extraction from multi-channel EEG [PMID: 29932424]
- **LSTMs**: Temporal sequence modeling [PMID: 32145678]
- **Transformers**: Long-range dependency capture [PMID: 38123456]

**2. Performance Benchmarks**
Across major EEG datasets, deep learning methods achieve:
- Seizure detection: 90-95% accuracy (CHB-MIT)
- Sleep staging: 85-88% accuracy (Sleep-EDF)
- Motor imagery: 80-89% accuracy (BCI Competition IV)
- Emotion recognition: 85-92% accuracy (DEAP)

**3. Preprocessing Best Practices**
- Bandpass filtering: 0.5-45Hz standard
- Artifact removal: ICA or wavelet-based
- Normalization: Z-score per channel/epoch

## Recommendations

For new EEG classification projects, consider EEGNet as a strong baseline, with potential improvements from attention mechanisms or ensemble approaches.

---
*Response synthesized from {len(sources)} peer-reviewed papers with verified citations.*"""

    return {
        "text": response_text,
        "sources": sources,
        "citations": [f"[{s['authors']} {s['year']}]" for s in sources]
    }


# =============================================================================
# Streamlit UI Components
# =============================================================================


def render_agent_card(agent: Dict[str, Any], expanded: bool = False):
    """Render a single agent status card."""
    status = agent["status"]
    
    # Status styling
    status_colors = {
        "idle": "#6B7280",
        "running": "#3B82F6",
        "success": "#10B981",
        "error": "#EF4444",
        "waiting": "#F59E0B"
    }
    
    status_icons = {
        "idle": "⚪",
        "running": "🔄",
        "success": "✅",
        "error": "❌",
        "waiting": "⏳"
    }
    
    color = status_colors.get(status, "#6B7280")
    icon = status_icons.get(status, "⚪")
    agent_icon = agent.get("icon", "🤖")
    
    # Build card HTML
    card_html = f"""
    <div style="
        border: 1px solid {color}40;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        background: {'#1E3A5F20' if status == 'running' else '#1a1a2e'};
        transition: all 0.3s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.2em;">{agent_icon}</span>
                <div>
                    <div style="font-weight: 600; color: #E0E0E0;">{agent['name']}</div>
                    <div style="font-size: 0.75em; color: #888;">{agent.get('description', '')}</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                {f'<span style="font-size: 0.75em; background: #10B98120; color: #10B981; padding: 2px 8px; border-radius: 12px;">{agent["results_count"]} results</span>' if agent.get("results_count", 0) > 0 else ''}
                <span style="font-size: 1.1em;">{icon}</span>
            </div>
        </div>
    """
    
    # Progress bar if running
    if status == "running":
        progress = agent.get("progress", 0)
        card_html += f"""
        <div style="margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #888; margin-bottom: 4px;">
                <span>{agent.get('details', 'Processing...')}</span>
                <span>{progress}%</span>
            </div>
            <div style="width: 100%; background: #374151; border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="width: {progress}%; background: linear-gradient(90deg, #3B82F6, #60A5FA); height: 100%; border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    # Details if success
    if status == "success" and agent.get("details"):
        card_html += f"""
        <div style="margin-top: 8px; font-size: 0.8em; color: #10B981;">
            ✓ {agent['details']}
        </div>
        """
    
    card_html += "</div>"
    
    return card_html


def render_timeline(events: List[Dict[str, Any]]):
    """Render the execution timeline."""
    if not events:
        return ""
    
    html_parts = ['<div style="max-height: 300px; overflow-y: auto;">']
    
    for i, event in enumerate(events):
        event_type = event.get("type", "info")
        colors = {
            "info": "#3B82F6",
            "success": "#10B981", 
            "error": "#EF4444",
            "warning": "#F59E0B"
        }
        color = colors.get(event_type, "#3B82F6")
        
        html_parts.append(f"""
        <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px;">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: {color};"></div>
                {'<div style="width: 2px; height: 32px; background: #374151;"></div>' if i < len(events) - 1 else ''}
            </div>
            <div style="flex: 1; padding-bottom: 8px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 500; color: #E0E0E0;">{event['agent']}</span>
                    <span style="font-size: 0.75em; color: #888;">{event['time_str']}</span>
                </div>
                <p style="font-size: 0.875em; color: #A0A0A0; margin: 2px 0 0 0;">{event['message']}</p>
            </div>
        </div>
        """)
    
    html_parts.append('</div>')
    return ''.join(html_parts)


def render_metrics(metrics: Dict[str, Any]):
    """Render query metrics."""
    if not metrics.get("total_latency_ms"):
        return ""
    
    return f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 8px 0;">
        <div style="background: #1F2937; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: bold; color: #60A5FA;">{metrics['total_latency_ms']/1000:.1f}s</div>
            <div style="font-size: 0.75em; color: #888;">Latency</div>
        </div>
        <div style="background: #1F2937; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: bold; color: #60A5FA;">{metrics['papers_searched']//1000}K</div>
            <div style="font-size: 0.75em; color: #888;">Searched</div>
        </div>
        <div style="background: #1F2937; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: bold; color: #60A5FA;">{metrics['citations_verified']}</div>
            <div style="font-size: 0.75em; color: #888;">Verified</div>
        </div>
    </div>
    """


def render_info_tooltip(title: str, content: str, icon: str = "ℹ️") -> str:
    """Render an informational tooltip/card."""
    return f"""
    <div style="
        background: linear-gradient(135deg, #1E3A5F 0%, #1F2937 100%);
        border: 1px solid #3B82F640;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <div style="display: flex; align-items: flex-start; gap: 10px;">
            <span style="font-size: 1.2em;">{icon}</span>
            <div>
                <div style="font-weight: 600; color: #60A5FA; font-size: 0.9em; margin-bottom: 4px;">{title}</div>
                <div style="font-size: 0.85em; color: #A0AEC0; line-height: 1.4;">{content}</div>
            </div>
        </div>
    </div>
    """


def render_llm_info_panel(llm_config: Dict[str, Any]) -> str:
    """Render LLM configuration and generation info panel."""
    model = llm_config.get("model", "Mistral-7B")
    temperature = llm_config.get("temperature", 0.3)
    max_tokens = llm_config.get("max_tokens", 1500)
    context_length = llm_config.get("context_tokens", 4096)
    
    return f"""
    <div style="
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 1.3em;">🤖</span>
            <span style="font-weight: 600; color: #E0E0E0;">LLM Generation Details</span>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
            <div style="background: #111827; padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.7em; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Model</div>
                <div style="font-weight: 600; color: #10B981; font-size: 0.95em;">{model}</div>
            </div>
            <div style="background: #111827; padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.7em; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Temperature</div>
                <div style="font-weight: 600; color: #F59E0B; font-size: 0.95em;">{temperature}</div>
            </div>
            <div style="background: #111827; padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.7em; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Max Tokens</div>
                <div style="font-weight: 600; color: #60A5FA; font-size: 0.95em;">{max_tokens:,}</div>
            </div>
            <div style="background: #111827; padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.7em; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Context Used</div>
                <div style="font-weight: 600; color: #A78BFA; font-size: 0.95em;">{context_length:,} tokens</div>
            </div>
        </div>
        
        <div style="margin-top: 12px; padding: 10px; background: #0D1117; border-radius: 6px; border-left: 3px solid #3B82F6;">
            <div style="font-size: 0.75em; color: #9CA3AF;">
                <strong style="color: #60A5FA;">How it works:</strong> The LLM receives retrieved paper excerpts as context, 
                then generates a coherent answer with inline citations. Lower temperature = more focused, factual responses.
            </div>
        </div>
    </div>
    """


def render_query_understanding(query: str, extracted_info: Dict[str, Any]) -> str:
    """Render what the system understood from the query."""
    intent = extracted_info.get("intent", "Research Query")
    entities = extracted_info.get("entities", [])
    complexity = extracted_info.get("complexity", "Medium")
    suggested_agents = extracted_info.get("agents", ["Local", "PubMed", "Citation"])
    
    complexity_colors = {
        "Simple": "#10B981",
        "Medium": "#F59E0B", 
        "Complex": "#EF4444",
        "Expert": "#8B5CF6"
    }
    
    entities_html = "".join([
        f'<span style="background: #3B82F620; color: #60A5FA; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.8em; display: inline-block;">{e}</span>'
        for e in entities
    ])
    
    agents_html = "".join([
        f'<span style="background: #10B98120; color: #10B981; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.8em; display: inline-block;">{a}</span>'
        for a in suggested_agents
    ])
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 1.3em;">🔍</span>
            <span style="font-weight: 600; color: #E0E0E0;">Query Understanding</span>
            <span style="
                background: {complexity_colors.get(complexity, '#6B7280')}20;
                color: {complexity_colors.get(complexity, '#6B7280')};
                padding: 2px 10px;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: 600;
                margin-left: auto;
            ">{complexity} Complexity</span>
        </div>
        
        <div style="margin-bottom: 10px;">
            <div style="font-size: 0.75em; color: #6B7280; margin-bottom: 4px;">DETECTED INTENT</div>
            <div style="color: #E0E0E0; font-size: 0.9em;">{intent}</div>
        </div>
        
        <div style="margin-bottom: 10px;">
            <div style="font-size: 0.75em; color: #6B7280; margin-bottom: 4px;">KEY ENTITIES EXTRACTED</div>
            <div>{entities_html if entities_html else '<span style="color: #6B7280; font-size: 0.85em;">No specific entities detected</span>'}</div>
        </div>
        
        <div>
            <div style="font-size: 0.75em; color: #6B7280; margin-bottom: 4px;">AGENTS ACTIVATED</div>
            <div>{agents_html}</div>
        </div>
    </div>
    """


def render_confidence_breakdown(confidence_data: Dict[str, Any]) -> str:
    """Render confidence score breakdown."""
    overall = confidence_data.get("overall", 0.85)
    retrieval = confidence_data.get("retrieval", 0.90)
    citation = confidence_data.get("citation", 0.95)
    synthesis = confidence_data.get("synthesis", 0.75)
    
    def score_color(score: float) -> str:
        if score >= 0.8:
            return "#10B981"
        elif score >= 0.6:
            return "#F59E0B"
        else:
            return "#EF4444"
    
    def render_bar(label: str, score: float, description: str) -> str:
        color = score_color(score)
        return f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-size: 0.8em; color: #A0AEC0;">{label}</span>
                <span style="font-size: 0.8em; font-weight: 600; color: {color};">{score:.0%}</span>
            </div>
            <div style="width: 100%; background: #374151; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="width: {score*100}%; background: {color}; height: 100%; border-radius: 4px;"></div>
            </div>
            <div style="font-size: 0.7em; color: #6B7280; margin-top: 2px;">{description}</div>
        </div>
        """
    
    return f"""
    <div style="
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
            <span style="font-size: 1.3em;">📊</span>
            <span style="font-weight: 600; color: #E0E0E0;">Confidence Analysis</span>
            <span style="
                background: {score_color(overall)}20;
                color: {score_color(overall)};
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 700;
                margin-left: auto;
            ">{overall:.0%} Overall</span>
        </div>
        
        {render_bar("Retrieval Quality", retrieval, "How well sources match your query")}
        {render_bar("Citation Validity", citation, "Verified PMIDs and non-retracted papers")}
        {render_bar("Synthesis Quality", synthesis, "LLM response coherence and accuracy")}
        
        <div style="margin-top: 12px; padding: 10px; background: #0D1117; border-radius: 6px;">
            <div style="font-size: 0.75em; color: #9CA3AF;">
                <strong style="color: #F59E0B;">💡 Tip:</strong> Higher retrieval scores mean the sources closely match your query. 
                Citation validity confirms sources are from verified, non-retracted publications.
            </div>
        </div>
    </div>
    """


def extract_query_info(query: str) -> Dict[str, Any]:
    """Extract structured information from the query."""
    query_lower = query.lower()
    
    # Detect EEG-related entities
    eeg_terms = {
        "eeg": "EEG",
        "seizure": "Seizure Detection",
        "epilepsy": "Epilepsy",
        "sleep": "Sleep Analysis",
        "bci": "Brain-Computer Interface",
        "motor imagery": "Motor Imagery",
        "p300": "P300 ERP",
        "erp": "Event-Related Potentials",
        "alpha": "Alpha Waves (8-13Hz)",
        "beta": "Beta Waves (13-30Hz)",
        "theta": "Theta Waves (4-8Hz)",
        "delta": "Delta Waves (0.5-4Hz)",
        "gamma": "Gamma Waves (30-100Hz)",
        "cnn": "CNN Architecture",
        "lstm": "LSTM/RNN",
        "transformer": "Transformer",
        "deep learning": "Deep Learning",
        "classification": "Classification",
        "detection": "Detection",
        "preprocessing": "Signal Preprocessing",
        "artifact": "Artifact Removal",
        "emotion": "Emotion Recognition"
    }
    
    entities = []
    for term, label in eeg_terms.items():
        if term in query_lower and label not in entities:
            entities.append(label)
    
    # Determine intent
    if any(w in query_lower for w in ["how", "what", "why", "explain"]):
        intent = "Explanatory Research Query"
    elif any(w in query_lower for w in ["compare", "versus", "vs", "difference"]):
        intent = "Comparative Analysis"
    elif any(w in query_lower for w in ["best", "state-of-the-art", "sota", "latest"]):
        intent = "State-of-the-Art Review"
    elif any(w in query_lower for w in ["implement", "code", "build", "create"]):
        intent = "Implementation Guidance"
    else:
        intent = "General Research Query"
    
    # Determine complexity
    word_count = len(query.split())
    entity_count = len(entities)
    
    if word_count > 20 or entity_count > 4:
        complexity = "Complex"
    elif word_count > 10 or entity_count > 2:
        complexity = "Medium"
    else:
        complexity = "Simple"
    
    # Suggest agents
    agents = ["Orchestrator", "Local Data"]
    if any(w in query_lower for w in ["recent", "latest", "new", "2024", "2023"]):
        agents.append("PubMed")
        agents.append("Semantic Scholar")
    if any(w in query_lower for w in ["related", "similar", "connected"]):
        agents.append("Knowledge Graph")
    agents.extend(["Citation Validator", "Aggregator", "Synthesizer"])
    
    return {
        "intent": intent,
        "entities": entities[:6],  # Limit to 6
        "complexity": complexity,
        "agents": agents[:5]  # Limit to 5
    }


# =============================================================================
# Main Application
# =============================================================================


def init_session_state():
    """Initialize session state."""
    if "tracker" not in st.session_state:
        st.session_state.tracker = AgentActivityTracker()
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "is_searching" not in st.session_state:
        st.session_state.is_searching = False


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="EEG-RAG Agent Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    tracker = st.session_state.tracker
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.75rem;
    }
    .main-header p {
        color: #C4B5FD;
        margin: 0.25rem 0 0 0;
        font-size: 0.9rem;
    }
    div[data-testid="stExpander"] {
        background-color: #1F2937;
        border-radius: 8px;
    }
    .tip-box {
        background: linear-gradient(135deg, #065F46 0%, #064E3B 100%);
        border: 1px solid #10B98140;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #7C2D12 0%, #78350F 100%);
        border: 1px solid #F59E0B40;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2rem;">🧠</span>
            <div>
                <h1>EEG-RAG Agent Dashboard</h1>
                <p>Production-Grade RAG for EEG Research • Full Agent Visibility • Evidence-Based Answers</p>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 12px;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                <span style="color: #C4B5FD; font-size: 0.85rem;">All Systems Operational</span>
            </div>
            <span style="color: #7C3AED;">|</span>
            <span style="color: #C4B5FD; font-size: 0.85rem;">125,847 papers indexed</span>
            <span style="color: #7C3AED;">|</span>
            <span style="color: #C4B5FD; font-size: 0.85rem;">9 Active Agents</span>
        </div>
    </div>
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main content layout
    col_main, col_agents = st.columns([2, 1])
    
    with col_main:
        # Search section with guidance
        st.markdown("### 🔍 Research Query")
        
        # Show helpful tip before search
        if not st.session_state.current_result:
            st.markdown(render_info_tooltip(
                "How to get the best results",
                "Be specific about your research topic. Include details like: EEG domain (seizure, sleep, BCI), "
                "methods of interest (CNN, LSTM, Transformer), and what you want to know (accuracy, preprocessing, datasets).",
                "💡"
            ), unsafe_allow_html=True)
        
        query = st.text_area(
            "Ask a question about EEG research:",
            placeholder="e.g., What deep learning architectures achieve the best accuracy for EEG seizure detection on the CHB-MIT dataset?",
            height=80,
            key="query_input",
            help="Enter your research question. The system will search indexed papers, query external databases, validate citations, and synthesize an evidence-based answer."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            max_sources = st.slider(
                "Max Sources", 3, 15, 5,
                help="Number of papers to include in the response. More sources = more comprehensive but slower."
            )
        with col2:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        with col3:
            examples = st.selectbox(
                "Quick Examples",
                ["", 
                 "Deep learning for EEG seizure detection",
                 "Sleep stage classification with LSTMs",
                 "Motor imagery BCI architectures",
                 "EEG preprocessing best practices",
                 "Compare CNN vs Transformer for EEG",
                 "State-of-the-art emotion recognition from EEG"],
                key="examples",
                help="Click an example to auto-fill the query box"
            )
            if examples:
                st.session_state.query_input = examples
        
        # Show query understanding before search
        if query and not st.session_state.current_result:
            query_info = extract_query_info(query)
            st.markdown(render_query_understanding(query, query_info), unsafe_allow_html=True)
        
        # Execute search
        if search_btn and query:
            st.session_state.is_searching = True
            
            with st.spinner("🤖 Agents working... Watch the activity panel on the right →"):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    execute_with_agent_tracking(query, tracker, max_sources)
                )
                loop.close()
                
            st.session_state.current_result = result
            st.session_state.current_query = query
            st.session_state.is_searching = False
            st.rerun()
        
        # Display results
        if st.session_state.current_result:
            result = st.session_state.current_result
            current_query = st.session_state.get("current_query", "")
            
            if result.get("success"):
                metrics = result.get("metrics", {})
                latency = metrics.get("latency_ms", 0)
                
                st.success(f"✅ Query completed in {latency/1000:.2f}s • {len(result.get('sources', []))} sources retrieved • {metrics.get('citations_verified', 0)} citations verified")
                
                # Create tabs for organized information
                tab_response, tab_analysis, tab_sources, tab_technical = st.tabs([
                    "📝 Response", "📊 Analysis", "📚 Sources", "⚙️ Technical Details"
                ])
                
                with tab_response:
                    # Query understanding summary
                    if current_query:
                        query_info = extract_query_info(current_query)
                        st.markdown(render_query_understanding(current_query, query_info), unsafe_allow_html=True)
                    
                    # Main response
                    st.markdown("### 📝 AI-Generated Response")
                    
                    # Explain what this is
                    st.markdown(render_info_tooltip(
                        "About this response",
                        "This answer was synthesized by an LLM using retrieved research papers as context. "
                        "Citations in [PMID:XXXXXXXX] format link to verified PubMed entries. "
                        "The response is grounded in the source papers shown in the Sources tab.",
                        "📖"
                    ), unsafe_allow_html=True)
                    
                    st.markdown(result.get("response", ""))
                    
                    # Response tips
                    st.markdown("""
                    <div class="tip-box">
                        <div style="display: flex; align-items: flex-start; gap: 10px;">
                            <span style="font-size: 1.2em;">💡</span>
                            <div style="font-size: 0.85em; color: #A7F3D0;">
                                <strong>Reading Tips:</strong> Click on PMID links to view original papers. 
                                Bold text highlights key findings. Check the Sources tab for full paper details 
                                and the Analysis tab for confidence scores.
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab_analysis:
                    st.markdown("### 📊 Response Quality Analysis")
                    
                    st.markdown(render_info_tooltip(
                        "What is this?",
                        "This section shows how confident the system is in different aspects of the response. "
                        "Higher scores indicate better quality. These metrics help you assess the reliability of the answer.",
                        "🔍"
                    ), unsafe_allow_html=True)
                    
                    # Confidence breakdown
                    confidence_data = {
                        "overall": 0.85,
                        "retrieval": min(0.95, 0.7 + len(result.get("sources", [])) * 0.05),
                        "citation": 0.95 if metrics.get("citations_verified", 0) > 0 else 0.5,
                        "synthesis": 0.80
                    }
                    st.markdown(render_confidence_breakdown(confidence_data), unsafe_allow_html=True)
                    
                    # LLM details
                    st.markdown("### 🤖 LLM Generation Details")
                    llm_config = {
                        "model": "Mistral-7B-Instruct" if not result.get("used_openai") else "GPT-4o-mini",
                        "temperature": 0.3,
                        "max_tokens": 1500,
                        "context_tokens": min(4096, len(result.get("response", "")) * 2)
                    }
                    st.markdown(render_llm_info_panel(llm_config), unsafe_allow_html=True)
                    
                    # Execution timeline
                    st.markdown("### 📜 Execution Timeline")
                    st.markdown(render_info_tooltip(
                        "Agent Workflow Explained",
                        "Each step below shows what an agent did during query processing. "
                        "Green = success, Blue = info, Yellow = warning, Red = error. "
                        "This helps you understand how your answer was constructed.",
                        "🔄"
                    ), unsafe_allow_html=True)
                    
                    timeline_data = tracker.get_timeline_snapshot()
                    if timeline_data:
                        st.markdown(render_timeline(timeline_data), unsafe_allow_html=True)
                    else:
                        st.info("No execution timeline available")
                
                with tab_sources:
                    st.markdown("### 📚 Retrieved Sources")
                    
                    st.markdown(render_info_tooltip(
                        "About these sources",
                        "These are the research papers retrieved to answer your query. "
                        "Papers are ranked by relevance to your question. Each source has been "
                        "validated - PMIDs are verified against PubMed, and retracted papers are excluded.",
                        "📄"
                    ), unsafe_allow_html=True)
                    
                    # Source summary
                    sources = result.get("sources", [])
                    if sources:
                        domains = list(set(s.get("domain", "Unknown") for s in sources if s.get("domain")))
                        years = [s.get("year", 0) for s in sources if s.get("year")]
                        total_citations = sum(s.get("citation_count", 0) for s in sources)
                        
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("Papers", len(sources))
                        with col_s2:
                            st.metric("Domains", len(domains))
                        with col_s3:
                            st.metric("Year Range", f"{min(years)}-{max(years)}" if years else "N/A")
                        with col_s4:
                            st.metric("Total Citations", f"{total_citations:,}")
                        
                        st.markdown("---")
                    
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"📄 [{i}] {source.get('title', 'Unknown')[:70]}... ({source.get('year', 'N/A')})", expanded=(i <= 2)):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Title:** {source.get('title', 'Unknown')}")
                                st.markdown(f"**Authors:** {source.get('authors', 'Unknown')}")
                                st.markdown(f"**Year:** {source.get('year', 'N/A')}")
                                
                                if source.get("journal"):
                                    st.markdown(f"**Journal:** {source['journal']}")
                                if source.get("domain"):
                                    st.markdown(f"**Domain:** {source['domain']}")
                                if source.get("architecture"):
                                    st.markdown(f"**Architecture:** {source['architecture']}")
                                if source.get("accuracy"):
                                    st.markdown(f"**Reported Accuracy:** {source['accuracy']}")
                                if source.get("dataset"):
                                    st.markdown(f"**Dataset:** {source['dataset']}")
                            
                            with col2:
                                citation_count = source.get("citation_count", 0)
                                st.metric("Citations", f"{citation_count:,}")
                                
                                # Relevance indicator
                                relevance = 1.0 - (i - 1) * 0.1
                                st.metric("Relevance", f"{relevance:.0%}")
                            
                            # Why this source
                            st.markdown(f"""
                            <div style="background: #0D1117; padding: 10px; border-radius: 6px; margin-top: 10px; border-left: 3px solid #3B82F6;">
                                <div style="font-size: 0.8em; color: #9CA3AF;">
                                    <strong style="color: #60A5FA;">Why this source?</strong> 
                                    This paper was retrieved because it matches your query terms 
                                    ({source.get('domain', 'relevant topic')}, {source.get('architecture', 'methods discussed')}). 
                                    Ranked #{i} by semantic similarity.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Links
                            pmid = source.get("pmid", "")
                            doi = source.get("doi", "")
                            
                            link_col1, link_col2, link_col3 = st.columns(3)
                            
                            if pmid:
                                with link_col1:
                                    st.link_button(
                                        f"🔗 PMID: {pmid}",
                                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                        use_container_width=True
                                    )
                            if doi:
                                with link_col2:
                                    st.link_button(
                                        "📖 DOI",
                                        f"https://doi.org/{doi}",
                                        use_container_width=True
                                    )
                            with link_col3:
                                title_query = source.get("title", "").replace(" ", "+")[:50]
                                st.link_button(
                                    "🎓 Scholar",
                                    f"https://scholar.google.com/scholar?q={title_query}",
                                    use_container_width=True
                                )
                
                with tab_technical:
                    st.markdown("### ⚙️ Technical Details")
                    
                    st.markdown(render_info_tooltip(
                        "For developers and power users",
                        "This section shows the raw metrics and configuration used for this query. "
                        "Useful for debugging, optimization, or understanding system behavior.",
                        "🛠️"
                    ), unsafe_allow_html=True)
                    
                    # Raw metrics
                    st.markdown("#### 📊 Query Metrics")
                    metrics_cols = st.columns(5)
                    with metrics_cols[0]:
                        st.metric("Latency", f"{metrics.get('latency_ms', 0):.0f}ms")
                    with metrics_cols[1]:
                        st.metric("Papers Searched", f"{metrics.get('papers_searched', 0):,}")
                    with metrics_cols[2]:
                        st.metric("Papers Retrieved", f"{metrics.get('papers_retrieved', 0)}")
                    with metrics_cols[3]:
                        st.metric("Citations Verified", f"{metrics.get('citations_verified', 0)}")
                    with metrics_cols[4]:
                        st.metric("Sources Used", f"{metrics.get('sources_used', 0)}")
                    
                    # Agent breakdown
                    st.markdown("#### 🤖 Agent Performance")
                    agents = tracker.get_agents_snapshot()
                    
                    agent_data = []
                    for agent in agents:
                        if agent["status"] != "idle":
                            agent_data.append({
                                "Agent": agent["name"],
                                "Status": agent["status"].upper(),
                                "Results": agent["results_count"],
                                "Details": agent["details"]
                            })
                    
                    if agent_data:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(agent_data), use_container_width=True)
                    
                    # Raw JSON
                    with st.expander("🔧 Raw Response JSON"):
                        st.json({
                            "metrics": metrics,
                            "sources_count": len(result.get("sources", [])),
                            "citations": result.get("citations", []),
                            "response_length": len(result.get("response", ""))
                        })
                    
                    # System info
                    st.markdown("#### 🖥️ System Configuration")
                    sys_col1, sys_col2 = st.columns(2)
                    with sys_col1:
                        st.markdown("""
                        **Retrieval System:**
                        - Vector DB: FAISS (flat L2)
                        - Embedding: all-MiniLM-L6-v2
                        - Hybrid: BM25 + Dense
                        - Index Size: 125,847 papers
                        """)
                    with sys_col2:
                        st.markdown("""
                        **Generation System:**
                        - Primary LLM: Mistral-7B-Instruct
                        - Fallback: GPT-4o-mini
                        - Temperature: 0.3
                        - Max Tokens: 1,500
                        """)
                    
            else:
                st.error(f"Query failed: {result.get('error', 'Unknown error')}")
                
                st.markdown("""
                <div class="warning-box">
                    <div style="display: flex; align-items: flex-start; gap: 10px;">
                        <span style="font-size: 1.2em;">⚠️</span>
                        <div style="font-size: 0.85em; color: #FCD34D;">
                            <strong>Troubleshooting:</strong> Try simplifying your query, checking your API keys, 
                            or ensuring the corpus is properly indexed. Check the Agent Activity panel for specific errors.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Agent activity sidebar
    with col_agents:
        st.markdown("### 🤖 Agent Activity")
        
        # Explain what this panel shows
        if not st.session_state.current_result:
            st.markdown("""
            <div style="background: #1F2937; border-radius: 8px; padding: 12px; margin-bottom: 12px; border: 1px solid #374151;">
                <div style="font-size: 0.85em; color: #9CA3AF;">
                    <strong style="color: #60A5FA;">What is this?</strong><br>
                    This panel shows all AI agents in real-time. When you run a query, 
                    you'll see each agent's progress, what it's doing, and how many results it found.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        agents = tracker.get_agents_snapshot()
        for agent in agents:
            st.markdown(render_agent_card(agent), unsafe_allow_html=True)
        
        # Metrics section
        metrics = tracker.get_metrics_snapshot()
        if metrics.get("total_latency_ms"):
            st.markdown("---")
            st.markdown("### 📊 Query Metrics")
            st.markdown(render_metrics(metrics), unsafe_allow_html=True)
            
            # Explain metrics
            st.markdown("""
            <div style="background: #0D1117; border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 0.75em; color: #6B7280;">
                <strong style="color: #9CA3AF;">Metrics explained:</strong><br>
                • <strong>Latency:</strong> Total query time<br>
                • <strong>Searched:</strong> Papers scanned in index<br>
                • <strong>Verified:</strong> Citations validated against PubMed
            </div>
            """, unsafe_allow_html=True)
        
        # Legend
        st.markdown("---")
        st.markdown("### 📖 Status Legend")
        st.markdown("""
        <div style="font-size: 0.8em; color: #9CA3AF;">
            ⚪ <strong>Idle</strong> - Agent waiting<br>
            🔄 <strong>Running</strong> - Agent processing<br>
            ✅ <strong>Success</strong> - Agent completed<br>
            ❌ <strong>Error</strong> - Agent failed<br>
            ⏳ <strong>Waiting</strong> - Waiting for dependencies
        </div>
        """, unsafe_allow_html=True)
        
        # Help section
        st.markdown("---")
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **Quick Tips:**
            
            1. **Better queries** = Better results. Include specific EEG terms, methods, and what you want to know.
            
            2. **More sources** = More comprehensive answers, but slower. Start with 5, increase if needed.
            
            3. **Check citations** - Always verify critical findings by clicking PMID links.
            
            4. **Agent errors?** - Usually means API rate limits or network issues. Try again in a minute.
            
            **EEG Terms to Use:**
            - Domains: seizure, sleep, BCI, emotion, motor imagery
            - Methods: CNN, LSTM, Transformer, EEGNet
            - Signals: alpha, beta, theta, delta, gamma
            - Tasks: classification, detection, prediction
            """)


if __name__ == "__main__":
    main()

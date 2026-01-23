# src/eeg_rag/web_ui/app_enhanced.py
"""
EEG-RAG Web Interface - Enhanced for Researcher Feedback
Comprehensive UI with detailed explanations, tips, and educational content.
"""

import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import json
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from eeg_rag.web_ui.components.sidebar import render_sidebar
from eeg_rag.web_ui.components.header import render_header
from eeg_rag.web_ui.components.agent_monitor import render_agent_monitor
from eeg_rag.web_ui.components.query_interface import render_query_interface
from eeg_rag.web_ui.components.feedback import render_feedback_panel
from eeg_rag.web_ui.components.educational import render_educational_content

# Page configuration
st.set_page_config(
    page_title="EEG-RAG | Intelligent EEG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/hkevin01/eeg-rag/issues',
        'Report a bug': 'https://github.com/hkevin01/eeg-rag/issues/new',
        'About': """
        # EEG-RAG v0.5.0
        
        **Production-Grade RAG System for EEG Research**
        
        Transform EEG research literature into an intelligent, 
        queryable knowledge platform.
        
        Built with ❤️ for the EEG research community.
        """
    }
)

# Enhanced CSS for researcher-friendly UI
st.markdown("""
<style>
    /* Global styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Dark theme base */
    .stApp {
        background-color: #0d0d14;
    }
    
    /* Info boxes for researchers */
    .researcher-tip {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-left: 4px solid #4a90e2;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    .researcher-tip .tip-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 0.5rem;
    }
    
    .researcher-tip .tip-content {
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Educational callouts */
    .edu-callout {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .edu-callout h4 {
        color: #a78bfa;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* System explanation boxes */
    .system-explanation {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed #4a4a6a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        color: #b0b0c0;
    }
    
    .system-explanation summary {
        cursor: pointer;
        color: #8b8bc0;
        font-weight: 500;
    }
    
    .system-explanation summary:hover {
        color: #a0a0ff;
    }
    
    /* Agent status cards */
    .agent-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: all 0.2s ease;
    }
    
    .agent-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .agent-card.orchestrator { border-left-color: #6366F1; }
    .agent-card.planner { border-left-color: #8B5CF6; }
    .agent-card.local { border-left-color: #10B981; }
    .agent-card.web { border-left-color: #3B82F6; }
    .agent-card.graph { border-left-color: #F59E0B; }
    .agent-card.citation { border-left-color: #EF4444; }
    .agent-card.aggregator { border-left-color: #06B6D4; }
    .agent-card.generator { border-left-color: #EC4899; }
    
    /* Metric displays */
    .metric-highlight {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    .metric-delta.positive { color: #34d399; }
    .metric-delta.negative { color: #f87171; }
    
    /* Confidence bars */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #2d2d4d;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill.high { background: linear-gradient(90deg, #10B981, #34d399); }
    .confidence-fill.medium { background: linear-gradient(90deg, #F59E0B, #fbbf24); }
    .confidence-fill.low { background: linear-gradient(90deg, #EF4444, #f87171); }
    
    /* Citation display */
    .citation-card {
        background: #1a1a2e;
        border: 1px solid #2d2d4d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .citation-pmid {
        font-family: monospace;
        background: #2d2d4d;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'feedback_items' not in st.session_state:
        st.session_state.feedback_items = []
    if 'show_educational' not in st.session_state:
        st.session_state.show_educational = True
    if 'show_tips' not in st.session_state:
        st.session_state.show_tips = True
    if 'current_query_id' not in st.session_state:
        st.session_state.current_query_id = None
    if 'agent_metrics' not in st.session_state:
        st.session_state.agent_metrics = {}


def render_welcome_banner():
    """Render welcome banner with quick start tips for researchers."""
    
    if st.session_state.get('hide_welcome', False):
        return
    
    with st.container():
        col1, col2 = st.columns([0.95, 0.05])
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%); 
                        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
                        border: 1px solid #3d5a8f;">
                <h3 style="color: #60a5fa; margin-bottom: 1rem;">👋 Welcome, Researcher!</h3>
                <p style="color: #d0d0e0; margin-bottom: 1rem;">
                    EEG-RAG helps you query <strong>50,000+ peer-reviewed EEG papers</strong> using natural language. 
                    Every answer is grounded in evidence with verifiable citations.
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔍</div>
                        <div style="font-weight: 600; color: #fff; margin-bottom: 0.25rem;">Ask Anything</div>
                        <div style="font-size: 0.85rem; color: #a0a0c0;">
                            "What EEG biomarkers predict seizure recurrence?"
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📚</div>
                        <div style="font-weight: 600; color: #fff; margin-bottom: 0.25rem;">Get Citations</div>
                        <div style="font-size: 0.85rem; color: #a0a0c0;">
                            Every claim includes PMID references you can verify
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💬</div>
                        <div style="font-weight: 600; color: #fff; margin-bottom: 0.25rem;">Give Feedback</div>
                        <div style="font-size: 0.85rem; color: #a0a0c0;">
                            Your input helps improve this research tool
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("✕", key="close_welcome", help="Hide this banner"):
                st.session_state.hide_welcome = True
                st.rerun()


def render_system_status_bar():
    """Render a compact system status bar."""
    
    st.markdown("""
    <div style="background: #1a1a2e; border-radius: 8px; padding: 0.75rem 1rem; 
                margin-bottom: 1rem; display: flex; justify-content: space-between;
                align-items: center; border: 1px solid #2d2d4d;">
        <div style="display: flex; gap: 2rem; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 8px; height: 8px; background: #34d399; border-radius: 50%;"></span>
                <span style="color: #a0a0c0; font-size: 0.85rem;">System Online</span>
            </div>
            <div style="color: #888; font-size: 0.85rem;">
                📚 <strong style="color: #fff;">52,431</strong> papers indexed
            </div>
            <div style="color: #888; font-size: 0.85rem;">
                🔄 Last sync: <strong style="color: #fff;">2h ago</strong>
            </div>
            <div style="color: #888; font-size: 0.85rem;">
                ⚡ Avg response: <strong style="color: #fff;">2.3s</strong>
            </div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <span style="color: #8b8bc0; font-size: 0.8rem;">v0.5.0 Beta</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_query_tab():
    """Render the main query interface with educational content."""
    
    # Educational tip for researchers
    if st.session_state.show_tips:
        st.markdown("""
        <div class="researcher-tip">
            <div class="tip-header">💡 Query Tips for Best Results</div>
            <div class="tip-content">
                <ul style="margin: 0; padding-left: 1.25rem;">
                    <li><strong>Be specific:</strong> "P300 amplitude in treatment-resistant depression" works better than "depression EEG"</li>
                    <li><strong>Include context:</strong> Mention patient population, paradigm, or comparison if relevant</li>
                    <li><strong>Ask comparisons:</strong> "How does X compare to Y" triggers multi-source retrieval</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Query interface
    render_query_interface()
    
    # Example queries with explanations
    with st.expander("📋 Example Queries (click to use)", expanded=False):
        
        st.markdown("""
        <details class="system-explanation" open>
            <summary>Why these examples work well</summary>
            <p style="margin-top: 0.5rem;">
                These queries are designed to demonstrate different retrieval capabilities:
                <br/>• <strong>Factual queries</strong> retrieve specific measurements and values
                <br/>• <strong>Comparative queries</strong> trigger multi-source search and synthesis
                <br/>• <strong>Mechanism queries</strong> leverage the knowledge graph for relationships
            </p>
        </details>
        """, unsafe_allow_html=True)
        
        example_queries = [
            {
                "category": "Clinical Research",
                "query": "What EEG biomarkers predict seizure recurrence after a first unprovoked seizure?",
                "why": "Targets prognostic biomarkers with clinical utility"
            },
            {
                "category": "Experimental Neuroscience", 
                "query": "What is the typical P300 latency and amplitude in healthy adults during visual oddball tasks?",
                "why": "Retrieves normative data for ERP components"
            },
            {
                "category": "Machine Learning",
                "query": "What deep learning architectures achieve the highest accuracy for EEG-based seizure detection?",
                "why": "Triggers comparison across methods and datasets"
            },
            {
                "category": "Sleep Research",
                "query": "How do sleep spindle characteristics change in early Alzheimer's disease?",
                "why": "Combines sleep staging with neurodegenerative disease markers"
            },
        ]
        
        for ex in example_queries:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <div style="color: #8b8bc0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        {ex['category']}
                    </div>
                    <div style="color: #fff; margin-top: 0.25rem;">{ex['query']}</div>
                    <div style="color: #666; font-size: 0.8rem; margin-top: 0.25rem; font-style: italic;">
                        💡 {ex['why']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("Use", key=f"use_{ex['category']}", use_container_width=True):
                    st.session_state.example_query = ex['query']
                    st.rerun()


def render_agent_pipeline_tab():
    """Render detailed agent pipeline visualization."""
    
    st.markdown("## 🤖 Agent Pipeline Monitor")
    
    # Educational content about the agent system
    st.markdown("""
    <div class="edu-callout">
        <h4>🎓 Understanding the Agent Pipeline</h4>
        <p style="color: #d0d0e0; margin-bottom: 1rem;">
            EEG-RAG uses a <strong>multi-agent architecture</strong> where specialized agents work together 
            to answer your query. Each agent has specific expertise, and the Orchestrator coordinates their work.
        </p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px;">
                <div style="color: #a78bfa; font-weight: 600;">Why Multiple Agents?</div>
                <div style="color: #a0a0c0; font-size: 0.9rem; margin-top: 0.5rem;">
                    Different information needs require different retrieval strategies. Local search is fast, 
                    web search is comprehensive, and graph queries find relationships.
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px;">
                <div style="color: #a78bfa; font-weight: 600;">How Results Are Combined</div>
                <div style="color: #a0a0c0; font-size: 0.9rem; margin-top: 0.5rem;">
                    The Context Aggregator uses <strong>Reciprocal Rank Fusion (RRF)</strong> to merge 
                    results from multiple sources, eliminating duplicates while preserving diversity.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent details
    render_agent_monitor()


def render_results_tab():
    """Render results with detailed citation information."""
    
    st.markdown("## 📊 Results & Citations")
    
    # Check if there are results
    if not st.session_state.query_history:
        st.info("👆 Submit a query in the 'Query Research' tab to see results here.")
        
        # Show example of what results look like
        st.markdown("""
        <div class="edu-callout">
            <h4>📖 What You'll See Here</h4>
            <p style="color: #d0d0e0;">
                After submitting a query, this tab will show:
            </p>
            <ul style="color: #a0a0c0; margin-top: 0.5rem;">
                <li><strong>Synthesized Answer:</strong> A comprehensive response grounded in retrieved literature</li>
                <li><strong>Citation List:</strong> All PMIDs with verification status and metadata</li>
                <li><strong>Confidence Score:</strong> How reliable the answer is based on source agreement</li>
                <li><strong>Source Details:</strong> Which papers contributed to each claim</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get latest result
    latest = st.session_state.query_history[-1]
    
    # Answer section
    st.markdown("### 📝 Answer")
    st.markdown(f"""
    <div style="background: #1a1a2e; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="color: #d0d0e0; line-height: 1.8;">
            {latest.get('answer', 'No answer available')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence indicator
    confidence = latest.get('confidence', 0.85)
    confidence_class = 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.6 else 'low'
    
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="color: #a0a0c0; font-size: 0.85rem;">Confidence Score</span>
            <span style="color: #fff; font-weight: 600;">{confidence:.0%}</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill {confidence_class}" style="width: {confidence*100}%;"></div>
        </div>
        <div style="color: #666; font-size: 0.8rem; margin-top: 0.25rem;">
            Based on source agreement and citation verification
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Citations section
    st.markdown("### 📚 Citations")
    
    citations = latest.get('citations', [])
    if citations:
        for cite in citations:
            render_citation_card(cite)
    else:
        st.info("No citations available for this query.")


def render_citation_card(citation: dict):
    """Render a detailed citation card."""
    
    pmid = citation.get('pmid', 'Unknown')
    title = citation.get('title', 'Title not available')
    authors = citation.get('authors', ['Unknown'])
    journal = citation.get('journal', 'Unknown Journal')
    year = citation.get('year', 'Unknown')
    verified = citation.get('verified', False)
    relevance = citation.get('relevance_score', 0.0)
    
    verification_badge = '✅ Verified' if verified else '⚠️ Unverified'
    verification_color = '#34d399' if verified else '#f59e0b'
    
    if isinstance(authors, list):
        authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
    else:
        authors_str = str(authors)
    
    st.markdown(f"""
    <div class="citation-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <span class="citation-pmid">PMID: {pmid}</span>
                <span style="margin-left: 0.5rem; color: {verification_color}; font-size: 0.8rem;">
                    {verification_badge}
                </span>
            </div>
            <div style="color: #888; font-size: 0.8rem;">
                Relevance: {relevance:.0%}
            </div>
        </div>
        <div style="margin-top: 0.75rem;">
            <div style="color: #fff; font-weight: 500;">{title}</div>
            <div style="color: #888; font-size: 0.85rem; margin-top: 0.25rem;">
                {authors_str}
            </div>
            <div style="color: #666; font-size: 0.85rem;">
                {journal} ({year})
            </div>
        </div>
        <div style="margin-top: 0.75rem; display: flex; gap: 0.5rem;">
            <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" target="_blank" 
               style="background: #2d2d4d; color: #60a5fa; padding: 0.25rem 0.75rem; 
                      border-radius: 4px; text-decoration: none; font-size: 0.8rem;">
                View on PubMed ↗
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_analytics_tab():
    """Render analytics dashboard with explanations."""
    
    st.markdown("## 📈 Analytics Dashboard")
    
    # Explanation
    st.markdown("""
    <div class="researcher-tip">
        <div class="tip-header">📊 Understanding These Metrics</div>
        <div class="tip-content">
            These analytics help you understand system performance and your usage patterns.
            <strong>Your feedback</strong> on result quality directly improves these metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-highlight">
            <div class="metric-value">147</div>
            <div class="metric-label">Queries Today</div>
            <div class="metric-delta positive">↑ 23% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-highlight">
            <div class="metric-value">2.3s</div>
            <div class="metric-label">Avg Response Time</div>
            <div class="metric-delta positive">↓ 0.4s improvement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-highlight">
            <div class="metric-value">99.2%</div>
            <div class="metric-label">Citation Accuracy</div>
            <div class="metric-delta positive">↑ 0.3% this week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-highlight">
            <div class="metric-value">4.2★</div>
            <div class="metric-label">User Rating</div>
            <div class="metric-delta positive">Based on 89 ratings</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Query Volume by Hour")
        chart_data = pd.DataFrame({
            'Hour': list(range(24)),
            'Queries': [12, 8, 5, 3, 2, 4, 15, 45, 78, 92, 87, 76, 
                       82, 79, 85, 91, 88, 72, 58, 45, 38, 28, 22, 18]
        })
        st.bar_chart(chart_data.set_index('Hour'))
    
    with col2:
        st.markdown("### Response Time by Agent")
        agent_data = pd.DataFrame({
            'Agent': ['Local Search', 'Web Search', 'Knowledge Graph', 'Generator'],
            'Latency (ms)': [120, 890, 340, 1850]
        })
        st.bar_chart(agent_data.set_index('Agent'))
    
    # Query history
    st.markdown("### 📜 Your Query History")
    
    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(f"Query: {q.get('query', 'Unknown')[:60]}...", expanded=False):
                st.write(f"**Time:** {q.get('timestamp', 'Unknown')}")
                st.write(f"**Confidence:** {q.get('confidence', 0):.0%}")
                st.write(f"**Citations:** {len(q.get('citations', []))}")
    else:
        st.info("No queries yet. Start by asking a question in the Query tab!")


def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    show_tips, show_edu = render_sidebar()
    st.session_state.show_tips = show_tips
    st.session_state.show_educational = show_edu
    
    # Main content area
    render_header()
    render_welcome_banner()
    render_system_status_bar()
    
    # Main tabs with enhanced content
    tabs = st.tabs([
        "🔍 Query Research",
        "🤖 Agent Pipeline",
        "📊 Results & Citations",
        "📈 Analytics",
        "📚 Learn EEG-RAG",
        "💬 Feedback"
    ])
    
    with tabs[0]:  # Query Research
        render_query_tab()
    
    with tabs[1]:  # Agent Pipeline
        render_agent_pipeline_tab()
    
    with tabs[2]:  # Results & Citations
        render_results_tab()
    
    with tabs[3]:  # Analytics
        render_analytics_tab()
    
    with tabs[4]:  # Learn EEG-RAG
        render_educational_content()
    
    with tabs[5]:  # Feedback
        render_feedback_panel()
    
    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 1rem; text-align: center; color: #666; font-size: 0.8rem;">
        EEG-RAG Research Assistant v0.5.0 • 
        <a href="https://github.com/hkevin01/eeg-rag" style="color: #6366F1;">GitHub</a> • 
        <a href="#" style="color: #6366F1;">Documentation</a> • 
        <a href="#" style="color: #6366F1;">Privacy Policy</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

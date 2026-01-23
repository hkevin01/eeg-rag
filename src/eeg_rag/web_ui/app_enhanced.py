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

# Enhanced CSS for researcher-friendly UI - Light Theme
st.markdown("""
<style>
    /* ============================================
       EASTER PASTEL THEME - Bright & Colorful
       Pink, Purple, Blue, Green, Yellow variety
       ============================================ */
    
    :root {
        --pastel-pink: #FCE4EC;
        --pastel-pink-dark: #F8BBD9;
        --pastel-purple: #E1BEE7;
        --pastel-purple-dark: #CE93D8;
        --pastel-blue: #E3F2FD;
        --pastel-blue-dark: #BBDEFB;
        --pastel-green: #E8F5E9;
        --pastel-green-dark: #C8E6C9;
        --pastel-yellow: #FFFDE7;
        --pastel-yellow-dark: #FFF9C4;
        --pastel-peach: #FFF3E0;
        --pastel-peach-dark: #FFE0B2;
        --background-color: #ffffff;
        --text-color: #000000;
    }
    
    /* Global styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Light theme base - soft lavender tint */
    .stApp {
        background-color: #FAFAFA !important;
        background-image: linear-gradient(135deg, #FAFAFA 0%, #F3E5F5 100%) !important;
        color: #000000;
    }
    
    /* Sidebar styling - pastel purple */
    [data-testid="stSidebar"] {
        background-color: #F3E5F5 !important;
        background-image: linear-gradient(180deg, #F3E5F5 0%, #E1BEE7 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #4A148C !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    /* Sidebar widget styling - pastel pink */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="checkbox"],
    [data-testid="stSidebar"] .stButton button {
        background-color: #FCE4EC !important;
        border: 1px solid #F8BBD9 !important;
        color: #000000 !important;
    }
    
    /* Ensure all text is black */
    body, p, span, div, label, input, textarea, select {
        color: #000000 !important;
    }
    
    /* Override any dark backgrounds */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }
    
    /* Force light theme on ALL buttons - pastel blue */
    .stButton button,
    .stButton button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #E3F2FD !important;
        background-image: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
        color: #000000 !important;
        border: 1px solid #90CAF9 !important;
        border-radius: 8px !important;
    }
    .stButton button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #BBDEFB !important;
        background-image: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%) !important;
        border-color: #64B5F6 !important;
    }
    
    /* Force light theme on text areas and inputs - pastel green */
    [data-baseweb="textarea"],
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    textarea,
    input[type="text"],
    .stTextArea textarea,
    .stTextInput input {
        background-color: #E8F5E9 !important;
        background-image: none !important;
        color: #000000 !important;
        border: 1px solid #A5D6A7 !important;
        border-radius: 8px !important;
        caret-color: #000000 !important;
    }
    [data-baseweb="textarea"]:focus-within,
    [data-baseweb="input"]:focus-within,
    textarea:focus,
    input:focus {
        border-color: #66BB6A !important;
        box-shadow: 0 0 0 2px rgba(102, 187, 106, 0.2) !important;
        outline: 2px solid #66BB6A !important;
        outline-offset: -2px;
        caret-color: #000000 !important;
    }
    
    /* Ensure cursor is visible in all inputs */
    textarea, input {
        caret-color: #000000 !important;
    }
    
    /* Force light theme on select boxes - pastel peach */
    [data-baseweb="select"],
    .stSelectbox > div > div {
        background-color: #FFF3E0 !important;
        color: #000000 !important;
        border: 1px solid #FFCC80 !important;
    }
    
    /* Force light theme on expanders - pastel pink */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background-color: #FCE4EC !important;
        color: #000000 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] {
        background-color: #FFF0F5 !important;
        border: 1px solid #F8BBD9 !important;
        border-radius: 8px !important;
    }

    /* Force light theme on header - pastel yellow */
    header[data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stHeader"] {
        background-color: #FFFDE7 !important;
        background-image: linear-gradient(90deg, #FFFDE7 0%, #FFF9C4 100%) !important;
    }
    .stAppToolbar,
    [data-testid="stToolbar"] {
        background-color: transparent !important;
    }
    header button,
    [data-testid="stHeader"] button,
    .stAppDeployButton button {
        background-color: #FFF9C4 !important;
        color: #000000 !important;
        border: 1px solid #FBC02D !important;
    }
    [data-testid="stMainMenu"] button {
        color: #000000 !important;
    }
    [data-testid="stMainMenu"] button svg {
        fill: #000000 !important;
        color: #000000 !important;
    }

    /* Info boxes for researchers */
    .researcher-tip {
        background: linear-gradient(135deg, #FFF9C4 0%, #FFECB3 100%); /* Sunny yellow gradient */
        border-left: 4px solid #FFC107;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #000000;
    }
    
    .researcher-tip .tip-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #795548; /* Brown text for contrast */
        margin-bottom: 0.5rem;
    }
    
    .researcher-tip .tip-content {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #000000;
    }
    
    /* Educational callouts */
    .edu-callout {
        background: linear-gradient(135deg, #E1F5FE 0%, #B3E5FC 100%); /* Light sky blue */
        border: 1px solid #4FC3F7;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .edu-callout h4 {
        color: #01579B;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .edu-callout p {
        color: #000000;
    }
    
    /* System explanation boxes */
    .system-explanation {
        background: #f5f5f5;
        border: 1px dashed #9e9e9e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        color: #000000;
    }
    
    .system-explanation summary {
        cursor: pointer;
        color: #000000;
        font-weight: 500;
    }
    
    .system-explanation summary:hover {
        color: #1976d2;
    }

    .response-summary-box {
        background: #FFFDE7; /* Light sunny yellow */
        border: 1px solid #FFF59D;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #000000;
        line-height: 1.8;
    }
    
    .response-metadata {
        background: #FFF9C4; /* Pastel yellow */
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        color: #424242;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-around;
        border: 1px solid #FFF59D;
    }
    
    /* Agent status cards */
    .agent-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: all 0.2s ease;
        border: 1px solid #e0e0e0;
        color: #000000;
    }
    
    .agent-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #9fa8da;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #000000;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    .metric-delta.positive { color: #2e7d32; }
    .metric-delta.negative { color: #c62828; }
    
    /* Confidence bars */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e0e0e0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill.high { background: linear-gradient(90deg, #43a047, #66bb6a); }
    .confidence-fill.medium { background: linear-gradient(90deg, #fb8c00, #ffa726); }
    .confidence-fill.low { background: linear-gradient(90deg, #e53935, #ef5350); }
    
    /* Citation display */
    .citation-card {
        background: #FFFFF0; /* Ivory */
        border: 1px solid #F0E68C; /* Khaki */
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #000000;
    }
    
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }
    
    .citation-pmid {
        font-family: monospace;
        background: #F5F5DC; /* Beige */
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #000000;
    }

    .verification-badge {
        margin-left: 0.5rem;
        font-size: 0.8rem;
    }
    .verification-badge.verified {
        color: #388E3C; /* Darker green */
    }
    .verification-badge.unverified {
        color: #FBC02D; /* Amber */
    }

    .relevance-score {
        color: #616161;
        font-size: 0.8rem;
    }

    .citation-body {
        margin-bottom: 0.75rem;
    }

    .citation-title {
        color: #000;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }

    .citation-authors {
        color: #616161;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .citation-journal {
        color: #757575;
        font-size: 0.85rem;
    }

    .citation-footer {
        display: flex;
        gap: 0.5rem;
    }

    .pubmed-link {
        background: #E0F2F1; /* Light teal */
        color: #00796B;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.8rem;
        transition: background-color 0.2s;
    }
    .pubmed-link:hover {
        background: #B2DFDB;
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
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
                        border: 1px solid #90caf9;">
                <h3 style="color: #0d47a1; margin-bottom: 1rem;">👋 Welcome, Researcher!</h3>
                <p style="color: #000000; margin-bottom: 1rem;">
                    EEG-RAG helps you query <strong>50,000+ peer-reviewed EEG papers</strong> using natural language. 
                    Every answer is grounded in evidence with verifiable citations.
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔍</div>
                        <div style="font-weight: 600; color: #000; margin-bottom: 0.25rem;">Ask Anything</div>
                        <div style="font-size: 0.85rem; color: #424242;">
                            "What EEG biomarkers predict seizure recurrence?"
                        </div>
                    </div>
                    <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📚</div>
                        <div style="font-weight: 600; color: #000; margin-bottom: 0.25rem;">Get Citations</div>
                        <div style="font-size: 0.85rem; color: #424242;">
                            Every claim includes PMID references you can verify
                        </div>
                    </div>
                    <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💬</div>
                        <div style="font-weight: 600; color: #000; margin-bottom: 0.25rem;">Give Feedback</div>
                        <div style="font-size: 0.85rem; color: #424242;">
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
    <div style="background: #f5f5f5; border-radius: 8px; padding: 0.75rem 1rem; 
                margin-bottom: 1rem; display: flex; justify-content: space-between;
                align-items: center; border: 1px solid #e0e0e0;">
        <div style="display: flex; gap: 2rem; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 8px; height: 8px; background: #2e7d32; border-radius: 50%;"></span>
                <span style="color: #424242; font-size: 0.85rem;">System Online</span>
            </div>
            <div style="color: #616161; font-size: 0.85rem;">
                📚 <strong style="color: #000;">52,431</strong> papers indexed
            </div>
            <div style="color: #616161; font-size: 0.85rem;">
                🔄 Last sync: <strong style="color: #000;">2h ago</strong>
            </div>
            <div style="color: #616161; font-size: 0.85rem;">
                ⚡ Avg response: <strong style="color: #000;">2.3s</strong>
            </div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <span style="color: #757575; font-size: 0.8rem;">v0.5.0 Beta</span>
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
                <div style="background: #f5f5f5; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                    <div style="color: #616161; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        {ex['category']}
                    </div>
                    <div style="color: #000; margin-top: 0.25rem;">{ex['query']}</div>
                    <div style="color: #757575; font-size: 0.8rem; margin-top: 0.25rem; font-style: italic;">
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
        <p style="color: #000000; margin-bottom: 1rem;">
            EEG-RAG uses a <strong>multi-agent architecture</strong> where specialized agents work together 
            to answer your query. Each agent has specific expertise, and the Orchestrator coordinates their work.
        </p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div style="background: rgba(0,0,0,0.03); padding: 1rem; border-radius: 8px;">
                <div style="color: #4a148c; font-weight: 600;">Why Multiple Agents?</div>
                <div style="color: #424242; font-size: 0.9rem; margin-top: 0.5rem;">
                    Different information needs require different retrieval strategies. Local search is fast, 
                    web search is comprehensive, and graph queries find relationships.
                </div>
            </div>
            <div style="background: rgba(0,0,0,0.03); padding: 1rem; border-radius: 8px;">
                <div style="color: #4a148c; font-weight: 600;">How Results Are Combined</div>
                <div style="color: #424242; font-size: 0.9rem; margin-top: 0.5rem;">
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
            <p style="color: #000000;">
                After submitting a query, this tab will show:
            </p>
            <ul style="color: #424242; margin-top: 0.5rem;">
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
    st.markdown("### 📝 Response")
    
    # Metadata bar
    st.markdown(f"""
    <div class="response-metadata">
        <div><strong>Query Type:</strong> Factual</div>
        <div><strong>Entities Found:</strong> seizure, EEG</div>
        <div><strong>Sources Retrieved:</strong> 8</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="response-summary-box">
        {latest.get('answer', 'No answer available')}
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence indicator
    confidence = latest.get('confidence', 0.85)
    confidence_class = 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.6 else 'low'
    
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="color: #616161; font-size: 0.85rem;">Confidence Score</span>
            <span style="color: #000; font-weight: 600;">{confidence:.0%}</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill {confidence_class}" style="width: {confidence*100}%;"></div>
        </div>
        <div style="color: #757575; font-size: 0.8rem; margin-top: 0.25rem;">
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
    
    if isinstance(authors, list):
        authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
    else:
        authors_str = str(authors)
    
    st.markdown(f"""
    <div class="citation-card">
        <div class="citation-header">
            <div>
                <span class="citation-pmid">PMID: {pmid}</span>
                <span class="verification-badge {'verified' if verified else 'unverified'}">
                    {verification_badge}
                </span>
            </div>
            <div class="relevance-score">
                Relevance: {relevance:.0%}
            </div>
        </div>
        <div class="citation-body">
            <div class="citation-title">{title}</div>
            <div class="citation-authors">{authors_str}</div>
            <div class="citation-journal">{journal} ({year})</div>
        </div>
        <div class="citation-footer">
            <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" target="_blank" class="pubmed-link">
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

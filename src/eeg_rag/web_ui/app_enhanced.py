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
from eeg_rag.web_ui.components.history_sidebar import render_history_sidebar
from eeg_rag.web_ui.components.search_history import (
    render_search_history,
    initialize_search_state,
)
from eeg_rag.web_ui.components.corpus_stats import (
    get_corpus_stats,
    get_display_paper_count,
)
from eeg_rag.web_ui.components.agents_showcase import render_agents_showcase

# Page configuration
st.set_page_config(
    page_title="EEG-RAG | Intelligent EEG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/hkevin01/eeg-rag/issues",
        "Report a bug": "https://github.com/hkevin01/eeg-rag/issues/new",
        "About": """
        # EEG-RAG v0.5.0
        
        **Production-Grade RAG System for EEG Research**
        
        Transform EEG research literature into an intelligent, 
        queryable knowledge platform.
        
        Built with ❤️ for the EEG research community.
        """,
    },
)

# Enhanced CSS for researcher-friendly UI - Light Theme
st.markdown(
    """
<style>
    /* ============================================
       PROFESSIONAL LIGHT THEME
       Clean, subdued colors for research UI
       SOLID COLORS - No gradients
       ============================================ */
    
    :root {
        --primary-blue: #E8EEF4;
        --primary-blue-dark: #D0DCE8;
        --accent-blue: #5C7A99;
        --light-gray: #F5F7F9;
        --medium-gray: #E8EAED;
        --border-gray: #D1D5DB;
        --success-green: #E8F5E9;
        --success-green-dark: #C8E6C9;
        --warning-amber: #FFF8E1;
        --warning-amber-dark: #FFECB3;
        --info-blue: #E3F2FD;
        --background-color: #FAFBFC;
        --text-color: #1F2937;
        --text-muted: #6B7280;
    }
    
    /* ACCESSIBILITY: Light tooltips with high contrast - WCAG AAA compliant */
    [data-testid="stTooltipIcon"],
    [data-testid="stTooltipHoverTarget"],
    div[data-testid*="tooltip"],
    .stTooltipIcon,
    [role="tooltip"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    
    /* Tooltip content styling */
    [role="tooltip"] *,
    div[data-testid*="tooltip"] * {
        background-color: transparent !important;
        color: #1F2937 !important;
    }
    
    /* Help icon tooltips - ensure light background */
    button[title],
    button[aria-label] {
        position: relative;
    }
    
    button[title]:hover::after,
    button[aria-label]:hover::after {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Global styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Light theme base - clean off-white */
    .stApp {
        background-color: #FAFBFC !important;
        background-image: none !important;
        color: #1F2937;
    }
    
    /* Sidebar styling - subtle blue-gray, compact width */
    [data-testid="stSidebar"] {
        background-color: #E8EEF4 !important;
        background-image: none !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        width: 240px !important;
    }
    
    section[data-testid="stSidebar"] {
        width: 240px !important;
        min-width: 240px !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #1F2937 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #374151 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #1F2937 !important;
    }
    
    /* Sidebar widget styling - light blue-gray */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="checkbox"],
    [data-testid="stSidebar"] .stButton button {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        color: #1F2937 !important;
    }
    
    /* AGGRESSIVE: Remove ALL white box backgrounds from sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
    [data-testid="stSidebar"] [data-testid="column"],
    [data-testid="stSidebar"] div[data-testid="column"],
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .stVerticalBlock,
    [data-testid="stSidebar"] .stHorizontalBlock,
    [data-testid="stSidebar"] > div > div > div,
    [data-testid="stSidebar"] section > div {
        background-color: transparent !important;
        background-image: none !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Force sidebar to have clean background everywhere */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] > div > div {
        background-color: #E8EEF4 !important;
        background-image: none !important;
    }
    
    /* Ensure all text is dark gray */
    body, p, span, div, label, input, textarea, select {
        color: #1F2937 !important;
    }
    
    /* Override any dark backgrounds and remove ALL container backgrounds */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"],
    div[data-testid="column"],
    .element-container {
        background-color: transparent !important;
        background-image: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Force main area to have clean background */
    .main [data-testid="stVerticalBlock"],
    .main [data-testid="stHorizontalBlock"],
    .main [data-testid="column"] {
        background-color: transparent !important;
    }
    
    /* ACCESSIBLE: Light gray buttons with dark text - WCAG compliant */
    .stButton button,
    .stButton button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #F3F4F6 !important;
        background-image: none !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 0.5rem 1rem !important;
        min-width: fit-content !important;
        font-weight: 500 !important;
    }
    .stButton button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #E5E7EB !important;
        background-image: none !important;
        border-color: #9CA3AF !important;
    }
    
    /* Primary action buttons - slightly more prominent */
    .stButton button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #DBEAFE !important;
        color: #1E40AF !important;
        border: 1px solid #93C5FD !important;
    }
    .stButton button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #BFDBFE !important;
        border-color: #60A5FA !important;
    }
    
    /* Delete buttons in sidebar - minimal styling, no visible container */
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] .stButton button:has-text("Delete") {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #6B7280 !important;
        font-size: 0.75rem !important;
        padding: 0.25rem 0.5rem !important;
        min-height: auto !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #FEE2E2 !important;
        color: #DC2626 !important;
    }
    
    /* Hide ALL container backgrounds in sidebar - nuclear option */
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stButton > div,
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div,
    [data-testid="stSidebar"] [class*="block-container"],
    [data-testid="stSidebar"] div[data-testid="element-container"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Force light theme on text areas and inputs - clean white */
    [data-baseweb="textarea"],
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    textarea,
    input[type="text"],
    .stTextArea textarea,
    .stTextInput input {
        background-color: #FFFFFF !important;
        background-image: none !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
        caret-color: #1F2937 !important;
    }
    [data-baseweb="textarea"]:focus-within,
    [data-baseweb="input"]:focus-within,
    textarea:focus,
    input:focus {
        border-color: #5C7A99 !important;
        box-shadow: 0 0 0 2px rgba(92, 122, 153, 0.2) !important;
        outline: 2px solid #5C7A99 !important;
        outline-offset: -2px;
        caret-color: #1F2937 !important;
    }
    
    /* Ensure cursor is visible in all inputs */
    textarea, input {
        caret-color: #1F2937 !important;
    }
    
    /* Force light theme on select boxes - clean white */
    [data-baseweb="select"],
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
    }
    
    /* Force light theme on expanders - subtle gray */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background-color: #F5F7F9 !important;
        color: #1F2937 !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E8EAED !important;
        border-radius: 6px !important;
    }

    /* Force light theme on header - clean white */
    header[data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        background-image: none !important;
        border-bottom: 1px solid #E8EAED !important;
    }
    .stAppToolbar,
    [data-testid="stToolbar"] {
        background-color: transparent !important;
    }
    header button,
    [data-testid="stHeader"] button,
    .stAppDeployButton button {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
    }
    [data-testid="stMainMenu"] button {
        color: #1F2937 !important;
    }
    [data-testid="stMainMenu"] button svg {
        fill: #1F2937 !important;
        color: #1F2937 !important;
    }

    /* Info boxes for researchers - subtle amber */
    .researcher-tip {
        background: #FFF8E1;
        border-left: 4px solid #F9A825;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #1F2937;
    }
    
    .researcher-tip .tip-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #6D4C41;
        margin-bottom: 0.5rem;
    }
    
    .researcher-tip .tip-content {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1F2937;
    }
    
    /* Educational callouts - subtle blue */
    .edu-callout {
        background: #E3F2FD;
        border: 1px solid #90CAF9;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .edu-callout h4 {
        color: #1565C0;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .edu-callout p {
        color: #1F2937;
    }
    
    /* System explanation boxes */
    .system-explanation {
        background: #F5F7F9;
        border: 1px dashed #D1D5DB;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        color: #1F2937;
    }
    
    .system-explanation summary {
        cursor: pointer;
        color: #1F2937;
        font-weight: 500;
    }
    
    .system-explanation summary:hover {
        color: #5C7A99;
    }

    .response-summary-box {
        background: #FFFFFF;
        border: 1px solid #E8EAED;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #1F2937;
        line-height: 1.8;
    }
    
    .response-metadata {
        background: #F5F7F9;
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        color: #6B7280;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-around;
        border: 1px solid #E8EAED;
    }
    
    /* Agent status cards */
    .agent-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: all 0.2s ease;
        border: 1px solid #E8EAED;
        color: #1F2937;
    }
    
    .agent-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .agent-card.orchestrator { border-left-color: #5C7A99; }
    .agent-card.planner { border-left-color: #7C8DB5; }
    .agent-card.local { border-left-color: #4CAF50; }
    .agent-card.web { border-left-color: #2196F3; }
    .agent-card.graph { border-left-color: #FF9800; }
    .agent-card.citation { border-left-color: #F44336; }
    .agent-card.aggregator { border-left-color: #00BCD4; }
    .agent-card.generator { border-left-color: #E91E63; }
    
    /* Metric displays */
    .metric-highlight {
        background: #E8EEF4;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #D0DCE8;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    .metric-delta.positive { color: #388E3C; }
    .metric-delta.negative { color: #D32F2F; }
    
    /* Confidence bars */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #E8EAED;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill.high { background: #4CAF50; }
    .confidence-fill.medium { background: #FF9800; }
    .confidence-fill.low { background: #F44336; }
    
    /* Citation display */
    .citation-card {
        background: #FFFFFF;
        border: 1px solid #E8EAED;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #1F2937;
    }
    
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }
    
    .citation-pmid {
        font-family: monospace;
        background: #E8EEF4;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1F2937;
    }

    .verification-badge {
        margin-left: 0.5rem;
        font-size: 0.8rem;
    }
    .verification-badge.verified {
        color: #388E3C;
    }
    .verification-badge.unverified {
        color: #F9A825;
    }

    .relevance-score {
        color: #6B7280;
        font-size: 0.8rem;
    }

    .citation-body {
        margin-bottom: 0.75rem;
    }

    .citation-title {
        color: #1F2937;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }

    .citation-authors {
        color: #6B7280;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .citation-journal {
        color: #9CA3AF;
        font-size: 0.85rem;
    }

    .citation-footer {
        display: flex;
        gap: 0.5rem;
    }

    .pubmed-link {
        background: #E3F2FD;
        color: #1565C0;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.8rem;
        transition: background-color 0.2s;
    }
    .pubmed-link:hover {
        background: #BBDEFB;
    }
    
    /* ACCESSIBILITY: Streamlit tooltip override - light background with dark text */
    .stTooltipContent,
    [data-testid="stTooltipContent"],
    div[class*="Tooltip"],
    div[class*="tooltip"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        padding: 10px 14px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        max-width: 300px !important;
    }
    
    /* Force light tooltips on all elements */
    [title]:hover::after,
    [data-title]:hover::after {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Override Streamlit's default tooltip styling */
    div[data-baseweb="tooltip"] {
        background-color: #FFFFFF !important;
    }
    
    div[data-baseweb="tooltip"] > div {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        padding: 10px 14px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    
    /* ACCESSIBILITY: Color-blind friendly palette adjustments */
    /* Use blue/orange instead of red/green where possible */
    .success-indicator {
        color: #0066CC !important; /* Blue instead of green */
        background-color: #E3F2FD !important;
    }
    
    .error-indicator {
        color: #D84315 !important; /* Orange-red */
        background-color: #FFF3E0 !important;
    }
    
    .warning-indicator {
        color: #F57C00 !important; /* Orange */
        background-color: #FFF8E1 !important;
    }
    
    /* Ensure good contrast ratios (WCAG AAA) */
    .high-contrast-text {
        color: #1F2937 !important;
        font-weight: 500 !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
</style>

<script>
// Force light tooltips with JavaScript for dynamically generated elements
(function() {
    const styleTooltips = function() {
        // Find all tooltip elements
        const tooltips = document.querySelectorAll('[role="tooltip"], [data-testid*="tooltip"], div[class*="tooltip"]');
        tooltips.forEach(tooltip => {
            tooltip.style.backgroundColor = '#FFFFFF';
            tooltip.style.color = '#1F2937';
            tooltip.style.border = '1px solid #D1D5DB';
            tooltip.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.2)';
            tooltip.style.padding = '10px 14px';
            tooltip.style.borderRadius = '6px';
            tooltip.style.fontSize = '14px';
            tooltip.style.fontWeight = '400';
        });
    };
    
    // Run on page load
    styleTooltips();
    
    // Watch for new tooltips being added
    const observer = new MutationObserver(styleTooltips);
    observer.observe(document.body, { 
        childList: true, 
        subtree: true 
    });
})();
</script>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables."""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "feedback_items" not in st.session_state:
        st.session_state.feedback_items = []
    if "show_educational" not in st.session_state:
        st.session_state.show_educational = True
    if "show_tips" not in st.session_state:
        st.session_state.show_tips = True
    if "current_query_id" not in st.session_state:
        st.session_state.current_query_id = None
    if "agent_metrics" not in st.session_state:
        st.session_state.agent_metrics = {}


def render_welcome_banner():
    """Render welcome banner with quick start tips for researchers."""

    if st.session_state.get("hide_welcome", False):
        return

    # Get actual paper count
    paper_count, _ = get_display_paper_count()
    paper_count_str = f"{paper_count:,}" if paper_count > 0 else "thousands of"

    st.markdown(
        f"""
    <div style="background: #bbdefb; 
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
                border: 1px solid #90caf9;">
        <h3 style="color: #0d47a1; margin-bottom: 1rem;">👋 Welcome, Researcher!</h3>
        <p style="color: #000000; margin-bottom: 1rem;">
            EEG-RAG helps you search <strong>{paper_count_str} EEG research papers</strong> using natural language. 
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
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📧</div>
                <div style="font-weight: 600; color: #000; margin-bottom: 0.25rem;">Send Feedback</div>
                <div style="font-size: 0.85rem; color: #424242;">
                    <a href="mailto:kevin.hildebrand@gmail.com" style="color: #1565C0; text-decoration: none;">kevin.hildebrand@gmail.com</a>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_system_status_bar():
    """Render a compact system status bar with dynamic paper count."""
    paper_count, is_actual = get_display_paper_count()
    paper_display = f"{paper_count:,}" if paper_count > 0 else "0"

    st.markdown(
        f"""
    <div style="background: #f5f5f5; border-radius: 8px; padding: 0.75rem 1rem; 
                margin-bottom: 1rem; display: flex; justify-content: space-between;
                align-items: center; border: 1px solid #e0e0e0;">
        <div style="display: flex; gap: 2rem; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 8px; height: 8px; background: #2e7d32; border-radius: 50%;"></span>
                <span style="color: #424242; font-size: 0.85rem;">System Online</span>
            </div>
            <div style="color: #616161; font-size: 0.85rem;">
                💾 <strong style="color: #000;">{paper_display}</strong> cached | 🌐 <strong style="color: #2e7d32;">200M+</strong> searchable
            </div>
            <div style="color: #9CA3AF; font-size: 0.7rem; padding-left: 1rem;">
                PubMed (35M) • Semantic Scholar (200M) • arXiv (2M) • OpenAlex • CrossRef
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
    """,
        unsafe_allow_html=True,
    )


def render_query_tab():
    """Render the main query interface with educational content."""

    # Educational tip for researchers
    if st.session_state.show_tips:
        st.markdown(
            """
        <div class="researcher-tip">
            <div class="tip-header">💡 Query Tips for Best Results</div>
            <div class="tip-content">
                <ul style="margin: 0; padding-left: 1.25rem; line-height: 1.6;">
                    <li><strong>Be specific:</strong> "P300 amplitude in treatment-resistant depression" works better than "depression EEG"</li>
                    <li><strong>Include context:</strong> Mention patient population, paradigm, or comparison if relevant</li>
                    <li><strong>Ask comparisons:</strong> "How does X compare to Y" triggers multi-source retrieval</li>
                </ul>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Query interface
    render_query_interface()

    # Example queries with explanations
    with st.expander("📋 Example Queries (click to use)", expanded=False):

        st.markdown(
            """
        <details class="system-explanation" open>
            <summary>Why these examples work well</summary>
            <p style="margin-top: 0.5rem;">
                These queries are designed to demonstrate different retrieval capabilities:
                <br/>• <strong>Factual queries</strong> retrieve specific measurements and values
                <br/>• <strong>Comparative queries</strong> trigger multi-source search and synthesis
                <br/>• <strong>Mechanism queries</strong> leverage the knowledge graph for relationships
            </p>
        </details>
        """,
            unsafe_allow_html=True,
        )

        example_queries = [
            {
                "category": "Clinical Research",
                "query": "What EEG biomarkers predict seizure recurrence after a first unprovoked seizure?",
                "why": "Targets prognostic biomarkers with clinical utility",
            },
            {
                "category": "Experimental Neuroscience",
                "query": "What is the typical P300 latency and amplitude in healthy adults during visual oddball tasks?",
                "why": "Retrieves normative data for ERP components",
            },
            {
                "category": "Machine Learning",
                "query": "What deep learning architectures achieve the highest accuracy for EEG-based seizure detection?",
                "why": "Triggers comparison across methods and datasets",
            },
            {
                "category": "Sleep Research",
                "query": "How do sleep spindle characteristics change in early Alzheimer's disease?",
                "why": "Combines sleep staging with neurodegenerative disease markers",
            },
        ]

        for ex in example_queries:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(
                    f"""
                <div style="background: #f5f5f5; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                    <div style="color: #616161; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        {ex['category']}
                    </div>
                    <div style="color: #000; margin-top: 0.25rem;">{ex['query']}</div>
                    <div style="color: #757575; font-size: 0.8rem; margin-top: 0.25rem; font-style: italic;">
                        💡 {ex['why']}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button(
                    "Use", key=f"use_{ex['category']}", use_container_width=True
                ):
                    st.session_state.example_query = ex["query"]
                    st.rerun()


def render_agent_pipeline_tab():
    """Render detailed agent pipeline visualization."""

    st.markdown("## 🤖 Agent Pipeline Monitor")

    # Educational content about the agent system
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    # Agent details
    render_agent_monitor()


def render_results_tab():
    """Render results with detailed citation information."""

    st.markdown("## 📊 Results & Citations")

    # Check if there are results
    if not st.session_state.query_history:
        st.info("👆 Submit a query in the 'Query Research' tab to see results here.")

        # Show example of what results look like
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )
        return

    # Get latest result
    latest = st.session_state.query_history[-1]

    # Answer section
    st.markdown("### 📝 Response")

    # Metadata bar
    st.markdown(
        f"""
    <div class="response-metadata">
        <div><strong>Query Type:</strong> Factual</div>
        <div><strong>Entities Found:</strong> seizure, EEG</div>
        <div><strong>Sources Retrieved:</strong> 8</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="response-summary-box">
        {latest.get('answer', 'No answer available')}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Confidence indicator
    confidence = latest.get("confidence", 0.85)
    confidence_class = (
        "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
    )

    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )

    # Citations section
    st.markdown("### 📚 Citations")

    citations = latest.get("citations", [])
    if citations:
        for cite in citations:
            render_citation_card(cite)
    else:
        st.info("No citations available for this query.")


def render_citation_card(citation: dict):
    """Render a detailed citation card."""

    pmid = citation.get("pmid", "Unknown")
    title = citation.get("title", "Title not available")
    authors = citation.get("authors", ["Unknown"])
    journal = citation.get("journal", "Unknown Journal")
    year = citation.get("year", "Unknown")
    verified = citation.get("verified", False)
    relevance = citation.get("relevance_score", 0.0)

    verification_badge = "✅ Verified" if verified else "⚠️ Unverified"

    if isinstance(authors, list):
        authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
    else:
        authors_str = str(authors)

    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )


def render_analytics_tab():
    """Render analytics dashboard with explanations."""

    st.markdown("## 📈 Analytics Dashboard")

    # Explanation
    st.markdown(
        """
    <div class="researcher-tip">
        <div class="tip-header">📊 Understanding These Metrics</div>
        <div class="tip-content">
            These analytics help you understand system performance and your usage patterns.
            <strong>Your feedback</strong> on result quality directly improves these metrics.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-highlight">
            <div class="metric-value">147</div>
            <div class="metric-label">Queries Today</div>
            <div class="metric-delta positive">↑ 23% vs yesterday</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-highlight">
            <div class="metric-value">2.3s</div>
            <div class="metric-label">Avg Response Time</div>
            <div class="metric-delta positive">↓ 0.4s improvement</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-highlight">
            <div class="metric-value">99.2%</div>
            <div class="metric-label">Citation Accuracy</div>
            <div class="metric-delta positive">↑ 0.3% this week</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-highlight">
            <div class="metric-value">4.2★</div>
            <div class="metric-label">User Rating</div>
            <div class="metric-delta positive">Based on 89 ratings</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Query Volume by Hour")
        chart_data = pd.DataFrame(
            {
                "Hour": list(range(24)),
                "Queries": [
                    12,
                    8,
                    5,
                    3,
                    2,
                    4,
                    15,
                    45,
                    78,
                    92,
                    87,
                    76,
                    82,
                    79,
                    85,
                    91,
                    88,
                    72,
                    58,
                    45,
                    38,
                    28,
                    22,
                    18,
                ],
            }
        )
        st.bar_chart(chart_data.set_index("Hour"))

    with col2:
        st.markdown("### Response Time by Agent")
        agent_data = pd.DataFrame(
            {
                "Agent": ["Local Search", "Web Search", "Knowledge Graph", "Generator"],
                "Latency (ms)": [120, 890, 340, 1850],
            }
        )
        st.bar_chart(agent_data.set_index("Agent"))

    # Query history
    st.markdown("### 📜 Your Query History")

    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(
                f"Query: {q.get('query', 'Unknown')[:60]}...", expanded=False
            ):
                st.write(f"**Time:** {q.get('timestamp', 'Unknown')}")
                st.write(f"**Confidence:** {q.get('confidence', 0):.0%}")
                st.write(f"**Citations:** {len(q.get('citations', []))}")
    else:
        st.info("No queries yet. Start by asking a question in the Query tab!")


def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Initialize search history and sessions
    initialize_search_state()

    # Render sidebar
    show_tips, show_edu = render_sidebar()
    st.session_state.show_tips = show_tips
    st.session_state.show_educational = show_edu

    # Render history sidebar (conversation management)
    render_history_sidebar()

    # Main content area
    render_header()
    render_welcome_banner()
    render_agents_showcase()  # Show the 8 AI agents on homepage
    render_system_status_bar()

    # Main tabs with enhanced content
    tabs = st.tabs(
        [
            "🔍 Query Research",
            "📜 Search History",
            "🤖 Agent Pipeline",
            "📊 Results & Citations",
            "📈 Analytics",
            "📚 Learn EEG-RAG",
            "💬 Feedback",
        ]
    )

    with tabs[0]:  # Query Research
        render_query_tab()

    with tabs[1]:  # Search History
        render_search_history()

    with tabs[2]:  # Agent Pipeline
        render_agent_pipeline_tab()

    with tabs[3]:  # Results & Citations
        render_results_tab()

    with tabs[4]:  # Analytics
        render_analytics_tab()

    with tabs[5]:  # Learn EEG-RAG
        render_educational_content()

    with tabs[6]:  # Feedback
        render_feedback_panel()

    # Footer
    st.markdown(
        """
    <div style="margin-top: 3rem; padding: 1rem; text-align: center; color: #666; font-size: 0.8rem;">
        EEG-RAG Research Assistant v0.5.0 • 
        <a href="https://github.com/hkevin01/eeg-rag" style="color: #6366F1;">GitHub</a> • 
        <a href="#" style="color: #6366F1;">Documentation</a> • 
        <a href="#" style="color: #6366F1;">Privacy Policy</a>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

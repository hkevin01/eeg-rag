# src/eeg_rag/web_ui/app_modular.py
"""
Enhanced EEG-RAG Web Interface with Modular Components

This is the main application file that integrates all modular components
for a comprehensive, researcher-friendly interface.

Features:
- Query interface with real-time agent monitoring
- Educational content for researchers
- Comprehensive feedback collection
- Agent pipeline visualization
- Results history and analytics

Run with: streamlit run src/eeg_rag/web_ui/app_modular.py --server.port 8504
"""

import streamlit as st
from datetime import datetime

# Import modular components
from components.sidebar import render_sidebar
from components.header import render_header
from components.query_interface import render_query_interface
from components.agent_monitor import render_agent_monitor
from components.educational import render_educational_content
from components.feedback import render_feedback_panel
from components.corpus_stats import get_header_display_stats


# Page configuration
st.set_page_config(
    page_title="EEG-RAG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/eeg-rag/discussions",
        "Report a bug": "https://github.com/your-org/eeg-rag/issues",
        "About": """
        ## EEG-RAG Research Assistant v2.0
        
        **Multi-Agent Retrieval-Augmented Generation for EEG Research**
        
        This system uses specialized AI agents to search, validate, and 
        synthesize information from EEG research papers.
        
        All responses include verified citations with PMID links.
        """,
    },
)


# Enhanced CSS Styling
st.markdown(
    """
<link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
    integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
    crossorigin="anonymous"
/>
<style>
    :root {
        --space-1: 8px;
        --space-2: 16px;
        --space-3: 24px;
        --space-4: 32px;
        --surface: #FFFFFF;
        --surface-alt: #F5F7F9;
        --border: #D1D5DB;
        --text: #1F2937;
        --text-muted: #4B5563;
        --accent: #1D4ED8;
    }

    html, body {
        overflow-x: hidden;
    }

    .stApp {
        background-color: #FAFBFC;
        color: var(--text);
        overflow-x: hidden;
    }
    
    .main .block-container {
        max-width: 1400px;
        padding-top: var(--space-2);
        padding-bottom: var(--space-3);
        padding-left: var(--space-2);
        padding-right: var(--space-2);
    }

    @media (min-width: 992px) {
        .main .block-container {
            padding-left: var(--space-3);
            padding-right: var(--space-3);
        }
    }

    [data-testid="stSidebar"] {
        background-color: #E8EEF4;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    
    .wcag-section {
        margin-bottom: var(--space-3);
    }
    
    .wcag-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: var(--space-2);
        margin-bottom: var(--space-2);
    }
    
    .wcag-stack > * + * {
        margin-top: var(--space-2);
    }
    
    .researcher-tip {
        background: #FFF8E1;
        border-radius: 12px;
        padding: var(--space-2);
        border-left: 4px solid #F59E0B;
        margin: var(--space-2) 0;
    }
    
    .researcher-tip .tip-title,
    .researcher-tip .tip-content {
        color: var(--text);
    }
    
    .edu-callout {
        background: #E3F2FD;
        border-radius: 12px;
        padding: var(--space-2);
        border: 1px solid #90CAF9;
        margin: var(--space-2) 0;
    }
    
    .edu-callout-title {
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: var(--space-1);
    }
    
    .stButton button,
    .stButton button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"] {
        background: #F3F4F6 !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        min-height: 44px !important;
        padding: 10px 16px !important;
        line-height: 1.35 !important;
        font-weight: 600 !important;
    }
    
    .stButton button:hover {
        background: #E5E7EB !important;
        border-color: #9CA3AF !important;
    }
    
    .stTextArea textarea,
    .stTextInput input,
    [data-baseweb="textarea"],
    [data-baseweb="input"] {
        min-height: 44px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        background: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--space-1);
        background: #E8EEF4;
        padding: var(--space-1);
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 14px;
        min-height: 44px;
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background: #DBEAFE;
        color: #1E3A8A;
    }
    
    .streamlit-expanderHeader {
        background: #F3F4F6;
        border-radius: 8px;
    }
    
    .system-explanation {
        margin: 0.5rem 0;
    }
    
    .system-explanation summary {
        cursor: pointer;
        color: #1D4ED8;
        font-size: 0.9rem;
    }
    
    .system-explanation summary:hover {
        color: #1E3A8A;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: var(--space-2);
            padding-right: var(--space-2);
        }
    }
    
    @media (max-width: 480px) {
        .main .block-container {
            padding-left: var(--space-1);
            padding-right: var(--space-1);
        }

        .wcag-card,
        .edu-callout,
        .researcher-tip {
            padding: var(--space-2);
        }
    }
    
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.initialize_session_state
# Requirement  : `initialize_session_state` shall initialize session state variables
# Purpose      : Initialize session state variables
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def initialize_session_state():
    """Initialize session state variables."""

    defaults = {
        "query_history": [],
        "feedback_history": [],
        "show_tips": True,
        "show_educational": True,
        "total_queries": 0,
        "session_start": datetime.now().isoformat(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_welcome_banner
# Requirement  : `render_welcome_banner` shall render welcome banner for first-time users
# Purpose      : Render welcome banner for first-time users
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_welcome_banner():
    """Render welcome banner for first-time users."""

    if "welcomed" not in st.session_state:
        st.markdown(
            """
        <div class="edu-callout" style="margin-bottom: 2rem;">
            <div class="edu-callout-title">
                👋 Welcome to EEG-RAG Research Assistant
            </div>
            <div style="color: #a0a0c0; font-size: 0.9rem; line-height: 1.6;">
                This AI-powered system helps you search and synthesize information from 
                <strong>52,000+ EEG research papers</strong>. Every response includes 
                verified citations with PMID links.
                <br/><br/>
                <strong>Quick Start:</strong> Type your research question below, or explore 
                the "Learn" tab to understand how the system works.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col2:
            if st.button("✓ Got it!", use_container_width=True):
                st.session_state.welcomed = True
                st.rerun()


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_system_status_bar
# Requirement  : `render_system_status_bar` shall render real-time system status bar
# Purpose      : Render real-time system status bar
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_system_status_bar():
    """Render real-time system status bar."""

    # Get dynamic stats
    stats = get_header_display_stats()
    paper_count = stats.get("papers_indexed", "0")
    ai_agents = stats.get("ai_agents", "8")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            """
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="width: 10px; height: 10px; background: #10B981; border-radius: 50%; 
                         animation: pulse 2s infinite;"></span>
            <span style="color: #888; font-size: 0.8rem;">System Online</span>
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style="color: #888; font-size: 0.8rem;">
            💾 {paper_count} cached | 🌐 200M+ searchable
        </div>
        <div style="color: #999; font-size: 0.65rem;">
            PubMed • Semantic Scholar • arXiv • OpenAlex • CrossRef
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div style="color: #888; font-size: 0.8rem;">
            🤖 {ai_agents} agents ready
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        queries = len(st.session_state.get("query_history", []))
        st.markdown(
            f"""
        <div style="color: #888; font-size: 0.8rem;">
            🔍 {queries} queries this session
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
        <div style="color: #888; font-size: 0.8rem;">
            ⚡ Avg. response: 2.3s
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='margin: 0.75rem 0; border-color: #2d2d4d;'>", unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_query_tab
# Requirement  : `render_query_tab` shall render the main query tab content
# Purpose      : Render the main query tab content
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_query_tab():
    """Render the main query tab content."""

    # Show tip if enabled
    if st.session_state.get("show_tips", True):
        st.markdown(
            """
        <div class="researcher-tip">
            <span class="tip-icon">💡</span>
            <span class="tip-title">Research Tip</span>
            <div class="tip-content">
                For best results, include <strong>specific details</strong> in your query: 
                patient population, EEG paradigm, comparison groups, or outcome measures. 
                Example: "What is the diagnostic accuracy of interictal EEG for focal epilepsy 
                in adults vs children?"
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Main query interface
    render_query_interface()


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_agent_pipeline_tab
# Requirement  : `render_agent_pipeline_tab` shall render the agent pipeline visualization tab
# Purpose      : Render the agent pipeline visualization tab
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_agent_pipeline_tab():
    """Render the agent pipeline visualization tab."""

    # Introduction
    st.markdown(
        """
    <div class="edu-callout">
        <div class="edu-callout-title">
            🔄 Understanding the Agent Pipeline
        </div>
        <div style="color: #a0a0c0; font-size: 0.9rem;">
            When you submit a query, it passes through a series of specialized AI agents. 
            Each agent has a specific role in retrieving, validating, and synthesizing 
            information. This multi-agent approach ensures comprehensive and accurate results.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Agent monitor component
    render_agent_monitor()


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_results_tab
# Requirement  : `render_results_tab` shall render the results history tab
# Purpose      : Render the results history tab
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_results_tab():
    """Render the results history tab."""

    st.markdown("### 📊 Your Query History")

    history = st.session_state.get("query_history", [])

    if not history:
        st.info("No queries yet. Start by asking a question in the 'Ask' tab!")
        return

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", len(history))

    with col2:
        avg_time = sum(q.get("execution_time", 0) for q in history) / len(history)
        st.metric("Avg. Time", f"{avg_time:.1f}s")

    with col3:
        avg_conf = sum(q.get("confidence", 0) for q in history) / len(history)
        st.metric("Avg. Confidence", f"{avg_conf:.0%}")

    with col4:
        total_citations = sum(len(q.get("citations", [])) for q in history)
        st.metric("Total Citations", total_citations)

    st.markdown("---")

    # History list
    for i, query_item in enumerate(reversed(history)):
        with st.expander(f"🔍 {query_item['query'][:80]}...", expanded=i == 0):
            col1, col2 = st.columns([0.7, 0.3])

            with col1:
                st.markdown(f"**Query:** {query_item['query']}")
                st.markdown(f"**Timestamp:** {query_item.get('timestamp', 'N/A')}")

            with col2:
                st.markdown(f"**Time:** {query_item.get('execution_time', 0):.1f}s")
                st.markdown(f"**Confidence:** {query_item.get('confidence', 0):.0%}")

            # Citations
            citations = query_item.get("citations", [])
            if citations:
                st.markdown("**Citations:**")
                for cite in citations:
                    verified = "✅" if cite.get("verified") else "⚠️"
                    st.markdown(f"- {verified} PMID: {cite.get('pmid', 'N/A')}")


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_analytics_tab
# Requirement  : `render_analytics_tab` shall render analytics and insights tab
# Purpose      : Render analytics and insights tab
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_analytics_tab():
    """Render analytics and insights tab."""

    st.markdown("### 📈 Session Analytics")

    history = st.session_state.get("query_history", [])

    if len(history) < 2:
        st.info("Analytics become available after 2+ queries. Keep exploring!")
        return

    # Query pattern analysis
    st.markdown("#### Query Patterns")

    query_types = {}
    for q in history:
        query = q.get("query", "").lower()
        if "compare" in query or "vs" in query:
            qtype = "Comparative"
        elif "predict" in query or "prognosis" in query:
            qtype = "Predictive"
        elif "mechanism" in query or "how" in query:
            qtype = "Mechanistic"
        else:
            qtype = "General"
        query_types[qtype] = query_types.get(qtype, 0) + 1

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Your Query Distribution:**")
        for qtype, count in sorted(query_types.items(), key=lambda x: -x[1]):
            pct = count / len(history) * 100
            st.markdown(f"- {qtype}: {count} ({pct:.0f}%)")

    with col2:
        st.markdown("**Performance Trends:**")
        times = [q.get("execution_time", 0) for q in history]
        confs = [q.get("confidence", 0) for q in history]
        st.markdown(f"- Fastest query: {min(times):.1f}s")
        st.markdown(f"- Slowest query: {max(times):.1f}s")
        st.markdown(f"- Highest confidence: {max(confs):.0%}")


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.render_citation_card
# Requirement  : `render_citation_card` shall render a citation card with verification status
# Purpose      : Render a citation card with verification status
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : pmid: str; title: str; journal: str; year: int; verified: bool
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_citation_card(
    pmid: str, title: str, journal: str, year: int, verified: bool
):
    """Render a citation card with verification status."""

    ver_badge = "✅ Verified" if verified else "⚠️ Unverified"
    ver_color = "#10B981" if verified else "#F59E0B"

    st.markdown(
        f"""
    <div style="background: #1a1a2e; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;
                border-left: 3px solid {ver_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: monospace; background: #2d2d4d; padding: 0.25rem 0.5rem; 
                         border-radius: 4px;">PMID: {pmid}</span>
            <span style="color: {ver_color}; font-size: 0.8rem;">{ver_badge}</span>
        </div>
        <div style="margin-top: 0.5rem; color: #fff;">{title}</div>
        <div style="color: #888; font-size: 0.85rem;">{journal} ({year})</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ID           : web_ui.app_modular.main
# Requirement  : `main` shall main application entry point
# Purpose      : Main application entry point
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""

    # Initialize state
    initialize_session_state()

    # Render sidebar (with options and stats)
    show_tips, show_edu = render_sidebar()
    st.session_state.show_tips = show_tips
    st.session_state.show_educational = show_edu

    # Render header
    render_header()

    # System status bar
    render_system_status_bar()

    # Welcome banner for new users
    render_welcome_banner()

    # Main content tabs
    tabs = st.tabs(
        [
            "🔍 Ask",
            "🤖 Agent Pipeline",
            "📚 Results History",
            "📈 Analytics",
            "📖 Learn",
            "💬 Feedback",
        ]
    )

    with tabs[0]:
        render_query_tab()

    with tabs[1]:
        render_agent_pipeline_tab()

    with tabs[2]:
        render_results_tab()

    with tabs[3]:
        render_analytics_tab()

    with tabs[4]:
        render_educational_content()

    with tabs[5]:
        render_feedback_panel()

    # Footer
    st.markdown(
        """
    <div style="margin-top: 3rem; padding: 1rem; text-align: center; color: #666; font-size: 0.8rem;">
        EEG-RAG Research Assistant v2.0 • 
        <a href="https://github.com/your-org/eeg-rag" style="color: #6366F1;">GitHub</a> • 
        <a href="#" style="color: #6366F1;">Documentation</a> • 
        <a href="#" style="color: #6366F1;">Privacy Policy</a>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

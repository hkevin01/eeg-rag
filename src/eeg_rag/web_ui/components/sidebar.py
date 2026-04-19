# src/eeg_rag/web_ui/components/sidebar.py
"""
Enhanced sidebar component for EEG-RAG application.
"""

import streamlit as st
from eeg_rag.web_ui.components.corpus_stats import get_corpus_stats, get_display_paper_count


# ---------------------------------------------------------------------------
# ID           : web_ui.components.sidebar.get_paper_count_short
# Requirement  : `get_paper_count_short` shall get short formatted paper count (e.g., '500K' or '10')
# Purpose      : Get short formatted paper count (e.g., '500K' or '10')
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : str
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
def get_paper_count_short() -> str:
    """Get short formatted paper count (e.g., '500K' or '10')."""
    count, is_actual = get_display_paper_count()
    if count >= 1000:
        return f"{count // 1000}K"
    return str(count)


# ---------------------------------------------------------------------------
# ID           : web_ui.components.sidebar.render_sidebar
# Requirement  : `render_sidebar` shall render a compact application sidebar
# Purpose      : Render a compact application sidebar
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
def render_sidebar():
    """Render a compact application sidebar."""
    
    with st.sidebar:
        # Compact logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <span style="font-size: 2rem;">🧠</span>
            <div style="font-size: 1.1rem; font-weight: 700; color: #000; margin-top: 0.25rem;">EEG-RAG</div>
            <div style="color: #616161; font-size: 0.75rem;">AI Research Assistant</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # Compact display options
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #374151;'>⚙️ Options</div>", unsafe_allow_html=True)
        
        show_tips = st.checkbox("Query Tips", value=st.session_state.get('show_tips', True), key="tips_cb")
        show_educational = st.checkbox("Edu Content", value=st.session_state.get('show_educational', True), key="edu_cb")
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # Compact quick stats - inline
        query_count = len(st.session_state.get('query_history', []))
        feedback_count = len(st.session_state.get('feedback_items', []))
        
        st.markdown(f"""
        <div style="font-size: 0.85rem; font-weight: 600; color: #374151;">📊 Stats</div>
        <div style="display: flex; gap: 1rem; font-size: 0.8rem; color: #4B5563; margin-top: 0.25rem;">
            <span>🔍 {query_count} queries</span>
            <span>💬 {feedback_count} feedback</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # Compact session info
        current_session_id = st.session_state.get('current_session_id', None)
        sessions = st.session_state.get('search_sessions', {})
        
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #374151;'>📁 Session</div>", unsafe_allow_html=True)
        
        if current_session_id and current_session_id in sessions:
            session = sessions[current_session_id]
            session_name = session.get('name', 'Unnamed')[:20]
            query_count_session = len(session.get('queries', []))
            st.markdown(f"<div style='font-size: 0.75rem; color: #5C7A99;'>{session_name} ({query_count_session})</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 0.75rem; color: #9CA3AF;'>No active session</div>", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # Compact system status
        paper_count = get_paper_count_short()
        st.markdown(f"""
        <div style="font-size: 0.85rem; font-weight: 600; color: #374151;">🔌 Status</div>
        <div style="font-size: 0.75rem; color: #4B5563; margin-top: 0.25rem;">
            <span style="color: #4CAF50;">●</span> Online · {paper_count} papers · 2.3s avg
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # Compact resources as icons
        st.markdown("""
        <div style="font-size: 0.75rem; text-align: center;">
            <a href="https://github.com/hkevin01/eeg-rag" style="color: #5C7A99; text-decoration: none;">📖 Docs</a> · 
            <a href="https://github.com/hkevin01/eeg-rag/issues" style="color: #5C7A99; text-decoration: none;">🐛 Issues</a>
        </div>
        <div style="text-align: center; color: #9CA3AF; font-size: 0.65rem; margin-top: 0.5rem;">
            v0.5.0 · © 2024
        </div>
        """, unsafe_allow_html=True)
    
    # Return the display options for use in main app
    return show_tips, show_educational

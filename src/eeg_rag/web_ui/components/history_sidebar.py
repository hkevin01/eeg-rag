"""
History Sidebar Component
Displays conversation history with search and session management.
"""

import streamlit as st
from datetime import datetime
from typing import Optional
from .search_history import HistoryManager, HistorySession


# ---------------------------------------------------------------------------
# ID           : web_ui.components.history_sidebar.render_history_sidebar
# Requirement  : `render_history_sidebar` shall render the conversation history sidebar
# Purpose      : Render the conversation history sidebar
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
def render_history_sidebar():
    """Render the conversation history sidebar."""

    # Initialize history manager
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = HistoryManager()

    manager = st.session_state.history_manager

    with st.sidebar:
        st.markdown("### 📚 Conversation History")

        # Search history
        search_query = st.text_input(
            "🔍 Search history", placeholder="Search queries..."
        )

        if search_query:
            results = manager.search_history(search_query)
            if results:
                st.markdown(f"**{len(results)} results found**")
                for result in results:
                    with st.expander(f"📄 {result['session_title'][:30]}..."):
                        st.markdown(f"**Query:** {result['content'][:100]}...")
                        st.caption(f"📅 {result['timestamp']}")
                        if st.button("Load", key=f"load_{result['message_id']}"):
                            st.session_state.active_session_id = result["session_id"]
                            st.rerun()
            else:
                st.info("No results found")

        # Statistics
        stats = manager.get_statistics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sessions", stats["total_sessions"])
        with col2:
            st.metric("Queries", stats["total_queries"])

        st.divider()

        # Session list
        st.markdown("**Recent Sessions**")

        # New session button
        if st.button("➕ New Session", use_container_width=True):
            st.session_state.active_session_id = None
            st.session_state.pop("current_filtered_papers", None)
            st.session_state.pop("last_query_cache_key", None)
            st.rerun()

        sessions = manager.get_sessions(limit=20)

        if not sessions:
            st.info("No sessions yet. Start by asking a question!")
        else:
            for session in sessions:
                # Format timestamp
                created = datetime.fromisoformat(session.created_at)
                time_str = created.strftime("%b %d, %H:%M")

                # Active session indicator
                is_active = st.session_state.get("active_session_id") == session.id

                # Shorter title to prevent cutoff (20 chars max)
                short_title = session.title[:20]
                if len(session.title) > 20:
                    short_title += "..."

                button_label = f"✓ {short_title}" if is_active else f"📂 {short_title}"

                # Single load button
                if st.button(
                    button_label,
                    key=f"session_{session.id}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    st.session_state.active_session_id = session.id
                    _load_session_data(manager, session.id)
                    st.rerun()

                # Metadata with delete in expander
                with st.expander(
                    f"📅 {time_str} • {session.query_count}q", expanded=False
                ):
                    if st.button(
                        "🗑️ Delete Session",
                        key=f"del_{session.id}",
                        use_container_width=True,
                    ):
                        manager.delete_session(session.id)
                        if st.session_state.get("active_session_id") == session.id:
                            st.session_state.active_session_id = None
                        st.rerun()

        # Export options
        st.markdown("**Export**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Markdown", use_container_width=True):
                _export_session_markdown(manager)
        with col2:
            if st.button("📊 JSON", use_container_width=True):
                _export_session_json(manager)


# ---------------------------------------------------------------------------
# ID           : web_ui.components.history_sidebar._load_session_data
# Requirement  : `_load_session_data` shall load session data into session state
# Purpose      : Load session data into session state
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : manager: HistoryManager; session_id: str
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
def _load_session_data(manager: HistoryManager, session_id: str):
    """Load session data into session state."""
    messages = manager.get_session_messages(session_id)

    # Find the last query and its response
    for i, msg in enumerate(messages):
        if msg.role == "user" and i + 1 < len(messages):
            response = messages[i + 1]
            # Store in session state for display
            st.session_state.loaded_query = msg.content
            st.session_state.loaded_response = response.content
            st.session_state.loaded_paper_count = response.paper_count
            st.session_state.loaded_execution_time = response.execution_time
            # Also pre-fill the query input
            st.session_state.query_input = msg.content
            # Clear any cached results from previous queries
            st.session_state.pop("current_filtered_papers", None)
            st.session_state.pop("last_executed_query", None)


# ---------------------------------------------------------------------------
# ID           : web_ui.components.history_sidebar._export_session_markdown
# Requirement  : `_export_session_markdown` shall export current session to Markdown
# Purpose      : Export current session to Markdown
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : manager: HistoryManager
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
def _export_session_markdown(manager: HistoryManager):
    """Export current session to Markdown."""
    session_id = st.session_state.get("active_session_id")
    if not session_id:
        st.warning("No active session to export")
        return

    sessions = manager.get_sessions()
    session = next((s for s in sessions if s.id == session_id), None)
    if not session:
        st.error("Session not found")
        return

    messages = manager.get_session_messages(session_id)

    # Generate markdown
    md = f"# {session.title}\n\n"
    md += f"**Created:** {session.created_at}\n"
    md += f"**Last Updated:** {session.updated_at}\n"
    md += f"**Total Queries:** {session.query_count}\n\n"
    md += "---\n\n"

    for msg in messages:
        role_emoji = "🙋" if msg.role == "user" else "🤖"
        md += f"## {role_emoji} {msg.role.title()}\n\n"
        md += f"**Time:** {msg.timestamp}\n\n"
        md += f"{msg.content}\n\n"
        if msg.role == "assistant":
            md += f"*Papers: {msg.paper_count} | Time: {msg.execution_time:.2f}s*\n\n"
        md += "---\n\n"

    # Download button
    st.download_button(
        label="📥 Download Markdown",
        data=md,
        file_name=f"session_{session_id}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# ID           : web_ui.components.history_sidebar._export_session_json
# Requirement  : `_export_session_json` shall export current session to JSON
# Purpose      : Export current session to JSON
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : manager: HistoryManager
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
def _export_session_json(manager: HistoryManager):
    """Export current session to JSON."""
    import json

    session_id = st.session_state.get("active_session_id")
    if not session_id:
        st.warning("No active session to export")
        return

    sessions = manager.get_sessions()
    session = next((s for s in sessions if s.id == session_id), None)
    if not session:
        st.error("Session not found")
        return

    messages = manager.get_session_messages(session_id)

    # Build JSON structure
    data = {
        "session": {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "tags": session.tags,
            "query_count": session.query_count,
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "paper_count": msg.paper_count,
                "execution_time": msg.execution_time,
                "relevance_threshold": msg.relevance_threshold,
            }
            for msg in messages
        ],
    }

    # Download button
    st.download_button(
        label="📥 Download JSON",
        data=json.dumps(data, indent=2),
        file_name=f"session_{session_id}.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# ID           : web_ui.components.history_sidebar.get_or_create_session
# Requirement  : `get_or_create_session` shall get active session or create new one based on query
# Purpose      : Get active session or create new one based on query
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : manager: HistoryManager; query: str
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
def get_or_create_session(manager: HistoryManager, query: str) -> str:
    """Get active session or create new one based on query."""

    # Check if we have an active session
    if "active_session_id" in st.session_state and st.session_state.active_session_id:
        return st.session_state.active_session_id

    # Create new session with query as title
    title = query[:50] + "..." if len(query) > 50 else query
    session = manager.create_session(title)
    st.session_state.active_session_id = session.id

    return session.id

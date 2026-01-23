# src/eeg_rag/web_ui/components/search_history.py
"""
Search History Component - Manages search sessions, history tracking, and recall.
Provides persistent storage and session-based organization of searches.
"""

import streamlit as st
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, List, Any


# History storage path
HISTORY_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "search_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def initialize_search_state():
    """Initialize search history and session state."""
    if 'search_sessions' not in st.session_state:
        st.session_state.search_sessions = {}
    
    if 'current_session_id' not in st.session_state:
        # Create a new session on app load
        st.session_state.current_session_id = create_new_session()
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Load persisted history
    load_persisted_history()


def create_new_session(name: Optional[str] = None) -> str:
    """Create a new search session."""
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now()
    
    session_name = name or f"Session {timestamp.strftime('%b %d, %H:%M')}"
    
    st.session_state.search_sessions[session_id] = {
        'id': session_id,
        'name': session_name,
        'created_at': timestamp.isoformat(),
        'updated_at': timestamp.isoformat(),
        'queries': [],
        'tags': [],
        'notes': '',
        'is_favorite': False
    }
    
    return session_id


def get_current_session() -> Dict[str, Any]:
    """Get the current active session."""
    if st.session_state.current_session_id not in st.session_state.search_sessions:
        st.session_state.current_session_id = create_new_session()
    
    return st.session_state.search_sessions[st.session_state.current_session_id]


def add_search_to_session(
    query: str,
    query_id: str,
    results: Optional[Dict[str, Any]] = None,
    confidence: float = 0.0,
    citations: Optional[List[Dict]] = None,
    execution_time_ms: int = 0,
    agents_used: Optional[List[str]] = None
):
    """Add a search to the current session."""
    session = get_current_session()
    
    search_entry = {
        'id': query_id,
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'confidence': confidence,
        'execution_time_ms': execution_time_ms,
        'agents_used': agents_used or ['Local', 'PubMed'],
        'citation_count': len(citations) if citations else 0,
        'citations': citations or [],
        'results_summary': results.get('summary', '') if results else '',
        'query_type': results.get('query_type', 'factual') if results else 'factual',
        'entities_found': results.get('entities', []) if results else [],
    }
    
    session['queries'].append(search_entry)
    session['updated_at'] = datetime.now().isoformat()
    
    # Also add to flat history for easy access
    st.session_state.query_history.append(search_entry)
    
    # Persist to disk
    save_session_to_disk(st.session_state.current_session_id)


def save_session_to_disk(session_id: str):
    """Save a session to persistent storage."""
    if session_id not in st.session_state.search_sessions:
        return
    
    session = st.session_state.search_sessions[session_id]
    filepath = HISTORY_DIR / f"session_{session_id}.json"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(session, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save session: {e}")


def load_persisted_history():
    """Load all persisted sessions from disk."""
    try:
        for filepath in HISTORY_DIR.glob("session_*.json"):
            with open(filepath, 'r') as f:
                session = json.load(f)
                session_id = session['id']
                
                # Only load if not already in memory
                if session_id not in st.session_state.search_sessions:
                    st.session_state.search_sessions[session_id] = session
                    
                    # Add queries to flat history
                    for query_entry in session.get('queries', []):
                        if query_entry not in st.session_state.query_history:
                            st.session_state.query_history.append(query_entry)
    except Exception as e:
        pass  # Silent fail on load errors


def delete_session(session_id: str):
    """Delete a session and its persisted data."""
    if session_id in st.session_state.search_sessions:
        del st.session_state.search_sessions[session_id]
    
    filepath = HISTORY_DIR / f"session_{session_id}.json"
    if filepath.exists():
        filepath.unlink()
    
    # Remove queries from flat history
    st.session_state.query_history = [
        q for q in st.session_state.query_history 
        if not any(
            s.get('id') == session_id and q in s.get('queries', [])
            for s in st.session_state.search_sessions.values()
        )
    ]


def render_search_history():
    """Render the search history and session management interface."""
    initialize_search_state()
    
    st.markdown("## 📜 Search History & Sessions")
    st.markdown("""
    <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid #90CAF9;">
        <div style="color: #1565C0; font-weight: 600; margin-bottom: 0.5rem;">💡 About Search Sessions</div>
        <div style="color: #000; font-size: 0.9rem;">
            Sessions help you organize related searches together. Each session tracks your queries, 
            results, citations, and timing. You can create new sessions for different research topics,
            recall past searches, and export your research history.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Session management row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Session selector
        session_options = {
            sid: f"{s['name']} ({len(s['queries'])} queries)"
            for sid, s in st.session_state.search_sessions.items()
        }
        
        if session_options:
            selected_session = st.selectbox(
                "📁 Current Session",
                options=list(session_options.keys()),
                format_func=lambda x: session_options[x],
                index=list(session_options.keys()).index(st.session_state.current_session_id) 
                      if st.session_state.current_session_id in session_options else 0
            )
            
            if selected_session != st.session_state.current_session_id:
                st.session_state.current_session_id = selected_session
                st.rerun()
    
    with col2:
        if st.button("➕ New Session", use_container_width=True):
            new_id = create_new_session()
            st.session_state.current_session_id = new_id
            st.success("New session created!")
            st.rerun()
    
    with col3:
        if st.button("💾 Save All", use_container_width=True):
            for sid in st.session_state.search_sessions:
                save_session_to_disk(sid)
            st.success("All sessions saved!")
    
    with col4:
        if st.button("📥 Export", use_container_width=True):
            export_current_session()
    
    st.markdown("---")
    
    # Create tabs for different views
    history_tabs = st.tabs(["📋 Current Session", "📚 All Sessions", "🔍 Search All", "📊 Statistics"])
    
    with history_tabs[0]:
        render_current_session_view()
    
    with history_tabs[1]:
        render_all_sessions_view()
    
    with history_tabs[2]:
        render_search_across_sessions()
    
    with history_tabs[3]:
        render_history_statistics()


def render_current_session_view():
    """Render the current session's queries."""
    session = get_current_session()
    
    # Session info header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_name = st.text_input(
            "Session Name",
            value=session['name'],
            key="session_name_input"
        )
        if new_name != session['name']:
            session['name'] = new_name
            save_session_to_disk(session['id'])
    
    with col2:
        st.markdown(f"""
        <div style="background: #C8E6C9; padding: 0.75rem; border-radius: 8px; text-align: center; margin-top: 1.5rem;">
            <div style="color: #1B5E20; font-size: 1.5rem; font-weight: 700;">{len(session['queries'])}</div>
            <div style="color: #2E7D32; font-size: 0.8rem;">Searches</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Session tags
    st.text_input(
        "Tags (comma-separated)",
        value=", ".join(session.get('tags', [])),
        key="session_tags",
        help="Add tags to organize your sessions",
        on_change=lambda: update_session_tags(session['id'])
    )
    
    # Session notes
    st.text_area(
        "Session Notes",
        value=session.get('notes', ''),
        key="session_notes",
        height=80,
        help="Add notes about this research session",
        on_change=lambda: update_session_notes(session['id'])
    )
    
    st.markdown("---")
    
    # List queries in this session
    if not session['queries']:
        st.info("🔍 No searches yet in this session. Go to the Query Research tab to start searching!")
    else:
        st.markdown("### 📋 Search History")
        
        # Reverse to show most recent first
        for idx, query_entry in enumerate(reversed(session['queries'])):
            render_search_entry_card(query_entry, idx, allow_recall=True)


def render_search_entry_card(query_entry: Dict[str, Any], idx: int, allow_recall: bool = True):
    """Render a single search entry card."""
    timestamp = datetime.fromisoformat(query_entry['timestamp'])
    time_ago = get_time_ago(timestamp)
    
    confidence = query_entry.get('confidence', 0)
    conf_color = "#4CAF50" if confidence >= 0.8 else "#FF9800" if confidence >= 0.6 else "#F44336"
    conf_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
    
    with st.container():
        st.markdown(f"""
        <div style="background: #FFFDE7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; 
                    border-left: 4px solid {conf_color}; border: 1px solid #FFF59D;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <div style="color: #000; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">
                        "{query_entry['query'][:100]}{'...' if len(query_entry['query']) > 100 else ''}"
                    </div>
                    <div style="color: #666; font-size: 0.85rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                        <span>🕐 {time_ago}</span>
                        <span>{conf_emoji} {confidence:.0%} confidence</span>
                        <span>📚 {query_entry.get('citation_count', 0)} citations</span>
                        <span>⚡ {query_entry.get('execution_time_ms', 0)}ms</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if allow_recall and st.button("🔄 Recall", key=f"recall_{query_entry['id']}_{idx}", use_container_width=True):
                st.session_state.example_query = query_entry['query']
                st.rerun()
        
        with col2:
            if st.button("📋 Details", key=f"details_{query_entry['id']}_{idx}", use_container_width=True):
                st.session_state[f"show_details_{query_entry['id']}"] = not st.session_state.get(f"show_details_{query_entry['id']}", False)
                st.rerun()
        
        # Show details if expanded
        if st.session_state.get(f"show_details_{query_entry['id']}", False):
            render_search_details(query_entry)


def render_search_details(query_entry: Dict[str, Any]):
    """Render detailed view of a search entry."""
    with st.expander("📊 Full Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Query Analysis**")
            st.markdown(f"- **Type:** {query_entry.get('query_type', 'Unknown').title()}")
            st.markdown(f"- **Entities:** {', '.join(query_entry.get('entities_found', [])) or 'None detected'}")
            st.markdown(f"- **Agents Used:** {', '.join(query_entry.get('agents_used', []))}")
        
        with col2:
            st.markdown("**Performance**")
            st.markdown(f"- **Execution Time:** {query_entry.get('execution_time_ms', 0)}ms")
            st.markdown(f"- **Confidence:** {query_entry.get('confidence', 0):.1%}")
            st.markdown(f"- **Citations Found:** {query_entry.get('citation_count', 0)}")
        
        # Citations list
        if query_entry.get('citations'):
            st.markdown("---")
            st.markdown("**📚 Citations Retrieved**")
            for cite in query_entry['citations'][:5]:
                pmid = cite.get('pmid', 'N/A')
                title = cite.get('title', 'Unknown')[:60]
                st.markdown(f"- [PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}) - {title}...")


def render_all_sessions_view():
    """Render a view of all saved sessions."""
    st.markdown("### 📚 All Search Sessions")
    
    sessions = list(st.session_state.search_sessions.values())
    
    if not sessions:
        st.info("No sessions found. Create a new session to get started!")
        return
    
    # Sort by updated_at descending
    sessions.sort(key=lambda s: s.get('updated_at', ''), reverse=True)
    
    for session in sessions:
        is_current = session['id'] == st.session_state.current_session_id
        bg_color = "#E8F5E9" if is_current else "#FAFAFA"
        border_color = "#4CAF50" if is_current else "#E0E0E0"
        
        created = datetime.fromisoformat(session['created_at'])
        updated = datetime.fromisoformat(session['updated_at'])
        
        with st.container():
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
                        border: 2px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #000; font-weight: 600; font-size: 1.1rem;">
                            {'⭐ ' if session.get('is_favorite') else ''}{session['name']}
                            {' (Active)' if is_current else ''}
                        </div>
                        <div style="color: #666; font-size: 0.85rem; margin-top: 0.25rem;">
                            📅 Created: {created.strftime('%b %d, %Y %H:%M')} | 
                            📝 Updated: {get_time_ago(updated)} |
                            🔍 {len(session['queries'])} searches
                        </div>
                        {f"<div style='color: #1976D2; font-size: 0.85rem; margin-top: 0.25rem;'>🏷️ {', '.join(session.get('tags', []))}</div>" if session.get('tags') else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if not is_current:
                    if st.button("📂 Open", key=f"open_{session['id']}", use_container_width=True):
                        st.session_state.current_session_id = session['id']
                        st.rerun()
            
            with col2:
                fav_label = "⭐ Unfavorite" if session.get('is_favorite') else "☆ Favorite"
                if st.button(fav_label, key=f"fav_{session['id']}", use_container_width=True):
                    toggle_favorite(session['id'])
                    st.rerun()
            
            with col3:
                if st.button("📥 Export", key=f"export_{session['id']}", use_container_width=True):
                    export_session(session['id'])
            
            with col4:
                if not is_current:
                    if st.button("🗑️ Delete", key=f"delete_{session['id']}", use_container_width=True):
                        delete_session(session['id'])
                        st.success(f"Session '{session['name']}' deleted")
                        st.rerun()


def render_search_across_sessions():
    """Search across all sessions for specific queries."""
    st.markdown("### 🔍 Search Across All Sessions")
    
    search_term = st.text_input(
        "Search queries, citations, or notes",
        placeholder="Enter keywords to search...",
        key="history_search_input"
    )
    
    if search_term:
        results = []
        search_lower = search_term.lower()
        
        for session in st.session_state.search_sessions.values():
            for query_entry in session.get('queries', []):
                # Search in query text
                if search_lower in query_entry.get('query', '').lower():
                    results.append({
                        'session': session,
                        'entry': query_entry,
                        'match_type': 'Query'
                    })
                # Search in entities
                elif any(search_lower in e.lower() for e in query_entry.get('entities_found', [])):
                    results.append({
                        'session': session,
                        'entry': query_entry,
                        'match_type': 'Entity'
                    })
                # Search in citations
                elif any(search_lower in str(c).lower() for c in query_entry.get('citations', [])):
                    results.append({
                        'session': session,
                        'entry': query_entry,
                        'match_type': 'Citation'
                    })
        
        if results:
            st.success(f"Found {len(results)} matching searches")
            
            for idx, result in enumerate(results):
                session = result['session']
                entry = result['entry']
                
                st.markdown(f"""
                <div style="background: #FFF3E0; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                            border-left: 4px solid #FF9800;">
                    <div style="color: #E65100; font-size: 0.8rem; margin-bottom: 0.25rem;">
                        📁 {session['name']} | Match: {result['match_type']}
                    </div>
                    <div style="color: #000; font-weight: 500;">
                        "{entry['query'][:80]}..."
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Recall this search", key=f"recall_search_{idx}"):
                    st.session_state.current_session_id = session['id']
                    st.session_state.example_query = entry['query']
                    st.rerun()
        else:
            st.info("No matches found. Try different keywords.")
    else:
        st.info("Enter keywords above to search through your search history.")


def render_history_statistics():
    """Render statistics about search history."""
    st.markdown("### 📊 Search History Statistics")
    
    all_queries = st.session_state.query_history
    all_sessions = st.session_state.search_sessions
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: #E1BEE7; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="color: #6A1B9A; font-size: 2rem; font-weight: 700;">{len(all_sessions)}</div>
            <div style="color: #7B1FA2; font-size: 0.9rem;">Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #BBDEFB; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="color: #1565C0; font-size: 2rem; font-weight: 700;">{len(all_queries)}</div>
            <div style="color: #1976D2; font-size: 0.9rem;">Total Searches</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_citations = sum(q.get('citation_count', 0) for q in all_queries)
        st.markdown(f"""
        <div style="background: #C8E6C9; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="color: #2E7D32; font-size: 2rem; font-weight: 700;">{total_citations}</div>
            <div style="color: #388E3C; font-size: 0.9rem;">Citations Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_conf = sum(q.get('confidence', 0) for q in all_queries) / len(all_queries) if all_queries else 0
        st.markdown(f"""
        <div style="background: #FFE0B2; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="color: #E65100; font-size: 2rem; font-weight: 700;">{avg_conf:.0%}</div>
            <div style="color: #F57C00; font-size: 0.9rem;">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    if all_queries:
        st.markdown("---")
        
        # Most common entities
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏷️ Most Searched Topics")
            entity_counts = {}
            for q in all_queries:
                for entity in q.get('entities_found', []):
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            if entity_counts:
                sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for entity, count in sorted_entities:
                    st.markdown(f"- **{entity}**: {count} searches")
            else:
                st.info("No entity data yet")
        
        with col2:
            st.markdown("#### ⏱️ Recent Activity")
            recent = sorted(all_queries, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
            for q in recent:
                ts = datetime.fromisoformat(q['timestamp'])
                st.markdown(f"- {get_time_ago(ts)}: *{q['query'][:40]}...*")


def get_time_ago(timestamp: datetime) -> str:
    """Get human-readable time ago string."""
    now = datetime.now()
    diff = now - timestamp
    
    if diff < timedelta(minutes=1):
        return "Just now"
    elif diff < timedelta(hours=1):
        mins = int(diff.total_seconds() / 60)
        return f"{mins}m ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours}h ago"
    elif diff < timedelta(days=7):
        days = diff.days
        return f"{days}d ago"
    else:
        return timestamp.strftime("%b %d, %Y")


def update_session_tags(session_id: str):
    """Update session tags from input."""
    tags_str = st.session_state.get("session_tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    
    if session_id in st.session_state.search_sessions:
        st.session_state.search_sessions[session_id]['tags'] = tags
        save_session_to_disk(session_id)


def update_session_notes(session_id: str):
    """Update session notes from input."""
    notes = st.session_state.get("session_notes", "")
    
    if session_id in st.session_state.search_sessions:
        st.session_state.search_sessions[session_id]['notes'] = notes
        save_session_to_disk(session_id)


def toggle_favorite(session_id: str):
    """Toggle favorite status of a session."""
    if session_id in st.session_state.search_sessions:
        current = st.session_state.search_sessions[session_id].get('is_favorite', False)
        st.session_state.search_sessions[session_id]['is_favorite'] = not current
        save_session_to_disk(session_id)


def export_current_session():
    """Export the current session as JSON."""
    export_session(st.session_state.current_session_id)


def export_session(session_id: str):
    """Export a specific session as downloadable JSON."""
    if session_id not in st.session_state.search_sessions:
        st.error("Session not found")
        return
    
    session = st.session_state.search_sessions[session_id]
    json_str = json.dumps(session, indent=2)
    
    st.download_button(
        label="📥 Download Session JSON",
        data=json_str,
        file_name=f"eeg_rag_session_{session_id}.json",
        mime="application/json",
        key=f"download_{session_id}"
    )

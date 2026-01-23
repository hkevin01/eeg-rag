# src/eeg_rag/web_ui/components/sidebar.py
"""
Enhanced sidebar component for EEG-RAG application.
"""

import streamlit as st


def render_sidebar():
    """Render the application sidebar with navigation and settings."""
    
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🧠</span>
            <h2 style="margin: 0.5rem 0 0 0; color: #000;">EEG-RAG</h2>
            <p style="color: #616161; font-size: 0.85rem; margin: 0;">
                AI Research Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display options
        st.markdown("### ⚙️ Display Options")
        
        show_tips = st.checkbox(
            "Show Query Tips",
            value=st.session_state.get('show_tips', True),
            help="Display helpful tips for writing effective queries"
        )
        
        show_educational = st.checkbox(
            "Show Educational Content",
            value=st.session_state.get('show_educational', True),
            help="Display explanations about how the system works"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        
        query_count = len(st.session_state.get('query_history', []))
        feedback_count = len(st.session_state.get('feedback_items', []))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", query_count)
        with col2:
            st.metric("Feedback", feedback_count)
        
        st.markdown("---")
        
        # Current Session Info
        st.markdown("### 📁 Current Session")
        
        current_session_id = st.session_state.get('current_session_id', None)
        sessions = st.session_state.get('search_sessions', {})
        
        if current_session_id and current_session_id in sessions:
            session = sessions[current_session_id]
            session_name = session.get('name', 'Unnamed Session')
            query_count_session = len(session.get('queries', []))
            
            st.markdown(f"""
            <div style="background: #E8F5E9; padding: 0.75rem; border-radius: 8px; border: 1px solid #A5D6A7;">
                <div style="color: #1B5E20; font-weight: 600; font-size: 0.9rem;">{session_name}</div>
                <div style="color: #2E7D32; font-size: 0.8rem; margin-top: 0.25rem;">
                    🔍 {query_count_session} searches in session
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #FFF3E0; padding: 0.75rem; border-radius: 8px; border: 1px solid #FFE0B2;">
                <div style="color: #E65100; font-size: 0.85rem;">No active session</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 🔌 System Status")
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="width: 8px; height: 8px; background: #2e7d32; border-radius: 50%;"></span>
            <span style="color: #424242; font-size: 0.85rem;">All agents online</span>
        </div>
        <div style="color: #616161; font-size: 0.8rem;">
            📚 52,431 papers indexed<br/>
            🔄 Last sync: 2h ago<br/>
            ⚡ Avg response: 2.3s
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Resources
        st.markdown("### 📚 Resources")
        
        st.markdown("""
        - [Documentation](https://github.com/hkevin01/eeg-rag/wiki)
        - [API Reference](https://github.com/hkevin01/eeg-rag/docs/api)
        - [Report Issue](https://github.com/hkevin01/eeg-rag/issues/new)
        """)
        
        st.markdown("---")
        
        # Version info
        st.markdown("""
        <div style="text-align: center; color: #616161; font-size: 0.8rem;">
            EEG-RAG v0.5.0 Beta<br/>
            © 2024 EEG Research Team
        </div>
        """, unsafe_allow_html=True)
    
    # Return the display options for use in main app
    return show_tips, show_educational

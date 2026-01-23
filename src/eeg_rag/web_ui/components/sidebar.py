# src/eeg_rag/web_ui/components/sidebar.py
"""
Sidebar component for EEG-RAG application.
"""

import streamlit as st
from pathlib import Path


def render_sidebar():
    """Render the application sidebar with navigation and settings."""
    
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🧠</span>
            <h2 style="margin: 0.5rem 0 0 0; color: #fff;">EEG-RAG</h2>
            <p style="color: #888; font-size: 0.85rem; margin: 0;">
                AI Research Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display options
        st.markdown("### ⚙️ Display Options")
        
        st.session_state.show_tips = st.checkbox(
            "Show Query Tips",
            value=st.session_state.get('show_tips', True),
            help="Display helpful tips for writing effective queries"
        )
        
        st.session_state.show_educational = st.checkbox(
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
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            EEG-RAG v0.5.0 Beta<br/>
            © 2024 EEG Research Team
        </div>
        """, unsafe_allow_html=True)

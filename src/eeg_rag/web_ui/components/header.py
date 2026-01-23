# src/eeg_rag/web_ui/components/header.py
"""
Enhanced header component for EEG-RAG application.
"""

import streamlit as st


def render_header():
    """Render the application header with title and quick metrics."""
    
    st.markdown("""
    <div style="background: #c5cae9;
                border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1rem;
                border: 1px solid #9fa8da;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #000; font-size: 2rem;">
                    🧠 EEG-RAG Research Assistant
                </h1>
                <p style="color: #424242; margin: 0.5rem 0 0 0; font-size: 1rem;">
                    AI-powered literature search for EEG research with verified citations
                </p>
            </div>
            <div style="display: flex; gap: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #6366F1;">52,431</div>
                    <div style="font-size: 0.75rem; color: #616161; text-transform: uppercase;">Papers Indexed</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #10B981;">8</div>
                    <div style="font-size: 0.75rem; color: #616161; text-transform: uppercase;">AI Agents</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #F59E0B;">99.2%</div>
                    <div style="font-size: 0.75rem; color: #616161; text-transform: uppercase;">Citation Accuracy</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

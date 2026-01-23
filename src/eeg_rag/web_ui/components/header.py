# src/eeg_rag/web_ui/components/header.py
"""
Header component for EEG-RAG application.
"""

import streamlit as st


def render_header():
    """Render the application header."""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid #3d3d5c;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #fff; font-size: 1.75rem;">
                    🧠 EEG-RAG Research Assistant
                </h1>
                <p style="margin: 0.5rem 0 0 0; color: #a0a0c0; font-size: 0.95rem;">
                    Query 50,000+ peer-reviewed EEG papers with AI-powered search and verified citations
                </p>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <div style="text-align: center; padding: 0.5rem 1rem; 
                                background: rgba(16, 185, 129, 0.1); border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: #10B981;">52,431</div>
                        <div style="font-size: 0.75rem; color: #a0a0c0;">Papers Indexed</div>
                    </div>
                    <div style="text-align: center; padding: 0.5rem 1rem; 
                                background: rgba(99, 102, 241, 0.1); border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: #6366F1;">8</div>
                        <div style="font-size: 0.75rem; color: #a0a0c0;">AI Agents</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

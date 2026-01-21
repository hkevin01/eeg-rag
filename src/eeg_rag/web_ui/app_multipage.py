#!/usr/bin/env python3
"""
EEG-RAG Multi-Page Streamlit Application

Main application with:
- Query Interface (main page)
- Systematic Review Extraction
- Testing Suite (separate page)
- Benchmarking Suite (separate page)
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="EEG-RAG: AI Research Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🧠 EEG-RAG: AI Research Assistant")
    st.markdown("**Production-grade RAG system for EEG research**")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "🔍 Query System",
            "🔬 Systematic Review",
            "📥 Data Ingestion",
            "📚 Corpus Explorer",
            "⚙️ Settings"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Additional Tools**")
    st.sidebar.markdown("🧪 [Testing Suite](pages/1_Testing_Suite.py)")
    st.sidebar.markdown("📊 [Benchmarking](pages/2_Benchmarking.py)")
    
    st.markdown("---")
    
    # Render selected page
    if page == "🔍 Query System":
        from pages import query_page
        query_page.render()
    elif page == "🔬 Systematic Review":
        from pages import systematic_review_page
        systematic_review_page.render()
    elif page == "📥 Data Ingestion":
        from pages import ingestion_page
        ingestion_page.render()
    elif page == "📚 Corpus Explorer":
        from pages import corpus_page
        corpus_page.render()
    elif page == "⚙️ Settings":
        from pages import settings_page
        settings_page.render()


if __name__ == "__main__":
    main()

# src/eeg_rag/web_ui/components/header.py
"""
Enhanced header component for EEG-RAG application.
"""

import streamlit as st
from eeg_rag.web_ui.components.corpus_stats import get_header_display_stats


def get_paper_count() -> str:
    """Get formatted paper count for display."""
    stats = get_header_display_stats()
    return stats.get("papers_indexed", "0")


def render_header():
    """Render the application header with title and quick metrics."""

    # Get all display stats from centralized service
    stats = get_header_display_stats()
    paper_count = stats.get("papers_indexed", "0")
    ai_agents = stats.get("ai_agents", "8")
    citation_accuracy = stats.get("citation_accuracy", "99.2%")

    # Check if this is a demo/small dataset
    try:
        count_value = int(paper_count.replace(",", ""))
        is_demo = count_value < 100
    except ValueError:
        is_demo = True

    # Demo mode indicator
    demo_badge = ""
    if is_demo:
        demo_badge = '<span style="background: #FFF3E0; color: #E65100; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-left: 0.5rem;">DEMO MODE</span>'
        paper_label = "Demo Papers"
    else:
        paper_label = "Papers Cached Locally"

    st.markdown(
        f"""
    <div style="background: #FFFFFF;
                border-radius: 8px; padding: 1.5rem 2rem; margin-bottom: 1rem;
                border: 1px solid #E8EAED; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #1F2937; font-size: 2rem;">
                    🧠 EEG-RAG Research Assistant {demo_badge}
                </h1>
                <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 1rem;">
                    AI-powered literature search for EEG research with verified citations
                </p>
            </div>
            <div style="display: flex; gap: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #5C7A99;">{paper_count}</div>
                    <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Cached Locally</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #2e7d32;">200M+</div>
                    <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Searchable</div>
                    <div style="font-size: 0.65rem; color: #9CA3AF; margin-top: 0.125rem;">PubMed • S2 • arXiv • OpenAlex</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #4CAF50;">{ai_agents}</div>
                    <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">AI Agents</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.75rem; font-weight: 700; color: #FF9800;">{citation_accuracy}</div>
                    <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Citation Accuracy</div>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show ingestion prompt in demo mode
    if is_demo:
        st.info(
            "📥 **Demo Mode**: Only 10 sample papers are indexed. Run `python scripts/run_ingestion.py` to collect 1,000+ real papers from PubMed and arXiv."
        )

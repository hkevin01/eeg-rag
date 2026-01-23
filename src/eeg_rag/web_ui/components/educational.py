# src/eeg_rag/web_ui/components/educational.py
"""
Educational Content Component - Comprehensive learning resources for researchers.
Helps users understand RAG systems, EEG domain, and how to use EEG-RAG effectively.
"""

import streamlit as st
from eeg_rag.web_ui.components.corpus_stats import get_corpus_stats, get_display_paper_count


def get_paper_count_formatted() -> str:
    """Get formatted paper count."""
    count, is_actual = get_display_paper_count()
    return f"{count:,}" if count > 0 else "0"


def render_educational_content():
    """Render comprehensive educational content about EEG-RAG."""
    
    st.markdown("## 📚 Learn About EEG-RAG")
    
    # PASTEL PEACH header
    st.markdown("""
    <div style="background: #F5F7F9; 
                border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #E8EAED;">
        <p style="color: #1F2937; margin: 0;">
            This section helps you understand how EEG-RAG works, master query strategies, 
            and interpret results with confidence. Select a topic below to learn more.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Educational tabs
    edu_tabs = st.tabs([
        "🔄 RAG Basics",
        "🧠 EEG Domain",
        "🎯 Query Strategies", 
        "📊 Interpreting Results",
        "⚠️ Limitations"
    ])
    
    with edu_tabs[0]:
        render_rag_basics()
    
    with edu_tabs[1]:
        render_eeg_domain()
    
    with edu_tabs[2]:
        render_query_strategies()
    
    with edu_tabs[3]:
        render_interpreting_results()
    
    with edu_tabs[4]:
        render_limitations()


def render_rag_basics():
    """Explain RAG fundamentals for researchers."""
    
    st.markdown("### What is Retrieval-Augmented Generation (RAG)?")
    
    # PASTEL BLUE callout
    st.markdown("""
    <div style="background: #E3F2FD; 
                border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #BBDEFB;">
        <p style="color: #1F2937; font-size: 1.1rem; line-height: 1.8; margin: 0;">
            <strong>RAG</strong> combines the power of large language models (LLMs) with 
            real-time document retrieval. Instead of relying solely on the model's training data,
            RAG systems <em>retrieve</em> relevant documents first, then <em>generate</em> 
            answers based on that specific evidence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚫 Traditional LLM (Without RAG)")
        st.markdown("""
        - Uses only information from training data
        - Training data has a cutoff date
        - Cannot cite specific sources
        - May "hallucinate" facts confidently
        - No domain specialization
        """)
    
    with col2:
        st.markdown("#### ✅ RAG System (EEG-RAG)")
        st.markdown("""
        - Retrieves current literature before answering
        - Accesses papers published today via PubMed
        - Every claim linked to verifiable PMID
        - Grounded in actual research documents
        - Specialized for EEG/neuroscience domain
        """)
    
    st.markdown("---")
    
    st.markdown("#### How EEG-RAG Works (Simplified)")
    
    st.markdown("""
    ```
    Your Query → Query Analysis → Multi-Source Retrieval → Context Aggregation → Response Generation → Citation Verification → Final Answer
    ```
    
    1. **Query Analysis**: Your question is analyzed to identify key concepts, entities, and intent
    2. **Multi-Source Retrieval**: Relevant passages are retrieved from local index, PubMed, and knowledge graph
    3. **Context Aggregation**: Results are merged, deduplicated, and ranked by relevance
    4. **Response Generation**: An LLM synthesizes the context into a coherent answer
    5. **Citation Verification**: Every PMID is validated against PubMed before inclusion
    """)


def render_eeg_domain():
    """Explain EEG-specific domain knowledge encoded in the system."""
    
    st.markdown("### EEG Domain Knowledge in EEG-RAG")
    
    # Professional callout
    st.markdown("""
    <div style="background: #E8F5E9; 
                border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #C8E6C9;">
        <p style="color: #1F2937; margin: 0;">
            EEG-RAG is built with deep understanding of EEG research terminology, concepts, 
            and relationships. This domain knowledge improves retrieval accuracy and 
            response quality.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Entity types
    st.markdown("#### 🏷️ Recognized Entity Types")
    
    entity_types = {
        "Brain Regions": ["Frontal cortex", "Temporal lobe", "Occipital area", "Hippocampus"],
        "Frequency Bands": ["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)", "Gamma (30-100+ Hz)"],
        "ERP Components": ["P300", "N400", "P600", "MMN", "N170", "ERN"],
        "Clinical Conditions": ["Epilepsy", "Alzheimer's", "Depression", "Schizophrenia", "Sleep disorders"],
        "Paradigms/Tasks": ["Oddball task", "Go/No-Go", "Stroop", "N-back", "Resting-state"],
        "Electrode Systems": ["10-20 system", "10-10 system", "High-density EEG", "Source localization"]
    }
    
    cols = st.columns(3)
    for idx, (category, examples) in enumerate(entity_types.items()):
        with cols[idx % 3]:
            st.markdown(f"**{category}**")
            for ex in examples[:4]:
                st.markdown(f"- {ex}")
    
    st.markdown("---")
    
    st.markdown("#### 📊 Literature Coverage")
    
    paper_count = get_paper_count_formatted()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Indexed Papers", paper_count)
        st.caption("Peer-reviewed EEG research")
    
    with col2:
        st.metric("EEG Terms", "458")
        st.caption("In knowledge graph")
    
    with col3:
        st.metric("Relationships", "2,340")
        st.caption("Concept connections")


def render_query_strategies():
    """Teach effective query formulation strategies."""
    
    st.markdown("### Crafting Effective Queries")
    
    st.markdown("""
    <div class="researcher-tip">
        <div class="tip-header">💡 The Key to Great Results</div>
        <div class="tip-content">
            The quality of your query directly affects the quality of results. 
            Being specific, using domain terminology, and including context 
            significantly improves retrieval accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Query type examples
    st.markdown("#### Query Types and Examples")
    
    query_examples = [
        {
            "type": "Factual Query",
            "description": "Retrieve specific facts, measurements, or definitions",
            "good": "What is the typical P300 latency range in healthy young adults during visual oddball tasks?",
            "bad": "Tell me about P300",
            "tip": "Include population, paradigm, and specific measurement"
        },
        {
            "type": "Comparative Query",
            "description": "Compare conditions, methods, or findings",
            "good": "How does alpha power asymmetry in major depression compare to healthy controls?",
            "bad": "Depression vs normal EEG",
            "tip": "Specify what aspect you're comparing"
        },
        {
            "type": "Mechanism Query",
            "description": "Understand how or why something works",
            "good": "What neural mechanisms generate sleep spindles and how do they relate to memory consolidation?",
            "bad": "How do spindles work",
            "tip": "Ask about specific mechanisms or pathways"
        },
        {
            "type": "Clinical Query",
            "description": "Clinical applications, diagnosis, prognosis",
            "good": "What EEG biomarkers have shown utility for predicting conversion from MCI to Alzheimer's disease?",
            "bad": "EEG for Alzheimer's",
            "tip": "Specify clinical application context"
        }
    ]
    
    for ex in query_examples:
        with st.expander(f"**{ex['type']}**: {ex['description']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("✅ **Good Query:**")
                st.info(ex['good'])
            with col2:
                st.markdown("❌ **Vague Query:**")
                st.error(ex['bad'])
            st.markdown(f"💡 **Tip:** {ex['tip']}")
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Query Optimization Checklist")
    
    st.markdown("""
    - [ ] **Be specific**: Include specific EEG components, frequencies, or paradigms
    - [ ] **Define population**: Mention age, condition, or comparison groups
    - [ ] **State measurement**: Specify amplitude, latency, power, or connectivity
    - [ ] **Provide context**: Include task, condition, or clinical setting
    - [ ] **Use domain terms**: Use standard EEG terminology when possible
    """)


def render_interpreting_results():
    """Guide users on interpreting EEG-RAG responses."""
    
    st.markdown("### Understanding Your Results")
    
    # Professional gray callout
    st.markdown("""
    <div style="background: #F5F7F9; 
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #E8EAED;">
        <p style="color: #1F2937; margin: 0;">
            EEG-RAG provides structured responses with confidence scores and citations. 
            Here's how to interpret each component of your results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence scores
    st.markdown("#### 📊 Confidence Scores Explained")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Professional green for high confidence
        st.markdown("""
        <div style="background: #E8F5E9; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #C8E6C9;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #2E7D32;">80-100%</div>
            <div style="color: #388E3C; font-size: 0.9rem;">High Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Multiple sources agree
        - Citations verified
        - Well-established finding
        """)
    
    with col2:
        # Professional amber for medium confidence
        st.markdown("""
        <div style="background: #FFF8E1; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #FFE082;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #E65100;">60-79%</div>
            <div style="color: #F57C00; font-size: 0.9rem;">Medium Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Some source disagreement
        - Limited evidence
        - Emerging research area
        """)
    
    with col3:
        # Professional rose for low confidence
        st.markdown("""
        <div style="background: #FFEBEE; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #FFCDD2;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #C62828;"><60%</div>
            <div style="color: #D32F2F; font-size: 0.9rem;">Low Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Conflicting evidence
        - Few sources found
        - Speculative content
        """)
    
    st.markdown("---")
    
    # Citations
    st.markdown("#### 📚 Reading Citations")
    
    st.markdown("""
    Each citation includes:
    
    - **PMID**: PubMed identifier - click to verify the source
    - **Verification Status**: ✅ means we confirmed the paper exists and supports the claim
    - **Relevance Score**: How closely the cited passage matches your query
    - **Publication Info**: Title, authors, journal, and year
    
    **Always verify critical information** by clicking the PubMed link and reviewing the original paper.
    """)


def render_limitations():
    """Transparently communicate system limitations."""
    
    st.markdown("### System Limitations & Best Practices")
    
    # Professional amber warning box
    st.markdown("""
    <div style="background: #FFF8E1; 
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
                border: 1px solid #FFE082;">
        <h4 style="color: #E65100; margin-bottom: 0.75rem;">⚠️ Important Limitations</h4>
        <p style="color: #4E342E; margin: 0;">
            While EEG-RAG strives for accuracy, it's a research assistance tool - not a replacement 
            for expert judgment or primary literature review. Always verify critical information.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Limitation categories
    limitations = [
        {
            "category": "📚 Literature Coverage",
            "items": [
                "Indexed corpus may not include very recent publications (PubMed indexing delay)",
                "Some specialized sub-fields may have limited coverage",
                "Non-English publications are underrepresented",
                "Preprints are included but not peer-reviewed"
            ]
        },
        {
            "category": "🧠 Domain Boundaries",
            "items": [
                "Optimized for EEG research - may be less accurate for other modalities",
                "Clinical recommendations require expert interpretation",
                "Rare conditions may have insufficient training data",
                "Cutting-edge methods may not be well-represented"
            ]
        },
        {
            "category": "🤖 AI Limitations",
            "items": [
                "LLMs can occasionally produce plausible-sounding but incorrect information",
                "Complex multi-step reasoning may have errors",
                "Quantitative claims should be verified against source",
                "Context window limits may truncate very long documents"
            ]
        },
        {
            "category": "✅ Citation Verification",
            "items": [
                "Verification confirms paper exists, not that interpretation is correct",
                "Claim-source alignment is probabilistic, not guaranteed",
                "Some citations may be tangentially related",
                "Always read the original paper for critical decisions"
            ]
        }
    ]
    
    for lim in limitations:
        with st.expander(lim["category"], expanded=False):
            for item in lim["items"]:
                st.markdown(f"- {item}")
    
    st.markdown("---")
    
    st.markdown("#### ✅ Best Practices for Research Use")
    
    st.markdown("""
    1. **Use as a starting point**: EEG-RAG helps you discover relevant literature faster
    2. **Verify critical claims**: Always click through to the original papers for important findings
    3. **Cross-reference**: Compare EEG-RAG results with your own literature searches
    4. **Report issues**: Use the Feedback tab to report inaccuracies - this improves the system
    5. **Consider confidence**: Weight information by confidence score and citation quality
    """)

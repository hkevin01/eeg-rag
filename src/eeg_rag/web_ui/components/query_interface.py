# src/eeg_rag/web_ui/components/query_interface.py
"""
Query Interface Component - Main query input and execution with live agent monitoring.
"""

import streamlit as st
from datetime import datetime
import uuid
import time
import re


def render_query_interface():
    """Render the main query input interface with options."""
    
    # Initialize query_input in session state if not present
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ''
    
    # If an example query was selected, update the query_input
    if 'example_query' in st.session_state:
        st.session_state.query_input = st.session_state.example_query
        del st.session_state.example_query
        st.rerun()
    
    # Query input
    query = st.text_area(
        "Enter your EEG research question",
        placeholder="Example: What are the P300 amplitude differences between patients with treatment-resistant depression and healthy controls?",
        height=100,
        key="query_input"
    )
    
    # Query options
    with st.expander("⚙️ Query Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            execution_mode = st.selectbox(
                "Execution Mode",
                ["Balanced", "Fast (Local Only)", "Comprehensive (All Sources)"],
                help="Trade-off between speed and completeness"
            )
        
        with col2:
            max_sources = st.slider(
                "Max Citations",
                min_value=3,
                max_value=20,
                value=10,
                help="Maximum number of sources to cite"
            )
        
        with col3:
            include_trials = st.checkbox(
                "Include Clinical Trials",
                value=False,
                help="Search ClinicalTrials.gov for relevant trials"
            )
        
        validate_citations = st.checkbox(
            "Validate All Citations",
            value=True,
            help="Verify each PMID against PubMed (slower but more reliable)"
        )
    
    # Submit button
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        submit = st.button(
            "🔍 Search EEG Literature",
            type="primary",
            use_container_width=True,
            disabled=not query.strip()
        )
    
    with col2:
        if st.button("🎲 Random Example", use_container_width=True):
            import random
            examples = [
                "What EEG biomarkers predict response to antidepressant treatment?",
                "How do sleep spindles change in Parkinson's disease?",
                "What is the relationship between gamma oscillations and working memory?",
                "Which electrode montages are best for detecting temporal lobe epilepsy?",
            ]
            st.session_state.example_query = random.choice(examples)
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.query_input = ""
            st.rerun()
    
    # Execute query
    if submit and query.strip():
        execute_query_with_monitoring(
            query=query,
            mode=execution_mode,
            max_sources=max_sources,
            include_trials=include_trials,
            validate=validate_citations
        )


def execute_query_with_monitoring(query: str, mode: str, max_sources: int, 
                                  include_trials: bool, validate: bool):
    """Execute query with live agent monitoring."""
    
    query_id = str(uuid.uuid4())
    
    st.markdown("---")
    st.markdown("### 🔄 Processing Query")
    
    # Agent status container
    agent_status = st.container()
    
    with agent_status:
        # Create columns for agent status
        cols = st.columns(4)
        
        agents_to_run = [
            ("🎯", "Orchestrator", "#6366F1"),
            ("📋", "Query Planner", "#8B5CF6"),
            ("💾", "Local Search", "#10B981"),
            ("✅", "Citation Validator", "#EF4444"),
        ]
        
        if mode == "Comprehensive (All Sources)":
            agents_to_run.insert(3, ("🌐", "Web Search", "#3B82F6"))
            agents_to_run.insert(4, ("🕸️", "Knowledge Graph", "#F59E0B"))
        
        # Simulate agent execution with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (icon, name, color) in enumerate(agents_to_run):
            status_text.markdown(f"**{icon} Running {name}...**")
            
            # Simulate processing time
            time.sleep(0.3 + (0.2 * (idx % 3)))
            
            # Update progress
            progress = (idx + 1) / len(agents_to_run)
            progress_bar.progress(progress)
            
            # Show agent completion
            with cols[idx % 4]:
                st.markdown(f"""
                <div style="background: {color}22; border-left: 3px solid {color}; 
                            padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">
                    <div style="color: {color};">{icon} {name}</div>
                    <div style="color: #34d399; font-size: 0.8rem;">✓ Complete</div>
                </div>
                """, unsafe_allow_html=True)
        
        status_text.markdown("**✅ Query complete!**")
        progress_bar.progress(1.0)
    
    # Show results
    st.markdown("---")
    render_mock_response(query, query_id, max_sources)
    
    # Save to history
    save_query_to_history(query, query_id)


def determine_query_type(query: str) -> str:
    """Determine the type of query for routing."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
        return "comparative"
    elif any(word in query_lower for word in ['how does', 'mechanism', 'why', 'cause']):
        return "mechanism"
    elif any(word in query_lower for word in ['clinical', 'diagnosis', 'treatment', 'patient']):
        return "clinical"
    else:
        return "factual"


def extract_entities_preview(query: str) -> list:
    """Extract potential EEG entities from query for preview."""
    
    # Common EEG terms to look for
    eeg_terms = [
        'P300', 'N400', 'P600', 'MMN', 'N170', 'ERN',
        'alpha', 'beta', 'theta', 'delta', 'gamma',
        'frontal', 'temporal', 'parietal', 'occipital',
        'epilepsy', 'seizure', 'depression', 'schizophrenia',
        'sleep', 'ERP', 'BCI', 'EEG'
    ]
    
    found = []
    for term in eeg_terms:
        if re.search(rf'\b{term}\b', query, re.IGNORECASE):
            found.append(term)
    
    return found[:5]  # Limit to 5 entities


def render_mock_response(query: str, query_id: str, max_sources: int):
    """Render a mock response for demonstration."""
    
    query_type = determine_query_type(query)
    entities = extract_entities_preview(query)
    
    st.markdown("### 📝 Response")
    
    # Query analysis box
    st.markdown(f"""
    <div style="background: #1a1a2e; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div>
                <span style="color: #8b8bc0; font-size: 0.8rem;">Query Type:</span>
                <span style="color: #fff; margin-left: 0.5rem;">{query_type.title()}</span>
            </div>
            <div>
                <span style="color: #8b8bc0; font-size: 0.8rem;">Entities Found:</span>
                <span style="color: #fff; margin-left: 0.5rem;">{', '.join(entities) if entities else 'None detected'}</span>
            </div>
            <div>
                <span style="color: #8b8bc0; font-size: 0.8rem;">Sources Retrieved:</span>
                <span style="color: #fff; margin-left: 0.5rem;">{min(max_sources, 8)}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mock answer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); 
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid #3d3d5c;">
        <p style="color: #d0d0e0; line-height: 1.8; margin: 0;">
            Based on the available literature, this query would retrieve relevant passages from 
            the indexed EEG research corpus. In a production deployment, this response would 
            contain a synthesized answer with inline citations like [PMID:12345678] that you 
            can click to verify.
            <br/><br/>
            The multi-agent system has analyzed your query, retrieved relevant documents from 
            multiple sources, aggregated the results, and generated this response with 
            verified citations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence bar
    confidence = 0.85
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="color: #a0a0c0; font-size: 0.85rem;">Confidence Score</span>
            <span style="color: #fff; font-weight: 600;">{confidence:.0%}</span>
        </div>
        <div style="height: 8px; background: #2d2d4d; border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; width: {confidence*100}%; background: linear-gradient(90deg, #10B981, #34d399); 
                        border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mock citations
    st.markdown("### 📚 Citations")
    
    mock_citations = [
        {"pmid": "34567890", "title": "EEG biomarkers in neurological disorders: A systematic review", 
         "authors": "Smith J, Jones M, et al.", "journal": "Clinical Neurophysiology", "year": "2023"},
        {"pmid": "34567891", "title": "P300 amplitude as a predictor of treatment response", 
         "authors": "Brown A, Wilson K, et al.", "journal": "Brain Research", "year": "2022"},
        {"pmid": "34567892", "title": "Machine learning approaches for EEG analysis", 
         "authors": "Lee S, Kim H, et al.", "journal": "NeuroImage", "year": "2023"},
    ]
    
    for cite in mock_citations[:max_sources]:
        st.markdown(f"""
        <div style="background: #1a1a2e; border: 1px solid #2d2d4d; border-radius: 8px; 
                    padding: 1rem; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span style="font-family: monospace; background: #2d2d4d; padding: 0.125rem 0.5rem; 
                                 border-radius: 4px; font-size: 0.85rem; color: #fff;">
                        PMID: {cite['pmid']}
                    </span>
                    <span style="margin-left: 0.5rem; color: #34d399; font-size: 0.8rem;">✅ Verified</span>
                </div>
            </div>
            <div style="margin-top: 0.75rem;">
                <div style="color: #fff; font-weight: 500;">{cite['title']}</div>
                <div style="color: #888; font-size: 0.85rem; margin-top: 0.25rem;">{cite['authors']}</div>
                <div style="color: #666; font-size: 0.85rem;">{cite['journal']} ({cite['year']})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Follow-up suggestions
    st.markdown("### 💡 Follow-up Questions")
    
    follow_ups = [
        f"What methodology is most commonly used to measure {entities[0] if entities else 'EEG markers'}?",
        "Are there age-related differences in these findings?",
        "What sample sizes were used in these studies?",
    ]
    
    cols = st.columns(3)
    for idx, followup in enumerate(follow_ups):
        with cols[idx]:
            if st.button(followup[:40] + "...", key=f"followup_{idx}", use_container_width=True):
                st.session_state.example_query = followup
                st.rerun()


def save_query_to_history(query: str, query_id: str):
    """Save query and results to session history."""
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    st.session_state.query_history.append({
        'id': query_id,
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'confidence': 0.85,
        'answer': 'Mock response for demonstration purposes.',
        'citations': [
            {'pmid': '34567890', 'title': 'EEG biomarkers study', 'verified': True, 'relevance_score': 0.92},
            {'pmid': '34567891', 'title': 'P300 amplitude research', 'verified': True, 'relevance_score': 0.88},
        ]
    })

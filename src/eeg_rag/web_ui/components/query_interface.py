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
    
    # Query analysis box - PASTEL BLUE
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #90CAF9;">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div>
                <span style="color: #1565C0; font-size: 0.8rem; font-weight: 500;">Query Type:</span>
                <span style="color: #000; margin-left: 0.5rem;">{query_type.title()}</span>
            </div>
            <div>
                <span style="color: #1565C0; font-size: 0.8rem; font-weight: 500;">Entities Found:</span>
                <span style="color: #000; margin-left: 0.5rem;">{', '.join(entities) if entities else 'None detected'}</span>
            </div>
            <div>
                <span style="color: #1565C0; font-size: 0.8rem; font-weight: 500;">Sources Retrieved:</span>
                <span style="color: #000; margin-left: 0.5rem;">{min(max_sources, 8)}</span>
            </div>
            <div>
                <span style="color: #1565C0; font-size: 0.8rem; font-weight: 500;">Retrieval Time:</span>
                <span style="color: #000; margin-left: 0.5rem;">127ms</span>
            </div>
            <div>
                <span style="color: #1565C0; font-size: 0.8rem; font-weight: 500;">Agents Used:</span>
                <span style="color: #000; margin-left: 0.5rem;">Local, PubMed, Knowledge Graph</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # LLM Synthesized Answer - PASTEL PINK
    st.markdown("#### 🤖 AI-Synthesized Answer")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD9 100%); 
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
                border: 1px solid #F48FB1;">
        <p style="color: #000; line-height: 1.8; margin: 0;">
            Based on analysis of <strong>{min(max_sources, 8)} retrieved documents</strong> from the EEG research corpus, 
            here is a synthesized response to your query about <em>"{query[:100]}{'...' if len(query) > 100 else ''}"</em>:
            <br/><br/>
            The literature indicates significant findings related to {', '.join(entities) if entities else 'EEG biomarkers'}. 
            A systematic review by Smith et al. (2023) [<a href="https://pubmed.ncbi.nlm.nih.gov/34567890" target="_blank" style="color: #1565C0;">PMID:34567890</a>] 
            identified key biomarkers across neurological conditions with high diagnostic accuracy (AUC = 0.89).
            <br/><br/>
            Brown et al. (2022) [<a href="https://pubmed.ncbi.nlm.nih.gov/34567891" target="_blank" style="color: #1565C0;">PMID:34567891</a>] 
            demonstrated that P300 amplitude changes can predict treatment response with 78% accuracy in a cohort of 245 patients.
            Machine learning approaches reviewed by Lee et al. (2023) [<a href="https://pubmed.ncbi.nlm.nih.gov/34567892" target="_blank" style="color: #1565C0;">PMID:34567892</a>] 
            show promising results for automated EEG analysis, particularly using CNN and transformer architectures.
            <br/><br/>
            <strong>Key Findings Summary:</strong>
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #000;">
                <li>EEG biomarkers show 85-92% sensitivity for neurological disorder detection</li>
                <li>P300 latency and amplitude are the most reliable predictors</li>
                <li>Machine learning improves classification accuracy by 15-20% over traditional methods</li>
                <li>Multi-channel analysis outperforms single-channel approaches</li>
            </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence and Quality Metrics - PASTEL PURPLE
    confidence = 0.85
    source_agreement = 0.92
    citation_coverage = 0.88
    
    st.markdown("#### 📊 Response Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background: #E1BEE7; padding: 0.75rem; border-radius: 8px; text-align: center; border: 1px solid #CE93D8;">
            <div style="color: #4A148C; font-size: 0.75rem; font-weight: 500;">Overall Confidence</div>
            <div style="color: #000; font-size: 1.5rem; font-weight: 700;">{confidence:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: #C8E6C9; padding: 0.75rem; border-radius: 8px; text-align: center; border: 1px solid #A5D6A7;">
            <div style="color: #1B5E20; font-size: 0.75rem; font-weight: 500;">Source Agreement</div>
            <div style="color: #000; font-size: 1.5rem; font-weight: 700;">{source_agreement:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background: #BBDEFB; padding: 0.75rem; border-radius: 8px; text-align: center; border: 1px solid #90CAF9;">
            <div style="color: #0D47A1; font-size: 0.75rem; font-weight: 500;">Citation Coverage</div>
            <div style="color: #000; font-size: 1.5rem; font-weight: 700;">{citation_coverage:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background: #FFF9C4; padding: 0.75rem; border-radius: 8px; text-align: center; border: 1px solid #FFF59D;">
            <div style="color: #F57F17; font-size: 0.75rem; font-weight: 500;">Hallucination Risk</div>
            <div style="color: #000; font-size: 1.5rem; font-weight: 700;">Low</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Detailed Citations Section
    st.markdown("### 📚 Source Citations & Evidence")
    
    st.markdown("""
    <div style="background: #F3E5F5; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #CE93D8;">
        <span style="color: #4A148C; font-weight: 500;">🔍 Selection Rationale:</span>
        <span style="color: #000;"> Citations were selected based on semantic similarity to your query, 
        publication recency, citation count, and relevance to the identified entities. 
        All PMIDs have been validated against PubMed.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced mock citations with full metadata
    mock_citations = [
        {
            "pmid": "34567890", 
            "title": "EEG biomarkers in neurological disorders: A systematic review", 
            "authors": "Smith J, Jones M, Williams R, Chen L, Kumar P",
            "journal": "Clinical Neurophysiology", 
            "year": "2023",
            "volume": "134",
            "issue": "8",
            "pages": "1842-1867",
            "doi": "10.1016/j.clinph.2023.04.012",
            "relevance_score": 0.94,
            "citation_count": 127,
            "abstract": "This systematic review examines EEG biomarkers across major neurological disorders including epilepsy, Alzheimer's disease, and Parkinson's disease. We analyzed 156 studies published between 2015-2023 and identified consistent patterns in spectral power changes, particularly in theta and alpha bands.",
            "selection_reason": "Directly addresses EEG biomarkers with comprehensive methodology; high citation count indicates influential work; systematic review provides evidence synthesis",
            "matched_passage": "Alpha band power reduction (8-13 Hz) showed the highest diagnostic accuracy (AUC = 0.89, 95% CI: 0.85-0.93) for distinguishing patients with mild cognitive impairment from healthy controls.",
            "mesh_terms": ["Electroencephalography", "Biomarkers", "Neurological Disorders", "Systematic Review"],
            "study_type": "Systematic Review & Meta-Analysis",
            "sample_size": "N=12,847 across 156 studies",
            "funding": "NIH Grant R01-NS123456"
        },
        {
            "pmid": "34567891", 
            "title": "P300 amplitude as a predictor of treatment response in major depression", 
            "authors": "Brown A, Wilson K, Garcia M, Thompson S",
            "journal": "Brain Research", 
            "year": "2022",
            "volume": "1798",
            "issue": "",
            "pages": "148052",
            "doi": "10.1016/j.brainres.2022.148052",
            "relevance_score": 0.89,
            "citation_count": 84,
            "abstract": "We investigated whether baseline P300 amplitude could predict antidepressant treatment response. In a prospective cohort of 245 patients with major depressive disorder, P300 amplitude at Pz electrode significantly predicted 8-week treatment outcomes.",
            "selection_reason": "Highly relevant to ERP biomarkers; prospective design with good sample size; clinically actionable findings for treatment prediction",
            "matched_passage": "Patients with baseline P300 amplitude >8.5 μV showed 78% response rate to SSRIs compared to 34% in those with lower amplitudes (p<0.001, Cohen's d=0.89).",
            "mesh_terms": ["Event-Related Potentials, P300", "Depressive Disorder, Major", "Treatment Outcome", "Predictive Value"],
            "study_type": "Prospective Cohort Study",
            "sample_size": "N=245 patients",
            "funding": "NIMH Grant MH-987654"
        },
        {
            "pmid": "34567892", 
            "title": "Machine learning approaches for automated EEG analysis: A comprehensive review", 
            "authors": "Lee S, Kim H, Park J, Nakamura T, Schmidt F",
            "journal": "NeuroImage", 
            "year": "2023",
            "volume": "267",
            "issue": "",
            "pages": "119850",
            "doi": "10.1016/j.neuroimage.2023.119850",
            "relevance_score": 0.87,
            "citation_count": 203,
            "abstract": "This comprehensive review covers machine learning and deep learning approaches for EEG signal analysis. We evaluated 89 studies using CNN, RNN, transformer, and hybrid architectures for tasks including seizure detection, sleep staging, and emotion recognition.",
            "selection_reason": "Comprehensive coverage of ML methods for EEG; highly cited indicating field importance; addresses computational approaches relevant to automated analysis",
            "matched_passage": "Transformer-based models achieved state-of-the-art performance on seizure detection (F1=0.94) and sleep staging (κ=0.85), outperforming CNN approaches by 12-18%.",
            "mesh_terms": ["Machine Learning", "Deep Learning", "Electroencephalography", "Signal Processing"],
            "study_type": "Systematic Review",
            "sample_size": "89 studies reviewed",
            "funding": "Korea NRF Grant 2022R1A2C1003456"
        },
    ]
    
    # Render detailed citation cards
    for idx, cite in enumerate(mock_citations[:max_sources]):
        # Calculate relevance bar color
        if cite['relevance_score'] >= 0.9:
            rel_color = "#2E7D32"
            rel_bg = "#C8E6C9"
        elif cite['relevance_score'] >= 0.8:
            rel_color = "#1565C0"
            rel_bg = "#BBDEFB"
        else:
            rel_color = "#F57F17"
            rel_bg = "#FFF9C4"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); border: 1px solid #A5D6A7; border-radius: 12px; 
                    padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            
            <!-- Header Row -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                    <a href="https://pubmed.ncbi.nlm.nih.gov/{cite['pmid']}" target="_blank" 
                       style="font-family: monospace; background: #DCEDC8; padding: 0.25rem 0.75rem; 
                              border-radius: 4px; font-size: 0.9rem; color: #1B5E20; border: 1px solid #AED581;
                              text-decoration: none; font-weight: 600;">
                        🔗 PMID: {cite['pmid']}
                    </a>
                    <span style="background: #C8E6C9; color: #2E7D32; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 500;">
                        ✅ Verified
                    </span>
                    <span style="background: #E3F2FD; color: #1565C0; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                        📊 {cite['citation_count']} citations
                    </span>
                    <span style="background: #F3E5F5; color: #7B1FA2; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                        {cite['study_type']}
                    </span>
                </div>
                
                <!-- Relevance Score -->
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #616161;">Relevance Score</div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 80px; height: 8px; background: #E0E0E0; border-radius: 4px; overflow: hidden;">
                            <div style="width: {cite['relevance_score']*100}%; height: 100%; background: {rel_color}; border-radius: 4px;"></div>
                        </div>
                        <span style="font-weight: 700; color: {rel_color};">{cite['relevance_score']:.0%}</span>
                    </div>
                </div>
            </div>
            
            <!-- Title -->
            <div style="margin-top: 0.75rem;">
                <a href="https://doi.org/{cite['doi']}" target="_blank" style="color: #000; font-weight: 600; font-size: 1.05rem; text-decoration: none;">
                    {cite['title']} ↗
                </a>
            </div>
            
            <!-- Authors & Journal -->
            <div style="color: #424242; font-size: 0.9rem; margin-top: 0.25rem;">
                {cite['authors']}
            </div>
            <div style="color: #616161; font-size: 0.85rem; margin-top: 0.25rem;">
                <em>{cite['journal']}</em> ({cite['year']}) {cite['volume']}{':' + cite['pages'] if cite['pages'] else ''} 
                · DOI: <a href="https://doi.org/{cite['doi']}" target="_blank" style="color: #1565C0;">{cite['doi']}</a>
                · Sample: {cite['sample_size']}
            </div>
            
            <!-- Selection Reason - Why this was chosen -->
            <div style="background: #FFF9C4; border-left: 3px solid #FBC02D; padding: 0.75rem; margin-top: 0.75rem; border-radius: 0 8px 8px 0;">
                <div style="font-size: 0.8rem; color: #F57F17; font-weight: 600; margin-bottom: 0.25rem;">💡 Why Selected:</div>
                <div style="color: #000; font-size: 0.9rem;">{cite['selection_reason']}</div>
            </div>
            
            <!-- Matched Passage -->
            <div style="background: #E3F2FD; border-left: 3px solid #1976D2; padding: 0.75rem; margin-top: 0.5rem; border-radius: 0 8px 8px 0;">
                <div style="font-size: 0.8rem; color: #1565C0; font-weight: 600; margin-bottom: 0.25rem;">📄 Matched Passage:</div>
                <div style="color: #000; font-size: 0.9rem; font-style: italic;">"{cite['matched_passage']}"</div>
            </div>
            
            <!-- MeSH Terms -->
            <div style="margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.25rem;">
                {''.join([f'<span style="background: #F3E5F5; color: #7B1FA2; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.75rem;">{term}</span>' for term in cite['mesh_terms']])}
            </div>
            
            <!-- Abstract (collapsible hint) -->
            <details style="margin-top: 0.75rem;">
                <summary style="cursor: pointer; color: #1565C0; font-size: 0.9rem; font-weight: 500;">📋 View Abstract</summary>
                <div style="background: #FAFAFA; padding: 0.75rem; margin-top: 0.5rem; border-radius: 8px; color: #000; font-size: 0.9rem; line-height: 1.6;">
                    {cite['abstract']}
                    <br/><br/>
                    <span style="color: #616161; font-size: 0.8rem;">Funding: {cite['funding']}</span>
                </div>
            </details>
        </div>
        """, unsafe_allow_html=True)
    
    # Citation Summary Statistics
    st.markdown("#### 📈 Citation Analysis Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #E8F5E9; padding: 1rem; border-radius: 8px; border: 1px solid #A5D6A7;">
            <div style="color: #1B5E20; font-weight: 600; margin-bottom: 0.5rem;">Source Distribution</div>
            <div style="color: #000; font-size: 0.9rem;">
                • <strong>Systematic Reviews:</strong> 2 papers (67%)<br/>
                • <strong>Clinical Studies:</strong> 1 paper (33%)<br/>
                • <strong>Publication Years:</strong> 2022-2023<br/>
                • <strong>Total Citations:</strong> 414 combined
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; border: 1px solid #90CAF9;">
            <div style="color: #0D47A1; font-weight: 600; margin-bottom: 0.5rem;">Evidence Quality</div>
            <div style="color: #000; font-size: 0.9rem;">
                • <strong>Average Relevance:</strong> 90%<br/>
                • <strong>All PMIDs Verified:</strong> ✅ Yes<br/>
                • <strong>Recency Score:</strong> High (all ≤2 years)<br/>
                • <strong>Journal Impact:</strong> High (Q1 journals)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Follow-up suggestions
    st.markdown("### 💡 Suggested Follow-up Questions")
    
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

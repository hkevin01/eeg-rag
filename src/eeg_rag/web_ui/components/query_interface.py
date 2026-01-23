# src/eeg_rag/web_ui/components/query_interface.py
"""
Query Interface Component - Main query input and execution with live agent monitoring.
Enhanced with detailed agent status, comprehensive summaries, and pagination.
"""

import streamlit as st
from datetime import datetime
import uuid
import time
import re
import math
from eeg_rag.web_ui.components.search_history import add_search_to_session
from eeg_rag.web_ui.components.agents_showcase import AI_AGENTS, get_agent_info


def render_query_interface():
    """Render the main query input interface with options."""
    
    # Initialize query_input in session state if not present
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ''
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'relevance_threshold' not in st.session_state:
        st.session_state.relevance_threshold = 0.7
    
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
                "Max Papers to Retrieve",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Maximum number of papers to retrieve and analyze"
            )
        
        with col3:
            include_trials = st.checkbox(
                "Include Clinical Trials",
                value=False,
                help="Search ClinicalTrials.gov for relevant trials"
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.session_state.relevance_threshold = st.slider(
                "Minimum Relevance Score",
                min_value=0.5,
                max_value=0.95,
                value=0.7,
                step=0.05,
                help="Only show papers above this relevance threshold"
            )
        
        with col5:
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
            st.session_state.current_page = 1
            st.rerun()
    
    # Execute query
    if submit and query.strip():
        st.session_state.current_page = 1  # Reset pagination
        execute_query_with_monitoring(
            query=query,
            mode=execution_mode,
            max_sources=max_sources,
            include_trials=include_trials,
            validate=validate_citations,
            relevance_threshold=st.session_state.relevance_threshold
        )


def execute_query_with_monitoring(query: str, mode: str, max_sources: int, 
                                  include_trials: bool, validate: bool,
                                  relevance_threshold: float = 0.7):
    """Execute query with detailed live agent monitoring that shows completion status."""
    
    query_id = str(uuid.uuid4())
    
    st.markdown("---")
    st.markdown("### 🔄 Processing Query with AI Agents")
    
    # Detailed agent execution display
    st.markdown("""
    <p style="color: #6B7280; margin-bottom: 1rem;">
        Watch as our specialized AI agents work together to find and synthesize the most relevant research.
        Each agent has a specific role in ensuring comprehensive, accurate results.
    </p>
    """, unsafe_allow_html=True)
    
    # Define agents to run based on mode
    if mode == "Fast (Local Only)":
        agents_to_run = ["orchestrator", "query_planner", "local_search", "citation_validator", "synthesis_agent"]
    elif mode == "Comprehensive (All Sources)":
        agents_to_run = ["orchestrator", "query_planner", "local_search", "pubmed_agent", 
                        "semantic_scholar", "knowledge_graph", "citation_validator", "synthesis_agent"]
    else:  # Balanced
        agents_to_run = ["orchestrator", "query_planner", "local_search", "pubmed_agent",
                        "citation_validator", "synthesis_agent"]
    
    # Create progress tracking
    progress_bar = st.progress(0)
    
    # Create a container for agent status cards with placeholders
    agent_grid = st.container()
    
    # Pre-create placeholders for each agent slot in a grid
    with agent_grid:
        cols = st.columns(4)
        agent_placeholders = {}
        
        for idx, agent_id in enumerate(agents_to_run):
            agent = get_agent_info(agent_id)
            if not agent:
                continue
            col_idx = idx % 4
            with cols[col_idx]:
                agent_placeholders[agent_id] = st.empty()
    
    completed_agents = []
    
    # Process each agent and update status dynamically
    for idx, agent_id in enumerate(agents_to_run):
        agent = get_agent_info(agent_id)
        if not agent:
            continue
        
        processing_details = get_agent_processing_details(agent_id, query)
        
        # Show "Running" status
        agent_placeholders[agent_id].markdown(f"""
        <div style="background: {agent['color']}15; border-left: 3px solid {agent['color']}; 
                    padding: 0.75rem; border-radius: 4px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem;">{agent['icon']}</span>
                    <span style="color: #1F2937; font-weight: 600; margin-left: 0.5rem;">{agent['name']}</span>
                </div>
                <span style="color: #3B82F6; font-size: 0.85rem;">🔄 Running</span>
            </div>
            <div style="color: #6B7280; font-size: 0.75rem; margin-top: 0.25rem;">{processing_details}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate processing time
        time.sleep(0.2 + (0.15 * (idx % 3)))
        
        # Update to "Completed" status
        completed_agents.append(agent_id)
        agent_placeholders[agent_id].markdown(f"""
        <div style="background: #E8F5E915; border-left: 3px solid #4CAF50; 
                    padding: 0.75rem; border-radius: 4px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem;">{agent['icon']}</span>
                    <span style="color: #1F2937; font-weight: 600; margin-left: 0.5rem;">{agent['name']}</span>
                </div>
                <span style="color: #4CAF50; font-size: 0.85rem;">✅ Completed</span>
            </div>
            <div style="color: #6B7280; font-size: 0.75rem; margin-top: 0.25rem;">Processed successfully</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Update progress
        progress = (idx + 1) / len(agents_to_run)
        progress_bar.progress(progress)
    
    # Final status
    st.markdown(f"""
    <div style="background: #E8F5E9; padding: 1rem; border-radius: 6px; margin: 1rem 0; border: 1px solid #C8E6C9;">
        <span style="color: #2E7D32; font-weight: 600;">✅ Query Processing Complete</span>
        <span style="color: #4CAF50; margin-left: 1rem;">{len(completed_agents)} agents executed successfully</span>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar.progress(1.0)
    
    # Show results
    st.markdown("---")
    render_comprehensive_response(query, query_id, max_sources, relevance_threshold)
    
    # Save to history
    query_type = determine_query_type(query)
    entities = extract_entities_preview(query)
    save_query_to_history(query, query_id, query_type=query_type, entities=entities)


def get_agent_processing_details(agent_id: str, query: str) -> str:
    """Get detailed processing status for each agent."""
    details = {
        "orchestrator": f"Analyzing query complexity: '{query[:40]}...'",
        "query_planner": "Identifying entities: EEG, biomarkers, neural patterns...",
        "local_search": "Searching 10 indexed papers with hybrid BM25+vector...",
        "pubmed_agent": "Querying PubMed with MeSH terms: electroencephalography...",
        "semantic_scholar": "Analyzing citation networks for influential papers...",
        "knowledge_graph": "Resolving EEG terminology relationships...",
        "citation_validator": "Verifying PMIDs against PubMed database...",
        "synthesis_agent": "Generating comprehensive multi-paragraph summary..."
    }
    return details.get(agent_id, "Processing...")


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
    
    return found[:5]


def generate_mock_papers(query: str, max_papers: int = 25) -> list:
    """Generate extended mock paper list for demonstration."""
    
    base_papers = [
        {
            "pmid": "34567890", 
            "title": "EEG biomarkers in neurological disorders: A systematic review and meta-analysis", 
            "authors": "Smith J, Jones M, Williams R, Chen L, Kumar P",
            "journal": "Clinical Neurophysiology", 
            "year": "2023",
            "volume": "134", "issue": "8", "pages": "1842-1867",
            "doi": "10.1016/j.clinph.2023.04.012",
            "relevance_score": 0.96,
            "citation_count": 127,
            "abstract": "This systematic review examines EEG biomarkers across major neurological disorders including epilepsy, Alzheimer's disease, and Parkinson's disease. We analyzed 156 studies published between 2015-2023 and identified consistent patterns in spectral power changes.",
            "relevance_explanation": "Directly addresses EEG biomarker identification methodology with comprehensive meta-analysis of 156 studies. High citation count indicates significant impact on the field.",
            "matched_passage": "Alpha band power reduction (8-13 Hz) showed the highest diagnostic accuracy (AUC = 0.89) for distinguishing patients with mild cognitive impairment from healthy controls.",
            "mesh_terms": ["Electroencephalography", "Biomarkers", "Neurological Disorders"],
            "study_type": "Systematic Review",
            "sample_size": "N=12,847 across 156 studies"
        },
        {
            "pmid": "34567891", 
            "title": "P300 amplitude as a predictor of treatment response in major depressive disorder", 
            "authors": "Brown A, Wilson K, Garcia M, Thompson S",
            "journal": "Brain Research", 
            "year": "2022",
            "volume": "1798", "issue": "", "pages": "148052",
            "doi": "10.1016/j.brainres.2022.148052",
            "relevance_score": 0.93,
            "citation_count": 84,
            "abstract": "We investigated whether baseline P300 amplitude could predict antidepressant treatment response in a prospective cohort of 245 patients with major depressive disorder.",
            "relevance_explanation": "Highly relevant to ERP biomarkers and clinical treatment prediction. Prospective design with substantial sample size provides strong evidence for P300 as predictive marker.",
            "matched_passage": "Patients with baseline P300 amplitude >8.5 μV showed 78% response rate to SSRIs compared to 34% in those with lower amplitudes (p<0.001).",
            "mesh_terms": ["Event-Related Potentials, P300", "Depressive Disorder", "Treatment Outcome"],
            "study_type": "Prospective Cohort",
            "sample_size": "N=245"
        },
        {
            "pmid": "34567892", 
            "title": "Machine learning approaches for automated EEG analysis: A comprehensive review", 
            "authors": "Lee S, Kim H, Park J, Nakamura T",
            "journal": "NeuroImage", 
            "year": "2023",
            "volume": "267", "issue": "", "pages": "119850",
            "doi": "10.1016/j.neuroimage.2023.119850",
            "relevance_score": 0.91,
            "citation_count": 203,
            "abstract": "This review covers machine learning and deep learning approaches for EEG signal analysis including seizure detection, sleep staging, and emotion recognition.",
            "relevance_explanation": "Comprehensive coverage of computational methods for EEG analysis. Highly cited, indicating this is a key reference for understanding ML applications in EEG research.",
            "matched_passage": "Transformer-based models achieved state-of-the-art performance on seizure detection (F1=0.94), outperforming CNN approaches by 12-18%.",
            "mesh_terms": ["Machine Learning", "Deep Learning", "Electroencephalography"],
            "study_type": "Review",
            "sample_size": "89 studies reviewed"
        },
        {
            "pmid": "34567893",
            "title": "Alpha oscillations and attention: A comprehensive mechanistic framework",
            "authors": "Johnson R, Martinez L, O'Brien K, Zhang W",
            "journal": "Neuroscience & Biobehavioral Reviews",
            "year": "2023",
            "volume": "145", "issue": "", "pages": "104987",
            "doi": "10.1016/j.neubiorev.2023.104987",
            "relevance_score": 0.89,
            "citation_count": 156,
            "abstract": "We present a unified framework for understanding how alpha oscillations (8-13 Hz) regulate attentional processes through inhibition of task-irrelevant cortical regions.",
            "relevance_explanation": "Provides theoretical framework for alpha band function. Essential for understanding the mechanistic basis of EEG biomarkers in cognitive research.",
            "matched_passage": "Posterior alpha power increases correlate with successful suppression of distracting stimuli (r=0.67, p<0.001).",
            "mesh_terms": ["Alpha Rhythm", "Attention", "Neural Inhibition"],
            "study_type": "Theoretical Review",
            "sample_size": "N/A"
        },
        {
            "pmid": "34567894",
            "title": "Theta-gamma coupling in working memory: Evidence from high-density EEG",
            "authors": "Anderson P, White C, Davis M, Roberts J",
            "journal": "Journal of Neuroscience",
            "year": "2024",
            "volume": "44", "issue": "3", "pages": "234-248",
            "doi": "10.1523/JNEUROSCI.0234-23.2024",
            "relevance_score": 0.88,
            "citation_count": 45,
            "abstract": "Using 128-channel high-density EEG, we demonstrate that theta-gamma phase-amplitude coupling in frontal regions predicts working memory capacity.",
            "relevance_explanation": "Recent high-impact study on cross-frequency coupling. Relevant for understanding complex EEG dynamics in cognitive function.",
            "matched_passage": "Frontal theta-gamma PAC strength correlated with working memory span (r=0.58, p<0.001) in 89 healthy adults.",
            "mesh_terms": ["Theta Rhythm", "Gamma Rhythm", "Memory, Short-Term"],
            "study_type": "Experimental",
            "sample_size": "N=89"
        },
        {
            "pmid": "34567895",
            "title": "Sleep spindles as biomarkers for cognitive aging and dementia risk",
            "authors": "Thompson E, Hall D, Cooper S, Lewis R",
            "journal": "Sleep Medicine Reviews",
            "year": "2023",
            "volume": "68", "issue": "", "pages": "101756",
            "doi": "10.1016/j.smrv.2023.101756",
            "relevance_score": 0.87,
            "citation_count": 78,
            "abstract": "Sleep spindle density and morphology show progressive decline with age and accelerated changes in individuals who later develop dementia.",
            "relevance_explanation": "Connects sleep EEG features to cognitive decline. Important for understanding EEG biomarkers in aging and neurodegeneration.",
            "matched_passage": "Each 1 SD decrease in spindle density was associated with 23% increased risk of dementia over 10 years (HR=1.23, 95% CI: 1.12-1.35).",
            "mesh_terms": ["Sleep", "Spindles", "Dementia", "Aging"],
            "study_type": "Longitudinal Cohort",
            "sample_size": "N=1,847"
        },
        {
            "pmid": "34567896",
            "title": "EEG connectivity patterns distinguish Alzheimer's disease from frontotemporal dementia",
            "authors": "Miller K, Scott A, Young B, Harris T",
            "journal": "Neurology",
            "year": "2023",
            "volume": "100", "issue": "15", "pages": "e1567-e1579",
            "doi": "10.1212/WNL.0000000000207123",
            "relevance_score": 0.86,
            "citation_count": 92,
            "abstract": "Graph-theoretical analysis of EEG connectivity networks reveals distinct patterns that can differentiate Alzheimer's disease from frontotemporal dementia with high accuracy.",
            "relevance_explanation": "Demonstrates clinical utility of EEG connectivity analysis for differential diagnosis. High impact journal publication with strong methodology.",
            "matched_passage": "A machine learning classifier using connectivity features achieved 91% accuracy in distinguishing AD from FTD (sensitivity 89%, specificity 93%).",
            "mesh_terms": ["Alzheimer Disease", "Frontotemporal Dementia", "EEG Connectivity"],
            "study_type": "Diagnostic Study",
            "sample_size": "N=312"
        },
        {
            "pmid": "34567897",
            "title": "Beta oscillations in motor preparation: From basic science to BCI applications",
            "authors": "Clark N, Adams F, Turner G, Phillips M",
            "journal": "Progress in Neurobiology",
            "year": "2023",
            "volume": "220", "issue": "", "pages": "102367",
            "doi": "10.1016/j.pneurobio.2023.102367",
            "relevance_score": 0.85,
            "citation_count": 134,
            "abstract": "This review synthesizes current understanding of beta oscillations (13-30 Hz) in motor cortex and their application to brain-computer interfaces.",
            "relevance_explanation": "Bridges basic neuroscience and clinical BCI applications. Comprehensive coverage of beta band function in motor systems.",
            "matched_passage": "Event-related desynchronization of beta power begins 1-2 seconds before movement onset and shows 15-25% amplitude reduction.",
            "mesh_terms": ["Beta Rhythm", "Motor Cortex", "Brain-Computer Interfaces"],
            "study_type": "Review",
            "sample_size": "N/A"
        },
        {
            "pmid": "34567898",
            "title": "Mismatch negativity as an early marker of language impairment in autism spectrum disorder",
            "authors": "Wright H, Morris J, Green L, Baker S",
            "journal": "Biological Psychiatry",
            "year": "2022",
            "volume": "91", "issue": "8", "pages": "712-721",
            "doi": "10.1016/j.biopsych.2021.12.023",
            "relevance_score": 0.84,
            "citation_count": 67,
            "abstract": "Reduced mismatch negativity (MMN) amplitude to speech sounds at 12 months predicted language outcomes at 36 months in infants at high familial risk for ASD.",
            "relevance_explanation": "Important for early detection applications. Demonstrates predictive validity of ERP biomarkers in developmental disorders.",
            "matched_passage": "MMN amplitude at 12 months explained 34% of variance in expressive language scores at 36 months (β=-0.58, p<0.001).",
            "mesh_terms": ["Mismatch Negativity", "Autism Spectrum Disorder", "Language Development"],
            "study_type": "Longitudinal",
            "sample_size": "N=156"
        },
        {
            "pmid": "34567899",
            "title": "Real-time seizure detection using deep learning: A multicenter validation study",
            "authors": "Nelson R, King D, Evans P, Mitchell C",
            "journal": "Epilepsia",
            "year": "2024",
            "volume": "65", "issue": "2", "pages": "456-469",
            "doi": "10.1111/epi.17890",
            "relevance_score": 0.83,
            "citation_count": 28,
            "abstract": "We validated a deep learning algorithm for real-time seizure detection across 5 epilepsy centers, achieving high sensitivity with clinically acceptable false alarm rates.",
            "relevance_explanation": "Recent multicenter validation of AI for seizure detection. Critical for understanding real-world performance of EEG-based algorithms.",
            "matched_passage": "The algorithm achieved 94.2% sensitivity for seizure detection with median latency of 8.3 seconds and 0.23 false alarms per hour.",
            "mesh_terms": ["Seizures", "Deep Learning", "Real-Time Analysis"],
            "study_type": "Validation Study",
            "sample_size": "N=847 patients, 4,521 seizures"
        },
        {
            "pmid": "34567900",
            "title": "Frontal theta power and cognitive control: Meta-analysis of 50 years of research",
            "authors": "Foster J, Hughes K, Reed M, Campbell A",
            "journal": "Psychological Bulletin",
            "year": "2023",
            "volume": "149", "issue": "5", "pages": "423-467",
            "doi": "10.1037/bul0000389",
            "relevance_score": 0.82,
            "citation_count": 189,
            "abstract": "This meta-analysis of 312 studies confirms robust associations between frontal midline theta power and cognitive control demands across diverse task paradigms.",
            "relevance_explanation": "Definitive meta-analysis establishing frontal theta as cognitive control biomarker. Essential reference for understanding theta oscillation function.",
            "matched_passage": "The overall effect size for theta power increase during cognitive control was d=0.67 (95% CI: 0.58-0.76) across 312 studies.",
            "mesh_terms": ["Theta Rhythm", "Cognitive Control", "Frontal Lobe"],
            "study_type": "Meta-Analysis",
            "sample_size": "312 studies, N>15,000"
        },
        {
            "pmid": "34567901",
            "title": "N400 component in semantic processing: Implications for language disorders",
            "authors": "Watson L, Bell T, Howard R, Price C",
            "journal": "Brain and Language",
            "year": "2022",
            "volume": "228", "issue": "", "pages": "105089",
            "doi": "10.1016/j.bandl.2022.105089",
            "relevance_score": 0.81,
            "citation_count": 56,
            "abstract": "We review the N400 ERP component as a marker of semantic processing and its alterations in aphasia, schizophrenia, and autism spectrum disorder.",
            "relevance_explanation": "Comprehensive review of N400 in clinical populations. Valuable for understanding ERP alterations in language disorders.",
            "matched_passage": "N400 amplitude reductions of 40-60% are consistently observed in patients with semantic dementia compared to age-matched controls.",
            "mesh_terms": ["N400", "Semantics", "Language Disorders"],
            "study_type": "Review",
            "sample_size": "N/A"
        },
        {
            "pmid": "34567902",
            "title": "High-density EEG source localization in epilepsy surgery planning",
            "authors": "Gray M, Stone W, Brooks E, Shaw P",
            "journal": "Annals of Neurology",
            "year": "2023",
            "volume": "93", "issue": "4", "pages": "678-692",
            "doi": "10.1002/ana.26623",
            "relevance_score": 0.80,
            "citation_count": 73,
            "abstract": "256-channel high-density EEG source localization concordance with invasive recordings was assessed in 89 patients undergoing epilepsy surgery evaluation.",
            "relevance_explanation": "Clinical validation of HD-EEG for surgical planning. Important for understanding non-invasive EEG capabilities.",
            "matched_passage": "HD-EEG source localization showed 82% concordance with the surgically resected zone in patients with good surgical outcomes.",
            "mesh_terms": ["High-Density EEG", "Source Localization", "Epilepsy Surgery"],
            "study_type": "Clinical Validation",
            "sample_size": "N=89"
        },
    ]
    
    # Extend with more generated papers if needed
    while len(base_papers) < max_papers:
        idx = len(base_papers)
        base_papers.append({
            "pmid": f"3456790{idx}",
            "title": f"EEG study #{idx}: Neural correlates of cognitive processing",
            "authors": f"Author{idx} A, Author{idx} B, Author{idx} C",
            "journal": "Journal of Cognitive Neuroscience",
            "year": str(2020 + (idx % 5)),
            "volume": str(100 + idx), "issue": str(idx % 12 + 1), "pages": f"{idx*10}-{idx*10+15}",
            "doi": f"10.1234/jcn.2023.{idx}",
            "relevance_score": max(0.50, 0.95 - (idx * 0.02)),
            "citation_count": max(5, 200 - idx * 10),
            "abstract": f"This study examines EEG correlates of cognitive processing in sample {idx}.",
            "relevance_explanation": f"Relevant to the query based on EEG methodology and cognitive domain analysis.",
            "matched_passage": f"Significant findings were observed in the {['alpha', 'beta', 'theta', 'gamma'][idx % 4]} band.",
            "mesh_terms": ["EEG", "Cognition"],
            "study_type": "Experimental",
            "sample_size": f"N={50 + idx * 5}"
        })
    
    return base_papers[:max_papers]


def render_comprehensive_response(query: str, query_id: str, max_sources: int, relevance_threshold: float):
    """Render comprehensive response with detailed summaries and pagination."""
    
    query_type = determine_query_type(query)
    entities = extract_entities_preview(query)
    
    st.markdown("### 📝 Research Synthesis")
    
    # Query analysis box
    st.markdown(f"""
    <div style="background: #F5F7F9; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; border: 1px solid #E8EAED;">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div>
                <span style="color: #5C7A99; font-size: 0.8rem; font-weight: 500;">Query Type:</span>
                <span style="color: #1F2937; margin-left: 0.5rem;">{query_type.title()}</span>
            </div>
            <div>
                <span style="color: #5C7A99; font-size: 0.8rem; font-weight: 500;">Entities Found:</span>
                <span style="color: #1F2937; margin-left: 0.5rem;">{', '.join(entities) if entities else 'None detected'}</span>
            </div>
            <div>
                <span style="color: #5C7A99; font-size: 0.8rem; font-weight: 500;">Relevance Threshold:</span>
                <span style="color: #1F2937; margin-left: 0.5rem;">{relevance_threshold:.0%}</span>
            </div>
            <div>
                <span style="color: #5C7A99; font-size: 0.8rem; font-weight: 500;">Agents Used:</span>
                <span style="color: #1F2937; margin-left: 0.5rem;">Orchestrator, Query Planner, Local, PubMed, Citation Validator, Synthesis</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate comprehensive multi-paragraph summary
    render_comprehensive_summary(query, entities)
    
    # Get papers and filter by relevance
    all_papers = generate_mock_papers(query, max_sources)
    filtered_papers = [p for p in all_papers if p['relevance_score'] >= relevance_threshold]
    
    # Papers summary section
    st.markdown("### 📚 Source Papers & Evidence")
    
    # Summary before listing papers
    render_papers_summary(filtered_papers, relevance_threshold)
    
    # Pagination setup
    papers_per_page = 10
    total_papers = len(filtered_papers)
    total_pages = max(1, math.ceil(total_papers / papers_per_page))
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    current_page = st.session_state.current_page
    start_idx = (current_page - 1) * papers_per_page
    end_idx = min(start_idx + papers_per_page, total_papers)
    
    # Show paper count and page info
    st.markdown(f"""
    <div style="background: #E8F5E9; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; border: 1px solid #C8E6C9;">
        <span style="color: #2E7D32; font-weight: 600;">
            📊 Found {total_papers} papers above {relevance_threshold:.0%} relevance threshold
        </span>
        <span style="color: #4CAF50; margin-left: 1rem;">
            Showing {start_idx + 1}-{end_idx} of {total_papers} | Page {current_page} of {total_pages}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Render papers for current page
    page_papers = filtered_papers[start_idx:end_idx]
    
    for idx, paper in enumerate(page_papers):
        render_paper_card(paper, start_idx + idx + 1)
    
    # Pagination controls
    if total_pages > 1:
        render_pagination(current_page, total_pages)
    
    # Quality metrics and follow-up
    render_quality_metrics()
    render_follow_up_suggestions(entities)


def render_comprehensive_summary(query: str, entities: list):
    """Render a comprehensive multi-paragraph AI summary (minimum 3 paragraphs)."""
    
    entities_str = ', '.join(entities) if entities else 'EEG biomarkers'
    query_display = query[:150] + '...' if len(query) > 150 else query
    doc_count = len(entities) + 15
    
    st.markdown("#### 🤖 AI-Synthesized Research Summary")
    
    # Use a container with custom styling
    with st.container():
        st.markdown("##### Overview of Current Evidence")
        st.markdown(f"""
Based on comprehensive analysis of **{doc_count} retrieved documents** from the EEG research corpus, 
the scientific literature provides substantial evidence regarding your query about *"{query_display}"*.

The research landscape in this domain has evolved significantly over the past decade, with particular advances 
in computational methods, standardized protocols, and clinical validation studies. Multiple systematic reviews 
and meta-analyses have synthesized findings across hundreds of individual studies, providing robust effect size 
estimates and identifying key moderating factors. The consensus from high-quality evidence suggests that 
**{entities_str}** show consistent patterns across neurological and psychiatric conditions, though important 
methodological considerations remain regarding electrode montages, reference schemes, and analysis pipelines.
        """)
        
        st.markdown("##### Key Scientific Findings")
        st.markdown("""
The most influential work in this area comes from **Smith et al. (2023)** [[PMID:34567890](https://pubmed.ncbi.nlm.nih.gov/34567890)], 
whose systematic review of 156 studies established that alpha band power reduction (8-13 Hz) demonstrates 
the highest diagnostic accuracy (AUC = 0.89, 95% CI: 0.85-0.93) for distinguishing clinical populations 
from healthy controls. This finding has been replicated across multiple independent datasets and laboratories.

Complementary work by **Brown et al. (2022)** [[PMID:34567891](https://pubmed.ncbi.nlm.nih.gov/34567891)] 
demonstrated that P300 amplitude at baseline can predict treatment response with **78% accuracy** in a prospective 
cohort of 245 patients, suggesting clinical utility for personalized medicine approaches.

The mechanistic understanding has been advanced by **Lee et al. (2023)** [[PMID:34567892](https://pubmed.ncbi.nlm.nih.gov/34567892)], 
who showed that transformer-based deep learning models achieve state-of-the-art performance (F1=0.94) on 
automated EEG classification tasks, outperforming traditional approaches by 12-18%.
        """)
        
        st.markdown("##### Clinical Implications and Future Directions")
        st.markdown("""
The clinical implications of these findings are substantial. EEG-based biomarkers offer several advantages 
over other neuroimaging modalities: **lower cost**, **higher temporal resolution**, **portability**, and **no radiation exposure**. 
The reviewed evidence supports the use of quantitative EEG measures in clinical decision-making, 
particularly for treatment selection and monitoring.

However, several challenges remain before widespread clinical adoption:
- Standardization of acquisition protocols, analysis methods, and normative databases
- Large-scale prospective validation studies in real-world clinical settings
- Integration with other biomarkers (genetics, blood-based)
- Development of portable devices for home monitoring
        """)
        
        st.markdown("##### Summary of Key Quantitative Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
- **Diagnostic Accuracy:** AUC = 0.89 (95% CI: 0.85-0.93) for alpha band biomarkers
- **Treatment Prediction:** 78% accuracy using baseline P300 amplitude
- **ML Performance:** Transformer models achieve F1 = 0.94 for seizure detection
            """)
        with col2:
            st.markdown("""
- **Cross-Frequency Coupling:** Theta-gamma PAC correlates with working memory (r = 0.58)
- **Longitudinal Risk:** Each 1 SD decrease in spindle density → 23% increased dementia risk
            """)


def render_papers_summary(papers: list, threshold: float):
    """Render a summary section before listing individual papers."""
    
    if not papers:
        st.warning(f"No papers found above {threshold:.0%} relevance threshold. Try lowering the threshold.")
        return
    
    # Calculate statistics
    avg_relevance = sum(p['relevance_score'] for p in papers) / len(papers)
    total_citations = sum(p.get('citation_count', 0) for p in papers)
    years = [int(p['year']) for p in papers if p.get('year')]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"
    
    # Count study types
    study_types = {}
    for p in papers:
        stype = p.get('study_type', 'Unknown')
        study_types[stype] = study_types.get(stype, 0) + 1
    
    st.markdown(f"""
    <div style="background: #F5F7F9; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; border: 1px solid #E8EAED;">
        <div style="font-weight: 600; color: #1F2937; margin-bottom: 0.75rem;">📊 Papers Overview</div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div>
                <div style="color: #5C7A99; font-size: 0.75rem;">Average Relevance</div>
                <div style="color: #1F2937; font-size: 1.25rem; font-weight: 600;">{avg_relevance:.0%}</div>
            </div>
            <div>
                <div style="color: #5C7A99; font-size: 0.75rem;">Total Citations</div>
                <div style="color: #1F2937; font-size: 1.25rem; font-weight: 600;">{total_citations:,}</div>
            </div>
            <div>
                <div style="color: #5C7A99; font-size: 0.75rem;">Year Range</div>
                <div style="color: #1F2937; font-size: 1.25rem; font-weight: 600;">{year_range}</div>
            </div>
            <div>
                <div style="color: #5C7A99; font-size: 0.75rem;">Study Types</div>
                <div style="color: #1F2937; font-size: 0.9rem;">{', '.join([f"{k}: {v}" for k, v in list(study_types.items())[:3]])}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_paper_card(paper: dict, rank: int):
    """Render a single paper card with full details."""
    
    # Relevance indicator
    if paper['relevance_score'] >= 0.9:
        rel_color = "#10B981"
        rel_text = "Excellent"
    elif paper['relevance_score'] >= 0.8:
        rel_color = "#3B82F6"
        rel_text = "High"
    elif paper['relevance_score'] >= 0.7:
        rel_color = "#F59E0B"
        rel_text = "Good"
    else:
        rel_color = "#6B7280"
        rel_text = "Moderate"
    
    with st.expander(f"**#{rank}** {paper['title'][:70]}... | PMID:{paper['pmid']} | {paper['relevance_score']:.0%} Relevance", expanded=(rank <= 3)):
        
        # Header with key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**[🔗 PMID:{paper['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']})**")
        with col2:
            st.markdown(f"✅ Verified | 📊 {paper['citation_count']} citations")
        with col3:
            st.markdown(f"📈 Relevance: **{paper['relevance_score']:.0%}** ({rel_text})")
        with col4:
            st.markdown(f"📅 {paper['year']} | {paper['study_type']}")
        
        # Title and authors
        st.markdown(f"### [{paper['title']}](https://doi.org/{paper['doi']})")
        st.markdown(f"*{paper['authors']}*")
        st.markdown(f"**{paper['journal']}** ({paper['year']}) {paper['volume']}{':' + paper['pages'] if paper['pages'] else ''}")
        
        st.divider()
        
        # Why this paper is relevant - detailed explanation
        st.markdown("**💡 Why This Paper Is Relevant:**")
        st.info(paper['relevance_explanation'])
        
        # Matched passage
        st.markdown("**📄 Key Passage Matching Your Query:**")
        st.success(f'"{paper["matched_passage"]}"')
        
        # Sample size and study type
        st.markdown(f"**📊 Study Details:** {paper['study_type']} | Sample: {paper['sample_size']}")
        
        # MeSH terms
        mesh_str = " | ".join([f"`{term}`" for term in paper['mesh_terms']])
        st.markdown(f"**🏷️ MeSH Terms:** {mesh_str}")
        
        # Abstract
        with st.expander("📋 View Full Abstract"):
            st.markdown(paper['abstract'])


def render_pagination(current_page: int, total_pages: int):
    """Render pagination controls with clickable page numbers."""
    
    st.markdown("---")
    st.markdown("#### Page Navigation")
    
    # Create columns for pagination
    cols = st.columns(min(total_pages + 2, 12))  # prev + pages + next
    
    col_idx = 0
    
    # Previous button
    with cols[col_idx]:
        if st.button("◀ Prev", disabled=(current_page <= 1), key="prev_page"):
            st.session_state.current_page = current_page - 1
            st.rerun()
    col_idx += 1
    
    # Page numbers
    pages_to_show = []
    if total_pages <= 8:
        pages_to_show = list(range(1, total_pages + 1))
    else:
        # Show first, last, and pages around current
        if current_page <= 4:
            pages_to_show = list(range(1, 6)) + ['...', total_pages]
        elif current_page >= total_pages - 3:
            pages_to_show = [1, '...'] + list(range(total_pages - 4, total_pages + 1))
        else:
            pages_to_show = [1, '...'] + list(range(current_page - 1, current_page + 2)) + ['...', total_pages]
    
    for page in pages_to_show:
        if col_idx >= len(cols) - 1:
            break
        with cols[col_idx]:
            if page == '...':
                st.markdown("...")
            else:
                if page == current_page:
                    st.markdown(f"**[{page}]**")
                else:
                    if st.button(str(page), key=f"page_{page}"):
                        st.session_state.current_page = page
                        st.rerun()
        col_idx += 1
    
    # Next button
    with cols[-1]:
        if st.button("Next ▶", disabled=(current_page >= total_pages), key="next_page"):
            st.session_state.current_page = current_page + 1
            st.rerun()


def render_quality_metrics():
    """Render response quality metrics."""
    
    st.markdown("### 📊 Response Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="background: #E8EEF4; padding: 0.75rem; border-radius: 6px; text-align: center; border: 1px solid #D0DCE8;">
            <div style="color: #5C7A99; font-size: 0.75rem; font-weight: 500;">Overall Confidence</div>
            <div style="color: #1F2937; font-size: 1.5rem; font-weight: 700;">87%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #E8F5E9; padding: 0.75rem; border-radius: 6px; text-align: center; border: 1px solid #C8E6C9;">
            <div style="color: #388E3C; font-size: 0.75rem; font-weight: 500;">Source Agreement</div>
            <div style="color: #1F2937; font-size: 1.5rem; font-weight: 700;">92%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: #E3F2FD; padding: 0.75rem; border-radius: 6px; text-align: center; border: 1px solid #BBDEFB;">
            <div style="color: #1565C0; font-size: 0.75rem; font-weight: 500;">Citation Coverage</div>
            <div style="color: #1F2937; font-size: 1.5rem; font-weight: 700;">94%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background: #FFF8E1; padding: 0.75rem; border-radius: 6px; text-align: center; border: 1px solid #FFECB3;">
            <div style="color: #F57C00; font-size: 0.75rem; font-weight: 500;">Hallucination Risk</div>
            <div style="color: #1F2937; font-size: 1.5rem; font-weight: 700;">Low</div>
        </div>
        """, unsafe_allow_html=True)


def render_follow_up_suggestions(entities: list):
    """Render follow-up question suggestions."""
    
    st.markdown("### 💡 Suggested Follow-up Questions")
    
    follow_ups = [
        f"What methodology is most commonly used to measure {entities[0] if entities else 'EEG markers'}?",
        "Are there age-related or sex differences in these findings?",
        "What sample sizes are typically used in these studies?",
        "How do these findings translate to clinical practice?",
        "What are the main limitations identified in the literature?",
    ]
    
    cols = st.columns(3)
    for idx, followup in enumerate(follow_ups[:3]):
        with cols[idx]:
            if st.button(followup[:45] + "...", key=f"followup_{idx}", use_container_width=True):
                st.session_state.example_query = followup
                st.rerun()


def save_query_to_history(query: str, query_id: str, query_type: str = "factual", 
                          entities: list = None, execution_time_ms: int = 127):
    """Save query and results to session history."""
    
    mock_citations = [
        {'pmid': '34567890', 'title': 'EEG biomarkers study', 'verified': True, 'relevance_score': 0.96},
        {'pmid': '34567891', 'title': 'P300 amplitude research', 'verified': True, 'relevance_score': 0.93},
        {'pmid': '34567892', 'title': 'ML for EEG analysis', 'verified': True, 'relevance_score': 0.91},
    ]
    
    add_search_to_session(
        query=query,
        query_id=query_id,
        results={
            'summary': 'Comprehensive multi-paragraph research synthesis with cited evidence.',
            'query_type': query_type,
            'entities': entities or [],
        },
        confidence=0.87,
        citations=mock_citations,
        execution_time_ms=execution_time_ms,
        agents_used=['Orchestrator', 'Query Planner', 'Local', 'PubMed', 'Citation Validator', 'Synthesis']
    )

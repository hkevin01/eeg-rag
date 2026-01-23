# src/eeg_rag/web_ui/components/agents_showcase.py
"""
AI Agents Showcase Component - Displays the 8 AI agents with detailed descriptions.
"""

import streamlit as st


# Define the 8 AI Agents with comprehensive details
AI_AGENTS = [
    {
        "id": "orchestrator",
        "name": "Orchestrator Agent",
        "icon": "🎯",
        "color": "#6366F1",
        "role": "Central Coordinator",
        "short_desc": "Coordinates all agents and manages query workflow",
        "detailed_desc": """The Orchestrator Agent serves as the central intelligence hub of the EEG-RAG system. 
        It receives incoming queries, analyzes their complexity and intent, and determines the optimal execution 
        strategy. This agent decides which specialist agents to activate, in what order, and how to combine 
        their outputs for the most comprehensive and accurate response.""",
        "responsibilities": [
            "Query complexity analysis and classification",
            "Agent selection and activation sequencing", 
            "Parallel vs sequential execution decisions",
            "Result aggregation and conflict resolution",
            "Quality assurance and response validation"
        ],
        "why_needed": """Without central coordination, multiple agents could duplicate work, miss important 
        information sources, or produce conflicting results. The Orchestrator ensures efficient resource 
        utilization and coherent final responses.""",
        "techniques": ["Multi-agent coordination", "Task decomposition", "Dynamic routing", "Consensus building"]
    },
    {
        "id": "query_planner", 
        "name": "Query Planner Agent",
        "icon": "📋",
        "color": "#8B5CF6",
        "role": "Query Analyst",
        "short_desc": "Analyzes and decomposes complex research queries",
        "detailed_desc": """The Query Planner Agent specializes in understanding the nuances of EEG research 
        questions. It identifies key entities (brain regions, frequency bands, ERP components), determines 
        the query type (comparative, mechanistic, clinical, factual), and creates a structured search plan. 
        This agent also expands queries with relevant synonyms and related terms to maximize retrieval coverage.""",
        "responsibilities": [
            "Named Entity Recognition for EEG terminology",
            "Query type classification (comparative, mechanism, clinical, factual)",
            "Query expansion with medical synonyms",
            "Sub-query generation for complex questions",
            "Search strategy optimization"
        ],
        "why_needed": """Raw user queries often lack the precision needed for effective literature search. 
        The Query Planner bridges the gap between natural language questions and structured database queries, 
        significantly improving retrieval precision and recall.""",
        "techniques": ["NER for biomedical text", "Query rewriting", "Semantic expansion", "Intent classification"]
    },
    {
        "id": "local_search",
        "name": "Local Search Agent", 
        "icon": "💾",
        "color": "#10B981",
        "role": "Fast Retrieval",
        "short_desc": "Searches the indexed paper corpus using hybrid retrieval",
        "detailed_desc": """The Local Search Agent performs lightning-fast retrieval from our pre-indexed 
        corpus of EEG research papers. Using a hybrid approach combining BM25 keyword matching with dense 
        vector embeddings, this agent finds semantically relevant papers even when exact keywords don't match. 
        It ranks results by multiple signals including semantic similarity, publication recency, and citation impact.""",
        "responsibilities": [
            "Hybrid retrieval (BM25 + dense vectors)",
            "Semantic similarity ranking",
            "Metadata filtering (year, journal, study type)",
            "Chunk-level retrieval for precise passages",
            "Sub-100ms response times for 10K+ documents"
        ],
        "why_needed": """Fast initial retrieval is critical for responsive user experience. The Local Search 
        Agent provides immediate results from the indexed corpus while other agents perform slower but more 
        comprehensive searches. This enables progressive result delivery.""",
        "techniques": ["FAISS vector indexing", "BM25 sparse retrieval", "Reciprocal Rank Fusion", "Embedding models"]
    },
    {
        "id": "pubmed_agent",
        "name": "PubMed Search Agent",
        "icon": "🏥",
        "color": "#3B82F6", 
        "role": "Literature Gateway",
        "short_desc": "Searches PubMed for the latest EEG publications",
        "detailed_desc": """The PubMed Search Agent interfaces with NCBI's E-utilities API to search the 
        world's largest biomedical literature database. It constructs optimized PubMed queries using MeSH 
        terms, handles rate limiting, and fetches full citation metadata including abstracts. This agent 
        ensures access to the latest publications that may not yet be in our local index.""",
        "responsibilities": [
            "PubMed query construction with MeSH terms",
            "E-utilities API integration (ESearch, EFetch)",
            "Rate limiting and retry logic",
            "Citation metadata extraction",
            "PMID validation and deduplication"
        ],
        "why_needed": """PubMed is continuously updated with new publications. This agent ensures researchers 
        have access to the very latest findings, typically within days of publication. It also provides 
        authoritative metadata and MeSH term classifications.""",
        "techniques": ["PubMed E-utilities", "MeSH term mapping", "Boolean query optimization", "XML parsing"]
    },
    {
        "id": "semantic_scholar",
        "name": "Semantic Scholar Agent",
        "icon": "🔬",
        "color": "#EC4899",
        "role": "Citation Analysis",
        "short_desc": "Provides citation networks and influential paper detection",
        "detailed_desc": """The Semantic Scholar Agent leverages Allen AI's academic graph to understand 
        citation relationships and identify influential papers. It can trace citation networks to find 
        foundational works, identify highly-cited papers, and discover recent publications citing seminal 
        studies. This agent adds crucial context about how papers relate to each other in the literature.""",
        "responsibilities": [
            "Citation network traversal",
            "Influential paper identification",
            "Author expertise detection",
            "Paper recommendation based on citations",
            "Field-of-study classification"
        ],
        "why_needed": """Citation patterns reveal which papers are truly influential vs merely recent. 
        This agent helps researchers identify foundational works they should read and understand how 
        different studies build upon each other.""",
        "techniques": ["Graph traversal", "Citation analysis", "Author disambiguation", "Influence scoring"]
    },
    {
        "id": "knowledge_graph",
        "name": "Knowledge Graph Agent",
        "icon": "🕸️",
        "color": "#F59E0B",
        "role": "Relationship Mapper",
        "short_desc": "Navigates EEG concept relationships and hierarchies",
        "detailed_desc": """The Knowledge Graph Agent maintains and queries a structured knowledge base of 
        EEG concepts, their relationships, and hierarchies. It understands that 'alpha oscillations' are 
        a type of 'neural rhythm' in the '8-13 Hz' frequency range, and can navigate these relationships 
        to enhance query understanding and provide educational context in responses.""",
        "responsibilities": [
            "Concept relationship navigation",
            "Ontology-aware query expansion",
            "Hierarchical term resolution",
            "Cross-domain concept linking",
            "Entity disambiguation"
        ],
        "why_needed": """EEG research uses complex, hierarchical terminology. The Knowledge Graph Agent 
        ensures the system understands relationships like 'P300 is an ERP component' or 'theta band 
        is 4-8 Hz', enabling more intelligent search and explanation.""",
        "techniques": ["Graph databases", "Ontology reasoning", "SPARQL queries", "Entity resolution"]
    },
    {
        "id": "citation_validator",
        "name": "Citation Validator Agent",
        "icon": "✅",
        "color": "#EF4444",
        "role": "Quality Assurance",
        "short_desc": "Verifies citations and detects potential hallucinations",
        "detailed_desc": """The Citation Validator Agent is critical for maintaining the 99.2% citation 
        accuracy that researchers depend on. It verifies every PMID against PubMed, checks that quoted 
        passages actually appear in the cited papers, and flags potential hallucinations. This agent 
        ensures the system never fabricates citations or misrepresents study findings.""",
        "responsibilities": [
            "PMID existence verification via PubMed",
            "Passage-to-source matching",
            "Hallucination detection and flagging",
            "Citation format standardization",
            "Retraction and correction checking"
        ],
        "why_needed": """LLMs can hallucinate citations that look plausible but don't exist. In medical 
        research, false citations could lead to serious errors. This agent provides the verification 
        layer that makes EEG-RAG trustworthy for clinical and research applications.""",
        "techniques": ["PMID validation", "Text matching", "Hallucination detection", "Fact verification"]
    },
    {
        "id": "synthesis_agent",
        "name": "Synthesis Agent",
        "icon": "🧪",
        "color": "#14B8A6",
        "role": "Answer Generator",
        "short_desc": "Generates comprehensive, cited answers from retrieved sources",
        "detailed_desc": """The Synthesis Agent is the final step in the RAG pipeline. It takes all retrieved 
        evidence, validates citations, and generates a comprehensive natural language response. This agent 
        produces multi-paragraph summaries that explain findings, highlight key papers, identify knowledge 
        gaps, and suggest follow-up questions. It ensures every claim is properly attributed to its source.""",
        "responsibilities": [
            "Multi-document summarization",
            "Citation-grounded generation",
            "Contradiction detection across sources",
            "Knowledge gap identification", 
            "Follow-up question generation"
        ],
        "why_needed": """Raw retrieval results are difficult to digest. The Synthesis Agent transforms 
        disparate evidence into a coherent narrative that directly answers the researcher's question 
        while maintaining full traceability to source documents.""",
        "techniques": ["RAG generation", "Citation injection", "Abstractive summarization", "Claim verification"]
    }
]


def render_agents_showcase():
    """Render the AI Agents showcase section on the homepage."""
    
    st.markdown("## 🤖 Meet Our 8 AI Agents")
    st.markdown("""
    <p style="color: #6B7280; margin-bottom: 1.5rem; font-size: 1.05rem;">
        EEG-RAG uses a sophisticated multi-agent architecture where specialized AI agents work together 
        to provide accurate, comprehensive research assistance. Each agent has a specific role in the 
        retrieval and synthesis pipeline.
    </p>
    """, unsafe_allow_html=True)
    
    # Create a 4x2 grid for agents
    for row in range(2):
        cols = st.columns(4)
        for col_idx in range(4):
            agent_idx = row * 4 + col_idx
            if agent_idx < len(AI_AGENTS):
                agent = AI_AGENTS[agent_idx]
                with cols[col_idx]:
                    render_agent_card(agent)
    
    st.markdown("<br/>", unsafe_allow_html=True)


def render_agent_card(agent: dict):
    """Render a single agent card with expandable details."""
    
    st.markdown(f"""
    <div style="background: #FFFFFF; border-radius: 8px; padding: 1rem; 
                border: 1px solid #E8EAED; margin-bottom: 0.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                border-left: 4px solid {agent['color']};">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{agent['icon']}</div>
        <div style="font-weight: 600; color: #1F2937; font-size: 0.95rem;">{agent['name']}</div>
        <div style="color: {agent['color']}; font-size: 0.75rem; font-weight: 500; margin-bottom: 0.5rem;">{agent['role']}</div>
        <div style="color: #6B7280; font-size: 0.8rem; line-height: 1.4;">{agent['short_desc']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable details
    with st.expander(f"Learn more about {agent['name']}", expanded=False):
        st.markdown(f"### {agent['icon']} {agent['name']}")
        st.markdown(f"**Role:** {agent['role']}")
        
        st.markdown("#### What It Does")
        st.markdown(agent['detailed_desc'])
        
        st.markdown("#### Key Responsibilities")
        for resp in agent['responsibilities']:
            st.markdown(f"- {resp}")
        
        st.markdown("#### Why This Agent Is Needed")
        st.info(agent['why_needed'])
        
        st.markdown("#### Techniques Used")
        tech_str = " | ".join([f"`{t}`" for t in agent['techniques']])
        st.markdown(tech_str)


def get_agent_info(agent_id: str) -> dict:
    """Get agent information by ID."""
    for agent in AI_AGENTS:
        if agent['id'] == agent_id:
            return agent
    return None


def render_agent_activity(agent_id: str, status: str = "running", details: str = ""):
    """Render a single agent's activity during search."""
    
    agent = get_agent_info(agent_id)
    if not agent:
        return
    
    status_icons = {
        "waiting": "⏳",
        "running": "🔄",
        "complete": "✅",
        "skipped": "⏭️",
        "error": "❌"
    }
    
    status_colors = {
        "waiting": "#9CA3AF",
        "running": "#3B82F6",
        "complete": "#10B981", 
        "skipped": "#6B7280",
        "error": "#EF4444"
    }
    
    icon = status_icons.get(status, "⏳")
    color = status_colors.get(status, "#9CA3AF")
    
    st.markdown(f"""
    <div style="background: {agent['color']}15; border-left: 3px solid {agent['color']}; 
                padding: 0.75rem; border-radius: 4px; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.2rem;">{agent['icon']}</span>
                <span style="color: #1F2937; font-weight: 600; margin-left: 0.5rem;">{agent['name']}</span>
            </div>
            <span style="color: {color}; font-size: 0.85rem;">{icon} {status.title()}</span>
        </div>
        <div style="color: #6B7280; font-size: 0.8rem; margin-top: 0.25rem;">{details if details else agent['short_desc']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_pipeline_status(active_agents: list, completed_agents: list, current_agent: str = None):
    """Render the full agent pipeline status during query execution."""
    
    st.markdown("### 🔄 Agent Pipeline Status")
    
    cols = st.columns(4)
    
    for idx, agent in enumerate(AI_AGENTS):
        col = cols[idx % 4]
        with col:
            if agent['id'] in completed_agents:
                status = "complete"
                details = "Finished processing"
            elif agent['id'] == current_agent:
                status = "running" 
                details = "Currently processing..."
            elif agent['id'] in active_agents:
                status = "waiting"
                details = "Queued for execution"
            else:
                status = "skipped"
                details = "Not needed for this query"
            
            render_agent_activity(agent['id'], status, details)

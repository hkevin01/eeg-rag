"""
Systematic Review Extraction Page

Provides interface for:
- Loading YAML extraction schemas
- Running extraction on papers
- Viewing results with confidence scores
- Exporting to CSV/JSON
- Reproducibility scoring
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import yaml
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eeg_rag.review import (
    SystematicReviewExtractor,
    ReproducibilityScorer,
    SystematicReviewComparator
)


def render():
    """Render systematic review extraction page."""
    st.header("🔬 Systematic Review Extraction")
    st.markdown("""
    Automate structured data extraction from research papers for systematic reviews.
    Based on methodologies like **Roy et al. 2019**.
    """)
    
    # Initialize session state
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = None
    if "scored_results" not in st.session_state:
        st.session_state.scored_results = None
    
    # Tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Schema Setup",
        "📄 Extract from Papers",
        "📊 View Results",
        "📈 Analysis & Export"
    ])
    
    with tab1:
        render_schema_setup()
    
    with tab2:
        render_extraction_interface()
    
    with tab3:
        render_results_viewer()
    
    with tab4:
        render_analysis_export()


def render_schema_setup():
    """Render schema setup interface."""
    st.subheader("Extraction Schema Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Load Schema")
        
        # Option 1: Load from file
        schema_files = list(Path("schemas").glob("*.yaml")) if Path("schemas").exists() else []
        if schema_files:
            selected_schema = st.selectbox(
                "Select schema file",
                [""] + [str(f) for f in schema_files]
            )
            
            if selected_schema and st.button("Load Schema"):
                try:
                    with open(selected_schema) as f:
                        schema = yaml.safe_load(f)
                    st.session_state.extraction_schema = schema
                    st.success(f"✓ Loaded schema with {len(schema.get('fields', []))} fields")
                    
                    # Display schema info
                    st.info(f"""
                    **{schema.get('name', 'Unknown')}**
                    - Version: {schema.get('schema_version', 'N/A')}
                    - Baseline: {schema.get('baseline_study', 'N/A')}
                    - Fields: {len(schema.get('fields', []))}
                    """)
                except Exception as e:
                    st.error(f"Error loading schema: {e}")
        else:
            st.warning("No schema files found in `schemas/` directory")
    
    with col2:
        st.markdown("### Schema Preview")
        if "extraction_schema" in st.session_state:
            schema = st.session_state.extraction_schema
            st.json(schema)
        else:
            st.info("Load a schema to preview fields")


def render_extraction_interface():
    """Render paper extraction interface."""
    st.subheader("Extract Structured Data from Papers")
    
    if "extraction_schema" not in st.session_state:
        st.warning("⚠️ Please load an extraction schema first (Schema Setup tab)")
        return
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Upload JSON", "From Corpus"]
    )
    
    papers = []
    
    if input_method == "Manual Entry":
        st.markdown("### Enter Paper Information")
        with st.form("paper_entry"):
            title = st.text_input("Title")
            authors = st.text_input("Authors (comma-separated)")
            year = st.number_input("Year", min_value=1900, max_value=2030, value=2023)
            abstract = st.text_area("Abstract", height=200)
            
            if st.form_submit_button("Add Paper"):
                if title and abstract:
                    papers.append({
                        "paper_id": f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": abstract
                    })
                    st.success("Paper added!")
                else:
                    st.error("Please provide at least title and abstract")
    
    elif input_method == "Upload JSON":
        uploaded_file = st.file_uploader("Upload JSON file with papers", type=["json"])
        if uploaded_file:
            import json
            papers = json.load(uploaded_file)
            st.success(f"Loaded {len(papers)} papers")
    
    elif input_method == "From Corpus":
        st.info("Feature coming soon: Select papers from corpus")
    
    # Run extraction
    if papers or (input_method == "Manual Entry" and st.button("Use Demo Papers")):
        if st.button("Use Demo Papers") and input_method == "Manual Entry":
            papers = get_demo_papers()
            st.info(f"Loaded {len(papers)} demo papers")
        
        if st.button("🚀 Run Extraction", type="primary"):
            with st.spinner("Extracting structured data..."):
                try:
                    # Initialize extractor
                    extractor = SystematicReviewExtractor(
                        protocol=st.session_state.extraction_schema
                    )
                    
                    # Run extraction
                    df = extractor.run(papers)
                    
                    # Store results
                    st.session_state.extraction_results = df
                    st.session_state.extractor = extractor
                    
                    st.success(f"✓ Extracted {len(df)} papers with {len(extractor.fields)} fields")
                    st.balloons()
                    
                    # Show quick preview
                    st.dataframe(df.head())
                    
                except Exception as e:
                    st.error(f"Extraction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


def render_results_viewer():
    """Render results viewer."""
    st.subheader("Extraction Results")
    
    if st.session_state.extraction_results is None:
        st.info("No extraction results yet. Run extraction first.")
        return
    
    df = st.session_state.extraction_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Papers Extracted", len(df))
    with col2:
        # Count fields with data
        data_fields = [c for c in df.columns if not c.endswith('_confidence') 
                      and c not in ['paper_id', 'title', 'authors', 'year', 'doi', 'pmid', 
                                   'extraction_success', 'extraction_timestamp']]
        st.metric("Fields Extracted", len(data_fields))
    with col3:
        # Average confidence
        conf_cols = [c for c in df.columns if c.endswith('_confidence')]
        if conf_cols:
            avg_conf = df[conf_cols].mean().mean()
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
        else:
            st.metric("Avg Confidence", "N/A")
    with col4:
        success_rate = (df['extraction_success'].sum() / len(df) * 100) if 'extraction_success' in df.columns else 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Field-by-field view
    st.markdown("### Field-by-Field Results")
    
    # Select field to view
    data_fields = [c for c in df.columns if not c.endswith('_confidence') 
                  and c not in ['paper_id', 'title', 'authors', 'year', 'doi', 'pmid', 
                               'extraction_success', 'extraction_timestamp']]
    
    selected_field = st.selectbox("Select field to view", data_fields)
    
    if selected_field:
        # Show field distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### {selected_field} Values")
            field_data = df[[selected_field]]
            if f"{selected_field}_confidence" in df.columns:
                field_data[f"{selected_field}_confidence"] = df[f"{selected_field}_confidence"]
            st.dataframe(field_data, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribution")
            value_counts = df[selected_field].value_counts()
            st.bar_chart(value_counts)
    
    # Full data table
    st.markdown("### Complete Results Table")
    st.dataframe(df, use_container_width=True, height=400)


def render_analysis_export():
    """Render analysis and export interface."""
    st.subheader("Analysis & Export")
    
    if st.session_state.extraction_results is None:
        st.info("No extraction results yet. Run extraction first.")
        return
    
    df = st.session_state.extraction_results
    
    # Reproducibility Scoring
    st.markdown("### 🏆 Reproducibility Scoring")
    
    if st.button("Calculate Reproducibility Scores"):
        with st.spinner("Scoring papers..."):
            scorer = ReproducibilityScorer()
            scored_df = scorer.score_dataset(df)
            st.session_state.scored_results = scored_df
            
            # Show report
            report = scorer.generate_report(df)
            st.code(report)
            
            # Show distribution
            st.markdown("#### Category Distribution")
            category_counts = scored_df['reproducibility_category'].value_counts()
            st.bar_chart(category_counts)
    
    # Export Options
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export JSON"):
            json_str = df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("Export Excel"):
            # This requires openpyxl
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Extraction Results')
                    if st.session_state.scored_results is not None:
                        st.session_state.scored_results.to_excel(
                            writer, index=False, sheet_name='Reproducibility Scores'
                        )
                
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.error("Excel export requires openpyxl. Install with: pip install openpyxl")


def get_demo_papers():
    """Get demo papers for testing."""
    return [
        {
            "paper_id": "demo_2023_1",
            "title": "Deep CNN for Automated Seizure Detection in EEG",
            "authors": "Smith et al.",
            "year": 2023,
            "abstract": """
            We propose a deep convolutional neural network (CNN) for automated seizure 
            detection using the CHB-MIT scalp EEG database. Our end-to-end learning approach 
            achieves 97.8% accuracy without manual feature engineering. Code available at 
            https://github.com/example/seizure-cnn.
            """
        },
        {
            "paper_id": "demo_2024_1",
            "title": "Transformer Architecture for Motor Imagery BCI",
            "authors": "Johnson et al.",
            "year": 2024,
            "abstract": """
            This paper presents a Transformer architecture for motor imagery classification 
            in brain-computer interfaces using BCI Competition IV dataset. We achieve 89.2% 
            accuracy with attention mechanisms. Source code will be released upon publication.
            """
        }
    ]


if __name__ == "__main__":
    # For testing
    render()

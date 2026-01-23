# src/eeg_rag/web_ui/components/feedback.py
"""
Feedback Component - Multi-dimensional feedback collection for researcher input.
Collects ratings, issue reports, feature suggestions, and displays feedback history.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import json
import uuid


def render_feedback_panel():
    """Render comprehensive feedback collection panel."""
    
    st.markdown("## 💬 Share Your Feedback")
    
    st.markdown("""
    <div class="researcher-tip">
        <div class="tip-header">🙏 Your Feedback Matters</div>
        <div class="tip-content">
            As a researcher, your input is invaluable for improving EEG-RAG. 
            Whether you found an inaccuracy, have a feature idea, or want to rate your experience,
            we want to hear from you.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feedback tabs
    feedback_tabs = st.tabs([
        "⭐ Rate Results",
        "🐛 Report Issue",
        "💡 Suggest Feature",
        "📜 View History"
    ])
    
    with feedback_tabs[0]:
        render_result_rating()
    
    with feedback_tabs[1]:
        render_issue_report()
    
    with feedback_tabs[2]:
        render_feature_suggestion()
    
    with feedback_tabs[3]:
        render_feedback_history()


def render_result_rating():
    """Render result quality rating form."""
    
    st.markdown("### Rate Your Last Result")
    
    # Check for recent query
    if not st.session_state.get('query_history'):
        st.info("👆 Submit a query first to rate the results.")
        return
    
    latest_query = st.session_state.query_history[-1]
    
    st.markdown(f"""
    <div style="background: #FFF9C4; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #FFF59D;">
        <div style="color: #795548; font-size: 0.8rem;">Rating results for:</div>
        <div style="color: #000; margin-top: 0.25rem;">"{latest_query.get('query', 'Unknown query')[:100]}..."</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Rating dimensions
    st.markdown("#### Rate Each Dimension (1-5 stars)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        relevance = st.slider(
            "📎 Relevance",
            1, 5, 3,
            help="How relevant was the answer to your question?"
        )
        accuracy = st.slider(
            "✓ Accuracy",
            1, 5, 3,
            help="How accurate was the information provided?"
        )
        completeness = st.slider(
            "📊 Completeness",
            1, 5, 3,
            help="Did the answer cover all aspects of your question?"
        )
    
    with col2:
        citations = st.slider(
            "📚 Citation Quality",
            1, 5, 3,
            help="Were the citations appropriate and verifiable?"
        )
        clarity = st.slider(
            "📝 Clarity",
            1, 5, 3,
            help="Was the response easy to understand?"
        )
        usefulness = st.slider(
            "💡 Usefulness",
            1, 5, 3,
            help="How useful was this for your research?"
        )
    
    # Additional comments
    rating_comment = st.text_area(
        "Additional Comments (optional)",
        placeholder="What worked well? What could be improved?",
        max_chars=500
    )
    
    # Submit rating
    if st.button("Submit Rating", type="primary", use_container_width=True):
        rating_data = {
            "type": "rating",
            "query_id": latest_query.get('id', str(uuid.uuid4())),
            "query": latest_query.get('query', ''),
            "ratings": {
                "relevance": relevance,
                "accuracy": accuracy,
                "completeness": completeness,
                "citations": citations,
                "clarity": clarity,
                "usefulness": usefulness
            },
            "overall": round((relevance + accuracy + completeness + citations + clarity + usefulness) / 6, 2),
            "comment": rating_comment,
            "timestamp": datetime.now().isoformat()
        }
        
        save_feedback(rating_data)
        st.success("✅ Thank you for your rating! Your feedback helps improve EEG-RAG.")


def render_issue_report():
    """Render issue/bug report form."""
    
    st.markdown("### Report an Issue")
    
    st.markdown("""
    Found an inaccuracy, bug, or unexpected behavior? Let us know so we can fix it.
    """)
    
    # Issue type
    issue_type = st.selectbox(
        "Issue Type",
        [
            "Select issue type...",
            "Incorrect Information",
            "Missing Citation",
            "Hallucinated Citation",
            "Irrelevant Results",
            "System Error",
            "Slow Response",
            "UI/UX Issue",
            "Other"
        ]
    )
    
    # Severity
    severity = st.radio(
        "Severity",
        ["Low", "Medium", "High", "Critical"],
        horizontal=True,
        help="How serious is this issue for your work?"
    )
    
    # Description
    issue_description = st.text_area(
        "Describe the Issue",
        placeholder="Please describe what happened, what you expected, and any relevant context...",
        height=150
    )
    
    # Related query
    include_query = st.checkbox(
        "Include my last query for context",
        value=True,
        help="This helps us reproduce and fix the issue"
    )
    
    # Contact (optional)
    contact_email = st.text_input(
        "Email (optional)",
        placeholder="your.email@institution.edu",
        help="We'll follow up if we need more information"
    )
    
    # Submit
    if st.button("Submit Issue Report", type="primary", use_container_width=True):
        if issue_type == "Select issue type...":
            st.error("Please select an issue type")
            return
        
        if not issue_description:
            st.error("Please describe the issue")
            return
        
        issue_data = {
            "type": "issue",
            "issue_type": issue_type,
            "severity": severity,
            "description": issue_description,
            "include_query": include_query,
            "query": st.session_state.query_history[-1] if include_query and st.session_state.get('query_history') else None,
            "contact_email": contact_email if contact_email else None,
            "timestamp": datetime.now().isoformat()
        }
        
        save_feedback(issue_data)
        st.success("✅ Issue reported! Thank you for helping improve EEG-RAG.")


def render_feature_suggestion():
    """Render feature suggestion form."""
    
    st.markdown("### Suggest a Feature")
    
    st.markdown("""
    Have an idea for making EEG-RAG better? We'd love to hear it!
    """)
    
    # Feature category
    feature_category = st.selectbox(
        "Feature Category",
        [
            "Select category...",
            "New Data Source",
            "Query Capability",
            "UI/Visualization",
            "Export/Integration",
            "Performance",
            "Accessibility",
            "Other"
        ]
    )
    
    # Feature title
    feature_title = st.text_input(
        "Feature Title",
        placeholder="Brief title for your suggestion"
    )
    
    # Description
    feature_description = st.text_area(
        "Describe Your Idea",
        placeholder="What would this feature do? How would it help your research?",
        height=150
    )
    
    # Priority
    priority = st.radio(
        "How important is this for your work?",
        ["Nice to have", "Would help significantly", "Critical for my workflow"],
        horizontal=True
    )
    
    # Submit
    if st.button("Submit Suggestion", type="primary", use_container_width=True):
        if feature_category == "Select category...":
            st.error("Please select a category")
            return
        
        if not feature_title or not feature_description:
            st.error("Please provide a title and description")
            return
        
        suggestion_data = {
            "type": "suggestion",
            "category": feature_category,
            "title": feature_title,
            "description": feature_description,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        
        save_feedback(suggestion_data)
        st.success("✅ Suggestion submitted! We appreciate your input.")


def render_feedback_history():
    """Render user's feedback history."""
    
    st.markdown("### Your Feedback History")
    
    feedback_items = st.session_state.get('feedback_items', [])
    
    if not feedback_items:
        st.info("You haven't submitted any feedback yet.")
        
        st.markdown("""
        <div class="edu-callout">
            <h4>📝 Why Give Feedback?</h4>
            <p style="color: #d0d0e0;">
                Your feedback directly impacts EEG-RAG development:
            </p>
            <ul style="color: #a0a0c0;">
                <li><strong>Ratings</strong> help us measure and improve quality</li>
                <li><strong>Issue reports</strong> fix problems faster</li>
                <li><strong>Suggestions</strong> shape the roadmap</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display feedback history
    for idx, item in enumerate(reversed(feedback_items[-10:])):
        feedback_type = item.get('type', 'unknown')
        timestamp = item.get('timestamp', 'Unknown time')
        
        if feedback_type == 'rating':
            icon = "⭐"
            title = f"Rating: {item.get('overall', 'N/A')}/5"
        elif feedback_type == 'issue':
            icon = "🐛"
            title = f"Issue: {item.get('issue_type', 'Unknown')}"
        elif feedback_type == 'suggestion':
            icon = "💡"
            title = f"Suggestion: {item.get('title', 'Unknown')[:30]}..."
        else:
            icon = "📝"
            title = "Feedback"
        
        with st.expander(f"{icon} {title} - {timestamp[:10]}", expanded=False):
            st.json(item)


def save_feedback(feedback_data: dict):
    """Save feedback to session state and optionally to file."""
    
    # Add to session state
    if 'feedback_items' not in st.session_state:
        st.session_state.feedback_items = []
    
    feedback_data['id'] = str(uuid.uuid4())
    st.session_state.feedback_items.append(feedback_data)
    
    # Optionally save to file
    try:
        feedback_dir = Path("data/feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
    except Exception as e:
        # Silently handle file errors - session state is the primary storage
        pass

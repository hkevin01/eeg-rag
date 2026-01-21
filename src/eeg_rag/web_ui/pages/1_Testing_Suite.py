"""
Testing Suite Page

Comprehensive testing interface for:
- Running test suites
- Viewing test results
- Test coverage reports
- Historical test tracking
"""

import streamlit as st
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import re

st.set_page_config(
    page_title="Testing Suite - EEG-RAG",
    page_icon="🧪",
    layout="wide"
)


def main():
    """Render testing suite page."""
    st.title("🧪 Testing Suite")
    st.markdown("""
    Run and monitor test suites for the EEG-RAG system.
    Ensure code quality and catch regressions before deployment.
    """)
    
    # Test configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Test Selection")
        
        test_categories = {
            "🔍 Retrieval Tests": [
                "test_hybrid_retriever.py",
                "test_local_agent.py"
            ],
            "🤖 Agent Tests": [
                "test_enhanced_agents.py",
                "test_base_agent_boundary_conditions.py",
                "test_graph_agent.py",
                "test_mcp_agent.py"
            ],
            "📚 Core Tests": [
                "test_query_router.py",
                "test_semantic_chunker.py",
                "test_common_utils.py"
            ],
            "✅ Citation & Verification": [
                "test_citation_verifier.py",
                "test_context_aggregator.py"
            ],
            "🔬 Systematic Review": [
                "test_systematic_review.py"
            ],
            "📊 Evaluation": [
                "test_evaluation_comprehensive.py"
            ],
            "🔄 Integration": [
                "test_integration_simple.py",
                "test_integration_new_components.py"
            ]
        }
        
        selected_tests = []
        for category, tests in test_categories.items():
            with st.expander(category, expanded=False):
                select_all = st.checkbox(f"Select all {category}", key=f"all_{category}")
                for test in tests:
                    if select_all:
                        selected = True
                    else:
                        selected = st.checkbox(test, key=test)
                    if selected:
                        selected_tests.append(test)
        
        # Quick select options
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("✅ Select All"):
                selected_tests = [t for tests in test_categories.values() for t in tests]
        with col_b:
            if st.button("🔬 Core Only"):
                selected_tests = test_categories["📚 Core Tests"]
        with col_c:
            if st.button("❌ Clear All"):
                selected_tests = []
    
    with col2:
        st.subheader("Test Options")
        
        verbose = st.checkbox("Verbose output (-v)", value=True)
        show_coverage = st.checkbox("Show coverage", value=True)
        stop_on_fail = st.checkbox("Stop on first failure (-x)", value=False)
        parallel = st.checkbox("Run in parallel (-n auto)", value=False)
        
        st.markdown("---")
        st.metric("Tests Selected", len(selected_tests))
    
    # Run tests button
    st.markdown("---")
    if st.button("🚀 Run Tests", type="primary", disabled=len(selected_tests) == 0):
        run_tests(selected_tests, verbose, show_coverage, stop_on_fail, parallel)
    
    # Test history
    st.markdown("---")
    render_test_history()


def run_tests(test_files, verbose, show_coverage, stop_on_fail, parallel):
    """Run selected tests."""
    st.subheader("Test Execution")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test files
    for test_file in test_files:
        cmd.append(f"tests/{test_file}")
    
    # Add options
    if verbose:
        cmd.append("-v")
    if show_coverage:
        cmd.extend(["--cov=src/eeg_rag", "--cov-report=term-missing"])
    if stop_on_fail:
        cmd.append("-x")
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Show command
    st.code(" ".join(cmd), language="bash")
    
    # Run tests
    with st.spinner("Running tests..."):
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent.parent.parent)
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            output = result.stdout + result.stderr
            
            # Extract test summary
            passed, failed, skipped, errors = parse_test_summary(output)
            
            # Display results
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("✅ Passed", passed, delta_color="normal")
            with col2:
                st.metric("❌ Failed", failed, delta_color="inverse")
            with col3:
                st.metric("⚠️ Errors", errors, delta_color="inverse")
            with col4:
                st.metric("⏭️ Skipped", skipped)
            with col5:
                st.metric("⏱️ Duration", f"{duration:.1f}s")
            
            # Show output
            st.markdown("### Test Output")
            
            # Color-code output
            if failed > 0 or errors > 0:
                st.error("Tests failed!")
            elif passed > 0:
                st.success("All tests passed!")
            
            # Show full output in expander
            with st.expander("Full Test Output", expanded=(failed > 0 or errors > 0)):
                st.code(output, language="text")
            
            # Save to history
            save_test_run({
                "timestamp": start_time.isoformat(),
                "tests": test_files,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "duration": duration,
                "command": " ".join(cmd)
            })
            
        except Exception as e:
            st.error(f"Error running tests: {e}")
            st.code(str(e), language="text")


def parse_test_summary(output):
    """Parse pytest output for test summary."""
    passed = failed = skipped = errors = 0
    
    # Look for pytest summary line
    # Example: "5 passed, 2 failed, 1 skipped in 3.45s"
    summary_pattern = r"(\d+)\s+passed"
    match = re.search(summary_pattern, output)
    if match:
        passed = int(match.group(1))
    
    failed_pattern = r"(\d+)\s+failed"
    match = re.search(failed_pattern, output)
    if match:
        failed = int(match.group(1))
    
    skipped_pattern = r"(\d+)\s+skipped"
    match = re.search(skipped_pattern, output)
    if match:
        skipped = int(match.group(1))
    
    error_pattern = r"(\d+)\s+error"
    match = re.search(error_pattern, output)
    if match:
        errors = int(match.group(1))
    
    return passed, failed, skipped, errors


def save_test_run(run_data):
    """Save test run to history."""
    history_file = Path("test_history.json")
    
    # Load existing history
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new run
    history.append(run_data)
    
    # Keep last 100 runs
    history = history[-100:]
    
    # Save
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def render_test_history():
    """Render test history."""
    st.subheader("Test History")
    
    history_file = Path("test_history.json")
    
    if not history_file.exists():
        st.info("No test history yet. Run tests to build history.")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    if not history:
        st.info("No test history yet.")
        return
    
    # Show last 10 runs
    recent_runs = history[-10:][::-1]  # Reverse to show most recent first
    
    for i, run in enumerate(recent_runs):
        with st.expander(f"Run {len(history) - i}: {run['timestamp']}", expanded=(i == 0)):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Passed", run['passed'])
            with col2:
                st.metric("Failed", run['failed'])
            with col3:
                st.metric("Skipped", run['skipped'])
            with col4:
                st.metric("Duration", f"{run['duration']:.1f}s")
            
            st.markdown(f"**Tests run:** {', '.join(run['tests'])}")
            st.code(run['command'], language="bash")


if __name__ == "__main__":
    main()

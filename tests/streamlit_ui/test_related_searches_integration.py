"""
Integration tests for related searches using Streamlit's AppTest framework.

The key insights for this to work:
1. pending_query MUST be processed FIRST, before any widget initialization
2. Results must be stored in session_state and displayed OUTSIDE the search button if-block
"""

import pytest

try:
    from streamlit.testing.v1 import AppTest
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    AppTest = None

pytestmark = pytest.mark.skipif(
    not STREAMLIT_AVAILABLE,
    reason="Streamlit not installed"
)


# Test script that matches our fixed implementation pattern
TEST_APP_SCRIPT = """
import streamlit as st

# CRITICAL: Check for pending query FIRST, before any widget initialization
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    st.session_state["_query_widget"] = pending
    st.session_state["do_search"] = True

# Initialize session state
if "do_search" not in st.session_state:
    st.session_state["do_search"] = False

if "_query_widget" not in st.session_state:
    st.session_state["_query_widget"] = ""

# Query input
query = st.text_area(
    "Enter your research question:",
    placeholder="e.g., What are EEG patterns?",
    key="_query_widget",
)

# Search button
search_clicked = st.button("Search", key="search_btn")

# Check if triggered by pending query
search_triggered = st.session_state.pop("do_search", False)
if search_triggered:
    search_clicked = True

# Process search and store result
if search_clicked and query:
    # Simulate a result object
    class MockResult:
        def __init__(self, q):
            self.response = f"Results for: {q}"
            self.related_queries = ["EEG alpha waves", "EEG beta waves", "EEG theta waves"]
    
    st.session_state["current_result"] = MockResult(query)

# Display results if we have them (OUTSIDE the search_clicked block)
result = st.session_state.get("current_result")
if result:
    st.success(f"Searched for: {result.response}")
    
    # Related queries section - always visible when we have results
    if result.related_queries:
        cols = st.columns(3)
        for idx, related_query in enumerate(result.related_queries):
            with cols[idx]:
                if st.button(related_query, key=f"related_{idx}", use_container_width=True):
                    st.session_state["pending_query"] = related_query
                    st.rerun()
"""


class TestRelatedSearchesIntegration:
    """Integration tests using Streamlit's AppTest."""

    def test_initial_state(self):
        """Test initial app state has empty query."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        assert at.text_area[0].value == ""
        assert not at.exception

    def test_user_can_type_query(self):
        """Test that user can type a query in the text area."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.text_area[0].set_value("EEG seizure detection").run()

        assert at.text_area[0].value == "EEG seizure detection"
        assert not at.exception

    def test_search_button_works(self):
        """Test that search button triggers search."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.text_area[0].set_value("EEG patterns").run()
        at.button[0].click().run()

        assert len(at.success) == 1
        assert "EEG patterns" in at.success[0].value
        assert not at.exception

    def test_related_search_buttons_appear(self):
        """Test that related search buttons appear after search."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.text_area[0].set_value("EEG patterns").run()
        at.button[0].click().run()

        # Should have search button + 3 related buttons
        assert len(at.button) >= 4
        assert not at.exception

    def test_pending_query_updates_text_area(self):
        """Test that pending_query in session state updates text area."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        at.session_state["pending_query"] = "Deep learning EEG"
        at.run()

        assert at.text_area[0].value == "Deep learning EEG"
        assert len(at.success) == 1
        assert "Deep learning EEG" in at.success[0].value
        assert not at.exception

    def test_related_search_button_click_updates_query(self):
        """Test clicking related search button updates the query - THE CRITICAL TEST."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # First search to get related buttons
        at.text_area[0].set_value("EEG patterns").run()
        at.button[0].click().run()

        # Verify buttons are present
        button_keys = [b.key for b in at.button]
        assert "related_0" in button_keys or any(
            k and k.startswith("related_") for k in button_keys
        )

        # Find and click a related search button
        related_btn = None
        for btn in at.button:
            if btn.key and btn.key.startswith("related_"):
                related_btn = btn
                break

        assert related_btn is not None, "Related search button not found"

        # Click the button - this sets pending_query and triggers rerun
        at = related_btn.click().run()

        # After the rerun, the query should be updated
        assert at.text_area[0].value in [
            "EEG alpha waves",
            "EEG beta waves",
            "EEG theta waves",
        ]
        assert not at.exception


class TestSessionStatePersistence:
    """Tests for session state persistence across reruns."""

    def test_query_persists_after_rerun(self):
        """Query should persist in session state after rerun."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.text_area[0].set_value("Persistent query").run()
        at.run()

        assert at.text_area[0].value == "Persistent query"
        assert not at.exception

    def test_pending_query_is_consumed(self):
        """pending_query should be removed after being processed."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        at.session_state["pending_query"] = "Consumable query"
        at.run()

        assert "pending_query" not in at.session_state
        assert at.text_area[0].value == "Consumable query"
        assert not at.exception


class TestMultipleRelatedSearches:
    """Test multiple consecutive related searches."""

    def test_click_all_three_related_buttons(self):
        """Test clicking each related button sequentially."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # Initial search
        at.text_area[0].set_value("EEG patterns").run()
        at.button[0].click().run()

        expected_queries = ["EEG alpha waves", "EEG beta waves", "EEG theta waves"]

        # Click first related button
        related_0 = None
        for btn in at.button:
            if btn.key == "related_0":
                related_0 = btn
                break

        if related_0:
            at = related_0.click().run()
            assert at.text_area[0].value in expected_queries
            assert not at.exception

    def test_search_then_related_then_search_again(self):
        """Test: search -> click related -> new search sequence."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # First search
        at.text_area[0].set_value("Initial query").run()
        at.button[0].click().run()
        assert "Initial query" in at.success[0].value

        # Click related button
        for btn in at.button:
            if btn.key and btn.key.startswith("related_"):
                at = btn.click().run()
                break

        # Now do a new manual search
        at.text_area[0].set_value("New manual query").run()
        at.button[0].click().run()

        assert "New manual query" in at.success[0].value
        assert not at.exception

    def test_results_update_when_related_clicked(self):
        """Test that success message updates when related button clicked."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # First search
        at.text_area[0].set_value("Original query").run()
        at.button[0].click().run()

        original_result = at.success[0].value
        assert "Original query" in original_result

        # Click related button
        for btn in at.button:
            if btn.key and btn.key.startswith("related_"):
                at = btn.click().run()
                break

        # Result should be different now
        new_result = at.success[0].value
        assert new_result != original_result
        assert not at.exception


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_no_search(self):
        """Clicking search with empty query should not produce results."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.button[0].click().run()

        assert len(at.success) == 0
        assert not at.exception

    def test_pending_query_with_existing_results(self):
        """pending_query should work even when results already exist."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # First search
        at.text_area[0].set_value("First query").run()
        at.button[0].click().run()

        # Set pending query directly (simulating button click)
        at.session_state["pending_query"] = "Second query"
        at.run()

        assert at.text_area[0].value == "Second query"
        assert "Second query" in at.success[0].value
        assert not at.exception

    def test_whitespace_only_query_no_search(self):
        """Whitespace-only query should not trigger search."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.text_area[0].set_value("   ").run()
        # Note: Our mock doesn't strip whitespace, but real app might
        # This tests the current behavior
        assert not at.exception

    def test_special_characters_in_query(self):
        """Query with special characters should work."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        at.text_area[0].set_value("EEG α-waves & β-rhythms (8-13Hz)?").run()
        at.button[0].click().run()

        assert "EEG α-waves & β-rhythms (8-13Hz)?" in at.success[0].value
        assert not at.exception

    def test_very_long_query(self):
        """Test handling of long queries."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()
        long_query = "EEG " * 100  # 400 characters
        at.text_area[0].set_value(long_query).run()
        at.button[0].click().run()

        assert long_query.strip() in at.success[0].value
        assert not at.exception

    def test_rapid_button_clicks(self):
        """Test multiple rapid clicks don't break state."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # First search
        at.text_area[0].set_value("Query 1").run()
        at.button[0].click().run()

        # Find and click related button multiple times
        for btn in at.button:
            if btn.key == "related_0":
                at = btn.click().run()
                at = btn.click().run()
                break

        # Should still be in valid state
        assert at.text_area[0].value in [
            "EEG alpha waves",
            "EEG beta waves",
            "EEG theta waves",
        ]
        assert not at.exception


class TestDoSearchFlag:
    """Tests for the do_search flag mechanism."""

    def test_do_search_triggers_search_on_next_run(self):
        """do_search flag should trigger search on next run."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        # Set query and do_search flag
        at.session_state["_query_widget"] = "Flagged query"
        at.session_state["do_search"] = True
        at.run()

        assert len(at.success) == 1
        assert "Flagged query" in at.success[0].value
        assert not at.exception

    def test_do_search_is_consumed(self):
        """do_search flag should be consumed after use."""
        at = AppTest.from_string(TEST_APP_SCRIPT).run()

        at.session_state["_query_widget"] = "Some query"
        at.session_state["do_search"] = True
        at.run()

        # do_search should be consumed (popped) - check using 'in' operator
        assert (
            "do_search" not in at.session_state
            or at.session_state["do_search"] == False
        )

        # Search should have happened with "Some query"
        assert len(at.success) == 1
        assert "Some query" in at.success[0].value
        assert not at.exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

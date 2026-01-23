"""CLI module for EEG-RAG system."""

from eeg_rag.cli.history_cli import history_cli
from eeg_rag.cli.verify_stats import stats as stats_cli

__all__ = ['history_cli', 'stats_cli']

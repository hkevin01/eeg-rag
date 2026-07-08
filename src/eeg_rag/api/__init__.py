"""
FastAPI Web Service for EEG Literature RAG System.
"""

try:
	from .main import app
except ModuleNotFoundError as exc:
	# Keep api package importable when FastAPI extras are not installed.
	if exc.name == "fastapi":
		app = None
	else:
		raise

__all__ = ["app"]

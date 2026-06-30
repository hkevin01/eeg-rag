"""Centrality enrichment for retrieval metadata.

Maps retrieved paper identifiers onto live citation-network centrality scores
computed by the bibliometrics subsystem. The enricher is optional so retrieval
can still operate when no citation graph is available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CentralityEnricher:
    """Attach citation-network centrality scores to metadata dictionaries."""

    def __init__(
        self,
        biblionet: Optional[Any] = None,
        method: str = "pagerank",
    ) -> None:
        self._biblionet = biblionet
        self._method = method
        self._loaded = False
        self._openalex_scores: Dict[str, float] = {}
        self._pmid_scores: Dict[str, float] = {}
        self._doi_scores: Dict[str, float] = {}

    def enrich(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Return metadata with a centrality score attached when available."""
        if metadata.get("centrality_score") is not None:
            return metadata

        self._ensure_loaded()
        if not self._loaded:
            return metadata

        openalex_id = str(metadata.get("openalex_id") or metadata.get("id") or "")
        pmid = str(metadata.get("pmid") or metadata.get("PMID") or "")
        doi = str(metadata.get("doi") or "").lower()

        score = None
        if openalex_id:
            score = self._openalex_scores.get(openalex_id)
        if score is None and pmid:
            score = self._pmid_scores.get(pmid)
        if score is None and doi:
            score = self._doi_scores.get(doi)

        if score is not None:
            metadata["centrality_score"] = float(score)

        return metadata

    def _ensure_loaded(self) -> None:
        """Compute and cache centrality maps from the bibliometric graph."""
        if self._loaded or self._biblionet is None:
            return

        articles = getattr(self._biblionet, "articles", None)
        graph = getattr(self._biblionet, "citation_graph", None)
        if not articles or graph is None:
            return

        centrality = self._biblionet.compute_citation_centrality(
            method=self._method,
        )
        self._openalex_scores = {
            str(article.openalex_id): float(centrality.get(article.openalex_id, 0.0))
            for article in articles
            if getattr(article, "openalex_id", None)
        }
        self._pmid_scores = {
            str(article.pmid): float(article.centrality_score)
            for article in articles
            if getattr(article, "pmid", None)
        }
        self._doi_scores = {
            str(article.doi).lower(): float(article.centrality_score)
            for article in articles
            if getattr(article, "doi", None)
        }
        self._loaded = True

"""Centrality enrichment for retrieval metadata.

Maps retrieved paper identifiers onto live citation-network centrality scores
computed by the bibliometrics subsystem. The enricher is optional so retrieval
can still operate when no citation graph is available.
"""

from __future__ import annotations

import json
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
        if not articles:
            articles = self._load_cached_articles()
            if articles:
                self._biblionet.articles = articles

        if not articles:
            return

        if graph is None:
            try:
                self._biblionet.build_citation_network()
                graph = getattr(self._biblionet, "citation_graph", None)
            except Exception:
                graph = None

        if graph is None:
            self._populate_identifier_maps_from_articles(articles)
            self._loaded = True
            return

        centrality = self._biblionet.compute_citation_centrality(
            method=self._method,
        )
        self._openalex_scores = {}
        self._pmid_scores = {}
        self._doi_scores = {}
        for article in articles:
            openalex_id = getattr(article, "openalex_id", None)
            score = 0.0
            if openalex_id is not None:
                score = float(centrality.get(openalex_id, getattr(article, "centrality_score", 0.0) or 0.0))
            else:
                score = float(getattr(article, "centrality_score", 0.0) or 0.0)

            if openalex_id:
                self._openalex_scores[str(openalex_id)] = score
            pmid = getattr(article, "pmid", None)
            if pmid:
                self._pmid_scores[str(pmid)] = score
            doi = getattr(article, "doi", None)
            if doi:
                self._doi_scores[str(doi).lower()] = score
        self._loaded = True

    def _populate_identifier_maps_from_articles(self, articles: Any) -> None:
        """Populate score maps from precomputed article centrality values."""
        self._openalex_scores = {}
        self._pmid_scores = {}
        self._doi_scores = {}
        for article in articles:
            score = float(getattr(article, "centrality_score", 0.0) or 0.0)
            openalex_id = getattr(article, "openalex_id", None)
            if openalex_id:
                self._openalex_scores[str(openalex_id)] = score
            pmid = getattr(article, "pmid", None)
            if pmid:
                self._pmid_scores[str(pmid)] = score
            doi = getattr(article, "doi", None)
            if doi:
                self._doi_scores[str(doi).lower()] = score

    def _load_cached_articles(self) -> Optional[Any]:
        """Load cached bibliometric articles when runtime memory is empty."""
        cache_dir = getattr(self._biblionet, "cache_dir", None)
        if cache_dir is None:
            return None

        try:
            from eeg_rag.bibliometrics.eeg_biblionet import EEGArticle
        except Exception:
            return None

        json_candidates = sorted(cache_dir.glob("*.json"), reverse=True)
        for cache_file in json_candidates:
            try:
                with open(cache_file, "r") as handle:
                    payload = json.load(handle)
            except Exception:
                continue

            if not isinstance(payload, list) or not payload:
                continue
            if not isinstance(payload[0], dict):
                continue
            if "openalex_id" not in payload[0]:
                continue

            try:
                return [EEGArticle(**article) for article in payload]
            except Exception:
                continue

        return None

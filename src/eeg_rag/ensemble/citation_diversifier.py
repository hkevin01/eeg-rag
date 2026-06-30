r"""
Citation diversification for aggregated evidence sets.

Applies a lightweight Maximal Marginal Relevance (MMR) strategy to reduce
redundancy among highly similar citations while preserving topical relevance.
The implementation is dependency-free and uses token-set Jaccard similarity as
the redundancy term, which keeps latency low inside the aggregation layer.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np


class CitationDiversifier:
    r"""Diversify citations with an MMR-style reranking objective.

    The selection objective is:

        $MMR(d) = \lambda \cdot U(d) - (1 - \lambda) \cdot
        \max_{s \in S} Sim(d, s)$

    where:
        - $U(d)$ is a utility score derived from relevance and bibliometrics
        - $Sim(d, s)$ is token-level Jaccard similarity between citations
        - $S$ is the set of already selected citations

    Utility is defined as a convex combination of:
        - retrieval relevance
        - citation-network centrality
        - cross-source support
        - recency
    """

    def __init__(
        self,
        diversity_lambda: float = 0.7,
        centrality_weight: float = 0.15,
        support_weight: float = 0.10,
        recency_weight: float = 0.05,
    ) -> None:
        """Initialize the diversifier.

        Args:
            diversity_lambda: Tradeoff between utility and redundancy penalty.
                Higher values favor pure relevance; lower values favor novelty.
            centrality_weight: Weight for citation-network centrality.
            support_weight: Weight for multi-source support.
            recency_weight: Weight for publication recency.

        Raises:
            ValueError: If weights are outside valid ranges.
        """
        if not 0.0 <= diversity_lambda <= 1.0:
            raise ValueError("diversity_lambda must be between 0.0 and 1.0")

        for name, value in (
            ("centrality_weight", centrality_weight),
            ("support_weight", support_weight),
            ("recency_weight", recency_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        total_aux_weight = (
            centrality_weight + support_weight + recency_weight
        )
        if total_aux_weight > 1.0:
            raise ValueError("auxiliary weights must sum to <= 1.0")

        self.diversity_lambda = diversity_lambda
        self.centrality_weight = centrality_weight
        self.support_weight = support_weight
        self.recency_weight = recency_weight

    def diversify(
        self,
        citations: Sequence[Any],
        max_results: Optional[int] = None,
    ) -> List[Any]:
        """Return citations reordered to reduce redundancy.

        Args:
            citations: Citation-like objects with title, abstract,
                relevance_score, source_agents, year, and metadata fields.
            max_results: Optional cap on the number of selected results.

        Returns:
            Diversified citation list in selected order.
        """
        if not citations:
            return []

        limit = max_results or len(citations)
        remaining = list(citations)
        selected: List[Any] = []
        token_cache = {
            id(citation): self._tokenize(self._citation_text(citation))
            for citation in remaining
        }

        while remaining and len(selected) < limit:
            best_citation = None
            best_mmr = -float("inf")

            for citation in remaining:
                utility = self._utility(citation)
                redundancy = 0.0

                if selected:
                    redundancy = max(
                        self._similarity(
                            citation,
                            other,
                            token_cache[id(citation)],
                            token_cache[id(other)],
                        )
                        for other in selected
                    )

                mmr_score = (
                    self.diversity_lambda * utility
                    - (1.0 - self.diversity_lambda) * redundancy
                )

                metadata = self._metadata(citation)
                metadata["utility_score"] = round(utility, 6)
                metadata["redundancy_penalty"] = round(redundancy, 6)
                metadata["mmr_score"] = round(mmr_score, 6)

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_citation = citation

            if best_citation is None:
                break

            selected.append(best_citation)
            remaining.remove(best_citation)

        return selected

    def analyze_set(
        self,
        citations: Sequence[Any],
    ) -> Dict[str, float]:
        """Compute redundancy and diversity diagnostics for a result set."""
        if len(citations) <= 1:
            return {
                "redundancy_score": 0.0,
                "diversity_score": 1.0,
                "average_pairwise_similarity": 0.0,
            }

        token_cache = {
            id(citation): self._tokenize(self._citation_text(citation))
            for citation in citations
        }
        pairwise: List[float] = []
        max_per_doc: List[float] = []

        for i, left in enumerate(citations):
            local_max = 0.0
            for j, right in enumerate(citations):
                if i == j:
                    continue
                similarity = self._similarity(
                    left,
                    right,
                    token_cache[id(left)],
                    token_cache[id(right)],
                )
                pairwise.append(similarity)
                local_max = max(local_max, similarity)
            max_per_doc.append(local_max)

        redundancy = sum(max_per_doc) / len(max_per_doc)
        average_pairwise = sum(pairwise) / len(pairwise) if pairwise else 0.0
        return {
            "redundancy_score": round(redundancy, 6),
            "diversity_score": round(1.0 - redundancy, 6),
            "average_pairwise_similarity": round(average_pairwise, 6),
        }

    def _utility(self, citation: Any) -> float:
        """Compute utility score in the range [0, 1]."""
        metadata = self._metadata(citation)

        relevance = self._clamp(getattr(citation, "relevance_score", 0.0))
        centrality = self._clamp(metadata.get("centrality_score", 0.0))
        support = self._clamp(len(getattr(citation, "source_agents", [])) / 3.0)
        recency = self._recency_score(getattr(citation, "year", None))

        base_weight = 1.0 - (
            self.centrality_weight
            + self.support_weight
            + self.recency_weight
        )

        utility = (
            base_weight * relevance
            + self.centrality_weight * centrality
            + self.support_weight * support
            + self.recency_weight * recency
        )
        return self._clamp(utility)

    @staticmethod
    def _metadata(citation: Any) -> Dict[str, Any]:
        """Return a mutable metadata mapping for a citation-like object."""
        metadata = getattr(citation, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(citation, "metadata", metadata)
        return metadata

    @staticmethod
    def _citation_text(citation: Any) -> str:
        """Join the text used for redundancy comparison."""
        title = str(getattr(citation, "title", "") or "")
        abstract = str(getattr(citation, "abstract", "") or "")
        return f"{title} {abstract}".strip()

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Tokenize text into a small lexical signature."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _jaccard_similarity(left: Set[str], right: Set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _similarity(
        self,
        left_citation: Any,
        right_citation: Any,
        left_tokens: Set[str],
        right_tokens: Set[str],
    ) -> float:
        """Use cosine similarity when embeddings are present, else Jaccard."""
        left_embedding = self._extract_embedding(left_citation)
        right_embedding = self._extract_embedding(right_citation)

        if left_embedding is not None and right_embedding is not None:
            if left_embedding.shape == right_embedding.shape and left_embedding.size > 0:
                denom = float(np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding))
                if denom > 0.0:
                    cosine = float(np.dot(left_embedding, right_embedding) / denom)
                    return max(0.0, min(cosine, 1.0))

        return self._jaccard_similarity(left_tokens, right_tokens)

    def _extract_embedding(self, citation: Any) -> Optional[np.ndarray]:
        """Read a dense embedding vector from citation metadata when present."""
        metadata = self._metadata(citation)
        for key in ("embedding_vector", "dense_embedding", "embedding"):
            value = metadata.get(key)
            if isinstance(value, list) and value:
                return np.asarray(value, dtype=np.float32)
        return None

    @staticmethod
    def _recency_score(year: Optional[int]) -> float:
        """Map publication year to a smooth [0, 1] recency score."""
        if not isinstance(year, int):
            return 0.0
        # Logistic recency curve centered near recent EEG literature.
        age = max(0, 2026 - year)
        return 1.0 / (1.0 + math.exp((age - 5.0) / 2.0))

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp numeric values to [0, 1]."""
        return max(0.0, min(float(value), 1.0))

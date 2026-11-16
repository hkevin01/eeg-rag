"""Optional contradiction-aware claim evaluation with pinned NLI checkpoints."""

import re
from typing import Any, Optional

import numpy as np

from eeg_rag.verification.citation_verifier import ClaimEvidence, ClaimStatus


class NLIClaimEvaluator:
    """Evaluate source-to-claim entailment rather than topical cosine similarity.

    Args:
        model_name: NLI checkpoint using contradiction, entailment, neutral labels.
        revision: Checkpoint revision; pin a commit for deployment reproducibility.
        threshold: Minimum decision confidence, requiring EEG-specific calibration.
        backend: Sentence Transformers inference backend.
        model: Optional already-loaded compatible model.

    Raises:
        ValueError: Confidence threshold is outside the valid range.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        revision: Optional[str] = None,
        threshold: float = 0.9,
        backend: str = "torch",
        model: Any = None,
    ) -> None:
        if not 0.5 < threshold <= 1.0:
            raise ValueError("NLI threshold must be greater than 0.5 and at most 1")
        if model_name != "cross-encoder/nli-deberta-v3-base" and model is None:
            raise ValueError(
                "Other NLI checkpoints require an explicitly adapted model"
            )
        self.model_name = model_name
        self.revision = revision
        self.threshold = threshold
        self.backend = backend
        self._model = model

    def __call__(self, claim: str, abstract: str) -> ClaimEvidence:
        """Return an entailment decision and exact source span, or unknown on error.

        Args:
            claim: Finding attributed to the publication.
            abstract: Authoritative abstract text.

        Returns:
            Contradiction, supported, or unknown evidence with source provenance.
        """
        if not claim.strip() or not abstract.strip():
            return ClaimEvidence(ClaimStatus.UNKNOWN, 0.0)
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name, revision=self.revision, backend=self.backend
                )
            spans = [
                span for span in re.split(r"(?<=[.!?])\s+", abstract) if span.strip()
            ]
            scores = np.asarray(
                self._model.predict(
                    [(span, claim) for span in spans],
                    apply_softmax=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            if (
                scores.shape != (len(spans), 3)
                or not np.isfinite(scores).all()
                or (scores < 0).any()
                or (scores > 1).any()
                or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3)
            ):
                return ClaimEvidence(ClaimStatus.UNKNOWN, 0.0)
            contradiction = int(scores[:, 0].argmax())
            if scores[contradiction, 0] >= self.threshold:
                return ClaimEvidence(
                    ClaimStatus.UNSUPPORTED,
                    float(scores[contradiction, 0]),
                    spans[contradiction],
                )
            entailment = int(scores[:, 1].argmax())
            status = (
                ClaimStatus.SUPPORTED
                if scores[entailment, 1] >= self.threshold
                else ClaimStatus.UNKNOWN
            )
            return ClaimEvidence(
                status, float(scores[entailment, 1]), spans[entailment]
            )
        except Exception:
            return ClaimEvidence(ClaimStatus.UNKNOWN, 0.0)

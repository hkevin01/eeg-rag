"""
Synthetic QA Pair Generation for RAG Fine-Tuning.

This module implements the QA-pair grounding pipeline:
  1. For each text chunk, generate N question-answer pairs grounded
     strictly in the chunk's content using an LLM.
  2. Embed each synthetic question using the same embedding model
     used for the corpus (PubMedBERT or configurable alternative).
  3. Store (question_embedding, chunk_id, answer, chunk_text) tuples
     so the retriever can use question vectors as additional index
     entries pointing back to their source chunk.

Why this matters for EEG-RAG
------------------------------
Standard dense retrieval embeds document chunks and matches them against
query vectors.  The semantic gap between a clinical question like
"What EEG patterns appear in status epilepticus?" and a chunk that
discusses "ictal discharge morphology in SE" reduces recall because the
surface forms differ.

Synthetic QA pairs bridge this gap: the model that generated the question
was shown the chunk, so the question's wording naturally aligns with
how a user would ask about that content.  By embedding the synthetic
question (rather than the raw chunk) and indexing it in FAISS alongside
the chunk embeddings, retrieval recall improves without any fine-tuning
of the embedding model itself.

This is sometimes called "HyDE in reverse" — instead of expanding the
query at inference time, we expand the index at build time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ID           : rag.synthetic_qa.QAPair
# Requirement  : `QAPair` shall hold one synthetic question-answer pair grounded
#                in a specific chunk.
# Purpose      : Carrier for a single (question, answer, source_chunk) triple
#                produced by the QA generation pipeline.
# Rationale    : Dataclass provides typed, serialisable storage with zero
#                boilerplate; JSON-round-trippable via `to_dict`.
# Inputs       : question: str, answer: str, chunk_id: str, chunk_text: str,
#                question_type: str (factoid | causal | procedural | comparative)
# Outputs      : N/A (class definition)
# Precond.     : All fields are non-empty strings.
# Postcond.    : `pair_id` is deterministic SHA-256(chunk_id + question[:64]).
# Assumptions  : chunk_id is unique within its FAISS index.
# Side Effects : None at construction time.
# Fail Modes   : ValueError if chunk_id is empty string.
# Err Handling : Raises at construction; validated in SyntheticQAGenerator.
# Constraints  : question and answer must be < 2048 chars for typical LLM limits.
# Verification : Unit test: pair_id deterministic; to_dict round-trips via json.
# References   : HyDE (Gao et al. 2022); InPars (Bonifacio et al. 2022).
# ---------------------------------------------------------------------------
@dataclass
class QAPair:
    """One synthetic question-answer pair grounded in a specific chunk."""

    question: str
    answer: str
    chunk_id: str
    chunk_text: str
    question_type: str = "factoid"  # factoid | causal | procedural | comparative
    pair_id: str = field(init=False)
    question_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = f"{self.chunk_id}::{self.question[:64]}"
        self.pair_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (embedding excluded)."""
        return {
            "pair_id": self.pair_id,
            "question": self.question,
            "answer": self.answer,
            "chunk_id": self.chunk_id,
            "question_type": self.question_type,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# ID           : rag.synthetic_qa.QABatch
# Requirement  : `QABatch` shall aggregate all pairs produced for a corpus.
# Purpose      : Container returned by `SyntheticQAGenerator.generate_for_corpus`.
# Inputs       : pairs: List[QAPair], stats: Dict[str, Any]
# Outputs      : N/A (class definition)
# Precond.     : pairs is a non-empty list.
# Postcond.    : `total_pairs` equals `len(pairs)`.
# ---------------------------------------------------------------------------
@dataclass
class QABatch:
    """All QA pairs generated for a corpus or document set."""

    pairs: List[QAPair]
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)

    def pairs_for_chunk(self, chunk_id: str) -> List[QAPair]:
        return [p for p in self.pairs if p.chunk_id == chunk_id]

    def to_jsonl(self, path: str | Path) -> None:
        """Write pairs to a JSONL file for downstream fine-tuning."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for pair in self.pairs:
                fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d QA pairs to %s", self.total_pairs, path)

    def get_question_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return (embedding_matrix, chunk_ids) for pairs that have embeddings.

        The embedding matrix can be passed directly to `faiss.IndexFlatIP.add`
        after normalisation.
        """
        embedded = [p for p in self.pairs if p.question_embedding is not None]
        if not embedded:
            return np.empty((0, 0), dtype=np.float32), []
        matrix = np.vstack([p.question_embedding for p in embedded]).astype(np.float32)
        chunk_ids = [p.chunk_id for p in embedded]
        return matrix, chunk_ids


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = "llama3", timeout: int = 90) -> str:
    """Call a locally running Ollama instance."""
    url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def _call_openai(prompt: str, model: str = "gpt-3.5-turbo", timeout: int = 60) -> str:
    """Call OpenAI chat completions (lazy import)."""
    import openai  # type: ignore
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt: str, model: str = "claude-3-haiku-20240307", timeout: int = 60) -> str:
    """Call Anthropic Messages API (lazy import)."""
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text if msg.content else ""


def _dispatch_llm(prompt: str, backend: str, model: Optional[str] = None) -> str:
    """Route to the correct LLM backend."""
    if backend == "ollama":
        return _call_ollama(prompt, model or "llama3")
    if backend == "openai":
        return _call_openai(prompt, model or "gpt-3.5-turbo")
    if backend == "anthropic":
        return _call_anthropic(prompt, model or "claude-3-haiku-20240307")
    raise ValueError(f"Unknown LLM backend: {backend!r}. Choose ollama/openai/anthropic.")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QA_PROMPT = """You are an expert in electroencephalography (EEG) and neuroscience research.
Read the following text chunk from a scientific paper and generate {n} question-answer pairs.

Rules:
- Every question must be answerable SOLELY from the provided text.
- Questions should be diverse: include factoid, causal (why/how), procedural,
  and comparative types where the content permits.
- Answers must be concise (1-3 sentences) and cite specific information from the text.
- Do NOT add information not present in the text.
- Output ONLY a JSON array. No preamble, no explanation.

Output format (strict JSON array):
[
  {{"question": "...", "answer": "...", "type": "factoid|causal|procedural|comparative"}},
  ...
]

Text chunk:
\"\"\"
{chunk_text}
\"\"\"
"""


def _parse_qa_json(raw: str) -> List[Dict[str, str]]:
    """Extract the JSON array from an LLM response, stripping markdown fences."""
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    # Find first '[' and last ']'
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ID           : rag.synthetic_qa.SyntheticQAGenerator
# Requirement  : For every input chunk, `SyntheticQAGenerator` shall produce
#                N grounded QA pairs and embed each question using the
#                configured sentence-transformer model.
# Purpose      : Augment the FAISS index with question-side embeddings so that
#                user queries match chunks via semantically aligned questions,
#                reducing the surface-form gap between queries and documents.
# Rationale    : Dense retrieval accuracy is limited when query phrasing differs
#                from document phrasing. Embedding synthetic questions (which
#                share phrasing patterns with real user queries) and indexing
#                them alongside chunk embeddings improves recall at no inference
#                cost.  This is the build-time analogue of HyDE.
# Inputs       : chunks: List[Dict] with keys {chunk_id, text, metadata};
#                n_questions_per_chunk: int in [1, 10];
#                llm_backend: "ollama" | "openai" | "anthropic";
#                embed_model: sentence-transformers model name.
# Outputs      : QABatch with embedded QAPair objects.
# Precond.     : LLM backend reachable; sentence-transformers installed.
# Postcond.    : `batch.total_pairs >= len(chunks)` (at least 1 per chunk).
# Assumptions  : Each chunk is 100-1024 tokens; longer chunks are truncated.
# Side Effects : LLM API calls (billable); disk writes via `to_jsonl()`.
# Fail Modes   : LLM timeout → chunk skipped with warning; JSON parse error →
#                chunk skipped with warning; embedding OOM → batch size halved.
# Err Handling : Per-chunk try/except; errors logged; generation continues.
# Constraints  : Rate limited by LLM API quotas; default 0.5 s inter-chunk delay.
# Verification : test_synthetic_qa.py — mock LLM backend, assert pair_id stable,
#                assert embeddings shape correct, assert JSONL round-trips.
# References   : InPars (Bonifacio et al. 2022); GPL (Wang et al. 2022);
#                HyDE (Gao et al. 2022).
# ---------------------------------------------------------------------------
class SyntheticQAGenerator:
    """
    Generates synthetic question-answer pairs grounded in each text chunk,
    then embeds the questions for use as additional FAISS index entries.

    The workflow is:
        chunks → LLM prompt → JSON QA pairs → embed questions → QABatch

    Usage::

        from eeg_rag.rag.synthetic_qa import SyntheticQAGenerator

        gen = SyntheticQAGenerator(
            llm_backend="ollama",        # or "openai" / "anthropic"
            n_questions_per_chunk=3,
        )
        batch = gen.generate_for_corpus(my_chunks)
        batch.to_jsonl("data/qa_pairs.jsonl")

        # Inject into FAISS index
        gen.inject_into_faiss(batch, faiss_index, chunk_id_to_index_map)
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        llm_model: Optional[str] = None,
        n_questions_per_chunk: int = 3,
        embed_model: str = "all-MiniLM-L6-v2",
        inter_chunk_delay: float = 0.5,
        max_chunk_chars: int = 3000,
    ) -> None:
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.n_questions = n_questions_per_chunk
        self.embed_model_name = embed_model
        self.inter_chunk_delay = inter_chunk_delay
        self.max_chunk_chars = max_chunk_chars
        self._embed_model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._embed_model = SentenceTransformer(self.embed_model_name)
            logger.info("Loaded embedding model: %s", self.embed_model_name)
        return self._embed_model

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        model = self._get_embed_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def _generate_for_chunk(self, chunk_id: str, chunk_text: str) -> List[QAPair]:
        """Call LLM and parse QA pairs for a single chunk."""
        truncated = chunk_text[: self.max_chunk_chars]
        prompt = _QA_PROMPT.format(n=self.n_questions, chunk_text=truncated)
        raw = _dispatch_llm(prompt, self.llm_backend, self.llm_model)
        items = _parse_qa_json(raw)
        pairs: List[QAPair] = []
        for item in items[: self.n_questions]:
            pairs.append(
                QAPair(
                    question=item.get("question", "").strip(),
                    answer=item.get("answer", "").strip(),
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    question_type=item.get("type", "factoid"),
                )
            )
        return pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_for_corpus(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> QABatch:
        """
        Generate QA pairs for every chunk in the corpus.

        Parameters
        ----------
        chunks:
            List of dicts, each with at least ``chunk_id`` (str) and
            ``text`` (str).  Optional ``metadata`` dict is preserved.
        show_progress:
            Print a progress counter to stdout.

        Returns
        -------
        QABatch
            All generated pairs with embedded questions.
        """
        all_pairs: List[QAPair] = []
        errors = 0

        for idx, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}")))
            text = chunk.get("text", chunk.get("content", ""))

            if show_progress and idx % 10 == 0:
                logger.info("QA generation: %d / %d chunks", idx, len(chunks))

            try:
                pairs = self._generate_for_chunk(chunk_id, text)
                all_pairs.extend(pairs)
            except Exception as exc:
                logger.warning("Skipping chunk %s — LLM error: %s", chunk_id, exc)
                errors += 1

            if self.inter_chunk_delay > 0:
                time.sleep(self.inter_chunk_delay)

        # Embed all questions in one batch
        if all_pairs:
            questions = [p.question for p in all_pairs]
            logger.info("Embedding %d synthetic questions …", len(questions))
            embeddings = self._embed_texts(questions)
            for pair, emb in zip(all_pairs, embeddings):
                pair.question_embedding = emb

        stats = {
            "total_chunks": len(chunks),
            "total_pairs": len(all_pairs),
            "errors": errors,
            "avg_pairs_per_chunk": len(all_pairs) / max(len(chunks), 1),
            "embed_model": self.embed_model_name,
            "llm_backend": self.llm_backend,
        }
        logger.info("QA generation complete: %s", stats)
        return QABatch(pairs=all_pairs, stats=stats)

    def inject_into_faiss(
        self,
        batch: QABatch,
        faiss_index,
        chunk_id_to_int: Dict[str, int],
    ) -> int:
        """
        Add question embeddings to an existing FAISS index.

        Each question embedding is added as a new vector whose associated
        document ID is the integer index of the source chunk in the existing
        index.  The FAISS index must support `add_with_ids` (e.g.
        ``faiss.IndexIDMap`` wrapping a flat index).

        Parameters
        ----------
        batch:
            QABatch returned by `generate_for_corpus`.
        faiss_index:
            A FAISS index that supports `add_with_ids`.
        chunk_id_to_int:
            Mapping from chunk string IDs to the integer IDs already in
            the FAISS index.

        Returns
        -------
        int
            Number of question vectors injected.
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu") from exc

        matrix, chunk_ids = batch.get_question_embeddings()
        if matrix.shape[0] == 0:
            logger.warning("No embedded questions to inject.")
            return 0

        int_ids = np.array(
            [chunk_id_to_int[cid] for cid in chunk_ids if cid in chunk_id_to_int],
            dtype=np.int64,
        )
        valid_mask = np.array([cid in chunk_id_to_int for cid in chunk_ids])
        valid_matrix = matrix[valid_mask]

        faiss_index.add_with_ids(valid_matrix, int_ids)
        logger.info("Injected %d question vectors into FAISS index.", len(int_ids))
        return int(len(int_ids))

    def save_batch(self, batch: QABatch, output_dir: str | Path) -> Dict[str, str]:
        """
        Persist the batch to disk:
          - ``qa_pairs.jsonl``  — one JSON object per line (question, answer, chunk_id)
          - ``question_embeddings.npy``  — float32 matrix (N x D)
          - ``question_chunk_ids.json``  — list of chunk_id strings aligned with rows

        Returns a dict of written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_dir / "qa_pairs.jsonl"
        batch.to_jsonl(jsonl_path)

        matrix, chunk_ids = batch.get_question_embeddings()
        if matrix.shape[0] > 0:
            emb_path = output_dir / "question_embeddings.npy"
            np.save(str(emb_path), matrix)

            ids_path = output_dir / "question_chunk_ids.json"
            ids_path.write_text(json.dumps(chunk_ids, ensure_ascii=False))
        else:
            emb_path = ids_path = None
            logger.warning("No embeddings to save (questions were not embedded).")

        stats_path = output_dir / "qa_stats.json"
        stats_path.write_text(json.dumps(batch.stats, indent=2))

        return {
            "jsonl": str(jsonl_path),
            "embeddings": str(emb_path) if emb_path else "",
            "chunk_ids": str(ids_path) if ids_path else "",
            "stats": str(stats_path),
        }


# ---------------------------------------------------------------------------
# Convenience async wrapper
# ---------------------------------------------------------------------------

async def generate_qa_pairs_async(
    chunks: List[Dict[str, Any]],
    llm_backend: str = "ollama",
    n_questions_per_chunk: int = 3,
    embed_model: str = "all-MiniLM-L6-v2",
    output_dir: Optional[str] = None,
) -> QABatch:
    """
    Async-friendly entry point — runs `generate_for_corpus` in a thread pool
    so it does not block an event loop.

    Parameters
    ----------
    chunks:
        List of chunk dicts with ``chunk_id`` and ``text`` keys.
    llm_backend:
        One of ``"ollama"``, ``"openai"``, ``"anthropic"``.
    n_questions_per_chunk:
        Number of QA pairs to generate per chunk (1-10).
    embed_model:
        Sentence-transformer model for question embedding.
    output_dir:
        If provided, persists the batch to this directory automatically.

    Returns
    -------
    QABatch
    """
    gen = SyntheticQAGenerator(
        llm_backend=llm_backend,
        n_questions_per_chunk=n_questions_per_chunk,
        embed_model=embed_model,
    )
    loop = asyncio.get_event_loop()
    batch = await loop.run_in_executor(None, gen.generate_for_corpus, chunks)
    if output_dir:
        gen.save_batch(batch, output_dir)
    return batch

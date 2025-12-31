from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from agentic_resume_tailor.core.keyword_matcher import latex_to_plain_for_matching


@dataclass(frozen=True)
class QueryItem:
    query: str
    purpose: str = "general"
    boost_keywords: Tuple[str, ...] = ()
    weight: float = 1.0


@dataclass
class Hit:
    query: str
    purpose: str
    weight: float
    cosine: float
    weighted: float


@dataclass
class Candidate:
    bullet_id: str
    source: str
    text_latex: str
    meta: Dict[str, Any]
    best_hit: Hit
    total_weighted: float
    effective_total_weighted: float
    selection_score: float
    quant_bonus: float
    hits: List[Hit]


def _l2norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (_l2norm(a) * _l2norm(b)))


def normalize_query_text(q: str) -> str:
    # Keep it simple and embedding-friendly (no boolean logic).
    q = (q or "").strip()
    q = " ".join(q.split())
    return q.lower()


_QUANT_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?%?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(ms|s|sec|secs|seconds|min|mins|minutes|hrs|hours)\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(kb|mb|gb|tb)\b"),
    re.compile(r"\bx\d+\b"),
    re.compile(r"\b\d+\s?(users|customers|clients|requests|rps|qps)\b"),
    re.compile(r"\bauc\s?\d+(?:\.\d+)?\b"),
    re.compile(r"\b(reduced|improved|increased|cut|decreased|grew)\s+by\s+\d+(?:\.\d+)?%?\b"),
    re.compile(r"\bfrom\s+\d+(?:\.\d+)?\s+\w*\s+to\s+\d+(?:\.\d+)?\b"),
]


def _compute_quant_bonus(text_latex: str) -> float:
    plain = latex_to_plain_for_matching(text_latex or "")
    if not plain:
        return 0.0
    text = plain.lower()
    hits = sum(1 for pat in _QUANT_PATTERNS if pat.search(text))
    if hits <= 0:
        return 0.0
    return min(0.05 * hits, 0.20)


def _build_query_items(jd_parser_result: Any) -> List[QueryItem]:
    """
    Accepts multiple shapes:
    1) TargetProfileV1: profile.retrieval_plan.experience_queries = [{query, purpose, boost_keywords, weight}, ...]
    2) JobRequirements-like: result.experience_queries = [str, ...]
    3) plain list[str]
    """
    # Case 3: list[str]
    if isinstance(jd_parser_result, list) and all(isinstance(x, str) for x in jd_parser_result):
        return [QueryItem(query=normalize_query_text(q)) for q in jd_parser_result if q.strip()]

    # Case 2: has attribute experience_queries (list[str])
    if hasattr(jd_parser_result, "experience_queries"):
        eq = getattr(jd_parser_result, "experience_queries", None)
        if isinstance(eq, list) and all(isinstance(x, str) for x in eq):
            return [QueryItem(query=normalize_query_text(q)) for q in eq if q.strip()]

    # Case 1: TargetProfileV1-like dict/pydantic
    # Try dict-style access first
    profile = jd_parser_result
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()

    if isinstance(profile, dict):
        rp = profile.get("retrieval_plan", {})
        eq = rp.get("experience_queries", [])
        items: List[QueryItem] = []
        if isinstance(eq, list):
            for it in eq:
                if isinstance(it, dict) and it.get("query"):
                    q = normalize_query_text(it["query"])
                    purpose = it.get("purpose", "general") or "general"
                    boost = tuple((it.get("boost_keywords") or []))
                    weight = float(it.get("weight", 1.0))
                    items.append(
                        QueryItem(query=q, purpose=purpose, boost_keywords=boost, weight=weight)
                    )
        if items:
            return items

    raise ValueError(
        "Unsupported JD parser result shape. Need experience_queries or retrieval_plan.experience_queries."
    )


def multi_query_retrieve(
    collection: Any,
    embedding_fn: Any,
    jd_parser_result: Any,
    per_query_k: int = 10,
    final_k: int = 30,
) -> List[Candidate]:
    """
    Node 2 retrieval:
    - runs multi-query over Chroma
    - merges by bullet_id (Chroma record id)
    - reranks using cosine similarity computed explicitly from embeddings
    - returns ranked candidates + provenance (best query, hits)

    Assumptions:
    - Chroma record id is your deterministic bullet_id
    - metadata contains company OR project name
    - text is either meta['text_latex'] or doc
    """
    query_items = _build_query_items(jd_parser_result)

    merged: Dict[str, Dict[str, Any]] = {}

    for qi in query_items:
        # Apply boosts by appending canonical boosts to the query text (generic, no hardcoding)
        parts = [qi.query]
        if qi.boost_keywords:
            parts.extend([k for k in qi.boost_keywords if k])
        boosted_query = " ".join(parts).strip()

        # Embed query explicitly so we can compute cosine ourselves
        q_emb = np.array(embedding_fn([boosted_query])[0], dtype=np.float32)

        res = collection.query(
            query_texts=[boosted_query],
            n_results=per_query_k,
            # ids returned automatically
            include=["documents", "metadatas", "embeddings"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0]

        for i in range(len(ids)):
            bullet_id = ids[i]
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            emb = embs[i] if i < len(embs) else None

            if emb is None:
                # If embeddings are not returned, we cannot compute cosine; skip (or fallback later).
                continue

            d_emb = np.array(emb, dtype=np.float32)
            cos = cosine_similarity(q_emb, d_emb)  # [-1, 1], higher is better
            weighted = qi.weight * cos

            source = (
                meta.get("company") or meta.get("name") or meta.get("project") or "Unknown Source"
            )
            text = meta.get("text_latex") or doc

            hit = Hit(
                query=boosted_query,
                purpose=qi.purpose,
                weight=float(qi.weight),
                cosine=float(cos),
                weighted=float(weighted),
            )

            if bullet_id not in merged:
                merged[bullet_id] = {
                    "bullet_id": bullet_id,
                    "source": source,
                    "text_latex": text,
                    "meta": meta,
                    "hits": [hit],
                    "best_hit": hit,
                    "total_weighted": weighted,
                }
            else:
                merged[bullet_id]["hits"].append(hit)
                merged[bullet_id]["total_weighted"] += weighted
                if hit.weighted > merged[bullet_id]["best_hit"].weighted:
                    merged[bullet_id]["best_hit"] = hit

    # Rerank:
    # - primary: best weighted cosine + quant bonus
    # - secondary: total weighted + quant bonus (reward multi-hit)
    candidates: List[Candidate] = []
    for _, v in merged.items():
        # sort hits desc for debugging/provenance
        v["hits"].sort(key=lambda h: h.weighted, reverse=True)
        total_weighted = float(v["total_weighted"])
        quant_bonus = _compute_quant_bonus(v["text_latex"])
        selection_score = float(v["best_hit"].weighted) + quant_bonus
        effective_total_weighted = total_weighted + quant_bonus
        candidates.append(
            Candidate(
                bullet_id=v["bullet_id"],
                source=v["source"],
                text_latex=v["text_latex"],
                meta=v["meta"],
                best_hit=v["best_hit"],
                total_weighted=total_weighted,
                effective_total_weighted=effective_total_weighted,
                selection_score=selection_score,
                quant_bonus=quant_bonus,
                hits=v["hits"],
            )
        )

    candidates.sort(
        key=lambda c: (
            -c.selection_score,
            -c.effective_total_weighted,
            -c.total_weighted,
            c.bullet_id,
        )
    )

    return candidates[:final_k]

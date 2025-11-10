from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import faiss
import numpy as np
from langchain_core.documents import Document

from python.helpers.tokens import approximate_tokens

if TYPE_CHECKING:  # pragma: no cover - import cycles guard
    from agent import Agent
    from python.helpers.vector_db import VectorDB
    from python.helpers.neo4j_memory import Neo4jMemory

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievalConfig:
    """Configuration for the hybrid retriever."""

    top_k: int = 8
    graph_hops: int = 2
    graph_branching: int = 4
    rerank_top_k: int = 12
    token_budget: int = 1400
    reranker: str = "hybrid"  # "hybrid", "utility", or "none"
    reranker_model: str = ""
    ivf_nlist: int = 100
    pq_m: int = 16
    pq_bits: int = 8
    min_train_vectors: int = 64
    min_similarity: float = 0.0


@dataclass
class QueryPlan:
    """Expanded query plan used during retrieval."""

    original_query: str
    rewritten_queries: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class RetrievedSegment:
    """Final retrieval payload segment."""

    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    hop: int
    dom_path: Optional[str] = None
    screenshot: Optional[str] = None


@dataclass
class HybridRetrievalResult:
    """Aggregate result returned by :class:`HybridRetriever`."""

    query: str
    plan: QueryPlan
    segments: List[RetrievedSegment]


@dataclass
class _Candidate:
    """Internal representation of a retrieval candidate."""

    doc_id: str
    document: Document
    score: float
    hop: int = 0
    keywords_matched: int = 0


class HybridRetriever:
    """Perform hybrid (vector + graph) retrieval with optional reranking."""

    def __init__(
        self,
        agent: Optional["Agent"],
        vector_store: "VectorDB",
        graph_store: Optional["Neo4jMemory"],
        *,
        config: Optional[HybridRetrievalConfig] = None,
        query_rewriter: Optional[Callable[[str], Awaitable[QueryPlan]]] = None,
    ) -> None:
        self.agent = agent
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.config = config or HybridRetrievalConfig()
        self._query_rewriter = query_rewriter
        self._pq_index: Optional[faiss.IndexIDMap] = None
        self._pq_docs: List[Document] = []
        self._pq_total: int = 0
        self._index_lock = asyncio.Lock()

    async def retrieve(self, query: str) -> HybridRetrievalResult:
        """Execute the full hybrid retrieval pipeline for ``query``."""

        start = time.perf_counter()
        plan = await self._rewrite_query(query)
        base_results = await self._vector_search(plan)
        expanded = await self._expand_graph(base_results, plan)
        ranked = await self._rerank_candidates(expanded, query, plan)
        segments = self._package_segments(ranked)
        elapsed = time.perf_counter() - start
        logger.debug("Hybrid retrieval for '%s' completed in %.2f s", query, elapsed)
        return HybridRetrievalResult(query=query, plan=plan, segments=segments)

    async def _rewrite_query(self, query: str) -> QueryPlan:
        if self._query_rewriter:
            try:
                plan = await self._query_rewriter(query)
                if plan.rewritten_queries:
                    return plan
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Custom query rewriter failed: %s", exc)

        if self.agent:
            try:
                plan = await self._rewrite_with_agent(query)
                if plan.rewritten_queries:
                    return plan
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Utility model query rewrite failed: %s", exc)

        return self._heuristic_plan(query)

    async def _rewrite_with_agent(self, query: str) -> QueryPlan:
        assert self.agent  # for type checker
        system_prompt = (
            "You expand search queries for a retrieval engine. Respond in JSON with the keys "
            "'expansions', 'boolean', and 'keywords'. 'expansions' is a list of rewritten "
            "queries, 'boolean' is an optional boolean expression string, and 'keywords' is a "
            "list of important keywords."
        )
        response = await self.agent.call_utility_model(system=system_prompt, message=query)
        plan = self._parse_rewrite_response(query, response)
        if plan.rewritten_queries:
            return plan
        return self._heuristic_plan(query)

    def _parse_rewrite_response(self, original: str, text: str) -> QueryPlan:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return QueryPlan(original_query=original)

        expansions: List[str] = []
        keywords: List[str] = []

        if isinstance(data, dict):
            boolean_expr = data.get("boolean")
            if isinstance(boolean_expr, str) and boolean_expr.strip():
                expansions.append(boolean_expr.strip())
            raw_expansions = data.get("expansions")
            if isinstance(raw_expansions, list):
                expansions.extend(str(item).strip() for item in raw_expansions if str(item).strip())
            raw_keywords = data.get("keywords")
            if isinstance(raw_keywords, list):
                keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
        elif isinstance(data, list):
            expansions.extend(str(item).strip() for item in data if str(item).strip())

        expansions = self._deduplicate([original] + expansions)
        return QueryPlan(original_query=original, rewritten_queries=expansions, keywords=self._deduplicate(keywords))

    def _heuristic_plan(self, query: str) -> QueryPlan:
        tokens = re.findall(r"[\w-]+", query.lower())
        keywords = [tok for tok in tokens if len(tok) > 2]
        boolean_expr = " AND ".join(f'"{kw}"' for kw in keywords) if keywords else ""
        expansions = [query]
        if boolean_expr:
            expansions.append(boolean_expr)
        return QueryPlan(
            original_query=query,
            rewritten_queries=self._deduplicate(expansions),
            keywords=self._deduplicate(keywords),
        )

    async def _ensure_ann_index(self) -> None:
        async with self._index_lock:
            docs_map = getattr(self.vector_store.db, "get_all_docs", lambda: {})()
            docs = list(docs_map.values())
            total = len(docs)
            if total < self.config.min_train_vectors:
                self._pq_index = None
                self._pq_docs = docs
                self._pq_total = total
                return

            if self._pq_index is not None and self._pq_total == total:
                return

            embeddings: List[List[float]] = []
            for doc in docs:
                text = doc.page_content or ""
                embeddings.append(self.vector_store.embeddings.embed_query(text))

            matrix = np.asarray(embeddings, dtype="float32")
            if matrix.size == 0:
                self._pq_index = None
                self._pq_docs = docs
                self._pq_total = total
                return

            nlist = min(self.config.ivf_nlist, max(1, total // 2))
            quantizer = faiss.IndexFlatIP(matrix.shape[1])
            try:
                pq = faiss.IndexIVFPQ(quantizer, matrix.shape[1], nlist, self.config.pq_m, self.config.pq_bits)
                pq.train(matrix)
                id_map = faiss.IndexIDMap(pq)
                ids = np.arange(total, dtype="int64")
                id_map.add_with_ids(matrix, ids)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to prepare PQ index, falling back to exact search: %s", exc)
                self._pq_index = None
                self._pq_docs = docs
                self._pq_total = total
                return

            self._pq_index = id_map
            self._pq_docs = docs
            self._pq_total = total

    async def _vector_search(self, plan: QueryPlan) -> List[_Candidate]:
        await self._ensure_ann_index()
        aggregated: Dict[str, _Candidate] = {}
        queries = plan.rewritten_queries or [plan.original_query]

        for variant in queries:
            vector = None
            try:
                vector = self.vector_store.embeddings.embed_query(variant)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Embedding query failed: %s", exc)
                continue

            if self._pq_index is not None and self._pq_docs:
                scores, indices = self._pq_index.search(np.asarray([vector], dtype="float32"), self.config.top_k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    base_doc = self._pq_docs[idx]
                    doc = Document(page_content=base_doc.page_content, metadata=dict(base_doc.metadata))
                    doc_id = str(doc.metadata.get("id", idx))
                    candidate = aggregated.get(doc_id)
                    normalized = float(score)
                    if candidate is None or normalized > candidate.score:
                        aggregated[doc_id] = _Candidate(
                            doc_id=doc_id,
                            document=doc,
                            score=normalized,
                            hop=0,
                            keywords_matched=self._count_keyword_matches(doc.page_content, plan.keywords),
                        )
            else:
                results = await self.vector_store.search_by_similarity_threshold(
                    variant,
                    limit=self.config.top_k,
                    threshold=self.config.min_similarity,
                )
                for rank, doc in enumerate(results):
                    doc_copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                    doc_id = str(doc_copy.metadata.get("id", rank))
                    score = float(doc_copy.metadata.get("score") or doc_copy.metadata.get("relevance_score") or (1.0 - rank / max(1, self.config.top_k)))
                    candidate = aggregated.get(doc_id)
                    if candidate is None or score > candidate.score:
                        aggregated[doc_id] = _Candidate(
                            doc_id=doc_id,
                            document=doc_copy,
                            score=score,
                            hop=0,
                            keywords_matched=self._count_keyword_matches(doc_copy.page_content, plan.keywords),
                        )

        return sorted(aggregated.values(), key=lambda item: item.score, reverse=True)[: self.config.top_k]

    async def _expand_graph(self, base: List[_Candidate], plan: QueryPlan) -> List[_Candidate]:
        if not self.graph_store:
            return base

        expanded: Dict[str, _Candidate] = {candidate.doc_id: candidate for candidate in base}
        keywords = plan.keywords

        for candidate in base:
            node_id = candidate.doc_id
            queue: List[_Candidate] = [candidate]
            seen = {node_id}

            while queue:
                current_candidate = queue.pop(0)
                current_id = current_candidate.doc_id
                if current_candidate.hop >= self.config.graph_hops:
                    continue
                try:
                    neighbors = await self.graph_store.get_related_entities(current_id, limit=self.config.graph_branching)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Graph expansion failed for %s: %s", current_id, exc)
                    break

                for neighbor in neighbors:
                    neighbor_id = str(neighbor.get("id"))
                    if not neighbor_id or neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)
                    metadata = dict(neighbor)
                    content = metadata.pop("content", "")
                    hop = current_candidate.hop + 1
                    base_score = max(0.0, current_candidate.score * (0.8 - 0.1 * current_candidate.hop))
                    match_count = self._count_keyword_matches(content, keywords)
                    score = base_score + (0.02 * match_count)
                    metadata.setdefault("id", neighbor_id)
                    doc = Document(page_content=content, metadata=metadata)
                    new_candidate = _Candidate(
                        doc_id=neighbor_id,
                        document=doc,
                        score=score,
                        hop=hop,
                        keywords_matched=match_count,
                    )
                    expanded[neighbor_id] = new_candidate
                    queue.append(new_candidate)

        return list(expanded.values())

    async def _rerank_candidates(
        self,
        candidates: List[_Candidate],
        query: str,
        plan: QueryPlan,
    ) -> List[_Candidate]:
        if not candidates:
            return []

        for candidate in candidates:
            hop_bonus = 0.0
            if candidate.hop == 0:
                hop_bonus = 0.05
            elif candidate.hop == 1:
                hop_bonus = 0.03
            elif candidate.hop == 2:
                hop_bonus = 0.01
            candidate.score = candidate.score + hop_bonus + (0.02 * candidate.keywords_matched)

        candidates.sort(key=lambda item: item.score, reverse=True)

        reranker_mode = self.config.reranker.lower()
        if reranker_mode == "utility" and self.agent:
            try:
                candidates = await self._rerank_with_utility(candidates, query)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Utility reranker failed: %s", exc)
        elif reranker_mode == "none":
            pass

        return sorted(candidates, key=lambda item: item.score, reverse=True)[: self.config.rerank_top_k]

    async def _rerank_with_utility(self, candidates: List[_Candidate], query: str) -> List[_Candidate]:
        assert self.agent  # for type checker
        sample = candidates[: self.config.rerank_top_k]
        payload = {
            "query": query,
            "segments": [
                {
                    "id": cand.doc_id,
                    "hop": cand.hop,
                    "score": cand.score,
                    "content": (cand.document.page_content or "")[:5000],
                    "metadata": cand.document.metadata,
                }
                for cand in sample
            ],
        }
        system_prompt = (
            "Return JSON with a 'scores' list containing objects with keys 'id' and 'score'. "
            "Higher scores indicate higher relevance."
        )
        response = await self.agent.call_utility_model(system=system_prompt, message=json.dumps(payload))
        mapping = self._parse_rerank_scores(response)
        if not mapping:
            return candidates

        for candidate in candidates:
            if candidate.doc_id in mapping:
                candidate.score = float(mapping[candidate.doc_id])
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _parse_rerank_scores(self, text: str) -> Dict[str, float]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}

        if isinstance(data, dict):
            scores = data.get("scores")
        else:
            scores = data

        result: Dict[str, float] = {}
        if isinstance(scores, list):
            for entry in scores:
                if isinstance(entry, dict) and "id" in entry and "score" in entry:
                    doc_id = str(entry["id"]).strip()
                    try:
                        result[doc_id] = float(entry["score"])
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        continue
        return result

    def _package_segments(self, candidates: List[_Candidate]) -> List[RetrievedSegment]:
        if not candidates:
            return []

        budget = max(0, int(self.config.token_budget))
        used_tokens = 0
        packaged: List[RetrievedSegment] = []

        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            content = candidate.document.page_content or ""
            token_cost = approximate_tokens(content)
            if budget and used_tokens + token_cost > budget and packaged:
                break
            used_tokens += token_cost
            metadata = dict(candidate.document.metadata)
            metadata.update(
                {
                    "score": candidate.score,
                    "hop": candidate.hop,
                    "keywords_matched": candidate.keywords_matched,
                }
            )
            dom_path = metadata.get("dom_path") or metadata.get("domPath")
            screenshot = metadata.get("screenshot") or metadata.get("screenshot_url")
            source = metadata.get("source") or metadata.get("document_uri") or "vector"
            packaged.append(
                RetrievedSegment(
                    id=str(metadata.get("id", candidate.doc_id)),
                    content=content,
                    score=candidate.score,
                    metadata=metadata,
                    source=str(source),
                    hop=candidate.hop,
                    dom_path=dom_path,
                    screenshot=screenshot,
                )
            )

        return packaged

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for item in items:
            clean = item.strip()
            if clean and clean not in seen:
                seen.add(clean)
                result.append(clean)
        return result

    @staticmethod
    def _count_keyword_matches(text: str, keywords: Sequence[str]) -> int:
        if not text or not keywords:
            return 0
        lowered = text.lower()
        return sum(1 for keyword in keywords if keyword.lower() in lowered)

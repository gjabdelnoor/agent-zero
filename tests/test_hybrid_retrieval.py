import asyncio
import os
import sys
import time

import numpy as np
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid import HybridRetrievalConfig, HybridRetriever


class DummyEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        seed = max(1, len(text))
        return [float(((seed + i) % 11) / 10) for i in range(8)]


class StubDocStore:
    def __init__(self, docs: list[Document]):
        self._docs = {doc.metadata["id"]: doc for doc in docs}

    def get_all_docs(self):
        return self._docs


class StubVectorStore:
    def __init__(self):
        self.embeddings = DummyEmbeddings()
        self._docs = [
            Document(page_content=f"Synthetic knowledge chunk {idx}", metadata={"id": f"chunk-{idx}", "source": "vector"})
            for idx in range(6)
        ]
        self.db = StubDocStore(self._docs)

    async def search_by_similarity_threshold(self, query: str, limit: int, threshold: float, filter: str = ""):
        return self._docs[:limit]


class StubGraphStore:
    async def get_related_entities(self, entity_id: str, limit: int = 10):
        return [
            {
                "id": f"{entity_id}-neighbor-{idx}",
                "content": f"Additional context {idx} for {entity_id}",
                "source": "graph",
                "dom_path": f"#node-{idx}",
            }
            for idx in range(min(limit, 2))
        ]


def test_hybrid_retrieval_p95_under_4s():
    retriever = HybridRetriever(
        agent=None,
        vector_store=StubVectorStore(),
        graph_store=StubGraphStore(),
        config=HybridRetrievalConfig(top_k=3, graph_hops=2, rerank_top_k=5, token_budget=600),
    )

    async def run_once():
        result = await retriever.retrieve("test query about hybrid retrieval")
        assert result.segments, "Expected at least one retrieved segment"
        assert result.plan.rewritten_queries, "Query planner should provide rewritten queries"
        assert all(segment.hop >= 0 for segment in result.segments)

    durations: list[float] = []
    for _ in range(10):
        start = time.perf_counter()
        asyncio.run(run_once())
        durations.append(time.perf_counter() - start)

    percentile = float(np.percentile(np.asarray(durations), 95))
    assert percentile < 4.0, f"p95 latency {percentile:.2f}s exceeds 4s budget"

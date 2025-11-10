"""Utility script to migrate historical FAISS memories into Neo4j."""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from typing import Dict, Iterable, List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

import initialize
import models
from python.helpers import files, guids
from python.helpers.memory import Memory, abs_db_dir
from python.helpers.neo4j_memory import Neo4jMemory


async def migrate_memory(memory_subdir: str = "default") -> int:
    """Migrate documents from a FAISS index into Neo4j."""

    agent_config = initialize.initialize_agent()

    memory_dir = abs_db_dir(memory_subdir)
    if not files.exists(memory_dir, "index.faiss"):
        print(f"No FAISS index found for '{memory_subdir}'. Nothing to migrate.")
        return 0

    embeddings_model = models.get_embedding_model(
        agent_config.embeddings_model.provider,
        agent_config.embeddings_model.name,
        **agent_config.embeddings_model.build_kwargs(),
    )

    embeddings_dir = files.get_abs_path("memory", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    store = LocalFileStore(embeddings_dir)
    embeddings_model_id = files.safe_file_name(
        agent_config.embeddings_model.provider + "_" + agent_config.embeddings_model.name
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_model, store, namespace=embeddings_model_id
    )

    faiss_db = FAISS.load_local(
        folder_path=memory_dir,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE,
    )

    all_docs: Dict[str, Document] = faiss_db.docstore._dict  # type: ignore[attr-defined]
    print(f"Found {len(all_docs)} documents in FAISS for '{memory_subdir}'.")

    neo4j_memory = await Neo4jMemory.get(agent_config, memory_subdir)

    documents: List[Document] = []
    ids: List[str] = []
    for doc_id, doc in all_docs.items():
        metadata = doc.metadata.copy()
        metadata.setdefault("id", doc_id or guids.generate_id(10))
        metadata.setdefault("memory_subdir", memory_subdir)
        metadata.setdefault("timestamp", Memory.get_timestamp())
        if not metadata.get("area"):
            metadata["area"] = Memory.Area.MAIN.value
        documents.append(Document(page_content=doc.page_content, metadata=metadata))
        ids.append(metadata["id"])

    if documents:
        await neo4j_memory.aadd_documents(documents=documents, ids=ids)

    await _create_relationships_from_metadata(neo4j_memory, documents)

    print(f"Migration complete! Migrated {len(documents)} documents to Neo4j.")
    return len(documents)


async def _create_relationships_from_metadata(
    neo4j_memory: Neo4jMemory, docs: Iterable[Document]
) -> None:
    by_area: Dict[str, List[Document]] = defaultdict(list)
    for doc in docs:
        area = doc.metadata.get("area", Memory.Area.MAIN.value)
        by_area[area].append(doc)

    for area, documents in by_area.items():
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get("timestamp", ""),
        )
        for first, second in zip(sorted_docs, sorted_docs[1:]):
            await neo4j_memory.create_relationship(
                first.metadata["id"],
                second.metadata["id"],
                "NEXT",
                {"area": area},
            )


def main(memory_subdir: str = "default") -> None:
    asyncio.run(migrate_memory(memory_subdir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate FAISS memories into Neo4j")
    parser.add_argument("memory_subdir", nargs="?", default="default")
    args = parser.parse_args()
    main(args.memory_subdir)


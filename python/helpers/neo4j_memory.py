from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.documents import Document
from neo4j import AsyncDriver, AsyncGraphDatabase

import models
from agent import Agent, AgentConfig

try:  # The optional import keeps unit tests runnable without the dependency.
    from langchain_community.vectorstores.neo4j_vector import Neo4jVector
except Exception as exc:  # pragma: no cover - dependency errors are surfaced at runtime
    raise RuntimeError(
        "Neo4j vector store dependencies are missing. Ensure langchain-community "
        "extras are installed."
    ) from exc


class Neo4jMemoryError(RuntimeError):
    """Raised when memory operations fail."""


class Neo4jMemory:
    """Graph-based memory system backed by Neo4j vector search."""

    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions"
        INSTRUMENTS = "instruments"

    _drivers: Dict[str, AsyncDriver] = {}
    _vector_stores: Dict[str, Neo4jVector] = {}

    def __init__(self, driver: AsyncDriver, vector_store: Neo4jVector, memory_subdir: str):
        self.driver = driver
        self.vector_store = vector_store
        self.memory_subdir = memory_subdir

    @staticmethod
    async def get(agent_or_config: Union[Agent, AgentConfig], memory_subdir: str) -> "Neo4jMemory":
        """Return a Neo4j memory instance for ``memory_subdir``."""

        if memory_subdir not in Neo4jMemory._drivers:
            await Neo4jMemory._initialize(agent_or_config, memory_subdir)

        return Neo4jMemory(
            driver=Neo4jMemory._drivers[memory_subdir],
            vector_store=Neo4jMemory._vector_stores[memory_subdir],
            memory_subdir=memory_subdir,
        )

    @staticmethod
    async def _initialize(agent_or_config: Union[Agent, AgentConfig], memory_subdir: str) -> None:
        """Initialise driver, constraints and vector index for ``memory_subdir``."""

        config = agent_or_config.config if isinstance(agent_or_config, Agent) else agent_or_config

        neo4j_uri = config.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = config.neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = config.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        await driver.verify_connectivity()

        async with driver.session() as session:
            await session.run(
                """
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """
            )
            await session.run(
                """
                CREATE INDEX entity_timestamp IF NOT EXISTS
                FOR (e:Entity) ON (e.timestamp)
                """
            )
            await session.run(
                """
                CREATE INDEX entity_area IF NOT EXISTS
                FOR (e:Entity) ON (e.area)
                """
            )
            await session.run(
                """
                CREATE INDEX entity_memory_subdir IF NOT EXISTS
                FOR (e:Entity) ON (e.memory_subdir)
                """
            )

        embeddings_model = models.get_embedding_model(
            config.embeddings_model.provider,
            config.embeddings_model.name,
            **config.embeddings_model.build_kwargs(),
        )

        embedding_dims = int(os.getenv("NEO4J_VECTOR_DIMENSIONS", getattr(config, "neo4j_vector_dimensions", 4096)))

        async with driver.session() as session:
            await session.run(
                """
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dims,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """,
                dims=embedding_dims,
            )

        try:
            vector_store = await Neo4jVector.afrom_existing_index(
                embedding=embeddings_model,
                url=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
                index_name="entity_embeddings",
                node_label="Entity",
                embedding_node_property="embedding",
                text_node_property="content",
            )
        except Exception:
            vector_store = await Neo4jVector.afrom_documents(
                documents=[],
                embedding=embeddings_model,
                url=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
                index_name="entity_embeddings",
                node_label="Entity",
                embedding_node_property="embedding",
                text_node_property="content",
            )

        Neo4jMemory._drivers[memory_subdir] = driver
        Neo4jMemory._vector_stores[memory_subdir] = vector_store

    async def close(self) -> None:
        driver = Neo4jMemory._drivers.pop(self.memory_subdir, None)
        Neo4jMemory._vector_stores.pop(self.memory_subdir, None)
        if driver:
            await driver.close()

    async def aadd_documents(self, documents: List[Document], ids: Iterable[str]) -> List[str]:
        ids_list = list(ids)
        for doc, doc_id in zip(documents, ids_list):
            doc.metadata.setdefault("id", doc_id)
            doc.metadata.setdefault("memory_subdir", self.memory_subdir)
        await self.vector_store.aadd_documents(documents=documents, ids=ids_list)
        return ids_list

    async def aadd_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Iterable[str]) -> List[str]:
        ids_list = list(ids)
        for metadata, doc_id in zip(metadatas, ids_list):
            metadata.setdefault("id", doc_id)
            metadata.setdefault("memory_subdir", self.memory_subdir)
        await self.vector_store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids_list)
        return ids_list

    async def asearch(
        self,
        query: str,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: Optional[float] = None,
        filter: Optional[Any] = None,
    ) -> List[Document]:
        docs = await self.vector_store.asearch(query, search_type="similarity", k=max(k, 5))
        filtered: List[Document] = []
        for doc in docs:
            metadata = doc.metadata or {}
            if metadata.get("memory_subdir") not in (self.memory_subdir, None):
                continue
            score = metadata.get("score")
            if score_threshold is not None and score is not None and score < score_threshold:
                continue
            if filter and callable(filter) and not filter(metadata):
                continue
            filtered.append(doc)
        return filtered[:k]

    async def adelete(self, ids: Iterable[str]) -> None:
        ids_list = list(ids)
        await self.vector_store.adelete(ids=ids_list)
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids AND e.memory_subdir = $memory_subdir
                DETACH DELETE e
                """,
                ids=ids_list,
                memory_subdir=self.memory_subdir,
            )

    async def aget_by_ids(self, ids: Iterable[str]) -> List[Document]:
        ids_list = list(ids)
        if not ids_list:
            return []
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids AND e.memory_subdir = $memory_subdir
                RETURN e.id AS id, e.content AS content, e
                """,
                ids=ids_list,
                memory_subdir=self.memory_subdir,
            )
            docs: List[Document] = []
            async for record in result:
                metadata = dict(record["e"])
                docs.append(Document(page_content=record["content"], metadata=metadata))
            return docs

    async def get_all_docs(self) -> Dict[str, Document]:
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {memory_subdir: $memory_subdir})
                RETURN e.id AS id, e.content AS content, e
                """,
                memory_subdir=self.memory_subdir,
            )
            docs: Dict[str, Document] = {}
            async for record in result:
                metadata = dict(record["e"])
                docs[record["id"]] = Document(
                    page_content=record["content"], metadata=metadata
                )
            return docs

    async def insert_text(self, text: str, metadata: Dict[str, Any]) -> str:
        doc_id = metadata.get("id") or metadata.get("guid")
        if not doc_id:
            raise Neo4jMemoryError("Metadata must include an 'id' before insertion")
        await self.aadd_texts(texts=[text], metadatas=[metadata], ids=[doc_id])
        return doc_id

    async def delete_documents_by_ids(self, ids: Iterable[str]) -> List[Document]:
        docs = await self.aget_by_ids(ids)
        if docs:
            await self.adelete([doc.metadata["id"] for doc in docs])
        return docs

    async def create_relationship(
        self, from_id: str, to_id: str, rel_type: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        properties = properties or {}
        async with self.driver.session() as session:
            await session.run(
                f"""
                MATCH (a:Entity {{id: $from_id, memory_subdir: $memory_subdir}}),
                      (b:Entity {{id: $to_id, memory_subdir: $memory_subdir}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                """,
                from_id=from_id,
                to_id=to_id,
                memory_subdir=self.memory_subdir,
                properties=properties,
            )

    async def get_related_entities(
        self, entity_id: str, rel_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        rel_pattern = f"[:{rel_type}]" if rel_type else ""
        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH (e:Entity {{id: $entity_id, memory_subdir: $memory_subdir}})
                      -{rel_pattern}-(related:Entity {{memory_subdir: $memory_subdir}})
                RETURN related.id AS id, related.content AS content,
                       related.timestamp AS timestamp
                LIMIT $limit
                """,
                entity_id=entity_id,
                memory_subdir=self.memory_subdir,
                limit=limit,
            )
            related: List[Dict[str, Any]] = []
            async for record in result:
                related.append(dict(record))
            return related

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    async def list_memory_subdirs(agent_or_config: Union[Agent, AgentConfig]) -> List[str]:
        config = agent_or_config.config if isinstance(agent_or_config, Agent) else agent_or_config

        neo4j_uri = config.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = config.neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = config.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        await driver.verify_connectivity()
        try:
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    RETURN DISTINCT e.memory_subdir AS memory_subdir
                    """
                )
                subdirs: List[str] = []
                async for record in result:
                    value = record.get("memory_subdir")
                    if value:
                        subdirs.append(value)
                if "default" not in subdirs:
                    subdirs.insert(0, "default")
                return subdirs
        finally:
            await driver.close()


__all__ = ["Neo4jMemory", "Neo4jMemoryError"]


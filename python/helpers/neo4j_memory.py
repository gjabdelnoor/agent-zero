"""Neo4j-backed memory implementation."""

from __future__ import annotations
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from neo4j import AsyncDriver, AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import Neo4jError

import models
from agent import Agent, AgentConfig
from python.helpers import files, guids, knowledge_import
from python.helpers.log import LogItem
from python.helpers.print_style import PrintStyle

LOGGER = logging.getLogger(__name__)


def _safe_index_name(name: str) -> str:
    """Return a Neo4j-safe index name for a memory subdirectory."""

    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return f"entity_embeddings_{safe}"[:62]


def _safe_label(name: str) -> str:
    """Return a Neo4j label that uniquely maps to the memory subdirectory."""

    safe = re.sub(r"[^0-9a-zA-Z]", "_", name)
    if not safe:
        safe = "Default"
    if not safe[0].isalpha():
        safe = f"M_{safe}"
    return f"Memory_{safe}"[:50]


def _safe_relationship_type(rel_type: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", rel_type or "RELATED")
    if not cleaned:
        cleaned = "RELATED"
    if cleaned[0].isdigit():
        cleaned = f"R_{cleaned}"
    return cleaned.upper()[:50]


def _abs_db_dir(memory_subdir: str) -> str:
    """Replicates memory.abs_db_dir to avoid circular imports."""

    if memory_subdir.startswith("projects/"):
        from python.helpers.projects import get_project_meta_folder

        return files.get_abs_path(get_project_meta_folder(memory_subdir[9:]), "memory")
    return files.get_abs_path("memory", memory_subdir)


def _default_similarity_normalizer(val: float) -> float:
    res = (1 + val) / 2
    return max(0.0, min(1.0, res))


def _get_comparator(condition: str):
    from simpleeval import simple_eval

    def comparator(data: Dict[str, Any]):
        try:
            return bool(simple_eval(condition, names=data))
        except Exception as exc:  # pragma: no cover - defensive
            PrintStyle.error(f"Error evaluating condition: {exc}")
            return False

    return comparator


@dataclass
class _Neo4jResources:
    driver: AsyncDriver
    vector_store: Neo4jVector
    node_label: str
    index_name: str


class Neo4jMemory:
    """Graph-based memory system backed by Neo4j."""

    _resources: Dict[str, _Neo4jResources] = {}
    _instances: Dict[str, "Neo4jMemory"] = {}

    def __init__(self, resources: _Neo4jResources, memory_subdir: str):
        self._resources = resources
        self.memory_subdir = memory_subdir

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    async def get(
        cls,
        agent: Agent,
        memory_subdir: str,
        *,
        log_item: LogItem | None = None,
        knowledge_subdirs: Optional[Iterable[str]] = None,
        preload_knowledge: bool = True,
    ) -> "Neo4jMemory":
        config = agent.config
        instance = await cls._ensure_instance(
            config,
            memory_subdir,
            log_item=log_item,
        )

        if preload_knowledge and knowledge_subdirs:
            await instance.preload_knowledge(log_item, list(knowledge_subdirs), memory_subdir)

        return instance

    @classmethod
    async def get_for_config(
        cls,
        config: AgentConfig,
        memory_subdir: str,
        *,
        log_item: LogItem | None = None,
        knowledge_subdirs: Optional[Iterable[str]] = None,
        preload_knowledge: bool = True,
    ) -> "Neo4jMemory":
        instance = await cls._ensure_instance(config, memory_subdir, log_item=log_item)
        if preload_knowledge and knowledge_subdirs:
            await instance.preload_knowledge(log_item, list(knowledge_subdirs), memory_subdir)
        return instance

    @classmethod
    async def _ensure_instance(
        cls,
        config: AgentConfig,
        memory_subdir: str,
        *,
        log_item: LogItem | None = None,
    ) -> "Neo4jMemory":
        if memory_subdir not in cls._instances:
            resources = await cls._initialize_resources(config, memory_subdir, log_item=log_item)
            cls._instances[memory_subdir] = Neo4jMemory(resources, memory_subdir)
        return cls._instances[memory_subdir]

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    @classmethod
    async def _initialize_resources(
        cls,
        config: AgentConfig,
        memory_subdir: str,
        *,
        log_item: LogItem | None = None,
    ) -> _Neo4jResources:
        if memory_subdir in cls._resources:
            return cls._resources[memory_subdir]

        neo4j_uri = config.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = config.neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = config.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        neo4j_database = getattr(config, "neo4j_database", "") or os.getenv("NEO4J_DATABASE")

        driver_kwargs: Dict[str, Any] = {}
        pool_size = os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE")
        if pool_size:
            driver_kwargs["max_connection_pool_size"] = int(pool_size)
        acquisition_timeout = os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT")
        if acquisition_timeout:
            driver_kwargs["connection_acquisition_timeout"] = int(acquisition_timeout)

        driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password),
            database=neo4j_database or None,
            **driver_kwargs,
        )

        await driver.verify_connectivity()

        node_label = _safe_label(memory_subdir)
        index_name = _safe_index_name(memory_subdir)
        embedding_dims = int(
            getattr(config, "neo4j_vector_dimensions", 0)
            or os.getenv("NEO4J_VECTOR_DIMENSIONS", "4096")
        )

        await cls._prepare_schema(
            driver,
            node_label=node_label,
            index_name=index_name,
            embedding_dims=embedding_dims,
            memory_subdir=memory_subdir,
        )

        embeddings_model = models.get_embedding_model(
            config.embeddings_model.provider,
            config.embeddings_model.name,
            **config.embeddings_model.build_kwargs(),
        )

        vector_store = Neo4jVector(
            embedding=embeddings_model,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database or None,
            index_name=index_name,
            node_label=node_label,
            embedding_node_property="embedding",
            text_node_property="content",
            distance_strategy=DistanceStrategy.COSINE,
            relevance_score_fn=_default_similarity_normalizer,
        )

        resources = _Neo4jResources(
            driver=driver, vector_store=vector_store, node_label=node_label, index_name=index_name
        )
        cls._resources[memory_subdir] = resources

        abs_dir = _abs_db_dir(memory_subdir)
        os.makedirs(abs_dir, exist_ok=True)
        if log_item:
            log_item.update(heading=f"Connected to Neo4j memory '/{memory_subdir}'")

        return resources

    @staticmethod
    async def _prepare_schema(
        driver: AsyncDriver,
        *,
        node_label: str,
        index_name: str,
        embedding_dims: int,
        memory_subdir: str,
    ) -> None:
        async with driver.session() as session:
            await session.run(
                f"""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (e:{node_label}) REQUIRE e.id IS UNIQUE
                """
            )

            await session.run(
                f"""
                CREATE INDEX IF NOT EXISTS
                FOR (e:{node_label}) ON (e.timestamp)
                """
            )

            await session.run(
                f"""
                CREATE INDEX IF NOT EXISTS
                FOR (e:{node_label}) ON (e.area)
                """
            )

            await session.run(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (e:{node_label}) ON (e.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dims,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dims=embedding_dims,
            )

            await session.run(
                f"""
                CREATE INDEX IF NOT EXISTS
                FOR (e:{node_label}) ON (e.memory_subdir)
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def preload_knowledge(
        self,
        log_item: LogItem | None,
        kn_dirs: List[str],
        memory_subdir: str,
    ) -> None:
        if log_item:
            log_item.update(heading="Preloading knowledge...")

        db_dir = _abs_db_dir(memory_subdir)
        os.makedirs(db_dir, exist_ok=True)
        index_path = files.get_abs_path(db_dir, "knowledge_import.json")

        index: dict[str, knowledge_import.KnowledgeImport]
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        else:
            index = {}

        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        for file in list(index.keys()):
            entry = index[file]
            ids = entry.get("ids", [])
            if entry.get("state") in {"changed", "removed"} and ids:
                await self.delete_documents_by_ids(ids)
            if entry.get("state") == "changed":
                entry["ids"] = await self.insert_documents(entry.get("documents", []))

        index = {k: v for k, v in index.items() if v.get("state") != "removed"}
        for file in index:
            index[file].pop("documents", None)
            index[file].pop("state", None)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)

    def _preload_knowledge_folders(
        self,
        log_item: LogItem | None,
        kn_dirs: List[str],
        index: dict[str, knowledge_import.KnowledgeImport],
    ) -> dict[str, knowledge_import.KnowledgeImport]:
        from python.helpers.memory import Memory  # local import to avoid cycle

        for kn_dir in kn_dirs:
            for area in Memory.Area:
                index = knowledge_import.load_knowledge(
                    log_item,
                    files.get_abs_path("knowledge", kn_dir, area.value),
                    index,
                    {"area": area.value},
                )

        index = knowledge_import.load_knowledge(
            log_item,
            files.get_abs_path("instruments"),
            index,
            {"area": Memory.Area.INSTRUMENTS.value},
            filename_pattern="**/*.md",
        )
        return index

    async def get_document_by_id(self, id: str) -> Optional[Document]:
        docs = await self.aget_by_ids([id])
        return docs[0] if docs else None

    async def get_all_documents(self) -> Dict[str, Document]:
        query = f"""
            MATCH (e:{self._resources.node_label})
            WHERE e.memory_subdir = $memory_subdir
            RETURN e.id AS id, e.content AS content, e
        """

        docs: Dict[str, Document] = {}
        async with self._resources.driver.session() as session:
            result = await session.run(query, memory_subdir=self.memory_subdir)
            async for record in result:
                metadata = dict(record["e"].items())
                metadata["id"] = record["id"]
                metadata.setdefault("memory_subdir", self.memory_subdir)
                doc = Document(page_content=record["content"], metadata=metadata)
                docs[record["id"]] = doc
        return docs

    async def search_similarity_threshold(
        self,
        query: str,
        limit: int,
        threshold: float,
        filter: str = "",
    ) -> List[Document]:
        comparator = _get_comparator(filter) if filter else None
        docs = await self._resources.vector_store.asearch(
            query,
            search_type="similarity_score_threshold",
            k=limit,
            score_threshold=threshold,
        )

        filtered: List[Document] = []
        for doc in docs:
            metadata = doc.metadata or {}
            metadata.setdefault("memory_subdir", metadata.get("memory_subdir", self.memory_subdir))
            if metadata.get("memory_subdir") != self.memory_subdir:
                continue
            if comparator and not comparator(metadata):
                continue
            filtered.append(doc)

        return filtered[:limit]

    async def delete_documents_by_query(
        self,
        query: str,
        threshold: float,
        filter: str = "",
    ) -> List[Document]:
        removed = []
        batch_size = 100

        while True:
            docs = await self.search_similarity_threshold(query, batch_size, threshold, filter)
            if not docs:
                break
            removed.extend(docs)
            ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
            if ids:
                await self._delete_by_ids(ids)
            if len(docs) < batch_size:
                break

        return removed

    async def delete_documents_by_ids(self, ids: List[str]) -> List[Document]:
        return await self._delete_by_ids(ids)

    async def _delete_by_ids(self, ids: Sequence[str]) -> List[Document]:
        docs = await self.aget_by_ids(ids)
        if not docs:
            return []

        async with self._resources.driver.session() as session:
            await session.run(
                f"""
                MATCH (e:{self._resources.node_label})
                WHERE e.memory_subdir = $memory_subdir AND e.id IN $ids
                DETACH DELETE e
                """,
                memory_subdir=self.memory_subdir,
                ids=list(ids),
            )
        return docs

    async def insert_text(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        doc = Document(text, metadata=metadata or {})
        ids = await self.insert_documents([doc])
        return ids[0]

    async def insert_documents(self, docs: List[Document]) -> List[str]:
        ids: List[str] = []
        timestamp = self.get_timestamp()
        for doc in docs:
            doc_id = await self._generate_doc_id()
            ids.append(doc_id)
            doc.metadata["id"] = doc_id
            doc.metadata.setdefault("timestamp", timestamp)
            doc.metadata.setdefault("area", "main")
            doc.metadata["memory_subdir"] = self.memory_subdir

        if ids:
            await self._resources.vector_store.aadd_documents(docs, ids=ids)

            async with self._resources.driver.session() as session:
                await session.run(
                    f"""
                    MATCH (e:{self._resources.node_label})
                    WHERE e.id IN $ids
                    SET e.memory_subdir = $memory_subdir
                    """,
                    ids=ids,
                    memory_subdir=self.memory_subdir,
                )

        return ids

    async def update_documents(self, docs: List[Document]) -> List[str]:
        ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
        if ids:
            await self._delete_by_ids(ids)
        return await self.insert_documents(docs)

    async def aget_by_ids(self, ids: Sequence[str]) -> List[Document]:
        if not ids:
            return []
        query = f"""
            MATCH (e:{self._resources.node_label})
            WHERE e.memory_subdir = $memory_subdir AND e.id IN $ids
            RETURN e.id AS id, e.content AS content, e
        """
        async with self._resources.driver.session() as session:
            result = await session.run(query, memory_subdir=self.memory_subdir, ids=list(ids))
            fetched: Dict[str, Document] = {}
            async for record in result:
                metadata = dict(record["e"].items())
                metadata["id"] = record["id"]
                metadata.setdefault("memory_subdir", self.memory_subdir)
                fetched[record["id"]] = Document(page_content=record["content"], metadata=metadata)

        ordered = [fetched[id] for id in ids if id in fetched]
        return ordered

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        properties = properties or {}
        safe_type = _safe_relationship_type(rel_type)
        async with self._resources.driver.session() as session:
            await session.run(
                f"""
                MATCH (a:{self._resources.node_label} {{id: $from_id, memory_subdir: $memory_subdir}}),
                      (b:{self._resources.node_label} {{id: $to_id, memory_subdir: $memory_subdir}})
                MERGE (a)-[r:{safe_type}]->(b)
                SET r += $properties
                """,
                from_id=from_id,
                to_id=to_id,
                properties=properties,
                memory_subdir=self.memory_subdir,
            )

    async def get_related_entities(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        rel_pattern = f":{_safe_relationship_type(rel_type)}" if rel_type else ""
        query = f"""
            MATCH (e:{self._resources.node_label} {{id: $entity_id, memory_subdir: $memory_subdir}})-[r{rel_pattern}]-(related:{self._resources.node_label})
            WHERE related.memory_subdir = $memory_subdir
            RETURN related.id as id, related.content as content, type(r) as relationship, related.timestamp as timestamp
            LIMIT $limit
        """
        async with self._resources.driver.session() as session:
            result = await session.run(
                query,
                entity_id=entity_id,
                limit=limit,
                memory_subdir=self.memory_subdir,
            )
            related: List[Dict[str, Any]] = []
            async for record in result:
                related.append(dict(record))
        return related

    async def close(self) -> None:
        await self._resources.driver.close()

    @classmethod
    async def reset(cls, memory_subdir: Optional[str] = None) -> None:
        targets = [memory_subdir] if memory_subdir else list(cls._resources.keys())
        for target in targets:
            resources = cls._resources.pop(target, None)
            if resources:
                try:
                    await resources.driver.close()
                except Neo4jError:  # pragma: no cover - defensive
                    LOGGER.debug("Failed to close Neo4j driver", exc_info=True)
            cls._instances.pop(target, None)

    async def _generate_doc_id(self) -> str:
        while True:
            doc_id = guids.generate_id(10)
            docs = await self.aget_by_ids([doc_id])
            if not docs:
                return doc_id

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_memory_subdirs(config: AgentConfig) -> List[str]:
    subdirs = {key for key in Neo4jMemory._resources.keys() if key}

    uri = config.neo4j_uri or os.getenv("NEO4J_URI")
    if not uri:
        return sorted(subdirs)

    username = config.neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
    password = config.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
    database = getattr(config, "neo4j_database", "") or os.getenv("NEO4J_DATABASE")

    driver = GraphDatabase.driver(uri, auth=(username, password), database=database or None)
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE exists(n.memory_subdir) RETURN DISTINCT n.memory_subdir AS memory_subdir"
            )
            for record in result:
                sub = record.get("memory_subdir")
                if sub:
                    subdirs.add(sub)
    except Neo4jError:
        LOGGER.debug("Failed to list Neo4j memory subdirectories", exc_info=True)
    finally:
        driver.close()

    return sorted(subdirs)


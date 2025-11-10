from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, List

from langchain_core.documents import Document

from agent import Agent, AgentContext
from python.helpers import files, guids, knowledge_import
from python.helpers.log import LogItem
from python.helpers.neo4j_memory import Neo4jMemory
from python.helpers.print_style import PrintStyle


class Memory:
    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions"
        INSTRUMENTS = "instruments"

    index: dict[str, "Memory"] = {}

    def __init__(self, backend: Neo4jMemory, memory_subdir: str):
        self.backend = backend
        self.memory_subdir = memory_subdir

    @staticmethod
    async def get(agent: Agent) -> "Memory":
        memory_subdir = get_agent_memory_subdir(agent)
        if Memory.index.get(memory_subdir) is None:
            log_item = agent.context.log.log(
                type="util",
                heading=f"Initializing Neo4j memory in '/{memory_subdir}'",
            )
            backend = await Neo4jMemory.get(agent, memory_subdir)
            wrapper = Memory(backend=backend, memory_subdir=memory_subdir)
            if agent.config.knowledge_subdirs:
                await wrapper.preload_knowledge(
                    log_item, agent.config.knowledge_subdirs, memory_subdir
                )
            Memory.index[memory_subdir] = wrapper
        return Memory.index[memory_subdir]

    @staticmethod
    async def get_by_subdir(
        memory_subdir: str,
        log_item: LogItem | None = None,
        preload_knowledge: bool = True,
    ) -> "Memory":
        instance = Memory.index.get(memory_subdir)
        if not instance:
            import initialize

            agent_config = initialize.initialize_agent()
            backend = await Neo4jMemory.get(agent_config, memory_subdir)
            instance = Memory(backend=backend, memory_subdir=memory_subdir)
            if preload_knowledge and agent_config.knowledge_subdirs:
                await instance.preload_knowledge(
                    log_item, agent_config.knowledge_subdirs, memory_subdir
                )
            Memory.index[memory_subdir] = instance
        return instance

    @staticmethod
    async def reload(agent: Agent) -> "Memory":
        memory_subdir = agent.config.memory_subdir or "default"
        if Memory.index.get(memory_subdir):
            instance = Memory.index.pop(memory_subdir)
            await instance.backend.close()
        return await Memory.get(agent)

    async def preload_knowledge(
        self, log_item: LogItem | None, kn_dirs: list[str], memory_subdir: str
    ) -> None:
        if log_item:
            log_item.update(heading="Preloading knowledge...")

        db_dir = abs_db_dir(memory_subdir)
        os.makedirs(db_dir, exist_ok=True)

        index_path = files.get_abs_path(db_dir, "knowledge_import.json")
        index: dict[str, knowledge_import.KnowledgeImport] = {}
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        for file in index:
            if index[file]["state"] in ["changed", "removed"] and index[file].get("ids", []):
                await self.delete_documents_by_ids(index[file]["ids"])
            if index[file]["state"] == "changed":
                index[file]["ids"] = await self.insert_documents(index[file]["documents"])

        index = {k: v for k, v in index.items() if v["state"] != "removed"}

        for file in index:
            index[file].pop("documents", None)
            index[file].pop("state", None)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)

    def _preload_knowledge_folders(
        self,
        log_item: LogItem | None,
        kn_dirs: list[str],
        index: dict[str, knowledge_import.KnowledgeImport],
    ) -> dict[str, knowledge_import.KnowledgeImport]:
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

    async def get_document_by_id(self, id: str) -> Document | None:
        docs = await self.backend.aget_by_ids([id])
        return docs[0] if docs else None

    async def get_documents_by_ids(self, ids: Iterable[str]) -> list[Document]:
        return await self.backend.aget_by_ids(ids)

    async def get_all_docs(self) -> dict[str, Document]:
        return await self.backend.get_all_docs()

    async def search_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = ""
    ) -> List[Document]:
        comparator = Memory._get_comparator(filter) if filter else None
        return await self.backend.asearch(
            query,
            search_type="similarity",
            k=limit,
            score_threshold=threshold,
            filter=comparator,
        )

    async def delete_documents_by_query(
        self, query: str, threshold: float, filter: str = ""
    ) -> List[Document]:
        k = 100
        removed: List[Document] = []

        while True:
            docs = await self.search_similarity_threshold(
                query, limit=k, threshold=threshold, filter=filter
            )
            if not docs:
                break
            removed.extend(docs)
            ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
            if ids:
                await self.backend.adelete(ids)
            if len(docs) < k:
                break
        return removed

    async def delete_documents_by_ids(self, ids: Iterable[str]) -> List[Document]:
        return await self.backend.delete_documents_by_ids(ids)

    async def insert_text(self, text: str, metadata: dict | None = None) -> str:
        metadata = metadata.copy() if metadata else {}
        doc = Document(page_content=text, metadata=metadata)
        ids = await self.insert_documents([doc])
        return ids[0]

    async def insert_documents(self, docs: List[Document]) -> List[str]:
        ids = [self._generate_doc_id() for _ in docs]
        timestamp = Memory.get_timestamp()
        for doc, id in zip(docs, ids):
            doc.metadata["id"] = id
            doc.metadata.setdefault("timestamp", timestamp)
            if not doc.metadata.get("area"):
                doc.metadata["area"] = Memory.Area.MAIN.value
            doc.metadata["memory_subdir"] = self.memory_subdir
        await self.backend.aadd_documents(documents=docs, ids=ids)
        return ids

    async def update_documents(self, docs: List[Document]) -> List[str]:
        ids = [doc.metadata["id"] for doc in docs if doc.metadata.get("id")]
        if ids:
            await self.backend.adelete(ids)
        for doc in docs:
            doc.metadata.setdefault("memory_subdir", self.memory_subdir)
        await self.backend.aadd_documents(documents=docs, ids=ids)
        return ids

    def _generate_doc_id(self) -> str:
        return guids.generate_id(10)

    @staticmethod
    def _get_comparator(condition: str):
        def comparator(data: dict[str, Any]):
            try:
                from simpleeval import simple_eval

                return bool(simple_eval(condition, names=data))
            except Exception as exc:  # pragma: no cover - best effort filtering
                PrintStyle.error(f"Error evaluating condition: {exc}")
                return False

        return comparator

    @staticmethod
    def format_docs_plain(docs: list[Document]) -> list[str]:
        result = []
        for doc in docs:
            text = ""
            for k, v in doc.metadata.items():
                text += f"{k}: {v}\n"
            text += f"Content: {doc.page_content}"
            result.append(text)
        return result

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_custom_knowledge_subdir_abs(agent: Agent) -> str:
    for dir in agent.config.knowledge_subdirs:
        if dir != "default":
            return files.get_abs_path("knowledge", dir)
    raise Exception("No custom knowledge subdir set")


def reload() -> None:
    Memory.index = {}


def abs_db_dir(memory_subdir: str) -> str:
    if memory_subdir.startswith("projects/"):
        from python.helpers.projects import get_project_meta_folder

        return files.get_abs_path(get_project_meta_folder(memory_subdir[9:]), "memory")
    return files.get_abs_path("memory", memory_subdir)


def get_memory_subdir_abs(agent: Agent) -> str:
    subdir = get_agent_memory_subdir(agent)
    return abs_db_dir(subdir)


def get_agent_memory_subdir(agent: Agent) -> str:
    return get_context_memory_subdir(agent.context)


def get_context_memory_subdir(context: AgentContext) -> str:
    from python.helpers.projects import get_context_memory_subdir as get_project_memory_subdir

    memory_subdir = get_project_memory_subdir(context)
    if memory_subdir:
        return memory_subdir
    return context.config.memory_subdir or "default"


def get_existing_memory_subdirs() -> list[str]:
    try:
        from python.helpers.projects import (
            get_project_meta_folder,
            get_projects_parent_folder,
        )

        subdirs = files.get_subdirectories("memory", exclude="embeddings")
        project_subdirs = files.get_subdirectories(get_projects_parent_folder())
        for project_subdir in project_subdirs:
            meta_folder = get_project_meta_folder(project_subdir)
            if files.exists(meta_folder, "memory"):
                subdirs.append(f"projects/{project_subdir}")

        if "default" not in subdirs:
            subdirs.insert(0, "default")
        return subdirs
    except Exception as exc:
        PrintStyle.error(f"Failed to get memory subdirectories: {exc}")
        return ["default"]


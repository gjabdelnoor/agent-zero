import os
import sys
from pathlib import Path

import pytest

try:
    import neo4j  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.skip("Neo4j Python driver is not installed", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent import AgentConfig
from models import ModelConfig, ModelType
from python.helpers.neo4j_memory import Neo4jMemory

NEO4J_TEST_URI = os.getenv("NEO4J_TEST_URI") or os.getenv("NEO4J_URI")
NEO4J_TEST_USERNAME = os.getenv("NEO4J_TEST_USERNAME", os.getenv("NEO4J_USERNAME", "neo4j"))
NEO4J_TEST_PASSWORD = os.getenv("NEO4J_TEST_PASSWORD", os.getenv("NEO4J_PASSWORD", "password"))
NEO4J_TEST_DATABASE = os.getenv("NEO4J_TEST_DATABASE", os.getenv("NEO4J_DATABASE", ""))
QWEN_API_KEY = os.getenv("OPENAI_API_KEY")

if not (NEO4J_TEST_URI and QWEN_API_KEY):
    pytest.skip(
        "Neo4j memory tests require NEO4J_URI (or NEO4J_TEST_URI) and OPENAI_API_KEY to be set",
        allow_module_level=True,
    )


def _build_agent_config(memory_subdir: str) -> AgentConfig:
    chat = ModelConfig(type=ModelType.CHAT, provider="openai", name="gpt-4o-mini")
    util = ModelConfig(type=ModelType.CHAT, provider="openai", name="gpt-4o-mini")
    embed = ModelConfig(
        type=ModelType.EMBEDDING,
        provider="openai",
        name="Qwen/Qwen3-Embedding-8B",
        api_base="https://chutes-qwen-qwen3-embedding-8b.chutes.ai/v1",
    )
    browser = ModelConfig(type=ModelType.CHAT, provider="openai", name="gpt-4o-mini")
    return AgentConfig(
        chat_model=chat,
        utility_model=util,
        embeddings_model=embed,
        browser_model=browser,
        mcp_servers="{}",
        memory_subdir=memory_subdir,
        knowledge_subdirs=["default"],
        memory_backend="neo4j",
        neo4j_uri=NEO4J_TEST_URI,
        neo4j_username=NEO4J_TEST_USERNAME,
        neo4j_password=NEO4J_TEST_PASSWORD,
        neo4j_database=NEO4J_TEST_DATABASE,
        neo4j_vector_dimensions=4096,
    )


@pytest.mark.asyncio
async def test_insert_and_retrieve_neo4j_memory():
    memory_subdir = "pytest-memory"
    config = _build_agent_config(memory_subdir)
    await Neo4jMemory.reset(memory_subdir)
    memory = await Neo4jMemory.get_for_config(
        config,
        memory_subdir,
        preload_knowledge=False,
        knowledge_subdirs=[],
    )

    doc_id = await memory.insert_text(
        "AI agents use embeddings for semantic search",
        {"area": "main", "tags": ["ai", "embeddings"]},
    )

    results = await memory.search_similarity_threshold(
        "semantic search",
        limit=5,
        threshold=0.2,
    )

    assert any(doc.metadata.get("id") == doc_id for doc in results)
    await Neo4jMemory.reset(memory_subdir)


@pytest.mark.asyncio
async def test_relationship_management():
    memory_subdir = "pytest-relationships"
    config = _build_agent_config(memory_subdir)
    await Neo4jMemory.reset(memory_subdir)
    memory = await Neo4jMemory.get_for_config(
        config,
        memory_subdir,
        preload_knowledge=False,
        knowledge_subdirs=[],
    )

    id1 = await memory.insert_text("Document 1", {"area": "main"})
    id2 = await memory.insert_text("Document 2", {"area": "main"})

    await memory.create_relationship(id1, id2, "RELATED", {"strength": 0.9})

    related = await memory.get_related_entities(id1)

    assert related
    assert any(item["id"] == id2 for item in related)
    await Neo4jMemory.reset(memory_subdir)

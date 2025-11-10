import os

import pytest

initialize = pytest.importorskip("initialize")
neo4j_module = pytest.importorskip("python.helpers.neo4j_memory")
Neo4jMemory = neo4j_module.Neo4jMemory


pytestmark = pytest.mark.skipif(
    not os.getenv("NEO4J_URI"), reason="Neo4j instance not available"
)


@pytest.mark.asyncio
async def test_neo4j_memory_initialization():
    config = initialize.initialize_agent()
    memory = await Neo4jMemory.get(config, "default")
    assert memory.memory_subdir == "default"

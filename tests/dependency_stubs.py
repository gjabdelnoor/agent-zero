"""Minimal stubs for optional third-party dependencies used in tests."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np


def install_dependency_stubs() -> None:
    """Inject lightweight stand-ins for optional runtime dependencies.

    These stubs provide just enough surface area for imports to succeed
    without pulling large native libraries into the test environment.
    """

    if "cryptography" not in sys.modules:
        crypto_module = types.ModuleType("cryptography")
        hazmat_module = types.ModuleType("cryptography.hazmat")
        primitives_module = types.ModuleType("cryptography.hazmat.primitives")
        asymmetric_module = types.ModuleType("cryptography.hazmat.primitives.asymmetric")
        rsa_module = types.ModuleType("cryptography.hazmat.primitives.asymmetric.rsa")
        padding_module = types.ModuleType("cryptography.hazmat.primitives.asymmetric.padding")
        hashes_module = types.ModuleType("cryptography.hazmat.primitives.hashes")
        serialization_module = types.ModuleType("cryptography.hazmat.primitives.serialization")

        class _DummySHA256:
            def __call__(self) -> "_DummySHA256":
                return self

        class _DummyMGF1:
            def __init__(self, algorithm: Any):
                self.algorithm = algorithm

        class _DummyOAEP:
            def __init__(self, mgf: Any, algorithm: Any, label: Any):
                self.mgf = mgf
                self.algorithm = algorithm
                self.label = label

        class _DummyPublicKey:
            def public_bytes(self, encoding: Any = None, format: Any = None) -> bytes:
                return b""

            def encrypt(self, data: bytes, pad: Any) -> bytes:
                return b""

        class _DummyPrivateKey(_DummyPublicKey):
            def public_key(self) -> "_DummyPublicKey":
                return _DummyPublicKey()

            def decrypt(self, data: bytes, pad: Any) -> bytes:
                return b""

        class RSAPrivateKey(_DummyPrivateKey):
            pass

        class RSAPublicKey(_DummyPublicKey):
            pass

        def generate_private_key(public_exponent: int, key_size: int) -> RSAPrivateKey:
            return RSAPrivateKey()

        rsa_module.RSAPrivateKey = RSAPrivateKey
        rsa_module.RSAPublicKey = RSAPublicKey
        rsa_module.generate_private_key = generate_private_key

        padding_module.MGF1 = _DummyMGF1
        padding_module.OAEP = _DummyOAEP

        hashes_module.SHA256 = _DummySHA256

        class _Encoding:
            PEM = "PEM"

        class _PublicFormat:
            SubjectPublicKeyInfo = "SubjectPublicKeyInfo"

        def load_pem_public_key(data: bytes) -> RSAPublicKey:
            return RSAPublicKey()

        serialization_module.Encoding = _Encoding
        serialization_module.PublicFormat = _PublicFormat
        serialization_module.load_pem_public_key = load_pem_public_key

        asymmetric_module.rsa = rsa_module
        asymmetric_module.padding = padding_module

        primitives_module.asymmetric = asymmetric_module
        primitives_module.hashes = hashes_module
        primitives_module.serialization = serialization_module

        hazmat_module.primitives = primitives_module
        crypto_module.hazmat = hazmat_module

        sys.modules["cryptography"] = crypto_module
        sys.modules["cryptography.hazmat"] = hazmat_module
        sys.modules["cryptography.hazmat.primitives"] = primitives_module
        sys.modules["cryptography.hazmat.primitives.asymmetric"] = asymmetric_module
        sys.modules["cryptography.hazmat.primitives.asymmetric.rsa"] = rsa_module
        sys.modules["cryptography.hazmat.primitives.asymmetric.padding"] = padding_module
        sys.modules["cryptography.hazmat.primitives.hashes"] = hashes_module
        sys.modules["cryptography.hazmat.primitives.serialization"] = serialization_module

    if "browser_use" not in sys.modules:
        browser_use = types.ModuleType("browser_use")
        llm_module = types.ModuleType("browser_use.llm")

        class _BaseChat:
            _fix_gemini_schema = staticmethod(lambda schema: schema)

        class ChatGoogle(_BaseChat):
            pass

        class ChatOllama(_BaseChat):
            pass

        class ChatOpenRouter(_BaseChat):
            pass

        class ChatAnthropic(_BaseChat):
            pass

        class ChatGroq(_BaseChat):
            pass

        class ChatOpenAI(_BaseChat):
            pass

        llm_module.ChatGoogle = ChatGoogle
        llm_module.ChatOllama = ChatOllama
        llm_module.ChatOpenRouter = ChatOpenRouter
        llm_module.ChatAnthropic = ChatAnthropic
        llm_module.ChatGroq = ChatGroq
        llm_module.ChatOpenAI = ChatOpenAI
        browser_use.llm = llm_module

        sys.modules["browser_use"] = browser_use
        sys.modules["browser_use.llm"] = llm_module

    if "faiss" not in sys.modules:
        faiss_module = types.ModuleType("faiss")

        class _IndexBase:
            def __init__(self, dim: int):
                self.dim = dim
                self._vectors: np.ndarray | None = None
                self._ids: np.ndarray | None = None

            def add_with_ids(self, matrix: np.ndarray, ids: np.ndarray) -> None:
                self._vectors = np.asarray(matrix, dtype="float32")
                self._ids = np.asarray(ids, dtype="int64")

            def search(self, queries: np.ndarray, k: int):
                if self._vectors is None or self._vectors.size == 0 or self._ids is None:
                    return (
                        np.zeros((len(queries), k), dtype="float32"),
                        -np.ones((len(queries), k), dtype="int64"),
                    )
                scores = np.matmul(np.asarray(queries, dtype="float32"), self._vectors.T)
                order = np.argsort(-scores, axis=1)[:, :k]
                top_scores = np.take_along_axis(scores, order, axis=1)
                repeated_ids = np.broadcast_to(self._ids, (len(queries), self._ids.size))
                top_ids = np.take_along_axis(repeated_ids, order, axis=1)
                return top_scores.astype("float32"), top_ids.astype("int64")

        class IndexFlatIP(_IndexBase):
            pass

        class IndexIVFPQ(_IndexBase):
            def __init__(self, quantizer: Any, dim: int, nlist: int, m: int, bits: int):
                super().__init__(dim)
                self.quantizer = quantizer

            def train(self, matrix: np.ndarray) -> None:
                self._vectors = np.asarray(matrix, dtype="float32")
                self._ids = np.arange(len(matrix), dtype="int64")

        class IndexIDMap:
            def __init__(self, index: _IndexBase):
                self.index = index

            def add_with_ids(self, matrix: np.ndarray, ids: np.ndarray) -> None:
                self.index.add_with_ids(matrix, ids)

            def search(self, queries: np.ndarray, k: int):
                return self.index.search(queries, k)

        faiss_module.IndexFlatIP = IndexFlatIP
        faiss_module.IndexIVFPQ = IndexIVFPQ
        faiss_module.IndexIDMap = IndexIDMap

        sys.modules["faiss"] = faiss_module

    if "whisper" not in sys.modules:
        whisper_module = types.ModuleType("whisper")

        class _DummyWhisperModel:
            def transcribe(self, path: str, fp16: bool = False):
                return {"text": ""}

        def load_model(name: str, download_root: str | None = None):
            return _DummyWhisperModel()

        whisper_module.load_model = load_model
        sys.modules["whisper"] = whisper_module

    if "langchain" not in sys.modules:
        langchain_module = types.ModuleType("langchain")
        embeddings_module = types.ModuleType("langchain.embeddings")
        base_module = types.ModuleType("langchain.embeddings.base")
        prompts_module = types.ModuleType("langchain.prompts")
        schema_module = types.ModuleType("langchain.schema")

        class Embeddings:
            pass

        class ChatPromptTemplate:
            def __init__(self, messages):
                self.messages = messages

            @classmethod
            def from_messages(cls, messages):
                return cls(messages)

            def __or__(self, other):
                class _DummyChain:
                    async def astream(self_inner, _input):
                        if hasattr(other, "astream"):
                            async for chunk in other.astream(_input):
                                yield chunk
                        else:
                            if isinstance(other, str):
                                yield other
                            else:
                                yield ""

                return _DummyChain()

        class FewShotChatMessagePromptTemplate:
            def __init__(self, example_prompt=None, examples=None, input_variables=None):
                self.example_prompt = example_prompt
                self.examples = examples or []
                self.input_variables = input_variables or []

            def format(self, *args, **kwargs):
                return ""

        class AIMessage:
            def __init__(self, content: str):
                self.content = content

        prompts_module.ChatPromptTemplate = ChatPromptTemplate
        prompts_module.FewShotChatMessagePromptTemplate = FewShotChatMessagePromptTemplate
        schema_module.AIMessage = AIMessage
        base_module.Embeddings = Embeddings
        embeddings_module.base = base_module
        langchain_module.prompts = prompts_module
        langchain_module.schema = schema_module
        langchain_module.embeddings = embeddings_module

        sys.modules["langchain"] = langchain_module
        sys.modules["langchain.embeddings"] = embeddings_module
        sys.modules["langchain.embeddings.base"] = base_module
        sys.modules["langchain.prompts"] = prompts_module
        sys.modules["langchain.schema"] = schema_module

    if "webcolors" not in sys.modules:
        webcolors_module = types.ModuleType("webcolors")

        class _RGB:
            def __init__(self, red: int, green: int, blue: int):
                self.red = red
                self.green = green
                self.blue = blue

        def name_to_rgb(name: str) -> _RGB:
            return _RGB(255, 255, 255)

        webcolors_module.name_to_rgb = name_to_rgb
        sys.modules["webcolors"] = webcolors_module

    if "sentence_transformers" not in sys.modules:
        st_module = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            def __init__(self, model: str, **kwargs):
                self.model = model

            def encode(self, texts, convert_to_numpy: bool = False, batch_size: int = 32):
                if isinstance(texts, str):
                    texts = [texts]
                vectors = np.zeros((len(texts), 768), dtype="float32")
                if convert_to_numpy:
                    return vectors
                return vectors.tolist()

        st_module.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = st_module

    if "git" not in sys.modules:
        git_module = types.ModuleType("git")

        class Repo:
            @staticmethod
            def init(*args, **kwargs):
                return Repo()

            @staticmethod
            def clone_from(*args, **kwargs):
                return Repo()

        git_module.Repo = Repo
        sys.modules["git"] = git_module

    if "mcp" not in sys.modules:
        mcp_module = types.ModuleType("mcp")

        class ClientSession:
            pass

        class StdioServerParameters:
            pass

        mcp_module.ClientSession = ClientSession
        mcp_module.StdioServerParameters = StdioServerParameters

        client_module = types.ModuleType("mcp.client")
        stdio_module = types.ModuleType("mcp.client.stdio")
        sse_module = types.ModuleType("mcp.client.sse")
        streamable_module = types.ModuleType("mcp.client.streamable_http")
        shared_module = types.ModuleType("mcp.shared")
        message_module = types.ModuleType("mcp.shared.message")
        types_module = types.ModuleType("mcp.types")

        def stdio_client(*args, **kwargs):
            return None

        def sse_client(*args, **kwargs):
            return None

        def streamablehttp_client(*args, **kwargs):
            return None

        class SessionMessage:
            pass

        class CallToolResult:
            pass

        class ListToolsResult:
            pass

        stdio_module.stdio_client = stdio_client
        sse_module.sse_client = sse_client
        streamable_module.streamablehttp_client = streamablehttp_client
        message_module.SessionMessage = SessionMessage
        types_module.CallToolResult = CallToolResult
        types_module.ListToolsResult = ListToolsResult
        client_module.stdio = stdio_module
        client_module.sse = sse_module
        client_module.streamable_http = streamable_module
        mcp_module.client = client_module
        shared_module.message = message_module
        mcp_module.types = types_module

        sys.modules["mcp"] = mcp_module
        sys.modules["mcp.client"] = client_module
        sys.modules["mcp.client.stdio"] = stdio_module
        sys.modules["mcp.client.sse"] = sse_module
        sys.modules["mcp.client.streamable_http"] = streamable_module
        sys.modules["mcp.shared"] = shared_module
        sys.modules["mcp.shared.message"] = message_module
        sys.modules["mcp.types"] = types_module

    if "nest_asyncio" not in sys.modules:
        nest_module = types.ModuleType("nest_asyncio")

        def apply():
            return None

        nest_module.apply = apply
        sys.modules["nest_asyncio"] = nest_module

    if "pytz" not in sys.modules:
        pytz_module = types.ModuleType("pytz")

        class _Timezone:
            def localize(self, dt, is_dst=False):
                return dt

        def timezone(name: str) -> _Timezone:
            return _Timezone()

        pytz_module.timezone = timezone
        sys.modules["pytz"] = pytz_module

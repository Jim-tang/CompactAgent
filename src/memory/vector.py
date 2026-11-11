"""向量检索：加载 embedding / reranker 模型，读写 Chroma 中的 L1 原子与 L2 场景集合。

任何失败都不向外抛出，只把 ``available`` 置为 False 并记录 ``error``，检索侧退化为纯 FTS5。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VectorIndex:
    """持有 embedding / reranker 模型与 Chroma 集合，提供向量写入、检索与精排。"""

    def __init__(self, config: dict) -> None:
        self.config = config or {}
        self.available = False
        self.error: Optional[str] = None

        self.persist_directory = self._resolve_dir(
            self.config.get("persist_directory", "./memory_db/vectors")
        )
        self.collection_name = self.config.get("collection_name", "memory_l1")
        self.scene_collection_name = self.config.get(
            "scene_collection_name", "memory_l2"
        )
        self.embedding_model = self.config.get("embedding_model", "BAAI/bge-m3")
        self.reranker_model = self.config.get("reranker_model", "BAAI/bge-reranker-base")
        self.local_files_only = bool(self.config.get("local_files_only", True))
        self.use_fp16 = bool(self.config.get("reranker_fp16", True))

        self._embedder: Any = None
        self._reranker: Any = None
        self._collection: Any = None
        self._scene_collection: Any = None
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_dir(raw: str) -> Path:
        from ..common_utils import PROJECT_ROOT

        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _prepare_imports(self) -> None:
        """构造模型前做准备：离线时设置 HF 离线环境变量，并把 transformers 日志降到 error"""
        import os

        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            import transformers  # noqa: F401
        except Exception:
            return
        transformers.logging.set_verbosity_error()

    def load_embedder(self) -> bool:
        """加载 embedding 模型，已加载则直接返回 True，加载失败记录 error 并返回 False。"""
        if self._embedder is not None:
            return True
        with self._lock:
            if self._embedder is not None:
                return True
            try:
                self._prepare_imports()
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embedder = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("embedding 模型已加载: %s", self.embedding_model)
                return True
            except Exception as exc:
                self.error = f"embedding 加载失败: {exc}"
                logger.warning(self.error)
                self._embedder = None
                return False

    def load_reranker(self) -> bool:
        """加载 reranker 模型，已加载则直接返回 True，加载失败记录 error 并返回 False。"""
        if self._reranker is not None:
            return True
        with self._lock:
            if self._reranker is not None:
                return True
            try:
                self._prepare_imports()
                from FlagEmbedding import FlagReranker

                self._reranker = FlagReranker(self.reranker_model, use_fp16=self.use_fp16)
                logger.info("reranker 模型已加载: %s", self.reranker_model)
                return True
            except Exception as exc:
                self.error = f"reranker 加载失败: {exc}"
                logger.warning(self.error)
                self._reranker = None
                return False

    def open_collection(self) -> bool:
        """打开或创建 L1 原子与 L2 场景两个 Chroma 集合，成功返回 True，embedding 未加载或建库失败返回 False。"""
        if self._collection is not None and self._scene_collection is not None:
            return True
        if self._embedder is None:
            return False
        with self._lock:
            if self._collection is not None and self._scene_collection is not None:
                return True
            try:
                import chromadb
                from chromadb.config import Settings

                self.persist_directory.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._scene_collection = client.get_or_create_collection(
                    name=self.scene_collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self.available = True
                return True
            except Exception as exc:
                self.error = f"Chroma 打开失败: {exc}"
                logger.warning(self.error)
                self._collection = None
                self._scene_collection = None
                self.available = False
                return False

    def load(self) -> bool:
        """依次加载 embedding 与 Chroma 集合，任一失败则 available 为 False，返回是否可用。"""
        ok = self.load_embedder() and self.open_collection()
        self.available = bool(ok)
        return self.available

    # ------------------------------------------------------------ 读写
    def _upsert(self, collection: Any, ids: list[str], texts: list[str],
                metadatas: list[dict], label: str) -> bool:
        """把 ids/texts/metadatas 嵌入后 upsert 进指定集合，成功返回 True，失败只记日志返回 False。"""
        if not collection or not ids:
            return False
        try:
            embeddings = self._embedder.embed_documents(texts)
            collection.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )
            return True
        except Exception as exc:
            logger.warning("%s 向量 upsert 失败: %s", label, exc)
            return False

    def _search(self, collection: Any, query: str, limit: int,
                id_key: str, label: str) -> list[dict]:
        """在指定集合里做向量检索，返回按 ``id_key`` 标注 id 的条目列表，失败或未就绪返回空列表。"""
        if not collection or not self._embedder:
            return []
        try:
            vector = self._embedder.embed_query(query)
            result = collection.query(
                query_embeddings=[vector], n_results=max(1, limit)
            )
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            out = []
            for i, doc_id in enumerate(ids):
                out.append(
                    {
                        id_key: doc_id,
                        "content": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": dists[i] if i < len(dists) else None,
                    }
                )
            return out
        except Exception as exc:
            logger.warning("%s 向量检索失败（降级为纯 FTS5）: %s", label, exc)
            return []

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> bool:
        """按 id 幂等写入 L1 原子向量，成功返回 True，失败返回 False。"""
        return self._upsert(self._collection, ids, texts, metadatas, "L1")

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """在 L1 集合里按 query 检索，返回最多 limit 条 ``atom_id`` 命中的条目；不可用时返回空列表。"""
        return self._search(self._collection, query, limit, "atom_id", "L1")

    def upsert_scenes(self, ids: list[str], texts: list[str],
                      metadatas: list[dict]) -> bool:
        """把 L2 场景写入场景集合，texts 必须是 ``name + summary``（不含正文），成功返回 True。"""
        return self._upsert(self._scene_collection, ids, texts, metadatas, "L2")

    def delete_scenes(self, ids: list[str]) -> bool:
        """按 id 从场景集合删除向量，成功返回 True，失败只记日志返回 False。"""
        if not self._scene_collection or not ids:
            return False
        try:
            self._scene_collection.delete(ids=ids)
            return True
        except Exception as exc:
            logger.warning("L2 向量删除失败: %s", exc)
            return False

    def search_scenes(self, query: str, limit: int = 20) -> list[dict]:
        """在场景集合里按 query 检索，返回项的 id 键为 ``scene_id``；不可用时返回空列表。"""
        return self._search(self._scene_collection, query, limit, "scene_id", "L2")

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """用 reranker 给候选打分并取前 top_n 条，返回副本且带 ``rerank_score``（0~1）。"""
        if not candidates:
            return []
        if self._reranker is None:
            return candidates[:top_n]
        try:
            pairs = [(query, c.get("content", "")) for c in candidates]
            scores = self._reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):
                scores = [scores]
            ranked = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )
            result = []
            for item, score in ranked[:top_n]:
                enriched = dict(item)
                enriched["rerank_score"] = float(score)
                result.append(enriched)
            return result
        except Exception as exc:
            logger.warning("reranker 精排失败（保持 RRF 顺序）: %s", exc)
            return candidates[:top_n]

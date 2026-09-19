import os
os.environ['ANONYMIZED_TELEMETRY'] = 'false'  # 禁用 Chroma 遥测数据收集

import re
import json
import rjieba
import numpy as np
import asyncio
import threading
import transformers
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from langchain_chroma import Chroma
from langchain_classic.schema import Document
from FlagEmbedding import FlagReranker
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from llm_client import OpenAIClient
from common_utils import get_messages_text

transformers.logging.set_verbosity_error()

CONSOLIDATE_PROMPT = """
以下是一个包含若干条近期会话记录的列表，请逐条分析每条记录的重要性（0.0 ~ 1.0 分），然后将其概括为语义简洁、自包含的长期记忆，务必保留关键信息。

---
{memories_text}
---

请严格按照以下 JSON 格式输出，返回一个 JSON 数组，每个元素对应一条记忆，顺序与输入一致：
[
  {{
    "importance_score": <重要性分数>,
    "summary": "<记忆摘要>"
  }},
  ...
]
"""

RECORD_SESSION_PROMPT = """
请仔细阅读下面用户与 Agent 之间的完整对话历史，从中提炼会话主题并概括会话内容，最后以 JSON 字符串形式输出

---
### 对话历史
{conversation_history}
---

输出格式要求：
{{
  "topic": <会话主题>（不超过 40 字）"
  "content": <会话内容>（不超过 500 字）
}}

概括会话内容时注意语义连贯，言简意赅，保持客观准确，避免冗余细节，需要包含以下关键信息：
- 用户的核心请求或任务目标
- Assistant 调用的所有工具名称、关键参数
- 工具调用的返回结果中的重要数据
- Assistant 据此做出的决策、推理或最终回答
- 任何遇到的问题、异常或需要用户澄清的内容
"""

class LocalEmbeddings:
    def __init__(self, model_name='BAAI/bge-m3'):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """兼容原生 ChromaDB 的接口"""
        return self.embed_documents(input)

    def name(self):
        return self.model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str = '', input: List[str] = None) -> List:
        """
        对查询进行embedding，Chroma查询时传入单个文本，而原生 chromadb 接口传入的是input列表
        """
        if text:
            return self.embed_documents([text])[0]
        elif input:
            return self.embed_documents(input)
        else:
            return []

class BM25Index:
    """
    基于 rank_bm25.BM25Okapi 的关键词检索器，所有文档以 List[Document] 形式存储，支持全量重建、得分计算、top-k 检索。
    """
    def __init__(self):
        self.docs: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None

    def rebuild(self, docs: List[Document]):
        """用新的文档列表重建索引"""
        self.docs = docs
        tokenized_corpus = [list(rjieba.cut(doc.page_content)) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_scores(self, query: str) -> np.ndarray:
        """返回查询与所有文档的 BM25 得分"""
        if self.bm25 is None or not self.docs:
            return np.array([])
        tokenized_query = list(rjieba.cut(query))
        return np.array(self.bm25.get_scores(tokenized_query))

class MemoryManager:
    """
    记忆管理器：负责短期记忆与长期记忆的添加、检索、遗忘、整合。
    - 短期记忆：带 TTL 自动遗忘。
    - 长期记忆：由重要短期记忆整合生成。
    - 检索流程：采用语义检索 + BM25关键词的混合检索策略，先查短期，若最佳融合得分低于阈值则查长期。
    - 整合：使用 LLM 评估短期记忆重要性，重要者改写后存入长期。
    """

    def __init__(
        self,
        llm_client: OpenAIClient,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-m3",
        session_ttl_seconds: int = 3600,         # 短期记忆存活时间（秒）
        retrieve_threshold: float = 0.25,           # 混合检索的重排筛选阈值
        importance_threshold: float = 0.6,          # 重要性阈值（0~1），高于此值才转为长期记忆
        hybrid_weights: Tuple[float, float] = (0.6, 0.4),  # (语义权重, BM25权重)
    ):
        self.llm_client = llm_client

        # 嵌入模型（本地部署）
        embeddings = LocalEmbeddings(embedding_model)
        # 短期记忆向量库
        session_collection = Chroma(
            collection_name="session_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        # 长期记忆向量库
        long_term_collection = Chroma(
            collection_name="long_term_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )


        self.session_dir = "session"
        self.memory_collections = {"session": session_collection, "long_term": long_term_collection}
        self.session_ttl = session_ttl_seconds
        self.retrieve_threshold = retrieve_threshold
        self.importance_threshold = importance_threshold
        self.hybrid_weights = hybrid_weights

        self.bm25_retrievers = {"session": BM25Index(), "long_term": BM25Index()}
        self._rebuild_bm25_retriever("session")
        self._rebuild_bm25_retriever("long_term")

        self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        self.reranker_lock = threading.Lock()

    def _rebuild_bm25_retriever(self, memory_type: str = "session"):
        """从指定的向量库重建 BM25 检索器（每次增删都重建的开销较大，可优化为增量更新）"""
        # 获取所有文档
        all_docs = self.memory_collections[memory_type].get()
        if not all_docs['documents']:
            return None
        # 构建 Document 列表
        docs = [
            Document(id=doc_id, page_content=text, metadata=meta)
            for doc_id, text, meta in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas'])
        ]
        self.bm25_retrievers[memory_type].rebuild(docs)

    def _hybrid_search(
        self,
        vector_store: Chroma,
        bm25_retriever: BM25Index,
        query: str,
        top_k: int
    ) -> List[Document]:
        """
        对指定的存储执行混合检索（语义 + BM25），返回 (文本, 综合得分) 列表
        """
        # 语义检索
        semantic_results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

        # 若无 BM25 检索器，仅返回语义结果
        if bm25_retriever.bm25 is None:
            return [doc for doc, score in semantic_results]

        # BM25 关键词检索
        bm25_scores = bm25_retriever.get_scores(query)
        max_bm25 = float(np.max(bm25_scores))
        norm_bm25 = (bm25_scores / max_bm25).tolist() if max_bm25 > 0 else [0.0] * len(bm25_scores)

        # 建立 BM25 得分映射: 文档 id -> bm25 得分
        bm25_map = {}
        for doc, bm25_score in zip(bm25_retriever.docs, norm_bm25):
            bm25_map[doc.id] = bm25_score

        # 建立综合得分映射: 文档 id -> (文档内容, 综合得分)
        fusion_map = {}
        for doc, sem_score in semantic_results:
            bm25_score = bm25_map.get(doc.id, 0.0)
            # 加权融合
            fused_score = self.hybrid_weights[0] * sem_score + self.hybrid_weights[1] * bm25_score
            fusion_map[doc.id] = (doc, fused_score)

        # 重新排序
        sorted_items = sorted(fusion_map.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_items]

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, memory_type: str = "session") -> str:
        """
        添加记忆
        :param text: 记忆内容
        :param metadata: 附加元数据（会自动添加 timestamp）
        :param memory_type: 记忆类型
        :return: 文档 ID
        """
        if metadata is None:
            metadata = {}
        # 自动添加时间戳
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['memory_type'] = memory_type

        doc_id = self.memory_collections[memory_type].add_texts(
            texts=[text],
            metadatas=[metadata]
        )[0]

        # 更新 BM25 索引
        self._rebuild_bm25_retriever(memory_type)

        return doc_id

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """单个查询的检索"""
        short_collection = self.memory_collections['session']
        long_collection = self.memory_collections['long_term']
        short_bm25 = self.bm25_retrievers['session']
        long_bm25 = self.bm25_retrievers['long_term']

        # 并行检索短期和长期记忆
        with ThreadPoolExecutor(max_workers=2) as executor:
            short_future = executor.submit(self._hybrid_search,short_collection, short_bm25, query, top_k)
            long_future = executor.submit(self._hybrid_search,long_collection, long_bm25, query, top_k)
            short_results = short_future.result()
            long_results = long_future.result()

        all_docs = short_results + long_results
        candidates = [doc.page_content for doc in all_docs]

        # 利用 rerank 模型筛选混合检索的结果
        with self.reranker_lock:
            # 确保同一时间只有一个线程执行重排，避免多线程并发调用导致模型内部冲突
            scores = self.reranker.compute_score([(query, text) for text in candidates], normalize=True)
        sorted_pairs = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
        result = [item for item in sorted_pairs if item[1] >= self.retrieve_threshold][:top_k]
        return result

    async def forget(self) -> int:
        """基于 TTL 的遗忘机制：找出所有过期的短期记忆，对这些记忆进行整合（重要的转为长期记忆），删除过期记忆"""
        all_short = self.memory_collections['session'].get(include=["documents", "metadatas"])
        if not all_short['ids']:
            return 0

        # 找出所有过期的短期记忆
        now = datetime.now()
        expired = []  # 存储 (id, text, metadata)
        for idx, doc_id in enumerate(all_short['ids']):
            meta = all_short['metadatas'][idx]
            ts = datetime.fromisoformat(meta['timestamp'])
            if now - ts > timedelta(seconds=self.session_ttl):
                expired.append((doc_id, meta.get('content')))

        if not expired:
            return 0

        # 整合即将被遗忘的记忆
        consolidated_ids = await self._consolidate_memories(expired)
        print("已整合 %d 条短期记忆" % len(consolidated_ids))

        # 删除过期记忆
        expired_ids = [item[0] for item in expired]
        self.memory_collections['session'].delete(ids=expired_ids)

        # 重建短期 BM25 索引
        self._rebuild_bm25_retriever(memory_type='session')
        
        return len(expired)

    async def _consolidate_memories(self, memories: List[Tuple], batch_size: int = 10, max_concurrent: int = 5) -> List[str]:
        """异步整合记忆，每个元素包含（id,记忆文本,元数据），将重要的转为长期记忆"""
        semaphore = asyncio.Semaphore(max_concurrent)  # 控制并发数
        async def process_batch(batch, batch_idx):
            async with semaphore:
                memories_text = "\n---\n".join(f"### 记忆 {idx+1} ###\n{item[1]}" for idx, item in enumerate(batch))
                messages = [
                    {"role": "system", "content": "你是一个记忆管理与评估专家，负责一个 Agent 系统的记忆模块，该模块具备将短期的会话记忆评估后整合为长期记忆的功能。"},
                    {"role": "user", "content": CONSOLIDATE_PROMPT.format(memories_text=memories_text)},
                ]
                try:
                    response = await self.llm_client.invoke(messages)
                    content = response.get("content", "").strip()
                    if content.startswith("```json"):
                        content = re.sub(r'^```(?:json)?\s*', '', content).replace("```", "")
                    items = json.loads(content)
                    return items
                except Exception as e:
                    print(f"❌ 处理批次 {batch_idx} 时发生错误: {e}")
                    return []

        # 分批处理
        batches = [memories[i:i + batch_size] for i in range(0, len(memories), batch_size)]
        # 并发执行所有批次的 LLM 请求
        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        # 结果列表的顺序与传入的 tasks 序列顺序一致
        results = await asyncio.gather(*tasks)

        # 串行处理所有 add 长期记忆的操作
        long_mem_ids = []
        for i, items in enumerate(results):
            for idx, item in enumerate(items):
                score = item.get("importance_score", 0.0)
                summary = item.get("summary")
                if score >= self.importance_threshold and summary:
                    new_meta = {"importance_score": score}
                    long_mem_id = self.add_memory(summary, metadata=new_meta, memory_type="long_term")
                    long_mem_ids.append(long_mem_id)
        return long_mem_ids

    async def record_session(self, session_msg: list[dict]) -> str:
        """提炼并存储本次会话的对话历史并记录至 Sessions.md"""
        if not session_msg:
            return "Empty session history, skip recording"
        history_text = get_messages_text(session_msg)
        session_note = await self.extract_session(history_text)
        topic = session_note.get("topic", "")
        summary = session_note.get("content", "")
        timestamp = datetime.now().isoformat()
        await asyncio.to_thread(self.add_memory, topic, metadata={"content": summary})
        await asyncio.to_thread(self.update_session_md, topic, summary, timestamp)
        await asyncio.to_thread(self.save_session_json, topic, summary, timestamp, session_msg)
        return topic

    async def extract_session(self, history_text: str):
        messages = [
            {"role": "system", "content": "你是一个会话分析与记忆提取专家。"},
            {"role": "user", "content": RECORD_SESSION_PROMPT.format(conversation_history=history_text)},
        ]
        try:
            response = await self.llm_client.invoke(messages)
            content = response.get("content", "").strip()
            if content.startswith("```json"):
                content = re.sub(r'^```(?:json)?\s*', '', content).replace("```", "")
            return json.loads(content)
        except Exception as e:
            print(f"记录本次会话历史时发生错误: {e}\n============== content ==============\n{content}")
            return {}

    def save_session_json(self, topic: str, summary: str, timestamp: str, session_msg: list[dict]) -> None:
        """将完整对话历史保存为 JSON 文件"""
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            safe_topic = re.sub(r'[<>:"/\\|?*]', '_', topic)
            safe_timestamp = datetime.fromisoformat(timestamp).strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.session_dir, f"{safe_timestamp}_{safe_topic}.json")
            session_data = {
                "topic": topic,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "messages": session_msg,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            print(f"已保存会话 JSON: {filepath}")
        except Exception as e:
            print(f"⚠️ 保存会话 JSON 失败: {e}")

    def update_session_md(self, topic: str, content: str, timestamp: str) -> None:
        """将新的会话条目增量追加到 Sessions.md（插入到文件开头）"""
        entry_lines = [
            f"### {topic}\n",
            f"> **{datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}**\n",
            f"> {content}\n",
            "---\n",
        ]
        entry_text = "".join(entry_lines)
        session_file_path = os.path.join(self.session_dir, "Sessions.md")
        if os.path.exists(session_file_path):
            with open(session_file_path, "r", encoding="utf-8") as f:
                existing = f.read()
            new_content = entry_text + existing
        else:
            new_content = entry_text

        with open(session_file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    def read_sessions(self) -> List[Dict]:
        """从 sessions.md 解析近期的会话列表"""
        session_file_path = os.path.join(self.session_dir, "Sessions.md")
        if not os.path.exists(session_file_path):
            return []
        with open(session_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        sessions = []
        # 正则匹配格式： ### topic\n\n> **timestamp**\n> content\n\n---
        pattern = r"### (.*?)\n\n> \*\*([^*]+)\*\*\n> (.*?)\n\n---"
        matches = re.findall(pattern, content, re.DOTALL)
        for topic, timestamp, content_text in matches:
            sessions.append({
                "timestamp": timestamp.strip(),
                "topic": topic.strip(),
                "content": content_text.strip()
            })
        return sessions
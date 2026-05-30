"""
L2: Semantic Cache
基于 embedding 向量的语义相似缓存。
语义相近的 prompt（如 "解释 AMM" vs "AMM 是什么"）
命中同一条缓存响应，省掉重复调用。
"""

import threading
import time
from typing import Any, Optional

import numpy as np

from .config import CacheConfig, DEFAULT_CONFIG


class SemanticCache:
    """语义相似匹配缓存 —— L2 层

    核心思路：
    1. prompt → embedding 向量
    2. 新请求与库中所有向量计算 cosine similarity
    3. 最高分超过阈值 → 命中，返回缓存响应
    4. 否则 → miss，新 embedding 入库

    特点：
    - 对同义不同文的查询同样生效
    - 相似度阈值可调（默认 0.95，语义等价级别）
    - 自动过期 + 容量淘汰
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # embedding 模型（延迟加载）
        self._encoder = None

        # 向量存储: list[embedding]
        self._embeddings: list[np.ndarray] = []
        # 响应存储: list[(expire_at, response)]
        self._responses: list[tuple[float, Any]] = []
        # prompt 原文（用于调试/排重）
        self._prompts: list[str] = []
        # 索引顺序 = 写入顺序，与以上三表对齐

        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    # ── 公共接口 ──────────────────────────────────────────────

    def get(self, prompt: str, threshold: Optional[float] = None) -> Optional[Any]:
        """语义匹配查询。找到相似度最高的缓存并返回。"""
        emb = self._encode(prompt)
        if emb is None:
            return None

        threshold = threshold if threshold is not None else self.config.l2_similarity_threshold
        now = time.time()

        with self._lock:
            if not self._embeddings:
                self._stats["misses"] += 1
                return None

            # 清理过期 + 找最相似
            best_idx = -1
            best_sim = 0.0
            valid_indices = []

            for i in range(len(self._responses) - 1, -1, -1):
                expire_at, _ = self._responses[i]
                if expire_at < now:
                    # 过期 → 标记删除
                    continue

                valid_indices.append(i)
                sim = self._cosine_similarity(emb, self._embeddings[i])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx >= 0 and best_sim >= threshold:
                self._stats["hits"] += 1
                return self._responses[best_idx][1]

            self._stats["misses"] += 1
            return None

    def set(
        self,
        prompt: str,
        response: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """写入语义缓存。自动编码 prompt 并存储。"""
        emb = self._encode(prompt)
        if emb is None:
            return

        ttl = ttl if ttl is not None else self.config.l2_ttl_seconds
        expire_at = time.time() + ttl

        with self._lock:
            # 容量淘汰
            while len(self._responses) >= self.config.l2_max_size:
                self._evict_one()

            self._embeddings.append(emb)
            self._responses.append((expire_at, response))
            self._prompts.append(prompt)
            self._stats["sets"] += 1

    def get_similarity(self, prompt_a: str, prompt_b: str) -> float:
        """计算两个 prompt 的语义相似度（辅助调试用）。"""
        emb_a = self._encode(prompt_a)
        emb_b = self._encode(prompt_b)
        if emb_a is None or emb_b is None:
            return 0.0
        return float(self._cosine_similarity(emb_a, emb_b))

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock:
            self._embeddings.clear()
            self._responses.clear()
            self._prompts.clear()

    def stats(self) -> dict:
        """获取缓存统计。"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0.0
            return {
                **self._stats,
                "hit_rate": round(hit_rate, 2),
                "size": len(self._responses),
                "max_size": self.config.l2_max_size,
                "threshold": self.config.l2_similarity_threshold,
                "embedding_dim": self.config.l2_embedding_dim,
            }

    def __len__(self) -> int:
        return len(self._responses)

    # ── 内部方法 ──────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """将文本编码为 embedding 向量。"""
        try:
            encoder = self._get_encoder()
            emb = encoder.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            # 如果 sentence-transformers 未安装，报错但不崩溃
            import warnings
            warnings.warn(f"Embedding failed: {e}. "
                          f"Install with: pip install 'llm-response-cache[embedding]'")
            return None

    def _get_encoder(self):
        """延迟加载 embedding 模型。"""
        if self._encoder is not None:
            return self._encoder

        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.config.l2_embedding_model)
        except ImportError:
            # fallback: 简单的 TF-IDF + sklearn
            self._encoder = self._SklearnFallbackEncoder()
        return self._encoder

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个归一化向量的 cosine similarity。"""
        # a: (1, dim), b: (1, dim) → 拉平后内积
        a_flat = a.flatten()
        b_flat = b.flatten()
        return float(np.dot(a_flat, b_flat))

    def _evict_one(self) -> None:
        """淘汰最旧的缓存项（FIFO）。"""
        if self._responses:
            self._embeddings.pop(0)
            self._responses.pop(0)
            self._prompts.pop(0)
            self._stats["evictions"] += 1

    # ── 无依赖 fallback embedding ────────────────────────────

    class _SklearnFallbackEncoder:
        """无 sentence-transformers 时的 fallback：CountVectorizer + TfidfTransformer"""

        def __init__(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=384,
                analyzer="char_wb",
                ngram_range=(2, 4),
            )
            self._fitted = False

        def encode(self, texts, normalize_embeddings=True):
            """编码文本。支持单字符串或列表。"""
            if isinstance(texts, str):
                texts = [texts]
            if not self._fitted:
                emb = self._vectorizer.fit_transform(texts).toarray()
                self._fitted = True
            else:
                emb = self._vectorizer.transform(texts).toarray()
            if normalize_embeddings:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms
            return emb

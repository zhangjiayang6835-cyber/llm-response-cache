"""
L1: Exact Match Cache
基于 prompt 全文 hash 的精确匹配缓存。
相同 prompt → 零 token 开销直接返回。
"""

import hashlib
import json
import threading
import time
from typing import Any, Optional

from .config import CacheConfig, DEFAULT_CONFIG


class ExactMatchCache:
    """精确匹配缓存 —— L1 层

    核心思路：对 prompt 取 SHA256 hash 作为 key，
    相同请求命中缓存，不走 LLM API。

    特点：
    - O(1) 查找，极致速度
    - 自动过期 + LRU 淘汰
    - 可选 Redis 后端
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # 内存存储: {hash_key: (expire_at, response)}
        self._store: dict[str, tuple[float, Any]] = {}
        # LRU 顺序列表（最近使用的在前面）
        self._lru_order: list[str] = []
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    # ── 公共接口 ──────────────────────────────────────────────

    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        """获取缓存。命中返回缓存数据，未命中返回 None。"""
        key = self._hash(prompt)
        now = time.time()

        with self._lock:
            if key in self._store:
                expire_at, response = self._store[key]
                if expire_at > now:
                    # 命中 → 移到 LRU 最前
                    self._promote(key)
                    self._stats["hits"] += 1
                    return response
                else:
                    # 过期 → 删除
                    del self._store[key]
                    self._lru_remove(key)

            self._stats["misses"] += 1
            return None

    def set(self, prompt: str, response: Any, ttl: Optional[int] = None) -> None:
        """写入缓存。"""
        key = self._hash(prompt)
        ttl = ttl if ttl is not None else self.config.l1_ttl_seconds
        expire_at = time.time() + ttl

        with self._lock:
            # 如果 key 已存在，先移除旧记录
            if key in self._store:
                self._lru_remove(key)

            # 淘汰：超过最大容量
            while len(self._store) >= self.config.l1_max_size:
                self._evict_one()

            self._store[key] = (expire_at, response)
            self._lru_order.insert(0, key)
            self._stats["sets"] += 1

    def delete(self, prompt: str) -> bool:
        """删除指定缓存。"""
        key = self._hash(prompt)
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._lru_remove(key)
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock:
            self._store.clear()
            self._lru_order.clear()

    def stats(self) -> dict:
        """获取缓存统计。"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0.0
            return {
                **self._stats,
                "hit_rate": round(hit_rate, 2),
                "size": len(self._store),
                "max_size": self.config.l1_max_size,
            }

    def __len__(self) -> int:
        return len(self._store)

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _hash(prompt: str) -> str:
        """对 prompt 取 SHA256 哈希作为缓存 key。"""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _promote(self, key: str) -> None:
        """将 key 移到 LRU 最前面（最近使用）。"""
        self._lru_remove(key)
        self._lru_order.insert(0, key)

    def _lru_remove(self, key: str) -> None:
        """从 LRU 列表中移除 key。"""
        try:
            self._lru_order.remove(key)
        except ValueError:
            pass

    def _evict_one(self) -> None:
        """淘汰 LRU 末尾（最久未使用）的一项。"""
        if self._lru_order:
            evict_key = self._lru_order.pop()
            if evict_key in self._store:
                del self._store[evict_key]
                self._stats["evictions"] += 1

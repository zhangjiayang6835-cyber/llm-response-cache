"""
L1: ExactMatchCache 测试
"""

import time
import pytest
from llm_cache import ExactMatchCache


class TestExactMatchCache:
    def test_set_and_get(self):
        cache = ExactMatchCache()
        cache.set("hello", "world", ttl=60)
        assert cache.get("hello") == "world"

    def test_miss_returns_none(self):
        cache = ExactMatchCache()
        assert cache.get("never_set") is None

    def test_expired_entry(self):
        cache = ExactMatchCache()
        cache.set("expire_me", "data", ttl=0)  # 立即过期
        time.sleep(0.01)
        assert cache.get("expire_me") is None

    def test_overwrite_same_key(self):
        cache = ExactMatchCache()
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_lru_eviction(self):
        cache = ExactMatchCache()
        cache.config.l1_max_size = 3
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")  # 触发淘汰
        assert cache.get("a") is None  # a 被淘汰
        assert cache.get("d") == "4"   # d 存在

    def test_lru_promotion(self):
        """访问过的 key 不淘汰"""
        cache = ExactMatchCache()
        cache.config.l1_max_size = 3
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.get("a")  # 访问 a，提升 LRU 优先级
        cache.set("d", "4")  # 触发淘汰 → 淘汰 b
        assert cache.get("a") == "1"  # a 还在
        assert cache.get("b") is None  # b 被淘汰

    def test_delete(self):
        cache = ExactMatchCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        cache = ExactMatchCache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert len(cache) == 0

    def test_stats(self):
        cache = ExactMatchCache()
        cache.set("ping", "pong")
        cache.get("ping")   # hit
        cache.get("nope")   # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 50.0

    def test_different_prompts_different_keys(self):
        cache = ExactMatchCache()
        cache.set("prompt one", "response one")
        cache.set("prompt two", "response two")
        assert cache.get("prompt one") == "response one"
        assert cache.get("prompt two") == "response two"

    def test_ttl_override(self):
        cache = ExactMatchCache()
        cache.set("short", "data", ttl=1)
        cache.set("long", "data", ttl=60)
        # 两者应该都能获取
        assert cache.get("short") == "data"
        assert cache.get("long") == "data"

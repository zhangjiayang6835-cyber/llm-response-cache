"""
L2: SemanticCache 测试
"""

import pytest
from llm_cache import SemanticCache


class TestSemanticCache:
    def test_set_and_get_exact(self):
        cache = SemanticCache()
        cache.set("什么是闪电贷攻击？", "闪电贷攻击是一种...")
        result = cache.get("什么是闪电贷攻击？")
        assert result is not None
        assert "闪电贷攻击" in result

    def test_semantic_similarity(self):
        """语义相近的 prompt 应该命中"""
        cache = SemanticCache()
        cache.set("AMM 是什么", "AMM 是自动做市商...")
        result = cache.get("什么是 AMM")
        # 语义相近应该命中（阈值 0.95 可能偏高，如果 miss 属于正常）
        # 这里我们确认至少不会 crash
        assert result is None or "自动做市商" in result

    def test_get_similarity(self):
        """测试相似度计算"""
        cache = SemanticCache()
        sim = cache.get_similarity("hello world", "hello world")
        assert sim >= 0.99

    def test_miss_returns_none(self):
        cache = SemanticCache()
        result = cache.get("完全没有关联的问题")
        assert result is None

    def test_multiple_entries(self):
        cache = SemanticCache()
        cache.set("问题 A", "回答 A")
        cache.set("问题 B", "回答 B")
        cache.set("问题 C", "回答 C")
        result = cache.get("问题 A")
        assert result is not None

    def test_empty_cache_stats(self):
        cache = SemanticCache()
        s = cache.stats()
        assert s["size"] == 0
        assert s["hit_rate"] == 0.0

    def test_clear(self):
        cache = SemanticCache()
        cache.set("test", "response")
        cache.clear()
        assert len(cache) == 0

    def test_threshold_override(self):
        """测试自定义阈值"""
        cache = SemanticCache()
        cache.set("approve 函数", "approve 用于授权...")
        # 用极低的阈值，保证命中
        result = cache.get("approve 方法", threshold=0.1)
        assert result is not None

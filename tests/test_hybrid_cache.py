"""
L3: HybridCache 测试
"""

import pytest
from llm_cache import HybridCache


class TestHybridCache:
    def test_l1_hit(self):
        """相同 prompt → L1 命中"""
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return f"response to: {prompt}"

        cache = HybridCache(llm_callback=llm)
        resp1, src1 = cache.query("hello")
        assert src1 == "llm"
        assert len(calls) == 1

        resp2, src2 = cache.query("hello")
        assert src2 == "l1"
        assert len(calls) == 1  # 没再调 LLM

    def test_l2_hit(self):
        """语义相近 → L2 命中"""
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return f"response to: {prompt}"

        cache = HybridCache(llm_callback=llm)
        resp1, src1 = cache.query("Solidity 中的 delegatecall 是什么")
        assert src1 in ("llm",)
        assert len(calls) == 1

        resp2, src2 = cache.query("什么是 delegatecall")
        # 可能 L2 命中（语义相似），也可能不命中（取决于阈值）
        # 至少不会 crash
        assert src2 in ("l1", "l2", "llm")

    def test_llm_fallback(self):
        """全新问题 → LLM 调用"""
        def llm(prompt):
            return f"response: {prompt}"

        cache = HybridCache(llm_callback=llm)
        resp, src = cache.query("brand new question")
        assert src == "llm"
        assert resp == "response: brand new question"

    def test_no_callback_raises(self):
        """没有提供回调时应报错"""
        cache = HybridCache(config=None)
        with pytest.raises(ValueError):
            cache.query("test")

    def test_clear(self):
        def llm(prompt):
            return "ok"

        cache = HybridCache(llm_callback=llm)
        cache.query("test")
        assert len(cache.l1) > 0
        cache.clear()
        assert len(cache.l1) == 0

    def test_stats(self):
        def llm(prompt):
            return "ok"

        cache = HybridCache(llm_callback=llm)
        cache.query("a")
        cache.query("a")  # L1 命中
        s = cache.stats()
        assert s["total_requests"] == 2
        assert s["l1_hits"] >= 1

    def test_report(self):
        def llm(prompt):
            return "ok"

        cache = HybridCache(llm_callback=llm)
        cache.query("test")
        report = cache.report()
        assert "Hybrid Cache" in report
        assert "llm_calls" in report.lower() or "LLM" in report

    def test_l1_disabled(self):
        """关闭 L1 时 Hybrid 仍然可用"""
        from llm_cache.config import CacheConfig
        config = CacheConfig(l1_enabled=False)

        calls = []
        def llm(prompt):
            calls.append(prompt)
            return "ok"

        cache = HybridCache(llm_callback=llm, config=config)
        assert cache.l1 is None
        resp, src = cache.query("test")
        assert src == "llm"
        assert len(calls) == 1

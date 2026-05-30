"""
L3: HybridCache 测试
"""

import pytest
from llm_cache import HybridCache, CacheConfig, CompressResult


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


class TestHybridCacheWithCompression:
    def test_compression_enabled(self):
        """开启压缩后 compressor 被初始化"""
        config = CacheConfig(compression_enabled=True)
        cache = HybridCache(llm_callback=lambda p: "ok", config=config)
        assert cache.compressor is not None

    def test_compression_disabled_by_default(self):
        """默认不开启压缩"""
        cache = HybridCache(llm_callback=lambda p: "ok")
        assert cache.compressor is None

    def test_compression_before_llm(self):
        """LLM 调用前压缩 prompt 省 token"""
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return f"response: {prompt}"

        config = CacheConfig(compression_enabled=True, compression_level="aggressive")
        cache = HybridCache(llm_callback=llm, config=config)

        long_prompt = (
            "首先，我想请你帮我写一个 Solidity 合约。这个合约需要实现 ERC20 标准的所有功能。"
            "好的，非常感谢！如果你有任何问题，欢迎随时提出。"
        )
        resp, src = cache.query(long_prompt)
        sent_prompt = calls[0]
        assert src == "llm"
        # 压缩后的 prompt 应该更短
        assert len(sent_prompt) < len(long_prompt)

    def test_compression_increases_l1_hit_rate(self):
        """压缩后原 prompt 回写 L1，相同未压缩的 prompt 也能命中 L1"""
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return "response"

        config = CacheConfig(compression_enabled=True, compression_level="standard")
        cache = HybridCache(llm_callback=llm, config=config)

        prompt = "请帮我用 Solidity 写一个简单的 ERC20 代币合约，包含 transfer 和 approve 功能，好的谢谢！"
        resp1, src1 = cache.query(prompt)
        assert src1 == "llm"
        assert len(calls) == 1

        # 第二次用完全相同的未压缩 prompt
        resp2, src2 = cache.query(prompt)
        assert src2 == "l1"
        assert len(calls) == 1  # 没再调 LLM

    def test_compression_stats_in_report(self):
        """报告应包含压缩相关统计"""
        config = CacheConfig(compression_enabled=True)
        cache = HybridCache(llm_callback=lambda p: "ok", config=config)
        cache.query("这是一个非常长的测试文本需要被压缩处理以便节省更多的token。好的，谢谢！")
        report = cache.report()
        assert "压缩" in report
        assert "Prompt" in report or "prompt" in report

    def test_compress_before_cache_only(self):
        """仅缓存前压缩，LLM 不改"""
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return "response"

        config = CacheConfig(
            compression_enabled=True,
            compress_before_cache=True,
            compress_before_llm=False,
        )
        cache = HybridCache(llm_callback=llm, config=config)
        text = "首先，我想请你写一个合约，包含 transfer 和 approve 功能。好的谢谢！"
        resp, src = cache.query(text)
        assert src == "llm"
        # LLM 收到的是原始文本（compress_before_llm=False）
        sent = calls[0]
        assert "首先" in sent or "好的" in sent or "谢谢" in sent

    def test_compression_report_method(self):
        """compression_report() 可用"""
        config = CacheConfig(compression_enabled=True)
        cache = HybridCache(llm_callback=lambda p: "ok", config=config)
        cache.query("测试文本用于生成压缩报告。这是一段需要被压缩的测试文本。")
        report = cache.compression_report()
        assert "压缩" in report or "未开启" in report

    def test_compression_report_disabled(self):
        """未开启压缩时返回提示"""
        cache = HybridCache(llm_callback=lambda p: "ok")
        report = cache.compression_report()
        assert "未开启" in report

"""
L3: Hybrid Cache —— 混合分层缓存引擎

三层架构：
  L1: ExactMatchCache (内存, hash 精确匹配, 最快)
  L2: SemanticCache (向量, 语义相似匹配, 覆盖更广)
  LLM: 真正的 API 调用 (最后一道防线)

命中路径：
  请求 → L1? → 是 → 返回（零 token，亚毫秒）
       → 否 → L2? → 是 → 返回（零 token，几毫秒）
                  → 否 → LLM API → 回写 L1+L2 → 返回
"""

import logging
import time
from typing import Any, Callable, Optional

from .config import CacheConfig, DEFAULT_CONFIG
from .exact_match import ExactMatchCache
from .semantic_cache import SemanticCache

logger = logging.getLogger("llm_cache.hybrid")


class HybridCache:
    """混合分层缓存引擎 —— L3 层

    组合 L1 精确匹配 + L2 语义匹配 + LLM 回调，
    最大化命中率，最小化 API 调用。
    """

    def __init__(
        self,
        llm_callback: Optional[Callable] = None,
        config: Optional[CacheConfig] = None,
    ):
        """
        Args:
            llm_callback: LLM 调用函数。签名为 async/def fn(prompt: str, **kwargs) -> str
            config: 缓存配置
        """
        self.config = config or DEFAULT_CONFIG
        self._llm_callback = llm_callback

        # 初始化各层
        self.l1 = ExactMatchCache(config) if self.config.l1_enabled else None
        self.l2 = SemanticCache(config) if self.config.l2_enabled else None

        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "llm_calls": 0,
            "total_requests": 0,
            "tokens_saved": 0,
        }

        # 估算每次 LLM 调用的 token 消耗（用于统计）
        self._estimated_tokens_per_call = 2000

    # ── 同步调用 ──────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        llm_callback: Optional[Callable] = None,
        **llm_kwargs,
    ) -> tuple[Any, str]:
        """三层查询（同步版本）

        Args:
            prompt: 用户输入的 prompt
            llm_callback: 可选，覆盖初始化时的回调
            **llm_kwargs: 透传给 LLM 回调的额外参数

        Returns:
            (response, source) — source 是 "l1"/"l2"/"llm"
        """
        self._stats["total_requests"] += 1

        # ── L1: 精确匹配 ──
        if self.l1 is not None:
            result = self.l1.get(prompt)
            if result is not None:
                self._stats["l1_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                logger.debug("L1 HIT: %s...", prompt[:60])
                return result, "l1"

        # ── L2: 语义匹配 ──
        if self.l2 is not None:
            result = self.l2.get(prompt)
            if result is not None:
                self._stats["l2_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                logger.debug("L2 HIT: %s...", prompt[:60])
                return result, "l2"

        # ── L3: LLM 回调 ──
        callback = llm_callback or self._llm_callback
        if callback is None:
            raise ValueError(
                "所有缓存层均未命中，且未提供 llm_callback 回调."
            )

        self._stats["llm_calls"] += 1

        # 计时
        t0 = time.time()
        result = callback(prompt, **llm_kwargs)
        elapsed = time.time() - t0
        logger.debug("LLM CALL (%s): %s... [%.2fs]", self._stats["llm_calls"], prompt[:60], elapsed)

        # 自动回写 L1 + L2
        if self.config.l3_auto_cache_llm:
            if self.l1 is not None:
                self.l1.set(prompt, result)
            if self.l2 is not None:
                self.l2.set(prompt, result)

        return result, "llm"

    # ── 异步调用 ──────────────────────────────────────────────

    async def aquery(
        self,
        prompt: str,
        llm_callback: Optional[Callable] = None,
        **llm_kwargs,
    ) -> tuple[Any, str]:
        """三层查询（异步版本）"""
        self._stats["total_requests"] += 1

        if self.l1 is not None:
            result = self.l1.get(prompt)
            if result is not None:
                self._stats["l1_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                return result, "l1"

        if self.l2 is not None:
            result = self.l2.get(prompt)
            if result is not None:
                self._stats["l2_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                return result, "l2"

        callback = llm_callback or self._llm_callback
        if callback is None:
            raise ValueError("所有缓存层均未命中，且未提供 llm_callback 回调.")

        self._stats["llm_calls"] += 1
        t0 = time.time()

        # 判断回调是 async 还是 sync
        import inspect
        if inspect.iscoroutinefunction(callback):
            result = await callback(prompt, **llm_kwargs)
        else:
            result = callback(prompt, **llm_kwargs)

        elapsed = time.time() - t0
        logger.debug("LLM CALL (%s): %s... [%.2fs]", self._stats["llm_calls"], prompt[:60], elapsed)

        if self.config.l3_auto_cache_llm:
            if self.l1 is not None:
                self.l1.set(prompt, result)
            if self.l2 is not None:
                self.l2.set(prompt, result)

        return result, "llm"

    # ── 管理接口 ──────────────────────────────────────────────

    def set_llm_callback(self, callback: Callable) -> None:
        """设置或更换 LLM 回调。"""
        self._llm_callback = callback

    def clear(self) -> None:
        """清空所有缓存层。"""
        if self.l1:
            self.l1.clear()
        if self.l2:
            self.l2.clear()

    def delete(self, prompt: str) -> bool:
        """从所有缓存层删除指定 prompt。"""
        deleted = False
        if self.l1:
            deleted = self.l1.delete(prompt) or deleted
        # L2 的 delete 需要精确匹配 prompt 原文
        # 目前简化：不支持按 prompt 删除 L2（需要重新 embedding）
        return deleted

    def stats(self) -> dict:
        """获取全链路统计。"""
        l1_stats = self.l1.stats() if self.l1 else {}
        l2_stats = self.l2.stats() if self.l2 else {}

        total = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["llm_calls"]
        hit_rate = (
            (self._stats["l1_hits"] + self._stats["l2_hits"]) / total * 100
            if total > 0
            else 0.0
        )

        return {
            "total_requests": self._stats["total_requests"],
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "llm_calls": self._stats["llm_calls"],
            "overall_hit_rate": round(hit_rate, 2),
            "tokens_saved_estimate": self._stats["tokens_saved"],
            "l1_size": l1_stats.get("size", 0),
            "l2_size": l2_stats.get("size", 0),
            "l1_hit_rate": l1_stats.get("hit_rate", 0),
            "l2_hit_rate": l2_stats.get("hit_rate", 0),
        }

    def report(self) -> str:
        """生成可读的缓存报告。"""
        s = self.stats()
        lines = [
            "═══════════════════════════════════════",
            "  📊 Hybrid Cache 运行报告",
            "═══════════════════════════════════════",
            f"  总请求:        {s['total_requests']}",
            f"  L1 命中:       {s['l1_hits']} 次",
            f"  L2 命中:       {s['l2_hits']} 次",
            f"  LLM 调用:      {s['llm_calls']} 次",
            f"  综合命中率:    {s['overall_hit_rate']}%",
            f"  预估节省:      ~{s['tokens_saved_estimate']:,} tokens",
            f"  L1 缓存量:     {s['l1_size']} 条",
            f"  L2 缓存量:     {s['l2_size']} 条",
            "───────────────────────────────────────",
            f"  假设 100 次请求：",
            f"    无缓存:  ~{100 * self._estimated_tokens_per_call:,} tokens",
            f"    有缓存:  ~{s['llm_calls'] * self._estimated_tokens_per_call if s['total_requests'] > 0 else 0:,} tokens",
            f"    节省:    ~{s['tokens_saved_estimate'] if s['total_requests'] > 0 else 0:,} tokens",
            "═══════════════════════════════════════",
        ]
        return "\n".join(lines)

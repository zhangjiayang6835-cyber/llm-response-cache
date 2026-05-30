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

内置 Prompt 压缩（可选开启）：
  开启后缓存查前先压缩 prompt，提高 L1 命中率；
  LLM 调用前再压缩，省 token 开销。
"""

import inspect
import logging
import time
from typing import Any, Callable, Optional

from .config import CacheConfig, DEFAULT_CONFIG
from .exact_match import ExactMatchCache
from .prompt_compressor import CompressionLevel, PromptCompressor
from .semantic_cache import SemanticCache

logger = logging.getLogger("llm_cache.hybrid")


class HybridCache:
    """混合分层缓存引擎 —— L3 层

    组合 L1 精确匹配 + L2 语义匹配 + LLM 回调 + Prompt 压缩，
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

        # 初始化 Prompt 压缩引擎
        if self.config.compression_enabled:
            comp_level = CompressionLevel(self.config.compression_level)
            self.compressor = PromptCompressor(level=comp_level)
            self._compression_stats = {
                "cache_side": 0,
                "llm_side": 0,
                "cache_compression_saved": 0,
                "llm_compression_saved": 0,
            }
        else:
            self.compressor = None

        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "llm_calls": 0,
            "total_requests": 0,
            "tokens_saved": 0,
        }

        # 估算每次 LLM 调用的 token 消耗（用于统计）
        self._estimated_tokens_per_call = 2000

    # ── Prompt 压缩 ──────────────────────────────────────────

    def _maybe_compress(self, prompt: str, side: str) -> str:
        """如果开启了压缩，返回压缩后的 prompt。"""
        if self.compressor is None:
            return prompt

        result = self.compressor.compress(prompt)
        if side == "cache":
            self._compression_stats["cache_side"] += 1
            self._compression_stats["cache_compression_saved"] += result.saved_tokens
        else:
            self._compression_stats["llm_side"] += 1
            self._compression_stats["llm_compression_saved"] += result.saved_tokens

        if result.saved_tokens > 0:
            logger.debug(
                "COMPRESS [%s]: %d→%d tokens (%.0f%%)",
                side,
                result.original_tokens,
                result.compressed_tokens,
                result.compression_ratio * 100,
            )

        return result.compressed

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
            (response, source) — source 是 "compressed" / "l1" / "l2" / "llm"
        """
        self._stats["total_requests"] += 1

        # ── 缓存前压缩（提高命中率） ──
        cache_prompt = self._maybe_compress(prompt, "cache") if self.config.compress_before_cache else prompt

        # ── L1: 精确匹配 ──
        if self.l1 is not None:
            result = self.l1.get(cache_prompt)
            if result is not None:
                self._stats["l1_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                logger.debug("L1 HIT: %s...", cache_prompt[:60])
                return result, "l1"

        # ── L2: 语义匹配 ──
        if self.l2 is not None:
            result = self.l2.get(cache_prompt)
            if result is not None:
                self._stats["l2_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                logger.debug("L2 HIT: %s...", cache_prompt[:60])
                return result, "l2"

        # ── LLM 回调 ──
        callback = llm_callback or self._llm_callback
        if callback is None:
            raise ValueError(
                "所有缓存层均未命中，且未提供 llm_callback 回调."
            )

        self._stats["llm_calls"] += 1

        # LLM 调用前压缩（省 token）
        # compress_before_llm=True 时基于原始 prompt 压缩
        # compress_before_llm=False 时直接用原始 prompt
        if self.config.compress_before_llm:
            llm_prompt = self._maybe_compress(prompt, "llm")
        else:
            llm_prompt = prompt

        # 计时 + 调用
        t0 = time.time()
        result = callback(llm_prompt, **llm_kwargs)
        elapsed = time.time() - t0
        logger.debug(
            "LLM CALL (%s): %s... [%.2fs]",
            self._stats["llm_calls"],
            llm_prompt[:60],
            elapsed,
        )

        # 自动回写 L1 + L2（用 cache_prompt 回写，下次压缩后精确命中）
        if self.config.l3_auto_cache_llm:
            if self.l1 is not None:
                self.l1.set(cache_prompt, result)
            if self.l2 is not None:
                self.l2.set(cache_prompt, result)

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

        cache_prompt = self._maybe_compress(prompt, "cache") if self.config.compress_before_cache else prompt

        if self.l1 is not None:
            result = self.l1.get(cache_prompt)
            if result is not None:
                self._stats["l1_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                return result, "l1"

        if self.l2 is not None:
            result = self.l2.get(cache_prompt)
            if result is not None:
                self._stats["l2_hits"] += 1
                self._stats["tokens_saved"] += self._estimated_tokens_per_call
                return result, "l2"

        callback = llm_callback or self._llm_callback
        if callback is None:
            raise ValueError("所有缓存层均未命中，且未提供 llm_callback 回调.")

        self._stats["llm_calls"] += 1
        if self.config.compress_before_llm:
            llm_prompt = self._maybe_compress(prompt, "llm")
        else:
            llm_prompt = prompt

        t0 = time.time()
        if inspect.iscoroutinefunction(callback):
            result = await callback(llm_prompt, **llm_kwargs)
        else:
            result = callback(llm_prompt, **llm_kwargs)

        elapsed = time.time() - t0
        logger.debug(
            "LLM CALL (%s): %s... [%.2fs]",
            self._stats["llm_calls"],
            llm_prompt[:60],
            elapsed,
        )

        if self.config.l3_auto_cache_llm:
            if self.l1 is not None:
                self.l1.set(cache_prompt, result)
            if self.l2 is not None:
                self.l2.set(cache_prompt, result)

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
        return deleted

    def stats(self) -> dict:
        """获取全链路统计。"""
        l1_stats = self.l1.stats() if self.l1 else {}
        l2_stats = self.l2.stats() if self.l2 else {}
        comp_stats = getattr(self, "_compression_stats", {})
        compressor_stats = self.compressor.stats() if self.compressor else {}

        total_cache = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["llm_calls"]
        hit_rate = (
            (self._stats["l1_hits"] + self._stats["l2_hits"]) / total_cache * 100
            if total_cache > 0
            else 0.0
        )

        return {
            "total_requests": self._stats["total_requests"],
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "llm_calls": self._stats["llm_calls"],
            "overall_hit_rate": round(hit_rate, 2),
            "tokens_saved_by_cache": self._stats["tokens_saved"],
            "tokens_saved_by_compression": compressor_stats.get("total_saved_tokens", 0),
            "tokens_saved_total": self._stats["tokens_saved"] + compressor_stats.get("total_saved_tokens", 0),
            "compression_enabled": self.compressor is not None,
            "compression_avg_ratio": compressor_stats.get("avg_compression_ratio", 0),
            "l1_size": l1_stats.get("size", 0),
            "l2_size": l2_stats.get("size", 0),
            "l1_hit_rate": l1_stats.get("hit_rate", 0),
            "l2_hit_rate": l2_stats.get("hit_rate", 0),
        }

    def report(self) -> str:
        """生成可读的缓存报告。"""
        s = self.stats()
        lines = [
            "═" * 50,
            "  📊 Hybrid Cache 运行报告",
            "═" * 50,
            f"  总请求:        {s['total_requests']}",
            f"  L1 命中:       {s['l1_hits']} 次",
            f"  L2 命中:       {s['l2_hits']} 次",
            f"  LLM 调用:      {s['llm_calls']} 次",
            f"  综合命中率:    {s['overall_hit_rate']}%",
            f"  L1 缓存量:     {s['l1_size']} 条",
            f"  L2 缓存量:     {s['l2_size']} 条",
            "",
            f"  ┌─ Token 节省明细 ─────────────────────",
            f"  缓存命中节省:   ~{s['tokens_saved_by_cache']:,} tokens",
        ]

        if s["compression_enabled"]:
            lines.extend([
                f"  Prompt 压缩节省: ~{s['tokens_saved_by_compression']:,} tokens  (平均 {s['compression_avg_ratio'] * 100:.1f}%)",
            ])
        lines.extend([
            f"  总计节省:       ~{s['tokens_saved_total']:,} tokens",
            "─" * 50,
        ])

        if s["total_requests"] > 0:
            est_per_call = self._estimated_tokens_per_call
            no_cache = s["total_requests"] * est_per_call
            with_cache = s["llm_calls"] * est_per_call - s["tokens_saved_by_compression"]
            with_cache = max(with_cache, 0)
            lines.extend([
                f"  对比（{s['total_requests']} 次请求）：",
                f"    无缓存无压缩:  ~{no_cache:,} tokens",
                f"    有缓存+压缩:   ~{with_cache:,} tokens",
                f"    实际节省:      ~{no_cache - with_cache:,} tokens  ({((no_cache - with_cache) / no_cache * 100):.0f}%)",
            ])

        lines.append("═" * 50)
        return "\n".join(lines)

    def compression_report(self) -> str:
        """获取压缩引擎专项报告。"""
        if self.compressor is None:
            return "⚠️  压缩引擎未开启。设置 config.compression_enabled=True 启用。"
        return self.compressor.report()

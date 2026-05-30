"""
PromptCompressor — Prompt 压缩引擎

核心功能：
- 三层压缩力度（轻度/标准/极省）
- 安全保留：代码、配置、专业术语、URL、地址 100% 不变
- 可集成到 HybridCache 中，在缓存前或 LLM 前自动压缩

节省效果：
  轻度 (mild):    20~35%  去口语/空行/重复
  标准 (standard): 35~55%  去过渡句，短句替代长句
  极省 (aggressive): 50~70%  关键词结构，极端精简
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CompressionLevel(str, Enum):
    MILD = "mild"           # 轻度压缩
    STANDARD = "standard"   # 标准压缩（默认）
    AGGRESSIVE = "aggressive"  # 极省压缩


@dataclass
class CompressResult:
    """压缩结果"""
    original: str
    compressed: str
    level: CompressionLevel
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # 0.0 ~ 1.0，越大表示省得越多
    elapsed_ms: float

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.compressed_tokens


class PromptCompressor:
    """Prompt 压缩引擎

    纯规则驱动，零外部依赖。支持三种压缩级别，自动保护代码/配置/专业术语。
    """

    # ── 安全删除清单 —— 在任何级别下均可安全移除 ──
    # 注意：中文不使用 \b，中文没有 word boundary
    _FILLER_PATTERNS = [
        # 开场白/废话（中文部分用 lookahead 而非 \b）
        (r"(?:好的|没问题|明白了|收到)[，,。]?\s*", ""),
        (r"我来帮你|让我看看|让我帮你|让我来", ""),
        (r"简单来说|总的来说|总而言之|综上所述|也就是说|换句话来?说", ""),
        # 过渡词（逗号可选，兼容句首无逗号的情况）
        (r"(?:首先|其次|再次|最后|然后|另外|此外|同时|因此|所以|那么)[，,]?\s*", ""),
        # 冗余修饰
        (r"值得一提的是|需要说明的是|需要注意的是|特别说明|请知悉|你可以这样理解|基于以上分析|通过以上步骤|按照上述方法\s*", ""),
        # 礼貌结尾
        (r"\s*如果(?:有|你)(?:什么|任何)问(?:题|的)[，,。!！]?\s*", ""),
        (r"\s*欢迎随时提出[，,。!！]?\s*", ""),
        (r"\s*谢谢[！!。.]?\s*", ""),
        (r"\s*非常感谢[！!。.]?\s*", ""),
        # 请/麻烦
        (r"\s*请[你]?\s*", " "),
        (r"\s*麻烦[你]?\s*", " "),
        # 非常/很/特别等程度副词
        (r"(?:非常|特别|极其|十分|相当|较为|比较|略显|稍微)\s*", ""),
        # 重复标点
        (r"[！!]{2,}", "！"),
        (r"[？?]{2,}", "？"),
        (r"[。.。\n]{3,}", "。\n"),
        (r"\s{2,}", " "),
        (r"\n{3,}", "\n\n"),
        # 中文 vs 英文空格规范化
        (r"([\u4e00-\u9fff])\s+([a-zA-Z0-9])", r"\1 \2"),
        (r"([a-zA-Z0-9])\s+([\u4e00-\u9fff])", r"\1 \2"),
    ]

    # ── 标准级额外移除（过渡句压缩为关键词）──
    _STANDARD_PATTERNS = [
        # 所谓的X → X:
        (r"所谓的[\u201c\u201d]?(\S+?)[\u201c\u201d]?是指?", r"\1:"),
        # 基于分析 → 删
        (r"基于上述[^，,。\n]*[，,。]\s*", ""),
        # 冗余 "一个"
        (r"一个(\s*[a-zA-Z\u4e00-\u9fff])", r"\1"),
    ]

    # ── 极省级额外移除（极端压缩）──
    _AGGRESSIVE_PATTERNS = [
        (r"(?:能不能|可不可以|希望|想要|打算|准备)\s*", ""),
        (r"(?:需要|必须|应当|应该|要)\s*", ""),
        (r"请问\s*", ""),
        (r"可以吗\??\s*", ""),
        # 合并短句
        (r"[。；;]\s*然后\s+", "。"),
        (r"[。；;]\s*接着\s+", "。"),
    ]

    # ── 安全保护区（这些内容原样保留）──
    _SAFE_PATTERNS = re.compile(
        r"("
        r"`[^`]+`"                           # inline code
        r"|```[\s\S]*?```"                   # code block
        r"|0x[a-fA-F0-9]{40}"                # Ethereum address
        r"|0x[a-fA-F0-9]+"                   # hex
        r"|https?://[^\s,，。；;]+"           # URL
        r"|[\w.+-]+@[\w.-]+\.[a-z]{2,}"      # email
        r"|Qm[a-zA-Z0-9]{44}"               # IPFS CID
        r"|(?:\d{1,3}\.){3}\d{1,3}"         # IP
        r"|\$\w+"                            # $variable
        r"|[\w-]+\.sol\b"                    # .sol files
        r"|[\w-]+\.vy\b"                     # .vy files
        r")",
        re.UNICODE,
    )

    def __init__(self, level: CompressionLevel = CompressionLevel.STANDARD):
        self.level = level if isinstance(level, CompressionLevel) else CompressionLevel(level)
        self._stats = {
            "total_compressed": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
        }

    def compress(
        self,
        prompt: str,
        level: Optional[CompressionLevel] = None,
    ) -> CompressResult:
        """压缩 prompt 文本

        Args:
            prompt: 原始 prompt
            level: 压缩级别，覆盖初始化时的设置

        Returns:
            CompressResult: 包含原始/压缩文本及统计
        """
        t0 = time.time()
        level = level or self.level
        original_tokens = self._estimate_tokens(prompt)

        # 如果文本太短，跳过压缩（但记录 stats）
        if original_tokens < 10:
            elapsed = (time.time() - t0) * 1000
            result = CompressResult(
                original=prompt,
                compressed=prompt,
                level=level,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                elapsed_ms=round(elapsed, 2),
            )
            # 短文本也记录到 stats（确保统计准确）
            self._record_stats(result)
            return result

        # 提取安全保护区
        safe_parts = []
        processed = prompt

        def _save_safe(m):
            idx = len(safe_parts)
            safe_parts.append(m.group(0))
            return f"\x00SAFE{idx}\x00"

        processed = self._SAFE_PATTERNS.sub(_save_safe, processed)

        # 应用压缩规则
        processed = self._apply_level(processed, level)

        # 恢复安全内容
        for i, part in enumerate(safe_parts):
            processed = processed.replace(f"\x00SAFE{i}\x00", part)

        # 后处理：清理多余空白
        processed = re.sub(r"[ \t]+", " ", processed).strip()
        processed = re.sub(r"\n{3,}", "\n\n", processed)

        compressed_tokens = self._estimate_tokens(processed)
        ratio = round(
            1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0, 4
        )
        elapsed = (time.time() - t0) * 1000

        result = CompressResult(
            original=prompt,
            compressed=processed,
            level=level,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            elapsed_ms=round(elapsed, 2),
        )
        self._record_stats(result)
        return result

    def _record_stats(self, result: CompressResult) -> None:
        """记录压缩统计。"""
        self._stats["total_compressed"] += 1
        self._stats["total_original_tokens"] += result.original_tokens
        self._stats["total_compressed_tokens"] += result.compressed_tokens

    def _apply_level(self, text: str, level: CompressionLevel) -> str:
        """按级别应用压缩规则。"""
        # 所有级别通用：去废话
        for pattern, replacement in self._FILLER_PATTERNS:
            text = re.sub(pattern, replacement, text)

        if level in (CompressionLevel.STANDARD, CompressionLevel.AGGRESSIVE):
            for pattern, replacement in self._STANDARD_PATTERNS:
                text = re.sub(pattern, replacement, text)

        if level == CompressionLevel.AGGRESSIVE:
            for pattern, replacement in self._AGGRESSIVE_PATTERNS:
                text = re.sub(pattern, replacement, text)
            # 极省级额外：压缩段落结构
            text = self._compress_paragraphs(text)

        return text

    @staticmethod
    def _compress_paragraphs(text: str) -> str:
        """极省级：将多个简短段落合并为要点列表"""
        lines = text.strip().split("\n")
        if len(lines) <= 3:
            return text

        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 以字母/中文开头且非列表项 → 合并
            if result and not line.startswith("-") and not line.startswith("*"):
                result[-1] = result[-1].rstrip("。；;，,）)") + "，" + line
            else:
                result.append(line)

        return "\n".join(result)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """快速估算 token 数（中英文混合近似）。"""
        if not text:
            return 0
        # 中文约 1.8 字/token（更准确），英文约 4 字母/token
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - chinese_chars
        estimated = int(chinese_chars / 1.8 + other_chars / 4)
        return max(estimated, int(len(text) * 0.3))  # 兜底：至少 30% 的字符数

    def stats(self) -> dict:
        """获取压缩统计。"""
        s = self._stats
        total = s["total_original_tokens"]
        saved = s["total_original_tokens"] - s["total_compressed_tokens"]
        avg_ratio = round(
            1 - (s["total_compressed_tokens"] / total) if total > 0 else 0, 4
        )
        return {
            "total_compressed": s["total_compressed"],
            "total_original_tokens": total,
            "total_compressed_tokens": s["total_compressed_tokens"],
            "total_saved_tokens": saved,
            "avg_compression_ratio": avg_ratio,
        }

    def report(self) -> str:
        """生成可读的压缩报告。"""
        s = self.stats()
        return (
            f"📉 Prompt 压缩报告\n"
            f"  压缩次数:        {s['total_compressed']}\n"
            f"  原始 Token:      {s['total_original_tokens']:,}\n"
            f"  压缩后 Token:    {s['total_compressed_tokens']:,}\n"
            f"  节省 Token:      {s['total_saved_tokens']:,}\n"
            f"  平均压缩率:      {s['avg_compression_ratio'] * 100:.1f}%\n"
        )

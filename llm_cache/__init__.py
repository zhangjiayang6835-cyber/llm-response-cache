"""LLM Response Cache — 三层缓存引擎：Exact Match → Semantic → Hybrid + Prompt 压缩"""

from .config import CacheConfig
from .exact_match import ExactMatchCache
from .hybrid_cache import HybridCache
from .prompt_compressor import CompressionLevel, PromptCompressor, CompressResult
from .semantic_cache import SemanticCache

__all__ = [
    "ExactMatchCache",
    "SemanticCache",
    "HybridCache",
    "CacheConfig",
    "PromptCompressor",
    "CompressionLevel",
    "CompressResult",
]
__version__ = "0.2.0"

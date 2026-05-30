"""缓存配置"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheConfig:
    """全局缓存配置"""

    # Exact Match Cache (L1)
    l1_enabled: bool = True
    l1_max_size: int = 1000
    l1_ttl_seconds: int = 300  # 5 分钟

    # Semantic Cache (L2)
    l2_enabled: bool = True
    l2_similarity_threshold: float = 0.95
    l2_max_size: int = 5000
    l2_ttl_seconds: int = 1800  # 30 分钟
    l2_embedding_model: str = "all-MiniLM-L6-v2"  # 本地 embedding 模型
    l2_embedding_dim: int = 384

    # Hybrid
    l3_llm_fallback: bool = True
    l3_auto_cache_llm: bool = True  # LLM 返回后自动回写 L1+L2

    # Redis (可选)
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0

    def __post_init__(self):
        _redis_host = os.environ.get("LLM_CACHE_REDIS_HOST")
        if _redis_host:
            self.redis_host = _redis_host

    @property
    def use_redis(self) -> bool:
        return self.redis_host is not None


# 默认全局配置
DEFAULT_CONFIG = CacheConfig()

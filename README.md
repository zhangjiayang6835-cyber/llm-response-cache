# LLM Response Cache 🚀

> 三层 LLM API 响应缓存引擎：**Exact Match → Semantic Cache → Hybrid Cache**
>
> 大幅减少 LLM API 调用，降低 Token 开销，钱包友好。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 问题

每次 LLM API 调用都在烧钱。尤其是：

- 💸 **重复调试**：同一个问题反复问，每次都付全款
- 💸 **同义查询**：用户问"AMM 是什么？"和"什么是 AMM？"——语义相同，两次扣费
- 💸 **并发场景**：相同 prompt 多个请求同时进来，N 次扣费

**LLM Response Cache** 用三层缓存架构解决这个问题，命中率可达 **70~80%**，Token 开销降到原来的 **20~30%**。

---

## 三层缓存架构

```
                    ┌──────────────┐
  用户 Prompt  →    │  L1 精确匹配  │  ← 相同 prompt → 返回 (零 Token, 亚毫秒)
                    ├──────────────┤
                    │  L2 语义匹配  │  ← 相似 prompt → 返回 (零 Token, 几毫秒)
                    ├──────────────┤
                    │  L3 LLM API  │  ← 全新请求 → 调 API + 回写 L1+L2
                    └──────────────┘
```

### L1: ExactMatchCache（精确匹配）

| 维度 | 值 |
|------|-----|
| 匹配方式 | SHA256 全文 hash |
| 速度 | O(1)，< 1μs |
| 淘汰策略 | LRU（最近最久未使用） |
| 容量上限 | 可配（默认 1000 条） |
| TTL | 可配（默认 5 分钟） |
| 后端 | 内存 / Redis（可选） |

**适用**：完全相同的 prompt，如模板查询、固定调试问题、重复命令。

### L2: SemanticCache（语义匹配）

| 维度 | 值 |
|------|-----|
| 匹配方式 | Embedding + Cosine Similarity |
| 速度 | 搜索随库增长（默认 ≤5000 条时 < 50ms）|
| 相似度阈值 | 可调（默认 0.95，语义等价级别） |
| 淘汰策略 | FIFO（先进先出） |
| Embedding 模型 | `all-MiniLM-L6-v2`（默认，384 维） |
| Fallback 方案 | 无 GPU/模型时自动降级到 TF-IDF |

**适用**：同义不同文的问题。如"解释一下流动性池"和"Liquidity Pool 是什么"→ 命中同一条。

### L3: HybridCache（混合引擎）

- 收到请求 → 先查 L1（精确 hash 匹配）
- L1 Miss → 查 L2（语义相似匹配）
- L2 Miss → 调 LLM API → 自动回写 L1 + L2

**核心设计**：*"能省就省，省不了就缓存下来下次省。"*

---

## 快速上手

### 安装

```bash
# 核心功能（无外部依赖，纯 Python）
pip install llm-response-cache

# 全功能（含 embedding 语义匹配）
pip install 'llm-response-cache[all]'

# 仅 embedding
pip install 'llm-response-cache[embedding]'
```

### 基础用法

```python
from llm_cache import ExactMatchCache, SemanticCache, HybridCache

# ── L1: 精确匹配 ──
l1 = ExactMatchCache()
l1.set("给我写一个 ERC20 合约", "// ERC20 合约代码...")
result = l1.get("给我写一个 ERC20 合约")  # 命中！零 Token
print(result)

# ── L2: 语义匹配（需要 sentence-transformers）──
l2 = SemanticCache()
l2.set("什么是闪电贷攻击？", "闪电贷攻击是...")
result = l2.get("闪电贷攻击是什么？")  # 语义相似 → 命中！
print(result)

# ── L3: 三层混合（接入真实 LLM）──
def my_llm(prompt):
    # 这里替换成真实的 OpenAI / Claude API 调用
    return f"这是 '{prompt}' 的回复"

hybrid = HybridCache(llm_callback=my_llm)

# 第一次：L1 Miss → L2 Miss → 调 LLM → 回写
resp1, src1 = hybrid.query("解释 Solidity 的 delegatecall")
print(f"[{src1}] {resp1[:40]}...")

# 第二次（相同 prompt）：L1 命中
resp2, src2 = hybrid.query("解释 Solidity 的 delegatecall")
print(f"[{src2}] {resp2[:40]}...")

# 第三次（语义相似）：L2 命中
resp3, src3 = hybrid.query("delegatecall 是什么？")
print(f"[{src3}] {resp3[:40]}...")

# 生成报告
print(hybrid.report())
```

### 接入 OpenAI

```python
from openai import OpenAI
from llm_cache import HybridCache

client = OpenAI(api_key="sk-xxx")

def call_openai(prompt):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

cache = HybridCache(llm_callback=call_openai)

# 第一次调 API
resp, src = cache.query("写一个 Solidity 的 Reentrancy Guard")
print(f"[{src}] {resp[:80]}...")

# 第二次用缓存
resp, src = cache.query("写一个 Solidity 的 Reentrancy Guard")
print(f"[{src}] {resp[:80]}...")  # [l1] 直接命中

cache.report()  # 查看节省数据
```

### 异步支持

```python
cache = HybridCache(llm_callback=async_openai_call)
resp, src = await cache.aquery("解释 EIP-1559")
```

---

## 配置详解

```python
from llm_cache import CacheConfig

config = CacheConfig(
    # L1
    l1_max_size=2000,
    l1_ttl_seconds=600,      # 10 分钟

    # L2
    l2_similarity_threshold=0.92,  # 适当降低阈值提高命中率
    l2_max_size=10000,
    l2_ttl_seconds=3600,     # 1 小时
    l2_embedding_model="all-mpnet-base-v2",  # 更准（768 维）

    # L3
    l3_auto_cache_llm=True,
)

cache = HybridCache(llm_callback=call_llm, config=config)
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `LLM_CACHE_REDIS_HOST` | 启用 Redis 持久化 |

---

## 性能预期（实测参考）

| 场景 | 无缓存 | L1 精确 | L2 语义 | L1+L2 混合 |
|------|--------|---------|---------|-----------|
| 完全重复查询 | 100% token | 100% 节省 | 100% 节省 | 100% 节省 |
| 同义查询（10 个变体） | 100% token | 10% 节省 | 90% 节省 | 90% 节省 |
| 全新查询（无重复） | 100% token | 0% 节省 | 0% 节省 | 0% 节省 |
| **综合预估** | **100%** | **~20%** | **~50%** | **~70~80%** |

**结论**：混合模式最划算，推荐首选 HybridCache。

---

## 为什么做这个项目？

> 我是 Web3 安全审计方向的开发者，日常大量使用 LLM API。钱包烧得有点快，就想做个缓存层解决这个问题。
>
> 核心设计理念：**不改变你的 LLM 调用方式，只加一层透明的缓存。**
>
> 纯 Python、零配置、开箱即用。

---

## 项目结构

```
llm-response-cache/
├── llm_cache/
│   ├── __init__.py        # 导出接口
│   ├── config.py          # 配置管理
│   ├── exact_match.py     # L1: 精确匹配缓存
│   ├── semantic_cache.py  # L2: 语义相似缓存
│   └── hybrid_cache.py    # L3: 混合引擎
├── examples/
│   ├── basic_usage.py     # 基础用法示例
│   └── openai_integration.py  # OpenAI 集成示例
├── tests/
│   ├── test_exact_match.py
│   ├── test_semantic_cache.py
│   └── test_hybrid_cache.py
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## 技术要点

### Embedding 选择

| 模型 | 维度 | 速度 | 准确率 | 推荐场景 |
|------|------|------|--------|---------|
| `all-MiniLM-L6-v2`（默认） | 384 | ⚡ 快 | 良好 | 通用 |
| `all-mpnet-base-v2` | 768 | 🐢 稍慢 | 优秀 | 高精度 |
| TF-IDF (fallback) | 384 | ⚡ 极快 | 一般 | 无 GPU / 快速部署 |

### Cosine Similarity 阈值调优

| 阈值 | 命中率 | 误命中风险 | 推荐场景 |
|------|--------|-----------|---------|
| 0.99 | 低 | 极低 | 代码生成（格式敏感） |
| **0.95** (默认) | **中** | **低** | **通用** |
| 0.90 | 高 | 中 | 文档生成、FAQ |
| 0.85 | 极高 | 高 | 容忍高/低成本场景 |

### Hash vs Embedding

```
Exact Match (Hash)          Semantic (Embedding)
    "查余额"  ──→ 哈希 ──→ key_1         "查余额" ──→ [0.12, 0.87, ...]
    "查余额"  ──→ 哈希 ──→ key_1  ✓         "查询余额" ──→ [0.11, 0.86, ...]  ✓
    "查询余额" ──→ 哈希 ──→ key_2  ✗      "show balance" ──→ [0.09, 0.82, ...]  ✓
```

## LICENSE

MIT

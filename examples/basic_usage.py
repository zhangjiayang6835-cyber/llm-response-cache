"""
基础用法示例
"""

from llm_cache import ExactMatchCache, SemanticCache, HybridCache


def demo_l1():
    """L1: 精确匹配缓存"""
    print("=" * 50)
    print("L1: ExactMatchCache 演示")
    print("=" * 50)

    cache = ExactMatchCache()
    cache.set("给我写一个 ERC20 合约", "// SPDX-License-Identifier: MIT\n// ERC20 合约模板...")
    cache.set("Solidity 有哪些数据类型", "uint, int, address, string, bytes, bool, enum, struct...")

    # 精确命中
    result = cache.get("给我写一个 ERC20 合约")
    print(f"[命中] 精确匹配: {result[:40]}...")

    # 精确未命中（即使语义相同，拼写不同也不行）
    result = cache.get("写一个 ERC20 合约给我")
    print(f"[未命中] {result}")

    print(f"统计: {cache.stats()}\n")


def demo_l2():
    """L2: 语义匹配缓存"""
    print("=" * 50)
    print("L2: SemanticCache 演示")
    print("=" * 50)

    cache = SemanticCache()
    cache.set("什么是闪电贷攻击？", "闪电贷攻击（Flash Loan Attack）是...")
    cache.set("解释 Solidity 的 delegatecall", "delegatecall 是一种低级调用...")

    # 精确命中
    result = cache.get("什么是闪电贷攻击？")
    if result:
        print(f"[命中] 精确: {result[:50]}...")

    # 语义命中（如果相似度够高）
    result = cache.get("闪电贷攻击是什么？")
    if result:
        print(f"[命中] 语义: {result[:50]}...")
    else:
        print("[未命中] 语义相似但低于阈值")

    print(f"统计: {cache.stats()}\n")


def demo_l3():
    """L3: 混合缓存"""
    print("=" * 50)
    print("L3: HybridCache 演示")
    print("=" * 50)

    call_count = 0

    def fake_llm(prompt):
        """模拟 LLM API 调用"""
        nonlocal call_count
        call_count += 1
        return f"[LLM 回复] 你问了: '{prompt[:40]}...'"

    cache = HybridCache(llm_callback=fake_llm)

    # 第一次：Miss → Miss → LLM
    resp, src = cache.query("解释 AMM 自动做市商机制")
    print(f"[{src}] {resp}")

    # 第二次：L1 命中（完全相同）
    resp, src = cache.query("解释 AMM 自动做市商机制")
    print(f"[{src}] {resp} (零 Token 开销)")

    # 第三次：L2 命中（语义相似）
    resp, src = cache.query("AMM 自动做市商是什么原理")
    if src == "l2":
        print(f"[{src}] 语义命中！(零 Token 开销)")
    else:
        print(f"[{src}] {resp}")

    print(f"总 LLM 调用次数: {call_count}")
    print()
    print(cache.report())


if __name__ == "__main__":
    demo_l1()
    demo_l2()
    demo_l3()

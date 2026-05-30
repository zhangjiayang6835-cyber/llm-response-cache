"""
OpenAI 集成示例

运行前请设置环境变量：
  export OPENAI_API_KEY=sk-xxx

或者直接替换下面的 api_key。
"""

import os

from llm_cache import HybridCache

# ── 真实 OpenAI 集成 ──────────────────────────────────────────

def openai_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    """调用 OpenAI Chat Completion API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-your-key"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except ImportError:
        return f"[模拟 OpenAI 响应] '{prompt}' 的回复"
    except Exception as e:
        return f"[API Error] {e}"


def main():
    print("=" * 50)
    print("OpenAI + HybridCache 集成")
    print("=" * 50)

    cache = HybridCache(llm_callback=openai_chat)

    questions = [
        "用 Solidity 写一个简单的 ERC20 代币合约",
        "用 Solidity 写一个简单的 ERC20 代币合约",  # 重复 → L1
        "Solidity 实现一个 ERC20 代币合约",          # 相似 → L2
        "什么是 Reentrancy Attack？",
        "什么是 Reentrancy Attack？",                # 重复 → L1
        "重入攻击是什么？",                           # 相似 → L2？
        "Solidity 的 modifier 是什么？",
    ]

    for q in questions:
        resp, src = cache.query(q)
        status = f"[{src.upper()}]" if src in ("l1", "l2") else f"[{src.upper()} 💸]"
        print(f"{status} {q[:50]:<50} → {resp[:50]}...")

    print()
    print(cache.report())


if __name__ == "__main__":
    main()

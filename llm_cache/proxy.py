"""
LLM API 缓存代理服务器

在 Hermes Agent 和 LLM Provider 之间架设一层透明缓存代理。
Hermes 把 base_url 指向这个代理，所有 API 调用自动走缓存。

使用方式：
  # 启动代理（默认端口 8080）
  python -m llm_cache.proxy

  # 配置 Hermes 指向代理
  hermes config set model.base_url http://localhost:8080/v1

  # 验证代理是否正常工作
  curl http://localhost:8080/v1/models

架构：
  Hermes Agent  ──→  Proxy (localhost:8080)  ──→  LLM Provider (DeepSeek/OpenAI/...)
                        │
                        ├─ L1 ExactMatchCache (内存)
                        ├─ L2 SemanticCache (向量)
                        └─ PromptCompressor (可选压缩)
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# 将项目根目录加入 sys.path（支持直接运行）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_cache import CacheConfig, HybridCache

logger = logging.getLogger("llm_cache.proxy")

# ── FastAPI App ──────────────────────────────────────────────

app = FastAPI(title="LLM Response Cache Proxy", version="0.2.0")

# ── 全局缓存实例 ────────────────────────────────────────────

_cache: Optional[HybridCache] = None
_upstream_base_url: str = ""
_api_key: str = ""
_compression_enabled: bool = False
_http_client: Optional[httpx.AsyncClient] = None


def _init_cache(config: CacheConfig, upstream_url: str, api_key: str) -> None:
    """初始化全局缓存。"""
    global _cache, _upstream_base_url, _api_key, _http_client, _compression_enabled

    # 修正 upstream URL（去掉 /v1/chat/completions 等后缀，只留 base）
    parsed = urlparse(upstream_url)
    _upstream_base_url = f"{parsed.scheme}://{parsed.netloc}"

    if parsed.path and parsed.path != "/":
        # 如果路径包含 /v1，保留到 /v1
        if "/v1" in parsed.path:
            _upstream_base_url += parsed.path[: parsed.path.index("/v1") + 2]
        else:
            _upstream_base_url += parsed.path.rstrip("/")

    _api_key = api_key
    _compression_enabled = config.compression_enabled

    _cache = HybridCache(config=config)

    # 创建 HTTP 客户端（复用连接）
    _http_client = httpx.AsyncClient(
        base_url=_upstream_base_url,
        timeout=httpx.Timeout(120.0, connect=10.0),
        follow_redirects=True,
    )

    logger.info(
        "缓存代理已启动: 上游=%s | 压缩=%s | L1(%d条) | L2(%d条, 阈值%.2f)",
        _upstream_base_url,
        "开启" if _compression_enabled else "关闭",
        config.l1_max_size,
        config.l2_max_size,
        config.l2_similarity_threshold,
    )


# ── 路由 ────────────────────────────────────────────────────


@app.api_route("/v1/models", methods=["GET", "HEAD"])
async def list_models(request: Request):
    """透传模型列表查询，不做缓存。"""
    headers = _build_headers(request)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_upstream_base_url}/v1/models", headers=headers, timeout=30.0
        )
        return JSONResponse(
            content=resp.json(), status_code=resp.status_code,
            headers={"content-type": "application/json"},
        )


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """核心：缓存 chat completions 请求。"""
    if _cache is None:
        return JSONResponse(
            content={"error": "缓存未初始化"}, status_code=500
        )

    # 提取请求体
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            content={"error": "缺少 messages 字段"}, status_code=400
        )

    # 从 messages 中提取最终的 user prompt
    prompt = _extract_prompt(messages)

    # 检查是否请求流式输出
    stream = body.get("stream", False)

    # 构造 LLM 回调（发给真正的上游）
    async def call_upstream(p: str, **kwargs) -> str:
        """发送请求到真正的 LLM Provider。"""
        # 替换 messages 中的最后一条 user 内容
        modified_body = _update_messages(body, p)
        headers = _build_headers(request)

        resp = await _http_client.post(
            "/v1/chat/completions",
            json=modified_body,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        # 提取回复文本
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "")

        return str(data)

    # L1/L2 缓存查询 + LLM fallback（自动回写缓存）
    response_text, source = await _cache.aquery(prompt, llm_callback=call_upstream)

    # 构造 OpenAI 格式响应
    result = {
        "id": "chatcmpl-cached",
        "object": "chat.completion",
        "created": 1234567890,
        "model": body.get("model", "unknown"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_source": source,
        },
        "x-cache-source": source,  # 自定义 header（调试用）
    }

    if stream:
        # 流式支持的简化版：非流式响应但标记 stream
        result["x-cache-stream"] = "simulated"

    return JSONResponse(content=result)


@app.api_route("/health", methods=["GET"])
async def health():
    """健康检查。"""
    s = _cache.stats() if _cache else {}
    return JSONResponse(content={
        "status": "ok",
        "cache_hit_rate": s.get("overall_hit_rate", 0),
        "l1_size": s.get("l1_size", 0),
        "l2_size": s.get("l2_size", 0),
        "compression_enabled": _compression_enabled,
    })


@app.api_route("/stats", methods=["GET"])
async def stats():
    """缓存统计报告。"""
    if _cache is None:
        return JSONResponse(content={"error": "缓存未初始化"}, status_code=500)
    return JSONResponse(content=_cache.stats())


@app.api_route("/report", methods=["GET"])
async def report():
    """缓存运行报告（文本格式）。"""
    if _cache is None:
        return JSONResponse(content={"error": "缓存未初始化"}, status_code=500)
    return JSONResponse(content={"report": _cache.report()})


@app.api_route("/clear", methods=["POST"])
async def clear():
    """清空缓存。"""
    if _cache:
        _cache.clear()
    return JSONResponse(content={"status": "ok", "message": "缓存已清空"})


# ── 辅助函数 ────────────────────────────────────────────────


def _extract_prompt(messages: list[dict]) -> str:
    """从 messages 列表中提取最终的 user prompt 文本。"""
    # 优先取最后一条 user/assistant 消息
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and isinstance(content, str) and content.strip():
            return content
        # 处理 content 为 list（多模态消息）
        if role == "user" and isinstance(content, list):
            texts = [
                c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"
            ]
            if texts:
                return " ".join(texts)

    # fallback: 取最后一条非 system 消息的 content
    for msg in reversed(messages):
        if msg.get("role") != "system":
            c = msg.get("content", "")
            if isinstance(c, str) and c.strip():
                return c

    return messages[-1].get("content", "") if messages else ""


def _update_messages(messages: list[dict], new_prompt: str) -> list[dict]:
    """将 messages 中最后一条 user 消息的内容替换为新 prompt。"""
    modified = list(messages)
    for i in range(len(modified) - 1, -1, -1):
        if modified[i].get("role") == "user":
            content = modified[i].get("content", "")
            if isinstance(content, str) and content.strip():
                modified[i] = {**modified[i], "content": new_prompt}
                break
    return modified


def _build_headers(request: Request) -> dict:
    """构建转发请求头。"""
    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }
    # 透传部分原始 header
    for key in ["x-request-id", "x-api-key", "user-agent"]:
        if key in request.headers:
            headers[key] = request.headers[key]
    return headers


# ── CLI 入口 ────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LLM Response Cache Proxy")
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--upstream", type=str,
                        default=os.environ.get("LLM_CACHE_UPSTREAM", ""),
                        help="上游 LLM API base URL (如 https://api.deepseek.com)")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("LLM_CACHE_API_KEY", ""),
                        help="上游 API Key")
    parser.add_argument("--compress", action="store_true",
                        help="开启 Prompt 压缩")
    parser.add_argument("--compression-level", type=str, default="standard",
                        choices=["mild", "standard", "aggressive"],
                        help="压缩级别 (默认: standard)")
    parser.add_argument("--l1-size", type=int, default=1000, help="L1 缓存容量")
    parser.add_argument("--l2-size", type=int, default=5000, help="L2 缓存容量")
    parser.add_argument("--l2-threshold", type=float, default=0.95, help="L2 相似度阈值")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # 日志配置
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 确定上游地址
    upstream = args.upstream
    if not upstream:
        # 尝试从 Hermes 配置获取
        config_yaml = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(config_yaml):
            try:
                import yaml
                with open(config_yaml) as f:
                    cfg = yaml.safe_load(f)
                upstream = (cfg.get("model", {}).get("base_url", "")
                            or cfg.get("providers", {}).get("deepseek", {}).get("base_url", ""))
            except Exception:
                pass

    if not upstream:
        print("⚠️  未指定上游地址。使用 --upstream 或设置 LLM_CACHE_UPSTREAM")
        print(f"   示例: python -m llm_cache.proxy --upstream https://api.deepseek.com")
        sys.exit(1)

    # 确定 API Key
    api_key = args.api_key
    if not api_key:
        # 尝试从 Hermes .env 获取
        env_path = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if "DEEPSEEK_API_KEY" in line:
                        api_key = line.split("=", 1)[1].strip().strip("\"'")
                        break

    if not api_key:
        print("⚠️  未指定 API Key。使用 --api-key 或设置 LLM_CACHE_API_KEY")
        sys.exit(1)

    # 构建缓存配置
    config = CacheConfig(
        compression_enabled=args.compress,
        compression_level=args.compression_level,
        l1_max_size=args.l1_size,
        l2_max_size=args.l2_size,
        l2_similarity_threshold=args.l2_threshold,
    )

    # 初始化缓存
    _init_cache(config, upstream, api_key)

    # 启动服务器
    print(f"\n🚀 LLM Response Cache Proxy")
    print(f"   监听:     http://{args.host}:{args.port}")
    print(f"   上游:     {upstream}")
    print(f"   压缩:     {'开启' if args.compress else '关闭'} ({args.compression_level})")
    print(f"   L1:       {args.l1_size} 条 | L2: {args.l2_size} 条 (阈值 {args.l2_threshold})")
    print(f"\n   配置 Hermes:")
    print(f"     hermes config set model.base_url http://{args.host}:{args.port}/v1")
    print(f"     hermes config set model.api_key sk-cache-proxy")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()

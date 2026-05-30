"""
缓存代理集成测试
"""

import json
import sys
import threading
import time
import uvicorn
from multiprocessing import Process

import httpx
import pytest


def _run_proxy(port: int):
    """在子进程中启动代理。"""
    from llm_cache.proxy import app, _init_cache
    from llm_cache.config import CacheConfig

    config = CacheConfig(l1_max_size=100, l2_enabled=False)
    _init_cache(config, "https://api.deepseek.com", "test-key")

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


class TestProxy:
    @pytest.fixture(autouse=True)
    def setup_proxy(self):
        """启动和停止代理。"""
        import random
        port = random.randint(18000, 19000)
        self.proc = Process(target=_run_proxy, args=(port,), daemon=True)
        self.proc.start()
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        time.sleep(2)  # 等待启动
        yield
        self.proc.terminate()
        self.proc.join(timeout=5)

    def test_health(self):
        """健康检查"""
        resp = httpx.get(f"{self.base}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_stats(self):
        """统计接口"""
        resp = httpx.get(f"{self.base}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data

    def test_clear(self):
        """清空缓存"""
        resp = httpx.post(f"{self.base}/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_chat_completion_missing_messages(self):
        """缺少 messages 时返回 400"""
        resp = httpx.post(
            f"{self.base}/v1/chat/completions",
            json={"model": "deepseek-chat"},
        )
        assert resp.status_code == 400

    def test_chat_completion_calls_upstream(self):
        """缓存未命中时尝试调用上游（会失败但不会 crash）"""
        resp = httpx.post(
            f"{self.base}/v1/chat/completions",
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Hello, this is a test prompt"}
                ],
            },
        )
        # 上游调用会失败（api-key 无效），但代理不应该 crash
        assert resp.status_code in (200, 401, 500)

#!/usr/bin/env python
"""
启动 Hermes 缓存代理包装器。
从 Hermes .env 读取 API Key，启动缓存代理，配置 Hermes。
"""
import os
import re
import subprocess
import sys
import time

HERMES_HOME = os.environ.get(
    "HERMES_HOME",
    os.path.expandvars(r"%LOCALAPPDATA%\hermes"),
)
ENV_PATH = os.path.join(HERMES_HOME, ".env")
PROXY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_env_key(key_name: str) -> str | None:
    """从 Hermes .env 文件中读取指定 key 的值。"""
    if not os.path.exists(ENV_PATH):
        return None
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(key_name + "="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value and value != "***":
                    return value
    return None


def main():
    print("=" * 50)
    print("  LLM Response Cache Proxy — Hermes 集成启动器")
    print("=" * 50)

    # 1. 读取 API Key
    api_key = read_env_key("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY")
        print(f"   请确认 {ENV_PATH} 中存在该变量")
        sys.exit(1)

    print(f"✅ API Key 已读取 (长度: {len(api_key)})")

    # 2. 构建环境变量
    env = os.environ.copy()
    env["LLM_CACHE_API_KEY"] = api_key
    env["LLM_CACHE_UPSTREAM"] = "https://api.deepseek.com"

    # 3. 启动代理
    print("🚀 启动缓存代理...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_cache.proxy",
         "--upstream", "https://api.deepseek.com",
         "--compress",
         "--compression-level", "standard",
         "--l1-size", "2000",
         "--l2-size", "10000",
         "--port", "8080",
         "--log-level", "INFO"],
        cwd=PROXY_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # 4. 等待启动
    time.sleep(3)
    if proc.poll() is not None:
        stdout, _ = proc.communicate(timeout=2)
        print(f"❌ 代理启动失败: {stdout.decode()}")
        sys.exit(1)

    print(f"✅ 代理已启动 (PID: {proc.pid})")

    # 5. 健康检查
    try:
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=3)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        data = resp.read().decode()
        conn.close()
        print(f"✅ 健康检查通过: {data}")
    except Exception as e:
        print(f"⚠️  健康检查失败: {e}")
        proc.kill()
        sys.exit(1)

    # 6. 配置 Hermes
    try:
        result = subprocess.run(
            ["hermes", "config", "set", "model.base_url", "http://localhost:8080/v1"],
            capture_output=True, text=True, shell=True,
        )
        print(f"✅ Hermes 已配置: {result.stdout.strip()}")
    except Exception as e:
        print(f"⚠️  Hermes 配置失败: {e}")

    print()
    print("═" * 50)
    print("  Hermes 现在走缓存代理对话！")
    print()
    print("  代理地址: http://localhost:8080/v1")
    print("  上游:     https://api.deepseek.com")
    print("  三层缓存: ✅")
    print("  Prompt 压缩: ✅ (standard)")
    print()
    print("  管理接口:")
    print("    curl http://localhost:8080/stats  # 缓存统计")
    print("    curl http://localhost:8080/report # 运行报告")
    print("    curl -X POST http://localhost:8080/clear  # 清空缓存")
    print()
    print("  停止代理: 按 Ctrl+C")
    print("═" * 50)

    # 7. 保持运行
    try:
        for line in iter(proc.stdout.readline, b""):
            print(line.decode(), end="")
    except KeyboardInterrupt:
        print("\n\n⏹️  正在关闭代理...")
        proc.terminate()
        proc.wait(timeout=5)
        print("✅ 代理已关闭")
        print()
        print("恢复 Hermes 直连上游：")
        print("  hermes config set model.base_url https://api.deepseek.com/v1")


if __name__ == "__main__":
    main()

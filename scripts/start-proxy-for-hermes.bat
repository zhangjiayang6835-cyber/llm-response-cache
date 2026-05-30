@echo off
REM 启动 LLM 缓存代理（在 Hermes 和 DeepSeek 之间）
cd /d "C:\Users\25936\projects\llm-response-cache"

REM 从 .env 读取 API Key
for /f "tokens=*" %%a in ('type "C:\Users\25936\AppData\Local\hermes\.env" ^| findstr "^DEEPSEEK_API_KEY="') do set %%a

if "%DEEPSEEK_API_KEY%"=="" (
    echo [ERROR] 未找到 DEEPSEEK_API_KEY
    pause
    exit /b 1
)

echo [INFO] 启动缓存代理...
echo [INFO] 上游: https://api.deepseek.com
echo [INFO] 端口: 127.0.0.1:8080

set LLM_CACHE_API_KEY=%DEEPSEEK_API_KEY%
set LLM_CACHE_UPSTREAM=https://api.deepseek.com

python -m llm_cache.proxy --compress --port 8080 --log-level INFO

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 代理退出，错误码: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

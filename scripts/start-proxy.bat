@echo off
REM LLM Response Cache Proxy — Windows 启动脚本
REM 在 Hermes Agent 和 LLM Provider 之间架设缓存代理

echo ============================================
echo  LLM Response Cache Proxy
echo ============================================
echo.

REM 配置区 — 根据你的实际情况修改
set UPSTREAM=https://api.deepseek.com
set PORT=8080
set HOST=127.0.0.1
set COMPRESS=--compress
set COMPRESSION_LEVEL=--compression-level standard
set L1_SIZE=--l1-size 2000
set L2_SIZE=--l2-size 10000

REM 切换到项目目录
cd /d "%~dp0.."

REM 启动代理
echo [INFO] 启动缓存代理...
echo [INFO] 上游: %UPSTREAM%
echo [INFO] 端口: %HOST%:%PORT%
echo [INFO] 压缩: 开启 (standard)
echo.

python -m llm_cache.proxy ^
    --upstream %UPSTREAM% ^
    --host %HOST% ^
    --port %PORT% ^
    %COMPRESS% %COMPRESSION_LEVEL% ^
    %L1_SIZE% %L2_SIZE%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 代理启动失败，错误码: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

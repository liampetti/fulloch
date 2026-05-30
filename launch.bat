@echo off
setlocal EnableDelayedExpansion

echo Fulloch — Windows launcher
echo.

REM -----------------------------------------------------------------------
REM 1. Dependency checks
REM -----------------------------------------------------------------------
echo Checking dependencies...

where /q docker 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: docker not found. Install Docker Desktop first.
    echo   https://docs.docker.com/desktop/install/windows-install/
    exit /b 1
)

docker compose version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: docker compose not found. Docker Desktop should include it.
    exit /b 1
)

REM Prefer the repo venv; fall back to system Python.
set PYTHON=
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    where /q python 2>nul
    if !ERRORLEVEL! equ 0 (
        set PYTHON=python
    ) else (
        echo ERROR: Python not found.
        echo   Create a venv and run: pip install -r requirements.txt
        exit /b 1
    )
)
echo Using Python: %PYTHON%

REM huggingface-cli comes with transformers / huggingface_hub (already in reqs).
set HF_CLI=
if exist ".venv\Scripts\huggingface-cli.exe" (
    set HF_CLI=.venv\Scripts\huggingface-cli.exe
) else (
    where /q huggingface-cli 2>nul
    if !ERRORLEVEL! equ 0 (
        set HF_CLI=huggingface-cli
    ) else (
        echo huggingface-cli not found — installing via pip...
        "%PYTHON%" -m pip install "huggingface_hub[cli]" --quiet
        if !ERRORLEVEL! neq 0 (
            echo ERROR: could not install huggingface-cli.
            exit /b 1
        )
        if exist ".venv\Scripts\huggingface-cli.exe" (
            set HF_CLI=.venv\Scripts\huggingface-cli.exe
        ) else (
            set HF_CLI=huggingface-cli
        )
    )
)
echo All dependencies found.
echo.

REM -----------------------------------------------------------------------
REM 2. Directory structure
REM -----------------------------------------------------------------------
echo Checking directory structure...
set BASE_DIR=%CD%\data\models
set GRAMMAR_DIR=%BASE_DIR%\grammars
set HUB_DIR=%BASE_DIR%\hub

if not exist "%BASE_DIR%"    mkdir "%BASE_DIR%"
if not exist "%GRAMMAR_DIR%" mkdir "%GRAMMAR_DIR%"
if not exist "%HUB_DIR%"     mkdir "%HUB_DIR%"

REM -----------------------------------------------------------------------
REM 2a. Config / env templates
REM -----------------------------------------------------------------------
set CREATED_MSG=
if not exist "data\config.yml" (
    echo config.yml not found — creating from template...
    copy "data\config.example.yml" "data\config.yml" >nul
    set CREATED_MSG=!CREATED_MSG! data\config.yml
)
if not exist ".env" (
    echo .env not found — creating from template...
    copy ".env.example" ".env" >nul
    set CREATED_MSG=!CREATED_MSG! .env
)
if not "!CREATED_MSG!"=="" (
    echo.
    echo Created:!CREATED_MSG!
    echo.
    set /p CONT=Continue with defaults or exit to edit first? (c)ontinue / (e)xit:
    if /i "!CONT!"=="e" (
        echo Edit the files and run launch.bat again when ready.
        exit /b 0
    )
    echo Continuing with defaults.
    echo.
)

REM -----------------------------------------------------------------------
REM 3. Grammar files
REM -----------------------------------------------------------------------
if not exist "%GRAMMAR_DIR%\json.gbnf" (
    echo Downloading json.gbnf...
    curl -L --silent --show-error -o "%GRAMMAR_DIR%\json.gbnf" ^
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/json.gbnf"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: failed to download json.gbnf.
        exit /b 1
    )
) else (
    echo json.gbnf exists.
)

REM -----------------------------------------------------------------------
REM 4. Model downloads
REM -----------------------------------------------------------------------
if not exist "%BASE_DIR%\Qwen3.5-9B-Q5_K_M.gguf" (
    echo Downloading Qwen3.5 9B SLM ^(6.6GB^)...
    "%HF_CLI%" download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q5_K_M.gguf --local-dir "%BASE_DIR%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: SLM download failed. & exit /b 1 )
) else (
    echo Qwen3.5-9B-Q5_K_M.gguf exists.
)

if not exist "%HUB_DIR%\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" (
    echo Downloading Qwen3 TTS Base ^(3.4GB^)...
    "%HF_CLI%" download Qwen/Qwen3-TTS-12Hz-1.7B-Base --cache-dir "%HUB_DIR%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: TTS download failed. & exit /b 1 )
) else (
    echo Qwen3 TTS Base model exists.
)

if not exist "%HUB_DIR%\models--Qwen--Qwen3-ASR-1.7B" (
    echo Downloading Qwen3 ASR ^(3.4GB^)...
    "%HF_CLI%" download Qwen/Qwen3-ASR-1.7B --cache-dir "%HUB_DIR%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: ASR download failed. & exit /b 1 )
) else (
    echo Qwen3 ASR exists.
)

if not exist "%HUB_DIR%\models--BAAI--bge-small-en-v1.5" (
    echo Downloading BGE-small-en-v1.5 for semantic note search ^(~130MB^)...
    "%HF_CLI%" download BAAI/bge-small-en-v1.5 --cache-dir "%HUB_DIR%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: BGE download failed. & exit /b 1 )
) else (
    echo BGE-small-en-v1.5 exists.
)

echo.
echo All files checked.
echo.

REM -----------------------------------------------------------------------
REM 5. Start SearXNG (Docker)
REM -----------------------------------------------------------------------
set /p REBUILD=Rebuild SearXNG image before starting? (y/N):
set BUILD_FLAG=
if /i "%REBUILD%"=="y" set BUILD_FLAG=--build

echo Starting SearXNG...
docker compose -f compose.searxng.yml up -d %BUILD_FLAG%
if %ERRORLEVEL% neq 0 (
    echo ERROR: failed to start SearXNG. Is Docker Desktop running?
    exit /b 1
)
echo SearXNG started at http://localhost:8080
echo.

REM -----------------------------------------------------------------------
REM 6. Launch Fulloch (runs in foreground; Ctrl+C to stop)
REM -----------------------------------------------------------------------
echo Launching Fulloch...
echo Press Ctrl+C to stop. SearXNG will keep running in Docker.
echo.
"%PYTHON%" app.py

endlocal

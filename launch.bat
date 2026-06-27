@echo off
setlocal EnableDelayedExpansion

echo Fulloch — Windows launcher
echo.

REM -----------------------------------------------------------------------
REM First run boots a web setup wizard (in your browser) that picks a tier
REM and downloads the models, so config templating and model/grammar
REM downloads no longer happen here — the app seeds its own scaffolding
REM (config.yml + grammar) via core/bootstrap.py and the wizard fetches the
REM models. This launcher just checks Python, optionally starts SearXNG for
REM web search, and runs the app.
REM -----------------------------------------------------------------------

REM 1. Python (required) — prefer the repo venv, fall back to system Python.
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
echo.

REM 2. SearXNG (optional) — web search degrades gracefully without it, so a
REM missing Docker is only a warning, not a hard failure.
where /q docker 2>nul
if !ERRORLEVEL! equ 0 (
    docker compose version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set /p STARTSX=Start SearXNG for web search? ^(Y/n^):
        if /i not "!STARTSX!"=="n" (
            echo Starting SearXNG...
            docker compose -f compose.searxng.yml up -d
            if !ERRORLEVEL! equ 0 (
                echo SearXNG started at http://localhost:8080
            ) else (
                echo WARNING: could not start SearXNG; web search will be unavailable.
            )
        )
    ) else (
        echo Docker Compose not found — skipping SearXNG ^(web search unavailable^).
    )
) else (
    echo Docker not found — skipping SearXNG ^(web search unavailable^).
)
echo.

REM 3. Launch Fulloch. First run opens the setup wizard in your browser at the
REM    URL printed below; later runs go straight to the assistant.
echo Launching Fulloch...
echo On first run, open the setup wizard URL printed below in your browser.
echo Press Ctrl+C to stop. SearXNG (if started) keeps running in Docker.
echo.
"%PYTHON%" app.py

endlocal

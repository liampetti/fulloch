@echo off
setlocal EnableDelayedExpansion

echo Fulloch Voice Designer
echo.

REM -----------------------------------------------------------------------
REM 1. Pick Python — prefer the repo's venv to match runtime deps.
REM -----------------------------------------------------------------------
set PYTHON=
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    where /q python 2>nul
    if !ERRORLEVEL! equ 0 (
        set PYTHON=python
    ) else (
        echo ERROR: Python not found.
        echo   Activate the project venv or install Python 3.10+ and run:
        echo     pip install -r requirements.txt
        exit /b 1
    )
)
echo Using Python: %PYTHON%

REM -----------------------------------------------------------------------
REM 2. huggingface-cli (needed to download the VoiceDesign model)
REM -----------------------------------------------------------------------
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

REM -----------------------------------------------------------------------
REM 3. Ensure model cache layout exists.
REM -----------------------------------------------------------------------
set HUB_DIR=%CD%\data\models\hub
if not exist "%HUB_DIR%" mkdir "%HUB_DIR%"

REM -----------------------------------------------------------------------
REM 4. Download VoiceDesign model if not already cached.
REM -----------------------------------------------------------------------
set VOICE_DESIGN_DIR=%HUB_DIR%\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign
if not exist "%VOICE_DESIGN_DIR%" (
    echo Qwen3-TTS-12Hz-1.7B-VoiceDesign not found in %HUB_DIR%
    echo.
    set /p DLDL=Download it now? ~3.4GB (Y/n):
    if /i "!DLDL!"=="n" (
        echo Aborted — model is required.
        exit /b 0
    )
    "%HF_CLI%" download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --cache-dir "%HUB_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: VoiceDesign model download failed.
        exit /b 1
    )
) else (
    echo VoiceDesign model present.
)

REM -----------------------------------------------------------------------
REM 5. Hand off to the Python helper (interactive generate/play/save loop).
REM -----------------------------------------------------------------------
echo.
"%PYTHON%" scripts\voice_design.py

endlocal

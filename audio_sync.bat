@echo off
setlocal enabledelayedexpansion
set "CP=python"
cd /d "%~dp0"

echo.
echo   ============================================
echo     Audio Emotion Synchrony Analysis
echo   ============================================
echo.

echo   Available audio CSVs in output\csv\:
echo   -----------------------------------
set N=0
for /f "delims=" %%f in ('dir /b /od output\csv\audio_emotions_*.csv 2^>nul') do (
    set /a N+=1
    echo   [!N!] %%f
    set "FILE_!N!=%%f"
)
if !N!==0 (
    echo   No audio CSV files found in output\csv\
    pause
    goto :end
)
if !N! LSS 2 (
    echo   Need at least 2 audio CSV files.
    pause
    goto :end
)

echo.
set /p PA="  Audio A (number): "
set /p PB="  Audio B (number): "

set "FA=!FILE_%PA%!"
set "FB=!FILE_%PB%!"

if "!FA!"=="" (echo Invalid choice & pause & goto :end)
if "!FB!"=="" (echo Invalid choice & pause & goto :end)

echo.
echo   A: !FA!
echo   B: !FB!
echo   Running analysis...
echo.

"%CP%" tools\emotion_sync.py "output\csv\!FA!" "output\csv\!FB!"
echo.
echo   Report saved to output\sync\ folder.
echo.
pause
:end

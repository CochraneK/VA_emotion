@echo off
setlocal enabledelayedexpansion
set "CP=D:/AVATAR/SOFTWARE/Anaconda/envs/face-emotion/python.exe"
cd /d "D:/AVATAR/SOFTWARE/face-emotion"

echo.
echo   ============================================
echo     Emotion Synchrony Analysis
echo   ============================================
echo.

echo   Available CSVs in output\csv\:
echo   ---------------------------
set N=0
for /f "delims=" %%f in ('dir /b /od output\csv\emotions_*.csv 2^>nul') do (
    set /a N+=1
    echo   [!N!] %%f
    set "FILE_!N!=%%f"
)
if !N!==0 (
    echo   No CSV files found in output\csv\
    pause
    goto :end
)
if !N! LSS 2 (
    echo   Need at least 2 CSV files.
    pause
    goto :end
)

echo.
set /p PA="  Person A (number): "
set /p PB="  Person B (number): "

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

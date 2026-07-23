@echo off
set "CP=D:/AVATAR/SOFTWARE/Anaconda/envs/face-emotion/python.exe"
cd /d "D:/AVATAR/SOFTWARE/face-emotion"

echo.
echo   ============================================
echo     Video Emotion Analysis (HSEmotion GPU)
echo   ============================================
echo.

set /p VPATH="  Video path (drag file here): "

set "VPATH=%VPATH:"=%"

if not exist "%VPATH%" (
    echo   File not found!
    pause
    goto :end
)

echo.
echo   Skip every N frames? (default 10)
set /p SKIP="  Skip interval: "
if "%SKIP%"=="" set SKIP=10

echo.
set /p LABEL="  Session label (optional): "

echo.
echo   Analyzing...
echo.

if "%LABEL%"=="" (
    "%CP%" src/video_to_csv.py --video "%VPATH%" --skip %SKIP%
) else (
    "%CP%" src/video_to_csv.py --video "%VPATH%" --skip %SKIP% --label "%LABEL%"
)

echo.
echo   CSV saved to output/csv/ folder.
echo   Run sync.bat to compare sessions.
echo.
pause
:end
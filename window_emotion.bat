@echo off
cd /d "%~dp0"
python src\window_emotion.py --window-title Render --layout auto --label flp
pause

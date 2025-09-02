@echo off
REM Move to the folder where this .bat is located
cd /d "%~dp0"

REM Log start timestamp
echo ===== Starting Smart Pothole App at %date% %time% ===== > run_log.txt

REM Use the venv python to run streamlit (no need to activate)
venv\Scripts\python -m streamlit run app.py --server.port=8501 --server.address=127.0.0.1 >> run_log.txt 2>&1

REM If you want to auto-open the browser after a short wait, uncomment the next two lines:
REM timeout /t 3 /nobreak >nul
REM start "" "http://127.0.0.1:8501"

echo ===== Streamlit exited. Check run_log.txt for details =====
pause

@echo off
echo Starting Kelly Trading Dashboard...
echo Dashboard will be available at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo ================================================

REM Set PYTHONPATH to include src directory
set PYTHONPATH=%~dp0src

REM Run streamlit with config
echo. | python -m streamlit run src/ui/dashboard.py --server.headless=true --browser.gatherUsageStats=false --server.port=8501

pause
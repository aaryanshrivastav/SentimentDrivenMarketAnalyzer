@echo off
REM Startup script for Sentiment-Driven Market Analyzer
REM Starts backend API server

echo.
echo ================================
echo  Backend API Server
echo ================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.9+ and add to PATH.
    pause
    exit /b 1
)

echo Starting Backend API Server on port 8000...
echo.
echo Press Ctrl+C to stop the server
echo.
echo API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.

python api_server.py

pause

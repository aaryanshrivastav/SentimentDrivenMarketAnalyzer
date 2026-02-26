@echo off
REM Startup script for Frontend Dev Server

echo.
echo ================================
echo  Frontend Dev Server
echo ================================
echo.

REM Check if Node.js is available
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found. Please install Node.js 18+ and add to PATH.
    pause
    exit /b 1
)

cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    echo.
    npm install
    echo.
)

echo Starting Frontend Dev Server on port 3000...
echo.
echo Press Ctrl+C to stop the server
echo.
echo Frontend will be available at: http://localhost:3000
echo.

npm run dev

pause

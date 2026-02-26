#!/usr/bin/env pwsh
# Startup script for Sentiment-Driven Market Analyzer
# Starts both backend API and frontend dev server

Write-Host "🚀 Starting Sentiment-Driven Market Analyzer..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python not found. Please install Python 3.9+ and add to PATH." -ForegroundColor Red
    exit 1
}

# Check if Node.js is available
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ and add to PATH." -ForegroundColor Red
    exit 1
}

# Start backend API server in background
Write-Host "📊 Starting Backend API Server (port 8000)..." -ForegroundColor Cyan
$backend = Start-Process python -ArgumentList "api_server.py" -PassThru -NoNewWindow
Start-Sleep -Seconds 2

# Check if backend started successfully
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✅ Backend API running on http://localhost:8000" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend API may not be ready yet. Check terminal for errors." -ForegroundColor Yellow
}

Write-Host ""

# Start frontend dev server
Write-Host "🎨 Starting Frontend Dev Server (port 3000)..." -ForegroundColor Cyan
Set-Location frontend

# Check if node_modules exists
if (!(Test-Path "node_modules")) {
    Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
    
    # Try pnpm first, fallback to npm
    if (Get-Command pnpm -ErrorAction SilentlyContinue) {
        pnpm install
    } else {
        npm install
    }
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "✅ Sentiment-Driven Market Analyzer is ready!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "🔌 Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "📖 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host ""

# Start frontend (this will block)
try {
    if (Get-Command pnpm -ErrorAction SilentlyContinue) {
        pnpm dev
    } else {
        npm run dev
    }
} finally {
    # Cleanup: Stop backend when frontend stops
    Write-Host ""
    Write-Host "🛑 Stopping servers..." -ForegroundColor Yellow
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
    Write-Host "✅ Servers stopped." -ForegroundColor Green
}

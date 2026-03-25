"""
FastAPI server to run the sentiment analysis pipeline and stream logs to frontend.

Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Sentiment Market Analyzer API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent
PYTHON_EXE = sys.executable  # Use the same Python as the API server


async def stream_pipeline_logs(request: Request) -> AsyncIterator[str]:
    """
    Run test.py and stream logs in real-time as Server-Sent Events (SSE).
    
    Each log line is sent as:
    data: {"timestamp": "...", "level": "...", "source": "...", "message": "..."}
    """
    import json
    
    # Run test.py with all stages
    cmd = [PYTHON_EXE, str(PROJECT_ROOT / "test.py"), "--stages", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
    
    # Start log with pipeline start event
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_data = json.dumps({"type": "start", "timestamp": start_time})
    yield f"data: {start_data}\n\n"
    
    process = None

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT
        )
        
        if process.stdout:
            async for line_bytes in process.stdout:
                # Stop streaming immediately if the client closes the SSE connection.
                if await request.is_disconnected():
                    if process.returncode is None:
                        process.terminate()
                        await process.wait()
                    return

                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                # Parse log line format: "2026-02-24 12:37:54 | INFO    | PIPELINE             | message"
                log_match = re.match(
                    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(.+)$',
                    line
                )
                
                if log_match:
                    timestamp, level, source, message = log_match.groups()
                    
                    # Determine stage from message content
                    stage_id = determine_stage(message)
                    
                    log_data = {
                        "type": "log",
                        "timestamp": timestamp,
                        "level": level.strip(),
                        "source": source.strip(),
                        "message": message.strip(),
                        "stage": stage_id
                    }
                    
                    # Send as SSE (Server-Sent Events)
                    yield f"data: {json.dumps(log_data)}\n\n"
                else:
                    # Send raw line if not matching standard format
                    log_data = {
                        "type": "log",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "source": "PIPELINE",
                        "message": line,
                        "stage": None
                    }
                    yield f"data: {json.dumps(log_data)}\n\n"
                
                # Small delay to prevent overwhelming the frontend
                await asyncio.sleep(0.01)
        
        # Wait for process to complete
        await process.wait()

        # If the client disconnected while the process ended, skip final SSE frame.
        if await request.is_disconnected():
            return
        
        # Send completion event
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if process.returncode == 0:
            complete_data = json.dumps({"type": "complete", "timestamp": end_time, "success": True})
            yield f"data: {complete_data}\n\n"
        else:
            complete_data = json.dumps({"type": "complete", "timestamp": end_time, "success": False, "exit_code": process.returncode})
            yield f"data: {complete_data}\n\n"
    
    except asyncio.CancelledError:
        # Raised when the response stream is cancelled (e.g., browser tab closed).
        if process and process.returncode is None:
            process.terminate()
            await process.wait()
        return
    except Exception as e:
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_data = json.dumps({"type": "error", "timestamp": error_time, "message": str(e)})
        yield f"data: {error_data}\n\n"


def determine_stage(message: str) -> str:
    """Determine which stage a log message belongs to based on keywords."""
    message_lower = message.lower()
    
    if "stage 1a" in message_lower or "data cleaning" in message_lower or "bot detection" in message_lower:
        return "1A"
    elif "stage 1b" in message_lower or "finbert" in message_lower or "sentiment analysis" in message_lower:
        return "1B"
    elif "stage 1c" in message_lower or "aggregat" in message_lower or "sentiment feature" in message_lower:
        return "1C"
    elif "stage 2a" in message_lower or "market data" in message_lower or "technical indicator" in message_lower:
        return "2A"
    elif "stage 2b" in message_lower or "granger" in message_lower:
        return "2B"
    elif "stage 2c" in message_lower or "feature fusion" in message_lower or "fusing" in message_lower:
        return "2C"
    elif "stage 3" in message_lower or "prediction model" in message_lower or "ensemble" in message_lower:
        return "3"
    elif "pipeline" in message_lower and ("complete" in message_lower or "summary" in message_lower):
        return "complete"
    else:
        return None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Sentiment-Driven Market Analyzer API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/api/pipeline/run")
async def run_pipeline(request: Request):
    """
    Start the pipeline execution and stream logs in real-time.
    
    Returns a Server-Sent Events (SSE) stream.
    """
    return StreamingResponse(
        stream_pipeline_logs(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/api/pipeline/status")
async def pipeline_status():
    """Get the current pipeline status and recent outputs."""
    output_dir = PROJECT_ROOT / "output"
    
    if not output_dir.exists():
        return {"status": "not_run", "outputs": []}
    
    # List recent output files
    outputs = []
    for file in output_dir.glob("*.csv"):
        outputs.append({
            "name": file.name,
            "size": file.stat().st_size,
            "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })
    
    return {
        "status": "idle",
        "outputs": sorted(outputs, key=lambda x: x["modified"], reverse=True)[:10]
    }


if __name__ == "__main__":
    import uvicorn
    if sys.platform == "win32":
        # Selector policy avoids noisy Proactor shutdown tracebacks on client disconnect.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("🚀 Starting Sentiment Market Analyzer API Server...")
    print("📊 Frontend: http://localhost:3000")
    print("🔌 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

"""
Demo API routes for running Stage 1A -> Stage 3A from the frontend demo page.

Run directly:
  uvicorn api.demo_api:app --host 0.0.0.0 --port 8010 --reload
"""

from __future__ import annotations

import asyncio
import sys
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = PROJECT_ROOT / "test.py"
PYTHON_EXE = sys.executable


class DemoRunRequest(BaseModel):
    include_granger: bool = True


@dataclass
class DemoJob:
    job_id: str
    status: str = "queued"
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    stages: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


JOBS: Dict[str, DemoJob] = {}


async def _run_pipeline(job: DemoJob) -> None:
    job.status = "running"
    job.started_at = datetime.now(timezone.utc).isoformat()

    cmd = [PYTHON_EXE, str(TEST_SCRIPT), "--stages", *job.stages]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        # Read logs line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                job.logs.append(line)

        process.wait()

        job.exit_code = process.returncode
        job.ended_at = datetime.now(timezone.utc).isoformat()
        job.status = "completed" if process.returncode == 0 else "failed"

    except Exception as e:
        job.logs.append(f"ERROR: {str(e)}")
        job.status = "failed"
        job.ended_at = datetime.now(timezone.utc).isoformat()


app = FastAPI(title="Sentiment Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health() -> dict:
    return {
        "service": "demo_api",
        "status": "running",
        "project_root": str(PROJECT_ROOT),
    }


@app.post("/api/demo/run")
async def run_demo_pipeline(request: DemoRunRequest) -> dict:
    if not TEST_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Missing test.py at {TEST_SCRIPT}")

    stages = ["1A", "1B", "1C", "2A", "2C", "3A"]
    if request.include_granger:
        stages.insert(5, "2B")

    job_id = str(uuid.uuid4())
    job = DemoJob(job_id=job_id, stages=stages)
    JOBS[job_id] = job

    asyncio.create_task(_run_pipeline(job))

    return {
        "job_id": job_id,
        "status": job.status,
        "stages": stages,
        "message": "Pipeline execution started.",
    }


@app.get("/api/demo/run/{job_id}")
def get_demo_pipeline_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "exit_code": job.exit_code,
        "stages": job.stages,
        "log_line_count": len(job.logs),
        "latest_logs": job.logs[-200:],
    }


@app.delete("/api/demo/run/{job_id}")
def delete_demo_pipeline_job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    del JOBS[job_id]
    return {"deleted": job_id}

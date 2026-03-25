"use client";

import { useState } from "react";
import { startDemo, getDemoStatus } from "@/lib/api/demo";

export default function DemoPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [logs, setLogs] = useState<string[]>([]);

  const runDemo = async () => {
    setStatus("starting");
    setLogs([]);

    try {
      const res = await startDemo(false);
      const id = res.job_id;

      setJobId(id);
      setStatus("running");

      const interval = setInterval(async () => {
        const data = await getDemoStatus(id);

        setStatus(data.status);

        if (data.latest_logs) {
          setLogs(data.latest_logs);
        }

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
        }
      }, 3000);

    } catch {
      setStatus("failed");
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center px-4 text-white">
      <div className="w-full max-w-3xl space-y-6">

        <h1 className="text-3xl font-bold text-center">
          Pipeline Demo
        </h1>

        <button
          onClick={runDemo}
          className="w-full py-3 rounded-lg bg-green-500 hover:bg-green-600 transition font-semibold"
        >
          Run Pipeline
        </button>

        <div className="text-center text-sm text-zinc-400">
          Status: {status}
        </div>

        <div className=" p-4 h-64 overflow-y-auto text-xs w-full hp-3 rounded-xl bg-card border border-border focus:outline-none focus:ring-2 focus:ring-primary transition">
          {logs.length === 0 ? (
            <p className="text-zinc-500">No logs yet...</p>
          ) : (
            logs.map((log, i) => <p key={i}>{log}</p>)
          )}
        </div>

      </div>
    </main>
  );
}
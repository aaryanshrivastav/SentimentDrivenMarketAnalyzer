"use client"

import { useEffect, useRef, useState } from "react"
import type { PipelineStage, LogLine } from "@/lib/pipeline-data"
import { CheckCircle2, Loader2 } from "lucide-react"

function LogEntry({ log, isNew }: { log: LogLine; isNew: boolean }) {
  const isSuccess = log.message.includes("\u2713")

  return (
    <div
      className={`flex items-start gap-3 font-mono text-xs leading-relaxed transition-opacity duration-300 ${
        isNew ? "animate-fade-in" : ""
      }`}
    >
      <span className="shrink-0 text-muted-foreground/50 select-none">
        {log.timestamp.split(" ")[1]}
      </span>
      <span className="shrink-0 text-[10px] uppercase tracking-wider text-muted-foreground/60 min-w-[90px]">
        {log.source}
      </span>
      <span
        className={
          isSuccess
            ? "text-[#00e676]"
            : log.level === "ERROR"
            ? "text-[#ff5252]"
            : "text-foreground/80"
        }
      >
        {log.message}
      </span>
    </div>
  )
}

export function StageCard({
  stage,
  isActive,
  isComplete,
  visibleLogs,
}: {
  stage: PipelineStage
  isActive: boolean
  isComplete: boolean
  visibleLogs: LogLine[]
}) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [visibleLogs.length])

  return (
    <div
      className={`
        relative overflow-hidden rounded-lg border transition-all duration-500
        ${
          isActive
            ? "border-[#00e676]/30 shadow-[0_0_30px_rgba(0,230,118,0.08)]"
            : isComplete
            ? "border-[#00e676]/10 opacity-80"
            : "border-border/40"
        }
      `}
      style={{
        background:
          "linear-gradient(135deg, rgba(16,28,22,0.65) 0%, rgba(10,20,16,0.55) 100%)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
      }}
    >
      {/* Glass shimmer overlay */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(135deg, rgba(0,230,118,0.03) 0%, transparent 50%, rgba(0,230,118,0.01) 100%)",
        }}
      />

      {/* Header */}
      <div className="relative flex items-center justify-between border-b border-[#00e676]/8 px-5 py-3">
        <div className="flex items-center gap-3">
          {isActive && !isComplete ? (
            <Loader2 className="h-4 w-4 animate-spin text-[#00e676]" />
          ) : isComplete ? (
            <CheckCircle2 className="h-4 w-4 text-[#00e676]" />
          ) : (
            <div className="h-4 w-4 rounded-full border border-muted-foreground/30" />
          )}
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-foreground">
              {stage.title}
            </h3>
            <p className="text-xs text-muted-foreground">{stage.subtitle}</p>
          </div>
        </div>
        {isActive && !isComplete && (
          <div className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[#00e676] animate-pulse" />
            <span className="text-[10px] font-medium uppercase tracking-wider text-[#00e676]">
              Running
            </span>
          </div>
        )}
        {isComplete && (
          <span className="text-[10px] font-medium uppercase tracking-wider text-[#00e676]/70">
            Complete
          </span>
        )}
      </div>

      {/* Logs */}
      <div ref={scrollRef} className="relative max-h-[220px] overflow-y-auto px-5 py-3 space-y-1.5 scrollbar-thin">
        {visibleLogs.map((log, i) => (
          <LogEntry
            key={`${stage.id}-${i}`}
            log={log}
            isNew={i === visibleLogs.length - 1 && isActive}
          />
        ))}
      </div>

      {/* Bottom glow line when active */}
      {isActive && !isComplete && (
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#00e676]/40 to-transparent" />
      )}
    </div>
  )
}

"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { StageCard } from "@/components/stage-card"
import type { LogLine, PipelineStage } from "@/lib/pipeline-data"
import { Play, RotateCcw, AlertCircle } from "lucide-react"

type StageState = {
  visibleLogs: LogLine[]
  isActive: boolean
  isComplete: boolean
}

// Stage definitions matching test.py stages
const STAGE_DEFINITIONS: Record<string, { title: string; subtitle: string }> = {
  "1A": { title: "Stage 1A", subtitle: "Data Cleaning & Noise Removal" },
  "1B": { title: "Stage 1B", subtitle: "Sentiment Analysis Engine (FinBERT)" },
  "1C": { title: "Stage 1C", subtitle: "Sentiment Feature Aggregation" },
  "2A": { title: "Stage 2A", subtitle: "Market Data & Technical Indicators" },
  "2B": { title: "Stage 2B", subtitle: "Granger Causality Test" },
  "2C": { title: "Stage 2C", subtitle: "Feature Fusion" },
  "3": { title: "Stage 3", subtitle: "Prediction Model (XGBoost + LSTM Ensemble)" },
  "complete": { title: "Pipeline Complete", subtitle: "Execution Summary" },
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export function PipelineRunner() {
  const [isRunning, setIsRunning] = useState(false)
  const [isDone, setIsDone] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stages, setStages] = useState<Map<string, StageState>>(new Map())
  const [visibleStageIds, setVisibleStageIds] = useState<string[]>([])
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      })
    }
  }, [])

  const cleanup = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }, [])

  const reset = useCallback(() => {
    cleanup()
    setIsRunning(false)
    setIsDone(false)
    setError(null)
    setStages(new Map())
    setVisibleStageIds([])
    setCurrentStage(null)
  }, [cleanup])

  const ensureStageVisible = useCallback((stageId: string) => {
    setVisibleStageIds((prev) => {
      if (!prev.includes(stageId)) {
        return [...prev, stageId]
      }
      return prev
    })
    
    setStages((prev) => {
      const updated = new Map(prev)
      if (!updated.has(stageId)) {
        updated.set(stageId, {
          visibleLogs: [],
          isActive: true,
          isComplete: false,
        })
      }
      return updated
    })
  }, [])

  const runPipeline = useCallback(() => {
    reset()
    setIsRunning(true)
    setError(null)

    // Connect to SSE endpoint
    const eventSource = new EventSource(`${API_URL}/api/pipeline/run`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === "start") {
          console.log("Pipeline started at:", data.timestamp)
        } else if (data.type === "log") {
          const stageId = data.stage || "general"
          
          // Ensure stage card is visible
          if (stageId && stageId !== "general") {
            ensureStageVisible(stageId)
            setCurrentStage(stageId)
            
            // Mark previous stage as complete
            setStages((prev) => {
              const updated = new Map(prev)
              prev.forEach((state, id) => {
                if (id !== stageId && state.isActive) {
                  updated.set(id, {
                    ...state,
                    isActive: false,
                    isComplete: true,
                  })
                }
              })
              return updated
            })
          }

          // Add log to stage
          const logLine: LogLine = {
            timestamp: data.timestamp,
            level: data.level as "INFO" | "ERROR",
            source: data.source,
            message: data.message,
          }

          setStages((prev) => {
            const updated = new Map(prev)
            const current = updated.get(stageId)
            if (current) {
              updated.set(stageId, {
                ...current,
                visibleLogs: [...current.visibleLogs, logLine],
              })
            }
            return updated
          })

          setTimeout(scrollToBottom, 50)
        } else if (data.type === "complete") {
          // Mark last stage as complete
          setStages((prev) => {
            const updated = new Map(prev)
            prev.forEach((state, id) => {
              if (state.isActive) {
                updated.set(id, {
                  ...state,
                  isActive: false,
                  isComplete: true,
                })
              }
            })
            return updated
          })

          setIsRunning(false)
          setIsDone(true)
          cleanup()
        } else if (data.type === "error") {
          setError(data.message)
          setIsRunning(false)
          cleanup()
        }
      } catch (err) {
        console.error("Error parsing SSE message:", err)
      }
    }

    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err)
      setError("Connection to pipeline server lost. Make sure the API server is running on port 8000.")
      setIsRunning(false)
      cleanup()
    }
  }, [reset, cleanup, scrollToBottom, ensureStageVisible])

  useEffect(() => {
    return cleanup
  }, [cleanup])

  return (
    <div className="flex flex-col items-center gap-8 w-full max-w-3xl mx-auto">
      {/* Action Button */}
      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-4">
          {!isRunning && !isDone && (
            <button
              onClick={runPipeline}
              className="group relative flex items-center gap-2.5 rounded-lg border border-[#00e676]/30 bg-[#00e676]/10 px-8 py-3.5 text-sm font-semibold text-[#00e676] transition-all duration-300 hover:bg-[#00e676]/20 hover:border-[#00e676]/50 hover:shadow-[0_0_30px_rgba(0,230,118,0.15)] active:scale-[0.98]"
            >
              <Play className="h-4 w-4 fill-current" />
              Run Pipeline
            </button>
          )}
          {isDone && (
            <button
              onClick={reset}
              className="group relative flex items-center gap-2.5 rounded-lg border border-[#00e676]/30 bg-[#00e676]/10 px-8 py-3.5 text-sm font-semibold text-[#00e676] transition-all duration-300 hover:bg-[#00e676]/20 hover:border-[#00e676]/50 hover:shadow-[0_0_30px_rgba(0,230,118,0.15)] active:scale-[0.98]"
            >
              <RotateCcw className="h-4 w-4" />
              Run Again
            </button>
          )}
          {isRunning && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-[#00e676] animate-pulse" />
              Pipeline executing...
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="flex items-start gap-3 rounded-lg border border-[#ff5252]/30 bg-[#ff5252]/10 px-4 py-3 max-w-xl">
            <AlertCircle className="h-5 w-5 text-[#ff5252] shrink-0 mt-0.5" />
            <div className="flex flex-col gap-1">
              <p className="text-sm font-medium text-[#ff5252]">Pipeline Error</p>
              <p className="text-xs text-[#ff5252]/80">{error}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Make sure the API server is running: <code className="text-[#00e676]">python api_server.py</code>
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Stage Cards */}
      <div
        ref={containerRef}
        className="flex flex-col gap-4 w-full max-h-[calc(100vh-280px)] overflow-y-auto pr-1 scrollbar-thin"
      >
        {visibleStageIds.map((stageId) => {
          const stageDef = STAGE_DEFINITIONS[stageId] || {
            title: `Stage ${stageId}`,
            subtitle: "Processing...",
          }
          const state = stages.get(stageId)
          if (!state) return null

          const stage: PipelineStage = {
            id: stageId,
            title: stageDef.title,
            subtitle: stageDef.subtitle,
            logs: state.visibleLogs,
            duration: 200,
          }

          return (
            <div key={stageId} className="animate-slide-up">
              <StageCard
                stage={stage}
                isActive={state.isActive}
                isComplete={state.isComplete}
                visibleLogs={state.visibleLogs}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

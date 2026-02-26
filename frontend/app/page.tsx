import { PipelineRunner } from "@/components/pipeline-runner"
import { GridBackground } from "@/components/grid-background"
import { TrendingUp, Activity, BarChart3 } from "lucide-react"

export default function Page() {
  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      <GridBackground />

      {/* Subtle radial gradient */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, rgba(0,230,118,0.06) 0%, transparent 60%)",
        }}
        aria-hidden="true"
      />

      <main className="relative z-10 flex flex-col items-center px-4 py-12 md:py-16">
        {/* Header */}
        <header className="flex flex-col items-center gap-5 mb-10">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 rounded-md border border-[#00e676]/20 bg-[#00e676]/5 px-3 py-1.5">
              <Activity className="h-3.5 w-3.5 text-[#00e676]" />
              <TrendingUp className="h-3.5 w-3.5 text-[#00e676]" />
              <BarChart3 className="h-3.5 w-3.5 text-[#00e676]" />
            </div>
          </div>

          <h1 className="text-balance text-center text-3xl font-bold tracking-tight text-foreground md:text-4xl lg:text-5xl">
            Sentiment Driven
            <br />
            <span className="text-[#00e676]">Market Analyser</span>
          </h1>

          <p className="max-w-lg text-center text-sm leading-relaxed text-muted-foreground md:text-base">
            AI-powered pipeline combining FinBERT sentiment analysis, NLP noise filtering,
            and XGBoost + LSTM ensemble models for stock market prediction.
          </p>

          {/* Tech tags */}
          <div className="flex flex-wrap items-center justify-center gap-2">
            {["FinBERT", "RoBERTa", "SpaCy NER", "XGBoost", "LSTM"].map(
              (tag) => (
                <span
                  key={tag}
                  className="rounded-md border border-[#00e676]/10 bg-[#00e676]/5 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wider text-[#00e676]/70"
                >
                  {tag}
                </span>
              )
            )}
          </div>
        </header>

        {/* Pipeline Runner */}
        <PipelineRunner />
      </main>
    </div>
  )
}

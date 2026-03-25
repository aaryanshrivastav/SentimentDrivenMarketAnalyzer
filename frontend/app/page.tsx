"use client";

import { GridBackground } from "@/components/grid-background";
import { TrendingUp, Activity, BarChart3 } from "lucide-react";

export default function Page() {
  return (
    <div className="relative min-h-screen bg-background overflow-hidden">

      {/* 🌌 Background */}
      <GridBackground />

      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, rgba(0,230,118,0.08) 0%, transparent 60%)",
        }}
      />

      {/* 🌐 Container */}
      <main className="relative z-10 flex min-h-screen items-center justify-center px-4">

        {/* 📦 Content Wrapper */}
        <div className="w-full max-w-3xl flex flex-col items-center text-center">

          {/* HEADER */}
          <header className="flex flex-col items-center gap-6">

            {/* ICON STRIP */}
            <div className="flex items-center gap-2 rounded-md border border-[#00e676]/20 bg-[#00e676]/5 px-3 py-1.5">
              <Activity className="h-3 w-3 md:h-3.5 md:w-3.5 text-[#00e676]" />
              <TrendingUp className="h-3 w-3 md:h-3.5 md:w-3.5 text-[#00e676]" />
              <BarChart3 className="h-3 w-3 md:h-3.5 md:w-3.5 text-[#00e676]" />
            </div>

            {/* TITLE */}
            <h1 className="text-center font-bold tracking-tight leading-tight
              text-3xl sm:text-4xl md:text-5xl lg:text-6xl text-foreground">
              Sentiment Driven
              <br />
              <span className="text-[#00e676]">
                Market Analyser
              </span>
            </h1>

            {/* DESCRIPTION */}
            <p className="max-w-xl text-center text-sm sm:text-base md:text-lg leading-relaxed text-muted-foreground">
              AI-powered pipeline combining FinBERT sentiment analysis, NLP noise filtering,
              and XGBoost + LSTM ensemble models for stock market prediction.
            </p>

            {/* TECH TAGS */}
            <div className="flex flex-wrap justify-center gap-2 max-w-md">
              {["FinBERT", "RoBERTa", "SpaCy NER", "XGBoost", "LSTM"].map((tag) => (
                <span
                  key={tag}
                  className="rounded-md border border-[#00e676]/10 bg-[#00e676]/5 
                  px-2.5 py-1 text-[10px] sm:text-xs font-medium uppercase 
                  tracking-wider text-[#00e676]/70"
                >
                  {tag}
                </span>
              ))}
            </div>

            {/* CTA */}
            <div className="pt-4">
              <a
                href="/demo"
                className="inline-flex items-center justify-center
                rounded-lg border border-[#00e676]/20 bg-[#00e676]/10
                px-6 py-3 text-sm sm:text-base font-medium text-[#00e676]
                transition-all duration-300
                hover:bg-[#00e676]/20 hover:shadow-[0_0_30px_rgba(0,230,118,0.35)]
                hover:scale-105"
              >
                Explore Demo →
              </a>
            </div>

          </header>
        </div>

      </main>
    </div>
  );
}
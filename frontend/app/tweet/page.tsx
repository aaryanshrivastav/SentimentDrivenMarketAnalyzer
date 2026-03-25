"use client";

import { useState } from "react";
import { scoreTweet } from "@/lib/api/tweet";

export default function TweetPage() {
  const [tweet, setTweet] = useState("");
  const [ticker, setTicker] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await scoreTweet({
        tweet,
        ticker,
        user_credibility: 1.0,
      });

      console.log(res);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <main className="flex min-h-screen items-center justify-center px-4 text-white">
      <div className="w-full max-w-2xl space-y-6">
        
        {/* Title */}
        <h1 className="text-3xl font-bold text-center">
          Tweet Sentiment Analyzer
        </h1>

        {/* Input Box */}
        <textarea
          value={tweet}
          onChange={(e) => setTweet(e.target.value)}
          placeholder="Enter tweet..."
          className="w-full h-32 p-4 rounded-xl bg-card border border-border focus:outline-none focus:ring-2 focus:ring-primary transition"
        />

        {/* Ticker Input */}
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Ticker (e.g. NVDA)"
          className="w-full hp-3 rounded-xl bg-card border border-border focus:outline-none focus:ring-2 focus:ring-primary transition"
        />

        {/* Button */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-3 rounded-lg bg-green-500 hover:bg-green-600 transition font-semibold"
        >
          {loading ? "Analyzing..." : "Analyze Tweet"}
        </button>

        {/* Error */}
        {error && (
          <div className="text-red-400 text-sm text-center">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
  <div className="bg-card backdrop-blur-md border border-border rounded-xl p-6 space-y-5 animate-slide-up">

    {/* 🎯 Direction Badge */}
    <div className="flex justify-center">
      <span
        className={`px-4 py-1.5 rounded-full text-sm font-semibold tracking-wide
        ${
          result.predicted_direction === "UP"
            ? "bg-green-500/10 text-green-400 border border-green-500/30"
            : "bg-red-500/10 text-red-400 border border-red-500/30"
        }`}
      >
        {result.predicted_direction === "UP" ? "📈 BULLISH" : "📉 BEARISH"}
      </span>
    </div>

    {/* 📊 Score */}
    <div className="text-center">
      <p className="text-xs text-muted-foreground">Model Score</p>
      <p className="text-3xl font-bold text-primary">
        {(result.main_score * 100).toFixed(1)}%
      </p>
    </div>

    {/* 📊 Confidence Bar */}
    <div>
      <p className="text-xs text-muted-foreground mb-1">Confidence</p>
      <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-500"
          style={{
            width: `${(result.scores.finbert_confidence * 100).toFixed(0)}%`,
          }}
        />
      </div>
      <p className="text-xs text-right text-muted-foreground mt-1">
        {(result.scores.finbert_confidence * 100).toFixed(1)}%
      </p>
    </div>

    {/* 🧠 Sentiment */}
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">Sentiment</span>
      <span className="font-medium capitalize text-primary">
        {result.scores.finbert_label}
      </span>
    </div>

  </div>
)}

      </div>
    </main>
  );
}
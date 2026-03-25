"use client";

import { useState } from "react";
import { scoreCompany } from "@/lib/api/company";

export default function CompanyPage() {
  const [company, setCompany] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await scoreCompany({
        company_name: company,
        posts_per_query: 20,
        fetch_comments: true,
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

        <h1 className="text-3xl font-bold text-center">
          Company Analyzer
        </h1>

        <input
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          placeholder="Enter company name (e.g. Reliance)"
          className="w-full p-3 hp-3 rounded-xl bg-card border border-border focus:outline-none focus:ring-2 focus:ring-primary transition"
        />

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-3 rounded-lg bg-green-500 hover:bg-green-600 transition font-semibold"
        >
          {loading ? "Analyzing..." : "Analyze Company"}
        </button>

        {error && (
          <div className="text-red-400 text-center">
            {error}
          </div>
        )}

        {result && (
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-4 space-y-2">
            <p><strong>Score:</strong> {result.main_score}</p>
            <p><strong>Direction:</strong> {result.predicted_direction}</p>
            <p><strong>Mentions:</strong> {result.mention_volume}</p>
            <p><strong>Confidence:</strong> {result.avg_confidence}</p>
          </div>
        )}

      </div>
    </main>
  );
}
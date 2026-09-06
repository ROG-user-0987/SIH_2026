"use client";

import { AnalysisResult } from "@/lib/websocket-client";

interface ExplanationPanelProps {
  result: AnalysisResult;
}

const CATEGORY_LABELS: Record<string, { label: string; color: string }> = {
  spectral: { label: "Spectral Pattern", color: "#8b5cf6" },
  prosody: { label: "Voice Rhythm & Pitch", color: "#06b6d4" },
  speaker_match: { label: "Speaker Identity", color: "#f59e0b" },
  transcript_risk: { label: "Message Content", color: "#ef4444" },
  meta: { label: "Overall Assessment", color: "#6b7280" },
  none: { label: "No Signal", color: "#4b5563" },
  unknown: { label: "Unknown", color: "#4b5563" },
};

const TAG_TO_PHRASE: Record<string, string> = {
  unnatural_spectral_pattern: "Unusual spectral pattern typical of synthetic speech",
  low_pitch_variance: "Unnaturally flat pitch variance",
  high_spectral_artifact: "High-frequency spectral artifacts detected",
  jitter_anomaly: "Unnatural pitch perturbation detected",
  shimmer_anomaly: "Unnatural amplitude perturbation detected",
  natural_spectral_pattern: "Spectral pattern consistent with natural speech",
  healthy_pitch_variation: "Pitch variation within normal range",
  inconsistent_detector_signals: "Detectors disagree on authenticity",
  speech_rate_anomaly: "Unusual speech rate pattern",
  low_confidence_signal: "Weak signal strength — result may be less reliable",
};

export default function ExplanationPanel({ result }: ExplanationPanelProps) {
  const { explanation, signals, risk_score } = result;
  const { dominant_signal, dominant_category, rationale_tags } = explanation || {};

  const categoryInfo = CATEGORY_LABELS[dominant_category] || CATEGORY_LABELS.unknown;

  const rankedSignals = [...signals].sort(
    (a, b) => Math.abs(b.score * b.weight) - Math.abs(a.score * a.weight)
  );

  return (
    <div className="glass-card rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
        Why This Result
      </h3>

      <div className="flex items-center gap-3 mb-4">
        <div
          className="px-3 py-1.5 rounded-lg text-xs font-bold uppercase"
          style={{ backgroundColor: categoryInfo.color + "20", color: categoryInfo.color }}
        >
          {categoryInfo.label}
        </div>
        <span className="text-xs text-slate-400">
          Dominant signal: {dominant_signal?.replace(/_/g, " ") || "none"}
        </span>
      </div>

      {rationale_tags && rationale_tags.length > 0 && (
        <div className="mb-4">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
            Rationale
          </p>
          <div className="flex flex-wrap gap-2">
            {rationale_tags.map((tag) => (
              <span
                key={tag}
                className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700/50"
              >
                {TAG_TO_PHRASE[tag] || tag.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>
      )}

      {rankedSignals.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
            Signal Ranking (by contribution)
          </p>
          <div className="space-y-2">
            {rankedSignals.map((signal) => {
              const contribution = Math.abs(signal.score * signal.weight);
              const catInfo = CATEGORY_LABELS[signal.category] || CATEGORY_LABELS.unknown;
              return (
                <div key={signal.name} className="flex items-center gap-3">
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: catInfo.color }}
                  />
                  <span className="text-xs text-slate-300 w-24 flex-shrink-0">
                    {signal.name.replace(/_/g, " ")}
                  </span>
                  <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(contribution * 100, 100)}%`,
                        backgroundColor: catInfo.color,
                      }}
                    />
                  </div>
                  <span className="text-[10px] text-slate-500 w-14 text-right">
                    {(contribution * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="mt-4 pt-3 border-t border-slate-700/50 flex items-center justify-between">
        <p className="text-[10px] text-slate-600">
          {explanation.explainability_version || "N/A"}
        </p>
        <p className="text-[10px] text-slate-600 italic">
          Advisory signal — not a definitive authenticity verdict
        </p>
      </div>
    </div>
  );
}

"use client";

import { AnalysisResult } from "@/lib/websocket-client";

interface ResultPanelProps {
  result: AnalysisResult;
}

export default function ResultPanel({ result }: ResultPanelProps) {
  const { signals, confidence, model_versions, window_ts_start } = result;

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Detector Signals
        </h3>
        <span className="text-xs text-slate-500">
          Confidence: {(confidence * 100).toFixed(1)}%
        </span>
      </div>

      {signals.length === 0 ? (
        <p className="text-sm text-slate-500 italic">Waiting for speech data...</p>
      ) : (
        <div className="space-y-3">
          {signals.map((signal) => (
            <div key={signal.name} className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">
                  {signal.name.replace(/_/g, " ").toUpperCase()}
                </span>
                <span className="text-slate-400">
                  {(signal.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full signal-bar"
                  style={{
                    width: `${signal.score * 100}%`,
                    backgroundColor:
                      signal.category === "spectral" ? "#8b5cf6" : "#06b6d4",
                  }}
                />
              </div>
              {signal.top_feature && (
                <span className="text-[10px] text-slate-500">
                  Top feature: {signal.top_feature.replace(/_/g, " ")}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 pt-3 border-t border-slate-700/50">
        <p className="text-[10px] text-slate-600">
          Fusion: {model_versions.fusion || "N/A"} | Window:{" "}
          {new Date(window_ts_start).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}

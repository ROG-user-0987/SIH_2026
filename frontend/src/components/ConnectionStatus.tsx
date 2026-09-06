"use client";

interface ConnectionStatusProps {
  state: "disconnected" | "connecting" | "connected" | "processing" | "error";
}

const STATE_CONFIG: Record<string, { color: string; label: string; dotClass: string }> = {
  disconnected: { color: "text-slate-500", label: "Disconnected", dotClass: "bg-slate-500" },
  connecting: { color: "text-yellow-400", label: "Connecting...", dotClass: "bg-yellow-400 animate-pulse" },
  connected: { color: "text-green-400", label: "Connected", dotClass: "bg-green-400" },
  processing: { color: "text-blue-400", label: "Processing", dotClass: "bg-blue-400 animate-pulse" },
  error: { color: "text-red-400", label: "Error", dotClass: "bg-red-400" },
};

export default function ConnectionStatus({ state }: ConnectionStatusProps) {
  const config = STATE_CONFIG[state] || STATE_CONFIG.disconnected;

  return (
    <div className={`flex items-center gap-2 ${config.color}`}>
      <div className={`w-2.5 h-2.5 rounded-full ${config.dotClass}`} />
      <span className="text-xs font-medium uppercase tracking-wider">{config.label}</span>
    </div>
  );
}

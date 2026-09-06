"use client";

interface RiskGaugeProps {
  riskScore: number;
  verdict: string | null;
}

export default function RiskGauge({ riskScore, verdict }: RiskGaugeProps) {
  const radius = 70;
  const stroke = 10;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const progress = Math.min(riskScore / 100, 1);
  const strokeDashoffset = circumference - progress * circumference;

  const getColor = (score: number) => {
    if (score < 30) return "#22c55e";
    if (score < 60) return "#eab308";
    if (score < 75) return "#f97316";
    return "#ef4444";
  };

  const color = getColor(riskScore);

  const getVerdictClass = (v: string | null) => {
    if (v === "FAKE") return "verdict-fake";
    if (v === "REAL") return "verdict-real";
    if (v === "INSUFFICIENT_SPEECH") return "verdict-insufficient";
    return "";
  };

  return (
    <div className="glass-card rounded-2xl p-6 flex flex-col items-center">
      <div className="relative" style={{ width: radius * 2, height: radius * 2 }}>
        <svg height={radius * 2} width={radius * 2} className="-rotate-90">
          <circle
            stroke="rgba(45, 55, 72, 0.4)"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          <circle
            stroke={color}
            fill="transparent"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${circumference} ${circumference}`}
            style={{
              strokeDashoffset,
              transition: "stroke-dashoffset 0.6s cubic-bezier(0.4,0,0.2,1), stroke 0.3s ease",
            }}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-4xl font-bold tabular-nums"
            style={{ color }}
          >
            {riskScore}
          </span>
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mt-0.5">
            Risk
          </span>
        </div>
      </div>

      {verdict && (
        <div className="mt-4 text-center">
          <span
            className={`text-2xl font-bold uppercase tracking-wider ${getVerdictClass(
              verdict
            )}`}
          >
            {verdict === "INSUFFICIENT_SPEECH" ? "Listening..." : verdict}
          </span>
        </div>
      )}
    </div>
  );
}

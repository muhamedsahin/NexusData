"use client";

import { cn } from "@/lib/cn";

type HeroFallbackProps = {
  className?: string;
};

/** Static CSS/SVG fallback when WebGL is unavailable or still probing. */
export function HeroFallback({ className }: HeroFallbackProps) {
  return (
    <div
      className={cn("absolute inset-0 overflow-hidden", className)}
      aria-hidden
    >
      {/* Rich multi-glow background */}
      <div
        className="absolute inset-0"
        style={{
          background: [
            "radial-gradient(ellipse 720px 420px at 68% 28%, rgba(47,191,176,0.18), transparent 68%)",
            "#090a0c",
          ].join(", "),
        }}
      />

      {/* Animated lattice-like SVG grid */}
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 900 700"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="gridStroke" x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#2fbfb0" stopOpacity="0.85" />
            <stop offset="1" stopColor="#f3eee6" stopOpacity="0.35" />
          </linearGradient>
          <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#2fbfb0" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#2fbfb0" stopOpacity="0" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Concentric neon rectangles */}
        {Array.from({ length: 10 }).map((_, i) => (
          <rect
            key={i}
            x={270 + i * 10}
            y={150 + i * 10}
            width={400 - i * 20}
            height={320 - i * 20}
            rx="14"
            stroke="url(#gridStroke)"
            strokeOpacity={0.5 - i * 0.04}
            strokeWidth={i === 0 ? 1.8 : 1.0}
            filter={i === 0 ? "url(#glow)" : undefined}
          />
        ))}

        {/* Glowing node dots */}
        {Array.from({ length: 48 }).map((_, i) => {
          const cols = 8;
          const col = i % cols;
          const row = Math.floor(i / cols);
          const x = 305 + col * 46;
          const y = 200 + row * 36;
          const opacity = 0.35 + (i % 7) * 0.08;
          const r = 2.4 + (i % 3) * 0.6;
          return (
            <g key={`n-${i}`}>
              <circle cx={x} cy={y} r={r + 4} fill="url(#nodeGlow)" opacity={opacity * 0.4} />
              <circle
                cx={x}
                cy={y}
                r={r}
                fill="#2fbfb0"
                opacity={opacity}
                filter="url(#glow)"
              />
            </g>
          );
        })}

        {/* Connection lines */}
        {Array.from({ length: 18 }).map((_, i) => {
          const x1 = 305 + (i % 6) * 46;
          const y1 = 200 + Math.floor(i / 6) * 36;
          const x2 = x1 + 46;
          const y2 = y1 + 36;
          return (
            <line
              key={`l-${i}`}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="#2fbfb0"
              strokeOpacity={0.15 + (i % 4) * 0.05}
              strokeWidth={0.8}
            />
          );
        })}
      </svg>
    </div>
  );
}

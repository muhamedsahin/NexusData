import { create } from "zustand";

export type QualityLevel = "low" | "medium" | "high";

type QualityState = {
  level: QualityLevel;
  dprCap: number;
  particleScale: number;
  enablePost: boolean;
  setLevel: (level: QualityLevel) => void;
  degrade: () => void;
};

const PRESETS: Record<
  QualityLevel,
  { dprCap: number; particleScale: number; enablePost: boolean }
> = {
  low: { dprCap: 1, particleScale: 0.35, enablePost: false },
  medium: { dprCap: 1.5, particleScale: 0.65, enablePost: true },
  high: { dprCap: 2, particleScale: 1, enablePost: true },
};

function nextLower(level: QualityLevel): QualityLevel {
  if (level === "high") return "medium";
  if (level === "medium") return "low";
  return "low";
}

export function initialQualityForDevice(): QualityLevel {
  if (typeof window === "undefined") return "medium";
  const cores = navigator.hardwareConcurrency ?? 4;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const isMobile = window.matchMedia("(max-width: 768px)").matches;
  if (isMobile || cores <= 4 || (mem !== undefined && mem <= 4)) return "low";
  if (cores >= 8 && (mem === undefined || mem >= 8)) return "high";
  return "medium";
}

export const useQualityStore = create<QualityState>((set, get) => {
  const level = "medium";
  const preset = PRESETS[level];
  return {
    level,
    ...preset,
    setLevel: (next) => set({ level: next, ...PRESETS[next] }),
    degrade: () => {
      const next = nextLower(get().level);
      set({ level: next, ...PRESETS[next] });
    },
  };
});

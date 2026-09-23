export type SceneId =
  | "hero"
  | "sources"
  | "shuffle"
  | "batch"
  | "gpu"
  | "ecosystem"
  | "features"
  | "performance"
  | "code"
  | "cta";

export type SceneRange = {
  id: SceneId;
  start: number;
  end: number;
};

/** Normalized scroll progress → continuous story beats (no hard cuts). */
export const SCENE_RANGES: SceneRange[] = [
  { id: "hero", start: 0, end: 0.08 },
  { id: "sources", start: 0.08, end: 0.18 },
  { id: "shuffle", start: 0.18, end: 0.3 },
  { id: "batch", start: 0.3, end: 0.42 },
  { id: "gpu", start: 0.42, end: 0.54 },
  { id: "ecosystem", start: 0.54, end: 0.66 },
  { id: "features", start: 0.66, end: 0.76 },
  { id: "performance", start: 0.76, end: 0.86 },
  { id: "code", start: 0.86, end: 0.94 },
  { id: "cta", start: 0.94, end: 1 },
];

export function sceneIndex(id: SceneId): number {
  return SCENE_RANGES.findIndex((s) => s.id === id);
}

export function sceneAt(progress: number): SceneRange {
  const p = Math.min(1, Math.max(0, progress));
  for (const range of SCENE_RANGES) {
    if (p >= range.start && p < range.end) return range;
  }
  return SCENE_RANGES[SCENE_RANGES.length - 1]!;
}

/** Soft weight in [0,1] for how deep we are inside a scene. */
export function sceneLocal(progress: number, id: SceneId): number {
  const range = SCENE_RANGES.find((s) => s.id === id);
  if (!range) return 0;
  const span = range.end - range.start;
  if (span <= 0) return 0;
  return Math.min(1, Math.max(0, (progress - range.start) / span));
}

/** Smoothstep blend between adjacent scenes for shader morphs. */
export function continuousScene(progress: number): number {
  const p = Math.min(1, Math.max(0, progress));
  for (let i = 0; i < SCENE_RANGES.length; i++) {
    const r = SCENE_RANGES[i]!;
    if (p <= r.end || i === SCENE_RANGES.length - 1) {
      const local = (p - r.start) / Math.max(1e-6, r.end - r.start);
      return i + Math.min(1, Math.max(0, local));
    }
  }
  return SCENE_RANGES.length - 1;
}

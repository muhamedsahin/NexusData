import { create } from "zustand";

export type LoaderPhase = "booting" | "loading" | "exploding" | "ready" | "skipped";

export type LoaderLogId =
  | "mmap"
  | "shaders"
  | "rng"
  | "warmup"
  | "done";

type LoaderState = {
  phase: LoaderPhase;
  progress: number;
  minElapsed: boolean;
  assetsReady: boolean;
  sceneReady: boolean;
  fontsReady: boolean;
  reducedMotion: boolean;
  webgl: boolean | null;
  logs: LoaderLogId[];
  morph: number;
  setProgress: (value: number) => void;
  markAssetProgress: (ratio: number) => void;
  markFontsReady: () => void;
  markSceneReady: () => void;
  markMinElapsed: () => void;
  pushLog: (id: LoaderLogId) => void;
  setWebgl: (ok: boolean) => void;
  setReducedMotion: (value: boolean) => void;
  setMorph: (value: number) => void;
  complete: () => void;
  skip: () => void;
  beginExplode: () => void;
};

const SESSION_KEY = "nexusdata-loader-seen";

export function hasSeenLoader(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return sessionStorage.getItem(SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

export function markLoaderSeen(): void {
  try {
    sessionStorage.setItem(SESSION_KEY, "1");
  } catch {
    /* ignore */
  }
}

export const useLoaderStore = create<LoaderState>((set, get) => ({
  phase: "booting",
  progress: 0,
  minElapsed: false,
  assetsReady: false,
  sceneReady: false,
  fontsReady: false,
  reducedMotion: false,
  webgl: null,
  logs: [],
  morph: 0,
  setProgress: (value) => set({ progress: Math.min(100, Math.max(0, value)) }),
  markAssetProgress: (ratio) => {
    const clamped = Math.min(1, Math.max(0, ratio));
    const { fontsReady, sceneReady, logs } = get();
    const fontPart = fontsReady ? 1 : 0;
    const scenePart = sceneReady ? 1 : 0;
    const combined = clamped * 0.7 + fontPart * 0.15 + scenePart * 0.15;
    const progress = Math.round(combined * 100);
    const nextLogs = [...logs];
    if (clamped > 0.05 && !nextLogs.includes("mmap")) nextLogs.push("mmap");
    if (clamped > 0.45 && !nextLogs.includes("shaders")) nextLogs.push("shaders");
    set({
      progress,
      assetsReady: clamped >= 0.99,
      logs: nextLogs,
      phase: get().phase === "booting" ? "loading" : get().phase,
    });
  },
  markFontsReady: () => {
    const next = [...get().logs];
    if (!next.includes("rng")) next.push("rng");
    set({ fontsReady: true, logs: next });
    get().markAssetProgress(get().assetsReady ? 1 : get().progress / 100 / 0.7);
  },
  markSceneReady: () => {
    const next = [...get().logs];
    if (!next.includes("warmup")) next.push("warmup");
    set({ sceneReady: true, logs: next });
    const { fontsReady, assetsReady } = get();
    const assetRatio = assetsReady ? 1 : Math.min(0.98, get().progress / 70);
    get().markAssetProgress(assetRatio);
    void fontsReady;
  },
  markMinElapsed: () => set({ minElapsed: true }),
  pushLog: (id) => {
    const logs = get().logs;
    if (logs.includes(id)) return;
    set({ logs: [...logs, id] });
  },
  setWebgl: (ok) => set({ webgl: ok }),
  setReducedMotion: (value) => set({ reducedMotion: value }),
  setMorph: (value) => set({ morph: Math.min(1, Math.max(0, value)) }),
  beginExplode: () => set({ phase: "exploding", progress: 100 }),
  complete: () => {
    markLoaderSeen();
    const next = [...get().logs];
    if (!next.includes("done")) next.push("done");
    set({ phase: "ready", progress: 100, morph: 1, logs: next });
  },
  skip: () => {
    markLoaderSeen();
    set({
      phase: "skipped",
      progress: 100,
      morph: 1,
      minElapsed: true,
      assetsReady: true,
      sceneReady: true,
      fontsReady: true,
    });
  },
}));

export function computeReadyToFinish(state: LoaderState): boolean {
  if (state.phase === "ready" || state.phase === "skipped" || state.phase === "exploding") {
    return false;
  }
  if (state.reducedMotion || state.webgl === false) {
    return state.minElapsed && state.fontsReady;
  }
  return (
    state.minElapsed &&
    state.fontsReady &&
    state.sceneReady &&
    (state.assetsReady || state.progress >= 98)
  );
}

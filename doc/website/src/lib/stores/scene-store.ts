import { create } from "zustand";

type PointerState = {
  /** NDC -1..1 */
  x: number;
  y: number;
  /** Smoothed */
  sx: number;
  sy: number;
  /** World-ish target used by shaders */
  worldX: number;
  worldY: number;
  setTarget: (x: number, y: number) => void;
  tick: (dt: number) => void;
};

export const usePointerStore = create<PointerState>((set, get) => ({
  x: 0,
  y: 0,
  sx: 0,
  sy: 0,
  worldX: 0,
  worldY: 0,
  setTarget: (x, y) => set({ x, y }),
  tick: (dt) => {
    const { x, y, sx, sy } = get();
    const lerp = 1 - Math.exp(-dt * 6);
    const nsx = sx + (x - sx) * lerp;
    const nsy = sy + (y - sy) * lerp;
    set({
      sx: nsx,
      sy: nsy,
      worldX: nsx * 4.5,
      worldY: nsy * 2.8,
    });
  },
}));

type ScrollState = {
  progress: number;
  velocity: number;
  setScroll: (progress: number, velocity?: number) => void;
};

/** Single scroll progress source for R3F (Lenis writes here in Phase 1+). */
export const useScrollStore = create<ScrollState>((set) => ({
  progress: 0,
  velocity: 0,
  setScroll: (progress, velocity = 0) =>
    set({ progress: Math.min(1, Math.max(0, progress)), velocity }),
}));

import { create } from "zustand";

export type HoveredCellInfo = {
  index: number;
  x: number;
  y: number;
  z: number;
  value: number;
  screenX: number;
  screenY: number;
};

type LatticeState = {
  hoveredCell: HoveredCellInfo | null;
  setHoveredCell: (cell: HoveredCellInfo | null) => void;
  shockwaveOrigin: [number, number, number];
  shockwaveTriggerTime: number;
  triggerShockwave: (x: number, y: number, z?: number) => void;
};

export const useLatticeStore = create<LatticeState>((set) => ({
  hoveredCell: null,
  setHoveredCell: (cell) => set({ hoveredCell: cell }),
  shockwaveOrigin: [0, 0, 0],
  shockwaveTriggerTime: -999,
  triggerShockwave: (x, y, z = 0) =>
    set({
      shockwaveOrigin: [x, y, z],
      shockwaveTriggerTime: performance.now() * 0.001,
    }),
}));


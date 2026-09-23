import { create } from "zustand";
import { planBatches, shuffledIndices, type BatchPlan } from "@/lib/rng-pcg32";
import { continuousScene, sceneAt, type SceneId } from "@/lib/story-scenes";

const DATASET_SIZE = 48;

type StoryState = {
  seed: number;
  batchSize: number;
  dropLast: boolean;
  indices: number[];
  batchPlan: BatchPlan;
  activeScene: SceneId;
  sceneT: number;
  setSeed: (seed: number) => void;
  setBatchSize: (size: number) => void;
  setDropLast: (value: boolean) => void;
  syncFromScroll: (progress: number) => void;
  recompute: () => void;
};

function buildPlan(seed: number, batchSize: number, dropLast: boolean) {
  const indices = shuffledIndices(DATASET_SIZE, seed);
  const batchPlan = planBatches(indices, batchSize, dropLast);
  return { indices, batchPlan };
}

export const useStoryStore = create<StoryState>((set, get) => {
  const initial = buildPlan(42, 8, true);
  return {
    seed: 42,
    batchSize: 8,
    dropLast: true,
    ...initial,
    activeScene: "hero",
    sceneT: 0,
    setSeed: (seed) => {
      const next = buildPlan(seed >>> 0, get().batchSize, get().dropLast);
      set({ seed: seed >>> 0, ...next });
    },
    setBatchSize: (batchSize) => {
      const size = Math.min(24, Math.max(1, Math.floor(batchSize)));
      const next = buildPlan(get().seed, size, get().dropLast);
      set({ batchSize: size, ...next });
    },
    setDropLast: (dropLast) => {
      const next = buildPlan(get().seed, get().batchSize, dropLast);
      set({ dropLast, ...next });
    },
    syncFromScroll: (progress) => {
      const active = sceneAt(progress).id;
      set({ activeScene: active, sceneT: continuousScene(progress) });
    },
    recompute: () => {
      const { seed, batchSize, dropLast } = get();
      set(buildPlan(seed, batchSize, dropLast));
    },
  };
});

export const STORY_DATASET_SIZE = DATASET_SIZE;

import { describe, expect, it } from "vitest";
import {
  computeReadyToFinish,
  type LoaderLogId,
} from "@/lib/stores/loader-store";

function baseState(
  overrides: Partial<Parameters<typeof computeReadyToFinish>[0]> = {},
) {
  return {
    phase: "loading" as const,
    progress: 100,
    minElapsed: true,
    assetsReady: true,
    sceneReady: true,
    fontsReady: true,
    reducedMotion: false,
    webgl: true as boolean | null,
    logs: [] as LoaderLogId[],
    morph: 0,
    setProgress: () => undefined,
    markAssetProgress: () => undefined,
    markFontsReady: () => undefined,
    markSceneReady: () => undefined,
    markMinElapsed: () => undefined,
    pushLog: () => undefined,
    setWebgl: () => undefined,
    setReducedMotion: () => undefined,
    setMorph: () => undefined,
    complete: () => undefined,
    skip: () => undefined,
    beginExplode: () => undefined,
    ...overrides,
  };
}

describe("computeReadyToFinish", () => {
  it("requires min elapsed and readiness gates", () => {
    expect(computeReadyToFinish(baseState({ minElapsed: false }))).toBe(false);
    expect(computeReadyToFinish(baseState())).toBe(true);
  });

  it("allows reduced-motion finish without scene", () => {
    expect(
      computeReadyToFinish(
        baseState({
          reducedMotion: true,
          sceneReady: false,
          assetsReady: false,
          fontsReady: true,
          minElapsed: true,
        }),
      ),
    ).toBe(true);
  });

  it("does not re-trigger when already ready", () => {
    expect(computeReadyToFinish(baseState({ phase: "ready" }))).toBe(false);
  });
});

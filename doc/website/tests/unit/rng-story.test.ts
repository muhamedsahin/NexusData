import { describe, expect, it } from "vitest";
import { Pcg32, planBatches, shuffledIndices } from "@/lib/rng-pcg32";
import { continuousScene, sceneAt } from "@/lib/story-scenes";

describe("Pcg32 determinism", () => {
  it("same seed yields same stream", () => {
    const a = new Pcg32(42);
    const b = new Pcg32(42);
    const seqA = Array.from({ length: 8 }, () => a.nextU32());
    const seqB = Array.from({ length: 8 }, () => b.nextU32());
    expect(seqA).toEqual(seqB);
  });

  it("shuffle is deterministic for a seed", () => {
    expect(shuffledIndices(16, 7)).toEqual(shuffledIndices(16, 7));
    expect(shuffledIndices(16, 7)).not.toEqual(shuffledIndices(16, 8));
  });

  it("drop_last removes incomplete trailing batch", () => {
    const indices = Array.from({ length: 10 }, (_, i) => i);
    const withDrop = planBatches(indices, 4, true);
    const without = planBatches(indices, 4, false);
    expect(withDrop.batches).toHaveLength(2);
    expect(withDrop.dropped).toEqual([8, 9]);
    expect(without.batches).toHaveLength(3);
  });
});

describe("story scenes", () => {
  it("maps progress to scene ids", () => {
    expect(sceneAt(0).id).toBe("hero");
    expect(sceneAt(0.2).id).toBe("shuffle");
    expect(sceneAt(0.99).id).toBe("cta");
  });

  it("produces continuous scene parameter", () => {
    expect(continuousScene(0)).toBeGreaterThanOrEqual(0);
    expect(continuousScene(1)).toBeGreaterThan(8);
  });
});

import { describe, expect, it } from "vitest";
import { benchmarkEntrySchema, benchmarkFileSchema } from "@/lib/benchmarks/schema";
import { loadBenchmarkSuite } from "@/lib/benchmarks/load";
import { planBatches, shuffledIndices } from "@/lib/rng-pcg32";

describe("benchmark schema", () => {
  it("accepts illustrative entries with null hardware", () => {
    const parsed = benchmarkEntrySchema.parse({
      id: "t",
      title: { en: "T", tr: "T" },
      metric: "throughput",
      unit: "GB/s",
      higherIsBetter: true,
      illustrative: true,
      date: null,
      libraryVersion: null,
      commit: null,
      hardware: null,
      methodology: { en: "m", tr: "m" },
      reproduce: "echo",
      results: [{ name: "a", value: 1, stddev: 0, runs: 0, isBaseline: true }],
    });
    expect(parsed.chartKind).toBe("bar");
    expect(parsed.illustrative).toBe(true);
  });

  it("loads suite from disk with unique ids", () => {
    const suite = loadBenchmarkSuite();
    expect(suite.length).toBeGreaterThanOrEqual(4);
    const ids = new Set(suite.map((s) => s.id));
    expect(ids.size).toBe(suite.length);
    for (const entry of suite) {
      expect(() => benchmarkEntrySchema.parse(entry)).not.toThrow();
    }
  });

  it("parses array files", () => {
    const arr = benchmarkFileSchema.parse([
      {
        id: "a",
        title: { en: "A", tr: "A" },
        metric: "m",
        unit: "u",
        higherIsBetter: true,
        illustrative: true,
        date: null,
        libraryVersion: null,
        commit: null,
        hardware: null,
        methodology: { en: "m", tr: "m" },
        reproduce: "r",
        results: [{ name: "x", value: 1 }],
      },
    ]);
    expect(Array.isArray(arr)).toBe(true);
  });
});

describe("dataloader simulator math", () => {
  it("drop_last removes incomplete final batch", () => {
    const idx = shuffledIndices(10, 42);
    const plan = planBatches(idx, 4, true);
    expect(plan.batches.every((b) => b.length === 4)).toBe(true);
    expect(plan.dropped.length).toBe(2);
  });
});

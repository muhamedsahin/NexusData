import fs from "node:fs";
import path from "node:path";
import {
  benchmarkEntrySchema,
  benchmarkFileSchema,
  type BenchmarkEntry,
} from "@/lib/benchmarks/schema";

const BENCH_ROOT = path.join(process.cwd(), "data", "benchmarks");

export function loadBenchmarkSuite(): BenchmarkEntry[] {
  if (!fs.existsSync(BENCH_ROOT)) return [];

  const files = fs
    .readdirSync(BENCH_ROOT)
    .filter((f) => f.endsWith(".json"))
    .sort();

  const byId = new Map<string, BenchmarkEntry>();

  for (const file of files) {
    const raw = JSON.parse(
      fs.readFileSync(path.join(BENCH_ROOT, file), "utf8"),
    ) as unknown;
    const parsed = benchmarkFileSchema.parse(raw);
    const entries = Array.isArray(parsed) ? parsed : [parsed];
    for (const entry of entries) {
      const validated = benchmarkEntrySchema.parse(entry);
      byId.set(validated.id, validated);
    }
  }

  return Array.from(byId.values()).sort((a, b) => a.id.localeCompare(b.id));
}

export function loadBenchmarkById(id: string): BenchmarkEntry | null {
  return loadBenchmarkSuite().find((b) => b.id === id) ?? null;
}

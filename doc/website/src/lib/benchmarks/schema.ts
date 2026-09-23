import { z } from "zod";

export const localizedStringSchema = z.object({
  en: z.string().min(1),
  tr: z.string().min(1),
});

export const hardwareSchema = z
  .object({
    cpu: z.string().min(1),
    cores: z.number().positive(),
    ram: z.string().min(1),
    gpu: z.string().nullable(),
    os: z.string().min(1),
    compiler: z.string().min(1),
    flags: z.string(),
  })
  .nullable();

export const benchmarkResultSchema = z.object({
  name: z.string().min(1),
  value: z.number(),
  stddev: z.number().nonnegative().default(0),
  runs: z.number().int().nonnegative().default(0),
  isBaseline: z.boolean().default(false),
});

export const chartKindSchema = z.enum(["bar", "scaling", "breakeven", "table"]);

export const benchmarkEntrySchema = z.object({
  id: z.string().min(1),
  title: localizedStringSchema,
  metric: z.string().min(1),
  unit: z.string().min(1),
  higherIsBetter: z.boolean(),
  illustrative: z.boolean(),
  date: z.string().nullable(),
  libraryVersion: z.string().nullable(),
  commit: z.string().nullable(),
  hardware: hardwareSchema,
  methodology: localizedStringSchema,
  reproduce: z.string().min(1),
  results: z.array(benchmarkResultSchema).min(1),
  chartKind: chartKindSchema.default("bar"),
});

export const benchmarkFileSchema = z.union([
  benchmarkEntrySchema,
  z.array(benchmarkEntrySchema).min(1),
]);

export type BenchmarkEntry = z.infer<typeof benchmarkEntrySchema>;
export type BenchmarkResult = z.infer<typeof benchmarkResultSchema>;
export type LocalizedString = z.infer<typeof localizedStringSchema>;

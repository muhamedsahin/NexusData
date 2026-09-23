"use client";

import { useMemo, useState } from "react";
import { useLocale } from "next-intl";
import { planBatches, shuffledIndices } from "@/lib/rng-pcg32";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/cn";

const copy = {
  en: {
    title: "DataLoader simulator",
    dataset: "Dataset size",
    batch: "Batch size",
    seed: "Seed",
    epoch: "Epoch",
    shuffle: "Shuffle",
    dropLast: "drop_last",
    batches: "Batches",
    dropped: "Dropped indices",
    note: "Uses the same TypeScript PCG32 + Fisher–Yates as the homepage. Illustrative until verified against C++ goldens.",
  },
  tr: {
    title: "DataLoader simülatörü",
    dataset: "Dataset boyutu",
    batch: "Batch boyutu",
    seed: "Seed",
    epoch: "Epoch",
    shuffle: "Shuffle",
    dropLast: "drop_last",
    batches: "Batch'ler",
    dropped: "Atılan indeksler",
    note: "Ana sayfadaki TypeScript PCG32 + Fisher–Yates ile aynı. C++ golden ile doğrulanana kadar örnek.",
  },
} as const;

export function DataLoaderSimulator() {
  const locale = useLocale() === "tr" ? "tr" : "en";
  const t = copy[locale];
  const [datasetSize, setDatasetSize] = useState(10);
  const [batchSize, setBatchSize] = useState(4);
  const [seed, setSeed] = useState(42);
  const [epoch, setEpoch] = useState(0);
  const [shuffle, setShuffle] = useState(true);
  const [dropLast, setDropLast] = useState(true);

  const plan = useMemo(() => {
    const n = Math.max(1, Math.min(256, Math.floor(datasetSize)));
    const bs = Math.max(1, Math.min(n, Math.floor(batchSize)));
    const effectiveSeed = (seed + epoch) >>> 0;
    const indices = shuffle
      ? shuffledIndices(n, effectiveSeed)
      : Array.from({ length: n }, (_, i) => i);
    return planBatches(indices, bs, dropLast);
  }, [datasetSize, batchSize, seed, epoch, shuffle, dropLast]);

  return (
    <div className="my-6 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)]/50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="font-semibold text-[color:var(--fg)]">{t.title}</h3>
        <StatusBadge status="illustrative" label={locale === "tr" ? "Örnek" : "Illustrative"} />
      </div>
      <p className="mb-4 text-xs text-[color:var(--fg-muted)]">{t.note}</p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">{t.dataset}</span>
          <input
            type="number"
            min={1}
            max={256}
            value={datasetSize}
            onChange={(e) => setDatasetSize(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">{t.batch}</span>
          <input
            type="number"
            min={1}
            max={256}
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">{t.seed}</span>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">{t.epoch}</span>
          <input
            type="number"
            min={0}
            value={epoch}
            onChange={(e) => setEpoch(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-[color:var(--fg)]">
          <input type="checkbox" checked={shuffle} onChange={(e) => setShuffle(e.target.checked)} />
          {t.shuffle}
        </label>
        <label className="flex items-center gap-2 text-xs text-[color:var(--fg)]">
          <input type="checkbox" checked={dropLast} onChange={(e) => setDropLast(e.target.checked)} />
          {t.dropLast}
        </label>
      </div>

      <p className="mt-4 text-xs font-medium text-[color:var(--fg)]">{t.batches}</p>
      <ul className="mt-2 space-y-1 font-mono text-xs">
        {plan.batches.map((b, i) => (
          <li
            key={i}
            className={cn(
              "rounded-md border border-[color:var(--border)] px-2 py-1",
              b.length < batchSize && "border-[color:var(--status-warn-border)]",
            )}
          >
            [{i}] {b.join(", ")}
          </li>
        ))}
      </ul>
      {plan.dropped.length > 0 ? (
        <p className="mt-3 text-xs text-[color:var(--status-warn)]">
          {t.dropped}: {plan.dropped.join(", ")}
        </p>
      ) : null}
    </div>
  );
}

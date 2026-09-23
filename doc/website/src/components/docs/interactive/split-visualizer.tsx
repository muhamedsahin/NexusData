"use client";

import { useMemo, useState } from "react";
import { useLocale } from "next-intl";
import { Pcg32 } from "@/lib/rng-pcg32";
import { StatusBadge } from "@/components/ui/status-badge";

function splitIndices(n: number, fractions: number[], seed: number): number[][] {
  const rng = new Pcg32(seed >>> 0);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    const tmp = idx[i]!;
    idx[i] = idx[j]!;
    idx[j] = tmp;
  }
  const sum = fractions.reduce((a, b) => a + b, 0) || 1;
  const norm = fractions.map((f) => f / sum);
  const cuts: number[] = [];
  let acc = 0;
  for (let i = 0; i < norm.length - 1; i++) {
    acc += norm[i]!;
    cuts.push(Math.round(acc * n));
  }
  const parts: number[][] = [];
  let start = 0;
  for (const c of cuts) {
    parts.push(idx.slice(start, c));
    start = c;
  }
  parts.push(idx.slice(start));
  return parts;
}

function classBalance(
  labels: number[],
  indices: number[],
  nClasses: number,
): number[] {
  const counts = Array.from({ length: nClasses }, () => 0);
  for (const i of indices) {
    const y = labels[i] ?? 0;
    counts[y]! += 1;
  }
  return counts;
}

export function SplitVisualizer() {
  const locale = useLocale() === "tr" ? "tr" : "en";
  const [n, setN] = useState(100);
  const [train, setTrain] = useState(0.7);
  const [val, setVal] = useState(0.15);
  const [seed, setSeed] = useState(7);
  const [stratified, setStratified] = useState(false);
  const nClasses = 3;

  const labels = useMemo(() => {
    const rng = new Pcg32(99);
    return Array.from({ length: n }, () => rng.nextInt(nClasses));
  }, [n]);

  const parts = useMemo(() => {
    const size = Math.max(3, Math.min(500, Math.floor(n)));
    if (!stratified) {
      return splitIndices(size, [train, val, 1 - train - val], seed);
    }
    // Approximate stratified: split per class then merge
    const byClass: number[][] = Array.from({ length: nClasses }, () => []);
    for (let i = 0; i < size; i++) {
      byClass[labels[i]!]!.push(i);
    }
    const trainI: number[] = [];
    const valI: number[] = [];
    const testI: number[] = [];
    for (const cls of byClass) {
      const [tr, va, te] = splitIndices(cls.length, [train, val, 1 - train - val], seed + cls.length);
      trainI.push(...tr.map((j) => cls[j]!));
      valI.push(...va.map((j) => cls[j]!));
      testI.push(...te.map((j) => cls[j]!));
    }
    return [trainI, valI, testI];
  }, [n, train, val, seed, stratified, labels]);

  const balances = parts.map((p) => classBalance(labels, p, nClasses));
  const maxCount = Math.max(...balances.flat(), 1);
  const names = locale === "tr" ? ["Train", "Val", "Test"] : ["Train", "Val", "Test"];

  return (
    <div className="my-6 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)]/50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="font-semibold text-[color:var(--fg)]">
          {locale === "tr" ? "Split görselleştirici" : "Split visualizer"}
        </h3>
        <StatusBadge status="illustrative" label={locale === "tr" ? "Örnek" : "Illustrative"} />
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">N</span>
          <input
            type="number"
            min={3}
            max={500}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">Train</span>
          <input
            type="number"
            step={0.05}
            min={0.1}
            max={0.9}
            value={train}
            onChange={(e) => setTrain(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">Val</span>
          <input
            type="number"
            step={0.05}
            min={0.05}
            max={0.5}
            value={val}
            onChange={(e) => setVal(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">Seed</span>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
      </div>
      <label className="mt-3 flex items-center gap-2 text-xs text-[color:var(--fg)]">
        <input
          type="checkbox"
          checked={stratified}
          onChange={(e) => setStratified(e.target.checked)}
        />
        {locale === "tr" ? "Stratified (yaklaşık)" : "Stratified (approx)"}
      </label>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        {balances.map((counts, i) => (
          <div key={names[i]}>
            <p className="mb-2 text-xs font-medium text-[color:var(--fg)]">
              {names[i]} (n={parts[i]!.length})
            </p>
            <div className="flex h-24 items-end gap-1">
              {counts.map((c, ci) => (
                <div
                  key={ci}
                  className="flex-1 rounded-t bg-[color:var(--accent)]"
                  style={{ height: `${(c / maxCount) * 100}%`, opacity: 0.5 + ci * 0.15 }}
                  title={`class ${ci}: ${c}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

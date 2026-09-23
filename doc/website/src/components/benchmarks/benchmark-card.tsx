"use client";

import { useId, useMemo, useState } from "react";
import type { BenchmarkEntry } from "@/lib/benchmarks/schema";
import type { SiteLocale } from "@/config/site.config";
import { StatusBadge } from "@/components/ui/status-badge";
import { Callout } from "@/components/ui/callout";
import { cn } from "@/lib/cn";

type Props = {
  entry: BenchmarkEntry;
  locale: SiteLocale;
};

function BarChart({ entry }: { entry: BenchmarkEntry }) {
  const max = Math.max(...entry.results.map((r) => r.value), 1e-9);
  const w = 560;
  const h = 220;
  const pad = { t: 16, r: 16, b: 48, l: 48 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const gap = 12;
  const barW = (innerW - gap * (entry.results.length - 1)) / entry.results.length;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-auto w-full" role="img" aria-hidden>
      {entry.results.map((r, i) => {
        const bh = (r.value / max) * innerH;
        const x = pad.l + i * (barW + gap);
        const y = pad.t + innerH - bh;
        return (
          <g key={r.name}>
            <rect
              x={x}
              y={y}
              width={barW}
              height={Math.max(bh, 1)}
              rx={4}
              className={r.isBaseline ? "fill-[color:var(--accent)]" : "fill-[color:var(--fg-muted)]"}
              opacity={0.85}
            />
            <text
              x={x + barW / 2}
              y={h - 12}
              textAnchor="middle"
              className="fill-[color:var(--fg-muted)] text-[10px]"
            >
              {r.name.length > 18 ? `${r.name.slice(0, 16)}…` : r.name}
            </text>
            <title>
              {r.name}: {r.value} {entry.unit}
            </title>
          </g>
        );
      })}
    </svg>
  );
}

function ScalingChart({ entry }: { entry: BenchmarkEntry }) {
  const w = 560;
  const h = 220;
  const pad = { t: 16, r: 24, b: 40, l: 48 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const max = Math.max(...entry.results.map((r) => r.value), 1e-9);
  const n = entry.results.length;
  const baseline = entry.results[0]?.value ?? 1;

  const points = entry.results.map((r, i) => {
    const x = pad.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
    const y = pad.t + innerH - (r.value / max) * innerH;
    return { x, y, r };
  });

  const ideal = entry.results.map((_, i) => {
    const workers = i + 1;
    const x = pad.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
    const idealVal = baseline * workers;
    const y = pad.t + innerH - (Math.min(idealVal, max * 1.05) / (max * 1.05)) * innerH;
    return `${x},${y}`;
  });

  const poly = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-auto w-full" role="img" aria-hidden>
      <polyline
        fill="none"
        stroke="currentColor"
        strokeDasharray="4 4"
        opacity={0.35}
        points={ideal.join(" ")}
        className="text-[color:var(--fg-muted)]"
      />
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        points={poly}
        className="text-[color:var(--accent)]"
      />
      {points.map((p) => (
        <circle
          key={p.r.name}
          cx={p.x}
          cy={p.y}
          r={4}
          className="fill-[color:var(--accent)]"
        >
          <title>
            {p.r.name}: {p.r.value} {entry.unit}
          </title>
        </circle>
      ))}
    </svg>
  );
}

function BreakEvenChart({ entry }: { entry: BenchmarkEntry }) {
  return <ScalingChart entry={entry} />;
}

function ResultsTable({
  entry,
  locale,
}: {
  entry: BenchmarkEntry;
  locale: SiteLocale;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-[color:var(--border)]">
      <table className="w-full min-w-[28rem] border-collapse text-sm">
        <caption className="sr-only">
          {entry.title[locale]} — {entry.unit}
        </caption>
        <thead>
          <tr>
            <th className="border-b border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-left">
              Name
            </th>
            <th className="border-b border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-left">
              Value ({entry.unit})
            </th>
            <th className="border-b border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-left">
              Stddev
            </th>
            <th className="border-b border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-left">
              Runs
            </th>
          </tr>
        </thead>
        <tbody>
          {entry.results.map((r) => (
            <tr key={r.name}>
              <td className="border-b border-[color:var(--border)] px-3 py-2">
                {r.name}
                {r.isBaseline ? " ★" : ""}
              </td>
              <td className="border-b border-[color:var(--border)] px-3 py-2 font-mono">
                {r.value}
              </td>
              <td className="border-b border-[color:var(--border)] px-3 py-2 font-mono">
                {r.stddev}
              </td>
              <td className="border-b border-[color:var(--border)] px-3 py-2 font-mono">
                {r.runs}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function BenchmarkCard({ entry, locale }: Props) {
  const [tab, setTab] = useState<"chart" | "table">("chart");
  const titleId = useId();
  const kind = entry.chartKind ?? "bar";

  const chart = useMemo(() => {
    if (kind === "table") return null;
    if (kind === "scaling") return <ScalingChart entry={entry} />;
    if (kind === "breakeven") return <BreakEvenChart entry={entry} />;
    return <BarChart entry={entry} />;
  }, [entry, kind]);

  return (
    <article
      className="my-8 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)]/40 p-5"
      aria-labelledby={titleId}
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 id={titleId} className="text-lg font-semibold text-[color:var(--fg)]">
          {entry.title[locale]}
        </h3>
        {entry.illustrative ? (
          <StatusBadge status="illustrative" label={locale === "tr" ? "Örnek" : "Illustrative"} />
        ) : null}
      </div>

      {entry.illustrative ? (
        <Callout tone="warning" className="mb-4" title={locale === "tr" ? "Ölçülmedi" : "Not measured"}>
          {entry.methodology[locale]}
        </Callout>
      ) : (
        <p className="mb-4 text-sm text-[color:var(--fg-muted)]">{entry.methodology[locale]}</p>
      )}

      {entry.hardware ? (
        <dl className="mb-4 grid gap-1 text-xs text-[color:var(--fg-muted)] sm:grid-cols-2">
          <div>
            <dt className="inline font-medium text-[color:var(--fg)]">CPU: </dt>
            <dd className="inline">
              {entry.hardware.cpu} ({entry.hardware.cores} cores)
            </dd>
          </div>
          <div>
            <dt className="inline font-medium text-[color:var(--fg)]">RAM: </dt>
            <dd className="inline">{entry.hardware.ram}</dd>
          </div>
          <div>
            <dt className="inline font-medium text-[color:var(--fg)]">OS: </dt>
            <dd className="inline">{entry.hardware.os}</dd>
          </div>
          <div>
            <dt className="inline font-medium text-[color:var(--fg)]">Compiler: </dt>
            <dd className="inline">
              {entry.hardware.compiler} {entry.hardware.flags}
            </dd>
          </div>
        </dl>
      ) : null}

      {kind !== "table" ? (
        <div className="mb-3 flex gap-2" role="tablist" aria-label="Chart or table">
          {(["chart", "table"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium",
                tab === t
                  ? "bg-[color:var(--accent)] text-[color:var(--accent-fg,#fff)]"
                  : "bg-[color:var(--surface)] text-[color:var(--fg-muted)]",
              )}
              onClick={() => setTab(t)}
            >
              {t === "chart"
                ? locale === "tr"
                  ? "Grafik"
                  : "Chart"
                : locale === "tr"
                  ? "Tablo"
                  : "Table"}
            </button>
          ))}
        </div>
      ) : null}

      {kind === "table" || tab === "table" ? (
        <ResultsTable entry={entry} locale={locale} />
      ) : (
        <div className="rounded-xl border border-[color:var(--border)] bg-[color:var(--bg)] p-3">
          {chart}
        </div>
      )}

      <div className="mt-4 space-y-1 text-xs text-[color:var(--fg-muted)]">
        {entry.libraryVersion ? <p>Library: {entry.libraryVersion}</p> : null}
        {entry.commit ? <p>Commit: {entry.commit}</p> : null}
        {entry.date ? <p>Date: {entry.date}</p> : null}
        <p>
          <span className="font-medium text-[color:var(--fg)]">
            {locale === "tr" ? "Yeniden üret:" : "Reproduce:"}
          </span>{" "}
          <code className="font-mono">{entry.reproduce}</code>
        </p>
      </div>
    </article>
  );
}

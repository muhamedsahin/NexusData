"use client";

import { useMemo, useState } from "react";
import { useLocale } from "next-intl";
import { StatusBadge } from "@/components/ui/status-badge";

const DTYPE_BYTES: Record<string, number> = {
  Bool: 1,
  Int8: 1,
  UInt8: 1,
  Int16: 2,
  UInt16: 2,
  Int32: 4,
  UInt32: 4,
  Int64: 8,
  UInt64: 8,
  Float16: 2,
  BFloat16: 2,
  Float32: 4,
  Float64: 8,
};

function parseShape(raw: string): number[] | null {
  const parts = raw
    .split(/[,\s×x]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return null;
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => !Number.isFinite(n) || n < 0 || !Number.isInteger(n))) return null;
  return nums;
}

export function ShapeExplorer() {
  const locale = useLocale() === "tr" ? "tr" : "en";
  const [shapeRaw, setShapeRaw] = useState("64, 3, 224, 224");
  const [dtype, setDtype] = useState("Float32");
  const [reshapeRaw, setReshapeRaw] = useState("64, 150528");

  const shape = useMemo(() => parseShape(shapeRaw), [shapeRaw]);
  const reshape = useMemo(() => parseShape(reshapeRaw), [reshapeRaw]);

  const numel = shape?.reduce((a, b) => a * b, 1) ?? 0;
  const itemsize = DTYPE_BYTES[dtype] ?? 4;
  const nbytes = numel * itemsize;
  const reshapeOk =
    shape && reshape ? reshape.reduce((a, b) => a * b, 1) === numel : false;

  return (
    <div className="my-6 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)]/50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="font-semibold text-[color:var(--fg)]">
          {locale === "tr" ? "Shape gezgini" : "Shape explorer"}
        </h3>
        <StatusBadge status="stable" label="NDArray" />
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <label className="text-xs sm:col-span-2">
          <span className="mb-1 block text-[color:var(--fg-muted)]">Shape</span>
          <input
            value={shapeRaw}
            onChange={(e) => setShapeRaw(e.target.value)}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">DType</span>
          <select
            value={dtype}
            onChange={(e) => setDtype(e.target.value)}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 text-sm"
          >
            {Object.keys(DTYPE_BYTES).map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs sm:col-span-3">
          <span className="mb-1 block text-[color:var(--fg-muted)]">Reshape to</span>
          <input
            value={reshapeRaw}
            onChange={(e) => setReshapeRaw(e.target.value)}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 font-mono text-sm"
          />
        </label>
      </div>

      <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-3">
        <div className="rounded-lg border border-[color:var(--border)] px-3 py-2">
          <dt className="text-xs text-[color:var(--fg-muted)]">numel</dt>
          <dd className="font-mono">{shape ? numel : "—"}</dd>
        </div>
        <div className="rounded-lg border border-[color:var(--border)] px-3 py-2">
          <dt className="text-xs text-[color:var(--fg-muted)]">nbytes</dt>
          <dd className="font-mono">
            {shape ? `${nbytes} (${(nbytes / (1024 * 1024)).toFixed(3)} MiB)` : "—"}
          </dd>
        </div>
        <div className="rounded-lg border border-[color:var(--border)] px-3 py-2">
          <dt className="text-xs text-[color:var(--fg-muted)]">reshape</dt>
          <dd className="font-mono">
            {!shape || !reshape
              ? "—"
              : reshapeOk
                ? locale === "tr"
                  ? "geçerli"
                  : "valid"
                : locale === "tr"
                  ? "geçersiz (numel uyuşmaz)"
                  : "invalid (numel mismatch)"}
          </dd>
        </div>
      </dl>
    </div>
  );
}

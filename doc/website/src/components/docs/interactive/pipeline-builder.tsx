"use client";

import { useMemo, useState } from "react";
import { useLocale } from "next-intl";
import { StatusBadge } from "@/components/ui/status-badge";

type NodeId = "source" | "transform" | "sampler" | "loader";

const NODE_ORDER: NodeId[] = ["source", "transform", "sampler", "loader"];

const LABELS = {
  en: {
    title: "Pipeline builder",
    source: "Source",
    transform: "Transforms",
    sampler: "Sampler",
    loader: "DataLoader",
    copy: "Copy C++",
    copied: "Copied",
    note: "Generated snippet is a Design preview — verify against public headers before shipping.",
  },
  tr: {
    title: "Pipeline oluşturucu",
    source: "Source",
    transform: "Transform",
    sampler: "Sampler",
    loader: "DataLoader",
    copy: "C++ kopyala",
    copied: "Kopyalandı",
    note: "Üretilen snippet Design preview — yayınlamadan önce public başlıklarla doğrulayın.",
  },
} as const;

const OPTIONS: Record<NodeId, { id: string; label: string }[]> = {
  source: [
    { id: "csv", label: "CSVDataset" },
    { id: "image", label: "ImageFolderDataset" },
    { id: "mnist", label: "MNIST" },
  ],
  transform: [
    { id: "none", label: "(none)" },
    { id: "resize", label: "Resize+ToTensor" },
    { id: "clip", label: "ClipTransform" },
  ],
  sampler: [
    { id: "seq", label: "Sequential" },
    { id: "rand", label: "Random / shuffle" },
  ],
  loader: [
    { id: "stack", label: "collate_stack" },
    { id: "pad", label: "collate_pad_sequence" },
  ],
};

function generateCpp(sel: Record<NodeId, string>): string {
  const lines: string[] = [
    '#include "nexusdata/nexusdata.hpp"',
    "using namespace nexusdata;",
    "",
  ];

  if (sel.source === "csv") {
    lines.push('CSVOptions opt; opt.label_column = "target";');
    lines.push('auto raw = std::make_shared<CSVDataset>("train.csv", opt);');
  } else if (sel.source === "image") {
    lines.push('auto raw = std::make_shared<ImageFolderDataset>("data/train");');
  } else {
    lines.push('auto raw = std::make_shared<MNIST>("data/mnist");');
  }

  if (sel.transform === "resize") {
    lines.push(
      "auto tfm = std::make_shared<Compose>({",
      "  std::make_shared<Resize>(224, 224),",
      "  std::make_shared<ToTensor>(),",
      "});",
      "auto ds = std::make_shared<MapDataset>(raw, tfm);",
    );
  } else if (sel.transform === "clip") {
    lines.push(
      "auto tfm = std::make_shared<ClipTransform>(-1.0, 1.0);",
      "auto ds = std::make_shared<MapDataset>(raw, tfm);",
    );
  } else {
    lines.push("auto ds = raw;");
  }

  lines.push("");
  lines.push("DataLoaderOptions lo;");
  lines.push("lo.batch_size = 64;");
  lines.push(`lo.shuffle = ${sel.sampler === "rand" ? "true" : "false"};`);
  lines.push("lo.seed = 42;");
  if (sel.loader === "pad") {
    lines.push("DataLoader loader(ds, lo, collate_pad_sequence);");
  } else {
    lines.push("DataLoader loader(ds, lo); // collate_stack");
  }
  lines.push("for (const Batch& b : loader) { /* ... */ }");
  return lines.join("\n");
}

export function PipelineBuilder() {
  const locale = useLocale() === "tr" ? "tr" : "en";
  const t = LABELS[locale];
  const [sel, setSel] = useState<Record<NodeId, string>>({
    source: "csv",
    transform: "none",
    sampler: "rand",
    loader: "stack",
  });
  const [copied, setCopied] = useState(false);
  const code = useMemo(() => generateCpp(sel), [sel]);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="my-6 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)]/50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="font-semibold text-[color:var(--fg)]">{t.title}</h3>
        <StatusBadge status="design-preview" label="Design preview" />
      </div>
      <p className="mb-4 text-xs text-[color:var(--fg-muted)]">{t.note}</p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {NODE_ORDER.map((node) => (
          <label key={node} className="text-xs">
            <span className="mb-1 block text-[color:var(--fg-muted)]">{t[node]}</span>
            <select
              value={sel[node]}
              onChange={(e) => setSel((s) => ({ ...s, [node]: e.target.value }))}
              className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 text-sm"
            >
              {OPTIONS[node].map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      <div className="mt-4 flex items-center justify-between gap-2">
        <p className="text-xs text-[color:var(--fg-muted)]">
          {NODE_ORDER.map((n) => OPTIONS[n].find((o) => o.id === sel[n])?.label).join(" → ")}
        </p>
        <button
          type="button"
          onClick={onCopy}
          className="rounded-md bg-[color:var(--accent)] px-3 py-1.5 text-xs font-medium text-white"
        >
          {copied ? t.copied : t.copy}
        </button>
      </div>
      <pre className="mt-3 overflow-x-auto rounded-xl bg-[#0d1117] p-3 font-mono text-xs text-white/85">
        {code}
      </pre>
    </div>
  );
}

"use client";

import { useState } from "react";
import { cn } from "@/lib/cn";
import { ArrowRight, Database, Layers, Shuffle, Cpu, Box, Sparkles } from "lucide-react";

interface PipelineNode {
  id: string;
  name: string;
  shortTag: string;
  detail: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NODES: PipelineNode[] = [
  {
    id: "source",
    name: "Source",
    shortTag: "mmap / S3",
    detail: "Zero-copy mmap & asynchronous io_uring streaming",
    icon: Database,
  },
  {
    id: "dataset",
    name: "Dataset",
    shortTag: "Arrow/Parquet",
    detail: "High-throughput columnar record decoding",
    icon: Layers,
  },
  {
    id: "sampler",
    name: "Sampler",
    shortTag: "Deterministic",
    detail: "SIMD seeded pseudo-random permutation",
    icon: Shuffle,
  },
  {
    id: "dataloader",
    name: "DataLoader",
    shortTag: "Lock-Free",
    detail: "Worker pool prefetch with ring buffer",
    icon: Cpu,
  },
  {
    id: "batch",
    name: "Batch",
    shortTag: "SIMD Collate",
    detail: "Fused memory collation & drop_last handling",
    icon: Box,
  },
  {
    id: "ndarray",
    name: "NDArray",
    shortTag: "CUDA / Metal",
    detail: "Direct GPU tensor dispatch without copy",
    icon: Sparkles,
  },
];

export function PipelineIndicator({ className }: { className?: string }) {
  const [activeNode, setActiveNode] = useState<string | null>(null);

  return (
    <div className={cn("w-full min-w-0 max-w-full py-1", className)}>
      <div className="flex items-center justify-between pb-1.5">
        <span className="font-mono text-[11px] font-medium uppercase tracking-[0.2em] text-[color:var(--accent)]">
          Architecture Pipeline
        </span>
        <span className="font-mono text-[10px] text-[#94a3b8] hidden sm:inline">
          Hover node to inspect
        </span>
      </div>

      {/* Pipeline Track */}
      <div className="relative flex flex-wrap items-center gap-1.5 rounded-xl border border-[color:var(--border)] bg-[color-mix(in_oklab,var(--bg)_82%,transparent)] p-1.5 backdrop-blur-md">
        {NODES.map((node, index) => {
          const isHovered = activeNode === node.id;
          const Icon = node.icon;
          const isLast = index === NODES.length - 1;

          return (
            <div key={node.id} className="flex items-center shrink-0">
              <div
                onMouseEnter={() => setActiveNode(node.id)}
                onMouseLeave={() => setActiveNode(null)}
                className={cn(
                  "group relative flex cursor-pointer items-center gap-1.5 rounded-lg px-2.5 py-1.5 transition-all duration-200",
                  isHovered
                    ? "bg-[color:var(--accent-soft)] ring-1 ring-[color:var(--accent)]"
                    : "bg-white/[0.03] hover:bg-white/[0.07]",
                )}
              >
                <Icon
                  className={cn(
                    "h-3.5 w-3.5 transition-colors duration-200",
                    isHovered
                      ? "text-[color:var(--accent)]"
                      : index === NODES.length - 1
                        ? "text-[color:var(--accent)]"
                        : "text-[#94a3b8]",
                  )}
                />
                <span
                  className={cn(
                    "font-mono text-xs font-medium transition-colors",
                    isHovered ? "text-[color:var(--fg)]" : "text-[#cbd5e1]",
                  )}
                >
                  {node.name}
                </span>

                {/* Floating tooltip on active node */}
                {isHovered ? (
                  <div className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2 whitespace-nowrap rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2.5 py-1 text-[11px] font-mono text-[color:var(--fg)] shadow-xl backdrop-blur-lg">
                    <span className="font-semibold text-[color:var(--accent)]">
                      {node.shortTag}:
                    </span>{" "}
                    {node.detail}
                    <div className="absolute -bottom-1 left-1/2 h-2 w-2 -translate-x-1/2 rotate-45 border-b border-r border-[color:var(--border)] bg-[color:var(--bg)]" />
                  </div>
                ) : null}
              </div>

              {/* Animated connector chevron / pulse line */}
              {!isLast ? (
                <div className="flex items-center px-1 text-[color:var(--fg-muted)]">
                  <ArrowRight className="h-3 w-3" />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}


"use client";

import { useLatticeStore } from "@/lib/stores/lattice-store";
import { cn } from "@/lib/cn";

export function LatticeTooltip() {
  const hoveredCell = useLatticeStore((s) => s.hoveredCell);

  if (!hoveredCell) return null;

  return (
    <div
      className="pointer-events-none fixed z-50 transition-transform duration-75 ease-out"
      style={{
        left: `${hoveredCell.screenX + 16}px`,
        top: `${hoveredCell.screenY - 32}px`,
      }}
    >
      <div className="flex flex-col gap-0.5 rounded-lg border border-[#2DE2D0]/50 bg-[#04060B]/90 px-2.5 py-1.5 shadow-[0_0_20px_rgba(45,226,208,0.3)] backdrop-blur-md">
        <div className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-[#2DE2D0] animate-pulse" />
          <span className="font-mono text-xs font-semibold text-[#E8F1FF]">
            [{hoveredCell.x}, {hoveredCell.y}, {hoveredCell.z}] ={" "}
            <span className="text-[#2DE2D0]">{hoveredCell.value.toFixed(3)}</span>
          </span>
          <span className="font-mono text-[10px] text-[#94a3b8]">float32</span>
        </div>
        <div className="flex items-center justify-between gap-3 border-t border-white/10 pt-0.5 font-mono text-[10px] text-[#94a3b8]">
          <span>shape: (8, 8, 8)</span>
          <span className="text-[#5B4BDB]">layout: row-major</span>
        </div>
      </div>
    </div>
  );
}


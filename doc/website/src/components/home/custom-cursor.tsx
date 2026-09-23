"use client";

import { useEffect, useRef, useState } from "react";
import { usePointerStore } from "@/lib/stores/scene-store";
import { useLoaderStore } from "@/lib/stores/loader-store";
import { cn } from "@/lib/cn";

/**
 * Desktop custom cursor:
 * - Inner point with direct tracking
 * - Outer ring with smooth spring damping
 * - Interactive state expansion over links & buttons
 * - Disappears on mouseleave / touch / reduced motion
 */
export function CustomCursor() {
  const [enabled, setEnabled] = useState(false);
  const [visible, setVisible] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [inspecting, setInspecting] = useState(false);

  const ringPosRef = useRef({ x: -100, y: -100 });
  const targetPosRef = useRef({ x: -100, y: -100 });
  const ringElRef = useRef<HTMLDivElement>(null);
  const dotElRef = useRef<HTMLDivElement>(null);

  const phase = useLoaderStore((s) => s.phase);
  const reduced = useLoaderStore((s) => s.reducedMotion);

  useEffect(() => {
    const fine = window.matchMedia("(pointer: fine)").matches;
    setEnabled(fine && !reduced);
  }, [reduced]);

  useEffect(() => {
    if (!enabled) return;

    const onMouseMove = (e: MouseEvent) => {
      targetPosRef.current = { x: e.clientX, y: e.clientY };
      if (!visible) setVisible(true);

      // Direct update on dot for zero input lag
      if (dotElRef.current) {
        dotElRef.current.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0) translate(-50%, -50%)`;
      }
    };

    const onMouseLeave = () => setVisible(false);
    const onMouseEnter = () => setVisible(true);

    const onOver = (e: Event) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const interactive = target.closest(
        "a,button,[role='button'],input,textarea,summary,[data-interactive='true']",
      );
      const canvas = target.closest("canvas");
      setHovering(Boolean(interactive));
      setInspecting(Boolean(canvas && !interactive));
    };

    window.addEventListener("mousemove", onMouseMove, { passive: true });
    document.addEventListener("mouseleave", onMouseLeave);
    document.addEventListener("mouseenter", onMouseEnter);
    document.addEventListener("mouseover", onOver);

    // Spring animation loop for outer ring
    let rafId: number;
    const animateRing = () => {
      const cur = ringPosRef.current;
      const target = targetPosRef.current;
      // Spring lerp factor
      cur.x += (target.x - cur.x) * 0.18;
      cur.y += (target.y - cur.y) * 0.18;

      if (ringElRef.current) {
        ringElRef.current.style.transform = `translate3d(${cur.x}px, ${cur.y}px, 0) translate(-50%, -50%)`;
      }
      rafId = requestAnimationFrame(animateRing);
    };
    rafId = requestAnimationFrame(animateRing);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseleave", onMouseLeave);
      document.removeEventListener("mouseenter", onMouseEnter);
      document.removeEventListener("mouseover", onOver);
      cancelAnimationFrame(rafId);
    };
  }, [enabled, visible]);

  if (!enabled || phase === "loading" || phase === "booting") return null;

  return (
    <div
      className={cn(
        "pointer-events-none fixed inset-0 z-[70] hidden transition-opacity duration-300 md:block",
        visible ? "opacity-100" : "opacity-0",
      )}
      aria-hidden
    >
      {/* Outer spring ring */}
      <div
        ref={ringElRef}
        className={cn(
          "absolute left-0 top-0 rounded-full border transition-[width,height,border-color,background-color] duration-200 ease-out will-change-transform",
          hovering
            ? "h-12 w-12 border-[color:var(--accent)] bg-[color:var(--accent)]/15 shadow-[0_0_16px_rgba(45,226,208,0.35)]"
            : inspecting
              ? "h-10 w-10 border-[color:var(--accent)]/80 border-dashed"
              : "h-8 w-8 border-[color:var(--accent)]/60 bg-transparent shadow-[0_0_8px_rgba(45,226,208,0.15)]",
        )}
      />

      {/* Center sharp dot */}
      <div
        ref={dotElRef}
        className={cn(
          "absolute left-0 top-0 h-2 w-2 rounded-full bg-[color:var(--accent)] shadow-[0_0_6px_var(--accent)] will-change-transform",
        )}
      />
    </div>
  );
}

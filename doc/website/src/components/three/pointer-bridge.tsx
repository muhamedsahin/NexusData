"use client";

import { useEffect } from "react";
import { usePointerStore } from "@/lib/stores/scene-store";
import { useLatticeStore } from "@/lib/stores/lattice-store";

type PointerBridgeProps = {
  targetRef: React.RefObject<HTMLElement | null>;
};

export function PointerBridge({ targetRef }: PointerBridgeProps) {
  useEffect(() => {
    const onMove = (clientX: number, clientY: number) => {
      const w = window.innerWidth || 1;
      const h = window.innerHeight || 1;
      const x = (clientX / w) * 2 - 1;
      const y = -((clientY / h) * 2 - 1);
      usePointerStore.getState().setTarget(x, y);
    };

    const onPointer = (e: PointerEvent) => onMove(e.clientX, e.clientY);
    const onTouch = (e: TouchEvent) => {
      const t = e.touches[0];
      if (t) onMove(t.clientX, t.clientY);
    };

    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        target.closest("a,button,input,textarea,summary,[role='button']")
      ) {
        return;
      }
      const w = window.innerWidth || 1;
      const h = window.innerHeight || 1;
      const ndcX = (e.clientX / w) * 2 - 1;
      const ndcY = -((e.clientY / h) * 2 - 1);
      useLatticeStore.getState().triggerShockwave(ndcX * 3.5, ndcY * 2.2, 0);
    };

    window.addEventListener("pointermove", onPointer, { passive: true });
    window.addEventListener("touchmove", onTouch, { passive: true });
    window.addEventListener("click", onClick);

    return () => {
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("click", onClick);
    };
  }, [targetRef]);

  return null;
}

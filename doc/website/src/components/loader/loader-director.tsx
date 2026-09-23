"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import {
  computeReadyToFinish,
  hasSeenLoader,
  useLoaderStore,
} from "@/lib/stores/loader-store";
import { prefersReducedMotion } from "@/lib/webgl";

/** Drives morph / explode / complete once real readiness gates pass. */
export function LoaderDirector() {
  const phase = useLoaderStore((s) => s.phase);
  const morphTween = useRef<gsap.core.Tween | null>(null);

  useEffect(() => {
    useLoaderStore.getState().setReducedMotion(prefersReducedMotion());
    if (hasSeenLoader()) {
      useLoaderStore.getState().skip();
      return;
    }

    const minMs = prefersReducedMotion() ? 400 : 1200;
    const timer = window.setTimeout(() => {
      useLoaderStore.getState().markMinElapsed();
    }, minMs);

    const fonts = document.fonts?.ready;
    if (fonts) {
      void fonts.then(() => useLoaderStore.getState().markFontsReady());
    } else {
      useLoaderStore.getState().markFontsReady();
    }

    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      const state = useLoaderStore.getState();
      if (!computeReadyToFinish(state)) return;
      if (state.reducedMotion || state.webgl === false) {
        state.complete();
        return;
      }
      state.beginExplode();
    }, 50);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (phase !== "exploding") return;
    morphTween.current?.kill();
    const obj = { v: useLoaderStore.getState().morph };
    morphTween.current = gsap.to(obj, {
      v: 1,
      duration: 1.1,
      ease: "power3.out",
      onUpdate: () => useLoaderStore.getState().setMorph(obj.v),
      onComplete: () => useLoaderStore.getState().complete(),
    });
    return () => {
      morphTween.current?.kill();
    };
  }, [phase]);

  return null;
}

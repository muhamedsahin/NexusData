"use client";

import { useEffect } from "react";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useScrollStore } from "@/lib/stores/scene-store";
import { useStoryStore } from "@/lib/stores/story-store";
import { prefersReducedMotion } from "@/lib/webgl";

gsap.registerPlugin(ScrollTrigger);

/**
 * Single scroll progress source: Lenis drives ScrollTrigger via scrollerProxy,
 * both write normalized progress into zustand for R3F.
 */
export function HomeLenis() {
  useEffect(() => {
    const reduced = prefersReducedMotion();

    if (reduced) {
      const onScroll = () => {
        const max = Math.max(
          1,
          document.documentElement.scrollHeight - window.innerHeight,
        );
        const progress = window.scrollY / max;
        useScrollStore.getState().setScroll(progress, 0);
        useStoryStore.getState().syncFromScroll(progress);
      };
      onScroll();
      window.addEventListener("scroll", onScroll, { passive: true });
      return () => window.removeEventListener("scroll", onScroll);
    }

    const lenis = new Lenis({
      lerp: 0.08,
      smoothWheel: true,
    });

    lenis.on("scroll", (e) => {
      const limit = Math.max(1, lenis.limit);
      const progress = e.scroll / limit;
      useScrollStore.getState().setScroll(progress, e.velocity);
      useStoryStore.getState().syncFromScroll(progress);
      ScrollTrigger.update();
    });

    if (typeof window !== "undefined") {
      (window as unknown as { __setScrollProgress?: (p: number) => void }).__setScrollProgress = (
        p: number,
      ) => {
        const clamped = Math.min(1, Math.max(0, p));
        const limit = Math.max(1, lenis.limit);
        lenis.scrollTo(clamped * limit, { immediate: true });
        useScrollStore.getState().setScroll(clamped, 0);
        useStoryStore.getState().syncFromScroll(clamped);
        ScrollTrigger.update();
      };
    }

    ScrollTrigger.scrollerProxy(document.body, {
      scrollTop(value) {
        if (typeof value === "number") {
          lenis.scrollTo(value, { immediate: true });
        }
        return lenis.scroll;
      },
      getBoundingClientRect() {
        return {
          top: 0,
          left: 0,
          width: window.innerWidth,
          height: window.innerHeight,
        };
      },
    });

    ScrollTrigger.defaults({ scroller: document.body });
    ScrollTrigger.refresh();

    let frame = 0;
    const raf = (time: number) => {
      lenis.raf(time);
      frame = requestAnimationFrame(raf);
    };
    frame = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(frame);
      ScrollTrigger.getAll().forEach((t) => t.kill());
      lenis.destroy();
    };
  }, []);

  return null;
}

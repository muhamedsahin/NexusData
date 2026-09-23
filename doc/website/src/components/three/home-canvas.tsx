"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { detectWebGL } from "@/lib/webgl";
import { useLoaderStore } from "@/lib/stores/loader-store";
import { HeroFallback } from "@/components/three/hero-fallback";

const HomeCanvasInner = dynamic(
  () =>
    import("@/components/three/home-canvas-inner").then(
      (m) => m.HomeCanvasInner,
    ),
  { ssr: false, loading: () => null },
);

type HomeCanvasProps = {
  className?: string;
};

export function HomeCanvas({ className }: HomeCanvasProps) {
  const [webgl, setWebgl] = useState<boolean | null>(null);
  const setWebglStore = useLoaderStore((s) => s.setWebgl);

  useEffect(() => {
    const ok = detectWebGL();
    setWebgl(ok);
    setWebglStore(ok);
    if (!ok) {
      useLoaderStore.getState().markSceneReady();
      useLoaderStore.getState().markAssetProgress(1);
    }
  }, [setWebglStore]);

  if (webgl === false) {
    return <HeroFallback className={className} />;
  }

  return (
    <div className={className} aria-hidden>
      {webgl ? <HomeCanvasInner /> : <HeroFallback className="opacity-40" />}
    </div>
  );
}

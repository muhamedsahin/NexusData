"use client";

import {
  EffectComposer,
  Bloom,
  Vignette,
} from "@react-three/postprocessing";
import { useQualityStore } from "@/lib/stores/quality-store";
import { useLoaderStore } from "@/lib/stores/loader-store";

export function PostFX() {
  const enablePost = useQualityStore((s) => s.enablePost);
  const level = useQualityStore((s) => s.level);
  const reduced = useLoaderStore((s) => s.reducedMotion);
  const morph = useLoaderStore((s) => s.morph);

  if (!enablePost || reduced || morph < 0.2) return null;

  return (
      <EffectComposer multisampling={level === "high" ? 2 : 0}>
        <Bloom
          intensity={level === "high" ? 0.32 : 0.2}
          luminanceThreshold={0.92}
          luminanceSmoothing={0.2}
          mipmapBlur
        />
        <Vignette eskil={false} offset={0.35} darkness={0.45} />
      </EffectComposer>
  );
}

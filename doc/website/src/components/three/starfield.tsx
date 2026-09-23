"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { buildStarGeometry } from "@/components/three/geometry";
import { useLoaderStore } from "@/lib/stores/loader-store";
import { useQualityStore } from "@/lib/stores/quality-store";

const starAvoidVertexShader = /* glsl */ `
uniform float uTime;
uniform float uReduced;
attribute float aSeed;
varying float vAlpha;
varying vec2 vNdc;

void main() {
  vec3 p = position;
  if (uReduced < 0.5) {
    p.z += sin(uTime * 0.04 + aSeed * 25.0) * 0.12;
  }
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vec4 clip = projectionMatrix * mv;
  vNdc = clip.xy / clip.w;

  // Gentle twinkle
  float twinkle = sin(uTime * 0.8 + aSeed * 50.0) * 0.25 + 0.75;
  vAlpha = mix(0.1, 0.45, aSeed) * twinkle;

  gl_Position = clip;
  gl_PointSize = mix(1.0, 2.0, aSeed) * (140.0 / max(30.0, -mv.z));
}
`;

const starAvoidFragmentShader = /* glsl */ `
varying float vAlpha;
varying vec2 vNdc;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float d = dot(uv, uv);
  if (d > 1.0) discard;

  // Text avoidance mask: dim stars on left half where text sits (vNdc.x < 0.1)
  float textAvoidance = smoothstep(-0.8, 0.2, vNdc.x);

  // Colour variation: cyan-tinted for bright stars, warm-white for dim
  float tint = step(0.65, vAlpha / 0.55); // fraction 0..1
  vec3 starCol = mix(vec3(0.82, 0.90, 1.0), vec3(0.72, 0.95, 1.0), tint);

  float alpha = smoothstep(1.0, 0.0, d) * vAlpha * mix(0.3, 1.0, textAvoidance);

  gl_FragColor = vec4(starCol, alpha);
}
`;

export function Starfield() {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const level = useQualityStore((s) => s.level);
  // ART_DIRECTION: 800 subtle points max
  const count = level === "high" ? 800 : level === "medium" ? 550 : 320;

  const { geometry, material } = useMemo(() => {
    const geo = buildStarGeometry(count);
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: starAvoidVertexShader,
      fragmentShader: starAvoidFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uReduced: { value: 0 },
      },
    });
    return { geometry: geo, material: mat };
  }, [count]);

  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  useFrame((state) => {
    const mat = materialRef.current ?? material;
    mat.uniforms.uTime.value = state.clock.elapsedTime;
    mat.uniforms.uReduced.value = useLoaderStore.getState().reducedMotion
      ? 1
      : 0;
  });

  return (
    <points geometry={geometry} frustumCulled={false}>
      <primitive object={material} ref={materialRef} attach="material" />
    </points>
  );
}

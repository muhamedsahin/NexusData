"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useLoaderStore } from "@/lib/stores/loader-store";

/* ── Shader 1: Central Volumetric Glow (cyan/violet double-lobe) ── */
const glowVertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const glowFragmentShader = /* glsl */ `
uniform float uTime;
uniform float uMorph;
uniform vec3 uColorCore;
uniform vec3 uColorAura;
uniform vec3 uColorAccent3;
varying vec2 vUv;

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  p.x *= 1.35;

  float dist = length(p);

  // Organic pulsation — two frequencies beating together
  float pulse  = 0.9 + sin(uTime * 0.65) * 0.08 + sin(uTime * 1.1) * 0.04;
  float pulse2 = 0.9 + sin(uTime * 0.4 + 1.2) * 0.06;

  // Triple-layer glow: tight core, medium aura, wide haze
  float core  = exp(-dist * 2.8) * 0.9 * pulse;
  float aura  = exp(-dist * 1.4) * 0.5 * pulse2;
  float haze  = exp(-dist * 0.7) * 0.22;

  // Shifted lobe for secondary color
  vec2 p2 = p + vec2(0.35, -0.2);
  float lobe = exp(-length(p2) * 3.0) * 0.6;

  vec3 col = uColorCore  * core
           + uColorAura  * aura
           + uColorAccent3 * lobe
           + uColorAura  * haze * 0.3;

  float alpha = (core + aura * 0.6 + lobe * 0.4 + haze * 0.25)
                * clamp(uMorph, 0.0, 1.0) * 0.16;

  gl_FragColor = vec4(col, alpha);
}
`;

/* ── Shader 2: Secondary ambient haze (warm side glow) ── */
const hazeFragmentShader = /* glsl */ `
uniform float uTime;
uniform float uMorph;
uniform vec3 uColorA;
uniform vec3 uColorB;
varying vec2 vUv;

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float dist = length(p * vec2(1.0, 0.55)); // squish vertically

  float drift = sin(uTime * 0.3 + vUv.x * 2.8) * 0.04;
  float glow  = exp(-(dist + drift) * 2.0) * 0.7;
  float pulse = 0.8 + sin(uTime * 0.5) * 0.12;

  vec3 col = mix(uColorA, uColorB, vUv.x * 0.6 + 0.2);
  float alpha = glow * pulse * clamp(uMorph, 0.0, 1.0) * 0.4;
  gl_FragColor = vec4(col, alpha);
}
`;

export function VolumetricGlow() {
  const primaryRef = useRef<THREE.ShaderMaterial>(null);
  const hazeRef    = useRef<THREE.ShaderMaterial>(null);
  const morph      = useLoaderStore((s) => s.morph);

  const { primary, haze } = useMemo(() => {
    const primary = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: glowVertexShader,
      fragmentShader: glowFragmentShader,
      uniforms: {
        uTime:        { value: 0 },
        uMorph:       { value: 0 },
        uColorCore:   { value: new THREE.Color("#2fbfb0") },
        uColorAura:   { value: new THREE.Color("#143532") },
        uColorAccent3:{ value: new THREE.Color("#1d4a46") },
      },
    });

    const haze = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: glowVertexShader,
      fragmentShader: hazeFragmentShader,
      uniforms: {
        uTime:   { value: 0 },
        uMorph:  { value: 0 },
        uColorA: { value: new THREE.Color("#2fbfb0") },
        uColorB: { value: new THREE.Color("#143532") },
      },
    });

    return { primary, haze };
  }, []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (primaryRef.current) {
      primaryRef.current.uniforms.uTime.value  = t;
      primaryRef.current.uniforms.uMorph.value = morph;
    }
    if (hazeRef.current) {
      hazeRef.current.uniforms.uTime.value  = t;
      hazeRef.current.uniforms.uMorph.value = morph;
    }
  });

  return (
    <>
      {/* Main glow — centre-right behind lattice */}
      <mesh position={[1.1, 0.15, -2.2]} frustumCulled={false}>
        <planeGeometry args={[8, 6, 1, 1]} />
        <primitive object={primary} ref={primaryRef} attach="material" />
      </mesh>
      {/* Secondary warm haze — far left bottom */}
      <mesh position={[-2.2, -1.2, -4.5]} frustumCulled={false}>
        <planeGeometry args={[8, 5, 1, 1]} />
        <primitive object={haze} ref={hazeRef} attach="material" />
      </mesh>
    </>
  );
}

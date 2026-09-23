"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useLoaderStore } from "@/lib/stores/loader-store";

const groundVertexShader = /* glsl */ `
varying vec3 vWorldPos;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPos = worldPos.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const groundFragmentShader = /* glsl */ `
uniform float uTime;
uniform float uMorph;
uniform vec3 uGridColor;
uniform vec3 uAccentColor;
varying vec3 vWorldPos;

float gridLine(vec2 p, float size, float lineWidth) {
  vec2 coord = p / size;
  vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
  float line = min(grid.x, grid.y);
  return 1.0 - min(line * (1.0 / lineWidth), 1.0);
}

void main() {
  vec2 p = vWorldPos.xz;
  float dist = length(p);

  // Radial fade off into the void
  float alphaMask = smoothstep(14.0, 1.5, dist);
  if (alphaMask <= 0.001) discard;

  // Multi-tier grid lines: major and minor
  float minor = gridLine(p, 0.5, 1.2) * 0.35;
  float major = gridLine(p, 2.5, 1.8) * 0.9;
  float grid = max(minor, major);

  // Wave ripple traveling across ground
  float wave = sin(dist * 1.2 - uTime * 1.5) * 0.5 + 0.5;
  vec3 col = mix(uGridColor, uAccentColor, wave * 0.35);

  float finalAlpha = grid * alphaMask * 0.75 * clamp(uMorph, 0.0, 1.0);
  gl_FragColor = vec4(col, finalAlpha);
}
`;

export function GroundGrid() {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const morph = useLoaderStore((s) => s.morph);

  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: groundVertexShader,
      fragmentShader: groundFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uMorph: { value: 0 },
        uGridColor:   { value: new THREE.Color("#1c2228") },
        uAccentColor: { value: new THREE.Color("#2fbfb0") },
      },
    });
  }, []);

  useFrame((state) => {
    if (!materialRef.current) return;
    materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    materialRef.current.uniforms.uMorph.value = morph;
  });

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, -2.6, 0]}
      frustumCulled={false}
    >
      <planeGeometry args={[32, 32, 1, 1]} />
      <primitive object={material} ref={materialRef} attach="material" />
    </mesh>
  );
}


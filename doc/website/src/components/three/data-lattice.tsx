"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three-stdlib";
import {
  latticeFragmentShader,
  latticeVertexShader,
} from "@/components/three/shaders/lattice-shader";
import { useLoaderStore } from "@/lib/stores/loader-store";
import { usePointerStore } from "@/lib/stores/scene-store";
import { useStoryStore } from "@/lib/stores/story-store";
import { useLatticeStore } from "@/lib/stores/lattice-store";

const GRID_DIM = 8;
const CELL_COUNT = GRID_DIM * GRID_DIM * GRID_DIM; // 512 cells
const CELL_SPACING = 0.40;

export function DataLattice() {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const [hoverIndex, setHoverIndex] = useState<number>(-1);

  const reducedMotion = useLoaderStore((s) => s.reducedMotion);
  const { viewport } = useThree();
  const isPortrait = viewport.aspect < 1.1;

  const { geometry, material } = useMemo(() => {
    // 1. Base Rounded Box geometry with subtle bevel
    const geo = new RoundedBoxGeometry(0.20, 0.20, 0.20, 2, 0.035);

    // 2. Custom Instanced Attributes for the 512 cells
    const gridPositions = new Float32Array(CELL_COUNT * 3);
    const cellCoords = new Float32Array(CELL_COUNT * 3);
    const cellSeeds = new Float32Array(CELL_COUNT);
    const cellIndices = new Float32Array(CELL_COUNT);

    const half = (GRID_DIM - 1) * 0.5;
    let idx = 0;

    for (let z = 0; z < GRID_DIM; z++) {
      for (let y = 0; y < GRID_DIM; y++) {
        for (let x = 0; x < GRID_DIM; x++) {
          const pIdx = idx * 3;
          // Grid position centered around (0, 0, 0)
          gridPositions[pIdx] = (x - half) * CELL_SPACING;
          gridPositions[pIdx + 1] = (y - half) * CELL_SPACING;
          gridPositions[pIdx + 2] = (z - half) * CELL_SPACING;

          cellCoords[pIdx] = x;
          cellCoords[pIdx + 1] = y;
          cellCoords[pIdx + 2] = z;

          // Deterministic pseudo-random seed per cell
          cellSeeds[idx] = Math.sin(idx * 12.9898 + 78.233) * 43758.5453 % 1;
          cellIndices[idx] = idx;
          idx++;
        }
      }
    }

    geo.setAttribute(
      "aGridPos",
      new THREE.InstancedBufferAttribute(gridPositions, 3),
    );
    geo.setAttribute(
      "aCellCoord",
      new THREE.InstancedBufferAttribute(cellCoords, 3),
    );
    geo.setAttribute(
      "aCellSeed",
      new THREE.InstancedBufferAttribute(cellSeeds, 1),
    );
    geo.setAttribute(
      "aCellIndex",
      new THREE.InstancedBufferAttribute(cellIndices, 1),
    );

    // 3. Obsidian silicon glass shader with cyan glow & violet rim
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: true,
      vertexShader: latticeVertexShader,
      fragmentShader: latticeFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uMorph: { value: 0 },
        uScene: { value: 0 },
        uReduced: { value: reducedMotion ? 1 : 0 },
        uMouse: { value: new THREE.Vector3(0, 0, 0) },
        uHoverIndex: { value: -1 },
        uShockwaveOrigin: { value: new THREE.Vector3(0, 0, 0) },
        uShockwaveAge: { value: -999 },
        uBatchSize: { value: 8 },
        uDropLast: { value: 1 },
        uSeed: { value: 42 },
        uColorBase: { value: new THREE.Color("#101318") },
        uColorCyan: { value: new THREE.Color("#2fbfb0") },
        uColorViolet: { value: new THREE.Color("#3d4654") },
        uColorAmber: { value: new THREE.Color("#c6a15b") },
      },
    });

    return { geometry: geo, material: mat };
  }, [reducedMotion]);

  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  useFrame((state, delta) => {
    usePointerStore.getState().tick(delta);
    const morph = useLoaderStore.getState().morph;
    const story = useStoryStore.getState();
    const pointer = usePointerStore.getState();
    const lattice = useLatticeStore.getState();
    const mat = materialRef.current ?? material;

    mat.uniforms.uTime.value = state.clock.elapsedTime;
    mat.uniforms.uMorph.value = morph;
    mat.uniforms.uScene.value = story.sceneT;
    mat.uniforms.uSeed.value = story.seed;
    mat.uniforms.uBatchSize.value = story.batchSize;
    mat.uniforms.uDropLast.value = story.dropLast ? 1 : 0;
    mat.uniforms.uReduced.value = useLoaderStore.getState().reducedMotion ? 1 : 0;
    mat.uniforms.uHoverIndex.value = hoverIndex;

    // Pointer repulsion target
    mat.uniforms.uMouse.value.set(pointer.worldX, pointer.worldY, 0.2);

    // Shockwave timing
    if (lattice.shockwaveTriggerTime > 0) {
      const shockAge = performance.now() * 0.001 - lattice.shockwaveTriggerTime;
      mat.uniforms.uShockwaveAge.value = shockAge;
      mat.uniforms.uShockwaveOrigin.value.set(
        lattice.shockwaveOrigin[0],
        lattice.shockwaveOrigin[1],
        lattice.shockwaveOrigin[2],
      );
    }

    // Responsive position & rotation: right-offset on desktop, centred on mobile
    if (groupRef.current) {
      const portrait = viewport.aspect < 1.1;
      const targetPosX = portrait ? 0 : 1.05;
      const targetPosY = portrait ? 0.85 : 0.05;
      const targetScale = portrait ? 0.62 : 0.9;

      groupRef.current.position.x += (targetPosX - groupRef.current.position.x) * 0.08;
      groupRef.current.position.y += (targetPosY - groupRef.current.position.y) * 0.08;

      const curScale = groupRef.current.scale.x;
      const nextScale = curScale + (targetScale - curScale) * 0.08;
      groupRef.current.scale.set(nextScale, nextScale, nextScale);

      if (!useLoaderStore.getState().reducedMotion) {
        groupRef.current.rotation.x = pointer.sy * 0.12;
        groupRef.current.rotation.z = pointer.sx * 0.08;
      }
    }
  });

  return (
    <group
      ref={groupRef}
      position={[isPortrait ? 0 : 1.05, isPortrait ? 0.85 : 0.05, 0]}
      scale={isPortrait ? 0.62 : 0.9}
    >
      <instancedMesh
        ref={meshRef}
        args={[geometry, material, CELL_COUNT]}
        frustumCulled={false}
        onPointerMove={(e) => {
          e.stopPropagation();
          if (typeof e.instanceId === "number") {
            const id = e.instanceId;
            const ix = id % GRID_DIM;
            const iy = Math.floor(id / GRID_DIM) % GRID_DIM;
            const iz = Math.floor(id / (GRID_DIM * GRID_DIM));
            const val = 0.1 + ((ix * 73 + iy * 37 + iz * 19) % 100) / 115;

            useLatticeStore.getState().setHoveredCell({
              index: id,
              x: ix,
              y: iy,
              z: iz,
              value: val,
              screenX: e.clientX,
              screenY: e.clientY,
            });
            setHoverIndex(id);
          }
        }}
        onPointerOut={() => {
          useLatticeStore.getState().setHoveredCell(null);
          setHoverIndex(-1);
        }}
        onClick={(e) => {
          e.stopPropagation();
          if (e.point) {
            useLatticeStore
              .getState()
              .triggerShockwave(e.point.x, e.point.y, e.point.z);
          }
        }}
      >
        <primitive object={material} ref={materialRef} attach="material" />
      </instancedMesh>
    </group>
  );
}

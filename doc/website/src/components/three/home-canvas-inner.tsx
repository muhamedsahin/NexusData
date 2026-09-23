"use client";

import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect } from "react";
import { DataLattice } from "@/components/three/data-lattice";
import { Starfield } from "@/components/three/starfield";
import { GroundGrid } from "@/components/three/ground-grid";
import { VolumetricGlow } from "@/components/three/volumetric-glow";
import { PostFX } from "@/components/three/post-fx";
import {
  AssetProgressBridge,
  CameraRig,
  QualityController,
  SceneReadySignal,
} from "@/components/three/scene-controllers";
import {
  initialQualityForDevice,
  useQualityStore,
} from "@/lib/stores/quality-store";
const STAGE_BG = "#090a0c";

function SceneContent() {
  return (
    <>
      <color attach="background" args={["#090a0c"]} />
      <ambientLight intensity={0.42} color="#f3eee6" />
      <directionalLight position={[5, 8, 6]} intensity={1.15} color="#f3eee6" />
      <directionalLight position={[-5, 2, 4]} intensity={0.55} color="#2fbfb0" />
      <VolumetricGlow />
      <GroundGrid />
      <Starfield />
      <DataLattice />
      <CameraRig />
      <QualityController />
      <SceneReadySignal />
      <AssetProgressBridge />
      <PostFX />
    </>
  );
}

export function HomeCanvasInner() {
  const dprCap = useQualityStore((s) => s.dprCap);

  useEffect(() => {
    useQualityStore.getState().setLevel(initialQualityForDevice());
  }, []);

  const bg = STAGE_BG;

  return (
    <Canvas
      aria-hidden
      role="presentation"
      dpr={[1, dprCap]}
      frameloop="always"
      gl={{
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
        stencil: false,
      }}
      camera={{ position: [-0.15, 0.22, 7.4], fov: 38, near: 0.1, far: 80 }}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        background: bg,
      }}
      onCreated={({ gl }) => {
        gl.setClearColor(bg, 1);
      }}
    >
      <Suspense fallback={null}>
        <SceneContent />
      </Suspense>
    </Canvas>
  );
}

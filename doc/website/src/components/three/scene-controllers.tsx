"use client";

import { useEffect, useRef } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { PerformanceMonitor } from "@react-three/drei";
import * as THREE from "three";
import {
  DESKTOP_HERO_PATH,
  PORTRAIT_HERO_PATH,
  sampleCameraPath,
} from "@/lib/camera-path";
import { useQualityStore } from "@/lib/stores/quality-store";
import { useLoaderStore } from "@/lib/stores/loader-store";
import { usePointerStore, useScrollStore } from "@/lib/stores/scene-store";

export function QualityController() {
  const setLevel = useQualityStore((s) => s.setLevel);
  const degrade = useQualityStore((s) => s.degrade);
  const dprCap = useQualityStore((s) => s.dprCap);
  const setDpr = useThree((s) => s.setDpr);

  useEffect(() => {
    setDpr(Math.min(window.devicePixelRatio, dprCap));
  }, [dprCap, setDpr]);

  return (
    <PerformanceMonitor
      onDecline={() => degrade()}
      onIncline={() => {
        const level = useQualityStore.getState().level;
        if (level === "low") setLevel("medium");
        else if (level === "medium") setLevel("high");
      }}
      flipflops={3}
      factor={0.8}
    />
  );
}

export function SceneReadySignal() {
  const marked = useRef(false);
  useFrame(() => {
    if (marked.current) return;
    marked.current = true;
    const store = useLoaderStore.getState();
    store.markSceneReady();
    store.markAssetProgress(1);
  });
  return null;
}

const desiredPosition = new THREE.Vector3();
const desiredLook = new THREE.Vector3();

export function CameraRig() {
  const camera = useThree((s) => s.camera);
  const aspect = useThree((s) => s.viewport.aspect);
  const width = useThree((s) => s.size.width);
  const lookTarget = useRef(new THREE.Vector3(0.05, 0.02, 0));

  useFrame((_, delta) => {
    const { sx, sy } = usePointerStore.getState();
    const progress = useScrollStore.getState().progress;
    const reduced = useLoaderStore.getState().reducedMotion;
    const portrait = aspect < 1.05 || width < 1024;
    const pose = sampleCameraPath(
      progress,
      portrait ? PORTRAIT_HERO_PATH : DESKTOP_HERO_PATH,
    );
    const parallax = reduced ? 0 : portrait ? 0.08 : 0.14;
    desiredPosition.set(
      pose.position[0] + sx * parallax,
      pose.position[1] + sy * parallax * 0.55,
      pose.position[2],
    );
    desiredLook.set(pose.target[0], pose.target[1], pose.target[2]);
    const k = 1 - Math.exp(-delta * (reduced ? 14 : 3.4));
    camera.position.lerp(desiredPosition, k);
    lookTarget.current.lerp(desiredLook, k);
    camera.lookAt(lookTarget.current);
  });

  return null;
}

export function AssetProgressBridge() {
  useEffect(() => {
    const started = performance.now();
    let frame = 0;
    const tick = () => {
      const elapsed = performance.now() - started;
      const soft = Math.min(0.95, elapsed / 900);
      const store = useLoaderStore.getState();
      if (
        !store.assetsReady &&
        store.phase !== "ready" &&
        store.phase !== "skipped"
      ) {
        store.markAssetProgress(Math.max(soft, store.progress / 100));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);
  return null;
}

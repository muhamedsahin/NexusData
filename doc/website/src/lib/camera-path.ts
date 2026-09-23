export type Vec3 = [number, number, number];

export type CameraKeyframe = {
  /** Normalized page scroll, 0–1. */
  progress: number;
  position: Vec3;
  target: Vec3;
};

/**
 * One controlled move: the lattice holds the right side of the hero,
 * then the camera rises and yaws as the sources section arrives.
 * Later scroll holds the arrival pose until more beats are keyed.
 */
export const DESKTOP_HERO_PATH: CameraKeyframe[] = [
  {
    progress: 0,
    position: [-0.15, 0.22, 7.4],
    target: [0.05, 0.02, 0],
  },
  {
    progress: 0.18,
    position: [-0.4, 0.58, 7.95],
    target: [0.2, 0.1, 0],
  },
];

/** Narrow viewports: lattice sits in the upper frame, copy on a solid field below. */
export const PORTRAIT_HERO_PATH: CameraKeyframe[] = [
  {
    progress: 0,
    position: [0, 0.05, 8.7],
    target: [0, -1.45, 0],
  },
  {
    progress: 0.18,
    position: [0, 0.75, 9.1],
    target: [0, -0.35, 0],
  },
];

function smoothstep(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

export function sampleCameraPath(
  progress: number,
  keys: CameraKeyframe[],
): { position: Vec3; target: Vec3 } {
  if (keys.length === 0) {
    return { position: [0, 0, 6], target: [0, 0, 0] };
  }
  const first = keys[0]!;
  const last = keys[keys.length - 1]!;
  if (progress <= first.progress) {
    return { position: first.position, target: first.target };
  }
  if (progress >= last.progress) {
    return { position: last.position, target: last.target };
  }
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i]!;
    const b = keys[i + 1]!;
    if (progress >= a.progress && progress <= b.progress) {
      const span = b.progress - a.progress;
      const t = smoothstep(span <= 0 ? 0 : (progress - a.progress) / span);
      return {
        position: lerp3(a.position, b.position, t),
        target: lerp3(a.target, b.target, t),
      };
    }
  }
  return { position: last.position, target: last.target };
}

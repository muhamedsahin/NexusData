import { describe, expect, it } from "vitest";
import {
  DESKTOP_HERO_PATH,
  sampleCameraPath,
} from "@/lib/camera-path";

describe("sampleCameraPath", () => {
  it("holds the hero pose before the transition", () => {
    const pose = sampleCameraPath(0, DESKTOP_HERO_PATH);
    expect(pose.position).toEqual(DESKTOP_HERO_PATH[0]!.position);
    expect(pose.target).toEqual(DESKTOP_HERO_PATH[0]!.target);
  });

  it("holds the arrival pose after the single transition", () => {
    const pose = sampleCameraPath(0.9, DESKTOP_HERO_PATH);
    const last = DESKTOP_HERO_PATH[DESKTOP_HERO_PATH.length - 1]!;
    expect(pose.position).toEqual(last.position);
    expect(pose.target).toEqual(last.target);
  });

  it("eases through the midpoint of the hero move", () => {
    const pose = sampleCameraPath(0.09, DESKTOP_HERO_PATH);
    const a = DESKTOP_HERO_PATH[0]!.position;
    const b = DESKTOP_HERO_PATH[1]!.position;
    expect(pose.position[1]).toBeCloseTo((a[1]! + b[1]!) / 2, 5);
    expect(pose.position[2]).toBeGreaterThan(a[2]!);
    expect(pose.position[2]).toBeLessThan(b[2]!);
  });
});

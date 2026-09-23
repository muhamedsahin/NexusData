export const particleVertexShader = /* glsl */ `
uniform float uTime;
uniform float uMorph;
uniform float uScene;
uniform float uReduced;
uniform vec3 uMouse;
uniform float uForce;
uniform float uSeed;
uniform float uBatchSize;
uniform float uDropLast;

attribute vec3 aCube;
attribute vec3 aGrid;
attribute float aSeed;
attribute float aIndex;

varying float vGlow;
varying float vDepth;
varying float vBatch;

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec3 forceField(vec3 p, vec3 mouse) {
  vec3 d = p - mouse;
  float dist = length(d) + 0.001;
  float strength = uForce / (dist * dist + 0.35);
  return normalize(d) * strength * 0.35;
}

vec3 sourcesPos(vec3 grid, float idx) {
  float orbit = idx / 48.0 * 6.2831;
  float radius = 2.2 + hash11(idx + uSeed) * 0.8;
  float y = (hash11(idx * 1.7 + uSeed) - 0.5) * 2.4;
  vec3 orbitP = vec3(cos(orbit + uTime * 0.25) * radius, y, sin(orbit + uTime * 0.25) * radius);
  float funnel = smoothstep(1.0, 3.0, uScene);
  vec3 funnelP = mix(orbitP, vec3(0.0, -1.8, 0.0) + grid * 0.15, funnel * 0.65);
  return mix(grid, funnelP, smoothstep(0.8, 2.0, uScene));
}

vec3 shufflePos(vec3 grid, float idx) {
  float h = hash11(idx * 13.1 + floor(uSeed));
  float h2 = hash11(idx * 7.3 + floor(uSeed) * 1.9);
  float h3 = hash11(idx * 3.1 + floor(uSeed) * 0.7);
  vec3 scattered = vec3((h - 0.5) * 5.0, (h2 - 0.5) * 3.2, (h3 - 0.5) * 4.0);
  return mix(grid, scattered, smoothstep(1.6, 2.6, uScene));
}

vec3 batchPos(vec3 grid, float idx) {
  float bs = max(1.0, uBatchSize);
  float batchId = floor(idx / bs);
  float slot = mod(idx, bs);
  float total = 48.0;
  float lastStart = floor((total - 0.001) / bs) * bs;
  float isDropped = step(lastStart, idx) * step(mod(total, bs), bs - 0.5) * uDropLast;
  vec3 lane = vec3(
    (slot - bs * 0.5) * 0.28,
    1.2 - batchId * 0.45,
    -batchId * 0.15
  );
  lane.x += isDropped * 3.5;
  lane.y -= isDropped * 0.8;
  float gate = smoothstep(2.6, 3.4, uScene);
  vec3 throughGate = lane + vec3(0.0, 0.0, gate * 2.5);
  return mix(shufflePos(grid, idx), throughGate, smoothstep(2.4, 3.2, uScene));
}

vec3 gpuPos(vec3 grid, float idx) {
  float ring = floor(idx / 16.0);
  float slot = mod(idx, 16.0);
  vec3 chip = vec3((slot - 7.5) * 0.22, (ring - 1.0) * 0.35, 0.2);
  float pulse = sin(uTime * 2.0 + ring) * 0.05;
  chip.z += pulse;
  return mix(batchPos(grid, idx), chip, smoothstep(3.6, 4.4, uScene));
}

vec3 ecoPos(vec3 grid, float idx) {
  float node = mod(floor(idx / 7.0), 7.0);
  float ang = node / 7.0 * 6.2831 + uTime * 0.08;
  float rad = 2.6;
  vec3 center = vec3(cos(ang) * rad, sin(ang) * rad * 0.55, sin(ang * 0.5) * 0.4);
  vec3 local = grid * 0.08;
  return mix(gpuPos(grid, idx), center + local, smoothstep(4.6, 5.5, uScene));
}

vec3 ctaPos(vec3 grid, float idx) {
  return mix(ecoPos(grid, idx), grid, smoothstep(8.4, 9.2, uScene));
}

void main() {
  vec3 grid = aGrid;
  vec3 base = mix(aCube, grid, clamp(uMorph, 0.0, 1.0));

  if (uMorph > 0.95) {
    base = ctaPos(grid, aIndex);
    // Ease earlier scenes when scrolling back
    if (uScene < 1.2) base = mix(grid, base, smoothstep(0.0, 1.2, uScene));
  }

  float breathe = sin(uTime * 0.6 + aSeed * 6.2831) * 0.04 * (1.0 - uReduced);
  base += normalize(base + 0.0001) * breathe;

  if (uReduced < 0.5) {
    base += forceField(base, uMouse);
  }

  vec4 mv = modelViewMatrix * vec4(base, 1.0);
  vDepth = -mv.z;
  vGlow = mix(0.55, 1.0, aSeed);
  vBatch = floor(aIndex / max(1.0, uBatchSize));
  gl_Position = projectionMatrix * mv;
  float size = mix(3.5, 2.2, uMorph) * (1.0 + aSeed * 0.5);
  gl_PointSize = size * (300.0 / max(60.0, -mv.z));
}
`;

export const particleFragmentShader = /* glsl */ `
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform float uMorph;
uniform float uScene;

varying float vGlow;
varying float vDepth;
varying float vBatch;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float d = dot(uv, uv);
  if (d > 1.0) discard;
  float alpha = smoothstep(1.0, 0.1, d) * mix(0.85, 0.55, uMorph);
  vec3 col = mix(uColorA, uColorB, fract(vBatch * 0.17 + vGlow));
  float fog = smoothstep(18.0, 4.0, vDepth);
  float ecoBoost = smoothstep(4.5, 5.5, uScene) * 0.15;
  gl_FragColor = vec4(col + ecoBoost, alpha * fog);
}
`;

export const starVertexShader = /* glsl */ `
uniform float uTime;
uniform float uReduced;
attribute float aSeed;
varying float vAlpha;

void main() {
  vec3 p = position;
  if (uReduced < 0.5) {
    p.z += sin(uTime * 0.05 + aSeed * 20.0) * 0.15;
  }
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vAlpha = mix(0.15, 0.55, aSeed);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = mix(1.0, 2.2, aSeed) * (180.0 / max(40.0, -mv.z));
}
`;

export const starFragmentShader = /* glsl */ `
varying float vAlpha;
void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float d = dot(uv, uv);
  if (d > 1.0) discard;
  float a = smoothstep(1.0, 0.0, d) * vAlpha;
  gl_FragColor = vec4(0.75, 0.85, 1.0, a);
}
`;

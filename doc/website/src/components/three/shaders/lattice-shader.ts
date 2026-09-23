export const latticeVertexShader = /* glsl */ `
uniform float uTime;
uniform float uMorph;
uniform float uScene;
uniform float uReduced;
uniform vec3 uMouse;
uniform float uHoverIndex;
uniform vec3 uShockwaveOrigin;
uniform float uShockwaveAge;
uniform float uBatchSize;
uniform float uDropLast;
uniform float uSeed;

attribute vec3 aGridPos;
attribute vec3 aCellCoord;
attribute float aCellSeed;
attribute float aCellIndex;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying float vHover;
varying float vShock;
varying float vBreathe;
varying float vCellSeed;
varying float vBatch;

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

// Scene 1: 3 Columnar Source Clusters (Parquet, Arrow, S3)
vec3 computeSourcesPos(vec3 grid, float idx) {
  float cluster = mod(idx, 3.0);
  vec3 center = vec3((cluster - 1.0) * 2.2, 0.2, 0.0);
  float ang = idx * 0.35 + uTime * 0.2;
  float rad = 0.4 + hash11(idx) * 0.55;
  vec3 swirl = vec3(cos(ang) * rad, sin(ang) * rad, (hash11(idx * 2.1) - 0.5) * 1.2);
  return center + swirl;
}

// Scene 2: Deterministic Shuffle Scatter
vec3 computeShufflePos(vec3 grid, float idx) {
  float h1 = hash11(idx * 17.1 + floor(uSeed) * 3.7);
  float h2 = hash11(idx * 31.3 + floor(uSeed) * 5.9);
  float h3 = hash11(idx * 7.7 + floor(uSeed) * 2.3);
  return vec3((h1 - 0.5) * 4.4, (h2 - 0.5) * 3.2, (h3 - 0.5) * 3.6);
}

// Scene 3: Batch Lanes & Drop_Last Outlier
vec3 computeBatchPos(vec3 grid, float idx) {
  float bs = max(1.0, uBatchSize);
  float batchId = floor(idx / bs);
  float slot = mod(idx, bs);
  float total = 512.0;
  float lastStart = floor((total - 0.001) / bs) * bs;
  float isDropped = step(lastStart, idx) * step(mod(total, bs), bs - 0.5) * uDropLast;

  vec3 lane = vec3(
    (slot - bs * 0.5) * 0.24,
    1.4 - mod(batchId, 8.0) * 0.42,
    -floor(batchId / 8.0) * 0.35
  );
  lane.x += isDropped * 3.2;
  lane.y -= isDropped * 0.7;
  return lane;
}

// Scene 4: GPU Tensor Register Grid
vec3 computeGpuPos(vec3 grid, float idx) {
  float col = mod(idx, 32.0);
  float row = floor(idx / 32.0);
  vec3 reg = vec3((col - 15.5) * 0.12, (row - 7.5) * 0.22, 0.0);
  float pulse = sin(uTime * 3.0 + row * 0.5) * 0.06;
  reg.z += pulse;
  return reg;
}

// Scene 5: Ecosystem Orbit
vec3 computeEcoPos(vec3 grid, float idx) {
  float ring = mod(floor(idx / 64.0), 8.0);
  float ang = idx * 0.12 + uTime * (0.05 + ring * 0.02);
  float rad = 1.8 + ring * 0.35;
  return vec3(cos(ang) * rad, sin(ang) * rad * 0.6, sin(ang * 0.5) * 0.4);
}

void main() {
  // Base 8x8x8 Lattice coordinate
  vec3 base = aGridPos;

  // Diagonal breathing wave in Hero
  float diag = (aCellCoord.x + aCellCoord.y + aCellCoord.z);
  float wavePhase = diag * 0.42 - uTime * 1.6;
  float breathe = sin(wavePhase) * (1.0 - uReduced);
  vBreathe = breathe;

  if (uScene < 0.1) {
    base.y += breathe * 0.07;
  } else {
    // Smooth transitions between scenes
    vec3 s1 = computeSourcesPos(base, aCellIndex);
    vec3 s2 = computeShufflePos(base, aCellIndex);
    vec3 s3 = computeBatchPos(base, aCellIndex);
    vec3 s4 = computeGpuPos(base, aCellIndex);
    vec3 s5 = computeEcoPos(base, aCellIndex);

    vec3 morphTarget = base;
    if (uScene < 1.5) morphTarget = mix(base, s1, smoothstep(0.0, 1.2, uScene));
    else if (uScene < 2.5) morphTarget = mix(s1, s2, smoothstep(1.5, 2.3, uScene));
    else if (uScene < 3.5) morphTarget = mix(s2, s3, smoothstep(2.5, 3.3, uScene));
    else if (uScene < 4.5) morphTarget = mix(s3, s4, smoothstep(3.5, 4.3, uScene));
    else morphTarget = mix(s4, s5, smoothstep(4.5, 5.5, uScene));

    base = morphTarget;
  }

  // Interactive mouse repulsion in Hero
  if (uReduced < 0.5 && uScene < 1.0) {
    vec3 toMouse = base - uMouse;
    float mouseDist = length(toMouse) + 0.001;
    float repulse = 0.35 / (mouseDist * mouseDist + 0.45);
    base += normalize(toMouse) * min(repulse, 0.40);
  }

  // Click shockwave ring
  float shock = 0.0;
  if (uShockwaveAge >= 0.0 && uShockwaveAge < 2.5) {
    float shockDist = length(base - uShockwaveOrigin);
    float waveRadius = uShockwaveAge * 4.2;
    float waveDelta = abs(shockDist - waveRadius);
    float ring = exp(-pow(waveDelta / 0.32, 2.0));
    float decay = max(0.0, 1.0 - uShockwaveAge * 0.4);
    shock = ring * decay;
    base += normalize(base - uShockwaveOrigin + vec3(0.001)) * shock * 0.30;
  }
  vShock = shock;

  // Hover detection
  float isHovered = (abs(aCellIndex - uHoverIndex) < 0.5) ? 1.0 : 0.0;
  vHover = isHovered;

  // Dynamic cell scale
  float baseScale = mix(0.96, 1.18, isHovered);
  baseScale *= (1.0 + breathe * 0.08 + shock * 0.22);

  // Transform local vertex
  vec3 transformed = position * baseScale;
  vec4 worldInstance = modelMatrix * vec4(base + transformed, 1.0);
  vWorldPos = worldInstance.xyz;

  vec3 worldNorm = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
  vNormal = worldNorm;
  vViewDir = normalize(cameraPosition - vWorldPos);

  vCellSeed = aCellSeed;
  vBatch = floor(aCellIndex / max(1.0, uBatchSize));

  gl_Position = projectionMatrix * viewMatrix * worldInstance;
}
`;

export const latticeFragmentShader = /* glsl */ `
uniform vec3 uColorBase;
uniform vec3 uColorCyan;
uniform vec3 uColorViolet;
uniform vec3 uColorAmber;
uniform float uScene;
uniform float uTime;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying float vHover;
varying float vShock;
varying float vBreathe;
varying float vCellSeed;
varying float vBatch;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);

  float NdotV = max(0.0, dot(N, V));

  // 1. Triple specular highlights — key (cyan), fill (violet), accent (ember)
  vec3 L1 = normalize(vec3(0.5, 0.8, 0.6));
  vec3 H1 = normalize(L1 + V);
  float spec1 = pow(max(0.0, dot(N, H1)), 22.0) * 1.2;

  vec3 L2 = normalize(vec3(-0.6, -0.2, -0.5));
  vec3 H2 = normalize(L2 + V);
  float spec2 = pow(max(0.0, dot(N, H2)), 14.0) * 0.65;

  vec3 L3 = normalize(vec3(1.0, 0.0, 0.3));
  vec3 H3 = normalize(L3 + V);
  float spec3 = pow(max(0.0, dot(N, H3)), 32.0) * 0.45;

  // 2. Dark glass base
  vec3 baseCol = vec3(0.012, 0.02, 0.04);

  // 3. Violet Fresnel Rim (wider, brighter)
  float fresnel = pow(1.0 - NdotV, 2.8);
  vec3 rimViolet = uColorViolet * (fresnel * 2.2);

  // 4. Electric Cyan Edge & Core (stronger)
  float edgeCyan = pow(1.0 - NdotV, 4.5) * 1.4;
  float internalPulse = pow(NdotV, 2.0) * (0.14 + vBreathe * 0.08 + vCellSeed * 0.06);
  vec3 cyanAccents = uColorCyan * (edgeCyan + internalPulse);

  // 5. Amber/ember side lobe (from right)
  float amberLobe = pow(max(0.0, dot(N, normalize(vec3(1.0, 0.0, 0.3)))), 6.0) * 0.55 * fresnel;
  vec3 amberTint = uColorAmber * amberLobe;

  // 6. Combine
  vec3 specular = uColorCyan * spec1 * 0.9 + uColorViolet * spec2 + uColorAmber * spec3;
  vec3 finalColor = baseCol + rimViolet + cyanAccents + amberTint + specular;

  // 7. Click Shockwave Flash (intense)
  finalColor += uColorCyan * (vShock * 3.5);
  finalColor += uColorAmber * (vShock * 1.2);

  // 8. Hover Highlight
  if (vHover > 0.5) {
    finalColor = mix(finalColor, uColorCyan * 3.2 + vec3(0.1, 0.6, 0.7), 0.92);
  }

  // 9. Batch coloring hint (subtle violet tint per batch)
  float batchHue = fract(vBatch * 0.137);
  finalColor += uColorViolet * batchHue * 0.08;

  // Soft depth fog
  float depth = length(vWorldPos - cameraPosition);
  float fog = smoothstep(22.0, 4.0, depth);

  gl_FragColor = vec4(finalColor, fog);
}
`;


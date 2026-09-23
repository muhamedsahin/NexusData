import * as THREE from "three";

const GRID = 12;

export function buildParticleGeometry(countScale = 1): {
  geometry: THREE.BufferGeometry;
  count: number;
} {
  const dim = Math.max(6, Math.round(GRID * Math.sqrt(countScale)));
  const count = dim * dim * dim;
  const cube = new Float32Array(count * 3);
  const grid = new Float32Array(count * 3);
  const seeds = new Float32Array(count);
  const indices = new Float32Array(count);

  const half = (dim - 1) * 0.5;
  const spacing = 0.38;
  let i = 0;
  for (let z = 0; z < dim; z++) {
    for (let y = 0; y < dim; y++) {
      for (let x = 0; x < dim; x++) {
        const idx = i * 3;
        const gx = (x - half) * spacing;
        const gy = (y - half) * spacing;
        const gz = (z - half) * spacing;
        grid[idx] = gx;
        grid[idx + 1] = gy;
        grid[idx + 2] = gz;

        // Map to wireframe-ish cube shell for loading morph source
        const onShell =
          x === 0 ||
          y === 0 ||
          z === 0 ||
          x === dim - 1 ||
          y === dim - 1 ||
          z === dim - 1;
        const edge = 1.55;
        if (onShell) {
          cube[idx] = ((x / (dim - 1)) * 2 - 1) * edge;
          cube[idx + 1] = ((y / (dim - 1)) * 2 - 1) * edge;
          cube[idx + 2] = ((z / (dim - 1)) * 2 - 1) * edge;
        } else {
          // Interior points start near cube center and explode outward
          const n = Math.random();
          cube[idx] = (Math.random() * 2 - 1) * 0.2 * n;
          cube[idx + 1] = (Math.random() * 2 - 1) * 0.2 * n;
          cube[idx + 2] = (Math.random() * 2 - 1) * 0.2 * n;
        }
        seeds[i] = Math.random();
        indices[i] = i % 48;
        i++;
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(grid, 3));
  geometry.setAttribute("aCube", new THREE.BufferAttribute(cube, 3));
  geometry.setAttribute("aGrid", new THREE.BufferAttribute(grid, 3));
  geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
  geometry.setAttribute("aIndex", new THREE.BufferAttribute(indices, 1));
  return { geometry, count };
}

export function buildStarGeometry(count: number): THREE.BufferGeometry {
  const positions = new Float32Array(count * 3);
  const seeds = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const r = 8 + Math.random() * 22;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi) - 6;
    seeds[i] = Math.random();
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
  return geometry;
}

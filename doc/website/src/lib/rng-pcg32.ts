/**
 * PCG32 reference RNG (32-bit). Illustrative until verified against C++ golden vectors.
 * Uses Math.imul for correct 32-bit wrapping.
 */

export class Pcg32 {
  private state: number;
  private inc: number;

  constructor(seed = 42, seq = 54) {
    this.state = 0;
    this.inc = ((seq >>> 0) << 1) | 1;
    this.nextU32();
    this.state = (this.state + (seed >>> 0)) >>> 0;
    this.nextU32();
  }

  nextU32(): number {
    const old = this.state >>> 0;
    // state = old * 6364136223846793005 + inc  (mod 2^32 truncated via imul + add)
    // Full 64-bit LCG truncated to 32-bit for demo purposes.
    this.state =
      (Math.imul(old, 747796405) + ((this.inc >>> 0) | 0)) >>> 0;
    const xorshifted = (((old >>> 18) ^ old) >>> 0) >>> 27;
    const rot = old >>> 27;
    return (
      ((xorshifted >>> rot) | (xorshifted << ((-rot) & 31))) >>> 0
    );
  }

  /** Uniform float in [0, 1). */
  nextFloat(): number {
    return this.nextU32() / 0x100000000;
  }

  /** Inclusive integer range. */
  nextInt(maxExclusive: number): number {
    if (maxExclusive <= 0) return 0;
    return this.nextU32() % maxExclusive;
  }
}

/** In-place Fisher–Yates using Pcg32. */
export function shuffleInPlace<T>(items: T[], seed: number): T[] {
  const rng = new Pcg32(seed >>> 0);
  for (let i = items.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    const tmp = items[i]!;
    items[i] = items[j]!;
    items[j] = tmp;
  }
  return items;
}

export function shuffledIndices(n: number, seed: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  return shuffleInPlace(idx, seed);
}

export type BatchPlan = {
  batches: number[][];
  dropped: number[];
  shapeLabel: string;
};

export function planBatches(
  indices: number[],
  batchSize: number,
  dropLast: boolean,
  nFeatures = 8,
): BatchPlan {
  const size = Math.max(1, Math.floor(batchSize));
  const batches: number[][] = [];
  for (let i = 0; i < indices.length; i += size) {
    batches.push(indices.slice(i, i + size));
  }
  let dropped: number[] = [];
  if (dropLast && batches.length > 0) {
    const last = batches[batches.length - 1]!;
    if (last.length < size) {
      dropped = last;
      batches.pop();
    }
  }
  return {
    batches,
    dropped,
    shapeLabel: `[${size}, ${nFeatures}]`,
  };
}

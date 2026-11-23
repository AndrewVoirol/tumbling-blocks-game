/**
 * contactEquilibrium.ts — Coordinate-Free 3D Contact-Hull Equilibrium, Rounded-Bevel Escape,
 * Oriented-OBB Support & Quasi-Static Brittle Glass Yield
 *
 * Pure, zero-GC physical domain laws governing all 360-degree orientations across standing towers
 * (1L..36L+), post-collapse rubble heaps, and Sandbox block-on-shard builds:
 * - Law 1A: All-orientation polygon-chamfer knife-edge release (59-deg longitudinal, 79-deg upright-end,
 *           8-corner & block-on-block saddles) when tilted > 10 deg with all-contact 2D hull width < 2.4 mm.
 * - Law 1B: Coordinate-free 360-deg cantilever & true rounded-bevel lip escape (L_tail <= 6.5 mm or
 *           nose-down bevel slip >= 7.5 deg), preserving counterweighted corbels & suspenseful slot hangs.
 * - Law 1C: Quasi-static concentrated-load Griffith edge crazing & fracture at v ~ 0 on narrow (< 4 mm)
 *           overloaded tall-tower corner/edge slivers and extreme three-point spans (>= 5x margin above
 *           centered 36L '+' crosses).
 */

import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';
import type { MujocoTowerIndexMap } from './mujocoTowerBuilder';

export const EQ_BLOCK_HALF_X_M = 0.0375;
export const EQ_BLOCK_HALF_Y_M = 0.0075;
export const EQ_BLOCK_HALF_Z_M = 0.0125;
export const EQ_BEVEL_RADIUS_M = 0.00125;

export const EQ_BLOCK_HALF_X_DM = 0.375;
export const EQ_BLOCK_HALF_Y_DM = 0.075;
export const EQ_BLOCK_HALF_Z_DM = 0.125;

export const KNIFE_EDGE_MAX_AXIS_ALIGN_Y = 0.985; // cos(9.94 deg)
export const KNIFE_EDGE_MIN_HULL_WIDTH_M = 0.0024; // 2.4 mm
export const KNIFE_EDGE_MAX_OMEGA_SI = 0.85; // rad/s

export const BEVEL_UNROOFED_MIN_OVERHANG_M = 0.0010; // 1.0 mm
export const BEVEL_ESCAPE_TAIL_LIMIT_M = 0.0065; // 6.5 mm remaining inboard tail
export const BEVEL_SLIP_TILT_LIMIT_DEG = 7.5; // deg nose-down outward pitch
export const MU_GLASS_BEVEL_SLIP = 0.080;
export const MU_SLOT_PINCH_ESCAPE = 0.018;
export const MAX_CAM_SLIP_SPEED_SI = 0.12; // m/s

export const QUASI_STATIC_MIN_CLAMP_N = 16.0; // N (>= 23 blocks squeezed through thickness)
export const QUASI_STATIC_NARROW_WIDTH_M = 0.0040; // 4.0 mm edge/sliver minor width
export const QUASI_STATIC_NARROW_AREA_M2 = 4.8e-5; // 48 mm^2 corner patch
export const QUASI_STATIC_CRAZE_STRESS_MPA = 1.10; // MPa (> 8x above 0.135 MPa centered 36L '+' cross)
export const QUASI_STATIC_CRAZED_FRACTURE_MPA = 1.65; // MPa crazed edge spall limit
export const QUASI_STATIC_INSTANT_FRACTURE_MPA = 3.40; // MPa extreme razor-point crush

export const MAX_CONTACTS_PER_BLOCK = 32;
export const CONTACT_STRIDE = 10;
const INTERNAL_STRIDE = 14; // px, py, pz, dist, fn, nx, ny, nz, fx, fy, fz, otherGeomId, otherIsShard, otherIsTable

export interface ContactHull2DMetrics {
  pointCount: number;
  hullCount: number;
  minorWidthM: number;
  majorSpanM: number;
  areaM2: number;
  effectiveAreaM2: number;
  stripTanX: number;
  stripTanZ: number;
  stripCenterX: number;
  stripCenterZ: number;
  dOutM: number;
  fulcrumX: number;
  fulcrumZ: number;
  outUnitX: number;
  outUnitZ: number;
}

export interface BlockEquilibriumAction {
  isUnbalanced: boolean;
  isKnifeEdgeRelease: boolean;
  isCantileverRelease: boolean;
  overrideFrictionMu: number; // 0 if no override, else 0.018 or 0.080
  tauWorldX: number;
  tauWorldZ: number;
  forceWorldX: number;
  forceWorldZ: number;
  outUnitX: number;
  outUnitZ: number;
  allHullMinorWidthM: number;
  upwardForceN: number;
  downwardRoofForceN: number;
  dOutM: number;
  lTailM: number;
  tiltOutDeg: number;
  tauRestoreNm: number;
  tauGravityNm: number;
}

export interface BlockQuasiStaticYieldResult {
  overloaded: boolean;
  shouldCraze: boolean;
  shouldFracture: boolean;
  effectiveStressMpa: number;
  edgeBearingStressMpa: number;
  flexuralBendingStressMpa: number;
  targetDamage: number;
  localOriginDmX: number;
  localOriginDmY: number;
  localOriginDmZ: number;
  worldHitDmX: number;
  worldHitDmY: number;
  worldHitDmZ: number;
  worldNormalX: number;
  worldNormalY: number;
  worldNormalZ: number;
}

export interface GateBBlockAudit {
  pass: boolean;
  reason: string;
  upwardSupportN: number;
  hasTableContact: boolean;
  maxAxisAlignY: number;
  allHullMinorWidthMm: number;
  dOutMm: number;
  lTailMm: number;
  tiltOutDeg: number;
  tauRestoreNm: number;
  tauGravityNm: number;
}

// Pre-allocated zero-GC module scratch buffers
const scratchPtsX = new Float64Array(64);
const scratchPtsZ = new Float64Array(64);
const scratchOrder = new Int32Array(64);
const scratchHullIdx = new Int32Array(128);
const scratchHullX = new Float64Array(64);
const scratchHullZ = new Float64Array(64);

const scratchUpPtsX = new Float64Array(64);
const scratchUpPtsZ = new Float64Array(64);
const scratchDnPtsX = new Float64Array(64);
const scratchDnPtsZ = new Float64Array(64);

export const scratchAllHullMetrics: ContactHull2DMetrics = {
  pointCount: 0,
  hullCount: 0,
  minorWidthM: 0,
  majorSpanM: 0,
  areaM2: 0,
  effectiveAreaM2: 2.0e-6,
  stripTanX: 1,
  stripTanZ: 0,
  stripCenterX: 0,
  stripCenterZ: 0,
  dOutM: 0,
  fulcrumX: 0,
  fulcrumZ: 0,
  outUnitX: 1,
  outUnitZ: 0,
};

export const scratchUpHullMetrics: ContactHull2DMetrics = {
  pointCount: 0,
  hullCount: 0,
  minorWidthM: 0,
  majorSpanM: 0,
  areaM2: 0,
  effectiveAreaM2: 2.0e-6,
  stripTanX: 1,
  stripTanZ: 0,
  stripCenterX: 0,
  stripCenterZ: 0,
  dOutM: 0,
  fulcrumX: 0,
  fulcrumZ: 0,
  outUnitX: 1,
  outUnitZ: 0,
};

export const scratchDnHullMetrics: ContactHull2DMetrics = {
  pointCount: 0,
  hullCount: 0,
  minorWidthM: 0,
  majorSpanM: 0,
  areaM2: 0,
  effectiveAreaM2: 2.0e-6,
  stripTanX: 1,
  stripTanZ: 0,
  stripCenterX: 0,
  stripCenterZ: 0,
  dOutM: 0,
  fulcrumX: 0,
  fulcrumZ: 0,
  outUnitX: 1,
  outUnitZ: 0,
};

export const scratchEquilibriumAction: BlockEquilibriumAction = {
  isUnbalanced: false,
  isKnifeEdgeRelease: false,
  isCantileverRelease: false,
  overrideFrictionMu: 0,
  tauWorldX: 0,
  tauWorldZ: 0,
  forceWorldX: 0,
  forceWorldZ: 0,
  outUnitX: 1,
  outUnitZ: 0,
  allHullMinorWidthM: 0,
  upwardForceN: 0,
  downwardRoofForceN: 0,
  dOutM: 0,
  lTailM: 0,
  tiltOutDeg: 0,
  tauRestoreNm: 0,
  tauGravityNm: 0,
};

export const scratchQuasiStaticYield: BlockQuasiStaticYieldResult = {
  overloaded: false,
  shouldCraze: false,
  shouldFracture: false,
  effectiveStressMpa: 0,
  edgeBearingStressMpa: 0,
  flexuralBendingStressMpa: 0,
  targetDamage: 0,
  localOriginDmX: 0,
  localOriginDmY: 0,
  localOriginDmZ: 0,
  worldHitDmX: 0,
  worldHitDmY: 0,
  worldHitDmZ: 0,
  worldNormalX: 0,
  worldNormalY: 1,
  worldNormalZ: 0,
};

/**
 * Reusable zero-GC flat CSR contact collector for up to `maxBlocks` intact blocks.
 */
export class BlockContactScratchBuffer {
  public readonly counts: Int32Array;
  public readonly data: Float64Array;
  public readonly hasTableContact: Uint8Array;
  public readonly hasShardContact: Uint8Array;
  public readonly maxBlocks: number;

  constructor(maxBlocks: number = 128) {
    this.maxBlocks = maxBlocks;
    this.counts = new Int32Array(maxBlocks);
    this.data = new Float64Array(maxBlocks * MAX_CONTACTS_PER_BLOCK * INTERNAL_STRIDE);
    this.hasTableContact = new Uint8Array(maxBlocks);
    this.hasShardContact = new Uint8Array(maxBlocks);
  }

  public clear(activeBlockCount: number): void {
    const n = Math.min(this.maxBlocks, activeBlockCount);
    this.counts.fill(0, 0, n);
    this.hasTableContact.fill(0, 0, n);
    this.hasShardContact.fill(0, 0, n);
  }

  public addContact(
    blockId: number,
    px: number,
    py: number,
    pz: number,
    dist: number,
    fn: number,
    nx: number,
    ny: number,
    nz: number,
    fx: number,
    fy: number,
    fz: number,
    isTable: boolean,
    otherGeomId: number = -1,
    otherIsShard: boolean = false
  ): void {
    if (blockId < 0 || blockId >= this.maxBlocks) return;
    if (isTable && dist < 0.0015) {
      this.hasTableContact[blockId] = 1;
    }
    if (otherIsShard && dist < 0.0015) {
      this.hasShardContact[blockId] = 1;
    }
    const c = this.counts[blockId];
    if (otherIsShard && otherGeomId >= 0 && c >= 3) {
      const prev1 = (blockId * MAX_CONTACTS_PER_BLOCK + (c - 1)) * INTERNAL_STRIDE;
      const prev2 = (blockId * MAX_CONTACTS_PER_BLOCK + (c - 2)) * INTERNAL_STRIDE;
      const prev3 = (blockId * MAX_CONTACTS_PER_BLOCK + (c - 3)) * INTERNAL_STRIDE;
      if (
        this.data[prev1 + 11] === otherGeomId &&
        this.data[prev2 + 11] === otherGeomId &&
        this.data[prev3 + 11] === otherGeomId
      ) {
        const oldFn = this.data[prev1 + 4];
        const newFn = oldFn + fn;
        if (newFn > 1e-6 && fn > 0) {
          const w = fn / newFn;
          this.data[prev1 + 0] += (px - this.data[prev1 + 0]) * w;
          this.data[prev1 + 1] += (py - this.data[prev1 + 1]) * w;
          this.data[prev1 + 2] += (pz - this.data[prev1 + 2]) * w;
        }
        if (dist < this.data[prev1 + 3]) this.data[prev1 + 3] = dist;
        this.data[prev1 + 4] = newFn;
        this.data[prev1 + 8] += fx;
        this.data[prev1 + 9] += fy;
        this.data[prev1 + 10] += fz;
        return;
      }
    }
    if (c >= MAX_CONTACTS_PER_BLOCK) return;
    const base = (blockId * MAX_CONTACTS_PER_BLOCK + c) * INTERNAL_STRIDE;
    this.data[base + 0] = px;
    this.data[base + 1] = py;
    this.data[base + 2] = pz;
    this.data[base + 3] = dist;
    this.data[base + 4] = fn;
    this.data[base + 5] = nx;
    this.data[base + 6] = ny;
    this.data[base + 7] = nz;
    this.data[base + 8] = fx;
    this.data[base + 9] = fy;
    this.data[base + 10] = fz;
    this.data[base + 11] = otherGeomId;
    this.data[base + 12] = otherIsShard ? 1 : 0;
    this.data[base + 13] = isTable ? 1 : 0;
    this.counts[blockId] = c + 1;
  }
}

/**
 * Computes 2D CCW convex hull, Rotating-Calipers minimum strip width, Shoelace area,
 * and signed outward distance + closest fulcrum point from query `(qx, qz)`.
 * Zero heap allocation.
 */
export function analyzeContactPolygon2D(
  ptsX: Float64Array,
  ptsZ: Float64Array,
  rawCount: number,
  queryX: number,
  queryZ: number,
  out: ContactHull2DMetrics = scratchAllHullMetrics
): ContactHull2DMetrics {
  out.pointCount = rawCount;
  out.hullCount = 0;
  out.minorWidthM = 0;
  out.majorSpanM = 0;
  out.areaM2 = 0;
  out.effectiveAreaM2 = 2.0e-6;
  out.stripTanX = 1;
  out.stripTanZ = 0;
  out.stripCenterX = queryX;
  out.stripCenterZ = queryZ;
  out.dOutM = 0;
  out.fulcrumX = queryX;
  out.fulcrumZ = queryZ;
  out.outUnitX = 1;
  out.outUnitZ = 0;

  if (rawCount <= 0) return out;

  // Insertion-sort indices lexicographically by (x, z) and deduplicate within 0.02 mm
  const nCap = Math.min(rawCount, 64);
  for (let i = 0; i < nCap; i++) {
    const x = ptsX[i];
    const z = ptsZ[i];
    let j = i - 1;
    while (j >= 0) {
      const idxJ = scratchOrder[j];
      const xj = ptsX[idxJ];
      const zj = ptsZ[idxJ];
      if (xj > x || (xj === x && zj > z)) {
        scratchOrder[j + 1] = idxJ;
        j--;
      } else {
        break;
      }
    }
    scratchOrder[j + 1] = i;
  }

  let uCount = 0;
  for (let i = 0; i < nCap; i++) {
    const idx = scratchOrder[i];
    const x = ptsX[idx];
    const z = ptsZ[idx];
    if (
      uCount > 0 &&
      Math.abs(x - scratchHullX[uCount - 1]) < 2e-5 &&
      Math.abs(z - scratchHullZ[uCount - 1]) < 2e-5
    ) {
      continue;
    }
    scratchHullX[uCount] = x;
    scratchHullZ[uCount] = z;
    uCount++;
  }

  if (uCount === 1) {
    const fx = scratchHullX[0];
    const fz = scratchHullZ[0];
    out.hullCount = 1;
    out.stripCenterX = fx;
    out.stripCenterZ = fz;
    out.fulcrumX = fx;
    out.fulcrumZ = fz;
    const dx = queryX - fx;
    const dz = queryZ - fz;
    const d = Math.hypot(dx, dz);
    out.dOutM = d;
    if (d > 1e-9) {
      out.outUnitX = dx / d;
      out.outUnitZ = dz / d;
    }
    return out;
  }

  if (uCount === 2) {
    const ax = scratchHullX[0];
    const az = scratchHullZ[0];
    const bx = scratchHullX[1];
    const bz = scratchHullZ[1];
    const ex = bx - ax;
    const ez = bz - az;
    const len = Math.hypot(ex, ez);
    out.hullCount = 2;
    out.majorSpanM = len;
    out.minorWidthM = 0;
    out.effectiveAreaM2 = Math.max(2.0e-6, Math.max(len, 0.0025) * 0.0008);
    out.stripTanX = len > 1e-9 ? ex / len : 1;
    out.stripTanZ = len > 1e-9 ? ez / len : 0;
    out.stripCenterX = 0.5 * (ax + bx);
    out.stripCenterZ = 0.5 * (az + bz);

    const t =
      len > 1e-9
        ? Math.max(0, Math.min(1, ((queryX - ax) * ex + (queryZ - az) * ez) / (len * len)))
        : 0;
    const cx = ax + ex * t;
    const cz = az + ez * t;
    out.fulcrumX = cx;
    out.fulcrumZ = cz;
    const dx = queryX - cx;
    const dz = queryZ - cz;
    const d = Math.hypot(dx, dz);
    out.dOutM = d;
    if (d > 1e-9) {
      out.outUnitX = dx / d;
      out.outUnitZ = dz / d;
    } else {
      out.outUnitX = -out.stripTanZ;
      out.outUnitZ = out.stripTanX;
    }
    return out;
  }

  // Andrew's monotone chain CCW hull in 2D (X, Z) where 2D cross(O->A, O->B) = (ax-ox)*(bz-oz) - (az-oz)*(bx-ox)
  let k = 0;
  for (let i = 0; i < uCount; i++) {
    const px = scratchHullX[i];
    const pz = scratchHullZ[i];
    while (k >= 2) {
      const i1 = scratchHullIdx[k - 2];
      const i2 = scratchHullIdx[k - 1];
      const cross =
        (scratchHullX[i2] - scratchHullX[i1]) * (pz - scratchHullZ[i1]) -
        (scratchHullZ[i2] - scratchHullZ[i1]) * (px - scratchHullX[i1]);
      if (cross <= 1e-12) k--;
      else break;
    }
    scratchHullIdx[k++] = i;
  }
  for (let i = uCount - 2, t = k + 1; i >= 0; i--) {
    const px = scratchHullX[i];
    const pz = scratchHullZ[i];
    while (k >= t) {
      const i1 = scratchHullIdx[k - 2];
      const i2 = scratchHullIdx[k - 1];
      const cross =
        (scratchHullX[i2] - scratchHullX[i1]) * (pz - scratchHullZ[i1]) -
        (scratchHullZ[i2] - scratchHullZ[i1]) * (px - scratchHullX[i1]);
      if (cross <= 1e-12) k--;
      else break;
    }
    scratchHullIdx[k++] = i;
  }
  const H = k - 1;
  for (let i = 0; i < H; i++) {
    const idx = scratchHullIdx[i];
    scratchPtsX[i] = scratchHullX[idx];
    scratchPtsZ[i] = scratchHullZ[idx];
  }

  if (H === 2) {
    return analyzeContactPolygon2D(scratchPtsX, scratchPtsZ, 2, queryX, queryZ, out);
  }

  out.hullCount = H;

  // Major span, Shoelace area, Rotating-Calipers minimum strip width, and closest boundary point
  let maxSpanSq = 0;
  for (let i = 0; i < H; i++) {
    for (let j = i + 1; j < H; j++) {
      const dx = scratchPtsX[j] - scratchPtsX[i];
      const dz = scratchPtsZ[j] - scratchPtsZ[i];
      const dSq = dx * dx + dz * dz;
      if (dSq > maxSpanSq) maxSpanSq = dSq;
    }
  }
  out.majorSpanM = Math.sqrt(maxSpanSq);

  let shoelace2 = 0;
  let minWidth = Infinity;
  let bestTanX = 1;
  let bestTanZ = 0;
  let bestCenterX = scratchPtsX[0];
  let bestCenterZ = scratchPtsZ[0];

  let insideAll = true;
  let minInteriorMargin = Infinity;
  let minOutsideDistSq = Infinity;
  let bestFulcrumX = queryX;
  let bestFulcrumZ = queryZ;

  for (let i = 0; i < H; i++) {
    const j = (i + 1) % H;
    const ax = scratchPtsX[i];
    const az = scratchPtsZ[i];
    const bx = scratchPtsX[j];
    const bz = scratchPtsZ[j];
    shoelace2 += ax * bz - az * bx;

    const ex = bx - ax;
    const ez = bz - az;
    const eLen = Math.hypot(ex, ez);
    if (eLen > 1e-9) {
      const tx = ex / eLen;
      const tz = ez / eLen;
      // Inward normal for CCW polygon in (X, Z) is (-tz, tx)
      const inX = -tz;
      const inZ = tx;

      let maxProj = 0;
      let sumAlong = 0;
      for (let m = 0; m < H; m++) {
        const rx = scratchPtsX[m] - ax;
        const rz = scratchPtsZ[m] - az;
        const p = rx * inX + rz * inZ;
        if (p > maxProj) maxProj = p;
        sumAlong += rx * tx + rz * tz;
      }
      if (maxProj < minWidth) {
        minWidth = maxProj;
        bestTanX = tx;
        bestTanZ = tz;
        const meanAlong = sumAlong / H;
        bestCenterX = ax + tx * meanAlong + inX * (0.5 * maxProj);
        bestCenterZ = az + tz * meanAlong + inZ * (0.5 * maxProj);
      }

      const qInward = (queryX - ax) * inX + (queryZ - az) * inZ;
      if (qInward < 0) {
        insideAll = false;
      } else if (qInward < minInteriorMargin) {
        minInteriorMargin = qInward;
      }

      const tSeg = Math.max(0, Math.min(1, ((queryX - ax) * ex + (queryZ - az) * ez) / (eLen * eLen)));
      const cx = ax + ex * tSeg;
      const cz = az + ez * tSeg;
      const dx = queryX - cx;
      const dz = queryZ - cz;
      const dSq = dx * dx + dz * dz;
      if (dSq < minOutsideDistSq) {
        minOutsideDistSq = dSq;
        bestFulcrumX = cx;
        bestFulcrumZ = cz;
      }
    }
  }

  const area = 0.5 * Math.abs(shoelace2);
  const safeMinW = Number.isFinite(minWidth) ? minWidth : 0;
  out.areaM2 = area;
  out.minorWidthM = safeMinW;
  out.effectiveAreaM2 = Math.max(
    area,
    Math.max(out.majorSpanM, 0.0025) * Math.max(safeMinW, 0.0008),
    2.0e-6
  );
  out.stripTanX = bestTanX;
  out.stripTanZ = bestTanZ;
  out.stripCenterX = bestCenterX;
  out.stripCenterZ = bestCenterZ;
  out.fulcrumX = bestFulcrumX;
  out.fulcrumZ = bestFulcrumZ;

  if (insideAll) {
    out.dOutM = -(Number.isFinite(minInteriorMargin) ? minInteriorMargin : 0);
    const dx = queryX - bestFulcrumX;
    const dz = queryZ - bestFulcrumZ;
    const d = Math.hypot(dx, dz);
    if (d > 1e-9) {
      out.outUnitX = -dx / d;
      out.outUnitZ = -dz / d;
    }
  } else {
    const d = Math.sqrt(minOutsideDistSq);
    out.dOutM = d;
    if (d > 1e-9) {
      out.outUnitX = (queryX - bestFulcrumX) / d;
      out.outUnitZ = (queryZ - bestFulcrumZ) / d;
    }
  }

  return out;
}

/**
 * Evaluates Law 1A (Knife-Edge Chamfer Release) and Law 1B (360-deg Cantilever & Rounded-Bevel Escape)
 * for a single awake non-grabbed intact block. Zero heap allocation.
 */
export function evaluateBlockContactEquilibrium(
  blockId: number,
  contactsBuf: BlockContactScratchBuffer,
  xM: number,
  yM: number,
  zM: number,
  qw: number,
  qx: number,
  qy: number,
  qz: number,
  vxSi: number,
  vzSi: number,
  wxBody: number,
  wyBody: number,
  wzBody: number,
  massKg: number,
  out: BlockEquilibriumAction = scratchEquilibriumAction,
  allowElevatedTowerSlip: boolean = false
): BlockEquilibriumAction {
  out.isUnbalanced = false;
  out.isKnifeEdgeRelease = false;
  out.isCantileverRelease = false;
  out.overrideFrictionMu = 0;
  out.tauWorldX = 0;
  out.tauWorldZ = 0;
  out.forceWorldX = 0;
  out.forceWorldZ = 0;
  out.outUnitX = 1;
  out.outUnitZ = 0;
  out.allHullMinorWidthM = 0;
  out.upwardForceN = 0;
  out.downwardRoofForceN = 0;
  out.dOutM = 0;
  out.lTailM = 0;
  out.tiltOutDeg = 0;
  out.tauRestoreNm = 0;
  out.tauGravityNm = 0;

  const totalContacts = blockId >= 0 && blockId < contactsBuf.maxBlocks ? contactsBuf.counts[blockId] : 0;
  if (totalContacts <= 0) return out;

  // Principal axes e_x, e_y, e_z in world coordinates
  const nxX = 1.0 - 2.0 * (qy * qy + qz * qz);
  const nxY = 2.0 * (qx * qy + qw * qz);
  const nxZ = 2.0 * (qx * qz - qw * qy);

  const nyX = 2.0 * (qx * qy - qw * qz);
  const nyY = 1.0 - 2.0 * (qx * qx + qz * qz);
  const nyZ = 2.0 * (qy * qz + qw * qx);

  const nzX = 2.0 * (qx * qz + qw * qy);
  const nzY = 2.0 * (qy * qz - qw * qx);
  const nzZ = 1.0 - 2.0 * (qx * qx + qy * qy);

  const absNxY = Math.abs(nxY);
  const absNyY = Math.abs(nyY);
  const absNzY = Math.abs(nzY);
  const maxAxisAlignY = Math.max(absNxY, absNyY, absNzY);

  let allCount = 0;
  let upCount = 0;
  let upAllOnShards = true;
  let sumAllFn = 0;
  let sumUpFn = 0;
  let sumUpPy = 0;
  let sumDnFn = 0;
  let meanSupNx = 0;
  let meanSupNy = 0;
  let meanSupNz = 0;
  let minAllPy = Infinity;
  let maxAllPy = -Infinity;
  let loadCount = 0;
  let minLoadPy = Infinity;
  let maxLoadPy = -Infinity;
  let minUpPy = Infinity;
  let maxUpPy = -Infinity;
  let firstSupportGeomId = -2;
  let hasDistinctSupportGeoms = false;

  const baseBlock = blockId * MAX_CONTACTS_PER_BLOCK * INTERNAL_STRIDE;
  const cd = contactsBuf.data;

  for (let c = 0; c < totalContacts; c++) {
    const b = baseBlock + c * INTERNAL_STRIDE;
    const dist = cd[b + 3];
    if (dist >= 0.0015) continue;

    const px = cd[b + 0];
    const py = cd[b + 1];
    const pz = cd[b + 2];
    const fn = cd[b + 4];
    const cnx = cd[b + 5];
    const cny = cd[b + 6];
    const cnz = cd[b + 7];
    const otherGeom = cd[b + 11];
    if (firstSupportGeomId === -2) {
      firstSupportGeomId = otherGeom;
    } else if (otherGeom !== firstSupportGeomId) {
      hasDistinctSupportGeoms = true;
    }

    scratchPtsX[allCount] = px;
    scratchPtsZ[allCount] = pz;
    allCount++;
    sumAllFn += fn;
    if (py < minAllPy) minAllPy = py;
    if (py > maxAllPy) maxAllPy = py;
    if (fn >= 0.035 || totalContacts <= 2) {
      loadCount++;
      if (py < minLoadPy) minLoadPy = py;
      if (py > maxLoadPy) maxLoadPy = py;
    }

    if (cny > 0.20) {
      scratchUpPtsX[upCount] = px;
      scratchUpPtsZ[upCount] = pz;
      upCount++;
      if (py < minUpPy) minUpPy = py;
      if (py > maxUpPy) maxUpPy = py;
      if (cd[b + 12] === 0) {
        upAllOnShards = false;
      }
      const w = Math.max(1e-4, fn);
      sumUpFn += fn;
      sumUpPy += py * w;
      meanSupNx += cnx * w;
      meanSupNy += cny * w;
      meanSupNz += cnz * w;
    } else if (cny < -0.20) {
      sumDnFn += fn;
    }
  }

  out.upwardForceN = sumUpFn;
  out.downwardRoofForceN = sumDnFn;
  if (allCount === 0) {
    out.isUnbalanced = true;
    return out;
  }

  const allHull = analyzeContactPolygon2D(scratchPtsX, scratchPtsZ, allCount, xM, zM, scratchAllHullMetrics);
  const contactVerticalSpanM =
    loadCount >= 2 ? maxLoadPy - minLoadPy : allCount >= 2 ? maxAllPy - minAllPy : 0;
  const effMaxPropPy = loadCount >= 1 ? maxLoadPy : maxAllPy;
  let minTanProj = Infinity;
  let maxTanProj = -Infinity;
  for (let k = 0; k < allCount; k++) {
    const tp = (scratchPtsX[k] - xM) * allHull.stripTanX + (scratchPtsZ[k] - zM) * allHull.stripTanZ;
    if (tp < minTanProj) minTanProj = tp;
    if (tp > maxTanProj) maxTanProj = tp;
  }
  const isTwoGeomPitchedRamp =
    hasDistinctSupportGeoms &&
    allHull.majorSpanM >= 0.012 &&
    minTanProj < -0.0035 &&
    maxTanProj > 0.0035 &&
    contactVerticalSpanM >= 0.0040;
  const isSingleLineKnifeEdge =
    !isTwoGeomPitchedRamp &&
    (contactVerticalSpanM < 0.0040 || (effMaxPropPy < yM - 0.0025 && contactVerticalSpanM < 0.0120));
  out.allHullMinorWidthM =
    !isSingleLineKnifeEdge
      ? Math.max(allHull.minorWidthM, 0.0040)
      : allHull.minorWidthM;

  const wWorldX = nxX * wxBody + nyX * wyBody + nzX * wzBody;
  const wWorldZ = nxZ * wxBody + nyZ * wyBody + nzZ * wzBody;
  const omegaMag = Math.hypot(wxBody, wyBody, wzBody);

  // =========================================================================
  // LAW 1A: All-Orientation Polygon-Chamfer Knife-Edge Release
  // Single-plane or strictly-sub-CoM 1D edge/corner/bevel-saddle contacts (`width < 2.4 mm`)
  // =========================================================================
  if (
    maxAxisAlignY < KNIFE_EDGE_MAX_AXIS_ALIGN_Y &&
    allHull.minorWidthM < KNIFE_EDGE_MIN_HULL_WIDTH_M &&
    isSingleLineKnifeEdge
  ) {
    out.isUnbalanced = true;
    if (omegaMag < KNIFE_EDGE_MAX_OMEGA_SI) {
      out.isKnifeEdgeRelease = true;

      let rollDirX = 0;
      let rollDirZ = 0;
      let effLeverM = 0.00125;

      // Support normal (defaults to world +Y)
      const supLen = Math.hypot(meanSupNx, meanSupNy, meanSupNz);
      const sNx = supLen > 1e-6 ? meanSupNx / supLen : 0.0;
      const sNy = supLen > 1e-6 ? meanSupNy / supLen : 1.0;
      const sNz = supLen > 1e-6 ? meanSupNz / supLen : 0.0;

      // Local direction pointing into the support (-R^T * sN)
      const uInLocalX = -(nxX * sNx + nxY * sNy + nxZ * sNz);
      const uInLocalY = -(nyX * sNx + nyY * sNy + nyZ * sNz);
      const uInLocalZ = -(nzX * sNx + nzY * sNy + nzZ * sNz);

      // Rounded-fillet inner-core center of curvature (smooth tanh avoids step jumps on flat axes)
      const cxLocal = Math.tanh(uInLocalX * 20.0) * (EQ_BLOCK_HALF_X_M - EQ_BEVEL_RADIUS_M);
      const cyLocal = Math.tanh(uInLocalY * 20.0) * (EQ_BLOCK_HALF_Y_M - EQ_BEVEL_RADIUS_M);
      const czLocal = Math.tanh(uInLocalZ * 20.0) * (EQ_BLOCK_HALF_Z_M - EQ_BEVEL_RADIUS_M);

      const coreOffX = nxX * cxLocal + nyX * cyLocal + nzX * czLocal;
      const coreOffZ = nxZ * cxLocal + nyZ * cyLocal + nzZ * czLocal;
      const dCoreX = -coreOffX;
      const dCoreZ = -coreOffZ;

      const sY = nyY >= 0 ? 1.0 : -1.0;
      const broadRollX = -sY * nyX;
      const broadRollZ = -sY * nyZ;

      if (allHull.majorSpanM >= 0.0030) {
        // Edge strip: roll strictly perpendicular to the contact line
        const perpX = -allHull.stripTanZ;
        const perpZ = allHull.stripTanX;
        const projCore = dCoreX * perpX + dCoreZ * perpZ;
        const projHull = (xM - allHull.stripCenterX) * perpX + (zM - allHull.stripCenterZ) * perpZ;
        const combined = Math.abs(projCore) >= 1.2e-4 ? projCore : projHull;
        let sign = 1.0;
        if (Math.abs(combined) >= 1.0e-4) {
          sign = combined >= 0 ? 1.0 : -1.0;
        } else {
          const broadProj = broadRollX * perpX + broadRollZ * perpZ;
          sign = broadProj >= 0 ? 1.0 : -1.0;
        }
        rollDirX = sign * perpX;
        rollDirZ = sign * perpZ;
        effLeverM = Math.max(Math.abs(projCore), 0.00085) + EQ_BEVEL_RADIUS_M;
      } else {
        // Single-corner or sub-3mm point contact: roll radially outward from contact / core center
        const comDx = xM - allHull.stripCenterX;
        const comDz = zM - allHull.stripCenterZ;
        const useX = Math.hypot(dCoreX, dCoreZ) >= 1.5e-4 ? dCoreX : comDx;
        const useZ = Math.hypot(dCoreX, dCoreZ) >= 1.5e-4 ? dCoreZ : comDz;
        const rLen = Math.hypot(useX, useZ);
        if (rLen >= 1.0e-4) {
          rollDirX = useX / rLen;
          rollDirZ = useZ / rLen;
          effLeverM = Math.max(rLen, 0.00095) + EQ_BEVEL_RADIUS_M;
        } else {
          const bLen = Math.hypot(broadRollX, broadRollZ);
          rollDirX = bLen > 1e-6 ? broadRollX / bLen : 1.0;
          rollDirZ = bLen > 1e-6 ? broadRollZ / bLen : 0.0;
          effLeverM = 0.0022;
        }
      }

      const tauDirX = rollDirZ;
      const tauDirZ = -rollDirX;
      const wRollAlong = wWorldX * tauDirX + wWorldZ * tauDirZ;
      if (omegaMag < 0.28 && wRollAlong >= -0.02) {
        const omegaFade = Math.max(0.0, 1.0 - omegaMag / 0.28);
        const fLoad = Math.min(3.5, Math.max(massKg * 9.81, sumAllFn * 0.35));
        const tauMag =
          Math.min(
            0.0065,
            Math.max(0.0028, fLoad * Math.max(effLeverM, allHull.dOutM + EQ_BEVEL_RADIUS_M))
          ) * omegaFade;
        out.tauWorldX = tauDirX * tauMag;
        out.tauWorldZ = tauDirZ * tauMag;
      }
    }
    return out;
  }

  // =========================================================================
  // LAW 1B: Coordinate-Free 360-deg Cantilever & True Rounded-Bevel Escape
  // =========================================================================
  if (upCount === 0) {
    out.isUnbalanced = true;
    return out;
  }

  // A block resting flat on a principal face (`maxAxisAlignY > 0.985`, `yM < 18.5 mm`) inside the 43 cm
  // circular oak table (`r < 41 cm`) has continuous solid wood beneath its CoM — 1-edge micro-rocking is Housner
  // flat-face rocking, not a cliff overhang! (Upright or >10-deg tilted blocks MUST still evaluate Law 1B.)
  const hasTable = blockId >= 0 && blockId < contactsBuf.maxBlocks && contactsBuf.hasTableContact[blockId] !== 0;
  if (
    hasTable &&
    maxAxisAlignY > KNIFE_EDGE_MAX_AXIS_ALIGN_Y &&
    yM < 0.0185 &&
    xM * xM + zM * zM < 0.41 * 0.41
  ) {
    return out;
  }

  const upHull = analyzeContactPolygon2D(scratchUpPtsX, scratchUpPtsZ, upCount, xM, zM, scratchUpHullMetrics);
  const dOut = upHull.dOutM;
  out.dOutM = dOut;
  if (dOut <= BEVEL_UNROOFED_MIN_OVERHANG_M) {
    return out;
  }

  const ux = upHull.outUnitX;
  const uz = upHull.outUnitZ;
  out.outUnitX = ux;
  out.outUnitZ = uz;

  // Projected 3D half-span R_span(u_out) across any 360-deg block orientation
  const rSpanM =
    Math.abs(nxX * ux + nxZ * uz) * EQ_BLOCK_HALF_X_M +
    Math.abs(nyX * ux + nyZ * uz) * EQ_BLOCK_HALF_Y_M +
    Math.abs(nzX * ux + nzZ * uz) * EQ_BLOCK_HALF_Z_M;
  const lTailM = rSpanM - dOut;
  out.lTailM = lTailM;

  const supLenForTilt = Math.hypot(meanSupNx, meanSupNy, meanSupNz);
  const supNx = supLenForTilt > 1e-6 ? meanSupNx / supLenForTilt : 0.0;
  const supNy = supLenForTilt > 1e-6 ? meanSupNy / supLenForTilt : 1.0;
  const supNz = supLenForTilt > 1e-6 ? meanSupNz / supLenForTilt : 0.0;

  // Nose-down tilt angle along u_out using whichever principal face normal points most upward
  let upFaceX = nyX;
  let upFaceZ = nyZ;
  if (absNyY >= absNxY && absNyY >= absNzY) {
    const s = nyY >= 0 ? 1.0 : -1.0;
    upFaceX = s * nyX;
    upFaceZ = s * nyZ;
  } else if (absNzY >= absNxY) {
    const s = nzY >= 0 ? 1.0 : -1.0;
    upFaceX = s * nzX;
    upFaceZ = s * nzZ;
  } else {
    const s = nxY >= 0 ? 1.0 : -1.0;
    upFaceX = s * nxX;
    upFaceZ = s * nxZ;
  }
  const sinTiltOut = Math.max(-1.0, Math.min(1.0, upFaceX * ux + upFaceZ * uz));
  const tiltOutDeg = (Math.asin(sinTiltOut) * 180.0) / Math.PI;
  const sinTiltRelSup = Math.max(
    -1.0,
    Math.min(1.0, (upFaceX - supNx) * ux + (upFaceZ - supNz) * uz)
  );
  const tiltRelSupDeg = (Math.asin(sinTiltRelSup) * 180.0) / Math.PI;
  out.tiltOutDeg = tiltOutDeg;

  // Nose-up Housner / lateral edge rocking (`-8.5 < tiltOutDeg < -0.20 deg`, `dOut < 15 mm`, e.g. 6.5-deg
  // fulcrum seesaw or a pillar rocked backward onto its inner edge over a solid course) has solid support
  // beneath its raised nose rather than a cliff abyss — preserve natural rocking & static friction!
  if (
    (tiltRelSupDeg < -0.25 && tiltRelSupDeg > -8.5 && omegaMag >= 0.03 && dOut < 0.015) ||
    (tiltOutDeg < -0.005 && tiltOutDeg > -8.5 && dOut < 0.015)
  ) {
    return out;
  }

  const qxFulc = upHull.fulcrumX;
  const qzFulc = upHull.fulcrumZ;
  let qyFulc = sumUpFn > 1e-5 ? sumUpPy / sumUpFn : yM - EQ_BLOCK_HALF_Y_M;
  let bestFulcDistSq = Infinity;
  for (let c = 0; c < totalContacts; c++) {
    const b = baseBlock + c * INTERNAL_STRIDE;
    if (cd[b + 3] >= 0.0015 || cd[b + 6] <= 0.20) continue;
    const dxF = cd[b + 0] - qxFulc;
    const dzF = cd[b + 2] - qzFulc;
    const dSq = dxF * dxF + dzF * dzF;
    if (dSq < bestFulcDistSq) {
      bestFulcDistSq = dSq;
      qyFulc = cd[b + 1];
    }
  }

  let tauRestorePosNm = 0;
  let tauRestoreAlgNm = 0;
  let inboardRoofFn = 0;
  let meanRoofNx = 0;
  let meanRoofNy = 0;
  let meanRoofNz = 0;
  let roofWeightSum = 0;
  let hasOutboardPropBeyondCom = false;
  let hasInboardPropBehindCom = false;
  for (let c = 0; c < totalContacts; c++) {
    const b = baseBlock + c * INTERNAL_STRIDE;
    const dist = cd[b + 3];
    if (dist >= 0.0015) continue;
    const px = cd[b + 0];
    const py = cd[b + 1];
    const pz = cd[b + 2];
    const fn = cd[b + 4];
    const cnx = cd[b + 5];
    const cny = cd[b + 6];
    const cnz = cd[b + 7];
    const fx = cd[b + 8];
    const fy = cd[b + 9];
    const fz = cd[b + 10];

    const sFromCom = (px - xM) * ux + (pz - zM) * uz;
    if (cny > -0.15 && fn >= 0.10 * massKg * 9.81 && sFromCom > 0.0015) {
      hasOutboardPropBeyondCom = true;
    }
    if (cny > -0.15 && fn >= 0.10 * massKg * 9.81 && sFromCom < -0.0015) {
      hasInboardPropBehindCom = true;
    }

    if (fn <= 1e-5) continue;
    const sAlong = (px - qxFulc) * ux + (pz - qzFulc) * uz;
    const ry = py - qyFulc;

    if (cny < -0.20 && cd[b + 12] === 0) {
      const rw = Math.max(1e-4, fn);
      meanRoofNx += cnx * rw;
      meanRoofNy += cny * rw;
      meanRoofNz += cnz * rw;
      roofWeightSum += rw;
      if (sAlong < -0.0015) {
        inboardRoofFn += -fn * cny;
      }
    }
    const isHorizCeilingPinch =
      supNy > 0.985 &&
      cny < -0.97 &&
      cd[b + 12] === 0 &&
      rSpanM >= 0.020 &&
      lTailM <= 0.0220 &&
      dOut >= 0.0150;
    const exactMom = isHorizCeilingPinch
      ? fy * sAlong
      : fy * sAlong - ry * (fx * ux + fz * uz);
    tauRestoreAlgNm += exactMom;
    if (Math.abs(sAlong) > 0.0012 || Math.abs(ry) > 0.0015) {
      if (exactMom > 0) {
        tauRestorePosNm += exactMom;
      } else if (
        rSpanM >= 0.020 &&
        lTailM <= 0.0220 &&
        dOut >= 0.0150 &&
        cny < -0.85 &&
        fy < 0 &&
        sAlong > 0.0015
      ) {
        tauRestorePosNm += fy * sAlong;
      }
    }
  }
  const tauRestoreNm = Math.max(tauRestorePosNm, tauRestoreAlgNm);

  // Two-end bridge / 3D pocket whose contact hull brackets CoM (`allHull.dOutM <= 1.0 mm` without ceiling slot pinch)
  // is supported on both -u_out and +u_out sides — not an overhanging cliff cantilever!
  if (
    (allHull.dOutM <= 0.0010 && inboardRoofFn <= 0.04) ||
    (hasOutboardPropBeyondCom && (hasInboardPropBehindCom || allHull.dOutM <= 0.0015))
  ) {
    return out;
  }

  const tauGravityNm = massKg * 9.81 * dOut;
  out.tauRestoreNm = tauRestoreNm;
  out.tauGravityNm = tauGravityNm;

  // Measure wedge angle relative to the overhead ceiling face (-n_roof):
  // - Flush counterweighted corbels have n_roof = -n_top -> slotWedgeTiltDeg = 0 deg (even if the stack leans!),
  // - Nose-down two-bevel slot wedges hit the horizontal ceiling on their rear 1.25 mm fillet -> slotWedgeTiltDeg > 0.
  const roofLen = Math.hypot(meanRoofNx, meanRoofNy, meanRoofNz);
  const rNx = roofWeightSum > 1e-6 && roofLen > 1e-6 ? meanRoofNx / roofLen : 0.0;
  const rNy = roofWeightSum > 1e-6 && roofLen > 1e-6 ? meanRoofNy / roofLen : -1.0;
  const rNz = roofWeightSum > 1e-6 && roofLen > 1e-6 ? meanRoofNz / roofLen : 0.0;
  const sinSlotWedgeTilt = Math.max(
    -1.0,
    Math.min(1.0, (upFaceX + rNx) * ux + (upFaceZ + rNz) * uz)
  );
  const slotWedgeTiltDeg = (Math.asin(sinSlotWedgeTilt) * 180.0) / Math.PI;

  // Interior cross-pillar guard: in a standing tower (`margin = 0`), a centered block resting across a
  // 25 mm single-pillar `+` cross (`dOut < 14.5 mm`, `L_tail > 22.0 mm`) drops its opposite +/-12.5 mm edge
  // contact out of `d.contact` on any 10-micron (`0.01 deg`) sway while carrying overhead courses (`sumDnFn > 0.15 N`);
  // never drop static glass friction on a flat centered interior course!
  if (
    yM >= 0.045 &&
    sumDnFn > 0.15 &&
    inboardRoofFn <= 0.04 &&
    maxAxisAlignY > KNIFE_EDGE_MAX_AXIS_ALIGN_Y &&
    lTailM > 0.0220 &&
    dOut < 0.0145
  ) {
    return out;
  }

  const isUnroofedOrUnpropped = tauRestoreNm < 0.96 * tauGravityNm;
  // Horizontal tower-slot bevel/lip escape applies when pinched by an overhead course (`inboardRoofFn > 0.04` or `sumDnFn >= 2.8`)
  // inside a near-horizontal slot (`supNy > 0.985 && rNy < -0.990`), never on inclined multi-block rubble ramps!
  // Includes deep nose-down two-bevel slot wedges (`dOut >= 14.5 mm`, `L_tail <= 22.0 mm`, `tiltOut > 0.15 deg` with
  // `slotWedgeTilt >= 1.05 deg` or heavy-load buckling outer pillar), while preserving flat counterweighted corbels (`68%` overhang).
  const effBevelEscapeTailM = rSpanM >= 0.020 ? BEVEL_ESCAPE_TAIL_LIMIT_M : 0.0020;
  const isDeepTiltedSlotBevelWedge = allowElevatedTowerSlip
    ? rSpanM >= 0.020 &&
      dOut >= 0.0145 &&
      lTailM <= 0.0220 &&
      tiltOutDeg > 0.15 &&
      rNy < -0.990 &&
      (slotWedgeTiltDeg >= 1.05 || (yM >= 0.055 && sumDnFn >= 2.8 && tiltOutDeg >= 0.45))
    : rSpanM >= 0.020 && lTailM <= 0.0220 && slotWedgeTiltDeg >= 1.05;
  const isBevelLipOrSteepTiltEscape =
    maxAxisAlignY > 0.90 &&
    (inboardRoofFn > 0.04 || (allowElevatedTowerSlip && yM >= 0.055 && sumDnFn >= 2.8)) &&
    supNy > 0.985 &&
    rNy < (allowElevatedTowerSlip ? -0.990 : -0.970) &&
    (lTailM <= effBevelEscapeTailM ||
      slotWedgeTiltDeg >= BEVEL_SLIP_TILT_LIMIT_DEG ||
      isDeepTiltedSlotBevelWedge);

  if (!isUnroofedOrUnpropped && !isBevelLipOrSteepTiltEscape) {
    // Genuine counterweighted corbel or suspenseful slot-held drawer (L_tail > 6.5 mm & tilt < 7.5 deg):
    // preserve natural physics with zero artificial ejection!
    return out;
  }

  const vHorizSi = Math.hypot(vxSi, vzSi);
  // When a nearly-stationary horizontal block (`vHoriz < 55 mm/s`, `maxAxisAlignY > 0.965`)
  // rests solely on sub-gram Voronoi shards (`margin = 0`), 10-micron GJK micro-bounces transiently drop 1 of its
  // shard contacts out of `d.contact`; avoid artificial bevel-fillet pogo resonance on shard beds while letting
  // MuJoCo's native gravity tip genuine overhangs.
  const isStationaryFlatShardBed =
    upAllOnShards &&
    vHorizSi < 0.055 &&
    omegaMag < 2.2 &&
    maxAxisAlignY > 0.965 &&
    yM < 0.042 &&
    dOut < 0.022 &&
    inboardRoofFn <= 0.04;
  if (isStationaryFlatShardBed) {
    return out;
  }

  out.isUnbalanced = true;
  const isCeilingPinched = inboardRoofFn > 0.04 || (isDeepTiltedSlotBevelWedge && sumDnFn >= 2.8);
  const isElevatedTowerCourse = allowElevatedTowerSlip && yM >= 0.055;
  const maxActiveOmega =
    isCeilingPinched || isDeepTiltedSlotBevelWedge || isElevatedTowerCourse
      ? 3.2
      : KNIFE_EDGE_MAX_OMEGA_SI;
  const maxActiveHorizV =
    isCeilingPinched || isDeepTiltedSlotBevelWedge || isElevatedTowerCourse ? 0.40 : 0.25;
  if (omegaMag >= maxActiveOmega || vHorizSi >= maxActiveHorizV) {
    return out;
  }
  out.isCantileverRelease = true;

  // While ceiling-pinched past bevel/tilt escape, use 0.018 slot-pinch bevel friction to clear the
  // horizontal ceiling Painleve lock; once unroofed/clear, use 0.080 polished glass bevel friction.
  out.overrideFrictionMu =
    (!allowElevatedTowerSlip || isBevelLipOrSteepTiltEscape) && isCeilingPinched
      ? MU_SLOT_PINCH_ESCAPE
      : MU_GLASS_BEVEL_SLIP;

  const pitchAxisX = uz;
  const pitchAxisZ = -ux;
  const wPitchAlong = wWorldX * pitchAxisX + wWorldZ * pitchAxisZ;

  const ramp01 = isUnroofedOrUnpropped
    ? Math.min(1.0, (dOut - BEVEL_UNROOFED_MIN_OVERHANG_M) / 0.0025)
    : Math.min(
        1.0,
        Math.max(
          (BEVEL_ESCAPE_TAIL_LIMIT_M - lTailM + 0.0010) / 0.0025,
          (tiltOutDeg - (BEVEL_SLIP_TILT_LIMIT_DEG - 1.5)) / 3.0,
          isDeepTiltedSlotBevelWedge
            ? Math.max(0.55, (0.0225 - lTailM) / 0.0040)
            : 0.0
        )
      );

  const maxTipOmega = isCeilingPinched ? KNIFE_EDGE_MAX_OMEGA_SI : 0.28;
  if (
    omegaMag < maxTipOmega &&
    vHorizSi < 0.22 &&
    wPitchAlong < 1.2 &&
    (isCeilingPinched || wPitchAlong >= -0.02)
  ) {
    // Gravity-consistent R_bevel = 1.25 mm rounded-fillet tipping moment (smoothly tapers to 0 to prevent bang-bang chatter)
    const omegaFade = isCeilingPinched ? 1.0 : Math.max(0.0, 1.0 - omegaMag / maxTipOmega);
    const tauFillet = isCeilingPinched
      ? Math.min(0.0085, (0.0032 + Math.min(12.0, inboardRoofFn) * 0.00045) * ramp01)
      : Math.min(0.0038, Math.max(0.0020, massKg * 9.81 * 0.0035 * ramp01)) * omegaFade;
    out.tauWorldX = pitchAxisX * tauFillet;
    out.tauWorldZ = pitchAxisZ * tauFillet;
  }

  // Gravity-bounded bevel-cam outward slip when ceiling-pinched OR deep-slot-wedged on the 1.25 mm lip (v_out <= 0.12 m/s)
  if ((inboardRoofFn > 0.02 || isDeepTiltedSlotBevelWedge) && omegaMag < 3.0) {
    const vOutSi = vxSi * ux + vzSi * uz;
    if (vOutSi < MAX_CAM_SLIP_SPEED_SI) {
      const fCamOutN = Math.min(
        0.95,
        (0.24 + Math.min(20.0, inboardRoofFn) * 0.055) * Math.max(0.35, ramp01)
      );
      out.forceWorldX = ux * fCamOutN;
      out.forceWorldZ = uz * fCamOutN;
    }
  }

  return out;
}

const clipPolyAx = new Float64Array(16);
const clipPolyAz = new Float64Array(16);
const clipPolyBx = new Float64Array(16);
const clipPolyBz = new Float64Array(16);

/**
 * Coordinate-free 3D micro-gap support check (`<= 1.8 mm` OBB clearance):
 * In MuJoCo with `margin = 0`, a `0.02 deg` (`5-micron`) elastic sway lifts one side of a centered or
 * 2-pillar-bridged flat block out of `d.contact`, making `upHull` appear one-sided even though a solid
 * supporting block sits microns below the `+u_out` side. Returns `true` iff `ob` is within `1.8 mm` 3D OBB
 * clearance below `b` AND `ob`'s bevel-inset top footprint overlaps `b`'s footprint at least `+1.2 mm`
 * outboard of `b`'s CoM along `+u_out` (strictly `false` for any genuine `>= 0.1 mm` CoM overhang).
 */
export function hasOutboardMicroGapBlockSupportM(
  bxM: number,
  byM: number,
  bzM: number,
  qw: number,
  qx: number,
  qy: number,
  qz: number,
  obxM: number,
  obyM: number,
  obzM: number,
  obQw: number,
  obQx: number,
  obQy: number,
  obQz: number,
  ux: number,
  uz: number
): boolean {
  if (
    obyM > byM - 0.0055 ||
    obyM < byM - 0.045 ||
    Math.abs(obxM - bxM) > 0.080 ||
    Math.abs(obzM - bzM) > 0.080
  ) {
    return false;
  }

  // Support block must sit on one of its 6 principal faces (`< 15 deg` tilt)
  const obNxX = 1.0 - 2.0 * (obQy * obQy + obQz * obQz);
  const obNxY = 2.0 * (obQx * obQy + obQw * obQz);
  const obNxZ = 2.0 * (obQx * obQz - obQw * obQy);
  const obNyX = 2.0 * (obQx * obQy - obQw * obQz);
  const obNyY = 1.0 - 2.0 * (obQx * obQx + obQz * obQz);
  const obNyZ = 2.0 * (obQy * obQz + obQw * obQx);
  const obNzX = 2.0 * (obQx * obQz + obQw * obQy);
  const obNzY = 2.0 * (obQy * obQz - obQw * obQx);
  const obNzZ = 1.0 - 2.0 * (obQx * obQx + obQy * obQy);

  const absObNxY = Math.abs(obNxY);
  const absObNyY = Math.abs(obNyY);
  const absObNzY = Math.abs(obNzY);
  if (Math.max(absObNxY, absObNyY, absObNzY) < 0.965) return false;

  if (
    !areBlockObbsWithinGapDm(
      bxM * 10.0,
      byM * 10.0,
      bzM * 10.0,
      qx,
      qy,
      qz,
      qw,
      obxM * 10.0,
      obyM * 10.0,
      obzM * 10.0,
      obQx,
      obQy,
      obQz,
      obQw,
      0.018
    )
  ) {
    return false;
  }

  const tx = bxM + ux * 0.016 - obxM;
  const tz = bzM + uz * 0.016 - obzM;
  const hxIn = EQ_BLOCK_HALF_X_M - EQ_BEVEL_RADIUS_M;
  const hyIn = EQ_BLOCK_HALF_Y_M - EQ_BEVEL_RADIUS_M;
  const hzIn = EQ_BLOCK_HALF_Z_M - EQ_BEVEL_RADIUS_M;

  let cx = obxM;
  let cz = obzM;
  if (absObNyY >= absObNxY && absObNyY >= absObNzY) {
    const lx = Math.max(-hxIn, Math.min(hxIn, tx * obNxX + tz * obNxZ));
    const lz = Math.max(-hzIn, Math.min(hzIn, tx * obNzX + tz * obNzZ));
    cx += lx * obNxX + lz * obNzX;
    cz += lx * obNxZ + lz * obNzZ;
  } else if (absObNzY >= absObNxY) {
    const lx = Math.max(-hxIn, Math.min(hxIn, tx * obNxX + tz * obNxZ));
    const ly = Math.max(-hyIn, Math.min(hyIn, tx * obNyX + tz * obNyZ));
    cx += lx * obNxX + ly * obNyX;
    cz += lx * obNxZ + ly * obNyZ;
  } else {
    const ly = Math.max(-hyIn, Math.min(hyIn, tx * obNyX + tz * obNyZ));
    const lz = Math.max(-hzIn, Math.min(hzIn, tx * obNzX + tz * obNzZ));
    cx += ly * obNyX + lz * obNzX;
    cz += ly * obNyZ + lz * obNzZ;
  }

  if ((cx - bxM) * ux + (cz - bzM) * uz < 0.0012) return false;

  // Verify the support point also lies inside `b`'s own horizontal bevel-inset footprint
  const bNxX = 1.0 - 2.0 * (qy * qy + qz * qz);
  const bNxY = 2.0 * (qx * qy + qw * qz);
  const bNxZ = 2.0 * (qx * qz - qw * qy);
  const bNyX = 2.0 * (qx * qy - qw * qz);
  const bNyY = 1.0 - 2.0 * (qx * qx + qz * qz);
  const bNyZ = 2.0 * (qy * qz + qw * qx);
  const bNzX = 2.0 * (qx * qz + qw * qy);
  const bNzY = 2.0 * (qy * qz - qw * qx);
  const bNzZ = 1.0 - 2.0 * (qx * qx + qy * qy);

  const relX = cx - bxM;
  const relZ = cz - bzM;
  const absBNxY = Math.abs(bNxY);
  const absBNyY = Math.abs(bNyY);
  const absBNzY = Math.abs(bNzY);
  if (absBNyY >= absBNxY && absBNyY >= absBNzY) {
    return (
      Math.abs(relX * bNxX + relZ * bNxZ) <= hxIn + 0.0005 &&
      Math.abs(relX * bNzX + relZ * bNzZ) <= hzIn + 0.0005
    );
  }
  if (absBNzY >= absBNxY) {
    return (
      Math.abs(relX * bNxX + relZ * bNxZ) <= hxIn + 0.0005 &&
      Math.abs(relX * bNyX + relZ * bNyZ) <= hyIn + 0.0005
    );
  }
  return (
    Math.abs(relX * bNyX + relZ * bNyZ) <= hyIn + 0.0005 &&
    Math.abs(relX * bNzX + relZ * bNzZ) <= hzIn + 0.0005
  );
}

/**
 * Computes the exact 2D `(X, Z)` Sutherland-Hodgman intersection area (`m^2`) of the bevel-inset
 * horizontal faces of two blocks when both faces are coplanar within ~7.2 deg (`|uA · uB| > 0.992`).
 * Returns `0` when either block is tilted onto an edge/corner relative to the other, so narrow
 * tilted slivers use their true point/line contact patch. Strictly zero heap allocation.
 */
export function computeCoplanarBlockFaceOverlapAreaM2(
  axM: number,
  azM: number,
  aqw: number,
  aqx: number,
  aqy: number,
  aqz: number,
  bxM: number,
  bzM: number,
  bqw: number,
  bqx: number,
  bqy: number,
  bqz: number
): number {
  const a0x = 1.0 - 2.0 * (aqy * aqy + aqz * aqz);
  const a0y = 2.0 * (aqx * aqy + aqw * aqz);
  const a0z = 2.0 * (aqx * aqz - aqw * aqy);
  const a1x = 2.0 * (aqx * aqy - aqw * aqz);
  const a1y = 1.0 - 2.0 * (aqx * aqx + aqz * aqz);
  const a1z = 2.0 * (aqy * aqz + aqw * aqx);
  const a2x = 2.0 * (aqx * aqz + aqw * aqy);
  const a2y = 2.0 * (aqy * aqz - aqw * aqx);
  const a2z = 1.0 - 2.0 * (aqx * aqx + aqy * aqy);

  const b0x = 1.0 - 2.0 * (bqy * bqy + bqz * bqz);
  const b0y = 2.0 * (bqx * bqy + bqw * bqz);
  const b0z = 2.0 * (bqx * bqz - bqw * bqy);
  const b1x = 2.0 * (bqx * bqy - bqw * bqz);
  const b1y = 1.0 - 2.0 * (bqx * bqx + bqz * bqz);
  const b1z = 2.0 * (bqy * bqz + bqw * bqx);
  const b2x = 2.0 * (bqx * bqz + bqw * bqy);
  const b2y = 2.0 * (bqy * bqz - bqw * bqx);
  const b2z = 1.0 - 2.0 * (bqx * bqx + bqy * bqy);

  // Determine which principal face of A and B is horizontal (must be within ~7 deg of world Y)
  const aAbs0 = Math.abs(a0y), aAbs1 = Math.abs(a1y), aAbs2 = Math.abs(a2y);
  const aMaxY = Math.max(aAbs0, aAbs1, aAbs2);
  const bAbs0 = Math.abs(b0y), bAbs1 = Math.abs(b1y), bAbs2 = Math.abs(b2y);
  const bMaxY = Math.max(bAbs0, bAbs1, bAbs2);
  if (aMaxY < 0.992 || bMaxY < 0.992) return 0;

  let auX: number, auZ: number, avX: number, avZ: number, huA: number, hvA: number;
  if (aAbs1 >= aAbs0 && aAbs1 >= aAbs2) {
    auX = a0x; auZ = a0z; huA = EQ_BLOCK_HALF_X_M - EQ_BEVEL_RADIUS_M;
    avX = a2x; avZ = a2z; hvA = EQ_BLOCK_HALF_Z_M - EQ_BEVEL_RADIUS_M;
  } else if (aAbs2 >= aAbs0) {
    auX = a0x; auZ = a0z; huA = EQ_BLOCK_HALF_X_M - EQ_BEVEL_RADIUS_M;
    avX = a1x; avZ = a1z; hvA = EQ_BLOCK_HALF_Y_M - EQ_BEVEL_RADIUS_M;
  } else {
    auX = a2x; auZ = a2z; huA = EQ_BLOCK_HALF_Z_M - EQ_BEVEL_RADIUS_M;
    avX = a1x; avZ = a1z; hvA = EQ_BLOCK_HALF_Y_M - EQ_BEVEL_RADIUS_M;
  }

  let buX: number, buZ: number, bvX: number, bvZ: number, huB: number, hvB: number;
  if (bAbs1 >= bAbs0 && bAbs1 >= bAbs2) {
    buX = b0x; buZ = b0z; huB = EQ_BLOCK_HALF_X_M - EQ_BEVEL_RADIUS_M;
    bvX = b2x; bvZ = b2z; hvB = EQ_BLOCK_HALF_Z_M - EQ_BEVEL_RADIUS_M;
  } else if (bAbs2 >= bAbs0) {
    buX = b0x; buZ = b0z; huB = EQ_BLOCK_HALF_X_M - EQ_BEVEL_RADIUS_M;
    bvX = b1x; bvZ = b1z; hvB = EQ_BLOCK_HALF_Y_M - EQ_BEVEL_RADIUS_M;
  } else {
    buX = b2x; buZ = b2z; huB = EQ_BLOCK_HALF_Z_M - EQ_BEVEL_RADIUS_M;
    bvX = b1x; bvZ = b1z; hvB = EQ_BLOCK_HALF_Y_M - EQ_BEVEL_RADIUS_M;
  }

  const uLenB = Math.hypot(buX, buZ) || 1;
  buX /= uLenB; buZ /= uLenB;
  const vLenB = Math.hypot(bvX, bvZ) || 1;
  bvX /= vLenB; bvZ /= vLenB;

  // Initialize polygon with 4 corners of A's horizontal face
  clipPolyAx[0] = axM - auX * huA - avX * hvA; clipPolyAz[0] = azM - auZ * huA - avZ * hvA;
  clipPolyAx[1] = axM + auX * huA - avX * hvA; clipPolyAz[1] = azM + auZ * huA - avZ * hvA;
  clipPolyAx[2] = axM + auX * huA + avX * hvA; clipPolyAz[2] = azM + auZ * huA + avZ * hvA;
  clipPolyAx[3] = axM - auX * huA + avX * hvA; clipPolyAz[3] = azM - auZ * huA + avZ * hvA;
  let count = 4;

  // Clip against 4 half-planes of B: (p - bCenter) · n <= limit
  for (let side = 0; side < 4; side++) {
    if (count === 0) return 0;
    const nx = side === 0 ? buX : side === 1 ? -buX : side === 2 ? bvX : -bvX;
    const nz = side === 0 ? buZ : side === 1 ? -buZ : side === 2 ? bvZ : -bvZ;
    const lim = side < 2 ? huB : hvB;
    const srcX = side % 2 === 0 ? clipPolyAx : clipPolyBx;
    const srcZ = side % 2 === 0 ? clipPolyAz : clipPolyBz;
    const dstX = side % 2 === 0 ? clipPolyBx : clipPolyAx;
    const dstZ = side % 2 === 0 ? clipPolyBz : clipPolyAz;

    let outCnt = 0;
    let px = srcX[count - 1];
    let pz = srcZ[count - 1];
    let pDist = (px - bxM) * nx + (pz - bzM) * nz - lim;
    for (let i = 0; i < count; i++) {
      const cx = srcX[i];
      const cz = srcZ[i];
      const cDist = (cx - bxM) * nx + (cz - bzM) * nz - lim;
      if (cDist <= 0) {
        if (pDist > 0) {
          const t = pDist / (pDist - cDist);
          dstX[outCnt] = px + (cx - px) * t;
          dstZ[outCnt] = pz + (cz - pz) * t;
          outCnt++;
        }
        dstX[outCnt] = cx;
        dstZ[outCnt] = cz;
        outCnt++;
      } else if (pDist <= 0) {
        const t = pDist / (pDist - cDist);
        dstX[outCnt] = px + (cx - px) * t;
        dstZ[outCnt] = pz + (cz - pz) * t;
        outCnt++;
      }
      px = cx;
      pz = cz;
      pDist = cDist;
    }
    count = outCnt;
  }

  if (count < 3) return 0;
  // After 4 passes (`side = 0..3`), result is back in `clipPolyAx / clipPolyAz`
  let area2 = 0;
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    area2 += clipPolyAx[i] * clipPolyAz[j] - clipPolyAz[i] * clipPolyAx[j];
  }
  return 0.5 * Math.abs(area2);
}

/**
 * Law 1C: Evaluates quasi-static concentrated bearing stress (F_n / A_support_eff on narrow < 4 mm
 * edge/corner slivers) and static three-point flexural bending at v ~ 0 across 12L..36L+ tall stacks.
 * Strictly zero allocation.
 */
export function evaluateBlockQuasiStaticStress(
  blockId: number,
  contactsBuf: BlockContactScratchBuffer,
  xM: number,
  yM: number,
  zM: number,
  qw: number,
  qx: number,
  qy: number,
  qz: number,
  priorDamage: number,
  dtSinceLastCrazeSec: number,
  out: BlockQuasiStaticYieldResult = scratchQuasiStaticYield,
  upCoplanarAreaM2: number = 0,
  dnCoplanarAreaM2: number = 0,
  maxPlausibleColumnLoadN: number = Infinity
): BlockQuasiStaticYieldResult {
  out.overloaded = false;
  out.shouldCraze = false;
  out.shouldFracture = false;
  out.effectiveStressMpa = 0;
  out.edgeBearingStressMpa = 0;
  out.flexuralBendingStressMpa = 0;
  out.targetDamage = priorDamage;
  out.localOriginDmX = 0;
  out.localOriginDmY = 0;
  out.localOriginDmZ = 0;
  out.worldHitDmX = xM * 10.0;
  out.worldHitDmY = yM * 10.0;
  out.worldHitDmZ = zM * 10.0;
  out.worldNormalX = 0;
  out.worldNormalY = 1;
  out.worldNormalZ = 0;

  const totalContacts = blockId >= 0 && blockId < contactsBuf.maxBlocks ? contactsBuf.counts[blockId] : 0;
  if (totalContacts < 2) return out;

  // Principal axes e_x, e_y, e_z in world coordinates
  const nxX = 1.0 - 2.0 * (qy * qy + qz * qz);
  const nxY = 2.0 * (qx * qy + qw * qz);
  const nxZ = 2.0 * (qx * qz - qw * qy);

  const nyX = 2.0 * (qx * qy - qw * qz);
  const nyY = 1.0 - 2.0 * (qx * qx + qz * qz);
  const nyZ = 2.0 * (qy * qz + qw * qx);

  const nzX = 2.0 * (qx * qz + qw * qy);
  const nzY = 2.0 * (qy * qz - qw * qx);
  const nzZ = 1.0 - 2.0 * (qx * qx + qy * qy);

  let upCount = 0;
  let dnCount = 0;
  let sumUpFn = 0;
  let sumDnFn = 0;
  let upWeightedX = 0, upWeightedY = 0, upWeightedZ = 0;
  let dnWeightedX = 0, dnWeightedY = 0, dnWeightedZ = 0;

  // Local-X three-point bending accumulators (meters)
  let upNegOuterFn = 0;
  let upPosOuterFn = 0;
  let upMidSpanFn = 0;
  let dnNegOuterFn = 0;
  let dnPosOuterFn = 0;
  let dnMidSpanFn = 0;
  let upOuterAbsXSum = 0;
  let dnOuterAbsXSum = 0;

  const baseBlock = blockId * MAX_CONTACTS_PER_BLOCK * INTERNAL_STRIDE;
  const cd = contactsBuf.data;

  for (let c = 0; c < totalContacts; c++) {
    const b = baseBlock + c * INTERNAL_STRIDE;
    const dist = cd[b + 3];
    if (dist >= 0.0015) continue;
    // Exclude sub-gram Voronoi shard contacts from quasi-static column-weight bearing stress
    if (cd[b + 12] !== 0) continue;
    const fn = cd[b + 4];
    if (fn <= 0.01) continue;

    const px = cd[b + 0];
    const py = cd[b + 1];
    const pz = cd[b + 2];
    const cny = cd[b + 6];

    const rx = px - xM;
    const ry = py - yM;
    const rz = pz - zM;
    const localXM = rx * nxX + ry * nxY + rz * nxZ;

    if (cny > 0.25) {
      scratchUpPtsX[upCount] = px;
      scratchUpPtsZ[upCount] = pz;
      upCount++;
      sumUpFn += fn;
      upWeightedX += px * fn;
      upWeightedY += py * fn;
      upWeightedZ += pz * fn;

      if (localXM < -0.021) {
        upNegOuterFn += fn;
        upOuterAbsXSum += fn * Math.abs(localXM);
      } else if (localXM > 0.021) {
        upPosOuterFn += fn;
        upOuterAbsXSum += fn * Math.abs(localXM);
      } else {
        upMidSpanFn += fn;
      }
    } else if (cny < -0.25) {
      scratchDnPtsX[dnCount] = px;
      scratchDnPtsZ[dnCount] = pz;
      dnCount++;
      sumDnFn += fn;
      dnWeightedX += px * fn;
      dnWeightedY += py * fn;
      dnWeightedZ += pz * fn;

      if (localXM < -0.021) {
        dnNegOuterFn += fn;
        dnOuterAbsXSum += fn * Math.abs(localXM);
      } else if (localXM > 0.021) {
        dnPosOuterFn += fn;
        dnOuterAbsXSum += fn * Math.abs(localXM);
      } else {
        dnMidSpanFn += fn;
      }
    }
  }

  // Require genuine through-thickness multi-course clamping bounded by Newtonian weight + preload above block
  const clampedFn = Math.min(sumUpFn, sumDnFn, maxPlausibleColumnLoadN);
  if (clampedFn < QUASI_STATIC_MIN_CLAMP_N || upCount === 0 || dnCount === 0) {
    return out;
  }

  const upHull = analyzeContactPolygon2D(scratchUpPtsX, scratchUpPtsZ, upCount, xM, zM, scratchUpHullMetrics);
  const dnHull = analyzeContactPolygon2D(scratchDnPtsX, scratchDnPtsZ, dnCount, xM, zM, scratchDnHullMetrics);

  let sigmaEdgeMpa = 0;
  let hitWorldX = xM;
  let hitWorldY = yM;
  let hitWorldZ = zM;
  let hitNormY = 1.0;

  const effUpAreaM2 = Math.max(upHull.effectiveAreaM2, upCoplanarAreaM2);
  const effDnAreaM2 = Math.max(dnHull.effectiveAreaM2, dnCoplanarAreaM2);

  const upIsNarrowSliver =
    (upHull.minorWidthM < QUASI_STATIC_NARROW_WIDTH_M &&
      upCoplanarAreaM2 < QUASI_STATIC_NARROW_AREA_M2) ||
    effUpAreaM2 < QUASI_STATIC_NARROW_AREA_M2;
  if (upIsNarrowSliver) {
    const sUp = (clampedFn / effUpAreaM2) * 1e-6;
    if (sUp > sigmaEdgeMpa) {
      sigmaEdgeMpa = sUp;
      hitWorldX = upWeightedX / sumUpFn;
      hitWorldY = upWeightedY / sumUpFn;
      hitWorldZ = upWeightedZ / sumUpFn;
      hitNormY = 1.0;
    }
  }

  const dnIsNarrowSliver =
    (dnHull.minorWidthM < QUASI_STATIC_NARROW_WIDTH_M &&
      dnCoplanarAreaM2 < QUASI_STATIC_NARROW_AREA_M2) ||
    effDnAreaM2 < QUASI_STATIC_NARROW_AREA_M2;
  if (dnIsNarrowSliver) {
    const sDn = (clampedFn / effDnAreaM2) * 1e-6;
    if (sDn > sigmaEdgeMpa) {
      sigmaEdgeMpa = sDn;
      hitWorldX = dnWeightedX / sumDnFn;
      hitWorldY = dnWeightedY / sumDnFn;
      hitWorldZ = dnWeightedZ / sumDnFn;
      hitNormY = -1.0;
    }
  }

  // Static three-point Euler-Bernoulli flexural stress across clear end-supported span
  let sigmaBendMpa = 0;
  const absNyY = Math.abs(nyY);
  const absNzY = Math.abs(nzY);
  // Section modulus Z_mod = b * h^2 / 6 (flat: 0.025 * 0.015^2 / 6 = 9.375e-7 m^3; side: 1.5625e-6 m^3)
  const sectionModulusM3 = absNyY >= absNzY ? 9.375e-7 : 1.5625e-6;

  if (
    upNegOuterFn > 12.0 &&
    upPosOuterFn > 12.0 &&
    upMidSpanFn < 1.0 &&
    dnMidSpanFn > 24.0 &&
    dnNegOuterFn + dnPosOuterFn < 0.25 * dnMidSpanFn
  ) {
    const fBend = Math.min(dnMidSpanFn, upNegOuterFn + upPosOuterFn, maxPlausibleColumnLoadN);
    const halfSpanM = Math.max(0.022, upOuterAbsXSum / (upNegOuterFn + upPosOuterFn));
    // Clear unsupported span beyond the central 25 mm pillar width
    const clearHalfSpanM = Math.max(0.0, halfSpanM - 0.012);
    const mBendNm = fBend * 0.5 * clearHalfSpanM;
    sigmaBendMpa = (mBendNm / sectionModulusM3) * 1e-6;
    if (sigmaBendMpa > sigmaEdgeMpa) {
      hitWorldX = xM;
      hitWorldY = yM - (nyY >= 0 ? 1 : -1) * EQ_BLOCK_HALF_Y_M;
      hitWorldZ = zM;
      hitNormY = -1.0;
    }
  } else if (
    dnNegOuterFn > 12.0 &&
    dnPosOuterFn > 12.0 &&
    dnMidSpanFn < 1.0 &&
    upMidSpanFn > 24.0 &&
    upNegOuterFn + upPosOuterFn < 0.25 * upMidSpanFn
  ) {
    const fBend = Math.min(upMidSpanFn, dnNegOuterFn + dnPosOuterFn, maxPlausibleColumnLoadN);
    const halfSpanM = Math.max(0.022, dnOuterAbsXSum / (dnNegOuterFn + dnPosOuterFn));
    const clearHalfSpanM = Math.max(0.0, halfSpanM - 0.012);
    const mBendNm = fBend * 0.5 * clearHalfSpanM;
    sigmaBendMpa = (mBendNm / sectionModulusM3) * 1e-6;
    if (sigmaBendMpa > sigmaEdgeMpa) {
      hitWorldX = xM;
      hitWorldY = yM + (nyY >= 0 ? 1 : -1) * EQ_BLOCK_HALF_Y_M;
      hitWorldZ = zM;
      hitNormY = 1.0;
    }
  }

  const effectiveStressMpa = Math.max(sigmaEdgeMpa, sigmaBendMpa);
  out.edgeBearingStressMpa = sigmaEdgeMpa;
  out.flexuralBendingStressMpa = sigmaBendMpa;
  out.effectiveStressMpa = effectiveStressMpa;

  if (effectiveStressMpa < QUASI_STATIC_CRAZE_STRESS_MPA) {
    return out;
  }

  out.overloaded = true;
  out.worldHitDmX = hitWorldX * 10.0;
  out.worldHitDmY = hitWorldY * 10.0;
  out.worldHitDmZ = hitWorldZ * 10.0;
  out.worldNormalX = 0;
  out.worldNormalY = hitNormY;
  out.worldNormalZ = 0;

  const relX = hitWorldX - xM;
  const relY = hitWorldY - yM;
  const relZ = hitWorldZ - zM;
  out.localOriginDmX = Math.max(
    -EQ_BLOCK_HALF_X_DM,
    Math.min(EQ_BLOCK_HALF_X_DM, (relX * nxX + relY * nxY + relZ * nxZ) * 10.0)
  );
  out.localOriginDmY = Math.max(
    -EQ_BLOCK_HALF_Y_DM,
    Math.min(EQ_BLOCK_HALF_Y_DM, (relX * nyX + relY * nyY + relZ * nyZ) * 10.0)
  );
  out.localOriginDmZ = Math.max(
    -EQ_BLOCK_HALF_Z_DM,
    Math.min(EQ_BLOCK_HALF_Z_DM, (relX * nzX + relY * nzY + relZ * nzZ) * 10.0)
  );

  const clampedPrior = Math.max(0.0, Math.min(0.95, priorDamage));
  const crazeRatio = Math.min(
    1.0,
    (effectiveStressMpa - QUASI_STATIC_CRAZE_STRESS_MPA) / QUASI_STATIC_CRAZE_STRESS_MPA
  );
  const computedDmg = Math.min(0.86, 0.42 + 0.38 * crazeRatio);
  out.targetDamage = Math.max(priorDamage, computedDmg);

  const crazedLimitMpa = QUASI_STATIC_CRAZED_FRACTURE_MPA * (1.0 - 0.24 * clampedPrior);
  if (
    effectiveStressMpa >= QUASI_STATIC_INSTANT_FRACTURE_MPA ||
    (priorDamage >= 0.36 && dtSinceLastCrazeSec >= 0.10 && effectiveStressMpa >= crazedLimitMpa)
  ) {
    out.shouldFracture = true;
  } else if (out.targetDamage > priorDamage + 0.06 && dtSinceLastCrazeSec >= 0.30) {
    out.shouldCraze = true;
  }

  return out;
}

/**
 * Exact 15-axis 3D Separating-Axis-Theorem (Gottschalk 1996) OBB proximity check.
 * Returns true iff the surface-to-surface Euclidean gap along every separating axis is <= `maxGapDm`.
 * Zero heap allocation.
 */
export function areBlockObbsWithinGapDm(
  ax: number,
  ay: number,
  az: number,
  aqx: number,
  aqy: number,
  aqz: number,
  aqw: number,
  bx: number,
  by: number,
  bz: number,
  bqx: number,
  bqy: number,
  bqz: number,
  bqw: number,
  maxGapDm: number = 0.035
): boolean {
  const txW = bx - ax;
  const tyW = by - ay;
  const tzW = bz - az;
  // Fast bounding-sphere reject (r_bound = hypot(0.375, 0.075, 0.125) = 0.4023 dm)
  const maxCenterDist = 0.805 + maxGapDm;
  if (txW * txW + tyW * tyW + tzW * tzW > maxCenterDist * maxCenterDist) {
    return false;
  }

  const halfGap = 0.5 * Math.max(0, maxGapDm);
  const a0 = EQ_BLOCK_HALF_X_DM + halfGap;
  const a1 = EQ_BLOCK_HALF_Y_DM + halfGap;
  const a2 = EQ_BLOCK_HALF_Z_DM + halfGap;
  const b0 = a0;
  const b1 = a1;
  const b2 = a2;

  // Rotation columns of A in world
  const a0x = 1.0 - 2.0 * (aqy * aqy + aqz * aqz);
  const a0y = 2.0 * (aqx * aqy + aqw * aqz);
  const a0z = 2.0 * (aqx * aqz - aqw * aqy);

  const a1x = 2.0 * (aqx * aqy - aqw * aqz);
  const a1y = 1.0 - 2.0 * (aqx * aqx + aqz * aqz);
  const a1z = 2.0 * (aqy * aqz + aqw * aqx);

  const a2x = 2.0 * (aqx * aqz + aqw * aqy);
  const a2y = 2.0 * (aqy * aqz - aqw * aqx);
  const a2z = 1.0 - 2.0 * (aqx * aqx + aqy * aqy);

  // Rotation columns of B in world
  const b0x = 1.0 - 2.0 * (bqy * bqy + bqz * bqz);
  const b0y = 2.0 * (bqx * bqy + bqw * bqz);
  const b0z = 2.0 * (bqx * bqz - bqw * bqy);

  const b1x = 2.0 * (bqx * bqy - bqw * bqz);
  const b1y = 1.0 - 2.0 * (bqx * bqx + bqz * bqz);
  const b1z = 2.0 * (bqy * bqz + bqw * bqx);

  const b2x = 2.0 * (bqx * bqz + bqw * bqy);
  const b2y = 2.0 * (bqy * bqz - bqw * bqx);
  const b2z = 1.0 - 2.0 * (bqx * bqx + bqy * bqy);

  // Translation in A's local frame
  const t0 = txW * a0x + tyW * a0y + tzW * a0z;
  const t1 = txW * a1x + tyW * a1y + tzW * a1z;
  const t2 = txW * a2x + tyW * a2y + tzW * a2z;

  // R = A^T * B
  const r00 = a0x * b0x + a0y * b0y + a0z * b0z;
  const r01 = a0x * b1x + a0y * b1y + a0z * b1z;
  const r02 = a0x * b2x + a0y * b2y + a0z * b2z;

  const r10 = a1x * b0x + a1y * b0y + a1z * b0z;
  const r11 = a1x * b1x + a1y * b1y + a1z * b1z;
  const r12 = a1x * b2x + a1y * b2y + a1z * b2z;

  const r20 = a2x * b0x + a2y * b0y + a2z * b0z;
  const r21 = a2x * b1x + a2y * b1y + a2z * b1z;
  const r22 = a2x * b2x + a2y * b2y + a2z * b2z;

  const eps = 1e-6;
  const ar00 = Math.abs(r00) + eps, ar01 = Math.abs(r01) + eps, ar02 = Math.abs(r02) + eps;
  const ar10 = Math.abs(r10) + eps, ar11 = Math.abs(r11) + eps, ar12 = Math.abs(r12) + eps;
  const ar20 = Math.abs(r20) + eps, ar21 = Math.abs(r21) + eps, ar22 = Math.abs(r22) + eps;

  // 3 face axes of A
  if (Math.abs(t0) > a0 + b0 * ar00 + b1 * ar01 + b2 * ar02) return false;
  if (Math.abs(t1) > a1 + b0 * ar10 + b1 * ar11 + b2 * ar12) return false;
  if (Math.abs(t2) > a2 + b0 * ar20 + b1 * ar21 + b2 * ar22) return false;

  // 3 face axes of B
  if (Math.abs(t0 * r00 + t1 * r10 + t2 * r20) > a0 * ar00 + a1 * ar10 + a2 * ar20 + b0) return false;
  if (Math.abs(t0 * r01 + t1 * r11 + t2 * r21) > a0 * ar01 + a1 * ar11 + a2 * ar21 + b1) return false;
  if (Math.abs(t0 * r02 + t1 * r12 + t2 * r22) > a0 * ar02 + a1 * ar12 + a2 * ar22 + b2) return false;

  // 9 edge-edge cross axes
  if (Math.abs(t2 * r10 - t1 * r20) > a1 * ar20 + a2 * ar10 + b1 * ar02 + b2 * ar01) return false;
  if (Math.abs(t2 * r11 - t1 * r21) > a1 * ar21 + a2 * ar11 + b0 * ar02 + b2 * ar00) return false;
  if (Math.abs(t2 * r12 - t1 * r22) > a1 * ar22 + a2 * ar12 + b0 * ar01 + b1 * ar00) return false;

  if (Math.abs(t0 * r20 - t2 * r00) > a0 * ar20 + a2 * ar00 + b1 * ar12 + b2 * ar11) return false;
  if (Math.abs(t0 * r21 - t2 * r01) > a0 * ar21 + a2 * ar01 + b0 * ar12 + b2 * ar10) return false;
  if (Math.abs(t0 * r22 - t2 * r02) > a0 * ar22 + a2 * ar02 + b0 * ar11 + b1 * ar10) return false;

  if (Math.abs(t1 * r00 - t0 * r10) > a0 * ar10 + a1 * ar00 + b1 * ar22 + b2 * ar21) return false;
  if (Math.abs(t1 * r01 - t0 * r11) > a0 * ar11 + a1 * ar01 + b0 * ar22 + b2 * ar20) return false;
  if (Math.abs(t1 * r02 - t0 * r12) > a0 * ar12 + a1 * ar02 + b0 * ar21 + b1 * ar20) return false;

  return true;
}

/**
 * Checks whether a shard centered at `(sx, sy, sz)` (dm) with bounding radius `shardRadDm`
 * lies beneath and within `maxGapDm` of block A's oriented OBB.
 */
export function isShardSupportingBlockObbDm(
  ax: number,
  ay: number,
  az: number,
  aqx: number,
  aqy: number,
  aqz: number,
  aqw: number,
  sx: number,
  sy: number,
  sz: number,
  shardRadDm: number = 0.18,
  maxGapDm: number = 0.035
): boolean {
  const dy = ay - sy;
  if (dy <= -0.18 || dy > 0.56 || sy - shardRadDm > ay + 0.015) return false;

  const dx = sx - ax;
  const dyRel = sy - ay;
  const dz = sz - az;

  const a0x = 1.0 - 2.0 * (aqy * aqy + aqz * aqz);
  const a0y = 2.0 * (aqx * aqy + aqw * aqz);
  const a0z = 2.0 * (aqx * aqz - aqw * aqy);

  const a1x = 2.0 * (aqx * aqy - aqw * aqz);
  const a1y = 1.0 - 2.0 * (aqx * aqx + aqz * aqz);
  const a1z = 2.0 * (aqy * aqz + aqw * aqx);

  const a2x = 2.0 * (aqx * aqz + aqw * aqy);
  const a2y = 2.0 * (aqy * aqz - aqw * aqx);
  const a2z = 1.0 - 2.0 * (aqx * aqx + aqy * aqy);

  const lx = Math.abs(dx * a0x + dyRel * a0y + dz * a0z);
  const ly = Math.abs(dx * a1x + dyRel * a1y + dz * a1z);
  const lz = Math.abs(dx * a2x + dyRel * a2y + dz * a2z);

  const ex = Math.max(0, lx - EQ_BLOCK_HALF_X_DM);
  const ey = Math.max(0, ly - EQ_BLOCK_HALF_Y_DM);
  const ez = Math.max(0, lz - EQ_BLOCK_HALF_Z_DM);
  const limit = shardRadDm + maxGapDm;
  return ex * ex + ey * ey + ez * ez <= limit * limit;
}

/**
 * Oracle Gate B: Coordinate-Free 3D Contact-Hull & Knife-Edge Plausibility Invariant for a settled block.
 */
export function auditBlockEquilibriumGateB(
  blockId: number,
  contactsBuf: BlockContactScratchBuffer,
  xM: number,
  yM: number,
  zM: number,
  qw: number,
  qx: number,
  qy: number,
  qz: number,
  massKg: number,
  totalUpwardConstraintForceN: number = 0
): GateBBlockAudit {
  const totalContacts = blockId >= 0 && blockId < contactsBuf.maxBlocks ? contactsBuf.counts[blockId] : 0;
  const hasTable = blockId >= 0 && blockId < contactsBuf.maxBlocks && contactsBuf.hasTableContact[blockId] !== 0;

  const nxX = 1.0 - 2.0 * (qy * qy + qz * qz);
  const nxY = 2.0 * (qx * qy + qw * qz);
  const nxZ = 2.0 * (qx * qz - qw * qy);
  const nyX = 2.0 * (qx * qy - qw * qz);
  const nyY = 1.0 - 2.0 * (qx * qx + qz * qz);
  const nyZ = 2.0 * (qy * qz + qw * qx);
  const nzX = 2.0 * (qx * qz + qw * qy);
  const nzY = 2.0 * (qy * qz - qw * qx);
  const nzZ = 1.0 - 2.0 * (qx * qx + qy * qy);

  const absNxY = Math.abs(nxY);
  const absNyY = Math.abs(nyY);
  const absNzY = Math.abs(nzY);
  const maxAxisAlignY = Math.max(absNxY, absNyY, absNzY);

  let allCount = 0;
  let upCount = 0;
  let sumPosCy = 0;
  let sumUpFn = 0;
  let sumUpPy = 0;
  const baseBlock = blockId * MAX_CONTACTS_PER_BLOCK * INTERNAL_STRIDE;
  const cd = contactsBuf.data;

  let upAllOnShards = true;
  let minAllPy = Infinity;
  let maxAllPy = -Infinity;
  let loadCount = 0;
  let minLoadPy = Infinity;
  let maxLoadPy = -Infinity;
  let minUpPy = Infinity;
  let maxUpPy = -Infinity;
  let firstSupportGeomId = -2;
  let hasDistinctSupportGeoms = false;
  for (let c = 0; c < totalContacts; c++) {
    const b = baseBlock + c * INTERNAL_STRIDE;
    const dist = cd[b + 3];
    if (dist >= 0.0022) continue;
    const px = cd[b + 0];
    const py = cd[b + 1];
    const pz = cd[b + 2];
    const fn = cd[b + 4];
    const cny = cd[b + 6];
    const fy = cd[b + 9];
    const otherGeom = cd[b + 11];
    if (firstSupportGeomId === -2) {
      firstSupportGeomId = otherGeom;
    } else if (otherGeom !== firstSupportGeomId) {
      hasDistinctSupportGeoms = true;
    }

    scratchPtsX[allCount] = px;
    scratchPtsZ[allCount] = pz;
    allCount++;
    if (py < minAllPy) minAllPy = py;
    if (py > maxAllPy) maxAllPy = py;
    if (fn >= 0.035 || totalContacts <= 2) {
      loadCount++;
      if (py < minLoadPy) minLoadPy = py;
      if (py > maxLoadPy) maxLoadPy = py;
    }

    const upComp = Math.max(fy, cny > 0.15 ? fn * cny : 0);
    if (upComp > 0) sumPosCy += upComp;
    if (cny > 0.20) {
      scratchUpPtsX[upCount] = px;
      scratchUpPtsZ[upCount] = pz;
      upCount++;
      if (py < minUpPy) minUpPy = py;
      if (py > maxUpPy) maxUpPy = py;
      if (cd[b + 12] === 0) {
        upAllOnShards = false;
      }
      const w = Math.max(1e-4, fn);
      sumUpFn += fn;
      sumUpPy += py * w;
    }
  }

  const allHull = analyzeContactPolygon2D(scratchPtsX, scratchPtsZ, allCount, xM, zM, scratchAllHullMetrics);
  const weightN = massKg * 9.81;
  const effUpwardN = Math.max(sumPosCy, totalUpwardConstraintForceN);

  // 1. Upward support invariant: sum(F_cy) >= 0.80 * m * g or table contact
  if (!hasTable && effUpwardN < 0.80 * weightN) {
    return {
      pass: false,
      reason: `Insufficient upward support sum(F_cy)=${effUpwardN.toFixed(3)}N < ${(0.80 * weightN).toFixed(3)}N`,
      upwardSupportN: effUpwardN,
      hasTableContact: hasTable,
      maxAxisAlignY,
      allHullMinorWidthMm: allHull.minorWidthM * 1000,
      dOutMm: 0,
      lTailMm: 0,
      tiltOutDeg: 0,
      tauRestoreNm: 0,
      tauGravityNm: 0,
    };
  }

  // 2. Knife-edge chamfer balance invariant (< 2.4 mm minor principal width across single-elevation contacts when tilted > 10 deg)
  const contactVerticalSpanM =
    loadCount >= 2 ? maxLoadPy - minLoadPy : allCount >= 2 ? maxAllPy - minAllPy : 0;
  const effMaxPropPy = loadCount >= 1 ? maxLoadPy : maxAllPy;
  let minTanProj = Infinity;
  let maxTanProj = -Infinity;
  for (let k = 0; k < allCount; k++) {
    const tp = (scratchPtsX[k] - xM) * allHull.stripTanX + (scratchPtsZ[k] - zM) * allHull.stripTanZ;
    if (tp < minTanProj) minTanProj = tp;
    if (tp > maxTanProj) maxTanProj = tp;
  }
  const isTwoGeomPitchedRamp =
    hasDistinctSupportGeoms &&
    allHull.majorSpanM >= 0.012 &&
    minTanProj < -0.0035 &&
    maxTanProj > 0.0035 &&
    contactVerticalSpanM >= 0.0040;
  const isSingleLineKnifeEdge =
    !isTwoGeomPitchedRamp &&
    (contactVerticalSpanM < 0.0040 || (effMaxPropPy < yM - 0.0025 && contactVerticalSpanM < 0.0120));
  if (
    maxAxisAlignY < KNIFE_EDGE_MAX_AXIS_ALIGN_Y &&
    allHull.minorWidthM < KNIFE_EDGE_MIN_HULL_WIDTH_M &&
    isSingleLineKnifeEdge
  ) {
    return {
      pass: false,
      reason: `Knife-edge chamfer balance maxAxisAlignY=${maxAxisAlignY.toFixed(4)} < 0.985 with hullMinorWidth=${(allHull.minorWidthM * 1000).toFixed(2)}mm < 2.4mm`,
      upwardSupportN: effUpwardN,
      hasTableContact: hasTable,
      maxAxisAlignY,
      allHullMinorWidthMm: allHull.minorWidthM * 1000,
      dOutMm: 0,
      lTailMm: 0,
      tiltOutDeg: 0,
      tauRestoreNm: 0,
      tauGravityNm: 0,
    };
  }

  // 3. Unpropped CoM overhang invariant (d_out > 1.5 mm without balancing counterweight/prop wrench)
  const rCenterTableM = Math.hypot(xM, zM);
  let rMaxTableM = rCenterTableM + EQ_BLOCK_HALF_X_M;
  if (rCenterTableM > 1e-6) {
    const urx = xM / rCenterTableM;
    const urz = zM / rCenterTableM;
    rMaxTableM =
      rCenterTableM +
      Math.abs(nxX * urx + nxZ * urz) * EQ_BLOCK_HALF_X_M +
      Math.abs(nyX * urx + nyZ * urz) * EQ_BLOCK_HALF_Y_M +
      Math.abs(nzX * urx + nzZ * urz) * EQ_BLOCK_HALF_Z_M;
  }
  if (
    upCount > 0 &&
    hasTable &&
    maxAxisAlignY > KNIFE_EDGE_MAX_AXIS_ALIGN_Y &&
    yM < 0.0390 &&
    rMaxTableM <= 0.4295
  ) {
    return {
      pass: true,
      reason: 'ok',
      upwardSupportN: effUpwardN,
      hasTableContact: hasTable,
      maxAxisAlignY,
      allHullMinorWidthMm: allHull.minorWidthM * 1000,
      dOutMm: 0,
      lTailMm: 0,
      tiltOutDeg: 0,
      tauRestoreNm: 0,
      tauGravityNm: 0,
    };
  }
  const upHull = analyzeContactPolygon2D(scratchUpPtsX, scratchUpPtsZ, upCount, xM, zM, scratchUpHullMetrics);
  const dOut = upHull.dOutM;
  const isFlatShardBed = upCount > 0 && upAllOnShards && maxAxisAlignY > 0.960 && yM < 0.035 && dOut < 0.016;
  if (!isFlatShardBed && (upCount === 0 || dOut > 0.0015)) {
    const ux = upHull.outUnitX;
    const uz = upHull.outUnitZ;
    const rSpanM =
      Math.abs(nxX * ux + nxZ * uz) * EQ_BLOCK_HALF_X_M +
      Math.abs(nyX * ux + nyZ * uz) * EQ_BLOCK_HALF_Y_M +
      Math.abs(nzX * ux + nzZ * uz) * EQ_BLOCK_HALF_Z_M;
    const lTailM = upCount === 0 ? -rSpanM : rSpanM - dOut;

    let upFaceX = nyX;
    let upFaceZ = nyZ;
    if (absNyY >= absNxY && absNyY >= absNzY) {
      const s = nyY >= 0 ? 1.0 : -1.0;
      upFaceX = s * nyX;
      upFaceZ = s * nyZ;
    } else if (absNzY >= absNxY) {
      const s = nzY >= 0 ? 1.0 : -1.0;
      upFaceX = s * nzX;
      upFaceZ = s * nzZ;
    } else {
      const s = nxY >= 0 ? 1.0 : -1.0;
      upFaceX = s * nxX;
      upFaceZ = s * nxZ;
    }
    const tiltOutDeg =
      (Math.asin(Math.max(-1.0, Math.min(1.0, upFaceX * ux + upFaceZ * uz))) * 180.0) / Math.PI;

    const qxFulc = upHull.fulcrumX;
    const qzFulc = upHull.fulcrumZ;
    let qyFulc = sumUpFn > 1e-5 ? sumUpPy / sumUpFn : yM - EQ_BLOCK_HALF_Y_M;
    let bestFulcDistSq = Infinity;
    for (let c = 0; c < totalContacts; c++) {
      const b = baseBlock + c * INTERNAL_STRIDE;
      if (cd[b + 3] >= 0.0022 || cd[b + 6] <= 0.20) continue;
      const dxF = cd[b + 0] - qxFulc;
      const dzF = cd[b + 2] - qzFulc;
      const dSq = dxF * dxF + dzF * dzF;
      if (dSq < bestFulcDistSq) {
        bestFulcDistSq = dSq;
        qyFulc = cd[b + 1];
      }
    }

    let tauRestorePosNm = 0;
    let tauRestoreAlgNm = 0;
    let inboardRoofFn = 0;
    let meanRoofNx = 0;
    let meanRoofNy = 0;
    let meanRoofNz = 0;
    let roofWeightSum = 0;
    let hasOutboardPropBeyondCom = false;
    let hasInboardPropBehindCom = false;
    let meanSupNx = 0;
    let meanSupNy = 0;
    let meanSupNz = 0;
    for (let c = 0; c < totalContacts; c++) {
      const b = baseBlock + c * INTERNAL_STRIDE;
      const dist = cd[b + 3];
      if (dist >= 0.0022) continue;
      const px = cd[b + 0];
      const py = cd[b + 1];
      const pz = cd[b + 2];
      const fn = cd[b + 4];
      const cnx = cd[b + 5];
      const cny = cd[b + 6];
      const cnz = cd[b + 7];
      const fx = cd[b + 8];
      const fy = cd[b + 9];
      const fz = cd[b + 10];
      if (cny > 0.20) {
        const w = Math.max(1e-4, fn);
        meanSupNx += cnx * w;
        meanSupNy += cny * w;
        meanSupNz += cnz * w;
      }
      const sFromCom = (px - xM) * ux + (pz - zM) * uz;
      if (cny > -0.35 && fn >= 0.03 * weightN && sFromCom > 0.0015) {
        hasOutboardPropBeyondCom = true;
      }
      if (cny > -0.35 && fn >= 0.03 * weightN && sFromCom < -0.0015) {
        hasInboardPropBehindCom = true;
      }
      const sAlong = (px - qxFulc) * ux + (pz - qzFulc) * uz;
      const ry = py - qyFulc;
      if (cny < -0.20 && cd[b + 12] === 0) {
        const rw = Math.max(1e-4, fn);
        meanRoofNx += cnx * rw;
        meanRoofNy += cny * rw;
        meanRoofNz += cnz * rw;
        roofWeightSum += rw;
        if (fy < 0 && sAlong < -0.0015) {
          inboardRoofFn += -fy;
        }
      }
      const exactMom = fy * sAlong - ry * (fx * ux + fz * uz);
      tauRestoreAlgNm += exactMom;
      if (Math.abs(sAlong) > 0.0012 || Math.abs(ry) > 0.0015) {
        if (exactMom > 0) {
          tauRestorePosNm += exactMom;
        } else if (
          rSpanM >= 0.020 &&
          lTailM <= 0.0220 &&
          dOut >= 0.0150 &&
          cny < -0.85 &&
          fy < 0 &&
          sAlong > 0.0015
        ) {
          tauRestorePosNm += fy * sAlong;
        }
      }
    }
    const tauRestoreNm = Math.max(tauRestorePosNm, tauRestoreAlgNm);

    const supLen = Math.hypot(meanSupNx, meanSupNy, meanSupNz);
    const supNy = supLen > 1e-6 ? meanSupNy / supLen : 1.0;
    const roofLen = Math.hypot(meanRoofNx, meanRoofNy, meanRoofNz);
    const rNx = roofWeightSum > 1e-6 && roofLen > 1e-6 ? meanRoofNx / roofLen : 0.0;
    const rNy = roofWeightSum > 1e-6 && roofLen > 1e-6 ? meanRoofNy / roofLen : -1.0;
    const rNz = roofWeightSum > 1e-6 && roofLen > 1e-6 ? meanRoofNz / roofLen : 0.0;
    const slotWedgeTiltDeg =
      (Math.asin(Math.max(-1.0, Math.min(1.0, (upFaceX + rNx) * ux + (upFaceZ + rNz) * uz))) *
        180.0) /
      Math.PI;
    const effBevelEscapeTailM = rSpanM >= 0.020 ? BEVEL_ESCAPE_TAIL_LIMIT_M : 0.0020;
    const isDeepTiltedSlotBevelWedge =
      rSpanM >= 0.020 &&
      dOut >= 0.0145 &&
      lTailM <= 0.0220 &&
      tiltOutDeg > 0.15 &&
      rNy < -0.994 &&
      slotWedgeTiltDeg >= 1.05;
    const isSlotBevelEscape =
      maxAxisAlignY > 0.960 &&
      inboardRoofFn > 0.04 &&
      supNy > 0.985 &&
      rNy < -0.990 &&
      (lTailM <= effBevelEscapeTailM ||
        slotWedgeTiltDeg >= BEVEL_SLIP_TILT_LIMIT_DEG ||
        isDeepTiltedSlotBevelWedge);
    const isProppedTwoEndBridge =
      (hasOutboardPropBeyondCom && (hasInboardPropBehindCom || allHull.dOutM <= 0.0015)) ||
      (!isSlotBevelEscape && allHull.dOutM <= 0.0015 && (hasTable || effUpwardN >= 0.80 * weightN));

    const effDOut = upCount === 0 ? 0.010 : dOut;
    const effNetOverhangM = upCount === 0 ? 0.010 : Math.max(0.0, dOut - EQ_BEVEL_RADIUS_M);
    const tauGravityNm = weightN * effDOut;
    const tauCondimMarginNm = upCount === 0 ? 0.0 : 0.0003 * Math.min(2.0, effUpwardN);
    const tauRequiredNm = 0.80 * weightN * effNetOverhangM;
    if (
      !isProppedTwoEndBridge &&
      (upCount === 0 || tauRestoreNm + tauCondimMarginNm < tauRequiredNm || isSlotBevelEscape)
    ) {
      return {
        pass: false,
        reason: isSlotBevelEscape
          ? `Slot/bevel overhang escape d_out=${(effDOut * 1000).toFixed(2)}mm, L_tail=${(lTailM * 1000).toFixed(2)}mm, slotTilt=${slotWedgeTiltDeg.toFixed(2)}deg`
          : `Unpropped CoM overhang d_out=${(effDOut * 1000).toFixed(2)}mm > 1.5mm with tauRestore=${tauRestoreNm.toFixed(5)}Nm < ${(0.80 * tauGravityNm).toFixed(5)}Nm`,
        upwardSupportN: sumPosCy,
        hasTableContact: hasTable,
        maxAxisAlignY,
        allHullMinorWidthMm: allHull.minorWidthM * 1000,
        dOutMm: effDOut * 1000,
        lTailMm: lTailM * 1000,
        tiltOutDeg,
        tauRestoreNm,
        tauGravityNm,
      };
    }
  }

  return {
    pass: true,
    reason: 'ok',
    upwardSupportN: sumPosCy,
    hasTableContact: hasTable,
    maxAxisAlignY,
    allHullMinorWidthMm: allHull.minorWidthM * 1000,
    dOutMm: Math.max(0, dOut * 1000),
    lTailMm: 0,
    tiltOutDeg: 0,
    tauRestoreNm: 0,
    tauGravityNm: 0,
  };
}

export interface EquilibriumBlockRef {
  readonly id: number;
  readonly bodyHandle: number;
  readonly colliderHandle: number;
  readonly shattered: boolean;
  readonly extracted: boolean;
}

export interface EquilibriumShardRef {
  readonly slotIndex: number;
  readonly bodyHandle: number;
  readonly colliderHandle: number;
}

/**
 * Zero-GC active-scene soft-contact elastic potential accumulator.
 * Replaces `d.energy[0]` (`mj_energyPos`), which contains `0.0 J` of soft-contact penalty energy
 * and sums `-1,319 J` across `-50 m` parked pool bodies.
 */
export class ContactPotentialAccumulator {
  public cachedPot = new Float64Array(1024);
  public nextPot = new Float64Array(1024);
  public maxSupPen = new Float64Array(1024);
  public sigTime = -1;
  public sigNcon = -1;
  public sigNefc = -1;
  public sigEfc0 = 0;

  private readonly potPairHashKey = new Int32Array(4096);
  private readonly potPairHashStamp = new Uint32Array(4096);
  private readonly potPairPrevUPair = new Float64Array(4096);
  private potPairHashEpoch = 0;

  public reset(): void {
    this.cachedPot.fill(0);
    this.sigTime = -1;
  }

  public invalidateSignature(): void {
    this.sigTime = -1;
  }

  public zeroBody(bodyId: number): void {
    if (bodyId >= 0 && bodyId < this.cachedPot.length) {
      this.cachedPot[bodyId] = 0;
    }
  }

  public ensureCapacity(nbody: number): void {
    if (this.cachedPot.length < nbody) {
      const nextCap = Math.max(nbody, 1024);
      const prev = this.cachedPot;
      this.cachedPot = new Float64Array(nextCap);
      this.cachedPot.set(prev);
      this.nextPot = new Float64Array(nextCap);
      this.maxSupPen = new Float64Array(nextCap);
    }
  }

  public begin(nbody: number): void {
    this.ensureCapacity(nbody);
    this.nextPot.fill(0, 0, nbody);
    this.maxSupPen.fill(0, 0, nbody);
    let potEpoch = (this.potPairHashEpoch + 1) >>> 0;
    if (potEpoch === 0) {
      this.potPairHashStamp.fill(0);
      potEpoch = 1;
    }
    this.potPairHashEpoch = potEpoch;
  }

  public recordPair(
    g1: number,
    g2: number,
    penM: number,
    r0y: number,
    geomBodyId: ArrayLike<number>,
    bodyMass: ArrayLike<number>
  ): void {
    const b1 = geomBodyId[g1];
    const b2 = geomBodyId[g2];
    const absRy = Math.abs(r0y);
    if (absRy > 0.15) {
      const bSup = r0y >= 0 ? b2 : b1;
      if (bSup >= 2) {
        const vPen = penM * absRy;
        if (vPen > this.maxSupPen[bSup]) {
          this.maxSupPen[bSup] = vPen;
        }
      }
    }
    const horizWeight = Math.max(0, 1.0 - r0y * r0y);
    if (horizWeight <= 1e-4) return;
    const m1 = b1 >= 2 ? bodyMass[b1] : 0;
    const m2 = b2 >= 2 ? bodyMass[b2] : 0;
    const mEff = m1 > 0 && m2 > 0 ? (m1 * m2) / (m1 + m2) : m1 > 0 ? m1 : m2;
    if (!(mEff > 0)) return;
    const penEff = Math.min(penM, 0.0008);
    const uHoriz = 0.5 * mEff * horizWeight * 12000.0 * penEff * penEff;
    const loG = g1 < g2 ? g1 : g2;
    const hiG = g1 < g2 ? g2 : g1;
    const pKey = loG * 2048 + hiG;
    const potEpoch = this.potPairHashEpoch;
    const potHKey = this.potPairHashKey;
    const potHStamp = this.potPairHashStamp;
    const potPrevU = this.potPairPrevUPair;
    let pSlot = (Math.imul(pKey, 0x9e3779b1) >>> 20) & 4095;
    let uDelta = uHoriz;
    let seenPair = false;
    while (potHStamp[pSlot] === potEpoch) {
      if (potHKey[pSlot] === pKey) {
        seenPair = true;
        if (uHoriz > potPrevU[pSlot]) {
          uDelta = uHoriz - potPrevU[pSlot];
          potPrevU[pSlot] = uHoriz;
        } else {
          uDelta = 0;
        }
        break;
      }
      pSlot = (pSlot + 1) & 4095;
    }
    if (!seenPair) {
      potHStamp[pSlot] = potEpoch;
      potHKey[pSlot] = pKey;
      potPrevU[pSlot] = uHoriz;
    }
    if (uDelta > 0) {
      const nextPot = this.nextPot;
      if (b1 >= 2 && b2 >= 2) {
        nextPot[b1] += 0.5 * uDelta;
        nextPot[b2] += 0.5 * uDelta;
      } else if (b1 >= 2) {
        nextPot[b1] += uDelta;
      } else if (b2 >= 2) {
        nextPot[b2] += uDelta;
      }
    }
  }

  public commit(
    blocks: readonly EquilibriumBlockRef[],
    shards: readonly EquilibriumShardRef[],
    bodyTreeId: ArrayLike<number>,
    treeAsleep: ArrayLike<number>,
    bodyMass: ArrayLike<number>,
    sigTime: number,
    sigNcon: number,
    sigNefc: number,
    sigEfc0: number
  ): void {
    const cachedPot = this.cachedPot;
    const nextPot = this.nextPot;
    const maxSupPen = this.maxSupPen;
    const hasShards = shards.length > 0;

    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.shattered || b.extracted) continue;
      const bodyId = b.bodyHandle;
      const tid = bodyTreeId[bodyId];
      if (tid < 0 || treeAsleep[tid] < 0) {
        const vPen = maxSupPen[bodyId];
        let uVert = 0;
        if (vPen > 0) {
          const mB = bodyMass[bodyId];
          if (hasShards) {
            const penEff = Math.min(vPen, 0.0008);
            uVert = mB * 9.81 * vPen + 0.5 * mB * 12000.0 * penEff * penEff;
          } else {
            const penEff = Math.min(vPen, 0.0012);
            uVert = 0.5 * mB * 23668.6 * penEff * penEff;
          }
        }
        cachedPot[bodyId] = nextPot[bodyId] + uVert;
      }
    }
    for (let i = 0; i < shards.length; i++) {
      const s = shards[i];
      const bodyId = s.bodyHandle;
      const tid = bodyTreeId[bodyId];
      if (tid < 0 || treeAsleep[tid] < 0) {
        const vPen = maxSupPen[bodyId];
        let uVert = 0;
        if (vPen > 0) {
          const mS = bodyMass[bodyId];
          const penEff = Math.min(vPen, 0.0008);
          uVert = mS * 9.81 * vPen + 0.5 * mS * 12000.0 * penEff * penEff;
        }
        cachedPot[bodyId] = nextPot[bodyId] + uVert;
      }
    }

    this.sigTime = sigTime;
    this.sigNcon = sigNcon;
    this.sigNefc = sigNefc;
    this.sigEfc0 = sigEfc0;
  }
}

/**
 * Populates `BlockContactScratchBuffer` directly from `MjData` + `MjModel` active contacts with
 * hoisted typed-array views and zero heap allocation (disposing every Embind `mjContact` wrapper).
 */
export function populateBlockContactScratchFromMjData<TBlock extends EquilibriumBlockRef>(
  d: MjData,
  m: MjModel,
  buf: BlockContactScratchBuffer,
  blockCount: number,
  map: MujocoTowerIndexMap,
  bodyIdToBlock: ArrayLike<TBlock | undefined>,
  minShardGid: number,
  maxShardGid: number,
  activeBlockId: number | null,
  blockUnbalancedMask: Uint8Array,
  shardTouchingLiveBlockMask: Uint8Array | null,
  shardSupportingElevatedBlockMask: Uint8Array | null,
  updateLiveShardMask: boolean = false,
  maxDistM: number = 0.0015,
  vcQpos?: Float64Array,
  vcQvel?: Float64Array,
  vcGeomBodyId?: Int32Array
): void {
  buf.clear(blockCount);
  if (updateLiveShardMask) {
    shardTouchingLiveBlockMask?.fill(0);
    shardSupportingElevatedBlockMask?.fill(0);
  }
  const ncon = d.ncon;
  if (ncon === 0) return;

  const contacts = d.contact;
  const efc = d.efc_force;
  const qpos = vcQpos ?? d.qpos;
  const qvel = vcQvel ?? d.qvel;
  const geomBodyId = vcGeomBodyId ?? m.geom_bodyid;
  const parkingGid = map.parkingShelfGeomId;
  const parkingNegGid = map.parkingShelfNegGeomId;
  const firstDynGid = map.blockGeomIds[0] ?? 4;
  const tableGid = map.tableGeomId;
  const shardTableGid = map.shardTableGeomId;
  const blockQposAdr = map.blockQposAdr;
  const blockDofAdr = map.blockDofAdr;
  const shardQposAdr = map.shardQposAdr;
  const maxContactRadiusSq = 0.052 * 0.052;

  for (let i = 0; i < ncon; i++) {
    const c = contacts.get(i);
    if (!c) continue;
    const g1 = c.geom1;
    const g2 = c.geom2;
    const nonBlk1 = g1 < firstDynGid || (g1 >= minShardGid && g1 <= maxShardGid);
    const nonBlk2 = g2 < firstDynGid || (g2 >= minShardGid && g2 <= maxShardGid);
    if (
      (nonBlk1 && nonBlk2) ||
      g1 === parkingGid ||
      g2 === parkingGid ||
      g1 === parkingNegGid ||
      g2 === parkingNegGid
    ) {
      c.delete();
      continue;
    }
    const dist = c.dist;
    if (dist >= maxDistM) {
      c.delete();
      continue;
    }
    const b1 = geomBodyId[g1];
    const b2 = geomBodyId[g2];
    const blk1 = !nonBlk1 && b1 >= 2 ? bodyIdToBlock[b1] : undefined;
    const blk2 = !nonBlk2 && b2 >= 2 ? bodyIdToBlock[b2] : undefined;
    if (
      (!blk1 || blk1.shattered || blk1.extracted) &&
      (!blk2 || blk2.shattered || blk2.extracted)
    ) {
      c.delete();
      continue;
    }

    const adr = c.efc_address;
    const fn = adr >= 0 ? Math.max(0, efc[adr]) : 0;
    const hasTangents = adr >= 0 && c.dim >= 3;
    const ft1 = hasTangents ? efc[adr + 1] : 0;
    const ft2 = hasTangents ? efc[adr + 2] : 0;
    const cPos = c.pos;
    const px = cPos[0];
    const py = cPos[1];
    const pz = cPos[2];
    const f = c.frame;
    const r0x = f[0];
    const r0y = f[1];
    const r0z = f[2];
    const rawFx = fn * r0x + ft1 * f[3] + ft2 * f[6];
    const rawFy = fn * r0y + ft1 * f[4] + ft2 * f[7];
    const rawFz = fn * r0z + ft1 * f[5] + ft2 * f[8];

    if (blk1 && !blk1.shattered && !blk1.extracted) {
      const b1Qa = blockQposAdr[blk1.id];
      const dx1 = px - qpos[b1Qa];
      const dy1 = py - qpos[b1Qa + 1];
      const dz1 = pz - qpos[b1Qa + 2];
      if (dx1 * dx1 + dy1 * dy1 + dz1 * dz1 <= maxContactRadiusSq) {
        const g2IsShard = g2 >= minShardGid && g2 <= maxShardGid;
        if (updateLiveShardMask && g2IsShard && fn > 0.02) {
          const shdSlot = g2 - minShardGid;
          const shdQa = shardQposAdr[shdSlot];
          const b1Dof = blockDofAdr[blk1.id];
          const vx1 = qvel[b1Dof + 0];
          const vy1 = qvel[b1Dof + 1];
          const vz1 = qvel[b1Dof + 2];
          const wx1 = qvel[b1Dof + 3];
          const wy1 = qvel[b1Dof + 4];
          const wz1 = qvel[b1Dof + 5];
          const b1Unbal =
            blk1.id === activeBlockId ||
            (blk1.id < blockUnbalancedMask.length && blockUnbalancedMask[blk1.id] !== 0);
          if (
            shardSupportingElevatedBlockMask &&
            b1Unbal &&
            -r0y > 0.45 &&
            qpos[b1Qa + 1] > 0.0092 &&
            py < qpos[b1Qa + 1] - 0.002 &&
            qpos[shdQa + 1] < qpos[b1Qa + 1] - 0.002
          ) {
            shardSupportingElevatedBlockMask[shdSlot] = 1;
          }
          const b1Active =
            b1Unbal ||
            vx1 * vx1 + vy1 * vy1 + vz1 * vz1 >= 0.0004 ||
            wx1 * wx1 + wy1 * wy1 + wz1 * wz1 >= 0.0784;
          if (b1Active && shardTouchingLiveBlockMask) {
            shardTouchingLiveBlockMask[shdSlot] = 1;
          }
        }
        buf.addContact(
          blk1.id,
          px,
          py,
          pz,
          dist,
          fn,
          -r0x,
          -r0y,
          -r0z,
          -rawFx,
          -rawFy,
          -rawFz,
          g2 === tableGid || g2 === shardTableGid,
          g2,
          g2IsShard
        );
      }
    }
    if (blk2 && !blk2.shattered && !blk2.extracted) {
      const b2Qa = blockQposAdr[blk2.id];
      const dx2 = px - qpos[b2Qa];
      const dy2 = py - qpos[b2Qa + 1];
      const dz2 = pz - qpos[b2Qa + 2];
      if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 <= maxContactRadiusSq) {
        const g1IsShard = g1 >= minShardGid && g1 <= maxShardGid;
        if (updateLiveShardMask && g1IsShard && fn > 0.02) {
          const shdSlot = g1 - minShardGid;
          const shdQa = shardQposAdr[shdSlot];
          const b2Dof = blockDofAdr[blk2.id];
          const vx2 = qvel[b2Dof + 0];
          const vy2 = qvel[b2Dof + 1];
          const vz2 = qvel[b2Dof + 2];
          const wx2 = qvel[b2Dof + 3];
          const wy2 = qvel[b2Dof + 4];
          const wz2 = qvel[b2Dof + 5];
          const b2Unbal =
            blk2.id === activeBlockId ||
            (blk2.id < blockUnbalancedMask.length && blockUnbalancedMask[blk2.id] !== 0);
          if (
            shardSupportingElevatedBlockMask &&
            b2Unbal &&
            r0y > 0.45 &&
            qpos[b2Qa + 1] > 0.0092 &&
            py < qpos[b2Qa + 1] - 0.002 &&
            qpos[shdQa + 1] < qpos[b2Qa + 1] - 0.002
          ) {
            shardSupportingElevatedBlockMask[shdSlot] = 1;
          }
          const b2Active =
            b2Unbal ||
            vx2 * vx2 + vy2 * vy2 + vz2 * vz2 >= 0.0004 ||
            wx2 * wx2 + wy2 * wy2 + wz2 * wz2 >= 0.0784;
          if (b2Active && shardTouchingLiveBlockMask) {
            shardTouchingLiveBlockMask[shdSlot] = 1;
          }
        }
        buf.addContact(
          blk2.id,
          px,
          py,
          pz,
          dist,
          fn,
          r0x,
          r0y,
          r0z,
          rawFx,
          rawFy,
          rawFz,
          g1 === tableGid || g1 === shardTableGid,
          g1,
          g1IsShard
        );
      }
    }
    c.delete();
  }
  contacts.delete();
}

const qsSeenUpGeoms = new Int32Array(16);
const qsSeenDnGeoms = new Int32Array(16);
export const scratchCoplanarFaceAreas = new Float64Array(2);

/**
 * Sums upward and downward coplanar bevel-inset face overlap areas (`m^2`) for `blockId` across
 * all distinct supporting/loading blocks & table in `contactsBuf` into `outAreas[0..1]`.
 */
export function sumCoplanarBlockFaceAreasForBlock<TBlock extends EquilibriumBlockRef>(
  blockId: number,
  qa: number,
  contactsBuf: BlockContactScratchBuffer,
  qpos: ArrayLike<number>,
  geomBodyId: ArrayLike<number>,
  bodyIdToBlock: ArrayLike<TBlock | undefined>,
  blockQposAdr: ArrayLike<number>,
  outAreas: Float64Array = scratchCoplanarFaceAreas
): Float64Array {
  let upCoplanarAreaM2 = 0;
  let dnCoplanarAreaM2 = 0;
  const cntC = blockId >= 0 && blockId < contactsBuf.maxBlocks ? contactsBuf.counts[blockId] : 0;
  if (cntC >= 2) {
    const cd = contactsBuf.data;
    const baseC = blockId * MAX_CONTACTS_PER_BLOCK * INTERNAL_STRIDE;
    let seenUpCount = 0;
    let seenDnCount = 0;
    const qwA = qpos[qa + 3];
    const qxA = qpos[qa + 4];
    const qyA = qpos[qa + 5];
    const qzA = qpos[qa + 6];
    const maxAyA = Math.max(
      Math.abs(2.0 * (qxA * qyA + qwA * qzA)),
      Math.abs(1.0 - 2.0 * (qxA * qxA + qzA * qzA)),
      Math.abs(2.0 * (qyA * qzA - qwA * qxA))
    );
    for (let ci = 0; ci < cntC; ci++) {
      const off = baseC + ci * INTERNAL_STRIDE;
      if (cd[off + 3] >= 0.0015 || cd[off + 12] !== 0 || cd[off + 4] <= 0.01) continue;
      const cny = cd[off + 6];
      const otherGid = cd[off + 11] | 0;
      const isTbl = cd[off + 13] !== 0;
      if (cny > 0.25) {
        let dup = false;
        for (let s = 0; s < seenUpCount; s++) {
          if (qsSeenUpGeoms[s] === otherGid) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
        if (seenUpCount < qsSeenUpGeoms.length) qsSeenUpGeoms[seenUpCount++] = otherGid;
        if (isTbl && maxAyA > 0.992) {
          upCoplanarAreaM2 += 1.6e-3;
        } else if (!isTbl && otherGid >= 0 && otherGid < geomBodyId.length) {
          const ob = bodyIdToBlock[geomBodyId[otherGid]];
          if (ob && !ob.shattered && !ob.extracted) {
            const oqa = blockQposAdr[ob.id];
            upCoplanarAreaM2 += computeCoplanarBlockFaceOverlapAreaM2(
              qpos[qa + 0],
              qpos[qa + 2],
              qwA,
              qxA,
              qyA,
              qzA,
              qpos[oqa + 0],
              qpos[oqa + 2],
              qpos[oqa + 3],
              qpos[oqa + 4],
              qpos[oqa + 5],
              qpos[oqa + 6]
            );
          }
        }
      } else if (cny < -0.25) {
        let dup = false;
        for (let s = 0; s < seenDnCount; s++) {
          if (qsSeenDnGeoms[s] === otherGid) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
        if (seenDnCount < qsSeenDnGeoms.length) qsSeenDnGeoms[seenDnCount++] = otherGid;
        if (!isTbl && otherGid >= 0 && otherGid < geomBodyId.length) {
          const ob = bodyIdToBlock[geomBodyId[otherGid]];
          if (ob && !ob.shattered && !ob.extracted) {
            const oqa = blockQposAdr[ob.id];
            dnCoplanarAreaM2 += computeCoplanarBlockFaceOverlapAreaM2(
              qpos[qa + 0],
              qpos[qa + 2],
              qwA,
              qxA,
              qyA,
              qzA,
              qpos[oqa + 0],
              qpos[oqa + 2],
              qpos[oqa + 3],
              qpos[oqa + 4],
              qpos[oqa + 5],
              qpos[oqa + 6]
            );
          }
        }
      }
    }
  }
  outAreas[0] = upCoplanarAreaM2;
  outAreas[1] = dnCoplanarAreaM2;
  return outAreas;
}

/**
 * Reusable zero-GC scratch arrays for `detectSleepingUnbalancedBlocksInScratch` and
 * `evaluateTwoGateSettledPlausibilityInScratch`.
 */
export class ScratchEquilibriumOracleBuffers {
  public contactBuf = new BlockContactScratchBuffer(264);
  public savedPriority = new Int32Array(0);
  public savedFriction = new Float64Array(0);
  public savedDofDamping = new Float64Array(0);
  public savedContype = new Int32Array(0);
  public savedConaffinity = new Int32Array(0);
  public savedBodyContype = new Int32Array(0);
  public savedBodyConaffinity = new Int32Array(0);
  public nearShardSlotMask = new Uint8Array(512);
  public nearShardDofAdrs: number[] = [];
  public candidateBlockIds: number[] = [];
  public toWakeBlockIds: number[] = [];
  public startPose = new Float64Array(256 * 7);
  public step6Pose = new Float64Array(256 * 7);
  public wakeFlags = new Uint8Array(256);
  public gateBPending = new Uint8Array(256);
  public peakDyMm = new Float64Array(256);
  public peakDPosMm = new Float64Array(256);
  public peakDThetaDeg = new Float64Array(256);

  public ensureCapacity(m: MjModel, blockCount: number, maxShards: number = 512): void {
    const ngeom = m.ngeom;
    const nbody = m.nbody;
    const nv = m.nv;
    if (this.savedPriority.length < ngeom) {
      this.savedPriority = new Int32Array(ngeom);
      this.savedFriction = new Float64Array(ngeom * 3);
      this.savedContype = new Int32Array(ngeom);
      this.savedConaffinity = new Int32Array(ngeom);
    }
    if (this.savedBodyContype.length < nbody) {
      this.savedBodyContype = new Int32Array(nbody);
      this.savedBodyConaffinity = new Int32Array(nbody);
    }
    if (this.savedDofDamping.length < nv) {
      this.savedDofDamping = new Float64Array(nv);
    }
    if (this.nearShardSlotMask.length < maxShards) {
      this.nearShardSlotMask = new Uint8Array(maxShards);
    }
    if (this.contactBuf.maxBlocks < blockCount + 8) {
      this.contactBuf = new BlockContactScratchBuffer(Math.max(264, blockCount + 16));
    }
    if (this.wakeFlags.length < blockCount) {
      const cap = Math.max(256, blockCount + 16);
      this.startPose = new Float64Array(cap * 7);
      this.step6Pose = new Float64Array(cap * 7);
      this.wakeFlags = new Uint8Array(cap);
      this.gateBPending = new Uint8Array(cap);
      this.peakDyMm = new Float64Array(cap);
      this.peakDPosMm = new Float64Array(cap);
      this.peakDThetaDeg = new Float64Array(cap);
    }
  }
}

export interface TwoGatePlausibilityReport {
  pass: boolean;
  gateAPass: boolean;
  gateBPass: boolean;
  maxAbsDyMm: number;
  maxDPosMm: number;
  maxDThetaDeg: number;
  gateAViolations: Array<{
    blockId: number;
    absDyMm: number;
    dPosMm: number;
    dThetaDeg: number;
    posMm: [number, number, number];
  }>;
  gateBViolations: Array<{ blockId: number; audit: GateBBlockAudit }>;
}

/**
 * Non-destructively clones `(m, d)` into `sd` on tree sleep to rebuild sleeping contact manifolds
 * and detect any intact block resting on a knife-edge chamfer, unpropped overhang, or soft cascade
 * contact. Zero live `MjData`/`MjModel` mutation (except optional `onRelieveEmbeddedShards`).
 */
export function detectSleepingUnbalancedBlocksInScratch<TBlock extends EquilibriumBlockRef>(params: {
  mj: MainModule;
  m: MjModel;
  d: MjData;
  sd: MjData;
  bufs: ScratchEquilibriumOracleBuffers;
  map: MujocoTowerIndexMap;
  blocks: readonly TBlock[];
  shards: readonly EquilibriumShardRef[];
  bodyIdToBlock: ArrayLike<TBlock | undefined>;
  minShardGid: number;
  maxShardGid: number;
  offRimBlockConaffinity: number;
  tableRadiusM: number;
  glassFriction: readonly [number, number, number];
  sleepTolerance: number;
  postSleepSweepPassCount: number;
  blockEqWakeRetries: Uint8Array;
  attachThreadPool: (sd: MjData) => void;
  onRelieveEmbeddedShards?: (sd: MjData) => boolean;
}): number[] {
  const {
    mj,
    m,
    d,
    sd,
    bufs,
    map,
    blocks,
    shards,
    bodyIdToBlock,
    minShardGid,
    maxShardGid,
    offRimBlockConaffinity,
    tableRadiusM,
    glassFriction,
    sleepTolerance,
    postSleepSweepPassCount,
    blockEqWakeRetries,
    attachThreadPool,
    onRelieveEmbeddedShards,
  } = params;

  bufs.ensureCapacity(m, blocks.length);
  const toWakeBlockIds = bufs.toWakeBlockIds;
  toWakeBlockIds.length = 0;

  const ngeom = m.ngeom;
  const nbody = m.nbody;
  const nv = m.nv;
  const savedIntegrator = m.opt.integrator;
  const savedTimestep = m.opt.timestep;
  const savedIterations = m.opt.iterations;
  const savedLsIterations = m.opt.ls_iterations;
  const savedSleepTolerance = m.opt.sleep_tolerance;
  const savedDisableFlags = m.opt.disableflags;
  const savedEnableFlags = m.opt.enableflags;

  const mPrio = m.geom_priority;
  const mFric = m.geom_friction;
  const mDamp = m.dof_damping;
  const mGContype = m.geom_contype;
  const mGConaff = m.geom_conaffinity;

  bufs.savedPriority.subarray(0, ngeom).set(mPrio);
  bufs.savedFriction.subarray(0, ngeom * 3).set(mFric);
  bufs.savedDofDamping.subarray(0, nv).set(mDamp);
  let hasSavedPostEscape = false;

  try {
    const sig = mj.mjtState.mjSTATE_FULLPHYSICS.value;
    mj.mj_resetData(m, sd);
    attachThreadPool(sd);
    mj.mj_copyState(m, d, sd, sig);
    const dTreeAsleep = d.tree_asleep;
    const sdTreeAsleep = sd.tree_asleep;
    for (let t = 0; t < m.ntree; t++) {
      sdTreeAsleep[t] = dTreeAsleep[t] >= 0 ? t : -10;
    }
    sd.qvel.fill(0);
    sd.qacc_warmstart.fill(0);
    if (d.mocap_pos.length > 0) sd.mocap_pos.set(d.mocap_pos);
    if (d.mocap_quat.length > 0) sd.mocap_quat.set(d.mocap_quat);
    sd.qfrc_applied.fill(0);
    sd.xfrc_applied.fill(0);
    sd.eq_active[map.grabEqualityId] = 0;
    m.opt.timestep = 1.0 / 600.0;
    m.opt.iterations = 50;
    m.opt.ls_iterations = 16;
    m.opt.sleep_tolerance = sleepTolerance;
    m.opt.disableflags &= ~mj.mjtDisableBit.mjDSBL_MULTICCD.value;

    const sf = glassFriction;
    const candidateBlockIds = bufs.candidateBlockIds;
    candidateBlockIds.length = 0;
    const sdQpos = sd.qpos;
    const sdQfrcApp = sd.qfrc_applied;

    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.shattered || b.extracted) continue;
      const gid = b.colliderHandle;
      if (
        mGContype[gid] === 0 ||
        mGConaff[gid] === offRimBlockConaffinity ||
        (mGConaff[gid] & 0x01) === 0
      ) {
        continue;
      }
      const qa = map.blockQposAdr[b.id];
      const xM = sdQpos[qa + 0];
      const yM = sdQpos[qa + 1];
      const zM = sdQpos[qa + 2];
      const rCenterM = Math.hypot(xM, zM);
      let rMaxM = rCenterM + EQ_BLOCK_HALF_X_M;
      if (rCenterM > 1e-6) {
        const urx = xM / rCenterM;
        const urz = zM / rCenterM;
        const qw = sdQpos[qa + 3];
        const qx = sdQpos[qa + 4];
        const qy = sdQpos[qa + 5];
        const qz = sdQpos[qa + 6];
        const axX = 1.0 - 2.0 * (qy * qy + qz * qz);
        const axZ = 2.0 * (qx * qz - qw * qy);
        const ayX = 2.0 * (qx * qy - qw * qz);
        const ayZ = 2.0 * (qy * qz + qw * qx);
        const azX = 2.0 * (qx * qz + qw * qy);
        const azZ = 1.0 - 2.0 * (qx * qx + qy * qy);
        rMaxM =
          rCenterM +
          Math.abs(axX * urx + axZ * urz) * EQ_BLOCK_HALF_X_M +
          Math.abs(ayX * urx + ayZ * urz) * EQ_BLOCK_HALF_Y_M +
          Math.abs(azX * urx + azZ * urz) * EQ_BLOCK_HALF_Z_M;
      }
      if (yM < -0.005 || rMaxM > tableRadiusM) continue;
      mPrio[gid] = 0;
      const g3 = gid * 3;
      if (mFric[g3 + 0] < sf[0]) mFric[g3 + 0] = sf[0];
      if (mFric[g3 + 1] < sf[1]) mFric[g3 + 1] = sf[1];
      if (mFric[g3 + 2] < sf[2]) mFric[g3 + 2] = sf[2];
      sdQfrcApp[map.blockDofAdr[b.id] + 1] = 1e-9;
      candidateBlockIds.push(i);
    }

    const nearShardDofAdrs = bufs.nearShardDofAdrs;
    nearShardDofAdrs.length = 0;
    const nearShardSlotMask = bufs.nearShardSlotMask;
    nearShardSlotMask.fill(0);
    if (candidateBlockIds.length > 0 && shards.length > 0) {
      for (let sIdx = 0; sIdx < shards.length; sIdx++) {
        const shd = shards[sIdx];
        const sGid = shd.colliderHandle;
        if (mGContype[sGid] === 0 || (mGConaff[sGid] & 0x21) === 0) continue;
        const sQa = map.shardQposAdr[shd.slotIndex];
        const sx = sdQpos[sQa + 0];
        const sy = sdQpos[sQa + 1];
        const sz = sdQpos[sQa + 2];
        if (sy < -0.02) continue;
        let nearAnyBlock = false;
        for (let cIdx = 0; cIdx < candidateBlockIds.length; cIdx++) {
          const bQa = map.blockQposAdr[blocks[candidateBlockIds[cIdx]].id];
          const dx = sx - sdQpos[bQa + 0];
          const dy = sy - sdQpos[bQa + 1];
          const dz = sz - sdQpos[bQa + 2];
          if (Math.abs(dx) > 0.078 || Math.abs(dz) > 0.078 || Math.abs(dy) > 0.060) {
            continue;
          }
          const qw = sdQpos[bQa + 3];
          const qx = sdQpos[bQa + 4];
          const qy = sdQpos[bQa + 5];
          const qz = sdQpos[bQa + 6];
          const lx =
            (1.0 - 2.0 * (qy * qy + qz * qz)) * dx +
            2.0 * (qx * qy + qw * qz) * dy +
            2.0 * (qx * qz - qw * qy) * dz;
          const ly =
            2.0 * (qx * qy - qw * qz) * dx +
            (1.0 - 2.0 * (qx * qx + qz * qz)) * dy +
            2.0 * (qy * qz + qw * qx) * dz;
          const lz =
            2.0 * (qx * qz + qw * qy) * dx +
            2.0 * (qy * qz - qw * qx) * dy +
            (1.0 - 2.0 * (qx * qx + qy * qy)) * dz;
          if (Math.abs(lx) < 0.072 && Math.abs(ly) < 0.048 && Math.abs(lz) < 0.050) {
            nearAnyBlock = true;
            break;
          }
        }
        if (nearAnyBlock) {
          const sDof = map.shardDofAdr[shd.slotIndex];
          sdQfrcApp[sDof + 1] = 1e-9;
          nearShardDofAdrs.push(sDof);
          nearShardSlotMask[shd.slotIndex] = 1;
        }
      }
    }

    if (candidateBlockIds.length > 0) {
      mj.mj_step1(m, sd);
      sd.qfrc_applied.fill(0);
      mj.mj_forward(m, sd);

      if (shards.length > 0 && sd.ncon > 0 && onRelieveEmbeddedShards) {
        if (onRelieveEmbeddedShards(sd)) {
          sd.qpos.set(d.qpos);
          sd.qvel.fill(0);
          sd.qacc_warmstart.fill(0);
          mj.mj_forward(m, sd);
        }
      }

      const mGCtPost = m.geom_contype;
      const mGCaPost = m.geom_conaffinity;
      const mBCtPost = m.body_contype;
      const mBCaPost = m.body_conaffinity;
      bufs.savedContype.subarray(0, ngeom).set(mGCtPost);
      bufs.savedConaffinity.subarray(0, ngeom).set(mGCaPost);
      bufs.savedBodyContype.subarray(0, nbody).set(mBCtPost);
      bufs.savedBodyConaffinity.subarray(0, nbody).set(mBCaPost);
      hasSavedPostEscape = true;

      if (shards.length > 0) {
        for (let sIdx = 0; sIdx < shards.length; sIdx++) {
          const shd = shards[sIdx];
          if (nearShardSlotMask[shd.slotIndex] === 0) {
            mGCtPost[shd.colliderHandle] = 0;
            mGCaPost[shd.colliderHandle] = 0;
            mBCtPost[shd.bodyHandle] = 0;
            mBCaPost[shd.bodyHandle] = 0;
          }
        }
      }

      const startPose = bufs.startPose;
      const step6Pose = bufs.step6Pose;
      const wakeFlags = bufs.wakeFlags;
      const gateBPending = bufs.gateBPending;
      wakeFlags.fill(0, 0, blocks.length);
      gateBPending.fill(0, 0, blocks.length);

      const sdQposInit = sd.qpos;
      for (let idx = 0; idx < candidateBlockIds.length; idx++) {
        const i = candidateBlockIds[idx];
        const qa = map.blockQposAdr[blocks[i].id];
        const b7 = i * 7;
        for (let k = 0; k < 7; k++) startPose[b7 + k] = sdQposInit[qa + k];
      }

      const hasNearShards = nearShardDofAdrs.length > 0;
      const scratchBuf = bufs.contactBuf;

      const useRk4Verify = hasNearShards && d.ntree_awake === 0;
      m.opt.integrator = useRk4Verify
        ? mj.mjtIntegrator.mjINT_RK4.value
        : mj.mjtIntegrator.mjINT_EULER.value;
      m.opt.timestep = 1.0 / 600.0;
      m.opt.iterations = 50;
      m.opt.ls_iterations = 16;

      mj.mj_resetData(m, sd);
      attachThreadPool(sd);
      mj.mj_copyState(m, d, sd, sig);
      const dTreeAsleep2 = d.tree_asleep;
      const sdTreeAsleep2 = sd.tree_asleep;
      for (let t = 0; t < m.ntree; t++) {
        sdTreeAsleep2[t] = dTreeAsleep2[t] >= 0 ? t : -10;
      }
      sd.qvel.fill(0);
      sd.qacc_warmstart.fill(0);
      if (d.mocap_pos.length > 0) sd.mocap_pos.set(d.mocap_pos);
      if (d.mocap_quat.length > 0) sd.mocap_quat.set(d.mocap_quat);
      const sdQfrcApp2 = sd.qfrc_applied;
      sdQfrcApp2.fill(0);
      sd.xfrc_applied.fill(0);
      sd.eq_active[map.grabEqualityId] = 0;
      for (let ns = 0; ns < nearShardDofAdrs.length; ns++) {
        sdQfrcApp2[nearShardDofAdrs[ns] + 1] = 1e-9;
      }
      for (let idx = 0; idx < candidateBlockIds.length; idx++) {
        const b = blocks[candidateBlockIds[idx]];
        sdQfrcApp2[map.blockDofAdr[b.id] + 1] = 1e-9;
      }

      let wokeCount = 0;
      const emptyMask = wakeFlags;

      mj.mj_step1(m, sd);
      sd.qfrc_applied.fill(0);
      mj.mj_forward(m, sd);
      let sqpos = sd.qpos;
      let sqvel = sd.qvel;
      let sqacc = sd.qacc;
      let bodyMass = m.body_mass;
      let geomBodyId = m.geom_bodyid;
      populateBlockContactScratchFromMjData(
        sd,
        m,
        scratchBuf,
        blocks.length,
        map,
        bodyIdToBlock,
        minShardGid,
        maxShardGid,
        null,
        emptyMask,
        null,
        null,
        false,
        0.0022,
        sqpos,
        sqvel,
        geomBodyId
      );
      const sdQfrcCon0 = sd.qfrc_constraint;
      for (let idx = 0; idx < candidateBlockIds.length; idx++) {
        const i = candidateBlockIds[idx];
        const b = blocks[i];
        const qa = map.blockQposAdr[b.id];
        const dof = map.blockDofAdr[b.id];
        const gateB0 = auditBlockEquilibriumGateB(
          b.id,
          scratchBuf,
          sqpos[qa + 0],
          sqpos[qa + 1],
          sqpos[qa + 2],
          sqpos[qa + 3],
          sqpos[qa + 4],
          sqpos[qa + 5],
          sqpos[qa + 6],
          bodyMass[b.bodyHandle],
          Math.max(0, sdQfrcCon0[dof + 1])
        );
        if (!gateB0.pass) {
          gateBPending[i] = 1;
        }
      }

      const isHeavyShardPass0 =
        hasNearShards &&
        !useRk4Verify &&
        shards.length >= 160 &&
        postSleepSweepPassCount === 0;
      const totalScratchSteps = isHeavyShardPass0 ? 96 : 180;
      const gateBCheckStep = 5;
      const limDy = 1.55;
      const limPos = 2.35;
      const limTheta = 2.85;

      for (let stepIdx = 0; stepIdx < totalScratchSteps; stepIdx++) {
        if (isHeavyShardPass0 && stepIdx === 12) {
          m.opt.iterations = 28;
          m.opt.ls_iterations = 6;
        }
        mj.mj_step(m, sd);
        if (sqpos.buffer.byteLength === 0) {
          sqpos = sd.qpos;
          sqvel = sd.qvel;
          sqacc = sd.qacc;
          bodyMass = m.body_mass;
          geomBodyId = m.geom_bodyid;
        }
        if (hasNearShards && !useRk4Verify) {
          for (let ns = 0; ns < nearShardDofAdrs.length; ns++) {
            const sDof = nearShardDofAdrs[ns];
            const svx = sqvel[sDof + 0];
            const svy = sqvel[sDof + 1];
            const svz = sqvel[sDof + 2];
            const svSq = svx * svx + svy * svy + svz * svz;
            if (svSq > 0.81) {
              const kv = 0.90 / Math.hypot(svx, svy, svz);
              sqvel[sDof + 0] = svx * kv;
              sqvel[sDof + 1] = svy * kv;
              sqvel[sDof + 2] = svz * kv;
            }
            const swx = sqvel[sDof + 3];
            const swy = sqvel[sDof + 4];
            const swz = sqvel[sDof + 5];
            const swSq = swx * swx + swy * swy + swz * swz;
            if (swSq > 3025.0) {
              const kw = 55.0 / Math.hypot(swx, swy, swz);
              sqvel[sDof + 3] = swx * kw;
              sqvel[sDof + 4] = swy * kw;
              sqvel[sDof + 5] = swz * kw;
            }
          }
        }
        if (stepIdx === 6) {
          for (let idx = 0; idx < candidateBlockIds.length; idx++) {
            const ci = candidateBlockIds[idx];
            const qa = map.blockQposAdr[blocks[ci].id];
            const b7 = ci * 7;
            for (let k = 0; k < 7; k++) step6Pose[b7 + k] = sqpos[qa + k];
          }
        }
        if (stepIdx === (useRk4Verify ? 24 : isHeavyShardPass0 ? 26 : 30)) {
          let maxBlkV = 0;
          let maxBlkW = 0;
          let maxBlkLinAcc = 0;
          let maxBlkAngAcc = 0;
          let maxDyChk = 0;
          let maxDPosChk = 0;
          let maxDThetaChk = 0;
          let maxDy6To16 = 0;
          let maxDPos6To16 = 0;
          let maxDTheta6To16 = 0;
          for (let idx = 0; idx < candidateBlockIds.length; idx++) {
            const ci = candidateBlockIds[idx];
            const dof = map.blockDofAdr[blocks[ci].id];
            const qa = map.blockQposAdr[blocks[ci].id];
            const b7 = ci * 7;
            const v = Math.hypot(sqvel[dof + 0], sqvel[dof + 1], sqvel[dof + 2]);
            const w = Math.hypot(sqvel[dof + 3], sqvel[dof + 4], sqvel[dof + 5]);
            const la = Math.hypot(sqacc[dof + 0], sqacc[dof + 1], sqacc[dof + 2]);
            const aa = Math.hypot(sqacc[dof + 3], sqacc[dof + 4], sqacc[dof + 5]);
            const dy = Math.abs(sqpos[qa + 1] - startPose[b7 + 1]) * 1000.0;
            const dp =
              Math.hypot(
                sqpos[qa + 0] - startPose[b7 + 0],
                sqpos[qa + 1] - startPose[b7 + 1],
                sqpos[qa + 2] - startPose[b7 + 2]
              ) * 1000.0;
            const dotQChk = Math.min(
              1.0,
              Math.abs(
                sqpos[qa + 3] * startPose[b7 + 3] +
                  sqpos[qa + 4] * startPose[b7 + 4] +
                  sqpos[qa + 5] * startPose[b7 + 5] +
                  sqpos[qa + 6] * startPose[b7 + 6]
              )
            );
            const dTh = (2.0 * Math.acos(dotQChk) * 180.0) / Math.PI;
            const dy6 = Math.abs(sqpos[qa + 1] - step6Pose[b7 + 1]) * 1000.0;
            const dp6 =
              Math.hypot(
                sqpos[qa + 0] - step6Pose[b7 + 0],
                sqpos[qa + 1] - step6Pose[b7 + 1],
                sqpos[qa + 2] - step6Pose[b7 + 2]
              ) * 1000.0;
            const dotQ6 = Math.min(
              1.0,
              Math.abs(
                sqpos[qa + 3] * step6Pose[b7 + 3] +
                  sqpos[qa + 4] * step6Pose[b7 + 4] +
                  sqpos[qa + 5] * step6Pose[b7 + 5] +
                  sqpos[qa + 6] * step6Pose[b7 + 6]
              )
            );
            const dTh6 = (2.0 * Math.acos(dotQ6) * 180.0) / Math.PI;
            if (v > maxBlkV) maxBlkV = v;
            if (w > maxBlkW) maxBlkW = w;
            if (la > maxBlkLinAcc) maxBlkLinAcc = la;
            if (aa > maxBlkAngAcc) maxBlkAngAcc = aa;
            if (dy > maxDyChk) maxDyChk = dy;
            if (dp > maxDPosChk) maxDPosChk = dp;
            if (dTh > maxDThetaChk) maxDThetaChk = dTh;
            if (dy6 > maxDy6To16) maxDy6To16 = dy6;
            if (dp6 > maxDPos6To16) maxDPos6To16 = dp6;
            if (dTh6 > maxDTheta6To16) maxDTheta6To16 = dTh6;
          }
          if (useRk4Verify) {
            if (
              wokeCount === 0 &&
              maxDyChk < 0.030 &&
              maxDPosChk < 0.050 &&
              maxDThetaChk < 0.15 &&
              maxDy6To16 < 0.022 &&
              maxDPos6To16 < 0.035 &&
              maxDTheta6To16 < 0.14
            ) {
              break;
            }
          } else if (
            wokeCount === 0 &&
            maxDPosChk < 0.12 &&
            maxDThetaChk < 0.16 &&
            maxBlkV < 0.0008 &&
            maxBlkW < 0.012 &&
            maxBlkLinAcc < 0.025 &&
            maxBlkAngAcc < 0.35
          ) {
            break;
          }
        }
        if (stepIdx === gateBCheckStep) {
          populateBlockContactScratchFromMjData(
            sd,
            m,
            scratchBuf,
            blocks.length,
            map,
            bodyIdToBlock,
            minShardGid,
            maxShardGid,
            null,
            emptyMask,
            null,
            null,
            false,
            0.0022,
            sqpos,
            sqvel,
            geomBodyId
          );
          const sdQfrcCon5 = sd.qfrc_constraint;
          for (let idx = 0; idx < candidateBlockIds.length; idx++) {
            const i = candidateBlockIds[idx];
            if (gateBPending[i] === 0) continue;
            const b = blocks[i];
            const qa = map.blockQposAdr[b.id];
            const dof = map.blockDofAdr[b.id];
            const gateB5 = auditBlockEquilibriumGateB(
              b.id,
              scratchBuf,
              sqpos[qa + 0],
              sqpos[qa + 1],
              sqpos[qa + 2],
              sqpos[qa + 3],
              sqpos[qa + 4],
              sqpos[qa + 5],
              sqpos[qa + 6],
              bodyMass[b.bodyHandle],
              Math.max(0, sdQfrcCon5[dof + 1])
            );
            if (!gateB5.pass && wakeFlags[i] === 0) {
              wakeFlags[i] = 1;
              wokeCount++;
            }
          }
        }
        for (let idx = 0; idx < candidateBlockIds.length; idx++) {
          const i = candidateBlockIds[idx];
          if (wakeFlags[i] !== 0) continue;
          const qa = map.blockQposAdr[blocks[i].id];
          const b7 = i * 7;
          const dxMm = (sqpos[qa + 0] - startPose[b7 + 0]) * 1000.0;
          const dyMm = (sqpos[qa + 1] - startPose[b7 + 1]) * 1000.0;
          const dzMm = (sqpos[qa + 2] - startPose[b7 + 2]) * 1000.0;
          const absDy = Math.abs(dyMm);
          const dPosSq = dxMm * dxMm + dyMm * dyMm + dzMm * dzMm;
          const dotQ = Math.min(
            1.0,
            Math.abs(
              sqpos[qa + 3] * startPose[b7 + 3] +
                sqpos[qa + 4] * startPose[b7 + 4] +
                sqpos[qa + 5] * startPose[b7 + 5] +
                sqpos[qa + 6] * startPose[b7 + 6]
            )
          );
          // cos(2.85 deg / 2) = cos(1.425 deg) = 0.99969076
          if (absDy <= limDy && dPosSq <= limPos * limPos && dotQ >= 0.9996908) {
            continue;
          }
          const dPos = Math.sqrt(dPosSq);
          const dTheta = (2.0 * Math.acos(dotQ) * 180.0) / Math.PI;
          const bidChk = blocks[i].id;
          const retries = bidChk < blockEqWakeRetries.length ? blockEqWakeRetries[bidChk] : 0;
          if (
            (absDy > limDy || dPos > limPos || dTheta > limTheta) &&
            !(
              retries >= 1 &&
              (useRk4Verify
                ? absDy < 1.85 && dPos < 2.80 && dTheta < 3.80
                : absDy < 1.75 && dPos < 2.68 && dTheta < 3.65)
            )
          ) {
            wakeFlags[i] = 1;
            wokeCount++;
          }
        }
      }
      for (let idx = 0; idx < candidateBlockIds.length; idx++) {
        const i = candidateBlockIds[idx];
        if (wakeFlags[i] !== 0) {
          toWakeBlockIds.push(blocks[i].id);
        }
      }
    }
  } finally {
    m.opt.integrator = savedIntegrator;
    m.opt.timestep = savedTimestep;
    m.opt.iterations = savedIterations;
    m.opt.ls_iterations = savedLsIterations;
    m.opt.sleep_tolerance = savedSleepTolerance;
    m.opt.disableflags = savedDisableFlags;
    m.opt.enableflags = savedEnableFlags;
    m.geom_priority.set(bufs.savedPriority.subarray(0, ngeom));
    m.geom_friction.set(bufs.savedFriction.subarray(0, ngeom * 3));
    m.dof_damping.set(bufs.savedDofDamping.subarray(0, nv));
    if (hasSavedPostEscape) {
      m.geom_contype.set(bufs.savedContype.subarray(0, ngeom));
      m.geom_conaffinity.set(bufs.savedConaffinity.subarray(0, ngeom));
      m.body_contype.set(bufs.savedBodyContype.subarray(0, nbody));
      m.body_conaffinity.set(bufs.savedBodyConaffinity.subarray(0, nbody));
    }
  }

  return toWakeBlockIds;
}

/**
 * Dual Anti-Goodharting Settled Plausibility Oracle (Zero live `MjData`/`MjModel` mutation):
 * - Gate A: Pure-MuJoCo Wake-and-Step Ground Truth (`600 Hz / 50 Newton iters / 16 ls_iters`, `0` custom JS hooks).
 * - Gate B: Coordinate-Free 3D Contact-Hull & Knife-Edge Invariant on the woken converged contact manifold.
 */
export function evaluateTwoGateSettledPlausibilityInScratch<TBlock extends EquilibriumBlockRef>(params: {
  mj: MainModule;
  m: MjModel;
  d: MjData;
  sd: MjData;
  bufs: ScratchEquilibriumOracleBuffers;
  map: MujocoTowerIndexMap;
  blocks: readonly TBlock[];
  shards: readonly EquilibriumShardRef[];
  bodyIdToBlock: ArrayLike<TBlock | undefined>;
  minShardGid: number;
  maxShardGid: number;
  activeBlockId: number | null;
  offRimBlockConaffinity: number;
  tableRadiusM: number;
  glassFriction: readonly [number, number, number];
  sleepTolerance: number;
  substepsCount: number;
  attachThreadPool: (sd: MjData) => void;
}): TwoGatePlausibilityReport {
  const {
    mj,
    m,
    d,
    sd,
    bufs,
    map,
    blocks,
    shards,
    bodyIdToBlock,
    minShardGid,
    maxShardGid,
    activeBlockId,
    offRimBlockConaffinity,
    tableRadiusM,
    glassFriction,
    sleepTolerance,
    substepsCount,
    attachThreadPool,
  } = params;

  bufs.ensureCapacity(m, blocks.length);
  const ngeom = m.ngeom;
  const nbody = m.nbody;
  const nv = m.nv;

  const savedIntegrator = m.opt.integrator;
  const savedTimestep = m.opt.timestep;
  const savedIterations = m.opt.iterations;
  const savedLsIterations = m.opt.ls_iterations;
  const savedSleepTolerance = m.opt.sleep_tolerance;
  const savedDisableFlags = m.opt.disableflags;
  const savedEnableFlags = m.opt.enableflags;

  const mPrio = m.geom_priority;
  const mFric = m.geom_friction;
  const mDamp = m.dof_damping;
  const mGContype = m.geom_contype;
  const mGConaff = m.geom_conaffinity;

  bufs.savedPriority.subarray(0, ngeom).set(mPrio);
  bufs.savedFriction.subarray(0, ngeom * 3).set(mFric);
  bufs.savedDofDamping.subarray(0, nv).set(mDamp);
  bufs.savedContype.subarray(0, ngeom).set(mGContype);
  bufs.savedConaffinity.subarray(0, ngeom).set(mGConaff);
  bufs.savedBodyContype.subarray(0, nbody).set(m.body_contype);
  bufs.savedBodyConaffinity.subarray(0, nbody).set(m.body_conaffinity);

  const gateAViolations: TwoGatePlausibilityReport['gateAViolations'] = [];
  const gateBViolations: TwoGatePlausibilityReport['gateBViolations'] = [];
  let maxAbsDyMm = 0;
  let maxDPosMm = 0;
  let maxDThetaDeg = 0;

  try {
    const sf = glassFriction;
    const activeBlockIds = bufs.candidateBlockIds;
    activeBlockIds.length = 0;
    const initPose = bufs.startPose;
    const peakDyMm = bufs.peakDyMm;
    const peakDPosMm = bufs.peakDPosMm;
    const peakDThetaDeg = bufs.peakDThetaDeg;
    peakDyMm.fill(0, 0, blocks.length);
    peakDPosMm.fill(0, 0, blocks.length);
    peakDThetaDeg.fill(0, 0, blocks.length);

    const dQpos = d.qpos;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.shattered || b.extracted || b.id === activeBlockId) continue;
      const gid = b.colliderHandle;
      if (
        mGContype[gid] === 0 ||
        mGConaff[gid] === offRimBlockConaffinity ||
        (mGConaff[gid] & 0x01) === 0
      ) {
        continue;
      }
      const qa = map.blockQposAdr[b.id];
      const xM = dQpos[qa + 0];
      const yM = dQpos[qa + 1];
      const zM = dQpos[qa + 2];
      const rCenterM = Math.hypot(xM, zM);
      let rMaxM = rCenterM + EQ_BLOCK_HALF_X_M;
      if (rCenterM > 1e-6) {
        const urx = xM / rCenterM;
        const urz = zM / rCenterM;
        const qw = dQpos[qa + 3];
        const qx = dQpos[qa + 4];
        const qy = dQpos[qa + 5];
        const qz = dQpos[qa + 6];
        const axX = 1.0 - 2.0 * (qy * qy + qz * qz);
        const axZ = 2.0 * (qx * qz - qw * qy);
        const ayX = 2.0 * (qx * qy - qw * qz);
        const ayZ = 2.0 * (qy * qz + qw * qx);
        const azX = 2.0 * (qx * qz + qw * qy);
        const azZ = 1.0 - 2.0 * (qx * qx + qy * qy);
        rMaxM =
          rCenterM +
          Math.abs(axX * urx + axZ * urz) * EQ_BLOCK_HALF_X_M +
          Math.abs(ayX * urx + ayZ * urz) * EQ_BLOCK_HALF_Y_M +
          Math.abs(azX * urx + azZ * urz) * EQ_BLOCK_HALF_Z_M;
      }
      if (yM < -0.005 || rMaxM > tableRadiusM) continue;

      mPrio[gid] = 0;
      const g3 = gid * 3;
      if (mFric[g3 + 0] < sf[0]) mFric[g3 + 0] = sf[0];
      if (mFric[g3 + 1] < sf[1]) mFric[g3 + 1] = sf[1];
      if (mFric[g3 + 2] < sf[2]) mFric[g3 + 2] = sf[2];

      const base7 = i * 7;
      for (let k = 0; k < 7; k++) {
        initPose[base7 + k] = dQpos[qa + k];
      }
      activeBlockIds.push(i);
    }

    const sig = mj.mjtState.mjSTATE_FULLPHYSICS.value;
    mj.mj_resetData(m, sd);
    attachThreadPool(sd);
    mj.mj_copyState(m, d, sd, sig);
    const dTreeAsleep = d.tree_asleep;
    const sdTreeAsleep = sd.tree_asleep;
    for (let t = 0; t < m.ntree; t++) {
      sdTreeAsleep[t] = dTreeAsleep[t] >= 0 ? t : -10;
    }
    const sdQvel = sd.qvel;
    const dofBodyId = m.dof_bodyid;
    const bodyTreeId = m.body_treeid;
    for (let dofIdx = 0; dofIdx < sdQvel.length; dofIdx++) {
      const bId = dofBodyId[dofIdx];
      const tId = bodyTreeId[bId];
      if (tId >= 0 && dTreeAsleep[tId] >= 0) {
        sdQvel[dofIdx] = 0;
      }
    }
    sd.qacc_warmstart.fill(0);
    if (d.mocap_pos.length > 0) sd.mocap_pos.set(d.mocap_pos);
    if (d.mocap_quat.length > 0) sd.mocap_quat.set(d.mocap_quat);
    const sdQfrcApp = sd.qfrc_applied;
    sdQfrcApp.fill(0);
    sd.xfrc_applied.fill(0);
    sd.eq_active[map.grabEqualityId] = 0;

    const sdQpos = sd.qpos;
    const mGCt2 = m.geom_contype;
    const mGCa2 = m.geom_conaffinity;
    const mBCt2 = m.body_contype;
    const mBCa2 = m.body_conaffinity;
    let nearShardCount = 0;
    if (activeBlockIds.length > 0) {
      for (let sIdx = 0; sIdx < shards.length; sIdx++) {
        const shd = shards[sIdx];
        const sGid = shd.colliderHandle;
        if (mGCt2[sGid] === 0 || (mGCa2[sGid] & 0x21) === 0) continue;
        const sQa = map.shardQposAdr[shd.slotIndex];
        const sx = sdQpos[sQa + 0];
        const sy = sdQpos[sQa + 1];
        const sz = sdQpos[sQa + 2];
        if (sy < -0.02) continue;
        let nearAnyBlk = false;
        for (let a = 0; a < activeBlockIds.length; a++) {
          const bQa = map.blockQposAdr[blocks[activeBlockIds[a]].id];
          const dx = sx - sdQpos[bQa + 0];
          const dy = sy - sdQpos[bQa + 1];
          const dz = sz - sdQpos[bQa + 2];
          if (Math.abs(dx) > 0.078 || Math.abs(dz) > 0.078 || Math.abs(dy) > 0.060) {
            continue;
          }
          const qw = sdQpos[bQa + 3];
          const qx = sdQpos[bQa + 4];
          const qy = sdQpos[bQa + 5];
          const qz = sdQpos[bQa + 6];
          const lx =
            (1.0 - 2.0 * (qy * qy + qz * qz)) * dx +
            2.0 * (qx * qy + qw * qz) * dy +
            2.0 * (qx * qz - qw * qy) * dz;
          const ly =
            2.0 * (qx * qy - qw * qz) * dx +
            (1.0 - 2.0 * (qx * qx + qz * qz)) * dy +
            2.0 * (qy * qz + qw * qx) * dz;
          const lz =
            2.0 * (qx * qz + qw * qy) * dx +
            2.0 * (qy * qz - qw * qx) * dy +
            (1.0 - 2.0 * (qx * qx + qy * qy)) * dz;
          if (Math.abs(lx) < 0.072 && Math.abs(ly) < 0.048 && Math.abs(lz) < 0.050) {
            nearAnyBlk = true;
            break;
          }
        }
        if (nearAnyBlk) {
          sdQfrcApp[map.shardDofAdr[shd.slotIndex] + 1] = 1e-9;
          nearShardCount++;
        } else {
          mGCt2[sGid] = 0;
          mGCa2[sGid] = 0;
          mBCt2[shd.bodyHandle] = 0;
          mBCa2[shd.bodyHandle] = 0;
        }
      }
    } else {
      for (let sIdx = 0; sIdx < shards.length; sIdx++) {
        const shd = shards[sIdx];
        mGCt2[shd.colliderHandle] = 0;
        mGCa2[shd.colliderHandle] = 0;
        mBCt2[shd.bodyHandle] = 0;
        mBCa2[shd.bodyHandle] = 0;
      }
    }

    m.opt.integrator =
      nearShardCount > 0
        ? mj.mjtIntegrator.mjINT_RK4.value
        : mj.mjtIntegrator.mjINT_EULER.value;
    m.opt.timestep = 1.0 / 600.0;
    m.opt.iterations = 50;
    m.opt.ls_iterations = 16;
    m.opt.sleep_tolerance = sleepTolerance;
    m.opt.disableflags &= ~mj.mjtDisableBit.mjDSBL_MULTICCD.value;
    for (let a = 0; a < activeBlockIds.length; a++) {
      const b = blocks[activeBlockIds[a]];
      sdQfrcApp[map.blockDofAdr[b.id] + 1] = 1e-9;
    }

    const scratchBuf = bufs.contactBuf;
    const emptyMask = bufs.wakeFlags;
    emptyMask.fill(0, 0, blocks.length);
    const pendingGateB = new Map<number, GateBBlockAudit>();

    mj.mj_step1(m, sd);
    sd.qfrc_applied.fill(0);
    mj.mj_forward(m, sd);
    let sqpos = sd.qpos;
    let sqvel = sd.qvel;
    let bodyMass = m.body_mass;
    let geomBodyId = m.geom_bodyid;
    populateBlockContactScratchFromMjData(
      sd,
      m,
      scratchBuf,
      blocks.length,
      map,
      bodyIdToBlock,
      minShardGid,
      maxShardGid,
      null,
      emptyMask,
      null,
      null,
      false,
      0.0022,
      sqpos,
      sqvel,
      geomBodyId
    );
    const sdQfrcCon0 = sd.qfrc_constraint;
    for (let a = 0; a < activeBlockIds.length; a++) {
      const idx = activeBlockIds[a];
      const b = blocks[idx];
      const qa = map.blockQposAdr[b.id];
      const dof = map.blockDofAdr[b.id];
      const audit0 = auditBlockEquilibriumGateB(
        b.id,
        scratchBuf,
        sqpos[qa + 0],
        sqpos[qa + 1],
        sqpos[qa + 2],
        sqpos[qa + 3],
        sqpos[qa + 4],
        sqpos[qa + 5],
        sqpos[qa + 6],
        bodyMass[b.bodyHandle],
        Math.max(0, sdQfrcCon0[dof + 1])
      );
      if (!audit0.pass) {
        pendingGateB.set(idx, audit0);
      }
    }

    const totalSubsteps = Math.max(12, substepsCount);
    for (let stepIdx = 0; stepIdx < totalSubsteps; stepIdx++) {
      mj.mj_step(m, sd);
      if (sqpos.buffer.byteLength === 0) {
        sqpos = sd.qpos;
        sqvel = sd.qvel;
        bodyMass = m.body_mass;
        geomBodyId = m.geom_bodyid;
      }
      if (stepIdx === 5 && pendingGateB.size > 0) {
        populateBlockContactScratchFromMjData(
          sd,
          m,
          scratchBuf,
          blocks.length,
          map,
          bodyIdToBlock,
          minShardGid,
          maxShardGid,
          null,
          emptyMask,
          null,
          null,
          false,
          0.0022,
          sqpos,
          sqvel,
          geomBodyId
        );
        const sdQfrcCon5 = sd.qfrc_constraint;
        for (const [idx] of pendingGateB) {
          const b = blocks[idx];
          const qa = map.blockQposAdr[b.id];
          const dof = map.blockDofAdr[b.id];
          const audit5 = auditBlockEquilibriumGateB(
            b.id,
            scratchBuf,
            sqpos[qa + 0],
            sqpos[qa + 1],
            sqpos[qa + 2],
            sqpos[qa + 3],
            sqpos[qa + 4],
            sqpos[qa + 5],
            sqpos[qa + 6],
            bodyMass[b.bodyHandle],
            Math.max(0, sdQfrcCon5[dof + 1])
          );
          if (audit5.pass) {
            pendingGateB.delete(idx);
          } else {
            pendingGateB.set(idx, audit5);
          }
        }
      }

      for (let a = 0; a < activeBlockIds.length; a++) {
        const idx = activeBlockIds[a];
        const b = blocks[idx];
        const qa = map.blockQposAdr[b.id];
        const base7 = idx * 7;
        const dxMm = (sqpos[qa + 0] - initPose[base7 + 0]) * 1000.0;
        const dyMm = (sqpos[qa + 1] - initPose[base7 + 1]) * 1000.0;
        const dzMm = (sqpos[qa + 2] - initPose[base7 + 2]) * 1000.0;
        const absDy = Math.abs(dyMm);
        const dPos = Math.sqrt(dxMm * dxMm + dyMm * dyMm + dzMm * dzMm);
        const dotQ = Math.min(
          1.0,
          Math.abs(
            sqpos[qa + 3] * initPose[base7 + 3] +
              sqpos[qa + 4] * initPose[base7 + 4] +
              sqpos[qa + 5] * initPose[base7 + 5] +
              sqpos[qa + 6] * initPose[base7 + 6]
          )
        );
        const dTheta = (2.0 * Math.acos(dotQ) * 180.0) / Math.PI;
        if (absDy > peakDyMm[idx]) peakDyMm[idx] = absDy;
        if (dPos > peakDPosMm[idx]) peakDPosMm[idx] = dPos;
        if (dTheta > peakDThetaDeg[idx]) peakDThetaDeg[idx] = dTheta;
      }
    }

    for (const [idx, audit] of pendingGateB) {
      gateBViolations.push({ blockId: blocks[idx].id, audit });
    }

    for (let a = 0; a < activeBlockIds.length; a++) {
      const idx = activeBlockIds[a];
      const b = blocks[idx];
      const absDy = peakDyMm[idx];
      const dPos = peakDPosMm[idx];
      const dTheta = peakDThetaDeg[idx];
      if (absDy > maxAbsDyMm) maxAbsDyMm = absDy;
      if (dPos > maxDPosMm) maxDPosMm = dPos;
      if (dTheta > maxDThetaDeg) maxDThetaDeg = dTheta;
      if (absDy > 2.0 || dPos > 3.0 || dTheta > 4.0) {
        const base7 = idx * 7;
        gateAViolations.push({
          blockId: b.id,
          absDyMm: +absDy.toFixed(3),
          dPosMm: +dPos.toFixed(3),
          dThetaDeg: +dTheta.toFixed(3),
          posMm: [
            +(initPose[base7 + 0] * 1000).toFixed(2),
            +(initPose[base7 + 1] * 1000).toFixed(2),
            +(initPose[base7 + 2] * 1000).toFixed(2),
          ],
        });
      }
    }
  } finally {
    m.opt.integrator = savedIntegrator;
    m.opt.timestep = savedTimestep;
    m.opt.iterations = savedIterations;
    m.opt.ls_iterations = savedLsIterations;
    m.opt.sleep_tolerance = savedSleepTolerance;
    m.opt.disableflags = savedDisableFlags;
    m.opt.enableflags = savedEnableFlags;
    m.geom_priority.set(bufs.savedPriority.subarray(0, ngeom));
    m.geom_friction.set(bufs.savedFriction.subarray(0, ngeom * 3));
    m.dof_damping.set(bufs.savedDofDamping.subarray(0, nv));
    m.geom_contype.set(bufs.savedContype.subarray(0, ngeom));
    m.geom_conaffinity.set(bufs.savedConaffinity.subarray(0, ngeom));
    m.body_contype.set(bufs.savedBodyContype.subarray(0, nbody));
    m.body_conaffinity.set(bufs.savedBodyConaffinity.subarray(0, nbody));
  }

  const gateAPass = gateAViolations.length === 0;
  const gateBPass = gateBViolations.length === 0;
  return {
    pass: gateAPass && gateBPass,
    gateAPass,
    gateBPass,
    maxAbsDyMm: +maxAbsDyMm.toFixed(3),
    maxDPosMm: +maxDPosMm.toFixed(3),
    maxDThetaDeg: +maxDThetaDeg.toFixed(3),
    gateAViolations,
    gateBViolations,
  };
}

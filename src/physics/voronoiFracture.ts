import { Vec3 } from '../math/vecMat';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  GLASS_DENSITY,
  SHARD_SLOT_STRIDE_VERTS,
} from './constants';

export const MAX_SHARD_PLANES = 12;

interface PolyFace {
  verts: Vec3[];
  normal: Vec3;
  isOuter: boolean;
}

export interface GlassShardGeometry {
  /** Centroid of this shard in the parent block's local space */
  localCentroid: Vec3;
  /** Exact 3D polyhedron volume in dm^3 */
  volume: number;
  /** Distance from shard centroid to localImpactPoint in dm */
  impactDist: number;
  /** Unique convex hull points relative to localCentroid (for Rapier ColliderDesc.convexHull) */
  hullPoints: Float32Array;
  /**
   * Up to MAX_SHARD_PLANES (12) bounding half-space planes in shard-centroid space:
   * [nx, ny, nz, signedD] where abs(signedD) > 0 is distance from centroid to face,
   * and signedD < 0 indicates an original exterior leaded face (`isOuter = true`).
   */
  planes: Float32Array;
  planeCount: number;
  /**
   * Triangulated flat-shaded vertex data: 10 floats per vertex (40 bytes):
   * [cx, cy, cz] (pos relative to shard centroid),
   * [nx, ny, nz] (outward face normal in shard local frame),
   * [bx, by, bz] (original pos in parent block local space for stained-glass motif & lead-came continuity),
   * [isOuterFace] (1.0 = original exterior leaded face, 0.0 = fresh internal glass cleavage face)
   */
  renderVertices: Float32Array;
  vertexCount: number;
}

const EPS = 1e-5;

function createInitialBoxFaces(hx: number, hy: number, hz: number): PolyFace[] {
  return [
    // +Z
    {
      verts: [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
      normal: [0, 0, 1],
      isOuter: true,
    },
    // -Z
    {
      verts: [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]],
      normal: [0, 0, -1],
      isOuter: true,
    },
    // +X
    {
      verts: [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]],
      normal: [1, 0, 0],
      isOuter: true,
    },
    // -X
    {
      verts: [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]],
      normal: [-1, 0, 0],
      isOuter: true,
    },
    // +Y
    {
      verts: [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]],
      normal: [0, 1, 0],
      isOuter: true,
    },
    // -Y
    {
      verts: [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]],
      normal: [0, -1, 0],
      isOuter: true,
    },
  ];
}

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function computePolygonArea(verts: Vec3[]): number {
  const n = verts.length;
  if (n < 3) return 0;
  let area = 0;
  const v0 = verts[0];
  const v0x = v0[0];
  const v0y = v0[1];
  const v0z = v0[2];
  for (let i = 1; i < n - 1; i++) {
    const vi = verts[i];
    const vj = verts[i + 1];
    const e1x = vi[0] - v0x;
    const e1y = vi[1] - v0y;
    const e1z = vi[2] - v0z;
    const e2x = vj[0] - v0x;
    const e2y = vj[1] - v0y;
    const e2z = vj[2] - v0z;
    const cx = e1y * e2z - e1z * e2y;
    const cy = e1z * e2x - e1x * e2z;
    const cz = e1x * e2y - e1y * e2x;
    area += 0.5 * Math.hypot(cx, cy, cz);
  }
  return area;
}

const scratchCapPoints: Vec3[] = [];
const scratchClipped: Vec3[] = [];
let scratchCapAngles = new Float64Array(64);
const scratchCapOrder: number[] = [];
const scratchVertexDa = new Float64Array(64);

function distExceeds1e4Exact(ax: number, ay: number, az: number, bx: number, by: number, bz: number): boolean {
  const dx = ax - bx;
  const dy = ay - by;
  const dz = az - bz;
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > 1.0001e-8) return true;
  if (d2 < 0.9999e-8) return false;
  return Math.hypot(dx, dy, dz) > 1e-4;
}

function distBelow2e4Exact(ax: number, ay: number, az: number, bx: number, by: number, bz: number): boolean {
  const dx = ax - bx;
  const dy = ay - by;
  const dz = az - bz;
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 < 3.9999e-8) return true;
  if (d2 > 4.0001e-8) return false;
  return Math.hypot(dx, dy, dz) < 2e-4;
}

/**
 * Clips a closed 3D convex polyhedron against the half-space dot(n, p) <= planeD.
 * Any clipped cross-section receives a watertight CCW cap face with outward normal `n` and `isOuter = false`.
 */
function clipConvexPolyhedron(
  faces: PolyFace[],
  planeNormal: Vec3,
  planeD: number,
  mutateIfUnclipped: boolean = false
): PolyFace[] {
  const pnx = planeNormal[0];
  const pny = planeNormal[1];
  const pnz = planeNormal[2];
  let anyOutsideOrBoundary = false;
  outer: for (let fIdx = 0; fIdx < faces.length; fIdx++) {
    const verts = faces[fIdx].verts;
    for (let i = 0; i < verts.length; i++) {
      const v = verts[i];
      if (pnx * v[0] + pny * v[1] + pnz * v[2] - planeD >= -EPS) {
        anyOutsideOrBoundary = true;
        break outer;
      }
    }
  }
  if (!anyOutsideOrBoundary) {
    if (mutateIfUnclipped) {
      for (let fIdx = 0; fIdx < faces.length; fIdx++) {
        const verts = faces[fIdx].verts;
        const n = verts.length;
        if (n >= 3) {
          const v0 = verts[0];
          for (let i = 0; i < n - 1; i++) verts[i] = verts[i + 1];
          verts[n - 1] = v0;
        }
      }
      return faces;
    }
    const copyFaces: PolyFace[] = new Array(faces.length);
    for (let fIdx = 0; fIdx < faces.length; fIdx++) {
      const face = faces[fIdx];
      const verts = face.verts;
      const n = verts.length;
      const rotVerts: Vec3[] = new Array(n);
      for (let i = 0; i < n - 1; i++) rotVerts[i] = verts[i + 1];
      rotVerts[n - 1] = verts[0];
      copyFaces[fIdx] = { verts: rotVerts, normal: face.normal, isOuter: face.isOuter };
    }
    return copyFaces;
  }

  const nextFaces: PolyFace[] = [];
  const capPoints = scratchCapPoints;
  capPoints.length = 0;

  const addCapPoint = (p: Vec3) => {
    const px = p[0];
    const py = p[1];
    const pz = p[2];
    for (let i = 0; i < capPoints.length; i++) {
      const q = capPoints[i];
      if (
        Math.abs(px - q[0]) < 1e-4 &&
        Math.abs(py - q[1]) < 1e-4 &&
        Math.abs(pz - q[2]) < 1e-4
      ) {
        return;
      }
    }
    capPoints.push(p);
  };

  let daBuf = scratchVertexDa;

  for (let fIdx = 0; fIdx < faces.length; fIdx++) {
    const face = faces[fIdx];
    const verts = face.verts;
    const count = verts.length;
    if (count < 3) continue;

    if (daBuf.length < count) {
      daBuf = new Float64Array(Math.max(64, count * 2));
    }
    let allStrictlyInside = true;
    let allStrictlyOutside = true;
    for (let i = 0; i < count; i++) {
      const v = verts[i];
      const d = pnx * v[0] + pny * v[1] + pnz * v[2] - planeD;
      daBuf[i] = d;
      if (d >= -EPS) allStrictlyInside = false;
      if (d <= EPS) allStrictlyOutside = false;
    }
    if (allStrictlyOutside) continue;
    if (allStrictlyInside) {
      if (mutateIfUnclipped) {
        const v0 = verts[0];
        for (let i = 0; i < count - 1; i++) verts[i] = verts[i + 1];
        verts[count - 1] = v0;
        nextFaces.push(face);
      } else {
        const rotVerts: Vec3[] = new Array(count);
        for (let i = 0; i < count - 1; i++) rotVerts[i] = verts[i + 1];
        rotVerts[count - 1] = verts[0];
        nextFaces.push({
          verts: rotVerts,
          normal: face.normal,
          isOuter: face.isOuter,
        });
      }
      continue;
    }

    const clipped = scratchClipped;
    clipped.length = 0;
    for (let i = 0; i < count; i++) {
      const nextIdx = i + 1 === count ? 0 : i + 1;
      const a = verts[i];
      const b = verts[nextIdx];
      const da = daBuf[i];
      const db = daBuf[nextIdx];

      const aIn = da <= EPS;
      const bIn = db <= EPS;

      if (aIn && bIn) {
        clipped.push(b);
        if (Math.abs(db) <= EPS) addCapPoint(b);
      } else if (aIn && !bIn) {
        const t = da / (da - db);
        const inter: Vec3 = [
          a[0] + (b[0] - a[0]) * t,
          a[1] + (b[1] - a[1]) * t,
          a[2] + (b[2] - a[2]) * t,
        ];
        clipped.push(inter);
        addCapPoint(inter);
      } else if (!aIn && bIn) {
        const t = da / (da - db);
        const inter: Vec3 = [
          a[0] + (b[0] - a[0]) * t,
          a[1] + (b[1] - a[1]) * t,
          a[2] + (b[2] - a[2]) * t,
        ];
        clipped.push(inter);
        addCapPoint(inter);
        clipped.push(b);
        if (Math.abs(db) <= EPS) addCapPoint(b);
      }
    }

    // Deduplicate adjacent near-identical vertices
    const clean: Vec3[] = [];
    for (let i = 0; i < clipped.length; i++) {
      const cur = clipped[i];
      if (clean.length === 0) {
        clean.push(cur);
      } else {
        const prev = clean[clean.length - 1];
        if (distExceeds1e4Exact(cur[0], cur[1], cur[2], prev[0], prev[1], prev[2])) {
          clean.push(cur);
        }
      }
    }
    if (clean.length >= 3) {
      const first = clean[0];
      const last = clean[clean.length - 1];
      if (!distExceeds1e4Exact(first[0], first[1], first[2], last[0], last[1], last[2])) {
        clean.pop();
      }
    }

    if (clean.length >= 3) {
      nextFaces.push({
        verts: clean,
        normal: face.normal,
        isOuter: face.isOuter,
      });
    }
  }

  // Construct the newly exposed internal fracture cleavage cap face
  const capLen = capPoints.length;
  if (capLen >= 3) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < capLen; i++) {
      const p = capPoints[i];
      cx += p[0];
      cy += p[1];
      cz += p[2];
    }
    cx /= capLen;
    cy /= capLen;
    cz /= capLen;

    // Build orthonormal tangent basis (u, v) on plane such that cross(u, v) == planeNormal
    const ref: Vec3 = Math.abs(planeNormal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = normalize3(cross3(ref, planeNormal));
    const v = cross3(planeNormal, u);

    if (scratchCapAngles.length < capLen) {
      scratchCapAngles = new Float64Array(capLen * 2);
    }
    scratchCapOrder.length = capLen;
    for (let i = 0; i < capLen; i++) {
      const p = capPoints[i];
      const rx = p[0] - cx;
      const ry = p[1] - cy;
      const rz = p[2] - cz;
      const pu = rx * u[0] + ry * u[1] + rz * u[2];
      const pv = rx * v[0] + ry * v[1] + rz * v[2];
      scratchCapAngles[i] = Math.atan2(pv, pu);
      scratchCapOrder[i] = i;
    }
    scratchCapOrder.sort((a, b) => scratchCapAngles[a] - scratchCapAngles[b]);
    const sortedCap: Vec3[] = new Array(capLen);
    for (let i = 0; i < capLen; i++) {
      sortedCap[i] = capPoints[scratchCapOrder[i]];
    }

    nextFaces.push({
      verts: sortedCap,
      normal: [planeNormal[0], planeNormal[1], planeNormal[2]],
      isOuter: false,
    });
  }

  return nextFaces;
}

/**
 * Deterministic fast PRNG (Mulberry32) so fractures are reproducible and unit-testable.
 */
function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Evaluates A * v where A = sum_k lambda_k * (u_k outer u_k) is a 3x3 symmetric positive-definite
 * brittle cleavage metric tensor. Guarantees exact 100% volume-conserving convex Voronoi partitioning
 * while producing acute radial wedges and elongated glass splinters instead of isotropic cubes.
 */
function applyAnisotropicMetric(
  v: Vec3,
  u1: Vec3,
  u2: Vec3,
  u3: Vec3,
  l1: number,
  l2: number,
  l3: number
): Vec3 {
  const d1 = dot3(v, u1) * l1;
  const d2 = dot3(v, u2) * l2;
  const d3 = dot3(v, u3) * l3;
  return [
    u1[0] * d1 + u2[0] * d2 + u3[0] * d3,
    u1[1] * d1 + u2[1] * d2 + u3[1] * d3,
    u1[2] * d1 + u2[2] * d2 + u3[2] * d3,
  ];
}

function buildShardGeometryFromFaces(
  faces: PolyFace[],
  clampedImpact: Vec3
): GlassShardGeometry | null {
  if (faces.length < 4) return null;

  const uniqueVerts: Vec3[] = [];
  for (let fi = 0; fi < faces.length; fi++) {
    const fv = faces[fi].verts;
    for (let vi = 0; vi < fv.length; vi++) {
      const v = fv[vi];
      const vx = v[0];
      const vy = v[1];
      const vz = v[2];
      let exists = false;
      for (let ui = 0; ui < uniqueVerts.length; ui++) {
        const u = uniqueVerts[ui];
        if (distBelow2e4Exact(vx, vy, vz, u[0], u[1], u[2])) {
          exists = true;
          break;
        }
      }
      if (!exists) uniqueVerts.push(v);
    }
  }
  if (uniqueVerts.length < 4) return null;

  let refX = 0, refY = 0, refZ = 0;
  for (let ui = 0; ui < uniqueVerts.length; ui++) {
    const u = uniqueVerts[ui];
    refX += u[0];
    refY += u[1];
    refZ += u[2];
  }
  refX /= uniqueVerts.length;
  refY /= uniqueVerts.length;
  refZ /= uniqueVerts.length;

  let totalVolume = 0;
  let cx = 0, cy = 0, cz = 0;

  for (let fi = 0; fi < faces.length; fi++) {
    const fv = faces[fi].verts;
    const v0 = fv[0];
    const v0x = v0[0], v0y = v0[1], v0z = v0[2];
    const ax = v0x - refX, ay = v0y - refY, az = v0z - refZ;
    for (let t = 1; t < fv.length - 1; t++) {
      const v1 = fv[t];
      const v2 = fv[t + 1];

      const bx = v1[0] - refX, by = v1[1] - refY, bz = v1[2] - refZ;
      const dx = v2[0] - refX, dy = v2[1] - refY, dz = v2[2] - refZ;

      const det =
        ax * (by * dz - bz * dy) -
        ay * (bx * dz - bz * dx) +
        az * (bx * dy - by * dx);
      const tetVol = Math.abs(det) / 6.0;
      totalVolume += tetVol;
      cx += tetVol * (refX + v0x + v1[0] + v2[0]) * 0.25;
      cy += tetVol * (refY + v0y + v1[1] + v2[1]) * 0.25;
      cz += tetVol * (refZ + v0z + v1[2] + v2[2]) * 0.25;
    }
  }

  if (totalVolume < 1e-6) return null;
  const lc0 = cx / totalVolume;
  const lc1 = cy / totalVolume;
  const lc2 = cz / totalVolume;
  const localCentroid: Vec3 = [lc0, lc1, lc2];
  const impactDist = Math.hypot(
    lc0 - clampedImpact[0],
    lc1 - clampedImpact[1],
    lc2 - clampedImpact[2]
  );

  let triCount = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    triCount += Math.max(0, faces[fi].verts.length - 2);
  }
  if (triCount < 4) {
    return null;
  }
  const vertexCount = Math.min(SHARD_SLOT_STRIDE_VERTS, triCount * 3);

  const hullPoints = new Float32Array(uniqueVerts.length * 3);
  for (let k = 0; k < uniqueVerts.length; k++) {
    const u = uniqueVerts[k];
    hullPoints[k * 3 + 0] = u[0] - lc0;
    hullPoints[k * 3 + 1] = u[1] - lc1;
    hullPoints[k * 3 + 2] = u[2] - lc2;
  }

  // Pack up to MAX_SHARD_PLANES (12) bounding half-space planes sorted by face area descending
  const faceCount = faces.length;
  const sortedFacesForPlanes: Array<{ f: PolyFace; area: number }> = new Array(faceCount);
  for (let fi = 0; fi < faceCount; fi++) {
    const f = faces[fi];
    sortedFacesForPlanes[fi] = { f, area: computePolygonArea(f.verts) };
  }
  sortedFacesForPlanes.sort((a, b) => b.area - a.area);
  const planeCount = Math.min(MAX_SHARD_PLANES, faceCount);
  const planes = new Float32Array(MAX_SHARD_PLANES * 4);
  for (let pIdx = 0; pIdx < planeCount; pIdx++) {
    const f = sortedFacesForPlanes[pIdx].f;
    const v0 = f.verts[0];
    const distFromCentroid = Math.max(
      2e-4,
      f.normal[0] * (v0[0] - lc0) +
        f.normal[1] * (v0[1] - lc1) +
        f.normal[2] * (v0[2] - lc2)
    );
    planes[pIdx * 4 + 0] = f.normal[0];
    planes[pIdx * 4 + 1] = f.normal[1];
    planes[pIdx * 4 + 2] = f.normal[2];
    planes[pIdx * 4 + 3] = f.isOuter ? -distFromCentroid : distFromCentroid;
  }

  const renderVertices = new Float32Array(vertexCount * 10);
  let ptr = 0;
  const maxFloatLen = vertexCount * 10;
  for (let si = 0; si < faceCount; si++) {
    if (ptr + 30 > maxFloatLen) break;
    const f = sortedFacesForPlanes[si].f;
    const fv = f.verts;
    const nx = f.normal[0];
    const ny = f.normal[1];
    const nz = f.normal[2];
    const outerFlag = f.isOuter ? 1.0 : 0.0;
    const v0 = fv[0];
    const v0x = v0[0], v0y = v0[1], v0z = v0[2];
    const v0rx = v0x - lc0, v0ry = v0y - lc1, v0rz = v0z - lc2;

    for (let t = 1; t < fv.length - 1; t++) {
      if (ptr + 30 > maxFloatLen) break;
      const v1 = fv[t];
      const v2 = fv[t + 1];
      renderVertices[ptr++] = v0rx;
      renderVertices[ptr++] = v0ry;
      renderVertices[ptr++] = v0rz;
      renderVertices[ptr++] = nx;
      renderVertices[ptr++] = ny;
      renderVertices[ptr++] = nz;
      renderVertices[ptr++] = v0x;
      renderVertices[ptr++] = v0y;
      renderVertices[ptr++] = v0z;
      renderVertices[ptr++] = outerFlag;

      renderVertices[ptr++] = v1[0] - lc0;
      renderVertices[ptr++] = v1[1] - lc1;
      renderVertices[ptr++] = v1[2] - lc2;
      renderVertices[ptr++] = nx;
      renderVertices[ptr++] = ny;
      renderVertices[ptr++] = nz;
      renderVertices[ptr++] = v1[0];
      renderVertices[ptr++] = v1[1];
      renderVertices[ptr++] = v1[2];
      renderVertices[ptr++] = outerFlag;

      renderVertices[ptr++] = v2[0] - lc0;
      renderVertices[ptr++] = v2[1] - lc1;
      renderVertices[ptr++] = v2[2] - lc2;
      renderVertices[ptr++] = nx;
      renderVertices[ptr++] = ny;
      renderVertices[ptr++] = nz;
      renderVertices[ptr++] = v2[0];
      renderVertices[ptr++] = v2[1];
      renderVertices[ptr++] = v2[2];
      renderVertices[ptr++] = outerFlag;
    }
  }

  return {
    localCentroid,
    volume: totalVolume,
    impactDist,
    hullPoints,
    planes,
    planeCount,
    renderVertices,
    vertexCount,
  };
}

/**
 * Fractures a stained-glass Jenga block into watertight 3D Voronoi convex polyhedron shards:
 * - Zone 1 (Hertzian crush core): small angular glass chips tightly clustered at `localImpactPoint`
 * - Zone 2 (Radial cleavage fan): elongated, acute diagonal glass wedges & splinters
 * - Zone 3 (Far-end remnants): larger cast-glass end pieces (with optional 1-generation 2-plane cleavage > 10g on high-energy impacts)
 */
export function fractureBlockVoronoi(
  localImpactPoint: Vec3,
  shardCount: number = 12,
  seed: number = 1337,
  hx: number = BLOCK_HALF_X,
  hy: number = BLOCK_HALF_Y,
  hz: number = BLOCK_HALF_Z,
  _motif: number = 0,
  impactEnergy01: number = 0.65,
  allowSecondaryCleavage: boolean = false
): GlassShardGeometry[] {
  const rand = mulberry32(seed);
  const energy = Math.max(0.0, Math.min(1.0, impactEnergy01));

  const clampedImpact: Vec3 = [
    Math.max(-hx * 0.92, Math.min(hx * 0.92, localImpactPoint[0])),
    Math.max(-hy * 0.90, Math.min(hy * 0.90, localImpactPoint[1])),
    Math.max(-hz * 0.92, Math.min(hz * 0.92, localImpactPoint[2])),
  ];

  // Build an oblique brittle cleavage basis oriented away from the impact point along the glass bar:
  // Compressing metric eigenvalue l1 along u1 elongates shards into sharp radial lancet wedges & splinters,
  // while oblique yaw/pitch shear prevents unnatural 90-degree bread-loaf slices.
  const dirFromImpact: Vec3 = [
    -clampedImpact[0] * 0.85 + (rand() - 0.5) * 0.35,
    -clampedImpact[1] * 0.35 + (rand() - 0.5) * 0.28,
    -clampedImpact[2] * 0.65 + (rand() - 0.5) * 0.42,
  ];
  const rawLen = Math.hypot(dirFromImpact[0], dirFromImpact[1], dirFromImpact[2]);
  const shearSign = rand() > 0.5 ? 1 : -1;
  const shearYaw = shearSign * (0.34 + rand() * 0.22); // ~20-32 deg oblique brittle glass cleavage angle
  const shearPitch = (rand() - 0.5) * 0.32;

  const baseAxis: Vec3 =
    rawLen > 0.08
      ? normalize3([
          dirFromImpact[0] + Math.sin(shearYaw) * 0.35,
          dirFromImpact[1] + shearPitch * 0.25,
          dirFromImpact[2] + Math.sin(shearYaw) * 0.42,
        ])
      : normalize3([Math.cos(shearYaw), shearPitch, Math.sin(shearYaw)]);

  const upRef: Vec3 = Math.abs(baseAxis[1]) < 0.85 ? [0, 1, 0] : [0, 0, 1];
  const u3 = normalize3(cross3(baseAxis, upRef)); // Lateral cross-bar axis
  const u2 = normalize3(cross3(u3, baseAxis));    // Thickness/oblique cant axis
  const u1 = baseAxis;                            // Primary longitudinal splinter propagation axis

  // Eigenvalues of SPD metric A: smaller l1 => longer radial lancet splinters; larger l3 => narrower acute wedges
  const l1 = 0.38;
  const l2 = 1.35;
  const l3 = 2.25;

  const seeds: Vec3[] = [];
  const seedZones: number[] = [];
  const targetCount = Math.max(4, Math.min(22, shardCount));

  const farCount = energy > 0.8 ? 2 : 1;
  const crushCount = Math.max(2, Math.round((targetCount - farCount) * (0.44 + energy * 0.12)));
  const fanCount = Math.max(1, targetCount - farCount - crushCount);

  for (let i = 0; i < targetCount; i++) {
    const zone = i < crushCount ? 1 : i < crushCount + fanCount ? 2 : 3;
    const minSep = zone === 1 ? 0.013 : 0.022;

    for (let attempt = 0; attempt < 6; attempt++) {
      let sx = 0;
      let sy = 0;
      let sz = 0;

      if (zone === 1) {
        const spreadBoost = attempt * 0.065;
        const rFrac = Math.pow(rand(), 1.45) * (0.21 + energy * 0.09 + spreadBoost);
        const rx = (rand() * 2 - 1) * hx * 0.94;
        const ry = (rand() * 2 - 1) * hy * 0.90;
        const rz = (rand() * 2 - 1) * hz * 0.92;
        sx = clampedImpact[0] + (rx - clampedImpact[0]) * rFrac;
        sy = clampedImpact[1] + (ry - clampedImpact[1]) * rFrac;
        sz = clampedImpact[2] + (rz - clampedImpact[2]) * rFrac;
      } else if (zone === 2) {
        const t = 0.22 + rand() * (0.42 + energy * 0.22);
        const rx = (rand() * 2 - 1) * hx * 0.92;
        const ry = (rand() * 2 - 1) * hy * 0.88;
        const rz = (rand() * 2 - 1) * hz * 0.92;
        sx = clampedImpact[0] + (rx - clampedImpact[0]) * t;
        sy = clampedImpact[1] + (ry - clampedImpact[1]) * t;
        sz = clampedImpact[2] + (rz - clampedImpact[2]) * t;
      } else {
        const farSignX = clampedImpact[0] >= 0 ? -1 : 1;
        const spread = 0.58 + rand() * 0.30;
        sx = farSignX * hx * spread + (rand() - 0.5) * hx * 0.22;
        sy = (rand() * 2 - 1) * hy * 0.68;
        sz = (rand() * 2 - 1) * hz * 0.74;
      }

      sx = Math.max(-hx * 0.95, Math.min(hx * 0.95, sx));
      sy = Math.max(-hy * 0.92, Math.min(hy * 0.92, sy));
      sz = Math.max(-hz * 0.95, Math.min(hz * 0.95, sz));

      let tooClose = false;
      for (const prev of seeds) {
        if (Math.hypot(sx - prev[0], sy - prev[1], sz - prev[2]) < minSep) {
          tooClose = true;
          break;
        }
      }
      if (!tooClose) {
        seeds.push([sx, sy, sz]);
        seedZones.push(zone);
        break;
      }
    }
  }

  if (seeds.length < 4) {
    seeds.push(
      [-hx * 0.55, 0, -hz * 0.35],
      [hx * 0.55, 0, hz * 0.35],
      [0, hy * 0.35, -hz * 0.35],
      [0, -hy * 0.35, hz * 0.35]
    );
    seedZones.push(2, 2, 2, 3);
  }

  const aSeeds: Vec3[] = seeds.map((s) => applyAnisotropicMetric(s, u1, u2, u3, l1, l2, l3));
  const quadSeeds: number[] = seeds.map((s, idx) => dot3(s, aSeeds[idx]));

  const shards: GlassShardGeometry[] = [];

  const neighbors: Array<{
    j: number;
    dax: number;
    day: number;
    daz: number;
    metricDist2: number;
  }> = new Array(Math.max(0, seeds.length - 1));

  for (let i = 0; i < seeds.length; i++) {
    let faces = createInitialBoxFaces(hx, hy, hz);
    const si = seeds[i];
    const asi = aSeeds[i];
    const qi = quadSeeds[i];

    let nbCount = 0;
    for (let j = 0; j < seeds.length; j++) {
      if (j === i) continue;
      const sj = seeds[j];
      const asj = aSeeds[j];
      const dx = sj[0] - si[0];
      const dy = sj[1] - si[1];
      const dz = sj[2] - si[2];
      const dax = asj[0] - asi[0];
      const day = asj[1] - asi[1];
      const daz = asj[2] - asi[2];
      neighbors[nbCount++] = {
        j,
        dax,
        day,
        daz,
        metricDist2: dx * dax + dy * day + dz * daz,
      };
    }
    neighbors.length = nbCount;
    neighbors.sort((a, b) => a.metricDist2 - b.metricDist2);

    const normalTmp: Vec3 = [0, 0, 0];
    for (let nIdx = 0; nIdx < nbCount; nIdx++) {
      if (faces.length < 4) break;
      const nb = neighbors[nIdx];
      const gLen = Math.hypot(nb.dax, nb.day, nb.daz);
      if (gLen < 1e-6) continue;

      normalTmp[0] = nb.dax / gLen;
      normalTmp[1] = nb.day / gLen;
      normalTmp[2] = nb.daz / gLen;
      const planeD = (0.5 * (quadSeeds[nb.j] - qi)) / gLen;
      faces = clipConvexPolyhedron(faces, normalTmp, planeD, true);
    }

    const baseGeom = buildShardGeometryFromFaces(faces, clampedImpact);
    if (!baseGeom) continue;

    // Secondary 2-plane cleavage on Zone-3 remnants > 10g (0.010 kg) on high-speed impacts
    if (
      allowSecondaryCleavage &&
      seedZones[i] === 3 &&
      baseGeom.volume * GLASS_DENSITY > 0.01
    ) {
      const c0 = baseGeom.localCentroid;
      const n1 = normalize3([
        u3[0] * 0.82 + u1[0] * (0.35 + (rand() - 0.5) * 0.25),
        u2[1] * 0.35 + (rand() - 0.5) * 0.3,
        u3[2] * 0.82 + u1[2] * (0.35 + (rand() - 0.5) * 0.25),
      ]);
      const d1 = dot3(n1, c0);
      const negN1: Vec3 = [-n1[0], -n1[1], -n1[2]];
      const facesA = clipConvexPolyhedron(faces, n1, d1);
      const facesB = clipConvexPolyhedron(faces, negN1, -d1);
      const geomA = buildShardGeometryFromFaces(facesA, clampedImpact);
      const geomB = buildShardGeometryFromFaces(facesB, clampedImpact);

      if (
        geomA &&
        geomB &&
        Math.abs(geomA.volume + geomB.volume - baseGeom.volume) < 5e-6
      ) {
        // Second oblique cleavage plane across the heavier sub-remnant
        const targetFaces = geomA.volume >= geomB.volume ? facesA : facesB;
        const targetGeom = geomA.volume >= geomB.volume ? geomA : geomB;
        const keepGeom = geomA.volume >= geomB.volume ? geomB : geomA;

        const n2 = normalize3([
          u1[0] * 0.72 - u3[0] * 0.52 + (rand() - 0.5) * 0.24,
          u2[1] * 0.48 + (rand() - 0.5) * 0.28,
          u1[2] * 0.72 - u3[2] * 0.52 + (rand() - 0.5) * 0.24,
        ]);
        const d2 = dot3(n2, targetGeom.localCentroid);
        const negN2: Vec3 = [-n2[0], -n2[1], -n2[2]];
        const sub1 = buildShardGeometryFromFaces(
          clipConvexPolyhedron(targetFaces, n2, d2),
          clampedImpact
        );
        const sub2 = buildShardGeometryFromFaces(
          clipConvexPolyhedron(targetFaces, negN2, -d2),
          clampedImpact
        );

        if (
          sub1 &&
          sub2 &&
          Math.abs(keepGeom.volume + sub1.volume + sub2.volume - baseGeom.volume) < 8e-6
        ) {
          shards.push(keepGeom, sub1, sub2);
          continue;
        }
        shards.push(geomA, geomB);
        continue;
      }
    }

    shards.push(baseGeom);
  }

  return shards;
}

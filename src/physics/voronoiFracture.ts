import { Vec3 } from '../math/vecMat';
import { BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z } from './constants';

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
  /** Unique convex hull points relative to localCentroid (for Rapier ColliderDesc.convexHull) */
  hullPoints: Float32Array;
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

/**
 * Clips a closed 3D convex polyhedron against the half-space dot(n, p) <= planeD.
 * Any clipped cross-section receives a watertight CCW cap face with outward normal `n` and `isOuter = false`.
 */
function clipConvexPolyhedron(
  faces: PolyFace[],
  planeNormal: Vec3,
  planeD: number
): PolyFace[] {
  const nextFaces: PolyFace[] = [];
  const capPoints: Vec3[] = [];

  const addCapPoint = (p: Vec3) => {
    for (let i = 0; i < capPoints.length; i++) {
      const q = capPoints[i];
      if (
        Math.abs(p[0] - q[0]) < 1e-4 &&
        Math.abs(p[1] - q[1]) < 1e-4 &&
        Math.abs(p[2] - q[2]) < 1e-4
      ) {
        return;
      }
    }
    capPoints.push(p);
  };

  for (let fIdx = 0; fIdx < faces.length; fIdx++) {
    const face = faces[fIdx];
    const verts = face.verts;
    const count = verts.length;
    if (count < 3) continue;

    const clipped: Vec3[] = [];
    for (let i = 0; i < count; i++) {
      const a = verts[i];
      const b = verts[(i + 1) % count];
      const da = dot3(planeNormal, a) - planeD;
      const db = dot3(planeNormal, b) - planeD;

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
      const prev = clean.length > 0 ? clean[clean.length - 1] : clipped[clipped.length - 1];
      if (
        clean.length === 0 ||
        Math.hypot(cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]) > 1e-4
      ) {
        clean.push(cur);
      }
    }
    if (
      clean.length >= 3 &&
      Math.hypot(
        clean[0][0] - clean[clean.length - 1][0],
        clean[0][1] - clean[clean.length - 1][1],
        clean[0][2] - clean[clean.length - 1][2]
      ) <= 1e-4
    ) {
      clean.pop();
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
  if (capPoints.length >= 3) {
    let cx = 0, cy = 0, cz = 0;
    for (const p of capPoints) {
      cx += p[0];
      cy += p[1];
      cz += p[2];
    }
    cx /= capPoints.length;
    cy /= capPoints.length;
    cz /= capPoints.length;

    // Build orthonormal tangent basis (u, v) on plane such that cross(u, v) == planeNormal
    const ref: Vec3 = Math.abs(planeNormal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const uRaw = cross3(ref, planeNormal);
    const uLen = Math.hypot(uRaw[0], uRaw[1], uRaw[2]) || 1;
    const u: Vec3 = [uRaw[0] / uLen, uRaw[1] / uLen, uRaw[2] / uLen];
    const v = cross3(planeNormal, u);

    const sortedCap = capPoints
      .map((p) => {
        const rx = p[0] - cx;
        const ry = p[1] - cy;
        const rz = p[2] - cz;
        const pu = rx * u[0] + ry * u[1] + rz * u[2];
        const pv = rx * v[0] + ry * v[1] + rz * v[2];
        return { p, angle: Math.atan2(pv, pu) };
      })
      .sort((a, b) => a.angle - b.angle)
      .map((item) => item.p);

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
 * Fractures a stained-glass Jenga block into watertight 3D Voronoi convex polyhedron shards,
 * clustering smaller sharp splinters near `localImpactPoint` and larger prismatic wedges further out.
 */
export function fractureBlockVoronoi(
  localImpactPoint: Vec3,
  shardCount: number = 12,
  seed: number = 1337,
  hx: number = BLOCK_HALF_X,
  hy: number = BLOCK_HALF_Y,
  hz: number = BLOCK_HALF_Z
): GlassShardGeometry[] {
  const rand = mulberry32(seed);
  const clampedImpact: Vec3 = [
    Math.max(-hx * 0.92, Math.min(hx * 0.92, localImpactPoint[0])),
    Math.max(-hy * 0.92, Math.min(hy * 0.92, localImpactPoint[1])),
    Math.max(-hz * 0.92, Math.min(hz * 0.92, localImpactPoint[2])),
  ];

  // Generate 3D Voronoi seeds: ~45% clustered near impact point, ~55% distributed across block length
  const seeds: Vec3[] = [];
  const targetCount = Math.max(4, Math.min(20, shardCount));

  for (let i = 0; i < targetCount; i++) {
    const clusterWeight = i < Math.ceil(targetCount * 0.45) ? Math.pow(rand(), 1.8) * 0.42 : 0.35 + rand() * 0.65;
    const rx = (rand() * 2 - 1) * hx * 0.92;
    const ry = (rand() * 2 - 1) * hy * 0.88;
    const rz = (rand() * 2 - 1) * hz * 0.90;

    const sx = clampedImpact[0] + (rx - clampedImpact[0]) * clusterWeight;
    const sy = clampedImpact[1] + (ry - clampedImpact[1]) * clusterWeight;
    const sz = clampedImpact[2] + (rz - clampedImpact[2]) * clusterWeight;

    // Ensure minimum separation between Voronoi seeds to prevent zero-thickness degenerate slivers
    let tooClose = false;
    for (const prev of seeds) {
      if (Math.hypot(sx - prev[0], sy - prev[1], sz - prev[2]) < 0.025) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      seeds.push([sx, sy, sz]);
    }
  }

  // Fallback guarantee of at least 4 seeds
  if (seeds.length < 4) {
    seeds.push([-hx * 0.5, 0, -hz * 0.3], [hx * 0.5, 0, hz * 0.3], [0, hy * 0.3, -hz * 0.3], [0, -hy * 0.3, hz * 0.3]);
  }

  const shards: GlassShardGeometry[] = [];

  for (let i = 0; i < seeds.length; i++) {
    let faces = createInitialBoxFaces(hx, hy, hz);
    const si = seeds[i];

    // Sort other seeds by distance to si so closer neighbors clip first (faster early culling)
    const neighbors = seeds
      .map((sj, j) => ({
        j,
        sj,
        dist2: (sj[0] - si[0]) ** 2 + (sj[1] - si[1]) ** 2 + (sj[2] - si[2]) ** 2,
      }))
      .filter((item) => item.j !== i)
      .sort((a, b) => a.dist2 - b.dist2);

    for (const { sj } of neighbors) {
      if (faces.length < 4) break;
      const dx = sj[0] - si[0];
      const dy = sj[1] - si[1];
      const dz = sj[2] - si[2];
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-6) continue;

      const normal: Vec3 = [dx / len, dy / len, dz / len];
      const mid: Vec3 = [
        (si[0] + sj[0]) * 0.5,
        (si[1] + sj[1]) * 0.5,
        (si[2] + sj[2]) * 0.5,
      ];
      const planeD = dot3(normal, mid);
      faces = clipConvexPolyhedron(faces, normal, planeD);
    }

    if (faces.length < 4) continue;

    // Gather unique hull vertices & compute exact volume + centroid via tetrahedral decomposition
    const uniqueVerts: Vec3[] = [];
    for (const f of faces) {
      for (const v of f.verts) {
        let exists = false;
        for (const u of uniqueVerts) {
          if (Math.hypot(v[0] - u[0], v[1] - u[1], v[2] - u[2]) < 2e-4) {
            exists = true;
            break;
          }
        }
        if (!exists) uniqueVerts.push(v);
      }
    }
    if (uniqueVerts.length < 4) continue;

    // Approximate interior reference point
    let refX = 0, refY = 0, refZ = 0;
    for (const u of uniqueVerts) {
      refX += u[0];
      refY += u[1];
      refZ += u[2];
    }
    refX /= uniqueVerts.length;
    refY /= uniqueVerts.length;
    refZ /= uniqueVerts.length;

    let totalVolume = 0;
    let cx = 0, cy = 0, cz = 0;

    for (const f of faces) {
      const fv = f.verts;
      const v0 = fv[0];
      for (let t = 1; t < fv.length - 1; t++) {
        const v1 = fv[t];
        const v2 = fv[t + 1];

        const ax = v0[0] - refX, ay = v0[1] - refY, az = v0[2] - refZ;
        const bx = v1[0] - refX, by = v1[1] - refY, bz = v1[2] - refZ;
        const dx = v2[0] - refX, dy = v2[1] - refY, dz = v2[2] - refZ;

        // Signed volume of tetrahedron (ref, v0, v1, v2); outward CCW faces give positive volume
        const det =
          ax * (by * dz - bz * dy) -
          ay * (bx * dz - bz * dx) +
          az * (bx * dy - by * dx);
        const tetVol = Math.abs(det) / 6.0;
        totalVolume += tetVol;
        cx += tetVol * (refX + v0[0] + v1[0] + v2[0]) * 0.25;
        cy += tetVol * (refY + v0[1] + v1[1] + v2[1]) * 0.25;
        cz += tetVol * (refZ + v0[2] + v1[2] + v2[2]) * 0.25;
      }
    }

    if (totalVolume < 1e-6) continue;
    const localCentroid: Vec3 = [cx / totalVolume, cy / totalVolume, cz / totalVolume];

    const hullPoints = new Float32Array(uniqueVerts.length * 3);
    for (let k = 0; k < uniqueVerts.length; k++) {
      hullPoints[k * 3 + 0] = uniqueVerts[k][0] - localCentroid[0];
      hullPoints[k * 3 + 1] = uniqueVerts[k][1] - localCentroid[1];
      hullPoints[k * 3 + 2] = uniqueVerts[k][2] - localCentroid[2];
    }

    // Build flat-faceted render triangle list (10 floats/vertex)
    let triCount = 0;
    for (const f of faces) {
      triCount += Math.max(0, f.verts.length - 2);
    }
    const renderVertices = new Float32Array(triCount * 3 * 10);
    let ptr = 0;
    for (const f of faces) {
      const fv = f.verts;
      const nx = f.normal[0];
      const ny = f.normal[1];
      const nz = f.normal[2];
      const outerFlag = f.isOuter ? 1.0 : 0.0;
      const v0 = fv[0];

      for (let t = 1; t < fv.length - 1; t++) {
        const tri = [v0, fv[t], fv[t + 1]];
        for (const v of tri) {
          renderVertices[ptr++] = v[0] - localCentroid[0];
          renderVertices[ptr++] = v[1] - localCentroid[1];
          renderVertices[ptr++] = v[2] - localCentroid[2];
          renderVertices[ptr++] = nx;
          renderVertices[ptr++] = ny;
          renderVertices[ptr++] = nz;
          renderVertices[ptr++] = v[0];
          renderVertices[ptr++] = v[1];
          renderVertices[ptr++] = v[2];
          renderVertices[ptr++] = outerFlag;
        }
      }
    }

    shards.push({
      localCentroid,
      volume: totalVolume,
      hullPoints,
      renderVertices,
      vertexCount: triCount * 3,
    });
  }

  return shards;
}

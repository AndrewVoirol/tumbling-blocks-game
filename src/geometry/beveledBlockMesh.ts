import { BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z } from '../physics/constants';

export interface MeshData {
  /** Interleaved [px, py, pz, nx, ny, nz] per vertex (6 floats = 24 bytes) */
  vertices: Float32Array;
  indices: Uint16Array;
  vertexCount: number;
  indexCount: number;
}

/**
 * Generates a smooth-chamfered rectangular glass prism mesh at exact Jenga scale
 * (7.5 cm x 1.5 cm x 2.5 cm -> half-extents 0.375 x 0.075 x 0.125 dm).
 * A rounded bevel along all 12 edges and 8 corners produces authentic specular highlights
 * along the cut glass edges and solder seams.
 */
export function createBeveledBlockMesh(
  hx: number = BLOCK_HALF_X,
  hy: number = BLOCK_HALF_Y,
  hz: number = BLOCK_HALF_Z,
  bevel: number = 0.0085, // ~0.85 mm glass bevel chamfer
  segments: number = 3
): MeshData {
  const verts: number[] = [];
  const indices: number[] = [];

  // Parameterize each face on [-1, 1]^2 with clustered edge coordinates and project point onto inner box + bevel sphere!
  const paramStops: number[] = [-1.0];
  for (let i = 1; i <= segments; i++) {
    paramStops.push(-1.0 + (i / segments) * 0.16);
  }
  for (let i = segments; i >= 1; i--) {
    paramStops.push(1.0 - (i / segments) * 0.16);
  }
  paramStops.push(1.0);

  const mSteps = paramStops.length;
  const mapCubePointToRoundedBox = (cx: number, cy: number, cz: number) => {
    // Map unit cube [-1, 1]^3 onto exact box [-hx, hx] x [-hy, hy] x [-hz, hz],
    // then clamp to inner core [-hx+b, hx-b] to get exact signed distance surface point + outward normal!
    const px = cx * hx;
    const py = cy * hy;
    const pz = cz * hz;

    const ix = Math.max(-(hx - bevel), Math.min(hx - bevel, px));
    const iy = Math.max(-(hy - bevel), Math.min(hy - bevel, py));
    const iz = Math.max(-(hz - bevel), Math.min(hz - bevel, pz));

    let dx = px - ix;
    let dy = py - iy;
    let dz = pz - iz;
    const dLen = Math.hypot(dx, dy, dz);
    if (dLen < 1e-7) {
      // Pure flat face center
      if (Math.abs(cx) >= Math.abs(cy) && Math.abs(cx) >= Math.abs(cz)) {
        dx = Math.sign(cx); dy = 0; dz = 0;
      } else if (Math.abs(cy) >= Math.abs(cx) && Math.abs(cy) >= Math.abs(cz)) {
        dx = 0; dy = Math.sign(cy); dz = 0;
      } else {
        dx = 0; dy = 0; dz = Math.sign(cz);
      }
    } else {
      dx /= dLen;
      dy /= dLen;
      dz /= dLen;
    }

    return {
      x: ix + dx * bevel,
      y: iy + dy * bevel,
      z: iz + dz * bevel,
      nx: dx,
      ny: dy,
      nz: dz,
    };
  };

  const emitCubeFace = (
    evalCube: (u: number, v: number) => [number, number, number],
    flip: boolean
  ) => {
    const base = verts.length / 6;
    for (let iv = 0; iv < mSteps; iv++) {
      const v = paramStops[iv];
      for (let iu = 0; iu < mSteps; iu++) {
        const u = paramStops[iu];
        const [cx, cy, cz] = evalCube(u, v);
        const p = mapCubePointToRoundedBox(cx, cy, cz);
        verts.push(p.x, p.y, p.z, p.nx, p.ny, p.nz);
      }
    }
    for (let iv = 0; iv < mSteps - 1; iv++) {
      for (let iu = 0; iu < mSteps - 1; iu++) {
        const i00 = base + iv * mSteps + iu;
        const i10 = i00 + 1;
        const i01 = i00 + mSteps;
        const i11 = i01 + 1;
        if (!flip) {
          indices.push(i00, i10, i11, i00, i11, i01);
        } else {
          indices.push(i00, i11, i10, i00, i01, i11);
        }
      }
    }
  };

  // +Z (front), -Z (back), +X (right), -X (left), +Y (top), -Y (bottom) with outward CCW winding
  emitCubeFace((u, v) => [u, v, 1], false);
  emitCubeFace((u, v) => [u, v, -1], true);
  emitCubeFace((u, v) => [1, v, -u], false);
  emitCubeFace((u, v) => [-1, v, u], false);
  emitCubeFace((u, v) => [u, 1, -v], false);
  emitCubeFace((u, v) => [u, -1, v], false);

  return {
    vertices: new Float32Array(verts),
    indices: new Uint16Array(indices),
    vertexCount: verts.length / 6,
    indexCount: indices.length,
  };
}

/**
 * Crisp 24-vertex sharp box mesh for shadow/caustic transmission pass and ghost slot guides.
 */
export function createSimpleBoxMesh(
  hx: number = BLOCK_HALF_X,
  hy: number = BLOCK_HALF_Y,
  hz: number = BLOCK_HALF_Z
): MeshData {
  // prettier-ignore
  const vertices = new Float32Array([
    // +Z
    -hx, -hy,  hz,  0,  0,  1,
     hx, -hy,  hz,  0,  0,  1,
     hx,  hy,  hz,  0,  0,  1,
    -hx,  hy,  hz,  0,  0,  1,
    // -Z
     hx, -hy, -hz,  0,  0, -1,
    -hx, -hy, -hz,  0,  0, -1,
    -hx,  hy, -hz,  0,  0, -1,
     hx,  hy, -hz,  0,  0, -1,
    // +X
     hx, -hy,  hz,  1,  0,  0,
     hx, -hy, -hz,  1,  0,  0,
     hx,  hy, -hz,  1,  0,  0,
     hx,  hy,  hz,  1,  0,  0,
    // -X
    -hx, -hy, -hz, -1,  0,  0,
    -hx, -hy,  hz, -1,  0,  0,
    -hx,  hy,  hz, -1,  0,  0,
    -hx,  hy, -hz, -1,  0,  0,
    // +Y
    -hx,  hy,  hz,  0,  1,  0,
     hx,  hy,  hz,  0,  1,  0,
     hx,  hy, -hz,  0,  1,  0,
    -hx,  hy, -hz,  0,  1,  0,
    // -Y
    -hx, -hy, -hz,  0, -1,  0,
     hx, -hy, -hz,  0, -1,  0,
     hx, -hy,  hz,  0, -1,  0,
    -hx, -hy,  hz,  0, -1,  0,
  ]);

  const indices = new Uint16Array(36);
  for (let f = 0; f < 6; f++) {
    const b = f * 4;
    const o = f * 6;
    indices[o + 0] = b + 0;
    indices[o + 1] = b + 1;
    indices[o + 2] = b + 2;
    indices[o + 3] = b + 0;
    indices[o + 4] = b + 2;
    indices[o + 5] = b + 3;
  }

  return {
    vertices,
    indices,
    vertexCount: 24,
    indexCount: 36,
  };
}

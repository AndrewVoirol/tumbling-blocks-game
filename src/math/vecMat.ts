// Zero-allocation column-major 4x4 matrix, 3D vector, quaternion, and ray-OBB math
// Uses WebGPU standard right-handed view space with [0, 1] NDC depth range.

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // [x, y, z, w]

export function mat4Create(): Float32Array {
  const out = new Float32Array(16);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  return out;
}

export function mat4Identity(out: Float32Array): Float32Array {
  out.fill(0);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  return out;
}

export function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
  out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
  out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
  out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  return out;
}

/**
 * WebGPU right-handed perspective matrix mapping view Z in [-near, -far] to NDC [0, 1].
 */
export function mat4PerspectiveZO(
  out: Float32Array,
  fovyRad: number,
  aspect: number,
  near: number,
  far: number
): Float32Array {
  const f = 1.0 / Math.tan(fovyRad * 0.5);
  const nf = 1.0 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1.0;
  out[14] = far * near * nf;
  return out;
}

/**
 * WebGPU right-handed orthographic matrix mapping view Z in [-near, -far] to NDC [0, 1].
 */
export function mat4OrthoZO(
  out: Float32Array,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number
): Float32Array {
  const lr = 1.0 / (left - right);
  const bt = 1.0 / (bottom - top);
  const nf = 1.0 / (near - far);
  out.fill(0);
  out[0] = -2.0 * lr;
  out[5] = -2.0 * bt;
  out[10] = nf;
  out[12] = (left + right) * lr;
  out[13] = (top + bottom) * bt;
  out[14] = near * nf;
  out[15] = 1.0;
  return out;
}

export function mat4LookAt(
  out: Float32Array,
  eye: Vec3,
  center: Vec3,
  up: Vec3
): Float32Array {
  let zx = eye[0] - center[0];
  let zy = eye[1] - center[1];
  let zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz);
  if (len < 1e-8) {
    return mat4Identity(out);
  }
  zx /= len;
  zy /= len;
  zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz);
  if (len < 1e-8) {
    xx = 1; xy = 0; xz = 0;
  } else {
    xx /= len; xy /= len; xz /= len;
  }

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;
  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;
  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

export function mat4FromRotationTranslationScale(
  out: Float32Array,
  q: Quat,
  v: Vec3,
  s: Vec3
): Float32Array {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const sx = s[0], sy = s[1], sz = s[2];

  out[0] = (1 - (yy + zz)) * sx;
  out[1] = (xy + wz) * sx;
  out[2] = (xz - wy) * sx;
  out[3] = 0;
  out[4] = (xy - wz) * sy;
  out[5] = (1 - (xx + zz)) * sy;
  out[6] = (yz + wx) * sy;
  out[7] = 0;
  out[8] = (xz + wy) * sz;
  out[9] = (yz - wx) * sz;
  out[10] = (1 - (xx + yy)) * sz;
  out[11] = 0;
  out[12] = v[0];
  out[13] = v[1];
  out[14] = v[2];
  out[15] = 1;
  return out;
}

/**
 * Rigid inverse (transpose rotation + inverse translation) for unscaled rigid body transforms.
 */
export function mat4FromRigidInverse(
  out: Float32Array,
  q: Quat,
  v: Vec3
): Float32Array {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;

  // Transposed rotation columns
  const r00 = 1 - (yy + zz);
  const r01 = xy - wz;
  const r02 = xz + wy;

  const r10 = xy + wz;
  const r11 = 1 - (xx + zz);
  const r12 = yz - wx;

  const r20 = xz - wy;
  const r21 = yz + wx;
  const r22 = 1 - (xx + yy);

  out[0] = r00;
  out[1] = r01;
  out[2] = r02;
  out[3] = 0;
  out[4] = r10;
  out[5] = r11;
  out[6] = r12;
  out[7] = 0;
  out[8] = r20;
  out[9] = r21;
  out[10] = r22;
  out[11] = 0;
  out[12] = -(r00 * v[0] + r10 * v[1] + r20 * v[2]);
  out[13] = -(r01 * v[0] + r11 * v[1] + r21 * v[2]);
  out[14] = -(r02 * v[0] + r12 * v[1] + r22 * v[2]);
  out[15] = 1;
  return out;
}

export function mat4Invert(out: Float32Array, a: Float32Array): Float32Array | null {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return null;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

export function quatFromAxisAngle(axis: Vec3, rad: number): Quat {
  const half = rad * 0.5;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

export function quatRotateVec3(q: Quat, v: Vec3): Vec3 {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  ];
}

export function quatInverseRotateVec3(q: Quat, v: Vec3): Vec3 {
  return quatRotateVec3([-q[0], -q[1], -q[2], q[3]], v);
}

export function unprojectScreenRay(
  ndcX: number,
  ndcY: number,
  invViewProj: Float32Array
): { origin: Vec3; dir: Vec3 } {
  const pNear = transformPointProjective(ndcX, ndcY, 0.0, invViewProj);
  const pFar = transformPointProjective(ndcX, ndcY, 1.0, invViewProj);
  const dx = pFar[0] - pNear[0];
  const dy = pFar[1] - pNear[1];
  const dz = pFar[2] - pNear[2];
  const len = Math.hypot(dx, dy, dz) || 1;
  return {
    origin: pNear,
    dir: [dx / len, dy / len, dz / len],
  };
}

function transformPointProjective(
  x: number,
  y: number,
  z: number,
  m: Float32Array
): Vec3 {
  const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1;
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

export interface RayOBBHit {
  t: number;
  worldPoint: Vec3;
  localPoint: Vec3;
  localNormal: Vec3;
  worldNormal: Vec3;
}

/**
 * Exact Ray vs Oriented Bounding Box intersection in world space.
 */
export function intersectRayOBB(
  rayOrigin: Vec3,
  rayDir: Vec3,
  boxPos: Vec3,
  boxQuat: Quat,
  halfExtents: Vec3
): RayOBBHit | null {
  const relOrigin: Vec3 = [
    rayOrigin[0] - boxPos[0],
    rayOrigin[1] - boxPos[1],
    rayOrigin[2] - boxPos[2],
  ];
  const localOrigin = quatInverseRotateVec3(boxQuat, relOrigin);
  const localDir = quatInverseRotateVec3(boxQuat, rayDir);

  let tMin = -Infinity;
  let tMax = Infinity;
  let hitAxis = 0;
  let hitSign = 1;

  for (let i = 0; i < 3; i++) {
    const o = localOrigin[i];
    const d = localDir[i];
    const h = halfExtents[i];
    if (Math.abs(d) < 1e-8) {
      if (o < -h || o > h) return null;
    } else {
      let t1 = (-h - o) / d;
      let t2 = (h - o) / d;
      let sign = -1;
      if (t1 > t2) {
        const tmp = t1;
        t1 = t2;
        t2 = tmp;
        sign = 1;
      }
      if (t1 > tMin) {
        tMin = t1;
        hitAxis = i;
        hitSign = sign;
      }
      if (t2 < tMax) {
        tMax = t2;
      }
      if (tMin > tMax) return null;
    }
  }

  if (tMax < 0) return null;
  const t = tMin >= 0 ? tMin : tMax;
  const localPoint: Vec3 = [
    localOrigin[0] + localDir[0] * t,
    localOrigin[1] + localDir[1] * t,
    localOrigin[2] + localDir[2] * t,
  ];
  const localNormal: Vec3 = [0, 0, 0];
  localNormal[hitAxis] = hitSign;
  const worldNormal = quatRotateVec3(boxQuat, localNormal);
  const worldPoint: Vec3 = [
    rayOrigin[0] + rayDir[0] * t,
    rayOrigin[1] + rayDir[1] * t,
    rayOrigin[2] + rayDir[2] * t,
  ];

  return { t, worldPoint, localPoint, localNormal, worldNormal };
}

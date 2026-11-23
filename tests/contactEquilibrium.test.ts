import { beforeAll, describe, expect, it, vi } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { OrbitCamera } from '../src/engine/OrbitCamera';
import { quatFromAxisAngle, quatRotateVec3, type Quat, type Vec3 } from '../src/math/vecMat';
import { BLOCK_HALF_X, BLOCK_HEIGHT } from '../src/physics/constants';
import {
  analyzeContactPolygon2D,
  areBlockObbsWithinGapDm,
  auditBlockEquilibriumGateB,
  BEVEL_ESCAPE_TAIL_LIMIT_M,
  BlockContactScratchBuffer,
  EQ_BLOCK_HALF_X_M,
  evaluateBlockContactEquilibrium,
  evaluateBlockQuasiStaticStress,
  isShardSupportingBlockObbDm,
  MU_GLASS_BEVEL_SLIP,
  MU_SLOT_PINCH_ESCAPE,
  QUASI_STATIC_CRAZE_STRESS_MPA,
} from '../src/physics/contactEquilibrium';
import { ensureMujocoInitialized, getMujocoRuntimeSync } from '../src/physics/mujocoRuntime';
import { PhysicsWorld } from '../src/physics/PhysicsWorld';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function quatNorm(q: Quat): Quat {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function quatMul(a: Quat, b: Quat): Quat {
  return quatNorm([
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ]);
}

function toWxyz(q: Quat): [number, number, number, number] {
  const n = quatNorm(q);
  return [n[3], n[0], n[1], n[2]];
}

describe('Coordinate-Free 3D Contact-Hull Equilibrium, Rounded-Bevel Escape & Quasi-Static Yield', () => {
  beforeAll(async () => {
    await ensureMujocoInitialized(false);
  }, 30000);

  it('Law 1A (Pure): releases 59-deg longitudinal, 79-deg upright-end, 8-corner & bevel-saddle knife-edges across 360 deg while preserving 2-block ramps and 6.5-deg seesaws', () => {
    const buf = new BlockContactScratchBuffer(8);
    const massKg = 0.0711;
    const weightN = massKg * 9.81;

    const azimuthsDeg = [0, 37, 90, 143, 215, 312];
    for (const azDeg of azimuthsDeg) {
      const yawQ = quatFromAxisAngle([0, 1, 0], (azDeg * Math.PI) / 180);

      // 1. 59-deg longitudinal chamfer edge stand along rotated X axis
      buf.clear(4);
      const roll59Q = quatMul(yawQ, quatFromAxisAngle([1, 0, 0], (59 * Math.PI) / 180));
      const [qw59, qx59, qy59, qz59] = toWxyz(roll59Q);
      const ex59 = quatRotateVec3(roll59Q, [1, 0, 0]);
      for (const s of [-0.032, 0.032]) {
        buf.addContact(
          0,
          ex59[0] * s,
          0,
          ex59[2] * s,
          -0.0001,
          0.5 * weightN,
          0,
          1,
          0,
          0,
          0.5 * weightN,
          0,
          true,
          0,
          false
        );
      }
      const act59 = evaluateBlockContactEquilibrium(
        0,
        buf,
        0,
        0.014,
        0,
        qw59,
        qx59,
        qy59,
        qz59,
        0,
        0,
        0,
        0,
        0,
        massKg
      );
      expect(act59.isUnbalanced).toBe(true);
      expect(act59.isKnifeEdgeRelease).toBe(true);
      expect(act59.allHullMinorWidthM).toBeLessThan(0.0024);
      expect(Math.hypot(act59.tauWorldX, act59.tauWorldZ)).toBeGreaterThan(1e-4);

      const gb59 = auditBlockEquilibriumGateB(
        0,
        buf,
        0,
        0.014,
        0,
        qw59,
        qx59,
        qy59,
        qz59,
        massKg,
        weightN
      );
      expect(gb59.pass).toBe(false);

      // 2. 79-deg upright-end pitch chamfer edge stand
      buf.clear(4);
      const pitch79Q = quatMul(yawQ, quatFromAxisAngle([0, 0, 1], (79 * Math.PI) / 180));
      const [qw79, qx79, qy79, qz79] = toWxyz(pitch79Q);
      const ez79 = quatRotateVec3(pitch79Q, [0, 0, 1]);
      for (const s of [-0.010, 0.010]) {
        buf.addContact(
          1,
          ez79[0] * s,
          0,
          ez79[2] * s,
          -0.0001,
          0.5 * weightN,
          0,
          1,
          0,
          0,
          0.5 * weightN,
          0,
          true,
          0,
          false
        );
      }
      const act79 = evaluateBlockContactEquilibrium(
        1,
        buf,
        0,
        0.036,
        0,
        qw79,
        qx79,
        qy79,
        qz79,
        0,
        0,
        0,
        0,
        0,
        massKg
      );
      expect(act79.isUnbalanced).toBe(true);
      expect(act79.isKnifeEdgeRelease).toBe(true);

      // 3. Single 8-corner vertex stand
      buf.clear(4);
      const cornerQ = quatMul(
        yawQ,
        quatMul(
          quatFromAxisAngle([0, 0, 1], (52 * Math.PI) / 180),
          quatFromAxisAngle([1, 0, 0], (38 * Math.PI) / 180)
        )
      );
      const [qwc, qxc, qyc, qzc] = toWxyz(cornerQ);
      buf.addContact(2, 0, 0, 0, -0.0001, weightN, 0, 1, 0, 0, weightN, 0, true, 0, false);
      const actCorner = evaluateBlockContactEquilibrium(
        2,
        buf,
        0,
        0.039,
        0,
        qwc,
        qxc,
        qyc,
        qzc,
        0,
        0,
        0,
        0,
        0,
        massKg
      );
      expect(actCorner.isUnbalanced).toBe(true);
      expect(actCorner.isKnifeEdgeRelease).toBe(true);
    }

    // 4. Preserve multi-support 2-block leaning ramp (table bottom edge + upper block edge bracketing CoM)
    buf.clear(4);
    const rampQ = quatFromAxisAngle([0, 0, 1], (26 * Math.PI) / 180);
    const [qwr, qxr, qyr, qzr] = toWxyz(rampQ);
    buf.addContact(0, -0.028, 0.0, -0.011, -0.0001, 0.22, 0, 1, 0, 0, 0.22, 0, true, 0, false);
    buf.addContact(0, -0.028, 0.0, 0.011, -0.0001, 0.22, 0, 1, 0, 0, 0.22, 0, true, 0, false);
    buf.addContact(0, 0.024, 0.015, -0.011, -0.0001, 0.14, -0.44, 0.90, 0, 0, 0.14, 0, false, 5, false);
    buf.addContact(0, 0.024, 0.015, 0.011, -0.0001, 0.14, -0.44, 0.90, 0, 0, 0.14, 0, false, 5, false);
    const actRamp = evaluateBlockContactEquilibrium(
      0,
      buf,
      0,
      0.014,
      0,
      qwr,
      qxr,
      qyr,
      qzr,
      0,
      0,
      0,
      0,
      0,
      massKg
    );
    expect(actRamp.isKnifeEdgeRelease).toBe(false);
    expect(actRamp.isCantileverRelease).toBe(false);

    // 5. Preserve dynamic 6.5-deg Housner fulcrum seesaw (`|w| >= 0.03 rad/s`, nose-up rocking)
    buf.clear(4);
    const seesawQ = quatFromAxisAngle([0, 0, 1], (6.5 * Math.PI) / 180);
    const [qws, qxs, qys, qzs] = toWxyz(seesawQ);
    buf.addContact(1, -0.008, 0.015, -0.011, -0.0001, 0.35, 0, 1, 0, 0, 0.35, 0, false, 3, false);
    buf.addContact(1, -0.008, 0.015, 0.011, -0.0001, 0.35, 0, 1, 0, 0, 0.35, 0, false, 3, false);
    const actSeesaw = evaluateBlockContactEquilibrium(
      1,
      buf,
      0,
      0.023,
      0,
      qws,
      qxs,
      qys,
      qzs,
      0,
      0,
      0,
      0,
      0.25,
      massKg
    );
    expect(actSeesaw.isKnifeEdgeRelease).toBe(false);
    expect(actSeesaw.isCantileverRelease).toBe(false);
  });

  it('Law 1B (Pure): releases unroofed overhangs > 1.0 mm across 360 deg, preserves 68% counterweighted corbels (zero 62% magic cutoff), and releases L_tail <= 6.5 mm slot bevel escapes', () => {
    const buf = new BlockContactScratchBuffer(8);
    const massKg = 0.0711;
    const weightN = massKg * 9.81;

    // 1. Unroofed overhang across 360 deg (d_out = 4.0 mm > 1.0 mm)
    for (const azDeg of [0, 45, 90, 135, 225, 315]) {
      buf.clear(4);
      const azRad = (azDeg * Math.PI) / 180;
      const ux = Math.cos(azRad);
      const uz = Math.sin(azRad);
      const px = -uz;
      const pz = ux;
      const q = quatFromAxisAngle([0, 1, 0], -azRad);
      const [qw, qx, qy, qz] = toWxyz(q);

      for (const lat of [-0.011, 0.011]) {
        buf.addContact(
          0,
          -ux * 0.004 + px * lat,
          0.030,
          -uz * 0.004 + pz * lat,
          -0.0001,
          0.5 * weightN,
          0,
          1,
          0,
          0,
          0.5 * weightN,
          0,
          false,
          4,
          false
        );
      }
      buf.addContact(
        0,
        -ux * 0.020,
        0.030,
        -uz * 0.020,
        -0.0001,
        0.05,
        0,
        1,
        0,
        0,
        0.05,
        0,
        false,
        4,
        false
      );

      const act = evaluateBlockContactEquilibrium(
        0,
        buf,
        0,
        0.0375,
        0,
        qw,
        qx,
        qy,
        qz,
        0,
        0,
        0,
        0,
        0,
        massKg
      );
      expect(act.dOutM).toBeCloseTo(0.004, 3);
      expect(act.isUnbalanced).toBe(true);
      expect(act.isCantileverRelease).toBe(true);
      expect(act.overrideFrictionMu).toBeCloseTo(MU_GLASS_BEVEL_SLIP, 5);

      const gb = auditBlockEquilibriumGateB(0, buf, 0, 0.0375, 0, qw, qx, qy, qz, massKg, weightN);
      expect(gb.pass).toBe(false);
    }

    // 2. Counterweighted corbel at 68% overhang (`51 mm` out of `75 mm` -> `d_out = 13.5 mm`, `L_tail = 24.0 mm > 6.5 mm`!)
    //    with heavy downward roof counterweight at `sAlong = -18 mm` inboard of fulcrum: MUST HOLD (no 62% magic ejection!)
    buf.clear(4);
    const dOutCorbel = 0.0135;
    const roofForceN = 1.40;
    buf.addContact(
      1,
      0.0,
      0.030,
      -0.011,
      -0.0001,
      0.5 * (weightN + roofForceN),
      0,
      1,
      0,
      0,
      0.5 * (weightN + roofForceN),
      0,
      false,
      4,
      false
    );
    buf.addContact(
      1,
      0.0,
      0.030,
      0.011,
      -0.0001,
      0.5 * (weightN + roofForceN),
      0,
      1,
      0,
      0,
      0.5 * (weightN + roofForceN),
      0,
      false,
      4,
      false
    );
    buf.addContact(
      1,
      -0.018,
      0.045,
      -0.008,
      -0.0001,
      0.5 * roofForceN,
      0,
      -1,
      0,
      0,
      -0.5 * roofForceN,
      0,
      false,
      7,
      false
    );
    buf.addContact(
      1,
      -0.018,
      0.045,
      0.008,
      -0.0001,
      0.5 * roofForceN,
      0,
      -1,
      0,
      0,
      -0.5 * roofForceN,
      0,
      false,
      7,
      false
    );

    const actCorbel = evaluateBlockContactEquilibrium(
      1,
      buf,
      dOutCorbel,
      0.0375,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      massKg
    );
    expect(actCorbel.lTailM).toBeGreaterThan(BEVEL_ESCAPE_TAIL_LIMIT_M);
    expect(actCorbel.isUnbalanced).toBe(false);
    expect(actCorbel.isCantileverRelease).toBe(false);
    expect(actCorbel.overrideFrictionMu).toBe(0);

    const gbCorbel = auditBlockEquilibriumGateB(
      1,
      buf,
      dOutCorbel,
      0.0375,
      0,
      1,
      0,
      0,
      0,
      massKg,
      weightN + roofForceN
    );
    expect(gbCorbel.pass).toBe(true);

    // 3. Slot-pinched rounded-bevel lip escape (`L_tail = 5.0 mm <= 6.5 mm`): releases with MU_SLOT_PINCH_ESCAPE (0.018)
    buf.clear(4);
    const dOutLip = EQ_BLOCK_HALF_X_M - 0.0050;
    const pinchRoofN = 8.0;
    buf.addContact(
      2,
      0.0,
      0.030,
      -0.011,
      -0.0001,
      0.5 * (weightN + pinchRoofN),
      0,
      1,
      0,
      0,
      0.5 * (weightN + pinchRoofN),
      0,
      false,
      4,
      false
    );
    buf.addContact(
      2,
      0.0,
      0.030,
      0.011,
      -0.0001,
      0.5 * (weightN + pinchRoofN),
      0,
      1,
      0,
      0,
      0.5 * (weightN + pinchRoofN),
      0,
      false,
      4,
      false
    );
    buf.addContact(
      2,
      -0.004,
      0.045,
      0.0,
      -0.0001,
      pinchRoofN,
      0,
      -1,
      0,
      0,
      -pinchRoofN,
      0,
      false,
      7,
      false
    );

    const actLip = evaluateBlockContactEquilibrium(
      2,
      buf,
      dOutLip,
      0.0375,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      massKg
    );
    expect(actLip.lTailM).toBeCloseTo(0.005, 4);
    expect(actLip.isUnbalanced).toBe(true);
    expect(actLip.isCantileverRelease).toBe(true);
    expect(actLip.overrideFrictionMu).toBeCloseTo(MU_SLOT_PINCH_ESCAPE, 5);
  });

  it('Law 1C (Pure): keeps >= 5x safety margin on centered 36L "+" crosses while crazing & fracturing overloaded 1.5-2.5 mm corner slivers at v = 0', () => {
    const buf = new BlockContactScratchBuffer(4);

    // 1. Centered 36L single-pillar '+' cross (`75 N` across `23.6 x 23.6 mm` ~ `557 mm^2` -> `~0.135 MPa`)
    buf.clear(2);
    const load36LN = 75.0;
    const halfPatch = 0.0118;
    for (const sx of [-halfPatch, halfPatch]) {
      for (const sz of [-halfPatch, halfPatch]) {
        buf.addContact(
          0,
          sx,
          0.0,
          sz,
          -0.0001,
          0.25 * load36LN,
          0,
          1,
          0,
          0,
          0.25 * load36LN,
          0,
          false,
          2,
          false
        );
        buf.addContact(
          0,
          sx,
          0.015,
          sz,
          -0.0001,
          0.25 * load36LN,
          0,
          -1,
          0,
          0,
          -0.25 * load36LN,
          0,
          false,
          5,
          false
        );
      }
    }
    const resCentered36L = evaluateBlockQuasiStaticStress(
      0,
      buf,
      0,
      0.0075,
      0,
      1,
      0,
      0,
      0,
      0,
      10.0
    );
    expect(resCentered36L.overloaded).toBe(false);
    expect(resCentered36L.shouldCraze).toBe(false);
    expect(resCentered36L.shouldFracture).toBe(false);
    expect(resCentered36L.effectiveStressMpa).toBeLessThan(QUASI_STATIC_CRAZE_STRESS_MPA / 5.0);

    // 2. 30L (~60 N) concentrated onto a narrow 2.5 mm x 16 mm corner sliver (40 mm^2 -> 1.50 MPa in [1.10, 3.40) MPa) at v = 0
    buf.clear(2);
    const load30LN = 60.0;
    const cornerX = 0.035;
    const cornerZ = 0.008;
    for (const dx of [-0.00125, 0.00125]) {
      for (const dz of [-0.008, 0.008]) {
        buf.addContact(
          1,
          cornerX + dx,
          0.0,
          cornerZ + dz,
          -0.0002,
          0.25 * load30LN,
          0,
          1,
          0,
          0,
          0.25 * load30LN,
          0,
          false,
          2,
          false
        );
      }
    }
    for (const rdx of [-0.012, 0.012]) {
      for (const rdz of [-0.011, 0.011]) {
        buf.addContact(
          1,
          rdx,
          0.015,
          rdz,
          -0.0001,
          0.25 * load30LN,
          0,
          -1,
          0,
          0,
          -0.25 * load30LN,
          0,
          false,
          5,
          false
        );
      }
    }

    const resCraze = evaluateBlockQuasiStaticStress(1, buf, 0, 0.0075, 0, 1, 0, 0, 0, 0, 1.0);
    expect(resCraze.overloaded).toBe(true);
    expect(resCraze.effectiveStressMpa).toBeGreaterThanOrEqual(QUASI_STATIC_CRAZE_STRESS_MPA);
    expect(resCraze.shouldCraze).toBe(true);
    expect(resCraze.targetDamage).toBeGreaterThan(0.30);
    expect(resCraze.localOriginDmX).toBeGreaterThan(0.25);
    expect(resCraze.localOriginDmZ).toBeGreaterThan(0.05);

    // Rate-limiting guard: immediate re-evaluation within 20 ms (`dtSinceLastCrazeSec = 0.02`) must NOT re-emit craze audio buzz
    const resRateLimited = evaluateBlockQuasiStaticStress(
      1,
      buf,
      0,
      0.0075,
      0,
      1,
      0,
      0,
      0,
      resCraze.targetDamage,
      0.02
    );
    expect(resRateLimited.shouldCraze).toBe(false);

    // Once crazed (`priorDamage = 0.72`), concentrated stress exceeding crazed glass limit fractures the block
    const resFracture = evaluateBlockQuasiStaticStress(1, buf, 0, 0.0075, 0, 1, 0, 0, 0, 0.72, 1.0);
    expect(resFracture.shouldFracture).toBe(true);
  });

  it('15-Axis Gottschalk OBB Support: rejects world-AABB diagonal false positives while detecting true 360-deg OBB & shard support', () => {
    const q45 = quatFromAxisAngle([0, 1, 0], Math.PI / 4);
    const perpDistDm = 0.25 + 0.045; // 29.5 mm center-to-center along 25 mm width axis -> 4.5 mm air gap > 3.5 mm
    const dxDm = perpDistDm * Math.SQRT1_2;
    const dzDm = perpDistDm * Math.SQRT1_2;
    expect(Math.abs(dxDm)).toBeLessThan(0.85);
    expect(
      areBlockObbsWithinGapDm(
        dxDm,
        0.30,
        dzDm,
        q45[0],
        q45[1],
        q45[2],
        q45[3],
        0,
        0.15,
        0,
        q45[0],
        q45[1],
        q45[2],
        q45[3],
        0.035
      )
    ).toBe(false);

    expect(
      areBlockObbsWithinGapDm(
        0,
        0.225,
        0,
        q45[0],
        q45[1],
        q45[2],
        q45[3],
        0,
        0.075,
        0,
        q45[0],
        q45[1],
        q45[2],
        q45[3],
        0.035
      )
    ).toBe(true);

    expect(
      isShardSupportingBlockObbDm(
        0,
        0.30,
        0,
        q45[0],
        q45[1],
        q45[2],
        q45[3],
        0.18,
        0.12,
        0.18,
        0.18,
        0.035
      )
    ).toBe(true);
    expect(
      isShardSupportingBlockObbDm(
        0,
        0.30,
        0,
        q45[0],
        q45[1],
        q45[2],
        q45[3],
        0.32,
        0.12,
        0.32,
        0.12,
        0.035
      )
    ).toBe(false);

    const xs = new Float64Array([-0.012, 0.012, 0.012, -0.012]);
    const zs = new Float64Array([-0.012, -0.012, 0.012, 0.012]);
    const poly = analyzeContactPolygon2D(xs, zs, 4, 0, 0);
    expect(poly.minorWidthM).toBeCloseTo(0.024, 4);
    expect(poly.dOutM).toBeCloseTo(-0.012, 4);
  });

  it('MuJoCo Integration (Law 1A & 1B): releases 360-deg chamfer/corner/saddle balances, restores 0.275 static friction, and preserves >65% counterweighted corbels', () => {
    const mj = getMujocoRuntimeSync().mj;
    const audio = new GlassAudioEngine();
    audio.setMuted(true);
    const phys = new PhysicsWorld(audio, 6, { seed: 4242 });

    // 1. Place blocks on 59-deg longitudinal chamfer, 79-deg upright-end chamfer, and 8-corner across oblique azimuths
    phys.rebuildTower('sandbox', 2, 3, 4242);
    const wCompat = phys.getWorld();
    const m = wCompat.getModel();
    const d = wCompat.getData();

    for (let i = 3; i < phys.blocks.length; i++) {
      const qa = wCompat.getMapping(phys.blocks[i].bodyHandle)!.qposAdr;
      d.qpos[qa + 0] = 0.22 + (i - 3) * 0.03;
      d.qpos[qa + 1] = 0.0075;
      d.qpos[qa + 2] = 0.22;
      d.qpos[qa + 3] = 1;
      d.qpos[qa + 4] = 0;
      d.qpos[qa + 5] = 0;
      d.qpos[qa + 6] = 0;
    }

    const poses: Array<{ xM: number; yM: number; zM: number; q: Quat }> = [
      {
        xM: -0.13,
        yM: 0.0151,
        zM: -0.05,
        q: quatMul(
          quatFromAxisAngle([0, 1, 0], (33 * Math.PI) / 180),
          quatFromAxisAngle([1, 0, 0], (59 * Math.PI) / 180)
        ),
      },
      {
        xM: 0.0,
        yM: 0.0385,
        zM: -0.05,
        q: quatMul(
          quatFromAxisAngle([0, 1, 0], (128 * Math.PI) / 180),
          quatFromAxisAngle([0, 0, 1], (79 * Math.PI) / 180)
        ),
      },
      {
        xM: 0.13,
        yM: 0.041,
        zM: -0.05,
        q: quatMul(
          quatFromAxisAngle([0, 1, 0], (241 * Math.PI) / 180),
          quatMul(
            quatFromAxisAngle([0, 0, 1], (54 * Math.PI) / 180),
            quatFromAxisAngle([1, 0, 0], (36 * Math.PI) / 180)
          )
        ),
      },
    ];

    for (let i = 0; i < 3; i++) {
      const qa = wCompat.getMapping(phys.blocks[i].bodyHandle)!.qposAdr;
      const [qw, qx, qy, qz] = toWxyz(poses[i].q);
      d.qpos[qa + 0] = poses[i].xM;
      d.qpos[qa + 1] = poses[i].yM;
      d.qpos[qa + 2] = poses[i].zM;
      d.qpos[qa + 3] = qw;
      d.qpos[qa + 4] = qx;
      d.qpos[qa + 5] = qy;
      d.qpos[qa + 6] = qz;
      wCompat.wakeBody(phys.blocks[i].bodyHandle);
    }
    mj.mj_forward(m, d);

    for (let f = 0; f < 180; f++) {
      phys.step(1 / 60);
      if (f > 30 && d.ntree_awake === 0) break;
    }

    expect(d.ntree_awake).toBe(0);
    for (let i = 0; i < 3; i++) {
      const qa = wCompat.getMapping(phys.blocks[i].bodyHandle)!.qposAdr;
      const qw = d.qpos[qa + 3];
      const qx = d.qpos[qa + 4];
      const qy = d.qpos[qa + 5];
      const qz = d.qpos[qa + 6];
      const maxAxisY = Math.max(
        Math.abs(2 * (qx * qy + qw * qz)),
        Math.abs(1 - 2 * (qx * qx + qz * qz)),
        Math.abs(2 * (qy * qz - qw * qx))
      );
      expect(maxAxisY).toBeGreaterThan(0.995);
      const gid = phys.blocks[i].colliderHandle;
      expect(m.geom_friction[gid * 3]).toBeGreaterThanOrEqual(0.275 - 1e-6);
    }
    const twoGate1A = phys.auditSettledPlausibilityTwoGate();
    expect(twoGate1A.pass).toBe(true);

    // 2. Counterweighted 66.7% corbel overhang (`50 mm` overhanging cliff -> `d_out = 12.5 mm`, `L_tail = 25 mm > 6.5 mm`)
    //    propped by 3 counterweight blocks on its inboard tail: MUST hold in suspense and pass Two-Gate!
    phys.rebuildTower('sandbox', 2, 2, 5151);
    const w2 = phys.getWorld();
    const m2 = w2.getModel();
    const d2 = w2.getData();
    const setBlkPose = (idx: number, xM: number, yM: number, zM: number, yawDeg: number = 0) => {
      const mapInfo = w2.getMapping(phys.blocks[idx].bodyHandle)!;
      const qa = mapInfo.qposAdr;
      const dof = mapInfo.dofAdr;
      const q = quatFromAxisAngle([0, 1, 0], (yawDeg * Math.PI) / 180);
      const [qw, qx, qy, qz] = toWxyz(q);
      d2.qpos[qa + 0] = xM;
      d2.qpos[qa + 1] = yM;
      d2.qpos[qa + 2] = zM;
      d2.qpos[qa + 3] = qw;
      d2.qpos[qa + 4] = qx;
      d2.qpos[qa + 5] = qy;
      d2.qpos[qa + 6] = qz;
      for (let k = 0; k < 6; k++) d2.qvel[dof + k] = 0;
      w2.wakeBody(phys.blocks[idx].bodyHandle);
    };
    setBlkPose(0, -0.0125, 0.0075, 0.0, 90);
    setBlkPose(1, -0.0385, 0.0075, 0.0, 90);
    setBlkPose(2, 0.0125, 0.0225, 0.0, 0);
    setBlkPose(3, -0.025, 0.0375, 0.0, 90);
    setBlkPose(4, -0.025, 0.0525, 0.0, 90);
    setBlkPose(5, -0.025, 0.0675, 0.0, 90);
    mj.mj_forward(m2, d2);

    for (let f = 0; f < 150; f++) {
      phys.step(1 / 60);
      if (f > 30 && d2.ntree_awake === 0) break;
    }
    const corbelQa = w2.getMapping(phys.blocks[2].bodyHandle)!.qposAdr;
    // Suspenseful corbel tilts nose-down slightly (`y ~ 18.2 mm` vs `7.5 mm` fallen on table) while holding aloft!
    expect(d2.qpos[corbelQa + 1]).toBeGreaterThan(0.016);
    expect(d2.qpos[corbelQa + 0]).toBeGreaterThan(0.010);
    const twoGateCorbel = phys.auditSettledPlausibilityTwoGate();
    expect(twoGateCorbel.pass).toBe(true);

    phys.dispose();
  });

  it('MuJoCo Integration (Law 1C, Polariscope & OrbitCamera): scales cleanly to 36L tall stacks while crazing overloaded corner slivers at v ~ 0', () => {
    const mj = getMujocoRuntimeSync().mj;
    const audio = new GlassAudioEngine();
    audio.setMuted(true);
    const crackSpy = vi.spyOn(audio, 'playGlassCrack');

    // 1. 36L (108-block, 54 cm) tall tower: 0 false crazing/fracture + smooth Polariscope lower-course contrast
    const phys36 = new PhysicsWorld(audio, 36, { seed: 3636 });
    const d36 = phys36.getWorld().getData();
    for (let f = 0; f < 90; f++) {
      phys36.step(1 / 60);
      if (f > 20 && d36.ntree_awake === 0) break;
    }
    expect(phys36.shatteredCount).toBe(0);
    expect(phys36.blocks.every((b) => b.damage === 0)).toBe(true);
    expect(crackSpy).not.toHaveBeenCalled();

    const wTable = new Float32Array(4);
    const wBase = new Float32Array(4);
    const wMid = new Float32Array(4);
    const wTop = new Float32Array(4);
    phys36.writeBlockPhotoelasticWrench(0, wTable, 0);
    phys36.writeBlockPhotoelasticWrench(2 * 3 + 0, wBase, 0);
    phys36.writeBlockPhotoelasticWrench(18 * 3 + 0, wMid, 0);
    phys36.writeBlockPhotoelasticWrench(35 * 3 + 0, wTop, 0);
    expect(Math.max(wTable[0], wTable[2])).toBeGreaterThan(0.85);
    const meanBase = (wBase[0] + wBase[1] + wBase[2]) / 3;
    const meanMid = (wMid[0] + wMid[1] + wMid[2]) / 3;
    const meanTop = (wTop[0] + wTop[1] + wTop[2]) / 3;
    expect(meanBase).toBeGreaterThan(meanMid + 0.12);
    expect(meanMid).toBeLessThan(0.85);
    expect(meanTop).toBeLessThan(0.05);

    const cam = new OrbitCamera();
    cam.syncAutoFramingForTowerHeight(18 * BLOCK_HEIGHT, 18);
    const dist18 = (cam as unknown as { targetDistance: number }).targetDistance;
    expect(dist18).toBeCloseTo(4.55, 3);

    cam.syncAutoFramingForTowerHeight(36 * BLOCK_HEIGHT, 36);
    const dist36 = (cam as unknown as { targetDistance: number }).targetDistance;
    expect(dist36).toBeGreaterThan(7.2);

    cam.zoomBy(-0.20);
    const manualDist = (cam as unknown as { targetDistance: number }).targetDistance;
    cam.syncAutoFramingForTowerHeight(36 * BLOCK_HEIGHT, 36);
    expect((cam as unknown as { targetDistance: number }).targetDistance).toBeCloseTo(manualDist, 5);

    phys36.dispose();

    // 2. Quasi-static concentrated corner-sliver overload at v ~ 0:
    //    Park side blocks of Layers 0 & 1 (`0, 2, 3, 5`) aside so the entire 28-course (~58.6 N) upper stack
    //    channels through Layer 1's center block (`id=4`, X in [-12.5, +12.5] mm) onto a ~1.5 mm edge sliver
    //    of Layer 0's Block 1 (`id=1`, shifted to x = +47.2 mm -> -X flat edge at +10.95 mm vs +11.25 mm!)
    const physOverload = new PhysicsWorld(audio, 30, { seed: 3030 });
    const wO = physOverload.getWorld();
    const mO = wO.getModel();
    const dO = wO.getData();
    for (const [k, sideId] of [0, 2, 3, 5].entries()) {
      const qa = wO.getMapping(physOverload.blocks[sideId].bodyHandle)!.qposAdr;
      dO.qpos[qa + 0] = 0.22 + k * 0.035;
      dO.qpos[qa + 1] = 0.0075;
      dO.qpos[qa + 2] = 0.22;
    }
    const qa1 = wO.getMapping(physOverload.blocks[1].bodyHandle)!.qposAdr;
    dO.qpos[qa1 + 0] = 0.0472;
    dO.qpos[qa1 + 2] = 0.0;
    dO.time = 0.20;
    (physOverload as unknown as { simTime: number }).simTime = 0.20;
    for (let i = 0; i < physOverload.blocks.length; i++) {
      wO.wakeBody(physOverload.blocks[i].bodyHandle);
    }
    mj.mj_forward(mO, dO);

    let sawQuasiStaticCrazeOrFracture = false;
    for (let f = 0; f < 18; f++) {
      physOverload.step(1 / 60);
      if (physOverload.blocks[1].damage > 0.25 || physOverload.blocks[1].shattered) {
        sawQuasiStaticCrazeOrFracture = true;
        break;
      }
    }
    expect(sawQuasiStaticCrazeOrFracture).toBe(true);
    expect(crackSpy.mock.calls.length).toBeGreaterThanOrEqual(1);
    physOverload.dispose();
  });

  it('100-Episode 360-deg Sandbox, 30L+ Tall-Stack & Collapse Plausibility Campaign: 0/100 Gate A + Gate B violations', () => {
    const mj = getMujocoRuntimeSync().mj;
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    let passCount = 0;
    const failures: Array<{
      ep: number;
      category: string;
      awake: number;
      gA: unknown;
      gB: unknown;
    }> = [];

    const physByLayers = new Map<number, PhysicsWorld>();
    const getPhys = (layers: number, initSeed: number): PhysicsWorld => {
      let pw = physByLayers.get(layers);
      if (!pw) {
        pw = new PhysicsWorld(audio, layers, { seed: initSeed });
        physByLayers.set(layers, pw);
      }
      return pw;
    };
    const phys18 = getPhys(18, 1001);
    const phys30 = getPhys(30, 3001);

    try {
      for (let ep = 0; ep < 100; ep++) {
        const rng = mulberry32((0x9e3779b9 ^ (ep * 2654435761)) >>> 0);
        let phys: PhysicsWorld;
        let category: string;
        let maxSettleFrames = 360;

        if (ep < 45) {
          category = 'sandbox_360';
          const layers = 4 + Math.floor(rng() * 6);
          phys = getPhys(layers, (10000 + ep * 977) >>> 0);
          phys.rebuildTower('sandbox', 2, layers, (10000 + ep * 977) >>> 0);
          const w = phys.getWorld();
          const m = w.getModel();
          const d = w.getData();

          if (ep % 3 === 0) {
            const count = Math.min(3, phys.blocks.length);
            for (let k = 0; k < count; k++) {
              const bid = phys.blocks.length - 1 - k;
              const qa = w.getMapping(phys.blocks[bid].bodyHandle)!.qposAdr;
              const az = rng() * Math.PI * 2;
              const r = 0.06 + rng() * 0.16;
              const tiltA = ((15 + rng() * 68) * Math.PI) / 180;
              const tiltB = ((rng() - 0.5) * 50 * Math.PI) / 180;
              const q = quatMul(
                quatFromAxisAngle([0, 1, 0], az),
                quatMul(quatFromAxisAngle([1, 0, 0], tiltA), quatFromAxisAngle([0, 0, 1], tiltB))
              );
              const [qw, qx, qy, qz] = toWxyz(q);
              d.qpos[qa + 0] = Math.cos(az) * r;
              d.qpos[qa + 1] = 0.042 + k * 0.025;
              d.qpos[qa + 2] = Math.sin(az) * r;
              d.qpos[qa + 3] = qw;
              d.qpos[qa + 4] = qx;
              d.qpos[qa + 5] = qy;
              d.qpos[qa + 6] = qz;
              w.wakeBody(phys.blocks[bid].bodyHandle);
            }
            mj.mj_forward(m, d);
          } else if (ep % 3 === 1) {
            const targetIdx = Math.floor(rng() * phys.blocks.length);
            const blk = phys.blocks[targetIdx];
            const localX = (rng() - 0.5) * 2.0 * BLOCK_HALF_X * 0.9;
            const localPt: Vec3 = [localX, 0, 0];
            const grabOff = quatRotateVec3(blk.rotation, localPt);
            const grabStart: Vec3 = [
              blk.position[0] + grabOff[0],
              blk.position[1] + grabOff[1],
              blk.position[2] + grabOff[2],
            ];
            const dragAz = rng() * Math.PI * 2;
            const dragDistDm = 0.35 + rng() * 0.85;
            phys.startGrabBlock(blk.id, localPt, grabStart);
            for (let s = 1; s <= 8; s++) {
              const frac = s / 8;
              phys.updateGrabTarget([
                grabStart[0] + Math.cos(dragAz) * dragDistDm * frac,
                grabStart[1],
                grabStart[2] + Math.sin(dragAz) * dragDistDm * frac,
              ]);
              phys.step(1 / 60);
            }
            phys.endGrab(false);
          } else {
            const topBid = phys.blocks.length - 1;
            const supBid = phys.blocks.length - 4;
            const qaTop = w.getMapping(phys.blocks[topBid].bodyHandle)!.qposAdr;
            const qaSup = w.getMapping(phys.blocks[supBid].bodyHandle)!.qposAdr;
            const az = rng() * Math.PI * 2;
            const overhangM = 0.004 + rng() * 0.032;
            const q = quatFromAxisAngle([0, 1, 0], -az);
            const [qw, qx, qy, qz] = toWxyz(q);
            d.qpos[qaTop + 0] = d.qpos[qaSup + 0] + Math.cos(az) * overhangM;
            d.qpos[qaTop + 1] = d.qpos[qaSup + 1] + BLOCK_HEIGHT * 0.1 + 0.0005;
            d.qpos[qaTop + 2] = d.qpos[qaSup + 2] + Math.sin(az) * overhangM;
            d.qpos[qaTop + 3] = qw;
            d.qpos[qaTop + 4] = qx;
            d.qpos[qaTop + 5] = qy;
            d.qpos[qaTop + 6] = qz;
            for (let i = 0; i < phys.blocks.length; i++) {
              w.wakeBody(phys.blocks[i].bodyHandle);
            }
            mj.mj_forward(m, d);
          }
          maxSettleFrames = 480;
        } else if (ep < 75) {
          category = 'tall_stack_30L';
          phys = phys30;
          phys.rebuildTower('sandbox', 2, 30, (30000 + ep * 1237) >>> 0);
          const w = phys.getWorld();
          const m = w.getModel();
          const d = w.getData();
          if (ep % 2 === 0) {
            const row = 4 + Math.floor(rng() * 20);
            const cand = phys.blocks[row * 3 + 1];
            const axisX = quatRotateVec3(cand.rotation, [1, 0, 0]);
            const sign = rng() < 0.5 ? 1 : -1;
            const yawJitter = (rng() - 0.5) * 0.30;
            const dirX = axisX[0] * Math.cos(yawJitter) - axisX[2] * Math.sin(yawJitter);
            const dirZ = axisX[0] * Math.sin(yawJitter) + axisX[2] * Math.cos(yawJitter);
            const pullDm = 0.22 + rng() * 0.55;
            const startPt: Vec3 = [...cand.position];
            phys.startGrabBlock(cand.id, [0, 0, 0], startPt);
            for (let s = 1; s <= 8; s++) {
              const frac = s / 8;
              phys.updateGrabTarget([
                startPt[0] + sign * dirX * pullDm * frac,
                startPt[1],
                startPt[2] + sign * dirZ * pullDm * frac,
              ]);
              phys.step(1 / 60);
            }
            phys.endGrab(false);
          } else {
            const topBid = phys.blocks.length - 1;
            const qa = w.getMapping(phys.blocks[topBid].bodyHandle)!.qposAdr;
            const offAz = rng() * Math.PI * 2;
            const offM = rng() * 0.014;
            d.qpos[qa + 0] += Math.cos(offAz) * offM;
            d.qpos[qa + 1] += 0.004;
            d.qpos[qa + 2] += Math.sin(offAz) * offM;
            for (let i = Math.max(0, phys.blocks.length - 9); i < phys.blocks.length; i++) {
              w.wakeBody(phys.blocks[i].bodyHandle);
            }
            mj.mj_forward(m, d);
          }
          maxSettleFrames = 660;
        } else {
          category = 'collapse_360';
          phys = phys18;
          const azRad = (((ep - 75) / 25) * Math.PI * 2 + rng() * 0.08) % (Math.PI * 2);
          phys.rebuildTower('sandbox', 2, 18, (75000 + ep * 7919) >>> 0);
          phys.triggerSandboxCollapseImpulse({
            azimuthRad: azRad,
            energyScale: 0.92 + rng() * 0.14,
          });
          maxSettleFrames = 960;
        }

        const dCheck = phys.getWorld().getData();
        const minFrames = category === 'collapse_360' ? 150 : 36;
        for (let f = 0; f < maxSettleFrames; f++) {
          phys.step(1 / 60);
          if (f >= minFrames && dCheck.ntree_awake === 0) {
            phys.step(1 / 60);
            if (dCheck.ntree_awake === 0) break;
          }
        }

        const awake = dCheck.ntree_awake;
        const oracle = phys.auditSettledPlausibilityTwoGate();
        if (awake === 0 && oracle.pass) {
          passCount++;
        } else {
          failures.push({
            ep,
            category,
            awake,
            gA: oracle.gateAViolations,
            gB: oracle.gateBViolations,
          });
        }
      }
    } finally {
      for (const pw of physByLayers.values()) {
        pw.dispose();
      }
    }

    if (failures.length > 0) {
      throw new Error(
        `100-Episode Plausibility Campaign had ${failures.length}/100 violations: ${JSON.stringify(failures)}`
      );
    }
    expect(passCount).toBe(100);
  }, 600000);
});

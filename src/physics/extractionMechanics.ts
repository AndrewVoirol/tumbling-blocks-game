/**
 * extractionMechanics.ts — Competitive Push vs. Force-Bounded Pinch Extraction Mechanics (Track 4)
 *
 * Physical contact-mechanics distinction for Solo, Pass & Play (`AI`), and Online Multiplayer
 * (leaving Sandbox free-drag untouched):
 * - One-Finger End Push (`|localX| >= BLOCK_HALF_X - 0.005`, pushing inward):
 *   Normal compressive fingertip contact up to `F_PUSH_MAX = 18.0 N` without friction slip,
 *   nudging clamped blocks until the far nose pokes out.
 * - Two-Finger Side Pinch / Outward Pull (`|localX| < BLOCK_HALF_X - 0.005` or pulling outward):
 *   Slick soda-lime glass friction contact bounded by `F_SLIP = 2 * MU_FINGER * F_PINCH` (~3.42 N).
 *   When applied pull force / clamping resistance exceeds `F_SLIP`, fingers slip cleanly off
 *   with tactile/audio slip feedback (`slipAudio = true`).
 *
 * Note on units:
 * - Physical slip equations and `BLOCK_HALF_X = 0.0375` use SI meters (matching `Newtons`, `MU_FINGER`).
 * - Vitrail world/local coordinates (`constants.ts`) use decimeters (`1 dm = 0.1 m`), converted via
 *   `localDmToMeters()`.
 */

import { Quat, quatRotateVec3, Vec3 } from '../math/vecMat';
import { SIM_TIME_SCALE } from './constants';

export const BLOCK_HALF_X = 0.0375; // 37.5 mm in SI meters
export const END_FACE_MARGIN_M = 0.005; // 5.0 mm end-face threshold margin
export const END_FACE_THRESHOLD_X = BLOCK_HALF_X - END_FACE_MARGIN_M; // 0.0325 m (32.5 mm)

export const MU_FINGER = 0.38;
export const F_PINCH = 4.5; // N
export const F_SLIP = 2 * MU_FINGER * F_PINCH; // 3.42 N
export const F_PUSH_MAX = 18.0; // N

export const DM_TO_METERS = 0.1;
export const ZONE_BOUNDARY_DM = 0.2175; // Exact 21.75 mm (0.58 * 0.375 dm; avoids IEEE-754 1-ULP 0.21749999999999997)
export const ZONE_NONE = 0;
export const ZONE_BODY = 1.0;
export const ZONE_END_TIP = 2.0;

export const COMPETITIVE_MAX_PULL_FORCE_SOLO_N = 5.2;
export const FINGER_TETHER_STIFFNESS_N_PER_M = 360.0;
export const END_PUSH_CLAMP_EMA_ALPHA = 0.08;
export const END_PUSH_CLAMP_EMA_TAU_UP_SEC = 0.32;
export const END_PUSH_CLAMP_EMA_TAU_DOWN_SEC = 0.06;
export const END_PUSH_CLAMP_MAX_SLEW_UP_N_PER_SEC = 40.0;

/**
 * Time-normalized asymmetric EMA with upward spike slew-rate damping for end-push clamping load.
 * Rejects brief 1–3 frame MuJoCo contact oscillations near the `F_PUSH_MAX / 0.65` (`~27.6923 N`)
 * boundary at both 60 Hz and 120 Hz while allowing sustained overloads (`> 27.6923 N`) to slip
 * within ~150–250 ms and post-spike relaxations to decay rapidly.
 */
export function updateEndPushSmoothedClampingLoadN(
  prevSmoothedN: number,
  liveLoadN: number,
  dt: number = 1 / 60
): number {
  if (!Number.isFinite(liveLoadN)) return prevSmoothedN;
  if (!Number.isFinite(prevSmoothedN) || prevSmoothedN <= 0) return liveLoadN;
  const clampedDt = Math.min(0.085, Math.max(0.001, Number.isFinite(dt) ? dt : 1 / 60));
  if (liveLoadN > prevSmoothedN) {
    const alphaUp = 1 - Math.exp(-clampedDt / END_PUSH_CLAMP_EMA_TAU_UP_SEC);
    const rawNext = prevSmoothedN * (1 - alphaUp) + liveLoadN * alphaUp;
    return Math.min(rawNext, prevSmoothedN + END_PUSH_CLAMP_MAX_SLEW_UP_N_PER_SEC * clampedDt);
  }
  const alphaDown = 1 - Math.exp(-clampedDt / END_PUSH_CLAMP_EMA_TAU_DOWN_SEC);
  return prevSmoothedN * (1 - alphaDown) + liveLoadN * alphaDown;
}

export type LocalHitInput =
  | { x: number; y: number; z: number }
  | readonly [number, number, number];

export interface ExtractionSlipResult {
  slipped: boolean;
  maxForceN: number;
  slipAudio: boolean;
}

export interface CompetitiveDragEvaluation {
  active: boolean;
  isPush: boolean;
  isEndFace: boolean;
  pullForceN: number;
  tetherStretchDm: number;
  slip: ExtractionSlipResult;
}

export interface ClampingEstimateBlockLike {
  id: number;
  position: Vec3;
  assignedLayer: number;
  isLooseSlot: boolean;
  damage: number;
  shattered: boolean;
  extracted: boolean;
}

/**
 * Extracts local X coordinate (in SI meters) from an object `{ x, y, z }` or tuple `[x, y, z]`.
 */
export function extractLocalX(localHit: LocalHitInput): number {
  return Array.isArray(localHit) ? localHit[0] : (localHit as { x: number }).x;
}

/**
 * Converts Vitrail block-local coordinates from decimeters (`dm`) to SI meters (`m`).
 */
export function localDmToMeters(localPointDm: readonly [number, number, number]): [number, number, number] {
  return [
    localPointDm[0] * DM_TO_METERS,
    localPointDm[1] * DM_TO_METERS,
    localPointDm[2] * DM_TO_METERS,
  ];
}

/**
 * Returns true if the local hit point (in SI meters) lies on either end-face zone
 * (`|localX| >= BLOCK_HALF_X - 0.005`, i.e., within 5 mm of +/-37.5 mm end faces).
 */
export function isEndFaceHit(localHit: LocalHitInput): boolean {
  const lx = extractLocalX(localHit);
  return Number.isFinite(lx) && Math.abs(lx) >= END_FACE_THRESHOLD_X;
}

/**
 * Computes the 2-zone grip enum encoded by `WebGPURenderer` (`stateFlags.x` / `stateFlags.y`):
 * - `0`: no hit
 * - `1.0`: Body / Broadside (`|localX_dm| <= 0.2175 dm` = `21.75 mm`)
 * - `2.0`: End-Tip (`|localX_dm| > 0.2175 dm`)
 */
export function computeGripZoneFromLocalDm(
  localPointDm: readonly [number, number, number] | null | undefined
): 0 | 1 | 2 {
  if (!localPointDm) return ZONE_NONE;
  const hx = Math.abs(localPointDm[0]);
  if (!Number.isFinite(hx)) return ZONE_NONE;
  return hx > ZONE_BOUNDARY_DM ? ZONE_END_TIP : ZONE_BODY;
}

/**
 * Evaluates whether fingertip extraction force slips off the glass block.
 *
 * @param localHit Local hit coordinates on the block in SI meters (`BLOCK_HALF_X = 0.0375` m)
 * @param pullForceN Applied extraction/push force magnitude in SI Newtons
 * @param isPush True if the gesture is an inward normal push into the touched end face
 */
export function evaluateExtractionSlip(
  localHit: LocalHitInput,
  pullForceN: number,
  isPush: boolean
): ExtractionSlipResult {
  const forceN = Number.isFinite(pullForceN) ? Math.abs(pullForceN) : 0;
  const endFace = isEndFaceHit(localHit);
  const maxForceN = isPush && endFace ? F_PUSH_MAX : F_SLIP;
  const slipped = forceN > maxForceN;
  return {
    slipped,
    maxForceN,
    slipAudio: slipped,
  };
}

/**
 * Classifies a 3D world drag gesture against a block's grabbed local point as an inward
 * compressive end-face push (`true`) vs. an outward pull or side-face pinch (`false`).
 */
export function isEndPushGesture(
  localPointDm: readonly [number, number, number],
  blockRotation: Quat,
  anchorOrBlockWorldDm: readonly [number, number, number],
  targetWorldDm: readonly [number, number, number]
): boolean {
  if (!isEndFaceHit(localDmToMeters(localPointDm))) {
    return false;
  }
  const axisX = quatRotateVec3(blockRotation, [1, 0, 0]);
  const hLen = Math.hypot(axisX[0], axisX[2]);
  if (hLen < 1e-4) return false;

  const endSign = localPointDm[0] >= 0 ? 1 : -1;
  const outX = (axisX[0] / hLen) * endSign;
  const outZ = (axisX[2] / hLen) * endSign;

  const dx = targetWorldDm[0] - anchorOrBlockWorldDm[0];
  const dz = targetWorldDm[2] - anchorOrBlockWorldDm[2];
  const dragLen = Math.hypot(dx, dz);
  if (dragLen < 1e-4) return false;

  const outDot = dx * outX + dz * outZ;
  // Inward compressive push must point inward opposite the end-face outward normal
  return outDot < 0 && -outDot >= 0.38 * dragLen;
}

/**
 * Estimates per-block normal clamping load (SI Newtons) from photoelastic contact wrench
 * and tributary upper-stack column weight for worker/guest mirrors that do not hold raw `MjData`.
 */
export function estimateClampingLoadFromWrenchAndStack(
  blockId: number,
  blocks: ReadonlyArray<ClampingEstimateBlockLike>,
  blockPhotoelasticWrench: Float32Array
): number {
  const blk = blocks[blockId];
  if (!blk || blk.shattered || blk.extracted) return 0;

  const wBase = blockId * 4;
  let maxTowerLayer = 0;
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    if (!b.shattered && !b.extracted && b.assignedLayer + 1 > maxTowerLayer) {
      maxTowerLayer = b.assignedLayer + 1;
    }
  }
  const photoColNormN =
    7.5 * (maxTowerLayer > 30 ? Math.pow(maxTowerLayer / 30.0, 1.45) : 1.0);
  const invertCompressCol = (s: number): number =>
    maxTowerLayer <= 30 || s <= 0.72
      ? s
      : 0.72 + 0.55 * Math.atanh(Math.min(0.995, (s - 0.72) / 0.28));

  let approxDirectN = 0;
  if (wBase + 2 < blockPhotoelasticWrench.length) {
    const dmgLift = blk.damage * 0.28;
    const cNeg = invertCompressCol(Math.max(0, blockPhotoelasticWrench[wBase + 0] - dmgLift));
    const cCtr = invertCompressCol(Math.max(0, blockPhotoelasticWrench[wBase + 1] - dmgLift));
    const cPos = invertCompressCol(Math.max(0, blockPhotoelasticWrench[wBase + 2] - dmgLift));
    // Inverse of `MujocoPhysicsEngine` photoelastic scaling (`colForce / photoColNormN`, where `colForce = 1.75*min + 0.18*max - 0.38`)
    approxDirectN = ((cNeg + cCtr + cPos) * photoColNormN) / 1.70;
  }

  if (approxDirectN <= 0.25) {
    return approxDirectN;
  }

  let blocksAbove = 0;
  let maxTightDirectN = 0;
  for (let i = 0; i < blocks.length; i++) {
    const ob = blocks[i];
    if (!ob || ob.shattered || ob.extracted) continue;
    if (ob.position[1] > blk.position[1] + 0.0825 && Math.hypot(ob.position[0], ob.position[2]) < 0.85) {
      blocksAbove++;
    } else if (ob.assignedLayer === blk.assignedLayer && !ob.isLooseSlot) {
      const obBase = ob.id * 4;
      const obDmgLift = ob.damage * 0.28;
      const obDirectN =
        obBase + 2 < blockPhotoelasticWrench.length
          ? ((invertCompressCol(Math.max(0, blockPhotoelasticWrench[obBase + 0] - obDmgLift)) +
              invertCompressCol(Math.max(0, blockPhotoelasticWrench[obBase + 1] - obDmgLift)) +
              invertCompressCol(Math.max(0, blockPhotoelasticWrench[obBase + 2] - obDmgLift))) *
              photoColNormN) /
            1.70
          : 0;
      if (obDirectN > maxTightDirectN) maxTightDirectN = obDirectN;
    }
  }

  let bearingInSameRow = 0;
  for (let i = 0; i < blocks.length; i++) {
    const ob = blocks[i];
    if (!ob || ob.shattered || ob.extracted || ob.assignedLayer !== blk.assignedLayer) continue;
    const obBase = ob.id * 4;
    const obDmgLift = ob.damage * 0.28;
    const obDirectN =
      obBase + 2 < blockPhotoelasticWrench.length
        ? ((invertCompressCol(Math.max(0, blockPhotoelasticWrench[obBase + 0] - obDmgLift)) +
            invertCompressCol(Math.max(0, blockPhotoelasticWrench[obBase + 1] - obDmgLift)) +
            invertCompressCol(Math.max(0, blockPhotoelasticWrench[obBase + 2] - obDmgLift))) *
            photoColNormN) /
          1.70
        : 0;
    // Account for colForce's -0.38 per-zone offset when detecting bearing neighbors in the same row
    if (obDirectN > 0.25 || (!ob.isLooseSlot && obDirectN > 0.20)) {
      bearingInSameRow +=
        !blk.isLooseSlot && ob.isLooseSlot && maxTightDirectN > 0 && obDirectN < 0.65 * maxTightDirectN
          ? (obDirectN / maxTightDirectN) * 0.5
          : 1.0;
    }
  }

  const tributaryN =
    bearingInSameRow > 0 ? (blocksAbove * 0.0711 * 9.81) / bearingInSameRow : approxDirectN;
  const blended = approxDirectN * 0.20 + tributaryN * 0.80;
  return bearingInSameRow > 0 ? Math.max(approxDirectN * 0.72, blended) : approxDirectN;
}

/**
 * Returns true if the current competitive drag state represents an inward compressive
 * end-face push gesture (`|localX| >= 32.5 mm` and dragging inward along the block axis).
 */
export function isCompetitiveInwardEndPush(params: {
  gameMode: string;
  turnPhase: string;
  blockRotation: Quat;
  grabLocalPointDm: readonly [number, number, number];
  grabCurrentWorldOnBlockDm: readonly [number, number, number];
  grabAnchorStartDm?: readonly [number, number, number];
  targetWorldDm: readonly [number, number, number];
}): boolean {
  if (params.gameMode === 'sandbox' || params.turnPhase !== 'PULLING') {
    return false;
  }
  const tetherStretchDm = Math.hypot(
    params.targetWorldDm[0] - params.grabCurrentWorldOnBlockDm[0],
    params.targetWorldDm[2] - params.grabCurrentWorldOnBlockDm[2]
  );
  // Use instantaneous tether direction when >= 0.3 mm, otherwise fallback to initial anchor -> target
  const pushRef =
    tetherStretchDm >= 0.003 || !params.grabAnchorStartDm
      ? params.grabCurrentWorldOnBlockDm
      : params.grabAnchorStartDm;
  return isEndPushGesture(
    params.grabLocalPointDm,
    params.blockRotation,
    pushRef,
    params.targetWorldDm
  );
}

/**
 * Evaluates competitive extraction slip during an active drag in Solo, Pass & Play, or Online modes.
 * Sandbox mode always returns `{ active: false, ..., slip: { slipped: false, ... } }`.
 */
export function evaluateCompetitiveDragSlip(params: {
  gameMode: string;
  turnPhase: string;
  blockRotation: Quat;
  grabLocalPointDm: readonly [number, number, number];
  grabCurrentWorldOnBlockDm: readonly [number, number, number];
  grabAnchorStartDm?: readonly [number, number, number];
  targetWorldDm: readonly [number, number, number];
  clampingLoadN: number;
  grabResistance01: number;
}): CompetitiveDragEvaluation {
  if (params.gameMode === 'sandbox' || params.turnPhase !== 'PULLING') {
    return {
      active: false,
      isPush: false,
      isEndFace: false,
      pullForceN: 0,
      tetherStretchDm: 0,
      slip: {
        slipped: false,
        maxForceN: F_PUSH_MAX,
        slipAudio: false,
      },
    };
  }

  const localMeters = localDmToMeters(params.grabLocalPointDm);
  const endFace = isEndFaceHit(localMeters);

  const tetherStretchDm = Math.hypot(
    params.targetWorldDm[0] - params.grabCurrentWorldOnBlockDm[0],
    params.targetWorldDm[2] - params.grabCurrentWorldOnBlockDm[2]
  );

  const isPush = isCompetitiveInwardEndPush(params);

  const measuredForceN =
    Math.max(0, params.grabResistance01) * COMPETITIVE_MAX_PULL_FORCE_SOLO_N * 0.72;
  const stretchForceN = tetherStretchDm * DM_TO_METERS * FINGER_TETHER_STIFFNESS_N_PER_M;

  let pullForceN: number;
  if (isPush && endFace) {
    pullForceN = Math.max(
      measuredForceN,
      Math.min(F_PUSH_MAX, stretchForceN),
      params.clampingLoadN * 0.65
    );
  } else if (params.clampingLoadN > F_SLIP && (tetherStretchDm >= 0.08 || measuredForceN > F_SLIP)) {
    // Pulling outward or side-pinching a clamped block whose bearing load exceeds F_SLIP (3.42 N)
    pullForceN = Math.max(measuredForceN, stretchForceN, params.clampingLoadN);
  } else {
    pullForceN = measuredForceN;
  }

  const slip = evaluateExtractionSlip(localMeters, pullForceN, isPush);
  return {
    active: true,
    isPush,
    isEndFace: endFace,
    pullForceN,
    tetherStretchDm,
    slip,
  };
}

/**
 * Computes effective clamping load (N) for an actively grabbed block, enforcing:
 * - loose-slot negligible-wrench cap (`wSum < 0.18` -> `<= F_SLIP * 0.5`)
 * - monotonic slide relief once a block has slid >= 1.8 cm (`0.18 dm`) out of its initial pocket
 *   or started below `F_SLIP`
 * - EMA-smoothed clamping load during active continuous inward end-push (`isEndPush`) so
 *   transient 1-substep normal-force spikes do not cause false slips while sustained load shifts
 *   still propagate cleanly.
 */
export function computeEffectiveGrabClampingLoadN(params: {
  liveLoadN: number;
  isLooseSlot: boolean;
  wrenchSum: number;
  blockPos: readonly [number, number, number];
  grabStartBlockPos: readonly [number, number, number] | null;
  grabInitialClampingLoadN: number;
  isEndPush?: boolean;
  smoothedLoadN?: number;
}): number {
  if (params.isLooseSlot && params.wrenchSum < 0.18) {
    return Math.min(params.liveLoadN, F_SLIP * 0.5);
  }
  if (params.grabStartBlockPos) {
    const slidDm = Math.hypot(
      params.blockPos[0] - params.grabStartBlockPos[0],
      params.blockPos[2] - params.grabStartBlockPos[2]
    );
    if (slidDm >= 0.18 || params.grabInitialClampingLoadN <= F_SLIP) {
      return Math.min(params.liveLoadN, params.grabInitialClampingLoadN);
    }
    if (
      params.isEndPush &&
      params.grabInitialClampingLoadN > 0 &&
      params.smoothedLoadN !== undefined &&
      Number.isFinite(params.smoothedLoadN) &&
      params.liveLoadN * 0.65 > F_PUSH_MAX
    ) {
      return params.smoothedLoadN;
    }
  }
  return Math.max(params.liveLoadN, params.grabInitialClampingLoadN);
}

/**
 * Counts remaining intact, non-extracted blocks sharing the target block's course within the tower radius.
 */
export function countOtherBlocksInLayer(
  blocks: ReadonlyArray<ClampingEstimateBlockLike>,
  blk: Pick<ClampingEstimateBlockLike, 'id' | 'assignedLayer'>
): number {
  let count = 0;
  for (let i = 0; i < blocks.length; i++) {
    const ob = blocks[i];
    if (
      ob &&
      ob.id !== blk.id &&
      !ob.shattered &&
      !ob.extracted &&
      ob.assignedLayer === blk.assignedLayer &&
      Math.hypot(ob.position[0], ob.position[2]) < 0.72
    ) {
      count++;
    }
  }
  return count;
}

/**
 * Unified competitive slip release predicate across `PhysicsWorld`, `MujocoWorkerBridge`,
 * `GameEngine`, and `MultiplayerGuestWorld`: enforces `F_SLIP = 3.42 N` consistently across
 * 1-block, 2-block, and 3-block rows in competitive modes.
 */
export function shouldReleaseCompetitiveSlip(
  evalRes: Pick<CompetitiveDragEvaluation, 'isEndFace' | 'slip'>,
  _sameRowOtherCount?: number
): boolean {
  return evalRes.slip.slipped;
}

/**
 * Normalizes an `ExtractionSlipResult` so `slipped` and `slipAudio` strictly reflect whether
 * `shouldReleaseCompetitiveSlip` triggers a release.
 */
export function normalizeCompetitiveSlipResult(
  evalRes: Pick<CompetitiveDragEvaluation, 'isEndFace' | 'slip'>,
  _sameRowOtherCount?: number
): ExtractionSlipResult {
  return evalRes.slip;
}

/**
 * Shared Track 4 one-finger compressive end-push impulse helper (`F_PUSH_MAX = 18.0 N`)
 * executed identically by synchronous `PhysicsWorld.step` and Worker-mode `mujocoWorker` `step`.
 */
export function applyAssistiveEndPushImpulse(params: {
  gameMode: string;
  turnPhase: string;
  isPointerDraggingBlock: boolean;
  blockPos: readonly [number, number, number];
  blockRotation: Quat;
  blockLinVel: readonly [number, number, number];
  grabLocalPointDm: readonly [number, number, number];
  grabCurrentWorldOnBlockDm: readonly [number, number, number];
  grabWorldAnchorDm: readonly [number, number, number];
  grabAnchorStartDm?: readonly [number, number, number];
  grabStartBlockPos: readonly [number, number, number] | null;
  clampingLoadN: number;
  grabResistance01: number;
  dt: number;
  applyImpulseAtPoint: (
    impulseSim: { x: number; y: number; z: number },
    pointDm: { x: number; y: number; z: number }
  ) => void;
}): CompetitiveDragEvaluation {
  const evalRes = evaluateCompetitiveDragSlip({
    gameMode: params.gameMode,
    turnPhase: params.turnPhase,
    blockRotation: params.blockRotation,
    grabLocalPointDm: params.grabLocalPointDm,
    grabCurrentWorldOnBlockDm: params.grabCurrentWorldOnBlockDm,
    grabAnchorStartDm: params.grabAnchorStartDm,
    targetWorldDm: params.grabWorldAnchorDm,
    clampingLoadN: params.clampingLoadN,
    grabResistance01: params.grabResistance01,
  });

  if (
    !evalRes.active ||
    !params.isPointerDraggingBlock ||
    !evalRes.isPush ||
    !evalRes.isEndFace ||
    evalRes.slip.slipped ||
    params.clampingLoadN <= F_SLIP ||
    params.clampingLoadN * 0.65 > F_PUSH_MAX ||
    evalRes.tetherStretchDm <= 0.025
  ) {
    return evalRes;
  }

  const endPushSlidDm = params.grabStartBlockPos
    ? Math.hypot(
        params.blockPos[0] - params.grabStartBlockPos[0],
        params.blockPos[2] - params.grabStartBlockPos[2]
      )
    : 0;
  if (endPushSlidDm >= 0.08) {
    return evalRes;
  }

  const axisX = quatRotateVec3(params.blockRotation, [1, 0, 0]);
  const hLen = Math.hypot(axisX[0], axisX[2]);
  if (hLen <= 0.35) {
    return evalRes;
  }

  const endSign = params.grabLocalPointDm[0] >= 0 ? 1 : -1;
  const pushDirX = -(endSign * axisX[0]) / hLen;
  const pushDirZ = -(endSign * axisX[2]) / hLen;
  const invFroude = 1.0 / SIM_TIME_SCALE;
  const vAlongSi =
    (params.blockLinVel[0] * pushDirX + params.blockLinVel[2] * pushDirZ) * invFroude;
  const vTargetSi = Math.min(0.032, evalRes.tetherStretchDm * DM_TO_METERS * 0.35);
  if (vAlongSi < vTargetSi) {
    const stretchForceN =
      evalRes.tetherStretchDm * DM_TO_METERS * FINGER_TETHER_STIFFNESS_N_PER_M;
    const pushForceN = Math.min(
      F_PUSH_MAX,
      Math.max(params.clampingLoadN * 0.65, stretchForceN)
    );
    const clampedDt = Math.min(0.085, Math.max(0.001, params.dt));
    const loadBoostBase = 0.025 + Math.max(0, params.clampingLoadN - 12.0) * 0.012;
    const stepBoostSi = loadBoostBase * Math.max(1.0, clampedDt * 60.0);
    const jSim = Math.min(
      pushForceN * clampedDt * SIM_TIME_SCALE,
      (vTargetSi - vAlongSi + stepBoostSi) * 0.0711 * SIM_TIME_SCALE
    );
    if (jSim > 0) {
      params.applyImpulseAtPoint(
        { x: pushDirX * jSim, y: 0, z: pushDirZ * jSim },
        { x: params.blockPos[0], y: params.blockPos[1], z: params.blockPos[2] }
      );
    }
  }

  return evalRes;
}

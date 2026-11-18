/**
 * solderCameTribology.ts — Proud Perimeter 60/40 Solder Came Tribology & Acoustic Coupling
 *
 * Models the ~0.15 mm proud perimeter 60/40 solder came (2.0 mm border on 75 x 25 x 15 mm blocks).
 * - Smooth kinetic glass sliding on the recessed interior pane (mu_k = 0.125).
 * - 60/40 Sn-Pb solder friction on the proud perimeter border (mu_solder = 0.320).
 * - Energy-passive catch-and-release force pulse on perimeter crossing (delta E <= 0, e <= 0.35).
 * - Computes seamCrossingIntensity in [0.0, 1.0] for GlassAudioEngine slide texture modulation.
 *
 * ZERO-GC: All evaluations write to pre-allocated scratch structs.
 * MEMORYLESS: Operates purely on instantaneous contact kinematics, fully compatible with
 * Glazier AI counterfactual rollouts (evaluateCounterfactualBlockCriticality).
 */

export interface SolderCameContactResult {
  /** True if contact is traversing the came boundary transition band */
  isSeamCrossing: boolean;
  /** Normalized seam crossing intensity in [0.0, 1.0] for audio texture modulation */
  seamCrossingIntensity: number;
  /** Energy-passive tangential resistance force in SI Newtons (opposing relative velocity) */
  tangentialResistForceN: number;
  /** Effective sliding friction coefficient (blended between glass 0.125 and solder 0.320) */
  effectiveFriction: number;
  /** True if the contact point is on the proud perimeter came */
  isOnCame: boolean;
}

export interface SolderCameAggregateResult {
  /** Peak seam crossing intensity across all active block contacts in [0.0, 1.0] */
  maxSeamCrossingIntensity: number;
  /** Total tangential resistance force to apply to sliding block in SI Newtons */
  totalResistForceN: number;
  /** Normal-load-weighted effective friction coefficient */
  weightedEffectiveFriction: number;
  /** Total block-on-block contacts evaluated */
  contactCount: number;
  /** Contacts currently on the proud came border */
  cameBorderContactCount: number;
  /** Internal re-entrant accumulator: sum of normal-load weights */
  _frictionWeightSum?: number;
  /** Internal re-entrant accumulator: sum of normal-load-weighted friction */
  _weightedFrictionSum?: number;
}

// Block dimensions (dm)
export const CAME_BLOCK_HALF_X_DM = 0.375; // 37.5 mm
export const CAME_BLOCK_HALF_Y_DM = 0.075; // 7.5 mm
export const CAME_BLOCK_HALF_Z_DM = 0.125; // 12.5 mm

// Solder came geometry
export const SOLDER_CAME_WIDTH_DM = 0.020;   // 2.0 mm border width
export const SOLDER_CAME_HEIGHT_DM = 0.0015; // 0.15 mm proud ridge height
export const SOLDER_CAME_SLOPE = SOLDER_CAME_HEIGHT_DM / SOLDER_CAME_WIDTH_DM; // tan(theta) = 0.075

// Friction coefficients
export const MU_KINETIC_GLASS = 0.125;   // Smooth polished soda-lime glass
export const MU_KINETIC_SOLDER = 0.320;  // 60/40 Sn-Pb solder alloy
export const SOLDER_CAME_MAX_RESTITUTION = 0.35; // Maximum recovery restitution (e <= 0.35)

// Boundary thresholds (dm)
export const CAME_INNER_BOUND_X_DM = CAME_BLOCK_HALF_X_DM - SOLDER_CAME_WIDTH_DM; // 0.355 dm = 35.5 mm
export const CAME_INNER_BOUND_Z_DM = CAME_BLOCK_HALF_Z_DM - SOLDER_CAME_WIDTH_DM; // 0.105 dm = 10.5 mm
export const CAME_SEAM_TRANSITION_HALF_WIDTH_DM = 0.0075; // +/- 0.75 mm transition band across seam

// Pre-allocated zero-GC scratch singletons
export const scratchSolderCameContactResult: SolderCameContactResult = {
  isSeamCrossing: false,
  seamCrossingIntensity: 0,
  tangentialResistForceN: 0,
  effectiveFriction: MU_KINETIC_GLASS,
  isOnCame: false,
};

export const scratchSolderCameAggregate: SolderCameAggregateResult = {
  maxSeamCrossingIntensity: 0,
  totalResistForceN: 0,
  weightedEffectiveFriction: MU_KINETIC_GLASS,
  contactCount: 0,
  cameBorderContactCount: 0,
  _frictionWeightSum: 0,
  _weightedFrictionSum: 0,
};

export function resetSolderCameAggregate(agg: SolderCameAggregateResult = scratchSolderCameAggregate): void {
  agg.maxSeamCrossingIntensity = 0;
  agg.totalResistForceN = 0;
  agg.weightedEffectiveFriction = MU_KINETIC_GLASS;
  agg.contactCount = 0;
  agg.cameBorderContactCount = 0;
  agg._frictionWeightSum = 0;
  agg._weightedFrictionSum = 0;
}

/**
 * Evaluates solder came tribology for a single contact point on a block.
 *
 * @param localHitDm Contact point in block-local coordinates (decimeters)
 * @param relVelLocalSi Relative sliding velocity in block-local coordinates (SI m/s: v_block - v_other)
 * @param normalForceN Normal contact force in SI Newtons
 * @param out Scratch struct to write results to (zero GC)
 */
export function evaluateSolderCamePoint(
  localHitDm: readonly [number, number, number] | Float64Array,
  relVelLocalSi: readonly [number, number, number] | Float64Array,
  normalForceN: number,
  out: SolderCameContactResult = scratchSolderCameContactResult
): SolderCameContactResult {
  const absY = Math.abs(localHitDm[1]);
  const isFace = absY >= CAME_BLOCK_HALF_Y_DM - 0.015;

  // Compute signed distance past inner came boundary
  const absX = Math.abs(localHitDm[0]);
  const absZ = Math.abs(localHitDm[2]);
  const dx = absX - CAME_INNER_BOUND_X_DM;
  const dz = absZ - CAME_INNER_BOUND_Z_DM;

  const absVx = Math.abs(relVelLocalSi[0]);
  const absVz = Math.abs(relVelLocalSi[2]);
  const hasSlidingMotion = absVx > 1e-4 || absVz > 1e-4;

  let dCame: number;
  let dotD: number;
  if (isFace && hasSlidingMotion) {
    // When sliding along the surface, evaluate seam traversal along the continuous
    // velocity-weighted trajectory. Quadratic weighting (w_x = vx^2 / (vx^2 + vz^2))
    // eliminates knife-edge chatter during diagonal sliding (|vx| ~ |vz|) while strictly
    // preserving orthogonal course decoupling (w_x = 1.0, w_z = 0.0 when vz = 0).
    const vSqX = relVelLocalSi[0] * relVelLocalSi[0];
    const vSqZ = relVelLocalSi[2] * relVelLocalSi[2];
    const vSqTotal = vSqX + vSqZ;
    if (vSqTotal > 1e-12) {
      const wx = vSqX / vSqTotal;
      const wz = 1.0 - wx;
      dCame = wx * dx + wz * dz;
      const dotDx = (localHitDm[0] >= 0 ? 1 : -1) * relVelLocalSi[0];
      const dotDz = (localHitDm[2] >= 0 ? 1 : -1) * relVelLocalSi[2];
      dotD = wx * dotDx + wz * dotDz;
    } else {
      dCame = dx;
      dotD = (localHitDm[0] >= 0 ? 1 : -1) * relVelLocalSi[0];
    }
  } else {
    dCame = isFace ? Math.max(dx, dz) : Math.max(dx, dz, absY - (CAME_BLOCK_HALF_Y_DM - SOLDER_CAME_WIDTH_DM));
    dotD = dx >= dz
      ? (localHitDm[0] >= 0 ? 1 : -1) * relVelLocalSi[0]
      : (localHitDm[2] >= 0 ? 1 : -1) * relVelLocalSi[2];
  }

  const isOnCame = dCame >= 0;
  const absDCame = Math.abs(dCame);
  const isSeamCrossing = absDCame <= CAME_SEAM_TRANSITION_HALF_WIDTH_DM;
  const seamWeight = isSeamCrossing ? 1.0 - absDCame / CAME_SEAM_TRANSITION_HALF_WIDTH_DM : 0.0;

  // Relative sliding speed in local XZ plane (m/s)
  const slideSpeedMps = Math.hypot(relVelLocalSi[0], relVelLocalSi[2]);
  const speedWeight = Math.min(1.0, Math.max(0.0, (slideSpeedMps - 0.002) / 0.038));

  // Continuous C1 climb ratio: smooth transition as dotD crosses 0 (eliminates corner diagonal step jump)
  const climbRatio = slideSpeedMps > 1e-4 ? Math.max(0.0, Math.min(1.0, dotD / slideSpeedMps)) : 0.0;

  const safeNormalN = Number.isFinite(normalForceN) ? Math.max(0.0, normalForceN) : 0.0;

  // Catch phase: opposing tangential resistance force (energy-passive, delta E <= 0)
  let tangentialResistForceN = 0;
  if (climbRatio > 0.001 && isSeamCrossing && safeNormalN > 0.005 && speedWeight > 0.01) {
    tangentialResistForceN = safeNormalN * SOLDER_CAME_SLOPE * seamWeight * speedWeight * climbRatio;
  }

  // Effective friction coefficient blending
  let effectiveFriction = MU_KINETIC_GLASS;
  if (dCame <= -CAME_SEAM_TRANSITION_HALF_WIDTH_DM) {
    effectiveFriction = MU_KINETIC_GLASS;
  } else if (dCame >= CAME_SEAM_TRANSITION_HALF_WIDTH_DM) {
    effectiveFriction = MU_KINETIC_SOLDER;
  } else {
    const t = (dCame + CAME_SEAM_TRANSITION_HALF_WIDTH_DM) / (2.0 * CAME_SEAM_TRANSITION_HALF_WIDTH_DM);
    effectiveFriction = MU_KINETIC_GLASS + (MU_KINETIC_SOLDER - MU_KINETIC_GLASS) * t;
  }

  // Audio seam crossing intensity in [0.0, 1.0] (strictly 0 when normal contact force is absent)
  const loadWeight = Math.min(1.0, safeNormalN / 3.0);
  const seamCrossingIntensity =
    safeNormalN > 0.005
      ? Math.min(1.0, seamWeight * speedWeight * (0.30 + 0.70 * loadWeight))
      : 0.0;

  out.isSeamCrossing = isSeamCrossing;
  out.seamCrossingIntensity = seamCrossingIntensity;
  out.tangentialResistForceN = tangentialResistForceN;
  out.effectiveFriction = effectiveFriction;
  out.isOnCame = isOnCame;
  return out;
}

/**
 * Aggregates a single contact point into the multi-contact aggregate struct.
 */
export function accumulateSolderCameContact(
  localHitDm: readonly [number, number, number] | Float64Array,
  relVelLocalSi: readonly [number, number, number] | Float64Array,
  normalForceN: number,
  agg: SolderCameAggregateResult = scratchSolderCameAggregate
): void {
  const safeNormalN = Number.isFinite(normalForceN) ? Math.max(0.0, normalForceN) : 0.0;
  const pt = evaluateSolderCamePoint(localHitDm, relVelLocalSi, safeNormalN, scratchSolderCameContactResult);
  agg.contactCount++;
  if (pt.isOnCame) agg.cameBorderContactCount++;
  if (pt.seamCrossingIntensity > agg.maxSeamCrossingIntensity) {
    agg.maxSeamCrossingIntensity = pt.seamCrossingIntensity;
  }
  agg.totalResistForceN += pt.tangentialResistForceN;

  const w = Math.max(0.001, safeNormalN);
  const prevW = agg._frictionWeightSum ?? 0;
  const prevWeighted = agg._weightedFrictionSum ?? 0;
  const nextW = prevW + w;
  const nextWeighted = prevWeighted + w * pt.effectiveFriction;
  agg._frictionWeightSum = nextW;
  agg._weightedFrictionSum = nextWeighted;
  agg.weightedEffectiveFriction = nextWeighted / nextW;
}

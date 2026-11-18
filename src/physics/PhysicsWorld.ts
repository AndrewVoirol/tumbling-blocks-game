import { ensureMujocoInitialized } from './mujocoRuntime';
import {
  AdversarialPlacementPlan,
  BlockCriticalityCandidate,
  MujocoPhysicsEngine,
  SceneRaycastHit,
  SceneRaycastOptions,
} from './MujocoPhysicsEngine';
import { INITIAL_LAYERS, TraceryMotif } from './constants';
import { Quat, Vec3 } from '../math/vecMat';
import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import {
  applyAssistiveEndPushImpulse,
  computeEffectiveGrabClampingLoadN,
  countOtherBlocksInLayer,
  estimateClampingLoadFromWrenchAndStack,
  evaluateCompetitiveDragSlip,
  evaluateExtractionSlip,
  type ExtractionSlipResult,
  F_PINCH,
  F_PUSH_MAX,
  F_SLIP,
  isCompetitiveInwardEndPush,
  MU_FINGER,
  normalizeCompetitiveSlipResult,
  shouldReleaseCompetitiveSlip,
  updateEndPushSmoothedClampingLoadN,
} from './extractionMechanics';

export type {
  AdversarialPlacementPlan,
  BlockCriticalityCandidate,
  ExtractionSlipResult,
  SceneRaycastHit,
  SceneRaycastOptions,
};

export {
  evaluateExtractionSlip,
  F_PINCH,
  F_PUSH_MAX,
  F_SLIP,
  MU_FINGER,
};

export {
  type PhysicsFidelityConfig,
  SHIPPED_FIDELITY_CONFIG,
  FAITHFUL_GLASS_FIDELITY_CONFIG,
  cloneFidelityConfig,
} from './physicsFidelityProfile';

export type GameMode = 'solo' | 'party' | 'sandbox';

export type TurnPhase =
  | 'SELECT_BLOCK'   // Waiting for player to tap-test or pull a legal block
  | 'PULLING'        // Actively sliding a block inside the tower
  | 'EXTRACTED'      // Block pulled clear; carry/hover over top slots to stack
  | 'SETTLING'       // Block dropped onto top; waiting for tower wobble to settle
  | 'COLLAPSED';     // Tower collapsed! Physics & shattering continue live

export interface GlassBlockState {
  id: number;
  bodyHandle: number;
  colliderHandle: number;
  swatchId: number;
  motif: TraceryMotif;
  rippleSeed: number;
  initialLayer: number;
  assignedLayer: number;
  assignedSlot: number; // 0, 1, 2
  isLooseSlot: boolean;
  position: Vec3;
  rotation: Quat;
  prePos: Vec3;
  preQuat: Quat;
  linVel: Vec3;
  angVel: Vec3;
  currentFriction: number;
  isMicroDamped: boolean;
  /** Accumulated internal crazing damage in [0, 1) before full shatter */
  damage: number;
  /** Local 3D origin of internal crazing spiderweb cracks */
  crackLocalOrigin: Vec3;
  shattered: boolean;
  extracted: boolean;
  /** Whether this block is currently legal to pull under Jenga rules */
  isLegalTarget: boolean;
}

export interface GlassShardState {
  id: number;
  slotIndex: number;
  bodyHandle: number;
  colliderHandle: number;
  swatchId: number;
  motif: TraceryMotif;
  rippleSeed: number;
  position: Vec3;
  rotation: Quat;
  prePos: Vec3;
  preQuat: Quat;
  preLinVel: Vec3;
  preAngVel: Vec3;
  volume: number;
  massKg: number;
  preVy: number;
  impactLocalPos: Vec3;
  planes: Float32Array;
  planeCount: number;
  /** Fixed-stride offset (`slotIndex * SHARD_SLOT_STRIDE_VERTS`) & vertex count inside GPU shard vertex buffer */
  firstVertex: number;
  vertexCount: number;
  quietSubsteps: number;
  /** Simulation timestamp (`simTime` in seconds) when this Voronoi shard was spawned */
  spawnSimTime?: number;
}

export interface ShardImpactEvent {
  /** Contact / shard position in Vitrail world decimeters (dm) */
  pos: [number, number, number];
  /** Closing normal speed in SI m/s (directly matches GlassAudioEngine ShardChatterImpact.speed) */
  speed: number;
  /** Shard mass in SI kg */
  massKg: number;
  /** Impacted substrate: oak table ('wood') vs intact block / shard ('glass') */
  surface: 'wood' | 'glass';
  /** Active 0..511 shard pool slot index */
  slotIndex: number;
}

export interface WobbleRockEvent {
  blockId: number;
  tiltAngleRad: number;
  angularSpeedRadPerSec: number;
  clampingLoadN: number;
  worldPos: [number, number, number];
  surface: 'glass' | 'wood';
  damage: number;
}

export interface PhysicsStepResult {
  substeps: number;
  simTime: number;
  /** Top <= 8 energetic post-spawn secondary 3D shard bounces (ignoring initial fracture separation window) */
  shardImpacts: ShardImpactEvent[];
  /** Beveled-block Housner rocking / micro-tilt settle impulses */
  wobbleEvents: WobbleRockEvent[];
}

export interface PhysicsShardSnapshotMeta {
  id: number;
  slotIndex: number;
  swatchId: number;
  motif: TraceryMotif;
  rippleSeed: number;
  volume: number;
  massKg: number;
  impactLocalPos: [number, number, number];
  planeCount: number;
  planes: Float32Array;
  vertexCount: number;
  renderVertices?: Float32Array;
}

export interface PhysicsNetworkSnapshot {
  version: 1;
  simTime: number;
  towerSeed: number;
  initialLayers: number;
  gameMode: GameMode;
  turnPhase: TurnPhase;
  currentPlayer: number;
  playerCount: number;
  blocksStacked: number;
  shatteredCount: number;
  collapseReason: string | null;
  activeBlockId: number | null;
  hoveredSlotIndex: number | null;
  isPointerDraggingBlock: boolean;
  grabLocalPoint: [number, number, number];
  grabWorldAnchor: [number, number, number];
  grabCurrentWorldOnBlock: [number, number, number];
  grabResistance01: number;
  carriedTargetPos: [number, number, number];
  carriedTargetQuat: [number, number, number, number];
  carriedLandingPos: [number, number, number];
  carriedYawOffsetRad: number;
  /**
   * Stride = 19 floats per block:
   * [0..2] posDm, [3..6] quat(xyzw), [7..9] linVelSim, [10..12] angVelSim,
   * [13] damage, [14..16] crackLocalOrigin, [17] rippleSeed,
   * [18] packedIntBits (bit0=shattered, bit1=extracted, bit2=isLegalTarget, bit3=isLooseSlot,
   *      bits 4..9=(assignedLayer + 1) & 0x3f, bits 10..11=assignedSlot, bits 12..15=swatchId, bits 16..17=motif)
   */
  blockStride: 19;
  blockCount: number;
  blockData: Float32Array;
  /** 4 floats per block: [-X, 0, +X, bend] Brewster photoelastic stress wrench */
  blockPhotoelasticWrench: Float32Array;
  /** Stride = 15 floats per active shard: [0] id, [1] slotIndex, [2..4] posDm, [5..8] quat(xyzw), [9..11] linVelSim, [12..14] angVelSim */
  shardStride: 15;
  shardCount: number;
  shardData: Float32Array;
  shardMetas?: PhysicsShardSnapshotMeta[];
}

export interface MicroSplinter {
  pos: Vec3;
  vel: Vec3;
  rot: number;
  rotSpeed: number;
  scale: number;
  swatchId: number;
  life: number;
  maxLife: number;
}

export interface TopPlacementSlot {
  layer: number;
  slotIndex: number; // 0, 1, 2
  position: Vec3;
  rotation: Quat;
  yawRad: number;
}

export interface TopSupportFrame {
  targetLayer: number;
  surfaceCenter: Vec3;
  layerCenter: Vec3;
  upNormal: Vec3;
  tiltQuat: Quat;
  defaultYawRad: number;
  defaultRotation: Quat;
  lengthAxis: Vec3;
  spacingAxis: Vec3;
}

export interface TowerTelemetry {
  turnPhase: TurnPhase;
  gameMode: GameMode;
  currentPlayer: number; // 0-indexed
  playerCount: number;
  blocksStacked: number;
  currentLayers: number;
  peakLayers: number;
  liveHeightCm: number;
  peakHeightCm: number;
  leanAngleDeg: number;
  wobbleSpeedDegPerSec: number;
  centerOfMassXZ: [number, number]; // in cm relative to base center
  shatteredCount: number;
  crazedCount: number;
  intactCount: number;
  activeShardCount: number;
  grabResistance01: number; // 0 = loose, 0.5 = snug, 0.9+ = pinned load-bearing
  hoveredBlockId: number | null;
  activeBlockId: number | null;
  hoveredSlotIndex: number | null;
  carriedYawDeg: number;
  lockedMinRow: number | null;
  settleProgress01: number;
  collapseReason: string | null;
  aiOpponentEnabled?: boolean;
  aiThinking?: boolean;
  aiActionLabel?: string | null;
  polariscopeMode?: boolean;
  physicsWorkerActive?: boolean;
}

export interface ShardMeshAppendEvent {
  firstVertex: number;
  vertexData: Float32Array;
}

export type PhysicsBackendKind = 'mujoco';

export interface PhysicsWorldOptions {
  backend?: PhysicsBackendKind;
  seed?: number;
}

export async function ensurePhysicsInitialized(): Promise<void> {
  await ensureMujocoInitialized();
}

/** Backward-compatible alias for test suites calling `ensureRapierInitialized()`. */
export async function ensureRapierInitialized(): Promise<void> {
  await ensureMujocoInitialized();
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export interface PhysicsWorld extends MujocoPhysicsEngine {
  hoverLocalPoint?: Vec3 | null;
  lastExtractionSlip?: ExtractionSlipResult | null;
  evaluateActiveExtractionSlip?: (targetWorldOnPlane?: Vec3) => ExtractionSlipResult | null;
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export class PhysicsWorld {
  constructor(
    audio: GlassAudioEngine,
    initialLayers: number = INITIAL_LAYERS,
    options?: PhysicsWorldOptions
  ) {
    const engine = new MujocoPhysicsEngine(
      audio,
      initialLayers,
      undefined,
      options?.seed
    ) as unknown as PhysicsWorld;

    engine.hoverLocalPoint = null;
    engine.lastExtractionSlip = null;

    let grabStartAnchor: Vec3 | null = null;
    let grabStartBlockPos: Vec3 | null = null;
    let grabInitialClampingLoadN = 0;
    let grabSmoothedClampingLoadN = 0;
    let slipAudioClearTicks = 0;

    const wrenchScratch = new Float32Array(4);
    const getEffectiveGrabClampingLoadN = (
      blkId: number,
      blkPos: Vec3,
      useEngineDirect: boolean = true,
      isEndPush: boolean = false
    ): number => {
      const blk = engine.blocks[blkId];
      const liveLoadN = useEngineDirect
        ? engine.computeBlockClampingLoadN(blkId)
        : estimateClampingLoadFromWrenchAndStack(
            blkId,
            engine.blocks,
            engine.blockPhotoelasticWrench
          );
      let wrenchSum = 0;
      if (blk?.isLooseSlot) {
        wrenchScratch.fill(0);
        engine.writeBlockPhotoelasticWrench(blkId, wrenchScratch, 0);
        wrenchSum = wrenchScratch[0] + wrenchScratch[1] + wrenchScratch[2];
      }
      return computeEffectiveGrabClampingLoadN({
        liveLoadN,
        isLooseSlot: Boolean(blk?.isLooseSlot),
        wrenchSum,
        blockPos: blkPos,
        grabStartBlockPos,
        grabInitialClampingLoadN,
        isEndPush,
        smoothedLoadN: isEndPush ? grabSmoothedClampingLoadN : undefined,
      });
    };

    const evaluateActiveSlip = (targetWorldOnPlane?: Vec3): ExtractionSlipResult | null => {
      if (engine.activeBlockId === null) return null;
      const blk = engine.blocks[engine.activeBlockId];
      if (!blk || blk.shattered) return null;
      const target = targetWorldOnPlane ?? engine.grabWorldAnchor;
      const isEndPush = isCompetitiveInwardEndPush({
        gameMode: engine.gameMode,
        turnPhase: engine.turnPhase,
        blockRotation: blk.rotation,
        grabLocalPointDm: engine.grabLocalPoint,
        grabCurrentWorldOnBlockDm: engine.grabCurrentWorldOnBlock,
        grabAnchorStartDm: grabStartAnchor ?? undefined,
        targetWorldDm: target,
      });
      const clampingLoadN = getEffectiveGrabClampingLoadN(blk.id, blk.position, true, isEndPush);
      const res = evaluateCompetitiveDragSlip({
        gameMode: engine.gameMode,
        turnPhase: engine.turnPhase,
        blockRotation: blk.rotation,
        grabLocalPointDm: engine.grabLocalPoint,
        grabCurrentWorldOnBlockDm: engine.grabCurrentWorldOnBlock,
        grabAnchorStartDm: grabStartAnchor ?? undefined,
        targetWorldDm: target,
        clampingLoadN,
        grabResistance01: engine.grabResistance01,
      });
      const sameRowOtherCount = countOtherBlocksInLayer(engine.blocks, blk);
      const normalized = normalizeCompetitiveSlipResult(res, sameRowOtherCount);
      engine.lastExtractionSlip = normalized;
      return normalized;
    };

    engine.evaluateActiveExtractionSlip = evaluateActiveSlip;

    const origRebuild = engine.rebuildTower.bind(engine);
    engine.rebuildTower = (
      mode?: GameMode,
      players?: number,
      layers?: number,
      seed?: number
    ): void => {
      engine.hoverLocalPoint = null;
      engine.lastExtractionSlip = null;
      grabStartAnchor = null;
      grabStartBlockPos = null;
      grabInitialClampingLoadN = 0;
      grabSmoothedClampingLoadN = 0;
      slipAudioClearTicks = 0;
      origRebuild(mode, players, layers, seed);
    };

    const origStartGrab = engine.startGrabBlock.bind(engine);
    engine.startGrabBlock = (
      blockId: number,
      localPoint: Vec3,
      worldPoint: Vec3,
      timestampMs?: number
    ): boolean => {
      const ok = origStartGrab(blockId, localPoint, worldPoint, timestampMs);
      if (ok) {
        engine.lastExtractionSlip = null;
        grabStartAnchor = [...worldPoint];
        const blk = engine.blocks[blockId];
        grabStartBlockPos = null;
        grabInitialClampingLoadN = 0;
        grabInitialClampingLoadN = blk
          ? getEffectiveGrabClampingLoadN(blockId, blk.position)
          : 0;
        grabSmoothedClampingLoadN = grabInitialClampingLoadN;
        grabStartBlockPos = blk ? [...blk.position] : null;
      }
      return ok;
    };

    const origEndGrab = engine.endGrab.bind(engine);
    engine.endGrab = (dropIfPoisedOverTower: boolean = false, timestampMs?: number): void => {
      grabStartAnchor = null;
      grabStartBlockPos = null;
      grabInitialClampingLoadN = 0;
      grabSmoothedClampingLoadN = 0;
      slipAudioClearTicks = 0;
      origEndGrab(dropIfPoisedOverTower, timestampMs);
    };

    const origUpdateGrabTarget = engine.updateGrabTarget.bind(engine);
    engine.updateGrabTarget = (worldTargetOnPlane: Vec3, timestampMs?: number): void => {
      origUpdateGrabTarget(worldTargetOnPlane, timestampMs);
      if (
        engine.gameMode !== 'sandbox' &&
        engine.turnPhase === 'PULLING' &&
        engine.activeBlockId !== null
      ) {
        const blk = engine.blocks[engine.activeBlockId];
        if (!blk || blk.shattered) return;
        const isEndPush = isCompetitiveInwardEndPush({
          gameMode: engine.gameMode,
          turnPhase: engine.turnPhase,
          blockRotation: blk.rotation,
          grabLocalPointDm: engine.grabLocalPoint,
          grabCurrentWorldOnBlockDm: engine.grabCurrentWorldOnBlock,
          grabAnchorStartDm: grabStartAnchor ?? undefined,
          targetWorldDm: worldTargetOnPlane,
        });
        const clampingLoadN = getEffectiveGrabClampingLoadN(blk.id, blk.position, true, isEndPush);
        const evalRes = evaluateCompetitiveDragSlip({
          gameMode: engine.gameMode,
          turnPhase: engine.turnPhase,
          blockRotation: blk.rotation,
          grabLocalPointDm: engine.grabLocalPoint,
          grabCurrentWorldOnBlockDm: engine.grabCurrentWorldOnBlock,
          grabAnchorStartDm: grabStartAnchor ?? undefined,
          targetWorldDm: worldTargetOnPlane,
          clampingLoadN,
          grabResistance01: engine.grabResistance01,
        });
        const sameRowOtherCount = countOtherBlocksInLayer(engine.blocks, blk);
        const normalizedSlip = normalizeCompetitiveSlipResult(evalRes, sameRowOtherCount);
        engine.lastExtractionSlip = normalizedSlip;

        if (shouldReleaseCompetitiveSlip(evalRes, sameRowOtherCount)) {
          grabStartAnchor = null;
          grabStartBlockPos = null;
          grabInitialClampingLoadN = 0;
          grabSmoothedClampingLoadN = 0;
          origEndGrab(false, timestampMs);
          audio.updateSlideFriction(
            1.25,
            0.35,
            16.0,
            0,
            Math.min(F_SLIP, clampingLoadN),
            0.38
          );
          audio.playUiKeyUp?.('step');
          slipAudioClearTicks = 3;
        }
      }
    };

    const origStep = engine.step.bind(engine);
    engine.step = (dt: number): PhysicsStepResult => {
      if (slipAudioClearTicks > 0 && !engine.isPointerDraggingBlock) {
        slipAudioClearTicks--;
        if (slipAudioClearTicks === 0) {
          audio.updateSlideFriction(0, 0, 0, 0, 0, 0);
        }
      }

      if (
        engine.gameMode !== 'sandbox' &&
        engine.turnPhase === 'PULLING' &&
        engine.isPointerDraggingBlock &&
        engine.activeBlockId !== null
      ) {
        const blk = engine.blocks[engine.activeBlockId];
        if (blk && !blk.shattered && !blk.extracted) {
          const isEndPush = isCompetitiveInwardEndPush({
            gameMode: engine.gameMode,
            turnPhase: engine.turnPhase,
            blockRotation: blk.rotation,
            grabLocalPointDm: engine.grabLocalPoint,
            grabCurrentWorldOnBlockDm: engine.grabCurrentWorldOnBlock,
            grabAnchorStartDm: grabStartAnchor ?? undefined,
            targetWorldDm: engine.grabWorldAnchor,
          });
          if (isEndPush) {
            grabSmoothedClampingLoadN = updateEndPushSmoothedClampingLoadN(
              grabSmoothedClampingLoadN,
              engine.computeBlockClampingLoadN(blk.id),
              dt
            );
          }
          const clampingLoadN = getEffectiveGrabClampingLoadN(blk.id, blk.position, true, isEndPush);
          const rb = engine.getWorld()?.getRigidBody(blk.bodyHandle);
          const evalRes = applyAssistiveEndPushImpulse({
            gameMode: engine.gameMode,
            turnPhase: engine.turnPhase,
            isPointerDraggingBlock: engine.isPointerDraggingBlock,
            blockPos: blk.position,
            blockRotation: blk.rotation,
            blockLinVel: blk.linVel,
            grabLocalPointDm: engine.grabLocalPoint,
            grabCurrentWorldOnBlockDm: engine.grabCurrentWorldOnBlock,
            grabWorldAnchorDm: engine.grabWorldAnchor,
            grabAnchorStartDm: grabStartAnchor ?? undefined,
            grabStartBlockPos,
            clampingLoadN,
            grabResistance01: engine.grabResistance01,
            dt,
            applyImpulseAtPoint: (impulseSim, pointDm) => {
              rb?.applyImpulseAtPoint(impulseSim, pointDm, true);
            },
          });
          const sameRowOtherCount = countOtherBlocksInLayer(engine.blocks, blk);
          engine.lastExtractionSlip = normalizeCompetitiveSlipResult(evalRes, sameRowOtherCount);
        }
      }

      return origStep(dt);
    };

    return engine;
  }
}

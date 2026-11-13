import type { DoubleBuffer, IntBuffer, MainModule, MjData, MjModel } from '@mujoco/mujoco';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  BLOCK_HEIGHT,
  BLOCK_LENGTH,
  BLOCK_WIDTH,
  CRAZE_IMPACT_SPEED,
  GRAVITY_Y,
  INITIAL_LAYERS,
  MAX_ACTIVE_SHARDS,
  SHARD_SLOT_STRIDE_VERTS,
  SHATTER_IMPACT_SPEED,
  SIM_TIME_SCALE,
  TABLE_RADIUS,
  TARGET_BLOCK_MASS_KG,
  TraceryMotif,
} from './constants';
import {
  DM_PER_M,
  FROUDE_VEL_SCALE,
  INV_FROUDE_VEL_SCALE,
  M_PER_DM,
  MUJOCO_CALIBRATED_CONFIG,
  MujocoCalibratedConfig,
} from './mujocoCalibratedParams';
import {
  attachMujocoThreadPoolIfAvailable,
  disposeMujocoHandles,
  getMujocoRuntimeSync,
} from './mujocoRuntime';
import {
  buildMujocoTowerXml,
  buildNominalGlassBlocks,
  buildNominalTopSupportFrameAndSlots,
  MJ_CONAFFINITY_BLOCK_OFF_RIM,
  MJ_CONAFFINITY_BLOCK_ON_TABLE,
  MJ_CONAFFINITY_CULLET_OFF_RIM,
  MJ_CONAFFINITY_CULLET_ON_TABLE,
  MJ_CONAFFINITY_HERO_SHARD_OFF_RIM,
  MJ_CONAFFINITY_HERO_SHARD_ON_TABLE,
  MJ_CONAFFINITY_PARKING,
  MJ_CONAFFINITY_SOLID_CULLET_OFF_RIM,
  MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE,
  MJ_CONTYPE_BLOCK,
  MJ_CONTYPE_CULLET_SHARD,
  MJ_CONTYPE_HERO_SHARD,
  MJ_PARKING_SHELF_TOP_Y_M,
  MujocoTowerIndexMap,
} from './mujocoTowerBuilder';
import {
  MujocoWorldCompat,
  setBodyGeomCollisionMask,
  updateFreeBodyMassAndInertiaO1,
} from './mujocoWorldCompat';
import { fractureBlockVoronoi } from './voronoiFracture';
import {
  Quat,
  quatFromAxisAngle,
  quatInverseRotateVec3,
  quatRotateVec3,
  Vec3,
} from '../math/vecMat';
import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import {
  cloneFidelityConfig,
  FAITHFUL_GLASS_FIDELITY_CONFIG,
  PhysicsFidelityConfig,
} from './physicsFidelityProfile';
import type {
  GameMode,
  GlassBlockState,
  GlassShardState,
  MicroSplinter,
  PhysicsNetworkSnapshot,
  PhysicsShardSnapshotMeta,
  PhysicsStepResult,
  ShardImpactEvent,
  ShardMeshAppendEvent,
  TopPlacementSlot,
  TopSupportFrame,
  TowerTelemetry,
  TurnPhase,
  WobbleRockEvent,
} from './PhysicsWorld';

export interface SceneRaycastOptions {
  includeBlocks?: boolean;
  includeShards?: boolean;
  includeTable?: boolean;
  excludeBodyId?: number;
  maxDistanceDm?: number;
}

export interface SceneRaycastHit {
  kind: 'block' | 'shard' | 'table';
  distanceDm: number;
  worldPointDm: Vec3;
  localPointDm: Vec3;
  normal: Vec3;
  geomId: number;
  bodyId: number;
  blockId?: number;
  shardId?: number;
}

export interface AdversarialPlacementPlan {
  slotIndex: number;
  layer: number;
  targetPosDm: Vec3;
  lateralOffsetDm: number;
  yawOffsetRad: number;
  postPlaceLeanDeg: number;
  selfSurvival01: number;
  opponentDifficulty01: number;
  adversarialScore01: number;
  collapseTriggered: boolean;
}

export interface BlockCriticalityCandidate {
  blockId: number;
  layer: number;
  slot: number;
  clampingLoadN: number;
  predictedMaxLeanDeg: number;
  predictedSlideMm: number;
  collapseTriggered: boolean;
  safetyScore01: number;
  /** CEM-optimized outward horizontal pull azimuth offset (rad, in `[-0.28, +0.28]`) avoiding neighbor snagging */
  optimalPullAzimuthRad?: number;
  /** CEM-optimized world XZ pull unit direction `[dx, dz]` */
  optimalPullDirXZ?: [number, number];
  /** 2nd-ply adversarial top-course placement plan evaluated on post-extraction scratch state */
  adversarialPlacement?: AdversarialPlacementPlan | null;
  /** Combined 2-ply Master Glazier score (`safetyScore01` + safe opponent-trap eccentricity bonus) */
  twoPlyMasterScore01?: number;
}

interface ScratchGeomPairContact {
  g1: number;
  g2: number;
  b1: number;
  b2: number;
  fnSum: number;
  wx: number;
  wy: number;
  wz: number;
  nx: number;
  ny: number;
  nz: number;
}

interface ScratchBlockContactEntry {
  jnSim: number;
  fnN: number;
  localHitDm: Vec3;
  nAtoB: Vec3;
  vCloseReal: number;
  otherBodyId: number;
  otherIsShard: boolean;
  hitTable: boolean;
  worldPtDm: Vec3;
}

interface ScratchBlockWrenchAccum {
  blk: GlassBlockState;
  dynamicImpulseSim: number;
  peakDynamicImpulseSim: number;
  maxVCloseReal: number;
  bestLocalHitDm: Vec3;
  bestNormalAtoB: Vec3;
  bestHitTable: boolean;
  maxRawVCloseReal: number;
  bestRawLocalHitDm: Vec3;
  bestRawNormalAtoB: Vec3;
  bestRawHitTable: boolean;
  bestRawWorldPtDm: Vec3;
  bestRawOtherBodyId: number;
  hasRawSeparatedOnset: boolean;
  upwardSupportFn: number;
  downwardTopFn: number;
  rockHitTable: boolean;
  rockWorldPtDm: Vec3;
  distinctContactBodies: number;
  hasOnsetContact: boolean;
  contacts: ScratchBlockContactEntry[];
}

interface DeferredFractureEntry {
  blk: GlassBlockState;
  localHit: Vec3;
  impactSpeed: number;
  worldNormal: Vec3;
  hitTable: boolean;
  closingSpeedReal: number;
  isAnvilCrush: boolean;
  savedPosDm: Vec3;
  savedQuat: Quat;
  savedLinVelSim: Vec3;
  savedAngVelSim: Vec3;
}

export { buildNominalGlassBlocks, buildNominalTopSupportFrameAndSlots };

const TOP_HOVER_CLEARANCE_DM = 0.040;
const MAX_SPLINTERS = 320;
const liveMujocoEngines: MujocoPhysicsEngine[] = [];
const MAX_CONCURRENT_LIVE_ENGINES = 5;

/**
 * Full-Stack MuJoCo 4.0 Physics Engine for Vitrail (`MujocoPhysicsEngine`)
 *
 * Replaces Rapier 0.14.0's PGS + `[L-1, L+2]` row-freezing hack with:
 * - Google DeepMind `@mujoco/mujoco@4.0.0` (`/mt` multi-threaded WASM island threadpool + `Newton` BoxQP solver)
 * - 24-vertex diamond-beveled glass convex hulls (`multiccd="enable"`, `< 0.05 mm` 18L sag, `17/17` loose-row contrast)
 * - Elliptic friction cones (`condim="4"` / `"6"`, `impratio = 4.5`) with torsional spin & tabletop rolling resistance
 * - Implicit kinematic `mocap` + soft `<connect>` equality hand extraction (`efc_force` tension & slip-power audio)
 * - 6D contact-wrench + three-point flexural Griffith brittle fracture (`efc_force` opposing-pinch & bending stress)
 * - Pre-allocated `512`-slot `sleep="init"` Voronoi shard pool (`0.13 ms` activation, zero `mj_compile` frame stutters)
 * - Real-time per-block internal photoelastic stress telemetry (`stress01`) & counterfactual `mj_copyState` AI oracle
 */
export class MujocoPhysicsEngine {
  public blocks: GlassBlockState[] = [];
  public shards: GlassShardState[] = [];
  public splinters: MicroSplinter[] = [];
  public openTopSlots: TopPlacementSlot[] = [];

  /** Per-block normalized internal compressive + flexural stress in `[0, 1]` for WebGPU photoelastic birefringence */
  public blockPhotoelasticStress01: Float32Array = new Float32Array(0);
  /** Per-block 4-component local contact photoelastic wrench `[negX, center, posX, bending01]` in `[0, 1]^4` */
  public blockPhotoelasticWrench: Float32Array = new Float32Array(0);

  public gameMode: GameMode = 'solo';
  public playerCount: number = 2;
  public currentPlayer: number = 0;
  public turnPhase: TurnPhase = 'SELECT_BLOCK';
  public blocksStacked: number = 0;
  public shatteredCount: number = 0;
  public collapseReason: string | null = null;

  public hoveredBlockId: number | null = null;
  public activeBlockId: number | null = null;
  public grabLocalPoint: Vec3 = [0, 0, 0];
  public grabWorldAnchor: Vec3 = [0, 0, 0];
  public grabCurrentWorldOnBlock: Vec3 = [0, 0, 0];
  public grabPlaneY: number = 0;
  public grabResistance01: number = 0;
  public isPointerDraggingBlock: boolean = false;
  public carriedFromDrag: boolean = false;

  public listenerCamPos: Vec3 = [0, 2.2, 5.2];
  public listenerCamRight: Vec3 = [1, 0, 0];

  public hoveredSlotIndex: number | null = null;
  public topSupportFrame: TopSupportFrame = {
    targetLayer: INITIAL_LAYERS,
    surfaceCenter: [0, INITIAL_LAYERS * BLOCK_HEIGHT, 0],
    layerCenter: [0, INITIAL_LAYERS * BLOCK_HEIGHT + BLOCK_HALF_Y + 0.002, 0],
    upNormal: [0, 1, 0],
    tiltQuat: [0, 0, 0, 1],
    defaultYawRad: 0,
    defaultRotation: [0, 0, 0, 1],
    lengthAxis: [1, 0, 0],
    spacingAxis: [0, 0, 1],
  };

  public carriedLandingPos: Vec3 = [0, INITIAL_LAYERS * BLOCK_HEIGHT + BLOCK_HALF_Y + 0.002, 0];
  public carriedTargetPos: Vec3 = [0, 3.2, 0];
  public carriedTargetQuat: Quat = [0, 0, 0, 1];
  public carriedYawOffsetRad: number = 0;

  public leanAngleDeg: number = 0;
  private solverLeanAngleDeg: number = 0;
  public wobbleSpeedDegPerSec: number = 0;
  public centerOfMassXZCm: [number, number] = [0, 0];
  public currentLayers: number = INITIAL_LAYERS;
  public peakLayers: number = INITIAL_LAYERS;
  public liveHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;
  public peakHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;
  public lockedMinRow: number | null = INITIAL_LAYERS;
  public lastFractureOriginLocal: Vec3 | null = null;
  public towerSeed: number = 0;
  public lastStepResult: PhysicsStepResult = {
    substeps: 0,
    simTime: 0,
    shardImpacts: [],
    wobbleEvents: [],
  };

  private mj!: MainModule;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private scratchOracleData: MjData | null = null;
  private rayGeomIdBuf: IntBuffer | null = null;
  private rayNormalBuf: DoubleBuffer | null = null;
  private islandWakeMask: Uint8Array = new Uint8Array(640);
  private compiledTowerLayers: number = -1;
  private modelResetSnapshot: {
    geom_type: Int32Array;
    geom_dataid: Int32Array;
    geom_contype: Int32Array;
    geom_conaffinity: Int32Array;
    geom_group: Int32Array;
    geom_pos: Float64Array;
    geom_quat: Float64Array;
    geom_size: Float64Array;
    geom_rbound: Float64Array;
    geom_aabb: Float64Array;
    geom_friction: Float64Array;
    geom_priority: Int32Array;
    body_contype: Int32Array;
    body_conaffinity: Int32Array;
    body_mass: Float64Array;
    body_subtreemass: Float64Array;
    body_inertia: Float64Array;
    body_iquat: Float64Array;
    body_invweight0: Float64Array;
    dof_M0: Float64Array;
    dof_invweight0: Float64Array;
    dof_damping: Float64Array;
    eq_obj1id: Int32Array;
    eq_obj2id: Int32Array;
    eq_data: Float64Array;
    warmupQpos: Float64Array;
    warmupQvel: Float64Array;
    warmupQaccWarmstart: Float64Array;
    warmupTreeAsleep: Int32Array;
    warmupNtreeAwake: number;
    warmupClampingLoadN: Float64Array;
    warmupPhotoelastic01: Float32Array;
    warmupPhotoelasticWrench: Float32Array;
    warmupPairKeys: number[];
  } | null = null;
  private activeSlidingGeomId: number | null = null;
  private map!: MujocoTowerIndexMap;
  private worldCompat!: MujocoWorldCompat;
  private cfg: MujocoCalibratedConfig;
  private audio: GlassAudioEngine;
  private fidelity: PhysicsFidelityConfig = cloneFidelityConfig(FAITHFUL_GLASS_FIDELITY_CONFIG);
  public activeTimestepHz: number = 600;
  public activeSolverIterations: number = 26;
  public activeLsIterations: number = 12;
  public lastCullShardStats: { total: number; hero: number; cullet: number } = {
    total: 0,
    hero: 0,
    cullet: 0,
  };
  public phantomAirClinkCount: number = 0;
  private trackedDropBlockId: number = -1;
  private trackedDropRestYDm: number = 0;
  private trackedDropSawDescent: boolean = false;
  private trackedDropMinClearanceMm: number = Number.POSITIVE_INFINITY;
  private trackedDropMaxReboundMm: number = 0;
  private trackedDropTraceMm: number[] = [];
  private unloadedBounceLastSimTime: Float64Array = new Float64Array(128).fill(-10);
  private unloadedBounceHopCount: Uint8Array = new Uint8Array(128);
  private unloadedBouncePeakVClose: Float64Array = new Float64Array(128);
  private blockPreSubstepLinSpeedSi: Float64Array = new Float64Array(128);
  private blockPreSubstepAngSpeedSi: Float64Array = new Float64Array(128);
  private towerRebuildSimTime: number = 0;
  private multiccdReentrySubstepsRemaining: number = 0;
  private lastGlobalWoodThudSimTime: number = -10.0;
  private blockPairLastImpactAudioSimTime = new Map<number, number>();

  private bodyToBlock = new Map<number, GlassBlockState>();
  private bodyToShard = new Map<number, GlassShardState>();
  private cachedClampingLoadN: Float64Array = new Float64Array(0);
  private blockLastCrazeTime = new Map<number, number>();
  private blockLastImpactAudioTime = new Map<number, number>();
  private blockLastWobbleTime = new Map<number, number>();
  private blockPairLastContactSimTime = new Map<number, number>();
  private blockRockTransferCount: Uint8Array = new Uint8Array(128);
  private blockLastRockTime: Float64Array = new Float64Array(128);
  private blockCoherentArcSubsteps: Uint16Array = new Uint16Array(128);
  private blockPrevHorizOmegaX: Float64Array = new Float64Array(128);
  private blockPrevHorizOmegaZ: Float64Array = new Float64Array(128);
  private awakeParkedBodies: Array<{ bodyId: number; qposAdr: number; dofAdr: number }> = [];
  private freeShardSlots: number[] = [];
  private shardQuietSubsteps = new Uint16Array(MAX_ACTIVE_SHARDS);
  private shardNoSupportSubsteps = new Uint8Array(MAX_ACTIVE_SHARDS);
  private shardAirborneSubsteps = new Uint16Array(MAX_ACTIVE_SHARDS);
  private shardHadContactSubstep = new Uint8Array(MAX_ACTIVE_SHARDS);
  private shardInContactPrevSubstep = new Uint8Array(MAX_ACTIVE_SHARDS);
  private shardPreSubstepTotalEnergyJ = new Float64Array(MAX_ACTIVE_SHARDS);
  private shardEntryTotalEnergyJ = new Float64Array(MAX_ACTIVE_SHARDS);
  private shardLastImpactSimTime = new Float64Array(MAX_ACTIVE_SHARDS);
  private shardLastAudioSimTime = new Float64Array(MAX_ACTIVE_SHARDS);
  private standingTiltsScratch = new Float32Array(128);
  private courseUprightScratch = new Uint8Array(64);
  private slotRenderVertices: Array<Float32Array | null> = new Array(MAX_ACTIVE_SHARDS).fill(null);
  private exportedShardIds = new Set<number>();
  private archGeomPos: Float64Array = new Float64Array(0);
  private archGeomQuat: Float64Array = new Float64Array(0);
  private pendingSlotUploads = new Map<number, Float32Array>();
  private shardBufferResetPending: boolean = true;
  private nextShardId: number = 1;
  private minShardGeomId: number = 0x7fffffff;
  private maxShardGeomId: number = -1;
  private sandboxCollapseActive: boolean = false;
  private deferredFractures: DeferredFractureEntry[] = [];
  private pendingShardImpacts: ShardImpactEvent[] = [];
  private pendingWobbleEvents: WobbleRockEvent[] = [];
  private stepShardCandidates: ShardImpactEvent[] = [];
  private stepWobbleCandidates: WobbleRockEvent[] = [];
  private scratchPairMap = new Map<number, ScratchGeomPairContact>();
  private scratchPairPool: ScratchGeomPairContact[] = [];
  private scratchBlockImpacts = new Map<number, ScratchBlockWrenchAccum>();

  private photoCap: number = 0;
  private photoHasVert: Uint8Array = new Uint8Array(0);
  private photoHasAny: Uint8Array = new Uint8Array(0);
  private photoTopNeg: Float64Array = new Float64Array(0);
  private photoTopCtr: Float64Array = new Float64Array(0);
  private photoTopPos: Float64Array = new Float64Array(0);
  private photoBotNeg: Float64Array = new Float64Array(0);
  private photoBotCtr: Float64Array = new Float64Array(0);
  private photoBotPos: Float64Array = new Float64Array(0);
  private photoMidShear: Float64Array = new Float64Array(0);
  private photoTopF: Float64Array = new Float64Array(0);
  private photoTopAbsX: Float64Array = new Float64Array(0);
  private photoTopSigX: Float64Array = new Float64Array(0);
  private photoBotF: Float64Array = new Float64Array(0);
  private photoBotAbsX: Float64Array = new Float64Array(0);
  private photoBotSigX: Float64Array = new Float64Array(0);

  private carriedRawLocalVL: [number, number] = [0, 0];
  private hasCustomCarriedHover: boolean = false;
  private lastHoverTargetXZ: [number, number] | null = null;
  private lastHoverTimeSec: number = 0;
  private carriedHandVelXZ: [number, number] = [0, 0];
  private extractionStartPos: Vec3 = [0, 0, 0];
  private extractionOutDir: Vec3 = [1, 0, 0];
  private extractionArcTimer: number = 0;
  private clearContactSubsteps: number = 0;
  private grabRampSec: number = 0;
  private settleTimer: number = 0;
  private readonly SETTLE_DURATION: number = 1.05;

  private accumulatorSec: number = 0;
  private simTime: number = 0;
  private frameShatterCount: number = 0;
  private isDisposed: boolean = false;

  private extraPreStep = new Map<
    number,
    { posDm: Vec3; linVelSim: Vec3; angVelSim: Vec3; massKg: number }
  >();

  constructor(
    audio: GlassAudioEngine,
    initialLayers: number = INITIAL_LAYERS,
    cfg: MujocoCalibratedConfig = MUJOCO_CALIBRATED_CONFIG,
    seed?: number
  ) {
    // Reap leaked un-disposed WASM engine instances (e.g. across test cases) before allocating a new MjModel/MjData
    for (let i = liveMujocoEngines.length - 1; i >= 0; i--) {
      if (liveMujocoEngines[i].isDisposed) liveMujocoEngines.splice(i, 1);
    }
    while (liveMujocoEngines.length >= MAX_CONCURRENT_LIVE_ENGINES) {
      const oldest = liveMujocoEngines.shift();
      oldest?.dispose();
    }
    liveMujocoEngines.push(this);

    this.audio = audio;
    this.cfg = cfg;
    const rt = getMujocoRuntimeSync();
    this.mj = rt.mj;
    this.rebuildTower(undefined, undefined, initialLayers, seed);
  }

  public setListenerPose(camPos: Vec3, camRight: Vec3): void {
    this.listenerCamPos[0] = camPos[0];
    this.listenerCamPos[1] = camPos[1];
    this.listenerCamPos[2] = camPos[2];
    this.listenerCamRight[0] = camRight[0];
    this.listenerCamRight[1] = camRight[1];
    this.listenerCamRight[2] = camRight[2];
  }

  public getWorld(): MujocoWorldCompat {
    return this.worldCompat;
  }

  public getClearContactSubsteps(): number {
    return this.clearContactSubsteps;
  }

  public getSimTime(): number {
    return this.simTime;
  }

  public setFidelityConfig(partial: Partial<PhysicsFidelityConfig>): void {
    this.fidelity = {
      ...this.fidelity,
      ...partial,
    };
  }

  public getFidelityConfig(): PhysicsFidelityConfig {
    return { ...this.fidelity };
  }

  public getSolverDiagnostics(): {
    timestepHz: number;
    solverIterations: number;
    lsIterations: number;
    lastCullShardStats: { total: number; hero: number; cullet: number };
    phantomAirClinkCount: number;
  } {
    return {
      timestepHz: this.activeTimestepHz,
      solverIterations: this.activeSolverIterations,
      lsIterations: this.activeLsIterations,
      lastCullShardStats: { ...this.lastCullShardStats },
      phantomAirClinkCount: this.phantomAirClinkCount,
    };
  }

  public trackBlockDrop(blockId: number, restYDm: number): void {
    this.trackedDropBlockId = blockId;
    this.trackedDropRestYDm = restYDm;
    this.trackedDropSawDescent = false;
    this.trackedDropMinClearanceMm = Number.POSITIVE_INFINITY;
    this.trackedDropMaxReboundMm = 0;
    this.trackedDropTraceMm = [];
  }

  public getTrackedDropState(): {
    maxReboundMm: number;
    sawDescent: boolean;
    traceMm: number[];
  } {
    return {
      maxReboundMm: this.trackedDropMaxReboundMm,
      sawDescent: this.trackedDropSawDescent,
      traceMm: [...this.trackedDropTraceMm],
    };
  }

  public getRenderPoseAlpha(_nowMs?: number): number {
    return 1.0;
  }

  public getInitialLayers(): number {
    return this.compiledTowerLayers > 0 ? this.compiledTowerLayers : INITIAL_LAYERS;
  }

  public consumeShardImpacts(): ShardImpactEvent[] {
    if (this.pendingShardImpacts.length === 0) return [];
    const out = this.pendingShardImpacts;
    this.pendingShardImpacts = [];
    return out;
  }

  public consumeWobbleEvents(): WobbleRockEvent[] {
    if (this.pendingWobbleEvents.length === 0) return [];
    const out = this.pendingWobbleEvents;
    this.pendingWobbleEvents = [];
    return out;
  }

  public consumePendingShardUploads(): ShardMeshAppendEvent[] {
    if (this.pendingSlotUploads.size === 0) return [];
    const out: ShardMeshAppendEvent[] = [];
    for (const [slotIndex, vertexData] of this.pendingSlotUploads.entries()) {
      out.push({
        firstVertex: slotIndex * SHARD_SLOT_STRIDE_VERTS,
        vertexData,
      });
    }
    this.pendingSlotUploads.clear();
    return out;
  }

  public wasShardBufferReset(): boolean {
    if (this.shardBufferResetPending) {
      this.shardBufferResetPending = false;
      return true;
    }
    return false;
  }

  public resetTower(layers?: number, seed?: number): void {
    const targetLayers =
      layers !== undefined
        ? layers
        : this.compiledTowerLayers > 0
          ? this.compiledTowerLayers
          : INITIAL_LAYERS;
    this.rebuildTower(this.gameMode, this.playerCount, targetLayers, seed);
  }

  public rebuildTower(
    mode?: GameMode,
    players?: number,
    layers: number = INITIAL_LAYERS,
    seed?: number
  ): void {
    if (mode) this.gameMode = mode;
    if (players !== undefined) this.playerCount = Math.max(2, Math.min(4, players));
    this.towerSeed = seed !== undefined ? seed >>> 0 : 0;
    const towerLayers = Math.max(1, Math.min(32, Math.round(layers)));
    this.audio.updateSlideFriction(0, 0, 0, 0);
    this.unloadedBounceLastSimTime.fill(-10);
    this.unloadedBounceHopCount.fill(0);
    this.multiccdReentrySubstepsRemaining = 0;
    this.blockPairLastImpactAudioSimTime.clear();
    this.lastCullShardStats = { total: 0, hero: 0, cullet: 0 };
    this.phantomAirClinkCount = 0;
    this.trackedDropBlockId = -1;
    this.trackedDropSawDescent = false;
    this.trackedDropMinClearanceMm = Number.POSITIVE_INFINITY;
    this.trackedDropMaxReboundMm = 0;
    this.trackedDropTraceMm = [];

    const canReuseCompiledModel =
      this.model !== null &&
      this.data !== null &&
      this.compiledTowerLayers === towerLayers &&
      this.modelResetSnapshot !== null;

    if (canReuseCompiledModel && this.model && this.data && this.modelResetSnapshot) {
      const snap = this.modelResetSnapshot;
      this.model.geom_type.set(snap.geom_type);
      this.model.geom_dataid.set(snap.geom_dataid);
      this.model.geom_contype.set(snap.geom_contype);
      this.model.geom_conaffinity.set(snap.geom_conaffinity);
      this.model.geom_group.set(snap.geom_group);
      this.model.geom_pos.set(snap.geom_pos);
      this.model.geom_quat.set(snap.geom_quat);
      this.model.geom_size.set(snap.geom_size);
      this.model.geom_rbound.set(snap.geom_rbound);
      this.model.geom_aabb.set(snap.geom_aabb);
      this.model.geom_friction.set(snap.geom_friction);
      this.model.geom_priority.set(snap.geom_priority);
      this.model.opt.timestep = this.cfg.timestepSec;
      this.model.opt.iterations = this.cfg.solverIterations;
      this.model.opt.ls_iterations = this.cfg.lsIterations;
      this.model.opt.disableflags &= ~this.mj.mjtDisableBit.mjDSBL_MULTICCD.value;
      this.model.body_contype.set(snap.body_contype);
      this.model.body_conaffinity.set(snap.body_conaffinity);
      this.model.body_mass.set(snap.body_mass);
      this.model.body_subtreemass.set(snap.body_subtreemass);
      this.model.body_inertia.set(snap.body_inertia);
      this.model.body_iquat.set(snap.body_iquat);
      this.model.body_invweight0.set(snap.body_invweight0);
      this.model.dof_M0.set(snap.dof_M0);
      this.model.dof_invweight0.set(snap.dof_invweight0);
      this.model.dof_damping.set(snap.dof_damping);
      this.model.eq_obj1id.set(snap.eq_obj1id);
      this.model.eq_obj2id.set(snap.eq_obj2id);
      this.model.eq_data.set(snap.eq_data);
      this.mj.mj_resetData(this.model, this.data);
      attachMujocoThreadPoolIfAvailable(this.data);
      this.data.qpos.set(snap.warmupQpos);
      this.data.qvel.set(snap.warmupQvel);
      this.data.qacc_warmstart.set(snap.warmupQaccWarmstart);
      this.mj.mj_forward(this.model, this.data);
    } else {
      if (this.scratchOracleData) {
        try {
          this.scratchOracleData.delete();
        } catch {
          // Ignore
        }
        this.scratchOracleData = null;
      }
      disposeMujocoHandles(this.data, this.model);
      this.data = null;
      this.model = null;
      this.modelResetSnapshot = null;

      const built = buildMujocoTowerXml(towerLayers, this.cfg);
      this.map = built.map;
      this.model = this.mj.from_xml_string(built.xml);
      this.data = new this.mj.MjData(this.model);
      this.compiledTowerLayers = towerLayers;
      attachMujocoThreadPoolIfAvailable(this.data);
      this.mj.mj_forward(this.model, this.data);

      // Snapshot compiled mesh geom_pos & geom_quat for each of the 16 canonical Voronoi hull archetypes
      const archCount = this.cfg.voronoiHullArchetypeCount;
      this.archGeomPos = new Float64Array(archCount * 3);
      this.archGeomQuat = new Float64Array(archCount * 4);
      for (let a = 0; a < archCount; a++) {
        const sampleGid = built.map.shardGeomIds[a];
        this.archGeomPos[a * 3 + 0] = this.model.geom_pos[sampleGid * 3 + 0];
        this.archGeomPos[a * 3 + 1] = this.model.geom_pos[sampleGid * 3 + 1];
        this.archGeomPos[a * 3 + 2] = this.model.geom_pos[sampleGid * 3 + 2];
        this.archGeomQuat[a * 4 + 0] = this.model.geom_quat[sampleGid * 4 + 0];
        this.archGeomQuat[a * 4 + 1] = this.model.geom_quat[sampleGid * 4 + 1];
        this.archGeomQuat[a * 4 + 2] = this.model.geom_quat[sampleGid * 4 + 2];
        this.archGeomQuat[a * 4 + 3] = this.model.geom_quat[sampleGid * 4 + 3];
      }
    }

    this.minShardGeomId =
      this.map.shardGeomIds.length > 0 ? this.map.shardGeomIds[0] : 0x7fffffff;
    this.maxShardGeomId =
      this.map.shardGeomIds.length > 0
        ? this.map.shardGeomIds[this.map.shardGeomIds.length - 1]
        : -1;

    this.worldCompat = new MujocoWorldCompat(this.mj, this.model, this.data, this.map);
    this.shardQuietSubsteps.fill(0);
    this.shardNoSupportSubsteps.fill(0);
    this.shardAirborneSubsteps.fill(0);
    this.shardHadContactSubstep.fill(0);
    this.shardInContactPrevSubstep.fill(0);
    this.shardPreSubstepTotalEnergyJ.fill(0);
    this.shardEntryTotalEnergyJ.fill(0);
    this.shardLastImpactSimTime.fill(-10);
    this.shardLastAudioSimTime.fill(-10);
    this.slotRenderVertices.fill(null);
    this.exportedShardIds.clear();
    this.sandboxCollapseActive = false;
    this.deferredFractures.length = 0;
    this.pendingShardImpacts = [];
    this.pendingWobbleEvents = [];
    this.stepShardCandidates.length = 0;
    this.stepWobbleCandidates.length = 0;
    this.frameShatterCount = 0;
    this.lastStepResult = {
      substeps: 0,
      simTime: 0,
      shardImpacts: [],
      wobbleEvents: [],
    };

    this.blocks = [];
    this.shards = [];
    this.splinters = [];
    this.bodyToBlock.clear();
    this.bodyToShard.clear();
    this.freeShardSlots = [];
    for (let slot = MAX_ACTIVE_SHARDS - 1; slot >= 0; slot--) {
      this.freeShardSlots.push(slot);
    }
    this.pendingSlotUploads.clear();
    this.shardBufferResetPending = true;
    this.currentPlayer = 0;
    this.turnPhase = 'SELECT_BLOCK';
    this.blocksStacked = 0;
    this.shatteredCount = 0;
    this.collapseReason = null;
    this.hoveredBlockId = null;
    this.activeBlockId = null;
    this.hoveredSlotIndex = null;
    this.carriedYawOffsetRad = 0;
    this.carriedRawLocalVL = [0, 0];
    this.hasCustomCarriedHover = false;
    this.lastHoverTargetXZ = null;
    this.lastHoverTimeSec = 0;
    this.carriedHandVelXZ = [0, 0];
    this.isPointerDraggingBlock = false;
    this.grabResistance01 = 0;
    this.clearContactSubsteps = 0;
    this.extractionArcTimer = 0;
    this.grabRampSec = 0;
    this.blockLastCrazeTime.clear();
    this.blockLastImpactAudioTime.clear();
    this.blockLastWobbleTime.clear();
    this.blockPairLastContactSimTime.clear();
    this.awakeParkedBodies.length = 0;
    this.settleTimer = 0;
    this.accumulatorSec = 0;
    this.simTime = 0;
    this.towerRebuildSimTime = 0;
    this.multiccdReentrySubstepsRemaining = 0;
    this.lastGlobalWoodThudSimTime = -10.0;
    this.leanAngleDeg = 0;
    this.solverLeanAngleDeg = 0;
    this.wobbleSpeedDegPerSec = 0;
    this.currentLayers = towerLayers;
    this.peakLayers = towerLayers;
    this.liveHeightCm = towerLayers * BLOCK_HEIGHT * 10;
    this.peakHeightCm = towerLayers * BLOCK_HEIGHT * 10;
    this.lockedMinRow = this.gameMode === 'sandbox' ? null : towerLayers;

    this.activeSlidingGeomId = null;
    const totalBlocks = this.map.initialBlocks.length;
    const blockCap = totalBlocks + this.cfg.extraBodyPoolCapacity;
    this.cachedClampingLoadN = new Float64Array(blockCap);
    this.blockPhotoelasticStress01 = new Float32Array(blockCap);
    this.blockPhotoelasticWrench = new Float32Array(blockCap * 4);
    if (this.blockRockTransferCount.length < blockCap) {
      this.blockRockTransferCount = new Uint8Array(blockCap);
      this.blockLastRockTime = new Float64Array(blockCap);
      this.blockCoherentArcSubsteps = new Uint16Array(blockCap);
      this.blockPrevHorizOmegaX = new Float64Array(blockCap);
      this.blockPrevHorizOmegaZ = new Float64Array(blockCap);
    }
    if (this.unloadedBounceLastSimTime.length < blockCap) {
      this.unloadedBounceLastSimTime = new Float64Array(blockCap);
      this.unloadedBounceHopCount = new Uint8Array(blockCap);
      this.unloadedBouncePeakVClose = new Float64Array(blockCap);
    }
    if (this.blockPreSubstepLinSpeedSi.length < blockCap) {
      this.blockPreSubstepLinSpeedSi = new Float64Array(blockCap);
      this.blockPreSubstepAngSpeedSi = new Float64Array(blockCap);
    }
    this.unloadedBounceLastSimTime.fill(-10);
    this.unloadedBounceHopCount.fill(0);
    this.unloadedBouncePeakVClose.fill(0);
    this.blockPreSubstepLinSpeedSi.fill(0);
    this.blockPreSubstepAngSpeedSi.fill(0);
    this.blockRockTransferCount.fill(0);
    this.blockLastRockTime.fill(-10);
    this.blockCoherentArcSubsteps.fill(0);
    this.blockPrevHorizOmegaX.fill(0);
    this.blockPrevHorizOmegaZ.fill(0);

    this.blocks = buildNominalGlassBlocks(
      towerLayers,
      this.towerSeed,
      this.cfg.glassFriction[0],
      this.map.initialBlocks,
      this.map.blockBodyIds,
      this.map.blockGeomIds
    );
    for (let i = 0; i < this.blocks.length; i++) {
      this.bodyToBlock.set(this.blocks[i].bodyHandle, this.blocks[i]);
    }

    const seedWarmupContactPairs = () => {
      if (!this.model || !this.data) return;
      const ncon = this.data.ncon;
      if (ncon <= 0) return;
      const contacts = this.data.contact;
      const geomBodyId = this.model.geom_bodyid;
      for (let ci = 0; ci < ncon; ci++) {
        const c = contacts.get(ci);
        if (!c) continue;
        const cGeom = c.geom;
        const g1 = cGeom[0];
        const g2 = cGeom[1];
        c.delete();
        if (g1 < 0 || g2 < 0) continue;
        const b1 = geomBodyId[g1];
        const b2 = geomBodyId[g2];
        if ((b1 >= 2 && this.bodyToBlock.has(b1)) || (b2 >= 2 && this.bodyToBlock.has(b2))) {
          const pairKey = g1 < g2 ? g1 * 4096 + g2 : g2 * 4096 + g1;
          this.blockPairLastContactSimTime.set(pairKey, 0.0);
        }
      }
      contacts.delete();
    };

    if (canReuseCompiledModel && this.modelResetSnapshot) {
      this.cachedClampingLoadN.set(this.modelResetSnapshot.warmupClampingLoadN);
      this.blockPhotoelasticStress01.set(this.modelResetSnapshot.warmupPhotoelastic01);
      this.blockPhotoelasticWrench.set(this.modelResetSnapshot.warmupPhotoelasticWrench);
      for (const pk of this.modelResetSnapshot.warmupPairKeys) {
        this.blockPairLastContactSimTime.set(pk, 0.0);
      }
      // 2 quick steps re-establish warmstart contacts cleanly on the restored settled pose
      for (let i = 0; i < 2; i++) {
        this.mj.mj_step(this.model, this.data);
        if (this.data.ntree_awake > 0) seedWarmupContactPairs();
      }
    } else {
      // Quiet warm-up steps with fractures suppressed so beveled stack settles & caches initial clamping forces
      // IMPORTANT: capture clamping loads INSIDE the warmup loop while `ntree_awake > 0` before island sleep sets `ncon = 0`!
      const warmupSteps = Math.min(48, 16 + towerLayers * 2);
      for (let i = 0; i < warmupSteps; i++) {
        this.mj.mj_step(this.model, this.data);
        if (this.data.ntree_awake > 0) {
          this.refreshClampingLoadsAndPhotoelasticStress(true);
          seedWarmupContactPairs();
        }
      }
      this.modelResetSnapshot = {
        geom_type: new Int32Array(this.model.geom_type),
        geom_dataid: new Int32Array(this.model.geom_dataid),
        geom_contype: new Int32Array(this.model.geom_contype),
        geom_conaffinity: new Int32Array(this.model.geom_conaffinity),
        geom_group: new Int32Array(this.model.geom_group),
        geom_pos: new Float64Array(this.model.geom_pos),
        geom_quat: new Float64Array(this.model.geom_quat),
        geom_size: new Float64Array(this.model.geom_size),
        geom_rbound: new Float64Array(this.model.geom_rbound),
        geom_aabb: new Float64Array(this.model.geom_aabb),
        geom_friction: new Float64Array(this.model.geom_friction),
        geom_priority: new Int32Array(this.model.geom_priority),
        body_contype: new Int32Array(this.model.body_contype),
        body_conaffinity: new Int32Array(this.model.body_conaffinity),
        body_mass: new Float64Array(this.model.body_mass),
        body_subtreemass: new Float64Array(this.model.body_subtreemass),
        body_inertia: new Float64Array(this.model.body_inertia),
        body_iquat: new Float64Array(this.model.body_iquat),
        body_invweight0: new Float64Array(this.model.body_invweight0),
        dof_M0: new Float64Array(this.model.dof_M0),
        dof_invweight0: new Float64Array(this.model.dof_invweight0),
        dof_damping: new Float64Array(this.model.dof_damping),
        eq_obj1id: new Int32Array(this.model.eq_obj1id),
        eq_obj2id: new Int32Array(this.model.eq_obj2id),
        eq_data: new Float64Array(this.model.eq_data),
        warmupQpos: new Float64Array(this.data.qpos),
        warmupQvel: new Float64Array(this.data.qvel),
        warmupQaccWarmstart: new Float64Array(this.data.qacc_warmstart),
        warmupTreeAsleep: new Int32Array(this.data.tree_asleep),
        warmupNtreeAwake: this.data.ntree_awake,
        warmupClampingLoadN: new Float64Array(this.cachedClampingLoadN),
        warmupPhotoelastic01: new Float32Array(this.blockPhotoelasticStress01),
        warmupPhotoelasticWrench: new Float32Array(this.blockPhotoelasticWrench),
        warmupPairKeys: Array.from(this.blockPairLastContactSimTime.keys()),
      };
    }
    this.syncTransformsAndRules();
  }

  /**
   * Registers any dynamically appended block in `this.blocks` (e.g. `T3d` pool stress test) into `bodyToBlock`.
   */
  private ensureBlockRegistered(block: GlassBlockState): void {
    if (!this.bodyToBlock.has(block.bodyHandle)) {
      this.bodyToBlock.set(block.bodyHandle, block);
    }
  }

  /**
   * Refreshes per-block upper clamping pinch force (`cachedClampingLoadN`, in SI Newtons) and
   * normalized internal photoelastic stress (`blockPhotoelasticStress01`, `[0..1]`) directly from
   * MuJoCo 4.0's active `efc_force` constraint solution.
   *
   * Only updates blocks whose island is currently awake so sleeping blocks retain exact pre-sleep loads!
   */
  private resetActiveSlidingGeomFriction(): void {
    if (this.activeSlidingGeomId !== null && this.model) {
      const gid = this.activeSlidingGeomId;
      this.model.geom_priority[gid] = 0;
      const g3 = gid * 3;
      this.model.geom_friction[g3 + 0] = this.cfg.glassFriction[0];
      this.model.geom_friction[g3 + 1] = this.cfg.glassFriction[1];
      this.model.geom_friction[g3 + 2] = this.cfg.glassFriction[2];
      this.activeSlidingGeomId = null;
    }
  }

  private ensurePhotoScratchCapacity(nBlocks: number): void {
    if (this.photoCap >= nBlocks) {
      this.photoHasVert.fill(0, 0, nBlocks);
      this.photoHasAny.fill(0, 0, nBlocks);
      this.photoTopNeg.fill(0, 0, nBlocks);
      this.photoTopCtr.fill(0, 0, nBlocks);
      this.photoTopPos.fill(0, 0, nBlocks);
      this.photoBotNeg.fill(0, 0, nBlocks);
      this.photoBotCtr.fill(0, 0, nBlocks);
      this.photoBotPos.fill(0, 0, nBlocks);
      this.photoMidShear.fill(0, 0, nBlocks);
      this.photoTopF.fill(0, 0, nBlocks);
      this.photoTopAbsX.fill(0, 0, nBlocks);
      this.photoTopSigX.fill(0, 0, nBlocks);
      this.photoBotF.fill(0, 0, nBlocks);
      this.photoBotAbsX.fill(0, 0, nBlocks);
      this.photoBotSigX.fill(0, 0, nBlocks);
      return;
    }
    this.photoCap = Math.max(nBlocks, 96);
    this.photoHasVert = new Uint8Array(this.photoCap);
    this.photoHasAny = new Uint8Array(this.photoCap);
    this.photoTopNeg = new Float64Array(this.photoCap);
    this.photoTopCtr = new Float64Array(this.photoCap);
    this.photoTopPos = new Float64Array(this.photoCap);
    this.photoBotNeg = new Float64Array(this.photoCap);
    this.photoBotCtr = new Float64Array(this.photoCap);
    this.photoBotPos = new Float64Array(this.photoCap);
    this.photoMidShear = new Float64Array(this.photoCap);
    this.photoTopF = new Float64Array(this.photoCap);
    this.photoTopAbsX = new Float64Array(this.photoCap);
    this.photoTopSigX = new Float64Array(this.photoCap);
    this.photoBotF = new Float64Array(this.photoCap);
    this.photoBotAbsX = new Float64Array(this.photoCap);
    this.photoBotSigX = new Float64Array(this.photoCap);
  }

  private refreshClampingLoadsAndPhotoelasticStress(isWarmup: boolean = false): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d || d.ntree_awake === 0) return;

    const nBlocks = this.blocks.length;
    this.ensurePhotoScratchCapacity(nBlocks);
    const hasVerticalContact = this.photoHasVert;
    const hasAnyPhotoContact = this.photoHasAny;
    const topZoneNeg = this.photoTopNeg;
    const topZoneCtr = this.photoTopCtr;
    const topZonePos = this.photoTopPos;
    const botZoneNeg = this.photoBotNeg;
    const botZoneCtr = this.photoBotCtr;
    const botZonePos = this.photoBotPos;
    const midShear = this.photoMidShear;
    const topF = this.photoTopF;
    const topAbsX = this.photoTopAbsX;
    const topSigX = this.photoTopSigX;
    const botF = this.photoBotF;
    const botAbsX = this.photoBotAbsX;
    const botSigX = this.photoBotSigX;

    const contacts = d.contact;
    const efc = d.efc_force;
    const geomBodyId = m.geom_bodyid;
    const xpos = d.xpos;
    const qpos = d.qpos;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const ncon = d.ncon;
    const minShardGid = this.minShardGeomId;
    const maxShardGid = this.maxShardGeomId;
    const parkingGid = this.map.parkingShelfGeomId;

    const accumBlockLocalContact = (
      blk: GlassBlockState,
      wxDm: number,
      wyDm: number,
      wzDm: number,
      fn: number,
      ft: number
    ) => {
      const id = blk.id;
      if (id < 0 || id >= nBlocks || blk.shattered || blk.extracted) return;
      const info = this.worldCompat.getMapping(blk.bodyHandle);
      if (!info) return;
      const adr = info.qposAdr;
      const bxDm = qpos[adr + 0] * DM_PER_M;
      const byDm = qpos[adr + 1] * DM_PER_M;
      const bzDm = qpos[adr + 2] * DM_PER_M;
      const bQuat: Quat = [qpos[adr + 4], qpos[adr + 5], qpos[adr + 6], qpos[adr + 3]];
      const localPt = quatInverseRotateVec3(bQuat, [wxDm - bxDm, wyDm - byDm, wzDm - bzDm]);
      const uX = Math.max(-1.0, Math.min(1.0, localPt[0] / BLOCK_HALF_X));

      hasAnyPhotoContact[id] = 1;
      const fEff = fn + 0.42 * ft;
      const dNeg = (uX + 0.667) * 3.1;
      const dCtr = uX * 3.1;
      const dPos = (uX - 0.667) * 3.1;
      const wNeg = Math.exp(-dNeg * dNeg);
      const wCtr = Math.exp(-dCtr * dCtr);
      const wPos = Math.exp(-dPos * dPos);
      const invWSum = 1.0 / (wNeg + wCtr + wPos || 1.0);

      if (localPt[1] > 0.012) {
        topZoneNeg[id] += fEff * wNeg * invWSum;
        topZoneCtr[id] += fEff * wCtr * invWSum;
        topZonePos[id] += fEff * wPos * invWSum;
        topF[id] += fn;
        topAbsX[id] += fn * Math.abs(uX);
        topSigX[id] += fn * uX;
      } else if (localPt[1] < -0.012) {
        botZoneNeg[id] += fEff * wNeg * invWSum;
        botZoneCtr[id] += fEff * wCtr * invWSum;
        botZonePos[id] += fEff * wPos * invWSum;
        botF[id] += fn;
        botAbsX[id] += fn * Math.abs(uX);
        botSigX[id] += fn * uX;
      } else {
        midShear[id] += ft * 0.5;
      }
    };

    for (let i = 0; i < ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      const cGeom = c.geom;
      const g1 = cGeom[0];
      const g2 = cGeom[1];
      const nonBlock1 = g1 < 2 || g1 === parkingGid || (g1 >= minShardGid && g1 <= maxShardGid);
      const nonBlock2 = g2 < 2 || g2 === parkingGid || (g2 >= minShardGid && g2 <= maxShardGid);
      if (nonBlock1 && nonBlock2) {
        c.delete();
        continue;
      }
      const adr = c.efc_address;
      if (adr >= 0) {
        const fn = Math.max(0, efc[adr]);
        if (fn > 1e-4) {
          const b1 = geomBodyId[g1];
          const b2 = geomBodyId[g2];
          if (!isWarmup && fn > 2.2) {
            if (g1 >= minShardGid && g1 <= maxShardGid) {
              const s1 = this.bodyToShard.get(b1);
              if (s1 && fn > Math.max(2.2, s1.massKg * 420.0)) {
                this.demotePinchedHeroShardIfWedged(s1, fn);
              }
            }
            if (g2 >= minShardGid && g2 <= maxShardGid) {
              const s2 = this.bodyToShard.get(b2);
              if (s2 && fn > Math.max(2.2, s2.massKg * 420.0)) {
                this.demotePinchedHeroShardIfWedged(s2, fn);
              }
            }
          }
          const blk1 = !nonBlock1 && b1 >= 2 ? this.bodyToBlock.get(b1) : undefined;
          const blk2 = !nonBlock2 && b2 >= 2 ? this.bodyToBlock.get(b2) : undefined;
          if ((blk1 && !blk1.shattered && !blk1.extracted) || (blk2 && !blk2.shattered && !blk2.extracted)) {
            const ft = c.dim >= 3 ? Math.hypot(efc[adr + 1], efc[adr + 2]) : 0;
            const cPos = c.pos;
            const wxDm = cPos[0] * DM_PER_M;
            const wyDm = cPos[1] * DM_PER_M;
            const wzDm = cPos[2] * DM_PER_M;

            if (blk1 && !blk1.shattered && !blk1.extracted) {
              accumBlockLocalContact(blk1, wxDm, wyDm, wzDm, fn, ft);
            }
            if (blk2 && !blk2.shattered && !blk2.extracted) {
              accumBlockLocalContact(blk2, wxDm, wyDm, wzDm, fn, ft);
            }

            if (blk1 && blk2 && !blk1.shattered && !blk2.shattered && !blk1.extracted && !blk2.extracted) {
              const y1 = xpos[b1 * 3 + 1];
              const y2 = xpos[b2 * 3 + 1];
              if (y2 > y1 + 0.0068 && blk1.id < nBlocks) {
                if (hasVerticalContact[blk1.id] === 0) {
                  hasVerticalContact[blk1.id] = 1;
                  this.cachedClampingLoadN[blk1.id] = 0;
                }
                this.cachedClampingLoadN[blk1.id] += fn;
              } else if (y1 > y2 + 0.0068 && blk2.id < nBlocks) {
                if (hasVerticalContact[blk2.id] === 0) {
                  hasVerticalContact[blk2.id] = 1;
                  this.cachedClampingLoadN[blk2.id] = 0;
                }
                this.cachedClampingLoadN[blk2.id] += fn;
              }
            }
          }
        }
      }
      c.delete();
    }
    contacts.delete();

    const alpha = isWarmup ? 1.0 : 0.38;
    const keep = 1.0 - alpha;

    for (let i = 0; i < nBlocks; i++) {
      const b = this.blocks[i];
      const id = b.id;
      if (id >= this.blockPhotoelasticStress01.length) continue;
      const wBase = id * 4;
      if (b.shattered || b.extracted) {
        this.cachedClampingLoadN[id] = 0;
        this.blockPhotoelasticStress01[id] = 0;
        this.blockPhotoelasticWrench[wBase + 0] = 0;
        this.blockPhotoelasticWrench[wBase + 1] = 0;
        this.blockPhotoelasticWrench[wBase + 2] = 0;
        this.blockPhotoelasticWrench[wBase + 3] = 0;
        continue;
      }

      const treeId = bodyTreeId[b.bodyHandle];
      const isAwake = treeId >= 0 && treeAsleep[treeId] < 0;

      if (isAwake && id < nBlocks && hasVerticalContact[id] === 0) {
        let hasUpperNeighbor = false;
        for (let j = 0; j < nBlocks; j++) {
          const ob = this.blocks[j];
          if (!ob.shattered && !ob.extracted && ob.assignedLayer === b.assignedLayer + 1) {
            if (Math.hypot(ob.position[0] - b.position[0], ob.position[2] - b.position[2]) < 0.72) {
              hasUpperNeighbor = true;
              break;
            }
          }
        }
        if (!hasUpperNeighbor) {
          this.cachedClampingLoadN[id] = 0;
        }
      }

      // Only update photoelastic contact wrench when this block's island is awake (preserving exact pre-sleep forces!)
      if (isAwake || isWarmup) {
        if (hasAnyPhotoContact[id] === 0) {
          const dmgRes = Math.min(1.0, b.damage * 0.28);
          this.blockPhotoelasticWrench[wBase + 0] = this.blockPhotoelasticWrench[wBase + 0] * keep + dmgRes * alpha;
          this.blockPhotoelasticWrench[wBase + 1] = this.blockPhotoelasticWrench[wBase + 1] * keep + dmgRes * alpha;
          this.blockPhotoelasticWrench[wBase + 2] = this.blockPhotoelasticWrench[wBase + 2] * keep + dmgRes * alpha;
          this.blockPhotoelasticWrench[wBase + 3] = this.blockPhotoelasticWrench[wBase + 3] * keep;
        } else {
          const dmgLift = b.damage * 0.28;
          // True through-thickness compressive bearing column (`min(top, bot)`): an unloaded top block or
          // unsupported cantilever overhang (`bot === 0` or `top === 0`) has zero fake vertical column band!
          const isUnloadedTopCourse =
            !b.isLegalTarget &&
            b.assignedLayer >= Math.max(1, this.currentLayers - 1) &&
            this.cachedClampingLoadN[id] <= 0.05 &&
            topF[id] <= 0.05;
          const colForce = (fTop: number, fBot: number): number =>
            isUnloadedTopCourse
              ? 0.0
              : Math.max(0.0, 1.75 * Math.min(fTop, fBot) + 0.18 * Math.max(fTop, fBot) - 0.38);
          const tNeg = Math.min(1.0, colForce(topZoneNeg[id], botZoneNeg[id]) / 7.5 + dmgLift);
          const tCtr = Math.min(1.0, colForce(topZoneCtr[id], botZoneCtr[id]) / 7.5 + dmgLift);
          const tPos = Math.min(1.0, colForce(topZonePos[id], botZonePos[id]) / 7.5 + dmgLift);

          let bend01 = Math.min(
            0.45,
            (isUnloadedTopCourse ? Math.max(0.0, midShear[id] - 0.18) : midShear[id]) / 4.5
          );
          if (topF[id] > 0.25 && botF[id] > 0.25) {
            const pairedLoadN = Math.min(topF[id], botF[id]);
            const topMeanX = topSigX[id] / topF[id];
            const botMeanX = botSigX[id] / botF[id];
            const topSpreadX = topAbsX[id] / topF[id];
            const botSpreadX = botAbsX[id] / botF[id];
            const cantileverLever = Math.abs(topMeanX - botMeanX);
            const spanBridgeLever = Math.abs(botSpreadX - topSpreadX) * 0.85;
            const bendMomentProxy = pairedLoadN * Math.max(cantileverLever, spanBridgeLever);
            bend01 = Math.min(1.0, Math.max(bend01, bendMomentProxy / 2.8) + b.damage * 0.20);
          }

          this.blockPhotoelasticWrench[wBase + 0] = this.blockPhotoelasticWrench[wBase + 0] * keep + tNeg * alpha;
          this.blockPhotoelasticWrench[wBase + 1] = this.blockPhotoelasticWrench[wBase + 1] * keep + tCtr * alpha;
          this.blockPhotoelasticWrench[wBase + 2] = this.blockPhotoelasticWrench[wBase + 2] * keep + tPos * alpha;
          this.blockPhotoelasticWrench[wBase + 3] = this.blockPhotoelasticWrench[wBase + 3] * keep + bend01 * alpha;
        }
        this.blockPhotoelasticStress01[id] = Math.max(
          this.blockPhotoelasticWrench[wBase + 0],
          this.blockPhotoelasticWrench[wBase + 1],
          this.blockPhotoelasticWrench[wBase + 2],
          this.blockPhotoelasticWrench[wBase + 3]
        );
      }
    }
  }

  public getBlockPhotoelasticStress01(blockId: number): number {
    if (blockId < 0 || blockId >= this.blockPhotoelasticStress01.length) return 0;
    return this.blockPhotoelasticStress01[blockId];
  }

  public writeBlockPhotoelasticWrench(blockId: number, out: Float32Array, offset: number): void {
    const wBase = blockId * 4;
    if (blockId < 0 || wBase + 3 >= this.blockPhotoelasticWrench.length) {
      out[offset + 0] = 0;
      out[offset + 1] = 0;
      out[offset + 2] = 0;
      out[offset + 3] = 0;
      return;
    }
    out[offset + 0] = this.blockPhotoelasticWrench[wBase + 0];
    out[offset + 1] = this.blockPhotoelasticWrench[wBase + 1];
    out[offset + 2] = this.blockPhotoelasticWrench[wBase + 2];
    out[offset + 3] = this.blockPhotoelasticWrench[wBase + 3];
  }

  public computeBlockClampingLoadN(blockId: number): number {
    const blk = this.blocks[blockId];
    if (!blk || blk.shattered || blk.extracted) return 0;
    const directContactN = blockId < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[blockId] : 0;
    // In Froude (`Fr = 1`) decimeter units (`g_sim = -9.81 dm/s^2`, `dt_sim = sqrt(10) * dt_si`),
    // Vitrail's clamping load convention (`F_froude = J_sim / dt_sim` in `PhysicsWorld.ts`) equals SI normal force
    // across exact 2-column hydrostatic sharing, whereas MuJoCo's 3D Newton elastic brick lattice can arch ~35% of
    // upper load across slightly stiffer diagonal corner contacts (`S1` vs `S0/S2`). Blend direct `efc_force`
    // contact normal force with tributary upper-stack column weight whenever the block is actively in contact (`directContactN > 0.85 N`),
    // while keeping genuine loose blocks (`directContactN <= 0.50 N`) strictly at their exact `efc_force` reading!
    if (directContactN <= 0.65) {
      return directContactN;
    }
    let blocksAbove = 0;
    let maxTightDirectN = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const ob = this.blocks[i];
      if (ob.shattered || ob.extracted) continue;
      if (ob.position[1] > blk.position[1] + BLOCK_HEIGHT * 0.55 && Math.hypot(ob.position[0], ob.position[2]) < 0.85) {
        blocksAbove++;
      } else if (ob.assignedLayer === blk.assignedLayer && !ob.isLooseSlot) {
        const obLoad = ob.id < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[ob.id] : 0;
        if (obLoad > maxTightDirectN) maxTightDirectN = obLoad;
      }
    }
    let bearingInSameRow = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const ob = this.blocks[i];
      if (ob.shattered || ob.extracted || ob.assignedLayer !== blk.assignedLayer) continue;
      const obLoad = ob.id < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[ob.id] : 0;
      if (obLoad > 0.65) {
        bearingInSameRow +=
          !blk.isLooseSlot && ob.isLooseSlot && maxTightDirectN > 0 && obLoad < 0.65 * maxTightDirectN
            ? (obLoad / maxTightDirectN) * 0.5
            : 1.0;
      }
    }
    const tributaryN =
      bearingInSameRow > 0 ? (blocksAbove * TARGET_BLOCK_MASS_KG * 9.81) / bearingInSameRow : directContactN;
    return Math.max(directContactN, directContactN * 0.28 + tributaryN * 0.72);
  }

  private wakeTowerForInteraction(): void {
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (!b.shattered && !b.extracted) {
        if (this.model) {
          const info = this.worldCompat.getMapping(b.bodyHandle);
          if (info) {
            if (this.sandboxCollapseActive) {
              info.quietTicks = 0;
              info.settledAsleep = false;
            }
            for (let k = 0; k < 6; k++) this.model.dof_damping[info.dofAdr + k] = 0;
          }
        }
        this.worldCompat.wakeBody(b.bodyHandle);
      }
    }
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      if (s.position[1] > 0.18) {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        this.shardNoSupportSubsteps[s.slotIndex] = 0;
        this.worldCompat.wakeBody(s.bodyHandle);
      }
    }
  }

  public tapNudgeBlock(blockId: number, localNormal: Vec3, localPoint: Vec3): boolean {
    const block = this.blocks[blockId];
    if (!block || block.shattered || block.extracted || !this.data) return false;
    if (this.gameMode !== 'sandbox' && this.turnPhase !== 'SELECT_BLOCK') return false;
    if (this.gameMode !== 'sandbox' && !block.isLegalTarget) return false;

    const clampingLoadN = this.computeBlockClampingLoadN(blockId);
    const pinch01 = Math.max(0.0, Math.min(1.0, clampingLoadN / 3.5));

    const axisX = quatRotateVec3(block.rotation, [1, 0, 0]);
    let dirSign = 1;
    if (Math.abs(localNormal[0]) > 0.65) {
      dirSign = -Math.sign(localNormal[0]);
    } else {
      dirSign = localPoint[0] >= 0 ? 1 : -1;
    }

    // Convert Vitrail Froude impulse (0.0235 / 0.14 kg*dm/s_sim) to SI velocity jump dv_si = J_sim / (m * sqrt(10))
    // (~0.104 m/s on a 71.1 g loose prism -> physical ~2.0 mm slide distance under mu = 0.275!)
    const impulseMagSim = this.gameMode === 'sandbox' ? 0.14 : 0.0235;
    const massKg = Math.max(0.01, this.model!.body_mass[block.bodyHandle]);
    const dvSi = (impulseMagSim * INV_FROUDE_VEL_SCALE) / massKg;

    const info = this.worldCompat.getMapping(block.bodyHandle);
    if (info) {
      this.worldCompat.wakeBody(block.bodyHandle);
      this.data.qvel[info.dofAdr + 0] += axisX[0] * dirSign * dvSi;
      this.data.qvel[info.dofAdr + 2] += axisX[2] * dirSign * dvSi;
    }

    this.audio.playTapTestResonance(
      pinch01,
      block.position,
      this.listenerCamPos,
      this.listenerCamRight,
      block.damage
    );
    return true;
  }

  public startGrabBlock(blockId: number, localPoint: Vec3, worldPoint: Vec3): boolean {
    const block = this.blocks[blockId];
    if (!block || block.shattered || !this.model || !this.data) return false;

    if (this.gameMode !== 'sandbox') {
      if (this.turnPhase === 'COLLAPSED' || this.turnPhase === 'SETTLING') return false;
      if (this.turnPhase === 'EXTRACTED' && block.id !== this.activeBlockId) return false;
      if (this.turnPhase === 'SELECT_BLOCK' && !block.isLegalTarget) return false;
    }

    // Wake adjacent layers cleanly (no artificial Fixed-body freeze window needed in MuJoCo Newton!)
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (!b.shattered && !b.extracted && Math.abs(b.assignedLayer - block.assignedLayer) <= 1) {
        this.worldCompat.wakeBody(b.bodyHandle);
      }
    }

    this.activeBlockId = blockId;
    this.grabLocalPoint = [...localPoint];
    this.grabPlaneY = worldPoint[1];
    this.grabWorldAnchor = [...worldPoint];
    this.grabCurrentWorldOnBlock = [...worldPoint];
    this.isPointerDraggingBlock = true;
    this.clearContactSubsteps = 0;
    this.grabRampSec = 0;

    if (!block.extracted) {
      // Configure MuJoCo soft `<connect>` equality constraint between `hand_anchor` (body 1) and `block.bodyHandle`
      const eqId = this.map.grabEqualityId;
      this.model.eq_obj1id[eqId] = this.map.handMocapBodyId;
      this.model.eq_obj2id[eqId] = block.bodyHandle;
      // eq_data[0..2] = anchor on mocap hand (0,0,0); eq_data[3..5] = local grab point on block in SI meters!
      const eqBase = eqId * 11;
      this.model.eq_data[eqBase + 0] = 0;
      this.model.eq_data[eqBase + 1] = 0;
      this.model.eq_data[eqBase + 2] = 0;
      this.model.eq_data[eqBase + 3] = localPoint[0] * M_PER_DM;
      this.model.eq_data[eqBase + 4] = localPoint[1] * M_PER_DM;
      this.model.eq_data[eqBase + 5] = localPoint[2] * M_PER_DM;

      this.data.mocap_pos[0] = worldPoint[0] * M_PER_DM;
      this.data.mocap_pos[1] = worldPoint[1] * M_PER_DM;
      this.data.mocap_pos[2] = worldPoint[2] * M_PER_DM;
      this.data.eq_active[eqId] = 1;

      if (this.turnPhase === 'SELECT_BLOCK') {
        this.turnPhase = 'PULLING';
      }
    }

    return true;
  }

  public updateGrabTarget(worldTargetOnPlane: Vec3): void {
    this.grabWorldAnchor[0] = worldTargetOnPlane[0];
    this.grabWorldAnchor[1] = this.grabPlaneY;
    this.grabWorldAnchor[2] = worldTargetOnPlane[2];
  }

  public get isExtractionLiftReady(): boolean {
    return this.turnPhase === 'EXTRACTED' && this.extractionArcTimer >= 0.20;
  }

  private static quatMul(a: Quat, b: Quat): Quat {
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
  }

  private static solve3x3(m: number[], b: Vec3): Vec3 {
    const a00 = m[0], a01 = m[1], a02 = m[2];
    const a10 = m[3], a11 = m[4], a12 = m[5];
    const a20 = m[6], a21 = m[7], a22 = m[8];
    const det =
      a00 * (a11 * a22 - a12 * a21) -
      a01 * (a10 * a22 - a12 * a20) +
      a02 * (a10 * a21 - a11 * a20);
    if (Math.abs(det) < 1e-12) return [0, 0, 0];
    const invDet = 1.0 / det;
    return [
      ((a11 * a22 - a12 * a21) * b[0] +
        (a02 * a21 - a01 * a22) * b[1] +
        (a01 * a12 - a02 * a11) * b[2]) *
        invDet,
      ((a12 * a20 - a10 * a22) * b[0] +
        (a00 * a22 - a02 * a20) * b[1] +
        (a02 * a10 - a00 * a12) * b[2]) *
        invDet,
      ((a10 * a21 - a11 * a20) * b[0] +
        (a01 * a20 - a00 * a21) * b[1] +
        (a00 * a11 - a01 * a10) * b[2]) *
        invDet,
    ];
  }

  private static wrapYawRad(rad: number): number {
    let r = rad;
    while (r > Math.PI) r -= Math.PI * 2;
    while (r < -Math.PI) r += Math.PI * 2;
    return r;
  }

  private static getBlockXZAxes(q: Quat): { uX: [number, number]; uZ: [number, number] } {
    const ax = quatRotateVec3(q, [1, 0, 0]);
    const az = quatRotateVec3(q, [0, 0, 1]);
    const lx = Math.hypot(ax[0], ax[2]);
    const lz = Math.hypot(az[0], az[2]);
    return {
      uX: lx > 1e-5 ? [ax[0] / lx, ax[2] / lx] : [1, 0],
      uZ: lz > 1e-5 ? [az[0] / lz, az[2] / lz] : [0, 1],
    };
  }

  private static obb2DOverlapDepth(
    ax: number,
    az: number,
    uAx: [number, number],
    uAz: [number, number],
    hxA: number,
    hzA: number,
    bx: number,
    bz: number,
    uBx: [number, number],
    uBz: [number, number],
    hxB: number,
    hzB: number,
    centerRefX: number,
    centerRefZ: number
  ): { overlap: number; dirX: number; dirZ: number } | null {
    const axes: Array<[number, number]> = [uAx, uAz, uBx, uBz];
    const dx = ax - bx;
    const dz = az - bz;
    let minOverlap = Infinity;
    let bestAxis: [number, number] = [1, 0];
    let bestProj = 0;

    for (let i = 0; i < 4; i++) {
      const lx = axes[i][0];
      const lz = axes[i][1];
      const rA =
        hxA * Math.abs(uAx[0] * lx + uAx[1] * lz) +
        hzA * Math.abs(uAz[0] * lx + uAz[1] * lz);
      const rB =
        hxB * Math.abs(uBx[0] * lx + uBx[1] * lz) +
        hzB * Math.abs(uBz[0] * lx + uBz[1] * lz);
      const proj = dx * lx + dz * lz;
      const ov = rA + rB - Math.abs(proj);
      if (ov <= 0) return null;
      if (ov < minOverlap) {
        minOverlap = ov;
        bestAxis = [lx, lz];
        bestProj = proj;
      }
    }

    let sign = bestProj >= 0 ? 1 : -1;
    if (Math.abs(bestProj) < 1e-4) {
      const pPlusX = ax + bestAxis[0] * minOverlap - centerRefX;
      const pPlusZ = az + bestAxis[1] * minOverlap - centerRefZ;
      const pMinusX = ax - bestAxis[0] * minOverlap - centerRefX;
      const pMinusZ = az - bestAxis[1] * minOverlap - centerRefZ;
      sign =
        pPlusX * pPlusX + pPlusZ * pPlusZ <= pMinusX * pMinusX + pMinusZ * pMinusZ ? 1 : -1;
    }

    return {
      overlap: minOverlap,
      dirX: bestAxis[0] * sign,
      dirZ: bestAxis[1] * sign,
    };
  }

  private static getCarriedSupportBounds(yawOffsetRad: number): {
    maxAbsV: number;
    maxAbsU: number;
  } {
    const cosY = Math.abs(Math.cos(yawOffsetRad));
    const sinY = Math.abs(Math.sin(yawOffsetRad));
    const projHalfV = BLOCK_HALF_X * cosY + BLOCK_HALF_Z * sinY;
    const projHalfU = BLOCK_HALF_X * sinY + BLOCK_HALF_Z * cosY;
    return {
      maxAbsV: Math.max(0.24, 0.375 - projHalfV + 0.17),
      maxAbsU: Math.max(0.24, 0.375 - projHalfU + 0.17),
    };
  }

  public rotateCarriedBlock(deltaYawRad: number): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    const prevYaw = this.carriedYawOffsetRad;
    const rawV = this.carriedRawLocalVL[0];
    const rawU = this.carriedRawLocalVL[1];
    this.hasCustomCarriedHover = true;

    const targetYaw = MujocoPhysicsEngine.wrapYawRad(prevYaw + deltaYawRad);
    this.carriedYawOffsetRad = targetYaw;
    if (this.resolveCarriedPlacementPose(rawV, rawU, false)) return;

    for (let step = 11; step >= 0; step--) {
      const candYaw = MujocoPhysicsEngine.wrapYawRad(prevYaw + deltaYawRad * (step / 12));
      this.carriedYawOffsetRad = candYaw;
      if (this.resolveCarriedPlacementPose(rawV, rawU, false)) return;
    }

    this.carriedYawOffsetRad = 0;
    this.resolveCarriedPlacementPose(rawV, rawU, true);
  }

  public resetCarriedBlockRotation(): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    this.carriedYawOffsetRad = 0;
    this.resolveCarriedPlacementPose(this.carriedRawLocalVL[0], this.carriedRawLocalVL[1], true);
  }

  private resolveCarriedPlacementPose(
    desiredV: number,
    desiredU: number,
    allowYawRelaxFallback: boolean = true
  ): boolean {
    const frame = this.topSupportFrame;
    const totalYaw = frame.defaultYawRad + this.carriedYawOffsetRad;
    const yawQuat = quatFromAxisAngle([0, 1, 0], totalYaw);
    const carriedQuat = MujocoPhysicsEngine.quatMul(frame.tiltQuat, yawQuat);
    const { uX: uAx, uZ: uAz } = MujocoPhysicsEngine.getBlockXZAxes(carriedQuat);
    const { maxAbsV, maxAbsU } = MujocoPhysicsEngine.getCarriedSupportBounds(this.carriedYawOffsetRad);

    const colHx = BLOCK_HALF_X - 0.0008;
    const colHz = BLOCK_HALF_Z - 0.0014;
    const abutmentClearance = 0.0040;
    const minSafeClearance = 0.0022;

    const obstacles: Array<{ x: number; z: number; uX: [number, number]; uZ: [number, number] }> = [];
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId && b.assignedLayer === frame.targetLayer) {
        if (Math.hypot(b.position[0] - frame.layerCenter[0], b.position[2] - frame.layerCenter[2]) < 0.95) {
          const axes = MujocoPhysicsEngine.getBlockXZAxes(b.rotation);
          obstacles.push({ x: b.position[0], z: b.position[2], uX: axes.uX, uZ: axes.uZ });
        }
      }
    }

    const maxOverlapAt = (px: number, pz: number, clearance: number): number => {
      let worst = 0;
      for (let i = 0; i < obstacles.length; i++) {
        const ob = obstacles[i];
        const hit = MujocoPhysicsEngine.obb2DOverlapDepth(
          px, pz, uAx, uAz, colHx + clearance, colHz + clearance,
          ob.x, ob.z, ob.uX, ob.uZ, colHx, colHz,
          frame.layerCenter[0], frame.layerCenter[2]
        );
        if (hit && hit.overlap > worst) worst = hit.overlap;
      }
      return worst;
    };

    const clampedV = Math.max(-maxAbsV, Math.min(maxAbsV, desiredV));
    const clampedU = Math.max(-maxAbsU, Math.min(maxAbsU, desiredU));

    let px = frame.layerCenter[0] + frame.lengthAxis[0] * clampedV + frame.spacingAxis[0] * clampedU;
    let pz = frame.layerCenter[2] + frame.lengthAxis[2] * clampedV + frame.spacingAxis[2] * clampedU;

    if (obstacles.length > 0) {
      for (let iter = 0; iter < 6; iter++) {
        let moved = false;
        for (let i = 0; i < obstacles.length; i++) {
          const ob = obstacles[i];
          const mtv = MujocoPhysicsEngine.obb2DOverlapDepth(
            px, pz, uAx, uAz, colHx + abutmentClearance, colHz + abutmentClearance,
            ob.x, ob.z, ob.uX, ob.uZ, colHx, colHz,
            frame.layerCenter[0], frame.layerCenter[2]
          );
          if (mtv && mtv.overlap > 1e-5) {
            px += mtv.dirX * mtv.overlap;
            pz += mtv.dirZ * mtv.overlap;
            moved = true;
          }
        }
        if (!moved) break;
      }
    }

    const relX = px - frame.layerCenter[0];
    const relZ = pz - frame.layerCenter[2];
    let finalV = relX * frame.lengthAxis[0] + relZ * frame.lengthAxis[2];
    let finalU = relX * frame.spacingAxis[0] + relZ * frame.spacingAxis[2];

    let poseValid = true;
    const outOfSupportBounds = Math.abs(finalV) > maxAbsV + 0.01 || Math.abs(finalU) > maxAbsU + 0.01;
    if (obstacles.length > 0 && (outOfSupportBounds || maxOverlapAt(px, pz, minSafeClearance) > 1e-5)) {
      const candVs: number[] = [clampedV, Math.max(-0.14, Math.min(0.14, clampedV)), 0];
      if (maxAbsV > 0.26) {
        const vSteps = Math.round(maxAbsV / 0.03);
        for (let kv = -vSteps; kv <= vSteps; kv++) candVs.push(kv * 0.03);
      }
      const candUs: number[] = [];
      const obUs: number[] = [];
      for (let i = 0; i < obstacles.length; i++) {
        const ob = obstacles[i];
        const obRelX = ob.x - frame.layerCenter[0];
        const obRelZ = ob.z - frame.layerCenter[2];
        const obU = obRelX * frame.spacingAxis[0] + obRelZ * frame.spacingAxis[2];
        obUs.push(obU);
        const rSpan =
          (colHx + abutmentClearance) * Math.abs(uAx[0] * frame.spacingAxis[0] + uAx[1] * frame.spacingAxis[2]) +
          (colHz + abutmentClearance) * Math.abs(uAz[0] * frame.spacingAxis[0] + uAz[1] * frame.spacingAxis[2]) +
          colHx * Math.abs(ob.uX[0] * frame.spacingAxis[0] + ob.uX[1] * frame.spacingAxis[2]) +
          colHz * Math.abs(ob.uZ[0] * frame.spacingAxis[0] + ob.uZ[1] * frame.spacingAxis[2]);
        candUs.push(obU - rSpan, obU + rSpan);
      }
      for (let i = 0; i < obUs.length; i++) {
        for (let j = i + 1; j < obUs.length; j++) candUs.push((obUs[i] + obUs[j]) * 0.5);
      }
      const uSteps = Math.floor(maxAbsU / 0.015);
      for (let k = -uSteps; k <= uSteps; k++) candUs.push(k * 0.015);

      let bestCost = Infinity;
      let foundSafe = false;
      for (let iv = 0; iv < candVs.length; iv++) {
        const cv = candVs[iv];
        if (Math.abs(cv) > maxAbsV) continue;
        for (let iu = 0; iu < candUs.length; iu++) {
          const cu = candUs[iu];
          if (Math.abs(cu) > maxAbsU) continue;
          const cpx = frame.layerCenter[0] + frame.lengthAxis[0] * cv + frame.spacingAxis[0] * cu;
          const cpz = frame.layerCenter[2] + frame.lengthAxis[2] * cv + frame.spacingAxis[2] * cu;
          if (maxOverlapAt(cpx, cpz, minSafeClearance) <= 1e-5) {
            const dv = cv - clampedV;
            const du = cu - clampedU;
            const cost = dv * dv * 1.4 + du * du;
            if (cost < bestCost) {
              bestCost = cost;
              finalV = cv;
              finalU = cu;
              foundSafe = true;
            }
          }
        }
      }
      if (!foundSafe) {
        if (allowYawRelaxFallback && Math.abs(this.carriedYawOffsetRad) > 1e-4) {
          const origYaw = this.carriedYawOffsetRad;
          for (let step = 9; step >= 0; step--) {
            this.carriedYawOffsetRad = origYaw * (step / 10);
            if (this.resolveCarriedPlacementPose(desiredV, desiredU, false)) return true;
          }
        }
        poseValid = false;
      }
    } else {
      finalV = Math.max(-maxAbsV, Math.min(maxAbsV, finalV));
      finalU = Math.max(-maxAbsU, Math.min(maxAbsU, finalU));
    }

    const landX = frame.layerCenter[0] + frame.lengthAxis[0] * finalV + frame.spacingAxis[0] * finalU;
    const landY = frame.layerCenter[1] + frame.lengthAxis[1] * finalV + frame.spacingAxis[1] * finalU;
    const landZ = frame.layerCenter[2] + frame.lengthAxis[2] * finalV + frame.spacingAxis[2] * finalU;

    this.carriedLandingPos = [landX, landY, landZ];
    this.carriedTargetPos = [
      landX + frame.upNormal[0] * TOP_HOVER_CLEARANCE_DM,
      landY + frame.upNormal[1] * TOP_HOVER_CLEARANCE_DM,
      landZ + frame.upNormal[2] * TOP_HOVER_CLEARANCE_DM,
    ];
    this.carriedTargetQuat = [...carriedQuat];

    if (this.openTopSlots.length > 0) {
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < this.openTopSlots.length; i++) {
        const s = this.openTopSlots[i];
        const d = Math.hypot(landX - s.position[0], landZ - s.position[2]);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      this.hoveredSlotIndex = this.openTopSlots[bestIdx].slotIndex;
    } else {
      this.hoveredSlotIndex = null;
    }
    return poseValid;
  }

  public updateCarriedHover(rayOrigin: Vec3, rayDir: Vec3): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    if (this.openTopSlots.length === 0) this.computeOpenTopSlots();

    const frame = this.topSupportFrame;
    const lc = frame.layerCenter;
    const n = frame.upNormal;

    let hitX = lc[0], hitY = lc[1], hitZ = lc[2];
    const denom = rayDir[0] * n[0] + rayDir[1] * n[1] + rayDir[2] * n[2];
    if (Math.abs(denom) > 1e-4) {
      const t = ((lc[0] - rayOrigin[0]) * n[0] + (lc[1] - rayOrigin[1]) * n[1] + (lc[2] - rayOrigin[2]) * n[2]) / denom;
      if (t > 0) {
        hitX = rayOrigin[0] + rayDir[0] * t;
        hitY = rayOrigin[1] + rayDir[1] * t;
        hitZ = rayOrigin[2] + rayDir[2] * t;
      }
    } else if (Math.abs(rayDir[1]) > 1e-4) {
      const t = (lc[1] - rayOrigin[1]) / rayDir[1];
      if (t > 0) {
        hitX = rayOrigin[0] + rayDir[0] * t;
        hitY = lc[1];
        hitZ = rayOrigin[2] + rayDir[2] * t;
      }
    }

    const rx = hitX - lc[0];
    const ry = hitY - lc[1];
    const rz = hitZ - lc[2];
    const rawV = rx * frame.lengthAxis[0] + ry * frame.lengthAxis[1] + rz * frame.lengthAxis[2];
    const rawU = rx * frame.spacingAxis[0] + ry * frame.spacingAxis[1] + rz * frame.spacingAxis[2];

    const { maxAbsV, maxAbsU } = MujocoPhysicsEngine.getCarriedSupportBounds(this.carriedYawOffsetRad);
    const clampedV = Math.max(-maxAbsV, Math.min(maxAbsV, rawV));
    const clampedU = Math.max(-maxAbsU, Math.min(maxAbsU, rawU));
    this.carriedRawLocalVL = [clampedV, clampedU];
    this.hasCustomCarriedHover = true;

    this.resolveCarriedPlacementPose(clampedV, clampedU, true);

    if (this.lastHoverTargetXZ === null) {
      this.lastHoverTargetXZ = [this.carriedLandingPos[0], this.carriedLandingPos[2]];
      this.lastHoverTimeSec = this.simTime;
    } else {
      const dtReal = this.simTime - this.lastHoverTimeSec;
      if (dtReal > 1e-4) {
        if (dtReal < 0.25) {
          const vx = (this.carriedLandingPos[0] - this.lastHoverTargetXZ[0]) / dtReal;
          const vz = (this.carriedLandingPos[2] - this.lastHoverTargetXZ[1]) / dtReal;
          const blend = 1.0 - Math.exp(-dtReal / 0.022);
          this.carriedHandVelXZ[0] += (vx - this.carriedHandVelXZ[0]) * blend;
          this.carriedHandVelXZ[1] += (vz - this.carriedHandVelXZ[1]) * blend;
        }
        this.lastHoverTargetXZ = [this.carriedLandingPos[0], this.carriedLandingPos[2]];
        this.lastHoverTimeSec = this.simTime;
      }
    }
  }

  public endGrab(dropIfPoisedOverTower: boolean = false): void {
    this.isPointerDraggingBlock = false;
    this.grabResistance01 = 0;
    this.audio.updateSlideFriction(0, 0, 0, 0);
    this.resetActiveSlidingGeomFriction();

    if (this.data) {
      this.data.eq_active[this.map.grabEqualityId] = 0;
    }

    if (this.turnPhase !== 'EXTRACTED') {
      const hadActiveGrab = this.activeBlockId !== null;
      this.activeBlockId = null;
      if (this.turnPhase === 'PULLING') {
        this.turnPhase = 'SELECT_BLOCK';
      }
      if (hadActiveGrab) {
        // Wake active tower blocks/shards on released/aborted pull so any precarious overhang resolves under gravity!
        this.worldCompat.wakeActiveTowerBlocks();
      }
      return;
    }

    if (dropIfPoisedOverTower) {
      this.releaseExtractedBlockOntoTop();
    }
  }

  public releaseExtractedBlockOntoTop(): boolean {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null || !this.data) return false;
    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) return false;

    this.computeOpenTopSlots();
    const frame = this.topSupportFrame;

    if (this.hasCustomCarriedHover) {
      this.resolveCarriedPlacementPose(this.carriedRawLocalVL[0], this.carriedRawLocalVL[1]);
    } else {
      const chosenSlot =
        this.openTopSlots.find((s) => s.slotIndex === this.hoveredSlotIndex) ??
        this.openTopSlots[Math.min(1, Math.max(0, this.openTopSlots.length - 1))];
      if (!chosenSlot) return false;
      const dx = chosenSlot.position[0] - frame.layerCenter[0];
      const dy = chosenSlot.position[1] - frame.layerCenter[1];
      const dz = chosenSlot.position[2] - frame.layerCenter[2];
      const sv = dx * frame.lengthAxis[0] + dy * frame.lengthAxis[1] + dz * frame.lengthAxis[2];
      const su = dx * frame.spacingAxis[0] + dy * frame.spacingAxis[1] + dz * frame.spacingAxis[2];
      this.carriedYawOffsetRad = 0;
      this.resolveCarriedPlacementPose(sv, su);
    }

    const rb = this.worldCompat.getRigidBody(block.bodyHandle);
    const col = this.worldCompat.getCollider(block.colliderHandle);
    if (!rb || !col) return false;

    this.isPointerDraggingBlock = false;
    this.data.eq_active[this.map.grabEqualityId] = 0;
    col.setSensor(false);
    col.setFriction(this.cfg.glassFriction[0]);
    block.currentFriction = this.cfg.glassFriction[0];
    rb.setGravityScale(1.0, true);

    const relX = this.carriedLandingPos[0] + frame.upNormal[0] * TOP_HOVER_CLEARANCE_DM;
    const relY = this.carriedLandingPos[1] + frame.upNormal[1] * TOP_HOVER_CLEARANCE_DM;
    const relZ = this.carriedLandingPos[2] + frame.upNormal[2] * TOP_HOVER_CLEARANCE_DM;
    rb.setTranslation({ x: relX, y: relY, z: relZ }, true);
    rb.setRotation(
      {
        x: this.carriedTargetQuat[0],
        y: this.carriedTargetQuat[1],
        z: this.carriedTargetQuat[2],
        w: this.carriedTargetQuat[3],
      },
      true
    );
    block.position[0] = relX;
    block.position[1] = relY;
    block.position[2] = relZ;
    block.rotation[0] = this.carriedTargetQuat[0];
    block.rotation[1] = this.carriedTargetQuat[1];
    block.rotation[2] = this.carriedTargetQuat[2];
    block.rotation[3] = this.carriedTargetQuat[3];

    let relVxReal = this.carriedHandVelXZ[0] * 0.20;
    let relVzReal = this.carriedHandVelXZ[1] * 0.20;
    const relHorizReal = Math.hypot(relVxReal, relVzReal);
    if (relHorizReal > 2.2) {
      const scale = 2.2 / relHorizReal;
      relVxReal *= scale;
      relVzReal *= scale;
    }
    rb.setLinvel(
      {
        x: relVxReal / SIM_TIME_SCALE,
        y: -0.32,
        z: relVzReal / SIM_TIME_SCALE,
      },
      true
    );
    rb.setAngvel({ x: 0, y: 0, z: 0 }, true);

    const usedSlots = new Set<number>();
    for (const ob of this.blocks) {
      if (!ob.shattered && !ob.extracted && ob.id !== block.id && ob.assignedLayer === frame.targetLayer) {
        usedSlots.add(ob.assignedSlot);
      }
    }
    const relDx = this.carriedLandingPos[0] - frame.layerCenter[0];
    const relDz = this.carriedLandingPos[2] - frame.layerCenter[2];
    const placedU = relDx * frame.spacingAxis[0] + relDz * frame.spacingAxis[2];
    const idealSlot = Math.max(0, Math.min(2, Math.round(placedU / BLOCK_WIDTH + 1)));
    let assignedSlot = idealSlot;
    if (usedSlots.has(assignedSlot)) {
      const candidates = [0, 1, 2].sort((a, b) => Math.abs(a - idealSlot) - Math.abs(b - idealSlot));
      for (const cand of candidates) {
        if (!usedSlots.has(cand)) {
          assignedSlot = cand;
          break;
        }
      }
    }

    block.extracted = false;
    block.assignedLayer = frame.targetLayer;
    block.assignedSlot = assignedSlot;

    this.wakeTowerForInteraction();
    this.activeBlockId = null;
    this.hoveredSlotIndex = null;
    this.carriedYawOffsetRad = 0;
    this.carriedRawLocalVL = [0, 0];
    this.hasCustomCarriedHover = false;
    this.lastHoverTargetXZ = null;
    this.carriedHandVelXZ = [0, 0];
    this.carriedFromDrag = false;
    this.clearContactSubsteps = 0;
    this.turnPhase = 'SETTLING';
    this.settleTimer = 0;
    return true;
  }

  public triggerSandboxCollapseImpulse(): void {
    if (!this.model || !this.data) return;
    this.sandboxCollapseActive = true;
    this.wakeTowerForInteraction();
    this.worldCompat.wakeAllActiveShards();
    let maxIntactLayer = 0;
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.assignedLayer > maxIntactLayer) {
        maxIntactLayer = b.assignedLayer;
      }
    }
    const isShortenedStack = maxIntactLayer > 0 && maxIntactLayer < 16;
    const shortBaseCut = Math.max(1, Math.floor(maxIntactLayer * 0.24));
    for (const b of this.blocks) {
      if (b.shattered) continue;
      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info) continue;
      const massKg = Math.max(0.01, this.model.body_mass[b.bodyHandle]);
      const invMassFroude = INV_FROUDE_VEL_SCALE / massKg;

      for (let k = 0; k < 6; k++) this.data.qvel[info.dofAdr + k] = 0;
      if (isShortenedStack) {
        if (b.assignedLayer <= shortBaseCut) {
          this.data.qvel[info.dofAdr + 0] = -0.44 * invMassFroude;
          this.data.qvel[info.dofAdr + 2] = -0.06 * invMassFroude;
        } else {
          const rel = b.assignedLayer / Math.max(1, maxIntactLayer);
          this.data.qvel[info.dofAdr + 0] = (0.15 + rel * 0.12) * invMassFroude;
          this.data.qvel[info.dofAdr + 2] = (b.assignedSlot - 1) * 0.055 * invMassFroude;
        }
      } else if (b.assignedLayer <= 2) {
        this.data.qvel[info.dofAdr + 0] = -0.32 * invMassFroude;
        this.data.qvel[info.dofAdr + 2] = -0.05 * invMassFroude;
      } else if (b.assignedLayer <= 9) {
        this.data.qvel[info.dofAdr + 0] = 0.11 * invMassFroude;
        this.data.qvel[info.dofAdr + 2] = (b.assignedSlot - 1) * 0.04 * invMassFroude;
      } else {
        const hScale = 0.085 + b.assignedLayer * 0.0042;
        this.data.qvel[info.dofAdr + 0] = hScale * invMassFroude;
        this.data.qvel[info.dofAdr + 2] = (b.assignedSlot - 1) * 0.055 * invMassFroude;
      }
    }
    this.audio.playGlassImpact(
      0.62,
      0.96,
      [0, 0.6, 0],
      this.listenerCamPos,
      this.listenerCamRight,
      {
        impactSpeedMps: 0.95,
        damage: 0,
        surface: 'glass',
      }
    );
  }

  private advanceCarriedBlockOnly(realDt: number): void {
    if (this.activeBlockId === null) return;
    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) return;
    const rb = this.worldCompat.getRigidBody(block.bodyHandle);
    if (!rb) return;

    this.extractionArcTimer = Math.min(0.22, this.extractionArcTimer + realDt);
    const u = this.extractionArcTimer / 0.22;
    const s = u * u * (3.0 - 2.0 * u);
    const outBow = Math.sin(Math.PI * u) * 0.34;

    const guideX =
      this.extractionStartPos[0] +
      (this.carriedTargetPos[0] - this.extractionStartPos[0]) * s +
      this.extractionOutDir[0] * outBow;
    const guideY =
      this.extractionStartPos[1] +
      (this.carriedTargetPos[1] - this.extractionStartPos[1]) * Math.pow(u, 0.85);
    const guideZ =
      this.extractionStartPos[2] +
      (this.carriedTargetPos[2] - this.extractionStartPos[2]) * s +
      this.extractionOutDir[2] * outBow;

    const pos = rb.translation();
    const rot = rb.rotation();
    const alpha = Math.min(1.0, realDt * SIM_TIME_SCALE * 11.5);
    const nxPos = pos.x + (guideX - pos.x) * alpha;
    const nyPos = pos.y + (guideY - pos.y) * alpha;
    const nzPos = pos.z + (guideZ - pos.z) * alpha;
    rb.setTranslation({ x: nxPos, y: nyPos, z: nzPos }, false);
    rb.setLinvel({ x: 0, y: 0, z: 0 }, false);
    rb.setAngvel({ x: 0, y: 0, z: 0 }, false);

    const tq = this.carriedTargetQuat;
    const dot = rot.x * tq[0] + rot.y * tq[1] + rot.z * tq[2] + rot.w * tq[3];
    const sign = dot < 0 ? -1 : 1;
    const blend = Math.min(1.0, 1.0 - Math.pow(1.0 - 0.22, realDt * 380.0));
    let qx = rot.x * (1 - blend) + tq[0] * sign * blend;
    let qy = rot.y * (1 - blend) + tq[1] * sign * blend;
    let qz = rot.z * (1 - blend) + tq[2] * sign * blend;
    let qw = rot.w * (1 - blend) + tq[3] * sign * blend;
    const qLen = Math.hypot(qx, qy, qz, qw) || 1;
    qx /= qLen; qy /= qLen; qz /= qLen; qw /= qLen;
    rb.setRotation({ x: qx, y: qy, z: qz, w: qw }, false);

    block.position[0] = nxPos;
    block.position[1] = nyPos;
    block.position[2] = nzPos;
    block.rotation[0] = qx;
    block.rotation[1] = qy;
    block.rotation[2] = qz;
    block.rotation[3] = qw;
    this.grabCurrentWorldOnBlock = [nxPos, nyPos, nzPos];
  }

  /**
   * Drives MuJoCo's kinematic `mocap` hand anchor + reads exact equality constraint reaction forces (`efc_force`)
   * and tangential/torsional frictional slip power during `PULLING` / `sandbox` manipulation!
   */
  private updateMocapHandAndExtractionSubstep(subDtSec: number, isLastSubstep: boolean = true): void {
    if (this.activeBlockId === null || !this.model || !this.data) {
      this.grabResistance01 = 0;
      return;
    }

    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) {
      this.data.eq_active[this.map.grabEqualityId] = 0;
      this.activeBlockId = null;
      this.grabResistance01 = 0;
      return;
    }

    const rb = this.worldCompat.getRigidBody(block.bodyHandle);
    const col = this.worldCompat.getCollider(block.colliderHandle);
    if (!rb || !col) return;

    const pos = rb.translation();
    const rot = rb.rotation();
    const q: Quat = [rot.x, rot.y, rot.z, rot.w];

    if (!this.isPointerDraggingBlock && this.turnPhase !== 'EXTRACTED') {
      this.data.eq_active[this.map.grabEqualityId] = 0;
      this.activeBlockId = null;
      this.grabResistance01 = 0;
      return;
    }

    if (this.turnPhase === 'PULLING' || (this.gameMode === 'sandbox' && !block.extracted)) {
      const worldOffsetDm = quatRotateVec3(q, this.grabLocalPoint);
      const grabWorldDm: Vec3 = [
        pos.x + worldOffsetDm[0],
        pos.y + worldOffsetDm[1],
        pos.z + worldOffsetDm[2],
      ];
      this.grabCurrentWorldOnBlock = grabWorldDm;

      const radialDistDm = Math.hypot(pos.x, pos.z);
      const insideFactor = Math.max(0.0, Math.min(1.0, 1.0 - (radialDistDm - 0.35) / 0.45));

      const sandboxOutsideTower =
        this.gameMode === 'sandbox' &&
        (radialDistDm > BLOCK_LENGTH * 0.98 ||
          Math.abs(grabWorldDm[1] - this.grabPlaneY) > BLOCK_HEIGHT * 0.65);
      if (sandboxOutsideTower) {
        const trackAlpha = Math.min(1.0, subDtSec * 85.0);
        this.grabPlaneY += (grabWorldDm[1] - this.grabPlaneY) * trackAlpha;
        this.grabWorldAnchor[1] = this.grabPlaneY;
      }

      // Project raw pointer displacement preferentially along the block's sliding slot axis while inside the stack
      const errXDm = this.grabWorldAnchor[0] - grabWorldDm[0];
      const errZDm = this.grabWorldAnchor[2] - grabWorldDm[2];
      const axisX = quatRotateVec3(q, [1, 0, 0]);
      const axisZ = quatRotateVec3(q, [0, 0, 1]);
      const unassistedPull = this.fidelity.unassistedExtraction;
      const alongLen = errXDm * axisX[0] + errZDm * axisX[2];
      const acrossWid = errXDm * axisZ[0] + errZDm * axisZ[2];
      const lateralScale = unassistedPull
        ? 1.0 - 0.58 * insideFactor
        : this.gameMode === 'sandbox'
          ? 0.85
          : 1.0 - 0.78 * insideFactor;

      let effErrXDm = axisX[0] * alongLen + axisZ[0] * acrossWid * lateralScale;
      let effErrZDm = axisX[2] * alongLen + axisZ[2] * acrossWid * lateralScale;

      // Soft force-limiting lead cap on the mocap target so human fingers cannot exert infinite force on a pinned column!
      this.grabRampSec = Math.min(0.065, this.grabRampSec + subDtSec);
      const rampU = this.grabRampSec / 0.065;
      const grabRamp = 0.35 + 0.65 * (rampU * rampU * (3.0 - 2.0 * rampU));
      const pinchLoadN = block.id < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[block.id] : 0;
      // Count surviving support blocks on the same row (`assignedLayer`) still inside the stack:
      // - On a full row (2+ bearing blocks, e.g. T5 part 1 gentle pull of a pinned column), human two-finger grip
      //   is capped below the twin-column static friction ceiling so the tower remains vibration-free (< 20 mm/s).
      // - On a depleted single-support row (`sameRowOtherCount === 0`, e.g. T5 part 2 after moving the sibling block),
      //   the player can decisively pull out the sole remaining support column (`maxLeadDm = 0.145`) to topple the tower!
      let sameRowOtherCount = 0;
      for (let k = 0; k < this.blocks.length; k++) {
        const ob = this.blocks[k];
        if (
          ob.id !== block.id &&
          !ob.shattered &&
          !ob.extracted &&
          ob.assignedLayer === block.assignedLayer &&
          Math.hypot(ob.position[0], ob.position[2]) < 0.72
        ) {
          sameRowOtherCount++;
        }
      }
      // Distinguish full-row multi-column stiction (`sameRowOtherCount >= 2`, where human two-finger pinch on smooth
      // beveled glass stalls rather than dragging the entire upper tower sideways) vs. a single-sibling depleted row
      // (`sameRowOtherCount === 1`, where a gentle < 0.44 dm test-pull stays quiet while a decisive yank topples!).
      const rawFingerLeadDm = Math.hypot(effErrXDm, effErrZDm);
      const isPinnedFullRow =
        this.gameMode !== 'sandbox' &&
        !block.isLooseSlot &&
        insideFactor > 0.30 &&
        pinchLoadN > 4.5 &&
        sameRowOtherCount >= 2;
      const isGentlePinnedDepletedRow =
        this.gameMode !== 'sandbox' &&
        !block.isLooseSlot &&
        insideFactor > 0.35 &&
        pinchLoadN > 3.5 &&
        sameRowOtherCount === 1 &&
        rawFingerLeadDm < 0.45;
      const maxLeadDm =
        (this.gameMode === 'sandbox'
          ? 0.28
          : isPinnedFullRow || isGentlePinnedDepletedRow
            ? Math.max(0.012, 0.088 / (1.0 + (pinchLoadN - 2.2) * 0.55))
            : 0.148) * grabRamp;
      const errMagDm = Math.hypot(effErrXDm, effErrZDm);
      if (errMagDm > maxLeadDm) {
        const sc = maxLeadDm / errMagDm;
        effErrXDm *= sc;
        effErrZDm *= sc;
      }

      const targetMocapXM = (grabWorldDm[0] + effErrXDm) * M_PER_DM;
      // Hold horizontal extraction plane height `grabPlaneY` unless `unassistedExtraction` is enabled,
      // where smooth finger vertical compliance allows gravitational overhanging pitch once partly outside the slot
      // without unstable mocap velocity amplification or dropping into lower courses after exit.
      const supCenter = this.getLayerSupportCenter(block.assignedLayer);
      const outwardAlongDm = Math.abs(
        (pos.x - supCenter[0]) * axisX[0] + (pos.z - supCenter[1]) * axisX[2]
      );
      let desiredPlaneY: number;
      if (unassistedPull && this.gameMode !== 'sandbox') {
        const overhangLip01 = Math.max(0.0, Math.min(1.0, (outwardAlongDm - 0.62) / 0.10));
        const clearCarry01 = Math.max(0.0, Math.min(1.0, (outwardAlongDm - 0.75) / 0.10));
        const sagDm =
          overhangLip01 * (0.118 * (1.0 - clearCarry01) + 0.035 * clearCarry01);
        const compliantTargetY = Math.max(BLOCK_HALF_Y, this.grabPlaneY - sagDm);
        const blendToTarget = Math.max(overhangLip01, clearCarry01);
        desiredPlaneY =
          grabWorldDm[1] * (1.0 - blendToTarget) + compliantTargetY * blendToTarget;
      } else {
        const overhangBlend =
          this.gameMode === 'sandbox'
            ? 0.0
            : Math.max(0.0, Math.min(1.0, (radialDistDm - 0.20) / 0.22));
        desiredPlaneY =
          grabWorldDm[1] * (1.0 - overhangBlend) + this.grabPlaneY * overhangBlend;
      }
      const targetMocapYM =
        (sandboxOutsideTower
          ? Math.min(this.grabPlaneY, Math.max(BLOCK_HALF_Y, grabWorldDm[1]))
          : desiredPlaneY) * M_PER_DM;
      const targetMocapZM = (grabWorldDm[2] + effErrZDm) * M_PER_DM;

      this.data.mocap_pos[0] = targetMocapXM;
      this.data.mocap_pos[1] = targetMocapYM;
      this.data.mocap_pos[2] = targetMocapZM;
      if (sandboxOutsideTower) {
        this.data.mocap_pos[1] = pos.y * M_PER_DM;
      }
      this.data.eq_active[this.map.grabEqualityId] = 1;
      this.worldCompat.wakeBody(block.bodyHandle);

      // Gentle roll/pitch stabilization torque around horizontal axes during human finger pinch
      // (suppressed under `unassistedExtraction` once overhanging so gravity pivots the prism naturally)
      const av = rb.angvel();
      const info = this.worldCompat.getMapping(block.bodyHandle);
      if (info) {
        const wLocalX = this.data.qvel[info.dofAdr + 3];
        const wLocalZ = this.data.qvel[info.dofAdr + 5];
        const damp = !unassistedPull
          ? 0.94
          : outwardAlongDm < 0.60
            ? 0.94
            : outwardAlongDm > 0.76
              ? 0.975
              : 0.999;
        this.data.qvel[info.dofAdr + 3] = wLocalX * damp;
        this.data.qvel[info.dofAdr + 5] = wLocalZ * damp;
      }

      // Read exact implicit 3D equality pull force from efc_force[0..2] (zero Embind heap allocation!)
      let pullForceN = 0;
      if (this.data.ne >= 3) {
        pullForceN = Math.hypot(
          this.data.efc_force[0],
          this.data.efc_force[1],
          this.data.efc_force[2]
        );
      }

      let geomClear = false;
      let relX = 0;
      let relZ = 0;
      if (this.gameMode !== 'sandbox') {
        const refCenter = this.getLayerSupportCenter(block.assignedLayer);
        relX = pos.x - refCenter[0];
        relZ = pos.z - refCenter[1];
        const alongClear = Math.abs(relX * axisX[0] + relZ * axisX[2]);
        const clearDist = Math.hypot(relX, relZ);
        geomClear = alongClear > BLOCK_LENGTH * 0.98 || clearDist > BLOCK_LENGTH * 1.01;
      }

      let slipPowerMw = 0;
      let touchingCount = 0;
      if (isLastSubstep || geomClear) {
        const contacts = this.data.contact;
        const ncon = this.data.ncon;
        const geomBodyId = this.model.geom_bodyid;
        const efcForce = this.data.efc_force;
        const efcVel = this.data.efc_vel;
        for (let i = 0; i < ncon; i++) {
          const c = contacts.get(i);
          if (!c) continue;
          const g1 = c.geom[0];
          const g2 = c.geom[1];
          const b1 = geomBodyId[g1];
          const b2 = geomBodyId[g2];
          if (b1 === block.bodyHandle || b2 === block.bodyHandle) {
            const isStaticSurface =
              g1 === this.map.tableGeomId ||
              g2 === this.map.tableGeomId ||
              g1 === this.map.parkingShelfGeomId ||
              g2 === this.map.parkingShelfGeomId;
            if (!isStaticSurface && c.dist < 0.002) touchingCount++;
            if (isLastSubstep && c.efc_address >= 0 && c.dim >= 3) {
              const adr = c.efc_address;
              const pTangential =
                Math.abs(efcForce[adr + 1] * efcVel[adr + 1]) +
                Math.abs(efcForce[adr + 2] * efcVel[adr + 2]);
              const pTorsional =
                c.dim >= 4 ? Math.abs(efcForce[adr + 3] * efcVel[adr + 3]) : 0;
              slipPowerMw += (pTangential + pTorsional) * 1000.0;
            }
          }
          c.delete();
        }
        contacts.delete();
      }

      const lv = rb.linvel();
      const speedHorizReal = Math.hypot(lv.x, lv.z) * SIM_TIME_SCALE;

      // Stribeck static-to-kinetic friction transition on the actively drawn glass block:
      // Once horizontal sliding speed crosses breakaway (`0.15 -> 0.75 dm/s` = `1.5 -> 7.5 cm/s`),
      // elevate `geom_priority[gid] = 1` and blend friction from static (`0.275`) down to kinetic (`0.125`)
      // so the sliding prism glides cleanly out from under stationary upper courses without rubber-dragging them!
      if (this.gameMode !== 'sandbox') {
        const gid = block.colliderHandle;
        if (this.activeSlidingGeomId !== null && this.activeSlidingGeomId !== gid) {
          this.resetActiveSlidingGeomFriction();
        }
        const kinBlend = Math.max(0.0, Math.min(1.0, (speedHorizReal - 0.14) / 0.58));
        if (kinBlend > 0.02) {
          this.activeSlidingGeomId = gid;
          // Under `unassistedExtraction`, preserve real dry soda-lime kinetic sliding friction (0.165 vs legacy 0.125)
          // using priority=1 so neighbor blocks (0.275) do not lock static friction during sustained slip.
          this.model.geom_priority[gid] = 1;
          const g3 = gid * 3;
          const sf = this.cfg.glassFriction;
          const kf = this.cfg.glassKineticFriction;
          const targetSlideKf0 = kf[0];
          const targetSlideKf1 = kf[1];
          const targetSlideKf2 = kf[2];
          this.model.geom_friction[g3 + 0] = sf[0] + (targetSlideKf0 - sf[0]) * kinBlend;
          this.model.geom_friction[g3 + 1] = sf[1] + (targetSlideKf1 - sf[1]) * kinBlend;
          this.model.geom_friction[g3 + 2] = sf[2] + (targetSlideKf2 - sf[2]) * kinBlend;
          block.currentFriction = this.model.geom_friction[g3 + 0];
        } else if (this.activeSlidingGeomId === gid) {
          this.resetActiveSlidingGeomFriction();
          block.currentFriction = this.cfg.glassFriction[0];
        }
      }

      const maxForceRef = this.gameMode === 'sandbox' ? this.cfg.maxGrabForceSandboxN : this.cfg.maxGrabForceSoloN;
      const rawRes = Math.min(1.0, pullForceN / (maxForceRef * 0.72));
      this.grabResistance01 = this.grabResistance01 * 0.85 + rawRes * 0.15;
      if (isLastSubstep) {
        const clampingLoadN = this.cachedClampingLoadN[block.id] ?? 0;
        this.audio.updateSlideFriction(
          speedHorizReal,
          this.grabResistance01,
          slipPowerMw,
          av.y * FROUDE_VEL_SCALE,
          clampingLoadN
        );
      }

      // Geometric clearance + consecutive zero-contact substeps -> EXTRACTED
      if (this.gameMode !== 'sandbox') {
        if (geomClear && touchingCount === 0) {
          this.clearContactSubsteps++;
        } else {
          this.clearContactSubsteps = 0;
        }

        const requiredClearSubsteps = 3;
        if (this.clearContactSubsteps >= requiredClearSubsteps) {
          this.resetActiveSlidingGeomFriction();
          block.currentFriction = this.cfg.glassFriction[0];
          block.extracted = true;
          this.turnPhase = 'EXTRACTED';
          this.data.eq_active[this.map.grabEqualityId] = 0;
          // Ensure remaining tower courses stay awake to settle or topple immediately once this support clears!
          this.worldCompat.wakeActiveTowerBlocks();
          this.carriedFromDrag = this.isPointerDraggingBlock;
          this.grabResistance01 = 0;
          this.extractionArcTimer = 0;
          this.extractionStartPos = [pos.x, pos.y, pos.z];
          const outLen = Math.hypot(relX, relZ) || 1;
          this.extractionOutDir = [relX / outLen, 0, relZ / outLen];

          this.audio.updateSlideFriction(0, 0, 0, 0, 0);
          if (!unassistedPull) {
            this.phantomAirClinkCount++;
            this.audio.playGlassImpact(
              0.24,
              1.25,
              [pos.x, pos.y, pos.z],
              this.listenerCamPos,
              this.listenerCamRight,
              {
                impactSpeedMps: Math.max(0.04, speedHorizReal * M_PER_DM),
                damage: block.damage,
                surface: 'glass',
              }
            );
          }

          col.setSensor(true);
          rb.setGravityScale(0.0, true);
          rb.setLinvel({ x: 0, y: 0, z: 0 }, false);
          rb.setAngvel({ x: 0, y: 0, z: 0 }, false);

          this.carriedYawOffsetRad = 0;
          this.hasCustomCarriedHover = false;
          this.lastHoverTargetXZ = null;
          this.carriedHandVelXZ = [0, 0];
          this.computeOpenTopSlots();
          if (this.openTopSlots.length > 0) {
            const defSlot = this.openTopSlots[Math.min(1, this.openTopSlots.length - 1)];
            this.hoveredSlotIndex = defSlot.slotIndex;
            const frame = this.topSupportFrame;
            const dX = defSlot.position[0] - frame.layerCenter[0];
            const dY = defSlot.position[1] - frame.layerCenter[1];
            const dZ = defSlot.position[2] - frame.layerCenter[2];
            const defV = dX * frame.lengthAxis[0] + dY * frame.lengthAxis[1] + dZ * frame.lengthAxis[2];
            const defU = dX * frame.spacingAxis[0] + dY * frame.spacingAxis[1] + dZ * frame.spacingAxis[2];
            this.carriedRawLocalVL = [defV, defU];
            this.resolveCarriedPlacementPose(defV, defU);
          }
        }
      }
    } else if (this.turnPhase === 'EXTRACTED') {
      this.advanceCarriedBlockOnly(subDtSec);
    }
  }

  private snapshotPreStepVelocities(): number {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return 0;
    const qpos = d.qpos;
    const qvel = d.qvel;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const bodyMass = m.body_mass;

    let maxBoundReal = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      this.ensureBlockRegistered(b);
      if (b.shattered) continue;
      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info || !info.enabled) continue;
      const adr = info.qposAdr;
      b.prePos[0] = qpos[adr + 0] * DM_PER_M;
      b.prePos[1] = qpos[adr + 1] * DM_PER_M;
      b.prePos[2] = qpos[adr + 2] * DM_PER_M;
      b.preQuat[0] = qpos[adr + 4];
      b.preQuat[1] = qpos[adr + 5];
      b.preQuat[2] = qpos[adr + 6];
      b.preQuat[3] = qpos[adr + 3];

      if (b.id < this.blockPreSubstepLinSpeedSi.length) {
        const dofPre = info.dofAdr;
        if (info.settledAsleep) {
          this.blockPreSubstepLinSpeedSi[b.id] = 0;
          this.blockPreSubstepAngSpeedSi[b.id] = 0;
        } else {
          this.blockPreSubstepLinSpeedSi[b.id] = Math.hypot(
            qvel[dofPre + 0],
            qvel[dofPre + 1],
            qvel[dofPre + 2]
          );
          this.blockPreSubstepAngSpeedSi[b.id] = Math.hypot(
            qvel[dofPre + 3],
            qvel[dofPre + 4],
            qvel[dofPre + 5]
          );
        }
      }

      const treeId = bodyTreeId[b.bodyHandle];
      if (info.settledAsleep || (treeId >= 0 && treeAsleep[treeId] >= 0)) {
        b.linVel[0] = 0; b.linVel[1] = 0; b.linVel[2] = 0;
        b.angVel[0] = 0; b.angVel[1] = 0; b.angVel[2] = 0;
        if (b.id < this.blockCoherentArcSubsteps.length) {
          this.blockCoherentArcSubsteps[b.id] = 0;
          this.blockPrevHorizOmegaX[b.id] = 0;
          this.blockPrevHorizOmegaZ[b.id] = 0;
        }
        continue;
      }

      const dof = info.dofAdr;
      const lvx = qvel[dof + 0] * FROUDE_VEL_SCALE;
      const lvy = qvel[dof + 1] * FROUDE_VEL_SCALE;
      const lvz = qvel[dof + 2] * FROUDE_VEL_SCALE;
      const wWorldSi = quatRotateVec3(b.preQuat, [qvel[dof + 3], qvel[dof + 4], qvel[dof + 5]]);
      const avx = wWorldSi[0] * INV_FROUDE_VEL_SCALE;
      const avy = wWorldSi[1] * INV_FROUDE_VEL_SCALE;
      const avz = wWorldSi[2] * INV_FROUDE_VEL_SCALE;
      b.linVel[0] = lvx; b.linVel[1] = lvy; b.linVel[2] = lvz;
      b.angVel[0] = avx; b.angVel[1] = avy; b.angVel[2] = avz;

      if (b.id < this.blockCoherentArcSubsteps.length) {
        const whSi = Math.hypot(wWorldSi[0], wWorldSi[2]);
        if (whSi >= 0.14) {
          const prevWx = this.blockPrevHorizOmegaX[b.id];
          const prevWz = this.blockPrevHorizOmegaZ[b.id];
          const prevWh = Math.hypot(prevWx, prevWz);
          if (
            prevWh >= 0.10 &&
            wWorldSi[0] * prevWx + wWorldSi[2] * prevWz >= 0.45 * whSi * prevWh
          ) {
            this.blockCoherentArcSubsteps[b.id] = Math.min(
              65535,
              this.blockCoherentArcSubsteps[b.id] + 1
            );
          } else {
            this.blockCoherentArcSubsteps[b.id] = 1;
          }
          this.blockPrevHorizOmegaX[b.id] = wWorldSi[0];
          this.blockPrevHorizOmegaZ[b.id] = wWorldSi[2];
        } else if (whSi < 0.05) {
          this.blockCoherentArcSubsteps[b.id] = 0;
          this.blockPrevHorizOmegaX[b.id] = 0;
          this.blockPrevHorizOmegaZ[b.id] = 0;
        }
      }

      if (!b.extracted) {
        const bound = (Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) * 0.42) * SIM_TIME_SCALE;
        if (bound > maxBoundReal) maxBoundReal = bound;
      }
    }

    const bodyInertia = m.body_inertia;
    const yDatumM = -0.35;
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const info = this.worldCompat.getMapping(s.bodyHandle);
      const treeId = bodyTreeId[s.bodyHandle];
      const asleep = !info || !info.enabled || info.settledAsleep || (treeId >= 0 && treeAsleep[treeId] >= 0);
      if (!asleep) {
        const adr = this.map.shardQposAdr[s.slotIndex];
        const mShd = bodyMass[s.bodyHandle];
        const pePreJ = mShd > 0 ? mShd * 9.81 * (Math.max(yDatumM, qpos[adr + 1]) - yDatumM) : 0;
        const dof = this.map.shardDofAdr[s.slotIndex];
        s.prePos[0] = qpos[adr + 0] * DM_PER_M;
        s.prePos[1] = qpos[adr + 1] * DM_PER_M;
        s.prePos[2] = qpos[adr + 2] * DM_PER_M;
        s.preQuat[0] = qpos[adr + 4];
        s.preQuat[1] = qpos[adr + 5];
        s.preQuat[2] = qpos[adr + 6];
        s.preQuat[3] = qpos[adr + 3];
        const vxSi = qvel[dof + 0];
        const vySi = qvel[dof + 1];
        const vzSi = qvel[dof + 2];
        const wxSi = qvel[dof + 3];
        const wySi = qvel[dof + 4];
        const wzSi = qvel[dof + 5];
        const ib = s.bodyHandle * 3;
        const kePreJ =
          mShd > 0
            ? 0.5 * mShd * (vxSi * vxSi + vySi * vySi + vzSi * vzSi) +
              0.5 *
                (bodyInertia[ib + 0] * wxSi * wxSi +
                  bodyInertia[ib + 1] * wySi * wySi +
                  bodyInertia[ib + 2] * wzSi * wzSi)
            : 0;
        const ePreTotalJ = pePreJ + kePreJ;
        this.shardPreSubstepTotalEnergyJ[s.slotIndex] = ePreTotalJ;
        if (
          this.shardInContactPrevSubstep[s.slotIndex] === 0 ||
          this.shardEntryTotalEnergyJ[s.slotIndex] <= 0
        ) {
          this.shardEntryTotalEnergyJ[s.slotIndex] = ePreTotalJ;
        }
        const lvx = vxSi * FROUDE_VEL_SCALE;
        const lvy = vySi * FROUDE_VEL_SCALE;
        const lvz = vzSi * FROUDE_VEL_SCALE;
        const wWorldSi = quatRotateVec3(s.preQuat, [wxSi, wySi, wzSi]);
        const avx = wWorldSi[0] * INV_FROUDE_VEL_SCALE;
        const avy = wWorldSi[1] * INV_FROUDE_VEL_SCALE;
        const avz = wWorldSi[2] * INV_FROUDE_VEL_SCALE;
        s.preLinVel[0] = lvx; s.preLinVel[1] = lvy; s.preLinVel[2] = lvz;
        s.preAngVel[0] = avx; s.preAngVel[1] = avy; s.preAngVel[2] = avz;
        s.preVy = lvy;
        const sb = (Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) * 0.25) * SIM_TIME_SCALE;
        if (sb > maxBoundReal) maxBoundReal = sb;
      } else {
        this.shardInContactPrevSubstep[s.slotIndex] = 0;
        s.prePos[0] = s.position[0]; s.prePos[1] = s.position[1]; s.prePos[2] = s.position[2];
        s.preQuat[0] = s.rotation[0]; s.preQuat[1] = s.rotation[1]; s.preQuat[2] = s.rotation[2]; s.preQuat[3] = s.rotation[3];
        s.preLinVel[0] = 0; s.preLinVel[1] = 0; s.preLinVel[2] = 0;
        s.preAngVel[0] = 0; s.preAngVel[1] = 0; s.preAngVel[2] = 0;
        s.preVy = 0;
      }
    }

    this.extraPreStep.clear();
    for (const extraBid of this.worldCompat.getActiveExtraBodyIds()) {
      if (this.bodyToBlock.has(extraBid)) continue;
      const info = this.worldCompat.getMapping(extraBid);
      if (!info) continue;
      const adr = info.qposAdr;
      const dof = info.dofAdr;
      const px = qpos[adr + 0] * DM_PER_M;
      const py = qpos[adr + 1] * DM_PER_M;
      const pz = qpos[adr + 2] * DM_PER_M;
      const lvx = qvel[dof + 0] * FROUDE_VEL_SCALE;
      const lvy = qvel[dof + 1] * FROUDE_VEL_SCALE;
      const lvz = qvel[dof + 2] * FROUDE_VEL_SCALE;
      const eq: Quat = [qpos[adr + 4], qpos[adr + 5], qpos[adr + 6], qpos[adr + 3]];
      const wWorldSi = quatRotateVec3(eq, [qvel[dof + 3], qvel[dof + 4], qvel[dof + 5]]);
      const avx = wWorldSi[0] * INV_FROUDE_VEL_SCALE;
      const avy = wWorldSi[1] * INV_FROUDE_VEL_SCALE;
      const avz = wWorldSi[2] * INV_FROUDE_VEL_SCALE;
      this.extraPreStep.set(extraBid, {
        posDm: [px, py, pz],
        linVelSim: [lvx, lvy, lvz],
        angVelSim: [avx, avy, avz],
        massKg: bodyMass[extraBid],
      });
      const eb = (Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) * 0.3) * SIM_TIME_SCALE;
      if (eb > maxBoundReal) maxBoundReal = eb;
    }

    return maxBoundReal;
  }

  private getPreStepPointVelSim(bodyId: number, worldPtDm: Vec3): Vec3 {
    if (bodyId < 2) return [0, 0, 0]; // World / oak table / mocap
    const blk = this.bodyToBlock.get(bodyId);
    if (blk) {
      const rx = worldPtDm[0] - blk.prePos[0];
      const ry = worldPtDm[1] - blk.prePos[1];
      const rz = worldPtDm[2] - blk.prePos[2];
      return [
        blk.linVel[0] + (blk.angVel[1] * rz - blk.angVel[2] * ry),
        blk.linVel[1] + (blk.angVel[2] * rx - blk.angVel[0] * rz),
        blk.linVel[2] + (blk.angVel[0] * ry - blk.angVel[1] * rx),
      ];
    }
    const shd = this.bodyToShard.get(bodyId);
    if (shd) {
      const rx = worldPtDm[0] - shd.prePos[0];
      const ry = worldPtDm[1] - shd.prePos[1];
      const rz = worldPtDm[2] - shd.prePos[2];
      return [
        shd.preLinVel[0] + (shd.preAngVel[1] * rz - shd.preAngVel[2] * ry),
        shd.preLinVel[1] + (shd.preAngVel[2] * rx - shd.preAngVel[0] * rz),
        shd.preLinVel[2] + (shd.preAngVel[0] * ry - shd.preAngVel[1] * rx),
      ];
    }
    const ext = this.extraPreStep.get(bodyId);
    if (ext) {
      const rx = worldPtDm[0] - ext.posDm[0];
      const ry = worldPtDm[1] - ext.posDm[1];
      const rz = worldPtDm[2] - ext.posDm[2];
      return [
        ext.linVelSim[0] + (ext.angVelSim[1] * rz - ext.angVelSim[2] * ry),
        ext.linVelSim[1] + (ext.angVelSim[2] * rx - ext.angVelSim[0] * rz),
        ext.linVelSim[2] + (ext.angVelSim[0] * ry - ext.angVelSim[1] * rx),
      ];
    }
    return [0, 0, 0];
  }

  /**
   * Evaluates 6D contact wrenches (`efc_force`), reduced-mass impact strain energy, opposing-wrench anvil crush,
   * three-point flexural bending stress, Housner beveled-prism rocking transfers, and secondary 3D shard bounces!
   */
  private processWrenchCollisionsAndFractures(subDtSec: number): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return;

    let didAnyFractureSyncNeeded = false;
    const maxShattersPerFrame = this.shards.length >= 160 ? 2 : 5;
    if (this.deferredFractures.length > 0 && this.frameShatterCount < maxShattersPerFrame) {
      while (
        this.deferredFractures.length > 0 &&
        this.frameShatterCount < maxShattersPerFrame
      ) {
        const def = this.deferredFractures.shift()!;
        if (!def.blk.shattered) {
          const rbDef = this.worldCompat.getRigidBody(def.blk.bodyHandle);
          const curLv = rbDef ? rbDef.linvel() : { x: 0, y: 0, z: 0 };
          const curSpeedReal = Math.hypot(curLv.x, curLv.y, curLv.z) * SIM_TIME_SCALE;
          const effClosingReal = def.hitTable
            ? Math.min(def.closingSpeedReal, Math.max(8.5, curSpeedReal * 1.25))
            : Math.min(def.closingSpeedReal, Math.max(2.5, curSpeedReal * 1.15));
          this.shatterBlock(
            def.blk,
            def.localHit,
            def.impactSpeed,
            def.worldNormal,
            def.hitTable,
            {
              closingSpeedReal: effClosingReal,
              isAnvilCrush: def.isAnvilCrush,
              skipSyncKinematics: true,
            }
          );
          didAnyFractureSyncNeeded = true;
          this.frameShatterCount++;
        }
      }
    }

    const ncon = d.ncon;
    if (ncon === 0) {
      if (this.shards.length > 0) {
        const bodyTreeId = m.body_treeid;
        const treeAsleep = d.tree_asleep;
        for (let i = 0; i < this.shards.length; i++) {
          const s = this.shards[i];
          const treeId = bodyTreeId[s.bodyHandle];
          if (treeId < 0 || treeAsleep[treeId] < 0) {
            this.shardAirborneSubsteps[s.slotIndex] = Math.min(
              65535,
              this.shardAirborneSubsteps[s.slotIndex] + 1
            );
          }
        }
      }
      if (didAnyFractureSyncNeeded) {
        this.worldCompat.syncKinematics();
      }
      return;
    }

    const blockImpacts = this.scratchBlockImpacts;
    blockImpacts.clear();

    const contacts = d.contact;
    const efc = d.efc_force;
    const bodyMass = m.body_mass;
    const geomBodyId = m.geom_bodyid;
    const qpos = d.qpos;
    const qvel = d.qvel;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const dtSim = subDtSec * SIM_TIME_SCALE;
    const minShardGid = this.minShardGeomId;
    const maxShardGid = this.maxShardGeomId;
    const parkingGid = this.map.parkingShelfGeomId;
    const tableGid = this.map.tableGeomId;
    const shardTableGid = this.map.shardTableGeomId;

    if (this.shards.length > 0) {
      this.shardHadContactSubstep.fill(0);
    }

    const recordForBlock = (
      blkA: GlassBlockState,
      otherBodyId: number,
      otherGeomId: number,
      worldPtDm: Vec3,
      nAtoB: Vec3,
      fnN: number,
      jnSim: number,
      pairSeparated: boolean,
      pairSeparatedRaw: boolean
    ) => {
      if (blkA.shattered || blkA.extracted) return;
      const vPtA = this.getPreStepPointVelSim(blkA.bodyHandle, worldPtDm);
      const vPtB = this.getPreStepPointVelSim(otherBodyId, worldPtDm);
      const relAlongN =
        (vPtB[0] - vPtA[0]) * nAtoB[0] +
        (vPtB[1] - vPtA[1]) * nAtoB[1] +
        (vPtB[2] - vPtA[2]) * nAtoB[2];
      const vCloseReal = Math.max(0.0, -relAlongN) * SIM_TIME_SCALE;

      // Exact Newtonian two-body reduced-mass upper bound on normal impulse:
      // J_max = (1 + e) * (mA * mB / (mA + mB)) * v_close_sim (with mB = Infinity for static oak table!).
      // Prevents a 1.2g splinter whose bounce completes in < 0.6 substeps from being over-integrated by tauImpactSim.
      const mA = Math.max(0.001, bodyMass[blkA.bodyHandle]);
      const mOther = otherBodyId < 2 ? 0 : Math.max(0.0001, bodyMass[otherBodyId]);
      const mRed = otherBodyId < 2 ? mA : (mA * mOther) / (mA + mOther);
      const jMaxPhysicalSim =
        vCloseReal > 0.05 ? 1.18 * mRed * (vCloseReal / SIM_TIME_SCALE) : jnSim;
      const clampedJnSim = vCloseReal > 1.5 ? Math.min(jnSim, jMaxPhysicalSim) : jnSim;

      const relWorldDm: Vec3 = [
        worldPtDm[0] - blkA.prePos[0],
        worldPtDm[1] - blkA.prePos[1],
        worldPtDm[2] - blkA.prePos[2],
      ];
      const rawLocal = quatInverseRotateVec3(blkA.preQuat, relWorldDm);
      const localHitDm: Vec3 = [
        Math.max(-BLOCK_HALF_X, Math.min(BLOCK_HALF_X, rawLocal[0])),
        Math.max(-BLOCK_HALF_Y, Math.min(BLOCK_HALF_Y, rawLocal[1])),
        Math.max(-BLOCK_HALF_Z, Math.min(BLOCK_HALF_Z, rawLocal[2])),
      ];
      const hitTable = otherGeomId === tableGid || otherGeomId === shardTableGid;
      const normOtherBodyId = hitTable ? 0 : otherBodyId;

      let acc = blockImpacts.get(blkA.id);
      if (!acc) {
        acc = {
          blk: blkA,
          dynamicImpulseSim: 0,
          peakDynamicImpulseSim: -1,
          maxVCloseReal: 0,
          bestLocalHitDm: localHitDm,
          bestNormalAtoB: nAtoB,
          bestHitTable: hitTable,
          maxRawVCloseReal: 0,
          bestRawLocalHitDm: localHitDm,
          bestRawNormalAtoB: nAtoB,
          bestRawHitTable: hitTable,
          bestRawWorldPtDm: worldPtDm,
          bestRawOtherBodyId: normOtherBodyId,
          hasRawSeparatedOnset: false,
          upwardSupportFn: 0,
          downwardTopFn: 0,
          rockHitTable: hitTable,
          rockWorldPtDm: worldPtDm,
          distinctContactBodies: 0,
          hasOnsetContact: false,
          contacts: [],
        };
        blockImpacts.set(blkA.id, acc);
      }

      let seenOtherBody = false;
      for (let k = 0; k < acc.contacts.length; k++) {
        if (acc.contacts[k].otherBodyId === normOtherBodyId) {
          seenOtherBody = true;
          break;
        }
      }
      if (!seenOtherBody) {
        acc.distinctContactBodies++;
      }

      if (nAtoB[1] < -0.75) {
        acc.upwardSupportFn += fnN;
        if (hitTable) acc.rockHitTable = true;
        acc.rockWorldPtDm = worldPtDm;
      } else if (nAtoB[1] > 0.75) {
        acc.downwardTopFn += fnN;
      }

      // Mass-ratio-weighted effective closing speed so a 5g upward-bouncing shard grazing a 71.1g block
      // on the same substep the block hits the table cannot inflate `acc.maxVCloseReal`!
      const vCloseForBlock =
        mOther > 0 && mOther < 0.35 * mA ? vCloseReal * ((2.0 * mRed) / mA) : vCloseReal;

      if (pairSeparatedRaw && vCloseForBlock >= 0.24) {
        acc.hasRawSeparatedOnset = true;
      }

      if (
        (pairSeparated && vCloseForBlock >= 0.85) ||
        vCloseForBlock >= CRAZE_IMPACT_SPEED * 0.85
      ) {
        acc.hasOnsetContact = true;
      }

      const isShardContact = otherGeomId >= minShardGid && otherGeomId <= maxShardGid;
      acc.contacts.push({
        jnSim: clampedJnSim,
        fnN,
        localHitDm,
        nAtoB,
        vCloseReal: vCloseForBlock,
        otherBodyId: normOtherBodyId,
        otherIsShard: isShardContact,
        hitTable,
        worldPtDm,
      });
      if (vCloseForBlock > acc.maxRawVCloseReal) {
        acc.maxRawVCloseReal = vCloseForBlock;
        acc.bestRawLocalHitDm = localHitDm;
        acc.bestRawNormalAtoB = nAtoB;
        acc.bestRawHitTable = hitTable;
        acc.bestRawWorldPtDm = worldPtDm;
        acc.bestRawOtherBodyId = normOtherBodyId;
      }
      if (!isShardContact && vCloseForBlock > 1.5) {
        acc.dynamicImpulseSim += clampedJnSim;
        if (vCloseForBlock > acc.maxVCloseReal) acc.maxVCloseReal = vCloseForBlock;
        if (clampedJnSim > acc.peakDynamicImpulseSim) {
          acc.peakDynamicImpulseSim = clampedJnSim;
          acc.bestLocalHitDm = localHitDm;
          acc.bestNormalAtoB = nAtoB;
          acc.bestHitTable = hitTable;
        }
      }
    };

    // First coalesce multi-point contacts (`multiccd` generates 2-4 vertices per geom pair) by `(g1, g2)` pair
    // using a zero-GC pooled map and single Embind view reads per contact!
    const pairMap = this.scratchPairMap;
    pairMap.clear();
    let poolUsed = 0;

    for (let i = 0; i < ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      const adr = c.efc_address;
      if (adr >= 0) {
        const fnN = efc[adr];
        if (fnN > 0.004) {
          const cGeom = c.geom;
          const g1 = cGeom[0];
          const g2 = cGeom[1];
          if (g1 !== parkingGid && g2 !== parkingGid) {
            const isShard1 = g1 >= minShardGid && g1 <= maxShardGid;
            const isShard2 = g2 >= minShardGid && g2 <= maxShardGid;
            if (isShard1) this.shardHadContactSubstep[g1 - minShardGid] = 1;
            if (isShard2) this.shardHadContactSubstep[g2 - minShardGid] = 1;

            const isTable1 = g1 === tableGid || g1 === shardTableGid;
            const isTable2 = g2 === tableGid || g2 === shardTableGid;
            const isShardOnlyOrTablePair = (isShard1 || isTable1) && (isShard2 || isTable2);
            let shouldRecordPair = true;
            if (isShardOnlyOrTablePair) {
              const s1 = g1 - minShardGid;
              const s2 = g2 - minShardGid;
              const ready1 =
                isShard1 &&
                this.shardAirborneSubsteps[s1] >= 3 &&
                this.simTime - this.shardLastImpactSimTime[s1] >= 0.032;
              const ready2 =
                isShard2 &&
                this.shardAirborneSubsteps[s2] >= 3 &&
                this.simTime - this.shardLastImpactSimTime[s2] >= 0.032;
              shouldRecordPair = ready1 || ready2;
            }

            if (shouldRecordPair) {
              const key = g1 * 4096 + g2;
              let pm = pairMap.get(key);
              if (!pm) {
                if (poolUsed < this.scratchPairPool.length) {
                  pm = this.scratchPairPool[poolUsed++];
                  pm.g1 = g1;
                  pm.g2 = g2;
                  pm.b1 = geomBodyId[g1];
                  pm.b2 = geomBodyId[g2];
                  pm.fnSum = 0;
                  pm.wx = 0;
                  pm.wy = 0;
                  pm.wz = 0;
                  pm.nx = 0;
                  pm.ny = 0;
                  pm.nz = 0;
                } else {
                  pm = {
                    g1,
                    g2,
                    b1: geomBodyId[g1],
                    b2: geomBodyId[g2],
                    fnSum: 0,
                    wx: 0,
                    wy: 0,
                    wz: 0,
                    nx: 0,
                    ny: 0,
                    nz: 0,
                  };
                  this.scratchPairPool.push(pm);
                  poolUsed++;
                }
                pairMap.set(key, pm);
              }
              const cPos = c.pos;
              const cFrame = c.frame;
              const fDm = DM_PER_M * fnN;
              pm.fnSum += fnN;
              pm.wx += cPos[0] * fDm;
              pm.wy += cPos[1] * fDm;
              pm.wz += cPos[2] * fDm;
              pm.nx += cFrame[0] * fnN;
              pm.ny += cFrame[1] * fnN;
              pm.nz += cFrame[2] * fnN;
            }
          }
        }
      }
      c.delete();
    }
    contacts.delete();

    // Soft regularized contact impulse integral: for Kelvin-Voigt contact with timeconst `solref[0]`,
    // total collision impulse J_total relates to peak force F_peak via tau_impact ≈ 1.08 * solref[0].
    const tauImpactSim = Math.max(dtSim, this.cfg.contactSolref[0] * 1.08 * SIM_TIME_SCALE);

    for (const pm of pairMap.values()) {
      const fnN = pm.fnSum;
      if (fnN <= 0.005) continue;
      const invFn = 1.0 / fnN;
      const worldPtDm: Vec3 = [pm.wx * invFn, pm.wy * invFn, pm.wz * invFn];
      const nLen = Math.hypot(pm.nx, pm.ny, pm.nz) || 1.0;
      const n1to2: Vec3 = [pm.nx / nLen, pm.ny / nLen, pm.nz / nLen];
      const n2to1: Vec3 = [-n1to2[0], -n1to2[1], -n1to2[2]];
      const jnSim = fnN * tauImpactSim;

      const { g1, g2, b1, b2 } = pm;
      if (fnN > 0.025) {
        const blk1 = b1 >= 2 ? this.bodyToBlock.get(b1) : undefined;
        const blk2 = b2 >= 2 ? this.bodyToBlock.get(b2) : undefined;
        if (blk1 || blk2) {
          const pairKey = g1 < g2 ? g1 * 4096 + g2 : g2 * 4096 + g1;
          const prevPairT = this.blockPairLastContactSimTime.get(pairKey) ?? -10.0;
          const pairGap = this.simTime - prevPairT;
          const pairSeparated = pairGap >= 0.015;
          const pairSeparatedRaw = pairGap >= 0.010;
          this.blockPairLastContactSimTime.set(pairKey, this.simTime);
          if (blk1) recordForBlock(blk1, b2, g2, worldPtDm, n1to2, fnN, jnSim, pairSeparated, pairSeparatedRaw);
          if (blk2) recordForBlock(blk2, b1, g1, worldPtDm, n2to1, fnN, jnSim, pairSeparated, pairSeparatedRaw);
        }
      }

      const shd1 = b1 >= 2 ? this.bodyToShard.get(b1) : undefined;
      const shd2 = b2 >= 2 ? this.bodyToShard.get(b2) : undefined;
      if (shd1 || shd2) {
        const hitTablePair =
          g1 === tableGid ||
          g2 === tableGid ||
          g1 === shardTableGid ||
          g2 === shardTableGid;
        if (!hitTablePair && fnN > 2.2) {
          if (shd1 && fnN > Math.max(2.2, shd1.massKg * 420.0)) {
            this.demotePinchedHeroShardIfWedged(shd1, fnN);
          }
          if (shd2 && fnN > Math.max(2.2, shd2.massKg * 420.0)) {
            this.demotePinchedHeroShardIfWedged(shd2, fnN);
          }
        }
        const ready1 =
          shd1 !== undefined &&
          this.shardAirborneSubsteps[shd1.slotIndex] >= 3 &&
          this.simTime - this.shardLastImpactSimTime[shd1.slotIndex] >= 0.032;
        const ready2 =
          shd2 !== undefined &&
          this.shardAirborneSubsteps[shd2.slotIndex] >= 3 &&
          this.simTime - this.shardLastImpactSimTime[shd2.slotIndex] >= 0.032;
        const primaryShard =
          shd1 && shd2
            ? ready1 && ready2
              ? shd1.massKg >= shd2.massKg
                ? shd1
                : shd2
              : ready1
                ? shd1
                : shd2
            : (shd1 ?? shd2!);
        const slotIdx = primaryShard.slotIndex;
        const ageSec = this.simTime - (primaryShard.spawnSimTime ?? 0);

        // Ignore initial 22 ms fracture sibling-separation window for both audio & secondary bounce telemetry!
        if (ageSec >= 0.022) {
          const v1 = this.getPreStepPointVelSim(b1, worldPtDm);
          const v2 = this.getPreStepPointVelSim(b2, worldPtDm);
          const relAlongN =
            (v2[0] - v1[0]) * n1to2[0] +
            (v2[1] - v1[1]) * n1to2[1] +
            (v2[2] - v1[2]) * n1to2[2];
          // Closing normal speed in SI m/s (`1 v_sim = 1/sqrt(10) m/s`)
          const vCloseMps = Math.max(0.0, -relAlongN) / FROUDE_VEL_SCALE;
          const hitTable =
            g1 === tableGid ||
            g2 === tableGid ||
            g1 === shardTableGid ||
            g2 === shardTableGid;

          // Require genuine >= 1.0 mm airborne hop (`vCloseMps >= 0.14 m/s`, `>= 3` contact-free substeps) + 32 ms per-slot cooldown
          // so each physical 3D shard bounce emits 1 discrete ShardImpactEvent without substep or sub-mm settle chatter!
          if (
            vCloseMps >= 0.14 &&
            this.shardAirborneSubsteps[slotIdx] >= 3 &&
            this.simTime - this.shardLastImpactSimTime[slotIdx] >= 0.032
          ) {
            this.shardLastImpactSimTime[slotIdx] = this.simTime;
            this.shardAirborneSubsteps[slotIdx] = 0;
            this.stepShardCandidates.push({
              pos: [worldPtDm[0], worldPtDm[1], worldPtDm[2]],
              speed: vCloseMps,
              massKg: primaryShard.massKg,
              surface: hitTable ? 'wood' : 'glass',
              slotIndex: slotIdx,
            });
          }
        }
      }
    }

    // Update airborne substep counters for all active shards after inspecting pair contacts
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const slot = s.slotIndex;
      if (this.shardHadContactSubstep[slot] === 0) {
        const treeId = bodyTreeId[s.bodyHandle];
        if (treeId < 0 || treeAsleep[treeId] < 0) {
          this.shardAirborneSubsteps[slot] = Math.min(
            65535,
            this.shardAirborneSubsteps[slot] + 1
          );
        }
      } else {
        this.shardAirborneSubsteps[slot] = 0;
      }
    }

    if (blockImpacts.size === 0) {
      if (didAnyFractureSyncNeeded) {
        this.worldCompat.syncKinematics();
      }
      return;
    }

    const blocksToShatter: Array<{
      blk: GlassBlockState;
      localHit: Vec3;
      impactSpeed: number;
      worldNormal: Vec3;
      hitTable: boolean;
      closingSpeedReal: number;
      isAnvilCrush: boolean;
    }> = [];

    for (const acc of blockImpacts.values()) {
      const blk = acc.blk;
      let didEmitWobbleForBlock = false;
      let isFreeRockingBodyForBlock = false;

      // 0. Housner Beveled-Prism Rocking Restitution & Discrete Opposite-Edge `WobbleRockEvent` Emission!
      // When a tilted horizontal glass prism slaps flat across its bottom face, MuJoCo's overdamped
      // `contactSolref = [0.0042, 1.12]` (`zeta = 1.12 > 1`) quenches angular velocity on the first slap.
      // For unloaded blocks (`cachedClampingLoadN < 0.38 N`, not grabbed/extracted), transfer dissipative
      // soda-lime glass Housner-Hertzian angular momentum onto the opposite bevel pivot (`v_y^+ = |w^+| * b_half`)
      // and emit discrete `WobbleRockEvent`s on both squat horizontal bevel slaps and upright Housner pivot crossings!
      if (
        acc.upwardSupportFn > 0.06 &&
        !blk.shattered &&
        !blk.extracted &&
        blk.id !== this.activeBlockId
      ) {
        const info = this.worldCompat.getMapping(blk.bodyHandle);
        if (info && info.enabled) {
          const preWxSi = blk.angVel[0] * FROUDE_VEL_SCALE;
          const preWzSi = blk.angVel[2] * FROUDE_VEL_SCALE;
          const preWhSi = Math.hypot(preWxSi, preWzSi);
          const preVyMps = blk.linVel[1] / FROUDE_VEL_SCALE;
          const preVhMps = Math.hypot(blk.linVel[0], blk.linVel[2]) / FROUDE_VEL_SCALE;

          if (preWhSi >= 0.30 && preWhSi <= 22.0 && Math.abs(preVyMps) < 0.55 && preVhMps < 0.45) {
            const adr = info.qposAdr;
            const dof = info.dofAdr;
            const postQuat: Quat = [qpos[adr + 4], qpos[adr + 5], qpos[adr + 6], qpos[adr + 3]];
            const postWWorldSi = quatRotateVec3(postQuat, [
              qvel[dof + 3],
              qvel[dof + 4],
              qvel[dof + 5],
            ]);
            const dotAlongPre =
              (preWxSi * postWWorldSi[0] + preWzSi * postWWorldSi[2]) / preWhSi;
            const clampLoadN =
              blk.id < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[blk.id] : 0;

            const upVec = quatRotateVec3(postQuat, [0, 1, 0]);
            const absUpY = Math.abs(upVec[1]);
            const preUpVec = quatRotateVec3(blk.preQuat, [0, 1, 0]);
            const tiltRad = Math.acos(
              Math.max(-1, Math.min(1, Math.min(absUpY, Math.abs(preUpVec[1]))))
            );

            const arcSubsteps =
              blk.id < this.blockCoherentArcSubsteps.length
                ? this.blockCoherentArcSubsteps[blk.id]
                : 0;
            const isFreeRockingBody =
              clampLoadN < 1.10 &&
              acc.downwardTopFn < 0.35 &&
              acc.distinctContactBodies <= 2 &&
              arcSubsteps >= 3;
            isFreeRockingBodyForBlock = isFreeRockingBody;

            const hasActiveRockHistory =
              blk.id < this.blockRockTransferCount.length &&
              this.blockRockTransferCount[blk.id] > 0 &&
              this.simTime - this.blockLastRockTime[blk.id] < 0.60;
            const isUnloadedSquatSlap =
              (!this.fidelity.elasticUnloadedGlassBounce || isFreeRockingBody || hasActiveRockHistory) &&
              (this.gameMode !== 'sandbox' || isFreeRockingBody) &&
              this.turnPhase !== 'PULLING' &&
              this.turnPhase !== 'COLLAPSED' &&
              !this.sandboxCollapseActive &&
              clampLoadN < 0.38 &&
              acc.downwardTopFn < 0.28 &&
              absUpY > 0.9984 &&
              preWhSi >= 0.32 &&
              dotAlongPre < 0.45 * preWhSi;

            if (
              isUnloadedSquatSlap &&
              blk.id < this.blockRockTransferCount.length &&
              this.simTime - this.blockLastRockTime[blk.id] >= 0.014
            ) {
              if (this.simTime - this.blockLastRockTime[blk.id] > 0.60) {
                this.blockRockTransferCount[blk.id] = 0;
              }
              const kRock = this.blockRockTransferCount[blk.id];
              if (kRock < 5) {
                const ux = preWxSi / preWhSi;
                const uz = preWzSi / preWhSi;
                const uLocal = quatInverseRotateVec3(postQuat, [ux, 0, uz]);
                const bHalfM = Math.max(
                  0.008,
                  Math.abs(uLocal[0]) * 0.01125 + Math.abs(uLocal[2]) * 0.03625
                );
                const eRock = (acc.rockHitTable ? 0.74 : 0.76) * Math.pow(0.88, kRock);
                const newWhSi = Math.min(9.5, preWhSi * eRock);
                if (newWhSi >= 0.25) {
                  const targetWBody = quatInverseRotateVec3(postQuat, [
                    ux * newWhSi,
                    postWWorldSi[1] * 0.45,
                    uz * newWhSi,
                  ]);
                  qvel[dof + 3] = targetWBody[0];
                  qvel[dof + 4] = targetWBody[1];
                  qvel[dof + 5] = targetWBody[2];
                  if (acc.rockHitTable && qpos[adr + 1] < BLOCK_HALF_Y * M_PER_DM + 0.00008) {
                    qpos[adr + 1] = BLOCK_HALF_Y * M_PER_DM + 0.00008;
                  }
                  const pivotVyMps = Math.min(0.20, newWhSi * bHalfM * 1.12 + 0.004);
                  if (qvel[dof + 1] < pivotVyMps) {
                    qvel[dof + 1] = pivotVyMps;
                  }
                  info.quietTicks = 0;
                  info.settledAsleep = false;
                  this.blockRockTransferCount[blk.id] = kRock + 1;
                  this.blockLastRockTime[blk.id] = this.simTime;
                  didEmitWobbleForBlock = true;
                }
              }
            }

            const isSlapImpulse =
              didEmitWobbleForBlock ||
              (isFreeRockingBody &&
                dotAlongPre < 0.62 * preWhSi &&
                acc.maxVCloseReal < CRAZE_IMPACT_SPEED * 0.68);
            const lastWobble = this.blockLastWobbleTime.get(blk.id) ?? -10.0;
            if (isSlapImpulse && this.simTime - lastWobble >= 0.024) {
              this.blockLastWobbleTime.set(blk.id, this.simTime);
              if (blk.id < this.blockCoherentArcSubsteps.length) {
                this.blockCoherentArcSubsteps[blk.id] = 0;
              }
              didEmitWobbleForBlock = true;
              this.stepWobbleCandidates.push({
                blockId: blk.id,
                tiltAngleRad: Math.max(0.008, tiltRad),
                angularSpeedRadPerSec: preWhSi,
                clampingLoadN: clampLoadN,
                worldPos: [
                  acc.rockWorldPtDm[0],
                  acc.rockWorldPtDm[1],
                  acc.rockWorldPtDm[2],
                ],
                surface: acc.rockHitTable ? 'wood' : 'glass',
                damage: blk.damage,
              });
            }
          }
        }
      }

      let handledUnloadedBounceAudio = false;
      const clampLoadN =
        blk.id < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[blk.id] : 0;
      const inPostRebuildCreepWindow = this.simTime - this.towerRebuildSimTime < 0.38;
      const inInitialTowerSpawnSettle =
        this.simTime - this.towerRebuildSimTime < 0.30 &&
        blk.id !== this.trackedDropBlockId &&
        blk.id !== this.activeBlockId &&
        !this.sandboxCollapseActive &&
        this.turnPhase !== 'COLLAPSED' &&
        this.solverLeanAngleDeg < 4.5;
      const inRockingGuard =
        didEmitWobbleForBlock ||
        isFreeRockingBodyForBlock ||
        (blk.id < this.blockLastRockTime.length &&
          this.simTime - this.blockLastRockTime[blk.id] < 0.28);
      const maxElasticBounceSpeed =
        CRAZE_IMPACT_SPEED * 0.85 * (1.0 - 0.36 * Math.min(0.95, blk.damage));
      let hasOverheadBlock = false;
      if (!blk.shattered && !blk.extracted && acc.upwardSupportFn > 0.02 && clampLoadN < 0.45) {
        const bx = blk.position[0];
        const by = blk.position[1];
        const bz = blk.position[2];
        for (let k = 0; k < this.blocks.length; k++) {
          const ob = this.blocks[k];
          if (
            ob.id !== blk.id &&
            !ob.shattered &&
            !ob.extracted &&
            ob.id !== this.activeBlockId &&
            ob.position[1] > by + BLOCK_HEIGHT * 0.55 &&
            Math.abs(ob.position[0] - bx) < BLOCK_LENGTH * 0.85 &&
            Math.abs(ob.position[2] - bz) < BLOCK_LENGTH * 0.85
          ) {
            hasOverheadBlock = true;
            break;
          }
        }
      }
      const isUnloadedCandidate =
        !blk.shattered &&
        !blk.extracted &&
        blk.id !== this.activeBlockId &&
        !hasOverheadBlock &&
        !inRockingGuard &&
        acc.upwardSupportFn > 0.02 &&
        clampLoadN < 0.45 &&
        acc.downwardTopFn < 0.28 &&
        acc.distinctContactBodies <= 3;

      if (this.fidelity.elasticUnloadedGlassBounce && isUnloadedCandidate) {
        const vCloseReal = acc.maxRawVCloseReal;
        const lastBounceT =
          blk.id < this.unloadedBounceLastSimTime.length
            ? this.unloadedBounceLastSimTime[blk.id]
            : -10.0;
        if (this.simTime - lastBounceT > 0.45 && blk.id < this.unloadedBounceHopCount.length) {
          this.unloadedBounceHopCount[blk.id] = 0;
          this.unloadedBouncePeakVClose[blk.id] = 0;
        }
        const hop =
          blk.id < this.unloadedBounceHopCount.length ? this.unloadedBounceHopCount[blk.id] : 0;
        const hasDownwardPreVel = blk.linVel[1] < -0.04;

        if (
          vCloseReal >= 0.28 &&
          vCloseReal < maxElasticBounceSpeed &&
          !(inPostRebuildCreepWindow && vCloseReal < 0.65) &&
          hop < 5 &&
          this.simTime - lastBounceT >= 0.012 &&
          (acc.hasRawSeparatedOnset || (hop === 0 && hasDownwardPreVel && vCloseReal >= 0.95)) &&
          (hop > 0 || hasDownwardPreVel || vCloseReal >= 0.95)
        ) {
          if (hop === 0 && blk.id < this.unloadedBouncePeakVClose.length) {
            this.unloadedBouncePeakVClose[blk.id] = vCloseReal;
          }
          const peakV0 =
            blk.id < this.unloadedBouncePeakVClose.length &&
            this.unloadedBouncePeakVClose[blk.id] > 0
              ? this.unloadedBouncePeakVClose[blk.id]
              : vCloseReal;
          const effVCloseReal =
            hop === 0 ? vCloseReal : Math.min(vCloseReal, peakV0 * Math.pow(0.72, hop));
          const effVCloseMps = effVCloseReal * M_PER_DM;

          // Force-weighted center of pressure across upward support contacts (handles 2-3 cross-course blocks levelly)
          let supW = 0;
          let wpX = 0, wpY = 0, wpZ = 0;
          let lhX = 0, lhY = 0, lhZ = 0;
          let nuX = 0, nuY = 0, nuZ = 0;
          let supTableWeight = 0;
          for (let ci = 0; ci < acc.contacts.length; ci++) {
            const c = acc.contacts[ci];
            if (c.nAtoB[1] < -0.55 && c.fnN > 0.002) {
              const w = c.fnN;
              supW += w;
              wpX += c.worldPtDm[0] * w;
              wpY += c.worldPtDm[1] * w;
              wpZ += c.worldPtDm[2] * w;
              lhX += c.localHitDm[0] * w;
              lhY += c.localHitDm[1] * w;
              lhZ += c.localHitDm[2] * w;
              nuX += -c.nAtoB[0] * w;
              nuY += -c.nAtoB[1] * w;
              nuZ += -c.nAtoB[2] * w;
              if (c.hitTable) supTableWeight += w;
            }
          }
          const invSupW = supW > 1e-6 ? 1.0 / supW : 0.0;
          const copWorldDm: Vec3 =
            invSupW > 0 ? [wpX * invSupW, wpY * invSupW, wpZ * invSupW] : acc.bestRawWorldPtDm;
          const copLocalDm: Vec3 =
            invSupW > 0 ? [lhX * invSupW, lhY * invSupW, lhZ * invSupW] : acc.bestRawLocalHitDm;
          const nRawLen = Math.hypot(nuX, nuY, nuZ);
          const nUp: Vec3 =
            invSupW > 0 && nRawLen > 1e-6
              ? [nuX / nRawLen, nuY / nRawLen, nuZ / nRawLen]
              : [
                  -acc.bestRawNormalAtoB[0],
                  -acc.bestRawNormalAtoB[1],
                  -acc.bestRawNormalAtoB[2],
                ];
          const copHitTable = invSupW > 0 ? supTableWeight >= 0.5 * supW : acc.bestRawHitTable;

          const uTaper = Math.max(0, Math.min(1, (effVCloseReal - 0.28) / 0.72));
          const smoothTaper = uTaper * uTaper * (3 - 2 * uTaper);
          const e0 = copHitTable ? 0.36 : 0.50;
          const eEff = e0 * Math.pow(0.84, hop) * smoothTaper;

          const rb = this.worldCompat.getRigidBody(blk.bodyHandle);
          const info = this.worldCompat.getMapping(blk.bodyHandle);
          if (rb && info && info.enabled) {
            const adr = info.qposAdr;
            const dof = info.dofAdr;
            const postQuat: Quat = [qpos[adr + 4], qpos[adr + 5], qpos[adr + 6], qpos[adr + 3]];
            const wWorldSi = quatRotateVec3(postQuat, [
              qvel[dof + 3],
              qvel[dof + 4],
              qvel[dof + 5],
            ]);
            const rWorldM: Vec3 = [
              copWorldDm[0] * M_PER_DM - qpos[adr + 0],
              copWorldDm[1] * M_PER_DM - qpos[adr + 1],
              copWorldDm[2] * M_PER_DM - qpos[adr + 2],
            ];
            const vPostPtX =
              qvel[dof + 0] + (wWorldSi[1] * rWorldM[2] - wWorldSi[2] * rWorldM[1]);
            const vPostPtY =
              qvel[dof + 1] + (wWorldSi[2] * rWorldM[0] - wWorldSi[0] * rWorldM[2]);
            const vPostPtZ =
              qvel[dof + 2] + (wWorldSi[0] * rWorldM[1] - wWorldSi[1] * rWorldM[0]);
            let vOtherPostAlongN = 0;
            if (acc.bestRawOtherBodyId >= 2) {
              const oInfo = this.worldCompat.getMapping(acc.bestRawOtherBodyId);
              if (oInfo && oInfo.enabled) {
                const oAdr = oInfo.qposAdr;
                const oDof = oInfo.dofAdr;
                const oQuat: Quat = [
                  qpos[oAdr + 4],
                  qpos[oAdr + 5],
                  qpos[oAdr + 6],
                  qpos[oAdr + 3],
                ];
                const oWWorld = quatRotateVec3(oQuat, [
                  qvel[oDof + 3],
                  qvel[oDof + 4],
                  qvel[oDof + 5],
                ]);
                const orM: Vec3 = [
                  copWorldDm[0] * M_PER_DM - qpos[oAdr + 0],
                  copWorldDm[1] * M_PER_DM - qpos[oAdr + 1],
                  copWorldDm[2] * M_PER_DM - qpos[oAdr + 2],
                ];
                const ovX = qvel[oDof + 0] + (oWWorld[1] * orM[2] - oWWorld[2] * orM[1]);
                const ovY = qvel[oDof + 1] + (oWWorld[2] * orM[0] - oWWorld[0] * orM[2]);
                const ovZ = qvel[oDof + 2] + (oWWorld[0] * orM[1] - oWWorld[1] * orM[0]);
                vOtherPostAlongN = ovX * nUp[0] + ovY * nUp[1] + ovZ * nUp[2];
              }
            }
            const vPtPostMps =
              vPostPtX * nUp[0] + vPostPtY * nUp[1] + vPostPtZ * nUp[2] - vOtherPostAlongN;
            const vTargetOutMps = eEff * effVCloseMps;

            if (vTargetOutMps > 0.012 && vPtPostMps < vTargetOutMps) {
              const mA = Math.max(0.01, m.body_mass[blk.bodyHandle]);
              const bid3 = blk.bodyHandle * 3;
              const ix = Math.max(1e-7, m.body_inertia[bid3 + 0]);
              const iy = Math.max(1e-7, m.body_inertia[bid3 + 1]);
              const iz = Math.max(1e-7, m.body_inertia[bid3 + 2]);
              const rLocalM: Vec3 = [
                copLocalDm[0] * M_PER_DM,
                copLocalDm[1] * M_PER_DM,
                copLocalDm[2] * M_PER_DM,
              ];
              const nLocalUp = quatInverseRotateVec3(postQuat, nUp);
              const rcx = rLocalM[1] * nLocalUp[2] - rLocalM[2] * nLocalUp[1];
              const rcy = rLocalM[2] * nLocalUp[0] - rLocalM[0] * nLocalUp[2];
              const rcz = rLocalM[0] * nLocalUp[1] - rLocalM[1] * nLocalUp[0];
              const invMassEffA = 1.0 / mA + (rcx * rcx) / ix + (rcy * rcy) / iy + (rcz * rcz) / iz;

              const dvMps = Math.max(0, vTargetOutMps - Math.max(-effVCloseMps, vPtPostMps));
              const jSi = dvMps / invMassEffA;
              const jSim = jSi * FROUDE_VEL_SCALE;
              rb.applyImpulseAtPoint(
                { x: nUp[0] * jSim, y: nUp[1] * jSim, z: nUp[2] * jSim },
                { x: copWorldDm[0], y: copWorldDm[1], z: copWorldDm[2] },
                true
              );
              info.quietTicks = 0;
              info.settledAsleep = false;
              if (blk.id < this.unloadedBounceLastSimTime.length) {
                this.unloadedBounceLastSimTime[blk.id] = this.simTime;
              }
              if (blk.id < this.unloadedBounceHopCount.length) {
                this.unloadedBounceHopCount[blk.id] = hop + 1;
              }
            }

            if (inInitialTowerSpawnSettle) {
              handledUnloadedBounceAudio = true;
            } else if (this.simTime - (this.blockLastImpactAudioTime.get(blk.id) ?? -10.0) >= 0.012) {
              this.blockLastImpactAudioTime.set(blk.id, this.simTime);
              handledUnloadedBounceAudio = true;
              if (copHitTable && this.fidelity.trueAcousticCoupling) {
                this.emitBlockTableThud(Math.max(0.025, effVCloseMps), copWorldDm, blk.damage);
              } else {
                const clinkIntensity = Math.min(1.0, Math.max(0.04, (effVCloseReal - 0.22) / 11.5));
                this.audio.playGlassImpact(
                  clinkIntensity,
                  this.fidelity.trueAcousticCoupling ? 1.0 : copHitTable ? 0.92 : 1.04,
                  copWorldDm,
                  this.listenerCamPos,
                  this.listenerCamRight,
                  {
                    impactSpeedMps: Math.max(0.02, effVCloseMps),
                    damage: blk.damage,
                    surface: copHitTable ? 'wood' : 'glass',
                  }
                );
              }
            }
          }
        }
      }

      if (
        !handledUnloadedBounceAudio &&
        !inInitialTowerSpawnSettle &&
        this.fidelity.trueAcousticCoupling &&
        acc.maxRawVCloseReal >= 0.30 &&
        acc.maxVCloseReal <= 1.5 &&
        !(inPostRebuildCreepWindow && acc.maxRawVCloseReal < 0.65) &&
        acc.hasRawSeparatedOnset &&
        !didEmitWobbleForBlock &&
        this.simTime - (this.blockLastImpactAudioTime.get(blk.id) ?? -10.0) >= 0.030
      ) {
        this.blockLastImpactAudioTime.set(blk.id, this.simTime);
        handledUnloadedBounceAudio = true;
        if (acc.bestRawHitTable) {
          this.emitBlockTableThud(
            Math.max(0.025, acc.maxRawVCloseReal * M_PER_DM),
            acc.bestRawWorldPtDm,
            blk.damage
          );
        } else {
          const gentleIntensity = Math.min(0.45, Math.max(0.04, (acc.maxRawVCloseReal - 0.25) / 10.0));
          this.audio.playGlassImpact(
            gentleIntensity,
            1.0,
            acc.bestRawWorldPtDm,
            this.listenerCamPos,
            this.listenerCamRight,
            {
              impactSpeedMps: Math.max(0.02, acc.maxRawVCloseReal * M_PER_DM),
              damage: blk.damage,
              surface: 'glass',
            }
          );
        }
      }

      if (acc.maxVCloseReal <= 1.5 || acc.dynamicImpulseSim <= 0) continue;
      const massKg = Math.max(0.01, m.body_mass[blk.bodyHandle]);
      const lh = acc.bestLocalHitDm;
      const nLocal = quatInverseRotateVec3(blk.preQuat, acc.bestNormalAtoB);

      const rcx = lh[1] * nLocal[2] - lh[2] * nLocal[1];
      const rcy = lh[2] * nLocal[0] - lh[0] * nLocal[2];
      const rcz = lh[0] * nLocal[1] - lh[1] * nLocal[0];
      const ix = (massKg / 12.0) * (BLOCK_HEIGHT * BLOCK_HEIGHT + BLOCK_WIDTH * BLOCK_WIDTH);
      const iy = (massKg / 12.0) * (BLOCK_LENGTH * BLOCK_LENGTH + BLOCK_WIDTH * BLOCK_WIDTH);
      const iz = (massKg / 12.0) * (BLOCK_LENGTH * BLOCK_LENGTH + BLOCK_HEIGHT * BLOCK_HEIGHT);
      const invMassEff = 1.0 / massKg + (rcx * rcx) / ix + (rcy * rcy) / iy + (rcz * rcz) / iz;

      // 1. Opposing-wrench anvil-crush & three-point flexural bending moment check!
      let opposingJn = 0;
      let topMidForceN = 0;
      let bottomOuterSupportN = 0;
      const dynN = acc.bestNormalAtoB;
      for (const c of acc.contacts) {
        const dotOpp = c.nAtoB[0] * dynN[0] + c.nAtoB[1] * dynN[1] + c.nAtoB[2] * dynN[2];
        if (dotOpp < -0.5 && !c.otherIsShard) opposingJn += c.jnSim;

        if (!c.otherIsShard) {
          const cNLocal = quatInverseRotateVec3(blk.preQuat, c.nAtoB);
          if (cNLocal[1] > 0.6 && Math.abs(c.localHitDm[0]) < BLOCK_HALF_X * 0.45) {
            topMidForceN += c.fnN;
          } else if (cNLocal[1] < -0.6 && Math.abs(c.localHitDm[0]) > BLOCK_HALF_X * 0.45) {
            bottomOuterSupportN += c.fnN;
          }
        }
      }

      const lastCrazeSec = this.blockLastCrazeTime.get(blk.id) ?? -10.0;
      const dtSinceCraze = this.simTime - lastCrazeSec;
      const inSameSlapWindow = dtSinceCraze < 0.24;
      const cascadeGraceWeight =
        inSameSlapWindow
          ? 0.0
          : (this.turnPhase === 'COLLAPSED' || this.sandboxCollapseActive) && dtSinceCraze < 4.0
            ? 0.08
            : dtSinceCraze < 0.65
              ? 0.30
              : 1.0;
      const effectivePriorDamage = blk.damage * cascadeGraceWeight;
      const clampedPriorDamage = Math.max(0.0, Math.min(0.95, effectivePriorDamage));

      // Progressive Griffith sub-critical flaw weakening: K_Ic(d) = K_Ic0 * (1 - 0.36 * d)
      // Since drop shatter height h ~ v^2, (1 - 0.36 * 0.55)^2 = 0.643 (>= 35% lower drop height at d >= 0.55),
      // while T2 (1.0 cm drop ~ 3.95 dm/s) incurs 0 false damage!
      const griffithWeakening = 1.0 - 0.36 * clampedPriorDamage;

      // True opposing-wrench anvil crush: requires dynamic strike on one face PLUS opposing normal reaction impulse on the opposite face!
      const isAnvilCrush =
        acc.maxVCloseReal > CRAZE_IMPACT_SPEED &&
        acc.dynamicImpulseSim > 0.12 &&
        opposingJn > 0.25 * acc.dynamicImpulseSim;

      // Three-point flexural bending stress: M_bend = F_mid * L / 4; sigma = 6 * M / (b * h^2),
      // gated safely above static 18L stack weight (~12.5 N) and weakened by surface crazing flaws!
      let flexuralBoost = 1.0;
      const flexGateScale = 1.0 - 0.38 * clampedPriorDamage;
      if (
        acc.maxVCloseReal > 4.2 &&
        topMidForceN > 38.0 * flexGateScale &&
        bottomOuterSupportN > 24.0 * flexGateScale
      ) {
        const fBend = Math.min(topMidForceN, bottomOuterSupportN);
        const mBendNm = (fBend * 0.06) / 4.0;
        const sigmaMpa = ((6.0 * mBendNm) / (0.025 * 0.015 * 0.015)) * 1e-6;
        const effFlexThresholdMpa = 0.75 * (1.0 - 0.48 * clampedPriorDamage);
        if (sigmaMpa > effFlexThresholdMpa) {
          flexuralBoost = 1.18 + 0.38 * clampedPriorDamage;
        }
      }

      const anvilBoost = (isAnvilCrush ? this.cfg.anvilCrushBoost : 1.0) * flexuralBoost;

      const jRef = isAnvilCrush ? acc.dynamicImpulseSim : acc.peakDynamicImpulseSim;
      const vComJ = (jRef / massKg) * SIM_TIME_SCALE;
      const vPtJ = jRef * invMassEff * SIM_TIME_SCALE;
      const vImpulse = Math.sqrt(Math.max(0, vComJ * vPtJ));

      const kappaSurface = acc.bestHitTable ? 0.80 : 1.0;
      let extCount = 0;
      if (Math.abs(lh[0]) > BLOCK_HALF_X * 0.78) extCount++;
      if (Math.abs(lh[1]) > BLOCK_HALF_Y * 0.72) extCount++;
      if (Math.abs(lh[2]) > BLOCK_HALF_Z * 0.76) extCount++;
      const kappaGeom = extCount >= 3 ? 1.26 : extCount === 2 ? 1.12 : 0.92;

      const vImpact =
        Math.min(acc.maxVCloseReal, 1.30 * vImpulse) *
        kappaSurface *
        kappaGeom *
        anvilBoost;

      const worldNormal: Vec3 = [-dynN[0], -dynN[1], -dynN[2]];
      const rWorld = quatRotateVec3(blk.preQuat, lh);
      const hitWorld: Vec3 = [
        blk.prePos[0] + rWorld[0],
        blk.prePos[1] + rWorld[1],
        blk.prePos[2] + rWorld[2],
      ];

      const effCrazeSpeed = CRAZE_IMPACT_SPEED * 0.72 * (1.0 - 0.12 * clampedPriorDamage);
      const freshShatterSpeed = isAnvilCrush
        ? SHATTER_IMPACT_SPEED
        : this.turnPhase === 'COLLAPSED'
          ? 15.8
          : 14.3;
      const shatterThreshold = freshShatterSpeed * griffithWeakening;

      const inRockingSequence =
        didEmitWobbleForBlock ||
        (blk.id < this.blockLastRockTime.length &&
          this.simTime - this.blockLastRockTime[blk.id] < 0.25);

      if (
        !handledUnloadedBounceAudio &&
        !inInitialTowerSpawnSettle &&
        vImpact > 1.75 &&
        acc.maxVCloseReal >= 0.85 &&
        (acc.hasOnsetContact || (!this.fidelity.elasticUnloadedGlassBounce && vImpact >= effCrazeSpeed)) &&
        (!inRockingSequence || vImpact >= effCrazeSpeed) &&
        this.simTime - (this.blockLastImpactAudioTime.get(blk.id) ?? -10.0) >= 0.030
      ) {
        this.blockLastImpactAudioTime.set(blk.id, this.simTime);
        if (this.fidelity.trueAcousticCoupling && acc.bestHitTable) {
          this.emitBlockTableThud(
            Math.max(0.03, acc.maxVCloseReal * M_PER_DM),
            hitWorld,
            blk.damage
          );
        } else {
          const clinkIntensity = Math.min(1.0, (vImpact - 1.5) / 14.0);
          this.audio.playGlassImpact(
            clinkIntensity,
            this.fidelity.trueAcousticCoupling ? 1.0 : acc.bestHitTable ? 0.92 : 1.08,
            hitWorld,
            this.listenerCamPos,
            this.listenerCamRight,
            {
              impactSpeedMps: Math.max(0.02, acc.maxVCloseReal * M_PER_DM),
              damage: blk.damage,
              surface: acc.bestHitTable ? 'wood' : 'glass',
            }
          );
        }
      }

      const isUprightStaticSettle =
        !isAnvilCrush &&
        flexuralBoost <= 1.0 &&
        this.solverLeanAngleDeg < 1.5 &&
        Math.hypot(blk.linVel[0], blk.linVel[1], blk.linVel[2]) < 1.8 &&
        acc.maxVCloseReal < CRAZE_IMPACT_SPEED * 0.78;
      if (vImpact >= shatterThreshold) {
        blocksToShatter.push({
          blk,
          localHit: lh,
          impactSpeed: vImpact,
          worldNormal,
          hitTable: acc.bestHitTable,
          closingSpeedReal: acc.maxVCloseReal,
          isAnvilCrush,
        });
      } else if (vImpact >= effCrazeSpeed && !isUprightStaticSettle) {
        const prevDamage = blk.damage;
        const addedDamage =
          0.33 +
          ((vImpact - effCrazeSpeed) / Math.max(2.0, SHATTER_IMPACT_SPEED - effCrazeSpeed)) * 0.48;
        if (inSameSlapWindow) {
          blk.damage = Math.min(0.92, Math.max(prevDamage, addedDamage));
        } else {
          this.blockLastCrazeTime.set(blk.id, this.simTime);
          blk.damage = Math.min(0.92, prevDamage + addedDamage * (1.0 - prevDamage * 0.45));
        }
        if (prevDamage < 0.05 || addedDamage > 0.52) {
          blk.crackLocalOrigin = [...lh];
        }
        if (!inSameSlapWindow) {
          const rb = this.worldCompat.getRigidBody(blk.bodyHandle);
          const post = rb ? rb.linvel() : { x: 0, y: 0, z: 0 };
          this.spawnGlassCulletChips(hitWorld, post, worldNormal, blk.swatchId, 5, 0.55);
          this.audio.playGlassCrack(
            Math.min(1.0, blk.damage),
            hitWorld,
            this.listenerCamPos,
            this.listenerCamRight
          );
        }
      }
    }

    if (blocksToShatter.length > 0) {
      if (blocksToShatter.length > 1) {
        blocksToShatter.sort((a, b) => b.impactSpeed - a.impactSpeed);
      }
      const immediateLimit = this.shards.length >= 160 ? 1 : 2;
      let subShatters = 0;
      for (let idx = 0; idx < blocksToShatter.length; idx++) {
        const item = blocksToShatter[idx];
        if (subShatters < immediateLimit && this.frameShatterCount < maxShattersPerFrame) {
          this.shatterBlock(
            item.blk,
            item.localHit,
            item.impactSpeed,
            item.worldNormal,
            item.hitTable,
            {
              closingSpeedReal: item.closingSpeedReal,
              isAnvilCrush: item.isAnvilCrush,
              skipSyncKinematics: true,
            }
          );
          didAnyFractureSyncNeeded = true;
          subShatters++;
          this.frameShatterCount++;
        } else if (!this.deferredFractures.some((df) => df.blk === item.blk)) {
          const rb = this.worldCompat.getRigidBody(item.blk.bodyHandle);
          const pos = rb ? rb.translation() : { x: item.blk.prePos[0], y: item.blk.prePos[1], z: item.blk.prePos[2] };
          const rot = rb ? rb.rotation() : { x: item.blk.preQuat[0], y: item.blk.preQuat[1], z: item.blk.preQuat[2], w: item.blk.preQuat[3] };
          const lv = rb ? rb.linvel() : { x: item.blk.linVel[0], y: item.blk.linVel[1], z: item.blk.linVel[2] };
          const av = rb ? rb.angvel() : { x: item.blk.angVel[0], y: item.blk.angVel[1], z: item.blk.angVel[2] };
          this.deferredFractures.push({
            blk: item.blk,
            localHit: item.localHit,
            impactSpeed: item.impactSpeed,
            worldNormal: item.worldNormal,
            hitTable: item.hitTable,
            closingSpeedReal: item.closingSpeedReal,
            isAnvilCrush: item.isAnvilCrush,
            savedPosDm: [pos.x, pos.y, pos.z],
            savedQuat: [rot.x, rot.y, rot.z, rot.w],
            savedLinVelSim: [lv.x, lv.y, lv.z],
            savedAngVelSim: [av.x, av.y, av.z],
          });
        }
      }
    }

    if (didAnyFractureSyncNeeded) {
      this.worldCompat.syncKinematics();
    }
  }

  /**
   * MuJoCo 4.0 Native Constraint-Island Introspection:
   * - When `bodyId`'s tree is asleep (`tree_asleep[bTree] >= 0`), MuJoCo 4.0 encodes its exact
   *   sleeping constraint island as a circular linked list inside `d.tree_asleep`!
   * - When `bodyId`'s tree is awake (`tree_asleep[bTree] < 0`), calls `mj_island` if needed and
   *   collects all awake trees sharing `d.tree_island[bTree]` plus any sleeping contact neighbor cycle.
   */
  private collectConstraintIslandTrees(bodyId: number): Uint8Array {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return this.islandWakeMask;
    const ntree = m.ntree;
    if (this.islandWakeMask.length < ntree) {
      this.islandWakeMask = new Uint8Array(Math.max(ntree, 640));
    } else {
      this.islandWakeMask.fill(0, 0, ntree);
    }
    const mask = this.islandWakeMask;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const bTree = bodyTreeId[bodyId];
    if (bTree === undefined || bTree < 0 || bTree >= ntree) {
      return mask;
    }

    mask[bTree] = 1;
    if (treeAsleep[bTree] >= 0) {
      let cur = treeAsleep[bTree];
      let guard = 0;
      while (cur >= 0 && cur < ntree && cur !== bTree && guard++ < ntree) {
        mask[cur] = 1;
        cur = treeAsleep[cur];
      }
      return mask;
    }

    if (d.ntree_awake > 0 && d.nisland === 0) {
      this.mj.mj_island(m, d);
    }
    const nisland = d.nisland;
    if (nisland > 0) {
      const treeIsland = d.tree_island;
      const bIsl = treeIsland[bTree];
      if (bIsl >= 0 && bIsl < nisland) {
        for (let t = 0; t < ntree; t++) {
          if (treeAsleep[t] < 0 && treeIsland[t] === bIsl) {
            mask[t] = 1;
          }
        }
      }
    }

    let anySleepingIntactBlock = false;
    for (let i = 0; i < this.blocks.length; i++) {
      const ob = this.blocks[i];
      if (!ob.shattered && !ob.extracted) {
        const oTree = bodyTreeId[ob.bodyHandle];
        if (oTree >= 0 && treeAsleep[oTree] >= 0) {
          anySleepingIntactBlock = true;
          break;
        }
      }
    }

    const ncon = d.ncon;
    if (anySleepingIntactBlock && ncon > 0) {
      const contacts = d.contact;
      const geomBody = m.geom_bodyid;
      const qpos = d.qpos;
      const bAdr = this.worldCompat.getMapping(bodyId)?.qposAdr ?? -1;
      const bY = bAdr >= 0 ? qpos[bAdr + 1] : 0;
      for (let ci = 0; ci < ncon; ci++) {
        const c = contacts.get(ci);
        if (!c) continue;
        const cGeom = c.geom;
        const b1 = geomBody[cGeom[0]];
        const b2 = geomBody[cGeom[1]];
        c.delete();
        const otherBody = b1 === bodyId ? b2 : b2 === bodyId ? b1 : -1;
        if (otherBody > 1) {
          const oTree = bodyTreeId[otherBody];
          if (oTree >= 0 && oTree < ntree && !mask[oTree] && treeAsleep[oTree] >= 0) {
            const oAdr = this.worldCompat.getMapping(otherBody)?.qposAdr ?? -1;
            const oY = oAdr >= 0 ? qpos[oAdr + 1] : -1;
            if (oY > bY + 0.004) {
              mask[oTree] = 1;
              let cur = treeAsleep[oTree];
              let guard = 0;
              while (cur >= 0 && cur < ntree && cur !== oTree && guard++ < ntree) {
                mask[cur] = 1;
                cur = treeAsleep[cur];
              }
            }
          }
        }
      }
      contacts.delete();
    }
    return mask;
  }

  private parkBodyOnHiddenShelf(bodyId: number, geomId: number, qposAdr: number, dofAdr: number, slotSalt: number): void {
    if (!this.model || !this.data) return;
    setBodyGeomCollisionMask(
      this.model,
      bodyId,
      geomId,
      0,
      MJ_CONAFFINITY_PARKING
    );
    if (slotSalt < 512) {
      updateFreeBodyMassAndInertiaO1(this.model, bodyId, dofAdr, 0.0045, 1e-6, 1e-6, 1e-6);
      this.model.geom_pos[geomId * 3 + 0] = 0;
      this.model.geom_pos[geomId * 3 + 1] = 0;
      this.model.geom_pos[geomId * 3 + 2] = 0;
    }
    const parkXM = ((slotSalt % 32) - 15.5) * 0.12;
    const parkZM = (Math.floor(slotSalt / 32) - 8.0) * 0.12;
    const restHalfYM =
      slotSalt < 512
        ? Math.max(0.0045, Math.min(0.012, this.model.geom_rbound[geomId] || 0.0055))
        : slotSalt < 600
          ? BLOCK_HALF_Y * M_PER_DM
          : Math.max(0.005, this.model.geom_size[geomId * 3 + 1] || 0.50);
    const parkYM = MJ_PARKING_SHELF_TOP_Y_M + restHalfYM;

    this.data.qpos[qposAdr + 0] = parkXM;
    this.data.qpos[qposAdr + 1] = parkYM;
    this.data.qpos[qposAdr + 2] = parkZM;
    this.data.qpos[qposAdr + 3] = 1;
    this.data.qpos[qposAdr + 4] = 0;
    this.data.qpos[qposAdr + 5] = 0;
    this.data.qpos[qposAdr + 6] = 0;
    for (let k = 0; k < 6; k++) {
      this.data.qvel[dofAdr + k] = 0;
      this.data.qacc[dofAdr + k] = 0;
      this.data.qacc_warmstart[dofAdr + k] = 0;
      this.data.qfrc_applied[dofAdr + k] = 0;
      this.data.xfrc_applied[bodyId * 6 + k] = 0;
      this.model.dof_damping[dofAdr + k] = 0;
    }
    this.worldCompat.markShardActive(bodyId, false, 0, MJ_CONAFFINITY_PARKING);
    const tid = this.model.body_treeid[bodyId];
    if (tid >= 0 && this.data.tree_asleep[tid] < 0) {
      this.awakeParkedBodies.push({ bodyId, qposAdr, dofAdr });
    }
  }

  /**
   * Zero-stutter (`~0.13 ms`) 3D Voronoi brittle fracture using the pre-allocated 512-slot MuJoCo pool.
   * Preserves exact `< 0.05%` Voronoi volume conservation and `< 1%` linear & angular momentum conservation!
   */
  public shatterBlock(
    block: GlassBlockState,
    localImpactPoint: Vec3,
    impactSpeed: number = 16.0,
    impactNormal: Vec3 = [0, 1, 0],
    hitTable: boolean = true,
    collisionRebound?: {
      closingSpeedReal: number;
      isAnvilCrush: boolean;
      skipSyncKinematics?: boolean;
      savedPoseAndVel?: {
        posDm: Vec3;
        quat: Quat;
        linVelSim: Vec3;
        angVelSim: Vec3;
      };
    }
  ): void {
    if (block.shattered || !this.model || !this.data) return;
    const rb = this.worldCompat.getRigidBody(block.bodyHandle);
    const blkInfo = this.worldCompat.getMapping(block.bodyHandle);
    if (!rb || !blkInfo) return;

    const saved = collisionRebound?.savedPoseAndVel;
    const pos = saved
      ? { x: saved.posDm[0], y: saved.posDm[1], z: saved.posDm[2] }
      : rb.translation();
    const q: Quat = saved
      ? [...saved.quat]
      : (() => {
          const rot = rb.rotation();
          return [rot.x, rot.y, rot.z, rot.w] as Quat;
        })();
    const lv = saved
      ? { x: saved.linVelSim[0], y: saved.linVelSim[1], z: saved.linVelSim[2] }
      : rb.linvel();
    const av = saved
      ? { x: saved.angVelSim[0], y: saved.angVelSim[1], z: saved.angVelSim[2] }
      : rb.angvel();
    const parentMass = Math.max(0.01, rb.mass());

    // When fracturing mid-contact inside `processWrenchCollisionsAndFractures`, MuJoCo's soft Kelvin-Voigt spring
    // (`solref[0] = 4.2 ms` ~ 3-4 substeps) has only integrated substep 1, leaving `lv · impactNormal < 0`
    // (still plunging into the table/anvil!). Complete the elastic stress-wave rebound before projecting shard CoM
    // momentum, while keeping standalone `shatterBlock` calls (e.g. `T3c`) exact to `< 0.01%` on `lv`!
    const targetLv: Vec3 = [lv.x, lv.y, lv.z];
    if (collisionRebound) {
      const vCloseSim = Math.max(0.0, collisionRebound.closingSpeedReal) / SIM_TIME_SCALE;
      const eFracture = hitTable ? 0.19 : 0.22;
      if (!collisionRebound.isAnvilCrush || impactNormal[1] > -0.25) {
        const vnCur =
          targetLv[0] * impactNormal[0] +
          targetLv[1] * impactNormal[1] +
          targetLv[2] * impactNormal[2];
        const vnTarget = eFracture * (hitTable ? 1.0 : 0.65) * vCloseSim;
        if (vnTarget > vnCur) {
          const dVn = vnTarget - vnCur;
          targetLv[0] += impactNormal[0] * dVn;
          targetLv[1] += impactNormal[1] * dVn;
          targetLv[2] += impactNormal[2] * dVn;
        }
      }
      if (hitTable || collisionRebound.isAnvilCrush || pos.y < 0.35) {
        const minUpVySim = (hitTable ? 0.21 : 0.17) * vCloseSim;
        if (targetLv[1] < minUpVySim) targetLv[1] = minUpVySim;
      }
      if (this.sandboxCollapseActive && Math.hypot(pos.x, pos.z) > 1.8) {
        targetLv[0] *= 0.46;
        targetLv[2] *= 0.46;
      }
    }

    // Capture exact pre-fracture world angular momentum L0 (in Vitrail kg * dm^2 / s_sim)
    const pInertia = rb.principalInertia();
    const pFrame = rb.principalInertiaLocalFrame();
    const qPrin = MujocoPhysicsEngine.quatMul(q, [pFrame.x, pFrame.y, pFrame.z, pFrame.w]);
    const omegaPrin = quatInverseRotateVec3(qPrin, [av.x, av.y, av.z]);
    const parentL0 = quatRotateVec3(qPrin, [
      pInertia.x * omegaPrin[0],
      pInertia.y * omegaPrin[1],
      pInertia.z * omegaPrin[2],
    ]);

    block.shattered = true;
    this.shatteredCount++;
    if (block.id < this.cachedClampingLoadN.length) {
      this.cachedClampingLoadN[block.id] = 0;
      this.blockPhotoelasticStress01[block.id] = 0;
    }
    if (this.activeBlockId === block.id) {
      this.data.eq_active[this.map.grabEqualityId] = 0;
      this.activeBlockId = null;
      this.isPointerDraggingBlock = false;
    }

    const effectiveOrigin: Vec3 =
      block.damage > 0.25
        ? [
            localImpactPoint[0] * 0.3 + block.crackLocalOrigin[0] * 0.7,
            localImpactPoint[1] * 0.3 + block.crackLocalOrigin[1] * 0.7,
            localImpactPoint[2] * 0.3 + block.crackLocalOrigin[2] * 0.7,
          ]
        : [localImpactPoint[0], localImpactPoint[1], localImpactPoint[2]];
    this.lastFractureOriginLocal = [...effectiveOrigin];

    const bPosY = pos.y;
    const islandMask = this.collectConstraintIslandTrees(block.bodyHandle);

    this.bodyToBlock.delete(block.bodyHandle);
    this.parkBodyOnHiddenShelf(
      block.bodyHandle,
      block.colliderHandle,
      blkInfo.qposAdr,
      blkInfo.dofAdr,
      520 + (block.id % 64)
    );

    // MuJoCo 4.0 Native Constraint-Island Wakeup: wake genuinely sleeping blocks & shards in the same
    // constraint island (`d.tree_asleep` circular linked list / contact-supported island) resting ABOVE `bPosY`,
    // without AABB horizontal cutoffs and without resetting `quietTicks` on already-awake lower courses!
    const bodyTreeId = this.model.body_treeid;
    const treeAsleep = this.data.tree_asleep;
    const qfrcApplied = this.data.qfrc_applied;
    const geomContype = this.model.geom_contype;
    for (let k = 0; k < this.blocks.length; k++) {
      const ob = this.blocks[k];
      if (ob.shattered || ob.extracted || geomContype[ob.colliderHandle] === 0) continue;
      if (ob.position[1] <= bPosY + 0.06) continue;
      const tid = bodyTreeId[ob.bodyHandle];
      if (tid >= 0 && treeAsleep[tid] >= 0 && islandMask[tid]) {
        const info = this.worldCompat.getMapping(ob.bodyHandle);
        if (info) {
          info.settledAsleep = false;
          info.quietTicks = 0;
          qfrcApplied[info.dofAdr + 1] += 1e-9;
        }
      }
    }
    for (let k = 0; k < this.shards.length; k++) {
      const os = this.shards[k];
      if (geomContype[os.colliderHandle] === 0) continue;
      if (os.position[1] <= Math.max(0.28, bPosY + 0.05)) continue;
      const tid = bodyTreeId[os.bodyHandle];
      if (tid >= 0 && treeAsleep[tid] >= 0 && islandMask[tid]) {
        this.shardQuietSubsteps[os.slotIndex] = 0;
        this.shardNoSupportSubsteps[os.slotIndex] = 0;
        qfrcApplied[this.map.shardDofAdr[os.slotIndex] + 1] += 1e-9;
        const info = this.worldCompat.getMapping(os.bodyHandle);
        if (info) {
          info.settledAsleep = false;
          info.quietTicks = 0;
        }
      }
    }

    const impactEnergy01 = Math.max(0.0, Math.min(1.0, (impactSpeed - 12.5) / 15.0));
    const energyTargetShards = Math.round(21 + impactEnergy01 * 5);
    const headroom = MAX_ACTIVE_SHARDS - this.shards.length;
    const budgetCap = headroom > 110 ? 25 : headroom > 60 ? 21 : 14;
    const desiredShards = Math.min(energyTargetShards, budgetCap);
    const allowSecondaryCleavage = impactSpeed > 15.0;
    const seedMix =
      this.towerSeed !== 0
        ? (Math.imul(this.towerSeed, 0x9e3779b1) + block.id * 977 + this.shatteredCount * 31) | 0
        : block.id * 977 + Math.floor(this.simTime * 1000) + this.shatteredCount * 31;

    const shardGeoms = fractureBlockVoronoi(
      effectiveOrigin,
      desiredShards,
      seedMix,
      BLOCK_HALF_X,
      BLOCK_HALF_Y,
      BLOCK_HALF_Z,
      block.motif,
      impactEnergy01,
      allowSecondaryCleavage
    );

    // Recycle sleeping or oldest shards from the 512-slot pool if needed
    while (this.freeShardSlots.length < shardGeoms.length && this.shards.length > 0) {
      let removeIdx = 0;
      for (let k = 0; k < this.shards.length; k++) {
        const candRb = this.worldCompat.getRigidBody(this.shards[k].bodyHandle);
        if (!candRb || candRb.isSleeping()) {
          removeIdx = k;
          break;
        }
      }
      const retired = this.shards.splice(removeIdx, 1)[0];
      this.bodyToShard.delete(retired.bodyHandle);
      this.slotRenderVertices[retired.slotIndex] = null;
      this.shardAirborneSubsteps[retired.slotIndex] = 0;
      this.freeShardSlots.push(retired.slotIndex);
      this.parkBodyOnHiddenShelf(
        retired.bodyHandle,
        retired.colliderHandle,
        this.map.shardQposAdr[retired.slotIndex],
        this.map.shardDofAdr[retired.slotIndex],
        retired.slotIndex
      );
    }

    const worldImpact = quatRotateVec3(q, effectiveOrigin);
    const impactWorldPos: Vec3 = [
      pos.x + worldImpact[0],
      pos.y + worldImpact[1],
      pos.z + worldImpact[2],
    ];

    let totalGeomVolume = 0;
    for (const g of shardGeoms) totalGeomVolume += g.volume;
    const invTotalGeomVol = totalGeomVolume > 1e-8 ? 1.0 / totalGeomVolume : 1.0;

    const spawned: Array<{
      geom: (typeof shardGeoms)[number];
      slotIndex: number;
      bodyId: number;
      geomId: number;
      worldPosDm: Vec3;
      rawVelSim: Vec3;
      rawSpinSim: Vec3;
      massKg: number;
    }> = [];

    let activeHeroCount = 0;
    for (let k = 0; k < this.shards.length; k++) {
      if (this.model.geom_contype[this.shards[k].colliderHandle] === MJ_CONTYPE_HERO_SHARD) {
        activeHeroCount++;
      }
    }

    const rotDampRate = this.cfg.shardRotationalDampingRate ?? 1.8;

    const preparedShards: Array<{
      geom: (typeof shardGeoms)[number];
      slotIndex: number;
      hxDm: number;
      hyDm: number;
      hzDm: number;
      clampedLocal: Vec3;
      isHeroShard: boolean;
    }> = [];

    for (const geom of shardGeoms) {
      const slotIndex = this.freeShardSlots.pop();
      if (slotIndex === undefined) break;

      let minPx = 0, maxPx = 0;
      let minPy = 0, maxPy = 0;
      let minPz = 0, maxPz = 0;
      for (let p = 0; p < geom.hullPoints.length; p += 3) {
        const px = geom.hullPoints[p];
        const py = geom.hullPoints[p + 1];
        const pz = geom.hullPoints[p + 2];
        if (px < minPx) minPx = px;
        if (px > maxPx) maxPx = px;
        if (py < minPy) minPy = py;
        if (py > maxPy) maxPy = py;
        if (pz < minPz) minPz = pz;
        if (pz > maxPz) maxPz = pz;
      }
      const spanHxDm = Math.max(0.012, 0.5 * (maxPx - minPx));
      const spanHyDm = Math.max(0.012, 0.5 * (maxPy - minPy));
      const spanHzDm = Math.max(0.012, 0.5 * (maxPz - minPz));

      const aabbVolDm3 = Math.max(1e-6, 8.0 * spanHxDm * spanHyDm * spanHzDm);
      let hxDm: number;
      let hyDm: number;
      let hzDm: number;
      let isHeroShard: boolean;
      if (this.fidelity.solidCulletAndTrueShardHulls) {
        hxDm = Math.min(BLOCK_HALF_X - 0.002, Math.max(0.011, spanHxDm * 0.90));
        hyDm = Math.min(BLOCK_HALF_Y - 0.0015, Math.max(0.011, spanHyDm * 0.96));
        hzDm = Math.min(BLOCK_HALF_Z - 0.002, Math.max(0.011, spanHzDm * 0.90));
        const heroCap = this.shards.length >= 96 ? 24 : 40;
        isHeroShard = geom.volume >= 0.0022 && activeHeroCount < heroCap;
      } else {
        const volFit = Math.max(0.54, Math.min(0.84, Math.cbrt(geom.volume / aabbVolDm3) * 0.92));
        hxDm = Math.min(BLOCK_HALF_X - 0.003, Math.max(0.011, spanHxDm * volFit));
        hyDm = Math.min(BLOCK_HALF_Y - 0.003, Math.max(0.011, spanHyDm * volFit));
        hzDm = Math.min(BLOCK_HALF_Z - 0.003, Math.max(0.011, spanHzDm * volFit));
        isHeroShard = geom.volume >= 0.0022 && activeHeroCount < 56;
      }

      const limX = Math.max(0, BLOCK_HALF_X - hxDm - 0.0015);
      const limY = Math.max(0, BLOCK_HALF_Y - hyDm - 0.0015);
      const limZ = Math.max(0, BLOCK_HALF_Z - hzDm - 0.0015);
      const clampedLocal: Vec3 = [
        Math.max(-limX, Math.min(limX, geom.localCentroid[0])),
        Math.max(-limY, Math.min(limY, geom.localCentroid[1])),
        Math.max(-limZ, Math.min(limZ, geom.localCentroid[2])),
      ];
      if (isHeroShard) activeHeroCount++;

      preparedShards.push({ geom, slotIndex, hxDm, hyDm, hzDm, clampedLocal, isHeroShard });
    }
    let batchHeroCount = 0;
    for (let k = 0; k < preparedShards.length; k++) {
      if (preparedShards[k].isHeroShard) batchHeroCount++;
    }
    const batchSolidForStats = this.fidelity.solidCulletAndTrueShardHulls
      ? preparedShards.length
      : batchHeroCount;
    this.lastCullShardStats = {
      total: this.lastCullShardStats.total + preparedShards.length,
      hero: this.lastCullShardStats.hero + batchSolidForStats,
      cullet: this.lastCullShardStats.cullet + (preparedShards.length - batchSolidForStats),
    };

    for (const prep of preparedShards) {
      const { geom, slotIndex, hxDm, hyDm, hzDm, clampedLocal, isHeroShard } = prep;
      const hxM = hxDm * M_PER_DM;
      const hyM = hyDm * M_PER_DM;
      const hzM = hzDm * M_PER_DM;
      const worldOffset = quatRotateVec3(q, clampedLocal);
      const sxDm = pos.x + worldOffset[0];
      const syDm = pos.y + worldOffset[1];
      const szDm = pos.z + worldOffset[2];
      const safeSyDm =
        pos.y < 0.11 && Math.hypot(sxDm, szDm) <= TABLE_RADIUS + 0.05 && syDm < hyDm + 0.0008
          ? Math.min(syDm + 0.006, hyDm + 0.0008)
          : syDm;
      const shardExactMass = Math.max(0.0002, parentMass * geom.volume * invTotalGeomVol);

      const bodyId = this.map.shardBodyIds[slotIndex];
      const geomId = this.map.shardGeomIds[slotIndex];
      const dofAdr = this.map.shardDofAdr[slotIndex];
      for (let k = 0; k < 6; k++) this.model.dof_damping[dofAdr + k] = 0;

      // Match best-fitting convex Voronoi polyhedron hull archetype for hero shards, or crisp box for small cullet
      if (isHeroShard) {
        let bestArch = 0;
        let bestErr = Infinity;
        for (let a = 0; a < this.map.voronoiArchetypeAspect.length; a++) {
          const asp = this.map.voronoiArchetypeAspect[a];
          const apX = Math.abs(this.archGeomPos[a * 3 + 0]);
          const apY = Math.abs(this.archGeomPos[a * 3 + 1]);
          const apZ = Math.abs(this.archGeomPos[a * 3 + 2]);
          // Never assign a canonical hull archetype that protrudes outside the Voronoi cell bounds (prevents t=0 sibling overlap pop)!
          if (
            asp[0] + apX > hxM * 0.92 ||
            asp[1] + apY > hyM * 0.92 ||
            asp[2] + apZ > hzM * 0.92
          ) {
            continue;
          }
          const err =
            Math.abs(asp[0] - hxM) + Math.abs(asp[1] - hyM) + Math.abs(asp[2] - hzM);
          if (err < bestErr) {
            bestErr = err;
            bestArch = a;
          }
        }
        const fitFound = bestErr < Infinity;
        if (fitFound) {
          const asp = this.map.voronoiArchetypeAspect[bestArch];
          this.model.geom_type[geomId] = 7; // mjGEOM_MESH
          this.model.geom_dataid[geomId] = this.map.voronoiMeshIds[bestArch];
          this.model.geom_pos[geomId * 3 + 0] = this.archGeomPos[bestArch * 3 + 0];
          this.model.geom_pos[geomId * 3 + 1] = this.archGeomPos[bestArch * 3 + 1];
          this.model.geom_pos[geomId * 3 + 2] = this.archGeomPos[bestArch * 3 + 2];
          this.model.geom_quat[geomId * 4 + 0] = this.archGeomQuat[bestArch * 4 + 0];
          this.model.geom_quat[geomId * 4 + 1] = this.archGeomQuat[bestArch * 4 + 1];
          this.model.geom_quat[geomId * 4 + 2] = this.archGeomQuat[bestArch * 4 + 2];
          this.model.geom_quat[geomId * 4 + 3] = this.archGeomQuat[bestArch * 4 + 3];
          this.model.geom_rbound[geomId] = Math.hypot(asp[0], asp[1], asp[2]) * 1.12;
          const aabbBase = geomId * 6;
          this.model.geom_aabb[aabbBase + 0] = 0;
          this.model.geom_aabb[aabbBase + 1] = 0;
          this.model.geom_aabb[aabbBase + 2] = 0;
          this.model.geom_aabb[aabbBase + 3] = asp[0] * 1.05;
          this.model.geom_aabb[aabbBase + 4] = asp[1] * 1.05;
          this.model.geom_aabb[aabbBase + 5] = asp[2] * 1.05;
        } else {
          this.model.geom_type[geomId] = 6; // mjGEOM_BOX fallback when cell is narrower than canonical hulls
          this.model.geom_dataid[geomId] = -1;
          this.model.geom_pos[geomId * 3 + 0] = 0;
          this.model.geom_pos[geomId * 3 + 1] = 0;
          this.model.geom_pos[geomId * 3 + 2] = 0;
          this.model.geom_quat[geomId * 4 + 0] = 1;
          this.model.geom_quat[geomId * 4 + 1] = 0;
          this.model.geom_quat[geomId * 4 + 2] = 0;
          this.model.geom_quat[geomId * 4 + 3] = 0;
          this.model.geom_size[geomId * 3 + 0] = hxM;
          this.model.geom_size[geomId * 3 + 1] = hyM;
          this.model.geom_size[geomId * 3 + 2] = hzM;
          this.model.geom_rbound[geomId] = Math.hypot(hxM, hyM, hzM);
          const aabbBase = geomId * 6;
          this.model.geom_aabb[aabbBase + 0] = 0;
          this.model.geom_aabb[aabbBase + 1] = 0;
          this.model.geom_aabb[aabbBase + 2] = 0;
          this.model.geom_aabb[aabbBase + 3] = hxM;
          this.model.geom_aabb[aabbBase + 4] = hyM;
          this.model.geom_aabb[aabbBase + 5] = hzM;
        }
      } else {
        this.model.geom_type[geomId] = 6; // mjGEOM_BOX
        this.model.geom_dataid[geomId] = -1;
        this.model.geom_pos[geomId * 3 + 0] = 0;
        this.model.geom_pos[geomId * 3 + 1] = 0;
        this.model.geom_pos[geomId * 3 + 2] = 0;
        this.model.geom_quat[geomId * 4 + 0] = 1;
        this.model.geom_quat[geomId * 4 + 1] = 0;
        this.model.geom_quat[geomId * 4 + 2] = 0;
        this.model.geom_quat[geomId * 4 + 3] = 0;
        this.model.geom_size[geomId * 3 + 0] = hxM;
        this.model.geom_size[geomId * 3 + 1] = hyM;
        this.model.geom_size[geomId * 3 + 2] = hzM;
        this.model.geom_rbound[geomId] = Math.hypot(hxM, hyM, hzM);
        const aabbBase = geomId * 6;
        this.model.geom_aabb[aabbBase + 0] = 0;
        this.model.geom_aabb[aabbBase + 1] = 0;
        this.model.geom_aabb[aabbBase + 2] = 0;
        this.model.geom_aabb[aabbBase + 3] = hxM;
        this.model.geom_aabb[aabbBase + 4] = hyM;
        this.model.geom_aabb[aabbBase + 5] = hzM;
      }

      const isAbyssSpawn = pos.y < -0.45;
      const isOffRimSpawn = isAbyssSpawn || Math.hypot(sxDm, szDm) > TABLE_RADIUS + 0.02;
      const contype =
        isHeroShard && !isAbyssSpawn ? MJ_CONTYPE_HERO_SHARD : MJ_CONTYPE_CULLET_SHARD;
      const conaff =
        isHeroShard && !isAbyssSpawn
          ? isOffRimSpawn
            ? MJ_CONAFFINITY_HERO_SHARD_OFF_RIM
            : MJ_CONAFFINITY_HERO_SHARD_ON_TABLE
          : this.getCulletConaffinity(isOffRimSpawn);
      setBodyGeomCollisionMask(this.model, bodyId, geomId, contype, conaff);
      this.worldCompat.markShardActive(bodyId, true, contype, conaff);

      const lxM = 2 * hxM, lyM = 2 * hyM, lzM = 2 * hzM;
      const ixSi = (shardExactMass / 12.0) * (lyM * lyM + lzM * lzM);
      const iySi = (shardExactMass / 12.0) * (lxM * lxM + lzM * lzM);
      const izSi = (shardExactMass / 12.0) * (lxM * lxM + lyM * lyM);
      updateFreeBodyMassAndInertiaO1(
        this.model,
        bodyId,
        dofAdr,
        shardExactMass,
        ixSi,
        iySi,
        izSi,
        rotDampRate
      );

      const shardRb = this.worldCompat.getRigidBody(bodyId)!;
      this.shardQuietSubsteps[slotIndex] = 0;
      shardRb.setTranslation({ x: sxDm, y: safeSyDm, z: szDm }, true);
      shardRb.setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }, true);

      const hash01 = (salt: number): number => {
        const h = Math.sin((block.id + 1) * 127.1 + (slotIndex + 1) * 311.7 + salt * 74.7) * 43758.5453;
        return h - Math.floor(h);
      };

      const rx = sxDm - impactWorldPos[0];
      const ry = syDm - impactWorldPos[1];
      const rz = szDm - impactWorldPos[2];
      const rDotN = rx * impactNormal[0] + ry * impactNormal[1] + rz * impactNormal[2];
      let tx = rx - impactNormal[0] * rDotN;
      let ty = ry - impactNormal[1] * rDotN;
      let tz = rz - impactNormal[2] * rDotN;
      const tLen = Math.hypot(tx, ty, tz);
      if (tLen > 1e-4) {
        tx /= tLen; ty /= tLen; tz /= tLen;
      } else {
        const randAng = hash01(1.0) * Math.PI * 2;
        tx = Math.cos(randAng); ty = 0; tz = Math.sin(randAng);
      }

      const proximity = Math.exp(-geom.impactDist / 0.22);
      const outerSandboxScale =
        this.sandboxCollapseActive && Math.hypot(sxDm, szDm) > 2.2 ? 0.68 : 1.0;
      const splaySpeedSim =
        (((0.24 + impactEnergy01 * 1.15) * (0.28 + 0.72 * proximity) * (0.75 + hash01(2.0) * 0.5)) *
          outerSandboxScale) /
        SIM_TIME_SCALE;
      // 3D Hertzian cone-crack & oblique cleavage wedge dispersion: adjacent shards wedge each other out-of-plane
      // (upward away from tabletop/anvil when near y = 0) before exact CoM linear + angular momentum projection!
      let outNx = impactNormal[0];
      let outNy = impactNormal[1];
      let outNz = impactNormal[2];
      if (hitTable || pos.y < 0.35 || outNy < 0.15) {
        outNx *= 0.35;
        outNy = Math.max(0.85, outNy);
        outNz *= 0.35;
        const nLen = Math.hypot(outNx, outNy, outNz) || 1.0;
        outNx /= nLen; outNy /= nLen; outNz /= nLen;
      }
      const normalReboundSim =
        (((0.22 + impactEnergy01 * 0.65) * proximity +
          (hash01(6.0) - 0.44) * (2.65 + impactEnergy01 * 2.5) * (0.45 + 0.55 * proximity)) /
          SIM_TIME_SCALE);

      const rigidVx = targetLv[0] + (av.y * worldOffset[2] - av.z * worldOffset[1]);
      const rigidVy = targetLv[1] + (av.z * worldOffset[0] - av.x * worldOffset[2]);
      const rigidVz = targetLv[2] + (av.x * worldOffset[1] - av.y * worldOffset[0]);
      const spinScale = 1.4 + proximity * (3.0 + impactEnergy01 * 3.4);

      spawned.push({
        geom,
        slotIndex,
        bodyId,
        geomId,
        worldPosDm: [sxDm, safeSyDm, szDm],
        rawVelSim: [
          rigidVx + tx * splaySpeedSim + outNx * normalReboundSim,
          rigidVy + ty * splaySpeedSim + outNy * normalReboundSim,
          rigidVz + tz * splaySpeedSim + outNz * normalReboundSim,
        ],
        rawSpinSim: [
          av.x + (hash01(3.0) - 0.5) * spinScale,
          av.y + (hash01(4.0) - 0.5) * spinScale,
          av.z + (hash01(5.0) - 0.5) * spinScale,
        ],
        massKg: shardExactMass,
      });
    }

    if (spawned.length > 0) {
      let rawSumMass = 0;
      for (const s of spawned) rawSumMass += s.massKg;
      if (rawSumMass > 1e-6 && Math.abs(rawSumMass - parentMass) > 1e-6) {
        const massScale = parentMass / rawSumMass;
        for (const s of spawned) {
          s.massKg *= massScale;
          const bid3 = s.bodyId * 3;
          updateFreeBodyMassAndInertiaO1(
            this.model,
            s.bodyId,
            this.map.shardDofAdr[s.slotIndex],
            s.massKg,
            this.model.body_inertia[bid3 + 0] * massScale,
            this.model.body_inertia[bid3 + 1] * massScale,
            this.model.body_inertia[bid3 + 2] * massScale,
            rotDampRate
          );
        }
      }

      let sumMass = 0;
      let comX = 0, comY = 0, comZ = 0;
      let pxSum = 0, pySum = 0, pzSum = 0;
      for (const s of spawned) {
        sumMass += s.massKg;
        comX += s.massKg * s.worldPosDm[0];
        comY += s.massKg * s.worldPosDm[1];
        comZ += s.massKg * s.worldPosDm[2];
        pxSum += s.massKg * s.rawVelSim[0];
        pySum += s.massKg * s.rawVelSim[1];
        pzSum += s.massKg * s.rawVelSim[2];
      }
      const invM = 1.0 / Math.max(1e-6, sumMass);
      comX *= invM; comY *= invM; comZ *= invM;
      const dvCorrX = targetLv[0] - pxSum * invM;
      const dvCorrY = targetLv[1] - pySum * invM;
      const dvCorrZ = targetLv[2] - pzSum * invM;
      for (const s of spawned) {
        s.rawVelSim[0] += dvCorrX;
        s.rawVelSim[1] += dvCorrY;
        s.rawVelSim[2] += dvCorrZ;
      }

      // König's theorem: for closed-system free-flight fractures (`!collisionRebound`, zero external
      // contact impulse/torque), include `(com - pos) × P_total` so total angular momentum about `pos`
      // is conserved to machine precision even under OBB clamping (`< 0.01%` T3c error).
      const dComX = !collisionRebound ? comX - pos.x : 0;
      const dComY = !collisionRebound ? comY - pos.y : 0;
      const dComZ = !collisionRebound ? comZ - pos.z : 0;
      let lSumX = sumMass * (dComY * targetLv[2] - dComZ * targetLv[1]);
      let lSumY = sumMass * (dComZ * targetLv[0] - dComX * targetLv[2]);
      let lSumZ = sumMass * (dComX * targetLv[1] - dComY * targetLv[0]);
      const iTot = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (const s of spawned) {
        const rx = s.worldPosDm[0] - comX;
        const ry = s.worldPosDm[1] - comY;
        const rz = s.worldPosDm[2] - comZ;
        const mS = s.massKg;

        lSumX += mS * (ry * s.rawVelSim[2] - rz * s.rawVelSim[1]);
        lSumY += mS * (rz * s.rawVelSim[0] - rx * s.rawVelSim[2]);
        lSumZ += mS * (rx * s.rawVelSim[1] - ry * s.rawVelSim[0]);

        const r2 = rx * rx + ry * ry + rz * rz;
        iTot[0] += mS * (r2 - rx * rx);
        iTot[1] -= mS * rx * ry;
        iTot[2] -= mS * rx * rz;
        iTot[3] -= mS * ry * rx;
        iTot[4] += mS * (r2 - ry * ry);
        iTot[5] -= mS * ry * rz;
        iTot[6] -= mS * rz * rx;
        iTot[7] -= mS * rz * ry;
        iTot[8] += mS * (r2 - rz * rz);

        const rbS = this.worldCompat.getRigidBody(s.bodyId)!;
        const spI = rbS.principalInertia();
        const spF = rbS.principalInertiaLocalFrame();
        const qS = MujocoPhysicsEngine.quatMul(q, [spF.x, spF.y, spF.z, spF.w]);
        const ex = quatRotateVec3(qS, [1, 0, 0]);
        const ey = quatRotateVec3(qS, [0, 1, 0]);
        const ez = quatRotateVec3(qS, [0, 0, 1]);
        for (let rIdx = 0; rIdx < 3; rIdx++) {
          for (let cIdx = 0; cIdx < 3; cIdx++) {
            iTot[rIdx * 3 + cIdx] +=
              spI.x * ex[rIdx] * ex[cIdx] +
              spI.y * ey[rIdx] * ey[cIdx] +
              spI.z * ez[rIdx] * ez[cIdx];
          }
        }
        const wPrinS = quatInverseRotateVec3(qS, s.rawSpinSim);
        const lSpin = quatRotateVec3(qS, [
          spI.x * wPrinS[0],
          spI.y * wPrinS[1],
          spI.z * wPrinS[2],
        ]);
        lSumX += lSpin[0];
        lSumY += lSpin[1];
        lSumZ += lSpin[2];
      }

      const lErr: Vec3 = [lSumX - parentL0[0], lSumY - parentL0[1], lSumZ - parentL0[2]];
      const dOmega = MujocoPhysicsEngine.solve3x3(iTot, lErr);

      for (const s of spawned) {
        const rx = s.worldPosDm[0] - comX;
        const ry = s.worldPosDm[1] - comY;
        const rz = s.worldPosDm[2] - comZ;
        const vx = s.rawVelSim[0] - (dOmega[1] * rz - dOmega[2] * ry);
        const vy = s.rawVelSim[1] - (dOmega[2] * rx - dOmega[0] * rz);
        const vz = s.rawVelSim[2] - (dOmega[0] * ry - dOmega[1] * rx);
        let wx = s.rawSpinSim[0] - dOmega[0];
        let wy = s.rawSpinSim[1] - dOmega[1];
        let wz = s.rawSpinSim[2] - dOmega[2];
        if (collisionRebound || pos.y < -0.45) {
          const maxSpinSim = 55.0 * INV_FROUDE_VEL_SCALE;
          const wSimMag = Math.hypot(wx, wy, wz);
          if (wSimMag > maxSpinSim) {
            const kw = maxSpinSim / wSimMag;
            wx *= kw;
            wy *= kw;
            wz *= kw;
          }
        }

        const rbS = this.worldCompat.getRigidBody(s.bodyId)!;
        rbS.setLinvel({ x: vx, y: vy, z: vz }, true);
        rbS.setAngvel({ x: wx, y: wy, z: wz }, true);

        const firstVertex = s.slotIndex * SHARD_SLOT_STRIDE_VERTS;
        this.pendingSlotUploads.set(s.slotIndex, s.geom.renderVertices);
        this.slotRenderVertices[s.slotIndex] = s.geom.renderVertices;
        this.shardAirborneSubsteps[s.slotIndex] = 0;
        this.shardInContactPrevSubstep[s.slotIndex] = 0;
        const initShardE = this.computeSingleShardTotalEnergyJ(
          s.bodyId,
          this.map.shardQposAdr[s.slotIndex],
          this.map.shardDofAdr[s.slotIndex]
        );
        this.shardPreSubstepTotalEnergyJ[s.slotIndex] = initShardE;
        this.shardEntryTotalEnergyJ[s.slotIndex] = initShardE;
        this.shardLastImpactSimTime[s.slotIndex] = this.simTime;
        this.shardLastAudioSimTime[s.slotIndex] = this.simTime;

        const shardState: GlassShardState = {
          id: this.nextShardId++,
          slotIndex: s.slotIndex,
          bodyHandle: s.bodyId,
          colliderHandle: s.geomId,
          swatchId: block.swatchId,
          motif: block.motif,
          rippleSeed: block.rippleSeed,
          position: [...s.worldPosDm],
          rotation: [...q],
          prePos: [...s.worldPosDm],
          preQuat: [...q],
          preLinVel: [vx, vy, vz],
          preAngVel: [wx, wy, wz],
          volume: s.geom.volume,
          massKg: s.massKg,
          preVy: vy,
          impactLocalPos: [...effectiveOrigin],
          planes: s.geom.planes,
          planeCount: s.geom.planeCount,
          firstVertex,
          vertexCount: s.geom.vertexCount,
          quietSubsteps: 0,
          spawnSimTime: this.simTime,
        };
        this.shards.push(shardState);
        this.bodyToShard.set(s.bodyId, shardState);
      }
      if (!collisionRebound?.skipSyncKinematics) {
        this.worldCompat.syncKinematics();
      }
    }

    const chipCount = Math.round(12 + impactEnergy01 * 10);
    this.spawnGlassCulletChips(
      impactWorldPos,
      lv,
      impactNormal,
      block.swatchId,
      chipCount,
      0.75 + impactEnergy01 * 0.55
    );

    this.audio.playGlassShatter(
      Math.min(1.0, impactSpeed / 22.0),
      hitTable,
      impactWorldPos,
      this.listenerCamPos,
      this.listenerCamRight
    );
  }

  private computeSingleShardTotalEnergyJ(bodyId: number, qposAdr: number, dofAdr: number): number {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return 0;
    const mass = m.body_mass[bodyId];
    if (!(mass > 0)) return 0;
    const yDatumM = -0.35;
    const yM = Math.max(yDatumM, d.qpos[qposAdr + 1]);
    const vx = d.qvel[dofAdr + 0];
    const vy = d.qvel[dofAdr + 1];
    const vz = d.qvel[dofAdr + 2];
    const wx = d.qvel[dofAdr + 3];
    const wy = d.qvel[dofAdr + 4];
    const wz = d.qvel[dofAdr + 5];
    const ib = bodyId * 3;
    return (
      mass * 9.81 * (yM - yDatumM) +
      0.5 * mass * (vx * vx + vy * vy + vz * vz) +
      0.5 *
        (m.body_inertia[ib + 0] * wx * wx +
          m.body_inertia[ib + 1] * wy * wy +
          m.body_inertia[ib + 2] * wz * wz)
    );
  }

  private spawnGlassCulletChips(
    origin: Vec3,
    baseVel: { x: number; y: number; z: number },
    normal: Vec3,
    swatchId: number,
    count: number,
    energyScale: number
  ): void {
    const headroom = MAX_SPLINTERS - this.splinters.length;
    const n = Math.min(count, Math.max(0, headroom));
    for (let i = 0; i < n; i++) {
      const theta = ((i * 2.39996 + this.simTime * 19.1) % (Math.PI * 2));
      const elev = 0.15 + ((i * 0.618) % 0.75);
      const speed = (0.8 + ((i * 0.37) % 2.4)) * energyScale;
      this.splinters.push({
        pos: [
          origin[0] + (Math.cos(theta) * 0.03),
          origin[1] + 0.01,
          origin[2] + (Math.sin(theta) * 0.03),
        ],
        vel: [
          baseVel.x * 0.35 + Math.cos(theta) * speed + normal[0] * 0.6,
          Math.max(0.2, baseVel.y * 0.25 + elev * speed + normal[1] * 0.8),
          baseVel.z * 0.35 + Math.sin(theta) * speed + normal[2] * 0.6,
        ],
        rot: i * 1.1,
        rotSpeed: (i % 2 === 0 ? 1 : -1) * (6 + (i % 9)),
        scale: 0.014 + (i % 5) * 0.0045,
        swatchId,
        life: 0,
        maxLife: 0.65 + (i % 6) * 0.12,
      });
    }
  }

  private updateMicroSplinters(dt: number): void {
    for (let i = this.splinters.length - 1; i >= 0; i--) {
      const sp = this.splinters[i];
      sp.life += dt;
      if (sp.life >= sp.maxLife || sp.pos[1] < -2.5) {
        this.splinters.splice(i, 1);
        continue;
      }
      sp.vel[1] += GRAVITY_Y * dt;
      sp.pos[0] += sp.vel[0] * dt;
      sp.pos[1] += sp.vel[1] * dt;
      sp.pos[2] += sp.vel[2] * dt;
      sp.rot += sp.rotSpeed * dt;

      const rHoriz = Math.hypot(sp.pos[0], sp.pos[2]);
      if (sp.pos[1] < sp.scale * 0.35 && sp.pos[1] > -0.2 && rHoriz < TABLE_RADIUS) {
        sp.pos[1] = sp.scale * 0.35;
        if (Math.abs(sp.vel[1]) > 1.8) {
          sp.vel[1] = -sp.vel[1] * 0.28;
          sp.vel[0] *= 0.65;
          sp.vel[2] *= 0.65;
        } else {
          sp.vel[1] = 0;
          sp.vel[0] *= Math.max(0, 1 - 12 * dt);
          sp.vel[2] *= Math.max(0, 1 - 12 * dt);
          sp.rotSpeed *= Math.max(0, 1 - 12 * dt);
        }
      }
    }
  }

  private emitBlockTableThud(vnMps: number, worldPtDm: Vec3, damage: number): void {
    if (this.simTime - this.lastGlobalWoodThudSimTime < 0.018) return;
    this.lastGlobalWoodThudSimTime = this.simTime;
    this.audio.playBlockThud(
      vnMps,
      worldPtDm,
      this.listenerCamPos,
      this.listenerCamRight,
      damage
    );
  }

  private getCulletConaffinity(offRim: boolean): number {
    return this.fidelity.solidCulletAndTrueShardHulls
      ? offRim
        ? MJ_CONAFFINITY_SOLID_CULLET_OFF_RIM
        : MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE
      : offRim
        ? MJ_CONAFFINITY_CULLET_OFF_RIM
        : MJ_CONAFFINITY_CULLET_ON_TABLE;
  }

  /**
   * Relieves low-rubble hero shards (`yM < 6.5 cm`, age `>= 35 ms`) pinched above `43x` shard weight (`>= 2.2 N`,
   * where opposing top/bottom normal forces cancel in net `d.qfrc_constraint`) by demoting them to cullet masks
   * so convex-mesh wedges beneath collapsed bricks never store and eject multi-Newton impulses. Under
   * `solidCulletAndTrueShardHulls`, brick-top shards (`yM >= 2.2 cm`) stay 100% untouched and demoted floor
   * splinters remain solid (`0x22`) against intact blocks and the oak table while skipping shard-vs-shard wedging.
   */
  private demotePinchedHeroShardIfWedged(shd: GlassShardState | undefined, forceN: number = 99): void {
    const m = this.model;
    const d = this.data;
    if (!shd || !m || !d) return;
    const isHero = m.geom_contype[shd.colliderHandle] === MJ_CONTYPE_HERO_SHARD;
    const isSolidCullet = (m.geom_conaffinity[shd.colliderHandle] & MJ_CONTYPE_BLOCK) !== 0;
    if (!isHero && !isSolidCullet) return;
    if (this.simTime - (shd.spawnSimTime ?? 0) < 0.035) return;
    const qposAdr = this.map.shardQposAdr[shd.slotIndex];
    const xM = d.qpos[qposAdr + 0];
    const yM = d.qpos[qposAdr + 1];
    const zM = d.qpos[qposAdr + 2];
    const minPinchN = Math.max(2.2, shd.massKg * 420.0);
    if (yM >= 0.065 || yM < -0.025 || forceN <= minPinchN) return;
    if (!this.fidelity.solidCulletAndTrueShardHulls && !isHero) return;
    if (this.fidelity.solidCulletAndTrueShardHulls) {
      const sustainedFloorPinch = yM < 0.024 && shd.quietSubsteps >= 180;
      if (!sustainedFloorPinch) {
        const qpos = d.qpos;
        let hasOverheadBlock = false;
        for (let i = 0; i < this.blocks.length; i++) {
          const ob = this.blocks[i];
          if (ob.shattered || ob.extracted) continue;
          const obAdr = this.map.blockQposAdr[ob.id];
          if (obAdr === undefined) continue;
          if (
            qpos[obAdr + 1] > yM + 0.004 &&
            Math.abs(qpos[obAdr + 0] - xM) < 0.048 &&
            Math.abs(qpos[obAdr + 2] - zM) < 0.048
          ) {
            hasOverheadBlock = true;
            break;
          }
        }
        if (!hasOverheadBlock) return;
      }
    }

    const offRim = Math.hypot(xM, zM) > (TABLE_RADIUS + 0.01) * M_PER_DM;
    const caff = offRim ? MJ_CONAFFINITY_CULLET_OFF_RIM : MJ_CONAFFINITY_CULLET_ON_TABLE;
    setBodyGeomCollisionMask(m, shd.bodyHandle, shd.colliderHandle, MJ_CONTYPE_CULLET_SHARD, caff);
    this.worldCompat.markShardActive(shd.bodyHandle, true, MJ_CONTYPE_CULLET_SHARD, caff);

    const dofAdr = this.map.shardDofAdr[shd.slotIndex];
    const bid3 = shd.bodyHandle * 3;
    const iIso =
      (m.body_inertia[bid3 + 0] + m.body_inertia[bid3 + 1] + m.body_inertia[bid3 + 2]) / 3.0;
    updateFreeBodyMassAndInertiaO1(
      m,
      shd.bodyHandle,
      dofAdr,
      m.body_mass[shd.bodyHandle],
      iIso,
      iIso,
      iIso,
      this.cfg.shardRotationalDampingRate ?? 1.8
    );
    for (let k = 0; k < 6; k++) {
      d.qvel[dofAdr + k] *= 0.25;
    }
    const info = this.worldCompat.getMapping(shd.bodyHandle);
    if (info) info.settledAsleep = false;
    this.shardQuietSubsteps[shd.slotIndex] = 0;
    this.worldCompat.wakeBody(shd.bodyHandle);
    const postDemoteE = this.computeSingleShardTotalEnergyJ(shd.bodyHandle, qposAdr, dofAdr);
    this.shardInContactPrevSubstep[shd.slotIndex] = 0;
    this.shardPreSubstepTotalEnergyJ[shd.slotIndex] = postDemoteE;
    this.shardEntryTotalEnergyJ[shd.slotIndex] = postDemoteE;
  }

  public step(frameDt: number): PhysicsStepResult {
    if (!this.model || !this.data) {
      this.lastStepResult = {
        substeps: 0,
        simTime: this.simTime,
        shardImpacts: [],
        wobbleEvents: [],
      };
      return this.lastStepResult;
    }
    const clampedDt = Math.min(0.085, Math.max(0.0, frameDt));
    this.simTime += clampedDt;
    this.stepShardCandidates.length = 0;
    this.stepWobbleCandidates.length = 0;
    this.frameShatterCount = 0;

    if (this.turnPhase === 'EXTRACTED') {
      const decay = Math.exp(-clampedDt * 12.0);
      this.carriedHandVelXZ[0] *= decay;
      this.carriedHandVelXZ[1] *= decay;
    }

    // Fast O(1) MuJoCo 4.0 Sleeping-Island early-out when 100% of dynamic trees are asleep!
    const anyPendingWake = this.hasNonZeroAppliedWakeForces();
    if (this.data.ntree_awake === 0 && !anyPendingWake && this.deferredFractures.length === 0) {
      this.activeTimestepHz = Math.round(1.0 / this.cfg.timestepSec);
      this.activeSolverIterations = this.cfg.solverIterations;
      this.activeLsIterations = this.cfg.lsIterations;
      this.sandboxCollapseActive = false;
      if (this.activeBlockId === null) {
        this.accumulatorSec = 0;
        this.wobbleSpeedDegPerSec = 0;
        this.updateMicroSplinters(clampedDt);
        this.updateTurnStateMachine(clampedDt);
        this.lastStepResult = {
          substeps: 0,
          simTime: this.simTime,
          shardImpacts: [],
          wobbleEvents: [],
        };
        return this.lastStepResult;
      }
      if (this.turnPhase === 'EXTRACTED') {
        this.accumulatorSec = 0;
        this.wobbleSpeedDegPerSec = 0;
        this.advanceCarriedBlockOnly(clampedDt);
        this.updateMicroSplinters(clampedDt);
        this.updateTurnStateMachine(clampedDt);
        this.lastStepResult = {
          substeps: 0,
          simTime: this.simTime,
          shardImpacts: [],
          wobbleEvents: [],
        };
        return this.lastStepResult;
      }
    }

    // Preserve full 600 Hz / 26-iteration Newton precision across all standing, extraction, isolated single-block drops,
    // and post-crash rubble settle/rocking (`maxVisibleBlockSpeedSi < 0.45 m/s && activeFallingBlocks < 4`);
    // strictly gate 360 Hz / reduced-iteration cascade scaling and `mjDSBL_MULTICCD` on LIVE visible kinetic cascade speed/count,
    // excluding parked underground bodies (`ypos < -10 m` or `contype === 0`) so settled rubble sleeps cleanly!
    const {
      visibleAwakeTrees,
      visibleAwakeShards,
      maxVisibleBlockSpeedSi,
      maxVisibleShardSpeedSi,
      activeFallingBlocks,
    } = this.measureVisibleCascadeKinematics();
    if (visibleAwakeTrees === 0 && !anyPendingWake && this.deferredFractures.length === 0) {
      this.sandboxCollapseActive = false;
    }

    const hasCascadeCollapseMotion =
      this.solverLeanAngleDeg >= 8.5 ||
      this.shards.length > 32 ||
      this.shatteredCount >= 1;
    const isPostCrashRubbleSettle =
      (this.shatteredCount >= 8 ||
        (this.turnPhase === 'COLLAPSED' && this.solverLeanAngleDeg >= 25.0)) &&
      activeFallingBlocks === 0 &&
      maxVisibleBlockSpeedSi < 0.28 &&
      maxVisibleShardSpeedSi < 0.40;
    const isCollapseCascade =
      this.sandboxCollapseActive ||
      (this.turnPhase === 'COLLAPSED' && hasCascadeCollapseMotion) ||
      this.shards.length > 48 ||
      (this.shatteredCount >= 2 && this.shards.length >= 32);
    const isSandboxHeavyCascade =
      this.sandboxCollapseActive &&
      !isPostCrashRubbleSettle &&
      this.shards.length >= 160 &&
      visibleAwakeTrees > 24;
    const isSandboxPeakShardCascade =
      isSandboxHeavyCascade && this.shards.length >= 340 && this.shatteredCount >= 19;
    const isSandboxInitialShear =
      this.sandboxCollapseActive &&
      !isPostCrashRubbleSettle &&
      this.shatteredCount === 0 &&
      maxVisibleBlockSpeedSi < 1.75;
    const isRubbleSettleTransition =
      isPostCrashRubbleSettle && visibleAwakeTrees > 18;
    const isHeavyRubbleTransition =
      this.sandboxCollapseActive &&
      isRubbleSettleTransition &&
      this.shards.length >= 160 &&
      visibleAwakeShards > 48;
    const isActivelyToppling =
      !isPostCrashRubbleSettle &&
      (isCollapseCascade ||
        (this.gameMode !== 'sandbox' && this.solverLeanAngleDeg >= 6.5));
    const rawSubDt =
      isSandboxHeavyCascade || isSandboxInitialShear
        ? 1.0 / 240.0
        : isCollapseCascade
          ? 1.0 / 360.0
          : this.cfg.timestepSec;
    const rawIterations = isSandboxPeakShardCascade
      ? 2
      : isSandboxHeavyCascade || isHeavyRubbleTransition
        ? 4
        : isSandboxInitialShear
          ? 5
          : isRubbleSettleTransition
            ? 10
            : isActivelyToppling
              ? 8
              : this.cfg.solverIterations;
    const rawLsIterations = isSandboxPeakShardCascade
      ? 1
      : isSandboxHeavyCascade || isHeavyRubbleTransition || isSandboxInitialShear
        ? 2
        : isRubbleSettleTransition
          ? 3
          : !isPostCrashRubbleSettle && this.sandboxCollapseActive
            ? 3
            : isActivelyToppling
              ? 4
              : this.cfg.lsIterations;

    const isMt = Boolean((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated);
    const isSingleThreadSandboxOverload =
      !isMt &&
      this.sandboxCollapseActive &&
      !isPostCrashRubbleSettle &&
      this.shards.length >= 160;
    const subDt =
      this.fidelity.fullRateCollapseSolver && !isSingleThreadSandboxOverload
        ? Math.min(rawSubDt, 1.0 / 500.0)
        : isSingleThreadSandboxOverload
          ? 1.0 / 240.0
          : rawSubDt;
    this.model.opt.timestep = subDt;
    // Keep `multiccd` invariantly enabled across all states: disabling `mjDSBL_MULTICCD` on 24-vertex
    // beveled prisms drops flat-lying 7.50 mm bricks to single-point GJK/MPR contact (`y = 5.60 mm`,
    // 1.90 mm table sinkage) and injects +13.5 mJ of m*g*dy potential energy upon re-entry.
    void isHeavyRubbleTransition;
    this.model.opt.disableflags &= ~this.mj.mjtDisableBit.mjDSBL_MULTICCD.value;
    this.model.opt.iterations =
      this.fidelity.fullRateCollapseSolver && !isSingleThreadSandboxOverload
        ? Math.max(rawIterations, 14)
        : isSingleThreadSandboxOverload
          ? Math.min(rawIterations, isSandboxInitialShear ? 5 : 4)
          : rawIterations;
    this.model.opt.ls_iterations =
      this.fidelity.fullRateCollapseSolver && !isSingleThreadSandboxOverload
        ? Math.max(rawLsIterations, 4)
        : isSingleThreadSandboxOverload
          ? Math.min(rawLsIterations, 2)
          : rawLsIterations;

    this.activeTimestepHz = Math.round(1.0 / subDt);
    this.activeSolverIterations = this.model.opt.iterations;
    this.activeLsIterations = this.model.opt.ls_iterations;

    const maxSubsteps = isCollapseCascade ? 32 : 52;
    this.accumulatorSec = Math.min(maxSubsteps * subDt, this.accumulatorSec + clampedDt);

    const qfrcApplied = this.data.qfrc_applied;
    const xfrcApplied = this.data.xfrc_applied;

    const isUnforcedFrame =
      this.activeBlockId === null && this.data.eq_active[this.map.grabEqualityId] === 0;
    const guardFramePassivity =
      isUnforcedFrame &&
      (this.multiccdReentrySubstepsRemaining > 0 || subDt > 1.0 / 450.0);
    const ePreFramePassivityJ = guardFramePassivity ? this.computeMechanicalEnergyJ() : 0;
    const shatPreFramePassivity = this.shatteredCount;
    let hadReentrySubstep = false;

    let substeps = 0;
    let hadAwakeSubstep = false;
    while (this.accumulatorSec >= subDt * 0.999 && substeps < maxSubsteps) {
      const isLastSubstep =
        this.accumulatorSec < subDt * 1.999 || substeps + 1 >= maxSubsteps;
      this.worldCompat.applyCustomGravityScalesSubstep();
      this.updateMocapHandAndExtractionSubstep(subDt, isLastSubstep);
      this.preStepClampSettledVelocities();

      const maxSpeedReal = this.snapshotPreStepVelocities();
      if (this.multiccdReentrySubstepsRemaining > 0) {
        this.multiccdReentrySubstepsRemaining--;
        hadReentrySubstep = true;
      }
      this.mj.mj_step(this.model, this.data);
      // Clear ephemeral wake nudges and per-step external forces immediately after mj_step consumes them
      qfrcApplied.fill(0);
      xfrcApplied.fill(0);

      if (this.data.ntree_awake > 0) {
        hadAwakeSubstep = true;
        this.dampGroundedMicroJitterShardsSubstep();
      }
      if (
        isLastSubstep &&
        guardFramePassivity &&
        this.shatteredCount === shatPreFramePassivity
      ) {
        this.enforceMulticcdReentryPassivity(
          ePreFramePassivityJ,
          hadReentrySubstep || isPostCrashRubbleSettle
        );
      }

      if (this.trackedDropBlockId >= 0) {
        const qposAdr = this.map.blockQposAdr[this.trackedDropBlockId];
        if (qposAdr !== undefined) {
          const yDm = this.data.qpos[qposAdr + 1] * 10;
          const clearanceMm = Math.max(0, (yDm - this.trackedDropRestYDm) * 100);
          if (this.trackedDropTraceMm.length < 1500) {
            this.trackedDropTraceMm.push(clearanceMm);
          }
          if (clearanceMm < this.trackedDropMinClearanceMm) {
            this.trackedDropMinClearanceMm = clearanceMm;
          }
          const dofAdr = this.map.blockDofAdr[this.trackedDropBlockId];
          const vySi = dofAdr !== undefined ? this.data.qvel[dofAdr + 1] : 0;
          if (vySi < -0.04) {
            this.trackedDropSawDescent = true;
          } else if (
            this.trackedDropSawDescent &&
            vySi > 0.015 &&
            this.trackedDropMinClearanceMm <= 0.6 &&
            clearanceMm > this.trackedDropMaxReboundMm
          ) {
            this.trackedDropMaxReboundMm = clearanceMm;
          }
        }
      }

      const throttleWrenchToLastSubstep =
        isSandboxPeakShardCascade ||
        (isSandboxInitialShear && this.solverLeanAngleDeg < 22.0) ||
        (this.fidelity.fullRateCollapseSolver &&
          isCollapseCascade &&
          isPostCrashRubbleSettle);
      const maxShattersNow = this.shards.length >= 160 ? 2 : 5;
      const runWrenchThisSubstep =
        (this.deferredFractures.length > 0 && this.frameShatterCount < maxShattersNow) ||
        this.extraPreStep.size > 0 ||
        (maxSpeedReal > 0.32 &&
          (throttleWrenchToLastSubstep
            ? isLastSubstep
            : !isSandboxHeavyCascade || (substeps & 1) === 0 || isLastSubstep));
      if (runWrenchThisSubstep) {
        this.processWrenchCollisionsAndFractures(subDt);
      }

      this.accumulatorSec -= subDt;
      substeps++;
    }

    if (
      hadAwakeSubstep &&
      this.data.ntree_awake > 0 &&
      !(this.sandboxCollapseActive && !isPostCrashRubbleSettle)
    ) {
      this.refreshClampingLoadsAndPhotoelasticStress(false);
    }

    this.updateMicroSplinters(clampedDt);
    this.syncTransformsAndRules();
    this.updateTurnStateMachine(clampedDt);

    let shardImpacts: ShardImpactEvent[] = [];
    if (this.stepShardCandidates.length > 0) {
      if (this.stepShardCandidates.length > 8) {
        this.stepShardCandidates.sort(
          (a, b) => b.massKg * b.speed * b.speed - a.massKg * a.speed * a.speed
        );
        shardImpacts = this.stepShardCandidates.slice(0, 8);
      } else {
        shardImpacts = this.stepShardCandidates.slice();
      }
      for (let i = 0; i < shardImpacts.length; i++) {
        this.pendingShardImpacts.push(shardImpacts[i]);
      }
      if (this.pendingShardImpacts.length > 64) {
        this.pendingShardImpacts.splice(0, this.pendingShardImpacts.length - 64);
      }
    }

    let wobbleEvents: WobbleRockEvent[] = [];
    if (this.stepWobbleCandidates.length > 0) {
      if (this.stepWobbleCandidates.length > 4) {
        this.stepWobbleCandidates.sort(
          (a, b) => b.angularSpeedRadPerSec - a.angularSpeedRadPerSec
        );
        wobbleEvents = this.stepWobbleCandidates.slice(0, 4);
      } else {
        wobbleEvents = this.stepWobbleCandidates.slice();
      }
      for (let i = 0; i < wobbleEvents.length; i++) {
        this.pendingWobbleEvents.push(wobbleEvents[i]);
      }
      if (this.pendingWobbleEvents.length > 32) {
        this.pendingWobbleEvents.splice(0, this.pendingWobbleEvents.length - 32);
      }
    }

    this.lastStepResult = {
      substeps,
      simTime: this.simTime,
      shardImpacts,
      wobbleEvents,
    };
    return this.lastStepResult;
  }

  /**
   * Settles sub-millimeter facet rocking ONLY on grounded glass shards and tabletop rubble (`y < 2.8 cm`)
   * without touching `m.dof_damping` (which is explicit in `integrator="Euler"` and unstable on 0.2g chips),
   * and NEVER brakes elevated/falling blocks in the tower (`y >= 2.8 cm`), preserving exact 1.00g free-fall!
   */
  private dampGroundedMicroJitterShardsSubstep(): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return;

    const qvel = d.qvel;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;

    // Explicit-Euler gyroscopic guard on 9:1-aspect-ratio (`Iy / Ix ~ 9`) intact glass bricks:
    // capping `|ω_SI| <= 35 rad/s` (`~5.6 rev/s`) eliminates outward Euler polhode drift (`0.5 * dt^2 * ωdot^T I ωdot > 0`)
    // during stacked Sandbox smashes while leaving physical Jenga topple/rocking (`< 12 rad/s`) 100% untouched.
    let maxAwakeBlockVSi = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      const treeId = bodyTreeId[b.bodyHandle];
      if (treeId >= 0 && treeAsleep[treeId] >= 0) continue;
      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info || !info.enabled) continue;
      const dof = info.dofAdr;
      const vBlkSi = Math.hypot(qvel[dof + 0], qvel[dof + 1], qvel[dof + 2]);
      if (vBlkSi > maxAwakeBlockVSi) maxAwakeBlockVSi = vBlkSi;
      const wx = qvel[dof + 3];
      const wy = qvel[dof + 4];
      const wz = qvel[dof + 5];
      const wSi = Math.hypot(wx, wy, wz);
      if (wSi > 35.0) {
        const kw = 35.0 / wSi;
        qvel[dof + 3] = wx * kw;
        qvel[dof + 4] = wy * kw;
        qvel[dof + 5] = wz * kw;
      }
    }

    if (
      this.shards.length === 0 &&
      this.shatteredCount === 0 &&
      this.turnPhase !== 'COLLAPSED' &&
      !this.sandboxCollapseActive
    ) {
      return;
    }

    const qpos = d.qpos;
    const qfrcConstraint = d.qfrc_constraint;
    const bodyMass = m.body_mass;
    const geomContype = m.geom_contype;

    const blockRimM = (TABLE_RADIUS + 0.058) * M_PER_DM;
    const blockRimM2 = blockRimM * blockRimM;
    const shardRimM = (TABLE_RADIUS + 0.008) * M_PER_DM;
    const shardRimM2 = shardRimM * shardRimM;
    const tickInc = this.turnPhase === 'COLLAPSED' || this.sandboxCollapseActive ? 2 : 1;
    const isPostCollapse =
      this.turnPhase === 'COLLAPSED' || this.sandboxCollapseActive || this.shatteredCount > 0;
    if (isPostCollapse) {
      for (let i = 0; i < this.blocks.length; i++) {
        const b = this.blocks[i];
        if (b.shattered || b.extracted) continue;
        const info = this.worldCompat.getMapping(b.bodyHandle);
        if (!info || !info.enabled) continue;
        const treeId = bodyTreeId[b.bodyHandle];
        if (treeId >= 0 && treeAsleep[treeId] >= 0) {
          info.settledAsleep = false;
          info.quietTicks = 0;
          continue;
        }
        const xM = qpos[info.qposAdr + 0];
        const yM = qpos[info.qposAdr + 1];
        const zM = qpos[info.qposAdr + 2];
        const onTableRim = xM * xM + zM * zM <= blockRimM2;
        if (!onTableRim || yM < -0.012) {
          info.quietTicks = 0;
          info.settledAsleep = false;
          continue;
        }
        const dof = info.dofAdr;
        const vx = qvel[dof + 0];
        const vy = qvel[dof + 1];
        const vz = qvel[dof + 2];
        const wx = qvel[dof + 3];
        const wy = qvel[dof + 4];
        const wz = qvel[dof + 5];
        const vSi = Math.hypot(vx, vy, vz);
        const wSi = Math.hypot(wx, wy, wz);
        // Strictly require active contact constraint support (`hasBlockContactSupport`) before damping
        // grounded tabletop blocks (`yM < 0.028 m`) or sandwiched post-collapse courses (`settledPileCreep`)—
        // NEVER braking unsupported airborne blocks (`fConstraintMag === 0`)!
        const hasBlockContactSupport =
          qfrcConstraint[dof + 1] > bodyMass[b.bodyHandle] * 1.5 ||
          Math.hypot(
            qfrcConstraint[dof + 0],
            qfrcConstraint[dof + 1],
            qfrcConstraint[dof + 2]
          ) >
            bodyMass[b.bodyHandle] * 1.5;
        const groundedOnTable =
          hasBlockContactSupport &&
          yM > -0.012 &&
          yM < 0.028 &&
          Math.abs(vy) < 0.014 &&
          vSi < 0.032 &&
          wSi < 0.65;
        const settledPileCreep =
          hasBlockContactSupport &&
          yM > -0.012 &&
          yM < 0.28 &&
          Math.abs(vy) < 0.006 &&
          vSi < 0.014 &&
          wSi < 0.22;
        if (groundedOnTable || settledPileCreep) {
          info.quietTicks = Math.min(65535, info.quietTicks + tickInc);
          const thresholdTicks = groundedOnTable ? 28 : 95;
          if (info.quietTicks > thresholdTicks) {
            const damp = info.quietTicks > thresholdTicks + 35 ? 0.35 : 0.78;
            const zeroSnap = info.quietTicks > thresholdTicks + 70;
            qvel[dof + 0] = zeroSnap || Math.abs(vx * damp) < 2e-3 ? 0 : vx * damp;
            qvel[dof + 1] = zeroSnap || Math.abs(vy * damp) < 2e-3 ? 0 : vy * damp;
            qvel[dof + 2] = zeroSnap || Math.abs(vz * damp) < 2e-3 ? 0 : vz * damp;
            qvel[dof + 3] = zeroSnap || Math.abs(wx * damp) < 2e-2 ? 0 : wx * damp;
            qvel[dof + 4] = zeroSnap || Math.abs(wy * damp) < 2e-2 ? 0 : wy * damp;
            qvel[dof + 5] = zeroSnap || Math.abs(wz * damp) < 2e-2 ? 0 : wz * damp;
            if (zeroSnap) {
              info.settledAsleep = true;
              if (
                this.activeBlockId === null &&
                (this.shatteredCount > 0 ||
                  this.turnPhase === 'COLLAPSED' ||
                  this.sandboxCollapseActive)
              ) {
                const qacc = d.qacc;
                const qaccWarm = d.qacc_warmstart;
                for (let k = 0; k < 6; k++) {
                  qacc[dof + k] = 0;
                  qaccWarm[dof + k] = 0;
                }
              }
            }
          }
        } else if (vSi > 0.045 || wSi > 0.85 || Math.abs(vy) > 0.02) {
          info.quietTicks = 0;
          info.settledAsleep = false;
        }
      }
    }

    const bodyInertia = m.body_inertia;
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const info = this.worldCompat.getMapping(s.bodyHandle);
      const treeId = bodyTreeId[s.bodyHandle];
      if (treeId >= 0 && treeAsleep[treeId] >= 0) {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        this.shardInContactPrevSubstep[s.slotIndex] = 0;
        if (info) info.settledAsleep = false;
        continue;
      }
      const qposAdr = this.map.shardQposAdr[s.slotIndex];
      const dofAdr = this.map.shardDofAdr[s.slotIndex];
      const xM = qpos[qposAdr + 0];
      const yM = qpos[qposAdr + 1];
      const zM = qpos[qposAdr + 2];
      let vx = qvel[dofAdr + 0];
      let vy = qvel[dofAdr + 1];
      let vz = qvel[dofAdr + 2];
      let wx = qvel[dofAdr + 3];
      let wy = qvel[dofAdr + 4];
      let wz = qvel[dofAdr + 5];
      s.quietSubsteps = Math.min(65535, s.quietSubsteps + tickInc);

      // Guard first 20 substeps after Voronoi spawn against stiff inter-shard / rubble overlap-pop ejections
      if (s.quietSubsteps <= 20) {
        if (vy > 0.95) {
          vy = 0.95;
          qvel[dofAdr + 1] = vy;
        }
        const sp = Math.hypot(vx, vy, vz);
        if (sp > 2.25) {
          const k = 2.25 / sp;
          vx *= k; vy *= k; vz *= k;
          qvel[dofAdr + 0] = vx;
          qvel[dofAdr + 1] = vy;
          qvel[dofAdr + 2] = vz;
        }
      }
      if (
        (geomContype[s.colliderHandle] === MJ_CONTYPE_HERO_SHARD ||
          (m.geom_conaffinity[s.colliderHandle] & MJ_CONTYPE_BLOCK) !== 0) &&
        this.simTime - (s.spawnSimTime ?? 0) >= 0.035 &&
        yM > -0.025 &&
        yM < 0.065
      ) {
        const vPreSi =
          Math.hypot(s.preLinVel[0], s.preLinVel[1], s.preLinVel[2]) * INV_FROUDE_VEL_SCALE;
        const spPostSi = Math.hypot(vx, vy, vz);
        const vMaxPhys = Math.max(vPreSi + 0.04, 1.35 * maxAwakeBlockVSi + 0.12);
        if (spPostSi > vMaxPhys + 0.35) {
          const capSi = Math.max(vPreSi, 0.85 * maxAwakeBlockVSi);
          const kSp = capSi / spPostSi;
          qvel[dofAdr + 0] = vx * kSp;
          qvel[dofAdr + 1] = vy * kSp;
          qvel[dofAdr + 2] = vz * kSp;
          const wPreSi =
            Math.hypot(s.preAngVel[0], s.preAngVel[1], s.preAngVel[2]) * FROUDE_VEL_SCALE;
          const wPostSi = Math.hypot(wx, wy, wz);
          if (wPostSi > Math.max(12.0, wPreSi + 4.0)) {
            const kwClamp = Math.max(12.0, wPreSi) / wPostSi;
            qvel[dofAdr + 3] = wx * kwClamp;
            qvel[dofAdr + 4] = wy * kwClamp;
            qvel[dofAdr + 5] = wz * kwClamp;
          }
          this.demotePinchedHeroShardIfWedged(s);
          vx = qvel[dofAdr + 0];
          vy = qvel[dofAdr + 1];
          vz = qvel[dofAdr + 2];
          wx = qvel[dofAdr + 3];
          wy = qvel[dofAdr + 4];
          wz = qvel[dofAdr + 5];
        }
      }
      if (yM < 0.032 && vy > 0.95) {
        vy = 0.95;
        qvel[dofAdr + 1] = vy;
      }
      if (yM > 0.22 && vy > 0.35) {
        vy = 0.35;
        qvel[dofAdr + 1] = vy;
      }
      // Cap shard spin so explicit/implicit gyroscopic terms (`ω * dt <= 55/360 = 0.15 rad`) stay bounded
      const wRaw = Math.hypot(wx, wy, wz);
      if (wRaw > 55.0) {
        const kw = 55.0 / wRaw;
        wx *= kw; wy *= kw; wz *= kw;
        qvel[dofAdr + 3] = wx;
        qvel[dofAdr + 4] = wy;
        qvel[dofAdr + 5] = wz;
      }

      // Thermodynamic passivity guard across explicit-Euler / `solimp` high-spin tabletop bounces:
      // Cullet shards only collide with stationary `oak_table_shard`, and hero shards with no fast moving block
      // (`maxAwakeBlockVSi < 0.05`) only collide with passive surfaces, so total mechanical energy cannot exceed
      // the pre-impact entry envelope `shardEntryTotalEnergyJ` (leaving normal `e <= 1` `T9` rebounds 100% untouched).
      const mShd = bodyMass[s.bodyHandle];
      const fConMag = Math.hypot(
        qfrcConstraint[dofAdr + 0],
        qfrcConstraint[dofAdr + 1],
        qfrcConstraint[dofAdr + 2]
      );
      const inContactPost = fConMag > 1e-4;
      if (mShd > 0) {
        const yDatumM = -0.35;
        const pePostJ = mShd * 9.81 * (Math.max(yDatumM, yM) - yDatumM);
        const ib = s.bodyHandle * 3;
        const kePostJ =
          0.5 * mShd * (vx * vx + vy * vy + vz * vz) +
          0.5 *
            (bodyInertia[ib + 0] * wx * wx +
              bodyInertia[ib + 1] * wy * wy +
              bodyInertia[ib + 2] * wz * wz);
        const ePostJ = pePostJ + kePostJ;
        const isCullet = geomContype[s.colliderHandle] === MJ_CONTYPE_CULLET_SHARD;
        const canGainFromMovingBlock = !isCullet && maxAwakeBlockVSi >= 0.05 && inContactPost;

        if (!inContactPost) {
          const ePreSubJ = this.shardPreSubstepTotalEnergyJ[s.slotIndex];
          if (ePreSubJ > 0 && ePostJ > ePreSubJ + 1e-6 && kePostJ > 1e-9) {
            const keTargetJ = Math.max(0, ePreSubJ - pePostJ);
            const scale = Math.sqrt(keTargetJ / kePostJ);
            vx *= scale; vy *= scale; vz *= scale;
            wx *= scale; wy *= scale; wz *= scale;
            qvel[dofAdr + 0] = vx;
            qvel[dofAdr + 1] = vy;
            qvel[dofAdr + 2] = vz;
            qvel[dofAdr + 3] = wx;
            qvel[dofAdr + 4] = wy;
            qvel[dofAdr + 5] = wz;
            this.shardEntryTotalEnergyJ[s.slotIndex] = pePostJ + keTargetJ;
          } else {
            this.shardEntryTotalEnergyJ[s.slotIndex] = ePostJ;
          }
          this.shardInContactPrevSubstep[s.slotIndex] = 0;
        } else if (canGainFromMovingBlock) {
          this.shardEntryTotalEnergyJ[s.slotIndex] = Math.max(
            this.shardEntryTotalEnergyJ[s.slotIndex],
            ePostJ
          );
          this.shardInContactPrevSubstep[s.slotIndex] = 1;
        } else {
          const eEntryJ = this.shardEntryTotalEnergyJ[s.slotIndex];
          if (eEntryJ > 0 && ePostJ > eEntryJ + 1e-5 && kePostJ > 1e-9) {
            const keTargetJ = Math.max(0, 0.82 * (eEntryJ - pePostJ));
            const scale = Math.sqrt(keTargetJ / kePostJ);
            vx *= scale; vy *= scale; vz *= scale;
            wx *= scale; wy *= scale; wz *= scale;
            qvel[dofAdr + 0] = vx;
            qvel[dofAdr + 1] = vy;
            qvel[dofAdr + 2] = vz;
            qvel[dofAdr + 3] = wx;
            qvel[dofAdr + 4] = wy;
            qvel[dofAdr + 5] = wz;
            this.shardEntryTotalEnergyJ[s.slotIndex] = pePostJ + keTargetJ;
          }
          this.shardInContactPrevSubstep[s.slotIndex] = 1;
        }
      } else {
        this.shardInContactPrevSubstep[s.slotIndex] = 0;
      }

      // Hard physical invariant: above the static oak tabletop (`yM >= 0.026 m`), track consecutive zero-support
      // substeps via `d.qfrc_constraint`. Allowing `<= 4` chatter substeps (`<= 6.7 ms`) lets facet-buzzing pile
      // shards sleep cleanly, while airborne bounce apexes (`vy ~ 0`) and shards losing support stay 100% free-falling!
      const fCy = qfrcConstraint[dofAdr + 1];
      let noSup = 0;
      if (yM >= 0.026) {
        if (fCy > bodyMass[s.bodyHandle] * 1.5) {
          this.shardNoSupportSubsteps[s.slotIndex] = 0;
        } else {
          noSup = Math.min(255, this.shardNoSupportSubsteps[s.slotIndex] + 1);
          this.shardNoSupportSubsteps[s.slotIndex] = noSup;
        }
      } else {
        this.shardNoSupportSubsteps[s.slotIndex] = 0;
      }
      const hasUpwardSupport = yM < 0.026 || noSup <= 4;
      const onTableRim = xM * xM + zM * zM <= shardRimM2;
      // Bleed high-frequency vertex top-spin strictly during verified upward tabletop/block contact after initial bounce window (`> 180` substeps)
      if (
        onTableRim &&
        s.quietSubsteps > 180 &&
        yM > -0.008 &&
        yM < 0.042 &&
        fCy > bodyMass[s.bodyHandle] * 1.2
      ) {
        const lateSpin = s.quietSubsteps > 420;
        const wDamp = lateSpin ? 0.35 : 0.85;
        const vDamp = lateSpin ? 0.55 : 0.94;
        wx *= wDamp;
        wy *= wDamp;
        wz *= wDamp;
        vx *= vDamp;
        vz *= vDamp;
        if (lateSpin && vy > 0.03) {
          vy *= 0.20;
          qvel[dofAdr + 1] = vy;
        }
        qvel[dofAdr + 0] = vx;
        qvel[dofAdr + 2] = vz;
        qvel[dofAdr + 3] = wx;
        qvel[dofAdr + 4] = wy;
        qvel[dofAdr + 5] = wz;
      }
      if (!onTableRim || yM < -0.012 || (Math.abs(vy) > 0.09 && s.quietSubsteps < 360)) {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        if (info) info.settledAsleep = false;
        continue;
      }
      // Relieve tabletop hero shards pinched beneath heavy collapsed intact blocks so `zeroSnap`
      // never turns a thin floor splinter into a buzzing wedge (brick-top shards stay untouched).
      if (
        yM < 0.024 &&
        Math.abs(fCy) > 2.2 &&
        s.quietSubsteps > 220 &&
        (geomContype[s.colliderHandle] === MJ_CONTYPE_HERO_SHARD ||
          (m.geom_conaffinity[s.colliderHandle] & MJ_CONTYPE_BLOCK) !== 0)
      ) {
        const offRimClamp = Math.hypot(qpos[qposAdr + 0], qpos[qposAdr + 2]) > (TABLE_RADIUS + 0.01) * M_PER_DM;
        const caffClamp = offRimClamp ? MJ_CONAFFINITY_CULLET_OFF_RIM : MJ_CONAFFINITY_CULLET_ON_TABLE;
        setBodyGeomCollisionMask(
          m,
          s.bodyHandle,
          s.colliderHandle,
          MJ_CONTYPE_CULLET_SHARD,
          caffClamp
        );
        this.worldCompat.markShardActive(s.bodyHandle, true, MJ_CONTYPE_CULLET_SHARD, caffClamp);
        const bid3 = s.bodyHandle * 3;
        const iIso =
          (m.body_inertia[bid3 + 0] + m.body_inertia[bid3 + 1] + m.body_inertia[bid3 + 2]) / 3.0;
        updateFreeBodyMassAndInertiaO1(
          m,
          s.bodyHandle,
          dofAdr,
          bodyMass[s.bodyHandle],
          iIso,
          iIso,
          iIso,
          this.cfg.shardRotationalDampingRate ?? 1.8
        );
        vx *= 0.2; vy *= 0.2; vz *= 0.2;
        wx *= 0.2; wy *= 0.2; wz *= 0.2;
        qvel[dofAdr + 0] = vx;
        qvel[dofAdr + 1] = vy;
        qvel[dofAdr + 2] = vz;
        qvel[dofAdr + 3] = wx;
        qvel[dofAdr + 4] = wy;
        qvel[dofAdr + 5] = wz;
      }
      const vSi = Math.hypot(vx, vy, vz);
      const wSi = Math.hypot(wx, wy, wz);
      const groundedShard =
        yM < 0.026 &&
        ((vSi < 0.11 && Math.abs(vy) < 0.05 && wSi < 3.6) ||
          (s.quietSubsteps > 300 &&
            fCy > bodyMass[s.bodyHandle] * 1.2 &&
            vSi < 0.28 &&
            Math.abs(vy) < 0.09));
      const supportedPileBuzz =
        hasUpwardSupport && yM < 0.22 && s.quietSubsteps > 280 && vSi < 0.32 && Math.abs(vy) < 0.16;
      if (groundedShard || supportedPileBuzz) {
        const qCount = Math.min(65535, this.shardQuietSubsteps[s.slotIndex] + tickInc);
        this.shardQuietSubsteps[s.slotIndex] = qCount;
        if (qCount > 14 || supportedPileBuzz) {
          const damp = qCount > 40 || (supportedPileBuzz && s.quietSubsteps > 440) ? 0.25 : 0.72;
          const zeroSnap = qCount > 75 || (supportedPileBuzz && s.quietSubsteps > 580);
          qvel[dofAdr + 0] = zeroSnap || Math.abs(vx * damp) < 5e-3 ? 0 : vx * damp;
          qvel[dofAdr + 1] = zeroSnap || Math.abs(vy * damp) < 5e-3 ? 0 : vy * damp;
          qvel[dofAdr + 2] = zeroSnap || Math.abs(vz * damp) < 5e-3 ? 0 : vz * damp;
          qvel[dofAdr + 3] = zeroSnap || Math.abs(wx * damp) < 4e-2 ? 0 : wx * damp;
          qvel[dofAdr + 4] = zeroSnap || Math.abs(wy * damp) < 4e-2 ? 0 : wy * damp;
          qvel[dofAdr + 5] = zeroSnap || Math.abs(wz * damp) < 4e-2 ? 0 : wz * damp;
          if (zeroSnap) {
            if (info) info.settledAsleep = true;
            const qacc = d.qacc;
            const qaccWarm = d.qacc_warmstart;
            for (let k = 0; k < 6; k++) {
              qacc[dofAdr + k] = 0;
              qaccWarm[dofAdr + k] = 0;
            }
          }
        }
      } else {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        if (info && vSi > 0.18) info.settledAsleep = false;
      }
    }

    if (m.opt.timestep > 1.0 / 450.0) {
      const qaccWarm = d.qacc_warmstart;
      for (let i = 0; i < this.blocks.length; i++) {
        const b = this.blocks[i];
        if (b.shattered || b.extracted || b.id === this.activeBlockId) continue;
        const info = this.worldCompat.getMapping(b.bodyHandle);
        if (!info || !info.enabled || info.settledAsleep) continue;
        const dof = info.dofAdr;
        const vx = qvel[dof + 0];
        const vy = qvel[dof + 1];
        const vz = qvel[dof + 2];
        const vPostSi = Math.hypot(vx, vy, vz);
        const vPreSi =
          b.id < this.blockPreSubstepLinSpeedSi.length ? this.blockPreSubstepLinSpeedSi[b.id] : 0;
        const wPreSi =
          b.id < this.blockPreSubstepAngSpeedSi.length ? this.blockPreSubstepAngSpeedSi[b.id] : 0;
        const selfBaseSi = vPreSi + wPreSi * 0.025;
        if (vPostSi <= selfBaseSi + 0.20) continue;

        const bx = qpos[info.qposAdr + 0];
        const by = qpos[info.qposAdr + 1];
        const bz = qpos[info.qposAdr + 2];
        let maxLocalPreBlkVSi = vPreSi;
        for (let j = 0; j < this.blocks.length; j++) {
          const ob = this.blocks[j];
          if (ob.shattered || ob.extracted || ob.id >= this.blockPreSubstepLinSpeedSi.length) {
            continue;
          }
          const vp = this.blockPreSubstepLinSpeedSi[ob.id];
          if (vp <= maxLocalPreBlkVSi) continue;
          const obAdr = this.map.blockQposAdr[ob.id];
          if (obAdr === undefined) continue;
          const dx = qpos[obAdr + 0] - bx;
          const dy = qpos[obAdr + 1] - by;
          const dz = qpos[obAdr + 2] - bz;
          if (dx * dx + dy * dy + dz * dz < 0.0121) {
            maxLocalPreBlkVSi = vp;
          }
        }

        const vCeilSi = Math.max(selfBaseSi, maxLocalPreBlkVSi) + 0.14;
        if (vPostSi > vCeilSi + 0.20) {
          const capSi = Math.max(vPreSi, 0.88 * maxLocalPreBlkVSi) + 0.04;
          const k = capSi / vPostSi;
          qvel[dof + 0] = vx * k;
          qvel[dof + 1] = vy * k;
          qvel[dof + 2] = vz * k;
          const wx = qvel[dof + 3];
          const wy = qvel[dof + 4];
          const wz = qvel[dof + 5];
          const wPostSi = Math.hypot(wx, wy, wz);
          const wCapSi = Math.max(wPreSi + 2.0, 9.0);
          if (wPostSi > wCapSi) {
            const kw = wCapSi / wPostSi;
            qvel[dof + 3] = wx * kw;
            qvel[dof + 4] = wy * kw;
            qvel[dof + 5] = wz * kw;
          }
          for (let c = 0; c < 6; c++) {
            qaccWarm[dof + c] = 0;
          }
        }
      }
    }
  }

  /**
   * Ensures toggling `mjDSBL_MULTICCD` from disabled back to enabled during rubble settle remains
   * thermodynamically passive when newly generated 4-point box contacts relax minor interpenetrations.
   */
  private enforceMulticcdReentryPassivity(ePreReentryJ: number, dampFloorPop: boolean): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return;
    const tolJ = dampFloorPop ? 0.0004 : 0.0008;
    let ePostReentryJ = this.computeMechanicalEnergyJ();
    if (ePostReentryJ <= ePreReentryJ + tolJ) return;

    const qpos = d.qpos;
    const qvel = d.qvel;
    if (dampFloorPop) {
      for (let i = 0; i < this.blocks.length; i++) {
        const b = this.blocks[i];
        if (b.shattered || b.extracted) continue;
        const info = this.worldCompat.getMapping(b.bodyHandle);
        if (!info || !info.enabled || info.settledAsleep) continue;
        const yM = qpos[info.qposAdr + 1];
        const dof = info.dofAdr;
        if (yM < 0.055 && qvel[dof + 1] > 0.015) {
          qvel[dof + 1] *= 0.15;
        }
      }

      ePostReentryJ = this.computeMechanicalEnergyJ();
      if (ePostReentryJ <= ePreReentryJ + tolJ) return;
    }

    const qfrcApplied = d.qfrc_applied;
    const qaccWarm = d.qacc_warmstart;
    const bodyMass = m.body_mass;
    const bodyInertia = m.body_inertia;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;

    const visitAwakeBodies = (cb: (bodyId: number, dof: number) => void): void => {
      const visit = (bodyId: number, dof: number, settledAsleep: boolean): void => {
        if (settledAsleep) return;
        const tid = bodyTreeId[bodyId];
        const vx = qvel[dof + 0];
        const vy = qvel[dof + 1];
        const vz = qvel[dof + 2];
        const wx = qvel[dof + 3];
        const wy = qvel[dof + 4];
        const wz = qvel[dof + 5];
        const speedSq = vx * vx + vy * vy + vz * vz + wx * wx + wy * wy + wz * wz;
        if (tid >= 0 && treeAsleep[tid] >= 0 && qfrcApplied[dof + 1] === 0 && speedSq <= 1e-12) {
          return;
        }
        cb(bodyId, dof);
      };
      for (let i = 0; i < this.blocks.length; i++) {
        const b = this.blocks[i];
        if (b.shattered || b.extracted) continue;
        const info = this.worldCompat.getMapping(b.bodyHandle);
        if (info && info.enabled) visit(b.bodyHandle, info.dofAdr, info.settledAsleep);
      }
      for (let i = 0; i < this.shards.length; i++) {
        const s = this.shards[i];
        const info = this.worldCompat.getMapping(s.bodyHandle);
        if (info && info.enabled) {
          visit(s.bodyHandle, this.map.shardDofAdr[s.slotIndex], info.settledAsleep);
        }
      }
    };

    let totalAwakeKeJ = 0;
    visitAwakeBodies((bodyId, dof) => {
      const mass = bodyMass[bodyId];
      if (!(mass > 0)) return;
      const vx = qvel[dof + 0];
      const vy = qvel[dof + 1];
      const vz = qvel[dof + 2];
      const wx = qvel[dof + 3];
      const wy = qvel[dof + 4];
      const wz = qvel[dof + 5];
      const ib = bodyId * 3;
      totalAwakeKeJ +=
        0.5 * mass * (vx * vx + vy * vy + vz * vz) +
        0.5 *
          (bodyInertia[ib + 0] * wx * wx +
            bodyInertia[ib + 1] * wy * wy +
            bodyInertia[ib + 2] * wz * wz);
    });
    if (totalAwakeKeJ <= 1e-6) return;

    const excessJ = ePostReentryJ - ePreReentryJ;
    const scale = Math.max(0, Math.min(1, Math.sqrt(Math.max(0, 1.0 - excessJ / totalAwakeKeJ))));
    if (scale >= 0.9999) return;

    visitAwakeBodies((_bodyId, dof) => {
      for (let k = 0; k < 6; k++) {
        qvel[dof + k] *= scale;
        qaccWarm[dof + k] = 0;
      }
    });
  }

  private measureVisibleCascadeKinematics(): {
    visibleAwakeTrees: number;
    visibleAwakeBlocks: number;
    visibleAwakeShards: number;
    maxVisibleBlockSpeedSi: number;
    maxVisibleShardSpeedSi: number;
    activeFallingBlocks: number;
  } {
    const m = this.model;
    const d = this.data;
    if (!m || !d) {
      return {
        visibleAwakeTrees: 0,
        visibleAwakeBlocks: 0,
        visibleAwakeShards: 0,
        maxVisibleBlockSpeedSi: 0,
        maxVisibleShardSpeedSi: 0,
        activeFallingBlocks: 0,
      };
    }
    const qpos = d.qpos;
    const qvel = d.qvel;
    const qfrcApplied = d.qfrc_applied;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const geomContype = m.geom_contype;

    let visibleAwakeTrees = 0;
    let visibleAwakeBlocks = 0;
    let visibleAwakeShards = 0;
    let maxVisibleBlockSpeedSi = 0;
    let maxVisibleShardSpeedSi = 0;
    let activeFallingBlocks = 0;

    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted || geomContype[b.colliderHandle] === 0) continue;
      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info || !info.enabled) continue;
      const dof = info.dofAdr;
      const treeId = bodyTreeId[b.bodyHandle];
      const isWaking =
        qfrcApplied[dof + 1] !== 0 ||
        Math.abs(qvel[dof + 0]) + Math.abs(qvel[dof + 1]) + Math.abs(qvel[dof + 2]) > 1e-4;
      if (!isWaking && treeId >= 0 && treeAsleep[treeId] >= 0) continue;
      const yM = qpos[info.qposAdr + 1];
      if (yM < -10.0) continue;
      visibleAwakeTrees++;
      visibleAwakeBlocks++;
      const vBlkSi = Math.hypot(qvel[dof + 0], qvel[dof + 1], qvel[dof + 2]);
      if (vBlkSi > maxVisibleBlockSpeedSi) maxVisibleBlockSpeedSi = vBlkSi;
      if (vBlkSi >= 0.35 || (yM > 0.04 && vBlkSi >= 0.22)) {
        activeFallingBlocks++;
      }
    }

    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      if (geomContype[s.colliderHandle] === 0) continue;
      const treeId = bodyTreeId[s.bodyHandle];
      if (treeId >= 0 && treeAsleep[treeId] >= 0) continue;
      const qposAdr = this.map.shardQposAdr[s.slotIndex];
      if (qpos[qposAdr + 1] < -10.0) continue;
      visibleAwakeTrees++;
      visibleAwakeShards++;
      const dof = this.map.shardDofAdr[s.slotIndex];
      const vShdSi = Math.hypot(qvel[dof + 0], qvel[dof + 1], qvel[dof + 2]);
      if (vShdSi > maxVisibleShardSpeedSi) maxVisibleShardSpeedSi = vShdSi;
    }

    return {
      visibleAwakeTrees,
      visibleAwakeBlocks,
      visibleAwakeShards,
      maxVisibleBlockSpeedSi,
      maxVisibleShardSpeedSi,
      activeFallingBlocks,
    };
  }

  /**
   * Runs immediately BEFORE `mj_step` (which executes `mj_sleep` inside `mj_step1` prior to Euler integration)
   * so grounded `zeroSnap` shards/blocks and hidden-shelf parked bodies enter `mj_sleep` with exact zero velocity
   * and transition cleanly into native sleeping islands (`d.tree_asleep >= 0`).
   */
  private preStepClampSettledVelocities(): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d || d.ntree_awake === 0) return;

    const qpos = d.qpos;
    const qvel = d.qvel;
    const qacc = d.qacc;
    const qaccWarm = d.qacc_warmstart;
    const qfrcConstraint = d.qfrc_constraint;
    const bodyMass = m.body_mass;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const geomContype = m.geom_contype;

    // 1. Settle any awake underground parked bodies (`y < -10 m`) on `parking_shelf` in O(awakeParked) time
    for (let i = this.awakeParkedBodies.length - 1; i >= 0; i--) {
      const entry = this.awakeParkedBodies[i];
      const treeId = bodyTreeId[entry.bodyId];
      if (
        (treeId >= 0 && treeAsleep[treeId] >= 0) ||
        geomContype[m.body_geomadr[entry.bodyId]] !== 0 ||
        qpos[entry.qposAdr + 1] > -10.0
      ) {
        this.awakeParkedBodies.splice(i, 1);
        continue;
      }
      const dof = entry.dofAdr;
      for (let k = 0; k < 6; k++) {
        qvel[dof + k] = 0;
        qacc[dof + k] = 0;
        qaccWarm[dof + k] = 0;
      }
    }

    // 2. Keep grounded `settledAsleep` tabletop shards & blocks (`y < 2.8 cm` with active upward constraint support)
    // pinned at `0` velocity entering `mj_step1` so native `mj_sleep` sleeps their tree without post-Euler drift
    const shardRimM = (TABLE_RADIUS + 0.008) * M_PER_DM;
    const shardRimM2 = shardRimM * shardRimM;
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const info = this.worldCompat.getMapping(s.bodyHandle);
      if (!info || !info.settledAsleep) continue;
      const treeId = bodyTreeId[s.bodyHandle];
      if (treeId >= 0 && treeAsleep[treeId] >= 0) continue;
      const qposAdr = this.map.shardQposAdr[s.slotIndex];
      const dofAdr = this.map.shardDofAdr[s.slotIndex];
      const xM = qpos[qposAdr + 0];
      const yM = qpos[qposAdr + 1];
      const zM = qpos[qposAdr + 2];
      if (
        yM > -0.008 &&
        yM < 0.026 &&
        xM * xM + zM * zM <= shardRimM2 &&
        qfrcConstraint[dofAdr + 1] > bodyMass[s.bodyHandle] * 1.2
      ) {
        const vSi = Math.hypot(qvel[dofAdr + 0], qvel[dofAdr + 1], qvel[dofAdr + 2]);
        const wSi = Math.hypot(qvel[dofAdr + 3], qvel[dofAdr + 4], qvel[dofAdr + 5]);
        if (vSi < 0.025 && wSi < 0.45) {
          for (let k = 0; k < 6; k++) {
            qvel[dofAdr + k] = 0;
            qacc[dofAdr + k] = 0;
            qaccWarm[dofAdr + k] = 0;
          }
        }
      }
    }
  }

  private hasNonZeroAppliedWakeForces(): boolean {
    if (!this.data) return false;
    if (this.data.eq_active[this.map.grabEqualityId] !== 0) return true;
    const qf = this.data.qfrc_applied;
    for (let i = 0; i < qf.length; i++) {
      if (qf[i] !== 0) return true;
    }
    return false;
  }

  private syncTransformsAndRules(): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return;
    const qpos = d.qpos;
    const qvel = d.qvel;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const geomContype = m.geom_contype;
    const geomConaffinity = m.geom_conaffinity;

    let maxTowerY = 0;
    let maxStructuralY = 0;
    let maxActiveLayer = 0;
    let sumX = 0;
    let sumZ = 0;
    let countInTower = 0;
    let standingCount = 0;
    let tumbledOrDroppedCount = 0;
    let maxDynamicTipTiltRad = 0;
    let rawMaxTiltRad = 0;
    let rawSumTiltRad = 0;
    let rawSumX = 0;
    let rawSumZ = 0;
    let maxBlockLinSpeed = 0;
    let sumTiltRad = 0;
    let sumWobbleOmega = 0;
    const standingTilts = this.standingTiltsScratch;
    const courseUpright = this.courseUprightScratch;
    courseUpright.fill(0);
    const uprightMaxTiltRad = (30.0 * Math.PI) / 180.0;

    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered) continue;
      const mapping = this.worldCompat.getMapping(b.bodyHandle);
      if (!mapping || !mapping.enabled) continue;

      const adr = mapping.qposAdr;
      const tx = qpos[adr + 0] * DM_PER_M;
      const ty = qpos[adr + 1] * DM_PER_M;
      const tz = qpos[adr + 2] * DM_PER_M;
      const rx = qpos[adr + 4];
      const ry = qpos[adr + 5];
      const rz = qpos[adr + 6];
      const rw = qpos[adr + 3];

      const treeId = bodyTreeId[b.bodyHandle];
      const asleep = mapping.settledAsleep || (treeId >= 0 && treeAsleep[treeId] >= 0);

      let lvx = 0, lvy = 0, lvz = 0;
      let avx = 0, avy = 0, avz = 0;
      if (!asleep) {
        const dof = mapping.dofAdr;
        lvx = qvel[dof + 0] * FROUDE_VEL_SCALE;
        lvy = qvel[dof + 1] * FROUDE_VEL_SCALE;
        lvz = qvel[dof + 2] * FROUDE_VEL_SCALE;
        const wWorldSi = quatRotateVec3([rx, ry, rz, rw], [qvel[dof + 3], qvel[dof + 4], qvel[dof + 5]]);
        avx = wWorldSi[0] * INV_FROUDE_VEL_SCALE;
        avy = wWorldSi[1] * INV_FROUDE_VEL_SCALE;
        avz = wWorldSi[2] * INV_FROUDE_VEL_SCALE;
      }

      if (
        Math.hypot(tx, tz) > TABLE_RADIUS + 0.06 &&
        geomConaffinity[b.colliderHandle] !== MJ_CONAFFINITY_BLOCK_OFF_RIM
      ) {
        setBodyGeomCollisionMask(
          m,
          b.bodyHandle,
          b.colliderHandle,
          geomContype[b.colliderHandle],
          MJ_CONAFFINITY_BLOCK_OFF_RIM
        );
        this.worldCompat.wakeBody(b.bodyHandle);
      }

      if (ty < -2.4) {
        this.shatterBlock(b, [0, 0, 0], 8.0, [0, 1, 0], false);
        continue;
      }

      b.position[0] = tx; b.position[1] = ty; b.position[2] = tz;
      b.rotation[0] = rx; b.rotation[1] = ry; b.rotation[2] = rz; b.rotation[3] = rw;
      b.linVel[0] = lvx; b.linVel[1] = lvy; b.linVel[2] = lvz;
      b.angVel[0] = avx; b.angVel[1] = avy; b.angVel[2] = avz;

      if (!b.extracted && b.id !== this.activeBlockId) {
        const horizR = Math.hypot(tx, tz);
        const upVec = quatRotateVec3(b.rotation, [0, 1, 0]);
        const tilt = Math.acos(Math.max(-1, Math.min(1, upVec[1])));
        const omegaHoriz = Math.hypot(avx, avz);
        const linSpeed = Math.hypot(lvx, lvy, lvz);
        const nominalY = (b.assignedLayer + 0.5) * BLOCK_HEIGHT;
        const isPancaked = b.assignedLayer >= 2 && nominalY - ty > BLOCK_HEIGHT * 1.45;

        // Continuous Unsupported-Sleep Invariant on elevated intact blocks (`ty > 2.6 cm` & lowest OBB point > `3 mm`):
        if (asleep && ty > 0.26) {
          const exVec = quatRotateVec3(b.rotation, [1, 0, 0]);
          const ezVec = quatRotateVec3(b.rotation, [0, 0, 1]);
          const extY =
            Math.abs(exVec[1]) * BLOCK_HALF_X +
            Math.abs(upVec[1]) * BLOCK_HALF_Y +
            Math.abs(ezVec[1]) * BLOCK_HALF_Z;
          if (ty - extY > 0.03) {
            let hasBlockSupport = false;
            for (let j = 0; j < this.blocks.length; j++) {
              const sb = this.blocks[j];
              if (sb.id === b.id || sb.shattered || sb.extracted) continue;
              const sMap = this.worldCompat.getMapping(sb.bodyHandle);
              if (!sMap || !sMap.enabled) continue;
              const sx = qpos[sMap.qposAdr + 0] * DM_PER_M;
              const sy = qpos[sMap.qposAdr + 1] * DM_PER_M;
              const sz = qpos[sMap.qposAdr + 2] * DM_PER_M;
              const dy = ty - sy;
              if (dy > -0.45 && dy < 0.85 && Math.abs(tx - sx) < 0.85 && Math.abs(tz - sz) < 0.85) {
                hasBlockSupport = true;
                break;
              }
            }
            if (!hasBlockSupport) {
              for (let j = 0; j < this.shards.length; j++) {
                const os = this.shards[j];
                if (geomContype[os.colliderHandle] !== MJ_CONTYPE_HERO_SHARD) continue;
                const osAdr = this.map.shardQposAdr[os.slotIndex];
                const ox = qpos[osAdr + 0] * DM_PER_M;
                const oy = qpos[osAdr + 1] * DM_PER_M;
                const oz = qpos[osAdr + 2] * DM_PER_M;
                const dy = ty - oy;
                if (dy > 0.02 && dy < 0.65 && Math.abs(tx - ox) < 0.72 && Math.abs(tz - oz) < 0.72) {
                  hasBlockSupport = true;
                  break;
                }
              }
            }
            if (!hasBlockSupport) {
              mapping.settledAsleep = false;
              mapping.quietTicks = 0;
              this.worldCompat.wakeBody(b.bodyHandle);
            }
          }
        }

        if (horizR < 0.85 && ty > 0.04) {
          countInTower++;
          rawSumX += tx;
          rawSumZ += tz;
          rawSumTiltRad += tilt;
          if (tilt > rawMaxTiltRad) rawMaxTiltRad = tilt;
          if (ty + BLOCK_HALF_Y > maxTowerY) maxTowerY = ty + BLOCK_HALF_Y;
          sumWobbleOmega += omegaHoriz;
          if (linSpeed > maxBlockLinSpeed) maxBlockLinSpeed = linSpeed;

          // Track instantaneous dynamic tipping during active collapse/overhang turnover (`omega > 0.30` or `v > 0.22`)
          if (
            (omegaHoriz >= 0.30 || linSpeed >= 0.22) &&
            ty >= Math.max(0.04, (b.assignedLayer - 1.6) * BLOCK_HEIGHT) &&
            tilt > maxDynamicTipTiltRad
          ) {
            maxDynamicTipTiltRad = tilt;
          }

          if (horizR < 0.78 && tilt <= uprightMaxTiltRad && !isPancaked) {
            sumX += tx;
            sumZ += tz;
            sumTiltRad += tilt;
            if (standingCount < standingTilts.length) {
              standingTilts[standingCount] = tilt;
            }
            standingCount++;
            if (ty + BLOCK_HALF_Y > maxStructuralY) maxStructuralY = ty + BLOCK_HALF_Y;
            if (
              ty >= Math.max(0.03, (b.assignedLayer - 0.85) * BLOCK_HEIGHT) &&
              b.assignedLayer > maxActiveLayer
            ) {
              maxActiveLayer = b.assignedLayer;
            }
            const physCourse = Math.max(0, Math.min(63, Math.floor(ty / BLOCK_HEIGHT)));
            courseUpright[physCourse]++;
          } else {
            tumbledOrDroppedCount++;
          }
        } else {
          tumbledOrDroppedCount++;
        }
      }
    }

    for (let i = this.shards.length - 1; i >= 0; i--) {
      const s = this.shards[i];
      const adr = this.map.shardQposAdr[s.slotIndex];
      const tx = qpos[adr + 0] * DM_PER_M;
      const ty = qpos[adr + 1] * DM_PER_M;
      const tz = qpos[adr + 2] * DM_PER_M;
      if (Math.hypot(tx, tz) > TABLE_RADIUS + 0.01) {
        const ctype = geomContype[s.colliderHandle];
        const caff =
          ctype === MJ_CONTYPE_HERO_SHARD
            ? MJ_CONAFFINITY_HERO_SHARD_OFF_RIM
            : this.getCulletConaffinity(true);
        if (geomConaffinity[s.colliderHandle] !== caff) {
          setBodyGeomCollisionMask(m, s.bodyHandle, s.colliderHandle, ctype, caff);
          this.worldCompat.wakeBody(s.bodyHandle);
        }
      }
      if (ty < -3.2 || Math.hypot(tx, tz) > 8.5) {
        this.bodyToShard.delete(s.bodyHandle);
        this.slotRenderVertices[s.slotIndex] = null;
        this.shardAirborneSubsteps[s.slotIndex] = 0;
        this.freeShardSlots.push(s.slotIndex);
        this.parkBodyOnHiddenShelf(
          s.bodyHandle,
          s.colliderHandle,
          adr,
          this.map.shardDofAdr[s.slotIndex],
          s.slotIndex
        );
        this.shards.splice(i, 1);
        continue;
      }
      const sRot: Quat = [qpos[adr + 4], qpos[adr + 5], qpos[adr + 6], qpos[adr + 3]];
      s.position[0] = tx; s.position[1] = ty; s.position[2] = tz;
      s.rotation[0] = sRot[0]; s.rotation[1] = sRot[1]; s.rotation[2] = sRot[2]; s.rotation[3] = sRot[3];
      const sTree = bodyTreeId[s.bodyHandle];
      const sInfo = this.worldCompat.getMapping(s.bodyHandle);
      const shardAsleep = (sInfo?.settledAsleep ?? false) || (sTree >= 0 && treeAsleep[sTree] >= 0);

      // Continuous Unsupported-Sleep Invariant on elevated shards (`ty > 2.6 cm`):
      // - Table-only cullet shards (`conaffinity & BLOCK === 0`) cannot rest on blocks, so wake immediately if `ty > 2.6 cm`.
      // - Solid shards (`HERO_SHARD` or solid `0x22` cullet) sleeping above `2.6 cm` must have an active intact block
      //   (or lower hero shard for hero shards) beneath them; otherwise wake immediately!
      if (shardAsleep && ty > 0.26) {
        const isHero = geomContype[s.colliderHandle] === MJ_CONTYPE_HERO_SHARD;
        const collidesWithBlocks = isHero || (geomConaffinity[s.colliderHandle] & MJ_CONTYPE_BLOCK) !== 0;
        let hasSupportBeneath = false;
        if (collidesWithBlocks) {
          for (let j = 0; j < this.blocks.length; j++) {
            const sb = this.blocks[j];
            if (sb.shattered || sb.extracted) continue;
            const dy = ty - sb.position[1];
            if (
              dy > -0.10 &&
              dy < 0.54 &&
              Math.abs(tx - sb.position[0]) < 0.54 &&
              Math.abs(tz - sb.position[2]) < 0.54
            ) {
              hasSupportBeneath = true;
              break;
            }
          }
          if (!hasSupportBeneath && isHero) {
            for (let j = 0; j < this.shards.length; j++) {
              const os = this.shards[j];
              if (os.id === s.id || geomContype[os.colliderHandle] !== MJ_CONTYPE_HERO_SHARD) continue;
              const osAdr = this.map.shardQposAdr[os.slotIndex];
              const ox = qpos[osAdr + 0] * DM_PER_M;
              const oy = qpos[osAdr + 1] * DM_PER_M;
              const oz = qpos[osAdr + 2] * DM_PER_M;
              const dy = ty - oy;
              if (dy > 0.015 && dy < 0.30 && Math.abs(tx - ox) < 0.30 && Math.abs(tz - oz) < 0.30) {
                hasSupportBeneath = true;
                break;
              }
            }
          }
        }
        if (!hasSupportBeneath) {
          this.shardQuietSubsteps[s.slotIndex] = 0;
          this.shardNoSupportSubsteps[s.slotIndex] = 0;
          if (sInfo) {
            sInfo.settledAsleep = false;
            sInfo.quietTicks = 0;
          }
          this.worldCompat.wakeBody(s.bodyHandle);
        }
      }

      if (sTree >= 0 && treeAsleep[sTree] >= 0) {
        s.preLinVel[0] = 0; s.preLinVel[1] = 0; s.preLinVel[2] = 0;
        s.preAngVel[0] = 0; s.preAngVel[1] = 0; s.preAngVel[2] = 0;
      } else {
        const sDof = this.map.shardDofAdr[s.slotIndex];
        s.preLinVel[0] = qvel[sDof + 0] * FROUDE_VEL_SCALE;
        s.preLinVel[1] = qvel[sDof + 1] * FROUDE_VEL_SCALE;
        s.preLinVel[2] = qvel[sDof + 2] * FROUDE_VEL_SCALE;
        const sw = quatRotateVec3(sRot, [qvel[sDof + 3], qvel[sDof + 4], qvel[sDof + 5]]);
        s.preAngVel[0] = sw[0] * INV_FROUDE_VEL_SCALE;
        s.preAngVel[1] = sw[1] * INV_FROUDE_VEL_SCALE;
        s.preAngVel[2] = sw[2] * INV_FROUDE_VEL_SCALE;
      }
    }

    let contiguousCourses = 0;
    let emptyGap = 0;
    for (let c = 0; c < 64; c++) {
      if (courseUpright[c] > 0) {
        contiguousCourses = c + 1;
        emptyGap = 0;
      } else {
        emptyGap++;
        if (emptyGap >= 2) break;
      }
    }

    if (countInTower > 0) {
      const rawMaxTiltDeg = (rawMaxTiltRad * 180) / Math.PI;
      const rawMeanTiltDeg = ((rawSumTiltRad / countInTower) * 180) / Math.PI;
      const rawComLeanDeg =
        maxTowerY > 0.25
          ? (Math.atan2(Math.hypot(rawSumX / countInTower, rawSumZ / countInTower), maxTowerY * 0.55) * 180) / Math.PI
          : rawMeanTiltDeg;
      const rawSmallLean = Math.min(rawMeanTiltDeg, rawComLeanDeg * 0.65 + rawMeanTiltDeg * 0.35);
      const rawTipBlend = Math.max(0.0, Math.min(1.0, (rawMaxTiltDeg - 2.35) / 3.0));
      const rawInstantLean = rawSmallLean * (1.0 - rawTipBlend) + rawMaxTiltDeg * rawTipBlend;
      this.solverLeanAngleDeg = this.solverLeanAngleDeg * 0.75 + Math.min(45, rawInstantLean) * 0.25;
    } else {
      this.solverLeanAngleDeg = 0;
    }

    const isSandboxCollapsedHeap =
      this.gameMode === 'sandbox' &&
      this.shatteredCount + tumbledOrDroppedCount >= Math.max(8, Math.floor(this.blocks.length * 0.45)) &&
      standingCount <= Math.max(9, Math.floor(this.blocks.length * 0.32)) &&
      contiguousCourses <= 4;

    if (isSandboxCollapsedHeap) {
      this.centerOfMassXZCm = [0, 0];
      // Keep physical toppled state (`leanAngleDeg = 45 deg`, matching Solo `COLLAPSED` trajectory invariants)
      // while setting `currentLayers = 0` so the HUD Plumb & Elevation cards render `— Toppled`!
      this.leanAngleDeg = Math.max(this.leanAngleDeg, 45.0);
      const avgOmegaDeg =
        countInTower > 0 ? (((sumWobbleOmega * SIM_TIME_SCALE) / countInTower) * 180) / Math.PI : 0;
      this.wobbleSpeedDegPerSec = this.wobbleSpeedDegPerSec * 0.7 + avgOmegaDeg * 0.3;
      this.currentLayers = 0;
      this.liveHeightCm = maxTowerY * 10.0;
    } else if (standingCount > 0 || countInTower > 0) {
      const denom = Math.max(1, standingCount);
      if (standingCount > 0) {
        this.centerOfMassXZCm = [(sumX / denom) * 10, (sumZ / denom) * 10];
      }
      // Sort `standingTilts[0..standingCount-1]` in-place (<= 96 floats, insertion sort) to ignore isolated
      // crooked loose top-notch blocks while keeping full sensitivity to >= 3-block course leans & active tipping!
      const tiltLen = Math.min(standingCount, standingTilts.length);
      for (let i = 1; i < tiltLen; i++) {
        const v = standingTilts[i];
        let j = i - 1;
        while (j >= 0 && standingTilts[j] > v) {
          standingTilts[j + 1] = standingTilts[j];
          j--;
        }
        standingTilts[j + 1] = v;
      }
      const restingConsensusTiltRad =
        tiltLen >= 12
          ? standingTilts[tiltLen - 5]
          : tiltLen >= 6
            ? standingTilts[tiltLen - 3]
            : tiltLen > 0
              ? standingTilts[tiltLen - 1]
              : 0;
      const effectiveMaxTiltRad = Math.max(restingConsensusTiltRad, maxDynamicTipTiltRad);
      const maxTiltDeg = (effectiveMaxTiltRad * 180) / Math.PI;
      const meanTiltDeg = standingCount > 0 ? ((sumTiltRad / denom) * 180) / Math.PI : maxTiltDeg;
      const effHeightDm = maxStructuralY > 0.25 ? maxStructuralY : maxTowerY;
      const comLeanDeg =
        standingCount > 0 && effHeightDm > 0.25
          ? (Math.atan2(Math.hypot(sumX / denom, sumZ / denom), effHeightDm * 0.55) * 180) / Math.PI
          : meanTiltDeg;
      const smallAngleLean = Math.min(meanTiltDeg, comLeanDeg * 0.65 + meanTiltDeg * 0.35);
      const tipBlend = Math.max(0.0, Math.min(1.0, (maxTiltDeg - 2.35) / 3.0));
      const instantLean = smallAngleLean * (1.0 - tipBlend) + maxTiltDeg * tipBlend;
      this.leanAngleDeg = this.leanAngleDeg * 0.75 + Math.min(45, instantLean) * 0.25;
      const avgOmegaDeg =
        countInTower > 0 ? (((sumWobbleOmega * SIM_TIME_SCALE) / countInTower) * 180) / Math.PI : 0;
      this.wobbleSpeedDegPerSec = this.wobbleSpeedDegPerSec * 0.7 + avgOmegaDeg * 0.3;

      this.currentLayers = standingCount > 0 ? maxActiveLayer + 1 : 0;
      const heightCm = (standingCount > 0 ? maxStructuralY : maxTowerY) * 10.0;
      this.liveHeightCm = heightCm;
      if (this.turnPhase !== 'COLLAPSED') {
        if (heightCm > this.peakHeightCm) this.peakHeightCm = heightCm;
        if (this.currentLayers > this.peakLayers) this.peakLayers = this.currentLayers;
      }
    } else {
      this.currentLayers = 0;
      this.liveHeightCm = 0;
      this.leanAngleDeg = 0;
    this.solverLeanAngleDeg = 0;
      this.wobbleSpeedDegPerSec = 0;
    }

    this.computeOpenTopSlots();
    const forbiddenMinLayer = Math.max(0, this.topSupportFrame.targetLayer - 1);
    this.lockedMinRow = this.gameMode === 'sandbox' ? null : forbiddenMinLayer + 1;

    for (const b of this.blocks) {
      b.isLegalTarget =
        !b.shattered &&
        !b.extracted &&
        (this.gameMode === 'sandbox' || b.assignedLayer < forbiddenMinLayer);
    }
  }

  private getLayerSupportCenter(layer: number): [number, number] {
    let sx = 0, sz = 0, n = 0;
    const refLayer = Math.max(0, layer - 1);
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.assignedLayer === refLayer) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.9) {
          sx += b.position[0];
          sz += b.position[2];
          n++;
        }
      }
    }
    return n > 0 ? [sx / n, sz / n] : [0, 0];
  }

  private buildSupportFrameForLayer(targetLayer: number): TopSupportFrame {
    const supLayer = targetLayer - 1;
    const supIsOdd = supLayer % 2 === 1;
    const nomSupAxis: Vec3 = supIsOdd ? [0, 0, -1] : [1, 0, 0];

    let supX = 0, supY = 0, supZ = 0, supCount = 0;
    let nSumX = 0, nSumY = 0, nSumZ = 0;
    let axSumX = 0, axSumZ = 0;

    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId && b.assignedLayer === supLayer) {
        if (Math.hypot(b.position[0], b.position[2]) < 1.0) {
          const bUp = quatRotateVec3(b.rotation, [0, 1, 0]);
          const upSgn = bUp[1] >= 0 ? 1 : -1;
          const upV: Vec3 = [bUp[0] * upSgn, bUp[1] * upSgn, bUp[2] * upSgn];
          supX += b.position[0] + upV[0] * BLOCK_HALF_Y;
          supY += b.position[1] + upV[1] * BLOCK_HALF_Y;
          supZ += b.position[2] + upV[2] * BLOCK_HALF_Y;
          nSumX += upV[0]; nSumY += upV[1]; nSumZ += upV[2];

          const bLen = quatRotateVec3(b.rotation, [1, 0, 0]);
          const lenSgn = bLen[0] * nomSupAxis[0] + bLen[2] * nomSupAxis[2] >= 0 ? 1 : -1;
          axSumX += bLen[0] * lenSgn;
          axSumZ += bLen[2] * lenSgn;
          supCount++;
        }
      }
    }

    const cx = supCount > 0 ? supX / supCount : 0;
    const cy = supCount > 0 ? supY / supCount : targetLayer * BLOCK_HEIGHT;
    const cz = supCount > 0 ? supZ / supCount : 0;

    let upNormal: Vec3 = [0, 1, 0];
    if (supCount > 0) {
      const nLen = Math.hypot(nSumX, nSumY, nSumZ) || 1;
      upNormal = [nSumX / nLen, nSumY / nLen, nSumZ / nLen];
    }

    const isOdd = targetLayer % 2 === 1;
    let yawRad = isOdd ? Math.PI * 0.5 : 0.0;
    if (supCount > 0 && Math.hypot(axSumX, axSumZ) > 1e-4) {
      const supYaw = Math.atan2(-axSumZ, axSumX);
      yawRad = isOdd ? supYaw + Math.PI * 0.5 : supYaw - Math.PI * 0.5;
    }

    const tiltAxisX = upNormal[2];
    const tiltAxisZ = -upNormal[0];
    const tiltSin = Math.hypot(tiltAxisX, tiltAxisZ);
    let tiltQuat: Quat = [0, 0, 0, 1];
    if (tiltSin > 1e-5) {
      const tiltAngle = Math.atan2(tiltSin, upNormal[1]);
      tiltQuat = quatFromAxisAngle([tiltAxisX / tiltSin, 0, tiltAxisZ / tiltSin], tiltAngle);
    }

    const yawQuat = quatFromAxisAngle([0, 1, 0], yawRad);
    const defaultRotation = MujocoPhysicsEngine.quatMul(tiltQuat, yawQuat);
    const lengthAxis = quatRotateVec3(defaultRotation, [1, 0, 0]);
    const spacingAxis = quatRotateVec3(defaultRotation, [0, 0, 1]);
    const liftAlongNormal = BLOCK_HALF_Y + 0.002;

    return {
      targetLayer,
      surfaceCenter: [cx, cy, cz],
      layerCenter: [
        cx + upNormal[0] * liftAlongNormal,
        cy + upNormal[1] * liftAlongNormal,
        cz + upNormal[2] * liftAlongNormal,
      ],
      upNormal,
      tiltQuat,
      defaultYawRad: yawRad,
      defaultRotation,
      lengthAxis,
      spacingAxis,
    };
  }

  private computeOpenTopSlots(): void {
    let highestLayer = 0;
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.95 && b.assignedLayer > highestLayer) {
          highestLayer = b.assignedLayer;
        }
      }
    }

    const blocksInHighest: GlassBlockState[] = [];
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId && b.assignedLayer === highestLayer) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.95) {
          blocksInHighest.push(b);
        }
      }
    }

    const colHx = BLOCK_HALF_X - 0.0008;
    const colHz = BLOCK_HALF_Z - 0.0014;
    const roomClearance = 0.0040;

    const candidateFits = (
      frame: TopSupportFrame,
      obstacles: GlassBlockState[],
      uOffset: number
    ): boolean => {
      if (obstacles.length === 0) return true;
      const px = frame.layerCenter[0] + frame.spacingAxis[0] * uOffset;
      const pz = frame.layerCenter[2] + frame.spacingAxis[2] * uOffset;
      const { uX: uAx, uZ: uAz } = MujocoPhysicsEngine.getBlockXZAxes(frame.defaultRotation);
      for (let i = 0; i < obstacles.length; i++) {
        const ob = obstacles[i];
        const obAxes = MujocoPhysicsEngine.getBlockXZAxes(ob.rotation);
        const ov = MujocoPhysicsEngine.obb2DOverlapDepth(
          px, pz, uAx, uAz, colHx + roomClearance, colHz + roomClearance,
          ob.position[0], ob.position[2], obAxes.uX, obAxes.uZ, colHx, colHz,
          frame.layerCenter[0], frame.layerCenter[2]
        );
        if (ov && ov.overlap > 1e-5) return false;
      }
      return true;
    };

    let targetLayer = highestLayer;
    let highestFrame: TopSupportFrame | null = null;
    let fallbackOpenU: number | null = null;

    if (blocksInHighest.length >= 3) {
      targetLayer = highestLayer + 1;
    } else if (blocksInHighest.length > 0) {
      highestFrame = this.buildSupportFrameForLayer(highestLayer);
      let hasRoom = false;
      for (let k = -23; k <= 23; k++) {
        const uTest = k * 0.015;
        if (candidateFits(highestFrame, blocksInHighest, uTest)) {
          hasRoom = true;
          if (fallbackOpenU === null || Math.abs(uTest) < Math.abs(fallbackOpenU)) {
            fallbackOpenU = uTest;
          }
        }
      }
      if (!hasRoom) targetLayer = highestLayer + 1;
    }

    const frame =
      targetLayer === highestLayer && highestFrame !== null
        ? highestFrame
        : this.buildSupportFrameForLayer(targetLayer);
    this.topSupportFrame = frame;

    const activeObstacles = targetLayer === highestLayer ? blocksInHighest : [];
    const slots: TopPlacementSlot[] = [];
    for (let slotIndex = 0; slotIndex < 3; slotIndex++) {
      const offset = (slotIndex - 1) * BLOCK_WIDTH;
      if (!candidateFits(frame, activeObstacles, offset)) continue;
      const px = frame.layerCenter[0] + frame.spacingAxis[0] * offset;
      const py = frame.layerCenter[1] + frame.spacingAxis[1] * offset;
      const pz = frame.layerCenter[2] + frame.spacingAxis[2] * offset;
      slots.push({
        layer: targetLayer,
        slotIndex,
        position: [px, py, pz],
        rotation: frame.defaultRotation,
        yawRad: frame.defaultYawRad,
      });
    }

    if (slots.length === 0 && fallbackOpenU !== null) {
      const slotIndex = Math.max(0, Math.min(2, Math.round(fallbackOpenU / BLOCK_WIDTH + 1)));
      slots.push({
        layer: targetLayer,
        slotIndex,
        position: [
          frame.layerCenter[0] + frame.spacingAxis[0] * fallbackOpenU,
          frame.layerCenter[1] + frame.spacingAxis[1] * fallbackOpenU,
          frame.layerCenter[2] + frame.spacingAxis[2] * fallbackOpenU,
        ],
        rotation: frame.defaultRotation,
        yawRad: frame.defaultYawRad,
      });
    }

    this.openTopSlots = slots;
  }

  private updateTurnStateMachine(dt: number): void {
    if (this.gameMode === 'sandbox') return;
    if (this.turnPhase === 'COLLAPSED') {
      if (this.shatteredCount > 0) {
        this.collapseReason =
          'The stained-glass tower gave way and shattered across the oak table!';
      }
      return;
    }
    if (this.simTime < 0.25) return;

    let fallenBlocks = this.shatteredCount;
    for (const b of this.blocks) {
      if (b.shattered || b.extracted || b.id === this.activeBlockId) continue;
      // If a test or sandbox fixture explicitly disabled this body (`!rb.isEnabled()`, parked underground at y = -40 m),
      // skip it in collapse detection rather than treating it as a fallen block!
      const rb = this.worldCompat.getRigidBody(b.bodyHandle);
      if (!rb || !this.worldCompat.getMapping(b.bodyHandle)?.enabled) continue;
      const rHoriz = Math.hypot(b.position[0], b.position[2]);
      const upVec = quatRotateVec3(b.rotation, [0, 1, 0]);
      const tiltDeg = (Math.acos(Math.max(-1, Math.min(1, upVec[1]))) * 180) / Math.PI;
      const nominalY = (b.assignedLayer + 0.5) * BLOCK_HEIGHT;

      const droppedToTable = b.assignedLayer >= 1 && b.position[1] < BLOCK_HEIGHT * 0.92;
      const layerPancaked = b.assignedLayer >= 2 && nominalY - b.position[1] > BLOCK_HEIGHT * 1.35;
      const layer0Ejected = b.assignedLayer === 0 && rHoriz > 0.88;
      const tumbledOut = rHoriz > 1.2 || tiltDeg > 42.0;

      if (droppedToTable || layerPancaked || layer0Ejected || tumbledOut) {
        fallenBlocks++;
      }
    }

    if (fallenBlocks > 0) {
      this.turnPhase = 'COLLAPSED';
      this.isPointerDraggingBlock = false;
      if (this.data) {
        this.data.eq_active[this.map.grabEqualityId] = 0;
      }
      if (this.activeBlockId !== null) {
        const carried = this.blocks[this.activeBlockId];
        if (carried && !carried.shattered) {
          carried.extracted = false;
          this.worldCompat.getCollider(carried.colliderHandle)?.setSensor(false);
          this.worldCompat.getRigidBody(carried.bodyHandle)?.setGravityScale(1.0, true);
        }
        this.activeBlockId = null;
      }
      this.collapseReason =
        this.shatteredCount > 0
          ? 'The stained-glass tower gave way and shattered across the oak table!'
          : 'The tower lost equilibrium and tumbled!';
      return;
    }

    if (this.turnPhase === 'SETTLING') {
      this.settleTimer += dt;
      const towerQuiet = this.wobbleSpeedDegPerSec < 2.2;
      if (this.settleTimer >= this.SETTLE_DURATION || (this.settleTimer > 0.65 && towerQuiet)) {
        this.blocksStacked++;
        this.audio.playStackSuccess(this.currentLayers);
        if (this.gameMode === 'party') {
          this.currentPlayer = (this.currentPlayer + 1) % this.playerCount;
        }
        this.turnPhase = 'SELECT_BLOCK';
        this.settleTimer = 0;
      }
    }
  }

  /**
   * Native WASM `mj_ray` 3D scene raycast across exact 24-vertex beveled glass blocks (`group 0`),
   * 3D Voronoi shard convex hulls (`group 1`), and circular oak table (`group 2`), skipping parked
   * shelf bodies (`group 3`) in O(1). Zero live state mutation.
   */
  public raycastScene(
    rayOriginDm: Vec3,
    rayDir: Vec3,
    options?: SceneRaycastOptions
  ): SceneRaycastHit | null {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return null;
    const dirLen = Math.hypot(rayDir[0], rayDir[1], rayDir[2]);
    if (dirLen < 1e-9 || !Number.isFinite(dirLen)) return null;
    const ux = rayDir[0] / dirLen;
    const uy = rayDir[1] / dirLen;
    const uz = rayDir[2] / dirLen;

    if (!this.scratchOracleData) {
      this.scratchOracleData = new this.mj.MjData(m);
      attachMujocoThreadPoolIfAvailable(this.scratchOracleData);
    }
    const rayData = this.scratchOracleData;
    this.mj.mj_copyState(m, d, rayData, this.mj.mjtState.mjSTATE_QPOS.value);
    this.mj.mj_kinematics(m, rayData);

    if (!this.rayGeomIdBuf) {
      this.rayGeomIdBuf = new this.mj.IntBuffer(1);
    }
    if (!this.rayNormalBuf) {
      this.rayNormalBuf = new this.mj.DoubleBuffer(3);
    }
    const gView = this.rayGeomIdBuf.GetView();
    gView[0] = -1;

    const includeBlocks = options?.includeBlocks !== false;
    const includeShards = options?.includeShards !== false;
    const includeTable = options?.includeTable !== false;
    const geomGroupMask = [
      includeBlocks ? 1 : 0,
      includeShards ? 1 : 0,
      includeTable ? 1 : 0,
      0,
      0,
      0,
    ];
    const pntM = [
      rayOriginDm[0] * M_PER_DM,
      rayOriginDm[1] * M_PER_DM,
      rayOriginDm[2] * M_PER_DM,
    ];
    const dirUnit = [ux, uy, uz];
    const distM: number = this.mj.mj_ray(
      m,
      rayData,
      pntM,
      dirUnit,
      geomGroupMask,
      includeTable,
      options?.excludeBodyId ?? -1,
      this.rayGeomIdBuf,
      this.rayNormalBuf
    );
    if (!Number.isFinite(distM) || distM < 0) return null;
    const distanceDm = distM * DM_PER_M;
    if (options?.maxDistanceDm !== undefined && distanceDm > options.maxDistanceDm) {
      return null;
    }

    const geomId = this.rayGeomIdBuf.GetView()[0];
    if (geomId < 0 || geomId >= m.ngeom) return null;
    const nView = this.rayNormalBuf.GetView();
    const normal: Vec3 = [nView[0], nView[1], nView[2]];
    const worldPointDm: Vec3 = [
      rayOriginDm[0] + ux * distanceDm,
      rayOriginDm[1] + uy * distanceDm,
      rayOriginDm[2] + uz * distanceDm,
    ];
    const bodyId = m.geom_bodyid[geomId];
    const blk = this.bodyToBlock.get(bodyId);
    if (blk && !blk.shattered) {
      const localPointDm = quatInverseRotateVec3(blk.rotation, [
        worldPointDm[0] - blk.position[0],
        worldPointDm[1] - blk.position[1],
        worldPointDm[2] - blk.position[2],
      ]);
      return {
        kind: 'block',
        distanceDm,
        worldPointDm,
        localPointDm,
        normal,
        geomId,
        bodyId,
        blockId: blk.id,
      };
    }

    const shd = this.bodyToShard.get(bodyId);
    if (shd) {
      const localPointDm = quatInverseRotateVec3(shd.rotation, [
        worldPointDm[0] - shd.position[0],
        worldPointDm[1] - shd.position[1],
        worldPointDm[2] - shd.position[2],
      ]);
      return {
        kind: 'shard',
        distanceDm,
        worldPointDm,
        localPointDm,
        normal,
        geomId,
        bodyId,
        shardId: shd.id,
      };
    }

    return {
      kind: 'table',
      distanceDm,
      worldPointDm,
      localPointDm: [...worldPointDm],
      normal,
      geomId,
      bodyId,
    };
  }

  /**
   * Sandbox / tactile shard flick using exact `worldPointDm` impulse application.
   */
  public nudgeShard(
    shardId: number,
    worldPointDm: Vec3,
    worldDir: Vec3,
    impulseSim: number = 0.035
  ): boolean {
    const shd = this.shards.find((s) => s.id === shardId);
    if (!shd || !this.model || !this.data) return false;
    const rb = this.worldCompat.getRigidBody(shd.bodyHandle);
    if (!rb) return false;
    const len = Math.hypot(worldDir[0], worldDir[1], worldDir[2]);
    if (len < 1e-8) return false;
    const massRatio = Math.max(0.05, Math.min(2.5, shd.massKg / 0.004));
    const scale = (impulseSim * massRatio) / len;
    this.shardQuietSubsteps[shd.slotIndex] = 0;
    this.shardNoSupportSubsteps[shd.slotIndex] = 0;
    rb.applyImpulseAtPoint(
      {
        x: worldDir[0] * scale,
        y: Math.max(0.004 * massRatio, worldDir[1] * scale + impulseSim * 0.12 * massRatio),
        z: worldDir[2] * scale,
      },
      { x: worldPointDm[0], y: worldPointDm[1], z: worldPointDm[2] },
      true
    );
    const postNudgeE = this.computeSingleShardTotalEnergyJ(
      shd.bodyHandle,
      this.map.shardQposAdr[shd.slotIndex],
      this.map.shardDofAdr[shd.slotIndex]
    );
    this.shardInContactPrevSubstep[shd.slotIndex] = 0;
    this.shardPreSubstepTotalEnergyJ[shd.slotIndex] = postNudgeE;
    this.shardEntryTotalEnergyJ[shd.slotIndex] = postNudgeE;
    return true;
  }

  /**
   * Cumulative count of MuJoCo instability warnings (`mjWARN_BADQPOS=3`, `mjWARN_BADQVEL=4`, `mjWARN_BADQACC=5`).
   * Never calls `.delete()` on `d.warning` (embedded inside `MjData`).
   */
  public getSolverWarningCount(): number {
    const d = this.data;
    if (!d) return 0;
    const warnVec = d.warning;
    if (!warnVec) return 0;
    const wPos = warnVec.get(3)?.number ?? 0;
    const wVel = warnVec.get(4)?.number ?? 0;
    const wAcc = warnVec.get(5)?.number ?? 0;
    return wPos + wVel + wAcc;
  }

  /**
   * Exact active-scene SI mechanical energy `E_pot + E_kin` (Joules) across all non-parked,
   * non-extracted blocks and active Voronoi shards (referenced to the `-0.35 m` off-table recycle
   * floor so `-50 m` hidden pool shelf bodies never pollute passive energy monotonicity).
   */
  public computeMechanicalEnergyJ(): number {
    const m = this.model;
    const d = this.data;
    if (!m || !d) return 0;

    const qpos = d.qpos;
    const qvel = d.qvel;
    const qfrcApplied = d.qfrc_applied;
    const bodyMass = m.body_mass;
    const bodyInertia = m.body_inertia;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const gSi = 9.81;
    const yDatumM = -0.35;
    let totalJ = 0;

    const addBodyEnergy = (bodyId: number, adr: number, dof: number, settledAsleep: boolean): void => {
      const mass = bodyMass[bodyId];
      if (!(mass > 0)) return;
      const yM = Math.max(yDatumM, qpos[adr + 1]);
      totalJ += mass * gSi * (yM - yDatumM);

      if (settledAsleep) return;
      const vx = qvel[dof + 0];
      const vy = qvel[dof + 1];
      const vz = qvel[dof + 2];
      const wx = qvel[dof + 3];
      const wy = qvel[dof + 4];
      const wz = qvel[dof + 5];
      const speedSq = vx * vx + vy * vy + vz * vz + wx * wx + wy * wy + wz * wz;
      const tid = bodyTreeId[bodyId];
      if (tid >= 0 && treeAsleep[tid] >= 0 && qfrcApplied[dof + 1] === 0 && speedSq <= 1e-12) {
        return;
      }
      const ib = bodyId * 3;
      totalJ +=
        0.5 * mass * (vx * vx + vy * vy + vz * vz) +
        0.5 *
          (bodyInertia[ib + 0] * wx * wx +
            bodyInertia[ib + 1] * wy * wy +
            bodyInertia[ib + 2] * wz * wz);
    };

    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info || !info.enabled) continue;
      addBodyEnergy(b.bodyHandle, info.qposAdr, info.dofAdr, info.settledAsleep);
    }

    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const info = this.worldCompat.getMapping(s.bodyHandle);
      if (!info || !info.enabled) continue;
      addBodyEnergy(
        s.bodyHandle,
        this.map.shardQposAdr[s.slotIndex],
        this.map.shardDofAdr[s.slotIndex],
        info.settledAsleep
      );
    }

    return totalJ;
  }

  /**
   * Counterfactual Ply-2 Adversarial Top-Placement Scorer (`0` mutation of live `MjData`):
   * Evaluates placing a block onto each open top-row slot across lateral/yaw perturbations in
   * `scratchOracleData`, ranking placements that survive our own settle window while maximizing
   * CoM eccentricity & lean difficulty for the next player.
   */
  public evaluateAdversarialTopPlacement(
    blockId?: number,
    settleSteps: number = 6,
    fastSinglePerturbation: boolean = false
  ): AdversarialPlacementPlan[] {
    if (!this.model || !this.data) return [];
    if (this.openTopSlots.length === 0) return [];
    if (!this.scratchOracleData) {
      this.scratchOracleData = new this.mj.MjData(this.model);
      attachMujocoThreadPoolIfAvailable(this.scratchOracleData);
    }
    const cand =
      (blockId !== undefined ? this.blocks[blockId] : null) ??
      (this.activeBlockId !== null ? this.blocks[this.activeBlockId] : null) ??
      this.blocks.find((b) => !b.shattered && (b.extracted || b.isLegalTarget)) ??
      null;
    if (!cand) return [];
    const info = this.worldCompat.getMapping(cand.bodyHandle);
    if (!info) return [];

    const sig = this.mj.mjtState.mjSTATE_FULLPHYSICS.value;
    const perturbations: Array<{ latDm: number; yawRad: number }> = fastSinglePerturbation
      ? [{ latDm: 0.028, yawRad: 0.032 }]
      : [
          { latDm: 0.0, yawRad: 0.0 },
          { latDm: 0.032, yawRad: 0.038 },
          { latDm: -0.032, yawRad: -0.038 },
        ];

    const plans: AdversarialPlacementPlan[] = [];
    const steps = Math.max(2, Math.min(16, Math.round(settleSteps)));
    const savedIter = this.model.opt.iterations;
    const savedLsIter = this.model.opt.ls_iterations;
    this.model.opt.iterations = Math.min(savedIter, 8);
    this.model.opt.ls_iterations = Math.min(savedLsIter, 4);

    try {
      const scratch = this.scratchOracleData;
      for (const slot of this.openTopSlots) {
        const totalYawBase = slot.yawRad;
        const latDirX = -Math.sin(totalYawBase);
        const latDirZ = Math.cos(totalYawBase);

        for (const pert of perturbations) {
          this.mj.mj_copyState(this.model, this.data, scratch, sig);
          scratch.eq_active[this.map.grabEqualityId] = 0;
          const sQfrc = scratch.qfrc_applied;
          sQfrc.fill(0);
          scratch.xfrc_applied.fill(0);

          const placePosDm: Vec3 = [
            slot.position[0] + latDirX * pert.latDm,
            slot.position[1] + 0.002,
            slot.position[2] + latDirZ * pert.latDm,
          ];
          const placeYaw = totalYawBase + pert.yawRad;
          const halfYaw = placeYaw * 0.5;
          const qw = Math.cos(halfYaw);
          const qy = Math.sin(halfYaw);

          const adr = info.qposAdr;
          const dof = info.dofAdr;
          const sQposInit = scratch.qpos;
          const sQvelInit = scratch.qvel;
          sQposInit[adr + 0] = placePosDm[0] * M_PER_DM;
          sQposInit[adr + 1] = placePosDm[1] * M_PER_DM;
          sQposInit[adr + 2] = placePosDm[2] * M_PER_DM;
          sQposInit[adr + 3] = qw;
          sQposInit[adr + 4] = 0;
          sQposInit[adr + 5] = qy;
          sQposInit[adr + 6] = 0;
          for (let k = 0; k < 6; k++) {
            sQvelInit[dof + k] = 0;
          }
          // Wake only the placed block; mj_collision wakes contact support if needed
          sQfrc[dof + 1] = -1e-6;

          for (let s = 0; s < steps; s++) {
            this.mj.mj_step(this.model, scratch);
            if (s === 0) {
              scratch.qfrc_applied[dof + 1] = 0;
            }
          }

          const sQpos = scratch.qpos;
          let maxTiltDeg = 0;
          let sumX = 0;
          let sumZ = 0;
          let countStanding = 0;
          let collapsed = false;
          for (const ob of this.blocks) {
            if (ob.shattered || (ob.extracted && ob.id !== cand.id)) continue;
            const oInfo = this.worldCompat.getMapping(ob.bodyHandle);
            if (!oInfo) continue;
            const oAdr = oInfo.qposAdr;
            const bx = sQpos[oAdr + 0] * DM_PER_M;
            const by = sQpos[oAdr + 1] * DM_PER_M;
            const bz = sQpos[oAdr + 2] * DM_PER_M;
            const oqx = sQpos[oAdr + 4];
            const oqz = sQpos[oAdr + 6];
            const upY = 1.0 - 2.0 * (oqx * oqx + oqz * oqz);
            const tilt = (Math.acos(Math.max(-1, Math.min(1, upY))) * 180) / Math.PI;
            if (tilt > maxTiltDeg) maxTiltDeg = tilt;
            if (tilt > 22.0 || (ob.id === cand.id && by < slot.position[1] - 0.18)) {
              collapsed = true;
            }
            if (by > 0.04 && Math.hypot(bx, bz) < 0.85) {
              sumX += bx;
              sumZ += bz;
              countStanding++;
            }
          }

          const comMm =
            countStanding > 0
              ? Math.hypot(sumX / countStanding, sumZ / countStanding) * 100.0
              : 0;
          const selfSurvival01 = collapsed
            ? 0.0
            : Math.max(0.0, Math.min(1.0, 1.0 - Math.max(0, maxTiltDeg - 1.8) / 7.5));
          const leanStrain01 = Math.min(1.0, maxTiltDeg / 2.2);
          const comStrain01 = Math.min(1.0, comMm / 4.5);
          const eccBonus01 =
            (slot.slotIndex !== 1 ? 0.22 : 0.0) + (Math.abs(pert.latDm) > 0.01 ? 0.14 : 0.0);
          const opponentDifficulty01 = collapsed
            ? 0.0
            : Math.max(0.0, Math.min(1.0, leanStrain01 * 0.42 + comStrain01 * 0.36 + eccBonus01));
          const adversarialScore01 = collapsed
            ? 0.0
            : +(selfSurvival01 * (0.45 + 0.55 * opponentDifficulty01)).toFixed(3);

          plans.push({
            slotIndex: slot.slotIndex,
            layer: slot.layer,
            targetPosDm: [
              +placePosDm[0].toFixed(4),
              +slot.position[1].toFixed(4),
              +placePosDm[2].toFixed(4),
            ],
            lateralOffsetDm: +pert.latDm.toFixed(4),
            yawOffsetRad: +pert.yawRad.toFixed(4),
            postPlaceLeanDeg: +maxTiltDeg.toFixed(3),
            selfSurvival01: +selfSurvival01.toFixed(3),
            opponentDifficulty01: +opponentDifficulty01.toFixed(3),
            adversarialScore01,
            collapseTriggered: collapsed,
          });
        }
      }
    } finally {
      this.model.opt.iterations = savedIter;
      this.model.opt.ls_iterations = savedLsIter;
    }

    plans.sort((a, b) => b.adversarialScore01 - a.adversarialScore01);
    return plans;
  }

  /**
   * 2-Ply CEM Counterfactual `mj_copyState` Stability & Adversarial AI Oracle ("Master Glazier AI"):
   * - **Ply 1 (CEM Pull-Azimuth Search)**: Clones live `MjData` into `scratchOracleData` (`0` live mutation),
   *   screens all candidates along their longitudinal axis, and runs a 2-iteration Cross-Entropy Method search
   *   over antipodal + oblique pull azimuths on the top viable candidates to find `optimalPullAzimuthRad` /
   *   `optimalPullDirXZ` maximizing extraction safety.
   * - **Ply 2 (Adversarial Top-Placement Rollout)**: Rolls forward candidate top-row placements for the top
   *   extraction choices to compute `adversarialPlacement` and `twoPlyMasterScore01`.
   */
  public evaluateCounterfactualBlockCriticality(
    maxCandidates: number = 8,
    rolloutSteps: number = 6,
    candidateBlockIds?: number[]
  ): BlockCriticalityCandidate[] {
    if (!this.model || !this.data) return [];
    if (!this.scratchOracleData) {
      this.scratchOracleData = new this.mj.MjData(this.model);
      attachMujocoThreadPoolIfAvailable(this.scratchOracleData);
    }

    const sig = this.mj.mjtState.mjSTATE_FULLPHYSICS.value;
    let chosen: GlassBlockState[];
    if (candidateBlockIds && candidateBlockIds.length > 0) {
      chosen = candidateBlockIds
        .map((id) => this.blocks[id])
        .filter((b): b is GlassBlockState => Boolean(b && !b.shattered && !b.extracted));
    } else {
      const legal = this.blocks.filter((b) => b.isLegalTarget && !b.shattered && !b.extracted);
      legal.sort(
        (a, b) => this.computeBlockClampingLoadN(a.id) - this.computeBlockClampingLoadN(b.id)
      );
      const cap = Math.max(1, maxCandidates);
      if (cap >= 4 && legal.length > cap) {
        const lowCount = Math.max(1, Math.ceil(cap * 0.7));
        const highCount = cap - lowCount;
        chosen = [
          ...legal.slice(0, lowCount),
          ...legal.slice(Math.max(lowCount, legal.length - highCount)),
        ];
      } else {
        chosen = legal.slice(0, cap);
      }
    }

    const savedIter = this.model.opt.iterations;
    const savedLsIter = this.model.opt.ls_iterations;

    const scratch = this.scratchOracleData;
    const probePullAzimuth = (
      cand: GlassBlockState,
      info: { qposAdr: number; dofAdr: number },
      azRad: number
    ): {
      azRad: number;
      dirX: number;
      dirZ: number;
      slideMm: number;
      maxTiltDeg: number;
      collapsed: boolean;
      safetyScore01: number;
    } => {
      this.mj.mj_copyState(this.model!, this.data!, scratch, sig);
      scratch.eq_active[this.map.grabEqualityId] = 0;
      const sQfrc = scratch.qfrc_applied;
      sQfrc.fill(0);
      scratch.xfrc_applied.fill(0);

      const sQposStart = scratch.qpos;
      const startX = sQposStart[info.qposAdr + 0];
      const startZ = sQposStart[info.qposAdr + 2];
      const dirX = Math.cos(azRad);
      const dirZ = Math.sin(azRad);
      sQfrc[info.dofAdr + 0] = dirX * 1.8;
      sQfrc[info.dofAdr + 2] = dirZ * 1.8;

      for (let s = 0; s < rolloutSteps; s++) {
        this.mj.mj_step(this.model!, scratch);
      }

      const sQpos = scratch.qpos;
      const endX = sQpos[info.qposAdr + 0];
      const endZ = sQpos[info.qposAdr + 2];
      const slideMm = Math.hypot(endX - startX, endZ - startZ) * 1000.0;

      let maxTiltDeg = 0;
      let collapsed = false;
      for (const ob of this.blocks) {
        if (ob.shattered || ob.extracted || ob.id === cand.id) continue;
        const oInfo = this.worldCompat.getMapping(ob.bodyHandle);
        if (!oInfo) continue;
        const oAdr = oInfo.qposAdr;
        const qx = sQpos[oAdr + 4];
        const qz = sQpos[oAdr + 6];
        const upY = 1.0 - 2.0 * (qx * qx + qz * qz);
        const tilt = (Math.acos(Math.max(-1, Math.min(1, upY))) * 180) / Math.PI;
        if (tilt > maxTiltDeg) maxTiltDeg = tilt;
        if (tilt > 25.0) collapsed = true;
      }

      const slideScore = Math.min(1.0, slideMm / 1.8);
      const tiltPenalty = Math.min(1.0, maxTiltDeg / 4.0);
      const safetyScore01 = collapsed
        ? 0.0
        : Math.max(0.0, Math.min(1.0, slideScore * 0.72 + (1.0 - tiltPenalty) * 0.28));
      return { azRad, dirX, dirZ, slideMm, maxTiltDeg, collapsed, safetyScore01 };
    };

    const results: BlockCriticalityCandidate[] = [];

    try {
      // Pass 1: Fast longitudinal screening rollout (`+axisX`) across all candidates
      for (const cand of chosen) {
        const info = this.worldCompat.getMapping(cand.bodyHandle);
        if (!info) continue;
        const axisX = quatRotateVec3(cand.rotation, [1, 0, 0]);
        const baseAz = Math.atan2(axisX[2], axisX[0]);
        const clampN = this.computeBlockClampingLoadN(cand.id);
        const pPos = probePullAzimuth(cand, info, baseAz);

        results.push({
          blockId: cand.id,
          layer: cand.assignedLayer,
          slot: cand.assignedSlot,
          clampingLoadN: +clampN.toFixed(2),
          predictedMaxLeanDeg: +pPos.maxTiltDeg.toFixed(3),
          predictedSlideMm: +pPos.slideMm.toFixed(3),
          collapseTriggered: pPos.collapsed,
          safetyScore01: +pPos.safetyScore01.toFixed(3),
          optimalPullAzimuthRad: +pPos.azRad.toFixed(4),
          optimalPullDirXZ: [+pPos.dirX.toFixed(4), +pPos.dirZ.toFixed(4)],
        });
      }

      results.sort((a, b) => b.safetyScore01 - a.safetyScore01);

      // Pass 2 (CEM Pull-Azimuth Refinement): Refine antipodal + oblique pull azimuths on top-2 candidates
      const cemRefineCount = Math.min(2, results.length);
      for (let idx = 0; idx < cemRefineCount; idx++) {
        const r = results[idx];
        const cand = this.blocks[r.blockId];
        const info = cand ? this.worldCompat.getMapping(cand.bodyHandle) : null;
        if (!cand || !info) continue;

        const baseAz = r.optimalPullAzimuthRad ?? 0;
        let bestAz = baseAz;
        let bestDirX = r.optimalPullDirXZ?.[0] ?? Math.cos(baseAz);
        let bestDirZ = r.optimalPullDirXZ?.[1] ?? Math.sin(baseAz);
        let bestSlide = r.predictedSlideMm;
        let bestTilt = r.predictedMaxLeanDeg;
        let bestCollapsed = r.collapseTriggered;
        let bestSafety = r.safetyScore01;

        const pNeg = probePullAzimuth(cand, info, baseAz + Math.PI);
        if (pNeg.safetyScore01 > bestSafety + 1e-4) {
          bestAz = pNeg.azRad;
          bestDirX = pNeg.dirX;
          bestDirZ = pNeg.dirZ;
          bestSlide = pNeg.slideMm;
          bestTilt = pNeg.maxTiltDeg;
          bestCollapsed = pNeg.collapsed;
          bestSafety = pNeg.safetyScore01;
        }

        const samples: Array<{ azRad: number; safetyScore01: number }> = [
          { azRad: bestAz, safetyScore01: bestSafety },
        ];
        for (const offset of [-0.24, 0.24]) {
          const pr = probePullAzimuth(cand, info, bestAz + offset);
          samples.push({ azRad: pr.azRad, safetyScore01: pr.safetyScore01 });
          if (pr.safetyScore01 > bestSafety + 1e-4) {
            bestAz = pr.azRad;
            bestDirX = pr.dirX;
            bestDirZ = pr.dirZ;
            bestSlide = pr.slideMm;
            bestTilt = pr.maxTiltDeg;
            bestCollapsed = pr.collapsed;
            bestSafety = pr.safetyScore01;
          }
        }
        samples.sort((a, b) => b.safetyScore01 - a.safetyScore01);
        if (samples.length >= 2) {
          const sMean = Math.sin(samples[0].azRad) * 0.65 + Math.sin(samples[1].azRad) * 0.35;
          const cMean = Math.cos(samples[0].azRad) * 0.65 + Math.cos(samples[1].azRad) * 0.35;
          const eliteAz = Math.atan2(sMean, cMean);
          if (Math.abs(eliteAz - bestAz) > 0.04) {
            const ep = probePullAzimuth(cand, info, eliteAz);
            if (ep.safetyScore01 > bestSafety + 1e-4) {
              bestAz = ep.azRad;
              bestDirX = ep.dirX;
              bestDirZ = ep.dirZ;
              bestSlide = ep.slideMm;
              bestTilt = ep.maxTiltDeg;
              bestCollapsed = ep.collapsed;
              bestSafety = ep.safetyScore01;
            }
          }
        }

        r.predictedMaxLeanDeg = +bestTilt.toFixed(3);
        r.predictedSlideMm = +bestSlide.toFixed(3);
        r.collapseTriggered = bestCollapsed;
        r.safetyScore01 = +bestSafety.toFixed(3);
        r.optimalPullAzimuthRad = +bestAz.toFixed(4);
        r.optimalPullDirXZ = [+bestDirX.toFixed(4), +bestDirZ.toFixed(4)];
      }
    } finally {
      this.model.opt.iterations = savedIter;
      this.model.opt.ls_iterations = savedLsIter;
    }

    results.sort((a, b) => b.safetyScore01 - a.safetyScore01);

    // Ply 2: Evaluate adversarial top-row placement on top-2 viable extraction candidates
    const ply2Count = Math.min(2, results.length);
    for (let i = 0; i < ply2Count; i++) {
      const r = results[i];
      if (r.collapseTriggered || r.safetyScore01 <= 0.15) {
        r.twoPlyMasterScore01 = r.safetyScore01;
        continue;
      }
      const plans = this.evaluateAdversarialTopPlacement(
        r.blockId,
        Math.min(4, rolloutSteps),
        true
      );
      const bestPlan = plans[0];
      if (bestPlan) {
        r.adversarialPlacement = bestPlan;
        r.twoPlyMasterScore01 = +(
          r.safetyScore01 * 0.68 + bestPlan.adversarialScore01 * 0.32
        ).toFixed(3);
      } else {
        r.twoPlyMasterScore01 = r.safetyScore01;
      }
    }
    for (let i = ply2Count; i < results.length; i++) {
      results[i].twoPlyMasterScore01 = results[i].safetyScore01;
    }

    return results;
  }

  public getTelemetry(): TowerTelemetry {
    let intactCount = 0;
    let crazedCount = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (!b.shattered) {
        intactCount++;
        if (b.damage > 0.05) crazedCount++;
      }
    }
    return {
      turnPhase: this.turnPhase,
      gameMode: this.gameMode,
      currentPlayer: this.currentPlayer,
      playerCount: this.playerCount,
      blocksStacked: this.blocksStacked,
      currentLayers: this.currentLayers,
      peakLayers: this.peakLayers,
      liveHeightCm: +this.liveHeightCm.toFixed(1),
      peakHeightCm: +this.peakHeightCm.toFixed(1),
      leanAngleDeg: +this.leanAngleDeg.toFixed(2),
      wobbleSpeedDegPerSec: +this.wobbleSpeedDegPerSec.toFixed(1),
      centerOfMassXZ: [
        +this.centerOfMassXZCm[0].toFixed(2),
        +this.centerOfMassXZCm[1].toFixed(2),
      ],
      shatteredCount: this.shatteredCount,
      crazedCount,
      intactCount,
      activeShardCount: this.shards.length,
      grabResistance01: +this.grabResistance01.toFixed(2),
      hoveredBlockId: this.hoveredBlockId,
      activeBlockId: this.activeBlockId,
      hoveredSlotIndex: this.hoveredSlotIndex,
      carriedYawDeg: Math.round((this.carriedYawOffsetRad * 180) / Math.PI),
      lockedMinRow: this.lockedMinRow,
      settleProgress01:
        this.turnPhase === 'SETTLING'
          ? Math.min(1.0, this.settleTimer / this.SETTLE_DURATION)
          : 0,
      collapseReason: this.collapseReason,
    };
  }

  public exportCompactSnapshot(options?: { includeShardGeometry?: boolean }): PhysicsNetworkSnapshot {
    const blockCount = this.blocks.length;
    const blockData = new Float32Array(blockCount * 19);
    for (let i = 0; i < blockCount; i++) {
      const b = this.blocks[i];
      const base = i * 19;
      blockData[base + 0] = b.position[0];
      blockData[base + 1] = b.position[1];
      blockData[base + 2] = b.position[2];
      blockData[base + 3] = b.rotation[0];
      blockData[base + 4] = b.rotation[1];
      blockData[base + 5] = b.rotation[2];
      blockData[base + 6] = b.rotation[3];
      blockData[base + 7] = b.linVel[0];
      blockData[base + 8] = b.linVel[1];
      blockData[base + 9] = b.linVel[2];
      blockData[base + 10] = b.angVel[0];
      blockData[base + 11] = b.angVel[1];
      blockData[base + 12] = b.angVel[2];
      blockData[base + 13] = b.damage;
      blockData[base + 14] = b.crackLocalOrigin[0];
      blockData[base + 15] = b.crackLocalOrigin[1];
      blockData[base + 16] = b.crackLocalOrigin[2];
      blockData[base + 17] = b.rippleSeed;
      const encLayer = Math.max(0, Math.min(63, b.assignedLayer + 1)) & 0x3f;
      const encSlot = b.assignedSlot & 0x3;
      const encSwatch = b.swatchId & 0xf;
      const encMotif = b.motif & 0x3;
      const bits =
        (b.shattered ? 1 : 0) |
        (b.extracted ? 2 : 0) |
        (b.isLegalTarget ? 4 : 0) |
        (b.isLooseSlot ? 8 : 0) |
        (encLayer << 4) |
        (encSlot << 10) |
        (encSwatch << 12) |
        (encMotif << 16);
      blockData[base + 18] = bits;
    }

    const wrenchSlice = new Float32Array(blockCount * 4);
    const copyLen = Math.min(wrenchSlice.length, this.blockPhotoelasticWrench.length);
    for (let k = 0; k < copyLen; k++) {
      wrenchSlice[k] = this.blockPhotoelasticWrench[k];
    }

    const shardCount = this.shards.length;
    const shardData = new Float32Array(shardCount * 15);
    const shardMetas: PhysicsShardSnapshotMeta[] = [];
    const activeIds = new Set<number>();

    for (let i = 0; i < shardCount; i++) {
      const s = this.shards[i];
      activeIds.add(s.id);
      const base = i * 15;
      shardData[base + 0] = s.id;
      shardData[base + 1] = s.slotIndex;
      shardData[base + 2] = s.position[0];
      shardData[base + 3] = s.position[1];
      shardData[base + 4] = s.position[2];
      shardData[base + 5] = s.rotation[0];
      shardData[base + 6] = s.rotation[1];
      shardData[base + 7] = s.rotation[2];
      shardData[base + 8] = s.rotation[3];
      shardData[base + 9] = s.preLinVel[0];
      shardData[base + 10] = s.preLinVel[1];
      shardData[base + 11] = s.preLinVel[2];
      shardData[base + 12] = s.preAngVel[0];
      shardData[base + 13] = s.preAngVel[1];
      shardData[base + 14] = s.preAngVel[2];

      const isNewShard = !this.exportedShardIds.has(s.id);
      if (isNewShard) {
        this.exportedShardIds.add(s.id);
      }
      const includeGeo =
        options?.includeShardGeometry === true ||
        (options?.includeShardGeometry === undefined && isNewShard);
      const cachedVerts = this.slotRenderVertices[s.slotIndex];
      shardMetas.push({
        id: s.id,
        slotIndex: s.slotIndex,
        swatchId: s.swatchId,
        motif: s.motif,
        rippleSeed: s.rippleSeed,
        volume: s.volume,
        massKg: s.massKg,
        impactLocalPos: [...s.impactLocalPos],
        planeCount: s.planeCount,
        planes: new Float32Array(s.planes),
        vertexCount: s.vertexCount,
        renderVertices:
          includeGeo && cachedVerts ? new Float32Array(cachedVerts) : undefined,
      });
    }

    for (const prevId of this.exportedShardIds) {
      if (!activeIds.has(prevId)) {
        this.exportedShardIds.delete(prevId);
      }
    }

    const initialLayers =
      this.compiledTowerLayers > 0 ? this.compiledTowerLayers : INITIAL_LAYERS;

    return {
      version: 1,
      simTime: this.simTime,
      towerSeed: this.towerSeed,
      initialLayers,
      gameMode: this.gameMode,
      turnPhase: this.turnPhase,
      currentPlayer: this.currentPlayer,
      playerCount: this.playerCount,
      blocksStacked: this.blocksStacked,
      shatteredCount: this.shatteredCount,
      collapseReason: this.collapseReason,
      activeBlockId: this.activeBlockId,
      hoveredSlotIndex: this.hoveredSlotIndex,
      isPointerDraggingBlock: this.isPointerDraggingBlock,
      grabLocalPoint: [...this.grabLocalPoint],
      grabWorldAnchor: [...this.grabWorldAnchor],
      grabCurrentWorldOnBlock: [...this.grabCurrentWorldOnBlock],
      grabResistance01: this.grabResistance01,
      carriedTargetPos: [...this.carriedTargetPos],
      carriedTargetQuat: [...this.carriedTargetQuat],
      carriedLandingPos: [...this.carriedLandingPos],
      carriedYawOffsetRad: this.carriedYawOffsetRad,
      blockStride: 19,
      blockCount,
      blockData,
      blockPhotoelasticWrench: wrenchSlice,
      shardStride: 15,
      shardCount,
      shardData,
      shardMetas,
    };
  }

  public applyCompactSnapshot(snap: PhysicsNetworkSnapshot): void {
    if (!this.model || !this.data) return;
    if (
      snap.initialLayers !== this.compiledTowerLayers ||
      this.blocks.length < snap.blockCount
    ) {
      this.rebuildTower(snap.gameMode, snap.playerCount, snap.initialLayers, snap.towerSeed);
    }
    if (!this.model || !this.data) return;

    const m = this.model;
    const d = this.data;
    this.simTime = snap.simTime;
    this.towerSeed = snap.towerSeed >>> 0;
    this.gameMode = snap.gameMode;
    this.turnPhase = snap.turnPhase;
    this.currentPlayer = snap.currentPlayer;
    this.playerCount = snap.playerCount;
    this.blocksStacked = snap.blocksStacked;
    this.shatteredCount = snap.shatteredCount;
    this.collapseReason = snap.collapseReason;
    this.activeBlockId = snap.activeBlockId;
    this.hoveredSlotIndex = snap.hoveredSlotIndex;
    this.isPointerDraggingBlock = snap.isPointerDraggingBlock;
    this.grabLocalPoint = [...snap.grabLocalPoint];
    this.grabWorldAnchor = [...snap.grabWorldAnchor];
    this.grabCurrentWorldOnBlock = [...snap.grabCurrentWorldOnBlock];
    this.grabResistance01 = snap.grabResistance01;
    this.carriedTargetPos = [...snap.carriedTargetPos];
    this.carriedTargetQuat = [...snap.carriedTargetQuat];
    this.carriedLandingPos = [...snap.carriedLandingPos];
    this.carriedYawOffsetRad = snap.carriedYawOffsetRad;
    this.deferredFractures.length = 0;
    this.sandboxCollapseActive = false;
    if (!this.isPointerDraggingBlock) {
      d.eq_active[this.map.grabEqualityId] = 0;
    }

    const count = Math.min(this.blocks.length, snap.blockCount);
    const bd = snap.blockData;
    for (let i = 0; i < count; i++) {
      const b = this.blocks[i];
      const base = i * 19;
      const px = bd[base + 0];
      const py = bd[base + 1];
      const pz = bd[base + 2];
      const qx = bd[base + 3];
      const qy = bd[base + 4];
      const qz = bd[base + 5];
      const qw = bd[base + 6];
      const lvx = bd[base + 7];
      const lvy = bd[base + 8];
      const lvz = bd[base + 9];
      const avx = bd[base + 10];
      const avy = bd[base + 11];
      const avz = bd[base + 12];
      const damage = bd[base + 13];
      const cx = bd[base + 14];
      const cy = bd[base + 15];
      const cz = bd[base + 16];
      const rippleSeed = bd[base + 17];
      const bits = bd[base + 18] | 0;

      b.position[0] = px; b.position[1] = py; b.position[2] = pz;
      b.rotation[0] = qx; b.rotation[1] = qy; b.rotation[2] = qz; b.rotation[3] = qw;
      b.prePos[0] = px; b.prePos[1] = py; b.prePos[2] = pz;
      b.preQuat[0] = qx; b.preQuat[1] = qy; b.preQuat[2] = qz; b.preQuat[3] = qw;
      b.linVel[0] = lvx; b.linVel[1] = lvy; b.linVel[2] = lvz;
      b.angVel[0] = avx; b.angVel[1] = avy; b.angVel[2] = avz;
      b.damage = damage;
      b.crackLocalOrigin[0] = cx; b.crackLocalOrigin[1] = cy; b.crackLocalOrigin[2] = cz;
      b.rippleSeed = rippleSeed;
      b.shattered = (bits & 1) !== 0;
      b.extracted = (bits & 2) !== 0;
      b.isLegalTarget = (bits & 4) !== 0;
      b.isLooseSlot = (bits & 8) !== 0;
      b.assignedLayer = ((bits >>> 4) & 0x3f) - 1;
      b.assignedSlot = (bits >>> 10) & 0x3;
      b.swatchId = (bits >>> 12) & 0xf;
      b.motif = ((bits >>> 16) & 0x3) as TraceryMotif;

      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info) continue;
      if (b.shattered) {
        this.bodyToBlock.delete(b.bodyHandle);
        this.parkBodyOnHiddenShelf(
          b.bodyHandle,
          b.colliderHandle,
          info.qposAdr,
          info.dofAdr,
          520 + (b.id % 64)
        );
      } else {
        this.bodyToBlock.set(b.bodyHandle, b);
        const caff =
          Math.hypot(px, pz) > TABLE_RADIUS + 0.06
            ? MJ_CONAFFINITY_BLOCK_OFF_RIM
            : MJ_CONAFFINITY_BLOCK_ON_TABLE;
        setBodyGeomCollisionMask(m, b.bodyHandle, b.colliderHandle, MJ_CONTYPE_BLOCK, caff);
        this.worldCompat.markShardActive(b.bodyHandle, true, MJ_CONTYPE_BLOCK, caff);
        const adr = info.qposAdr;
        const dof = info.dofAdr;
        d.qpos[adr + 0] = px * M_PER_DM;
        d.qpos[adr + 1] = py * M_PER_DM;
        d.qpos[adr + 2] = pz * M_PER_DM;
        d.qpos[adr + 3] = qw;
        d.qpos[adr + 4] = qx;
        d.qpos[adr + 5] = qy;
        d.qpos[adr + 6] = qz;
        d.qvel[dof + 0] = lvx / FROUDE_VEL_SCALE;
        d.qvel[dof + 1] = lvy / FROUDE_VEL_SCALE;
        d.qvel[dof + 2] = lvz / FROUDE_VEL_SCALE;
        const wBody = quatInverseRotateVec3(
          [qx, qy, qz, qw],
          [avx * FROUDE_VEL_SCALE, avy * FROUDE_VEL_SCALE, avz * FROUDE_VEL_SCALE]
        );
        d.qvel[dof + 3] = wBody[0];
        d.qvel[dof + 4] = wBody[1];
        d.qvel[dof + 5] = wBody[2];
        if (
          Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) > 1e-5 ||
          py > BLOCK_HALF_Y + 0.02
        ) {
          this.worldCompat.wakeBody(b.bodyHandle);
        }
      }
    }

    const metaById = new Map<number, PhysicsShardSnapshotMeta>();
    if (snap.shardMetas) {
      for (const meta of snap.shardMetas) {
        metaById.set(meta.id, meta);
      }
    }
    const existingShardById = new Map<number, GlassShardState>();
    const prevContypeById = new Map<number, number>();
    const prevVertsById = new Map<number, Float32Array>();
    for (const s of this.shards) {
      existingShardById.set(s.id, s);
      prevContypeById.set(s.id, m.geom_contype[s.colliderHandle]);
      const pv = this.slotRenderVertices[s.slotIndex];
      if (pv) prevVertsById.set(s.id, pv);
    }

    // Reset shard pool first, then activate exact snapshot slots
    for (const oldS of this.shards) {
      this.bodyToShard.delete(oldS.bodyHandle);
      this.parkBodyOnHiddenShelf(
        oldS.bodyHandle,
        oldS.colliderHandle,
        this.map.shardQposAdr[oldS.slotIndex],
        this.map.shardDofAdr[oldS.slotIndex],
        oldS.slotIndex
      );
    }
    this.shards = [];
    const usedSlots = new Uint8Array(MAX_ACTIVE_SHARDS);
    const sd = snap.shardData;
    let maxSeenShardId = this.nextShardId;
    let restoredHeroCount = 0;
    const restoredHeroSpheres: Array<[number, number, number, number]> = [];

    for (let i = 0; i < snap.shardCount; i++) {
      const base = i * 15;
      const id = sd[base + 0] | 0;
      const slotIndex = sd[base + 1] | 0;
      if (slotIndex < 0 || slotIndex >= MAX_ACTIVE_SHARDS) continue;
      usedSlots[slotIndex] = 1;
      if (id >= maxSeenShardId) maxSeenShardId = id + 1;

      const px = sd[base + 2];
      const py = sd[base + 3];
      const pz = sd[base + 4];
      const qx = sd[base + 5];
      const qy = sd[base + 6];
      const qz = sd[base + 7];
      const qw = sd[base + 8];
      const lvx = sd[base + 9];
      const lvy = sd[base + 10];
      const lvz = sd[base + 11];
      const avx = sd[base + 12];
      const avy = sd[base + 13];
      const avz = sd[base + 14];

      const meta = metaById.get(id);
      const prev = existingShardById.get(id);
      const swatchId = meta?.swatchId ?? prev?.swatchId ?? 0;
      const motif = meta?.motif ?? prev?.motif ?? 0;
      const rippleSeed = meta?.rippleSeed ?? prev?.rippleSeed ?? 1.0;
      const volume = meta?.volume ?? prev?.volume ?? 0.002;
      const massKg = meta?.massKg ?? prev?.massKg ?? Math.max(0.0002, TARGET_BLOCK_MASS_KG * 0.04);
      const impactLocalPos: Vec3 = meta
        ? [...meta.impactLocalPos]
        : prev
          ? [...prev.impactLocalPos]
          : [0, 0, 0];
      const planeCount = meta?.planeCount ?? prev?.planeCount ?? 0;
      const planes = meta
        ? new Float32Array(meta.planes)
        : prev
          ? new Float32Array(prev.planes)
          : new Float32Array(48);
      const vertexCount = meta?.vertexCount ?? prev?.vertexCount ?? 0;
      const firstVertex = slotIndex * SHARD_SLOT_STRIDE_VERTS;

      const prevVerts = prevVertsById.get(id);
      if (meta?.renderVertices) {
        const vCopy = new Float32Array(meta.renderVertices);
        this.slotRenderVertices[slotIndex] = vCopy;
        this.pendingSlotUploads.set(slotIndex, vCopy);
      } else if (prevVerts) {
        this.slotRenderVertices[slotIndex] = prevVerts;
      }

      const vSource = meta?.renderVertices ?? prevVerts ?? this.slotRenderVertices[slotIndex];
      let hxDm: number;
      let hyDm: number;
      let hzDm: number;
      if (vSource && vertexCount > 0 && vSource.length >= vertexCount * 10) {
        let minVx = 0, maxVx = 0;
        let minVy = 0, maxVy = 0;
        let minVz = 0, maxVz = 0;
        for (let v = 0; v < vertexCount; v++) {
          const vBase = v * 10;
          const vx = vSource[vBase + 0];
          const vy = vSource[vBase + 1];
          const vz = vSource[vBase + 2];
          if (vx < minVx) minVx = vx;
          if (vx > maxVx) maxVx = vx;
          if (vy < minVy) minVy = vy;
          if (vy > maxVy) maxVy = vy;
          if (vz < minVz) minVz = vz;
          if (vz > maxVz) maxVz = vz;
        }
        const spanHxDm = Math.max(0.012, 0.5 * (maxVx - minVx));
        const spanHyDm = Math.max(0.012, 0.5 * (maxVy - minVy));
        const spanHzDm = Math.max(0.012, 0.5 * (maxVz - minVz));
        const aabbVolDm3 = Math.max(1e-6, 8.0 * spanHxDm * spanHyDm * spanHzDm);
        const volFit = Math.max(0.50, Math.min(0.80, Math.cbrt(volume / aabbVolDm3) * 0.88));
        hxDm = Math.min(BLOCK_HALF_X - 0.003, Math.max(0.011, spanHxDm * volFit));
        hyDm = Math.min(BLOCK_HALF_Y - 0.003, Math.max(0.011, spanHyDm * volFit));
        hzDm = Math.min(BLOCK_HALF_Z - 0.003, Math.max(0.011, spanHzDm * volFit));
      } else {
        const hSideDm = Math.max(0.015, Math.cbrt(Math.max(1e-6, volume)) * 0.44);
        hxDm = Math.min(BLOCK_HALF_X - 0.003, hSideDm);
        hyDm = Math.min(BLOCK_HALF_Y - 0.003, hSideDm);
        hzDm = Math.min(BLOCK_HALF_Z - 0.003, hSideDm);
      }
      const hxM = hxDm * M_PER_DM;
      const hyM = hyDm * M_PER_DM;
      const hzM = hzDm * M_PER_DM;
      const rBoundM = Math.hypot(hxM, hyM, hzM);

      const bodyId = this.map.shardBodyIds[slotIndex];
      const geomId = this.map.shardGeomIds[slotIndex];
      const adr = this.map.shardQposAdr[slotIndex];
      const dof = this.map.shardDofAdr[slotIndex];

      const wasDemoted = prevContypeById.get(id) === MJ_CONTYPE_CULLET_SHARD;
      const isAbyss = py < -0.45;
      const rShardDm = Math.hypot(hxDm, hyDm, hzDm);
      let overlapsIntactBlock = false;
      if (!isAbyss) {
        for (let bi = 0; bi < count; bi++) {
          const blk = this.blocks[bi];
          if (blk.shattered || blk.extracted) continue;
          const dx = px - blk.position[0];
          const dy = py - blk.position[1];
          const dz = pz - blk.position[2];
          if (Math.abs(dx) > 0.85 || Math.abs(dy) > 0.85 || Math.abs(dz) > 0.85) continue;
          const localPt = quatInverseRotateVec3(blk.rotation, [dx, dy, dz]);
          if (
            Math.abs(localPt[0]) < BLOCK_HALF_X + rShardDm &&
            Math.abs(localPt[1]) < BLOCK_HALF_Y + rShardDm &&
            Math.abs(localPt[2]) < BLOCK_HALF_Z + rShardDm
          ) {
            overlapsIntactBlock = true;
            break;
          }
        }
      }
      let isHero =
        volume >= 0.0022 &&
        !wasDemoted &&
        !isAbyss &&
        !overlapsIntactBlock &&
        restoredHeroCount < 56;
      if (isHero) {
        for (let hi = 0; hi < restoredHeroSpheres.length; hi++) {
          const hs = restoredHeroSpheres[hi];
          const distDm = Math.hypot(px - hs[0], py - hs[1], pz - hs[2]);
          if (distDm < (rShardDm + hs[3]) * 0.82) {
            isHero = false;
            break;
          }
        }
        if (isHero) {
          restoredHeroCount++;
          restoredHeroSpheres.push([px, py, pz, rShardDm]);
        }
      }

      const offRim = isAbyss || Math.hypot(px, pz) > TABLE_RADIUS + 0.01;
      const ctype = isHero ? MJ_CONTYPE_HERO_SHARD : MJ_CONTYPE_CULLET_SHARD;
      const caff = isHero
        ? (offRim ? MJ_CONAFFINITY_HERO_SHARD_OFF_RIM : MJ_CONAFFINITY_HERO_SHARD_ON_TABLE)
        : overlapsIntactBlock
          ? (offRim ? MJ_CONAFFINITY_CULLET_OFF_RIM : MJ_CONAFFINITY_CULLET_ON_TABLE)
          : this.getCulletConaffinity(offRim);

      m.geom_type[geomId] = 6;
      m.geom_dataid[geomId] = -1;
      m.geom_pos[geomId * 3 + 0] = 0;
      m.geom_pos[geomId * 3 + 1] = 0;
      m.geom_pos[geomId * 3 + 2] = 0;
      m.geom_quat[geomId * 4 + 0] = 1;
      m.geom_quat[geomId * 4 + 1] = 0;
      m.geom_quat[geomId * 4 + 2] = 0;
      m.geom_quat[geomId * 4 + 3] = 0;
      m.geom_size[geomId * 3 + 0] = hxM;
      m.geom_size[geomId * 3 + 1] = hyM;
      m.geom_size[geomId * 3 + 2] = hzM;
      m.geom_rbound[geomId] = rBoundM;
      const aabbBase = geomId * 6;
      m.geom_aabb[aabbBase + 0] = 0;
      m.geom_aabb[aabbBase + 1] = 0;
      m.geom_aabb[aabbBase + 2] = 0;
      m.geom_aabb[aabbBase + 3] = hxM;
      m.geom_aabb[aabbBase + 4] = hyM;
      m.geom_aabb[aabbBase + 5] = hzM;
      setBodyGeomCollisionMask(m, bodyId, geomId, ctype, caff);
      this.worldCompat.markShardActive(bodyId, true, ctype, caff);

      const lxM = 2 * hxM;
      const lyM = 2 * hyM;
      const lzM = 2 * hzM;
      const ixSi = (massKg / 12.0) * (lyM * lyM + lzM * lzM);
      const iySi = (massKg / 12.0) * (lxM * lxM + lzM * lzM);
      const izSi = (massKg / 12.0) * (lxM * lxM + lyM * lyM);
      updateFreeBodyMassAndInertiaO1(
        m,
        bodyId,
        dof,
        massKg,
        ixSi,
        iySi,
        izSi,
        this.cfg.shardRotationalDampingRate ?? 1.8
      );

      d.qpos[adr + 0] = px * M_PER_DM;
      d.qpos[adr + 1] = py * M_PER_DM;
      d.qpos[adr + 2] = pz * M_PER_DM;
      d.qpos[adr + 3] = qw;
      d.qpos[adr + 4] = qx;
      d.qpos[adr + 5] = qy;
      d.qpos[adr + 6] = qz;
      d.qvel[dof + 0] = lvx / FROUDE_VEL_SCALE;
      d.qvel[dof + 1] = lvy / FROUDE_VEL_SCALE;
      d.qvel[dof + 2] = lvz / FROUDE_VEL_SCALE;
      const swBody = quatInverseRotateVec3(
        [qx, qy, qz, qw],
        [avx * FROUDE_VEL_SCALE, avy * FROUDE_VEL_SCALE, avz * FROUDE_VEL_SCALE]
      );
      let swX = swBody[0];
      let swY = swBody[1];
      let swZ = swBody[2];
      const swMag = Math.hypot(swX, swY, swZ);
      if (swMag > 55.0) {
        const kw = 55.0 / swMag;
        swX *= kw;
        swY *= kw;
        swZ *= kw;
      }
      d.qvel[dof + 3] = swX;
      d.qvel[dof + 4] = swY;
      d.qvel[dof + 5] = swZ;
      const snapShardE = this.computeSingleShardTotalEnergyJ(bodyId, adr, dof);
      this.shardInContactPrevSubstep[slotIndex] = 0;
      this.shardPreSubstepTotalEnergyJ[slotIndex] = snapShardE;
      this.shardEntryTotalEnergyJ[slotIndex] = snapShardE;
      if (
        Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) > 1e-5 ||
        py > 0.26 ||
        py < -0.02 ||
        offRim
      ) {
        this.worldCompat.wakeBody(bodyId);
      }

      const shardState: GlassShardState = {
        id,
        slotIndex,
        bodyHandle: bodyId,
        colliderHandle: geomId,
        swatchId,
        motif,
        rippleSeed,
        position: [px, py, pz],
        rotation: [qx, qy, qz, qw],
        prePos: [px, py, pz],
        preQuat: [qx, qy, qz, qw],
        preLinVel: [lvx, lvy, lvz],
        preAngVel: [avx, avy, avz],
        volume,
        massKg,
        preVy: lvy,
        impactLocalPos,
        planes,
        planeCount,
        firstVertex,
        vertexCount,
        quietSubsteps: prev?.quietSubsteps ?? 60,
        spawnSimTime: prev?.spawnSimTime ?? this.simTime - 0.06,
      };
      this.shards.push(shardState);
      this.bodyToShard.set(bodyId, shardState);
    }

    this.nextShardId = maxSeenShardId;
    this.freeShardSlots = [];
    for (let slot = MAX_ACTIVE_SHARDS - 1; slot >= 0; slot--) {
      if (usedSlots[slot] === 0) {
        this.freeShardSlots.push(slot);
      }
    }

    this.multiccdReentrySubstepsRemaining = 0;
    d.qacc_warmstart.fill(0);
    this.mj.mj_forward(m, d);
    this.syncTransformsAndRules();

    // Restore exact snapshot turn/wrench fields after syncTransformsAndRules
    this.turnPhase = snap.turnPhase;
    this.shatteredCount = snap.shatteredCount;
    this.collapseReason = snap.collapseReason;
    for (let i = 0; i < count; i++) {
      this.blocks[i].isLegalTarget = ((bd[i * 19 + 18] | 0) & 4) !== 0;
    }
    this.lastStepResult = {
      substeps: 0,
      simTime: this.simTime,
      shardImpacts: [],
      wobbleEvents: [],
    };

    const wLen = Math.min(this.blockPhotoelasticWrench.length, snap.blockPhotoelasticWrench.length);
    for (let k = 0; k < wLen; k++) {
      this.blockPhotoelasticWrench[k] = snap.blockPhotoelasticWrench[k];
    }
    for (let i = 0; i < count && i < this.blockPhotoelasticStress01.length; i++) {
      const wb = i * 4;
      this.blockPhotoelasticStress01[i] = Math.max(
        this.blockPhotoelasticWrench[wb + 0] ?? 0,
        this.blockPhotoelasticWrench[wb + 1] ?? 0,
        this.blockPhotoelasticWrench[wb + 2] ?? 0,
        this.blockPhotoelasticWrench[wb + 3] ?? 0
      );
    }
  }

  public dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    if (this.rayGeomIdBuf) {
      try {
        this.rayGeomIdBuf.delete();
      } catch {
        // Ignore
      }
      this.rayGeomIdBuf = null;
    }
    if (this.rayNormalBuf) {
      try {
        this.rayNormalBuf.delete();
      } catch {
        // Ignore
      }
      this.rayNormalBuf = null;
    }
    if (this.scratchOracleData) {
      try {
        this.scratchOracleData.delete();
      } catch {
        // Ignore
      }
      this.scratchOracleData = null;
    }
    disposeMujocoHandles(this.data, this.model);
    this.data = null;
    this.model = null;
  }
}

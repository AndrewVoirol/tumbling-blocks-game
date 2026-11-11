import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';
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
  STAINED_GLASS_PALETTE,
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
  MJ_CONAFFINITY_BLOCK_OFF_RIM,
  MJ_CONAFFINITY_CULLET_ON_TABLE,
  MJ_CONAFFINITY_HERO_SHARD_OFF_RIM,
  MJ_CONAFFINITY_HERO_SHARD_ON_TABLE,
  MJ_CONAFFINITY_PARKING,
  MJ_CONTYPE_CULLET_SHARD,
  MJ_CONTYPE_HERO_SHARD,
  MJ_CONTYPE_PARKING,
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
import type {
  GameMode,
  GlassBlockState,
  GlassShardState,
  MicroSplinter,
  ShardMeshAppendEvent,
  TopPlacementSlot,
  TopSupportFrame,
  TowerTelemetry,
  TurnPhase,
} from './PhysicsWorld';

export interface BlockCriticalityCandidate {
  blockId: number;
  layer: number;
  slot: number;
  clampingLoadN: number;
  predictedMaxLeanDeg: number;
  predictedSlideMm: number;
  collapseTriggered: boolean;
  safetyScore01: number;
}

const TOP_HOVER_CLEARANCE_DM = 0.040;
const MAX_SPLINTERS = 320;
const liveMujocoEngines: MujocoPhysicsEngine[] = [];
const MAX_CONCURRENT_LIVE_ENGINES = 3;

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
  public wobbleSpeedDegPerSec: number = 0;
  public centerOfMassXZCm: [number, number] = [0, 0];
  public currentLayers: number = INITIAL_LAYERS;
  public peakLayers: number = INITIAL_LAYERS;
  public liveHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;
  public peakHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;
  public lockedMinRow: number | null = INITIAL_LAYERS;
  public lastFractureOriginLocal: Vec3 | null = null;

  private mj!: MainModule;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private scratchOracleData: MjData | null = null;
  private compiledTowerLayers: number = -1;
  private modelResetSnapshot: {
    geom_type: Int32Array;
    geom_dataid: Int32Array;
    geom_contype: Int32Array;
    geom_conaffinity: Int32Array;
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
  } | null = null;
  private activeSlidingGeomId: number | null = null;
  private map!: MujocoTowerIndexMap;
  private worldCompat!: MujocoWorldCompat;
  private cfg: MujocoCalibratedConfig;
  private audio: GlassAudioEngine;

  private bodyToBlock = new Map<number, GlassBlockState>();
  private bodyToShard = new Map<number, GlassShardState>();
  private cachedClampingLoadN: Float64Array = new Float64Array(0);
  private blockLastCrazeTime = new Map<number, number>();
  private freeShardSlots: number[] = [];
  private shardQuietSubsteps = new Uint16Array(MAX_ACTIVE_SHARDS);
  private archGeomPos: Float64Array = new Float64Array(0);
  private archGeomQuat: Float64Array = new Float64Array(0);
  private pendingSlotUploads = new Map<number, Float32Array>();
  private shardBufferResetPending: boolean = true;
  private nextShardId: number = 1;

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
  private isDisposed: boolean = false;

  private extraPreStep = new Map<
    number,
    { posDm: Vec3; linVelSim: Vec3; angVelSim: Vec3; massKg: number }
  >();

  constructor(
    audio: GlassAudioEngine,
    initialLayers: number = INITIAL_LAYERS,
    cfg: MujocoCalibratedConfig = MUJOCO_CALIBRATED_CONFIG
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
    this.rebuildTower(undefined, undefined, initialLayers);
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

  public rebuildTower(
    mode?: GameMode,
    players?: number,
    layers: number = INITIAL_LAYERS
  ): void {
    if (mode) this.gameMode = mode;
    if (players !== undefined) this.playerCount = Math.max(2, Math.min(4, players));
    const towerLayers = Math.max(1, Math.min(32, Math.round(layers)));
    this.audio.updateSlideFriction(0, 0, 0, 0);

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
      this.model.body_contype.set(snap.body_contype);
      this.model.body_conaffinity.set(snap.body_conaffinity);
      this.model.body_mass.set(snap.body_mass);
      this.model.body_subtreemass.set(snap.body_subtreemass);
      this.model.body_inertia.set(snap.body_inertia);
      this.model.body_iquat.set(snap.body_iquat);
      this.model.body_invweight0.set(snap.body_invweight0);
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

    this.worldCompat = new MujocoWorldCompat(this.mj, this.model, this.data, this.map);
    this.shardQuietSubsteps.fill(0);

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
    this.settleTimer = 0;
    this.accumulatorSec = 0;
    this.simTime = 0;
    this.leanAngleDeg = 0;
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

    for (const spec of this.map.initialBlocks) {
      const bodyId = this.map.blockBodyIds[spec.id];
      const geomId = this.map.blockGeomIds[spec.id];
      const swatchId =
        (spec.layer * 3 + spec.slot * 2 + Math.floor(spec.layer / 2)) %
        STAINED_GLASS_PALETTE.length;
      const motif: TraceryMotif = 0;

      const blk: GlassBlockState = {
        id: spec.id,
        bodyHandle: bodyId,
        colliderHandle: geomId,
        swatchId,
        motif,
        rippleSeed: (spec.id + 1) * 1.6180339,
        initialLayer: spec.layer,
        assignedLayer: spec.layer,
        assignedSlot: spec.slot,
        isLooseSlot: spec.isLoose,
        position: [...spec.posDm],
        rotation: [...spec.quat],
        prePos: [...spec.posDm],
        preQuat: [...spec.quat],
        linVel: [0, 0, 0],
        angVel: [0, 0, 0],
        currentFriction: this.cfg.glassFriction[0],
        isMicroDamped: false,
        damage: 0,
        crackLocalOrigin: [0, 0, 0],
        shattered: false,
        extracted: false,
        isLegalTarget: spec.layer < towerLayers - 1,
      };
      this.blocks.push(blk);
      this.bodyToBlock.set(bodyId, blk);
    }

    if (canReuseCompiledModel && this.modelResetSnapshot) {
      this.cachedClampingLoadN.set(this.modelResetSnapshot.warmupClampingLoadN);
      this.blockPhotoelasticStress01.set(this.modelResetSnapshot.warmupPhotoelastic01);
      this.blockPhotoelasticWrench.set(this.modelResetSnapshot.warmupPhotoelasticWrench);
      // 2 quick steps re-establish warmstart contacts cleanly on the restored settled pose
      for (let i = 0; i < 2; i++) {
        this.mj.mj_step(this.model, this.data);
      }
    } else {
      // Quiet warm-up steps with fractures suppressed so beveled stack settles & caches initial clamping forces
      // IMPORTANT: capture clamping loads INSIDE the warmup loop while `ntree_awake > 0` before island sleep sets `ncon = 0`!
      const warmupSteps = Math.min(48, 16 + towerLayers * 2);
      for (let i = 0; i < warmupSteps; i++) {
        this.mj.mj_step(this.model, this.data);
        if (this.data.ntree_awake > 0) {
          this.refreshClampingLoadsAndPhotoelasticStress(true);
        }
      }
      this.modelResetSnapshot = {
        geom_type: new Int32Array(this.model.geom_type),
        geom_dataid: new Int32Array(this.model.geom_dataid),
        geom_contype: new Int32Array(this.model.geom_contype),
        geom_conaffinity: new Int32Array(this.model.geom_conaffinity),
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

  private refreshClampingLoadsAndPhotoelasticStress(isWarmup: boolean = false): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d || d.ntree_awake === 0) return;

    const nBlocks = this.blocks.length;
    const hasVerticalContact = new Uint8Array(nBlocks);
    const hasAnyPhotoContact = new Uint8Array(nBlocks);
    const topZoneNeg = new Float64Array(nBlocks);
    const topZoneCtr = new Float64Array(nBlocks);
    const topZonePos = new Float64Array(nBlocks);
    const botZoneNeg = new Float64Array(nBlocks);
    const botZoneCtr = new Float64Array(nBlocks);
    const botZonePos = new Float64Array(nBlocks);
    const midShear = new Float64Array(nBlocks);
    const topF = new Float64Array(nBlocks);
    const topAbsX = new Float64Array(nBlocks);
    const topSigX = new Float64Array(nBlocks);
    const botF = new Float64Array(nBlocks);
    const botAbsX = new Float64Array(nBlocks);
    const botSigX = new Float64Array(nBlocks);

    const contacts = d.contact;
    const efc = d.efc_force;
    const geomBodyId = m.geom_bodyid;
    const xpos = d.xpos;
    const qpos = d.qpos;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;
    const ncon = d.ncon;

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
      const adr = c.efc_address;
      if (adr >= 0) {
        const fn = Math.max(0, efc[adr]);
        if (fn > 1e-4) {
          const b1 = geomBodyId[c.geom[0]];
          const b2 = geomBodyId[c.geom[1]];
          const ft = c.dim >= 3 ? Math.hypot(efc[adr + 1], efc[adr + 2]) : 0;
          const cPos = c.pos;
          const wxDm = cPos[0] * DM_PER_M;
          const wyDm = cPos[1] * DM_PER_M;
          const wzDm = cPos[2] * DM_PER_M;

          const blk1 = b1 >= 2 ? this.bodyToBlock.get(b1) : undefined;
          const blk2 = b2 >= 2 ? this.bodyToBlock.get(b2) : undefined;

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
          const colForce = (fTop: number, fBot: number): number =>
            Math.max(0.0, 1.75 * Math.min(fTop, fBot) + 0.18 * Math.max(fTop, fBot) - 0.38);
          const tNeg = Math.min(1.0, colForce(topZoneNeg[id], botZoneNeg[id]) / 7.5 + dmgLift);
          const tCtr = Math.min(1.0, colForce(topZoneCtr[id], botZoneCtr[id]) / 7.5 + dmgLift);
          const tPos = Math.min(1.0, colForce(topZonePos[id], botZonePos[id]) / 7.5 + dmgLift);

          let bend01 = Math.min(0.45, midShear[id] / 4.5);
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
    let bearingInSameRow = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const ob = this.blocks[i];
      if (ob.shattered || ob.extracted) continue;
      if (ob.position[1] > blk.position[1] + BLOCK_HEIGHT * 0.55 && Math.hypot(ob.position[0], ob.position[2]) < 0.85) {
        blocksAbove++;
      } else if (
        ob.assignedLayer === blk.assignedLayer &&
        (ob.id < this.cachedClampingLoadN.length ? this.cachedClampingLoadN[ob.id] : 0) > 0.65
      ) {
        bearingInSameRow++;
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
            for (let k = 0; k < 6; k++) this.model.dof_damping[info.dofAdr + k] = 0;
          }
        }
        this.worldCompat.wakeBody(b.bodyHandle);
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
      this.listenerCamRight
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

    if (this.turnPhase === 'PULLING') {
      this.activeBlockId = null;
      this.turnPhase = 'SELECT_BLOCK';
      // Wake active tower blocks on aborted pull so any precarious overhang immediately resolves under gravity!
      this.worldCompat.wakeActiveTowerBlocks();
      return;
    }

    if (this.turnPhase === 'EXTRACTED' && dropIfPoisedOverTower) {
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
    this.wakeTowerForInteraction();
    for (const b of this.blocks) {
      if (b.shattered) continue;
      const info = this.worldCompat.getMapping(b.bodyHandle);
      if (!info) continue;
      const massKg = Math.max(0.01, this.model.body_mass[b.bodyHandle]);
      const invMassFroude = INV_FROUDE_VEL_SCALE / massKg;

      if (b.assignedLayer === 0) {
        if (b.assignedSlot === 0) {
          this.data.qvel[info.dofAdr + 0] += -0.055 * invMassFroude;
          this.data.qvel[info.dofAdr + 2] += -0.02 * invMassFroude;
        }
      } else if (b.assignedLayer <= 2) {
        if (b.assignedSlot <= 1) {
          const baseScale = b.assignedSlot === 0 ? -0.235 : -0.192;
          this.data.qvel[info.dofAdr + 0] += baseScale * invMassFroude;
          this.data.qvel[info.dofAdr + 1] += 0.004 * invMassFroude;
          this.data.qvel[info.dofAdr + 2] += -0.054 * invMassFroude;
        } else {
          this.data.qvel[info.dofAdr + 0] += 0.032 * invMassFroude;
          this.data.qvel[info.dofAdr + 2] += 0.017 * invMassFroude;
        }
      } else {
        const hScale = 0.088 + Math.min(16, b.assignedLayer) * 0.0052;
        this.data.qvel[info.dofAdr + 0] += hScale * invMassFroude;
        this.data.qvel[info.dofAdr + 2] += hScale * 0.34 * invMassFroude;
      }
    }
    this.audio.playGlassImpact(0.62, 0.96, [0, 0.6, 0], this.listenerCamPos, this.listenerCamRight);
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
      const alongLen = errXDm * axisX[0] + errZDm * axisX[2];
      const acrossWid = errXDm * axisZ[0] + errZDm * axisZ[2];
      const lateralScale = this.gameMode === 'sandbox' ? 0.85 : 1.0 - 0.78 * insideFactor;

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
      // Hold horizontal extraction plane height `grabPlaneY` only AFTER the block has pulled partially out (> 18 mm)
      // so an overhanging block does not droop; while inside or barely emerging, track the block's natural slot Y
      // (`grabWorldDm[1]`) so the mocap equality constraint NEVER wedges the block upward into the row above!
      const overhangBlend =
        this.gameMode === 'sandbox'
          ? 0.0
          : Math.max(0.0, Math.min(1.0, (radialDistDm - 0.20) / 0.22));
      const desiredPlaneY =
        grabWorldDm[1] * (1.0 - overhangBlend) + this.grabPlaneY * overhangBlend;
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
      const av = rb.angvel();
      const info = this.worldCompat.getMapping(block.bodyHandle);
      if (info) {
        const wLocalX = this.data.qvel[info.dofAdr + 3];
        const wLocalZ = this.data.qvel[info.dofAdr + 5];
        this.data.qvel[info.dofAdr + 3] = wLocalX * 0.94;
        this.data.qvel[info.dofAdr + 5] = wLocalZ * 0.94;
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
          this.model.geom_priority[gid] = 1;
          const g3 = gid * 3;
          const sf = this.cfg.glassFriction;
          const kf = this.cfg.glassKineticFriction;
          this.model.geom_friction[g3 + 0] = sf[0] + (kf[0] - sf[0]) * kinBlend;
          this.model.geom_friction[g3 + 1] = sf[1] + (kf[1] - sf[1]) * kinBlend;
          this.model.geom_friction[g3 + 2] = sf[2] + (kf[2] - sf[2]) * kinBlend;
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
        this.audio.updateSlideFriction(speedHorizReal, this.grabResistance01, slipPowerMw, av.y * FROUDE_VEL_SCALE);
      }

      // Geometric clearance + 3 consecutive zero-contact substeps -> EXTRACTED
      if (this.gameMode !== 'sandbox') {
        if (geomClear && touchingCount === 0) {
          this.clearContactSubsteps++;
        } else {
          this.clearContactSubsteps = 0;
        }

        if (this.clearContactSubsteps >= 3) {
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

          this.audio.updateSlideFriction(0, 0, 0, 0);
          this.audio.playGlassImpact(
            0.24,
            1.25,
            [pos.x, pos.y, pos.z],
            this.listenerCamPos,
            this.listenerCamRight
          );

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

      const treeId = bodyTreeId[b.bodyHandle];
      if (info.settledAsleep || (treeId >= 0 && treeAsleep[treeId] >= 0)) {
        b.linVel[0] = 0; b.linVel[1] = 0; b.linVel[2] = 0;
        b.angVel[0] = 0; b.angVel[1] = 0; b.angVel[2] = 0;
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

      if (!b.extracted) {
        const bound = (Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) * 0.42) * SIM_TIME_SCALE;
        if (bound > maxBoundReal) maxBoundReal = bound;
      }
    }

    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const info = this.worldCompat.getMapping(s.bodyHandle);
      const treeId = bodyTreeId[s.bodyHandle];
      const asleep = !info || !info.enabled || info.settledAsleep || (treeId >= 0 && treeAsleep[treeId] >= 0);
      if (!asleep) {
        const adr = this.map.shardQposAdr[s.slotIndex];
        const dof = this.map.shardDofAdr[s.slotIndex];
        s.prePos[0] = qpos[adr + 0] * DM_PER_M;
        s.prePos[1] = qpos[adr + 1] * DM_PER_M;
        s.prePos[2] = qpos[adr + 2] * DM_PER_M;
        s.preQuat[0] = qpos[adr + 4];
        s.preQuat[1] = qpos[adr + 5];
        s.preQuat[2] = qpos[adr + 6];
        s.preQuat[3] = qpos[adr + 3];
        const lvx = qvel[dof + 0] * FROUDE_VEL_SCALE;
        const lvy = qvel[dof + 1] * FROUDE_VEL_SCALE;
        const lvz = qvel[dof + 2] * FROUDE_VEL_SCALE;
        const wWorldSi = quatRotateVec3(s.preQuat, [qvel[dof + 3], qvel[dof + 4], qvel[dof + 5]]);
        const avx = wWorldSi[0] * INV_FROUDE_VEL_SCALE;
        const avy = wWorldSi[1] * INV_FROUDE_VEL_SCALE;
        const avz = wWorldSi[2] * INV_FROUDE_VEL_SCALE;
        s.preLinVel[0] = lvx; s.preLinVel[1] = lvy; s.preLinVel[2] = lvz;
        s.preAngVel[0] = avx; s.preAngVel[1] = avy; s.preAngVel[2] = avz;
        s.preVy = lvy;
        const sb = (Math.hypot(lvx, lvy, lvz) + Math.hypot(avx, avy, avz) * 0.25) * SIM_TIME_SCALE;
        if (sb > maxBoundReal) maxBoundReal = sb;
      } else {
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
   * and three-point flexural bending stress across all active MuJoCo contacts!
   */
  private processWrenchCollisionsAndFractures(subDtSec: number): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d || d.ncon === 0) return;

    interface BlockWrenchAccum {
      blk: GlassBlockState;
      dynamicImpulseSim: number;
      peakDynamicImpulseSim: number;
      maxVCloseReal: number;
      bestLocalHitDm: Vec3;
      bestNormalAtoB: Vec3;
      bestHitTable: boolean;
      contacts: Array<{
        jnSim: number;
        fnN: number;
        localHitDm: Vec3;
        nAtoB: Vec3;
        vCloseReal: number;
      }>;
    }

    const blockImpacts = new Map<number, BlockWrenchAccum>();
    const shardAudioEvents: Array<{
      energy: number;
      intensity01: number;
      volumeDm3: number;
      worldPos: [number, number, number];
      hitGlass: boolean;
    }> = [];

    const contacts = d.contact;
    const efc = d.efc_force;
    const bodyMass = m.body_mass;
    const geomBodyId = m.geom_bodyid;
    const ncon = d.ncon;
    const dtSim = subDtSec * SIM_TIME_SCALE;

    const recordForBlock = (
      blkA: GlassBlockState,
      otherBodyId: number,
      otherGeomId: number,
      worldPtDm: Vec3,
      nAtoB: Vec3,
      fnN: number,
      jnSim: number
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
      const hitTable = otherGeomId === this.map.tableGeomId;

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
          contacts: [],
        };
        blockImpacts.set(blkA.id, acc);
      }

      acc.contacts.push({ jnSim: clampedJnSim, fnN, localHitDm, nAtoB, vCloseReal });
      if (vCloseReal > 1.5) {
        acc.dynamicImpulseSim += clampedJnSim;
        if (vCloseReal > acc.maxVCloseReal) acc.maxVCloseReal = vCloseReal;
        if (clampedJnSim > acc.peakDynamicImpulseSim) {
          acc.peakDynamicImpulseSim = clampedJnSim;
          acc.bestLocalHitDm = localHitDm;
          acc.bestNormalAtoB = nAtoB;
          acc.bestHitTable = hitTable;
        }
      }
    };

    // First coalesce multi-point contacts (`multiccd` generates 2-4 vertices per geom pair) by `(g1, g2)` pair
    // so total pair normal force `fnPair` matches the physical manifold resultant!
    interface GeomPairContact {
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
    const pairMap = new Map<number, GeomPairContact>();

    for (let i = 0; i < ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      const adr = c.efc_address;
      if (adr >= 0) {
        const fnN = Math.max(0, efc[adr]);
        if (fnN > 0.02) {
          const g1 = c.geom[0];
          const g2 = c.geom[1];
          if (g1 !== this.map.parkingShelfGeomId && g2 !== this.map.parkingShelfGeomId) {
            const key = g1 * 4096 + g2;
            let pm = pairMap.get(key);
            if (!pm) {
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
              pairMap.set(key, pm);
            }
            pm.fnSum += fnN;
            pm.wx += c.pos[0] * DM_PER_M * fnN;
            pm.wy += c.pos[1] * DM_PER_M * fnN;
            pm.wz += c.pos[2] * DM_PER_M * fnN;
            pm.nx += c.frame[0] * fnN;
            pm.ny += c.frame[1] * fnN;
            pm.nz += c.frame[2] * fnN;
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
      if (fnN <= 0.05) continue;
      const invFn = 1.0 / fnN;
      const worldPtDm: Vec3 = [pm.wx * invFn, pm.wy * invFn, pm.wz * invFn];
      const nLen = Math.hypot(pm.nx, pm.ny, pm.nz) || 1.0;
      const n1to2: Vec3 = [pm.nx / nLen, pm.ny / nLen, pm.nz / nLen];
      const n2to1: Vec3 = [-n1to2[0], -n1to2[1], -n1to2[2]];
      const jnSim = fnN * tauImpactSim;

      const { g1, g2, b1, b2 } = pm;
      const blk1 = this.bodyToBlock.get(b1);
      const blk2 = this.bodyToBlock.get(b2);
      if (blk1) recordForBlock(blk1, b2, g2, worldPtDm, n1to2, fnN, jnSim);
      if (blk2) recordForBlock(blk2, b1, g1, worldPtDm, n2to1, fnN, jnSim);

      const shd1 = this.bodyToShard.get(b1);
      const shd2 = this.bodyToShard.get(b2);
      if (shd1 || shd2) {
        const primaryShard = shd1 ?? shd2!;
        const v1 = this.getPreStepPointVelSim(b1, worldPtDm);
        const v2 = this.getPreStepPointVelSim(b2, worldPtDm);
        const dvReal =
          Math.hypot(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]) * SIM_TIME_SCALE;
        if (dvReal > 2.2 && jnSim > 0.001) {
          const hitTable = g1 === this.map.tableGeomId || g2 === this.map.tableGeomId;
          const effMass =
            shd1 && shd2 ? Math.min(shd1.massKg, shd2.massKg) : primaryShard.massKg;
          shardAudioEvents.push({
            energy: effMass * dvReal * dvReal,
            intensity01: Math.min(1.0, (dvReal - 1.8) / 14.0),
            volumeDm3: primaryShard.volume,
            worldPos: [...primaryShard.prePos],
            hitGlass: !hitTable,
          });
        }
      }
    }

    if (shardAudioEvents.length > 0) {
      this.audio.playRankedShardCollisions(
        shardAudioEvents,
        this.listenerCamPos,
        this.listenerCamRight
      );
    }

    if (blockImpacts.size === 0) return;

    const blocksToShatter: Array<{
      blk: GlassBlockState;
      localHit: Vec3;
      impactSpeed: number;
      worldNormal: Vec3;
      hitTable: boolean;
    }> = [];

    for (const acc of blockImpacts.values()) {
      if (acc.maxVCloseReal <= 1.5 || acc.dynamicImpulseSim <= 0) continue;
      const blk = acc.blk;
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
        if (dotOpp < -0.5) opposingJn += c.jnSim;

        const cNLocal = quatInverseRotateVec3(blk.preQuat, c.nAtoB);
        if (cNLocal[1] > 0.6 && Math.abs(c.localHitDm[0]) < BLOCK_HALF_X * 0.45) {
          topMidForceN += c.fnN;
        } else if (cNLocal[1] < -0.6 && Math.abs(c.localHitDm[0]) > BLOCK_HALF_X * 0.45) {
          bottomOuterSupportN += c.fnN;
        }
      }

      // True opposing-wrench anvil crush: requires dynamic strike on one face PLUS opposing normal reaction impulse on the opposite face!
      const isAnvilCrush =
        acc.maxVCloseReal > CRAZE_IMPACT_SPEED &&
        acc.dynamicImpulseSim > 0.12 &&
        opposingJn > 0.25 * acc.dynamicImpulseSim;

      // Three-point flexural bending stress: M_bend = F_mid * L / 4; sigma = 6 * M / (b * h^2)
      let flexuralBoost = 1.0;
      if (topMidForceN > 40.0 && bottomOuterSupportN > 25.0) {
        const fBend = Math.min(topMidForceN, bottomOuterSupportN);
        const mBendNm = (fBend * 0.06) / 4.0;
        const sigmaMpa = ((6.0 * mBendNm) / (0.025 * 0.015 * 0.015)) * 1e-6;
        if (sigmaMpa > this.cfg.flexuralStrengthMpa * 0.5) {
          flexuralBoost = 1.18;
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

      if (vImpact > 1.75) {
        const clinkIntensity = Math.min(1.0, (vImpact - 1.5) / 14.0);
        this.audio.playGlassImpact(
          clinkIntensity,
          acc.bestHitTable ? 0.92 : 1.08,
          hitWorld,
          this.listenerCamPos,
          this.listenerCamRight
        );
      }

      const lastCrazeSec = this.blockLastCrazeTime.get(blk.id) ?? -10.0;
      const inSameSlapWindow = this.simTime - lastCrazeSec < 0.22;
      const effectivePriorDamage = inSameSlapWindow ? 0.0 : blk.damage;
      const effCrazeSpeed = CRAZE_IMPACT_SPEED * 0.72;
      const baseShatterSpeed = isAnvilCrush ? SHATTER_IMPACT_SPEED : SHATTER_IMPACT_SPEED * 1.12;
      const shatterThreshold = Math.max(
        isAnvilCrush ? 11.4 : 14.3,
        baseShatterSpeed * (1.0 - Math.min(0.85, effectivePriorDamage) * 0.16)
      );

      if (vImpact >= shatterThreshold) {
        blocksToShatter.push({
          blk,
          localHit: lh,
          impactSpeed: vImpact,
          worldNormal,
          hitTable: acc.bestHitTable,
        });
      } else if (vImpact >= effCrazeSpeed) {
        const prevDamage = blk.damage;
        const addedDamage =
          0.33 +
          ((vImpact - effCrazeSpeed) / (SHATTER_IMPACT_SPEED - effCrazeSpeed)) * 0.48;
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

    for (const item of blocksToShatter) {
      this.shatterBlock(item.blk, item.localHit, item.impactSpeed, item.worldNormal, item.hitTable);
    }
  }

  private parkBodyOnHiddenShelf(bodyId: number, geomId: number, qposAdr: number, dofAdr: number, slotSalt: number): void {
    if (!this.model || !this.data) return;
    setBodyGeomCollisionMask(
      this.model,
      bodyId,
      geomId,
      MJ_CONTYPE_PARKING,
      MJ_CONAFFINITY_PARKING
    );
    const parkXM = ((slotSalt % 32) - 15.5) * 0.12;
    const parkZM = (Math.floor(slotSalt / 32) - 8.0) * 0.12;
    const parkYM = MJ_PARKING_SHELF_TOP_Y_M + 0.005;

    this.data.qpos[qposAdr + 0] = parkXM;
    this.data.qpos[qposAdr + 1] = parkYM;
    this.data.qpos[qposAdr + 2] = parkZM;
    this.data.qpos[qposAdr + 3] = 1;
    this.data.qpos[qposAdr + 4] = 0;
    this.data.qpos[qposAdr + 5] = 0;
    this.data.qpos[qposAdr + 6] = 0;
    for (let k = 0; k < 6; k++) {
      this.data.qvel[dofAdr + k] = 0;
      this.data.qfrc_applied[dofAdr + k] = 0;
      this.data.xfrc_applied[bodyId * 6 + k] = 0;
      this.model.dof_damping[dofAdr + k] = 0;
    }
    this.worldCompat.markShardActive(bodyId, false, MJ_CONTYPE_PARKING, MJ_CONAFFINITY_PARKING);
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
    hitTable: boolean = true
  ): void {
    if (block.shattered || !this.model || !this.data) return;
    const rb = this.worldCompat.getRigidBody(block.bodyHandle);
    const blkInfo = this.worldCompat.getMapping(block.bodyHandle);
    if (!rb || !blkInfo) return;

    const pos = rb.translation();
    const rot = rb.rotation();
    const q: Quat = [rot.x, rot.y, rot.z, rot.w];
    const lv = rb.linvel();
    const av = rb.angvel();
    const parentMass = Math.max(0.01, rb.mass());

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

    this.bodyToBlock.delete(block.bodyHandle);
    this.parkBodyOnHiddenShelf(
      block.bodyHandle,
      block.colliderHandle,
      blkInfo.qposAdr,
      blkInfo.dofAdr,
      520 + (block.id % 64)
    );

    const impactEnergy01 = Math.max(0.0, Math.min(1.0, (impactSpeed - 12.5) / 15.0));
    const energyTargetShards = Math.round(20 + impactEnergy01 * 4);
    const headroom = MAX_ACTIVE_SHARDS - this.shards.length;
    const budgetCap = headroom > 120 ? 23 : headroom > 65 ? 19 : 12;
    const desiredShards = Math.min(energyTargetShards, budgetCap);
    const allowSecondaryCleavage = impactSpeed > 15.0;

    const shardGeoms = fractureBlockVoronoi(
      effectiveOrigin,
      desiredShards,
      block.id * 977 + Math.floor(this.simTime * 1000) + this.shatteredCount * 31,
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

    for (const geom of shardGeoms) {
      const slotIndex = this.freeShardSlots.pop();
      if (slotIndex === undefined) break;

      const worldOffset = quatRotateVec3(q, geom.localCentroid);
      const sxDm = pos.x + worldOffset[0];
      const syDm = pos.y + worldOffset[1];
      const szDm = pos.z + worldOffset[2];

      let maxHxDm = 0.012, maxHyDm = 0.012, maxHzDm = 0.012;
      for (let p = 0; p < geom.hullPoints.length; p += 3) {
        maxHxDm = Math.max(maxHxDm, Math.abs(geom.hullPoints[p]));
        maxHyDm = Math.max(maxHyDm, Math.abs(geom.hullPoints[p + 1]));
        maxHzDm = Math.max(maxHzDm, Math.abs(geom.hullPoints[p + 2]));
      }

      const aabbVolDm3 = Math.max(1e-6, 8.0 * maxHxDm * maxHyDm * maxHzDm);
      const volFit = Math.max(0.54, Math.min(0.84, Math.cbrt(geom.volume / aabbVolDm3) * 0.92));
      const hxM = maxHxDm * M_PER_DM * volFit;
      const hyM = maxHyDm * M_PER_DM * volFit;
      const hzM = maxHzDm * M_PER_DM * volFit;
      const safeSyDm =
        pos.y < 0.16 && Math.hypot(sxDm, szDm) <= TABLE_RADIUS + 0.05
          ? Math.max(syDm, hyM * DM_PER_M + 0.001)
          : syDm;
      const shardExactMass = Math.max(0.0002, parentMass * geom.volume * invTotalGeomVol);
      const isHeroShard = geom.volume >= 0.0072 && this.shards.length + spawned.length < 24;

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
          // Never assign a canonical hull archetype that protrudes substantially outside the Voronoi cell bounds!
          if (asp[0] > hxM * 1.12 || asp[1] > hyM * 1.12 || asp[2] > hzM * 1.12) continue;
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

      const contype = isHeroShard ? MJ_CONTYPE_HERO_SHARD : MJ_CONTYPE_CULLET_SHARD;
      const conaff = isHeroShard
        ? MJ_CONAFFINITY_HERO_SHARD_ON_TABLE
        : MJ_CONAFFINITY_CULLET_ON_TABLE;
      setBodyGeomCollisionMask(this.model, bodyId, geomId, contype, conaff);
      this.worldCompat.markShardActive(bodyId, true, contype, conaff);

      const lxM = 2 * hxM, lyM = 2 * hyM, lzM = 2 * hzM;
      const ixSi = (shardExactMass / 12.0) * (lyM * lyM + lzM * lzM);
      const iySi = (shardExactMass / 12.0) * (lxM * lxM + lzM * lzM);
      const izSi = (shardExactMass / 12.0) * (lxM * lxM + lyM * lyM);
      updateFreeBodyMassAndInertiaO1(this.model, bodyId, dofAdr, shardExactMass, ixSi, iySi, izSi);

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
      const splaySpeedSim =
        ((0.24 + impactEnergy01 * 1.25) * (0.28 + 0.72 * proximity) * (0.75 + hash01(2.0) * 0.5)) /
        SIM_TIME_SCALE;
      const normalReboundSim = ((0.06 + impactEnergy01 * 0.16) * proximity) / SIM_TIME_SCALE;

      const rigidVx = lv.x + (av.y * worldOffset[2] - av.z * worldOffset[1]);
      const rigidVy = lv.y + (av.z * worldOffset[0] - av.x * worldOffset[2]);
      const rigidVz = lv.z + (av.x * worldOffset[1] - av.y * worldOffset[0]);
      const spinScale = 1.2 + proximity * (2.8 + impactEnergy01 * 3.2);

      spawned.push({
        geom,
        slotIndex,
        bodyId,
        geomId,
        worldPosDm: [sxDm, syDm, szDm],
        rawVelSim: [
          rigidVx + tx * splaySpeedSim + impactNormal[0] * normalReboundSim,
          rigidVy + ty * splaySpeedSim + impactNormal[1] * normalReboundSim,
          rigidVz + tz * splaySpeedSim + impactNormal[2] * normalReboundSim,
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
            this.model.body_inertia[bid3 + 2] * massScale
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
      const dvCorrX = lv.x - pxSum * invM;
      const dvCorrY = lv.y - pySum * invM;
      const dvCorrZ = lv.z - pzSum * invM;
      for (const s of spawned) {
        s.rawVelSim[0] += dvCorrX;
        s.rawVelSim[1] += dvCorrY;
        s.rawVelSim[2] += dvCorrZ;
      }

      let lSumX = 0, lSumY = 0, lSumZ = 0;
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
        const wx = s.rawSpinSim[0] - dOmega[0];
        const wy = s.rawSpinSim[1] - dOmega[1];
        const wz = s.rawSpinSim[2] - dOmega[2];

        const rbS = this.worldCompat.getRigidBody(s.bodyId)!;
        rbS.setLinvel({ x: vx, y: vy, z: vz }, true);
        rbS.setAngvel({ x: wx, y: wy, z: wz }, true);

        const firstVertex = s.slotIndex * SHARD_SLOT_STRIDE_VERTS;
        this.pendingSlotUploads.set(s.slotIndex, s.geom.renderVertices);

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
        };
        this.shards.push(shardState);
        this.bodyToShard.set(s.bodyId, shardState);
      }
      this.worldCompat.syncKinematics();
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

  public step(frameDt: number): void {
    if (!this.model || !this.data) return;
    const clampedDt = Math.min(0.085, Math.max(0.0, frameDt));
    this.simTime += clampedDt;

    if (this.turnPhase === 'EXTRACTED') {
      const decay = Math.exp(-clampedDt * 12.0);
      this.carriedHandVelXZ[0] *= decay;
      this.carriedHandVelXZ[1] *= decay;
    }

    // Fast O(1) MuJoCo 4.0 Sleeping-Island early-out when 100% of dynamic trees are asleep!
    const anyPendingWake = this.hasNonZeroAppliedWakeForces();
    if (this.data.ntree_awake === 0 && !anyPendingWake) {
      if (this.activeBlockId === null) {
        this.accumulatorSec = 0;
        this.wobbleSpeedDegPerSec = 0;
        this.updateMicroSplinters(clampedDt);
        this.updateTurnStateMachine(clampedDt);
        return;
      }
      if (this.turnPhase === 'EXTRACTED') {
        this.accumulatorSec = 0;
        this.wobbleSpeedDegPerSec = 0;
        this.advanceCarriedBlockOnly(clampedDt);
        this.updateMicroSplinters(clampedDt);
        this.updateTurnStateMachine(clampedDt);
        return;
      }
    }

    // Preserve full 600 Hz / 26-iteration Newton precision across all standing, extraction, and sandbox regimes;
    // once `turnPhase === 'COLLAPSED'`, step at 360 Hz (`6` substeps/frame) and scale Newton iterations (`26 -> 8`)
    // during active toppling so full-stack collapses run at 1.00x real-time wall-clock speed with zero slow-motion lag!
    const isCollapsed = this.turnPhase === 'COLLAPSED';
    const isActivelyToppling =
      isCollapsed ||
      (this.gameMode !== 'sandbox' && (this.leanAngleDeg > 2.2 || this.wobbleSpeedDegPerSec > 12.0)) ||
      this.shards.length > 24;
    const subDt = isCollapsed ? 1.0 / 360.0 : this.cfg.timestepSec;
    this.model.opt.timestep = subDt;
    this.model.opt.iterations = isActivelyToppling ? 8 : this.cfg.solverIterations;
    this.model.opt.ls_iterations = isActivelyToppling ? 4 : this.cfg.lsIterations;

    const maxSubsteps = isCollapsed ? 32 : 52;
    this.accumulatorSec = Math.min(maxSubsteps * subDt, this.accumulatorSec + clampedDt);

    let substeps = 0;
    let hadAwakeSubstep = false;
    while (this.accumulatorSec >= subDt * 0.999 && substeps < maxSubsteps) {
      const isLastSubstep =
        this.accumulatorSec < subDt * 1.999 || substeps + 1 >= maxSubsteps;
      this.worldCompat.applyCustomGravityScalesSubstep();
      this.updateMocapHandAndExtractionSubstep(subDt, isLastSubstep);

      const maxSpeedReal = this.snapshotPreStepVelocities();
      this.mj.mj_step(this.model, this.data);
      // Clear ephemeral wake nudges and per-step external forces immediately after mj_step consumes them
      this.data.qfrc_applied.fill(0);
      this.data.xfrc_applied.fill(0);

      if (this.data.ntree_awake > 0) {
        hadAwakeSubstep = true;
        this.dampGroundedMicroJitterShardsSubstep();
      }

      if (maxSpeedReal > 1.25 || this.extraPreStep.size > 0) {
        this.processWrenchCollisionsAndFractures(subDt);
      }

      this.accumulatorSec -= subDt;
      substeps++;
    }

    if (hadAwakeSubstep && this.data.ntree_awake > 0) {
      this.refreshClampingLoadsAndPhotoelasticStress(false);
    }

    this.updateMicroSplinters(clampedDt);
    this.syncTransformsAndRules();
    this.updateTurnStateMachine(clampedDt);
  }

  /**
   * Settles sub-millimeter facet rocking ONLY on grounded glass shards and tabletop rubble (`y < 2.8 cm`)
   * without touching `m.dof_damping` (which is explicit in `integrator="Euler"` and unstable on 0.2g chips),
   * and NEVER brakes elevated/falling blocks in the tower (`y >= 2.8 cm`), preserving exact 1.00g free-fall!
   */
  private dampGroundedMicroJitterShardsSubstep(): void {
    const m = this.model;
    const d = this.data;
    if (!m || !d || (this.shards.length === 0 && this.shatteredCount === 0 && this.turnPhase !== 'COLLAPSED')) return;

    const qpos = d.qpos;
    const qvel = d.qvel;
    const bodyTreeId = m.body_treeid;
    const treeAsleep = d.tree_asleep;

    const tickInc = this.turnPhase === 'COLLAPSED' ? 2 : 1;
    const isPostCollapse = this.turnPhase === 'COLLAPSED' || this.shatteredCount > 0;
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
        const yM = qpos[info.qposAdr + 1];
        const dof = info.dofAdr;
        const vx = qvel[dof + 0];
        const vy = qvel[dof + 1];
        const vz = qvel[dof + 2];
        const wx = qvel[dof + 3];
        const wy = qvel[dof + 4];
        const wz = qvel[dof + 5];
        const vSi = Math.hypot(vx, vy, vz);
        const wSi = Math.hypot(wx, wy, wz);
        // Strictly restrict intact-block micro-jitter damping to blocks already resting on the tabletop (`yM < 0.028 m`),
        // or after >95 quiet substeps of ultra-slow residual creep (`vSi < 0.014 m/s && Math.abs(vy) < 0.006 m/s`)
        // on post-collapse rubble piles—NEVER braking blocks starting to tip or fall!
        const groundedOnTable = yM > -0.015 && yM < 0.028 && Math.abs(vy) < 0.014 && vSi < 0.032 && wSi < 0.65;
        const settledPileCreep = yM > -0.015 && yM < 0.28 && Math.abs(vy) < 0.006 && vSi < 0.014 && wSi < 0.22;
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
            if (zeroSnap) info.settledAsleep = true;
          }
        } else if (vSi > 0.045 || wSi > 0.85 || Math.abs(vy) > 0.02) {
          info.quietTicks = 0;
          info.settledAsleep = false;
        }
      }
    }

    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const info = this.worldCompat.getMapping(s.bodyHandle);
      const treeId = bodyTreeId[s.bodyHandle];
      if (treeId >= 0 && treeAsleep[treeId] >= 0) {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        if (info) info.settledAsleep = false;
        continue;
      }
      const qposAdr = this.map.shardQposAdr[s.slotIndex];
      const dofAdr = this.map.shardDofAdr[s.slotIndex];
      const yM = qpos[qposAdr + 1];
      let vx = qvel[dofAdr + 0];
      let vy = qvel[dofAdr + 1];
      let vz = qvel[dofAdr + 2];
      let wx = qvel[dofAdr + 3];
      let wy = qvel[dofAdr + 4];
      let wz = qvel[dofAdr + 5];
      s.quietSubsteps = Math.min(65535, s.quietSubsteps + tickInc);

      // Guard first 14 substeps after Voronoi spawn against stiff inter-shard / rubble overlap-pop ejections
      if (s.quietSubsteps <= 14) {
        if (vy > 0.95) {
          vy = 0.95;
          qvel[dofAdr + 1] = vy;
        }
        const sp = Math.hypot(vx, vy, vz);
        if (sp > 2.35) {
          const k = 2.35 / sp;
          vx *= k; vy *= k; vz *= k;
          qvel[dofAdr + 0] = vx;
          qvel[dofAdr + 1] = vy;
          qvel[dofAdr + 2] = vz;
        }
      }
      if (yM > 0.23 && vy > 0.35) {
        vy = 0.35;
        qvel[dofAdr + 1] = vy;
      }

      if (yM < -0.04 || (Math.abs(vy) > 0.09 && s.quietSubsteps < 360)) {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        if (info) info.settledAsleep = false;
        continue;
      }
      const vSi = Math.hypot(vx, vy, vz);
      const wSi = Math.hypot(wx, wy, wz);
      const longLivedLowEnergy = s.quietSubsteps > 280 && vSi < 0.35 && Math.abs(vy) < 0.20;
      if ((vSi < 0.13 && wSi < 3.8) || longLivedLowEnergy) {
        const qCount = Math.min(65535, this.shardQuietSubsteps[s.slotIndex] + tickInc);
        this.shardQuietSubsteps[s.slotIndex] = qCount;
        if (qCount > 14 || longLivedLowEnergy) {
          const damp = qCount > 40 || s.quietSubsteps > 440 ? 0.25 : 0.72;
          const zeroSnap = qCount > 75 || s.quietSubsteps > 580;
          qvel[dofAdr + 0] = zeroSnap || Math.abs(vx * damp) < 5e-3 ? 0 : vx * damp;
          qvel[dofAdr + 1] = zeroSnap || Math.abs(vy * damp) < 5e-3 ? 0 : vy * damp;
          qvel[dofAdr + 2] = zeroSnap || Math.abs(vz * damp) < 5e-3 ? 0 : vz * damp;
          qvel[dofAdr + 3] = zeroSnap || Math.abs(wx * damp) < 4e-2 ? 0 : wx * damp;
          qvel[dofAdr + 4] = zeroSnap || Math.abs(wy * damp) < 4e-2 ? 0 : wy * damp;
          qvel[dofAdr + 5] = zeroSnap || Math.abs(wz * damp) < 4e-2 ? 0 : wz * damp;
          if (zeroSnap && info) info.settledAsleep = true;
        }
      } else {
        this.shardQuietSubsteps[s.slotIndex] = 0;
        if (info && vSi > 0.18) info.settledAsleep = false;
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
    let maxActiveLayer = 0;
    let sumX = 0;
    let sumZ = 0;
    let countInTower = 0;
    let maxTiltRad = 0;
    let sumTiltRad = 0;
    let sumWobbleOmega = 0;

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

      if (Math.hypot(tx, tz) > TABLE_RADIUS + 0.08 && geomConaffinity[b.colliderHandle] !== MJ_CONAFFINITY_BLOCK_OFF_RIM) {
        setBodyGeomCollisionMask(
          m,
          b.bodyHandle,
          b.colliderHandle,
          geomContype[b.colliderHandle],
          MJ_CONAFFINITY_BLOCK_OFF_RIM
        );
      }

      if (ty < -2.4) {
        this.shatterBlock(b, [0, 0, 0], 22.0, [0, 1, 0]);
        continue;
      }

      b.position[0] = tx; b.position[1] = ty; b.position[2] = tz;
      b.rotation[0] = rx; b.rotation[1] = ry; b.rotation[2] = rz; b.rotation[3] = rw;
      b.linVel[0] = lvx; b.linVel[1] = lvy; b.linVel[2] = lvz;
      b.angVel[0] = avx; b.angVel[1] = avy; b.angVel[2] = avz;

      if (!b.extracted && b.id !== this.activeBlockId) {
        const horizR = Math.hypot(tx, tz);
        if (horizR < 0.85 && ty > 0.04) {
          countInTower++;
          sumX += tx;
          sumZ += tz;
          if (ty + BLOCK_HALF_Y > maxTowerY) maxTowerY = ty + BLOCK_HALF_Y;
          if (
            ty >= Math.max(0.03, (b.assignedLayer - 0.85) * BLOCK_HEIGHT) &&
            b.assignedLayer > maxActiveLayer
          ) {
            maxActiveLayer = b.assignedLayer;
          }
          const upVec = quatRotateVec3(b.rotation, [0, 1, 0]);
          const tilt = Math.acos(Math.max(-1, Math.min(1, upVec[1])));
          if (tilt > maxTiltRad) maxTiltRad = tilt;
          sumTiltRad += tilt;
          sumWobbleOmega += Math.hypot(avx, avz);
        }
      }
    }

    for (let i = this.shards.length - 1; i >= 0; i--) {
      const s = this.shards[i];
      const adr = this.map.shardQposAdr[s.slotIndex];
      const tx = qpos[adr + 0] * DM_PER_M;
      const ty = qpos[adr + 1] * DM_PER_M;
      const tz = qpos[adr + 2] * DM_PER_M;
      if (Math.hypot(tx, tz) > TABLE_RADIUS + 0.04) {
        const ctype = geomContype[s.colliderHandle];
        const caff = ctype === MJ_CONTYPE_HERO_SHARD ? MJ_CONAFFINITY_HERO_SHARD_OFF_RIM : 0x02;
        if (geomConaffinity[s.colliderHandle] !== caff) {
          setBodyGeomCollisionMask(m, s.bodyHandle, s.colliderHandle, ctype, caff);
        }
      }
      if (ty < -0.65 || Math.hypot(tx, tz) > 8.5) {
        this.bodyToShard.delete(s.bodyHandle);
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
      s.position[0] = tx; s.position[1] = ty; s.position[2] = tz;
      s.rotation[0] = qpos[adr + 4]; s.rotation[1] = qpos[adr + 5]; s.rotation[2] = qpos[adr + 6]; s.rotation[3] = qpos[adr + 3];
    }

    if (countInTower > 0) {
      this.centerOfMassXZCm = [(sumX / countInTower) * 10, (sumZ / countInTower) * 10];
      const maxTiltDeg = (maxTiltRad * 180) / Math.PI;
      const meanTiltDeg = ((sumTiltRad / countInTower) * 180) / Math.PI;
      // Below 3.5 deg, weight global CoM column lean (`atan2(r_com, h)`) + trimmed row tilt so a single upper brick
      // settling 0.135 mm across a loose-slot gap does not misreport a > 0.5 deg global tower lean;
      // above 2.0 deg, transition smoothly to peak tipping row tilt (`maxTiltDeg`)!
      const comLeanDeg =
        maxTowerY > 0.25
          ? (Math.atan2(Math.hypot(sumX / countInTower, sumZ / countInTower), maxTowerY * 0.55) * 180) / Math.PI
          : meanTiltDeg;
      const smallAngleLean = Math.min(meanTiltDeg, comLeanDeg * 0.65 + meanTiltDeg * 0.35);
      const tipBlend = Math.max(0.0, Math.min(1.0, (maxTiltDeg - 2.35) / 3.0));
      const instantLean = smallAngleLean * (1.0 - tipBlend) + maxTiltDeg * tipBlend;
      this.leanAngleDeg = this.leanAngleDeg * 0.75 + Math.min(45, instantLean) * 0.25;
      const avgOmegaDeg = (((sumWobbleOmega * SIM_TIME_SCALE) / countInTower) * 180) / Math.PI;
      this.wobbleSpeedDegPerSec = this.wobbleSpeedDegPerSec * 0.7 + avgOmegaDeg * 0.3;
    }

    this.currentLayers = countInTower > 0 ? maxActiveLayer + 1 : 0;
    const heightCm = maxTowerY * 10.0;
    this.liveHeightCm = heightCm;
    if (this.turnPhase !== 'COLLAPSED') {
      if (heightCm > this.peakHeightCm) this.peakHeightCm = heightCm;
      if (this.currentLayers > this.peakLayers) this.peakLayers = this.currentLayers;
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
          'The stained-glass tower gave way and shattered across the walnut table!';
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
          ? 'The stained-glass tower gave way and shattered across the walnut table!'
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
   * Counterfactual `mj_copyState` Stability & AI Oracle ("Master Glazier AI"):
   * Clones the live `MjData` state into an isolated scratch workspace (`0` mutation of live gameplay!),
   * applies a virtual test pull to candidate legal blocks, steps MuJoCo's Newton solver forward, and ranks
   * blocks by physical slide ease vs. tower tilt risk!
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
        // Sample both low-load candidates (safe extractions) and high-load bearing columns (criticality contrast)
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
    const results: BlockCriticalityCandidate[] = [];

    for (const cand of chosen) {
      this.mj.mj_copyState(this.model, this.data, this.scratchOracleData, sig);
      this.scratchOracleData.eq_active[this.map.grabEqualityId] = 0;
      this.scratchOracleData.qfrc_applied.fill(0);
      this.scratchOracleData.xfrc_applied.fill(0);

      const info = this.worldCompat.getMapping(cand.bodyHandle);
      if (!info) continue;
      const startX = this.scratchOracleData.qpos[info.qposAdr + 0];
      const startZ = this.scratchOracleData.qpos[info.qposAdr + 2];
      const axisX = quatRotateVec3(cand.rotation, [1, 0, 0]);

      // Apply a calibrated 1.8 N outward probe force for `rolloutSteps`
      for (let s = 0; s < rolloutSteps; s++) {
        this.scratchOracleData.qfrc_applied[info.dofAdr + 0] = axisX[0] * 1.8;
        this.scratchOracleData.qfrc_applied[info.dofAdr + 2] = axisX[2] * 1.8;
        this.mj.mj_step(this.model, this.scratchOracleData);
      }

      const endX = this.scratchOracleData.qpos[info.qposAdr + 0];
      const endZ = this.scratchOracleData.qpos[info.qposAdr + 2];
      const slideMm = Math.hypot(endX - startX, endZ - startZ) * 1000.0;

      let maxTiltDeg = 0;
      let collapsed = false;
      for (const ob of this.blocks) {
        if (ob.shattered || ob.extracted || ob.id === cand.id) continue;
        const oInfo = this.worldCompat.getMapping(ob.bodyHandle);
        if (!oInfo) continue;
        const qw = this.scratchOracleData.qpos[oInfo.qposAdr + 3];
        const qx = this.scratchOracleData.qpos[oInfo.qposAdr + 4];
        const qy = this.scratchOracleData.qpos[oInfo.qposAdr + 5];
        const qz = this.scratchOracleData.qpos[oInfo.qposAdr + 6];
        const upVec = quatRotateVec3([qx, qy, qz, qw], [0, 1, 0]);
        const tilt = (Math.acos(Math.max(-1, Math.min(1, upVec[1]))) * 180) / Math.PI;
        if (tilt > maxTiltDeg) maxTiltDeg = tilt;
        if (tilt > 25.0) collapsed = true;
      }

      const clampN = this.computeBlockClampingLoadN(cand.id);
      const slideScore = Math.min(1.0, slideMm / 1.8);
      const tiltPenalty = Math.min(1.0, maxTiltDeg / 4.0);
      const safetyScore01 = collapsed
        ? 0.0
        : Math.max(0.0, Math.min(1.0, slideScore * 0.72 + (1.0 - tiltPenalty) * 0.28));

      results.push({
        blockId: cand.id,
        layer: cand.assignedLayer,
        slot: cand.assignedSlot,
        clampingLoadN: +clampN.toFixed(2),
        predictedMaxLeanDeg: +maxTiltDeg.toFixed(3),
        predictedSlideMm: +slideMm.toFixed(3),
        collapseTriggered: collapsed,
        safetyScore01: +safetyScore01.toFixed(3),
      });
    }

    results.sort((a, b) => b.safetyScore01 - a.safetyScore01);
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

  public dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
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

import RAPIER from '@dimforge/rapier3d-compat';
import {
  BLOCK_BEVEL_RADIUS,
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
  PHYSICS_DT,
  SHARD_SLOT_STRIDE_VERTS,
  SHATTER_IMPACT_SPEED,
  SIM_GRAVITY_Y,
  SIM_TIME_SCALE,
  STAINED_GLASS_PALETTE,
  TABLE_RADIUS,
  TABLE_THICKNESS,
  TARGET_BLOCK_MASS_KG,
  TraceryMotif,
} from './constants';
import { fractureBlockVoronoi } from './voronoiFracture';
import {
  Quat,
  quatFromAxisAngle,
  quatInverseRotateVec3,
  quatRotateVec3,
  Vec3,
} from '../math/vecMat';
import { GlassAudioEngine } from '../audio/GlassAudioEngine';

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
}

export interface ShardMeshAppendEvent {
  firstVertex: number;
  vertexData: Float32Array;
}

// Collision membership/filter bitmasks (16-bit upper = membership, 16-bit lower = filter)
// Group 0x0001 = Table, Group 0x0002 = Intact Glass Blocks, Group 0x0004 = Glass Shards
const GROUP_TABLE = (0x0001 << 16) | 0xffff;
const GROUP_BLOCK_ON_TABLE = (0x0002 << 16) | 0xffff;
const GROUP_FREEFALL_OFF_RIM = (0x0002 << 16) | 0xfffe; // Excludes GROUP_TABLE once past circular table radius!
const GROUP_SHARD_ON_TABLE = (0x0004 << 16) | 0x0007;        // Hero shards collide with Table + Blocks + Hero Shards
const GROUP_SHARD_CULLET_ON_TABLE = (0x0008 << 16) | 0x0003; // Small cullet shards collide with Table + Blocks
const GROUP_SHARD_OFF_RIM = (0x0004 << 16) | 0x0006;         // Excludes Table once past circular rim
const MAX_SPLINTERS = 320;

// Calibrated polished cathedral soda-lime glass friction (~15% lower stiction than wood)
const GLASS_STATIC_FRICTION = 0.34;
const GLASS_KINETIC_FRICTION = 0.25;
const SHARD_FRICTION = 0.33;
const TOP_HOVER_CLEARANCE_DM = 0.040; // 4.0 mm hover/drop clearance (+0.2 mm resting offset = 4.2 mm above lower glass)

let rapierInitialized = false;

export async function ensureRapierInitialized(): Promise<void> {
  if (!rapierInitialized) {
    await RAPIER.init();
    rapierInitialized = true;
  }
}

export class PhysicsWorld {
  private world!: RAPIER.World;
  private eventQueue!: RAPIER.EventQueue;
  private tableColliderHandles = new Set<number>();

  public blocks: GlassBlockState[] = [];
  public shards: GlassShardState[] = [];
  public splinters: MicroSplinter[] = [];
  public openTopSlots: TopPlacementSlot[] = [];

  // Fixed-stride O(1) 512-slot GPU shard vertex buffer allocator (`slotIndex * 132`)
  private freeShardSlots: number[] = [];
  private pendingSlotUploads = new Map<number, Float32Array>();
  private shardBufferResetPending: boolean = true;
  private nextShardId: number = 1;

  // Collider handle -> block or shard lookup
  private colliderToBlock = new Map<number, GlassBlockState>();
  private colliderToShard = new Map<number, GlassShardState>();

  // Game rules & turn state
  public gameMode: GameMode = 'solo';
  public playerCount: number = 2;
  public currentPlayer: number = 0;
  public turnPhase: TurnPhase = 'SELECT_BLOCK';
  public blocksStacked: number = 0;
  public shatteredCount: number = 0;
  public collapseReason: string | null = null;

  // Grab / Hand state
  public hoveredBlockId: number | null = null;
  public activeBlockId: number | null = null;
  public grabLocalPoint: Vec3 = [0, 0, 0];
  public grabWorldAnchor: Vec3 = [0, 0, 0];
  public grabCurrentWorldOnBlock: Vec3 = [0, 0, 0];
  public grabPlaneY: number = 0;
  public grabResistance01: number = 0;
  public isPointerDraggingBlock: boolean = false;
  public carriedFromDrag: boolean = false;
  private clearContactSubsteps: number = 0;
  private extractionArcTimer: number = 0;
  private grabRampSec: number = 0;
  private extractionStartPos: Vec3 = [0, 0, 0];
  private extractionOutDir: Vec3 = [1, 0, 0];
  private localWindowFrozenHandles = new Set<number>();
  private localWindowDisabledCols: number[] = [];
  private localWindowRefPos = new Map<number, Vec3>();
  private localWindowSavedClampingLoadN = new Map<number, number>();
  private localWindowCapBodyHandles: number[] = [];
  private localWindowBrokenForPull: boolean = false;
  private pendingQuietManifoldRestore: boolean = false;
  private blockDampingTier = new Map<number, number>();
  private blockQuietFrames = new Map<number, number>();
  private blockLastCrazeTime = new Map<number, number>();

  // Listener camera pose for stereo-panned modal glass audio
  public listenerCamPos: Vec3 = [0, 2.2, 5.2];
  public listenerCamRight: Vec3 = [1, 0, 0];

  // Continuous skill-based free-hand top placement state
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
  private carriedRawLocalVL: [number, number] = [0, 0];
  private hasCustomCarriedHover: boolean = false;
  private lastHoverTargetXZ: [number, number] | null = null;
  private lastHoverTimeSec: number = 0;
  private carriedHandVelXZ: [number, number] = [0, 0];
  private settleTimer: number = 0;
  private readonly SETTLE_DURATION = 1.35;

  // Telemetry
  public leanAngleDeg: number = 0;
  public wobbleSpeedDegPerSec: number = 0;
  public centerOfMassXZCm: [number, number] = [0, 0];
  public currentLayers: number = INITIAL_LAYERS;
  public peakLayers: number = INITIAL_LAYERS;
  public liveHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;
  public peakHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;
  public lockedMinRow: number | null = INITIAL_LAYERS;

  private accumulator: number = 0;
  private simTime: number = 0;
  private audio: GlassAudioEngine;

  constructor(audio: GlassAudioEngine, initialLayers: number = INITIAL_LAYERS) {
    this.audio = audio;
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

  public getWorld(): RAPIER.World {
    return this.world;
  }

  public rebuildTower(
    mode?: GameMode,
    players?: number,
    layers: number = INITIAL_LAYERS
  ): void {
    if (mode) this.gameMode = mode;
    if (players !== undefined) this.playerCount = Math.max(2, Math.min(4, players));
    const towerLayers = Math.max(1, Math.min(32, Math.round(layers)));

    if (this.eventQueue) {
      this.eventQueue.free();
    }
    if (this.world) {
      this.world.free();
    }

    this.world = new RAPIER.World({ x: 0.0, y: SIM_GRAVITY_Y, z: 0.0 });
    this.eventQueue = new RAPIER.EventQueue(true);

    // Calibrated 120 Hz Froude-scaled Rapier SmallStepsPGS solver (< 0.15 mm 18L sag, 100% sleep <= 1.2s)
    const params = this.world.integrationParameters;
    params.dt = PHYSICS_DT;
    params.lengthUnit = 0.25;
    params.normalizedAllowedLinearError = 0.000008;
    params.numSolverIterations = 6;
    params.numAdditionalFrictionIterations = 1;
    params.numInternalPgsIterations = 2;
    params.contact_natural_frequency = 219;
    params.maxCcdSubsteps = 1;

    // Analytical SAT cuboid oak tabletop (y = 0.0); Min-combined 0.08 wood restitution vs 0.18 glass-on-glass
    this.tableColliderHandles.clear();
    const tableBody = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(0, -TABLE_THICKNESS * 0.5, 0)
    );
    const tablePlate = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(TABLE_RADIUS + 0.5, TABLE_THICKNESS * 0.5, TABLE_RADIUS + 0.5)
        .setFriction(0.54)
        .setRestitution(0.08)
        .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Min)
        .setCollisionGroups(GROUP_TABLE),
      tableBody
    );
    this.tableColliderHandles.add(tablePlate.handle);

    // Reset state & O(1) 512-slot fixed-stride pool
    this.blocks = [];
    this.shards = [];
    this.splinters = [];
    this.colliderToBlock.clear();
    this.colliderToShard.clear();
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
    this.localWindowFrozenHandles.clear();
    this.localWindowDisabledCols = [];
    this.localWindowRefPos.clear();
    this.localWindowSavedClampingLoadN.clear();
    this.localWindowCapBodyHandles = [];
    this.localWindowBrokenForPull = false;
    this.pendingQuietManifoldRestore = false;
    this.blockDampingTier.clear();
    this.blockQuietFrames.clear();
    this.blockLastCrazeTime.clear();
    this.currentSubstepDt = PHYSICS_DT;
    this.settleTimer = 0;
    this.accumulator = 0;
    this.simTime = 0;
    this.leanAngleDeg = 0;
    this.wobbleSpeedDegPerSec = 0;
    this.currentLayers = towerLayers;
    this.peakLayers = towerLayers;
    this.liveHeightCm = towerLayers * BLOCK_HEIGHT * 10;
    this.peakHeightCm = towerLayers * BLOCK_HEIGHT * 10;
    this.lockedMinRow = this.gameMode === 'sandbox' ? null : towerLayers;

    const rBevel = BLOCK_BEVEL_RADIUS;

    let id = 0;
    for (let layer = 0; layer < towerLayers; layer++) {
      const isOdd = layer % 2 === 1;
      const yaw = isOdd ? Math.PI * 0.5 : 0.0;

      // Alternate loose block between middle (slot 1) and symmetric outer pair so tower CoM stays balanced
      const centerIsLoose = layer % 2 === 0;

      for (let slot = 0; slot < 3; slot++) {
        const offset = (slot - 1) * BLOCK_WIDTH;
        const jx = Math.sin(id * 12.9898) * 0.0022;
        const jz = Math.cos(id * 78.233) * 0.0022;
        const jYaw = Math.sin(id * 39.17) * 0.0032;

        const px = (isOdd ? offset : 0) + jx;
        const py = (layer + 0.5) * BLOCK_HEIGHT;
        const pz = (isOdd ? 0 : offset) + jz;
        const quat = quatFromAxisAngle([0, 1, 0], yaw + jYaw);

        // Calibrated Jenga thickness tolerance: 1 block per row has -0.08 mm clearance (0 N upper pinch)
        // while the other 2 blocks carry the column load in static equilibrium.
        const isLoose = centerIsLoose ? slot === 1 : slot === (layer % 4 === 1 ? 0 : 2);
        const heightMicroDelta = isLoose ? -0.0008 : 0.0;
        const colHalfX = BLOCK_HALF_X - 0.0008;
        const colHalfY = BLOCK_HALF_Y + heightMicroDelta;
        const colHalfZ = BLOCK_HALF_Z - 0.0014;

        const innerHx = Math.max(0.01, colHalfX - rBevel);
        const innerHy = Math.max(0.01, colHalfY - rBevel);
        const innerHz = Math.max(0.01, colHalfZ - rBevel);
        // Rapier 0.14.0 roundCuboid computes mass from the inner box only; compensate density for exact 71.10g mass
        const innerVol = 8.0 * innerHx * innerHy * innerHz;
        const compensatedDensity = TARGET_BLOCK_MASS_KG / innerVol;

        const rbDesc = RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(px, py, pz)
          .setRotation({ x: quat[0], y: quat[1], z: quat[2], w: quat[3] })
          .setLinearDamping(0.025)
          .setAngularDamping(0.035);

        const body = this.world.createRigidBody(rbDesc);

        const initStaticFriction = GLASS_STATIC_FRICTION;
        const colDesc = RAPIER.ColliderDesc.roundCuboid(innerHx, innerHy, innerHz, rBevel)
          .setDensity(compensatedDensity)
          .setFriction(initStaticFriction)
          .setRestitution(0.18)
          .setCollisionGroups(GROUP_BLOCK_ON_TABLE)
          .setActiveEvents(RAPIER.ActiveEvents.CONTACT_FORCE_EVENTS)
          .setContactForceEventThreshold(2.2);

        const collider = this.world.createCollider(colDesc, body);

        // Harmonious stained-glass palette distribution; pure hand-rolled cathedral waterglass (`motif = 0`)
        const swatchId = (layer * 3 + slot * 2 + Math.floor(layer / 2)) % STAINED_GLASS_PALETTE.length;
        const motif: TraceryMotif = 0;

        const blockState: GlassBlockState = {
          id,
          bodyHandle: body.handle,
          colliderHandle: collider.handle,
          swatchId,
          motif,
          rippleSeed: (id + 1) * 1.6180339,
          initialLayer: layer,
          assignedLayer: layer,
          assignedSlot: slot,
          isLooseSlot: isLoose,
          position: [px, py, pz],
          rotation: [...quat],
          prePos: [px, py, pz],
          preQuat: [...quat],
          linVel: [0, 0, 0],
          angVel: [0, 0, 0],
          currentFriction: initStaticFriction,
          isMicroDamped: false,
          damage: 0,
          crackLocalOrigin: [0, 0, 0],
          shattered: false,
          extracted: false,
          isLegalTarget: layer < towerLayers - 1,
        };

        this.blocks.push(blockState);
        this.colliderToBlock.set(collider.handle, blockState);
        id++;
      }
    }

    // Run quiet warm-up substeps with fracture events suppressed so stack settles cleanly
    const warmupSteps = Math.min(24, 6 + towerLayers);
    for (let i = 0; i < warmupSteps; i++) {
      this.world.step();
    }
    this.syncTransformsAndRules();
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

  private wakeTowerForInteraction(): void {
    for (const b of this.blocks) {
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (rb && rb.isSleeping()) {
        rb.wakeUp();
      }
    }
  }

  /**
   * Computes upper-row clamping pinch force `F_n` (in SI Newtons) resting on `blockId`.
   * Works both when bodies are awake and when sleeping (Rapier preserves last resting manifold impulses).
   */
  public computeBlockClampingLoadN(blockId: number): number {
    const block = this.blocks[blockId];
    if (!block || block.shattered || block.extracted) return 0;
    if (this.localWindowFrozenHandles.has(block.bodyHandle)) {
      return this.localWindowSavedClampingLoadN.get(blockId) ?? 0;
    }
    const col = this.world.getCollider(block.colliderHandle);
    if (!col) return 0;

    let upperImpulseSum = 0;
    this.world.contactPairsWith(col, (otherCol) => {
      const otherBlk = this.colliderToBlock.get(otherCol.handle);
      if (!otherBlk || otherBlk.shattered || otherBlk.extracted) return;
      if (otherBlk.position[1] <= block.position[1] + BLOCK_HEIGHT * 0.45) return;

      this.world.contactPair(col, otherCol, (manifold) => {
        const nc = manifold.numContacts();
        for (let k = 0; k < nc; k++) {
          const imp = manifold.contactImpulse(k);
          if (imp > 0) upperImpulseSum += imp;
        }
      });
    });

    const dtDiv = this.currentSubstepDt > 0 ? this.currentSubstepDt : PHYSICS_DT;
    return upperImpulseSum / dtDiv;
  }

  /**
   * Calibrated finger tap / poke to test block looseness with load-coupled modal resonance (`Q=48` loose vs `Q=14` pinched).
   */
  public tapNudgeBlock(blockId: number, localNormal: Vec3, localPoint: Vec3): boolean {
    const block = this.blocks[blockId];
    if (!block || block.shattered || block.extracted) return false;
    if (this.gameMode !== 'sandbox' && this.turnPhase !== 'SELECT_BLOCK') return false;
    if (this.gameMode !== 'sandbox' && !block.isLegalTarget) return false;

    const rb = this.world.getRigidBody(block.bodyHandle);
    if (!rb) return false;

    const clampingLoadN = this.computeBlockClampingLoadN(blockId);
    const pinch01 = Math.max(0.0, Math.min(1.0, clampingLoadN / 3.5));

    // Determine lengthwise push/pull direction in world space
    const axisX = quatRotateVec3(block.rotation, [1, 0, 0]);
    let dirSign = 1;
    if (Math.abs(localNormal[0]) > 0.65) {
      dirSign = -Math.sign(localNormal[0]);
    } else {
      dirSign = localPoint[0] >= 0 ? 1 : -1;
    }

    const impulseMag = this.gameMode === 'sandbox' ? 0.14 : 0.048;
    rb.wakeUp();
    rb.applyImpulse(
      {
        x: axisX[0] * dirSign * impulseMag,
        y: 0.0,
        z: axisX[2] * dirSign * impulseMag,
      },
      true
    );
    this.audio.playTapTestResonance(
      pinch01,
      block.position,
      this.listenerCamPos,
      this.listenerCamRight
    );
    return true;
  }

  /**
   * Starts pulling/grabbing a block at a specific local contact point.
   */
  public startGrabBlock(blockId: number, localPoint: Vec3, worldPoint: Vec3): boolean {
    const block = this.blocks[blockId];
    if (!block || block.shattered) return false;

    if (this.gameMode !== 'sandbox') {
      if (this.turnPhase === 'COLLAPSED' || this.turnPhase === 'SETTLING') return false;
      if (this.turnPhase === 'EXTRACTED' && block.id !== this.activeBlockId) return false;
      if (this.turnPhase === 'SELECT_BLOCK' && !block.isLegalTarget) return false;
    }

    this.releaseLocalSolverWindow(false, true);
    this.localWindowBrokenForPull = false;
    for (const b of this.blocks) {
      if (!b.shattered && Math.abs(b.assignedLayer - block.assignedLayer) <= 1) {
        this.world.getRigidBody(b.bodyHandle)?.wakeUp();
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

    if (!block.extracted && this.turnPhase === 'SELECT_BLOCK') {
      this.turnPhase = 'PULLING';
    }
    return true;
  }

  public updateGrabTarget(worldTargetOnPlane: Vec3): void {
    this.grabWorldAnchor[0] = worldTargetOnPlane[0];
    this.grabWorldAnchor[1] = this.grabPlaneY;
    this.grabWorldAnchor[2] = worldTargetOnPlane[2];
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
      // Tie-break dead-center hover toward tower support center so outer blocks push inward onto the stack
      const pPlusX = ax + bestAxis[0] * minOverlap - centerRefX;
      const pPlusZ = az + bestAxis[1] * minOverlap - centerRefZ;
      const pMinusX = ax - bestAxis[0] * minOverlap - centerRefX;
      const pMinusZ = az - bestAxis[1] * minOverlap - centerRefZ;
      sign =
        pPlusX * pPlusX + pPlusZ * pPlusZ <= pMinusX * pMinusX + pMinusZ * pMinusZ
          ? 1
          : -1;
    }

    return {
      overlap: minOverlap,
      dirX: bestAxis[0] * sign,
      dirZ: bestAxis[1] * sign,
    };
  }

  public get isExtractionLiftReady(): boolean {
    return this.turnPhase === 'EXTRACTED' && this.extractionArcTimer >= 0.20;
  }

  private static wrapYawRad(rad: number): number {
    let r = rad;
    while (r > Math.PI) r -= Math.PI * 2;
    while (r < -Math.PI) r += Math.PI * 2;
    return r;
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

  /**
   * Adjusts the carried block's yaw angle relative to the top layer's perpendicular row axis.
   * Supports smooth mouse-wheel/Q/E micro-angling and R 90-degree toggles during `EXTRACTED`,
   * stopping flush against existing top-row blocks if a rotation has no overlap-free planar room.
   */
  public rotateCarriedBlock(deltaYawRad: number): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    const prevYaw = this.carriedYawOffsetRad;
    const rawV = this.carriedRawLocalVL[0];
    const rawU = this.carriedRawLocalVL[1];
    this.hasCustomCarriedHover = true;

    const targetYaw = PhysicsWorld.wrapYawRad(prevYaw + deltaYawRad);
    this.carriedYawOffsetRad = targetYaw;
    if (this.resolveCarriedPlacementPose(rawV, rawU, false)) {
      return;
    }

    for (let step = 11; step >= 0; step--) {
      const candYaw = PhysicsWorld.wrapYawRad(prevYaw + deltaYawRad * (step / 12));
      this.carriedYawOffsetRad = candYaw;
      if (this.resolveCarriedPlacementPose(rawV, rawU, false)) {
        return;
      }
    }

    this.carriedYawOffsetRad = 0;
    this.resolveCarriedPlacementPose(rawV, rawU, true);
  }

  public resetCarriedBlockRotation(): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    this.carriedYawOffsetRad = 0;
    this.resolveCarriedPlacementPose(this.carriedRawLocalVL[0], this.carriedRawLocalVL[1], true);
  }

  /**
   * Resolves continuous free-hand `(desiredV, desiredU)` placement coordinates on the active top support
   * plane with 2D OBB side-abutment (`+0.4 mm` collider clearance) so blocks kiss flush bevel-to-bevel
   * against existing top-row blocks without ever spawning intersecting colliders.
   */
  private resolveCarriedPlacementPose(
    desiredV: number,
    desiredU: number,
    allowYawRelaxFallback: boolean = true
  ): boolean {
    const frame = this.topSupportFrame;
    const totalYaw = frame.defaultYawRad + this.carriedYawOffsetRad;
    const yawQuat = quatFromAxisAngle([0, 1, 0], totalYaw);
    const carriedQuat = PhysicsWorld.quatMul(frame.tiltQuat, yawQuat);
    const { uX: uAx, uZ: uAz } = PhysicsWorld.getBlockXZAxes(carriedQuat);
    const { maxAbsV, maxAbsU } = PhysicsWorld.getCarriedSupportBounds(this.carriedYawOffsetRad);

    const colHx = BLOCK_HALF_X - 0.0008;
    const colHz = BLOCK_HALF_Z - 0.0014;
    const abutmentClearance = 0.0040; // +0.40 mm bevel-to-bevel collider clearance
    const minSafeClearance = 0.0022;  // >= 0.22 mm strict positive collider separation

    const obstacles: Array<{
      x: number;
      z: number;
      uX: [number, number];
      uZ: [number, number];
    }> = [];
    for (const b of this.blocks) {
      if (
        !b.shattered &&
        !b.extracted &&
        b.id !== this.activeBlockId &&
        b.assignedLayer === frame.targetLayer
      ) {
        if (Math.hypot(b.position[0] - frame.layerCenter[0], b.position[2] - frame.layerCenter[2]) < 0.95) {
          const axes = PhysicsWorld.getBlockXZAxes(b.rotation);
          obstacles.push({
            x: b.position[0],
            z: b.position[2],
            uX: axes.uX,
            uZ: axes.uZ,
          });
        }
      }
    }

    const maxOverlapAt = (px: number, pz: number, clearance: number): number => {
      let worst = 0;
      for (let i = 0; i < obstacles.length; i++) {
        const ob = obstacles[i];
        const hit = PhysicsWorld.obb2DOverlapDepth(
          px,
          pz,
          uAx,
          uAz,
          colHx + clearance,
          colHz + clearance,
          ob.x,
          ob.z,
          ob.uX,
          ob.uZ,
          colHx,
          colHz,
          frame.layerCenter[0],
          frame.layerCenter[2]
        );
        if (hit && hit.overlap > worst) {
          worst = hit.overlap;
        }
      }
      return worst;
    };

    const clampedV = Math.max(-maxAbsV, Math.min(maxAbsV, desiredV));
    const clampedU = Math.max(-maxAbsU, Math.min(maxAbsU, desiredU));

    let px =
      frame.layerCenter[0] +
      frame.lengthAxis[0] * clampedV +
      frame.spacingAxis[0] * clampedU;
    let pz =
      frame.layerCenter[2] +
      frame.lengthAxis[2] * clampedV +
      frame.spacingAxis[2] * clampedU;

    if (obstacles.length > 0) {
      for (let iter = 0; iter < 6; iter++) {
        let moved = false;
        for (let i = 0; i < obstacles.length; i++) {
          const ob = obstacles[i];
          const mtv = PhysicsWorld.obb2DOverlapDepth(
            px,
            pz,
            uAx,
            uAz,
            colHx + abutmentClearance,
            colHz + abutmentClearance,
            ob.x,
            ob.z,
            ob.uX,
            ob.uZ,
            colHx,
            colHz,
            frame.layerCenter[0],
            frame.layerCenter[2]
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
    const outOfSupportBounds =
      Math.abs(finalV) > maxAbsV + 0.01 || Math.abs(finalU) > maxAbsU + 0.01;
    if (
      obstacles.length > 0 &&
      (outOfSupportBounds || maxOverlapAt(px, pz, minSafeClearance) > 1e-5)
    ) {
      // Cursor aimed into a narrow blind seam between 2 blocks or outward past an edge block:
      // find the closest completely overlap-free flush pose on the top surface.
      const candVs: number[] = [clampedV, Math.max(-0.14, Math.min(0.14, clampedV)), 0];
      if (maxAbsV > 0.26) {
        const vSteps = Math.round(maxAbsV / 0.03);
        for (let kv = -vSteps; kv <= vSteps; kv++) {
          candVs.push(kv * 0.03);
        }
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
          (colHx + abutmentClearance) *
            Math.abs(uAx[0] * frame.spacingAxis[0] + uAx[1] * frame.spacingAxis[2]) +
          (colHz + abutmentClearance) *
            Math.abs(uAz[0] * frame.spacingAxis[0] + uAz[1] * frame.spacingAxis[2]) +
          colHx * Math.abs(ob.uX[0] * frame.spacingAxis[0] + ob.uX[1] * frame.spacingAxis[2]) +
          colHz * Math.abs(ob.uZ[0] * frame.spacingAxis[0] + ob.uZ[1] * frame.spacingAxis[2]);
        candUs.push(obU - rSpan, obU + rSpan);
      }
      for (let i = 0; i < obUs.length; i++) {
        for (let j = i + 1; j < obUs.length; j++) {
          candUs.push((obUs[i] + obUs[j]) * 0.5);
        }
      }
      const uSteps = Math.floor(maxAbsU / 0.015);
      for (let k = -uSteps; k <= uSteps; k++) {
        candUs.push(k * 0.015);
      }

      let bestCost = Infinity;
      let foundSafe = false;
      for (let iv = 0; iv < candVs.length; iv++) {
        const cv = candVs[iv];
        if (Math.abs(cv) > maxAbsV) continue;
        for (let iu = 0; iu < candUs.length; iu++) {
          const cu = candUs[iu];
          if (Math.abs(cu) > maxAbsU) continue;
          const cpx =
            frame.layerCenter[0] +
            frame.lengthAxis[0] * cv +
            frame.spacingAxis[0] * cu;
          const cpz =
            frame.layerCenter[2] +
            frame.lengthAxis[2] * cv +
            frame.spacingAxis[2] * cu;
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
            if (this.resolveCarriedPlacementPose(desiredV, desiredU, false)) {
              return true;
            }
          }
        }
        poseValid = false;
        // Preserve MTV-separated pose rather than ever clamping back into collider overlap
      }
    } else {
      finalV = Math.max(-maxAbsV, Math.min(maxAbsV, finalV));
      finalU = Math.max(-maxAbsU, Math.min(maxAbsU, finalU));
    }

    const landX =
      frame.layerCenter[0] +
      frame.lengthAxis[0] * finalV +
      frame.spacingAxis[0] * finalU;
    const landY =
      frame.layerCenter[1] +
      frame.lengthAxis[1] * finalV +
      frame.spacingAxis[1] * finalU;
    const landZ =
      frame.layerCenter[2] +
      frame.lengthAxis[2] * finalV +
      frame.spacingAxis[2] * finalU;

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

  /**
   * Updates continuous free-hand X/Z placement pose & side-abutment while carrying an extracted block.
   */
  public updateCarriedHover(rayOrigin: Vec3, rayDir: Vec3): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    if (this.openTopSlots.length === 0) {
      this.computeOpenTopSlots();
    }

    const frame = this.topSupportFrame;
    const lc = frame.layerCenter;
    const n = frame.upNormal;

    let hitX = lc[0];
    let hitY = lc[1];
    let hitZ = lc[2];

    const denom = rayDir[0] * n[0] + rayDir[1] * n[1] + rayDir[2] * n[2];
    if (Math.abs(denom) > 1e-4) {
      const t =
        ((lc[0] - rayOrigin[0]) * n[0] +
          (lc[1] - rayOrigin[1]) * n[1] +
          (lc[2] - rayOrigin[2]) * n[2]) /
        denom;
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
    const rawV =
      rx * frame.lengthAxis[0] + ry * frame.lengthAxis[1] + rz * frame.lengthAxis[2];
    const rawU =
      rx * frame.spacingAxis[0] + ry * frame.spacingAxis[1] + rz * frame.spacingAxis[2];

    const { maxAbsV, maxAbsU } = PhysicsWorld.getCarriedSupportBounds(this.carriedYawOffsetRad);
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

  /**
   * Releases pointer drag.
   * - If block is still inside tower (PULLING), lets go of block (player may nudge another block!).
   * - If block was dragged directly over the top surface after extraction, places it at the live free-hand pose!
   */
  public endGrab(dropIfPoisedOverTower: boolean = false): void {
    this.isPointerDraggingBlock = false;
    this.grabResistance01 = 0;
    this.audio.updateSlideFriction(0, 0);

    if (this.activeBlockId !== null) {
      const blk = this.blocks[this.activeBlockId];
      if (blk && !blk.shattered && blk.currentFriction !== GLASS_STATIC_FRICTION) {
        const col = this.world.getCollider(blk.colliderHandle);
        if (col) {
          col.setFriction(GLASS_STATIC_FRICTION);
          blk.currentFriction = GLASS_STATIC_FRICTION;
        }
      }
    }

    if (this.turnPhase === 'PULLING') {
      // Block was not extracted yet; rebuild resting contact manifolds across the stack and return to SELECT_BLOCK
      this.releaseLocalSolverWindow(true, true);
      this.activeBlockId = null;
      this.turnPhase = 'SELECT_BLOCK';
      return;
    }

    this.releaseLocalSolverWindow(true, true);
    if (this.turnPhase === 'EXTRACTED' && dropIfPoisedOverTower) {
      this.releaseExtractedBlockOntoTop();
    }
  }

  /**
   * Releases the carried extracted block from ~4.2 mm above the active top support surface at the player's
   * continuous free-hand X/Z/yaw pose (with 20% clamped hand-velocity coupling) under real Rapier gravity!
   */
  public releaseExtractedBlockOntoTop(): boolean {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return false;
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

    const rb = this.world.getRigidBody(block.bodyHandle);
    const col = this.world.getCollider(block.colliderHandle);
    if (!rb || !col) return false;

    this.releaseLocalSolverWindow(true, false);

    // Re-enable solid collision, static glass friction, & gravity, poised ~4.0 mm (+0.2 mm resting gap = 4.2 mm) above support row
    this.isPointerDraggingBlock = false;
    col.setSensor(false);
    col.setFriction(GLASS_STATIC_FRICTION);
    block.currentFriction = GLASS_STATIC_FRICTION;
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

    // Subtle natural hand-velocity coupling (20% horizontal release speed, clamped <= 2.2 dm/s real)
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
      if (
        !ob.shattered &&
        !ob.extracted &&
        ob.id !== block.id &&
        ob.assignedLayer === frame.targetLayer
      ) {
        usedSlots.add(ob.assignedSlot);
      }
    }
    const relDx = this.carriedLandingPos[0] - frame.layerCenter[0];
    const relDz = this.carriedLandingPos[2] - frame.layerCenter[2];
    const placedU = relDx * frame.spacingAxis[0] + relDz * frame.spacingAxis[2];
    const idealSlot = Math.max(0, Math.min(2, Math.round(placedU / BLOCK_WIDTH + 1)));
    let assignedSlot = idealSlot;
    if (usedSlots.has(assignedSlot)) {
      const candidates = [0, 1, 2].sort(
        (a, b) => Math.abs(a - idealSlot) - Math.abs(b - idealSlot)
      );
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

  /**
   * Sandbox topple impulse: asymmetric hinge buckle (kicks out slots 0-1 on layers 1-2 while leaving
   * layer 0 + slot 2 as a natural hinge stump and tipping upper stories) so 0 blocks shatter inside
   * the upright stack, upper rows shatter on table impact (>= 300 shards), and surviving lower/mid
   * bricks retain realistic tumble, internal crazing fractures, and precarious balance.
   */
  public triggerSandboxCollapseImpulse(): void {
    this.releaseLocalSolverWindow(true, false);
    this.wakeTowerForInteraction();
    for (const b of this.blocks) {
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb) continue;
      const t = rb.translation();
      if (b.assignedLayer === 0) {
        if (b.assignedSlot === 0) {
          rb.applyImpulse({ x: -0.055, y: 0.0, z: -0.02 }, true);
        }
      } else if (b.assignedLayer <= 2) {
        if (b.assignedSlot <= 1) {
          const baseScale = b.assignedSlot === 0 ? -0.215 : -0.175;
          rb.applyImpulse({ x: baseScale, y: 0.004, z: -0.048 }, true);
        } else {
          rb.applyImpulse({ x: 0.028, y: 0.0, z: 0.015 }, true);
        }
      } else {
        const hScale = 0.076 + Math.min(16, b.assignedLayer) * 0.0044;
        rb.applyImpulseAtPoint(
          { x: hScale, y: 0.0, z: hScale * 0.32 },
          { x: t.x, y: t.y + 0.02, z: t.z + 0.025 },
          true
        );
      }
    }
    this.audio.playGlassImpact(
      0.62,
      0.96,
      [0, 0.6, 0],
      this.listenerCamPos,
      this.listenerCamRight
    );
  }

  private currentSubstepDt: number = PHYSICS_DT;
  private maxSubstepSpeedBoundReal: number = 0;

  private areNonCarriedBodiesSleeping(): boolean {
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (rb && !rb.isSleeping()) return false;
    }
    for (let i = 0; i < this.shards.length; i++) {
      const rb = this.world.getRigidBody(this.shards[i].bodyHandle);
      if (rb && !rb.isSleeping()) return false;
    }
    return true;
  }

  private advanceCarriedBlockOnly(realDt: number): void {
    if (this.activeBlockId === null) return;
    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) return;
    const rb = this.world.getRigidBody(block.bodyHandle);
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
    const blend = Math.min(1.0, 1.0 - Math.pow(1.0 - 0.22, (realDt * SIM_TIME_SCALE) / PHYSICS_DT));
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

  public step(frameDt: number): void {
    const clampedDt = Math.min(0.025, Math.max(0.0, frameDt));
    this.simTime += clampedDt;
    if (this.turnPhase === 'EXTRACTED') {
      const decay = Math.exp(-clampedDt * 12.0);
      this.carriedHandVelXZ[0] *= decay;
      this.carriedHandVelXZ[1] *= decay;
    }

    if (this.areNonCarriedBodiesSleeping()) {
      if (this.activeBlockId === null) {
        this.accumulator = 0;
        this.wobbleSpeedDegPerSec = 0;
        this.updateMicroSplinters(clampedDt);
        this.updateTurnStateMachine(clampedDt);
        return;
      }
      if (this.turnPhase === 'EXTRACTED') {
        this.accumulator = 0;
        this.wobbleSpeedDegPerSec = 0;
        this.advanceCarriedBlockOnly(clampedDt);
        this.updateMicroSplinters(clampedDt);
        this.updateTurnStateMachine(clampedDt);
        return;
      }
    }

    const activeBlkForWin =
      this.activeBlockId !== null ? this.blocks[this.activeBlockId] : undefined;
    if (
      activeBlkForWin &&
      !activeBlkForWin.shattered &&
      !activeBlkForWin.extracted &&
      (this.turnPhase === 'PULLING' || this.gameMode === 'sandbox') &&
      this.shards.length === 0 &&
      !this.localWindowBrokenForPull &&
      this.leanAngleDeg < 1.2 &&
      this.wobbleSpeedDegPerSec < 35.0
    ) {
      this.engageOrValidateLocalSolverWindow();
    } else if (this.localWindowFrozenHandles.size > 0) {
      this.releaseLocalSolverWindow(true, false);
    }

    const massiveShardPile = this.shards.length > 200;
    const heavyShardPile = !massiveShardPile && this.shards.length > 80;
    const mediumShardPile = !massiveShardPile && !heavyShardPile && this.shards.length >= 1;
    const gentleActivePlay =
      this.shards.length === 0 &&
      this.activeBlockId !== null &&
      (this.turnPhase === 'PULLING' || this.turnPhase === 'EXTRACTED') &&
      this.leanAngleDeg < 2.5;

    // Never jump dtScale by 6.32x or drop friction/solver iterations below Euler/Baumgarte stability:
    // - Intact play & topple (0 shards): invariant dt=1/120, iters=6, fric=1, pgs=2, freq=200 (0 compliance jumps!)
    // - Shard piles: bounded dtScale <= 2.0 with proportionally scaled sub-Nyquist contact_natural_frequency (erp <= 0.35)
    const dtScale = massiveShardPile
      ? 2.0
      : heavyShardPile
        ? 1.6
        : mediumShardPile
          ? 1.25
          : 1.0;
    const stepDt = PHYSICS_DT * dtScale;
    this.currentSubstepDt = stepDt;
    const maxSubsteps = massiveShardPile
      ? 4
      : heavyShardPile
        ? 5
        : mediumShardPile
          ? 6
          : 8;

    this.accumulator = Math.min(
      maxSubsteps * stepDt,
      this.accumulator + clampedDt * SIM_TIME_SCALE
    );

    const params = this.world.integrationParameters;
    params.dt = stepDt;
    if (massiveShardPile) {
      params.numSolverIterations = 4;
      params.numAdditionalFrictionIterations = 1;
      params.numInternalPgsIterations = 1;
      params.contact_natural_frequency = 34;
    } else if (heavyShardPile) {
      params.numSolverIterations = 4;
      params.numAdditionalFrictionIterations = 1;
      params.numInternalPgsIterations = 2;
      params.contact_natural_frequency = 50;
    } else if (mediumShardPile) {
      params.numSolverIterations = 5;
      params.numAdditionalFrictionIterations = 1;
      params.numInternalPgsIterations = 2;
      params.contact_natural_frequency = 85;
    } else if (this.localWindowFrozenHandles.size > 0) {
      // 4-layer local window on loose (<0.75 N) pull: 4 solver x 2 PGS sub-Nyquist converges cleanly in < 1.0 ms
      params.numSolverIterations = 4;
      params.numAdditionalFrictionIterations = 1;
      params.numInternalPgsIterations = 2;
      params.contact_natural_frequency = 195;
    } else {
      params.numSolverIterations = 6;
      params.numAdditionalFrictionIterations = 1;
      params.numInternalPgsIterations = 2;
      params.contact_natural_frequency = 219;
    }

    this.updateStribeckAndMicroDampingFrame();

    let substeps = 0;
    while (this.accumulator >= stepDt && substeps < maxSubsteps) {
      this.applyHandForcesSubstep();
      if (
        substeps === 0 ||
        (!gentleActivePlay && this.maxSubstepSpeedBoundReal > 0.72) ||
        this.maxSubstepSpeedBoundReal > 1.65 ||
        this.extraBodyPreStep.size > 0
      ) {
        this.snapshotPreStepVelocities();
      }
      const canProduceDynamicCollision =
        (gentleActivePlay
          ? this.maxSubstepSpeedBoundReal > 1.65
          : this.maxSubstepSpeedBoundReal > 1.15) || this.extraBodyPreStep.size > 0;
      if (canProduceDynamicCollision) {
        this.world.step(this.eventQueue);
        this.processCollisionsAndFractures();
      } else {
        this.world.step();
      }
      if (this.localWindowFrozenHandles.size > 0) {
        this.validateLocalSolverWindowSubstep();
      }
      this.accumulator -= stepDt;
      substeps++;
    }

    this.updateMicroSplinters(clampedDt);
    this.syncTransformsAndRules();
    if (
      this.turnPhase === 'EXTRACTED' &&
      this.activeBlockId !== null &&
      this.shards.length === 0 &&
      this.leanAngleDeg < 1.0
    ) {
      let towerQuiet = true;
      let anyAwake = false;
      for (let i = 0; i < this.blocks.length; i++) {
        const ob = this.blocks[i];
        if (ob.shattered || ob.extracted || ob.id === this.activeBlockId) continue;
        const orb = this.world.getRigidBody(ob.bodyHandle);
        if (!orb || orb.isFixed() || orb.isSleeping()) continue;
        anyAwake = true;
        const olv = orb.linvel();
        const oav = orb.angvel();
        if (
          Math.hypot(olv.x, olv.y, olv.z) * SIM_TIME_SCALE >= 0.14 ||
          Math.hypot(oav.x, oav.y, oav.z) >= 0.28
        ) {
          towerQuiet = false;
          break;
        }
      }
      if (towerQuiet && (anyAwake || this.pendingQuietManifoldRestore)) {
        this.pendingQuietManifoldRestore = false;
        this.releaseLocalSolverWindow(false, false);
        this.restoreRestingContactManifolds(true);
      }
    } else if (this.pendingQuietManifoldRestore) {
      this.pendingQuietManifoldRestore = false;
      this.restoreRestingContactManifolds(false);
    }
    this.updateTurnStateMachine(clampedDt);
  }

  private validateLocalSolverWindowSubstep(): void {
    if (this.activeBlockId === null || this.localWindowFrozenHandles.size === 0) return;
    const activeBlk = this.blocks[this.activeBlockId];
    if (!activeBlk || activeBlk.shattered || activeBlk.extracted) {
      this.releaseLocalSolverWindow(true, false);
      return;
    }
    const L = activeBlk.assignedLayer;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted || b.id === activeBlk.id) continue;
      if (b.assignedLayer === L + 1 || b.assignedLayer === L + 2) {
        const rb = this.world.getRigidBody(b.bodyHandle);
        const ref = this.localWindowRefPos.get(b.id);
        if (!rb || !ref) continue;
        const t = rb.translation();
        const r = rb.rotation();
        const upY = 1.0 - 2.0 * (r.x * r.x + r.z * r.z);
        const tiltDeg = (Math.acos(Math.max(-1, Math.min(1, Math.abs(upY)))) * 180) / Math.PI;
        if (
          Math.abs(t.y - ref[1]) > 0.015 ||
          Math.hypot(t.x - ref[0], t.z - ref[2]) > 0.028 ||
          tiltDeg > 1.45
        ) {
          this.releaseLocalSolverWindow(true, false);
          this.localWindowBrokenForPull = true;
          return;
        }
      }
    }
  }

  private engageOrValidateLocalSolverWindow(): void {
    if (this.activeBlockId === null) return;
    const activeBlk = this.blocks[this.activeBlockId];
    if (!activeBlk || activeBlk.shattered || activeBlk.extracted) {
      this.releaseLocalSolverWindow(true, false);
      return;
    }
    if (this.localWindowFrozenHandles.size > 0) {
      this.validateLocalSolverWindowSubstep();
      return;
    }

    const L = activeBlk.assignedLayer;
    this.localWindowRefPos.clear();
    this.localWindowSavedClampingLoadN.clear();
    this.localWindowCapBodyHandles = [];
    this.localWindowDisabledCols = [];

    const layerCounts = new Map<number, number>();
    const preFreezeClampingN = new Map<number, number>();
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      layerCounts.set(b.assignedLayer, (layerCounts.get(b.assignedLayer) ?? 0) + 1);
      preFreezeClampingN.set(b.id, this.computeBlockClampingLoadN(b.id));
    }

    // Only engage local window when pulling a genuinely loose block (< 0.75 N clamping load).
    // When pulling a load-bearing block (>= 0.75 N), solve the full continuous 54-block dynamic stack
    // so upper rows deform smoothly with zero mid-pull unfreeze drop or vibration!
    if ((preFreezeClampingN.get(activeBlk.id) ?? 0) >= 0.75) {
      this.localWindowBrokenForPull = true;
      return;
    }

    const isBearingUpperLoad = (b: (typeof this.blocks)[number]): boolean =>
      !b.isLooseSlot ||
      (layerCounts.get(b.assignedLayer) ?? 0) <= 1 ||
      (preFreezeClampingN.get(b.id) ?? 0) >= 0.35;

    let hasL2Support = false;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (
        !b.shattered &&
        !b.extracted &&
        b.id !== activeBlk.id &&
        b.assignedLayer === L + 2 &&
        isBearingUpperLoad(b)
      ) {
        hasL2Support = true;
        break;
      }
    }
    const maxWinLayer = hasL2Support ? L + 2 : L + 1;
    let aboveCount = 0;

    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb) continue;

      if (b.assignedLayer < L - 1 || b.assignedLayer > maxWinLayer) {
        this.localWindowSavedClampingLoadN.set(b.id, preFreezeClampingN.get(b.id) ?? 0);
        rb.setBodyType(RAPIER.RigidBodyType.Fixed, false);
        this.localWindowFrozenHandles.add(b.bodyHandle);
        if (b.assignedLayer > maxWinLayer) {
          aboveCount++;
        }
        // Keep L - 2 enabled as the rigid support shelf directly beneath L - 1;
        // disable non-boundary frozen colliders so NarrowPhase skips Fixed-Fixed roundCuboid pairs
        // and leaves maxWinLayer's top surface unconstrained to tip/drop if support is pulled.
        if (b.assignedLayer < L - 2 || b.assignedLayer > maxWinLayer) {
          const col = this.world.getCollider(b.colliderHandle);
          if (col && col.isEnabled()) {
            col.setEnabled(false);
            this.localWindowDisabledCols.push(b.colliderHandle);
          }
        }
      } else {
        const t = rb.translation();
        this.localWindowRefPos.set(b.id, [t.x, t.y, t.z]);
        if ((layerCounts.get(L + 1) ?? 3) < 2 || (layerCounts.get(L + 2) ?? 3) < 2) {
          rb.wakeUp();
        }
        if (b.id !== activeBlk.id && b.assignedLayer === maxWinLayer && isBearingUpperLoad(b)) {
          this.localWindowCapBodyHandles.push(b.bodyHandle);
        }
      }
    }

    if (aboveCount > 0) {
      if (this.localWindowCapBodyHandles.length === 0) {
        // Never allow upper-tower weight to vanish if maxWinLayer has no eligible cap body
        this.releaseLocalSolverWindow(true, false);
        this.localWindowBrokenForPull = true;
        return;
      }
      const gScale = 1.0 + aboveCount / this.localWindowCapBodyHandles.length;
      const wakeCap = (layerCounts.get(L + 1) ?? 3) < 2 || (layerCounts.get(L + 2) ?? 3) < 2;
      for (let k = 0; k < this.localWindowCapBodyHandles.length; k++) {
        this.world.getRigidBody(this.localWindowCapBodyHandles[k])?.setGravityScale(gScale, wakeCap);
      }
    }
  }

  private releaseLocalSolverWindow(
    wakeRestored: boolean,
    rebuildRestingManifolds: boolean = false
  ): void {
    const hadFrozenBodies =
      this.localWindowFrozenHandles.size > 0 || this.localWindowDisabledCols.length > 0;
    if (!hadFrozenBodies && this.localWindowCapBodyHandles.length === 0) {
      return;
    }
    for (let k = 0; k < this.localWindowDisabledCols.length; k++) {
      this.world.getCollider(this.localWindowDisabledCols[k])?.setEnabled(true);
    }
    for (let k = 0; k < this.localWindowCapBodyHandles.length; k++) {
      const capRb = this.world.getRigidBody(this.localWindowCapBodyHandles[k]);
      if (capRb) {
        capRb.setGravityScale(1.0, false);
        const lv = capRb.linvel();
        if (lv.y < -0.6) {
          capRb.setLinvel({ x: lv.x, y: -0.6, z: lv.z }, true);
        }
      }
    }
    for (const handle of this.localWindowFrozenHandles) {
      const rb = this.world.getRigidBody(handle);
      if (rb && rb.isFixed()) {
        rb.setBodyType(RAPIER.RigidBodyType.Dynamic, wakeRestored || rebuildRestingManifolds);
        if (!wakeRestored && !rebuildRestingManifolds) {
          rb.sleep();
        }
      }
    }
    this.localWindowFrozenHandles.clear();
    this.localWindowDisabledCols = [];
    this.localWindowRefPos.clear();
    this.localWindowSavedClampingLoadN.clear();
    this.localWindowCapBodyHandles = [];

    if (hadFrozenBodies && rebuildRestingManifolds) {
      this.restoreRestingContactManifolds(!wakeRestored);
    }
  }

  /**
   * Runs 2 quiet 120 Hz solver warm-up substeps after re-enabling frozen window colliders so
   * Rapier's NarrowPhase & PGS solver repopulate true resting contact manifolds and clamping loads
   * across all layers (keeping play-by-ear tap-test resonance accurate after pulls/extractions).
   */
  private restoreRestingContactManifolds(sleepAllQuietAfter: boolean): void {
    if (this.shards.length > 0 || this.leanAngleDeg >= 1.5) return;

    const params = this.world.integrationParameters;
    const savedDt = params.dt;
    const savedSolver = params.numSolverIterations;
    const savedFriction = params.numAdditionalFrictionIterations;
    const savedPgs = params.numInternalPgsIterations;
    const savedFreq = params.contact_natural_frequency;

    params.dt = PHYSICS_DT;
    params.numSolverIterations = 6;
    params.numAdditionalFrictionIterations = 1;
    params.numInternalPgsIterations = 2;
    params.contact_natural_frequency = 219;
    this.currentSubstepDt = PHYSICS_DT;

    const quietHandles: number[] = [];
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb || rb.isFixed()) continue;
      const lv = rb.linvel();
      const av = rb.angvel();
      const isQuiet =
        sleepAllQuietAfter ||
        rb.isSleeping() ||
        (Math.hypot(lv.x, lv.y, lv.z) * SIM_TIME_SCALE < 0.08 &&
          Math.hypot(av.x, av.y, av.z) < 0.16);
      if (isQuiet) {
        quietHandles.push(b.bodyHandle);
      }
      rb.wakeUp();
    }

    this.world.step();
    this.world.step();

    for (let k = 0; k < quietHandles.length; k++) {
      const rb = this.world.getRigidBody(quietHandles[k]);
      if (!rb || rb.isFixed()) continue;
      rb.setLinvel({ x: 0, y: 0, z: 0 }, false);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, false);
      if (sleepAllQuietAfter) {
        rb.sleep();
      }
    }

    params.dt = savedDt;
    params.numSolverIterations = savedSolver;
    params.numAdditionalFrictionIterations = savedFriction;
    params.numInternalPgsIterations = savedPgs;
    params.contact_natural_frequency = savedFreq;
  }

  /**
   * Hysteresis-guarded Stribeck breakaway on ONLY the actively pulled block (never perturbing
   * warm-started tangential friction manifolds across resting/swaying stack neighbors!), paired
   * with tumble-preserving oak-table rolling & inelastic contact settling dissipation on fallen bricks.
   */
  private updateStribeckAndMicroDampingFrame(): void {
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered || b.extracted) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb || rb.isSleeping()) continue;

      const t = rb.translation();
      const r = rb.rotation();
      const lv = rb.linvel();
      const av = rb.angvel();
      const horizSlideReal = Math.hypot(lv.x, lv.z) * SIM_TIME_SCALE;
      const speed3DReal = Math.hypot(lv.x, lv.y, lv.z) * SIM_TIME_SCALE;
      const angSpeedReal = Math.hypot(av.x, av.y, av.z) * SIM_TIME_SCALE;

      if (b.id === this.activeBlockId) {
        // Smooth breakaway only on the pulled block with hysteresis (2.8 cm/s engage, 0.7 cm/s re-stick)
        let targetMu = b.currentFriction;
        if (horizSlideReal >= 0.28) {
          targetMu =
            GLASS_KINETIC_FRICTION +
            (GLASS_STATIC_FRICTION - GLASS_KINETIC_FRICTION) *
              Math.exp(-Math.pow((horizSlideReal - 0.28) / 0.35, 2));
        } else if (horizSlideReal <= 0.07) {
          targetMu = GLASS_STATIC_FRICTION;
        }
        if (
          Math.abs(targetMu - b.currentFriction) > 0.02 ||
          (targetMu === GLASS_STATIC_FRICTION && b.currentFriction !== GLASS_STATIC_FRICTION)
        ) {
          const col = this.world.getCollider(b.colliderHandle);
          if (col) {
            col.setFriction(targetMu);
            b.currentFriction = targetMu;
          }
        }
        if ((this.blockDampingTier.get(b.id) ?? 0) !== 0) {
          this.blockDampingTier.set(b.id, 0);
          b.isMicroDamped = false;
          rb.setLinearDamping(0.025);
          rb.setAngularDamping(0.035);
        }
        this.blockQuietFrames.set(b.id, 0);
        continue;
      }

      if (b.currentFriction !== GLASS_STATIC_FRICTION) {
        const col = this.world.getCollider(b.colliderHandle);
        if (col) {
          col.setFriction(GLASS_STATIC_FRICTION);
          b.currentFriction = GLASS_STATIC_FRICTION;
        }
      }

      const upY = 1.0 - 2.0 * (r.x * r.x + r.z * r.z);
      const tiltDeg = (Math.acos(Math.max(-1, Math.min(1, Math.abs(upY)))) * 180) / Math.PI;
      const rHoriz = Math.hypot(t.x, t.z);
      const nominalY = (b.assignedLayer + 0.5) * BLOCK_HEIGHT;
      const hasFallenOntoPile =
        (b.assignedLayer >= 2 && t.y < Math.min(1.15, nominalY - BLOCK_HEIGHT * 0.9)) ||
        (b.assignedLayer === 1 && (t.y < BLOCK_HEIGHT * 0.95 || tiltDeg > 35.0 || rHoriz > 0.72)) ||
        (b.assignedLayer === 0 && (tiltDeg > 25.0 || rHoriz > 0.72)) ||
        tiltDeg > 42.0 ||
        rHoriz > 0.85;
      const isInStandingTower = !hasFallenOntoPile && tiltDeg < 18.0 && rHoriz < 0.72;

      const isPostCollapseActive =
        this.turnPhase === 'COLLAPSED' || this.shatteredCount > 0 || this.leanAngleDeg > 38.0;

      let targetTier = 0; // 0 = crisp free-fall / active tip / active tumble (0.025 / 0.035)
      if (isPostCollapseActive && t.y > -0.05 && t.y < 2.55 && speed3DReal < 0.16 && angSpeedReal < 0.55) {
        targetTier = 2; // Quiet surviving post-collapse stump or perched leaning block
      } else if (isInStandingTower) {
        if (speed3DReal < 0.025 && angSpeedReal < 0.45) {
          targetTier = 1; // 1 = gentle in-tower micro-settling (0.14 / 0.20)
        }
      } else if (hasFallenOntoPile) {
        // Fallen block on oak table or rubble pile: preserve 100% free-fall & active tumble energy (>= 2.0 dm/s),
        // then apply hardwood-oak & pile inelastic rolling dissipation once supported!
        const inRubbleZone = t.y > -0.05 && t.y < 1.85;
        const vertSpeedReal = Math.abs(lv.y) * SIM_TIME_SCALE;
        if (inRubbleZone && speed3DReal < 0.42 && angSpeedReal < 1.45) {
          targetTier = 3; // 3 = final bevel-rocking arrest (1.65 / 2.40)
        } else if (inRubbleZone && speed3DReal < 2.0 && angSpeedReal < 4.5 && vertSpeedReal < 1.25) {
          targetTier = 2; // 2 = inelastic rolling/clatter decay (0.42 / 0.68)
        }
      }

      const prevTier = this.blockDampingTier.get(b.id) ?? 0;
      if (targetTier !== prevTier) {
        this.blockDampingTier.set(b.id, targetTier);
        b.isMicroDamped = targetTier > 0;
        if (targetTier === 0) {
          rb.setLinearDamping(0.025);
          rb.setAngularDamping(0.035);
        } else if (targetTier === 1) {
          rb.setLinearDamping(0.14);
          rb.setAngularDamping(0.20);
        } else if (targetTier === 2) {
          rb.setLinearDamping(0.42);
          rb.setAngularDamping(0.68);
        } else {
          rb.setLinearDamping(1.65);
          rb.setAngularDamping(2.40);
        }
      }

      // Put settled fallen/pile blocks (and quiet surviving stump / perched blocks after collapse) cleanly to sleep
      // so coupled block+shard contact islands never buzz indefinitely. Active tipping columns never sleep!
      if (
        (hasFallenOntoPile || isPostCollapseActive) &&
        t.y > -0.05 &&
        t.y < 2.55 &&
        speed3DReal < 0.16 &&
        angSpeedReal < 0.52
      ) {
        const qf = (this.blockQuietFrames.get(b.id) ?? 0) + 1;
        this.blockQuietFrames.set(b.id, qf);
        if (qf >= 14) {
          rb.setLinvel({ x: 0, y: 0, z: 0 }, false);
          rb.setAngvel({ x: 0, y: 0, z: 0 }, false);
          rb.sleep();
        }
      } else {
        this.blockQuietFrames.set(b.id, 0);
      }
    }
  }

  private countTouchingContacts(col: RAPIER.Collider): number {
    let touches = 0;
    this.world.contactPairsWith(col, (otherCol) => {
      this.world.contactPair(col, otherCol, (manifold) => {
        const nc = manifold.numContacts();
        for (let k = 0; k < nc; k++) {
          if (manifold.contactDist(k) <= 0.0025) {
            touches++;
          }
        }
      });
    });
    return touches;
  }

  public getClearContactSubsteps(): number {
    return this.clearContactSubsteps;
  }

  private applyHandForcesSubstep(): void {
    if (this.activeBlockId === null) {
      this.grabResistance01 = 0;
      return;
    }

    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) {
      this.activeBlockId = null;
      this.grabResistance01 = 0;
      return;
    }

    const rb = this.world.getRigidBody(block.bodyHandle);
    const col = this.world.getCollider(block.colliderHandle);
    if (!rb || !col) return;

    const pos = rb.translation();
    const rot = rb.rotation();
    const q: Quat = [rot.x, rot.y, rot.z, rot.w];
    const lv = rb.linvel();
    const av = rb.angvel();

    if (this.turnPhase === 'PULLING' || (this.gameMode === 'sandbox' && !block.extracted)) {
      const worldOffset = quatRotateVec3(q, this.grabLocalPoint);
      const grabWorld: Vec3 = [
        pos.x + worldOffset[0],
        pos.y + worldOffset[1],
        pos.z + worldOffset[2],
      ];
      this.grabCurrentWorldOnBlock = grabWorld;

      const errX = this.grabWorldAnchor[0] - grabWorld[0];
      const errZ = this.grabWorldAnchor[2] - grabWorld[2];

      const axisX = quatRotateVec3(q, [1, 0, 0]);
      const axisZ = quatRotateVec3(q, [0, 0, 1]);
      const radialDist = Math.hypot(pos.x, pos.z);
      const insideFactor = Math.max(0.0, Math.min(1.0, 1.0 - (radialDist - 0.35) / 0.45));

      const alongLen = errX * axisX[0] + errZ * axisX[2];
      const acrossWid = errX * axisZ[0] + errZ * axisZ[2];
      const lateralScale = this.gameMode === 'sandbox' ? 0.85 : 1.0 - 0.78 * insideFactor;

      const effErrX = axisX[0] * alongLen + axisZ[0] * acrossWid * lateralScale;
      const effErrZ = axisX[2] * alongLen + axisZ[2] * acrossWid * lateralScale;

      const vx = lv.x + (av.y * worldOffset[2] - av.z * worldOffset[1]);
      const vz = lv.z + (av.x * worldOffset[1] - av.y * worldOffset[0]);

      const subDt = this.currentSubstepDt;
      this.grabRampSec = Math.min(0.065, this.grabRampSec + subDt / SIM_TIME_SCALE);
      const rampU = this.grabRampSec / 0.065;
      const grabRamp = 0.35 + 0.65 * (rampU * rampU * (3.0 - 2.0 * rampU));

      const dtRatio = subDt / PHYSICS_DT;
      const implicitDampScale = 1.0 / (1.0 + 0.65 * Math.max(0.0, dtRatio - 1.0));
      const kp = (this.gameMode === 'sandbox' ? 28.0 : 18.0) * grabRamp;

      // Critical damping scaled to true rotational effective point mass at grabOffset (m_eff ~ 21g at tip vs 71g at center)
      const mBlk = Math.max(0.01, rb.mass());
      const iyBlk = (mBlk / 12.0) * (BLOCK_LENGTH * BLOCK_LENGTH + BLOCK_WIDTH * BLOCK_WIDTH);
      const errNorm = Math.hypot(effErrX, effErrZ);
      const dirX = errNorm > 1e-4 ? effErrX / errNorm : axisX[0];
      const dirZ = errNorm > 1e-4 ? effErrZ / errNorm : axisX[2];
      const rCrossPull = worldOffset[0] * dirZ - worldOffset[2] * dirX;
      const invMassGrab = 1.0 / mBlk + (rCrossPull * rCrossPull) / iyBlk;
      const mEffGrab = 1.0 / invMassGrab;
      const kd = 2.15 * Math.sqrt(kp * mEffGrab) * implicitDampScale;

      let fx = kp * effErrX - kd * vx;
      let fz = kp * effErrZ - kd * vz;
      const fy = -1.6 * implicitDampScale * lv.y;

      const maxForce = (this.gameMode === 'sandbox' ? 11.5 : 5.2) * grabRamp;
      const fMag = Math.hypot(fx, fz);
      if (fMag > maxForce) {
        fx = (fx / fMag) * maxForce;
        fz = (fz / fMag) * maxForce;
      }

      rb.applyImpulseAtPoint(
        { x: fx * subDt, y: fy * subDt, z: fz * subDt },
        { x: grabWorld[0], y: grabWorld[1], z: grabWorld[2] },
        true
      );

      // Gentle roll/pitch stabilization only; yaw is already critically damped by point force (fx, fz) at grabWorld
      const dampXZ = Math.min(0.00028, 0.0005 * dtRatio * implicitDampScale);
      const dampY = Math.min(0.00022, 0.00022 * dtRatio * implicitDampScale);
      rb.applyTorqueImpulse(
        {
          x: -av.x * dampXZ,
          y: -av.y * dampY,
          z: -av.z * dampXZ,
        },
        true
      );

      const postLv = rb.linvel();
      const postAv = rb.angvel();
      const postBoundReal =
        (Math.hypot(postLv.x, postLv.y, postLv.z) +
          Math.hypot(postAv.x, postAv.y, postAv.z) * 0.42) *
        SIM_TIME_SCALE;
      if (postBoundReal > this.maxSubstepSpeedBoundReal) {
        this.maxSubstepSpeedBoundReal = postBoundReal;
      }

      const speedHorizReal = Math.hypot(postLv.x, postLv.z) * SIM_TIME_SCALE;
      const rawRes = Math.min(1.0, fMag / (maxForce * 0.72)) * Math.max(0.08, 1.0 - speedHorizReal / 3.8);
      this.grabResistance01 = this.grabResistance01 * 0.86 + rawRes * 0.14;
      this.audio.updateSlideFriction(speedHorizReal, this.grabResistance01);

      // Require geometric clearance AND 3 consecutive zero-contact substeps (25 ms) before EXTRACTED
      if (this.gameMode !== 'sandbox') {
        const refCenter = this.getLayerSupportCenter(block.assignedLayer);
        const relX = pos.x - refCenter[0];
        const relZ = pos.z - refCenter[1];
        const alongClear = Math.abs(relX * axisX[0] + relZ * axisX[2]);
        const clearDist = Math.hypot(relX, relZ);
        const geomClear = alongClear > BLOCK_LENGTH * 0.98 || clearDist > BLOCK_LENGTH * 1.01;

        if (geomClear && this.countTouchingContacts(col) === 0) {
          this.clearContactSubsteps++;
        } else {
          this.clearContactSubsteps = 0;
        }

        if (this.clearContactSubsteps >= 3) {
          block.extracted = true;
          this.turnPhase = 'EXTRACTED';
          this.carriedFromDrag = this.isPointerDraggingBlock;
          this.grabResistance01 = 0;
          this.extractionArcTimer = 0;
          this.extractionStartPos = [pos.x, pos.y, pos.z];
          const outLen = Math.hypot(relX, relZ) || 1;
          this.extractionOutDir = [relX / outLen, 0, relZ / outLen];

          this.audio.updateSlideFriction(0, 0);
          this.audio.playGlassImpact(
            0.24,
            1.25,
            [pos.x, pos.y, pos.z],
            this.listenerCamPos,
            this.listenerCamRight
          );

          col.setSensor(true);
          rb.setGravityScale(0.0, true);
          const hadFrozen = this.localWindowFrozenHandles.size > 0;
          this.releaseLocalSolverWindow(true, false);
          if (hadFrozen) {
            this.pendingQuietManifoldRestore = true;
          }
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
      // Smooth 220 ms outward-then-upward clearance lift arc to poise above hovered top slot
      this.extractionArcTimer = Math.min(0.22, this.extractionArcTimer + this.currentSubstepDt / SIM_TIME_SCALE);
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

      const dx = guideX - pos.x;
      const dy = guideY - pos.y;
      const dz = guideZ - pos.z;

      rb.setLinvel(
        {
          x: dx * 11.5,
          y: dy * 11.5,
          z: dz * 11.5,
        },
        true
      );

      const tq = this.carriedTargetQuat;
      const dot = q[0] * tq[0] + q[1] * tq[1] + q[2] * tq[2] + q[3] * tq[3];
      const sign = dot < 0 ? -1 : 1;
      const blend = 0.22;
      let nx = q[0] * (1 - blend) + tq[0] * sign * blend;
      let ny = q[1] * (1 - blend) + tq[1] * sign * blend;
      let nz = q[2] * (1 - blend) + tq[2] * sign * blend;
      let nw = q[3] * (1 - blend) + tq[3] * sign * blend;
      const nLen = Math.hypot(nx, ny, nz, nw) || 1;
      nx /= nLen; ny /= nLen; nz /= nLen; nw /= nLen;
      rb.setRotation({ x: nx, y: ny, z: nz, w: nw }, true);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, true);
      this.grabCurrentWorldOnBlock = [pos.x, pos.y, pos.z];
      this.grabWorldAnchor = [...this.carriedTargetPos];
    }
  }

  private snapshotPreStepVelocities(): void {
    let maxBoundSim = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb) continue;
      if (rb.isSleeping()) {
        b.prePos[0] = b.position[0];
        b.prePos[1] = b.position[1];
        b.prePos[2] = b.position[2];
        b.preQuat[0] = b.rotation[0];
        b.preQuat[1] = b.rotation[1];
        b.preQuat[2] = b.rotation[2];
        b.preQuat[3] = b.rotation[3];
        b.linVel[0] = 0;
        b.linVel[1] = 0;
        b.linVel[2] = 0;
        b.angVel[0] = 0;
        b.angVel[1] = 0;
        b.angVel[2] = 0;
        continue;
      }
      const t = rb.translation();
      const r = rb.rotation();
      const lv = rb.linvel();
      const av = rb.angvel();
      b.prePos[0] = t.x;
      b.prePos[1] = t.y;
      b.prePos[2] = t.z;
      b.preQuat[0] = r.x;
      b.preQuat[1] = r.y;
      b.preQuat[2] = r.z;
      b.preQuat[3] = r.w;
      b.linVel[0] = lv.x;
      b.linVel[1] = lv.y;
      b.linVel[2] = lv.z;
      b.angVel[0] = av.x;
      b.angVel[1] = av.y;
      b.angVel[2] = av.z;
      if (!b.extracted) {
        const bound =
          Math.hypot(lv.x, lv.y, lv.z) + Math.hypot(av.x, av.y, av.z) * 0.42;
        if (bound > maxBoundSim) maxBoundSim = bound;
      }
    }
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const rb = this.world.getRigidBody(s.bodyHandle);
      if (!rb) continue;
      if (!rb.isSleeping()) {
        const t = rb.translation();
        const r = rb.rotation();
        const lv = rb.linvel();
        const av = rb.angvel();
        s.prePos[0] = t.x;
        s.prePos[1] = t.y;
        s.prePos[2] = t.z;
        s.preQuat[0] = r.x;
        s.preQuat[1] = r.y;
        s.preQuat[2] = r.z;
        s.preQuat[3] = r.w;
        s.preLinVel[0] = lv.x;
        s.preLinVel[1] = lv.y;
        s.preLinVel[2] = lv.z;
        s.preAngVel[0] = av.x;
        s.preAngVel[1] = av.y;
        s.preAngVel[2] = av.z;
        s.preVy = lv.y;
        const sBound =
          Math.hypot(lv.x, lv.y, lv.z) + Math.hypot(av.x, av.y, av.z) * 0.25;
        if (sBound > maxBoundSim) maxBoundSim = sBound;

        if (
          t.y > -0.04 &&
          t.y < 2.55 &&
          lv.x * lv.x + lv.y * lv.y + lv.z * lv.z < 0.016 &&
          av.x * av.x + av.y * av.y + av.z * av.z < 0.42
        ) {
          s.quietSubsteps++;
          const sleepSubsteps = this.shards.length > 64 ? 12 : 20;
          if (s.quietSubsteps >= sleepSubsteps) {
            rb.setLinvel({ x: 0, y: 0, z: 0 }, false);
            rb.setAngvel({ x: 0, y: 0, z: 0 }, false);
            rb.sleep();
          }
        } else {
          s.quietSubsteps = 0;
        }
      } else {
        s.prePos[0] = s.position[0];
        s.prePos[1] = s.position[1];
        s.prePos[2] = s.position[2];
        s.preQuat[0] = s.rotation[0];
        s.preQuat[1] = s.rotation[1];
        s.preQuat[2] = s.rotation[2];
        s.preQuat[3] = s.rotation[3];
        s.preLinVel[0] = 0;
        s.preLinVel[1] = 0;
        s.preLinVel[2] = 0;
        s.preAngVel[0] = 0;
        s.preAngVel[1] = 0;
        s.preAngVel[2] = 0;
        s.preVy = 0;
      }
    }
    this.maxSubstepSpeedBoundReal = maxBoundSim * SIM_TIME_SCALE;

    // Also snapshot any external dynamic/kinematic rigid bodies (e.g., sandbox/test strikers)
    this.extraBodyPreStep.clear();
    if (this.world.bodies.len() > this.blocks.length + this.shards.length + 1) {
      this.world.bodies.forEach((rb) => {
        if (!rb.isDynamic() && !rb.isKinematic()) return;
        const t = rb.translation();
        const lv = rb.linvel();
        const av = rb.angvel();
        this.extraBodyPreStep.set(rb.handle, {
          pos: [t.x, t.y, t.z],
          linVel: [lv.x, lv.y, lv.z],
          angVel: [av.x, av.y, av.z],
        });
      });
    }
  }

  private extraBodyPreStep = new Map<
    number,
    { pos: Vec3; linVel: Vec3; angVel: Vec3 }
  >();

  public lastFractureOriginLocal: Vec3 | null = null;

  private static quatMul(a: Quat, b: Quat): Quat {
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
  }

  private static solve3x3(m: number[], rhs: Vec3): Vec3 {
    const a00 = m[0], a01 = m[1], a02 = m[2];
    const a10 = m[3], a11 = m[4], a12 = m[5];
    const a20 = m[6], a21 = m[7], a22 = m[8];
    const det =
      a00 * (a11 * a22 - a12 * a21) -
      a01 * (a10 * a22 - a12 * a20) +
      a02 * (a10 * a21 - a11 * a20);
    if (Math.abs(det) < 1e-12) return [0, 0, 0];
    const inv = 1.0 / det;
    const b0 = rhs[0], b1 = rhs[1], b2 = rhs[2];
    return [
      (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) * inv,
      (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) * inv,
      (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) * inv,
    ];
  }

  private resolveBlockContactGeometry(
    blkA: GlassBlockState,
    colA: RAPIER.Collider,
    otherCol: RAPIER.Collider,
    forceDirWorld: Vec3
  ): { localHit: Vec3; nAtoB: Vec3 } {
    let foundManifold = false;
    let bestImpulse = -1;
    let bestDist = Infinity;
    let localHit: Vec3 = [0, 0, 0];
    let nAtoB: Vec3 = [0, -1, 0];

    this.world.contactPair(colA, otherCol, (manifold, flipped) => {
      const nc = manifold.numContacts();
      if (nc <= 0) return;
      const mn = manifold.normal();
      const candN: Vec3 = flipped ? [-mn.x, -mn.y, -mn.z] : [mn.x, mn.y, mn.z];
      for (let k = 0; k < nc; k++) {
        const imp = manifold.contactImpulse(k);
        const dist = manifold.contactDist(k);
        const isBetter =
          imp > bestImpulse + 1e-6 ||
          (Math.abs(imp - bestImpulse) <= 1e-6 && dist < bestDist);
        if (isBetter) {
          const p = flipped ? manifold.localContactPoint2(k) : manifold.localContactPoint1(k);
          if (p) {
            foundManifold = true;
            bestImpulse = imp;
            bestDist = dist;
            localHit = [p.x, p.y, p.z];
            nAtoB = candN;
          }
        }
      }
    });

    if (foundManifold) {
      const worldOff = quatRotateVec3(blkA.preQuat, localHit);
      const worldPtX = blkA.prePos[0] + worldOff[0];
      const worldPtY = blkA.prePos[1] + worldOff[1];
      const worldPtZ = blkA.prePos[2] + worldOff[2];
      const oc = otherCol.translation();
      const toOtherX = oc.x - worldPtX;
      const toOtherY = oc.y - worldPtY;
      const toOtherZ = oc.z - worldPtZ;
      const dotSign =
        nAtoB[0] * toOtherX + nAtoB[1] * toOtherY + nAtoB[2] * toOtherZ;
      if (dotSign < 0) {
        nAtoB = [-nAtoB[0], -nAtoB[1], -nAtoB[2]];
      }
      return { localHit, nAtoB };
    }

    // Intra-step rebound fallback (`SmallStepsPgsSolver` rebounded > predictionDistance before substep 12):
    // Compute exact closest point on blkA's local prism to the impacting body's pre-step geometry!
    if (this.tableColliderHandles.has(otherCol.handle)) {
      let minWorldY = Infinity;
      const cornerHits: Array<{ wy: number; lp: Vec3 }> = [];
      for (let sx = -1; sx <= 1; sx += 2) {
        for (let sy = -1; sy <= 1; sy += 2) {
          for (let sz = -1; sz <= 1; sz += 2) {
            const lp: Vec3 = [sx * BLOCK_HALF_X, sy * BLOCK_HALF_Y, sz * BLOCK_HALF_Z];
            const wp = quatRotateVec3(blkA.preQuat, lp);
            const wy = blkA.prePos[1] + wp[1];
            if (wy < minWorldY) minWorldY = wy;
            cornerHits.push({ wy, lp });
          }
        }
      }
      let ax = 0, ay = 0, az = 0, cnt = 0;
      for (const c of cornerHits) {
        if (c.wy <= minWorldY + 0.004) {
          ax += c.lp[0];
          ay += c.lp[1];
          az += c.lp[2];
          cnt++;
        }
      }
      return {
        localHit: cnt > 0 ? [ax / cnt, ay / cnt, az / cnt] : [0, -BLOCK_HALF_Y, 0],
        nAtoB: [0, -1, 0],
      };
    }

    const otherBlk = this.colliderToBlock.get(otherCol.handle);
    if (otherBlk) {
      let minD2 = Infinity;
      const samples: Array<{ d2: number; clamped: Vec3 }> = [];
      for (let ix = -1; ix <= 1; ix++) {
        for (let iy = -1; iy <= 1; iy++) {
          for (let iz = -1; iz <= 1; iz++) {
            if (ix === 0 && iy === 0 && iz === 0) continue;
            const bLocal: Vec3 = [ix * BLOCK_HALF_X, iy * BLOCK_HALF_Y, iz * BLOCK_HALF_Z];
            const bOff = quatRotateVec3(otherBlk.preQuat, bLocal);
            const relWorld: Vec3 = [
              otherBlk.prePos[0] + bOff[0] - blkA.prePos[0],
              otherBlk.prePos[1] + bOff[1] - blkA.prePos[1],
              otherBlk.prePos[2] + bOff[2] - blkA.prePos[2],
            ];
            const inA = quatInverseRotateVec3(blkA.preQuat, relWorld);
            const cx = Math.max(-BLOCK_HALF_X, Math.min(BLOCK_HALF_X, inA[0]));
            const cy = Math.max(-BLOCK_HALF_Y, Math.min(BLOCK_HALF_Y, inA[1]));
            const cz = Math.max(-BLOCK_HALF_Z, Math.min(BLOCK_HALF_Z, inA[2]));
            const d2 = (inA[0] - cx) ** 2 + (inA[1] - cy) ** 2 + (inA[2] - cz) ** 2;
            if (d2 < minD2) minD2 = d2;
            samples.push({ d2, clamped: [cx, cy, cz] });
          }
        }
      }
      let ax = 0, ay = 0, az = 0, cnt = 0;
      for (const s of samples) {
        if (s.d2 <= minD2 + 1e-5) {
          ax += s.clamped[0];
          ay += s.clamped[1];
          az += s.clamped[2];
          cnt++;
        }
      }
      const hitLocal: Vec3 = cnt > 0 ? [ax / cnt, ay / cnt, az / cnt] : [0, 0, 0];
      const deltaAB: Vec3 = [
        otherBlk.prePos[0] - blkA.prePos[0],
        otherBlk.prePos[1] - blkA.prePos[1],
        otherBlk.prePos[2] - blkA.prePos[2],
      ];
      const fLen = Math.hypot(forceDirWorld[0], forceDirWorld[1], forceDirWorld[2]);
      let dir: Vec3;
      if (fLen > 1e-4) {
        const sgn =
          forceDirWorld[0] * deltaAB[0] +
            forceDirWorld[1] * deltaAB[1] +
            forceDirWorld[2] * deltaAB[2] >=
          0
            ? 1
            : -1;
        dir = [(forceDirWorld[0] / fLen) * sgn, (forceDirWorld[1] / fLen) * sgn, (forceDirWorld[2] / fLen) * sgn];
      } else {
        const dLen = Math.hypot(deltaAB[0], deltaAB[1], deltaAB[2]) || 1;
        dir = [deltaAB[0] / dLen, deltaAB[1] / dLen, deltaAB[2] / dLen];
      }
      return { localHit: hitLocal, nAtoB: dir };
    }

    // Shard or external striker collider
    let otherPrePos: Vec3 = [0, 0, 0];
    const otherShard = this.colliderToShard.get(otherCol.handle);
    if (otherShard) {
      otherPrePos = otherShard.prePos;
    } else {
      const rawParent = otherCol.parent() as unknown as number | RAPIER.RigidBody | null;
      const parentHandle =
        typeof rawParent === 'number' ? rawParent : rawParent?.handle ?? null;
      const extra = parentHandle !== null ? this.extraBodyPreStep.get(parentHandle) : undefined;
      if (extra) {
        otherPrePos = extra.pos;
      } else {
        const ct = otherCol.translation();
        otherPrePos = [ct.x, ct.y, ct.z];
      }
    }
    const relWorld: Vec3 = [
      otherPrePos[0] - blkA.prePos[0],
      otherPrePos[1] - blkA.prePos[1],
      otherPrePos[2] - blkA.prePos[2],
    ];
    const inA = quatInverseRotateVec3(blkA.preQuat, relWorld);
    const clamped: Vec3 = [
      Math.max(-BLOCK_HALF_X, Math.min(BLOCK_HALF_X, inA[0])),
      Math.max(-BLOCK_HALF_Y, Math.min(BLOCK_HALF_Y, inA[1])),
      Math.max(-BLOCK_HALF_Z, Math.min(BLOCK_HALF_Z, inA[2])),
    ];
    const clampedWorldOff = quatRotateVec3(blkA.preQuat, clamped);
    const surfDelta: Vec3 = [
      otherPrePos[0] - (blkA.prePos[0] + clampedWorldOff[0]),
      otherPrePos[1] - (blkA.prePos[1] + clampedWorldOff[1]),
      otherPrePos[2] - (blkA.prePos[2] + clampedWorldOff[2]),
    ];
    const refVec: Vec3 =
      Math.hypot(surfDelta[0], surfDelta[1], surfDelta[2]) > 1e-4 ? surfDelta : relWorld;
    const fLen = Math.hypot(forceDirWorld[0], forceDirWorld[1], forceDirWorld[2]);
    let nOut: Vec3;
    if (fLen > 1e-4) {
      const sgn =
        forceDirWorld[0] * refVec[0] +
          forceDirWorld[1] * refVec[1] +
          forceDirWorld[2] * refVec[2] >=
        0
          ? 1
          : -1;
      nOut = [
        (forceDirWorld[0] / fLen) * sgn,
        (forceDirWorld[1] / fLen) * sgn,
        (forceDirWorld[2] / fLen) * sgn,
      ];
    } else {
      const rLen = Math.hypot(refVec[0], refVec[1], refVec[2]) || 1;
      nOut = [refVec[0] / rLen, refVec[1] / rLen, refVec[2] / rLen];
    }
    return {
      localHit: clamped,
      nAtoB: nOut,
    };
  }

  private getPreStepPointVel(
    colHandle: number,
    otherCol: RAPIER.Collider,
    worldPt: Vec3
  ): Vec3 {
    if (this.tableColliderHandles.has(colHandle)) {
      return [0, 0, 0];
    }
    const blk = this.colliderToBlock.get(colHandle);
    if (blk) {
      const rx = worldPt[0] - blk.prePos[0];
      const ry = worldPt[1] - blk.prePos[1];
      const rz = worldPt[2] - blk.prePos[2];
      return [
        blk.linVel[0] + (blk.angVel[1] * rz - blk.angVel[2] * ry),
        blk.linVel[1] + (blk.angVel[2] * rx - blk.angVel[0] * rz),
        blk.linVel[2] + (blk.angVel[0] * ry - blk.angVel[1] * rx),
      ];
    }
    const shd = this.colliderToShard.get(colHandle);
    if (shd) {
      const rx = worldPt[0] - shd.prePos[0];
      const ry = worldPt[1] - shd.prePos[1];
      const rz = worldPt[2] - shd.prePos[2];
      return [
        shd.preLinVel[0] + (shd.preAngVel[1] * rz - shd.preAngVel[2] * ry),
        shd.preLinVel[1] + (shd.preAngVel[2] * rx - shd.preAngVel[0] * rz),
        shd.preLinVel[2] + (shd.preAngVel[0] * ry - shd.preAngVel[1] * rx),
      ];
    }
    const rawParent = otherCol.parent() as unknown as number | RAPIER.RigidBody | null;
    const parentHandle =
      typeof rawParent === 'number' ? rawParent : rawParent?.handle ?? null;
    if (parentHandle !== null) {
      const extra = this.extraBodyPreStep.get(parentHandle);
      if (extra) {
        const rx = worldPt[0] - extra.pos[0];
        const ry = worldPt[1] - extra.pos[1];
        const rz = worldPt[2] - extra.pos[2];
        return [
          extra.linVel[0] + (extra.angVel[1] * rz - extra.angVel[2] * ry),
          extra.linVel[1] + (extra.angVel[2] * rx - extra.angVel[0] * rz),
          extra.linVel[2] + (extra.angVel[0] * ry - extra.angVel[1] * rx),
        ];
      }
    }
    return [0, 0, 0];
  }

  private spawnGlassCulletChips(
    worldOrigin: Vec3,
    baseLinVelSim: { x: number; y: number; z: number },
    impactNormal: Vec3,
    swatchId: number,
    count: number,
    speedScale: number
  ): void {
    for (let i = 0; i < count; i++) {
      if (this.splinters.length >= MAX_SPLINTERS) {
        this.splinters.shift();
      }
      const theta = Math.random() * Math.PI * 2;
      const latSpd = (0.8 + Math.random() * 4.6) * speedScale;
      const vx =
        baseLinVelSim.x * SIM_TIME_SCALE * 0.35 +
        Math.cos(theta) * latSpd +
        impactNormal[0] * latSpd * 0.25;
      const vy =
        Math.max(-1.5, baseLinVelSim.y * SIM_TIME_SCALE * 0.15) +
        (0.45 + Math.random() * 2.1) * speedScale +
        Math.max(0, impactNormal[1]) * 0.8;
      const vz =
        baseLinVelSim.z * SIM_TIME_SCALE * 0.35 +
        Math.sin(theta) * latSpd +
        impactNormal[2] * latSpd * 0.25;

      this.splinters.push({
        pos: [
          worldOrigin[0] + (Math.random() - 0.5) * 0.1,
          Math.max(0.012, worldOrigin[1] + (Math.random() - 0.5) * 0.05),
          worldOrigin[2] + (Math.random() - 0.5) * 0.1,
        ],
        vel: [vx, vy, vz],
        rot: Math.random() * Math.PI * 2,
        rotSpeed: (Math.random() - 0.5) * 28,
        scale: 0.008 + Math.random() * 0.013,
        swatchId,
        life: 0,
        maxLife: 6.5 + Math.random() * 4.5,
      });
    }
  }

  private processCollisionsAndFractures(): void {
    interface BlockImpactAccum {
      blk: GlassBlockState;
      dynamicImpulseJn: number;
      peakDynamicImpulseJn: number;
      maxVClose: number;
      bestLocalHit: Vec3;
      bestNormalAtoB: Vec3;
      bestHitTable: boolean;
      contacts: Array<{ jn: number; nAtoB: Vec3; vClose: number }>;
    }

    const blockImpacts = new Map<number, BlockImpactAccum>();
    const shardAudioEvents: Array<{
      energy: number;
      intensity01: number;
      volumeDm3: number;
      worldPos: [number, number, number];
      hitGlass: boolean;
    }> = [];

    const accumulateForBlock = (
      blkA: GlassBlockState,
      colA: RAPIER.Collider,
      otherHandle: number,
      otherCol: RAPIER.Collider,
      jn: number,
      forceDirWorld: Vec3
    ) => {
      if (blkA.shattered || blkA.extracted) return;
      const { localHit, nAtoB } = this.resolveBlockContactGeometry(
        blkA,
        colA,
        otherCol,
        forceDirWorld
      );
      const rWorld = quatRotateVec3(blkA.preQuat, localHit);
      const worldPt: Vec3 = [
        blkA.prePos[0] + rWorld[0],
        blkA.prePos[1] + rWorld[1],
        blkA.prePos[2] + rWorld[2],
      ];
      const vPtA = this.getPreStepPointVel(colA.handle, colA, worldPt);
      const vPtB = this.getPreStepPointVel(otherHandle, otherCol, worldPt);
      const relAlongN =
        (vPtB[0] - vPtA[0]) * nAtoB[0] +
        (vPtB[1] - vPtA[1]) * nAtoB[1] +
        (vPtB[2] - vPtA[2]) * nAtoB[2];
      const vClose = Math.max(0.0, -relAlongN) * SIM_TIME_SCALE;
      const hitTable = this.tableColliderHandles.has(otherHandle);

      let acc = blockImpacts.get(blkA.id);
      if (!acc) {
        acc = {
          blk: blkA,
          dynamicImpulseJn: 0,
          peakDynamicImpulseJn: -1,
          maxVClose: 0,
          bestLocalHit: localHit,
          bestNormalAtoB: nAtoB,
          bestHitTable: hitTable,
          contacts: [],
        };
        blockImpacts.set(blkA.id, acc);
      }

      acc.contacts.push({ jn, nAtoB, vClose });
      if (vClose > 1.5) {
        acc.dynamicImpulseJn += jn;
        if (vClose > acc.maxVClose) {
          acc.maxVClose = vClose;
        }
        if (jn > acc.peakDynamicImpulseJn) {
          acc.peakDynamicImpulseJn = jn;
          acc.bestLocalHit = localHit;
          acc.bestNormalAtoB = nAtoB;
          acc.bestHitTable = hitTable;
        }
      }
    };

    const upperSpeedBoundSim = (h: number): number => {
      if (this.tableColliderHandles.has(h)) return 0;
      const b = this.colliderToBlock.get(h);
      if (b) {
        return (
          Math.hypot(b.linVel[0], b.linVel[1], b.linVel[2]) +
          Math.hypot(b.angVel[0], b.angVel[1], b.angVel[2]) * 0.42
        );
      }
      const s = this.colliderToShard.get(h);
      if (s) {
        return (
          Math.hypot(s.preLinVel[0], s.preLinVel[1], s.preLinVel[2]) +
          Math.hypot(s.preAngVel[0], s.preAngVel[1], s.preAngVel[2]) * 0.25
        );
      }
      return 100.0; // Unknown external striker -> never skip!
    };

    this.eventQueue.drainContactForceEvents((ev) => {
      const h1 = ev.collider1();
      const h2 = ev.collider2();
      const totalForceMag = ev.totalForceMagnitude();
      if (!(totalForceMag > 0)) return;

      // Fast pure-JS upper-bound closing speed gate (avoids WASM NarrowPhase calls on resting contacts!)
      const maxRelReal = (upperSpeedBoundSim(h1) + upperSpeedBoundSim(h2)) * SIM_TIME_SCALE;
      if (maxRelReal <= 1.45) return;

      const jn = totalForceMag * this.currentSubstepDt;
      const md = ev.maxForceDirection();
      const forceDir: Vec3 = [md.x, md.y, md.z];

      const col1 = this.world.getCollider(h1);
      const col2 = this.world.getCollider(h2);
      if (!col1 || !col2) return;

      const blk1 = this.colliderToBlock.get(h1);
      const blk2 = this.colliderToBlock.get(h2);
      if (blk1) accumulateForBlock(blk1, col1, h2, col2, jn, forceDir);
      if (blk2) accumulateForBlock(blk2, col2, h1, col1, jn, forceDir);

      // Mid-air & pile shard-on-shard, shard-on-block, and shard-on-table modal collisions
      if (maxRelReal > 2.15) {
        const shd1 = this.colliderToShard.get(h1);
        const shd2 = this.colliderToShard.get(h2);
        if (shd1 || shd2) {
          const primaryShard = shd1 ?? shd2!;
          const worldPt: Vec3 = [...primaryShard.prePos];
          const v1 = this.getPreStepPointVel(h1, col1, worldPt);
          const v2 = this.getPreStepPointVel(h2, col2, worldPt);
          const dvReal =
            Math.hypot(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]) * SIM_TIME_SCALE;
          if (dvReal > 2.2 && jn > 0.001) {
            const hitTable =
              this.tableColliderHandles.has(h1) || this.tableColliderHandles.has(h2);
            const effMass =
              shd1 && shd2
                ? Math.min(shd1.massKg, shd2.massKg)
                : primaryShard.massKg;
            const energy = effMass * dvReal * dvReal;
            const intensity01 = Math.min(1.0, (dvReal - 1.8) / 14.0);
            shardAudioEvents.push({
              energy,
              intensity01,
              volumeDm3: primaryShard.volume,
              worldPos: [
                primaryShard.prePos[0],
                primaryShard.prePos[1],
                primaryShard.prePos[2],
              ],
              hitGlass: !hitTable,
            });
          }
        }
      }
    });

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
      if (acc.maxVClose <= 1.5 || acc.dynamicImpulseJn <= 0) continue;
      const blk = acc.blk;
      const rb = this.world.getRigidBody(blk.bodyHandle);
      if (!rb) continue;

      const massKg = Math.max(0.01, rb.mass());
      const lh = acc.bestLocalHit;
      const nLocal = quatInverseRotateVec3(blk.preQuat, acc.bestNormalAtoB);
      const rcx = lh[1] * nLocal[2] - lh[2] * nLocal[1];
      const rcy = lh[2] * nLocal[0] - lh[0] * nLocal[2];
      const rcz = lh[0] * nLocal[1] - lh[1] * nLocal[0];
      const ix = (massKg / 12.0) * (BLOCK_HEIGHT * BLOCK_HEIGHT + BLOCK_WIDTH * BLOCK_WIDTH);
      const iy = (massKg / 12.0) * (BLOCK_LENGTH * BLOCK_LENGTH + BLOCK_WIDTH * BLOCK_WIDTH);
      const iz = (massKg / 12.0) * (BLOCK_LENGTH * BLOCK_LENGTH + BLOCK_HEIGHT * BLOCK_HEIGHT);
      const invMassEff = 1.0 / massKg + (rcx * rcx) / ix + (rcy * rcy) / iy + (rcz * rcz) / iz;

      // Dynamic opposing-face anvil-crush detection (e.g., stationary bottom block pinched against oak table by falling row)
      let opposingJn = 0;
      const dynN = acc.bestNormalAtoB;
      for (const c of acc.contacts) {
        const dotOpp =
          c.nAtoB[0] * dynN[0] + c.nAtoB[1] * dynN[1] + c.nAtoB[2] * dynN[2];
        if (dotOpp < -0.5) {
          opposingJn += c.jn;
        }
      }
      const restingTablePinch =
        blk.prePos[1] < BLOCK_HEIGHT * 0.85 && dynN[1] < -0.6 && acc.dynamicImpulseJn > 0.12;
      const isAnvilCrush =
        acc.maxVClose > CRAZE_IMPACT_SPEED &&
        acc.dynamicImpulseJn > 0.12 &&
        (opposingJn > 0.3 * acc.dynamicImpulseJn || restingTablePinch);
      const anvilBoost = isAnvilCrush ? 1.28 : 1.0;

      // Strain-energy-equivalent impulse speed: geometric mean of CoM velocity jump and contact-point velocity jump.
      // Prevents 21g brick-tip rotational leverage (3.35x invMassEff) from detonating low-energy glancing tumbles.
      const jRef = isAnvilCrush ? acc.dynamicImpulseJn : acc.peakDynamicImpulseJn;
      const vComJ = (jRef / massKg) * SIM_TIME_SCALE;
      const vPtJ = jRef * invMassEff * SIM_TIME_SCALE;
      const vImpulse = Math.sqrt(Math.max(0, vComJ * vPtJ));

      // Surface hardness factor: glass-on-glass = 1.0 vs glass-on-oak-wood = 0.88
      const kappaSurface = acc.bestHitTable ? 0.88 : 1.0;

      // Stress-concentration geometry factor at localHit: corner = 1.32, edge = 1.14, flat face = 0.94
      let extCount = 0;
      if (Math.abs(lh[0]) > BLOCK_HALF_X * 0.78) extCount++;
      if (Math.abs(lh[1]) > BLOCK_HALF_Y * 0.72) extCount++;
      if (Math.abs(lh[2]) > BLOCK_HALF_Z * 0.76) extCount++;
      const kappaGeom = extCount >= 3 ? 1.32 : extCount === 2 ? 1.14 : 0.94;

      const vImpact =
        Math.min(acc.maxVClose, 1.30 * vImpulse) *
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

      // Griffith brittle crack propagation: pre-crazed glass has reduced fracture toughness (0.38) on distinct secondary impacts,
      // coalescing single-bounce corner slaps (55 ms) and preserving spiderweb-crazed blocks on low-energy (< 10.35 dm/s) rollovers.
      const lastCrazeSec = this.blockLastCrazeTime.get(blk.id) ?? -10.0;
      const inSameSlapWindow = this.simTime - lastCrazeSec < 0.055;
      const effectivePriorDamage = inSameSlapWindow ? 0.0 : blk.damage;
      const shatterThreshold = Math.max(
        10.35,
        SHATTER_IMPACT_SPEED * (1.0 - Math.min(0.85, effectivePriorDamage) * 0.38)
      );

      if (vImpact >= shatterThreshold) {
        blocksToShatter.push({
          blk,
          localHit: lh,
          impactSpeed: vImpact,
          worldNormal,
          hitTable: acc.bestHitTable,
        });
      } else if (vImpact >= CRAZE_IMPACT_SPEED) {
        const prevDamage = blk.damage;
        const addedDamage =
          0.33 +
          ((vImpact - CRAZE_IMPACT_SPEED) / (SHATTER_IMPACT_SPEED - CRAZE_IMPACT_SPEED)) * 0.48;
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
          const post = rb.linvel();
          this.spawnGlassCulletChips(hitWorld, post, worldNormal, blk.swatchId, 5, 0.55);
          this.audio.playGlassCrack(Math.min(1.0, blk.damage));
        }
      }
    }

    for (const item of blocksToShatter) {
      this.shatterBlock(item.blk, item.localHit, item.impactSpeed, item.worldNormal, item.hitTable);
    }
  }

  /**
   * Executes exact 3D Voronoi brittle fracture on `block`, replacing it with dynamic convex glass shards
   * in the 512-slot fixed-stride pool while strictly conserving mass, linear momentum, and angular momentum.
   */
  public shatterBlock(
    block: GlassBlockState,
    localImpactPoint: Vec3,
    impactSpeed: number = 16.0,
    impactNormal: Vec3 = [0, 1, 0],
    hitTable: boolean = true
  ): void {
    if (block.shattered) return;
    const rb = this.world.getRigidBody(block.bodyHandle);
    if (!rb) return;

    const pos = rb.translation();
    const rot = rb.rotation();
    const q: Quat = [rot.x, rot.y, rot.z, rot.w];
    const lv = rb.linvel();
    const av = rb.angvel();
    const parentMass = Math.max(0.01, rb.mass());

    // Capture exact pre-fracture world angular momentum L0 about parent CoM
    const pInertia = rb.principalInertia();
    const pFrame = rb.principalInertiaLocalFrame();
    const qPrin = PhysicsWorld.quatMul(q, [pFrame.x, pFrame.y, pFrame.z, pFrame.w]);
    const omegaPrin = quatInverseRotateVec3(qPrin, [av.x, av.y, av.z]);
    const parentL0 = quatRotateVec3(qPrin, [
      pInertia.x * omegaPrin[0],
      pInertia.y * omegaPrin[1],
      pInertia.z * omegaPrin[2],
    ]);

    block.shattered = true;
    this.shatteredCount++;
    if (this.activeBlockId === block.id) {
      this.activeBlockId = null;
      this.isPointerDraggingBlock = false;
    }

    // Seed Voronoi fracture origin 70% toward pre-existing crackLocalOrigin when damage > 0.25
    const effectiveOrigin: Vec3 =
      block.damage > 0.25
        ? [
            localImpactPoint[0] * 0.3 + block.crackLocalOrigin[0] * 0.7,
            localImpactPoint[1] * 0.3 + block.crackLocalOrigin[1] * 0.7,
            localImpactPoint[2] * 0.3 + block.crackLocalOrigin[2] * 0.7,
          ]
        : [localImpactPoint[0], localImpactPoint[1], localImpactPoint[2]];
    this.lastFractureOriginLocal = [...effectiveOrigin];

    this.colliderToBlock.delete(block.colliderHandle);
    this.world.removeRigidBody(rb);

    const impactEnergy01 = Math.max(0.0, Math.min(1.0, (impactSpeed - 12.5) / 15.0));
    const energyTargetShards = Math.round(11 + impactEnergy01 * 6);
    const headroom = MAX_ACTIVE_SHARDS - this.shards.length;
    const budgetCap = headroom > 160 ? 17 : headroom > 80 ? 13 : 9;
    const desiredShards = Math.min(energyTargetShards, budgetCap);
    const allowSecondaryCleavage = impactSpeed > 18.5;

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

    // Ensure enough free slots exist in the 512-slot fixed-stride pool (evict sleeping first, then oldest FIFO)
    while (this.freeShardSlots.length < shardGeoms.length && this.shards.length > 0) {
      let removeIdx = 0;
      for (let k = 0; k < this.shards.length; k++) {
        const candRb = this.world.getRigidBody(this.shards[k].bodyHandle);
        if (!candRb || candRb.isSleeping()) {
          removeIdx = k;
          break;
        }
      }
      const retired = this.shards.splice(removeIdx, 1)[0];
      this.colliderToShard.delete(retired.colliderHandle);
      this.freeShardSlots.push(retired.slotIndex);
      const oldRb = this.world.getRigidBody(retired.bodyHandle);
      if (oldRb) this.world.removeRigidBody(oldRb);
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
      body: RAPIER.RigidBody;
      col: RAPIER.Collider;
      worldPos: Vec3;
      rawVel: Vec3;
      rawSpin: Vec3;
      mass: number;
    }> = [];

    for (const geom of shardGeoms) {
      const slotIndex = this.freeShardSlots.pop();
      if (slotIndex === undefined) break;

      const worldOffset = quatRotateVec3(q, geom.localCentroid);
      const sx = pos.x + worldOffset[0];
      const sy = pos.y + worldOffset[1];
      const sz = pos.z + worldOffset[2];

      const shardRbDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(sx, sy, sz)
        .setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] })
        .setLinearDamping(0.14)
        .setAngularDamping(0.32)
        .setCcdEnabled(false);

      const shardBody = this.world.createRigidBody(shardRbDesc);

      const collisionHull = new Float32Array(geom.hullPoints.length);
      let maxHx = 0.012, maxHy = 0.012, maxHz = 0.012;
      for (let p = 0; p < geom.hullPoints.length; p += 3) {
        const hxVal = geom.hullPoints[p] * 0.982;
        const hyVal = geom.hullPoints[p + 1] * 0.982;
        const hzVal = geom.hullPoints[p + 2] * 0.982;
        collisionHull[p] = hxVal;
        collisionHull[p + 1] = hyVal;
        collisionHull[p + 2] = hzVal;
        maxHx = Math.max(maxHx, Math.abs(hxVal));
        maxHy = Math.max(maxHy, Math.abs(hyVal));
        maxHz = Math.max(maxHz, Math.abs(hzVal));
      }

      const shardExactMass = Math.max(0.0002, parentMass * geom.volume * invTotalGeomVol);
      const isHeroShard = geom.volume >= 0.0048 && this.shards.length < 200;
      const rawHullDesc = isHeroShard ? RAPIER.ColliderDesc.convexHull(collisionHull) : null;
      const shardColDesc = (
        rawHullDesc ?? RAPIER.ColliderDesc.cuboid(maxHx * 0.88, maxHy * 0.88, maxHz * 0.88)
      )
        .setMass(shardExactMass)
        .setFriction(SHARD_FRICTION)
        .setRestitution(0.12)
        .setCollisionGroups(isHeroShard ? GROUP_SHARD_ON_TABLE : GROUP_SHARD_CULLET_ON_TABLE);
      if (isHeroShard) {
        shardColDesc
          .setActiveEvents(RAPIER.ActiveEvents.CONTACT_FORCE_EVENTS)
          .setContactForceEventThreshold(0.55);
      }

      const shardCol = this.world.createCollider(shardColDesc, shardBody);

      const hash01 = (salt: number): number => {
        const h = Math.sin((block.id + 1) * 127.1 + (slotIndex + 1) * 311.7 + salt * 74.7) * 43758.5453;
        return h - Math.floor(h);
      };

      const rx = sx - impactWorldPos[0];
      const ry = sy - impactWorldPos[1];
      const rz = sz - impactWorldPos[2];
      const rDotN = rx * impactNormal[0] + ry * impactNormal[1] + rz * impactNormal[2];
      let tx = rx - impactNormal[0] * rDotN;
      let ty = ry - impactNormal[1] * rDotN;
      let tz = rz - impactNormal[2] * rDotN;
      const tLen = Math.hypot(tx, ty, tz);
      if (tLen > 1e-4) {
        tx /= tLen;
        ty /= tLen;
        tz /= tLen;
      } else {
        const randAng = hash01(1.0) * Math.PI * 2;
        tx = Math.cos(randAng);
        ty = 0;
        tz = Math.sin(randAng);
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
        body: shardBody,
        col: shardCol,
        worldPos: [sx, sy, sz],
        rawVel: [
          rigidVx + tx * splaySpeedSim + impactNormal[0] * normalReboundSim,
          rigidVy + ty * splaySpeedSim + impactNormal[1] * normalReboundSim,
          rigidVz + tz * splaySpeedSim + impactNormal[2] * normalReboundSim,
        ],
        rawSpin: [
          av.x + (hash01(3.0) - 0.5) * spinScale,
          av.y + (hash01(4.0) - 0.5) * spinScale,
          av.z + (hash01(5.0) - 0.5) * spinScale,
        ],
        mass: shardBody.mass(),
      });
    }

    if (spawned.length > 0) {
      // 1. Exact mass normalization so sum(m_k) == parentMass (71.10 g)
      let rawSumMass = 0;
      for (const s of spawned) rawSumMass += s.mass;
      if (rawSumMass > 1e-6 && Math.abs(rawSumMass - parentMass) > 1e-6) {
        const massScale = parentMass / rawSumMass;
        for (const s of spawned) {
          s.col.setMass(s.mass * massScale);
          s.mass = s.body.mass();
        }
      }

      // 2. Subtract mass-weighted mean burst velocity so linear momentum sum(m_k * v_k) === M0 * v0
      let sumMass = 0;
      let comX = 0, comY = 0, comZ = 0;
      let pxSum = 0, pySum = 0, pzSum = 0;
      for (const s of spawned) {
        sumMass += s.mass;
        comX += s.mass * s.worldPos[0];
        comY += s.mass * s.worldPos[1];
        comZ += s.mass * s.worldPos[2];
        pxSum += s.mass * s.rawVel[0];
        pySum += s.mass * s.rawVel[1];
        pzSum += s.mass * s.rawVel[2];
      }
      const invM = 1.0 / Math.max(1e-6, sumMass);
      comX *= invM;
      comY *= invM;
      comZ *= invM;
      const dvCorrX = lv.x - pxSum * invM;
      const dvCorrY = lv.y - pySum * invM;
      const dvCorrZ = lv.z - pzSum * invM;
      for (const s of spawned) {
        s.rawVel[0] += dvCorrX;
        s.rawVel[1] += dvCorrY;
        s.rawVel[2] += dvCorrZ;
      }

      // 3. Project out net angular momentum drift (orbital + spin) via rigid cluster tensor I_total
      let lSumX = 0, lSumY = 0, lSumZ = 0;
      const iTot = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (const s of spawned) {
        const rx = s.worldPos[0] - comX;
        const ry = s.worldPos[1] - comY;
        const rz = s.worldPos[2] - comZ;
        const m = s.mass;

        // Orbital r x (m v)
        lSumX += m * (ry * s.rawVel[2] - rz * s.rawVel[1]);
        lSumY += m * (rz * s.rawVel[0] - rx * s.rawVel[2]);
        lSumZ += m * (rx * s.rawVel[1] - ry * s.rawVel[0]);

        // Orbital contribution m * (|r|^2 I - r r^T) to I_total
        const r2 = rx * rx + ry * ry + rz * rz;
        iTot[0] += m * (r2 - rx * rx);
        iTot[1] -= m * rx * ry;
        iTot[2] -= m * rx * rz;
        iTot[3] -= m * ry * rx;
        iTot[4] += m * (r2 - ry * ry);
        iTot[5] -= m * ry * rz;
        iTot[6] -= m * rz * rx;
        iTot[7] -= m * rz * ry;
        iTot[8] += m * (r2 - rz * rz);

        // Spin inertia tensor R * diag(I_p) * R^T in world frame
        const spI = s.body.principalInertia();
        const spF = s.body.principalInertiaLocalFrame();
        const qS = PhysicsWorld.quatMul(q, [spF.x, spF.y, spF.z, spF.w]);
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
        const wPrin = quatInverseRotateVec3(qS, s.rawSpin);
        const lSpin = quatRotateVec3(qS, [
          spI.x * wPrin[0],
          spI.y * wPrin[1],
          spI.z * wPrin[2],
        ]);
        lSumX += lSpin[0];
        lSumY += lSpin[1];
        lSumZ += lSpin[2];
      }

      const lErr: Vec3 = [
        lSumX - parentL0[0],
        lSumY - parentL0[1],
        lSumZ - parentL0[2],
      ];
      const dOmega = PhysicsWorld.solve3x3(iTot, lErr);

      for (const s of spawned) {
        const rx = s.worldPos[0] - comX;
        const ry = s.worldPos[1] - comY;
        const rz = s.worldPos[2] - comZ;
        const vx = s.rawVel[0] - (dOmega[1] * rz - dOmega[2] * ry);
        const vy = s.rawVel[1] - (dOmega[2] * rx - dOmega[0] * rz);
        const vz = s.rawVel[2] - (dOmega[0] * ry - dOmega[1] * rx);
        const wx = s.rawSpin[0] - dOmega[0];
        const wy = s.rawSpin[1] - dOmega[1];
        const wz = s.rawSpin[2] - dOmega[2];

        s.body.setLinvel({ x: vx, y: vy, z: vz }, true);
        s.body.setAngvel({ x: wx, y: wy, z: wz }, true);

        const firstVertex = s.slotIndex * SHARD_SLOT_STRIDE_VERTS;
        this.pendingSlotUploads.set(s.slotIndex, s.geom.renderVertices);

        const shardState: GlassShardState = {
          id: this.nextShardId++,
          slotIndex: s.slotIndex,
          bodyHandle: s.body.handle,
          colliderHandle: s.col.handle,
          swatchId: block.swatchId,
          motif: block.motif,
          rippleSeed: block.rippleSeed,
          position: [...s.worldPos],
          rotation: [...q],
          prePos: [...s.worldPos],
          preQuat: [...q],
          preLinVel: [vx, vy, vz],
          preAngVel: [wx, wy, wz],
          volume: s.geom.volume,
          massKg: s.mass,
          preVy: vy,
          impactLocalPos: [...effectiveOrigin],
          planes: s.geom.planes,
          planeCount: s.geom.planeCount,
          firstVertex,
          vertexCount: s.geom.vertexCount,
          quietSubsteps: 0,
        };
        this.shards.push(shardState);
        this.colliderToShard.set(s.col.handle, shardState);
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

  private syncTransformsAndRules(): void {
    let maxTowerY = 0;
    let maxActiveLayer = 0;
    let sumX = 0;
    let sumZ = 0;
    let countInTower = 0;
    let maxTiltRad = 0;
    let sumWobbleOmega = 0;

    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb) continue;

      const t = rb.translation();
      const r = rb.rotation();
      const lv = rb.linvel();
      const av = rb.angvel();

      if (Math.hypot(t.x, t.z) > TABLE_RADIUS + 0.08) {
        const col = this.world.getCollider(b.colliderHandle);
        if (col) col.setCollisionGroups(GROUP_FREEFALL_OFF_RIM);
      }

      if (t.y < -2.4) {
        this.shatterBlock(b, [0, 0, 0], 22.0, [0, 1, 0]);
        continue;
      }

      b.position[0] = t.x;
      b.position[1] = t.y;
      b.position[2] = t.z;
      b.rotation[0] = r.x;
      b.rotation[1] = r.y;
      b.rotation[2] = r.z;
      b.rotation[3] = r.w;
      b.linVel[0] = lv.x;
      b.linVel[1] = lv.y;
      b.linVel[2] = lv.z;
      b.angVel[0] = av.x;
      b.angVel[1] = av.y;
      b.angVel[2] = av.z;

      if (!b.extracted && b.id !== this.activeBlockId) {
        const horizR = Math.hypot(t.x, t.z);
        if (horizR < 0.85 && t.y > 0.04) {
          countInTower++;
          sumX += t.x;
          sumZ += t.z;
          if (t.y + BLOCK_HALF_Y > maxTowerY) {
            maxTowerY = t.y + BLOCK_HALF_Y;
          }
          if (
            t.y >= Math.max(0.03, (b.assignedLayer - 0.85) * BLOCK_HEIGHT) &&
            b.assignedLayer > maxActiveLayer
          ) {
            maxActiveLayer = b.assignedLayer;
          }
          const upVec = quatRotateVec3(b.rotation, [0, 1, 0]);
          const tilt = Math.acos(Math.max(-1, Math.min(1, upVec[1])));
          if (tilt > maxTiltRad) maxTiltRad = tilt;
          sumWobbleOmega += Math.hypot(av.x, av.z);
        }
      }
    }

    // Sync shards & recycle slots for shards that fell off table edge
    for (let i = this.shards.length - 1; i >= 0; i--) {
      const s = this.shards[i];
      const rb = this.world.getRigidBody(s.bodyHandle);
      if (!rb) {
        this.freeShardSlots.push(s.slotIndex);
        this.shards.splice(i, 1);
        continue;
      }
      const t = rb.translation();
      if (Math.hypot(t.x, t.z) > TABLE_RADIUS + 0.04) {
        const col = this.world.getCollider(s.colliderHandle);
        if (col) col.setCollisionGroups(GROUP_SHARD_OFF_RIM);
      }
      if (t.y < -3.2) {
        this.colliderToShard.delete(s.colliderHandle);
        this.freeShardSlots.push(s.slotIndex);
        this.world.removeRigidBody(rb);
        this.shards.splice(i, 1);
        continue;
      }
      const r = rb.rotation();
      s.position[0] = t.x;
      s.position[1] = t.y;
      s.position[2] = t.z;
      s.rotation[0] = r.x;
      s.rotation[1] = r.y;
      s.rotation[2] = r.z;
      s.rotation[3] = r.w;
    }

    if (countInTower > 0) {
      this.centerOfMassXZCm = [(sumX / countInTower) * 10, (sumZ / countInTower) * 10];
      const instantLean = (maxTiltRad * 180) / Math.PI;
      this.leanAngleDeg = this.leanAngleDeg * 0.75 + Math.min(45, instantLean) * 0.25;
      const avgOmegaDeg = (((sumWobbleOmega * SIM_TIME_SCALE) / countInTower) * 180) / Math.PI;
      this.wobbleSpeedDegPerSec = this.wobbleSpeedDegPerSec * 0.7 + avgOmegaDeg * 0.3;
    }

    this.currentLayers = countInTower > 0 ? maxActiveLayer + 1 : 0;
    const heightCm = maxTowerY * 10.0;
    this.liveHeightCm = heightCm;
    if (this.turnPhase !== 'COLLAPSED') {
      if (heightCm > this.peakHeightCm) {
        this.peakHeightCm = heightCm;
      }
      if (this.currentLayers > this.peakLayers) {
        this.peakLayers = this.currentLayers;
      }
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
          nSumX += upV[0];
          nSumY += upV[1];
          nSumZ += upV[2];

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

    // Shortest-arc tilt quaternion mapping [0, 1, 0] -> upNormal
    const tiltAxisX = upNormal[2];
    const tiltAxisZ = -upNormal[0];
    const tiltSin = Math.hypot(tiltAxisX, tiltAxisZ);
    let tiltQuat: Quat = [0, 0, 0, 1];
    if (tiltSin > 1e-5) {
      const tiltAngle = Math.atan2(tiltSin, upNormal[1]);
      tiltQuat = quatFromAxisAngle([tiltAxisX / tiltSin, 0, tiltAxisZ / tiltSin], tiltAngle);
    }

    const yawQuat = quatFromAxisAngle([0, 1, 0], yawRad);
    const defaultRotation = PhysicsWorld.quatMul(tiltQuat, yawQuat);
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

  /**
   * Updates `topSupportFrame` and reference `openTopSlots` on the active top layer, advancing cleanly
   * once 3 blocks occupy the top row OR once wide-spaced/angled blocks leave no non-overlapping room.
   */
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
      if (
        !b.shattered &&
        !b.extracted &&
        b.id !== this.activeBlockId &&
        b.assignedLayer === highestLayer
      ) {
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
      const { uX: uAx, uZ: uAz } = PhysicsWorld.getBlockXZAxes(frame.defaultRotation);
      for (let i = 0; i < obstacles.length; i++) {
        const ob = obstacles[i];
        const obAxes = PhysicsWorld.getBlockXZAxes(ob.rotation);
        const ov = PhysicsWorld.obb2DOverlapDepth(
          px,
          pz,
          uAx,
          uAz,
          colHx + roomClearance,
          colHz + roomClearance,
          ob.position[0],
          ob.position[2],
          obAxes.uX,
          obAxes.uZ,
          colHx,
          colHz,
          frame.layerCenter[0],
          frame.layerCenter[2]
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
        const uTest = k * 0.015; // [-0.345, +0.345] dm supported CoM span
        if (candidateFits(highestFrame, blocksInHighest, uTest)) {
          hasRoom = true;
          if (
            fallbackOpenU === null ||
            Math.abs(uTest) < Math.abs(fallbackOpenU)
          ) {
            fallbackOpenU = uTest;
          }
        }
      }
      if (!hasRoom) {
        // Existing wide-spaced or angled blocks leave no supported non-overlapping room -> advance row!
        targetLayer = highestLayer + 1;
      }
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
      this.releaseLocalSolverWindow(true, false);
      this.turnPhase = 'COLLAPSED';
      this.isPointerDraggingBlock = false;
      if (this.activeBlockId !== null) {
        const carried = this.blocks[this.activeBlockId];
        if (carried && !carried.shattered) {
          carried.extracted = false;
          const col = this.world.getCollider(carried.colliderHandle);
          const rb = this.world.getRigidBody(carried.bodyHandle);
          if (col) col.setSensor(false);
          if (rb) rb.setGravityScale(1.0, true);
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

  public getTelemetry(): TowerTelemetry {
    let intactCount = 0;
    let crazedCount = 0;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (!b.shattered) {
        intactCount++;
        if (b.damage > 0.05) {
          crazedCount++;
        }
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

  private isDisposed: boolean = false;

  public dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    if (this.eventQueue) {
      this.eventQueue.free();
      (this as unknown as { eventQueue: null }).eventQueue = null;
    }
    if (this.world) {
      this.world.free();
      (this as unknown as { world: null }).world = null;
    }
  }
}

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

export type {
  AdversarialPlacementPlan,
  BlockCriticalityCandidate,
  SceneRaycastHit,
  SceneRaycastOptions,
};

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
export interface PhysicsWorld extends MujocoPhysicsEngine {}

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export class PhysicsWorld {
  constructor(
    audio: GlassAudioEngine,
    initialLayers: number = INITIAL_LAYERS,
    options?: PhysicsWorldOptions
  ) {
    return new MujocoPhysicsEngine(
      audio,
      initialLayers,
      undefined,
      options?.seed
    ) as unknown as PhysicsWorld;
  }
}

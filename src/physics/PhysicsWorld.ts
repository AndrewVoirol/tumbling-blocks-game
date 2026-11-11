import { ensureMujocoInitialized } from './mujocoRuntime';
import { BlockCriticalityCandidate, MujocoPhysicsEngine } from './MujocoPhysicsEngine';
import { INITIAL_LAYERS, TraceryMotif } from './constants';
import { Quat, Vec3 } from '../math/vecMat';
import { GlassAudioEngine } from '../audio/GlassAudioEngine';

export type { BlockCriticalityCandidate };

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
    _options?: PhysicsWorldOptions
  ) {
    return new MujocoPhysicsEngine(audio, initialLayers) as unknown as PhysicsWorld;
  }
}

import { Quat, Vec3 } from '../math/vecMat';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  BLOCK_HEIGHT,
  SHARD_SLOT_STRIDE_VERTS,
  STAINED_GLASS_PALETTE,
  TraceryMotif,
} from '../physics/constants';
import {
  GameMode,
  GlassBlockState,
  MicroSplinter,
  TurnPhase,
} from '../physics/PhysicsWorld';

export const MP_PROTOCOL_MAGIC = 0x564d; // 'VM'
export const MP_PROTOCOL_VERSION = 1;
export const MP_HEADER_BYTES = 140;
export const MP_BLOCK_STRIDE_BYTES = 32;
export const MP_SHARD_STRIDE_BYTES = 16;
export const MP_SPLINTER_STRIDE_BYTES = 12;
export const MP_MAX_SHARDS_PER_FRAME = 120;
export const MP_MAX_SPLINTERS_PER_FRAME = 10;

export const ROOM_CODE_ALPHABET = '23456789ABCDEFGHJKMNPQRSTUVWXYZ';

export type MultiplayerRole = 'host' | 'guest';

export type MultiplayerConnectionState =
  | 'idle'
  | 'connecting'
  | 'waiting_peer'
  | 'connected'
  | 'reconnecting'
  | 'error';

export type MultiplayerTransportKind = 'none' | 'webrtc-p2p' | 'ws-relay';

export interface MultiplayerStatusSnapshot {
  active: boolean;
  role: MultiplayerRole | null;
  roomCode: string | null;
  roomSeed: number;
  connectionState: MultiplayerConnectionState;
  transport: MultiplayerTransportKind;
  peerConnected: boolean;
  pingMs: number | null;
  errorMessage: string | null;
}

export interface SerializedShardMeta {
  id: number;
  slotIndex: number;
  swatchId: number;
  motif: TraceryMotif;
  rippleSeed: number;
  localCentroid: Vec3;
  impactLocalPos: Vec3;
  planeCount: number;
  /** Length = planeCount * 4: [nx, ny, nz, signedD] */
  planes: number[];
}

export type NetworkAudioEvent =
  | {
      kind: 'slideFriction';
      slideSpeedDmPerSec: number;
      resistance01: number;
      slipPowerMw: number;
      torsionRadPerSec: number;
      clampingLoadN?: number;
    }
  | {
      kind: 'tapTest';
      pinch01: number;
      worldPos?: Vec3;
      damage?: number;
    }
  | {
      kind: 'glassImpact';
      intensity01: number;
      pitchMul: number;
      worldPos?: Vec3;
      impactSpeedMps?: number;
      damage?: number;
      surface?: 'glass' | 'wood';
    }
  | {
      kind: 'glassCrack';
      damage01: number;
      worldPos?: Vec3;
    }
  | {
      kind: 'glassShatter';
      intensity01: number;
      hitTable: boolean;
      worldPos?: Vec3;
    }
  | {
      kind: 'rankedShards';
      events: Array<{
        energy: number;
        intensity01: number;
        volumeDm3: number;
        worldPos: Vec3;
        hitGlass: boolean;
      }>;
    }
  | {
      kind: 'shardImpacts';
      impacts: Array<{
        pos: Vec3;
        speed: number;
        massKg: number;
        surface: 'wood' | 'glass';
        slotIndex: number;
      }>;
    }
  | {
      kind: 'wobbleRock';
      tiltAngleRad: number;
      angularSpeedRadPerSec: number;
      clampingLoadN: number;
      worldPos: Vec3;
      surface: 'glass' | 'wood';
      damage: number;
    }
  | {
      kind: 'stackSuccess';
      layerCount: number;
    };

export type GuestRemoteCommand =
  | { kind: 'hover_block'; blockId: number | null }
  | { kind: 'tap_block'; blockId: number; localNormal: Vec3; localPoint: Vec3 }
  | { kind: 'begin_pull'; blockId: number; localPoint: Vec3; worldPoint: Vec3 }
  | { kind: 'update_pull'; worldTarget: Vec3 }
  | { kind: 'end_pull'; dropIfPoisedOverTower: boolean }
  | { kind: 'nudge_placement'; rayOrigin: Vec3; rayDir: Vec3 }
  | { kind: 'rotate_placement'; deltaYawRad: number }
  | { kind: 'reset_rotation' }
  | { kind: 'confirm_placement' };

export type MultiplayerControlMessage =
  | {
      type: 'hello';
      protocol: number;
      role: MultiplayerRole;
      clientId: string;
      roomCode: string;
    }
  | {
      type: 'room_welcome';
      roomCode: string;
      roomSeed: number;
      initialLayers: number;
      collapseReason: string | null;
      activeShardMetas: SerializedShardMeta[];
      retiredShardIds: number[];
    }
  | {
      type: 'request_resync';
      lastSeq: number;
    }
  | {
      type: 'shard_spawn';
      spawned: SerializedShardMeta[];
      retiredIds: number[];
    }
  | {
      type: 'collapse_info';
      collapseReason: string | null;
    }
  | {
      type: 'audio_batch';
      events: NetworkAudioEvent[];
    }
  | {
      type: 'guest_cmd';
      seq: number;
      cmd: GuestRemoteCommand;
    }
  | {
      type: 'rematch_request';
    }
  | {
      type: 'ping';
      id: number;
      t0: number;
    }
  | {
      type: 'pong';
      id: number;
      t0: number;
    }
  | {
      type: 'peer_bye';
      clientId: string;
    }
  | {
      type: 'rtc_signal';
      from: MultiplayerRole;
      signalType: 'offer' | 'answer' | 'candidate';
      sdp?: RTCSessionDescriptionInit;
      candidate?: RTCIceCandidateInit;
    };

export interface DecodedBlockFrameState {
  position: Vec3;
  rotation: Quat;
  damage: number;
  shattered: boolean;
  extracted: boolean;
  isLegalTarget: boolean;
  isLooseSlot: boolean;
  swatchId: number;
  motif: TraceryMotif;
  assignedLayer: number;
  assignedSlot: number;
  stress01: number;
  crackLocalOrigin: Vec3;
  photoelasticWrench: [number, number, number, number];
}

export interface DecodedShardFrameState {
  id: number;
  position: Vec3;
  rotation: Quat;
}

export interface MultiplayerStateFrame {
  seq: number;
  hostElapsedSec: number;
  roomSeed: number;
  turnPhase: TurnPhase;
  gameMode: GameMode;
  currentPlayer: number;
  playerCount: number;
  blocksStacked: number;
  shatteredCount: number;
  currentLayers: number;
  peakLayers: number;
  activeBlockId: number | null;
  hoveredBlockId: number | null;
  hoveredSlotIndex: number | null;
  lockedMinRow: number | null;
  isPointerDraggingBlock: boolean;
  isExtractionLiftReady: boolean;
  polariscopeMode: boolean;
  grabResistance01: number;
  settleProgress01: number;
  leanAngleDeg: number;
  wobbleSpeedDegPerSec: number;
  liveHeightCm: number;
  peakHeightCm: number;
  centerOfMassXZCm: [number, number];
  carriedYawOffsetRad: number;
  grabCurrentWorldOnBlock: Vec3;
  grabWorldAnchor: Vec3;
  carriedLandingPos: Vec3;
  carriedTargetQuat: Quat;
  topSupportSurfaceCenter: Vec3;
  topSupportUpNormal: Vec3;
  topSupportDefaultRot: Quat;
  topSupportTargetLayer: number;
  blocks: DecodedBlockFrameState[];
  shards: DecodedShardFrameState[];
  splinters: MicroSplinter[];
}

const TURN_PHASES: readonly TurnPhase[] = [
  'SELECT_BLOCK',
  'PULLING',
  'EXTRACTED',
  'SETTLING',
  'COLLAPSED',
];

const GAME_MODES: readonly GameMode[] = ['solo', 'party', 'sandbox'];

/**
 * Deterministic 32-bit Mulberry32 PRNG (`[0, 1)`).
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateRoomCode(length: number = 5): string {
  const bytes = new Uint8Array(length);
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < length; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  let out = '';
  for (let i = 0; i < length; i++) {
    out += ROOM_CODE_ALPHABET[bytes[i] % ROOM_CODE_ALPHABET.length];
  }
  return out;
}

export function normalizeRoomCode(raw: string): string {
  return raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, '')
    .slice(0, 8);
}

export function isValidRoomCode(code: string): boolean {
  const norm = normalizeRoomCode(code);
  return norm.length >= 4 && norm.length <= 8 && /^[A-Z0-9]+$/.test(norm);
}

export function generateRoomSeed(): number {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const arr = new Uint32Array(1);
    crypto.getRandomValues(arr);
    return (arr[0] >>> 0) || 0x56495452;
  }
  return ((Math.random() * 0xffffffff) >>> 0) || 0x56495452;
}

/**
 * Deterministically builds the sequential stained-glass palette swatch, tracery motif, and
 * waterglass ripple phase table for an `N`-layer tower from `roomSeed`, matching
 * `MujocoPhysicsEngine.rebuildTower`'s sequential `mulberry32` stream 1:1 (`54/54` parity
 * across Worker Host, Sync Host, and Guest).
 */
export function buildDeterministicTowerMaterials(
  roomSeed: number,
  towerLayers: number = 18
): Array<{ swatchId: number; motif: TraceryMotif; rippleSeed: number }> {
  const paletteLen = STAINED_GLASS_PALETTE.length;
  const seed = roomSeed >>> 0;
  const total = Math.max(0, towerLayers * 3);
  const out: Array<{ swatchId: number; motif: TraceryMotif; rippleSeed: number }> = new Array(
    total
  );
  const seedRng =
    seed !== 0 ? mulberry32((seed ^ Math.imul(towerLayers, 0x9e3779b9)) >>> 0) : null;
  let prevRowSwatch0 = -1;
  let prevRowSwatch1 = -1;

  for (let id = 0; id < total; id++) {
    const layer = Math.floor(id / 3);
    const slot = id % 3;
    let swatchId: number;
    let motif: TraceryMotif;
    let rippleSeed: number;
    if (seedRng) {
      if (slot === 0) {
        prevRowSwatch0 = -1;
        prevRowSwatch1 = -1;
      }
      swatchId = Math.floor(seedRng() * paletteLen) % paletteLen;
      if (slot === 2 && swatchId === prevRowSwatch0 && swatchId === prevRowSwatch1) {
        swatchId = (swatchId + 1) % paletteLen;
      }
      if (slot === 0) prevRowSwatch0 = swatchId;
      if (slot === 1) prevRowSwatch1 = swatchId;
      motif = (Math.floor(seedRng() * 4) & 3) as TraceryMotif;
      rippleSeed = +(0.25 + seedRng() * 96.0).toFixed(5);
    } else {
      swatchId = (layer * 3 + slot * 2 + Math.floor(layer / 2)) % paletteLen;
      motif = 0;
      rippleSeed = (id + 1) * 1.6180339;
    }
    out[id] = { swatchId, motif, rippleSeed };
  }
  return out;
}

export function computeDeterministicBlockMaterial(
  roomSeed: number,
  blockId: number,
  layer: number,
  slot: number,
  towerLayers: number = 18
): { swatchId: number; motif: TraceryMotif; rippleSeed: number } {
  const neededLayers = Math.max(towerLayers, Math.floor(blockId / 3) + 1);
  if (blockId >= 0 && blockId < towerLayers * 3) {
    return buildDeterministicTowerMaterials(roomSeed, towerLayers)[blockId];
  }
  const rng = mulberry32(
    ((roomSeed >>> 0) ^ Math.imul(blockId + 1, 0x9e3779b9) ^ (neededLayers * 131 + layer * 17 + slot)) >>> 0
  );
  const paletteLen = STAINED_GLASS_PALETTE.length;
  const swatchId = Math.floor(rng() * paletteLen) % paletteLen;
  const motif = (Math.floor(rng() * 4) & 3) as TraceryMotif;
  const rippleSeed = +(0.25 + rng() * 96.0).toFixed(5);
  return { swatchId, motif, rippleSeed };
}

export function applyDeterministicTowerPalette(
  blocks: GlassBlockState[],
  roomSeed: number,
  towerLayers?: number
): void {
  const inferredLayers =
    towerLayers ?? (blocks.length >= 3 ? Math.round(blocks.length / 3) : 18);
  const materials = buildDeterministicTowerMaterials(roomSeed >>> 0, inferredLayers);
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    const mat =
      (b.id >= 0 && b.id < materials.length ? materials[b.id] : undefined) ??
      computeDeterministicBlockMaterial(
        roomSeed >>> 0,
        b.id,
        b.initialLayer,
        b.assignedSlot,
        inferredLayers
      );
    b.swatchId = mat.swatchId;
    b.motif = mat.motif;
    b.rippleSeed = mat.rippleSeed;
  }
}

/**
 * Canonicalizes a unit quaternion into the `w >= 0` closed hemisphere (`S^3 / {+1, -1}`)
 * for compact `Int16` wire encoding.
 */
export function canonicalizeQuatHemisphere(q: Quat): Quat {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1.0;
  let x = q[0] / len;
  let y = q[1] / len;
  let z = q[2] / len;
  let w = q[3] / len;
  const flip =
    w < 0 ||
    (w === 0 && (z < 0 || (z === 0 && (y < 0 || (y === 0 && x < 0)))));
  if (flip) {
    x = -x;
    y = -y;
    z = -z;
    w = -w;
  }
  return [x, y, z, w];
}

/**
 * Shortest-arc hemisphere-safe spherical linear interpolation (`dot(q0, q1) < 0` antipodal guard).
 * Eliminates `360 deg` spin pops when a rotating prism or shard crosses the `w = 0` equator.
 */
export function quatSlerpHemisphere(q0: Quat, q1: Quat, t: number): Quat {
  const u = Math.max(0, Math.min(1, t));
  const l0 = Math.hypot(q0[0], q0[1], q0[2], q0[3]) || 1.0;
  const l1 = Math.hypot(q1[0], q1[1], q1[2], q1[3]) || 1.0;
  const ax = q0[0] / l0;
  const ay = q0[1] / l0;
  const az = q0[2] / l0;
  const aw = q0[3] / l0;

  let bx = q1[0] / l1;
  let by = q1[1] / l1;
  let bz = q1[2] / l1;
  let bw = q1[3] / l1;

  let dot = ax * bx + ay * by + az * bz + aw * bw;
  if (dot < 0.0) {
    dot = -dot;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }

  let s0: number;
  let s1: number;
  if (dot > 0.9995) {
    s0 = 1.0 - u;
    s1 = u;
  } else {
    const theta = Math.acos(Math.min(1.0, dot));
    const invSin = 1.0 / Math.sin(theta);
    s0 = Math.sin((1.0 - u) * theta) * invSin;
    s1 = Math.sin(u * theta) * invSin;
  }

  const rx = s0 * ax + s1 * bx;
  const ry = s0 * ay + s1 * by;
  const rz = s0 * az + s1 * bz;
  const rw = s0 * aw + s1 * bw;
  const rLen = Math.hypot(rx, ry, rz, rw) || 1.0;
  return [rx / rLen, ry / rLen, rz / rLen, rw / rLen];
}

function clamp01(v: number): number {
  return v <= 0 ? 0 : v >= 1 ? 1 : v;
}

function quantizeSnorm16(v: number): number {
  const c = v <= -1 ? -1 : v >= 1 ? 1 : v;
  return Math.round(c * 32767);
}

function dequantizeSnorm16(q: number): number {
  return Math.max(-1, Math.min(1, q / 32767));
}

function writeQuatI16(view: DataView, byteOffset: number, q: Quat): void {
  const cq = canonicalizeQuatHemisphere(q);
  view.setInt16(byteOffset + 0, quantizeSnorm16(cq[0]), true);
  view.setInt16(byteOffset + 2, quantizeSnorm16(cq[1]), true);
  view.setInt16(byteOffset + 4, quantizeSnorm16(cq[2]), true);
  view.setInt16(byteOffset + 6, quantizeSnorm16(cq[3]), true);
}

function readQuatI16(view: DataView, byteOffset: number): Quat {
  const x = dequantizeSnorm16(view.getInt16(byteOffset + 0, true));
  const y = dequantizeSnorm16(view.getInt16(byteOffset + 2, true));
  const z = dequantizeSnorm16(view.getInt16(byteOffset + 4, true));
  const w = dequantizeSnorm16(view.getInt16(byteOffset + 6, true));
  const len = Math.hypot(x, y, z, w) || 1.0;
  return [x / len, y / len, z / len, w / len];
}

/**
 * Encodes the authoritative Host simulation snapshot into a compact `< 4 KB` binary `ArrayBuffer`
 * (`~1,868 B` for a 54-block tower; `<= 3,908 B` even during a 120-shard collapse burst).
 */
export function encodeMultiplayerStateFrame(frame: MultiplayerStateFrame): ArrayBuffer {
  const blockCount = Math.min(128, frame.blocks.length);
  const shardCount = Math.min(MP_MAX_SHARDS_PER_FRAME, frame.shards.length);
  const splinterCount = Math.min(MP_MAX_SPLINTERS_PER_FRAME, frame.splinters.length);

  const totalBytes =
    MP_HEADER_BYTES +
    blockCount * MP_BLOCK_STRIDE_BYTES +
    shardCount * MP_SHARD_STRIDE_BYTES +
    splinterCount * MP_SPLINTER_STRIDE_BYTES;

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);

  const flags =
    (frame.isPointerDraggingBlock ? 1 : 0) |
    (frame.isExtractionLiftReady ? 2 : 0) |
    (frame.polariscopeMode ? 4 : 0) |
    (frame.activeBlockId !== null ? 8 : 0) |
    (frame.hoveredBlockId !== null ? 16 : 0) |
    (frame.hoveredSlotIndex !== null ? 32 : 0) |
    (frame.lockedMinRow !== null ? 64 : 0);

  const phaseIdx = Math.max(0, TURN_PHASES.indexOf(frame.turnPhase));
  const modeIdx = Math.max(0, GAME_MODES.indexOf(frame.gameMode));

  view.setUint16(0, MP_PROTOCOL_MAGIC, true);
  view.setUint8(2, MP_PROTOCOL_VERSION);
  view.setUint8(3, flags);
  view.setUint32(4, frame.seq >>> 0, true);
  view.setFloat32(8, frame.hostElapsedSec, true);
  view.setUint32(12, frame.roomSeed >>> 0, true);
  view.setUint8(16, phaseIdx);
  view.setUint8(17, modeIdx);
  view.setUint8(18, frame.currentPlayer & 0xff);
  view.setUint8(19, frame.playerCount & 0xff);
  view.setUint16(20, Math.max(0, Math.min(65535, frame.blocksStacked | 0)), true);
  view.setUint8(22, Math.max(0, Math.min(255, frame.shatteredCount | 0)));
  view.setUint8(23, Math.max(0, Math.min(255, frame.currentLayers | 0)));
  view.setUint8(24, Math.max(0, Math.min(255, frame.peakLayers | 0)));
  view.setUint8(25, Math.round(clamp01(frame.grabResistance01) * 255));
  view.setUint8(26, Math.round(clamp01(frame.settleProgress01) * 255));
  view.setUint8(27, splinterCount);
  view.setInt16(28, frame.activeBlockId ?? -1, true);
  view.setInt16(30, frame.hoveredBlockId ?? -1, true);
  view.setInt16(32, frame.hoveredSlotIndex ?? -1, true);
  view.setInt16(34, frame.lockedMinRow ?? -1, true);
  view.setUint16(36, blockCount, true);
  view.setUint16(38, shardCount, true);

  view.setFloat32(40, frame.leanAngleDeg, true);
  view.setFloat32(44, frame.wobbleSpeedDegPerSec, true);
  view.setFloat32(48, frame.liveHeightCm, true);
  view.setFloat32(52, frame.peakHeightCm, true);
  view.setFloat32(56, frame.centerOfMassXZCm[0], true);
  view.setFloat32(60, frame.centerOfMassXZCm[1], true);
  view.setFloat32(64, frame.carriedYawOffsetRad, true);

  view.setFloat32(68, frame.grabCurrentWorldOnBlock[0], true);
  view.setFloat32(72, frame.grabCurrentWorldOnBlock[1], true);
  view.setFloat32(76, frame.grabCurrentWorldOnBlock[2], true);

  view.setFloat32(80, frame.grabWorldAnchor[0], true);
  view.setFloat32(84, frame.grabWorldAnchor[1], true);
  view.setFloat32(88, frame.grabWorldAnchor[2], true);

  view.setFloat32(92, frame.carriedLandingPos[0], true);
  view.setFloat32(96, frame.carriedLandingPos[1], true);
  view.setFloat32(100, frame.carriedLandingPos[2], true);

  writeQuatI16(view, 104, frame.carriedTargetQuat);

  view.setFloat32(112, frame.topSupportSurfaceCenter[0], true);
  view.setFloat32(116, frame.topSupportSurfaceCenter[1], true);
  view.setFloat32(120, frame.topSupportSurfaceCenter[2], true);

  view.setInt16(124, quantizeSnorm16(frame.topSupportUpNormal[0]), true);
  view.setInt16(126, quantizeSnorm16(frame.topSupportUpNormal[1]), true);
  view.setInt16(128, quantizeSnorm16(frame.topSupportUpNormal[2]), true);

  writeQuatI16(view, 130, frame.topSupportDefaultRot);
  view.setUint16(138, Math.max(0, Math.min(65535, frame.topSupportTargetLayer | 0)), true);

  let offset = MP_HEADER_BYTES;
  for (let i = 0; i < blockCount; i++) {
    const b = frame.blocks[i];
    view.setFloat32(offset + 0, b.position[0], true);
    view.setFloat32(offset + 4, b.position[1], true);
    view.setFloat32(offset + 8, b.position[2], true);
    writeQuatI16(view, offset + 12, b.rotation);

    view.setUint8(offset + 20, Math.round(clamp01(b.damage) * 255));
    const stateBits =
      (b.shattered ? 1 : 0) |
      (b.extracted ? 2 : 0) |
      (b.isLegalTarget ? 4 : 0) |
      (b.isLooseSlot ? 8 : 0) |
      ((b.swatchId & 0x07) << 4);
    view.setUint8(offset + 21, stateBits);
    view.setUint8(
      offset + 22,
      (Math.max(0, Math.min(63, b.assignedLayer | 0)) & 0x3f) | ((b.assignedSlot & 0x03) << 6)
    );
    view.setUint8(offset + 23, Math.round(clamp01(b.stress01) * 255));

    const cx = Math.max(-1, Math.min(1, b.crackLocalOrigin[0] / BLOCK_HALF_X));
    const cy = Math.max(-1, Math.min(1, b.crackLocalOrigin[1] / BLOCK_HALF_Y));
    const cz = Math.max(-1, Math.min(1, b.crackLocalOrigin[2] / BLOCK_HALF_Z));
    view.setInt8(offset + 24, Math.round(cx * 127));
    view.setInt8(offset + 25, Math.round(cy * 127));
    view.setInt8(offset + 26, Math.round(cz * 127));
    view.setUint8(offset + 27, b.motif & 0x03);

    view.setUint8(offset + 28, Math.round(clamp01(b.photoelasticWrench[0]) * 255));
    view.setUint8(offset + 29, Math.round(clamp01(b.photoelasticWrench[1]) * 255));
    view.setUint8(offset + 30, Math.round(clamp01(b.photoelasticWrench[2]) * 255));
    view.setUint8(offset + 31, Math.round(clamp01(b.photoelasticWrench[3]) * 255));
    offset += MP_BLOCK_STRIDE_BYTES;
  }

  for (let i = 0; i < shardCount; i++) {
    const s = frame.shards[i];
    view.setUint16(offset + 0, s.id & 0xffff, true);
    const px = Math.round(Math.max(-16.0, Math.min(16.0, s.position[0])) * 2000);
    const py = Math.round(Math.max(-16.0, Math.min(16.0, s.position[1])) * 2000);
    const pz = Math.round(Math.max(-16.0, Math.min(16.0, s.position[2])) * 2000);
    view.setInt16(offset + 2, px, true);
    view.setInt16(offset + 4, py, true);
    view.setInt16(offset + 6, pz, true);
    writeQuatI16(view, offset + 8, s.rotation);
    offset += MP_SHARD_STRIDE_BYTES;
  }

  for (let i = 0; i < splinterCount; i++) {
    const sp = frame.splinters[i];
    const px = Math.round(Math.max(-16.0, Math.min(16.0, sp.pos[0])) * 2000);
    const py = Math.round(Math.max(-16.0, Math.min(16.0, sp.pos[1])) * 2000);
    const pz = Math.round(Math.max(-16.0, Math.min(16.0, sp.pos[2])) * 2000);
    view.setInt16(offset + 0, px, true);
    view.setInt16(offset + 2, py, true);
    view.setInt16(offset + 4, pz, true);
    view.setUint8(offset + 6, Math.round(clamp01(sp.scale / 0.12) * 255));
    const lifeRatio = sp.maxLife > 1e-5 ? clamp01(sp.life / sp.maxLife) : 0;
    view.setUint8(offset + 7, Math.round(lifeRatio * 255));
    view.setUint8(offset + 8, sp.swatchId & 0x07);
    const normRot = ((sp.rot % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
    view.setUint8(offset + 9, Math.round((normRot / (Math.PI * 2)) * 255));
    view.setUint16(offset + 10, 0, true);
    offset += MP_SPLINTER_STRIDE_BYTES;
  }

  return buffer;
}

export function decodeMultiplayerStateFrame(buffer: ArrayBuffer): MultiplayerStateFrame | null {
  if (buffer.byteLength < MP_HEADER_BYTES) return null;
  const view = new DataView(buffer);
  if (view.getUint16(0, true) !== MP_PROTOCOL_MAGIC) return null;
  if (view.getUint8(2) !== MP_PROTOCOL_VERSION) return null;

  const flags = view.getUint8(3);
  const seq = view.getUint32(4, true);
  const hostElapsedSec = view.getFloat32(8, true);
  const roomSeed = view.getUint32(12, true);
  const phaseIdx = view.getUint8(16);
  const modeIdx = view.getUint8(17);
  const currentPlayer = view.getUint8(18);
  const playerCount = view.getUint8(19);
  const blocksStacked = view.getUint16(20, true);
  const shatteredCount = view.getUint8(22);
  const currentLayers = view.getUint8(23);
  const peakLayers = view.getUint8(24);
  const grabResistance01 = view.getUint8(25) / 255;
  const settleProgress01 = view.getUint8(26) / 255;
  const splinterCount = view.getUint8(27);
  const rawActiveId = view.getInt16(28, true);
  const rawHoverId = view.getInt16(30, true);
  const rawHoverSlot = view.getInt16(32, true);
  const rawLockedRow = view.getInt16(34, true);
  const blockCount = view.getUint16(36, true);
  const shardCount = view.getUint16(38, true);

  const expectedBytes =
    MP_HEADER_BYTES +
    blockCount * MP_BLOCK_STRIDE_BYTES +
    shardCount * MP_SHARD_STRIDE_BYTES +
    splinterCount * MP_SPLINTER_STRIDE_BYTES;
  if (buffer.byteLength < expectedBytes) return null;

  const leanAngleDeg = view.getFloat32(40, true);
  const wobbleSpeedDegPerSec = view.getFloat32(44, true);
  const liveHeightCm = view.getFloat32(48, true);
  const peakHeightCm = view.getFloat32(52, true);
  const comX = view.getFloat32(56, true);
  const comZ = view.getFloat32(60, true);
  const carriedYawOffsetRad = view.getFloat32(64, true);

  const grabCurrentWorldOnBlock: Vec3 = [
    view.getFloat32(68, true),
    view.getFloat32(72, true),
    view.getFloat32(76, true),
  ];
  const grabWorldAnchor: Vec3 = [
    view.getFloat32(80, true),
    view.getFloat32(84, true),
    view.getFloat32(88, true),
  ];
  const carriedLandingPos: Vec3 = [
    view.getFloat32(92, true),
    view.getFloat32(96, true),
    view.getFloat32(100, true),
  ];
  const carriedTargetQuat = readQuatI16(view, 104);

  const topSupportSurfaceCenter: Vec3 = [
    view.getFloat32(112, true),
    view.getFloat32(116, true),
    view.getFloat32(120, true),
  ];
  const nx = dequantizeSnorm16(view.getInt16(124, true));
  const ny = dequantizeSnorm16(view.getInt16(126, true));
  const nz = dequantizeSnorm16(view.getInt16(128, true));
  const nLen = Math.hypot(nx, ny, nz) || 1.0;
  const topSupportUpNormal: Vec3 = [nx / nLen, ny / nLen, nz / nLen];
  const topSupportDefaultRot = readQuatI16(view, 130);
  const topSupportTargetLayer = view.getUint16(138, true);

  let offset = MP_HEADER_BYTES;
  const blocks: DecodedBlockFrameState[] = new Array(blockCount);
  for (let i = 0; i < blockCount; i++) {
    const px = view.getFloat32(offset + 0, true);
    const py = view.getFloat32(offset + 4, true);
    const pz = view.getFloat32(offset + 8, true);
    const rotation = readQuatI16(view, offset + 12);
    const damage = view.getUint8(offset + 20) / 255;
    const stateBits = view.getUint8(offset + 21);
    const layerAndSlot = view.getUint8(offset + 22);
    const stress01 = view.getUint8(offset + 23) / 255;
    const cx = (view.getInt8(offset + 24) / 127) * BLOCK_HALF_X;
    const cy = (view.getInt8(offset + 25) / 127) * BLOCK_HALF_Y;
    const cz = (view.getInt8(offset + 26) / 127) * BLOCK_HALF_Z;
    const motif = (view.getUint8(offset + 27) & 0x03) as TraceryMotif;

    blocks[i] = {
      position: [px, py, pz],
      rotation,
      damage,
      shattered: (stateBits & 1) !== 0,
      extracted: (stateBits & 2) !== 0,
      isLegalTarget: (stateBits & 4) !== 0,
      isLooseSlot: (stateBits & 8) !== 0,
      swatchId: (stateBits >>> 4) & 0x07,
      motif,
      assignedLayer: layerAndSlot & 0x3f,
      assignedSlot: (layerAndSlot >>> 6) & 0x03,
      stress01,
      crackLocalOrigin: [cx, cy, cz],
      photoelasticWrench: [
        view.getUint8(offset + 28) / 255,
        view.getUint8(offset + 29) / 255,
        view.getUint8(offset + 30) / 255,
        view.getUint8(offset + 31) / 255,
      ],
    };
    offset += MP_BLOCK_STRIDE_BYTES;
  }

  const shards: DecodedShardFrameState[] = new Array(shardCount);
  for (let i = 0; i < shardCount; i++) {
    const id = view.getUint16(offset + 0, true);
    const sx = view.getInt16(offset + 2, true) / 2000;
    const sy = view.getInt16(offset + 4, true) / 2000;
    const sz = view.getInt16(offset + 6, true) / 2000;
    const rotation = readQuatI16(view, offset + 8);
    shards[i] = {
      id,
      position: [sx, sy, sz],
      rotation,
    };
    offset += MP_SHARD_STRIDE_BYTES;
  }

  const splinters: MicroSplinter[] = new Array(splinterCount);
  for (let i = 0; i < splinterCount; i++) {
    const sx = view.getInt16(offset + 0, true) / 2000;
    const sy = view.getInt16(offset + 2, true) / 2000;
    const sz = view.getInt16(offset + 4, true) / 2000;
    const scale = (view.getUint8(offset + 6) / 255) * 0.12;
    const life = view.getUint8(offset + 7) / 255;
    const swatchId = view.getUint8(offset + 8) & 0x07;
    const rot = (view.getUint8(offset + 9) / 255) * Math.PI * 2;
    splinters[i] = {
      pos: [sx, sy, sz],
      vel: [0, 0, 0],
      rot,
      rotSpeed: 0,
      scale,
      swatchId,
      life,
      maxLife: 1.0,
    };
    offset += MP_SPLINTER_STRIDE_BYTES;
  }

  return {
    seq,
    hostElapsedSec,
    roomSeed,
    turnPhase: TURN_PHASES[phaseIdx] ?? 'SELECT_BLOCK',
    gameMode: GAME_MODES[modeIdx] ?? 'party',
    currentPlayer,
    playerCount,
    blocksStacked,
    shatteredCount,
    currentLayers,
    peakLayers,
    activeBlockId: (flags & 8) !== 0 && rawActiveId >= 0 ? rawActiveId : null,
    hoveredBlockId: (flags & 16) !== 0 && rawHoverId >= 0 ? rawHoverId : null,
    hoveredSlotIndex: (flags & 32) !== 0 && rawHoverSlot >= 0 ? rawHoverSlot : null,
    lockedMinRow: (flags & 64) !== 0 && rawLockedRow >= 0 ? rawLockedRow : null,
    isPointerDraggingBlock: (flags & 1) !== 0,
    isExtractionLiftReady: (flags & 2) !== 0,
    polariscopeMode: (flags & 4) !== 0,
    grabResistance01,
    settleProgress01,
    leanAngleDeg,
    wobbleSpeedDegPerSec,
    liveHeightCm,
    peakHeightCm,
    centerOfMassXZCm: [comX, comZ],
    carriedYawOffsetRad,
    grabCurrentWorldOnBlock,
    grabWorldAnchor,
    carriedLandingPos,
    carriedTargetQuat,
    topSupportSurfaceCenter,
    topSupportUpNormal,
    topSupportDefaultRot,
    topSupportTargetLayer,
    blocks,
    shards,
    splinters,
  };
}

interface ConvexPolyFace {
  verts: Vec3[];
  normal: Vec3;
  isOuter: boolean;
}

function crossVec3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normVec3(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1.0;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function polyArea(verts: Vec3[]): number {
  if (verts.length < 3) return 0;
  let area = 0;
  const v0 = verts[0];
  for (let i = 1; i < verts.length - 1; i++) {
    const e1: Vec3 = [verts[i][0] - v0[0], verts[i][1] - v0[1], verts[i][2] - v0[2]];
    const e2: Vec3 = [verts[i + 1][0] - v0[0], verts[i + 1][1] - v0[1], verts[i + 1][2] - v0[2]];
    const c = crossVec3(e1, e2);
    area += 0.5 * Math.hypot(c[0], c[1], c[2]);
  }
  return area;
}

function clipConvexFaces(
  faces: ConvexPolyFace[],
  planeNormal: Vec3,
  planeD: number,
  capIsOuter: boolean
): ConvexPolyFace[] {
  const nextFaces: ConvexPolyFace[] = [];
  const capPoints: Vec3[] = [];
  const eps = 1e-5;

  const addCap = (p: Vec3) => {
    for (let i = 0; i < capPoints.length; i++) {
      const q = capPoints[i];
      if (
        Math.abs(p[0] - q[0]) < 1e-4 &&
        Math.abs(p[1] - q[1]) < 1e-4 &&
        Math.abs(p[2] - q[2]) < 1e-4
      ) {
        return;
      }
    }
    capPoints.push(p);
  };

  for (const face of faces) {
    const verts = face.verts;
    const n = verts.length;
    if (n < 3) continue;
    const dists = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = verts[i];
      dists[i] =
        planeNormal[0] * v[0] + planeNormal[1] * v[1] + planeNormal[2] * v[2] - planeD;
    }

    const clipped: Vec3[] = [];
    for (let i = 0; i < n; i++) {
      const curr = verts[i];
      const next = verts[(i + 1) % n];
      const dCurr = dists[i];
      const dNext = dists[(i + 1) % n];
      const currIn = dCurr <= eps;
      const nextIn = dNext <= eps;

      if (currIn) {
        clipped.push(curr);
        if (Math.abs(dCurr) <= eps) addCap(curr);
      }
      if (currIn !== nextIn) {
        const t = dCurr / (dCurr - dNext);
        const inter: Vec3 = [
          curr[0] + (next[0] - curr[0]) * t,
          curr[1] + (next[1] - curr[1]) * t,
          curr[2] + (next[2] - curr[2]) * t,
        ];
        clipped.push(inter);
        addCap(inter);
      }
    }

    const clean: Vec3[] = [];
    for (const v of clipped) {
      if (
        clean.length === 0 ||
        Math.hypot(
          v[0] - clean[clean.length - 1][0],
          v[1] - clean[clean.length - 1][1],
          v[2] - clean[clean.length - 1][2]
        ) > 1e-4
      ) {
        clean.push(v);
      }
    }
    if (
      clean.length >= 3 &&
      Math.hypot(
        clean[0][0] - clean[clean.length - 1][0],
        clean[0][1] - clean[clean.length - 1][1],
        clean[0][2] - clean[clean.length - 1][2]
      ) <= 1e-4
    ) {
      clean.pop();
    }
    if (clean.length >= 3) {
      nextFaces.push({ verts: clean, normal: face.normal, isOuter: face.isOuter });
    }
  }

  if (capPoints.length >= 3) {
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const p of capPoints) {
      cx += p[0];
      cy += p[1];
      cz += p[2];
    }
    cx /= capPoints.length;
    cy /= capPoints.length;
    cz /= capPoints.length;

    const ref: Vec3 = Math.abs(planeNormal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = normVec3(crossVec3(ref, planeNormal));
    const v = crossVec3(planeNormal, u);
    const sorted = capPoints
      .map((p) => {
        const rx = p[0] - cx;
        const ry = p[1] - cy;
        const rz = p[2] - cz;
        return {
          p,
          ang: Math.atan2(rx * v[0] + ry * v[1] + rz * v[2], rx * u[0] + ry * u[1] + rz * u[2]),
        };
      })
      .sort((a, b) => a.ang - b.ang)
      .map((x) => x.p);

    nextFaces.push({
      verts: sorted,
      normal: [planeNormal[0], planeNormal[1], planeNormal[2]],
      isOuter: capIsOuter,
    });
  }

  return nextFaces;
}

/**
 * Reconstructs the exact 40-byte-per-vertex flat-shaded Voronoi convex shard GPU mesh
 * (`[cx,cy,cz, nx,ny,nz, bx,by,bz, isOuter]`) on the Guest client from the compact
 * half-space plane descriptor (`~110 B` over `"ctrl"` vs `~5.2 KB` raw triangle buffer).
 */
export function reconstructShardVerticesFromMeta(meta: SerializedShardMeta): {
  renderVertices: Float32Array;
  vertexCount: number;
  planesFloat32: Float32Array;
} {
  const lc = meta.localCentroid;
  const xMin = -BLOCK_HALF_X - lc[0];
  const xMax = BLOCK_HALF_X - lc[0];
  const yMin = -BLOCK_HALF_Y - lc[1];
  const yMax = BLOCK_HALF_Y - lc[1];
  const zMin = -BLOCK_HALF_Z - lc[2];
  const zMax = BLOCK_HALF_Z - lc[2];

  let faces: ConvexPolyFace[] = [
    {
      verts: [
        [xMin, yMin, zMax],
        [xMax, yMin, zMax],
        [xMax, yMax, zMax],
        [xMin, yMax, zMax],
      ],
      normal: [0, 0, 1],
      isOuter: true,
    },
    {
      verts: [
        [xMax, yMin, zMin],
        [xMin, yMin, zMin],
        [xMin, yMax, zMin],
        [xMax, yMax, zMin],
      ],
      normal: [0, 0, -1],
      isOuter: true,
    },
    {
      verts: [
        [xMax, yMin, zMax],
        [xMax, yMin, zMin],
        [xMax, yMax, zMin],
        [xMax, yMax, zMax],
      ],
      normal: [1, 0, 0],
      isOuter: true,
    },
    {
      verts: [
        [xMin, yMin, zMin],
        [xMin, yMin, zMax],
        [xMin, yMax, zMax],
        [xMin, yMax, zMin],
      ],
      normal: [-1, 0, 0],
      isOuter: true,
    },
    {
      verts: [
        [xMin, yMax, zMax],
        [xMax, yMax, zMax],
        [xMax, yMax, zMin],
        [xMin, yMax, zMin],
      ],
      normal: [0, 1, 0],
      isOuter: true,
    },
    {
      verts: [
        [xMin, yMin, zMin],
        [xMax, yMin, zMin],
        [xMax, yMin, zMax],
        [xMin, yMin, zMax],
      ],
      normal: [0, -1, 0],
      isOuter: true,
    },
  ];

  const planesFloat32 = new Float32Array(48);
  const count = Math.min(12, meta.planeCount, Math.floor(meta.planes.length / 4));
  for (let i = 0; i < count; i++) {
    const nx = meta.planes[i * 4 + 0];
    const ny = meta.planes[i * 4 + 1];
    const nz = meta.planes[i * 4 + 2];
    const signedD = meta.planes[i * 4 + 3];
    planesFloat32[i * 4 + 0] = nx;
    planesFloat32[i * 4 + 1] = ny;
    planesFloat32[i * 4 + 2] = nz;
    planesFloat32[i * 4 + 3] = signedD;

    // Exterior parent-block faces (`signedD < 0`) are already exact in `faces` above;
    // clip against all interior Voronoi cleavage planes (`signedD > 0`)!
    if (signedD > 0) {
      const n = normVec3([nx, ny, nz]);
      faces = clipConvexFaces(faces, n, Math.abs(signedD), false);
    }
  }

  const sortedFaces = faces
    .map((f) => ({ f, area: polyArea(f.verts) }))
    .sort((a, b) => b.area - a.area);

  let triCount = 0;
  for (const item of sortedFaces) {
    triCount += Math.max(0, item.f.verts.length - 2);
  }
  const vertexCount = Math.min(SHARD_SLOT_STRIDE_VERTS, Math.max(12, triCount * 3));
  const renderVertices = new Float32Array(vertexCount * 10);
  let ptr = 0;
  const maxFloats = vertexCount * 10;

  for (const item of sortedFaces) {
    if (ptr + 30 > maxFloats) break;
    const f = item.f;
    const fv = f.verts;
    const nx = f.normal[0];
    const ny = f.normal[1];
    const nz = f.normal[2];
    const outerFlag = f.isOuter ? 1.0 : 0.0;
    const v0 = fv[0];

    for (let t = 1; t < fv.length - 1; t++) {
      if (ptr + 30 > maxFloats) break;
      const tri = [v0, fv[t], fv[t + 1]];
      for (const c of tri) {
        renderVertices[ptr++] = c[0];
        renderVertices[ptr++] = c[1];
        renderVertices[ptr++] = c[2];
        renderVertices[ptr++] = nx;
        renderVertices[ptr++] = ny;
        renderVertices[ptr++] = nz;
        renderVertices[ptr++] = c[0] + lc[0];
        renderVertices[ptr++] = c[1] + lc[1];
        renderVertices[ptr++] = c[2] + lc[2];
        renderVertices[ptr++] = outerFlag;
      }
    }
  }

  const actualVerts = Math.floor(ptr / 10);
  return {
    renderVertices: actualVerts === vertexCount ? renderVertices : renderVertices.subarray(0, ptr),
    vertexCount: actualVerts,
    planesFloat32,
  };
}

/**
 * Infers `localCentroid` of a Voronoi shard from its `ShardMeshAppendEvent` (`bx - cx`),
 * or falls back to exterior plane distances if `renderVertices` was already flushed.
 */
export function inferShardLocalCentroid(
  planes: Float32Array,
  planeCount: number,
  impactLocalPos: Vec3,
  renderVertices?: Float32Array
): Vec3 {
  if (renderVertices && renderVertices.length >= 9) {
    return [
      renderVertices[6] - renderVertices[0],
      renderVertices[7] - renderVertices[1],
      renderVertices[8] - renderVertices[2],
    ];
  }
  const lc: Vec3 = [impactLocalPos[0] * 0.5, impactLocalPos[1] * 0.5, impactLocalPos[2] * 0.5];
  const n = Math.min(12, planeCount);
  for (let i = 0; i < n; i++) {
    const signedD = planes[i * 4 + 3];
    if (signedD >= 0) continue;
    const dist = Math.abs(signedD);
    const nx = planes[i * 4 + 0];
    const ny = planes[i * 4 + 1];
    const nz = planes[i * 4 + 2];
    if (Math.abs(nx) > 0.95) {
      lc[0] = nx > 0 ? BLOCK_HALF_X - dist : -BLOCK_HALF_X + dist;
    } else if (Math.abs(ny) > 0.95) {
      lc[1] = ny > 0 ? BLOCK_HALF_Y - dist : -BLOCK_HALF_Y + dist;
    } else if (Math.abs(nz) > 0.95) {
      lc[2] = nz > 0 ? BLOCK_HALF_Z - dist : -BLOCK_HALF_Z + dist;
    }
  }
  return lc;
}

export function computeNominalTowerBlockPose(
  layer: number,
  slot: number
): { position: Vec3; rotation: Quat } {
  const y = (layer + 0.5) * BLOCK_HEIGHT;
  const offset = (slot - 1) * 0.252;
  if (layer % 2 === 0) {
    return {
      position: [0, y, offset],
      rotation: [0, 0, 0, 1],
    };
  }
  const s = Math.SQRT1_2;
  return {
    position: [offset, y, 0],
    rotation: [0, s, 0, s],
  };
}

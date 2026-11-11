/// <reference lib="webworker" />
import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import { Quat, Vec3 } from '../math/vecMat';
import { INITIAL_LAYERS } from './constants';
import {
  BlockCriticalityCandidate,
  MujocoPhysicsEngine,
} from './MujocoPhysicsEngine';
import { ensureMujocoInitialized } from './mujocoRuntime';
import {
  GameMode,
  GlassBlockState,
  MicroSplinter,
  ShardMeshAppendEvent,
  TopPlacementSlot,
  TopSupportFrame,
  TowerTelemetry,
  TurnPhase,
} from './PhysicsWorld';

export type WorkerAudioEvent =
  | {
      kind: 'slideFriction';
      slideSpeedDmPerSec: number;
      resistance01: number;
      slipPowerMw: number;
      torsionRadPerSec: number;
    }
  | {
      kind: 'tapTest';
      pinch01: number;
      worldPos?: [number, number, number];
    }
  | {
      kind: 'glassImpact';
      intensity01: number;
      pitchMul: number;
      worldPos?: [number, number, number];
    }
  | {
      kind: 'glassCrack';
      damage01: number;
      worldPos?: [number, number, number];
    }
  | {
      kind: 'glassShatter';
      intensity01: number;
      hitTable: boolean;
      worldPos?: [number, number, number];
    }
  | {
      kind: 'rankedShards';
      events: Array<{
        energy: number;
        intensity01: number;
        volumeDm3: number;
        worldPos: [number, number, number];
        hitGlass: boolean;
      }>;
    }
  | {
      kind: 'stackSuccess';
      layerCount: number;
    };

export interface WorkerShardStaticMeta {
  id: number;
  slotIndex: number;
  swatchId: number;
  motif: 0 | 1 | 2 | 3;
  rippleSeed: number;
  firstVertex: number;
  vertexCount: number;
  planeCount: number;
  impactLocalPos: Vec3;
  planes: Float32Array;
}

export interface WorkerSnapshotPayload {
  type: 'snapshot';
  isFullRebuild: boolean;
  threadingMode: 'mt' | 'single';
  threadCount: number;
  initialBlocks?: GlassBlockState[];
  /** Packed stride = 12 floats per block: pos(3), quat(4), crackOrigin(3), damage(1), flagsBits(1) */
  blockTransforms: Float32Array;
  blockPhotoelasticStress01: Float32Array;
  blockPhotoelasticWrench: Float32Array;
  /** Packed stride = 8 floats per active shard: id(1), pos(3), quat(4) */
  shardTransforms: Float32Array;
  newShardMetas: WorkerShardStaticMeta[];
  shardUploads: ShardMeshAppendEvent[];
  splinters: MicroSplinter[];
  openTopSlots: TopPlacementSlot[];
  topSupportFrame: TopSupportFrame;
  telemetry: TowerTelemetry;
  turnPhase: TurnPhase;
  gameMode: GameMode;
  playerCount: number;
  currentPlayer: number;
  blocksStacked: number;
  shatteredCount: number;
  collapseReason: string | null;
  hoveredBlockId: number | null;
  activeBlockId: number | null;
  hoveredSlotIndex: number | null;
  grabLocalPoint: Vec3;
  grabWorldAnchor: Vec3;
  grabCurrentWorldOnBlock: Vec3;
  grabPlaneY: number;
  grabResistance01: number;
  isPointerDraggingBlock: boolean;
  isExtractionLiftReady: boolean;
  carriedLandingPos: Vec3;
  carriedTargetPos: Vec3;
  carriedTargetQuat: Quat;
  carriedYawOffsetRad: number;
  leanAngleDeg: number;
  wobbleSpeedDegPerSec: number;
  centerOfMassXZCm: [number, number];
  currentLayers: number;
  peakLayers: number;
  liveHeightCm: number;
  peakHeightCm: number;
  lockedMinRow: number | null;
  audioEvents: WorkerAudioEvent[];
}

export type WorkerInboundMessage =
  | { type: 'init'; initialLayers?: number }
  | {
      type: 'step';
      dt: number;
      camPos: Vec3;
      camRight: Vec3;
      hoveredBlockId: number | null;
    }
  | { type: 'rebuildTower'; mode?: GameMode; players?: number; layers?: number }
  | { type: 'tapNudgeBlock'; blockId: number; localNormal: Vec3; localPoint: Vec3 }
  | { type: 'startGrabBlock'; reqId: number; blockId: number; localPoint: Vec3; worldPoint: Vec3 }
  | { type: 'updateGrabTarget'; worldTargetOnPlane: Vec3 }
  | { type: 'updateCarriedHover'; rayOrigin: Vec3; rayDir: Vec3 }
  | { type: 'rotateCarriedBlock'; deltaYawRad: number }
  | { type: 'resetCarriedBlockRotation' }
  | { type: 'endGrab'; dropIfPoisedOverTower: boolean }
  | { type: 'releaseExtractedBlockOntoTop' }
  | { type: 'triggerSandboxCollapseImpulse' }
  | {
      type: 'evaluateOracle';
      reqId: number;
      maxCandidates: number;
      rolloutSteps: number;
      candidateBlockIds?: number[];
    }
  | { type: 'dispose' };

export type WorkerOutboundMessage =
  | { type: 'ready'; threadingMode: 'mt' | 'single'; threadCount: number; snapshot: WorkerSnapshotPayload }
  | WorkerSnapshotPayload
  | { type: 'grabStarted'; reqId: number; started: boolean }
  | { type: 'oracleResult'; reqId: number; candidates: BlockCriticalityCandidate[] };

class WorkerAudioProxy extends GlassAudioEngine {
  public pendingEvents: WorkerAudioEvent[] = [];
  private latestSlide: Extract<WorkerAudioEvent, { kind: 'slideFriction' }> | null = null;

  public override updateSlideFriction(
    slideSpeedDmPerSec: number,
    resistance01: number,
    slipPowerMilliwatts: number = 0,
    torsionRadPerSec: number = 0
  ): void {
    this.latestSlide = {
      kind: 'slideFriction',
      slideSpeedDmPerSec,
      resistance01,
      slipPowerMw: slipPowerMilliwatts,
      torsionRadPerSec,
    };
  }

  public override playTapTestResonance(
    pinch01: number,
    worldPos?: [number, number, number]
  ): void {
    this.pendingEvents.push({
      kind: 'tapTest',
      pinch01,
      worldPos: worldPos ? [...worldPos] : undefined,
    });
  }

  public override playGlassImpact(
    intensity01: number,
    pitchMul: number = 1.0,
    worldPos?: [number, number, number]
  ): void {
    this.pendingEvents.push({
      kind: 'glassImpact',
      intensity01,
      pitchMul,
      worldPos: worldPos ? [...worldPos] : undefined,
    });
  }

  public override playGlassCrack(
    damage01: number,
    worldPos?: [number, number, number]
  ): void {
    this.pendingEvents.push({
      kind: 'glassCrack',
      damage01,
      worldPos: worldPos ? [...worldPos] : undefined,
    });
  }

  public override playGlassShatter(
    intensity01: number,
    hitTable: boolean = false,
    worldPos?: [number, number, number]
  ): void {
    this.pendingEvents.push({
      kind: 'glassShatter',
      intensity01,
      hitTable,
      worldPos: worldPos ? [...worldPos] : undefined,
    });
  }

  public override playRankedShardCollisions(
    events: Array<{
      energy: number;
      intensity01: number;
      volumeDm3: number;
      worldPos: [number, number, number];
      hitGlass: boolean;
    }>
  ): void {
    if (events.length === 0) return;
    this.pendingEvents.push({
      kind: 'rankedShards',
      events: events.slice(0, 6),
    });
  }

  public override playStackSuccess(layerCount: number): void {
    this.pendingEvents.push({
      kind: 'stackSuccess',
      layerCount,
    });
  }

  public flush(): WorkerAudioEvent[] {
    const out = this.pendingEvents;
    this.pendingEvents = [];
    if (this.latestSlide) {
      out.push(this.latestSlide);
      this.latestSlide = null;
    }
    return out;
  }
}

let engine: MujocoPhysicsEngine | null = null;
let audioProxy: WorkerAudioProxy | null = null;
let threadingMode: 'mt' | 'single' = 'single';
let threadCount = 1;
const knownShardIds = new Set<number>();

function buildSnapshot(isFullRebuild: boolean): {
  snapshot: WorkerSnapshotPayload;
  transfer: Transferable[];
} {
  const eng = engine!;
  const audio = audioProxy!;

  if (isFullRebuild) {
    knownShardIds.clear();
  }

  const blockCount = eng.blocks.length;
  const blockTransforms = new Float32Array(blockCount * 12);
  for (let i = 0; i < blockCount; i++) {
    const b = eng.blocks[i];
    const base = i * 12;
    blockTransforms[base + 0] = b.position[0];
    blockTransforms[base + 1] = b.position[1];
    blockTransforms[base + 2] = b.position[2];
    blockTransforms[base + 3] = b.rotation[0];
    blockTransforms[base + 4] = b.rotation[1];
    blockTransforms[base + 5] = b.rotation[2];
    blockTransforms[base + 6] = b.rotation[3];
    blockTransforms[base + 7] = b.crackLocalOrigin[0];
    blockTransforms[base + 8] = b.crackLocalOrigin[1];
    blockTransforms[base + 9] = b.crackLocalOrigin[2];
    blockTransforms[base + 10] = b.damage;
    const bits =
      (b.shattered ? 1 : 0) |
      (b.extracted ? 2 : 0) |
      (b.isLegalTarget ? 4 : 0) |
      (b.isLooseSlot ? 8 : 0);
    blockTransforms[base + 11] = bits;
  }

  const stressCopy = new Float32Array(eng.blockPhotoelasticStress01);
  const wrenchCopy = new Float32Array(eng.blockPhotoelasticWrench);

  const shardCount = eng.shards.length;
  const shardTransforms = new Float32Array(shardCount * 8);
  const newShardMetas: WorkerShardStaticMeta[] = [];
  const activeIdsNow = new Set<number>();

  for (let i = 0; i < shardCount; i++) {
    const s = eng.shards[i];
    activeIdsNow.add(s.id);
    const base = i * 8;
    shardTransforms[base + 0] = s.id;
    shardTransforms[base + 1] = s.position[0];
    shardTransforms[base + 2] = s.position[1];
    shardTransforms[base + 3] = s.position[2];
    shardTransforms[base + 4] = s.rotation[0];
    shardTransforms[base + 5] = s.rotation[1];
    shardTransforms[base + 6] = s.rotation[2];
    shardTransforms[base + 7] = s.rotation[3];

    if (!knownShardIds.has(s.id)) {
      knownShardIds.add(s.id);
      newShardMetas.push({
        id: s.id,
        slotIndex: s.slotIndex,
        swatchId: s.swatchId,
        motif: s.motif,
        rippleSeed: s.rippleSeed,
        firstVertex: s.firstVertex,
        vertexCount: s.vertexCount,
        planeCount: s.planeCount,
        impactLocalPos: [...s.impactLocalPos],
        planes: new Float32Array(s.planes),
      });
    }
  }

  // Prune recycled shard IDs
  for (const oldId of knownShardIds) {
    if (!activeIdsNow.has(oldId)) {
      knownShardIds.delete(oldId);
    }
  }

  const shardUploads = eng.consumePendingShardUploads();
  const transfer: Transferable[] = [
    blockTransforms.buffer,
    stressCopy.buffer,
    wrenchCopy.buffer,
    shardTransforms.buffer,
  ];
  for (const up of shardUploads) {
    transfer.push(up.vertexData.buffer);
  }
  for (const meta of newShardMetas) {
    transfer.push(meta.planes.buffer);
  }

  const snapshot: WorkerSnapshotPayload = {
    type: 'snapshot',
    isFullRebuild,
    threadingMode,
    threadCount,
    initialBlocks: isFullRebuild ? eng.blocks.map((b) => ({ ...b })) : undefined,
    blockTransforms,
    blockPhotoelasticStress01: stressCopy,
    blockPhotoelasticWrench: wrenchCopy,
    shardTransforms,
    newShardMetas,
    shardUploads,
    splinters: eng.splinters.map((sp) => ({ ...sp, pos: [...sp.pos], vel: [...sp.vel] })),
    openTopSlots: eng.openTopSlots.map((sl) => ({
      ...sl,
      position: [...sl.position],
      rotation: [...sl.rotation],
    })),
    topSupportFrame: {
      targetLayer: eng.topSupportFrame.targetLayer,
      surfaceCenter: [...eng.topSupportFrame.surfaceCenter],
      layerCenter: [...eng.topSupportFrame.layerCenter],
      upNormal: [...eng.topSupportFrame.upNormal],
      tiltQuat: [...eng.topSupportFrame.tiltQuat],
      defaultYawRad: eng.topSupportFrame.defaultYawRad,
      defaultRotation: [...eng.topSupportFrame.defaultRotation],
      lengthAxis: [...eng.topSupportFrame.lengthAxis],
      spacingAxis: [...eng.topSupportFrame.spacingAxis],
    },
    telemetry: eng.getTelemetry(),
    turnPhase: eng.turnPhase,
    gameMode: eng.gameMode,
    playerCount: eng.playerCount,
    currentPlayer: eng.currentPlayer,
    blocksStacked: eng.blocksStacked,
    shatteredCount: eng.shatteredCount,
    collapseReason: eng.collapseReason,
    hoveredBlockId: eng.hoveredBlockId,
    activeBlockId: eng.activeBlockId,
    hoveredSlotIndex: eng.hoveredSlotIndex,
    grabLocalPoint: [...eng.grabLocalPoint],
    grabWorldAnchor: [...eng.grabWorldAnchor],
    grabCurrentWorldOnBlock: [...eng.grabCurrentWorldOnBlock],
    grabPlaneY: eng.grabPlaneY,
    grabResistance01: eng.grabResistance01,
    isPointerDraggingBlock: eng.isPointerDraggingBlock,
    isExtractionLiftReady: eng.isExtractionLiftReady,
    carriedLandingPos: [...eng.carriedLandingPos],
    carriedTargetPos: [...eng.carriedTargetPos],
    carriedTargetQuat: [...eng.carriedTargetQuat],
    carriedYawOffsetRad: eng.carriedYawOffsetRad,
    leanAngleDeg: eng.leanAngleDeg,
    wobbleSpeedDegPerSec: eng.wobbleSpeedDegPerSec,
    centerOfMassXZCm: [...eng.centerOfMassXZCm],
    currentLayers: eng.currentLayers,
    peakLayers: eng.peakLayers,
    liveHeightCm: eng.liveHeightCm,
    peakHeightCm: eng.peakHeightCm,
    lockedMinRow: eng.lockedMinRow,
    audioEvents: audio.flush(),
  };

  return { snapshot, transfer };
}

function postSnapshot(isFullRebuild: boolean): void {
  if (!engine) return;
  const { snapshot, transfer } = buildSnapshot(isFullRebuild);
  self.postMessage(snapshot, transfer);
}

self.onmessage = async (ev: MessageEvent<WorkerInboundMessage>) => {
  const msg = ev.data;
  if (msg.type === 'init') {
    const rt = await ensureMujocoInitialized(true);
    threadingMode = rt.mode;
    threadCount = rt.threadCount;
    audioProxy = new WorkerAudioProxy();
    engine = new MujocoPhysicsEngine(audioProxy, msg.initialLayers ?? INITIAL_LAYERS);
    const { snapshot, transfer } = buildSnapshot(true);
    self.postMessage(
      {
        type: 'ready',
        threadingMode,
        threadCount,
        snapshot,
      } satisfies WorkerOutboundMessage,
      transfer
    );
    return;
  }

  if (!engine) return;

  switch (msg.type) {
    case 'step': {
      engine.setListenerPose(msg.camPos, msg.camRight);
      if (engine.turnPhase === 'SELECT_BLOCK' || engine.gameMode === 'sandbox') {
        engine.hoveredBlockId = msg.hoveredBlockId;
      }
      engine.step(msg.dt);
      postSnapshot(false);
      break;
    }
    case 'rebuildTower': {
      engine.rebuildTower(msg.mode, msg.players, msg.layers);
      postSnapshot(true);
      break;
    }
    case 'tapNudgeBlock': {
      engine.tapNudgeBlock(msg.blockId, msg.localNormal, msg.localPoint);
      postSnapshot(false);
      break;
    }
    case 'startGrabBlock': {
      const started = engine.startGrabBlock(msg.blockId, msg.localPoint, msg.worldPoint);
      self.postMessage({ type: 'grabStarted', reqId: msg.reqId, started } satisfies WorkerOutboundMessage);
      postSnapshot(false);
      break;
    }
    case 'updateGrabTarget': {
      engine.updateGrabTarget(msg.worldTargetOnPlane);
      break;
    }
    case 'updateCarriedHover': {
      engine.updateCarriedHover(msg.rayOrigin, msg.rayDir);
      break;
    }
    case 'rotateCarriedBlock': {
      engine.rotateCarriedBlock(msg.deltaYawRad);
      postSnapshot(false);
      break;
    }
    case 'resetCarriedBlockRotation': {
      engine.resetCarriedBlockRotation();
      postSnapshot(false);
      break;
    }
    case 'endGrab': {
      engine.endGrab(msg.dropIfPoisedOverTower);
      postSnapshot(false);
      break;
    }
    case 'releaseExtractedBlockOntoTop': {
      engine.releaseExtractedBlockOntoTop();
      postSnapshot(false);
      break;
    }
    case 'triggerSandboxCollapseImpulse': {
      engine.triggerSandboxCollapseImpulse();
      postSnapshot(false);
      break;
    }
    case 'evaluateOracle': {
      const candidates = engine.evaluateCounterfactualBlockCriticality(
        msg.maxCandidates,
        msg.rolloutSteps,
        msg.candidateBlockIds
      );
      self.postMessage({
        type: 'oracleResult',
        reqId: msg.reqId,
        candidates,
      } satisfies WorkerOutboundMessage);
      break;
    }
    case 'dispose': {
      engine.dispose();
      engine = null;
      self.close();
      break;
    }
  }
};

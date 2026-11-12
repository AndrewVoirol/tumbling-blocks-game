/// <reference lib="webworker" />
import { GlassAudioEngine, type GlassImpactOptions } from '../audio/GlassAudioEngine';
import { Quat, Vec3 } from '../math/vecMat';
import { INITIAL_LAYERS } from './constants';
import {
  type AdversarialPlacementPlan,
  type BlockCriticalityCandidate,
  MujocoPhysicsEngine,
  type SceneRaycastHit,
  type SceneRaycastOptions,
} from './MujocoPhysicsEngine';
import { ensureMujocoInitialized } from './mujocoRuntime';
import {
  GameMode,
  GlassBlockState,
  MicroSplinter,
  PhysicsNetworkSnapshot,
  ShardImpactEvent,
  ShardMeshAppendEvent,
  TopPlacementSlot,
  TopSupportFrame,
  TowerTelemetry,
  TurnPhase,
  WobbleRockEvent,
} from './PhysicsWorld';

export type WorkerAudioEvent =
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
      worldPos?: [number, number, number];
      damage?: number;
    }
  | {
      kind: 'glassImpact';
      intensity01: number;
      pitchMul: number;
      worldPos?: [number, number, number];
      options?: GlassImpactOptions;
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
  volume: number;
  massKg: number;
  spawnSimTime: number;
  firstVertex: number;
  vertexCount: number;
  planeCount: number;
  impactLocalPos: Vec3;
  planes: Float32Array;
}

export interface WorkerSnapshotPayload {
  type: 'snapshot';
  epoch: number;
  ackSeq: number;
  isFullRebuild: boolean;
  threadingMode: 'mt' | 'single';
  threadCount: number;
  simTime: number;
  towerSeed: number;
  initialLayers: number;
  initialBlocks?: GlassBlockState[];
  /** Packed stride = 18 floats per block: pos(3), quat(4), crackOrigin(3), damage(1), packedIntBits(1), linVel(3), angVel(3) */
  blockTransforms: Float32Array;
  blockPhotoelasticStress01: Float32Array;
  blockPhotoelasticWrench: Float32Array;
  /** Packed stride = 14 floats per active shard: id(1), pos(3), quat(4), linVel(3), angVel(3) */
  shardTransforms: Float32Array;
  newShardMetas: WorkerShardStaticMeta[];
  shardUploads: ShardMeshAppendEvent[];
  shardImpacts: ShardImpactEvent[];
  wobbleEvents: WobbleRockEvent[];
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
  solverWarningCount: number;
  mechanicalEnergyJ: number;
  audioEvents: WorkerAudioEvent[];
}

export type WorkerInboundMessage =
  | { type: 'init'; epoch: number; seq?: number; initialLayers?: number; seed?: number }
  | {
      type: 'step';
      epoch: number;
      seq?: number;
      dt: number;
      camPos: Vec3;
      camRight: Vec3;
      hoveredBlockId: number | null;
    }
  | {
      type: 'rebuildTower';
      epoch: number;
      seq?: number;
      mode?: GameMode;
      players?: number;
      layers?: number;
      seed?: number;
    }
  | { type: 'resetTower'; epoch: number; seq?: number; layers?: number; seed?: number }
  | { type: 'applyCompactSnapshot'; epoch: number; seq?: number; snapshot: PhysicsNetworkSnapshot }
  | {
      type: 'tapNudgeBlock';
      epoch: number;
      seq?: number;
      blockId: number;
      localNormal: Vec3;
      localPoint: Vec3;
    }
  | {
      type: 'startGrabBlock';
      epoch: number;
      seq?: number;
      reqId: number;
      blockId: number;
      localPoint: Vec3;
      worldPoint: Vec3;
    }
  | { type: 'updateGrabTarget'; epoch: number; seq?: number; worldTargetOnPlane: Vec3 }
  | { type: 'updateCarriedHover'; epoch: number; seq?: number; rayOrigin: Vec3; rayDir: Vec3 }
  | { type: 'rotateCarriedBlock'; epoch: number; seq?: number; deltaYawRad: number }
  | { type: 'resetCarriedBlockRotation'; epoch: number; seq?: number }
  | { type: 'endGrab'; epoch: number; seq?: number; dropIfPoisedOverTower: boolean }
  | { type: 'releaseExtractedBlockOntoTop'; epoch: number; seq?: number }
  | { type: 'triggerSandboxCollapseImpulse'; epoch: number; seq?: number }
  | {
      type: 'nudgeShard';
      epoch: number;
      seq?: number;
      shardId: number;
      worldPoint: Vec3;
      impulse: Vec3;
    }
  | {
      type: 'evaluateOracle';
      epoch: number;
      seq?: number;
      reqId: number;
      maxCandidates: number;
      rolloutSteps: number;
      candidateBlockIds?: number[];
    }
  | {
      type: 'evaluateAdversarialPlacement';
      epoch: number;
      seq?: number;
      reqId: number;
      blockId: number;
      rolloutSteps?: number;
      assumeAlreadyExtracted?: boolean;
    }
  | {
      type: 'raycastScene';
      epoch: number;
      seq?: number;
      reqId: number;
      rayOriginDm: Vec3;
      rayDir: Vec3;
      options?: SceneRaycastOptions;
    }
  | { type: 'dispose' };

export type WorkerOutboundMessage =
  | {
      type: 'ready';
      epoch: number;
      threadingMode: 'mt' | 'single';
      threadCount: number;
      snapshot: WorkerSnapshotPayload;
    }
  | WorkerSnapshotPayload
  | { type: 'grabStarted'; epoch: number; reqId: number; seq: number; started: boolean }
  | { type: 'oracleResult'; epoch: number; reqId: number; candidates: BlockCriticalityCandidate[] }
  | {
      type: 'placementResult';
      epoch: number;
      reqId: number;
      plans: AdversarialPlacementPlan[];
    }
  | { type: 'raycastResult'; epoch: number; reqId: number; hit: SceneRaycastHit | null };

export interface WorkerHostScope {
  onmessage: ((ev: MessageEvent<WorkerInboundMessage>) => void) | null;
  postMessage: (message: WorkerOutboundMessage, transfer?: Transferable[]) => void;
  close?: () => void;
}

class WorkerAudioProxy extends GlassAudioEngine {
  public pendingEvents: WorkerAudioEvent[] = [];
  private latestSlide: Extract<WorkerAudioEvent, { kind: 'slideFriction' }> | null = null;

  public override updateSlideFriction(
    slideSpeedDmPerSec: number,
    resistance01: number,
    slipPowerMilliwatts: number = 0,
    torsionRadPerSec: number = 0,
    clampingLoadN?: number
  ): void {
    this.latestSlide = {
      kind: 'slideFriction',
      slideSpeedDmPerSec,
      resistance01,
      slipPowerMw: slipPowerMilliwatts,
      torsionRadPerSec,
      clampingLoadN,
    };
  }

  public override playTapTestResonance(
    pinch01: number,
    worldPos?: [number, number, number],
    _camPos?: [number, number, number],
    _camRight?: [number, number, number],
    damage: number = 0
  ): void {
    this.pendingEvents.push({
      kind: 'tapTest',
      pinch01,
      worldPos: worldPos ? [...worldPos] : undefined,
      damage,
    });
  }

  public override playGlassImpact(
    intensity01: number,
    pitchMul: number = 1.0,
    worldPos?: [number, number, number],
    _camPos?: [number, number, number],
    _camRight?: [number, number, number],
    options?: GlassImpactOptions
  ): void {
    this.pendingEvents.push({
      kind: 'glassImpact',
      intensity01,
      pitchMul,
      worldPos: worldPos ? [...worldPos] : undefined,
      options: options ? { ...options } : undefined,
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

/**
 * Attaches an isolated MuJoCo 4.0 worker command loop with monotonic `epoch` fencing
 * to either a dedicated `DedicatedWorkerGlobalScope` or an in-process `MessagePort` adapter.
 */
export function attachMujocoWorkerHost(scope: WorkerHostScope): { dispose: () => void } {
  let engine: MujocoPhysicsEngine | null = null;
  let audioProxy: WorkerAudioProxy | null = null;
  let threadingMode: 'mt' | 'single' = 'single';
  let threadCount = 1;
  let workerEpoch = 0;
  let lastProcessedSeq = 0;
  let disposed = false;
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
    const blockTransforms = new Float32Array(blockCount * 18);
    for (let i = 0; i < blockCount; i++) {
      const b = eng.blocks[i];
      const base = i * 18;
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
      blockTransforms[base + 11] = bits;
      blockTransforms[base + 12] = b.linVel[0];
      blockTransforms[base + 13] = b.linVel[1];
      blockTransforms[base + 14] = b.linVel[2];
      blockTransforms[base + 15] = b.angVel[0];
      blockTransforms[base + 16] = b.angVel[1];
      blockTransforms[base + 17] = b.angVel[2];
    }

    const stressCopy = new Float32Array(eng.blockPhotoelasticStress01);
    const wrenchCopy = new Float32Array(eng.blockPhotoelasticWrench);

    const shardCount = eng.shards.length;
    const shardTransforms = new Float32Array(shardCount * 14);
    const newShardMetas: WorkerShardStaticMeta[] = [];
    const activeIdsNow = new Set<number>();

    for (let i = 0; i < shardCount; i++) {
      const s = eng.shards[i];
      activeIdsNow.add(s.id);
      const base = i * 14;
      shardTransforms[base + 0] = s.id;
      shardTransforms[base + 1] = s.position[0];
      shardTransforms[base + 2] = s.position[1];
      shardTransforms[base + 3] = s.position[2];
      shardTransforms[base + 4] = s.rotation[0];
      shardTransforms[base + 5] = s.rotation[1];
      shardTransforms[base + 6] = s.rotation[2];
      shardTransforms[base + 7] = s.rotation[3];
      shardTransforms[base + 8] = s.preLinVel[0];
      shardTransforms[base + 9] = s.preLinVel[1];
      shardTransforms[base + 10] = s.preLinVel[2];
      shardTransforms[base + 11] = s.preAngVel[0];
      shardTransforms[base + 12] = s.preAngVel[1];
      shardTransforms[base + 13] = s.preAngVel[2];

      if (!knownShardIds.has(s.id)) {
        knownShardIds.add(s.id);
        newShardMetas.push({
          id: s.id,
          slotIndex: s.slotIndex,
          swatchId: s.swatchId,
          motif: s.motif,
          rippleSeed: s.rippleSeed,
          volume: s.volume,
          massKg: s.massKg,
          spawnSimTime: s.spawnSimTime ?? eng.getSimTime(),
          firstVertex: s.firstVertex,
          vertexCount: s.vertexCount,
          planeCount: s.planeCount,
          impactLocalPos: [...s.impactLocalPos],
          planes: new Float32Array(s.planes),
        });
      }
    }

    for (const oldId of knownShardIds) {
      if (!activeIdsNow.has(oldId)) {
        knownShardIds.delete(oldId);
      }
    }

    // Clone vertexData buffers so Transferable ownership never detaches engine.slotRenderVertices inside the worker
    const rawShardUploads = eng.consumePendingShardUploads();
    const shardUploads: ShardMeshAppendEvent[] = new Array(rawShardUploads.length);
    for (let i = 0; i < rawShardUploads.length; i++) {
      const up = rawShardUploads[i];
      shardUploads[i] = {
        firstVertex: up.firstVertex,
        vertexData: new Float32Array(up.vertexData),
      };
    }
    const shardImpacts = eng.consumeShardImpacts();
    const wobbleEvents = eng.consumeWobbleEvents();

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
      epoch: workerEpoch,
      ackSeq: lastProcessedSeq,
      isFullRebuild,
      threadingMode,
      threadCount,
      simTime: eng.getSimTime(),
      towerSeed: eng.towerSeed,
      initialLayers: eng.getInitialLayers(),
      initialBlocks: isFullRebuild ? eng.blocks.map((b) => ({ ...b })) : undefined,
      blockTransforms,
      blockPhotoelasticStress01: stressCopy,
      blockPhotoelasticWrench: wrenchCopy,
      shardTransforms,
      newShardMetas,
      shardUploads,
      shardImpacts,
      wobbleEvents,
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
      solverWarningCount: eng.getSolverWarningCount(),
      mechanicalEnergyJ: eng.computeMechanicalEnergyJ(),
      audioEvents: audio.flush(),
    };

    return { snapshot, transfer };
  }

  function postSnapshot(isFullRebuild: boolean): void {
    if (!engine || disposed) return;
    const { snapshot, transfer } = buildSnapshot(isFullRebuild);
    scope.postMessage(snapshot, transfer);
  }

  // Serialize async `init` against subsequent messages so `MessagePort` queues never race WASM boot
  let messageChain: Promise<void> = Promise.resolve();

  const handleMessage = async (msg: WorkerInboundMessage): Promise<void> => {
    if (disposed) return;

    if (msg.type === 'dispose') {
      disposed = true;
      if (engine) {
        engine.dispose();
        engine = null;
      }
      scope.close?.();
      return;
    }

    if (msg.type === 'init') {
      if (msg.epoch < workerEpoch) return;
      workerEpoch = msg.epoch;
      if (msg.seq !== undefined && msg.seq > lastProcessedSeq) {
        lastProcessedSeq = msg.seq;
      }
      const rt = await ensureMujocoInitialized(true);
      if (disposed) return;
      threadingMode = rt.mode;
      threadCount = rt.threadCount;
      audioProxy = new WorkerAudioProxy();
      if (engine) {
        engine.dispose();
      }
      engine = new MujocoPhysicsEngine(
        audioProxy,
        msg.initialLayers ?? INITIAL_LAYERS,
        undefined,
        msg.seed
      );
      const { snapshot, transfer } = buildSnapshot(true);
      scope.postMessage(
        {
          type: 'ready',
          epoch: workerEpoch,
          threadingMode,
          threadCount,
          snapshot,
        } satisfies WorkerOutboundMessage,
        transfer
      );
      return;
    }

    if (!engine) return;

    // Epoch-advancing reset barriers update `workerEpoch`; all other commands must match `workerEpoch` exactly
    if (
      msg.type === 'rebuildTower' ||
      msg.type === 'resetTower' ||
      msg.type === 'applyCompactSnapshot'
    ) {
      if (msg.epoch < workerEpoch) return;
      workerEpoch = msg.epoch;
    } else if (msg.epoch !== workerEpoch) {
      return;
    }

    if (msg.seq !== undefined && msg.seq > lastProcessedSeq) {
      lastProcessedSeq = msg.seq;
    }

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
        audioProxy?.flush();
        engine.rebuildTower(msg.mode, msg.players, msg.layers, msg.seed);
        postSnapshot(true);
        break;
      }
      case 'resetTower': {
        audioProxy?.flush();
        engine.resetTower(msg.layers, msg.seed);
        postSnapshot(true);
        break;
      }
      case 'applyCompactSnapshot': {
        audioProxy?.flush();
        knownShardIds.clear();
        engine.applyCompactSnapshot(msg.snapshot);
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
        scope.postMessage({
          type: 'grabStarted',
          epoch: workerEpoch,
          reqId: msg.reqId,
          seq: lastProcessedSeq,
          started,
        } satisfies WorkerOutboundMessage);
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
      case 'nudgeShard': {
        engine.nudgeShard(msg.shardId, msg.worldPoint, msg.impulse);
        postSnapshot(false);
        break;
      }
      case 'evaluateOracle': {
        const candidates = engine.evaluateCounterfactualBlockCriticality(
          msg.maxCandidates,
          msg.rolloutSteps,
          msg.candidateBlockIds
        );
        scope.postMessage({
          type: 'oracleResult',
          epoch: workerEpoch,
          reqId: msg.reqId,
          candidates,
        } satisfies WorkerOutboundMessage);
        break;
      }
      case 'evaluateAdversarialPlacement': {
        const plans = engine.evaluateAdversarialTopPlacement(
          msg.blockId,
          msg.rolloutSteps,
          msg.assumeAlreadyExtracted
        );
        scope.postMessage({
          type: 'placementResult',
          epoch: workerEpoch,
          reqId: msg.reqId,
          plans,
        } satisfies WorkerOutboundMessage);
        break;
      }
      case 'raycastScene': {
        const hit = engine.raycastScene(msg.rayOriginDm, msg.rayDir, msg.options);
        scope.postMessage({
          type: 'raycastResult',
          epoch: workerEpoch,
          reqId: msg.reqId,
          hit,
        } satisfies WorkerOutboundMessage);
        break;
      }
    }
  };

  scope.onmessage = (ev: MessageEvent<WorkerInboundMessage>) => {
    const data = ev.data;
    messageChain = messageChain.then(() => handleMessage(data));
  };

  return {
    dispose: () => {
      disposed = true;
      scope.onmessage = null;
      if (engine) {
        engine.dispose();
        engine = null;
      }
    },
  };
}

const globalWorkerScope = globalThis as unknown as {
  WorkerGlobalScope?: unknown;
  onmessage?: ((ev: MessageEvent<WorkerInboundMessage>) => void) | null;
  postMessage?: (message: WorkerOutboundMessage, transfer?: Transferable[]) => void;
  close?: () => void;
};

if (
  typeof globalWorkerScope.WorkerGlobalScope !== 'undefined' &&
  typeof globalWorkerScope.postMessage === 'function'
) {
  attachMujocoWorkerHost(globalWorkerScope as unknown as WorkerHostScope);
}

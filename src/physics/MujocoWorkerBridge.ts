import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import { Quat, Vec3 } from '../math/vecMat';
import { BLOCK_HALF_Y, BLOCK_HEIGHT, INITIAL_LAYERS } from './constants';
import { BlockCriticalityCandidate } from './MujocoPhysicsEngine';
import {
  WorkerAudioEvent,
  WorkerInboundMessage,
  WorkerOutboundMessage,
  WorkerShardStaticMeta,
  WorkerSnapshotPayload,
} from './mujocoWorker';
import {
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

/**
 * Main-thread Non-Blocking Physics Mirror (`MujocoWorkerBridge`)
 *
 * Runs Google DeepMind's `@mujoco/mujoco@4.0.0` (`/mt` multi-threaded WASM island threadpool)
 * inside a dedicated ES Module Web Worker so main-thread WebGPU `requestAnimationFrame`
 * submits draw calls at `0.0 ms` solver stall on desktop and iPad Pro.
 */
export class MujocoWorkerBridge {
  public readonly isWorkerBacked = true;
  public workerThreadingMode: 'mt' | 'single' = 'single';
  public workerThreadCount: number = 1;

  public blocks: GlassBlockState[] = [];
  public shards: GlassShardState[] = [];
  public splinters: MicroSplinter[] = [];
  public openTopSlots: TopPlacementSlot[] = [];
  public blockPhotoelasticStress01: Float32Array = new Float32Array(0);
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
  public hoveredSlotIndex: number | null = null;

  public grabLocalPoint: Vec3 = [0, 0, 0];
  public grabWorldAnchor: Vec3 = [0, 0, 0];
  public grabCurrentWorldOnBlock: Vec3 = [0, 0, 0];
  public grabPlaneY: number = 0;
  public grabResistance01: number = 0;
  public isPointerDraggingBlock: boolean = false;
  public isExtractionLiftReady: boolean = false;

  public listenerCamPos: Vec3 = [0, 2.2, 5.2];
  public listenerCamRight: Vec3 = [1, 0, 0];

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

  private worker: Worker;
  private audio: GlassAudioEngine;
  private shardMetaById = new Map<number, WorkerShardStaticMeta>();
  private pendingShardUploads: ShardMeshAppendEvent[] = [];
  private latestTelemetry: TowerTelemetry | null = null;
  private stepInFlight: boolean = false;
  private accumDtWhileInFlight: number = 0;
  private nextReqId: number = 1;
  private oracleCallbacks = new Map<number, (res: BlockCriticalityCandidate[]) => void>();
  private disposed: boolean = false;

  private constructor(worker: Worker, audio: GlassAudioEngine) {
    this.worker = worker;
    this.audio = audio;
    this.worker.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
      this.handleWorkerMessage(ev.data);
    };
  }

  public static async create(
    audio: GlassAudioEngine,
    initialLayers: number = INITIAL_LAYERS
  ): Promise<MujocoWorkerBridge> {
    const worker = new Worker(new URL('./mujocoWorker.ts', import.meta.url), {
      type: 'module',
    });
    const bridge = new MujocoWorkerBridge(worker, audio);

    await new Promise<void>((resolve, reject) => {
      const onError = (err: ErrorEvent) => {
        worker.removeEventListener('error', onError);
        reject(new Error(err.message || 'MuJoCo physics worker failed to initialize'));
      };
      worker.addEventListener('error', onError, { once: true });

      const prevHandler = worker.onmessage;
      worker.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
        const msg = ev.data;
        if (msg.type === 'ready') {
          worker.removeEventListener('error', onError);
          bridge.workerThreadingMode = msg.threadingMode;
          bridge.workerThreadCount = msg.threadCount;
          bridge.applySnapshot(msg.snapshot);
          worker.onmessage = prevHandler;
          resolve();
        } else if (prevHandler) {
          prevHandler.call(worker, ev);
        }
      };

      const initMsg: WorkerInboundMessage = { type: 'init', initialLayers };
      worker.postMessage(initMsg);
    });

    return bridge;
  }

  private post(msg: WorkerInboundMessage): void {
    if (this.disposed) return;
    this.worker.postMessage(msg);
  }

  private handleWorkerMessage(msg: WorkerOutboundMessage): void {
    if (this.disposed) return;
    if (msg.type === 'snapshot') {
      this.stepInFlight = false;
      this.applySnapshot(msg);
      return;
    }
    if (msg.type === 'oracleResult') {
      const cb = this.oracleCallbacks.get(msg.reqId);
      if (cb) {
        this.oracleCallbacks.delete(msg.reqId);
        cb(msg.candidates);
      }
      return;
    }
    if (msg.type === 'grabStarted' && !msg.started) {
      this.isPointerDraggingBlock = false;
      this.activeBlockId = null;
    }
  }

  private applySnapshot(snap: WorkerSnapshotPayload): void {
    this.workerThreadingMode = snap.threadingMode;
    this.workerThreadCount = snap.threadCount;

    if (snap.isFullRebuild && snap.initialBlocks) {
      this.blocks = snap.initialBlocks.map((b) => ({
        ...b,
        position: [...b.position],
        rotation: [...b.rotation],
        prePos: [...b.prePos],
        preQuat: [...b.preQuat],
        linVel: [...b.linVel],
        angVel: [...b.angVel],
        crackLocalOrigin: [...b.crackLocalOrigin],
      }));
      this.shardMetaById.clear();
      this.shards = [];
    }

    const bt = snap.blockTransforms;
    const count = Math.min(this.blocks.length, Math.floor(bt.length / 12));
    for (let i = 0; i < count; i++) {
      const b = this.blocks[i];
      const base = i * 12;
      b.position[0] = bt[base + 0];
      b.position[1] = bt[base + 1];
      b.position[2] = bt[base + 2];
      b.rotation[0] = bt[base + 3];
      b.rotation[1] = bt[base + 4];
      b.rotation[2] = bt[base + 5];
      b.rotation[3] = bt[base + 6];
      b.crackLocalOrigin[0] = bt[base + 7];
      b.crackLocalOrigin[1] = bt[base + 8];
      b.crackLocalOrigin[2] = bt[base + 9];
      b.damage = bt[base + 10];
      const bits = bt[base + 11] | 0;
      b.shattered = (bits & 1) !== 0;
      b.extracted = (bits & 2) !== 0;
      b.isLegalTarget = (bits & 4) !== 0;
      b.isLooseSlot = (bits & 8) !== 0;
    }

    this.blockPhotoelasticStress01 = snap.blockPhotoelasticStress01;
    this.blockPhotoelasticWrench = snap.blockPhotoelasticWrench;

    for (const meta of snap.newShardMetas) {
      this.shardMetaById.set(meta.id, meta);
    }

    for (const up of snap.shardUploads) {
      this.pendingShardUploads.push(up);
    }

    const st = snap.shardTransforms;
    const shardCount = Math.floor(st.length / 8);
    const nextShards: GlassShardState[] = [];
    for (let i = 0; i < shardCount; i++) {
      const base = i * 8;
      const id = st[base + 0] | 0;
      const meta = this.shardMetaById.get(id);
      if (!meta) continue;
      const pos: Vec3 = [st[base + 1], st[base + 2], st[base + 3]];
      const rot: Quat = [st[base + 4], st[base + 5], st[base + 6], st[base + 7]];
      nextShards.push({
        id,
        slotIndex: meta.slotIndex,
        bodyHandle: 0,
        colliderHandle: 0,
        swatchId: meta.swatchId,
        motif: meta.motif,
        rippleSeed: meta.rippleSeed,
        position: pos,
        rotation: rot,
        prePos: pos,
        preQuat: rot,
        preLinVel: [0, 0, 0],
        preAngVel: [0, 0, 0],
        volume: 0.001,
        massKg: 0.0025,
        preVy: 0,
        impactLocalPos: meta.impactLocalPos,
        planes: meta.planes,
        planeCount: meta.planeCount,
        firstVertex: meta.firstVertex,
        vertexCount: meta.vertexCount,
        quietSubsteps: 0,
      });
    }
    this.shards = nextShards;

    this.splinters = snap.splinters;
    this.openTopSlots = snap.openTopSlots;
    this.topSupportFrame = snap.topSupportFrame;
    this.latestTelemetry = snap.telemetry;

    this.turnPhase = snap.turnPhase;
    this.gameMode = snap.gameMode;
    this.playerCount = snap.playerCount;
    this.currentPlayer = snap.currentPlayer;
    this.blocksStacked = snap.blocksStacked;
    this.shatteredCount = snap.shatteredCount;
    this.collapseReason = snap.collapseReason;

    this.activeBlockId = snap.activeBlockId;
    this.hoveredSlotIndex = snap.hoveredSlotIndex;
    this.grabLocalPoint = snap.grabLocalPoint;
    this.grabWorldAnchor = snap.grabWorldAnchor;
    this.grabCurrentWorldOnBlock = snap.grabCurrentWorldOnBlock;
    this.grabPlaneY = snap.grabPlaneY;
    this.grabResistance01 = snap.grabResistance01;
    this.isPointerDraggingBlock = snap.isPointerDraggingBlock;
    this.isExtractionLiftReady = snap.isExtractionLiftReady;

    this.carriedLandingPos = snap.carriedLandingPos;
    this.carriedTargetPos = snap.carriedTargetPos;
    this.carriedTargetQuat = snap.carriedTargetQuat;
    this.carriedYawOffsetRad = snap.carriedYawOffsetRad;

    this.leanAngleDeg = snap.leanAngleDeg;
    this.wobbleSpeedDegPerSec = snap.wobbleSpeedDegPerSec;
    this.centerOfMassXZCm = snap.centerOfMassXZCm;
    this.currentLayers = snap.currentLayers;
    this.peakLayers = snap.peakLayers;
    this.liveHeightCm = snap.liveHeightCm;
    this.peakHeightCm = snap.peakHeightCm;
    this.lockedMinRow = snap.lockedMinRow;

    this.replayAudioEvents(snap.audioEvents);
  }

  private replayAudioEvents(events: WorkerAudioEvent[]): void {
    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      switch (ev.kind) {
        case 'slideFriction':
          this.audio.updateSlideFriction(
            ev.slideSpeedDmPerSec,
            ev.resistance01,
            ev.slipPowerMw,
            ev.torsionRadPerSec
          );
          break;
        case 'tapTest':
          this.audio.playTapTestResonance(
            ev.pinch01,
            ev.worldPos,
            this.listenerCamPos,
            this.listenerCamRight
          );
          break;
        case 'glassImpact':
          this.audio.playGlassImpact(
            ev.intensity01,
            ev.pitchMul,
            ev.worldPos,
            this.listenerCamPos,
            this.listenerCamRight
          );
          break;
        case 'glassCrack':
          this.audio.playGlassCrack(
            ev.damage01,
            ev.worldPos,
            this.listenerCamPos,
            this.listenerCamRight
          );
          break;
        case 'glassShatter':
          this.audio.playGlassShatter(
            ev.intensity01,
            ev.hitTable,
            ev.worldPos,
            this.listenerCamPos,
            this.listenerCamRight
          );
          break;
        case 'rankedShards':
          this.audio.playRankedShardCollisions(
            ev.events,
            this.listenerCamPos,
            this.listenerCamRight
          );
          break;
        case 'stackSuccess':
          this.audio.playStackSuccess(ev.layerCount);
          break;
      }
    }
  }

  public setListenerPose(camPos: Vec3, camRight: Vec3): void {
    this.listenerCamPos[0] = camPos[0];
    this.listenerCamPos[1] = camPos[1];
    this.listenerCamPos[2] = camPos[2];
    this.listenerCamRight[0] = camRight[0];
    this.listenerCamRight[1] = camRight[1];
    this.listenerCamRight[2] = camRight[2];
  }

  public consumePendingShardUploads(): ShardMeshAppendEvent[] {
    if (this.pendingShardUploads.length === 0) return [];
    const out = this.pendingShardUploads;
    this.pendingShardUploads = [];
    return out;
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

  public step(frameDt: number): void {
    if (this.disposed) return;
    if (this.stepInFlight) {
      this.accumDtWhileInFlight = Math.min(0.085, this.accumDtWhileInFlight + frameDt);
      return;
    }
    const totalDt = Math.min(0.085, frameDt + this.accumDtWhileInFlight);
    this.accumDtWhileInFlight = 0;
    this.stepInFlight = true;
    this.post({
      type: 'step',
      dt: totalDt,
      camPos: [...this.listenerCamPos],
      camRight: [...this.listenerCamRight],
      hoveredBlockId: this.hoveredBlockId,
    });
  }

  public rebuildTower(mode?: GameMode, players?: number, layers?: number): void {
    if (mode !== undefined) this.gameMode = mode;
    if (players !== undefined) this.playerCount = Math.max(2, Math.min(4, players));
    this.turnPhase = 'SELECT_BLOCK';
    this.currentPlayer = 0;
    this.blocksStacked = 0;
    this.shatteredCount = 0;
    this.collapseReason = null;
    this.activeBlockId = null;
    this.hoveredBlockId = null;
    this.isPointerDraggingBlock = false;
    this.post({ type: 'rebuildTower', mode, players, layers });
  }

  public tapNudgeBlock(blockId: number, localNormal: Vec3, localPoint: Vec3): boolean {
    const blk = this.blocks[blockId];
    if (!blk || blk.shattered) return false;
    if (this.gameMode !== 'sandbox' && (!blk.isLegalTarget || this.turnPhase !== 'SELECT_BLOCK')) {
      return false;
    }
    this.post({
      type: 'tapNudgeBlock',
      blockId,
      localNormal: [...localNormal],
      localPoint: [...localPoint],
    });
    return true;
  }

  public startGrabBlock(blockId: number, localPoint: Vec3, worldPoint: Vec3): boolean {
    const blk = this.blocks[blockId];
    if (!blk || blk.shattered) return false;
    if (this.gameMode !== 'sandbox' && (!blk.isLegalTarget || this.turnPhase !== 'SELECT_BLOCK')) {
      return false;
    }
    // Optimistically mirror grab state for immediate spring-filament feedback on current frame
    this.activeBlockId = blockId;
    this.isPointerDraggingBlock = true;
    this.grabLocalPoint = [...localPoint];
    this.grabWorldAnchor = [...worldPoint];
    this.grabCurrentWorldOnBlock = [...worldPoint];
    this.grabPlaneY = worldPoint[1];
    if (this.gameMode !== 'sandbox') {
      this.turnPhase = 'PULLING';
    }
    const reqId = this.nextReqId++;
    this.post({
      type: 'startGrabBlock',
      reqId,
      blockId,
      localPoint: [...localPoint],
      worldPoint: [...worldPoint],
    });
    return true;
  }

  public updateGrabTarget(worldTargetOnPlane: Vec3): void {
    this.grabWorldAnchor = [...worldTargetOnPlane];
    this.post({ type: 'updateGrabTarget', worldTargetOnPlane: [...worldTargetOnPlane] });
  }

  public updateCarriedHover(rayOrigin: Vec3, rayDir: Vec3): void {
    this.post({
      type: 'updateCarriedHover',
      rayOrigin: [...rayOrigin],
      rayDir: [...rayDir],
    });
  }

  public rotateCarriedBlock(deltaYawRad: number): void {
    this.carriedYawOffsetRad += deltaYawRad;
    this.post({ type: 'rotateCarriedBlock', deltaYawRad });
  }

  public resetCarriedBlockRotation(): void {
    this.carriedYawOffsetRad = 0;
    this.post({ type: 'resetCarriedBlockRotation' });
  }

  public endGrab(dropIfPoisedOverTower: boolean = false): void {
    this.isPointerDraggingBlock = false;
    this.post({ type: 'endGrab', dropIfPoisedOverTower });
  }

  public releaseExtractedBlockOntoTop(): boolean {
    if (this.turnPhase !== 'EXTRACTED') return false;
    this.isPointerDraggingBlock = false;
    this.post({ type: 'releaseExtractedBlockOntoTop' });
    return true;
  }

  public triggerSandboxCollapseImpulse(): void {
    this.post({ type: 'triggerSandboxCollapseImpulse' });
  }

  public evaluateCounterfactualBlockCriticalityAsync(
    maxCandidates: number = 10,
    rolloutSteps: number = 6,
    candidateBlockIds?: number[]
  ): Promise<BlockCriticalityCandidate[]> {
    if (this.disposed) return Promise.resolve([]);
    const reqId = this.nextReqId++;
    return new Promise<BlockCriticalityCandidate[]>((resolve) => {
      this.oracleCallbacks.set(reqId, resolve);
      this.post({
        type: 'evaluateOracle',
        reqId,
        maxCandidates,
        rolloutSteps,
        candidateBlockIds,
      });
    });
  }

  public getTelemetry(): TowerTelemetry {
    if (this.latestTelemetry) {
      return {
        ...this.latestTelemetry,
        hoveredBlockId: this.hoveredBlockId,
        physicsWorkerActive: true,
      };
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
      centerOfMassXZ: [...this.centerOfMassXZCm],
      shatteredCount: this.shatteredCount,
      crazedCount: 0,
      intactCount: this.blocks.length,
      activeShardCount: this.shards.length,
      grabResistance01: +this.grabResistance01.toFixed(2),
      hoveredBlockId: this.hoveredBlockId,
      activeBlockId: this.activeBlockId,
      hoveredSlotIndex: this.hoveredSlotIndex,
      carriedYawDeg: Math.round((this.carriedYawOffsetRad * 180) / Math.PI),
      lockedMinRow: this.lockedMinRow,
      settleProgress01: 0,
      collapseReason: this.collapseReason,
      physicsWorkerActive: true,
    };
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const cb of this.oracleCallbacks.values()) {
      cb([]);
    }
    this.oracleCallbacks.clear();
    try {
      this.worker.postMessage({ type: 'dispose' } satisfies WorkerInboundMessage);
      this.worker.terminate();
    } catch {
      // Ignore already-terminated worker
    }
  }
}

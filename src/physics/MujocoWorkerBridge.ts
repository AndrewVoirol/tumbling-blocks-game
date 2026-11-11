import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import { Quat, Vec3 } from '../math/vecMat';
import {
  BLOCK_HALF_Y,
  BLOCK_HEIGHT,
  INITIAL_LAYERS,
  MAX_ACTIVE_SHARDS,
  SHARD_SLOT_STRIDE_VERTS,
  TARGET_BLOCK_MASS_KG,
  TraceryMotif,
} from './constants';
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
  public towerSeed: number = 0;
  public lastStepResult: PhysicsStepResult = {
    substeps: 0,
    simTime: 0,
    shardImpacts: [],
    wobbleEvents: [],
  };

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
  private simTime: number = 0;
  private initialLayers: number = INITIAL_LAYERS;
  private shardMetaById = new Map<number, WorkerShardStaticMeta>();
  private slotRenderVertices: Array<Float32Array | null> = new Array(MAX_ACTIVE_SHARDS).fill(null);
  private exportedShardIds = new Set<number>();
  private pendingShardUploads: ShardMeshAppendEvent[] = [];
  private pendingShardImpacts: ShardImpactEvent[] = [];
  private pendingWobbleEvents: WobbleRockEvent[] = [];
  private stepResultShardQueue: ShardImpactEvent[] = [];
  private stepResultWobbleQueue: WobbleRockEvent[] = [];
  private shardBufferResetPending: boolean = false;
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
    initialLayers: number = INITIAL_LAYERS,
    seed?: number
  ): Promise<MujocoWorkerBridge> {
    const worker = new Worker(new URL('./mujocoWorker.ts', import.meta.url), {
      type: 'module',
    });
    const bridge = new MujocoWorkerBridge(worker, audio);
    bridge.initialLayers = initialLayers;
    if (seed !== undefined) {
      bridge.towerSeed = seed >>> 0;
    }

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

      const initMsg: WorkerInboundMessage = { type: 'init', initialLayers, seed };
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
    this.simTime = snap.simTime;
    this.towerSeed = snap.towerSeed >>> 0;
    this.initialLayers = snap.initialLayers;

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
      this.slotRenderVertices.fill(null);
      this.exportedShardIds.clear();
      this.shards = [];
      this.shardBufferResetPending = true;
    }

    const bt = snap.blockTransforms;
    const count = Math.min(this.blocks.length, Math.floor(bt.length / 18));
    for (let i = 0; i < count; i++) {
      const b = this.blocks[i];
      const base = i * 18;
      b.prePos[0] = b.position[0];
      b.prePos[1] = b.position[1];
      b.prePos[2] = b.position[2];
      b.preQuat[0] = b.rotation[0];
      b.preQuat[1] = b.rotation[1];
      b.preQuat[2] = b.rotation[2];
      b.preQuat[3] = b.rotation[3];
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
      b.assignedLayer = ((bits >>> 4) & 0x3f) - 1;
      b.assignedSlot = (bits >>> 10) & 0x3;
      b.swatchId = (bits >>> 12) & 0xf;
      b.motif = ((bits >>> 16) & 0x3) as TraceryMotif;
      b.linVel[0] = bt[base + 12];
      b.linVel[1] = bt[base + 13];
      b.linVel[2] = bt[base + 14];
      b.angVel[0] = bt[base + 15];
      b.angVel[1] = bt[base + 16];
      b.angVel[2] = bt[base + 17];
    }

    this.blockPhotoelasticStress01 = snap.blockPhotoelasticStress01;
    this.blockPhotoelasticWrench = snap.blockPhotoelasticWrench;

    for (const meta of snap.newShardMetas) {
      this.shardMetaById.set(meta.id, meta);
    }

    for (const up of snap.shardUploads) {
      const slotIdx = Math.floor(up.firstVertex / SHARD_SLOT_STRIDE_VERTS);
      if (slotIdx >= 0 && slotIdx < MAX_ACTIVE_SHARDS) {
        this.slotRenderVertices[slotIdx] = new Float32Array(up.vertexData);
      }
      this.pendingShardUploads.push(up);
    }

    if (snap.shardImpacts.length > 0) {
      for (let i = 0; i < snap.shardImpacts.length; i++) {
        const ev = snap.shardImpacts[i];
        this.pendingShardImpacts.push(ev);
        this.stepResultShardQueue.push(ev);
      }
      if (this.pendingShardImpacts.length > 64) {
        this.pendingShardImpacts.splice(0, this.pendingShardImpacts.length - 64);
      }
      if (this.stepResultShardQueue.length > 64) {
        this.stepResultShardQueue.splice(0, this.stepResultShardQueue.length - 64);
      }
    }

    if (snap.wobbleEvents.length > 0) {
      for (let i = 0; i < snap.wobbleEvents.length; i++) {
        const ev = snap.wobbleEvents[i];
        this.pendingWobbleEvents.push(ev);
        this.stepResultWobbleQueue.push(ev);
      }
      if (this.pendingWobbleEvents.length > 32) {
        this.pendingWobbleEvents.splice(0, this.pendingWobbleEvents.length - 32);
      }
      if (this.stepResultWobbleQueue.length > 32) {
        this.stepResultWobbleQueue.splice(0, this.stepResultWobbleQueue.length - 32);
      }
    }

    const st = snap.shardTransforms;
    const shardCount = Math.floor(st.length / 14);
    const nextShards: GlassShardState[] = [];
    for (let i = 0; i < shardCount; i++) {
      const base = i * 14;
      const id = st[base + 0] | 0;
      const meta = this.shardMetaById.get(id);
      if (!meta) continue;
      const pos: Vec3 = [st[base + 1], st[base + 2], st[base + 3]];
      const rot: Quat = [st[base + 4], st[base + 5], st[base + 6], st[base + 7]];
      const preLinVel: Vec3 = [st[base + 8], st[base + 9], st[base + 10]];
      const preAngVel: Vec3 = [st[base + 11], st[base + 12], st[base + 13]];
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
        preLinVel,
        preAngVel,
        volume: meta.volume,
        massKg: meta.massKg,
        preVy: preLinVel[1],
        impactLocalPos: meta.impactLocalPos,
        planes: meta.planes,
        planeCount: meta.planeCount,
        firstVertex: meta.firstVertex,
        vertexCount: meta.vertexCount,
        quietSubsteps: 0,
        spawnSimTime: this.simTime,
      });
    }
    this.shards = nextShards;
    if (this.shardMetaById.size > 1024) {
      const liveIds = new Set<number>();
      for (let i = 0; i < nextShards.length; i++) liveIds.add(nextShards[i].id);
      for (const id of this.shardMetaById.keys()) {
        if (!liveIds.has(id)) this.shardMetaById.delete(id);
      }
    }

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

  public getSimTime(): number {
    return this.simTime;
  }

  public getInitialLayers(): number {
    return this.initialLayers;
  }

  public wasShardBufferReset(): boolean {
    if (this.shardBufferResetPending) {
      this.shardBufferResetPending = false;
      return true;
    }
    return false;
  }

  public consumePendingShardUploads(): ShardMeshAppendEvent[] {
    if (this.pendingShardUploads.length === 0) return [];
    const out = this.pendingShardUploads;
    this.pendingShardUploads = [];
    return out;
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

  public step(frameDt: number): PhysicsStepResult {
    if (this.disposed) {
      this.lastStepResult = {
        substeps: 0,
        simTime: this.simTime,
        shardImpacts: [],
        wobbleEvents: [],
      };
      return this.lastStepResult;
    }

    let shardImpacts: ShardImpactEvent[] = [];
    if (this.stepResultShardQueue.length > 0) {
      if (this.stepResultShardQueue.length > 8) {
        this.stepResultShardQueue.sort(
          (a, b) => b.massKg * b.speed * b.speed - a.massKg * a.speed * a.speed
        );
        shardImpacts = this.stepResultShardQueue.slice(0, 8);
      } else {
        shardImpacts = this.stepResultShardQueue;
      }
      this.stepResultShardQueue = [];
    }

    let wobbleEvents: WobbleRockEvent[] = [];
    if (this.stepResultWobbleQueue.length > 0) {
      if (this.stepResultWobbleQueue.length > 4) {
        this.stepResultWobbleQueue.sort(
          (a, b) => b.angularSpeedRadPerSec - a.angularSpeedRadPerSec
        );
        wobbleEvents = this.stepResultWobbleQueue.slice(0, 4);
      } else {
        wobbleEvents = this.stepResultWobbleQueue;
      }
      this.stepResultWobbleQueue = [];
    }

    if (this.stepInFlight) {
      this.accumDtWhileInFlight = Math.min(0.085, this.accumDtWhileInFlight + frameDt);
      this.lastStepResult = {
        substeps: 0,
        simTime: this.simTime,
        shardImpacts,
        wobbleEvents,
      };
      return this.lastStepResult;
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
    this.lastStepResult = {
      substeps: 1,
      simTime: this.simTime,
      shardImpacts,
      wobbleEvents,
    };
    return this.lastStepResult;
  }

  public rebuildTower(mode?: GameMode, players?: number, layers?: number, seed?: number): void {
    if (mode !== undefined) this.gameMode = mode;
    if (players !== undefined) this.playerCount = Math.max(2, Math.min(4, players));
    if (layers !== undefined) this.initialLayers = layers;
    this.towerSeed = seed !== undefined ? seed >>> 0 : 0;
    this.simTime = 0;
    this.turnPhase = 'SELECT_BLOCK';
    this.currentPlayer = 0;
    this.blocksStacked = 0;
    this.shatteredCount = 0;
    this.collapseReason = null;
    this.activeBlockId = null;
    this.hoveredBlockId = null;
    this.isPointerDraggingBlock = false;
    this.pendingShardImpacts.length = 0;
    this.pendingWobbleEvents.length = 0;
    this.stepResultShardQueue.length = 0;
    this.stepResultWobbleQueue.length = 0;
    this.exportedShardIds.clear();
    this.slotRenderVertices.fill(null);
    this.lastStepResult = {
      substeps: 0,
      simTime: 0,
      shardImpacts: [],
      wobbleEvents: [],
    };
    this.post({ type: 'rebuildTower', mode, players, layers, seed });
  }

  public resetTower(layers?: number, seed?: number): void {
    const targetLayers = layers ?? this.initialLayers;
    this.rebuildTower(this.gameMode, this.playerCount, targetLayers, seed);
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

    return {
      version: 1,
      simTime: this.simTime,
      towerSeed: this.towerSeed,
      initialLayers: this.initialLayers,
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
    this.simTime = snap.simTime;
    this.towerSeed = snap.towerSeed >>> 0;
    this.initialLayers = snap.initialLayers;
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

    while (this.blocks.length < snap.blockCount) {
      const id = this.blocks.length;
      const initialLayer = Math.floor(id / 3);
      this.blocks.push({
        id,
        bodyHandle: 0,
        colliderHandle: 0,
        swatchId: 0,
        motif: 0,
        rippleSeed: 0,
        initialLayer,
        assignedLayer: initialLayer,
        assignedSlot: id % 3,
        isLooseSlot: false,
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        prePos: [0, 0, 0],
        preQuat: [0, 0, 0, 1],
        linVel: [0, 0, 0],
        angVel: [0, 0, 0],
        currentFriction: 0.275,
        isMicroDamped: false,
        damage: 0,
        crackLocalOrigin: [0, 0, 0],
        shattered: false,
        extracted: false,
        isLegalTarget: false,
      });
    }
    if (this.blocks.length > snap.blockCount) {
      this.blocks.length = snap.blockCount;
    }

    const bd = snap.blockData;
    for (let i = 0; i < snap.blockCount; i++) {
      const b = this.blocks[i];
      const base = i * 19;
      const px = bd[base + 0];
      const py = bd[base + 1];
      const pz = bd[base + 2];
      const qx = bd[base + 3];
      const qy = bd[base + 4];
      const qz = bd[base + 5];
      const qw = bd[base + 6];
      b.position = [px, py, pz];
      b.rotation = [qx, qy, qz, qw];
      b.prePos = [px, py, pz];
      b.preQuat = [qx, qy, qz, qw];
      b.linVel = [bd[base + 7], bd[base + 8], bd[base + 9]];
      b.angVel = [bd[base + 10], bd[base + 11], bd[base + 12]];
      b.damage = bd[base + 13];
      b.crackLocalOrigin = [bd[base + 14], bd[base + 15], bd[base + 16]];
      b.rippleSeed = bd[base + 17];
      const bits = bd[base + 18] | 0;
      b.shattered = (bits & 1) !== 0;
      b.extracted = (bits & 2) !== 0;
      b.isLegalTarget = (bits & 4) !== 0;
      b.isLooseSlot = (bits & 8) !== 0;
      b.assignedLayer = ((bits >>> 4) & 0x3f) - 1;
      b.assignedSlot = (bits >>> 10) & 0x3;
      b.swatchId = (bits >>> 12) & 0xf;
      b.motif = ((bits >>> 16) & 0x3) as TraceryMotif;
    }

    this.blockPhotoelasticWrench = new Float32Array(snap.blockPhotoelasticWrench);
    this.blockPhotoelasticStress01 = new Float32Array(snap.blockCount);
    for (let i = 0; i < snap.blockCount; i++) {
      const wb = i * 4;
      this.blockPhotoelasticStress01[i] = Math.max(
        this.blockPhotoelasticWrench[wb + 0] ?? 0,
        this.blockPhotoelasticWrench[wb + 1] ?? 0,
        this.blockPhotoelasticWrench[wb + 2] ?? 0,
        this.blockPhotoelasticWrench[wb + 3] ?? 0
      );
    }

    if (snap.shardMetas) {
      for (const meta of snap.shardMetas) {
        const firstVertex = meta.slotIndex * SHARD_SLOT_STRIDE_VERTS;
        this.shardMetaById.set(meta.id, {
          id: meta.id,
          slotIndex: meta.slotIndex,
          swatchId: meta.swatchId,
          motif: meta.motif,
          rippleSeed: meta.rippleSeed,
          volume: meta.volume,
          massKg: meta.massKg,
          firstVertex,
          vertexCount: meta.vertexCount,
          planeCount: meta.planeCount,
          impactLocalPos: [...meta.impactLocalPos],
          planes: new Float32Array(meta.planes),
        });
        if (meta.renderVertices) {
          const vCopy = new Float32Array(meta.renderVertices);
          if (meta.slotIndex >= 0 && meta.slotIndex < MAX_ACTIVE_SHARDS) {
            this.slotRenderVertices[meta.slotIndex] = vCopy;
          }
          this.pendingShardUploads.push({
            firstVertex,
            vertexData: vCopy,
          });
        }
      }
    }

    const sd = snap.shardData;
    const nextShards: GlassShardState[] = [];
    for (let i = 0; i < snap.shardCount; i++) {
      const base = i * 15;
      const id = sd[base + 0] | 0;
      const slotIndex = sd[base + 1] | 0;
      const meta = this.shardMetaById.get(id);
      const pos: Vec3 = [sd[base + 2], sd[base + 3], sd[base + 4]];
      const rot: Quat = [sd[base + 5], sd[base + 6], sd[base + 7], sd[base + 8]];
      const preLinVel: Vec3 = [sd[base + 9], sd[base + 10], sd[base + 11]];
      const preAngVel: Vec3 = [sd[base + 12], sd[base + 13], sd[base + 14]];
      nextShards.push({
        id,
        slotIndex,
        bodyHandle: 0,
        colliderHandle: 0,
        swatchId: meta?.swatchId ?? 0,
        motif: meta?.motif ?? 0,
        rippleSeed: meta?.rippleSeed ?? 1.0,
        position: pos,
        rotation: rot,
        prePos: pos,
        preQuat: rot,
        preLinVel,
        preAngVel,
        volume: meta?.volume ?? 0.002,
        massKg: meta?.massKg ?? Math.max(0.0002, TARGET_BLOCK_MASS_KG * 0.04),
        preVy: preLinVel[1],
        impactLocalPos: meta ? [...meta.impactLocalPos] : [0, 0, 0],
        planes: meta ? meta.planes : new Float32Array(48),
        planeCount: meta?.planeCount ?? 0,
        firstVertex: slotIndex * SHARD_SLOT_STRIDE_VERTS,
        vertexCount: meta?.vertexCount ?? 0,
        quietSubsteps: 0,
        spawnSimTime: this.simTime - 0.05,
      });
    }
    this.shards = nextShards;
    if (this.shardMetaById.size > 1024) {
      const liveIds = new Set<number>();
      for (let i = 0; i < nextShards.length; i++) liveIds.add(nextShards[i].id);
      for (const id of this.shardMetaById.keys()) {
        if (!liveIds.has(id)) this.shardMetaById.delete(id);
      }
    }
    this.lastStepResult = {
      substeps: 0,
      simTime: this.simTime,
      shardImpacts: [],
      wobbleEvents: [],
    };
    this.post({ type: 'applyCompactSnapshot', snapshot: snap });
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

import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import { Quat, quatRotateVec3, Vec3 } from '../math/vecMat';
import {
  BLOCK_HALF_Y,
  BLOCK_HEIGHT,
  INITIAL_LAYERS,
  SHARD_SLOT_STRIDE_VERTS,
} from '../physics/constants';
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
} from '../physics/PhysicsWorld';
import {
  computeEffectiveGrabClampingLoadN,
  countOtherBlocksInLayer,
  estimateClampingLoadFromWrenchAndStack,
  evaluateCompetitiveDragSlip,
  isCompetitiveInwardEndPush,
  normalizeCompetitiveSlipResult,
  type ExtractionSlipResult,
  updateEndPushSmoothedClampingLoadN,
} from '../physics/extractionMechanics';
import {
  applyDeterministicTowerPalette,
  canonicalizeQuatHemisphere,
  computeDeterministicBlockMaterial,
  computeNominalTowerBlockPose,
  MultiplayerStateFrame,
  NetworkAudioEvent,
  quatSlerpHemisphere,
  reconstructShardVerticesFromMeta,
  SerializedShardMeta,
} from './protocol';

interface ShardCacheEntry {
  meta: SerializedShardMeta;
  planesFloat32: Float32Array;
  firstVertex: number;
  vertexCount: number;
  currentPos: Vec3;
  currentQuat: Quat;
  targetPos: Vec3;
  targetQuat: Quat;
  hasPose: boolean;
}

interface ExtendedGuestAudioHooks {
  playShardImpacts?: (
    impacts: ReadonlyArray<{
      pos: Vec3;
      speed: number;
      massKg: number;
      surface: 'wood' | 'glass';
      slotIndex: number;
    }>,
    camPos?: Vec3,
    camRight?: Vec3
  ) => number;
  playWobbleRock?: (
    tiltAngleRad: number,
    angularSpeedRadPerSec?: number,
    clampingLoadN?: number,
    worldPos?: Vec3,
    options?: {
      blockId?: number;
      surface?: 'glass' | 'wood';
      damage?: number;
      camPos?: Vec3;
      camRight?: Vec3;
    }
  ) => number;
  playGlassImpact: (
    intensity01: number,
    pitchScale?: number,
    worldPos?: Vec3,
    camPos?: Vec3,
    camRight?: Vec3,
    options?: {
      impactSpeedMps?: number;
      damage?: number;
      surface?: 'glass' | 'wood';
    }
  ) => void;
  playBlockThud?: (
    impactSpeedMps: number,
    worldPos?: Vec3,
    camPosOrOptions?: Vec3 | { damage?: number; camPos?: Vec3; camRight?: Vec3 },
    camRight?: Vec3,
    damage?: number
  ) => void;
  playTapTestResonance: (
    pinch01: number,
    worldPos?: Vec3,
    camPos?: Vec3,
    camRight?: Vec3,
    damage?: number
  ) => void;
  updateSlideFriction: (
    slideSpeedDmPerSec: number,
    resistance01: number,
    slipPowerMilliwatts?: number,
    torsionRadPerSec?: number,
    clampingLoadN?: number,
    seamCrossingIntensity?: number
  ) => void;
}

/**
 * Render-interpolated Guest Physics Mirror (`~36 ms` exponential + hemisphere-safe quaternion slerp).
 *
 * Implements the `PhysicsWorld` surface consumed by `WebGPURenderer` and `GameEngine` so
 * `30 Hz` authoritative Host snapshots render smoothly at `60–120 Hz` on ProMotion displays
 * with zero quaternion sign-flip pops (`dot(q0, q1) < 0` guard) and identical `roomSeed` glass colors.
 */
export class MultiplayerGuestWorld {
  public readonly isRemoteGuestMirror = true;
  public roomSeed: number = 0;

  public blocks: GlassBlockState[] = [];
  public shards: GlassShardState[] = [];
  public splinters: MicroSplinter[] = [];
  public openTopSlots: TopPlacementSlot[] = [];
  public blockPhotoelasticStress01: Float32Array = new Float32Array(0);
  public blockPhotoelasticWrench: Float32Array = new Float32Array(0);
  private blockClampingLoadsN: Float32Array = new Float32Array(0);

  public gameMode: GameMode = 'party';
  public playerCount: number = 2;
  public currentPlayer: number = 0;
  public turnPhase: TurnPhase = 'SELECT_BLOCK';
  public blocksStacked: number = 0;
  public shatteredCount: number = 0;
  public collapseReason: string | null = null;

  public hoveredBlockId: number | null = null;
  public hoverLocalPoint: Vec3 | null = null;
  public activeBlockId: number | null = null;
  public hoveredSlotIndex: number | null = null;

  public grabLocalPoint: Vec3 = [0, 0, 0];
  public grabWorldAnchor: Vec3 = [0, 0, 0];
  public grabCurrentWorldOnBlock: Vec3 = [0, 0, 0];
  public grabPlaneY: number = 0;
  public grabResistance01: number = 0;
  public isPointerDraggingBlock: boolean = false;
  public isExtractionLiftReady: boolean = false;
  public lastExtractionSlip: ExtractionSlipResult | null = null;

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
  public settleProgress01: number = 0;

  /** Exponential smoothing time constant (`36 ms`, inside `30–45 ms` target window) */
  public readonly interpolationTauSec: number = 0.036;

  private audio: GlassAudioEngine;
  private targetBlockPos: Vec3[] = [];
  private targetBlockQuat: Quat[] = [];
  private targetGrabOnBlock: Vec3 = [0, 0, 0];
  private targetGrabAnchor: Vec3 = [0, 0, 0];
  private targetCarriedLandingPos: Vec3 = [0, INITIAL_LAYERS * BLOCK_HEIGHT + BLOCK_HALF_Y, 0];
  private targetCarriedQuat: Quat = [0, 0, 0, 1];

  private shardMap = new Map<number, ShardCacheEntry>();
  private pendingShardUploads: ShardMeshAppendEvent[] = [];
  private lastAcceptedSeq: number = 0;
  private acceptedFrameCount: number = 0;
  private rejectedFrameCount: number = 0;
  private audioBatchSeqFloor: number = 0;
  private receivedAudioBatchSeqs = new Set<number>();
  private duplicateAudioBatchCount: number = 0;
  private hasReceivedWelcome: boolean = false;
  private hasCurrentSeedStateFrame: boolean = false;
  private snapNextPose: boolean = true;

  private localGuestHoverBlockId: number | null = null;
  private localGuestHoverLocalPoint: Vec3 | null = null;
  private localOptimisticPullAnchor: Vec3 | null = null;
  private localOptimisticPullStartAnchor: Vec3 | null = null;
  private localOptimisticGrabBlockId: number | null = null;
  private localOptimisticGrabLocalPoint: Vec3 = [0, 0, 0];
  private localOptimisticGrabBlockStartPos: Vec3 | null = null;
  private localOptimisticGrabInitialClampN: number = 0;
  private localOptimisticGrabSmoothedClampN: number = 0;
  private localOptimisticGrabLastSmoothHostSec: number = -1;
  private lastAcceptedHostElapsedSec: number = -1;
  private localOptimisticGrabConfirmedByHost: boolean = false;
  private localOptimisticGrabPreAckFrames: number = 0;
  private localOptimisticGrabStartedAtMs: number = 0;
  private lastEvaluatedClampingLoadN: number = 0;
  private hostHoverBlockId: number | null = null;

  constructor(audio: GlassAudioEngine, initialLayers: number = INITIAL_LAYERS, roomSeed: number = 0x56495452) {
    this.audio = audio;
    this.initDeterministicTower(initialLayers, roomSeed);
  }

  public initDeterministicTower(
    layers: number = INITIAL_LAYERS,
    roomSeed: number = 0x56495452,
    options?: { preserveSequenceState?: boolean }
  ): void {
    this.roomSeed = roomSeed >>> 0;
    const total = layers * 3;
    this.blocks = [];
    this.targetBlockPos = [];
    this.targetBlockQuat = [];
    this.shards = [];
    this.splinters = [];
    this.shardMap.clear();
    this.pendingShardUploads = [];
    if (!options?.preserveSequenceState) {
      this.lastAcceptedSeq = 0;
      this.acceptedFrameCount = 0;
      this.rejectedFrameCount = 0;
      this.audioBatchSeqFloor = 0;
      this.receivedAudioBatchSeqs.clear();
      this.duplicateAudioBatchCount = 0;
      this.hasReceivedWelcome = false;
    }
    this.hasCurrentSeedStateFrame = false;
    this.lastAcceptedHostElapsedSec = -1;
    this.localOptimisticGrabLastSmoothHostSec = -1;
    this.snapNextPose = true;
    this.collapseReason = null;
    this.turnPhase = 'SELECT_BLOCK';
    this.currentPlayer = 0;
    this.blocksStacked = 0;
    this.shatteredCount = 0;
    this.currentLayers = layers;
    this.peakLayers = layers;
    this.liveHeightCm = layers * BLOCK_HEIGHT * 10;
    this.peakHeightCm = layers * BLOCK_HEIGHT * 10;
    this.lockedMinRow = layers;
    this.clearOptimisticGrabState();
    this.localGuestHoverLocalPoint = null;
    this.hoverLocalPoint = null;
    this.lastExtractionSlip = null;

    this.blockPhotoelasticStress01 = new Float32Array(total + 32);
    this.blockPhotoelasticWrench = new Float32Array((total + 32) * 4);
    this.blockClampingLoadsN = new Float32Array(total + 32);
    this.blockClampingLoadsN.fill(Number.NaN);

    for (let layer = 0; layer < layers; layer++) {
      const centerIsLoose = layer % 2 === 0;
      for (let slot = 0; slot < 3; slot++) {
        const id = layer * 3 + slot;
        const isLoose = centerIsLoose ? slot === 1 : slot === (layer % 4 === 1 ? 0 : 2);
        const nominal = computeNominalTowerBlockPose(layer, slot);
        const blk: GlassBlockState = {
          id,
          bodyHandle: id + 1,
          colliderHandle: id + 1,
          swatchId: 0,
          motif: 0,
          rippleSeed: 1.0,
          initialLayer: layer,
          assignedLayer: layer,
          assignedSlot: slot,
          isLooseSlot: isLoose,
          position: [...nominal.position],
          rotation: [...nominal.rotation],
          prePos: [...nominal.position],
          preQuat: [...nominal.rotation],
          linVel: [0, 0, 0],
          angVel: [0, 0, 0],
          currentFriction: 0.3,
          isMicroDamped: false,
          damage: 0,
          crackLocalOrigin: [0, 0, 0],
          shattered: false,
          extracted: false,
          isLegalTarget: layer < layers - 1,
        };
        this.blocks.push(blk);
        this.targetBlockPos.push([...nominal.position]);
        this.targetBlockQuat.push([...nominal.rotation]);
      }
    }

    applyDeterministicTowerPalette(this.blocks, this.roomSeed, layers);
    this.rebuildOpenTopSlotsFromSupportFrame();
  }

  public applyRoomWelcome(params: {
    roomSeed: number;
    initialLayers: number;
    collapseReason: string | null;
    activeShardMetas: SerializedShardMeta[];
    retiredShardIds: number[];
    totalActiveShards?: number;
    minSnapshotSeq?: number;
    currentAudioBatchSeq?: number;
  }): void {
    const nextSeed = params.roomSeed >>> 0;
    const isFreshWelcome = !this.hasReceivedWelcome && this.lastAcceptedSeq === 0;
    const seedChanged = this.roomSeed !== nextSeed;
    const hostEpochReset =
      typeof params.minSnapshotSeq === 'number' &&
      Number.isFinite(params.minSnapshotSeq) &&
      params.minSnapshotSeq >>> 0 < this.lastAcceptedSeq;
    this.hasReceivedWelcome = true;

    if (
      seedChanged ||
      hostEpochReset ||
      this.blocks.length !== params.initialLayers * 3
    ) {
      this.initDeterministicTower(params.initialLayers, nextSeed, {
        preserveSequenceState: !isFreshWelcome && !hostEpochReset,
      });
      this.hasReceivedWelcome = true;
    } else {
      applyDeterministicTowerPalette(this.blocks, nextSeed, params.initialLayers);
      if (isFreshWelcome) {
        this.snapNextPose = true;
      }
    }

    if (params.totalActiveShards === 0 && params.activeShardMetas.length === 0) {
      this.shardMap.clear();
      this.rebuildShardList();
    }

    if (
      typeof params.minSnapshotSeq === 'number' &&
      Number.isFinite(params.minSnapshotSeq) &&
      (hostEpochReset || params.minSnapshotSeq > this.lastAcceptedSeq)
    ) {
      this.lastAcceptedSeq = params.minSnapshotSeq >>> 0;
    }

    if (
      (isFreshWelcome || seedChanged || hostEpochReset) &&
      typeof params.currentAudioBatchSeq === 'number' &&
      Number.isFinite(params.currentAudioBatchSeq)
    ) {
      const targetFloor = params.currentAudioBatchSeq >>> 0;
      if (hostEpochReset) {
        this.audioBatchSeqFloor = targetFloor;
        this.receivedAudioBatchSeqs.clear();
      } else if (targetFloor > this.audioBatchSeqFloor) {
        this.audioBatchSeqFloor = targetFloor;
        for (const seq of this.receivedAudioBatchSeqs) {
          if (seq <= targetFloor) {
            this.receivedAudioBatchSeqs.delete(seq);
          }
        }
      }
    }

    this.collapseReason = params.collapseReason;
    if (params.activeShardMetas.length > 0 || params.retiredShardIds.length > 0) {
      this.applyShardSpawn(params.activeShardMetas, params.retiredShardIds);
    }
  }

  public applyShardSpawn(spawned: SerializedShardMeta[], retiredIds: number[]): void {
    for (const oldId of retiredIds) {
      this.shardMap.delete(oldId);
    }
    for (const meta of spawned) {
      const hasInitPose =
        Array.isArray(meta.initialPosition) &&
        meta.initialPosition.length === 3 &&
        Array.isArray(meta.initialRotation) &&
        meta.initialRotation.length === 4;

      const existing = this.shardMap.get(meta.id);
      if (existing) {
        if (hasInitPose && (!existing.hasPose || this.snapNextPose)) {
          existing.currentPos = [...meta.initialPosition!];
          existing.targetPos = [...meta.initialPosition!];
          existing.currentQuat = canonicalizeQuatHemisphere(meta.initialRotation!);
          existing.targetQuat = [...existing.currentQuat];
          existing.hasPose = true;
        }
        continue;
      }

      // Evict any stale shard occupying the same 512-slot GPU vertex ring slot
      for (const [oldId, entry] of this.shardMap.entries()) {
        if (oldId !== meta.id && entry.meta.slotIndex === meta.slotIndex) {
          this.shardMap.delete(oldId);
        }
      }

      const built = reconstructShardVerticesFromMeta(meta);
      const firstVertex = meta.slotIndex * SHARD_SLOT_STRIDE_VERTS;
      this.pendingShardUploads.push({
        firstVertex,
        vertexData: built.renderVertices,
      });
      const initPos: Vec3 = hasInitPose
        ? [meta.initialPosition![0], meta.initialPosition![1], meta.initialPosition![2]]
        : [0, -10, 0];
      const initQuat: Quat = hasInitPose
        ? canonicalizeQuatHemisphere([
            meta.initialRotation![0],
            meta.initialRotation![1],
            meta.initialRotation![2],
            meta.initialRotation![3],
          ])
        : [0, 0, 0, 1];
      this.shardMap.set(meta.id, {
        meta,
        planesFloat32: built.planesFloat32,
        firstVertex,
        vertexCount: built.vertexCount,
        currentPos: [...initPos],
        currentQuat: [...initQuat],
        targetPos: [...initPos],
        targetQuat: [...initQuat],
        hasPose: hasInitPose,
      });
    }
    this.rebuildShardList();
  }

  public applyNetworkStateFrame(frame: MultiplayerStateFrame): boolean {
    if (!Number.isFinite(frame.seq) || frame.seq <= this.lastAcceptedSeq) {
      // Strictly discard ANY stale or out-of-order 0x564d UDP packet across both normal play and rapid rematches
      this.rejectedFrameCount++;
      return false;
    }

    const nextSeed = frame.roomSeed >>> 0;
    const seedChanged = nextSeed !== this.roomSeed && nextSeed !== 0;
    if (seedChanged) {
      const layers = Math.max(1, Math.round(frame.blocks.length / 3) || INITIAL_LAYERS);
      this.initDeterministicTower(layers, nextSeed, { preserveSequenceState: true });
      this.hasReceivedWelcome = true;
    }

    this.lastAcceptedSeq = frame.seq >>> 0;
    this.acceptedFrameCount++;
    this.hasCurrentSeedStateFrame = true;
    const prevHostElapsedSec = this.lastAcceptedHostElapsedSec;
    this.lastAcceptedHostElapsedSec = frame.hostElapsedSec;

    const snapImmediate = this.snapNextPose;
    this.snapNextPose = false;

    this.turnPhase = frame.turnPhase;
    this.gameMode = frame.gameMode;
    this.currentPlayer = frame.currentPlayer;
    this.playerCount = frame.playerCount;
    this.blocksStacked = frame.blocksStacked;
    this.shatteredCount = frame.shatteredCount;
    this.currentLayers = frame.currentLayers;
    this.peakLayers = frame.peakLayers;
    this.activeBlockId = frame.activeBlockId;
    this.hostHoverBlockId = frame.hoveredBlockId;
    this.hoveredSlotIndex = frame.hoveredSlotIndex;
    this.lockedMinRow = frame.lockedMinRow;
    this.isPointerDraggingBlock = frame.isPointerDraggingBlock;
    this.isExtractionLiftReady = frame.isExtractionLiftReady;
    this.grabResistance01 = frame.grabResistance01;
    this.settleProgress01 = frame.settleProgress01;
    this.leanAngleDeg = frame.leanAngleDeg;
    this.wobbleSpeedDegPerSec = frame.wobbleSpeedDegPerSec;
    this.liveHeightCm = frame.liveHeightCm;
    this.peakHeightCm = frame.peakHeightCm;
    this.centerOfMassXZCm = [...frame.centerOfMassXZCm];
    this.carriedYawOffsetRad = frame.carriedYawOffsetRad;

    if (
      this.currentPlayer !== 1 ||
      this.turnPhase === 'COLLAPSED' ||
      this.turnPhase === 'SETTLING' ||
      this.turnPhase === 'EXTRACTED'
    ) {
      this.clearOptimisticGrabState();
    } else if (this.localOptimisticGrabBlockId !== null) {
      const hostConfirmedGrab =
        (frame.isPointerDraggingBlock &&
          frame.activeBlockId === this.localOptimisticGrabBlockId) ||
        (frame.turnPhase === 'PULLING' &&
          frame.activeBlockId === this.localOptimisticGrabBlockId);
      if (hostConfirmedGrab) {
        this.localOptimisticGrabConfirmedByHost = true;
        if (!frame.isPointerDraggingBlock && frame.turnPhase !== 'PULLING') {
          this.clearOptimisticGrabState();
        }
      } else if (this.localOptimisticGrabConfirmedByHost) {
        if (!frame.isPointerDraggingBlock) {
          this.clearOptimisticGrabState();
        }
      } else {
        const targetBlk = this.blocks[this.localOptimisticGrabBlockId];
        const invalidTarget =
          !targetBlk ||
          targetBlk.shattered ||
          targetBlk.extracted ||
          (frame.activeBlockId !== null &&
            frame.activeBlockId !== this.localOptimisticGrabBlockId);
        this.localOptimisticGrabPreAckFrames++;
        const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
        const expired =
          this.localOptimisticGrabPreAckFrames > 15 &&
          nowMs - this.localOptimisticGrabStartedAtMs > 1000;
        if (invalidTarget || expired) {
          this.clearOptimisticGrabState();
        } else {
          this.activeBlockId = this.localOptimisticGrabBlockId;
          this.isPointerDraggingBlock = true;
        }
      }
    }

    if (this.currentPlayer === 1 && this.localGuestHoverBlockId !== null) {
      this.hoveredBlockId = this.localGuestHoverBlockId;
      this.hoverLocalPoint = this.localGuestHoverLocalPoint
        ? [...this.localGuestHoverLocalPoint]
        : null;
    } else {
      this.hoveredBlockId = this.hostHoverBlockId;
      this.hoverLocalPoint = null;
    }

    if (
      this.currentPlayer === 1 &&
      this.localOptimisticGrabBlockId !== null &&
      (this.activeBlockId === this.localOptimisticGrabBlockId ||
        !this.localOptimisticGrabConfirmedByHost)
    ) {
      this.grabLocalPoint = [...this.localOptimisticGrabLocalPoint];
    }

    const preservePreAckGrabPose =
      this.currentPlayer === 1 &&
      this.localOptimisticGrabBlockId !== null &&
      !this.localOptimisticGrabConfirmedByHost;
    this.targetGrabOnBlock =
      preservePreAckGrabPose && this.localOptimisticPullStartAnchor
        ? [...this.localOptimisticPullStartAnchor]
        : [...frame.grabCurrentWorldOnBlock];
    this.targetGrabAnchor = this.localOptimisticPullAnchor
      ? [...this.localOptimisticPullAnchor]
      : [...frame.grabWorldAnchor];
    this.grabPlaneY =
      preservePreAckGrabPose && this.localOptimisticPullAnchor
        ? this.localOptimisticPullAnchor[1]
        : frame.grabWorldAnchor[1];

    this.targetCarriedLandingPos = [...frame.carriedLandingPos];
    this.targetCarriedQuat = [...frame.carriedTargetQuat];
    this.carriedTargetPos = [
      frame.carriedLandingPos[0],
      frame.carriedLandingPos[1] + 0.12,
      frame.carriedLandingPos[2],
    ];

    const sc = frame.topSupportSurfaceCenter;
    const up = frame.topSupportUpNormal;
    const defRot = frame.topSupportDefaultRot;
    const lenAxis = quatRotateVec3(defRot, [1, 0, 0]);
    const spAxis = quatRotateVec3(defRot, [0, 0, 1]);
    this.topSupportFrame = {
      targetLayer: frame.topSupportTargetLayer,
      surfaceCenter: [...sc],
      layerCenter: [
        sc[0] + up[0] * (BLOCK_HALF_Y + 0.002),
        sc[1] + up[1] * (BLOCK_HALF_Y + 0.002),
        sc[2] + up[2] * (BLOCK_HALF_Y + 0.002),
      ],
      upNormal: [...up],
      tiltQuat: [0, 0, 0, 1],
      defaultYawRad: frame.topSupportTargetLayer % 2 === 1 ? Math.PI * 0.5 : 0,
      defaultRotation: [...defRot],
      lengthAxis: lenAxis,
      spacingAxis: spAxis,
    };
    this.rebuildOpenTopSlotsFromSupportFrame();

    if (snapImmediate) {
      this.grabCurrentWorldOnBlock = [...this.targetGrabOnBlock];
      this.grabWorldAnchor = [...this.targetGrabAnchor];
      this.carriedLandingPos = [...this.targetCarriedLandingPos];
      this.carriedTargetQuat = quatSlerpHemisphere(
        this.carriedTargetQuat,
        this.targetCarriedQuat,
        1.0
      );
    }

    const neededCap = frame.blocks.length + 8;
    if (this.blockPhotoelasticStress01.length < neededCap) {
      this.blockPhotoelasticStress01 = new Float32Array(neededCap);
      this.blockPhotoelasticWrench = new Float32Array(neededCap * 4);
    }
    if (this.blockClampingLoadsN.length < neededCap) {
      this.blockClampingLoadsN = new Float32Array(neededCap);
      this.blockClampingLoadsN.fill(Number.NaN);
    }

    for (let i = 0; i < frame.blocks.length; i++) {
      const fb = frame.blocks[i];
      let blk = this.blocks[i];
      if (!blk) {
        const mat = computeDeterministicBlockMaterial(
          this.roomSeed,
          i,
          fb.assignedLayer,
          fb.assignedSlot,
          INITIAL_LAYERS
        );
        blk = {
          id: i,
          bodyHandle: i + 1,
          colliderHandle: i + 1,
          swatchId: fb.swatchId,
          motif: fb.motif,
          rippleSeed: mat.rippleSeed,
          initialLayer: fb.assignedLayer,
          assignedLayer: fb.assignedLayer,
          assignedSlot: fb.assignedSlot,
          isLooseSlot: fb.isLooseSlot,
          position: [...fb.position],
          rotation: [...fb.rotation],
          prePos: [...fb.position],
          preQuat: [...fb.rotation],
          linVel: [0, 0, 0],
          angVel: [0, 0, 0],
          currentFriction: 0.3,
          isMicroDamped: false,
          damage: fb.damage,
          crackLocalOrigin: [...fb.crackLocalOrigin],
          shattered: fb.shattered,
          extracted: fb.extracted,
          isLegalTarget: fb.isLegalTarget,
        };
        this.blocks.push(blk);
        this.targetBlockPos.push([...fb.position]);
        this.targetBlockQuat.push([...fb.rotation]);
      }

      this.targetBlockPos[i] = [...fb.position];
      this.targetBlockQuat[i] = [...fb.rotation];

      const distJump = Math.hypot(
        fb.position[0] - blk.position[0],
        fb.position[1] - blk.position[1],
        fb.position[2] - blk.position[2]
      );
      if (snapImmediate || distJump > 1.2) {
        blk.position[0] = fb.position[0];
        blk.position[1] = fb.position[1];
        blk.position[2] = fb.position[2];
        blk.rotation = quatSlerpHemisphere(blk.rotation, fb.rotation, 1.0);
      }

      blk.swatchId = fb.swatchId;
      blk.motif = fb.motif;
      blk.damage = fb.damage;
      blk.shattered = fb.shattered;
      blk.extracted = fb.extracted;
      blk.isLegalTarget = fb.isLegalTarget;
      blk.isLooseSlot = fb.isLooseSlot;
      blk.assignedLayer = fb.assignedLayer;
      blk.assignedSlot = fb.assignedSlot;
      blk.crackLocalOrigin[0] = fb.crackLocalOrigin[0];
      blk.crackLocalOrigin[1] = fb.crackLocalOrigin[1];
      blk.crackLocalOrigin[2] = fb.crackLocalOrigin[2];

      this.blockPhotoelasticStress01[i] = fb.stress01;
      const wBase = i * 4;
      this.blockPhotoelasticWrench[wBase + 0] = fb.photoelasticWrench[0];
      this.blockPhotoelasticWrench[wBase + 1] = fb.photoelasticWrench[1];
      this.blockPhotoelasticWrench[wBase + 2] = fb.photoelasticWrench[2];
      this.blockPhotoelasticWrench[wBase + 3] = fb.photoelasticWrench[3];
      this.blockClampingLoadsN[i] =
        typeof fb.clampingLoadN === 'number' && Number.isFinite(fb.clampingLoadN)
          ? fb.clampingLoadN
          : Number.NaN;
    }

    for (let i = 0; i < frame.shards.length; i++) {
      const fs = frame.shards[i];
      const entry = this.shardMap.get(fs.id);
      if (!entry) continue;
      entry.targetPos = [...fs.position];
      entry.targetQuat = [...fs.rotation];
      if (!entry.hasPose || snapImmediate) {
        entry.currentPos = [...fs.position];
        entry.currentQuat = entry.hasPose
          ? quatSlerpHemisphere(entry.currentQuat, fs.rotation, 1.0)
          : [...fs.rotation];
        entry.hasPose = true;
      }
    }
    this.rebuildShardList();

    const trackedId = this.localOptimisticGrabBlockId ?? this.activeBlockId;
    if (trackedId !== null && this.gameMode !== 'sandbox') {
      const trackedBlk = this.blocks[trackedId];
      if (trackedBlk && !trackedBlk.shattered) {
        const localPt =
          this.localOptimisticGrabBlockId === trackedId
            ? this.localOptimisticGrabLocalPoint
            : this.grabLocalPoint;
        const targetPt = this.localOptimisticPullAnchor ?? this.grabWorldAnchor;
        const curOnBlk =
          this.grabCurrentWorldOnBlock[0] === 0 &&
          this.grabCurrentWorldOnBlock[1] === 0 &&
          this.grabCurrentWorldOnBlock[2] === 0 &&
          this.localOptimisticPullStartAnchor
            ? this.localOptimisticPullStartAnchor
            : this.grabCurrentWorldOnBlock;
        const isEndPush = isCompetitiveInwardEndPush({
          gameMode: this.gameMode,
          turnPhase:
            this.turnPhase === 'SELECT_BLOCK' && this.localOptimisticGrabBlockId !== null
              ? 'PULLING'
              : this.turnPhase,
          blockRotation: trackedBlk.rotation,
          grabLocalPointDm: localPt,
          grabCurrentWorldOnBlockDm: curOnBlk,
          grabAnchorStartDm: this.localOptimisticPullStartAnchor ?? undefined,
          targetWorldDm: targetPt,
        });
        if (isEndPush && frame.hostElapsedSec !== this.localOptimisticGrabLastSmoothHostSec) {
          const dtSmooth =
            this.localOptimisticGrabLastSmoothHostSec >= 0 &&
            frame.hostElapsedSec > this.localOptimisticGrabLastSmoothHostSec
              ? frame.hostElapsedSec - this.localOptimisticGrabLastSmoothHostSec
              : prevHostElapsedSec >= 0 && frame.hostElapsedSec > prevHostElapsedSec
                ? frame.hostElapsedSec - prevHostElapsedSec
                : 1 / 30;
          this.localOptimisticGrabSmoothedClampN = updateEndPushSmoothedClampingLoadN(
            this.localOptimisticGrabSmoothedClampN,
            this.computeBlockClampingLoadN(trackedId),
            dtSmooth
          );
          this.localOptimisticGrabLastSmoothHostSec = frame.hostElapsedSec;
        }
      }
    }

    this.splinters = frame.splinters;
    return true;
  }

  private rebuildOpenTopSlotsFromSupportFrame(): void {
    const f = this.topSupportFrame;
    const slots: TopPlacementSlot[] = [];
    for (let s = 0; s < 3; s++) {
      const offset = (s - 1) * 0.252;
      slots.push({
        layer: f.targetLayer,
        slotIndex: s,
        position: [
          f.layerCenter[0] + f.spacingAxis[0] * offset,
          f.layerCenter[1] + f.spacingAxis[1] * offset,
          f.layerCenter[2] + f.spacingAxis[2] * offset,
        ],
        rotation: [...f.defaultRotation],
        yawRad: f.defaultYawRad,
      });
    }
    this.openTopSlots = slots;
  }

  private rebuildShardList(): void {
    if (!this.hasCurrentSeedStateFrame) {
      this.shards = [];
      return;
    }
    const next: GlassShardState[] = [];
    for (const [id, entry] of this.shardMap.entries()) {
      if (!entry.hasPose) continue;
      next.push({
        id,
        slotIndex: entry.meta.slotIndex,
        bodyHandle: 0,
        colliderHandle: 0,
        swatchId: entry.meta.swatchId,
        motif: entry.meta.motif,
        rippleSeed: entry.meta.rippleSeed,
        position: entry.currentPos,
        rotation: entry.currentQuat,
        prePos: entry.currentPos,
        preQuat: entry.currentQuat,
        preLinVel: [0, 0, 0],
        preAngVel: [0, 0, 0],
        volume: entry.meta.volume ?? 0.001,
        massKg: entry.meta.massKg ?? 0.0025,
        preVy: 0,
        impactLocalPos: entry.meta.impactLocalPos,
        planes: entry.planesFloat32,
        planeCount: entry.meta.planeCount,
        firstVertex: entry.firstVertex,
        vertexCount: entry.vertexCount,
        quietSubsteps: 0,
      });
    }
      this.shards = next;
  }

  /**
   * Advances exponential + hemisphere-safe quaternion slerp interpolation (`~36 ms` time constant)
   * at the local display's native refresh rate (`60–120 Hz`).
   */
  public step(frameDt: number): void {
    const dt = Math.max(0.0005, Math.min(0.08, frameDt));
    const alpha = 1.0 - Math.exp(-dt / this.interpolationTauSec);

    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered) continue;
      const tp = this.targetBlockPos[i];
      const tq = this.targetBlockQuat[i];
      if (!tp || !tq) continue;

      b.position[0] += (tp[0] - b.position[0]) * alpha;
      b.position[1] += (tp[1] - b.position[1]) * alpha;
      b.position[2] += (tp[2] - b.position[2]) * alpha;
      b.rotation = quatSlerpHemisphere(b.rotation, tq, alpha);
    }

    for (const entry of this.shardMap.values()) {
      if (!entry.hasPose) continue;
      entry.currentPos[0] += (entry.targetPos[0] - entry.currentPos[0]) * alpha;
      entry.currentPos[1] += (entry.targetPos[1] - entry.currentPos[1]) * alpha;
      entry.currentPos[2] += (entry.targetPos[2] - entry.currentPos[2]) * alpha;
      entry.currentQuat = quatSlerpHemisphere(entry.currentQuat, entry.targetQuat, alpha);
    }

    this.grabCurrentWorldOnBlock[0] +=
      (this.targetGrabOnBlock[0] - this.grabCurrentWorldOnBlock[0]) * alpha;
    this.grabCurrentWorldOnBlock[1] +=
      (this.targetGrabOnBlock[1] - this.grabCurrentWorldOnBlock[1]) * alpha;
    this.grabCurrentWorldOnBlock[2] +=
      (this.targetGrabOnBlock[2] - this.grabCurrentWorldOnBlock[2]) * alpha;

    if (this.localOptimisticPullAnchor) {
      this.grabWorldAnchor[0] = this.localOptimisticPullAnchor[0];
      this.grabWorldAnchor[1] = this.localOptimisticPullAnchor[1];
      this.grabWorldAnchor[2] = this.localOptimisticPullAnchor[2];
    } else {
      this.grabWorldAnchor[0] += (this.targetGrabAnchor[0] - this.grabWorldAnchor[0]) * alpha;
      this.grabWorldAnchor[1] += (this.targetGrabAnchor[1] - this.grabWorldAnchor[1]) * alpha;
      this.grabWorldAnchor[2] += (this.targetGrabAnchor[2] - this.grabWorldAnchor[2]) * alpha;
    }

    this.carriedLandingPos[0] +=
      (this.targetCarriedLandingPos[0] - this.carriedLandingPos[0]) * alpha;
    this.carriedLandingPos[1] +=
      (this.targetCarriedLandingPos[1] - this.carriedLandingPos[1]) * alpha;
    this.carriedLandingPos[2] +=
      (this.targetCarriedLandingPos[2] - this.carriedLandingPos[2]) * alpha;
    this.carriedTargetQuat = quatSlerpHemisphere(
      this.carriedTargetQuat,
      this.targetCarriedQuat,
      alpha
    );
  }

  public setOptimisticGuestHover(
    blockId: number | null,
    localPoint: Vec3 | null = null
  ): void {
    this.localGuestHoverBlockId = blockId;
    this.localGuestHoverLocalPoint =
      blockId !== null && localPoint ? [...localPoint] : null;
    if (this.currentPlayer === 1) {
      this.hoveredBlockId = blockId;
      this.hoverLocalPoint = this.localGuestHoverLocalPoint
        ? [...this.localGuestHoverLocalPoint]
        : null;
    }
  }

  private clearOptimisticGrabState(): void {
    this.localOptimisticPullAnchor = null;
    this.localOptimisticPullStartAnchor = null;
    this.localOptimisticGrabBlockId = null;
    this.localOptimisticGrabLocalPoint = [0, 0, 0];
    this.localOptimisticGrabBlockStartPos = null;
    this.localOptimisticGrabInitialClampN = 0;
    this.localOptimisticGrabSmoothedClampN = 0;
    this.localOptimisticGrabLastSmoothHostSec = -1;
    this.localOptimisticGrabConfirmedByHost = false;
    this.localOptimisticGrabPreAckFrames = 0;
    this.localOptimisticGrabStartedAtMs = 0;
  }

  private getEffectiveGuestGrabClampingLoadN(
    blkId: number,
    blkPos: Vec3,
    isEndPush: boolean = false
  ): number {
    const blk = this.blocks[blkId];
    const liveLoadN = this.computeBlockClampingLoadN(blkId);
    let wrenchSum = 0;
    if (blk?.isLooseSlot) {
      const wBase = blkId * 4;
      wrenchSum =
        wBase + 2 < this.blockPhotoelasticWrench.length
          ? (this.blockPhotoelasticWrench[wBase + 0] ?? 0) +
            (this.blockPhotoelasticWrench[wBase + 1] ?? 0) +
            (this.blockPhotoelasticWrench[wBase + 2] ?? 0)
          : 0;
    }
    return computeEffectiveGrabClampingLoadN({
      liveLoadN,
      isLooseSlot: Boolean(blk?.isLooseSlot),
      wrenchSum,
      blockPos: blkPos,
      grabStartBlockPos: this.localOptimisticGrabBlockStartPos,
      grabInitialClampingLoadN: this.localOptimisticGrabInitialClampN,
      isEndPush,
      smoothedLoadN: isEndPush ? this.localOptimisticGrabSmoothedClampN : undefined,
    });
  }

  public setOptimisticGuestGrab(
    blockId: number | null,
    localPoint?: Vec3,
    worldAnchor?: Vec3
  ): void {
    this.lastExtractionSlip = null;
    if (blockId !== null && localPoint && worldAnchor) {
      this.localOptimisticGrabBlockId = blockId;
      this.localOptimisticGrabLocalPoint = [...localPoint];
      this.localOptimisticPullStartAnchor = [...worldAnchor];
      this.localOptimisticPullAnchor = [...worldAnchor];
      this.localOptimisticGrabConfirmedByHost = false;
      this.localOptimisticGrabPreAckFrames = 0;
      this.localOptimisticGrabStartedAtMs =
        typeof performance !== 'undefined' ? performance.now() : Date.now();
      const gBlk = this.blocks[blockId];
      this.localOptimisticGrabBlockStartPos = null;
      this.localOptimisticGrabInitialClampN = 0;
      this.localOptimisticGrabInitialClampN = gBlk
        ? this.getEffectiveGuestGrabClampingLoadN(blockId, gBlk.position)
        : 0;
      this.localOptimisticGrabSmoothedClampN = this.localOptimisticGrabInitialClampN;
      this.localOptimisticGrabLastSmoothHostSec = this.lastAcceptedHostElapsedSec;
      this.localOptimisticGrabBlockStartPos = gBlk ? [...gBlk.position] : null;
      if (this.currentPlayer === 1) {
        this.activeBlockId = blockId;
        this.grabLocalPoint = [...localPoint];
        this.grabWorldAnchor = [...worldAnchor];
        this.grabCurrentWorldOnBlock = [...worldAnchor];
        this.targetGrabOnBlock = [...worldAnchor];
        this.targetGrabAnchor = [...worldAnchor];
        this.grabPlaneY = worldAnchor[1];
        this.isPointerDraggingBlock = true;
      }
    } else {
      this.clearOptimisticGrabState();
      if (this.currentPlayer === 1 && this.turnPhase !== 'EXTRACTED') {
        this.targetGrabOnBlock = [0, 0, 0];
        this.targetGrabAnchor = [0, 0, 0];
      }
    }
  }

  public setOptimisticGuestPullAnchor(worldAnchor: Vec3 | null): void {
    this.localOptimisticPullAnchor = worldAnchor ? [...worldAnchor] : null;
    if (worldAnchor) {
      if (!this.localOptimisticPullStartAnchor) {
        this.localOptimisticPullStartAnchor = [...worldAnchor];
      }
      this.grabWorldAnchor = [...worldAnchor];
      this.targetGrabAnchor = [...worldAnchor];
      this.grabPlaneY = worldAnchor[1];
    } else {
      this.clearOptimisticGrabState();
      if (this.currentPlayer === 1 && this.turnPhase !== 'EXTRACTED') {
        this.isPointerDraggingBlock = false;
        this.activeBlockId = null;
        this.targetGrabOnBlock = [0, 0, 0];
        this.targetGrabAnchor = [0, 0, 0];
        if (this.turnPhase === 'PULLING') {
          this.turnPhase = 'SELECT_BLOCK';
        }
      }
    }
  }

  public computeBlockClampingLoadN(blockId: number): number {
    const blk = this.blocks[blockId];
    if (!blk || blk.shattered || blk.extracted) return 0;
    if (
      this.hasCurrentSeedStateFrame &&
      blockId >= 0 &&
      blockId < this.blockClampingLoadsN.length
    ) {
      const wireLoad = this.blockClampingLoadsN[blockId];
      if (Number.isFinite(wireLoad)) {
        return wireLoad;
      }
    }
    return estimateClampingLoadFromWrenchAndStack(
      blockId,
      this.blocks,
      this.blockPhotoelasticWrench
    );
  }

  public getLastEvaluatedClampingLoadN(): number {
    return this.lastEvaluatedClampingLoadN;
  }

  /**
   * Evaluates Track 4 competitive extraction slip on the Guest using identical physical
   * thresholds (`F_SLIP = 3.42 N`, `F_PUSH_MAX = 18.0 N`) as the authoritative Host.
   */
  public evaluateGuestDragSlip(targetWorldOnPlane: Vec3): ExtractionSlipResult | null {
    const blkId = this.localOptimisticGrabBlockId ?? this.activeBlockId;
    if (blkId === null) return null;
    const blk = this.blocks[blkId];
    if (!blk || blk.shattered) return null;

    if (this.localOptimisticGrabBlockStartPos === null) {
      this.localOptimisticGrabInitialClampN = this.getEffectiveGuestGrabClampingLoadN(
        blk.id,
        blk.position
      );
      this.localOptimisticGrabSmoothedClampN = this.localOptimisticGrabInitialClampN;
      this.localOptimisticGrabLastSmoothHostSec = this.lastAcceptedHostElapsedSec;
      this.localOptimisticGrabBlockStartPos = [...blk.position];
    }
    const localPt =
      this.localOptimisticGrabBlockId === blkId
        ? this.localOptimisticGrabLocalPoint
        : this.grabLocalPoint;
    const currentOnBlock =
      this.grabCurrentWorldOnBlock[0] === 0 &&
      this.grabCurrentWorldOnBlock[1] === 0 &&
      this.grabCurrentWorldOnBlock[2] === 0 &&
      this.localOptimisticPullStartAnchor
        ? this.localOptimisticPullStartAnchor
        : this.grabCurrentWorldOnBlock;
    const evalTurnPhase =
      this.turnPhase === 'SELECT_BLOCK' && this.localOptimisticGrabBlockId !== null
        ? 'PULLING'
        : this.turnPhase;
    const isEndPush = isCompetitiveInwardEndPush({
      gameMode: this.gameMode,
      turnPhase: evalTurnPhase,
      blockRotation: blk.rotation,
      grabLocalPointDm: localPt,
      grabCurrentWorldOnBlockDm: currentOnBlock,
      grabAnchorStartDm: this.localOptimisticPullStartAnchor ?? undefined,
      targetWorldDm: targetWorldOnPlane,
    });
    const clampingLoadN = this.getEffectiveGuestGrabClampingLoadN(blk.id, blk.position, isEndPush);
    this.lastEvaluatedClampingLoadN = clampingLoadN;

    const res = evaluateCompetitiveDragSlip({
      gameMode: this.gameMode,
      turnPhase: evalTurnPhase,
      blockRotation: blk.rotation,
      grabLocalPointDm: localPt,
      grabCurrentWorldOnBlockDm: currentOnBlock,
      grabAnchorStartDm: this.localOptimisticPullStartAnchor ?? undefined,
      targetWorldDm: targetWorldOnPlane,
      clampingLoadN,
      grabResistance01: this.grabResistance01,
    });
    const sameRowOtherCount = countOtherBlocksInLayer(this.blocks, blk);
    const normalized = normalizeCompetitiveSlipResult(res, sameRowOtherCount);
    this.lastExtractionSlip = normalized;
    return normalized;
  }

  public getLastAcceptedSeq(): number {
    return this.lastAcceptedSeq;
  }

  public getAcceptedFrameCount(): number {
    return this.acceptedFrameCount;
  }

  public getRejectedFrameCount(): number {
    return this.rejectedFrameCount;
  }

  public getLastAcceptedAudioBatchSeq(): number {
    return this.audioBatchSeqFloor;
  }

  public getDuplicateAudioBatchCount(): number {
    return this.duplicateAudioBatchCount;
  }

  public hasCompletedInitialWelcome(): boolean {
    return this.hasReceivedWelcome;
  }

  public replayNetworkAudioEvents(
    events: NetworkAudioEvent[],
    batchSeq?: number,
    roomSeed?: number
  ): boolean {
    if (
      roomSeed !== undefined &&
      roomSeed !== 0 &&
      (roomSeed >>> 0) !== this.roomSeed
    ) {
      this.duplicateAudioBatchCount++;
      return false;
    }
    if (batchSeq !== undefined) {
      const cleanBatch = batchSeq >>> 0;
      if (
        !Number.isFinite(batchSeq) ||
        cleanBatch <= this.audioBatchSeqFloor ||
        this.receivedAudioBatchSeqs.has(cleanBatch)
      ) {
        this.duplicateAudioBatchCount++;
        return false;
      }
      this.receivedAudioBatchSeqs.add(cleanBatch);
      while (this.receivedAudioBatchSeqs.has((this.audioBatchSeqFloor + 1) >>> 0)) {
        this.audioBatchSeqFloor = (this.audioBatchSeqFloor + 1) >>> 0;
        this.receivedAudioBatchSeqs.delete(this.audioBatchSeqFloor);
      }
      if (this.receivedAudioBatchSeqs.size > 256) {
        const sorted = Array.from(this.receivedAudioBatchSeqs).sort((a, b) => a - b);
        const trimTo = sorted[sorted.length - 128];
        this.audioBatchSeqFloor = Math.max(this.audioBatchSeqFloor, trimTo);
        for (const seq of sorted) {
          if (seq <= this.audioBatchSeqFloor) {
            this.receivedAudioBatchSeqs.delete(seq);
          }
        }
      }
    }
    const ext = this.audio as unknown as ExtendedGuestAudioHooks;
    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      switch (ev.kind) {
        case 'slideFriction':
          ext.updateSlideFriction(
            ev.slideSpeedDmPerSec,
            ev.resistance01,
            ev.slipPowerMw,
            ev.torsionRadPerSec,
            ev.clampingLoadN,
            ev.seamCrossingIntensity
          );
          break;
        case 'tapTest':
          ext.playTapTestResonance(
            ev.pinch01,
            ev.worldPos,
            this.listenerCamPos,
            this.listenerCamRight,
            ev.damage
          );
          break;
        case 'glassImpact':
          ext.playGlassImpact(
            ev.intensity01,
            ev.pitchMul,
            ev.worldPos,
            this.listenerCamPos,
            this.listenerCamRight,
            {
              impactSpeedMps: ev.impactSpeedMps,
              damage: ev.damage,
              surface: ev.surface,
            }
          );
          break;
        case 'blockThud':
          ext.playBlockThud?.(
            ev.impactSpeedMps,
            ev.worldPos,
            {
              camPos: this.listenerCamPos,
              camRight: this.listenerCamRight,
              damage: ev.damage,
            }
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
        case 'shardImpacts':
          ext.playShardImpacts?.(ev.impacts, this.listenerCamPos, this.listenerCamRight);
          break;
        case 'wobbleRock':
          ext.playWobbleRock?.(
            ev.tiltAngleRad,
            ev.angularSpeedRadPerSec,
            ev.clampingLoadN,
            ev.worldPos,
            {
              blockId: ev.blockId,
              surface: ev.surface,
              damage: ev.damage,
              camPos: this.listenerCamPos,
              camRight: this.listenerCamRight,
            }
          );
          break;
        case 'stackSuccess':
          this.audio.playStackSuccess(ev.layerCount);
          break;
      }
    }
    return true;
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

  public getTelemetry(): TowerTelemetry {
    let crazedCount = 0;
    let intactCount = 0;
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
      centerOfMassXZ: [...this.centerOfMassXZCm],
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
      settleProgress01: +this.settleProgress01.toFixed(2),
      collapseReason: this.collapseReason,
      physicsWorkerActive: false,
    };
  }

  public dispose(): void {
    this.audio.updateSlideFriction(0, 0);
    this.shardMap.clear();
    this.pendingShardUploads = [];
  }
}

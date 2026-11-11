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
  applyDeterministicTowerPalette,
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
    clampingLoadN?: number
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

  public gameMode: GameMode = 'party';
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
  private snapNextPose: boolean = true;

  private localGuestHoverBlockId: number | null = null;
  private localOptimisticPullAnchor: Vec3 | null = null;
  private hostHoverBlockId: number | null = null;

  constructor(audio: GlassAudioEngine, initialLayers: number = INITIAL_LAYERS, roomSeed: number = 0x56495452) {
    this.audio = audio;
    this.initDeterministicTower(initialLayers, roomSeed);
  }

  public initDeterministicTower(layers: number = INITIAL_LAYERS, roomSeed: number = 0x56495452): void {
    this.roomSeed = roomSeed >>> 0;
    const total = layers * 3;
    this.blocks = [];
    this.targetBlockPos = [];
    this.targetBlockQuat = [];
    this.shards = [];
    this.splinters = [];
    this.shardMap.clear();
    this.pendingShardUploads = [];
    this.lastAcceptedSeq = 0;
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
    this.localOptimisticPullAnchor = null;

    this.blockPhotoelasticStress01 = new Float32Array(total + 32);
    this.blockPhotoelasticWrench = new Float32Array((total + 32) * 4);

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

    applyDeterministicTowerPalette(this.blocks, this.roomSeed);
    this.rebuildOpenTopSlotsFromSupportFrame();
  }

  public applyRoomWelcome(params: {
    roomSeed: number;
    initialLayers: number;
    collapseReason: string | null;
    activeShardMetas: SerializedShardMeta[];
    retiredShardIds: number[];
  }): void {
    if (
      this.roomSeed !== (params.roomSeed >>> 0) ||
      this.blocks.length !== params.initialLayers * 3
    ) {
      this.initDeterministicTower(params.initialLayers, params.roomSeed);
    } else {
      applyDeterministicTowerPalette(this.blocks, params.roomSeed);
      this.shardMap.clear();
      this.shards = [];
      this.lastAcceptedSeq = 0;
      this.snapNextPose = true;
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
      if (this.shardMap.has(meta.id)) continue;
      const built = reconstructShardVerticesFromMeta(meta);
      const firstVertex = meta.slotIndex * SHARD_SLOT_STRIDE_VERTS;
      this.pendingShardUploads.push({
        firstVertex,
        vertexData: built.renderVertices,
      });
      this.shardMap.set(meta.id, {
        meta,
        planesFloat32: built.planesFloat32,
        firstVertex,
        vertexCount: built.vertexCount,
        currentPos: [0, -10, 0],
        currentQuat: [0, 0, 0, 1],
        targetPos: [0, -10, 0],
        targetQuat: [0, 0, 0, 1],
        hasPose: false,
      });
    }
    this.rebuildShardList();
  }

  public applyNetworkStateFrame(frame: MultiplayerStateFrame): boolean {
    const seedChanged = (frame.roomSeed >>> 0) !== this.roomSeed && frame.roomSeed !== 0;
    if (seedChanged) {
      const layers = Math.max(1, Math.round(frame.blocks.length / 3) || INITIAL_LAYERS);
      this.initDeterministicTower(layers, frame.roomSeed);
    } else if (!this.snapNextPose && frame.seq <= this.lastAcceptedSeq) {
      // Guard against out-of-order UDP packets on the unordered WebRTC DataChannel
      return false;
    }
    this.lastAcceptedSeq = frame.seq;

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

    if (this.currentPlayer !== 1 || !this.isPointerDraggingBlock) {
      this.localOptimisticPullAnchor = null;
    }

    this.hoveredBlockId =
      this.currentPlayer === 1 && this.localGuestHoverBlockId !== null
        ? this.localGuestHoverBlockId
        : this.hostHoverBlockId;

    this.targetGrabOnBlock = [...frame.grabCurrentWorldOnBlock];
    this.targetGrabAnchor = this.localOptimisticPullAnchor
      ? [...this.localOptimisticPullAnchor]
      : [...frame.grabWorldAnchor];
    this.grabPlaneY = frame.grabWorldAnchor[1];

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
      this.carriedTargetQuat = [...this.targetCarriedQuat];
    }

    const neededCap = frame.blocks.length + 8;
    if (this.blockPhotoelasticStress01.length < neededCap) {
      this.blockPhotoelasticStress01 = new Float32Array(neededCap);
      this.blockPhotoelasticWrench = new Float32Array(neededCap * 4);
    }

    for (let i = 0; i < frame.blocks.length; i++) {
      const fb = frame.blocks[i];
      let blk = this.blocks[i];
      if (!blk) {
        const mat = { swatchId: fb.swatchId, motif: fb.motif, rippleSeed: 1.0 + i * 0.618 };
        blk = {
          id: i,
          bodyHandle: i + 1,
          colliderHandle: i + 1,
          swatchId: mat.swatchId,
          motif: mat.motif,
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
        applyDeterministicTowerPalette([blk], this.roomSeed);
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
    }

    for (let i = 0; i < frame.shards.length; i++) {
      const fs = frame.shards[i];
      const entry = this.shardMap.get(fs.id);
      if (!entry) continue;
      entry.targetPos = [...fs.position];
      entry.targetQuat = [...fs.rotation];
      if (!entry.hasPose || snapImmediate) {
        entry.currentPos = [...fs.position];
        entry.currentQuat = [...fs.rotation];
        entry.hasPose = true;
      }
    }
    this.rebuildShardList();

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
        volume: 0.001,
        massKg: 0.0025,
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

  public setOptimisticGuestHover(blockId: number | null): void {
    this.localGuestHoverBlockId = blockId;
    if (this.currentPlayer === 1) {
      this.hoveredBlockId = blockId;
    }
  }

  public setOptimisticGuestPullAnchor(worldAnchor: Vec3 | null): void {
    this.localOptimisticPullAnchor = worldAnchor ? [...worldAnchor] : null;
    if (worldAnchor) {
      this.grabWorldAnchor = [...worldAnchor];
      this.grabPlaneY = worldAnchor[1];
    }
  }

  public replayNetworkAudioEvents(events: NetworkAudioEvent[]): void {
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
            ev.clampingLoadN
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

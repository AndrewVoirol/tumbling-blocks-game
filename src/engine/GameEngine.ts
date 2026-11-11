import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  INITIAL_LAYERS,
  LIGHTING_PRESETS,
  SHARD_SLOT_STRIDE_VERTS,
} from '../physics/constants';
import {
  BlockCriticalityCandidate,
  ensureRapierInitialized,
  GameMode,
  PhysicsWorld,
  ShardMeshAppendEvent,
  TowerTelemetry,
} from '../physics/PhysicsWorld';
import { MujocoWorkerBridge } from '../physics/MujocoWorkerBridge';
import { WebGPURenderer } from '../renderer/WebGPURenderer';
import { intersectRayOBB, quatRotateVec3, RayOBBHit, Vec3 } from '../math/vecMat';
import { OrbitCamera } from './OrbitCamera';
import {
  applyDeterministicTowerPalette,
  DecodedBlockFrameState,
  DecodedShardFrameState,
  generateRoomCode,
  generateRoomSeed,
  GuestRemoteCommand,
  inferShardLocalCentroid,
  isValidRoomCode,
  MP_MAX_SHARDS_PER_FRAME,
  MP_MAX_SPLINTERS_PER_FRAME,
  MultiplayerConnectionState,
  MultiplayerControlMessage,
  MultiplayerGuestWorld,
  MultiplayerRole,
  MultiplayerStateFrame,
  MultiplayerStatusSnapshot,
  MultiplayerTransportKind,
  NetworkAudioEvent,
  normalizeRoomCode,
  SerializedShardMeta,
  VitrailMultiplayerClient,
  VitrailMultiplayerClientOptions,
} from '../multiplayer';

export interface VitrailHudTelemetry extends TowerTelemetry {
  mpActive?: boolean;
  mpRole?: MultiplayerRole | null;
  mpRoomCode?: string | null;
  mpConnectionState?: MultiplayerConnectionState;
  mpTransport?: MultiplayerTransportKind;
  mpPeerConnected?: boolean;
  mpPingMs?: number | null;
  mpError?: string | null;
}

export type TelemetryListener = (telemetry: VitrailHudTelemetry) => void;

interface OptionalPhysicsHooks {
  consumeShardImpacts?: () => Array<{
    pos: Vec3;
    speed: number;
    massKg: number;
    surface: 'wood' | 'glass';
    slotIndex: number;
  }>;
  consumeWobbleEvents?: () => Array<{
    blockId: number;
    tiltAngleRad: number;
    angularSpeedRadPerSec: number;
    clampingLoadN: number;
    worldPos: Vec3;
    surface: 'glass' | 'wood';
    damage: number;
  }>;
  rebuildTower: (mode?: GameMode, players?: number, layers?: number, seed?: number) => void;
}

interface OptionalAcousticsHooks {
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
}

export type OnlineSessionOverrides = Omit<
  VitrailMultiplayerClientOptions,
  'role' | 'roomCode' | 'onControlMessage' | 'onStateFrame' | 'onStatusChange'
>;

export class GameEngine {
  private canvas: HTMLCanvasElement;
  public audio: GlassAudioEngine;
  public physics!: PhysicsWorld;
  public renderer!: WebGPURenderer;
  public camera: OrbitCamera;

  private rafId: number = 0;
  private running: boolean = false;
  private disposed: boolean = false;
  private lastFrameMs: number = 0;
  private elapsedSec: number = 0;
  private lastTelemetryEmitMs: number = 0;
  private listener: TelemetryListener | null = null;

  // Pointer interaction tracking
  private activePointerId: number | null = null;
  private activeTouches = new Map<number, [number, number]>();
  private lastPinchDist: number = 0;
  private pointerDownPos: [number, number] = [0, 0];
  private lastPointerPos: [number, number] = [0, 0];
  private pointerMovedDist: number = 0;
  private interactionMode: 'none' | 'orbit' | 'block_pending' | 'block' | 'place_top' = 'none';
  private pendingTapHit: { blockId: number; hit: RayOBBHit } | null = null;
  private grabAnchorOrigin: Vec3 = [0, 0, 0];

  public modalOpen: boolean = false;
  public benchPaused: boolean = false;
  public benchCameraLocked: boolean = false;
  public aiOpponentEnabled: boolean = false;
  public aiThinking: boolean = false;
  public aiActionLabel: string | null = null;
  public polariscopeMode: boolean = false;
  public physicsWorkerActive: boolean = false;
  private resizeObserver: ResizeObserver | null = null;

  // Master Glazier AI turn controller state
  private aiStage:
    | 'idle'
    | 'scouting'
    | 'awaiting_oracle'
    | 'probe_tight'
    | 'post_tap'
    | 'pulling'
    | 'placing' = 'idle';
  private aiTimerSec: number = 0;
  private aiTargetBlockId: number | null = null;
  private aiPullStartWorld: Vec3 = [0, 0, 0];
  private aiPullDirXZ: [number, number] = [1, 0];
  private aiHoverTargetXZ: [number, number] = [0, 0];
  private aiHoverCurrentXZ: [number, number] = [0, 0];

  // Internet Multiplayer (Host-Authoritative + Render-Interpolated Guest Mirror)
  private authoritativePhysics: PhysicsWorld | null = null;
  private authoritativeWorkerActive: boolean = false;
  private guestWorld: MultiplayerGuestWorld | null = null;
  private mpClient: VitrailMultiplayerClient | null = null;
  public mpRole: MultiplayerRole | null = null;
  public mpRoomCode: string | null = null;
  public mpRoomSeed: number = 0;
  private mpStatus: MultiplayerStatusSnapshot = {
    active: false,
    role: null,
    roomCode: null,
    roomSeed: 0,
    connectionState: 'idle',
    transport: 'none',
    peerConnected: false,
    pingMs: null,
    errorMessage: null,
  };
  private mpStateSeq: number = 0;
  private mpLastSnapshotSentMs: number = 0;
  private mpForceImmediateSnapshot: boolean = false;
  private mpLastCollapseReasonSent: string | null = null;
  private mpGuestCmdSeq: number = 0;
  private mpLastGuestCmdSentMs: number = 0;
  private mpLastSentGuestHoverId: number | null = null;
  private mpPendingPullTarget: Vec3 | null = null;

  // Host Voronoi shard & discrete spatial audio replication buffers
  private hostRawConsumeShardUploads: (() => ShardMeshAppendEvent[]) | null = null;
  private hostStagedRendererUploads: ShardMeshAppendEvent[] = [];
  private hostSlotLocalCentroids = new Map<number, Vec3>();
  private hostActiveShardMetas = new Map<number, SerializedShardMeta>();
  private hostAudioTapInstalled: boolean = false;
  private mpOutboundAudioQueue: NetworkAudioEvent[] = [];
  private mpLastSlideAudioSentMs: number = 0;
  private mpLastSlideWasNonZero: boolean = false;
  private scratchWrench4 = new Float32Array(4);

  constructor(canvas?: HTMLCanvasElement, onTelemetry?: TelemetryListener) {
    this.canvas = canvas ?? ({ style: {} } as HTMLCanvasElement);
    this.listener = onTelemetry ?? null;
    this.audio = new GlassAudioEngine();
    this.camera = new OrbitCamera();
  }

  /**
   * Initializes MuJoCo + PhysicsWorld + OrbitCamera without WebGPU for fast headless playability & multiplayer tests.
   */
  public async initHeadless(): Promise<void> {
    await ensureRapierInitialized();
    this.physics = new PhysicsWorld(this.audio);
    this.authoritativePhysics = this.physics;
    this.authoritativeWorkerActive = false;
    this.camera.update(0.016, 16 / 9);
    this.physics.setListenerPose(this.camera.position, this.camera.getRightVector());
  }

  public async init(): Promise<void> {
    this.renderer = new WebGPURenderer(this.canvas);
    await this.renderer.init();

    if (this.disposed) {
      this.renderer.dispose();
      return;
    }

    const params =
      typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
    const benchScene = params?.get('bench') ?? null;
    const disableWorker = benchScene !== null || params?.get('worker') === '0';

    if (!disableWorker && typeof Worker !== 'undefined') {
      try {
        const bridge = await MujocoWorkerBridge.create(this.audio);
        if (this.disposed) {
          bridge.dispose();
          this.renderer.dispose();
          return;
        }
        this.physics = bridge as unknown as PhysicsWorld;
        this.physicsWorkerActive = true;
      } catch {
        await ensureRapierInitialized();
        this.physics = new PhysicsWorld(this.audio);
        this.physicsWorkerActive = false;
      }
    } else {
      await ensureRapierInitialized();
      this.physics = new PhysicsWorld(this.audio);
      this.physicsWorkerActive = false;
    }

    this.authoritativePhysics = this.physics;
    this.authoritativeWorkerActive = this.physicsWorkerActive;

    if (benchScene && params) {
      const presetId = params.get('preset') ?? undefined;
      const darkParam = params.get('darkroom');
      const darkScale = darkParam !== null ? Number(darkParam) : undefined;
      this.configureBenchScene(benchScene, presetId, darkScale);
    } else {
      const roomParam = params?.get('room') ?? null;
      if (roomParam && isValidRoomCode(roomParam)) {
        this.joinOnlineRoom(roomParam);
      }
    }

    this.attachEvents();
    this.running = true;
    this.lastFrameMs = performance.now();
    this.emitTelemetry(true);
    this.rafId = requestAnimationFrame(this.tick);

    // Expose engine handle for DevTools runtime profiling & automated verification
    (window as unknown as { __VITRAIL_ENGINE__?: GameEngine }).__VITRAIL_ENGINE__ = this;
  }

  /**
   * Deterministic headless step helper for unit & two-peer multiplayer integration tests.
   */
  public stepHeadless(dt: number = 1 / 60): void {
    const clampedDt = Math.min(0.05, Math.max(0.001, dt));
    this.elapsedSec += clampedDt;
    if (!this.benchCameraLocked) {
      this.camera.update(clampedDt, 16 / 9);
    } else {
      this.camera.setPoseImmediate(
        this.camera.azimuth,
        this.camera.elevation,
        this.camera.distance,
        this.camera.target,
        16 / 9
      );
    }
    this.physics.setListenerPose(this.camera.position, this.camera.getRightVector());
    if (!this.benchPaused) {
      this.updateMasterGlazierAi(clampedDt);
      this.stepPhysicsAndSecondaryAcoustics(clampedDt);
    }
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    this.pumpMultiplayerStep(now);
  }

  private stepPhysicsAndSecondaryAcoustics(dt: number): void {
    this.physics.step(dt);

    if ((this.physics as unknown as { isRemoteGuestMirror?: boolean }).isRemoteGuestMirror) {
      return;
    }

    const physHooks = this.physics as unknown as OptionalPhysicsHooks;
    const audioHooks = this.audio as unknown as OptionalAcousticsHooks;

    if (typeof physHooks.consumeShardImpacts === 'function') {
      const impacts = physHooks.consumeShardImpacts();
      if (impacts && impacts.length > 0) {
        if (typeof audioHooks.playShardImpacts === 'function') {
          audioHooks.playShardImpacts(
            impacts,
            this.camera.position,
            this.camera.getRightVector()
          );
        } else if (this.mpRole === 'host' && this.mpStatus.peerConnected) {
          this.enqueueHostAudioEvent({
            kind: 'shardImpacts',
            impacts: impacts.slice(0, 8),
          });
        }
      }
    }

    if (typeof physHooks.consumeWobbleEvents === 'function') {
      const wobbles = physHooks.consumeWobbleEvents();
      if (wobbles && wobbles.length > 0) {
        for (let i = 0; i < wobbles.length; i++) {
          const w = wobbles[i];
          if (typeof audioHooks.playWobbleRock === 'function') {
            audioHooks.playWobbleRock(
              w.tiltAngleRad,
              w.angularSpeedRadPerSec,
              w.clampingLoadN,
              w.worldPos,
              {
                surface: w.surface,
                damage: w.damage,
                camPos: this.camera.position,
                camRight: this.camera.getRightVector(),
              }
            );
          } else if (this.mpRole === 'host' && this.mpStatus.peerConnected) {
            this.enqueueHostAudioEvent({
              kind: 'wobbleRock',
              tiltAngleRad: w.tiltAngleRad,
              angularSpeedRadPerSec: w.angularSpeedRadPerSec,
              clampingLoadN: w.clampingLoadN,
              worldPos: w.worldPos,
              surface: w.surface,
              damage: w.damage,
            });
          }
        }
      }
    }
  }

  private tick = (nowMs: number): void => {
    if (!this.running) return;
    const rawDt = (nowMs - this.lastFrameMs) / 1000;
    this.lastFrameMs = nowMs;
    const dt = Math.min(0.05, Math.max(0.001, rawDt));
    this.elapsedSec += dt;

    const cw = this.canvas.clientWidth || window.innerWidth;
    const ch = Math.max(1, this.canvas.clientHeight || window.innerHeight);
    if (cw > 0 && ch > 0) {
      this.renderer.resize(cw, ch);
    }
    const aspect = cw / ch;
    if (!this.benchCameraLocked) {
      this.camera.update(dt, aspect);
    } else {
      this.camera.setPoseImmediate(
        this.camera.azimuth,
        this.camera.elevation,
        this.camera.distance,
        this.camera.target,
        aspect
      );
    }
    this.physics.setListenerPose(this.camera.position, this.camera.getRightVector());

    if (!this.benchPaused) {
      this.updateMasterGlazierAi(dt);
      this.stepPhysicsAndSecondaryAcoustics(dt);
    }

    this.pumpMultiplayerStep(nowMs);

    // Smooth camera focus adjustment: frame top of tower when placing an extracted block!
    const topSlotY =
      this.physics.openTopSlots.length > 0
        ? this.physics.openTopSlots[0].position[1]
        : this.physics.currentLayers * 0.15;

    if (!this.benchCameraLocked) {
      if (this.physics.turnPhase === 'EXTRACTED') {
        this.camera.focusTopForPlacement(topSlotY);
      } else if (this.physics.turnPhase !== 'COLLAPSED') {
        this.camera.setFocusHeight(Math.max(1.32, topSlotY * 0.52));
      }
    }

    this.renderer.render(this.physics, this.camera, this.elapsedSec);

    if (nowMs - this.lastTelemetryEmitMs >= 45) {
      this.emitTelemetry(false);
      this.lastTelemetryEmitMs = nowMs;
    }

    this.rafId = requestAnimationFrame(this.tick);
  };

  private emitTelemetry(force: boolean): void {
    if (!this.listener || !this.physics) return;
    void force;
    const base = this.physics.getTelemetry();
    this.listener({
      ...base,
      aiOpponentEnabled: this.aiOpponentEnabled,
      aiThinking: this.aiThinking,
      aiActionLabel: this.aiActionLabel,
      polariscopeMode: this.polariscopeMode,
      physicsWorkerActive: this.physicsWorkerActive,
      mpActive: this.mpStatus.active,
      mpRole: this.mpRole,
      mpRoomCode: this.mpRoomCode,
      mpConnectionState: this.mpStatus.connectionState,
      mpTransport: this.mpStatus.transport,
      mpPeerConnected: this.mpStatus.peerConnected,
      mpPingMs: this.mpStatus.pingMs,
      mpError: this.mpStatus.errorMessage,
    });
  }

  public getTelemetrySnapshot(): VitrailHudTelemetry {
    const base = this.physics.getTelemetry();
    return {
      ...base,
      aiOpponentEnabled: this.aiOpponentEnabled,
      aiThinking: this.aiThinking,
      aiActionLabel: this.aiActionLabel,
      polariscopeMode: this.polariscopeMode,
      physicsWorkerActive: this.physicsWorkerActive,
      mpActive: this.mpStatus.active,
      mpRole: this.mpRole,
      mpRoomCode: this.mpRoomCode,
      mpConnectionState: this.mpStatus.connectionState,
      mpTransport: this.mpStatus.transport,
      mpPeerConnected: this.mpStatus.peerConnected,
      mpPingMs: this.mpStatus.pingMs,
      mpError: this.mpStatus.errorMessage,
    };
  }

  // ============================================================================
  // INTERNET MULTIPLAYER ORCHESTRATION (HOST + GUEST + TURN LOCK)
  // ============================================================================

  public getOnlineInviteUrl(): string | null {
    if (!this.mpRoomCode || typeof window === 'undefined' || !window.location) {
      return null;
    }
    const url = new URL(window.location.href);
    url.searchParams.delete('bench');
    url.searchParams.set('room', this.mpRoomCode);
    return url.toString();
  }

  private syncBrowserRoomQueryParam(code: string | null): void {
    if (typeof window === 'undefined' || !window.history?.replaceState || !window.location) {
      return;
    }
    try {
      const url = new URL(window.location.href);
      if (url.searchParams.has('bench')) return;
      if (code) {
        url.searchParams.set('room', code);
      } else {
        url.searchParams.delete('room');
      }
      window.history.replaceState(null, '', url.toString());
    } catch {
      // Ignore history security errors in sandboxed frames
    }
  }

  private restoreAuthoritativePhysics(): void {
    if (this.authoritativePhysics) {
      this.physics = this.authoritativePhysics;
      this.physicsWorkerActive = this.authoritativeWorkerActive;
    }
    if (this.guestWorld) {
      this.guestWorld.dispose();
      this.guestWorld = null;
    }
  }

  private shutdownMultiplayerSessionOnly(): void {
    if (this.mpClient) {
      this.mpClient.dispose();
      this.mpClient = null;
    }
    this.mpRole = null;
    this.mpRoomCode = null;
    this.mpStatus = {
      active: false,
      role: null,
      roomCode: null,
      roomSeed: 0,
      connectionState: 'idle',
      transport: 'none',
      peerConnected: false,
      pingMs: null,
      errorMessage: null,
    };
    this.mpOutboundAudioQueue = [];
    this.mpPendingPullTarget = null;
    this.mpLastSentGuestHoverId = null;
    this.hostStagedRendererUploads = [];
    this.hostSlotLocalCentroids.clear();
    this.hostActiveShardMetas.clear();
  }

  private rebuildAuthoritativeTowerWithSeed(seed: number): void {
    const physHooks = this.physics as unknown as OptionalPhysicsHooks;
    physHooks.rebuildTower('party', 2, INITIAL_LAYERS, seed >>> 0);
    applyDeterministicTowerPalette(this.physics.blocks, seed >>> 0);
    this.hostStagedRendererUploads = [];
    this.hostSlotLocalCentroids.clear();
    this.hostActiveShardMetas.clear();
    this.mpStateSeq = 0;
    this.mpLastCollapseReasonSent = null;
    this.mpForceImmediateSnapshot = true;
  }

  private installHostShardUploadHook(targetPhysics: PhysicsWorld): void {
    const tagged = targetPhysics as unknown as {
      _vitrailMpShardHooked?: boolean;
      consumePendingShardUploads: () => ShardMeshAppendEvent[];
    };
    if (tagged._vitrailMpShardHooked) return;

    const rawConsume = targetPhysics.consumePendingShardUploads.bind(targetPhysics);
    this.hostRawConsumeShardUploads = rawConsume;
    tagged._vitrailMpShardHooked = true;

    targetPhysics.consumePendingShardUploads = (): ShardMeshAppendEvent[] => {
      const fresh = rawConsume();
      if (fresh.length > 0) {
        this.recordShardUploadCentroids(fresh);
      }
      if (this.hostStagedRendererUploads.length > 0) {
        const combined =
          fresh.length > 0
            ? this.hostStagedRendererUploads.concat(fresh)
            : this.hostStagedRendererUploads;
        this.hostStagedRendererUploads = [];
        return combined;
      }
      return fresh;
    };
  }

  private recordShardUploadCentroids(uploads: ShardMeshAppendEvent[]): void {
    for (let i = 0; i < uploads.length; i++) {
      const ev = uploads[i];
      const slotIndex = Math.floor(ev.firstVertex / SHARD_SLOT_STRIDE_VERTS);
      if (ev.vertexData.length >= 9) {
        this.hostSlotLocalCentroids.set(slotIndex, [
          ev.vertexData[6] - ev.vertexData[0],
          ev.vertexData[7] - ev.vertexData[1],
          ev.vertexData[8] - ev.vertexData[2],
        ]);
      }
    }
  }

  private harvestHostShardUploadsBeforeNetwork(): void {
    if (!this.hostRawConsumeShardUploads) return;
    const fresh = this.hostRawConsumeShardUploads();
    if (fresh.length === 0) return;
    this.recordShardUploadCentroids(fresh);
    if (this.renderer) {
      for (let i = 0; i < fresh.length; i++) {
        this.hostStagedRendererUploads.push(fresh[i]);
      }
    }
  }

  private enqueueHostAudioEvent(ev: NetworkAudioEvent): void {
    if (this.mpRole !== 'host' || !this.mpStatus.peerConnected) return;
    if (this.mpOutboundAudioQueue.length < 32) {
      this.mpOutboundAudioQueue.push(ev);
    }
  }

  private installHostAudioTap(): void {
    if (this.hostAudioTapInstalled) return;
    this.hostAudioTapInstalled = true;

    const audioAny = this.audio as unknown as {
      updateSlideFriction: (
        speed: number,
        res: number,
        power?: number,
        torsion?: number,
        loadN?: number
      ) => void;
      playTapTestResonance: (
        pinch01: number,
        worldPos?: Vec3,
        camPos?: Vec3,
        camRight?: Vec3,
        damage?: number
      ) => void;
      playGlassImpact: (
        intensity01: number,
        pitchMul?: number,
        worldPos?: Vec3,
        camPos?: Vec3,
        camRight?: Vec3,
        options?: { impactSpeedMps?: number; damage?: number; surface?: 'glass' | 'wood' }
      ) => void;
      playGlassCrack: (
        damage01: number,
        worldPos?: Vec3,
        camPos?: Vec3,
        camRight?: Vec3
      ) => void;
      playGlassShatter: (
        intensity01: number,
        hitTable?: boolean,
        worldPos?: Vec3,
        camPos?: Vec3,
        camRight?: Vec3
      ) => void;
      playRankedShardCollisions: (
        events: Array<{
          energy: number;
          intensity01: number;
          volumeDm3: number;
          worldPos: Vec3;
          hitGlass: boolean;
        }>,
        camPos?: Vec3,
        camRight?: Vec3
      ) => void;
      playStackSuccess: (layerCount: number) => void;
      playShardImpacts?: OptionalAcousticsHooks['playShardImpacts'];
      playWobbleRock?: OptionalAcousticsHooks['playWobbleRock'];
    };

    const origSlide = audioAny.updateSlideFriction.bind(this.audio);
    audioAny.updateSlideFriction = (speed, res, power = 0, torsion = 0, loadN) => {
      origSlide(speed, res, power, torsion, loadN);
      if (this.mpRole !== 'host' || !this.mpStatus.peerConnected) return;
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const isZero = speed <= 0.001;
      if (isZero && this.mpLastSlideWasNonZero) {
        this.mpLastSlideWasNonZero = false;
        this.mpLastSlideAudioSentMs = now;
        this.enqueueHostAudioEvent({
          kind: 'slideFriction',
          slideSpeedDmPerSec: 0,
          resistance01: 0,
          slipPowerMw: 0,
          torsionRadPerSec: 0,
          clampingLoadN: loadN,
        });
      } else if (!isZero && now - this.mpLastSlideAudioSentMs >= 48) {
        this.mpLastSlideWasNonZero = true;
        this.mpLastSlideAudioSentMs = now;
        this.enqueueHostAudioEvent({
          kind: 'slideFriction',
          slideSpeedDmPerSec: speed,
          resistance01: res,
          slipPowerMw: power,
          torsionRadPerSec: torsion,
          clampingLoadN: loadN,
        });
      }
    };

    const origTap = audioAny.playTapTestResonance.bind(this.audio);
    audioAny.playTapTestResonance = (pinch01, worldPos, camPos, camRight, damage) => {
      origTap(pinch01, worldPos, camPos, camRight, damage);
      this.enqueueHostAudioEvent({ kind: 'tapTest', pinch01, worldPos, damage });
    };

    const origImpact = audioAny.playGlassImpact.bind(this.audio);
    audioAny.playGlassImpact = (intensity01, pitchMul = 1.0, worldPos, camPos, camRight, opts) => {
      origImpact(intensity01, pitchMul, worldPos, camPos, camRight, opts);
      this.enqueueHostAudioEvent({
        kind: 'glassImpact',
        intensity01,
        pitchMul,
        worldPos,
        impactSpeedMps: opts?.impactSpeedMps,
        damage: opts?.damage,
        surface: opts?.surface,
      });
    };

    const origCrack = audioAny.playGlassCrack.bind(this.audio);
    audioAny.playGlassCrack = (damage01, worldPos, camPos, camRight) => {
      origCrack(damage01, worldPos, camPos, camRight);
      this.enqueueHostAudioEvent({ kind: 'glassCrack', damage01, worldPos });
    };

    const origShatter = audioAny.playGlassShatter.bind(this.audio);
    audioAny.playGlassShatter = (intensity01, hitTable = false, worldPos, camPos, camRight) => {
      origShatter(intensity01, hitTable, worldPos, camPos, camRight);
      this.enqueueHostAudioEvent({ kind: 'glassShatter', intensity01, hitTable, worldPos });
    };

    const origRanked = audioAny.playRankedShardCollisions.bind(this.audio);
    audioAny.playRankedShardCollisions = (events, camPos, camRight) => {
      origRanked(events, camPos, camRight);
      if (events.length > 0) {
        this.enqueueHostAudioEvent({ kind: 'rankedShards', events: events.slice(0, 6) });
      }
    };

    const origStack = audioAny.playStackSuccess.bind(this.audio);
    audioAny.playStackSuccess = (layerCount) => {
      origStack(layerCount);
      this.enqueueHostAudioEvent({ kind: 'stackSuccess', layerCount });
    };

    if (typeof audioAny.playShardImpacts === 'function') {
      const origShardImpacts = audioAny.playShardImpacts.bind(this.audio);
      audioAny.playShardImpacts = (impacts, camPos, camRight) => {
        const played = origShardImpacts(impacts, camPos, camRight);
        if (impacts.length > 0) {
          this.enqueueHostAudioEvent({ kind: 'shardImpacts', impacts: impacts.slice(0, 8) });
        }
        return played;
      };
    }

    if (typeof audioAny.playWobbleRock === 'function') {
      const origWobble = audioAny.playWobbleRock.bind(this.audio);
      audioAny.playWobbleRock = (tilt, speed = 0, load = 0, worldPos = [0, 0, 0], opts) => {
        const played = origWobble(tilt, speed, load, worldPos, opts);
        this.enqueueHostAudioEvent({
          kind: 'wobbleRock',
          tiltAngleRad: tilt,
          angularSpeedRadPerSec: speed,
          clampingLoadN: load,
          worldPos,
          surface: opts?.surface ?? 'glass',
          damage: opts?.damage ?? 0,
        });
        return played;
      };
    }
  }

  /**
   * Hosts a new 2-player Online Vitrail room (Player 1 = Host, Player 2 = Guest).
   */
  public startOnlineHost(
    customRoomCode?: string,
    overrides?: OnlineSessionOverrides
  ): string {
    if (!this.physics && !this.authoritativePhysics) {
      throw new Error('GameEngine must be initialized before hosting an online room.');
    }
    this.handlePointerCancel();
    this.shutdownMultiplayerSessionOnly();
    this.restoreAuthoritativePhysics();

    this.aiOpponentEnabled = false;
    this.resetAiController();
    this.benchPaused = false;
    this.benchCameraLocked = false;

    const code =
      customRoomCode && isValidRoomCode(customRoomCode)
        ? normalizeRoomCode(customRoomCode)
        : generateRoomCode(5);
    const seed = overrides?.roomSeed !== undefined ? overrides.roomSeed >>> 0 : generateRoomSeed();

    this.mpRole = 'host';
    this.mpRoomCode = code;
    this.mpRoomSeed = seed;

    this.installHostShardUploadHook(this.physics);
    this.installHostAudioTap();
    this.rebuildAuthoritativeTowerWithSeed(seed);
    this.camera.resetView(27.0);
    this.syncBrowserRoomQueryParam(code);

    const client = new VitrailMultiplayerClient({
      ...overrides,
      role: 'host',
      roomCode: code,
      roomSeed: seed,
      onStatusChange: (status) => {
        this.mpStatus = status;
        this.emitTelemetry(true);
      },
      onPeerConnected: () => {
        this.sendHostRoomWelcome();
        this.mpForceImmediateSnapshot = true;
        this.pumpMultiplayerStep(
          typeof performance !== 'undefined' ? performance.now() : Date.now()
        );
        overrides?.onPeerConnected?.();
      },
      onPeerDisconnected: () => {
        if (this.physics.currentPlayer === 1 && this.physics.isPointerDraggingBlock) {
          this.physics.endGrab(false);
        }
        overrides?.onPeerDisconnected?.();
      },
      onControlMessage: (msg) => {
        this.handleHostControlMessage(msg);
      },
    });
    this.mpClient = client;
    client.connect();
    this.mpStatus = client.getStatus();
    this.emitTelemetry(true);
    return code;
  }

  /**
   * Joins an existing 2-player Online Vitrail room as Player 2 (`Guest`).
   */
  public joinOnlineRoom(
    rawRoomCode: string,
    overrides?: OnlineSessionOverrides
  ): boolean {
    const code = normalizeRoomCode(rawRoomCode);
    if (!isValidRoomCode(code)) {
      return false;
    }
    this.handlePointerCancel();
    this.shutdownMultiplayerSessionOnly();
    this.restoreAuthoritativePhysics();

    this.aiOpponentEnabled = false;
    this.resetAiController();
    this.benchPaused = false;
    this.benchCameraLocked = false;

    const initialSeed = overrides?.roomSeed ?? 0x56495452;
    const guestMirror = new MultiplayerGuestWorld(this.audio, INITIAL_LAYERS, initialSeed);
    guestMirror.setListenerPose(this.camera.position, this.camera.getRightVector());

    this.guestWorld = guestMirror;
    this.physics = guestMirror as unknown as PhysicsWorld;
    this.physicsWorkerActive = false;
    this.mpRole = 'guest';
    this.mpRoomCode = code;
    this.mpRoomSeed = guestMirror.roomSeed;
    this.camera.resetView(27.0);
    this.syncBrowserRoomQueryParam(code);

    const client = new VitrailMultiplayerClient({
      ...overrides,
      role: 'guest',
      roomCode: code,
      roomSeed: initialSeed,
      onStatusChange: (status) => {
        this.mpStatus = status;
        this.emitTelemetry(true);
      },
      onPeerConnected: () => {
        client.sendControl({ type: 'request_resync', lastSeq: 0 });
        overrides?.onPeerConnected?.();
      },
      onPeerDisconnected: () => {
        this.audio.updateSlideFriction(0, 0);
        this.guestWorld?.setOptimisticGuestPullAnchor(null);
        overrides?.onPeerDisconnected?.();
      },
      onControlMessage: (msg) => {
        this.handleGuestControlMessage(msg);
      },
      onStateFrame: (frame) => {
        this.handleGuestStateFrame(frame);
      },
    });
    this.mpClient = client;
    client.connect();
    this.mpStatus = client.getStatus();
    this.emitTelemetry(true);
    return true;
  }

  public leaveOnlineRoom(nextMode: GameMode = 'solo', players: number = 2): void {
    this.handlePointerCancel();
    this.shutdownMultiplayerSessionOnly();
    this.restoreAuthoritativePhysics();
    this.syncBrowserRoomQueryParam(null);
    this.resetGame(nextMode, players);
  }

  public requestOnlineRematch(): void {
    if (!this.mpRole || !this.mpClient) return;
    if (this.mpRole === 'host') {
      this.executeHostRematchReset();
    } else {
      this.mpClient.sendControl({ type: 'rematch_request' });
    }
  }

  private executeHostRematchReset(): void {
    if (this.mpRole !== 'host' || !this.mpClient) return;
    const retiredIds = Array.from(this.hostActiveShardMetas.keys());
    const nextSeed = generateRoomSeed();
    this.mpRoomSeed = nextSeed;
    this.mpClient.roomSeed = nextSeed;
    this.audio.updateSlideFriction(0, 0);
    this.rebuildAuthoritativeTowerWithSeed(nextSeed);
    this.camera.resetView(27.0);
    this.sendHostRoomWelcome(retiredIds);
    this.mpForceImmediateSnapshot = true;
    this.pumpMultiplayerStep(
      typeof performance !== 'undefined' ? performance.now() : Date.now()
    );
    this.emitTelemetry(true);
  }

  private sendChunkedShardSpawns(
    spawned: SerializedShardMeta[],
    retiredIds: number[]
  ): void {
    if (!this.mpClient || !this.mpStatus.peerConnected) return;
    const CHUNK_SIZE = 48; // Strictly < 16 KB per message (< 64 KB WebRTC SCTP maxMessageSize across all browsers)
    if (spawned.length === 0) {
      if (retiredIds.length > 0) {
        this.mpClient.sendControl({ type: 'shard_spawn', spawned: [], retiredIds });
      }
      return;
    }
    for (let i = 0; i < spawned.length; i += CHUNK_SIZE) {
      this.mpClient.sendControl({
        type: 'shard_spawn',
        spawned: spawned.slice(i, i + CHUNK_SIZE),
        retiredIds: i === 0 ? retiredIds : [],
      });
    }
  }

  private syncHostShardMetasAndBroadcast(): void {
    this.harvestHostShardUploadsBeforeNetwork();

    const currentIds = new Set<number>();
    const spawned: SerializedShardMeta[] = [];

    for (let i = 0; i < this.physics.shards.length; i++) {
      const s = this.physics.shards[i];
      currentIds.add(s.id);
      if (!this.hostActiveShardMetas.has(s.id)) {
        const cachedLc =
          this.hostSlotLocalCentroids.get(s.slotIndex) ??
          inferShardLocalCentroid(s.planes, s.planeCount, s.impactLocalPos);
        const planeFloats = Math.min(48, s.planeCount * 4);
        const planes: number[] = new Array(planeFloats);
        for (let k = 0; k < planeFloats; k++) {
          planes[k] = +s.planes[k].toFixed(4);
        }
        const meta: SerializedShardMeta = {
          id: s.id,
          slotIndex: s.slotIndex,
          swatchId: s.swatchId,
          motif: s.motif,
          rippleSeed: +s.rippleSeed.toFixed(4),
          localCentroid: [
            +cachedLc[0].toFixed(4),
            +cachedLc[1].toFixed(4),
            +cachedLc[2].toFixed(4),
          ],
          impactLocalPos: [
            +s.impactLocalPos[0].toFixed(4),
            +s.impactLocalPos[1].toFixed(4),
            +s.impactLocalPos[2].toFixed(4),
          ],
          planeCount: s.planeCount,
          planes,
        };
        this.hostActiveShardMetas.set(s.id, meta);
        spawned.push(meta);
      }
    }

    const retiredIds: number[] = [];
    for (const existingId of this.hostActiveShardMetas.keys()) {
      if (!currentIds.has(existingId)) {
        retiredIds.push(existingId);
        this.hostActiveShardMetas.delete(existingId);
      }
    }

    if (spawned.length > 0 || retiredIds.length > 0) {
      this.sendChunkedShardSpawns(spawned, retiredIds);
    }
  }

  private sendHostRoomWelcome(explicitRetiredIds: number[] = []): void {
    if (this.mpRole !== 'host' || !this.mpClient || !this.mpRoomCode) return;
    this.syncHostShardMetasAndBroadcast();
    const allActiveShards = Array.from(this.hostActiveShardMetas.values());
    this.mpClient.sendControl({
      type: 'room_welcome',
      roomCode: this.mpRoomCode,
      roomSeed: this.mpRoomSeed,
      initialLayers: INITIAL_LAYERS,
      collapseReason: this.physics.collapseReason,
      activeShardMetas: [],
      retiredShardIds: explicitRetiredIds,
    });
    if (allActiveShards.length > 0) {
      this.sendChunkedShardSpawns(allActiveShards, []);
    }
  }

  private buildHostStateFrame(): MultiplayerStateFrame {
    const blocksLen = this.physics.blocks.length;
    const blocks: DecodedBlockFrameState[] = new Array(blocksLen);
    const wScratch = this.scratchWrench4;

    for (let i = 0; i < blocksLen; i++) {
      const b = this.physics.blocks[i];
      this.physics.writeBlockPhotoelasticWrench(i, wScratch, 0);
      blocks[i] = {
        position: [b.position[0], b.position[1], b.position[2]],
        rotation: [b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]],
        damage: b.damage,
        shattered: b.shattered,
        extracted: b.extracted,
        isLegalTarget: b.isLegalTarget,
        isLooseSlot: b.isLooseSlot,
        swatchId: b.swatchId,
        motif: b.motif,
        assignedLayer: b.assignedLayer,
        assignedSlot: b.assignedSlot,
        stress01: this.physics.getBlockPhotoelasticStress01(i),
        crackLocalOrigin: [
          b.crackLocalOrigin[0],
          b.crackLocalOrigin[1],
          b.crackLocalOrigin[2],
        ],
        photoelasticWrench: [wScratch[0], wScratch[1], wScratch[2], wScratch[3]],
      };
    }

    const shardLimit = Math.min(MP_MAX_SHARDS_PER_FRAME, this.physics.shards.length);
    const shards: DecodedShardFrameState[] = new Array(shardLimit);
    for (let i = 0; i < shardLimit; i++) {
      const s = this.physics.shards[i];
      shards[i] = {
        id: s.id,
        position: [s.position[0], s.position[1], s.position[2]],
        rotation: [s.rotation[0], s.rotation[1], s.rotation[2], s.rotation[3]],
      };
    }

    const splinters = this.physics.splinters.slice(0, MP_MAX_SPLINTERS_PER_FRAME);
    const tf = this.physics.topSupportFrame;
    const tel = this.physics.getTelemetry();
    this.mpStateSeq = (this.mpStateSeq + 1) >>> 0;

    return {
      seq: this.mpStateSeq,
      hostElapsedSec: this.elapsedSec,
      roomSeed: this.mpRoomSeed,
      turnPhase: this.physics.turnPhase,
      gameMode: this.physics.gameMode,
      currentPlayer: this.physics.currentPlayer,
      playerCount: this.physics.playerCount,
      blocksStacked: this.physics.blocksStacked,
      shatteredCount: tel.shatteredCount,
      currentLayers: tel.currentLayers,
      peakLayers: tel.peakLayers,
      activeBlockId: this.physics.activeBlockId,
      hoveredBlockId: this.physics.hoveredBlockId,
      hoveredSlotIndex: this.physics.hoveredSlotIndex,
      lockedMinRow: tel.lockedMinRow,
      isPointerDraggingBlock: this.physics.isPointerDraggingBlock,
      isExtractionLiftReady: this.physics.isExtractionLiftReady,
      polariscopeMode: this.polariscopeMode,
      grabResistance01: this.physics.grabResistance01,
      settleProgress01: tel.settleProgress01,
      leanAngleDeg: tel.leanAngleDeg,
      wobbleSpeedDegPerSec: tel.wobbleSpeedDegPerSec,
      liveHeightCm: tel.liveHeightCm,
      peakHeightCm: tel.peakHeightCm,
      centerOfMassXZCm: [
        this.physics.centerOfMassXZCm[0],
        this.physics.centerOfMassXZCm[1],
      ],
      carriedYawOffsetRad: this.physics.carriedYawOffsetRad,
      grabCurrentWorldOnBlock: [...this.physics.grabCurrentWorldOnBlock],
      grabWorldAnchor: [...this.physics.grabWorldAnchor],
      carriedLandingPos: [...this.physics.carriedLandingPos],
      carriedTargetQuat: [...this.physics.carriedTargetQuat],
      topSupportSurfaceCenter: [...tf.surfaceCenter],
      topSupportUpNormal: [...tf.upNormal],
      topSupportDefaultRot: [...tf.defaultRotation],
      topSupportTargetLayer: tf.targetLayer,
      blocks,
      shards,
      splinters,
    };
  }

  private pumpMultiplayerStep(nowMs: number): void {
    if (!this.mpClient || !this.mpRole) return;

    if (this.mpRole === 'guest') {
      if (
        this.mpPendingPullTarget &&
        this.mpStatus.peerConnected &&
        nowMs - this.mpLastGuestCmdSentMs >= 15
      ) {
        const target = this.mpPendingPullTarget;
        this.mpPendingPullTarget = null;
        this.sendGuestRemoteCommand({ kind: 'update_pull', worldTarget: target }, nowMs);
      }
      return;
    }

    // Host role: synchronize new Voronoi shard geometries even while waiting for Guest so reconnects/joins have complete metadata
    this.syncHostShardMetasAndBroadcast();

    if (!this.mpStatus.peerConnected) {
      this.mpOutboundAudioQueue = [];
      return;
    }

    if (this.physics.collapseReason !== this.mpLastCollapseReasonSent) {
      this.mpLastCollapseReasonSent = this.physics.collapseReason;
      this.mpClient.sendControl({
        type: 'collapse_info',
        collapseReason: this.physics.collapseReason,
      });
    }

    if (this.mpOutboundAudioQueue.length > 0) {
      const batch = this.mpOutboundAudioQueue;
      this.mpOutboundAudioQueue = [];
      this.mpClient.sendControl({ type: 'audio_batch', events: batch });
    }

    // Stream compact `< 4 KB` binary snapshots at `30 Hz` (`~32 ms` cadence) or immediately on discrete state transitions
    if (this.mpForceImmediateSnapshot || nowMs - this.mpLastSnapshotSentMs >= 32) {
      this.mpForceImmediateSnapshot = false;
      this.mpLastSnapshotSentMs = nowMs;
      const frame = this.buildHostStateFrame();
      this.mpClient.sendStateFrame(frame);
    }
  }

  private handleHostControlMessage(msg: MultiplayerControlMessage): void {
    if (this.mpRole !== 'host' || !this.physics) return;
    switch (msg.type) {
      case 'hello':
      case 'request_resync': {
        this.sendHostRoomWelcome();
        this.mpForceImmediateSnapshot = true;
        this.pumpMultiplayerStep(
          typeof performance !== 'undefined' ? performance.now() : Date.now()
        );
        break;
      }
      case 'guest_cmd': {
        this.executeAuthoritativeGuestCommand(msg.cmd);
        break;
      }
      case 'rematch_request': {
        this.executeHostRematchReset();
        break;
      }
    }
  }

  /**
   * Authoritative Host command gate: strictly enforces turn ownership (`currentPlayer === 1`)
   * before applying any Guest pointer/keyboard action to the MuJoCo simulation.
   */
  private executeAuthoritativeGuestCommand(cmd: GuestRemoteCommand): void {
    if (this.mpRole !== 'host' || !this.physics) return;
    if (
      this.physics.currentPlayer !== 1 ||
      this.physics.turnPhase === 'COLLAPSED' ||
      this.physics.turnPhase === 'SETTLING'
    ) {
      return;
    }

    switch (cmd.kind) {
      case 'hover_block': {
        if (this.physics.turnPhase === 'SELECT_BLOCK') {
          const blk = cmd.blockId !== null ? this.physics.blocks[cmd.blockId] : undefined;
          this.physics.hoveredBlockId = blk && !blk.shattered ? blk.id : null;
        }
        break;
      }
      case 'tap_block': {
        if (this.physics.turnPhase === 'SELECT_BLOCK') {
          this.physics.tapNudgeBlock(cmd.blockId, cmd.localNormal, cmd.localPoint);
          this.mpForceImmediateSnapshot = true;
        }
        break;
      }
      case 'begin_pull': {
        if (this.physics.turnPhase === 'SELECT_BLOCK') {
          const started = this.physics.startGrabBlock(
            cmd.blockId,
            cmd.localPoint,
            cmd.worldPoint
          );
          if (started) {
            this.mpForceImmediateSnapshot = true;
            this.emitTelemetry(true);
          }
        }
        break;
      }
      case 'update_pull': {
        if (this.physics.turnPhase === 'PULLING') {
          this.physics.updateGrabTarget(cmd.worldTarget);
        }
        break;
      }
      case 'end_pull': {
        if (
          this.physics.turnPhase === 'PULLING' ||
          this.physics.turnPhase === 'EXTRACTED'
        ) {
          this.physics.endGrab(cmd.dropIfPoisedOverTower);
          this.mpForceImmediateSnapshot = true;
          this.emitTelemetry(true);
        }
        break;
      }
      case 'nudge_placement': {
        if (this.physics.turnPhase === 'EXTRACTED') {
          this.physics.updateCarriedHover(cmd.rayOrigin, cmd.rayDir);
        }
        break;
      }
      case 'rotate_placement': {
        if (this.physics.turnPhase === 'EXTRACTED') {
          this.physics.rotateCarriedBlock(cmd.deltaYawRad);
          this.mpForceImmediateSnapshot = true;
        }
        break;
      }
      case 'reset_rotation': {
        if (this.physics.turnPhase === 'EXTRACTED') {
          this.physics.resetCarriedBlockRotation();
          this.mpForceImmediateSnapshot = true;
        }
        break;
      }
      case 'confirm_placement': {
        if (
          this.physics.turnPhase === 'EXTRACTED' &&
          this.physics.isExtractionLiftReady
        ) {
          this.physics.releaseExtractedBlockOntoTop();
          this.mpForceImmediateSnapshot = true;
          this.emitTelemetry(true);
        }
        break;
      }
    }
  }

  private handleGuestControlMessage(msg: MultiplayerControlMessage): void {
    if (this.mpRole !== 'guest' || !this.guestWorld) return;
    switch (msg.type) {
      case 'room_welcome': {
        this.mpRoomSeed = msg.roomSeed >>> 0;
        this.guestWorld.applyRoomWelcome({
          roomSeed: msg.roomSeed,
          initialLayers: msg.initialLayers,
          collapseReason: msg.collapseReason,
          activeShardMetas: msg.activeShardMetas,
          retiredShardIds: msg.retiredShardIds,
        });
        this.emitTelemetry(true);
        break;
      }
      case 'shard_spawn': {
        this.guestWorld.applyShardSpawn(msg.spawned, msg.retiredIds);
        break;
      }
      case 'collapse_info': {
        this.guestWorld.collapseReason = msg.collapseReason;
        this.emitTelemetry(true);
        break;
      }
      case 'audio_batch': {
        this.guestWorld.setListenerPose(this.camera.position, this.camera.getRightVector());
        this.guestWorld.replayNetworkAudioEvents(msg.events);
        break;
      }
    }
  }

  private handleGuestStateFrame(frame: MultiplayerStateFrame): void {
    if (this.mpRole !== 'guest' || !this.guestWorld) return;
    const prevPhase = this.guestWorld.turnPhase;
    const prevPlayer = this.guestWorld.currentPlayer;
    const applied = this.guestWorld.applyNetworkStateFrame(frame);
    if (!applied) return;
    this.mpRoomSeed = this.guestWorld.roomSeed;

    if (
      prevPhase !== this.guestWorld.turnPhase ||
      prevPlayer !== this.guestWorld.currentPlayer
    ) {
      if (this.guestWorld.currentPlayer !== 1) {
        this.guestWorld.setOptimisticGuestHover(null);
        this.guestWorld.setOptimisticGuestPullAnchor(null);
        if (this.interactionMode !== 'orbit' && this.interactionMode !== 'none') {
          this.interactionMode = 'none';
          this.pendingTapHit = null;
        }
      }
      this.updateCursorStyle();
      this.emitTelemetry(true);
    }
  }

  private sendGuestRemoteCommand(cmd: GuestRemoteCommand, nowOverrideMs?: number): void {
    if (this.mpRole !== 'guest' || !this.mpClient || !this.mpStatus.peerConnected) return;
    this.mpGuestCmdSeq = (this.mpGuestCmdSeq + 1) >>> 0;
    this.mpLastGuestCmdSentMs =
      nowOverrideMs ?? (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this.mpClient.sendControl({
      type: 'guest_cmd',
      seq: this.mpGuestCmdSeq,
      cmd,
    });
  }

  private canLocalPlayerManipulateBlocks(): boolean {
    if (!this.physics) return false;
    if (this.isAiTurnActive()) return false;
    if (this.mpRole === 'host') {
      return (
        this.physics.currentPlayer === 0 &&
        this.physics.turnPhase !== 'COLLAPSED' &&
        this.physics.turnPhase !== 'SETTLING'
      );
    }
    if (this.mpRole === 'guest') {
      return (
        this.mpStatus.peerConnected &&
        this.physics.currentPlayer === 1 &&
        this.physics.turnPhase !== 'COLLAPSED' &&
        this.physics.turnPhase !== 'SETTLING'
      );
    }
    return true;
  }

  public setPolariscopeMode(enabled: boolean): void {
    this.polariscopeMode = enabled;
    if (this.renderer) {
      this.renderer.polariscopeMode = enabled;
    }
    this.emitTelemetry(true);
  }

  public setAiOpponentEnabled(enabled: boolean): void {
    if (enabled && this.mpRole) {
      this.leaveOnlineRoom('party', 2);
    }
    this.aiOpponentEnabled = enabled;
    this.resetAiController();
    if (enabled && this.physics && (this.physics.gameMode !== 'party' || this.physics.playerCount !== 2)) {
      this.resetGame('party', 2);
      return;
    }
    this.emitTelemetry(true);
  }

  private resetAiController(): void {
    this.aiStage = 'idle';
    this.aiTimerSec = 0;
    this.aiTargetBlockId = null;
    this.aiThinking = false;
    this.aiActionLabel = null;
  }

  private isAiTurnActive(): boolean {
    return (
      !this.mpRole &&
      this.aiOpponentEnabled &&
      !this.modalOpen &&
      Boolean(this.physics) &&
      this.physics.gameMode === 'party' &&
      this.physics.currentPlayer === 1 &&
      this.physics.turnPhase !== 'COLLAPSED' &&
      this.physics.turnPhase !== 'SETTLING'
    );
  }

  private requestAiCounterfactualOracle(): Promise<BlockCriticalityCandidate[]> {
    const maybeWorker = this.physics as unknown as {
      evaluateCounterfactualBlockCriticalityAsync?: (
        maxC: number,
        steps: number
      ) => Promise<BlockCriticalityCandidate[]>;
    };
    if (typeof maybeWorker.evaluateCounterfactualBlockCriticalityAsync === 'function') {
      return maybeWorker.evaluateCounterfactualBlockCriticalityAsync(12, 8);
    }
    return Promise.resolve(this.physics.evaluateCounterfactualBlockCriticality(12, 8));
  }

  private updateMasterGlazierAi(dt: number): void {
    if (!this.isAiTurnActive()) {
      if (this.aiStage !== 'idle') {
        this.resetAiController();
      }
      return;
    }

    if (this.aiStage === 'idle' && this.physics.turnPhase === 'SELECT_BLOCK') {
      this.aiStage = 'scouting';
      this.aiTimerSec = 0;
      this.aiThinking = true;
      this.aiActionLabel = 'Glazier AI Reading Stack…';
      this.emitTelemetry(true);
      return;
    }

    if (this.aiStage === 'scouting') {
      this.aiTimerSec += dt;
      if (this.aiTimerSec >= 0.80) {
        this.aiStage = 'awaiting_oracle';
        void this.requestAiCounterfactualOracle().then((candidates) => {
          if (!this.isAiTurnActive() || this.aiStage !== 'awaiting_oracle') return;
          const chosen =
            candidates.find((c) => !c.collapseTriggered && c.safetyScore01 > 0.35) ?? candidates[0];
          const blk = chosen ? this.physics.blocks[chosen.blockId] : undefined;
          if (!blk || blk.shattered || !blk.isLegalTarget) {
            this.resetAiController();
            return;
          }
          this.aiTargetBlockId = blk.id;

          // Pick a tighter/distinct candidate first so the player sees & hears the AI probe and reject a bearing column!
          const tighterProbe = candidates.find(
            (c) =>
              c.blockId !== blk.id &&
              this.physics.blocks[c.blockId] &&
              !this.physics.blocks[c.blockId].shattered &&
              this.physics.blocks[c.blockId].isLegalTarget &&
              (c.safetyScore01 < chosen.safetyScore01 - 0.14 ||
                this.physics.blocks[c.blockId].assignedLayer !== blk.assignedLayer)
          );
          if (tighterProbe) {
            this.physics.hoveredBlockId = tighterProbe.blockId;
            this.physics.tapNudgeBlock(tighterProbe.blockId, [1, 0, 0], [BLOCK_HALF_X * 0.82, 0, 0]);
            this.aiStage = 'probe_tight';
            this.aiTimerSec = 0;
            this.aiActionLabel = 'Glazier AI Testing Column…';
          } else {
            this.physics.hoveredBlockId = blk.id;
            this.physics.tapNudgeBlock(blk.id, [1, 0, 0], [BLOCK_HALF_X * 0.82, 0, 0]);
            this.aiStage = 'post_tap';
            this.aiTimerSec = 0;
            this.aiActionLabel = 'Glazier AI Selecting Prism…';
          }
          this.emitTelemetry(true);
        });
      }
      return;
    }

    if (this.aiStage === 'probe_tight') {
      this.aiTimerSec += dt;
      if (this.aiTimerSec >= 0.95) {
        const blk = this.aiTargetBlockId !== null ? this.physics.blocks[this.aiTargetBlockId] : undefined;
        if (!blk || blk.shattered || !blk.isLegalTarget) {
          this.resetAiController();
          return;
        }
        this.physics.hoveredBlockId = blk.id;
        this.physics.tapNudgeBlock(blk.id, [1, 0, 0], [BLOCK_HALF_X * 0.82, 0, 0]);
        this.aiStage = 'post_tap';
        this.aiTimerSec = 0;
        this.aiActionLabel = 'Glazier AI Selecting Prism…';
        this.emitTelemetry(true);
      }
      return;
    }

    if (this.aiStage === 'post_tap') {
      this.aiTimerSec += dt;
      if (this.aiTimerSec >= 0.90 && this.aiTargetBlockId !== null) {
        const blk = this.physics.blocks[this.aiTargetBlockId];
        if (!blk || blk.shattered) {
          this.resetAiController();
          return;
        }
        const axisX = quatRotateVec3(blk.rotation, [1, 0, 0]);
        const horizLen = Math.hypot(axisX[0], axisX[2]) || 1.0;
        const dirSign =
          blk.position[0] * axisX[0] + blk.position[2] * axisX[2] >= 0 ? 1.0 : -1.0;
        const pullX = (axisX[0] / horizLen) * dirSign;
        const pullZ = (axisX[2] / horizLen) * dirSign;
        const localGrab: Vec3 = [BLOCK_HALF_X * 0.84 * dirSign, 0, 0];
        const grabOffset = quatRotateVec3(blk.rotation, localGrab);
        const startWorld: Vec3 = [
          blk.position[0] + grabOffset[0],
          blk.position[1] + grabOffset[1],
          blk.position[2] + grabOffset[2],
        ];
        if (this.physics.startGrabBlock(blk.id, localGrab, startWorld)) {
          this.aiPullStartWorld = startWorld;
          this.aiPullDirXZ = [pullX, pullZ];
          this.aiStage = 'pulling';
          this.aiTimerSec = 0;
          this.aiThinking = false;
          this.aiActionLabel = 'Glazier AI Drawing…';
          this.emitTelemetry(true);
        } else {
          this.resetAiController();
        }
      }
      return;
    }

    if (this.aiStage === 'pulling') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        const comCm = this.physics.centerOfMassXZCm;
        const slots = this.physics.openTopSlots;
        let bestXZ: [number, number] = [
          this.physics.topSupportFrame.layerCenter[0],
          this.physics.topSupportFrame.layerCenter[2],
        ];
        if (slots.length > 0) {
          // Choose the open top slot that best counterbalances current tower CoM eccentricity
          let bestScore = Infinity;
          for (const sl of slots) {
            const projX = comCm[0] + sl.position[0] * 1.8;
            const projZ = comCm[1] + sl.position[2] * 1.8;
            const err = Math.hypot(projX, projZ);
            if (err < bestScore) {
              bestScore = err;
              bestXZ = [sl.position[0], sl.position[2]];
            }
          }
        }
        this.aiHoverCurrentXZ = [
          this.physics.carriedTargetPos[0],
          this.physics.carriedTargetPos[2],
        ];
        this.aiHoverTargetXZ = bestXZ;
        this.aiStage = 'placing';
        this.aiTimerSec = 0;
        this.aiActionLabel = 'Glazier AI Stacking…';
        this.emitTelemetry(true);
        return;
      }

      this.aiTimerSec += dt;
      const tPull = Math.min(1.0, this.aiTimerSec / 1.58);
      const easePull = tPull * tPull * (3.0 - 2.0 * tPull);
      const pullDistDm = 1.38 * easePull;
      const targetWorld: Vec3 = [
        this.aiPullStartWorld[0] + this.aiPullDirXZ[0] * pullDistDm,
        this.aiPullStartWorld[1],
        this.aiPullStartWorld[2] + this.aiPullDirXZ[1] * pullDistDm,
      ];
      this.physics.updateGrabTarget(targetWorld);
      if (this.aiTimerSec > 3.8 && this.physics.turnPhase === 'PULLING') {
        // If unexpectedly jammed, abort cleanly and re-scout
        this.physics.endGrab(false);
        this.aiStage = 'idle';
      }
      return;
    }

    if (this.aiStage === 'placing') {
      if (this.physics.turnPhase !== 'EXTRACTED') {
        this.resetAiController();
        return;
      }
      this.aiTimerSec += dt;
      const u = Math.min(1.0, this.aiTimerSec / 0.95);
      const smoothU = u * u * (3.0 - 2.0 * u);
      const hx =
        this.aiHoverCurrentXZ[0] + (this.aiHoverTargetXZ[0] - this.aiHoverCurrentXZ[0]) * smoothU;
      const hz =
        this.aiHoverCurrentXZ[1] + (this.aiHoverTargetXZ[1] - this.aiHoverCurrentXZ[1]) * smoothU;
      const topY = this.physics.topSupportFrame.layerCenter[1] + 1.0;
      this.physics.updateCarriedHover([hx, topY, hz], [0, -1, 0]);
      // Hold a visible 0.50 s alignment poise over the chosen landing slot before releasing (`0.95s -> 1.45s`)
      if (this.aiTimerSec >= 1.45 && this.physics.isExtractionLiftReady) {
        this.physics.releaseExtractedBlockOntoTop();
        this.resetAiController();
        this.emitTelemetry(true);
      }
    }
  }

  public resetGame(mode?: GameMode, players?: number): void {
    if (!this.physics) return;

    // In active Online mode, clicking Rebuild triggers a synchronized deterministic rematch
    if (this.mpRole && mode === undefined) {
      this.requestOnlineRematch();
      return;
    }
    if (this.mpRole) {
      this.shutdownMultiplayerSessionOnly();
      this.restoreAuthoritativePhysics();
      this.syncBrowserRoomQueryParam(null);
    }

    this.benchPaused = false;
    this.benchCameraLocked = false;
    this.resetAiController();
    if (mode !== undefined && mode !== 'party') {
      this.aiOpponentEnabled = false;
    }
    if (players !== undefined && players !== 2) {
      this.aiOpponentEnabled = false;
    }
    if (this.renderer) {
      this.renderer.lightingOverride = null;
      this.renderer.referenceCardLuminance = 0;
      this.renderer.benchHideBlocks = false;
    }
    this.interactionMode = 'none';
    this.activePointerId = null;
    this.activeTouches.clear();
    this.lastPinchDist = 0;
    this.pendingTapHit = null;
    this.audio.updateSlideFriction(0, 0);
    this.physics.rebuildTower(mode, players);
    this.camera.resetView(27.0);
    this.updateCursorStyle();
    this.emitTelemetry(true);
  }

  public setModalOpen(open: boolean): void {
    this.modalOpen = open;
    if (open) {
      this.handlePointerCancel();
    }
  }

  public setLightingPreset(presetId: string): void {
    const found = LIGHTING_PRESETS.find((p) => p.id === presetId);
    if (found && this.renderer) {
      this.renderer.lightingPreset = found;
    }
  }

  /**
   * Deterministic Tier-2 WebGPU Optical Bench configurator (`?bench=<scene>&preset=<id>`).
   */
  public configureBenchScene(
    scene: string,
    presetId?: string,
    darkroomSkyScale?: number
  ): void {
    if (!this.physics || !this.renderer) return;

    this.audio.setMuted(true);
    const preset =
      LIGHTING_PRESETS.find((p) => p.id === presetId) ?? LIGHTING_PRESETS[0];
    this.renderer.lightingPreset = preset;
    this.renderer.lightingOverride = null;
    this.renderer.referenceCardLuminance = 0;
    this.renderer.benchHideBlocks = false;
    this.benchPaused = true;
    this.benchCameraLocked = true;

    const aspect =
      (this.canvas.clientWidth || 1280) / Math.max(1, this.canvas.clientHeight || 800);

    if (scene === 'two-block-cross') {
      this.physics.rebuildTower('sandbox');
      const s = Math.SQRT1_2;
      // qHoriz: 7.5 cm (X) x 2.5 cm (Y) broad face facing +Z, 1.5 cm (0.15 dm) thick along Z
      const qHoriz: [number, number, number, number] = [s, 0, 0, s];
      // qVert: 2.5 cm (X) x 7.5 cm (Y) broad face facing +Z, 1.5 cm (0.15 dm) thick along Z
      const qVert: [number, number, number, number] = [0.5, 0.5, 0.5, 0.5];
      const qIdent: [number, number, number, number] = [0, 0, 0, 1];

      const specs: Array<{
        swatchId: number;
        pos: Vec3;
        rot: [number, number, number, number];
      }> = [
        // Pair 0 (Left): Selenium Ruby (#0) x Chartres Cobalt (#1)
        { swatchId: 0, pos: [-0.76, 0.80, 0.09], rot: qHoriz },
        { swatchId: 1, pos: [-0.76, 0.80, -0.09], rot: qVert },
        // Pair 1 (Center): Silver-Stain Amber (#2) x Chartres Cobalt (#1)
        { swatchId: 2, pos: [0.00, 0.80, 0.09], rot: qHoriz },
        { swatchId: 1, pos: [0.00, 0.80, -0.09], rot: qVert },
        // Pair 2 (Right): Stacked Chartres Cobalt (#1) x Chartres Cobalt (#1) sharing 1.25 mm bevel seam at y = 0.80
        { swatchId: 1, pos: [0.76, 0.725, 0.0], rot: qIdent },
        { swatchId: 1, pos: [0.76, 0.875, 0.0], rot: qIdent },
      ];

      for (let i = 0; i < this.physics.blocks.length; i++) {
        const blk = this.physics.blocks[i];
        if (i < specs.length) {
          const sp = specs[i];
          blk.shattered = false;
          blk.damage = 0;
          blk.swatchId = sp.swatchId;
          blk.motif = 0;
          blk.rippleSeed = 1.0 + i * 0.37;
          blk.position = [...sp.pos];
          blk.rotation = [...sp.rot];
        } else {
          blk.shattered = true;
        }
      }
      this.physics.hoveredBlockId = null;
      this.physics.activeBlockId = null;
      this.physics.openTopSlots = [];

      this.renderer.referenceCardLuminance = 0.80;
      this.renderer.lightingOverride = {
        sunAzimuthDeg: 135,
        sunElevationDeg: 40,
        sunColor: [0, 0, 0],
        skyTint: [0.032, 0.035, 0.042],
      };
      this.camera.setPoseImmediate(0, 0, 2.75, [0, 0.80, 0], aspect);
      return;
    }

    if (scene === 'backlit-tower') {
      this.physics.rebuildTower('solo');
      for (let i = 0; i < 100; i++) {
        this.physics.step(1 / 120);
      }
      if (darkroomSkyScale !== undefined && !Number.isNaN(darkroomSkyScale)) {
        this.renderer.lightingOverride = {
          sunColor: [0, 0, 0],
          skyTint: [
            preset.skyTint[0] * darkroomSkyScale,
            preset.skyTint[1] * darkroomSkyScale,
            preset.skyTint[2] * darkroomSkyScale,
          ],
        };
      }
      const camAz = ((preset.sunAzimuthDeg + 180) * Math.PI) / 180;
      const camEl = (14 * Math.PI) / 180;
      this.camera.setPoseImmediate(camAz, camEl, 4.25, [0, 1.25, 0], aspect);
      return;
    }

    if (scene === 'caustic-pool') {
      this.physics.rebuildTower('solo');
      for (let i = 0; i < 100; i++) {
        this.physics.step(1 / 120);
      }
      const az = (preset.sunAzimuthDeg * Math.PI) / 180;
      const el = (preset.sunElevationDeg * Math.PI) / 180;
      const lx = Math.cos(el) * Math.sin(az);
      const ly = Math.max(0.25, Math.sin(el));
      const lz = Math.cos(el) * Math.cos(az);
      const poolX = -1.35 * (lx / ly);
      const poolZ = -1.35 * (lz / ly);
      const camAz = ((preset.sunAzimuthDeg + 90) * Math.PI) / 180;
      this.camera.setPoseImmediate(camAz, 1.34, 4.65, [poolX * 0.82, 0.04, poolZ * 0.82], aspect);
      return;
    }

    if (scene === 'collapse-300') {
      this.physics.rebuildTower('sandbox');
      for (let i = 0; i < 60; i++) {
        this.physics.step(1 / 60);
      }
      this.physics.triggerSandboxCollapseImpulse();
      for (let f = 0; f < 165; f++) {
        this.physics.step(1 / 60);
        // Capture the mid-cascade Voronoi shard peak (`~409` active shards on the 43 cm oak table
        // before outer radial shards drop past the circular rim onto the hidden shelf).
        if (f >= 34 && this.physics.shards.length >= 340) {
          break;
        }
      }
      let sx = 0;
      let sz = 0;
      const nSh = this.physics.shards.length;
      if (nSh > 0) {
        for (let i = 0; i < nSh; i++) {
          sx += this.physics.shards[i].position[0];
          sz += this.physics.shards[i].position[2];
        }
        sx /= nSh;
        sz /= nSh;
      }
      this.camera.setPoseImmediate(0.58, 0.52, 3.65, [sx * 0.85, 0.22, sz * 0.85], aspect);
      return;
    }

    // Default bench scene: 'morning-front'
    this.physics.rebuildTower('solo');
    for (let i = 0; i < 100; i++) {
      this.physics.step(1 / 120);
    }
    this.camera.setPoseImmediate(0.64, 0.36, 4.45, [0, 1.35, 0], aspect);
  }

  public triggerCollapseDemo(): void {
    if (this.mpRole) return;
    this.audio.unlock();
    this.physics.triggerSandboxCollapseImpulse();
    this.emitTelemetry(true);
  }

  private raycastIntactBlocks(rayOrigin: Vec3, rayDir: Vec3): { blockId: number; hit: RayOBBHit } | null {
    let bestT = Infinity;
    let bestResult: { blockId: number; hit: RayOBBHit } | null = null;
    const halfExt: Vec3 = [BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z];

    for (let i = 0; i < this.physics.blocks.length; i++) {
      const b = this.physics.blocks[i];
      if (b.shattered) continue;
      const hit = intersectRayOBB(rayOrigin, rayDir, b.position, b.rotation, halfExt);
      if (hit && hit.t < bestT) {
        bestT = hit.t;
        bestResult = { blockId: b.id, hit };
      }
    }
    return bestResult;
  }

  private intersectHorizontalPlane(rayOrigin: Vec3, rayDir: Vec3, planeY: number): Vec3 | null {
    if (Math.abs(rayDir[1]) < 0.04) {
      const dist = Math.hypot(rayOrigin[0], rayOrigin[2]);
      return [
        rayOrigin[0] + rayDir[0] * dist,
        planeY,
        rayOrigin[2] + rayDir[2] * dist,
      ];
    }
    const t = (planeY - rayOrigin[1]) / rayDir[1];
    if (t < 0) return null;
    return [
      rayOrigin[0] + rayDir[0] * t,
      planeY,
      rayOrigin[2] + rayDir[2] * t,
    ];
  }

  /**
   * Grazing- & upward-pitch-safe isometric drag-plane projection:
   * Below `25 deg` camera elevation, tilts the proxy plane normal toward the camera by
   * `tiltBeta = min(90 deg, 25 deg - elevRad)` and unfolds the in-plane displacement
   * `dRad = (relX * hCamX + relZ * hCamZ) * cosB - relY * sinB` isometrically onto `y = anchorPoint[1]`.
   * Preserves `0.0 mm` grab-start jump, C^1 continuity at `25 deg`, and exact `1.00x` radial pull gain
   * even at steep upward camera pitches (`-25 deg` to `-65 deg`) without `cos(tiltBeta)` foreshortening.
   */
  public projectBlockDragPlane(
    rayOrigin: Vec3,
    rayDir: Vec3,
    anchorPoint: Vec3
  ): Vec3 | null {
    const dxCam = rayOrigin[0] - anchorPoint[0];
    const dyCam = rayOrigin[1] - anchorPoint[1];
    const dzCam = rayOrigin[2] - anchorPoint[2];
    const horizDist = Math.max(1e-3, Math.hypot(dxCam, dzCam));
    const elevRad = Math.atan2(dyCam, horizDist);
    const minGrazeRad = (25.0 * Math.PI) / 180.0;

    if (elevRad >= minGrazeRad) {
      return this.intersectHorizontalPlane(rayOrigin, rayDir, anchorPoint[1]);
    }

    const tiltBeta = Math.min(Math.PI * 0.5, minGrazeRad - elevRad);
    const hCamX = dxCam / horizDist;
    const hCamZ = dzCam / horizDist;
    const sinB = Math.sin(tiltBeta);
    const cosB = Math.cos(tiltBeta);
    const nx = hCamX * sinB;
    const ny = cosB;
    const nz = hCamZ * sinB;

    const num =
      (anchorPoint[0] - rayOrigin[0]) * nx +
      (anchorPoint[1] - rayOrigin[1]) * ny +
      (anchorPoint[2] - rayOrigin[2]) * nz;
    const rawDenom = rayDir[0] * nx + rayDir[1] * ny + rayDir[2] * nz;

    let denom = rawDenom;
    if (num <= 0) {
      if (rawDenom > -0.08) denom = -0.08;
    } else if (Math.abs(rawDenom) < 1e-4) {
      return null;
    }

    const t = num / denom;
    if (t <= 0) return null;

    const relX = rayOrigin[0] + rayDir[0] * t - anchorPoint[0];
    const relY = rayOrigin[1] + rayDir[1] * t - anchorPoint[1];
    const relZ = rayOrigin[2] + rayDir[2] * t - anchorPoint[2];

    let dLat = -relX * hCamZ + relZ * hCamX;
    let dRad = (relX * hCamX + relZ * hCamZ) * cosB - relY * sinB;

    const excursion = Math.hypot(dRad, dLat);
    const maxExcursionDm = 2.6;
    if (excursion > maxExcursionDm) {
      const scale = maxExcursionDm / excursion;
      dRad *= scale;
      dLat *= scale;
    }

    return [
      anchorPoint[0] + hCamX * dRad - hCamZ * dLat,
      anchorPoint[1],
      anchorPoint[2] + hCamZ * dRad + hCamX * dLat,
    ];
  }

  private updateCursorStyle(): void {
    if (!this.canvas?.style || !this.physics) return;
    if (this.interactionMode === 'orbit') {
      this.canvas.style.cursor = 'move';
      return;
    }
    if (!this.canLocalPlayerManipulateBlocks()) {
      this.canvas.style.cursor = 'default';
      return;
    }
    if (
      (this.physics.turnPhase === 'PULLING' && this.physics.isPointerDraggingBlock) ||
      (this.mpRole === 'guest' && this.interactionMode === 'block')
    ) {
      this.canvas.style.cursor = 'grabbing';
      return;
    }
    if (this.physics.turnPhase === 'EXTRACTED') {
      this.canvas.style.cursor = 'pointer';
      return;
    }
    if (this.physics.hoveredBlockId !== null) {
      const blk = this.physics.blocks[this.physics.hoveredBlockId];
      this.canvas.style.cursor = blk && blk.isLegalTarget ? 'grab' : 'not-allowed';
      return;
    }
    this.canvas.style.cursor = 'default';
  }

  private getViewportRect(overrideRect?: {
    left: number;
    top: number;
    width: number;
    height: number;
  }): { left: number; top: number; width: number; height: number } {
    if (overrideRect) return overrideRect;
    if (typeof this.canvas.getBoundingClientRect === 'function') {
      return this.canvas.getBoundingClientRect();
    }
    return { left: 0, top: 0, width: 1280, height: 800 };
  }

  private intersectTopSupportPlane(rayOrigin: Vec3, rayDir: Vec3): Vec3 | null {
    const frame = this.physics.topSupportFrame;
    const lc = frame.layerCenter;
    const n = frame.upNormal;
    const denom = rayDir[0] * n[0] + rayDir[1] * n[1] + rayDir[2] * n[2];
    if (Math.abs(denom) > 1e-4) {
      const t =
        ((lc[0] - rayOrigin[0]) * n[0] +
          (lc[1] - rayOrigin[1]) * n[1] +
          (lc[2] - rayOrigin[2]) * n[2]) /
        denom;
      if (t < 0) return null;
      return [
        rayOrigin[0] + rayDir[0] * t,
        rayOrigin[1] + rayDir[1] * t,
        rayOrigin[2] + rayDir[2] * t,
      ];
    }
    return this.intersectHorizontalPlane(rayOrigin, rayDir, lc[1] || 2.7);
  }

  public handlePointerDown(
    clientX: number,
    clientY: number,
    pointerId: number = 1,
    button: number = 0,
    altKey: boolean = false,
    overrideRect?: { left: number; top: number; width: number; height: number }
  ): void {
    this.audio.unlock();
    if (this.activePointerId !== null) return;

    this.activePointerId = pointerId;
    this.pointerDownPos = [clientX, clientY];
    this.lastPointerPos = [clientX, clientY];
    this.pointerMovedDist = 0;

    const rect = this.getViewportRect(overrideRect);
    const { origin, dir } = this.camera.screenToRay(clientX, clientY, rect);

    if (button === 2 || altKey || !this.canLocalPlayerManipulateBlocks()) {
      this.interactionMode = 'orbit';
      this.camera.isUserOrbiting = true;
      this.updateCursorStyle();
      return;
    }

    if (this.physics.turnPhase === 'EXTRACTED') {
      const lc = this.physics.topSupportFrame.layerCenter;
      const ptTop = this.intersectTopSupportPlane(origin, dir);
      const nearTopTower =
        ptTop !== null && Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < 1.15;

      if (nearTopTower) {
        if (this.mpRole === 'guest') {
          this.sendGuestRemoteCommand({
            kind: 'nudge_placement',
            rayOrigin: origin,
            rayDir: dir,
          });
        } else {
          this.physics.updateCarriedHover(origin, dir);
        }
        this.interactionMode = 'place_top';
      } else {
        this.interactionMode = 'orbit';
        this.camera.isUserOrbiting = true;
      }
      this.updateCursorStyle();
      return;
    }

    // Defer SELECT_BLOCK -> PULLING until >= 5.0 px drag so clicks tap-test blocks without locking in a turn!
    const pick = this.raycastIntactBlocks(origin, dir);
    if (pick) {
      const blk = this.physics.blocks[pick.blockId];
      if (
        blk &&
        ((blk.isLegalTarget && this.physics.turnPhase === 'SELECT_BLOCK') ||
          (!this.mpRole && this.physics.gameMode === 'sandbox'))
      ) {
        this.pendingTapHit = pick;
        this.grabAnchorOrigin = [...pick.hit.worldPoint];
        this.interactionMode = 'block_pending';
        this.updateCursorStyle();
        return;
      }
    }

    this.interactionMode = 'orbit';
    this.camera.isUserOrbiting = true;
    this.updateCursorStyle();
  }

  public handlePointerMove(
    clientX: number,
    clientY: number,
    pointerId: number = 1,
    overrideRect?: { left: number; top: number; width: number; height: number }
  ): void {
    const rect = this.getViewportRect(overrideRect);
    const { origin, dir } = this.camera.screenToRay(clientX, clientY, rect);
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();

    if (this.activePointerId === null || pointerId !== this.activePointerId) {
      if (!this.canLocalPlayerManipulateBlocks()) {
        if (this.mpRole === 'guest') {
          this.guestWorld?.setOptimisticGuestHover(null);
        }
        this.updateCursorStyle();
        return;
      }

      if (this.mpRole === 'guest') {
        if (this.physics.turnPhase === 'EXTRACTED') {
          if (now - this.mpLastGuestCmdSentMs >= 15) {
            this.sendGuestRemoteCommand(
              { kind: 'nudge_placement', rayOrigin: origin, rayDir: dir },
              now
            );
          }
        } else if (this.physics.turnPhase === 'SELECT_BLOCK') {
          const pick = this.raycastIntactBlocks(origin, dir);
          const pickId = pick ? pick.blockId : null;
          this.guestWorld?.setOptimisticGuestHover(pickId);
          if (pickId !== this.mpLastSentGuestHoverId && now - this.mpLastGuestCmdSentMs >= 15) {
            this.mpLastSentGuestHoverId = pickId;
            this.sendGuestRemoteCommand({ kind: 'hover_block', blockId: pickId }, now);
          }
        } else {
          this.guestWorld?.setOptimisticGuestHover(null);
        }
        this.updateCursorStyle();
        return;
      }

      if (this.physics.turnPhase === 'EXTRACTED') {
        this.physics.updateCarriedHover(origin, dir);
      }
      if (this.physics.turnPhase === 'SELECT_BLOCK' || this.physics.gameMode === 'sandbox') {
        const pick = this.raycastIntactBlocks(origin, dir);
        this.physics.hoveredBlockId = pick ? pick.blockId : null;
      } else {
        this.physics.hoveredBlockId = null;
      }
      this.updateCursorStyle();
      return;
    }

    const dx = clientX - this.lastPointerPos[0];
    const dy = clientY - this.lastPointerPos[1];
    this.lastPointerPos = [clientX, clientY];
    this.pointerMovedDist = Math.hypot(
      clientX - this.pointerDownPos[0],
      clientY - this.pointerDownPos[1]
    );

    if (this.interactionMode === 'orbit') {
      this.camera.orbitBy(-dx * 0.0072, dy * 0.0058);
      return;
    }

    if (!this.canLocalPlayerManipulateBlocks()) {
      this.handlePointerCancel(pointerId);
      return;
    }

    if (this.interactionMode === 'place_top' && this.physics.turnPhase === 'EXTRACTED') {
      if (this.mpRole === 'guest') {
        if (now - this.mpLastGuestCmdSentMs >= 15) {
          this.sendGuestRemoteCommand(
            { kind: 'nudge_placement', rayOrigin: origin, rayDir: dir },
            now
          );
        }
      } else {
        this.physics.updateCarriedHover(origin, dir);
      }
      this.updateCursorStyle();
      return;
    }

    if (this.interactionMode === 'block_pending' && this.pointerMovedDist >= 5.0 && this.pendingTapHit) {
      if (this.mpRole === 'guest') {
        this.sendGuestRemoteCommand(
          {
            kind: 'begin_pull',
            blockId: this.pendingTapHit.blockId,
            localPoint: this.pendingTapHit.hit.localPoint,
            worldPoint: this.pendingTapHit.hit.worldPoint,
          },
          now
        );
        this.guestWorld?.setOptimisticGuestPullAnchor(this.pendingTapHit.hit.worldPoint);
        this.interactionMode = 'block';
        this.emitTelemetry(true);
      } else {
        const started = this.physics.startGrabBlock(
          this.pendingTapHit.blockId,
          this.pendingTapHit.hit.localPoint,
          this.pendingTapHit.hit.worldPoint
        );
        if (started) {
          this.interactionMode = 'block';
          this.mpForceImmediateSnapshot = true;
          this.emitTelemetry(true);
        } else {
          this.interactionMode = 'none';
        }
      }
    }

    if (this.interactionMode === 'block') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        const lc = this.physics.topSupportFrame.layerCenter;
        const ptTop = this.intersectTopSupportPlane(origin, dir);
        const nearTopPlane =
          ptTop !== null && Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < 1.35;
        if (this.physics.isExtractionLiftReady && nearTopPlane) {
          if (this.mpRole === 'guest') {
            if (now - this.mpLastGuestCmdSentMs >= 15) {
              this.sendGuestRemoteCommand(
                { kind: 'nudge_placement', rayOrigin: origin, rayDir: dir },
                now
              );
            }
          } else {
            this.physics.updateCarriedHover(origin, dir);
          }
        }
      } else {
        const anchorPlaneOrigin: Vec3 =
          !this.mpRole && this.physics.gameMode === 'sandbox'
            ? [this.grabAnchorOrigin[0], this.physics.grabPlaneY, this.grabAnchorOrigin[2]]
            : this.grabAnchorOrigin;
        const planePt = this.projectBlockDragPlane(origin, dir, anchorPlaneOrigin);
        if (planePt) {
          if (this.mpRole === 'guest') {
            this.guestWorld?.setOptimisticGuestPullAnchor(planePt);
            if (now - this.mpLastGuestCmdSentMs >= 15) {
              this.mpPendingPullTarget = null;
              this.sendGuestRemoteCommand({ kind: 'update_pull', worldTarget: planePt }, now);
            } else {
              this.mpPendingPullTarget = planePt;
            }
          } else {
            this.physics.updateGrabTarget(planePt);
          }
        }
      }
      this.updateCursorStyle();
    }
  }

  public handlePointerUp(
    clientX: number,
    clientY: number,
    pointerId: number = 1,
    overrideRect?: { left: number; top: number; width: number; height: number }
  ): void {
    if (this.activePointerId === null || pointerId !== this.activePointerId) return;

    const rect = this.getViewportRect(overrideRect);
    const { origin, dir } = this.camera.screenToRay(clientX, clientY, rect);

    if (this.interactionMode === 'place_top') {
      const lc = this.physics.topSupportFrame.layerCenter;
      const ptTop = this.intersectTopSupportPlane(origin, dir);
      const stillOverTop =
        ptTop !== null && Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < 0.95;
      if (this.pointerMovedDist < 18.0 || stillOverTop) {
        if (this.mpRole === 'guest') {
          this.sendGuestRemoteCommand({
            kind: 'nudge_placement',
            rayOrigin: origin,
            rayDir: dir,
          });
          this.sendGuestRemoteCommand({ kind: 'confirm_placement' });
        } else {
          this.physics.updateCarriedHover(origin, dir);
          this.physics.releaseExtractedBlockOntoTop();
          this.mpForceImmediateSnapshot = true;
        }
      }
    } else if (this.interactionMode === 'block_pending') {
      // Pointer moved < 5.0 px: acoustic & tactile tap-test without leaving SELECT_BLOCK!
      if (this.pendingTapHit && this.canLocalPlayerManipulateBlocks()) {
        if (this.mpRole === 'guest') {
          this.sendGuestRemoteCommand({
            kind: 'tap_block',
            blockId: this.pendingTapHit.blockId,
            localNormal: this.pendingTapHit.hit.localNormal,
            localPoint: this.pendingTapHit.hit.localPoint,
          });
        } else {
          this.physics.setListenerPose(this.camera.position, this.camera.getRightVector());
          this.physics.tapNudgeBlock(
            this.pendingTapHit.blockId,
            this.pendingTapHit.hit.localNormal,
            this.pendingTapHit.hit.localPoint
          );
          this.mpForceImmediateSnapshot = true;
        }
      }
    } else if (this.interactionMode === 'block') {
      if (this.mpRole === 'guest') {
        this.guestWorld?.setOptimisticGuestPullAnchor(null);
        if (this.mpPendingPullTarget) {
          const finalTarget = this.mpPendingPullTarget;
          this.mpPendingPullTarget = null;
          this.sendGuestRemoteCommand({ kind: 'update_pull', worldTarget: finalTarget });
        }
        const lc = this.physics.topSupportFrame.layerCenter;
        const ptTop = this.intersectTopSupportPlane(origin, dir);
        const overTop =
          this.physics.turnPhase === 'EXTRACTED' &&
          this.physics.isExtractionLiftReady &&
          ptTop !== null &&
          Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < 0.78;
        this.sendGuestRemoteCommand({ kind: 'end_pull', dropIfPoisedOverTower: overTop });
      } else if (this.physics.turnPhase === 'EXTRACTED') {
        const lc = this.physics.topSupportFrame.layerCenter;
        const ptTop = this.intersectTopSupportPlane(origin, dir);
        const overTop =
          this.physics.isExtractionLiftReady &&
          ptTop !== null &&
          Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < 0.78;
        this.physics.endGrab(overTop);
        this.mpForceImmediateSnapshot = true;
      } else {
        this.physics.endGrab(false);
        this.mpForceImmediateSnapshot = true;
      }
    }

    this.camera.isUserOrbiting = false;
    this.activePointerId = null;
    this.interactionMode = 'none';
    this.pendingTapHit = null;
    this.updateCursorStyle();
    this.emitTelemetry(true);
  }

  public handlePointerCancel(pointerId?: number): void {
    if (this.activePointerId === null) return;
    if (pointerId !== undefined && pointerId !== this.activePointerId) return;

    if (this.interactionMode === 'block') {
      if (this.mpRole === 'guest') {
        this.mpPendingPullTarget = null;
        this.guestWorld?.setOptimisticGuestPullAnchor(null);
        this.sendGuestRemoteCommand({ kind: 'end_pull', dropIfPoisedOverTower: false });
      } else {
        this.physics.endGrab(false);
        this.mpForceImmediateSnapshot = true;
      }
    }

    this.camera.isUserOrbiting = false;
    this.activePointerId = null;
    this.interactionMode = 'none';
    this.pendingTapHit = null;
    this.updateCursorStyle();
    this.emitTelemetry(true);
  }

  public handleRotateCarried(deltaYawRad: number): void {
    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;
    if (this.mpRole === 'guest') {
      this.sendGuestRemoteCommand({ kind: 'rotate_placement', deltaYawRad });
      return;
    }
    this.physics.rotateCarriedBlock(deltaYawRad);
    this.mpForceImmediateSnapshot = true;
    this.emitTelemetry(true);
  }

  public handleResetCarriedRotation(): void {
    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;
    if (this.mpRole === 'guest') {
      this.sendGuestRemoteCommand({ kind: 'reset_rotation' });
      return;
    }
    this.physics.resetCarriedBlockRotation();
    this.mpForceImmediateSnapshot = true;
    this.emitTelemetry(true);
  }

  public handleReleaseOntoTop(): void {
    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;
    if (this.mpRole === 'guest') {
      this.sendGuestRemoteCommand({ kind: 'confirm_placement' });
      return;
    }
    this.physics.releaseExtractedBlockOntoTop();
    this.mpForceImmediateSnapshot = true;
    this.emitTelemetry(true);
  }

  private onPointerDown = (e: PointerEvent): void => {
    if ((e as unknown as { _vitrailHandled?: boolean })._vitrailHandled) return;
    (e as unknown as { _vitrailHandled?: boolean })._vitrailHandled = true;
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // Synthetic pointer events may not have an active pointerId
    }
    if (e.pointerType === 'touch') {
      this.activeTouches.set(e.pointerId, [e.clientX, e.clientY]);
      if (this.activeTouches.size === 2) {
        const pts = Array.from(this.activeTouches.values());
        this.lastPinchDist = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
        if (this.interactionMode === 'orbit' || this.interactionMode === 'block_pending') {
          this.handlePointerCancel();
        }
        return;
      }
    }
    this.handlePointerDown(e.clientX, e.clientY, e.pointerId, e.button, e.altKey);
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (e.pointerType === 'touch' && this.activeTouches.has(e.pointerId)) {
      this.activeTouches.set(e.pointerId, [e.clientX, e.clientY]);
      if (this.activeTouches.size >= 2) {
        const pts = Array.from(this.activeTouches.values());
        const dist = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
        if (this.lastPinchDist > 4) {
          const pinchDelta = (this.lastPinchDist - dist) * 0.014;
          this.camera.zoomBy(Math.max(-0.22, Math.min(0.22, pinchDelta)));
        }
        this.lastPinchDist = dist;
        return;
      }
    }
    this.handlePointerMove(e.clientX, e.clientY, e.pointerId);
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') {
      this.activeTouches.delete(e.pointerId);
      if (this.activeTouches.size < 2) {
        this.lastPinchDist = 0;
      }
    }
    this.handlePointerUp(e.clientX, e.clientY, e.pointerId);
  };

  private onPointerCancel = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') {
      this.activeTouches.delete(e.pointerId);
      if (this.activeTouches.size < 2) {
        this.lastPinchDist = 0;
      }
    }
    this.handlePointerCancel(e.pointerId);
  };

  private onWindowBlur = (): void => {
    this.activeTouches.clear();
    this.lastPinchDist = 0;
    this.handlePointerCancel();
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    if (
      this.canLocalPlayerManipulateBlocks() &&
      this.physics.turnPhase === 'EXTRACTED' &&
      !e.altKey &&
      !e.ctrlKey &&
      !e.metaKey
    ) {
      const wheelAxis = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      const yawStep = Math.max(-0.16, Math.min(0.16, wheelAxis * 0.0009));
      this.handleRotateCarried(yawStep);
      return;
    }
    const zoomDelta = Math.max(-0.22, Math.min(0.22, e.deltaY * 0.0012));
    this.camera.zoomBy(zoomDelta);
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (this.modalOpen || !this.physics) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const targetEl = e.target as HTMLElement | null;
    const tag = targetEl?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    if (e.code === 'Escape') {
      if (this.activePointerId !== null) {
        e.preventDefault();
        this.handlePointerCancel();
      }
      return;
    }

    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;

    if (e.code === 'KeyQ' || e.code === 'BracketLeft') {
      e.preventDefault();
      this.handleRotateCarried(-Math.PI / 24);
    } else if (e.code === 'KeyE' || e.code === 'BracketRight') {
      e.preventDefault();
      this.handleRotateCarried(Math.PI / 24);
    } else if (e.code === 'KeyR') {
      e.preventDefault();
      this.handleRotateCarried(Math.PI * 0.5);
    } else if (e.code === 'Digit0') {
      e.preventDefault();
      this.handleResetCarriedRotation();
    } else if (
      (e.code === 'Space' || e.code === 'Enter') &&
      tag !== 'BUTTON' &&
      tag !== 'A' &&
      this.physics.isExtractionLiftReady
    ) {
      e.preventDefault();
      this.handleReleaseOntoTop();
    }
  };

  private onContextMenu = (e: MouseEvent): void => {
    e.preventDefault();
  };

  private onPageHide = (ev: PageTransitionEvent): void => {
    if (!ev.persisted) {
      this.dispose();
    }
  };

  private attachEvents(): void {
    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('pointercancel', this.onPointerCancel);
    window.addEventListener('blur', this.onWindowBlur);
    window.addEventListener('keydown', this.onKeyDown);
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
    this.canvas.addEventListener('contextmenu', this.onContextMenu);
    window.addEventListener('pagehide', this.onPageHide);

    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          this.renderer.resize(width, height);
        }
      }
    });
    this.resizeObserver.observe(this.canvas);
  }

  public dispose(): void {
    this.disposed = true;
    this.running = false;
    if (typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(this.rafId);
    }

    if (typeof this.canvas.removeEventListener === 'function') {
      this.canvas.removeEventListener('pointerdown', this.onPointerDown);
      this.canvas.removeEventListener('wheel', this.onWheel);
      this.canvas.removeEventListener('contextmenu', this.onContextMenu);
    }
    if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
      window.removeEventListener('pointermove', this.onPointerMove);
      window.removeEventListener('pointerup', this.onPointerUp);
      window.removeEventListener('pointercancel', this.onPointerCancel);
      window.removeEventListener('blur', this.onWindowBlur);
      window.removeEventListener('keydown', this.onKeyDown);
      window.removeEventListener('pagehide', this.onPageHide);
    }
    this.resizeObserver?.disconnect();

    this.shutdownMultiplayerSessionOnly();
    if (this.guestWorld) {
      this.guestWorld.dispose();
      this.guestWorld = null;
    }
    if (this.authoritativePhysics && this.authoritativePhysics !== this.physics) {
      this.authoritativePhysics.dispose();
      this.authoritativePhysics = null;
    }
    this.renderer?.dispose();
    this.physics?.dispose();
    this.audio?.dispose();
  }
}

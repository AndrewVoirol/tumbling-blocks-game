import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  INITIAL_LAYERS,
  LIGHTING_PRESETS,
  SHARD_SLOT_STRIDE_VERTS,
  TraceryMotif,
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
  buildDeterministicTowerMaterials,
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

export type AiDifficulty = 'apprentice' | 'master';

export interface TouchExtractionPlan {
  blockId: number;
  pullDirXZ: [number, number];
  azimuthRad: number;
  azimuthOffsetDeg: number;
  dirSign: 1 | -1;
  localGrab: Vec3;
  startWorld: Vec3;
  minNeighborClearanceDm: number;
}

export interface AdversarialPlacementPlan {
  targetXZ: [number, number];
  localVU: [number, number];
  yawOffsetRad: number;
  predictedComXZCm: [number, number];
  predictedMinCourseMarginDm: number;
}

export interface VitrailHudTelemetry extends TowerTelemetry {
  aiDifficulty?: AiDifficulty;
  aiVsAiEnabled?: boolean;
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
  /** Second-finger arrival window (ms) within which an optimistic 1-finger grab is canceled into pinch/pan */
  public static readonly MULTI_TOUCH_GRAB_CANCEL_WINDOW_MS = 80;

  private canvas: HTMLCanvasElement;
  public audio: GlassAudioEngine;
  public physics!: PhysicsWorld;
  public renderer!: WebGPURenderer;
  public camera: OrbitCamera;

  private rafId: number = 0;
  private running: boolean = false;
  private disposed: boolean = false;
  private lastFrameMs: number = 0;
  private lastForegroundFrameMs: number = 0;
  private lastBgPumpStepMs: number = 0;
  private hostOccludedOverride: boolean = false;
  private elapsedSec: number = 0;
  private lastTelemetryEmitMs: number = 0;
  private listener: TelemetryListener | null;

  // Pointer & multi-touch interaction tracking
  private activePointerId: number | null = null;
  private activeTouches = new Map<number, [number, number]>();
  private primaryTouchDownMs: number = -Infinity;
  private lastPinchDist: number = 0;
  private lastPinchMidpoint: [number, number] = [0, 0];
  private pointerDownPos: [number, number] = [0, 0];
  private lastPointerPos: [number, number] = [0, 0];
  private pointerMovedDist: number = 0;
  private interactionMode:
    | 'none'
    | 'orbit'
    | 'block_pending'
    | 'block'
    | 'place_top'
    | 'pinch_pan' = 'none';
  private pendingTapHit: { blockId: number; hit: RayOBBHit } | null = null;
  private grabAnchorOrigin: Vec3 = [0, 0, 0];

  public modalOpen: boolean = false;
  public benchPaused: boolean = false;
  public benchCameraLocked: boolean = false;
  public aiOpponentEnabled: boolean = false;
  public aiDifficulty: AiDifficulty = 'master';
  public aiVsAiEnabled: boolean = false;
  public aiFastCadence: boolean = false;
  private aiPlayerDifficulties: [AiDifficulty, AiDifficulty] = ['master', 'master'];
  public aiThinking: boolean = false;
  public aiActionLabel: string | null = null;
  public polariscopeMode: boolean = false;
  public physicsWorkerActive: boolean = false;
  private resizeObserver: ResizeObserver | null = null;
  private rendererInitialized: boolean = false;
  private gpuYieldChannel: BroadcastChannel | null = null;
  private gpuYieldUntilMs: number = 0;
  private readonly gpuInstanceId: string = `eng_${Math.random().toString(36).slice(2, 10)}`;

  // Glazier AI 2-ply turn controller state
  private aiStage:
    | 'idle'
    | 'scouting'
    | 'awaiting_oracle'
    | 'probe_tight'
    | 'post_tap'
    | 'pulling'
    | 'placing' = 'idle';
  private aiTimerSec: number = 0;
  private aiPullProgressSec: number = 0;
  private aiTargetBlockId: number | null = null;
  private aiPullStartWorld: Vec3 = [0, 0, 0];
  private aiPullBlockInitXZ: [number, number] = [0, 0];
  private aiPullDirXZ: [number, number] = [1, 0];
  private aiHoverTargetXZ: [number, number] = [0, 0];
  private aiHoverCurrentXZ: [number, number] = [0, 0];
  private aiTargetYawOffsetRad: number = 0;
  private aiAbortedBlockIds: Set<number> = new Set();
  private aiLastBlocksStacked: number = -1;

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
  private mpLastAcceptedGuestCmdSeq: number = 0;
  private mpActiveGuestClientId: string | null = null;
  private mpLastGuestCmdSentMs: number = 0;
  private mpLastSentGuestHoverId: number | null;
  private mpPendingPullTarget: Vec3 | null = null;
  private mpActiveGuestPullGrab: {
    blockId: number;
    localPoint: Vec3;
    worldPoint: Vec3;
  } | null = null;
  private mpPendingDeferredGuestCmds: GuestRemoteCommand[] = [];
  private mpExpectedPalette: Array<{
    swatchId: number;
    motif: TraceryMotif;
    rippleSeed: number;
  }> | null = null;
  private mpHostBgTimer: ReturnType<typeof setInterval> | null = null;
  private mpHostDisconnectGraceTimer: ReturnType<typeof setTimeout> | null = null;

  // Host Voronoi shard & discrete spatial audio replication buffers
  private hostRawConsumeShardUploads: (() => ShardMeshAppendEvent[]) | null = null;
  private hostStagedRendererUploads: ShardMeshAppendEvent[] = [];
  private hostSlotLocalCentroids = new Map<number, Vec3>();
  private hostActiveShardMetas = new Map<number, SerializedShardMeta>();
  private hostAudioTapInstalled: boolean = false;
  private hostAudioTapDepth: number = 0;
  private mpOutboundAudioQueue: NetworkAudioEvent[] = [];
  private mpAudioBatchSeq: number = 0;
  private mpRecentAudioBatches: Array<{
    batchSeq: number;
    roomSeed: number;
    events: NetworkAudioEvent[];
  }> = [];
  private mpLastSlideAudioSentMs: number = 0;
  private mpLastSlideWasNonZero: boolean = false;
  private scratchWrench4 = new Float32Array(4);

  constructor(canvas?: HTMLCanvasElement, onTelemetry?: TelemetryListener) {
    this.canvas = canvas ?? ({ style: {} } as HTMLCanvasElement);
    this.listener = onTelemetry ?? null;
    this.mpLastSentGuestHoverId = null;
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

  private ensureGpuYieldChannel(): BroadcastChannel | null {
    if (this.disposed || typeof BroadcastChannel === 'undefined') return null;
    if (this.gpuYieldChannel) return this.gpuYieldChannel;
    try {
      const ch = new BroadcastChannel('vitrail_webgpu_yield');
      ch.onmessage = (ev: MessageEvent) => {
        if (this.disposed) return;
        const data = ev.data as { type?: string; senderId?: string } | null;
        if (!data || data.senderId === this.gpuInstanceId) return;
        const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
        if (data.type === 'gpu-init-start') {
          this.gpuYieldUntilMs = Math.max(this.gpuYieldUntilMs, now + 350);
        } else if (data.type === 'gpu-init-done') {
          this.gpuYieldUntilMs = 0;
        }
      };
      this.gpuYieldChannel = ch;
      return ch;
    } catch {
      return null;
    }
  }

  private closeGpuYieldChannel(): void {
    if (!this.gpuYieldChannel) return;
    try {
      this.gpuYieldChannel.onmessage = null;
      this.gpuYieldChannel.close();
    } catch {
      // Ignore BroadcastChannel close errors
    }
    this.gpuYieldChannel = null;
  }

  public async init(): Promise<void> {
    // Yield one microtask so React 19 StrictMode's synchronous mount -> cleanup -> remount
    // disposes throwaway Mount #1 BEFORE touching `navigator.gpu.requestAdapter()`.
    await Promise.resolve();
    if (this.disposed) {
      return;
    }

    const yieldCh = this.ensureGpuYieldChannel();
    if (yieldCh) {
      try {
        yieldCh.postMessage({ type: 'gpu-init-start', senderId: this.gpuInstanceId });
      } catch {
        // Ignore postMessage errors
      }
      // Allow sibling tabs' in-flight Retina 4x MSAA rAF command buffers (~4-16 ms) to drain
      // on Dawn's GPU process before entering `navigator.gpu.requestAdapter()`.
      await new Promise<void>((resolve) => setTimeout(resolve, 20));
      if (this.disposed) {
        try {
          yieldCh.postMessage({ type: 'gpu-init-done', senderId: this.gpuInstanceId });
        } catch {
          // Ignore
        }
        this.closeGpuYieldChannel();
        return;
      }
    }

    this.renderer = new WebGPURenderer(this.canvas);
    try {
      await this.renderer.init();
      this.rendererInitialized = true;
    } finally {
      try {
        this.gpuYieldChannel?.postMessage({
          type: 'gpu-init-done',
          senderId: this.gpuInstanceId,
        });
      } catch {
        // Ignore
      }
    }

    if (this.disposed) {
      this.renderer.dispose();
      this.closeGpuYieldChannel();
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
    const bootNow = performance.now();
    this.lastFrameMs = bootNow;
    this.lastForegroundFrameMs = bootNow;
    this.lastBgPumpStepMs = bootNow;
    this.emitTelemetry(true);
    this.rafId = requestAnimationFrame(this.tick);

    // Expose engine handle for DevTools runtime profiling & automated verification
    (window as unknown as { __VITRAIL_ENGINE__?: GameEngine }).__VITRAIL_ENGINE__ = this;
  }

  /**
   * Explicitly toggles Host tab occlusion state (used alongside `document.hidden` to drive
   * the `30 Hz` background physics & snapshot pump when foreground `rAF` is halted).
   */
  public setHostTabOccluded(occluded: boolean): void {
    this.hostOccludedOverride = occluded;
    if (occluded) {
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      if (this.lastBgPumpStepMs <= 0) {
        this.lastBgPumpStepMs = now;
      }
      if (this.mpRole === 'host' && this.mpHostBgTimer === null && !this.disposed) {
        this.startHostBackgroundPump();
      }
    }
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
    this.lastFrameMs = now;
    this.lastForegroundFrameMs = now;
    this.lastBgPumpStepMs = now;
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
        } else if (this.mpRole === 'host') {
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
          } else if (this.mpRole === 'host') {
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
    this.lastForegroundFrameMs = nowMs;
    this.lastBgPumpStepMs = nowMs;
    const dt = Math.min(0.05, Math.max(0.001, rawDt));
    this.elapsedSec += dt;

    const cw = this.canvas.clientWidth || window.innerWidth;
    const ch = Math.max(1, this.canvas.clientHeight || window.innerHeight);
    const shouldYieldGpu =
      (typeof document !== 'undefined' && Boolean(document.hidden)) ||
      nowMs < this.gpuYieldUntilMs;
    if (!shouldYieldGpu && this.rendererInitialized && cw > 0 && ch > 0) {
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

    if (!shouldYieldGpu && this.rendererInitialized) {
      this.renderer.render(this.physics, this.camera, this.elapsedSec);
    }

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
      aiDifficulty: this.aiDifficulty,
      aiVsAiEnabled: this.aiVsAiEnabled,
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
      aiDifficulty: this.aiDifficulty,
      aiVsAiEnabled: this.aiVsAiEnabled,
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
      const loc = window.location;
      const rawHref =
        typeof loc.href === 'string' && loc.href.length > 0
          ? loc.href
          : `${loc.pathname ?? '/'}${loc.search ?? ''}${loc.hash ?? ''}`;
      const isAbsolute = /^https?:\/\//i.test(rawHref);
      const url = new URL(rawHref, 'http://localhost');
      if (url.searchParams.has('bench')) return;
      if (code) {
        url.searchParams.set('room', code);
      } else {
        url.searchParams.delete('room');
      }
      const nextUrl = isAbsolute
        ? url.toString()
        : `${url.pathname}${url.search}${url.hash}`;
      window.history.replaceState(null, '', nextUrl);
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
    if (this.mpHostBgTimer !== null) {
      clearInterval(this.mpHostBgTimer);
      this.mpHostBgTimer = null;
    }
    if (this.mpHostDisconnectGraceTimer !== null) {
      clearTimeout(this.mpHostDisconnectGraceTimer);
      this.mpHostDisconnectGraceTimer = null;
    }
    if (this.mpClient) {
      this.mpClient.dispose();
      this.mpClient = null;
    }
    this.mpRole = null;
    this.mpRoomCode = null;
    this.mpExpectedPalette = null;
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
    this.mpStateSeq = 0;
    this.mpLastSnapshotSentMs = 0;
    this.mpForceImmediateSnapshot = false;
    this.mpLastCollapseReasonSent = null;
    this.mpGuestCmdSeq = 0;
    this.mpLastAcceptedGuestCmdSeq = 0;
    this.mpActiveGuestClientId = null;
    this.mpOutboundAudioQueue = [];
    this.mpAudioBatchSeq = 0;
    this.mpRecentAudioBatches = [];
    this.mpPendingPullTarget = null;
    this.mpActiveGuestPullGrab = null;
    this.mpPendingDeferredGuestCmds = [];
    this.mpLastSentGuestHoverId = null;
    this.hostOccludedOverride = false;
    this.hostStagedRendererUploads = [];
    this.hostSlotLocalCentroids.clear();
    this.hostActiveShardMetas.clear();
  }

  private enforceHostPaletteParity(): void {
    const palette = this.mpExpectedPalette;
    if (!palette || !this.physics) return;
    const blocks = this.physics.blocks;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      const mat = b.id >= 0 && b.id < palette.length ? palette[b.id] : undefined;
      if (mat) {
        b.swatchId = mat.swatchId;
        b.motif = mat.motif;
        b.rippleSeed = mat.rippleSeed;
      }
    }
  }

  private startHostBackgroundPump(): void {
    if (this.mpHostBgTimer !== null) {
      clearInterval(this.mpHostBgTimer);
    }
    this.mpHostBgTimer = setInterval(() => {
      if (this.disposed || this.mpRole !== 'host') return;
      const docHidden = typeof document !== 'undefined' && Boolean(document.hidden);
      const activeSession = this.running || this.hostOccludedOverride || docHidden;
      if (!activeSession) return;
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const isOccludedOrStalled =
        this.hostOccludedOverride ||
        docHidden ||
        (this.running && now - this.lastForegroundFrameMs > 55);
      if (!isOccludedOrStalled) return;
      const prevStepMs = this.lastBgPumpStepMs > 0 ? this.lastBgPumpStepMs : now - 1000 / 30;
      const dt = Math.min(0.05, Math.max(0.008, (now - prevStepMs) / 1000));
      this.lastBgPumpStepMs = now;
      this.lastFrameMs = now;
      this.elapsedSec += dt;
      if (!this.benchPaused) {
        this.updateMasterGlazierAi(dt);
        this.stepPhysicsAndSecondaryAcoustics(dt);
      }
      if (now - this.mpLastSnapshotSentMs >= 30) {
        this.mpForceImmediateSnapshot = true;
      }
      this.pumpMultiplayerStep(now);
      if (now - this.lastTelemetryEmitMs >= 60) {
        this.emitTelemetry(false);
        this.lastTelemetryEmitMs = now;
      }
    }, 33);
  }

  private rebuildAuthoritativeTowerWithSeed(seed: number): void {
    const cleanSeed = seed >>> 0;
    this.mpExpectedPalette = buildDeterministicTowerMaterials(cleanSeed, INITIAL_LAYERS);
    const physHooks = this.physics as unknown as OptionalPhysicsHooks;
    physHooks.rebuildTower('party', 2, INITIAL_LAYERS, cleanSeed);
    applyDeterministicTowerPalette(this.physics.blocks, cleanSeed, INITIAL_LAYERS);
    this.hostStagedRendererUploads = [];
    this.hostSlotLocalCentroids.clear();
    this.hostActiveShardMetas.clear();
    // Preserve strictly monotonic `mpStateSeq` across rematches so delayed pre-rematch `0x564d`
    // packets (`frame.seq <= lastAcceptedSeq`) are 100% discarded by `MultiplayerGuestWorld`!
    this.mpOutboundAudioQueue = [];
    this.mpRecentAudioBatches = [];
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
    if (
      this.mpRole !== 'host' ||
      (!this.mpStatus.peerConnected && this.mpStatus.connectionState !== 'reconnecting')
    ) {
      return;
    }
    if (ev.kind === 'slideFriction' && this.mpOutboundAudioQueue.length > 0) {
      const tail = this.mpOutboundAudioQueue[this.mpOutboundAudioQueue.length - 1];
      if (tail.kind === 'slideFriction') {
        this.mpOutboundAudioQueue[this.mpOutboundAudioQueue.length - 1] = ev;
        return;
      }
    }
    if (this.mpOutboundAudioQueue.length >= 256) {
      const dropIdx = this.mpOutboundAudioQueue.findIndex(
        (item) => item.kind === 'slideFriction' || item.kind === 'wobbleRock'
      );
      if (dropIdx >= 0) {
        this.mpOutboundAudioQueue.splice(dropIdx, 1);
      } else {
        return;
      }
    }
    this.mpOutboundAudioQueue.push(ev);
  }

  public getOutboundAudioQueueSnapshot(): ReadonlyArray<NetworkAudioEvent> {
    return [...this.mpOutboundAudioQueue];
  }

  public installHostAudioTap(): void {
    if (this.hostAudioTapInstalled) return;
    this.hostAudioTapInstalled = true;

    const runTopLevelGuarded = <T>(fn: () => T, onTopLevel: (res: T) => void): T => {
      const isTopLevel = this.hostAudioTapDepth === 0;
      this.hostAudioTapDepth++;
      try {
        const result = fn();
        if (isTopLevel) {
          onTopLevel(result);
        }
        return result;
      } finally {
        this.hostAudioTapDepth--;
      }
    };

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
      playBlockThud?: (
        impactSpeedMps: number,
        worldPos?: Vec3,
        camPosOrOptions?: Vec3 | { damage?: number; camPos?: Vec3; camRight?: Vec3 },
        camRight?: Vec3,
        damage?: number
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
      runTopLevelGuarded(
        () => origSlide(speed, res, power, torsion, loadN),
        () => {
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
        }
      );
    };

    const origTap = audioAny.playTapTestResonance.bind(this.audio);
    audioAny.playTapTestResonance = (pinch01, worldPos, camPos, camRight, damage) => {
      runTopLevelGuarded(
        () => origTap(pinch01, worldPos, camPos, camRight, damage),
        () => {
          this.enqueueHostAudioEvent({ kind: 'tapTest', pinch01, worldPos, damage });
        }
      );
    };

    const origImpact = audioAny.playGlassImpact.bind(this.audio);
    audioAny.playGlassImpact = (intensity01, pitchMul = 1.0, worldPos, camPos, camRight, opts) => {
      runTopLevelGuarded(
        () => origImpact(intensity01, pitchMul, worldPos, camPos, camRight, opts),
        () => {
          this.enqueueHostAudioEvent({
            kind: 'glassImpact',
            intensity01,
            pitchMul,
            worldPos,
            impactSpeedMps: opts?.impactSpeedMps,
            damage: opts?.damage,
            surface: opts?.surface,
          });
        }
      );
    };

    if (typeof audioAny.playBlockThud === 'function') {
      const origThud = audioAny.playBlockThud.bind(this.audio);
      audioAny.playBlockThud = (impactSpeedMps, worldPos, camPosOrOptions, camRight, damage = 0) => {
        runTopLevelGuarded(
          () => origThud(impactSpeedMps, worldPos, camPosOrOptions, camRight, damage),
          () => {
            const resolvedDamage =
              camPosOrOptions && !Array.isArray(camPosOrOptions)
                ? camPosOrOptions.damage ?? damage
                : damage;
            this.enqueueHostAudioEvent({
              kind: 'blockThud',
              impactSpeedMps,
              worldPos,
              damage: resolvedDamage,
            });
          }
        );
      };
    }

    const origCrack = audioAny.playGlassCrack.bind(this.audio);
    audioAny.playGlassCrack = (damage01, worldPos, camPos, camRight) => {
      runTopLevelGuarded(
        () => origCrack(damage01, worldPos, camPos, camRight),
        () => {
          this.enqueueHostAudioEvent({ kind: 'glassCrack', damage01, worldPos });
        }
      );
    };

    const origShatter = audioAny.playGlassShatter.bind(this.audio);
    audioAny.playGlassShatter = (intensity01, hitTable = false, worldPos, camPos, camRight) => {
      runTopLevelGuarded(
        () => origShatter(intensity01, hitTable, worldPos, camPos, camRight),
        () => {
          this.enqueueHostAudioEvent({ kind: 'glassShatter', intensity01, hitTable, worldPos });
        }
      );
    };

    const origRanked = audioAny.playRankedShardCollisions.bind(this.audio);
    audioAny.playRankedShardCollisions = (events, camPos, camRight) => {
      runTopLevelGuarded(
        () => origRanked(events, camPos, camRight),
        () => {
          if (events.length > 0) {
            this.enqueueHostAudioEvent({ kind: 'rankedShards', events: events.slice(0, 6) });
          }
        }
      );
    };

    const origStack = audioAny.playStackSuccess.bind(this.audio);
    audioAny.playStackSuccess = (layerCount) => {
      runTopLevelGuarded(
        () => origStack(layerCount),
        () => {
          this.enqueueHostAudioEvent({ kind: 'stackSuccess', layerCount });
        }
      );
    };

    if (typeof audioAny.playShardImpacts === 'function') {
      const origShardImpacts = audioAny.playShardImpacts.bind(this.audio);
      audioAny.playShardImpacts = (impacts, camPos, camRight) =>
        runTopLevelGuarded(
          () => origShardImpacts(impacts, camPos, camRight),
          () => {
            if (impacts.length > 0) {
              this.enqueueHostAudioEvent({ kind: 'shardImpacts', impacts: impacts.slice(0, 8) });
            }
          }
        );
    }

    if (typeof audioAny.playWobbleRock === 'function') {
      const origWobble = audioAny.playWobbleRock.bind(this.audio);
      audioAny.playWobbleRock = (tilt, speed = 0, load = 0, worldPos = [0, 0, 0], opts) =>
        runTopLevelGuarded(
          () => origWobble(tilt, speed, load, worldPos, opts),
          () => {
            this.enqueueHostAudioEvent({
              kind: 'wobbleRock',
              tiltAngleRad: tilt,
              angularSpeedRadPerSec: speed,
              clampingLoadN: load,
              worldPos,
              surface: opts?.surface ?? 'glass',
              damage: opts?.damage ?? 0,
            });
          }
        );
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
        if (this.mpHostDisconnectGraceTimer !== null) {
          clearTimeout(this.mpHostDisconnectGraceTimer);
          this.mpHostDisconnectGraceTimer = null;
        }
        this.sendHostRoomWelcome();
        this.mpForceImmediateSnapshot = true;
        this.pumpMultiplayerStep(
          typeof performance !== 'undefined' ? performance.now() : Date.now()
        );
        overrides?.onPeerConnected?.();
      },
      onTransportFailover: (nextTransport, reason) => {
        if (this.mpHostDisconnectGraceTimer !== null) {
          clearTimeout(this.mpHostDisconnectGraceTimer);
          this.mpHostDisconnectGraceTimer = null;
        }
        this.sendHostRoomWelcome();
        this.mpForceImmediateSnapshot = true;
        this.pumpMultiplayerStep(
          typeof performance !== 'undefined' ? performance.now() : Date.now()
        );
        overrides?.onTransportFailover?.(nextTransport, reason);
      },
      onRemotePeerBye: (remoteClientId) => {
        if (this.mpHostDisconnectGraceTimer !== null) {
          clearTimeout(this.mpHostDisconnectGraceTimer);
          this.mpHostDisconnectGraceTimer = null;
        }
        if (this.physics.currentPlayer === 1 && this.physics.isPointerDraggingBlock) {
          this.physics.endGrab(false);
          this.mpForceImmediateSnapshot = true;
        }
        overrides?.onRemotePeerBye?.(remoteClientId);
      },
      onPeerDisconnected: () => {
        if (this.mpHostDisconnectGraceTimer !== null) {
          clearTimeout(this.mpHostDisconnectGraceTimer);
        }
        // Grace window across transient WebRTC -> WS relay failover so mid-drag Guest pulls survive seamlessly
        this.mpHostDisconnectGraceTimer = setTimeout(() => {
          this.mpHostDisconnectGraceTimer = null;
          if (
            !this.disposed &&
            this.mpRole === 'host' &&
            !this.mpStatus.peerConnected &&
            this.physics.currentPlayer === 1 &&
            this.physics.isPointerDraggingBlock
          ) {
            this.physics.endGrab(false);
            this.mpForceImmediateSnapshot = true;
          }
        }, 1500);
        overrides?.onPeerDisconnected?.();
      },
      onControlMessage: (msg) => {
        this.handleHostControlMessage(msg);
      },
    });
    this.mpClient = client;
    client.connect();
    this.startHostBackgroundPump();
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
        this.triggerGuestResyncAndPullRecovery();
        overrides?.onPeerConnected?.();
      },
      onTransportFailover: (nextTransport, reason) => {
        this.triggerGuestResyncAndPullRecovery();
        overrides?.onTransportFailover?.(nextTransport, reason);
      },
      onPeerDisconnected: () => {
        this.audio.updateSlideFriction(0, 0);
        if (this.mpStatus.connectionState !== 'reconnecting') {
          this.guestWorld?.setOptimisticGuestPullAnchor(null);
        }
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

  private triggerGuestResyncAndPullRecovery(): void {
    if (this.mpRole !== 'guest' || !this.mpClient || !this.guestWorld) return;
    const hadWelcome = this.guestWorld.hasCompletedInitialWelcome();
    this.mpClient.sendControl({
      type: 'request_resync',
      lastSeq: this.guestWorld.getLastAcceptedSeq(),
      lastAudioBatchSeq: hadWelcome ? this.guestWorld.getLastAcceptedAudioBatchSeq() : undefined,
    });

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (this.mpPendingDeferredGuestCmds.length > 0) {
      const queued = this.mpPendingDeferredGuestCmds;
      this.mpPendingDeferredGuestCmds = [];
      for (let i = 0; i < queued.length; i++) {
        this.sendGuestRemoteCommand(queued[i], now);
      }
    }

    if (
      this.interactionMode === 'block' &&
      this.mpActiveGuestPullGrab &&
      this.physics.currentPlayer === 1 &&
      this.physics.turnPhase !== 'COLLAPSED' &&
      this.physics.turnPhase !== 'SETTLING'
    ) {
      const grab = this.mpActiveGuestPullGrab;
      this.sendGuestRemoteCommand(
        {
          kind: 'begin_pull',
          blockId: grab.blockId,
          localPoint: grab.localPoint,
          worldPoint: grab.worldPoint,
        },
        now
      );
      if (this.mpPendingPullTarget) {
        const latestTarget = this.mpPendingPullTarget;
        this.mpPendingPullTarget = null;
        this.sendGuestRemoteCommand({ kind: 'update_pull', worldTarget: latestTarget }, now);
      }
    }
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
    // Cancel any active Host pointer drag before rebuilding the authoritative rematch tower
    this.handlePointerCancel();
    if (this.mpHostDisconnectGraceTimer !== null) {
      clearTimeout(this.mpHostDisconnectGraceTimer);
      this.mpHostDisconnectGraceTimer = null;
    }
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
          volume: +s.volume.toFixed(6),
          massKg: +s.massKg.toFixed(6),
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
      minSnapshotSeq: this.mpStateSeq,
      currentAudioBatchSeq: this.mpAudioBatchSeq,
    });
    if (allActiveShards.length > 0) {
      this.sendChunkedShardSpawns(allActiveShards, []);
    }
  }

  private buildHostStateFrame(): MultiplayerStateFrame {
    this.enforceHostPaletteParity();
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

    // Host role: synchronize seeded palette & new Voronoi shard geometries even while waiting for Guest so reconnects/joins have complete metadata
    this.enforceHostPaletteParity();
    this.syncHostShardMetasAndBroadcast();

    if (!this.mpStatus.peerConnected) {
      if (this.mpStatus.connectionState !== 'reconnecting') {
        this.mpOutboundAudioQueue = [];
      }
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
      this.mpAudioBatchSeq = (this.mpAudioBatchSeq + 1) >>> 0;
      const batchSeq = this.mpAudioBatchSeq;
      this.mpRecentAudioBatches.push({
        batchSeq,
        roomSeed: this.mpRoomSeed,
        events: batch,
      });
      if (this.mpRecentAudioBatches.length > 128) {
        this.mpRecentAudioBatches.shift();
      }
      this.mpClient.sendControl({
        type: 'audio_batch',
        batchSeq,
        roomSeed: this.mpRoomSeed,
        events: batch,
      });
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
      case 'hello': {
        if (msg.role === 'guest') {
          if (
            this.mpActiveGuestClientId !== null &&
            this.mpActiveGuestClientId !== msg.clientId &&
            this.physics.currentPlayer === 1 &&
            this.physics.isPointerDraggingBlock
          ) {
            this.physics.endGrab(false);
          }
          if (this.mpActiveGuestClientId !== msg.clientId) {
            this.mpActiveGuestClientId = msg.clientId;
            this.mpLastAcceptedGuestCmdSeq = 0;
          }
        }
        this.sendHostRoomWelcome();
        this.mpForceImmediateSnapshot = true;
        this.pumpMultiplayerStep(
          typeof performance !== 'undefined' ? performance.now() : Date.now()
        );
        break;
      }
      case 'request_resync': {
        this.sendHostRoomWelcome();
        if (typeof msg.lastAudioBatchSeq === 'number' && this.mpClient) {
          const watermark = msg.lastAudioBatchSeq >>> 0;
          for (let i = 0; i < this.mpRecentAudioBatches.length; i++) {
            const entry = this.mpRecentAudioBatches[i];
            if (entry.roomSeed === this.mpRoomSeed && entry.batchSeq > watermark) {
              this.mpClient.sendControl({
                type: 'audio_batch',
                batchSeq: entry.batchSeq,
                roomSeed: entry.roomSeed,
                events: entry.events,
              });
            }
          }
        }
        this.mpForceImmediateSnapshot = true;
        this.pumpMultiplayerStep(
          typeof performance !== 'undefined' ? performance.now() : Date.now()
        );
        break;
      }
      case 'guest_cmd': {
        if (
          this.physics.currentPlayer !== 1 ||
          this.physics.turnPhase === 'COLLAPSED' ||
          this.physics.turnPhase === 'SETTLING'
        ) {
          break;
        }
        if (
          msg.roomSeed !== undefined &&
          msg.roomSeed !== 0 &&
          (msg.roomSeed >>> 0) !== (this.mpRoomSeed >>> 0)
        ) {
          break;
        }
        if (!Number.isFinite(msg.seq) || msg.seq <= this.mpLastAcceptedGuestCmdSeq) {
          break;
        }
        this.mpLastAcceptedGuestCmdSeq = msg.seq >>> 0;
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
   * and single-owner pointer mutual exclusion before applying any Guest action to MuJoCo.
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
        if (
          this.physics.turnPhase === 'SELECT_BLOCK' &&
          !this.physics.isPointerDraggingBlock &&
          this.physics.activeBlockId === null
        ) {
          this.physics.tapNudgeBlock(cmd.blockId, cmd.localNormal, cmd.localPoint);
          this.mpForceImmediateSnapshot = true;
        }
        break;
      }
      case 'begin_pull': {
        if (
          this.physics.turnPhase === 'SELECT_BLOCK' &&
          !this.physics.isPointerDraggingBlock &&
          this.physics.activeBlockId === null &&
          !(this.interactionMode === 'block' && this.activePointerId !== null)
        ) {
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
        if (
          this.physics.turnPhase === 'EXTRACTED' &&
          this.isRayNearTopPlacementZone(cmd.rayOrigin, cmd.rayDir, 0.80)
        ) {
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

  private clearGuestLocalBlockInteraction(): void {
    this.mpPendingPullTarget = null;
    this.mpActiveGuestPullGrab = null;
    this.mpPendingDeferredGuestCmds = [];
    this.guestWorld?.setOptimisticGuestHover(null);
    this.guestWorld?.setOptimisticGuestPullAnchor(null);
    if (this.interactionMode !== 'orbit' && this.interactionMode !== 'none') {
      this.interactionMode = 'none';
      this.pendingTapHit = null;
      this.activePointerId = null;
    }
  }

  private handleGuestControlMessage(msg: MultiplayerControlMessage): void {
    if (this.mpRole !== 'guest' || !this.guestWorld) return;
    switch (msg.type) {
      case 'room_welcome': {
        const prevSeed = this.guestWorld.roomSeed;
        const hadWelcome = this.guestWorld.hasCompletedInitialWelcome();
        const nextSeed = msg.roomSeed >>> 0;
        if (hadWelcome && nextSeed !== prevSeed) {
          this.clearGuestLocalBlockInteraction();
        }
        this.mpRoomSeed = nextSeed;
        this.guestWorld.applyRoomWelcome({
          roomSeed: msg.roomSeed,
          initialLayers: msg.initialLayers,
          collapseReason: msg.collapseReason,
          activeShardMetas: msg.activeShardMetas,
          retiredShardIds: msg.retiredShardIds,
          minSnapshotSeq: msg.minSnapshotSeq,
          currentAudioBatchSeq: msg.currentAudioBatchSeq,
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
        this.guestWorld.replayNetworkAudioEvents(msg.events, msg.batchSeq, msg.roomSeed);
        break;
      }
    }
  }

  private handleGuestStateFrame(frame: MultiplayerStateFrame): void {
    if (this.mpRole !== 'guest' || !this.guestWorld) return;
    const prevSeed = this.guestWorld.roomSeed;
    const prevPhase = this.guestWorld.turnPhase;
    const prevPlayer = this.guestWorld.currentPlayer;
    const applied = this.guestWorld.applyNetworkStateFrame(frame);
    if (!applied) return;
    this.mpRoomSeed = this.guestWorld.roomSeed;

    if (prevSeed !== this.guestWorld.roomSeed) {
      this.clearGuestLocalBlockInteraction();
      this.updateCursorStyle();
      this.emitTelemetry(true);
      return;
    }

    if (
      prevPhase !== this.guestWorld.turnPhase ||
      prevPlayer !== this.guestWorld.currentPlayer
    ) {
      if (this.guestWorld.currentPlayer !== 1) {
        this.clearGuestLocalBlockInteraction();
      }
      this.updateCursorStyle();
      this.emitTelemetry(true);
    }
  }

  private sendGuestRemoteCommand(cmd: GuestRemoteCommand, nowOverrideMs?: number): void {
    if (this.mpRole !== 'guest' || !this.mpClient) return;
    if (!this.mpStatus.peerConnected) {
      if (
        this.mpStatus.connectionState === 'reconnecting' &&
        (cmd.kind === 'end_pull' ||
          cmd.kind === 'confirm_placement' ||
          cmd.kind === 'tap_block')
      ) {
        if (this.mpPendingDeferredGuestCmds.length < 8) {
          this.mpPendingDeferredGuestCmds.push(cmd);
        }
      }
      return;
    }
    this.mpGuestCmdSeq = (this.mpGuestCmdSeq + 1) >>> 0;
    this.mpLastGuestCmdSentMs =
      nowOverrideMs ?? (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this.mpClient.sendControl({
      type: 'guest_cmd',
      seq: this.mpGuestCmdSeq,
      roomSeed: this.mpRoomSeed,
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
      const linkReadyOrMidDragFailover =
        this.mpStatus.peerConnected ||
        (this.mpStatus.connectionState === 'reconnecting' && this.interactionMode === 'block');
      return (
        linkReadyOrMidDragFailover &&
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

  public setAiDifficulty(difficulty: AiDifficulty): void {
    this.aiDifficulty = difficulty;
    this.aiPlayerDifficulties[1] = difficulty;
    if (this.aiStage === 'scouting' || this.aiStage === 'idle') {
      this.resetAiController();
    }
    this.emitTelemetry(true);
  }

  public setAiOpponentEnabled(enabled: boolean, difficulty?: AiDifficulty): void {
    if (difficulty) {
      this.aiDifficulty = difficulty;
      this.aiPlayerDifficulties[1] = difficulty;
    }
    if (enabled && this.mpRole) {
      this.leaveOnlineRoom('party', 2);
    }
    if (!enabled) {
      this.aiVsAiEnabled = false;
      this.aiFastCadence = false;
    }
    this.aiOpponentEnabled = enabled;
    this.resetAiController();
    if (enabled && this.physics && (this.physics.gameMode !== 'party' || this.physics.playerCount !== 2)) {
      this.resetGame('party', 2);
      return;
    }
    this.emitTelemetry(true);
  }

  /**
   * Configures autonomous 2-player AI-vs-AI play for headless 30-turn Vitest benchmarks or spectator matches.
   */
  public setAiVsAiMode(
    enabled: boolean,
    options?: {
      fastCadence?: boolean;
      p0Difficulty?: AiDifficulty;
      p1Difficulty?: AiDifficulty;
    }
  ): void {
    if (enabled && this.mpRole) {
      this.leaveOnlineRoom('party', 2);
    }
    this.aiVsAiEnabled = enabled;
    this.aiOpponentEnabled = enabled;
    this.aiFastCadence = enabled ? Boolean(options?.fastCadence) : false;
    if (options?.p0Difficulty) {
      this.aiPlayerDifficulties[0] = options.p0Difficulty;
    }
    if (options?.p1Difficulty) {
      this.aiPlayerDifficulties[1] = options.p1Difficulty;
      this.aiDifficulty = options.p1Difficulty;
    }
    this.resetAiController();
    if (enabled && this.physics && (this.physics.gameMode !== 'party' || this.physics.playerCount !== 2)) {
      this.physics.rebuildTower('party', 2);
    }
    this.emitTelemetry(true);
  }

  private getActiveTurnAiDifficulty(): AiDifficulty {
    if (this.aiVsAiEnabled && this.physics) {
      return this.aiPlayerDifficulties[this.physics.currentPlayer === 0 ? 0 : 1];
    }
    return this.aiDifficulty;
  }

  private resetAiController(): void {
    this.aiStage = 'idle';
    this.aiTimerSec = 0;
    this.aiPullProgressSec = 0;
    this.aiTargetBlockId = null;
    this.aiTargetYawOffsetRad = 0;
    this.aiThinking = false;
    this.aiActionLabel = null;
  }

  private isAiTurnActive(): boolean {
    if (this.mpRole || !this.aiOpponentEnabled || this.modalOpen || !this.physics) {
      return false;
    }
    if (this.physics.gameMode !== 'party') return false;
    if (this.physics.turnPhase === 'COLLAPSED' || this.physics.turnPhase === 'SETTLING') {
      return false;
    }
    if (this.aiVsAiEnabled) {
      return this.physics.currentPlayer === 0 || this.physics.currentPlayer === 1;
    }
    return this.physics.currentPlayer === 1;
  }

  /**
   * Computes exact per-course beveled 2D convex support margins across every active course `k`
   * when `excludeBlockId` is removed and an optional prospective top block is stacked at `extraTopXZ`.
   */
  private evaluateCourseSupportMargins(
    excludeBlockId: number | null,
    extraTopXZ?: [number, number]
  ): {
    valid: boolean;
    minMarginDm: number;
    postComXZDm: [number, number];
    weakestLayer: number;
  } {
    const blocks = this.physics.blocks;
    const excludedLayer =
      excludeBlockId != null && blocks[excludeBlockId]
        ? blocks[excludeBlockId].assignedLayer
        : -1;
    const byLayer = new Map<number, typeof blocks>();
    let maxLayer = 0;
    let totalMass = 0;
    let sumX = 0;
    let sumZ = 0;

    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.shattered || b.extracted || b.id === excludeBlockId) continue;
      if (Math.hypot(b.position[0], b.position[2]) > 1.25) continue;
      let arr = byLayer.get(b.assignedLayer);
      if (!arr) {
        arr = [];
        byLayer.set(b.assignedLayer, arr);
      }
      arr.push(b);
      if (b.assignedLayer > maxLayer) maxLayer = b.assignedLayer;
      totalMass += 1;
      sumX += b.position[0];
      sumZ += b.position[2];
    }

    const topTargetLayer = this.physics.topSupportFrame.targetLayer;
    if (extraTopXZ) {
      totalMass += 1;
      sumX += extraTopXZ[0];
      sumZ += extraTopXZ[1];
      if (topTargetLayer > maxLayer) maxLayer = topTargetLayer;
    }

    if (totalMass <= 0) {
      return { valid: false, minMarginDm: -1, postComXZDm: [0, 0], weakestLayer: 0 };
    }

    const postComXZDm: [number, number] = [sumX / totalMass, sumZ / totalMass];
    const effHx = Math.max(0.1, BLOCK_HALF_X - 0.018);
    const effHz = Math.max(0.05, BLOCK_HALF_Z - 0.018);

    let minMarginDm = Infinity;
    let weakestLayer = 0;

    for (let k = 0; k <= maxLayer; k++) {
      const courseBlocks = byLayer.get(k);
      let aboveCount = 0;
      let aboveX = 0;
      let aboveZ = 0;

      for (let m = k + 1; m <= maxLayer; m++) {
        const upper = byLayer.get(m);
        if (!upper) continue;
        for (let j = 0; j < upper.length; j++) {
          aboveCount += 1;
          aboveX += upper[j].position[0];
          aboveZ += upper[j].position[2];
        }
      }
      if (extraTopXZ && topTargetLayer > k) {
        aboveCount += 1;
        aboveX += extraTopXZ[0];
        aboveZ += extraTopXZ[1];
      }

      if (aboveCount === 0) continue;
      if (!courseBlocks || courseBlocks.length === 0) {
        return { valid: false, minMarginDm: -1, postComXZDm, weakestLayer: k };
      }

      const refAxisX = quatRotateVec3(courseBlocks[0].rotation, [1, 0, 0]);
      const lenX = Math.hypot(refAxisX[0], refAxisX[2]) || 1.0;
      const vAxisX = refAxisX[0] / lenX;
      const vAxisZ = refAxisX[2] / lenX;
      const uAxisX = -vAxisZ;
      const uAxisZ = vAxisX;

      let uMin = Infinity;
      let uMax = -Infinity;
      let vMin = Infinity;
      let vMax = -Infinity;

      for (let j = 0; j < courseBlocks.length; j++) {
        const cb = courseBlocks[j];
        const bx = quatRotateVec3(cb.rotation, [1, 0, 0]);
        const bz = quatRotateVec3(cb.rotation, [0, 0, 1]);
        const cU = cb.position[0] * uAxisX + cb.position[2] * uAxisZ;
        const cV = cb.position[0] * vAxisX + cb.position[2] * vAxisZ;
        const spanU =
          effHx * Math.abs(bx[0] * uAxisX + bx[2] * uAxisZ) +
          effHz * Math.abs(bz[0] * uAxisX + bz[2] * uAxisZ);
        const spanV =
          effHx * Math.abs(bx[0] * vAxisX + bx[2] * vAxisZ) +
          effHz * Math.abs(bz[0] * vAxisX + bz[2] * vAxisZ);
        if (cU - spanU < uMin) uMin = cU - spanU;
        if (cU + spanU > uMax) uMax = cU + spanU;
        if (cV - spanV < vMin) vMin = cV - spanV;
        if (cV + spanV > vMax) vMax = cV + spanV;
      }

      const comAboveX = aboveX / aboveCount;
      const comAboveZ = aboveZ / aboveCount;
      const comU = comAboveX * uAxisX + comAboveZ * uAxisZ;
      const comV = comAboveX * vAxisX + comAboveZ * vAxisZ;
      const marginK = Math.min(comU - uMin, uMax - comU, comV - vMin, vMax - comV);

      if (marginK < minMarginDm) {
        minMarginDm = marginK;
        weakestLayer = k;
      }

      // Dry-masonry contact-patch check: each unbonded prism in layer `k + 1` only receives upward
      // normal force from blocks in layer `k` whose 2D footprint actually overlaps that prism!
      const minIndivSupportDm = 0.048;
      const evalPrismContactMargin = (
        pU: number,
        pV: number,
        pSpanU: number,
        pSpanV: number
      ): number => {
        let patchUMin = Infinity;
        let patchUMax = -Infinity;
        let patchVMin = Infinity;
        let patchVMax = -Infinity;
        for (let ci = 0; ci < courseBlocks.length; ci++) {
          const cb = courseBlocks[ci];
          const bx = quatRotateVec3(cb.rotation, [1, 0, 0]);
          const bz = quatRotateVec3(cb.rotation, [0, 0, 1]);
          const cU = cb.position[0] * uAxisX + cb.position[2] * uAxisZ;
          const cV = cb.position[0] * vAxisX + cb.position[2] * vAxisZ;
          const spanU =
            effHx * Math.abs(bx[0] * uAxisX + bx[2] * uAxisZ) +
            effHz * Math.abs(bz[0] * uAxisX + bz[2] * uAxisZ);
          const spanV =
            effHx * Math.abs(bx[0] * vAxisX + bx[2] * vAxisZ) +
            effHz * Math.abs(bz[0] * vAxisX + bz[2] * vAxisZ);
          const iu0 = Math.max(cU - spanU, pU - pSpanU);
          const iu1 = Math.min(cU + spanU, pU + pSpanU);
          const iv0 = Math.max(cV - spanV, pV - pSpanV);
          const iv1 = Math.min(cV + spanV, pV + pSpanV);
          if (iu0 < iu1 && iv0 < iv1) {
            if (iu0 < patchUMin) patchUMin = iu0;
            if (iu1 > patchUMax) patchUMax = iu1;
            if (iv0 < patchVMin) patchVMin = iv0;
            if (iv1 > patchVMax) patchVMax = iv1;
          }
        }
        if (!Number.isFinite(patchUMin)) return -0.5;
        return Math.min(pU - patchUMin, patchUMax - pU, pV - patchVMin, patchVMax - pV);
      };

      const directAbove = k === excludedLayer ? byLayer.get(k + 1) : undefined;
      if (directAbove) {
        for (let j = 0; j < directAbove.length; j++) {
          const db = directAbove[j];
          const dbx = quatRotateVec3(db.rotation, [1, 0, 0]);
          const dbz = quatRotateVec3(db.rotation, [0, 0, 1]);
          const dbU = db.position[0] * uAxisX + db.position[2] * uAxisZ;
          const dbV = db.position[0] * vAxisX + db.position[2] * vAxisZ;
          const dbSpanU =
            effHx * Math.abs(dbx[0] * uAxisX + dbx[2] * uAxisZ) +
            effHz * Math.abs(dbz[0] * uAxisX + dbz[2] * uAxisZ);
          const dbSpanV =
            effHx * Math.abs(dbx[0] * vAxisX + dbx[2] * vAxisZ) +
            effHz * Math.abs(dbz[0] * vAxisX + dbz[2] * vAxisZ);
          const indivMargin = evalPrismContactMargin(dbU, dbV, dbSpanU, dbSpanV);
          if (indivMargin < minIndivSupportDm) {
            const deficit = indivMargin - minIndivSupportDm;
            if (deficit < minMarginDm) {
              minMarginDm = deficit;
              weakestLayer = k;
            }
          }
        }
      }
      if (extraTopXZ && topTargetLayer === k + 1) {
        // Orthogonal top course has length along course k's spacing axis `u` (`effHx`) and width along `v` (`effHz`)
        const exU = extraTopXZ[0] * uAxisX + extraTopXZ[1] * uAxisZ;
        const exV = extraTopXZ[0] * vAxisX + extraTopXZ[1] * vAxisZ;
        const exMargin = evalPrismContactMargin(exU, exV, effHx, effHz);
        if (exMargin < minIndivSupportDm) {
          const deficit = exMargin - minIndivSupportDm;
          if (deficit < minMarginDm) {
            minMarginDm = deficit;
            weakestLayer = k;
          }
        }
      }
    }

    if (!Number.isFinite(minMarginDm)) {
      minMarginDm = 0.25;
    }

    return {
      valid: minMarginDm > 0,
      minMarginDm,
      postComXZDm,
      weakestLayer,
    };
  }

  /**
   * Optimizes continuous extraction azimuth (`pullDirXZ`) and grab end (`dirSign`) for a target block.
   * `Master Glazier` combines Brewster photoelastic end-bearing wrenches with continuous OBB neighbor-channel
   * clearance sweeping (`±13.75 deg`) so asymmetric or micro-rotated courses extract without lateral rubbing.
   */
  public computeOptimalExtractionPlan(
    blockId: number,
    difficulty: AiDifficulty = this.getActiveTurnAiDifficulty()
  ): TouchExtractionPlan | null {
    if (!this.physics) return null;
    const blk = this.physics.blocks[blockId];
    if (!blk || blk.shattered) return null;

    const axisX = quatRotateVec3(blk.rotation, [1, 0, 0]);
    const horizLen = Math.hypot(axisX[0], axisX[2]) || 1.0;
    const ux = axisX[0] / horizLen;
    const uz = axisX[2] / horizLen;

    const sameLayerNeighbors = this.physics.blocks.filter(
      (nb) =>
        !nb.shattered &&
        !nb.extracted &&
        nb.id !== blk.id &&
        nb.assignedLayer === blk.assignedLayer &&
        Math.hypot(nb.position[0] - blk.position[0], nb.position[2] - blk.position[2]) < 0.85
    );

    let layerCenterX = blk.position[0];
    let layerCenterZ = blk.position[2];
    if (sameLayerNeighbors.length > 0) {
      for (let i = 0; i < sameLayerNeighbors.length; i++) {
        layerCenterX += sameLayerNeighbors[i].position[0];
        layerCenterZ += sameLayerNeighbors[i].position[2];
      }
      layerCenterX /= sameLayerNeighbors.length + 1;
      layerCenterZ /= sameLayerNeighbors.length + 1;
    }

    if (difficulty === 'apprentice') {
      const dirSign: 1 | -1 =
        blk.position[0] * ux + blk.position[2] * uz >= 0 ? 1 : -1;
      const pullX = ux * dirSign;
      const pullZ = uz * dirSign;
      const localGrab: Vec3 = [BLOCK_HALF_X * 0.84 * dirSign, 0, 0];
      const grabOffset = quatRotateVec3(blk.rotation, localGrab);
      return {
        blockId: blk.id,
        pullDirXZ: [pullX, pullZ],
        azimuthRad: Math.atan2(pullZ, pullX),
        azimuthOffsetDeg: 0,
        dirSign,
        localGrab,
        startWorld: [
          blk.position[0] + grabOffset[0],
          blk.position[1] + grabOffset[1],
          blk.position[2] + grabOffset[2],
        ],
        minNeighborClearanceDm: 0.02,
      };
    }

    // Master Glazier: inspect photoelastic end-bearing wrench [-X, mid, +X, bend] + continuous azimuth clearance
    this.scratchWrench4.fill(0);
    if (typeof this.physics.writeBlockPhotoelasticWrench === 'function') {
      this.physics.writeBlockPhotoelasticWrench(blk.id, this.scratchWrench4, 0);
    }
    const wNeg = this.scratchWrench4[0];
    const wPos = this.scratchWrench4[2];

    const blkAxisZ = quatRotateVec3(blk.rotation, [0, 0, 1]);
    const bzHorizLen = Math.hypot(blkAxisZ[0], blkAxisZ[2]) || 1.0;
    const wzX = blkAxisZ[0] / bzHorizLen;
    const wzZ = blkAxisZ[2] / bzHorizLen;
    const effHx = BLOCK_HALF_X - 0.002;
    const effHz = BLOCK_HALF_Z - 0.002;

    const evaluateSlideClearance = (dirX: number, dirZ: number): number => {
      if (sameLayerNeighbors.length === 0) return 0.35;
      let worstClearance = Infinity;
      const samples = [0.06, 0.18, 0.34, 0.52, 0.68];
      for (let si = 0; si < samples.length; si++) {
        const s = samples[si];
        const cx = blk.position[0] + dirX * s;
        const cz = blk.position[2] + dirZ * s;
        for (let ni = 0; ni < sameLayerNeighbors.length; ni++) {
          const nb = sameLayerNeighbors[ni];
          const nbxRaw = quatRotateVec3(nb.rotation, [1, 0, 0]);
          const nbzRaw = quatRotateVec3(nb.rotation, [0, 0, 1]);
          const nbxLen = Math.hypot(nbxRaw[0], nbxRaw[2]) || 1.0;
          const nbzLen = Math.hypot(nbzRaw[0], nbzRaw[2]) || 1.0;
          const nxX = nbxRaw[0] / nbxLen;
          const nxZ = nbxRaw[2] / nbxLen;
          const nzX = nbzRaw[0] / nbzLen;
          const nzZ = nbzRaw[2] / nbzLen;

          const relX = nb.position[0] - cx;
          const relZ = nb.position[2] - cz;

          // Exact 4-axis 2D Separating Axis Theorem signed OBB-to-OBB clearance across {bx, bz, nbx, nbz}
          const axes: ReadonlyArray<[number, number]> = [
            [ux, uz],
            [wzX, wzZ],
            [nxX, nxZ],
            [nzX, nzZ],
          ];
          let satGap = -Infinity;
          for (let ai = 0; ai < 4; ai++) {
            const ax = axes[ai][0];
            const az = axes[ai][1];
            const centerProj = Math.abs(relX * ax + relZ * az);
            const rBlk =
              effHx * Math.abs(ux * ax + uz * az) + effHz * Math.abs(wzX * ax + wzZ * az);
            const rNb =
              effHx * Math.abs(nxX * ax + nxZ * az) + effHz * Math.abs(nzX * ax + nzZ * az);
            const axisGap = centerProj - rBlk - rNb;
            if (axisGap > satGap) {
              satGap = axisGap;
            }
          }

          if (satGap < worstClearance) {
            worstClearance = satGap;
          }
        }
      }
      return Number.isFinite(worstClearance) ? worstClearance : 0.35;
    };

    let bestSign: 1 | -1 = 1;
    let bestAzimuthRad = Math.atan2(uz, ux);
    let bestOffsetRad = 0;
    let bestScore = -Infinity;
    let bestMinClearance = 0;

    const signs: Array<1 | -1> = [1, -1];
    const maxSweepRad = 0.24; // ±13.75 deg continuous azimuth search
    const sweepSteps = 32;

    for (let si = 0; si < signs.length; si++) {
      const sign = signs[si];
      const baseAz = Math.atan2(uz * sign, ux * sign);
      const protrusion =
        (blk.position[0] - layerCenterX) * (ux * sign) +
        (blk.position[2] - layerCenterZ) * (uz * sign);
      // Prefer pulling toward the end that already protrudes and away from heavier trailing pinch
      const wrenchBias = (wPos - wNeg) * sign * 0.08;

      for (let k = 0; k <= sweepSteps; k++) {
        const deltaAz = -maxSweepRad + (2 * maxSweepRad * k) / sweepSteps;
        const candAz = baseAz + deltaAz;
        const dirX = Math.cos(candAz);
        const dirZ = Math.sin(candAz);
        const minClear = evaluateSlideClearance(dirX, dirZ);

        // Steep penalty if trajectory grazes an adjacent neighbor (< 1.5 mm), smooth reward for wider channel up to 12 mm
        const contactPenalty = minClear < 0.015 ? (minClear - 0.015) * 18.0 : 0;
        const channelReward = Math.min(0.12, Math.max(-0.12, minClear)) * 2.2;
        const axialReg = -0.32 * deltaAz * deltaAz;

        const totalScore =
          protrusion * 0.65 + wrenchBias + contactPenalty + channelReward + axialReg;

        if (totalScore > bestScore) {
          bestScore = totalScore;
          bestSign = sign;
          bestAzimuthRad = candAz;
          bestOffsetRad = deltaAz;
          bestMinClearance = minClear;
        }
      }
    }

    const pullDirXZ: [number, number] = [Math.cos(bestAzimuthRad), Math.sin(bestAzimuthRad)];
    const localGrab: Vec3 = [BLOCK_HALF_X * 0.84 * bestSign, 0, 0];
    const grabOffset = quatRotateVec3(blk.rotation, localGrab);
    return {
      blockId: blk.id,
      pullDirXZ,
      azimuthRad: bestAzimuthRad,
      azimuthOffsetDeg: +((bestOffsetRad * 180) / Math.PI).toFixed(2),
      dirSign: bestSign,
      localGrab,
      startWorld: [
        blk.position[0] + grabOffset[0],
        blk.position[1] + grabOffset[1],
        blk.position[2] + grabOffset[2],
      ],
      minNeighborClearanceDm: +bestMinClearance.toFixed(4),
    };
  }

  /**
   * Computes top-course placement pose `(targetXZ, localVU, yawOffsetRad)`.
   * - `Apprentice`: counterbalances global CoM toward `[0, 0]` to stabilize the stack.
   * - `Master Glazier`: searches continuous eccentric longitudinal & lateral poses `(v, u, yaw)` that
   *   MAXIMIZE surviving global CoM eccentricity & opponent bottleneck stress while guaranteeing both
   *   local beveled support on `targetLayer - 1` and global `P-Delta` stability across every course `0..targetLayer-1`.
   */
  public computeAdversarialTopPlacement(
    excludeBlockId?: number | null,
    difficulty: AiDifficulty = this.getActiveTurnAiDifficulty()
  ): AdversarialPlacementPlan {
    const frame = this.physics.topSupportFrame;
    const lc = frame.layerCenter;
    const activeExcludeId =
      excludeBlockId !== undefined ? excludeBlockId : this.physics.activeBlockId;

    if (difficulty === 'apprentice') {
      const slots = this.physics.openTopSlots;
      let bestXZ: [number, number] = [lc[0], lc[2]];
      if (slots.length > 0) {
        let bestSlotScore = -Infinity;
        for (let i = 0; i < slots.length; i++) {
          const sl = slots[i];
          const candXZ: [number, number] = [sl.position[0], sl.position[2]];
          const postCand = this.evaluateCourseSupportMargins(activeExcludeId, candXZ);
          const candEccDm = Math.hypot(postCand.postComXZDm[0], postCand.postComXZDm[1]);
          const slotScore =
            (postCand.valid ? 0 : -2.0) + postCand.minMarginDm * 1.5 - candEccDm * 4.5;
          if (slotScore > bestSlotScore) {
            bestSlotScore = slotScore;
            bestXZ = candXZ;
          }
        }
      }
      const rx = bestXZ[0] - lc[0];
      const rz = bestXZ[1] - lc[2];
      const v = rx * frame.lengthAxis[0] + rz * frame.lengthAxis[2];
      const u = rx * frame.spacingAxis[0] + rz * frame.spacingAxis[2];
      const post = this.evaluateCourseSupportMargins(activeExcludeId, bestXZ);
      return {
        targetXZ: bestXZ,
        localVU: [v, u],
        yawOffsetRad: 0,
        predictedComXZCm: [post.postComXZDm[0] * 10, post.postComXZDm[1] * 10],
        predictedMinCourseMarginDm: post.minMarginDm,
      };
    }

    // Master Glazier adversarial eccentric search
    const sameLayerObstacles = this.physics.blocks.filter(
      (b) =>
        !b.shattered &&
        !b.extracted &&
        b.id !== activeExcludeId &&
        b.assignedLayer === frame.targetLayer &&
        Math.hypot(b.position[0] - lc[0], b.position[2] - lc[2]) < 0.95
    );

    const supportBelowBlocks = this.physics.blocks.filter(
      (b) =>
        !b.shattered &&
        !b.extracted &&
        b.id !== activeExcludeId &&
        b.assignedLayer === frame.targetLayer - 1 &&
        Math.hypot(b.position[0] - lc[0], b.position[2] - lc[2]) < 0.95
    );

    // Compute local beveled support envelope on `targetLayer - 1` in `(v, u)` coordinates
    let supVMin = -0.34;
    let supVMax = 0.34;
    let supUMin = -0.34;
    let supUMax = 0.34;
    if (supportBelowBlocks.length > 0) {
      supVMin = Infinity;
      supVMax = -Infinity;
      supUMin = Infinity;
      supUMax = -Infinity;
      const effHx = BLOCK_HALF_X - 0.022;
      const effHz = BLOCK_HALF_Z - 0.022;
      for (let i = 0; i < supportBelowBlocks.length; i++) {
        const sb = supportBelowBlocks[i];
        const bx = quatRotateVec3(sb.rotation, [1, 0, 0]);
        const bz = quatRotateVec3(sb.rotation, [0, 0, 1]);
        const rx = sb.position[0] - lc[0];
        const rz = sb.position[2] - lc[2];
        const sv = rx * frame.lengthAxis[0] + rz * frame.lengthAxis[2];
        const su = rx * frame.spacingAxis[0] + rz * frame.spacingAxis[2];
        const spanV =
          effHx * Math.abs(bx[0] * frame.lengthAxis[0] + bx[2] * frame.lengthAxis[2]) +
          effHz * Math.abs(bz[0] * frame.lengthAxis[0] + bz[2] * frame.lengthAxis[2]);
        const spanU =
          effHx * Math.abs(bx[0] * frame.spacingAxis[0] + bx[2] * frame.spacingAxis[2]) +
          effHz * Math.abs(bz[0] * frame.spacingAxis[0] + bz[2] * frame.spacingAxis[2]);
        if (sv - spanV < supVMin) supVMin = sv - spanV;
        if (sv + spanV > supVMax) supVMax = sv + spanV;
        if (su - spanU < supUMin) supUMin = su - spanU;
        if (su + spanU > supUMax) supUMax = su + spanU;
      }
    }

    const obstacleUs = sameLayerObstacles.map((ob) => {
      const rx = ob.position[0] - lc[0];
      const rz = ob.position[2] - lc[2];
      return rx * frame.spacingAxis[0] + rz * frame.spacingAxis[2];
    });

    const minObstacleGapU = 2 * BLOCK_HALF_Z + 0.010; // >= 2.60 cm center-to-center (~1.0 mm clear gap)
    const leanDeg = this.physics.getTelemetry().leanAngleDeg;
    const minRequiredCourseMarginDm = Math.min(
      0.132,
      0.068 + 0.006 * leanDeg + this.physics.blocksStacked * 0.0014
    );
    // P-Delta elastic compliance envelope in decimeters (0.058 dm = 0.58 cm up to 21L, tapering to 0.38 cm at 28L+):
    // drives CoM strongly into the adversarial danger band while preventing runaway uphill-hollowing collapse
    const maxSafeGlobalEccDm = Math.max(
      0.038,
      0.058 - Math.max(0, this.physics.currentLayers - 21) * 0.0028 - leanDeg * 0.015
    );

    // Keep lateral u aligned with clean 3-block course lanes (+/- eccentric outer edge shifts) so courses pack 3 blocks cleanly
    const candidateUs: number[] = [
      -0.285, -0.268, -0.25, -0.232,
      -0.025, 0.0, 0.025,
      0.232, 0.25, 0.268, 0.285,
    ];
    for (let i = 0; i < this.physics.openTopSlots.length; i++) {
      const sl = this.physics.openTopSlots[i];
      const rx = sl.position[0] - lc[0];
      const rz = sl.position[2] - lc[2];
      candidateUs.push(rx * frame.spacingAxis[0] + rz * frame.spacingAxis[2]);
    }

    const candidateVs: number[] = [
      -0.175, -0.145, -0.11, -0.075, -0.035, 0.0, 0.035, 0.075, 0.11, 0.145, 0.175,
    ];
    const yawCandidates = sameLayerObstacles.length >= 2 ? [-0.075, 0, 0.075] : [0];

    let bestPlan: AdversarialPlacementPlan | null = null;
    let bestScore = -Infinity;

    for (let iu = 0; iu < candidateUs.length; iu++) {
      const cu = candidateUs[iu];
      if (cu < supUMin + 0.065 || cu > supUMax - 0.065) continue;

      let collidesSameLayer = false;
      for (let oi = 0; oi < obstacleUs.length; oi++) {
        if (Math.abs(cu - obstacleUs[oi]) < minObstacleGapU) {
          collidesSameLayer = true;
          break;
        }
      }
      if (collidesSameLayer) continue;

      for (let iv = 0; iv < candidateVs.length; iv++) {
        const cv = candidateVs[iv];
        if (cv < supVMin + 0.085 || cv > supVMax - 0.085) continue;

        const px = lc[0] + frame.lengthAxis[0] * cv + frame.spacingAxis[0] * cu;
        const pz = lc[2] + frame.lengthAxis[2] * cv + frame.spacingAxis[2] * cu;
        if (Math.hypot(px, pz) > 0.44) continue;
        const post = this.evaluateCourseSupportMargins(activeExcludeId, [px, pz]);
        if (!post.valid || post.minMarginDm < minRequiredCourseMarginDm) continue;

        const postEccDm = Math.hypot(post.postComXZDm[0], post.postComXZDm[1]);
        if (postEccDm > maxSafeGlobalEccDm) continue;

        const localEccDm = Math.hypot(cv, cu);
        // Reward driving global CoM eccentricity outward while preserving safe settling margin
        const marginSafetyBonus = Math.min(0.045, post.minMarginDm - minRequiredCourseMarginDm) * 1.4;

        for (let iy = 0; iy < yawCandidates.length; iy++) {
          const candYaw = yawCandidates[iy];
          let yawClearOk = true;
          if (Math.abs(candYaw) > 1e-4 && obstacleUs.length > 0) {
            const extraWidth = BLOCK_HALF_X * Math.abs(Math.sin(candYaw));
            for (let oi = 0; oi < obstacleUs.length; oi++) {
              if (Math.abs(cu - obstacleUs[oi]) < minObstacleGapU + extraWidth) {
                yawClearOk = false;
                break;
              }
            }
          }
          if (!yawClearOk) continue;

          const score =
            postEccDm * 2.4 +
            localEccDm * 0.42 +
            marginSafetyBonus +
            Math.abs(candYaw) * 0.12;

          if (score > bestScore) {
            bestScore = score;
            bestPlan = {
              targetXZ: [px, pz],
              localVU: [cv, cu],
              yawOffsetRad: candYaw,
              predictedComXZCm: [
                +(post.postComXZDm[0] * 10).toFixed(3),
                +(post.postComXZDm[1] * 10).toFixed(3),
              ],
              predictedMinCourseMarginDm: +post.minMarginDm.toFixed(4),
            };
          }
        }
      }
    }

    if (bestPlan) {
      return bestPlan;
    }

    // Fallback when tower is already at/above its P-Delta envelope: hold near maxSafeGlobalEccDm with high course margin
    let safestPlan: AdversarialPlacementPlan = this.computeAdversarialTopPlacement(
      activeExcludeId,
      'apprentice'
    );
    const initFallbackEccDm =
      Math.hypot(safestPlan.predictedComXZCm[0], safestPlan.predictedComXZCm[1]) * 0.1;
    let safestScore =
      safestPlan.predictedMinCourseMarginDm * 2.4 -
      Math.max(0, initFallbackEccDm - maxSafeGlobalEccDm) * 6.0 -
      initFallbackEccDm * 1.2;

    for (let iu = 0; iu < candidateUs.length; iu++) {
      const cu = candidateUs[iu];
      if (cu < supUMin + 0.065 || cu > supUMax - 0.065) continue;
      let collides = false;
      for (let oi = 0; oi < obstacleUs.length; oi++) {
        if (Math.abs(cu - obstacleUs[oi]) < minObstacleGapU) {
          collides = true;
          break;
        }
      }
      if (collides) continue;
      for (let iv = 0; iv < candidateVs.length; iv++) {
        const cv = candidateVs[iv];
        if (cv < supVMin + 0.085 || cv > supVMax - 0.085) continue;
        const px = lc[0] + frame.lengthAxis[0] * cv + frame.spacingAxis[0] * cu;
        const pz = lc[2] + frame.lengthAxis[2] * cv + frame.spacingAxis[2] * cu;
        const post = this.evaluateCourseSupportMargins(activeExcludeId, [px, pz]);
        if (!post.valid) continue;
        const postEccDm = Math.hypot(post.postComXZDm[0], post.postComXZDm[1]);
        const candScore =
          post.minMarginDm * 2.4 -
          Math.max(0, postEccDm - maxSafeGlobalEccDm) * 6.0 -
          postEccDm * 1.2;
        if (candScore > safestScore) {
          safestScore = candScore;
          safestPlan = {
            targetXZ: [px, pz],
            localVU: [cv, cu],
            yawOffsetRad: 0,
            predictedComXZCm: [
              +(post.postComXZDm[0] * 10).toFixed(3),
              +(post.postComXZDm[1] * 10).toFixed(3),
            ],
            predictedMinCourseMarginDm: +post.minMarginDm.toFixed(4),
          };
        }
      }
    }

    return safestPlan;
  }

  private selectAiTargetFromOracle(
    candidates: BlockCriticalityCandidate[],
    difficulty: AiDifficulty
  ): BlockCriticalityCandidate | undefined {
    const rawLegal = candidates.filter((c) => {
      const b = this.physics.blocks[c.blockId];
      return Boolean(b && !b.shattered && !b.extracted && b.isLegalTarget);
    });
    if (rawLegal.length === 0) return undefined;
    const nonAborted = rawLegal.filter((c) => !this.aiAbortedBlockIds.has(c.blockId));
    const legal = nonAborted.length > 0 ? nonAborted : rawLegal;

    // Count active blocks per layer so we know which lower courses are narrow 1-prism hinges
    const layerCounts = new Map<number, number>();
    for (let i = 0; i < this.physics.blocks.length; i++) {
      const b = this.physics.blocks[i];
      if (!b || b.shattered || b.extracted) continue;
      layerCounts.set(b.assignedLayer, (layerCounts.get(b.assignedLayer) ?? 0) + 1);
    }
    const countCrossHingesBelow = (layer: number): number => {
      let hinges = 0;
      for (let m = 0; m < layer; m++) {
        if ((layerCounts.get(m) ?? 0) === 1 && m % 2 !== layer % 2) {
          hinges++;
        }
      }
      return hinges;
    };

    const leanDeg = this.physics.getTelemetry().leanAngleDeg;
    const minExtractMarginDm = Math.min(
      0.122,
      0.078 + 0.005 * leanDeg + this.physics.blocksStacked * 0.0008
    );

    const topClampFloorLayer = Math.max(2, this.physics.currentLayers - 3);

    if (difficulty === 'apprentice') {
      // Apprentice prioritizes wide remaining course support, free-sliding prisms (predictedSlideMm >= 0.35), and low CoM drift
      let bestApp = legal[0];
      let bestAppScore = -Infinity;
      for (let i = 0; i < legal.length; i++) {
        const c = legal[i];
        const blk = this.physics.blocks[c.blockId];
        const surv = this.evaluateCourseSupportMargins(c.blockId);
        const survEccDm = Math.hypot(surv.postComXZDm[0], surv.postComXZDm[1]);
        const underClampedTopPen =
          (blk.assignedLayer >= topClampFloorLayer ? -1.8 : 0) +
          (blk.assignedLayer >= 18 ? -0.55 : 0) -
          countCrossHingesBelow(blk.assignedLayer) * 0.48;
        const jammedOrUnsafe =
          c.collapseTriggered ||
          !surv.valid ||
          surv.minMarginDm < minExtractMarginDm ||
          c.safetyScore01 < 0.45 ||
          c.predictedSlideMm < 0.35;
        const safeGatePenalty = jammedOrUnsafe ? -3.2 : 0;
        const s =
          safeGatePenalty +
          underClampedTopPen +
          Math.min(0.36, surv.minMarginDm) * 2.8 +
          c.safetyScore01 * 0.95 -
          survEccDm * 3.8;
        if (s > bestAppScore) {
          bestAppScore = s;
          bestApp = c;
        }
      }
      return bestApp;
    }

    // Master Glazier 2-ply joint (extractCandidate + adversarialTopPlacement) evaluation
    const maxSafeGlobalEccDm = Math.max(
      0.038,
      0.058 - Math.max(0, this.physics.currentLayers - 21) * 0.0028 - leanDeg * 0.015
    );
    let bestCand: BlockCriticalityCandidate | undefined;
    let best2PlyScore = -Infinity;

    for (let i = 0; i < legal.length; i++) {
      const cand = legal[i];
      const blk = this.physics.blocks[cand.blockId];
      if (cand.collapseTriggered || cand.safetyScore01 < 0.45 || cand.predictedSlideMm < 0.35) {
        continue;
      }
      const midExtract = this.evaluateCourseSupportMargins(cand.blockId);
      if (!midExtract.valid || midExtract.minMarginDm < minExtractMarginDm) continue;

      const plan = this.computeAdversarialTopPlacement(cand.blockId, 'master');
      if (plan.predictedMinCourseMarginDm < minExtractMarginDm) continue;

      const postEccDm =
        Math.hypot(plan.predictedComXZCm[0], plan.predictedComXZCm[1]) * 0.1;
      const extractPlan = this.computeOptimalExtractionPlan(cand.blockId, 'master');
      const channelBonus = extractPlan
        ? Math.min(0.08, Math.max(-0.08, extractPlan.minNeighborClearanceDm)) * 1.5
        : 0;
      const eccReward =
        postEccDm <= maxSafeGlobalEccDm
          ? postEccDm * 2.4
          : maxSafeGlobalEccDm * 2.4 - (postEccDm - maxSafeGlobalEccDm) * 5.5;
      const underClampedTopPen =
        (blk.assignedLayer >= topClampFloorLayer ? -1.8 : 0) +
        (blk.assignedLayer >= 18 ? -0.55 : 0) -
        countCrossHingesBelow(blk.assignedLayer) * 0.48;

      const cemTwoPlyBonus =
        cand.adversarialPlacement?.collapseTriggered
          ? -0.35
          : (cand.twoPlyMasterScore01 ?? cand.safetyScore01) * 0.05;

      const score2Ply =
        eccReward +
        underClampedTopPen +
        cand.safetyScore01 * 0.92 +
        Math.min(0.32, midExtract.minMarginDm) * 1.65 +
        channelBonus +
        cemTwoPlyBonus;

      if (score2Ply > best2PlyScore) {
        best2PlyScore = score2Ply;
        bestCand = cand;
      }
    }

    if (bestCand) return bestCand;

    // High-stress late-game fallback: maximize combined survival margin + actual slide ease while avoiding cross-axis hinges
    let fallbackCand = legal[0];
    let fallbackScore = -Infinity;
    for (let i = 0; i < legal.length; i++) {
      const cand = legal[i];
      const blk = this.physics.blocks[cand.blockId];
      const surv = this.evaluateCourseSupportMargins(cand.blockId);
      const survEccDm = Math.hypot(surv.postComXZDm[0], surv.postComXZDm[1]);
      const underClampedTopPen =
        (blk.assignedLayer >= topClampFloorLayer ? -1.4 : 0) -
        countCrossHingesBelow(blk.assignedLayer) * 0.45;
      const jamPen = cand.predictedSlideMm < 0.25 ? -1.8 : 0;
      const s =
        (cand.collapseTriggered ? -2.5 : 0) +
        (surv.valid ? 0 : -1.8) +
        jamPen +
        underClampedTopPen +
        surv.minMarginDm * 3.5 +
        cand.safetyScore01 * 1.3 -
        survEccDm * 2.2;
      if (s > fallbackScore) {
        fallbackScore = s;
        fallbackCand = cand;
      }
    }
    return fallbackCand;
  }

  private applyAiOracleDecision(candidates: BlockCriticalityCandidate[]): void {
    if (!this.isAiTurnActive() || this.aiStage !== 'awaiting_oracle') return;
    const difficulty = this.getActiveTurnAiDifficulty();
    const labelPrefix = difficulty === 'master' ? 'Master Glazier' : 'Apprentice AI';
    const chosen = this.selectAiTargetFromOracle(candidates, difficulty);
    const blk = chosen ? this.physics.blocks[chosen.blockId] : undefined;
    if (!chosen || !blk || blk.shattered || !blk.isLegalTarget) {
      this.resetAiController();
      return;
    }
    this.aiTargetBlockId = blk.id;

    if (!this.aiFastCadence) {
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
        this.aiActionLabel = `${labelPrefix} Testing Column…`;
        this.emitTelemetry(true);
        return;
      }
    }

    this.physics.hoveredBlockId = blk.id;
    if (!this.aiFastCadence) {
      this.physics.tapNudgeBlock(blk.id, [1, 0, 0], [BLOCK_HALF_X * 0.82, 0, 0]);
    }
    this.aiStage = 'post_tap';
    this.aiTimerSec = 0;
    this.aiActionLabel = `${labelPrefix} Selecting Prism…`;
    this.emitTelemetry(true);
  }

  private dispatchAiCounterfactualOracle(difficulty: AiDifficulty): void {
    const maxC = this.aiFastCadence
      ? difficulty === 'master'
        ? 10
        : 8
      : difficulty === 'master'
        ? 14
        : 10;
    // Keep >= 6 solver steps so `predictedSlideMm / 1.8` cleanly separates free prisms from jammed bearing columns
    const rolloutSteps = 6;

    const layerCounts = new Map<number, number>();
    for (let i = 0; i < this.physics.blocks.length; i++) {
      const b = this.physics.blocks[i];
      if (!b || b.shattered || b.extracted) continue;
      layerCounts.set(b.assignedLayer, (layerCounts.get(b.assignedLayer) ?? 0) + 1);
    }

    const topClampFloorLayer = Math.max(2, this.physics.currentLayers - 3);
    const scoredLegal: Array<{ id: number; rankScore: number }> = [];
    for (let i = 0; i < this.physics.blocks.length; i++) {
      const b = this.physics.blocks[i];
      if (!b || b.shattered || b.extracted || !b.isLegalTarget) continue;
      if (this.aiAbortedBlockIds.has(b.id)) continue;
      const surv = this.evaluateCourseSupportMargins(b.id);
      if (!surv.valid) continue;
      this.scratchWrench4.fill(0);
      if (typeof this.physics.writeBlockPhotoelasticWrench === 'function') {
        this.physics.writeBlockPhotoelasticWrench(b.id, this.scratchWrench4, 0);
      }
      const clampStress =
        this.scratchWrench4[0] + this.scratchWrench4[1] + this.scratchWrench4[2];
      let crossHinges = 0;
      for (let m = 0; m < b.assignedLayer; m++) {
        if ((layerCounts.get(m) ?? 0) === 1 && m % 2 !== b.assignedLayer % 2) {
          crossHinges++;
        }
      }
      const survEccDm = Math.hypot(surv.postComXZDm[0], surv.postComXZDm[1]);
      const underClampedTopPen =
        (b.assignedLayer >= topClampFloorLayer ? -1.8 : 0) +
        (b.assignedLayer >= 18 ? -0.55 : 0) -
        crossHinges * 0.48;
      const rankScore =
        underClampedTopPen +
        Math.min(0.36, surv.minMarginDm) * 2.4 -
        survEccDm * 2.0 -
        clampStress * 0.45;
      scoredLegal.push({ id: b.id, rankScore });
    }
    scoredLegal.sort((a, b) => b.rankScore - a.rankScore);
    const candidateIds =
      scoredLegal.length > 0 ? scoredLegal.slice(0, maxC).map((s) => s.id) : undefined;

    const maybeWorker = this.physics as unknown as {
      evaluateCounterfactualBlockCriticalityAsync?: (
        c: number,
        s: number,
        candidateBlockIds?: number[]
      ) => Promise<BlockCriticalityCandidate[]>;
    };
    if (typeof maybeWorker.evaluateCounterfactualBlockCriticalityAsync === 'function') {
      void maybeWorker
        .evaluateCounterfactualBlockCriticalityAsync(maxC, rolloutSteps, candidateIds)
        .then((candidates) => this.applyAiOracleDecision(candidates));
      return;
    }

    const syncCandidates = this.physics.evaluateCounterfactualBlockCriticality(
      maxC,
      rolloutSteps,
      candidateIds
    );
    this.applyAiOracleDecision(syncCandidates);
  }

  private updateMasterGlazierAi(dt: number): void {
    if (!this.isAiTurnActive()) {
      if (this.aiStage !== 'idle') {
        this.resetAiController();
      }
      return;
    }

    if (this.physics.blocksStacked !== this.aiLastBlocksStacked) {
      this.aiLastBlocksStacked = this.physics.blocksStacked;
      this.aiAbortedBlockIds.clear();
    }

    const difficulty = this.getActiveTurnAiDifficulty();
    const labelPrefix = difficulty === 'master' ? 'Master Glazier' : 'Apprentice AI';
    const scoutDelay = this.aiFastCadence ? 0.02 : 0.72;
    const probeDelay = this.aiFastCadence ? 0.02 : 0.88;
    const postTapDelay = this.aiFastCadence ? 0.02 : 0.82;

    if (this.aiStage === 'idle' && this.physics.turnPhase === 'SELECT_BLOCK') {
      this.aiStage = 'scouting';
      this.aiTimerSec = 0;
      this.aiThinking = true;
      this.aiActionLabel = `${labelPrefix} Reading Stack…`;
      this.emitTelemetry(true);
      return;
    }

    if (this.aiStage === 'scouting') {
      this.aiTimerSec += dt;
      if (this.aiTimerSec >= scoutDelay) {
        this.aiStage = 'awaiting_oracle';
        this.dispatchAiCounterfactualOracle(difficulty);
      }
      return;
    }

    if (this.aiStage === 'probe_tight') {
      this.aiTimerSec += dt;
      if (this.aiTimerSec >= probeDelay) {
        const blk = this.aiTargetBlockId !== null ? this.physics.blocks[this.aiTargetBlockId] : undefined;
        if (!blk || blk.shattered || !blk.isLegalTarget) {
          this.resetAiController();
          return;
        }
        this.physics.hoveredBlockId = blk.id;
        this.physics.tapNudgeBlock(blk.id, [1, 0, 0], [BLOCK_HALF_X * 0.82, 0, 0]);
        this.aiStage = 'post_tap';
        this.aiTimerSec = 0;
        this.aiActionLabel = `${labelPrefix} Selecting Prism…`;
        this.emitTelemetry(true);
      }
      return;
    }

    if (this.aiStage === 'post_tap') {
      this.aiTimerSec += dt;
      if (this.aiTimerSec >= postTapDelay && this.aiTargetBlockId !== null) {
        const plan = this.computeOptimalExtractionPlan(this.aiTargetBlockId, difficulty);
        if (!plan) {
          this.resetAiController();
          return;
        }
        const blkRef = this.physics.blocks[plan.blockId];
        if (blkRef && this.physics.startGrabBlock(plan.blockId, plan.localGrab, plan.startWorld)) {
          this.aiPullStartWorld = plan.startWorld;
          this.aiPullBlockInitXZ = [blkRef.position[0], blkRef.position[2]];
          this.aiPullDirXZ = plan.pullDirXZ;
          this.aiStage = 'pulling';
          this.aiTimerSec = 0;
          this.aiPullProgressSec = 0;
          this.aiThinking = false;
          this.aiActionLabel = `${labelPrefix} Drawing…`;
          this.emitTelemetry(true);
        } else {
          this.resetAiController();
        }
      }
      return;
    }

    if (this.aiStage === 'pulling') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        const placePlan = this.computeAdversarialTopPlacement(
          this.physics.activeBlockId,
          difficulty
        );
        this.aiHoverCurrentXZ = [
          this.physics.carriedTargetPos[0],
          this.physics.carriedTargetPos[2],
        ];
        this.aiHoverTargetXZ = placePlan.targetXZ;
        this.aiTargetYawOffsetRad = placePlan.yawOffsetRad;
        this.physics.resetCarriedBlockRotation();
        this.aiStage = 'placing';
        this.aiTimerSec = 0;
        this.aiActionLabel = `${labelPrefix} Stacking…`;
        this.emitTelemetry(true);
        return;
      }

      // Bind-and-release tactician reflex: if a grabbed prism binds hard (< 8 mm slide after 0.14 s)
      // while tension or tower sway rises, release gently before clearing the channel and pick a looser prism
      const pulledBlk =
        this.aiTargetBlockId !== null ? this.physics.blocks[this.aiTargetBlockId] : undefined;
      if (pulledBlk && this.aiTimerSec >= 0.14 && this.physics.turnPhase === 'PULLING') {
        const movedDm = Math.hypot(
          pulledBlk.position[0] - this.aiPullBlockInitXZ[0],
          pulledBlk.position[2] - this.aiPullBlockInitXZ[1]
        );
        const liveLeanDeg = this.physics.getTelemetry().leanAngleDeg;
        if (movedDm < 0.08 && (this.physics.grabResistance01 > 0.72 || liveLeanDeg > 0.90)) {
          this.aiAbortedBlockIds.add(pulledBlk.id);
          this.physics.endGrab(false);
          this.resetAiController();
          return;
        }
      }

      this.aiTimerSec += dt;
      const resistanceThrottle = this.physics.grabResistance01 > 0.55 ? 0.55 : 1.0;
      this.aiPullProgressSec += dt * resistanceThrottle;

      const nominalPullDur = this.aiFastCadence ? 0.56 : 1.52;
      const tPull = Math.min(1.0, this.aiPullProgressSec / nominalPullDur);
      const easePull = tPull * tPull * (3.0 - 2.0 * tPull);
      const extraReach =
        this.aiPullProgressSec > nominalPullDur
          ? Math.min(0.45, (this.aiPullProgressSec - nominalPullDur) * 0.35)
          : 0;
      const pullDistDm = 1.62 * easePull + extraReach;
      const targetWorld: Vec3 = [
        this.aiPullStartWorld[0] + this.aiPullDirXZ[0] * pullDistDm,
        this.aiPullStartWorld[1],
        this.aiPullStartWorld[2] + this.aiPullDirXZ[1] * pullDistDm,
      ];
      this.physics.updateGrabTarget(targetWorld);
      if (this.aiTimerSec > (this.aiFastCadence ? 2.4 : 4.2) && this.physics.turnPhase === 'PULLING') {
        if (this.aiTargetBlockId !== null) {
          this.aiAbortedBlockIds.add(this.aiTargetBlockId);
        }
        this.physics.endGrab(false);
        this.resetAiController();
      }
      return;
    }

    if (this.aiStage === 'placing') {
      if (this.physics.turnPhase !== 'EXTRACTED') {
        this.resetAiController();
        return;
      }
      this.aiTimerSec += dt;
      const hoverDur = this.aiFastCadence ? 0.09 : 0.92;
      const releaseTime = this.aiFastCadence ? 0.13 : 1.42;
      const u = Math.min(1.0, this.aiTimerSec / hoverDur);
      const smoothU = u * u * (3.0 - 2.0 * u);
      const hx =
        this.aiHoverCurrentXZ[0] + (this.aiHoverTargetXZ[0] - this.aiHoverCurrentXZ[0]) * smoothU;
      const hz =
        this.aiHoverCurrentXZ[1] + (this.aiHoverTargetXZ[1] - this.aiHoverCurrentXZ[1]) * smoothU;
      const topY = this.physics.topSupportFrame.layerCenter[1] + 1.0;
      this.physics.updateCarriedHover([hx, topY, hz], [0, -1, 0]);

      if (Math.abs(this.aiTargetYawOffsetRad) > 1e-4 && u >= 0.85) {
        const deltaYaw = this.aiTargetYawOffsetRad - this.physics.carriedYawOffsetRad;
        if (Math.abs(deltaYaw) > 1e-3) {
          this.physics.rotateCarriedBlock(deltaYaw);
        }
      }

      if (this.aiTimerSec >= releaseTime && this.physics.isExtractionLiftReady) {
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
      this.aiVsAiEnabled = false;
      this.aiFastCadence = false;
    }
    if (players !== undefined && players !== 2) {
      this.aiOpponentEnabled = false;
      this.aiVsAiEnabled = false;
      this.aiFastCadence = false;
    }
    if (this.renderer) {
      this.renderer.lightingOverride = null;
      this.renderer.referenceCardLuminance = 0;
      this.renderer.benchHideBlocks = false;
    }
    this.interactionMode = 'none';
    this.activePointerId = null;
    this.activeTouches.clear();
    this.primaryTouchDownMs = -Infinity;
    this.lastPinchDist = 0;
    this.lastPinchMidpoint = [0, 0];
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

  /**
   * Resolves a 1-finger / pointer ray against the top placement zone without requiring prior hover.
   * Combines direct OBB picking against open top guide slots and the floating carried prism with
   * grazing-safe plane projection so 1-finger placement works reliably across mobile, iPad, and desktop
   * even before camera elevation smoothing finishes.
   */
  private resolveTopPlacementRay(
    rayOrigin: Vec3,
    rayDir: Vec3,
    maxRadiusDm: number = 1.18
  ): { hitPoint: Vec3; synthOrigin: Vec3; synthDir: Vec3 } | null {
    const frame = this.physics.topSupportFrame;
    const lc = frame.layerCenter;
    const n = frame.upNormal;

    const makeVerticalRay = (pt: Vec3) => ({
      hitPoint: pt,
      synthOrigin: [pt[0] + n[0], pt[1] + n[1], pt[2] + n[2]] as Vec3,
      synthDir: [-n[0], -n[1], -n[2]] as Vec3,
    });

    const denom = rayDir[0] * n[0] + rayDir[1] * n[1] + rayDir[2] * n[2];

    // 1. For all downward-pointing rays (>= ~4.6 deg onto top support plane, including 10 deg grazing),
    // intersect the exact top support plane first so outer-slot front-face OBBs never introduce overhang parallax.
    if (denom <= -0.08) {
      const ptTop = this.intersectTopSupportPlane(rayOrigin, rayDir);
      if (ptTop && Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < maxRadiusDm) {
        return makeVerticalRay([ptTop[0], lc[1], ptTop[2]]);
      }
    }

    // 2. For upward/horizon rays (e.g. tall 24L+ tower viewed from below), pick open top slot OBBs and snap to slot center
    const slotHalfExt: Vec3 = [BLOCK_HALF_X * 1.12, BLOCK_HALF_Y * 1.65, BLOCK_HALF_Z * 1.24];
    let bestSlotT = Infinity;
    let bestSlotPt: Vec3 | null = null;
    for (let i = 0; i < this.physics.openTopSlots.length; i++) {
      const sl = this.physics.openTopSlots[i];
      const hit = intersectRayOBB(rayOrigin, rayDir, sl.position, sl.rotation, slotHalfExt);
      if (hit && hit.t < bestSlotT) {
        bestSlotT = hit.t;
        bestSlotPt = [sl.position[0], lc[1], sl.position[2]];
      }
    }
    if (bestSlotPt && Math.hypot(bestSlotPt[0] - lc[0], bestSlotPt[2] - lc[2]) < maxRadiusDm) {
      return makeVerticalRay(bestSlotPt);
    }

    // 3. Direct OBB hit on the floating carried prism itself -> use its clamped landing pose
    const carriedHit = intersectRayOBB(
      rayOrigin,
      rayDir,
      this.physics.carriedTargetPos,
      this.physics.carriedTargetQuat,
      slotHalfExt
    );
    if (carriedHit) {
      const pt: Vec3 = [this.physics.carriedLandingPos[0], lc[1], this.physics.carriedLandingPos[2]];
      if (Math.hypot(pt[0] - lc[0], pt[2] - lc[2]) < maxRadiusDm) {
        return makeVerticalRay(pt);
      }
    }

    // 4. Horizon / upward drag-plane projection fallback
    const ptProj = this.projectBlockDragPlane(rayOrigin, rayDir, lc);
    if (ptProj && Math.hypot(ptProj[0] - lc[0], ptProj[2] - lc[2]) < maxRadiusDm) {
      return makeVerticalRay([ptProj[0], lc[1], ptProj[2]]);
    }

    return null;
  }

  public getInteractionMode(): string {
    return this.interactionMode;
  }

  public getActiveTouchCount(): number {
    return this.activeTouches.size;
  }

  private cancelActiveSinglePointerForPinch(): void {
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
    if (this.physics) {
      this.physics.hoveredBlockId = null;
    }
    this.camera.isUserOrbiting = false;
    this.activePointerId = null;
    this.pendingTapHit = null;
  }

  public handlePointerDown(
    clientX: number,
    clientY: number,
    pointerId: number = 1,
    button: number = 0,
    altKey: boolean = false,
    overrideRect?: { left: number; top: number; width: number; height: number },
    pointerType: 'mouse' | 'touch' | 'pen' = 'mouse',
    timestampMs?: number
  ): void {
    this.audio.unlock();
    const nowMs =
      timestampMs ?? (typeof performance !== 'undefined' ? performance.now() : Date.now());

    if (pointerType === 'touch') {
      if (this.activeTouches.size === 0) {
        this.activeTouches.set(pointerId, [clientX, clientY]);
        this.primaryTouchDownMs = nowMs;
      } else {
        const elapsedSincePrimaryMs = nowMs - this.primaryTouchDownMs;
        // Protect established (> 80 ms) deliberate block pulls from accidental secondary bezel/palm touches
        if (
          this.interactionMode === 'block' &&
          elapsedSincePrimaryMs > GameEngine.MULTI_TOUCH_GRAB_CANCEL_WINDOW_MS
        ) {
          return;
        }

        this.activeTouches.set(pointerId, [clientX, clientY]);
        if (this.activeTouches.size >= 2) {
          this.cancelActiveSinglePointerForPinch();
          const pts = Array.from(this.activeTouches.values());
          this.lastPinchDist = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
          this.lastPinchMidpoint = [
            (pts[0][0] + pts[1][0]) * 0.5,
            (pts[0][1] + pts[1][1]) * 0.5,
          ];
          this.interactionMode = 'pinch_pan';
          this.updateCursorStyle();
          this.emitTelemetry(true);
          return;
        }
      }
    }

    if (this.interactionMode === 'pinch_pan' || this.activePointerId !== null) return;

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
      const resolvedTop = this.resolveTopPlacementRay(origin, dir, 1.18);
      if (resolvedTop) {
        if (this.mpRole === 'guest') {
          this.sendGuestRemoteCommand({
            kind: 'nudge_placement',
            rayOrigin: resolvedTop.synthOrigin,
            rayDir: resolvedTop.synthDir,
          });
        } else {
          this.physics.updateCarriedHover(resolvedTop.synthOrigin, resolvedTop.synthDir);
        }
        this.interactionMode = 'place_top';
      } else {
        this.interactionMode = 'orbit';
        this.camera.isUserOrbiting = true;
      }
      this.updateCursorStyle();
      return;
    }

    // Defer SELECT_BLOCK -> PULLING until >= 5.0 px drag so clicks/taps tap-test blocks without locking in a turn!
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

  private isRayNearTopPlacementZone(
    origin: Vec3,
    dir: Vec3,
    maxRadiusDm: number = 0.80
  ): boolean {
    const ptTop = this.intersectTopSupportPlane(origin, dir);
    if (!ptTop) return false;
    const lc = this.physics.topSupportFrame.layerCenter;
    return Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < maxRadiusDm;
  }

  public handlePointerMove(
    clientX: number,
    clientY: number,
    pointerId: number = 1,
    overrideRect?: { left: number; top: number; width: number; height: number },
    pointerType: 'mouse' | 'touch' | 'pen' = 'mouse',
    timestampMs?: number
  ): void {
    const rect = this.getViewportRect(overrideRect);
    const now =
      timestampMs ?? (typeof performance !== 'undefined' ? performance.now() : Date.now());

    if (pointerType === 'touch') {
      if (this.activeTouches.has(pointerId)) {
        this.activeTouches.set(pointerId, [clientX, clientY]);
      }
      if (this.interactionMode === 'pinch_pan') {
        if (this.activeTouches.size >= 2) {
          const pts = Array.from(this.activeTouches.values());
          const midX = (pts[0][0] + pts[1][0]) * 0.5;
          const midY = (pts[0][1] + pts[1][1]) * 0.5;
          const dist = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);

          const dMidX = midX - this.lastPinchMidpoint[0];
          const dMidY = midY - this.lastPinchMidpoint[1];
          if (Math.abs(dMidX) > 0.01 || Math.abs(dMidY) > 0.01) {
            this.camera.panBy(dMidX, dMidY, rect.height);
          }

          if (this.lastPinchDist > 8 && dist > 8) {
            const scaleDelta = Math.max(
              -0.22,
              Math.min(0.22, this.lastPinchDist / dist - 1.0)
            );
            if (Math.abs(scaleDelta) > 1e-4) {
              this.camera.zoomBy(scaleDelta);
            }
          }

          this.lastPinchDist = dist;
          this.lastPinchMidpoint = [midX, midY];
        }
        return;
      }
      // Touch input has no passive hover state when no primary touch is active
      if (this.activePointerId === null || pointerId !== this.activePointerId) {
        return;
      }
    }

    const { origin, dir } = this.camera.screenToRay(clientX, clientY, rect);

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
          if (
            now - this.mpLastGuestCmdSentMs >= 15 &&
            this.isRayNearTopPlacementZone(origin, dir, 0.80)
          ) {
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

      if (
        this.physics.turnPhase === 'EXTRACTED' &&
        this.isRayNearTopPlacementZone(origin, dir, 0.80)
      ) {
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
      this.handlePointerCancel(pointerId, pointerType);
      return;
    }

    if (this.interactionMode === 'place_top' && this.physics.turnPhase === 'EXTRACTED') {
      const resolvedTop = this.resolveTopPlacementRay(origin, dir, 1.45);
      const rOrig = resolvedTop ? resolvedTop.synthOrigin : origin;
      const rDir = resolvedTop ? resolvedTop.synthDir : dir;
      if (this.mpRole === 'guest') {
        if (now - this.mpLastGuestCmdSentMs >= 15) {
          this.sendGuestRemoteCommand(
            { kind: 'nudge_placement', rayOrigin: rOrig, rayDir: rDir },
            now
          );
        }
      } else {
        this.physics.updateCarriedHover(rOrig, rDir);
      }
      this.updateCursorStyle();
      return;
    }

    if (this.interactionMode === 'block_pending' && this.pointerMovedDist >= 5.0 && this.pendingTapHit) {
      if (this.mpRole === 'guest') {
        const localPt: Vec3 = [...this.pendingTapHit.hit.localPoint];
        const worldPt: Vec3 = [...this.pendingTapHit.hit.worldPoint];
        this.mpActiveGuestPullGrab = {
          blockId: this.pendingTapHit.blockId,
          localPoint: localPt,
          worldPoint: worldPt,
        };
        this.sendGuestRemoteCommand(
          {
            kind: 'begin_pull',
            blockId: this.pendingTapHit.blockId,
            localPoint: localPt,
            worldPoint: worldPt,
          },
          now
        );
        this.guestWorld?.setOptimisticGuestPullAnchor(worldPt);
        this.interactionMode = 'block';
        this.emitTelemetry(true);
      } else if (
        this.mpRole === 'host' &&
        (this.physics.currentPlayer !== 0 ||
          this.physics.turnPhase !== 'SELECT_BLOCK' ||
          this.physics.isPointerDraggingBlock ||
          this.physics.activeBlockId !== null)
      ) {
        this.interactionMode = 'none';
        this.pendingTapHit = null;
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
        const resolvedTop = this.resolveTopPlacementRay(origin, dir, 1.35);
        if (this.physics.isExtractionLiftReady && resolvedTop) {
          if (this.mpRole === 'guest') {
            if (this.mpStatus.peerConnected && now - this.mpLastGuestCmdSentMs >= 15) {
              this.sendGuestRemoteCommand(
                {
                  kind: 'nudge_placement',
                  rayOrigin: resolvedTop.synthOrigin,
                  rayDir: resolvedTop.synthDir,
                },
                now
              );
            }
          } else {
            this.physics.updateCarriedHover(resolvedTop.synthOrigin, resolvedTop.synthDir);
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
            if (this.mpStatus.peerConnected && now - this.mpLastGuestCmdSentMs >= 15) {
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
    overrideRect?: { left: number; top: number; width: number; height: number },
    pointerType: 'mouse' | 'touch' | 'pen' = 'mouse',
    timestampMs?: number
  ): void {
    void timestampMs;
    if (pointerType === 'touch') {
      this.activeTouches.delete(pointerId);
      if (this.interactionMode === 'pinch_pan') {
        if (this.activeTouches.size < 2) {
          this.lastPinchDist = 0;
        }
        if (this.activeTouches.size === 0) {
          this.interactionMode = 'none';
          this.primaryTouchDownMs = -Infinity;
          this.updateCursorStyle();
        }
        return;
      }
      if (this.activeTouches.size === 0) {
        this.primaryTouchDownMs = -Infinity;
      }
    }

    if (this.activePointerId === null || pointerId !== this.activePointerId) return;

    const rect = this.getViewportRect(overrideRect);
    const { origin, dir } = this.camera.screenToRay(clientX, clientY, rect);

    if (this.interactionMode === 'place_top') {
      const resolvedTop = this.resolveTopPlacementRay(origin, dir, 1.05);
      if (
        (this.pointerMovedDist < 18.0 || resolvedTop !== null) &&
        this.canLocalPlayerManipulateBlocks()
      ) {
        const rOrig = resolvedTop ? resolvedTop.synthOrigin : origin;
        const rDir = resolvedTop ? resolvedTop.synthDir : dir;
        if (this.mpRole === 'guest') {
          this.sendGuestRemoteCommand({
            kind: 'nudge_placement',
            rayOrigin: rOrig,
            rayDir: rDir,
          });
          this.sendGuestRemoteCommand({ kind: 'confirm_placement' });
        } else {
          if (resolvedTop) {
            this.physics.updateCarriedHover(rOrig, rDir);
          }
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
        this.mpActiveGuestPullGrab = null;
        this.guestWorld?.setOptimisticGuestPullAnchor(null);
        if (this.mpPendingPullTarget) {
          const finalTarget = this.mpPendingPullTarget;
          this.mpPendingPullTarget = null;
          this.sendGuestRemoteCommand({ kind: 'update_pull', worldTarget: finalTarget });
        }
        const resolvedTop =
          this.physics.turnPhase === 'EXTRACTED'
            ? this.resolveTopPlacementRay(origin, dir, 0.82)
            : null;
        const overTop =
          this.physics.turnPhase === 'EXTRACTED' &&
          this.physics.isExtractionLiftReady &&
          resolvedTop !== null;
        this.sendGuestRemoteCommand({ kind: 'end_pull', dropIfPoisedOverTower: overTop });
      } else if (this.mpRole === 'host' && this.physics.currentPlayer !== 0) {
        // Host lost turn ownership (e.g. rematch or turn handoff); never terminate Guest's pull
      } else if (this.physics.turnPhase === 'EXTRACTED') {
        const resolvedTop = this.resolveTopPlacementRay(origin, dir, 0.82);
        const overTop = this.physics.isExtractionLiftReady && resolvedTop !== null;
        if (overTop && resolvedTop) {
          this.physics.updateCarriedHover(resolvedTop.synthOrigin, resolvedTop.synthDir);
        }
        this.physics.endGrab(overTop);
        this.mpForceImmediateSnapshot = true;
      } else {
        this.physics.endGrab(false);
        this.mpForceImmediateSnapshot = true;
      }
    } else if (
      this.interactionMode === 'orbit' &&
      this.pointerMovedDist < 5.0 &&
      !this.mpRole &&
      this.physics.gameMode === 'sandbox' &&
      typeof this.physics.raycastScene === 'function' &&
      typeof this.physics.nudgeShard === 'function'
    ) {
      const shardHit = this.physics.raycastScene(origin, dir, {
        includeBlocks: false,
        includeShards: true,
        includeTable: false,
      });
      if (shardHit?.kind === 'shard' && shardHit.shardId !== undefined) {
        const nudged = this.physics.nudgeShard(shardHit.shardId, shardHit.worldPointDm, dir);
        if (nudged) {
          const shd = this.physics.shards.find((s) => s.id === shardHit.shardId);
          const audioHooks = this.audio as unknown as OptionalAcousticsHooks;
          audioHooks.playShardImpacts?.(
            [
              {
                pos: shardHit.worldPointDm,
                speed: 0.42,
                massKg: shd?.massKg ?? 0.0025,
                surface: 'wood',
                slotIndex: shd?.slotIndex ?? 0,
              },
            ],
            this.camera.position,
            this.camera.getRightVector()
          );
        }
      }
    }

    this.camera.isUserOrbiting = false;
    this.activePointerId = null;
    this.interactionMode = 'none';
    this.pendingTapHit = null;
    this.updateCursorStyle();
    this.emitTelemetry(true);
  }

  public handlePointerCancel(
    pointerId?: number,
    pointerType: 'mouse' | 'touch' | 'pen' = 'mouse'
  ): void {
    if (pointerId === undefined) {
      this.activeTouches.clear();
      this.primaryTouchDownMs = -Infinity;
      this.lastPinchDist = 0;
      if (this.interactionMode === 'pinch_pan') {
        this.interactionMode = 'none';
        this.updateCursorStyle();
        return;
      }
    } else if (pointerType === 'touch') {
      this.activeTouches.delete(pointerId);
      if (this.interactionMode === 'pinch_pan') {
        if (this.activeTouches.size < 2) {
          this.lastPinchDist = 0;
        }
        if (this.activeTouches.size === 0) {
          this.interactionMode = 'none';
          this.primaryTouchDownMs = -Infinity;
          this.updateCursorStyle();
        }
        return;
      }
      if (this.activeTouches.size === 0) {
        this.primaryTouchDownMs = -Infinity;
      }
    }

    if (this.activePointerId === null) return;
    if (pointerId !== undefined && pointerId !== this.activePointerId) return;

    if (this.interactionMode === 'block') {
      if (this.mpRole === 'guest') {
        this.mpActiveGuestPullGrab = null;
        this.mpPendingPullTarget = null;
        this.guestWorld?.setOptimisticGuestPullAnchor(null);
        this.sendGuestRemoteCommand({ kind: 'end_pull', dropIfPoisedOverTower: false });
      } else if (this.mpRole !== 'host' || this.physics.currentPlayer === 0) {
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

  private syncCarriedHoverToCurrentLanding(): void {
    const physicsAny = this.physics as unknown as { hasCustomCarriedHover?: boolean };
    if (physicsAny.hasCustomCarriedHover) return;
    const land = this.physics.carriedLandingPos;
    const n = this.physics.topSupportFrame.upNormal;
    this.physics.updateCarriedHover(
      [land[0] + n[0], land[1] + n[1], land[2] + n[2]],
      [-n[0], -n[1], -n[2]]
    );
  }

  public handleRotateCarried(deltaYawRad: number): void {
    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;
    this.audio.unlock();
    if (this.mpRole === 'guest') {
      this.sendGuestRemoteCommand({ kind: 'rotate_placement', deltaYawRad });
      return;
    }
    this.syncCarriedHoverToCurrentLanding();
    this.physics.rotateCarriedBlock(deltaYawRad);
    this.mpForceImmediateSnapshot = true;
    this.emitTelemetry(true);
  }

  public handleResetCarriedRotation(): void {
    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;
    this.audio.unlock();
    if (this.mpRole === 'guest') {
      this.sendGuestRemoteCommand({ kind: 'reset_rotation' });
      return;
    }
    this.syncCarriedHoverToCurrentLanding();
    this.physics.resetCarriedBlockRotation();
    this.mpForceImmediateSnapshot = true;
    this.emitTelemetry(true);
  }

  public handleReleaseOntoTop(): void {
    if (!this.canLocalPlayerManipulateBlocks() || this.physics.turnPhase !== 'EXTRACTED') return;
    this.audio.unlock();
    if (this.mpRole === 'guest') {
      this.sendGuestRemoteCommand({ kind: 'confirm_placement' });
      return;
    }
    this.physics.releaseExtractedBlockOntoTop();
    this.mpForceImmediateSnapshot = true;
    this.emitTelemetry(true);
  }

  private onFirstGestureUnlock = (): void => {
    this.audio.unlock();
  };

  private onPointerDown = (e: PointerEvent): void => {
    if ((e as unknown as { _vitrailHandled?: boolean })._vitrailHandled) return;
    (e as unknown as { _vitrailHandled?: boolean })._vitrailHandled = true;
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // Synthetic pointer events may not have an active pointerId
    }
    const pType = (e.pointerType as 'mouse' | 'touch' | 'pen') || 'mouse';
    this.handlePointerDown(
      e.clientX,
      e.clientY,
      e.pointerId,
      e.button,
      e.altKey,
      undefined,
      pType,
      e.timeStamp
    );
  };

  private onPointerMove = (e: PointerEvent): void => {
    const pType = (e.pointerType as 'mouse' | 'touch' | 'pen') || 'mouse';
    if (
      this.activePointerId === null &&
      this.activeTouches.size === 0 &&
      e.target !== this.canvas
    ) {
      return;
    }
    this.handlePointerMove(
      e.clientX,
      e.clientY,
      e.pointerId,
      undefined,
      pType,
      e.timeStamp
    );
  };

  private onPointerUp = (e: PointerEvent): void => {
    const pType = (e.pointerType as 'mouse' | 'touch' | 'pen') || 'mouse';
    this.handlePointerUp(
      e.clientX,
      e.clientY,
      e.pointerId,
      undefined,
      pType,
      e.timeStamp
    );
  };

  private onPointerCancel = (e: PointerEvent): void => {
    const pType = (e.pointerType as 'mouse' | 'touch' | 'pen') || 'mouse';
    this.handlePointerCancel(e.pointerId, pType);
  };

  private onWindowBlur = (): void => {
    this.activeTouches.clear();
    this.primaryTouchDownMs = -Infinity;
    this.lastPinchDist = 0;
    this.lastPinchMidpoint = [0, 0];
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
      if (this.activePointerId !== null || this.activeTouches.size > 0) {
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

    // iOS/WebKit CoreAudio first-gesture unlock across both canvas and HUD controls
    window.addEventListener('touchstart', this.onFirstGestureUnlock, {
      capture: true,
      passive: true,
    });
    window.addEventListener('touchend', this.onFirstGestureUnlock, {
      capture: true,
      passive: true,
    });
    window.addEventListener('pointerdown', this.onFirstGestureUnlock, {
      capture: true,
      passive: true,
    });
    window.addEventListener('pointerup', this.onFirstGestureUnlock, {
      capture: true,
      passive: true,
    });
    window.addEventListener('keydown', this.onFirstGestureUnlock, {
      capture: true,
      passive: true,
    });

    this.resizeObserver = new ResizeObserver((entries) => {
      if (this.disposed || !this.rendererInitialized) return;
      if (typeof document !== 'undefined' && document.hidden) return;
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      if (now < this.gpuYieldUntilMs) return;
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
      window.removeEventListener('touchstart', this.onFirstGestureUnlock, true);
      window.removeEventListener('touchend', this.onFirstGestureUnlock, true);
      window.removeEventListener('pointerdown', this.onFirstGestureUnlock, true);
      window.removeEventListener('pointerup', this.onFirstGestureUnlock, true);
      window.removeEventListener('keydown', this.onFirstGestureUnlock, true);
    }
    this.resizeObserver?.disconnect();
    this.closeGpuYieldChannel();

    this.shutdownMultiplayerSessionOnly();
    if (this.guestWorld) {
      this.guestWorld.dispose();
      this.guestWorld = null;
    }
    if (this.authoritativePhysics && this.authoritativePhysics !== this.physics) {
      this.authoritativePhysics.dispose();
      this.authoritativePhysics = null;
    }
    if (this.rendererInitialized) {
      this.renderer?.dispose();
    }
    this.physics?.dispose();
    this.audio?.dispose();
  }
}

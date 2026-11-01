import { GlassAudioEngine } from '../audio/GlassAudioEngine';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  LIGHTING_PRESETS,
} from '../physics/constants';
import {
  ensureRapierInitialized,
  GameMode,
  PhysicsWorld,
  TowerTelemetry,
} from '../physics/PhysicsWorld';
import { WebGPURenderer } from '../renderer/WebGPURenderer';
import { intersectRayOBB, RayOBBHit, Vec3 } from '../math/vecMat';
import { OrbitCamera } from './OrbitCamera';

export type TelemetryListener = (telemetry: TowerTelemetry) => void;

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
  private pointerDownPos: [number, number] = [0, 0];
  private lastPointerPos: [number, number] = [0, 0];
  private pointerMovedDist: number = 0;
  private interactionMode: 'none' | 'orbit' | 'block_pending' | 'block' | 'place_top' = 'none';
  private pendingTapHit: { blockId: number; hit: RayOBBHit } | null = null;
  private grabAnchorOrigin: Vec3 = [0, 0, 0];

  public benchPaused: boolean = false;
  public benchCameraLocked: boolean = false;
  private resizeObserver: ResizeObserver | null = null;

  constructor(canvas?: HTMLCanvasElement, onTelemetry?: TelemetryListener) {
    this.canvas = (canvas ?? ({ style: {} } as HTMLCanvasElement));
    this.listener = onTelemetry ?? null;
    this.audio = new GlassAudioEngine();
    this.camera = new OrbitCamera();
  }

  /**
   * Initializes Rapier + PhysicsWorld + OrbitCamera without WebGPU for fast headless playability tests (`T4`).
   */
  public async initHeadless(): Promise<void> {
    await ensureRapierInitialized();
    this.physics = new PhysicsWorld(this.audio);
    this.camera.update(0.016, 16 / 9);
    this.physics.setListenerPose(this.camera.position, this.camera.getRightVector());
  }

  public async init(): Promise<void> {
    await ensureRapierInitialized();
    if (this.disposed) return;

    this.renderer = new WebGPURenderer(this.canvas);
    await this.renderer.init();

    if (this.disposed) {
      this.renderer.dispose();
      return;
    }

    this.physics = new PhysicsWorld(this.audio);

    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const benchScene = params.get('bench');
      if (benchScene) {
        const presetId = params.get('preset') ?? undefined;
        const darkParam = params.get('darkroom');
        const darkScale = darkParam !== null ? Number(darkParam) : undefined;
        this.configureBenchScene(benchScene, presetId, darkScale);
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

  private tick = (nowMs: number): void => {
    if (!this.running) return;
    const rawDt = (nowMs - this.lastFrameMs) / 1000;
    this.lastFrameMs = nowMs;
    const dt = Math.min(0.05, Math.max(0.001, rawDt));
    this.elapsedSec += dt;

    const aspect =
      (this.canvas.clientWidth || window.innerWidth) /
      Math.max(1, this.canvas.clientHeight || window.innerHeight);
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
      this.physics.step(dt);
    }

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
    this.listener(this.physics.getTelemetry());
  }

  public resetGame(mode?: GameMode, players?: number): void {
    if (!this.physics) return;
    this.benchPaused = false;
    this.benchCameraLocked = false;
    if (this.renderer) {
      this.renderer.lightingOverride = null;
      this.renderer.referenceCardLuminance = 0;
      this.renderer.benchHideBlocks = false;
    }
    this.interactionMode = 'none';
    this.activePointerId = null;
    this.pendingTapHit = null;
    this.physics.rebuildTower(mode, players);
    this.camera.resetView(27.0);
    this.updateCursorStyle();
    this.emitTelemetry(true);
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
        if (f > 115 && this.physics.shards.length >= 320) {
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
   * Grazing-safe axial drag plane projection:
   * Below 25 deg camera pitch, tilts the drag plane normal toward the camera so the ray-plane angle
   * never drops below 25 deg, eliminating the 5.2x horizon depth blowup while staying 0-jump at grab start.
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

    const denom = rayDir[0] * nx + rayDir[1] * ny + rayDir[2] * nz;
    if (Math.abs(denom) < 1e-4) return null;
    const t =
      ((anchorPoint[0] - rayOrigin[0]) * nx +
        (anchorPoint[1] - rayOrigin[1]) * ny +
        (anchorPoint[2] - rayOrigin[2]) * nz) /
      denom;
    if (t <= 0) return null;

    return [
      rayOrigin[0] + rayDir[0] * t,
      anchorPoint[1],
      rayOrigin[2] + rayDir[2] * t,
    ];
  }

  private updateCursorStyle(): void {
    if (!this.canvas?.style) return;
    if (this.interactionMode === 'orbit') {
      this.canvas.style.cursor = 'move';
      return;
    }
    if (this.physics.turnPhase === 'PULLING' && this.physics.isPointerDraggingBlock) {
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

    if (button === 2 || altKey) {
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
        this.physics.updateCarriedHover(origin, dir);
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
          this.physics.gameMode === 'sandbox')
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

    if (this.activePointerId === null || pointerId !== this.activePointerId) {
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

    if (this.interactionMode === 'place_top' && this.physics.turnPhase === 'EXTRACTED') {
      this.physics.updateCarriedHover(origin, dir);
      this.updateCursorStyle();
      return;
    }

    if (this.interactionMode === 'block_pending' && this.pointerMovedDist >= 5.0 && this.pendingTapHit) {
      const started = this.physics.startGrabBlock(
        this.pendingTapHit.blockId,
        this.pendingTapHit.hit.localPoint,
        this.pendingTapHit.hit.worldPoint
      );
      if (started) {
        this.interactionMode = 'block';
        this.emitTelemetry(true);
      } else {
        this.interactionMode = 'none';
      }
    }

    if (this.interactionMode === 'block') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        this.physics.updateCarriedHover(origin, dir);
      } else {
        const planePt = this.projectBlockDragPlane(origin, dir, this.grabAnchorOrigin);
        if (planePt) {
          this.physics.updateGrabTarget(planePt);
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
        this.physics.updateCarriedHover(origin, dir);
        this.physics.releaseExtractedBlockOntoTop();
      }
    } else if (this.interactionMode === 'block_pending') {
      // Pointer moved < 5.0 px: acoustic & tactile tap-test without leaving SELECT_BLOCK!
      if (this.pendingTapHit) {
        this.physics.setListenerPose(this.camera.position, this.camera.getRightVector());
        this.physics.tapNudgeBlock(
          this.pendingTapHit.blockId,
          this.pendingTapHit.hit.localNormal,
          this.pendingTapHit.hit.localPoint
        );
      }
    } else if (this.interactionMode === 'block') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        const lc = this.physics.topSupportFrame.layerCenter;
        const ptTop = this.intersectTopSupportPlane(origin, dir);
        const overTop =
          this.physics.isExtractionLiftReady &&
          ptTop !== null &&
          Math.hypot(ptTop[0] - lc[0], ptTop[2] - lc[2]) < 0.78;
        this.physics.endGrab(overTop);
      } else {
        this.physics.endGrab(false);
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
      this.physics.endGrab(false);
    }

    this.camera.isUserOrbiting = false;
    this.activePointerId = null;
    this.interactionMode = 'none';
    this.pendingTapHit = null;
    this.updateCursorStyle();
    this.emitTelemetry(true);
  }

  public handleRotateCarried(deltaYawRad: number): void {
    if (this.physics.turnPhase !== 'EXTRACTED') return;
    this.physics.rotateCarriedBlock(deltaYawRad);
    this.emitTelemetry(true);
  }

  public handleResetCarriedRotation(): void {
    if (this.physics.turnPhase !== 'EXTRACTED') return;
    this.physics.resetCarriedBlockRotation();
    this.emitTelemetry(true);
  }

  public handleReleaseOntoTop(): void {
    if (this.physics.turnPhase !== 'EXTRACTED') return;
    this.physics.releaseExtractedBlockOntoTop();
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
    this.handlePointerDown(e.clientX, e.clientY, e.pointerId, e.button, e.altKey);
  };

  private onPointerMove = (e: PointerEvent): void => {
    this.handlePointerMove(e.clientX, e.clientY, e.pointerId);
  };

  private onPointerUp = (e: PointerEvent): void => {
    this.handlePointerUp(e.clientX, e.clientY, e.pointerId);
  };

  private onPointerCancel = (e: PointerEvent): void => {
    this.handlePointerCancel(e.pointerId);
  };

  private onWindowBlur = (): void => {
    this.handlePointerCancel();
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    if (this.physics.turnPhase === 'EXTRACTED' && !e.altKey && !e.ctrlKey && !e.metaKey) {
      const yawStep = Math.max(-0.16, Math.min(0.16, e.deltaY * 0.0009));
      this.handleRotateCarried(yawStep);
      return;
    }
    const zoomDelta = Math.max(-0.22, Math.min(0.22, e.deltaY * 0.0012));
    this.camera.zoomBy(zoomDelta);
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (this.physics?.turnPhase !== 'EXTRACTED') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

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
      this.physics.resetCarriedBlockRotation();
      this.emitTelemetry(true);
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

    this.renderer?.dispose();
    this.physics?.dispose();
    this.audio?.dispose();
  }
}

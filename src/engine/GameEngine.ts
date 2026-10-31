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
  private pointerDownTimeMs: number = 0;
  private pointerMovedDist: number = 0;
  private interactionMode: 'none' | 'orbit' | 'block' | 'place_top' = 'none';
  private pendingTapHit: { blockId: number; hit: RayOBBHit } | null = null;

  private resizeObserver: ResizeObserver | null = null;

  constructor(canvas: HTMLCanvasElement, onTelemetry?: TelemetryListener) {
    this.canvas = canvas;
    this.listener = onTelemetry ?? null;
    this.audio = new GlassAudioEngine();
    this.camera = new OrbitCamera();
  }

  public async init(): Promise<void> {
    await ensureRapierInitialized();
    if (this.disposed) return;

    this.physics = new PhysicsWorld(this.audio);
    this.renderer = new WebGPURenderer(this.canvas);
    await this.renderer.init();

    if (this.disposed) {
      this.renderer.dispose();
      this.physics.dispose();
      return;
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

    this.physics.step(dt);

    // Smooth camera focus adjustment: frame top of tower when placing an extracted block!
    const topSlotY =
      this.physics.openTopSlots.length > 0
        ? this.physics.openTopSlots[0].position[1]
        : (this.physics.currentLayers * 0.15);

    if (this.physics.turnPhase === 'EXTRACTED') {
      this.camera.focusTopForPlacement(topSlotY);
    } else if (this.physics.turnPhase !== 'COLLAPSED') {
      this.camera.setFocusHeight(Math.max(1.32, topSlotY * 0.52));
    }

    const aspect =
      (this.canvas.clientWidth || window.innerWidth) /
      Math.max(1, this.canvas.clientHeight || window.innerHeight);
    this.camera.update(dt, aspect);
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
      // Grazing horizontal view fallback: project along camera distance to tower axis
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

  private updateCursorStyle(): void {
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

  private onPointerDown = (e: PointerEvent): void => {
    if ((e as unknown as { _vitrailHandled?: boolean })._vitrailHandled) return;
    (e as unknown as { _vitrailHandled?: boolean })._vitrailHandled = true;

    this.audio.unlock();
    if (this.activePointerId !== null) return;

    this.activePointerId = e.pointerId;
    this.pointerDownPos = [e.clientX, e.clientY];
    this.lastPointerPos = [e.clientX, e.clientY];
    this.pointerDownTimeMs = performance.now();
    this.pointerMovedDist = 0;

    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // Synthetic pointer events in headless tests may not have an active pointerId
    }

    const rect = this.canvas.getBoundingClientRect();
    const { origin, dir } = this.camera.screenToRay(e.clientX, e.clientY, rect);

    // Right-click or Alt/Shift drag always orbits camera
    if (e.button === 2 || e.altKey) {
      this.interactionMode = 'orbit';
      this.camera.isUserOrbiting = true;
      this.updateCursorStyle();
      return;
    }

    // If holding an extracted block (`EXTRACTED`), clicking near the top slots places it;
    // dragging far outside orbits camera so player can inspect tower lean before clicking!
    if (this.physics.turnPhase === 'EXTRACTED') {
      this.physics.updateCarriedHover(origin, dir);
      const topY =
        this.physics.openTopSlots.length > 0 ? this.physics.openTopSlots[0].position[1] : 2.7;
      const ptTop = this.intersectHorizontalPlane(origin, dir, topY);
      const nearTopTower = ptTop !== null && Math.hypot(ptTop[0], ptTop[2]) < 1.15;

      if (nearTopTower) {
        this.interactionMode = 'place_top';
      } else {
        this.interactionMode = 'orbit';
        this.camera.isUserOrbiting = true;
      }
      this.updateCursorStyle();
      return;
    }

    // Check raycast against stained-glass blocks
    const pick = this.raycastIntactBlocks(origin, dir);
    if (pick) {
      const blk = this.physics.blocks[pick.blockId];
      if (blk && blk.isLegalTarget && this.physics.turnPhase === 'SELECT_BLOCK') {
        this.pendingTapHit = pick;
        const started = this.physics.startGrabBlock(
          pick.blockId,
          pick.hit.localPoint,
          pick.hit.worldPoint
        );
        if (started) {
          this.interactionMode = 'block';
          this.updateCursorStyle();
          this.emitTelemetry(true);
          return;
        }
      } else if (blk && this.physics.gameMode === 'sandbox') {
        this.pendingTapHit = pick;
        this.physics.startGrabBlock(pick.blockId, pick.hit.localPoint, pick.hit.worldPoint);
        this.interactionMode = 'block';
        this.updateCursorStyle();
        return;
      }
    }

    // Clicked background or locked block -> Orbit camera!
    this.interactionMode = 'orbit';
    this.camera.isUserOrbiting = true;
    this.updateCursorStyle();
  };

  private onPointerMove = (e: PointerEvent): void => {
    const rect = this.canvas.getBoundingClientRect();
    const { origin, dir } = this.camera.screenToRay(e.clientX, e.clientY, rect);

    // Always update top-slot hover preview when a block is extracted!
    if (this.physics.turnPhase === 'EXTRACTED') {
      this.physics.updateCarriedHover(origin, dir);
    }

    if (this.activePointerId === null || e.pointerId !== this.activePointerId) {
      // Passive hover raycast
      if (this.physics.turnPhase === 'SELECT_BLOCK' || this.physics.gameMode === 'sandbox') {
        const pick = this.raycastIntactBlocks(origin, dir);
        this.physics.hoveredBlockId = pick ? pick.blockId : null;
      } else {
        this.physics.hoveredBlockId = null;
      }
      this.updateCursorStyle();
      return;
    }

    const dx = e.clientX - this.lastPointerPos[0];
    const dy = e.clientY - this.lastPointerPos[1];
    this.lastPointerPos = [e.clientX, e.clientY];
    this.pointerMovedDist = Math.hypot(
      e.clientX - this.pointerDownPos[0],
      e.clientY - this.pointerDownPos[1]
    );

    if (this.interactionMode === 'orbit') {
      this.camera.orbitBy(-dx * 0.0072, dy * 0.0058);
      return;
    }

    if (this.interactionMode === 'block') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        // Block just popped free mid-drag! Let player drag straight up onto a top slot!
        this.physics.updateCarriedHover(origin, dir);
      } else {
        const planePt = this.intersectHorizontalPlane(origin, dir, this.physics.grabPlaneY);
        if (planePt) {
          this.physics.updateGrabTarget(planePt);
        }
      }
      this.updateCursorStyle();
    }
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (this.activePointerId === null || e.pointerId !== this.activePointerId) return;

    const elapsedMs = performance.now() - this.pointerDownTimeMs;
    const wasQuickClick = elapsedMs < 210 && this.pointerMovedDist < 7.0;

    const rect = this.canvas.getBoundingClientRect();
    const { origin, dir } = this.camera.screenToRay(e.clientX, e.clientY, rect);

    if (this.interactionMode === 'place_top') {
      if (this.pointerMovedDist < 18.0) {
        this.physics.updateCarriedHover(origin, dir);
        this.physics.releaseExtractedBlockOntoTop();
      }
    } else if (this.interactionMode === 'block') {
      if (this.physics.turnPhase === 'EXTRACTED') {
        // Check if cursor ended right over the top of the tower -> drop seamlessly in one gesture!
        const topY =
          this.physics.openTopSlots.length > 0 ? this.physics.openTopSlots[0].position[1] : 2.7;
        const ptTop = this.intersectHorizontalPlane(origin, dir, topY);
        const overTop = ptTop !== null && Math.hypot(ptTop[0], ptTop[2]) < 0.72;
        this.physics.endGrab(overTop);
      } else {
        this.physics.endGrab(false);
        // If it was a crisp finger click (< 210 ms, < 7 px), perform a physical tap-test poke!
        if (wasQuickClick && this.pendingTapHit) {
          this.physics.tapNudgeBlock(
            this.pendingTapHit.blockId,
            this.pendingTapHit.hit.localNormal,
            this.pendingTapHit.hit.localPoint
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
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const zoomDelta = Math.max(-0.22, Math.min(0.22, e.deltaY * 0.0012));
    this.camera.zoomBy(zoomDelta);
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
    window.addEventListener('pointercancel', this.onPointerUp);
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
    cancelAnimationFrame(this.rafId);

    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('pointercancel', this.onPointerUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.canvas.removeEventListener('contextmenu', this.onContextMenu);
    window.removeEventListener('pagehide', this.onPageHide);
    this.resizeObserver?.disconnect();

    this.renderer?.dispose();
    this.physics?.dispose();
    this.audio?.dispose();
  }
}

import { createServer, type ViteDevServer } from 'vite';
import { describe, expect, it } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { GameEngine } from '../src/engine/GameEngine';
import { OrbitCamera } from '../src/engine/OrbitCamera';
import { BLOCK_HALF_X, BLOCK_HALF_Z } from '../src/physics/constants';
import { quatRotateVec3, type Vec3 } from '../src/math/vecMat';

interface MinimalChildProcess {
  exitCode: number | null;
  stderr: {
    on(event: 'data', cb: (chunk: Uint8Array) => void): void;
  } | null;
  once(event: 'exit', cb: (code: number | null) => void): void;
  kill(signal?: string): boolean;
}

interface MinimalChildProcessModule {
  spawn(
    cmd: string,
    args: string[],
    opts: { stdio: Array<'ignore' | 'pipe'> }
  ): MinimalChildProcess;
}

interface MinimalFsModule {
  readFileSync(filePath: string, encoding: 'utf8'): string;
  mkdtempSync(prefix: string): string;
  rmSync(target: string, opts: { recursive: boolean; force: boolean }): void;
}

interface MinimalHttpResponse {
  setEncoding(enc: 'utf8'): void;
  on(event: 'data', cb: (chunk: string) => void): void;
  on(event: 'end', cb: () => void): void;
}

interface MinimalHttpModule {
  get(
    opts: { hostname: string; port: number; path: string; agent: boolean },
    cb: (res: MinimalHttpResponse) => void
  ): { on(event: 'error', cb: (err: unknown) => void): void };
}

interface MinimalOsModule {
  tmpdir(): string;
}

interface MinimalPathModule {
  resolve(...parts: string[]): string;
  join(...parts: string[]): string;
}

function getNodeCwd(): string {
  const g = globalThis as unknown as { process?: { cwd(): string } };
  return g.process ? g.process.cwd() : '.';
}

function raycastIntactBlocksForTest(
  engine: GameEngine,
  origin: Vec3,
  dir: Vec3
): { blockId: number } | null {
  return (
    engine as unknown as {
      raycastIntactBlocks(o: Vec3, d: Vec3): { blockId: number } | null;
    }
  ).raycastIntactBlocks(origin, dir);
}

function getCameraTargetPose(cam: OrbitCamera): {
  targetAzimuth: number;
  targetElevation: number;
  targetDistance: number;
  targetPanOffset: Vec3;
} {
  return cam as unknown as {
    targetAzimuth: number;
    targetElevation: number;
    targetDistance: number;
    targetPanOffset: Vec3;
  };
}

const VIEWPORTS: ReadonlyArray<{
  name: string;
  width: number;
  height: number;
}> = [
  { name: 'iPad Pro Portrait (1024x1366)', width: 1024, height: 1366 },
  { name: 'iPad Pro Landscape (1366x1024)', width: 1366, height: 1024 },
  { name: 'Mobile Portrait (390x844)', width: 390, height: 844 },
  { name: 'Desktop Studio (1440x900)', width: 1440, height: 900 },
];

function worldToScreenPx(
  engine: GameEngine,
  worldPt: Vec3,
  vpWidth: number,
  vpHeight: number
): [number, number] {
  const m = engine.camera.viewProjMatrix;
  const x = worldPt[0];
  const y = worldPt[1];
  const z = worldPt[2];
  const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
  const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  const ndcX = cx / cw;
  const ndcY = cy / cw;
  return [(ndcX * 0.5 + 0.5) * vpWidth, (0.5 - ndcY * 0.5) * vpHeight];
}

function findFrontSelectableBlock(
  engine: GameEngine,
  rect: { left: number; top: number; width: number; height: number },
  preferLooseOuter: boolean = true
): { blockId: number; screenPt: [number, number]; endWorldPt: Vec3 } {
  const candidates = engine.physics.blocks.filter(
    (b) =>
      !b.shattered &&
      !b.extracted &&
      b.isLegalTarget &&
      b.assignedLayer >= 4 &&
      b.assignedLayer <= 14 &&
      (!preferLooseOuter || b.assignedSlot !== 1)
  );

  for (const blk of candidates) {
    const axisX = quatRotateVec3(blk.rotation, [1, 0, 0]);
    for (const sign of [1, -1] as const) {
      const endWorld: Vec3 = [
        blk.position[0] + axisX[0] * BLOCK_HALF_X * 0.72 * sign,
        blk.position[1] + axisX[1] * BLOCK_HALF_X * 0.72 * sign,
        blk.position[2] + axisX[2] * BLOCK_HALF_X * 0.72 * sign,
      ];
      const [sx, sy] = worldToScreenPx(engine, endWorld, rect.width, rect.height);
      if (sx < 20 || sx > rect.width - 20 || sy < 20 || sy > rect.height - 20) continue;
      const { origin, dir } = engine.camera.screenToRay(sx, sy, rect);
      const hit = raycastIntactBlocksForTest(engine, origin, dir);
      if (hit && hit.blockId === blk.id) {
        return { blockId: blk.id, screenPt: [sx, sy], endWorldPt: endWorld };
      }
    }
  }

  throw new Error(`No front-visible legal block found for viewport ${rect.width}x${rect.height}`);
}

function findBackgroundOrbitPoint(
  engine: GameEngine,
  rect: { left: number; top: number; width: number; height: number }
): [number, number] {
  const probeCoords: Array<[number, number]> = [
    [rect.width * 0.14, rect.height * 0.48],
    [rect.width * 0.86, rect.height * 0.48],
    [rect.width * 0.08, rect.height * 0.32],
    [rect.width * 0.92, rect.height * 0.32],
  ];
  for (const [sx, sy] of probeCoords) {
    const { origin, dir } = engine.camera.screenToRay(sx, sy, rect);
    if (!raycastIntactBlocksForTest(engine, origin, dir)) {
      return [sx, sy];
    }
  }
  throw new Error(`No empty background ray found for viewport ${rect.width}x${rect.height}`);
}

function httpGetJson<T>(httpMod: MinimalHttpModule, port: number, urlPath: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = httpMod.get(
      { hostname: '127.0.0.1', port, path: urlPath, agent: false },
      (res: MinimalHttpResponse) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk: string) => {
          body += chunk;
        });
        res.on('end', () => {
          try {
            resolve(JSON.parse(body) as T);
          } catch (err) {
            reject(err);
          }
        });
      }
    );
    req.on('error', reject);
  });
}

class SimpleCdpPageClient {
  private ws: WebSocket;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (val: unknown) => void; reject: (err: Error) => void }
  >();

  private constructor(ws: WebSocket) {
    this.ws = ws;
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(String(ev.data)) as {
        id?: number;
        result?: unknown;
        error?: { message?: string };
      };
      if (msg.id !== undefined) {
        const entry = this.pending.get(msg.id);
        if (entry) {
          this.pending.delete(msg.id);
          if (msg.error) {
            entry.reject(new Error(msg.error.message ?? 'CDP command failed'));
          } else {
            entry.resolve(msg.result);
          }
        }
      }
    });
  }

  public static async connect(wsUrl: string): Promise<SimpleCdpPageClient> {
    const ws = new WebSocket(wsUrl);
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener('open', () => resolve(), { once: true });
      ws.addEventListener('error', () => reject(new Error(`Failed to open CDP WebSocket ${wsUrl}`)), {
        once: true,
      });
    });
    return new SimpleCdpPageClient(ws);
  }

  public send<T = Record<string, unknown>>(
    method: string,
    params: Record<string, unknown> = {}
  ): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (v) => resolve(v as T),
        reject,
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  public async evaluate<T>(expression: string): Promise<T> {
    const res = await this.send<{
      result: { value?: T; description?: string };
      exceptionDetails?: { text?: string; exception?: { description?: string } };
    }>('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (res.exceptionDetails) {
      throw new Error(
        res.exceptionDetails.exception?.description ??
          res.exceptionDetails.text ??
          'Runtime.evaluate exception'
      );
    }
    return res.result.value as T;
  }

  public close(): void {
    try {
      this.ws.close();
    } catch {
      // Ignore close error
    }
  }
}

const WEBGPU_HEADLESS_SHIM_SCRIPT = `
(() => {
  if (typeof globalThis.GPUBufferUsage === 'undefined') {
    globalThis.GPUBufferUsage = {
      MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
      INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
      INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
    };
  }
  if (typeof globalThis.GPUTextureUsage === 'undefined') {
    globalThis.GPUTextureUsage = {
      COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
      STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
    };
  }
  if (typeof globalThis.GPUShaderStage === 'undefined') {
    globalThis.GPUShaderStage = { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };
  }
  if (typeof globalThis.GPUMapMode === 'undefined') {
    globalThis.GPUMapMode = { READ: 0x1, WRITE: 0x2 };
  }
  if (typeof globalThis.GPUColorWrite === 'undefined') {
    globalThis.GPUColorWrite = { RED: 1, GREEN: 2, BLUE: 4, ALPHA: 8, ALL: 15 };
  }

  const noop = () => {};
  const makePass = () => ({
    setPipeline: noop,
    setBindGroup: noop,
    setVertexBuffer: noop,
    setIndexBuffer: noop,
    setViewport: noop,
    setScissorRect: noop,
    draw: noop,
    drawIndexed: noop,
    dispatchWorkgroups: noop,
    end: noop,
  });
  const makeTexture = () => ({
    createView: () => ({}),
    destroy: noop,
    width: 64,
    height: 64,
  });
  const makeBuffer = (desc) => {
    const ab = new ArrayBuffer(Math.max(16, (desc && desc.size) || 64));
    return {
      size: ab.byteLength,
      getMappedRange: () => ab,
      unmap: noop,
      mapAsync: async () => {},
      destroy: noop,
    };
  };
  const fakeDevice = {
    features: new Set(),
    limits: {},
    queue: {
      submit: noop,
      writeBuffer: noop,
      writeTexture: noop,
      onSubmittedWorkDone: async () => {},
    },
    createBuffer: (desc) => makeBuffer(desc),
    createTexture: () => makeTexture(),
    createSampler: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createBindGroup: () => ({}),
    createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createRenderPipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createCommandEncoder: () => ({
      beginRenderPass: () => makePass(),
      beginComputePass: () => makePass(),
      copyTextureToBuffer: noop,
      copyTextureToTexture: noop,
      copyBufferToBuffer: noop,
      resolveQuerySet: noop,
      finish: () => ({}),
    }),
    createQuerySet: () => ({ destroy: noop }),
    pushErrorScope: noop,
    popErrorScope: async () => null,
    destroy: noop,
    lost: new Promise(() => {}),
  };
  const fakeAdapter = {
    features: new Set(),
    limits: {},
    requestDevice: async () => fakeDevice,
  };

  Object.defineProperty(navigator, 'gpu', {
    configurable: true,
    value: {
      requestAdapter: async () => fakeAdapter,
      getPreferredCanvasFormat: () => 'bgra8unorm',
    },
  });

  const origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, ...rest) {
    if (type === 'webgpu') {
      return {
        configure: noop,
        unconfigure: noop,
        getCurrentTexture: () => makeTexture(),
      };
    }
    return origGetContext.call(this, type, ...rest);
  };
})();
`;

describe('Vitrail Multi-Touch Disambiguation, Responsive HUD, Audio Guards & Glazier 2-Ply AI', () => {
  it('1. disambiguates 1-finger tap vs 1-finger pull vs 1-finger background orbit vs 2-finger pinch-zoom/pan across all 4 viewports', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    try {
      for (const vp of VIEWPORTS) {
        engine.resetGame('solo', 2);
        const rect = { left: 0, top: 0, width: vp.width, height: vp.height };
        engine.camera.update(0.25, vp.width / vp.height);

        const target = findFrontSelectableBlock(engine, rect, true);

        // A. 1-finger tap (< 5 px movement) -> acoustic tap-test WITHOUT leaving SELECT_BLOCK
        let tapResonanceCalls = 0;
        const origTapRes = engine.audio.playTapTestResonance.bind(engine.audio);
        engine.audio.playTapTestResonance = (...args) => {
          tapResonanceCalls++;
          origTapRes(...args);
        };

        engine.handlePointerDown(
          target.screenPt[0],
          target.screenPt[1],
          11,
          0,
          false,
          rect,
          'touch',
          1000
        );
        expect(engine.getInteractionMode()).toBe('block_pending');
        expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');

        engine.handlePointerMove(
          target.screenPt[0] + 2.5,
          target.screenPt[1] - 1.5,
          11,
          rect,
          'touch'
        );
        expect(engine.getInteractionMode()).toBe('block_pending');

        engine.handlePointerUp(
          target.screenPt[0] + 2.5,
          target.screenPt[1] - 1.5,
          11,
          rect,
          'touch',
          1040
        );
        expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
        expect(engine.physics.activeBlockId).toBeNull();
        expect(tapResonanceCalls).toBe(1);
        engine.audio.playTapTestResonance = origTapRes;

        // B. 1-finger pull (>= 5 px movement) -> promotes SELECT_BLOCK -> PULLING
        engine.handlePointerDown(
          target.screenPt[0],
          target.screenPt[1],
          12,
          0,
          false,
          rect,
          'touch',
          2000
        );
        engine.handlePointerMove(
          target.screenPt[0] + 14,
          target.screenPt[1] + 2,
          12,
          rect,
          'touch'
        );
        expect(engine.getInteractionMode()).toBe('block');
        expect(engine.physics.turnPhase).toBe('PULLING');
        expect(engine.physics.activeBlockId).toBe(target.blockId);

        // Release early before extraction -> aborts cleanly back to SELECT_BLOCK
        engine.handlePointerUp(
          target.screenPt[0] + 14,
          target.screenPt[1] + 2,
          12,
          rect,
          'touch',
          2120
        );
        expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
        expect(engine.getInteractionMode()).toBe('none');

        // C. 1-finger background drag -> orbits camera without grabbing any block
        const bgPt = findBackgroundOrbitPoint(engine, rect);
        const camTargets = getCameraTargetPose(engine.camera);
        const azBefore = camTargets.targetAzimuth;
        const elBefore = camTargets.targetElevation;

        engine.handlePointerDown(bgPt[0], bgPt[1], 13, 0, false, rect, 'touch', 3000);
        expect(engine.getInteractionMode()).toBe('orbit');
        engine.handlePointerMove(bgPt[0] + 48, bgPt[1] - 24, 13, rect, 'touch');
        expect(camTargets.targetAzimuth).not.toBeCloseTo(azBefore, 4);
        expect(camTargets.targetElevation).not.toBeCloseTo(elBefore, 4);
        expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
        engine.handlePointerUp(bgPt[0] + 48, bgPt[1] - 24, 13, rect, 'touch', 3090);
        expect(engine.getInteractionMode()).toBe('none');

        // D. 2-finger pinch-zoom + midpoint pan -> updates distance + panOffset and locks until all fingers lift
        const distBefore = camTargets.targetDistance;
        const panBefore = [...camTargets.targetPanOffset] as Vec3;

        engine.handlePointerDown(
          vp.width * 0.4,
          vp.height * 0.5,
          21,
          0,
          false,
          rect,
          'touch',
          4000
        );
        engine.handlePointerDown(
          vp.width * 0.6,
          vp.height * 0.5,
          22,
          0,
          false,
          rect,
          'touch',
          4012
        );
        expect(engine.getInteractionMode()).toBe('pinch_pan');
        expect(engine.getActiveTouchCount()).toBe(2);

        // Spread fingers wider (zoom in -> smaller targetDistance) while translating midpoint (+40 px X, -30 px Y)
        engine.handlePointerMove(vp.width * 0.34 + 40, vp.height * 0.5 - 30, 21, rect, 'touch');
        engine.handlePointerMove(vp.width * 0.66 + 40, vp.height * 0.5 - 30, 22, rect, 'touch');

        expect(camTargets.targetDistance).toBeLessThan(distBefore);
        const panDelta = Math.hypot(
          camTargets.targetPanOffset[0] - panBefore[0],
          camTargets.targetPanOffset[1] - panBefore[1],
          camTargets.targetPanOffset[2] - panBefore[2]
        );
        expect(panDelta).toBeGreaterThan(0.05);

        engine.camera.update(0.12, vp.width / vp.height);
        expect(
          Math.hypot(
            engine.camera.panOffset[0],
            engine.camera.panOffset[1],
            engine.camera.panOffset[2]
          )
        ).toBeGreaterThan(0.02);

        // Lifting 1 of the 2 fingers keeps pinch_pan locked so the remaining finger cannot trigger orbit/grab
        engine.handlePointerUp(
          vp.width * 0.66 + 40,
          vp.height * 0.5 - 30,
          22,
          rect,
          'touch',
          4090
        );
        expect(engine.getActiveTouchCount()).toBe(1);
        expect(engine.getInteractionMode()).toBe('pinch_pan');

        const azDuringLift = camTargets.targetAzimuth;
        engine.handlePointerMove(vp.width * 0.34 + 95, vp.height * 0.5 + 20, 21, rect, 'touch');
        expect(camTargets.targetAzimuth).toBeCloseTo(azDuringLift, 6);
        expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');

        // Lifting the final finger cleanly exits pinch_pan
        engine.handlePointerUp(
          vp.width * 0.34 + 95,
          vp.height * 0.5 + 20,
          21,
          rect,
          'touch',
          4130
        );
        expect(engine.getActiveTouchCount()).toBe(0);
        expect(engine.getInteractionMode()).toBe('none');
      }

      // Portrait FOV adaptation check vs setPoseImmediate optical-bench parity
      const cam = new OrbitCamera();
      cam.update(0.2, 390 / 844);
      const portraitM5 = cam.projMatrix[5];
      cam.update(0.2, 1440 / 900);
      const landscapeM5 = cam.projMatrix[5];
      expect(portraitM5).toBeLessThan(landscapeM5); // Wider vertical FOV on narrow portrait viewport

      cam.setPoseImmediate(cam.azimuth, cam.elevation, cam.distance, cam.target, 390 / 844);
      expect(cam.projMatrix[5]).toBeCloseTo(landscapeM5, 5); // Exact 40 deg bench parity
    } finally {
      engine.dispose();
    }
  });

  it('2. cancels any accidental optimistic 1-finger block grab if a 2nd finger touches down within <= 80 ms while protecting established (> 80 ms) pulls', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    try {
      const rect = { left: 0, top: 0, width: 1024, height: 1366 };
      engine.camera.update(0.25, rect.width / rect.height);
      const target = findFrontSelectableBlock(engine, rect, true);

      // Case A: 2nd finger lands at +58 ms (<= 80 ms) AFTER 1st finger already crossed 5 px drag into PULLING
      engine.handlePointerDown(
        target.screenPt[0],
        target.screenPt[1],
        1,
        0,
        false,
        rect,
        'touch',
        1000
      );
      engine.handlePointerMove(
        target.screenPt[0] + 10,
        target.screenPt[1] + 1,
        1,
        rect,
        'touch'
      );
      expect(engine.getInteractionMode()).toBe('block');
      expect(engine.physics.turnPhase).toBe('PULLING');
      expect(engine.physics.activeBlockId).toBe(target.blockId);

      // Second finger touches down at t = 1058 ms (+58 ms <= 80 ms window)
      engine.handlePointerDown(
        target.screenPt[0] + 180,
        target.screenPt[1] - 40,
        2,
        0,
        false,
        rect,
        'touch',
        1058
      );
      expect(engine.getInteractionMode()).toBe('pinch_pan');
      expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
      expect(engine.physics.activeBlockId).toBeNull();

      engine.handlePointerUp(target.screenPt[0] + 180, target.screenPt[1] - 40, 2, rect, 'touch', 1100);
      engine.handlePointerUp(target.screenPt[0] + 10, target.screenPt[1] + 1, 1, rect, 'touch', 1110);
      expect(engine.getInteractionMode()).toBe('none');

      // Case B: Deliberate 1-finger pull held for 140 ms (> 80 ms) ignores stray secondary palm/finger touch
      engine.handlePointerDown(
        target.screenPt[0],
        target.screenPt[1],
        3,
        0,
        false,
        rect,
        'touch',
        2000
      );
      engine.handlePointerMove(
        target.screenPt[0] + 16,
        target.screenPt[1],
        3,
        rect,
        'touch'
      );
      expect(engine.getInteractionMode()).toBe('block');
      expect(engine.physics.turnPhase).toBe('PULLING');

      engine.handlePointerDown(
        960,
        1280,
        4,
        0,
        false,
        rect,
        'touch',
        2140
      );
      expect(engine.getInteractionMode()).toBe('block');
      expect(engine.physics.turnPhase).toBe('PULLING');
      expect(engine.physics.activeBlockId).toBe(target.blockId);

      engine.handlePointerCancel();
    } finally {
      engine.dispose();
    }
  });

  it('3. completes 1-finger EXTRACTED top-course rotation & placement without mouse hover (steep & grazing camera angles)', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    try {
      const rect = { left: 0, top: 0, width: 1024, height: 1366 };
      engine.camera.update(0.25, rect.width / rect.height);

      // Helper to extract a loose prism from an intact 3-block course into EXTRACTED state with ZERO hover events
      const extractLooseBlockWithoutHover = (): number => {
        const blk = engine.physics.blocks.find((b) => {
          if (b.shattered || b.extracted || !b.isLegalTarget || !b.isLooseSlot) return false;
          if (b.assignedLayer < 6 || b.assignedLayer > 14) return false;
          const sameLayerIntact = engine.physics.blocks.filter(
            (nb) => !nb.shattered && !nb.extracted && nb.assignedLayer === b.assignedLayer
          ).length;
          return sameLayerIntact === 3;
        })!;
        expect(blk).toBeDefined();

        const plan = engine.computeOptimalExtractionPlan(blk.id, 'master')!;
        expect(engine.physics.startGrabBlock(blk.id, plan.localGrab, plan.startWorld)).toBe(true);

        for (let step = 0; step < 150 && engine.physics.turnPhase !== 'EXTRACTED'; step++) {
          const dist = Math.min(1.9, (step + 1) * 0.028);
          engine.physics.updateGrabTarget([
            plan.startWorld[0] + plan.pullDirXZ[0] * dist,
            plan.startWorld[1],
            plan.startWorld[2] + plan.pullDirXZ[1] * dist,
          ]);
          engine.physics.step(1 / 60);
        }
        expect(engine.physics.turnPhase).toBe('EXTRACTED');
        engine.physics.endGrab(false);

        // Advance arc timer past 0.20 s so isExtractionLiftReady === true (still ZERO hover calls!)
        for (let i = 0; i < 18; i++) {
          engine.physics.step(1 / 60);
        }
        expect(engine.physics.isExtractionLiftReady).toBe(true);
        return blk.id;
      };

      // Turn 1: Place first block on center slot (slot 1) via grazing 10-deg 1-finger touch without hover!
      extractLooseBlockWithoutHover();
      engine.camera.setPoseImmediate(
        0.62,
        (10 * Math.PI) / 180,
        5.8,
        [0, engine.physics.topSupportFrame.layerCenter[1], 0],
        rect.width / rect.height
      );

      const openSlot0 = engine.physics.openTopSlots.find((s) => s.slotIndex === 1) ?? engine.physics.openTopSlots[0];
      const [slotSx, slotSy] = worldToScreenPx(engine, openSlot0.position, rect.width, rect.height);

      engine.handlePointerDown(slotSx, slotSy, 31, 0, false, rect, 'touch', 5000);
      expect(engine.getInteractionMode()).toBe('place_top');
      engine.handlePointerUp(slotSx, slotSy, 31, rect, 'touch', 5045);
      expect(engine.physics.turnPhase).not.toBe('EXTRACTED');

      for (let i = 0; i < 120 && engine.physics.turnPhase === 'SETTLING'; i++) {
        engine.physics.step(1 / 60);
      }
      expect(engine.physics.blocksStacked).toBe(1);

      // Turn 2: Center slot 1 is now OCCUPIED! Extract a second block with ZERO hover, rotate via Q / 90 R / E / 0,
      // and verify carriedTargetPos stays on an open outer slot rather than jumping onto occupied center [0, 0]!
      engine.camera.resetView(27.0);
      engine.camera.update(0.3, rect.width / rect.height);
      extractLooseBlockWithoutHover();

      const occupiedCenter = engine.physics.topSupportFrame.layerCenter;
      engine.handleRotateCarried(Math.PI / 2); // 90° R button without prior hover
      engine.handleRotateCarried(-Math.PI / 12); // Q button
      engine.handleRotateCarried(Math.PI / 12); // E button
      engine.handleResetCarriedRotation(); // 0° align button

      const distFromOccupiedCenter = Math.hypot(
        engine.physics.carriedTargetPos[0] - occupiedCenter[0],
        engine.physics.carriedTargetPos[2] - occupiedCenter[2]
      );
      expect(distFromOccupiedCenter).toBeGreaterThan(0.18); // Safely on open outer slot (+/- 0.25 dm), NOT [0, 0]!

      // Confirm HUD Place Prism button releases cleanly without hover
      engine.handleReleaseOntoTop();
      for (let i = 0; i < 120 && engine.physics.turnPhase === 'SETTLING'; i++) {
        engine.physics.step(1 / 60);
      }
      expect(engine.physics.blocksStacked).toBe(2);
      expect(engine.physics.shatteredCount).toBe(0);
    } finally {
      engine.dispose();
    }
  });

  it('4. prevents duplicate nested outbound audio events via installHostAudioTap re-entrancy guard', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    try {
      // Attach a running AudioContext so playGlassCrack, playGlassShatter, and playWobbleRock
      // execute their full bodies (which internally call playGlassImpact / playShardImpacts!)
      const makeParam = (init: number) => ({
        value: init,
        setValueAtTime: () => {},
        linearRampToValueAtTime: () => {},
        exponentialRampToValueAtTime: () => {},
        setTargetAtTime: () => {},
        cancelScheduledValues: () => {},
      });
      let mockTimeSec = 1.0;
      const mockCtx = {
        get currentTime() {
          return mockTimeSec;
        },
        sampleRate: 48000,
        state: 'running',
        destination: {},
        createGain: () => ({ gain: makeParam(1), connect: () => {}, disconnect: () => {} }),
        createDynamicsCompressor: () => ({
          threshold: makeParam(-24),
          knee: makeParam(30),
          ratio: makeParam(12),
          attack: makeParam(0.003),
          release: makeParam(0.25),
          connect: () => {},
          disconnect: () => {},
        }),
        createBiquadFilter: () => ({
          type: 'bandpass',
          frequency: makeParam(1000),
          Q: makeParam(1),
          gain: makeParam(0),
          connect: () => {},
          disconnect: () => {},
        }),
        createStereoPanner: () => ({
          pan: makeParam(0),
          connect: () => {},
          disconnect: () => {},
        }),
        createOscillator: () => ({
          type: 'sine',
          frequency: makeParam(440),
          onended: null,
          connect: () => {},
          disconnect: () => {},
          start: () => {},
          stop: () => {},
        }),
        createBuffer: (_ch: number, len: number, sr: number) => ({
          sampleRate: sr,
          length: len,
          duration: len / sr,
          numberOfChannels: 1,
          getChannelData: () => new Float32Array(len),
        }),
        createBufferSource: () => ({
          buffer: null,
          loop: false,
          onended: null,
          connect: () => {},
          disconnect: () => {},
          start: () => {},
          stop: () => {},
        }),
        resume: async () => {},
        close: async () => {},
      } as unknown as AudioContext;

      engine.audio.dispose();
      engine.audio = new GlassAudioEngine(mockCtx);

      // Simulate active authoritative Host with a connected Guest peer
      (engine as unknown as { mpRole: 'host' }).mpRole = 'host';
      (engine as unknown as { mpStatus: { peerConnected: boolean } }).mpStatus.peerConnected = true;

      // Calling installHostAudioTap multiple times must be strictly idempotent
      engine.installHostAudioTap();
      engine.installHostAudioTap();

      let glassImpactUnderlyingCalls = 0;
      const wrappedImpact = engine.audio.playGlassImpact.bind(engine.audio);
      engine.audio.playGlassImpact = (...args) => {
        glassImpactUnderlyingCalls++;
        wrappedImpact(...args);
      };

      // 1. playGlassCrack internally calls this.playGlassImpact(...)
      // 2. playGlassShatter internally calls this.playShardImpacts(...)
      // 3. playWobbleRock schedules Housner micro-taps
      // 4. Direct top-level playGlassImpact MUST still emit 'glassImpact'!
      mockTimeSec += 0.1;
      engine.audio.playGlassCrack(0.75, [0.1, 1.2, 0]);
      mockTimeSec += 0.1;
      engine.audio.playGlassShatter(0.92, true, [-0.8, 0.12, 0.4]);
      mockTimeSec += 0.1;
      engine.audio.playWobbleRock(0.09, 0.85, 1.4, [0.2, 0.45, 0], { surface: 'glass' });
      mockTimeSec += 0.1;
      engine.audio.playGlassImpact(0.7, 1.0, [0, 0.5, 0]);

      // Confirm nested playGlassImpact DID execute inside playGlassCrack (2 total calls = 1 nested + 1 direct),
      // yet ONLY the 1 direct top-level call was enqueued onto outbound multiplayer audio (and 0 nested shardImpacts)!
      expect(glassImpactUnderlyingCalls).toBeGreaterThanOrEqual(2);

      const outbound = engine.getOutboundAudioQueueSnapshot();
      expect(outbound.map((ev) => ev.kind)).toEqual([
        'glassCrack',
        'glassShatter',
        'wobbleRock',
        'glassImpact',
      ]);
    } finally {
      engine.dispose();
    }
  });

  it('5. enforces >= 44x44 CSS px tap targets, env(safe-area-inset-*), zero HUD overlap across 4 viewports, and ARIA/keyboard modal & QuarryButton accessibility in real Chromium', async () => {
    const cpModName = 'node:child_process';
    const fsModName = 'node:fs';
    const httpModName = 'node:http';
    const osModName = 'node:os';
    const pathModName = 'node:path';
    const cpMod = (await import(/* @vite-ignore */ cpModName)) as unknown as MinimalChildProcessModule;
    const fsMod = (await import(/* @vite-ignore */ fsModName)) as unknown as MinimalFsModule;
    const httpMod = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
    const osMod = (await import(/* @vite-ignore */ osModName)) as unknown as MinimalOsModule;
    const pathMod = (await import(/* @vite-ignore */ pathModName)) as unknown as MinimalPathModule;
    const cwd = getNodeCwd();

    const cssSource = fsMod.readFileSync(
      pathMod.resolve(cwd, 'src/styles/index.css'),
      'utf8'
    );
    expect(cssSource).toContain('env(safe-area-inset-top');
    expect(cssSource).toContain('env(safe-area-inset-right');
    expect(cssSource).toContain('env(safe-area-inset-bottom');
    expect(cssSource).toContain('env(safe-area-inset-left');
    expect(cssSource).toMatch(/\.webgpu-canvas\s*\{[^}]*touch-action:\s*none/s);

    let viteServer: ViteDevServer | null = null;
    let chromeProc: MinimalChildProcess | null = null;
    let cdp: SimpleCdpPageClient | null = null;
    const userDataDir = fsMod.mkdtempSync(pathMod.join(osMod.tmpdir(), 'vitrail-hud-cdp-'));

    try {
      viteServer = await createServer({
        root: cwd,
        logLevel: 'silent',
        server: { host: '127.0.0.1', port: 0, strictPort: false },
      });
      await viteServer.listen();
      const addr = viteServer.httpServer?.address();
      const vitePort = typeof addr === 'object' && addr ? addr.port : 0;
      expect(vitePort).toBeGreaterThan(0);

      chromeProc = cpMod.spawn(
        '/usr/bin/google-chrome',
        [
          '--headless=new',
          '--no-sandbox',
          '--disable-gpu',
          '--disable-dev-shm-usage',
          '--no-first-run',
          '--no-default-browser-check',
          '--no-proxy-server',
          `--user-data-dir=${userDataDir}`,
          '--remote-debugging-port=0',
          'about:blank',
        ],
        { stdio: ['ignore', 'ignore', 'pipe'] }
      );

      const decoder = new TextDecoder('utf-8');
      const cdpPort = await new Promise<number>((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error('Timed out waiting for Chrome DevTools port')),
          10000
        );
        let stderrBuf = '';
        chromeProc!.stderr!.on('data', (chunk: Uint8Array) => {
          stderrBuf += decoder.decode(chunk, { stream: true });
          const match = stderrBuf.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//);
          if (match) {
            clearTimeout(timeout);
            resolve(Number(match[1]));
          }
        });
        chromeProc!.once('exit', (code: number | null) => {
          clearTimeout(timeout);
          reject(new Error(`Chrome exited prematurely with code ${code}: ${stderrBuf}`));
        });
      });

      const targets = await httpGetJson<Array<{ type: string; webSocketDebuggerUrl?: string }>>(
        httpMod,
        cdpPort,
        '/json/list'
      );
      const pageTarget = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      expect(pageTarget?.webSocketDebuggerUrl).toBeDefined();

      cdp = await SimpleCdpPageClient.connect(pageTarget!.webSocketDebuggerUrl!);
      await cdp.send('Page.enable');
      await cdp.send('Runtime.enable');
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: 1440,
        height: 900,
        deviceScaleFactor: 2,
        mobile: false,
      });
      await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
        source: WEBGPU_HEADLESS_SHIM_SCRIPT,
      });

      await cdp.send('Page.navigate', {
        url: `http://127.0.0.1:${vitePort}/?worker=0`,
      });

      // Wait for App + __VITRAIL_ENGINE__ to mount with zero gpuError
      await cdp.evaluate(`new Promise((resolve, reject) => {
        const deadline = Date.now() + 12000;
        const poll = () => {
          const eng = window.__VITRAIL_ENGINE__;
          const errModal = document.querySelector('[role="alertdialog"]');
          if (errModal) {
            reject(new Error('Unexpected gpuError modal: ' + errModal.textContent));
            return;
          }
          if (eng && eng.physics && document.querySelector('.vitrail-lancet-window')) {
            resolve(true);
            return;
          }
          if (Date.now() > deadline) {
            reject(new Error('Timed out waiting for __VITRAIL_ENGINE__ in headless Chrome'));
            return;
          }
          setTimeout(poll, 40);
        };
        poll();
      })`);

      // Verify viewport-fit=cover injection & canvas computed touch-action: none
      const bootChecks = await cdp.evaluate<{
        viewportContent: string;
        canvasTouchAction: string;
      }>(`(() => ({
        viewportContent: document.querySelector('meta[name="viewport"]')?.getAttribute('content') ?? '',
        canvasTouchAction: getComputedStyle(document.querySelector('.webgpu-canvas')).touchAction,
      }))()`);
      expect(bootChecks.viewportContent).toContain('viewport-fit=cover');
      expect(bootChecks.canvasTouchAction).toBe('none');

      // Helper to audit all visible interactive controls (>= 44x44 CSS px) & 0 HUD zone overlap
      const auditViewportLayout = async (vpWidth: number, vpHeight: number, stateLabel: string) => {
        await cdp!.send('Emulation.setDeviceMetricsOverride', {
          width: vpWidth,
          height: vpHeight,
          deviceScaleFactor: 2,
          mobile: vpWidth < 1100,
        });
        await cdp!.evaluate(
          `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`
        );

        const report = await cdp!.evaluate<{
          undersized: Array<{ label: string; width: number; height: number }>;
          offscreen: Array<{ label: string; left: number; top: number; right: number; bottom: number }>;
          zoneOverlaps: Array<{ pair: string; area: number }>;
          controlCount: number;
        }>(`(() => {
          const isVisible = (el) => {
            const st = getComputedStyle(el);
            if (st.display === 'none' || st.visibility === 'hidden' || Number(st.opacity) === 0) return false;
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          };

          const controls = Array.from(document.querySelectorAll('button, input, [role="button"], [role="radio"], [role="tab"]'))
            .filter(isVisible);

          const undersized = [];
          const offscreen = [];
          for (const el of controls) {
            const r = el.getBoundingClientRect();
            const label = (el.getAttribute('aria-label') || el.textContent || el.tagName).trim().slice(0, 40);
            if (r.width < 43.9 || r.height < 43.9) {
              undersized.push({ label, width: +r.width.toFixed(2), height: +r.height.toFixed(2) });
            }
            if (r.left < -0.5 || r.top < -0.5 || r.right > window.innerWidth + 0.5 || r.bottom > window.innerHeight + 0.5) {
              offscreen.push({
                label,
                left: +r.left.toFixed(1),
                top: +r.top.toFixed(1),
                right: +r.right.toFixed(1),
                bottom: +r.bottom.toFixed(1),
              });
            }
          }

          const zones = [
            ['lancet', document.querySelector('.vitrail-lancet-window')],
            ['plate', document.querySelector('.crystal-optical-plate')],
            ['prism', document.querySelector('.cut-glass-prism-bar')],
          ].filter(([, el]) => el && isVisible(el));

          const zoneOverlaps = [];
          for (let i = 0; i < zones.length; i++) {
            for (let j = i + 1; j < zones.length; j++) {
              const a = zones[i][1].getBoundingClientRect();
              const b = zones[j][1].getBoundingClientRect();
              const ix = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
              const iy = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
              const area = +(ix * iy).toFixed(2);
              if (area > 0) {
                zoneOverlaps.push({ pair: zones[i][0] + '<->' + zones[j][0], area });
              }
            }
          }

          return { undersized, offscreen, zoneOverlaps, controlCount: controls.length };
        })()`);

        expect(
          report.undersized,
          `${stateLabel} @ ${vpWidth}x${vpHeight} undersized controls`
        ).toEqual([]);
        expect(
          report.offscreen,
          `${stateLabel} @ ${vpWidth}x${vpHeight} offscreen controls`
        ).toEqual([]);
        expect(
          report.zoneOverlaps,
          `${stateLabel} @ ${vpWidth}x${vpHeight} HUD zone overlaps`
        ).toEqual([]);
        expect(report.controlCount).toBeGreaterThanOrEqual(10);
      };

      // State 1: Pass & Play -> AI active (Apprentice vs Master Glazier radiogroup visible!)
      await cdp.evaluate(`(() => {
        const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
        const partyTab = tabs.find((el) => el.textContent?.includes('Pass & Play'));
        partyTab?.click();
      })()`);
      await cdp.evaluate(`(() => {
        const aiBtn = document.querySelector('button[aria-label*="Glazier AI"]');
        aiBtn?.click();
      })()`);

      const aiRadioCheck = await cdp.evaluate<{
        hasRadioGroup: boolean;
        radioLabels: string[];
        engineDifficultyAfterApprentice: string;
        engineDifficultyAfterMaster: string;
      }>(`(() => {
        const group = document.querySelector('[role="radiogroup"][aria-label="AI Difficulty"]');
        const radios = group ? Array.from(group.querySelectorAll('[role="radio"]')) : [];
        const appBtn = radios.find((r) => r.textContent?.includes('Apprentice'));
        const masBtn = radios.find((r) => r.textContent?.includes('Master'));
        appBtn?.click();
        const afterApp = window.__VITRAIL_ENGINE__.aiDifficulty;
        masBtn?.click();
        const afterMas = window.__VITRAIL_ENGINE__.aiDifficulty;
        return {
          hasRadioGroup: Boolean(group),
          radioLabels: radios.map((r) => r.textContent?.trim() ?? ''),
          engineDifficultyAfterApprentice: afterApp,
          engineDifficultyAfterMaster: afterMas,
        };
      })()`);
      expect(aiRadioCheck.hasRadioGroup).toBe(true);
      expect(aiRadioCheck.radioLabels).toEqual(['Apprentice', 'Master Glazier']);
      expect(aiRadioCheck.engineDifficultyAfterApprentice).toBe('apprentice');
      expect(aiRadioCheck.engineDifficultyAfterMaster).toBe('master');

      for (const vp of VIEWPORTS) {
        await auditViewportLayout(vp.width, vp.height, 'Pass & Play + AI Difficulty');
      }

      // State 2: EXTRACTED state (Q, 90° R, E, align 0°, and Place Prism controls visible!)
      const extractedTurnPhase = await cdp.evaluate<string>(`new Promise((resolve) => {
        const eng = window.__VITRAIL_ENGINE__;
        eng.resetGame('solo', 2);
        const blk = eng.physics.blocks.find((b) => b.isLegalTarget && b.isLooseSlot && b.assignedLayer === 8);
        const plan = eng.computeOptimalExtractionPlan(blk.id, 'master');
        eng.physics.startGrabBlock(blk.id, plan.localGrab, plan.startWorld);
        for (let i = 0; i < 140 && eng.physics.turnPhase !== 'EXTRACTED'; i++) {
          const dist = Math.min(1.9, (i + 1) * 0.028);
          eng.physics.updateGrabTarget([
            plan.startWorld[0] + plan.pullDirXZ[0] * dist,
            plan.startWorld[1],
            plan.startWorld[2] + plan.pullDirXZ[1] * dist,
          ]);
          eng.physics.step(1 / 60);
        }
        eng.physics.endGrab(false);
        for (let i = 0; i < 20; i++) eng.physics.step(1 / 60);
        eng.handleRotateCarried(Math.PI / 12); // Rotate slightly so 0° align button ALSO renders + emits telemetry!
        requestAnimationFrame(() => requestAnimationFrame(() => resolve(eng.physics.turnPhase)));
      })`);
      expect(extractedTurnPhase).toBe('EXTRACTED');

      const extractedButtonsPresent = await cdp.evaluate<string[]>(`(() => {
        return Array.from(document.querySelectorAll('.cut-glass-prism-bar button'))
          .map((b) => b.getAttribute('aria-label') || '');
      })()`);
      expect(extractedButtonsPresent.some((s) => s.includes('counter-clockwise 7.5 degrees (Q)'))).toBe(true);
      expect(extractedButtonsPresent.some((s) => s.includes('90 degrees (R)'))).toBe(true);
      expect(extractedButtonsPresent.some((s) => s.includes('clockwise 7.5 degrees (E)'))).toBe(true);
      expect(extractedButtonsPresent.some((s) => s.includes('Align prism square'))).toBe(true);
      expect(extractedButtonsPresent.some((s) => s.includes('Place Prism onto top course'))).toBe(true);

      for (const vp of VIEWPORTS) {
        await auditViewportLayout(vp.width, vp.height, 'EXTRACTED Controls (Q / 90 R / E / 0° / Place Prism)');
      }

      // State 3: Online Room Code controls (Host, CODE input, Join)
      await cdp.evaluate(`(() => {
        window.__VITRAIL_ENGINE__.resetGame('solo', 2);
        const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
        const onlineTab = tabs.find((el) => el.textContent?.includes('Online'));
        onlineTab?.click();
      })()`);

      for (const vp of VIEWPORTS) {
        await auditViewportLayout(vp.width, vp.height, 'Online Room Code Controls');
      }

      // Keyboard QuarryButton activation (Enter & Space) + Guide Modal ARIA & Focus Trap
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: 1024,
        height: 1366,
        deviceScaleFactor: 2,
        mobile: false,
      });

      await cdp.evaluate(`new Promise((resolve) => {
        const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
        const soloTab = tabs.find((el) => el.textContent?.includes('Solo'));
        soloTab?.click();
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      })`);

      await cdp.evaluate(`(() => {
        const polBtn = document.querySelector('button[aria-label*="Polariscope"]');
        polBtn.focus();
      })()`);

      // Real OS/CDP Enter keydown -> QuarryButton sets data-pressing="true" and fires native click
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyDown',
        key: 'Enter',
        code: 'Enter',
        text: '\r',
        unmodifiedText: '\r',
        windowsVirtualKeyCode: 13,
        nativeVirtualKeyCode: 13,
      });
      const enterPressedAttr = await cdp.evaluate<string | null>(
        `document.querySelector('button[aria-label*="Polariscope"]')?.getAttribute('data-pressing') ?? null`
      );
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key: 'Enter',
        code: 'Enter',
        windowsVirtualKeyCode: 13,
        nativeVirtualKeyCode: 13,
      });
      const polariscopeAfterEnter = await cdp.evaluate<boolean>(
        `Boolean(window.__VITRAIL_ENGINE__.polariscopeMode)`
      );

      // Real OS/CDP Space keydown + keyup -> toggles Polariscope back off
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyDown',
        key: ' ',
        code: 'Space',
        text: ' ',
        windowsVirtualKeyCode: 32,
        nativeVirtualKeyCode: 32,
      });
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key: ' ',
        code: 'Space',
        windowsVirtualKeyCode: 32,
        nativeVirtualKeyCode: 32,
      });
      const polariscopeAfterSpace = await cdp.evaluate<boolean>(
        `Boolean(window.__VITRAIL_ENGINE__.polariscopeMode)`
      );

      // Focus Guide button ("How to Play & Controls") and open via real CDP Enter
      await cdp.evaluate(`(() => {
        const guideBtn = document.querySelector('button[aria-label="How to Play & Controls"]');
        guideBtn.focus();
      })()`);
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyDown',
        key: 'Enter',
        code: 'Enter',
        text: '\r',
        unmodifiedText: '\r',
        windowsVirtualKeyCode: 13,
        nativeVirtualKeyCode: 13,
      });
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key: 'Enter',
        code: 'Enter',
        windowsVirtualKeyCode: 13,
        nativeVirtualKeyCode: 13,
      });
      await cdp.evaluate(`new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`);

      const modalOpenState = await cdp.evaluate<{
        dialogOpen: boolean;
        hudAriaHiddenWhileOpen: string | null;
        canvasAriaHiddenWhileOpen: string | null;
        initialFocusLabel: string;
      }>(`(() => {
        const dialog = document.querySelector('[role="dialog"][aria-modal="true"]');
        return {
          dialogOpen: Boolean(dialog),
          hudAriaHiddenWhileOpen: document.querySelector('.hud-layer')?.getAttribute('aria-hidden') ?? null,
          canvasAriaHiddenWhileOpen: document.querySelector('.webgpu-canvas')?.getAttribute('aria-hidden') ?? null,
          initialFocusLabel: document.activeElement?.getAttribute('aria-label') || document.activeElement?.textContent?.trim() || '',
        };
      })()`);

      // Focus last modal button and press Tab -> focus trap wraps to first button
      await cdp.evaluate(`(() => {
        const dialog = document.querySelector('[role="dialog"][aria-modal="true"]');
        const focusables = Array.from(dialog.querySelectorAll('button:not([disabled])'));
        focusables[focusables.length - 1].focus();
      })()`);
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyDown',
        key: 'Tab',
        code: 'Tab',
        windowsVirtualKeyCode: 9,
        nativeVirtualKeyCode: 9,
      });
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key: 'Tab',
        code: 'Tab',
        windowsVirtualKeyCode: 9,
        nativeVirtualKeyCode: 9,
      });
      const tabWrappedToFirst = await cdp.evaluate<boolean>(`(() => {
        const dialog = document.querySelector('[role="dialog"][aria-modal="true"]');
        const focusables = Array.from(dialog.querySelectorAll('button:not([disabled])'));
        return document.activeElement === focusables[0];
      })()`);

      // Focus first modal button and press Shift+Tab -> focus trap wraps to last button
      await cdp.evaluate(`(() => {
        const dialog = document.querySelector('[role="dialog"][aria-modal="true"]');
        const focusables = Array.from(dialog.querySelectorAll('button:not([disabled])'));
        focusables[0].focus();
      })()`);
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyDown',
        key: 'Tab',
        code: 'Tab',
        modifiers: 8,
        windowsVirtualKeyCode: 9,
        nativeVirtualKeyCode: 9,
      });
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key: 'Tab',
        code: 'Tab',
        modifiers: 8,
        windowsVirtualKeyCode: 9,
        nativeVirtualKeyCode: 9,
      });
      const shiftTabWrappedToLast = await cdp.evaluate<boolean>(`(() => {
        const dialog = document.querySelector('[role="dialog"][aria-modal="true"]');
        const focusables = Array.from(dialog.querySelectorAll('button:not([disabled])'));
        return document.activeElement === focusables[focusables.length - 1];
      })()`);

      // Press Escape -> modal closes and restores focus to Guide trigger button
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyDown',
        key: 'Escape',
        code: 'Escape',
        windowsVirtualKeyCode: 27,
        nativeVirtualKeyCode: 27,
      });
      await cdp.send('Input.dispatchKeyEvent', {
        type: 'keyUp',
        key: 'Escape',
        code: 'Escape',
        windowsVirtualKeyCode: 27,
        nativeVirtualKeyCode: 27,
      });
      await cdp.evaluate(`new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`);

      const modalCloseState = await cdp.evaluate<{
        dialogStillOpen: boolean;
        focusRestoredToGuideOnEscape: boolean;
        hudAriaHiddenAfterClose: string | null;
      }>(`(() => {
        const guideBtn = document.querySelector('button[aria-label="How to Play & Controls"]');
        return {
          dialogStillOpen: Boolean(document.querySelector('[role="dialog"][aria-modal="true"]')),
          focusRestoredToGuideOnEscape: document.activeElement === guideBtn,
          hudAriaHiddenAfterClose: document.querySelector('.hud-layer')?.getAttribute('aria-hidden') ?? null,
        };
      })()`);

      expect(enterPressedAttr).toBe('true');
      expect(polariscopeAfterEnter).toBe(true);
      expect(polariscopeAfterSpace).toBe(false);
      expect(modalOpenState.dialogOpen).toBe(true);
      expect(modalOpenState.hudAriaHiddenWhileOpen).toBe('true');
      expect(modalOpenState.canvasAriaHiddenWhileOpen).toBe('true');
      expect(modalOpenState.initialFocusLabel).toContain('Close Guide');
      expect(tabWrappedToFirst).toBe(true);
      expect(shiftTabWrappedToLast).toBe(true);
      expect(modalCloseState.dialogStillOpen).toBe(false);
      expect(modalCloseState.focusRestoredToGuideOnEscape).toBe(true);
      expect(modalCloseState.hudAriaHiddenAfterClose).toBeNull();
    } finally {
      cdp?.close();
      if (chromeProc && chromeProc.exitCode === null) {
        const exitPromise = new Promise<void>((resolve) => {
          chromeProc!.once('exit', () => resolve());
          setTimeout(resolve, 1500);
        });
        chromeProc.kill('SIGKILL');
        await exitPromise;
      }
      if (viteServer) {
        await viteServer.close();
      }
      try {
        fsMod.rmSync(userDataDir, { recursive: true, force: true });
      } catch {
        // Ignore residual temp cleanup race from Chromium sub-processes
      }
    }
  }, 35000);

  it('6. optimizes continuous extraction azimuth & adversarial eccentric placement and survives a 30-turn headless AI-vs-AI match', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    try {
      // A. Continuous extraction azimuth check under asymmetric neighbor crowding (one neighbor removed, remaining neighbor encroaching)
      const targetBlk = engine.physics.blocks.find(
        (b) => b.assignedLayer === 8 && b.assignedSlot === 1 && b.isLegalTarget
      )!;
      const sideNeighbors = engine.physics.blocks.filter(
        (b) => b.assignedLayer === 8 && b.id !== targetBlk.id
      );
      const crowdingNeighbor = sideNeighbors[0];
      const openSideNeighbor = sideNeighbors[1];
      const axisZ = quatRotateVec3(targetBlk.rotation, [0, 0, 1]);
      const savedPos: Vec3 = [...crowdingNeighbor.position];
      const savedExtracted = openSideNeighbor.extracted;

      // Simulate a 2-block course where one outer prism was already extracted and the remaining neighbor crowds targetBlk along -Z/+Z
      openSideNeighbor.extracted = true;
      const signToNeighbor =
        (crowdingNeighbor.position[0] - targetBlk.position[0]) * axisZ[0] +
          (crowdingNeighbor.position[2] - targetBlk.position[2]) * axisZ[2] >=
        0
          ? 1
          : -1;
      crowdingNeighbor.position = [
        targetBlk.position[0] + axisZ[0] * signToNeighbor * (2 * BLOCK_HALF_Z - 0.004),
        crowdingNeighbor.position[1],
        targetBlk.position[2] + axisZ[2] * signToNeighbor * (2 * BLOCK_HALF_Z - 0.004),
      ];

      const apprenticeExtract = engine.computeOptimalExtractionPlan(targetBlk.id, 'apprentice')!;
      const masterExtract = engine.computeOptimalExtractionPlan(targetBlk.id, 'master')!;
      crowdingNeighbor.position = savedPos;
      openSideNeighbor.extracted = savedExtracted;

      expect(apprenticeExtract.azimuthOffsetDeg).toBe(0);
      expect(Math.abs(masterExtract.azimuthOffsetDeg)).toBeGreaterThan(1.0);
      expect(masterExtract.minNeighborClearanceDm).toBeGreaterThan(0.0);

      // B. Adversarial top-course placement check: Master Glazier drives global CoM eccentricity > Apprentice
      const excludeCandidate = engine.physics.blocks.find(
        (b) => b.assignedLayer === 6 && b.assignedSlot === 0 && b.isLegalTarget
      )!;
      const apprenticePlace = engine.computeAdversarialTopPlacement(
        excludeCandidate.id,
        'apprentice'
      );
      const masterPlace = engine.computeAdversarialTopPlacement(excludeCandidate.id, 'master');

      const appEccCm = Math.hypot(
        apprenticePlace.predictedComXZCm[0],
        apprenticePlace.predictedComXZCm[1]
      );
      const masEccCm = Math.hypot(
        masterPlace.predictedComXZCm[0],
        masterPlace.predictedComXZCm[1]
      );
      expect(masEccCm).toBeGreaterThan(appEccCm + 0.05);
      expect(Math.hypot(masterPlace.localVU[0], masterPlace.localVU[1])).toBeGreaterThan(0.22);
      expect(masterPlace.predictedMinCourseMarginDm).toBeGreaterThan(0.05);

      // C. Full 30-turn headless AI-vs-AI benchmark (Apprentice vs Master Glazier -> 18L up to 28L!)
      engine.resetGame('party', 2);
      engine.setAiVsAiMode(true, {
        fastCadence: true,
        p0Difficulty: 'apprentice',
        p1Difficulty: 'master',
      });

      let peakComEccCm = 0;
      const maxSteps = 22000; // ~366 sim seconds upper cap at 60 Hz (~12 s/turn max)
      let stepsRun = 0;

      while (
        stepsRun < maxSteps &&
        engine.physics.blocksStacked < 30 &&
        engine.physics.turnPhase !== 'COLLAPSED'
      ) {
        engine.stepHeadless(1 / 60);
        stepsRun++;
        const com = engine.physics.centerOfMassXZCm;
        const ecc = Math.hypot(com[0], com[1]);
        if (ecc > peakComEccCm) {
          peakComEccCm = ecc;
        }
      }

      expect(engine.physics.turnPhase).not.toBe('COLLAPSED');
      expect(engine.physics.shatteredCount).toBe(0);
      expect(engine.physics.blocksStacked).toBeGreaterThanOrEqual(30);
      expect(engine.physics.currentLayers).toBeGreaterThanOrEqual(28);
      expect(peakComEccCm).toBeGreaterThan(0.14);
    } finally {
      engine.dispose();
    }
  }, 120000);
});

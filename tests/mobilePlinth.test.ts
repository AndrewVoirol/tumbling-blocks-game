import { createServer, type ViteDevServer } from 'vite';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { GameEngine } from '../src/engine/GameEngine';
import { quatRotateVec3, type Vec3 } from '../src/math/vecMat';
import { BLOCK_HALF_X, BLOCK_HALF_Z } from '../src/physics/constants';

interface MinimalChildProcess {
  exitCode: number | null;
  stderr: {
    on(event: 'data', cb: (chunk: Uint8Array) => void): void;
  } | null;
  once(event: 'exit', cb: (code: number | null) => void): void;
  once(event: 'error', cb: (err: unknown) => void): void;
  kill(signal?: string): boolean;
}

interface MinimalChildProcessModule {
  spawn(
    cmd: string,
    args: string[],
    opts: { stdio: Array<'ignore' | 'pipe'>; env?: Record<string, string | undefined> }
  ): MinimalChildProcess;
}

interface MinimalFsModule {
  existsSync(filePath: string): boolean;
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

function findFrontBlockScreenPt(
  engine: GameEngine,
  rect: { left: number; top: number; width: number; height: number },
  predicate: (blk: GameEngine['physics']['blocks'][number]) => boolean
): { blockId: number; screenPt: [number, number] } {
  const localOffsets: Vec3[] = [
    [BLOCK_HALF_X * 0.72, 0, 0],
    [-BLOCK_HALF_X * 0.72, 0, 0],
    [0, 0, BLOCK_HALF_Z * 0.72],
    [0, 0, -BLOCK_HALF_Z * 0.72],
    [0, 0.06, 0],
    [0, 0, 0],
  ];
  for (const blk of engine.physics.blocks) {
    if (!predicate(blk)) continue;
    for (const localOff of localOffsets) {
      const off = quatRotateVec3(blk.rotation, localOff);
      const worldPt: Vec3 = [
        blk.position[0] + off[0],
        blk.position[1] + off[1],
        blk.position[2] + off[2],
      ];
      const [sx, sy] = worldToScreenPx(engine, worldPt, rect.width, rect.height);
      if (sx < 10 || sx > rect.width - 10 || sy < 10 || sy > rect.height - 10) continue;
      const { origin, dir } = engine.camera.screenToRay(sx, sy, rect);
      const hit = raycastIntactBlocksForTest(engine, origin, dir);
      if (hit && hit.blockId === blk.id) {
        return { blockId: blk.id, screenPt: [sx, sy] };
      }
    }
  }
  throw new Error(`No visible matching block found in ${rect.width}x${rect.height}`);
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

describe('Cathedral Plinth Mobile HUD & E1–E8 Contracts', () => {
  it('1. [E1/E3 Headless Engine] verifies viewportBottomInset camera framing, < 1e-4 screenToRay accuracy, and touchInspect telemetry without locked-top audio', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    try {
      const rect = { left: 0, top: 0, width: 390, height: 844 };
      engine.setViewportBottomInset(0.30);
      engine.camera.update(0.8, rect.width / rect.height);
      expect(engine.camera.getViewportBottomInset()).toBeGreaterThan(0.18);
      expect(engine.camera.getViewportBottomInset()).toBeLessThanOrEqual(0.4);
      expect(engine.camera.projMatrix[9]).toBeCloseTo(-engine.camera.getViewportBottomInset(), 5);

      // Verify screenToRay unprojection residual (< 1e-4 NDC) at center and near corners
      const probePixels: Array<[number, number]> = [
        [rect.width * 0.5, rect.height * 0.5],
        [rect.width * 0.12, rect.height * 0.16],
        [rect.width * 0.88, rect.height * 0.82],
      ];
      const m = engine.camera.viewProjMatrix;
      for (const [px, py] of probePixels) {
        const { origin, dir } = engine.camera.screenToRay(px, py, rect);
        const targetPt: Vec3 = [origin[0] + dir[0] * 14.0, origin[1] + dir[1] * 14.0, origin[2] + dir[2] * 14.0];
        const cx = m[0] * targetPt[0] + m[4] * targetPt[1] + m[8] * targetPt[2] + m[12];
        const cy = m[1] * targetPt[0] + m[5] * targetPt[1] + m[9] * targetPt[2] + m[13];
        const cw = m[3] * targetPt[0] + m[7] * targetPt[1] + m[11] * targetPt[2] + m[15];
        const ndcX = cx / cw;
        const ndcY = cy / cw;
        const expectedNdcX = (px / rect.width) * 2 - 1;
        const expectedNdcY = 1 - (py / rect.height) * 2;
        expect(Math.abs(ndcX - expectedNdcX)).toBeLessThan(1e-4);
        expect(Math.abs(ndcY - expectedNdcY)).toBeLessThan(1e-4);
      }

      // Resetting the tower preserves viewportBottomInset
      engine.resetGame('solo', 2);
      engine.camera.update(0.25, rect.width / rect.height);
      expect(engine.camera.getViewportBottomInset()).toBeGreaterThan(0.18);

      // Spy on tap-test resonance audio calls
      let tapResonanceCalls = 0;
      const origPlayTapTest = engine.audio.playTapTestResonance.bind(engine.audio);
      engine.audio.playTapTestResonance = ((...args: Parameters<typeof origPlayTapTest>) => {
        tapResonanceCalls++;
        return origPlayTapTest(...args);
      }) as typeof engine.audio.playTapTestResonance;

      // Short touch tap on a loose legal block emits touchInspect telemetry & 1 resonance tap
      const loosePick = findFrontBlockScreenPt(
        engine,
        rect,
        (b) => !b.shattered && b.isLegalTarget && b.isLooseSlot && b.assignedLayer >= 4 && b.assignedLayer <= 14
      );
      engine.handlePointerDown(loosePick.screenPt[0], loosePick.screenPt[1], 101, 0, false, rect, 'touch', 100);
      engine.handlePointerMove(loosePick.screenPt[0] + 2, loosePick.screenPt[1] + 1, 101, rect, 'touch', 130);
      engine.handlePointerUp(loosePick.screenPt[0] + 2, loosePick.screenPt[1] + 1, 101, rect, 'touch', 150);

      const looseSnap = engine.getTelemetrySnapshot();
      expect(looseSnap.touchInspect).not.toBeNull();
      expect(looseSnap.touchInspect?.blockId).toBe(loosePick.blockId);
      expect(looseSnap.touchInspect?.status).toBe('loose');
      expect(tapResonanceCalls).toBe(1);

      // Short touch tap on a locked top-row block emits 'locked_top' with ZERO additional resonance calls
      const topPick = findFrontBlockScreenPt(
        engine,
        rect,
        (b) => !b.shattered && !b.isLegalTarget && b.assignedLayer >= 16
      );
      const seqAfterLoose = looseSnap.touchInspect?.seq ?? 0;
      engine.handlePointerDown(topPick.screenPt[0], topPick.screenPt[1], 102, 0, false, rect, 'touch', 300);
      engine.handlePointerMove(topPick.screenPt[0] + 2, topPick.screenPt[1] + 1, 102, rect, 'touch', 325);
      engine.handlePointerUp(topPick.screenPt[0] + 2, topPick.screenPt[1] + 1, 102, rect, 'touch', 345);

      const topSnap = engine.getTelemetrySnapshot();
      expect(topSnap.touchInspect).not.toBeNull();
      expect(topSnap.touchInspect?.seq).toBeGreaterThan(seqAfterLoose);
      expect(topSnap.touchInspect?.status).toBe('locked_top');
      expect(tapResonanceCalls).toBe(1);
    } finally {
      engine.dispose();
    }
  }, 60000);

  describe('Headless Chromium Cathedral Plinth E1–E8 Suite', () => {
    let fsMod: MinimalFsModule;
    let pathMod: MinimalPathModule;
    let cwd: string;
    let viteServer: ViteDevServer | null = null;
    let chromeProc: MinimalChildProcess | null = null;
    let cdp: SimpleCdpPageClient | null = null;
    let userDataDir = '';

    const cleanupBrowserSession = async () => {
      try {
        cdp?.close();
      } catch {
        // Ignore close error
      }
      cdp = null;
      if (chromeProc && chromeProc.exitCode === null) {
        try {
          chromeProc.kill('SIGKILL');
        } catch {
          // Ignore kill error
        }
      }
      chromeProc = null;
      if (viteServer) {
        await viteServer.close();
        viteServer = null;
      }
      if (userDataDir && fsMod) {
        try {
          fsMod.rmSync(userDataDir, { recursive: true, force: true });
        } catch {
          // Ignore cleanup error
        }
      }
    };

    beforeAll(async () => {
      const cpModName = 'node:child_process';
      const fsModName = 'node:fs';
      const httpModName = 'node:http';
      const osModName = 'node:os';
      const pathModName = 'node:path';
      const cpMod = (await import(/* @vite-ignore */ cpModName)) as unknown as MinimalChildProcessModule;
      fsMod = (await import(/* @vite-ignore */ fsModName)) as unknown as MinimalFsModule;
      const httpMod = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
      const osMod = (await import(/* @vite-ignore */ osModName)) as unknown as MinimalOsModule;
      pathMod = (await import(/* @vite-ignore */ pathModName)) as unknown as MinimalPathModule;
      cwd = getNodeCwd();
      userDataDir = fsMod.mkdtempSync(pathMod.join(osMod.tmpdir(), 'vitrail-plinth-cdp-'));

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

        const procObj = (
          globalThis as unknown as {
            process?: { env?: Record<string, string | undefined>; platform?: string };
          }
        ).process;
        const envChrome = procObj?.env?.CHROME_PATH;
        const chromeCandidates = [
          ...(envChrome ? [envChrome] : []),
          '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
          '/Applications/Chromium.app/Contents/MacOS/Chromium',
          '/usr/bin/google-chrome',
          '/usr/bin/google-chrome-stable',
          '/usr/bin/chromium',
          '/usr/bin/chromium-browser',
        ];
        const chromeBin =
          chromeCandidates.find((candidate) => fsMod.existsSync(candidate)) ??
          '/usr/bin/google-chrome';

        const chromeEnv =
          procObj?.platform === 'linux'
            ? { ...(procObj.env ?? {}), DBUS_SESSION_BUS_ADDRESS: '/dev/null' }
            : procObj?.env;

        chromeProc = cpMod.spawn(
          chromeBin,
          [
            '--headless=new',
            '--no-sandbox',
            '--disable-gpu-sandbox',
            '--disable-dev-shm-usage',
            '--remote-debugging-address=127.0.0.1',
            '--password-store=basic',
            '--use-mock-keychain',
            '--disable-dbus',
            '--disable-background-networking',
            '--disable-extensions',
            '--disable-sync',
            '--disable-component-update',
            '--disable-breakpad',
            `--user-data-dir=${userDataDir}`,
            '--remote-debugging-port=0',
            'about:blank',
          ],
          { stdio: ['ignore', 'ignore', 'pipe'], env: chromeEnv }
        );

        const decoder = new TextDecoder('utf-8');
        const activePortFile = pathMod.join(userDataDir, 'DevToolsActivePort');
        const cdpPort = await new Promise<number>((resolve, reject) => {
          let settled = false;
          let stderrBuf = '';
          let pollTimer: ReturnType<typeof setInterval> | null = null;
          const finish = (cb: () => void) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeout);
            if (pollTimer !== null) clearInterval(pollTimer);
            cb();
          };
          const timeout = setTimeout(
            () =>
              finish(() =>
                reject(new Error(`Timed out waiting for Chrome DevTools port: ${stderrBuf}`))
              ),
            30000
          );
          pollTimer = setInterval(() => {
            try {
              if (!fsMod.existsSync(activePortFile)) return;
              const raw = fsMod.readFileSync(activePortFile, 'utf8');
              const firstLine = Number.parseInt(raw.split(/\r?\n/)[0]?.trim() ?? '', 10);
              if (Number.isInteger(firstLine) && firstLine > 0) {
                finish(() => resolve(firstLine));
              }
            } catch {
              // Still writing
            }
          }, 200);
          chromeProc!.stderr!.on('data', (chunk: Uint8Array) => {
            stderrBuf += decoder.decode(chunk, { stream: true });
            const match = stderrBuf.match(/DevTools listening on ws:\/\/[^/:]+:(\d+)\//);
            if (match) finish(() => resolve(Number(match[1])));
          });
          chromeProc!.once('exit', (code: number | null) => {
            finish(() => reject(new Error(`Chrome exited prematurely (${code}): ${stderrBuf}`)));
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
          width: 390,
          height: 844,
          deviceScaleFactor: 3,
          mobile: true,
        });
        await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
          source: WEBGPU_HEADLESS_SHIM_SCRIPT,
        });
        await cdp.send('Page.navigate', {
          url: `http://127.0.0.1:${vitePort}/?worker=0`,
        });

        await cdp.evaluate(`new Promise((resolve, reject) => {
          const deadline = Date.now() + 25000;
          const poll = () => {
            const eng = window.__VITRAIL_ENGINE__;
            const errModal = document.querySelector('[role="alertdialog"]');
            if (errModal) {
              reject(new Error('Unexpected gpuError modal: ' + errModal.textContent));
              return;
            }
            if (eng && eng.physics && document.querySelector('.mobile-hud-root')) {
              resolve(true);
              return;
            }
            if (Date.now() > deadline) {
              reject(new Error('Timed out waiting for CathedralPlinthHud in headless Chrome'));
              return;
            }
            setTimeout(poll, 40);
          };
          poll();
        })`);
      } catch (err) {
        await cleanupBrowserSession();
        throw err;
      }
    }, 60000);

    afterAll(async () => {
      await cleanupBrowserSession();
    });

    const setViewport = async (width: number, height: number, dpr = 2, mobile = true) => {
      await cdp!.send('Emulation.setDeviceMetricsOverride', {
        width,
        height,
        deviceScaleFactor: dpr,
        mobile,
      });
      await cdp!.evaluate(`new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`);
    };

    const resetClosedMobileState = async () => {
      await setViewport(390, 844, 3, true);
      await cdp!.evaluate(`new Promise((resolve) => {
        const eng = window.__VITRAIL_ENGINE__;
        eng.setModalOpen(false);
        eng.setOnlineLobbyOpen(false);
        eng.resetGame('solo', 2);
        const closeFull = document.querySelector('button[aria-label="Close studio sheet"]');
        closeFull?.click();
        requestAnimationFrame(() => {
          const closeHalf = document.querySelector('button[aria-label="Collapse studio tray"]');
          closeHalf?.click();
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        });
      })`);
    };

    it('2. [Zero-Pill & Viewport Meta E8] verifies 0 occurrences of 999px, computed borderRadius === 0px across Closed/Detent 1/Detent 2, and clean viewport meta', async () => {
      const plinthCss = fsMod.readFileSync(pathMod.resolve(cwd, 'src/ui/cathedralPlinth.css'), 'utf8');
      expect(plinthCss.match(/999px/g) ?? []).toHaveLength(0);

      const indexHtml = fsMod.readFileSync(pathMod.resolve(cwd, 'index.html'), 'utf8');
      const viewportMatch = indexHtml.match(/<meta\s+name="viewport"[^>]*>/i);
      expect(viewportMatch).not.toBeNull();
      const viewportTag = viewportMatch![0];
      expect(viewportTag).toContain('viewport-fit=cover');
      expect(viewportTag).not.toContain('user-scalable=no');
      expect(viewportTag).not.toContain('maximum-scale=1.0');

      await resetClosedMobileState();

      const auditBorderRadius = async (label: string) => {
        const offenders = await cdp!.evaluate<Array<{ selector: string; br: string }>>(`(() => {
          const isVisible = (el) => {
            const st = getComputedStyle(el);
            if (st.display === 'none' || st.visibility === 'hidden' || Number(st.opacity) === 0) return false;
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          };
          const els = Array.from(document.querySelectorAll(
            '.mobile-hud-root, .plinth-transom-frame, .plinth-sill-assembly, .plinth-frame, .plinth-swipe-cap, .plinth-half-tray, .plinth-half-tabs, .plinth-full-modal, .mobile-hud-root button, .mobile-hud-root input'
          )).filter(isVisible);
          const bad = [];
          for (const el of els) {
            const st = getComputedStyle(el);
            const corners = [
              st.borderTopLeftRadius,
              st.borderTopRightRadius,
              st.borderBottomRightRadius,
              st.borderBottomLeftRadius,
            ];
            if (st.borderRadius !== '0px' || corners.some((c) => c !== '0px')) {
              bad.push({ selector: el.tagName + '.' + Array.from(el.classList).join('.'), br: st.borderRadius });
            }
          }
          return bad;
        })()`);
        expect(offenders, `${label} rounded corners`).toEqual([]);
      };

      // Closed sill
      await auditBorderRadius('Closed Sill @ 390x844');

      // Detent 1 (Studio tab + Crystal tab)
      await cdp!.evaluate(`new Promise((resolve) => {
        document.querySelector('button.plinth-sill-mode')?.click();
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      })`);
      await auditBorderRadius('Detent 1 Studio @ 390x844');

      await cdp!.evaluate(`new Promise((resolve) => {
        const crystalTab = Array.from(document.querySelectorAll('button.plinth-half-tab-btn'))
          .find((el) => el.textContent?.includes('Crystal'));
        crystalTab?.click();
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      })`);
      await auditBorderRadius('Detent 1 Crystal @ 390x844');

      // Detent 2 (full modal)
      await cdp!.evaluate(`new Promise((resolve) => {
        document.querySelector('button[aria-label="Expand full studio sheet"]')?.click();
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      })`);
      await auditBorderRadius('Detent 2 Full @ 390x844');
    });

    it('3. [Extreme Viewport Matrix & Dynamic matchMedia] checks 320x568..844x390 (>=44px, 0 overflow), 5-button EXTRACTED at 320x568, and single-HUD resize switch', async () => {
      await resetClosedMobileState();

      const matrix = [
        { width: 320, height: 568, dpr: 2 },
        { width: 375, height: 667, dpr: 2 },
        { width: 390, height: 844, dpr: 3 },
        { width: 430, height: 932, dpr: 3 },
        { width: 667, height: 375, dpr: 2 },
        { width: 844, height: 390, dpr: 3 },
      ];

      for (const vp of matrix) {
        await setViewport(vp.width, vp.height, vp.dpr, true);
        const audit = await cdp!.evaluate<{
          hasMobile: boolean;
          hasDesktop: boolean;
          overflowX: boolean;
          roomInputs: number;
          undersized: Array<{ label: string; w: number; h: number }>;
          offscreen: Array<{ label: string; l: number; t: number; r: number; b: number }>;
          zoneOverlaps: number;
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
            const label = (el.getAttribute('aria-label') || el.textContent || el.tagName).trim().slice(0, 32);
            if (r.width < 43.9 || r.height < 43.9) {
              undersized.push({ label, w: +r.width.toFixed(1), h: +r.height.toFixed(1) });
            }
            if (r.left < -0.5 || r.top < -0.5 || r.right > window.innerWidth + 0.5 || r.bottom > window.innerHeight + 0.5) {
              offscreen.push({ label, l: +r.left.toFixed(1), t: +r.top.toFixed(1), r: +r.right.toFixed(1), b: +r.bottom.toFixed(1) });
            }
          }
          const a = document.querySelector('.plinth-transom')?.getBoundingClientRect();
          const b = document.querySelector('.plinth-sill-assembly')?.getBoundingClientRect();
          let zoneOverlaps = 0;
          if (a && b) {
            const ix = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
            const iy = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
            zoneOverlaps = +(ix * iy).toFixed(2);
          }
          return {
            hasMobile: Boolean(document.querySelector('.mobile-hud-root')),
            hasDesktop: Boolean(document.querySelector('.hud-layer')),
            overflowX: document.documentElement.scrollWidth > window.innerWidth,
            roomInputs: document.querySelectorAll('#online-room-code-input').length,
            undersized,
            offscreen,
            zoneOverlaps,
          };
        })()`);

        expect(audit.hasMobile, `${vp.width}x${vp.height} mobile`).toBe(true);
        expect(audit.hasDesktop, `${vp.width}x${vp.height} desktop`).toBe(false);
        expect(audit.overflowX, `${vp.width}x${vp.height} horizontal overflow`).toBe(false);
        expect(audit.roomInputs).toBeLessThanOrEqual(1);
        expect(audit.undersized, `${vp.width}x${vp.height} undersized`).toEqual([]);
        expect(audit.offscreen, `${vp.width}x${vp.height} offscreen`).toEqual([]);
        expect(audit.zoneOverlaps, `${vp.width}x${vp.height} transom/sill overlap`).toBe(0);
      }

      // 320x568 5-button EXTRACTED state with carriedYawDeg !== 0
      await setViewport(320, 568, 2, true);
      const extracted320 = await cdp!.evaluate<{
        turnPhase: string;
        carriedYawDeg: number;
        btnCount: number;
        undersized: number;
        offscreen: number;
        overlaps: number;
        overflowX: boolean;
      }>(`new Promise((resolve) => {
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
        eng.handleRotateCarried(Math.PI / 12);
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const btns = Array.from(document.querySelectorAll('.plinth-place-keys button'));
          const rects = btns.map((b) => b.getBoundingClientRect());
          let undersized = 0;
          let offscreen = 0;
          let overlaps = 0;
          for (let i = 0; i < rects.length; i++) {
            const r = rects[i];
            if (r.width < 43.9 || r.height < 43.9) undersized++;
            if (r.left < -0.5 || r.top < -0.5 || r.right > window.innerWidth + 0.5 || r.bottom > window.innerHeight + 0.5) offscreen++;
            for (let j = i + 1; j < rects.length; j++) {
              const b = rects[j];
              const ix = Math.max(0, Math.min(r.right, b.right) - Math.max(r.left, b.left));
              const iy = Math.max(0, Math.min(r.bottom, b.bottom) - Math.max(r.top, b.top));
              if (ix * iy > 0.5) overlaps++;
            }
          }
          resolve({
            turnPhase: eng.physics.turnPhase,
            carriedYawDeg: eng.getTelemetrySnapshot().carriedYawDeg,
            btnCount: btns.length,
            undersized,
            offscreen,
            overlaps,
            overflowX: document.documentElement.scrollWidth > window.innerWidth,
          });
        }));
      })`);
      expect(extracted320.turnPhase).toBe('EXTRACTED');
      expect(Math.abs(extracted320.carriedYawDeg)).toBeGreaterThan(1);
      expect(extracted320.btnCount).toBe(5);
      expect(extracted320.undersized).toBe(0);
      expect(extracted320.offscreen).toBe(0);
      expect(extracted320.overlaps).toBe(0);
      expect(extracted320.overflowX).toBe(false);

      // Dynamic matchMedia switch: 390x844 -> 844x390 -> 1024x1366 -> 390x844 with #online-room-code-input uniqueness
      await resetClosedMobileState();
      const resizeSequence = [
        { width: 390, height: 844, dpr: 3, mobile: true, expectMobile: true },
        { width: 844, height: 390, dpr: 3, mobile: true, expectMobile: true },
        { width: 1024, height: 1366, dpr: 2, mobile: false, expectMobile: false },
        { width: 390, height: 844, dpr: 3, mobile: true, expectMobile: true },
      ];
      for (const step of resizeSequence) {
        await setViewport(step.width, step.height, step.dpr, step.mobile);
        if (!step.expectMobile) {
          // Also open Desktop Online drawer so #online-room-code-input mounts before switching back to mobile
          await cdp!.evaluate(`new Promise((resolve) => {
            const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
            tabs.find((el) => el.textContent?.includes('Online'))?.click();
            requestAnimationFrame(() => requestAnimationFrame(resolve));
          })`);
        }
        const state = await cdp!.evaluate<{
          hasMobile: boolean;
          hasDesktop: boolean;
          roomInputs: number;
        }>(`(() => ({
          hasMobile: Boolean(document.querySelector('.mobile-hud-root')),
          hasDesktop: Boolean(document.querySelector('.hud-layer')),
          roomInputs: document.querySelectorAll('#online-room-code-input').length,
        }))()`);
        expect(state.hasMobile).toBe(step.expectMobile);
        expect(state.hasDesktop).toBe(!step.expectMobile);
        expect(state.roomInputs).toBeLessThanOrEqual(1);
      }

      // Open Detent 2 while onlineDrawerOpen is still true, verify #online-room-code-input uniqueness, then switch back to Solo to close drawer cleanly
      const mobileOnlineRecover = await cdp!.evaluate<{
        roomInputsInFull: number;
        modeTextAfterSolo: string;
      }>(`new Promise((resolve) => {
        document.querySelector('button.plinth-sill-mode')?.click();
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const roomInputsInFull = document.querySelectorAll('#online-room-code-input').length;
          const soloTab = Array.from(document.querySelectorAll('.plinth-full-modal button.mode-course-btn'))
            .find((el) => el.textContent?.includes('Solo'));
          soloTab?.click();
          requestAnimationFrame(() => requestAnimationFrame(() => {
            document.querySelector('button[aria-label="Close studio sheet"]')?.click();
            requestAnimationFrame(() => requestAnimationFrame(() => {
              resolve({
                roomInputsInFull,
                modeTextAfterSolo: document.querySelector('button.plinth-sill-mode')?.textContent?.trim() ?? '',
              });
            }));
          }));
        }));
      })`);
      expect(mobileOnlineRecover.roomInputsInFull).toBe(1);
      expect(mobileOnlineRecover.modeTextAfterSolo).toContain('Solo');
    });

    it('4. [Detent 1 Framing, Auto-Tuck, E5 Swipe, & E3 Inspect Strip] verifies bottom inset, rebuild persistence, phase tuck, swipe threshold, and inspect strip', async () => {
      await resetClosedMobileState();

      // Open Detent 1 via sill mode button and wait for ResizeObserver + rAF settle
      const detent1Check = await cdp!.evaluate<{
        detent: string | null;
        bottomInset: number;
        modalOpen: boolean;
        rayResidual: number;
      }>(`new Promise((resolve) => {
        document.querySelector('button.plinth-sill-mode')?.click();
        const deadline = Date.now() + 1200;
        const poll = () => {
          const eng = window.__VITRAIL_ENGINE__;
          const detent = document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null;
          const inset = eng.camera.getViewportBottomInset();
          if (detent === 'half' && inset > 0.18) {
            const rect = { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
            const px = rect.width * 0.31;
            const py = rect.height * 0.27;
            const { origin, dir } = eng.camera.screenToRay(px, py, rect);
            const pt = [origin[0] + dir[0] * 12, origin[1] + dir[1] * 12, origin[2] + dir[2] * 12];
            const m = eng.camera.viewProjMatrix;
            const cx = m[0] * pt[0] + m[4] * pt[1] + m[8] * pt[2] + m[12];
            const cy = m[1] * pt[0] + m[5] * pt[1] + m[9] * pt[2] + m[13];
            const cw = m[3] * pt[0] + m[7] * pt[1] + m[11] * pt[2] + m[15];
            const errX = Math.abs(cx / cw - ((px / rect.width) * 2 - 1));
            const errY = Math.abs(cy / cw - (1 - (py / rect.height) * 2));
            resolve({
              detent,
              bottomInset: inset,
              modalOpen: Boolean(eng.modalOpen),
              rayResidual: Math.max(errX, errY),
            });
            return;
          }
          if (Date.now() > deadline) {
            resolve({ detent, bottomInset: inset, modalOpen: Boolean(eng.modalOpen), rayResidual: 999 });
            return;
          }
          requestAnimationFrame(poll);
        };
        requestAnimationFrame(poll);
      })`);
      expect(detent1Check.detent).toBe('half');
      expect(detent1Check.bottomInset).toBeGreaterThan(0.18);
      expect(detent1Check.modalOpen).toBe(false);
      expect(detent1Check.rayResidual).toBeLessThan(1e-4);

      // Rebuild preserves Detent 1 open & bottom inset active
      const afterRebuild = await cdp!.evaluate<{ detent: string | null; bottomInset: number }>(`new Promise((resolve) => {
        document.querySelector('button[aria-label="Rebuild 54-Prism Stained-Glass Tower"]')?.click();
        requestAnimationFrame(() => requestAnimationFrame(() => {
          resolve({
            detent: document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null,
            bottomInset: window.__VITRAIL_ENGINE__.camera.getViewportBottomInset(),
          });
        }));
      })`);
      expect(afterRebuild.detent).toBe('half');
      expect(afterRebuild.bottomInset).toBeGreaterThan(0.18);

      // Close tray and test E5 horizontal vs vertical swipe disambiguation via real CDP touch events on .plinth-frame
      await cdp!.evaluate(`new Promise((resolve) => {
        document.querySelector('button[aria-label="Collapse studio tray"]')?.click();
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      })`);

      const dispatchFrameTouchSwipe = async (dx: number, dy: number): Promise<string | null> => {
        const pt = await cdp!.evaluate<{ x: number; y: number }>(`(() => {
          const r = document.querySelector('.plinth-frame').getBoundingClientRect();
          return { x: r.left + r.width * 0.5, y: r.top + Math.min(18, r.height * 0.25) };
        })()`);
        await cdp!.send('Input.dispatchTouchEvent', {
          type: 'touchStart',
          touchPoints: [{ x: pt.x, y: pt.y, id: 1 }],
        });
        const steps = 4;
        for (let s = 1; s <= steps; s++) {
          await cdp!.send('Input.dispatchTouchEvent', {
            type: 'touchMove',
            touchPoints: [{ x: pt.x + (dx * s) / steps, y: pt.y + (dy * s) / steps, id: 1 }],
          });
        }
        await cdp!.send('Input.dispatchTouchEvent', {
          type: 'touchEnd',
          touchPoints: [],
        });
        return cdp!.evaluate<string | null>(
          `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null))))`
        );
      };

      expect(await dispatchFrameTouchSwipe(45, 6)).toBe('closed');
      expect(await dispatchFrameTouchSwipe(0, -45)).toBe('half');
      expect(await dispatchFrameTouchSwipe(-45, 4)).toBe('half');
      expect(await dispatchFrameTouchSwipe(0, 45)).toBe('closed');

      // E2 Auto-Tuck on PULLING, EXTRACTED, and COLLAPSED
      const autoTuckPull = await cdp!.evaluate<string | null>(`new Promise((resolve) => {
        const waitDetentClosed = () => {
          const deadline = Date.now() + 500;
          const poll = () => {
            const det = document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null;
            if (det === 'closed' || Date.now() > deadline) {
              resolve(det);
              return;
            }
            requestAnimationFrame(poll);
          };
          requestAnimationFrame(poll);
        };
        document.querySelector('button.plinth-sill-mode')?.click();
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') !== 'half') {
            resolve('not_half_before_pull');
            return;
          }
          const eng = window.__VITRAIL_ENGINE__;
          const blk = eng.physics.blocks.find((b) => b.isLegalTarget && b.isLooseSlot && b.assignedLayer >= 4 && b.assignedLayer <= 14);
          const plan = eng.computeOptimalExtractionPlan(blk.id, 'master');
          eng.physics.startGrabBlock(blk.id, plan.localGrab, plan.startWorld);
          eng.physics.updateGrabTarget([
            plan.startWorld[0] + plan.pullDirXZ[0] * 0.25,
            plan.startWorld[1],
            plan.startWorld[2] + plan.pullDirXZ[1] * 0.25,
          ]);
          eng.physics.step(1 / 60);
          waitDetentClosed();
        }));
      })`);
      expect(autoTuckPull).toBe('closed');

      const autoTuckExtracted = await cdp!.evaluate<string | null>(`new Promise((resolve) => {
        const eng = window.__VITRAIL_ENGINE__;
        eng.resetGame('solo', 2);
        requestAnimationFrame(() => {
          document.querySelector('button.plinth-sill-mode')?.click();
          requestAnimationFrame(() => requestAnimationFrame(() => {
            if (document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') !== 'half') {
              resolve('not_half_before_extracted');
              return;
            }
            const blk = eng.physics.blocks.find((b) => b.isLegalTarget && b.isLooseSlot && b.assignedLayer >= 4 && b.assignedLayer <= 14);
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
            for (let i = 0; i < 12; i++) eng.physics.step(1 / 60);
            const deadline = Date.now() + 500;
            const poll = () => {
              const det = document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null;
              if (det === 'closed' || Date.now() > deadline) {
                resolve(eng.physics.turnPhase + ':' + det);
                return;
              }
              requestAnimationFrame(poll);
            };
            requestAnimationFrame(poll);
          }));
        });
      })`);
      expect(autoTuckExtracted).toBe('EXTRACTED:closed');

      const autoTuckCollapsed = await cdp!.evaluate<string | null>(`new Promise((resolve) => {
        const eng = window.__VITRAIL_ENGINE__;
        eng.resetGame('solo', 2);
        const waitSillAndOpen = () => {
          const modeBtn = document.querySelector('button.plinth-sill-mode');
          if (!modeBtn) {
            requestAnimationFrame(waitSillAndOpen);
            return;
          }
          modeBtn.click();
          requestAnimationFrame(() => requestAnimationFrame(() => {
            if (document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') !== 'half') {
              resolve('not_half_before_collapse:' + (document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? ''));
              return;
            }
            // Shatter all 3 blocks on base layer 0 so tower naturally collapses to turnPhase === 'COLLAPSED'
            for (const b of eng.physics.blocks) {
              if (!b.shattered && b.assignedLayer === 0) {
                eng.physics.shatterBlock(b, [0, 0, 0], 12, [1, 0, 0], false);
              }
            }
            for (let i = 0; i < 240 && eng.physics.turnPhase !== 'COLLAPSED'; i++) {
              eng.physics.step(1 / 60);
            }
            const deadline = Date.now() + 500;
            const poll = () => {
              const det = document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null;
              if (det === 'closed' || Date.now() > deadline) {
                resolve(eng.physics.turnPhase + ':' + det);
                return;
              }
              requestAnimationFrame(poll);
            };
            requestAnimationFrame(poll);
          }));
        };
        requestAnimationFrame(waitSillAndOpen);
      })`);
      expect(autoTuckCollapsed).toBe('COLLAPSED:closed');

      // E3 Inspect Strip in live Chromium DOM on touch block tap
      await resetClosedMobileState();
      const inspectDom = await cdp!.evaluate<{
        rendered: boolean;
        courseText: string;
        statusText: string;
      }>(`new Promise((resolve) => {
        const eng = window.__VITRAIL_ENGINE__;
        const rect = { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
        eng.camera.update(0.25, rect.width / rect.height);
        const m = eng.camera.viewProjMatrix;
        const rotVec = (q, v) => {
          const [qx, qy, qz, qw] = q;
          const [vx, vy, vz] = v;
          const tx = 2 * (qy * vz - qz * vy);
          const ty = 2 * (qz * vx - qx * vz);
          const tz = 2 * (qx * vy - qy * vx);
          return [
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
          ];
        };
        const localPts = [[0.27, 0, 0], [-0.27, 0, 0], [0, 0, 0.09], [0, 0, -0.09], [0, 0, 0]];
        let hitPt = null;
        for (const b of eng.physics.blocks) {
          if (b.shattered || !b.isLegalTarget || !b.isLooseSlot || b.assignedLayer < 4 || b.assignedLayer > 14) continue;
          for (const lp of localPts) {
            const off = rotVec(b.rotation, lp);
            const p = [b.position[0] + off[0], b.position[1] + off[1], b.position[2] + off[2]];
            const cx = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
            const cy = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
            const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
            const sx = ((cx / cw) * 0.5 + 0.5) * rect.width;
            const sy = (0.5 - (cy / cw) * 0.5) * rect.height;
            if (sx > 12 && sx < rect.width - 12 && sy > 12 && sy < rect.height - 12) {
              const { origin, dir } = eng.camera.screenToRay(sx, sy, rect);
              const pick = eng.raycastIntactBlocks(origin, dir);
              if (pick && pick.blockId === b.id) {
                hitPt = [sx, sy];
                break;
              }
            }
          }
          if (hitPt) break;
        }
        if (!hitPt) {
          resolve({ rendered: false, courseText: '', statusText: '' });
          return;
        }
        eng.handlePointerDown(hitPt[0], hitPt[1], 77, 0, false, rect, 'touch', 100);
        eng.handlePointerMove(hitPt[0] + 2, hitPt[1], 77, rect, 'touch', 120);
        eng.handlePointerUp(hitPt[0] + 2, hitPt[1], 77, rect, 'touch', 140);
        const deadline = Date.now() + 400;
        const poll = () => {
          const strip = document.querySelector('.plinth-inspect-strip');
          if (strip || Date.now() > deadline) {
            resolve({
              rendered: Boolean(strip),
              courseText: strip?.querySelector('.plinth-inspect-course')?.textContent?.trim() ?? '',
              statusText: strip?.querySelector('.plinth-inspect-status')?.textContent?.trim() ?? '',
            });
            return;
          }
          requestAnimationFrame(poll);
        };
        requestAnimationFrame(poll);
      })`);
      expect(inspectDom.rendered).toBe(true);
      expect(inspectDom.courseText).toMatch(/^Course \d+$/);
      expect(inspectDom.statusText.length).toBeGreaterThan(0);
    });

    it('5. [E4/E6 Modal A11y, Escape Cascade, Guide Focus Restore, & Closed-Sill Online Error Banner]', async () => {
      await resetClosedMobileState();

      // Open Detent 2 via keystone button
      await cdp!.evaluate(`new Promise((resolve) => {
        document.querySelector('button.plinth-sill-keystone')?.click();
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      })`);

      const modalCheck = await cdp!.evaluate<{
        role: string | null;
        ariaModal: string | null;
        focusableCount: number;
      }>(`(() => {
        const modal = document.querySelector('.plinth-full-modal');
        const focusables = modal
          ? Array.from(modal.querySelectorAll('button:not([disabled]), input:not([disabled])')).filter((el) => {
              const st = getComputedStyle(el);
              return st.display !== 'none' && st.visibility !== 'hidden';
            })
          : [];
        return {
          role: modal?.getAttribute('role') ?? null,
          ariaModal: modal?.getAttribute('aria-modal') ?? null,
          focusableCount: focusables.length,
        };
      })()`);
      expect(modalCheck.role).toBe('dialog');
      expect(modalCheck.ariaModal).toBe('true');
      expect(modalCheck.focusableCount).toBeGreaterThan(2);

      // Focus last modal button & press Tab -> wraps to first
      await cdp!.evaluate(`(() => {
        const modal = document.querySelector('.plinth-full-modal');
        const focusables = Array.from(modal.querySelectorAll('button:not([disabled]), input:not([disabled])')).filter((el) => {
          const st = getComputedStyle(el);
          return st.display !== 'none' && st.visibility !== 'hidden';
        });
        focusables[focusables.length - 1].focus();
      })()`);
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9, nativeVirtualKeyCode: 9,
      });
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9, nativeVirtualKeyCode: 9,
      });
      const wrappedToFirst = await cdp!.evaluate<boolean>(`(() => {
        const modal = document.querySelector('.plinth-full-modal');
        const focusables = Array.from(modal.querySelectorAll('button:not([disabled]), input:not([disabled])')).filter((el) => {
          const st = getComputedStyle(el);
          return st.display !== 'none' && st.visibility !== 'hidden';
        });
        return document.activeElement === focusables[0];
      })()`);
      expect(wrappedToFirst).toBe(true);

      // Focus first modal button & press Shift+Tab -> wraps to last
      await cdp!.evaluate(`(() => {
        const modal = document.querySelector('.plinth-full-modal');
        const focusables = Array.from(modal.querySelectorAll('button:not([disabled]), input:not([disabled])')).filter((el) => {
          const st = getComputedStyle(el);
          return st.display !== 'none' && st.visibility !== 'hidden';
        });
        focusables[0].focus();
      })()`);
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyDown', key: 'Tab', code: 'Tab', modifiers: 8, windowsVirtualKeyCode: 9, nativeVirtualKeyCode: 9,
      });
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyUp', key: 'Tab', code: 'Tab', modifiers: 8, windowsVirtualKeyCode: 9, nativeVirtualKeyCode: 9,
      });
      const wrappedToLast = await cdp!.evaluate<boolean>(`(() => {
        const modal = document.querySelector('.plinth-full-modal');
        const focusables = Array.from(modal.querySelectorAll('button:not([disabled]), input:not([disabled])')).filter((el) => {
          const st = getComputedStyle(el);
          return st.display !== 'none' && st.visibility !== 'hidden';
        });
        return document.activeElement === focusables[focusables.length - 1];
      })()`);
      expect(wrappedToLast).toBe(true);

      // Escape cascade: full -> half -> closed (+ focus restored to keystone!)
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27,
      });
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27,
      });
      const afterFirstEsc = await cdp!.evaluate<string | null>(
        `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null))))`
      );
      expect(afterFirstEsc).toBe('half');

      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27,
      });
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27,
      });
      const afterSecondEsc = await cdp!.evaluate<{ detent: string | null; keystoneFocused: boolean }>(
        `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r({
          detent: document.querySelector('.mobile-hud-root')?.getAttribute('data-detent') ?? null,
          keystoneFocused: document.activeElement === document.querySelector('button.plinth-sill-keystone'),
        }))))`
      );
      expect(afterSecondEsc.detent).toBe('closed');
      expect(afterSecondEsc.keystoneFocused).toBe(true);

      // Open Detent 2 -> click Vitrail Guide -> press Escape -> closes Guide AND restores focus to keystone!
      await cdp!.evaluate(`new Promise((resolve) => {
        document.querySelector('button.plinth-sill-keystone')?.click();
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const guideBtn = Array.from(document.querySelectorAll('.plinth-full-modal button'))
            .find((b) => b.textContent?.includes('Vitrail Guide'));
          guideBtn?.click();
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        }));
      })`);
      const guideDialogOpen = await cdp!.evaluate<boolean>(
        `Boolean(document.querySelector('[role="dialog"][aria-modal="true"]'))`
      );
      expect(guideDialogOpen).toBe(true);

      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27,
      });
      await cdp!.send('Input.dispatchKeyEvent', {
        type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27,
      });
      const afterGuideEsc = await cdp!.evaluate<{ dialogStillOpen: boolean; keystoneFocused: boolean }>(
        `new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r({
          dialogStillOpen: Boolean(document.querySelector('[role="dialog"][aria-modal="true"]')),
          keystoneFocused: document.activeElement === document.querySelector('button.plinth-sill-keystone'),
        }))))`
      );
      expect(afterGuideEsc.dialogStillOpen).toBe(false);
      expect(afterGuideEsc.keystoneFocused).toBe(true);

      // E4 Closed-sill .online-error-banner visibility on mpError (simulated WebSocket 1008 room full)
      const bannerCheck = await cdp!.evaluate<{
        visible: boolean;
        insideViewport: boolean;
        text: string;
        dismissed: boolean;
      }>(`new Promise((resolve) => {
        const OrigWS = window.WebSocket;
        window.WebSocket = class FakeWs {
          constructor() {
            this.readyState = 0;
            this.listeners = {};
            setTimeout(() => {
              this.readyState = 3;
              const ev = { code: 1008, reason: 'Room full (2/2 players)', wasClean: false };
              if (typeof this.onclose === 'function') this.onclose(ev);
              (this.listeners.close || []).forEach((fn) => fn(ev));
            }, 25);
          }
          addEventListener(type, fn) {
            this.listeners[type] = this.listeners[type] || [];
            this.listeners[type].push(fn);
          }
          removeEventListener() {}
          send() {}
          close() {}
        };
        const eng = window.__VITRAIL_ENGINE__;
        eng.joinOnlineRoom('K7M4P', { forceRelay: true });
        const deadline = Date.now() + 2000;
        const poll = () => {
          const banner = document.querySelector('.online-error-banner.plinth-mp-banner[role="alert"]');
          if (banner) {
            const r = banner.getBoundingClientRect();
            const text = banner.textContent?.trim() ?? '';
            const insideViewport = r.width > 10 && r.height > 10 && r.left >= -0.5 && r.right <= window.innerWidth + 0.5 && r.top >= -0.5 && r.bottom <= window.innerHeight + 0.5;
            document.querySelector('button[aria-label="Dismiss multiplayer error"]')?.click();
            requestAnimationFrame(() => requestAnimationFrame(() => {
              window.WebSocket = OrigWS;
              eng.resetGame('solo', 2);
              resolve({
                visible: true,
                insideViewport,
                text,
                dismissed: !document.querySelector('.online-error-banner.plinth-mp-banner'),
              });
            }));
            return;
          }
          if (Date.now() > deadline) {
            window.WebSocket = OrigWS;
            resolve({ visible: false, insideViewport: false, text: '', dismissed: false });
            return;
          }
          setTimeout(poll, 30);
        };
        setTimeout(poll, 40);
      })`);
      expect(bannerCheck.visible).toBe(true);
      expect(bannerCheck.insideViewport).toBe(true);
      expect(bannerCheck.text.length).toBeGreaterThan(0);
      expect(bannerCheck.dismissed).toBe(true);
    });
  });
});

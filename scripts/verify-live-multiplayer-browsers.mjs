#!/usr/bin/env node
/**
 * Live Two-Browser Hardware-WebGPU Full Multiplayer Lifecycle Verifier
 * (`scripts/verify-live-multiplayer-browsers.mjs` — `npm run verify:mp:live`)
 *
 * Launches isolated real Apple Silicon Metal-WebGPU Chromium browser processes
 * against local Vite (`http://localhost:5173`) and verifies all 8 real-DOM
 * 2-player lifecycle stages over BOTH live internet transports:
 *   - Mode A: Live public `wss://0.peerjs.com/peerjs` + native WebRTC `RTCDataChannel` (`?relay=none`)
 *   - Mode B: Live Cloudflare Durable Object WebSocket Relay (`?forceRelay=1&relay=wss://vitrail-rooms.andrew9687.workers.dev/__vitrail_mp`)
 */

import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const LIVE_CF_RELAY = 'wss://vitrail-rooms.andrew9687.workers.dev/__vitrail_mp';
const SIM_UNIT_TO_MM = 100.0; // 0.75 sim units = 75 mm regulation prism length

async function resolvePlaywright() {
  try {
    return await import('playwright');
  } catch {
    const npxRoot = path.join(os.homedir(), '.npm', '_npx');
    if (fs.existsSync(npxRoot)) {
      for (const dir of fs.readdirSync(npxRoot)) {
        const candidate = path.join(npxRoot, dir, 'node_modules', 'playwright', 'index.mjs');
        if (fs.existsSync(candidate)) {
          return await import(pathToFileURL(candidate).href);
        }
      }
    }
    throw new Error('Playwright not found in ~/.npm/_npx cache.');
  }
}

function allocateEphemeralPort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

function checkServerHeaders(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve({
        alive: res.statusCode !== undefined && res.statusCode < 500,
        statusCode: res.statusCode,
        coop: res.headers['cross-origin-opener-policy'] ?? null,
        coep: res.headers['cross-origin-embedder-policy'] ?? null,
      });
    });
    req.on('error', () => resolve({ alive: false, statusCode: null, coop: null, coep: null }));
    req.setTimeout(1500, () => {
      req.destroy();
      resolve({ alive: false, statusCode: null, coop: null, coep: null });
    });
  });
}

async function ensureViteDevServer(explicitUrl) {
  if (explicitUrl) {
    const cleanUrl = explicitUrl.replace(/\/+$/, '');
    const initial = await checkServerHeaders(cleanUrl);
    if (initial.alive) {
      return { baseUrl: cleanUrl, proc: null, headers: initial };
    }
    throw new Error(`Explicit --url ${cleanUrl} is not reachable`);
  }

  const port = await allocateEphemeralPort();
  const baseUrl = `http://localhost:${port}`;
  const viteBin = path.resolve(process.cwd(), 'node_modules', '.bin', 'vite');
  const viteProc = spawn(viteBin, ['--port', String(port), '--strictPort'], {
    cwd: process.cwd(),
    detached: true,
    stdio: 'ignore',
  });
  viteProc.unref();

  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 150));
    const check = await checkServerHeaders(baseUrl);
    if (check.alive) {
      return { baseUrl, proc: viteProc, headers: check };
    }
  }
  if (viteProc.pid) {
    try {
      process.kill(-viteProc.pid, 'SIGTERM');
    } catch {}
  }
  throw new Error(`Failed to boot detached Vite dev server at ${baseUrl}`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function statsMm(arr) {
  if (!arr || arr.length === 0) return { count: 0, min: 0, median: 0, p95: 0, max: 0 };
  const sorted = [...arr].sort((a, b) => a - b);
  const pick = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
  return {
    count: sorted.length,
    min: +sorted[0].toFixed(3),
    median: +pick(0.5).toFixed(3),
    p95: +pick(0.95).toFixed(3),
    max: +sorted[sorted.length - 1].toFixed(3),
  };
}

async function waitForEngineBoot(page, label) {
  const t0 = performance.now();
  await page.waitForFunction(
    () => {
      const eng = window.__VITRAIL_ENGINE__;
      return Boolean(eng && eng.rendererInitialized && eng.physics && eng.physics.blocks?.length === 54);
    },
    undefined,
    { timeout: 15000 }
  );
  const bootMs = +(performance.now() - t0).toFixed(1);

  const info = await page.evaluate(async () => {
    const eng = window.__VITRAIL_ENGINE__;
    let adapterInfo = null;
    if (navigator.gpu) {
      const ad = await navigator.gpu.requestAdapter();
      if (ad?.info) {
        adapterInfo = {
          vendor: ad.info.vendor,
          architecture: ad.info.architecture,
          device: ad.info.device,
        };
      }
    }
    // Non-invasive audio spy (`0` src/ edits)
    if (!window.__AUDIO_SPY__) {
      const counts = {
        tapTest: 0,
        blockThud: 0,
        glassImpact: 0,
        stackSuccess: 0,
        towerCollapse: 0,
        shardImpacts: 0,
        wobbleRock: 0,
        slideFrictionActiveCalls: 0,
      };
      const wrap = (method, key, pred) => {
        const orig = eng.audio[method];
        if (typeof orig === 'function' && !orig.__spied) {
          const fn = function (...args) {
            if (!pred || pred(...args)) counts[key]++;
            return orig.apply(this, args);
          };
          fn.__spied = true;
          eng.audio[method] = fn;
        }
      };
      wrap('playTapTestResonance', 'tapTest');
      wrap('playBlockThud', 'blockThud');
      wrap('playGlassImpact', 'glassImpact');
      wrap('playStackSuccess', 'stackSuccess');
      wrap('playTowerCollapse', 'towerCollapse');
      wrap('playShardImpacts', 'shardImpacts');
      wrap('playWobbleRock', 'wobbleRock');
      wrap('updateSlideFriction', 'slideFrictionActiveCalls', (speed) => Number(speed) > 0.01);
      window.__AUDIO_SPY__ = counts;
    }
    return {
      crossOriginIsolated: window.crossOriginIsolated,
      rendererInitialized: eng.rendererInitialized,
      physicsWorkerActive: eng.physicsWorkerActive,
      supportsTimestamps: Boolean(eng.renderer?.supportsTimestamps),
      adapterInfo,
    };
  });

  return { label, bootMs, ...info };
}

async function readAudioCounts(page) {
  return page.evaluate(() => ({ ...(window.__AUDIO_SPY__ ?? {}) }));
}

function diffAudioCounts(before, after) {
  const out = {};
  for (const k of Object.keys(after)) {
    out[k] = (after[k] ?? 0) - (before[k] ?? 0);
  }
  return out;
}

async function readMatchState(page) {
  return page.evaluate(() => {
    const eng = window.__VITRAIL_ENGINE__;
    if (!eng || !eng.physics) return null;
    const blocks = eng.physics.blocks.map((b) => ({
      id: b.id,
      pos: [b.position[0], b.position[1], b.position[2]],
      rot: [b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]],
      shattered: Boolean(b.shattered),
      extracted: Boolean(b.extracted),
      swatchId: b.swatchId,
      damage: b.damage ?? 0,
      assignedLayer: b.assignedLayer,
      assignedSlot: b.assignedSlot,
      isLooseSlot: Boolean(b.isLooseSlot),
      isLegalTarget: Boolean(b.isLegalTarget),
    }));
    const shards = (eng.physics.shards ?? []).map((s) => ({
      id: s.id,
      pos: [s.position[0], s.position[1], s.position[2]],
    }));
    const phaseTag = document.querySelector('.prism-phase-tag')?.textContent?.trim() ?? null;
    const onlineStatusText =
      document.querySelector('.online-status-pane .party-turn-copy')?.textContent?.trim() ?? null;
    const pingPillText = document.querySelector('.online-ping-pill')?.textContent?.trim() ?? null;
    const monumentTitle = document.querySelector('.monument-title')?.textContent?.trim() ?? null;
    const monumentProse = document.querySelector('.monument-prose')?.textContent?.trim() ?? null;
    const errorBanner = document.querySelector('.online-error-banner')?.textContent?.trim() ?? null;

    return {
      href: window.location.href,
      mpRole: eng.mpRole,
      mpRoomCode: eng.mpRoomCode,
      mpRoomSeed: eng.mpRoomSeed >>> 0,
      mpStatus: eng.mpStatus ? { ...eng.mpStatus } : null,
      physicsWorkerActive: Boolean(eng.physicsWorkerActive),
      turnPhase: eng.physics.turnPhase,
      currentPlayer: eng.physics.currentPlayer,
      blocksStacked: eng.physics.blocksStacked,
      activeBlockId: eng.physics.activeBlockId,
      carriedLandingPos: eng.physics.carriedLandingPos ? [...eng.physics.carriedLandingPos] : null,
      carriedYawRad: eng.physics.carriedYawOffsetRad ?? 0,
      collapseReason: eng.physics.collapseReason ?? '',
      lastReceivedStateSeq: eng.guestWorld?.getLastAcceptedSeq?.() ?? null,
      blocks,
      shardCount: shards.length,
      shards,
      hud: {
        phaseTag,
        onlineStatusText,
        pingPillText,
        monumentTitle,
        monumentProse,
        errorBanner,
      },
    };
  });
}

function compareBlockPosesMm(hostBlocks, guestBlocks) {
  const posDiffsMm = [];
  let swatchMatches = 0;
  const n = Math.min(hostBlocks.length, guestBlocks.length);
  for (let i = 0; i < n; i++) {
    const hb = hostBlocks[i];
    const gb = guestBlocks[i];
    if (hb.swatchId === gb.swatchId) swatchMatches++;
    if (hb.shattered || gb.shattered) continue;
    const dx = hb.pos[0] - gb.pos[0];
    const dy = hb.pos[1] - gb.pos[1];
    const dz = hb.pos[2] - gb.pos[2];
    posDiffsMm.push(Math.hypot(dx, dy, dz) * SIM_UNIT_TO_MM);
  }
  return {
    swatchMatches,
    totalBlocks: n,
    posErrorMm: statsMm(posDiffsMm),
  };
}

function compareShardHeapsMm(hostShards, guestShards) {
  const guestById = new Map(guestShards.map((s) => [s.id, s.pos]));
  const diffsMm = [];
  let matchedIds = 0;
  for (const hs of hostShards) {
    const gp = guestById.get(hs.id);
    if (!gp) continue;
    matchedIds++;
    diffsMm.push(
      Math.hypot(hs.pos[0] - gp[0], hs.pos[1] - gp[1], hs.pos[2] - gp[2]) * SIM_UNIT_TO_MM
    );
  }
  return {
    hostShardCount: hostShards.length,
    guestShardCount: guestShards.length,
    matchedIds,
    posErrorMm: statsMm(diffsMm),
  };
}

/**
 * Finds a front-visible, unobstructed legal outer/loose block on `<canvas>` and computes exact CSS
 * screen coordinates `(sx0, sy0) -> (sx1, sy1)` along its axial extraction channel.
 */
async function planBlockExtractionGesture(page, excludeBlockIds = [], options = {}) {
  return page.evaluate(
    ({ excludeIds, preferLayerMin, preferLayerMax, onlyOuterSlots }) => {
      const eng = window.__VITRAIL_ENGINE__;
      const rect = eng.canvas.getBoundingClientRect();
      const vp = eng.camera.viewProjMatrix;
      const camPos = eng.camera.position;

      const projectWorld = (wx, wy, wz) => {
        const cx = vp[0] * wx + vp[4] * wy + vp[8] * wz + vp[12];
        const cy = vp[1] * wx + vp[5] * wy + vp[9] * wz + vp[13];
        const cw = vp[3] * wx + vp[7] * wy + vp[11] * wz + vp[15];
        if (cw <= 0.001) return null;
        const ndcX = cx / cw;
        const ndcY = cy / cw;
        return [
          rect.left + (ndcX * 0.5 + 0.5) * rect.width,
          rect.top + (0.5 - ndcY * 0.5) * rect.height,
        ];
      };

      const quatRotX = (q) => {
        const [x, y, z, w] = q;
        return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
      };

      const candidates = eng.physics.blocks
        .filter(
          (b) =>
            !b.shattered &&
            !b.extracted &&
            b.isLegalTarget &&
            !excludeIds.includes(b.id) &&
            b.assignedLayer >= preferLayerMin &&
            b.assignedLayer <= preferLayerMax &&
            (!onlyOuterSlots || b.assignedSlot !== 1)
        )
        .sort((a, b) => {
          // Prefer loose slots first, then middle courses
          const looseDiff = Number(Boolean(b.isLooseSlot)) - Number(Boolean(a.isLooseSlot));
          if (looseDiff !== 0) return looseDiff;
          return Math.abs(a.assignedLayer - 9) - Math.abs(b.assignedLayer - 9);
        });

      for (const blk of candidates) {
        const axisX = quatRotX(blk.rotation);
        const toCam = [
          camPos[0] - blk.position[0],
          0,
          camPos[2] - blk.position[2],
        ];
        const signs =
          axisX[0] * toCam[0] + axisX[2] * toCam[2] >= 0 ? [1, -1] : [-1, 1];

        for (const sign of signs) {
          const pullDir = [axisX[0] * sign, axisX[1] * sign, axisX[2] * sign];
          const grabWorld = [
            blk.position[0] + pullDir[0] * 0.24,
            blk.position[1] + pullDir[1] * 0.24,
            blk.position[2] + pullDir[2] * 0.24,
          ];
          const extractWorld = [
            blk.position[0] + pullDir[0] * 1.55,
            blk.position[1],
            blk.position[2] + pullDir[2] * 1.55,
          ];
          const p0 = projectWorld(grabWorld[0], grabWorld[1], grabWorld[2]);
          const p1 = projectWorld(extractWorld[0], extractWorld[1], extractWorld[2]);
          if (!p0 || !p1) continue;
          if (
            p0[0] < 60 ||
            p0[0] > rect.width - 60 ||
            p0[1] < 60 ||
            p0[1] > rect.height - 130 ||
            p1[0] < 40 ||
            p1[0] > rect.width - 40 ||
            p1[1] < 40 ||
            p1[1] > rect.height - 120
          ) {
            continue;
          }
          // Ensure HUD elements do not occlude the start coordinate
          const topEl = document.elementFromPoint(p0[0], p0[1]);
          if (topEl !== eng.canvas) continue;

          // Verify raycast hits this exact block!
          const { origin, dir } = eng.camera.screenToRay(p0[0], p0[1], rect);
          const hit = eng.raycastIntactBlocks(origin, dir);
          if (hit && hit.blockId === blk.id) {
            return {
              blockId: blk.id,
              assignedLayer: blk.assignedLayer,
              assignedSlot: blk.assignedSlot,
              isLooseSlot: Boolean(blk.isLooseSlot),
              initialPos: [...blk.position],
              pullDir,
              startScreen: [p0[0], p0[1]],
              endScreen: [p1[0], p1[1]],
            };
          }
        }
      }
      return null;
    },
    {
      excludeIds: excludeBlockIds,
      preferLayerMin: options.preferLayerMin ?? 5,
      preferLayerMax: options.preferLayerMax ?? 13,
      onlyOuterSlots: options.onlyOuterSlots ?? true,
    }
  );
}

async function runModeLifecycle(chromium, modeConfig) {
  const {
    modeId,
    label,
    hostBootUrl,
    guestJoinStyle, // 'url-auto' | 'manual-input'
    guestBootUrlBuilder,
    expectedTransport,
  } = modeConfig;

  console.error(`\n=== Running ${modeId}: ${label} ===`);
  const browserHost = await chromium.launch({
    headless: false,
    args: [
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--enable-dawn-features=allow_unsafe_apis',
      '--window-position=20,40',
      '--window-size=1160,780',
    ],
  });
  const browserGuest = await chromium.launch({
    headless: false,
    args: [
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--enable-dawn-features=allow_unsafe_apis',
      '--disable-backgrounding-occluded-windows',
      '--window-position=360,80',
      '--window-size=1160,780',
    ],
  });

  const result = {
    modeId,
    label,
    expectedTransport,
    guestJoinStyle,
  };

  try {
    const ctxHost = await browserHost.newContext({ viewport: { width: 1120, height: 720 }, deviceScaleFactor: 1 });
    const ctxGuest = await browserGuest.newContext({ viewport: { width: 1120, height: 720 }, deviceScaleFactor: 1 });
    const pageHost = await ctxHost.newPage();
    const pageGuest = await ctxGuest.newPage();

    const pageErrors = { host: [], guest: [] };
    pageHost.on('pageerror', (e) => pageErrors.host.push(e.message));
    pageGuest.on('pageerror', (e) => pageErrors.guest.push(e.message));

    // -----------------------------------------------------------------------
    // Step 1: Boot Host -> Click Online -> Click Host -> Guest Joins
    // -----------------------------------------------------------------------
    await pageHost.goto(hostBootUrl, { waitUntil: 'domcontentloaded' });
    result.hostBoot = await waitForEngineBoot(pageHost, 'host');

    // Real DOM click on Online tab -> Host button
    await pageHost.locator('button.mode-course-btn', { hasText: 'Online' }).click();
    await pageHost.locator('button.online-host-btn').waitFor({ state: 'visible', timeout: 5000 });
    await pageHost.locator('button.online-host-btn').click();

    await pageHost.locator('.online-code-value').waitFor({ state: 'visible', timeout: 5000 });
    const roomCode = (await pageHost.locator('.online-code-value').textContent()).trim();
    const hostInviteUrl = await pageHost.evaluate(() => window.__VITRAIL_ENGINE__.getOnlineInviteUrl());
    const hostUrlAfterStart = pageHost.url();

    const guestTargetUrl = guestBootUrlBuilder(roomCode);
    const tJoin0 = performance.now();

    if (guestJoinStyle === 'url-auto') {
      await pageGuest.goto(guestTargetUrl, { waitUntil: 'domcontentloaded' });
      result.guestBoot = await waitForEngineBoot(pageGuest, 'guest');
    } else {
      // Manual room code typing + Join button click!
      await pageGuest.goto(guestTargetUrl, { waitUntil: 'domcontentloaded' });
      result.guestBoot = await waitForEngineBoot(pageGuest, 'guest');
      await pageGuest.locator('button.mode-course-btn', { hasText: 'Online' }).click();
      const codeInput = pageGuest.locator('#online-room-code-input');
      await codeInput.waitFor({ state: 'visible', timeout: 5000 });
      await codeInput.click();
      await pageGuest.keyboard.type(roomCode.toLowerCase(), { delay: 30 });
      const joinBtn = pageGuest.locator('button.online-join-btn');
      await joinBtn.click();
    }

    // Wait for both Host & Guest to report mpPeerConnected === true and initial state snapshot received
    await Promise.all([
      pageHost.waitForFunction(
        () => window.__VITRAIL_ENGINE__?.mpStatus?.peerConnected === true,
        undefined,
        { timeout: 15000 }
      ),
      pageGuest.waitForFunction(
        () =>
          window.__VITRAIL_ENGINE__?.mpStatus?.peerConnected === true &&
          (window.__VITRAIL_ENGINE__?.guestWorld?.getLastAcceptedSeq?.() ?? 0) > 0,
        undefined,
        { timeout: 15000 }
      ),
    ]);
    const joinHandshakeMs = +(performance.now() - tJoin0).toFixed(1);
    // Allow 1 ping/pong cycle (`1500 ms` interval) so pingMs populates
    await sleep(1650);

    const hostS0 = await readMatchState(pageHost);
    const guestS0 = await readMatchState(pageGuest);
    const initCompare = compareBlockPosesMm(hostS0.blocks, guestS0.blocks);

    result.connection = {
      roomCode,
      hostUrlAfterStart,
      hostInviteUrl,
      joinHandshakeMs,
      hostTransport: hostS0.mpStatus?.transport,
      guestTransport: guestS0.mpStatus?.transport,
      hostPingMs: hostS0.mpStatus?.pingMs,
      guestPingMs: guestS0.mpStatus?.pingMs,
      roomSeedMatch: hostS0.mpRoomSeed === guestS0.mpRoomSeed,
      roomSeed: hostS0.mpRoomSeed,
      swatchParity: `${initCompare.swatchMatches}/${initCompare.totalBlocks}`,
      initialPoseErrorMm: initCompare.posErrorMm,
      hostHudTurn: hostS0.hud.phaseTag,
      guestHudTurn: guestS0.hud.phaseTag,
    };

    // -----------------------------------------------------------------------
    // Step 2: Turn 1 — Host Tap-Tests, Drags Loose Prism Out, Rotates 90°, Places
    // -----------------------------------------------------------------------
    await pageHost.bringToFront();
    const hostPlan = await planBlockExtractionGesture(pageHost, [], {
      preferLayerMin: 5,
      preferLayerMax: 13,
      onlyOuterSlots: true,
    });
    if (!hostPlan) throw new Error(`${modeId}: Failed to find selectable Host block`);

    const guestAudioBeforeTap1 = await readAudioCounts(pageGuest);
    const hostAudioBeforeTap1 = await readAudioCounts(pageHost);

    // 2a. Real < 5 px tap-test on Host canvas (allow 120 ms hover settle for worker mj_ray)
    await pageHost.mouse.move(hostPlan.startScreen[0], hostPlan.startScreen[1]);
    await sleep(120);
    await pageHost.mouse.down();
    await sleep(55);
    await pageHost.mouse.up();
    await sleep(260);

    const guestAudioAfterTap1 = await readAudioCounts(pageGuest);
    const hostAudioAfterTap1 = await readAudioCounts(pageHost);
    const postTapPhase = await pageHost.evaluate(() => window.__VITRAIL_ENGINE__.physics.turnPhase);

    // Start 120 Hz Guest & Host pose samplers during Host's extraction drag
    await pageHost.evaluate((bid) => {
      window.__TURN1_HOST_TRACE__ = [];
      const rec = () => {
        const eng = window.__VITRAIL_ENGINE__;
        const b = eng.physics.blocks[bid];
        window.__TURN1_HOST_TRACE__.push({
          wallMs: Date.now(),
          phase: eng.physics.turnPhase,
          pos: [...b.position],
          landing: eng.physics.carriedLandingPos ? [...eng.physics.carriedLandingPos] : null,
        });
        if (window.__TURN1_SAMPLING__) requestAnimationFrame(rec);
      };
      window.__TURN1_SAMPLING__ = true;
      requestAnimationFrame(rec);
    }, hostPlan.blockId);

    await pageGuest.evaluate((bid) => {
      window.__TURN1_GUEST_TRACE__ = [];
      const rec = () => {
        const eng = window.__VITRAIL_ENGINE__;
        const b = eng.physics.blocks[bid];
        window.__TURN1_GUEST_TRACE__.push({
          wallMs: Date.now(),
          phase: eng.physics.turnPhase,
          pos: [...b.position],
          landing: eng.physics.carriedLandingPos ? [...eng.physics.carriedLandingPos] : null,
        });
        if (window.__TURN1_SAMPLING__) requestAnimationFrame(rec);
      };
      window.__TURN1_SAMPLING__ = true;
      requestAnimationFrame(rec);
    }, hostPlan.blockId);

    const guestAudioBeforePull1 = await readAudioCounts(pageGuest);

    // 2b. Smooth horizontal drag out on Host canvas
    await pageHost.mouse.move(hostPlan.startScreen[0], hostPlan.startScreen[1]);
    await pageHost.mouse.down();
    const steps1 = 44;
    let extractedStep1 = null;
    for (let i = 1; i <= steps1; i++) {
      const u = i / steps1;
      const sx = hostPlan.startScreen[0] + (hostPlan.endScreen[0] - hostPlan.startScreen[0]) * u;
      const sy = hostPlan.startScreen[1] + (hostPlan.endScreen[1] - hostPlan.startScreen[1]) * u;
      await pageHost.mouse.move(sx, sy);
      await sleep(16);
      if (extractedStep1 === null) {
        const ph = await pageHost.evaluate(() => window.__VITRAIL_ENGINE__.physics.turnPhase);
        if (ph === 'EXTRACTED') extractedStep1 = i;
      }
    }
    await pageHost.mouse.up();

    // Ensure EXTRACTED reached
    await pageHost.waitForFunction(
      () => window.__VITRAIL_ENGINE__.physics.turnPhase === 'EXTRACTED',
      undefined,
      { timeout: 5000 }
    );
    await sleep(320); // Allow extraction arc + snapshot sync

    // Stop drag trace & pair wall-clock samples
    const [hostTrace1, guestTrace1] = await Promise.all([
      pageHost.evaluate(() => {
        window.__TURN1_SAMPLING__ = false;
        return window.__TURN1_HOST_TRACE__;
      }),
      pageGuest.evaluate(() => {
        window.__TURN1_SAMPLING__ = false;
        return window.__TURN1_GUEST_TRACE__;
      }),
    ]);

    const liveDragLagMm1 = [];
    for (const gs of guestTrace1) {
      let best = null;
      let bestDt = Infinity;
      for (const hs of hostTrace1) {
        const dt = Math.abs(hs.wallMs - gs.wallMs);
        if (dt < bestDt) {
          bestDt = dt;
          best = hs;
        }
      }
      if (best && bestDt <= 25 && (best.phase === 'PULLING' || gs.phase === 'PULLING')) {
        liveDragLagMm1.push(
          Math.hypot(
            best.pos[0] - gs.pos[0],
            best.pos[1] - gs.pos[1],
            best.pos[2] - gs.pos[2]
          ) * SIM_UNIT_TO_MM
        );
      }
    }

    // 2c. Rotate 90° R on Host HUD -> check carried landing guide + yaw sync on Guest
    await pageHost.locator('button[aria-label="Rotate prism 90 degrees (R)"]').click();
    await sleep(200);
    const hostCarried1 = await readMatchState(pageHost);
    const guestCarried1 = await readMatchState(pageGuest);
    const carriedGuideErrorMm1 =
      hostCarried1.carriedLandingPos && guestCarried1.carriedLandingPos
        ? +(
            Math.hypot(
              hostCarried1.carriedLandingPos[0] - guestCarried1.carriedLandingPos[0],
              hostCarried1.carriedLandingPos[1] - guestCarried1.carriedLandingPos[1],
              hostCarried1.carriedLandingPos[2] - guestCarried1.carriedLandingPos[2]
            ) * SIM_UNIT_TO_MM
          ).toFixed(3)
        : null;
    const carriedYawDiffDeg1 = +(
      (Math.abs(hostCarried1.carriedYawRad - guestCarried1.carriedYawRad) * 180) /
      Math.PI
    ).toFixed(2);

    // 2d. Click Place Prism on Host HUD -> wait for turn handoff to Guest (`currentPlayer === 1`)
    await pageHost.locator('button.prism-release-btn').click();
    await Promise.all([
      pageHost.waitForFunction(
        () =>
          window.__VITRAIL_ENGINE__.physics.turnPhase === 'SELECT_BLOCK' &&
          window.__VITRAIL_ENGINE__.physics.currentPlayer === 1 &&
          window.__VITRAIL_ENGINE__.physics.blocksStacked === 1,
        undefined,
        { timeout: 10000 }
      ),
      pageGuest.waitForFunction(
        () =>
          window.__VITRAIL_ENGINE__.physics.turnPhase === 'SELECT_BLOCK' &&
          window.__VITRAIL_ENGINE__.physics.currentPlayer === 1 &&
          window.__VITRAIL_ENGINE__.physics.blocksStacked === 1,
        undefined,
        { timeout: 10000 }
      ),
    ]);
    await sleep(250);

    const guestAudioAfterTurn1 = await readAudioCounts(pageGuest);
    const hostS1 = await readMatchState(pageHost);
    const guestS1 = await readMatchState(pageGuest);
    const turn1SettledCompare = compareBlockPosesMm(hostS1.blocks, guestS1.blocks);

    result.turn1Host = {
      blockId: hostPlan.blockId,
      layer: hostPlan.assignedLayer,
      slot: hostPlan.assignedSlot,
      tapTestKeptSelectPhase: postTapPhase === 'SELECT_BLOCK',
      hostTapAudioDelta: diffAudioCounts(hostAudioBeforeTap1, hostAudioAfterTap1).tapTest,
      guestTapAudioDelta: diffAudioCounts(guestAudioBeforeTap1, guestAudioAfterTap1).tapTest,
      extractedAtStep: extractedStep1,
      guestLiveDragTrackingLagMm: statsMm(liveDragLagMm1),
      carriedGuideSyncErrorMm: carriedGuideErrorMm1,
      carriedYawDiffDeg: carriedYawDiffDeg1,
      settled54BlockErrorMm: turn1SettledCompare.posErrorMm,
      guestTurn1AudioCounts: diffAudioCounts(guestAudioBeforePull1, guestAudioAfterTurn1),
      turnHandoffToGuest: {
        hostCurrentPlayer: hostS1.currentPlayer,
        guestCurrentPlayer: guestS1.currentPlayer,
        guestHudPhaseTag: guestS1.hud.phaseTag,
        guestOnlineStatus: guestS1.hud.onlineStatusText,
      },
    };

    // -----------------------------------------------------------------------
    // Step 3: Turn 2 — Guest Real DOM Pointer Pull, Rubber-Band Telemetry & Place
    // -----------------------------------------------------------------------
    await pageGuest.bringToFront();
    await sleep(150);

    // Find a loose block using Host's authoritative `isLooseSlot` metadata, projected in Guest's viewport!
    const hostLooseIds = hostS1.blocks
      .filter(
        (b) =>
          !b.shattered &&
          !b.extracted &&
          b.isLegalTarget &&
          b.id !== hostPlan.blockId &&
          b.assignedLayer >= 4 &&
          b.assignedLayer <= 13 &&
          b.isLooseSlot
      )
      .map((b) => b.id);

    const nonLooseExclude = hostS1.blocks
      .filter((b) => !hostLooseIds.includes(b.id))
      .map((b) => b.id);

    let guestPlan = await planBlockExtractionGesture(
      pageGuest,
      [hostPlan.blockId, ...nonLooseExclude],
      { preferLayerMin: 4, preferLayerMax: 13, onlyOuterSlots: false }
    );
    if (!guestPlan) {
      guestPlan = await planBlockExtractionGesture(pageGuest, [hostPlan.blockId], {
        preferLayerMin: 4,
        preferLayerMax: 13,
        onlyOuterSlots: true,
      });
    }
    if (!guestPlan) throw new Error(`${modeId}: Failed to find selectable Guest block`);

    const hostAudioBeforeTap2 = await readAudioCounts(pageHost);
    const guestAudioBeforeTap2 = await readAudioCounts(pageGuest);

    // 3a. Guest real < 5 px tap-test on Guest canvas
    await pageGuest.mouse.move(guestPlan.startScreen[0], guestPlan.startScreen[1]);
    await sleep(80);
    await pageGuest.mouse.down();
    await sleep(45);
    await pageGuest.mouse.up();
    await sleep(300);

    const hostAudioAfterTap2 = await readAudioCounts(pageHost);
    const guestAudioAfterTap2 = await readAudioCounts(pageGuest);

    // Instrument 120 Hz Guest rAF pull trace (onset latency + rubber-banding reversals + optimistic anchor lag)
    await pageGuest.evaluate(
      ({ bid, initPos, pullDir }) => {
        window.__TURN2_GUEST_PULL_TRACE__ = [];
        window.__TURN2_DRAG_THRESHOLD_MS__ = null;
        const rec = () => {
          const eng = window.__VITRAIL_ENGINE__;
          const b = eng.physics.blocks[bid];
          const gw = eng.guestWorld;
          const dx = b.position[0] - initPos[0];
          const dy = b.position[1] - initPos[1];
          const dz = b.position[2] - initPos[2];
          const axialProgressMm =
            (dx * pullDir[0] + dy * pullDir[1] + dz * pullDir[2]) * 100.0;
          const anchor = gw?.optimisticGuestPullAnchor ? [...gw.optimisticGuestPullAnchor] : null;
          window.__TURN2_GUEST_PULL_TRACE__.push({
            tMs: performance.now(),
            wallMs: Date.now(),
            phase: eng.physics.turnPhase,
            interactionMode: eng.interactionMode,
            axialProgressMm,
            pos: [...b.position],
            anchor,
            rxSeq: gw?.getLastAcceptedSeq?.() ?? 0,
          });
          if (window.__TURN2_SAMPLING__) requestAnimationFrame(rec);
        };
        window.__TURN2_SAMPLING__ = true;
        requestAnimationFrame(rec);
      },
      {
        bid: guestPlan.blockId,
        initPos: guestPlan.initialPos,
        pullDir: guestPlan.pullDir,
      }
    );

    // 3b. Guest smooth outward pointer drag on Guest `<canvas>`!
    await pageGuest.mouse.move(guestPlan.startScreen[0], guestPlan.startScreen[1]);
    await pageGuest.mouse.down();
    const steps2 = 52;
    let dragThresholdCrossWallMs = null;
    for (let i = 1; i <= steps2; i++) {
      const u = i / steps2;
      const sx = guestPlan.startScreen[0] + (guestPlan.endScreen[0] - guestPlan.startScreen[0]) * u;
      const sy = guestPlan.startScreen[1] + (guestPlan.endScreen[1] - guestPlan.startScreen[1]) * u;
      const movedPx = Math.hypot(sx - guestPlan.startScreen[0], sy - guestPlan.startScreen[1]);
      await pageGuest.mouse.move(sx, sy);
      if (dragThresholdCrossWallMs === null && movedPx >= 6.0) {
        dragThresholdCrossWallMs = Date.now();
        await pageGuest.evaluate(() => {
          if (window.__TURN2_DRAG_THRESHOLD_MS__ === null) {
            window.__TURN2_DRAG_THRESHOLD_MS__ = performance.now();
          }
        });
      }
      await sleep(16);
      const guestPhaseNow = await pageGuest.evaluate(
        () => window.__VITRAIL_ENGINE__.physics.turnPhase
      );
      if (guestPhaseNow === 'EXTRACTED') break;
    }
    await pageGuest.mouse.up();

    await Promise.all([
      pageHost.waitForFunction(
        () => window.__VITRAIL_ENGINE__.physics.turnPhase === 'EXTRACTED',
        undefined,
        { timeout: 6000 }
      ),
      pageGuest.waitForFunction(
        () => window.__VITRAIL_ENGINE__.physics.turnPhase === 'EXTRACTED',
        undefined,
        { timeout: 6000 }
      ),
    ]);
    await sleep(280);

    const guestPullAnalysis = await pageGuest.evaluate(() => {
      window.__TURN2_SAMPLING__ = false;
      const trace = window.__TURN2_GUEST_PULL_TRACE__ ?? [];
      const tThresh = window.__TURN2_DRAG_THRESHOLD_MS__;
      let motionOnsetLatencyMs = null;
      if (tThresh !== null) {
        const firstMove = trace.find((s) => s.tMs >= tThresh && s.axialProgressMm >= 0.5);
        if (firstMove) {
          motionOnsetLatencyMs = +(firstMove.tMs - tThresh).toFixed(2);
        }
      }

      // Analyze monotonic pull samples (while phase === 'SELECT_BLOCK' or 'PULLING' before 'EXTRACTED')
      const pullSamples = trace.filter(
        (s) =>
          tThresh !== null &&
          s.tMs >= tThresh &&
          (s.phase === 'SELECT_BLOCK' || s.phase === 'PULLING')
      );
      let maxAxialReachedMm = 0;
      let maxBackwardSnapFromPeakMm = 0;
      let frameToFrameReversalCount = 0;
      let maxFrameToFrameReversalMm = 0;
      for (let i = 0; i < pullSamples.length; i++) {
        const cur = pullSamples[i].axialProgressMm;
        if (cur > maxAxialReachedMm) {
          maxAxialReachedMm = cur;
        } else {
          const snapBack = maxAxialReachedMm - cur;
          if (snapBack > maxBackwardSnapFromPeakMm) maxBackwardSnapFromPeakMm = snapBack;
        }
        if (i > 0) {
          const stepDelta = cur - pullSamples[i - 1].axialProgressMm;
          if (stepDelta < -0.25) {
            frameToFrameReversalCount++;
            if (-stepDelta > maxFrameToFrameReversalMm) {
              maxFrameToFrameReversalMm = -stepDelta;
            }
          }
        }
      }
      return {
        sampleCount: pullSamples.length,
        motionOnsetLatencyMs,
        maxAxialPullProgressMm: +maxAxialReachedMm.toFixed(2),
        maxBackwardSnapFromPeakMm: +maxBackwardSnapFromPeakMm.toFixed(3),
        frameToFrameReversalCount,
        maxFrameToFrameReversalMm: +maxFrameToFrameReversalMm.toFixed(3),
      };
    });

    // 3c. Guest moves pointer on top course, clicks 90° R on Guest HUD, and clicks Place Prism!
    const topScreenPt = await pageGuest.evaluate(() => {
      const eng = window.__VITRAIL_ENGINE__;
      const lp = eng.physics.carriedLandingPos ?? [0, eng.physics.currentLayers * 0.15, 0];
      const rect = eng.canvas.getBoundingClientRect();
      const vp = eng.camera.viewProjMatrix;
      const cx = vp[0] * lp[0] + vp[4] * lp[1] + vp[8] * lp[2] + vp[12];
      const cy = vp[1] * lp[0] + vp[5] * lp[1] + vp[9] * lp[2] + vp[13];
      const cw = vp[3] * lp[0] + vp[7] * lp[1] + vp[11] * lp[2] + vp[15];
      return [
        rect.left + ((cx / cw) * 0.5 + 0.5) * rect.width,
        rect.top + (0.5 - (cy / cw) * 0.5) * rect.height,
      ];
    });
    await pageGuest.mouse.move(topScreenPt[0] + 18, topScreenPt[1]);
    await sleep(120);
    await pageGuest.locator('button[aria-label="Rotate prism 90 degrees (R)"]').click();
    await sleep(220);

    const hostCarried2 = await readMatchState(pageHost);
    const guestCarried2 = await readMatchState(pageGuest);
    const guestPlaceGuideErrorMm =
      hostCarried2.carriedLandingPos && guestCarried2.carriedLandingPos
        ? +(
            Math.hypot(
              hostCarried2.carriedLandingPos[0] - guestCarried2.carriedLandingPos[0],
              hostCarried2.carriedLandingPos[1] - guestCarried2.carriedLandingPos[1],
              hostCarried2.carriedLandingPos[2] - guestCarried2.carriedLandingPos[2]
            ) * SIM_UNIT_TO_MM
          ).toFixed(3)
        : null;

    await pageGuest.locator('button.prism-release-btn').click();

    await Promise.all([
      pageHost.waitForFunction(
        () =>
          window.__VITRAIL_ENGINE__.physics.turnPhase === 'SELECT_BLOCK' &&
          window.__VITRAIL_ENGINE__.physics.currentPlayer === 0 &&
          window.__VITRAIL_ENGINE__.physics.blocksStacked === 2,
        undefined,
        { timeout: 10000 }
      ),
      pageGuest.waitForFunction(
        () =>
          window.__VITRAIL_ENGINE__.physics.turnPhase === 'SELECT_BLOCK' &&
          window.__VITRAIL_ENGINE__.physics.currentPlayer === 0 &&
          window.__VITRAIL_ENGINE__.physics.blocksStacked === 2,
        undefined,
        { timeout: 10000 }
      ),
    ]);
    await sleep(250);

    const hostS2 = await readMatchState(pageHost);
    const guestS2 = await readMatchState(pageGuest);
    const turn2SettledCompare = compareBlockPosesMm(hostS2.blocks, guestS2.blocks);

    result.turn2Guest = {
      blockId: guestPlan.blockId,
      layer: guestPlan.assignedLayer,
      slot: guestPlan.assignedSlot,
      isLooseSlot: guestPlan.isLooseSlot,
      hostTapAudioDelta: diffAudioCounts(hostAudioBeforeTap2, hostAudioAfterTap2).tapTest,
      guestTapAudioDelta: diffAudioCounts(guestAudioBeforeTap2, guestAudioAfterTap2).tapTest,
      pullTelemetry: guestPullAnalysis,
      carriedGuideSyncErrorMm: guestPlaceGuideErrorMm,
      settled54BlockErrorMm: turn2SettledCompare.posErrorMm,
      turnHandoffBackToHost: {
        hostCurrentPlayer: hostS2.currentPlayer,
        guestCurrentPlayer: guestS2.currentPlayer,
        hostHudPhaseTag: hostS2.hud.phaseTag,
        guestHudPhaseTag: guestS2.hud.phaseTag,
        blocksStackedHost: hostS2.blocksStacked,
        blocksStackedGuest: guestS2.blocksStacked,
      },
    };

    // -----------------------------------------------------------------------
    // Step 4: Host Tab Occlusion & Halted-rAF `30 Hz` Background Pump Probe
    // -----------------------------------------------------------------------
    // Note: Playwright's internal CRPage CDP session holds focus emulation (`document.hidden === false`),
    // so we BOTH minimize the OS window via CDP (engaging OS background timer rules) AND halt foreground
    // `rAF` via `setHostTabOccluded(true)` so 100% of snapshots must originate from `startHostBackgroundPump()`.
    const cdpHost = await ctxHost.newCDPSession(pageHost);
    const { windowId: hostWinId } = await cdpHost.send('Browser.getWindowForTarget');
    await cdpHost.send('Browser.setWindowBounds', {
      windowId: hostWinId,
      bounds: { windowState: 'minimized' },
    });
    const preOccludeFgMs = await pageHost.evaluate(() => {
      const eng = window.__VITRAIL_ENGINE__;
      if (eng.rafId) {
        cancelAnimationFrame(eng.rafId);
        eng.rafId = 0;
      }
      eng.running = false;
      eng.setHostTabOccluded(true);
      return eng.lastForegroundFrameMs;
    });
    await sleep(250);
    // Measure occluded Host 30 Hz background pump rate across 1.5 s (and past Chromium's 10 s cliff in Mode B)
    const seqStart0 = await pageGuest.evaluate(
      () => window.__VITRAIL_ENGINE__.guestWorld?.getLastAcceptedSeq?.() ?? 0
    );
    await sleep(1500);
    const seqEndEarly = await pageGuest.evaluate(
      () => window.__VITRAIL_ENGINE__.guestWorld?.getLastAcceptedSeq?.() ?? 0
    );
    const earlyOccludedHz = +(((seqEndEarly - seqStart0) / 1.5)).toFixed(2);

    let post10sOccludedHz = null;
    if (modeId === 'ModeB_LiveCloudflareRelay') {
      // Wait past Chromium's 10-second background timer intensive-throttling cliff!
      await sleep(9200);
      const seq10a = await pageGuest.evaluate(
        () => window.__VITRAIL_ENGINE__.guestWorld?.getLastAcceptedSeq?.() ?? 0
      );
      await sleep(2000);
      const seq10b = await pageGuest.evaluate(
        () => window.__VITRAIL_ENGINE__.guestWorld?.getLastAcceptedSeq?.() ?? 0
      );
      post10sOccludedHz = +(((seq10b - seq10a) / 2.0)).toFixed(2);
    }
    const postOccludeFgMs = await pageHost.evaluate(() => {
      const eng = window.__VITRAIL_ENGINE__;
      const fgDuring = eng.lastForegroundFrameMs;
      eng.setHostTabOccluded(false);
      eng.running = true;
      eng.lastFrameMs = performance.now();
      eng.lastForegroundFrameMs = performance.now();
      eng.rafId = requestAnimationFrame(eng.tick);
      return fgDuring;
    });
    await cdpHost.send('Browser.setWindowBounds', {
      windowId: hostWinId,
      bounds: { windowState: 'normal' },
    });
    await sleep(200);
    await pageHost.bringToFront();

    result.hostTabOcclusion = {
      foregroundRafHaltedVerified: postOccludeFgMs === preOccludeFgMs,
      earlyOccludedSnapshotRateHz: earlyOccludedHz,
      post10SecBackgroundSnapshotRateHz: post10sOccludedHz,
    };

    // -----------------------------------------------------------------------
    // Step 5: Guest Mid-Match `F5` Reload on `?room=<CODE>` (`blocksStacked === 2`)
    // -----------------------------------------------------------------------
    await sleep(350);
    await pageGuest.bringToFront();
    const guestUrlBeforeReload = pageGuest.url();
    const tReload0 = performance.now();
    await pageGuest.reload({ waitUntil: 'domcontentloaded' });
    await waitForEngineBoot(pageGuest, 'guest-reloaded');

    let guestRejoinedCleanly = false;
    let guestRejoinMs = null;
    try {
      await pageGuest.waitForFunction(
        () =>
          window.__VITRAIL_ENGINE__?.mpStatus?.peerConnected === true &&
          (window.__VITRAIL_ENGINE__?.guestWorld?.getLastAcceptedSeq?.() ?? 0) > 0 &&
          window.__VITRAIL_ENGINE__?.physics?.blocksStacked === 2,
        undefined,
        { timeout: 10000 }
      );
      guestRejoinedCleanly = true;
      guestRejoinMs = +(performance.now() - tReload0).toFixed(1);
    } catch {
      guestRejoinedCleanly = false;
    }
    await sleep(250);
    const hostPostGuestReload = await readMatchState(pageHost);
    const guestPostReload = await readMatchState(pageGuest);
    const postReloadCompare = compareBlockPosesMm(
      hostPostGuestReload.blocks,
      guestPostReload.blocks
    );

    result.guestMidMatchF5Reload = {
      guestUrlBeforeReload,
      guestRejoinedCleanly,
      guestRejoinMs,
      blocksStackedRestored: guestPostReload.blocksStacked,
      currentPlayerRestored: guestPostReload.currentPlayer,
      roomSeedMatch: guestPostReload.mpRoomSeed === hostPostGuestReload.mpRoomSeed,
      swatchParity: `${postReloadCompare.swatchMatches}/${postReloadCompare.totalBlocks}`,
      poseErrorAfterRejoinMm: postReloadCompare.posErrorMm,
      guestTransportAfterReload: guestPostReload.mpStatus?.transport,
    };

    // -----------------------------------------------------------------------
    // Step 6: Full 300+ Shard Collapse + Winner/Loser Parity + Guest & Host Rematch
    // -----------------------------------------------------------------------
    // Trigger authoritative high-energy collapse on Host so >300 Voronoi shards spawn and settle.
    // On random roomSeeds where T5 asymmetric collapse leaves >=20 intact survivor blocks (~250-295 shards),
    // fire up to 2 follow-up impulses so shards.length >= 300 deterministically on 100% of seeds.
    const guestAudioBeforeCollapse = await readAudioCounts(pageGuest);
    const tCollapseAudioStart = performance.now();
    await pageHost.evaluate(() => {
      const eng = window.__VITRAIL_ENGINE__;
      eng.physics.triggerSandboxCollapseImpulse();
    });
    for (let extraImpulse = 0; extraImpulse < 2; extraImpulse++) {
      await sleep(750);
      const currentHostShards = await pageHost.evaluate(
        () => window.__VITRAIL_ENGINE__.physics.shards.length
      );
      if (currentHostShards >= 305) break;
      await pageHost.evaluate(() => {
        window.__VITRAIL_ENGINE__.physics.triggerSandboxCollapseImpulse();
      });
    }

    const [peakHostShards, peakGuestShards] = await Promise.all([
      pageHost
        .waitForFunction(
          () =>
            window.__VITRAIL_ENGINE__.physics.turnPhase === 'COLLAPSED' &&
            window.__VITRAIL_ENGINE__.physics.shards.length >= 300,
          undefined,
          { timeout: 12000 }
        )
        .then(() => pageHost.evaluate(() => window.__VITRAIL_ENGINE__.physics.shards.length)),
      pageGuest
        .waitForFunction(
          () =>
            window.__VITRAIL_ENGINE__.physics.turnPhase === 'COLLAPSED' &&
            window.__VITRAIL_ENGINE__.physics.shards.length >= 300,
          undefined,
          { timeout: 14000 }
        )
        .then(() => pageGuest.evaluate(() => window.__VITRAIL_ENGINE__.physics.shards.length)),
    ]);

    // Uncontaminated Settle Gate: wait until Host's 300+ Voronoi shard heap comes to complete rest
    // (off-rim shards past the 43 cm circular edge retired + 0.05 mm stability across 250 ms),
    // then allow 900 ms for round-robin `mpShardCursor` (120/pkt @ 30 Hz) + Cloudflare DO RTT + `tau=36ms` slerp convergence.
    let prevHostShards = await pageHost.evaluate(() =>
      window.__VITRAIL_ENGINE__.physics.shards.map((s) => ({
        id: s.id,
        p: [s.position[0], s.position[1], s.position[2]],
      }))
    );
    for (let settlePoll = 0; settlePoll < 22; settlePoll++) {
      await sleep(250);
      const curHostShards = await pageHost.evaluate(() =>
        window.__VITRAIL_ENGINE__.physics.shards.map((s) => ({
          id: s.id,
          p: [s.position[0], s.position[1], s.position[2]],
        }))
      );
      if (settlePoll >= 7 && curHostShards.length === prevHostShards.length) {
        const prevMap = new Map(prevHostShards.map((s) => [s.id, s.p]));
        let maxStepMm = 0;
        let allMatched = true;
        for (const s of curHostShards) {
          const pp = prevMap.get(s.id);
          if (!pp) {
            allMatched = false;
            break;
          }
          const dMm =
            Math.hypot(s.p[0] - pp[0], s.p[1] - pp[1], s.p[2] - pp[2]) * SIM_UNIT_TO_MM;
          if (dMm > maxStepMm) maxStepMm = dMm;
        }
        if (allMatched && maxStepMm < 0.05) {
          break;
        }
      }
      prevHostShards = curHostShards;
    }
    await sleep(900);
    const collapseWindowSec = Math.max(1.0, (performance.now() - tCollapseAudioStart) / 1000);

    const guestAudioAfterCollapse = await readAudioCounts(pageGuest);
    const hostCollapsed = await readMatchState(pageHost);
    const guestCollapsed = await readMatchState(pageGuest);
    const shardCompare = compareShardHeapsMm(hostCollapsed.shards, guestCollapsed.shards);

    const guestCollapseAudioDelta = diffAudioCounts(
      guestAudioBeforeCollapse,
      guestAudioAfterCollapse
    );
    const collapseTransientChatterCount =
      (guestCollapseAudioDelta.wobbleRock ?? 0) + (guestCollapseAudioDelta.shardImpacts ?? 0);
    const collapseTotalTransientCount =
      collapseTransientChatterCount + (guestCollapseAudioDelta.glassImpact ?? 0);

    result.collapse300PlusShards = {
      hostTurnPhase: hostCollapsed.turnPhase,
      guestTurnPhase: guestCollapsed.turnPhase,
      hostCollapseReason: hostCollapsed.collapseReason,
      guestCollapseReason: guestCollapsed.collapseReason,
      hostMonumentTitle: hostCollapsed.hud.monumentTitle,
      guestMonumentTitle: guestCollapsed.hud.monumentTitle,
      hostOnlineStatus: hostCollapsed.hud.onlineStatusText,
      guestOnlineStatus: guestCollapsed.hud.onlineStatusText,
      peakHostShards,
      peakGuestShards,
      collapseWindowSec: +collapseWindowSec.toFixed(2),
      shardSync: shardCompare,
      guestCollapseAudioDelta,
      guestCollapseTransientChatterHz: +(
        collapseTransientChatterCount / collapseWindowSec
      ).toFixed(2),
      guestCollapseTotalTransientHz: +(collapseTotalTransientCount / collapseWindowSec).toFixed(2),
    };

    // 6b. Guest clicks "Rematch Stained-Glass Tower" (`button.keystone-cta`)
    const seedBeforeGuestRematch = hostCollapsed.mpRoomSeed;
    await pageGuest.bringToFront();
    await pageGuest.locator('button.keystone-cta').click();

    await Promise.all([
      pageHost.waitForFunction(
        (prevSeed) =>
          window.__VITRAIL_ENGINE__.physics.turnPhase === 'SELECT_BLOCK' &&
          window.__VITRAIL_ENGINE__.physics.shards.length === 0 &&
          (window.__VITRAIL_ENGINE__.mpRoomSeed >>> 0) !== (prevSeed >>> 0),
        seedBeforeGuestRematch,
        { timeout: 8000 }
      ),
      pageGuest.waitForFunction(
        (prevSeed) =>
          window.__VITRAIL_ENGINE__.physics.turnPhase === 'SELECT_BLOCK' &&
          window.__VITRAIL_ENGINE__.physics.shards.length === 0 &&
          (window.__VITRAIL_ENGINE__.mpRoomSeed >>> 0) !== (prevSeed >>> 0),
        seedBeforeGuestRematch,
        { timeout: 8000 }
      ),
    ]);
    await sleep(300);
    const hostAfterGuestRematch = await readMatchState(pageHost);
    const guestAfterGuestRematch = await readMatchState(pageGuest);
    const guestRematchCompare = compareBlockPosesMm(
      hostAfterGuestRematch.blocks,
      guestAfterGuestRematch.blocks
    );

    // 6c. Host clicks "Rebuild" sill button (`button[aria-label="Rebuild 54-Prism Stained-Glass Tower"]`)
    const seedBeforeHostRematch = hostAfterGuestRematch.mpRoomSeed;
    await pageHost.bringToFront();
    await pageHost.locator('button[aria-label="Rebuild 54-Prism Stained-Glass Tower"]').click();

    await Promise.all([
      pageHost.waitForFunction(
        (prevSeed) => (window.__VITRAIL_ENGINE__.mpRoomSeed >>> 0) !== (prevSeed >>> 0),
        seedBeforeHostRematch,
        { timeout: 8000 }
      ),
      pageGuest.waitForFunction(
        (prevSeed) => (window.__VITRAIL_ENGINE__.mpRoomSeed >>> 0) !== (prevSeed >>> 0),
        seedBeforeHostRematch,
        { timeout: 8000 }
      ),
    ]);
    await sleep(300);
    const hostAfterHostRematch = await readMatchState(pageHost);
    const guestAfterHostRematch = await readMatchState(pageGuest);
    const hostRematchCompare = compareBlockPosesMm(
      hostAfterHostRematch.blocks,
      guestAfterHostRematch.blocks
    );

    result.rematches = {
      guestTriggeredRematch: {
        newSeed: hostAfterGuestRematch.mpRoomSeed,
        seedMatch: hostAfterGuestRematch.mpRoomSeed === guestAfterGuestRematch.mpRoomSeed,
        hostShards: hostAfterGuestRematch.shardCount,
        guestShards: guestAfterGuestRematch.shardCount,
        swatchParity: `${guestRematchCompare.swatchMatches}/${guestRematchCompare.totalBlocks}`,
        poseErrorMm: guestRematchCompare.posErrorMm,
      },
      hostTriggeredRematch: {
        newSeed: hostAfterHostRematch.mpRoomSeed,
        seedMatch: hostAfterHostRematch.mpRoomSeed === guestAfterHostRematch.mpRoomSeed,
        swatchParity: `${hostRematchCompare.swatchMatches}/${hostRematchCompare.totalBlocks}`,
        poseErrorMm: hostRematchCompare.posErrorMm,
      },
    };

    // -----------------------------------------------------------------------
    // Step 7: Adversarial 3rd-Peer URL Open + Host `F5` Reload Test
    // -----------------------------------------------------------------------
    const browserThird = await chromium.launch({
      headless: false,
      args: [
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--enable-dawn-features=allow_unsafe_apis',
        '--disable-backgrounding-occluded-windows',
        '--window-position=500,120',
        '--window-size=900,640',
      ],
    });
    try {
      const ctxThird = await browserThird.newContext({ viewport: { width: 900, height: 600 }, deviceScaleFactor: 1 });
      const pageThird = await ctxThird.newPage();

      // Count connection-state / peerConnected transitions on Guest1 while 3rd peer opens invite URL!
      await pageGuest.evaluate(() => {
        window.__GUEST1_STATUS_FLIPS__ = 0;
        let prev = window.__VITRAIL_ENGINE__.mpStatus?.peerConnected;
        window.__GUEST1_FLIP_TIMER__ = setInterval(() => {
          const cur = window.__VITRAIL_ENGINE__.mpStatus?.peerConnected;
          if (cur !== prev) {
            window.__GUEST1_STATUS_FLIPS__++;
            prev = cur;
          }
        }, 50);
      });

      const thirdJoinUrl =
        guestJoinStyle === 'url-auto'
          ? guestTargetUrl
          : `${guestTargetUrl}&room=${encodeURIComponent(roomCode)}`;
      await pageThird.goto(thirdJoinUrl, { waitUntil: 'domcontentloaded' });
      await waitForEngineBoot(pageThird, 'intruder-3rd');
      await sleep(2800);

      const guest1Flips = await pageGuest.evaluate(() => {
        clearInterval(window.__GUEST1_FLIP_TIMER__);
        return window.__GUEST1_STATUS_FLIPS__;
      });
      const thirdState = await readMatchState(pageThird);
      const guest1StateDuringIntrusion = await readMatchState(pageGuest);

      result.thirdPeerIntrusion = {
        thirdPeerUrl: thirdJoinUrl,
        thirdPeerConnectedHijackedRoom: Boolean(thirdState.mpStatus?.peerConnected),
        thirdPeerConnectionState: thirdState.mpStatus?.connectionState,
        thirdPeerErrorBanner: thirdState.hud.errorBanner,
        legitimateGuest1StillConnected: Boolean(guest1StateDuringIntrusion.mpStatus?.peerConnected),
        legitimateGuest1ConnectionState: guest1StateDuringIntrusion.mpStatus?.connectionState,
        legitimateGuest1PeerConnectedTransitionsIn2_8s: guest1Flips,
      };
    } finally {
      await browserThird.close();
    }

    // -----------------------------------------------------------------------
    // Step 8: Host `F5` Reload on Host's `?room=<CODE>` URL (`vitrail_mp_tab_session_v1` reclaim!)
    // -----------------------------------------------------------------------
    await sleep(400);
    const hostUrlPreF5 = pageHost.url();
    const tHostF5 = performance.now();
    await pageHost.reload({ waitUntil: 'domcontentloaded' });
    await waitForEngineBoot(pageHost, 'host-after-F5');

    let hostReclaimedCleanly = false;
    let hostReclaimMs = null;
    try {
      await Promise.all([
        pageHost.waitForFunction(
          () =>
            window.__VITRAIL_ENGINE__?.mpRole === 'host' &&
            window.__VITRAIL_ENGINE__?.mpStatus?.peerConnected === true,
          undefined,
          { timeout: 12000 }
        ),
        pageGuest.waitForFunction(
          () =>
            window.__VITRAIL_ENGINE__?.mpStatus?.peerConnected === true &&
            (window.__VITRAIL_ENGINE__?.guestWorld?.getLastAcceptedSeq?.() ?? 0) > 0,
          undefined,
          { timeout: 12000 }
        ),
      ]);
      hostReclaimedCleanly = true;
      hostReclaimMs = +(performance.now() - tHostF5).toFixed(1);
    } catch {
      hostReclaimedCleanly = false;
    }
    await sleep(300);
    const hostAfterF5 = await readMatchState(pageHost);
    const guestAfterHostF5 = await readMatchState(pageGuest);
    const postHostF5Compare = compareBlockPosesMm(hostAfterF5.blocks, guestAfterHostF5.blocks);

    result.hostMidMatchF5Reload = {
      hostUrlPreF5,
      hostReclaimedCleanly,
      hostReclaimMs,
      hostRoleAfterF5: hostAfterF5.mpRole,
      hostPeerConnectedAfterF5: hostAfterF5.mpStatus?.peerConnected,
      hostConnectionStateAfterF5: hostAfterF5.mpStatus?.connectionState,
      guestPeerConnectedAfterHostF5: guestAfterHostF5.mpStatus?.peerConnected,
      guestConnectionStateAfterHostF5: guestAfterHostF5.mpStatus?.connectionState,
      roomSeedMatch: hostAfterF5.mpRoomSeed === guestAfterHostF5.mpRoomSeed,
      swatchParity: `${postHostF5Compare.swatchMatches}/${postHostF5Compare.totalBlocks}`,
      poseErrorAfterHostF5Mm: postHostF5Compare.posErrorMm,
    };

    result.pageErrors = pageErrors;
    result.stageGates = evaluateModeGates(result);
    return result;
  } finally {
    await browserGuest.close();
    await browserHost.close();
  }
}

function evaluateModeGates(r) {
  const failures = [];
  const stages = {
    stage1_connectionAndSwatchParity: true,
    stage2_turn1HostPullRotatePlace: true,
    stage3_turn2GuestPullRotatePlace: true,
    stage4_hostTabOcclusion30HzPump: true,
    stage5_guestMidMatchF5Reload: true,
    stage6_collapse300ShardsAudioAndRematches: true,
    stage7_thirdPeerIntrusionRejected1008: true,
    stage8_hostMidMatchF5RoleReclaim: true,
  };

  const fail = (stageKey, msg) => {
    stages[stageKey] = false;
    failures.push(`${r.modeId} [${stageKey}]: ${msg}`);
  };

  // Stage 1: Connection, transport, seed & 54/54 swatch parity
  if (
    r.connection?.hostTransport !== r.expectedTransport ||
    r.connection?.guestTransport !== r.expectedTransport ||
    !r.connection?.roomSeedMatch ||
    r.connection?.swatchParity !== '54/54' ||
    (r.connection?.initialPoseErrorMm?.max ?? 999) >= 2.0
  ) {
    fail(
      'stage1_connectionAndSwatchParity',
      `transport=${r.connection?.hostTransport}/${r.connection?.guestTransport} (expected ${r.expectedTransport}), swatch=${r.connection?.swatchParity}, maxErr=${r.connection?.initialPoseErrorMm?.max} mm`
    );
  }

  // Stage 2: Turn 1 Host tap, pull, rotate, place & handoff to Guest
  if (
    !r.turn1Host?.tapTestKeptSelectPhase ||
    (r.turn1Host?.hostTapAudioDelta ?? 0) < 1 ||
    (r.turn1Host?.guestTapAudioDelta ?? 0) < 1 ||
    (r.turn1Host?.settled54BlockErrorMm?.max ?? 999) >= 2.0 ||
    r.turn1Host?.turnHandoffToGuest?.hostCurrentPlayer !== 1 ||
    r.turn1Host?.turnHandoffToGuest?.guestCurrentPlayer !== 1
  ) {
    fail(
      'stage2_turn1HostPullRotatePlace',
      `tap=${r.turn1Host?.hostTapAudioDelta}/${r.turn1Host?.guestTapAudioDelta}, settledMax=${r.turn1Host?.settled54BlockErrorMm?.max} mm, handoff=${r.turn1Host?.turnHandoffToGuest?.guestCurrentPlayer}`
    );
  }

  // Stage 3: Turn 2 Guest tap, pull (`0` reversals), rotate, place & handoff to Host
  if (
    (r.turn2Guest?.hostTapAudioDelta ?? 0) < 1 ||
    (r.turn2Guest?.guestTapAudioDelta ?? 0) < 1 ||
    (r.turn2Guest?.pullTelemetry?.frameToFrameReversalCount ?? 99) !== 0 ||
    (r.turn2Guest?.settled54BlockErrorMm?.max ?? 999) >= 2.0 ||
    r.turn2Guest?.turnHandoffBackToHost?.blocksStackedHost !== 2 ||
    r.turn2Guest?.turnHandoffBackToHost?.blocksStackedGuest !== 2
  ) {
    fail(
      'stage3_turn2GuestPullRotatePlace',
      `tap=${r.turn2Guest?.hostTapAudioDelta}/${r.turn2Guest?.guestTapAudioDelta}, reversals=${r.turn2Guest?.pullTelemetry?.frameToFrameReversalCount}, settledMax=${r.turn2Guest?.settled54BlockErrorMm?.max} mm`
    );
  }

  // Stage 4: Occluded Host (`foregroundRafHaltedVerified === true`) 30 Hz background pump
  if (
    !r.hostTabOcclusion?.foregroundRafHaltedVerified ||
    (r.hostTabOcclusion?.earlyOccludedSnapshotRateHz ?? 0) < 20.0 ||
    (r.hostTabOcclusion?.post10SecBackgroundSnapshotRateHz !== null &&
      r.hostTabOcclusion.post10SecBackgroundSnapshotRateHz < 20.0)
  ) {
    fail(
      'stage4_hostTabOcclusion30HzPump',
      `rafHalted=${r.hostTabOcclusion?.foregroundRafHaltedVerified}, earlyHz=${r.hostTabOcclusion?.earlyOccludedSnapshotRateHz}, post10sHz=${r.hostTabOcclusion?.post10SecBackgroundSnapshotRateHz}`
    );
  }

  // Stage 5: Guest mid-match `F5` reload (`blocksStacked === 2`)
  if (
    !r.guestMidMatchF5Reload?.guestRejoinedCleanly ||
    r.guestMidMatchF5Reload?.blocksStackedRestored !== 2 ||
    !r.guestMidMatchF5Reload?.roomSeedMatch ||
    r.guestMidMatchF5Reload?.swatchParity !== '54/54' ||
    (r.guestMidMatchF5Reload?.poseErrorAfterRejoinMm?.max ?? 999) >= 2.0
  ) {
    fail(
      'stage5_guestMidMatchF5Reload',
      `rejoined=${r.guestMidMatchF5Reload?.guestRejoinedCleanly}, stacked=${r.guestMidMatchF5Reload?.blocksStackedRestored}, swatch=${r.guestMidMatchF5Reload?.swatchParity}, maxErr=${r.guestMidMatchF5Reload?.poseErrorAfterRejoinMm?.max} mm`
    );
  }

  // Stage 6: 300+ Voronoi shard collapse + `<= 25 Hz` collapse transient chatter + Guest & Host rematches
  const sh = r.collapse300PlusShards?.shardSync;
  if (
    (r.collapse300PlusShards?.peakHostShards ?? 0) < 300 ||
    (r.collapse300PlusShards?.peakGuestShards ?? 0) < 300 ||
    (sh?.hostShardCount ?? 0) < 180 ||
    sh?.guestShardCount !== sh?.hostShardCount ||
    sh?.matchedIds !== sh?.hostShardCount ||
    (sh?.posErrorMm?.p95 ?? 999) >= 2.0 ||
    (r.collapse300PlusShards?.guestCollapseTransientChatterHz ?? 999) > 25.0 ||
    !r.rematches?.guestTriggeredRematch?.seedMatch ||
    r.rematches?.guestTriggeredRematch?.swatchParity !== '54/54' ||
    !r.rematches?.hostTriggeredRematch?.seedMatch ||
    r.rematches?.hostTriggeredRematch?.swatchParity !== '54/54'
  ) {
    fail(
      'stage6_collapse300ShardsAudioAndRematches',
      `peakShards=${r.collapse300PlusShards?.peakGuestShards}/${r.collapse300PlusShards?.peakHostShards}, settled=${sh?.guestShardCount}/${sh?.hostShardCount} (p95=${sh?.posErrorMm?.p95} mm), chatterHz=${r.collapse300PlusShards?.guestCollapseTransientChatterHz} Hz, rematches=${r.rematches?.guestTriggeredRematch?.swatchParity}/${r.rematches?.hostTriggeredRematch?.swatchParity}`
    );
  }

  // Stage 7: Adversarial 3rd-peer URL intrusion rejected with `1008 Room full (2/2 players)` & Guest 1 undisturbed
  if (
    r.thirdPeerIntrusion?.thirdPeerConnectedHijackedRoom !== false ||
    r.thirdPeerIntrusion?.thirdPeerConnectionState !== 'error' ||
    !String(r.thirdPeerIntrusion?.thirdPeerErrorBanner ?? '').includes('Room full') ||
    r.thirdPeerIntrusion?.legitimateGuest1StillConnected !== true ||
    r.thirdPeerIntrusion?.legitimateGuest1PeerConnectedTransitionsIn2_8s !== 0
  ) {
    fail(
      'stage7_thirdPeerIntrusionRejected1008',
      `hijacked=${r.thirdPeerIntrusion?.thirdPeerConnectedHijackedRoom}, state=${r.thirdPeerIntrusion?.thirdPeerConnectionState}, banner=${JSON.stringify(r.thirdPeerIntrusion?.thirdPeerErrorBanner)}, guest1Alive=${r.thirdPeerIntrusion?.legitimateGuest1StillConnected}, flips=${r.thirdPeerIntrusion?.legitimateGuest1PeerConnectedTransitionsIn2_8s}`
    );
  }

  // Stage 8: Host mid-match `F5` reload reclaims `host` role and reconnects with Guest
  if (
    !r.hostMidMatchF5Reload?.hostReclaimedCleanly ||
    r.hostMidMatchF5Reload?.hostRoleAfterF5 !== 'host' ||
    r.hostMidMatchF5Reload?.hostPeerConnectedAfterF5 !== true ||
    r.hostMidMatchF5Reload?.guestPeerConnectedAfterHostF5 !== true ||
    !r.hostMidMatchF5Reload?.roomSeedMatch ||
    r.hostMidMatchF5Reload?.swatchParity !== '54/54'
  ) {
    fail(
      'stage8_hostMidMatchF5RoleReclaim',
      `role=${r.hostMidMatchF5Reload?.hostRoleAfterF5}, hostConn=${r.hostMidMatchF5Reload?.hostPeerConnectedAfterF5}, guestConn=${r.hostMidMatchF5Reload?.guestPeerConnectedAfterHostF5}, swatch=${r.hostMidMatchF5Reload?.swatchParity}`
    );
  }

  if ((r.pageErrors?.host?.length ?? 0) > 0 || (r.pageErrors?.guest?.length ?? 0) > 0) {
    failures.push(`${r.modeId} [pageErrors]: ${JSON.stringify(r.pageErrors)}`);
  }

  return {
    all8StagesPassed: failures.length === 0,
    stages,
    gateFailures: failures,
  };
}

async function main() {
  const argv = process.argv.slice(2);
  let selectedMode = 'all';
  let explicitUrl = null;
  let outPath = '/tmp/vitrail-swarm-logs/e2e-two-browser-mp.json';
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--mode' && argv[i + 1]) {
      selectedMode = argv[++i].toLowerCase();
    } else if (argv[i] === '--url' && argv[i + 1]) {
      explicitUrl = argv[++i];
    } else if (argv[i] === '--out' && argv[i + 1]) {
      outPath = argv[++i];
    }
  }

  const tStartAll = performance.now();
  const { baseUrl, proc: viteProc, headers } = await ensureViteDevServer(explicitUrl);
  const { chromium } = await resolvePlaywright();

  let exitCode = 1;
  try {
    let modeA = null;
    let modeB = null;

    if (selectedMode === 'all' || selectedMode === 'a') {
      modeA = await runModeLifecycle(chromium, {
        modeId: 'ModeA_PublicPeerJsWebRTC',
        label: 'Real Public 0.peerjs.com Signaling + Native WebRTC RTCDataChannel (?relay=none)',
        hostBootUrl: `${baseUrl}/?relay=none`,
        guestJoinStyle: 'url-auto',
        guestBootUrlBuilder: (code) => `${baseUrl}/?relay=none&room=${encodeURIComponent(code)}`,
        expectedTransport: 'webrtc-p2p',
      });
    }

    if (selectedMode === 'all' || selectedMode === 'b') {
      if (modeA) await sleep(600);
      modeB = await runModeLifecycle(chromium, {
        modeId: 'ModeB_LiveCloudflareRelay',
        label:
          'Live Internet Cloudflare Durable Object WS Relay (?forceRelay=1&relay=wss://vitrail-rooms.andrew9687.workers.dev/__vitrail_mp)',
        hostBootUrl: `${baseUrl}/?forceRelay=1&relay=${encodeURIComponent(LIVE_CF_RELAY)}`,
        guestJoinStyle: 'manual-input',
        guestBootUrlBuilder: (_code) =>
          `${baseUrl}/?forceRelay=1&relay=${encodeURIComponent(LIVE_CF_RELAY)}`,
        expectedTransport: 'ws-relay',
      });
    }

    const allGateFailures = [
      ...(modeA?.stageGates?.gateFailures ?? []),
      ...(modeB?.stageGates?.gateFailures ?? []),
    ];

    const summary = {
      ts: new Date().toISOString(),
      baseUrl,
      totalElapsedSec: +((performance.now() - tStartAll) / 1000).toFixed(2),
      allStagesPassed: allGateFailures.length === 0,
      gateFailures: allGateFailures,
      viteHeaders: headers,
      modeA,
      modeB,
    };

    if (outPath) {
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
    }
    console.log(JSON.stringify(summary, null, 2));
    exitCode = allGateFailures.length === 0 ? 0 : 1;
  } finally {
    if (viteProc?.pid) {
      try {
        process.kill(-viteProc.pid, 'SIGTERM');
      } catch {}
    }
  }
  process.exit(exitCode);
}

main().catch((err) => {
  console.error('E2E TWO-BROWSER MP FAILED:', err);
  process.exit(1);
});

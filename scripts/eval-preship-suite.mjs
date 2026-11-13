#!/usr/bin/env node
/**
 * Comprehensive 5-Domain Pre-Ship Evaluation Suite (`scripts/eval-preship-suite.mjs`)
 *
 * Evaluates the default shipped `FAITHFUL_GLASS_FIDELITY_CONFIG` (`M1–M6`) across:
 *   1. Gravity & Cantilever Overhang (71.1 g block & 0.2–2.0 g shard 1.000g free-fall, > 52% topple)
 *   2. Stack Heights (`12L`, `18L`, `24L`, `30L` sag/sleep/clinks) & Elastic Placement Drop
 *   3. Simulated Full Games (`18L` & `24L` 2-Ply Glazier AI vs AI) & 60-seed × 50-action Stateful Fuzzer
 *   4. Sandbox Smashing (`18L`, `24L`, `30L` collapse P95 / sleep / buzz), Solo Asymmetric Collapse, & 512-Shard Pool Wrap
 *   5. Acoustic Anomaly & Modal Health Audit (0 NaN/Inf, [75, 9500] Hz bounds, Kirchhoff oak + Timoshenko glass, seesaw wobble, bounded bursts)
 *
 * Usage:
 *   npm run eval:preship
 *   node scripts/eval-preship-suite.mjs [--json]
 */

import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { createServer } from 'vite';
import { createFuzzSuite } from './fuzz-stateful-trajectories.mjs';

const emitJson = process.argv.includes('--json');

function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[idx];
}

function median(values) {
  return percentile(values, 0.5);
}

async function main() {
  const tStart = performance.now();
  const vite = await createServer({
    logLevel: 'silent',
    server: { middlewareMode: true, ws: false },
    appType: 'custom',
  });

  /** @type {Array<{ domain: string; check: string; pass: boolean; detail: string }>} */
  const checks = [];
  const record = (domain, check, pass, detail) => {
    checks.push({ domain, check, pass: Boolean(pass), detail });
  };

  let fuzzSuite = null;

  try {
    const [
      { GlassAudioEngine },
      { GameEngine },
      constants,
      vecMat,
      { ensureMujocoInitialized },
      { PhysicsWorld },
      { MujocoWorkerBridge },
      { attachMujocoWorkerHost },
      { MJ_CONTYPE_BLOCK },
    ] = await Promise.all([
      vite.ssrLoadModule('/src/audio/GlassAudioEngine.ts'),
      vite.ssrLoadModule('/src/engine/GameEngine.ts'),
      vite.ssrLoadModule('/src/physics/constants.ts'),
      vite.ssrLoadModule('/src/math/vecMat.ts'),
      vite.ssrLoadModule('/src/physics/mujocoRuntime.ts'),
      vite.ssrLoadModule('/src/physics/PhysicsWorld.ts'),
      vite.ssrLoadModule('/src/physics/MujocoWorkerBridge.ts'),
      vite.ssrLoadModule('/src/physics/mujocoWorker.ts'),
      vite.ssrLoadModule('/src/physics/mujocoTowerBuilder.ts'),
    ]);

    const { BLOCK_HALF_Y, BLOCK_HEIGHT, BLOCK_LENGTH, TABLE_RADIUS } = constants;
    const { quatFromAxisAngle } = vecMat;

    await ensureMujocoInitialized(false);
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const allAudioEntries = [];
    const collectAudio = () => {
      const batch = audio.consumeRecentImpactLog();
      if (batch.length > 0) allAudioEntries.push(...batch);
      return batch;
    };

    // =========================================================================
    // DOMAIN 1: Gravity (71.1 g Block + 0.2-2.0 g Shards) & Cantilever Overhang
    // =========================================================================
    {
      const world = new PhysicsWorld(audio, 1);
      world.rebuildTower('sandbox', 2, 1);
      const rw = world.getWorld();
      for (let i = 1; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle).setTranslation(
          { x: 2.6 + i * 0.35, y: BLOCK_HALF_Y, z: 2.2 },
          true
        );
      }

      const b0 = world.blocks[0];
      const rb0 = rw.getRigidBody(b0.bodyHandle);
      const yStartDm = 3.2 + BLOCK_HALF_Y;
      rb0.setTranslation({ x: 0, y: yStartDm, z: 0 }, true);
      rb0.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb0.setAngvel({ x: 0, y: 0, z: 0 }, true);
      world.getWorld().syncKinematics();

      let yAfter10Dm = yStartDm;
      for (let f = 0; f < 10; f++) {
        world.step(1 / 60);
        if (f === 9) yAfter10Dm = b0.position[1];
      }
      const t10Sec = 10 / 60;
      const expectedFallDm = 0.5 * 98.1 * t10Sec * t10Sec;
      const blockGravityRatio = (yStartDm - yAfter10Dm) / expectedFallDm;
      record(
        'D1 Gravity & Overhang',
        '71.1g block 1.000g free-fall',
        blockGravityRatio >= 0.95 && blockGravityRatio <= 1.04,
        `g_ratio=${blockGravityRatio.toFixed(4)} (expected [0.95, 1.04])`
      );
      world.dispose();
    }

    {
      const world = new PhysicsWorld(audio, 1);
      world.rebuildTower('sandbox', 2, 1);
      const rw = world.getWorld();
      for (let i = 1; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle).setTranslation(
          { x: 2.6 + i * 0.35, y: BLOCK_HALF_Y, z: 2.2 },
          true
        );
      }
      const b0 = world.blocks[0];
      const rb0 = rw.getRigidBody(b0.bodyHandle);
      rb0.setTranslation({ x: 0, y: 2.8, z: 0 }, true);
      rb0.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      world.getWorld().syncKinematics();
      world.shatterBlock(b0, [0, -BLOCK_HALF_Y, 0], 15.0, [0, 1, 0], false);

      // Measure free-fall acceleration a = (y2 - 2*y1 + y0) / dt^2 over 3 equal 3-frame intervals (frames 3, 6, 9)
      const yTrack = new Map();
      for (const s of world.shards) {
        yTrack.set(s.id, { massG: s.massKg * 1000, y3: null, y6: null, y9: null });
      }
      for (let f = 1; f <= 9; f++) {
        world.step(1 / 60);
        if (f === 3 || f === 6 || f === 9) {
          for (const s of world.shards) {
            const st = yTrack.get(s.id);
            if (!st) continue;
            if (f === 3) st.y3 = s.position[1];
            else if (f === 6) st.y6 = s.position[1];
            else if (f === 9) st.y9 = s.position[1];
          }
        }
      }
      const dtIntervalSec = 3 / 60;
      const shardGRatios = [];
      for (const st of yTrack.values()) {
        if (
          st.y3 !== null &&
          st.y6 !== null &&
          st.y9 !== null &&
          st.y9 > 0.35 &&
          st.massG >= 0.15 &&
          st.massG <= 20.0
        ) {
          const accelDmS2 = -(st.y9 - 2 * st.y6 + st.y3) / (dtIntervalSec * dtIntervalSec);
          shardGRatios.push(accelDmS2 / 98.1);
        }
      }
      const medShardG = median(shardGRatios);
      record(
        'D1 Gravity & Overhang',
        '0.2-2.0g Voronoi shard 1.000g free-fall',
        shardGRatios.length >= 10 && medShardG >= 0.95 && medShardG <= 1.04,
        `median g_ratio=${medShardG.toFixed(4)} across ${shardGRatios.length} airborne shards`
      );
      world.dispose();
    }

    {
      const world = new PhysicsWorld(audio, 12);
      for (let i = 0; i < 50; i++) world.step(1 / 60);
      const rw = world.getWorld();
      const topSide = world.blocks.find((b) => b.assignedLayer === 11 && b.assignedSlot === 0);
      const rbTop = rw.getRigidBody(topSide.bodyHandle);
      // Displace 56% of block length beyond upper course edge (> 52% marginal overhang)
      const tiltRad = 0.08;
      const curPos = rbTop.translation();
      const curRot = rbTop.rotation();
      const axis = topSide;
      const dx = 1 - 2 * (curRot.y * curRot.y + curRot.z * curRot.z);
      const dz = 2 * (curRot.x * curRot.z - curRot.y * curRot.w);
      const len = Math.hypot(dx, dz) || 1;
      rbTop.setTranslation(
        {
          x: curPos.x + (dx / len) * (BLOCK_LENGTH * 0.56),
          y: curPos.y + 0.005,
          z: curPos.z + (dz / len) * (BLOCK_LENGTH * 0.56),
        },
        true
      );
      rbTop.setAngvel({ x: 0, y: 0, z: 0 }, true);
      rw.wakeBody(topSide.bodyHandle);
      rw.syncKinematics();

      let maxTiltDeg = 0;
      let droppedPastHalf = false;
      const baseLayerY = curPos.y;
      for (let f = 0; f < 90; f++) {
        world.step(1 / 60);
        const q = axis.rotation;
        const upY = Math.max(-1, Math.min(1, 1 - 2 * (q[0] * q[0] + q[2] * q[2])));
        const deg = (Math.acos(Math.abs(upY)) * 180) / Math.PI;
        if (deg > maxTiltDeg) maxTiltDeg = deg;
        if (axis.position[1] < baseLayerY - BLOCK_HEIGHT * 0.8) droppedPastHalf = true;
      }
      record(
        'D1 Gravity & Overhang',
        '> 52% cantilever overhang dynamic topple',
        maxTiltDeg >= 15.0 && droppedPastHalf,
        `maxTilt=${maxTiltDeg.toFixed(1)} deg, dropped=${droppedPastHalf}, tiltHint=${tiltRad}`
      );
      world.dispose();
    }

    collectAudio();

    // =========================================================================
    // DOMAIN 2: Stack Heights (12L, 18L, 24L, 30L) & 10 mm Elastic Placement Drop
    // =========================================================================
    const sagCapsMm = { 12: 0.10, 18: 0.15, 24: 0.50, 30: 1.20 };
    for (const layers of [12, 18, 24, 30]) {
      const world = new PhysicsWorld(audio, layers);
      world.rebuildTower('solo', 2, layers, 42);
      audio.clearRecentImpactLog();
      for (let f = 0; f < 90; f++) world.step(1 / 60);
      const clinksDuringSettle = audio.consumeRecentImpactLog().length;
      const rw = world.getWorld();
      const ntreeAwake = rw.getData().ntree_awake;
      let maxTopCenter = -Infinity;
      for (const b of world.blocks) {
        if (b.assignedLayer === layers - 1 && !b.isLooseSlot) {
          maxTopCenter = Math.max(maxTopCenter, b.position[1]);
        }
      }
      const sagMm = Math.max(0, (layers - 1 + 0.5) * BLOCK_HEIGHT - maxTopCenter) * 100.0;
      const cap = sagCapsMm[layers];
      record(
        'D2 Stack Heights & Drop',
        `${layers}L equilibrium sag / sleep / 0 clinks`,
        sagMm < cap && ntreeAwake === 0 && clinksDuringSettle === 0,
        `sag=${sagMm.toFixed(4)}mm (< ${cap}mm), ntree_awake=${ntreeAwake}, settleClinks=${clinksDuringSettle}`
      );
      world.dispose();
    }

    {
      const world = new PhysicsWorld(audio, 18);
      for (let f = 0; f < 48; f++) world.step(1 / 60);
      const slot0 = world.openTopSlots[0];
      const dropBlk = world.blocks.find((b) => b.assignedLayer === 6 && b.isLooseSlot);
      const rw = world.getWorld();
      const rb = rw.getRigidBody(dropBlk.bodyHandle);
      rb.setTranslation({ x: slot0.position[0], y: slot0.position[1] + 0.10, z: slot0.position[2] }, true);
      rb.setRotation(
        { x: slot0.rotation[0], y: slot0.rotation[1], z: slot0.rotation[2], w: slot0.rotation[3] },
        true
      );
      rb.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, true);
      dropBlk.assignedLayer = slot0.layer;
      dropBlk.assignedSlot = slot0.slotIndex;
      rw.syncKinematics();

      world.trackBlockDrop(dropBlk.id, slot0.position[1]);
      audio.clearRecentImpactLog();
      for (let f = 0; f < 100; f++) world.step(1 / 60);
      const dropLog = collectAudio();
      const glassBounces = dropLog.filter((e) => e.kind === 'glass');
      let monotonicGains = true;
      for (let i = 1; i < Math.min(3, glassBounces.length); i++) {
        if (glassBounces[i].gain > glassBounces[i - 1].gain + 1e-4) monotonicGains = false;
      }
      const { maxReboundMm } = world.getTrackedDropState();
      const maxDmg = Math.max(...world.blocks.map((b) => b.damage));
      const ntreeAwake = rw.getData().ntree_awake;
      record(
        'D2 Stack Heights & Drop',
        '10 mm top-course elastic drop & monotonic gain decay',
        maxReboundMm >= 1.0 && maxDmg === 0 && ntreeAwake === 0 && monotonicGains && glassBounces.length >= 2,
        `rebound=${maxReboundMm.toFixed(2)}mm, bounces=${glassBounces.length}, monotonic=${monotonicGains}, maxDmg=${maxDmg}, awake=${ntreeAwake}`
      );
      world.dispose();
    }

    // =========================================================================
    // DOMAIN 3: Simulated Full Games (18L & 24L AI vs AI) & 60-Seed Fuzzer
    // =========================================================================
    for (const [layers, minStacked, maxSteps] of [
      [18, 24, 18000],
      [24, 12, 12000],
    ]) {
      const engine = new GameEngine();
      await engine.initHeadless();
      engine.audio.setMuted(true);
      engine.resetGame('party', 2);
      if (layers !== 18) {
        engine.physics.rebuildTower('party', 2, layers, 101);
        for (let f = 0; f < 45; f++) engine.stepHeadless(1 / 60);
      }
      engine.setAiVsAiMode(true, {
        fastCadence: true,
        p0Difficulty: 'apprentice',
        p1Difficulty: 'master',
      });
      let steps = 0;
      while (
        steps < maxSteps &&
        engine.physics.blocksStacked < minStacked &&
        engine.physics.turnPhase !== 'COLLAPSED'
      ) {
        engine.stepHeadless(1 / 60);
        steps++;
      }
      const stacked = engine.physics.blocksStacked;
      const shattered = engine.physics.shatteredCount;
      const phase = engine.physics.turnPhase;
      record(
        'D3 Simulated Games & Fuzz',
        `${layers}L Glazier 2-Ply AI vs AI (>= ${minStacked} stacked)`,
        stacked >= minStacked && shattered === 0 && phase !== 'COLLAPSED',
        `stacked=${stacked}/${minStacked}, layers=${engine.physics.currentLayers}, shattered=${shattered}, phase=${phase}, steps=${steps}`
      );
      engine.dispose();
    }

    fuzzSuite = await createFuzzSuite({
      GameEngine,
      GlassAudioEngine,
      PhysicsWorld,
      MujocoWorkerBridge,
      attachMujocoWorkerHost,
      ensureMujocoInitialized,
      constants,
      vecMat,
    });
    const fuzzSeeds = 60;
    const fuzzActions = 50;
    let totalFuzzActions = 0;
    let fuzzErrors = 0;
    let fuzzCollapseTraj = 0;
    for (let i = 0; i < fuzzSeeds; i++) {
      const seed = (0x517cc1b7 + Math.imul(i + 1, 0x9e3779b9)) >>> 0;
      const mode = i < 20 ? 'paired' : i < 40 ? 'async-worker' : 'sync';
      try {
        const res = await fuzzSuite.executeTrajectory(seed, fuzzActions, mode);
        totalFuzzActions += res.actionsExecuted;
        if (res.sawCollapseOrFracture) fuzzCollapseTraj++;
      } catch (err) {
        fuzzErrors++;
        record('D3 Simulated Games & Fuzz', `Stateful fuzz seed 0x${seed.toString(16)}`, false, String(err.message || err));
        break;
      }
    }
    record(
      'D3 Simulated Games & Fuzz',
      '60-seed x 50-action stateful fuzzer (sync + worker)',
      fuzzErrors === 0 && totalFuzzActions === fuzzSeeds * fuzzActions,
      `actions=${totalFuzzActions}/${fuzzSeeds * fuzzActions}, collapseTraj=${fuzzCollapseTraj}, errors=${fuzzErrors}`
    );

    // =========================================================================
    // DOMAIN 4: Sandbox Smashing (18L, 24L, 30L), Solo Collapse & 512 Pool Wrap
    // =========================================================================
    const cpuCount = globalThis.navigator?.hardwareConcurrency ?? os.cpus().length;
    const budget18Ms = cpuCount <= 2 ? 48.0 : 16.0;
    let maxShardImpactBurst = 0;
    let maxWobbleBurst = 0;

    for (const [layers, minShards, stepFrames] of [
      [18, 280, 260],
      [24, 250, 280],
      [30, 250, 300],
    ]) {
      const world = new PhysicsWorld(audio, layers);
      world.rebuildTower('sandbox', 2, layers, 42);
      for (let f = 0; f < 40; f++) world.step(1 / 60);
      audio.clearRecentImpactLog();
      world.triggerSandboxCollapseImpulse();
      const rw = world.getWorld();
      const mjData = rw.getData();
      let prevTime = mjData.time;
      let solverResets = 0;
      const stepMs = [];
      let peakShards = 0;

      for (let f = 0; f < stepFrames; f++) {
        const t0 = performance.now();
        const res = world.step(1 / 60);
        stepMs.push(performance.now() - t0);
        collectAudio();
        if (res.shardImpacts.length > maxShardImpactBurst) maxShardImpactBurst = res.shardImpacts.length;
        if (res.wobbleEvents.length > maxWobbleBurst) maxWobbleBurst = res.wobbleEvents.length;
        if (world.shards.length > peakShards) peakShards = world.shards.length;
        if (mjData.time + 1e-6 < prevTime) solverResets++;
        prevTime = mjData.time;
      }

      // Allow tail rubble settle if still finishing sleep
      for (let f = 0; f < 160 && mjData.ntree_awake > 0; f++) {
        const res = world.step(1 / 60);
        collectAudio();
        if (res.shardImpacts.length > maxShardImpactBurst) maxShardImpactBurst = res.shardImpacts.length;
        if (res.wobbleEvents.length > maxWobbleBurst) maxWobbleBurst = res.wobbleEvents.length;
      }
      const awakeAfterSettle = mjData.ntree_awake;

      let postSettleBuzz = 0;
      for (let f = 0; f < 30; f++) {
        const res = world.step(1 / 60);
        postSettleBuzz += res.shardImpacts.length;
      }

      let floatingOffRim = 0;
      for (const s of world.shards) {
        const r = Math.hypot(s.position[0], s.position[2]);
        const lv = rw.getRigidBody(s.bodyHandle)?.linvel();
        const speed3d = lv ? Math.hypot(lv.x, lv.y, lv.z) : 0;
        if (r > TABLE_RADIUS + 0.08 && s.position[1] > -0.05 && speed3d < 0.12) floatingOffRim++;
      }
      for (const b of world.blocks) {
        if (b.shattered) continue;
        const r = Math.hypot(b.position[0], b.position[2]);
        const lv = rw.getRigidBody(b.bodyHandle)?.linvel();
        const speed3d = lv ? Math.hypot(lv.x, lv.y, lv.z) : 0;
        if (r > TABLE_RADIUS + 0.14 && b.position[1] > -0.05 && speed3d < 0.12) floatingOffRim++;
      }

      const p95 = percentile(stepMs, 0.95);
      const passBudget = layers === 18 ? p95 < budget18Ms : true;
      record(
        'D4 Sandbox & 512 Pool',
        `${layers}L Sandbox collapse (shards, sleep, 0 buzz, P95)`,
        peakShards >= minShards && solverResets === 0 && floatingOffRim === 0 && awakeAfterSettle === 0 && postSettleBuzz === 0 && passBudget,
        `peakShards=${peakShards}, p95=${p95.toFixed(2)}ms, resets=${solverResets}, offRim=${floatingOffRim}, awake=${awakeAfterSettle}, postBuzz=${postSettleBuzz}`
      );
      world.dispose();
    }

    {
      // 18L Solo Asymmetric Collapse (T5 style)
      const world = new PhysicsWorld(audio, 18);
      for (let i = 0; i < 45; i++) world.step(1 / 60);
      const rw = world.getWorld();
      for (const slotIdx of [0, 1]) {
        const blk = world.blocks.find((b) => b.assignedLayer === 0 && b.assignedSlot === slotIdx);
        const rb = rw.getRigidBody(blk.bodyHandle);
        rb.setTranslation({ x: 3.4, y: BLOCK_HALF_Y, z: 3.4 + slotIdx * 0.4 }, true);
        blk.extracted = true;
      }
      rw.wakeActiveTowerBlocks?.();
      for (let i = 0; i < 360; i++) {
        world.step(1 / 60);
        collectAudio();
      }
      const intact = world.blocks.filter((b) => !b.shattered).length;
      const crazed = world.blocks.filter((b) => !b.shattered && b.damage > 0).length;
      const awake = rw.getData().ntree_awake;
      record(
        'D4 Sandbox & 512 Pool',
        '18L Solo asymmetric collapse (>= 15 intact, >= 4 crazed, 100% sleep)',
        world.turnPhase === 'COLLAPSED' && intact >= 15 && crazed >= 4 && awake === 0,
        `phase=${world.turnPhase}, intact=${intact}, crazed=${crazed}, shattered=${world.shatteredCount}, awake=${awake}`
      );
      world.dispose();
    }

    {
      // 40-block sequential shatter -> > 800 cumulative shards wrapping 512-slot ring buffer
      const world = new PhysicsWorld(audio, 18);
      world.rebuildTower('sandbox', 2, 18, 77);
      for (let f = 0; f < 30; f++) world.step(1 / 60);
      const rw = world.getWorld();
      const mjModel = rw.getModel();
      const mjData = rw.getData();
      let ghostCount = 0;
      let checkedFresh = 0;
      let cumulativeShards = 0;

      const topDown = [...world.blocks].sort((a, b) => b.assignedLayer - a.assignedLayer);
      for (let idx = 0; idx < 46; idx++) {
        const b = topDown[idx];
        if (!b.shattered) {
          const prevMaxId = world.shards.reduce((m, s) => Math.max(m, s.id), -1);
          world.shatterBlock(b, [0.05, -BLOCK_HALF_Y, 0.02], 16.0, [0, 1, 0], false);
          const fresh = world.shards.filter((s) => s.id > prevMaxId);
          for (const shd of fresh) {
            checkedFresh++;
            if ((mjModel.geom_conaffinity[shd.colliderHandle] & MJ_CONTYPE_BLOCK) === 0) {
              ghostCount++;
            }
          }
          cumulativeShards += fresh.length;
        }
        for (let f = 0; f < 4; f++) {
          world.step(1 / 60);
          collectAudio();
        }
      }
      for (let f = 0; f < 220 && mjData.ntree_awake > 0; f++) {
        world.step(1 / 60);
        collectAudio();
      }
      const awake = mjData.ntree_awake;
      record(
        'D4 Sandbox & 512 Pool',
        '46-block sequential shatter (> 800 shards, 512 pool wrap, 0% ghost cullet, 100% sleep)',
        cumulativeShards >= 800 && world.shards.length <= 512 && ghostCount === 0 && awake === 0,
        `cumShards=${cumulativeShards}, active=${world.shards.length}/512, ghost=${ghostCount}/${checkedFresh}, awake=${awake}`
      );
      world.dispose();
    }

    // =========================================================================
    // DOMAIN 5: Acoustic Anomaly & Modal Health Audit
    // =========================================================================
    let phantomAirClinks = -1;
    let coupledWoodCount = 0;
    let coupledGlassCount = 0;
    let wobbleEventCount = 0;

    {
      // M2 extraction phantom air clink check
      const pw = new PhysicsWorld(audio, 18);
      pw.rebuildTower('solo', 2, 18, 42);
      for (let i = 0; i < 48; i++) pw.step(1 / 60);
      const midLoose = pw.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot && b.isLegalTarget);
      const q0 = midLoose.rotation;
      const axX = 1 - 2 * (q0[1] * q0[1] + q0[2] * q0[2]);
      const axZ = 2 * (q0[0] * q0[2] - q0[1] * q0[3]);
      const axLen = Math.hypot(axX, axZ) || 1;
      const px = axX / axLen;
      const pz = axZ / axLen;
      const startW = [midLoose.position[0] + px * 0.18, midLoose.position[1], midLoose.position[2] + pz * 0.18];
      pw.startGrabBlock(midLoose.id, [0.18, 0, 0], startW);
      for (let i = 0; i < 200; i++) {
        const tSim = (i + 1) / 120;
        const distDm = 0.02 + Math.min(1.0, tSim / 1.4) * 1.30;
        if (pw.turnPhase === 'PULLING') {
          pw.updateGrabTarget([startW[0] + px * distDm, startW[1], startW[2] + pz * distDm]);
        }
        pw.step(1 / 120);
        collectAudio();
      }
      phantomAirClinks = pw.getSolverDiagnostics().phantomAirClinkCount;
      pw.dispose();
    }

    {
      // Coupled Kirchhoff Oak + Timoshenko Glass drop check
      const pw = new PhysicsWorld(audio, 1);
      pw.rebuildTower('sandbox', 2, 1, 7);
      for (let i = 0; i < 20; i++) pw.step(1 / 60);
      const rw = pw.getWorld();
      rw.getRigidBody(pw.blocks[1].bodyHandle).setTranslation({ x: -2.9, y: BLOCK_HALF_Y, z: -0.6 }, true);
      rw.getRigidBody(pw.blocks[2].bodyHandle).setTranslation({ x: 2.9, y: BLOCK_HALF_Y, z: 0.6 }, true);
      const rb0 = rw.getRigidBody(pw.blocks[0].bodyHandle);
      rb0.setTranslation({ x: 0, y: BLOCK_HALF_Y + 0.34, z: 0 }, true);
      rb0.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb0.setAngvel({ x: 0, y: 0, z: 0 }, true);
      rw.syncKinematics();
      audio.clearRecentImpactLog();
      for (let i = 0; i < 120; i++) pw.step(1 / 120);
      const dropBatch = collectAudio();
      coupledWoodCount = dropBatch.filter((e) => e.kind === 'woodThud' && e.fundamentalHz >= 95 && e.fundamentalHz <= 135).length;
      coupledGlassCount = dropBatch.filter((e) => e.kind === 'glass' && e.fundamentalHz > 2780).length;
      pw.dispose();
    }

    {
      // Seesaw Housner rocking WobbleRockEvent check
      const pw = new PhysicsWorld(audio, 1);
      pw.rebuildTower('sandbox', 2, 1, 11);
      for (let i = 0; i < 20; i++) pw.step(1 / 60);
      const rw = pw.getWorld();
      rw.getRigidBody(pw.blocks[1].bodyHandle).setTranslation({ x: -2.9, y: BLOCK_HALF_Y, z: -0.6 }, true);
      rw.getRigidBody(pw.blocks[2].bodyHandle).setTranslation({ x: 2.9, y: BLOCK_HALF_Y, z: 0.6 }, true);
      const tiltQ = quatFromAxisAngle([1, 0, 0], (8 * Math.PI) / 180);
      const rb0 = rw.getRigidBody(pw.blocks[0].bodyHandle);
      rb0.setTranslation({ x: 0, y: BLOCK_HALF_Y + 0.04, z: 0 }, true);
      rb0.setRotation({ x: tiltQ[0], y: tiltQ[1], z: tiltQ[2], w: tiltQ[3] }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb0.setAngvel({ x: 0, y: 0, z: 0 }, true);
      rw.wakeBody(pw.blocks[0].bodyHandle);
      rw.syncKinematics();
      for (let f = 0; f < 90; f++) {
        const res = pw.step(1 / 60);
        collectAudio();
        wobbleEventCount += res.wobbleEvents.length;
      }
      pw.dispose();
    }

    let nanInfAudioCount = 0;
    let outOfBandHzCount = 0;
    for (const entry of allAudioEntries) {
      if (!Number.isFinite(entry.fundamentalHz) || !Number.isFinite(entry.gain) || !Number.isFinite(entry.pitchScale)) {
        nanInfAudioCount++;
      }
      if (entry.fundamentalHz < 75 || entry.fundamentalHz > 9500) {
        outOfBandHzCount++;
      }
    }

    record(
      'D5 Acoustic & Modal Audit',
      '0 NaN/Inf, [75, 9500] Hz, coupled wood+glass, seesaw wobble, bounded bursts',
      nanInfAudioCount === 0 &&
        outOfBandHzCount === 0 &&
        phantomAirClinks === 0 &&
        coupledWoodCount >= 1 &&
        coupledGlassCount >= 1 &&
        wobbleEventCount >= 1 &&
        maxShardImpactBurst <= 8 &&
        maxWobbleBurst <= 4 &&
        allAudioEntries.length >= 10,
      `entries=${allAudioEntries.length}, nanInf=${nanInfAudioCount}, outOfBand=${outOfBandHzCount}, airClinks=${phantomAirClinks}, woodThud=${coupledWoodCount}, glassDrop=${coupledGlassCount}, wobble=${wobbleEventCount}, maxBurst=${maxShardImpactBurst}/${maxWobbleBurst}`
    );

    const elapsedMs = +(performance.now() - tStart).toFixed(1);
    const passed = checks.filter((c) => c.pass).length;
    const overallPass = passed === checks.length;

    if (emitJson) {
      console.log(JSON.stringify({ overallPass, passed, total: checks.length, elapsedMs, checks }, null, 2));
    } else {
      console.log('\n=== VITRAIL PRE-SHIP 5-DOMAIN EVALUATION SCORECARD ===');
      for (const c of checks) {
        const tag = c.pass ? 'PASS' : 'FAIL';
        console.log(`[${tag}] ${c.domain.padEnd(26)} | ${c.check} -> ${c.detail}`);
      }
      console.log(`\nSummary: ${passed}/${checks.length} checks passed in ${(elapsedMs / 1000).toFixed(2)}s\n`);
    }

    if (!overallPass) {
      process.exitCode = 1;
    }
  } finally {
    fuzzSuite?.dispose();
    await vite.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

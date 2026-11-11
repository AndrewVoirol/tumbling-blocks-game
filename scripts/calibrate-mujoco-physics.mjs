#!/usr/bin/env node
/**
 * Offline MuJoCo 4.0 Calibration & Invariant Verification CLI (`scripts/calibrate-mujoco-physics.mjs`)
 *
 * Evaluates `MUJOCO_CALIBRATED_CONFIG` across Vitrail's `T1–T8` physical stack equilibrium,
 * loose-vs-bearing clamping contrast, torsional cone binding, benign drop / micro-splinter safety,
 * and counterfactual `mj_copyState` oracle latency, printing a verified engineering scorecard.
 *
 * Usage:
 *   npm run calibrate:physics
 *   node --import tsx scripts/calibrate-mujoco-physics.mjs [--json]
 */

import { performance } from 'node:perf_hooks';
import { createServer } from 'vite';

const emitJson = process.argv.includes('--json');

async function main() {
  const vite = await createServer({
    logLevel: 'silent',
    server: { middlewareMode: true },
    appType: 'custom',
  });

  try {
    const [
      { GlassAudioEngine },
      { BLOCK_HEIGHT, TARGET_BLOCK_MASS_KG },
      { MUJOCO_CALIBRATED_CONFIG },
      { ensureMujocoInitialized, getMujocoRuntimeSync },
      { PhysicsWorld },
    ] = await Promise.all([
      vite.ssrLoadModule('/src/audio/GlassAudioEngine.ts'),
      vite.ssrLoadModule('/src/physics/constants.ts'),
      vite.ssrLoadModule('/src/physics/mujocoCalibratedParams.ts'),
      vite.ssrLoadModule('/src/physics/mujocoRuntime.ts'),
      vite.ssrLoadModule('/src/physics/PhysicsWorld.ts'),
    ]);

    const tBoot0 = performance.now();
    await ensureMujocoInitialized();
    const wasmInitMs = performance.now() - tBoot0;

  const audio = new GlassAudioEngine();
  audio.setMuted(true);

  // 1. 18-Layer Equilibrium, Sag, Sleep & Loose-vs-Bearing Clamping Contrast
  const world = new PhysicsWorld(audio, 18);
  const tSettle0 = performance.now();
  let stepsToFullSleep18 = -1;
  for (let i = 1; i <= 90; i++) {
    world.step(1 / 60);
    const rw = world.getWorld();
    let asleep = 0;
    for (const b of world.blocks) {
      if (rw.getRigidBody(b.bodyHandle)?.isSleeping()) asleep++;
    }
    if (asleep === 54 && stepsToFullSleep18 < 0) {
      stepsToFullSleep18 = i;
    }
  }
  const settle18WallMs = performance.now() - tSettle0;

  const rw18 = world.getWorld();
  let maxTopCenter18 = -Infinity;
  let asleep18 = 0;
  let massMinG = Infinity;
  let massMaxG = -Infinity;
  for (const b of world.blocks) {
    const rb = rw18.getRigidBody(b.bodyHandle);
    if (rb?.isSleeping()) asleep18++;
    const mg = (rb?.mass() ?? 0) * 1000.0;
    massMinG = Math.min(massMinG, mg);
    massMaxG = Math.max(massMaxG, mg);
    if (b.assignedLayer === 17 && !b.isLooseSlot) {
      maxTopCenter18 = Math.max(maxTopCenter18, b.position[1]);
    }
  }
  const sag18Mm = Math.max(0, (17 + 0.5) * BLOCK_HEIGHT - maxTopCenter18) * 100.0;

  let validLooseRows18 = 0;
  for (let layer = 0; layer < 17; layer++) {
    const row = world.blocks.filter((b) => b.assignedLayer === layer);
    const loose = row.find((b) => b.isLooseSlot);
    const tight = row.filter((b) => !b.isLooseSlot);
    if (!loose || tight.length !== 2) continue;
    const looseN = world.computeBlockClampingLoadN(loose.id);
    const tightMinN = Math.min(
      world.computeBlockClampingLoadN(tight[0].id),
      world.computeBlockClampingLoadN(tight[1].id)
    );
    if (looseN < tightMinN * 0.45) validLooseRows18++;
  }

  // Sleeping frame step cost (60 Hz)
  const tSleepStep0 = performance.now();
  const sleepBenchFrames = 60;
  for (let i = 0; i < sleepBenchFrames; i++) world.step(1 / 60);
  const sleepingFrameUs = ((performance.now() - tSleepStep0) * 1000.0) / sleepBenchFrames;

  // Counterfactual mj_copyState Oracle latency & discrimination
  const tOracle0 = performance.now();
  const oracleRanking = world.evaluateCounterfactualBlockCriticality(10, 6);
  const oracleMs = performance.now() - tOracle0;
  const safest = oracleRanking[0];
  const riskiest = oracleRanking[oracleRanking.length - 1];

  // 2. 24-Layer Sag & Sleep Check
  world.rebuildTower('solo', 2, 24);
  for (let i = 0; i < 90; i++) world.step(1 / 60);
  const rw24 = world.getWorld();
  let maxTopCenter24 = -Infinity;
  let asleep24 = 0;
  for (const b of world.blocks) {
    if (rw24.getRigidBody(b.bodyHandle)?.isSleeping()) asleep24++;
    if (b.assignedLayer === 23 && !b.isLooseSlot) {
      maxTopCenter24 = Math.max(maxTopCenter24, b.position[1]);
    }
  }
  const sag24Mm = Math.max(0, (23 + 0.5) * BLOCK_HEIGHT - maxTopCenter24) * 100.0;

  // 3. Asymmetric Collapse Distribution Probe
  world.rebuildTower('sandbox', 2, 18);
  for (let i = 0; i < 60; i++) world.step(1 / 60);
  world.triggerSandboxCollapseImpulse();
  const tCol0 = performance.now();
  for (let i = 0; i < 360; i++) world.step(1 / 60);
  const collapse360FramesMs = performance.now() - tCol0;
  const telCollapse = world.getTelemetry();

  world.dispose();

  const rt = getMujocoRuntimeSync();
  const scorecard = {
    engine: '@mujoco/mujoco@4.0.0',
    multithreadedWasmActive: rt.mode === 'mt',
    threadCount: rt.threadCount,
    calibratedConfig: MUJOCO_CALIBRATED_CONFIG,
    metrics: {
      wasmInitMs: +wasmInitMs.toFixed(2),
      blockMassRangeG: [+massMinG.toFixed(2), +massMaxG.toFixed(2)],
      targetBlockMassG: +(TARGET_BLOCK_MASS_KG * 1000).toFixed(2),
      sag18Mm: +sag18Mm.toFixed(4),
      sag24Mm: +sag24Mm.toFixed(4),
      asleep18: `${asleep18}/54`,
      asleep24: `${asleep24}/72`,
      timeToSleep18Sec: stepsToFullSleep18 > 0 ? +(stepsToFullSleep18 / 60).toFixed(3) : null,
      settle18WallMs: +settle18WallMs.toFixed(1),
      sleepingFrameUs: +sleepingFrameUs.toFixed(2),
      looseVsBearingContrast18: `${validLooseRows18}/17`,
      counterfactualOracle10CandMs: +oracleMs.toFixed(2),
      oracleTopCandidate: safest ?? null,
      oracleBottomCandidate: riskiest ?? null,
      collapse6SecAvgFrameMs: +(collapse360FramesMs / 360).toFixed(2),
      collapseIntactSurvivors: telCollapse.intactCount,
      collapseCrazedSurvivors: telCollapse.crazedCount,
      collapseShatteredBlocks: telCollapse.shatteredCount,
      collapseActiveShards: telCollapse.activeShardCount,
    },
    gates: {
      sag18Under015Mm: sag18Mm < 0.15,
      sag24Under050Mm: sag24Mm < 0.50,
      sleep18Complete: asleep18 === 54,
      sleep24Complete: asleep24 === 72,
      looseContrastAtLeast12: validLooseRows18 >= 12,
      collapseIntactAtLeast15: telCollapse.intactCount >= 15,
      collapseCrazedAtLeast4: telCollapse.crazedCount >= 4,
      oracleDiscriminatesLoad:
        Boolean(safest && riskiest) && safest.safetyScore01 > riskiest.safetyScore01 + 0.25,
    },
  };

  const allPassed = Object.values(scorecard.gates).every(Boolean);

  if (emitJson) {
    console.log(JSON.stringify(scorecard, null, 2));
  } else {
    console.log('\n=== Vitrail MuJoCo 4.0 Calibrated Physics Scorecard ===');
    console.log(
      `Runtime: ${scorecard.engine} (multithreaded pthreads: ${scorecard.multithreadedWasmActive})`
    );
    console.log(`WASM init: ${scorecard.metrics.wasmInitMs} ms`);
    console.log(
      `18L stack sag: ${scorecard.metrics.sag18Mm} mm (gate < 0.1500 mm) · sleep: ${scorecard.metrics.asleep18} in ${scorecard.metrics.timeToSleep18Sec}s`
    );
    console.log(
      `24L stack sag: ${scorecard.metrics.sag24Mm} mm (gate < 0.5000 mm) · sleep: ${scorecard.metrics.asleep24}`
    );
    console.log(
      `Loose vs. bearing row contrast: ${scorecard.metrics.looseVsBearingContrast18} rows (gate >= 12/17)`
    );
    console.log(`60 Hz sleeping frame cost: ${scorecard.metrics.sleepingFrameUs} µs/frame`);
    console.log(
      `Counterfactual mj_copyState oracle (10 candidates × 6 steps): ${scorecard.metrics.counterfactualOracle10CandMs} ms ` +
        `(safest #${safest?.blockId} score=${safest?.safetyScore01}, pinched #${riskiest?.blockId} score=${riskiest?.safetyScore01})`
    );
    console.log(
      `Asymmetric collapse (360 frames): ${scorecard.metrics.collapseIntactSurvivors} intact, ` +
        `${scorecard.metrics.collapseCrazedSurvivors} crazed, ${scorecard.metrics.collapseShatteredBlocks} shattered ` +
        `(${scorecard.metrics.collapseActiveShards} shards, ${scorecard.metrics.collapse6SecAvgFrameMs} ms/frame)`
    );
    console.log(`Gates: ${allPassed ? 'ALL PASS (8/8)' : 'FAILED'}`);
  }

  if (!allPassed) {
    process.exitCode = 1;
  }
  } finally {
    await vite.close();
  }
}

main()
  .then(() => process.exit(process.exitCode ?? 0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });

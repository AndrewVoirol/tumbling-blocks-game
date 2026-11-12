import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

const { createServer } = await import(
  pathToFileURL(process.cwd() + '/node_modules/vite/dist/node/index.js').href
);

function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[idx];
}

async function runAudit() {
  const vite = await createServer({
    logLevel: 'silent',
    server: { middlewareMode: true },
    appType: 'custom',
  });

  const failures = [];
  const report = {};

  try {
    const [
      { GlassAudioEngine },
      { quatFromAxisAngle },
      { BLOCK_HALF_Y, BLOCK_HEIGHT, BLOCK_WIDTH, TABLE_RADIUS },
      { ensureMujocoInitialized },
      { PhysicsWorld },
    ] = await Promise.all([
      vite.ssrLoadModule('/src/audio/GlassAudioEngine.ts'),
      vite.ssrLoadModule('/src/math/vecMat.ts'),
      vite.ssrLoadModule('/src/physics/constants.ts'),
      vite.ssrLoadModule('/src/physics/mujocoRuntime.ts'),
      vite.ssrLoadModule('/src/physics/PhysicsWorld.ts'),
    ]);

    await ensureMujocoInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    // 1. Flat 26 cm tabletop drop: pre-bounce fracture rebound + 3D hop + 100% sleep
    {
      const world = new PhysicsWorld(audio, 1);
      world.rebuildTower('sandbox', 2, 1);
      const rw = world.getWorld();
      for (let i = 1; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle).setTranslation(
          { x: 2.0 + i * 0.4, y: BLOCK_HALF_Y, z: 1.8 },
          true
        );
      }
      const b0 = world.blocks[0];
      const rb0 = rw.getRigidBody(b0.bodyHandle);
      rb0.setTranslation({ x: 0, y: 2.6 + BLOCK_HALF_Y, z: 0 }, true);
      rb0.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb0.setAngvel({ x: 0, y: 0, z: 0 }, true);

      // Measure free-fall gravity accuracy across first 8 frames (t = 8/60 s = 0.1333 s) before table impact
      const yStartDm = 2.6 + BLOCK_HALF_Y;
      let yAt8Dm = yStartDm;
      const shardStats = new Map();
      let shatterLogged = false;
      let avgInitVySim = 0;

      for (let f = 0; f < 150; f++) {
        world.step(1 / 60);
        if (f === 7 && !b0.shattered) {
          yAt8Dm = b0.position[1];
        }
        if (b0.shattered && !shatterLogged) {
          shatterLogged = true;
          let vySum = 0;
          for (const s of world.shards) {
            const vy = rw.getRigidBody(s.bodyHandle).linvel().y;
            vySum += vy;
            shardStats.set(s.id, { initY: s.position[1], peakY: s.position[1] });
          }
          avgInitVySim = vySum / Math.max(1, world.shards.length);
        }
        for (const s of world.shards) {
          const st = shardStats.get(s.id);
          if (st && s.position[1] > st.peakY) st.peakY = s.position[1];
        }
      }

      const t8Sec = 8 / 60;
      const expectedFallDm = 0.5 * 98.1 * t8Sec * t8Sec; // 9.81 m/s^2 = 98.1 dm/s^2
      const fallGravityRatio = (yStartDm - yAt8Dm) / expectedFallDm;

      const gainsMm = [...shardStats.values()].map((st) => Math.max(0, st.peakY - st.initY) * 100.0);
      const medianUpwardHopMm = +percentile(gainsMm, 0.5).toFixed(2);
      const p90UpwardHopMm = +percentile(gainsMm, 0.9).toFixed(2);
      const asleepShards = world.shards.filter((s) => rw.getRigidBody(s.bodyHandle)?.isSleeping()).length;

      const pass =
        b0.shattered &&
        avgInitVySim > 0.8 &&
        medianUpwardHopMm >= 6.0 &&
        p90UpwardHopMm >= 11.0 &&
        asleepShards === world.shards.length &&
        fallGravityRatio >= 0.92 &&
        fallGravityRatio <= 1.06;

      report.flat26cmDrop = {
        pass,
        avgInitVySim: +avgInitVySim.toFixed(3),
        medianUpwardHopMm,
        p90UpwardHopMm,
        asleepRatio: `${asleepShards}/${world.shards.length}`,
        fallGravityRatio: +fallGravityRatio.toFixed(3),
      };
      if (!pass) failures.push(`flat26cmDrop failed: ${JSON.stringify(report.flat26cmDrop)}`);
      world.dispose();
    }

    // 2. 15 cm mid-air shard drop onto oak_table_shard: secondary restitution + 100% sleep
    {
      const world = new PhysicsWorld(audio, 1);
      world.rebuildTower('sandbox', 2, 1);
      const rw = world.getWorld();
      for (let i = 1; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle).setTranslation(
          { x: 2.0 + i * 0.4, y: BLOCK_HALF_Y, z: 1.8 },
          true
        );
      }
      const b0 = world.blocks[0];
      const rb0 = rw.getRigidBody(b0.bodyHandle);
      rb0.setTranslation({ x: 0, y: 1.5, z: 0 }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      world.shatterBlock(b0, [0, -BLOCK_HALF_Y, 0], 15.0, [0, 1, 0], false);

      const stateMap = new Map();
      for (const s of world.shards) {
        stateMap.set(s.id, { touched: false, minY: Infinity, reboundPeakY: -Infinity });
      }
      for (let f = 0; f < 150; f++) {
        world.step(1 / 60);
        for (const s of world.shards) {
          const st = stateMap.get(s.id);
          if (!st) continue;
          const y = s.position[1];
          const vy = rw.getRigidBody(s.bodyHandle).linvel().y;
          if (!st.touched && f >= 6 && y < 0.11 && vy >= -0.02) {
            st.touched = true;
            st.minY = y;
            st.reboundPeakY = y;
          } else if (st.touched && y > st.reboundPeakY) {
            st.reboundPeakY = y;
          }
        }
      }
      const rebounds = [...stateMap.values()]
        .filter((st) => st.touched)
        .map((st) => Math.max(0, st.reboundPeakY - st.minY) * 100.0);
      const medianReboundHeightMm = +percentile(rebounds, 0.5).toFixed(2);
      const p90ReboundHeightMm = +percentile(rebounds, 0.9).toFixed(2);
      const asleepShards = world.shards.filter((s) => rw.getRigidBody(s.bodyHandle)?.isSleeping()).length;

      const pass =
        rebounds.length >= 16 &&
        medianReboundHeightMm >= 2.0 &&
        p90ReboundHeightMm >= 5.5 &&
        asleepShards === world.shards.length;

      report.midAir15cmShardDrop = {
        pass,
        shardsTracked: rebounds.length,
        medianReboundHeightMm,
        p90ReboundHeightMm,
        asleepRatio: `${asleepShards}/${world.shards.length}`,
      };
      if (!pass) failures.push(`midAir15cmShardDrop failed: ${JSON.stringify(report.midAir15cmShardDrop)}`);
      world.dispose();
    }

    // 3. 18L rimward collapse: 0 off-rim floaters past 4.30 dm, 0 solver resets, real-time step budget
    {
      const world = new PhysicsWorld(audio, 42);
      world.rebuildTower('sandbox', 2, 18);
      for (let f = 0; f < 40; f++) world.step(1 / 60);
      world.triggerSandboxCollapseImpulse();

      const rw = world.getWorld();
      const mjData = rw.getData();
      let prevTime = mjData ? mjData.time : 0;
      let solverTimeResets = 0;
      const stepMs = [];

      for (let f = 0; f < 260; f++) {
        const t0 = performance.now();
        world.step(1 / 60);
        stepMs.push(performance.now() - t0);
        if (mjData) {
          if (mjData.time + 1e-6 < prevTime) solverTimeResets++;
          prevTime = mjData.time;
        }
      }

      let endFloatingOffRimShards = 0;
      for (const s of world.shards) {
        const r = Math.hypot(s.position[0], s.position[2]);
        const vy = rw.getRigidBody(s.bodyHandle)?.linvel().y ?? 0;
        if (r > TABLE_RADIUS + 0.08 && s.position[1] > -0.05 && Math.abs(vy) < 0.08) {
          endFloatingOffRimShards++;
        }
      }
      let endFloatingOffRimBlocks = 0;
      for (const b of world.blocks) {
        if (b.shattered) continue;
        const r = Math.hypot(b.position[0], b.position[2]);
        const vy = rw.getRigidBody(b.bodyHandle)?.linvel().y ?? 0;
        if (r > TABLE_RADIUS + 0.14 && b.position[1] > -0.05 && Math.abs(vy) < 0.08) {
          endFloatingOffRimBlocks++;
        }
      }
      const stepMsAvg = +(stepMs.reduce((a, b) => a + b, 0) / Math.max(1, stepMs.length)).toFixed(2);
      const stepMsP95 = +percentile(stepMs, 0.95).toFixed(2);
      const pass =
        world.shatteredCount >= 18 &&
        endFloatingOffRimShards === 0 &&
        endFloatingOffRimBlocks === 0 &&
        solverTimeResets === 0 &&
        stepMsAvg < 9.5 &&
        stepMsP95 < 35.0;

      report.collapseOffRimAndClock = {
        pass,
        shatteredBlocks: world.shatteredCount,
        activeShards: world.shards.length,
        endFloatingOffRimShards,
        endFloatingOffRimBlocks,
        solverTimeResets,
        stepMsAvg,
        stepMsP95,
      };
      if (!pass) failures.push(`collapseOffRimAndClock failed: ${JSON.stringify(report.collapseOffRimAndClock)}`);
      world.dispose();
    }

    // 4. Marginal +3.9 cm (5.8 deg) cantilever overhang topple guard (never freezes asleep)
    {
      const world = new PhysicsWorld(audio, 7);
      world.rebuildTower('solo', 2, 18);
      const rw = world.getWorld();
      const tiltRad = (5.8 * Math.PI) / 180.0;
      const halfTilt = -tiltRad * 0.5;
      const sz = Math.sin(halfTilt);
      const cz = Math.cos(halfTilt);
      for (const b of world.blocks) {
        if (b.assignedLayer >= 12) {
          const dy = (b.assignedLayer - 12) * BLOCK_HEIGHT;
          const rb = rw.getRigidBody(b.bodyHandle);
          const p = rb.translation();
          const q = rb.rotation();
          rb.setTranslation({ x: p.x + 0.39 + Math.sin(tiltRad) * dy, y: p.y, z: p.z }, true);
          rb.setRotation(
            {
              x: cz * q.x - sz * q.y,
              y: sz * q.x + cz * q.y,
              z: cz * q.z + sz * q.w,
              w: -sz * q.z + cz * q.w,
            },
            true
          );
        }
      }
      let toppleFrame = -1;
      for (let f = 0; f < 120; f++) {
        world.step(1 / 60);
        if (world.turnPhase === 'COLLAPSED') {
          toppleFrame = f;
          break;
        }
      }
      const pass = world.turnPhase === 'COLLAPSED' && toppleFrame >= 0;
      report.marginalOverhangTopple = { pass, turnPhase: world.turnPhase, toppleFrame };
      if (!pass) failures.push(`marginalOverhangTopple failed: ${JSON.stringify(report.marginalOverhangTopple)}`);
      world.dispose();
    }

    // 5. Spatially-resolved Polariscope wrench accuracy (zero unloaded / cantilever false glow)
    {
      const world = new PhysicsWorld(audio, 11);
      world.rebuildTower('solo', 2, 18);
      for (let f = 0; f < 60; f++) world.step(1 / 60);
      const topBlk = world.blocks.find((b) => b.assignedLayer === 17);
      const looseRow0 = world.blocks.find((b) => b.assignedLayer === 0 && b.isLooseSlot);
      const topW = new Float32Array(4);
      const looseW = new Float32Array(4);
      world.writeBlockPhotoelasticWrench(topBlk.id, topW, 0);
      if (looseRow0) world.writeBlockPhotoelasticWrench(looseRow0.id, looseW, 0);
      const topMax = Math.max(topW[0], topW[1], topW[2], topW[3]);
      const looseMax = looseRow0 ? Math.max(looseW[0], looseW[1], looseW[2], looseW[3]) : 0;

      world.rebuildTower('sandbox', 2, 4);
      const rw = world.getWorld();
      for (let i = 0; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle).setTranslation(
          { x: -3.2, y: 0.075, z: (i - 5) * 0.32 },
          true
        );
      }
      const peUnder = world.blocks[0];
      const pePrism = world.blocks[1];
      rw.getRigidBody(peUnder.bodyHandle).setTranslation({ x: 0.25, y: BLOCK_HEIGHT * 0.5, z: 0 }, true);
      rw.getRigidBody(peUnder.bodyHandle).setRotation({ x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 }, true);
      rw.getRigidBody(pePrism.bodyHandle).setTranslation({ x: 0.0, y: BLOCK_HEIGHT * 1.5, z: 0 }, true);
      rw.getRigidBody(pePrism.bodyHandle).setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      for (let k = 0; k < 4; k++) {
        const wBlk = world.blocks[2 + k];
        rw.getRigidBody(wBlk.bodyHandle).setTranslation({ x: 0.25, y: BLOCK_HEIGHT * (2.5 + k), z: 0 }, true);
        rw.getRigidBody(wBlk.bodyHandle).setRotation({ x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 }, true);
      }
      for (let s = 0; s < 24; s++) world.step(1 / 60);
      const eccW = new Float32Array(4);
      world.writeBlockPhotoelasticWrench(pePrism.id, eccW, 0);

      const pass = topMax < 0.02 && looseMax < 0.05 && eccW[2] > 0.28 && eccW[0] < 0.02;
      report.polariscopeSpatialAccuracy = {
        pass,
        topCourseMax: +topMax.toFixed(4),
        looseSlotMax: +looseMax.toFixed(4),
        cantileverNegX: +eccW[0].toFixed(4),
        fulcrumPosX: +eccW[2].toFixed(4),
      };
      if (!pass) {
        failures.push(`polariscopeSpatialAccuracy failed: ${JSON.stringify(report.polariscopeSpatialAccuracy)}`);
      }
      world.dispose();
    }

    // 6. Full-row Stribeck grazing extraction isolation (< 2.0 mm upper-stack lateral drift + Layer-0 clear)
    {
      const world = new PhysicsWorld(audio, 19);
      world.rebuildTower('solo', 2, 18);
      for (let f = 0; f < 60; f++) world.step(1 / 60);

      const row0Loose = world.blocks.find((b) => b.assignedLayer === 0 && b.isLooseSlot) ?? world.blocks[0];
      const upperBlocks = world.blocks.filter((b) => b.assignedLayer >= 2);
      const initXZ = new Map(upperBlocks.map((b) => [b.id, [b.position[0], b.position[2]]]));

      const startX = row0Loose.position[0];
      const startY = row0Loose.position[1];
      const startZ = row0Loose.position[2];
      world.startGrabBlock(row0Loose.id, [0.34, 0, 0], [startX + 0.34, startY, startZ]);

      for (let f = 0; f < 95; f++) {
        const pullDist = Math.min(1.15, (f + 1) * 0.016);
        world.updateGrabTarget([startX + 0.34 + pullDist, startY, startZ]);
        world.step(1 / 60);
        if (world.turnPhase === 'EXTRACTED') break;
      }

      let maxUpperDriftMm = 0;
      for (const b of upperBlocks) {
        const [x0, z0] = initXZ.get(b.id);
        const driftMm = Math.hypot(b.position[0] - x0, b.position[2] - z0) * 100.0;
        if (driftMm > maxUpperDriftMm) maxUpperDriftMm = driftMm;
      }
      maxUpperDriftMm = +maxUpperDriftMm.toFixed(2);
      const pass = world.turnPhase === 'EXTRACTED' && maxUpperDriftMm < 2.0;

      report.grazingPullIsolation = {
        pass,
        turnPhase: world.turnPhase,
        maxUpperDriftMm,
      };
      if (!pass) failures.push(`grazingPullIsolation failed: ${JSON.stringify(report.grazingPullIsolation)}`);
      world.dispose();
    }

    // 7. Multi-stage Sandbox support-loss: 0 mid-air frozen shards after secondary collapse + outlier-resistant Plumb/Elevation
    {
      const world = new PhysicsWorld(audio, 21);
      world.rebuildTower('sandbox', 2, 18);
      for (let f = 0; f < 45; f++) world.step(1 / 60);

      // Stage A: Shatter upper courses (41..53) so shards sleep perched atop Course 13 (~20-22 cm),
      // and rest 1 crooked top block at 22 deg tilt to verify Plumb ignores single-brick top-notch outliers
      for (const id of [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41]) {
        world.shatterBlock(world.blocks[id], [0, 0, 0], 16.5, [0, 1, 0]);
      }
      const crookedTop = world.blocks[40];
      const rw = world.getWorld();
      const halfTilt = (22.0 * Math.PI) / 360.0;
      rw.getRigidBody(crookedTop.bodyHandle).setRotation(
        { x: 0, y: 0, z: Math.sin(halfTilt), w: Math.cos(halfTilt) },
        true
      );
      rw.getRigidBody(crookedTop.bodyHandle).setTranslation(
        { x: crookedTop.position[0], y: crookedTop.position[1] + 0.04, z: crookedTop.position[2] },
        true
      );
      for (let f = 0; f < 240; f++) world.step(1 / 60);

      const stageALeanDeg = +world.leanAngleDeg.toFixed(2);
      const stageALayers = world.currentLayers;

      // Stage B: Buckle & shatter the supporting tower underneath the perched sleeping shards!
      world.triggerSandboxCollapseImpulse();
      for (let f = 0; f < 360; f++) world.step(1 / 60);

      let unsupportedElevatedShards = 0;
      for (const s of world.shards) {
        const [sx, sy, sz] = s.position;
        const speed = Math.hypot(s.preLinVel[0], s.preLinVel[1], s.preLinVel[2]);
        const isAsleep = rw.getRigidBody(s.bodyHandle)?.isSleeping() ?? false;
        if (sy <= 0.26 || (!isAsleep && speed > 0.12)) continue;
        let supported = false;
        const ctype = world.model.geom_contype[s.colliderHandle];
        if (ctype === 4) {
          for (const b of world.blocks) {
            if (b.shattered || b.extracted) continue;
            const dy = sy - b.position[1];
            if (dy > -0.10 && dy < 0.54 && Math.abs(sx - b.position[0]) < 0.54 && Math.abs(sz - b.position[2]) < 0.54) {
              supported = true;
              break;
            }
          }
          if (!supported) {
            for (const os of world.shards) {
              if (os.id === s.id || world.model.geom_contype[os.colliderHandle] !== 4) continue;
              const dy = sy - os.position[1];
              if (dy > 0.01 && dy < 0.30 && Math.abs(sx - os.position[0]) < 0.30 && Math.abs(sz - os.position[2]) < 0.30) {
                supported = true;
                break;
              }
            }
          }
        }
        if (!supported) unsupportedElevatedShards++;
      }

      const stageBLayers = world.currentLayers;
      const isToppledHud = stageBLayers === 0 && world.shatteredCount > 0;
      const pass =
        stageALeanDeg < 2.0 &&
        stageALayers >= 13 &&
        unsupportedElevatedShards === 0 &&
        isToppledHud;

      report.multiStageSupportLossAndTelemetry = {
        pass,
        stageALeanDeg,
        stageALayers,
        unsupportedElevatedShards,
        stageBLayers,
        isToppledHud,
      };
      if (!pass) {
        failures.push(
          `multiStageSupportLossAndTelemetry failed: ${JSON.stringify(report.multiStageSupportLossAndTelemetry)}`
        );
      }
      world.dispose();
    }

    // 8. Post-collapse 100% tree sleep (visible rubble + hidden shelf) and zero post-settle audio buzz
    {
      let simT = 0;
      let postSettleImpacts = 0;
      let postSettleWobbles = 0;
      class CollapseSilenceAudio extends GlassAudioEngine {
        playGlassImpact() {
          if (simT >= 1.6) postSettleImpacts++;
        }
      }
      const probeAudio = new CollapseSilenceAudio();
      probeAudio.setMuted(true);
      const world = new PhysicsWorld(probeAudio, 18, { seed: 1 });
      world.rebuildTower('sandbox', 2, 18, 1);
      for (let f = 0; f < 30; f++) world.step(1 / 60);
      world.triggerSandboxCollapseImpulse();

      let fullSleepSec = null;
      for (let f = 1; f <= 180; f++) {
        simT = f / 60;
        world.step(1 / 60);
        const wob = world.consumeWobbleEvents();
        world.consumeShardImpacts();
        if (simT >= 1.6) postSettleWobbles += wob.length;
        if (fullSleepSec === null && simT > 0.35 && world.data.ntree_awake === 0) {
          fullSleepSec = +simT.toFixed(3);
        }
      }

      const ntreeAwakeEnd = world.data.ntree_awake;
      const pass =
        world.shatteredCount >= 15 &&
        ntreeAwakeEnd === 0 &&
        fullSleepSec !== null &&
        fullSleepSec <= 2.4 &&
        postSettleWobbles === 0 &&
        postSettleImpacts === 0;

      report.postCollapse100PctSleepAndSilence = {
        pass,
        shatteredCount: world.shatteredCount,
        fullSleepSec,
        ntreeAwakeEnd,
        postSettleWobbles,
        postSettleImpacts,
      };
      if (!pass) {
        failures.push(
          `postCollapse100PctSleepAndSilence failed: ${JSON.stringify(report.postCollapse100PctSleepAndSilence)}`
        );
      }
      world.dispose();
    }

    // 9. Fulcrum seesaw rocking vs. stationary fulcrum & heavily clamped bearing-course silence
    {
      const world = new PhysicsWorld(audio, 1, { seed: 1 });
      world.rebuildTower('sandbox', 2, 1, 1);
      const rw = world.getWorld();
      const [bSeesaw, bFulcrum, bSpare] = world.blocks;
      rw.getRigidBody(bSpare.bodyHandle).setTranslation({ x: 2.5, y: BLOCK_HALF_Y, z: 0 }, true);
      const qYaw90 = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
      rw.getRigidBody(bFulcrum.bodyHandle).setTranslation({ x: 0, y: BLOCK_HALF_Y, z: 0 }, true);
      rw.getRigidBody(bFulcrum.bodyHandle).setRotation(
        { x: qYaw90[0], y: qYaw90[1], z: qYaw90[2], w: qYaw90[3] },
        true
      );
      rw.getRigidBody(bSeesaw.bodyHandle).setTranslation({ x: -2.5, y: BLOCK_HALF_Y, z: 0 }, true);
      for (let k = 0; k < 120; k++) world.step(1 / 600);
      world.consumeWobbleEvents();

      const rad = (6.5 * Math.PI) / 180.0;
      const halfSpan = BLOCK_WIDTH * 0.5 - 0.0125;
      const yCenter =
        BLOCK_HEIGHT + BLOCK_HALF_Y * Math.cos(rad) + halfSpan * Math.sin(rad) + 0.0004;
      const qTilt = quatFromAxisAngle([0, 0, 1], rad);
      const rbSeesaw = rw.getRigidBody(bSeesaw.bodyHandle);
      rbSeesaw.setTranslation({ x: 0, y: yCenter, z: 0 }, true);
      rbSeesaw.setRotation({ x: qTilt[0], y: qTilt[1], z: qTilt[2], w: qTilt[3] }, true);
      rbSeesaw.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rbSeesaw.setAngvel({ x: 0, y: 0, z: 0 }, true);

      const sDof = world.model.jnt_dofadr[world.model.body_jntadr[bSeesaw.bodyHandle]];
      let prevSign = 0;
      let seesawFlips = 0;
      let seesawWobbles = 0;
      let fulcrumWobbles = 0;
      for (let k = 0; k < 420; k++) {
        world.step(1 / 600);
        const wz = world.data.qvel[sDof + 5];
        if (Math.abs(wz) > 0.03) {
          const sgn = wz > 0 ? 1 : -1;
          if (prevSign !== 0 && sgn !== prevSign) seesawFlips++;
          prevSign = sgn;
        }
        for (const w of world.consumeWobbleEvents()) {
          if (w.blockId === bSeesaw.id) seesawWobbles++;
          if (w.blockId === bFulcrum.id) fulcrumWobbles++;
        }
      }
      const seesawAsleep =
        world.data.tree_asleep[world.model.body_treeid[bSeesaw.bodyHandle]] >= 0;
      world.dispose();

      // Clamped lower-course check: nudging a clamped Layer-1 bearing block (> 10 N top load) emits 0 clamped wobbles
      const towerWorld = new PhysicsWorld(audio, 18, { seed: 7 });
      towerWorld.rebuildTower('solo', 2, 18, 7);
      for (let f = 0; f < 50; f++) towerWorld.step(1 / 60);
      towerWorld.consumeWobbleEvents();
      const bearingBlk =
        towerWorld.blocks.find((b) => b.assignedLayer === 1 && !b.isLooseSlot) ??
        towerWorld.blocks[4];
      const rbBearing = towerWorld.getWorld().getRigidBody(bearingBlk.bodyHandle);
      rbBearing.setLinvel({ x: 0.28, y: 0, z: 0.08 }, true);
      let clampedFalseWobbles = 0;
      for (let f = 0; f < 45; f++) {
        towerWorld.step(1 / 60);
        for (const w of towerWorld.consumeWobbleEvents()) {
          if (w.clampingLoadN >= 1.1) clampedFalseWobbles++;
        }
      }
      towerWorld.dispose();

      const pass =
        seesawFlips >= 5 &&
        seesawWobbles >= 3 &&
        fulcrumWobbles === 0 &&
        seesawAsleep &&
        clampedFalseWobbles === 0;

      report.fulcrumSeesawAndClampedWobbleIsolation = {
        pass,
        seesawFlips,
        seesawWobbles,
        fulcrumWobbles,
        seesawAsleep,
        clampedFalseWobbles,
      };
      if (!pass) {
        failures.push(
          `fulcrumSeesawAndClampedWobbleIsolation failed: ${JSON.stringify(report.fulcrumSeesawAndClampedWobbleIsolation)}`
        );
      }
    }

    console.log(
      JSON.stringify(
        {
          overallPass: failures.length === 0,
          passedProbes: `${9 - failures.length}/9`,
          probes: report,
          failures,
        },
        null,
        2
      )
    );
  } finally {
    await vite.close();
  }

  if (failures.length > 0) {
    process.exit(1);
  }
}

runAudit().then(() => process.exit(0));

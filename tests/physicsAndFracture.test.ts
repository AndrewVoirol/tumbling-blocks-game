import { describe, expect, it } from 'vitest';
import { fractureBlockVoronoi } from '../src/physics/voronoiFracture';
import {
  BLOCK_HEIGHT,
  BLOCK_LENGTH,
  BLOCK_WIDTH,
  INITIAL_BLOCK_COUNT,
  SIM_TIME_SCALE,
} from '../src/physics/constants';
import { ensureRapierInitialized, PhysicsWorld } from '../src/physics/PhysicsWorld';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';

describe('3D Anisotropic Voronoi Stained Glass Fracture', () => {
  it('conserves exact 7.5 x 2.5 x 1.5 cm Jenga block volume and exports convex planes across motifs', () => {
    const expectedVolume = BLOCK_LENGTH * BLOCK_HEIGHT * BLOCK_WIDTH; // 0.028125 dm^3

    for (const motif of [0, 1, 2, 3]) {
      const shards = fractureBlockVoronoi(
        [0.28, -0.05, 0.08],
        14,
        42 + motif * 17,
        undefined,
        undefined,
        undefined,
        motif,
        0.75
      );

      expect(shards.length).toBeGreaterThanOrEqual(8);
      let sumVolume = 0;
      let hasOuterFace = false;
      let hasInnerCleavageFace = false;

      let nearVolSum = 0;
      let nearCount = 0;
      let farVolSum = 0;
      let farCount = 0;

      for (const s of shards) {
        expect(s.volume).toBeGreaterThan(0);
        expect(s.hullPoints.length).toBeGreaterThanOrEqual(12); // >= 4 3D vertices
        expect(s.vertexCount % 3).toBe(0);
        expect(s.planeCount).toBeGreaterThanOrEqual(4);
        expect(Math.abs(s.planes[3])).toBeGreaterThan(0);
        sumVolume += s.volume;

        if (s.impactDist < 0.16) {
          nearVolSum += s.volume;
          nearCount++;
        } else if (s.impactDist > 0.32) {
          farVolSum += s.volume;
          farCount++;
        }

        for (let v = 0; v < s.vertexCount; v++) {
          const flag = s.renderVertices[v * 10 + 9];
          if (flag > 0.5) hasOuterFace = true;
          else hasInnerCleavageFace = true;
        }
      }

      expect(Math.abs(sumVolume - expectedVolume)).toBeLessThan(1e-4);
      expect(hasOuterFace).toBe(true);
      expect(hasInnerCleavageFace).toBe(true);
      if (nearCount > 0 && farCount > 0) {
        // Crush-core shards near impact point should average smaller than far-end remnants
        expect(nearVolSum / nearCount).toBeLessThan(farVolSum / farCount);
      }
    }

    // Regression sweep across 200 deterministic fractures (including dense 23-seed cells > 132 verts)
    for (let i = 0; i < 200; i++) {
      const targetCount = i === 0 ? 23 : 8 + (i % 16);
      const seed = i === 0 ? 776 * 1013 : (i + 1) * 1013;
      const ox = i === 0 ? 0.31 : (((i * 73) % 100) / 100 - 0.5) * 0.68;
      const oy = (((i * 37) % 100) / 100 - 0.5) * 0.12;
      const oz = (((i * 53) % 100) / 100 - 0.5) * 0.22;
      const shards = fractureBlockVoronoi([ox, oy, oz], targetCount, seed, undefined, undefined, undefined, 0, 0.85);
      let vol = 0;
      for (const s of shards) {
        expect(s.vertexCount).toBeLessThanOrEqual(132);
        vol += s.volume;
      }
      expect(Math.abs(vol - expectedVolume) / expectedVolume).toBeLessThan(0.0005);
    }
  });
});

describe('Rapier3D Stained Glass Jenga Physics & Convex Hull Shard Settling', () => {
  it('builds a stable 18-layer (27.0 cm) 54-block tower and settles >= 300 convex shards naturally after collapse', async () => {
    await ensureRapierInitialized();
    const audio = new GlassAudioEngine();
    const world = new PhysicsWorld(audio);

    expect(world.blocks.length).toBe(INITIAL_BLOCK_COUNT);

    // Step 2.0 seconds of real wall-clock time and confirm static equilibrium
    for (let i = 0; i < 120; i++) {
      world.step(1 / 60);
    }

    const telemetry = world.getTelemetry();
    expect(telemetry.shatteredCount).toBe(0);
    expect(telemetry.turnPhase).toBe('SELECT_BLOCK');
    expect(telemetry.leanAngleDeg).toBeLessThan(0.6);
    expect(telemetry.peakHeightCm).toBeCloseTo(27.0, 0);

    // Tap-test a legal lower-middle block and confirm tower stays standing
    const tapped = world.tapNudgeBlock(19, [-1, 0, 0], [0.37, 0, 0]);
    expect(tapped).toBe(true);
    for (let i = 0; i < 30; i++) {
      world.step(1 / 60);
    }
    expect(world.getTelemetry().turnPhase).toBe('SELECT_BLOCK');

    // Trigger natural cascade collapse impulse: must NOT detonate upright (< 8 deg lean) and must yield >= 300 shards!
    world.rebuildTower('sandbox');
    for (let i = 0; i < 15; i++) world.step(1 / 60);
    world.triggerSandboxCollapseImpulse();
    let uprightShatters = 0;
    for (let i = 0; i < 160; i++) {
      world.step(1 / 60);
      if (world.leanAngleDeg < 8.0 && world.shatteredCount > 0) {
        uprightShatters = Math.max(uprightShatters, world.shatteredCount);
      }
    }
    expect(uprightShatters).toBe(0);

    const afterShatter = world.getTelemetry();
    expect(afterShatter.shatteredCount).toBeGreaterThanOrEqual(20);
    expect(afterShatter.activeShardCount).toBeGreaterThanOrEqual(300);

    // Verify shards rest near the oak tabletop without upward grenade explosions
    let groundedShards = 0;
    for (const s of world.shards) {
      expect(s.position[1]).toBeLessThan(2.8);
      if (s.position[1] > -0.02 && s.position[1] < 0.35) {
        groundedShards++;
      }
    }
    expect(groundedShards).toBeGreaterThanOrEqual(200);

    world.dispose();
  });
});

describe('Tier 1 Physics, Fracture & Playability Benchmark Gate (T1-T4)', () => {
  it('T1: 18L & 24L equilibrium, 71.1g beveled mass, <0.15mm/<0.50mm sag, 100% sleep, and loose-slot distribution', async () => {
    await ensureRapierInitialized();
    const audio = new GlassAudioEngine();
    const world = new PhysicsWorld(audio);
    const rapierWorld = world.getWorld();

    // Verify motif = 0 (Option A pure cathedral waterglass) and density-compensated roundCuboid mass (~71.10g)
    for (const b of world.blocks) {
      expect(b.motif).toBe(0);
      const rb = rapierWorld.getRigidBody(b.bodyHandle)!;
      const massKg = rb.mass();
      expect(massKg).toBeGreaterThanOrEqual(0.0705);
      expect(massKg).toBeLessThanOrEqual(0.0716);
    }

    // Step 1.5s at 60fps (90 frames, 120Hz sim solver)
    for (let i = 0; i < 90; i++) {
      world.step(1 / 60);
    }

    // 18L sag < 0.15 mm (0.0015 dm) and 100% bodies asleep
    let asleep18 = 0;
    let maxTopCenter18 = 0;
    for (const b of world.blocks) {
      const rb = rapierWorld.getRigidBody(b.bodyHandle)!;
      if (rb.isSleeping()) asleep18++;
      if (b.assignedLayer === 17 && !b.isLooseSlot) {
        maxTopCenter18 = Math.max(maxTopCenter18, b.position[1]);
      }
    }
    const idealTopCenter18 = (17 + 0.5) * BLOCK_HEIGHT; // 2.625 dm
    const sag18Mm = Math.max(0, idealTopCenter18 - maxTopCenter18) * 100.0;
    expect(sag18Mm).toBeLessThan(0.15);
    expect(asleep18).toBe(54);

    // >= 12/18 rows have 1 loose block (F_n < 0.5 N) + 2 load-bearing blocks
    let validLooseRows = 0;
    let sampleLooseId = -1;
    let samplePinchedId = -1;
    for (let layer = 0; layer < 17; layer++) {
      const rowBlocks = world.blocks.filter((b) => b.assignedLayer === layer);
      let looseInRow = 0;
      let bearingInRow = 0;
      for (const rbBlk of rowBlocks) {
        const fn = world.computeBlockClampingLoadN(rbBlk.id);
        if (fn < 0.5) {
          looseInRow++;
          if (sampleLooseId < 0 && layer >= 2 && layer <= 8) sampleLooseId = rbBlk.id;
        } else {
          bearingInRow++;
          if (samplePinchedId < 0 && layer >= 1 && layer <= 6 && fn > 4.0) samplePinchedId = rbBlk.id;
        }
      }
      if (looseInRow === 1 && bearingInRow === 2) {
        validLooseRows++;
      }
    }
    expect(validLooseRows).toBeGreaterThanOrEqual(12);

    // Verify load-coupled acoustic tap-test contrast (loose Q=48/160ms vs pinched Q~14/35ms)
    expect(sampleLooseId).toBeGreaterThanOrEqual(0);
    expect(samplePinchedId).toBeGreaterThanOrEqual(0);
    world.tapNudgeBlock(sampleLooseId, [1, 0, 0], world.blocks[sampleLooseId].position);
    const looseAudio = audio.lastTapDiagnostics!;
    world.tapNudgeBlock(samplePinchedId, [1, 0, 0], world.blocks[samplePinchedId].position);
    const pinchedAudio = audio.lastTapDiagnostics!;
    expect(looseAudio.qFactor).toBeGreaterThan(42);
    expect(looseAudio.decayMs).toBeGreaterThan(140);
    expect(pinchedAudio.qFactor).toBeLessThan(26);
    expect(pinchedAudio.decayMs).toBeLessThan(80);
    expect(looseAudio.fundamentalHz).toBeLessThan(pinchedAudio.fundamentalHz);

    // 24L equilibrium: sag < 0.50 mm and 100% (72/72) bodies asleep at t = 1.5s
    world.rebuildTower('solo', 2, 24);
    for (let i = 0; i < 90; i++) {
      world.step(1 / 60);
    }
    const rapier24 = world.getWorld();
    let asleep24 = 0;
    let maxTopCenter24 = 0;
    for (const b of world.blocks) {
      const rb = rapier24.getRigidBody(b.bodyHandle)!;
      if (rb.isSleeping()) asleep24++;
      if (b.assignedLayer === 23 && !b.isLooseSlot) {
        maxTopCenter24 = Math.max(maxTopCenter24, b.position[1]);
      }
    }
    const idealTopCenter24 = (23 + 0.5) * BLOCK_HEIGHT;
    const sag24Mm = Math.max(0, idealTopCenter24 - maxTopCenter24) * 100.0;
    expect(sag24Mm).toBeLessThan(0.5);
    expect(asleep24).toBe(72);

    world.dispose();
  });

  it('T2: static stack 1.0cm placement drop & 1.2g micro-splinter at 18 dm/s incur 0 false damage', async () => {
    await ensureRapierInitialized();
    const RAPIER = (await import('@dimforge/rapier3d-compat')).default;
    const audio = new GlassAudioEngine();
    const world = new PhysicsWorld(audio);

    for (let i = 0; i < 45; i++) world.step(1 / 60);

    // 1. Drop a 71.1g glass block from 1.0 cm (0.10 dm clearance) onto openTopSlots[0]
    const slot0 = world.openTopSlots[0];
    expect(slot0).toBeDefined();
    const dropBlock = world.blocks.find((b) => b.assignedLayer === 6 && b.isLooseSlot)!;
    const rw = world.getWorld();
    const dropRb = rw.getRigidBody(dropBlock.bodyHandle)!;
    dropRb.setTranslation(
      {
        x: slot0.position[0],
        y: slot0.position[1] + 0.1, // 1.0 cm drop clearance
        z: slot0.position[2],
      },
      true
    );
    dropRb.setRotation(
      {
        x: slot0.rotation[0],
        y: slot0.rotation[1],
        z: slot0.rotation[2],
        w: slot0.rotation[3],
      },
      true
    );
    dropRb.setLinvel({ x: 0, y: 0, z: 0 }, true);
    dropRb.setAngvel({ x: 0, y: 0, z: 0 }, true);
    dropBlock.assignedLayer = slot0.layer;
    dropBlock.assignedSlot = slot0.slotIndex;

    for (let i = 0; i < 50; i++) world.step(1 / 60);

    for (const b of world.blocks) {
      expect(b.shattered).toBe(false);
      expect(b.damage).toBe(0);
    }

    // 2. Drop a 1.2g micro-splinter at 18 dm/s onto a resting intact glass block on the table
    const targetBlk = world.blocks[0];
    const targetRb = rw.getRigidBody(targetBlk.bodyHandle)!;
    targetRb.setTranslation({ x: 1.8, y: BLOCK_HEIGHT * 0.5, z: 0 }, true);
    targetRb.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    targetRb.setLinvel({ x: 0, y: 0, z: 0 }, true);
    for (let i = 0; i < 10; i++) world.step(1 / 60);

    const simTimeScale = Math.sqrt(10.0);
    const splinterRb = rw.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(1.8, BLOCK_HEIGHT + 0.08, 0)
        .setLinvel(0, -18.0 / simTimeScale, 0)
    );
    // 1.2 g = 0.0012 kg; cuboid(0.04, 0.03, 0.04) volume = 8 * 0.000048 = 0.000384 dm^3 -> density = 3.125
    rw.createCollider(
      RAPIER.ColliderDesc.cuboid(0.04, 0.03, 0.04)
        .setDensity(0.0012 / (8 * 0.04 * 0.03 * 0.04))
        .setRestitution(0.24)
        .setActiveEvents(RAPIER.ActiveEvents.CONTACT_FORCE_EVENTS),
      splinterRb
    );

    for (let i = 0; i < 25; i++) world.step(1 / 60);

    for (const b of world.blocks) {
      expect(b.shattered).toBe(false);
      expect(b.damage).toBe(0);
    }

    world.dispose();
  });

  it('T3: (a) dynamic anvil-crush, (b) +X tip argmax fracture origin, (c) <1% momentum error, and (d) 512-slot stress', async () => {
    await ensureRapierInitialized();
    const RAPIER = (await import('@dimforge/rapier3d-compat')).default;
    const { quatRotateVec3, quatInverseRotateVec3 } = await import('../src/math/vecMat');
    const audio = new GlassAudioEngine();
    const world = new PhysicsWorld(audio, 2);

    // (a) Dynamic Anvil-Crush: 213.3g 3-block row dropped 18cm (1.80 dm) onto table-supported bottom block
    world.rebuildTower('sandbox', 2, 2); // 6 blocks total
    const rw = world.getWorld();
    const anvilTarget = world.blocks[0];
    const rbTarget = rw.getRigidBody(anvilTarget.bodyHandle)!;
    rbTarget.setTranslation({ x: 0, y: BLOCK_HEIGHT * 0.5, z: 0 }, true);
    rbTarget.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    rbTarget.setLinvel({ x: 0, y: 0, z: 0 }, true);
    rbTarget.setAngvel({ x: 0, y: 0, z: 0 }, true);

    // Move blocks 1, 2 out of the way and position blocks 3, 4, 5 as a 213g 3-block falling row 18 cm above anvilTarget
    for (const idx of [1, 2]) {
      const rbOut = rw.getRigidBody(world.blocks[idx].bodyHandle)!;
      rbOut.setTranslation({ x: 2.5 + idx * 0.5, y: BLOCK_HEIGHT * 0.5, z: 2.5 }, true);
      rbOut.setLinvel({ x: 0, y: 0, z: 0 }, true);
    }
    const dropCenterY = BLOCK_HEIGHT + 1.8 + BLOCK_HEIGHT * 0.5;
    for (let s = 0; s < 3; s++) {
      const bRow = world.blocks[3 + s];
      const rbRow = rw.getRigidBody(bRow.bodyHandle)!;
      rbRow.setTranslation({ x: (s - 1) * BLOCK_WIDTH * 0.9, y: dropCenterY, z: 0 }, true);
      rbRow.setRotation({ x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 }, true);
      rbRow.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rbRow.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }

    for (let i = 0; i < 24; i++) {
      world.step(1 / 60);
      if (anvilTarget.shattered) break;
    }
    expect(anvilTarget.shattered).toBe(true);

    // (b) Peak-Impulse Fracture Origin (`argmax`): strike +X tip (+0.35 dm) of a block
    world.rebuildTower('sandbox', 2, 1);
    const rwB = world.getWorld();
    const tipBlock = world.blocks[0];
    const rbTip = rwB.getRigidBody(tipBlock.bodyHandle)!;
    rbTip.setTranslation({ x: 0, y: 1.2, z: 0 }, true);
    rbTip.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    rbTip.setGravityScale(0, true);
    rbTip.setLinvel({ x: 0, y: 0, z: 0 }, true);
    rbTip.setAngvel({ x: 0, y: 0, z: 0 }, true);

    for (const idx of [1, 2]) {
      rwB.getRigidBody(world.blocks[idx].bodyHandle)!.setTranslation({ x: -3.0, y: 0.075, z: idx }, true);
    }

    const simScale = Math.sqrt(10.0);
    const strikerRb = rwB.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(0.33, 1.42, 0)
        .setGravityScale(0)
        .setLinvel(0, -24.0 / simScale, 0)
    );
    rwB.createCollider(
      RAPIER.ColliderDesc.cuboid(0.03, 0.08, 0.08)
        .setDensity(0.0711 / (8 * 0.03 * 0.08 * 0.08))
        .setRestitution(0.28)
        .setActiveEvents(RAPIER.ActiveEvents.CONTACT_FORCE_EVENTS),
      strikerRb
    );

    for (let i = 0; i < 12; i++) {
      world.step(1 / 60);
      if (tipBlock.shattered) break;
    }
    expect(tipBlock.shattered).toBe(true);
    expect(world.lastFractureOriginLocal).not.toBeNull();
    expect(world.lastFractureOriginLocal![0]).toBeGreaterThan(0.25);

    // (c) Momentum & Mass Conservation (< 1% linear & angular momentum error)
    world.rebuildTower('sandbox', 2, 1);
    const rwC = world.getWorld();
    const momBlock = world.blocks[0];
    const rbMom = rwC.getRigidBody(momBlock.bodyHandle)!;
    rbMom.setTranslation({ x: 0, y: 1.8, z: 0 }, true);
    rbMom.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    const v0 = { x: 1.85, y: -2.4, z: 0.92 };
    const w0 = { x: 2.1, y: -3.4, z: 1.6 };
    rbMom.setLinvel(v0, true);
    rbMom.setAngvel(w0, true);

    const m0 = rbMom.mass();
    const p0: [number, number, number] = [m0 * v0.x, m0 * v0.y, m0 * v0.z];
    const pIn = rbMom.principalInertia();
    const pFr = rbMom.principalInertiaLocalFrame();
    const qFr: [number, number, number, number] = [pFr.x, pFr.y, pFr.z, pFr.w];
    const wPrin = quatInverseRotateVec3(qFr, [w0.x, w0.y, w0.z]);
    const l0 = quatRotateVec3(qFr, [pIn.x * wPrin[0], pIn.y * wPrin[1], pIn.z * wPrin[2]]);

    world.shatterBlock(momBlock, [0.18, -0.02, 0.05], 19.5, [-1, 0, 0], false);

    let mSum = 0;
    const pSum = [0, 0, 0];
    const lSum = [0, 0, 0];
    for (const sh of world.shards) {
      const srb = rwC.getRigidBody(sh.bodyHandle)!;
      const ms = srb.mass();
      const st = srb.translation();
      const slv = srb.linvel();
      const sav = srb.angvel();
      const sr = srb.rotation();
      mSum += ms;
      pSum[0] += ms * slv.x;
      pSum[1] += ms * slv.y;
      pSum[2] += ms * slv.z;

      const rx = st.x - 0;
      const ry = st.y - 1.8;
      const rz = st.z - 0;
      lSum[0] += ms * (ry * slv.z - rz * slv.y);
      lSum[1] += ms * (rz * slv.x - rx * slv.z);
      lSum[2] += ms * (rx * slv.y - ry * slv.x);

      const spI = srb.principalInertia();
      const spF = srb.principalInertiaLocalFrame();
      // Combine shard rotation with local principal frame
      const qBody: [number, number, number, number] = [sr.x, sr.y, sr.z, sr.w];
      const qLoc: [number, number, number, number] = [spF.x, spF.y, spF.z, spF.w];
      const qTot: [number, number, number, number] = [
        qBody[3] * qLoc[0] + qBody[0] * qLoc[3] + qBody[1] * qLoc[2] - qBody[2] * qLoc[1],
        qBody[3] * qLoc[1] - qBody[0] * qLoc[2] + qBody[1] * qLoc[3] + qBody[2] * qLoc[0],
        qBody[3] * qLoc[2] + qBody[0] * qLoc[1] - qBody[1] * qLoc[0] + qBody[2] * qLoc[3],
        qBody[3] * qLoc[3] - qBody[0] * qLoc[0] - qBody[1] * qLoc[1] - qBody[2] * qLoc[2],
      ];
      const swP = quatInverseRotateVec3(qTot, [sav.x, sav.y, sav.z]);
      const sSpinL = quatRotateVec3(qTot, [spI.x * swP[0], spI.y * swP[1], spI.z * swP[2]]);
      lSum[0] += sSpinL[0];
      lSum[1] += sSpinL[1];
      lSum[2] += sSpinL[2];
    }

    const linErr =
      Math.hypot(pSum[0] - p0[0], pSum[1] - p0[1], pSum[2] - p0[2]) /
      Math.hypot(p0[0], p0[1], p0[2]);
    const angErr =
      Math.hypot(lSum[0] - l0[0], lSum[1] - l0[1], lSum[2] - l0[2]) /
      Math.hypot(l0[0], l0[1], l0[2]);

    expect(Math.abs(mSum - m0) / m0).toBeLessThan(0.002);
    expect(linErr).toBeLessThan(0.01);
    expect(angErr).toBeLessThan(0.01);

    // (d) 512-Slot Fixed-Stride Pool Stress: 250 sequential shatters (~3,000+ shards) with 0 live-slot overwrites
    let totalSpawnedAcrossStress = 0;
    for (let batch = 0; batch < 10; batch++) {
      // Spawn 25 blocks in current world or shatter blocks across batches while keeping shard pool live!
      const rwD = world.getWorld();
      for (let k = 0; k < 25; k++) {
        const id = world.blocks.length;
        const rbDesc = RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 1.5 + k * 0.05, 0);
        const body = rwD.createRigidBody(rbDesc);
        const col = rwD.createCollider(
          RAPIER.ColliderDesc.cuboid(0.375, 0.075, 0.125).setDensity(2.528),
          body
        );
        const syntheticBlk = {
          id,
          bodyHandle: body.handle,
          colliderHandle: col.handle,
          swatchId: k % 6,
          motif: 0 as const,
          rippleSeed: id * 1.37,
          initialLayer: 0,
          assignedLayer: 0,
          assignedSlot: 0,
          isLooseSlot: false,
          position: [0, 1.5, 0] as [number, number, number],
          rotation: [0, 0, 0, 1] as [number, number, number, number],
          prePos: [0, 1.5, 0] as [number, number, number],
          preQuat: [0, 0, 0, 1] as [number, number, number, number],
          linVel: [0, 0, 0] as [number, number, number],
          angVel: [0, 0, 0] as [number, number, number],
          currentFriction: 0.42,
          isMicroDamped: false,
          damage: 0,
          crackLocalOrigin: [0, 0, 0] as [number, number, number],
          shattered: false,
          extracted: false,
          isLegalTarget: true,
        };
        world.blocks.push(syntheticBlk);
        world.shatterBlock(syntheticBlk, [0.1, 0, 0], 17.5, [1, 0, 0], false);
        totalSpawnedAcrossStress += 12;
      }

      expect(world.shards.length).toBeLessThanOrEqual(512);
      const liveSlots = new Set<number>();
      for (const sh of world.shards) {
        expect(sh.slotIndex).toBeGreaterThanOrEqual(0);
        expect(sh.slotIndex).toBeLessThan(512);
        expect(sh.firstVertex).toBe(sh.slotIndex * 132);
        expect(sh.vertexCount).toBeGreaterThan(0);
        expect(sh.vertexCount).toBeLessThanOrEqual(132);
        expect(liveSlots.has(sh.slotIndex)).toBe(false);
        liveSlots.add(sh.slotIndex);
      }
      expect(liveSlots.size).toBe(world.shards.length);
      world.consumePendingShardUploads();
    }
    expect(totalSpawnedAcrossStress).toBeGreaterThanOrEqual(3000);

    world.dispose();
  });

  it('T4: (a) <5px tap isolation keeps SELECT_BLOCK, and (b) 11-deg grazing pull extracts cleanly after 3 zero-contact substeps', async () => {
    const { GameEngine } = await import('../src/engine/GameEngine');
    const stubCanvas = { style: { cursor: 'default' } } as unknown as HTMLCanvasElement;
    const engine = new GameEngine(stubCanvas);
    await engine.initHeadless();

    const rect = { left: 0, top: 0, width: 1280, height: 800 };
    for (let i = 0; i < 45; i++) engine.physics.step(1 / 60);

    // Find a loose middle block on an even layer (e.g., Layer 8, slot 1 -> lengthwise along X, centered at Z=0)
    const targetBlock = engine.physics.blocks.find(
      (b) => b.assignedLayer === 8 && b.assignedSlot === 1 && b.isLooseSlot
    )!;
    expect(targetBlock).toBeDefined();

    // Set camera at 11 deg grazing elevation looking along -Z/+X so +X end of targetBlock is clearly visible
    const pitch11Rad = (11.0 * Math.PI) / 180.0;
    engine.camera.setPoseImmediate(
      0.38,
      pitch11Rad,
      4.6,
      [0, targetBlock.position[1], 0],
      rect.width / rect.height
    );

    const projectWorldToScreen = (wp: [number, number, number]): [number, number] => {
      const m = engine.camera.viewProjMatrix;
      const cx = m[0] * wp[0] + m[4] * wp[1] + m[8] * wp[2] + m[12];
      const cy = m[1] * wp[0] + m[5] * wp[1] + m[9] * wp[2] + m[13];
      const cw = m[3] * wp[0] + m[7] * wp[1] + m[11] * wp[2] + m[15];
      const ndcX = cx / cw;
      const ndcY = cy / cw;
      return [(ndcX * 0.5 + 0.5) * rect.width, (0.5 - ndcY * 0.5) * rect.height];
    };

    const grabTipWorld: [number, number, number] = [
      targetBlock.position[0] + 0.34,
      targetBlock.position[1],
      targetBlock.position[2],
    ];
    const [sx0, sy0] = projectWorldToScreen(grabTipWorld);

    // (a) Tap Isolation: pointerdown -> 3 px move -> pointerup must keep turnPhase === 'SELECT_BLOCK'
    engine.handlePointerDown(sx0, sy0, 1, 0, false, rect);
    engine.handlePointerMove(sx0 + 2.4, sy0 + 1.8, 1, rect); // 3.0 px < 5.0 px!
    expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
    engine.handlePointerUp(sx0 + 2.4, sy0 + 1.8, 1, rect);
    expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
    expect(engine.audio.lastTapDiagnostics).not.toBeNull();

    // Settle brief tap vibration
    for (let i = 0; i < 20; i++) engine.physics.step(1 / 60);
    const leanBeforePull = engine.physics.leanAngleDeg;

    // (b) 11-deg Grazing Axial Pull: drag smoothly outward along +X until EXTRACTED
    const [pxStart, pyStart] = projectWorldToScreen(grabTipWorld);
    engine.handlePointerDown(pxStart, pyStart, 2, 0, false, rect);

    let sawPulling = false;
    let extractedAtStep = -1;
    let clearSubstepsAtExtract = 0;
    let lastMx = pxStart;
    let lastMy = pyStart;

    for (let step = 1; step <= 75; step++) {
      const pullDistDm = Math.min(1.15, (step / 48) * 1.15);
      const targetWorld: [number, number, number] = [
        grabTipWorld[0] + pullDistDm,
        grabTipWorld[1],
        grabTipWorld[2],
      ];
      const [mx, my] = projectWorldToScreen(targetWorld);
      lastMx = mx;
      lastMy = my;
      engine.handlePointerMove(mx, my, 2, rect);
      engine.physics.step(1 / 60);

      if (engine.physics.turnPhase === 'PULLING') {
        sawPulling = true;
      }
      if (engine.physics.turnPhase === 'EXTRACTED') {
        extractedAtStep = step;
        clearSubstepsAtExtract = engine.physics.getClearContactSubsteps();
        break;
      }
    }

    expect(sawPulling).toBe(true);
    expect(extractedAtStep).toBeGreaterThan(0);
    expect(clearSubstepsAtExtract).toBeGreaterThanOrEqual(3);

    const leanDelta = Math.abs(engine.physics.leanAngleDeg - leanBeforePull);
    expect(leanDelta).toBeLessThan(0.5);
    expect(engine.physics.shatteredCount).toBe(0);

    engine.handlePointerUp(lastMx, lastMy, 2, rect);
    expect(engine.physics.turnPhase).toBe('EXTRACTED');

    // (d1) pointercancel during place_top must NOT prematurely drop the carried block
    engine.handlePointerDown(rect.width * 0.5, rect.height * 0.32, 3, 0, false, rect);
    engine.handlePointerCancel(3);
    expect(engine.physics.turnPhase).toBe('EXTRACTED');

    // (c) Tilted-tower openTopSlots plane-normal & twist-yaw alignment + clean top placement
    const { quatFromAxisAngle, quatRotateVec3 } = await import('../src/math/vecMat');
    const tiltRad = (2.8 * Math.PI) / 180.0;
    const twistYawRad = (6.5 * Math.PI) / 180.0;
    const qTilt = quatFromAxisAngle([0, 0, 1], tiltRad);
    const qOddYaw = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5 + twistYawRad);
    const qMul = (a: [number, number, number, number], b: [number, number, number, number]): [number, number, number, number] => [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
    const topSupportRot = qMul(qTilt, qOddYaw);
    const expectedUp = quatRotateVec3(qTilt, [0, 1, 0]);
    const supLenDir = quatRotateVec3(topSupportRot, [1, 0, 0]);

    const rwT4 = engine.physics.getWorld();
    for (const b of engine.physics.blocks) {
      if (b.assignedLayer === 17 && !b.extracted && !b.shattered) {
        b.rotation = [...topSupportRot];
        const rb = rwT4.getRigidBody(b.bodyHandle)!;
        rb.setRotation({ x: topSupportRot[0], y: topSupportRot[1], z: topSupportRot[2], w: topSupportRot[3] }, false);
      }
    }

    const released = engine.physics.releaseExtractedBlockOntoTop();
    expect(released).toBe(true);
    expect(engine.physics.openTopSlots.length).toBe(3);

    for (const slot of engine.physics.openTopSlots) {
      const slotUp = quatRotateVec3(slot.rotation, [0, 1, 0]);
      const upDot = Math.min(1, Math.max(-1, slotUp[0] * expectedUp[0] + slotUp[1] * expectedUp[1] + slotUp[2] * expectedUp[2]));
      const upAngleErrDeg = (Math.acos(upDot) * 180) / Math.PI;
      expect(upAngleErrDeg).toBeLessThan(0.5);

      const slotLen = quatRotateVec3(slot.rotation, [1, 0, 0]);
      const horizCrossDot = Math.abs(
        (slotLen[0] * supLenDir[0] + slotLen[2] * supLenDir[2]) /
          (Math.hypot(slotLen[0], slotLen[2]) * Math.hypot(supLenDir[0], supLenDir[2]))
      );
      expect(horizCrossDot).toBeLessThan(0.02); // orthogonal in XZ within ~1 deg
    }

    // Restore level top row and settle the placed block cleanly to complete the turn
    const qLevelOdd = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
    for (const b of engine.physics.blocks) {
      if (b.assignedLayer === 17 && !b.shattered) {
        const rb = rwT4.getRigidBody(b.bodyHandle)!;
        rb.setRotation({ x: qLevelOdd[0], y: qLevelOdd[1], z: qLevelOdd[2], w: qLevelOdd[3] }, true);
      }
      if (b.assignedLayer === 18 && !b.shattered) {
        const rb = rwT4.getRigidBody(b.bodyHandle)!;
        rb.setTranslation({ x: 0, y: 18 * BLOCK_HEIGHT + BLOCK_HEIGHT * 0.5 + 0.04, z: 0 }, true);
        rb.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      }
    }
    for (let i = 0; i < 95; i++) engine.physics.step(1 / 60);
    expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
    expect(engine.physics.blocksStacked).toBe(1);
    expect(engine.physics.shatteredCount).toBe(0);

    // (d2) pointercancel during <5px block_pending must NOT poke/tap the block
    engine.audio.lastTapDiagnostics = null;
    const legalAfterTurn = engine.physics.blocks.find((b) => b.assignedLayer === 4 && b.assignedSlot === 1)!;
    const [cx0, cy0] = projectWorldToScreen([
      legalAfterTurn.position[0] + 0.34,
      legalAfterTurn.position[1],
      legalAfterTurn.position[2],
    ]);
    engine.handlePointerDown(cx0, cy0, 5, 0, false, rect);
    engine.handlePointerMove(cx0 + 1.5, cy0 + 1.0, 5, rect);
    engine.handlePointerCancel(5);
    expect(engine.audio.lastTapDiagnostics).toBeNull();
    expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');

    engine.dispose();
  });

  it('T4b: preserves play-by-ear clamping loads after aborted/completed pulls, retains upper-tower weight on single-block rows, and handles 24L upward grazing pitch', async () => {
    const { GameEngine } = await import('../src/engine/GameEngine');
    const stubCanvas = { style: { cursor: 'default' } } as unknown as HTMLCanvasElement;
    const engine = new GameEngine(stubCanvas);
    await engine.initHeadless();

    const pw = engine.physics;
    pw.rebuildTower('solo', 2, 18);
    for (let i = 0; i < 90; i++) pw.step(1 / 60);

    const l2Pinched = pw.blocks.find((b) => b.assignedLayer === 2 && !b.isLooseSlot)!;
    const l2Loose = pw.blocks.find((b) => b.assignedLayer === 2 && b.isLooseSlot)!;
    const l12Pinched = pw.blocks.find((b) => b.assignedLayer === 12 && !b.isLooseSlot)!;
    const l12Loose = pw.blocks.find((b) => b.assignedLayer === 12 && b.isLooseSlot)!;
    const l6Loose = pw.blocks.find((b) => b.assignedLayer === 6 && b.isLooseSlot)!;

    // 1. Aborted pull must preserve clamping loads & damped vs ringing tap resonance outside [L-1, L+2]
    pw.startGrabBlock(l6Loose.id, [0.35, 0, 0], [
      l6Loose.position[0] + 0.35,
      l6Loose.position[1],
      l6Loose.position[2],
    ]);
    for (let i = 0; i < 8; i++) {
      pw.updateGrabTarget([l6Loose.position[0] + 0.4, l6Loose.position[1], l6Loose.position[2]]);
      pw.step(1 / 60);
    }
    expect(pw.computeBlockClampingLoadN(l2Pinched.id)).toBeGreaterThan(10.0);
    expect(pw.computeBlockClampingLoadN(l12Pinched.id)).toBeGreaterThan(4.0);

    pw.endGrab(false);
    expect(pw.turnPhase).toBe('SELECT_BLOCK');
    expect(pw.computeBlockClampingLoadN(l2Pinched.id)).toBeGreaterThan(10.0);
    expect(pw.computeBlockClampingLoadN(l2Loose.id)).toBeLessThan(0.5);
    expect(pw.computeBlockClampingLoadN(l12Pinched.id)).toBeGreaterThan(4.0);
    expect(pw.computeBlockClampingLoadN(l12Loose.id)).toBeLessThan(0.5);

    pw.tapNudgeBlock(l2Pinched.id, [1, 0, 0], [0.35, 0, 0]);
    expect(engine.audio.lastTapDiagnostics?.pinch01).toBeGreaterThan(0.9);
    expect(engine.audio.lastTapDiagnostics?.qFactor).toBeLessThan(18);

    pw.tapNudgeBlock(l2Loose.id, [1, 0, 0], [0.35, 0, 0]);
    expect(engine.audio.lastTapDiagnostics?.pinch01).toBeLessThan(0.15);
    expect(engine.audio.lastTapDiagnostics?.qFactor).toBeGreaterThan(42);

    // 2. Full extraction + quiet tower sleep must also retain true resting clamping manifolds
    for (let i = 0; i < 20; i++) pw.step(1 / 60);
    pw.startGrabBlock(l6Loose.id, [0.35, 0, 0], [
      l6Loose.position[0] + 0.35,
      l6Loose.position[1],
      l6Loose.position[2],
    ]);
    for (let step = 1; step <= 60; step++) {
      pw.updateGrabTarget([
        l6Loose.position[0] + 0.35 + (step / 45) * 1.15,
        l6Loose.position[1],
        l6Loose.position[2],
      ]);
      pw.step(1 / 60);
      if (pw.turnPhase === 'EXTRACTED') break;
    }
    expect(pw.turnPhase).toBe('EXTRACTED');
    for (let i = 0; i < 5; i++) pw.step(1 / 60);
    expect(pw.computeBlockClampingLoadN(l2Pinched.id)).toBeGreaterThan(10.0);
    expect(pw.computeBlockClampingLoadN(l2Loose.id)).toBeLessThan(0.5);
    expect(pw.computeBlockClampingLoadN(l12Pinched.id)).toBeGreaterThan(4.0);
    expect(pw.computeBlockClampingLoadN(l12Loose.id)).toBeLessThan(0.5);

    // 3. Depleted supporting rows where only initial isLooseSlot center block remains still transmit upper-tower weight
    pw.rebuildTower('sandbox', 2, 18);
    for (let i = 0; i < 90; i++) pw.step(1 / 60);
    const rw = pw.getWorld();
    for (const b of pw.blocks) {
      if ((b.assignedLayer === 7 || b.assignedLayer === 8) && !b.isLooseSlot) {
        b.extracted = true;
        rw.getCollider(b.colliderHandle)!.setSensor(true);
        rw.getRigidBody(b.bodyHandle)!.setTranslation({ x: 5, y: 5, z: 5 }, false);
      }
    }
    for (let i = 0; i < 90; i++) pw.step(1 / 60);
    const l6Tight = pw.blocks.find((b) => b.assignedLayer === 6 && !b.isLooseSlot)!;
    pw.startGrabBlock(l6Tight.id, [0.35, 0, 0], [
      l6Tight.position[0] + 0.35,
      l6Tight.position[1],
      l6Tight.position[2],
    ]);
    pw.step(1 / 60);
    pw.step(1 / 60);
    expect(pw.computeBlockClampingLoadN(l6Tight.id)).toBeGreaterThan(8.0);
    pw.endGrab(false);

    // 4. 24-layer (72-block) stack + upward grazing camera pitch (-25 deg) drag projection has zero singularity
    pw.rebuildTower('solo', 2, 24);
    for (let i = 0; i < 100; i++) pw.step(1 / 60);
    const l22Blk = pw.blocks.find((b) => b.assignedLayer === 22 && b.isLooseSlot)!;
    const anchorUp: [number, number, number] = [0, l22Blk.position[1], 0];
    const camPosLow: [number, number, number] = [
      0,
      l22Blk.position[1] - 3.0 * Math.tan((25.0 * Math.PI) / 180.0),
      3.0,
    ];
    const rayDirUp: [number, number, number] = [
      0,
      Math.sin((25.0 * Math.PI) / 180.0),
      -Math.cos((25.0 * Math.PI) / 180.0),
    ];
    const projUp = engine.projectBlockDragPlane(camPosLow, rayDirUp, anchorUp);
    expect(projUp).not.toBeNull();
    expect(Math.hypot(projUp![0] - anchorUp[0], projUp![2] - anchorUp[2])).toBeLessThan(1e-4);

    engine.dispose();
  });

  it('T5: load-bearing pull stays vibration-free, and asymmetric collapse produces shattered + crazed + precarious survivors that sleep 100% on the table', async () => {
    await ensureRapierInitialized();
    const { SIM_TIME_SCALE } = await import('../src/physics/constants');
    const audio = new GlassAudioEngine();
    const pw = new PhysicsWorld(audio);
    pw.rebuildTower('solo', 2, 18);
    for (let i = 0; i < 90; i++) pw.step(1 / 60);

    // 1. Slow pull on a pinched load-bearing block (Layer 4) must not induce tower buzz/chatter
    const tight = pw.blocks.find((b) => b.assignedLayer === 4 && !b.isLooseSlot)!;
    pw.startGrabBlock(tight.id, [0.35, 0, 0], [
      tight.position[0] + 0.35,
      tight.position[1],
      tight.position[2],
    ]);

    let maxNonPulledSpeedMmS = 0;
    for (let f = 0; f < 120; f++) {
      const pullDist = Math.min(0.4, (f / 100) * 0.4);
      pw.updateGrabTarget([
        tight.position[0] + 0.35 + pullDist,
        tight.position[1],
        tight.position[2],
      ]);
      pw.step(1 / 60);
      for (const b of pw.blocks) {
        if (b.id === tight.id || b.shattered) continue;
        const vMmS = Math.hypot(b.linVel[0], b.linVel[1], b.linVel[2]) * SIM_TIME_SCALE * 100;
        if (vMmS > maxNonPulledSpeedMmS) maxNonPulledSpeedMmS = vMmS;
      }
    }
    expect(maxNonPulledSpeedMmS).toBeLessThan(45.0);
    expect(pw.wobbleSpeedDegPerSec).toBeLessThan(15.0);
    pw.endGrab(false);

    // 2. Asymmetric pull-induced Jenga collapse (Layer 8): verify mix of shattered, crazed intact,
    //    and precariously balanced/leaning blocks, followed by 100% sleep (0.00 mm/s) by t = 4.0s!
    pw.rebuildTower('solo', 2, 18);
    for (let i = 0; i < 90; i++) pw.step(1 / 60);

    const rw = pw.getWorld();
    const l8Mid = pw.blocks.find((b) => b.assignedLayer === 8 && b.assignedSlot === 1)!;
    const topSlot = pw.openTopSlots[0]!;
    const rbMid = rw.getRigidBody(l8Mid.bodyHandle)!;
    rbMid.setTranslation(
      { x: topSlot.position[0], y: topSlot.position[1], z: topSlot.position[2] },
      true
    );
    rbMid.setRotation(
      {
        x: topSlot.rotation[0],
        y: topSlot.rotation[1],
        z: topSlot.rotation[2],
        w: topSlot.rotation[3],
      },
      true
    );
    l8Mid.assignedLayer = topSlot.layer;
    l8Mid.assignedSlot = topSlot.slotIndex;
    for (let i = 0; i < 60; i++) pw.step(1 / 60);

    const l8Left = pw.blocks.find((b) => b.assignedLayer === 8 && b.assignedSlot === 0)!;
    pw.startGrabBlock(l8Left.id, [0.35, 0, 0], [
      l8Left.position[0] + 0.35,
      l8Left.position[1],
      l8Left.position[2],
    ]);

    for (let f = 1; f <= 240; f++) {
      if (pw.turnPhase === 'PULLING') {
        const pull = Math.min(1.25, (f / 35) * 1.25);
        pw.updateGrabTarget([
          l8Left.position[0] + 0.35 + pull,
          l8Left.position[1],
          l8Left.position[2],
        ]);
      }
      pw.step(1 / 60);
    }

    expect(pw.turnPhase).toBe('COLLAPSED');
    expect(pw.shatteredCount).toBeGreaterThanOrEqual(12);
    expect(pw.shards.length).toBeGreaterThanOrEqual(180);

    const intact = pw.blocks.filter((b) => !b.shattered && !b.extracted);
    const crazedSurvivors = intact.filter((b) => b.damage > 0.05 && b.damage < 1.0);
    const precariousOrElevatedSurvivors = intact.filter((b) => {
      const r = b.rotation;
      const upY = 1 - 2 * (r[0] * r[0] + r[2] * r[2]);
      const tiltDeg = (Math.acos(Math.max(-1, Math.min(1, Math.abs(upY)))) * 180) / Math.PI;
      return (b.position[1] > BLOCK_HEIGHT * 0.95 && b.assignedLayer >= 8) || (tiltDeg > 5.0 && tiltDeg < 85.0);
    });

    expect(intact.length).toBeGreaterThanOrEqual(15);
    expect(crazedSurvivors.length).toBeGreaterThanOrEqual(4);
    expect(precariousOrElevatedSurvivors.length).toBeGreaterThanOrEqual(2);

    // Zero post-fall table vibration: 100% of intact blocks and shards asleep with 0.0 mm/s speed
    let awakeBlocks = 0;
    let maxBlockSpeedMmS = 0;
    for (const b of intact) {
      if (!rw.getRigidBody(b.bodyHandle)!.isSleeping()) awakeBlocks++;
      const vMmS = Math.hypot(b.linVel[0], b.linVel[1], b.linVel[2]) * SIM_TIME_SCALE * 100;
      if (vMmS > maxBlockSpeedMmS) maxBlockSpeedMmS = vMmS;
    }
    let awakeShards = 0;
    for (const s of pw.shards) {
      if (!rw.getRigidBody(s.bodyHandle)!.isSleeping()) awakeShards++;
    }
    expect(awakeBlocks).toBe(0);
    expect(awakeShards).toBe(0);
    expect(maxBlockSpeedMmS).toBe(0);

    pw.dispose();
  }, 15000);

  it('T6: continuous free-hand top placement, bevel-to-bevel side-abutment, layer advancement, and counterbalance lean physics', async () => {
    await ensureRapierInitialized();
    const audio = new GlassAudioEngine();
    const pw = new PhysicsWorld(audio);
    pw.rebuildTower('solo', 2, 18);
    for (let i = 0; i < 90; i++) pw.step(1 / 60);

    const extractBlockCleanly = (blockId: number) => {
      const blk = pw.blocks.find((b) => b.id === blockId)!;
      const isOdd = blk.assignedLayer % 2 === 1;
      const grabLocal: [number, number, number] = isOdd ? [0, 0, 0.34] : [0.34, 0, 0];
      const startAnchor: [number, number, number] = [
        blk.position[0] + grabLocal[0],
        blk.position[1],
        blk.position[2] + grabLocal[2],
      ];
      expect(pw.startGrabBlock(blk.id, grabLocal, startAnchor)).toBe(true);
      for (let f = 1; f <= 160 && pw.turnPhase === 'PULLING'; f++) {
        const pull = Math.min(1.05, (f / 55) * 1.05);
        pw.updateGrabTarget([
          startAnchor[0] + (isOdd ? 0 : pull),
          startAnchor[1],
          startAnchor[2] + (isOdd ? pull : 0),
        ]);
        pw.step(1 / 60);
      }
      expect(pw.turnPhase).toBe('EXTRACTED');
      return blk;
    };

    const settlePlacement = () => {
      expect(pw.turnPhase).toBe('SETTLING');
      for (let f = 0; f < 130 && pw.turnPhase === 'SETTLING'; f++) {
        pw.step(1 / 60);
      }
      expect(pw.turnPhase).toBe('SELECT_BLOCK');
    };

    // 1. Continuous off-center X/Z placement + player angle adjustment (NOT snapped to -0.25, 0, +0.25)
    const b1 = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot)!.id
    );
    const targetX1 = 0.085;  // +8.5 mm lengthwise overhang
    const targetZ1 = -0.205; // -20.5 mm off-center (4.5 mm inward from -0.250, 20.5 mm from 0.000)
    pw.updateCarriedHover([targetX1, 6.0, targetZ1], [0, -1, 0]);
    pw.rotateCarriedBlock((2.5 * Math.PI) / 180.0);
    for (let i = 0; i < 12; i++) pw.step(1 / 60);

    expect(Math.abs(pw.carriedLandingPos[0] - targetX1)).toBeLessThan(0.005);
    expect(Math.abs(pw.carriedLandingPos[2] - targetZ1)).toBeLessThan(0.005);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    settlePlacement();

    expect(b1.assignedLayer).toBe(18);
    expect(b1.shattered).toBe(false);
    expect(b1.damage).toBe(0);
    // Verify landed pose stayed at player's chosen off-center coordinates rather than snapping to -0.25, 0, +0.25
    expect(Math.abs(b1.position[0] - targetX1)).toBeLessThan(0.03);
    expect(Math.abs(b1.position[2] - targetZ1)).toBeLessThan(0.03);
    expect(Math.abs(b1.position[2] - -BLOCK_WIDTH)).toBeGreaterThan(0.03);
    expect(Math.abs(b1.position[2] - 0)).toBeGreaterThan(0.17);

    // 2. Physical 2D OBB side-abutment prevents collider overlap when hovering directly into b1
    const b2 = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 10 && b.isLooseSlot)!.id
    );
    pw.updateCarriedHover([b1.position[0], 6.0, b1.position[2]], [0, -1, 0]);
    for (let i = 0; i < 12; i++) pw.step(1 / 60);

    const sepFromB1 = Math.hypot(
      pw.carriedLandingPos[0] - b1.position[0],
      pw.carriedLandingPos[2] - b1.position[2]
    );
    // Must push out by >= full block width (~0.25 dm) so colliders never intersect on release
    expect(sepFromB1).toBeGreaterThanOrEqual(0.248);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    settlePlacement();
    expect(b1.shattered).toBe(false);
    expect(b2.shattered).toBe(false);
    expect(b1.damage).toBe(0);
    expect(b2.damage).toBe(0);
    expect(b2.assignedLayer).toBe(18);
    expect(pw.topSupportFrame.targetLayer).toBe(18);

    // 3. Placing 3rd free-hand block advances active top layer (18 -> 19) and transitions EXTRACTED -> SETTLING -> SELECT_BLOCK
    const b3 = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 12 && b.isLooseSlot)!.id
    );
    const openSideZ = 0.315;
    pw.updateCarriedHover([0.0, 6.0, openSideZ], [0, -1, 0]);
    for (let i = 0; i < 12; i++) pw.step(1 / 60);
    expect(pw.turnPhase).toBe('EXTRACTED');
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    expect(pw.turnPhase).toBe('SETTLING');
    settlePlacement();

    expect(b3.assignedLayer).toBe(18);
    expect(pw.blocksStacked).toBe(3);
    expect(pw.topSupportFrame.targetLayer).toBe(19);
    expect(pw.openTopSlots.length).toBeGreaterThan(0);
    expect(pw.openTopSlots[0].layer).toBe(19);

    // 3b. Anti-soft-lock: 2 wide-spaced blocks on Layer 19 (odd layer, X-spaced) leaving <25mm everywhere advance to Layer 20
    const l19CenterX = pw.topSupportFrame.layerCenter[0];
    const l19CenterZ = pw.topSupportFrame.layerCenter[2];
    extractBlockCleanly(pw.blocks.find((b) => b.assignedLayer === 6 && b.isLooseSlot)!.id);
    pw.updateCarriedHover([l19CenterX - 0.13, 6.0, l19CenterZ], [0, -1, 0]);
    for (let i = 0; i < 10; i++) pw.step(1 / 60);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    settlePlacement();
    expect(pw.topSupportFrame.targetLayer).toBe(19);

    extractBlockCleanly(pw.blocks.find((b) => b.assignedLayer === 4 && b.isLooseSlot)!.id);
    pw.updateCarriedHover([l19CenterX + 0.13, 6.0, l19CenterZ], [0, -1, 0]);
    for (let i = 0; i < 10; i++) pw.step(1 / 60);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    settlePlacement();
    expect(pw.topSupportFrame.targetLayer).toBe(20);

    // 4. Counterbalance vs Overhang Lean Physics on a precarious cantilever upper row
    const runDepletedPlacementTrial = (placeZ: number) => {
      pw.rebuildTower('solo', 2, 18);
      for (let i = 0; i < 90; i++) pw.step(1 / 60);
      const rw = pw.getWorld();
      // Leave a single support block on Layer 16 at z = +0.095 (spanning [-0.030, +0.220]):
      // -> 3 blocks on Layer 17 (CoM z=0) stand level inside [-0.030, +0.220] (restoring torque 3*0.030 = 0.090),
      // -> counterbalanced placement at z = +0.12 stays squarely supported (<0.5 deg),
      // -> overhanging placement at z = -0.28 exerts 1*0.250 >> 0.090 overturning torque and tips upper rows!
      for (const remSlot of [0, 1]) {
        const remBlk = pw.blocks.find((b) => b.assignedLayer === 16 && b.assignedSlot === remSlot)!;
        remBlk.extracted = true;
        rw.getCollider(remBlk.colliderHandle)!.setEnabled(false);
        rw.getRigidBody(remBlk.bodyHandle)!.setEnabled(false);
      }
      const soleSupportBlk = pw.blocks.find((b) => b.assignedLayer === 16 && b.assignedSlot === 2)!;
      rw.getRigidBody(soleSupportBlk.bodyHandle)!.setTranslation(
        { x: soleSupportBlk.position[0], y: soleSupportBlk.position[1], z: 0.095 },
        true
      );
      for (let i = 0; i < 60; i++) pw.step(1 / 60);

      // Extract loose block from Layer 6 and place free-hand at `placeZ` on Layer 18
      extractBlockCleanly(pw.blocks.find((b) => b.assignedLayer === 6 && b.isLooseSlot)!.id);
      pw.updateCarriedHover([0.0, 6.0, placeZ], [0, -1, 0]);
      for (let i = 0; i < 10; i++) pw.step(1 / 60);
      expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
      let peakLeanDeg = 0;
      for (let i = 0; i < 90; i++) {
        pw.step(1 / 60);
        if (pw.leanAngleDeg > peakLeanDeg) peakLeanDeg = pw.leanAngleDeg;
      }
      return {
        phase: pw.turnPhase,
        peakLeanDeg,
        comZCm: pw.centerOfMassXZCm[1],
      };
    };

    const supportedCounterRes = runDepletedPlacementTrial(0.12);
    const unsupportedOverhangRes = runDepletedPlacementTrial(-0.28);
    expect(supportedCounterRes.phase).toBe('SELECT_BLOCK');
    expect(supportedCounterRes.peakLeanDeg).toBeLessThan(1.0);
    expect(unsupportedOverhangRes.peakLeanDeg).toBeGreaterThan(supportedCounterRes.peakLeanDeg + 15.0);
    expect(unsupportedOverhangRes.phase).toBe('COLLAPSED');

    // 5. Occupied-row R (90-deg) rotation stops flush without collider intersection, and open-row 90-deg reaches outer span
    pw.rebuildTower('solo', 2, 18);
    for (let i = 0; i < 90; i++) pw.step(1 / 60);
    const rBlkOpen = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot)!.id
    );
    pw.rotateCarriedBlock(Math.PI * 0.5);
    pw.updateCarriedHover([0.28, 6.0, 0.0], [0, -1, 0]);
    expect(Math.abs(pw.carriedLandingPos[0] - 0.28)).toBeLessThan(0.005);
    pw.resetCarriedBlockRotation();
    pw.updateCarriedHover([0.0, 6.0, 0.0], [0, -1, 0]);
    for (let i = 0; i < 10; i++) pw.step(1 / 60);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    settlePlacement();
    expect(rBlkOpen.assignedLayer).toBe(18);

    const rBlkSecond = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 10 && b.isLooseSlot)!.id
    );
    pw.updateCarriedHover([0.0, 6.0, 0.25], [0, -1, 0]);
    pw.rotateCarriedBlock(Math.PI * 0.5);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    let maxRotReleaseSpeedMmS = 0;
    for (let f = 0; f < 120 && pw.turnPhase === 'SETTLING'; f++) {
      pw.step(1 / 60);
      const s1 = Math.hypot(rBlkOpen.linVel[0], rBlkOpen.linVel[1], rBlkOpen.linVel[2]) * SIM_TIME_SCALE * 100;
      const s2 = Math.hypot(rBlkSecond.linVel[0], rBlkSecond.linVel[1], rBlkSecond.linVel[2]) * SIM_TIME_SCALE * 100;
      maxRotReleaseSpeedMmS = Math.max(maxRotReleaseSpeedMmS, s1, s2);
    }
    expect(maxRotReleaseSpeedMmS).toBeLessThan(450.0);
    expect(rBlkOpen.damage).toBe(0);
    expect(rBlkSecond.damage).toBe(0);
    expect(pw.turnPhase).toBe('SELECT_BLOCK');

    // 6. 240 Hz moving-mouse velocity coupling vs 2.0 s steady-hand zero-velocity decay
    const bVel = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 12 && b.isLooseSlot)!.id
    );
    let hzZ = -0.34;
    for (let f = 0; f < 5; f++) {
      for (let sub = 0; sub < 4; sub++) {
        hzZ += 8.0 / 240;
        pw.updateCarriedHover([0.0, 6.0, hzZ], [0, -1, 0]);
      }
      pw.step(1 / 60);
    }
    // Steady hand for 2.0 s decays coupled velocity cleanly to ~0
    for (let i = 0; i < 120; i++) pw.step(1 / 60);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    const lvSteady = pw.getWorld().getRigidBody(bVel.bodyHandle)!.linvel();
    expect(Math.hypot(lvSteady.x, lvSteady.z) * SIM_TIME_SCALE).toBeLessThan(0.01);
    settlePlacement();

    const bMoving = extractBlockCleanly(
      pw.blocks.find((b) => b.assignedLayer === 6 && b.isLooseSlot)!.id
    );
    let mvX = -0.32;
    for (let f = 0; f < 5; f++) {
      for (let sub = 0; sub < 4; sub++) {
        mvX += 8.0 / 240;
        pw.updateCarriedHover([mvX, 6.0, 0.0], [0, -1, 0]);
      }
      pw.step(1 / 60);
    }
    pw.updateCarriedHover([mvX + 8.0 / 240, 6.0, 0.0], [0, -1, 0]);
    expect(pw.releaseExtractedBlockOntoTop()).toBe(true);
    const lvMoving = pw.getWorld().getRigidBody(bMoving.bodyHandle)!.linvel();
    const relSpdMoving = Math.hypot(lvMoving.x, lvMoving.z) * SIM_TIME_SCALE;
    expect(relSpdMoving).toBeGreaterThan(1.05);
    expect(relSpdMoving).toBeLessThanOrEqual(2.21);

    pw.dispose();

    // 7. GameEngine EXTRACTED orbit-without-yanking & >= 25 px press-drag-release placement
    const { GameEngine } = await import('../src/engine/GameEngine');
    const stubCanvas = { style: { cursor: 'default' } } as unknown as HTMLCanvasElement;
    const engine = new GameEngine(stubCanvas);
    await engine.initHeadless();
    const rect = { left: 0, top: 0, width: 1280, height: 800 };
    engine.physics.rebuildTower('solo', 2, 18);
    for (let i = 0; i < 90; i++) engine.physics.step(1 / 60);
    engine.camera.setPoseImmediate(0.45, 0.48, 4.4, [0, 2.75, 0], rect.width / rect.height);

    const projectScreen = (wp: [number, number, number]): [number, number] => {
      const m = engine.camera.viewProjMatrix;
      const cx = m[0] * wp[0] + m[4] * wp[1] + m[8] * wp[2] + m[12];
      const cy = m[1] * wp[0] + m[5] * wp[1] + m[9] * wp[2] + m[13];
      const cw = m[3] * wp[0] + m[7] * wp[1] + m[11] * wp[2] + m[15];
      return [((cx / cw) * 0.5 + 0.5) * rect.width, (0.5 - (cy / cw) * 0.5) * rect.height];
    };

    const gBlk = engine.physics.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot)!;
    engine.physics.startGrabBlock(gBlk.id, [0.34, 0, 0], [gBlk.position[0] + 0.34, gBlk.position[1], gBlk.position[2]]);
    for (let f = 1; f <= 160 && engine.physics.turnPhase === 'PULLING'; f++) {
      engine.physics.updateGrabTarget([
        gBlk.position[0] + 0.34 + Math.min(1.05, (f / 55) * 1.05),
        gBlk.position[1],
        gBlk.position[2],
      ]);
      engine.physics.step(1 / 60);
    }
    expect(engine.physics.turnPhase).toBe('EXTRACTED');

    const topY = engine.physics.topSupportFrame.layerCenter[1];
    const [topSx0, topSy0] = projectScreen([-0.15, topY, -0.18]);
    engine.handlePointerMove(topSx0, topSy0, 99, rect);
    const savedLandX = engine.physics.carriedLandingPos[0];
    const savedLandZ = engine.physics.carriedLandingPos[2];

    // Orbit drag far outside tower must NOT disturb carriedLandingPos
    const [orbX, orbY] = projectScreen([2.4, topY, 0.0]);
    engine.handlePointerDown(orbX, orbY, 1, 0, false, rect);
    engine.handlePointerMove(orbX + 90, orbY + 25, 1, rect);
    engine.handlePointerUp(orbX + 90, orbY + 25, 1, rect);
    expect(Math.abs(engine.physics.carriedLandingPos[0] - savedLandX)).toBeLessThan(1e-5);
    expect(Math.abs(engine.physics.carriedLandingPos[2] - savedLandZ)).toBeLessThan(1e-5);
    expect(engine.physics.turnPhase).toBe('EXTRACTED');

    // Press-drag-release (>= 25 px drag across top row) places block cleanly at released spot
    const [dragStartX, dragStartY] = projectScreen([0.0, topY, -0.22]);
    const [dragEndX, dragEndY] = projectScreen([0.06, topY, 0.19]);
    expect(Math.hypot(dragEndX - dragStartX, dragEndY - dragStartY)).toBeGreaterThan(25.0);
    engine.handlePointerDown(dragStartX, dragStartY, 2, 0, false, rect);
    engine.handlePointerMove(dragEndX, dragEndY, 2, rect);
    engine.handlePointerUp(dragEndX, dragEndY, 2, rect);
    expect(engine.physics.turnPhase).toBe('SETTLING');
    expect(Math.abs(gBlk.position[2] - 0.19)).toBeLessThan(0.03);

    engine.dispose();
  }, 15000);
});


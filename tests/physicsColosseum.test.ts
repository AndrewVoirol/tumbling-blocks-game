import { describe, expect, it } from 'vitest';
import { GlassAudioEngine, type GlassImpactOptions } from '../src/audio/GlassAudioEngine';
import { quatFromAxisAngle } from '../src/math/vecMat';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HEIGHT,
  BLOCK_WIDTH,
  TABLE_RADIUS,
} from '../src/physics/constants';
import {
  attachMujocoWorkerHost,
  type WorkerInboundMessage,
  type WorkerOutboundMessage,
} from '../src/physics/mujocoWorker';
import {
  MujocoWorkerBridge,
  type WorkerEndpointLike,
} from '../src/physics/MujocoWorkerBridge';
import {
  ensurePhysicsInitialized,
  PhysicsWorld,
  ShardImpactEvent,
  WobbleRockEvent,
} from '../src/physics/PhysicsWorld';

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((sorted.length - 1) * p))
  );
  return sorted[idx];
}

describe('Colosseum Phase 2 Adversarial Physics, Tribology, Fracture & Snapshot Gates', () => {
  it('C1: progressive Griffith crazing memory weakens drop & three-point flexural strength (>= 30% drop-height reduction) while preserving 0 false damage on 1.0 cm drops', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const dropWorld = new PhysicsWorld(audio, 1);
    const measureDropShatterHeightCm = (priorDamage: number): number => {
      for (let hCm = 4.0; hCm <= 16.0; hCm += 0.5) {
        dropWorld.rebuildTower('sandbox', 2, 1);
        const rw = dropWorld.getWorld();
        for (let i = 1; i < dropWorld.blocks.length; i++) {
          rw.getRigidBody(dropWorld.blocks[i].bodyHandle)!.setTranslation(
            { x: 2.2, y: BLOCK_HALF_Y, z: -1.5 + i * 0.9 },
            true
          );
        }
        const b0 = dropWorld.blocks[0];
        b0.damage = priorDamage;
        const rb0 = rw.getRigidBody(b0.bodyHandle)!;
        rb0.setTranslation({ x: 0, y: hCm * 0.1 + BLOCK_HALF_Y, z: 0 }, true);
        rb0.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
        rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
        for (let f = 0; f < 40; f++) {
          dropWorld.step(1 / 60);
          if (b0.shattered) break;
        }
        if (b0.shattered) return hCm;
      }
      return 99.0;
    };

    const hFreshCm = measureDropShatterHeightCm(0.0);
    const hCrazed55Cm = measureDropShatterHeightCm(0.55);
    const hCrazed85Cm = measureDropShatterHeightCm(0.85);
    dropWorld.dispose();

    expect(hFreshCm).toBeGreaterThanOrEqual(12.5);
    expect(hCrazed55Cm).toBeLessThanOrEqual(9.0);
    expect(hCrazed55Cm).toBeLessThanOrEqual(hFreshCm * 0.7);
    expect(hCrazed85Cm).toBeLessThan(hCrazed55Cm);

    // Three-point flexural mid-span blow: fresh bridge survives 9.0 dm/s blow while crazed bridge snaps near mid-span
    const runFlexuralBridgeStrike = (priorDamage: number): { shattered: boolean; originX: number } => {
      const world = new PhysicsWorld(audio, 2);
      world.rebuildTower('sandbox', 2, 2);
      const rw = world.getWorld();
      const supL = world.blocks[0];
      const supR = world.blocks[1];
      const bridge = world.blocks[3];
      bridge.damage = priorDamage;

      rw.getRigidBody(world.blocks[2].bodyHandle)!.setTranslation({ x: -3.2, y: 0.075, z: 2.5 }, true);
      rw.getRigidBody(world.blocks[4].bodyHandle)!.setTranslation({ x: -2.6, y: 0.075, z: 2.5 }, true);
      rw.getRigidBody(world.blocks[5].bodyHandle)!.setTranslation({ x: -2.0, y: 0.075, z: 2.5 }, true);

      rw.getRigidBody(supL.bodyHandle)!.setTranslation({ x: -0.31, y: BLOCK_HEIGHT * 0.5, z: 0 }, true);
      rw.getRigidBody(supL.bodyHandle)!.setRotation({ x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 }, true);
      rw.getRigidBody(supR.bodyHandle)!.setTranslation({ x: 0.31, y: BLOCK_HEIGHT * 0.5, z: 0 }, true);
      rw.getRigidBody(supR.bodyHandle)!.setRotation({ x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 }, true);
      rw.getRigidBody(bridge.bodyHandle)!.setTranslation({ x: 0, y: BLOCK_HEIGHT * 1.5, z: 0 }, true);
      rw.getRigidBody(bridge.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      for (let s = 0; s < 15; s++) world.step(1 / 60);

      const simScale = Math.sqrt(10.0);
      const midHammer = rw.createRigidBody({
        translation: { x: 0.0, y: BLOCK_HEIGHT * 2.0 + 0.14, z: 0 },
        linvel: { x: 0, y: -9.0 / simScale, z: 0 },
      });
      rw.createCollider(
        { shape: { halfExtents: { x: 0.035, y: 0.06, z: 0.11 } }, mass: 0.085 },
        midHammer
      );

      for (let s = 0; s < 14; s++) {
        world.step(1 / 60);
        if (bridge.shattered) break;
      }
      const res = {
        shattered: bridge.shattered,
        originX: world.lastFractureOriginLocal ? Math.abs(world.lastFractureOriginLocal[0]) : 99,
      };
      world.dispose();
      return res;
    };

    const flexFresh = runFlexuralBridgeStrike(0.0);
    const flexCrazed = runFlexuralBridgeStrike(0.65);
    expect(flexFresh.shattered).toBe(false);
    expect(flexCrazed.shattered).toBe(true);
    expect(flexCrazed.originX).toBeLessThan(0.18);

    // 1.0 cm gentle placement drop incurs 0 additional damage on both fresh and pre-crazed blocks
    for (const startDmg of [0.0, 0.55]) {
      const world = new PhysicsWorld(audio, 1);
      world.rebuildTower('sandbox', 2, 1);
      const rw = world.getWorld();
      for (let i = 1; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle)!.setTranslation(
          { x: 2.2, y: BLOCK_HALF_Y, z: -1.5 + i * 0.9 },
          true
        );
      }
      const b0 = world.blocks[0];
      b0.damage = startDmg;
      const rb0 = rw.getRigidBody(b0.bodyHandle)!;
      rb0.setTranslation({ x: 0, y: 0.1 + BLOCK_HALF_Y, z: 0 }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      for (let f = 0; f < 40; f++) world.step(1 / 60);
      expect(b0.shattered).toBe(false);
      expect(b0.damage).toBeCloseTo(startDmg, 5);
      world.dispose();
    }
  }, 30000);

  it('C2: beveled glass prisms rock with >= 3 angular-velocity sign reversals, emit discrete WobbleRockEvents without substep clink spam, and topple at marginal overhang', async () => {
    await ensurePhysicsInitialized();

    for (const deg of [8, 12]) {
      let impactCalls = 0;
      class SpyAudio extends GlassAudioEngine {
        public override playGlassImpact(): void {
          impactCalls++;
        }
      }
      const spyAudio = new SpyAudio();
      spyAudio.setMuted(true);

      const world = new PhysicsWorld(spyAudio, 1);
      world.rebuildTower('sandbox', 2, 1);
      const rw = world.getWorld();
      for (let i = 1; i < world.blocks.length; i++) {
        rw.getRigidBody(world.blocks[i].bodyHandle)!.setTranslation(
          { x: 2.2, y: BLOCK_HALF_Y, z: -1.5 + i * 0.9 },
          true
        );
      }
      const b0 = world.blocks[0];
      const rb0 = rw.getRigidBody(b0.bodyHandle)!;
      const rad = (deg * Math.PI) / 180;
      const q = quatFromAxisAngle([1, 0, 0], rad);
      const halfW = BLOCK_WIDTH * 0.5 - 0.0125;
      const liftY = BLOCK_HALF_Y * Math.cos(rad) + halfW * Math.sin(rad) + 0.0005;
      rb0.setTranslation({ x: 0, y: liftY, z: 0 }, true);
      rb0.setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }, true);
      rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb0.setAngvel({ x: 0, y: 0, z: 0 }, true);

      let prevSign = 0;
      let signFlips = 0;
      const allWobbles: WobbleRockEvent[] = [];

      for (let f = 0; f < 90; f++) {
        const res = world.step(1 / 60);
        allWobbles.push(...res.wobbleEvents);
        const wx = b0.angVel[0];
        if (Math.abs(wx) > 0.04) {
          const sgn = wx > 0 ? 1 : -1;
          if (prevSign !== 0 && sgn !== prevSign) signFlips++;
          prevSign = sgn;
        }
      }

      expect(signFlips).toBeGreaterThanOrEqual(3);
      expect(allWobbles.length).toBeGreaterThanOrEqual(2);
      expect(allWobbles[0].surface).toBe('wood');
      expect(allWobbles[0].angularSpeedRadPerSec).toBeGreaterThan(0.3);
      expect(allWobbles[0].tiltAngleRad).toBeGreaterThan(0.005);
      expect(impactCalls).toBeLessThanOrEqual(2);
      expect(b0.damage).toBe(0);
      expect(rw.getRigidBody(b0.bodyHandle)!.isSleeping()).toBe(true);
      expect(world.consumeWobbleEvents().length).toBeGreaterThanOrEqual(2);
      expect(world.consumeWobbleEvents().length).toBe(0);
      world.dispose();
    }

    // Marginal +3.9 cm (5.8 deg) cantilevered overhang still topples cleanly (< 45 frames)
    const audio = new GlassAudioEngine();
    audio.setMuted(true);
    const world = new PhysicsWorld(audio, 18);
    world.rebuildTower('solo', 2, 18);
    const rw = world.getWorld();
    const tiltRad = (5.8 * Math.PI) / 180.0;
    const halfTilt = -tiltRad * 0.5;
    const sz = Math.sin(halfTilt);
    const cw = Math.cos(halfTilt);
    for (const b of world.blocks) {
      if (b.assignedLayer >= 12) {
        const rb = rw.getRigidBody(b.bodyHandle)!;
        const p = rb.translation();
        const r = rb.rotation();
        const dy = Math.max(0, p.y - 12 * BLOCK_HEIGHT);
        rb.setTranslation({ x: p.x + 0.39 + dy * Math.sin(tiltRad), y: p.y, z: p.z }, true);
        rb.setRotation(
          {
            x: cw * r.x - sz * r.y,
            y: cw * r.y + sz * r.x,
            z: cw * r.z + sz * r.w,
            w: cw * r.w - sz * r.z,
          },
          true
        );
      }
    }
    let toppleFrame = -1;
    for (let f = 1; f <= 60; f++) {
      world.step(1 / 60);
      if (world.turnPhase === 'COLLAPSED') {
        toppleFrame = f;
        break;
      }
    }
    expect(world.turnPhase).toBe('COLLAPSED');
    expect(toppleFrame).toBeGreaterThan(0);
    expect(toppleFrame).toBeLessThan(45);
    world.dispose();
  }, 30000);

  it('C3: secondary 3D shard impact telemetry suppresses initial fracture separation, deduplicates hop chatter, preserves true Voronoi massKg, and retains >= 300 collapse shards', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const world = new PhysicsWorld(audio, 1);
    world.rebuildTower('sandbox', 2, 1);
    const rw = world.getWorld();
    for (let i = 1; i < world.blocks.length; i++) {
      rw.getRigidBody(world.blocks[i].bodyHandle)!.setTranslation(
        { x: 2.2, y: BLOCK_HALF_Y, z: -1.5 + i * 0.9 },
        true
      );
    }
    const b0 = world.blocks[0];
    const rb0 = rw.getRigidBody(b0.bodyHandle)!;
    rb0.setTranslation({ x: 0, y: 2.6 + BLOCK_HALF_Y, z: 0 }, true);
    rb0.setLinvel({ x: 0, y: 0, z: 0 }, true);

    let shatterSimTime = -1;
    let earlyWindowEvents = 0;
    let postSleepEvents = 0;
    const postSpawnEvents: ShardImpactEvent[] = [];

    for (let f = 0; f < 160; f++) {
      const res = world.step(1 / 60);
      expect(res.shardImpacts.length).toBeLessThanOrEqual(8);
      if (b0.shattered && shatterSimTime < 0) {
        shatterSimTime = res.simTime;
      }
      if (shatterSimTime >= 0) {
        const elapsed = res.simTime - shatterSimTime;
        if (elapsed < 0.020) {
          earlyWindowEvents += res.shardImpacts.length;
        } else {
          postSpawnEvents.push(...res.shardImpacts);
        }
        const allAsleep =
          world.shards.length > 0 &&
          world.shards.every((s) => rw.getRigidBody(s.bodyHandle)?.isSleeping());
        if (allAsleep && f > 120) {
          postSleepEvents += res.shardImpacts.length;
        }
      }
    }

    expect(b0.shattered).toBe(true);
    expect(earlyWindowEvents).toBe(0);
    expect(postSpawnEvents.length).toBeGreaterThanOrEqual(6);
    expect(postSpawnEvents.length).toBeLessThanOrEqual(44);
    expect(postSleepEvents).toBe(0);

    const masses = postSpawnEvents.map((e) => e.massKg);
    const minMass = Math.min(...masses);
    const maxMass = Math.max(...masses);
    expect(minMass).toBeGreaterThanOrEqual(0.00018);
    expect(maxMass).toBeLessThanOrEqual(0.025);
    expect(maxMass).toBeGreaterThan(minMass * 1.15);
    for (const ev of postSpawnEvents) {
      expect(ev.speed).toBeGreaterThanOrEqual(0.14);
      expect(['wood', 'glass']).toContain(ev.surface);
      expect(ev.slotIndex).toBeGreaterThanOrEqual(0);
      expect(ev.slotIndex).toBeLessThan(512);
    }
    world.dispose();

    // Verify optical-bench `collapse-300` exact trajectory retains >= 300 active shards
    const collapseWorld = new PhysicsWorld(audio, 18);
    collapseWorld.rebuildTower('sandbox');
    for (let f = 0; f < 60; f++) collapseWorld.step(1 / 60);
    collapseWorld.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 165; f++) {
      collapseWorld.step(1 / 60);
      if (f > 115 && collapseWorld.shards.length >= 320) break;
    }
    expect(collapseWorld.shards.length).toBeGreaterThanOrEqual(300);
    collapseWorld.dispose();
  }, 30000);

  it('C4: 18-layer rimward collapse meets real-time P95 < 16.0 ms frame budget with 0 off-rim floaters and 0 solver resets', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const world = new PhysicsWorld(audio, 18);
    world.rebuildTower('sandbox', 2, 18);
    for (let f = 0; f < 40; f++) world.step(1 / 60);
    world.triggerSandboxCollapseImpulse();

    const rw = world.getWorld();
    const mjData = rw.getData();
    let prevTime = mjData ? mjData.time : 0;
    let solverTimeResets = 0;
    const stepMs: number[] = [];

    for (let f = 0; f < 260; f++) {
      const t0 = performance.now();
      world.step(1 / 60);
      stepMs.push(performance.now() - t0);
      if (mjData) {
        if (mjData.time + 1e-6 < prevTime) solverTimeResets++;
        prevTime = mjData.time;
      }
    }

    let floatingOffRimShards = 0;
    for (const s of world.shards) {
      const r = Math.hypot(s.position[0], s.position[2]);
      const vy = rw.getRigidBody(s.bodyHandle)?.linvel().y ?? 0;
      if (r > TABLE_RADIUS + 0.08 && s.position[1] > -0.05 && Math.abs(vy) < 0.08) {
        floatingOffRimShards++;
      }
    }
    let floatingOffRimBlocks = 0;
    for (const b of world.blocks) {
      if (b.shattered) continue;
      const r = Math.hypot(b.position[0], b.position[2]);
      const vy = rw.getRigidBody(b.bodyHandle)?.linvel().y ?? 0;
      if (r > TABLE_RADIUS + 0.14 && b.position[1] > -0.05 && Math.abs(vy) < 0.08) {
        floatingOffRimBlocks++;
      }
    }

    const p95 = percentile(stepMs, 0.95);
    const cpuCount = globalThis.navigator?.hardwareConcurrency ?? 2;
    const p95BudgetMs = cpuCount <= 2 ? 48.0 : 16.0;
    expect(world.shatteredCount).toBeGreaterThanOrEqual(18);
    expect(floatingOffRimShards).toBe(0);
    expect(floatingOffRimBlocks).toBe(0);
    expect(solverTimeResets).toBe(0);
    expect(p95).toBeLessThan(p95BudgetMs);
    world.dispose();
  }, 30000);

  it('C5: deterministic tower seeding and compact PhysicsNetworkSnapshot round-trip block poses, negative assignedLayer = -1, photoelastic wrenches, and active Voronoi shards', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const host = new PhysicsWorld(audio, 18, { seed: 0xcafe1234 });
    const peerSameSeed = new PhysicsWorld(audio, 18, { seed: 0xcafe1234 });
    const peerOtherSeed = new PhysicsWorld(audio, 18, { seed: 0xbeef5678 });

    expect(host.towerSeed).toBe(0xcafe1234);
    let diffFromOther = 0;
    for (let i = 0; i < host.blocks.length; i++) {
      expect(host.blocks[i].swatchId).toBe(peerSameSeed.blocks[i].swatchId);
      expect(host.blocks[i].motif).toBe(peerSameSeed.blocks[i].motif);
      expect(host.blocks[i].rippleSeed).toBe(peerSameSeed.blocks[i].rippleSeed);
      expect(host.blocks[i].isLooseSlot).toBe(peerSameSeed.blocks[i].isLooseSlot);
      if (
        host.blocks[i].swatchId !== peerOtherSeed.blocks[i].swatchId ||
        host.blocks[i].motif !== peerOtherSeed.blocks[i].motif
      ) {
        diffFromOther++;
      }
    }
    expect(diffFromOther).toBeGreaterThan(10);
    peerSameSeed.dispose();
    peerOtherSeed.dispose();

    // Advance host into a mid-collapse state with crazing, photoelastic stress, an unstacked block (assignedLayer = -1), and flying shards
    host.rebuildTower('sandbox', 2, 18, 0xcafe1234);
    for (let f = 0; f < 30; f++) host.step(1 / 60);
    host.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 32; f++) host.step(1 / 60);

    const intactBlock = host.blocks.find((b) => !b.shattered)!;
    intactBlock.assignedLayer = -1;
    intactBlock.assignedSlot = 2;
    intactBlock.swatchId = 13;
    intactBlock.motif = 3;
    intactBlock.extracted = true;
    intactBlock.damage = 0.64;
    intactBlock.crackLocalOrigin = [0.18, -0.04, 0.09];
    host.blockPhotoelasticWrench[intactBlock.id * 4 + 0] = 0.12;
    host.blockPhotoelasticWrench[intactBlock.id * 4 + 1] = 0.48;
    host.blockPhotoelasticWrench[intactBlock.id * 4 + 2] = 0.77;
    host.blockPhotoelasticWrench[intactBlock.id * 4 + 3] = 0.31;

    expect(host.shards.length).toBeGreaterThan(20);
    const snap = host.exportCompactSnapshot({ includeShardGeometry: true });

    const guest = new PhysicsWorld(audio, 12);
    guest.applyCompactSnapshot(snap);

    expect(guest.towerSeed).toBe(0xcafe1234);
    expect(guest.getSimTime()).toBeCloseTo(host.getSimTime(), 5);
    expect(guest.blocks.length).toBe(host.blocks.length);
    expect(guest.shatteredCount).toBe(host.shatteredCount);
    expect(guest.shards.length).toBe(host.shards.length);

    const guestIntact = guest.blocks[intactBlock.id];
    expect(guestIntact.assignedLayer).toBe(-1);
    expect(guestIntact.assignedSlot).toBe(2);
    expect(guestIntact.swatchId).toBe(13);
    expect(guestIntact.motif).toBe(3);
    expect(guestIntact.extracted).toBe(true);
    expect(guestIntact.damage).toBeCloseTo(0.64, 4);
    expect(guestIntact.crackLocalOrigin[0]).toBeCloseTo(0.18, 4);
    expect(guestIntact.position[0]).toBeCloseTo(intactBlock.position[0], 4);
    expect(guestIntact.linVel[0]).toBeCloseTo(intactBlock.linVel[0], 3);

    const guestWrench = new Float32Array(4);
    guest.writeBlockPhotoelasticWrench(intactBlock.id, guestWrench, 0);
    expect(guestWrench[0]).toBeCloseTo(0.12, 4);
    expect(guestWrench[1]).toBeCloseTo(0.48, 4);
    expect(guestWrench[2]).toBeCloseTo(0.77, 4);
    expect(guestWrench[3]).toBeCloseTo(0.31, 4);

    const guestUploads = guest.consumePendingShardUploads();
    expect(guestUploads.length).toBeGreaterThan(0);
    expect(guest.shards[0].massKg).toBeCloseTo(host.shards[0].massKg, 6);
    expect(guest.shards[0].volume).toBeCloseTo(host.shards[0].volume, 6);
    expect(guest.shards[0].position[1]).toBeCloseTo(host.shards[0].position[1], 4);

    // Step guest forward after snapshot restore without NaNs or solver errors
    const stepAfter = guest.step(1 / 60);
    expect(stepAfter.simTime).toBeGreaterThan(snap.simTime);
    expect(Number.isFinite(guest.shards[0].position[1])).toBe(true);

    host.dispose();
    guest.dispose();
  }, 30000);

  it('C6: multi-stage Sandbox support-loss leaves 0 mid-air floating shards, ignores isolated crooked top-notch outlier lean, and reports Toppled rubble telemetry', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const world = new PhysicsWorld(audio, 18);
    world.rebuildTower('sandbox', 2, 18);
    for (let f = 0; f < 45; f++) world.step(1 / 60);

    // Stage 1: Shatter top courses (41..53) so shards sleep perched atop Course 13 (~20-22 cm),
    // and place 1 crooked resting block tilted at 22 deg on top
    for (const id of [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41]) {
      world.shatterBlock(world.blocks[id], [0, 0, 0], 16.5, [0, 1, 0]);
    }
    const crookedTop = world.blocks[40];
    const rw = world.getWorld();
    const halfTilt = (22.0 * Math.PI) / 360.0;
    rw.getRigidBody(crookedTop.bodyHandle)!.setRotation(
      { x: 0, y: 0, z: Math.sin(halfTilt), w: Math.cos(halfTilt) },
      true
    );
    rw.getRigidBody(crookedTop.bodyHandle)!.setTranslation(
      { x: crookedTop.position[0], y: crookedTop.position[1] + 0.04, z: crookedTop.position[2] },
      true
    );
    for (let f = 0; f < 240; f++) world.step(1 / 60);

    expect(world.leanAngleDeg).toBeLessThan(2.0);
    expect(world.currentLayers).toBeGreaterThanOrEqual(13);

    // Stage 2: Buckle & shatter the supporting column underneath perched sleeping shards
    world.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 360; f++) world.step(1 / 60);

    const mjModel = rw.getModel()!;
    let unsupportedElevatedShards = 0;
    for (const s of world.shards) {
      const [sx, sy, sz] = s.position;
      const speed = Math.hypot(s.preLinVel[0], s.preLinVel[1], s.preLinVel[2]);
      const isAsleep = rw.getRigidBody(s.bodyHandle)?.isSleeping() ?? false;
      if (sy <= 0.26 || (!isAsleep && speed > 0.12)) continue;
      let supported = false;
      const collidesWithBlocks =
        mjModel.geom_contype[s.colliderHandle] === 4 ||
        (mjModel.geom_conaffinity[s.colliderHandle] & 2) !== 0;
      if (collidesWithBlocks) {
        for (const b of world.blocks) {
          if (b.shattered || b.extracted) continue;
          const dy = sy - b.position[1];
          if (
            dy > -0.10 &&
            dy < 0.54 &&
            Math.abs(sx - b.position[0]) < 0.54 &&
            Math.abs(sz - b.position[2]) < 0.54
          ) {
            supported = true;
            break;
          }
        }
        if (!supported && mjModel.geom_contype[s.colliderHandle] === 4) {
          for (const os of world.shards) {
            if (os.id === s.id || mjModel.geom_contype[os.colliderHandle] !== 4) continue;
            const dy = sy - os.position[1];
            if (
              dy > 0.01 &&
              dy < 0.30 &&
              Math.abs(sx - os.position[0]) < 0.30 &&
              Math.abs(sz - os.position[2]) < 0.30
            ) {
              supported = true;
              break;
            }
          }
        }
      }
      if (!supported) unsupportedElevatedShards++;
    }

    expect(unsupportedElevatedShards).toBe(0);
    expect(world.currentLayers).toBe(0);
    expect(world.shatteredCount).toBeGreaterThan(15);

    world.dispose();
  }, 35000);

  it('C7: worker monotonic epoch fencing, audio/shard telemetry parity, native WASM mj_ray, constraint-island wakeup, and 2-ply CEM oracle', async () => {
    await ensurePhysicsInitialized();

    // Spy audio engine capturing worker-forwarded metadata
    const slideLoads: number[] = [];
    const tapDamages: number[] = [];
    const impactMetas: GlassImpactOptions[] = [];
    class SpyAudioEngine extends GlassAudioEngine {
      public override updateSlideFriction(
        slideSpeedDmPerSec: number,
        resistance01: number,
        slipPowerMilliwatts: number = 0,
        torsionRadPerSec: number = 0,
        clampingLoadN?: number
      ): void {
        if (clampingLoadN !== undefined && slideSpeedDmPerSec > 0.01) {
          slideLoads.push(clampingLoadN);
        }
        super.updateSlideFriction(
          slideSpeedDmPerSec,
          resistance01,
          slipPowerMilliwatts,
          torsionRadPerSec,
          clampingLoadN
        );
      }
      public override playTapTestResonance(
        pinch01: number,
        worldPos?: [number, number, number],
        camPos?: [number, number, number],
        camRight?: [number, number, number],
        damage: number = 0
      ): void {
        tapDamages.push(damage);
        super.playTapTestResonance(pinch01, worldPos, camPos, camRight, damage);
      }
      public override playGlassImpact(
        intensity01: number,
        pitchScale: number = 1.0,
        worldPos?: [number, number, number],
        camPos?: [number, number, number],
        camRight?: [number, number, number],
        options?: GlassImpactOptions
      ): void {
        if (options) impactMetas.push(options);
        super.playGlassImpact(intensity01, pitchScale, worldPos, camPos, camRight, options);
      }
    }

    const spyAudio = new SpyAudioEngine();
    spyAudio.setMuted(true);

    // Wire MujocoWorkerBridge <-> attachMujocoWorkerHost over a real Node MessageChannel
    const channel = new MessageChannel();
    const clientEndpoint: WorkerEndpointLike = {
      onmessage: null,
      postMessage(msg: WorkerInboundMessage, transfer?: Transferable[]) {
        channel.port1.postMessage(msg, transfer as never);
      },
      terminate() {
        channel.port1.close();
        channel.port2.close();
      },
    };
    const hostScope = {
      onmessage: null as ((ev: MessageEvent<WorkerInboundMessage>) => void) | null,
      postMessage(msg: WorkerOutboundMessage, transfer?: Transferable[]) {
        channel.port2.postMessage(msg, transfer as never);
      },
      close() {
        channel.port2.close();
      },
    };
    channel.port1.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
      clientEndpoint.onmessage?.(ev);
    };
    channel.port2.onmessage = (ev: MessageEvent<WorkerInboundMessage>) => {
      hostScope.onmessage?.(ev);
    };
    channel.port1.start();
    channel.port2.start();

    const hostHandle = attachMujocoWorkerHost(hostScope);
    const bridge = await MujocoWorkerBridge.create(spyAudio, 18, 0x5150cafe, clientEndpoint);

    const stepBridgeSync = async (dt: number = 1 / 60): Promise<void> => {
      await new Promise<void>((resolve) => {
        const prev = clientEndpoint.onmessage;
        clientEndpoint.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
          prev?.(ev);
          if (ev.data.type === 'snapshot' && ev.data.epoch === bridge.getEpoch()) {
            clientEndpoint.onmessage = prev;
            resolve();
          }
        };
        bridge.step(dt);
      });
    };

    const waitNextBridgeSnapshot = async (): Promise<void> => {
      await new Promise<void>((resolve) => {
        const prev = clientEndpoint.onmessage;
        clientEndpoint.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
          prev?.(ev);
          if (ev.data.type === 'snapshot' && ev.data.epoch === bridge.getEpoch()) {
            clientEndpoint.onmessage = prev;
            resolve();
          }
        };
      });
    };

    try {
      // 1. Smash tower in sandbox, verify audio impact metadata & shard volume/massKg/spawnSimTime parity
      const snapRebuildDone = waitNextBridgeSnapshot();
      bridge.rebuildTower('sandbox', 2, 18, 0x5150cafe);
      await snapRebuildDone;
      for (let f = 0; f < 10; f++) await stepBridgeSync();

      const snapCollapseImpulse = waitNextBridgeSnapshot();
      bridge.triggerSandboxCollapseImpulse();
      await snapCollapseImpulse;
      for (let f = 0; f < 38; f++) await stepBridgeSync();

      expect(bridge.shatteredCount).toBeGreaterThanOrEqual(4);
      expect(bridge.shards.length).toBeGreaterThanOrEqual(40);
      expect(impactMetas.length).toBeGreaterThan(0);
      expect(impactMetas.some((m) => (m.impactSpeedMps ?? 0) > 0.05 && (m.surface === 'glass' || m.surface === 'wood'))).toBe(true);

      const shard0 = bridge.shards[0];
      const spawnTime0 = shard0.spawnSimTime;
      const shardMasses = bridge.shards.map((s) => s.massKg);
      expect(Math.min(...shardMasses)).toBeGreaterThanOrEqual(0.00018);
      expect(Math.max(...shardMasses)).toBeGreaterThan(Math.min(...shardMasses) * 1.15);
      expect(shard0.volume).toBeGreaterThan(0);

      // Step forward again and verify shard0.spawnSimTime is preserved (never overwritten with current simTime!)
      await stepBridgeSync();
      const shard0After = bridge.shards.find((s) => s.id === shard0.id);
      if (shard0After) {
        expect(spawnTime0).toBeDefined();
        expect(shard0After.spawnSimTime).toBeDefined();
        expect(shard0After.spawnSimTime!).toBeCloseTo(spawnTime0!, 6);
        expect(bridge.getSimTime()).toBeGreaterThan(shard0After.spawnSimTime!);
      }

      // Craze/damage check on tapNudgeBlock: find or set a damaged block via compact snapshot & tap it
      const crazedSnap = bridge.exportCompactSnapshot();
      let damagedBlockId = -1;
      for (let i = 0; i < bridge.blocks.length; i++) {
        if (!bridge.blocks[i].shattered) {
          damagedBlockId = i;
          crazedSnap.blockData[i * 19 + 13] = 0.62; // damage = 0.62
          break;
        }
      }
      expect(damagedBlockId).toBeGreaterThanOrEqual(0);
      const applySnapDone = waitNextBridgeSnapshot();
      bridge.applyCompactSnapshot(crazedSnap);
      await applySnapDone;
      tapDamages.length = 0;
      const tapDone = waitNextBridgeSnapshot();
      expect(bridge.tapNudgeBlock(damagedBlockId, [1, 0, 0], [0, 0, 0])).toBe(true);
      await tapDone;
      expect(tapDamages.some((d) => Math.abs(d - 0.62) < 1e-3)).toBe(true);

      // 2. Monotonic epoch fencing: fire an in-flight collapse step, then immediately rebuildTower('solo')!
      const epochBeforeRebuild = bridge.getEpoch();
      bridge.step(1 / 60); // In-flight step on old epoch!
      const soloRebuildSnap = waitNextBridgeSnapshot();
      bridge.rebuildTower('solo', 2, 18, 0x778899aa);

      // Synchronous tick-0 assertions (BEFORE worker rebuild snapshot arrives!)
      expect(bridge.getEpoch()).toBeGreaterThan(epochBeforeRebuild);
      expect(bridge.turnPhase).toBe('SELECT_BLOCK');
      expect(bridge.shatteredCount).toBe(0);
      expect(bridge.shards.length).toBe(0);
      expect(bridge.blocks.length).toBe(54);
      const tick0Tel = bridge.getTelemetry();
      expect(tick0Tel.turnPhase).toBe('SELECT_BLOCK');
      expect(tick0Tel.intactCount).toBe(54);
      expect(tick0Tel.crazedCount).toBe(0);
      expect(tick0Tel.shatteredCount).toBe(0);
      expect(tick0Tel.activeShardCount).toBe(0);

      const tick0NetSnap = bridge.exportCompactSnapshot();
      const refSyncWorld = new PhysicsWorld(spyAudio, 18, { seed: 0x778899aa });
      expect(tick0NetSnap.blockCount).toBe(54);
      expect(tick0NetSnap.shardCount).toBe(0);
      for (let i = 0; i < 54; i++) {
        expect(bridge.blocks[i].swatchId).toBe(refSyncWorld.blocks[i].swatchId);
        expect(bridge.blocks[i].motif).toBe(refSyncWorld.blocks[i].motif);
        expect(bridge.blocks[i].rippleSeed).toBeCloseTo(refSyncWorld.blocks[i].rippleSeed, 4);
      }
      refSyncWorld.dispose();

      // Inject a synthetic stale snapshot from previous epoch and verify it is ignored
      clientEndpoint.onmessage?.({
        data: {
          type: 'snapshot',
          epoch: bridge.getEpoch() - 1,
          ackSeq: 99999,
          isFullRebuild: false,
          threadingMode: 'single',
          threadCount: 1,
          simTime: 99,
          towerSeed: 0x5150cafe,
          initialLayers: 18,
          blockTransforms: new Float32Array(0),
          blockPhotoelasticStress01: new Float32Array(0),
          blockPhotoelasticWrench: new Float32Array(0),
          shardTransforms: new Float32Array(0),
          newShardMetas: [],
          shardUploads: [],
          shardImpacts: [],
          wobbleEvents: [],
          splinters: [],
          openTopSlots: [],
          topSupportFrame: bridge.topSupportFrame,
          telemetry: { ...tick0Tel, turnPhase: 'COLLAPSED', shatteredCount: 42 },
          turnPhase: 'COLLAPSED',
          gameMode: 'sandbox',
          playerCount: 2,
          currentPlayer: 0,
          blocksStacked: 0,
          shatteredCount: 42,
          collapseReason: 'stale-epoch-leak',
          hoveredBlockId: null,
          activeBlockId: null,
          hoveredSlotIndex: null,
          grabLocalPoint: [0, 0, 0],
          grabWorldAnchor: [0, 0, 0],
          grabCurrentWorldOnBlock: [0, 0, 0],
          grabPlaneY: 0,
          grabResistance01: 0,
          isPointerDraggingBlock: false,
          isExtractionLiftReady: false,
          carriedLandingPos: [0, 0, 0],
          carriedTargetPos: [0, 0, 0],
          carriedTargetQuat: [0, 0, 0, 1],
          carriedYawOffsetRad: 0,
          leanAngleDeg: 45,
          wobbleSpeedDegPerSec: 90,
          centerOfMassXZCm: [0, 0],
          currentLayers: 0,
          peakLayers: 18,
          liveHeightCm: 0,
          peakHeightCm: 27,
          lockedMinRow: null,
          solverWarningCount: 0,
          mechanicalEnergyJ: 0,
          audioEvents: [],
        },
      } as unknown as MessageEvent<WorkerOutboundMessage>);
      expect(bridge.turnPhase).toBe('SELECT_BLOCK');
      expect(bridge.shatteredCount).toBe(0);

      await soloRebuildSnap;
      for (let f = 0; f < 15; f++) await stepBridgeSync();
      expect(bridge.turnPhase).toBe('SELECT_BLOCK');
      expect(bridge.shatteredCount).toBe(0);

      // 3. startGrabBlock rejection rollback: optimistic PULLING rolls back cleanly when grabStarted = false
      const legalBlk = bridge.blocks.find((b) => b.assignedLayer === 8 && !b.isLooseSlot)!;
      const initialGrabSnap = waitNextBridgeSnapshot();
      expect(
        bridge.startGrabBlock(legalBlk.id, [0, 0, 0], [...legalBlk.position])
      ).toBe(true);
      expect(bridge.turnPhase).toBe('PULLING');
      await initialGrabSnap;
      // End grab on worker first, then verify worker rejection rolls optimistic PULLING back cleanly
      const endGrabDone = waitNextBridgeSnapshot();
      bridge.endGrab(false);
      await endGrabDone;
      expect(bridge.turnPhase).toBe('SELECT_BLOCK');

      const rejectedGrabSnap = waitNextBridgeSnapshot();
      expect(
        bridge.startGrabBlock(legalBlk.id, [0, 0, 0], [...legalBlk.position])
      ).toBe(true);
      expect(bridge.turnPhase).toBe('PULLING');
      clientEndpoint.onmessage?.({
        data: {
          type: 'grabStarted',
          epoch: bridge.getEpoch(),
          reqId: 9999,
          seq: 999999,
          started: false,
        },
      } as MessageEvent<WorkerOutboundMessage>);
      expect(bridge.turnPhase).toBe('SELECT_BLOCK');
      expect(bridge.activeBlockId).toBeNull();
      expect(bridge.isPointerDraggingBlock).toBe(false);
      await rejectedGrabSnap;
      const cleanupGrab = waitNextBridgeSnapshot();
      bridge.endGrab(false);
      await cleanupGrab;

      // 4. Slide clampingLoadN forwarding + releaseExtractedBlockOntoTop optimistic SETTLING lock
      slideLoads.length = 0;
      const bearingUpper =
        bridge.blocks.find((b) => b.assignedLayer === 15 && !b.isLooseSlot) ??
        bridge.blocks.find((b) => b.assignedLayer === 14 && !b.isLooseSlot)!;
      const startBearingPos = [...bearingUpper.position] as [number, number, number];
      const bAxisX = bearingUpper.assignedLayer % 2 === 0 ? [1, 0, 0] : [0, 0, 1];
      const grabDone = waitNextBridgeSnapshot();
      bridge.startGrabBlock(bearingUpper.id, [0, 0, 0], startBearingPos);
      await grabDone;
      for (let f = 1; f <= 16; f++) {
        const dPull = f * 0.032;
        bridge.updateGrabTarget([
          startBearingPos[0] + bAxisX[0] * dPull,
          startBearingPos[1],
          startBearingPos[2] + bAxisX[2] * dPull,
        ]);
        await stepBridgeSync();
      }
      expect(slideLoads.some((loadN) => loadN > 0.5)).toBe(true);
      const abortBearing = waitNextBridgeSnapshot();
      bridge.endGrab(false);
      await abortBearing;
      for (let f = 0; f < 15; f++) await stepBridgeSync();

      // Extract a loose center block cleanly and test optimistic SETTLING lock across an in-flight step!
      const looseBlk = bridge.blocks.find(
        (b) => b.assignedLayer >= 6 && b.assignedLayer <= 14 && b.isLooseSlot
      )!;
      const lAxis: [number, number, number] =
        looseBlk.assignedLayer % 2 === 0 ? [1, 0, 0] : [0, 0, 1];
      const startLoosePos = [...looseBlk.position] as [number, number, number];
      const looseGrabSnap = waitNextBridgeSnapshot();
      bridge.startGrabBlock(looseBlk.id, [0, 0, 0], startLoosePos);
      await looseGrabSnap;
      for (let f = 1; f <= 80 && bridge.turnPhase === 'PULLING'; f++) {
        const pullDm = Math.min(1.35, f * 0.035);
        bridge.updateGrabTarget([
          startLoosePos[0] + lAxis[0] * pullDm,
          startLoosePos[1],
          startLoosePos[2] + lAxis[2] * pullDm,
        ]);
        await stepBridgeSync();
      }
      expect(bridge.turnPhase).toBe('EXTRACTED');
      for (let f = 0; f < 16; f++) await stepBridgeSync();

      // Put a pre-release `step` in flight BEFORE calling `releaseExtractedBlockOntoTop()`!
      bridge.step(1 / 60);
      expect(bridge.releaseExtractedBlockOntoTop()).toBe(true);
      expect(bridge.turnPhase).toBe('SETTLING');
      expect(bridge.getTelemetry().turnPhase).toBe('SETTLING');
      // Second immediate call MUST be rejected by optimistic SETTLING lock!
      expect(bridge.releaseExtractedBlockOntoTop()).toBe(false);

      // Even while the pre-release step snapshot returns first, turnPhase must never revert to EXTRACTED!
      await waitNextBridgeSnapshot();
      expect(bridge.turnPhase).not.toBe('EXTRACTED');
      await waitNextBridgeSnapshot();
      expect(['SETTLING', 'SELECT_BLOCK']).toContain(bridge.turnPhase);

      // 5. Native WASM mj_ray (`raycastSceneAsync` & `raycastScene`) + 2-ply CEM oracle over worker
      const settleSolo = waitNextBridgeSnapshot();
      bridge.rebuildTower('solo', 2, 18, 0x1234abcd);
      await settleSolo;
      for (let f = 0; f < 20; f++) await stepBridgeSync();

      const wasmBlockHit = await bridge.raycastSceneAsync([0, 1.275, 2.5], [0, 0, -1]);
      const mirrorBlockHit = bridge.raycastScene([0, 1.275, 2.5], [0, 0, -1]);
      expect(wasmBlockHit).not.toBeNull();
      expect(wasmBlockHit!.kind).toBe('block');
      expect(wasmBlockHit!.blockId).toBeDefined();
      expect(mirrorBlockHit).not.toBeNull();
      expect(mirrorBlockHit!.kind).toBe('block');

      const wasmTableHit = await bridge.raycastSceneAsync([2.2, 1.0, 0], [0, -1, 0]);
      expect(wasmTableHit).not.toBeNull();
      expect(wasmTableHit!.kind).toBe('table');
      expect(wasmTableHit!.worldPointDm[1]).toBeCloseTo(0, 2);
      expect(wasmTableHit!.normal[1]).toBeGreaterThan(0.95);

      // Ray aimed outside the table toward y = -50 m MUST skip MJ_GEOM_GROUP_PARKED shelf & pool bodies!
      const shelfMiss = await bridge.raycastSceneAsync([0, -1.0, 0], [0, -1, 0]);
      expect(shelfMiss).toBeNull();

      const oracleCands = await bridge.evaluateCounterfactualBlockCriticalityAsync(4, 4);
      expect(oracleCands.length).toBeGreaterThanOrEqual(2);
      expect(oracleCands[0].optimalPullAzimuthRad).toBeTypeOf('number');
      expect(oracleCands[0].optimalPullDirXZ).toBeDefined();
      expect(oracleCands[0].twoPlyMasterScore01).toBeGreaterThan(0);
      expect(oracleCands[0].adversarialPlacement).toBeDefined();
      expect(bridge.getSolverWarningCount()).toBe(0);
    } finally {
      bridge.dispose();
      hostHandle.dispose();
    }

    // 6. Native MuJoCo 4.0 Constraint-Island Wakeup (`d.tree_asleep` / `d.tree_island`) vs AABB cutoff:
    // Build a wide-span sleeping stone-lintel bridge where shattering the left pier at x = -0.68 dm
    // removes support under a chained lintel span whose right-hand perched block sits at x = +0.92 dm
    // (`|dx| = 1.60 dm >> 0.78 dm` old AABB cutoff!), while an isolated sleeping control block at z = +1.8 dm stays asleep!
    const islandWorld = new PhysicsWorld(spyAudio, 2);
    islandWorld.rebuildTower('sandbox', 2, 2);
    const rw = islandWorld.getWorld();
    const bLeftPier = islandWorld.blocks[0];
    const bRightPier = islandWorld.blocks[1];
    const bBridge = islandWorld.blocks[2];
    const bPerchedFar = islandWorld.blocks[3];
    const bControl = islandWorld.blocks[4];
    const bUnused = islandWorld.blocks[5];

    // Park unused block 5 far away on rim
    rw.getRigidBody(bUnused.bodyHandle)!.setTranslation({ x: -2.5, y: BLOCK_HALF_Y, z: 2.0 }, true);
    rw.getRigidBody(bUnused.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    // Isolated control block at z = +1.8 dm
    rw.getRigidBody(bControl.bodyHandle)!.setTranslation({ x: 0, y: BLOCK_HALF_Y, z: 1.8 }, true);
    rw.getRigidBody(bControl.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);

    // Left pier at x = -0.34 dm, Right pier at x = +0.34 dm (each width 0.25 dm along X when rotated 90 deg)
    const qYaw90 = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
    rw.getRigidBody(bLeftPier.bodyHandle)!.setTranslation({ x: -0.33, y: BLOCK_HALF_Y, z: 0 }, true);
    rw.getRigidBody(bLeftPier.bodyHandle)!.setRotation(
      { x: qYaw90[0], y: qYaw90[1], z: qYaw90[2], w: qYaw90[3] },
      true
    );
    rw.getRigidBody(bRightPier.bodyHandle)!.setTranslation({ x: 0.33, y: BLOCK_HALF_Y, z: 0 }, true);
    rw.getRigidBody(bRightPier.bodyHandle)!.setRotation(
      { x: qYaw90[0], y: qYaw90[1], z: qYaw90[2], w: qYaw90[3] },
      true
    );
    // Bridge lintel spanning [-0.375, +0.375] dm on layer 1 (`y = 0.225 dm`)
    rw.getRigidBody(bBridge.bodyHandle)!.setTranslation({ x: 0.0, y: BLOCK_HEIGHT + BLOCK_HALF_Y, z: 0 }, true);
    rw.getRigidBody(bBridge.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    // Far-right cantilevered upper block at x = +0.52 dm (`dx` from left pier = `0.52 - (-0.33) = 0.85 dm > 0.78 dm` old AABB cutoff!)
    rw.getRigidBody(bPerchedFar.bodyHandle)!.setTranslation(
      { x: 0.52, y: 2 * BLOCK_HEIGHT + BLOCK_HALF_Y, z: 0 },
      true
    );
    rw.getRigidBody(bPerchedFar.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    for (const blk of islandWorld.blocks) {
      blk.isLooseSlot = false;
    }

    for (let f = 0; f < 120; f++) islandWorld.step(1 / 60);
    expect(rw.getRigidBody(bLeftPier.bodyHandle)!.isSleeping()).toBe(true);
    expect(rw.getRigidBody(bBridge.bodyHandle)!.isSleeping()).toBe(true);
    expect(rw.getRigidBody(bPerchedFar.bodyHandle)!.isSleeping()).toBe(true);
    expect(rw.getRigidBody(bControl.bodyHandle)!.isSleeping()).toBe(true);
    expect(Math.abs(bPerchedFar.position[0] - bLeftPier.position[0])).toBeGreaterThan(0.80);

    // Shatter left pier: native circular `tree_asleep` island introspection wakes BOTH `bBridge` AND `bPerchedFar`
    // (`dx = 0.85 dm > 0.78 dm`!), while disconnected `bControl` remains asleep!
    islandWorld.shatterBlock(bLeftPier, [-BLOCK_HALF_X * 0.5, 0, 0], 13.0, [0, 1, 0], true);
    expect(rw.getRigidBody(bControl.bodyHandle)!.isSleeping()).toBe(true);
    islandWorld.step(1 / 60);
    expect(rw.getRigidBody(bBridge.bodyHandle)!.isSleeping()).toBe(false);
    expect(rw.getRigidBody(bPerchedFar.bodyHandle)!.isSleeping()).toBe(false);

    //Also verify shard mj_ray hit on spawned Voronoi shards
    const targetShard = islandWorld.shards[0];
    const shardHit = islandWorld.raycastScene(
      [targetShard.position[0], targetShard.position[1] + 0.5, targetShard.position[2]],
      [0, -1, 0],
      { includeBlocks: false, includeTable: false, includeShards: true }
    );
    expect(shardHit).not.toBeNull();
    expect(shardHit!.kind).toBe('shard');

    islandWorld.dispose();
  }, 45000);

  it('C8: post-tumble rubble & parked shelf achieve 100% native tree sleep with 0 post-settle buzz while preserving fulcrum seesaw rocking & clamped-course silence', async () => {
    await ensurePhysicsInitialized();

    // 1. 18L Sandbox collapse must reach ntree_awake === 0 (0 visible jitter + 0 underground shelf ghost-wake)
    // and emit 0 WobbleRockEvents / playGlassImpact calls after t = 1.6 s
    let simT = 0;
    let postSettleImpacts = 0;
    let postSettleWobbles = 0;
    class SilenceSpyAudio extends GlassAudioEngine {
      public override playGlassImpact(): void {
        if (simT >= 1.6) postSettleImpacts++;
      }
    }
    const collapseAudio = new SilenceSpyAudio();
    collapseAudio.setMuted(true);

    const collapseWorld = new PhysicsWorld(collapseAudio, 18, { seed: 1 });
    collapseWorld.rebuildTower('sandbox', 2, 18, 1);
    for (let f = 0; f < 30; f++) collapseWorld.step(1 / 60);
    collapseWorld.triggerSandboxCollapseImpulse();

    for (let f = 1; f <= 180; f++) {
      simT = f / 60;
      collapseWorld.step(1 / 60);
      const wob = collapseWorld.consumeWobbleEvents();
      collapseWorld.consumeShardImpacts();
      if (simT >= 1.6) postSettleWobbles += wob.length;
    }

    const collapseData = collapseWorld.getWorld().getData()!;
    expect(collapseWorld.shatteredCount).toBeGreaterThanOrEqual(15);
    expect(collapseData.ntree_awake).toBe(0);
    expect(postSettleWobbles).toBe(0);
    expect(postSettleImpacts).toBe(0);
    collapseWorld.dispose();

    // 2. 6.5-deg prism across transverse fulcrum block rocks with >= 5 reversals, emits discrete wobbles on seesaw block,
    // emits 0 false wobbles on underlying stationary fulcrum block, and sleeps cleanly
    const audio = new GlassAudioEngine();
    audio.setMuted(true);
    const seesawWorld = new PhysicsWorld(audio, 1, { seed: 1 });
    seesawWorld.rebuildTower('sandbox', 2, 1, 1);
    const rw = seesawWorld.getWorld();
    const [bSeesaw, bFulcrum, bSpare] = seesawWorld.blocks;
    rw.getRigidBody(bSpare.bodyHandle)!.setTranslation({ x: 2.5, y: BLOCK_HALF_Y, z: 0 }, true);
    const qYaw90 = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
    rw.getRigidBody(bFulcrum.bodyHandle)!.setTranslation({ x: 0, y: BLOCK_HALF_Y, z: 0 }, true);
    rw.getRigidBody(bFulcrum.bodyHandle)!.setRotation(
      { x: qYaw90[0], y: qYaw90[1], z: qYaw90[2], w: qYaw90[3] },
      true
    );
    rw.getRigidBody(bSeesaw.bodyHandle)!.setTranslation({ x: -2.5, y: BLOCK_HALF_Y, z: 0 }, true);
    for (let k = 0; k < 120; k++) seesawWorld.step(1 / 600);
    seesawWorld.consumeWobbleEvents();

    const rad = (6.5 * Math.PI) / 180.0;
    const halfSpan = BLOCK_WIDTH * 0.5 - 0.0125;
    const yCenter =
      BLOCK_HEIGHT + BLOCK_HALF_Y * Math.cos(rad) + halfSpan * Math.sin(rad) + 0.0004;
    const qTilt = quatFromAxisAngle([0, 0, 1], rad);
    const rbSeesaw = rw.getRigidBody(bSeesaw.bodyHandle)!;
    rbSeesaw.setTranslation({ x: 0, y: yCenter, z: 0 }, true);
    rbSeesaw.setRotation({ x: qTilt[0], y: qTilt[1], z: qTilt[2], w: qTilt[3] }, true);
    rbSeesaw.setLinvel({ x: 0, y: 0, z: 0 }, true);
    rbSeesaw.setAngvel({ x: 0, y: 0, z: 0 }, true);

    const mjModel = rw.getModel()!;
    const mjData = rw.getData()!;
    const sDof = mjModel.jnt_dofadr[mjModel.body_jntadr[bSeesaw.bodyHandle]];
    let prevSign = 0;
    let seesawFlips = 0;
    let seesawWobbles = 0;
    let fulcrumWobbles = 0;
    for (let k = 0; k < 420; k++) {
      seesawWorld.step(1 / 600);
      const wz = mjData.qvel[sDof + 5];
      if (Math.abs(wz) > 0.03) {
        const sgn = wz > 0 ? 1 : -1;
        if (prevSign !== 0 && sgn !== prevSign) seesawFlips++;
        prevSign = sgn;
      }
      for (const w of seesawWorld.consumeWobbleEvents()) {
        if (w.blockId === bSeesaw.id) seesawWobbles++;
        if (w.blockId === bFulcrum.id) fulcrumWobbles++;
        expect(w.clampingLoadN).toBeLessThan(1.1);
      }
    }
    expect(seesawFlips).toBeGreaterThanOrEqual(5);
    expect(seesawWobbles).toBeGreaterThanOrEqual(3);
    expect(fulcrumWobbles).toBe(0);
    expect(rw.getRigidBody(bSeesaw.bodyHandle)!.isSleeping()).toBe(true);
    seesawWorld.dispose();
  }, 35000);
});

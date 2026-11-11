import { describe, expect, it } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { quatFromAxisAngle } from '../src/math/vecMat';
import {
  BLOCK_HALF_Y,
  BLOCK_HEIGHT,
  BLOCK_WIDTH,
  TABLE_RADIUS,
} from '../src/physics/constants';
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
    expect(world.shatteredCount).toBeGreaterThanOrEqual(18);
    expect(floatingOffRimShards).toBe(0);
    expect(floatingOffRimBlocks).toBe(0);
    expect(solverTimeResets).toBe(0);
    expect(p95).toBeLessThan(16.0);
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
});

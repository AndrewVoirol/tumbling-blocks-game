import { describe, expect, it } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { GameEngine } from '../src/engine/GameEngine';
import { quatFromAxisAngle, quatInverseRotateVec3 } from '../src/math/vecMat';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  SIM_TIME_SCALE,
  TABLE_RADIUS,
} from '../src/physics/constants';
import { MJ_CONTYPE_BLOCK } from '../src/physics/mujocoTowerBuilder';
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
  FAITHFUL_GLASS_FIDELITY_CONFIG,
  PhysicsWorld,
  SHIPPED_FIDELITY_CONFIG,
} from '../src/physics/PhysicsWorld';

function syncBlockPoseClean(
  pw: PhysicsWorld,
  idx: number,
  pos: [number, number, number],
  quat?: [number, number, number, number],
  linVelSim: [number, number, number] = [0, 0, 0]
): void {
  const b = pw.blocks[idx];
  const q = quat ?? ([...b.rotation] as [number, number, number, number]);
  const rb = pw.getWorld().getRigidBody(b.bodyHandle)!;
  rb.setTranslation({ x: pos[0], y: pos[1], z: pos[2] }, true);
  rb.setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }, true);
  rb.setLinvel({ x: linVelSim[0], y: linVelSim[1], z: linVelSim[2] }, true);
  rb.setAngvel({ x: 0, y: 0, z: 0 }, true);
  b.position = [...pos];
  b.prePos = [...pos];
  b.rotation = [...q];
  b.preQuat = [...q];
  b.linVel = [...linVelSim];
  b.angVel = [0, 0, 0];
}

describe('Vitrail A/B Physics & Acoustic Fidelity Comparison (M1–M6)', () => {
  it('defaults PhysicsWorld to FAITHFUL_GLASS_FIDELITY_CONFIG (all 6 flags true) and allows toggling to Legacy', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    const pw = new PhysicsWorld(audio, 6);
    expect(pw.getFidelityConfig()).toEqual(FAITHFUL_GLASS_FIDELITY_CONFIG);
    for (const val of Object.values(pw.getFidelityConfig())) {
      expect(val).toBe(true);
    }
    pw.setFidelityConfig(SHIPPED_FIDELITY_CONFIG);
    expect(pw.getFidelityConfig()).toEqual(SHIPPED_FIDELITY_CONFIG);
    for (const val of Object.values(pw.getFidelityConfig())) {
      expect(val).toBe(false);
    }
    pw.dispose();
  });

  it('M1 (elasticUnloadedGlassBounce): eliminates zero-rebound drop stick & obeys passivity + sleep', async () => {
    await ensurePhysicsInitialized();

    const runDrop = (faithful: boolean, gapMm: number) => {
      const audio = new GlassAudioEngine();
      const pw = new PhysicsWorld(audio, 6);
      pw.setFidelityConfig(faithful ? FAITHFUL_GLASS_FIDELITY_CONFIG : SHIPPED_FIDELITY_CONFIG);
      pw.rebuildTower('solo', 2, 6, 42);
      for (let i = 0; i < 36; i++) pw.step(1 / 60);

      const idx = pw.blocks.findIndex((b) => b.assignedLayer === 5 && b.assignedSlot === 1);
      expect(idx).toBeGreaterThanOrEqual(0);
      const blk = pw.blocks[idx];
      const restY = blk.position[1];
      const startY = restY + gapMm / 100.0;
      syncBlockPoseClean(pw, idx, [blk.position[0], startY, blk.position[2]], [...blk.rotation]);
      pw.getWorld().syncKinematics();

      let touched = false;
      let minY = Infinity;
      let maxReboundMm = 0;
      let maxObservedY = -Infinity;
      for (let i = 0; i < 240; i++) {
        pw.step(1 / 120);
        const y = blk.position[1];
        if (y > maxObservedY) maxObservedY = y;
        const clrMm = (y - restY) * 100.0;
        if (!touched && clrMm <= 0.65) {
          touched = true;
          minY = y;
        } else if (touched) {
          if (y < minY) minY = y;
          if (blk.linVel[1] > 0.02 || y > minY + 0.002) {
            maxReboundMm = Math.max(maxReboundMm, Math.max(0, clrMm));
          }
        }
      }
      const ntreeAwake = pw.getWorld().getData().ntree_awake;
      pw.dispose();
      return { maxReboundMm, maxObservedY, startY, ntreeAwake };
    };

    const s12 = runDrop(false, 12);
    const f12 = runDrop(true, 12);
    expect(s12.maxReboundMm).toBeLessThan(0.05);
    expect(f12.maxReboundMm).toBeGreaterThanOrEqual(1.7);
    expect(f12.maxObservedY).toBeLessThanOrEqual(f12.startY + 1e-4);
    expect(f12.ntreeAwake).toBe(0);

    const s434 = runDrop(false, 4.34);
    const f434 = runDrop(true, 4.34);
    expect(s434.maxReboundMm).toBeLessThan(0.05);
    expect(f434.maxReboundMm).toBeGreaterThanOrEqual(0.42);
    expect(f434.maxObservedY).toBeLessThanOrEqual(f434.startY + 1e-4);
    expect(f434.ntreeAwake).toBe(0);
  });

  it('M2 (unassistedExtraction): restores 1.0g overhang pitch & eliminates phantom air clink', async () => {
    await ensurePhysicsInitialized();

    const runPull = (faithful: boolean) => {
      const audio = new GlassAudioEngine();
      const pw = new PhysicsWorld(audio, 18);
      pw.setFidelityConfig(faithful ? FAITHFUL_GLASS_FIDELITY_CONFIG : SHIPPED_FIDELITY_CONFIG);
      pw.rebuildTower('solo', 2, 18, 42);
      for (let i = 0; i < 48; i++) pw.step(1 / 60);

      const midLoose =
        pw.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot && b.isLegalTarget)!;
      const q0 = midLoose.rotation;
      const axRawX = 1 - 2 * (q0[1] * q0[1] + q0[2] * q0[2]);
      const axRawZ = 2 * (q0[0] * q0[2] - q0[1] * q0[3]);
      const axLen = Math.hypot(axRawX, axRawZ) || 1;
      const px = axRawX / axLen;
      const pz = axRawZ / axLen;
      const perpX = -pz;
      const perpZ = px;
      const startWorld: [number, number, number] = [
        midLoose.position[0] + px * 0.18,
        midLoose.position[1],
        midLoose.position[2] + pz * 0.18,
      ];
      pw.startGrabBlock(midLoose.id, [0.18, 0, 0], startWorld);

      let maxPitchDeg = 0;
      for (let i = 0; i < 220; i++) {
        const tSim = (i + 1) / 120;
        let distDm: number;
        if (tSim <= 0.65) distDm = 0.02 + (tSim / 0.65) * 0.56;
        else if (tSim <= 1.15) distDm = 0.58 + ((tSim - 0.65) / 0.5) * 0.06;
        else distDm = 0.64 + Math.min(1.0, (tSim - 1.15) / 0.5) * 0.68;
        const latDm = Math.sin(Math.min(1.0, tSim / 1.65) * Math.PI) * 0.12;
        if (pw.turnPhase === 'PULLING') {
          pw.updateGrabTarget([
            startWorld[0] + px * distDm + perpX * latDm,
            startWorld[1],
            startWorld[2] + pz * distDm + perpZ * latDm,
          ]);
        }
        pw.step(1 / 120);
        const q = midLoose.rotation;
        const upY = Math.max(-1, Math.min(1, 1 - 2 * (q[0] * q[0] + q[2] * q[2])));
        const pitchDeg = (Math.acos(Math.abs(upY)) * 180) / Math.PI;
        if ((pw.turnPhase === 'PULLING' || pw.turnPhase === 'EXTRACTED') && pitchDeg > maxPitchDeg) {
          maxPitchDeg = pitchDeg;
        }
      }
      const diag = pw.getSolverDiagnostics();
      const turnPhase = pw.turnPhase;
      pw.dispose();
      return { turnPhase, maxPitchDeg, phantomAirClinkCount: diag.phantomAirClinkCount };
    };

    const shipped = runPull(false);
    const faithful = runPull(true);
    expect(shipped.turnPhase).toBe('EXTRACTED');
    expect(faithful.turnPhase).toBe('EXTRACTED');
    expect(shipped.phantomAirClinkCount).toBeGreaterThanOrEqual(1);
    expect(shipped.maxPitchDeg).toBeLessThan(3.0);
    expect(faithful.phantomAirClinkCount).toBe(0);
    expect(faithful.maxPitchDeg).toBeGreaterThanOrEqual(6.0);
    expect(faithful.maxPitchDeg).toBeGreaterThanOrEqual(shipped.maxPitchDeg + 3.5);
  }, 25000);

  it('M3 (solidCulletAndTrueShardHulls): removes ghost cullet and keeps shards resting on Layer-0 bricks', async () => {
    await ensurePhysicsInitialized();

    const runShatter = (faithful: boolean) => {
      const audio = new GlassAudioEngine();
      const pw = new PhysicsWorld(audio, 2);
      pw.setFidelityConfig(faithful ? FAITHFUL_GLASS_FIDELITY_CONFIG : SHIPPED_FIDELITY_CONFIG);
      pw.rebuildTower('sandbox', 2, 2, 19);
      for (let i = 0; i < 30; i++) pw.step(1 / 60);

      syncBlockPoseClean(pw, 3, [-2.75, BLOCK_HALF_Y, -0.45], [0, 0, 0, 1]);
      syncBlockPoseClean(pw, 4, [-2.75, BLOCK_HALF_Y, 0.45], [0, 0, 0, 1]);
      syncBlockPoseClean(pw, 5, [0, 2 * BLOCK_HALF_Y + 0.92, 0], [0, 0, 0, 1], [
        0,
        -3.4 / SIM_TIME_SCALE,
        0,
      ]);
      pw.getWorld().syncKinematics();
      pw.shatterBlock(pw.blocks[5], [0.03, -BLOCK_HALF_Y, 0.02], 18.2, [0, 1, 0], false);
      for (let i = 0; i < 180; i++) pw.step(1 / 120);

      const diag = pw.getSolverDiagnostics();
      const supported = pw.shards.filter(
        (s) =>
          s.position[1] >= 2 * BLOCK_HALF_Y + 0.025 &&
          Math.abs(s.position[0]) < 0.55 &&
          Math.abs(s.position[2]) < 0.55
      ).length;
      pw.dispose();
      return { cullStats: diag.lastCullShardStats, supported };
    };

    const shipped = runShatter(false);
    const faithful = runShatter(true);
    expect(shipped.cullStats.cullet / Math.max(1, shipped.cullStats.total)).toBeGreaterThan(0.65);
    expect(shipped.supported).toBeLessThanOrEqual(2);
    expect(faithful.cullStats.cullet).toBe(0);
    expect(faithful.supported).toBeGreaterThanOrEqual(6);
  });

  it('M4 (trueAcousticCoupling): emits coupled oak plate thud & unshifted 2872 Hz glass frequency', async () => {
    await ensurePhysicsInitialized();

    const runOakDrop = (faithful: boolean) => {
      const audio = new GlassAudioEngine();
      const pw = new PhysicsWorld(audio, 1);
      pw.setFidelityConfig(faithful ? FAITHFUL_GLASS_FIDELITY_CONFIG : SHIPPED_FIDELITY_CONFIG);
      pw.rebuildTower('sandbox', 2, 1, 7);
      for (let i = 0; i < 20; i++) pw.step(1 / 60);

      syncBlockPoseClean(pw, 1, [-2.9, BLOCK_HALF_Y, -0.6], [0, 0, 0, 1]);
      syncBlockPoseClean(pw, 2, [2.9, BLOCK_HALF_Y, 0.6], [0, 0, 0, 1]);
      syncBlockPoseClean(pw, 0, [0, BLOCK_HALF_Y + 0.34, 0], [0, 0, 0, 1]);
      pw.getWorld().syncKinematics();
      audio.clearRecentImpactLog();

      for (let i = 0; i < 120; i++) pw.step(1 / 120);
      const log = audio.getRecentImpactLog();
      pw.dispose();
      return log;
    };

    const shippedLog = runOakDrop(false);
    const faithfulLog = runOakDrop(true);

    const shippedWood = shippedLog.filter((e) => e.kind === 'woodThud');
    const shippedGlass = shippedLog.filter((e) => e.kind === 'glass');
    expect(shippedWood.length).toBe(0);
    expect(shippedGlass.length).toBeGreaterThanOrEqual(1);
    expect(shippedGlass[0].fundamentalHz).toBeLessThan(2700);
    expect(shippedGlass[0].pitchScale).toBeCloseTo(0.92, 4);

    const faithfulWood = faithfulLog.filter((e) => e.kind === 'woodThud');
    const faithfulGlass = faithfulLog.filter((e) => e.kind === 'glass');
    expect(faithfulWood.length).toBeGreaterThanOrEqual(1);
    expect(faithfulWood[0].fundamentalHz).toBeGreaterThanOrEqual(95);
    expect(faithfulWood[0].fundamentalHz).toBeLessThanOrEqual(135);
    expect(faithfulGlass.length).toBeGreaterThanOrEqual(1);
    expect(faithfulGlass[0].fundamentalHz).toBeGreaterThan(2780);
    expect(faithfulGlass[0].pitchScale).toBeCloseTo(1.0, 4);
  });

  it('M5 & M6: preserves >= 500 Hz / >= 14 Newton collapse rate & sub-frame worker shard interpolation', async () => {
    await ensurePhysicsInitialized();

    const runCollapse = (faithful: boolean, crossOriginIsolated: boolean) => {
      const g = globalThis as { crossOriginIsolated?: boolean };
      const origDesc = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
      Object.defineProperty(globalThis, 'crossOriginIsolated', {
        value: crossOriginIsolated,
        configurable: true,
        writable: true,
      });
      try {
        const audio = new GlassAudioEngine();
        const pw = new PhysicsWorld(audio, 18);
        pw.setFidelityConfig(faithful ? FAITHFUL_GLASS_FIDELITY_CONFIG : SHIPPED_FIDELITY_CONFIG);
        pw.rebuildTower('sandbox', 2, 18, 42);
        for (let i = 0; i < 15; i++) pw.step(1 / 60);
        pw.triggerSandboxCollapseImpulse();

        let minHz = Infinity;
        let minIter = Infinity;
        for (let i = 0; i < 140; i++) {
          pw.step(1 / 60);
          const diag = pw.getSolverDiagnostics();
          if (diag.timestepHz < minHz) minHz = diag.timestepHz;
          if (diag.solverIterations < minIter) minIter = diag.solverIterations;
        }
        const alpha = pw.getRenderPoseAlpha();
        pw.dispose();
        return { minHz, minIter, alpha };
      } finally {
        if (origDesc) {
          Object.defineProperty(globalThis, 'crossOriginIsolated', origDesc);
        } else {
          delete g.crossOriginIsolated;
        }
      }
    };

    const shipped = runCollapse(false, true);
    const faithfulMt = runCollapse(true, true);
    const faithfulSt = runCollapse(true, false);
    expect(shipped.minHz).toBeLessThanOrEqual(360);
    expect(shipped.minIter).toBeLessThanOrEqual(8);
    expect(faithfulMt.minHz).toBeGreaterThanOrEqual(500);
    expect(faithfulMt.minIter).toBeGreaterThanOrEqual(14);
    expect(faithfulMt.alpha).toBeGreaterThanOrEqual(0);
    expect(faithfulMt.alpha).toBeLessThanOrEqual(1);
    expect(faithfulSt.alpha).toBeGreaterThanOrEqual(0);
    expect(faithfulSt.alpha).toBeLessThanOrEqual(1);

    // Worker snapshot shard prePos preservation & sub-frame poseAlpha interpolation
    const bridgeAudio = new GlassAudioEngine();
    bridgeAudio.setMuted(true);
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
      addEventListener() {},
      removeEventListener() {},
    };
    const hostScope = {
      onmessage: null as ((ev: MessageEvent<WorkerInboundMessage>) => void) | null,
      postMessage(msg: WorkerOutboundMessage, transfer?: Transferable[]) {
        channel.port2.postMessage(msg, transfer as never);
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
    const bridge = await MujocoWorkerBridge.create(bridgeAudio, 6, 42, clientEndpoint);
    bridge.setFidelityConfig(FAITHFUL_GLASS_FIDELITY_CONFIG);

    const stepBridge = async (dt = 1 / 60): Promise<void> => {
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

    bridge.rebuildTower('sandbox', 2, 18, 42);
    for (let i = 0; i < 6; i++) await stepBridge(1 / 60);
    bridge.triggerSandboxCollapseImpulse();
    let shardMovedDist = 0;
    for (let i = 0; i < 110; i++) {
      await stepBridge(1 / 60);
      for (const s of bridge.shards) {
        const d = Math.hypot(
          s.position[0] - s.prePos[0],
          s.position[1] - s.prePos[1],
          s.position[2] - s.prePos[2]
        );
        if (d > shardMovedDist) shardMovedDist = d;
      }
      if (shardMovedDist > 1e-4 && i >= 60) break;
    }
    expect(bridge.shards.length).toBeGreaterThan(0);
    expect(shardMovedDist).toBeGreaterThan(1e-4);

    const nowMs = performance.now();
    const alpha0 = bridge.getRenderPoseAlpha(nowMs - 4);
    const alpha1 = bridge.getRenderPoseAlpha(nowMs + 8);
    expect(alpha0).toBeGreaterThanOrEqual(0);
    expect(alpha0).toBeLessThanOrEqual(1);
    expect(alpha1).toBeGreaterThanOrEqual(alpha0);
    expect(alpha1).toBeLessThanOrEqual(1);

    bridge.dispose();
    hostHandle.dispose();
  }, 25000);

  it('M7: Sandbox post-collapse block drag, flick, tap-nudge, network snapshot & tablecloth snap preserve 100% solid block-vs-shard collisions, bevel tilt & acoustics', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    // 1. 18L Sandbox collapse + settle -> 0% on-table ghost shards + network snapshot round-trip parity
    const pw = new PhysicsWorld(audio, 18, { seed: 42 });
    pw.rebuildTower('sandbox', 2, 18, 42);
    for (let f = 0; f < 30; f++) pw.step(1 / 60);
    pw.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 180; f++) pw.step(1 / 60);

    const rw = pw.getWorld();
    const mjModel = rw.getModel()!;
    const mjData = rw.getData()!;
    expect(mjData.ntree_awake).toBe(0);

    const onTableShards = pw.shards.filter(
      (s) => Math.hypot(s.position[0], s.position[2]) <= TABLE_RADIUS - 0.04
    );
    expect(onTableShards.length).toBeGreaterThan(200);
    const postCollapseGhosts = onTableShards.filter(
      (s) => (mjModel.geom_conaffinity[s.colliderHandle] & MJ_CONTYPE_BLOCK) === 0
    ).length;
    expect(postCollapseGhosts).toBe(0);

    // Network snapshot apply preserves 0x22 solid cullet conaffinity on guest
    const snap = pw.exportCompactSnapshot({ includeShardGeometry: true });
    const guest = new PhysicsWorld(audio, 18, { seed: 42 });
    guest.applyCompactSnapshot(snap);
    const guestModel = guest.getWorld().getModel()!;
    const guestOnTable = guest.shards.filter(
      (s) => Math.hypot(s.position[0], s.position[2]) <= TABLE_RADIUS - 0.04
    );
    expect(guestOnTable.length).toBeGreaterThan(200);
    const guestGhosts = guestOnTable.filter(
      (s) => (guestModel.geom_conaffinity[s.colliderHandle] & MJ_CONTYPE_BLOCK) === 0
    ).length;
    expect(guestGhosts).toBe(0);
    guest.dispose();

    // 2. Broadside top-face drag sweep across settled Voronoi shards: >= 90% plowed, 0 pass-throughs, natural bevel tilt, shard acoustics
    const intact = pw.blocks.filter((b) => !b.shattered);
    const pusher = intact[0];
    for (let i = 1; i < intact.length; i++) {
      const ang = (i / intact.length) * Math.PI * 2;
      rw.getRigidBody(intact[i].bodyHandle)!.setTranslation(
        { x: Math.cos(ang) * 3.6, y: BLOCK_HALF_Y, z: Math.sin(ang) * 3.6 },
        true
      );
      rw.getRigidBody(intact[i].bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    }
    const qBroadside = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
    const startPos: [number, number, number] = [-1.2, BLOCK_HALF_Y, 0.0];
    const endPos: [number, number, number] = [2.0, BLOCK_HALF_Y, 0.0];
    rw.getRigidBody(pusher.bodyHandle)!.setTranslation(
      { x: startPos[0], y: startPos[1], z: startPos[2] },
      true
    );
    rw.getRigidBody(pusher.bodyHandle)!.setRotation(
      { x: qBroadside[0], y: qBroadside[1], z: qBroadside[2], w: qBroadside[3] },
      true
    );
    for (let f = 0; f < 45; f++) pw.step(1 / 60);
    pw.consumeShardImpacts();

    const corridorInit = new Map<number, [number, number, number]>();
    for (const s of pw.shards) {
      if (
        s.position[0] > startPos[0] + 0.25 &&
        s.position[0] < endPos[0] - 0.35 &&
        Math.abs(s.position[2]) < BLOCK_HALF_X - 0.10 &&
        s.position[1] < 0.15
      ) {
        corridorInit.set(s.id, [s.position[0], s.position[1], s.position[2]]);
      }
    }
    expect(corridorInit.size).toBeGreaterThanOrEqual(20);

    pw.startGrabBlock(
      pusher.id,
      [0, BLOCK_HALF_Y, 0],
      [startPos[0], BLOCK_HALF_Y * 2, startPos[2]]
    );
    let interiorPassThroughs = 0;
    let maxTiltDeg = 0;
    let dragShardImpacts = 0;
    for (let f = 1; f <= 95; f++) {
      const u = Math.min(1, f / 75);
      const gx = startPos[0] * (1 - u) + endPos[0] * u;
      pw.updateGrabTarget([gx, BLOCK_HALF_Y * 2, startPos[2]]);
      pw.step(1 / 60);
      dragShardImpacts += pw.consumeShardImpacts().length;
      const r = pusher.rotation;
      const upY = 1 - 2 * (r[0] * r[0] + r[2] * r[2]);
      const tilt = (Math.acos(Math.max(-1, Math.min(1, Math.abs(upY)))) * 180) / Math.PI;
      if (tilt > maxTiltDeg) maxTiltDeg = tilt;
      for (const s of pw.shards) {
        const rel: [number, number, number] = [
          s.position[0] - pusher.position[0],
          s.position[1] - pusher.position[1],
          s.position[2] - pusher.position[2],
        ];
        const loc = quatInverseRotateVec3(pusher.rotation, rel);
        if (
          Math.abs(loc[0]) < BLOCK_HALF_X - 0.04 &&
          Math.abs(loc[1]) < BLOCK_HALF_Y - 0.025 &&
          Math.abs(loc[2]) < BLOCK_HALF_Z - 0.03
        ) {
          interiorPassThroughs++;
        }
      }
    }
    pw.endGrab(false);
    for (let f = 0; f < 150; f++) pw.step(1 / 60);

    let movedCorridor = 0;
    for (const [id, p0] of corridorInit) {
      const cur = pw.shards.find((s) => s.id === id);
      if (!cur || Math.hypot(cur.position[0] - p0[0], cur.position[2] - p0[2]) > 0.05) {
        movedCorridor++;
      }
    }
    expect(interiorPassThroughs).toBe(0);
    expect(movedCorridor / corridorInit.size).toBeGreaterThanOrEqual(0.9);
    expect(maxTiltDeg).toBeGreaterThanOrEqual(1.5);
    expect(maxTiltDeg).toBeLessThanOrEqual(35.0);
    expect(dragShardImpacts).toBeGreaterThanOrEqual(10);
    expect(mjData.ntree_awake).toBe(0);

    // 3. Off-center Sandbox `tapNudgeBlock` imparts yaw + lateral velocity and scatters nearby shards
    rw.getRigidBody(pusher.bodyHandle)!.setTranslation({ x: 2.8, y: BLOCK_HALF_Y, z: 2.2 }, true);
    const tapper = pw.blocks.find((b) => !b.shattered && !b.extracted && b.id !== pusher.id)!;
    rw.getRigidBody(tapper.bodyHandle)!.setTranslation({ x: 0.0, y: BLOCK_HALF_Y, z: 0.0 }, true);
    rw.getRigidBody(tapper.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    for (let f = 0; f < 30; f++) pw.step(1 / 60);
    const yawBefore = Math.atan2(
      2 * (tapper.rotation[3] * tapper.rotation[1] + tapper.rotation[0] * tapper.rotation[2]),
      1 - 2 * (tapper.rotation[1] * tapper.rotation[1] + tapper.rotation[2] * tapper.rotation[2])
    );
    expect(pw.tapNudgeBlock(tapper.id, [0, 0, -1], [0.3, 0, -BLOCK_HALF_Z])).toBe(true);
    for (let f = 0; f < 45; f++) pw.step(1 / 60);
    const yawAfter = Math.atan2(
      2 * (tapper.rotation[3] * tapper.rotation[1] + tapper.rotation[0] * tapper.rotation[2]),
      1 - 2 * (tapper.rotation[1] * tapper.rotation[1] + tapper.rotation[2] * tapper.rotation[2])
    );
    expect(Math.abs(yawAfter - yawBefore)).toBeGreaterThan(0.02);
    expect(tapper.position[2]).toBeGreaterThan(0.02);
    pw.dispose();

    // 4. Perched shards on a 15 mm flat tabletop block: ride under gentle Coulomb static friction, drop 100% on tablecloth Z-snap
    const tcWorld = new PhysicsWorld(audio, 2, { seed: 19 });
    tcWorld.rebuildTower('sandbox', 2, 2, 19);
    for (let f = 0; f < 25; f++) tcWorld.step(1 / 60);
    const tcRw = tcWorld.getWorld();
    const baseBlk = tcWorld.blocks[0];
    for (let i = 1; i < tcWorld.blocks.length - 1; i++) {
      tcRw
        .getRigidBody(tcWorld.blocks[i].bodyHandle)!
        .setTranslation({ x: -2.8, y: BLOCK_HALF_Y, z: (i - 2) * 0.4 }, true);
    }
    tcRw.getRigidBody(baseBlk.bodyHandle)!.setTranslation({ x: 0, y: BLOCK_HALF_Y, z: 0 }, true);
    tcRw.getRigidBody(baseBlk.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    const shatterBlk = tcWorld.blocks[tcWorld.blocks.length - 1];
    tcRw
      .getRigidBody(shatterBlk.bodyHandle)!
      .setTranslation({ x: 0, y: BLOCK_HALF_Y * 3 + 0.05, z: 0 }, true);
    tcRw.syncKinematics();
    tcWorld.shatterBlock(shatterBlk, [0, -BLOCK_HALF_Y, 0], 15.0, [0, 1, 0], false);
    for (let f = 0; f < 150; f++) tcWorld.step(1 / 60);

    const perchedBefore = tcWorld.shards.filter(
      (s) =>
        s.position[1] > 0.155 &&
        Math.abs(s.position[0]) < BLOCK_HALF_X + 0.02 &&
        Math.abs(s.position[2]) < BLOCK_HALF_Z + 0.02
    );
    expect(perchedBefore.length).toBeGreaterThanOrEqual(6);

    // Gentle drag: static friction carries perched shards along on top of `baseBlk`
    tcWorld.startGrabBlock(baseBlk.id, [BLOCK_HALF_X, 0, 0], [BLOCK_HALF_X, BLOCK_HALF_Y, 0]);
    for (let f = 1; f <= 30; f++) {
      tcWorld.updateGrabTarget([BLOCK_HALF_X + Math.min(0.8, f * 0.04), BLOCK_HALF_Y, 0]);
      tcWorld.step(1 / 60);
    }
    tcWorld.endGrab(false);
    for (let f = 0; f < 90; f++) tcWorld.step(1 / 60);
    const carriedPassengers = perchedBefore.filter((p) => {
      const cur = tcWorld.shards.find((s) => s.id === p.id);
      return cur && cur.position[1] > 0.155 && cur.position[0] > 0.45;
    });
    expect(carriedPassengers.length).toBeGreaterThanOrEqual(4);

    // Fast +Z tablecloth yank: 100% of perched shards slip off the 25 mm width and fall to the oak table (0 floaters)
    tcRw.getRigidBody(baseBlk.bodyHandle)!.setLinvel({ x: 0, y: 0, z: 4.5 }, true);
    for (let f = 0; f < 120; f++) tcWorld.step(1 / 60);
    let unsupportedMidAir = 0;
    for (const p of carriedPassengers) {
      const cur = tcWorld.shards.find((s) => s.id === p.id);
      if (!cur) continue;
      const onBase =
        Math.abs(cur.position[0] - baseBlk.position[0]) < BLOCK_HALF_X + 0.08 &&
        Math.abs(cur.position[2] - baseBlk.position[2]) < BLOCK_HALF_Z + 0.08;
      if (!onBase && cur.position[1] > 0.145) unsupportedMidAir++;
    }
    expect(unsupportedMidAir).toBe(0);
    expect(tcWorld.getWorld().getData()!.ntree_awake).toBe(0);
    tcWorld.dispose();
  }, 40000);

  it('M8: Sandbox settled-shard tap-flick, post-collapse block slide/billiard parity, GameEngine/Worker whip-flick & airborne splinter anvil immunity', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    // 1. Default `GameEngine` `nudgeShard(..., 0.035)` on settled post-collapse `0x22` shards: 0% substep-1 velocity wipe
    const pw = new PhysicsWorld(audio, 18, { seed: 42 });
    pw.rebuildTower('sandbox', 2, 18, 42);
    for (let f = 0; f < 30; f++) pw.step(1 / 60);
    pw.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 180; f++) pw.step(1 / 60);

    let defaultKilled = 0;
    let totalNudged = 0;
    for (const shd of pw.shards.slice(0, 30)) {
      if (Math.hypot(shd.position[0], shd.position[2]) > 3.0) continue;
      for (let f = 0; f < 25; f++) pw.step(1 / 60);
      totalNudged++;
      const dof = pw.getWorld().getMapping(shd.bodyHandle)!.dofAdr;
      pw.nudgeShard(shd.id, shd.position, [1, -0.3, 0], 0.035);
      const vBefore = Math.hypot(
        pw.getWorld().getData()!.qvel[dof],
        pw.getWorld().getData()!.qvel[dof + 1],
        pw.getWorld().getData()!.qvel[dof + 2]
      );
      pw.step(1 / 60);
      const vAfter = Math.hypot(
        pw.getWorld().getData()!.qvel[dof],
        pw.getWorld().getData()!.qvel[dof + 1],
        pw.getWorld().getData()!.qvel[dof + 2]
      );
      if (vBefore > 0.52 && vAfter < 0.01) defaultKilled++;
    }
    expect(totalNudged).toBeGreaterThanOrEqual(20);
    expect(defaultKilled).toBe(0);

    // 2. Mocap block-into-block-into-shards plow: dragging Block A pushes sleeping Block B through settled shards
    const rw = pw.getWorld();
    const flatBlocks = pw.blocks.filter(
      (b) => !b.shattered && b.position[1] < 0.14 && Math.hypot(b.position[0], b.position[2]) < 2.2
    );
    const blkA = flatBlocks[0]!;
    const blkB = flatBlocks[1]!;
    for (let i = 2; i < flatBlocks.length; i++) {
      const ang = (i / flatBlocks.length) * Math.PI * 2;
      rw.getRigidBody(flatBlocks[i]!.bodyHandle)!.setTranslation(
        { x: Math.cos(ang) * 3.6, y: BLOCK_HALF_Y, z: Math.sin(ang) * 3.6 },
        true
      );
    }
    const targetShard = pw.shards.find(
      (s) => s.position[1] < 0.15 && Math.hypot(s.position[0], s.position[2]) < 1.5
    )!;
    const sx = targetShard.position[0];
    const sz = targetShard.position[2];
    rw.getRigidBody(blkB.bodyHandle)!.setTranslation({ x: sx - 0.55, y: BLOCK_HALF_Y, z: sz }, true);
    rw.getRigidBody(blkB.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    rw.getRigidBody(blkA.bodyHandle)!.setTranslation({ x: sx - 1.55, y: BLOCK_HALF_Y, z: sz }, true);
    rw.getRigidBody(blkA.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    for (let f = 0; f < 80; f++) pw.step(1 / 60);
    const blkBStart: [number, number, number] = [...blkB.position];
    const shardStart = new Map(pw.shards.map((s) => [s.id, [...s.position] as [number, number, number]]));
    pw.startGrabBlock(blkA.id, [0, BLOCK_HALF_Y, 0], [sx - 1.55, BLOCK_HALF_Y * 2, sz]);
    for (let f = 1; f <= 60; f++) {
      pw.updateGrabTarget([sx - 1.55 + (f / 60) * 2.15, BLOCK_HALF_Y * 2, sz]);
      pw.step(1 / 60);
    }
    pw.endGrab(false);
    for (let f = 0; f < 90; f++) pw.step(1 / 60);
    const blkBMovedMm = Math.hypot(blkB.position[0] - blkBStart[0], blkB.position[2] - blkBStart[2]) * 100;
    let secondaryPlowedShards = 0;
    for (const s of pw.shards) {
      const p0 = shardStart.get(s.id);
      if (p0 && Math.hypot(s.position[0] - p0[0], s.position[2] - p0[2]) * 100 > 5.0) {
        secondaryPlowedShards++;
      }
    }
    expect(blkBMovedMm).toBeGreaterThan(80);
    expect(secondaryPlowedShards).toBeGreaterThanOrEqual(20);
    expect(rw.getData()!.ntree_awake).toBe(0);

    // Post-collapse free-sliding secondary block preserves >= 65% of clean-table slide & yaw (no ridge-buzz over-braking)
    const measureSlideYaw = (collapsed: boolean) => {
      const w = new PhysicsWorld(audio, 18, { seed: 42 });
      w.rebuildTower('sandbox', 2, 18, 42);
      for (let f = 0; f < 25; f++) w.step(1 / 60);
      if (collapsed) {
        w.triggerSandboxCollapseImpulse();
        for (let f = 0; f < 180; f++) w.step(1 / 60);
      }
      const wrw = w.getWorld();
      const survivors = w.blocks.filter((b) => !b.shattered && !b.extracted);
      for (const b of survivors) {
        if (Math.abs(b.position[2] - 2.6) < 0.9) {
          wrw.getRigidBody(b.bodyHandle)!.setTranslation({ x: -2.5, y: BLOCK_HALF_Y, z: -2.0 }, true);
        }
      }
      for (const s of w.shards) {
        if (Math.abs(s.position[2] - 2.6) < 0.7) {
          wrw.getRigidBody(s.bodyHandle)!.setTranslation({ x: -2.5, y: 0.05, z: -2.5 }, true);
        }
      }
      const slider = survivors[1]!;
      wrw.getRigidBody(slider.bodyHandle)!.setTranslation({ x: -0.5, y: BLOCK_HALF_Y, z: 2.6 }, true);
      wrw.getRigidBody(slider.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      for (let f = 0; f < 70; f++) w.step(1 / 60);
      wrw.getRigidBody(slider.bodyHandle)!.setLinvel({ x: 0.75, y: 0, z: 0 }, true);
      wrw.getRigidBody(slider.bodyHandle)!.setAngvel({ x: 0, y: 0.25, z: 0 }, true);
      const x0 = slider.position[0];
      const yaw0 = Math.atan2(
        2 * (slider.rotation[3] * slider.rotation[1] + slider.rotation[0] * slider.rotation[2]),
        1 - 2 * (slider.rotation[1] * slider.rotation[1] + slider.rotation[2] * slider.rotation[2])
      );
      for (let f = 0; f < 30; f++) w.step(1 / 60);
      const slideMm = (slider.position[0] - x0) * 100;
      const yaw1 = Math.atan2(
        2 * (slider.rotation[3] * slider.rotation[1] + slider.rotation[0] * slider.rotation[2]),
        1 - 2 * (slider.rotation[1] * slider.rotation[1] + slider.rotation[2] * slider.rotation[2])
      );
      w.dispose();
      return { slideMm, yawDeg: Math.abs(yaw1 - yaw0) * (180 / Math.PI) };
    };
    const cleanSlide = measureSlideYaw(false);
    const postCollapseSlide = measureSlideYaw(true);
    expect(postCollapseSlide.slideMm).toBeGreaterThanOrEqual(cleanSlide.slideMm * 0.65);
    expect(postCollapseSlide.yawDeg).toBeGreaterThanOrEqual(cleanSlide.yawDeg * 0.65);
    pw.dispose();

    // 3. `GameEngine` rapid 1-frame pointer whip-flick launches free Sandbox block (> 100 mm) while paused release adds < 3 mm drift
    const canvasMock = {
      width: 1280,
      height: 800,
      style: { cursor: 'default' },
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 1280, height: 800 }),
      addEventListener: () => {},
      removeEventListener: () => {},
      setPointerCapture: () => {},
      releasePointerCapture: () => {},
    } as unknown as HTMLCanvasElement;
    const eng = new GameEngine(canvasMock);
    await eng.initHeadless();
    const engPhys = (eng as unknown as { physics: PhysicsWorld }).physics;
    engPhys.rebuildTower('sandbox', 2, 18, 42);
    for (let f = 0; f < 25; f++) engPhys.step(1 / 60);
    engPhys.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 180; f++) engPhys.step(1 / 60);
    const engRw = engPhys.getWorld();
    const flickBlk = engPhys.blocks.find((b) => !b.shattered && !b.extracted)!;
    engRw.getRigidBody(flickBlk.bodyHandle)!.setTranslation({ x: -1.5, y: BLOCK_HALF_Y, z: 2.5 }, true);
    engRw.getRigidBody(flickBlk.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    for (const s of engPhys.shards) {
      if (Math.abs(s.position[2] - 2.5) < 0.5) {
        engRw.getRigidBody(s.bodyHandle)!.setTranslation({ x: -2.0, y: 0.05, z: -2.0 }, true);
      }
    }
    for (const b of engPhys.blocks) {
      if (b.id !== flickBlk.id && !b.shattered && Math.abs(b.position[2] - 2.5) < 0.8) {
        engRw.getRigidBody(b.bodyHandle)!.setTranslation({ x: -2.0, y: BLOCK_HALF_Y, z: -2.0 }, true);
      }
    }
    for (let f = 0; f < 50; f++) engPhys.step(1 / 60);
    const cam = (eng as unknown as {
      camera: { update: (dt: number, aspect: number) => void; viewProjMatrix: Float32Array };
    }).camera;
    cam.update(0.016, 1280 / 800);
    const vp = cam.viewProjMatrix;
    const project = (p: [number, number, number]): [number, number] => {
      const cx = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
      const cy = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
      const cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
      return [((cx / cw) * 0.5 + 0.5) * 1280, (0.5 - (cy / cw) * 0.5) * 800];
    };
    const [sx0, sy0] = project(flickBlk.position);
    const [sx1, sy1] = project([flickBlk.position[0] + 2.4, flickBlk.position[1], flickBlk.position[2]]);
    eng.handlePointerDown(sx0, sy0, 1, 0, false, undefined, 'mouse', 1016);
    eng.handlePointerMove(sx0 + 8, sy0, 1, undefined, 'mouse', 1020);
    expect(canvasMock.style.cursor).toBe('grabbing');
    engPhys.step(1 / 60);
    const posBeforeWhip: [number, number, number] = [...flickBlk.position];
    eng.handlePointerMove(sx0 + (sx1 - sx0) * 0.5, sy0 + (sy1 - sy0) * 0.5, 1, undefined, 'mouse', 1054);
    eng.handlePointerMove(sx1, sy1, 1, undefined, 'mouse', 1062);
    eng.handlePointerUp(sx1, sy1, 1, undefined, 'mouse', 1066);
    for (let f = 0; f < 45; f++) engPhys.step(1 / 60);
    const whipTravelMm =
      Math.hypot(flickBlk.position[0] - posBeforeWhip[0], flickBlk.position[2] - posBeforeWhip[2]) * 100;
    expect(whipTravelMm).toBeGreaterThan(100);

    engRw.getRigidBody(flickBlk.bodyHandle)!.setTranslation({ x: -1.5, y: BLOCK_HALF_Y, z: 2.5 }, true);
    engRw.getRigidBody(flickBlk.bodyHandle)!.setLinvel({ x: 0, y: 0, z: 0 }, true);
    for (let f = 0; f < 40; f++) engPhys.step(1 / 60);
    const [px0, py0] = project(flickBlk.position);
    const [px1, py1] = project([flickBlk.position[0] + 0.6, flickBlk.position[1], flickBlk.position[2]]);
    eng.handlePointerDown(px0, py0, 1, 0, false, undefined, 'mouse', 3000);
    eng.handlePointerMove(px1, py1, 1, undefined, 'mouse', 3016);
    for (let f = 0; f < 12; f++) engPhys.step(1 / 60);
    const posAtPauseRelease: [number, number, number] = [...flickBlk.position];
    eng.handlePointerUp(px1, py1, 1, undefined, 'mouse', 3216);
    for (let f = 0; f < 30; f++) engPhys.step(1 / 60);
    const postPauseDriftMm =
      Math.hypot(flickBlk.position[0] - posAtPauseRelease[0], flickBlk.position[2] - posAtPauseRelease[2]) * 100;
    expect(postPauseDriftMm).toBeLessThan(3.0);
    eng.dispose();

    // 4. `MujocoWorkerBridge` queued `nudgeShard` + rapid whip-flick parity
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
      addEventListener() {},
      removeEventListener() {},
    };
    const hostScope = {
      onmessage: null as ((ev: MessageEvent<WorkerInboundMessage>) => void) | null,
      postMessage(msg: WorkerOutboundMessage, transfer?: Transferable[]) {
        channel.port2.postMessage(msg, transfer as never);
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
    const bridge = await MujocoWorkerBridge.create(audio, 18, 42, clientEndpoint);
    bridge.setFidelityConfig(FAITHFUL_GLASS_FIDELITY_CONFIG);
    const stepBridge = async (dt = 1 / 60): Promise<void> => {
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
    bridge.rebuildTower('sandbox', 2, 18, 42);
    for (let f = 0; f < 25; f++) await stepBridge();
    bridge.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 180; f++) await stepBridge();
    const workerTargetShards = bridge.shards
      .filter(
        (s) =>
          Math.hypot(s.position[0], s.position[2]) < 2.8 &&
          bridge.blocks.every(
            (b) =>
              b.shattered ||
              Math.hypot(b.position[0] - s.position[0], b.position[2] - s.position[2]) > 0.65
          )
      )
      .slice(0, 12);
    expect(workerTargetShards.length).toBe(12);
    // Verify synchronous main-thread `MujocoWorkerBridge.raycastScene` hits 12/12 Voronoi shards (`Math.abs(signedD)`)
    for (const s of workerTargetShards) {
      const syncRayHit = bridge.raycastScene(
        [s.position[0], s.position[1] + 2.0, s.position[2]],
        [0, -1, 0],
        { includeBlocks: false, includeShards: true, includeTable: false }
      );
      expect(syncRayHit?.kind).toBe('shard');
      expect(syncRayHit?.shardId).toBeDefined();
    }
    const workerBefore = new Map(
      workerTargetShards.map((s) => [s.id, [...s.position] as [number, number, number]])
    );
    for (const s of workerTargetShards) {
      bridge.nudgeShard(s.id, s.position, [1, -0.3, 0], 0.035);
    }
    for (let f = 0; f < 15; f++) await stepBridge();
    let workerMovedCount = 0;
    for (const [id, p0] of workerBefore) {
      const cur = bridge.shards.find((x) => x.id === id);
      if (cur && Math.hypot(cur.position[0] - p0[0], cur.position[2] - p0[2]) > 0.05) workerMovedCount++;
    }
    expect(workerMovedCount).toBe(workerTargetShards.length);

    const workerBlk = [...bridge.blocks]
      .filter((b) => !b.shattered && !b.extracted && Math.hypot(b.position[0], b.position[2]) < 2.4)
      .sort((a, b) => b.position[0] - a.position[0])[0]!;
    const wb0: [number, number, number] = [...workerBlk.position];
    bridge.startGrabBlock(workerBlk.id, [0, BLOCK_HALF_Y, 0], [wb0[0], wb0[1] + BLOCK_HALF_Y, wb0[2]], 5000);
    bridge.updateGrabTarget([wb0[0] + 1.8, wb0[1] + BLOCK_HALF_Y, wb0[2]], 5018);
    bridge.endGrab(false, 5022);
    for (let f = 0; f < 30; f++) await stepBridge();
    const workerFlickMm = Math.hypot(workerBlk.position[0] - wb0[0], workerBlk.position[2] - wb0[2]) * 100;
    expect(workerFlickMm).toBeGreaterThan(25);
    bridge.dispose();
    hostHandle.dispose();
    clientEndpoint.terminate?.();

    // 5. Airborne 0.2 g micro-splinter incurs 0 false anvil crazing, while grounded shard anvil crazes/shatters even past 21 shattered blocks
    const airWorld = new PhysicsWorld(audio, 6, { seed: 42 });
    airWorld.rebuildTower('sandbox', 2, 6, 42);
    for (let f = 0; f < 20; f++) airWorld.step(1 / 60);
    const airRw = airWorld.getWorld();
    for (let i = 2; i < airWorld.blocks.length; i++) {
      airRw.getRigidBody(airWorld.blocks[i].bodyHandle)!.setTranslation({ x: -5, y: -5, z: -5 }, false);
    }
    airRw.getRigidBody(airWorld.blocks[1].bodyHandle)!.setTranslation({ x: 2.0, y: 0.2, z: 2.0 }, true);
    airRw.syncKinematics();
    airWorld.shatterBlock(airWorld.blocks[1], [0, 0, 0], 11.0, [0, 1, 0], false);
    for (let f = 0; f < 60; f++) airWorld.step(1 / 60);
    const tinySplinter = [...airWorld.shards].sort((a, b) => a.massKg - b.massKg)[0]!;
    airRw.getRigidBody(tinySplinter.bodyHandle)!.setTranslation({ x: 0, y: 0.24, z: 0 }, true);
    airRw.getRigidBody(tinySplinter.bodyHandle)!.setLinvel({ x: 0, y: 0.2, z: 0 }, true);
    const airBlk = airWorld.blocks[0]!;
    airRw.getRigidBody(airBlk.bodyHandle)!.setTranslation({ x: 0, y: 0.35, z: 0 }, true);
    airRw.getRigidBody(airBlk.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    airRw.getRigidBody(airBlk.bodyHandle)!.setLinvel({ x: 0, y: -3.0, z: 0 }, true);
    airWorld.step(1 / 60);
    airWorld.step(1 / 60);
    expect(airBlk.damage).toBe(0);
    expect(airBlk.shattered).toBe(false);
    airWorld.dispose();

    const anvilWorld = new PhysicsWorld(audio, 18, { seed: 42 });
    anvilWorld.rebuildTower('sandbox', 2, 18, 42);
    for (let f = 0; f < 20; f++) anvilWorld.step(1 / 60);
    anvilWorld.triggerSandboxCollapseImpulse();
    for (let f = 0; f < 160; f++) anvilWorld.step(1 / 60);
    const anvilRw = anvilWorld.getWorld();
    const anvilSurvivors = anvilWorld.blocks.filter((b) => !b.shattered && !b.extracted);
    for (let k = 1; k <= 6 && k < anvilSurvivors.length; k++) {
      anvilWorld.shatterBlock(anvilSurvivors[k]!, [0, -BLOCK_HALF_Y, 0], 12.0, [0, 1, 0], true);
    }
    const dropBlk = anvilSurvivors[0]!;
    for (const b of anvilWorld.blocks) {
      if (!b.shattered && !b.extracted && b.id !== dropBlk.id && Math.hypot(b.position[0], b.position[2]) < 1.2) {
        anvilRw.getRigidBody(b.bodyHandle)!.setTranslation({ x: -2.8, y: BLOCK_HALF_Y, z: -2.2 }, true);
      }
    }
    for (const s of anvilWorld.shards) {
      if (Math.hypot(s.position[0], s.position[2]) < 0.8) {
        anvilRw.getRigidBody(s.bodyHandle)!.setTranslation({ x: 2.2, y: 0.05, z: 2.2 }, true);
      }
    }
    const groundedShard = [...anvilWorld.shards]
      .filter((s) => Math.hypot(s.position[0], s.position[2]) < 3.5)
      .sort((a, b) => b.massKg - a.massKg)[0]!;
    anvilRw.getRigidBody(groundedShard.bodyHandle)!.setTranslation({ x: 0, y: 0.05, z: 0 }, true);
    anvilRw.getRigidBody(groundedShard.bodyHandle)!.setLinvel({ x: 0, y: 0, z: 0 }, true);
    anvilRw.getRigidBody(dropBlk.bodyHandle)!.setTranslation({ x: -2.5, y: BLOCK_HALF_Y, z: 0 }, true);
    for (let f = 0; f < 160; f++) anvilWorld.step(1 / 60);
    expect(anvilWorld.blocks.filter((b) => b.shattered).length).toBeGreaterThan(21);
    expect(anvilWorld.shards.length).toBeGreaterThan(385);
    anvilRw.getRigidBody(dropBlk.bodyHandle)!.setTranslation({ x: 0, y: 0.24, z: 0 }, true);
    anvilRw.getRigidBody(dropBlk.bodyHandle)!.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    anvilRw.getRigidBody(dropBlk.bodyHandle)!.setLinvel({ x: 0, y: -3.2, z: 0 }, true);
    for (let f = 0; f < 6; f++) anvilWorld.step(1 / 60);
    expect(dropBlk.damage > 0.35 || dropBlk.shattered).toBe(true);
    anvilWorld.dispose();
  }, 55000);
});

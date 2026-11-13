import { describe, expect, it } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { BLOCK_HALF_Y, SIM_TIME_SCALE } from '../src/physics/constants';
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
});

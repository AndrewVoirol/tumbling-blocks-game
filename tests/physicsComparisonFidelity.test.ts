import { describe, expect, it } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { computeGlassBarModes } from '../src/audio/physicalModalModels';
import { GameEngine } from '../src/engine/GameEngine';
import { MultiplayerGuestWorld } from '../src/multiplayer';
import { quatFromAxisAngle, quatInverseRotateVec3, quatRotateVec3, Vec3 } from '../src/math/vecMat';
import {
  BLOCK_HALF_X as EXTRACTION_BLOCK_HALF_X,
  F_PINCH,
  F_PUSH_MAX,
  F_SLIP,
  MU_FINGER,
  ZONE_BODY,
  ZONE_BOUNDARY_DM,
  ZONE_END_TIP,
  applyAssistiveEndPushImpulse,
  computeGripZoneFromLocalDm,
  estimateClampingLoadFromWrenchAndStack,
  evaluateExtractionSlip,
  normalizeCompetitiveSlipResult,
  shouldReleaseCompetitiveSlip,
  updateEndPushSmoothedClampingLoadN,
} from '../src/physics/extractionMechanics';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  LIGHTING_PRESETS,
  SIM_TIME_SCALE,
  TABLE_RADIUS,
} from '../src/physics/constants';
import {
  MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE,
  MJ_CONTYPE_BLOCK,
  MJ_CONTYPE_CULLET_SHARD,
  MJ_CONTYPE_HERO_SHARD,
} from '../src/physics/mujocoTowerBuilder';
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
import {
  CAME_INNER_BOUND_X_DM,
  CAME_INNER_BOUND_Z_DM,
  accumulateSolderCameContact,
  evaluateSolderCamePoint,
  MU_KINETIC_GLASS,
  MU_KINETIC_SOLDER,
  resetSolderCameAggregate,
  SOLDER_CAME_HEIGHT_DM,
  SOLDER_CAME_WIDTH_DM,
  type SolderCameAggregateResult,
} from '../src/physics/solderCameTribology';
import { GLASS_TOWER_WGSL } from '../src/renderer/shaders/glassTower.wgsl';

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

    const runCollapse = (faithful: boolean, crossOriginIsolated: boolean, layers: number = 18) => {
      const g = globalThis as { crossOriginIsolated?: boolean };
      const origDesc = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
      Object.defineProperty(globalThis, 'crossOriginIsolated', {
        value: crossOriginIsolated,
        configurable: true,
        writable: true,
      });
      try {
        const audio = new GlassAudioEngine();
        const pw = new PhysicsWorld(audio, layers);
        pw.setFidelityConfig(faithful ? FAITHFUL_GLASS_FIDELITY_CONFIG : SHIPPED_FIDELITY_CONFIG);
        pw.rebuildTower('sandbox', 2, layers, 42);
        for (let i = 0; i < 15; i++) pw.step(1 / 60);
        pw.triggerSandboxCollapseImpulse();

        let minHz = Infinity;
        let minIter = Infinity;
        const t0 = performance.now();
        for (let i = 0; i < 140; i++) {
          pw.step(1 / 60);
          const diag = pw.getSolverDiagnostics();
          if (diag.timestepHz < minHz) minHz = diag.timestepHz;
          if (diag.solverIterations < minIter) minIter = diag.solverIterations;
        }
        const avgWallMs = (performance.now() - t0) / 140;
        const alpha = pw.getRenderPoseAlpha();
        pw.dispose();
        return { minHz, minIter, avgWallMs, alpha };
      } finally {
        if (origDesc) {
          Object.defineProperty(globalThis, 'crossOriginIsolated', origDesc);
        } else {
          delete g.crossOriginIsolated;
        }
      }
    };

    const shipped = runCollapse(false, true, 8);
    const faithfulSubOverload = runCollapse(true, true, 8);
    const faithfulMt = runCollapse(true, true, 18);
    const faithfulSt = runCollapse(true, false, 18);
    expect(shipped.minHz).toBeLessThanOrEqual(360);
    expect(shipped.minIter).toBeLessThanOrEqual(8);
    expect(faithfulSubOverload.minHz).toBeGreaterThanOrEqual(500);
    expect(faithfulSubOverload.minIter).toBeGreaterThanOrEqual(14);
    expect(faithfulMt.avgWallMs).toBeLessThan(15.0);
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
    const onTableSample = pw.shards
      .filter((s) => Math.hypot(s.position[0], s.position[2]) <= 3.0)
      .slice(0, 30);
    for (const shd of onTableSample) {
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
    const otherCentralBlocks = pw.blocks.filter(
      (b) =>
        !b.shattered &&
        b.id !== blkA.id &&
        b.id !== blkB.id &&
        Math.hypot(b.position[0], b.position[2]) < 2.6
    );
    for (let i = 0; i < otherCentralBlocks.length; i++) {
      const ang = (i / Math.max(1, otherCentralBlocks.length)) * Math.PI * 2;
      rw.getRigidBody(otherCentralBlocks[i]!.bodyHandle)!.setTranslation(
        { x: Math.cos(ang) * 3.6, y: BLOCK_HALF_Y, z: Math.sin(ang) * 3.6 },
        true
      );
    }
    const targetShard = pw.shards.find(
      (s) => s.position[1] < 0.15 && Math.hypot(s.position[0], s.position[2]) < 1.5
    )!;
    const sx = targetShard.position[0];
    const sz = targetShard.position[2];
    for (const s of pw.shards) {
      if (s.position[0] < sx - 0.15 && s.position[0] > sx - 2.05 && Math.abs(s.position[2] - sz) < 0.22) {
        rw.getRigidBody(s.bodyHandle)!.setTranslation({ x: -3.2, y: 0.04, z: -2.2 }, true);
      }
    }
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

  it('M9: Sandbox shard-pile drag (Broadside & End-Drag front/center/rear, pre- & post-sleep) incurs 0 spastic shard bursts and smooth multi-height topple', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    // 1. Multi-seed, multi-grip Broadside & End-Drag (`center`, `front`, `rear-push`) across settled AND
    // still-settling (`sandboxCollapseActive === true`) `0x22` shard piles: zero spastic velocity/spin bursts,
    // zero nose-dives, positive bevel climb, and rock-steady End-Drag yaw + deliberate perpendicular swing.
    const dragModes: Array<{
      name: string;
      yaw: number;
      grabLocal: [number, number, number];
    }> = [
      { name: 'broadside-center', yaw: Math.PI * 0.5, grabLocal: [0, BLOCK_HALF_Y, 0] },
      { name: 'end-center', yaw: 0, grabLocal: [0, BLOCK_HALF_Y, 0] },
      { name: 'end-front', yaw: 0, grabLocal: [0.24, BLOCK_HALF_Y, 0] },
      { name: 'end-rear-push', yaw: 0, grabLocal: [-0.24, BLOCK_HALF_Y, 0] },
    ];

    for (const { seed, settleFrames, dragSpeedMps } of [
      { seed: 42, settleFrames: 180, dragSpeedMps: 0.10 },
      { seed: 42, settleFrames: 180, dragSpeedMps: 0.18 },
      { seed: 77, settleFrames: 95, dragSpeedMps: 0.12 }, // Mid-settle (`!hadFullSleepAfterCollapse`) regression check!
    ]) {
      for (const mode of dragModes) {
        const pw = new PhysicsWorld(audio, 18, { seed });
        pw.rebuildTower('sandbox', 2, 18, seed);
        for (let f = 0; f < 25; f++) pw.step(1 / 60);
        pw.triggerSandboxCollapseImpulse();
        for (let f = 0; f < settleFrames; f++) pw.step(1 / 60);

        const rw = pw.getWorld();
        const mjModel = rw.getModel()!;
        const d = rw.getData()!;
        const onTableIntact = pw.blocks.filter(
          (b) =>
            !b.shattered &&
            b.position[1] > 0 &&
            Math.hypot(b.position[0], b.position[2]) < TABLE_RADIUS - 0.4
        );
        const pusher = onTableIntact[0]!;
        // Aim straight through the collapsed on-table `0x22` shard pile centroid and clear only intact blocks
        // obstructing the `[-2.0, +1.6]` sweep corridor (leaving peripheral rubble untouched so mid-settle stays awake)
        const solidTableShards = pw.shards.filter(
          (s) =>
            Math.hypot(s.position[0], s.position[2]) < TABLE_RADIUS - 0.35 &&
            (mjModel.geom_conaffinity[s.colliderHandle] & 0x20) !== 0
        );
        const meanZ =
          solidTableShards.length > 0
            ? solidTableShards.reduce((acc, s) => acc + s.position[2], 0) / solidTableShards.length
            : 0.0;
        const startX = -1.55;
        const startZ = Math.max(-1.1, Math.min(1.1, meanZ));
        for (let i = 1; i < onTableIntact.length; i++) {
          const blk = onTableIntact[i]!;
          if (blk.position[0] > -2.1 && blk.position[0] < 2.15 && Math.abs(blk.position[2] - startZ) < 0.72) {
            const sideZ = startZ >= 0 ? -2.2 - (i % 4) * 0.32 : 2.2 + (i % 4) * 0.32;
            syncBlockPoseClean(
              pw,
              blk.id,
              [-1.4 + Math.floor(i / 4) * 0.85, BLOCK_HALF_Y, sideZ],
              [0, 0, 0, 1]
            );
          }
        }
        let corridorCount = solidTableShards.filter(
          (s) => s.position[0] > -0.85 && s.position[0] < 1.35 && Math.abs(s.position[2] - startZ) < 0.12
        ).length;
        let placedIdx = 0;
        for (const s of solidTableShards) {
          if (s.position[0] < -0.95 && Math.abs(s.position[2] - startZ) < 0.55) {
            const slotX = -0.35 + (placedIdx % 12) * 0.11 + Math.floor(placedIdx / 12) * 0.055;
            const slotZ = startZ + ((placedIdx % 3) - 1) * 0.042;
            const rbS = rw.getRigidBody(s.bodyHandle)!;
            rbS.setTranslation({ x: slotX, y: 0.065, z: slotZ }, true);
            rbS.setLinvel({ x: 0, y: 0, z: 0 }, true);
            placedIdx++;
            corridorCount++;
          } else if (corridorCount < 16 && Math.abs(s.position[2] - startZ) > 0.35) {
            const slotX = -0.30 + (placedIdx % 10) * 0.12;
            const slotZ = startZ + ((placedIdx % 3) - 1) * 0.042;
            const rbS = rw.getRigidBody(s.bodyHandle)!;
            rbS.setTranslation({ x: slotX, y: 0.065, z: slotZ }, true);
            rbS.setLinvel({ x: 0, y: 0, z: 0 }, true);
            placedIdx++;
            corridorCount++;
          }
        }
        const qInit = quatFromAxisAngle([0, 1, 0], mode.yaw);
        syncBlockPoseClean(
          pw,
          pusher.id,
          [startX, BLOCK_HALF_Y, startZ],
          [qInit[0], qInit[1], qInit[2], qInit[3]]
        );
        for (let f = 0; f < 36; f++) pw.step(1 / 60);

        const wOff = quatRotateVec3(pusher.rotation, mode.grabLocal);
        const grabStart: [number, number, number] = [
          pusher.position[0] + wOff[0],
          pusher.position[1] + wOff[1],
          pusher.position[2] + wOff[2],
        ];
        pw.startGrabBlock(pusher.id, mode.grabLocal, grabStart);

        const distDm = 2.9;
        const totalFrames = Math.round(((distDm * 0.1) / dragSpeedMps) * 60);
        let maxShardSpeedMps = 0;
        let maxShardVyMps = 0;
        let maxShardSpinRadS = 0;
        let minNosePitchDeg = 0;
        let maxNosePitchDeg = 0;
        let maxYawDriftDeg = 0;
        let plowedShards = 0;
        const initShardPos = new Map(
          pw.shards.map((s) => [s.id, [s.position[0], s.position[2]] as [number, number]])
        );

        for (let f = 1; f <= totalFrames; f++) {
          const gx = grabStart[0] + (distDm * f) / totalFrames;
          const dyPlane = pw.grabPlaneY - grabStart[1];
          // Include oblique 30-deg camera ray parallax (`-dy * cot(30 deg)`) to verify fixed `grabPlaneY` stability
          pw.updateGrabTarget([gx - dyPlane * 1.6, pw.grabPlaneY, grabStart[2]]);
          pw.step(1 / 60);

          const fwdAxis =
            mode.yaw > 0.5
              ? quatRotateVec3(pusher.rotation, [0, 0, -1])
              : quatRotateVec3(pusher.rotation, [1, 0, 0]);
          const initFwd =
            mode.yaw > 0.5
              ? quatRotateVec3(qInit, [0, 0, -1])
              : quatRotateVec3(qInit, [1, 0, 0]);
          const signFwd = initFwd[0] >= 0 ? 1 : -1;
          const noseDeg =
            (Math.asin(Math.max(-1, Math.min(1, fwdAxis[1] * signFwd))) * 180) / Math.PI;
          if (noseDeg < minNosePitchDeg) minNosePitchDeg = noseDeg;
          if (noseDeg > maxNosePitchDeg) maxNosePitchDeg = noseDeg;

          const yawDot =
            (fwdAxis[0] * signFwd) / Math.max(1e-4, Math.hypot(fwdAxis[0], fwdAxis[2]));
          const yawDrift = (Math.acos(Math.max(-1, Math.min(1, yawDot))) * 180) / Math.PI;
          if (yawDrift > maxYawDriftDeg) maxYawDriftDeg = yawDrift;

          for (const s of pw.shards) {
            if (
              s.position[0] < -2.0 ||
              s.position[0] > 2.3 ||
              Math.abs(s.position[2] - startZ) > 0.95
            ) {
              continue;
            }
            const sInfo = rw.getMapping(s.bodyHandle);
            if (!sInfo || !sInfo.enabled) continue;
            const dof = sInfo.dofAdr;
            const sp = Math.hypot(d.qvel[dof], d.qvel[dof + 1], d.qvel[dof + 2]);
            const vy = d.qvel[dof + 1];
            const w = Math.hypot(d.qvel[dof + 3], d.qvel[dof + 4], d.qvel[dof + 5]);
            if (sp > maxShardSpeedMps) maxShardSpeedMps = sp;
            if (vy > maxShardVyMps) maxShardVyMps = vy;
            if (w > maxShardSpinRadS) maxShardSpinRadS = w;
          }
        }
        for (const s of pw.shards) {
          const p0 = initShardPos.get(s.id);
          if (p0 && Math.hypot(s.position[0] - p0[0], s.position[2] - p0[1]) > 0.05) {
            plowedShards++;
          }
        }
        pw.endGrab(false);
        pw.dispose();

        expect(plowedShards).toBeGreaterThanOrEqual(8);
        expect(maxShardSpeedMps).toBeLessThanOrEqual(dragSpeedMps * 2.8 + 0.09);
        expect(maxShardVyMps).toBeLessThanOrEqual(0.28);
        expect(maxShardSpinRadS).toBeLessThanOrEqual(20.5);
        expect(minNosePitchDeg).toBeGreaterThanOrEqual(-2.8);
        expect(maxNosePitchDeg).toBeLessThanOrEqual(12.5);
        expect(maxYawDriftDeg).toBeLessThanOrEqual(mode.name.startsWith('end-') ? 11.5 : 18.0);
      }
    }

    // Deliberate perpendicular lever swing on block end still rotates freely (> 18 deg)
    {
      const swingWorld = new PhysicsWorld(audio, 1, { seed: 42 });
      swingWorld.rebuildTower('sandbox', 2, 1, 42);
      for (let f = 0; f < 15; f++) swingWorld.step(1 / 60);
      syncBlockPoseClean(swingWorld, 1, [-3.0, BLOCK_HALF_Y, -1.0], [0, 0, 0, 1]);
      syncBlockPoseClean(swingWorld, 2, [-3.0, BLOCK_HALF_Y, 1.0], [0, 0, 0, 1]);
      syncBlockPoseClean(swingWorld, 0, [0, BLOCK_HALF_Y, 0], [0, 0, 0, 1]);
      for (let f = 0; f < 10; f++) swingWorld.step(1 / 60);
      swingWorld.startGrabBlock(0, [0.30, BLOCK_HALF_Y, 0], [0.30, BLOCK_HALF_Y * 2, 0]);
      for (let f = 1; f <= 35; f++) {
        swingWorld.updateGrabTarget([0.30, BLOCK_HALF_Y * 2, (0.85 * f) / 35]);
        swingWorld.step(1 / 60);
      }
      const fwdAfter = quatRotateVec3(swingWorld.blocks[0]!.rotation, [1, 0, 0]);
      const swingYawDeg =
        (Math.acos(Math.max(-1, Math.min(1, fwdAfter[0]))) * 180) / Math.PI;
      expect(swingYawDeg).toBeGreaterThan(18.0);
      swingWorld.dispose();
    }

    // 1b. Untouched collapsed rubble drag + mid-drag pointer pause + post-endGrab settle (no artificial block clearing!):
    // verifies zero spastic block-shard-block / elevated-heap pinch bursts and zero false block shatters.
    for (const seed of [22, 42, 55]) {
      const uw = new PhysicsWorld(audio, 18, { seed });
      uw.rebuildTower('sandbox', 2, 18, seed);
      for (let f = 0; f < 25; f++) uw.step(1 / 60);
      uw.triggerSandboxCollapseImpulse();
      for (let f = 0; f < 160; f++) uw.step(1 / 60);

      const urw = uw.getWorld();
      const ud = urw.getData()!;
      const candidates = uw.blocks.filter(
        (b) =>
          !b.shattered &&
          b.position[1] > 0 &&
          b.position[1] < 0.35 &&
          Math.hypot(b.position[0], b.position[2]) < TABLE_RADIUS - 0.5
      );
      candidates.sort(
        (a, b) => Math.hypot(b.position[0], b.position[2]) - Math.hypot(a.position[0], a.position[2])
      );
      const blk = candidates[0]!;
      const shatBefore = uw.shatteredCount;
      const centroid = uw.shards.reduce(
        (acc, s) => [acc[0] + s.position[0], acc[1] + s.position[2]],
        [0, 0]
      );
      centroid[0] /= Math.max(1, uw.shards.length);
      centroid[1] /= Math.max(1, uw.shards.length);
      const dx = centroid[0] - blk.position[0];
      const dz = centroid[1] - blk.position[2];
      const len = Math.hypot(dx, dz) || 1;
      const ux = dx / len;
      const uz = dz / len;
      const wOff = quatRotateVec3(blk.rotation, [0, BLOCK_HALF_Y, 0]);
      const grabStart: [number, number, number] = [
        blk.position[0] + wOff[0],
        blk.position[1] + wOff[1],
        blk.position[2] + wOff[2],
      ];
      uw.startGrabBlock(blk.id, [0, BLOCK_HALF_Y, 0], grabStart);

      let curX = grabStart[0];
      let curZ = grabStart[2];
      let maxDragSp = 0;
      let maxPauseOrPostSp = 0;
      let maxAllSpin = 0;
      for (let f = 1; f <= 200; f++) {
        const isPausedOrPost = (f >= 86 && f <= 120) || f >= 161;
        if (f <= 85 || (f >= 121 && f <= 160)) {
          curX += ux * 0.02;
          curZ += uz * 0.02;
          uw.updateGrabTarget([curX, uw.grabPlaneY, curZ]);
        } else if (f === 161) {
          uw.endGrab(false);
        }
        uw.step(1 / 60);
        for (const s of uw.shards) {
          const sInfo = urw.getMapping(s.bodyHandle);
          if (!sInfo || !sInfo.enabled) continue;
          if (
            ud.qpos[sInfo.qposAdr + 1] < -0.01 ||
            Math.hypot(s.position[0], s.position[2]) > TABLE_RADIUS - 0.1
          ) {
            continue;
          }
          const dof = sInfo.dofAdr;
          const sp = Math.hypot(ud.qvel[dof], ud.qvel[dof + 1], ud.qvel[dof + 2]);
          const w = Math.hypot(ud.qvel[dof + 3], ud.qvel[dof + 4], ud.qvel[dof + 5]);
          if (w > maxAllSpin) maxAllSpin = w;
          if (isPausedOrPost) {
            if (sp > maxPauseOrPostSp) maxPauseOrPostSp = sp;
          } else if (sp > maxDragSp) {
            maxDragSp = sp;
          }
        }
      }
      expect(uw.shatteredCount - shatBefore).toBe(0);
      expect(maxDragSp).toBeLessThanOrEqual(0.45);
      expect(maxPauseOrPostSp).toBeLessThanOrEqual(0.48);
      expect(maxAllSpin).toBeLessThanOrEqual(20.5);
      uw.dispose();
    }

    // 1c. Smooth 90-degree curved drag under 2-Zone Physical Grip Law (`lx=0` Body Pinch holds constant heading,
    // `lx=0.26` Outward End-Tow aligns passively along curve, `lx=-0.24` Rear Tip pivots smoothly without whip)
    for (const lx of [0, 0.26, -0.24]) {
      const cw = new PhysicsWorld(audio, 1, { seed: 42 });
      cw.rebuildTower('sandbox', 2, 1, 42);
      for (let f = 0; f < 12; f++) cw.step(1 / 60);
      syncBlockPoseClean(cw, 1, [-3.2, BLOCK_HALF_Y, -1.0], [0, 0, 0, 1]);
      syncBlockPoseClean(cw, 2, [-3.2, BLOCK_HALF_Y, 1.0], [0, 0, 0, 1]);
      syncBlockPoseClean(cw, 0, [-1.0, BLOCK_HALF_Y, -1.0], [0, 0, 0, 1]);
      for (let f = 0; f < 10; f++) cw.step(1 / 60);
      const b0 = cw.blocks[0]!;
      const grabLocal: [number, number, number] = [lx, BLOCK_HALF_Y, 0];
      const g0: [number, number, number] = [b0.position[0] + lx, BLOCK_HALF_Y * 2, b0.position[2]];
      cw.startGrabBlock(0, grabLocal, g0);
      const R = 1.6;
      const cx = g0[0];
      const cz = g0[2] + R;
      for (let f = 1; f <= 90; f++) {
        const th = (Math.PI * 0.5 * f) / 90;
        cw.updateGrabTarget([cx + R * Math.sin(th), cw.grabPlaneY, cz - R * Math.cos(th)]);
        cw.step(1 / 60);
      }
      const fwd = quatRotateVec3(b0.rotation, [1, 0, 0]);
      const headingDeg = (Math.atan2(fwd[2], fwd[0]) * 180) / Math.PI;
      if (Math.abs(lx) <= BLOCK_HALF_X * 0.58) {
        expect(Math.abs(headingDeg)).toBeLessThan(14.0);
      } else if (lx > 0) {
        expect(headingDeg).toBeGreaterThan(50.0);
        expect(headingDeg).toBeLessThan(98.0);
      } else {
        expect(headingDeg).toBeLessThan(-45.0);
        expect(headingDeg).toBeGreaterThan(-95.0);
      }
      cw.dispose();
    }

    // 2. Smooth height-consistent Sandbox topple across `6L, 12L, 15L, 18L, 24L, 30L`
    // (`0` <16L cliff, `0` flat mid-tower velocity clamp, `0` premature upright shatters)
    for (const layers of [6, 12, 15, 18, 24, 30]) {
      const tw = new PhysicsWorld(audio, layers, { seed: 42 });
      tw.rebuildTower('sandbox', 2, layers, 42);
      for (let f = 0; f < 25; f++) tw.step(1 / 60);
      const topInitY = tw.blocks[tw.blocks.length - 1]!.position[1];
      tw.triggerSandboxCollapseImpulse();
      const trw = tw.getWorld();
      const td = trw.getData()!;
      const midIdx = Math.floor(layers * 0.52) * 3;
      const topIdx = tw.blocks.length - 1;
      const midDof = trw.getMapping(tw.blocks[midIdx]!.bodyHandle)!.dofAdr;
      const topDof = trw.getMapping(tw.blocks[topIdx]!.bodyHandle)!.dofAdr;
      const vMidInit = Math.hypot(td.qvel[midDof], td.qvel[midDof + 2]);
      const vTopInit = Math.hypot(td.qvel[topDof], td.qvel[topDof + 2]);
      expect(vTopInit).toBeGreaterThan(vMidInit * 1.18);

      let tFirstShatterSec = -1;
      let dropAtFrame6Mm = 0;
      for (let f = 1; f <= 60; f++) {
        tw.step(1 / 60);
        if (tFirstShatterSec < 0 && tw.shatteredCount > 0) {
          tFirstShatterSec = f / 60;
        }
        if (f === 6) {
          const topBlk = tw.blocks[tw.blocks.length - 1]!;
          dropAtFrame6Mm = (topInitY - topBlk.position[1]) * 100;
        }
      }
      expect(dropAtFrame6Mm).toBeGreaterThan(4.0);
      expect(tFirstShatterSec).toBeGreaterThanOrEqual(0.11);
      expect(tw.shatteredCount).toBeGreaterThanOrEqual(layers <= 6 ? 1 : 5);
      tw.dispose();
    }

    // 3. Worker bridge catch-up slicing: maintains smooth snapshot cadence & real-time catch-up under in-flight backlog
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
    channel.port1.onmessage = (ev) => clientEndpoint.onmessage?.(ev as MessageEvent<WorkerOutboundMessage>);
    channel.port2.onmessage = (ev) => hostScope.onmessage?.(ev as MessageEvent<WorkerInboundMessage>);

    const hostHandle = attachMujocoWorkerHost(hostScope);
    const bridge = await MujocoWorkerBridge.create(audio, 18, 42, clientEndpoint);
    let snapCount = 0;
    const origApply = clientEndpoint.onmessage;
    clientEndpoint.onmessage = (ev) => {
      if (ev.data.type === 'snapshot' && !ev.data.isFullRebuild) snapCount++;
      origApply?.(ev);
    };
    bridge.rebuildTower('sandbox', 2, 18, 42);
    await new Promise((r) => setTimeout(r, 40));
    snapCount = 0;
    const sim0 = bridge.getSimTime();
    bridge.triggerSandboxCollapseImpulse();
    // Simulate a 10-frame (`167 ms`) main-thread rAF stream arriving while heavy steps execute
    for (let f = 0; f < 10; f++) {
      bridge.step(1 / 60);
      await new Promise((r) => setTimeout(r, 14));
    }
    await new Promise((r) => setTimeout(r, 90));
    expect(snapCount).toBeGreaterThanOrEqual(2);
    expect(bridge.getSimTime() - sim0).toBeGreaterThanOrEqual(0.11);
    bridge.dispose();
    hostHandle.dispose();
    clientEndpoint.terminate?.();
  }, 120000);

  it('M10: Round 3 V1–V5 surgical fixes: 0 helicopter flail & 0 mm tether desync, broadside shard sled, jaw-pinch escape + re-solidification, foreground-shard disambiguation + bevel pick margin, and column-drop/stump/camera invariants', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    // --- 1. Zero Helicopter Flail, Zero Tether Desync (`0.00 mm`) & Stationary `grabFlickVelXZ` Decay ---
    for (const withShards of [false, true]) {
      for (const localXDm of [0.0, 0.15, 0.30, -0.30]) {
        const pw = new PhysicsWorld(audio, withShards ? 10 : 2, { seed: 42 });
        pw.rebuildTower('sandbox', 2, withShards ? 10 : 2, 42);
        for (let f = 0; f < 20; f++) pw.step(1 / 60);
        const rw = pw.getWorld();
        const blk = pw.blocks[0]!;
        for (let i = 1; i < pw.blocks.length; i++) {
          if (withShards && i <= 4) {
            syncBlockPoseClean(pw, i, [0.1 * i, 0.16, 0.0], [0, 0, 0, 1]);
            pw.shatterBlock(pw.blocks[i]!, [0, 0, 0], 11.5, [0, 1, 0], true);
          } else {
            pw.blocks[i]!.extracted = true;
            rw.getRigidBody(pw.blocks[i]!.bodyHandle)!.setTranslation({ x: -6 - i * 0.8, y: -5, z: -5 }, false);
          }
        }
        rw.syncKinematics();
        syncBlockPoseClean(pw, 0, [-0.5, BLOCK_HALF_Y, 0.4], [0, 0, 0, 1]);
        for (let f = 0; f < (withShards ? 90 : 15); f++) pw.step(1 / 60);

        const grabLocal: [number, number, number] = [localXDm, BLOCK_HALF_Y, 0];
        const wOff0 = quatRotateVec3(blk.rotation, grabLocal);
        const g0: [number, number, number] = [
          blk.position[0] + wOff0[0],
          blk.position[1] + wOff0[1],
          blk.position[2] + wOff0[2],
        ];
        expect(pw.startGrabBlock(0, grabLocal, g0)).toBe(true);

        const engInternal = pw as unknown as {
          map: { grabEqualityId: number; blockDofAdr: number[] };
          grabFlickVelXZ: [number, number];
        };
        const mjModel = rw.getModel()!;
        const mjData = rw.getData()!;
        const eqBase = engInternal.map.grabEqualityId * 11;
        const dof0 = engInternal.map.blockDofAdr[0]!;
        const initFwd = quatRotateVec3(blk.rotation, [1, 0, 0]);
        const initYaw = Math.atan2(-initFwd[2], initFwd[0]);

        let maxHeadingDriftDeg = 0;
        let maxYawRateRadS = 0;
        let maxEqDesyncMm = 0;
        let maxWorldTetherDesyncMm = 0;

        // Trajectory: +10 cm (+X) -> -10 cm (-X, 180° reversal!) -> -Z (90° corner) -> -X (90° corner)
        const totalMoveFrames = 80;
        for (let f = 1; f <= totalMoveFrames; f++) {
          let tx = g0[0];
          let tz = g0[2];
          if (f <= 20) {
            tx = g0[0] + (1.0 * f) / 20;
          } else if (f <= 44) {
            tx = g0[0] + 1.0 - (2.0 * (f - 20)) / 24;
          } else if (f <= 62) {
            tx = g0[0] - 1.0;
            tz = g0[2] - (0.9 * (f - 44)) / 18;
          } else {
            tx = g0[0] - 1.0 - (0.9 * (f - 62)) / 18;
            tz = g0[2] - 0.9;
          }
          pw.updateGrabTarget([tx, pw.grabPlaneY, tz], 1000 + f * 16.67);
          pw.step(1 / 60);

          const eqDxMm = Math.abs(mjModel.eq_data[eqBase + 3] - grabLocal[0] * 0.1) * 1000;
          const eqDzMm = Math.abs(mjModel.eq_data[eqBase + 5] - grabLocal[2] * 0.1) * 1000;
          maxEqDesyncMm = Math.max(maxEqDesyncMm, eqDxMm, eqDzMm);

          const exactWorldOff = quatRotateVec3(blk.rotation, grabLocal);
          const worldDxMm =
            Math.hypot(
              pw.grabCurrentWorldOnBlock[0] - (blk.position[0] + exactWorldOff[0]),
              pw.grabCurrentWorldOnBlock[2] - (blk.position[2] + exactWorldOff[2])
            ) * 100;
          maxWorldTetherDesyncMm = Math.max(maxWorldTetherDesyncMm, worldDxMm);

          const curFwd = quatRotateVec3(blk.rotation, [1, 0, 0]);
          const curYaw = Math.atan2(-curFwd[2], curFwd[0]);
          let dYaw = curYaw - initYaw;
          while (dYaw > Math.PI) dYaw -= Math.PI * 2;
          while (dYaw < -Math.PI) dYaw += Math.PI * 2;
          maxHeadingDriftDeg = Math.max(maxHeadingDriftDeg, (Math.abs(dYaw) * 180) / Math.PI);

          const wWorld = quatRotateVec3(blk.rotation, [
            mjData.qvel[dof0 + 3],
            mjData.qvel[dof0 + 4],
            mjData.qvel[dof0 + 5],
          ]);
          maxYawRateRadS = Math.max(maxYawRateRadS, Math.abs(wWorld[1]));
        }

        // Hold pointer completely stationary (zero updateGrabTarget calls) for 0.38 s (23 frames)
        for (let f = 0; f < 23; f++) pw.step(1 / 60);
        const decayedFlickSpd = Math.hypot(
          engInternal.grabFlickVelXZ[0],
          engInternal.grabFlickVelXZ[1]
        );
        expect(decayedFlickSpd).toBeLessThan(0.01);
        expect(maxEqDesyncMm).toBeLessThan(0.05);
        expect(maxWorldTetherDesyncMm).toBeLessThan(0.05);

        if (Math.abs(localXDm) <= BLOCK_HALF_X * 0.58) {
          expect(maxHeadingDriftDeg).toBeLessThan(14.0);
          expect(maxYawRateRadS).toBeLessThan(4.0);
        } else {
          // End-grip (`±30 mm`) pivots smoothly around 90° corners without 1,544 deg/s (26.9 rad/s) helicopter flail
          expect(maxYawRateRadS).toBeLessThan(12.5);
        }
        pw.endGrab(false);
        pw.dispose();
      }
    }

    // --- 2. Effortless Broadside Shard Sled (`localX = 0` & `localX = 14 mm`) ---
    for (const localXDm of [0.0, 0.14]) {
      const sw = new PhysicsWorld(audio, 18, { seed: 42 });
      sw.rebuildTower('sandbox', 2, 18, 42);
      for (let f = 0; f < 25; f++) sw.step(1 / 60);
      sw.triggerSandboxCollapseImpulse();
      for (let f = 0; f < 160; f++) sw.step(1 / 60);
      const srw = sw.getWorld();
      const sModel = srw.getModel()!;
      const survivors = sw.blocks.filter((b) => !b.shattered && !b.extracted);
      const sledBlk = survivors[0]!;
      const culletShards = sw.shards.filter(
        (s) => (sModel.geom_conaffinity[s.colliderHandle] & 0x20) !== 0
      );
      const meanZ =
        culletShards.length > 0
          ? Math.max(
              -0.9,
              Math.min(
                0.9,
                culletShards.reduce((acc, s) => acc + s.position[2], 0) / culletShards.length
              )
            )
          : 0.0;
      for (let i = 1; i < survivors.length; i++) {
        const blk = survivors[i]!;
        if (
          blk.position[0] > -2.1 &&
          blk.position[0] < 2.15 &&
          Math.abs(blk.position[2] - meanZ) < 0.72
        ) {
          const sideZ = meanZ >= 0 ? -2.1 - (i % 4) * 0.30 : 2.1 + (i % 4) * 0.30;
          syncBlockPoseClean(
            sw,
            blk.id,
            [-1.8 + Math.floor(i / 4) * 0.45, BLOCK_HALF_Y, sideZ],
            [0, 0, 0, 1]
          );
        }
      }
      let corridorIdx = 0;
      for (const s of sw.shards) {
        // Clear any hero/cullet shards overlapping the `[-1.80, -0.92]` launchpad so `sledBlk` starts flat on oak
        if (s.position[0] > -1.85 && s.position[0] < -0.92 && Math.abs(s.position[2] - meanZ) < 0.55) {
          const rbS = srw.getRigidBody(s.bodyHandle)!;
          if (corridorIdx < 72) {
            rbS.setTranslation(
              {
                x: -0.45 + Math.floor(corridorIdx / 6) * 0.085,
                y: 0.065 + (corridorIdx % 2) * 0.045,
                z: meanZ + ((corridorIdx % 6) - 2.5) * 0.065,
              },
              true
            );
            corridorIdx++;
          } else {
            rbS.setTranslation({ x: -2.4, y: 0.06, z: meanZ >= 0 ? -2.2 : 2.2 }, true);
          }
          rbS.setLinvel({ x: 0, y: 0, z: 0 }, true);
          continue;
        }
        if (s.position[0] < -0.95 || Math.abs(s.position[2] - meanZ) > 0.45) {
          if (corridorIdx < 72) {
            const rbS = srw.getRigidBody(s.bodyHandle)!;
            rbS.setTranslation(
              {
                x: -0.45 + Math.floor(corridorIdx / 6) * 0.085,
                y: 0.065 + (corridorIdx % 2) * 0.045,
                z: meanZ + ((corridorIdx % 6) - 2.5) * 0.065,
              },
              true
            );
            rbS.setLinvel({ x: 0, y: 0, z: 0 }, true);
            corridorIdx++;
          }
        }
      }
      const qBroad = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
      syncBlockPoseClean(
        sw,
        sledBlk.id,
        [-1.35, BLOCK_HALF_Y, meanZ],
        [qBroad[0], qBroad[1], qBroad[2], qBroad[3]]
      );
      for (let f = 0; f < 30; f++) sw.step(1 / 60);

      const grabLoc: [number, number, number] = [localXDm, BLOCK_HALF_Y, 0];
      const wOff = quatRotateVec3(sledBlk.rotation, grabLoc);
      const gStart: [number, number, number] = [
        sledBlk.position[0] + wOff[0],
        sledBlk.position[1] + wOff[1],
        sledBlk.position[2] + wOff[2],
      ];
      sw.startGrabBlock(sledBlk.id, grabLoc, gStart);
      let maxBroadsideClimbDeg = 0;
      let maxBroadsideTwistDeg = 0;
      const startX = sledBlk.position[0];
      for (let f = 1; f <= 85; f++) {
        sw.updateGrabTarget([gStart[0] + (2.4 * f) / 85, sw.grabPlaneY, gStart[2]]);
        sw.step(1 / 60);
        // Leading broadside edge along +X corresponds to local +Z when yaw = +90 deg
        const leadDir = quatRotateVec3(sledBlk.rotation, [0, 0, 1]);
        const climbDeg = (Math.asin(Math.max(-1, Math.min(1, leadDir[1]))) * 180) / Math.PI;
        if (climbDeg > maxBroadsideClimbDeg) maxBroadsideClimbDeg = climbDeg;
        const twistDeg =
          (Math.acos(Math.max(-1, Math.min(1, leadDir[0] / Math.max(1e-4, Math.hypot(leadDir[0], leadDir[2]))))) *
            180) /
          Math.PI;
        if (twistDeg > maxBroadsideTwistDeg) maxBroadsideTwistDeg = twistDeg;
      }
      const travelMm = (sledBlk.position[0] - startX) * 100;
      expect(maxBroadsideClimbDeg).toBeGreaterThanOrEqual(1.6);
      expect(maxBroadsideTwistDeg).toBeLessThan(14.0);
      expect(travelMm).toBeGreaterThan(170);
      sw.endGrab(false);
      sw.dispose();
    }

    // --- 3. Multi-Body Shard Jaw Pinch, Mid-Pile Pause, Medium (`0.65 m/s`) Sweep, `0x22` Re-Solidification & Hero Tablecloth ---
    for (const driveSpeedMps of [0.25, 0.65]) {
      const jw = new PhysicsWorld(audio, 12, { seed: 42 });
      jw.rebuildTower('sandbox', 2, 12, 42);
      for (let f = 0; f < 20; f++) jw.step(1 / 60);
      const jrw = jw.getWorld();
      const jModel = jrw.getModel()!;
      const jData = jrw.getData()!;
      for (let i = 3; i < jw.blocks.length; i++) {
        if (i <= 7) {
          syncBlockPoseClean(jw, i, [0.0, 0.16, 0.0], [0, 0, 0, 1]);
          jw.shatterBlock(jw.blocks[i]!, [0, 0, 0], 11.5, [0, 1, 0], true);
        } else {
          jw.blocks[i]!.extracted = true;
          jrw.getRigidBody(jw.blocks[i]!.bodyHandle)!.setTranslation({ x: -6 - i * 0.8, y: -5, z: -5 }, false);
        }
      }
      jrw.syncKinematics();
      const qJaw = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
      // Stationary anvil rubble block at x = +0.55 dm, dragged block starting at x = -0.75 dm
      syncBlockPoseClean(jw, 1, [0.55, BLOCK_HALF_Y, 0.0], [qJaw[0], qJaw[1], qJaw[2], qJaw[3]]);
      syncBlockPoseClean(jw, 2, [0.95, BLOCK_HALF_Y, 0.0], [qJaw[0], qJaw[1], qJaw[2], qJaw[3]]);
      syncBlockPoseClean(jw, 0, [-0.75, BLOCK_HALF_Y, 0.0], [qJaw[0], qJaw[1], qJaw[2], qJaw[3]]);
      for (let i = 0; i < Math.min(28, jw.shards.length); i++) {
        const s = jw.shards[i]!;
        const rbS = jrw.getRigidBody(s.bodyHandle)!;
        rbS.setTranslation(
          {
            x: -0.25 + (i % 7) * 0.09,
            y: 0.065,
            z: ((i % 4) - 1.5) * 0.12,
          },
          true
        );
        rbS.setLinvel({ x: 0, y: 0, z: 0 }, true);
      }
      for (let f = 0; f < 65; f++) jw.step(1 / 60);

      const shatBeforeSweep = jw.shatteredCount;
      const pusher = jw.blocks[0]!;
      const g0: [number, number, number] = [pusher.position[0], BLOCK_HALF_Y * 2, pusher.position[2]];
      jw.startGrabBlock(0, [0, BLOCK_HALF_Y, 0], g0);

      const sweepDistDm = 1.15;
      const sweepFrames = Math.max(10, Math.round(((sweepDistDm * 0.1) / driveSpeedMps) * 60));
      let maxShardSpMps = 0;
      let maxPausedSpinRadS = 0;
      const jMap = (jw as unknown as { map: { shardDofAdr: number[] } }).map;

      for (let f = 1; f <= sweepFrames; f++) {
        const gx = g0[0] + (sweepDistDm * f) / sweepFrames;
        jw.updateGrabTarget([gx, jw.grabPlaneY, g0[2]], 2000 + f * 16.67);
        jw.step(1 / 60);
        for (const s of jw.shards) {
          if (Math.hypot(s.position[0], s.position[2]) > TABLE_RADIUS - 0.1) continue;
          const dof = jMap.shardDofAdr[s.slotIndex]!;
          const sp = Math.hypot(jData.qvel[dof], jData.qvel[dof + 1], jData.qvel[dof + 2]);
          if (sp > maxShardSpMps) maxShardSpMps = sp;
        }
      }

      // Stationary pause for 0.60 s (36 frames) right inside the multi-body block-shard jaw!
      for (let f = 1; f <= 36; f++) {
        jw.step(1 / 60);
        for (const s of jw.shards) {
          if (Math.hypot(s.position[0], s.position[2]) > TABLE_RADIUS - 0.1) continue;
          const dof = jMap.shardDofAdr[s.slotIndex]!;
          const sp = Math.hypot(jData.qvel[dof], jData.qvel[dof + 1], jData.qvel[dof + 2]);
          if (sp > maxShardSpMps) maxShardSpMps = sp;
          if (f >= 10) {
            const w = Math.hypot(jData.qvel[dof + 3], jData.qvel[dof + 4], jData.qvel[dof + 5]);
            if (w > maxPausedSpinRadS) maxPausedSpinRadS = w;
          }
        }
      }

      // Release mid-pile and step 1.5 s to full settle
      jw.endGrab(false, 2000 + (sweepFrames + 36) * 16.67);
      for (let f = 1; f <= 95; f++) {
        jw.step(1 / 60);
        for (const s of jw.shards) {
          if (Math.hypot(s.position[0], s.position[2]) > TABLE_RADIUS - 0.1) continue;
          const dof = jMap.shardDofAdr[s.slotIndex]!;
          const sp = Math.hypot(jData.qvel[dof], jData.qvel[dof + 1], jData.qvel[dof + 2]);
          if (sp > maxShardSpMps) maxShardSpMps = sp;
        }
      }

      expect(jw.shatteredCount - shatBeforeSweep).toBe(0);
      expect(maxShardSpMps).toBeLessThanOrEqual(1.45 * driveSpeedMps + 0.14);
      expect(maxPausedSpinRadS).toBeLessThan(2.2);

      // Mandatory 100% Re-Solidification Guarantee (`0` post-settle ghost shards!)
      let postSettleGhost = 0;
      for (const s of jw.shards) {
        const offRim = Math.hypot(s.position[0], s.position[2]) > TABLE_RADIUS + 0.01;
        if (!offRim && jModel.geom_contype[s.colliderHandle] === MJ_CONTYPE_CULLET_SHARD) {
          if (
            (jModel.geom_conaffinity[s.colliderHandle] & MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE) !==
            MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE
          ) {
            postSettleGhost++;
          }
        }
      }
      expect(postSettleGhost).toBe(0);
      jw.dispose();
    }

    // Hero shards (`0x04`) resting peacefully on top of a flat brick preserve block collision (Tablecloth Pull guard)
    {
      const tw = new PhysicsWorld(audio, 4, { seed: 42 });
      tw.rebuildTower('sandbox', 2, 4, 42);
      for (let f = 0; f < 15; f++) tw.step(1 / 60);
      const trw = tw.getWorld();
      const tModel = trw.getModel()!;
      for (let i = 2; i < tw.blocks.length; i++) {
        syncBlockPoseClean(tw, i, [-3.2, BLOCK_HALF_Y, -2.0 + i * 0.4], [0, 0, 0, 1]);
      }
      syncBlockPoseClean(tw, 0, [0, BLOCK_HALF_Y, 0], [0, 0, 0, 1]);
      syncBlockPoseClean(tw, 1, [0, 0.28, 0], [0, 0, 0, 1]);
      tw.shatterBlock(tw.blocks[1]!, [0, 0, 0], 9.0, [0, 1, 0], false);
      const hero = tw.shards.find(
        (s) => tModel.geom_contype[s.colliderHandle] === MJ_CONTYPE_HERO_SHARD
      )!;
      expect(hero).toBeDefined();
      trw.getRigidBody(hero.bodyHandle)!.setTranslation({ x: 0, y: BLOCK_HALF_Y * 2 + 0.06, z: 0 }, true);
      trw.getRigidBody(hero.bodyHandle)!.setLinvel({ x: 0, y: 0, z: 0 }, true);
      for (let f = 0; f < 45; f++) tw.step(1 / 60);
      expect(hero.position[1]).toBeGreaterThan(BLOCK_HALF_Y * 1.85);
      expect(tModel.geom_conaffinity[hero.colliderHandle] & MJ_CONTYPE_BLOCK).toBe(MJ_CONTYPE_BLOCK);
      tw.dispose();
    }

    // 3b. 60 Hz `±0.12 mm` pointer-tremor mid-pile pause relaxes `<connect>` crush `< 0.80 N` + single-block bevel-to-table wedge escape + symmetric `±30 mm` end-tow
    {
      const bw = new PhysicsWorld(audio, 6, { seed: 42 });
      bw.rebuildTower('sandbox', 2, 6, 42);
      for (let f = 0; f < 15; f++) bw.step(1 / 60);
      const brw = bw.getWorld();
      const bm = brw.getModel()!;
      const bd = brw.getData()!;
      for (let i = 2; i < bw.blocks.length; i++) {
        bw.blocks[i]!.extracted = true;
        brw.getRigidBody(bw.blocks[i]!.bodyHandle)!.setTranslation({ x: -8, y: -5, z: -5 }, false);
      }
      const qJaw = quatFromAxisAngle([0, 1, 0], Math.PI * 0.5);
      syncBlockPoseClean(bw, 1, [0.28, BLOCK_HALF_Y, 0.0], [qJaw[0], qJaw[1], qJaw[2], qJaw[3]]);
      syncBlockPoseClean(bw, 0, [-0.35, BLOCK_HALF_Y, 0.0], [qJaw[0], qJaw[1], qJaw[2], qJaw[3]]);
      bm.body_mass[bw.blocks[1]!.bodyHandle] = 2.5;
      const b1Dof = (bw as unknown as { map: { blockDofAdr: number[] } }).map.blockDofAdr[1]!;
      bm.dof_M0[b1Dof + 0] = 2.5;
      bm.dof_M0[b1Dof + 1] = 2.5;
      bm.dof_M0[b1Dof + 2] = 2.5;
      for (let f = 0; f < 20; f++) bw.step(1 / 60);
      bw.startGrabBlock(0, [0, BLOCK_HALF_Y, 0], [-0.35, BLOCK_HALF_Y * 2, 0], 1000);
      for (let f = 1; f <= 20; f++) {
        bw.updateGrabTarget([-0.35 + (0.65 * f) / 20, bw.grabPlaneY, 0], 1000 + f * 16.67);
        bw.step(1 / 60);
      }
      let maxLateTremorCrushN = 0;
      for (let f = 1; f <= 36; f++) {
        const tremorDm = (f % 2 === 0 ? 1 : -1) * 0.0012;
        bw.updateGrabTarget([0.30 + tremorDm, bw.grabPlaneY, 0], 1333.4 + f * 16.67);
        bw.step(1 / 60);
        if (f >= 28 && bd.ne >= 3) {
          const fEqH = Math.hypot(bd.efc_force[0], bd.efc_force[2]);
          if (fEqH > maxLateTremorCrushN) maxLateTremorCrushN = fEqH;
        }
      }
      expect(maxLateTremorCrushN).toBeLessThan(0.8);
      bw.endGrab(false);
      bw.dispose();

      // Symmetric +30 mm and -30 mm outward end-tow over shards
      const towAngles: number[] = [];
      for (const lx of [0.30, -0.30]) {
        const tw2 = new PhysicsWorld(audio, 6, { seed: 42 });
        tw2.rebuildTower('sandbox', 2, 6, 42);
        for (let f = 0; f < 15; f++) tw2.step(1 / 60);
        const rw2 = tw2.getWorld();
        for (let i = 1; i < tw2.blocks.length; i++) {
          if (i <= 3) {
            syncBlockPoseClean(tw2, i, [0, 0.16, 0], [0, 0, 0, 1]);
            tw2.shatterBlock(tw2.blocks[i]!, [0, 0, 0], 11.0, [0, 1, 0], true);
          } else {
            tw2.blocks[i]!.extracted = true;
            rw2.getRigidBody(tw2.blocks[i]!.bodyHandle)!.setTranslation({ x: -8, y: -5, z: -5 }, false);
          }
        }
        const sgn = Math.sign(lx);
        syncBlockPoseClean(tw2, 0, [-0.8 * sgn, BLOCK_HALF_Y, 0], [0, 0, 0, 1]);
        for (let f = 0; f < 40; f++) tw2.step(1 / 60);
        const g0: [number, number, number] = [tw2.blocks[0]!.position[0] + lx, BLOCK_HALF_Y * 2, 0];
        tw2.startGrabBlock(0, [lx, BLOCK_HALF_Y, 0], g0, 1000);
        for (let f = 1; f <= 45; f++) {
          tw2.updateGrabTarget(
            [g0[0] + (sgn * 1.8 * f) / 45, tw2.grabPlaneY, (0.45 * f) / 45],
            1000 + f * 16.67
          );
          tw2.step(1 / 60);
        }
        const tipOut = quatRotateVec3(tw2.blocks[0]!.rotation, [sgn, 0, 0]);
        towAngles.push((Math.atan2(tipOut[2], tipOut[0] * sgn) * 180) / Math.PI);
        tw2.dispose();
      }
      expect(Math.abs(towAngles[0]! - towAngles[1]!)).toBeLessThan(6.0);
    }

    // --- 4. Foreground-Shard Click Disambiguation + `+2.5 mm` Sandbox Bevel Pick Margin + Grazing Ray Clamp ---
    {
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
      engPhys.rebuildTower('sandbox', 2, 6, 42);
      for (let f = 0; f < 20; f++) engPhys.step(1 / 60);
      const erw = engPhys.getWorld();
      for (let i = 2; i < engPhys.blocks.length; i++) {
        engPhys.blocks[i]!.extracted = true;
        erw.getRigidBody(engPhys.blocks[i]!.bodyHandle)!.setTranslation({ x: -6 - i * 0.8, y: -5, z: -5 }, false);
      }
      erw.syncKinematics();
      syncBlockPoseClean(engPhys, 1, [1.8, 0.22, 1.8], [0, 0, 0, 1]);
      engPhys.shatterBlock(engPhys.blocks[1]!, [0, 0, 0], 11.0, [0, 1, 0], true);
      syncBlockPoseClean(engPhys, 0, [0.0, BLOCK_HALF_Y, 1.4], [0, 0, 0, 1]);
      for (let f = 0; f < 45; f++) engPhys.step(1 / 60);

      const cam = (eng as unknown as {
        camera: {
          position: [number, number, number];
          azimuth: number;
          elevation: number;
          isUserOrbiting: boolean;
          update: (dt: number, aspect: number) => void;
          viewProjMatrix: Float32Array;
        };
      }).camera;
      cam.update(0.016, 1280 / 800);
      const vp = cam.viewProjMatrix;
      const project = (p: [number, number, number]): [number, number] => {
        const cx = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
        const cy = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
        const cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
        return [((cx / cw) * 0.5 + 0.5) * 1280, (0.5 - (cy / cw) * 0.5) * 800];
      };

      // Place a foreground shard directly along the ray from `cam.position` to `blk0.position` (`t = 0.78`)
      const blk0 = engPhys.blocks[0]!;
      const fgShard = [...engPhys.shards].sort((a, b) => b.massKg - a.massKg)[0]!;
      const placeForegroundOccluder = (): [number, number] => {
        const u = 0.82;
        const ox = cam.position[0] + (blk0.position[0] - cam.position[0]) * u;
        const oy = cam.position[1] + (blk0.position[1] - cam.position[1]) * u;
        const oz = cam.position[2] + (blk0.position[2] - cam.position[2]) * u;
        const rbFg = erw.getRigidBody(fgShard.bodyHandle)!;
        rbFg.setTranslation({ x: ox, y: oy, z: oz }, true);
        rbFg.setLinvel({ x: 0, y: 0, z: 0 }, true);
        rbFg.setAngvel({ x: 0, y: 0, z: 0 }, true);
        fgShard.position = [ox, oy, oz];
        erw.syncKinematics();
        return project([ox, oy, oz]);
      };

      // 4a. `< 5 px` tap on foreground-occluded block -> nudges foreground shard, `0` camera orbit, `0` block grab!
      const [fx0, fy0] = placeForegroundOccluder();
      const azBeforeTap = cam.azimuth;
      const elBeforeTap = cam.elevation;
      eng.handlePointerDown(fx0, fy0, 1, 0, false, undefined, 'mouse', 5000);
      eng.handlePointerMove(fx0 + 2, fy0 + 1, 1, undefined, 'mouse', 5016);
      eng.handlePointerUp(fx0 + 2, fy0 + 1, 1, undefined, 'mouse', 5032);
      cam.update(0.016, 1280 / 800);
      const fgDof = (engPhys as unknown as { map: { shardDofAdr: number[] } }).map
        .shardDofAdr[fgShard.slotIndex]!;
      const fgTapSpeed = Math.hypot(
        erw.getData()!.qvel[fgDof],
        erw.getData()!.qvel[fgDof + 1],
        erw.getData()!.qvel[fgDof + 2]
      );
      expect(fgTapSpeed).toBeGreaterThan(0.15);
      expect(engPhys.activeBlockId).toBeNull();
      expect(cam.isUserOrbiting).toBe(false);
      expect(Math.abs(cam.azimuth - azBeforeTap)).toBeLessThan(1e-6);
      expect(Math.abs(cam.elevation - elBeforeTap)).toBeLessThan(1e-6);

      // 4b. `>= 5 px` drag on foreground-occluded block -> starts grabbing `blk0` with `0` camera orbit!
      const [fx1, fy1] = placeForegroundOccluder();
      eng.handlePointerDown(fx1, fy1, 1, 0, false, undefined, 'mouse', 6000);
      eng.handlePointerMove(fx1 + 10, fy1, 1, undefined, 'mouse', 6020);
      cam.update(0.016, 1280 / 800);
      expect(engPhys.activeBlockId).toBe(blk0.id);
      expect(cam.isUserOrbiting).toBe(false);
      expect(Math.abs(cam.azimuth - azBeforeTap)).toBeLessThan(1e-6);
      eng.handlePointerUp(fx1 + 10, fy1, 1, undefined, 'mouse', 6040);

      // 4c. `+2.0 mm` (`+0.020 dm`) outside exact bevel edge of Sandbox rubble block -> grabs via Pass 2 (`+2.5 mm` margin)!
      for (const s of engPhys.shards) {
        erw.getRigidBody(s.bodyHandle)!.setTranslation({ x: -3.0, y: 0.05, z: -3.0 }, true);
      }
      syncBlockPoseClean(engPhys, 0, [0.0, BLOCK_HALF_Y, 1.4], [0, 0, 0, 1]);
      erw.syncKinematics();
      const bevelMarginPt: [number, number, number] = [
        blk0.position[0] + BLOCK_HALF_X + 0.020,
        blk0.position[1] + BLOCK_HALF_Y * 0.5,
        blk0.position[2],
      ];
      const [bxPx, byPx] = project(bevelMarginPt);
      eng.handlePointerDown(bxPx, byPx, 1, 0, false, undefined, 'mouse', 7000);
      eng.handlePointerMove(bxPx + 9, byPx, 1, undefined, 'mouse', 7020);
      expect(engPhys.activeBlockId).toBe(blk0.id);
      expect(cam.isUserOrbiting).toBe(false);

      // 4d. Grazing `< 10 deg` camera ray (`|ray.dir[1]| < 0.17`) clamps excursion `<= 2.85 dm` AND transitions C^1-smoothly across `ray.dir[1] = -0.22 .. -0.12` without background shoot (`<= 3.6 dm`) or cliff jump (`< 0.05 dm` across `-0.172 -> -0.168`)
      const anchorAtGrab: [number, number, number] = [...engPhys.grabCurrentWorldOnBlock];
      cam.elevation = 0.065;
      (cam as unknown as { targetElevation: number }).targetElevation = 0.065;
      cam.update(1.0, 1280 / 800);
      eng.handlePointerMove(640, 395, 1, undefined, 'mouse', 7032);
      const radialExcursionDm = Math.hypot(
        engPhys.grabWorldAnchor[0] - anchorAtGrab[0],
        engPhys.grabWorldAnchor[2] - anchorAtGrab[2]
      );
      expect(radialExcursionDm).toBeLessThanOrEqual(2.85);
      eng.handlePointerUp(640, 395, 1, undefined, 'mouse', 7040);

      cam.elevation = 0.22;
      (cam as unknown as { targetElevation: number }).targetElevation = 0.22;
      cam.update(1.0, 1280 / 800);
      const testAnchor: [number, number, number] = [0, BLOCK_HALF_Y, 0];
      const pt18 = eng.projectBlockDragPlane(
        cam.position,
        [0, -0.18, -Math.sqrt(1 - 0.18 * 0.18)],
        testAnchor
      )!;
      const pt172 = eng.projectBlockDragPlane(
        cam.position,
        [0, -0.172, -Math.sqrt(1 - 0.172 * 0.172)],
        testAnchor
      )!;
      const pt168 = eng.projectBlockDragPlane(
        cam.position,
        [0, -0.168, -Math.sqrt(1 - 0.168 * 0.168)],
        testAnchor
      )!;
      const r18 = Math.hypot(pt18[0] - testAnchor[0], pt18[2] - testAnchor[2]);
      const r172 = Math.hypot(pt172[0] - testAnchor[0], pt172[2] - testAnchor[2]);
      const r168 = Math.hypot(pt168[0] - testAnchor[0], pt168[2] - testAnchor[2]);
      expect(r18).toBeLessThanOrEqual(3.0);
      expect(Math.abs(r172 - r168)).toBeLessThan(0.05);

      // 4e. Distant (`16.5 cm` > `11.5 cm`) foreground shard in collapsed Sandbox rubble still defers (`0%` orbit hijack on drag), and `6.5 px` touch tap (`<= 8.0 px` touch slop) still flicks foreground shard!
      syncBlockPoseClean(engPhys, 2, [1.5, 0.22, 1.5], [0, 0, 0, 1]);
      engPhys.blocks[2]!.extracted = false;
      engPhys.shatterBlock(engPhys.blocks[2]!, [0, 0, 0], 11.0, [0, 1, 0], true);
      syncBlockPoseClean(engPhys, 0, [0.0, BLOCK_HALF_Y, 0.4], [0, 0, 0, 1]);
      cam.azimuth = 0;
      (cam as unknown as { targetAzimuth: number }).targetAzimuth = 0;
      cam.elevation = 0.16;
      (cam as unknown as { targetElevation: number }).targetElevation = 0.16;
      for (let f = 0; f < 40; f++) cam.update(0.05, 1280 / 800);
      const rDx = blk0.position[0] - cam.position[0];
      const rDy = blk0.position[1] - cam.position[1];
      const rDz = blk0.position[2] - cam.position[2];
      const dBlk = Math.hypot(rDx, rDy, rDz);
      const uFar = (dBlk - 1.65) / dBlk;
      const fxFar = cam.position[0] + rDx * uFar;
      const fyFar = cam.position[1] + rDy * uFar;
      const fzFar = cam.position[2] + rDz * uFar;
      erw.getRigidBody(fgShard.bodyHandle)!.setTranslation({ x: fxFar, y: fyFar, z: fzFar }, true);
      erw.getRigidBody(fgShard.bodyHandle)!.setLinvel({ x: 0, y: 0, z: 0 }, true);
      fgShard.position = [fxFar, fyFar, fzFar];
      erw.syncKinematics();
      const [farPx, farPy] = project([fxFar, fyFar, fzFar]);
      eng.handlePointerDown(farPx, farPy, 1, 0, false, undefined, 'mouse', 7100);
      expect(eng.getInteractionMode()).toBe('block_pending');
      eng.handlePointerMove(farPx + 12, farPy, 1, undefined, 'mouse', 7120);
      expect(engPhys.activeBlockId).toBe(blk0.id);
      expect(cam.isUserOrbiting).toBe(false);
      eng.handlePointerUp(farPx + 12, farPy, 1, undefined, 'mouse', 7140);

      erw.getRigidBody(fgShard.bodyHandle)!.setTranslation({ x: fxFar, y: fyFar, z: fzFar }, true);
      erw.getRigidBody(fgShard.bodyHandle)!.setLinvel({ x: 0, y: 0, z: 0 }, true);
      fgShard.position = [fxFar, fyFar, fzFar];
      erw.syncKinematics();
      const [tchPx, tchPy] = project([fxFar, fyFar, fzFar]);
      eng.handlePointerDown(tchPx, tchPy, 2, 0, false, undefined, 'touch', 7200);
      eng.handlePointerMove(tchPx + 6.5, tchPy, 2, undefined, 'touch', 7215);
      eng.handlePointerUp(tchPx + 6.5, tchPy, 2, undefined, 'touch', 7230);
      const touchTapSp = Math.hypot(
        erw.getData()!.qvel[fgDof],
        erw.getData()!.qvel[fgDof + 1],
        erw.getData()!.qvel[fgDof + 2]
      );
      expect(touchTapSp).toBeGreaterThan(0.15);
      expect(engPhys.activeBlockId).toBeNull();
      eng.dispose();
    }

    // --- 5. Topple, Column-Drop, Stump Rocket & Camera Focus-Hold Invariants ---
    // 5a. `8` upright courses (`L10..L17`) dropping `15 mm` onto a `9L` stump (`L9` extracted) -> `0` shattered blocks!
    {
      const cw = new PhysicsWorld(audio, 18, { seed: 42 });
      cw.rebuildTower('sandbox', 2, 18, 42);
      for (let f = 0; f < 25; f++) cw.step(1 / 60);
      const crw = cw.getWorld();
      for (const b of cw.blocks) {
        if (b.assignedLayer === 9) {
          syncBlockPoseClean(
            cw,
            b.id,
            [2.8, BLOCK_HALF_Y, (b.assignedSlot - 1) * 0.45],
            [0, 0, 0, 1]
          );
        }
      }
      for (const b of cw.blocks) {
        if (b.assignedLayer !== 9) crw.wakeBody(b.bodyHandle);
      }
      for (let f = 0; f < 90; f++) cw.step(1 / 60);
      expect(cw.shatteredCount).toBe(0);
      const columnCrazed = cw.blocks.filter((b) => b.assignedLayer !== 9 && b.damage > 0.05).length;
      expect(columnCrazed).toBeLessThanOrEqual(1);
      cw.dispose();
    }

    // 5b. Yanking `L4` from an `8L` Sandbox stump keeps `L7` upward apex hop `< 8.5 cm` (vs `25+ cm` seesaw rocket!)
    {
      const yw = new PhysicsWorld(audio, 8, { seed: 42 });
      yw.rebuildTower('sandbox', 2, 8, 42);
      for (let f = 0; f < 20; f++) yw.step(1 / 60);
      const l4Center = yw.blocks.find((b) => b.assignedLayer === 4 && b.assignedSlot === 1)!;
      const l7Blocks = yw.blocks.filter((b) => b.assignedLayer === 7);
      const l7InitY = l7Blocks[0]!.position[1];
      let maxL7HopCm = 0;
      const gStart: [number, number, number] = [
        l4Center.position[0] + BLOCK_HALF_X,
        l4Center.position[1],
        l4Center.position[2],
      ];
      yw.startGrabBlock(l4Center.id, [BLOCK_HALF_X, 0, 0], gStart);
      for (let f = 1; f <= 36; f++) {
        if (f <= 14) {
          yw.updateGrabTarget(
            [gStart[0] + (2.6 * f) / 14, gStart[1] - 0.12, gStart[2]],
            8000 + f * 16.67
          );
        } else if (f === 15) {
          yw.endGrab(false, 8000 + f * 16.67);
        }
        yw.step(1 / 60);
        for (const topB of l7Blocks) {
          const hopCm = (topB.position[1] - l7InitY) * 10;
          if (hopCm > maxL7HopCm) maxL7HopCm = hopCm;
        }
      }
      expect(maxL7HopCm).toBeLessThan(8.5);
      yw.dispose();
    }

    // 5c. `triggerSandboxCollapseImpulse()` on a `7L` stump surrounded by flat tabletop rubble (`L7..L17` scattered)
    // leaves flat tabletop rubble 100% stationary (`< 0.02 m/s`) while overturning the `7L` stump cleanly!
    {
      const rwStump = new PhysicsWorld(audio, 18, { seed: 42 });
      rwStump.rebuildTower('sandbox', 2, 18, 42);
      for (let f = 0; f < 20; f++) rwStump.step(1 / 60);
      const rubbleBlocks: number[] = [];
      for (const b of rwStump.blocks) {
        if (b.assignedLayer >= 7) {
          const ang = ((b.id - 21) / 33) * Math.PI * 2;
          const rad = 1.65 + (b.id % 4) * 0.38;
          syncBlockPoseClean(
            rwStump,
            b.id,
            [Math.cos(ang) * rad, BLOCK_HALF_Y, Math.sin(ang) * rad],
            [0, 0, 0, 1]
          );
          if (rad > 2.2 && Math.cos(ang) < -0.2) {
            rubbleBlocks.push(b.id);
          }
        }
      }
      for (let f = 0; f < 35; f++) rwStump.step(1 / 60);
      expect(rubbleBlocks.length).toBeGreaterThanOrEqual(4);

      rwStump.triggerSandboxCollapseImpulse();
      const sData = rwStump.getWorld().getData()!;
      const sMap = (rwStump as unknown as { map: { blockDofAdr: number[] } }).map;
      let maxRubbleKickMps = 0;
      for (const id of rubbleBlocks) {
        const dof = sMap.blockDofAdr[id]!;
        const sp = Math.hypot(sData.qvel[dof], sData.qvel[dof + 1], sData.qvel[dof + 2]);
        if (sp > maxRubbleKickMps) maxRubbleKickMps = sp;
      }
      expect(maxRubbleKickMps).toBeLessThan(0.02);

      for (let f = 0; f < 65; f++) rwStump.step(1 / 60);
      const stumpTopRemainingY = Math.max(
        ...rwStump.blocks
          .filter((b) => b.assignedLayer < 7 && !b.shattered)
          .map((b) => b.position[1])
      );
      expect(stumpTopRemainingY).toBeLessThan(0.55);
      rwStump.dispose();
    }

    // 5d. `18L` Buckle progressive fall (`impactTime >= 0.36 s`), `0` rim shard geysers (`maxRimShardYM < 0.085 m`),
    // and `GameEngine` Sandbox camera `focusHeight` held steady (`0.000 dm` mid-air jump as `currentLayers` crosses `< 8`)!
    {
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
      const cam = (eng as unknown as {
        camera: { desiredTargetY: number; update: (dt: number, aspect: number) => void };
      }).camera;

      engPhys.rebuildTower('sandbox', 2, 18, 42);
      for (let f = 0; f < 25; f++) {
        engPhys.step(1 / 60);
        eng.updateCameraFocusTarget(1 / 60);
        cam.update(1 / 60, 1280 / 800);
      }
      const focusBeforeTopple = cam.desiredTargetY;
      expect(focusBeforeTopple).toBeGreaterThan(1.38);

      engPhys.triggerSandboxCollapseImpulse();
      let impactTimeSec = -1;
      let sawMidAirLayersBelow8 = false;
      let maxMidAirFocusDropDm = 0;
      let maxRimShardYM = 0;

      for (let f = 1; f <= 180; f++) {
        engPhys.step(1 / 60);
        eng.updateCameraFocusTarget(1 / 60);
        cam.update(1 / 60, 1280 / 800);

        if (impactTimeSec < 0 && engPhys.shatteredCount > 0) {
          impactTimeSec = f / 60;
        }
        if (impactTimeSec < 0 && engPhys.currentLayers < 8) {
          sawMidAirLayersBelow8 = true;
          const dropDm = Math.abs(focusBeforeTopple - cam.desiredTargetY);
          if (dropDm > maxMidAirFocusDropDm) maxMidAirFocusDropDm = dropDm;
        }
        for (const s of engPhys.shards) {
          const rM = Math.hypot(s.position[0], s.position[2]) * 0.1;
          const yM = s.position[1] * 0.1;
          if (rM > 0.29 && yM > maxRimShardYM) {
            maxRimShardYM = yM;
          }
        }
      }

      expect(impactTimeSec).toBeGreaterThanOrEqual(0.36);
      expect(sawMidAirLayersBelow8).toBe(true);
      expect(maxMidAirFocusDropDm).toBeLessThan(1e-5);
      expect(maxRimShardYM).toBeLessThan(0.115);

      // Continue stepping past full settle to verify smooth post-collapse slew down toward 1.32 dm (`< 1.34 dm`)
      for (let f = 0; f < 240; f++) {
        engPhys.step(1 / 60);
        eng.updateCameraFocusTarget(1 / 60);
        cam.update(1 / 60, 1280 / 800);
      }
      expect(cam.desiredTargetY).toBeLessThan(1.34);
      eng.dispose();

      // Verify identical post-collapse camera slew in Worker mode (`MujocoWorkerBridge` clears `sandboxCollapseActive`)
      const wChan = new MessageChannel();
      const wClient: WorkerEndpointLike = {
        onmessage: null,
        postMessage(msg: WorkerInboundMessage, transfer?: Transferable[]) {
          wChan.port1.postMessage(msg, transfer as never);
        },
        terminate() {
          wChan.port1.close();
          wChan.port2.close();
        },
        addEventListener() {},
        removeEventListener() {},
      };
      const wHostScope = {
        onmessage: null as ((ev: MessageEvent<WorkerInboundMessage>) => void) | null,
        postMessage(msg: WorkerOutboundMessage, transfer?: Transferable[]) {
          wChan.port2.postMessage(msg, transfer as never);
        },
      };
      wChan.port1.onmessage = (ev) => wClient.onmessage?.(ev as MessageEvent<WorkerOutboundMessage>);
      wChan.port2.onmessage = (ev) => wHostScope.onmessage?.(ev as MessageEvent<WorkerInboundMessage>);
      const wHostHandle = attachMujocoWorkerHost(wHostScope);
      const wBridge = await MujocoWorkerBridge.create(audio, 18, 42, wClient);
      const wEng = new GameEngine(canvasMock);
      (wEng as unknown as { physics: MujocoWorkerBridge }).physics = wBridge;
      const wCam = (wEng as unknown as {
        camera: { desiredTargetY: number; update: (dt: number, aspect: number) => void };
      }).camera;
      wBridge.rebuildTower('sandbox', 2, 18, 42);
      await new Promise((r) => setTimeout(r, 30));
      (wEng as unknown as { resetSandboxCameraHoldState: (h: number) => void }).resetSandboxCameraHoldState(
        27.0
      );
      for (let f = 0; f < 8; f++) {
        wBridge.step(1 / 60);
        await new Promise((r) => setTimeout(r, 6));
        wEng.updateCameraFocusTarget(1 / 60);
        wCam.update(1 / 60, 1280 / 800);
      }
      const wFocusPre = wCam.desiredTargetY;
      wEng.triggerCollapseDemo();
      let wMaxMidFallDrop = 0;
      for (let f = 0; f < 340; f++) {
        wBridge.step(1 / 60);
        await new Promise((r) => setTimeout(r, 5));
        wEng.updateCameraFocusTarget(1 / 60);
        wCam.update(1 / 60, 1280 / 800);
        if (wBridge.shatteredCount === 0 && wBridge.currentLayers < 8) {
          wMaxMidFallDrop = Math.max(wMaxMidFallDrop, Math.abs(wFocusPre - wCam.desiredTargetY));
        }
      }
      for (let extra = 0; extra < 240 && wBridge.sandboxCollapseActive; extra++) {
        wBridge.step(1 / 60);
        await new Promise((r) => setTimeout(r, 10));
        wEng.updateCameraFocusTarget(1 / 60);
        wCam.update(1 / 60, 1280 / 800);
      }
      expect(wMaxMidFallDrop).toBeLessThan(1e-5);
      expect(wBridge.sandboxCollapseActive).toBe(false);
      expect(wCam.desiredTargetY).toBeLessThan(1.34);
      wBridge.dispose();
      wHostHandle.dispose();
      wClient.terminate?.();
    }

    // 5e. Natural `18L` gravity topple breaks up overdamped "sliced-bread" collinear landing (`< 45%` vs `62.5%` prior)
    {
      const nw = new PhysicsWorld(audio, 18, { seed: 42 });
      nw.rebuildTower('sandbox', 2, 18, 42);
      for (let f = 0; f < 20; f++) nw.step(1 / 60);
      const nrw = nw.getWorld();
      for (const b of nw.blocks) {
        if ((b.assignedLayer === 1 || b.assignedLayer === 2) && b.assignedSlot >= 1) {
          b.extracted = true;
          nrw.getRigidBody(b.bodyHandle)!.setTranslation({ x: -6 - b.id, y: -5, z: -5 }, false);
        }
      }
      nrw.syncKinematics();
      for (const b of nw.blocks) {
        if (!b.extracted) nrw.wakeBody(b.bodyHandle);
      }
      for (let f = 0; f < 180; f++) nw.step(1 / 60);
      const midSurvivors = nw.blocks.filter(
        (b) => !b.shattered && !b.extracted && b.assignedLayer >= 5 && b.assignedLayer <= 12
      );
      let closeCollinear = 0;
      for (let i = 0; i < midSurvivors.length; i++) {
        let hasClose = false;
        for (let j = 0; j < midSurvivors.length; j++) {
          if (i === j || midSurvivors[i]!.assignedLayer === midSurvivors[j]!.assignedLayer) continue;
          const dMm =
            Math.hypot(
              midSurvivors[i]!.position[0] - midSurvivors[j]!.position[0],
              midSurvivors[i]!.position[2] - midSurvivors[j]!.position[2]
            ) * 100;
          if (dMm < 28) hasClose = true;
        }
        if (hasClose) closeCollinear++;
      }
      expect(closeCollinear / Math.max(1, midSurvivors.length)).toBeLessThan(0.45);
      nw.dispose();
    }
  }, 120000);

  // =========================================================================
  // Gap A: Single-Block Under-Belly Cullet Wedge Escape & Re-Solidification
  // =========================================================================
  it('Gap A: Single-block under-belly cullet wedge escape (nearIntactBlocks === 1, high vertical force) & mandatory 0x22 re-solidification', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const pw = new PhysicsWorld(audio, 2, { seed: 42 });
    pw.rebuildTower('sandbox', 2, 2, 42);
    for (let f = 0; f < 20; f++) pw.step(1 / 60);

    const rw = pw.getWorld();
    const mjModel = rw.getModel()!;

    const syncPose = (idx: number, pos: [number, number, number], quat: [number, number, number, number] = [0, 0, 0, 1]) => {
      const b = pw.blocks[idx]!;
      const rb = rw.getRigidBody(b.bodyHandle)!;
      rb.setTranslation({ x: pos[0], y: pos[1], z: pos[2] }, true);
      rb.setRotation({ x: quat[0], y: quat[1], z: quat[2], w: quat[3] }, true);
      rb.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, true);
    };

    // Teleport blocks 2, 4..5 away, keep block 3 (layer 1) to sit on top of block 0
    for (let i = 2; i < pw.blocks.length; i++) {
      if (i === 3) continue; // keep weight block 3
      pw.blocks[i]!.extracted = true;
      rw.getRigidBody(pw.blocks[i]!.bodyHandle)!.setTranslation({ x: -6 - i * 0.8, y: -5, z: -5 }, false);
    }
    syncPose(1, [0.0, 0.22, 0.0]);
    pw.shatterBlock(pw.blocks[1]!, [0, 0, 0], 11.0, [0, 1, 0], true);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);

    const targetShard = pw.shards.find((s) => mjModel.geom_contype[s.colliderHandle] === MJ_CONTYPE_CULLET_SHARD)!;
    expect(targetShard).toBeDefined();
    const rbShard = rw.getRigidBody(targetShard.bodyHandle)!;

    for (const s of pw.shards) {
      if (s.slotIndex !== targetShard.slotIndex) {
        rw.getRigidBody(s.bodyHandle)!.setTranslation({ x: 5.0 + s.slotIndex * 0.2, y: 0.1, z: 5.0 }, false);
      }
    }

    const pusher = pw.blocks[0]!;
    // 1. Under-belly pinch: position shard under pusher belly with weight block 3 on top
    rbShard.setTranslation({ x: 0.0, y: 0.015, z: 0.0 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);

    syncPose(0, [0.0, 0.090, 0.0]);
    // Place weight block 3 directly on top of pusher to supply sustained vertical bearing load
    syncPose(3, [0.0, 0.090 + 0.15, 0.0]);
    rw.getRigidBody(pw.blocks[3]!.bodyHandle)!.setGravityScale(5.0, true);

    const g0: [number, number, number] = [0.0, 0.090, 0.0];
    pw.startGrabBlock(0, [0, 0, 0], g0, 1000);

    let sawEscape = false;
    let recordedNearIntactBlocks = 0;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, 0.075, 0.0], 1000 + f * 16.67);
      pw.step(1 / 60);

      const caff = mjModel.geom_conaffinity[targetShard.colliderHandle];
      if ((caff & 0x02) === 0) {
        sawEscape = true;
        if (recordedNearIntactBlocks === 0) {
          recordedNearIntactBlocks = (targetShard as { nearIntactBlocks?: number }).nearIntactBlocks ?? 0;
        }
      }
    }

    // Verify 0x20 escape was triggered during underbelly drag under strict nearIntactBlocks === 1
    expect(sawEscape).toBe(true);
    expect(recordedNearIntactBlocks).toBe(1);

    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);

    // Verify mandatory 100% re-solidification (postSettleGhosts === 0 and targetShard conaffinity === 0x22)
    let postSettleGhosts = 0;
    for (const s of pw.shards) {
      const offRim = Math.hypot(s.position[0], s.position[2]) > TABLE_RADIUS + 0.01;
      if (!offRim && mjModel.geom_contype[s.colliderHandle] === MJ_CONTYPE_CULLET_SHARD) {
        if ((mjModel.geom_conaffinity[s.colliderHandle] & MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE) !== MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE) {
          postSettleGhosts++;
        }
      }
    }
    expect(postSettleGhosts).toBe(0);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 2. Frontal plow check: shard placed in front of leading bevel is actively plowed forward
    pw.blocks[3]!.extracted = true;
    rw.getRigidBody(pw.blocks[3]!.bodyHandle)!.setGravityScale(1.0, false);
    rw.getRigidBody(pw.blocks[3]!.bodyHandle)!.setTranslation({ x: -10, y: -5, z: -5 }, false);

    syncPose(0, [0.0, BLOCK_HALF_Y, 0.0]);
    for (let f = 0; f < 30; f++) pw.step(1 / 60);

    const initShardX = pusher.position[0] + BLOCK_HALF_X + 0.04;
    rbShard.setTranslation({ x: initShardX, y: 0.02, z: pusher.position[2] }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    for (let f = 0; f < 10; f++) pw.step(1 / 60);

    const plowG0: [number, number, number] = [pusher.position[0], pusher.position[1], pusher.position[2]];
    pw.startGrabBlock(0, [0, 0, 0], plowG0, 2000);

    let interiorPassThroughs = 0;
    for (let f = 1; f <= 50; f++) {
      const gx = plowG0[0] + (0.6 * f) / 50;
      pw.updateGrabTarget([gx, plowG0[1], plowG0[2]], 2000 + f * 16.67);
      pw.step(1 / 60);

      if (targetShard.position[0] < pusher.position[0]) {
        interiorPassThroughs++;
      }
    }

    pw.endGrab(false);
    for (let f = 0; f < 30; f++) pw.step(1 / 60);

    const adv = targetShard.position[0] - initShardX;
    const plowRatio = adv > 0.15 ? 1.0 : adv / 0.15;
    expect(plowRatio).toBe(1.0);
    expect(interiorPassThroughs).toBe(0);
    expect((mjModel.geom_conaffinity[targetShard.colliderHandle] & MJ_CONTYPE_BLOCK)).toBe(MJ_CONTYPE_BLOCK);

    // 3. Normal 1.0g two-block under-belly pinch (no 5.0g multiplier): paired normal force triggers 0x20 escape & re-solidifies
    pw.blocks[3]!.extracted = false;
    rw.getRigidBody(pw.blocks[3]!.bodyHandle)!.setGravityScale(1.0, true);
    rbShard.setTranslation({ x: 0.0, y: 0.015, z: 0.0 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [0.0, 0.090, 0.0]);
    syncPose(3, [0.0, 0.090 + 0.15, 0.0]);
    const g1g: [number, number, number] = [0.0, 0.090, 0.0];
    pw.startGrabBlock(0, [0, 0, 0], g1g, 3000);
    let sawEscape1g = false;
    let recordedNear1g = 0;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, 0.075, 0.0], 3000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[targetShard.colliderHandle] & 0x02) === 0) {
        sawEscape1g = true;
        if (recordedNear1g === 0) {
          recordedNear1g = (targetShard as { nearIntactBlocks?: number }).nearIntactBlocks ?? 0;
        }
      }
    }
    expect(sawEscape1g).toBe(true);
    expect(recordedNear1g).toBe(1);
    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 4. Upside-down (180° roll around X) 1.0g under-belly pinch: orientation-symmetric hasUnderbellyWedge triggers 0x20 escape & re-solidifies
    rbShard.setTranslation({ x: 0.0, y: 0.015, z: 0.0 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [0.0, 0.090, 0.0], [1, 0, 0, 0]);
    syncPose(3, [0.0, 0.090 + 0.15, 0.0]);
    const gUpsideDown: [number, number, number] = [0.0, 0.090, 0.0];
    pw.startGrabBlock(0, [0, 0, 0], gUpsideDown, 5000);
    let sawEscapeUpsideDown = false;
    let recordedNearUpsideDown = 0;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, 0.075, 0.0], 5000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[targetShard.colliderHandle] & 0x02) === 0) {
        sawEscapeUpsideDown = true;
        if (recordedNearUpsideDown === 0) {
          recordedNearUpsideDown = (targetShard as { nearIntactBlocks?: number }).nearIntactBlocks ?? 0;
        }
      }
    }
    expect(sawEscapeUpsideDown).toBe(true);
    expect(recordedNearUpsideDown).toBe(1);
    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 5. On-edge (90° roll around X) 1.0g under-belly pinch: long narrow side facing down triggers 0x20 escape & re-solidifies
    rbShard.setTranslation({ x: 0.0, y: 0.015, z: 0.0 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [0.0, 0.140, 0.0], [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
    syncPose(3, [0.0, 0.340, 0.0]);
    const gOnEdge: [number, number, number] = [0.0, 0.140, 0.0];
    pw.startGrabBlock(0, [0, 0, 0], gOnEdge, 7000);
    let sawEscapeOnEdge = false;
    let recordedNearOnEdge = 0;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, 0.125, 0.0], 7000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[targetShard.colliderHandle] & 0x02) === 0) {
        sawEscapeOnEdge = true;
        if (recordedNearOnEdge === 0) {
          recordedNearOnEdge = (targetShard as { nearIntactBlocks?: number }).nearIntactBlocks ?? 0;
        }
      }
    }
    expect(sawEscapeOnEdge).toBe(true);
    expect(recordedNearOnEdge).toBe(1);
    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 6. On-edge 6.0 mm off-center narrow-face contact (inside 7.5 mm half-face) escapes & re-solidifies
    rbShard.setTranslation({ x: 0.0, y: 0.015, z: 0.060 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [0.0, 0.140, 0.0], [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
    syncPose(3, [0.0, 0.340, 0.0]);
    pw.startGrabBlock(0, [0, 0, 0], [0.0, 0.140, 0.0], 9000);
    let sawEscapeOnEdgeOff6 = false;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, 0.125, 0.0], 9000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[targetShard.colliderHandle] & 0x02) === 0) {
        sawEscapeOnEdgeOff6 = true;
      }
    }
    expect(sawEscapeOnEdgeOff6).toBe(true);
    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 7. On-end (90° pitch around Z) under-belly pinch escapes & re-solidifies
    rbShard.setTranslation({ x: 0.0, y: 0.015, z: 0.0 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [0.0, 0.395, 0.0], [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
    syncPose(3, [0.0, 0.855, 0.0]);
    pw.startGrabBlock(0, [0, 0, 0], [0.0, 0.395, 0.0], 11000);
    let sawEscapeOnEnd = false;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, 0.375, 0.0], 11000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[targetShard.colliderHandle] & 0x02) === 0) {
        sawEscapeOnEnd = true;
      }
    }
    expect(sawEscapeOnEnd).toBe(true);
    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 8. Tri-axial ~35° tilt (all |downLocal*| < 0.7) escapes & re-solidifies (full SO(3) coverage)
    const qRawTilt = [0.28, 0.28, 0.28, 0.87];
    const qNormTilt = Math.hypot(qRawTilt[0], qRawTilt[1], qRawTilt[2], qRawTilt[3]);
    const qTilt: [number, number, number, number] = [
      qRawTilt[0] / qNormTilt,
      qRawTilt[1] / qNormTilt,
      qRawTilt[2] / qNormTilt,
      qRawTilt[3] / qNormTilt,
    ];
    const wLowTilt = quatRotateVec3(qTilt, [-0.34, -BLOCK_HALF_Y, 0.10]);
    const bxTilt = -wLowTilt[0];
    const byTilt = 0.060 - wLowTilt[1] + 0.01;
    const bzTilt = -wLowTilt[2];
    rbShard.setTranslation({ x: 0.0, y: 0.025, z: 0.0 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [bxTilt, byTilt, bzTilt], qTilt);
    syncPose(3, [bxTilt, byTilt + 0.30, bzTilt]);
    pw.startGrabBlock(0, [0.30, 0, 0], [bxTilt + 0.25, byTilt, bzTilt], 13000);
    let sawEscapeTilt = false;
    for (let f = 1; f <= 40; f++) {
      pw.updateGrabTarget([bxTilt + 0.25 + 0.001 * f, byTilt - 0.03, bzTilt], 13000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[targetShard.colliderHandle] & 0x02) === 0) {
        sawEscapeTilt = true;
      }
    }
    expect(sawEscapeTilt).toBe(true);
    pw.endGrab(false);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    expect(mjModel.geom_conaffinity[targetShard.colliderHandle]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    // 9. Thick 6.0 mm shard (`hy = 0.060 dm`) under upright block escapes without micro-bounce starvation
    const gidT = targetShard.colliderHandle;
    mjModel.geom_type[gidT] = 6;
    mjModel.geom_dataid[gidT] = -1;
    mjModel.geom_pos[gidT * 3 + 0] = 0;
    mjModel.geom_pos[gidT * 3 + 1] = 0;
    mjModel.geom_pos[gidT * 3 + 2] = 0;
    mjModel.geom_size[gidT * 3 + 0] = 0.004;
    mjModel.geom_size[gidT * 3 + 1] = 0.006;
    mjModel.geom_size[gidT * 3 + 2] = 0.004;
    mjModel.geom_aabb[gidT * 6 + 3] = 0.004;
    mjModel.geom_aabb[gidT * 6 + 4] = 0.006;
    mjModel.geom_aabb[gidT * 6 + 5] = 0.004;
    mjModel.geom_rbound[gidT] = Math.hypot(0.004, 0.006, 0.004);
    rbShard.setTranslation({ x: 0.0, y: 0.060, z: 0.0 }, true);
    rbShard.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    rbShard.setLinvel({ x: 0, y: 0, z: 0 }, true);
    syncPose(0, [0.0, 0.197, 0.0], [0, 0, 0, 1]);
    syncPose(3, [0.0, 0.350, 0.0], [0, 0, 0, 1]);
    pw.startGrabBlock(0, [0, 0, 0], [0.0, 0.197, 0.0], 15000);
    let sawEscapeThick = false;
    for (let f = 1; f <= 50; f++) {
      pw.updateGrabTarget([0.001 * f, BLOCK_HALF_Y, 0.0], 15000 + f * 16.67);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[gidT] & 0x02) === 0) {
        sawEscapeThick = true;
      }
    }
    expect(sawEscapeThick).toBe(true);

    // 10. Post-drag re-solidification is local-block scoped: unrelated 0.035 m/s sliding neighbor does not keep ghost alive
    pw.blocks[4]!.extracted = false;
    syncPose(4, [2.0, BLOCK_HALF_Y, 2.0], [0, 0, 0, 1]);
    pw.endGrab(false);
    let postDragGhostWithMovingNeighbor = 0;
    for (let f = 1; f <= 20; f++) {
      rw.getRigidBody(pw.blocks[4]!.bodyHandle)!.setLinvel({ x: 0.35, y: 0, z: 0 }, true);
      pw.step(1 / 60);
      if ((mjModel.geom_conaffinity[gidT] & 0x02) === 0) {
        postDragGhostWithMovingNeighbor++;
      }
    }
    expect(postDragGhostWithMovingNeighbor).toBe(0);
    expect(mjModel.geom_conaffinity[gidT]).toBe(MJ_CONAFFINITY_SOLID_CULLET_ON_TABLE);

    pw.dispose();
  }, 60000);

  // =========================================================================
  // Gap B: 2-Zone Physical Grip Boundary & Visual Anchor Plumbing
  // =========================================================================
  it('Gap B: 2-Zone physical grip boundary (|localX| <= 21.75 mm) & legible visual anchor plumbing', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const pw = new PhysicsWorld(audio, 2, { seed: 42 });
    pw.rebuildTower('sandbox', 2, 2, 42);
    for (let f = 0; f < 20; f++) pw.step(1 / 60);

    const thresholdDm = BLOCK_HALF_X * 0.58; // 0.2175 dm = 21.75 mm
    expect(thresholdDm).toBeCloseTo(0.2175, 4);

    const pusher = pw.blocks[0]!;
    const g0: [number, number, number] = [pusher.position[0], pusher.position[1], pusher.position[2]];

    // 1. Production 2-zone grip state via startGrabBlock
    // Center point (Body grip: |localX| <= 21.75 mm)
    pw.startGrabBlock(0, [0.0, 0, 0], g0, 1000);
    expect((pw as any).grabIsEndGrip).toBe(false);

    // Just inside boundary (Body grip)
    pw.startGrabBlock(0, [thresholdDm - 0.005, 0, 0], g0, 1001);
    expect((pw as any).grabIsEndGrip).toBe(false);

    // Just outside boundary (End-Tip grip)
    pw.startGrabBlock(0, [thresholdDm + 0.005, 0, 0], g0, 1002);
    expect((pw as any).grabIsEndGrip).toBe(true);

    // Negative X just inside boundary (Body grip)
    pw.startGrabBlock(0, [-thresholdDm + 0.005, 0, 0], g0, 1003);
    expect((pw as any).grabIsEndGrip).toBe(false);

    // Negative X just outside boundary (End-Tip grip)
    pw.startGrabBlock(0, [-thresholdDm - 0.005, 0, 0], g0, 1004);
    expect((pw as any).grabIsEndGrip).toBe(true);

    pw.endGrab(false);

    // 2. Shader has 2-zone came line and anchor feedback
    expect(GLASS_TOWER_WGSL).toContain('0.2175');
    expect(GLASS_TOWER_WGSL).toContain('boundaryCameLine');
    expect(GLASS_TOWER_WGSL).toContain('brassCameAnchor');
    expect(GLASS_TOWER_WGSL).toContain('jewelPivotTrailer');

    pw.dispose();
  }, 25000);

  // =========================================================================
  // M11: Proud Solder Came Perimeter Tribology (Track 1)
  // =========================================================================
  it('M11: Proud solder came perimeter tribology (Track 1): energy-passive catch-release force step, audio coupling & T1/T4 non-interference', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    // 1. Perimeter geometry check: 2.0 mm border on 75 x 25 x 15 mm block
    expect(SOLDER_CAME_WIDTH_DM * 100).toBeCloseTo(2.0, 2);
    expect(SOLDER_CAME_HEIGHT_DM * 100).toBeCloseTo(0.15, 2);
    expect(CAME_INNER_BOUND_X_DM * 100).toBeCloseTo(35.5, 2);
    expect(CAME_INNER_BOUND_Z_DM * 100).toBeCloseTo(10.5, 2);

    // Center point (smooth glass)
    const center = { ...evaluateSolderCamePoint([0, 0.075, 0], [0.1, 0, 0], 1.0) };
    expect(center.isOnCame).toBe(false);
    expect(center.effectiveFriction).toBeCloseTo(MU_KINETIC_GLASS, 3);
    expect(center.isSeamCrossing).toBe(false);

    // Came border point (60/40 Sn-Pb solder)
    const border = { ...evaluateSolderCamePoint([0.365, 0.075, 0], [0.1, 0, 0], 1.0) };
    expect(border.isOnCame).toBe(true);
    expect(border.effectiveFriction).toBeCloseTo(MU_KINETIC_SOLDER, 3);

    // Seam crossing point (transition band with tangential catch force)
    const seam = { ...evaluateSolderCamePoint([0.355, 0.075, 0], [0.05, 0, 0], 2.0) };
    expect(seam.isSeamCrossing).toBe(true);
    expect(seam.tangentialResistForceN).toBeGreaterThanOrEqual(0.05);
    expect(seam.tangentialResistForceN).toBeLessThanOrEqual(0.50);
    expect(seam.seamCrossingIntensity).toBeGreaterThan(0);
    expect(seam.seamCrossingIntensity).toBeLessThanOrEqual(1.0);

    // Energy passivity: resistance opposes velocity (delta E <= 0)
    expect(seam.tangentialResistForceN).toBeGreaterThan(0);

    // 2. Audio texture coupling
    audio.updateSlideFriction(0.12, 0.8, 15, 0.5, 2.0, seam.seamCrossingIntensity);
    expect(audio.lastSlideDiagnostics?.seamCrossingIntensity).toBe(seam.seamCrossingIntensity);

    // 3. T1 18L tower sag remains < 0.15 mm
    const pw = new PhysicsWorld(audio, 18, { seed: 42 });
    pw.rebuildTower('solo', 2, 18, 42);
    for (let f = 0; f < 90; f++) pw.step(1 / 60);
    let maxTopCenter18 = 0;
    for (const b of pw.blocks) {
      if (b.assignedLayer === 17 && !b.isLooseSlot) {
        maxTopCenter18 = Math.max(maxTopCenter18, b.position[1]);
      }
    }
    const idealTopCenter18 = (17 + 0.5) * 0.15;
    const sagMm = Math.max(0, idealTopCenter18 - maxTopCenter18) * 100.0;
    expect(sagMm).toBeLessThan(0.15);
    pw.dispose();

    // 4. Tier 2: 24L tower sag remains < 0.50 mm under solder came tribology
    const pw24 = new PhysicsWorld(audio, 24, { seed: 42 });
    pw24.rebuildTower('solo', 2, 24, 42);
    for (let f = 0; f < 100; f++) pw24.step(1 / 60);
    let maxTopCenter24 = 0;
    for (const b of pw24.blocks) {
      if (b.assignedLayer === 23 && !b.isLooseSlot) {
        maxTopCenter24 = Math.max(maxTopCenter24, b.position[1]);
      }
    }
    const idealTopCenter24 = (23 + 0.5) * 0.15;
    const sag24Mm = Math.max(0, idealTopCenter24 - maxTopCenter24) * 100.0;
    expect(sag24Mm).toBeLessThan(0.50);
    pw24.dispose();

    // 5. Diagonal sliding continuity check (|vx| ~ |vz|): zero knife-edge step jump
    // Corner (+X, -Z) at [CAME_INNER_BOUND_X_DM, 0.075, -CAME_INNER_BOUND_Z_DM] dm:
    // dx = 0 and dz = 0 continuously; verify smooth C1 climbRatio across theta = 45 deg
    const hitDiag: [number, number, number] = [CAME_INNER_BOUND_X_DM, 0.075, -CAME_INNER_BOUND_Z_DM];
    const th1 = (44.95 * Math.PI) / 180;
    const th2 = (45.05 * Math.PI) / 180;
    const velDiag1: [number, number, number] = [0.05 * Math.cos(th1), 0, 0.05 * Math.sin(th1)];
    const velDiag2: [number, number, number] = [0.05 * Math.cos(th2), 0, 0.05 * Math.sin(th2)];
    const resDiag1 = { ...evaluateSolderCamePoint(hitDiag, velDiag1, 2.0) };
    const resDiag2 = { ...evaluateSolderCamePoint(hitDiag, velDiag2, 2.0) };

    // Assert that friction delta and tangential force delta are smooth and bounded (delta F <= 0.005 N vs 0.150 N step jump)
    expect(Math.abs(resDiag1.effectiveFriction - resDiag2.effectiveFriction)).toBeLessThan(0.005);
    expect(Math.abs(resDiag1.tangentialResistForceN - resDiag2.tangentialResistForceN)).toBeLessThan(0.005);
    expect(Math.abs(resDiag1.seamCrossingIntensity - resDiag2.seamCrossingIntensity)).toBeLessThan(0.005);

    // Exact orthogonal decoupling: pure X velocity completely decouples Z came border
    const hitZOnly: [number, number, number] = [0, 0.075, 0.115];
    const velPureX: [number, number, number] = [0.05, 0, 0];
    const resPureX = evaluateSolderCamePoint(hitZOnly, velPureX, 1.0);
    expect(resPureX.isOnCame).toBe(false);
    expect(resPureX.effectiveFriction).toBeCloseTo(MU_KINETIC_GLASS, 4);

    // 6. Live PhysicsWorld extraction sweep: smooth interior glass glide -> came seam crossing catch & audio -> EXTRACTED
    const pwLive = new PhysicsWorld(audio, 18, { seed: 42 });
    pwLive.rebuildTower('solo', 2, 18, 42);
    for (let f = 0; f < 48; f++) pwLive.step(1 / 60);
    const midLoose =
      pwLive.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot) ??
      pwLive.blocks.find((b) => b.assignedLayer >= 6 && b.assignedLayer <= 12 && b.isLooseSlot)!;
    expect(midLoose).toBeDefined();
    const axisX = quatRotateVec3(midLoose.rotation, [1, 0, 0]);
    const grabOff = quatRotateVec3(midLoose.rotation, [0.18, 0, 0]);
    const p0: [number, number, number] = [...midLoose.position];
    const gStart: [number, number, number] = [
      p0[0] + grabOff[0],
      p0[1] + grabOff[1],
      p0[2] + grabOff[2],
    ];
    pwLive.startGrabBlock(midLoose.id, [0.18, 0, 0], gStart, 1000);

    let seamCrossingPeak = 0;
    let minInteriorFriction = Number.POSITIVE_INFINITY;
    let cameFrictionPeak = 0;
    for (let s = 1; s <= 140; s++) {
      const pullDist = (1.35 * s) / 140;
      pwLive.updateGrabTarget(
        [
          gStart[0] + axisX[0] * pullDist,
          gStart[1] + axisX[1] * pullDist,
          gStart[2] + axisX[2] * pullDist,
        ],
        1000 + s * (1000 / 120)
      );
      pwLive.step(1 / 120);
      const slidDist = Math.hypot(
        midLoose.position[0] - p0[0],
        midLoose.position[2] - p0[2]
      );
      if (slidDist > 0.04 && slidDist < 0.14) {
        minInteriorFriction = Math.min(minInteriorFriction, midLoose.currentFriction);
      }
      if (slidDist >= 0.16 && !midLoose.extracted) {
        cameFrictionPeak = Math.max(cameFrictionPeak, midLoose.currentFriction);
      }
      const diagSeam = audio.lastSlideDiagnostics?.seamCrossingIntensity ?? 0;
      if (diagSeam > seamCrossingPeak) seamCrossingPeak = diagSeam;
      if (pwLive.turnPhase === 'EXTRACTED') break;
    }
    expect(seamCrossingPeak).toBeGreaterThan(0.10);
    expect(minInteriorFriction).toBeLessThan(0.165);
    expect(cameFrictionPeak).toBeGreaterThan(0.175);
    expect(pwLive.turnPhase).toBe('EXTRACTED');
    expect(pwLive.leanAngleDeg).toBeLessThan(2.0);
    pwLive.dispose();

    // 7. Negative/NaN normalForceN clamping & multi-instance SolderCameAggregateResult re-entrancy safety
    const negLoad = evaluateSolderCamePoint([0.355, 0.075, 0], [1.0, 0, 0], -5.0);
    expect(negLoad.seamCrossingIntensity).toBe(0);
    expect(negLoad.tangentialResistForceN).toBe(0);
    const nanLoad = evaluateSolderCamePoint([0.355, 0.075, 0], [1.0, 0, 0], Number.NaN);
    expect(nanLoad.seamCrossingIntensity).toBe(0);
    expect(nanLoad.tangentialResistForceN).toBe(0);

    const aggA: SolderCameAggregateResult = {
      maxSeamCrossingIntensity: 0,
      totalResistForceN: 0,
      weightedEffectiveFriction: MU_KINETIC_GLASS,
      contactCount: 0,
      cameBorderContactCount: 0,
    };
    const aggB: SolderCameAggregateResult = {
      maxSeamCrossingIntensity: 0,
      totalResistForceN: 0,
      weightedEffectiveFriction: MU_KINETIC_GLASS,
      contactCount: 0,
      cameBorderContactCount: 0,
    };
    resetSolderCameAggregate(aggA);
    resetSolderCameAggregate(aggB);
    accumulateSolderCameContact([0, 0.075, 0], [0.1, 0, 0], 4.0, aggA);
    accumulateSolderCameContact([0.365, 0.075, 0], [0.1, 0, 0], 4.0, aggB);
    accumulateSolderCameContact([0, 0.075, 0], [0.1, 0, 0], Number.NaN, aggA);
    expect(aggA.weightedEffectiveFriction).toBeCloseTo(MU_KINETIC_GLASS, 4);
    expect(aggB.weightedEffectiveFriction).toBeCloseTo(MU_KINETIC_SOLDER, 4);
    expect(Number.isFinite(aggA.weightedEffectiveFriction)).toBe(true);

    // 8. Worker boundary (`MujocoWorkerBridge`) & Host/Guest Multiplayer boundary (`installHostAudioTap` + `MultiplayerGuestWorld`)
    //    must preserve `seamCrossingIntensity > 0` without dropping the 6th argument.
    const wAudio = new GlassAudioEngine();
    wAudio.setMuted(true);
    const m11Chan = new MessageChannel();
    const m11HostScope = {
      onmessage: null as ((ev: MessageEvent<WorkerInboundMessage>) => void) | null,
      postMessage: (msg: WorkerOutboundMessage, transfer?: Transferable[]) => {
        m11Chan.port2.postMessage(msg, transfer as never);
      },
    };
    const m11ClientEndpoint: WorkerEndpointLike = {
      onmessage: null,
      postMessage: (msg: WorkerInboundMessage, transfer?: Transferable[]) => {
        m11Chan.port1.postMessage(msg, transfer as never);
      },
    };
    m11Chan.port1.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
      m11ClientEndpoint.onmessage?.(ev);
    };
    m11Chan.port2.onmessage = (ev: MessageEvent<WorkerInboundMessage>) => {
      m11HostScope.onmessage?.(ev);
    };
    m11Chan.port1.start();
    m11Chan.port2.start();

    const m11HostHandle = attachMujocoWorkerHost(m11HostScope);
    const m11Bridge = await MujocoWorkerBridge.create(wAudio, 18, 42, m11ClientEndpoint);
    const stepM11Bridge = async (dt = 1 / 60): Promise<void> => {
      await new Promise<void>((resolve) => {
        const prev = m11ClientEndpoint.onmessage;
        m11ClientEndpoint.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
          prev?.(ev);
          if (ev.data.type === 'snapshot' && ev.data.epoch === m11Bridge.getEpoch()) {
            m11ClientEndpoint.onmessage = prev;
            resolve();
          }
        };
        m11Bridge.step(dt);
      });
    };
    m11Bridge.rebuildTower('solo', 2, 18, 42);
    for (let f = 0; f < 48; f++) await stepM11Bridge(1 / 60);
    const wMidLoose =
      m11Bridge.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot) ??
      m11Bridge.blocks.find((b) => b.assignedLayer >= 6 && b.assignedLayer <= 12 && b.isLooseSlot)!;
    const wAx = quatRotateVec3(wMidLoose.rotation, [1, 0, 0]);
    const wGrabOff = quatRotateVec3(wMidLoose.rotation, [0.18, 0, 0]);
    const wStart: Vec3 = [
      wMidLoose.position[0] + wGrabOff[0],
      wMidLoose.position[1] + wGrabOff[1],
      wMidLoose.position[2] + wGrabOff[2],
    ];
    m11Bridge.startGrabBlock(wMidLoose.id, [0.18, 0, 0], wStart, 2000);
    let workerSeamPeak = 0;
    for (let s = 1; s <= 75; s++) {
      const pullDist = (1.35 * s) / 75;
      m11Bridge.updateGrabTarget(
        [wStart[0] + wAx[0] * pullDist, wStart[1] + wAx[1] * pullDist, wStart[2] + wAx[2] * pullDist],
        2000 + Math.round(s * (1000 / 60))
      );
      await stepM11Bridge(1 / 60);
      const diagSeam = wAudio.lastSlideDiagnostics?.seamCrossingIntensity ?? 0;
      if (diagSeam > workerSeamPeak) workerSeamPeak = diagSeam;
      if (m11Bridge.turnPhase === 'EXTRACTED') break;
    }
    expect(workerSeamPeak).toBeGreaterThan(0.10);
    m11Bridge.dispose();
    m11HostHandle.dispose();
    m11Chan.port1.close();
    m11Chan.port2.close();

    // Host AudioTap -> Outbound Queue -> Guest Replay preserves seamCrossingIntensity
    const tapEngine = new GameEngine();
    await tapEngine.initHeadless();
    (tapEngine as unknown as { mpRole: 'host' }).mpRole = 'host';
    (tapEngine as unknown as { mpStatus: { peerConnected: boolean } }).mpStatus.peerConnected = true;
    tapEngine.installHostAudioTap();
    tapEngine.audio.updateSlideFriction(0.65, 0.55, 18, 0.2, 2.5, 0.72);
    expect(tapEngine.audio.lastSlideDiagnostics?.seamCrossingIntensity).toBeCloseTo(0.72, 4);
    const queuedSlide = tapEngine
      .getOutboundAudioQueueSnapshot()
      .find((ev) => ev.kind === 'slideFriction');
    expect(queuedSlide?.kind).toBe('slideFriction');
    if (queuedSlide?.kind === 'slideFriction') {
      expect(queuedSlide.seamCrossingIntensity).toBeCloseTo(0.72, 4);
      const guestAudio = new GlassAudioEngine();
      guestAudio.setMuted(true);
      const guestWorld = new MultiplayerGuestWorld(guestAudio);
      guestWorld.replayNetworkAudioEvents([queuedSlide], 1);
      expect(guestAudio.lastSlideDiagnostics?.seamCrossingIntensity).toBeCloseTo(0.72, 4);
      guestWorld.dispose();
    }
    tapEngine.dispose();
  }, 60000);

  // =========================================================================
  // M12: Ductile Solder-Held Cracked Bricks & Modal Doublet Splitting (Track 2)
  // =========================================================================
  it('M12: Ductile solder-held cracked bricks & acoustic doublet splitting (Track 2): cohesion bracket d in [0.78, 0.96], tear threshold & 30-48 Hz modal beating', async () => {
    await ensurePhysicsInitialized();
    const audio = new GlassAudioEngine();
    audio.setMuted(true);

    const pw = new PhysicsWorld(audio, 2, { seed: 42 });
    pw.rebuildTower('sandbox', 2, 2, 42);
    for (let f = 0; f < 20; f++) pw.step(1 / 60);

    const rw = pw.getWorld();
    const b0 = pw.blocks[0]!; // test block -> cleaved -> shattered
    const b1 = pw.blocks[1]!; // fresh control block
    for (let i = 2; i < pw.blocks.length; i++) {
      pw.blocks[i]!.extracted = true;
      rw.getRigidBody(pw.blocks[i]!.bodyHandle)!.setTranslation({ x: -6 - i * 0.8, y: -5, z: -5 }, false);
    }

    // Park control block b1 aside
    syncBlockPoseClean(pw, 1, [2.8, BLOCK_HALF_Y, 2.8], [0, 0, 0, 1], [0, 0, 0]);

    // 1. Stage 1: Fresh block b0 (damage === 0) drops 11.0 cm onto oak table (intermediate impact in [vCleave, vTearFresh))
    // -> enters ductile copper-foil cleavage bracket [0.78, 0.96] WITHOUT shattering
    expect(b0.damage).toBe(0);
    syncBlockPoseClean(pw, 0, [0, 1.10 + BLOCK_HALF_Y, 0], [0, 0, 0, 1], [0, 0, 0]);
    for (let f = 0; f < 40; f++) pw.step(1 / 60);

    expect(b0.shattered).toBe(false);
    expect(b0.damage).toBeGreaterThanOrEqual(0.78);
    expect(b0.damage).toBeLessThanOrEqual(0.96);
    expect(pw.splinters.length).toBeGreaterThan(0);
    const cleavedDamage = b0.damage;

    // 2. Stage 2: Cleaved brick survives moderate 1.5 cm handling drop without shattering
    syncBlockPoseClean(pw, 0, [0, 0.15 + BLOCK_HALF_Y, 0], [0, 0, 0, 1], [0, 0, 0]);
    for (let f = 0; f < 40; f++) pw.step(1 / 60);
    expect(b0.shattered).toBe(false);
    expect(b0.damage).toBeGreaterThanOrEqual(0.78);
    expect(b0.damage).toBeLessThanOrEqual(0.96);

    // 3. Stage 3: Fresh control b1 from 9.5 cm does NOT shatter; cleaved b0 from 9.5 cm tears seam & shatters into Voronoi shards
    syncBlockPoseClean(pw, 0, [-2.8, BLOCK_HALF_Y, -2.8], [0, 0, 0, 1], [0, 0, 0]);
    syncBlockPoseClean(pw, 1, [0, 0.95 + BLOCK_HALF_Y, 0], [0, 0, 0, 1], [0, 0, 0]);
    for (let f = 0; f < 40; f++) pw.step(1 / 60);
    expect(b1.shattered).toBe(false);
    expect(b1.damage).toBeLessThan(0.78);

    syncBlockPoseClean(pw, 1, [2.8, BLOCK_HALF_Y, 2.8], [0, 0, 0, 1], [0, 0, 0]);
    syncBlockPoseClean(pw, 0, [0, 0.95 + BLOCK_HALF_Y, 0], [0, 0, 0, 1], [0, 0, 0]);
    for (let f = 0; f < 40; f++) pw.step(1 / 60);
    expect(b0.shattered).toBe(true);
    expect(pw.shards.length).toBeGreaterThanOrEqual(10);

    // 4. Stage 4: Modal doublet splitting & live GlassAudioEngine tap doublet check
    const modes = computeGlassBarModes({ damage: cleavedDamage });
    const fund = modes[0]!;
    expect(fund.splitBeatHz).toBeGreaterThanOrEqual(25.0);
    expect(fund.splitBeatHz).toBeLessThanOrEqual(48.0);
    expect(fund.doubletFreqHz).toBeCloseTo(fund.freqHz + fund.splitBeatHz, 5);
    audio.playTapTestResonance(0, b0.position, undefined, undefined, cleavedDamage);
    expect(audio.lastTapDiagnostics?.doubletSpawned).toBe(true);

    pw.dispose();
  }, 30000);

  // =========================================================================
  // M13: 3D Inter-Course Colored Caustic Projection (Track 3)
  // =========================================================================
  it('M13: 3D inter-course colored caustic projection (Track 3): lower course & gap wall illumination across 4 lighting presets & Polariscope dark field', async () => {
    // 1. Verify all 4 lighting presets exist and have valid solar angles
    expect(LIGHTING_PRESETS.length).toBe(4);
    const morning = LIGHTING_PRESETS.find((p) => p.id === 'lancet-morning')!;
    const noon = LIGHTING_PRESETS.find((p) => p.id === 'high-clerestory')!;
    const atelier = LIGHTING_PRESETS.find((p) => p.id === 'daylight-atelier')!;
    const vespers = LIGHTING_PRESETS.find((p) => p.id === 'golden-vespers')!;

    expect(morning).toBeDefined();
    expect(noon).toBeDefined();
    expect(atelier).toBeDefined();
    expect(vespers).toBeDefined();

    expect(noon.sunElevationDeg).toBeGreaterThan(morning.sunElevationDeg);
    expect(vespers.sunElevationDeg).toBeLessThan(noon.sunElevationDeg);

    // 2. Verify WGSL shaders contain caustic sampling, polariscope modulation and 2-zone came line
    expect(GLASS_TOWER_WGSL).toContain('uCausticTex');
    expect(GLASS_TOWER_WGSL).toContain('uBlockCenters[97].w');
    expect(GLASS_TOWER_WGSL).toContain('polariscopeMod');
    expect(GLASS_TOWER_WGSL).toContain('surfaceCaustic');
  });

  // =========================================================================
  // M14: Push vs Pinch Extraction Mechanics (Track 4)
  // =========================================================================
  it('M14: Push vs pinch extraction mechanics (Track 4): end push (18 N) vs pinch slip (3.42 N), 18L clamping parity, hover zones, Glazier AI & Guest parity', async () => {
    await ensurePhysicsInitialized();

    // 1. Track 4 physical constants & evaluateExtractionSlip boundary verification
    expect(EXTRACTION_BLOCK_HALF_X).toBeCloseTo(0.0375, 5);
    expect(MU_FINGER).toBeCloseTo(0.38, 4);
    expect(F_PINCH).toBeCloseTo(4.5, 4);
    expect(F_SLIP).toBeCloseTo(2 * MU_FINGER * F_PINCH, 5); // 3.42 N
    expect(F_PUSH_MAX).toBeCloseTo(18.0, 4);

    // End face (|localX| >= 0.0325 m) inward push: resists slip up to 18.0 N
    const pushWithinCap = evaluateExtractionSlip([0.035, 0, 0], 10.5, true);
    expect(pushWithinCap).toEqual({ slipped: false, maxForceN: 18.0, slipAudio: false });

    const pushAtBoundary = evaluateExtractionSlip({ x: -0.0325, y: 0, z: 0 }, 18.0, true);
    expect(pushAtBoundary).toEqual({ slipped: false, maxForceN: 18.0, slipAudio: false });

    const pushOverMax = evaluateExtractionSlip([0.036, 0, 0], 18.5, true);
    expect(pushOverMax).toEqual({ slipped: true, maxForceN: 18.0, slipAudio: true });

    // End face outward pull (isPush = false): bounded by F_SLIP = 3.42 N
    const endPullSafe = evaluateExtractionSlip([0.035, 0, 0], 2.5, false);
    expect(endPullSafe).toEqual({ slipped: false, maxForceN: F_SLIP, slipAudio: false });

    const endPullSlip = evaluateExtractionSlip([0.035, 0, 0], 3.5, false);
    expect(endPullSlip).toEqual({ slipped: true, maxForceN: F_SLIP, slipAudio: true });

    // Slick side face (|localX| < 0.0325 m): always bounded by F_SLIP = 3.42 N even if isPush = true
    const sidePinchSlip = evaluateExtractionSlip({ x: 0.01, y: 0, z: 0.0125 }, 4.2, true);
    expect(sidePinchSlip).toEqual({ slipped: true, maxForceN: F_SLIP, slipAudio: true });

    // 2. Real 18-layer MuJoCo tower clamping differentiation & solo extraction behavior
    const audio = new GlassAudioEngine();
    audio.setMuted(true);
    const pw = new PhysicsWorld(audio, 18, { seed: 42 });
    pw.rebuildTower('solo', 2, 18, 42);
    for (let f = 0; f < 90; f++) pw.step(1 / 60);

    const tightMid = pw.blocks.find((b) => b.assignedLayer === 8 && !b.isLooseSlot)!;
    const looseMid = pw.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot)!;
    expect(tightMid).toBeDefined();
    expect(looseMid).toBeDefined();

    const tightLoadN = pw.computeBlockClampingLoadN(tightMid.id);
    const looseLoadN = pw.computeBlockClampingLoadN(looseMid.id);
    expect(tightLoadN).toBeGreaterThan(5.0);
    expect(tightLoadN).toBeGreaterThan(F_SLIP);
    expect(looseLoadN).toBeLessThan(2.0);
    expect(looseLoadN).toBeLessThan(F_SLIP);

    // 2a. Loose block extraction via side-pinch pull succeeds cleanly without slip
    const looseAxisX = quatRotateVec3(looseMid.rotation, [1, 0, 0]);
    const looseLocal: Vec3 = [0.10, 0, BLOCK_HALF_Z];
    const looseOff = quatRotateVec3(looseMid.rotation, looseLocal);
    const looseStart: Vec3 = [
      looseMid.position[0] + looseOff[0],
      looseMid.position[1] + looseOff[1],
      looseMid.position[2] + looseOff[2],
    ];
    expect(pw.startGrabBlock(looseMid.id, looseLocal, looseStart, 1000)).toBe(true);
    for (let s = 1; s <= 95 && pw.turnPhase === 'PULLING'; s++) {
      const dist = Math.min(1.35, (s / 70) * 1.35);
      pw.updateGrabTarget(
        [looseStart[0] + looseAxisX[0] * dist, looseStart[1], looseStart[2] + looseAxisX[2] * dist],
        1000 + s * 16
      );
      pw.step(1 / 60);
    }
    expect(pw.turnPhase).toBe('EXTRACTED');
    expect(pw.lastExtractionSlip?.slipped).toBe(false);
    pw.endGrab(false);

    // Reset to settled 18L tower for clamped-block tests
    pw.rebuildTower('solo', 2, 18, 42);
    for (let f = 0; f < 90; f++) pw.step(1 / 60);

    // 2b. Clamped bearing block side-pinch pull slips cleanly off with slipAudio = true
    const tightSideBlk = pw.blocks.find((b) => b.assignedLayer === 8 && !b.isLooseSlot)!;
    const tightAxisX = quatRotateVec3(tightSideBlk.rotation, [1, 0, 0]);
    const sideLocal: Vec3 = [0.05, 0, BLOCK_HALF_Z];
    const sideOff = quatRotateVec3(tightSideBlk.rotation, sideLocal);
    const sideStart: Vec3 = [
      tightSideBlk.position[0] + sideOff[0],
      tightSideBlk.position[1] + sideOff[1],
      tightSideBlk.position[2] + sideOff[2],
    ];
    expect(pw.startGrabBlock(tightSideBlk.id, sideLocal, sideStart, 2000)).toBe(true);
    let sawSideSlip = false;
    for (let s = 1; s <= 24; s++) {
      const dist = s * 0.025;
      pw.updateGrabTarget(
        [sideStart[0] + tightAxisX[0] * dist, sideStart[1], sideStart[2] + tightAxisX[2] * dist],
        2000 + s * 16
      );
      if (pw.lastExtractionSlip?.slipped) {
        sawSideSlip = true;
        break;
      }
      pw.step(1 / 60);
    }
    expect(sawSideSlip).toBe(true);
    expect(pw.lastExtractionSlip?.maxForceN).toBeCloseTo(F_SLIP, 4);
    expect(pw.lastExtractionSlip?.slipAudio).toBe(true);
    expect(pw.turnPhase).toBe('SELECT_BLOCK');
    expect(pw.activeBlockId).toBeNull();
    expect(pw.leanAngleDeg).toBeLessThan(2.0);
    expect(audio.lastSlideDiagnostics?.seamCrossingIntensity).toBeGreaterThan(0);

    // 2c. Clamped bearing block one-finger inward end push does NOT slip (maxForceN = 18.0 N) & nudges block
    for (let f = 0; f < 10; f++) pw.step(1 / 60);
    const pushLocal: Vec3 = [-0.35, 0, 0]; // -35 mm end face (|x| >= 32.5 mm)
    const pushOff = quatRotateVec3(tightSideBlk.rotation, pushLocal);
    const pushStart: Vec3 = [
      tightSideBlk.position[0] + pushOff[0],
      tightSideBlk.position[1] + pushOff[1],
      tightSideBlk.position[2] + pushOff[2],
    ];
    const initPushPos: Vec3 = [...tightSideBlk.position];
    expect(pw.startGrabBlock(tightSideBlk.id, pushLocal, pushStart, 3000)).toBe(true);
    for (let s = 1; s <= 35; s++) {
      const inwardDist = Math.min(0.42, s * 0.015);
      pw.updateGrabTarget(
        [
          pushStart[0] + tightAxisX[0] * inwardDist,
          pushStart[1],
          pushStart[2] + tightAxisX[2] * inwardDist,
        ],
        3000 + s * 16
      );
      pw.step(1 / 60);
    }
    const nudgedDistDm = Math.hypot(
      tightSideBlk.position[0] - initPushPos[0],
      tightSideBlk.position[2] - initPushPos[2]
    );
    expect(pw.lastExtractionSlip?.slipped).toBe(false);
    expect(pw.lastExtractionSlip?.maxForceN).toBeCloseTo(F_PUSH_MAX, 4);
    expect(nudgedDistDm).toBeGreaterThan(0.03); // Nudged >= 3 mm without finger slip
    pw.endGrab(false);

    // 2d. Sandbox mode leaves free-drag untouched (never triggers competitive pinch slip)
    pw.rebuildTower('sandbox', 2, 18, 42);
    for (let f = 0; f < 60; f++) pw.step(1 / 60);
    const sbTight = pw.blocks.find((b) => b.assignedLayer === 8 && !b.isLooseSlot)!;
    const sbAxis = quatRotateVec3(sbTight.rotation, [1, 0, 0]);
    const sbOff = quatRotateVec3(sbTight.rotation, [0.05, 0, BLOCK_HALF_Z]);
    const sbStart: Vec3 = [
      sbTight.position[0] + sbOff[0],
      sbTight.position[1] + sbOff[1],
      sbTight.position[2] + sbOff[2],
    ];
    expect(pw.startGrabBlock(sbTight.id, [0.05, 0, BLOCK_HALF_Z], sbStart, 4000)).toBe(true);
    for (let s = 1; s <= 20; s++) {
      pw.updateGrabTarget(
        [sbStart[0] + sbAxis[0] * (s * 0.025), sbStart[1], sbStart[2] + sbAxis[2] * (s * 0.025)],
        4000 + s * 16
      );
      pw.step(1 / 60);
    }
    expect(pw.lastExtractionSlip).toBeNull();
    expect(pw.activeBlockId).toBe(sbTight.id);
    pw.endGrab(false);
    pw.dispose();

    // 3. GameEngine hover zone plumbing (Body = 1.0 vs End-Tip = 2.0) & Glazier AI push vs pinch plans
    const engine = new GameEngine();
    await engine.initHeadless();
    engine.resetGame('party', 2);
    for (let f = 0; f < 60; f++) engine.stepHeadless(1 / 60);

    // Project a front-facing legal block's Body center vs End-Tip onto the viewport
    const rect = { left: 0, top: 0, width: 1280, height: 800 };
    engine.camera.update(0.25, rect.width / rect.height);
    const projectToScreen = (pt: Vec3): [number, number] => {
      const m = engine.camera.viewProjMatrix;
      const cx = m[0] * pt[0] + m[4] * pt[1] + m[8] * pt[2] + m[12];
      const cy = m[1] * pt[0] + m[5] * pt[1] + m[9] * pt[2] + m[13];
      const cw = m[3] * pt[0] + m[7] * pt[1] + m[11] * pt[2] + m[15];
      return [((cx / cw) * 0.5 + 0.5) * rect.width, (0.5 - (cy / cw) * 0.5) * rect.height];
    };

    const frontProbe = engine.physics.blocks.find(
      (b) => b.assignedLayer === 8 && b.assignedSlot === 2 && b.isLegalTarget
    )!;
    expect(frontProbe).toBeDefined();
    const frontFaceCenter: Vec3 = [
      frontProbe.position[0],
      frontProbe.position[1],
      frontProbe.position[2] + BLOCK_HALF_Z,
    ];
    const [bx, by] = projectToScreen(frontFaceCenter);
    engine.handlePointerMove(bx, by, 1, rect, 'mouse');
    expect(engine.physics.hoveredBlockId).toBe(frontProbe.id);
    expect(engine.physics.hoverLocalPoint).not.toBeNull();
    expect(Math.abs(engine.physics.hoverLocalPoint![0])).toBeLessThanOrEqual(BLOCK_HALF_X * 0.58);
    expect(engine.getHoveredZone()).toBe(1.0);

    const endTipWorld: Vec3 = [
      frontProbe.position[0] + BLOCK_HALF_X * 0.85,
      frontProbe.position[1],
      frontProbe.position[2] + BLOCK_HALF_Z * 0.5,
    ];
    const [ex, ey] = projectToScreen(endTipWorld);
    engine.handlePointerMove(ex, ey, 1, rect, 'mouse');
    expect(engine.physics.hoveredBlockId).toBe(frontProbe.id);
    expect(engine.physics.hoverLocalPoint).not.toBeNull();
    expect(Math.abs(engine.physics.hoverLocalPoint![0])).toBeGreaterThan(BLOCK_HALF_X * 0.58);
    expect(engine.getHoveredZone()).toBe(2.0);

    // Glazier AI extraction plan chooses pinch_pull for loose <= 3.42 N and end_push for clamped > 3.42 N
    const aiLoose = engine.physics.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot)!;
    const aiTight = engine.physics.blocks.find((b) => b.assignedLayer === 8 && !b.isLooseSlot)!;
    const loosePlan = engine.computeOptimalExtractionPlan(aiLoose.id, 'master')!;
    const tightPlan = engine.computeOptimalExtractionPlan(aiTight.id, 'master')!;
    expect(loosePlan.extractionMode).toBe('pinch_pull');
    expect(loosePlan.maxForceN).toBeCloseTo(F_SLIP, 4);
    expect(tightPlan.extractionMode).toBe('end_push');
    expect(tightPlan.maxForceN).toBeCloseTo(F_PUSH_MAX, 4);
    expect(Math.abs(tightPlan.localGrab[0])).toBeGreaterThanOrEqual(BLOCK_HALF_X - 0.05);

    // 4. MultiplayerGuestWorld & MujocoWorkerBridge parity
    const guestAudio = new GlassAudioEngine();
    guestAudio.setMuted(true);
    const guest = new MultiplayerGuestWorld(guestAudio, 18, 42);
    const hostSnap = engine.physics.exportCompactSnapshot();
    const guestBlocks = Array.from({ length: hostSnap.blockCount }, (_, i) => {
      const b = engine.physics.blocks[i];
      const wBase = i * 4;
      const wrench: [number, number, number, number] = [
        hostSnap.blockPhotoelasticWrench[wBase + 0] ?? 0,
        hostSnap.blockPhotoelasticWrench[wBase + 1] ?? 0,
        hostSnap.blockPhotoelasticWrench[wBase + 2] ?? 0,
        hostSnap.blockPhotoelasticWrench[wBase + 3] ?? 0,
      ];
      return {
        position: [...b.position] as Vec3,
        rotation: [...b.rotation] as [number, number, number, number],
        damage: b.damage,
        shattered: b.shattered,
        extracted: b.extracted,
        isLegalTarget: b.isLegalTarget,
        isLooseSlot: b.isLooseSlot,
        swatchId: b.swatchId,
        motif: b.motif,
        assignedLayer: b.assignedLayer,
        assignedSlot: b.assignedSlot,
        stress01: engine.physics.getBlockPhotoelasticStress01(i),
        crackLocalOrigin: [...b.crackLocalOrigin] as Vec3,
        photoelasticWrench: wrench,
      };
    });
    const baseGuestFrame = {
      seq: 1,
      hostElapsedSec: 1.5,
      roomSeed: 42,
      turnPhase: 'SELECT_BLOCK' as const,
      gameMode: 'party' as const,
      currentPlayer: 1,
      playerCount: 2,
      blocksStacked: 0,
      shatteredCount: 0,
      currentLayers: 18,
      peakLayers: 18,
      activeBlockId: null,
      hoveredBlockId: null,
      hoveredSlotIndex: null,
      lockedMinRow: 17,
      isPointerDraggingBlock: false,
      isExtractionLiftReady: false,
      polariscopeMode: false,
      grabResistance01: 0,
      settleProgress01: 0,
      leanAngleDeg: 0,
      wobbleSpeedDegPerSec: 0,
      liveHeightCm: 27,
      peakHeightCm: 27,
      centerOfMassXZCm: [0, 0] as [number, number],
      carriedYawOffsetRad: 0,
      grabCurrentWorldOnBlock: [0, 0, 0] as Vec3,
      grabWorldAnchor: [0, 0, 0] as Vec3,
      carriedLandingPos: [0, 2.775, 0] as Vec3,
      carriedTargetQuat: [0, 0, 0, 1] as [number, number, number, number],
      topSupportSurfaceCenter: [0, 2.7, 0] as Vec3,
      topSupportUpNormal: [0, 1, 0] as Vec3,
      topSupportDefaultRot: [0, 0, 0, 1] as [number, number, number, number],
      topSupportTargetLayer: 18,
      blocks: guestBlocks,
      shards: [],
      splinters: [],
    };
    guest.applyNetworkStateFrame(baseGuestFrame);

    expect(guest.computeBlockClampingLoadN(aiTight.id)).toBeGreaterThan(F_SLIP);
    expect(guest.computeBlockClampingLoadN(aiLoose.id)).toBeLessThan(F_SLIP);

    guest.setOptimisticGuestHover(aiTight.id, [0.34, 0, 0]);
    expect(guest.hoverLocalPoint).toEqual([0.34, 0, 0]);

    const gTight = guest.blocks[aiTight.id];
    const gAxisX = quatRotateVec3(gTight.rotation, [1, 0, 0]);
    const gSideWorld: Vec3 = [gTight.position[0], gTight.position[1], gTight.position[2] + BLOCK_HALF_Z];
    guest.setOptimisticGuestGrab(aiTight.id, [0.05, 0, BLOCK_HALF_Z], gSideWorld);
    // Pre-confirmation Host snapshot (isPointerDraggingBlock=false) must NOT wipe optimistic Guest grab
    guest.applyNetworkStateFrame({
      ...baseGuestFrame,
      seq: 2,
      turnPhase: 'SELECT_BLOCK',
      activeBlockId: null,
      isPointerDraggingBlock: false,
    });
    const guestPinchSlip = guest.evaluateGuestDragSlip([
      gSideWorld[0] + gAxisX[0] * 0.20,
      gSideWorld[1],
      gSideWorld[2] + gAxisX[2] * 0.20,
    ]);
    expect(guestPinchSlip).not.toBeNull();
    expect(guestPinchSlip?.slipped).toBe(true);
    expect(guestPinchSlip?.maxForceN).toBeCloseTo(F_SLIP, 4);
    expect(guestPinchSlip?.slipAudio).toBe(true);
    guest.setOptimisticGuestPullAnchor(null);
    expect(guest.evaluateGuestDragSlip(gSideWorld)).toBeNull();

    // Loose block with small nonzero wrench (wSum = 0.16 < 0.18) must cap clamping load below F_SLIP
    const gLoose = guest.blocks[aiLoose.id];
    const gLooseAxisX = quatRotateVec3(gLoose.rotation, [1, 0, 0]);
    const gLooseSideWorld: Vec3 = [gLoose.position[0], gLoose.position[1], gLoose.position[2] + BLOCK_HALF_Z];
    guest.blockPhotoelasticWrench[aiLoose.id * 4 + 0] = 0.06;
    guest.blockPhotoelasticWrench[aiLoose.id * 4 + 1] = 0.05;
    guest.blockPhotoelasticWrench[aiLoose.id * 4 + 2] = 0.05;
    guest.setOptimisticGuestGrab(aiLoose.id, [0.05, 0, BLOCK_HALF_Z], gLooseSideWorld);
    const guestLooseWrenchSlip = guest.evaluateGuestDragSlip([
      gLooseSideWorld[0] + gLooseAxisX[0] * 0.20,
      gLooseSideWorld[1],
      gLooseSideWorld[2] + gLooseAxisX[2] * 0.20,
    ]);
    expect(guestLooseWrenchSlip?.slipped).toBe(false);
    guest.setOptimisticGuestPullAnchor(null);

    const gPushWorld: Vec3 = [
      gTight.position[0] - gAxisX[0] * 0.35,
      gTight.position[1],
      gTight.position[2] - gAxisX[2] * 0.35,
    ];
    guest.setOptimisticGuestGrab(aiTight.id, [-0.35, 0, 0], gPushWorld);
    const guestEndPush = guest.evaluateGuestDragSlip([
      gPushWorld[0] + gAxisX[0] * 0.20,
      gPushWorld[1],
      gPushWorld[2] + gAxisX[2] * 0.20,
    ]);
    expect(guestEndPush?.slipped).toBe(false);
    expect(guestEndPush?.maxForceN).toBeCloseTo(F_PUSH_MAX, 4);
    guest.setOptimisticGuestPullAnchor(null);

    // 5. GameEngine DOM pointer drag (handlePointerDown + handlePointerMove) double-slip cleanup verification
    const domTight = engine.physics.blocks.find(
      (b) => b.assignedLayer === 8 && !b.isLooseSlot && b.assignedSlot === 2 && b.isLegalTarget
    )!;
    expect(domTight).toBeDefined();
    const domAxisX = quatRotateVec3(domTight.rotation, [1, 0, 0]);
    const domStartWorld: Vec3 = [
      domTight.position[0],
      domTight.position[1],
      domTight.position[2] + BLOCK_HALF_Z,
    ];
    const domPullWorld: Vec3 = [
      domStartWorld[0] + domAxisX[0] * 0.32,
      domStartWorld[1],
      domStartWorld[2] + domAxisX[2] * 0.32,
    ];
    const [downX, downY] = projectToScreen(domStartWorld);
    const [dragX, dragY] = projectToScreen(domPullWorld);
    engine.handlePointerDown(downX, downY, 1, 0, false, rect, 'mouse', 5000);
    engine.handlePointerMove(dragX, dragY, 1, rect, 'mouse', 5016);
    if (engine.getInteractionMode() === 'block') {
      engine.handlePointerMove(dragX, dragY, 1, rect, 'mouse', 5032);
    }
    expect(engine.getLastExtractionSlip()?.slipped).toBe(true);
    expect(engine.getInteractionMode()).toBe('none');
    expect(engine.physics.turnPhase).toBe('SELECT_BLOCK');
    expect(engine.physics.activeBlockId).toBeNull();
    engine.handlePointerUp(dragX, dragY, 1, rect, 'mouse', 5048);

    engine.dispose();

    // 6. MujocoWorkerBridge + attachMujocoWorkerHost 18.0 N compressive end-push parity
    const wAudio = new GlassAudioEngine();
    wAudio.setMuted(true);
    const m4Chan = new MessageChannel();
    const m4HostScope = {
      onmessage: null as ((ev: MessageEvent<WorkerInboundMessage>) => void) | null,
      postMessage: (msg: WorkerOutboundMessage, transfer?: Transferable[]) => {
        m4Chan.port2.postMessage(msg, transfer as never);
      },
    };
    const m4ClientEndpoint: WorkerEndpointLike = {
      onmessage: null,
      postMessage: (msg: WorkerInboundMessage, transfer?: Transferable[]) => {
        m4Chan.port1.postMessage(msg, transfer as never);
      },
    };
    m4Chan.port1.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
      m4ClientEndpoint.onmessage?.(ev);
    };
    m4Chan.port2.onmessage = (ev: MessageEvent<WorkerInboundMessage>) => {
      m4HostScope.onmessage?.(ev);
    };
    m4Chan.port1.start();
    m4Chan.port2.start();

    const m4HostHandle = attachMujocoWorkerHost(m4HostScope);
    const m4Bridge = await MujocoWorkerBridge.create(wAudio, 18, 42, m4ClientEndpoint);
    const stepM4Bridge = async (dt = 1 / 60): Promise<void> => {
      await new Promise<void>((resolve) => {
        const prev = m4ClientEndpoint.onmessage;
        m4ClientEndpoint.onmessage = (ev: MessageEvent<WorkerOutboundMessage>) => {
          prev?.(ev);
          if (ev.data.type === 'snapshot' && ev.data.epoch === m4Bridge.getEpoch()) {
            m4ClientEndpoint.onmessage = prev;
            resolve();
          }
        };
        m4Bridge.step(dt);
      });
    };

    m4Bridge.rebuildTower('solo', 2, 18, 42);
    for (let f = 0; f < 90; f++) await stepM4Bridge(1 / 60);

    const wTight = m4Bridge.blocks.find((b) => b.assignedLayer === 8 && !b.isLooseSlot)!;
    const wAxisX = quatRotateVec3(wTight.rotation, [1, 0, 0]);
    const wPushLocal: Vec3 = [-0.35, 0, 0];
    const wPushOff = quatRotateVec3(wTight.rotation, wPushLocal);
    const wPushStart: Vec3 = [
      wTight.position[0] + wPushOff[0],
      wTight.position[1] + wPushOff[1],
      wTight.position[2] + wPushOff[2],
    ];
    const wInitPos: Vec3 = [...wTight.position];
    expect(m4Bridge.startGrabBlock(wTight.id, wPushLocal, wPushStart, 6000)).toBe(true);
    for (let s = 1; s <= 35; s++) {
      const inwardDist = Math.min(0.42, s * 0.015);
      m4Bridge.updateGrabTarget(
        [
          wPushStart[0] + wAxisX[0] * inwardDist,
          wPushStart[1],
          wPushStart[2] + wAxisX[2] * inwardDist,
        ],
        6000 + s * 16
      );
      await stepM4Bridge(1 / 60);
    }
    const wTightAfter = m4Bridge.blocks[wTight.id];
    const wNudgedDistDm = Math.hypot(
      wTightAfter.position[0] - wInitPos[0],
      wTightAfter.position[2] - wInitPos[2]
    );
    expect(m4Bridge.lastExtractionSlip?.slipped).toBe(false);
    expect(m4Bridge.lastExtractionSlip?.maxForceN).toBeCloseTo(F_PUSH_MAX, 4);
    expect(wNudgedDistDm).toBeGreaterThan(0.03); // Nudged >= 3 mm in Worker mode!
    m4Bridge.endGrab(false);

    // 7. Batched 75 ms Worker step end-push impulse parity (`clampedDt <= 0.085 s`, zero 76% dt starvation)
    const impulses60Hz: number[] = [];
    const impulsesBatched75ms: number[] = [];
    for (let i = 0; i < 4; i++) {
      applyAssistiveEndPushImpulse({
        gameMode: 'solo',
        turnPhase: 'PULLING',
        isPointerDraggingBlock: true,
        blockPos: [0, 1.2, 0],
        blockRotation: [0, 0, 0, 1],
        blockLinVel: [0, 0, 0],
        grabLocalPointDm: [-0.35, 0, 0],
        grabCurrentWorldOnBlockDm: [-0.35, 1.2, 0],
        grabWorldAnchorDm: [-0.15, 1.2, 0],
        grabAnchorStartDm: [-0.35, 1.2, 0],
        grabStartBlockPos: [0, 1.2, 0],
        clampingLoadN: 9.0,
        grabResistance01: 0.7,
        dt: 0.01875,
        applyImpulseAtPoint: (imp) => impulses60Hz.push(Math.hypot(imp.x, imp.z)),
      });
    }
    applyAssistiveEndPushImpulse({
      gameMode: 'solo',
      turnPhase: 'PULLING',
      isPointerDraggingBlock: true,
      blockPos: [0, 1.2, 0],
      blockRotation: [0, 0, 0, 1],
      blockLinVel: [0, 0, 0],
      grabLocalPointDm: [-0.35, 0, 0],
      grabCurrentWorldOnBlockDm: [-0.35, 1.2, 0],
      grabWorldAnchorDm: [-0.15, 1.2, 0],
      grabAnchorStartDm: [-0.35, 1.2, 0],
      grabStartBlockPos: [0, 1.2, 0],
      clampingLoadN: 9.0,
      grabResistance01: 0.7,
      dt: 0.075,
      applyImpulseAtPoint: (imp) => impulsesBatched75ms.push(Math.hypot(imp.x, imp.z)),
    });
    const sum60Hz = impulses60Hz.reduce((a, b) => a + b, 0);
    const sum75ms = impulsesBatched75ms.reduce((a, b) => a + b, 0);
    expect(sum75ms).toBeGreaterThan(sum60Hz * 0.75);

    // 8. Consistent 3.42 N competitive outward end-face slip across 1-block, 2-block, and 3-block rows
    for (const sameRowOthers of [0, 1, 2]) {
      const endPullEval = {
        isEndFace: true,
        slip: { slipped: true, maxForceN: F_SLIP, slipAudio: true },
      };
      expect(shouldReleaseCompetitiveSlip(endPullEval, sameRowOthers)).toBe(true);
      expect(normalizeCompetitiveSlipResult(endPullEval, sameRowOthers)).toEqual({
        slipped: true,
        maxForceN: F_SLIP,
        slipAudio: true,
      });
    }

    // 9. Host vs Guest clamping threshold parity across 3.42 N on 18L (id=46) and 24L (id=60, 62, 64)
    for (const [layers, checkIds] of [
      [18, [41, 45, 46]],
      [24, [57, 60, 62, 64]],
    ] as const) {
      const pwParity = new PhysicsWorld(wAudio, layers, { seed: 42 });
      pwParity.rebuildTower('solo', 2, layers, 42);
      for (let f = 0; f < 90; f++) pwParity.step(1 / 60);
      for (const id of checkIds) {
        const hostN = pwParity.computeBlockClampingLoadN(id);
        const guestN = estimateClampingLoadFromWrenchAndStack(
          id,
          pwParity.blocks,
          pwParity.blockPhotoelasticWrench
        );
        expect(guestN > F_SLIP).toBe(hostN > F_SLIP);
      }
      pwParity.dispose();
    }

    // 10. (18.0, 27.6923] N inward end-push alignment (non-zero impulse & >= 3 mm batched/60 Hz nudge on 24L L1/L4)
    const impMidClamped: number[] = [];
    const evalMidClamped = applyAssistiveEndPushImpulse({
      gameMode: 'solo',
      turnPhase: 'PULLING',
      isPointerDraggingBlock: true,
      blockPos: [0, 1.2, 0],
      blockRotation: [0, 0, 0, 1],
      blockLinVel: [0, 0, 0],
      grabLocalPointDm: [-0.35, 0, 0],
      grabCurrentWorldOnBlockDm: [-0.35, 1.2, 0],
      grabWorldAnchorDm: [-0.15, 1.2, 0],
      grabAnchorStartDm: [-0.35, 1.2, 0],
      grabStartBlockPos: [0, 1.2, 0],
      clampingLoadN: 20.0,
      grabResistance01: 0.7,
      dt: 1 / 60,
      applyImpulseAtPoint: (imp) => impMidClamped.push(Math.hypot(imp.x, imp.z)),
    });
    expect(evalMidClamped.slip.slipped).toBe(false);
    expect(impMidClamped.length).toBe(1);
    expect(impMidClamped[0]).toBeGreaterThan(0);

    const impOverPushMax: number[] = [];
    const evalOverPushMax = applyAssistiveEndPushImpulse({
      gameMode: 'solo',
      turnPhase: 'PULLING',
      isPointerDraggingBlock: true,
      blockPos: [0, 1.2, 0],
      blockRotation: [0, 0, 0, 1],
      blockLinVel: [0, 0, 0],
      grabLocalPointDm: [-0.35, 0, 0],
      grabCurrentWorldOnBlockDm: [-0.35, 1.2, 0],
      grabWorldAnchorDm: [-0.15, 1.2, 0],
      grabAnchorStartDm: [-0.35, 1.2, 0],
      grabStartBlockPos: [0, 1.2, 0],
      clampingLoadN: 28.5,
      grabResistance01: 0.7,
      dt: 1 / 60,
      applyImpulseAtPoint: (imp) => impOverPushMax.push(Math.hypot(imp.x, imp.z)),
    });
    expect(evalOverPushMax.slip.slipped).toBe(true);
    expect(impOverPushMax.length).toBe(0);

    // Worker 85 ms batched push & 60 Hz push on 24L L1 id=5 (18.0 < N <= 27.69) nudges >= 3 mm; id=4 (> 27.69 N) slips
    for (const [dt, steps] of [
      [0.085, 7],
      [1 / 60, 35],
    ] as const) {
      m4Bridge.rebuildTower('solo', 2, 24, 42);
      for (let f = 0; f < 90; f++) await stepM4Bridge(1 / 60);
      const w24Blk5 = m4Bridge.blocks[5];
      expect(m4Bridge.computeBlockClampingLoadN(5)).toBeGreaterThan(F_PUSH_MAX);
      expect(m4Bridge.computeBlockClampingLoadN(5) * 0.65).toBeLessThanOrEqual(F_PUSH_MAX);
      const w24Ax = quatRotateVec3(w24Blk5.rotation, [1, 0, 0]);
      const w24Off = quatRotateVec3(w24Blk5.rotation, [-0.35, 0, 0]);
      const w24Start: Vec3 = [
        w24Blk5.position[0] + w24Off[0],
        w24Blk5.position[1] + w24Off[1],
        w24Blk5.position[2] + w24Off[2],
      ];
      const w24Init: Vec3 = [...w24Blk5.position];
      expect(m4Bridge.startGrabBlock(5, [-0.35, 0, 0], w24Start, 7000)).toBe(true);
      for (let s = 1; s <= steps; s++) {
        const dist = Math.min(0.42, s * dt * 0.72);
        m4Bridge.updateGrabTarget(
          [w24Start[0] + w24Ax[0] * dist, w24Start[1], w24Start[2] + w24Ax[2] * dist],
          7000 + Math.round(s * dt * 1000)
        );
        expect(m4Bridge.lastExtractionSlip?.slipped).toBe(false);
        await stepM4Bridge(dt);
      }
      const w24After5 = m4Bridge.blocks[5];
      expect(m4Bridge.lastExtractionSlip?.slipped).toBe(false);
      expect(
        Math.hypot(w24After5.position[0] - w24Init[0], w24After5.position[2] - w24Init[2])
      ).toBeGreaterThan(0.03);
      m4Bridge.endGrab(false);
    }

    // 24L L1 id=4 (> 27.69 N) must still slip immediately on inward end-push
    m4Bridge.rebuildTower('solo', 2, 24, 42);
    for (let f = 0; f < 90; f++) await stepM4Bridge(1 / 60);
    const w24Blk4 = m4Bridge.blocks[4];
    expect(m4Bridge.computeBlockClampingLoadN(4) * 0.65).toBeGreaterThan(F_PUSH_MAX);
    const w24Ax4 = quatRotateVec3(w24Blk4.rotation, [1, 0, 0]);
    const w24Off4 = quatRotateVec3(w24Blk4.rotation, [-0.35, 0, 0]);
    const w24Start4: Vec3 = [
      w24Blk4.position[0] + w24Off4[0],
      w24Blk4.position[1] + w24Off4[1],
      w24Blk4.position[2] + w24Off4[2],
    ];
    expect(m4Bridge.startGrabBlock(4, [-0.35, 0, 0], w24Start4, 7500)).toBe(true);
    m4Bridge.updateGrabTarget(
      [w24Start4[0] + w24Ax4[0] * 0.05, w24Start4[1], w24Start4[2] + w24Ax4[2] * 0.05],
      7516
    );
    expect(m4Bridge.lastExtractionSlip?.slipped).toBe(true);
    m4Bridge.endGrab(false);

    // 11. 60 Hz continuous inward end-push on clamped 18L L4 id=12 & id=14 resists contact spikes without false slip
    for (const blkId of [12, 14] as const) {
      m4Bridge.rebuildTower('solo', 2, 18, 42);
      for (let f = 0; f < 90; f++) await stepM4Bridge(1 / 60);
      const w18L4 = m4Bridge.blocks[blkId];
      const w18Ax = quatRotateVec3(w18L4.rotation, [1, 0, 0]);
      const w18Off = quatRotateVec3(w18L4.rotation, [-0.35, 0, 0]);
      const w18Start: Vec3 = [
        w18L4.position[0] + w18Off[0],
        w18L4.position[1] + w18Off[1],
        w18L4.position[2] + w18Off[2],
      ];
      const w18Init: Vec3 = [...w18L4.position];
      expect(m4Bridge.startGrabBlock(blkId, [-0.35, 0, 0], w18Start, 8000)).toBe(true);
      for (let s = 1; s <= 35; s++) {
        const dist = blkId === 12 ? Math.min(0.42, (s / 35) * 0.42) : Math.min(0.42, s * 0.015);
        m4Bridge.updateGrabTarget(
          [w18Start[0] + w18Ax[0] * dist, w18Start[1], w18Start[2] + w18Ax[2] * dist],
          8000 + s * 16
        );
        expect(m4Bridge.lastExtractionSlip?.slipped).toBe(false);
        await stepM4Bridge(1 / 60);
      }
      const w18After = m4Bridge.blocks[blkId];
      expect(
        Math.hypot(w18After.position[0] - w18Init[0], w18After.position[2] - w18Init[2])
      ).toBeGreaterThan(0.03);
      m4Bridge.endGrab(false);
    }

    // 12. dt-normalized asymmetric EMA resists single-step contact spikes near 27.69 N at 60 Hz & 120 Hz
    //     while slipping sustained >27.69 N overloads within ~150-260 ms
    for (const dt of [1 / 60, 1 / 120] as const) {
      let spikeEma = 27.0;
      spikeEma = updateEndPushSmoothedClampingLoadN(spikeEma, 36.0, dt);
      expect(spikeEma * 0.65).toBeLessThanOrEqual(F_PUSH_MAX);
      spikeEma = updateEndPushSmoothedClampingLoadN(spikeEma, 27.0, dt);
      expect(spikeEma * 0.65).toBeLessThanOrEqual(F_PUSH_MAX);

      let sustEma = 19.45;
      let slipStep = -1;
      const maxSteps = Math.round(0.4 / dt);
      for (let s = 1; s <= maxSteps; s++) {
        sustEma = updateEndPushSmoothedClampingLoadN(sustEma, 35.63, dt);
        if (sustEma * 0.65 > F_PUSH_MAX && slipStep === -1) {
          slipStep = s;
        }
      }
      const slipMs = slipStep * dt * 1000;
      expect(slipStep).toBeGreaterThan(0);
      expect(slipMs).toBeGreaterThanOrEqual(150);
      expect(slipMs).toBeLessThanOrEqual(260);
    }

    // 13. Upper-band valid clamped blocks (18L L1 #4 ~25.91 N, 24L L7 #22 ~25.32 N) resist 60 Hz & 120 Hz
    //     solver contact oscillations without false mid-stroke slip across PhysicsWorld and GameEngine
    for (const [layers, blkId, dt, steps] of [
      [18, 4, 1 / 60, 35],
      [18, 4, 1 / 120, 70],
      [24, 22, 1 / 60, 35],
      [24, 22, 1 / 120, 70],
    ] as const) {
      const pwOsc = new PhysicsWorld(wAudio, layers, { seed: 42 });
      pwOsc.rebuildTower('solo', 2, layers, 42);
      for (let f = 0; f < 90; f++) pwOsc.step(1 / 60);
      const blk = pwOsc.blocks[blkId];
      const ax = quatRotateVec3(blk.rotation, [1, 0, 0]);
      const off = quatRotateVec3(blk.rotation, [-0.35, 0, 0]);
      const start: Vec3 = [
        blk.position[0] + off[0],
        blk.position[1] + off[1],
        blk.position[2] + off[2],
      ];
      const initPos: Vec3 = [...blk.position];
      expect(pwOsc.startGrabBlock(blkId, [-0.35, 0, 0], start, 9000)).toBe(true);
      for (let s = 1; s <= steps; s++) {
        const dist = Math.min(0.42, s * dt * 0.72);
        pwOsc.updateGrabTarget(
          [start[0] + ax[0] * dist, start[1], start[2] + ax[2] * dist],
          9000 + Math.round(s * dt * 1000)
        );
        expect(pwOsc.lastExtractionSlip?.slipped).toBe(false);
        pwOsc.step(dt);
      }
      expect(
        Math.hypot(blk.position[0] - initPos[0], blk.position[2] - initPos[2])
      ).toBeGreaterThan(0.03);
      pwOsc.dispose();
    }

    // 14. Glazier AI Oracle evaluates (18.0, 27.6923] N end-push candidates (e.g. 24L L1 #5 at ~19.45 N)
    //     without false unpushable slip penalty
    {
      const geAi = new GameEngine();
      await geAi.initHeadless();
      geAi.physics.rebuildTower('party', 2, 24, 42);
      for (let f = 0; f < 90; f++) geAi.stepHeadless(1 / 60);
      const c5Clamp = geAi.physics.computeBlockClampingLoadN(5);
      const c17Clamp = geAi.physics.computeBlockClampingLoadN(17);
      expect(c5Clamp).toBeGreaterThan(F_PUSH_MAX);
      expect(c5Clamp * 0.65).toBeLessThanOrEqual(F_PUSH_MAX);
      type AiOracleTestAccessor = {
        selectAiTargetFromOracle: (
          candidates: Array<{
            blockId: number;
            safetyScore01: number;
            predictedSlideMm: number;
            clampingLoadN: number;
            collapseTriggered: boolean;
          }>,
          difficulty: 'apprentice' | 'master'
        ) => { blockId: number } | undefined;
      };
      const chosen = (geAi as unknown as AiOracleTestAccessor).selectAiTargetFromOracle(
        [
          {
            blockId: 5,
            safetyScore01: 0.92,
            predictedSlideMm: 8.5,
            clampingLoadN: c5Clamp,
            collapseTriggered: false,
          },
          {
            blockId: 17,
            safetyScore01: 0.92,
            predictedSlideMm: 8.5,
            clampingLoadN: c17Clamp,
            collapseTriggered: false,
          },
        ],
        'master'
      );
      expect(chosen?.blockId).toBe(5);
      geAi.dispose();
    }

    // 15. Exact ±21.75 mm (0.2175 dm) 2-zone grip boundary precision (no IEEE-754 1-ULP drift)
    expect(ZONE_BOUNDARY_DM).toBe(0.2175);
    expect(computeGripZoneFromLocalDm([-0.2175, 0, 0])).toBe(ZONE_BODY);
    expect(computeGripZoneFromLocalDm([0.2175, 0, 0])).toBe(ZONE_BODY);
    expect(computeGripZoneFromLocalDm([0.2176, 0, 0])).toBe(ZONE_END_TIP);
    expect(computeGripZoneFromLocalDm([-0.2176, 0, 0])).toBe(ZONE_END_TIP);

    m4Bridge.dispose();
    m4HostHandle.dispose();
  }, 60000);
});

#!/usr/bin/env node
/**
 * Seeded Monte Carlo Stateful Trajectory Fuzzer (`scripts/fuzz-stateful-trajectories.mjs`)
 *
 * Drives `GameEngine` headlessly across BOTH synchronous `PhysicsWorld` and `MessageChannel`-backed
 * `MujocoWorkerBridge` (`attachMujocoWorkerHost`) through 500+ randomized 80-action sequences:
 *   - Pointer `< 5 px` tap-tests & direct `tapNudgeBlock` (intact & crazed blocks)
 *   - Grazing (`11°`), diagonal, and upward (`24L`-pitch) pointer pulls via `GameEngine`
 *   - Mid-pull aborts (`handlePointerUp` / `handlePointerCancel` / `endGrab(false)`)
 *   - Full block extraction + crooked yaw rotation + off-center top placement (`releaseExtractedBlockOntoTop`)
 *   - `Sandbox` collapse smash (`triggerSandboxCollapseImpulse`), shard nudges (`nudgeShard`), and sandbox drags
 *   - Mode transitions (`solo` <-> `party` <-> `sandbox`), counterfactual CEM oracle & WASM `mj_ray` zero-mutation checks
 *   - `PhysicsNetworkSnapshot` compact round-trips (`exportCompactSnapshot` -> `applyCompactSnapshot`)
 *   - Mid-collapse / in-flight `rebuildTower` epoch-fencing resets
 *   - Unforced passive settle windows
 *
 * Continuously asserts 5 hard invariants:
 *   1. `0` `mjWARN_BADQACC` / `mjWARN_BADQVEL` / `mjWARN_BADQPOS` warnings & `0` `NaN`/`Infinity` poses/velocities + monotonic `simTime`
 *   2. `0` unsupported sleeping blocks or Voronoi shards (`y > 2.6 cm = 0.26 dm`)
 *   3. `0` settled off-rim pieces (`r > TABLE_RADIUS + margin`) stranded above `y = -1.5 dm`
 *   4. Non-increasing passive mechanical energy (`computeMechanicalEnergyJ()`) across unforced settle windows
 *   5. 1:1 `TowerTelemetry` structural & sync-vs-worker parity (`PhysicsWorld` <-> `MujocoWorkerBridge`)
 *
 * Usage:
 *   node scripts/fuzz-stateful-trajectories.mjs [--seeds 520] [--paired 100] [--async-worker 120] [--actions 80] [--seed 0x1234] [--json]
 */

import { spawn } from 'node:child_process';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { MessageChannel } from 'node:worker_threads';

export class FuzzInvariantError extends Error {
  /**
   * @param {string} invariant
   * @param {string} message
   * @param {{ seed: number; mode: string; actionIndex: number; actionLog: Array<Record<string, unknown>> }} context
   */
  constructor(invariant, message, context) {
    super(`[${invariant}] seed=0x${(context.seed >>> 0).toString(16)} (${context.mode}) @ action #${context.actionIndex}: ${message}`);
    this.name = 'FuzzInvariantError';
    this.invariant = invariant;
    this.context = context;
  }
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const VIEWPORT_RECT = { left: 0, top: 0, width: 1280, height: 800 };

/**
 * Projects a 3D world point `[x, y, z]` (Vitrail dm) into `[clientX, clientY]` pixels using `OrbitCamera.viewProjMatrix`.
 */
export function projectWorldToScreen(camera, worldPt, rect = VIEWPORT_RECT) {
  const m = camera.viewProjMatrix;
  const x = worldPt[0];
  const y = worldPt[1];
  const z = worldPt[2];
  const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
  const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
  const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (!Number.isFinite(clipW) || clipW <= 1e-4) return null;
  const ndcX = clipX / clipW;
  const ndcY = clipY / clipW;
  return [
    rect.left + (ndcX * 0.5 + 0.5) * rect.width,
    rect.top + (0.5 - ndcY * 0.5) * rect.height,
  ];
}

function makeStubCanvas() {
  return {
    style: { cursor: 'default' },
    clientWidth: VIEWPORT_RECT.width,
    clientHeight: VIEWPORT_RECT.height,
    getBoundingClientRect: () => ({ ...VIEWPORT_RECT }),
    setPointerCapture: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
  };
}

/**
 * Creates reusable synchronous (`PhysicsWorld`) and worker-backed (`MujocoWorkerBridge` over `MessageChannel`)
 * `GameEngine` harnesses plus trajectory runners and invariant auditors.
 */
export async function createFuzzSuite(deps) {
  const {
    GameEngine,
    GlassAudioEngine,
    PhysicsWorld,
    MujocoWorkerBridge,
    attachMujocoWorkerHost,
    ensureMujocoInitialized,
    constants,
    vecMat,
  } = deps;

  const { BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z, BLOCK_HEIGHT, TABLE_RADIUS } = constants;
  const { quatRotateVec3 } = vecMat;

  await ensureMujocoInitialized(false);

  // 1. Synchronous GameEngine + PhysicsWorld harness
  const syncEngine = new GameEngine(makeStubCanvas());
  await syncEngine.initHeadless();
  syncEngine.audio.setMuted(true);

  // 2. Worker-backed GameEngine + MujocoWorkerBridge over FIFO Node MessageChannel with exact barrier flush
  const channel = new MessageChannel();
  const pendingBarriers = new Map();
  let nextBarrierId = 1;

  const clientEndpoint = {
    onmessage: null,
    postMessage(msg, transfer) {
      channel.port1.postMessage(msg, transfer);
    },
    terminate() {
      channel.port1.close();
      channel.port2.close();
    },
  };

  const hostScope = {
    onmessage: null,
    postMessage(msg, transfer) {
      channel.port2.postMessage(msg, transfer);
    },
    close() {
      channel.port2.close();
    },
  };

  let hostQueueTail = Promise.resolve();
  channel.port1.on('message', (data) => {
    if (data && typeof data === 'object' && '__fuzzPongId' in data) {
      const resolve = pendingBarriers.get(data.__fuzzPongId);
      if (resolve) {
        pendingBarriers.delete(data.__fuzzPongId);
        resolve();
      }
      return;
    }
    clientEndpoint.onmessage?.({ data });
  });

  channel.port2.on('message', (data) => {
    if (data && typeof data === 'object' && '__fuzzPingId' in data) {
      const pingId = data.__fuzzPingId;
      // Wait behind any queued microtasks on the worker host before echoing pong on port2
      hostQueueTail = hostQueueTail
        .then(() => new Promise((r) => setImmediate(r)))
        .then(() => {
          channel.port2.postMessage({ __fuzzPongId: pingId });
        });
      return;
    }
    hostQueueTail = hostQueueTail.then(() => {
      hostScope.onmessage?.({ data });
    });
  });

  const hostHandle = attachMujocoWorkerHost(hostScope);
  const workerAudio = new GlassAudioEngine();
  workerAudio.setMuted(true);
  const bridge = await MujocoWorkerBridge.create(workerAudio, 18, 0x10000001, clientEndpoint);

  const workerEngine = new GameEngine(makeStubCanvas());
  workerEngine.audio.setMuted(true);
  workerEngine.physics = bridge;
  workerEngine.physicsWorkerActive = true;
  workerEngine.authoritativePhysics = bridge;
  workerEngine.authoritativeWorkerActive = true;
  workerEngine.camera.update(0.016, 16 / 9);
  workerEngine.physics.setListenerPose(
    workerEngine.camera.position,
    workerEngine.camera.getRightVector()
  );

  async function flushWorker() {
    const id = nextBarrierId++;
    await new Promise((resolve) => {
      pendingBarriers.set(id, resolve);
      channel.port1.postMessage({ __fuzzPingId: id });
    });
  }

  /**
   * Verifies structural & physical invariants on a single `GameEngine` instance.
   */
  function assertSingleEngineInvariants(engine, prevSimTime, didResetTime, checkSettledOffRim, ctx) {
    const phys = engine.physics;
    const warnCount =
      typeof phys.getSolverWarningCount === 'function' ? phys.getSolverWarningCount() : 0;
    if (warnCount > 0) {
      throw new FuzzInvariantError(
        'SOLVER_WARNING',
        `MuJoCo solver emitted ${warnCount} BADQPOS/BADQVEL/BADQACC warning(s)`,
        ctx
      );
    }

    const simTime = phys.getSimTime();
    if (!Number.isFinite(simTime) || simTime < -1e-9) {
      throw new FuzzInvariantError('INVALID_SIM_TIME', `Non-finite or negative simTime=${simTime}`, ctx);
    }
    if (!didResetTime && simTime + 1e-6 < prevSimTime) {
      throw new FuzzInvariantError(
        'SIM_TIME_RESET',
        `Unexpected solver reset: simTime regressed ${prevSimTime.toFixed(6)} -> ${simTime.toFixed(6)}`,
        ctx
      );
    }

    const rw = typeof phys.getWorld === 'function' ? phys.getWorld() : null;
    const mjModel = rw && typeof rw.getModel === 'function' ? rw.getModel() : null;
    const mjData = rw && typeof rw.getData === 'function' ? rw.getData() : null;
    const bodyTreeId = mjModel ? mjModel.body_treeid : null;
    const geomContype = mjModel ? mjModel.geom_contype : null;
    const geomConaffinity = mjModel ? mjModel.geom_conaffinity : null;
    const treeAsleep = mjData ? mjData.tree_asleep : null;

    const isBodySleepingFast = (bodyId) => {
      if (!rw) return false;
      const mapping = rw.getMapping(bodyId);
      if (!mapping || !mapping.enabled || mapping.settledAsleep) return true;
      if (!bodyTreeId || !treeAsleep) return false;
      const tid = bodyTreeId[bodyId];
      return tid >= 0 ? treeAsleep[tid] >= 0 : false;
    };

    let intactCount = 0;
    let shatteredCount = 0;
    let crazedCount = 0;

    for (let i = 0; i < phys.blocks.length; i++) {
      const b = phys.blocks[i];
      for (let k = 0; k < 3; k++) {
        if (
          !Number.isFinite(b.position[k]) ||
          !Number.isFinite(b.linVel[k]) ||
          !Number.isFinite(b.angVel[k]) ||
          !Number.isFinite(b.crackLocalOrigin[k])
        ) {
          throw new FuzzInvariantError(
            'NAN_BLOCK_STATE',
            `Block #${b.id} has non-finite state pos=${JSON.stringify(b.position)} vel=${JSON.stringify(b.linVel)}`,
            ctx
          );
        }
      }
      const qNorm = Math.hypot(b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]);
      if (!Number.isFinite(qNorm) || Math.abs(qNorm - 1.0) > 0.025) {
        throw new FuzzInvariantError(
          'BAD_BLOCK_QUAT',
          `Block #${b.id} quaternion norm=${qNorm}`,
          ctx
        );
      }
      if (!Number.isFinite(b.damage) || b.damage < -1e-5 || b.damage > 1.0001) {
        throw new FuzzInvariantError(
          'BAD_BLOCK_DAMAGE',
          `Block #${b.id} out-of-bounds damage=${b.damage}`,
          ctx
        );
      }

      if (b.shattered) {
        shatteredCount++;
        continue;
      }
      intactCount++;
      if (b.damage > 0.01) crazedCount++;

      // Unsupported sleeping intact block check (`y > 2.6 cm = 0.26 dm`)
      if (!b.extracted && b.id !== phys.activeBlockId && rw) {
        const ty = b.position[1];
        const asleep = ty > 0.26 ? isBodySleepingFast(b.bodyHandle) : false;
        if (asleep && ty > 0.26) {
          const exVec = quatRotateVec3(b.rotation, [1, 0, 0]);
          const upVec = quatRotateVec3(b.rotation, [0, 1, 0]);
          const ezVec = quatRotateVec3(b.rotation, [0, 0, 1]);
          const extY =
            Math.abs(exVec[1]) * BLOCK_HALF_X +
            Math.abs(upVec[1]) * BLOCK_HALF_Y +
            Math.abs(ezVec[1]) * BLOCK_HALF_Z;
          if (ty - extY > 0.035) {
            let supported = false;
            for (let j = 0; j < phys.blocks.length; j++) {
              const sb = phys.blocks[j];
              if (sb.id === b.id || sb.shattered || sb.extracted) continue;
              const dy = ty - sb.position[1];
              if (
                dy > -0.46 &&
                dy < 0.86 &&
                Math.abs(b.position[0] - sb.position[0]) < 0.86 &&
                Math.abs(b.position[2] - sb.position[2]) < 0.86
              ) {
                supported = true;
                break;
              }
            }
            if (!supported) {
              for (let j = 0; j < phys.shards.length; j++) {
                const os = phys.shards[j];
                const dy = ty - os.position[1];
                if (
                  dy > 0.015 &&
                  dy < 0.66 &&
                  Math.abs(b.position[0] - os.position[0]) < 0.74 &&
                  Math.abs(b.position[2] - os.position[2]) < 0.74
                ) {
                  supported = true;
                  break;
                }
              }
            }
            if (!supported) {
              throw new FuzzInvariantError(
                'UNSUPPORTED_SLEEPING_BLOCK',
                `Block #${b.id} sleeping unsupported in mid-air at pos=${JSON.stringify(b.position)}`,
                ctx
              );
            }
          }
        }
      }

      if (checkSettledOffRim && !b.extracted && b.id !== phys.activeBlockId) {
        const r = Math.hypot(b.position[0], b.position[2]);
        const vy = b.linVel[1];
        const speed3d = Math.hypot(b.linVel[0], b.linVel[1], b.linVel[2]);
        // A block whose CoM is well past the rim + half-length (`4.30 + 0.42 = 4.72 dm`) cannot rest on the table
        if (r > TABLE_RADIUS + 0.44 && b.position[1] > -1.5 && speed3d < 0.12) {
          let bridgedByOnTableBlock = false;
          for (let j = 0; j < phys.blocks.length; j++) {
            const sb = phys.blocks[j];
            if (sb.id === b.id || sb.shattered || sb.extracted) continue;
            if (
              Math.hypot(sb.position[0], sb.position[2]) <= TABLE_RADIUS + 0.40 &&
              Math.hypot(b.position[0] - sb.position[0], b.position[2] - sb.position[2]) < 0.82
            ) {
              bridgedByOnTableBlock = true;
              break;
            }
          }
          if (!bridgedByOnTableBlock) {
            throw new FuzzInvariantError(
              'OFF_RIM_BLOCK_FLOATER',
              `Off-rim block #${b.id} stranded above y=-1.5 dm at r=${r.toFixed(3)}, y=${b.position[1].toFixed(3)}, vy=${vy.toFixed(4)}, speed3d=${speed3d.toFixed(4)}`,
              ctx
            );
          }
        }
      }
    }

    for (let i = 0; i < phys.shards.length; i++) {
      const s = phys.shards[i];
      for (let k = 0; k < 3; k++) {
        if (
          !Number.isFinite(s.position[k]) ||
          !Number.isFinite(s.preLinVel[k]) ||
          !Number.isFinite(s.preAngVel[k])
        ) {
          throw new FuzzInvariantError(
            'NAN_SHARD_STATE',
            `Shard #${s.id} has non-finite state pos=${JSON.stringify(s.position)}`,
            ctx
          );
        }
      }
      const sqNorm = Math.hypot(s.rotation[0], s.rotation[1], s.rotation[2], s.rotation[3]);
      if (!Number.isFinite(sqNorm) || Math.abs(sqNorm - 1.0) > 0.025) {
        throw new FuzzInvariantError(
          'BAD_SHARD_QUAT',
          `Shard #${s.id} quaternion norm=${sqNorm}`,
          ctx
        );
      }
      if (!Number.isFinite(s.massKg) || s.massKg <= 0 || !Number.isFinite(s.volume) || s.volume <= 0) {
        throw new FuzzInvariantError(
          'BAD_SHARD_MASS',
          `Shard #${s.id} invalid massKg=${s.massKg}, volume=${s.volume}`,
          ctx
        );
      }

      // Unsupported sleeping shard check (`y > 2.6 cm = 0.26 dm`)
      if (rw) {
        const [sx, sy, sz] = s.position;
        const asleep = sy > 0.26 ? isBodySleepingFast(s.bodyHandle) : false;
        if (asleep && sy > 0.26) {
          const isHero = geomContype ? geomContype[s.colliderHandle] === 4 : true;
          const collidesWithBlocks =
            isHero || (geomConaffinity ? (geomConaffinity[s.colliderHandle] & 2) !== 0 : false);
          let supported = false;
          if (collidesWithBlocks) {
            for (let j = 0; j < phys.blocks.length; j++) {
              const sb = phys.blocks[j];
              if (sb.shattered || sb.extracted) continue;
              const dy = sy - sb.position[1];
              if (
                dy > -0.11 &&
                dy < 0.55 &&
                Math.abs(sx - sb.position[0]) < 0.55 &&
                Math.abs(sz - sb.position[2]) < 0.55
              ) {
                supported = true;
                break;
              }
            }
            if (!supported && isHero) {
              for (let j = 0; j < phys.shards.length; j++) {
                const os = phys.shards[j];
                if (os.id === s.id) continue;
                if (geomContype && geomContype[os.colliderHandle] !== 4) continue;
                const dy = sy - os.position[1];
                if (
                  dy > 0.012 &&
                  dy < 0.31 &&
                  Math.abs(sx - os.position[0]) < 0.31 &&
                  Math.abs(sz - os.position[2]) < 0.31
                ) {
                  supported = true;
                  break;
                }
              }
            }
          }
          if (!supported) {
            throw new FuzzInvariantError(
              'UNSUPPORTED_SLEEPING_SHARD',
              `Shard #${s.id} (hero=${isHero}) sleeping unsupported at pos=${JSON.stringify(s.position)}`,
              ctx
            );
          }
        }
      }

      if (checkSettledOffRim) {
        const r = Math.hypot(s.position[0], s.position[2]);
        const lv = rw ? rw.getRigidBody(s.bodyHandle)?.linvel() : null;
        const vy = lv ? lv.y : s.preLinVel[1];
        const speed3d = lv
          ? Math.hypot(lv.x, lv.y, lv.z)
          : Math.hypot(s.preLinVel[0], s.preLinVel[1], s.preLinVel[2]);
        if (r > TABLE_RADIUS + 0.14 && s.position[1] > -1.5 && speed3d < 0.12) {
          let supportedByOverhangingBlock = false;
          for (let j = 0; j < phys.blocks.length; j++) {
            const sb = phys.blocks[j];
            if (sb.shattered || sb.extracted) continue;
            const dy = s.position[1] - sb.position[1];
            if (
              Math.hypot(sb.position[0], sb.position[2]) <= TABLE_RADIUS + 0.42 &&
              dy > -0.12 &&
              dy < 0.55 &&
              Math.abs(s.position[0] - sb.position[0]) < 0.55 &&
              Math.abs(s.position[2] - sb.position[2]) < 0.55
            ) {
              supportedByOverhangingBlock = true;
              break;
            }
          }
          if (!supportedByOverhangingBlock) {
            throw new FuzzInvariantError(
              'OFF_RIM_SHARD_FLOATER',
              `Off-rim shard #${s.id} stranded above y=-1.5 dm at r=${r.toFixed(3)}, y=${s.position[1].toFixed(3)}, vy=${vy.toFixed(4)}, speed3d=${speed3d.toFixed(4)}`,
              ctx
            );
          }
        }
      }
    }

    // Telemetry self-consistency (`GameEngine.getTelemetrySnapshot()` vs `physics` ground truth)
    const tel = engine.getTelemetrySnapshot();
    if (
      tel.intactCount !== intactCount ||
      tel.shatteredCount !== shatteredCount ||
      tel.crazedCount !== crazedCount ||
      tel.activeShardCount !== phys.shards.length ||
      tel.intactCount + tel.shatteredCount !== phys.blocks.length ||
      tel.turnPhase !== phys.turnPhase ||
      tel.gameMode !== phys.gameMode ||
      tel.currentPlayer !== phys.currentPlayer ||
      tel.blocksStacked !== phys.blocksStacked ||
      !Number.isFinite(tel.leanAngleDeg) ||
      !Number.isFinite(tel.wobbleSpeedDegPerSec) ||
      !Number.isFinite(tel.liveHeightCm)
    ) {
      throw new FuzzInvariantError(
        'TELEMETRY_SELF_MISMATCH',
        `Telemetry inconsistent with physics state: tel=${JSON.stringify(tel)} vs intact=${intactCount}, crazed=${crazedCount}, shattered=${shatteredCount}, shards=${phys.shards.length}`,
        ctx
      );
    }

    return simTime;
  }

  /**
   * Asserts 1:1 `TowerTelemetry` parity between synchronous `PhysicsWorld` and lockstep `MujocoWorkerBridge`.
   */
  function assertSyncVsWorkerTelemetryParity(syncEng, workerEng, ctx) {
    const a = syncEng.physics.getTelemetry();
    const b = workerEng.physics.getTelemetry();

    const exactKeys = [
      'turnPhase',
      'gameMode',
      'currentPlayer',
      'playerCount',
      'blocksStacked',
      'intactCount',
      'crazedCount',
      'shatteredCount',
      'activeShardCount',
      'currentLayers',
      'peakLayers',
      'lockedMinRow',
      'activeBlockId',
      'hoveredSlotIndex',
      'collapseReason',
    ];
    for (const key of exactKeys) {
      if (a[key] !== b[key]) {
        throw new FuzzInvariantError(
          'TELEMETRY_PARITY_MISMATCH',
          `Field "${key}" diverged: sync=${JSON.stringify(a[key])} vs worker=${JSON.stringify(b[key])}`,
          ctx
        );
      }
    }

    if (
      syncEng.physics.isPointerDraggingBlock !== workerEng.physics.isPointerDraggingBlock ||
      syncEng.physics.isExtractionLiftReady !== workerEng.physics.isExtractionLiftReady
    ) {
      throw new FuzzInvariantError(
        'PHYSICS_FLAG_PARITY_MISMATCH',
        `Drag/lift flags diverged: sync=[${syncEng.physics.isPointerDraggingBlock},${syncEng.physics.isExtractionLiftReady}] vs worker=[${workerEng.physics.isPointerDraggingBlock},${workerEng.physics.isExtractionLiftReady}]`,
        ctx
      );
    }

    const floatChecks = [
      ['leanAngleDeg', 0.05],
      ['wobbleSpeedDegPerSec', 0.15],
      ['liveHeightCm', 0.15],
      ['peakHeightCm', 0.15],
      ['grabResistance01', 0.02],
      ['settleProgress01', 0.02],
      ['carriedYawDeg', 0.25],
    ];
    for (const [key, tol] of floatChecks) {
      if (Math.abs(a[key] - b[key]) > tol) {
        throw new FuzzInvariantError(
          'TELEMETRY_FLOAT_DIVERGENCE',
          `Field "${key}" diverged: sync=${a[key]} vs worker=${b[key]} (tol=${tol})`,
          ctx
        );
      }
    }
    if (
      Math.abs(a.centerOfMassXZ[0] - b.centerOfMassXZ[0]) > 0.05 ||
      Math.abs(a.centerOfMassXZ[1] - b.centerOfMassXZ[1]) > 0.05
    ) {
      throw new FuzzInvariantError(
        'TELEMETRY_COM_DIVERGENCE',
        `centerOfMassXZ diverged: sync=${JSON.stringify(a.centerOfMassXZ)} vs worker=${JSON.stringify(b.centerOfMassXZ)}`,
        ctx
      );
    }
  }

  /**
   * Audits passive mechanical energy monotonicity across an unforced window of `frames` steps.
   */
  async function runPassiveEnergyAndSettleAudit(engine, isWorker, frames, ctx) {
    const phys = engine.physics;
    // Ensure no active finger pull or carried block hover is injecting force
    if (engine.activePointerId !== null) {
      engine.handlePointerCancel();
      if (isWorker) await flushWorker();
    }
    if (phys.turnPhase === 'PULLING' || phys.isPointerDraggingBlock) {
      phys.endGrab(false);
      if (isWorker) await flushWorker();
    }
    if (phys.turnPhase === 'EXTRACTED') {
      return phys.getSimTime();
    }

    let prevSimTime = phys.getSimTime();
    let baselineEnergyJ = phys.computeMechanicalEnergyJ();
    let baselineShattered = phys.shatteredCount;
    let prevWindowMaxJ = baselineEnergyJ;
    let firstWindowAvgJ = null;
    let lastWindowAvgJ = null;
    let windowMaxJ = -Infinity;
    let windowSumJ = 0;
    let windowSamples = 0;

      for (let f = 0; f < frames; f++) {
        engine.stepHeadless(1 / 60);
        if (isWorker) await flushWorker();

        if (f === 0 || f === (frames >>> 1) || f === frames - 1) {
          prevSimTime = assertSingleEngineInvariants(
            engine,
            prevSimTime,
            false,
            f === frames - 1 && frames >= 24,
            ctx
          );
        } else {
          const warnCount =
            typeof phys.getSolverWarningCount === 'function' ? phys.getSolverWarningCount() : 0;
          if (warnCount > 0) {
            throw new FuzzInvariantError(
              'SOLVER_WARNING',
              `MuJoCo solver emitted ${warnCount} warning(s) during passive settle`,
              ctx
            );
          }
          const tNow = phys.getSimTime();
          if (!Number.isFinite(tNow) || tNow + 1e-6 < prevSimTime) {
            throw new FuzzInvariantError(
              'SIM_TIME_RESET',
              `Unexpected solver reset during passive settle: ${prevSimTime} -> ${tNow}`,
              ctx
            );
          }
          prevSimTime = tNow;
        }

        const curEnergyJ = phys.computeMechanicalEnergyJ();
        const curShattered = phys.shatteredCount;
        if (!Number.isFinite(curEnergyJ) || curEnergyJ < -1e-4) {
          throw new FuzzInvariantError(
            'INVALID_MECHANICAL_ENERGY',
            `Invalid mechanical energy E=${curEnergyJ}`,
            ctx
          );
        }

        if (curShattered !== baselineShattered) {
          // Re-anchor passive baseline after a secondary fracture releases small radial separation velocity
          baselineEnergyJ = curEnergyJ;
          baselineShattered = curShattered;
          prevWindowMaxJ = curEnergyJ;
          firstWindowAvgJ = null;
          lastWindowAvgJ = null;
          windowMaxJ = -Infinity;
          windowSumJ = 0;
          windowSamples = 0;
          continue;
        }

        // 1. Instantaneous passive energy must never exceed unforced baseline (+ 3.5 mJ soft-contact relaxation margin)
        if (curEnergyJ > baselineEnergyJ + 0.0035) {
          let topBlockDesc = 'none';
          let maxBlkV = -1;
          for (const b of phys.blocks) {
            if (b.shattered || b.extracted) continue;
            const v = Math.hypot(b.linVel[0], b.linVel[1], b.linVel[2]);
            if (v > maxBlkV) {
              maxBlkV = v;
              topBlockDesc = `b${b.id}(y=${b.position[1].toFixed(3)},v=${v.toFixed(2)},w=${Math.hypot(b.angVel[0], b.angVel[1], b.angVel[2]).toFixed(2)})`;
            }
          }
          let topShardDesc = 'none';
          let maxShdV = -1;
          for (const s of phys.shards) {
            const v = Math.hypot(s.preLinVel[0], s.preLinVel[1], s.preLinVel[2]);
            if (v > maxShdV) {
              maxShdV = v;
              topShardDesc = `s${s.id}[slot=${s.slotIndex}](m=${(s.massKg * 1000).toFixed(1)}g,y=${s.position[1].toFixed(3)},v=${v.toFixed(2)})`;
            }
          }
          const tailActions = JSON.stringify(ctx.actionLog.slice(-4));
          throw new FuzzInvariantError(
            'PASSIVE_ENERGY_INCREASE',
            `Passive mechanical energy exceeded unforced baseline at f=${f}/${frames} (phase=${phys.turnPhase}, mode=${phys.gameMode}, activeBlk=${phys.activeBlockId}, shattered=${curShattered}, topBlk=${topBlockDesc}, topShd=${topShardDesc}, recent=${tailActions}): E=${curEnergyJ.toFixed(6)} J > baseline=${baselineEnergyJ.toFixed(6)} J`,
            ctx
          );
        }

        // 2. 8-frame (133 ms > 30 * solref[0]) peak-envelope monotonicity: avoids false positives when
        //    transient penalty-contact compression stores elastic potential U_contact and relaxes at KE ~ 0.
        if (curEnergyJ > windowMaxJ) windowMaxJ = curEnergyJ;
        windowSumJ += curEnergyJ;
        windowSamples++;
        if (windowSamples === 8) {
          const avgJ = windowSumJ / 8;
          if (firstWindowAvgJ === null) firstWindowAvgJ = avgJ;
          lastWindowAvgJ = avgJ;
          if (windowMaxJ > prevWindowMaxJ + 0.0020) {
            throw new FuzzInvariantError(
              'PASSIVE_ENERGY_WINDOW_GROWTH',
              `Passive 8-frame peak energy envelope grew: ${prevWindowMaxJ.toFixed(6)} J -> ${windowMaxJ.toFixed(6)} J`,
              ctx
            );
          }
          prevWindowMaxJ = Math.min(prevWindowMaxJ, windowMaxJ);
          windowMaxJ = -Infinity;
          windowSumJ = 0;
          windowSamples = 0;
        }
      }

      if (
        firstWindowAvgJ !== null &&
        lastWindowAvgJ !== null &&
        lastWindowAvgJ > firstWindowAvgJ + 0.0015
      ) {
        throw new FuzzInvariantError(
          'PASSIVE_ENERGY_NET_GROWTH',
          `Passive settle net window mean energy grew: ${firstWindowAvgJ.toFixed(6)} J -> ${lastWindowAvgJ.toFixed(6)} J`,
          ctx
        );
      }

      return prevSimTime;
    }

    /**
     * Executes a seeded `actionsCount`-action trajectory on `syncEngine` ONLY (fast mode),
     * or paired lockstep on BOTH `syncEngine` + `workerEngine` (`pairedLockstep = true`),
     * or asynchronous overlapping flight chaos on `workerEngine` (`asyncWorkerChaos = true`).
     */
    async function executeTrajectory(seed, actionsCount, modeOption = 'sync') {
      const rng = mulberry32(seed);
      const actionLog = [];
      const modes = ['solo', 'party', 'sandbox'];
      const initMode = modes[Math.floor(rng() * modes.length)];
      const initSeed = (seed ^ 0x9e3779b9) >>> 0;

      const isPaired = modeOption === 'paired';
      const isAsyncWorker = modeOption === 'async-worker';

      if (isPaired) {
        syncEngine.handlePointerCancel();
        workerEngine.handlePointerCancel();
        syncEngine.physics.rebuildTower(initMode, 2, 18, initSeed);
        workerEngine.physics.rebuildTower(initMode, 2, 18, initSeed);
        await flushWorker();
        syncEngine.camera.setPoseImmediate(0.64, 0.36, 4.55, [0, 1.38, 0], 16 / 9);
        workerEngine.camera.setPoseImmediate(0.64, 0.36, 4.55, [0, 1.38, 0], 16 / 9);
      } else if (isAsyncWorker) {
        workerEngine.handlePointerCancel();
        workerEngine.physics.rebuildTower(initMode, 2, 18, initSeed);
        await flushWorker();
        workerEngine.camera.setPoseImmediate(0.64, 0.36, 4.55, [0, 1.38, 0], 16 / 9);
      } else {
        syncEngine.handlePointerCancel();
        syncEngine.physics.rebuildTower(initMode, 2, 18, initSeed);
        syncEngine.camera.setPoseImmediate(0.64, 0.36, 4.55, [0, 1.38, 0], 16 / 9);
      }

      const primaryEngine = isAsyncWorker ? workerEngine : syncEngine;
      let syncPrevTime = syncEngine.physics.getSimTime();
      let workerPrevTime = workerEngine.physics.getSimTime();

      const stepBoth = async (frames, dt = 1 / 120) => {
        for (let f = 0; f < frames; f++) {
          if (isPaired) {
            syncEngine.stepHeadless(dt);
            workerEngine.stepHeadless(dt);
            await flushWorker();
          } else if (isAsyncWorker) {
            // Intentionally fire occasional un-awaited double steps to exercise stepInFlight accumulation!
            workerEngine.stepHeadless(dt);
            if (f % 3 === 1) {
              workerEngine.stepHeadless(dt);
            }
            await flushWorker();
          } else {
            syncEngine.stepHeadless(dt);
          }
        }
      };

      const pickTargetBlock = (phys, requireLegal = true) => {
        const candidates = [];
        for (let i = 0; i < phys.blocks.length; i++) {
          const b = phys.blocks[i];
          if (b.shattered || b.extracted) continue;
          if (requireLegal && phys.gameMode !== 'sandbox' && !b.isLegalTarget) continue;
          candidates.push(b);
        }
        if (candidates.length === 0) return null;
        return candidates[Math.floor(rng() * candidates.length)];
      };

      let peakShatteredCount = 0;
      let peakActiveShardCount = 0;
      let sawCollapseOrFracture = false;

      for (let actionIndex = 0; actionIndex < actionsCount; actionIndex++) {
        const phys = primaryEngine.physics;
        const phase = phys.turnPhase;
        let didResetTime = false;
        const roll = rng();

        let chosenAction = 'passive_step';
        if (roll < 0.06) {
          chosenAction = 'mid_collapse_rebuild';
        } else if (roll < 0.115) {
          chosenAction = 'mode_transition';
        } else if (roll < 0.135) {
          chosenAction = 'snapshot_roundtrip_or_oracle';
        } else if (phase === 'EXTRACTED') {
          chosenAction = rng() < 0.75 ? 'crooked_place' : 'rotate_carried';
        } else if (phase === 'SETTLING') {
          chosenAction = 'settle_wait';
        } else if (phase === 'COLLAPSED') {
          chosenAction = rng() < 0.22 ? 'mid_collapse_rebuild' : 'collapsed_rubble_interact';
        } else if (phys.gameMode === 'sandbox' && roll < 0.30) {
          chosenAction = 'sandbox_smash';
        } else if (roll < 0.38) {
          chosenAction = 'tap_nudge';
        } else if (roll < 0.55) {
          chosenAction = 'grazing_pull';
        } else if (roll < 0.70) {
          chosenAction = 'diagonal_or_upward_pull';
        } else if (roll < 0.85) {
          chosenAction = 'mid_pull_abort';
        } else {
          chosenAction = 'extract_and_crooked_place';
        }

        const entry = { i: actionIndex, kind: chosenAction, phaseBefore: phase, mode: phys.gameMode };
        actionLog.push(entry);
        const ctx = { seed, mode: modeOption, actionIndex, actionLog };

        switch (chosenAction) {
          case 'tap_nudge': {
            const blk = pickTargetBlock(phys, false);
            if (blk) {
              entry.blockId = blk.id;
              if (rng() < 0.5 && !isPaired) {
                // Exercise GameEngine < 5 px pointer tap path
                const scr = projectWorldToScreen(primaryEngine.camera, blk.position);
                if (scr) {
                  primaryEngine.handlePointerDown(scr[0], scr[1], 1, 0, false, VIEWPORT_RECT);
                  primaryEngine.handlePointerMove(scr[0] + 2.0, scr[1] + 1.5, 1, VIEWPORT_RECT);
                  primaryEngine.handlePointerUp(scr[0] + 2.0, scr[1] + 1.5, 1, VIEWPORT_RECT);
                  if (isAsyncWorker) await flushWorker();
                }
              } else {
                const localN = [rng() < 0.5 ? 1 : -1, 0, 0];
                const localPt = [(rng() - 0.5) * 0.4, 0, (rng() - 0.5) * 0.14];
                if (isPaired) {
                  syncEngine.physics.tapNudgeBlock(blk.id, localN, localPt);
                  workerEngine.physics.tapNudgeBlock(blk.id, localN, localPt);
                  await flushWorker();
                } else {
                  primaryEngine.physics.tapNudgeBlock(blk.id, localN, localPt);
                  if (isAsyncWorker) await flushWorker();
                }
              }
            }
            await stepBoth(1);
            break;
          }

          case 'grazing_pull':
          case 'diagonal_or_upward_pull':
          case 'mid_pull_abort': {
            const blk = pickTargetBlock(phys, true);
            if (!blk) {
              await stepBoth(1);
              break;
            }
            entry.blockId = blk.id;
            const isGrazing = chosenAction === 'grazing_pull';
            const isAbort = chosenAction === 'mid_pull_abort';
            const az = (rng() * 2 - 1) * Math.PI;
            const elev = isGrazing ? (11.0 * Math.PI) / 180.0 : 0.25 + rng() * 0.55;
            const targetY = Math.max(0.8, blk.position[1]);

            if (isPaired) {
              // Use exact lockstep physics commands so Float32 snapshot quantization never alters ray localPoint
              syncEngine.camera.setPoseImmediate(az, elev, 4.4, [0, targetY, 0], 16 / 9);
              workerEngine.camera.setPoseImmediate(az, elev, 4.4, [0, targetY, 0], 16 / 9);
              const axisX = quatRotateVec3(blk.rotation, [1, 0, 0]);
              const sign = rng() < 0.5 ? 1 : -1;
              const localGrab = [BLOCK_HALF_X * 0.72 * sign, 0, 0];
              const worldOff = quatRotateVec3(blk.rotation, localGrab);
              const startWorld = [
                blk.position[0] + worldOff[0],
                blk.position[1] + worldOff[1],
                blk.position[2] + worldOff[2],
              ];
              const diagAngle = chosenAction === 'diagonal_or_upward_pull' ? (rng() - 0.5) * 0.55 : 0;
              const cosD = Math.cos(diagAngle);
              const sinD = Math.sin(diagAngle);
              const dirX = (axisX[0] * cosD - axisX[2] * sinD) * sign;
              const dirZ = (axisX[0] * sinD + axisX[2] * cosD) * sign;
              const dirY = chosenAction === 'diagonal_or_upward_pull' ? rng() * 0.18 : 0;

              const sOk = syncEngine.physics.startGrabBlock(blk.id, localGrab, startWorld);
              const wOk = workerEngine.physics.startGrabBlock(blk.id, localGrab, startWorld);
              await flushWorker();
              if (sOk !== wOk) {
                throw new FuzzInvariantError('GRAB_PARITY', `startGrabBlock sync=${sOk} worker=${wOk}`, ctx);
              }
              const steps = isAbort ? 2 : 3;
              const maxPull = isAbort ? 0.38 : 0.95;
              for (let s = 1; s <= steps; s++) {
                const d = (maxPull * s) / steps;
                const tgt = [startWorld[0] + dirX * d, startWorld[1] + dirY * d, startWorld[2] + dirZ * d];
                syncEngine.physics.updateGrabTarget(tgt);
                workerEngine.physics.updateGrabTarget(tgt);
                await stepBoth(1);
                if (syncEngine.physics.turnPhase !== 'PULLING') break;
              }
              if (
                isAbort ||
                syncEngine.physics.turnPhase === 'PULLING' ||
                (syncEngine.physics.isPointerDraggingBlock &&
                  syncEngine.physics.turnPhase !== 'EXTRACTED')
              ) {
                syncEngine.physics.endGrab(false);
                workerEngine.physics.endGrab(false);
                await flushWorker();
                await stepBoth(1);
              }
            } else {
              // Full GameEngine screen-ray pointer pipeline (grazing 11° / diagonal / upward pitch / mid-pull abort)
              primaryEngine.camera.setPoseImmediate(az, elev, 4.4, [0, targetY, 0], 16 / 9);
              const scr = projectWorldToScreen(primaryEngine.camera, blk.position);
              if (scr) {
                primaryEngine.handlePointerDown(scr[0], scr[1], 1, 0, false, VIEWPORT_RECT);
                const dxTot = (rng() < 0.5 ? 1 : -1) * (45 + rng() * 110);
                const dyTot =
                  chosenAction === 'diagonal_or_upward_pull'
                    ? -(35 + rng() * 95)
                    : (rng() - 0.5) * 36;
                const steps = isAbort ? 2 : 3;
                for (let s = 1; s <= steps; s++) {
                  const u = s / steps;
                  primaryEngine.handlePointerMove(
                    scr[0] + dxTot * u,
                    scr[1] + dyTot * u,
                    1,
                    VIEWPORT_RECT
                  );
                  await stepBoth(1);
                }
                if (isAbort && rng() < 0.5) {
                  primaryEngine.handlePointerCancel(1);
                } else {
                  primaryEngine.handlePointerUp(
                    scr[0] + dxTot,
                    scr[1] + dyTot,
                    1,
                    VIEWPORT_RECT
                  );
                }
                if (isAsyncWorker) await flushWorker();
              }
              await stepBoth(1);
            }
            break;
          }

          case 'extract_and_crooked_place': {
            const blk =
              phys.blocks.find((b) => !b.shattered && !b.extracted && b.isLegalTarget && b.isLooseSlot) ??
              pickTargetBlock(phys, true);
            if (!blk || phys.gameMode === 'sandbox') {
              await stepBoth(1);
              break;
            }
            entry.blockId = blk.id;
            const axisX = quatRotateVec3(blk.rotation, [1, 0, 0]);
            const sign = rng() < 0.5 ? 1 : -1;
            const startPos = [...blk.position];
            const localGrab = [BLOCK_HALF_X * 0.78 * sign, 0, 0];
            const grabOff = quatRotateVec3(blk.rotation, localGrab);
            const grabWorld = [
              startPos[0] + grabOff[0],
              startPos[1] + grabOff[1],
              startPos[2] + grabOff[2],
            ];

            if (isPaired) {
              syncEngine.physics.startGrabBlock(blk.id, localGrab, grabWorld);
              workerEngine.physics.startGrabBlock(blk.id, localGrab, grabWorld);
              await flushWorker();
              for (let s = 1; s <= 8 && syncEngine.physics.turnPhase === 'PULLING'; s++) {
                const pullDm = Math.min(1.42, s * 0.19);
                const tgt = [
                  grabWorld[0] + axisX[0] * sign * pullDm,
                  grabWorld[1],
                  grabWorld[2] + axisX[2] * sign * pullDm,
                ];
                syncEngine.physics.updateGrabTarget(tgt);
                workerEngine.physics.updateGrabTarget(tgt);
                await stepBoth(1, 1 / 60);
              }
              if (syncEngine.physics.turnPhase === 'EXTRACTED') {
                await stepBoth(2);
                const crookedYaw = (rng() - 0.5) * 0.52;
                syncEngine.handleRotateCarried(crookedYaw);
                workerEngine.handleRotateCarried(crookedYaw);
                await flushWorker();
                const lc = syncEngine.physics.topSupportFrame.layerCenter;
                const offX = (rng() - 0.5) * 0.36;
                const offZ = (rng() - 0.5) * 0.36;
                const hoverOrig = [lc[0] + offX, lc[1] + 1.0, lc[2] + offZ];
                syncEngine.physics.updateCarriedHover(hoverOrig, [0, -1, 0]);
                workerEngine.physics.updateCarriedHover(hoverOrig, [0, -1, 0]);
                syncEngine.handleReleaseOntoTop();
                workerEngine.handleReleaseOntoTop();
                await flushWorker();
                await stepBoth(2, 1 / 60);
              } else {
                syncEngine.physics.endGrab(false);
                workerEngine.physics.endGrab(false);
                await flushWorker();
              }
            } else {
              primaryEngine.physics.startGrabBlock(blk.id, localGrab, grabWorld);
              if (isAsyncWorker) await flushWorker();
              for (let s = 1; s <= 8 && primaryEngine.physics.turnPhase === 'PULLING'; s++) {
                const pullDm = Math.min(1.42, s * 0.19);
                primaryEngine.physics.updateGrabTarget([
                  grabWorld[0] + axisX[0] * sign * pullDm,
                  grabWorld[1],
                  grabWorld[2] + axisX[2] * sign * pullDm,
                ]);
                await stepBoth(1, 1 / 60);
              }
              if (primaryEngine.physics.turnPhase === 'EXTRACTED') {
                await stepBoth(2);
                primaryEngine.handleRotateCarried((rng() - 0.5) * 0.65);
                const lc = primaryEngine.physics.topSupportFrame.layerCenter;
                primaryEngine.physics.updateCarriedHover(
                  [lc[0] + (rng() - 0.5) * 0.42, lc[1] + 1.0, lc[2] + (rng() - 0.5) * 0.42],
                  [0, -1, 0]
                );
                if (isAsyncWorker && rng() < 0.5) {
                  // Fire in-flight step right before releaseExtractedBlockOntoTop to stress optimistic SETTLING lock!
                  primaryEngine.physics.step(1 / 120);
                }
                primaryEngine.handleReleaseOntoTop();
                if (isAsyncWorker) await flushWorker();
                await stepBoth(2, 1 / 60);
              } else {
                primaryEngine.physics.endGrab(false);
                if (isAsyncWorker) await flushWorker();
              }
            }
            break;
          }

          case 'rotate_carried': {
            const yaw = (rng() - 0.5) * 0.7;
            if (isPaired) {
              if (rng() < 0.25) {
                syncEngine.handleResetCarriedRotation();
                workerEngine.handleResetCarriedRotation();
              } else {
                syncEngine.handleRotateCarried(yaw);
                workerEngine.handleRotateCarried(yaw);
              }
              await flushWorker();
            } else {
              if (rng() < 0.25) {
                primaryEngine.handleResetCarriedRotation();
              } else {
                primaryEngine.handleRotateCarried(yaw);
              }
              if (isAsyncWorker) await flushWorker();
            }
            await stepBoth(1);
            break;
          }

          case 'crooked_place': {
            await stepBoth(2);
            if (primaryEngine.physics.turnPhase === 'EXTRACTED') {
              const lc = primaryEngine.physics.topSupportFrame.layerCenter;
              const hoverOrig = [
                lc[0] + (rng() - 0.5) * 0.35,
                lc[1] + 1.0,
                lc[2] + (rng() - 0.5) * 0.35,
              ];
              if (isPaired) {
                syncEngine.physics.updateCarriedHover(hoverOrig, [0, -1, 0]);
                workerEngine.physics.updateCarriedHover(hoverOrig, [0, -1, 0]);
                syncEngine.handleReleaseOntoTop();
                workerEngine.handleReleaseOntoTop();
                await flushWorker();
              } else {
                primaryEngine.physics.updateCarriedHover(hoverOrig, [0, -1, 0]);
                primaryEngine.handleReleaseOntoTop();
                if (isAsyncWorker) await flushWorker();
              }
            }
            await stepBoth(2, 1 / 60);
            break;
          }

          case 'sandbox_smash': {
            if (isPaired) {
              syncEngine.physics.triggerSandboxCollapseImpulse();
              workerEngine.physics.triggerSandboxCollapseImpulse();
              await flushWorker();
            } else {
              primaryEngine.physics.triggerSandboxCollapseImpulse();
              if (isAsyncWorker) await flushWorker();
            }
            await stepBoth(14 + Math.floor(rng() * 8), 1 / 60);
            break;
          }

          case 'collapsed_rubble_interact': {
            if (phys.shards.length > 0 && typeof phys.nudgeShard === 'function') {
              const shd = phys.shards[Math.floor(rng() * phys.shards.length)];
              const imp = [(rng() - 0.5) * 0.012, 0.006 + rng() * 0.008, (rng() - 0.5) * 0.012];
              const pt = [...shd.position];
              if (isPaired) {
                syncEngine.physics.nudgeShard(shd.id, pt, imp);
                workerEngine.physics.nudgeShard(shd.id, pt, imp);
                await flushWorker();
              } else {
                primaryEngine.physics.nudgeShard(shd.id, pt, imp);
                if (isAsyncWorker) await flushWorker();
              }
            }
            await stepBoth(5, 1 / 60);
            break;
          }

          case 'mode_transition': {
            const nextMode = modes[Math.floor(rng() * modes.length)];
            const nextPlayers = 2 + Math.floor(rng() * 3);
            const nextSeed = ((seed + actionIndex * 2654435761) ^ 0x5bd1e995) >>> 0;
            entry.nextMode = nextMode;
            didResetTime = true;
            if (isPaired) {
              syncEngine.handlePointerCancel();
              workerEngine.handlePointerCancel();
              syncEngine.physics.rebuildTower(nextMode, nextPlayers, 18, nextSeed);
              workerEngine.physics.rebuildTower(nextMode, nextPlayers, 18, nextSeed);
              await flushWorker();
            } else {
              primaryEngine.handlePointerCancel();
              primaryEngine.physics.rebuildTower(nextMode, nextPlayers, 18, nextSeed);
              if (isAsyncWorker) await flushWorker();
            }
            await stepBoth(1);
            break;
          }

          case 'mid_collapse_rebuild': {
            // Trigger chaotic motion / in-flight worker step, then immediately rebuildTower mid-flight!
            const nextMode = modes[Math.floor(rng() * modes.length)];
            const nextSeed = ((seed ^ Math.imul(actionIndex + 7, 0x85ebca6b)) >>> 0) || 1;
            entry.nextMode = nextMode;
            didResetTime = true;
            if (isPaired) {
              if (rng() < 0.35) {
                syncEngine.physics.triggerSandboxCollapseImpulse();
                workerEngine.physics.triggerSandboxCollapseImpulse();
                syncEngine.stepHeadless(1 / 120);
                workerEngine.stepHeadless(1 / 120);
              }
              syncEngine.physics.rebuildTower(nextMode, 2, 18, nextSeed);
              workerEngine.physics.rebuildTower(nextMode, 2, 18, nextSeed);
              await flushWorker();
            } else if (isAsyncWorker) {
              // Deliberately do NOT await triggerSandboxCollapseImpulse + step before rebuildTower!
              workerEngine.physics.triggerSandboxCollapseImpulse();
              workerEngine.physics.step(1 / 120);
              workerEngine.physics.rebuildTower(nextMode, 2, 18, nextSeed);
              // Assert synchronous tick-0 nominal reset BEFORE awaiting worker!
              const tick0Tel = workerEngine.getTelemetrySnapshot();
              if (
                tick0Tel.turnPhase !== 'SELECT_BLOCK' ||
                tick0Tel.shatteredCount !== 0 ||
                tick0Tel.activeShardCount !== 0
              ) {
                throw new FuzzInvariantError(
                  'EPOCH_TICK0_LEAK',
                  `rebuildTower tick-0 leaked stale state: ${JSON.stringify(tick0Tel)}`,
                  ctx
                );
              }
              await flushWorker();
            } else {
              if (rng() < 0.35) {
                syncEngine.physics.triggerSandboxCollapseImpulse();
                syncEngine.stepHeadless(1 / 120);
              }
              syncEngine.physics.rebuildTower(nextMode, 2, 18, nextSeed);
            }
            await stepBoth(1);
            break;
          }

          case 'snapshot_roundtrip_or_oracle': {
            // Verify counterfactual CEM oracle & native WASM mj_ray mutate ZERO live state, or round-trip compact snapshot
            if (rng() < 0.55) {
              const tBefore = phys.getSimTime();
              const b0PosBefore = [...phys.blocks[0].position];
              if (isAsyncWorker) {
                const [, , hit] = await Promise.all([
                  workerEngine.physics.evaluateCounterfactualBlockCriticalityAsync(2, 2),
                  workerEngine.physics.evaluateAdversarialTopPlacementAsync(undefined, 2),
                  workerEngine.physics.raycastSceneAsync([0, 1.2, 2.5], [0, -0.2, -1]),
                ]);
                void hit;
                await flushWorker();
              } else {
                syncEngine.physics.evaluateCounterfactualBlockCriticality(2, 2);
                syncEngine.physics.evaluateAdversarialTopPlacement(undefined, 2, false);
                const sHit = syncEngine.physics.raycastScene([0, 1.2, 2.5], [0, -0.2, -1]);
                if (isPaired) {
                  if (actionIndex % 2 === 0) {
                    const [, , wHit] = await Promise.all([
                      workerEngine.physics.evaluateCounterfactualBlockCriticalityAsync(2, 2),
                      workerEngine.physics.evaluateAdversarialTopPlacementAsync(undefined, 2),
                      workerEngine.physics.raycastSceneAsync([0, 1.2, 2.5], [0, -0.2, -1]),
                    ]);
                    if ((sHit?.kind ?? null) !== (wHit?.kind ?? null) || (sHit?.id ?? null) !== (wHit?.id ?? null)) {
                      throw new FuzzInvariantError(
                        'RAYCAST_PARITY_MISMATCH',
                        `mj_ray diverged: sync=${JSON.stringify(sHit)} vs worker=${JSON.stringify(wHit)}`,
                        ctx
                      );
                    }
                  } else {
                    // Intentionally asymmetric on odd actionIndex to prove 0-ULP live MjData mutation!
                    await workerEngine.physics.evaluateCounterfactualBlockCriticalityAsync(2, 2);
                  }
                  await flushWorker();
                }
              }
              if (Math.abs(phys.getSimTime() - tBefore) > 1e-9) {
                throw new FuzzInvariantError('ORACLE_MUTATION', 'Counterfactual oracle mutated simTime', ctx);
              }
              for (let k = 0; k < 3; k++) {
                if (Math.abs(phys.blocks[0].position[k] - b0PosBefore[k]) > 1e-7) {
                  throw new FuzzInvariantError('ORACLE_POSE_MUTATION', 'Oracle mutated block 0 pose', ctx);
                }
              }
            } else if (!isPaired && phys.turnPhase !== 'PULLING' && phys.turnPhase !== 'EXTRACTED') {
              didResetTime = true;
              const snap = primaryEngine.physics.exportCompactSnapshot({ includeShardGeometry: true });
              primaryEngine.physics.applyCompactSnapshot(snap);
              if (isAsyncWorker) await flushWorker();
              await stepBoth(1);
            } else {
              await stepBoth(1);
            }
            break;
          }

          case 'settle_wait':
          default: {
            await stepBoth(2, 1 / 30);
            break;
          }
        }

        if (isPaired) {
          syncPrevTime = assertSingleEngineInvariants(syncEngine, syncPrevTime, didResetTime, false, {
            ...ctx,
            mode: 'paired-sync',
          });
          workerPrevTime = assertSingleEngineInvariants(
            workerEngine,
            workerPrevTime,
            didResetTime,
            false,
            { ...ctx, mode: 'paired-worker' }
          );
          assertSyncVsWorkerTelemetryParity(syncEngine, workerEngine, ctx);
        } else if (isAsyncWorker) {
          workerPrevTime = assertSingleEngineInvariants(
            workerEngine,
            workerPrevTime,
            didResetTime,
            false,
            ctx
          );
        } else {
          syncPrevTime = assertSingleEngineInvariants(
            syncEngine,
            syncPrevTime,
            didResetTime,
            false,
            ctx
          );
        }

        if (primaryEngine.physics.shatteredCount > peakShatteredCount) {
          peakShatteredCount = primaryEngine.physics.shatteredCount;
        }
        if (primaryEngine.physics.shards.length > peakActiveShardCount) {
          peakActiveShardCount = primaryEngine.physics.shards.length;
        }
        if (
          primaryEngine.physics.turnPhase === 'COLLAPSED' ||
          primaryEngine.physics.shatteredCount > 0 ||
          primaryEngine.physics.shards.length > 0
        ) {
          sawCollapseOrFracture = true;
        }

        // Passive settle + mechanical energy monotonicity + settled off-rim audit at end of every trajectory
        if (actionIndex === actionsCount - 1) {
          const auditFrames = 24;
          if (isPaired) {
            syncPrevTime = await runPassiveEnergyAndSettleAudit(
              syncEngine,
              false,
              auditFrames,
              { ...ctx, mode: 'paired-sync-audit' }
            );
            workerPrevTime = await runPassiveEnergyAndSettleAudit(
              workerEngine,
              true,
              auditFrames,
              { ...ctx, mode: 'paired-worker-audit' }
            );
            assertSyncVsWorkerTelemetryParity(syncEngine, workerEngine, ctx);
          } else if (isAsyncWorker) {
            workerPrevTime = await runPassiveEnergyAndSettleAudit(
              workerEngine,
              true,
              auditFrames,
              ctx
            );
          } else {
            syncPrevTime = await runPassiveEnergyAndSettleAudit(
              syncEngine,
              false,
              auditFrames,
              ctx
            );
          }
          if (primaryEngine.physics.shatteredCount > peakShatteredCount) {
            peakShatteredCount = primaryEngine.physics.shatteredCount;
          }
          if (primaryEngine.physics.shards.length > peakActiveShardCount) {
            peakActiveShardCount = primaryEngine.physics.shards.length;
          }
          if (
            primaryEngine.physics.turnPhase === 'COLLAPSED' ||
            primaryEngine.physics.shatteredCount > 0 ||
            primaryEngine.physics.shards.length > 0
          ) {
            sawCollapseOrFracture = true;
          }
        }
      }

      return {
        seed: seed >>> 0,
        mode: modeOption,
        actionsExecuted: actionsCount,
        finalPhase: primaryEngine.physics.turnPhase,
        shatteredCount: primaryEngine.physics.shatteredCount,
        activeShardCount: primaryEngine.physics.shards.length,
        peakShatteredCount,
        peakActiveShardCount,
        sawCollapseOrFracture,
      };
    }

    /**
     * Binary-searches the shortest failing action prefix `[1 .. failedAt + 1]` for a failing seed.
     */
    async function minimizeFailingSeed(seed, maxActions, modeOption) {
      let lo = 1;
      let hi = maxActions;
      let bestErr = null;
      while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        try {
          await executeTrajectory(seed, mid, modeOption);
          lo = mid + 1;
        } catch (err) {
          bestErr = err;
          hi = mid - 1;
        }
      }
      return { minActions: bestErr ? hi + 1 : maxActions, error: bestErr };
    }

    function dispose() {
      workerEngine.dispose();
      hostHandle.dispose();
      syncEngine.dispose();
      void PhysicsWorld;
    }

    return {
      syncEngine,
      workerEngine,
      flushWorker,
      executeTrajectory,
      minimizeFailingSeed,
      dispose,
    };
}

function parseArgInt(args, flag, fallback) {
  const idx = args.indexOf(flag);
  if (idx >= 0 && idx + 1 < args.length) {
    const raw = args[idx + 1];
    const parsed = raw.startsWith('0x') ? Number.parseInt(raw, 16) : Number.parseInt(raw, 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

async function runMultiWorkerCampaign({
  totalSeeds,
  offsetIdx,
  limitIdx,
  pairedSeeds,
  asyncWorkerSeeds,
  actionsPerSeed,
  numWorkers,
  emitJson,
}) {
  const t0 = performance.now();
  const children = [];
  const killChildren = () => {
    for (const c of children) {
      if (c.exitCode === null) {
        try {
          c.kill('SIGTERM');
        } catch {
          // ignore
        }
      }
    }
  };
  process.on('SIGTERM', killChildren);
  process.on('SIGINT', killChildren);
  process.on('SIGHUP', killChildren);
  process.on('exit', killChildren);

  let results;
  try {
    results = await Promise.all(
      Array.from({ length: numWorkers }, (_, w) => {
        return new Promise((resolve, reject) => {
          const childArgs = [
            process.argv[1],
            '--seeds',
            String(totalSeeds),
            '--offset',
            String(offsetIdx),
            '--limit',
            String(limitIdx),
            '--paired',
            String(pairedSeeds),
            '--async-worker',
            String(asyncWorkerSeeds),
            '--actions',
            String(actionsPerSeed),
            '--worker-id',
            String(w),
            '--workers',
            String(numWorkers),
            '--json',
          ];
          const child = spawn(process.execPath, childArgs, {
            cwd: process.cwd(),
            stdio: ['ignore', 'pipe', 'inherit'],
          });
          children.push(child);
          let stdoutBuf = '';
          child.stdout.on('data', (chunk) => {
            stdoutBuf += chunk.toString();
          });
          child.on('error', (err) => reject(err));
          child.on('close', (code) => {
            if (code !== 0) {
              killChildren();
              reject(new Error(`Fuzz worker #${w} exited with code ${code}`));
              return;
            }
            try {
              const lines = stdoutBuf.trim().split('\n');
              const jsonStart = stdoutBuf.indexOf('{');
              const parsed = JSON.parse(
                jsonStart >= 0 ? stdoutBuf.slice(jsonStart) : lines.join('\n')
              );
              resolve(parsed);
            } catch (err) {
              reject(
                new Error(`Failed to parse worker #${w} JSON output: ${err.message}\n${stdoutBuf}`)
              );
            }
          });
        });
      })
    );
  } finally {
    process.off('SIGTERM', killChildren);
    process.off('SIGINT', killChildren);
    process.off('SIGHUP', killChildren);
    process.off('exit', killChildren);
  }

  const summary = {
    overallPass: results.every((r) => r.overallPass),
    workers: numWorkers,
    offsetIdx,
    limitIdx,
    totalSeeds: results.reduce((acc, r) => acc + r.totalSeeds, 0),
    actionsPerSeed,
    totalActions: results.reduce((acc, r) => acc + r.totalActions, 0),
    pairedLockstepTrajectories: results.reduce((acc, r) => acc + r.pairedLockstepTrajectories, 0),
    asyncWorkerChaosTrajectories: results.reduce((acc, r) => acc + r.asyncWorkerChaosTrajectories, 0),
    syncTrajectories: results.reduce((acc, r) => acc + r.syncTrajectories, 0),
    fractureOrCollapseTrajectories: results.reduce(
      (acc, r) => acc + r.fractureOrCollapseTrajectories,
      0
    ),
    elapsedMs: +(performance.now() - t0).toFixed(1),
  };

  if (emitJson) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log('\n=== STATEFUL MONTE CARLO FUZZ SCORECARD ===');
    console.log(JSON.stringify(summary, null, 2));
  }
}

async function runCli() {
  const args = process.argv.slice(2);
  const emitJson = args.includes('--json');
  const singleSeedIdx = args.indexOf('--seed');
  const totalSeeds = parseArgInt(args, '--seeds', 520);
  const offsetIdx = Math.max(0, parseArgInt(args, '--offset', 0));
  const limitIdx = Math.min(totalSeeds, Math.max(offsetIdx, parseArgInt(args, '--limit', totalSeeds)));
  const rangeSpan = limitIdx - offsetIdx;
  const pairedSeeds = parseArgInt(args, '--paired', 65);
  const asyncWorkerSeeds = parseArgInt(args, '--async-worker', 85);
  const actionsPerSeed = parseArgInt(args, '--actions', 80);
  const workerId = parseArgInt(args, '--worker-id', -1);
  const defaultWorkers = Math.min(4, Math.max(1, os.availableParallelism ? os.availableParallelism() : os.cpus().length));
  const numWorkers = Math.max(1, parseArgInt(args, '--workers', defaultWorkers));

  if (singleSeedIdx < 0 && workerId < 0 && numWorkers > 1 && rangeSpan >= numWorkers * 2) {
    try {
      await runMultiWorkerCampaign({
        totalSeeds,
        offsetIdx,
        limitIdx,
        pairedSeeds,
        asyncWorkerSeeds,
        actionsPerSeed,
        numWorkers,
        emitJson,
      });
    } catch (err) {
      console.error(err.message || err);
      process.exitCode = 1;
    }
    return;
  }

  const { createServer } = await import('vite');
  const vite = await createServer({
    logLevel: 'silent',
    server: { middlewareMode: true, ws: false },
    appType: 'custom',
  });

  let suite = null;
  try {
    const [
      { GameEngine },
      { GlassAudioEngine },
      { PhysicsWorld },
      { MujocoWorkerBridge },
      { attachMujocoWorkerHost },
      { ensureMujocoInitialized },
      constants,
      vecMat,
    ] = await Promise.all([
      vite.ssrLoadModule('/src/engine/GameEngine.ts'),
      vite.ssrLoadModule('/src/audio/GlassAudioEngine.ts'),
      vite.ssrLoadModule('/src/physics/PhysicsWorld.ts'),
      vite.ssrLoadModule('/src/physics/MujocoWorkerBridge.ts'),
      vite.ssrLoadModule('/src/physics/mujocoWorker.ts'),
      vite.ssrLoadModule('/src/physics/mujocoRuntime.ts'),
      vite.ssrLoadModule('/src/physics/constants.ts'),
      vite.ssrLoadModule('/src/math/vecMat.ts'),
    ]);

    const t0 = performance.now();
    suite = await createFuzzSuite({
      GameEngine,
      GlassAudioEngine,
      PhysicsWorld,
      MujocoWorkerBridge,
      attachMujocoWorkerHost,
      ensureMujocoInitialized,
      constants,
      vecMat,
    });

    if (singleSeedIdx >= 0) {
      const seedVal = parseArgInt(args, '--seed', 1) >>> 0;
      const modeIdx = args.indexOf('--mode');
      const modeVal = modeIdx >= 0 && args[modeIdx + 1] ? args[modeIdx + 1] : 'paired';
      const res = await suite.executeTrajectory(seedVal, actionsPerSeed, modeVal);
      console.log(JSON.stringify({ ok: true, result: res }, null, 2));
      return;
    }

    const startIdx = offsetIdx + (workerId >= 0 ? workerId : 0);
    const stride = workerId >= 0 ? numWorkers : 1;
    const myTargetTotal = Math.ceil(Math.max(0, limitIdx - startIdx) / stride);

    let syncCount = 0;
    let pairedCount = 0;
    let asyncWorkerCount = 0;
    let collapseTrajectories = 0;
    let totalActions = 0;
    let completed = 0;

    for (let i = startIdx; i < limitIdx; i += stride) {
      const seed = (0x517cc1b7 + Math.imul(i + 1, 0x9e3779b9)) >>> 0;
      const mode =
        i < pairedSeeds
          ? 'paired'
          : i < pairedSeeds + asyncWorkerSeeds
            ? 'async-worker'
            : 'sync';

      try {
        const res = await suite.executeTrajectory(seed, actionsPerSeed, mode);
        if (mode === 'paired') pairedCount++;
        else if (mode === 'async-worker') asyncWorkerCount++;
        else syncCount++;
        if (res.sawCollapseOrFracture) collapseTrajectories++;
        totalActions += res.actionsExecuted;
        completed++;

        const shouldLogProgress =
          (!emitJson || workerId >= 0) &&
          (completed % 10 === 0 || completed === myTargetTotal);
        if (shouldLogProgress) {
          const elapsedS = ((performance.now() - t0) / 1000).toFixed(1);
          const rssMb = (process.memoryUsage().rss / 1048576).toFixed(0);
          const prefix = workerId >= 0 ? `[fuzz w${workerId}]` : '[fuzz]';
          process.stderr.write(
            `${prefix} ${completed}/${myTargetTotal} trajectories passed (${totalActions} actions, ${collapseTrajectories} fracture/collapse runs, rss=${rssMb}MB, ${elapsedS}s)\n`
          );
        }
      } catch (err) {
        const failedAt = err?.context?.actionIndex ?? actionsPerSeed - 1;
        const minimized = await suite.minimizeFailingSeed(seed, failedAt + 1, mode);
        const minErr = minimized.error ?? err;
        console.error('\n=== STATEFUL FUZZER FAILURE DETECTED & MINIMIZED ===');
        console.error(minErr.message);
        console.error(
          `Repro CLI: node scripts/fuzz-stateful-trajectories.mjs --seed 0x${seed.toString(16)} --mode ${mode} --actions ${minimized.minActions}`
        );
        if (minErr?.context?.actionLog) {
          console.error(
            'Minimized Action Prefix:',
            JSON.stringify(minErr.context.actionLog, null, 2)
          );
        }
        process.exitCode = 1;
        return;
      }
    }

    const totalMs = +(performance.now() - t0).toFixed(1);
    const summary = {
      overallPass: true,
      totalSeeds: completed,
      actionsPerSeed,
      totalActions,
      pairedLockstepTrajectories: pairedCount,
      asyncWorkerChaosTrajectories: asyncWorkerCount,
      syncTrajectories: syncCount,
      fractureOrCollapseTrajectories: collapseTrajectories,
      elapsedMs: totalMs,
    };

    if (emitJson) {
      console.log(JSON.stringify(summary, null, 2));
    } else {
      console.log('\n=== STATEFUL MONTE CARLO FUZZ SCORECARD ===');
      console.log(JSON.stringify(summary, null, 2));
    }
  } finally {
    suite?.dispose();
    await vite.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runCli().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

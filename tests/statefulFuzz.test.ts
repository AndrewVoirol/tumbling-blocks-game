import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';
import { GameEngine } from '../src/engine/GameEngine';
import * as vecMat from '../src/math/vecMat';
import * as constants from '../src/physics/constants';
import { ensureMujocoInitialized } from '../src/physics/mujocoRuntime';
import { attachMujocoWorkerHost } from '../src/physics/mujocoWorker';
import { MujocoWorkerBridge } from '../src/physics/MujocoWorkerBridge';
import { PhysicsWorld } from '../src/physics/PhysicsWorld';

interface TrajectorySummary {
  seed: number;
  mode: 'sync' | 'paired' | 'async-worker';
  actionsExecuted: number;
  finalPhase: string;
  shatteredCount: number;
  activeShardCount: number;
  peakShatteredCount: number;
  peakActiveShardCount: number;
  sawCollapseOrFracture: boolean;
}

interface FuzzSuiteHandle {
  syncEngine: GameEngine;
  workerEngine: GameEngine;
  flushWorker: () => Promise<void>;
  executeTrajectory: (
    seed: number,
    actionsCount: number,
    modeOption?: 'sync' | 'paired' | 'async-worker'
  ) => Promise<TrajectorySummary>;
  minimizeFailingSeed: (
    seed: number,
    maxActions: number,
    modeOption: 'sync' | 'paired' | 'async-worker'
  ) => Promise<{ minActions: number; error: Error | null }>;
  dispose: () => void;
}

interface FuzzModuleExports {
  createFuzzSuite: (deps: {
    GameEngine: typeof GameEngine;
    GlassAudioEngine: typeof GlassAudioEngine;
    PhysicsWorld: typeof PhysicsWorld;
    MujocoWorkerBridge: typeof MujocoWorkerBridge;
    attachMujocoWorkerHost: typeof attachMujocoWorkerHost;
    ensureMujocoInitialized: typeof ensureMujocoInitialized;
    constants: typeof constants;
    vecMat: typeof vecMat;
  }) => Promise<FuzzSuiteHandle>;
}

describe('Stateful Monte Carlo Trajectory Fuzzer (Sync & Worker Parity + Physical Invariants)', () => {
  let suite: FuzzSuiteHandle | null = null;

  beforeAll(async () => {
    const modUrl = new URL('../scripts/fuzz-stateful-trajectories.mjs', import.meta.url).href;
    const fuzzMod = (await import(/* @vite-ignore */ modUrl)) as FuzzModuleExports;
    suite = await fuzzMod.createFuzzSuite({
      GameEngine,
      GlassAudioEngine,
      PhysicsWorld,
      MujocoWorkerBridge,
      attachMujocoWorkerHost,
      ensureMujocoInitialized,
      constants,
      vecMat,
    });
  }, 45000);

  afterAll(() => {
    suite?.dispose();
    suite = null;
  });

  it('F1: lockstep paired 80-action trajectories maintain 1:1 TowerTelemetry parity between PhysicsWorld and MujocoWorkerBridge', async () => {
    expect(suite).not.toBeNull();
    // Includes regression pins `0xe33723ea` (pinched-shard relief) and `0x4f1a29e7` (`47 rad/s` spinning cullet passivity)
    const pairedSeeds = [
      0xefb43b70,
      0xc4e7a4c8,
      0xe33723ea,
      0x4f1a29e7,
    ];
    let sawCollapseOrFracture = false;
    for (const seed of pairedSeeds) {
      const res = await suite!.executeTrajectory(seed, 80, 'paired');
      expect(res.actionsExecuted).toBe(80);
      if (res.sawCollapseOrFracture) {
        sawCollapseOrFracture = true;
      }
    }
    expect(sawCollapseOrFracture).toBe(true);
  }, 120000);

  it('F2: asynchronous overlapping MujocoWorkerBridge 80-action chaos trajectories enforce monotonic epoch fencing, 0 solver warnings, and passive energy decay', async () => {
    expect(suite).not.toBeNull();
    const asyncSeeds = [
      0xa742c506,
      0x45da869f,
      0xe4724838,
    ];
    for (const seed of asyncSeeds) {
      const res = await suite!.executeTrajectory(seed, 80, 'async-worker');
      expect(res.actionsExecuted).toBe(80);
    }
  }, 120000);

  it('F3: screen-ray pointer & Sandbox smash 80-action trajectories maintain 0 unsupported sleepers (y > 2.6 cm), 0 off-rim floaters (y > -1.5 dm), and monotonic passive energy', async () => {
    expect(suite).not.toBeNull();
    let collapseCount = 0;
    for (let i = 0; i < 3; i++) {
      const seed = (0x9c00d1ce + Math.imul(i + 1, 0x9e3779b9)) >>> 0;
      const res = await suite!.executeTrajectory(seed, 80, 'sync');
      expect(res.actionsExecuted).toBe(80);
      if (res.sawCollapseOrFracture) {
        collapseCount++;
      }
    }
    // Regression pin (`0x76a73ef9` @ 80 actions): mass-scaled hero-shard pinch relief under multi-block rubble.
    const pinchReliefRes = await suite!.executeTrajectory(0x76a73ef9, 80, 'sync');
    expect(pinchReliefRes.actionsExecuted).toBe(80);
    // Regression pin (`0x12ab3675` @ 77 actions): single dropped block triggers `COLLAPSED` while 53 blocks
    // remain standing upright (`shatteredCount === 0`), verifying 600 Hz / 26-iter Newton column stability.
    const singleDropCollapsedRes = await suite!.executeTrajectory(0x12ab3675, 77, 'sync');
    expect(singleDropCollapsedRes.actionsExecuted).toBe(77);
    expect(singleDropCollapsedRes.finalPhase).toBe('COLLAPSED');
    expect(collapseCount).toBeGreaterThan(0);
  }, 120000);
});

import { describe, expect, it } from 'vitest';
import { fractureBlockVoronoi } from '../src/physics/voronoiFracture';
import {
  BLOCK_HEIGHT,
  BLOCK_LENGTH,
  BLOCK_WIDTH,
  INITIAL_BLOCK_COUNT,
} from '../src/physics/constants';
import { ensureRapierInitialized, PhysicsWorld } from '../src/physics/PhysicsWorld';
import { GlassAudioEngine } from '../src/audio/GlassAudioEngine';

describe('3D Voronoi Stained Glass Fracture', () => {
  it('conserves exact 7.5 x 2.5 x 1.5 cm Jenga block volume across all convex shards', () => {
    const expectedVolume = BLOCK_LENGTH * BLOCK_HEIGHT * BLOCK_WIDTH; // 0.028125 dm^3
    const shards = fractureBlockVoronoi([0.28, -0.05, 0.08], 13, 42);

    expect(shards.length).toBeGreaterThanOrEqual(8);
    let sumVolume = 0;
    let hasOuterFace = false;
    let hasInnerCleavageFace = false;

    for (const s of shards) {
      expect(s.volume).toBeGreaterThan(0);
      expect(s.hullPoints.length).toBeGreaterThanOrEqual(12); // >= 4 3D vertices
      expect(s.vertexCount % 3).toBe(0);
      sumVolume += s.volume;

      for (let v = 0; v < s.vertexCount; v++) {
        const flag = s.renderVertices[v * 10 + 9];
        if (flag > 0.5) hasOuterFace = true;
        else hasInnerCleavageFace = true;
      }
    }

    expect(Math.abs(sumVolume - expectedVolume)).toBeLessThan(1e-4);
    expect(hasOuterFace).toBe(true);
    expect(hasInnerCleavageFace).toBe(true);
  });
});

describe('Rapier3D 240Hz Stained Glass Jenga Physics', () => {
  it('builds a stable 18-layer (27.0 cm) 54-block tower and shatters falling blocks on high impact', async () => {
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

    // Trigger natural cascade collapse impulse and verify falling blocks tumble & shatter on the oak table!
    world.rebuildTower('sandbox');
    for (let i = 0; i < 20; i++) world.step(1 / 60);
    world.triggerSandboxCollapseImpulse();
    // Step 1.5 real seconds as the tower topples and high blocks smash onto the hard oak table
    for (let i = 0; i < 90; i++) {
      world.step(1 / 60);
    }

    const afterShatter = world.getTelemetry();
    expect(afterShatter.shatteredCount).toBeGreaterThanOrEqual(1);
    expect(afterShatter.activeShardCount).toBeGreaterThanOrEqual(6);

    world.dispose();
  });
});

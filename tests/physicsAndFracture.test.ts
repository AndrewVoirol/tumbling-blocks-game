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
  });
});

describe('Rapier3D Stained Glass Jenga Physics & Convex Hull Shard Settling', () => {
  it('builds a stable 18-layer (27.0 cm) 54-block tower and settles convex shards naturally after collapse', async () => {
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

    // Trigger natural cascade collapse impulse and verify falling blocks tumble & shatter into convex shards
    world.rebuildTower('sandbox');
    for (let i = 0; i < 20; i++) world.step(1 / 60);
    world.triggerSandboxCollapseImpulse();
    // Step 2.0 real seconds as the tower topples and shards skitter to rest on the oak table
    for (let i = 0; i < 120; i++) {
      world.step(1 / 60);
    }

    const afterShatter = world.getTelemetry();
    expect(afterShatter.shatteredCount).toBeGreaterThanOrEqual(1);
    expect(afterShatter.activeShardCount).toBeGreaterThanOrEqual(8);

    // Verify shards rest near the oak tabletop without upward grenade explosions
    let groundedShards = 0;
    for (const s of world.shards) {
      expect(s.position[1]).toBeLessThan(2.8);
      if (s.position[1] > -0.02 && s.position[1] < 0.35) {
        groundedShards++;
      }
    }
    expect(groundedShards).toBeGreaterThan(0);

    world.dispose();
  });
});

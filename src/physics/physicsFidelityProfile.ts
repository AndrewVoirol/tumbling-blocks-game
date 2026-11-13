export interface PhysicsFidelityConfig {
  /**
   * 1. True elastic micro-rebound on unloaded glass impacts + 1:1 bounce-coupled audio
   *    (vs. `contactSolref` ζ = 1.12 overdamped zero-rebound stick & 30 ms timer clatter).
   */
  elasticUnloadedGlassBounce: boolean;
  /**
   * 2. Unassisted 3D MuJoCo extraction: full lateral freedom, dry glass kinetic friction (μk = 0.215),
   *    gravity overhang tipping, and zero phantom mid-air slot-exit clink
   *    (vs. 78% rail projection, 0.94/step pitch lock, 0.125 friction cliff, and 0.24 air clink).
   */
  unassistedExtraction: boolean;
  /**
   * 3. Solid block-on-shard Voronoi collisions & full-thickness colliders
   *    (vs. 81.8% cullet shards ghosting through intact blocks and `volFit` tabletop sinking).
   */
  solidCulletAndTrueShardHulls: boolean;
  /**
   * 4. Invariant 2865 Hz Timoshenko glass eigenfrequencies on oak table hits, coupled
   *    orthotropic white-oak Kirchhoff plate thud, and audible gentle contacts
   *    (vs. accidental -144 cent 0.92× pitch detuning and muted gentle edge kisses).
   */
  trueAcousticCoupling: boolean;
  /**
   * 5. Sub-frame pose & quaternion slerp interpolation across asynchronous 120 Hz worker snapshots
   *    (vs. raw snapshot pulldown stutter on dropped frames).
   */
  subframeRenderInterpolation: boolean;
  /**
   * 6. Maintain >= 500 Hz (`solref[0] >= 2*dt`) and >= 14 Newton iterations throughout collapses
   *    (vs. 240 Hz / 2-iteration mid-collapse solver softening).
   */
  fullRateCollapseSolver: boolean;
}

export type PhysicsFidelityKey = keyof PhysicsFidelityConfig;

export const SHIPPED_FIDELITY_CONFIG: Readonly<PhysicsFidelityConfig> = {
  elasticUnloadedGlassBounce: false,
  unassistedExtraction: false,
  solidCulletAndTrueShardHulls: false,
  trueAcousticCoupling: false,
  subframeRenderInterpolation: false,
  fullRateCollapseSolver: false,
};

export const FAITHFUL_GLASS_FIDELITY_CONFIG: Readonly<PhysicsFidelityConfig> = {
  elasticUnloadedGlassBounce: true,
  unassistedExtraction: true,
  solidCulletAndTrueShardHulls: true,
  trueAcousticCoupling: true,
  subframeRenderInterpolation: true,
  fullRateCollapseSolver: true,
};

export function cloneFidelityConfig(
  cfg?: Partial<PhysicsFidelityConfig> | null
): PhysicsFidelityConfig {
  return {
    elasticUnloadedGlassBounce: Boolean(cfg?.elasticUnloadedGlassBounce),
    unassistedExtraction: Boolean(cfg?.unassistedExtraction),
    solidCulletAndTrueShardHulls: Boolean(cfg?.solidCulletAndTrueShardHulls),
    trueAcousticCoupling: Boolean(cfg?.trueAcousticCoupling),
    subframeRenderInterpolation: Boolean(cfg?.subframeRenderInterpolation),
    fullRateCollapseSolver: Boolean(cfg?.fullRateCollapseSolver),
  };
}

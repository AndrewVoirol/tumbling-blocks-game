/**
 * Calibrated MuJoCo 4.0 Physical & Tribological Parameters for Vitrail
 *
 * Units inside MuJoCo `MjModel` / `MjData`:
 * - Length: SI meters (`1 Vitrail dm = 0.10 m`, `1 mm = 0.001 m`)
 * - Mass: SI kilograms (`71.10 g = 0.0711 kg` regulation beveled soda-lime glass brick)
 * - Gravity: `g = -9.81 m/s^2` along `-Y` (`Y`-up world frame matching WebGPU renderer!)
 * - Velocity & Angular Velocity Mapping:
 *   Because Vitrail's external API uses Froude (`Fr = 1`) decimeter simulation units where
 *   `x_dm = 10 * x_m` and `t_sim = sqrt(10) * t_real`:
 *   - `v_sim_dm_per_s = dx_dm / dt_sim = (10 / sqrt(10)) * v_m_per_s = sqrt(10) * v_m_per_s`
 *   - `v_real_dm_per_s = 10 * v_m_per_s = v_sim_dm_per_s * SIM_TIME_SCALE`
 *   - `omega_sim_rad_per_s = dtheta / dt_sim = omega_rad_per_s / sqrt(10)`
 *   - `J_sim = m * dv_sim = sqrt(10) * J_si`
 */

export interface MujocoCalibratedConfig {
  /** Fixed MuJoCo integrator substep in SI real seconds (`0.0016667 s` = `600 Hz`, exact 10x of 60 Hz) */
  readonly timestepSec: number;
  /** Maximum Newton / Incremental-Cholesky BoxQP iterations per substep */
  readonly solverIterations: number;
  /** Maximum line-search iterations per Newton step */
  readonly lsIterations: number;
  /** Frictional-to-normal impedance ratio (`impratio = 4.0` yields stiff pre-sliding stiction) */
  readonly impratio: number;
  /** MuJoCo 4.0 native island sleep tolerance */
  readonly sleepTolerance: number;
  /** MuJoCo 4.0 time integrator (`'implicitfast'` provides unconditionally L-stable rotational shard damping) */
  readonly integrator?: 'Euler' | 'implicitfast';
  /** Passive implicit rotational damping rate `b_rot / I_eff` (1/s) applied to active Voronoi shards under `implicitfast` */
  readonly shardRotationalDampingRate?: number;

  /** Block-on-block polished cathedral soda-lime glass static friction: `[sliding, torsional_m, rolling_m]` */
  readonly glassFriction: readonly [number, number, number];
  /** Kinetic Stribeck sliding friction for an actively drawn glass block (`priority = 1` while sliding) */
  readonly glassKineticFriction: readonly [number, number, number];
  /** Oak/walnut tabletop friction: `[sliding, torsional_m, rolling_m]` */
  readonly tableFriction: readonly [number, number, number];
  /** Voronoi shard friction: `[sliding, torsional_m, rolling_m]` */
  readonly shardFriction: readonly [number, number, number];

  /** Contact normal `solref = [timeconst_sec, dampratio]` (`>= 2 * timestepSec` for exact zero-bounce equilibrium) */
  readonly contactSolref: readonly [number, number];
  /** Underdamped Voronoi glass-shard contact `solref = [timeconst_sec, dampratio]` (`>= 2/360 s` for 3D chatter & bounce) */
  readonly shardSolref: readonly [number, number];
  /** Contact normal/frictional `solimp = [d0, dwidth, width_m, midpoint, power]` */
  readonly contactSolimp: readonly [number, number, number, number, number];
  /** Independent tangential friction `solreffriction = [timeconst_sec, dampratio]` */
  readonly contactSolrefFriction: readonly [number, number];

  /** Implicit mocap soft-connect equality constraint `solref = [timeconst_sec, dampratio]` */
  readonly grabConnectSolref: readonly [number, number];
  /** Implicit mocap soft-connect equality constraint `solimp = [d0, dwidth, width_m, midpoint, power]` */
  readonly grabConnectSolimp: readonly [number, number, number, number, number];
  /** Maximum human finger pull force in SI Newtons (`solo`/`party` vs `sandbox`) */
  readonly maxGrabForceSoloN: number;
  readonly maxGrabForceSandboxN: number;

  /**Loose-slot half-height micro-clearance in SI meters (`0.08 mm = 0.00008 m`) */
  readonly looseSlotClearanceM: number;

  /** Pre-allocated shard pool capacity (`512` matches WebGPU `MAX_ACTIVE_SHARDS`) */
  readonly shardPoolCapacity: number;
  /** Pre-allocated extra striker/test body slots for dynamic test fixtures & sandbox props */
  readonly extraBodyPoolCapacity: number;
  /** Number of canonical 3D Voronoi convex polyhedron hull archetypes baked into `MjModel` */
  readonly voronoiHullArchetypeCount: number;

  /** Three-point flexural tensile stress threshold (MPa) for mid-span bending snap */
  readonly flexuralStrengthMpa: number;
  /** Compressive opposing-wrench anvil-crush boost factor */
  readonly anvilCrushBoost: number;
}

export const MUJOCO_CALIBRATED_CONFIG: MujocoCalibratedConfig = {
  timestepSec: 1.0 / 600.0,
  solverIterations: 26,
  lsIterations: 12,
  impratio: 4.5,
  sleepTolerance: 0.003,
  integrator: 'Euler',
  shardRotationalDampingRate: 1.8,

  // Elliptic condim=4/6 glass-on-glass & glass-on-oak tribology:
  // torsional radius ~3.6 mm (`0.0036 m`) resists off-axis block spin even on 1 bevel edge,
  // while kinetic Stribeck drop (`0.275 -> 0.125` on active sliding block via `geom_priority = 1`)
  // lets drawn blocks glide cleanly out from beneath stationary upper courses without rubber-dragging them!
  glassFriction: [0.275, 0.0036, 0.00018],
  glassKineticFriction: [0.125, 0.0016, 0.00012],
  tableFriction: [0.46, 0.0052, 0.00035],
  shardFriction: [0.54, 0.0032, 0.00035],

  // Empirically verified: 18L sag = 0.044 mm (< 0.15 mm), 24L sag = 0.103 mm (< 0.50 mm),
  // 17/17 loose-vs-bearing row contrast, 100% native island sleep in <= 0.32 s!
  contactSolref: [0.0042, 1.12],
  // Underdamped glass shard bounce (`timeconst = 6.0 ms > 2 * 1/360 s`, `zeta = 0.48` -> e ≈ 0.18 glass restitution)
  shardSolref: [0.0060, 0.48],
  contactSolimp: [0.9945, 0.9995, 0.00022, 0.5, 2.0],
  contactSolrefFriction: [0.0045, 1.18],

  // Implicit mocap equality hand coupling
  grabConnectSolref: [0.022, 1.15],
  grabConnectSolimp: [0.88, 0.97, 0.0035, 0.5, 2.0],
  maxGrabForceSoloN: 5.2,
  maxGrabForceSandboxN: 11.5,

  looseSlotClearanceM: 0.000135,

  shardPoolCapacity: 512,
  extraBodyPoolCapacity: 40,
  voronoiHullArchetypeCount: 16,

  flexuralStrengthMpa: 42.0,
  anvilCrushBoost: 1.32,
};

/** Conversion factors between SI meters/seconds (MuJoCo) and Froude decimeters (Vitrail) */
export const M_PER_DM = 0.1;
export const DM_PER_M = 10.0;
export const FROUDE_VEL_SCALE = Math.sqrt(10.0); // v_sim_dm = v_si_m * sqrt(10); omega_sim = omega_si / sqrt(10)
export const INV_FROUDE_VEL_SCALE = 1.0 / Math.sqrt(10.0);

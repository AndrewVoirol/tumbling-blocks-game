---
name: vitrail-simulation-audit
description: >-
  Continuous dynamical verification harness, adversarial physics/optics/acoustics audit matrix,
  and file-disjoint native swarm playbook for Vitrail (stained-glass tumbling blocks on MuJoCo 4.0
  WASM + WebGPU). Activate whenever auditing, tuning, or orchestrating a swarm to elevate physics,
  fracture, caustics, polariscope optics, glass acoustics, or Glazier AI playability.
---

# Vitrail Simulation Mastery & Swarm Audit Playbook

Use this playbook and its executable 9-probe dynamical harness (`scripts/audit-dynamics.mjs`) to verify physical accuracy across continuous AND multi-stage stateful trajectories—catching defects that single-shot endpoint unit tests miss—and to partition multi-agent `swarm` work cleanly without file collisions.

---

## 1. Fast Verification Pipeline (Run Before & After Any Simulation Change)

1. **9-Probe Continuous & Multi-Stage Dynamics Harness (`~7 s`, exits `1` on any regression)**:
   ```bash
   node .agents/skills/vitrail-simulation-audit/scripts/audit-dynamics.mjs
   ```
   Verifies in flight:
   - `flat26cmDrop`: Post-fracture normal rebound (`avgInitVySim > 0.8`), 3D upward shard hops (`median >= 6 mm`, `p90 >= 11 mm`), `100%` sleep.
   - `midAir15cmShardDrop`: Secondary `oak_table_shard` restitution (`median >= 2.0 mm`, `p90 >= 5.5 mm`), `100%` sleep.
   - `collapseOffRimAndClock`: `0` floating off-rim blocks/shards past circular `TABLE_RADIUS = 4.30 dm`, `0` `mjWARN_BADQACC` (`d.time`) solver resets, `stepMsAvg < 9.5 ms` (`stepMsP95 < 35 ms`), `1.00×` free-fall gravity parity (`[0.92, 1.06]`).
   - `marginalOverhangTopple`: `+3.9 cm` (`5.8°`) cantilevered 6-course upper sub-tower dynamically topples rather than freezing asleep under `sleepTolerance`.
   - `polariscopeSpatialAccuracy`: Unloaded top course `[0,0,0,0]`, loose Row-0 slot `[0,0,0,0]`, cantilevered half `< 0.02` vs. `+X` bearing fulcrum `> 0.28`.
   - `grazingPullIsolation`: Full-row grazing extraction keeps upper-course lateral drift `< 2.0 mm` and cleanly clears Layer-0 tabletop prisms.
   - `multiStageSupportLossAndTelemetry`: Multi-stage Sandbox play (shatter upper courses so Voronoi shards sleep perched at `y ~ 20–22 cm` + rest 1 crooked `22°` top-notch outlier block -> verify consensus `leanAngleDeg < 2.0°` -> buckle/shatter supporting tower underneath -> verify `0` unsupported elevated shards freeze in mid-air and HUD telemetry transitions cleanly to `currentLayers = 0` / `Toppled`).
   - `postCollapse100PctSleepAndSilence`: Sandbox `18L` collapse reaches `ntree_awake === 0` (`<= 2.4 s`) with `0` post-`1.6 s` `wobbleRock` / `glassImpact` buzz (`condim="3"` frictioned `parking_shelf`, kinetic-speed-gated `mjDSBL_MULTICCD`).
   - `fulcrumSeesawAndClampedWobbleIsolation`: `6.5°` rubble fulcrum seesaw rocks (`>= 5` flips, `>= 3` `wobbleRock` on seesaw, `0` on fulcrum) and sleeps, while clamped Layer-1 bearing blocks (`>= 1.1 N` top load) emit `0` false wobbles.
2. **MuJoCo 4.0 Calibration Scorecard (`8/8` Gates, `~5 s`)**:
   ```bash
   npm run calibrate:physics
   ```
3. **Static WGSL AST & Byte-Offset Lint + Production Build (`~1 s`)**:
   ```bash
   npm run lint:wgsl && npm run build
   ```
4. **Tier-2 Hardware Metal Optical Bench (`~4–10 s`, keep `screenshots/` clean via `/tmp/`)**:
   ```bash
   node scripts/optical-bench.mjs --scene all --out /tmp/vitrail-bench-all.png
   ```
5. **Complete `64/64` Vitest Suite (`8` suites: `T1–T9`, `C1–C8`, `6` Faithful-Glass Fidelity Checks, `520`-Seed Stateful Fuzzer, `10` iPad Touch/HUD, `9` Acoustics, `11` Multiplayer & `5` `100`-Match Chaos), `17/17` 5-Domain Pre-Ship Eval + Live Two-Browser E2E & Standalone Trajectory Fuzzer**:
   ```bash
   npx vitest run
   npm run eval:preship
   npm run verify:mp:live
   node scripts/fuzz-stateful-trajectories.mjs --seeds 500 --actions 80
   ```

---

## 2. Non-Colliding Native `swarm` Roster Partition

When running a native `swarm` (`workspace: "share"` for concurrent authors, `workspace: "inherit"` for verifiers), enforce strict file-disjoint ownership and serialize any shared buffer-layout change (`BLOCK_FLOATS = 52`, `SHARD_SLOT_STRIDE_VERTS = 132`, Worker `postMessage` schema) before parallel fan-out:

| Worker Name | Owned Paths (Strict) | Core Simulation Mandate |
| :--- | :--- | :--- |
| `physics-tribology` | `src/physics/*` (`MujocoPhysicsEngine.ts`, `mujocoTowerBuilder.ts`, `mujocoCalibratedParams.ts`, `mujocoWorldCompat.ts`, `voronoiFracture.ts`) | Contact mechanics (`condim=4/6` elliptic cones, Stribeck kinetic sliding), mid-air vs. anvil-crush vs. three-point flexural Griffith fracture, `512`-slot Voronoi hull fidelity, `dof_M0` / Euler stability, off-rim & settle dynamics. |
| `optics-caustics` | `src/renderer/*` (`WebGPURenderer.ts`, `shaders/*.wgsl.ts`) | Cross-pass optical continuity across intact -> crazed -> shattered Voronoi shard states: caustic transmission in `causticPass.wgsl.ts`, Polariscope dark-field response in `glassTower.wgsl.ts` & `glassShard.wgsl.ts`, internal crazing crack optics, XDR/SDR tone mapping. |
| `gameplay-audio-ai` | `src/engine/*`, `src/audio/*`, `src/App.tsx`, `src/ui/*` | Secondary shard-on-table & shard-on-glass crystalline acoustic chattering in `GlassAudioEngine.ts`, steep/grazing camera pointer raycast & mocap pull feel in `GameEngine.ts` / `OrbitCamera.ts`, and `Glazier AI` multi-turn tactical depth at `20L–28L+`. |
| `adversarial-verifier` | `tests/*`, `.agents/skills/vitrail-simulation-audit/scripts/*`, `scripts/*` (+ live Chrome DevTools MCP) | Never edits `src/*`. Writes/runs falsification probes (`audit-dynamics.mjs`), live `120 Hz` browser DOM gesture traces, Polariscope/Natural mode hot-swaps, and temporal collapse strips; reports concrete metric regressions to `@<owner>`. |

> **Mandatory Worker Brief Header**: Instruct every spawned worker to read `AGENTS.md`, `/Users/andrewvoirol/.gemini/config/skills/webgpu-interactive-simulation/SKILL.md` (plus `references/mujoco-rigidbody-fracture-tribology.md` for physics workers), and `.agents/skills/vitrail-simulation-audit/SKILL.md` on Turn 0 before reading or editing code.

---

## 3. Mandatory Regression Invariants (Anti-Goodharting Checklist)

1. **Force-Verified Shard Support & Parked-Pool Hygiene (`MujocoPhysicsEngine.ts`, `mujocoWorldCompat.ts`)**:
   - Above `y = 2.6 cm`, never gate shard velocity settle or `zeroSnap` on elevation/age without `d.qfrc_constraint[dofAdr + 1] > bodyMass * 1.5` (`shardNoSupportSubsteps <= 4`).
   - Guard every block/shard pool wakeup with `if (geomContype[geomId] === 0) continue;` so parked hidden-shelf bodies stay asleep (`stepMsP95 < 16 ms`), require strictly lower CoM (`dy > 0.06 dm`) in unsupported-sleep sweeps, and keep `MAX_FRACTURES_PER_SUBSTEP = 2` (`deferredFractures` queue).
2. **Two-Sided Polariscope Translucency & Caustic Continuity (`causticPass`, `glassTower`, `glassShard`)**:
   - Verify both `uCausticTex.a` Voronoi shard caustic ribbons on the oak table when `intactCount === 0` AND smoked neutral-density crystal refraction (`~30%` passband background visibility on unloaded prisms/shards + `0.11` dark-field trough extinction inside stressed courses)—never pass Polariscope dark-field by crushing transmission into `< 1%` matte-black silhouettes, and never pass negative bases to WGSL `pow()`.
3. **Outlier-Trimmed `Plumb`, Sandbox `Toppled` Parity & Multiplayer Seed Sync (`App.tsx`, `src/multiplayer/*`)**:
   - Keep internal `solverLeanAngleDeg` separate from HUD `leanAngleDeg` (`standingTilts[tiltLen - 5]` column consensus when stationary, `maxDynamicTipTiltRad` when tipping), assert `isToppled` (`— Toppled`) on collapsed Sandbox heaps (`currentLayers === 0 && shatteredCount > 0`), and preserve sequential `mulberry32` `54/54` Host/Worker/Guest block palette parity.

---

## 4. Eye-and-Ear Realism & Two-Sided Physics/Audio Invariants (`M1–M6` + `eval:preship`)

1. **Unloaded Rebound vs. Spawn Silence (`M1`)**: Assert `>= 1.15–1.7 mm` crisp rebound on `10–12 mm` unloaded oak drops paired `1:1` with `hasOnsetContact` impulses (`0` timer buzz), while gating only `< 0.30 s` initial stack spawn audio (`inInitialTowerSpawnSettle`, `settleClinks = 0`) without damping physics across `12L–30L` towers.
2. **Free Extraction Tribology, Cullet Masks & Substrate Audio (`M2–M4`)**: Reject artificial slot-rail/pitch locks (`78%` / `0.94`), kinetic drop below `μk = 0.215`, phantom air clinks, `0x20` ghosting cullet (require `0x22` on-table / `0x02` off-rim), and `-8%` oak modal detuning (`2865 Hz` glass prism modes stay substrate-invariant + coupled `95–135 Hz` white-oak plate thud).
3. **Worker Interpolation, Multi-Domain Fuzzer & Two-Sided `multiccd` (`M5–M6`)**: Verify `getRenderPoseAlpha` slerp (`120 Hz` parity), `>= 500 Hz / >= 14` iters collapse stability on MT, static `0.275` breakaway friction inside `mj_copyState` rollouts, Glazier AI `currentLayers - 5` top-course guard + `>= 0.30 s && movedDm < 0.32` bind abort, 3D `speed3d < 0.12` off-rim checks, and two-sided `mjDSBL_MULTICCD` bracketing (`!(solverLeanAngleDeg < 8.5° && shards.length < 64)` onset AND `visibleAwakeShards <= 48` rubble settle).

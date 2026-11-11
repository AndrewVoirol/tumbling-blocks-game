---
name: vitrail-simulation-audit
description: >-
  Continuous dynamical verification harness, adversarial physics/optics/acoustics audit matrix,
  and file-disjoint native swarm playbook for Vitrail (stained-glass tumbling blocks on MuJoCo 4.0
  WASM + WebGPU). Activate whenever auditing, tuning, or orchestrating a swarm to elevate physics,
  fracture, caustics, polariscope optics, glass acoustics, or Glazier AI playability.
---

# Vitrail Simulation Mastery & Swarm Audit Playbook

Use this playbook and its executable 7-probe dynamical harness (`scripts/audit-dynamics.mjs`) to verify physical accuracy across continuous AND multi-stage stateful trajectories—catching defects that single-shot endpoint unit tests miss—and to partition multi-agent `swarm` work cleanly without file collisions.

---

## 1. Fast Verification Pipeline (Run Before & After Any Simulation Change)

1. **7-Probe Continuous & Multi-Stage Dynamics Harness (`~6 s`, exits `1` on any regression)**:
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
5. **Complete `34/34` Vitest Suite (`4` suites: `T1–T9`, `C1–C6`, Acoustics & Multiplayer, `~44 s`)**:
   ```bash
   npx vitest run
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

## 3. High-Leverage Adversarial Audit Targets for Simulation Mastery

Use these concrete cross-subsystem questions to drive adversarial inspection and upgrades:

1. **Cross-Pass Shard Optical Continuity (`causticPass.wgsl.ts` & `glassShard.wgsl.ts`)**:
   - Currently `causticPass.wgsl.ts` projects oblique sun caustics from intact `uBlockCenters` blocks. When a block shatters into 20+ stained-glass Voronoi shards on the oak table, verify whether colored caustic patches abruptly vanish or whether resting/tumbling shards also refract colored caustic pools onto the oak surface.
   - Verify `glassShard.wgsl.ts` under dark-field **Polariscope** mode (`uBlockCenters[97].w > 0.5`): intact blocks switch to dark-field cross-polarized transmission with stress fringes—ensure shattered shards exhibit physically consistent dark-field extinction / residual cleavage-edge stress rather than glowing as unpolarized bright blocks.
2. **Post-Shatter Secondary Acoustic Chattering (`GlassAudioEngine.ts` <-> `MujocoPhysicsEngine.ts`)**:
   - Initial fracture triggers `playGlassShatter`, while `oak_table_shard` (`shardSolref = [0.0060, 0.48]`) now produces genuine secondary 3D shard hops (`3–17 mm`) over `150–350 ms`. Verify whether secondary shard-table and shard-block impacts emit rate-limited, mass/pitch-scaled crystalline tinkle grains (`f ~ m^-1/3`) that decay naturally into silence as `ntree_awake -> 0`.
3. **Progressive Internal Crazing & Multi-Hit Structural Memory**:
   - Inspect how `block.damage` and `block.crackLocalOrigin` accumulate across sub-shatter impacts (`effCrazeSpeed = 5.33 dm/s`), whether `glassTower.wgsl.ts` spiderweb cleavage planes orient physically toward `crackLocalOrigin` with proper Fresnel internal reflection, and whether weakened crazed blocks exhibit lower flexural/impact fracture thresholds on subsequent collisions.
4. **Tall-Stack (`22L–28L`) Tribology, Mocap Pull Ergonomics & Glazier AI Strategy**:
   - Test real DOM pointer pulls across steep top-down, grazing horizontal, and off-center camera angles, Layer-0 bottom pulls, single-center-block supporting rows, and counterbalance top-row placement.
   - Verify `evaluateCounterfactualBlockCriticality()` and `updateMasterGlazierAi` across full 10-turn rallies so AI placement accurately counterbalances lean without edge overhang clips or stall states.
5. **Peak Collapse Burst Step Budget (`stepMsP95 ~ 27 ms -> < 16 ms`)**:
   - While `stepMsAvg` across a 260-frame 21-block collapse is `~5.5 ms`, `stepMsP95` spikes to `~27 ms` on frames where 4–5 blocks run `fractureBlockVoronoi()` + `d.contact.get(i)` across ~400 contacts simultaneously. Profile whether skipping shard-shard `contacts.get(i)` Embind wrapper allocation inside `processWrenchCollisionsAndFractures` (e.g., capping fractures to `<= 2` blocks per substep or early-filtering before `efc` math) brings `stepMsP95` under `16.6 ms`.

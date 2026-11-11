# `tumbling-blocks-game` (Vitrail) — Workspace Agent Runbook & Invariants

## 0. Fast Triage & Anti-Relearning Runbook

1. **Mandatory Turn-0 Skill Read**: Read `/Users/andrewvoirol/.gemini/config/skills/webgpu-interactive-simulation/SKILL.md` before designing, editing, or delegating any WGSL shader, optical pass, fracture geometry, or MuJoCo physics solver changes.
2. **Core Commands**:
   - **Physics, Tribology & Voronoi Fracture Suite (`11/11` `T1–T8` Invariants)**: `npx vitest run` (or `npm run test:blast -- <file-or-symbol>`)
   - **Offline MuJoCo 4.0 Calibration & Invariant Scorecard (`~7 s`)**: `npm run calibrate:physics` (`node scripts/calibrate-mujoco-physics.mjs`)
   - **Static WGSL AST, Struct Alignment & Control-Flow Lint**: `npm run lint:wgsl`
   - **Symbol Line-Number Indexer (Before Reading Monoliths)**: `npm run map:symbols -- <keyword>`
   - **Typecheck + Production Bundle (`TS 7` + `Vite 8` Rolldown)**: `npm run build`
   - **Tier-2 Hardware Metal Optical Bench (`Float32` HDR Readback + Contact Sheet)**:
     - Single scene (`~4 s`): `node scripts/optical-bench.mjs --scene two-block-cross --out /tmp/vitrail-bench.png`
     - Full 4-scene optical matrix (`~10 s`): `node scripts/optical-bench.mjs --scene all --out /tmp/vitrail-bench-all.png`
3. **Media & Git Hygiene**:
   - Always pass `--out /tmp/...` when running `scripts/optical-bench.mjs` during development so committed `screenshots/*.png` and `screenshots/demo.gif` remain clean (`git diff --stat screenshots/` empty) unless deliberately regenerating portfolio media.
   - Enforce `<= 2` edit-and-capture cycles before surfacing side-by-side optical metrics to the user.

---

## 1. WebGPU Multi-Pass Optical Architecture (`src/renderer/`)

- **Renderer Orchestrator**: [`src/renderer/WebGPURenderer.ts`](src/renderer/WebGPURenderer.ts) — single-encoder `rgba16float` HDR pipeline with `4x` MSAA, `timestamp-query` GPU pass profiling, and linear `Float32` HDR resolve readback (`readHdrResolveFloat32()`).
- **Shader Modules (`src/renderer/shaders/*.wgsl.ts`)**:
  - [`commonGlass.wgsl.ts`](src/renderer/shaders/commonGlass.wgsl.ts): Shared Beer-Lambert spectral extinction, dielectric Fresnel, Abbe chromatic dispersion, and copper-foil solder came helpers (`COMMON_GLASS_WGSL`).
  - [`causticPass.wgsl.ts`](src/renderer/shaders/causticPass.wgsl.ts): Oblique `2048 × 2048` sun transmission & waterglass caustic projection onto the beveled oak table.
  - [`environmentPass.wgsl.ts`](src/renderer/shaders/environmentPass.wgsl.ts): Studio backdrop and caustic-lit oak table surface.
  - [`glassTower.wgsl.ts`](src/renderer/shaders/glassTower.wgsl.ts): Raytraced beveled regulation Jenga blocks (`n = 1.522`, `BLOCK_FLOATS = 52` / `208 B` `GPUBlockData`) with multi-block interior refraction (`<= 2` interior blocks), Total Internal Reflection, 3D seed air bubbles, internal crazing spiderweb cracks, and 3-sample ray-chord Brewster stress-optic birefringence (`photoelastic: vec4<f32>` = local `-X`, `0`, `+X` through-thickness bearing stress + Euler-Bernoulli flexural bending, toggled into dark-field **Polariscope** mode via `uBlockCenters[97].w`).
  - [`glassShard.wgsl.ts`](src/renderer/shaders/glassShard.wgsl.ts): Convex 3D Voronoi glass shards (`MAX_ACTIVE_SHARDS = 512`, `SHARD_SLOT_STRIDE_VERTS = 132`) differentiating outer leaded faces from inner conchoidal cleavage planes.
  - [`overlayPass.wgsl.ts`](src/renderer/shaders/overlayPass.wgsl.ts) & [`postProcess.wgsl.ts`](src/renderer/shaders/postProcess.wgsl.ts): Top-row placement guides (`depthCompare: 'less-equal'`, `depthWriteEnabled: false`) and ACES HDR tone mapping.
- **Swapchain Color Contract**: [`postProcess.wgsl.ts`](src/renderer/shaders/postProcess.wgsl.ts) compiles dual pipelines (`pipeCompositeHdr` with `HDR_EXTENDED = 1` on `rgba16float` + `toneMapping: { mode: 'extended' }` for Apple Silicon XDR displays, and `pipeCompositeSdr` with `HDR_EXTENDED = 0` on `bgra8unorm` / `rgba8unorm` for SDR monitors), switching automatically on `window.matchMedia('(dynamic-range: high)')` or via `renderer.setExtendedHdr(boolean)`.

---

## 2. MuJoCo 4.0 WASM Physics, Tribology, Fracture & Worker Architecture (`src/physics/`)

- **Google DeepMind `@mujoco/mujoco@^4.0.0` Runtime & Multi-Threading**:
  - [`src/physics/mujocoRuntime.ts`](src/physics/mujocoRuntime.ts) loads `@mujoco/mujoco/mt` (`4` WASM pthreads via `mju_threadpool` when `crossOriginIsolated === true` via `COOP: same-origin` + `COEP: credentialless` in [`vite.config.ts`](vite.config.ts), [`public/_headers`](public/_headers), and [`vercel.json`](vercel.json)) with automatic single-threaded fallback.
  - [`src/physics/mujocoWorker.ts`](src/physics/mujocoWorker.ts) & [`src/physics/MujocoWorkerBridge.ts`](src/physics/MujocoWorkerBridge.ts) run physics and counterfactual AI rollouts inside a dedicated ES Module Web Worker on desktop and iPad Pro (`0.0 ms` main-thread `rAF` stall; bypass via `?worker=0` or `?bench=...` for deterministic synchronous captures).
- **Full-Stack Newton Solver, Elliptic `condim=4/6` Tribology & Implicit Mocap Hand**:
  - [`src/physics/mujocoTowerBuilder.ts`](src/physics/mujocoTowerBuilder.ts) & [`src/physics/mujocoCalibratedParams.ts`](src/physics/mujocoCalibratedParams.ts): `<size memory="24M"/>`, `jacobian="sparse"`, `solver="Newton"`, `cone="elliptic"`, `condim="4"` (blocks) / `"6"` (oak table), `multiccd="enable"`, `sleep="enable"`, `timestepSec = 1/600`, 24-vertex beveled prism hulls (`71.1 g`), kinematic `mocap` hand + implicit soft `<connect>` equality pull, and a `512`-slot `sleep="init"` Voronoi shard pool (`16` convex hull archetypes, `< 0.2 ms` activation).
  - **Euler Shard Damping & Free-Fall Invariant**: Never write explicit `m.dof_damping` on `0.2 g` Voronoi shards (`I ~ 10^-8 kg·m^2`) under `integrator="Euler"` ($1 - b\Delta t / I \ll 0$ explodes), and never damp elevated intact blocks (`y >= 2.8 cm`) during collapse. Hoist Embind typed-array views (`d.qpos`, `d.qvel`, `d.xpos`, `m.geom_bodyid`) outside per-body/per-contact loops so JS wrapper overhead stays `< 0.5 ms/frame`.
  - **Sleep Tolerance, Stribeck Sliding & Sleeping Clamping-Load Invariants**: Keep `sleepTolerance <= 0.003` so cantilevered overhangs dynamically topple instead of freezing asleep mid-air; elevate `geom_priority[gid] = 1` toward `glassKineticFriction` (`0.125`) only on actively sliding blocks so upper courses are not rubber-dragged sideways; and capture `cachedClampingLoadN` + 4-component `blockPhotoelasticWrench` inside [`MujocoPhysicsEngine.ts`](src/physics/MujocoPhysicsEngine.ts) while `d.ntree_awake > 0` before native island sleep sets `d.ncon = 0`.
- **Counterfactual `mj_copyState` Oracle & Master Glazier AI**:
  - `evaluateCounterfactualBlockCriticality()` clones `MjData` into an isolated scratch buffer (`0` live state mutation) to score extraction ease vs. tower lean risk, driving the playable **Pass & Play -> `AI`** (`Glazier AI`) opponent in [`src/engine/GameEngine.ts`](src/engine/GameEngine.ts).
- **Hard `T1–T8` Physical Tolerance Gates ([`tests/physicsAndFracture.test.ts`](tests/physicsAndFracture.test.ts))**:
  - `T1`: `18L` sag `< 0.15 mm` (`0.044 mm` calibrated), `24L` sag `< 0.50 mm` (`0.103 mm` calibrated), `100%` sleep (`54/54` & `72/72` bodies), loose-vs-bearing row contrast.
  - `T2`: `1.0 cm` placement drop and `1.2 g` micro-splinter at `18 dm/s` incur `0` false crazing/shatter damage.
  - `T3`: Opposing-wrench anvil-crush, `+X` tip argmax fracture origin, `< 1%` linear/angular momentum error, and `512`-slot ring-buffer stress.
  - `T4` / `T4b`: `< 5 px` tap-test isolation, `11°` grazing pull extraction, preserved clamping loads after aborted pulls, and `24L` upward grazing pitch.
  - `T5` / `T6` / `T7`: Asymmetric collapse (`>= 15` intact + `>= 4` crazed + `100%` sleeping survivors), bevel-to-bevel side-abutment, counterbalance lean, and mid-drag/sandbox parallax guards.
  - `T8`: Elliptic `condim=4` torsional spin resistance, photoelastic force-chain stress contrast, three-point flexural mid-span snap, and zero-mutation `mj_copyState` counterfactual oracle discrimination.

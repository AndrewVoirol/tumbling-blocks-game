# `tumbling-blocks-game` (Vitrail) — Workspace Agent Runbook & Invariants

## 0. Fast Triage & Anti-Relearning Runbook

1. **Mandatory Turn-0 Skill Read**: Read `/Users/andrewvoirol/.gemini/config/skills/webgpu-interactive-simulation/SKILL.md` before designing, editing, or delegating any WGSL shader, optical pass, fracture geometry, or physics solver changes.
2. **Core Commands**:
   - **Physics & Voronoi Fracture Suite (`9/9` `T1–T6` Invariants)**: `npx vitest run` (or `npm run test:blast -- <file-or-symbol>`)
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
  - [`glassTower.wgsl.ts`](src/renderer/shaders/glassTower.wgsl.ts): Raytraced beveled regulation Jenga blocks (`n = 1.522`) with multi-block interior refraction (`<= 2` interior blocks), Total Internal Reflection, 3D seed air bubbles, and internal crazing spiderweb cracks.
  - [`glassShard.wgsl.ts`](src/renderer/shaders/glassShard.wgsl.ts): Convex 3D Voronoi glass shards (`MAX_ACTIVE_SHARDS = 512`, `SHARD_SLOT_STRIDE_VERTS = 132`) differentiating outer leaded faces from inner conchoidal cleavage planes.
  - [`overlayPass.wgsl.ts`](src/renderer/shaders/overlayPass.wgsl.ts) & [`postProcess.wgsl.ts`](src/renderer/shaders/postProcess.wgsl.ts): Top-row placement guides (`depthCompare: 'less-equal'`, `depthWriteEnabled: false`) and ACES HDR tone mapping.
- **Swapchain Color Contract**: [`postProcess.wgsl.ts`](src/renderer/shaders/postProcess.wgsl.ts) currently outputs ACES-tonemapped, gamma-encoded sRGB `[0, 1]` to a `bgra8unorm` swapchain. Upgrading to Apple Silicon XDR (`rgba16float` + `toneMapping: { mode: 'extended' }`) requires branching `postProcess.wgsl.ts` to emit **linear** extended-sRGB (`> 1.0` specular/caustic headroom without double-gamma washout).

---

## 2. Physics, Fracture & Rapier Solver Pin (`src/physics/`)

- **1:1 Regulation Scale & Froude (`Fr = 1`) Scaling**: [`src/physics/constants.ts`](src/physics/constants.ts) models `7.5 × 2.5 × 1.5 cm` (`71.1 g` beveled soda-lime glass) across `18` rows (`27.0 cm`, `54` blocks).
- **Volume-Conserving 3D Voronoi Fracture**: [`src/physics/voronoiFracture.ts`](src/physics/voronoiFracture.ts) clips blocks via 3D Sutherland-Hodgman half-spaces (`< 0.05%` volume error across 200 deterministic seeds, `<= 132` vertices per shard).
- **Hard `T1–T6` Physical Tolerance Gates ([`tests/physicsAndFracture.test.ts`](tests/physicsAndFracture.test.ts))**:
  - `T1`: `18L` sag `< 0.15 mm` (`0.388 mm` fails!), `24L` sag `< 0.50 mm`, `100%` sleep (`54/54` bodies), exact loose-slot micro-clearance distribution.
  - `T2`: `1.0 cm` placement drop and `1.2 g` micro-splinter at `18 dm/s` incur `0` false crazing/shatter damage.
  - `T3`: Dynamic anvil-crush, `+X` tip argmax fracture origin, `< 1%` linear momentum conservation error, and `512`-slot ring-buffer stress.
  - `T4` / `T4b`: `< 5 px` tap-test isolation, `11°` grazing pull extraction, preserved clamping loads after aborted pulls, and `24L` upward grazing pitch.
  - `T5` / `T6`: Asymmetric collapse (`>= 15` intact + `>= 4` crazed + precarious survivors sleeping `100%`), bevel-to-bevel side-abutment, and counterbalance lean physics.
- **Why `@dimforge/rapier3d-compat` Is Pinned to `^0.14.0`**:
  - Empirically tested against `@dimforge/rapier3d-compat@0.21.0`: Rapier `0.15+` replaced velocity PGS (`erp`) with the TGS Soft regularized contact solver, increasing `18L` stack sag from `< 0.15 mm` to `0.388 mm` (`T1`) and raising peak transient contact-force impulses so `1.2 g` micro-splinters trigger false shatters (`T2`, `T5` intact count `4 < 15`). Do **not** bump `@dimforge/rapier3d-compat` past `0.14.x` unless jointly recalibrating `integrationParameters` (`contact_natural_frequency`, `numSolverIterations`) and `CRAZE_IMPACT_SPEED` / `SHATTER_IMPACT_SPEED` in a dedicated physics Round-2 pass.

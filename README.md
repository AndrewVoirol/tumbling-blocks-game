# Vitrail

![Vitrail gameplay demo](screenshots/demo.gif)

Stained-glass tumbling blocks built on raw WebGPU and Google DeepMind's `@mujoco/mujoco` (`4.0.0` `/mt` WASM). Every prism is modeled to 1:1 regulation scale (`7.5 × 2.5 × 1.5 cm`, `71.1 g` beveled soda-lime glass across 18 courses = `27.0 cm`) atop a quarter-sawn white oak pedestal table, raytraced through multi-block dielectric refraction with projected Snell waterglass caustics, cross-polarized Brewster stress-optic birefringence, 3D Voronoi glass shattering, procedural modal acoustics, counterfactual **Glazier AI**, and 2-player **Online Rooms**.

**[Play Live at vitrail.andrewvoirol.com →](https://vitrail.andrewvoirol.com)**

## Quick Start

Requires Node.js 20+ and a WebGPU browser (Chrome 113+, Edge 113+, or Safari 18+).

```bash
git clone https://github.com/andrewvoirol/tumbling-blocks-game.git
cd tumbling-blocks-game
npm install
npm run dev
```

## In Action

### 18-Course Equilibrium & Projected Stained-Glass Caustics
![Initial 18-course stained-glass tower](screenshots/initial-state.png)

### Pass & Play vs. Glazier AI & Spring-Tethered Extraction
![Extracting a stained-glass prism against Glazier AI](screenshots/mid-turn-stack.png)

### Cascade Collapse & 3D Voronoi Glass Fracture
![Shattered stained-glass prisms and Voronoi shards on the oak table](screenshots/shattered-collapse.png)

## How It Works

1. **Scout loose glass by ear**: Tap (`< 5 px`) any legal prism below the top completed course without committing your turn. Each 3-prism course has physical micro-tolerance clearance on loose slots versus bearing-clamped columns, shifting Winkler-stiffened Timoshenko bar resonance from a free `~2.86 kHz` ring to a damped `~4.21 kHz` tick.
2. **Draw smoothly along the channel**: Drag horizontally along a prism's length to pull it via a soft equality spring tether. Watch the 12-pane tension strip and **Plumb** spirit compass—if resistance spikes and the stack leans, release before clearing the tower to pick another prism (`Escape` cancels an active pull).
3. **Stack on top & switch modes**: Position an extracted prism across the top course with bevel-to-bevel side abutment, rotate with the scroll wheel or `Q` / `90° R` / `E` (`0` aligns square), and click **Place Prism** (or press `Space` / `Enter`) to release. From the left stained-glass lancet window, switch between **Solo**, **Pass & Play** (2–4 local players or `AI` vs. counterfactual **Glazier AI**), **Online** (share a 5-character room code or `?room=` link for 2-player internet play), and **Sandbox** (unrestricted pulling, tap-to-flick shards, and one-click tower shatter), or toggle the **Polariscope** lens on the right crystal plate to inspect live isochromatic force chains.

Under the hood, a single-encoder WebGPU `rgba16float` HDR pipeline projects a `2048×2048` oblique Snell waterglass caustic map across intact blocks and 12-plane Voronoi shards onto a ring-porous quarter-sawn white oak table, raytracing beveled prisms (`n = 1.522`) through up to two interior blocks with exact Maxwell-Fresnel reflectance, Cauchy dispersion ($V_d \approx 58$), hollow seed-bubble total internal reflection, Griffith crazing planes, 3-sample Brewster photoelastic stress-optic retardation, and automatic SDR / Apple Silicon XDR (`toneMapping: 'extended'`) output. Physics runs off-thread inside a dedicated Web Worker on `@mujoco/mujoco` (`4.0.0` 4-thread WASM `mju_threadpool` when `crossOriginIsolated`) using a full-stack Newton solver, elliptic `condim=4/6` torsional and rolling friction cones, 24-vertex beveled prism hulls (`0.098 mm` calibrated 18-course sag, `54/54` island sleep), a 2-ply Cross-Entropy Method `mj_copyState` counterfactual rollout oracle for Glazier AI, and 6D contact-wrench Griffith fracture clipping blocks into a `512`-slot convex 3D Voronoi shard pool (`58/58` physical, acoustic, touch, stateful-fuzz, and network-chaos invariant tests). Procedural Web Audio synthesizes Cowper-Timoshenko flexural/torsional/longitudinal bar modes, Hertzian velocity-bandwidth impact filters, orthotropic circular oak Kirchhoff plate modes, accelerating Housner bevel rocking ticks, and Stribeck stick-slip singing, while 2-player Online Rooms stream `< 4 KB` `0x564d` binary `30 Hz` snapshots (`quatSlerpHemisphere` `60–120 Hz` interpolation + round-robin `512`-shard collapse parity) over PeerJS WebRTC `RTCDataChannel` with automatic `<= 4.0 s` failover to a Cloudflare Durable Object SQLite WebSocket relay.

For full governing equations, numerical discretizations, and falsification benchmarks, see [**The Science and Mathematics of Vitrail**](docs/SCIENCE_AND_MATHEMATICS_OF_VITRAIL.md) and the [**Adversarial Colosseum Audit Report**](docs/COLOSSEUM_AUDIT_REPORT.md).

## Tech Stack

| Layer | Technology |
| --- | --- |
| Rendering | Raw WebGPU + WGSL (`rgba16float` HDR, 4× MSAA, `2048²` Snell block & shard caustics, Brewster Polariscope, SDR / Apple XDR extended tone mapping) |
| Physics & Fracture | Google DeepMind `@mujoco/mujoco` (`4.0.0` `/mt` WASM + Web Worker, elliptic `condim=4/6` Newton solver, 2-ply CEM `mj_copyState` AI, `512`-slot 3D Voronoi pool) |
| Acoustics | Procedural Web Audio API (Timoshenko thick-bar modes, Hertzian impact cutoff, orthotropic oak Kirchhoff plate, Housner rocking & Stribeck stick-slip) |
| Multiplayer | Dual-transport WebRTC `RTCDataChannel` (`0.peerjs.com`, `< 4 KB` `0x564d` binary frames) + Cloudflare Durable Object WebSocket relay (`<= 4.0 s` auto-failover) |
| UI & Build | React 19, inlaid SVG stained-glass & etched crystal HUD, TypeScript 7, Vite 8 (Rolldown), Vitest |

# Vitrail

![Vitrail gameplay demo](screenshots/demo.gif)

Stained-glass Jenga built from scratch on WebGPU and Rapier3D. Every block is modeled to 1:1 regulation scale (`7.5 × 2.5 × 1.5 cm`, `70.9 g` soda-lime glass across 18 rows = `27.0 cm`), leaded with copper-foil solder cames, and raytraced through multi-block dielectric refraction with projected caustics and 3D Voronoi glass shattering.

## Quick Start

Requires Node.js 20+ and a WebGPU browser (Chrome 113+, Edge 113+, or Safari 18+).

```bash
git clone https://github.com/andrewvoirol/tumbling-blocks-game.git
cd tumbling-blocks-game
npm install
npm run dev
```

## In Action

### 18-Row Equilibrium & Projected Stained-Glass Caustics
![Initial 18-row stained-glass tower](screenshots/initial-state.png)

### Pass & Play Extraction & Top-Row Placement
![Extracting and stacking a stained-glass block](screenshots/mid-turn-stack.png)

### Cascade Collapse & 3D Voronoi Glass Fracture
![Shattered stained-glass blocks and Voronoi shards](screenshots/shattered-collapse.png)

## How It Works

1. **Tap-test before you commit**: Click any legal block below the top complete row to poke it slightly. Each row has physical micro-tolerance variance (~`0.09 mm` clearance on loose blocks vs pinned load-bearing columns), so loose blocks slide visibly while pinned blocks shudder.
2. **Slide it clear**: Drag horizontally along a block's length to pull it via a physical spring-damper. Watch the rim tint and **Tower Plumb** spirit level—if resistance spikes red and the tower leans, let go and pick another block.
3. **Stack on top**: Once a block clears the tower, aim at one of the glowing placement guides on the top row and click to drop it under gravity. Switch between **Solo Tower**, **Pass & Play** (2–4 players), and **Sandbox** (unrestricted pulling + one-click tower smash) from the top-left panel, or drag the background to orbit 360°.

Under the hood, a custom single-encoder WebGPU HDR pipeline renders an oblique `2048×2048` Beer-Lambert sun transmission & waterglass caustic map onto the beveled oak table, then raytraces primary beveled blocks (`n = 1.522`) through up to two interior stained-glass blocks behind them with Abbe chromatic dispersion, total internal reflection, 3D seed air bubbles, and internal crazing cracks. Physics runs on `@dimforge/rapier3d-compat` using Froude-invariant (`Fr = 1`) scaling for sub-`0.3 mm` 18-story stack compliance at real-time `9.81 m/s²` gravity; high-velocity impacts clip parent blocks via 3D Sutherland-Hodgman Voronoi planes into exact volume-conserving convex shards paired with procedural Web Audio modal glass synthesis.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Rendering | Raw WebGPU + WGSL (`rgba16float` HDR, 4× MSAA, ACES tone mapping) |
| Physics & Fracture | `@dimforge/rapier3d-compat`, custom 3D Sutherland-Hodgman Voronoi clipper |
| Audio | Procedural Web Audio API modal glass impact & friction synthesis |
| UI & Build | React 19, TypeScript 5.8, Lucide Icons, Vite 6, Vitest |

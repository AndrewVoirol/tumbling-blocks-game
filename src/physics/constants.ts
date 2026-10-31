// Exact Jenga physical scale & Stained Glass optical material constants
// World coordinate scale: 1 world unit = 10 cm = 0.1 m (decimeters).
// Why decimeters: keeps block dimensions (0.75 x 0.15 x 0.25) in the optimal IEEE-754
// and Rapier PGS solver precision window while preserving 1:1 physical time with g = -98.1 dm/s^2.

export const CM_PER_UNIT = 10.0;
export const METERS_PER_UNIT = 0.1;

// Standard Regulation Jenga Block Dimensions: 7.5 cm (L) x 2.5 cm (W) x 1.5 cm (H)
// Exact 3:1 width-to-length ratio so 3 blocks form a 7.5 cm x 7.5 cm square layer.
export const BLOCK_LENGTH = 0.75; // 7.5 cm along local X
export const BLOCK_HEIGHT = 0.15; // 1.5 cm along local Y
export const BLOCK_WIDTH = 0.25;  // 2.5 cm along local Z

export const BLOCK_HALF_X = BLOCK_LENGTH * 0.5; // 0.375 dm (3.75 cm)
export const BLOCK_HALF_Y = BLOCK_HEIGHT * 0.5; // 0.075 dm (0.75 cm)
export const BLOCK_HALF_Z = BLOCK_WIDTH * 0.5;  // 0.125 dm (1.25 cm)

export const INITIAL_LAYERS = 18;
export const BLOCKS_PER_LAYER = 3;
export const INITIAL_BLOCK_COUNT = INITIAL_LAYERS * BLOCKS_PER_LAYER; // 54 blocks

// Soda-lime / Cathedral Stained Glass density: 2.52 g/cm^3 = 2.52 kg/dm^3
// Block volume = 0.75 * 0.15 * 0.25 = 0.028125 dm^3 (28.125 cm^3)
// Exact block mass = 0.028125 * 2.52 = 0.070875 kg (~70.9 grams)
export const GLASS_DENSITY = 2.52;

// Exact Froude-number similarity scaling (Fr = 1):
// World length is in decimeters (1 unit = 0.1 m) while Rapier solves with g_sim = -9.81
// at SIM_TIME_SCALE = sqrt(10) = 3.16227766 sim-sec per real wall-clock second:
// -> Effective real-time gravity = 9.81 * (sqrt(10))^2 = 98.1 dm/s^2 = 9.81 m/s^2 (exact 1.000g!)
// -> Keeps Rapier PGS contact compliance 10x stiffer (< 0.3 mm across 30+ stacked glass stories).
export const SIM_GRAVITY_Y = -9.81;
export const SIM_TIME_SCALE = Math.sqrt(10.0); // 3.16227766
export const GRAVITY_Y = -98.1; // Real wall-clock dm/s^2 for micro-splinter particles

// Tabletop dimensions: 86 cm diameter artisan oak & limestone pedestal table
export const TABLE_RADIUS = 4.3; // 43 cm radius
export const TABLE_THICKNESS = 0.32;

// Physics stepping (60 Hz per sim-second * sqrt(10) = ~190 Hz in real wall-clock time = ~3 substeps per 60fps frame!)
export const SIM_PHYSICS_HZ = 60;
export const PHYSICS_DT = 1.0 / SIM_PHYSICS_HZ;

// Fracture thresholds in real wall-clock decimeters/sec (v_real = v_sim * SIM_TIME_SCALE):
// Gentle 1.0 cm top drop = ~4.4 dm/s (0.44 m/s) -> completely safe clink
// Medium impact (~9.5 dm/s = 0.95 m/s, ~4.6 cm drop onto hard oak) -> micro-crazing cracks
// Hard impact (~13.5 dm/s = 1.35 m/s, ~9.3 cm free-fall onto hard oak) -> shatters into 3D Voronoi shards
export const CRAZE_IMPACT_SPEED = 9.5;
export const SHATTER_IMPACT_SPEED = 13.5;

export interface StainedGlassSwatch {
  id: number;
  name: string;
  /** Beer-Lambert absorption sigma_a (per dm of optical path in glass). Higher = deeper absorption of that channel. */
  absorption: [number, number, number];
  /** Complementary secondary pane absorption for multi-pane Gothic stained glass tracery motifs */
  accentAbsorption: [number, number, number];
  /** UI preview hex color */
  uiHex: string;
}

/**
 * Authentic Cathedral Mouth-Blown Stained Glass Palette.
 * Using Beer-Lambert law T = exp(-sigma_a * pathLength_dm):
 * Across 1.5 cm (0.15 dm) height vs 2.5 cm (0.25 dm) width vs 7.5 cm (0.75 dm) length,
 * thickness naturally shifts saturation from luminous jewel glow to deep cathedral mass-tone!
 */
export const STAINED_GLASS_PALETTE: StainedGlassSwatch[] = [
  {
    id: 0,
    name: 'Selenium Ruby',
    absorption: [0.32, 7.8, 9.2],
    accentAbsorption: [0.45, 3.4, 8.8], // Warm amber-ruby accent
    uiHex: '#e12b38',
  },
  {
    id: 1,
    name: 'Chartres Cobalt',
    absorption: [8.6, 4.4, 0.28],
    accentAbsorption: [6.2, 1.8, 0.35], // Sky sapphire accent
    uiHex: '#2563eb',
  },
  {
    id: 2,
    name: 'Silver-Stain Amber',
    absorption: [0.22, 1.65, 8.9],
    accentAbsorption: [0.35, 3.6, 9.4], // Deep honey topaz
    uiHex: '#f59e0b',
  },
  {
    id: 3,
    name: 'Cathedral Emerald',
    absorption: [7.6, 0.48, 5.4],
    accentAbsorption: [5.8, 0.72, 2.2], // Seafoam tourmaline
    uiHex: '#10b981',
  },
  {
    id: 4,
    name: 'Manganese Amethyst',
    absorption: [3.2, 8.4, 1.15],
    accentAbsorption: [1.4, 7.6, 2.8], // Rose plum accent
    uiHex: '#9333ea',
  },
  {
    id: 5,
    name: 'Peacock Teal',
    absorption: [8.2, 1.05, 1.1],
    accentAbsorption: [6.8, 2.5, 0.42], // Cerulean accent
    uiHex: '#06b6d4',
  },
  {
    id: 6,
    name: 'Gold-Ruby Rose',
    absorption: [0.42, 6.6, 4.1],
    accentAbsorption: [0.28, 2.4, 7.2], // Peach-gold accent
    uiHex: '#ec4899',
  },
  {
    id: 7,
    name: 'Antique Seeded Crystal',
    // Real antique soda-lime clear glass has a faint iron-oxide aqua-celadon tint over 7.5 cm path!
    absorption: [0.68, 0.34, 0.48],
    accentAbsorption: [0.42, 1.15, 3.8], // Subtle pale champagne accent
    uiHex: '#e2f1ee',
  },
];

export type TraceryMotif = 0 | 1 | 2 | 3;
// 0: Classic Leaded Quarry (beveled glass with perimeter copper-foil solder & subtle corner quarries)
// 1: Diamond Lozenge Vitrail (central jewel diamond framed by leaded diagonal cames)
// 2: Gothic Lancet Triptych (three leaded cathedral panels with arched transom)
// 3: Rose Medallion Inlay (central circular jewel rondel with radial lead cames)

export interface LightingPreset {
  id: string;
  name: string;
  sunAzimuthDeg: number;
  sunElevationDeg: number;
  sunColor: [number, number, number];
  skyTint: [number, number, number];
}

export const LIGHTING_PRESETS: LightingPreset[] = [
  {
    id: 'lancet-morning',
    name: 'Morning Lancet',
    sunAzimuthDeg: -42,
    sunElevationDeg: 38,
    sunColor: [3.4, 3.05, 2.55],
    skyTint: [0.44, 0.54, 0.72],
  },
  {
    id: 'golden-vespers',
    name: 'Golden Vespers',
    sunAzimuthDeg: 48,
    sunElevationDeg: 27,
    sunColor: [3.85, 2.65, 1.75],
    skyTint: [0.52, 0.44, 0.62],
  },
  {
    id: 'high-clerestory',
    name: 'Clerestory Noon',
    sunAzimuthDeg: -18,
    sunElevationDeg: 54,
    sunColor: [3.3, 3.25, 3.1],
    skyTint: [0.48, 0.58, 0.74],
  },
];

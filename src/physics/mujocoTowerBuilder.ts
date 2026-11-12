import {
  BLOCK_BEVEL_RADIUS,
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  BLOCK_HEIGHT,
  BLOCK_WIDTH,
  STAINED_GLASS_PALETTE,
  TABLE_RADIUS,
  TABLE_THICKNESS,
  TARGET_BLOCK_MASS_KG,
  TraceryMotif,
} from './constants';
import {
  M_PER_DM,
  MUJOCO_CALIBRATED_CONFIG,
  MujocoCalibratedConfig,
} from './mujocoCalibratedParams';
import type { GlassBlockState, TopPlacementSlot, TopSupportFrame } from './PhysicsWorld';
import { fractureBlockVoronoi } from './voronoiFracture';
import { quatFromAxisAngle, quatRotateVec3, Quat, Vec3 } from '../math/vecMat';

/**
 * MuJoCo Collision Bitmasks (`contype` & `conaffinity`):
 * - Bit 1 (`0x01`): Oak Tabletop
 * - Bit 2 (`0x02`): Intact Glass Blocks (on table)
 * - Bit 4 (`0x04`): Hero Voronoi Glass Shards
 * - Bit 8 (`0x08`): Small Glass Cullet Shards (collide with Table + Blocks, skip shard-shard)
 * - Bit 16 (`0x10`): Hidden Underground Parking Shelf (`y = -50 m`) for rapid `sleep="init"` pool parking
 */
export const MJ_CONTYPE_TABLE = 0x01;
// Must be 0x00 because MuJoCo tests `(contype1 & conaffinity2) || (contype2 & conaffinity1)`:
// keeping table.conaffinity === 0x00 ensures clearing table bits on off-rim blocks/shards immediately disables table support!
export const MJ_CONAFFINITY_TABLE = 0x00;
export const MJ_CONTYPE_SHARD_TABLE = 0x20;

export const MJ_CONTYPE_BLOCK = 0x02;
export const MJ_CONAFFINITY_BLOCK_ON_TABLE = 0x07; // Oak Table (1) + Blocks (2) + Hero Shards (4)
export const MJ_CONAFFINITY_BLOCK_OFF_RIM = 0x06;  // Excludes Oak Table (1) once past circular oak rim!

export const MJ_CONTYPE_HERO_SHARD = 0x04;
export const MJ_CONAFFINITY_HERO_SHARD_ON_TABLE = 0x26; // Shard Table (0x20) + Blocks (2) + Hero Shards (4)
export const MJ_CONAFFINITY_HERO_SHARD_OFF_RIM = 0x06;  // Excludes Shard Table (0x20) once past circular rim!

export const MJ_CONTYPE_CULLET_SHARD = 0x08;
export const MJ_CONAFFINITY_CULLET_ON_TABLE = 0x20;     // Shard Table (0x20): bounces on oak table without carpeting falling bricks
export const MJ_CONAFFINITY_CULLET_OFF_RIM = 0x00;      // Excludes Shard Table (0x20) once past circular rim!

export const MJ_CONTYPE_PARKING = 0x10;
export const MJ_CONAFFINITY_PARKING = 0x10;
export const MJ_PARKING_SHELF_TOP_Y_M = -50.0;

/**
 * MuJoCo `geom_group` assignments (`0..5`) for zero-cost C++ BVH filtering in `mj_ray`:
 * - Group 0: Active intact glass blocks (`g_blk_*`)
 * - Group 1: Active Voronoi glass shards (`g_shd_*`)
 * - Group 2: Static oak tabletop (`oak_table`)
 * - Group 3: Parked pool bodies (`contype === 0`), `parking_shelf`, and duplicate `oak_table_shard`
 */
export const MJ_GEOM_GROUP_BLOCK = 0;
export const MJ_GEOM_GROUP_SHARD = 1;
export const MJ_GEOM_GROUP_TABLE = 2;
export const MJ_GEOM_GROUP_PARKED = 3;

export interface InitialBlockSpec {
  id: number;
  layer: number;
  slot: number;
  isLoose: boolean;
  posDm: Vec3;
  quat: Quat;
}

/**
 * Pure deterministic generator of nominal regulation tower block poses (`7.5 x 2.5 x 1.5 cm` in Vitrail dm).
 * Shared by `buildMujocoTowerXml` and `MujocoWorkerBridge.rebuildTower` for synchronous nominal state resets.
 */
export function computeInitialBlockSpecs(towerLayers: number): InitialBlockSpec[] {
  const specs: InitialBlockSpec[] = [];
  let id = 0;
  for (let layer = 0; layer < towerLayers; layer++) {
    const isOdd = layer % 2 === 1;
    const yaw = isOdd ? Math.PI * 0.5 : 0.0;
    const centerIsLoose = layer % 2 === 0;

    for (let slot = 0; slot < 3; slot++) {
      const offset = (slot - 1) * BLOCK_WIDTH;
      const jx = Math.sin(id * 12.9898) * 0.0022;
      const jz = Math.cos(id * 78.233) * 0.0022;
      const jYaw = Math.sin(id * 39.17) * 0.0032;

      const pxDm = (isOdd ? offset : 0) + jx;
      const pyDm = (layer + 0.5) * BLOCK_HEIGHT;
      const pzDm = (isOdd ? 0 : offset) + jz;
      const quat = quatFromAxisAngle([0, 1, 0], yaw + jYaw);
      const isLoose = centerIsLoose ? slot === 1 : slot === (layer % 4 === 1 ? 0 : 2);

      specs.push({
        id,
        layer,
        slot,
        isLoose,
        posDm: [pxDm, pyDm, pzDm],
        quat,
      });
      id++;
    }
  }
  return specs;
}

function mulberry32Tower(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic generator of nominal `GlassBlockState[]` matching `MujocoPhysicsEngine.rebuildTower`
 * bit-for-bit (`mulberry32` palette, motif, rippleSeed, loose-slot pattern, and poses).
 */
export function buildNominalGlassBlocks(
  towerLayers: number,
  towerSeed: number,
  glassStaticFriction: number = MUJOCO_CALIBRATED_CONFIG.glassFriction[0],
  specs: InitialBlockSpec[] = computeInitialBlockSpecs(towerLayers),
  blockBodyIds?: Int32Array,
  blockGeomIds?: Int32Array
): GlassBlockState[] {
  const paletteLen = STAINED_GLASS_PALETTE.length;
  const normSeed = towerSeed >>> 0;
  const seedRng =
    normSeed !== 0
      ? mulberry32Tower((normSeed ^ Math.imul(towerLayers, 0x9e3779b9)) >>> 0)
      : null;
  let prevRowSwatch0 = -1;
  let prevRowSwatch1 = -1;

  const blocks: GlassBlockState[] = new Array(specs.length);
  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i];
    const bodyId = blockBodyIds ? blockBodyIds[spec.id] : 0;
    const geomId = blockGeomIds ? blockGeomIds[spec.id] : 0;
    let swatchId: number;
    let motif: TraceryMotif;
    let rippleSeed: number;
    if (seedRng) {
      if (spec.slot === 0) {
        prevRowSwatch0 = -1;
        prevRowSwatch1 = -1;
      }
      swatchId = Math.floor(seedRng() * paletteLen) % paletteLen;
      if (spec.slot === 2 && swatchId === prevRowSwatch0 && swatchId === prevRowSwatch1) {
        swatchId = (swatchId + 1) % paletteLen;
      }
      if (spec.slot === 0) prevRowSwatch0 = swatchId;
      if (spec.slot === 1) prevRowSwatch1 = swatchId;
      motif = (Math.floor(seedRng() * 4) & 3) as TraceryMotif;
      rippleSeed = +(0.25 + seedRng() * 96.0).toFixed(5);
    } else {
      swatchId = (spec.layer * 3 + spec.slot * 2 + Math.floor(spec.layer / 2)) % paletteLen;
      motif = 0;
      rippleSeed = (spec.id + 1) * 1.6180339;
    }

    blocks[i] = {
      id: spec.id,
      bodyHandle: bodyId,
      colliderHandle: geomId,
      swatchId,
      motif,
      rippleSeed,
      initialLayer: spec.layer,
      assignedLayer: spec.layer,
      assignedSlot: spec.slot,
      isLooseSlot: spec.isLoose,
      position: [...spec.posDm],
      rotation: [...spec.quat],
      prePos: [...spec.posDm],
      preQuat: [...spec.quat],
      linVel: [0, 0, 0],
      angVel: [0, 0, 0],
      currentFriction: glassStaticFriction,
      isMicroDamped: false,
      damage: 0,
      crackLocalOrigin: [0, 0, 0],
      shattered: false,
      extracted: false,
      isLegalTarget: spec.layer < towerLayers - 1,
    };
  }
  return blocks;
}

/**
 * Deterministic generator of nominal `topSupportFrame` and `openTopSlots` for a freshly built `towerLayers` stack.
 */
export function buildNominalTopSupportFrameAndSlots(towerLayers: number): {
  topSupportFrame: TopSupportFrame;
  openTopSlots: TopPlacementSlot[];
} {
  const targetLayer = Math.max(1, Math.min(32, Math.round(towerLayers)));
  const isOdd = targetLayer % 2 === 1;
  const yawRad = isOdd ? Math.PI * 0.5 : 0.0;
  const defaultRotation = quatFromAxisAngle([0, 1, 0], yawRad);
  const lengthAxis = quatRotateVec3(defaultRotation, [1, 0, 0]);
  const spacingAxis = quatRotateVec3(defaultRotation, [0, 0, 1]);
  const surfY = targetLayer * BLOCK_HEIGHT;
  const layerY = surfY + BLOCK_HALF_Y + 0.002;
  const topSupportFrame: TopSupportFrame = {
    targetLayer,
    surfaceCenter: [0, surfY, 0],
    layerCenter: [0, layerY, 0],
    upNormal: [0, 1, 0],
    tiltQuat: [0, 0, 0, 1],
    defaultYawRad: yawRad,
    defaultRotation,
    lengthAxis,
    spacingAxis,
  };
  const openTopSlots: TopPlacementSlot[] = [0, 1, 2].map((slotIndex) => {
    const uOffset = (slotIndex - 1) * BLOCK_WIDTH;
    return {
      layer: targetLayer,
      slotIndex,
      position: [spacingAxis[0] * uOffset, layerY, spacingAxis[2] * uOffset],
      rotation: [...defaultRotation],
      yawRad,
    };
  });
  return { topSupportFrame, openTopSlots };
}

export interface MujocoTowerIndexMap {
  /** Geom ID of the oak tabletop (`worldbody` static geom) */
  tableGeomId: number;
  /** Geom ID of the shard-restitution tabletop (`worldbody` static geom) */
  shardTableGeomId: number;
  /** Geom ID of the hidden parking shelf at `y = -50 m` */
  parkingShelfGeomId: number;
  /** Body ID of the `mocap="true"` hand anchor (`mocapId = 0`) */
  handMocapBodyId: number;
  /** Equality constraint index (`0`) for the soft `<connect>` hand grab */
  grabEqualityId: number;
  /** Body IDs for intact tower blocks (`0 .. blockCount - 1`) */
  blockBodyIds: Int32Array;
  /** Geom IDs for intact tower blocks (`0 .. blockCount - 1`) */
  blockGeomIds: Int32Array;
  /** `qpos` start offset (`7 * freeJointIdx`) for each intact block */
  blockQposAdr: Int32Array;
  /** `qvel` / `dof` start offset (`6 * freeJointIdx`) for each intact block */
  blockDofAdr: Int32Array;
  /** Body IDs for the `512` pre-allocated Voronoi shard slots */
  shardBodyIds: Int32Array;
  /** Geom IDs for the `512` pre-allocated Voronoi shard slots */
  shardGeomIds: Int32Array;
  /** `qpos` start offset for each shard slot */
  shardQposAdr: Int32Array;
  /** `qvel` / `dof` start offset for each shard slot */
  shardDofAdr: Int32Array;
  /** Body IDs for dynamic test/sandbox extra bodies (e.g. `T2` micro-splinter, `T3` striker, `T3d` extra blocks) */
  extraBodyIds: Int32Array;
  /** Geom IDs for dynamic extra bodies */
  extraGeomIds: Int32Array;
  /** `qpos` start offset for extra bodies */
  extraQposAdr: Int32Array;
  /** `qvel` / `dof` start offset for extra bodies */
  extraDofAdr: Int32Array;
  /** Mesh asset IDs (`geom_dataid`) for the 16 canonical Voronoi convex hull archetypes (`[id0..id15]`) */
  voronoiMeshIds: Int32Array;
  /** Half-extents (in SI meters) of each canonical Voronoi hull archetype for aspect-ratio matching */
  voronoiArchetypeAspect: Array<[number, number, number]>;
  /** Initial block specifications in Vitrail decimeters */
  initialBlocks: InitialBlockSpec[];
}

/**
 * Generates the exact 24 vertices of a diamond-beveled rectangular glass prism in SI meters.
 * Matches `createBeveledBlockMesh` (`1.25 mm` 45-degree chamfer along all 12 edges & 8 corners).
 */
function build24VertexBevelPrismVertexStr(
  hxM: number,
  hyM: number,
  hzM: number,
  bevelM: number
): string {
  const b = Math.min(bevelM, hxM * 0.45, hyM * 0.45, hzM * 0.45);
  const coords: string[] = [];
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        coords.push(
          `${(sx * hxM).toFixed(6)} ${(sy * (hyM - b)).toFixed(6)} ${(sz * (hzM - b)).toFixed(6)}`,
          `${(sx * (hxM - b)).toFixed(6)} ${(sy * hyM).toFixed(6)} ${(sz * (hzM - b)).toFixed(6)}`,
          `${(sx * (hxM - b)).toFixed(6)} ${(sy * (hyM - b)).toFixed(6)} ${(sz * hzM).toFixed(6)}`
        );
      }
    }
  }
  return coords.join('  ');
}

interface CachedVoronoiBank {
  xmlAssets: string;
  aspects: Array<[number, number, number]>;
}

let cachedVoronoiBank: CachedVoronoiBank | null = null;

/**
 * Pre-generates 16 diverse, authentic 3D Voronoi glass shard convex hulls at canonical SI meter scale
 * covering elongated splinters, flat tabular flakes, wedge corners, and chunky crush-core polyhedra.
 */
function getCanonicalVoronoiHullBank(archetypeCount: number): CachedVoronoiBank {
  if (cachedVoronoiBank && cachedVoronoiBank.aspects.length === archetypeCount) {
    return cachedVoronoiBank;
  }

  const rawShards = fractureBlockVoronoi([0.16, -0.02, 0.04], archetypeCount, 1337, BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z, 0, 0.65, true);
  const aspects: Array<[number, number, number]> = [];
  let xmlAssets = '';

  for (let i = 0; i < archetypeCount; i++) {
    const sh = rawShards[i % rawShards.length];
    const pts = sh.hullPoints;
    let cx = 0, cy = 0, cz = 0;
    const vCount = Math.max(1, pts.length / 3);
    for (let p = 0; p < pts.length; p += 3) {
      cx += pts[p];
      cy += pts[p + 1];
      cz += pts[p + 2];
    }
    cx /= vCount; cy /= vCount; cz /= vCount;
    let maxX = 0.01, maxY = 0.01, maxZ = 0.01;
    const vStr: string[] = [];
    for (let p = 0; p < pts.length; p += 3) {
      const xm = (pts[p] - cx) * M_PER_DM * 0.92;
      const ym = (pts[p + 1] - cy) * M_PER_DM * 0.92;
      const zm = (pts[p + 2] - cz) * M_PER_DM * 0.92;
      maxX = Math.max(maxX, Math.abs(xm));
      maxY = Math.max(maxY, Math.abs(ym));
      maxZ = Math.max(maxZ, Math.abs(zm));
      vStr.push(`${xm.toFixed(6)} ${ym.toFixed(6)} ${zm.toFixed(6)}`);
    }
    aspects.push([maxX, maxY, maxZ]);
    xmlAssets += `    <mesh name="voronoi_arch_${i}" vertex="${vStr.join('  ')}" inertia="convex"/>\n`;
  }

  cachedVoronoiBank = { xmlAssets, aspects };
  return cachedVoronoiBank;
}

/**
 * Builds a complete, deterministic MuJoCo 4.0 MJCF XML scene + O(1) index map for a `towerLayers`-high
 * regulation stained-glass tower, pre-allocated `512`-slot sleeping Voronoi shard pool, extra dynamic body
 * slots (for runtime block additions / test strikers), and implicit mocap hand equality constraint.
 */
export function buildMujocoTowerXml(
  towerLayers: number,
  cfg: MujocoCalibratedConfig = MUJOCO_CALIBRATED_CONFIG
): { xml: string; map: MujocoTowerIndexMap } {
  const colHalfXM = (BLOCK_HALF_X - 0.0008) * M_PER_DM;
  const colHalfYM = BLOCK_HALF_Y * M_PER_DM;
  const colHalfYLooseM = Math.max(0.002, colHalfYM - cfg.looseSlotClearanceM);
  const colHalfZM = (BLOCK_HALF_Z - 0.0014) * M_PER_DM;
  // Regulation 1.25 mm 45-deg edge bevel produces ~0.55 mm effective bearing flat inset after polishing
  // (preserving flush [-0.030, +0.220] dm cantilever restoring moment while keeping smooth bevel-to-bevel glide)
  const bevelM = BLOCK_BEVEL_RADIUS * M_PER_DM * 0.44;

  const lxM = 2.0 * colHalfXM;
  const lyM = 2.0 * colHalfYM;
  const lzM = 2.0 * colHalfZM;
  const blkIx = ((TARGET_BLOCK_MASS_KG / 12.0) * (lyM * lyM + lzM * lzM)).toExponential(6);
  const blkIy = ((TARGET_BLOCK_MASS_KG / 12.0) * (lxM * lxM + lzM * lzM)).toExponential(6);
  const blkIz = ((TARGET_BLOCK_MASS_KG / 12.0) * (lxM * lxM + lyM * lyM)).toExponential(6);

  const stdBrickVerts = build24VertexBevelPrismVertexStr(colHalfXM, colHalfYM, colHalfZM, bevelM);
  const looseBrickVerts = build24VertexBevelPrismVertexStr(
    colHalfXM,
    colHalfYLooseM,
    colHalfZM,
    bevelM
  );
  const voronoiBank = getCanonicalVoronoiHullBank(cfg.voronoiHullArchetypeCount);

  const fGlass = `${cfg.glassFriction[0]} ${cfg.glassFriction[1]} ${cfg.glassFriction[2]}`;
  const fTable = `${cfg.tableFriction[0]} ${cfg.tableFriction[1]} ${cfg.tableFriction[2]}`;
  const fShard = `${cfg.shardFriction[0]} ${cfg.shardFriction[1]} ${cfg.shardFriction[2]}`;
  const solref = `${cfg.contactSolref[0]} ${cfg.contactSolref[1]}`;
  const shardSolref = `${cfg.shardSolref[0]} ${cfg.shardSolref[1]}`;
  const solimp = `${cfg.contactSolimp[0]} ${cfg.contactSolimp[1]} ${cfg.contactSolimp[2]} ${cfg.contactSolimp[3]} ${cfg.contactSolimp[4]}`;
  const grabSolref = `${cfg.grabConnectSolref[0]} ${cfg.grabConnectSolref[1]}`;
  const grabSolimp = `${cfg.grabConnectSolimp[0]} ${cfg.grabConnectSolimp[1]} ${cfg.grabConnectSolimp[2]} ${cfg.grabConnectSolimp[3]} ${cfg.grabConnectSolimp[4]}`;

  // Extra block capacity allows dynamic synthetic block spawning (e.g. T3d 25-block batches)
  const blockCount = towerLayers * 3;
  const shardCap = cfg.shardPoolCapacity;
  const extraCap = cfg.extraBodyPoolCapacity;

  const blockBodyIds = new Int32Array(blockCount);
  const blockGeomIds = new Int32Array(blockCount);
  const blockQposAdr = new Int32Array(blockCount);
  const blockDofAdr = new Int32Array(blockCount);

  const shardBodyIds = new Int32Array(shardCap);
  const shardGeomIds = new Int32Array(shardCap);
  const shardQposAdr = new Int32Array(shardCap);
  const shardDofAdr = new Int32Array(shardCap);

  const extraBodyIds = new Int32Array(extraCap);
  const extraGeomIds = new Int32Array(extraCap);
  const extraQposAdr = new Int32Array(extraCap);
  const extraDofAdr = new Int32Array(extraCap);

  const voronoiMeshIds = new Int32Array(cfg.voronoiHullArchetypeCount);
  for (let i = 0; i < cfg.voronoiHullArchetypeCount; i++) {
    // Mesh 0 = brick_std, Mesh 1 = brick_loose, Mesh 2..17 = voronoi_arch_0..15
    voronoiMeshIds[i] = 2 + i;
  }

  // Body 0 = world, Body 1 = hand_anchor (mocap, 0 DOFs, 0 geoms)
  // Geom 0 = oak_table, Geom 1 = parking_shelf, Geom 2 = oak_table_shard
  let nextBodyId = 2;
  let nextGeomId = 3;
  let nextFreeJointIdx = 0;

  const initialBlocks = computeInitialBlockSpecs(towerLayers);
  const bodyXmlChunks: string[] = [];

  for (let i = 0; i < initialBlocks.length; i++) {
    const spec = initialBlocks[i];
    const id = spec.id;
    const meshName = spec.isLoose && spec.layer < towerLayers - 1 ? 'brick_loose' : 'brick_std';

    blockBodyIds[id] = nextBodyId++;
    blockGeomIds[id] = nextGeomId++;
    blockQposAdr[id] = nextFreeJointIdx * 7;
    blockDofAdr[id] = nextFreeJointIdx * 6;
    nextFreeJointIdx++;

    const pxM = spec.posDm[0] * M_PER_DM;
    const pyM = spec.posDm[1] * M_PER_DM;
    const pzM = spec.posDm[2] * M_PER_DM;
    // MuJoCo quaternion convention is [w, x, y, z]
    const qStr = `${spec.quat[3].toFixed(7)} ${spec.quat[0].toFixed(7)} ${spec.quat[1].toFixed(7)} ${spec.quat[2].toFixed(7)}`;

    bodyXmlChunks.push(
      `    <body name="blk_${id}" pos="${pxM.toFixed(6)} ${pyM.toFixed(6)} ${pzM.toFixed(6)}" quat="${qStr}">` +
        `<inertial pos="0 0 0" quat="1 0 0 0" mass="${TARGET_BLOCK_MASS_KG}" diaginertia="${blkIx} ${blkIy} ${blkIz}"/>` +
        `<freejoint/>` +
        `<geom name="g_blk_${id}" group="${MJ_GEOM_GROUP_BLOCK}" type="mesh" mesh="${meshName}" condim="4" friction="${fGlass}" solref="${solref}" solimp="${solimp}" contype="${MJ_CONTYPE_BLOCK}" conaffinity="${MJ_CONAFFINITY_BLOCK_ON_TABLE}"/>` +
        `</body>`
    );
  }

  // Pre-allocate 512 sleeping Voronoi shard bodies parked on the hidden underground shelf (0 µs step cost while asleep!)
  for (let s = 0; s < shardCap; s++) {
    shardBodyIds[s] = nextBodyId++;
    shardGeomIds[s] = nextGeomId++;
    shardQposAdr[s] = nextFreeJointIdx * 7;
    shardDofAdr[s] = nextFreeJointIdx * 6;
    nextFreeJointIdx++;

    const parkXM = ((s % 32) - 15.5) * 0.12;
    const parkZM = (Math.floor(s / 32) - 8.0) * 0.12;
    const parkYM = MJ_PARKING_SHELF_TOP_Y_M + 0.006;
    const archIdx = s % cfg.voronoiHullArchetypeCount;

    bodyXmlChunks.push(
      `    <body name="shd_${s}" pos="${parkXM.toFixed(4)} ${parkYM.toFixed(4)} ${parkZM.toFixed(4)}" sleep="init">` +
        `<inertial pos="0 0 0" quat="1 0 0 0" mass="0.0045" diaginertia="1e-6 1e-6 1e-6"/>` +
        `<freejoint/>` +
        `<geom name="g_shd_${s}" group="${MJ_GEOM_GROUP_PARKED}" type="mesh" mesh="voronoi_arch_${archIdx}" condim="3" friction="${fShard}" solref="${solref}" solimp="${solimp}" contype="0" conaffinity="${MJ_CONAFFINITY_PARKING}"/>` +
        `</body>`
    );
  }

  // Pre-allocate 40 sleeping extra dynamic body slots for runtime test strikers / synthetic blocks
  // Initialized with large 0.50m AABB (`size="0.50 0.50 0.50"`) in MJCF so `mj_setConst` allocates static BVH
  // bounding volume leaves large enough for any runtime `createCollider` box half-extents!
  for (let e = 0; e < extraCap; e++) {
    extraBodyIds[e] = nextBodyId++;
    extraGeomIds[e] = nextGeomId++;
    extraQposAdr[e] = nextFreeJointIdx * 7;
    extraDofAdr[e] = nextFreeJointIdx * 6;
    nextFreeJointIdx++;

    const parkXM = ((e % 10) - 4.5) * 1.25;
    const parkZM = 3.0 + Math.floor(e / 10) * 1.25;
    const parkYM = MJ_PARKING_SHELF_TOP_Y_M + 0.505;

    bodyXmlChunks.push(
      `    <body name="ext_${e}" pos="${parkXM.toFixed(4)} ${parkYM.toFixed(4)} ${parkZM.toFixed(4)}" sleep="init">` +
        `<inertial pos="0 0 0" quat="1 0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>` +
        `<freejoint/>` +
        `<geom name="g_ext_${e}" group="${MJ_GEOM_GROUP_PARKED}" type="box" size="0.50 0.50 0.50" condim="4" friction="${fGlass}" solref="${solref}" solimp="${solimp}" contype="0" conaffinity="${MJ_CONAFFINITY_PARKING}"/>` +
        `</body>`
    );
  }

  const tableHalfSpanM = (TABLE_RADIUS + 0.5) * M_PER_DM;
  const tableHalfThickM = TABLE_THICKNESS * 0.5 * M_PER_DM;
  const integrator = cfg.integrator ?? 'Euler';

  const xml = `<mujoco model="vitrail_mujoco_4">
  <size memory="24M"/>
  <compiler inertiafromgeom="auto" autolimits="true"/>
  <option timestep="${cfg.timestepSec}" gravity="0 -9.81 0" integrator="${integrator}" solver="Newton" jacobian="sparse" cone="elliptic" iterations="${cfg.solverIterations}" ls_iterations="${cfg.lsIterations}" impratio="${cfg.impratio}" sleep_tolerance="${cfg.sleepTolerance}">
    <flag multiccd="enable" sleep="enable" warmstart="enable" energy="enable"/>
  </option>
  <asset>
    <mesh name="brick_std" vertex="${stdBrickVerts}" inertia="exact"/>
    <mesh name="brick_loose" vertex="${looseBrickVerts}" inertia="exact"/>
${voronoiBank.xmlAssets}  </asset>
  <worldbody>
    <geom name="oak_table" group="${MJ_GEOM_GROUP_TABLE}" type="box" pos="0 ${(-tableHalfThickM).toFixed(5)} 0" size="${tableHalfSpanM.toFixed(4)} ${tableHalfThickM.toFixed(5)} ${tableHalfSpanM.toFixed(4)}" condim="6" friction="${fTable}" solref="${solref}" solimp="${solimp}" contype="${MJ_CONTYPE_TABLE}" conaffinity="${MJ_CONAFFINITY_TABLE}"/>
    <geom name="parking_shelf" group="${MJ_GEOM_GROUP_PARKED}" type="box" pos="0 ${(MJ_PARKING_SHELF_TOP_Y_M - 0.05).toFixed(4)} 0" size="12.0 0.05 12.0" condim="1" priority="10" solref="0.012 1.25" solimp="${solimp}" contype="${MJ_CONTYPE_PARKING}" conaffinity="0"/>
    <geom name="oak_table_shard" group="${MJ_GEOM_GROUP_PARKED}" type="box" pos="0 ${(-tableHalfThickM).toFixed(5)} 0" size="${tableHalfSpanM.toFixed(4)} ${tableHalfThickM.toFixed(5)} ${tableHalfSpanM.toFixed(4)}" condim="3" priority="1" friction="${fShard}" solref="${shardSolref}" solimp="${solimp}" contype="${MJ_CONTYPE_SHARD_TABLE}" conaffinity="0"/>
    <body name="hand_anchor" mocap="true" pos="0 2.0 0"/>
${bodyXmlChunks.join('\n')}
  </worldbody>
  <equality>
    <connect name="hand_grab" body1="hand_anchor" body2="blk_0" anchor="0 0 0" active="false" solref="${grabSolref}" solimp="${grabSolimp}"/>
  </equality>
</mujoco>`;

  return {
    xml,
    map: {
      tableGeomId: 0,
      parkingShelfGeomId: 1,
      shardTableGeomId: 2,
      handMocapBodyId: 1,
      grabEqualityId: 0,
      blockBodyIds,
      blockGeomIds,
      blockQposAdr,
      blockDofAdr,
      shardBodyIds,
      shardGeomIds,
      shardQposAdr,
      shardDofAdr,
      extraBodyIds,
      extraGeomIds,
      extraQposAdr,
      extraDofAdr,
      voronoiMeshIds,
      voronoiArchetypeAspect: voronoiBank.aspects,
      initialBlocks,
    },
  };
}

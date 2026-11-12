import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';
import { quatInverseRotateVec3, quatRotateVec3 } from '../math/vecMat';
import {
  DM_PER_M,
  FROUDE_VEL_SCALE,
  INV_FROUDE_VEL_SCALE,
  M_PER_DM,
} from './mujocoCalibratedParams';
import {
  MJ_CONAFFINITY_BLOCK_ON_TABLE,
  MJ_CONTYPE_BLOCK,
  MJ_CONTYPE_CULLET_SHARD,
  MJ_CONTYPE_HERO_SHARD,
  MJ_GEOM_GROUP_BLOCK,
  MJ_GEOM_GROUP_PARKED,
  MJ_GEOM_GROUP_SHARD,
  MujocoTowerIndexMap,
} from './mujocoTowerBuilder';

export interface Vec3Obj {
  x: number;
  y: number;
  z: number;
}

export interface QuatObj {
  x: number;
  y: number;
  z: number;
  w: number;
}

interface BodyMapping {
  bodyId: number;
  geomId: number;
  qposAdr: number;
  dofAdr: number;
  gravityScale: number;
  sensor: boolean;
  enabled: boolean;
  settledAsleep: boolean;
  quietTicks: number;
  savedContype: number;
  savedConaffinity: number;
}

/**
 * Lightweight, zero-overhead Rapier-compatible `World` / `RigidBody` / `Collider` facade backed directly
 * by MuJoCo 4.0 WASM `MjModel` & `MjData` typed arrays (`qpos`, `qvel`, `xfrc_applied`, `body_mass`, `body_inertia`).
 *
 * Keeps all existing Vitrail diagnostic & test-suite assertions (`world.getWorld().getRigidBody(handle).mass()`,
 * `isSleeping()`, `setTranslation()`, `createRigidBody()`, etc.) 100% compatible with zero boilerplate!
 */
export class MujocoWorldCompat {
  private mj: MainModule;
  private model: MjModel;
  private data: MjData;
  private map: MujocoTowerIndexMap;
  private bodyLookup = new Map<number, BodyMapping>();
  private customGravityBodies = new Set<BodyMapping>();
  private bodyFacadeCache = new Map<number, MujocoRigidBodyCompat>();
  private colliderFacadeCache = new Map<number, MujocoColliderCompat>();
  private nextExtraIdx = 0;

  constructor(mj: MainModule, model: MjModel, data: MjData, map: MujocoTowerIndexMap) {
    this.mj = mj;
    this.model = model;
    this.data = data;
    this.map = map;

    for (let i = 0; i < map.blockBodyIds.length; i++) {
      const bodyId = map.blockBodyIds[i];
      const geomId = map.blockGeomIds[i];
      this.bodyLookup.set(bodyId, {
        bodyId,
        geomId,
        qposAdr: map.blockQposAdr[i],
        dofAdr: map.blockDofAdr[i],
        gravityScale: 1.0,
        sensor: false,
        enabled: true,
        settledAsleep: false,
        quietTicks: 0,
        savedContype: model.geom_contype[geomId],
        savedConaffinity: model.geom_conaffinity[geomId],
      });
    }

    for (let s = 0; s < map.shardBodyIds.length; s++) {
      const bodyId = map.shardBodyIds[s];
      const geomId = map.shardGeomIds[s];
      this.bodyLookup.set(bodyId, {
        bodyId,
        geomId,
        qposAdr: map.shardQposAdr[s],
        dofAdr: map.shardDofAdr[s],
        gravityScale: 1.0,
        sensor: false,
        enabled: false,
        settledAsleep: false,
        quietTicks: 0,
        savedContype: model.geom_contype[geomId],
        savedConaffinity: model.geom_conaffinity[geomId],
      });
    }

    for (let e = 0; e < map.extraBodyIds.length; e++) {
      const bodyId = map.extraBodyIds[e];
      const geomId = map.extraGeomIds[e];
      this.bodyLookup.set(bodyId, {
        bodyId,
        geomId,
        qposAdr: map.extraQposAdr[e],
        dofAdr: map.extraDofAdr[e],
        gravityScale: 1.0,
        sensor: false,
        enabled: false,
        settledAsleep: false,
        quietTicks: 0,
        savedContype: MJ_CONTYPE_BLOCK,
        savedConaffinity: MJ_CONAFFINITY_BLOCK_ON_TABLE,
      });
    }
  }

  public getModel(): MjModel {
    return this.model;
  }

  public getData(): MjData {
    return this.data;
  }

  public getMapping(bodyId: number): BodyMapping | undefined {
    return this.bodyLookup.get(bodyId);
  }

  public markShardActive(bodyId: number, active: boolean, contype?: number, conaffinity?: number): void {
    const m = this.bodyLookup.get(bodyId);
    if (!m) return;
    m.enabled = active;
    m.settledAsleep = false;
    m.quietTicks = 0;
    if (!active) {
      m.gravityScale = 1.0;
      this.customGravityBodies.delete(m);
    }
    if (contype !== undefined) m.savedContype = contype;
    if (conaffinity !== undefined) m.savedConaffinity = conaffinity;
  }

  public setBodyGravityScale(info: BodyMapping, scale: number): void {
    info.gravityScale = scale;
    if (Math.abs(scale - 1.0) > 1e-5) {
      this.customGravityBodies.add(info);
    } else {
      this.customGravityBodies.delete(info);
    }
  }

  /**
   * Applies exact gravity-scale compensation (`xfrc_applied`) before each `mj_step` for any body with `gravityScale !== 1.0`.
   */
  public applyCustomGravityScalesSubstep(): void {
    if (this.customGravityBodies.size === 0) return;
    const xfrc = this.data.xfrc_applied;
    const bodyMass = this.model.body_mass;
    for (const info of this.customGravityBodies) {
      if (!info.enabled) continue;
      const mass = bodyMass[info.bodyId];
      // World gravity is (0, -9.81, 0), so net gravity = scale * (-m * 9.81) requires upward force (1 - scale) * m * 9.81
      xfrc[info.bodyId * 6 + 1] += (1.0 - info.gravityScale) * mass * 9.81;
    }
  }

  public wakeBody(bodyId: number): void {
    const info = this.bodyLookup.get(bodyId);
    if (!info) return;
    info.settledAsleep = false;
    info.quietTicks = 0;
    const treeId = this.model.body_treeid[bodyId];
    if (treeId >= 0 && this.data.tree_asleep[treeId] >= 0) {
      // Inject a sub-piconewton ephemeral generalized force to trigger MuJoCo 4.0's internal mj_wake() cleanly
      this.data.qfrc_applied[info.dofAdr + 1] += 1e-9;
    }
  }

  /**
   * Wakes all active intact tower blocks (`blk_0..N-1`) AND any elevated or near-column Voronoi shards
   * whenever a block's support pose or collision state is mutated externally (e.g. shattering or removing
   * support blocks, releasing an aborted pull, or sliding a sole support block).
   */
  public wakeActiveTowerBlocks(): void {
    const blockIds = this.map.blockBodyIds;
    const blockGeoms = this.map.blockGeomIds;
    const geomContype = this.model.geom_contype;
    const bodyTreeId = this.model.body_treeid;
    const treeAsleep = this.data.tree_asleep;
    const qfrc = this.data.qfrc_applied;
    const qpos = this.data.qpos;
    for (let i = 0; i < blockIds.length; i++) {
      if (geomContype[blockGeoms[i]] === 0) continue; // Never wake parked unused/shattered blocks (`contype === 0`)!
      const bid = blockIds[i];
      const info = this.bodyLookup.get(bid);
      if (info && info.enabled && !info.sensor) {
        info.settledAsleep = false;
        info.quietTicks = 0;
        const treeId = bodyTreeId[bid];
        if (treeId >= 0 && treeAsleep[treeId] >= 0) {
          qfrc[info.dofAdr + 1] += 1e-9;
        }
      }
    }
    const shardIds = this.map.shardBodyIds;
    const shardGeoms = this.map.shardGeomIds;
    for (let i = 0; i < shardIds.length; i++) {
      if (geomContype[shardGeoms[i]] === 0) continue; // Never wake parked pool shards (`contype === 0`)!
      const sid = shardIds[i];
      const info = this.bodyLookup.get(sid);
      if (info && info.enabled && !info.sensor) {
        const xM = qpos[info.qposAdr + 0];
        const yM = qpos[info.qposAdr + 1];
        const zM = qpos[info.qposAdr + 2];
        // Wake any active sleeping shard perched above the tabletop (> 1.8 cm) or inside the column footprint (< 14 cm)
        if (yM > -0.05 && (yM > 0.018 || xM * xM + zM * zM < 0.0196)) {
          info.settledAsleep = false;
          info.quietTicks = 0;
          const treeId = bodyTreeId[sid];
          if (treeId >= 0 && treeAsleep[treeId] >= 0) {
            qfrc[info.dofAdr + 1] += 1e-9;
          }
        }
      }
    }
  }

  /**
   * Wakes every active (non-parked) Voronoi shard in the 512-slot pool (used on full Sandbox collapse impulses).
   */
  public wakeAllActiveShards(): void {
    const shardIds = this.map.shardBodyIds;
    const shardGeoms = this.map.shardGeomIds;
    const geomContype = this.model.geom_contype;
    const bodyTreeId = this.model.body_treeid;
    const treeAsleep = this.data.tree_asleep;
    const qfrc = this.data.qfrc_applied;
    for (let i = 0; i < shardIds.length; i++) {
      if (geomContype[shardGeoms[i]] === 0) continue; // Never wake parked pool shards (`contype === 0`)!
      const sid = shardIds[i];
      const info = this.bodyLookup.get(sid);
      if (info && info.enabled && !info.sensor) {
        info.settledAsleep = false;
        info.quietTicks = 0;
        const treeId = bodyTreeId[sid];
        if (treeId >= 0 && treeAsleep[treeId] >= 0) {
          qfrc[info.dofAdr + 1] += 1e-9;
        }
      }
    }
  }

  public syncKinematics(): void {
    this.mj.mj_kinematics(this.model, this.data);
  }

  public getRigidBody(handle: number): MujocoRigidBodyCompat | undefined {
    const info = this.bodyLookup.get(handle);
    if (!info) return undefined;
    let facade = this.bodyFacadeCache.get(handle);
    if (!facade) {
      facade = new MujocoRigidBodyCompat(this, this.model, this.data, info);
      this.bodyFacadeCache.set(handle, facade);
    }
    return facade;
  }

  public getCollider(geomHandle: number): MujocoColliderCompat | undefined {
    let found: BodyMapping | undefined;
    for (const info of this.bodyLookup.values()) {
      if (info.geomId === geomHandle) {
        found = info;
        break;
      }
    }
    if (!found) return undefined;
    let facade = this.colliderFacadeCache.get(geomHandle);
    if (!facade) {
      facade = new MujocoColliderCompat(this, this.model, found);
      this.colliderFacadeCache.set(geomHandle, facade);
    }
    return facade;
  }

  /**
   * Allocates one of the 40 pre-allocated dynamic extra bodies in `MjModel` (`0.0 ms`, zero recompile!)
   * so tests like `T2` (1.2g splinter), `T3b` (+X tip striker), and `T3d` (synthetic blocks) work seamlessly!
   */
  public createRigidBody(desc: {
    translation?: Vec3Obj;
    linvel?: Vec3Obj;
    angvel?: Vec3Obj;
    rotation?: QuatObj;
    gravityScale?: number;
  }): MujocoRigidBodyCompat {
    const slot = this.nextExtraIdx % this.map.extraBodyIds.length;
    this.nextExtraIdx++;
    const bodyId = this.map.extraBodyIds[slot];
    const info = this.bodyLookup.get(bodyId)!;
    info.enabled = true;
    info.sensor = false;
    info.gravityScale = typeof desc.gravityScale === 'number' ? desc.gravityScale : 1.0;

    // Inspect Rapier RigidBodyDesc fields (which store `.translation`, `.linvel`, `.rotation`, `.gravityScale`)
    const tr = desc.translation ?? { x: 0, y: 1.0, z: 0 };
    const rot = desc.rotation ?? { x: 0, y: 0, z: 0, w: 1 };
    const lv = desc.linvel ?? { x: 0, y: 0, z: 0 };
    const av = desc.angvel ?? { x: 0, y: 0, z: 0 };

    const rb = this.getRigidBody(bodyId)!;
    rb.setTranslation(tr, true);
    rb.setRotation(rot, true);
    rb.setLinvel(lv, true);
    rb.setAngvel(av, true);
    return rb;
  }

  public createCollider(
    colDesc: {
      shape?: { halfExtents?: Vec3Obj };
      density?: number;
      mass?: number;
      friction?: number;
    },
    parentBody: MujocoRigidBodyCompat
  ): MujocoColliderCompat {
    const info = this.bodyLookup.get(parentBody.handle)!;
    const gid = info.geomId;

    // Extract cuboid half-extents (in Vitrail dm) from Rapier ColliderDesc
    const heDm = colDesc.shape?.halfExtents ?? { x: 0.04, y: 0.03, z: 0.04 };
    const hxM = heDm.x * M_PER_DM;
    const hyM = heDm.y * M_PER_DM;
    const hzM = heDm.z * M_PER_DM;

    // Switch geom to mjGEOM_BOX (6) and update exact mass + box principal inertia in O(1)
    this.model.geom_type[gid] = 6; // mjGEOM_BOX
    this.model.geom_dataid[gid] = -1;
    this.model.geom_size[gid * 3 + 0] = hxM;
    this.model.geom_size[gid * 3 + 1] = hyM;
    this.model.geom_size[gid * 3 + 2] = hzM;
    this.model.geom_rbound[gid] = Math.hypot(hxM, hyM, hzM);
    // Reset AABB `[center(3), halfSize(3)]` so MuJoCo's BVH broadphase uses exact box bounds!
    const aabbBase = gid * 6;
    this.model.geom_aabb[aabbBase + 0] = 0;
    this.model.geom_aabb[aabbBase + 1] = 0;
    this.model.geom_aabb[aabbBase + 2] = 0;
    this.model.geom_aabb[aabbBase + 3] = hxM;
    this.model.geom_aabb[aabbBase + 4] = hyM;
    this.model.geom_aabb[aabbBase + 5] = hzM;
    setBodyGeomCollisionMask(
      this.model,
      info.bodyId,
      gid,
      MJ_CONTYPE_BLOCK,
      MJ_CONAFFINITY_BLOCK_ON_TABLE
    );
    info.savedContype = MJ_CONTYPE_BLOCK;
    info.savedConaffinity = MJ_CONAFFINITY_BLOCK_ON_TABLE;

    const volDm3 = 8.0 * heDm.x * heDm.y * heDm.z;
    const massKg =
      typeof colDesc.mass === 'number' && colDesc.mass > 0
        ? colDesc.mass
        : (typeof colDesc.density === 'number' ? colDesc.density : 2.5) * volDm3;

    const lxM = 2 * hxM;
    const lyM = 2 * hyM;
    const lzM = 2 * hzM;
    const ix = (massKg / 12.0) * (lyM * lyM + lzM * lzM);
    const iy = (massKg / 12.0) * (lxM * lxM + lzM * lzM);
    const iz = (massKg / 12.0) * (lxM * lxM + lyM * lyM);

    updateFreeBodyMassAndInertiaO1(this.model, info.bodyId, info.dofAdr, massKg, ix, iy, iz);
    this.wakeBody(info.bodyId);
    this.mj.mj_kinematics(this.model, this.data);
    return this.getCollider(gid)!;
  }

  public getActiveExtraBodyIds(): number[] {
    const out: number[] = [];
    for (let e = 0; e < this.map.extraBodyIds.length; e++) {
      const bid = this.map.extraBodyIds[e];
      if (this.bodyLookup.get(bid)?.enabled) out.push(bid);
    }
    return out;
  }
}

export function setBodyGeomCollisionMask(
  model: MjModel,
  bodyId: number,
  geomId: number,
  contype: number,
  conaffinity: number
): void {
  model.geom_contype[geomId] = contype;
  model.geom_conaffinity[geomId] = conaffinity;
  // Keep `geom_group` synchronized so native WASM `mj_ray` skips parked shelf geoms (`contype === 0`) in O(1)
  if (contype === 0) {
    model.geom_group[geomId] = MJ_GEOM_GROUP_PARKED;
  } else if ((contype & (MJ_CONTYPE_HERO_SHARD | MJ_CONTYPE_CULLET_SHARD)) !== 0) {
    model.geom_group[geomId] = MJ_GEOM_GROUP_SHARD;
  } else {
    model.geom_group[geomId] = MJ_GEOM_GROUP_BLOCK;
  }
  // Critical MuJoCo broadphase invariant: mj_collideTree filters dynamic body pairs by body_contype & body_conaffinity!
  if (bodyId > 0) {
    model.body_contype[bodyId] = contype;
    model.body_conaffinity[bodyId] = conaffinity;
  }
}

export function updateFreeBodyMassAndInertiaO1(
  model: MjModel,
  bodyId: number,
  dofAdr: number,
  massKg: number,
  ixSi: number,
  iySi: number,
  izSi: number,
  rotDampingRateSecInv: number = 0
): void {
  const safeM = Math.max(1e-5, massKg);
  const safeIx = Math.max(1e-10, ixSi);
  const safeIy = Math.max(1e-10, iySi);
  const safeIz = Math.max(1e-10, izSi);

  const minRotM0 = safeM * 2.5e-5;
  const effIx = Math.max(minRotM0, safeIx);
  const effIy = Math.max(minRotM0, safeIy);
  const effIz = Math.max(minRotM0, safeIz);

  model.body_mass[bodyId] = safeM;
  model.body_subtreemass[bodyId] = safeM;
  model.body_inertia[bodyId * 3 + 0] = effIx;
  model.body_inertia[bodyId * 3 + 1] = effIy;
  model.body_inertia[bodyId * 3 + 2] = effIz;
  model.body_iquat[bodyId * 4 + 0] = 1;
  model.body_iquat[bodyId * 4 + 1] = 0;
  model.body_iquat[bodyId * 4 + 2] = 0;
  model.body_iquat[bodyId * 4 + 3] = 0;

  // Critical for `body_simple == 1` free joints in MuJoCo 4.0: `mj_crb` / `mj_factorM` populates `qM`
  // directly from `model.dof_M0[dofAdr + 0..5]`, while `mj_rne` gyroscopic `ω × (I ω)` reads `body_inertia`.
  // Keeping both synchronized with `r_gyr >= 5 mm` (`2.5e-5 * m`) preserves exact 1.000g free-fall and
  // prevents gyroscopic divergence on 0.2 g splinters!
  model.dof_M0[dofAdr + 0] = safeM;
  model.dof_M0[dofAdr + 1] = safeM;
  model.dof_M0[dofAdr + 2] = safeM;
  model.dof_M0[dofAdr + 3] = effIx;
  model.dof_M0[dofAdr + 4] = effIy;
  model.dof_M0[dofAdr + 5] = effIz;

  // Under MuJoCo 4.0 `integrator="implicitfast"` (`mjINT_IMPLICITFAST === 3`), `M + dt * D` is unconditionally
  // L-stable for any `D >= 0`. Keep linear `dof_damping[0..2] === 0` so 1.000g ballistic free-fall is untouched,
  // while applying passive proportional angular damping (`b_rot = rate * I_eff`) on Voronoi shards!
  model.dof_damping[dofAdr + 0] = 0;
  model.dof_damping[dofAdr + 1] = 0;
  model.dof_damping[dofAdr + 2] = 0;
  if (rotDampingRateSecInv > 0 && model.opt.integrator === 3) {
    model.dof_damping[dofAdr + 3] = rotDampingRateSecInv * effIx;
    model.dof_damping[dofAdr + 4] = rotDampingRateSecInv * effIy;
    model.dof_damping[dofAdr + 5] = rotDampingRateSecInv * effIz;
  } else {
    model.dof_damping[dofAdr + 3] = 0;
    model.dof_damping[dofAdr + 4] = 0;
    model.dof_damping[dofAdr + 5] = 0;
  }

  const invM = 1.0 / safeM;
  const invIx = 1.0 / effIx;
  const invIy = 1.0 / effIy;
  const invIz = 1.0 / effIz;
  const avgInvI = (invIx + invIy + invIz) / 3.0;
  model.body_invweight0[bodyId * 2 + 0] = invM;
  model.body_invweight0[bodyId * 2 + 1] = avgInvI;
  model.dof_invweight0[dofAdr + 0] = invM;
  model.dof_invweight0[dofAdr + 1] = invM;
  model.dof_invweight0[dofAdr + 2] = invM;
  model.dof_invweight0[dofAdr + 3] = invIx;
  model.dof_invweight0[dofAdr + 4] = invIy;
  model.dof_invweight0[dofAdr + 5] = invIz;
}

export class MujocoRigidBodyCompat {
  public readonly handle: number;
  private world: MujocoWorldCompat;
  private model: MjModel;
  private data: MjData;
  private info: BodyMapping;

  constructor(world: MujocoWorldCompat, model: MjModel, data: MjData, info: BodyMapping) {
    this.handle = info.bodyId;
    this.world = world;
    this.model = model;
    this.data = data;
    this.info = info;
  }

  public mass(): number {
    return this.model.body_mass[this.info.bodyId];
  }

  public isSleeping(): boolean {
    if (!this.info.enabled || this.info.settledAsleep) return true;
    const treeId = this.model.body_treeid[this.info.bodyId];
    return treeId >= 0 ? this.data.tree_asleep[treeId] >= 0 : false;
  }

  public wakeUp(): void {
    this.world.wakeBody(this.info.bodyId);
  }

  public translation(): Vec3Obj {
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    return {
      x: qpos[adr + 0] * DM_PER_M,
      y: qpos[adr + 1] * DM_PER_M,
      z: qpos[adr + 2] * DM_PER_M,
    };
  }

  public setTranslation(posDm: Vec3Obj, wake: boolean = true): void {
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    qpos[adr + 0] = posDm.x * M_PER_DM;
    qpos[adr + 1] = posDm.y * M_PER_DM;
    qpos[adr + 2] = posDm.z * M_PER_DM;
    // Also mirror into xpos immediately so synchronous queries before next step see exact translation
    const bid3 = this.info.bodyId * 3;
    const xpos = this.data.xpos;
    xpos[bid3 + 0] = posDm.x * M_PER_DM;
    xpos[bid3 + 1] = posDm.y * M_PER_DM;
    xpos[bid3 + 2] = posDm.z * M_PER_DM;
    if (wake) {
      this.world.wakeBody(this.info.bodyId);
      if (this.info.bodyId < 2 + this.world['map'].blockBodyIds.length) {
        this.world.wakeActiveTowerBlocks();
      }
    }
  }

  public rotation(): QuatObj {
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    return {
      x: qpos[adr + 4],
      y: qpos[adr + 5],
      z: qpos[adr + 6],
      w: qpos[adr + 3],
    };
  }

  public setRotation(q: QuatObj, wake: boolean = true): void {
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    const xquat = this.data.xquat;
    const len = Math.hypot(q.x, q.y, q.z, q.w) || 1;
    qpos[adr + 3] = q.w / len;
    qpos[adr + 4] = q.x / len;
    qpos[adr + 5] = q.y / len;
    qpos[adr + 6] = q.z / len;
    const bid4 = this.info.bodyId * 4;
    xquat[bid4 + 0] = q.w / len;
    xquat[bid4 + 1] = q.x / len;
    xquat[bid4 + 2] = q.y / len;
    xquat[bid4 + 3] = q.z / len;
    if (wake) {
      this.world.wakeBody(this.info.bodyId);
      if (this.info.bodyId < 2 + this.world['map'].blockBodyIds.length) {
        this.world.wakeActiveTowerBlocks();
      }
    }
  }

  public linvel(): Vec3Obj {
    const dof = this.info.dofAdr;
    const qvel = this.data.qvel;
    return {
      x: qvel[dof + 0] * FROUDE_VEL_SCALE,
      y: qvel[dof + 1] * FROUDE_VEL_SCALE,
      z: qvel[dof + 2] * FROUDE_VEL_SCALE,
    };
  }

  public setLinvel(vSimDm: Vec3Obj, wake: boolean = true): void {
    const dof = this.info.dofAdr;
    const qvel = this.data.qvel;
    qvel[dof + 0] = vSimDm.x * INV_FROUDE_VEL_SCALE;
    qvel[dof + 1] = vSimDm.y * INV_FROUDE_VEL_SCALE;
    qvel[dof + 2] = vSimDm.z * INV_FROUDE_VEL_SCALE;
    if (wake && (vSimDm.x !== 0 || vSimDm.y !== 0 || vSimDm.z !== 0)) {
      this.world.wakeBody(this.info.bodyId);
    }
  }

  public angvel(): Vec3Obj {
    const dof = this.info.dofAdr;
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    const qBody: [number, number, number, number] = [
      qpos[adr + 4],
      qpos[adr + 5],
      qpos[adr + 6],
      qpos[adr + 3],
    ];
    // MuJoCo qvel[3..5] is body-local angular velocity in SI rad/s -> rotate to world & scale to Froude sim rad/s
    const wLocalSi: [number, number, number] = [
      qvel[dof + 3],
      qvel[dof + 4],
      qvel[dof + 5],
    ];
    const wWorldSi = quatRotateVec3(qBody, wLocalSi);
    return {
      x: wWorldSi[0] * INV_FROUDE_VEL_SCALE,
      y: wWorldSi[1] * INV_FROUDE_VEL_SCALE,
      z: wWorldSi[2] * INV_FROUDE_VEL_SCALE,
    };
  }

  public setAngvel(wSimWorld: Vec3Obj, wake: boolean = true): void {
    const dof = this.info.dofAdr;
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    const qBody: [number, number, number, number] = [
      qpos[adr + 4],
      qpos[adr + 5],
      qpos[adr + 6],
      qpos[adr + 3],
    ];
    const wWorldSi: [number, number, number] = [
      wSimWorld.x * FROUDE_VEL_SCALE,
      wSimWorld.y * FROUDE_VEL_SCALE,
      wSimWorld.z * FROUDE_VEL_SCALE,
    ];
    const wLocalSi = quatInverseRotateVec3(qBody, wWorldSi);
    qvel[dof + 3] = wLocalSi[0];
    qvel[dof + 4] = wLocalSi[1];
    qvel[dof + 5] = wLocalSi[2];
    if (wake && (wSimWorld.x !== 0 || wSimWorld.y !== 0 || wSimWorld.z !== 0)) {
      this.world.wakeBody(this.info.bodyId);
    }
  }

  /**
   * Applies a Vitrail Froude-scaled impulse (`kg * dm / s_sim`) at `worldPointDm` (`dm`),
   * updating linear `qvel[0..2]` and principal-frame angular `qvel[3..5]` with `|ω| <= 55 rad/s` Euler guard.
   */
  public applyImpulseAtPoint(impulseSim: Vec3Obj, worldPointDm: Vec3Obj, wake: boolean = true): void {
    const dof = this.info.dofAdr;
    const adr = this.info.qposAdr;
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    const dofM0 = this.model.dof_M0;

    const jxSi = impulseSim.x * INV_FROUDE_VEL_SCALE;
    const jySi = impulseSim.y * INV_FROUDE_VEL_SCALE;
    const jzSi = impulseSim.z * INV_FROUDE_VEL_SCALE;
    if (jxSi === 0 && jySi === 0 && jzSi === 0) return;

    const invM = 1.0 / Math.max(1e-5, dofM0[dof + 0] || this.model.body_mass[this.info.bodyId]);
    qvel[dof + 0] += jxSi * invM;
    qvel[dof + 1] += jySi * invM;
    qvel[dof + 2] += jzSi * invM;

    const rxM = worldPointDm.x * M_PER_DM - qpos[adr + 0];
    const ryM = worldPointDm.y * M_PER_DM - qpos[adr + 1];
    const rzM = worldPointDm.z * M_PER_DM - qpos[adr + 2];
    const dLxWorld = ryM * jzSi - rzM * jySi;
    const dLyWorld = rzM * jxSi - rxM * jzSi;
    const dLzWorld = rxM * jySi - ryM * jxSi;

    const qBody: [number, number, number, number] = [
      qpos[adr + 4],
      qpos[adr + 5],
      qpos[adr + 6],
      qpos[adr + 3],
    ];
    const dLBody = quatInverseRotateVec3(qBody, [dLxWorld, dLyWorld, dLzWorld]);
    const bid4 = this.info.bodyId * 4;
    const qInert: [number, number, number, number] = [
      this.model.body_iquat[bid4 + 1],
      this.model.body_iquat[bid4 + 2],
      this.model.body_iquat[bid4 + 3],
      this.model.body_iquat[bid4 + 0],
    ];
    const dLPrin = quatInverseRotateVec3(qInert, dLBody);

    let wx = qvel[dof + 3] + dLPrin[0] / Math.max(1e-8, dofM0[dof + 3]);
    let wy = qvel[dof + 4] + dLPrin[1] / Math.max(1e-8, dofM0[dof + 4]);
    let wz = qvel[dof + 5] + dLPrin[2] / Math.max(1e-8, dofM0[dof + 5]);
    const wNorm = Math.hypot(wx, wy, wz);
    if (wNorm > 55.0) {
      const cap = 55.0 / wNorm;
      wx *= cap;
      wy *= cap;
      wz *= cap;
    }
    qvel[dof + 3] = wx;
    qvel[dof + 4] = wy;
    qvel[dof + 5] = wz;

    if (wake) {
      this.world.wakeBody(this.info.bodyId);
    }
  }

  /**
   * Returns principal inertia in Vitrail `kg * dm^2` (`1 m^2 = 100 dm^2`).
   */
  public principalInertia(): Vec3Obj {
    const bid3 = this.info.bodyId * 3;
    return {
      x: this.model.body_inertia[bid3 + 0] * 100.0,
      y: this.model.body_inertia[bid3 + 1] * 100.0,
      z: this.model.body_inertia[bid3 + 2] * 100.0,
    };
  }

  public principalInertiaLocalFrame(): QuatObj {
    const bid4 = this.info.bodyId * 4;
    return {
      x: this.model.body_iquat[bid4 + 1],
      y: this.model.body_iquat[bid4 + 2],
      z: this.model.body_iquat[bid4 + 3],
      w: this.model.body_iquat[bid4 + 0],
    };
  }

  public setGravityScale(scale: number, wake: boolean = true): void {
    this.world.setBodyGravityScale(this.info, scale);
    if (wake) {
      this.world.wakeBody(this.info.bodyId);
      if (this.info.bodyId < 2 + this.world['map'].blockBodyIds.length) {
        this.world.wakeActiveTowerBlocks();
      }
    }
  }

  public setEnabled(enabled: boolean): void {
    this.info.enabled = enabled;
    const gid = this.info.geomId;
    if (!enabled) {
      setBodyGeomCollisionMask(this.model, this.info.bodyId, gid, 0, 0);
      const adr = this.info.qposAdr;
      const dof = this.info.dofAdr;
      this.data.qpos[adr + 1] = -40.0;
      for (let k = 0; k < 6; k++) this.data.qvel[dof + k] = 0;
      this.world.wakeActiveTowerBlocks();
    } else if (!this.info.sensor) {
      setBodyGeomCollisionMask(
        this.model,
        this.info.bodyId,
        gid,
        this.info.savedContype,
        this.info.savedConaffinity
      );
      this.world.wakeActiveTowerBlocks();
    }
  }
}

export class MujocoColliderCompat {
  public readonly handle: number;
  private world: MujocoWorldCompat;
  private model: MjModel;
  private info: BodyMapping;

  constructor(world: MujocoWorldCompat, model: MjModel, info: BodyMapping) {
    this.handle = info.geomId;
    this.world = world;
    this.model = model;
    this.info = info;
  }

  public setSensor(isSensor: boolean): void {
    this.info.sensor = isSensor;
    const gid = this.info.geomId;
    if (isSensor) {
      setBodyGeomCollisionMask(this.model, this.info.bodyId, gid, 0, 0);
    } else if (this.info.enabled) {
      setBodyGeomCollisionMask(
        this.model,
        this.info.bodyId,
        gid,
        this.info.savedContype,
        this.info.savedConaffinity
      );
    }
  }

  public setEnabled(enabled: boolean): void {
    this.setSensor(!enabled);
    this.world.wakeActiveTowerBlocks();
    if (enabled) this.world.wakeBody(this.info.bodyId);
  }

  public setFriction(muSlide: number): void {
    this.model.geom_friction[this.info.geomId * 3 + 0] = muSlide;
  }
}

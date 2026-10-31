import RAPIER from '@dimforge/rapier3d-compat';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  BLOCK_HEIGHT,
  BLOCK_LENGTH,
  BLOCK_WIDTH,
  CRAZE_IMPACT_SPEED,
  GLASS_DENSITY,
  GRAVITY_Y,
  INITIAL_LAYERS,
  PHYSICS_DT,
  SHATTER_IMPACT_SPEED,
  SIM_GRAVITY_Y,
  SIM_TIME_SCALE,
  STAINED_GLASS_PALETTE,
  TABLE_RADIUS,
  TABLE_THICKNESS,
  TraceryMotif,
} from './constants';
import { fractureBlockVoronoi } from './voronoiFracture';
import { Quat, quatFromAxisAngle, quatRotateVec3, Vec3 } from '../math/vecMat';
import { GlassAudioEngine } from '../audio/GlassAudioEngine';

export type GameMode = 'solo' | 'party' | 'sandbox';

export type TurnPhase =
  | 'SELECT_BLOCK'   // Waiting for player to tap-test or pull a legal block
  | 'PULLING'        // Actively sliding a block inside the tower
  | 'EXTRACTED'      // Block pulled clear; carry/hover over top slots to stack
  | 'SETTLING'       // Block dropped onto top; waiting for tower wobble to settle
  | 'COLLAPSED';     // Tower collapsed! Physics & shattering continue live

export interface GlassBlockState {
  id: number;
  bodyHandle: number;
  colliderHandle: number;
  swatchId: number;
  motif: TraceryMotif;
  rippleSeed: number;
  initialLayer: number;
  assignedLayer: number;
  assignedSlot: number; // 0, 1, 2
  position: Vec3;
  rotation: Quat;
  linVel: Vec3;
  angVel: Vec3;
  /** Accumulated internal crazing damage in [0, 1) before full shatter */
  damage: number;
  /** Local 3D origin of internal crazing spiderweb cracks */
  crackLocalOrigin: Vec3;
  shattered: boolean;
  extracted: boolean;
  /** Whether this block is currently legal to pull under Jenga rules */
  isLegalTarget: boolean;
}

export interface GlassShardState {
  id: number;
  bodyHandle: number;
  colliderHandle: number;
  swatchId: number;
  motif: TraceryMotif;
  rippleSeed: number;
  position: Vec3;
  rotation: Quat;
  volume: number;
  preVy: number;
  impactLocalPos: Vec3;
  planes: Float32Array;
  planeCount: number;
  /** Offset & vertex count inside the shared GPU shard vertex buffer */
  firstVertex: number;
  vertexCount: number;
}

export interface MicroSplinter {
  pos: Vec3;
  vel: Vec3;
  rot: number;
  rotSpeed: number;
  scale: number;
  swatchId: number;
  life: number;
  maxLife: number;
}

export interface TopPlacementSlot {
  layer: number;
  slotIndex: number; // 0, 1, 2
  position: Vec3;
  rotation: Quat;
  yawRad: number;
}

export interface TowerTelemetry {
  turnPhase: TurnPhase;
  gameMode: GameMode;
  currentPlayer: number; // 0-indexed
  playerCount: number;
  blocksStacked: number;
  currentLayers: number;
  peakHeightCm: number;
  leanAngleDeg: number;
  wobbleSpeedDegPerSec: number;
  centerOfMassXZ: [number, number]; // in cm relative to base center
  shatteredCount: number;
  intactCount: number;
  activeShardCount: number;
  grabResistance01: number; // 0 = loose, 0.5 = snug, 0.9+ = pinned load-bearing
  hoveredBlockId: number | null;
  activeBlockId: number | null;
  hoveredSlotIndex: number | null;
  settleProgress01: number;
  collapseReason: string | null;
}

export interface ShardMeshAppendEvent {
  firstVertex: number;
  vertexData: Float32Array;
}

// Collision membership/filter bitmasks (16-bit upper = membership, 16-bit lower = filter)
// Group 0x0001 = Table, Group 0x0002 = Intact Glass Blocks, Group 0x0004 = Glass Shards
const GROUP_TABLE = (0x0001 << 16) | 0xffff;
const GROUP_BLOCK_ON_TABLE = (0x0002 << 16) | 0xffff;
const GROUP_FREEFALL_OFF_RIM = (0x0002 << 16) | 0xfffe; // Excludes GROUP_TABLE once past circular table radius!
const GROUP_SHARD_ON_TABLE = (0x0004 << 16) | 0x0007;   // Collides with Table + Intact Blocks + Other Shards!
const GROUP_SHARD_OFF_RIM = (0x0004 << 16) | 0x0006;    // Excludes Table once past circular rim
const MAX_ACTIVE_SHARDS = 220;
const MAX_SHARD_VERTICES = 65536;
const MAX_SPLINTERS = 320;

let rapierInitialized = false;

export async function ensureRapierInitialized(): Promise<void> {
  if (!rapierInitialized) {
    await RAPIER.init();
    rapierInitialized = true;
  }
}

export class PhysicsWorld {
  private world!: RAPIER.World;
  private tableColliderHandles = new Set<number>();

  public blocks: GlassBlockState[] = [];
  public shards: GlassShardState[] = [];
  public splinters: MicroSplinter[] = [];
  public openTopSlots: TopPlacementSlot[] = [];

  // Shared GPU shard vertex buffer cursor
  private shardVertexCursor: number = 0;
  private pendingShardUploads: ShardMeshAppendEvent[] = [];
  private nextShardId: number = 1;

  // Collider handle -> block or shard lookup
  private colliderToBlock = new Map<number, GlassBlockState>();
  private colliderToShard = new Map<number, GlassShardState>();

  // Game rules & turn state
  public gameMode: GameMode = 'solo';
  public playerCount: number = 2;
  public currentPlayer: number = 0;
  public turnPhase: TurnPhase = 'SELECT_BLOCK';
  public blocksStacked: number = 0;
  public shatteredCount: number = 0;
  public collapseReason: string | null = null;

  // Grab / Hand state
  public hoveredBlockId: number | null = null;
  public activeBlockId: number | null = null;
  public grabLocalPoint: Vec3 = [0, 0, 0];
  public grabWorldAnchor: Vec3 = [0, 0, 0];
  public grabCurrentWorldOnBlock: Vec3 = [0, 0, 0];
  public grabPlaneY: number = 0;
  public grabResistance01: number = 0;
  public isPointerDraggingBlock: boolean = false;
  public carriedFromDrag: boolean = false;

  // Top placement state
  public hoveredSlotIndex: number | null = null;
  private carriedTargetPos: Vec3 = [0, 3.2, 0];
  private carriedTargetQuat: Quat = [0, 0, 0, 1];
  private settleTimer: number = 0;
  private readonly SETTLE_DURATION = 1.35;

  // Telemetry
  public leanAngleDeg: number = 0;
  public wobbleSpeedDegPerSec: number = 0;
  public centerOfMassXZCm: [number, number] = [0, 0];
  public currentLayers: number = INITIAL_LAYERS;
  public peakHeightCm: number = INITIAL_LAYERS * BLOCK_HEIGHT * 10;

  private accumulator: number = 0;
  private simTime: number = 0;
  private wakeKeepAliveTimer: number = 0;
  private audio: GlassAudioEngine;

  constructor(audio: GlassAudioEngine) {
    this.audio = audio;
    this.rebuildTower();
  }

  public rebuildTower(mode?: GameMode, players?: number): void {
    if (mode) this.gameMode = mode;
    if (players !== undefined) this.playerCount = Math.max(2, Math.min(4, players));

    if (this.world) {
      this.world.free();
    }

    this.world = new RAPIER.World({ x: 0.0, y: SIM_GRAVITY_Y, z: 0.0 });

    // Calibrate Rapier PGS solver with Froude-scaled meter stiffness (SIM_TIME_SCALE = sqrt(10))
    const params = this.world.integrationParameters;
    params.dt = PHYSICS_DT;
    params.numSolverIterations = 10;
    params.numAdditionalFrictionIterations = 4;
    params.maxCcdSubsteps = 1;

    // Single analytical SAT cuboid tabletop (y = 0.0); objects rolling beyond TABLE_RADIUS
    // switch collision mask to fall cleanly off the circular oak rim!
    this.tableColliderHandles.clear();
    const tableBody = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(0, -TABLE_THICKNESS * 0.5, 0)
    );
    const tablePlate = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(TABLE_RADIUS + 0.5, TABLE_THICKNESS * 0.5, TABLE_RADIUS + 0.5)
        .setFriction(0.54)
        .setRestitution(0.12)
        .setCollisionGroups(GROUP_TABLE),
      tableBody
    );
    this.tableColliderHandles.add(tablePlate.handle);

    // Reset state
    this.blocks = [];
    this.shards = [];
    this.splinters = [];
    this.colliderToBlock.clear();
    this.colliderToShard.clear();
    this.shardVertexCursor = 0;
    this.pendingShardUploads = [];
    this.currentPlayer = 0;
    this.turnPhase = 'SELECT_BLOCK';
    this.blocksStacked = 0;
    this.shatteredCount = 0;
    this.collapseReason = null;
    this.hoveredBlockId = null;
    this.activeBlockId = null;
    this.hoveredSlotIndex = null;
    this.isPointerDraggingBlock = false;
    this.grabResistance01 = 0;
    this.settleTimer = 0;
    this.accumulator = 0;
    this.simTime = 0;
    this.wakeKeepAliveTimer = 1.0;

    // Build 18-layer 54-block Stained Glass Jenga Tower
    let id = 0;
    for (let layer = 0; layer < INITIAL_LAYERS; layer++) {
      const isOdd = layer % 2 === 1;
      const yaw = isOdd ? Math.PI * 0.5 : 0.0;

      // Alternate loose block between middle (slot 1) and symmetric outer pair so tower CoM stays balanced!
      const centerIsLoose = layer % 2 === 0;

      for (let slot = 0; slot < 3; slot++) {
        const offset = (slot - 1) * BLOCK_WIDTH;
        // Tiny realistic human-stacking imperfection (+/- 0.12 mm pos, +/- 0.08 deg yaw)
        const jx = Math.sin(id * 12.9898) * 0.0006;
        const jz = Math.cos(id * 78.233) * 0.0006;
        const jYaw = Math.sin(id * 39.17) * 0.0012;

        const px = (isOdd ? offset : 0) + jx;
        const py = (layer + 0.5) * BLOCK_HEIGHT;
        const pz = (isOdd ? 0 : offset) + jz;
        const quat = quatFromAxisAngle([0, 1, 0], yaw + jYaw);

        // Symmetric micro-tolerance: 1 block per row has ~0.09 mm clearance (real Jenga block variance)
        // while the other 2 blocks carry the column load in static equilibrium.
        const isLoose = centerIsLoose ? slot === 1 : slot === (layer % 4 === 1 ? 0 : 2);
        const heightMicroDelta = isLoose ? -0.00046 : 0.0;
        const colHalfX = BLOCK_HALF_X - 0.0008;
        const colHalfY = BLOCK_HALF_Y + heightMicroDelta;
        const colHalfZ = BLOCK_HALF_Z - 0.0014;

        const rbDesc = RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(px, py, pz)
          .setRotation({ x: quat[0], y: quat[1], z: quat[2], w: quat[3] })
          .setLinearDamping(0.18)
          .setAngularDamping(0.26);

        const body = this.world.createRigidBody(rbDesc);

        const friction = isLoose
          ? 0.22 + ((id * 13) % 5) * 0.008
          : 0.38 + ((id * 17) % 7) * 0.012;
        const colDesc = RAPIER.ColliderDesc.cuboid(colHalfX, colHalfY, colHalfZ)
          .setDensity(GLASS_DENSITY)
          .setFriction(friction)
          .setRestitution(0.08)
          .setCollisionGroups(GROUP_BLOCK_ON_TABLE);

        const collider = this.world.createCollider(colDesc, body);

        // Harmonious stained-glass palette distribution (no identical adjacent blocks in same row)
        const swatchId = (layer * 3 + slot * 2 + Math.floor(layer / 2)) % STAINED_GLASS_PALETTE.length;
        const motif = ((layer + slot * 3) % 4) as TraceryMotif;

        const blockState: GlassBlockState = {
          id,
          bodyHandle: body.handle,
          colliderHandle: collider.handle,
          swatchId,
          motif,
          rippleSeed: (id + 1) * 1.6180339,
          initialLayer: layer,
          assignedLayer: layer,
          assignedSlot: slot,
          position: [px, py, pz],
          rotation: quat,
          linVel: [0, 0, 0],
          angVel: [0, 0, 0],
          damage: 0,
          crackLocalOrigin: [0, 0, 0],
          shattered: false,
          extracted: false,
          isLegalTarget: layer < INITIAL_LAYERS - 1,
        };

        this.blocks.push(blockState);
        this.colliderToBlock.set(collider.handle, blockState);
        id++;
      }
    }

    // Run 30 quiet warm-up substeps (0.125s) with impact events suppressed so the stack settles into true static equilibrium
    for (let i = 0; i < 30; i++) {
      this.world.step();
    }
    this.syncTransformsAndRules();
  }

  public consumePendingShardUploads(): ShardMeshAppendEvent[] {
    if (this.pendingShardUploads.length === 0) return [];
    const out = this.pendingShardUploads;
    this.pendingShardUploads = [];
    return out;
  }

  public wasShardBufferReset(): boolean {
    return this.shardVertexCursor === 0 && this.shards.length === 0;
  }

  private wakeAllTowerBlocks(): void {
    this.wakeKeepAliveTimer = 2.2;
    for (const b of this.blocks) {
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (rb && rb.isSleeping()) {
        rb.wakeUp();
      }
    }
  }

  /**
   * Calibrated finger tap / poke to test block looseness (just like tapping blocks in real Jenga!).
   */
  public tapNudgeBlock(blockId: number, localNormal: Vec3, localPoint: Vec3): boolean {
    const block = this.blocks[blockId];
    if (!block || block.shattered || block.extracted) return false;
    if (this.gameMode !== 'sandbox' && this.turnPhase !== 'SELECT_BLOCK') return false;
    if (this.gameMode !== 'sandbox' && !block.isLegalTarget) return false;

    const rb = this.world.getRigidBody(block.bodyHandle);
    if (!rb) return false;

    this.wakeAllTowerBlocks();

    // Determine lengthwise push/pull direction in world space
    const axisX = quatRotateVec3(block.rotation, [1, 0, 0]);
    let dirSign = 1;
    if (Math.abs(localNormal[0]) > 0.65) {
      // Clicked one of the square end caps -> poke inward away from finger!
      dirSign = -Math.sign(localNormal[0]);
    } else {
      // Clicked long side/top -> nudge toward whichever end was clicked
      dirSign = localPoint[0] >= 0 ? 1 : -1;
    }

    const impulseMag = this.gameMode === 'sandbox' ? 0.16 : 0.082;
    rb.applyImpulse(
      {
        x: axisX[0] * dirSign * impulseMag,
        y: 0.0,
        z: axisX[2] * dirSign * impulseMag,
      },
      true
    );
    this.audio.playGlassImpact(0.34, 1.04);
    return true;
  }

  /**
   * Starts pulling/grabbing a block at a specific local contact point.
   */
  public startGrabBlock(blockId: number, localPoint: Vec3, worldPoint: Vec3): boolean {
    const block = this.blocks[blockId];
    if (!block || block.shattered) return false;

    if (this.gameMode !== 'sandbox') {
      if (this.turnPhase === 'COLLAPSED' || this.turnPhase === 'SETTLING') return false;
      if (this.turnPhase === 'EXTRACTED' && block.id !== this.activeBlockId) return false;
      if (this.turnPhase === 'SELECT_BLOCK' && !block.isLegalTarget) return false;
    }

    this.wakeAllTowerBlocks();
    this.activeBlockId = blockId;
    this.grabLocalPoint = [...localPoint];
    this.grabPlaneY = worldPoint[1];
    this.grabWorldAnchor = [...worldPoint];
    this.grabCurrentWorldOnBlock = [...worldPoint];
    this.isPointerDraggingBlock = true;

    if (!block.extracted && this.turnPhase === 'SELECT_BLOCK') {
      this.turnPhase = 'PULLING';
    }
    return true;
  }

  public updateGrabTarget(worldTargetOnPlane: Vec3): void {
    this.grabWorldAnchor[0] = worldTargetOnPlane[0];
    this.grabWorldAnchor[1] = this.grabPlaneY;
    this.grabWorldAnchor[2] = worldTargetOnPlane[2];
  }

  /**
   * Updates which open top slot is hovered while carrying an extracted block.
   */
  public updateCarriedHover(rayOrigin: Vec3, rayDir: Vec3): void {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return;
    if (this.openTopSlots.length === 0) return;

    // Intersect ray with horizontal plane at top layer height
    const topY = this.openTopSlots[0].position[1];
    let hitX = 0;
    let hitZ = 0;
    if (Math.abs(rayDir[1]) > 1e-4) {
      const t = (topY - rayOrigin[1]) / rayDir[1];
      if (t > 0) {
        hitX = rayOrigin[0] + rayDir[0] * t;
        hitZ = rayOrigin[2] + rayDir[2] * t;
      }
    }

    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < this.openTopSlots.length; i++) {
      const s = this.openTopSlots[i];
      const d = Math.hypot(hitX - s.position[0], hitZ - s.position[2]);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }

    const chosen = this.openTopSlots[bestIdx];
    this.hoveredSlotIndex = chosen.slotIndex;

    // Allow subtle +/- 1.5 mm natural hand offset around the slot center
    const fineDx = Math.max(-0.015, Math.min(0.015, (hitX - chosen.position[0]) * 0.12));
    const fineDz = Math.max(-0.015, Math.min(0.015, (hitZ - chosen.position[2]) * 0.12));

    // Poise 0.85 cm above the target slot so release drops cleanly onto the tower under real gravity
    this.carriedTargetPos = [
      chosen.position[0] + fineDx,
      chosen.position[1] + 0.085,
      chosen.position[2] + fineDz,
    ];
    this.carriedTargetQuat = [...chosen.rotation];
  }

  /**
   * Releases pointer drag.
   * - If block is still inside tower (PULLING), lets go of block (player may nudge another block!).
   * - If block was dragged directly onto a top slot after extraction, drops it onto the slot!
   */
  public endGrab(dropIfPoisedOverTower: boolean = false): void {
    this.isPointerDraggingBlock = false;
    this.grabResistance01 = 0;
    this.audio.updateSlideFriction(0, 0);

    if (this.turnPhase === 'PULLING') {
      // Block was not extracted yet; return to SELECT_BLOCK so user can try another block or keep pulling
      this.activeBlockId = null;
      this.turnPhase = 'SELECT_BLOCK';
      return;
    }

    if (this.turnPhase === 'EXTRACTED' && dropIfPoisedOverTower) {
      this.releaseExtractedBlockOntoTop();
    }
  }

  /**
   * Releases the carried extracted block from ~8.5 mm above the chosen top slot under 100% real Rapier gravity!
   */
  public releaseExtractedBlockOntoTop(): boolean {
    if (this.turnPhase !== 'EXTRACTED' || this.activeBlockId === null) return false;
    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) return false;

    const chosenSlot =
      this.openTopSlots.find((s) => s.slotIndex === this.hoveredSlotIndex) ??
      this.openTopSlots[0];
    if (!chosenSlot) return false;

    const rb = this.world.getRigidBody(block.bodyHandle);
    const col = this.world.getCollider(block.colliderHandle);
    if (!rb || !col) return false;

    // Re-enable solid collision & gravity and place cleanly poised 0.65 cm above the chosen top slot
    col.setSensor(false);
    rb.setGravityScale(1.0, true);
    rb.setTranslation(
      {
        x: this.carriedTargetPos[0],
        y: chosenSlot.position[1] + 0.065,
        z: this.carriedTargetPos[2],
      },
      true
    );
    rb.setRotation(
      {
        x: chosenSlot.rotation[0],
        y: chosenSlot.rotation[1],
        z: chosenSlot.rotation[2],
        w: chosenSlot.rotation[3],
      },
      true
    );
    rb.setLinvel({ x: 0, y: -0.6, z: 0 }, true);
    rb.setAngvel({ x: 0, y: 0, z: 0 }, true);

    block.extracted = false;
    block.assignedLayer = chosenSlot.layer;
    block.assignedSlot = chosenSlot.slotIndex;

    this.wakeAllTowerBlocks();
    this.activeBlockId = null;
    this.hoveredSlotIndex = null;
    this.carriedFromDrag = false;
    this.turnPhase = 'SETTLING';
    this.settleTimer = 0;
    return true;
  }

  /**
   * Sandbox fun tool: launches a crystal striker sphere or brisk impulse to test dramatic collapses & shattering!
   */
  public triggerSandboxCollapseImpulse(): void {
    this.wakeAllTowerBlocks();
    // Strike a lower-middle layer briskly to trigger a realistic leaning cascade collapse
    for (const b of this.blocks) {
      if (!b.shattered && b.assignedLayer >= 2 && b.assignedLayer <= 4) {
        const rb = this.world.getRigidBody(b.bodyHandle);
        if (rb) {
          rb.applyImpulse({ x: 0.48, y: 0.02, z: 0.16 }, true);
        }
      }
    }
    this.audio.playGlassImpact(0.85, 0.95);
  }

  public step(frameDt: number): void {
    const clampedDt = Math.min(0.05, Math.max(0.0, frameDt));
    this.accumulator += clampedDt * SIM_TIME_SCALE;
    this.simTime += clampedDt;

    if (this.wakeKeepAliveTimer > 0) {
      this.wakeKeepAliveTimer -= clampedDt;
      for (const b of this.blocks) {
        if (!b.shattered) {
          const rb = this.world.getRigidBody(b.bodyHandle);
          if (rb && rb.isSleeping()) rb.wakeUp();
        }
      }
    }

    let substeps = 0;
    while (this.accumulator >= PHYSICS_DT && substeps < 14) {
      this.applyHandForcesSubstep();
      this.snapshotPreStepVelocities();
      this.world.step();
      this.processCollisionsAndFractures();
      this.accumulator -= PHYSICS_DT;
      substeps++;
    }

    this.updateMicroSplinters(clampedDt);
    this.syncTransformsAndRules();
    this.updateTurnStateMachine(clampedDt);
  }

  private applyHandForcesSubstep(): void {
    if (this.activeBlockId === null) {
      this.grabResistance01 = 0;
      return;
    }

    const block = this.blocks[this.activeBlockId];
    if (!block || block.shattered) {
      this.activeBlockId = null;
      this.grabResistance01 = 0;
      return;
    }

    const rb = this.world.getRigidBody(block.bodyHandle);
    const col = this.world.getCollider(block.colliderHandle);
    if (!rb || !col) return;

    const pos = rb.translation();
    const rot = rb.rotation();
    const q: Quat = [rot.x, rot.y, rot.z, rot.w];
    const lv = rb.linvel();
    const av = rb.angvel();

    if (this.turnPhase === 'PULLING' || (this.gameMode === 'sandbox' && !block.extracted)) {
      const worldOffset = quatRotateVec3(q, this.grabLocalPoint);
      const grabWorld: Vec3 = [
        pos.x + worldOffset[0],
        pos.y + worldOffset[1],
        pos.z + worldOffset[2],
      ];
      this.grabCurrentWorldOnBlock = grabWorld;

      const errX = this.grabWorldAnchor[0] - grabWorld[0];
      const errZ = this.grabWorldAnchor[2] - grabWorld[2];

      // Project hand pull primarily along the block's natural lengthwise extraction axis while inside the tower
      // so slight lateral mouse drift doesn't jam adjacent columns, while preserving real sideways force if dragged hard!
      const axisX = quatRotateVec3(q, [1, 0, 0]);
      const axisZ = quatRotateVec3(q, [0, 0, 1]);
      const radialDist = Math.hypot(pos.x, pos.z);
      const insideFactor = Math.max(0.0, Math.min(1.0, 1.0 - (radialDist - 0.35) / 0.45));

      const alongLen = errX * axisX[0] + errZ * axisX[2];
      const acrossWid = errX * axisZ[0] + errZ * axisZ[2];
      const lateralScale = this.gameMode === 'sandbox' ? 0.85 : 1.0 - 0.72 * insideFactor;

      const effErrX = axisX[0] * alongLen + axisZ[0] * acrossWid * lateralScale;
      const effErrZ = axisX[2] * alongLen + axisZ[2] * acrossWid * lateralScale;

      // Point velocity at grab point: v + w x r
      const vx = lv.x + (av.y * worldOffset[2] - av.z * worldOffset[1]);
      const vz = lv.z + (av.x * worldOffset[1] - av.y * worldOffset[0]);

      const kp = this.gameMode === 'sandbox' ? 32.0 : 18.5;
      const kd = 2.3;

      let fx = kp * effErrX - kd * vx;
      let fz = kp * effErrZ - kd * vz;
      // Gentle vertical steadiness damping so pulling horizontally doesn't pop block upward
      const fy = -1.6 * lv.y;

      const maxForce = this.gameMode === 'sandbox' ? 11.5 : 5.4;
      const fMag = Math.hypot(fx, fz);
      if (fMag > maxForce) {
        fx = (fx / fMag) * maxForce;
        fz = (fz / fMag) * maxForce;
      }

      rb.applyImpulseAtPoint(
        { x: fx * PHYSICS_DT, y: fy * PHYSICS_DT, z: fz * PHYSICS_DT },
        { x: grabWorld[0], y: grabWorld[1], z: grabWorld[2] },
        true
      );

      // Gentle rotational steadying in-hand while gripping block
      rb.applyTorqueImpulse(
        {
          x: -av.x * 0.0008,
          y: -av.y * 0.0011,
          z: -av.z * 0.0008,
        },
        true
      );

      // Compute tactile resistance metric (high force + low speed = pinned load-bearing block!)
      const speedHorizReal = Math.hypot(lv.x, lv.z) * SIM_TIME_SCALE;
      const rawRes = Math.min(1.0, fMag / (maxForce * 0.72)) * Math.max(0.08, 1.0 - speedHorizReal / 3.8);
      this.grabResistance01 = this.grabResistance01 * 0.86 + rawRes * 0.14;
      this.audio.updateSlideFriction(speedHorizReal, this.grabResistance01);

      // Check if block has cleanly cleared the tower's horizontal footprint!
      if (this.gameMode !== 'sandbox') {
        const refCenter = this.getLayerSupportCenter(block.assignedLayer);
        const alongClear = Math.abs(
          (pos.x - refCenter[0]) * axisX[0] + (pos.z - refCenter[1]) * axisX[2]
        );
        const clearDist = Math.hypot(pos.x - refCenter[0], pos.z - refCenter[1]);
        if (alongClear > BLOCK_LENGTH * 0.99 || clearDist > BLOCK_LENGTH * 1.02) {
          // Block is extracted! Transition smoothly into carried top-placement mode!
          block.extracted = true;
          this.turnPhase = 'EXTRACTED';
          this.carriedFromDrag = this.isPointerDraggingBlock;
          this.grabResistance01 = 0;
          this.audio.updateSlideFriction(0, 0);
          this.audio.playGlassImpact(0.26, 1.25);

          // Temporarily make sensor while rising alongside tower so lifting to top never clips side rows
          col.setSensor(true);
          rb.setGravityScale(0.0, true);
          this.computeOpenTopSlots();
          if (this.openTopSlots.length > 0) {
            const defSlot = this.openTopSlots[Math.min(1, this.openTopSlots.length - 1)];
            this.hoveredSlotIndex = defSlot.slotIndex;
            this.carriedTargetPos = [
              defSlot.position[0],
              defSlot.position[1] + 0.085,
              defSlot.position[2],
            ];
            this.carriedTargetQuat = [...defSlot.rotation];
          }
        }
      }
    } else if (this.turnPhase === 'EXTRACTED') {
      // Smooth critically-damped kinematic-like PD glide to poise above hovered top slot
      const dx = this.carriedTargetPos[0] - pos.x;
      const dy = this.carriedTargetPos[1] - pos.y;
      const dz = this.carriedTargetPos[2] - pos.z;

      rb.setLinvel(
        {
          x: dx * 7.5,
          y: dy * 7.5,
          z: dz * 7.5,
        },
        true
      );

      // Smoothly slerp rotation toward slot alignment
      const tq = this.carriedTargetQuat;
      let dot = q[0] * tq[0] + q[1] * tq[1] + q[2] * tq[2] + q[3] * tq[3];
      const sign = dot < 0 ? -1 : 1;
      dot = Math.abs(dot);
      const blend = 0.22;
      let nx = q[0] * (1 - blend) + tq[0] * sign * blend;
      let ny = q[1] * (1 - blend) + tq[1] * sign * blend;
      let nz = q[2] * (1 - blend) + tq[2] * sign * blend;
      let nw = q[3] * (1 - blend) + tq[3] * sign * blend;
      const nLen = Math.hypot(nx, ny, nz, nw) || 1;
      nx /= nLen; ny /= nLen; nz /= nLen; nw /= nLen;
      rb.setRotation({ x: nx, y: ny, z: nz, w: nw }, true);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, true);
      this.grabCurrentWorldOnBlock = [pos.x, pos.y, pos.z];
      this.grabWorldAnchor = [...this.carriedTargetPos];
    }
  }

  private snapshotPreStepVelocities(): void {
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb) continue;
      const lv = rb.linvel();
      const av = rb.angvel();
      b.linVel[0] = lv.x;
      b.linVel[1] = lv.y;
      b.linVel[2] = lv.z;
      b.angVel[0] = av.x;
      b.angVel[1] = av.y;
      b.angVel[2] = av.z;
    }
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      const rb = this.world.getRigidBody(s.bodyHandle);
      if (rb && !rb.isSleeping()) {
        s.preVy = rb.linvel().y;
      } else {
        s.preVy = 0;
      }
    }
  }

  private spawnGlassCulletChips(
    worldOrigin: Vec3,
    baseLinVelSim: { x: number; y: number; z: number },
    impactNormal: Vec3,
    swatchId: number,
    count: number,
    speedScale: number
  ): void {
    for (let i = 0; i < count; i++) {
      if (this.splinters.length >= MAX_SPLINTERS) {
        this.splinters.shift();
      }
      const theta = Math.random() * Math.PI * 2;
      const latSpd = (0.8 + Math.random() * 4.6) * speedScale;
      // Tangent splay + modest bounce along impactNormal (all in real wall-clock dm/s)
      const vx =
        baseLinVelSim.x * SIM_TIME_SCALE * 0.35 +
        Math.cos(theta) * latSpd +
        impactNormal[0] * latSpd * 0.25;
      const vy =
        Math.max(-1.5, baseLinVelSim.y * SIM_TIME_SCALE * 0.15) +
        (0.45 + Math.random() * 2.1) * speedScale +
        Math.max(0, impactNormal[1]) * 0.8;
      const vz =
        baseLinVelSim.z * SIM_TIME_SCALE * 0.35 +
        Math.sin(theta) * latSpd +
        impactNormal[2] * latSpd * 0.25;

      this.splinters.push({
        pos: [
          worldOrigin[0] + (Math.random() - 0.5) * 0.10,
          Math.max(0.012, worldOrigin[1] + (Math.random() - 0.5) * 0.05),
          worldOrigin[2] + (Math.random() - 0.5) * 0.10,
        ],
        vel: [vx, vy, vz],
        rot: Math.random() * Math.PI * 2,
        rotSpeed: (Math.random() - 0.5) * 28,
        scale: 0.008 + Math.random() * 0.013,
        swatchId,
        life: 0,
        maxLife: 6.5 + Math.random() * 4.5,
      });
    }
  }

  private processCollisionsAndFractures(): void {
    if (this.simTime < 0.25) return;

    const blocksToShatter: Array<{
      blk: GlassBlockState;
      localHit: Vec3;
      impactSpeed: number;
      worldNormal: Vec3;
      hitTable: boolean;
    }> = [];

    for (let i = 0; i < this.blocks.length; i++) {
      const blk = this.blocks[i];
      if (blk.shattered || blk.extracted) continue;
      const rb = this.world.getRigidBody(blk.bodyHandle);
      if (!rb || rb.isSleeping()) continue;

      const post = rb.linvel();
      const postAv = rb.angvel();
      const dvx = post.x - blk.linVel[0];
      const dvy = post.y - blk.linVel[1];
      const dvz = post.z - blk.linVel[2];
      const dawx = postAv.x - blk.angVel[0];
      const dawy = postAv.y - blk.angVel[1];
      const dawz = postAv.z - blk.angVel[2];

      // Subtract free-fall gravity delta over 1 substep so pure airborne free-fall doesn't count as contact shock
      const nonGravDvy = dvy - SIM_GRAVITY_Y * PHYSICS_DT;
      const comShock = Math.hypot(dvx, nonGravDvy, dvz) * SIM_TIME_SCALE;
      const rotShockApprox = Math.hypot(dawx, dawy, dawz) * 0.22 * SIM_TIME_SCALE;

      if (comShock + rotShockApprox < 1.55) continue;

      // Only query narrowphase contacts for this specific impacting block!
      const col = this.world.getCollider(blk.colliderHandle);
      if (!col) continue;

      let hasRealContact = false;
      let hitTable = false;
      let localHit: Vec3 = [0, 0, 0];
      let worldNormal: Vec3 = [0, 1, 0];

      this.world.contactPairsWith(col, (otherCol) => {
        if (this.tableColliderHandles.has(otherCol.handle)) {
          hitTable = true;
        }
        this.world.contactPair(col, otherCol, (manifold, flipped) => {
          if (manifold.numContacts() > 0) {
            hasRealContact = true;
            const n = manifold.normal();
            worldNormal = flipped ? [n.x, n.y, n.z] : [-n.x, -n.y, -n.z];
            const p = flipped ? manifold.localContactPoint2(0) : manifold.localContactPoint1(0);
            if (p) {
              localHit = [p.x, p.y, p.z];
            }
          }
        });
      });

      if (!hasRealContact) continue;

      // Evaluate local contact-point shock including angular slap (dw x r_hit) for corner/end impacts
      const rot = rb.rotation();
      const q: Quat = [rot.x, rot.y, rot.z, rot.w];
      const rWorld = quatRotateVec3(q, localHit);
      const pvx = dvx + (dawy * rWorld[2] - dawz * rWorld[1]) * 0.65;
      const pvy = nonGravDvy + (dawz * rWorld[0] - dawx * rWorld[2]) * 0.65;
      const pvz = dvz + (dawx * rWorld[1] - dawy * rWorld[0]) * 0.65;
      const pointShock = Math.hypot(pvx, pvy, pvz) * SIM_TIME_SCALE;
      const shock = comShock * 0.48 + pointShock * 0.52;

      if (shock > 1.75) {
        const clinkIntensity = Math.min(1.0, (shock - 1.5) / 14.0);
        this.audio.playGlassImpact(clinkIntensity, hitTable ? 0.92 : 1.08);
      }

      const effectiveShock = shock * (hitTable ? 1.12 : 0.94);
      // Griffith brittle crack propagation: pre-crazed glass has reduced fracture toughness on subsequent impacts
      const shatterThreshold = SHATTER_IMPACT_SPEED * (1.0 - Math.min(0.85, blk.damage) * 0.42);

      if (effectiveShock >= shatterThreshold) {
        blocksToShatter.push({
          blk,
          localHit,
          impactSpeed: effectiveShock,
          worldNormal,
          hitTable,
        });
      } else if (effectiveShock >= CRAZE_IMPACT_SPEED) {
        const prevDamage = blk.damage;
        const addedDamage =
          0.38 +
          ((effectiveShock - CRAZE_IMPACT_SPEED) /
            (SHATTER_IMPACT_SPEED - CRAZE_IMPACT_SPEED)) *
            0.52;
        blk.damage += addedDamage;
        if (prevDamage < 0.05 || addedDamage > 0.52) {
          blk.crackLocalOrigin = [...localHit];
        }
        if (blk.damage >= 1.0) {
          blocksToShatter.push({
            blk,
            localHit,
            impactSpeed: effectiveShock,
            worldNormal,
            hitTable,
          });
        } else {
          // Spall a few tiny glass micro-chips right off the crazed impact point
          const pos = rb.translation();
          const hitWorld: Vec3 = [pos.x + rWorld[0], pos.y + rWorld[1], pos.z + rWorld[2]];
          this.spawnGlassCulletChips(hitWorld, post, worldNormal, blk.swatchId, 5, 0.55);
          this.audio.playGlassCrack(Math.min(1.0, blk.damage));
        }
      }
    }

    // Secondary crystalline shard-on-table clink tinks as airborne Voronoi shards bounce on the oak table
    for (let i = 0; i < this.shards.length; i++) {
      const s = this.shards[i];
      if (s.preVy > -0.85) continue;
      const rb = this.world.getRigidBody(s.bodyHandle);
      if (!rb || rb.isSleeping()) continue;
      const t = rb.translation();
      if (t.y > 0.24 || t.y < -0.1) continue;
      const postVy = rb.linvel().y;
      const dVyReal = (postVy - s.preVy - SIM_GRAVITY_Y * PHYSICS_DT) * SIM_TIME_SCALE;
      if (dVyReal > 3.2) {
        const intensity = Math.min(1.0, (dVyReal - 2.8) / 14.0);
        this.audio.playShardTableTink(intensity, s.volume);
        s.preVy = postVy;
      }
    }

    for (const item of blocksToShatter) {
      this.shatterBlock(item.blk, item.localHit, item.impactSpeed, item.worldNormal, item.hitTable);
    }
  }

  /**
   * Executes exact 3D Voronoi brittle fracture on `block`, replacing it with dynamic convex glass shards
   * and grounded cullet chips while conserving momentum.
   */
  public shatterBlock(
    block: GlassBlockState,
    localImpactPoint: Vec3,
    impactSpeed: number = 16.0,
    impactNormal: Vec3 = [0, 1, 0],
    hitTable: boolean = true
  ): void {
    if (block.shattered) return;
    const rb = this.world.getRigidBody(block.bodyHandle);
    if (!rb) return;

    const pos = rb.translation();
    const rot = rb.rotation();
    const q: Quat = [rot.x, rot.y, rot.z, rot.w];
    const lv = rb.linvel();
    const av = rb.angvel();

    block.shattered = true;
    this.shatteredCount++;
    if (this.activeBlockId === block.id) {
      this.activeBlockId = null;
      this.isPointerDraggingBlock = false;
    }

    this.colliderToBlock.delete(block.colliderHandle);
    this.world.removeRigidBody(rb);

    // Scale shard count & crush-zone density by BOTH impact kinetic energy and active shard budget:
    // Moderate snap (~14 dm/s) -> 8-10 larger recognizable stained-glass pieces + wedges;
    // High-velocity oak smash (22+ dm/s) -> 14-17 sharp lancet splinters, wedges, and crushed cullet!
    const impactEnergy01 = Math.max(0.0, Math.min(1.0, (impactSpeed - 12.5) / 15.0));
    const energyTargetShards = Math.round(8 + impactEnergy01 * 8);
    const headroom = MAX_ACTIVE_SHARDS - this.shards.length;
    const budgetCap = headroom > 110 ? 17 : headroom > 55 ? 12 : 8;
    const desiredShards = Math.min(energyTargetShards, budgetCap);

    // Retire sleeping shards first (then oldest) if needed to stay within physics & vertex buffer caps
    while (this.shards.length + desiredShards > MAX_ACTIVE_SHARDS && this.shards.length > 0) {
      let removeIdx = 0;
      for (let k = 0; k < this.shards.length; k++) {
        const candRb = this.world.getRigidBody(this.shards[k].bodyHandle);
        if (!candRb || candRb.isSleeping()) {
          removeIdx = k;
          break;
        }
      }
      const retired = this.shards.splice(removeIdx, 1)[0];
      this.colliderToShard.delete(retired.colliderHandle);
      const oldRb = this.world.getRigidBody(retired.bodyHandle);
      if (oldRb) this.world.removeRigidBody(oldRb);
    }

    const shardGeoms = fractureBlockVoronoi(
      localImpactPoint,
      desiredShards,
      block.id * 977 + Math.floor(this.simTime * 1000),
      BLOCK_HALF_X,
      BLOCK_HALF_Y,
      BLOCK_HALF_Z,
      block.motif,
      impactEnergy01
    );

    const worldImpact = quatRotateVec3(q, localImpactPoint);
    const impactWorldPos: Vec3 = [
      pos.x + worldImpact[0],
      pos.y + worldImpact[1],
      pos.z + worldImpact[2],
    ];

    for (const geom of shardGeoms) {
      if (this.shardVertexCursor + geom.vertexCount > MAX_SHARD_VERTICES) {
        break;
      }

      const worldOffset = quatRotateVec3(q, geom.localCentroid);
      const sx = pos.x + worldOffset[0];
      const sy = pos.y + worldOffset[1];
      const sz = pos.z + worldOffset[2];

      const shardRbDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(sx, sy, sz)
        .setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] })
        .setLinearDamping(0.24)
        .setAngularDamping(0.34)
        .setCcdEnabled(false);

      const shardBody = this.world.createRigidBody(shardRbDesc);

      // Build exact 3D polyhedral convex-hull collider (with 0.982 seam margin so sibling shards never
      // depenetrate-pop at t=0, while settling flat onto their true geometric cleavage facets on the oak table!)
      const collisionHull = new Float32Array(geom.hullPoints.length);
      let maxHx = 0.012, maxHy = 0.012, maxHz = 0.012;
      for (let p = 0; p < geom.hullPoints.length; p += 3) {
        const hxVal = geom.hullPoints[p] * 0.982;
        const hyVal = geom.hullPoints[p + 1] * 0.982;
        const hzVal = geom.hullPoints[p + 2] * 0.982;
        collisionHull[p] = hxVal;
        collisionHull[p + 1] = hyVal;
        collisionHull[p + 2] = hzVal;
        maxHx = Math.max(maxHx, Math.abs(hxVal));
        maxHy = Math.max(maxHy, Math.abs(hyVal));
        maxHz = Math.max(maxHz, Math.abs(hzVal));
      }

      const rawHullDesc = RAPIER.ColliderDesc.convexHull(collisionHull);
      const shardColDesc = (
        rawHullDesc ?? RAPIER.ColliderDesc.cuboid(maxHx * 0.85, maxHy * 0.85, maxHz * 0.85)
      )
        .setDensity(GLASS_DENSITY)
        .setFriction(0.48)
        .setRestitution(0.10)
        .setCollisionGroups(GROUP_SHARD_ON_TABLE);

      const shardCol = this.world.createCollider(shardColDesc, shardBody);

      // Physical momentum inheritance (v_0 + w_0 x r) + proximity-weighted lateral wedge splay
      // (eliminates artificial upward grenade explosions; crush-core splinters skitter laterally while
      // heavy far-end stained-glass chunks crack off and topple naturally under gravity!)
      const rx = sx - impactWorldPos[0];
      const ry = sy - impactWorldPos[1];
      const rz = sz - impactWorldPos[2];
      const rDotN = rx * impactNormal[0] + ry * impactNormal[1] + rz * impactNormal[2];
      // Project radial vector onto contact tangent plane for realistic lateral splatter
      let tx = rx - impactNormal[0] * rDotN;
      let ty = ry - impactNormal[1] * rDotN;
      let tz = rz - impactNormal[2] * rDotN;
      const tLen = Math.hypot(tx, ty, tz);
      if (tLen > 1e-4) {
        tx /= tLen;
        ty /= tLen;
        tz /= tLen;
      } else {
        const randAng = Math.random() * Math.PI * 2;
        tx = Math.cos(randAng);
        ty = 0;
        tz = Math.sin(randAng);
      }

      const proximity = Math.exp(-geom.impactDist / 0.22);
      const splaySpeedSim =
        ((0.45 + impactEnergy01 * 2.35) * (0.28 + 0.72 * proximity) * (0.75 + Math.random() * 0.5)) /
        SIM_TIME_SCALE;
      const normalReboundSim = (0.12 + impactEnergy01 * 0.28) * proximity / SIM_TIME_SCALE;

      const rigidVx = lv.x + (av.y * worldOffset[2] - av.z * worldOffset[1]);
      const rigidVy = lv.y + (av.z * worldOffset[0] - av.x * worldOffset[2]);
      const rigidVz = lv.z + (av.x * worldOffset[1] - av.y * worldOffset[0]);

      const vx = rigidVx + tx * splaySpeedSim + impactNormal[0] * normalReboundSim;
      const vy = rigidVy + ty * splaySpeedSim + impactNormal[1] * normalReboundSim;
      const vz = rigidVz + tz * splaySpeedSim + impactNormal[2] * normalReboundSim;

      const spinScale = 1.4 + proximity * (3.2 + impactEnergy01 * 3.8);
      shardBody.setLinvel({ x: vx, y: vy, z: vz }, true);
      shardBody.setAngvel(
        {
          x: av.x + (Math.random() - 0.5) * spinScale,
          y: av.y + (Math.random() - 0.5) * spinScale,
          z: av.z + (Math.random() - 0.5) * spinScale,
        },
        true
      );

      const firstVertex = this.shardVertexCursor;
      this.shardVertexCursor += geom.vertexCount;
      this.pendingShardUploads.push({
        firstVertex,
        vertexData: geom.renderVertices,
      });

      const shardState: GlassShardState = {
        id: this.nextShardId++,
        bodyHandle: shardBody.handle,
        colliderHandle: shardCol.handle,
        swatchId: block.swatchId,
        motif: block.motif,
        rippleSeed: block.rippleSeed,
        position: [sx, sy, sz],
        rotation: [...q],
        volume: geom.volume,
        preVy: vy,
        impactLocalPos: [...localImpactPoint],
        planes: geom.planes,
        planeCount: geom.planeCount,
        firstVertex,
        vertexCount: geom.vertexCount,
      };
      this.shards.push(shardState);
      this.colliderToShard.set(shardCol.handle, shardState);
    }

    // Spawn 14-20 tiny grounded cullet glass chips right at the impact crush zone
    const chipCount = Math.round(12 + impactEnergy01 * 10);
    this.spawnGlassCulletChips(
      impactWorldPos,
      lv,
      impactNormal,
      block.swatchId,
      chipCount,
      0.75 + impactEnergy01 * 0.55
    );

    this.audio.playGlassShatter(Math.min(1.0, impactSpeed / 22.0), hitTable);
    this.wakeAllTowerBlocks();
  }

  private updateMicroSplinters(dt: number): void {
    for (let i = this.splinters.length - 1; i >= 0; i--) {
      const sp = this.splinters[i];
      sp.life += dt;
      if (sp.life >= sp.maxLife || sp.pos[1] < -2.5) {
        this.splinters.splice(i, 1);
        continue;
      }

      sp.vel[1] += GRAVITY_Y * dt;
      sp.pos[0] += sp.vel[0] * dt;
      sp.pos[1] += sp.vel[1] * dt;
      sp.pos[2] += sp.vel[2] * dt;
      sp.rot += sp.rotSpeed * dt;

      // Bounce and skitter to rest on the oak table surface (y = 0)
      const rHoriz = Math.hypot(sp.pos[0], sp.pos[2]);
      if (sp.pos[1] < sp.scale * 0.35 && sp.pos[1] > -0.2 && rHoriz < TABLE_RADIUS) {
        sp.pos[1] = sp.scale * 0.35;
        if (Math.abs(sp.vel[1]) > 1.8) {
          sp.vel[1] = -sp.vel[1] * 0.28;
          sp.vel[0] *= 0.65;
          sp.vel[2] *= 0.65;
        } else {
          sp.vel[1] = 0;
          sp.vel[0] *= Math.max(0, 1 - 12 * dt);
          sp.vel[2] *= Math.max(0, 1 - 12 * dt);
          sp.rotSpeed *= Math.max(0, 1 - 12 * dt);
        }
      }
    }
  }

  private syncTransformsAndRules(): void {
    let maxTowerY = 0;
    let maxActiveLayer = 0;
    let sumX = 0;
    let sumZ = 0;
    let countInTower = 0;
    let maxTiltRad = 0;
    let sumWobbleOmega = 0;

    // Sync intact blocks
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.shattered) continue;
      const rb = this.world.getRigidBody(b.bodyHandle);
      if (!rb) continue;

      const t = rb.translation();
      const r = rb.rotation();
      const lv = rb.linvel();
      const av = rb.angvel();

      // Circular table rim drop-off: if a block slides past TABLE_RADIUS, let it fall off the circular edge!
      if (Math.hypot(t.x, t.z) > TABLE_RADIUS + 0.08) {
        const col = this.world.getCollider(b.colliderHandle);
        if (col) col.setCollisionGroups(GROUP_FREEFALL_OFF_RIM);
      }

      // Retire blocks that fell far off the pedestal table onto the stone floor
      if (t.y < -2.4) {
        this.shatterBlock(b, [0, 0, 0], 22.0, [0, 1, 0]);
        continue;
      }

      b.position[0] = t.x;
      b.position[1] = t.y;
      b.position[2] = t.z;
      b.rotation[0] = r.x;
      b.rotation[1] = r.y;
      b.rotation[2] = r.z;
      b.rotation[3] = r.w;
      b.linVel[0] = lv.x;
      b.linVel[1] = lv.y;
      b.linVel[2] = lv.z;
      b.angVel[0] = av.x;
      b.angVel[1] = av.y;
      b.angVel[2] = av.z;

      if (!b.extracted && b.id !== this.activeBlockId) {
        const horizR = Math.hypot(t.x, t.z);
        if (horizR < 0.85 && t.y > 0.04) {
          countInTower++;
          sumX += t.x;
          sumZ += t.z;
          if (t.y + BLOCK_HALF_Y > maxTowerY) {
            maxTowerY = t.y + BLOCK_HALF_Y;
          }
          if (b.assignedLayer > maxActiveLayer) {
            maxActiveLayer = b.assignedLayer;
          }
          // Measure local Y-axis deviation from world vertical [0, 1, 0]
          const upVec = quatRotateVec3(b.rotation, [0, 1, 0]);
          const tilt = Math.acos(Math.max(-1, Math.min(1, upVec[1])));
          if (tilt > maxTiltRad) maxTiltRad = tilt;
          sumWobbleOmega += Math.hypot(av.x, av.z);
        }
      }
    }

    // Sync shards & prune shards that fell off table edge
    for (let i = this.shards.length - 1; i >= 0; i--) {
      const s = this.shards[i];
      const rb = this.world.getRigidBody(s.bodyHandle);
      if (!rb) {
        this.shards.splice(i, 1);
        continue;
      }
      const t = rb.translation();
      if (Math.hypot(t.x, t.z) > TABLE_RADIUS + 0.04) {
        const col = this.world.getCollider(s.colliderHandle);
        if (col) col.setCollisionGroups(GROUP_SHARD_OFF_RIM);
      }
      if (t.y < -3.2) {
        this.colliderToShard.delete(s.colliderHandle);
        this.world.removeRigidBody(rb);
        this.shards.splice(i, 1);
        continue;
      }
      const r = rb.rotation();
      s.position[0] = t.x;
      s.position[1] = t.y;
      s.position[2] = t.z;
      s.rotation[0] = r.x;
      s.rotation[1] = r.y;
      s.rotation[2] = r.z;
      s.rotation[3] = r.w;
    }

    // Update telemetry
    if (countInTower > 0) {
      this.centerOfMassXZCm = [(sumX / countInTower) * 10, (sumZ / countInTower) * 10];
      const instantLean = (maxTiltRad * 180) / Math.PI;
      this.leanAngleDeg = this.leanAngleDeg * 0.75 + Math.min(45, instantLean) * 0.25;
      const avgOmegaDeg = (((sumWobbleOmega * SIM_TIME_SCALE) / countInTower) * 180) / Math.PI;
      this.wobbleSpeedDegPerSec = this.wobbleSpeedDegPerSec * 0.7 + avgOmegaDeg * 0.3;
    }

    this.currentLayers = maxActiveLayer + 1;
    const heightCm = maxTowerY * 10.0;
    if (this.turnPhase !== 'COLLAPSED' && heightCm > this.peakHeightCm) {
      this.peakHeightCm = heightCm;
    }

    // Update legal Jenga extraction flags:
    // Count how many intact in-tower blocks occupy `maxActiveLayer`
    let topLayerOccupancy = 0;
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.assignedLayer === maxActiveLayer) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.85) {
          topLayerOccupancy++;
        }
      }
    }
    const forbiddenMinLayer = topLayerOccupancy >= 3 ? maxActiveLayer : Math.max(0, maxActiveLayer - 1);

    for (const b of this.blocks) {
      b.isLegalTarget =
        !b.shattered &&
        !b.extracted &&
        (this.gameMode === 'sandbox' || b.assignedLayer < forbiddenMinLayer);
    }

    this.computeOpenTopSlots();
  }

  private getLayerSupportCenter(layer: number): [number, number] {
    let sx = 0, sz = 0, n = 0;
    const refLayer = Math.max(0, layer - 1);
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.assignedLayer === refLayer) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.9) {
          sx += b.position[0];
          sz += b.position[2];
          n++;
        }
      }
    }
    return n > 0 ? [sx / n, sz / n] : [0, 0];
  }

  /**
   * Computes the exact 3D poses of the 1..3 open placement slots on the current top layer,
   * tracking the live leaning center of the supporting layer directly underneath!
   */
  private computeOpenTopSlots(): void {
    let highestLayer = 0;
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.85 && b.assignedLayer > highestLayer) {
          highestLayer = b.assignedLayer;
        }
      }
    }

    const occupiedInHighest = new Set<number>();
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId && b.assignedLayer === highestLayer) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.85) {
          occupiedInHighest.add(b.assignedSlot);
        }
      }
    }

    const targetLayer = occupiedInHighest.size >= 3 ? highestLayer + 1 : highestLayer;
    const occupiedSlots = targetLayer === highestLayer ? occupiedInHighest : new Set<number>();

    // Measure live center and top surface Y of supporting layer (targetLayer - 1)
    let supX = 0, supZ = 0, supTopY = targetLayer * BLOCK_HEIGHT, supCount = 0;
    for (const b of this.blocks) {
      if (!b.shattered && !b.extracted && b.id !== this.activeBlockId && b.assignedLayer === targetLayer - 1) {
        if (Math.hypot(b.position[0], b.position[2]) < 0.9) {
          supX += b.position[0];
          supZ += b.position[2];
          supTopY = Math.max(supTopY, b.position[1] + BLOCK_HALF_Y);
          supCount++;
        }
      }
    }
    const cx = supCount > 0 ? supX / supCount : 0;
    const cz = supCount > 0 ? supZ / supCount : 0;

    const isOdd = targetLayer % 2 === 1;
    const yawRad = isOdd ? Math.PI * 0.5 : 0.0;
    const rot = quatFromAxisAngle([0, 1, 0], yawRad);

    const slots: TopPlacementSlot[] = [];
    for (let slotIndex = 0; slotIndex < 3; slotIndex++) {
      if (occupiedSlots.has(slotIndex)) continue;
      const offset = (slotIndex - 1) * BLOCK_WIDTH;
      const px = cx + (isOdd ? offset : 0);
      const py = supTopY + BLOCK_HALF_Y + 0.002;
      const pz = cz + (isOdd ? 0 : offset);
      slots.push({
        layer: targetLayer,
        slotIndex,
        position: [px, py, pz],
        rotation: rot,
        yawRad,
      });
    }
    this.openTopSlots = slots;
  }

  private updateTurnStateMachine(dt: number): void {
    if (this.gameMode === 'sandbox' || this.turnPhase === 'COLLAPSED') return;
    if (this.simTime < 0.4) return;

    // Check if any tower block has fallen onto the table or shattered!
    let fallenBlocks = this.shatteredCount;
    for (const b of this.blocks) {
      if (b.shattered || b.extracted || b.id === this.activeBlockId) continue;
      // Base layer (assignedLayer === 0) naturally rests at y = 0.075 dm;
      // any block from layer >= 1 dropping to table level (y < 0.14 dm) or tumbling outside 1.15 dm radius has fallen!
      const rHoriz = Math.hypot(b.position[0], b.position[2]);
      if ((b.assignedLayer >= 1 && b.position[1] < BLOCK_HEIGHT * 0.92) || rHoriz > 1.25) {
        fallenBlocks++;
      }
    }

    if (fallenBlocks > 0) {
      this.turnPhase = 'COLLAPSED';
      this.isPointerDraggingBlock = false;
      // If player was carrying an extracted block when tower collapsed, let it fall naturally too!
      if (this.activeBlockId !== null) {
        const carried = this.blocks[this.activeBlockId];
        if (carried && !carried.shattered) {
          carried.extracted = false;
          const col = this.world.getCollider(carried.colliderHandle);
          const rb = this.world.getRigidBody(carried.bodyHandle);
          if (col) col.setSensor(false);
          if (rb) rb.setGravityScale(1.0, true);
        }
        this.activeBlockId = null;
      }
      this.collapseReason =
        this.shatteredCount > 0
          ? 'The stained-glass tower gave way and shattered across the oak table!'
          : 'The tower lost equilibrium and tumbled!';
      return;
    }

    if (this.turnPhase === 'SETTLING') {
      this.settleTimer += dt;
      const towerQuiet = this.wobbleSpeedDegPerSec < 2.2;
      if (this.settleTimer >= this.SETTLE_DURATION || (this.settleTimer > 0.65 && towerQuiet)) {
        this.blocksStacked++;
        this.audio.playStackSuccess(this.currentLayers);
        if (this.gameMode === 'party') {
          this.currentPlayer = (this.currentPlayer + 1) % this.playerCount;
        }
        this.turnPhase = 'SELECT_BLOCK';
        this.settleTimer = 0;
      }
    }
  }

  public getTelemetry(): TowerTelemetry {
    return {
      turnPhase: this.turnPhase,
      gameMode: this.gameMode,
      currentPlayer: this.currentPlayer,
      playerCount: this.playerCount,
      blocksStacked: this.blocksStacked,
      currentLayers: this.currentLayers,
      peakHeightCm: +this.peakHeightCm.toFixed(1),
      leanAngleDeg: +this.leanAngleDeg.toFixed(2),
      wobbleSpeedDegPerSec: +this.wobbleSpeedDegPerSec.toFixed(1),
      centerOfMassXZ: [
        +this.centerOfMassXZCm[0].toFixed(2),
        +this.centerOfMassXZCm[1].toFixed(2),
      ],
      shatteredCount: this.shatteredCount,
      intactCount: this.blocks.filter((b) => !b.shattered).length,
      activeShardCount: this.shards.length,
      grabResistance01: +this.grabResistance01.toFixed(2),
      hoveredBlockId: this.hoveredBlockId,
      activeBlockId: this.activeBlockId,
      hoveredSlotIndex: this.hoveredSlotIndex,
      settleProgress01:
        this.turnPhase === 'SETTLING'
          ? Math.min(1.0, this.settleTimer / this.SETTLE_DURATION)
          : 0,
      collapseReason: this.collapseReason,
    };
  }

  public dispose(): void {
    if (this.world) this.world.free();
  }
}

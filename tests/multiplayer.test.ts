import { describe, expect, it } from 'vitest';
import { GameEngine } from '../src/engine/GameEngine';
import {
  BLOCK_HALF_X,
  BLOCK_HALF_Y,
  BLOCK_HALF_Z,
  SHARD_SLOT_STRIDE_VERTS,
  STAINED_GLASS_PALETTE,
} from '../src/physics/constants';
import { fractureBlockVoronoi } from '../src/physics/voronoiFracture';
import { quatRotateVec3, Vec3 } from '../src/math/vecMat';
import {
  applyDeterministicTowerPalette,
  buildDeterministicTowerMaterials,
  canonicalizeQuatHemisphere,
  decodeMultiplayerStateFrame,
  encodeMultiplayerStateFrame,
  inferShardLocalCentroid,
  MP_MAX_SHARDS_PER_FRAME,
  MP_MAX_SPLINTERS_PER_FRAME,
  mulberry32,
  MultiplayerControlMessage,
  MultiplayerGuestWorld,
  MultiplayerStateFrame,
  quatSlerpHemisphere,
  reconstructShardVerticesFromMeta,
  SerializedShardMeta,
  VitrailMultiplayerClient,
} from '../src/multiplayer';

interface MinimalHttpServer {
  listen(port: number, host: string, cb: () => void): void;
  address(): { port: number } | string | null;
  close(cb?: () => void): void;
}

interface MinimalHttpModule {
  createServer(): MinimalHttpServer;
}

interface NodeRelayModule {
  isAllowedRelayOrigin(origin: string | null | undefined): boolean;
  attachVitrailRoomRelay(
    server: MinimalHttpServer,
    pathPrefix?: string
  ): { detach(): void };
}

function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitUntil(
  predicate: () => boolean,
  timeoutMs: number = 4000,
  onTick?: () => void
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    onTick?.();
    if (predicate()) return;
    await sleepMs(12);
  }
  onTick?.();
  if (!predicate()) {
    throw new Error(`Timed out after ${timeoutMs}ms waiting for multiplayer condition.`);
  }
}

function worldToScreenPx(
  engine: GameEngine,
  worldPt: Vec3,
  vpWidth: number,
  vpHeight: number
): [number, number] {
  const m = engine.camera.viewProjMatrix;
  const x = worldPt[0];
  const y = worldPt[1];
  const z = worldPt[2];
  const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
  const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  const ndcX = cx / cw;
  const ndcY = cy / cw;
  return [(ndcX * 0.5 + 0.5) * vpWidth, (0.5 - ndcY * 0.5) * vpHeight];
}

describe('Vitrail Internet Multiplayer, Binary Codec, Ergonomics & Relay Integration', () => {
  it('T-MP1: encodes 54-block + 120-shard state snapshots under 4 KB with lossless field round-trip', () => {
    const blocks = Array.from({ length: 54 }, (_, i) => ({
      position: [(i % 3 - 1) * 0.252, (Math.floor(i / 3) + 0.5) * 0.15, 0.01 * (i % 5)] as Vec3,
      rotation: canonicalizeQuatHemisphere([0.05, i % 2 === 0 ? 0 : Math.SQRT1_2, -0.03, Math.SQRT1_2]),
      damage: (i % 7) / 6,
      shattered: i === 5,
      extracted: i === 12,
      isLegalTarget: i < 51,
      isLooseSlot: i % 3 === 1,
      swatchId: i % 8,
      motif: (i % 4) as 0 | 1 | 2 | 3,
      assignedLayer: Math.floor(i / 3),
      assignedSlot: i % 3,
      stress01: (i % 10) / 9,
      crackLocalOrigin: [0.22, -0.04, 0.08] as Vec3,
      photoelasticWrench: [0.25, 0.6, 0.85, 0.4] as [number, number, number, number],
    }));

    const baseFrame: MultiplayerStateFrame = {
      seq: 42,
      hostElapsedSec: 18.75,
      roomSeed: 0x9abc1234,
      turnPhase: 'PULLING',
      gameMode: 'party',
      currentPlayer: 1,
      playerCount: 2,
      blocksStacked: 7,
      shatteredCount: 1,
      currentLayers: 20,
      peakLayers: 20,
      activeBlockId: 12,
      hoveredBlockId: 12,
      hoveredSlotIndex: 2,
      lockedMinRow: 19,
      isPointerDraggingBlock: true,
      isExtractionLiftReady: false,
      polariscopeMode: true,
      grabResistance01: 0.64,
      settleProgress01: 0.2,
      leanAngleDeg: 1.45,
      wobbleSpeedDegPerSec: 3.8,
      liveHeightCm: 30.0,
      peakHeightCm: 30.0,
      centerOfMassXZCm: [0.42, -0.31],
      carriedYawOffsetRad: 0.26,
      grabCurrentWorldOnBlock: [0.31, 0.975, 0.02],
      grabWorldAnchor: [0.58, 0.975, 0.04],
      carriedLandingPos: [0.0, 3.075, -0.252],
      carriedTargetQuat: [0, Math.SQRT1_2, 0, Math.SQRT1_2],
      topSupportSurfaceCenter: [0, 3.0, 0],
      topSupportUpNormal: [0.01, 0.9999, -0.01],
      topSupportDefaultRot: [0, Math.SQRT1_2, 0, Math.SQRT1_2],
      topSupportTargetLayer: 20,
      blocks,
      shards: [],
      splinters: [],
    };

    const towerOnlyBuf = encodeMultiplayerStateFrame(baseFrame);
    expect(towerOnlyBuf.byteLength).toBe(1868);
    expect(towerOnlyBuf.byteLength).toBeLessThan(4096);

    const maxShards = Array.from({ length: MP_MAX_SHARDS_PER_FRAME }, (_, i) => ({
      id: 100 + i,
      position: [Math.sin(i) * 1.8, 0.08 + (i % 6) * 0.05, Math.cos(i) * 1.8] as Vec3,
      rotation: canonicalizeQuatHemisphere([0.2, -0.4, 0.5, -0.74]),
    }));
    const maxSplinters = Array.from({ length: MP_MAX_SPLINTERS_PER_FRAME }, (_, i) => ({
      pos: [0.1 * i, 0.3, -0.1 * i] as Vec3,
      vel: [0, 0, 0] as Vec3,
      rot: 1.2,
      rotSpeed: 0,
      scale: 0.06,
      swatchId: i % 8,
      life: 0.75,
      maxLife: 1.0,
    }));

    const peakCollapseBuf = encodeMultiplayerStateFrame({
      ...baseFrame,
      shards: maxShards,
      splinters: maxSplinters,
    });
    expect(peakCollapseBuf.byteLength).toBe(3908);
    expect(peakCollapseBuf.byteLength).toBeLessThan(4096);

    const decoded = decodeMultiplayerStateFrame(peakCollapseBuf);
    expect(decoded).not.toBeNull();
    if (!decoded) return;

    expect(decoded.seq).toBe(42);
    expect(decoded.roomSeed).toBe(0x9abc1234);
    expect(decoded.turnPhase).toBe('PULLING');
    expect(decoded.currentPlayer).toBe(1);
    expect(decoded.blocksStacked).toBe(7);
    expect(decoded.polariscopeMode).toBe(true);
    expect(decoded.activeBlockId).toBe(12);
    expect(decoded.hoveredSlotIndex).toBe(2);
    expect(decoded.blocks.length).toBe(54);
    expect(decoded.shards.length).toBe(MP_MAX_SHARDS_PER_FRAME);
    expect(decoded.splinters.length).toBe(MP_MAX_SPLINTERS_PER_FRAME);

    expect(decoded.blocks[12].extracted).toBe(true);
    expect(Math.abs(decoded.blocks[12].position[0] - blocks[12].position[0])).toBeLessThan(1e-6);
    expect(Math.abs(decoded.blocks[12].photoelasticWrench[2] - 0.85)).toBeLessThan(0.01);
    expect(Math.abs(decoded.blocks[12].crackLocalOrigin[0] - 0.22)).toBeLessThan(0.006);
    expect(Math.abs(decoded.shards[17].position[0] - maxShards[17].position[0])).toBeLessThan(0.001);
  });

  it('T-MP2: enforces antipodal quaternion hemisphere continuity (dot < 0 guard) and 36 ms 120 Hz Guest smoothing', () => {
    const qPos: [number, number, number, number] = [0.12, 0.985, -0.08, 0.095];
    const qLen = Math.hypot(...qPos);
    const qA: [number, number, number, number] = [
      qPos[0] / qLen,
      qPos[1] / qLen,
      qPos[2] / qLen,
      qPos[3] / qLen,
    ];
    // Slight 4-degree rotation across w = 0 equator represented with negated hemisphere (-q)
    const rawNext: [number, number, number, number] = [-0.13, -0.988, 0.075, 0.025];
    const nLen = Math.hypot(...rawNext);
    const qB: [number, number, number, number] = [
      rawNext[0] / nLen,
      rawNext[1] / nLen,
      rawNext[2] / nLen,
      rawNext[3] / nLen,
    ];

    // Raw dot product is strongly negative (~ -0.99), which would cause a 356-deg spin pop without hemisphere guard
    const rawDot = qA[0] * qB[0] + qA[1] * qB[1] + qA[2] * qB[2] + qA[3] * qB[3];
    expect(rawDot).toBeLessThan(-0.95);

    const vRef: Vec3 = [1, 0, 0];
    const vStart = quatRotateVec3(qA, vRef);
    const vEnd = quatRotateVec3(qB, vRef);
    for (let step = 1; step <= 8; step++) {
      const qMid = quatSlerpHemisphere(qA, qB, step / 8);
      const vMid = quatRotateVec3(qMid, vRef);
      const alignStart = vMid[0] * vStart[0] + vMid[1] * vStart[1] + vMid[2] * vStart[2];
      const alignEnd = vMid[0] * vEnd[0] + vMid[1] * vEnd[1] + vMid[2] * vEnd[2];
      expect(alignStart).toBeGreaterThan(0.96);
      expect(alignEnd).toBeGreaterThan(0.96);
    }

    const engine = new GameEngine();
    const guestWorld = new MultiplayerGuestWorld(engine.audio, 18, 0x55aa77cc);
    expect(guestWorld.interpolationTauSec).toBeGreaterThanOrEqual(0.03);
    expect(guestWorld.interpolationTauSec).toBeLessThanOrEqual(0.045);

    const makeMinimalFrame = (seq: number, xPos: number, quat: [number, number, number, number]): MultiplayerStateFrame => ({
      seq,
      hostElapsedSec: seq * 0.033,
      roomSeed: 0x55aa77cc,
      turnPhase: 'PULLING',
      gameMode: 'party',
      currentPlayer: 0,
      playerCount: 2,
      blocksStacked: 0,
      shatteredCount: 0,
      currentLayers: 18,
      peakLayers: 18,
      activeBlockId: 4,
      hoveredBlockId: 4,
      hoveredSlotIndex: null,
      lockedMinRow: 18,
      isPointerDraggingBlock: true,
      isExtractionLiftReady: false,
      polariscopeMode: false,
      grabResistance01: 0.3,
      settleProgress01: 0,
      leanAngleDeg: 0.1,
      wobbleSpeedDegPerSec: 0.2,
      liveHeightCm: 27,
      peakHeightCm: 27,
      centerOfMassXZCm: [0, 0],
      carriedYawOffsetRad: 0,
      grabCurrentWorldOnBlock: [xPos, 0.225, 0],
      grabWorldAnchor: [xPos + 0.1, 0.225, 0],
      carriedLandingPos: [0, 2.777, 0],
      carriedTargetQuat: [0, 0, 0, 1],
      topSupportSurfaceCenter: [0, 2.7, 0],
      topSupportUpNormal: [0, 1, 0],
      topSupportDefaultRot: [0, 0, 0, 1],
      topSupportTargetLayer: 18,
      blocks: guestWorld.blocks.map((b, idx) => ({
        position: idx === 4 ? [xPos, b.position[1], b.position[2]] : [...b.position],
        rotation: idx === 4 ? quat : [...b.rotation],
        damage: 0,
        shattered: false,
        extracted: false,
        isLegalTarget: b.isLegalTarget,
        isLooseSlot: b.isLooseSlot,
        swatchId: b.swatchId,
        motif: b.motif,
        assignedLayer: b.assignedLayer,
        assignedSlot: b.assignedSlot,
        stress01: 0,
        crackLocalOrigin: [0, 0, 0],
        photoelasticWrench: [0, 0, 0, 0],
      })),
      shards: [],
      splinters: [],
    });

    expect(guestWorld.applyNetworkStateFrame(makeMinimalFrame(10, 0.10, qA))).toBe(true);
    expect(guestWorld.blocks[4].position[0]).toBeCloseTo(0.10, 4);
    expect(guestWorld.applyNetworkStateFrame(makeMinimalFrame(11, 0.22, qB))).toBe(true);
    // Out-of-order stale frame seq=9 must be ignored
    expect(guestWorld.applyNetworkStateFrame(makeMinimalFrame(9, -0.5, qA))).toBe(false);

    let prevX = guestWorld.blocks[4].position[0];
    for (let f = 0; f < 6; f++) {
      guestWorld.step(1 / 120);
      const curX = guestWorld.blocks[4].position[0];
      expect(curX).toBeGreaterThan(prevX);
      expect(curX).toBeLessThan(0.22);
      prevX = curX;
    }
    guestWorld.dispose();
    engine.dispose();
  });

  it('T-MP3: reconstructs 3D Voronoi shard GPU meshes from compact plane metadata with outer/cleavage parity', () => {
    const pieces = fractureBlockVoronoi(
      [0.28, 0.01, -0.03],
      12,
      1337,
      BLOCK_HALF_X,
      BLOCK_HALF_Y,
      BLOCK_HALF_Z,
      2,
      0.85
    );
    expect(pieces.length).toBeGreaterThanOrEqual(8);

    let totalOuterTris = 0;
    let totalInnerCleavageTris = 0;

    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i];
      const lc = inferShardLocalCentroid(
        p.planes,
        p.planeCount,
        [0.28, 0.01, -0.03],
        p.renderVertices
      );
      expect(Math.hypot(lc[0] - p.localCentroid[0], lc[1] - p.localCentroid[1], lc[2] - p.localCentroid[2])).toBeLessThan(1e-4);

      const meta: SerializedShardMeta = {
        id: i + 1,
        slotIndex: i,
        swatchId: 3,
        motif: 2,
        rippleSeed: 4.2,
        localCentroid: lc,
        impactLocalPos: [0.28, 0.01, -0.03],
        planeCount: p.planeCount,
        planes: Array.from(p.planes.subarray(0, p.planeCount * 4)),
      };

      const rebuilt = reconstructShardVerticesFromMeta(meta);
      expect(rebuilt.vertexCount).toBeGreaterThanOrEqual(12);
      expect(rebuilt.vertexCount).toBeLessThanOrEqual(SHARD_SLOT_STRIDE_VERTS);
      expect(rebuilt.vertexCount % 3).toBe(0);

      for (let v = 0; v < rebuilt.vertexCount; v++) {
        const base = v * 10;
        const nx = rebuilt.renderVertices[base + 3];
        const ny = rebuilt.renderVertices[base + 4];
        const nz = rebuilt.renderVertices[base + 5];
        const bx = rebuilt.renderVertices[base + 6];
        const by = rebuilt.renderVertices[base + 7];
        const bz = rebuilt.renderVertices[base + 8];
        const isOuter = rebuilt.renderVertices[base + 9];

        expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1.0, 3);
        expect(Math.abs(bx)).toBeLessThanOrEqual(BLOCK_HALF_X + 1e-3);
        expect(Math.abs(by)).toBeLessThanOrEqual(BLOCK_HALF_Y + 1e-3);
        expect(Math.abs(bz)).toBeLessThanOrEqual(BLOCK_HALF_Z + 1e-3);
        if (v % 3 === 0) {
          if (isOuter > 0.5) totalOuterTris++;
          else totalInnerCleavageTris++;
        }
      }
    }

    expect(totalOuterTris).toBeGreaterThan(0);
    expect(totalInnerCleavageTris).toBeGreaterThan(0);
  });

  it('T-MP4: guarantees 100% deterministic stained-glass tower palette parity from roomSeed', async () => {
    const host = new GameEngine();
    await host.initHeadless();
    const guestMirror = new MultiplayerGuestWorld(host.audio, 18, 0x7a4e19c3);
    applyDeterministicTowerPalette(host.physics.blocks, 0x7a4e19c3);

    expect(host.physics.blocks.length).toBe(54);
    expect(guestMirror.blocks.length).toBe(54);

    for (let i = 0; i < 54; i++) {
      expect(guestMirror.blocks[i].swatchId).toBe(host.physics.blocks[i].swatchId);
      expect(guestMirror.blocks[i].motif).toBe(host.physics.blocks[i].motif);
      expect(guestMirror.blocks[i].rippleSeed).toBeCloseTo(host.physics.blocks[i].rippleSeed, 5);
      expect(guestMirror.blocks[i].isLooseSlot).toBe(host.physics.blocks[i].isLooseSlot);
    }

    const otherMirror = new MultiplayerGuestWorld(host.audio, 18, 0x11223344);
    let diffs = 0;
    for (let i = 0; i < 54; i++) {
      if (otherMirror.blocks[i].swatchId !== guestMirror.blocks[i].swatchId) diffs++;
    }
    expect(diffs).toBeGreaterThan(20);

    otherMirror.dispose();
    guestMirror.dispose();
    host.dispose();
  });

  it('T-MP5: preserves 1.0x isometric radial pull gain at steep upward pitches and >= 300 shards on collapse-300', async () => {
    const engine = new GameEngine();
    await engine.initHeadless();

    // 1. Test steep upward pitches (-25 deg and -65 deg where cos(tiltBeta) = 0)
    for (const elevDeg of [-25, -65]) {
      const anchor: Vec3 = [0, 3.3, 0];
      const horizDist = 2.8;
      const elevRad = (elevDeg * Math.PI) / 180;
      const camOrigin: Vec3 = [0, anchor[1] + horizDist * Math.tan(elevRad), horizDist];

      const toAnchor: Vec3 = [
        anchor[0] - camOrigin[0],
        anchor[1] - camOrigin[1],
        anchor[2] - camOrigin[2],
      ];
      const len0 = Math.hypot(toAnchor[0], toAnchor[1], toAnchor[2]);
      const dir0: Vec3 = [toAnchor[0] / len0, toAnchor[1] / len0, toAnchor[2] / len0];

      // Zero grab-start jump
      const pStart = engine.projectBlockDragPlane(camOrigin, dir0, anchor);
      expect(pStart).not.toBeNull();
      if (!pStart) continue;
      expect(
        Math.hypot(pStart[0] - anchor[0], pStart[1] - anchor[1], pStart[2] - anchor[2])
      ).toBeLessThan(1e-5);

      // Drag downward in view (toward camera along the tilted plane): must yield full radial +Z motion
      const tiltBeta = Math.min(Math.PI * 0.5, (25 * Math.PI) / 180 - elevRad);
      const targetPullDm = 0.95;
      const inPlaneSample: Vec3 = [
        anchor[0],
        anchor[1] - Math.sin(tiltBeta) * targetPullDm,
        anchor[2] + Math.cos(tiltBeta) * targetPullDm,
      ];
      const toSample: Vec3 = [
        inPlaneSample[0] - camOrigin[0],
        inPlaneSample[1] - camOrigin[1],
        inPlaneSample[2] - camOrigin[2],
      ];
      const lenS = Math.hypot(toSample[0], toSample[1], toSample[2]);
      const dirS: Vec3 = [toSample[0] / lenS, toSample[1] / lenS, toSample[2] / lenS];

      const pPulled = engine.projectBlockDragPlane(camOrigin, dirS, anchor);
      expect(pPulled).not.toBeNull();
      if (!pPulled) continue;
      // Exact 1.00x isometric unfolding (0.95 dm radial pull preserved even when cos(tiltBeta) == 0!)
      expect(pPulled[2] - anchor[2]).toBeCloseTo(targetPullDm, 3);
      expect(pPulled[1]).toBeCloseTo(anchor[1], 5);
    }

    // 2. Verify configureBenchScene('collapse-300') retains >= 300 active Voronoi shards on the 43 cm oak table
    (engine as unknown as { renderer: Record<string, unknown> }).renderer = {
      lightingPreset: null,
      lightingOverride: null,
      referenceCardLuminance: 0,
      benchHideBlocks: false,
      dispose: () => {},
    };
    engine.configureBenchScene('collapse-300');
    expect(engine.physics.shards.length).toBeGreaterThanOrEqual(300);

    engine.dispose();
  });

  it(
    'T-MP6: runs full two-peer Host <-> Guest online match, turn-lock, remote extraction, shatter sync, reconnect resync & rematch over real TCP WebSocket relay',
    async () => {
      const httpModName = 'node:http';
      const relayModPath = '../workers/vitrail-rooms/nodeRelay.mjs';
      const http = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
      const { attachVitrailRoomRelay } = (await import(
        /* @vite-ignore */ relayModPath
      )) as unknown as NodeRelayModule;

      const server = http.createServer();
      const relay = attachVitrailRoomRelay(server, '/__vitrail_mp');
      await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
      const addr = server.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      expect(port).toBeGreaterThan(0);
      const relayUrl = `ws://127.0.0.1:${port}/__vitrail_mp`;

      const host = new GameEngine();
      const guest = new GameEngine();
      let guest2: GameEngine | null = null;

      try {
        await host.initHeadless();
        await guest.initHeadless();

        const roomCode = host.startOnlineHost('K7M4P', {
          roomSeed: 0x31415926,
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: false,
        });
        expect(roomCode).toBe('K7M4P');

        const joined = guest.joinOnlineRoom('k7m4p', {
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: false,
        });
        expect(joined).toBe(true);

        const pumpBoth = (frames: number = 1) => {
          for (let i = 0; i < frames; i++) {
            host.stepHeadless(1 / 60);
            guest.stepHeadless(1 / 60);
          }
        };

        await waitUntil(
          () =>
            Boolean(host.getTelemetrySnapshot().mpPeerConnected) &&
            Boolean(guest.getTelemetrySnapshot().mpPeerConnected) &&
            guest.mpRoomSeed === 0x31415926 &&
            guest.getTelemetrySnapshot().mpPingMs !== null,
          4000,
          () => pumpBoth(2)
        );

        const hostTel0 = host.getTelemetrySnapshot();
        const guestTel0 = guest.getTelemetrySnapshot();
        expect(hostTel0.mpTransport).toBe('ws-relay');
        expect(guestTel0.mpTransport).toBe('ws-relay');
        expect(guestTel0.mpPingMs).toBeGreaterThanOrEqual(0);
        expect(guest.physics.blocks[7].swatchId).toBe(host.physics.blocks[7].swatchId);

        // --- 1. Turn-Lock Check on Turn 0 (Host's Turn: currentPlayer === 0) ---
        expect(host.physics.currentPlayer).toBe(0);
        expect(guest.physics.currentPlayer).toBe(0);

        const vp = { left: 0, top: 0, width: 1280, height: 800 };
        guest.camera.setPoseImmediate(0, 0.35, 4.2, [0, 1.35, 0], 1280 / 800);
        const looseTurn0 = host.physics.blocks.find(
          (b) => b.assignedLayer === 8 && b.isLooseSlot && b.isLegalTarget
        )!;
        const [gx0, gy0] = worldToScreenPx(guest, looseTurn0.position, vp.width, vp.height);

        // Guest attempts to grab on Host's turn -> must fall back to camera orbit locally
        guest.handlePointerDown(gx0, gy0, 1, 0, false, vp);
        guest.handlePointerMove(gx0 + 40, gy0, 1, vp);
        guest.handlePointerUp(gx0 + 40, gy0, 1, vp);

        // Also inject a spoofed raw wire command from Guest while currentPlayer === 0 -> Host must reject it!
        const rawGuestClient = (guest as unknown as { mpClient: VitrailMultiplayerClient }).mpClient;
        rawGuestClient.sendControl({
          type: 'guest_cmd',
          seq: 777,
          cmd: {
            kind: 'begin_pull',
            blockId: looseTurn0.id,
            localPoint: [0.3, 0, 0],
            worldPoint: [...looseTurn0.position],
          },
        });
        await sleepMs(30);
        pumpBoth(4);
        expect(host.physics.activeBlockId).toBeNull();
        expect(host.physics.turnPhase).toBe('SELECT_BLOCK');

        // --- 2. Host Completes Turn 0 -> Advances Turn Ownership to Guest (currentPlayer === 1) ---
        const hostAxisX = quatRotateVec3(looseTurn0.rotation, [1, 0, 0]);
        const startWorld0: Vec3 = [...looseTurn0.position];
        expect(host.physics.startGrabBlock(looseTurn0.id, [0, 0, 0], startWorld0)).toBe(true);
        for (let step = 1; step <= 95; step++) {
          const dist = Math.min(1.4, (step / 65) * 1.4);
          host.physics.updateGrabTarget([
            startWorld0[0] + hostAxisX[0] * dist,
            startWorld0[1],
            startWorld0[2] + hostAxisX[2] * dist,
          ]);
          pumpBoth(1);
          if (host.physics.turnPhase === 'EXTRACTED') break;
        }
        expect(host.physics.turnPhase).toBe('EXTRACTED');

        // Wait until lift ready and release onto top
        for (let i = 0; i < 45 && !host.physics.isExtractionLiftReady; i++) {
          pumpBoth(1);
        }
        expect(host.physics.isExtractionLiftReady).toBe(true);
        const slot0 = host.physics.openTopSlots[0].position;
        host.physics.updateCarriedHover([slot0[0], slot0[1] + 1.0, slot0[2]], [0, -1, 0]);
        host.handleReleaseOntoTop();

        await waitUntil(
          () =>
            host.physics.currentPlayer === 1 &&
            host.physics.turnPhase === 'SELECT_BLOCK' &&
            guest.physics.currentPlayer === 1 &&
            guest.physics.turnPhase === 'SELECT_BLOCK' &&
            guest.physics.blocksStacked === 1,
          5000,
          () => pumpBoth(4)
        );

        // --- 3. Guest Executes Turn 1 Remotely Over Network (`currentPlayer === 1`) ---
        // Host local grab must now be locked out to camera orbit!
        const candidateTurn1 = host.physics.blocks.find(
          (b) => b.assignedLayer === 6 && b.isLooseSlot && b.isLegalTarget && !b.extracted
        )!;
        host.camera.setPoseImmediate(0, 0.32, 4.2, [0, 1.1, 0], 1280 / 800);
        const [hx1, hy1] = worldToScreenPx(host, candidateTurn1.position, vp.width, vp.height);
        host.handlePointerDown(hx1, hy1, 1, 0, false, vp);
        host.handlePointerMove(hx1 + 35, hy1, 1, vp);
        host.handlePointerUp(hx1 + 35, hy1, 1, vp);
        expect(host.physics.activeBlockId).toBeNull();
        expect(host.physics.turnPhase).toBe('SELECT_BLOCK');

        // Position Guest camera facing candidateTurn1's exposed +X end face (completely unoccluded by sibling slots)
        const gAxisX = quatRotateVec3(candidateTurn1.rotation, [1, 0, 0]);
        const endFaceWorld: Vec3 = [
          candidateTurn1.position[0] + gAxisX[0] * (BLOCK_HALF_X * 0.85),
          candidateTurn1.position[1],
          candidateTurn1.position[2] + gAxisX[2] * (BLOCK_HALF_X * 0.85),
        ];
        const faceAz = Math.atan2(gAxisX[0], gAxisX[2]) + 0.38;
        guest.benchCameraLocked = true;
        guest.camera.setPoseImmediate(
          faceAz,
          0.34,
          3.8,
          [0, candidateTurn1.position[1], 0],
          vp.width / vp.height
        );

        const [gx1, gy1] = worldToScreenPx(guest, endFaceWorld, vp.width, vp.height);

        // 3a. Guest tap-tests block (< 5 px movement) -> forwards `tap_block` without locking turn
        guest.handlePointerDown(gx1, gy1, 1, 0, false, vp);
        guest.handlePointerUp(gx1 + 1, gy1, 1, vp);
        await sleepMs(25);
        pumpBoth(4);
        expect(host.physics.turnPhase).toBe('SELECT_BLOCK');

        // 3b. Guest drags outward from the exposed +X end face to extract `candidateTurn1` remotely!
        guest.camera.setPoseImmediate(
          faceAz,
          0.34,
          3.8,
          [0, candidateTurn1.position[1], 0],
          vp.width / vp.height
        );
        const [gxGrab, gyGrab] = worldToScreenPx(guest, endFaceWorld, vp.width, vp.height);
        const worldPlusEnd: Vec3 = [
          endFaceWorld[0] + gAxisX[0] * 1.45,
          endFaceWorld[1],
          endFaceWorld[2] + gAxisX[2] * 1.45,
        ];
        const [gxEnd, gyEnd] = worldToScreenPx(guest, worldPlusEnd, vp.width, vp.height);

        guest.handlePointerDown(gxGrab, gyGrab, 2, 0, false, vp);
        const dragSteps = 75;
        for (let s = 1; s <= dragSteps; s++) {
          const u = s / dragSteps;
          guest.handlePointerMove(
            gxGrab + (gxEnd - gxGrab) * u,
            gyGrab + (gyEnd - gyGrab) * u,
            2,
            vp
          );
          await sleepMs(16);
          pumpBoth(2);
          if (guest.physics.turnPhase === 'EXTRACTED') break;
        }
        guest.handlePointerUp(gxEnd, gyEnd, 2, vp);

        await waitUntil(
          () =>
            host.physics.turnPhase === 'EXTRACTED' &&
            host.physics.isExtractionLiftReady &&
            guest.physics.turnPhase === 'EXTRACTED' &&
            guest.physics.isExtractionLiftReady,
          4000,
          () => pumpBoth(3)
        );

        // 3c. Guest rotates carried block, resets alignment, and confirms placement onto top row!
        guest.handleRotateCarried(Math.PI / 12);
        await waitUntil(
          () => Math.abs(host.physics.carriedYawOffsetRad) > 0.1,
          2000,
          () => pumpBoth(2)
        );
        guest.handleResetCarriedRotation();
        await waitUntil(
          () => Math.abs(host.physics.carriedYawOffsetRad) < 1e-3,
          2000,
          () => pumpBoth(2)
        );
        guest.handleReleaseOntoTop();

        await waitUntil(
          () =>
            host.physics.blocksStacked === 2 &&
            host.physics.currentPlayer === 0 &&
            guest.physics.blocksStacked === 2 &&
            guest.physics.currentPlayer === 0,
          5000,
          () => pumpBoth(4)
        );

        // --- 4. Voronoi Shatter Replication + Disconnect/Reconnect Full-State Resync ---
        host.physics.triggerSandboxCollapseImpulse();
        await waitUntil(
          () =>
            host.physics.shards.length > 0 &&
            guest.physics.shards.length > 0,
          4000,
          () => pumpBoth(3)
        );
        const guestUploadedMeshes = guest.physics.consumePendingShardUploads();
        expect(guestUploadedMeshes.length).toBeGreaterThan(0);

        // Disconnect original Guest and reconnect a brand-new Guest2 into the live shattered room!
        guest.dispose();
        await sleepMs(40);

        guest2 = new GameEngine();
        await guest2.initHeadless();
        expect(
          guest2.joinOnlineRoom('K7M4P', {
            relayUrl,
            enablePeerJsFallback: false,
            enableWebRtcUpgrade: false,
          })
        ).toBe(true);

        await waitUntil(
          () =>
            Boolean(guest2?.getTelemetrySnapshot().mpPeerConnected) &&
            guest2?.mpRoomSeed === host.mpRoomSeed &&
            (guest2?.physics.shards.length ?? 0) > 0,
          4000,
          () => {
            host.stepHeadless(1 / 60);
            guest2?.stepHeadless(1 / 60);
          }
        );

        expect(guest2.physics.blocksStacked).toBe(host.physics.blocksStacked);
        expect(guest2.physics.shards.length).toBeGreaterThan(0);
        expect(guest2.physics.consumePendingShardUploads().length).toBeGreaterThan(0);

        // --- 5. Guest-Initiated Online Rematch Resets Both Peers with Fresh Deterministic Seed ---
        const oldSeed = host.mpRoomSeed;
        guest2.requestOnlineRematch();

        await waitUntil(
          () =>
            host.mpRoomSeed !== oldSeed &&
            guest2?.mpRoomSeed === host.mpRoomSeed &&
            host.physics.shards.length === 0 &&
            guest2?.physics.shards.length === 0 &&
            guest2?.physics.blocksStacked === 0,
          4000,
          () => {
            host.stepHeadless(1 / 60);
            guest2?.stepHeadless(1 / 60);
          }
        );

        expect(guest2.physics.blocks[0].swatchId).toBe(host.physics.blocks[0].swatchId);
      } finally {
        guest2?.dispose();
        guest.dispose();
        host.dispose();
        relay.detach();
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    },
    25000
  );

  it('T-MP7: negotiates WebRTC P2P dual DataChannels (ctrl id:0 + state id:1) over PeerJS signaling protocol', async () => {
    const origWs = globalThis.WebSocket;
    const origRtc = (globalThis as unknown as { RTCPeerConnection?: unknown }).RTCPeerConnection;

    // In-memory PeerJS signaling broker + paired negotiated RTCDataChannel mock
    const peerSockets = new Map<string, MockPeerJsWebSocket>();
    const rtcRegistry = new Map<string, MockRtcPeerConnection>();

    class MockDataChannel {
      public readonly label: string;
      public readonly id: number;
      public readonly ordered: boolean;
      public readonly maxRetransmits: number | null;
      public binaryType: string = 'arraybuffer';
      public bufferedAmount: number = 0;
      public readyState: 'connecting' | 'open' | 'closed' = 'connecting';
      public onopen: (() => void) | null = null;
      public onmessage: ((ev: { data: unknown }) => void) | null = null;
      public onclose: (() => void) | null = null;
      public peerChannel: MockDataChannel | null = null;

      constructor(label: string, opts: { id?: number; ordered?: boolean; maxRetransmits?: number }) {
        this.label = label;
        this.id = opts.id ?? 0;
        this.ordered = opts.ordered ?? true;
        this.maxRetransmits = opts.maxRetransmits ?? null;
      }

      public send(data: unknown): void {
        if (this.readyState !== 'open' || !this.peerChannel) return;
        const target = this.peerChannel;
        queueMicrotask(() => {
          if (target.readyState === 'open') {
            target.onmessage?.({ data });
          }
        });
      }

      public close(): void {
        if (this.readyState === 'closed') return;
        this.readyState = 'closed';
        this.onclose?.();
      }
    }

    class MockRtcPeerConnection {
      public readonly pcId = `pc_${Math.random().toString(36).slice(2, 9)}`;
      public localDescription: RTCSessionDescriptionInit | null = null;
      public remoteDescription: RTCSessionDescriptionInit | null = null;
      public signalingState: string = 'stable';
      public connectionState: string = 'new';
      public onicecandidate: ((ev: { candidate: { toJSON: () => RTCIceCandidateInit } | null }) => void) | null = null;
      public onconnectionstatechange: (() => void) | null = null;
      public channels = new Map<number, MockDataChannel>();
      public remotePc: MockRtcPeerConnection | null = null;

      constructor() {
        rtcRegistry.set(this.pcId, this);
      }

      public createDataChannel(
        label: string,
        opts: { id?: number; ordered?: boolean; maxRetransmits?: number }
      ): MockDataChannel {
        const ch = new MockDataChannel(label, opts);
        this.channels.set(ch.id, ch);
        return ch;
      }

      public async createOffer(): Promise<RTCSessionDescriptionInit> {
        return { type: 'offer', sdp: this.pcId };
      }

      public async createAnswer(): Promise<RTCSessionDescriptionInit> {
        return { type: 'answer', sdp: this.pcId };
      }

      public async setLocalDescription(desc: RTCSessionDescriptionInit): Promise<void> {
        this.localDescription = desc;
        if (desc.type === 'offer') this.signalingState = 'have-local-offer';
        else if (desc.type === 'answer') this.signalingState = 'stable';
        queueMicrotask(() => {
          this.onicecandidate?.({
            candidate: { toJSON: () => ({ candidate: 'candidate:1 1 UDP 2122252543 203.0.113.5 54321 typ host' }) },
          });
        });
      }

      public async setRemoteDescription(desc: RTCSessionDescriptionInit): Promise<void> {
        this.remoteDescription = desc;
        const other = desc.sdp ? rtcRegistry.get(desc.sdp) ?? null : null;
        if (other) {
          this.remotePc = other;
          other.remotePc = this;
        }
        if (desc.type === 'answer') {
          this.signalingState = 'stable';
          this.openNegotiatedChannels();
        }
      }

      public async addIceCandidate(): Promise<void> {
        if (this.localDescription && this.remoteDescription) {
          this.openNegotiatedChannels();
        }
      }

      private openNegotiatedChannels(): void {
        if (!this.remotePc) return;
        this.connectionState = 'connected';
        this.remotePc.connectionState = 'connected';
        for (const [id, chA] of this.channels.entries()) {
          const chB = this.remotePc.channels.get(id);
          if (chB && chA.readyState !== 'open') {
            chA.peerChannel = chB;
            chB.peerChannel = chA;
            chA.readyState = 'open';
            chB.readyState = 'open';
            queueMicrotask(() => {
              chA.onopen?.();
              chB.onopen?.();
            });
          }
        }
      }

      public close(): void {
        this.connectionState = 'closed';
        for (const ch of this.channels.values()) ch.close();
        rtcRegistry.delete(this.pcId);
      }
    }

    class MockPeerJsWebSocket {
      public static readonly OPEN = 1;
      public static readonly CLOSED = 3;
      public readyState: number = 0;
      public binaryType: string = 'arraybuffer';
      public peerId: string;
      public onopen: (() => void) | null = null;
      public onmessage: ((ev: { data: string }) => void) | null = null;
      public onclose: (() => void) | null = null;
      public onerror: (() => void) | null = null;

      constructor(urlStr: string) {
        const u = new URL(urlStr);
        this.peerId = u.searchParams.get('id') ?? '';
        queueMicrotask(() => {
          this.readyState = MockPeerJsWebSocket.OPEN;
          this.onopen?.();
          if (peerSockets.has(this.peerId)) {
            this.onmessage?.({ data: JSON.stringify({ type: 'ID-TAKEN' }) });
            return;
          }
          peerSockets.set(this.peerId, this);
          this.onmessage?.({ data: JSON.stringify({ type: 'OPEN' }) });
        });
      }

      public send(raw: string): void {
        const parsed = JSON.parse(raw) as {
          type: string;
          dst?: string;
          payload?: unknown;
        };
        if (parsed.type === 'HEARTBEAT') return;
        if (parsed.dst) {
          const dstSock = peerSockets.get(parsed.dst);
          if (dstSock && dstSock.readyState === MockPeerJsWebSocket.OPEN) {
            queueMicrotask(() => {
              dstSock.onmessage?.({
                data: JSON.stringify({
                  type: parsed.type,
                  src: this.peerId,
                  dst: parsed.dst,
                  payload: parsed.payload,
                }),
              });
            });
          } else {
            queueMicrotask(() => {
              this.onmessage?.({ data: JSON.stringify({ type: 'EXPIRE', src: parsed.dst }) });
            });
          }
        }
      }

      public close(): void {
        if (this.readyState === MockPeerJsWebSocket.CLOSED) return;
        this.readyState = MockPeerJsWebSocket.CLOSED;
        if (peerSockets.get(this.peerId) === this) {
          peerSockets.delete(this.peerId);
        }
        this.onclose?.();
      }
    }

    (globalThis as unknown as { WebSocket: unknown }).WebSocket = MockPeerJsWebSocket;
    (globalThis as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection =
      MockRtcPeerConnection;

    const receivedCtrlByGuest: MultiplayerControlMessage[] = [];
    let receivedBinaryBytesByGuest = 0;

    const hostClient = new VitrailMultiplayerClient({
      role: 'host',
      roomCode: 'R9X2Q',
      roomSeed: 0xabcdef01,
      relayUrl: null,
      enablePeerJsFallback: true,
    });
    const dupHostClient = new VitrailMultiplayerClient({
      role: 'host',
      roomCode: 'R9X2Q',
      relayUrl: null,
      enablePeerJsFallback: true,
    });
    const guestClient = new VitrailMultiplayerClient({
      role: 'guest',
      roomCode: 'R9X2Q',
      relayUrl: null,
      enablePeerJsFallback: true,
      onControlMessage: (msg) => receivedCtrlByGuest.push(msg),
      onStateFrame: (_frame, byteLen) => {
        receivedBinaryBytesByGuest = byteLen;
      },
    });

    try {
      hostClient.connect();
      await waitUntil(() => peerSockets.has('vitrail-v1-R9X2Q-host'), 1000);

      // Duplicate host on same room code receives ID-TAKEN error cleanly
      dupHostClient.connect();
      await waitUntil(() => dupHostClient.getStatus().connectionState === 'error', 1000);
      expect(dupHostClient.getStatus().errorMessage).toContain('already has an active Host');

      // Guest connects via PeerJS OFFER/ANSWER/CANDIDATE -> opens dual WebRTC DataChannels
      guestClient.connect();
      await waitUntil(
        () =>
          hostClient.getStatus().peerConnected &&
          guestClient.getStatus().peerConnected &&
          hostClient.getStatus().transport === 'webrtc-p2p' &&
          guestClient.getStatus().transport === 'webrtc-p2p' &&
          guestClient.getStatus().pingMs !== null,
        2000
      );

      hostClient.sendControl({ type: 'collapse_info', collapseReason: 'test-p2p' });
      const dummyBytes = hostClient.sendStateFrame({
        seq: 1,
        hostElapsedSec: 0.5,
        roomSeed: 0xabcdef01,
        turnPhase: 'SELECT_BLOCK',
        gameMode: 'party',
        currentPlayer: 0,
        playerCount: 2,
        blocksStacked: 0,
        shatteredCount: 0,
        currentLayers: 18,
        peakLayers: 18,
        activeBlockId: null,
        hoveredBlockId: null,
        hoveredSlotIndex: null,
        lockedMinRow: 18,
        isPointerDraggingBlock: false,
        isExtractionLiftReady: false,
        polariscopeMode: false,
        grabResistance01: 0,
        settleProgress01: 0,
        leanAngleDeg: 0,
        wobbleSpeedDegPerSec: 0,
        liveHeightCm: 27,
        peakHeightCm: 27,
        centerOfMassXZCm: [0, 0],
        carriedYawOffsetRad: 0,
        grabCurrentWorldOnBlock: [0, 0, 0],
        grabWorldAnchor: [0, 0, 0],
        carriedLandingPos: [0, 2.777, 0],
        carriedTargetQuat: [0, 0, 0, 1],
        topSupportSurfaceCenter: [0, 2.7, 0],
        topSupportUpNormal: [0, 1, 0],
        topSupportDefaultRot: [0, 0, 0, 1],
        topSupportTargetLayer: 18,
        blocks: [],
        shards: [],
        splinters: [],
      });
      expect(dummyBytes).toBe(140);

      await waitUntil(
        () =>
          receivedBinaryBytesByGuest === 140 &&
          receivedCtrlByGuest.some((m) => m.type === 'collapse_info'),
        1000
      );
    } finally {
      dupHostClient.dispose();
      guestClient.dispose();
      hostClient.dispose();
      (globalThis as unknown as { WebSocket: unknown }).WebSocket = origWs;
      (globalThis as unknown as { RTCPeerConnection?: unknown }).RTCPeerConnection = origRtc;
    }
  });

  it('T-MP8: verifies Colosseum M1 (54/54 worker sequential PRNG parity), M2 (zombie RTCDataChannel teardown on Guest reload), and M3 (Place Prism HUD hover isolation)', async () => {
    // 1. M1: Exact 54/54 sequential mulberry32 parity with MujocoPhysicsEngine.rebuildTower
    const seed = 0x9c41ef27;
    const layers = 18;
    const paletteLen = STAINED_GLASS_PALETTE.length;
    const refRng = mulberry32((seed ^ Math.imul(layers, 0x9e3779b9)) >>> 0);
    const materials = buildDeterministicTowerMaterials(seed, layers);
    expect(materials.length).toBe(54);

    let prev0 = -1;
    let prev1 = -1;
    for (let id = 0; id < 54; id++) {
      const slot = id % 3;
      if (slot === 0) {
        prev0 = -1;
        prev1 = -1;
      }
      let expectedSwatch = Math.floor(refRng() * paletteLen) % paletteLen;
      if (slot === 2 && expectedSwatch === prev0 && expectedSwatch === prev1) {
        expectedSwatch = (expectedSwatch + 1) % paletteLen;
      }
      if (slot === 0) prev0 = expectedSwatch;
      if (slot === 1) prev1 = expectedSwatch;
      const expectedMotif = Math.floor(refRng() * 4) & 3;
      const expectedRipple = +(0.25 + refRng() * 96.0).toFixed(5);

      expect(materials[id].swatchId).toBe(expectedSwatch);
      expect(materials[id].motif).toBe(expectedMotif);
      expect(materials[id].rippleSeed).toBe(expectedRipple);
    }

    // 2. M2: Zombie open RTCDataChannel on Host is torn down immediately when Guest reloads over relay
    const httpModName = 'node:http';
    const relayModPath = '../workers/vitrail-rooms/nodeRelay.mjs';
    const http = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
    const { attachVitrailRoomRelay } = (await import(
      /* @vite-ignore */ relayModPath
    )) as unknown as NodeRelayModule;

    const server = http.createServer();
    const relay = attachVitrailRoomRelay(server, '/__vitrail_mp');
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const addr = server.address();
    const port = typeof addr === 'object' && addr ? addr.port : 0;
    const relayUrl = `ws://127.0.0.1:${port}`;

    const host = new GameEngine();
    const guest1 = new GameEngine();
    const guestReloaded = new GameEngine();
    const createMemoryTabStorage = () => {
      const backing = new Map<string, string>();
      return {
        getItem: (k: string) => backing.get(k) ?? null,
        setItem: (k: string, v: string) => {
          backing.set(k, v);
        },
        removeItem: (k: string) => {
          backing.delete(k);
        },
      };
    };
    const guestTabStore = createMemoryTabStorage();
    guest1.setMpTabStorageForTesting(guestTabStore);
    guestReloaded.setMpTabStorageForTesting(guestTabStore);
    await host.initHeadless();
    await guest1.initHeadless();
    await guestReloaded.initHeadless();

    try {
      const code = host.startOnlineHost('RLD99', {
        relayUrl,
        roomSeed: seed,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });
      guest1.joinOnlineRoom(code, {
        relayUrl,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });

      const pumpPair = async (g: GameEngine, steps: number) => {
        for (let i = 0; i < steps; i++) {
          host.stepHeadless(1 / 60);
          g.stepHeadless(1 / 60);
          await new Promise((r) => setTimeout(r, 4));
        }
      };

      await waitUntil(
        () =>
          Boolean(host.getTelemetrySnapshot().mpPeerConnected) &&
          Boolean(guest1.getTelemetrySnapshot().mpPeerConnected),
        2500
      );
      await pumpPair(guest1, 8);

      // Verify 54/54 swatch parity between Host and Guest1 even if Host blocks were dirtied before frame
      host.physics.blocks[0].swatchId = (host.physics.blocks[0].swatchId + 5) % paletteLen;
      await pumpPair(guest1, 4);
      for (let i = 0; i < 54; i++) {
        expect(host.physics.blocks[i].swatchId).toBe(materials[i].swatchId);
        expect(guest1.physics.blocks[i].swatchId).toBe(materials[i].swatchId);
      }

      // Inject a zombie open RTCDataChannel onto Host's multiplayer client (simulating un-timed-out Chromium WebRTC on Guest reload)
      let zombieCtrlClosed = false;
      let zombieSentBlackholed = 0;
      const hostMpInternal = (host as unknown as { mpClient: Record<string, unknown> }).mpClient;
      hostMpInternal.pcRemoteClientId = 'guest-STALE01';
      hostMpInternal.ctrlChannel = {
        readyState: 'open',
        send: () => {
          zombieSentBlackholed++;
        },
        close: () => {
          zombieCtrlClosed = true;
        },
      };
      hostMpInternal.stateChannel = {
        readyState: 'open',
        bufferedAmount: 0,
        send: () => {
          zombieSentBlackholed++;
        },
        close: () => {},
      };

      // Simulate abrupt Guest1 tab F5 reload without clean dispose (same tab sessionStorage reclaims Guest1 clientId)
      guestReloaded.restoreOrJoinOnlineRoomFromUrl(code, {
        relayUrl,
        roomSeed: 0x11111111, // Different initial seed -> must receive room_welcome with `seed`!
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });

      await waitUntil(
        () =>
          zombieCtrlClosed &&
          Boolean(guestReloaded.getTelemetrySnapshot().mpPeerConnected) &&
          guestReloaded.mpRoomSeed === seed,
        2500
      );
      await pumpPair(guestReloaded, 8);

      expect(zombieCtrlClosed).toBe(true);
      expect(zombieSentBlackholed).toBe(0);
      expect(guestReloaded.mpRoomSeed).toBe(seed);
      for (let i = 0; i < 54; i++) {
        expect(guestReloaded.physics.blocks[i].swatchId).toBe(materials[i].swatchId);
      }

      // 3. M3: Passive cursor motion down toward `.prism-release-btn` at bottom HUD must NOT drag carriedLandingPos to corner overhang
      host.leaveOnlineRoom('solo', 2);
      for (let i = 0; i < 30; i++) host.stepHeadless(1 / 60);

      const rect = { left: 0, top: 0, width: 1280, height: 800 };
      host.camera.setPoseImmediate(0.68, 0.42, 4.9, [0, 1.44, 0], rect.width / rect.height);
      const loose = host.physics.blocks.find((b) => b.assignedLayer === 8 && b.isLooseSlot)!;
      host.physics.startGrabBlock(
        loose.id,
        [0.34, 0, 0],
        [loose.position[0] + 0.34, loose.position[1], loose.position[2]]
      );
      for (let f = 1; f <= 160 && host.physics.turnPhase === 'PULLING'; f++) {
        host.physics.updateGrabTarget([
          loose.position[0] + 0.34 + Math.min(1.15, (f / 48) * 1.15),
          loose.position[1],
          loose.position[2],
        ]);
        host.stepHeadless(1 / 60);
      }
      for (let f = 0; f < 30 && !host.physics.isExtractionLiftReady; f++) {
        host.stepHeadless(1 / 60);
      }
      expect(host.physics.turnPhase).toBe('EXTRACTED');
      expect(host.physics.isExtractionLiftReady).toBe(true);
      host.physics.endGrab(false);

      // Sweep passive cursor (`activePointerId === null`) across the lower canvas down to `.prism-release-btn`
      for (let y = 520; y <= 775; y += 25) {
        host.handlePointerMove(640, y, 99, rect);
      }
      expect(
        Math.hypot(host.physics.carriedLandingPos[0], host.physics.carriedLandingPos[2])
      ).toBeLessThan(0.01);

      // Click "Place Prism" (`handleReleaseOntoTop`) -> stacks cleanly with 0 collapse!
      host.handleReleaseOntoTop();
      for (let f = 0; f < 140 && host.physics.turnPhase === 'SETTLING'; f++) {
        host.stepHeadless(1 / 60);
      }
      expect(host.physics.turnPhase).toBe('SELECT_BLOCK');
      expect(host.physics.blocksStacked).toBe(1);
    } finally {
      guestReloaded.dispose();
      guest1.dispose();
      host.dispose();
      relay.detach();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  }, 15000);

  it('T-MP9: Public PeerJS signaling handshake complies with strict 0.peerjs.com DataConnection schema, blockThud forwards once, and Sandbox shard tap-to-flick works', async () => {
    const origWS = globalThis.WebSocket;
    const origRTC = (globalThis as unknown as { RTCPeerConnection?: unknown }).RTCPeerConnection;

    const peerSockets = new Map<string, MockPeerJsSocket>();
    const validatedUrls: string[] = [];
    let schemaRejections = 0;
    let offerForwarded = 0;
    let answerForwarded = 0;
    let candidateForwarded = 0;

    class MockPeerJsSocket {
      public static readonly OPEN = 1;
      public static readonly CLOSED = 3;
      public readyState = 0;
      public peerId = '';
      public onopen: (() => void) | null = null;
      public onmessage: ((ev: { data: string }) => void) | null = null;
      public onclose: ((ev: { code: number; reason: string }) => void) | null = null;
      public onerror: (() => void) | null = null;

      constructor(url: string) {
        validatedUrls.push(url);
        const u = new URL(url);
        this.peerId = u.searchParams.get('id') ?? '';
        const version = u.searchParams.get('version');
        queueMicrotask(() => {
          if (this.readyState === MockPeerJsSocket.CLOSED) return;
          if (!version) {
            schemaRejections++;
            this.close(1000, 'missing version');
            return;
          }
          this.readyState = MockPeerJsSocket.OPEN;
          peerSockets.set(this.peerId, this);
          this.onopen?.();
          this.onmessage?.({ data: JSON.stringify({ type: 'OPEN' }) });
        });
      }

      public send(raw: string): void {
        if (this.readyState !== MockPeerJsSocket.OPEN) return;
        const msg = JSON.parse(raw) as {
          type: string;
          dst?: string;
          payload?: Record<string, unknown>;
        };
        if (msg.type === 'HEARTBEAT') return;
        if (msg.type === 'OFFER') {
          const p = msg.payload ?? {};
          // Enforce strict 0.peerjs.com DataConnection schema (closes 1000 if missing label/reliable/serialization/browser)
          if (
            p.type !== 'data' ||
            typeof p.label !== 'string' ||
            typeof p.connectionId !== 'string' ||
            typeof p.reliable !== 'boolean' ||
            typeof p.serialization !== 'string' ||
            typeof p.browser !== 'string' ||
            !p.sdp
          ) {
            schemaRejections++;
            this.close(1000, 'invalid OFFER schema');
            return;
          }
          const target = msg.dst ? peerSockets.get(msg.dst) : undefined;
          if (target) {
            offerForwarded++;
            queueMicrotask(() =>
              target.onmessage?.({
                data: JSON.stringify({ ...msg, src: this.peerId }),
              })
            );
          }
        } else if (msg.type === 'ANSWER') {
          const p = msg.payload ?? {};
          if (
            p.type !== 'data' ||
            typeof p.connectionId !== 'string' ||
            typeof p.browser !== 'string' ||
            !p.sdp
          ) {
            schemaRejections++;
            this.close(1000, 'invalid ANSWER schema');
            return;
          }
          const target = msg.dst ? peerSockets.get(msg.dst) : undefined;
          if (target) {
            answerForwarded++;
            queueMicrotask(() =>
              target.onmessage?.({
                data: JSON.stringify({ ...msg, src: this.peerId }),
              })
            );
          }
        } else if (msg.type === 'CANDIDATE') {
          const target = msg.dst ? peerSockets.get(msg.dst) : undefined;
          if (target) {
            candidateForwarded++;
            queueMicrotask(() =>
              target.onmessage?.({
                data: JSON.stringify({ ...msg, src: this.peerId }),
              })
            );
          }
        }
      }

      public close(code: number = 1000, reason: string = ''): void {
        if (this.readyState === MockPeerJsSocket.CLOSED) return;
        this.readyState = MockPeerJsSocket.CLOSED;
        peerSockets.delete(this.peerId);
        this.onclose?.({ code, reason });
      }
    }

    const activePcs: MockRTC[] = [];
    class MockDataChannel {
      public binaryType = 'arraybuffer';
      public readyState: 'connecting' | 'open' | 'closed' = 'connecting';
      public onopen: (() => void) | null = null;
      public onmessage: ((ev: { data: unknown }) => void) | null = null;
      public onclose: (() => void) | null = null;
      public peerChannel: MockDataChannel | null = null;

      constructor(public readonly label: string, public readonly id: number) {}

      public send(data: unknown): void {
        if (this.readyState === 'open' && this.peerChannel?.readyState === 'open') {
          const peer = this.peerChannel;
          queueMicrotask(() => peer.onmessage?.({ data }));
        }
      }

      public close(): void {
        this.readyState = 'closed';
        this.onclose?.();
      }
    }

    class MockRTC {
      public signalingState: RTCSignalingState = 'stable';
      public connectionState: RTCPeerConnectionState = 'new';
      public iceConnectionState: RTCIceConnectionState = 'new';
      public localDescription: RTCSessionDescriptionInit | null = null;
      public remoteDescription: RTCSessionDescriptionInit | null = null;
      public onicecandidate: ((ev: { candidate: { toJSON: () => RTCIceCandidateInit } | null }) => void) | null = null;
      public onconnectionstatechange: (() => void) | null = null;
      public oniceconnectionstatechange: (() => void) | null = null;
      public channels = new Map<number, MockDataChannel>();

      constructor() {
        activePcs.push(this);
      }

      public createDataChannel(label: string, opts: { id?: number }): MockDataChannel {
        const ch = new MockDataChannel(label, opts.id ?? 0);
        this.channels.set(ch.id, ch);
        return ch;
      }

      public async createOffer(): Promise<RTCSessionDescriptionInit> {
        return { type: 'offer', sdp: 'v=0\r\n' };
      }

      public async createAnswer(): Promise<RTCSessionDescriptionInit> {
        return { type: 'answer', sdp: 'v=0\r\n' };
      }

      public async setLocalDescription(desc: RTCSessionDescriptionInit): Promise<void> {
        this.localDescription = desc;
        this.signalingState = desc.type === 'offer' ? 'have-local-offer' : 'stable';
        queueMicrotask(() => {
          this.onicecandidate?.({
            candidate: {
              toJSON: () => ({
                candidate: 'candidate:1 1 udp 2130706431 127.0.0.1 9999 typ host',
                sdpMid: '0',
                sdpMLineIndex: 0,
              }),
            },
          });
        });
      }

      public async setRemoteDescription(desc: RTCSessionDescriptionInit): Promise<void> {
        this.remoteDescription = desc;
        if (desc.type === 'answer' && activePcs.length >= 2) {
          this.signalingState = 'stable';
          const [pcA, pcB] = activePcs.slice(-2);
          for (const [id, chA] of pcA.channels.entries()) {
            const chB = pcB.channels.get(id);
            if (chB) {
              chA.peerChannel = chB;
              chB.peerChannel = chA;
              chA.readyState = 'open';
              chB.readyState = 'open';
              queueMicrotask(() => {
                chA.onopen?.();
                chB.onopen?.();
              });
            }
          }
        }
      }

      public async addIceCandidate(): Promise<void> {}

      public close(): void {
        this.connectionState = 'closed';
        for (const ch of this.channels.values()) ch.close();
      }
    }

    (globalThis as unknown as { WebSocket: unknown }).WebSocket = MockPeerJsSocket;
    (globalThis as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection = MockRTC;

    const host = new GameEngine();
    const guest = new GameEngine();
    await host.initHeadless();
    await guest.initHeadless();

    let guestThuds = 0;
    let guestImpacts = 0;

    try {
      const code = host.startOnlineHost('PJS99', {
        relayUrl: null,
        peerJsUrl: 'wss://0.peerjs.com/peerjs',
        enablePeerJsFallback: true,
        roomSeed: 0x424242,
      });
      guest.joinOnlineRoom(code, {
        relayUrl: null,
        peerJsUrl: 'wss://0.peerjs.com/peerjs',
        enablePeerJsFallback: true,
        roomSeed: 0x424242,
      });

      const origGuestThud = guest.audio.playBlockThud.bind(guest.audio);
      guest.audio.playBlockThud = (...args) => {
        guestThuds++;
        origGuestThud(...args);
      };
      const origGuestImpact = guest.audio.playGlassImpact.bind(guest.audio);
      guest.audio.playGlassImpact = (...args) => {
        guestImpacts++;
        origGuestImpact(...args);
      };

      await waitUntil(
        () =>
          host.getTelemetrySnapshot().mpTransport === 'webrtc-p2p' &&
          guest.getTelemetrySnapshot().mpTransport === 'webrtc-p2p',
        2000
      );

      expect(schemaRejections).toBe(0);
      expect(validatedUrls.every((u) => u.includes('version=1.5.4'))).toBe(true);
      expect(offerForwarded).toBeGreaterThanOrEqual(1);
      expect(answerForwarded).toBeGreaterThanOrEqual(1);
      expect(candidateForwarded).toBeGreaterThanOrEqual(1);

      // Pause background physics settle and flush any initial tower-settle audio before testing isolated playBlockThud
      host.benchPaused = true;
      host.stepHeadless(1 / 60);
      await sleepMs(25);
      guest.stepHeadless(1 / 60);
      guestThuds = 0;
      guestImpacts = 0;

      // Trigger playBlockThud on Host -> must emit exactly 1 blockThud to Guest and 0 duplicate glassImpacts
      host.audio.playBlockThud(0.95, [0.4, 0.08, -0.3], { damage: 0.5 });
      host.stepHeadless(1 / 60);
      await sleepMs(20);
      guest.stepHeadless(1 / 60);

      expect(guestThuds).toBe(1);
      expect(guestImpacts).toBe(0);

      // Leave online room into Sandbox mode, shatter an isolated 1L block on the oak table, and tap-flick a Voronoi shard!
      host.benchPaused = false;
      host.leaveOnlineRoom('sandbox', 2);
      host.physics.rebuildTower('sandbox', 2, 1, 777);
      for (const b of host.physics.blocks) {
        host.physics.shatterBlock(b, [0, 0, 0], 16.0, [0, 1, 0], true);
      }
      for (let f = 0; f < 120; f++) host.stepHeadless(1 / 60);
      expect(host.physics.shards.length).toBeGreaterThan(0);

      const targetShard = host.physics.shards.reduce((best, s) =>
        s.massKg > best.massKg ? s : best
      );
      const vp = { left: 0, top: 0, width: 1280, height: 800 };
      host.camera.setPoseImmediate(
        0,
        0.75,
        2.2,
        [...targetShard.position],
        vp.width / vp.height
      );
      const [sx, sy] = worldToScreenPx(host, targetShard.position, vp.width, vp.height);
      const prePos: [number, number, number] = [...targetShard.position];

      host.handlePointerDown(sx, sy, 1, 0, false, vp);
      host.handlePointerUp(sx + 1, sy, 1, vp);
      host.stepHeadless(1 / 60);

      const movedDist = Math.hypot(
        targetShard.position[0] - prePos[0],
        targetShard.position[1] - prePos[1],
        targetShard.position[2] - prePos[2]
      );
      expect(movedDist).toBeGreaterThan(0.001);
    } finally {
      guest.dispose();
      host.dispose();
      (globalThis as unknown as { WebSocket: unknown }).WebSocket = origWS;
      (globalThis as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection = origRTC;
    }
  });

  it('T-MP10: enforces 1008 Room full 3rd-peer Guest & PeerJS OFFER rejection, sessionStorage Host/Guest F5 reclaim, lobby keepalives, and cooldown/collapse audio gating', async () => {
    const httpModName = 'node:http';
    const relayModPath = '../workers/vitrail-rooms/nodeRelay.mjs';
    const http = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
    const { attachVitrailRoomRelay } = (await import(
      /* @vite-ignore */ relayModPath
    )) as unknown as NodeRelayModule;

    const server = http.createServer();
    const relay = attachVitrailRoomRelay(server, '/__vitrail_mp');
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const addr = server.address();
    const port = typeof addr === 'object' && addr ? addr.port : 0;
    const relayUrl = `ws://127.0.0.1:${port}`;

    const makeTabStorage = () => {
      const map = new Map<string, string>();
      return {
        getItem: (k: string) => map.get(k) ?? null,
        setItem: (k: string, v: string) => {
          map.set(k, v);
        },
        removeItem: (k: string) => {
          map.delete(k);
        },
      };
    };

    const hostTabStore = makeTabStorage();
    const guest1TabStore = makeTabStorage();
    const intruderTabStore = makeTabStorage();

    let intruderStatusChanges = 0;
    const host = new GameEngine();
    const guest1 = new GameEngine();
    const intruder3 = new GameEngine(undefined, () => {
      intruderStatusChanges++;
    });
    const dupHost = new GameEngine();
    const hostReloaded = new GameEngine();
    host.setMpTabStorageForTesting(hostTabStore);
    guest1.setMpTabStorageForTesting(guest1TabStore);
    intruder3.setMpTabStorageForTesting(intruderTabStore);
    hostReloaded.setMpTabStorageForTesting(hostTabStore);

    await Promise.all([
      host.initHeadless(),
      guest1.initHeadless(),
      intruder3.initHeadless(),
      dupHost.initHeadless(),
      hostReloaded.initHeadless(),
    ]);

    try {
      // 1. Host boots room with fast 60 ms lobby keepalive while `waiting_peer` (P1-4)
      const seed = 0x778899aa;
      const code = host.startOnlineHost('FULL5', {
        relayUrl,
        roomSeed: seed,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
        lobbyKeepaliveIntervalMs: 60,
      });

      await waitUntil(
        () =>
          host.getTelemetrySnapshot().mpConnectionState === 'waiting_peer' &&
          !host.getTelemetrySnapshot().mpPeerConnected,
        1500
      );
      await sleepMs(160);
      // Relay silently absorbed >= 2 `{ __relay: 'keepalive' }` frames while waiting_peer
      expect(host.getTelemetrySnapshot().mpConnectionState).toBe('waiting_peer');

      // 2. Guest 1 joins -> 2/2 room full
      guest1.restoreOrJoinOnlineRoomFromUrl(code, {
        relayUrl,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });
      await waitUntil(
        () =>
          Boolean(host.getTelemetrySnapshot().mpPeerConnected) &&
          Boolean(guest1.getTelemetrySnapshot().mpPeerConnected),
        2000
      );
      for (let i = 0; i < 6; i++) {
        host.stepHeadless(1 / 60);
        guest1.stepHeadless(1 / 60);
        await sleepMs(5);
      }
      expect(guest1.mpRoomSeed).toBe(seed);

      // 3. 3rd-peer Guest intrusion (empty tab storage -> new clientId) MUST be rejected with 1008 Room full & 0 reconnect storms (P0-1)
      intruder3.restoreOrJoinOnlineRoomFromUrl(code, {
        relayUrl,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });
      await waitUntil(() => intruder3.getTelemetrySnapshot().mpConnectionState === 'error', 2000);
      expect(intruder3.getTelemetrySnapshot().mpError).toContain('Room full (2/2 players)');
      expect(intruder3.getTelemetrySnapshot().mpPeerConnected).toBe(false);
      expect(guest1.getTelemetrySnapshot().mpPeerConnected).toBe(true);
      expect(host.getTelemetrySnapshot().mpPeerConnected).toBe(true);

      const changesAtTerminal = intruderStatusChanges;
      await sleepMs(750);
      expect(intruderStatusChanges).toBe(changesAtTerminal);
      expect(guest1.getTelemetrySnapshot().mpPeerConnected).toBe(true);

      // 4. Duplicate Host with different clientId is also rejected with 1008 while Host 1 stays connected
      dupHost.startOnlineHost(code, {
        relayUrl,
        roomSeed: 0x12345678,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });
      await waitUntil(() => dupHost.getTelemetrySnapshot().mpConnectionState === 'error', 2000);
      expect(dupHost.getTelemetrySnapshot().mpError).toContain('already has an active Host');
      expect(host.getTelemetrySnapshot().mpPeerConnected).toBe(true);
      expect(guest1.getTelemetrySnapshot().mpPeerConnected).toBe(true);

      // 5. P1-3: Cooldown-dropped `playGlassImpact` (< 12 ms) & `playWobbleRock` (< 65 ms per blockId) never transmit,
      //    and `turnPhase === 'COLLAPSED'` caps collapse transient network audio at <= 25 Hz (>= 40 ms spacing)
      let simClockSec = 10.0;
      const makeMockNode = () => ({
        gain: {
          value: 1,
          setValueAtTime() {},
          linearRampToValueAtTime() {},
          exponentialRampToValueAtTime() {},
          setTargetAtTime() {},
          cancelScheduledValues() {},
        },
        frequency: { value: 440, setValueAtTime() {}, exponentialRampToValueAtTime() {} },
        Q: { value: 1 },
        pan: { value: 0 },
        type: 'sine',
        buffer: null,
        loop: false,
        onended: null,
        connect() {
          return this;
        },
        disconnect() {},
        start() {},
        stop() {},
      });
      const mockAudioCtx = {
        sampleRate: 48000,
        state: 'running',
        get currentTime() {
          return simClockSec;
        },
        destination: makeMockNode(),
        createGain: makeMockNode,
        createOscillator: makeMockNode,
        createBiquadFilter: makeMockNode,
        createStereoPanner: makeMockNode,
        createBufferSource: makeMockNode,
        createBuffer(_c: number, len: number) {
          const buf = new Float32Array(Math.min(len, 2048));
          return { getChannelData: () => buf };
        },
        resume: () => Promise.resolve(),
        close: () => Promise.resolve(),
      };
      (host.audio as unknown as { ctx: unknown; masterGain: unknown; muted: boolean }).ctx =
        mockAudioCtx;
      (host.audio as unknown as { masterGain: unknown }).masterGain = makeMockNode();
      host.audio.setMuted(false);

      host.benchPaused = true;
      host.stepHeadless(1 / 60);
      await sleepMs(25);
      guest1.stepHeadless(1 / 60);

      let guestImpactsRx = 0;
      let guestWobblesRx = 0;
      let guestLastWobbleBlockId: number | undefined;
      const origGuestImpact = guest1.audio.playGlassImpact.bind(guest1.audio);
      guest1.audio.playGlassImpact = (...args) => {
        guestImpactsRx++;
        return origGuestImpact(...args);
      };
      const origGuestWobble = guest1.audio.playWobbleRock.bind(guest1.audio);
      guest1.audio.playWobbleRock = (tilt, speed, load, pos, opts) => {
        guestWobblesRx++;
        guestLastWobbleBlockId = opts?.blockId;
        return origGuestWobble(tilt, speed, load, pos, opts);
      };

      // Fire 8 impacts + 8 wobbles at identical `simClockSec = 10.5` -> only 1 of each schedules & transmits!
      simClockSec = 10.5;
      for (let k = 0; k < 8; k++) {
        host.audio.playGlassImpact(0.28, 1.0, [0, 0.4, 0], undefined, undefined, {
          impactSpeedMps: 0.25,
          surface: 'glass',
        });
        host.audio.playWobbleRock(0.06, 0.9, 0.2, [0, 0.4, 0], {
          blockId: 17,
          surface: 'glass',
        });
      }
      (host as unknown as { pumpMultiplayerStep: (ms: number) => void }).pumpMultiplayerStep(
        performance.now() + 100
      );
      await sleepMs(30);
      guest1.stepHeadless(1 / 60);

      expect(guestImpactsRx).toBe(1);
      expect(guestWobblesRx).toBe(1);
      expect(guestLastWobbleBlockId).toBe(17);

      // Enter COLLAPSED state and fire 12 scheduled wobbles on distinct blockIds within 120 ms -> <= 25 Hz cap yields <= 4
      guestWobblesRx = 0;
      host.physics.turnPhase = 'COLLAPSED';
      for (let k = 0; k < 12; k++) {
        simClockSec += 0.02;
        host.audio.playWobbleRock(0.07, 1.1, 0.15, [0, 0.2, 0], {
          blockId: k + 1,
          surface: 'glass',
        });
        await sleepMs(10);
      }
      (host as unknown as { pumpMultiplayerStep: (ms: number) => void }).pumpMultiplayerStep(
        performance.now() + 200
      );
      await sleepMs(35);
      guest1.stepHeadless(1 / 60);
      expect(guestWobblesRx).toBeGreaterThanOrEqual(1);
      expect(guestWobblesRx).toBeLessThanOrEqual(5);

      // 6. P0-2: Host `F5` reload (`dispose()` preserves `vitrail_mp_tab_session_v1`, `restoreOrJoinOnlineRoomFromUrl` reclaims `host`!)
      host.benchPaused = false;
      host.dispose();
      expect(hostTabStore.getItem('vitrail_mp_tab_session_v1')).not.toBeNull();

      const reclaimedRole = hostReloaded.restoreOrJoinOnlineRoomFromUrl(code, {
        relayUrl,
        enablePeerJsFallback: false,
        enableWebRtcUpgrade: false,
      });
      expect(reclaimedRole).toBe('host');
      expect(hostReloaded.mpRole).toBe('host');
      expect(hostReloaded.mpRoomSeed).toBe(seed);

      await waitUntil(
        () =>
          Boolean(hostReloaded.getTelemetrySnapshot().mpPeerConnected) &&
          Boolean(guest1.getTelemetrySnapshot().mpPeerConnected) &&
          guest1.physics.turnPhase === 'SELECT_BLOCK',
        2500
      );
      for (let i = 0; i < 6; i++) {
        hostReloaded.stepHeadless(1 / 60);
        guest1.stepHeadless(1 / 60);
        await sleepMs(5);
      }
      expect(guest1.mpRoomSeed).toBe(seed);

      // Leaving online room clears per-tab sessionStorage so future `?room=` opens join normally
      hostReloaded.leaveOnlineRoom('solo', 2);
      expect(hostTabStore.getItem('vitrail_mp_tab_session_v1')).toBeNull();
    } finally {
      host.dispose();
      guest1.dispose();
      intruder3.dispose();
      dupHost.dispose();
      hostReloaded.dispose();
      relay.detach();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }

    // 7. P0-1 PeerJS `OFFER` 3rd-peer intrusion guard (`LEAVE` code 1008 without disturbing active WebRTC Guest 1)
    const origWS = globalThis.WebSocket;
    const origRTC = globalThis.RTCPeerConnection;
    const peerSockets = new Map<string, MockPeerJsBusSocket>();

    class MockPeerJsBusSocket {
      public static readonly CONNECTING = 0;
      public static readonly OPEN = 1;
      public static readonly CLOSING = 2;
      public static readonly CLOSED = 3;
      public readyState = MockPeerJsBusSocket.CONNECTING;
      public binaryType = 'arraybuffer';
      public onopen: (() => void) | null = null;
      public onmessage: ((ev: { data: unknown }) => void) | null = null;
      public onerror: (() => void) | null = null;
      public onclose: ((ev: { code: number; reason: string }) => void) | null = null;
      public readonly peerId: string;

      constructor(url: string) {
        const parsed = new URL(url);
        this.peerId = parsed.searchParams.get('id') ?? '';
        queueMicrotask(() => {
          if (peerSockets.has(this.peerId)) {
            this.readyState = MockPeerJsBusSocket.OPEN;
            this.onopen?.();
            this.onmessage?.({ data: JSON.stringify({ type: 'ID-TAKEN' }) });
            return;
          }
          peerSockets.set(this.peerId, this);
          this.readyState = MockPeerJsBusSocket.OPEN;
          this.onopen?.();
          this.onmessage?.({ data: JSON.stringify({ type: 'OPEN' }) });
        });
      }

      public send(raw: string): void {
        const msg = JSON.parse(raw) as { type: string; dst?: string };
        if (msg.type === 'HEARTBEAT') {
          queueMicrotask(() => this.onmessage?.({ data: JSON.stringify({ type: 'HEARTBEAT' }) }));
          return;
        }
        const target = msg.dst ? peerSockets.get(msg.dst) : undefined;
        if (target) {
          queueMicrotask(() =>
            target.onmessage?.({ data: JSON.stringify({ ...msg, src: this.peerId }) })
          );
        }
      }

      public close(code: number = 1000, reason: string = ''): void {
        if (this.readyState === MockPeerJsBusSocket.CLOSED) return;
        this.readyState = MockPeerJsBusSocket.CLOSED;
        if (peerSockets.get(this.peerId) === this) {
          peerSockets.delete(this.peerId);
        }
        this.onclose?.({ code, reason });
      }
    }

    const activePcs: MockPeerRTC[] = [];
    class MockPeerDC {
      public binaryType = 'arraybuffer';
      public readyState: 'connecting' | 'open' | 'closed' = 'connecting';
      public onopen: (() => void) | null = null;
      public onmessage: ((ev: { data: unknown }) => void) | null = null;
      public onclose: (() => void) | null = null;
      public peerChannel: MockPeerDC | null = null;
      constructor(public readonly label: string, public readonly id: number) {}
      public send(data: unknown): void {
        if (this.readyState === 'open' && this.peerChannel?.readyState === 'open') {
          const peer = this.peerChannel;
          queueMicrotask(() => peer.onmessage?.({ data }));
        }
      }
      public close(): void {
        this.readyState = 'closed';
        this.onclose?.();
      }
    }

    class MockPeerRTC {
      public signalingState: RTCSignalingState = 'stable';
      public connectionState: RTCPeerConnectionState = 'new';
      public iceConnectionState: RTCIceConnectionState = 'new';
      public localDescription: RTCSessionDescriptionInit | null = null;
      public remoteDescription: RTCSessionDescriptionInit | null = null;
      public onicecandidate: ((ev: { candidate: null }) => void) | null = null;
      public onconnectionstatechange: (() => void) | null = null;
      public oniceconnectionstatechange: (() => void) | null = null;
      public channels = new Map<number, MockPeerDC>();
      constructor() {
        activePcs.push(this);
      }
      public createDataChannel(label: string, opts: { id?: number }): MockPeerDC {
        const ch = new MockPeerDC(label, opts.id ?? 0);
        this.channels.set(ch.id, ch);
        return ch;
      }
      public async createOffer(): Promise<RTCSessionDescriptionInit> {
        return { type: 'offer', sdp: 'v=0\r\n' };
      }
      public async createAnswer(): Promise<RTCSessionDescriptionInit> {
        return { type: 'answer', sdp: 'v=0\r\n' };
      }
      public async setLocalDescription(desc: RTCSessionDescriptionInit): Promise<void> {
        this.localDescription = desc;
        this.signalingState = desc.type === 'offer' ? 'have-local-offer' : 'stable';
      }
      public async setRemoteDescription(desc: RTCSessionDescriptionInit): Promise<void> {
        this.remoteDescription = desc;
        if (desc.type === 'answer' && activePcs.length >= 2) {
          this.signalingState = 'stable';
          const [pcA, pcB] = activePcs.slice(-2);
          for (const [id, chA] of pcA.channels.entries()) {
            const chB = pcB.channels.get(id);
            if (chB) {
              chA.peerChannel = chB;
              chB.peerChannel = chA;
              chA.readyState = 'open';
              chB.readyState = 'open';
              queueMicrotask(() => {
                chA.onopen?.();
                chB.onopen?.();
              });
            }
          }
        }
      }
      public async addIceCandidate(): Promise<void> {}
      public close(): void {
        this.connectionState = 'closed';
        for (const ch of this.channels.values()) ch.close();
      }
    }

    (globalThis as unknown as { WebSocket: unknown }).WebSocket = MockPeerJsBusSocket;
    (globalThis as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection = MockPeerRTC;

    const pjsHost = new GameEngine();
    const pjsGuest1 = new GameEngine();
    const pjsIntruder3 = new GameEngine();
    await Promise.all([pjsHost.initHeadless(), pjsGuest1.initHeadless(), pjsIntruder3.initHeadless()]);
    try {
      const pjsCode = pjsHost.startOnlineHost('PJS3P', {
        relayUrl: null,
        peerJsUrl: 'wss://0.peerjs.com/peerjs',
        enablePeerJsFallback: true,
        roomSeed: 0x334455,
      });
      pjsGuest1.joinOnlineRoom(pjsCode, {
        relayUrl: null,
        peerJsUrl: 'wss://0.peerjs.com/peerjs',
        enablePeerJsFallback: true,
        roomSeed: 0x334455,
      });
      await waitUntil(
        () =>
          pjsHost.getTelemetrySnapshot().mpTransport === 'webrtc-p2p' &&
          pjsGuest1.getTelemetrySnapshot().mpTransport === 'webrtc-p2p',
        2000
      );
      pjsHost.stepHeadless(1 / 60);
      await sleepMs(15);
      pjsGuest1.stepHeadless(1 / 60);

      pjsIntruder3.joinOnlineRoom(pjsCode, {
        relayUrl: null,
        peerJsUrl: 'wss://0.peerjs.com/peerjs',
        enablePeerJsFallback: true,
      });
      await waitUntil(
        () => pjsIntruder3.getTelemetrySnapshot().mpConnectionState === 'error',
        2000
      );
      expect(pjsIntruder3.getTelemetrySnapshot().mpError).toContain('Room full (2/2 players)');
      expect(pjsIntruder3.getTelemetrySnapshot().mpPeerConnected).toBe(false);
      expect(pjsGuest1.getTelemetrySnapshot().mpPeerConnected).toBe(true);
      expect(pjsGuest1.getTelemetrySnapshot().mpTransport).toBe('webrtc-p2p');
    } finally {
      pjsIntruder3.dispose();
      pjsGuest1.dispose();
      pjsHost.dispose();
      (globalThis as unknown as { WebSocket: unknown }).WebSocket = origWS;
      (globalThis as unknown as { RTCPeerConnection: unknown }).RTCPeerConnection = origRTC;
    }
  }, 15000);

  it('T-MP11: Non-inhibiting relay abuse guard rejects foreign Origins (403) & non-0x564d / >16 KB binary / >512 KB or non-JSON frames while passing 0x564d & ~400 KB 512-shard room_welcome', async () => {
    const httpModName = 'node:http';
    const netModName = 'node:net';
    const relayModPath = '../workers/vitrail-rooms/nodeRelay.mjs';
    const cfModPath = '../workers/vitrail-rooms/src/index';

    const http = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
    const net = (await import(/* @vite-ignore */ netModName)) as unknown as {
      createConnection(opts: { host: string; port: number }, cb: () => void): {
        write(data: string): void;
        on(ev: string, cb: (chunk?: { toString(enc?: string): string }) => void): void;
        destroy(): void;
      };
    };
    const { attachVitrailRoomRelay, isAllowedRelayOrigin: nodeIsAllowed } = (await import(
      /* @vite-ignore */ relayModPath
    )) as unknown as NodeRelayModule;

    interface MockCfSocket {
      sent: Array<string | ArrayBuffer>;
      closeCode: number | null;
      readyState: number;
      attachment: unknown;
      send(msg: string | ArrayBuffer): void;
      close(code?: number, reason?: string): void;
      serializeAttachment(v: unknown): void;
      deserializeAttachment(): unknown;
    }
    interface CfWorkerModule {
      isAllowedRelayOrigin(origin: string | null | undefined): boolean;
      default: {
        fetch(
          req: Request,
          env: {
            VITRAIL_ROOMS: {
              idFromName(name: string): string;
              get(id: string): { fetch(req: Request): Promise<Response> };
            };
          }
        ): Promise<Response>;
      };
      VitrailRoomDurableObject: new (state: {
        acceptWebSocket(ws: MockCfSocket, tags: string[]): void;
        getWebSockets(tag?: string): MockCfSocket[];
      }) => {
        webSocketMessage(ws: MockCfSocket, message: string | ArrayBuffer): void;
      };
    }

    const cfMod = (await import(/* @vite-ignore */ cfModPath)) as unknown as CfWorkerModule;

    // 1. Permissive Origin allowlist parity across Cloudflare Worker and Node dev relay
    const allowedOrigins: Array<string | null | undefined> = [
      null,
      undefined,
      '',
      'null',
      'https://vitrail.andrewvoirol.com',
      'https://preview.andrewvoirol.com',
      'https://tumbling-blocks-git-main-andrew.vercel.app',
      'http://localhost:5173',
      'http://127.0.0.1:5173',
      'http://[::1]:5173',
      'http://macbook-pro.local:5173',
      'http://10.0.1.15:5173',
      'http://192.168.1.42:5173',
      'http://172.16.0.4:5173',
      'http://172.31.255.200:5173',
    ];
    for (const orig of allowedOrigins) {
      expect(nodeIsAllowed(orig)).toBe(true);
      expect(cfMod.isAllowedRelayOrigin(orig)).toBe(true);
    }

    const blockedOrigins = [
      'https://evil.example.com',
      'https://10.evil.com',
      'https://192.168.evil.com',
      'http://172.15.0.1:5173',
      'http://172.32.0.1:5173',
      'https://vitrail.andrewvoirol.com.attacker.io',
      ':::bad-origin:::',
    ];
    for (const orig of blockedOrigins) {
      expect(nodeIsAllowed(orig)).toBe(false);
      expect(cfMod.isAllowedRelayOrigin(orig)).toBe(false);
    }

    // Verify Cloudflare Worker `fetch` rejects foreign Origin with 403 before touching DO namespace
    let doCalls = 0;
    const mockEnv = {
      VITRAIL_ROOMS: {
        idFromName: (n: string) => n,
        get: () => {
          doCalls++;
          return {
            fetch: async () => new Response('ok', { status: 200 }),
          };
        },
      },
    };
    const cfEvilRes = await cfMod.default.fetch(
      new Request('https://vitrail-rooms.andrew9687.workers.dev/__vitrail_mp?room=TEST9', {
        headers: {
          Origin: 'https://evil.example.com',
          Upgrade: 'websocket',
          Connection: 'Upgrade',
        },
      }),
      mockEnv
    );
    expect(cfEvilRes.status).toBe(403);
    expect(doCalls).toBe(0);

    const cfGoodRes = await cfMod.default.fetch(
      new Request('https://vitrail-rooms.andrew9687.workers.dev/__vitrail_mp?room=TEST9', {
        headers: {
          Origin: 'https://vitrail.andrewvoirol.com',
          Upgrade: 'websocket',
          Connection: 'Upgrade',
        },
      }),
      mockEnv
    );
    expect(cfGoodRes.status).toBe(200);
    expect(doCalls).toBe(1);

    // 2. Verify Cloudflare Durable Object `webSocketMessage` payload guard directly
    const makeCfRoomHarness = () => {
      const tagged = new Map<string, MockCfSocket[]>();
      const makeSock = (role: 'host' | 'guest'): MockCfSocket => {
        const s: MockCfSocket = {
          sent: [],
          closeCode: null,
          readyState: 1,
          attachment: { role, clientId: `${role}-1` },
          send(m) {
            s.sent.push(m);
          },
          close(code = 1000) {
            s.closeCode = code;
            s.readyState = 3;
          },
          serializeAttachment(v) {
            s.attachment = v;
          },
          deserializeAttachment() {
            return s.attachment;
          },
        };
        return s;
      };
      const doObj = new cfMod.VitrailRoomDurableObject({
        acceptWebSocket(ws, tags) {
          for (const t of tags) {
            const list = tagged.get(t) ?? [];
            list.push(ws);
            tagged.set(t, list);
          }
        },
        getWebSockets(tag) {
          if (!tag) return [...(tagged.get('host') ?? []), ...(tagged.get('guest') ?? [])];
          return tagged.get(tag) ?? [];
        },
      });
      const hostWs = makeSock('host');
      const guestWs = makeSock('guest');
      tagged.set('host', [hostWs]);
      tagged.set('guest', [guestWs]);
      return { doObj, hostWs, guestWs };
    };

    const validFrameBuf = encodeMultiplayerStateFrame({
      seq: 7,
      hostElapsedSec: 1.25,
      roomSeed: 0xabcde,
      turnPhase: 'PULLING',
      gameMode: 'party',
      currentPlayer: 0,
      playerCount: 2,
      blocksStacked: 0,
      shatteredCount: 0,
      currentLayers: 18,
      peakLayers: 18,
      activeBlockId: null,
      hoveredBlockId: null,
      hoveredSlotIndex: -1,
      lockedMinRow: null,
      isPointerDraggingBlock: false,
      isExtractionLiftReady: false,
      polariscopeMode: false,
      grabResistance01: 0,
      settleProgress01: 1,
      leanAngleDeg: 0,
      wobbleSpeedDegPerSec: 0,
      liveHeightCm: 27.0,
      peakHeightCm: 27.0,
      centerOfMassXZCm: [0, 0],
      carriedYawOffsetRad: 0,
      grabCurrentWorldOnBlock: [0, 0, 0],
      grabWorldAnchor: [0, 0, 0],
      carriedLandingPos: [0, 2.775, 0],
      carriedTargetQuat: [0, 0, 0, 1],
      topSupportSurfaceCenter: [0, 2.7, 0],
      topSupportUpNormal: [0, 1, 0],
      topSupportDefaultRot: [0, 0, 0, 1],
      topSupportTargetLayer: 18,
      blocks: [],
      shards: [],
      splinters: [],
    });
    const bigWelcomeJson = JSON.stringify({
      type: 'room_welcome',
      roomCode: 'GUARD',
      pad: 'x'.repeat(400_000),
    });

    {
      const { doObj, hostWs, guestWs } = makeCfRoomHarness();
      doObj.webSocketMessage(hostWs, validFrameBuf);
      doObj.webSocketMessage(hostWs, bigWelcomeJson);
      expect(hostWs.closeCode).toBeNull();
      expect(guestWs.sent.length).toBe(2);
      expect((guestWs.sent[0] as ArrayBuffer).byteLength).toBe(validFrameBuf.byteLength);
      expect(guestWs.sent[1]).toBe(bigWelcomeJson);
    }
    {
      const { doObj, hostWs, guestWs } = makeCfRoomHarness();
      const badMagic = new Uint8Array([0xde, 0xad, 0xbe, 0xef]).buffer;
      doObj.webSocketMessage(hostWs, badMagic);
      expect(hostWs.closeCode).toBe(1003);
      expect(guestWs.sent.filter((m) => typeof m !== 'string')).toHaveLength(0);
    }
    {
      const { doObj, hostWs, guestWs } = makeCfRoomHarness();
      const hugeBin = new Uint8Array(20_000);
      hugeBin[0] = 0x4d;
      hugeBin[1] = 0x56;
      doObj.webSocketMessage(hostWs, hugeBin.buffer);
      expect(hostWs.closeCode).toBe(1009);
      expect(guestWs.sent.filter((m) => typeof m !== 'string')).toHaveLength(0);
    }
    {
      const { doObj, hostWs } = makeCfRoomHarness();
      doObj.webSocketMessage(hostWs, 'not-json-at-all');
      expect(hostWs.closeCode).toBe(1003);
    }
    {
      const { doObj, hostWs } = makeCfRoomHarness();
      doObj.webSocketMessage(hostWs, `{"oversized":"${'y'.repeat(530_000)}"}`);
      expect(hostWs.closeCode).toBe(1009);
    }

    // 3. End-to-end socket verification against `nodeRelay.mjs` over real local TCP/WebSocket
    const server = http.createServer();
    const relay = attachVitrailRoomRelay(server, '/__vitrail_mp');
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const addr = server.address();
    const port = typeof addr === 'object' && addr ? addr.port : 0;

    try {
      const rawUpgradeStatus = await new Promise<number>((resolve) => {
        let resp = '';
        const sock = net.createConnection({ host: '127.0.0.1', port }, () => {
          sock.write(
            'GET /__vitrail_mp?room=GUARD&role=guest HTTP/1.1\r\n' +
              `Host: 127.0.0.1:${port}\r\n` +
              'Origin: https://evil.example.com\r\n' +
              'Upgrade: websocket\r\n' +
              'Connection: Upgrade\r\n\r\n'
          );
        });
        sock.on('data', (chunk) => {
          if (chunk) resp += chunk.toString('utf8');
        });
        sock.on('close', () => {
          const m = /^HTTP\/1\.1 (\d{3})/.exec(resp);
          resolve(m ? Number(m[1]) : 0);
        });
      });
      expect(rawUpgradeStatus).toBe(403);

      const openRelayWs = (role: 'host' | 'guest', clientId: string) =>
        new Promise<{
          ws: WebSocket;
          rxBin: number[];
          rxText: string[];
          closed: { code: number } | null;
          getClosed: () => { code: number } | null;
        }>((resolve, reject) => {
          const ws = new WebSocket(
            `ws://127.0.0.1:${port}/__vitrail_mp?room=GUARD&role=${role}&clientId=${clientId}`
          );
          ws.binaryType = 'arraybuffer';
          const rxBin: number[] = [];
          const rxText: string[] = [];
          let closed: { code: number } | null = null;
          ws.onmessage = (ev) => {
            if (typeof ev.data === 'string') {
              if (!ev.data.includes('"__relay"')) rxText.push(ev.data);
            } else if (ev.data instanceof ArrayBuffer) {
              rxBin.push(ev.data.byteLength);
            }
          };
          ws.onclose = (ev) => {
            closed = { code: ev.code };
          };
          ws.onopen = () =>
            resolve({ ws, rxBin, rxText, closed, getClosed: () => closed });
          ws.onerror = () => reject(new Error('WS error'));
        });

      const h1 = await openRelayWs('host', 'h-ok');
      const g1 = await openRelayWs('guest', 'g-ok');

      // Valid 0x564d binary + ~400 KB room_welcome JSON pass cleanly
      h1.ws.send(validFrameBuf);
      h1.ws.send(bigWelcomeJson);
      await waitUntil(() => g1.rxBin.length === 1 && g1.rxText.length === 1, 1500);
      expect(g1.rxBin[0]).toBe(validFrameBuf.byteLength);
      expect(g1.rxText[0].length).toBe(bigWelcomeJson.length);

      // Non-0x564d binary closes sender with 1003 without forwarding to Guest
      h1.ws.send(new Uint8Array([0x01, 0x02, 0x03, 0x04]));
      await waitUntil(() => h1.getClosed() !== null, 1500);
      expect(h1.getClosed()?.code).toBe(1003);
      expect(g1.rxBin.length).toBe(1);

      // Reconnect Host and send > 16 KB binary -> closes with 1009 without forwarding to Guest
      const h2 = await openRelayWs('host', 'h-oversize');
      const overBin = new Uint8Array(20_000);
      overBin[0] = 0x4d;
      overBin[1] = 0x56;
      h2.ws.send(overBin);
      await waitUntil(() => h2.getClosed() !== null, 1500);
      expect(h2.getClosed()?.code).toBe(1009);
      expect(g1.rxBin.length).toBe(1);

      g1.ws.close();
    } finally {
      relay.detach();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});

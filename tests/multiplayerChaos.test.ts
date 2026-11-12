import { describe, expect, it } from 'vitest';
import { GameEngine } from '../src/engine/GameEngine';
import { quatRotateVec3 } from '../src/math/vecMat';
import {
  buildDeterministicTowerMaterials,
  canonicalizeQuatHemisphere,
  DEFAULT_PUBLIC_ICE_SERVERS,
  deriveRelayIceEndpointUrl,
  fetchDynamicTurnIceServers,
  isForcedRelayEnabled,
  parseConfiguredIceServers,
  resolveRelayEndpoints,
  VitrailMultiplayerClient,
} from '../src/multiplayer';

interface MinimalHttpServer {
  listen: (port: number, host: string, cb: () => void) => void;
  address: () => { port: number } | string | null;
  close: (cb?: () => void) => void;
}

interface MinimalHttpModule {
  createServer: () => MinimalHttpServer;
}

interface NodeRelayModule {
  attachVitrailRoomRelay: (
    server: MinimalHttpServer,
    pathPrefix?: string
  ) => { detach: () => void };
}

function sleepWallMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitUntilWall(
  pred: () => boolean,
  timeoutMs: number = 4500,
  stepFn?: () => void
): Promise<boolean> {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    stepFn?.();
    if (pred()) return true;
    await sleepWallMs(15);
  }
  stepFn?.();
  return pred();
}

describe('Vitrail Multiplayer Resilience, <=4s ICE Fallback & 100-Match Chaos Suite', () => {
  it('T-CHAOS1: preserves localhost /__vitrail_mp priority over VITE_MP_RELAY_URL with fallback, parses VITE_MP_TURN_URLS + /ice, and honors forced-relay & query param preservation', async () => {
    const prodRelay = 'wss://vitrail-rooms.example.workers.dev/__vitrail_mp';

    // 1. Localhost Vite dev prioritizes local `/__vitrail_mp` while keeping `VITE_MP_RELAY_URL` as automatic fallback
    const localDev = resolveRelayEndpoints(
      {},
      { VITE_MP_RELAY_URL: prodRelay, DEV: true },
      {
        protocol: 'http:',
        host: 'localhost:5173',
        hostname: 'localhost',
        port: '5173',
        search: '',
      }
    );
    expect(localDev.primaryRelayUrl).toBe('ws://localhost:5173/__vitrail_mp');
    expect(localDev.fallbackRelayUrl).toBe(prodRelay);

    // 2. Production HTTPS host uses `VITE_MP_RELAY_URL` directly
    const prodEndpoints = resolveRelayEndpoints(
      {},
      { VITE_MP_RELAY_URL: prodRelay, DEV: false },
      {
        protocol: 'https:',
        host: 'vitrail.andrewvoirol.com',
        hostname: 'vitrail.andrewvoirol.com',
        port: '',
        search: '',
      }
    );
    expect(prodEndpoints.primaryRelayUrl).toBe(prodRelay);

    // 3. `?relay=` query param and `options.relayUrl` override cleanly
    const queryOverride = resolveRelayEndpoints(
      {},
      { VITE_MP_RELAY_URL: prodRelay, DEV: true },
      {
        protocol: 'http:',
        host: '127.0.0.1:5173',
        hostname: '127.0.0.1',
        port: '5173',
        search: '?relay=wss%3A%2F%2Fcustom-relay.example.com%2F__vitrail_mp',
      }
    );
    expect(queryOverride.primaryRelayUrl).toBe('wss://custom-relay.example.com/__vitrail_mp');

    // 4. Static localhost without `/__vitrail_mp` falls back to `VITE_MP_RELAY_URL` automatically when initial local WS fails to open
    const httpModName = 'node:http';
    const relayModPath = '../workers/vitrail-rooms/nodeRelay.mjs';
    const http = (await import(/* @vite-ignore */ httpModName)) as unknown as MinimalHttpModule;
    const { attachVitrailRoomRelay } = (await import(
      /* @vite-ignore */ relayModPath
    )) as unknown as NodeRelayModule;

    const fallbackServer = http.createServer();
    const fallbackRelay = attachVitrailRoomRelay(fallbackServer, '/__vitrail_mp');
    await new Promise<void>((resolve) => fallbackServer.listen(0, '127.0.0.1', resolve));
    const fbAddr = fallbackServer.address();
    const fbPort = typeof fbAddr === 'object' && fbAddr ? fbAddr.port : 0;
    const liveFallbackUrl = `ws://127.0.0.1:${fbPort}/__vitrail_mp`;

    // Pick a closed ephemeral port on 127.0.0.1 to simulate static localhost server without `/__vitrail_mp`
    const deadServer = http.createServer();
    await new Promise<void>((resolve) => deadServer.listen(0, '127.0.0.1', resolve));
    const deadAddr = deadServer.address();
    const deadPort = typeof deadAddr === 'object' && deadAddr ? deadAddr.port : 0;
    await new Promise<void>((resolve) => deadServer.close(() => resolve()));
    const deadLocalUrl = `ws://127.0.0.1:${deadPort}/__vitrail_mp`;

    const cHost = new VitrailMultiplayerClient({
      role: 'host',
      roomCode: 'DEVFB',
      relayUrl: deadLocalUrl,
      fallbackRelayUrl: liveFallbackUrl,
      enablePeerJsFallback: false,
      enableWebRtcUpgrade: false,
    });
    const cGuest = new VitrailMultiplayerClient({
      role: 'guest',
      roomCode: 'DEVFB',
      relayUrl: deadLocalUrl,
      fallbackRelayUrl: liveFallbackUrl,
      enablePeerJsFallback: false,
      enableWebRtcUpgrade: false,
    });

    try {
      cHost.connect();
      cGuest.connect();
      const connectedViaFallback = await waitUntilWall(
        () => cHost.getStatus().peerConnected && cGuest.getStatus().peerConnected,
        2500
      );
      expect(connectedViaFallback).toBe(true);
      expect(cHost.getStatus().transport).toBe('ws-relay');
      expect(cGuest.getStatus().transport).toBe('ws-relay');
    } finally {
      cHost.dispose();
      cGuest.dispose();
      fallbackRelay.detach();
      await new Promise<void>((resolve) => fallbackServer.close(() => resolve()));
    }

    // 5. `VITE_MP_TURN_URLS` + credentials + dynamic `/__vitrail_mp/ice` parsing
    const parsedIce = parseConfiguredIceServers({
      VITE_MP_TURN_URLS:
        'turn:turn.cloudflare.com:3478?transport=udp, turns:turn.cloudflare.com:5349?transport=tcp; stun:stun2.l.google.com:19302',
      VITE_MP_TURN_USERNAME: 'cf-ephemeral-user',
      VITE_MP_TURN_CREDENTIAL: 'cf-ephemeral-secret',
    });
    expect(parsedIce.length).toBe(DEFAULT_PUBLIC_ICE_SERVERS.length + 2);
    const turnEntry = parsedIce.find(
      (s) => Array.isArray(s.urls) && s.urls.some((u) => u.startsWith('turns:'))
    );
    expect(turnEntry?.username).toBe('cf-ephemeral-user');
    expect(turnEntry?.credential).toBe('cf-ephemeral-secret');

    // Without credentials, `turn:` URLs must NOT be injected (avoids browser `RTCPeerConnection` InvalidAccessError)
    const unauthTurn = parseConfiguredIceServers({
      VITE_MP_TURN_URLS: 'turn:turn.cloudflare.com:3478',
    });
    expect(unauthTurn.length).toBe(DEFAULT_PUBLIC_ICE_SERVERS.length);

    expect(deriveRelayIceEndpointUrl('wss://vitrail-rooms.workers.dev/__vitrail_mp')).toBe(
      'https://vitrail-rooms.workers.dev/__vitrail_mp/ice'
    );
    const dynamicTurn = await fetchDynamicTurnIceServers(
      'https://vitrail-rooms.workers.dev/__vitrail_mp/ice',
      async () => ({
        ok: true,
        json: async () => ({
          iceServers: [
            {
              urls: ['turns:turn.cloudflare.com:5349?transport=tcp'],
              username: 'dyn-u',
              credential: 'dyn-c',
            },
          ],
        }),
      })
    );
    expect(dynamicTurn).toHaveLength(1);
    expect(dynamicTurn[0].username).toBe('dyn-u');

    // 6. Forced Relay overrides (`VITE_MP_FORCE_RELAY=1`, `?forceRelay=1`, `?mpRelay=1`, `?transport=relay`)
    //    and `GameEngine` `syncBrowserRoomQueryParam` query preservation (`forceRelay=1&worker=0`)
    expect(isForcedRelayEnabled({}, { VITE_MP_FORCE_RELAY: '1' }, { search: '' })).toBe(true);
    expect(isForcedRelayEnabled({}, {}, { search: '?forceRelay=1' })).toBe(true);
    expect(isForcedRelayEnabled({}, {}, { search: '?mpRelay=true' })).toBe(true);
    expect(isForcedRelayEnabled({}, {}, { search: '?transport=relay' })).toBe(true);
    expect(isForcedRelayEnabled({}, {}, { search: '?worker=0' })).toBe(false);

    const savedWindow = (globalThis as unknown as { window?: unknown }).window;
    let replacedUrl = '';
    (globalThis as unknown as { window: unknown }).window = {
      location: {
        href: 'https://vitrail.andrewvoirol.com/?forceRelay=1&worker=0',
        protocol: 'https:',
        host: 'vitrail.andrewvoirol.com',
        hostname: 'vitrail.andrewvoirol.com',
        port: '',
        search: '?forceRelay=1&worker=0',
      },
      history: {
        replaceState: (_data: unknown, _unused: string, url?: string | URL | null) => {
          replacedUrl = String(url ?? '');
        },
      },
      addEventListener: () => {},
      removeEventListener: () => {},
    };
    try {
      const eng = new GameEngine();
      await eng.initHeadless();
      try {
        eng.startOnlineHost('QRY99', {
          relayUrl: 'ws://127.0.0.1:9/__vitrail_mp',
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: true,
        });
        expect(replacedUrl).toContain('forceRelay=1');
        expect(replacedUrl).toContain('worker=0');
        expect(replacedUrl).toContain('room=QRY99');
        const internalClient = (
          eng as unknown as { mpClient: VitrailMultiplayerClient }
        ).mpClient;
        expect(internalClient.isForcedRelay()).toBe(true);
      } finally {
        eng.dispose();
      }
    } finally {
      (globalThis as unknown as { window?: unknown }).window = savedWindow;
    }
  });

  it(
    'T-CHAOS2: enforces <= 4.0s automatic fallback to WebSocket room relay when WebRTC ICE stalls in checking or fails immediately',
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
      const relayUrl = `ws://127.0.0.1:${port}/__vitrail_mp`;

      const savedRtc = globalThis.RTCPeerConnection;
      let iceMode: 'stall' | 'fail' = 'stall';
      let closedPcCount = 0;

      // Mock RTCPeerConnection that negotiates SDP over relay/PeerJS but either stalls in `checking`
      // or transitions to `iceConnectionState = 'failed'` without ever opening `ctrl` DataChannel.
      class StallingOrFailingRTCPeerConnection {
        public signalingState = 'stable';
        public iceConnectionState = 'new';
        public connectionState = 'new';
        public localDescription: RTCSessionDescriptionInit | null = null;
        public remoteDescription: RTCSessionDescriptionInit | null = null;
        public onicecandidate: ((ev: unknown) => void) | null = null;
        public onconnectionstatechange: (() => void) | null = null;
        public oniceconnectionstatechange: (() => void) | null = null;
        private closed = false;

        createDataChannel(label: string) {
          return {
            label,
            readyState: 'connecting',
            binaryType: 'arraybuffer',
            bufferedAmount: 0,
            onopen: null,
            onmessage: null,
            onclose: null,
            onerror: null,
            send() {
              throw new Error('DataChannel never opened');
            },
            close() {
              this.readyState = 'closed';
            },
          };
        }

        async createOffer(): Promise<RTCSessionDescriptionInit> {
          return { type: 'offer', sdp: 'v=0\r\no=stall 1 1 IN IP4 127.0.0.1\r\n' };
        }

        async createAnswer(): Promise<RTCSessionDescriptionInit> {
          return { type: 'answer', sdp: 'v=0\r\no=stall 2 1 IN IP4 127.0.0.1\r\n' };
        }

        async setLocalDescription(desc: RTCSessionDescriptionInit): Promise<void> {
          this.localDescription = desc;
          this.signalingState = desc.type === 'offer' ? 'have-local-offer' : 'stable';
          this.maybeTriggerIceState();
        }

        async setRemoteDescription(desc: RTCSessionDescriptionInit): Promise<void> {
          this.remoteDescription = desc;
          this.signalingState = desc.type === 'offer' ? 'have-remote-offer' : 'stable';
          this.maybeTriggerIceState();
        }

        async addIceCandidate(): Promise<void> {}

        private maybeTriggerIceState(): void {
          if (this.closed || this.signalingState !== 'stable') return;
          this.iceConnectionState = 'checking';
          this.connectionState = 'connecting';
          if (iceMode === 'fail') {
            setTimeout(() => {
              if (this.closed) return;
              this.iceConnectionState = 'failed';
              this.connectionState = 'failed';
              this.oniceconnectionstatechange?.();
              this.onconnectionstatechange?.();
            }, 35);
          }
        }

        close(): void {
          if (this.closed) return;
          this.closed = true;
          closedPcCount++;
          this.connectionState = 'closed';
          this.iceConnectionState = 'closed';
        }
      }

      globalThis.RTCPeerConnection =
        StallingOrFailingRTCPeerConnection as unknown as typeof RTCPeerConnection;

      try {
        // Case A: WebRTC ICE stalls indefinitely in `checking` (PeerJS unreachable/stalled -> fallbackRelayUrl)
        // Using default `webrtcHandshakeTimeoutMs = 2800 ms` (`<= 3200 ms`), total fallback MUST finish `<= 4000 ms`!
        iceMode = 'stall';
        closedPcCount = 0;
        const tStartStall = performance.now();

        const hostStall = new VitrailMultiplayerClient({
          role: 'host',
          roomCode: 'ICE4S',
          relayUrl: null,
          fallbackRelayUrl: relayUrl,
          peerJsUrl: 'ws://127.0.0.1:9/peerjs', // Unreachable signaling / stalled P2P
          enablePeerJsFallback: true,
          enableWebRtcUpgrade: true,
        });
        const guestStall = new VitrailMultiplayerClient({
          role: 'guest',
          roomCode: 'ICE4S',
          relayUrl: null,
          fallbackRelayUrl: relayUrl,
          peerJsUrl: 'ws://127.0.0.1:9/peerjs',
          enablePeerJsFallback: true,
          enableWebRtcUpgrade: true,
        });

        let receivedWelcomeOnFallback = false;
        hostStall.connect();
        guestStall.connect();

        // Also test stalled `RTCPeerConnection` upgrade directly over relay (tears down stalled `pc` in <= 2.8 s)
        const hostUpgradeStall = new VitrailMultiplayerClient({
          role: 'host',
          roomCode: 'UPG4S',
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: true,
          webrtcHandshakeTimeoutMs: 1200,
        });
        const guestUpgradeStall = new VitrailMultiplayerClient({
          role: 'guest',
          roomCode: 'UPG4S',
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: true,
          webrtcHandshakeTimeoutMs: 1200,
          onControlMessage: (msg) => {
            if (msg.type === 'pong' || msg.type === 'hello') {
              receivedWelcomeOnFallback = true;
            }
          },
        });
        hostUpgradeStall.connect();
        guestUpgradeStall.connect();

        try {
          const ok = await waitUntilWall(
            () =>
              hostStall.getStatus().peerConnected &&
              guestStall.getStatus().peerConnected &&
              hostStall.getStatus().transport === 'ws-relay' &&
              guestStall.getStatus().transport === 'ws-relay' &&
              closedPcCount >= 2,
            3950
          );
          const elapsedStallMs = performance.now() - tStartStall;
          expect(ok).toBe(true);
          expect(elapsedStallMs).toBeLessThanOrEqual(4000);
          expect(hostUpgradeStall.getStatus().transport).toBe('ws-relay');
          expect(guestUpgradeStall.getStatus().transport).toBe('ws-relay');
          expect(receivedWelcomeOnFallback || guestUpgradeStall.getStatus().peerConnected).toBe(
            true
          );
        } finally {
          hostStall.dispose();
          guestStall.dispose();
          hostUpgradeStall.dispose();
          guestUpgradeStall.dispose();
        }

        // Case B: Immediate WebRTC ICE `'failed'` state -> tears down `pc` and stays/connects on `ws-relay` in < 1000 ms
        iceMode = 'fail';
        closedPcCount = 0;
        const tStartFail = performance.now();
        const hostFail = new VitrailMultiplayerClient({
          role: 'host',
          roomCode: 'ICEFL',
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: true,
        });
        const guestFail = new VitrailMultiplayerClient({
          role: 'guest',
          roomCode: 'ICEFL',
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: true,
        });
        try {
          hostFail.connect();
          guestFail.connect();
          const failRecovered = await waitUntilWall(
            () =>
              closedPcCount >= 2 &&
              hostFail.getStatus().peerConnected &&
              guestFail.getStatus().peerConnected &&
              guestFail.getStatus().transport === 'ws-relay',
            2000
          );
          const elapsedFailMs = performance.now() - tStartFail;
          expect(failRecovered).toBe(true);
          expect(elapsedFailMs).toBeLessThan(1500);
        } finally {
          hostFail.dispose();
          guestFail.dispose();
        }
      } finally {
        globalThis.RTCPeerConnection = savedRtc;
        relay.detach();
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    },
    15000
  );

  it(
    'T-CHAOS3: passes 100-seeded-match Adversarial Network Chaos Simulator (8% drop, 120+/-45ms jitter+reordering, mid-drag WebRTC->WS failover, occluded 30Hz Host pump, double-grab races & rapid rematches)',
    async () => {
      const simModPath = '../scripts/chaos-multiplayer-sim.mjs';
      const { createChaosMultiplayerSuite } = (await import(
        /* @vite-ignore */ simModPath
      )) as {
        createChaosMultiplayerSuite: (deps: Record<string, unknown>) => Promise<{
          runChaosSimulation: (opts: {
            matches: number;
            seed: number;
          }) => Promise<{
            matchesCompleted: number;
            rematchesCompleted: number;
            uncaughtExceptions: number;
            unhandledRejections: number;
            nanCount: number;
            antipodalFlipCount: number;
            equatorCrossingsTested: number;
            monotonicSeqViolations: number;
            staleRematchRollbacks: number;
            totalGuestRejectedStaleFrames: number;
            swatchParityChecks: number;
            swatchParityMismatches: number;
            doubleGrabViolations: number;
            midDragFailoversCompleted: number;
            occludedHostTurnsCompleted: number;
            audioParity: {
              host: { blockThud: number; glassImpact: number; glassShatter: number };
              guest: { blockThud: number; glassImpact: number; glassShatter: number };
              epochMismatches: number;
            };
            maxSteadyStateConvergenceErrorMm: number;
            convergenceViolationsOver2Mm: number;
            gateFailures: string[];
          }>;
        }>;
      };

      const suite = await createChaosMultiplayerSuite({
        GameEngine,
        buildDeterministicTowerMaterials,
        canonicalizeQuatHemisphere,
        quatRotateVec3,
      });

      const result = await suite.runChaosSimulation({
        matches: 100,
        seed: 0x56495452,
      });

      expect(result.gateFailures).toEqual([]);
      expect(result.matchesCompleted).toBe(100);
      expect(result.rematchesCompleted).toBe(100);
      expect(result.uncaughtExceptions).toBe(0);
      expect(result.unhandledRejections).toBe(0);
      expect(result.nanCount).toBe(0);
      expect(result.antipodalFlipCount).toBe(0);
      expect(result.equatorCrossingsTested).toBeGreaterThan(0);
      expect(result.monotonicSeqViolations).toBe(0);
      expect(result.staleRematchRollbacks).toBe(0);
      expect(result.totalGuestRejectedStaleFrames).toBeGreaterThan(0);
      expect(result.swatchParityMismatches).toBe(0);
      expect(result.swatchParityChecks).toBeGreaterThanOrEqual(200);
      expect(result.doubleGrabViolations).toBe(0);
      expect(result.midDragFailoversCompleted).toBe(100);
      expect(result.occludedHostTurnsCompleted).toBe(100);
      expect(result.audioParity.epochMismatches).toBe(0);
      expect(result.audioParity.guest.blockThud).toBe(result.audioParity.host.blockThud);
      expect(result.audioParity.guest.glassImpact).toBe(result.audioParity.host.glassImpact);
      expect(result.audioParity.guest.glassShatter).toBe(result.audioParity.host.glassShatter);
      expect(result.maxSteadyStateConvergenceErrorMm).toBeLessThan(2.0);
      expect(result.convergenceViolationsOver2Mm).toBe(0);
    },
    300000
  );
});

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

  it(
    'T-CHAOS4: static/preview Host without /__vitrail_mp survives PeerJS lobby wait (> deadline) & accepts late Guest OFFER; StrictMode unmount & BroadcastChannel WebGPU yield prevent Dawn adapter starvation',
    async () => {
      // --- Part 1: Static/Preview Host Lobby Survival & Late Guest PeerJS Offer (Bug 1 Regression) ---
      const savedWs = globalThis.WebSocket;
      const savedRtc = globalThis.RTCPeerConnection;

      let deadRelayDialCount = 0;
      const peerJsPeers = new Map<string, MockStaticPeerJsWebSocket>();
      const sdpMap = new Map<string, MockInstantRTCPeerConnection>();

      class MockStaticPeerJsWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        public readyState = MockStaticPeerJsWebSocket.CONNECTING;
        public binaryType = 'arraybuffer';
        public onopen: ((ev: unknown) => void) | null = null;
        public onmessage: ((ev: { data: unknown }) => void) | null = null;
        public onclose: ((ev: unknown) => void) | null = null;
        public onerror: ((ev: unknown) => void) | null = null;
        private peerId = '';
        private isPeerJs = false;

        constructor(url: string) {
          const parsed = new URL(url);
          if (parsed.pathname.includes('peerjs')) {
            this.isPeerJs = true;
            this.peerId = parsed.searchParams.get('id') ?? '';
            setTimeout(() => {
              if (this.readyState !== MockStaticPeerJsWebSocket.CONNECTING) return;
              this.readyState = MockStaticPeerJsWebSocket.OPEN;
              peerJsPeers.set(this.peerId, this);
              this.onopen?.({});
              setTimeout(() => {
                if (this.readyState === MockStaticPeerJsWebSocket.OPEN) {
                  this.onmessage?.({ data: JSON.stringify({ type: 'OPEN' }) });
                }
              }, 3);
            }, 5);
          } else {
            deadRelayDialCount++;
            setTimeout(() => {
              if (this.readyState === MockStaticPeerJsWebSocket.CLOSED) return;
              this.readyState = MockStaticPeerJsWebSocket.CLOSED;
              this.onerror?.({});
              this.onclose?.({ code: 1006, reason: 'Static server 404' });
            }, 6);
          }
        }

        send(data: string): void {
          if (!this.isPeerJs || this.readyState !== MockStaticPeerJsWebSocket.OPEN) return;
          const parsed = JSON.parse(data) as {
            type: string;
            dst?: string;
            payload?: unknown;
          };
          if (parsed.type === 'HEARTBEAT' || !parsed.dst) return;
          const target = peerJsPeers.get(parsed.dst);
          setTimeout(() => {
            if (target && target.readyState === MockStaticPeerJsWebSocket.OPEN) {
              target.onmessage?.({
                data: JSON.stringify({
                  type: parsed.type,
                  src: this.peerId,
                  dst: parsed.dst,
                  payload: parsed.payload,
                }),
              });
            }
          }, 5);
        }

        close(): void {
          if (this.readyState === MockStaticPeerJsWebSocket.CLOSED) return;
          this.readyState = MockStaticPeerJsWebSocket.CLOSED;
          if (this.isPeerJs && peerJsPeers.get(this.peerId) === this) {
            peerJsPeers.delete(this.peerId);
          }
          this.onclose?.({ code: 1000 });
        }
      }

      class MockInstantRTCPeerConnection {
        public signalingState = 'stable';
        public iceConnectionState = 'new';
        public connectionState = 'new';
        public localDescription: RTCSessionDescriptionInit | null = null;
        public remoteDescription: RTCSessionDescriptionInit | null = null;
        public onicecandidate: ((ev: unknown) => void) | null = null;
        public onconnectionstatechange: (() => void) | null = null;
        public oniceconnectionstatechange: (() => void) | null = null;
        public channels = new Map<
          number,
          {
            label: string;
            readyState: string;
            binaryType: string;
            bufferedAmount: number;
            onopen: (() => void) | null;
            onmessage: ((ev: { data: unknown }) => void) | null;
            onclose: (() => void) | null;
            onerror: (() => void) | null;
            send: (data: unknown) => void;
            close: () => void;
          }
        >();
        public partner: MockInstantRTCPeerConnection | null = null;
        private id = Math.random().toString(36).slice(2, 8);

        createDataChannel(label: string, opts?: { id?: number }) {
          const chId = opts?.id ?? (label === 'ctrl' ? 0 : 1);
          const self = this;
          const ch = {
            label,
            readyState: 'connecting',
            binaryType: 'arraybuffer',
            bufferedAmount: 0,
            onopen: null as (() => void) | null,
            onmessage: null as ((ev: { data: unknown }) => void) | null,
            onclose: null as (() => void) | null,
            onerror: null as (() => void) | null,
            send(data: unknown) {
              const peerCh = self.partner?.channels.get(chId);
              if (peerCh && peerCh.readyState === 'open') {
                setTimeout(() => peerCh.onmessage?.({ data }), 2);
              }
            },
            close() {
              this.readyState = 'closed';
            },
          };
          this.channels.set(chId, ch);
          return ch;
        }

        async createOffer(): Promise<RTCSessionDescriptionInit> {
          return { type: 'offer', sdp: `v=0\r\no=${this.id} 1 1 IN IP4 127.0.0.1\r\n` };
        }

        async createAnswer(): Promise<RTCSessionDescriptionInit> {
          return { type: 'answer', sdp: `v=0\r\no=${this.id} 2 1 IN IP4 127.0.0.1\r\n` };
        }

        async setLocalDescription(desc: RTCSessionDescriptionInit): Promise<void> {
          this.localDescription = desc;
          if (desc.sdp) sdpMap.set(desc.sdp, this);
          this.signalingState = desc.type === 'offer' ? 'have-local-offer' : 'stable';
          this.maybeOpen();
        }

        async setRemoteDescription(desc: RTCSessionDescriptionInit): Promise<void> {
          this.remoteDescription = desc;
          const remote = desc.sdp ? sdpMap.get(desc.sdp) : undefined;
          if (remote && remote !== this) {
            this.partner = remote;
            remote.partner = this;
          }
          this.signalingState = desc.type === 'offer' ? 'have-remote-offer' : 'stable';
          this.maybeOpen();
        }

        async addIceCandidate(): Promise<void> {}

        private maybeOpen(): void {
          if (!this.partner || this.signalingState !== 'stable' || this.partner.signalingState !== 'stable') {
            return;
          }
          setTimeout(() => {
            for (const p of [this, this.partner!]) {
              p.iceConnectionState = 'connected';
              p.connectionState = 'connected';
              p.oniceconnectionstatechange?.();
              p.onconnectionstatechange?.();
              for (const ch of p.channels.values()) {
                if (ch.readyState !== 'open') {
                  ch.readyState = 'open';
                  ch.onopen?.();
                }
              }
            }
          }, 5);
        }

        close(): void {
          this.connectionState = 'closed';
          this.iceConnectionState = 'closed';
          for (const ch of this.channels.values()) ch.readyState = 'closed';
        }
      }

      globalThis.WebSocket = MockStaticPeerJsWebSocket as unknown as typeof WebSocket;
      globalThis.RTCPeerConnection =
        MockInstantRTCPeerConnection as unknown as typeof RTCPeerConnection;

      try {
        // Simulate static/preview origin where primaryRelayUrl points to local `/__vitrail_mp` (404/closed)
        // and fallbackRelayUrl is null (`enablePeerJsFallback: true`)
        const staticHost = new VitrailMultiplayerClient({
          role: 'host',
          roomCode: 'PRVW1',
          relayUrl: 'ws://127.0.0.1:4173/__vitrail_mp',
          fallbackRelayUrl: null,
          peerJsUrl: 'wss://0.peerjs.com/peerjs',
          enablePeerJsFallback: true,
          enableWebRtcUpgrade: true,
          webrtcHandshakeTimeoutMs: 120,
        });
        staticHost.connect();

        // Wait > 2.6x `webrtcHandshakeTimeoutMs` before any Guest joins!
        await sleepWallMs(320);

        expect(staticHost.getStatus().connectionState).toBe('waiting_peer');
        expect(staticHost.getStatus().errorMessage).toBeNull();
        // Dead static `/__vitrail_mp` URL must NEVER be re-dialed after initial `!openedOnce` failure
        expect(deadRelayDialCount).toBe(1);

        // Now late Guest joins over PeerJS after 320 ms — Host MUST still accept OFFER & open WebRTC P2P!
        const staticGuest = new VitrailMultiplayerClient({
          role: 'guest',
          roomCode: 'PRVW1',
          relayUrl: 'ws://127.0.0.1:4173/__vitrail_mp',
          fallbackRelayUrl: null,
          peerJsUrl: 'wss://0.peerjs.com/peerjs',
          enablePeerJsFallback: true,
          enableWebRtcUpgrade: true,
          webrtcHandshakeTimeoutMs: 500,
        });
        staticGuest.connect();

        try {
          const paired = await waitUntilWall(
            () =>
              staticHost.getStatus().peerConnected &&
              staticGuest.getStatus().peerConnected &&
              staticHost.getStatus().transport === 'webrtc-p2p' &&
              staticGuest.getStatus().transport === 'webrtc-p2p',
            1500
          );
          expect(paired).toBe(true);
          expect(deadRelayDialCount).toBe(2); // 1 initial attempt on Host + 1 initial attempt on Guest
        } finally {
          staticHost.dispose();
          staticGuest.dispose();
        }
      } finally {
        globalThis.WebSocket = savedWs;
        globalThis.RTCPeerConnection = savedRtc;
      }

      // --- Part 2: React 19 StrictMode Synchronous Unmount & Sibling-Tab BroadcastChannel Yield (Bug 2 Regression) ---
      const navObj = globalThis.navigator as unknown as Record<string, unknown>;
      const prevGpuDesc = Object.getOwnPropertyDescriptor(navObj, 'gpu');
      let requestAdapterCalls = 0;
      Object.defineProperty(navObj, 'gpu', {
        configurable: true,
        writable: true,
        value: {
          requestAdapter: async () => {
            requestAdapterCalls++;
            return null;
          },
        },
      });

      try {
        // React 19 StrictMode Mount #1: `init()` followed synchronously by `dispose()` MUST exit at microtask fence
        // before constructing `WebGPURenderer` or invoking `navigator.gpu.requestAdapter()`!
        const strictMount1 = new GameEngine();
        const initPromise = strictMount1.init();
        strictMount1.dispose();
        await initPromise;
        expect(requestAdapterCalls).toBe(0);

        // Sibling-tab `BroadcastChannel('vitrail_webgpu_yield')` pulse sets & clears `gpuYieldUntilMs`
        if (typeof BroadcastChannel !== 'undefined') {
          const liveTabEngine = new GameEngine();
          await liveTabEngine.initHeadless();
          const engInternal = liveTabEngine as unknown as {
            ensureGpuYieldChannel: () => BroadcastChannel | null;
            gpuYieldUntilMs: number;
          };
          engInternal.ensureGpuYieldChannel();

          const siblingSender = new BroadcastChannel('vitrail_webgpu_yield');
          try {
            siblingSender.postMessage({ type: 'gpu-init-start', senderId: 'tab_guest_2' });
            const yielded = await waitUntilWall(() => engInternal.gpuYieldUntilMs > 0, 500);
            expect(yielded).toBe(true);

            siblingSender.postMessage({ type: 'gpu-init-done', senderId: 'tab_guest_2' });
            const resumed = await waitUntilWall(() => engInternal.gpuYieldUntilMs === 0, 500);
            expect(resumed).toBe(true);
          } finally {
            siblingSender.close();
            liveTabEngine.dispose();
          }
        }
      } finally {
        if (prevGpuDesc) {
          Object.defineProperty(navObj, 'gpu', prevGpuDesc);
        } else {
          delete navObj.gpu;
        }
      }
    }
  );

  it(
    'T-CHAOS5: replicates > 300 Voronoi shards (1:1 activeShardCount & < 2.0 mm pose convergence across all shards) via initial poses + round-robin < 4 KB 0x564d cursor',
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

      const host = new GameEngine();
      const guest = new GameEngine();
      const guestLate = new GameEngine();
      await host.initHeadless();
      await guest.initHeadless();
      await guestLate.initHeadless();

      let hostLastActiveShardCount = 0;
      let guestLastActiveShardCount = 0;
      (host as unknown as { listener: (t: { activeShardCount: number }) => void }).listener = (
        t
      ) => {
        hostLastActiveShardCount = t.activeShardCount;
      };
      (guest as unknown as { listener: (t: { activeShardCount: number }) => void }).listener = (
        t
      ) => {
        guestLastActiveShardCount = t.activeShardCount;
      };

      try {
        host.startOnlineHost('SHD33', {
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: false,
          roomSeed: 0x33557799,
        });
        guest.joinOnlineRoom('SHD33', {
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: false,
        });

        const paired = await waitUntilWall(
          () =>
            Boolean(
              host.getTelemetrySnapshot().mpPeerConnected &&
                guest.getTelemetrySnapshot().mpPeerConnected
            ),
          2500,
          () => {
            host.stepHeadless(1 / 60);
            guest.stepHeadless(1 / 120);
          }
        );
        expect(paired).toBe(true);

        // Shatter 16 blocks across the tower to spawn ~330–360 active Voronoi shards (> 2.7x MP_MAX_SHARDS_PER_FRAME=120)
        const hostPhysicsAny = host.physics as unknown as {
          shatterBlock: (blk: unknown, origin: [number, number, number], speed: number) => void;
        };
        let shatteredBlocks = 0;
        for (let i = 0; i < host.physics.blocks.length && shatteredBlocks < 16; i++) {
          const blk = host.physics.blocks[i];
          if (!blk.shattered && !blk.extracted && i % 3 === 1) {
            hostPhysicsAny.shatterBlock(
              blk,
              [blk.position[0], blk.position[1], blk.position[2]],
              1.4
            );
            shatteredBlocks++;
          }
        }

        // Step dynamic collapse for ~0.45 s so shards fall & scatter far from their initial fracture origins
        for (let step = 0; step < 28; step++) {
          host.stepHeadless(1 / 60);
          guest.stepHeadless(1 / 120);
          guest.stepHeadless(1 / 120);
          await sleepWallMs(35);
        }

        // Freeze Host physics so shard poses are stationary and verify round-robin 120-shard 0x564d cursor
        // converges ALL > 300 shards on Guest within < 2.0 mm AND HUD activeShardCount matches 1:1!
        host.benchPaused = true;
        for (let sweep = 0; sweep < 14; sweep++) {
          host.stepHeadless(1 / 60);
          for (let g = 0; g < 4; g++) {
            guest.stepHeadless(1 / 120);
          }
          await sleepWallMs(34);
        }

        const hostTel = host.getTelemetrySnapshot();
        const guestTel = guest.getTelemetrySnapshot();
        expect(hostTel.activeShardCount).toBeGreaterThan(300);
        expect(guestTel.activeShardCount).toBe(hostTel.activeShardCount);
        expect(guestLastActiveShardCount).toBe(hostLastActiveShardCount);

        const guestShardById = new Map<number, (typeof guest.physics.shards)[number]>();
        for (const gs of guest.physics.shards) {
          guestShardById.set(gs.id, gs);
        }
        expect(guestShardById.size).toBe(host.physics.shards.length);

        let maxShardErrMm = 0;
        for (let i = 0; i < host.physics.shards.length; i++) {
          const hs = host.physics.shards[i];
          const gs = guestShardById.get(hs.id);
          expect(gs).toBeDefined();
          const errMm =
            Math.hypot(
              hs.position[0] - gs!.position[0],
              hs.position[1] - gs!.position[1],
              hs.position[2] - gs!.position[2]
            ) * 100.0;
          if (errMm > maxShardErrMm) maxShardErrMm = errMm;
        }
        expect(maxShardErrMm).toBeLessThan(2.0);

        // Also verify late-joining / reconnecting Guest immediately gets 1:1 activeShardCount (> 300) on room_welcome!
        guest.leaveOnlineRoom();
        await sleepWallMs(40);
        guestLate.joinOnlineRoom('SHD33', {
          relayUrl,
          enablePeerJsFallback: false,
          enableWebRtcUpgrade: false,
        });
        const lateSynced = await waitUntilWall(
          () =>
            Boolean(
              guestLate.getTelemetrySnapshot().mpPeerConnected &&
                guestLate.getTelemetrySnapshot().activeShardCount === hostTel.activeShardCount
            ),
          2500,
          () => {
            host.stepHeadless(1 / 60);
            guestLate.stepHeadless(1 / 120);
          }
        );
        expect(lateSynced).toBe(true);
        expect(guestLate.getTelemetrySnapshot().activeShardCount).toBeGreaterThan(300);
      } finally {
        host.dispose();
        guest.dispose();
        guestLate.dispose();
        relay.detach();
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    },
    20000
  );
});

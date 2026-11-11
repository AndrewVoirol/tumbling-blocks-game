import {
  decodeMultiplayerStateFrame,
  encodeMultiplayerStateFrame,
  generateRoomSeed,
  MultiplayerConnectionState,
  MultiplayerControlMessage,
  MultiplayerRole,
  MultiplayerStateFrame,
  MultiplayerStatusSnapshot,
  MultiplayerTransportKind,
  normalizeRoomCode,
} from './protocol';

export const DEFAULT_PEERJS_SIGNAL_URL = 'wss://0.peerjs.com/peerjs';

export const DEFAULT_PUBLIC_ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.cloudflare.com:3478' },
  { urls: 'stun:stun.l.google.com:19302' },
];

export interface VitrailMultiplayerClientOptions {
  role: MultiplayerRole;
  roomCode: string;
  roomSeed?: number;
  /** Optional explicit WebSocket room relay URL (`ws://.../__vitrail_mp` or Cloudflare Worker URL) */
  relayUrl?: string | null;
  /** Optional override for PeerJS signaling WebSocket endpoint (`wss://0.peerjs.com/peerjs`) */
  peerJsUrl?: string | null;
  /** Disable PeerJS public fallback (e.g. in isolated local relay unit tests) */
  enablePeerJsFallback?: boolean;
  /** Disable WebRTC upgrade over relay (e.g. to force pure WS relay transport) */
  enableWebRtcUpgrade?: boolean;
  iceServers?: RTCIceServer[];
  onControlMessage?: (msg: MultiplayerControlMessage) => void;
  onStateFrame?: (frame: MultiplayerStateFrame, byteLength: number) => void;
  onStatusChange?: (status: MultiplayerStatusSnapshot) => void;
  onPeerConnected?: () => void;
  onPeerDisconnected?: () => void;
}

interface PeerJsWireMessage {
  type: string;
  src?: string;
  dst?: string;
  payload?: {
    connectionId?: string;
    type?: string;
    sdp?: RTCSessionDescriptionInit;
    candidate?: RTCIceCandidateInit;
    msg?: string;
  };
}

interface RelayEnvelopeMessage {
  __relay?: 'peer_joined' | 'peer_left' | 'room_state' | 'error';
  role?: MultiplayerRole;
  hasHost?: boolean;
  hasGuest?: boolean;
  message?: string;
}

function randomAlphaNum(len: number): string {
  const chars = 'abcdefghjkmnpqrstuvwxyz23456789';
  const bytes = new Uint8Array(len);
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < len; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  let out = '';
  for (let i = 0; i < len; i++) out += chars[bytes[i] % chars.length];
  return out;
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function resolveAutoRelayUrl(): string | null {
  const metaEnv = (
    import.meta as unknown as { env?: Record<string, string | boolean | undefined> }
  ).env;
  const configured = metaEnv?.VITE_MP_RELAY_URL;
  if (typeof configured === 'string' && configured.trim().length > 0) {
    return configured.trim();
  }
  if (typeof window === 'undefined' || !window.location) {
    return null;
  }
  const host = window.location.hostname;
  const port = window.location.port;
  const isLocalOrLan =
    Boolean(metaEnv?.DEV) ||
    host === 'localhost' ||
    host === '127.0.0.1' ||
    host === '[::1]' ||
    host.endsWith('.local') ||
    port === '5173' ||
    port === '4173' ||
    /^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[0-1])\.)/.test(host);

  if (!isLocalOrLan) {
    return null;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/__vitrail_mp`;
}

function buildRelayRoomSocketUrl(
  baseUrl: string,
  roomCode: string,
  role: MultiplayerRole,
  clientId: string
): string {
  try {
    const u = new URL(baseUrl);
    if (u.pathname === '' || u.pathname === '/') {
      u.pathname = '/__vitrail_mp';
    }
    u.searchParams.set('room', roomCode);
    u.searchParams.set('role', role);
    u.searchParams.set('clientId', clientId);
    return u.toString();
  } catch {
    const sep = baseUrl.includes('?') ? '&' : '?';
    return `${baseUrl}${sep}room=${encodeURIComponent(roomCode)}&role=${encodeURIComponent(
      role
    )}&clientId=${encodeURIComponent(clientId)}`;
  }
}

function buildPeerJsSocketUrl(baseUrl: string, peerId: string, token: string): string {
  const clean = baseUrl.replace(/\/+$/, '');
  const sep = clean.includes('?') ? '&' : '?';
  return `${clean}${sep}key=peerjs&id=${encodeURIComponent(peerId)}&token=${encodeURIComponent(
    token
  )}`;
}

/**
 * Dual-Transport Hybrid Internet Multiplayer Client (`VitrailMultiplayerClient`)
 *
 * 1. **Public Internet Zero-Backend P2P (`WebRTC DataChannel` + Public PeerJS Signaling)**:
 *    Works out-of-the-box from static `vitrail.andrewvoirol.com` (`COOP: same-origin`,
 *    `COEP: credentialless`) using public STUN (`stun.cloudflare.com`, `stun.l.google.com`)
 *    and PeerJS signaling (`wss://0.peerjs.com/peerjs`) keyed by 5-char Room Code (`K7M4P`).
 *    Uses two negotiated SCTP DataChannels:
 *    - `"ctrl"` (`id: 0`, reliable ordered) for handshake, seed, turn locks, shard spawns & audio
 *    - `"state"` (`id: 1`, unordered `maxRetransmits: 0`) for `30 Hz` `< 4 KB` binary snapshots
 *
 * 2. **WebSocket Room Relay Fast-Path / Fallback (`/__vitrail_mp` / `VITE_MP_RELAY_URL`)**:
 *    Relays binary `ArrayBuffer` state frames + JSON control messages directly on Vite dev/preview
 *    or Cloudflare Durable Objects, while opportunistically upgrading to P2P `RTCDataChannel`.
 */
export class VitrailMultiplayerClient {
  public readonly role: MultiplayerRole;
  public readonly roomCode: string;
  public readonly clientId: string;
  public roomSeed: number;

  private options: VitrailMultiplayerClientOptions;
  private connectionState: MultiplayerConnectionState = 'idle';
  private transport: MultiplayerTransportKind = 'none';
  private peerConnected: boolean = false;
  private hadPeerConnectionOnce: boolean = false;
  private pingMs: number | null = null;
  private errorMessage: string | null = null;
  private disposed: boolean = false;

  // WebSocket Room Relay state
  private relayWs: WebSocket | null = null;
  private relayConnected: boolean = false;
  private relayPeerPresent: boolean = false;
  private relayReconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // PeerJS public signaling WebSocket state
  private peerJsWs: WebSocket | null = null;
  private peerJsOpen: boolean = false;
  private localPeerId: string;
  private hostTargetPeerId: string;
  private remoteGuestPeerId: string | null = null;
  private peerJsToken: string;
  private peerJsHeartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private peerJsOfferRetryTimer: ReturnType<typeof setInterval> | null = null;
  private peerJsReconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // WebRTC RTCPeerConnection + Dual DataChannels
  private pc: RTCPeerConnection | null = null;
  private ctrlChannel: RTCDataChannel | null = null;
  private stateChannel: RTCDataChannel | null = null;
  private pendingIceCandidates: RTCIceCandidateInit[] = [];
  private activeConnectionId: string = '';
  private remoteClientId: string | null = null;
  private pcRemoteClientId: string | null = null;
  private unloadListenersAttached: boolean = false;

  // Ping / RTT measurement
  private pingIntervalTimer: ReturnType<typeof setInterval> | null = null;
  private nextPingId: number = 1;
  private pendingPings = new Map<number, number>();

  constructor(options: VitrailMultiplayerClientOptions) {
    this.options = options;
    this.role = options.role;
    this.roomCode = normalizeRoomCode(options.roomCode);
    this.roomSeed = (options.roomSeed ?? generateRoomSeed()) >>> 0;
    this.clientId = `${this.role}-${randomAlphaNum(8)}`;
    this.peerJsToken = `tok${randomAlphaNum(10)}`;
    this.hostTargetPeerId = `vitrail-v1-${this.roomCode}-host`;
    this.localPeerId =
      this.role === 'host'
        ? this.hostTargetPeerId
        : `vitrail-v1-${this.roomCode}-g-${randomAlphaNum(6)}`;
  }

  public getStatus(): MultiplayerStatusSnapshot {
    return {
      active: !this.disposed && this.connectionState !== 'idle',
      role: this.role,
      roomCode: this.roomCode,
      roomSeed: this.roomSeed,
      connectionState: this.connectionState,
      transport: this.transport,
      peerConnected: this.peerConnected,
      pingMs: this.pingMs !== null ? Math.round(this.pingMs) : null,
      errorMessage: this.errorMessage,
    };
  }

  public connect(): void {
    if (this.disposed) return;
    this.errorMessage = null;
    this.updateStatus(this.role === 'host' ? 'waiting_peer' : 'connecting');

    const explicitRelay = this.options.relayUrl;
    const autoRelay = explicitRelay !== undefined ? explicitRelay : resolveAutoRelayUrl();
    const allowPeerJs =
      this.options.enablePeerJsFallback ?? (explicitRelay === undefined || explicitRelay === null);

    if (autoRelay) {
      this.connectRelaySocket(autoRelay, allowPeerJs);
    } else if (allowPeerJs) {
      this.connectPeerJsSignaling();
    } else {
      this.errorMessage = 'No multiplayer relay or signaling endpoint configured.';
      this.updateStatus('error');
    }

    this.startPingLoop();
    this.attachUnloadListeners();
  }

  private readonly onWindowUnload = (): void => {
    this.sendBestEffortPeerBye();
    this.dispose();
  };

  private attachUnloadListeners(): void {
    if (this.unloadListenersAttached) return;
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('pagehide', this.onWindowUnload);
      window.addEventListener('beforeunload', this.onWindowUnload);
      this.unloadListenersAttached = true;
    }
  }

  private detachUnloadListeners(): void {
    if (!this.unloadListenersAttached) return;
    if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
      window.removeEventListener('pagehide', this.onWindowUnload);
      window.removeEventListener('beforeunload', this.onWindowUnload);
    }
    this.unloadListenersAttached = false;
  }

  private sendBestEffortPeerBye(): void {
    const raw = JSON.stringify({ type: 'peer_bye', clientId: this.clientId });
    if (this.ctrlChannel?.readyState === 'open') {
      try {
        this.ctrlChannel.send(raw);
      } catch {
        // Ignore
      }
    }
    if (this.relayWs?.readyState === WebSocket.OPEN) {
      try {
        this.relayWs.send(raw);
      } catch {
        // Ignore
      }
    }
  }

  private updateStatus(nextState?: MultiplayerConnectionState): void {
    if (this.disposed) return;
    if (nextState) {
      this.connectionState = nextState;
    }

    const p2pReady = this.ctrlChannel?.readyState === 'open';
    const relayReady = this.relayConnected && this.relayPeerPresent;
    const nowConnected = p2pReady || relayReady;

    this.transport = p2pReady ? 'webrtc-p2p' : relayReady ? 'ws-relay' : 'none';

    const wasConnected = this.peerConnected;
    this.peerConnected = nowConnected;

    if (nowConnected) {
      this.hadPeerConnectionOnce = true;
      this.connectionState = 'connected';
      this.errorMessage = null;
    } else if (this.connectionState !== 'error' && this.connectionState !== 'idle') {
      if (this.hadPeerConnectionOnce) {
        this.connectionState = 'reconnecting';
      } else {
        this.connectionState = this.role === 'host' ? 'waiting_peer' : 'connecting';
      }
    }

    this.options.onStatusChange?.(this.getStatus());

    if (!wasConnected && nowConnected) {
      this.sendControl({
        type: 'hello',
        protocol: 1,
        role: this.role,
        clientId: this.clientId,
        roomCode: this.roomCode,
      });
      this.sendPing();
      this.options.onPeerConnected?.();
    } else if (wasConnected && !nowConnected) {
      this.options.onPeerDisconnected?.();
    }
  }

  // ============================================================================
  // 1. WebSocket Room Relay (`/___vitrail_mp` & Cloudflare Durable Object Relay)
  // ============================================================================

  private connectRelaySocket(baseUrl: string, fallbackToPeerJsOnError: boolean): void {
    if (this.disposed || typeof WebSocket === 'undefined') return;
    this.cleanupRelaySocket();

    const wsUrl = buildRelayRoomSocketUrl(baseUrl, this.roomCode, this.role, this.clientId);
    let openedOnce = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch {
      if (fallbackToPeerJsOnError) {
        this.connectPeerJsSignaling();
      } else {
        this.errorMessage = 'Failed to open WebSocket room relay.';
        this.updateStatus('error');
      }
      return;
    }

    ws.binaryType = 'arraybuffer';
    this.relayWs = ws;

    ws.onopen = () => {
      if (this.disposed || this.relayWs !== ws) return;
      openedOnce = true;
      this.relayConnected = true;
      this.updateStatus();
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this.disposed || this.relayWs !== ws) return;
      this.handleIncomingTransportPayload(ev.data, 'relay');
    };

    ws.onerror = () => {
      // Handled in onclose
    };

    ws.onclose = () => {
      if (this.relayWs !== ws) return;
      this.relayWs = null;
      this.relayConnected = false;
      this.relayPeerPresent = false;
      this.updateStatus();

      if (this.disposed) return;

      if (!openedOnce && fallbackToPeerJsOnError && !this.peerJsWs) {
        // Same-origin static host without `/__vitrail_mp` -> seamlessly switch to public PeerJS P2P!
        this.connectPeerJsSignaling();
        return;
      }

      if (openedOnce) {
        this.relayReconnectTimer = setTimeout(() => {
          this.relayReconnectTimer = null;
          if (!this.disposed) {
            this.connectRelaySocket(baseUrl, fallbackToPeerJsOnError);
          }
        }, 1200);
      } else if (!fallbackToPeerJsOnError) {
        this.errorMessage = 'Room relay connection closed.';
        this.updateStatus('error');
      }
    };
  }

  private handleRelayMetaEnvelope(meta: RelayEnvelopeMessage): void {
    if (meta.__relay === 'error') {
      this.errorMessage = meta.message ?? 'Room error';
      this.updateStatus('error');
      return;
    }
    if (meta.__relay === 'room_state') {
      const peerHere = this.role === 'host' ? Boolean(meta.hasGuest) : Boolean(meta.hasHost);
      this.relayPeerPresent = peerHere;
      this.updateStatus();
      if (
        peerHere &&
        this.role === 'guest' &&
        (this.options.enableWebRtcUpgrade ?? true) &&
        typeof RTCPeerConnection !== 'undefined' &&
        this.ctrlChannel?.readyState !== 'open'
      ) {
        void this.initiateWebRtcOffer('relay');
      }
      return;
    }
    if (meta.__relay === 'peer_joined') {
      if (meta.role && meta.role !== this.role) {
        // Tear down any orphaned WebRTC DataChannel from a previous peer tab session before resyncing
        this.closePeerConnection();
        this.remoteClientId = null;
        this.relayPeerPresent = true;
        this.updateStatus();
        if (
          this.role === 'guest' &&
          (this.options.enableWebRtcUpgrade ?? true) &&
          typeof RTCPeerConnection !== 'undefined'
        ) {
          void this.initiateWebRtcOffer('relay');
        }
      }
      return;
    }
    if (meta.__relay === 'peer_left') {
      if (meta.role && meta.role !== this.role) {
        this.closePeerConnection();
        this.remoteClientId = null;
        this.relayPeerPresent = false;
        this.updateStatus();
      }
    }
  }

  // ============================================================================
  // 2. Public Internet PeerJS Signaling + Native WebRTC `RTCDataChannel` P2P
  // ============================================================================

  private connectPeerJsSignaling(): void {
    if (this.disposed || typeof WebSocket === 'undefined') return;
    this.cleanupPeerJsSocket();

    const baseUrl = this.options.peerJsUrl || DEFAULT_PEERJS_SIGNAL_URL;
    const wsUrl = buildPeerJsSocketUrl(baseUrl, this.localPeerId, this.peerJsToken);

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch {
      this.errorMessage = 'Unable to reach public signaling server.';
      this.updateStatus('error');
      return;
    }

    this.peerJsWs = ws;

    ws.onopen = () => {
      if (this.disposed || this.peerJsWs !== ws) return;
      this.peerJsHeartbeatTimer = setInterval(() => {
        if (this.peerJsWs?.readyState === WebSocket.OPEN) {
          this.peerJsWs.send(JSON.stringify({ type: 'HEARTBEAT' }));
        }
      }, 5000);
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this.disposed || this.peerJsWs !== ws || typeof ev.data !== 'string') return;
      let msg: PeerJsWireMessage;
      try {
        msg = JSON.parse(ev.data) as PeerJsWireMessage;
      } catch {
        return;
      }
      this.handlePeerJsSignalMessage(msg);
    };

    ws.onerror = () => {
      // Handled in onclose
    };

    ws.onclose = () => {
      if (this.peerJsWs !== ws) return;
      this.peerJsWs = null;
      this.peerJsOpen = false;
      if (this.peerJsHeartbeatTimer !== null) {
        clearInterval(this.peerJsHeartbeatTimer);
        this.peerJsHeartbeatTimer = null;
      }
      if (this.peerJsOfferRetryTimer !== null) {
        clearInterval(this.peerJsOfferRetryTimer);
        this.peerJsOfferRetryTimer = null;
      }
      if (this.disposed) return;

      if (this.connectionState !== 'error') {
        this.peerJsReconnectTimer = setTimeout(() => {
          this.peerJsReconnectTimer = null;
          if (!this.disposed) {
            this.connectPeerJsSignaling();
          }
        }, 1800);
      }
    };
  }

  private handlePeerJsSignalMessage(msg: PeerJsWireMessage): void {
    switch (msg.type) {
      case 'OPEN': {
        this.peerJsOpen = true;
        this.updateStatus();
        if (this.role === 'guest' && this.ctrlChannel?.readyState !== 'open') {
          void this.initiateWebRtcOffer('peerjs');
          if (this.peerJsOfferRetryTimer !== null) {
            clearInterval(this.peerJsOfferRetryTimer);
          }
          this.peerJsOfferRetryTimer = setInterval(() => {
            if (this.disposed || this.ctrlChannel?.readyState === 'open') {
              if (this.peerJsOfferRetryTimer !== null) {
                clearInterval(this.peerJsOfferRetryTimer);
                this.peerJsOfferRetryTimer = null;
              }
              return;
            }
            if (this.peerJsOpen) {
              void this.initiateWebRtcOffer('peerjs');
            }
          }, 2600);
        }
        break;
      }
      case 'ID-TAKEN': {
        if (this.role === 'host') {
          this.errorMessage = `Room ${this.roomCode} already has an active Host.`;
          this.updateStatus('error');
          this.cleanupPeerJsSocket();
        } else {
          this.localPeerId = `vitrail-v1-${this.roomCode}-g-${randomAlphaNum(6)}`;
          this.connectPeerJsSignaling();
        }
        break;
      }
      case 'OFFER': {
        if (this.role === 'host' && msg.src && msg.payload?.sdp) {
          this.remoteGuestPeerId = msg.src;
          this.activeConnectionId = msg.payload.connectionId ?? `dc_${randomAlphaNum(8)}`;
          void this.acceptWebRtcOffer(msg.payload.sdp, 'peerjs', msg.src);
        }
        break;
      }
      case 'ANSWER': {
        if (this.role === 'guest' && msg.payload?.sdp) {
          void this.acceptWebRtcAnswer(msg.payload.sdp);
        }
        break;
      }
      case 'CANDIDATE': {
        if (msg.payload?.candidate) {
          void this.addRemoteIceCandidate(msg.payload.candidate);
        }
        break;
      }
      case 'EXPIRE': {
        // Host has not opened room yet or hasn't answered yet; retry timer handles re-offer cleanly
        break;
      }
      case 'LEAVE': {
        if (msg.src && (msg.src === this.remoteGuestPeerId || msg.src === this.hostTargetPeerId)) {
          this.closePeerConnection();
          this.updateStatus();
        }
        break;
      }
      case 'ERROR': {
        this.errorMessage = msg.payload?.msg ?? 'Signaling error';
        this.updateStatus('error');
        break;
      }
    }
  }

  private createFreshPeerConnection(
    via: 'peerjs' | 'relay',
    peerJsTargetId?: string
  ): RTCPeerConnection {
    this.closePeerConnection();
    this.pendingIceCandidates = [];
    this.pcRemoteClientId = this.remoteClientId;

    const pc = new RTCPeerConnection({
      iceServers: this.options.iceServers ?? DEFAULT_PUBLIC_ICE_SERVERS,
    });
    this.pc = pc;

    const ctrl = pc.createDataChannel('ctrl', {
      negotiated: true,
      id: 0,
      ordered: true,
    });
    ctrl.binaryType = 'arraybuffer';
    this.ctrlChannel = ctrl;

    const state = pc.createDataChannel('state', {
      negotiated: true,
      id: 1,
      ordered: false,
      maxRetransmits: 0,
    });
    state.binaryType = 'arraybuffer';
    this.stateChannel = state;

    ctrl.onopen = () => {
      if (this.disposed || this.ctrlChannel !== ctrl) return;
      if (this.peerJsOfferRetryTimer !== null) {
        clearInterval(this.peerJsOfferRetryTimer);
        this.peerJsOfferRetryTimer = null;
      }
      this.updateStatus();
    };

    ctrl.onmessage = (ev: MessageEvent) => {
      if (this.disposed || this.ctrlChannel !== ctrl) return;
      this.handleIncomingTransportPayload(ev.data, 'webrtc');
    };

    ctrl.onclose = () => {
      if (this.ctrlChannel !== ctrl) return;
      this.updateStatus();
    };

    state.onmessage = (ev: MessageEvent) => {
      if (this.disposed || this.stateChannel !== state) return;
      this.handleIncomingTransportPayload(ev.data, 'webrtc');
    };

    pc.onicecandidate = (ev: RTCPeerConnectionIceEvent) => {
      if (this.disposed || this.pc !== pc || !ev.candidate) return;
      const candInit = ev.candidate.toJSON();
      if (via === 'peerjs') {
        const dst =
          this.role === 'host'
            ? peerJsTargetId ?? this.remoteGuestPeerId
            : this.hostTargetPeerId;
        if (dst && this.peerJsWs?.readyState === WebSocket.OPEN) {
          this.peerJsWs.send(
            JSON.stringify({
              type: 'CANDIDATE',
              dst,
              payload: {
                connectionId: this.activeConnectionId,
                type: 'data',
                candidate: candInit,
              },
            })
          );
        }
      } else if (this.relayWs?.readyState === WebSocket.OPEN) {
        const sig: MultiplayerControlMessage = {
          type: 'rtc_signal',
          from: this.role,
          signalType: 'candidate',
          candidate: candInit,
        };
        this.relayWs.send(JSON.stringify(sig));
      }
    };

    pc.onconnectionstatechange = () => {
      if (this.disposed || this.pc !== pc) return;
      if (
        pc.connectionState === 'failed' ||
        pc.connectionState === 'disconnected' ||
        pc.connectionState === 'closed'
      ) {
        this.closePeerConnection();
        this.updateStatus();
      }
    };

    pc.oniceconnectionstatechange = () => {
      if (this.disposed || this.pc !== pc) return;
      if (
        pc.iceConnectionState === 'failed' ||
        pc.iceConnectionState === 'disconnected' ||
        pc.iceConnectionState === 'closed'
      ) {
        this.closePeerConnection();
        this.updateStatus();
      }
    };

    return pc;
  }

  private async initiateWebRtcOffer(via: 'peerjs' | 'relay'): Promise<void> {
    if (this.disposed || typeof RTCPeerConnection === 'undefined') return;
    try {
      this.activeConnectionId = `dc_${randomAlphaNum(8)}`;
      const pc = this.createFreshPeerConnection(via, this.hostTargetPeerId);
      const offer = await pc.createOffer();
      if (this.disposed || this.pc !== pc) return;
      await pc.setLocalDescription(offer);
      if (this.disposed || this.pc !== pc || !pc.localDescription) return;

      const sdpInit: RTCSessionDescriptionInit = {
        type: pc.localDescription.type,
        sdp: pc.localDescription.sdp,
      };

      if (via === 'peerjs') {
        if (this.peerJsWs?.readyState === WebSocket.OPEN) {
          this.peerJsWs.send(
            JSON.stringify({
              type: 'OFFER',
              dst: this.hostTargetPeerId,
              payload: {
                connectionId: this.activeConnectionId,
                type: 'data',
                sdp: sdpInit,
              },
            })
          );
        }
      } else if (this.relayWs?.readyState === WebSocket.OPEN) {
        const sig: MultiplayerControlMessage = {
          type: 'rtc_signal',
          from: this.role,
          signalType: 'offer',
          sdp: sdpInit,
        };
        this.relayWs.send(JSON.stringify(sig));
      }
    } catch {
      // Next retry or relay fallback handles transient WebRTC errors cleanly
    }
  }

  private async acceptWebRtcOffer(
    offer: RTCSessionDescriptionInit,
    via: 'peerjs' | 'relay',
    peerJsSrc?: string
  ): Promise<void> {
    if (this.disposed || typeof RTCPeerConnection === 'undefined') return;
    try {
      const pc = this.createFreshPeerConnection(via, peerJsSrc);
      await pc.setRemoteDescription(offer);
      if (this.disposed || this.pc !== pc) return;
      await this.flushPendingIceCandidates(pc);

      const answer = await pc.createAnswer();
      if (this.disposed || this.pc !== pc) return;
      await pc.setLocalDescription(answer);
      if (this.disposed || this.pc !== pc || !pc.localDescription) return;

      const sdpInit: RTCSessionDescriptionInit = {
        type: pc.localDescription.type,
        sdp: pc.localDescription.sdp,
      };

      if (via === 'peerjs' && peerJsSrc) {
        if (this.peerJsWs?.readyState === WebSocket.OPEN) {
          this.peerJsWs.send(
            JSON.stringify({
              type: 'ANSWER',
              dst: peerJsSrc,
              payload: {
                connectionId: this.activeConnectionId,
                type: 'data',
                sdp: sdpInit,
              },
            })
          );
        }
      } else if (via === 'relay' && this.relayWs?.readyState === WebSocket.OPEN) {
        const sig: MultiplayerControlMessage = {
          type: 'rtc_signal',
          from: this.role,
          signalType: 'answer',
          sdp: sdpInit,
        };
        this.relayWs.send(JSON.stringify(sig));
      }
    } catch {
      // Keep relay active if WebRTC SDP fails
    }
  }

  private async acceptWebRtcAnswer(answer: RTCSessionDescriptionInit): Promise<void> {
    const pc = this.pc;
    if (!pc || this.disposed) return;
    if (pc.signalingState !== 'have-local-offer') return;
    try {
      await pc.setRemoteDescription(answer);
      if (this.disposed || this.pc !== pc) return;
      await this.flushPendingIceCandidates(pc);
    } catch {
      // Ignore duplicate/stale answer
    }
  }

  private async addRemoteIceCandidate(cand: RTCIceCandidateInit): Promise<void> {
    const pc = this.pc;
    if (!pc || this.disposed) return;
    if (!pc.remoteDescription) {
      this.pendingIceCandidates.push(cand);
      return;
    }
    try {
      await pc.addIceCandidate(cand);
    } catch {
      // Ignore benign late ICE candidate warnings
    }
  }

  private async flushPendingIceCandidates(pc: RTCPeerConnection): Promise<void> {
    const queued = this.pendingIceCandidates;
    this.pendingIceCandidates = [];
    for (const cand of queued) {
      try {
        await pc.addIceCandidate(cand);
      } catch {
        // Ignore
      }
    }
  }

  // ============================================================================
  // 3. Unified Control & Binary Frame Dispatch
  // ============================================================================

  private handleIncomingTransportPayload(
    data: unknown,
    source: 'webrtc' | 'relay'
  ): void {
    if (data instanceof ArrayBuffer) {
      this.processBinaryFrame(data);
      return;
    }
    if (ArrayBuffer.isView(data)) {
      const ab = data.buffer.slice(
        data.byteOffset,
        data.byteOffset + data.byteLength
      ) as ArrayBuffer;
      this.processBinaryFrame(ab);
      return;
    }
    if (typeof Blob !== 'undefined' && data instanceof Blob) {
      void data.arrayBuffer().then((ab) => {
        if (!this.disposed) this.processBinaryFrame(ab);
      });
      return;
    }
    if (typeof data !== 'string') return;

    let parsed: MultiplayerControlMessage & RelayEnvelopeMessage;
    try {
      parsed = JSON.parse(data) as MultiplayerControlMessage & RelayEnvelopeMessage;
    } catch {
      return;
    }

    if (parsed.__relay) {
      this.handleRelayMetaEnvelope(parsed);
      return;
    }

    if (parsed.type === 'rtc_signal' && source === 'relay') {
      if (parsed.from === this.role) return;
      if (parsed.signalType === 'offer' && this.role === 'host' && parsed.sdp) {
        void this.acceptWebRtcOffer(parsed.sdp, 'relay');
      } else if (parsed.signalType === 'answer' && this.role === 'guest' && parsed.sdp) {
        void this.acceptWebRtcAnswer(parsed.sdp);
      } else if (parsed.signalType === 'candidate' && parsed.candidate) {
        void this.addRemoteIceCandidate(parsed.candidate);
      }
      return;
    }

    if (parsed.type === 'ping') {
      this.sendControl({ type: 'pong', id: parsed.id, t0: parsed.t0 });
      return;
    }

    if (parsed.type === 'pong') {
      const sentAt = this.pendingPings.get(parsed.id);
      if (sentAt !== undefined) {
        this.pendingPings.delete(parsed.id);
        const rtt = Math.max(0.5, nowMs() - sentAt);
        this.pingMs = this.pingMs === null ? rtt : this.pingMs * 0.65 + rtt * 0.35;
        this.options.onStatusChange?.(this.getStatus());
      }
      return;
    }

    if (parsed.type === 'peer_bye') {
      if (this.remoteClientId === null || this.remoteClientId === parsed.clientId) {
        this.closePeerConnection();
        this.remoteClientId = null;
        if (source === 'relay') {
          this.relayPeerPresent = false;
        }
        this.updateStatus();
      } else if (this.pcRemoteClientId === parsed.clientId) {
        this.closePeerConnection();
        this.updateStatus();
      }
      return;
    }

    if (parsed.type === 'hello') {
      if (parsed.role !== this.role) {
        const prevClientId = this.remoteClientId;
        this.remoteClientId = parsed.clientId;
        if (source === 'webrtc') {
          this.pcRemoteClientId = parsed.clientId;
        } else if (source === 'relay') {
          if (
            this.ctrlChannel?.readyState === 'open' &&
            (this.pcRemoteClientId !== parsed.clientId ||
              (prevClientId !== null && prevClientId !== parsed.clientId))
          ) {
            // Reconnecting tab sent `hello` over relay with a new `clientId`: tear down zombie DataChannel!
            this.closePeerConnection();
          } else if (this.pc && this.pcRemoteClientId === null) {
            this.pcRemoteClientId = parsed.clientId;
          }
          if (!this.relayPeerPresent) {
            this.relayPeerPresent = true;
          }
          this.updateStatus();
        }
      }
      this.options.onControlMessage?.(parsed);
      return;
    }

    this.options.onControlMessage?.(parsed);
  }

  private processBinaryFrame(ab: ArrayBuffer): void {
    const decoded = decodeMultiplayerStateFrame(ab);
    if (!decoded) return;
    this.options.onStateFrame?.(decoded, ab.byteLength);
  }

  public sendControl(msg: MultiplayerControlMessage): boolean {
    if (this.disposed) return false;
    const raw = JSON.stringify(msg);
    if (this.ctrlChannel?.readyState === 'open') {
      try {
        this.ctrlChannel.send(raw);
        return true;
      } catch {
        // Fall through to relay if available
      }
    }
    if (this.relayWs?.readyState === WebSocket.OPEN) {
      try {
        this.relayWs.send(raw);
        return true;
      } catch {
        return false;
      }
    }
    return false;
  }

  public sendStateFrame(frame: MultiplayerStateFrame): number {
    if (this.disposed || !this.peerConnected) return 0;
    const buf = encodeMultiplayerStateFrame(frame);
    if (this.sendStateBuffer(buf)) {
      return buf.byteLength;
    }
    return 0;
  }

  public sendStateBuffer(buffer: ArrayBuffer): boolean {
    if (this.disposed || !this.peerConnected) return false;
    if (this.stateChannel?.readyState === 'open') {
      if (this.stateChannel.bufferedAmount <= 65536) {
        try {
          this.stateChannel.send(buffer);
          return true;
        } catch {
          // Fall through to relay if available
        }
      } else {
        return false;
      }
    }
    if (this.relayWs?.readyState === WebSocket.OPEN && this.relayPeerPresent) {
      if (this.relayWs.bufferedAmount <= 131072) {
        try {
          this.relayWs.send(buffer);
          return true;
        } catch {
          return false;
        }
      }
    }
    return false;
  }

  private startPingLoop(): void {
    if (this.pingIntervalTimer !== null) {
      clearInterval(this.pingIntervalTimer);
    }
    this.pingIntervalTimer = setInterval(() => {
      if (!this.disposed && this.peerConnected) {
        this.sendPing();
      }
    }, 1500);
  }

  private sendPing(): void {
    const id = this.nextPingId++;
    const t0 = nowMs();
    if (this.pendingPings.size > 16) {
      const oldest = this.pendingPings.keys().next().value;
      if (oldest !== undefined) this.pendingPings.delete(oldest);
    }
    this.pendingPings.set(id, t0);
    this.sendControl({ type: 'ping', id, t0 });
  }

  private closePeerConnection(): void {
    if (this.ctrlChannel) {
      try {
        this.ctrlChannel.onopen = null;
        this.ctrlChannel.onmessage = null;
        this.ctrlChannel.onclose = null;
        this.ctrlChannel.close();
      } catch {
        // Ignore
      }
      this.ctrlChannel = null;
    }
    if (this.stateChannel) {
      try {
        this.stateChannel.onmessage = null;
        this.stateChannel.close();
      } catch {
        // Ignore
      }
      this.stateChannel = null;
    }
    if (this.pc) {
      try {
        this.pc.onicecandidate = null;
        this.pc.onconnectionstatechange = null;
        this.pc.oniceconnectionstatechange = null;
        this.pc.close();
      } catch {
        // Ignore
      }
      this.pc = null;
    }
    this.pcRemoteClientId = null;
  }

  private cleanupRelaySocket(): void {
    if (this.relayReconnectTimer !== null) {
      clearTimeout(this.relayReconnectTimer);
      this.relayReconnectTimer = null;
    }
    if (this.relayWs) {
      const ws = this.relayWs;
      this.relayWs = null;
      this.relayConnected = false;
      this.relayPeerPresent = false;
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      } catch {
        // Ignore
      }
    }
  }

  private cleanupPeerJsSocket(): void {
    if (this.peerJsReconnectTimer !== null) {
      clearTimeout(this.peerJsReconnectTimer);
      this.peerJsReconnectTimer = null;
    }
    if (this.peerJsHeartbeatTimer !== null) {
      clearInterval(this.peerJsHeartbeatTimer);
      this.peerJsHeartbeatTimer = null;
    }
    if (this.peerJsOfferRetryTimer !== null) {
      clearInterval(this.peerJsOfferRetryTimer);
      this.peerJsOfferRetryTimer = null;
    }
    if (this.peerJsWs) {
      const ws = this.peerJsWs;
      this.peerJsWs = null;
      this.peerJsOpen = false;
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      } catch {
        // Ignore
      }
    }
  }

  public dispose(): void {
    if (this.disposed) return;
    this.detachUnloadListeners();
    this.sendBestEffortPeerBye();
    this.disposed = true;
    if (this.pingIntervalTimer !== null) {
      clearInterval(this.pingIntervalTimer);
      this.pingIntervalTimer = null;
    }
    this.pendingPings.clear();
    this.closePeerConnection();
    this.cleanupRelaySocket();
    this.cleanupPeerJsSocket();
    this.connectionState = 'idle';
    this.peerConnected = false;
    this.transport = 'none';
  }
}

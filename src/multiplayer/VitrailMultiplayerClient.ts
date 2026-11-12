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

/** Default WebRTC ICE/DataChannel open deadline (`2.8 s`, strictly `<= 3.2 s` so total WS relay fallback finishes `<= 4.0 s`) */
export const DEFAULT_WEBRTC_HANDSHAKE_TIMEOUT_MS = 2800;

export const DEFAULT_PUBLIC_ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.cloudflare.com:3478' },
  { urls: 'stun:stun.l.google.com:19302' },
];

export interface VitrailMultiplayerClientOptions {
  role: MultiplayerRole;
  roomCode: string;
  roomSeed?: number;
  /** Optional explicit per-tab persistent `clientId` (restored from `sessionStorage` across `F5` reloads) */
  clientId?: string;
  /** True when reclaiming an existing per-tab session across an `F5` reload */
  isSessionReclaim?: boolean;
  /** Lobby keepalive interval (`ms`) while waiting for remote peer (`default: 25000`) */
  lobbyKeepaliveIntervalMs?: number;
  /** Optional explicit WebSocket room relay URL (`ws://.../__vitrail_mp` or Cloudflare Worker URL) */
  relayUrl?: string | null;
  /** Optional secondary WebSocket relay URL dialed if primary relay fails or WebRTC ICE stalls/fails */
  fallbackRelayUrl?: string | null;
  /** Optional override for PeerJS signaling WebSocket endpoint (`wss://0.peerjs.com/peerjs`) */
  peerJsUrl?: string | null;
  /** Disable PeerJS public fallback (e.g. in isolated local relay unit tests) */
  enablePeerJsFallback?: boolean;
  /** Disable WebRTC upgrade over relay (e.g. to force pure WS relay transport) */
  enableWebRtcUpgrade?: boolean;
  /** Force pure WebSocket relay transport (`enableWebRtcUpgrade=false` & `enablePeerJsFallback=false`) */
  forceRelay?: boolean;
  /** Maximum time (`ms`) to wait for WebRTC `ctrl` DataChannel to open before tearing down `pc` and falling back to WS relay (default `2800 ms`, capped `<= 3200 ms`) */
  webrtcHandshakeTimeoutMs?: number;
  /** Optional HTTP(S) endpoint returning `{ iceServers: RTCIceServer[] }` short-lived TURN credentials */
  iceEndpointUrl?: string | null;
  iceServers?: RTCIceServer[];
  onControlMessage?: (msg: MultiplayerControlMessage) => void;
  onStateFrame?: (frame: MultiplayerStateFrame, byteLength: number) => void;
  onStatusChange?: (status: MultiplayerStatusSnapshot) => void;
  onPeerConnected?: () => void;
  onPeerDisconnected?: () => void;
  /** Fired when an active `webrtc-p2p` session downgrades seamlessly onto an already-open `ws-relay` socket */
  onTransportFailover?: (nextTransport: MultiplayerTransportKind, reason: string) => void;
  /** Optional notification when explicit `peer_bye` arrives from the remote peer */
  onRemotePeerBye?: (remoteClientId: string) => void;
}

export interface BrowserLocationLike {
  protocol?: string;
  host?: string;
  hostname?: string;
  port?: string;
  search?: string;
}

export interface ResolvedRelayEndpoints {
  primaryRelayUrl: string | null;
  fallbackRelayUrl: string | null;
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
    metadata?: { clientId?: string };
    code?: number;
    reason?: string;
    msg?: string;
  };
}

interface RelayEnvelopeMessage {
  __relay?: 'peer_joined' | 'peer_left' | 'room_state' | 'error' | 'keepalive';
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

function readViteMetaEnv(): Record<string, string | boolean | undefined> | undefined {
  return (
    import.meta as unknown as { env?: Record<string, string | boolean | undefined> }
  ).env;
}

function parseBooleanFlag(val: unknown): boolean | undefined {
  if (typeof val === 'boolean') return val;
  if (typeof val === 'string') {
    const clean = val.trim().toLowerCase();
    if (clean === '1' || clean === 'true' || clean === 'yes' || clean === 'on') return true;
    if (clean === '0' || clean === 'false' || clean === 'no' || clean === 'off') return false;
  }
  return undefined;
}

/**
 * Determines whether forced WebSocket relay mode is active via explicit client options,
 * browser URL query parameters (`?forceRelay=1`, `?mpRelay=1`, `?transport=relay`),
 * or `VITE_MP_FORCE_RELAY=1`.
 */
export function isForcedRelayEnabled(
  options?: Pick<VitrailMultiplayerClientOptions, 'forceRelay'>,
  envOverride?: Record<string, unknown>,
  searchOverride?: string | { search?: string }
): boolean {
  if (options?.forceRelay !== undefined) {
    return Boolean(options.forceRelay);
  }

  const search =
    searchOverride !== undefined
      ? typeof searchOverride === 'string'
        ? searchOverride
        : searchOverride.search ?? ''
      : typeof window !== 'undefined' && window.location
        ? window.location.search ?? ''
        : '';

  if (search) {
    try {
      const params = new URLSearchParams(search);
      const forceParam = parseBooleanFlag(params.get('forceRelay'));
      if (forceParam !== undefined) return forceParam;
      const mpRelayParam = parseBooleanFlag(params.get('mpRelay'));
      if (mpRelayParam !== undefined) return mpRelayParam;
      const transportParam = params.get('transport')?.trim().toLowerCase();
      if (
        transportParam === 'relay' ||
        transportParam === 'ws' ||
        transportParam === 'ws-relay'
      ) {
        return true;
      }
    } catch {
      // Ignore malformed search query
    }
  }

  const env = envOverride ?? readViteMetaEnv();
  const envForce = parseBooleanFlag(env?.VITE_MP_FORCE_RELAY);
  return envForce === true;
}

/**
 * Resolves primary and fallback WebSocket relay endpoints:
 * - Explicit `options.relayUrl` or `?relay=<url>` takes top precedence.
 * - On `localhost` / `127.0.0.1` / `[::1]` / LAN / Vite `DEV`, local `ws(s)://<host>/__vitrail_mp`
 *   is ALWAYS tried first even when `.env.production` sets `VITE_MP_RELAY_URL`, seamlessly falling
 *   back to `VITE_MP_RELAY_URL` if local `/__vitrail_mp` is absent (e.g. static localhost server).
 * - On production origins, uses `VITE_MP_RELAY_URL`.
 */
export function resolveRelayEndpoints(
  options?: Pick<VitrailMultiplayerClientOptions, 'relayUrl' | 'fallbackRelayUrl'>,
  envOverride?: Record<string, unknown>,
  locationOverride?: BrowserLocationLike | null
): ResolvedRelayEndpoints {
  const env = envOverride ?? readViteMetaEnv();
  const configuredRaw = env?.VITE_MP_RELAY_URL;
  const configuredEnv =
    typeof configuredRaw === 'string' && configuredRaw.trim().length > 0
      ? configuredRaw.trim()
      : null;

  const normalizeOpt = (v: string | null | undefined): string | null =>
    typeof v === 'string' && v.trim().length > 0 ? v.trim() : null;

  if (options?.relayUrl !== undefined) {
    return {
      primaryRelayUrl: normalizeOpt(options.relayUrl),
      fallbackRelayUrl:
        options.fallbackRelayUrl !== undefined ? normalizeOpt(options.fallbackRelayUrl) : null,
    };
  }

  const loc: BrowserLocationLike | null =
    locationOverride !== undefined
      ? locationOverride
      : typeof window !== 'undefined' && window.location
        ? window.location
        : null;

  if (loc?.search) {
    try {
      const params = new URLSearchParams(loc.search);
      const qRelay = params.get('relay');
      if (qRelay !== null) {
        const trimmed = qRelay.trim();
        const lower = trimmed.toLowerCase();
        if (trimmed === '' || lower === '0' || lower === 'none' || lower === 'off') {
          return {
            primaryRelayUrl: null,
            fallbackRelayUrl: normalizeOpt(options?.fallbackRelayUrl),
          };
        }
        return {
          primaryRelayUrl: trimmed,
          fallbackRelayUrl:
            options?.fallbackRelayUrl !== undefined
              ? normalizeOpt(options.fallbackRelayUrl)
              : configuredEnv && configuredEnv !== trimmed
                ? configuredEnv
                : null,
        };
      }
    } catch {
      // Ignore malformed search query
    }
  }

  if (loc && loc.hostname && loc.host) {
    const host = loc.hostname;
    const port = loc.port ?? '';
    const isLocalOrLan =
      Boolean(env?.DEV) ||
      host === 'localhost' ||
      host === '127.0.0.1' ||
      host === '[::1]' ||
      host.endsWith('.local') ||
      port === '5173' ||
      port === '4173' ||
      /^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[0-1])\.)/.test(host);

    if (isLocalOrLan) {
      const proto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
      const localRelay = `${proto}//${loc.host}/__vitrail_mp`;
      const fallback =
        options?.fallbackRelayUrl !== undefined
          ? normalizeOpt(options.fallbackRelayUrl)
          : configuredEnv && configuredEnv !== localRelay
            ? configuredEnv
            : null;
      return {
        primaryRelayUrl: localRelay,
        fallbackRelayUrl: fallback,
      };
    }
  }

  return {
    primaryRelayUrl: configuredEnv,
    fallbackRelayUrl:
      options?.fallbackRelayUrl !== undefined ? normalizeOpt(options.fallbackRelayUrl) : null,
  };
}

/**
 * Parses `VITE_MP_TURN_URLS` (comma/semicolon/whitespace separated `turn:`/`turns:`/`stun:` URLs)
 * plus optional `VITE_MP_TURN_USERNAME` and `VITE_MP_TURN_CREDENTIAL` alongside `DEFAULT_PUBLIC_ICE_SERVERS`.
 */
export function parseConfiguredIceServers(
  envOverride?: Record<string, unknown>
): RTCIceServer[] {
  const servers: RTCIceServer[] = DEFAULT_PUBLIC_ICE_SERVERS.map((s) => ({ ...s }));
  const seenStun = new Set<string>(
    DEFAULT_PUBLIC_ICE_SERVERS.flatMap((s) => (Array.isArray(s.urls) ? s.urls : [s.urls]))
  );

  const env = envOverride ?? readViteMetaEnv();
  const rawUrls = typeof env?.VITE_MP_TURN_URLS === 'string' ? env.VITE_MP_TURN_URLS.trim() : '';
  if (!rawUrls) {
    return servers;
  }

  const tokens = rawUrls
    .split(/[,;\s]+/)
    .map((u) => u.trim())
    .filter((u) => /^(stuns?|turns?):/i.test(u));

  const stunUrls: string[] = [];
  const turnUrls: string[] = [];
  for (const u of tokens) {
    if (/^turns?:/i.test(u)) {
      turnUrls.push(u);
    } else if (!seenStun.has(u)) {
      seenStun.add(u);
      stunUrls.push(u);
    }
  }

  for (const sUrl of stunUrls) {
    servers.push({ urls: sUrl });
  }

  const username =
    typeof env?.VITE_MP_TURN_USERNAME === 'string' ? env.VITE_MP_TURN_USERNAME.trim() : '';
  const credential =
    typeof env?.VITE_MP_TURN_CREDENTIAL === 'string' ? env.VITE_MP_TURN_CREDENTIAL.trim() : '';

  // W3C WebRTC §4.2.2 requires TURN/TURNS entries to supply both `username` and `credential`
  if (turnUrls.length > 0 && username.length > 0 && credential.length > 0) {
    servers.push({
      urls: turnUrls.length === 1 ? turnUrls[0] : turnUrls,
      username,
      credential,
    });
  }

  return servers;
}

export function deriveRelayIceEndpointUrl(
  relayBaseUrl: string | null,
  explicitIceUrl?: string | null,
  envOverride?: Record<string, unknown>
): string | null {
  if (typeof explicitIceUrl === 'string' && explicitIceUrl.trim().length > 0) {
    return explicitIceUrl.trim();
  }
  const env = envOverride ?? readViteMetaEnv();
  const envIce = env?.VITE_MP_ICE_URL;
  if (typeof envIce === 'string' && envIce.trim().length > 0) {
    return envIce.trim();
  }
  if (!relayBaseUrl) return null;
  try {
    const u = new URL(relayBaseUrl);
    u.protocol = u.protocol === 'wss:' || u.protocol === 'https:' ? 'https:' : 'http:';
    u.pathname = '/__vitrail_mp/ice';
    u.search = '';
    return u.toString();
  } catch {
    return null;
  }
}

export type MinimalFetchFn = (
  url: string,
  init?: RequestInit
) => Promise<{ ok: boolean; json: () => Promise<unknown> }>;

export async function fetchDynamicTurnIceServers(
  iceEndpointUrl: string,
  timeoutMsOrFetch: number | MinimalFetchFn = 1200,
  timeoutMsOverride: number = 1200
): Promise<RTCIceServer[]> {
  const customFetch = typeof timeoutMsOrFetch === 'function' ? timeoutMsOrFetch : null;
  const timeoutMs =
    typeof timeoutMsOrFetch === 'number' ? timeoutMsOrFetch : timeoutMsOverride;
  const fetchFn: MinimalFetchFn | null =
    customFetch ??
    (typeof fetch === 'function' ? (fetch.bind(globalThis) as MinimalFetchFn) : null);
  if (!fetchFn) return [];
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timer = controller
    ? setTimeout(() => {
        try {
          controller.abort();
        } catch {
          // Ignore
        }
      }, timeoutMs)
    : null;
  try {
    const res = await fetchFn(iceEndpointUrl, {
      method: 'GET',
      signal: controller?.signal,
    });
    if (!res.ok) return [];
    const payload = (await res.json()) as {
      iceServers?: RTCIceServer[];
      urls?: string | string[];
      username?: string;
      credential?: string;
    };
    const candidates: RTCIceServer[] = Array.isArray(payload.iceServers)
      ? payload.iceServers
      : payload.urls
        ? [
            {
              urls: payload.urls,
              username: payload.username,
              credential: payload.credential,
            },
          ]
        : [];
    return candidates.filter((entry) => {
      if (!entry || !entry.urls) return false;
      const list = Array.isArray(entry.urls) ? entry.urls : [entry.urls];
      const hasTurn = list.some((u) => typeof u === 'string' && /^turns?:/i.test(u));
      if (hasTurn) {
        return (
          typeof entry.username === 'string' &&
          entry.username.length > 0 &&
          typeof entry.credential === 'string' &&
          entry.credential.length > 0
        );
      }
      return list.length > 0;
    });
  } catch {
    return [];
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
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
  )}&version=1.5.4`;
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
  public static readonly LOBBY_KEEPALIVE_INTERVAL_MS = 25_000;
  private relayWs: WebSocket | null = null;
  private relayConnected: boolean = false;
  private relayPeerPresent: boolean = false;
  private relayReconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private relayKeepaliveTimer: ReturnType<typeof setInterval> | null = null;
  private terminalRoomError: boolean = false;
  private hostPeerJsIdTakenRetries: number = 0;
  private primaryRelayBaseUrl: string | null = null;
  private fallbackRelayBaseUrl: string | null = null;
  private activeRelayBaseUrl: string | null = null;
  private failedRelayBaseUrls = new Set<string>();
  private forcedRelayMode: boolean = false;

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
  private peerJsFallbackDeadlineTimer: ReturnType<typeof setTimeout> | null = null;

  // WebRTC RTCPeerConnection + Dual DataChannels
  private pc: RTCPeerConnection | null = null;
  private ctrlChannel: RTCDataChannel | null = null;
  private stateChannel: RTCDataChannel | null = null;
  private webrtcHandshakeTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly webrtcHandshakeTimeoutMs: number;
  private webrtcUpgradeSuppressed: boolean = false;
  private lastFailoverReason: string = '';
  private resolvedIceServers: RTCIceServer[];
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
    const providedId = options.clientId?.trim().slice(0, 48);
    this.clientId = providedId && providedId.length >= 4 ? providedId : `${this.role}-${randomAlphaNum(8)}`;
    this.peerJsToken = `tok${randomAlphaNum(10)}`;
    this.hostTargetPeerId = `vitrail-v1-${this.roomCode}-host`;
    const cleanGuestSuffix =
      this.clientId.replace(/[^a-zA-Z0-9]/g, '').slice(-8) || randomAlphaNum(6);
    this.localPeerId =
      this.role === 'host'
        ? this.hostTargetPeerId
        : `vitrail-v1-${this.roomCode}-g-${cleanGuestSuffix}`;

    this.forcedRelayMode = isForcedRelayEnabled(options);
    const endpoints = resolveRelayEndpoints(options);
    this.primaryRelayBaseUrl = endpoints.primaryRelayUrl;
    this.fallbackRelayBaseUrl = endpoints.fallbackRelayUrl;

    const rawTimeout = options.webrtcHandshakeTimeoutMs ?? DEFAULT_WEBRTC_HANDSHAKE_TIMEOUT_MS;
    this.webrtcHandshakeTimeoutMs = Math.max(50, Math.min(3200, Math.round(rawTimeout)));

    this.resolvedIceServers = options.iceServers
      ? options.iceServers.map((s) => ({ ...s }))
      : parseConfiguredIceServers();
  }

  public isForcedRelay(): boolean {
    return this.forcedRelayMode;
  }

  public getResolvedIceServers(): ReadonlyArray<RTCIceServer> {
    return this.resolvedIceServers;
  }

  private canUpgradeWebRtc(): boolean {
    if (this.forcedRelayMode || this.webrtcUpgradeSuppressed) {
      return false;
    }
    return (this.options.enableWebRtcUpgrade ?? true) && typeof RTCPeerConnection !== 'undefined';
  }

  private isPeerJsAllowed(): boolean {
    if (this.forcedRelayMode) {
      return false;
    }
    if (this.options.enablePeerJsFallback !== undefined) {
      return Boolean(this.options.enablePeerJsFallback);
    }
    const explicitRelay = this.options.relayUrl;
    return explicitRelay === undefined || explicitRelay === null;
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
    this.terminalRoomError = false;
    this.hostPeerJsIdTakenRetries = 0;
    this.webrtcUpgradeSuppressed = false;
    this.failedRelayBaseUrls.clear();
    this.updateStatus(this.role === 'host' ? 'waiting_peer' : 'connecting');

    const primaryRelay = this.primaryRelayBaseUrl;
    const secondaryRelay = this.fallbackRelayBaseUrl;
    const allowPeerJs = this.isPeerJsAllowed();

    // Non-blocking dynamic TURN credential probe when an explicit ICE endpoint or unauthenticated TURN env is configured
    if (!this.options.iceServers && !this.forcedRelayMode) {
      const env = readViteMetaEnv();
      const shouldProbeIce =
        Boolean(this.options.iceEndpointUrl) ||
        Boolean(env?.VITE_MP_ICE_URL) ||
        Boolean(env?.VITE_MP_TURN_URLS && (!env?.VITE_MP_TURN_USERNAME || !env?.VITE_MP_TURN_CREDENTIAL));
      if (shouldProbeIce) {
        const iceUrl = deriveRelayIceEndpointUrl(
          primaryRelay ?? secondaryRelay,
          this.options.iceEndpointUrl,
          env
        );
        if (iceUrl) {
          void fetchDynamicTurnIceServers(iceUrl).then((extra) => {
            if (!this.disposed && extra.length > 0) {
              this.resolvedIceServers = [...this.resolvedIceServers, ...extra];
            }
          });
        }
      }
    }

    if (primaryRelay) {
      this.connectRelaySocket(primaryRelay, allowPeerJs, secondaryRelay);
    } else if (allowPeerJs) {
      this.connectPeerJsSignaling();
    } else if (secondaryRelay) {
      this.connectRelaySocket(secondaryRelay, false, null);
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

    const prevTransport = this.transport;
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
    } else if (wasConnected && nowConnected && prevTransport === 'webrtc-p2p' && this.transport === 'ws-relay') {
      // Seamless transport downgrade (`webrtc-p2p -> ws-relay`) while `relayReady` stayed true:
      // Immediately announce presence and trigger full state/audio resync with zero interruption!
      this.sendControl({
        type: 'hello',
        protocol: 1,
        role: this.role,
        clientId: this.clientId,
        roomCode: this.roomCode,
      });
      this.sendPing();
      this.options.onTransportFailover?.('ws-relay', this.lastFailoverReason || 'webrtc-downgrade');
    } else if (wasConnected && !nowConnected) {
      this.options.onPeerDisconnected?.();
    }
  }

  // ============================================================================
  // 1. WebSocket Room Relay (`/___vitrail_mp` & Cloudflare Durable Object Relay)
  // ============================================================================

  private getPreferredFallbackRelayUrl(): string | null {
    const candidates = [
      this.activeRelayBaseUrl,
      this.fallbackRelayBaseUrl,
      this.primaryRelayBaseUrl,
    ];
    for (let i = 0; i < candidates.length; i++) {
      const url = candidates[i];
      if (url && !this.failedRelayBaseUrls.has(url)) {
        return url;
      }
    }
    return null;
  }

  private ensureRelaySocketConnectedImmediate(): void {
    if (this.disposed || typeof WebSocket === 'undefined') return;
    if (this.relayWs && (this.relayWs.readyState === WebSocket.OPEN || this.relayWs.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const targetUrl = this.getPreferredFallbackRelayUrl();
    if (!targetUrl) return;
    if (this.relayReconnectTimer !== null) {
      clearTimeout(this.relayReconnectTimer);
      this.relayReconnectTimer = null;
    }
    const secondaryCandidate =
      targetUrl === this.primaryRelayBaseUrl &&
      this.fallbackRelayBaseUrl &&
      !this.failedRelayBaseUrls.has(this.fallbackRelayBaseUrl)
        ? this.fallbackRelayBaseUrl
        : null;
    this.connectRelaySocket(targetUrl, false, secondaryCandidate);
  }

  private connectRelaySocket(
    baseUrl: string,
    fallbackToPeerJsOnError: boolean,
    nextFallbackRelayUrl: string | null = null
  ): void {
    if (this.disposed || typeof WebSocket === 'undefined') return;
    this.cleanupRelaySocket();

    const wsUrl = buildRelayRoomSocketUrl(baseUrl, this.roomCode, this.role, this.clientId);
    let openedOnce = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch {
      this.failedRelayBaseUrls.add(baseUrl);
      if (nextFallbackRelayUrl && nextFallbackRelayUrl !== baseUrl) {
        this.connectRelaySocket(nextFallbackRelayUrl, fallbackToPeerJsOnError, null);
      } else if (fallbackToPeerJsOnError) {
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
      this.activeRelayBaseUrl = baseUrl;
      this.failedRelayBaseUrls.delete(baseUrl);
      this.relayConnected = true;
      if (this.relayKeepaliveTimer !== null) {
        clearInterval(this.relayKeepaliveTimer);
      }
      const keepaliveMs = Math.max(
        25,
        this.options.lobbyKeepaliveIntervalMs ??
          VitrailMultiplayerClient.LOBBY_KEEPALIVE_INTERVAL_MS
      );
      this.relayKeepaliveTimer = setInterval(() => {
        if (
          !this.disposed &&
          !this.terminalRoomError &&
          this.relayWs === ws &&
          ws.readyState === WebSocket.OPEN &&
          this.relayConnected &&
          !this.peerConnected
        ) {
          try {
            ws.send(JSON.stringify({ __relay: 'keepalive' }));
          } catch {
            // Ignore transient socket close
          }
        }
      }, keepaliveMs);
      this.updateStatus();
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this.disposed || this.relayWs !== ws) return;
      this.handleIncomingTransportPayload(ev.data, 'relay');
    };

    ws.onerror = () => {
      // Handled in onclose
    };

    ws.onclose = (ev?: CloseEvent) => {
      if (this.relayWs !== ws) return;
      this.relayWs = null;
      this.relayConnected = false;
      this.relayPeerPresent = false;
      if (this.relayKeepaliveTimer !== null) {
        clearInterval(this.relayKeepaliveTimer);
        this.relayKeepaliveTimer = null;
      }
      if (!openedOnce) {
        this.failedRelayBaseUrls.add(baseUrl);
      }

      const closeCode = typeof ev?.code === 'number' ? ev.code : 1006;
      const closeReason = typeof ev?.reason === 'string' ? ev.reason.trim() : '';
      if (closeCode === 1008 || this.terminalRoomError) {
        this.terminalRoomError = true;
        if (!this.errorMessage) {
          this.errorMessage = closeReason || 'Room full (2/2 players)';
        }
        this.closePeerConnection();
        this.cleanupPeerJsSocket();
        this.updateStatus('error');
        return;
      }

      this.updateStatus();

      if (this.disposed) return;

      if (!openedOnce && nextFallbackRelayUrl && nextFallbackRelayUrl !== baseUrl) {
        // Localhost static preview without Vite `/__vitrail_mp` -> seamlessly switch to configured Durable Object WS relay!
        this.connectRelaySocket(nextFallbackRelayUrl, fallbackToPeerJsOnError, null);
        return;
      }

      if (!openedOnce && fallbackToPeerJsOnError && !this.peerJsWs) {
        // Static host without relay -> switch to public PeerJS P2P!
        this.connectPeerJsSignaling();
        return;
      }

      if (openedOnce) {
        this.relayReconnectTimer = setTimeout(() => {
          this.relayReconnectTimer = null;
          if (!this.disposed && !this.terminalRoomError) {
            this.connectRelaySocket(baseUrl, fallbackToPeerJsOnError, null);
          }
        }, 600);
      } else if (!fallbackToPeerJsOnError) {
        this.errorMessage = 'Room relay connection closed.';
        this.updateStatus('error');
      }
    };
  }

  private handleRelayMetaEnvelope(meta: RelayEnvelopeMessage): void {
    if (meta.__relay === 'keepalive') {
      return;
    }
    if (meta.__relay === 'error') {
      this.terminalRoomError = true;
      this.errorMessage = meta.message ?? 'Room error';
      this.closePeerConnection();
      this.cleanupPeerJsSocket();
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
        this.canUpgradeWebRtc() &&
        this.ctrlChannel?.readyState !== 'open' &&
        !this.pc
      ) {
        void this.initiateWebRtcOffer('relay');
      }
      return;
    }
    if (meta.__relay === 'peer_joined') {
      if (meta.role && meta.role !== this.role) {
        // Tear down any orphaned WebRTC DataChannel from a previous peer tab session before resyncing
        this.closePeerConnection();
        this.webrtcUpgradeSuppressed = false;
        this.remoteClientId = null;
        this.relayPeerPresent = true;
        this.updateStatus();
        if (this.role === 'guest' && this.canUpgradeWebRtc()) {
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
    if (this.disposed || typeof WebSocket === 'undefined' || this.forcedRelayMode) return;
    this.cleanupPeerJsSocket();

    const fallbackRelay = this.getPreferredFallbackRelayUrl();
    if (fallbackRelay) {
      this.peerJsFallbackDeadlineTimer = setTimeout(() => {
        this.peerJsFallbackDeadlineTimer = null;
        if (this.disposed || this.ctrlChannel?.readyState === 'open') return;
        if (this.role === 'host' && this.peerJsOpen && !this.pc && !this.hadPeerConnectionOnce) {
          // Host PeerJS lobby is open and waiting for a Guest OFFER (`webrtcHandshakeTimer` guards
          // ICE once OFFER arrives); connect any live backup room relay without suppressing WebRTC!
          if (this.getPreferredFallbackRelayUrl()) {
            this.ensureRelaySocketConnectedImmediate();
          }
          return;
        }
        this.handleWebRtcDisconnectOrFailover('peerjs-handshake-timeout');
      }, this.webrtcHandshakeTimeoutMs);
    }

    const baseUrl = this.options.peerJsUrl || DEFAULT_PEERJS_SIGNAL_URL;
    const wsUrl = buildPeerJsSocketUrl(baseUrl, this.localPeerId, this.peerJsToken);

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch {
      if (fallbackRelay) {
        this.ensureRelaySocketConnectedImmediate();
      } else {
        this.errorMessage = 'Unable to reach public signaling server.';
        this.updateStatus('error');
      }
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

      if (this.ctrlChannel?.readyState !== 'open' && this.getPreferredFallbackRelayUrl()) {
        this.ensureRelaySocketConnectedImmediate();
        return;
      }

      if (this.connectionState !== 'error' && !this.forcedRelayMode && !this.webrtcUpgradeSuppressed) {
        this.peerJsReconnectTimer = setTimeout(() => {
          this.peerJsReconnectTimer = null;
          if (!this.disposed) {
            this.connectPeerJsSignaling();
          }
        }, 1800);
      }
    };
  }

  private ensureGuestPeerJsOfferRetryLoop(): void {
    if (
      this.disposed ||
      this.terminalRoomError ||
      this.role !== 'guest' ||
      this.ctrlChannel?.readyState === 'open' ||
      !this.canUpgradeWebRtc() ||
      !this.isPeerJsAllowed()
    ) {
      return;
    }
    if (
      !this.peerJsWs ||
      this.peerJsWs.readyState === WebSocket.CLOSED ||
      this.peerJsWs.readyState === WebSocket.CLOSING
    ) {
      this.connectPeerJsSignaling();
      return;
    }
    if (this.peerJsOfferRetryTimer !== null) {
      clearInterval(this.peerJsOfferRetryTimer);
    }
    const retryCadenceMs = this.getPreferredFallbackRelayUrl()
      ? Math.max(this.webrtcHandshakeTimeoutMs + 400, 2600)
      : this.hadPeerConnectionOnce
        ? 850
        : 2200;
    this.peerJsOfferRetryTimer = setInterval(() => {
      if (
        this.disposed ||
        this.terminalRoomError ||
        this.ctrlChannel?.readyState === 'open' ||
        !this.canUpgradeWebRtc()
      ) {
        if (this.peerJsOfferRetryTimer !== null) {
          clearInterval(this.peerJsOfferRetryTimer);
          this.peerJsOfferRetryTimer = null;
        }
        return;
      }
      if (this.peerJsOpen) {
        void this.initiateWebRtcOffer('peerjs');
      }
    }, retryCadenceMs);
  }

  private handlePeerJsSignalMessage(msg: PeerJsWireMessage): void {
    switch (msg.type) {
      case 'OPEN': {
        this.peerJsOpen = true;
        this.hostPeerJsIdTakenRetries = 0;
        this.updateStatus();
        if (
          this.role === 'guest' &&
          this.ctrlChannel?.readyState !== 'open' &&
          this.canUpgradeWebRtc() &&
          !this.terminalRoomError
        ) {
          void this.initiateWebRtcOffer('peerjs');
          this.ensureGuestPeerJsOfferRetryLoop();
        }
        break;
      }
      case 'ID-TAKEN': {
        if (this.role === 'host') {
          if (
            this.options.isSessionReclaim &&
            !this.terminalRoomError &&
            this.hostPeerJsIdTakenRetries < 5
          ) {
            this.hostPeerJsIdTakenRetries++;
            this.cleanupPeerJsSocket();
            this.peerJsReconnectTimer = setTimeout(() => {
              this.peerJsReconnectTimer = null;
              if (!this.disposed && !this.terminalRoomError) {
                this.connectPeerJsSignaling();
              }
            }, 350);
            break;
          }
          this.terminalRoomError = true;
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
        if (this.role === 'host' && msg.src && msg.payload?.sdp && this.canUpgradeWebRtc()) {
          const incomingClientId = msg.payload?.metadata?.clientId?.trim() || null;
          const hasActiveBoundGuest =
            this.peerConnected &&
            (this.ctrlChannel?.readyState === 'open' || this.relayPeerPresent);
          const isSameBoundGuest =
            (incomingClientId !== null &&
              this.remoteClientId !== null &&
              incomingClientId === this.remoteClientId) ||
            (incomingClientId === null &&
              this.remoteGuestPeerId !== null &&
              msg.src === this.remoteGuestPeerId);
          if (hasActiveBoundGuest && !isSameBoundGuest) {
            // Reject 3rd-peer WebRTC OFFER without disturbing the connected Guest 1 session.
            // Send both `LEAVE` (custom PeerJS servers) AND a schema-valid `CANDIDATE` sentinel
            // because public `0.peerjs.com` drops custom `LEAVE` payloads while forwarding `CANDIDATE` verbatim.
            if (this.peerJsWs?.readyState === WebSocket.OPEN) {
              this.peerJsWs.send(
                JSON.stringify({
                  type: 'CANDIDATE',
                  dst: msg.src,
                  payload: {
                    connectionId: msg.payload.connectionId ?? 'dc_reject',
                    type: 'data',
                    candidate: {
                      candidate: 'vitrail-reject:1008:Room full (2/2 players)',
                      sdpMid: '0',
                      sdpMLineIndex: 0,
                    },
                  },
                })
              );
              this.peerJsWs.send(
                JSON.stringify({
                  type: 'LEAVE',
                  dst: msg.src,
                  payload: {
                    code: 1008,
                    reason: 'room-full',
                    msg: 'Room full (2/2 players)',
                  },
                })
              );
            }
            break;
          }
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
        const candStr = msg.payload?.candidate?.candidate;
        if (
          this.role === 'guest' &&
          typeof candStr === 'string' &&
          candStr.startsWith('vitrail-reject:1008:')
        ) {
          this.terminalRoomError = true;
          this.errorMessage =
            candStr.slice('vitrail-reject:1008:'.length).trim() || 'Room full (2/2 players)';
          this.closePeerConnection();
          this.cleanupPeerJsSocket();
          this.updateStatus('error');
          break;
        }
        if (msg.payload?.candidate) {
          void this.addRemoteIceCandidate(msg.payload.candidate);
        }
        break;
      }
      case 'EXPIRE': {
        // Host has not opened room yet or hasn't answered yet; retry or fallback deadline handles cleanly
        break;
      }
      case 'LEAVE': {
        if (
          this.role === 'guest' &&
          msg.src === this.hostTargetPeerId &&
          (!this.hadPeerConnectionOnce ||
            msg.payload?.code === 1008 ||
            msg.payload?.reason === 'room-full')
        ) {
          this.terminalRoomError = true;
          this.errorMessage = msg.payload?.msg ?? 'Room full (2/2 players)';
          this.closePeerConnection();
          this.cleanupRelaySocket();
          this.cleanupPeerJsSocket();
          this.updateStatus('error');
          break;
        }
        if (msg.src && (msg.src === this.remoteGuestPeerId || msg.src === this.hostTargetPeerId)) {
          this.handleWebRtcDisconnectOrFailover('peerjs-leave');
        }
        break;
      }
      case 'ERROR': {
        if (this.getPreferredFallbackRelayUrl()) {
          this.handleWebRtcDisconnectOrFailover('peerjs-error');
        } else {
          this.errorMessage = msg.payload?.msg ?? 'Signaling error';
          this.updateStatus('error');
        }
        break;
      }
    }
  }

  private handleWebRtcDisconnectOrFailover(reason: string): void {
    if (this.disposed || this.terminalRoomError) return;
    this.lastFailoverReason = reason;
    if (
      reason.startsWith('ice-') ||
      reason.startsWith('pc-') ||
      reason.includes('timeout') ||
      reason.includes('error') ||
      reason.includes('close') ||
      reason.includes('throw')
    ) {
      if (this.getPreferredFallbackRelayUrl() || this.relayConnected) {
        this.webrtcUpgradeSuppressed = true;
      }
    }
    if (this.peerJsOfferRetryTimer !== null) {
      clearInterval(this.peerJsOfferRetryTimer);
      this.peerJsOfferRetryTimer = null;
    }
    if (this.peerJsFallbackDeadlineTimer !== null) {
      clearTimeout(this.peerJsFallbackDeadlineTimer);
      this.peerJsFallbackDeadlineTimer = null;
    }
    this.closePeerConnection();
    this.ensureRelaySocketConnectedImmediate();
    this.updateStatus();
    if (
      this.role === 'guest' &&
      !this.getPreferredFallbackRelayUrl() &&
      !this.forcedRelayMode &&
      !this.webrtcUpgradeSuppressed
    ) {
      this.ensureGuestPeerJsOfferRetryLoop();
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
      iceServers: this.resolvedIceServers,
    });
    this.pc = pc;

    // Enforce strict <= 3.2 s ICE/DataChannel open deadline so stalled symmetric-NAT handshakes
    // tear down `pc` and guarantee <= 4.0 s fallback to the WebSocket room relay.
    this.webrtcHandshakeTimer = setTimeout(() => {
      if (this.disposed || this.pc !== pc) return;
      this.webrtcHandshakeTimer = null;
      if (this.ctrlChannel?.readyState !== 'open') {
        this.handleWebRtcDisconnectOrFailover('ice-handshake-timeout');
      }
    }, this.webrtcHandshakeTimeoutMs);

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
      if (this.webrtcHandshakeTimer !== null) {
        clearTimeout(this.webrtcHandshakeTimer);
        this.webrtcHandshakeTimer = null;
      }
      if (this.peerJsFallbackDeadlineTimer !== null) {
        clearTimeout(this.peerJsFallbackDeadlineTimer);
        this.peerJsFallbackDeadlineTimer = null;
      }
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
      if (this.disposed || this.ctrlChannel !== ctrl) return;
      this.handleWebRtcDisconnectOrFailover('ctrl-close');
    };

    ctrl.onerror = () => {
      if (this.disposed || this.ctrlChannel !== ctrl) return;
      this.handleWebRtcDisconnectOrFailover('ctrl-error');
    };

    state.onmessage = (ev: MessageEvent) => {
      if (this.disposed || this.stateChannel !== state) return;
      this.handleIncomingTransportPayload(ev.data, 'webrtc');
    };

    state.onclose = () => {
      if (this.disposed || this.stateChannel !== state) return;
      this.handleWebRtcDisconnectOrFailover('state-close');
    };

    state.onerror = () => {
      if (this.disposed || this.stateChannel !== state) return;
      this.handleWebRtcDisconnectOrFailover('state-error');
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
        this.handleWebRtcDisconnectOrFailover(`pc-${pc.connectionState}`);
      }
    };

    pc.oniceconnectionstatechange = () => {
      if (this.disposed || this.pc !== pc) return;
      if (
        pc.iceConnectionState === 'failed' ||
        pc.iceConnectionState === 'disconnected' ||
        pc.iceConnectionState === 'closed'
      ) {
        this.handleWebRtcDisconnectOrFailover(`ice-${pc.iceConnectionState}`);
      }
    };

    return pc;
  }

  private async initiateWebRtcOffer(via: 'peerjs' | 'relay'): Promise<void> {
    if (this.disposed || !this.canUpgradeWebRtc()) return;
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
                label: 'ctrl',
                connectionId: this.activeConnectionId,
                type: 'data',
                reliable: true,
                serialization: 'binary',
                browser: 'Chrome',
                metadata: { clientId: this.clientId },
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
    if (this.disposed || !this.canUpgradeWebRtc()) return;
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
                browser: 'Chrome',
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
      if (parsed.from === this.role || !this.canUpgradeWebRtc()) return;
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
        if (
          this.role === 'guest' &&
          !this.getPreferredFallbackRelayUrl() &&
          !this.forcedRelayMode &&
          !this.webrtcUpgradeSuppressed
        ) {
          this.ensureGuestPeerJsOfferRetryLoop();
        }
        this.options.onRemotePeerBye?.(parsed.clientId);
      } else if (this.pcRemoteClientId === parsed.clientId) {
        this.closePeerConnection();
        this.updateStatus();
      }
      return;
    }

    if (parsed.type === 'hello') {
      if (parsed.role !== this.role) {
        const prevClientId = this.remoteClientId;
        if (prevClientId !== null && prevClientId !== parsed.clientId) {
          this.webrtcUpgradeSuppressed = false;
        }
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
        this.handleWebRtcDisconnectOrFailover('send-throw');
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
          this.handleWebRtcDisconnectOrFailover('send-throw');
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
    if (this.webrtcHandshakeTimer !== null) {
      clearTimeout(this.webrtcHandshakeTimer);
      this.webrtcHandshakeTimer = null;
    }
    if (this.ctrlChannel) {
      try {
        this.ctrlChannel.onopen = null;
        this.ctrlChannel.onmessage = null;
        this.ctrlChannel.onclose = null;
        this.ctrlChannel.onerror = null;
        this.ctrlChannel.close();
      } catch {
        // Ignore
      }
      this.ctrlChannel = null;
    }
    if (this.stateChannel) {
      try {
        this.stateChannel.onmessage = null;
        this.stateChannel.onclose = null;
        this.stateChannel.onerror = null;
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
    if (this.relayKeepaliveTimer !== null) {
      clearInterval(this.relayKeepaliveTimer);
      this.relayKeepaliveTimer = null;
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
    if (this.peerJsFallbackDeadlineTimer !== null) {
      clearTimeout(this.peerJsFallbackDeadlineTimer);
      this.peerJsFallbackDeadlineTimer = null;
    }
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

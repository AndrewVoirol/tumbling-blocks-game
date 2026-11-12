/**
 * Vitrail Real-Time Multiplayer Edge Room Relay (Cloudflare Workers + Hibernatable Durable Objects)
 *
 * Relays both 30 Hz `< 4 KB` binary `ArrayBuffer` physics/photoelastic snapshots and ordered JSON
 * control / WebRTC signaling envelopes between `host` and `guest` peers keyed by 5-char Room Code.
 */

export interface Env {
  VITRAIL_ROOMS: DurableObjectNamespace;
}

type Role = 'host' | 'guest';

interface SocketAttachment {
  role: Role;
  clientId: string;
  closed?: boolean;
}

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Upgrade, Connection',
};

const MP_PROTOCOL_MAGIC = 0x564d;
const MAX_BINARY_FRAME_BYTES = 16_384;
const MAX_TEXT_FRAME_BYTES = 524_288;
const utf8Encoder = new TextEncoder();

export function isAllowedRelayOrigin(originHeader: string | null | undefined): boolean {
  if (!originHeader) return true;
  const trimmed = originHeader.trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'null') return true;
  let host: string;
  try {
    host = new URL(trimmed).hostname.toLowerCase();
  } catch {
    return false;
  }
  if (host.startsWith('[') && host.endsWith(']')) {
    host = host.slice(1, -1);
  }
  if (
    host === 'vitrail.andrewvoirol.com' ||
    host === 'andrewvoirol.com' ||
    host.endsWith('.andrewvoirol.com') ||
    host.endsWith('.vercel.app') ||
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host === '::1' ||
    host.endsWith('.local')
  ) {
    return true;
  }
  const ipv4Match = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (ipv4Match) {
    const o1 = Number(ipv4Match[1]);
    const o2 = Number(ipv4Match[2]);
    const o3 = Number(ipv4Match[3]);
    const o4 = Number(ipv4Match[4]);
    if (o1 > 255 || o2 > 255 || o3 > 255 || o4 > 255) return false;
    if (o1 === 127 || o1 === 10) return true;
    if (o1 === 192 && o2 === 168) return true;
    if (o1 === 172 && o2 >= 16 && o2 <= 31) return true;
  }
  return false;
}

function normalizeCode(raw: string | null): string {
  return (raw ?? '')
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, '')
    .slice(0, 8);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    const url = new URL(request.url);
    if (url.pathname === '/health') {
      return Response.json({ ok: true, service: 'vitrail-rooms', protocol: 1 }, { headers: CORS_HEADERS });
    }
    if (url.pathname !== '/__vitrail_mp' && url.pathname !== '/') {
      return new Response('Not Found', { status: 404, headers: CORS_HEADERS });
    }
    if (!isAllowedRelayOrigin(request.headers.get('Origin'))) {
      return new Response('Forbidden origin', { status: 403, headers: CORS_HEADERS });
    }
    const room = normalizeCode(url.searchParams.get('room'));
    if (room.length < 4) {
      return Response.json({ error: 'Invalid room code' }, { status: 400, headers: CORS_HEADERS });
    }
    const stub = env.VITRAIL_ROOMS.get(env.VITRAIL_ROOMS.idFromName(`room:${room}`));
    return stub.fetch(request);
  },
};

export class VitrailRoomDurableObject {
  private readonly state: DurableObjectState;

  constructor(state: DurableObjectState) {
    this.state = state;
  }

  async fetch(request: Request): Promise<Response> {
    if (request.headers.get('Upgrade')?.toLowerCase() !== 'websocket') {
      const sockets = this.state.getWebSockets();
      return Response.json(
        { activePeers: sockets.length },
        { status: 200, headers: CORS_HEADERS }
      );
    }
    if (!isAllowedRelayOrigin(request.headers.get('Origin'))) {
      return new Response('Forbidden origin', { status: 403, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const role: Role = url.searchParams.get('role') === 'host' ? 'host' : 'guest';
    const clientId = (url.searchParams.get('clientId') ?? `${role}-anon`).slice(0, 48);

    const pair = new WebSocketPair();
    const [clientWs, serverWs] = [pair[0], pair[1]];

    const existingSameRole = this.state.getWebSockets(role).filter((sock) => {
      const meta = sock.deserializeAttachment() as SocketAttachment | null;
      const rs = (sock as unknown as { readyState?: number }).readyState;
      return !meta?.closed && rs !== 2 && rs !== 3;
    });
    if (existingSameRole.length > 0) {
      const prevMeta = existingSameRole[0].deserializeAttachment() as SocketAttachment | null;
      if (prevMeta && prevMeta.clientId !== clientId) {
        serverWs.accept();
        if (role === 'host') {
          serverWs.send(
            JSON.stringify({ __relay: 'error', message: 'Room already has an active Host.' })
          );
          serverWs.close(1008, 'Host slot occupied');
        } else {
          serverWs.send(
            JSON.stringify({ __relay: 'error', message: 'Room full (2/2 players)' })
          );
          serverWs.close(1008, 'Room full (2/2 players)');
        }
        return new Response(null, { status: 101, webSocket: clientWs });
      }
    }

    // Evict stale duplicate connection for the same role on same-clientId reconnect
    for (const oldWs of this.state.getWebSockets(role)) {
      try {
        const oldMeta = oldWs.deserializeAttachment() as SocketAttachment | null;
        if (oldMeta && !oldMeta.closed) {
          oldWs.serializeAttachment({ ...oldMeta, closed: true });
        }
        oldWs.close(1000, 'Replaced by reconnecting session');
      } catch {
        // Ignore
      }
    }

    const attachment: SocketAttachment = { role, clientId };
    this.state.acceptWebSocket(serverWs, [role]);
    serverWs.serializeAttachment(attachment);

    const hasHost = role === 'host' || this.state.getWebSockets('host').length > 0;
    const hasGuest = role === 'guest' || this.state.getWebSockets('guest').length > 0;

    serverWs.send(JSON.stringify({ __relay: 'room_state', hasHost, hasGuest }));

    const oppositeRole: Role = role === 'host' ? 'guest' : 'host';
    for (const peerWs of this.state.getWebSockets(oppositeRole)) {
      try {
        peerWs.send(JSON.stringify({ __relay: 'peer_joined', role }));
      } catch {
        // Ignore closed socket
      }
    }

    return new Response(null, { status: 101, webSocket: clientWs });
  }

  private closeRejectedSocket(
    ws: WebSocket,
    meta: SocketAttachment,
    code: number,
    reason: string
  ): void {
    if (!meta.closed) {
      ws.serializeAttachment({ ...meta, closed: true });
    }
    this.notifyDisconnect(ws, meta);
    try {
      ws.close(code, reason);
    } catch {
      // Ignore already-closed socket
    }
  }

  webSocketMessage(ws: WebSocket, message: string | ArrayBuffer): void {
    const meta = ws.deserializeAttachment() as SocketAttachment | null;
    if (!meta || meta.closed) return;

    if (typeof message === 'string') {
      if (message.length > MAX_TEXT_FRAME_BYTES) {
        this.closeRejectedSocket(ws, meta, 1009, 'Message too big');
        return;
      }
      if (message.length * 3 > MAX_TEXT_FRAME_BYTES) {
        if (utf8Encoder.encode(message).byteLength > MAX_TEXT_FRAME_BYTES) {
          this.closeRejectedSocket(ws, meta, 1009, 'Message too big');
          return;
        }
      }
      if (!message.trimStart().startsWith('{')) {
        this.closeRejectedSocket(ws, meta, 1003, 'Unsupported text payload');
        return;
      }
      if (message.includes('"__relay"')) {
        try {
          const parsed = JSON.parse(message) as { __relay?: string };
          if (parsed?.__relay === 'keepalive') {
            return;
          }
        } catch {
          // Forward regular JSON/text frames unchanged
        }
      }
    } else {
      const bytes = ArrayBuffer.isView(message)
        ? new Uint8Array(message.buffer, message.byteOffset, message.byteLength)
        : new Uint8Array(message);
      if (bytes.byteLength > MAX_BINARY_FRAME_BYTES) {
        this.closeRejectedSocket(ws, meta, 1009, 'Message too big');
        return;
      }
      if (bytes.byteLength < 2 || (bytes[0] | (bytes[1] << 8)) !== MP_PROTOCOL_MAGIC) {
        this.closeRejectedSocket(ws, meta, 1003, 'Unsupported binary payload');
        return;
      }
    }

    const targetRole: Role = meta.role === 'host' ? 'guest' : 'host';
    for (const peerWs of this.state.getWebSockets(targetRole)) {
      const peerMeta = peerWs.deserializeAttachment() as SocketAttachment | null;
      if (peerMeta?.closed) continue;
      try {
        peerWs.send(message);
      } catch {
        // Ignore closed socket
      }
    }
  }

  webSocketClose(ws: WebSocket, code: number): void {
    const meta = ws.deserializeAttachment() as SocketAttachment | null;
    if (meta && !meta.closed) {
      ws.serializeAttachment({ ...meta, closed: true });
    }
    this.notifyDisconnect(ws, meta);
    const safeCode =
      typeof code === 'number' && code >= 1000 && code <= 4999 && code !== 1005 && code !== 1006
        ? code
        : 1000;
    try {
      ws.close(safeCode, 'Peer closed');
    } catch {
      // Ignore already-closed socket
    }
  }

  webSocketError(ws: WebSocket): void {
    const meta = ws.deserializeAttachment() as SocketAttachment | null;
    if (meta && !meta.closed) {
      ws.serializeAttachment({ ...meta, closed: true });
    }
    this.notifyDisconnect(ws, meta);
    try {
      ws.close(1011, 'Peer socket error');
    } catch {
      // Ignore
    }
  }

  private notifyDisconnect(ws: WebSocket, cachedMeta?: SocketAttachment | null): void {
    const meta = cachedMeta ?? (ws.deserializeAttachment() as SocketAttachment | null);
    if (!meta) return;
    const stillHasSameRole = this.state.getWebSockets(meta.role).some((candidate) => {
      if (candidate === ws) return false;
      const cMeta = candidate.deserializeAttachment() as SocketAttachment | null;
      const rs = (candidate as unknown as { readyState?: number }).readyState;
      return !cMeta?.closed && rs !== 2 && rs !== 3;
    });
    if (stillHasSameRole) return;

    const targetRole: Role = meta.role === 'host' ? 'guest' : 'host';
    for (const peerWs of this.state.getWebSockets(targetRole)) {
      const peerMeta = peerWs.deserializeAttachment() as SocketAttachment | null;
      if (peerMeta?.closed) continue;
      try {
        peerWs.send(JSON.stringify({ __relay: 'peer_left', role: meta.role }));
      } catch {
        // Ignore
      }
    }
  }
}

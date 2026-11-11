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
}

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Upgrade, Connection',
};

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

    const url = new URL(request.url);
    const role: Role = url.searchParams.get('role') === 'host' ? 'host' : 'guest';
    const clientId = (url.searchParams.get('clientId') ?? `${role}-anon`).slice(0, 48);

    const pair = new WebSocketPair();
    const [clientWs, serverWs] = [pair[0], pair[1]];

    const existingSameRole = this.state.getWebSockets(role);
    if (role === 'host' && existingSameRole.length > 0) {
      const prevMeta = existingSameRole[0].deserializeAttachment() as SocketAttachment | null;
      if (prevMeta && prevMeta.clientId !== clientId) {
        serverWs.accept();
        serverWs.send(
          JSON.stringify({ __relay: 'error', message: 'Room already has an active Host.' })
        );
        serverWs.close(1008, 'Host slot occupied');
        return new Response(null, { status: 101, webSocket: clientWs });
      }
    }

    // Evict stale duplicate connection for the same role on reconnect
    for (const oldWs of existingSameRole) {
      try {
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

  webSocketMessage(ws: WebSocket, message: string | ArrayBuffer): void {
    const meta = ws.deserializeAttachment() as SocketAttachment | null;
    if (!meta) return;
    const targetRole: Role = meta.role === 'host' ? 'guest' : 'host';
    for (const peerWs of this.state.getWebSockets(targetRole)) {
      try {
        peerWs.send(message);
      } catch {
        // Ignore closed socket
      }
    }
  }

  webSocketClose(ws: WebSocket): void {
    this.notifyDisconnect(ws);
  }

  webSocketError(ws: WebSocket): void {
    this.notifyDisconnect(ws);
  }

  private notifyDisconnect(ws: WebSocket): void {
    const meta = ws.deserializeAttachment() as SocketAttachment | null;
    if (!meta) return;
    const stillHasSameRole = this.state
      .getWebSockets(meta.role)
      .some((candidate) => candidate !== ws);
    if (stillHasSameRole) return;

    const targetRole: Role = meta.role === 'host' ? 'guest' : 'host';
    for (const peerWs of this.state.getWebSockets(targetRole)) {
      try {
        peerWs.send(JSON.stringify({ __relay: 'peer_left', role: meta.role }));
      } catch {
        // Ignore
      }
    }
  }
}

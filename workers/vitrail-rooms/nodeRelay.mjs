import { createHash } from 'node:crypto';

const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const MAX_FRAME_BYTES = 1048576; // 1 MB outer wire-frame safety cap
const MP_PROTOCOL_MAGIC = 0x564d;
const MAX_BINARY_FRAME_BYTES = 16_384;
const MAX_TEXT_FRAME_BYTES = 524_288;

export function isAllowedRelayOrigin(originHeader) {
  if (!originHeader) return true;
  const trimmed = String(originHeader).trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'null') return true;
  let host;
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

function encodeWsFrame(opcode, payloadBuf) {
  const len = payloadBuf.length;
  let header;
  if (len < 126) {
    header = Buffer.allocUnsafe(2);
    header[0] = 0x80 | opcode;
    header[1] = len;
  } else if (len < 65536) {
    header = Buffer.allocUnsafe(4);
    header[0] = 0x80 | opcode;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.allocUnsafe(10);
    header[0] = 0x80 | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return len === 0 ? header : Buffer.concat([header, payloadBuf]);
}

function createWsPeerConnection(socket, head, onMessage, onClose) {
  let rxBuf = head && head.length > 0 ? Buffer.from(head) : Buffer.alloc(0);
  let fragOpcode = 0;
  let fragChunks = [];
  let closed = false;

  const conn = {
    isOpen: true,
    sendText(str) {
      if (closed || socket.destroyed) return;
      socket.write(encodeWsFrame(0x01, Buffer.from(str, 'utf8')));
    },
    sendBinary(buf) {
      if (closed || socket.destroyed) return;
      socket.write(encodeWsFrame(0x02, buf));
    },
    close(code = 1000, reason = '') {
      if (closed) return;
      closed = true;
      conn.isOpen = false;
      if (!socket.destroyed) {
        const reasonBytes = Buffer.from(reason, 'utf8').subarray(0, 64);
        const closePayload = Buffer.allocUnsafe(2 + reasonBytes.length);
        closePayload.writeUInt16BE(code, 0);
        reasonBytes.copy(closePayload, 2);
        try {
          socket.write(encodeWsFrame(0x08, closePayload));
          socket.end();
        } catch {
          socket.destroy();
        }
      }
      onClose();
    },
  };

  const pumpFrames = () => {
    while (!closed && rxBuf.length >= 2) {
      const b0 = rxBuf[0];
      const b1 = rxBuf[1];
      const fin = (b0 & 0x80) !== 0;
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      const len7 = b1 & 0x7f;

      let headerBytes = 2;
      if (len7 === 126) headerBytes += 2;
      else if (len7 === 127) headerBytes += 8;
      if (masked) headerBytes += 4;
      if (rxBuf.length < headerBytes) return;

      let payloadLen = len7;
      let cursor = 2;
      if (len7 === 126) {
        payloadLen = rxBuf.readUInt16BE(2);
        cursor = 4;
      } else if (len7 === 127) {
        payloadLen = Number(rxBuf.readBigUInt64BE(2));
        cursor = 10;
      }

      if (payloadLen > MAX_FRAME_BYTES) {
        conn.close(1009, 'Frame too large');
        return;
      }
      if (rxBuf.length < headerBytes + payloadLen) return;

      let payload;
      if (masked) {
        const m0 = rxBuf[cursor + 0];
        const m1 = rxBuf[cursor + 1];
        const m2 = rxBuf[cursor + 2];
        const m3 = rxBuf[cursor + 3];
        const srcOff = cursor + 4;
        payload = Buffer.allocUnsafe(payloadLen);
        for (let i = 0; i < payloadLen; i++) {
          const maskByte = (i & 3) === 0 ? m0 : (i & 3) === 1 ? m1 : (i & 3) === 2 ? m2 : m3;
          payload[i] = rxBuf[srcOff + i] ^ maskByte;
        }
      } else {
        payload = Buffer.from(rxBuf.subarray(cursor, cursor + payloadLen));
      }

      rxBuf = rxBuf.subarray(headerBytes + payloadLen);

      if (opcode === 0x08) {
        const rxCode = payload.length >= 2 ? payload.readUInt16BE(0) : 1000;
        const safeCode =
          rxCode >= 1000 && rxCode <= 4999 && rxCode !== 1005 && rxCode !== 1006 ? rxCode : 1000;
        conn.close(safeCode, 'Peer closed');
        return;
      }
      if (opcode === 0x09) {
        if (!closed && !socket.destroyed) {
          socket.write(encodeWsFrame(0x0a, payload));
        }
        continue;
      }
      if (opcode === 0x0a) {
        continue;
      }

      if (opcode === 0x01 || opcode === 0x02) {
        if (fin) {
          onMessage(opcode, payload);
        } else {
          fragOpcode = opcode;
          fragChunks = [payload];
        }
      } else if (opcode === 0x00 && fragOpcode !== 0) {
        fragChunks.push(payload);
        if (fin) {
          const full = Buffer.concat(fragChunks);
          const doneOp = fragOpcode;
          fragOpcode = 0;
          fragChunks = [];
          onMessage(doneOp, full);
        }
      }
    }
  };

  socket.on('data', (chunk) => {
    if (closed) return;
    rxBuf = rxBuf.length === 0 ? chunk : Buffer.concat([rxBuf, chunk]);
    pumpFrames();
  });

  socket.on('close', () => {
    if (!closed) {
      closed = true;
      conn.isOpen = false;
      onClose();
    }
  });

  socket.on('error', () => {
    if (!closed) {
      closed = true;
      conn.isOpen = false;
      onClose();
    }
  });

  if (rxBuf.length > 0) {
    queueMicrotask(pumpFrames);
  }

  return conn;
}

/**
 * Attaches the zero-dependency RFC 6455 `/__vitrail_mp` WebSocket Room Relay onto a Node HTTP server.
 * Used synchronously by Vite `dev` & `preview` servers and by `tests/multiplayer.test.ts`.
 */
export function attachVitrailRoomRelay(httpServer, pathPrefix = '/__vitrail_mp') {
  const rooms = new Map();

  const onUpgrade = (req, socket, head) => {
    let url;
    try {
      url = new URL(req.url || '/', 'http://localhost');
    } catch {
      return;
    }
    if (url.pathname !== pathPrefix) {
      return; // Leave Vite HMR and any other upgrade paths untouched
    }

    const rawOrigin = req.headers['origin'];
    if (!isAllowedRelayOrigin(typeof rawOrigin === 'string' ? rawOrigin : null)) {
      socket.write(
        'HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 16\r\n\r\nForbidden origin'
      );
      socket.destroy();
      return;
    }

    const wsKey = req.headers['sec-websocket-key'];
    if (!wsKey || typeof wsKey !== 'string') {
      socket.write('HTTP/1.1 400 Bad Request\r\n\r\n');
      socket.destroy();
      return;
    }

    const roomCode = (url.searchParams.get('room') || '')
      .toUpperCase()
      .replace(/[^A-Z0-9]/g, '')
      .slice(0, 8);
    if (roomCode.length < 4) {
      socket.write('HTTP/1.1 400 Bad Request\r\n\r\n');
      socket.destroy();
      return;
    }

    const role = url.searchParams.get('role') === 'host' ? 'host' : 'guest';
    const clientId = (url.searchParams.get('clientId') || `${role}-anon`).slice(0, 48);

    const acceptKey = createHash('sha1').update(wsKey + WS_GUID).digest('base64');
    socket.setNoDelay(true);
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
        'Upgrade: websocket\r\n' +
        'Connection: Upgrade\r\n' +
        `Sec-WebSocket-Accept: ${acceptKey}\r\n\r\n`
    );

    let room = rooms.get(roomCode);
    if (!room) {
      room = { host: null, guest: null };
      rooms.set(roomCode, room);
    }

    if (role === 'host' && room.host && room.host.conn.isOpen && room.host.clientId !== clientId) {
      const rejectConn = createWsPeerConnection(socket, head, () => {}, () => {});
      rejectConn.sendText(
        JSON.stringify({ __relay: 'error', message: `Room ${roomCode} already has an active Host.` })
      );
      rejectConn.close(1008, 'Host occupied');
      return;
    }

    if (role === 'guest' && room.guest && room.guest.conn.isOpen && room.guest.clientId !== clientId) {
      const rejectConn = createWsPeerConnection(socket, head, () => {}, () => {});
      rejectConn.sendText(
        JSON.stringify({ __relay: 'error', message: 'Room full (2/2 players)' })
      );
      rejectConn.close(1008, 'Room full (2/2 players)');
      return;
    }

    const prevSlot = room[role];
    if (prevSlot && prevSlot.conn.isOpen) {
      room[role] = null;
      prevSlot.conn.close(1000, 'Replaced');
    }

    let slotEntry = null;
    const conn = createWsPeerConnection(
      socket,
      head,
      (opcode, payload) => {
        if (opcode === 0x01) {
          if (payload.length > MAX_TEXT_FRAME_BYTES) {
            conn.close(1009, 'Message too big');
            return;
          }
          const text = payload.toString('utf8');
          if (!text.trimStart().startsWith('{')) {
            conn.close(1003, 'Unsupported text payload');
            return;
          }
          if (text.includes('"__relay"')) {
            try {
              const parsed = JSON.parse(text);
              if (parsed && parsed.__relay === 'keepalive') {
                return;
              }
            } catch {
              // Forward regular text frames
            }
          }
          const currentRoom = rooms.get(roomCode);
          if (!currentRoom) return;
          const peerSlot = role === 'host' ? currentRoom.guest : currentRoom.host;
          if (!peerSlot || !peerSlot.conn.isOpen) return;
          peerSlot.conn.sendText(text);
          return;
        }
        if (opcode === 0x02) {
          if (payload.length > MAX_BINARY_FRAME_BYTES) {
            conn.close(1009, 'Message too big');
            return;
          }
          if (payload.length < 2 || payload.readUInt16LE(0) !== MP_PROTOCOL_MAGIC) {
            conn.close(1003, 'Unsupported binary payload');
            return;
          }
          const currentRoom = rooms.get(roomCode);
          if (!currentRoom) return;
          const peerSlot = role === 'host' ? currentRoom.guest : currentRoom.host;
          if (!peerSlot || !peerSlot.conn.isOpen) return;
          peerSlot.conn.sendBinary(payload);
        }
      },
      () => {
        const currentRoom = rooms.get(roomCode);
        if (!currentRoom) return;
        if (currentRoom[role] === slotEntry) {
          currentRoom[role] = null;
          const peerSlot = role === 'host' ? currentRoom.guest : currentRoom.host;
          if (peerSlot && peerSlot.conn.isOpen) {
            peerSlot.conn.sendText(JSON.stringify({ __relay: 'peer_left', role }));
          }
          if (!currentRoom.host && !currentRoom.guest) {
            rooms.delete(roomCode);
          }
        }
      }
    );

    slotEntry = { role, clientId, conn };
    room[role] = slotEntry;

    conn.sendText(
      JSON.stringify({
        __relay: 'room_state',
        hasHost: Boolean(room.host?.conn.isOpen),
        hasGuest: Boolean(room.guest?.conn.isOpen),
      })
    );

    const opposite = role === 'host' ? room.guest : room.host;
    if (opposite && opposite.conn.isOpen) {
      opposite.conn.sendText(JSON.stringify({ __relay: 'peer_joined', role }));
    }
  };

  httpServer.on('upgrade', onUpgrade);

  return {
    rooms,
    detach() {
      httpServer.off('upgrade', onUpgrade);
      for (const room of rooms.values()) {
        room.host?.conn.close(1001, 'Server shutting down');
        room.guest?.conn.close(1001, 'Server shutting down');
      }
      rooms.clear();
    },
  };
}

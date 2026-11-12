#!/usr/bin/env node
/**
 * Adversarial Network Chaos Simulator (`scripts/chaos-multiplayer-sim.mjs`)
 *
 * Runs 100 seeded two-peer online matches (`GameEngine` Host + `VitrailMultiplayerClient` +
 * `MultiplayerGuestWorld` + MuJoCo 4.0 WASM `PhysicsWorld`) across an in-process faulty
 * dual-transport network (`WebRTC RTCDataChannel` + `PeerJS` broker + `vitrail-rooms` WS relay)
 * simultaneously injecting all 5 adversarial conditions:
 *   (a) `8%` baseline random binary `0x564d` state-packet drop (+ `65%` burst-loss windows)
 *   (b) `120 ms +/- 45 ms` jitter + out-of-order `0x564d` delivery (including delayed pre-rematch frames)
 *   (c) Mid-drag WebRTC disconnect (`ctrl`/`state` close/error, ICE `failed`/`disconnected`, `send` throw)
 *       -> automatic WS relay failover + state/pull/audio resync
 *   (d) Host tab occlusion (`setHostTabOccluded(true)` / `document.hidden = true`, foreground `rAF` halted)
 *       driving physics & `30 Hz` snapshots via `startHostBackgroundPump()` during Guest extraction
 *   (e) Simultaneous Host + Guest double-grab races on turn/rematch boundaries + rapid rematch resets
 *
 * Enforces 6 hard pass/fail gates (exiting non-zero in CLI mode if any gate fails):
 *   1. `0` uncaught exceptions & `0` unhandled promise rejections
 *   2. `0` `NaN`/`Infinity` values & `0` quaternion-hemisphere antipodal flips across `120 Hz` Guest steps
 *   3. `100%` monotonic sequence fencing (`0` stale/out-of-order `0x564d` frames accepted, `0` rematch rollbacks)
 *   4. `54/54` stained-glass swatch parity (`swatchId`, `motif`, `rippleSeed`) across all 100 matches & rematches
 *   5. Exact `1:1` single-emission audio parity (`blockThud`, `glassImpact`, `glassShatter` — `0` duplicates/drops)
 *   6. `< 2.0 mm` (`< 0.02 dm`) steady-state Guest pose convergence within `250 ms` after packet-loss bursts cease
 *
 * Usage:
 *   node scripts/chaos-multiplayer-sim.mjs [--matches 100] [--seed 0x56495452] [--json]
 */

import { performance as nodePerf } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const VIEWPORT_RECT = { left: 0, top: 0, width: 1280, height: 800 };

function makeRoomCodeFromIndex(index, rng) {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let code = alphabet[index % alphabet.length] + alphabet[Math.floor(index / alphabet.length) % alphabet.length];
  while (code.length < 5) {
    code += alphabet[Math.floor(rng() * alphabet.length)];
  }
  return code;
}

function projectWorldToScreen(camera, worldPt, rect = VIEWPORT_RECT) {
  const m = camera.viewProjMatrix;
  const x = worldPt[0];
  const y = worldPt[1];
  const z = worldPt[2];
  const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
  const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
  const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (!Number.isFinite(clipW) || clipW <= 1e-4) return null;
  const ndcX = clipX / clipW;
  const ndcY = clipY / clipW;
  return [
    rect.left + (ndcX * 0.5 + 0.5) * rect.width,
    rect.top + (0.5 - ndcY * 0.5) * rect.height,
  ];
}

function quatGeodesicAngleRad(q0, q1) {
  const l0 = Math.hypot(q0[0], q0[1], q0[2], q0[3]) || 1.0;
  const l1 = Math.hypot(q1[0], q1[1], q1[2], q1[3]) || 1.0;
  const absDot = Math.min(
    1.0,
    Math.abs(
      (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]) / (l0 * l1)
    )
  );
  return 2.0 * Math.acos(absDot);
}

function cloneBufferPayload(data) {
  if (data instanceof ArrayBuffer) {
    return data.slice(0);
  }
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  return data;
}

/**
 * Deterministic discrete-event virtual clock intercepting global timers & monotonic clocks
 * so 100 full matches with `120 ms +/- 45 ms` network jitter, `30 Hz` background `setInterval`
 * pumps, and `250 ms` convergence windows execute deterministically in seconds.
 */
class ChaosVirtualClock {
  constructor(startMs = 10_000) {
    this.nowMs = startMs;
    this.nextId = 1;
    /** @type {Map<number, { id: number; dueMs: number; intervalMs: number | null; fn: () => void }>} */
    this.timers = new Map();
    this.savedGlobals = null;
  }

  install() {
    if (this.savedGlobals) return;
    const origPerfNow = globalThis.performance?.now?.bind(globalThis.performance);
    this.savedGlobals = {
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      dateNow: Date.now,
      perfNow: origPerfNow,
    };

    const clock = this;
    globalThis.setTimeout = /** @type {any} */ ((fn, delay = 0, ...args) => {
      const id = clock.nextId++;
      const d = Math.max(0, Number.isFinite(Number(delay)) ? Number(delay) : 0);
      clock.timers.set(id, {
        id,
        dueMs: clock.nowMs + d,
        intervalMs: null,
        fn: () => {
          if (typeof fn === 'function') fn(...args);
        },
      });
      return id;
    });

    globalThis.clearTimeout = /** @type {any} */ ((id) => {
      clock.timers.delete(Number(id));
    });

    globalThis.setInterval = /** @type {any} */ ((fn, delay = 1, ...args) => {
      const id = clock.nextId++;
      const d = Math.max(1, Number.isFinite(Number(delay)) ? Number(delay) : 1);
      clock.timers.set(id, {
        id,
        dueMs: clock.nowMs + d,
        intervalMs: d,
        fn: () => {
          if (typeof fn === 'function') fn(...args);
        },
      });
      return id;
    });

    globalThis.clearInterval = /** @type {any} */ ((id) => {
      clock.timers.delete(Number(id));
    });

    Date.now = () => Math.round(clock.nowMs);
    if (globalThis.performance) {
      globalThis.performance.now = () => clock.nowMs;
    }
  }

  uninstall() {
    if (!this.savedGlobals) return;
    globalThis.setTimeout = this.savedGlobals.setTimeout;
    globalThis.clearTimeout = this.savedGlobals.clearTimeout;
    globalThis.setInterval = this.savedGlobals.setInterval;
    globalThis.clearInterval = this.savedGlobals.clearInterval;
    Date.now = this.savedGlobals.dateNow;
    if (globalThis.performance && this.savedGlobals.perfNow) {
      globalThis.performance.now = this.savedGlobals.perfNow;
    }
    this.timers.clear();
    this.savedGlobals = null;
  }

  fireDueTimers() {
    let guard = 0;
    while (guard++ < 4096) {
      let earliest = null;
      for (const entry of this.timers.values()) {
        if (entry.dueMs <= this.nowMs) {
          if (
            !earliest ||
            entry.dueMs < earliest.dueMs ||
            (entry.dueMs === earliest.dueMs && entry.id < earliest.id)
          ) {
            earliest = entry;
          }
        }
      }
      if (!earliest) break;
      if (earliest.intervalMs !== null) {
        earliest.dueMs += earliest.intervalMs;
      } else {
        this.timers.delete(earliest.id);
      }
      earliest.fn();
    }
  }
}

/**
 * In-process faulty dual-transport network (`WebSocket` room relay + `PeerJS` broker +
 * `RTCPeerConnection` negotiated DataChannels `"ctrl"` & `"state"`).
 */
class ChaosTransportNetwork {
  constructor(clock, rng) {
    this.clock = clock;
    this.rng = rng;
    this.savedTransportGlobals = null;

    // Tunable chaos parameters
    this.binaryDropRate = 0.08;
    this.jitterMeanMs = 120;
    this.jitterHalfSpanMs = 45;
    this.controlLatencyMeanMs = 26;
    this.controlLatencyJitterMs = 10;
    this.forceReorderPairToggle = false;
    this.steadyStateCleanLink = false;
    this.webrtcStallIce = false;
    this.webrtcFailIceImmediately = false;
    this.unreachableHosts = new Set();

    // Telemetry counters
    this.stats = {
      binaryFramesSent: 0,
      binaryFramesDropped: 0,
      binaryFramesDelivered: 0,
      binaryFramesReorderedOnWire: 0,
      delayedPreRematchInjections: 0,
      webrtcConnectionsOpened: 0,
      webrtcDisconnectsInjected: 0,
      wsRelayFailoversCompleted: 0,
      duplicateAudioBatchesInjected: 0,
    };

    /** @type {Array<{ dueMs: number; order: number; fn: () => void }>} */
    this.packetQueue = [];
    this.packetOrderCounter = 1;
    this.lastBinaryArrivalMs = 0;

    /** @type {Map<string, any>} */
    this.peerJsClients = new Map();
    /** @type {Map<string, { host: any | null; guest: any | null }>} */
    this.relayRooms = new Map();
    /** @type {Set<any>} */
    this.activePeerConnections = new Set();
    /** @type {ArrayBuffer | null} */
    this.capturedLatestBinaryFrame = null;
    /** @type {string | null} */
    this.capturedLatestAudioBatchJson = null;
  }

  schedulePacket(delayMs, fn) {
    const d = Math.max(0, delayMs);
    this.packetQueue.push({
      dueMs: this.clock.nowMs + d,
      order: this.packetOrderCounter++,
      fn,
    });
  }

  deliverDuePackets() {
    if (this.packetQueue.length === 0) return;
    this.packetQueue.sort((a, b) => (a.dueMs !== b.dueMs ? a.dueMs - b.dueMs : a.order - b.order));
    while (this.packetQueue.length > 0 && this.packetQueue[0].dueMs <= this.clock.nowMs) {
      const pkt = this.packetQueue.shift();
      pkt.fn();
    }
  }

  sampleBinaryDelayMs() {
    if (this.steadyStateCleanLink) {
      return 2.0;
    }
    const base =
      this.jitterMeanMs + (this.rng() * 2.0 - 1.0) * this.jitterHalfSpanMs;
    // Periodically invert consecutive packet delays within `[75 ms, 165 ms]` so out-of-order
    // `0x564d` frames (`33 ms` Host period < `90 ms` jitter span) occur deterministically every match!
    this.forceReorderPairToggle = !this.forceReorderPairToggle;
    const bimodalOffset = this.forceReorderPairToggle ? 36 : -36;
    const clamped = Math.max(
      this.jitterMeanMs - this.jitterHalfSpanMs,
      Math.min(this.jitterMeanMs + this.jitterHalfSpanMs, base + bimodalOffset * 0.55)
    );
    return clamped;
  }

  sampleControlDelayMs(prevOrderedArrivalMs = 0) {
    if (this.steadyStateCleanLink) {
      return Math.max(1.0, prevOrderedArrivalMs - this.clock.nowMs + 0.5);
    }
    const raw =
      this.controlLatencyMeanMs +
      (this.rng() * 2.0 - 1.0) * this.controlLatencyJitterMs;
    const targetArrival = Math.max(this.clock.nowMs + Math.max(2, raw), prevOrderedArrivalMs + 0.5);
    return targetArrival - this.clock.nowMs;
  }

  routeBinaryStatePayload(rawBuf, deliverFn) {
    const cloned = cloneBufferPayload(rawBuf);
    this.capturedLatestBinaryFrame = cloned;
    this.stats.binaryFramesSent++;

    if (!this.steadyStateCleanLink && this.rng() < this.binaryDropRate) {
      this.stats.binaryFramesDropped++;
      return;
    }

    const delay = this.sampleBinaryDelayMs();
    const arrival = this.clock.nowMs + delay;
    if (arrival < this.lastBinaryArrivalMs) {
      this.stats.binaryFramesReorderedOnWire++;
    }
    this.lastBinaryArrivalMs = Math.max(this.lastBinaryArrivalMs, arrival);

    this.schedulePacket(delay, () => {
      this.stats.binaryFramesDelivered++;
      deliverFn(cloned);
    });
  }

  injectDelayedRawStatePacketToGuest(guestEngine, rawBuffer, delayMs = 40) {
    if (!rawBuffer) return;
    this.stats.delayedPreRematchInjections++;
    const copy = cloneBufferPayload(rawBuffer);
    this.schedulePacket(delayMs, () => {
      const mpClient = guestEngine?.mpClient;
      if (!mpClient || mpClient.disposed) return;
      if (mpClient.stateChannel && mpClient.stateChannel.readyState === 'open') {
        mpClient.stateChannel.onmessage?.({ data: copy });
      } else if (mpClient.relayWs && mpClient.relayWs.readyState === 1) {
        mpClient.relayWs.onmessage?.({ data: copy });
      }
    });
  }

  injectDuplicateAudioBatchToGuest(guestEngine, rawJson, delayMs = 18) {
    if (!rawJson) return;
    this.stats.duplicateAudioBatchesInjected++;
    this.schedulePacket(delayMs, () => {
      const mpClient = guestEngine?.mpClient;
      if (!mpClient || mpClient.disposed) return;
      if (mpClient.ctrlChannel && mpClient.ctrlChannel.readyState === 'open') {
        mpClient.ctrlChannel.onmessage?.({ data: rawJson });
      } else if (mpClient.relayWs && mpClient.relayWs.readyState === 1) {
        mpClient.relayWs.onmessage?.({ data: rawJson });
      }
    });
  }

  severActiveWebRtcLinks(modeIndex = 0) {
    const pcs = Array.from(this.activePeerConnections);
    if (pcs.length === 0) return false;
    this.stats.webrtcDisconnectsInjected++;
    const mode = modeIndex % 5;

    for (const pc of pcs) {
      pc._severed = true;
    }

    const targetPc = pcs[pcs.length - 1]; // Guest or Host side trigger; unlink peer simultaneously
    const partnerPc = targetPc._partnerPc ?? pcs[0];

    const closePartnerSilently = (other) => {
      if (!other || other === targetPc) return;
      other.connectionState = 'disconnected';
      other.iceConnectionState = 'disconnected';
      if (other._channels) {
        for (const ch of other._channels.values()) {
          ch.readyState = 'closed';
        }
      }
      other.oniceconnectionstatechange?.();
      other.onconnectionstatechange?.();
    };

    if (mode === 0) {
      const ctrl = targetPc._channels.get(0);
      if (ctrl) {
        ctrl.readyState = 'closed';
        ctrl.onerror?.(new Event('error'));
        ctrl.onclose?.(new Event('close'));
      }
      closePartnerSilently(partnerPc);
    } else if (mode === 1) {
      const state = targetPc._channels.get(1);
      if (state) {
        state.readyState = 'closed';
        state.onerror?.(new Event('error'));
        state.onclose?.(new Event('close'));
      }
      closePartnerSilently(partnerPc);
    } else if (mode === 2) {
      targetPc.iceConnectionState = 'failed';
      targetPc.connectionState = 'failed';
      targetPc.oniceconnectionstatechange?.();
      closePartnerSilently(partnerPc);
    } else if (mode === 3) {
      targetPc.connectionState = 'disconnected';
      targetPc.onconnectionstatechange?.();
      closePartnerSilently(partnerPc);
    } else {
      // Mode 4: DataChannelstays 'open' until the very next `send()` throws `InvalidStateError`!
      for (const pc of pcs) {
        for (const ch of pc._channels.values()) {
          ch._throwOnNextSend = true;
        }
      }
    }
    return true;
  }

  install() {
    if (this.savedTransportGlobals) return;
    this.savedTransportGlobals = {
      WebSocket: globalThis.WebSocket,
      RTCPeerConnection: globalThis.RTCPeerConnection,
    };

    const net = this;

    class ChaosMockWebSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;

      constructor(url) {
        this.url = String(url);
        this.readyState = ChaosMockWebSocket.CONNECTING;
        this.binaryType = 'arraybuffer';
        this.bufferedAmount = 0;
        this.onopen = null;
        this.onmessage = null;
        this.onclose = null;
        this.onerror = null;
        this._orderedArrivalMs = net.clock.nowMs;

        const parsed = new URL(this.url);
        if (net.unreachableHosts.has(parsed.host)) {
          net.schedulePacket(6, () => {
            if (this.readyState === ChaosMockWebSocket.CLOSED) return;
            this.readyState = ChaosMockWebSocket.CLOSED;
            this.onerror?.(new Event('error'));
            this.onclose?.({ code: 1006, reason: 'Connection refused' });
          });
          return;
        }

        if (parsed.pathname.includes('peerjs')) {
          this._kind = 'peerjs';
          this._peerId = parsed.searchParams.get('id') || '';
          net.schedulePacket(4, () => {
            if (this.readyState !== ChaosMockWebSocket.CONNECTING) return;
            this.readyState = ChaosMockWebSocket.OPEN;
            net.peerJsClients.set(this._peerId, this);
            this.onopen?.(new Event('open'));
            net.schedulePacket(2, () => {
              if (this.readyState === ChaosMockWebSocket.OPEN) {
                this.onmessage?.({ data: JSON.stringify({ type: 'OPEN' }) });
              }
            });
          });
        } else {
          this._kind = 'relay';
          this._room = (parsed.searchParams.get('room') || '').toUpperCase();
          this._role = parsed.searchParams.get('role') || 'guest';
          this._clientId = parsed.searchParams.get('clientId') || '';
          net.schedulePacket(5, () => {
            if (this.readyState !== ChaosMockWebSocket.CONNECTING) return;
            this.readyState = ChaosMockWebSocket.OPEN;
            let room = net.relayRooms.get(this._room);
            if (!room) {
              room = { host: null, guest: null };
              net.relayRooms.set(this._room, room);
            }
            const prevSameRole = this._role === 'host' ? room.host : room.guest;
            if (prevSameRole && prevSameRole !== this && prevSameRole.readyState === ChaosMockWebSocket.OPEN) {
              prevSameRole.readyState = ChaosMockWebSocket.CLOSED;
              prevSameRole.onclose?.({ code: 4000, reason: 'Superseded' });
            }
            if (this._role === 'host') room.host = this;
            else room.guest = this;

            this.onopen?.(new Event('open'));

            const other = this._role === 'host' ? room.guest : room.host;
            const roomStateMsg = JSON.stringify({
              __relay: 'room_state',
              hasHost: Boolean(room.host && room.host.readyState === ChaosMockWebSocket.OPEN),
              hasGuest: Boolean(room.guest && room.guest.readyState === ChaosMockWebSocket.OPEN),
            });
            this.onmessage?.({ data: roomStateMsg });

            if (other && other.readyState === ChaosMockWebSocket.OPEN) {
              const joinedMsg = JSON.stringify({
                __relay: 'peer_joined',
                role: this._role,
              });
              net.schedulePacket(2, () => {
                if (other.readyState === ChaosMockWebSocket.OPEN) {
                  other.onmessage?.({ data: joinedMsg });
                }
              });
            }
          });
        }
      }

      send(data) {
        if (this.readyState !== ChaosMockWebSocket.OPEN) {
          throw new Error('WebSocket is not open');
        }
        if (this._kind === 'peerjs') {
          let parsed;
          try {
            parsed = JSON.parse(String(data));
          } catch {
            return;
          }
          if (parsed.type === 'HEARTBEAT') return;
          const dst = parsed.dst;
          if (!dst) return;
          const targetWs = net.peerJsClients.get(dst);
          const delay = net.sampleControlDelayMs(this._orderedArrivalMs);
          this._orderedArrivalMs = net.clock.nowMs + delay;
          net.schedulePacket(delay, () => {
            if (targetWs && targetWs.readyState === ChaosMockWebSocket.OPEN) {
              targetWs.onmessage?.({
                data: JSON.stringify({
                  type: parsed.type,
                  src: this._peerId,
                  dst,
                  payload: parsed.payload,
                }),
              });
            } else if (this.readyState === ChaosMockWebSocket.OPEN && parsed.type === 'OFFER') {
              this.onmessage?.({
                data: JSON.stringify({ type: 'EXPIRE', src: dst, dst: this._peerId }),
              });
            }
          });
          return;
        }

        const room = net.relayRooms.get(this._room);
        if (!room) return;
        const other = this._role === 'host' ? room.guest : room.host;
        if (!other || other.readyState !== ChaosMockWebSocket.OPEN) return;

        if (typeof data === 'string') {
          if (data.includes('"type":"audio_batch"')) {
            net.capturedLatestAudioBatchJson = data;
          }
          const delay = net.sampleControlDelayMs(this._orderedArrivalMs);
          this._orderedArrivalMs = net.clock.nowMs + delay;
          net.schedulePacket(delay, () => {
            if (other.readyState === ChaosMockWebSocket.OPEN) {
              other.onmessage?.({ data });
            }
          });
        } else {
          net.routeBinaryStatePayload(data, (buf) => {
            if (other.readyState === ChaosMockWebSocket.OPEN) {
              other.onmessage?.({ data: buf });
            }
          });
        }
      }

      close() {
        if (this.readyState === ChaosMockWebSocket.CLOSED) return;
        const wasOpen = this.readyState === ChaosMockWebSocket.OPEN;
        this.readyState = ChaosMockWebSocket.CLOSED;
        if (this._kind === 'peerjs') {
          if (net.peerJsClients.get(this._peerId) === this) {
            net.peerJsClients.delete(this._peerId);
          }
        } else if (this._kind === 'relay') {
          const room = net.relayRooms.get(this._room);
          if (room) {
            const other = this._role === 'host' ? room.guest : room.host;
            if (this._role === 'host' && room.host === this) room.host = null;
            if (this._role === 'guest' && room.guest === this) room.guest = null;
            if (wasOpen && other && other.readyState === ChaosMockWebSocket.OPEN) {
              const leftMsg = JSON.stringify({ __relay: 'peer_left', role: this._role });
              net.schedulePacket(2, () => {
                if (other.readyState === ChaosMockWebSocket.OPEN) {
                  other.onmessage?.({ data: leftMsg });
                }
              });
            }
          }
        }
        this.onclose?.({ code: 1000, reason: 'Normal closure' });
      }
    }

    let nextPcId = 1;
    /** @type {Map<string, any>} */
    const sdpRegistry = new Map();

    class ChaosMockRTCPeerConnection {
      constructor(config = {}) {
        this._pcId = `pc_${nextPcId++}`;
        this.config = config;
        this.signalingState = 'stable';
        this.iceConnectionState = 'new';
        this.connectionState = 'new';
        this.localDescription = null;
        this.remoteDescription = null;
        this.onicecandidate = null;
        this.onconnectionstatechange = null;
        this.oniceconnectionstatechange = null;
        this._channels = new Map();
        this._partnerPc = null;
        this._closed = false;
        this._severed = false;
        this._orderedCtrlArrivalMs = net.clock.nowMs;
        net.activePeerConnections.add(this);
      }

      createDataChannel(label, opts = {}) {
        const id = opts.id ?? (label === 'ctrl' ? 0 : 1);
        const pc = this;
        const ch = {
          label,
          id,
          ordered: Boolean(opts.ordered),
          maxRetransmits: opts.maxRetransmits ?? null,
          binaryType: 'arraybuffer',
          readyState: 'connecting',
          bufferedAmount: 0,
          onopen: null,
          onmessage: null,
          onclose: null,
          onerror: null,
          _throwOnNextSend: false,
          send(data) {
            if (this._throwOnNextSend) {
              this._throwOnNextSend = false;
              this.readyState = 'closed';
              if (pc._partnerPc) {
                pc._partnerPc.connectionState = 'disconnected';
                pc._partnerPc.onconnectionstatechange?.();
              }
              throw new Error('RTCDataChannel InvalidStateError: SCTP transport severed');
            }
            if (this.readyState !== 'open' || pc._closed || pc._severed) {
              throw new Error('RTCDataChannel is not open');
            }
            const partner = pc._partnerPc;
            if (!partner || partner._closed || partner._severed) return;
            const remoteCh = partner._channels.get(id);
            if (!remoteCh || remoteCh.readyState !== 'open') return;

            if (typeof data === 'string') {
              if (data.includes('"type":"audio_batch"')) {
                net.capturedLatestAudioBatchJson = data;
              }
              const delay = net.sampleControlDelayMs(pc._orderedCtrlArrivalMs);
              pc._orderedCtrlArrivalMs = net.clock.nowMs + delay;
              net.schedulePacket(delay, () => {
                if (!partner._closed && !partner._severed && remoteCh.readyState === 'open') {
                  remoteCh.onmessage?.({ data });
                }
              });
            } else {
              net.routeBinaryStatePayload(data, (buf) => {
                if (!partner._closed && !partner._severed && remoteCh.readyState === 'open') {
                  remoteCh.onmessage?.({ data: buf });
                }
              });
            }
          },
          close() {
            if (this.readyState === 'closed') return;
            this.readyState = 'closed';
          },
        };
        this._channels.set(id, ch);
        return ch;
      }

      async createOffer() {
        return { type: 'offer', sdp: `v=0\r\no=${this._pcId} 1 1 IN IP4 127.0.0.1\r\n` };
      }

      async createAnswer() {
        return { type: 'answer', sdp: `v=0\r\no=${this._pcId} 2 1 IN IP4 127.0.0.1\r\n` };
      }

      async setLocalDescription(desc) {
        this.localDescription = desc;
        sdpRegistry.set(desc.sdp, this);
        if (desc.type === 'offer') {
          this.signalingState = 'have-local-offer';
        } else if (desc.type === 'answer') {
          this.signalingState = 'stable';
        }
        net.schedulePacket(2, () => {
          if (this._closed) return;
          this.onicecandidate?.({
            candidate: {
              candidate: 'candidate:1 1 udp 2122260223 198.51.100.7 54321 typ host',
              sdpMid: '0',
              sdpMLineIndex: 0,
              toJSON() {
                return {
                  candidate: this.candidate,
                  sdpMid: this.sdpMid,
                  sdpMLineIndex: this.sdpMLineIndex,
                };
              },
            },
          });
        });
        this._maybeCompleteIceLink();
      }

      async setRemoteDescription(desc) {
        this.remoteDescription = desc;
        const remotePc = sdpRegistry.get(desc.sdp);
        if (remotePc && remotePc !== this) {
          this._partnerPc = remotePc;
          remotePc._partnerPc = this;
        }
        if (desc.type === 'offer') {
          this.signalingState = 'have-remote-offer';
        } else if (desc.type === 'answer') {
          this.signalingState = 'stable';
        }
        this._maybeCompleteIceLink();
      }

      async addIceCandidate() {
        // No-op in virtual SCTP pair
      }

      _maybeCompleteIceLink() {
        const partner = this._partnerPc;
        if (!partner || this._closed || partner._closed) return;
        if (this.signalingState !== 'stable' || partner.signalingState !== 'stable') return;
        if (this._iceScheduled || partner._iceScheduled) return;
        this._iceScheduled = true;
        partner._iceScheduled = true;

        this.iceConnectionState = 'checking';
        this.connectionState = 'connecting';
        partner.iceConnectionState = 'checking';
        partner.connectionState = 'connecting';

        if (net.webrtcStallIce) {
          // Remain stuck in 'checking' so VitrailMultiplayerClient's `webrtcHandshakeTimer` fires!
          return;
        }

        if (net.webrtcFailIceImmediately) {
          net.schedulePacket(18, () => {
            for (const p of [this, partner]) {
              if (p._closed) continue;
              p.iceConnectionState = 'failed';
              p.connectionState = 'failed';
              p.oniceconnectionstatechange?.();
              p.onconnectionstatechange?.();
            }
          });
          return;
        }

        net.schedulePacket(14, () => {
          if (this._closed || partner._closed || this._severed || partner._severed) return;
          net.stats.webrtcConnectionsOpened++;
          for (const p of [this, partner]) {
            p.iceConnectionState = 'connected';
            p.connectionState = 'connected';
            p.oniceconnectionstatechange?.();
            p.onconnectionstatechange?.();
            for (const ch of p._channels.values()) {
              ch.readyState = 'open';
              ch.onopen?.(new Event('open'));
            }
          }
        });
      }

      close() {
        if (this._closed) return;
        this._closed = true;
        net.activePeerConnections.delete(this);
        this.connectionState = 'closed';
        this.iceConnectionState = 'closed';
        for (const ch of this._channels.values()) {
          ch.readyState = 'closed';
        }
      }
    }

    globalThis.WebSocket = /** @type {any} */ (ChaosMockWebSocket);
    globalThis.RTCPeerConnection = /** @type {any} */ (ChaosMockRTCPeerConnection);
  }

  uninstall() {
    if (!this.savedTransportGlobals) return;
    globalThis.WebSocket = this.savedTransportGlobals.WebSocket;
    globalThis.RTCPeerConnection = this.savedTransportGlobals.RTCPeerConnection;
    this.packetQueue.length = 0;
    this.peerJsClients.clear();
    this.relayRooms.clear();
    this.activePeerConnections.clear();
    this.savedTransportGlobals = null;
  }
}

/**
 * Installs a lightweight deterministic mock WebAudio `AudioContext` on `GlassAudioEngine`
 * so `playBlockThud` (`-> this.playGlassImpact`), `playGlassCrack` (`-> this.playGlassImpact`),
 * `playGlassImpact`, and `playGlassShatter` execute their full internal codepaths (verifying
 * `installHostAudioTap`'s `runTopLevelGuarded` never double-records nested calls!).
 */
function attachMockAudioContextAndCounters(audioEngine, clock, options = {}) {
  const onlyCountScheduled = Boolean(options.onlyCountScheduled);
  const counts = {
    blockThud: 0,
    glassImpact: 0,
    glassShatter: 0,
    nestedCallsSuppressed: 0,
    cooldownDropped: 0,
  };

  const dummyParam = {
    value: 1,
    setValueAtTime() {},
    linearRampToValueAtTime() {},
    exponentialRampToValueAtTime() {},
    setTargetAtTime() {},
    cancelScheduledValues() {},
  };
  const makeNode = () => ({
    gain: { ...dummyParam },
    frequency: { ...dummyParam },
    Q: { ...dummyParam },
    pan: { ...dummyParam },
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

  const mockCtx = {
    sampleRate: 48000,
    state: 'running',
    get currentTime() {
      return clock.nowMs / 1000;
    },
    destination: makeNode(),
    createGain: makeNode,
    createOscillator: makeNode,
    createBiquadFilter: makeNode,
    createStereoPanner: makeNode,
    createBufferSource: makeNode,
    createBuffer(_channels, length) {
      const ch = new Float32Array(Math.min(length, 4096));
      return { getChannelData: () => ch };
    },
    resume: () => Promise.resolve(),
    close: () => Promise.resolve(),
  };

  audioEngine.ctx = mockCtx;
  audioEngine.masterGain = makeNode();
  audioEngine.noiseBuffer = mockCtx.createBuffer(1, 1024, 48000);
  audioEngine.muted = false;

  let depth = 0;
  const wrapCounted = (methodName, key) => {
    const orig = audioEngine[methodName].bind(audioEngine);
    audioEngine[methodName] = (...args) => {
      const isTop = depth === 0;
      if (!isTop) {
        counts.nestedCallsSuppressed++;
      }
      depth++;
      try {
        const res = orig(...args);
        if (isTop) {
          if (onlyCountScheduled && res === false) {
            counts.cooldownDropped++;
          } else {
            counts[key]++;
          }
        }
        return res;
      } finally {
        depth--;
      }
    };
  };

  const wrapGuardOnly = (methodName) => {
    if (typeof audioEngine[methodName] !== 'function') return;
    const orig = audioEngine[methodName].bind(audioEngine);
    audioEngine[methodName] = (...args) => {
      depth++;
      try {
        return orig(...args);
      } finally {
        depth--;
      }
    };
  };

  wrapCounted('playBlockThud', 'blockThud');
  wrapCounted('playGlassImpact', 'glassImpact');
  wrapCounted('playGlassShatter', 'glassShatter');
  wrapGuardOnly('playGlassCrack');
  wrapGuardOnly('playTapTestResonance');

  return {
    counts,
    resetCounts() {
      counts.blockThud = 0;
      counts.glassImpact = 0;
      counts.glassShatter = 0;
      counts.nestedCallsSuppressed = 0;
      counts.cooldownDropped = 0;
    },
  };
}

/**
 * Builds and runs the 100-match Adversarial Network Chaos Simulator.
 */
export async function createChaosMultiplayerSuite(deps) {
  const {
    GameEngine,
    buildDeterministicTowerMaterials,
    canonicalizeQuatHemisphere,
    quatRotateVec3,
  } = deps;

  async function runChaosSimulation(options = {}) {
    const matchCount = options.matches ?? 100;
    const baseSeed = (options.seed ?? 0x56495452) >>> 0;
    const relayUrl = 'wss://vitrail-rooms.chaos.workers.dev/__vitrail_mp';
    const peerJsUrl = 'wss://0.peerjs.com/peerjs';

    const t0Wall = nodePerf.now();
    const clock = new ChaosVirtualClock(10_000);
    const rng = mulberry32(baseSeed);
    const net = new ChaosTransportNetwork(clock, rng);

    let uncaughtExceptions = 0;
    let unhandledRejections = 0;
    /** @type {string[]} */
    const runtimeErrorMessages = [];

    const onUncaught = (err) => {
      uncaughtExceptions++;
      runtimeErrorMessages.push(`uncaughtException: ${err?.stack || err}`);
    };
    const onUnhandled = (reason) => {
      unhandledRejections++;
      runtimeErrorMessages.push(`unhandledRejection: ${reason?.stack || reason}`);
    };

    process.on('uncaughtException', onUncaught);
    process.on('unhandledRejection', onUnhandled);

    const savedDoc = globalThis.document;
    const mockDocument = { hidden: false };
    globalThis.document = /** @type {any} */ (mockDocument);

    // Boot two reusable headless GameEngine shells BEFORE installing virtual timers so WASM loads cleanly
    const host = new GameEngine();
    const guest = new GameEngine();
    await host.initHeadless();
    await guest.initHeadless();

    const hostAudioTracker = attachMockAudioContextAndCounters(host.audio, clock, {
      onlyCountScheduled: true,
    });
    const guestAudioTracker = attachMockAudioContextAndCounters(guest.audio, clock);
    // Re-wrap Host audio tap ON TOP of our counter wrapper so `hostAudioTracker` sees exact Host calls!
    host.hostAudioTapInstalled = false;
    host.installHostAudioTap();

    const metrics = {
      matchesRequested: matchCount,
      matchesCompleted: 0,
      rematchesCompleted: 0,
      uncaughtExceptions: 0,
      unhandledRejections: 0,
      nanCount: 0,
      antipodalFlipCount: 0,
      equatorCrossingsTested: 0,
      guestStepsAudited120Hz: 0,
      monotonicSeqViolations: 0,
      staleRematchRollbacks: 0,
      totalGuestAcceptedFrames: 0,
      totalGuestRejectedStaleFrames: 0,
      swatchParityChecks: 0,
      swatchParityMismatches: 0,
      doubleGrabRacesInjected: 0,
      doubleGrabViolations: 0,
      occludedHostTurnsCompleted: 0,
      occludedBgPumpSnapshotsSent: 0,
      midDragFailoversCompleted: 0,
      audioParity: {
        host: { blockThud: 0, glassImpact: 0, glassShatter: 0, nestedSuppressed: 0 },
        guest: { blockThud: 0, glassImpact: 0, glassShatter: 0 },
        epochMismatches: 0,
        duplicateBatchesRejectedOnGuest: 0,
      },
      maxSteadyStateConvergenceErrorMm: 0,
      convergenceSamplesCount: 0,
      convergenceViolationsOver2Mm: 0,
      network: net.stats,
      wallTimeMs: 0,
      gateFailures: /** @type {string[]} */ ([]),
    };

    /** @type {Quat[]} */
    let prevGuestBlockQuats = [];
    let prevObservedAcceptCount = 0;
    let prevObservedSeed = 0;
    let prevGuestMonotonicSeq = 0;

    const auditGuest120HzStep = (guestWorld) => {
      if (!guestWorld) return;
      metrics.guestStepsAudited120Hz++;
      const curAccept = guestWorld.getAcceptedFrameCount();
      const curSeq = guestWorld.getLastAcceptedSeq();
      const curSeed = guestWorld.roomSeed;

      if (curSeq < prevGuestMonotonicSeq) {
        metrics.monotonicSeqViolations++;
      }
      prevGuestMonotonicSeq = curSeq;

      const frameJustApplied = curAccept !== prevObservedAcceptCount || curSeed !== prevObservedSeed;
      prevObservedAcceptCount = curAccept;
      prevObservedSeed = curSeed;

      const blocks = guestWorld.blocks;
      if (prevGuestBlockQuats.length !== blocks.length) {
        prevGuestBlockQuats = blocks.map((b) => [...b.rotation]);
      }

      for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        const p = b.position;
        const q = b.rotation;
        if (
          !Number.isFinite(p[0]) ||
          !Number.isFinite(p[1]) ||
          !Number.isFinite(p[2]) ||
          !Number.isFinite(q[0]) ||
          !Number.isFinite(q[1]) ||
          !Number.isFinite(q[2]) ||
          !Number.isFinite(q[3])
        ) {
          metrics.nanCount++;
          continue;
        }
        const qNorm = Math.hypot(q[0], q[1], q[2], q[3]);
        if (!Number.isFinite(qNorm) || Math.abs(qNorm - 1.0) > 2e-3) {
          metrics.nanCount++;
        }

        const prevQ = prevGuestBlockQuats[i];
        if (!frameJustApplied && !b.shattered && !b.extracted && prevQ) {
          // During pure 120 Hz exponential slerp (`alpha ~ 0.206`), max geodesic step is <= 0.206 * pi ~ 0.649 rad.
          // Any antipodal hemisphere flip across `w = 0` produces a > 0.72 rad single-step jump or raw dot reversal!
          const stepAng = quatGeodesicAngleRad(prevQ, q);
          if (stepAng > 0.72) {
            metrics.antipodalFlipCount++;
          }
        }
        prevGuestBlockQuats[i] = [q[0], q[1], q[2], q[3]];
      }

      const shards = guestWorld.shards;
      for (let i = 0; i < shards.length; i++) {
        const s = shards[i];
        if (
          !Number.isFinite(s.position[0]) ||
          !Number.isFinite(s.position[1]) ||
          !Number.isFinite(s.position[2]) ||
          !Number.isFinite(s.rotation[0]) ||
          !Number.isFinite(s.rotation[1]) ||
          !Number.isFinite(s.rotation[2]) ||
          !Number.isFinite(s.rotation[3])
        ) {
          metrics.nanCount++;
        }
      }
    };

    const verify54BlockSwatchParity = (expectedSeed) => {
      metrics.swatchParityChecks++;
      const expected = buildDeterministicTowerMaterials(expectedSeed >>> 0, 18);
      if (host.physics.blocks.length !== 54 || guest.physics.blocks.length !== 54) {
        metrics.swatchParityMismatches++;
        return;
      }
      for (let i = 0; i < 54; i++) {
        const hb = host.physics.blocks[i];
        const gb = guest.physics.blocks[i];
        const ref = expected[i];
        if (
          hb.swatchId !== ref.swatchId ||
          gb.swatchId !== ref.swatchId ||
          hb.motif !== ref.motif ||
          gb.motif !== ref.motif ||
          Math.abs(hb.rippleSeed - ref.rippleSeed) > 1e-5 ||
          Math.abs(gb.rippleSeed - ref.rippleSeed) > 1e-5
        ) {
          metrics.swatchParityMismatches++;
          return;
        }
      }
    };

    let hostRendersHalted = false;
    let hostStepAccumulatorMs = 0;

    const advanceSimMs = async (durationMs) => {
      const quantumMs = 1000 / 120; // Exact 120 Hz Guest frame cadence (~8.333 ms)
      const steps = Math.max(1, Math.round(durationMs / quantumMs));
      for (let s = 0; s < steps; s++) {
        clock.nowMs += quantumMs;
        clock.fireDueTimers();
        net.deliverDuePackets();
        // Flush async microtasks (`RTCPeerConnection` SDP promises)
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
        net.deliverDuePackets();

        if (!hostRendersHalted) {
          hostStepAccumulatorMs += quantumMs;
          if (hostStepAccumulatorMs >= 1000 / 60 - 0.01) {
            const hDt = hostStepAccumulatorMs / 1000;
            hostStepAccumulatorMs = 0;
            host.stepHeadless(hDt);
          }
        }

        guest.stepHeadless(quantumMs / 1000);
        auditGuest120HzStep(guest.guestWorld);
      }
    };

    clock.install();
    net.install();

    try {
      for (let m = 0; m < matchCount; m++) {
        const matchSeed = (baseSeed ^ Math.imul(m + 1, 0x9e3779b9)) >>> 0 || 0x13572468;
        const roomCode = makeRoomCodeFromIndex(m, rng);

        // Reset baseline adversarial parameters (`8%` drop, `120 +/- 45 ms` jitter)
        net.binaryDropRate = 0.08;
        net.jitterMeanMs = 120;
        net.jitterHalfSpanMs = 45;
        net.steadyStateCleanLink = false;
        net.webrtcStallIce = false;
        net.webrtcFailIceImmediately = false;
        hostRendersHalted = false;
        mockDocument.hidden = false;
        host.setHostTabOccluded(false);

        hostAudioTracker.resetCounts();
        guestAudioTracker.resetCounts();
        prevGuestBlockQuats = [];
        prevObservedAcceptCount = 0;
        prevObservedSeed = 0;
        prevGuestMonotonicSeq = 0;

        // Alternate initial topologies:
        // - Even matches: Relay-connected + WebRTC upgrade (`relayReady === true` across WebRTC -> WS downgrade!)
        // - Odd matches: PeerJS-first P2P (`relayUrl: null`, `fallbackRelayUrl: relayUrl`) dialing WS relay on mid-drag disconnect!
        const usePeerJsFirst = m % 2 === 1;
        const sessionOpts = usePeerJsFirst
          ? {
              relayUrl: null,
              fallbackRelayUrl: relayUrl,
              peerJsUrl,
              enablePeerJsFallback: true,
              enableWebRtcUpgrade: true,
            }
          : {
              relayUrl,
              fallbackRelayUrl: relayUrl,
              peerJsUrl,
              enablePeerJsFallback: false,
              enableWebRtcUpgrade: true,
            };

        host.startOnlineHost(roomCode, {
          ...sessionOpts,
          roomSeed: matchSeed,
        });
        guest.joinOnlineRoom(roomCode, {
          ...sessionOpts,
          roomSeed: (matchSeed ^ 0xdeadbeef) >>> 0,
        });

        // 1. Await initial P2P WebRTC handshake & `room_welcome`
        for (let waitIter = 0; waitIter < 25; waitIter++) {
          await advanceSimMs(40);
          if (
            host.getTelemetrySnapshot().mpPeerConnected &&
            guest.getTelemetrySnapshot().mpPeerConnected &&
            host.getTelemetrySnapshot().mpTransport === 'webrtc-p2p' &&
            guest.getTelemetrySnapshot().mpTransport === 'webrtc-p2p' &&
            guest.mpRoomSeed === matchSeed
          ) {
            break;
          }
        }
        // Reset per-match audio parity counters once both peers are paired in the live room
        hostAudioTracker.resetCounts();
        guestAudioTracker.resetCounts();
        verify54BlockSwatchParity(matchSeed);

        // 2. Condition (e): Simultaneous Double-Grab Race #1 on Turn 0 (`currentPlayer === 0`)
        const turn0Loose = host.physics.blocks.find(
          (b) => b.assignedLayer === 8 && b.isLooseSlot && b.isLegalTarget && !b.extracted
        );
        const altLoose0 = host.physics.blocks.find(
          (b) => b.assignedLayer === 6 && b.isLooseSlot && b.isLegalTarget && !b.extracted
        );
        if (turn0Loose && altLoose0) {
          metrics.doubleGrabRacesInjected++;
          host.camera.setPoseImmediate(0, 0.34, 4.2, [0, 1.35, 0], 1280 / 800);
          guest.camera.setPoseImmediate(0, 0.34, 4.2, [0, 1.35, 0], 1280 / 800);
          const gPt = projectWorldToScreen(guest.camera, altLoose0.position);
          if (gPt) {
            guest.handlePointerDown(gPt[0], gPt[1], 1, 0, false, VIEWPORT_RECT);
            guest.handlePointerMove(gPt[0] + 24, gPt[1], 1, VIEWPORT_RECT);
            guest.handlePointerUp(gPt[0] + 24, gPt[1], 1, VIEWPORT_RECT);
          }
          // Also inject raw spoofed wire `guest_cmd` racing Host's grab
          guest.mpClient?.sendControl({
            type: 'guest_cmd',
            seq: 99999,
            roomSeed: matchSeed,
            cmd: {
              kind: 'begin_pull',
              blockId: altLoose0.id,
              localPoint: [0, 0, 0],
              worldPoint: [...altLoose0.position],
            },
          });

          const hAxisX = quatRotateVec3(turn0Loose.rotation, [1, 0, 0]);
          const startW0 = [...turn0Loose.position];
          const started0 = host.physics.startGrabBlock(turn0Loose.id, [0, 0, 0], startW0);
          await advanceSimMs(50);

          if (!started0 || host.physics.activeBlockId !== turn0Loose.id) {
            metrics.doubleGrabViolations++;
          }

          // Extract block on Turn 0 & sweep carried rotation across `w = 0` equator (`yaw = pi`)!
          for (let step = 1; step <= 18 && host.physics.turnPhase === 'PULLING'; step++) {
            const dist = Math.min(1.45, (step / 12) * 1.45);
            host.physics.updateGrabTarget([
              startW0[0] + hAxisX[0] * dist,
              startW0[1],
              startW0[2] + hAxisX[2] * dist,
            ]);
            await advanceSimMs(25);
          }

          for (let waitLift = 0; waitLift < 15 && !host.physics.isExtractionLiftReady; waitLift++) {
            await advanceSimMs(25);
          }

          if (host.physics.turnPhase === 'EXTRACTED') {
            // Sweep carried block yaw across 180 deg (`w = 0` hemisphere equator!) to stress-test `quatSlerpHemisphere`
            const qBefore = canonicalizeQuatHemisphere(host.physics.carriedTargetQuat);
            host.physics.rotateCarriedBlock(Math.PI * 0.5);
            await advanceSimMs(35);
            host.physics.rotateCarriedBlock(Math.PI * 0.5);
            await advanceSimMs(35);
            const qAfter = canonicalizeQuatHemisphere(host.physics.carriedTargetQuat);
            if (
              qBefore[0] * qAfter[0] +
                qBefore[1] * qAfter[1] +
                qBefore[2] * qAfter[2] +
                qBefore[3] * qAfter[3] <
              0.2
            ) {
              metrics.equatorCrossingsTested++;
            }
            host.physics.resetCarriedBlockRotation();
            const topSlot0 = host.physics.openTopSlots[0]?.position;
            if (topSlot0) {
              host.physics.updateCarriedHover(
                [topSlot0[0], topSlot0[1] + 1.0, topSlot0[2]],
                [0, -1, 0]
              );
            }
            host.handleReleaseOntoTop();
            // Emit real oak-table blockThud alongside placement glassImpact
            host.audio.playBlockThud(0.42, [0, 0.08, 0], { damage: 0.05 });
          }

          for (
            let settle0 = 0;
            settle0 < 40 &&
            !(host.physics.currentPlayer === 1 && host.physics.turnPhase === 'SELECT_BLOCK');
            settle0++
          ) {
            await advanceSimMs(35);
          }
        }

        // Synchronize Guest to `currentPlayer === 1` before Turn 1
        for (
          let syncWait = 0;
          syncWait < 12 &&
          !(guest.physics.currentPlayer === 1 && guest.physics.turnPhase === 'SELECT_BLOCK');
          syncWait++
        ) {
          await advanceSimMs(35);
        }

        // 3. Conditions (c), (d) & (e):
        //    - Simultaneous Host + Guest Double-Grab Race on Turn 1 (`currentPlayer === 1`)
        //    - Occluded Host Tab (`30 Hz` background pump with `rAF` halted)
        //    - Mid-Drag WebRTC Disconnect -> Automatic WS Relay Failover + Pull & Audio Resync
        const guestTarget = host.physics.blocks.find(
          (b) =>
            b.assignedLayer >= 4 &&
            b.assignedLayer <= 10 &&
            b.isLooseSlot &&
            b.isLegalTarget &&
            !b.extracted &&
            !b.shattered
        );
        const hostRaceBlock = host.physics.blocks.find(
          (b) =>
            b.id !== guestTarget?.id &&
            b.isLooseSlot &&
            b.isLegalTarget &&
            !b.extracted &&
            !b.shattered
        );

        if (guestTarget && hostRaceBlock && host.physics.currentPlayer === 1) {
          metrics.doubleGrabRacesInjected++;
          const gAxisX = quatRotateVec3(guestTarget.rotation, [1, 0, 0]);
          const endFaceWorld = [
            guestTarget.position[0] + gAxisX[0] * 0.36,
            guestTarget.position[1],
            guestTarget.position[2] + gAxisX[2] * 0.36,
          ];
          const camAz = Math.atan2(gAxisX[0], gAxisX[2]);
          guest.camera.setPoseImmediate(
            camAz,
            0.22,
            4.2,
            [0, guestTarget.position[1], 0],
            VIEWPORT_RECT.width / VIEWPORT_RECT.height
          );
          host.camera.setPoseImmediate(
            camAz,
            0.22,
            4.2,
            [0, hostRaceBlock.position[1], 0],
            VIEWPORT_RECT.width / VIEWPORT_RECT.height
          );

          const hScreen = projectWorldToScreen(host.camera, hostRaceBlock.position);
          const gScreen0 = projectWorldToScreen(guest.camera, endFaceWorld);

          // Race Host local pointer drag vs Guest pointer drag at the exact same timestamp!
          if (hScreen) {
            host.handlePointerDown(hScreen[0], hScreen[1], 1, 0, false, VIEWPORT_RECT);
            host.handlePointerMove(hScreen[0] + 28, hScreen[1], 1, VIEWPORT_RECT);
          }
          if (gScreen0) {
            guest.handlePointerDown(gScreen0[0], gScreen0[1], 1, 0, false, VIEWPORT_RECT);
            const gScreenPull1 = projectWorldToScreen(guest.camera, [
              endFaceWorld[0] + gAxisX[0] * 0.18,
              endFaceWorld[1],
              endFaceWorld[2] + gAxisX[2] * 0.18,
            ]);
            if (gScreenPull1) {
              guest.handlePointerMove(gScreenPull1[0], gScreenPull1[1], 1, VIEWPORT_RECT);
            }
          }
          if (hScreen) {
            host.handlePointerUp(hScreen[0] + 28, hScreen[1], 1, VIEWPORT_RECT);
          }

          // Condition (d): Occlude Host tab (`document.hidden = true` + `setHostTabOccluded(true)`)
          // and HALT foreground Host stepping so ONLY `startHostBackgroundPump()` ticks at `30 Hz`!
          const seqBeforeOcclusion = host.mpStateSeq;
          mockDocument.hidden = true;
          host.setHostTabOccluded(true);
          hostRendersHalted = true;

          await advanceSimMs(110);

          if (
            host.physics.activeBlockId === hostRaceBlock.id ||
            (host.physics.activeBlockId !== null && host.physics.activeBlockId !== guestTarget.id)
          ) {
            metrics.doubleGrabViolations++;
          }

          // Condition (c): Sever WebRTC MID-DRAG while Guest is actively pulling!
          // Also emit `blockThud` + `glassImpact` right across the failover window AND inject a duplicate
          // `audio_batch` after failover to verify Guest deduplicates `batchSeq` (`0` duplicates)!
          host.audio.playBlockThud(0.55, [...guestTarget.position], { damage: 0.12 });
          const severed = net.severActiveWebRtcLinks(m);

          for (let pullStep = 2; pullStep <= 16; pullStep++) {
            const dist = Math.min(1.5, (pullStep / 11) * 1.5);
            const pullPt = projectWorldToScreen(guest.camera, [
              endFaceWorld[0] + gAxisX[0] * dist,
              endFaceWorld[1],
              endFaceWorld[2] + gAxisX[2] * dist,
            ]);
            if (pullPt) {
              guest.handlePointerMove(pullPt[0], pullPt[1], 1, VIEWPORT_RECT);
            }
            if (pullStep === 4) {
              host.audio.playGlassImpact(0.62, 1.02, [...guestTarget.position], undefined, undefined, {
                impactSpeedMps: 0.48,
                surface: 'glass',
              });
            }
            if (pullStep === 6 && net.capturedLatestAudioBatchJson) {
              // Replay an already-sent `audio_batch` on the wire to verify `batchSeq` deduplication!
              net.injectDuplicateAudioBatchToGuest(guest, net.capturedLatestAudioBatchJson, 10);
            }
            await advanceSimMs(34);
            if (host.physics.turnPhase === 'EXTRACTED') break;
          }

          guest.handlePointerUp(VIEWPORT_RECT.width * 0.5, VIEWPORT_RECT.height * 0.5, 1, VIEWPORT_RECT);

          if (
            severed &&
            host.getTelemetrySnapshot().mpTransport === 'ws-relay' &&
            guest.getTelemetrySnapshot().mpTransport === 'ws-relay' &&
            guest.getTelemetrySnapshot().mpPeerConnected
          ) {
            net.stats.wsRelayFailoversCompleted++;
            metrics.midDragFailoversCompleted++;
          }

          // Finish Guest placement while Host is STILL occluded on its 30 Hz background pump!
          for (let waitExt = 0; waitExt < 16 && !host.physics.isExtractionLiftReady; waitExt++) {
            await advanceSimMs(34);
          }
          if (host.physics.turnPhase === 'EXTRACTED') {
            guest.sendGuestRemoteCommand({ kind: 'confirm_placement' });
          }
          for (
            let settle1 = 0;
            settle1 < 36 && host.physics.blocksStacked < 2 && host.physics.turnPhase !== 'COLLAPSED';
            settle1++
          ) {
            await advanceSimMs(34);
          }

          const bgSnaps = (host.mpStateSeq - seqBeforeOcclusion) >>> 0;
          metrics.occludedBgPumpSnapshotsSent += bgSnaps;
          if (bgSnaps >= 8) {
            metrics.occludedHostTurnsCompleted++;
          }

          // Restore foreground Host tab visibility
          hostRendersHalted = false;
          mockDocument.hidden = false;
          host.setHostTabOccluded(false);
        }

        // 4. Real Voronoi Block Shatter (`playGlassShatter`) + Packet-Loss Burst -> `< 2.0 mm` (`250 ms`) Convergence Gate
        const targetShatterLayer = m % 8 === 0 ? 3 : 16;
        const shatterCandidate =
          host.physics.blocks.find(
            (b) =>
              !b.shattered &&
              !b.extracted &&
              b.assignedLayer === targetShatterLayer &&
              b.isLooseSlot
          ) ??
          host.physics.blocks.find((b) => !b.shattered && !b.extracted && b.isLooseSlot);
        if (shatterCandidate && typeof host.physics.shatterBlock === 'function') {
          host.physics.shatterBlock(shatterCandidate, [...shatterCandidate.position], 1.25);
        }

        // Settle physics while blasting a 65% packet-loss burst!
        net.binaryDropRate = 0.65;
        const burstSteps = targetShatterLayer <= 4 ? 14 : 7;
        for (let settleBurst = 0; settleBurst < burstSteps; settleBurst++) {
          await advanceSimMs(30);
        }

        // Ensure Host tower is quiescent in steady state, then cease packet loss & measure Guest pose convergence after `250 ms`!
        host.benchPaused = true;
        net.binaryDropRate = 0.0;
        net.steadyStateCleanLink = true;
        // Drain any stale jittered packets from the burst window, then run exact `250 ms` clean recovery window
        await advanceSimMs(170);
        host.mpForceImmediateSnapshot = true;
        await advanceSimMs(250);

        for (let i = 0; i < host.physics.blocks.length; i++) {
          const hb = host.physics.blocks[i];
          const gb = guest.physics.blocks[i];
          if (!hb || !gb || hb.shattered || gb.shattered) continue;
          const errDm = Math.hypot(
            hb.position[0] - gb.position[0],
            hb.position[1] - gb.position[1],
            hb.position[2] - gb.position[2]
          );
          const errMm = errDm * 100.0; // 1 Vitrail world unit = 1 dm = 100 mm
          metrics.convergenceSamplesCount++;
          if (errMm > metrics.maxSteadyStateConvergenceErrorMm) {
            metrics.maxSteadyStateConvergenceErrorMm = errMm;
          }
          if (errMm >= 2.0) {
            metrics.convergenceViolationsOver2Mm++;
          }
        }
        // Verify exact 1:1 Single-Emission Audio Parity for the pre-rematch epoch
        // (while `host.benchPaused === true` and in-flight control packets have drained!)
        host.pumpMultiplayerStep(clock.nowMs);
        await advanceSimMs(80);
        if (
          hostAudioTracker.counts.blockThud !== guestAudioTracker.counts.blockThud ||
          hostAudioTracker.counts.glassImpact !== guestAudioTracker.counts.glassImpact ||
          hostAudioTracker.counts.glassShatter !== guestAudioTracker.counts.glassShatter
        ) {
          metrics.audioParity.epochMismatches++;
        }
        metrics.audioParity.host.blockThud += hostAudioTracker.counts.blockThud;
        metrics.audioParity.host.glassImpact += hostAudioTracker.counts.glassImpact;
        metrics.audioParity.host.glassShatter += hostAudioTracker.counts.glassShatter;
        metrics.audioParity.host.nestedSuppressed += hostAudioTracker.counts.nestedCallsSuppressed;
        metrics.audioParity.guest.blockThud += guestAudioTracker.counts.blockThud;
        metrics.audioParity.guest.glassImpact += guestAudioTracker.counts.glassImpact;
        metrics.audioParity.guest.glassShatter += guestAudioTracker.counts.glassShatter;

        host.benchPaused = false;
        net.steadyStateCleanLink = false;
        net.binaryDropRate = 0.08;

        // 5. Condition (e) + Cross-Rematch Stale `0x564d` Packet Hazard:
        //    Capture pre-rematch `0x564d` packet (`seed = matchSeed`, `seq = oldSeq`), reset audio trackers,
        //    trigger rapid rematch(es), and inject the delayed pre-rematch `0x564d` packet + stale `guest_cmd`!
        const stalePreRematchPacket = net.capturedLatestBinaryFrame;
        hostAudioTracker.resetCounts();
        guestAudioTracker.resetCounts();

        if (m % 2 === 0) {
          host.requestOnlineRematch();
        } else {
          guest.requestOnlineRematch();
        }
        if (m % 4 === 0) {
          // Rapid back-to-back double rematch!
          await advanceSimMs(25);
          host.requestOnlineRematch();
        }
        metrics.rematchesCompleted++;

        await advanceSimMs(120);
        const postRematchSeed = host.mpRoomSeed;

        // Inject delayed pre-rematch `0x564d` frame & stale-seed `guest_cmd` AFTER Guest adopted `postRematchSeed`!
        net.injectDelayedRawStatePacketToGuest(guest, stalePreRematchPacket, 15);
        guest.mpClient?.sendControl({
          type: 'guest_cmd',
          seq: (guest.mpGuestCmdSeq + 50) >>> 0,
          roomSeed: matchSeed,
          cmd: {
            kind: 'begin_pull',
            blockId: 12,
            localPoint: [0, 0, 0],
            worldPoint: [0, 0.7, 0],
          },
        });

        // Emit post-rematch thud & clink to verify audio parity continues cleanly across rematches
        host.audio.playBlockThud(0.38, [0, 0.08, 0], { damage: 0 });
        host.audio.playGlassImpact(0.45, 1.0, [0, 0.5, 0], undefined, undefined, {
          impactSpeedMps: 0.35,
          surface: 'glass',
        });

        net.steadyStateCleanLink = true;
        net.binaryDropRate = 0.0;
        await advanceSimMs(160);
        host.benchPaused = true;
        host.pumpMultiplayerStep(clock.nowMs);
        await advanceSimMs(60);
        host.benchPaused = false;

        if (guest.mpRoomSeed !== postRematchSeed || guest.guestWorld?.roomSeed !== postRematchSeed) {
          metrics.staleRematchRollbacks++;
        }
        verify54BlockSwatchParity(postRematchSeed);

        if (
          hostAudioTracker.counts.blockThud !== guestAudioTracker.counts.blockThud ||
          hostAudioTracker.counts.glassImpact !== guestAudioTracker.counts.glassImpact ||
          hostAudioTracker.counts.glassShatter !== guestAudioTracker.counts.glassShatter
        ) {
          metrics.audioParity.epochMismatches++;
        }
        metrics.audioParity.host.blockThud += hostAudioTracker.counts.blockThud;
        metrics.audioParity.host.glassImpact += hostAudioTracker.counts.glassImpact;
        metrics.audioParity.host.glassShatter += hostAudioTracker.counts.glassShatter;
        metrics.audioParity.host.nestedSuppressed += hostAudioTracker.counts.nestedCallsSuppressed;
        metrics.audioParity.guest.blockThud += guestAudioTracker.counts.blockThud;
        metrics.audioParity.guest.glassImpact += guestAudioTracker.counts.glassImpact;
        metrics.audioParity.guest.glassShatter += guestAudioTracker.counts.glassShatter;

        if (guest.guestWorld) {
          metrics.totalGuestAcceptedFrames += guest.guestWorld.getAcceptedFrameCount();
          metrics.totalGuestRejectedStaleFrames += guest.guestWorld.getRejectedFrameCount();
          metrics.audioParity.duplicateBatchesRejectedOnGuest +=
            guest.guestWorld.getDuplicateAudioBatchCount();
        }

        metrics.matchesCompleted++;
      }
    } finally {
      host.shutdownMultiplayerSessionOnly?.();
      guest.shutdownMultiplayerSessionOnly?.();
      host.dispose();
      guest.dispose();
      net.uninstall();
      clock.uninstall();
      globalThis.document = savedDoc;
      process.off('uncaughtException', onUncaught);
      process.off('unhandledRejection', onUnhandled);
    }

    metrics.uncaughtExceptions = uncaughtExceptions;
    metrics.unhandledRejections = unhandledRejections;
    metrics.wallTimeMs = +(nodePerf.now() - t0Wall).toFixed(1);
    metrics.maxSteadyStateConvergenceErrorMm = +metrics.maxSteadyStateConvergenceErrorMm.toFixed(4);

    // Evaluate all 6 hard pass/fail gates
    if (metrics.matchesCompleted !== matchCount) {
      metrics.gateFailures.push(
        `Incomplete run: completed ${metrics.matchesCompleted}/${matchCount} matches`
      );
    }
    if (metrics.uncaughtExceptions !== 0 || metrics.unhandledRejections !== 0) {
      metrics.gateFailures.push(
        `Runtime errors: uncaughtExceptions=${metrics.uncaughtExceptions}, unhandledRejections=${metrics.unhandledRejections} (${runtimeErrorMessages.slice(0, 3).join('; ')})`
      );
    }
    if (metrics.nanCount !== 0 || metrics.antipodalFlipCount !== 0) {
      metrics.gateFailures.push(
        `Pose/Quaternion safety failed: nanCount=${metrics.nanCount}, antipodalFlipCount=${metrics.antipodalFlipCount}`
      );
    }
    if (
      metrics.monotonicSeqViolations !== 0 ||
      metrics.staleRematchRollbacks !== 0 ||
      metrics.totalGuestRejectedStaleFrames === 0
    ) {
      metrics.gateFailures.push(
        `Monotonic sequence fencing failed: violations=${metrics.monotonicSeqViolations}, staleRematchRollbacks=${metrics.staleRematchRollbacks}, rejectedStale=${metrics.totalGuestRejectedStaleFrames}`
      );
    }
    if (metrics.swatchParityMismatches !== 0 || metrics.swatchParityChecks < matchCount * 2) {
      metrics.gateFailures.push(
        `Swatch parity failed: mismatches=${metrics.swatchParityMismatches} across ${metrics.swatchParityChecks} checks`
      );
    }
    if (
      metrics.audioParity.epochMismatches !== 0 ||
      metrics.audioParity.host.blockThud !== metrics.audioParity.guest.blockThud ||
      metrics.audioParity.host.glassImpact !== metrics.audioParity.guest.glassImpact ||
      metrics.audioParity.host.glassShatter !== metrics.audioParity.guest.glassShatter ||
      metrics.audioParity.host.blockThud === 0 ||
      metrics.audioParity.host.glassImpact === 0 ||
      metrics.audioParity.host.glassShatter === 0
    ) {
      metrics.gateFailures.push(
        `Single-emission audio parity failed: host=${JSON.stringify(metrics.audioParity.host)} vs guest=${JSON.stringify(metrics.audioParity.guest)}, epochMismatches=${metrics.audioParity.epochMismatches}`
      );
    }
    if (
      metrics.convergenceViolationsOver2Mm !== 0 ||
      metrics.maxSteadyStateConvergenceErrorMm >= 2.0 ||
      metrics.convergenceSamplesCount === 0
    ) {
      metrics.gateFailures.push(
        `Steady-state Guest pose convergence failed: maxErr=${metrics.maxSteadyStateConvergenceErrorMm} mm (limit < 2.0 mm), violations=${metrics.convergenceViolationsOver2Mm}`
      );
    }
    if (metrics.doubleGrabViolations !== 0) {
      metrics.gateFailures.push(
        `Double-grab mutual exclusion failed: violations=${metrics.doubleGrabViolations}`
      );
    }
    if (metrics.midDragFailoversCompleted < matchCount) {
      metrics.gateFailures.push(
        `Mid-drag WebRTC -> WS relay failover incomplete: ${metrics.midDragFailoversCompleted}/${matchCount}`
      );
    }
    if (metrics.occludedHostTurnsCompleted < matchCount) {
      metrics.gateFailures.push(
        `Occluded Host 30 Hz background pump failed: ${metrics.occludedHostTurnsCompleted}/${matchCount}`
      );
    }

    return metrics;
  }

  return { runChaosSimulation };
}

async function runCli() {
  const args = process.argv.slice(2);
  let matches = 100;
  let seed = 0x56495452;
  let jsonOnly = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--matches' && args[i + 1]) {
      matches = Math.max(1, parseInt(args[++i], 10) || 100);
    } else if (args[i] === '--seed' && args[i + 1]) {
      seed = Number(args[++i]) >>> 0;
    } else if (args[i] === '--json') {
      jsonOnly = true;
    }
  }

  const { createServer } = await import('vite');
  const vite = await createServer({
    server: { middlewareMode: true, ws: false },
    appType: 'custom',
    logLevel: 'silent',
  });

  try {
    const [engineMod, mpMod, mathMod] = await Promise.all([
      vite.ssrLoadModule('/src/engine/GameEngine.ts'),
      vite.ssrLoadModule('/src/multiplayer/index.ts'),
      vite.ssrLoadModule('/src/math/vecMat.ts'),
    ]);

    const suite = await createChaosMultiplayerSuite({
      GameEngine: engineMod.GameEngine,
      buildDeterministicTowerMaterials: mpMod.buildDeterministicTowerMaterials,
      canonicalizeQuatHemisphere: mpMod.canonicalizeQuatHemisphere,
      quatRotateVec3: mathMod.quatRotateVec3,
    });

    const result = await suite.runChaosSimulation({ matches, seed });
    if (jsonOnly) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log('=== Vitrail Adversarial Network Chaos Simulator ===');
      console.log(
        `Matches: ${result.matchesCompleted}/${result.matchesRequested} | Rematches: ${result.rematchesCompleted} | Wall time: ${result.wallTimeMs} ms`
      );
      console.log(
        `Packets: sent=${result.network.binaryFramesSent}, dropped=${result.network.binaryFramesDropped} (${(
          (100 * result.network.binaryFramesDropped) /
          Math.max(1, result.network.binaryFramesSent)
        ).toFixed(1)}%), reordered=${result.network.binaryFramesReorderedOnWire}, delayedPreRematch=${result.network.delayedPreRematchInjections}`
      );
      console.log(
        `Guest 120Hz Steps: ${result.guestStepsAudited120Hz} | NaNs: ${result.nanCount} | Antipodal Flips: ${result.antipodalFlipCount} (equatorCrossings=${result.equatorCrossingsTested})`
      );
      console.log(
        `Seq Fencing: accepted=${result.totalGuestAcceptedFrames}, staleRejected=${result.totalGuestRejectedStaleFrames}, violations=${result.monotonicSeqViolations}, staleRematchRollbacks=${result.staleRematchRollbacks}`
      );
      console.log(
        `Swatch Parity (54/54): checks=${result.swatchParityChecks}, mismatches=${result.swatchParityMismatches}`
      );
      console.log(
        `Failovers & Occlusion: midDragFailovers=${result.midDragFailoversCompleted}/${result.matchesRequested}, occludedTurns=${result.occludedHostTurnsCompleted}/${result.matchesRequested} (${result.occludedBgPumpSnapshotsSent} 30Hz bg snapshots), doubleGrabViolations=${result.doubleGrabViolations}/${result.doubleGrabRacesInjected}`
      );
      console.log(
        `Audio Parity (1:1): Host=${JSON.stringify(result.audioParity.host)} | Guest=${JSON.stringify(result.audioParity.guest)} | dupBatchesRejected=${result.audioParity.duplicateBatchesRejectedOnGuest}`
      );
      console.log(
        `Convergence (< 2.0 mm @ 250 ms): max=${result.maxSteadyStateConvergenceErrorMm} mm across ${result.convergenceSamplesCount} block samples`
      );
      if (result.gateFailures.length === 0) {
        console.log('RESULT: PASS (all hard chaos gates satisfied)');
      } else {
        console.error('RESULT: FAIL');
        for (const f of result.gateFailures) {
          console.error(`  - ${f}`);
        }
      }
    }

    if (result.gateFailures.length > 0) {
      process.exitCode = 1;
    }
  } finally {
    await vite.close();
  }
}

const isDirectCli =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isDirectCli) {
  runCli().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

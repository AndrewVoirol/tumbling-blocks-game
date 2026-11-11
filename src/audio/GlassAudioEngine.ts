import {
  computeGlassBarModes,
  computeHertzianContact,
  computeHertzianModeGain,
  computeHousnerRockingSchedule,
  computeOakPlateModes,
  computeShardModalVoice,
  computeShardStereoPan,
  computeStribeckSlipProfile,
  createMulberry32,
  LEADED_CAME_PRISM_DEFAULTS,
} from './physicalModalModels';

export * from './physicalModalModels';

export interface GlassImpactOptions {
  /** Normal impact closing speed in SI m/s (governs Hertzian cutoff f_c = 1 / (2 * tau_c)) */
  impactSpeedMps?: number;
  /** Structural crazing damage in [0, 1] (-1.5%..-4.1% stiffness softening + doublet beating + faster high-mode damping) */
  damage?: number;
  /** Struck substrate: 'glass' (glass-on-glass) or 'wood' (quarter-sawn white oak circular table) */
  surface?: 'glass' | 'wood';
  /** Colliding prism mass in kg (default 0.0711 kg) */
  massKg?: number;
  /** Effective local Hertzian contact curvature radius in meters (default 0.0022 m beveled crown) */
  effectiveRadiusM?: number;
}

export interface BlockThudOptions {
  massKg?: number;
  damage?: number;
  /** Normalized radial strike coordinate r / R_table in [0, 1] across the R = 43 cm circular oak table */
  radialNorm?: number;
  camPos?: [number, number, number];
  camRight?: [number, number, number];
}

export interface ShardChatterImpact {
  /** World position in Vitrail decimeters (x in [-4.3, 4.3] across R = 43 cm table) */
  pos?: [number, number, number];
  /** Optional alias if caller supplies worldPos instead of pos */
  worldPos?: [number, number, number];
  /** Closing normal impact speed in SI m/s */
  speed: number;
  /** Voronoi shard mass in SI kg (e.g. 0.0002 .. 0.015 kg) */
  massKg: number;
  /** Struck substrate: 'wood' (quarter-sawn oak knock + tinkle) or 'glass' (crystalline ping) */
  surface: 'wood' | 'glass';
  /** Optional shard pool slot index [0..511] for per-shard bounce rate-limiting */
  slotIndex?: number;
}

export interface WobbleRockOptions {
  blockId?: number;
  surface?: 'glass' | 'wood';
  damage?: number;
  camPos?: [number, number, number];
  camRight?: [number, number, number];
}

export interface SlideScrapeParams {
  /** Tangential slip speed in SI m/s */
  slideSpeedMps: number;
  /** Normal clamping load in SI Newtons (0 N = airy micro-rasp; >2 N = Stribeck stick-slip singing/chirps) */
  clampingLoadN: number;
  slipPowerMilliwatts?: number;
  torsionRadPerSec?: number;
  damage?: number;
  worldPos?: [number, number, number];
}

interface TrackedVoice {
  id: number;
  endTimeSec: number;
  priority: number;
  isShardGrainPrimary: boolean;
  release: (forceStop: boolean) => void;
}

function clampFinite(val: number, min: number, max: number, fallback: number = min): number {
  if (!Number.isFinite(val)) return fallback;
  return val < min ? min : val > max ? max : val;
}

function safeFreqHz(freqHz: number, sampleRate: number = 48000): number {
  const nyquistSafe = Math.max(2000, Math.min(19500, sampleRate * 0.46));
  return clampFinite(freqHz, 24, nyquistSafe, 440);
}

function safeGain(gain: number, maxCeiling: number = 0.75): number {
  return clampFinite(gain, 0.0001, maxCeiling, 0.0001);
}

/**
 * First-Principles Procedural Modal, Hertzian & Tribological Acoustics Engine (Web Audio API).
 *
 * Synthesizes Timoshenko-corrected leaded soda-lime stained-glass bar modes, Hertzian velocity-coupled
 * contact spectra, orthotropic circular white-oak Kirchhoff plate modes, mass-scaled (f ~ m^-1/3)
 * Voronoi shard chattering, Housner beveled-edge rocking double-taps, and Stribeck stick-slip singing
 * with zero external audio assets, deterministic PRNG buffers, strict voice-stealing, and zero leaks.
 */
export class GlassAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;

  // Persistent zero-allocation sliding & Stribeck stick-slip nodes
  private slideSource: AudioBufferSourceNode | null = null;
  private slideRaspFilter: BiquadFilterNode | null = null;
  private slideRaspGain: GainNode | null = null;
  private slideFilterA: BiquadFilterNode | null = null;
  private slideFilterB: BiquadFilterNode | null = null;
  private slideStickModGain: GainNode | null = null;
  private slideStickLfo: OscillatorNode | null = null;
  private slideStickLfoDepth: GainNode | null = null;
  private slideGain: GainNode | null = null;

  private muted: boolean = false;
  private lastClinkTime: number = -1;
  private lastShardTinkTime: number = -1;
  private lastThudTime: number = -1;

  private activeVoices: number = 0;
  private activeShardGrains: number = 0;
  private liveConnectedNodes: number = 0;
  private nextVoiceId: number = 1;
  private trackedVoices: TrackedVoice[] = [];

  /** Total concurrent transient Web Audio oscillator/buffer voices ceiling (with priority voice-stealing) */
  public readonly MAX_VOICES = 24;
  /** Strict concurrent secondary Voronoi shard micro-grain cap (<= 12) */
  public readonly MAX_SHARD_GRAINS = 12;

  private shardSlotLastTimeSec: Float64Array = new Float64Array(512).fill(-10);
  private wobbleBlockLastTimeSec: Map<number, number> = new Map();
  private shardWindowStartSec: number = -1;
  private shardWindowGrainsUsed: number = 0;

  private readonly prng = createMulberry32(0x676c6173);

  public lastTapDiagnostics: {
    pinch01: number;
    qFactor: number;
    decayMs: number;
    fundamentalHz: number;
  } | null = null;

  public lastSlideDiagnostics: {
    frictionCoeff: number;
    stickSlipIndex01: number;
    airyRaspIndex01: number;
    stickSlipPulseHz: number;
    resonatorQ: number;
    singGain: number;
    raspGain: number;
  } | null = null;

  public constructor(injectedContext?: AudioContext) {
    if (injectedContext) {
      this.attachAudioContext(injectedContext);
    }
  }

  /**
   * Attaches an AudioContext (used by unlock() in browsers or deterministic mocks in Vitest).
   */
  public attachAudioContext(ctx: AudioContext): void {
    if (this.ctx === ctx) return;
    this.ctx = ctx;
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = this.muted ? 0 : 0.65;

    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -10;
    compressor.knee.value = 12;
    compressor.ratio.value = 6;
    compressor.attack.value = 0.002;
    compressor.release.value = 0.12;

    this.masterGain.connect(compressor);
    compressor.connect(ctx.destination);

    this.initNoiseBuffer();
    this.initSlideNodes();
  }

  private sweepExpiredVoices(nowSec?: number): void {
    if (this.trackedVoices.length === 0) return;
    const now = nowSec ?? (this.ctx ? this.ctx.currentTime : 0);
    for (let i = this.trackedVoices.length - 1; i >= 0; i--) {
      const v = this.trackedVoices[i];
      if (now >= v.endTimeSec) {
        v.release(false);
      }
    }
  }

  private acquireVoiceSlot(nowSec: number, priority: number): boolean {
    this.sweepExpiredVoices(nowSec);
    if (this.activeVoices < this.MAX_VOICES) return true;

    // Steal lowest-priority / soonest-expiring voice if incoming event has >= priority
    let victimIdx = -1;
    let bestScore = Infinity;
    for (let i = 0; i < this.trackedVoices.length; i++) {
      const cand = this.trackedVoices[i];
      if (cand.priority <= priority + 0.05) {
        const remaining = Math.max(0, cand.endTimeSec - nowSec);
        const score = cand.priority * 2.0 + remaining;
        if (score < bestScore) {
          bestScore = score;
          victimIdx = i;
        }
      }
    }
    if (victimIdx >= 0) {
      this.trackedVoices[victimIdx].release(true);
      return this.activeVoices < this.MAX_VOICES;
    }
    return false;
  }

  private registerTransientVoice(
    sourceNode: AudioScheduledSourceNode,
    auxNodes: AudioNode[],
    cleanupPan: () => void,
    endTimeSec: number,
    priority: number,
    isShardGrainPrimary: boolean = false
  ): void {
    let released = false;
    const id = this.nextVoiceId++;
    this.activeVoices++;
    if (isShardGrainPrimary) {
      this.activeShardGrains++;
    }

    const release = (forceStop: boolean): void => {
      if (released) return;
      released = true;
      const idx = this.trackedVoices.findIndex((v) => v.id === id);
      if (idx >= 0) {
        this.trackedVoices.splice(idx, 1);
      }
      this.activeVoices = Math.max(0, this.activeVoices - 1);
      if (isShardGrainPrimary) {
        this.activeShardGrains = Math.max(0, this.activeShardGrains - 1);
      }
      if (forceStop) {
        try {
          sourceNode.stop();
        } catch {
          // Ignore already-stopped source
        }
      }
      try {
        sourceNode.disconnect();
      } catch {
        // Ignore
      }
      for (let i = 0; i < auxNodes.length; i++) {
        try {
          auxNodes[i].disconnect();
        } catch {
          // Ignore
        }
      }
      cleanupPan();
    };

    this.trackedVoices.push({
      id,
      endTimeSec,
      priority,
      isShardGrainPrimary,
      release,
    });
    sourceNode.onended = () => release(false);
  }

  public getActiveVoiceCount(): number {
    this.sweepExpiredVoices();
    return this.activeVoices;
  }

  public getActiveShardGrainCount(): number {
    this.sweepExpiredVoices();
    return this.activeShardGrains;
  }

  public getConnectedTransientNodes(): number {
    this.sweepExpiredVoices();
    return this.liveConnectedNodes;
  }

  private connectWithExplicitPan(source: AudioNode, panValue: number): () => void {
    if (!this.ctx || !this.masterGain) return () => {};
    let cleaned = false;
    const clampedPan = clampFinite(panValue, -0.85, 0.85, 0.0);
    if (typeof this.ctx.createStereoPanner === 'function') {
      const panner = this.ctx.createStereoPanner();
      panner.pan.value = clampedPan;
      source.connect(panner);
      panner.connect(this.masterGain);
      this.liveConnectedNodes += 2;
      return () => {
        if (cleaned) return;
        cleaned = true;
        this.liveConnectedNodes = Math.max(0, this.liveConnectedNodes - 2);
        try {
          source.disconnect();
          panner.disconnect();
        } catch {
          // Ignore already-disconnected node
        }
      };
    }
    source.connect(this.masterGain);
    this.liveConnectedNodes += 1;
    return () => {
      if (cleaned) return;
      cleaned = true;
      this.liveConnectedNodes = Math.max(0, this.liveConnectedNodes - 1);
      try {
        source.disconnect();
      } catch {
        // Ignore already-disconnected node
      }
    };
  }

  private connectPanned(
    source: AudioNode,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): () => void {
    if (!this.ctx || !this.masterGain) return () => {};
    if (!worldPos) {
      let cleaned = false;
      source.connect(this.masterGain);
      this.liveConnectedNodes += 1;
      return () => {
        if (cleaned) return;
        cleaned = true;
        this.liveConnectedNodes = Math.max(0, this.liveConnectedNodes - 1);
        try {
          source.disconnect();
        } catch {
          // Ignore
        }
      };
    }

    let pan = 0.0;
    if (camPos && camRight) {
      const dx = (worldPos[0] ?? 0) - (camPos[0] ?? 0);
      const dy = (worldPos[1] ?? 0) - (camPos[1] ?? 0);
      const dz = (worldPos[2] ?? 0) - (camPos[2] ?? 0);
      const dist = Math.max(0.6, Math.hypot(dx, dy, dz));
      const rawPan =
        (dx * (camRight[0] ?? 1) + dy * (camRight[1] ?? 0) + dz * (camRight[2] ?? 0)) /
        (dist * 0.65);
      pan = clampFinite(rawPan, -0.85, 0.85, 0.0);
    } else {
      pan = computeShardStereoPan(worldPos);
    }
    return this.connectWithExplicitPan(source, pan);
  }

  public unlock(): void {
    if (typeof window === 'undefined') return;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      this.attachAudioContext(new AudioCtx());
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
  }

  public setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.masterGain && this.ctx) {
      const now = this.ctx.currentTime;
      this.masterGain.gain.setTargetAtTime(muted ? 0 : 0.65, now, 0.02);
      if (muted) {
        if (this.slideGain) {
          this.slideGain.gain.cancelScheduledValues(now);
          this.slideGain.gain.setValueAtTime(safeGain(this.slideGain.gain.value, 0.2), now);
          this.slideGain.gain.linearRampToValueAtTime(0, now + 0.018);
        }
        if (this.slideRaspGain) {
          this.slideRaspGain.gain.cancelScheduledValues(now);
          this.slideRaspGain.gain.setValueAtTime(safeGain(this.slideRaspGain.gain.value, 0.2), now);
          this.slideRaspGain.gain.linearRampToValueAtTime(0, now + 0.018);
        }
      }
    }
  }

  public isMuted(): boolean {
    return this.muted;
  }

  private initNoiseBuffer(): void {
    if (!this.ctx) return;
    const sampleRate = this.ctx.sampleRate || 48000;
    const length = Math.max(1024, Math.floor(sampleRate * 2));
    const buffer = this.ctx.createBuffer(1, length, sampleRate);
    const data = buffer.getChannelData(0);
    const noisePrng = createMulberry32(0x56495452);
    for (let i = 0; i < length; i++) {
      data[i] = (noisePrng() * 2.0 - 1.0) * 0.92;
    }
    this.noiseBuffer = buffer;
  }

  private initSlideNodes(): void {
    if (!this.ctx || !this.noiseBuffer || !this.masterGain) return;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;
    src.loop = true;

    // 1. Airy unloaded micro-rasp branch (loose block gliding lightly across glass)
    this.slideRaspFilter = this.ctx.createBiquadFilter();
    this.slideRaspFilter.type = 'bandpass';
    this.slideRaspFilter.frequency.value = 3280;
    this.slideRaspFilter.Q.value = 4.2;

    this.slideRaspGain = this.ctx.createGain();
    this.slideRaspGain.gain.value = 0;

    // 2. Stribeck stick-slip singing resonator branch (clamped load-bearing blocks)
    this.slideFilterA = this.ctx.createBiquadFilter();
    this.slideFilterA.type = 'bandpass';
    this.slideFilterA.frequency.value = 2865;
    this.slideFilterA.Q.value = 18;

    this.slideFilterB = this.ctx.createBiquadFilter();
    this.slideFilterB.type = 'bandpass';
    this.slideFilterB.frequency.value = 4125;
    this.slideFilterB.Q.value = 22;

    this.slideStickModGain = this.ctx.createGain();
    this.slideStickModGain.gain.value = 0.72;

    if (typeof this.ctx.createOscillator === 'function') {
      this.slideStickLfo = this.ctx.createOscillator();
      this.slideStickLfo.type = 'triangle';
      this.slideStickLfo.frequency.value = 36;
      this.slideStickLfoDepth = this.ctx.createGain();
      this.slideStickLfoDepth.gain.value = 0.0;
      this.slideStickLfo.connect(this.slideStickLfoDepth);
      this.slideStickLfoDepth.connect(this.slideStickModGain.gain);
      this.slideStickLfo.start();
    }

    this.slideGain = this.ctx.createGain();
    this.slideGain.gain.value = 0;

    this.slideSource = src;
    src.connect(this.slideRaspFilter);
    this.slideRaspFilter.connect(this.slideRaspGain);
    this.slideRaspGain.connect(this.masterGain);

    src.connect(this.slideFilterA);
    src.connect(this.slideFilterB);
    this.slideFilterA.connect(this.slideStickModGain);
    this.slideFilterB.connect(this.slideStickModGain);
    this.slideStickModGain.connect(this.slideGain);
    this.slideGain.connect(this.masterGain);
    src.start();
  }

  /**
   * Physical Stribeck velocity-weakening stick-slip & airy micro-rasp continuous synthesizer.
   * - Low clampingLoadN (~0 N): smooth airy low-Q micro-rasp glide.
   * - High clampingLoadN (>2 N): high-Q modal singing & Stribeck relaxation chirps at f_ss = v / d_c.
   */
  public updateSlideScrape(params: SlideScrapeParams): void {
    const profile = computeStribeckSlipProfile(params);
    this.lastSlideDiagnostics = {
      frictionCoeff: profile.frictionCoeff,
      stickSlipIndex01: profile.stickSlipIndex01,
      airyRaspIndex01: profile.airyRaspIndex01,
      stickSlipPulseHz: profile.stickSlipPulseHz,
      resonatorQ: profile.resonatorQ,
      singGain: profile.singGain,
      raspGain: profile.raspGain,
    };

    if (!this.ctx || !this.slideGain || !this.slideFilterA || !this.slideFilterB) return;
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;

    if (this.muted || !profile.active) {
      this.slideGain.gain.cancelScheduledValues(now);
      this.slideGain.gain.setValueAtTime(clampFinite(this.slideGain.gain.value, 0, 0.2, 0), now);
      this.slideGain.gain.linearRampToValueAtTime(0, now + 0.018);
      if (this.slideRaspGain) {
        this.slideRaspGain.gain.cancelScheduledValues(now);
        this.slideRaspGain.gain.setValueAtTime(
          clampFinite(this.slideRaspGain.gain.value, 0, 0.2, 0),
          now
        );
        this.slideRaspGain.gain.linearRampToValueAtTime(0, now + 0.018);
      }
      if (this.slideStickLfoDepth) {
        this.slideStickLfoDepth.gain.setTargetAtTime(0, now, 0.015);
      }
      return;
    }

    if (this.slideRaspGain && this.slideRaspFilter) {
      this.slideRaspFilter.frequency.setTargetAtTime(
        safeFreqHz(3150 + clampFinite(params.slideSpeedMps, 0, 1.5, 0) * 2400, sr),
        now,
        0.02
      );
      this.slideRaspGain.gain.setTargetAtTime(profile.raspGain, now, 0.018);
    }

    this.slideFilterA.frequency.setTargetAtTime(safeFreqHz(profile.singingPrimaryHz, sr), now, 0.02);
    this.slideFilterA.Q.setTargetAtTime(clampFinite(profile.resonatorQ, 3.5, 46, 14), now, 0.025);
    this.slideFilterB.frequency.setTargetAtTime(safeFreqHz(profile.singingSecondaryHz, sr), now, 0.02);
    this.slideFilterB.Q.setTargetAtTime(
      clampFinite(profile.resonatorQ * 1.12, 4.0, 50, 18),
      now,
      0.025
    );

    if (this.slideStickLfo && this.slideStickLfoDepth) {
      this.slideStickLfo.frequency.setTargetAtTime(
        clampFinite(profile.stickSlipPulseHz, 14, 190, 36),
        now,
        0.02
      );
      this.slideStickLfoDepth.gain.setTargetAtTime(
        0.28 * profile.stickSlipIndex01,
        now,
        0.02
      );
    }

    this.slideGain.gain.setTargetAtTime(profile.singGain, now, 0.018);
  }

  /**
   * Backwards-compatible sliding & torsional friction updater (Vitrail decimeters/sec + resistance01),
   * routed directly through the Stribeck stick-slip & clamping-load physical model.
   */
  public updateSlideFriction(
    slideSpeedDmPerSec: number,
    resistance01: number,
    slipPowerMilliwatts: number = 0,
    torsionRadPerSec: number = 0,
    clampingLoadN?: number
  ): void {
    const speedMps = clampFinite(slideSpeedDmPerSec, 0, 30, 0) * 0.1;
    const res = clampFinite(resistance01, 0, 1, 0);
    const inferredClampN =
      clampingLoadN !== undefined
        ? clampFinite(clampingLoadN, 0, 60, 0)
        : res * 9.5 + clampFinite(slipPowerMilliwatts, 0, 100, 0) * 0.04;
    this.updateSlideScrape({
      slideSpeedMps: speedMps,
      clampingLoadN: inferredClampN,
      slipPowerMilliwatts,
      torsionRadPerSec,
    });
  }

  /**
   * Load-coupled Winkler-Timoshenko tap-test resonance for scouting loose vs bearing-clamped blocks by ear:
   * - Loose (`pinch01 = 0`): free-free Timoshenko flexural ring (`Q ~ 56`, `~190 ms` decay, `~2865 Hz` fundamental)
   * - Pinched (`pinch01 = 1`): bearing-clamped Winkler-stiffened tick (`Q ~ 14.5`, `~34 ms` decay, `~4210 Hz`)
   */
  public playTapTestResonance(
    pinch01: number,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number],
    damage: number = 0
  ): void {
    const p = clampFinite(pinch01, 0.0, 1.0, 0.0);
    const dmg = clampFinite(damage, 0.0, 1.0, 0.0);
    const clampingLoadN = p * 3.5;

    const modes = computeGlassBarModes({ clampingLoadN, damage: dmg });
    const primary = modes[0];
    const secondary = modes.find((m) => m.kind === 'flexural-y' && m.order === 2) ?? modes[2];

    const qFactor = Math.min(56.0, primary.qFactor);
    const t60Ms = 1000.0 * 6.9078 * primary.decayTauSec;
    const decayMs = clampFinite(28.0 + 0.82 * t60Ms, 28.0, 240.0, 160.0);
    const fundamentalHz = primary.freqHz;

    this.lastTapDiagnostics = {
      pinch01: p,
      qFactor,
      decayMs,
      fundamentalHz,
    };

    if (!this.ctx || !this.masterGain || this.muted) return;
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    if (!this.acquireVoiceSlot(now, 0.55)) return;

    const durSec = decayMs * 0.001;
    const detune = 0.994 + this.prng() * 0.012;
    const looseGain = 0.23 * (1.0 - 0.34 * p);

    // 1. Primary Timoshenko flexural mode
    const osc0 = this.ctx.createOscillator();
    const g0 = this.ctx.createGain();
    osc0.type = 'sine';
    osc0.frequency.value = safeFreqHz(fundamentalHz * detune, sr);
    g0.gain.setValueAtTime(0.0001, now);
    g0.gain.linearRampToValueAtTime(safeGain(looseGain), now + 0.001);
    g0.gain.exponentialRampToValueAtTime(0.0001, now + durSec);
    osc0.connect(g0);
    const cleanup0 = this.connectPanned(g0, worldPos, camPos, camRight);
    const stop0 = now + durSec + 0.008;
    this.registerTransientVoice(osc0, [g0], cleanup0, stop0, 0.55);
    osc0.start(now);
    osc0.stop(stop0);

    // 2. Timoshenko second flexural overtone (suppressed under heavy clamping or crazing)
    if (secondary && this.acquireVoiceSlot(now, 0.4)) {
      const osc1 = this.ctx.createOscillator();
      const g1 = this.ctx.createGain();
      osc1.type = 'sine';
      osc1.frequency.value = safeFreqHz(secondary.freqHz * detune, sr);
      const harmonicWeight = (0.42 * Math.pow(1.0 - p, 1.35) + 0.04) * (1.0 - 0.35 * dmg);
      const dur1 = Math.max(0.014, Math.min(durSec * 0.62, 6.9078 * secondary.decayTauSec));
      g1.gain.setValueAtTime(0.0001, now);
      g1.gain.linearRampToValueAtTime(safeGain(looseGain * harmonicWeight), now + 0.0008);
      g1.gain.exponentialRampToValueAtTime(0.0001, now + dur1);
      osc1.connect(g1);
      const cleanup1 = this.connectPanned(g1, worldPos, camPos, camRight);
      const stop1 = now + dur1 + 0.008;
      this.registerTransientVoice(osc1, [g1], cleanup1, stop1, 0.4);
      osc1.start(now);
      osc1.stop(stop1);
    }

    // 3. Lead-came / fingernail tap excitation impulse
    if (this.noiseBuffer && this.acquireVoiceSlot(now, 0.35)) {
      const tapNoise = this.ctx.createBufferSource();
      tapNoise.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = safeFreqHz(fundamentalHz * (1.04 + 0.22 * p), sr);
      bp.Q.value = clampFinite(qFactor, 5.0, 56.0, 24.0);
      const ng = this.ctx.createGain();
      const transientAmp = safeGain(0.13 + 0.2 * p);
      const transientDur = 0.017 * (1.0 - 0.55 * p) + 0.005;
      ng.gain.setValueAtTime(transientAmp, now);
      ng.gain.exponentialRampToValueAtTime(0.0001, now + transientDur);
      tapNoise.connect(bp);
      bp.connect(ng);
      const cleanupNoise = this.connectPanned(ng, worldPos, camPos, camRight);
      const stopNoise = now + transientDur + 0.012;
      this.registerTransientVoice(tapNoise, [bp, ng], cleanupNoise, stopNoise, 0.35);
      tapNoise.start(now, this.prng() * 1.4, transientDur + 0.012);
    }
  }

  /**
   * Internal helper to voice Kirchhoff circular quarter-sawn white-oak plate modes (f_01, f_11, f_21, f_02)
   * plus a short orthotropic wood-grain indentation knock transient.
   */
  private triggerOakPlateModes(
    impactSpeedMps: number,
    ampScale: number,
    radialNorm: number,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;

    const plateModes = computeOakPlateModes({ radialNorm });
    const hertzWood = computeHertzianContact({
      impactSpeedMps,
      surface: 'wood',
    });

    for (let i = 0; i < plateModes.length; i++) {
      const pm = plateModes[i];
      const modeGain =
        ampScale *
        pm.excitationWeight *
        computeHertzianModeGain(pm.freqHz, hertzWood.cutoffHz) *
        (i === 0 ? 0.95 : 0.68);
      if (modeGain < 0.004) continue;
      if (!this.acquireVoiceSlot(now, 0.52 - i * 0.06)) break;

      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = safeFreqHz(pm.freqHz * (0.995 + this.prng() * 0.01), sr);
      const dur = clampFinite(6.9078 * pm.decayTauSec, 0.025, 0.22, 0.09);

      g.gain.setValueAtTime(0.0001, now);
      g.gain.linearRampToValueAtTime(safeGain(modeGain, 0.42), now + 0.0014);
      g.gain.exponentialRampToValueAtTime(0.0001, now + dur);

      osc.connect(g);
      const cleanup = this.connectPanned(g, worldPos, camPos, camRight);
      const stopT = now + dur + 0.008;
      this.registerTransientVoice(osc, [g], cleanup, stopT, 0.52 - i * 0.06);
      osc.start(now);
      osc.stop(stopT);
    }

    // Short woody radial-grain indentation knock transient
    if (this.noiseBuffer && ampScale > 0.02 && this.acquireVoiceSlot(now, 0.45)) {
      const knock = this.ctx.createBufferSource();
      knock.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = safeFreqHz(Math.min(1350, hertzWood.cutoffHz * 1.15), sr);
      bp.Q.value = 3.2;
      const kg = this.ctx.createGain();
      kg.gain.setValueAtTime(safeGain(ampScale * 0.55, 0.35), now);
      kg.gain.exponentialRampToValueAtTime(0.0001, now + 0.009);
      knock.connect(bp);
      bp.connect(kg);
      const cleanupKnock = this.connectPanned(kg, worldPos, camPos, camRight);
      const stopKnock = now + 0.014;
      this.registerTransientVoice(knock, [bp, kg], cleanupKnock, stopKnock, 0.45);
      knock.start(now, this.prng() * 1.3, 0.014);
    }
  }

  /**
   * Quarter-sawn white oak circular table (`R = 43 cm`, `E_L = 12 GPa`, `E_R = 1.6 GPa`) impact resonance
   * coupled with muted leaded stained-glass bar modes when a block strikes or lands on the oak tabletop.
   */
  public playBlockThud(
    impactSpeedMps: number,
    worldPos?: [number, number, number],
    camPosOrOptions?: [number, number, number] | BlockThudOptions,
    camRight?: [number, number, number],
    damage: number = 0
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const now = this.ctx.currentTime;
    const vn = clampFinite(impactSpeedMps, 0.015, 6.0, 0.25);
    if (now - this.lastThudTime < 0.012 && vn < 0.35) return;
    this.lastThudTime = now;

    let resolvedCamPos: [number, number, number] | undefined;
    let resolvedCamRight: [number, number, number] | undefined = camRight;
    let resolvedDamage = damage;
    let resolvedMassKg: number = LEADED_CAME_PRISM_DEFAULTS.massKg;
    let radialNorm = worldPos
      ? clampFinite(Math.hypot(worldPos[0] ?? 0, worldPos[2] ?? 0) / 4.3, 0, 1, 0.35)
      : 0.35;

    if (Array.isArray(camPosOrOptions)) {
      resolvedCamPos = camPosOrOptions;
    } else if (camPosOrOptions && typeof camPosOrOptions === 'object') {
      resolvedCamPos = camPosOrOptions.camPos;
      resolvedCamRight = camPosOrOptions.camRight ?? camRight;
      resolvedDamage = camPosOrOptions.damage ?? damage;
      resolvedMassKg = camPosOrOptions.massKg ?? resolvedMassKg;
      if (camPosOrOptions.radialNorm !== undefined) {
        radialNorm = clampFinite(camPosOrOptions.radialNorm, 0, 1, radialNorm);
      }
    }

    const energyNorm = clampFinite(
      Math.pow(vn / 1.15, 0.85) * (resolvedMassKg / LEADED_CAME_PRISM_DEFAULTS.massKg),
      0.03,
      1.0,
      0.25
    );
    const woodAmp = clampFinite(0.045 + 0.28 * energyNorm, 0.02, 0.34, 0.14);

    // 1. Axisymmetric & asymmetric Kirchhoff white-oak circular plate modes + woody knock
    this.triggerOakPlateModes(
      vn,
      woodAmp,
      radialNorm,
      worldPos,
      resolvedCamPos,
      resolvedCamRight
    );

    // 2. Coupled muted leaded-glass prism modes filtered by the softer glass-on-oak Hertzian cutoff
    this.playGlassImpact(
      energyNorm * 0.68,
      0.98,
      worldPos,
      resolvedCamPos,
      resolvedCamRight,
      {
        impactSpeedMps: vn,
        damage: resolvedDamage,
        surface: 'wood',
        massKg: resolvedMassKg,
      }
    );
  }

  /**
   * Timoshenko-corrected modal synthesis of a 7.5 x 2.5 x 1.5 cm leaded soda-lime stained-glass prism
   * excited by a Hertzian half-sine contact force pulse (tau_c ~ v_n^(-1/5)).
   */
  public playGlassImpact(
    intensity01: number,
    pitchScale: number = 1.0,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number],
    options?: GlassImpactOptions
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const intensity = clampFinite(intensity01, 0.0, 1.0, 0.0);
    if (intensity <= 0.002 && !options?.impactSpeedMps) return;

    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    if (now - this.lastClinkTime < 0.012 && intensity < 0.48) return;
    this.lastClinkTime = now;

    const pScale = clampFinite(pitchScale, 0.5, 2.5, 1.0);
    // Legacy MujocoPhysicsEngine passes pitchScale = 0.92 when bestHitTable === true
    const inferredWoodFromPitch = !options?.surface && pScale <= 0.94;
    const surface: 'glass' | 'wood' = options?.surface ?? (inferredWoodFromPitch ? 'wood' : 'glass');
    const vnMps = clampFinite(
      options?.impactSpeedMps ?? 0.035 + intensity * 1.65,
      0.015,
      5.0,
      0.25
    );
    const damage = clampFinite(options?.damage ?? 0.0, 0.0, 1.0, 0.0);
    const massKg = clampFinite(
      options?.massKg ?? LEADED_CAME_PRISM_DEFAULTS.massKg,
      0.0002,
      0.5,
      LEADED_CAME_PRISM_DEFAULTS.massKg
    );

    const hertz = computeHertzianContact({
      impactSpeedMps: vnMps,
      surface,
      bodyMassKg: massKg,
      effectiveRadiusM: options?.effectiveRadiusM,
    });

    // Select 4 distinct physical modes: flexural-y n=1, torsional n=1, flexural-y n=2, flexural-y n=3
    const allModes = computeGlassBarModes({ damage, massKg });
    const selectedModes = [
      allModes.find((m) => m.kind === 'flexural-y' && m.order === 1) ?? allModes[0],
      allModes.find((m) => m.kind === 'torsional' && m.order === 1) ?? allModes[1],
      allModes.find((m) => m.kind === 'flexural-y' && m.order === 2) ?? allModes[3],
      allModes.find((m) => m.kind === 'flexural-y' && m.order === 3) ?? allModes[5],
    ].filter((m): m is NonNullable<typeof m> => Boolean(m));

    const gainBase = clampFinite(0.018 + intensity * 0.33, 0.015, 0.36, 0.12);
    const detune = 0.988 + this.prng() * 0.024;
    const priorityBase = 0.45 + 0.35 * intensity;

    for (let i = 0; i < selectedModes.length; i++) {
      const m = selectedModes[i];
      const scaledFreq = m.freqHz * (inferredWoodFromPitch ? 1.0 : pScale) * detune;
      const hertzWeight = computeHertzianModeGain(scaledFreq, hertz.cutoffHz);
      const modeAmp = gainBase * m.relativeWeight * hertzWeight;
      if (modeAmp < 0.0025) continue;
      if (!this.acquireVoiceSlot(now, priorityBase - i * 0.05)) break;

      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = safeFreqHz(scaledFreq, sr);

      // Solder-came viscoelastic + crazing decay (-80 dB ~ 6.91 * tau_k)
      const dur = clampFinite(
        (6.9078 * m.decayTauSec) / Math.sqrt(pScale),
        0.009,
        0.21,
        0.065
      );
      g.gain.setValueAtTime(0.0001, now);
      g.gain.linearRampToValueAtTime(safeGain(modeAmp, 0.38), now + 0.001);
      g.gain.exponentialRampToValueAtTime(0.0001, now + dur);

      osc.connect(g);
      const cleanup = this.connectPanned(g, worldPos, camPos, camRight);
      const stopT = now + dur + 0.008;
      this.registerTransientVoice(osc, [g], cleanup, stopT, priorityBase - i * 0.05);
      osc.start(now);
      osc.stop(stopT);

      // Crazed glass doublet-splitting beat partner on the primary flexural mode
      if (i === 0 && damage > 0.12 && m.splitBeatHz > 1.0 && this.acquireVoiceSlot(now, priorityBase - 0.08)) {
        const oscSplit = this.ctx.createOscillator();
        const gSplit = this.ctx.createGain();
        oscSplit.type = 'sine';
        oscSplit.frequency.value = safeFreqHz(
          m.doubletFreqHz * (inferredWoodFromPitch ? 1.0 : pScale) * detune,
          sr
        );
        const splitDur = Math.max(0.012, dur * 0.86);
        gSplit.gain.setValueAtTime(0.0001, now);
        gSplit.gain.linearRampToValueAtTime(
          safeGain(modeAmp * (0.38 + 0.35 * damage), 0.25),
          now + 0.0012
        );
        gSplit.gain.exponentialRampToValueAtTime(0.0001, now + splitDur);
        oscSplit.connect(gSplit);
        const cleanupSplit = this.connectPanned(gSplit, worldPos, camPos, camRight);
        const stopSplit = now + splitDur + 0.008;
        this.registerTransientVoice(
          oscSplit,
          [gSplit],
          cleanupSplit,
          stopSplit,
          priorityBase - 0.08
        );
        oscSplit.start(now);
        oscSplit.stop(stopSplit);
      }
    }

    // Hertzian-shaped contact impulse: warm lead-came tap at low v_n vs bright bevel click at high v_n
    if (this.noiseBuffer && intensity > 0.06 && this.acquireVoiceSlot(now, priorityBase - 0.12)) {
      const click = this.ctx.createBufferSource();
      click.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = safeFreqHz(clampFinite(hertz.cutoffHz * 1.15, 650, 7200, 2400), sr);
      bp.Q.value = vnMps < 0.2 ? 2.4 : 4.2;
      const cg = this.ctx.createGain();
      const pulseDur = clampFinite(Math.max(0.004, hertz.contactTimeSec * 14.0), 0.004, 0.014, 0.006);
      cg.gain.setValueAtTime(safeGain(gainBase * 0.42, 0.25), now);
      cg.gain.exponentialRampToValueAtTime(0.0001, now + pulseDur);
      click.connect(bp);
      bp.connect(cg);
      const cleanupClick = this.connectPanned(cg, worldPos, camPos, camRight);
      const stopClick = now + pulseDur + 0.008;
      this.registerTransientVoice(click, [bp, cg], cleanupClick, stopClick, priorityBase - 0.12);
      click.start(now, this.prng() * 1.5, pulseDur + 0.008);
    }

    // Automatically voice circular white-oak Kirchhoff plate modes when legacy caller hits table (pitchScale <= 0.94)
    if (inferredWoodFromPitch) {
      const radialNorm = worldPos
        ? clampFinite(Math.hypot(worldPos[0] ?? 0, worldPos[2] ?? 0) / 4.3, 0, 1, 0.35)
        : 0.35;
      this.triggerOakPlateModes(vnMps, gainBase * 0.62, radialNorm, worldPos, camPos, camRight);
    }
  }

  /**
   * Housner (1963) beveled-edge rocking / wobble micro-clink train when a tilted block rocks back flat.
   * Returns the number of micro-clink taps scheduled.
   */
  public playWobbleRock(
    tiltAngleRad: number,
    angularSpeedRadPerSec: number = 0.6,
    clampingLoadN: number = 0.7,
    worldPos?: [number, number, number],
    options?: WobbleRockOptions
  ): number {
    const surface = options?.surface ?? 'glass';
    const schedule = computeHousnerRockingSchedule(
      tiltAngleRad,
      angularSpeedRadPerSec,
      clampingLoadN,
      surface
    );
    if (schedule.length === 0) return 0;
    if (!this.ctx || !this.masterGain || this.muted) return 0;

    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    if (options?.blockId !== undefined) {
      const prevT = this.wobbleBlockLastTimeSec.get(options.blockId) ?? -10;
      if (now - prevT < 0.065) return 0;
      this.wobbleBlockLastTimeSec.set(options.blockId, now);
    }

    const damage = clampFinite(options?.damage ?? 0.0, 0.0, 1.0, 0.0);
    const modes = computeGlassBarModes({ damage, clampingLoadN });
    const fPrimary = modes[0]?.freqHz ?? 2865;
    const tauPrimary = modes[0]?.decayTauSec ?? 0.024;
    const fTorsion = modes.find((m) => m.kind === 'torsional')?.freqHz ?? 4125;

    let scheduledCount = 0;
    for (let k = 0; k < schedule.length; k++) {
      const tap = schedule[k];
      const startT = now + tap.delaySec;
      if (!this.acquireVoiceSlot(now, 0.36 - k * 0.05)) break;

      const hertzGain0 = computeHertzianModeGain(fPrimary, tap.cutoffHz);
      const amp = safeGain((0.022 + 0.14 * tap.intensity01) * Math.max(0.25, hertzGain0), 0.22);
      const dur = clampFinite(4.8 * tauPrimary * (1.0 - 0.14 * k), 0.014, 0.095, 0.042);

      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = safeFreqHz(fPrimary * tap.pitchScale * (0.994 + this.prng() * 0.012), sr);
      g.gain.setValueAtTime(0.0001, startT);
      g.gain.linearRampToValueAtTime(amp, startT + 0.0009);
      g.gain.exponentialRampToValueAtTime(0.0001, startT + dur);

      osc.connect(g);
      const cleanup = this.connectPanned(g, worldPos, options?.camPos, options?.camRight);
      const stopT = startT + dur + 0.006;
      this.registerTransientVoice(osc, [g], cleanup, stopT, 0.36 - k * 0.05);
      osc.start(startT);
      osc.stop(stopT);
      scheduledCount++;

      // Subtle bevel edge torsional partial on initial hit if velocity opens upper Hertzian lobe
      const hertzGainT = computeHertzianModeGain(fTorsion, tap.cutoffHz);
      if (k === 0 && hertzGainT > 0.18 && this.acquireVoiceSlot(now, 0.28)) {
        const oscT = this.ctx.createOscillator();
        const gT = this.ctx.createGain();
        oscT.type = 'sine';
        oscT.frequency.value = safeFreqHz(fTorsion * tap.pitchScale, sr);
        const durT = Math.max(0.01, dur * 0.52);
        gT.gain.setValueAtTime(0.0001, startT);
        gT.gain.linearRampToValueAtTime(safeGain(amp * 0.45 * hertzGainT, 0.12), startT + 0.0007);
        gT.gain.exponentialRampToValueAtTime(0.0001, startT + durT);
        oscT.connect(gT);
        const cleanupT = this.connectPanned(gT, worldPos, options?.camPos, options?.camRight);
        const stopTors = startT + durT + 0.006;
        this.registerTransientVoice(oscT, [gT], cleanupT, stopTors, 0.28);
        oscT.start(startT);
        oscT.stop(stopTors);
      }
    }

    return scheduledCount;
  }

  /**
   * Internal stress craze / sub-critical Griffith cleavage snap + crazed doublet-split ring.
   */
  public playGlassCrack(
    severity01: number = 0.7,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.muted) return;
    const sev = clampFinite(severity01, 0.05, 1.0, 0.7);
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;

    if (this.acquireVoiceSlot(now, 0.78)) {
      const snap = this.ctx.createBufferSource();
      snap.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = safeFreqHz(5100 + 1100 * sev, sr);
      bp.Q.value = 4.8;

      const g = this.ctx.createGain();
      const peak = safeGain(0.18 + sev * 0.24, 0.44);
      g.gain.setValueAtTime(peak, now);
      g.gain.exponentialRampToValueAtTime(0.0001, now + 0.022);

      snap.connect(bp);
      bp.connect(g);
      const cleanupSnap = this.connectPanned(g, worldPos, camPos, camRight);
      const stopSnap = now + 0.028;
      this.registerTransientVoice(snap, [bp, g], cleanupSnap, stopSnap, 0.78);
      snap.start(now, this.prng() * 1.2, 0.028);
    }

    this.playGlassImpact(0.46 * sev, 1.0, worldPos, camPos, camRight, {
      impactSpeedMps: 0.55 + 0.85 * sev,
      damage: clampFinite(Math.max(0.38, sev), 0.38, 0.95, 0.65),
      surface: 'glass',
    });
  }

  /**
   * Full 3D Voronoi brittle glass fracture: Griffith cleavage acoustic burst + Kirchhoff white-oak
   * circular plate thud (`hitTable`) + initial mass-distributed (`f ~ m^-1/3`) shard cascade.
   */
  public playGlassShatter(
    energy01: number = 1.0,
    hitTable: boolean = true,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.muted) return;
    const energy = clampFinite(energy01, 0.05, 1.0, 1.0);
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    const amp = safeGain(0.25 + energy * 0.26, 0.52);

    // 1. Brittle Griffith cleavage acoustic emission snap (3.2-7.5 kHz burst)
    if (this.acquireVoiceSlot(now, 0.95)) {
      const burst = this.ctx.createBufferSource();
      burst.buffer = this.noiseBuffer;
      const hp = this.ctx.createBiquadFilter();
      hp.type = 'bandpass';
      hp.frequency.value = safeFreqHz(4200 + 1200 * energy, sr);
      hp.Q.value = 1.15;

      const bg = this.ctx.createGain();
      bg.gain.setValueAtTime(amp, now);
      bg.gain.exponentialRampToValueAtTime(0.0001, now + 0.04);

      burst.connect(hp);
      hp.connect(bg);
      const cleanupBurst = this.connectPanned(bg, worldPos, camPos, camRight);
      const stopBurst = now + 0.05;
      this.registerTransientVoice(burst, [hp, bg], cleanupBurst, stopBurst, 0.95);
      burst.start(now, this.prng(), 0.05);
    }

    // 2. Coupled quarter-sawn white oak circular plate resonance when smashing onto tabletop
    if (hitTable) {
      const radialNorm = worldPos
        ? clampFinite(Math.hypot(worldPos[0] ?? 0, worldPos[2] ?? 0) / 4.3, 0, 1, 0.35)
        : 0.35;
      this.triggerOakPlateModes(
        0.65 + 1.45 * energy,
        amp * 0.62,
        radialNorm,
        worldPos,
        camPos,
        camRight
      );
    }

    // 3. Initial mass-distributed Voronoi cleavage shard spray (0.28 g .. 6.2 g -> f_0 ~ m^-1/3)
    const shardMassesKg = [0.0058, 0.0029, 0.0014, 0.00065, 0.00028];
    const tinkCount = Math.min(shardMassesKg.length, Math.floor(3 + energy * 2.5));
    const basePos: [number, number, number] = worldPos ? [...worldPos] : [0, 0.08, 0];

    for (let i = 0; i < tinkCount; i++) {
      if (!this.acquireVoiceSlot(now, 0.72 - i * 0.05)) break;
      const delay = 0.006 + Math.pow((i + 0.35 * this.prng()) / tinkCount, 1.45) * 0.125;
      const startT = now + delay;
      const massKg = shardMassesKg[i] * (0.85 + 0.3 * this.prng());
      const spec = computeShardModalVoice(
        {
          pos: [basePos[0] + (this.prng() - 0.5) * 0.6, basePos[1], basePos[2]],
          speed: 0.55 + energy * 1.15,
          massKg,
          surface: hitTable && i % 2 === 1 ? 'wood' : 'glass',
        },
        camPos,
        camRight
      );

      const dur = clampFinite(4.2 * spec.decayTauSec, 0.018, 0.075, 0.036);
      const tinkAmp = safeGain(amp * (0.16 + 0.1 * this.prng()) * (1.0 - i / (tinkCount + 3)), 0.24);

      const osc0 = this.ctx.createOscillator();
      const g0 = this.ctx.createGain();
      osc0.type = 'sine';
      osc0.frequency.value = safeFreqHz(spec.fundamentalHz, sr);
      g0.gain.setValueAtTime(0.0001, startT);
      g0.gain.linearRampToValueAtTime(tinkAmp, startT + 0.0007);
      g0.gain.exponentialRampToValueAtTime(0.0001, startT + dur);
      osc0.connect(g0);
      const cleanup0 = this.connectWithExplicitPan(g0, spec.stereoPan);
      const stop0 = startT + dur + 0.006;
      this.registerTransientVoice(osc0, [g0], cleanup0, stop0, 0.72 - i * 0.05);
      osc0.start(startT);
      osc0.stop(stop0);

      if (spec.overtoneGainRatio > 0.12 && this.acquireVoiceSlot(now, 0.58 - i * 0.05)) {
        const osc1 = this.ctx.createOscillator();
        const g1 = this.ctx.createGain();
        osc1.type = 'sine';
        osc1.frequency.value = safeFreqHz(spec.overtoneHz, sr);
        const dur1 = Math.max(0.01, dur * 0.55);
        g1.gain.setValueAtTime(0.0001, startT);
        g1.gain.linearRampToValueAtTime(
          safeGain(tinkAmp * spec.overtoneGainRatio, 0.14),
          startT + 0.0006
        );
        g1.gain.exponentialRampToValueAtTime(0.0001, startT + dur1);
        osc1.connect(g1);
        const cleanup1 = this.connectWithExplicitPan(g1, spec.stereoPan);
        const stop1 = startT + dur1 + 0.006;
        this.registerTransientVoice(osc1, [g1], cleanup1, stop1, 0.58 - i * 0.05);
        osc1.start(startT);
        osc1.stop(stop1);
      }
    }
  }

  /**
   * Rate-limited, polyphony-capped (`<= 12` concurrent micro-grains) secondary 3D Voronoi shard
   * bounce synthesis driven by shard mass `m_i` (`f_0 ~ m_i^-1/3`, `2.2 kHz - 9.5 kHz` across `0.2 g - 8 g`)
   * and substrate (`'wood'` oak knock + tinkle vs `'glass'` crystalline ping) with stereo pan
   * `pan = clamp(x / 4.5, -0.85, 0.85)`.
   *
   * Returns the number of shard micro-grains scheduled.
   */
  public playShardImpacts(
    impacts: ReadonlyArray<ShardChatterImpact>,
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): number {
    if (!this.ctx || !this.masterGain || this.muted || impacts.length === 0) return 0;

    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    this.sweepExpiredVoices(now);

    if (this.shardWindowStartSec < 0 || now - this.shardWindowStartSec >= 0.045) {
      this.shardWindowStartSec = now;
      this.shardWindowGrainsUsed = 0;
    }

    // Filter non-finite / sub-audible micro-jitter & rank descending by kinetic energy 0.5 * m * v^2
    const candidates: Array<{
      pos: [number, number, number];
      speed: number;
      massKg: number;
      surface: 'wood' | 'glass';
      slotIndex?: number;
      ke: number;
    }> = [];

    for (let i = 0; i < impacts.length; i++) {
      const imp = impacts[i];
      if (!imp) continue;
      const speed = imp.speed;
      const massKg = imp.massKg;
      if (!Number.isFinite(speed) || !Number.isFinite(massKg)) continue;
      if (speed < 0.022 || massKg <= 0) continue;

      if (
        imp.slotIndex !== undefined &&
        Number.isInteger(imp.slotIndex) &&
        imp.slotIndex >= 0 &&
        imp.slotIndex < this.shardSlotLastTimeSec.length
      ) {
        if (now - this.shardSlotLastTimeSec[imp.slotIndex] < 0.018) continue;
      }

      const rawPos = imp.pos ?? imp.worldPos ?? [0, 0.05, 0];
      const pos: [number, number, number] = [
        clampFinite(rawPos[0], -20, 20, 0),
        clampFinite(rawPos[1], -20, 20, 0.05),
        clampFinite(rawPos[2], -20, 20, 0),
      ];
      const mClamped = clampFinite(massKg, 0.00005, 0.08, 0.001);
      const sClamped = clampFinite(speed, 0.022, 10.0, 0.2);
      candidates.push({
        pos,
        speed: sClamped,
        massKg: mClamped,
        surface: imp.surface === 'wood' ? 'wood' : 'glass',
        slotIndex: imp.slotIndex,
        ke: 0.5 * mClamped * sClamped * sClamped,
      });
    }

    if (candidates.length === 0) return 0;
    if (candidates.length > 1) {
      candidates.sort((a, b) => b.ke - a.ke);
    }

    let scheduledGrains = 0;

    for (let i = 0; i < candidates.length; i++) {
      if (this.activeShardGrains >= this.MAX_SHARD_GRAINS) break;
      if (this.shardWindowGrainsUsed >= this.MAX_SHARD_GRAINS) break;

      const ev = candidates[i];
      const priority = 0.22 + Math.min(0.25, ev.ke * 40.0);
      if (!this.acquireVoiceSlot(now, priority)) break;

      if (
        ev.slotIndex !== undefined &&
        ev.slotIndex >= 0 &&
        ev.slotIndex < this.shardSlotLastTimeSec.length
      ) {
        this.shardSlotLastTimeSec[ev.slotIndex] = now;
      }

      const spec = computeShardModalVoice(ev, camPos, camRight);
      const startT = now + scheduledGrains * 0.0018;
      const dur0 = clampFinite(4.6 * spec.decayTauSec, 0.016, 0.085, 0.036);
      const peak0 = safeGain(spec.peakAmplitude, 0.2);

      // Primary mass-scaled (f_0 ~ m^-1/3) crystalline mode (counts as 1 shard micro-grain)
      // Deterministic pitch per shard mass (zero per-bounce random pitch drift!)
      const osc0 = this.ctx.createOscillator();
      const g0 = this.ctx.createGain();
      osc0.type = 'sine';
      osc0.frequency.value = safeFreqHz(spec.fundamentalHz, sr);
      g0.gain.setValueAtTime(0.0001, startT);
      g0.gain.linearRampToValueAtTime(peak0, startT + 0.0006);
      g0.gain.exponentialRampToValueAtTime(0.0001, startT + dur0);

      osc0.connect(g0);
      const cleanup0 = this.connectWithExplicitPan(g0, spec.stereoPan);
      const stop0 = startT + dur0 + 0.006;
      this.registerTransientVoice(osc0, [g0], cleanup0, stop0, priority, true);
      osc0.start(startT);
      osc0.stop(stop0);

      scheduledGrains++;
      this.shardWindowGrainsUsed++;

      // Companion secondary timbre: woody oak tap on 'wood' vs bright 2.29x crystalline partial on 'glass'
      if (this.activeVoices < this.MAX_VOICES) {
        if (ev.surface === 'wood' && spec.woodKnockGainRatio > 0.05) {
          const oscWood = this.ctx.createOscillator();
          const gWood = this.ctx.createGain();
          oscWood.type = 'triangle';
          oscWood.frequency.value = safeFreqHz(spec.woodKnockHz, sr);
          const durWood = 0.014;
          gWood.gain.setValueAtTime(0.0001, startT);
          gWood.gain.linearRampToValueAtTime(
            safeGain(peak0 * spec.woodKnockGainRatio, 0.1),
            startT + 0.0005
          );
          gWood.gain.exponentialRampToValueAtTime(0.0001, startT + durWood);
          oscWood.connect(gWood);
          const cleanupWood = this.connectWithExplicitPan(gWood, spec.stereoPan);
          const stopWood = startT + durWood + 0.005;
          this.registerTransientVoice(oscWood, [gWood], cleanupWood, stopWood, priority - 0.05, false);
          oscWood.start(startT);
          oscWood.stop(stopWood);
        } else if (spec.overtoneGainRatio > 0.08) {
          const osc1 = this.ctx.createOscillator();
          const g1 = this.ctx.createGain();
          osc1.type = 'sine';
          osc1.frequency.value = safeFreqHz(spec.overtoneHz, sr);
          const dur1 = Math.max(0.009, dur0 * 0.52);
          g1.gain.setValueAtTime(0.0001, startT);
          g1.gain.linearRampToValueAtTime(
            safeGain(peak0 * spec.overtoneGainRatio, 0.12),
            startT + 0.0005
          );
          g1.gain.exponentialRampToValueAtTime(0.0001, startT + dur1);
          osc1.connect(g1);
          const cleanup1 = this.connectWithExplicitPan(g1, spec.stereoPan);
          const stop1 = startT + dur1 + 0.005;
          this.registerTransientVoice(osc1, [g1], cleanup1, stop1, priority - 0.05, false);
          osc1.start(startT);
          osc1.stop(stop1);
        }
      }
    }

    return scheduledGrains;
  }

  /**
   * Single-or-batch convenience alias for secondary 3D Voronoi shard chattering.
   */
  public playShardChatter(
    impacts: ShardChatterImpact | ReadonlyArray<ShardChatterImpact>,
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): number {
    const list = Array.isArray(impacts) ? impacts : [impacts];
    return this.playShardImpacts(list, camPos, camRight);
  }

  /**
   * Backwards-compatible ranked shard collision entrypoint (`volumeDm3` -> `massKg = 2.50 * volumeDm3`),
   * routed directly through the `<= 12`-grain `m^-1/3` physical shard chatter pool.
   */
  public playRankedShardCollisions(
    events: Array<{
      energy: number;
      intensity01: number;
      volumeDm3: number;
      worldPos: [number, number, number];
      hitGlass?: boolean;
    }>,
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || this.muted || events.length === 0) return;
    const mapped: ShardChatterImpact[] = new Array(events.length);
    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const volDm3 = clampFinite(ev.volumeDm3, 0.00004, 0.03, 0.0008);
      const massKg = volDm3 * 2.5; // 2.50 kg/dm^3 soda-lime glass density
      const speedMps = 0.08 + clampFinite(ev.intensity01, 0, 1, 0.25) * 1.45;
      mapped[i] = {
        pos: ev.worldPos,
        speed: speedMps,
        massKg,
        surface: ev.hitGlass ? 'glass' : 'wood',
      };
    }
    this.playShardImpacts(mapped, camPos, camRight);
  }

  /**
   * Backwards-compatible single shard tabletop clink helper.
   */
  public playShardTableTink(
    intensity01: number,
    volumeDm3: number,
    worldPos: [number, number, number] = [0, 0.05, 0],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const now = this.ctx.currentTime;
    if (now - this.lastShardTinkTime < 0.014) return;
    this.lastShardTinkTime = now;
    this.playRankedShardCollisions(
      [{ energy: intensity01 * volumeDm3, intensity01, volumeDm3, worldPos, hitGlass: false }],
      camPos,
      camRight
    );
  }

  /**
   * Subtle crystalline harmonic chime when a stacked block settles cleanly onto the top of the tower.
   */
  public playStackSuccess(layerCount: number): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    const notes = [1046.5, 1318.5]; // C6 -> E6 gentle glass bell
    const pitchShift = Math.min(1.25, 1 + Math.max(0, layerCount - 18) * 0.015);

    notes.forEach((f, idx) => {
      if (!this.ctx || !this.masterGain || !this.acquireVoiceSlot(now, 0.6)) return;
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = safeFreqHz(f * pitchShift, sr);
      const t0 = now + idx * 0.075;
      const stopT = t0 + 0.38;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.linearRampToValueAtTime(0.075, t0 + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.36);
      osc.connect(g);
      const cleanupBell = this.connectPanned(g);
      this.registerTransientVoice(osc, [g], cleanupBell, stopT, 0.6);
      osc.start(t0);
      osc.stop(stopT);
    });
  }

  /**
   * Tactile cut-glass prism key-down transient (brass/pewter came seat + polished bevel click).
   */
  public playUiKeyDown(
    tone: 'default' | 'tab' | 'primary' | 'danger' | 'step' = 'default'
  ): void {
    this.unlock();
    if (!this.ctx || !this.masterGain || this.muted) return;

    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    const pitchMap: Record<typeof tone, number> = {
      default: 1.0,
      tab: 1.08,
      primary: 0.94,
      danger: 0.82,
      step: 1.18,
    };
    const pScale = pitchMap[tone] ?? 1.0;

    if (this.acquireVoiceSlot(now, 0.5)) {
      const seat = this.ctx.createOscillator();
      const sg = this.ctx.createGain();
      seat.type = 'triangle';
      seat.frequency.setValueAtTime(safeFreqHz(210 * pScale, sr), now);
      seat.frequency.exponentialRampToValueAtTime(safeFreqHz(92 * pScale, sr), now + 0.022);
      sg.gain.setValueAtTime(0.068, now);
      sg.gain.exponentialRampToValueAtTime(0.0001, now + 0.024);
      seat.connect(sg);
      const cleanupSeat = this.connectPanned(sg);
      const stopSeat = now + 0.028;
      this.registerTransientVoice(seat, [sg], cleanupSeat, stopSeat, 0.5);
      seat.start(now);
      seat.stop(stopSeat);
    }

    if (this.noiseBuffer && this.acquireVoiceSlot(now, 0.45)) {
      const tick = this.ctx.createBufferSource();
      tick.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = safeFreqHz(3450 * pScale, sr);
      bp.Q.value = 8.5;
      const tg = this.ctx.createGain();
      tg.gain.setValueAtTime(0.075, now);
      tg.gain.exponentialRampToValueAtTime(0.0001, now + 0.008);
      tick.connect(bp);
      bp.connect(tg);
      const cleanupTick = this.connectPanned(tg);
      const stopTick = now + 0.012;
      this.registerTransientVoice(tick, [bp, tg], cleanupTick, stopTick, 0.45);
      tick.start(now, this.prng() * 1.2, 0.012);
    }
  }

  /**
   * Delicate crystalline glass key-up harmonic chime when a pressed UI glass block rebounds.
   */
  public playUiKeyUp(
    tone: 'default' | 'tab' | 'primary' | 'danger' | 'step' = 'default'
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const now = this.ctx.currentTime;
    const sr = this.ctx.sampleRate || 48000;
    if (!this.acquireVoiceSlot(now, 0.5)) return;

    const freqMap: Record<typeof tone, number> = {
      default: 2480,
      tab: 2640,
      primary: 2220,
      danger: 1960,
      step: 2940,
    };
    const f0 = (freqMap[tone] ?? 2480) * (0.992 + this.prng() * 0.016);
    const dur = tone === 'primary' ? 0.072 : 0.048;

    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(safeFreqHz(f0, sr), now);
    osc.frequency.exponentialRampToValueAtTime(safeFreqHz(f0 * 1.018, sr), now + dur);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(tone === 'primary' ? 0.062 : 0.045, now + 0.0012);
    g.gain.exponentialRampToValueAtTime(0.0001, now + dur);

    osc.connect(g);
    const cleanup = this.connectPanned(g);
    const stopT = now + dur + 0.008;
    this.registerTransientVoice(osc, [g], cleanup, stopT, 0.5);
    osc.start(now);
    osc.stop(stopT);
  }

  public dispose(): void {
    for (let i = this.trackedVoices.length - 1; i >= 0; i--) {
      this.trackedVoices[i].release(true);
    }
    this.trackedVoices.length = 0;

    if (this.slideSource) {
      try {
        this.slideSource.stop();
        this.slideSource.disconnect();
      } catch {
        // Ignore already-stopped source
      }
      this.slideSource = null;
    }
    if (this.slideStickLfo) {
      try {
        this.slideStickLfo.stop();
        this.slideStickLfo.disconnect();
      } catch {
        // Ignore
      }
      this.slideStickLfo = null;
    }
    this.slideStickLfoDepth = null;
    this.slideStickModGain = null;
    this.slideRaspFilter = null;
    this.slideRaspGain = null;
    this.slideGain = null;
    this.slideFilterA = null;
    this.slideFilterB = null;
    this.masterGain = null;
    this.noiseBuffer = null;
    this.activeVoices = 0;
    this.activeShardGrains = 0;
    this.liveConnectedNodes = 0;
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
    }
  }
}

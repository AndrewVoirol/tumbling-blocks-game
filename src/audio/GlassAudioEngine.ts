/**
 * Procedural Physical Modal Glass Audio Engine (Web Audio API).
 * Synthesizes soda-lime glass bar flexural modes, crystalline sliding friction,
 * internal crazing snaps, and multi-shard fracture cascades with zero external audio assets.
 */
export class GlassAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;

  // Persistent sliding friction loop nodes (zero allocation during rAF)
  private slideGain: GainNode | null = null;
  private slideFilterA: BiquadFilterNode | null = null;
  private slideFilterB: BiquadFilterNode | null = null;

  private muted: boolean = false;
  private lastClinkTime: number = 0;
  private lastShardTinkTime: number = 0;
  private activeVoices: number = 0;
  private readonly MAX_VOICES = 14;
  private shardWindowStartSec: number = -1;
  private shardWindowVoicesUsed: number = 0;

  public lastTapDiagnostics: {
    pinch01: number;
    qFactor: number;
    decayMs: number;
    fundamentalHz: number;
  } | null = null;

  private liveConnectedNodes: number = 0;

  public getActiveVoiceCount(): number {
    return this.activeVoices;
  }

  public getConnectedTransientNodes(): number {
    return this.liveConnectedNodes;
  }

  private connectPanned(
    source: AudioNode,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): () => void {
    if (!this.ctx || !this.masterGain) return () => {};
    let cleaned = false;
    if (worldPos && camPos && camRight && typeof this.ctx.createStereoPanner === 'function') {
      const dx = worldPos[0] - camPos[0];
      const dy = worldPos[1] - camPos[1];
      const dz = worldPos[2] - camPos[2];
      const dist = Math.max(0.6, Math.hypot(dx, dy, dz));
      const pan = Math.max(
        -0.85,
        Math.min(0.85, (dx * camRight[0] + dy * camRight[1] + dz * camRight[2]) / (dist * 0.65))
      );
      const panner = this.ctx.createStereoPanner();
      panner.pan.value = pan;
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
    } else {
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
  }

  public unlock(): void {
    if (typeof window === 'undefined') return;
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = this.muted ? 0 : 0.65;

      // Subtle compressor to prevent clipping during multi-block tower collapses
      const compressor = this.ctx.createDynamicsCompressor();
      compressor.threshold.value = -10;
      compressor.knee.value = 12;
      compressor.ratio.value = 6;
      compressor.attack.value = 0.002;
      compressor.release.value = 0.12;

      this.masterGain.connect(compressor);
      compressor.connect(this.ctx.destination);

      this.initNoiseBuffer();
      this.initSlideNodes();
    }
    if (this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
  }

  public setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(muted ? 0 : 0.65, this.ctx.currentTime, 0.02);
    }
  }

  public isMuted(): boolean {
    return this.muted;
  }

  private initNoiseBuffer(): void {
    if (!this.ctx) return;
    const sampleRate = this.ctx.sampleRate;
    const length = sampleRate * 2;
    const buffer = this.ctx.createBuffer(1, length, sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    this.noiseBuffer = buffer;
  }

  private initSlideNodes(): void {
    if (!this.ctx || !this.noiseBuffer || !this.masterGain) return;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;
    src.loop = true;

    this.slideFilterA = this.ctx.createBiquadFilter();
    this.slideFilterA.type = 'bandpass';
    this.slideFilterA.frequency.value = 2350;
    this.slideFilterA.Q.value = 14;

    this.slideFilterB = this.ctx.createBiquadFilter();
    this.slideFilterB.type = 'bandpass';
    this.slideFilterB.frequency.value = 4120;
    this.slideFilterB.Q.value = 18;

    this.slideGain = this.ctx.createGain();
    this.slideGain.gain.value = 0;

    src.connect(this.slideFilterA);
    src.connect(this.slideFilterB);
    this.slideFilterA.connect(this.slideGain);
    this.slideFilterB.connect(this.slideGain);
    this.slideGain.connect(this.masterGain);
    src.start();
  }

  /**
   * Modulates real-time glass-on-glass sliding resonance while pulling/nudging a block.
   */
  public updateSlideFriction(slideSpeedDmPerSec: number, resistance01: number): void {
    if (!this.ctx || !this.slideGain || !this.slideFilterA || !this.slideFilterB || this.muted) return;
    const now = this.ctx.currentTime;
    const speedNorm = Math.min(1.0, Math.max(0.0, (slideSpeedDmPerSec - 0.04) / 4.2));
    const targetGain = speedNorm * (0.025 + 0.062 * resistance01);
    this.slideGain.gain.setTargetAtTime(targetGain, now, 0.02);
    this.slideFilterA.frequency.setTargetAtTime(2100 + speedNorm * 650 + resistance01 * 420, now, 0.025);
    this.slideFilterB.frequency.setTargetAtTime(3850 + speedNorm * 900 + resistance01 * 560, now, 0.025);
  }

  /**
   * Accentuated load-coupled acoustic tap-test resonance so players can scout loose vs pinched blocks by ear:
   * - Loose (`pinch01 = 0`): free-bar flexural ring (`Q = 48`, `160 ms` decay, warm `2320 Hz` fundamental + bright 2.56x singing harmonic)
   * - Pinched (`pinch01 = 1`): clamped bar (`Q = 14`, `35 ms` decay, stiffened `3760 Hz` muted tick + dry damping transient)
   */
  public playTapTestResonance(
    pinch01: number,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    const p = Math.max(0.0, Math.min(1.0, pinch01));
    const qFactor = 48.0 - 34.0 * p; // Q = 48 (loose) -> Q = 14 (pinched)
    const decayMs = 160.0 - 125.0 * p; // 160 ms (loose) -> 35 ms (pinched)
    const fundamentalHz = 2320.0 + 1440.0 * Math.pow(p, 0.85); // 2320 Hz -> 3760 Hz (+8.4 semitones)

    this.lastTapDiagnostics = {
      pinch01: p,
      qFactor,
      decayMs,
      fundamentalHz,
    };

    if (!this.ctx || !this.masterGain || this.muted) return;
    const available = this.MAX_VOICES - this.activeVoices;
    if (available <= 0) return;

    const now = this.ctx.currentTime;
    const durSec = decayMs * 0.001;
    const jitter = 0.985 + Math.random() * 0.03;
    const looseGain = 0.24 * (1.0 - 0.35 * p);

    // Primary flexural bar mode
    const osc0 = this.ctx.createOscillator();
    const g0 = this.ctx.createGain();
    osc0.type = 'sine';
    osc0.frequency.value = fundamentalHz * jitter;
    g0.gain.setValueAtTime(0.0001, now);
    g0.gain.linearRampToValueAtTime(looseGain, now + 0.001);
    g0.gain.exponentialRampToValueAtTime(0.0001, now + durSec);
    this.activeVoices++;
    osc0.connect(g0);
    const cleanup0 = this.connectPanned(g0, worldPos, camPos, camRight);
    osc0.start(now);
    osc0.stop(now + durSec + 0.01);
    osc0.onended = () => {
      this.activeVoices = Math.max(0, this.activeVoices - 1);
      try {
        osc0.disconnect();
      } catch {
        // Ignore
      }
      cleanup0();
    };

    // Singing crystalline overtone (prominent when loose, strongly suppressed when pinched)
    if (this.activeVoices < this.MAX_VOICES) {
      const osc1 = this.ctx.createOscillator();
      const g1 = this.ctx.createGain();
      osc1.type = 'sine';
      const overtoneRatio = 2.56 - 0.38 * p;
      osc1.frequency.value = Math.min(15200, fundamentalHz * overtoneRatio * jitter);
      const harmonicWeight = 0.48 * Math.pow(1.0 - p, 1.35) + 0.05;
      const dur1 = durSec * (0.82 - 0.35 * p);
      g1.gain.setValueAtTime(0.0001, now);
      g1.gain.linearRampToValueAtTime(looseGain * harmonicWeight, now + 0.0008);
      g1.gain.exponentialRampToValueAtTime(0.0001, now + Math.max(0.012, dur1));
      this.activeVoices++;
      osc1.connect(g1);
      const cleanup1 = this.connectPanned(g1, worldPos, camPos, camRight);
      osc1.start(now);
      osc1.stop(now + Math.max(0.012, dur1) + 0.01);
      osc1.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          osc1.disconnect();
        } catch {
          // Ignore
        }
        cleanup1();
      };
    }

    // Resonant Q-coupled bandpass excitation impulse (Q=48 ringing vs Q=14 dry clamped tap)
    if (this.noiseBuffer && this.activeVoices < this.MAX_VOICES) {
      const tapNoise = this.ctx.createBufferSource();
      tapNoise.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = fundamentalHz * (1.05 + 0.25 * p);
      bp.Q.value = qFactor;
      const ng = this.ctx.createGain();
      const transientAmp = 0.14 + 0.22 * p;
      const transientDur = 0.018 * (1.0 - 0.55 * p) + 0.005;
      ng.gain.setValueAtTime(transientAmp, now);
      ng.gain.exponentialRampToValueAtTime(0.0001, now + transientDur);
      this.activeVoices++;
      tapNoise.connect(bp);
      bp.connect(ng);
      const cleanupNoise = this.connectPanned(ng, worldPos, camPos, camRight);
      tapNoise.start(now, Math.random() * 1.4, transientDur + 0.02);
      tapNoise.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          tapNoise.disconnect();
          bp.disconnect();
        } catch {
          // Ignore
        }
        cleanupNoise();
      };
    }
  }

  /**
   * Modal synthesis of a rectangular soda-lime glass prism impact.
   */
  public playGlassImpact(
    intensity01: number,
    pitchScale: number = 1.0,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    const allowedModes = Math.min(3, this.MAX_VOICES - this.activeVoices);
    if (allowedModes <= 0) return;

    const now = this.ctx.currentTime;
    if (now - this.lastClinkTime < 0.014 && intensity01 < 0.5) return;
    this.lastClinkTime = now;

    const gainVal = Math.min(0.38, Math.max(0.015, intensity01 * 0.34));
    // Fundamental & overtone flexural modes of a 7.5 x 2.5 x 1.5 cm glass bar
    const baseFreqs = [2780, 5290, 8140];
    const modeDecays = [0.095, 0.055, 0.028];
    const modeWeights = [1.0, 0.52, 0.28];

    const jitter = 0.96 + Math.random() * 0.08;

    for (let i = 0; i < allowedModes; i++) {
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = Math.min(15500, baseFreqs[i] * pitchScale * jitter);

      const dur = modeDecays[i] * (0.75 + 0.45 * intensity01) / Math.sqrt(pitchScale);
      g.gain.setValueAtTime(0.0001, now);
      g.gain.linearRampToValueAtTime(gainVal * modeWeights[i], now + 0.0012);
      g.gain.exponentialRampToValueAtTime(0.0001, now + dur);

      this.activeVoices++;
      osc.connect(g);
      const cleanup = this.connectPanned(g, worldPos, camPos, camRight);
      osc.start(now);
      osc.stop(now + dur + 0.01);
      osc.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          osc.disconnect();
        } catch {
          // Ignore
        }
        cleanup();
      };
    }

    // Crisp micro-transient contact click
    if (this.noiseBuffer && intensity01 > 0.15 && this.activeVoices < this.MAX_VOICES) {
      const click = this.ctx.createBufferSource();
      click.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'highpass';
      bp.frequency.value = 3600;
      const cg = this.ctx.createGain();
      cg.gain.setValueAtTime(gainVal * 0.45, now);
      cg.gain.exponentialRampToValueAtTime(0.0001, now + 0.006);
      this.activeVoices++;
      click.connect(bp);
      bp.connect(cg);
      const cleanupClick = this.connectPanned(cg, worldPos, camPos, camRight);
      click.start(now, Math.random() * 1.5, 0.01);
      click.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          click.disconnect();
          bp.disconnect();
        } catch {
          // Ignore
        }
        cleanupClick();
      };
    }
  }

  /**
   * Internal stress craze / hairline crack sound without full shatter.
   */
  public playGlassCrack(severity01: number = 0.7): void {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.muted) return;
    const now = this.ctx.currentTime;

    if (this.activeVoices < this.MAX_VOICES) {
      const snap = this.ctx.createBufferSource();
      snap.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 5200;
      bp.Q.value = 4.5;

      const g = this.ctx.createGain();
      const peak = Math.min(0.42, 0.18 + severity01 * 0.25);
      g.gain.setValueAtTime(peak, now);
      g.gain.exponentialRampToValueAtTime(0.0001, now + 0.024);

      this.activeVoices++;
      snap.connect(bp);
      bp.connect(g);
      const cleanupSnap = this.connectPanned(g);
      snap.start(now, Math.random() * 1.2, 0.03);
      snap.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          snap.disconnect();
          bp.disconnect();
        } catch {
          // Ignore
        }
        cleanupSnap();
      };
    }

    this.playGlassImpact(0.45 * severity01, 1.32);
  }

  /**
   * Full 3D glass block fracture burst + oak table thud + inharmonic shard cascade.
   */
  public playGlassShatter(
    energy01: number = 1.0,
    hitTable: boolean = true,
    worldPos?: [number, number, number],
    camPos?: [number, number, number],
    camRight?: [number, number, number]
  ): void {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.muted) return;
    const now = this.ctx.currentTime;
    const amp = Math.min(0.52, 0.26 + energy01 * 0.26);

    // 1. Razor-sharp initial brittle fracture snap transient (2-8 kHz bandpass crackle)
    if (this.activeVoices < this.MAX_VOICES) {
      const burst = this.ctx.createBufferSource();
      burst.buffer = this.noiseBuffer;
      const hp = this.ctx.createBiquadFilter();
      hp.type = 'bandpass';
      hp.frequency.value = 3800;
      hp.Q.value = 1.1;

      const bg = this.ctx.createGain();
      bg.gain.setValueAtTime(amp, now);
      bg.gain.exponentialRampToValueAtTime(0.0001, now + 0.042);

      this.activeVoices++;
      burst.connect(hp);
      hp.connect(bg);
      const cleanupBurst = this.connectPanned(bg, worldPos, camPos, camRight);
      burst.start(now, Math.random(), 0.055);
      burst.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          burst.disconnect();
          hp.disconnect();
        } catch {
          // Ignore
        }
        cleanupBurst();
      };
    }

    // 2. Subtle quarter-sawn oak tabletop impact resonance when smashing onto wood
    if (hitTable && this.activeVoices < this.MAX_VOICES) {
      const thud = this.ctx.createOscillator();
      const tg = this.ctx.createGain();
      thud.type = 'triangle';
      thud.frequency.setValueAtTime(195, now);
      thud.frequency.exponentialRampToValueAtTime(88, now + 0.055);
      tg.gain.setValueAtTime(amp * 0.34, now);
      tg.gain.exponentialRampToValueAtTime(0.0001, now + 0.06);
      this.activeVoices++;
      thud.connect(tg);
      const cleanupThud = this.connectPanned(tg, worldPos, camPos, camRight);
      thud.start(now);
      thud.stop(now + 0.07);
      thud.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          thud.disconnect();
        } catch {
          // Ignore
        }
        cleanupThud();
      };
    }

    // 3. Staggered inharmonic dual-mode crystalline shard clinks
    const tinkCount = Math.min(5, Math.floor(3 + energy01 * 3));
    for (let i = 0; i < tinkCount; i++) {
      if (this.activeVoices + 2 > this.MAX_VOICES) break;
      const delay = 0.006 + Math.pow(Math.random(), 1.6) * 0.14;
      const baseFreq = 3600 + Math.random() * 5200;
      const overtoneRatio = 2.18 + Math.random() * 0.36;
      const dur = 0.026 + Math.random() * 0.042;
      const startT = now + delay;
      const tinkAmp = amp * (0.12 + Math.random() * 0.16) * (1 - i / (tinkCount + 3));

      const freqs = [baseFreq, Math.min(15800, baseFreq * overtoneRatio)];
      const weights = [1.0, 0.42];
      for (let m = 0; m < 2; m++) {
        const osc = this.ctx.createOscillator();
        const g = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freqs[m];
        const mDur = m === 0 ? dur : dur * 0.55;
        g.gain.setValueAtTime(0.0001, startT);
        g.gain.linearRampToValueAtTime(tinkAmp * weights[m], startT + 0.0008);
        g.gain.exponentialRampToValueAtTime(0.0001, startT + mDur);

        this.activeVoices++;
        osc.connect(g);
        const cleanupTink = this.connectPanned(g, worldPos, camPos, camRight);
        osc.start(startT);
        osc.stop(startT + mDur + 0.008);
        osc.onended = () => {
          this.activeVoices = Math.max(0, this.activeVoices - 1);
          try {
            osc.disconnect();
          } catch {
            // Ignore
          }
          cleanupTink();
        };
      }
    }
  }

  /**
   * Stereo-panned 2-oscillator modal voice pool for mid-air and pile shard-on-shard,
   * shard-on-block, and shard-on-table collisions (capped at 14 voices / 50 ms ranked by m_s * v_close^2).
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

    const now = this.ctx.currentTime;
    if (this.shardWindowStartSec < 0 || now - this.shardWindowStartSec >= 0.05) {
      this.shardWindowStartSec = now;
      this.shardWindowVoicesUsed = 0;
    }

    if (events.length > 1) {
      events.sort((a, b) => b.energy - a.energy);
    }

    for (let i = 0; i < events.length; i++) {
      if (this.activeVoices + 2 > this.MAX_VOICES) break;
      if (this.shardWindowVoicesUsed + 2 > this.MAX_VOICES) break;

      const ev = events[i];
      const sizeRatio = Math.cbrt(Math.max(0.00012, ev.volumeDm3) / 0.0025);
      const glassBoost = ev.hitGlass ? 1.12 : 1.0;
      const pitchScale = Math.max(0.72, Math.min(2.45, glassBoost / sizeRatio));
      const f0 = Math.min(13800, (3450 + Math.random() * 980) * pitchScale);
      const f1 = Math.min(16200, f0 * (2.26 + Math.random() * 0.24));

      const jitterDelay = i * 0.0025;
      const startT = now + jitterDelay;
      const peak = Math.min(0.19, Math.max(0.01, ev.intensity01 * (ev.hitGlass ? 0.18 : 0.14)));
      const dur = ((ev.hitGlass ? 0.044 : 0.034) + Math.random() * 0.028) / Math.sqrt(pitchScale);

      const modes = [
        { f: f0, w: 1.0, d: dur },
        { f: f1, w: ev.hitGlass ? 0.44 : 0.36, d: dur * 0.54 },
      ];
      for (const m of modes) {
        const osc = this.ctx.createOscillator();
        const g = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = m.f;
        g.gain.setValueAtTime(0.0001, startT);
        g.gain.linearRampToValueAtTime(peak * m.w, startT + 0.0007);
        g.gain.exponentialRampToValueAtTime(0.0001, startT + m.d);

        this.activeVoices++;
        this.shardWindowVoicesUsed++;
        osc.connect(g);
        const cleanupMode = this.connectPanned(g, ev.worldPos, camPos, camRight);
        osc.start(startT);
        osc.stop(startT + m.d + 0.006);
        osc.onended = () => {
          this.activeVoices = Math.max(0, this.activeVoices - 1);
          try {
            osc.disconnect();
          } catch {
            // Ignore
          }
          cleanupMode();
        };
      }
    }
  }

  /**
   * Backwards-compatible single shard clink helper routed through the ranked modal voice pool.
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
    const notes = [1046.5, 1318.5]; // C6 -> E6 gentle glass bell
    const pitchShift = Math.min(1.25, 1 + Math.max(0, layerCount - 18) * 0.015);

    notes.forEach((f, idx) => {
      if (!this.ctx || !this.masterGain || this.activeVoices >= this.MAX_VOICES) return;
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = f * pitchShift;
      const t0 = now + idx * 0.075;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.linearRampToValueAtTime(0.075, t0 + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.36);
      this.activeVoices++;
      osc.connect(g);
      const cleanupBell = this.connectPanned(g);
      osc.start(t0);
      osc.stop(t0 + 0.38);
      osc.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          osc.disconnect();
        } catch {
          // Ignore
        }
        cleanupBell();
      };
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
    if (this.activeVoices + 2 > this.MAX_VOICES) return;

    const now = this.ctx.currentTime;
    const pitchMap: Record<typeof tone, number> = {
      default: 1.0,
      tab: 1.08,
      primary: 0.94,
      danger: 0.82,
      step: 1.18,
    };
    const pScale = pitchMap[tone];

    // 1. Subtle weighted brass/pewter socket seat thock
    const seat = this.ctx.createOscillator();
    const sg = this.ctx.createGain();
    seat.type = 'triangle';
    seat.frequency.setValueAtTime(210 * pScale, now);
    seat.frequency.exponentialRampToValueAtTime(92 * pScale, now + 0.022);
    sg.gain.setValueAtTime(0.068, now);
    sg.gain.exponentialRampToValueAtTime(0.0001, now + 0.024);
    this.activeVoices++;
    seat.connect(sg);
    const cleanupSeat = this.connectPanned(sg);
    seat.start(now);
    seat.stop(now + 0.028);
    seat.onended = () => {
      this.activeVoices = Math.max(0, this.activeVoices - 1);
      try {
        seat.disconnect();
      } catch {
        // Ignore
      }
      cleanupSeat();
    };

    // 2. Crisp glass bevel contact tick
    if (this.noiseBuffer && this.activeVoices < this.MAX_VOICES) {
      const tick = this.ctx.createBufferSource();
      tick.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 3450 * pScale;
      bp.Q.value = 8.5;
      const tg = this.ctx.createGain();
      tg.gain.setValueAtTime(0.075, now);
      tg.gain.exponentialRampToValueAtTime(0.0001, now + 0.008);
      this.activeVoices++;
      tick.connect(bp);
      bp.connect(tg);
      const cleanupTick = this.connectPanned(tg);
      tick.start(now, Math.random() * 1.2, 0.012);
      tick.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        try {
          tick.disconnect();
          bp.disconnect();
        } catch {
          // Ignore
        }
        cleanupTick();
      };
    }
  }

  /**
   * Delicate crystalline glass key-up harmonic chime when a pressed UI glass block rebounds.
   */
  public playUiKeyUp(
    tone: 'default' | 'tab' | 'primary' | 'danger' | 'step' = 'default'
  ): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    if (this.activeVoices >= this.MAX_VOICES) return;

    const now = this.ctx.currentTime;
    const freqMap: Record<typeof tone, number> = {
      default: 2480,
      tab: 2640,
      primary: 2220,
      danger: 1960,
      step: 2940,
    };
    const f0 = freqMap[tone] * (0.99 + Math.random() * 0.02);
    const dur = tone === 'primary' ? 0.072 : 0.048;

    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(f0, now);
    osc.frequency.exponentialRampToValueAtTime(f0 * 1.018, now + dur);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(tone === 'primary' ? 0.062 : 0.045, now + 0.0012);
    g.gain.exponentialRampToValueAtTime(0.0001, now + dur);

    this.activeVoices++;
    osc.connect(g);
    const cleanup = this.connectPanned(g);
    osc.start(now);
    osc.stop(now + dur + 0.008);
    osc.onended = () => {
      this.activeVoices = Math.max(0, this.activeVoices - 1);
      try {
        osc.disconnect();
      } catch {
        // Ignore
      }
      cleanup();
    };
  }

  public dispose(): void {
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
    }
  }
}

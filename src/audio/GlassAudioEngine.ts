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
  private activeVoices: number = 0;
  private readonly MAX_VOICES = 28;

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
    const speedNorm = Math.min(1.0, Math.max(0.0, (slideSpeedDmPerSec - 0.08) / 4.5));
    const targetGain = speedNorm * (0.025 + 0.055 * resistance01);
    this.slideGain.gain.setTargetAtTime(targetGain, now, 0.025);
    this.slideFilterA.frequency.setTargetAtTime(2100 + speedNorm * 650 + resistance01 * 380, now, 0.03);
    this.slideFilterB.frequency.setTargetAtTime(3850 + speedNorm * 900 + resistance01 * 520, now, 0.03);
  }

  /**
   * Modal synthesis of a rectangular soda-lime glass prism impact.
   */
  public playGlassImpact(intensity01: number, pitchScale: number = 1.0): void {
    if (!this.ctx || !this.masterGain || this.muted) return;
    if (this.activeVoices >= this.MAX_VOICES) return;

    const now = this.ctx.currentTime;
    if (now - this.lastClinkTime < 0.014 && intensity01 < 0.5) return;
    this.lastClinkTime = now;

    const gainVal = Math.min(0.38, Math.max(0.015, intensity01 * 0.34));
    // Fundamental & overtone flexural modes of a 7.5 x 2.5 x 1.5 cm glass bar
    const baseFreqs = [2780, 5290, 8140];
    const modeDecays = [0.095, 0.055, 0.028];
    const modeWeights = [1.0, 0.52, 0.28];

    const jitter = 0.96 + Math.random() * 0.08;
    this.activeVoices += 3;

    for (let i = 0; i < 3; i++) {
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = Math.min(15500, baseFreqs[i] * pitchScale * jitter);

      const dur = modeDecays[i] * (0.75 + 0.45 * intensity01) / Math.sqrt(pitchScale);
      g.gain.setValueAtTime(0.0001, now);
      g.gain.linearRampToValueAtTime(gainVal * modeWeights[i], now + 0.0012);
      g.gain.exponentialRampToValueAtTime(0.0001, now + dur);

      osc.connect(g);
      g.connect(this.masterGain);
      osc.start(now);
      osc.stop(now + dur + 0.01);
      osc.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
      };
    }

    // Crisp micro-transient contact click
    if (this.noiseBuffer && intensity01 > 0.15) {
      const click = this.ctx.createBufferSource();
      click.buffer = this.noiseBuffer;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'highpass';
      bp.frequency.value = 3600;
      const cg = this.ctx.createGain();
      cg.gain.setValueAtTime(gainVal * 0.45, now);
      cg.gain.exponentialRampToValueAtTime(0.0001, now + 0.006);
      click.connect(bp);
      bp.connect(cg);
      cg.connect(this.masterGain);
      click.start(now, Math.random() * 1.5, 0.01);
    }
  }

  /**
   * Internal stress craze / hairline crack sound without full shatter.
   */
  public playGlassCrack(severity01: number = 0.7): void {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.muted) return;
    const now = this.ctx.currentTime;

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

    snap.connect(bp);
    bp.connect(g);
    g.connect(this.masterGain);
    snap.start(now, Math.random() * 1.2, 0.03);

    this.playGlassImpact(0.45 * severity01, 1.32);
  }

  /**
   * Full 3D glass block fracture burst + cascading crystalline shard tinks.
   */
  public playGlassShatter(energy01: number = 1.0): void {
    if (!this.ctx || !this.masterGain || !this.noiseBuffer || this.muted) return;
    const now = this.ctx.currentTime;

    // Primary brittle fracture snap
    const burst = this.ctx.createBufferSource();
    burst.buffer = this.noiseBuffer;
    const hp = this.ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 2900;

    const bg = this.ctx.createGain();
    const amp = Math.min(0.55, 0.3 + energy01 * 0.25);
    bg.gain.setValueAtTime(amp, now);
    bg.gain.exponentialRampToValueAtTime(0.0001, now + 0.065);

    burst.connect(hp);
    hp.connect(bg);
    bg.connect(this.masterGain);
    burst.start(now, Math.random(), 0.08);

    // Staggered high-frequency shard tinks
    const tinkCount = Math.min(10, Math.floor(6 + energy01 * 5));
    for (let i = 0; i < tinkCount; i++) {
      if (this.activeVoices >= this.MAX_VOICES) break;
      const delay = 0.008 + Math.pow(Math.random(), 1.5) * 0.22;
      const freq = 4200 + Math.random() * 6800;
      const dur = 0.035 + Math.random() * 0.065;

      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;

      const startT = now + delay;
      const tinkAmp = amp * (0.14 + Math.random() * 0.2) * (1 - i / (tinkCount + 3));
      g.gain.setValueAtTime(0.0001, startT);
      g.gain.linearRampToValueAtTime(tinkAmp, startT + 0.001);
      g.gain.exponentialRampToValueAtTime(0.0001, startT + dur);

      this.activeVoices++;
      osc.connect(g);
      g.connect(this.masterGain);
      osc.start(startT);
      osc.stop(startT + dur + 0.01);
      osc.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
      };
    }
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
      if (!this.ctx || !this.masterGain) return;
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = f * pitchShift;
      const t0 = now + idx * 0.075;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.linearRampToValueAtTime(0.075, t0 + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.36);
      osc.connect(g);
      g.connect(this.masterGain);
      osc.start(t0);
      osc.stop(t0 + 0.38);
    });
  }

  public dispose(): void {
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
    }
  }
}

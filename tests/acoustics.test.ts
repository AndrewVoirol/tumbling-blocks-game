import { describe, expect, it } from 'vitest';
import {
  computeGlassBarModes,
  computeHertzianContact,
  computeHertzianCutoffHz,
  computeHertzianModeGain,
  computeHousnerRockingSchedule,
  computeOakPlateModes,
  computeShardFundamentalHz,
  computeShardModalVoice,
  computeShardStereoPan,
  computeStribeckSlipProfile,
  GlassAudioEngine,
  ShardChatterImpact,
} from '../src/audio/GlassAudioEngine';

/**
 * Deterministic instrumented Web Audio API test double that records every AudioParam value,
 * scheduled ramp, oscillator frequency, stereo pan, and connection lifecycle to catch any
 * NaN / Infinity / polyphony leak / runaway voice bug in Node/Vitest.
 */
class InstrumentedMockAudioContext {
  public currentTime: number = 1.0;
  public readonly sampleRate: number = 48000;
  public state: AudioContextState = 'running';
  public readonly destination = { kind: 'destination' } as unknown as AudioDestinationNode;

  public readonly recordedParamValues: number[] = [];
  public readonly createdOscillators: Array<{
    type: string;
    frequencyValues: number[];
    startedAt: number | null;
    stoppedAt: number | null;
    disconnected: boolean;
  }> = [];
  public readonly createdPanners: Array<{ panValues: number[]; disconnected: boolean }> = [];

  private makeParam(initial: number, sink?: number[]) {
    const record = (v: number) => {
      this.recordedParamValues.push(v);
      sink?.push(v);
      if (!Number.isFinite(v)) {
        throw new Error(`Non-finite AudioParam value written: ${v}`);
      }
    };
    let current = initial;
    return {
      get value() {
        return current;
      },
      set value(v: number) {
        record(v);
        current = v;
      },
      setValueAtTime: (v: number, t: number) => {
        record(v);
        record(t);
        current = v;
      },
      linearRampToValueAtTime: (v: number, t: number) => {
        record(v);
        record(t);
        current = v;
      },
      exponentialRampToValueAtTime: (v: number, t: number) => {
        if (v <= 0) {
          throw new RangeError(`exponentialRampToValueAtTime requires positive value, got ${v}`);
        }
        record(v);
        record(t);
        current = v;
      },
      setTargetAtTime: (target: number, t: number, tau: number) => {
        record(target);
        record(t);
        record(tau);
        current = target;
      },
      cancelScheduledValues: (t: number) => {
        record(t);
      },
    };
  }

  public createGain(): GainNode {
    const param = this.makeParam(1.0);
    return {
      gain: param,
      connect: () => {},
      disconnect: () => {},
    } as unknown as GainNode;
  }

  public createDynamicsCompressor(): DynamicsCompressorNode {
    return {
      threshold: this.makeParam(-24),
      knee: this.makeParam(30),
      ratio: this.makeParam(12),
      attack: this.makeParam(0.003),
      release: this.makeParam(0.25),
      connect: () => {},
      disconnect: () => {},
    } as unknown as DynamicsCompressorNode;
  }

  public createBiquadFilter(): BiquadFilterNode {
    return {
      type: 'bandpass',
      frequency: this.makeParam(1000),
      Q: this.makeParam(1),
      gain: this.makeParam(0),
      connect: () => {},
      disconnect: () => {},
    } as unknown as BiquadFilterNode;
  }

  public createStereoPanner(): StereoPannerNode {
    const entry = { panValues: [] as number[], disconnected: false };
    this.createdPanners.push(entry);
    return {
      pan: this.makeParam(0, entry.panValues),
      connect: () => {},
      disconnect: () => {
        entry.disconnected = true;
      },
    } as unknown as StereoPannerNode;
  }

  public createOscillator(): OscillatorNode {
    const entry = {
      type: 'sine',
      frequencyValues: [] as number[],
      startedAt: null as number | null,
      stoppedAt: null as number | null,
      disconnected: false,
    };
    this.createdOscillators.push(entry);
    const freqParam = this.makeParam(440, entry.frequencyValues);
    const node = {
      get type() {
        return entry.type as OscillatorType;
      },
      set type(v: OscillatorType) {
        entry.type = v;
      },
      frequency: freqParam,
      onended: null as (() => void) | null,
      connect: () => {},
      disconnect: () => {
        entry.disconnected = true;
      },
      start: (when: number = this.currentTime) => {
        this.recordedParamValues.push(when);
        entry.startedAt = when;
      },
      stop: (when: number = this.currentTime) => {
        this.recordedParamValues.push(when);
        entry.stoppedAt = when;
      },
    };
    return node as unknown as OscillatorNode;
  }

  public createBuffer(_channels: number, length: number, sampleRate: number): AudioBuffer {
    const data = new Float32Array(length);
    return {
      sampleRate,
      length,
      duration: length / sampleRate,
      numberOfChannels: 1,
      getChannelData: () => data,
    } as unknown as AudioBuffer;
  }

  public createBufferSource(): AudioBufferSourceNode {
    return {
      buffer: null,
      loop: false,
      onended: null,
      connect: () => {},
      disconnect: () => {},
      start: (when: number = this.currentTime, offset: number = 0, duration?: number) => {
        this.recordedParamValues.push(when, offset);
        if (duration !== undefined) this.recordedParamValues.push(duration);
      },
      stop: (when: number = this.currentTime) => {
        this.recordedParamValues.push(when);
      },
    } as unknown as AudioBufferSourceNode;
  }

  public async resume(): Promise<void> {
    this.state = 'running';
  }

  public async close(): Promise<void> {
    this.state = 'closed';
  }
}

describe('Vitrail First-Principles Physical Modal & Impact Acoustics', () => {
  it('1. models Timoshenko thick-bar dispersion (L/h = 5 => f2/f1 < 2.78) and solder-came boundary damping', () => {
    const modes = computeGlassBarModes();
    expect(modes.length).toBeGreaterThanOrEqual(6);

    const flexY1 = modes.find((m) => m.kind === 'flexural-y' && m.order === 1)!;
    const flexY2 = modes.find((m) => m.kind === 'flexural-y' && m.order === 2)!;
    const flexY3 = modes.find((m) => m.kind === 'flexural-y' && m.order === 3)!;
    const flexZ1 = modes.find((m) => m.kind === 'flexural-z' && m.order === 1)!;
    const torsion1 = modes.find((m) => m.kind === 'torsional' && m.order === 1)!;
    const long1 = modes.find((m) => m.kind === 'longitudinal' && m.order === 1)!;

    expect(flexY1).toBeDefined();
    expect(flexY2).toBeDefined();
    expect(flexY3).toBeDefined();
    expect(flexZ1).toBeDefined();
    expect(torsion1).toBeDefined();
    expect(long1).toBeDefined();

    // Physical audible kHz bounds for leaded stained-glass tracery prism (75 x 25 x 15 mm, 71.1 g)
    expect(flexY1.freqHz).toBeGreaterThan(2200);
    expect(flexY1.freqHz).toBeLessThan(3600);
    expect(flexZ1.freqHz).toBeGreaterThan(flexY1.freqHz * 1.3);
    expect(torsion1.freqHz).toBeGreaterThan(flexY1.freqHz * 1.25);
    expect(long1.freqHz).toBeGreaterThan(flexY2.freqHz);
    expect(flexY3.freqHz).toBeLessThan(16000);

    // Timoshenko rotary inertia & Cowper shear correction on thick L/h = 5 bar:
    // Uncorrected Euler-Bernoulli gives (lambda_2 / lambda_1)^2 = 2.7565 ~ (5/3)^2 = 2.7778;
    // Timoshenko must compress f_2 / f_1 strictly below (5/3)^2 = 2.78 (specifically ~2.37)!
    const f2OverF1 = flexY2.freqHz / flexY1.freqHz;
    expect(f2OverF1).toBeLessThan((5 / 3) ** 2);
    expect(f2OverF1).toBeGreaterThan(2.15);
    expect(f2OverF1).toBeLessThan(2.55);

    expect(flexY1.timoshenkoFactor).toBeLessThan(0.92);
    expect(flexY2.timoshenkoFactor).toBeLessThan(flexY1.timoshenkoFactor);
    expect(flexY3.timoshenkoFactor).toBeLessThan(flexY2.timoshenkoFactor);

    // Peripheral 60/40 Sn-Pb solder came viscoelastic damping shortens upper-mode decay vs fundamental
    // and damps much faster than pure un-leaded soda-lime crystal
    expect(flexY2.decayTauSec).toBeLessThan(flexY1.decayTauSec * 0.45);
    expect(flexY3.decayTauSec).toBeLessThan(flexY2.decayTauSec * 0.65);

    const pureCrystalModes = computeGlassBarModes({ cameSeamFraction: 0, cameLossRef: 0 });
    const pureY1 = pureCrystalModes.find((m) => m.kind === 'flexural-y' && m.order === 1)!;
    expect(pureY1.freqHz).toBeGreaterThan(12000); // Monolithic 7.5 cm glass bar is ~12.9 kHz
    expect(flexY1.lossFactor).toBeGreaterThan(pureY1.lossFactor * 4.0);
  });

  it('2. models structural crazing damage softening (-1.5% to -4.1%), doublet splitting, and increased damping', () => {
    const intact = computeGlassBarModes({ damage: 0.0 });
    const mildCraze = computeGlassBarModes({ damage: 0.45 });
    const heavyCraze = computeGlassBarModes({ damage: 0.92 });
    const maxCraze = computeGlassBarModes({ damage: 1.0 });

    const f0 = intact[0].freqHz;
    const fMild = mildCraze[0].freqHz;
    const fHeavy = heavyCraze[0].freqHz;
    const fMax = maxCraze[0].freqHz;

    // Monotonic stiffness softening across (-1.5% .. -4.2%)
    const dropMildPct = ((f0 - fMild) / f0) * 100;
    const dropHeavyPct = ((f0 - fHeavy) / f0) * 100;
    const dropMaxPct = ((f0 - fMax) / f0) * 100;

    expect(dropMildPct).toBeGreaterThan(1.4);
    expect(dropHeavyPct).toBeGreaterThan(dropMildPct);
    expect(dropHeavyPct).toBeLessThan(4.2);
    expect(dropMaxPct).toBeGreaterThan(3.8);
    expect(dropMaxPct).toBeLessThan(4.5);

    // Crack-asymmetry doublet splitting produces audible beating only when damaged
    expect(intact[0].splitBeatHz).toBe(0);
    expect(mildCraze[0].splitBeatHz).toBeGreaterThan(10);
    expect(heavyCraze[0].splitBeatHz).toBeGreaterThan(mildCraze[0].splitBeatHz);

    // Damping ratio increases monotonically and upper modes dull rapidly
    expect(mildCraze[0].dampingRatio).toBeGreaterThan(intact[0].dampingRatio);
    expect(heavyCraze[0].dampingRatio).toBeGreaterThan(mildCraze[0].dampingRatio);
    expect(heavyCraze[3].decayTauSec).toBeLessThan(intact[3].decayTauSec * 0.45);
  });

  it('3. enforces Hertzian contact duration tau_c ~ v_n^(-1/5) spectral scaling and oak vs glass modulus contrast', () => {
    const speeds = [0.02, 0.05, 0.15, 0.45, 1.2, 3.0];
    let prevGlassFc = 0;
    let prevWoodFc = 0;

    for (const vn of speeds) {
      const fcGlass = computeHertzianCutoffHz(vn, 'glass');
      const fcWood = computeHertzianCutoffHz(vn, 'wood');

      expect(fcGlass).toBeGreaterThan(prevGlassFc);
      expect(fcWood).toBeGreaterThan(prevWoodFc);
      // Transverse white oak E_R = 1.6 GPa << soda-lime glass E = 72 GPa => ~4.4x lower cutoff on oak
      expect(fcWood).toBeLessThan(fcGlass * 0.3);

      prevGlassFc = fcGlass;
      prevWoodFc = fcWood;
    }

    // Exact Hertzian v_n^(1/5) exponent check: doubling impact velocity scales f_c by 2^(0.2)
    const fc025 = computeHertzianCutoffHz(0.25, 'glass');
    const fc050 = computeHertzianCutoffHz(0.5, 'glass');
    expect(fc050 / fc025).toBeCloseTo(Math.pow(2.0, 0.2), 5);

    // Spectral brightness contrast: gentle 0.05 m/s settle vs sharp 1.5 m/s edge collision
    const modes = computeGlassBarModes();
    const fFund = modes.find((m) => m.kind === 'flexural-y' && m.order === 1)!.freqHz;
    const fUpper = modes.find((m) => m.kind === 'flexural-y' && m.order === 3)!.freqHz;

    const fcGentle = computeHertzianCutoffHz(0.05, 'glass');
    const fcSharp = computeHertzianCutoffHz(1.5, 'glass');
    const upperToFundGentle =
      computeHertzianModeGain(fUpper, fcGentle) / computeHertzianModeGain(fFund, fcGentle);
    const upperToFundSharp =
      computeHertzianModeGain(fUpper, fcSharp) / computeHertzianModeGain(fFund, fcSharp);

    expect(upperToFundSharp).toBeGreaterThan(upperToFundGentle * 2.5);

    const contactInfo = computeHertzianContact({ impactSpeedMps: 1.0, surface: 'glass' });
    expect(contactInfo.peakForceN).toBeGreaterThan(50);
    expect(contactInfo.contactTimeSec).toBeGreaterThan(0.00005);
    expect(contactInfo.contactTimeSec).toBeLessThan(0.001);
  });

  it('4. models orthotropic quarter-sawn white oak circular Kirchhoff plate modes (R = 43 cm)', () => {
    const centerModes = computeOakPlateModes({ radialNorm: 0.0 });
    const rimModes = computeOakPlateModes({ radialNorm: 0.9 });

    expect(centerModes.map((m) => m.label)).toEqual(['f01', 'f11', 'f21', 'f02']);
    expect(centerModes[0].freqHz).toBeGreaterThan(90);
    expect(centerModes[0].freqHz).toBeLessThan(150);
    expect(centerModes[1].freqHz).toBeGreaterThan(centerModes[0].freqHz);
    expect(centerModes[2].freqHz).toBeGreaterThan(centerModes[1].freqHz);
    expect(centerModes[3].freqHz).toBeGreaterThan(centerModes[2].freqHz);
    expect(centerModes[3].freqHz).toBeLessThan(650);

    // Asymmetric 1- and 2-nodal-diameter modes (f11, f21) excite much more strongly near the rim
    expect(rimModes[1].excitationWeight).toBeGreaterThan(centerModes[1].excitationWeight * 4.0);
    expect(rimModes[2].excitationWeight).toBeGreaterThan(centerModes[2].excitationWeight * 5.0);
  });

  it('5. enforces Voronoi shard f ~ m^(-1/3) scaling, stereo pan bounds, and <= 12 polyphony cap on 200-impact bursts', () => {
    const massesG = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0];
    const freqs = massesG.map((g) => computeShardFundamentalHz(g * 0.001, 'glass'));

    for (let i = 1; i < freqs.length; i++) {
      expect(freqs[i]).toBeLessThan(freqs[i - 1]);
    }
    // 0.2 g .. 8.0 g shards must lie strictly inside [2.2 kHz, 9.5 kHz]
    const f02g = computeShardFundamentalHz(0.0002, 'glass');
    const f8g = computeShardFundamentalHz(0.008, 'glass');
    expect(f02g).toBeGreaterThanOrEqual(2200);
    expect(f02g).toBeLessThanOrEqual(9500);
    expect(f8g).toBeGreaterThanOrEqual(2200);
    expect(f8g).toBeLessThanOrEqual(9500);

    // Exact 8x mass increase (0.2 g -> 1.6 g, and 1.0 g -> 8.0 g) halves frequency (8^(-1/3) = 0.5)
    expect(computeShardFundamentalHz(0.0002, 'glass') / computeShardFundamentalHz(0.0016, 'glass')).toBeCloseTo(2.0, 5);
    expect(computeShardFundamentalHz(0.001, 'glass') / computeShardFundamentalHz(0.008, 'glass')).toBeCloseTo(2.0, 5);
    expect(computeShardFundamentalHz(0.001, 'glass') / computeShardFundamentalHz(0.015, 'glass')).toBeCloseTo(
      Math.cbrt(15.0),
      5
    );

    // Stereo pan formula: pan = clamp(x / 4.5, -0.85, 0.85)
    expect(computeShardStereoPan([0, 0.1, 0])).toBeCloseTo(0.0, 6);
    expect(computeShardStereoPan([-2.25, 0.1, 0])).toBeCloseTo(-0.5, 6);
    expect(computeShardStereoPan([2.25, 0.1, 0])).toBeCloseTo(0.5, 6);
    expect(computeShardStereoPan([-9.0, 0.1, 0])).toBeCloseTo(-0.85, 6);
    expect(computeShardStereoPan([9.0, 0.1, 0])).toBeCloseTo(0.85, 6);

    // Wood vs glass shard timbre contrast
    const woodSpec = computeShardModalVoice({
      pos: [1.8, 0.05, 0],
      speed: 0.8,
      massKg: 0.0015,
      surface: 'wood',
    });
    const glassSpec = computeShardModalVoice({
      pos: [1.8, 0.05, 0],
      speed: 0.8,
      massKg: 0.0015,
      surface: 'glass',
    });
    expect(woodSpec.woodKnockGainRatio).toBeGreaterThan(0.25);
    expect(glassSpec.woodKnockGainRatio).toBe(0);
    expect(glassSpec.overtoneGainRatio).toBeGreaterThan(woodSpec.overtoneGainRatio);

    // Deterministic per-shard bounce pitch: repeated bounces of the same shard mass emit identical f0
    const repeatCtx = new InstrumentedMockAudioContext();
    const repeatAudio = new GlassAudioEngine(repeatCtx as unknown as AudioContext);
    repeatAudio.playShardImpacts([
      { pos: [1.0, 0.05, 0], speed: 1.2, massKg: 0.00045, surface: 'glass', slotIndex: 7 },
    ]);
    const bounce1Freq = repeatCtx.createdOscillators[repeatCtx.createdOscillators.length - 2].frequencyValues[0];
    repeatCtx.currentTime += 0.06;
    repeatAudio.playShardImpacts([
      { pos: [1.1, 0.05, 0], speed: 0.45, massKg: 0.00045, surface: 'glass', slotIndex: 7 },
    ]);
    const bounce2Freq = repeatCtx.createdOscillators[repeatCtx.createdOscillators.length - 2].frequencyValues[0];
    expect(bounce2Freq).toBeCloseTo(bounce1Freq, 6);
    repeatAudio.dispose();

    // Adversarial 200-impact burst stress test (including NaN / Infinity / negative edge cases)
    const mockCtx = new InstrumentedMockAudioContext();
    const audio = new GlassAudioEngine(mockCtx as unknown as AudioContext);

    const burst200: ShardChatterImpact[] = [];
    for (let i = 0; i < 200; i++) {
      burst200.push({
        pos: [((i % 19) - 9) * 0.5, 0.05, 0],
        speed: i === 3 ? Number.NaN : i === 7 ? Number.POSITIVE_INFINITY : 0.08 + (i % 15) * 0.12,
        massKg: i === 11 ? -0.002 : 0.0002 + (i % 20) * 0.0006,
        surface: i % 2 === 0 ? 'wood' : 'glass',
        slotIndex: i % 64,
      });
    }

    const scheduled = audio.playShardImpacts(burst200);
    expect(scheduled).toBeLessThanOrEqual(12);
    expect(scheduled).toBeGreaterThan(0);
    expect(audio.getActiveShardGrainCount()).toBeLessThanOrEqual(12);
    expect(audio.getActiveVoiceCount()).toBeLessThanOrEqual(audio.MAX_VOICES);

    // Firing a second immediate burst at t + 5 ms must still respect the <= 12 concurrent grain cap
    mockCtx.currentTime += 0.005;
    audio.playShardChatter(burst200);
    expect(audio.getActiveShardGrainCount()).toBeLessThanOrEqual(12);
    expect(audio.getActiveVoiceCount()).toBeLessThanOrEqual(audio.MAX_VOICES);

    // Every single AudioParam value ever written must be finite
    for (const v of mockCtx.recordedParamValues) {
      expect(Number.isFinite(v)).toBe(true);
    }
    for (const p of mockCtx.createdPanners) {
      for (const pv of p.panValues) {
        expect(pv).toBeGreaterThanOrEqual(-0.85);
        expect(pv).toBeLessThanOrEqual(0.85);
      }
    }

    // Advance clock past grain decay (+250 ms) -> natural decay to complete silence & 0 leaked nodes
    mockCtx.currentTime += 0.25;
    expect(audio.getActiveShardGrainCount()).toBe(0);
    expect(audio.getActiveVoiceCount()).toBe(0);
    expect(audio.getConnectedTransientNodes()).toBe(0);

    audio.dispose();
  });

  it('6. synthesizes Housner accelerating wobble micro-clinks, Stribeck stick-slip singing, and tap-test contrast', () => {
    // A. Housner beveled-block rocking schedule accelerates (dt_2 < dt_1) and softens on each rebound
    const rockTaps = computeHousnerRockingSchedule(0.11, 0.9, 1.4, 'glass');
    expect(rockTaps.length).toBe(3);
    const dt1 = rockTaps[1].delaySec - rockTaps[0].delaySec;
    const dt2 = rockTaps[2].delaySec - rockTaps[1].delaySec;
    expect(dt1).toBeGreaterThan(0.015);
    expect(dt2).toBeLessThan(dt1);
    expect(rockTaps[1].cutoffHz).toBeLessThan(rockTaps[0].cutoffHz);
    expect(rockTaps[2].cutoffHz).toBeLessThan(rockTaps[1].cutoffHz);

    // B. Stribeck stick-slip tension vs loose airy micro-rasp
    const looseGlide = computeStribeckSlipProfile({
      slideSpeedMps: 0.06,
      clampingLoadN: 0.15,
    });
    const clampedSlowSing = computeStribeckSlipProfile({
      slideSpeedMps: 0.025,
      clampingLoadN: 9.5,
    });
    const clampedFastSing = computeStribeckSlipProfile({
      slideSpeedMps: 0.09,
      clampingLoadN: 9.5,
    });

    expect(looseGlide.stickSlipIndex01).toBe(0);
    expect(looseGlide.airyRaspIndex01).toBeGreaterThan(0.3);
    expect(looseGlide.resonatorQ).toBeLessThan(8.0);

    expect(clampedSlowSing.stickSlipIndex01).toBeGreaterThan(0.65);
    expect(clampedSlowSing.resonatorQ).toBeGreaterThan(28.0);
    expect(clampedSlowSing.singGain).toBeGreaterThan(looseGlide.singGain);
    expect(clampedFastSing.stickSlipPulseHz).toBeGreaterThan(clampedSlowSing.stickSlipPulseHz);
    expect(clampedFastSing.frictionCoeff).toBeLessThan(clampedSlowSing.frictionCoeff);

    // C. Full engine integration check across wobble, thud, shatter, slide, and T1/T4 tap-test gates
    const mockCtx = new InstrumentedMockAudioContext();
    const audio = new GlassAudioEngine(mockCtx as unknown as AudioContext);

    audio.playTapTestResonance(0.05, [0, 1.0, 0]);
    const looseTap = audio.lastTapDiagnostics!;
    audio.playTapTestResonance(0.95, [0, 1.0, 0]);
    const pinchedTap = audio.lastTapDiagnostics!;

    expect(looseTap.qFactor).toBeGreaterThan(42);
    expect(looseTap.decayMs).toBeGreaterThan(140);
    expect(pinchedTap.qFactor).toBeLessThan(18);
    expect(pinchedTap.decayMs).toBeLessThan(80);
    expect(looseTap.fundamentalHz).toBeLessThan(pinchedTap.fundamentalHz);

    const scheduledWobble = audio.playWobbleRock(0.09, 0.8, 1.2, [0.2, 0.45, 0], {
      blockId: 12,
      surface: 'glass',
      damage: 0.3,
    });
    expect(scheduledWobble).toBeGreaterThanOrEqual(2);
    // Immediate duplicate frame on same blockId is suppressed by per-block cooldown
    expect(audio.playWobbleRock(0.09, 0.8, 1.2, [0.2, 0.45, 0], { blockId: 12 })).toBe(0);

    audio.playBlockThud(0.85, [1.5, 0.08, -1.2], { damage: 0.4, radialNorm: 0.7 });
    audio.playGlassCrack(0.75, [0.1, 1.2, 0]);
    audio.playGlassShatter(0.9, true, [-1.4, 0.1, 0.8]);
    audio.updateSlideScrape({ slideSpeedMps: 0.04, clampingLoadN: 8.5 });
    audio.updateSlideScrape({ slideSpeedMps: 0, clampingLoadN: 0 });

    mockCtx.currentTime += 0.5;
    expect(audio.getActiveVoiceCount()).toBe(0);
    expect(audio.getConnectedTransientNodes()).toBe(0);
    audio.dispose();
  });

  it('7. enforces priority voice-stealing under heavy tower collapse and clean mute/dispose lifecycle', () => {
    const mockCtx = new InstrumentedMockAudioContext();
    const audio = new GlassAudioEngine(mockCtx as unknown as AudioContext);

    // Fill voice pool with low-priority shard chatter + gentle impacts
    const shardWave: ShardChatterImpact[] = Array.from({ length: 24 }, (_, idx) => ({
      pos: [(idx % 9) - 4, 0.06, 0],
      speed: 0.35,
      massKg: 0.0008 + idx * 0.0002,
      surface: idx % 2 === 0 ? 'glass' : 'wood',
      slotIndex: idx,
    }));
    audio.playShardImpacts(shardWave);
    expect(audio.getActiveShardGrainCount()).toBeLessThanOrEqual(12);

    // Fire high-priority multi-block collapse shatters while pool is saturated:
    // Priority voice-stealing must evict low-priority tails without ever exceeding MAX_VOICES
    for (let b = 0; b < 8; b++) {
      mockCtx.currentTime += 0.002;
      audio.playGlassShatter(0.95, true, [b * 0.3 - 1.0, 0.1, 0]);
      expect(audio.getActiveVoiceCount()).toBeLessThanOrEqual(audio.MAX_VOICES);
      expect(audio.getActiveShardGrainCount()).toBeLessThanOrEqual(audio.MAX_SHARD_GRAINS);
    }

    // Muting ramps continuous slide branches cleanly and blocks new transient spawns
    audio.updateSlideFriction(2.4, 0.85, 32, 1.1);
    audio.setMuted(true);
    expect(audio.isMuted()).toBe(true);
    const voicesAtMute = audio.getActiveVoiceCount();
    audio.playGlassImpact(0.9, 1.0, [0, 0.5, 0]);
    expect(audio.getActiveVoiceCount()).toBeLessThanOrEqual(voicesAtMute);

    // Dispose immediately stops & disconnects all remaining voices with 0 leaked nodes
    audio.dispose();
    expect(audio.getActiveVoiceCount()).toBe(0);
    expect(audio.getActiveShardGrainCount()).toBe(0);
    expect(audio.getConnectedTransientNodes()).toBe(0);
  });

  it('8. preserves 100% backward-compatible subclassing for WorkerAudioProxy and legacy callers', () => {
    class TestWorkerProxy extends GlassAudioEngine {
      public impacts = 0;
      public override playGlassImpact(
        intensity01: number,
        pitchMul: number = 1.0,
        worldPos?: [number, number, number]
      ): void {
        void intensity01;
        void pitchMul;
        void worldPos;
        this.impacts++;
      }
    }

    const proxy = new TestWorkerProxy();
    proxy.playGlassImpact(0.5, 0.92, [0, 0.2, 0]);
    proxy.playGlassCrack(0.6, [0, 0.4, 0]); // no-op when headless without ctx
    proxy.updateSlideFriction(1.2, 0.4);
    expect(proxy.impacts).toBe(1);
    proxy.dispose();
  });
});

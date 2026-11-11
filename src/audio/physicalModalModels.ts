/**
 * First-Principles Computational & Continuum Acoustics Models for Vitrail.
 *
 * Implements closed-form elastodynamic & tribological models with zero external assets:
 * 1. Timoshenko (1921) / Cowper (1966) free-free rectangular bar flexural, Saint-Venant torsional,
 *    and longitudinal eigenmodes for a leaded soda-lime stained-glass prism (75 x 25 x 15 mm, 71.1 g)
 *    with Zener thermoelastic + 60/40 Sn-Pb solder came viscoelastic damping and micro-crack softening.
 * 2. Hertz (1882) / Hunter (1957) elastic impact contact duration tau_c ~ m_eff^(2/5) R_eff^(-1/5)
 *    v_n^(-1/5) (E*)^(-2/5) and low-pass spectral excitation f_c = 1 / (2 * tau_c).
 * 3. Lekhnitskii (1968) / Leissa (1969) orthotropic quarter-sawn white oak circular Kirchhoff plate
 *    Bessel eigenmodes (f_01, f_11, f_21, f_02) for the R = 43 cm beveled pedestal table.
 * 4. 3D elastodynamic similarity f_0 ~ m^(-1/3) for convex Voronoi stained-glass shards (0.2 g - 15 g).
 * 5. Stribeck velocity-weakening stick-slip friction & Housner (1963) beveled-block rocking trains.
 */

export const SODA_LIME_GLASS = {
  /** Young's modulus E (Pa): 72 GPa soda-lime cathedral glass */
  youngsModulusPa: 72.0e9,
  /** Bulk density rho (kg/m^3) */
  densityKgPerM3: 2500.0,
  /** Poisson's ratio nu */
  poissonRatio: 0.22,
  /** Intrinsic bulk soda-lime dielectric/internal friction loss factor */
  intrinsicLossFactor: 6.0e-4,
  /** Thermal diffusivity D_th (m^2/s) for Zener transverse thermoelastic relaxation */
  thermalDiffusivityM2PerS: 5.2e-7,
  /** Zener thermoelastic relaxation strength Delta_E */
  thermoelasticStrength: 1.4e-3,
} as const;

export const LEADED_CAME_PRISM_DEFAULTS = {
  /** Regulation block length L = 7.5 cm (0.075 m) */
  lengthM: 0.075,
  /** Regulation block width b = 2.5 cm (0.025 m) */
  widthM: 0.025,
  /** Regulation block height/thickness h = 1.5 cm (0.015 m), thick slenderness L/h = 5.0 */
  heightM: 0.015,
  /** Calibrated leaded stained-glass block mass m = 71.1 g (0.0711 kg) */
  massKg: 0.0711,
  /** Diamond-polished bevel contact radius R_b = 1.25 mm (0.00125 m) */
  bevelRadiusM: 0.00125,
  /** Cumulative copper-foil / 60-40 Sn-Pb solder came tracery joint volume fraction */
  cameSeamFraction: 0.034,
  /** Effective viscoelastic solder/copper-foil joint transfer modulus across tracery seams (Pa) */
  cameJointModulusPa: 0.128e9,
  /** Reference viscoelastic loss factor of peripheral copper-foil & 60/40 Sn-Pb solder came at 2.8 kHz */
  cameLossRef: 0.0034,
  /** Viscoelastic solder grain-boundary frequency exponent alpha_v */
  cameLossFreqExp: 0.42,
} as const;

export const WHITE_OAK_TABLE_DEFAULTS = {
  /** Circular beveled oak tabletop radius R = 43 cm (0.43 m) */
  radiusM: 0.43,
  /** Solid quarter-sawn white oak slab thickness h_p = 3.2 cm (0.032 m) */
  thicknessM: 0.032,
  /** Longitudinal Young's modulus along grain E_L = 12.0 GPa */
  youngsModulusLPa: 12.0e9,
  /** Radial Young's modulus across quarter-sawn face E_R = 1.6 GPa */
  youngsModulusRPa: 1.6e9,
  /** In-plane longitudinal-radial shear modulus G_LR = 1.1 GPa */
  shearModulusLRPa: 1.1e9,
  /** Major Poisson ratio nu_LR */
  poissonLR: 0.37,
  /** Air-dry (12% MC) white oak density (kg/m^3) */
  densityKgPerM3: 750.0,
  /** Chaigne-Doutaut baseline wood internal friction tan(delta_W) */
  baseLossFactor: 0.022,
} as const;

/** Free-free Euler-Bernoulli roots lambda_n = beta_n * L solving cos(lambda) * cosh(lambda) = 1 */
const FREE_FREE_BEAM_ROOTS = [
  4.730040744862704,  // n = 1 (~ 3*pi/2 = 4.71239)
  7.853204624095838,  // n = 2 (~ 5*pi/2 = 7.85398)
  10.995607838001671, // n = 3 (~ 7*pi/2 = 10.99557)
  14.137165491257464, // n = 4 (~ 9*pi/2 = 14.13717)
] as const;

export type GlassModeKind = 'flexural-y' | 'flexural-z' | 'torsional' | 'longitudinal';

export interface GlassBarModalParams {
  lengthM?: number;
  widthM?: number;
  heightM?: number;
  massKg?: number;
  youngsModulusPa?: number;
  densityKgPerM3?: number;
  poissonRatio?: number;
  /** Copper-foil / 60-40 Sn-Pb came seam compliance fraction (0 = monolithic fused bar) */
  cameSeamFraction?: number;
  cameJointModulusPa?: number;
  cameLossRef?: number;
  /** Structural crazing damage in [0, 1] */
  damage?: number;
  /** Optional distributed bearing clamping load in Newtons (Winkler foundation shift + Coulomb damping) */
  clampingLoadN?: number;
}

export interface GlassBarMode {
  kind: GlassModeKind;
  /** Mode order n >= 1 within its polarization family */
  order: number;
  /** Uncorrected Euler-Bernoulli (or elementary rod) eigenfrequency (Hz) */
  eulerBernoulliHz: number;
  /** Timoshenko rotary-inertia & shear-corrected eigenfrequency (Hz), including damage softening */
  freqHz: number;
  /** Upper orthogonal crack-split doublet partner frequency (Hz); equals freqHz when damage === 0 */
  doubletFreqHz: number;
  /** Mode-splitting beat frequency |doubletFreqHz - freqHz| (Hz) */
  splitBeatHz: number;
  /** Dimensionless Timoshenko correction factor freq / eulerBernoulli (<= 1.0) */
  timoshenkoFactor: number;
  /** Total dimensionless loss factor eta_k = 2 * zeta_k */
  lossFactor: number;
  /** Critical damping ratio zeta_k = eta_k / 2 */
  dampingRatio: number;
  /** Resonant quality factor Q_k = 1 / eta_k */
  qFactor: number;
  /** 1/e exponential amplitude decay constant tau_k = 1 / (pi * f_k * eta_k) in seconds */
  decayTauSec: number;
  /** Relative baseline radiation/coupling weight */
  relativeWeight: number;
}

export interface HertzianContactParams {
  /** Normal impact closing velocity v_n in SI m/s */
  impactSpeedMps: number;
  /** Target substrate ('glass' for glass-on-glass, 'wood' for glass-on-white-oak) */
  surface?: 'glass' | 'wood';
  /** Effective colliding mass m_eff in kg (if omitted, computed from block mass & substrate) */
  effectiveMassKg?: number;
  /** Colliding body mass m_1 in kg (default 0.0711 kg) */
  bodyMassKg?: number;
  /** Effective local contact curvature radius R_eff in meters (default 2.2 mm beveled edge/crown) */
  effectiveRadiusM?: number;
}

export interface HertzianContactResult {
  /** Contact duration tau_c (seconds) */
  contactTimeSec: number;
  /** Half-sine / Hunter spectral -3 dB power cutoff f_c = 1 / (2 * tau_c) (Hz) */
  cutoffHz: number;
  /** Combined Hertzian contact modulus E* (Pa) */
  effectiveModulusPa: number;
  /** Peak normal Hertzian contact force F_max (N) */
  peakForceN: number;
}

export interface OakPlateModalParams {
  radiusM?: number;
  thicknessM?: number;
  youngsModulusLPa?: number;
  youngsModulusRPa?: number;
  shearModulusLRPa?: number;
  poissonLR?: number;
  densityKgPerM3?: number;
  /** Normalized radial strike position r / R in [0, 1] (0 = center pedestal, 1 = outer beveled rim) */
  radialNorm?: number;
}

export interface OakPlateMode {
  /** Bessel modal indices (m = nodal diameters, n = nodal circles) */
  label: 'f01' | 'f11' | 'f21' | 'f02';
  nodalDiameters: number;
  nodalCircles: number;
  freqHz: number;
  lossFactor: number;
  qFactor: number;
  decayTauSec: number;
  /** Spatial strike coupling weight at radialNorm */
  excitationWeight: number;
}

export interface ShardVoiceSynthesisSpec {
  fundamentalHz: number;
  overtoneHz: number;
  hertzCutoffHz: number;
  overtoneGainRatio: number;
  decayTauSec: number;
  woodKnockHz: number;
  woodKnockGainRatio: number;
  peakAmplitude: number;
  stereoPan: number;
}

export interface StribeckSlipAudioProfile {
  active: boolean;
  /** Effective dynamic friction coefficient mu(v) along the Stribeck curve */
  frictionCoeff: number;
  /** Dimensionless stick-slip singing instability index in [0, 1] */
  stickSlipIndex01: number;
  /** Unloaded airy micro-rasp weight in [0, 1] */
  airyRaspIndex01: number;
  /** Stick-slip relaxation pulse rate f_ss (Hz) */
  stickSlipPulseHz: number;
  /** Primary flexural singing frequency (Hz) */
  singingPrimaryHz: number;
  /** Secondary torsional/edgewise singing frequency (Hz) */
  singingSecondaryHz: number;
  /** Resonator Q factor (low ~4.5 for loose airy rasp, high ~38 for clamped singing) */
  resonatorQ: number;
  /** Continuous rasp gain [0..0.12] */
  raspGain: number;
  /** Singing stick-slip gain [0..0.14] */
  singGain: number;
}

export interface HousnerRockTap {
  /** Scheduled delay from initial wobble event in seconds */
  delaySec: number;
  /** Normal bevel impact speed v_n (m/s) */
  impactSpeedMps: number;
  /** Hertzian spectral cutoff f_c (Hz) */
  cutoffHz: number;
  /** Normalized tap intensity in [0, 1] */
  intensity01: number;
  /** Subtle upward pitch shift as rocking amplitude narrows */
  pitchScale: number;
}

function clamp(val: number, min: number, max: number): number {
  if (!Number.isFinite(val)) return min;
  return val < min ? min : val > max ? max : val;
}

/**
 * Deterministic 32-bit Mulberry32 PRNG in [0, 1) so procedural noise buffers
 * and micro-variations never rely on non-deterministic Math.random() state.
 */
export function createMulberry32(seed: number = 0x56495452): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Computes exact Timoshenko (1921) / Cowper (1966) free-free flexural, Saint-Venant torsional,
 * and longitudinal eigenmodes for a leaded soda-lime stained-glass prism (75 x 25 x 15 mm, L/h = 5).
 *
 * Includes:
 * - Reuss composite series compliance across copper-foil / 60-40 Sn-Pb solder came tracery seams
 * - Exact positive-discriminant Timoshenko rotary-inertia & shear factor xi_n = 2 / (B_n + sqrt(B_n^2 - 4*A_T))
 * - Zener transverse thermoelastic + solder-came viscoelastic damping tau_k(f)
 * - Kachanov / Budiansky-O'Connell micro-crack stiffness softening (-1.5% to -4.1% frequency drop),
 *   asymmetric crack-plane doublet splitting, and crack-face Coulomb damping increase with `damage`.
 */
export function computeGlassBarModes(params: GlassBarModalParams = {}): GlassBarMode[] {
  const L = clamp(params.lengthM ?? LEADED_CAME_PRISM_DEFAULTS.lengthM, 0.01, 1.0);
  const b = clamp(params.widthM ?? LEADED_CAME_PRISM_DEFAULTS.widthM, 0.004, 0.5);
  const h = clamp(params.heightM ?? LEADED_CAME_PRISM_DEFAULTS.heightM, 0.004, 0.5);
  const nu = clamp(params.poissonRatio ?? SODA_LIME_GLASS.poissonRatio, 0.05, 0.45);
  const EGlass = clamp(params.youngsModulusPa ?? SODA_LIME_GLASS.youngsModulusPa, 1.0e9, 200.0e9);
  const rhoGlass = clamp(params.densityKgPerM3 ?? SODA_LIME_GLASS.densityKgPerM3, 500.0, 8000.0);

  const cameFrac = clamp(
    params.cameSeamFraction ?? LEADED_CAME_PRISM_DEFAULTS.cameSeamFraction,
    0.0,
    0.25
  );
  const cameJointE = clamp(
    params.cameJointModulusPa ?? LEADED_CAME_PRISM_DEFAULTS.cameJointModulusPa,
    0.02e9,
    EGlass
  );
  const cameLossRef = clamp(
    params.cameLossRef ?? LEADED_CAME_PRISM_DEFAULTS.cameLossRef,
    0.0,
    0.2
  );
  const damage = clamp(params.damage ?? 0.0, 0.0, 1.0);
  const clampingLoadN = clamp(params.clampingLoadN ?? 0.0, 0.0, 80.0);

  // Effective density accounting for peripheral copper-foil & 60/40 Sn-Pb came mass (71.1 g on beveled volume)
  const nominalVolM3 = L * b * h * 0.989;
  const targetMassKg = params.massKg ?? (cameFrac > 0 ? LEADED_CAME_PRISM_DEFAULTS.massKg : rhoGlass * nominalVolM3);
  const rhoEff = clamp(targetMassKg / nominalVolM3, 500.0, 9000.0);

  // Reuss series effective composite modulus across soda-lime panes + viscoelastic solder came joints
  const invE0 = (1.0 - cameFrac) / EGlass + (cameFrac > 0 ? cameFrac / cameJointE : 0.0);
  const EComposite0 = 1.0 / invE0;

  // Structural crazing stiffness degradation: -1.54% freq drop at damage=0.45 up to -3.98% at damage=1.0
  const crazeStiffnessScale = Math.max(0.88, 1.0 - (0.060 * damage + 0.018 * damage * damage));
  const EEff = EComposite0 * crazeStiffnessScale;

  // Cowper (1966) rectangular cross-section Timoshenko shear coefficient kappa_s = 10*(1+nu)/(12+11*nu)
  const kappaShear = (10.0 * (1.0 + nu)) / (12.0 + 11.0 * nu);
  const eOverKappaG = (2.0 * (1.0 + nu)) / kappaShear;
  const cLong = Math.sqrt(EEff / rhoEff);
  const GEff = EEff / (2.0 * (1.0 + nu));
  const cShear = Math.sqrt(GEff / rhoEff);

  // Winkler elastic contact foundation shift when block is clamped between bearing courses
  const pinchNorm = clamp(clampingLoadN / 3.5, 0.0, 1.0);
  // Distributed bearing stiffness k_w / (rho * A) shifts fundamental upward under heavy clamping
  const winklerOmegaSq = Math.pow(2.0 * Math.PI * 2680.0, 2) * Math.pow(pinchNorm, 0.95) * 1.18;

  // Zener transverse thermoelastic characteristic time tau_th = h^2 / (pi^2 * D_th)
  const tauThermo = (h * h) / (Math.PI * Math.PI * SODA_LIME_GLASS.thermalDiffusivityM2PerS);

  const buildMode = (
    kind: GlassModeKind,
    order: number,
    ebHz: number,
    timoshenkoFactor: number,
    baseWeight: number
  ): GlassBarMode => {
    const rawTimoHz = ebHz * timoshenkoFactor;
    const clampedOmegaSq =
      Math.pow(2.0 * Math.PI * rawTimoHz, 2) + (order === 1 ? winklerOmegaSq : winklerOmegaSq * 0.45);
    const freqHz = Math.sqrt(clampedOmegaSq) / (2.0 * Math.PI);

    // Crack-induced symmetry breaking splits modes into closely spaced beating doublets
    const splitRatio = damage > 0.001 ? damage * (0.0045 + 0.0105 * damage) * (1.0 + 0.12 * (order - 1)) : 0.0;
    const doubletFreqHz = freqHz * (1.0 + splitRatio);
    const splitBeatHz = doubletFreqHz - freqHz;

    // Total loss factor = bulk soda-lime + Zener thermoelastic + viscoelastic Sn-Pb came + crack Coulomb + clamping
    const omega = 2.0 * Math.PI * freqHz;
    const etaThermo =
      SODA_LIME_GLASS.thermoelasticStrength *
      ((omega * tauThermo) / (1.0 + omega * omega * tauThermo * tauThermo));
    const etaCame =
      cameFrac > 0
        ? cameLossRef *
          Math.pow(Math.max(200.0, freqHz) / 2850.0, LEADED_CAME_PRISM_DEFAULTS.cameLossFreqExp) *
          (1.0 + 0.22 * (order - 1))
        : 0.0;
    const etaCrack =
      damage > 0
        ? (0.0038 * damage + 0.0065 * damage * damage) * (0.65 + 0.85 * (freqHz / 2850.0))
        : 0.0;
    const etaClamp = pinchNorm > 0 ? 0.068 * Math.pow(pinchNorm, 2.2) : 0.0;

    const lossFactor =
      SODA_LIME_GLASS.intrinsicLossFactor + etaThermo + etaCame + etaCrack + etaClamp;
    const dampingRatio = 0.5 * lossFactor;
    const qFactor = 1.0 / lossFactor;
    const decayTauSec = 1.0 / (Math.PI * freqHz * lossFactor);

    return {
      kind,
      order,
      eulerBernoulliHz: ebHz,
      freqHz,
      doubletFreqHz,
      splitBeatHz,
      timoshenkoFactor,
      lossFactor,
      dampingRatio,
      qFactor,
      decayTauSec,
      relativeWeight: baseWeight,
    };
  };

  const computeTimoshenkoFlexural = (
    thicknessM: number,
    orderIdx: number
  ): { ebHz: number; timoFactor: number } => {
    const lambda = FREE_FREE_BEAM_ROOTS[orderIdx];
    const rg = thicknessM / Math.sqrt(12.0);
    const ebOmega = ((lambda * lambda) / (L * L)) * cLong * rg;
    const ebHz = ebOmega / (2.0 * Math.PI);

    const eta = Math.pow((lambda * rg) / L, 2);
    const B = 1.0 + eta * (1.0 + eOverKappaG);
    const fourAT = 4.0 * eOverKappaG * eta * eta;
    // Algebraically positive discriminant: 1 + 2*eta*(1+gamma) + eta^2*(gamma-1)^2 > 1
    const disc = Math.max(0.0, B * B - fourAT);
    const xiSq = 2.0 / (B + Math.sqrt(disc));
    return { ebHz, timoFactor: Math.sqrt(xiSq) };
  };

  const modes: GlassBarMode[] = [];

  // 1-3: Flatwise free-free flexural modes (deflection along h = 15 mm, L/h = 5.0)
  const weightsY = [1.0, 0.58, 0.34];
  for (let i = 0; i < 3; i++) {
    const { ebHz, timoFactor } = computeTimoshenkoFlexural(h, i);
    modes.push(buildMode('flexural-y', i + 1, ebHz, timoFactor, weightsY[i]));
  }

  // 4: Edgewise free-free flexural fundamental (deflection along b = 25 mm, L/b = 3.0)
  {
    const { ebHz, timoFactor } = computeTimoshenkoFlexural(b, 0);
    modes.push(buildMode('flexural-z', 1, ebHz, timoFactor, 0.44));
  }

  // 5: Saint-Venant rectangular bar torsional fundamental (Roark / Timoshenko & Goodier)
  {
    const longSide = Math.max(b, h);
    const shortSide = Math.min(b, h);
    const aspect = shortSide / longSide;
    // Exact 4-term Saint-Venant rectangular torsion factor beta_SV = J_t / (longSide * shortSide^3)
    const betaSV =
      1.0 / 3.0 - 0.21 * aspect * (1.0 - Math.pow(aspect, 4) / 12.0);
    const jTorsion = betaSV * longSide * Math.pow(shortSide, 3);
    const iPolar = (b * h * (b * b + h * h)) / 12.0;
    const warpFactor = Math.sqrt(jTorsion / iPolar);
    const fTorsion0 = (cShear / (2.0 * L)) * warpFactor;
    modes.push(buildMode('torsional', 1, fTorsion0, 0.985, 0.40));
  }

  // 6: Free-free axial longitudinal bar fundamental (Rayleigh lateral-inertia corrected)
  {
    const fLongEB = cLong / (2.0 * L);
    const rgPolarSq = (b * b + h * h) / 12.0;
    const rayleighAxialFactor = 1.0 / Math.sqrt(1.0 + (Math.PI * Math.PI * nu * nu * rgPolarSq) / (2.0 * L * L));
    modes.push(buildMode('longitudinal', 1, fLongEB, rayleighAxialFactor, 0.26));
  }

  modes.sort((mA, mB) => mA.freqHz - mB.freqHz);
  return modes;
}

/**
 * Computes exact Hertz (1882) elastic impact contact duration tau_c, half-sine spectral cutoff f_c,
 * combined contact modulus E*, and peak normal force F_max.
 *
 * tau_c = 2.8684 * m_eff^(2/5) * R_eff^(-1/5) * v_n^(-1/5) * (E*)^(-2/5)
 * f_c   = 1 / (2 * tau_c)
 */
export function computeHertzianContact(params: HertzianContactParams): HertzianContactResult {
  const vn = clamp(params.impactSpeedMps, 0.005, 12.0);
  const surface = params.surface ?? 'glass';
  const bodyMassKg = clamp(params.bodyMassKg ?? LEADED_CAME_PRISM_DEFAULTS.massKg, 0.00005, 5.0);
  // Equal-mass two-block collision gives m_eff = m/2; massive oak table gives m_eff = m
  const mEff = clamp(
    params.effectiveMassKg ?? (surface === 'glass' ? 0.5 * bodyMassKg : bodyMassKg),
    0.000025,
    5.0
  );
  // Local contact curvature radius across 1.25 mm diamond bevel / waterglass crown (~2.2 mm default)
  const rEff = clamp(params.effectiveRadiusM ?? 0.0022, 0.0002, 0.15);

  const invGlass =
    (1.0 - SODA_LIME_GLASS.poissonRatio * SODA_LIME_GLASS.poissonRatio) /
    SODA_LIME_GLASS.youngsModulusPa;

  let invTarget: number;
  if (surface === 'wood') {
    // Normal impact onto horizontal quarter-sawn oak compresses wood radially/transversely (E_R = 1.6 GPa)
    const nuOak = WHITE_OAK_TABLE_DEFAULTS.poissonLR;
    invTarget = (1.0 - nuOak * nuOak) / WHITE_OAK_TABLE_DEFAULTS.youngsModulusRPa;
  } else {
    invTarget = invGlass;
  }

  const effectiveModulusPa = 1.0 / (invGlass + invTarget);

  // Exact Landau & Lifshitz (§9, Eq. 9.14) Hertzian impact duration constant: 2.8684
  const contactTimeSec =
    2.8684 *
    Math.pow(mEff, 0.4) *
    Math.pow(rEff, -0.2) *
    Math.pow(vn, -0.2) *
    Math.pow(effectiveModulusPa, -0.4);

  const cutoffHz = 1.0 / (2.0 * contactTimeSec);

  // Peak Hertzian normal force: F_max = (4/3)*E* * sqrt(R_eff) * delta_max^(3/2), where (1/2)m*v^2 = (2/5)F_max*delta_max
  const stiffnessK = (4.0 / 3.0) * effectiveModulusPa * Math.sqrt(rEff);
  const deltaMax = Math.pow((1.25 * mEff * vn * vn) / stiffnessK, 0.4);
  const peakForceN = stiffnessK * Math.pow(deltaMax, 1.5);

  return {
    contactTimeSec,
    cutoffHz,
    effectiveModulusPa,
    peakForceN,
  };
}

/**
 * Convenience pure function returning the Hertzian half-sine spectral cutoff frequency f_c = 1 / (2 * tau_c) in Hz.
 */
export function computeHertzianCutoffHz(
  impactSpeedMps: number,
  surface: 'glass' | 'wood' = 'glass',
  bodyMassKg: number = LEADED_CAME_PRISM_DEFAULTS.massKg,
  effectiveRadiusM: number = 0.0022
): number {
  return computeHertzianContact({
    impactSpeedMps,
    surface,
    bodyMassKg,
    effectiveRadiusM,
  }).cutoffHz;
}

/**
 * Smooth Hertzian half-sine / Hunter (1957) modal spectral excitation envelope:
 * Flat across the main contact-force lobe (f <= f_c = 1 / (2 * tau_c)) and rolling off steeply
 * (-24 dB/oct) above the main lobe knee 1.35 * f_c without artificial comb-filter dropouts.
 */
export function computeHertzianModeGain(freqHz: number, cutoffHz: number): number {
  const f = Math.max(0.0, freqHz);
  const fc = Math.max(10.0, cutoffHz);
  return 1.0 / (1.0 + Math.pow(f / (1.35 * fc), 4.0));
}

/**
 * Computes axisymmetric and asymmetric Kirchhoff circular plate Bessel eigenmodes
 * (f_01, f_11, f_21, f_02) for the R = 43 cm, h = 3.2 cm orthotropic quarter-sawn white-oak table
 * using Lekhnitskii (1968) angularly homogenized flexural rigidity D_eff and Leissa (1969) eigenvalues.
 */
export function computeOakPlateModes(params: OakPlateModalParams = {}): OakPlateMode[] {
  const R = clamp(params.radiusM ?? WHITE_OAK_TABLE_DEFAULTS.radiusM, 0.1, 2.0);
  const hp = clamp(params.thicknessM ?? WHITE_OAK_TABLE_DEFAULTS.thicknessM, 0.005, 0.2);
  const EL = clamp(params.youngsModulusLPa ?? WHITE_OAK_TABLE_DEFAULTS.youngsModulusLPa, 1.0e9, 40.0e9);
  const ER = clamp(params.youngsModulusRPa ?? WHITE_OAK_TABLE_DEFAULTS.youngsModulusRPa, 0.2e9, 20.0e9);
  const GLR = clamp(params.shearModulusLRPa ?? WHITE_OAK_TABLE_DEFAULTS.shearModulusLRPa, 0.1e9, 10.0e9);
  const nuLR = clamp(params.poissonLR ?? WHITE_OAK_TABLE_DEFAULTS.poissonLR, 0.05, 0.48);
  const nuRL = nuLR * (ER / EL);
  const rho = clamp(params.densityKgPerM3 ?? WHITE_OAK_TABLE_DEFAULTS.densityKgPerM3, 300.0, 1300.0);
  const rNorm = clamp(params.radialNorm ?? 0.35, 0.0, 1.0);

  const denom = Math.max(0.2, 1.0 - nuLR * nuRL);
  const h3Over12 = (hp * hp * hp) / 12.0;
  const Dx = (EL * h3Over12) / denom;
  const Dy = (ER * h3Over12) / denom;
  const Dxy = Dx * nuRL + (GLR * hp * hp * hp) / 6.0;
  // Lekhnitskii orthotropic circular plate effective flexural rigidity
  const DEff = (3.0 * Dx + 3.0 * Dy + 2.0 * Dxy) / 8.0;

  const plateScaleHz =
    (1.0 / (2.0 * Math.PI * R * R)) * Math.sqrt(DEff / (rho * hp));

  // Leissa (1969, NASA SP-160 Table 2.24) pedestal-clamped (b/a ~ 0.22) free-rim circular plate lambda_mn^2
  const specs: ReadonlyArray<{
    label: OakPlateMode['label'];
    m: number;
    n: number;
    lambdaSq: number;
    radialWeight: (r: number) => number;
  }> = [
    {
      label: 'f01',
      m: 0,
      n: 1,
      lambdaSq: 4.97,
      radialWeight: (r) => 0.62 + 0.38 * r * r,
    },
    {
      label: 'f11',
      m: 1,
      n: 1,
      lambdaSq: 9.0,
      // m=1 has nodal line through center (r=0) and peaks toward rim
      radialWeight: (r) => 0.12 + 0.88 * r,
    },
    {
      label: 'f21',
      m: 2,
      n: 1,
      lambdaSq: 14.68,
      // m=2 saddle mode grows quadratically toward rim
      radialWeight: (r) => 0.08 + 0.82 * r * r,
    },
    {
      label: 'f02',
      m: 0,
      n: 2,
      lambdaSq: 22.79,
      // Axisymmetric 1-nodal-circle mode (node near r ~ 0.68)
      radialWeight: (r) => 0.22 + 0.58 * Math.abs(Math.cos(2.25 * r)),
    },
  ];

  return specs.map((sp) => {
    const freqHz = plateScaleHz * sp.lambdaSq;
    // Chaigne-Doutaut frequency-dependent viscoelastic wood damping + radiation loss
    const lossFactor =
      WHITE_OAK_TABLE_DEFAULTS.baseLossFactor * Math.pow(freqHz / 110.0, 0.24) + 0.006;
    const qFactor = 1.0 / lossFactor;
    const decayTauSec = 1.0 / (Math.PI * freqHz * lossFactor);
    return {
      label: sp.label,
      nodalDiameters: sp.m,
      nodalCircles: sp.n,
      freqHz,
      lossFactor,
      qFactor,
      decayTauSec,
      excitationWeight: sp.radialWeight(rNorm),
    };
  });
}

/**
 * 3D elastodynamic similarity law for convex Voronoi soda-lime glass shards:
 * Characteristic chord d_i = (m_i / rho)^(1/3) => f_0(m_i) = f_ref * (m_ref / m_i)^(1/3).
 *
 * Anchored at m_ref = 1.0 g -> f_ref = 5,320 Hz (glass) so shards across [0.2 g, 8.0 g]
 * span [2,660 Hz .. 9,097 Hz] strictly inside [2.2 kHz, 9.5 kHz], extending smoothly to
 * 2,157 Hz at 15.0 g with exact m^(-1/3) scaling.
 */
export function computeShardFundamentalHz(
  massKg: number,
  surface: 'wood' | 'glass' = 'glass'
): number {
  // Guard non-positive / NaN while keeping [0.05 g .. 71.1 g] completely unclamped for exact m^(-1/3) scaling
  const mSafeKg = clamp(massKg, 0.00005, 0.08);
  const fRef1gHz = surface === 'glass' ? 5320.0 : 5180.0;
  return fRef1gHz * Math.cbrt(0.001 / mSafeKg);
}

/**
 * Computes stereo pan across the R = 4.3 dm circular oak table:
 * pan = clamp(x / 4.5, -0.85, 0.85) in world frame (or projected onto camera right axis at table scale 4.5 dm).
 */
export function computeShardStereoPan(
  posDm: readonly [number, number, number],
  camPosDm?: readonly [number, number, number],
  camRight?: readonly [number, number, number]
): number {
  const px = Number.isFinite(posDm[0]) ? posDm[0] : 0.0;
  const py = Number.isFinite(posDm[1]) ? posDm[1] : 0.0;
  const pz = Number.isFinite(posDm[2]) ? posDm[2] : 0.0;

  if (camPosDm && camRight) {
    const rx = Number.isFinite(camRight[0]) ? camRight[0] : 1.0;
    const ry = Number.isFinite(camRight[1]) ? camRight[1] : 0.0;
    const rz = Number.isFinite(camRight[2]) ? camRight[2] : 0.0;
    const rLen = Math.hypot(rx, ry, rz);
    if (rLen > 1e-4) {
      const lateralDm = (px * rx + py * ry + pz * rz) / rLen;
      return clamp(lateralDm / 4.5, -0.85, 0.85);
    }
  }
  return clamp(px / 4.5, -0.85, 0.85);
}

/**
 * Synthesizes complete modal & Hertzian voice parameters for a single 3D Voronoi shard bounce.
 */
export function computeShardModalVoice(
  impact: {
    pos: readonly [number, number, number];
    speed: number;
    massKg: number;
    surface: 'wood' | 'glass';
  },
  camPosDm?: readonly [number, number, number],
  camRight?: readonly [number, number, number]
): ShardVoiceSynthesisSpec {
  const mKg = clamp(impact.massKg, 0.00005, 0.08);
  const speedMps = clamp(impact.speed, 0.01, 8.0);
  const f0 = computeShardFundamentalHz(mKg, impact.surface);
  // Inharmonic 3D polyhedral overtone ratio (~2.29 for irregular convex cleavage shard)
  const overtoneHz = Math.min(18500.0, f0 * 2.29);

  // Equivalent shard cleavage tip radius scales with d_i = (m / rho)^(1/3)
  const shardChordM = Math.cbrt(mKg / SODA_LIME_GLASS.densityKgPerM3);
  const tipRadiusM = clamp(0.18 * shardChordM, 0.00035, 0.003);
  const hertzCutoffHz = computeHertzianCutoffHz(speedMps, impact.surface, mKg, tipRadiusM);

  const g0 = Math.max(0.12, computeHertzianModeGain(f0, hertzCutoffHz));
  const g1 = computeHertzianModeGain(overtoneHz, hertzCutoffHz);
  const overtoneGainRatio =
    impact.surface === 'glass'
      ? clamp((0.52 * g1) / g0, 0.08, 0.62)
      : clamp((0.30 * g1) / g0, 0.03, 0.34);

  // Smaller shards radiate and damp faster (~14 ms at 0.2 g up to ~48 ms at 8 g)
  const sizeScale = Math.cbrt(mKg / 0.001);
  const decayTauSec = clamp(
    (impact.surface === 'glass' ? 0.026 : 0.020) * Math.pow(sizeScale, 0.65),
    0.009,
    0.068
  );

  // Kinetic energy / momentum perceptual loudness scaling
  const momentumNorm = clamp((Math.pow(mKg / 0.0015, 0.45) * speedMps) / 1.35, 0.0, 1.0);
  const peakAmplitude = clamp(0.012 + 0.165 * Math.pow(momentumNorm, 0.78), 0.008, 0.185);

  const rNorm = clamp(Math.hypot(impact.pos[0], impact.pos[2]) / 4.3, 0.0, 1.0);
  const woodKnockHz = 420.0 + 190.0 * (1.0 - 0.35 * rNorm) + 65.0 / Math.max(0.45, sizeScale);
  const woodKnockGainRatio = impact.surface === 'wood' ? 0.38 : 0.0;

  return {
    fundamentalHz: f0,
    overtoneHz,
    hertzCutoffHz,
    overtoneGainRatio,
    decayTauSec,
    woodKnockHz,
    woodKnockGainRatio,
    peakAmplitude,
    stereoPan: computeShardStereoPan(impact.pos, camPosDm, camRight),
  };
}

/**
 * Evaluates Stribeck velocity-weakening friction mu(v) = mu_k + (mu_s - mu_k)*exp(-(v/v_S)^delta)
 * and maps normal clamping load F_N + slip velocity v into stick-slip singing vs airy rasp parameters.
 */
export function computeStribeckSlipProfile(params: {
  slideSpeedMps: number;
  clampingLoadN: number;
  slipPowerMilliwatts?: number;
  torsionRadPerSec?: number;
  damage?: number;
}): StribeckSlipAudioProfile {
  const v = clamp(Math.abs(params.slideSpeedMps), 0.0, 2.0);
  const fn = clamp(params.clampingLoadN, 0.0, 60.0);
  const powerMw = clamp(params.slipPowerMilliwatts ?? fn * v * 180.0, 0.0, 250.0);
  const omega = clamp(Math.abs(params.torsionRadPerSec ?? 0.0), 0.0, 12.0);
  const damage = clamp(params.damage ?? 0.0, 0.0, 1.0);

  const effSpeed = v + 0.012 * omega;
  if (effSpeed < 0.0025 && powerMw < 0.4) {
    return {
      active: false,
      frictionCoeff: 0.275,
      stickSlipIndex01: 0,
      airyRaspIndex01: 0,
      stickSlipPulseHz: 24,
      singingPrimaryHz: 2865,
      singingSecondaryHz: 4125,
      resonatorQ: 5.0,
      raspGain: 0,
      singGain: 0,
    };
  }

  const muStatic = 0.275;
  const muKinetic = 0.125;
  const vStribeck = 0.045; // 4.5 cm/s characteristic Stribeck velocity on soda-lime glass
  const ratio = effSpeed / vStribeck;
  const expTerm = Math.exp(-Math.pow(ratio, 1.15));
  const frictionCoeff = muKinetic + (muStatic - muKinetic) * expTerm;

  // Velocity-weakening slope magnitude |d(mu)/dv| peaks in the slow tense extraction window (0.5 - 6 cm/s)
  const weakeningWindow = clamp(Math.pow(ratio + 0.12, 0.55) * Math.exp(-0.55 * ratio) * 1.45, 0.0, 1.0);
  const loadTension01 = clamp((fn - 0.85) / 8.5, 0.0, 1.0);
  const stickSlipIndex01 = clamp(loadTension01 * (0.35 + 0.65 * weakeningWindow), 0.0, 1.0);

  const speedDrive01 = clamp((effSpeed - 0.0025) / 0.16, 0.0, 1.0);
  const unloadedFactor = 1.0 - 0.72 * loadTension01;
  const airyRaspIndex01 = clamp(speedDrive01 * unloadedFactor, 0.0, 1.0);

  // Macroscopic mocap-pull tangential stick-slip displacement d_ss = (mu_s - mu_k)*F_N / k_pull (~0.65..1.8 mm)
  const dStickSlipMeters = 0.00065 + 0.00115 * loadTension01;
  const stickSlipPulseHz = clamp(14.0 + effSpeed / dStickSlipMeters, 14.0, 185.0);

  const barModes = computeGlassBarModes({ damage, clampingLoadN: fn });
  const fFlex1 = barModes[0]?.freqHz ?? 2865.0;
  const fUpper = barModes[1]?.freqHz ?? 4125.0;

  // Loose unloaded block -> broad airy Q ~ 4.8; heavily clamped stick-slip -> singing Q ~ 38 (dulled if crazed)
  const crazeQScale = 1.0 - 0.35 * damage;
  const resonatorQ = (4.8 + 33.2 * stickSlipIndex01) * crazeQScale;

  const powerBoost = clamp(powerMw / 55.0, 0.0, 1.0);
  const raspGain = clamp(
    airyRaspIndex01 * 0.036 + speedDrive01 * 0.012,
    0.0,
    0.055
  );
  const singGain = clamp(
    stickSlipIndex01 * (0.028 + 0.068 * Math.max(speedDrive01, powerBoost * 0.85)),
    0.0,
    0.115
  );

  return {
    active: true,
    frictionCoeff,
    stickSlipIndex01,
    airyRaspIndex01,
    stickSlipPulseHz,
    singingPrimaryHz: fFlex1 + 140.0 * speedDrive01,
    singingSecondaryHz: fUpper + 220.0 * clamp(omega / 2.5, 0.0, 1.0),
    resonatorQ,
    raspGain,
    singGain,
  };
}

/**
 * Computes an accelerating Housner (1963) inverted-pendulum beveled-edge rocking micro-clink train
 * when a tilted prism (tiltAngleRad) rocks back flat onto its supporting course or oak tabletop.
 *
 * Half-period T_k ~ sqrt(theta_k / g_eff) shrinks on each bounce by angular restitution e_theta,
 * producing an accelerating double/triple tap with decreasing Hertzian cutoff f_c.
 */
export function computeHousnerRockingSchedule(
  tiltAngleRad: number,
  angularSpeedRadPerSec: number = 0.6,
  clampingLoadN: number = 0.7,
  surface: 'glass' | 'wood' = 'glass'
): HousnerRockTap[] {
  const theta0 = clamp(Math.abs(tiltAngleRad), 0.0, 0.45);
  const omega0 = clamp(Math.abs(angularSpeedRadPerSec), 0.0, 12.0);
  if (theta0 < 0.006 && omega0 < 0.08) return [];

  const m = LEADED_CAME_PRISM_DEFAULTS.massKg;
  const bPivot = 0.5 * LEADED_CAME_PRISM_DEFAULTS.widthM - LEADED_CAME_PRISM_DEFAULTS.bevelRadiusM; // 11.25 mm
  const hHalf = 0.5 * LEADED_CAME_PRISM_DEFAULTS.heightM; // 7.5 mm
  const rRock = Math.hypot(bPivot, hHalf);
  const iPivot = (m / 3.0) * (4.0 * bPivot * bPivot + 4.0 * hHalf * hHalf);
  const gEff = 9.81 * (1.0 + clamp(clampingLoadN, 0.0, 40.0) / (m * 9.81));

  // Potential-to-kinetic angular impact speed when rocking down from theta0
  const omegaPotential = Math.sqrt(
    Math.max(0.0, (2.0 * m * gEff * rRock * theta0 * (bPivot / rRock)) / iPivot)
  );
  const omegaImpact0 = Math.max(omega0, omegaPotential);
  const vn0 = clamp(2.0 * bPivot * omegaImpact0, 0.018, 0.85);

  // Housner beveled-glass angular restitution factor e_theta in (0, 1)
  const eTheta = surface === 'glass' ? 0.54 : 0.44;
  // Base Housner half-cycle interval for first rebound arc (clamped to audible 22..92 ms micro-rocking window)
  const dtBase = clamp(
    2.6 * Math.sqrt((rRock * Math.max(0.008, theta0 * eTheta)) / gEff),
    0.022,
    0.092
  );

  const tapCount = theta0 > 0.045 || vn0 > 0.09 ? 3 : 2;
  const taps: HousnerRockTap[] = [];
  let tAccum = 0.0;

  for (let k = 0; k < tapCount; k++) {
    const decayPow = Math.pow(eTheta, k);
    const vnk = Math.max(0.012, vn0 * decayPow);
    const cutoffHz = computeHertzianCutoffHz(
      vnk,
      surface,
      m,
      LEADED_CAME_PRISM_DEFAULTS.bevelRadiusM
    );
    const intensity01 = clamp((vnk - 0.008) / 0.38, 0.03, 0.65);
    taps.push({
      delaySec: tAccum,
      impactSpeedMps: vnk,
      cutoffHz,
      intensity01,
      pitchScale: 1.0 + 0.028 * k,
    });
    // Next rebound interval shrinks by e_theta < 1 (accelerating Housner settle!)
    tAccum += Math.max(0.012, dtBase * Math.pow(eTheta, k));
  }

  return taps;
}

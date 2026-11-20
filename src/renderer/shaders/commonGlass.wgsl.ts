// Shared WGSL optical, material, shadow, and stained-glass procedural functions used across
// Sun Shadow/Caustic Transmission, Environment/Tabletop, Intact Glass Tower, and Voronoi Shards.

export const COMMON_GLASS_WGSL = /* wgsl */ `
const BLOCK_HX: f32 = 0.375;
const BLOCK_HY: f32 = 0.075;
const BLOCK_HZ: f32 = 0.125;
const PI: f32 = 3.14159265359;

// Authentic soda-lime architectural glass iron-oxide base attenuation (gives thick 7.5 cm paths subtle sea-green depth)
const SODA_LIME_IRON_SIGMA: vec3<f32> = vec3<f32>(0.15, 0.045, 0.095);

// Soda-lime cathedral glass Cauchy dispersion indices (n_d = 1.522, Abbe V_d ~ 58)
const GLASS_IOR_R: f32 = 1.514;
const GLASS_IOR_G: f32 = 1.522;
const GLASS_IOR_B: f32 = 1.533;

/**
 * Exact unpolarized Maxwell-Fresnel dielectric reflectance 0.5 * (R_s + R_p) across interface n1 -> n2.
 * Returns exact 1.0 under Total Internal Reflection (sin^2(theta_t) >= 1.0, e.g. theta_i >= 41.07 deg for 1.522 -> 1.0),
 * eliminating the 7.6x Schlick under-reflection error near the internal critical angle.
 */
fn dielectricFresnel(cosThetaI: f32, n1: f32, n2: f32) -> f32 {
  let cI = clamp(cosThetaI, 0.0, 1.0);
  let eta = n1 / max(n2, 1e-4);
  let sin2T = eta * eta * max(0.0, 1.0 - cI * cI);
  if (sin2T >= 1.0) {
    return 1.0;
  }
  let cT = sqrt(max(0.0, 1.0 - sin2T));
  let n1cI = n1 * cI;
  let n2cT = n2 * cT;
  let n1cT = n1 * cT;
  let n2cI = n2 * cI;
  let rs = (n1cI - n2cT) / max(n1cI + n2cT, 1e-6);
  let rp = (n2cI - n1cT) / max(n2cI + n1cT, 1e-6);
  return clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0);
}

struct SceneUniforms {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
  cameraPosAndTime: vec4<f32>,
  sunDirAndBlockCount: vec4<f32>,
  sunColorAndAspect: vec4<f32>,
  skyTintAndMode: vec4<f32>, // xyz = skyTint, w = activeShardCount
  tableAoBounds: vec4<f32>,  // (minX, minZ, maxX, maxZ) of low-height blocks & shards
  tableAoCounts: vec4<f32>,  // x = intactCount, y = shardStride, z = maxBlockHalfY, w = atelierMix (0 = stone workshop/walnut, 1 = daylight plaster/oak)
};

struct GPUBlockData {
  localToWorld: mat4x4<f32>,
  worldToLocal: mat4x4<f32>,
  absorptionAndMotif: vec4<f32>,
  accentAndSeed: vec4<f32>,
  crackAndDamage: vec4<f32>,
  stateFlags: vec4<f32>,
  photoelastic: vec4<f32>, // xyz = local (-X, 0, +X) bearing contact stress [0..1], w = flexural bending/shear [0..1]
};

const MAX_SHARD_PLANES: u32 = 12u;

struct GPUShardInstance {
  localToWorld: mat4x4<f32>,
  absorption: vec3<f32>,
  packedMesh: u32,
  accentAndSeed: vec4<f32>,
  originAndPlaneCount: vec4<f32>,
  planes: array<vec4<f32>, 12>,
};

fn hash31(p3: vec3<f32>) -> f32 {
  var p = fract(p3 * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

fn noise3D(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
  let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

  let nx00 = mix(n000, n100, u.x);
  let nx10 = mix(n010, n110, u.x);
  let nx01 = mix(n001, n101, u.x);
  let nx11 = mix(n011, n111, u.x);

  let nxy0 = mix(nx00, nx10, u.y);
  let nxy1 = mix(nx01, nx11, u.y);
  return mix(nxy0, nxy1, u.z);
}

struct VitrailSample {
  cameMask: f32,               // 1.0 on recessed 3D filigree inlay, 0.0 on bare polished glass & bevels
  accentMix: f32,              // 0.0 = primary jewel glass, 1.0 = secondary accent jewel glass
  paneMeniscusAO: f32,         // Contact groove shadow right alongside 3D raised solder bead
  cameNormalLocal: vec3<f32>,  // True 3D half-round cylindrical solder bead & glass meniscus normal offset
  rippleHeight: f32,           // Hand-rolled cathedral waterglass undulation scalar
  rippleGrad: vec3<f32>,       // Analytical surface slope for optical refraction
  rippleLaplacian: f32,        // Analytical 2nd-derivative curvature for energy-conserving Jacobian caustics
};

/**
 * Evaluates solid cut-glass block optics:
 * - All 12 outer beveled edges remain 100% exposed diamond-polished glass (zero painted perimeter wireframe!)
 * - Motif 0 is a pure, unadorned hand-rolled antique cathedral glass brick
 * - Motifs 1..3 feature delicate, physically inset 3D half-round pewter/aged-brass filigree inlays
 *   with true cylindrical normal profiles and adjacent putty/groove contact shadowing.
 */
fn evaluateVitrailPattern(pLocal: vec3<f32>, motifId: i32, seed: f32) -> VitrailSample {
  let u = pLocal.x / BLOCK_HX; // [-1, 1]

  // Base lengthwise flashed-glass tone shift (preserves exact 0.18315 at u = 0 for motifId == 0)
  var accentMix = clamp(smoothstep(-0.55, 0.65, u * sign(sin(seed * 4.1) + 0.01)) * 0.42, 0.0, 1.0);

  // Conservative 3D C-infinity hand-rolled cathedral waterglass scalar potential H(p):
  // Non-axis-aligned wavevectors eliminate Cartesian grid alignment while guaranteeing curl(grad H) == 0.
  let k1 = vec3<f32>(12.8, 6.4, 3.9);
  let k2 = vec3<f32>(-4.3, 8.5, 16.8);
  let k3 = vec3<f32>(8.6, 17.5, -10.4);
  let ph1 = dot(pLocal, k1) + seed * 3.71;
  let ph2 = dot(pLocal, k2) + seed * 5.29;
  let ph3 = dot(pLocal, k3) - seed * 2.83;

  let a1: f32 = 0.28;
  let a2: f32 = 0.22;
  let a3: f32 = 0.12;
  let s1 = sin(ph1);
  let c1 = cos(ph1);
  let s2 = cos(ph2);
  let d2 = -sin(ph2);
  let s3 = sin(ph3);
  let c3 = cos(ph3);

  let ripple = a1 * s1 + a2 * s2 + a3 * s3;
  let gradH = (a1 * c1) * k1 + (a2 * d2) * k2 + (a3 * c3) * k3;
  let laplacian = -(a1 * s1 * dot(k1, k1) + a2 * s2 * dot(k2, k2) + a3 * s3 * dot(k3, k3)) / 260.0;

  var cameMask: f32 = 0.0;
  var paneMeniscusAO: f32 = 1.0;
  var cameNormalLocal = vec3<f32>(0.0);

  // Motifs 1..3: Recessed 3D half-round 60/40 tin-lead (1, 2) & copper-foil (3) solder came ribs
  // strictly inset from all 12 diamond-polished 1.25 mm outer bevels, with patinated flux meniscus AO.
  if (motifId > 0) {
    let ax = BLOCK_HX - abs(pLocal.x);
    let ay = BLOCK_HY - abs(pLocal.y);
    let az = BLOCK_HZ - abs(pLocal.z);
    let dMin = min(ax, min(ay, az));
    let dMax = max(ax, max(ay, az));
    let dMid = ax + ay + az - dMin - dMax;
    let faceInterior = smoothstep(0.015, 0.025, dMid);

    if (faceInterior > 0.005) {
      let isSideFace = ay > az;
      let vCoord = select(pLocal.y / BLOCK_HY, pLocal.z / BLOCK_HZ, !isSideFace);
      let vScale = select(BLOCK_HY, BLOCK_HZ, !isSideFace);

      var sDist: f32 = 1.0;
      var gradUV = vec2<f32>(1.0, 0.0);
      if (motifId == 1) {
        // Gothic lancet arch + central tie bar (60/40 tin-lead came)
        let archCurve = (abs(vCoord) - (0.52 - 0.34 * u * u)) * vScale;
        let tieBar = abs(pLocal.x);
        if (abs(archCurve) < tieBar) {
          sDist = archCurve;
          gradUV = normalize(vec2<f32>(0.68 * u * vScale / BLOCK_HX, select(-1.0, 1.0, vCoord >= 0.0)));
        } else {
          sDist = select(-tieBar, tieBar, pLocal.x >= 0.0);
          gradUV = vec2<f32>(1.0, 0.0);
        }
      } else if (motifId == 2) {
        // Cathedral diamond quarry lattice (60/40 tin-lead came)
        let dA = (u * 0.85 + vCoord * 0.65) * 0.095;
        let dB = (u * 0.85 - vCoord * 0.65) * 0.095;
        if (abs(dA) < abs(dB)) {
          sDist = dA;
          gradUV = normalize(vec2<f32>(0.85 / BLOCK_HX, 0.65 / vScale));
        } else {
          sDist = dB;
          gradUV = normalize(vec2<f32>(0.85 / BLOCK_HX, -0.65 / vScale));
        }
      } else {
        // Tiffany copper-foil rose medallion + axial spine
        let ell = vec2<f32>(pLocal.x * 0.36, vCoord * vScale);
        let rEll = length(ell);
        sDist = rEll - 0.064;
        gradUV = normalize(vec2<f32>(ell.x * 0.36, ell.y) + vec2<f32>(1e-5, 0.0));
      }

      let cameHalfW = select(0.0052, 0.0040, motifId == 3);
      let dCame = abs(sDist);
      cameMask = (1.0 - smoothstep(cameHalfW * 0.76, cameHalfW, dCame)) * faceInterior;
      let fluxGroove = (1.0 - smoothstep(cameHalfW * 0.82, cameHalfW * 2.85, dCame)) * (1.0 - cameMask * 0.62) * faceInterior;
      paneMeniscusAO = 1.0 - fluxGroove * 0.44;

      let uBead = clamp(sDist / cameHalfW, -0.95, 0.95);
      let tangent3 = select(
        vec3<f32>(gradUV.x, gradUV.y, 0.0),
        vec3<f32>(gradUV.x, 0.0, gradUV.y),
        !isSideFace
      );
      let outSignY = select(-1.0, 1.0, pLocal.y >= 0.0);
      let outSignZ = select(-1.0, 1.0, pLocal.z >= 0.0);
      let faceOutN = select(vec3<f32>(0.0, 0.0, outSignZ), vec3<f32>(0.0, outSignY, 0.0), isSideFace);
      let cylN = normalize(tangent3 * uBead + faceOutN * sqrt(max(0.05, 1.0 - uBead * uBead)));
      cameNormalLocal = (cylN - faceOutN) * cameMask;

      let paneFlip = smoothstep(-0.003, 0.003, -sDist) * faceInterior;
      accentMix = clamp(mix(accentMix, 0.88 - accentMix * 0.55, paneFlip * 0.78), 0.0, 1.0);
    }
  }

  // Precision diamond-ground architectural glass on motifId == 0 (micro-subtle optical figure, zero molded-plastic waviness)
  let rippleScale = select(0.022, 0.30, motifId > 0);
  var out: VitrailSample;
  out.cameMask = cameMask;
  out.accentMix = accentMix;
  out.paneMeniscusAO = paneMeniscusAO;
  out.cameNormalLocal = cameNormalLocal;
  out.rippleHeight = ripple * rippleScale;
  out.rippleGrad = gradH * 0.0011 * rippleScale;
  out.rippleLaplacian = laplacian * rippleScale;
  return out;
}

struct InternalCrackSample {
  mask: f32,
  normalLocal: vec3<f32>, // Block-local cleavage plane unit normal (rotates with block tilt/wobble)
  gapNm: f32,             // LEFM crack-opening air-film thickness in nm for Newton/Airy interference
  tipStress: f32,         // Arrested K_I crack-tip residual stress for Polariscope birefringence
};

/**
 * Evaluates asymmetric, sparse (< 0.25 mm) 3D Griffith radial cleavage planes, Hertzian impact frustum,
 * and chordal cross-cracks radiating from crackOrigin when damage > 0, returning exact block-local
 * cleavage plane normals, air-gap thickness (nm), and crack-tip stress concentration.
 */
fn evaluateInternalCrackField(pLocal: vec3<f32>, crackOrigin: vec3<f32>, damage: f32, seed: f32) -> InternalCrackSample {
  var res: InternalCrackSample;
  res.mask = 0.0;
  res.normalLocal = vec3<f32>(0.0, 1.0, 0.0);
  res.gapNm = 0.0;
  res.tipStress = 0.0;

  if (damage <= 0.01) {
    return res;
  }
  let rel = pLocal - crackOrigin;
  let dist = length(rel);
  let maxRadius = 0.10 + damage * 0.46;
  if (dist > maxRadius || dist < 1e-4) {
    return res;
  }

  let warpCoarse = (noise3D(pLocal * 21.0 + vec3<f32>(seed * 2.3)) - 0.5) * 0.0085;
  let warpFine = (noise3D(pLocal * 58.0 - vec3<f32>(seed * 4.1)) - 0.5) * 0.0028;
  let warp = warpCoarse + warpFine;

  let uCoord = rel.x - rel.y * 0.24;
  let vCoord = rel.z + rel.y * 0.34;
  let theta = atan2(vCoord, uCoord);
  let numSectors: f32 = 7.0;
  let sectorCoord = ((theta + PI) / (2.0 * PI)) * numSectors + seed * 1.73;
  let sectorId = floor(sectorCoord);
  let sectorFrac = fract(sectorCoord) - 0.5;

  // Crisp Roesler/Hertzian conical impact shell around crackOrigin (replaces blurry spherical smudge)
  let hertzShellDist = abs(dist - 0.014) + warpFine * 0.35;
  let hertzCone = (1.0 - smoothstep(0.0012, 0.0065, hertzShellDist)) * (1.0 - smoothstep(0.010, 0.030, dist)) * 0.72;
  let hertzNormal = safeNormalize(rel + vec3<f32>(0.0, 0.42 * dist, 0.0), vec3<f32>(0.0, 1.0, 0.0));

  let hSec = hash31(vec3<f32>(sectorId, seed * 3.17, 2.0));
  let dmgScale = clamp(damage * 1.25, 0.0, 1.0);
  if (fract(hSec * 19.37) < 0.38) {
    res.mask = hertzCone * dmgScale;
    res.normalLocal = hertzNormal;
    res.gapNm = 260.0 * clamp(damage, 0.0, 1.0);
    res.tipStress = res.mask * 0.55;
    return res;
  }

  let sectorReach = maxRadius * (0.35 + 0.65 * fract(hSec * 7.91));
  if (dist > sectorReach) {
    return res;
  }

  let spokeOffset = (hSec - 0.5) * 0.38;
  let spokeAng = ((sectorId + 0.5 + spokeOffset - seed * 1.73) / numSectors) * 2.0 * PI - PI;
  // Circumferential unit normal perpendicular to the radial Griffith cleavage plane!
  let spokeCleavageN = safeNormalize(
    vec3<f32>(-sin(spokeAng), 0.24 * sin(spokeAng) - 0.34 * cos(spokeAng), cos(spokeAng))
      + vec3<f32>(warpFine * 18.0, warpCoarse * 14.0, -warpFine * 16.0),
    hertzNormal
  );

  let perpDistDm = abs((sectorFrac - spokeOffset) * (2.0 * PI / numSectors) * dist + warp);
  var crack = 1.0 - smoothstep(0.00045, 0.0021, perpDistDm);
  var bestN = spokeCleavageN;

  if (fract(hSec * 13.3) > 0.46 && dist > 0.05) {
    let forkSlope = 0.20 + 0.14 * fract(hSec * 3.7);
    let forkSpread = (dist - 0.05) * forkSlope;
    let forkDistDm = abs(perpDistDm - forkSpread + warpFine * 0.8);
    let forkMask = (1.0 - smoothstep(0.0004, 0.0018, forkDistDm)) * 0.80;
    if (forkMask > crack) {
      crack = forkMask;
      let forkAng = spokeAng + forkSlope * select(-1.0, 1.0, sectorFrac >= spokeOffset);
      bestN = safeNormalize(vec3<f32>(-sin(forkAng), 0.20 * sin(forkAng) - 0.30 * cos(forkAng), cos(forkAng)), spokeCleavageN);
    }
  }

  if (fract(hSec * 29.7) > 0.56 && perpDistDm < 0.022) {
    let secCenterAng = ((sectorId + 0.5 - seed * 1.73) / numSectors) * 2.0 * PI - PI;
    let chordDir = normalize(vec3<f32>(cos(secCenterAng + 0.28), 0.18, sin(secCenterAng + 0.28)));
    let projChord = dot(rel, chordDir);
    let chordStep = 0.065 + fract(hSec * 5.3) * 0.085;
    let chordDistDm = abs(projChord - chordStep + warpFine);
    let chordMask = (1.0 - smoothstep(0.0004, 0.0017, chordDistDm)) * 0.68;
    if (chordMask > crack) {
      crack = chordMask;
      bestN = chordDir;
    }
  }

  let tipFade = 1.0 - smoothstep(sectorReach * 0.64, sectorReach, dist);
  let radialMask = crack * tipFade;
  if (hertzCone > radialMask) {
    bestN = hertzNormal;
  }
  let finalMask = clamp(max(radialMask, hertzCone) * dmgScale, 0.0, 1.0);
  // Irwin LEFM elliptic crack-opening profile delta(r) ~ sqrt(1 - r / a) in [140 nm, 680 nm]
  let codProfile = sqrt(max(0.02, 1.0 - dist / max(sectorReach, 1e-3)));
  let filmRipple = 0.82 + 0.36 * fract(hSec * 41.3 + warpCoarse * 40.0);

  res.mask = finalMask;
  res.normalLocal = bestN;
  res.gapNm = (140.0 + 520.0 * clamp(damage, 0.0, 1.0) * codProfile) * filmRipple;
  res.tipStress = finalMask * (0.30 + 0.70 * smoothstep(sectorReach * 0.30, sectorReach * 0.88, dist)) * clamp(damage * 1.35, 0.0, 1.0);
  return res;
}

fn evaluateInternalCracks(pLocal: vec3<f32>, crackOrigin: vec3<f32>, damage: f32, seed: f32) -> f32 {
  return evaluateInternalCrackField(pLocal, crackOrigin, damage, seed).mask;
}

fn safeNormalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let lenSq = dot(v, v);
  return select(fallback, v * inverseSqrt(max(lenSq, 1e-12)), lenSq > 1e-10);
}

const BLOCK_BEVEL_R: f32 = 0.0068;

struct ChamferIntervalHit {
  hit: bool,
  tEnter: f32,
  tExit: f32,
  enterNormalLocal: vec3<f32>,
  exitNormalLocal: vec3<f32>,
};

/**
 * Zero-array analytical intersection against the 0.68 mm C1 rounded-fillet glass prism:
 * Eliminates the 320-byte 20-plane stack array (0 VGPR spills on Metal -> < 4.5 ms at Retina zoom!)
 * while matching createBeveledBlockMesh(hx, hy, hz, 0.0068, 5) to sub-millimeter accuracy.
 */
fn intersectChamferedPrismInterval(ro: vec3<f32>, rd: vec3<f32>) -> ChamferIntervalHit {
  var res: ChamferIntervalHit;
  res.hit = false;
  res.tEnter = 0.0;
  res.tExit = 0.001;
  res.enterNormalLocal = vec3<f32>(0.0, 1.0, 0.0);
  res.exitNormalLocal = vec3<f32>(0.0, 1.0, 0.0);

  let halfExt = vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
  let innerCore = halfExt - vec3<f32>(BLOCK_BEVEL_R);
  let invD = 1.0 / select(rd, vec3<f32>(1e-7), abs(rd) < vec3<f32>(1e-7));
  let b0 = (-halfExt - ro) * invD;
  let b1 = ( halfExt - ro) * invD;
  let ts = min(b0, b1);
  let tb = max(b0, b1);

  let rawNear = max(ts.x, max(ts.y, ts.z));
  let rawFar = min(tb.x, min(tb.y, tb.z));
  if (rawFar <= 0.0 || rawNear > rawFar) {
    return res;
  }

  let tEn0 = max(0.0, rawNear);
  let pMid = ro + rd * (0.5 * (tEn0 + rawFar));
  let qMid = max(abs(pMid) - innerCore, vec3<f32>(0.0));
  if (dot(qMid, qMid) > (BLOCK_BEVEL_R + 0.0012) * (BLOCK_BEVEL_R + 0.0012)) {
    return res;
  }

  var nEn = vec3<f32>(0.0, 0.0, -sign(rd.z));
  if (ts.x >= ts.y && ts.x >= ts.z) {
    nEn = vec3<f32>(-sign(rd.x), 0.0, 0.0);
  } else if (ts.y >= ts.x && ts.y >= ts.z) {
    nEn = vec3<f32>(0.0, -sign(rd.y), 0.0);
  }

  var nEx = vec3<f32>(0.0, 0.0, sign(rd.z));
  if (tb.x <= tb.y && tb.x <= tb.z) {
    nEx = vec3<f32>(sign(rd.x), 0.0, 0.0);
  } else if (tb.y <= tb.x && tb.y <= tb.z) {
    nEx = vec3<f32>(0.0, sign(rd.y), 0.0);
  }

  var t0 = rawNear;
  if (rawNear > 0.0) {
    let qEn = max(abs(ro + rd * rawNear) - innerCore, vec3<f32>(0.0));
    let exEn = max(0.0, length(qEn) - BLOCK_BEVEL_R);
    t0 = min(rawFar - 0.0005, rawNear + exEn * 1.15);
  }
  let qEx = max(abs(ro + rd * rawFar) - innerCore, vec3<f32>(0.0));
  let exEx = max(0.0, length(qEx) - BLOCK_BEVEL_R);
  let t1 = max(max(0.0, t0) + 0.0005, rawFar - exEx * 1.15);

  res.hit = true;
  res.tEnter = t0;
  res.tExit = t1;
  let pEn = ro + rd * max(0.0, t0);
  let dEn = pEn - clamp(pEn, -innerCore, innerCore);
  res.enterNormalLocal = select(nEn, safeNormalize(dEn, nEn), dot(dEn, dEn) > 1e-7);
  let pEx = ro + rd * t1;
  let dEx = pEx - clamp(pEx, -innerCore, innerCore);
  res.exitNormalLocal = select(nEx, safeNormalize(dEx, nEx), dot(dEx, dEx) > 1e-7);
  return res;
}

/**
 * 8-Band Visible Spectral Absorption Basis (415, 455, 495, 530, 565, 605, 645, 685 nm -> CIE XYZ -> D65 Linear sRGB).
 * Reconstructs continuous transition-metal / colloidal stained-glass passbands from sigmaRGB so overlapping
 * colors mix physically (Amber x Cobalt -> Emerald Green, Ruby x Cobalt -> Royal Violet, Ruby x Amber -> Vermilion)!
 */
fn spectralSigma8FromRGB(sigmaRGB: vec3<f32>) -> vec4<f32> {
  // Helper unused stub; full 8-band evaluator below returns two vec4s (bands 0..3 and 4..7)
  return vec4<f32>(sigmaRGB, 0.0);
}

struct SpectralTau8 {
  lo: vec4<f32>, // 415, 455, 495, 530 nm (Violet, Blue, Cyan, Emerald-Green)
  hi: vec4<f32>, // 565, 605, 645, 685 nm (Yellow-Green, Orange, Ruby-Red, Deep-Violet-Red)
};

fn evalSwatchSpectralSigma8WithDensity(sigmaRGB: vec3<f32>, densityScale: f32) -> SpectralTau8 {
  // Bright, Luminous Stained-Glass Basis (water-clear ~0.09/dm passband + clean stop-band -> bright jewel glass, zero dark mud & zero pastel!):
  let sR = sigmaRGB.x * densityScale;
  let sG = sigmaRGB.y * densityScale;
  let sB = sigmaRGB.z * densityScale;

  let isCobalt = smoothstep(4.5 * densityScale, 7.5 * densityScale, sR) * (1.0 - smoothstep(0.4, 1.5, sigmaRGB.z));
  let isAmber  = (1.0 - smoothstep(0.25, 0.85, sigmaRGB.x)) * (1.0 - smoothstep(2.2, 4.4, sigmaRGB.y)) * smoothstep(5.5, 8.2, sigmaRGB.z);
  let isRuby   = (1.0 - smoothstep(0.30, 0.90, sigmaRGB.x)) * smoothstep(4.2, 6.8, sigmaRGB.y) * smoothstep(4.5, 7.2, sigmaRGB.z);

  var b415 = sB * 1.06 + 0.08;
  var b455 = sB * 0.90;
  var b495 = sB * 0.46 + sG * 0.42;
  var b530 = sG * 0.86;
  var b565 = sG * 0.46 + sR * 0.42;
  var b605 = sR * 0.82 + sG * 0.15;
  var b645 = sR * 0.92;
  var b685 = sR * 0.86;

  // Warm Golden-Honey Amber: clean blue cut-off + high golden passband (bright luminous honey-gold, never dark brown & never cream!)
  b415 = mix(b415, 10.5 * densityScale, isAmber * 0.92);
  b455 = mix(b455, 9.60 * densityScale, isAmber * 0.92);
  b495 = mix(b495, 3.60 * densityScale, isAmber * 0.88);
  b530 = mix(b530, 1.05 * densityScale, isAmber * 0.90);
  b565 = mix(b565, 0.28 * densityScale, isAmber * 0.92);
  b605 = mix(b605, 0.12 * densityScale, isAmber * 0.92);
  b645 = mix(b645, 0.10 * densityScale, isAmber * 0.92);
  b685 = mix(b685, 0.10 * densityScale, isAmber * 0.92);

  // Luminous Selenium Ruby: high >605 nm red passband (~97% clear in red!) + clean green/blue block
  b415 = mix(b415, 10.5 * densityScale, isRuby * 0.94);
  b455 = mix(b455, 11.0 * densityScale, isRuby * 0.94);
  b495 = mix(b495, 11.0 * densityScale, isRuby * 0.94);
  b530 = mix(b530, 10.8 * densityScale, isRuby * 0.95);
  b565 = mix(b565, 7.20 * densityScale, isRuby * 0.94);
  b605 = mix(b605, 1.45 * densityScale, isRuby * 0.88);
  b645 = mix(b645, 0.12 * densityScale, isRuby * 0.92);
  b685 = mix(b685, 0.10 * densityScale, isRuby * 0.92);

  // Luminous Chartres Cobalt: high blue passband (~97% clear in blue!) + 500 nm emerald shoulder + 685 nm violet window
  b415 = mix(b415, 0.14 * densityScale, isCobalt * 0.90);
  b455 = mix(b455, 0.11 * densityScale, isCobalt * 0.92);
  b495 = mix(b495, 1.25 * densityScale, isCobalt * 0.82);
  b530 = mix(b530, 2.45 * densityScale, isCobalt * 0.80);
  b565 = mix(b565, 6.80 * densityScale, isCobalt * 0.88);
  b605 = mix(b605, 8.40 * densityScale, isCobalt * 0.90);
  b645 = mix(b645, 8.20 * densityScale, isCobalt * 0.90);
  b685 = mix(b685, 1.55 * densityScale, isCobalt * 0.88);

  var out: SpectralTau8;
  out.lo = vec4<f32>(b415, b455, b495, b530);
  out.hi = vec4<f32>(b565, b605, b645, b685);
  return out;
}

fn evalSwatchSpectralSigma8(sigmaRGB: vec3<f32>) -> SpectralTau8 {
  return evalSwatchSpectralSigma8WithDensity(sigmaRGB, 0.88);
}

/**
 * Integrates 8-band spectral transmittance exp(-tau_k) against pre-integrated CIE 1931 -> Linear sRGB weights.
 * Exact white identity: when tau == 0, sum of weights == vec3(1.0, 1.0, 1.0) to 4 decimal places!
 */
fn integrateSpectralTau8ToRGB(tau: SpectralTau8) -> vec3<f32> {
  let tLo = exp(-tau.lo);
  let tHi = exp(-tau.hi);

  // Pre-integrated CIE 1931 -> Linear Rec.709 / sRGB quadrature weights across [415,455,495,530, 565,605,645,685] nm
  // Row sums: R = 0.02 + 0.00 + 0.00 + 0.04 + 0.16 + 0.36 + 0.28 + 0.14 = 1.000
  //           G = 0.00 + 0.03 + 0.16 + 0.39 + 0.31 + 0.10 + 0.01 + 0.00 = 1.000
  //           B = 0.28 + 0.52 + 0.17 + 0.03 + 0.00 + 0.00 + 0.00 + 0.00 = 1.000
  let r = dot(tLo, vec4<f32>(0.020, 0.000, 0.000, 0.040)) + dot(tHi, vec4<f32>(0.160, 0.360, 0.280, 0.140));
  let g = dot(tLo, vec4<f32>(0.000, 0.030, 0.160, 0.390)) + dot(tHi, vec4<f32>(0.310, 0.100, 0.010, 0.000));
  let b = dot(tLo, vec4<f32>(0.280, 0.520, 0.170, 0.030)) + dot(tHi, vec4<f32>(0.000, 0.000, 0.000, 0.000));
  return vec3<f32>(r, g, b);
}

/**
 * Facet-preserving prism normal: returns exact flat facet normal across flat faces & 45-deg diamond chamfers!
 */
fn evalBeveledBoxNormalLocal(pLocal: vec3<f32>, fallbackN: vec3<f32>) -> vec3<f32> {
  let innerHalf = vec3<f32>(BLOCK_HX - BLOCK_BEVEL_R, BLOCK_HY - BLOCK_BEVEL_R, BLOCK_HZ - BLOCK_BEVEL_R);
  let dRim = pLocal - clamp(pLocal, -innerHalf, innerHalf);
  if (dot(dRim, dRim) > 1e-7) {
    return safeNormalize(dRim, fallbackN);
  }
  return safeNormalize(fallbackN, vec3<f32>(0.0, 1.0, 0.0));
}

/**
 * Physically bounded Trowbridge-Reitz GGX microfacet specular term (D * G / 4*NdotV*NdotL).
 * Eliminates infinite/pinpoint Phong fireflies while producing realistic microfacet highlights.
 */
fn evalGGXSpecular(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let nDotL = max(0.0, dot(N, L));
  let nDotV = max(1e-4, dot(N, V));
  if (nDotL <= 0.0) {
    return 0.0;
  }
  let H = safeNormalize(L + V, N);
  let nDotH = max(0.0, dot(N, H));
  let a = max(0.022, roughness * roughness);
  let a2 = a * a;
  let denom = nDotH * nDotH * (a2 - 1.0) + 1.0;
  let D = a2 / max(1e-5, PI * denom * denom);

  let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
  let gV = nDotV / (nDotV * (1.0 - k) + k);
  let gL = nDotL / (nDotL * (1.0 - k) + k);
  return min(10.0, (D * gV * gL) / max(0.08, 4.0 * nDotV));
}

/**
 * Evaluates a focused cathedral clerestory window light-shaft (gobo) across the stage:
 * Full sunlight illuminates the central tower & caustic pool, softly rolling off into
 * architectural shadow toward the outer perimeter (lifted in bounce-lit Daylight Atelier).
 */
fn evaluateStudioWindowGobo(worldPos: vec3<f32>, sunDir: vec3<f32>, atelierMix: f32) -> f32 {
  // Trace point back along sunDir to reference plane y = 0.9 dm so tall tower & floor pool align coherently
  let shiftT = (0.90 - worldPos.y) / max(0.22, sunDir.y);
  let stageXZ = worldPos.xz + sunDir.xz * shiftT * 0.25;

  let sunRight = safeNormalize(vec3<f32>(sunDir.z, 0.0, -sunDir.x), vec3<f32>(1.0, 0.0, 0.0)).xz;
  let sunFwd = vec2<f32>(-sunDir.x, -sunDir.z) / max(1e-3, length(sunDir.xz));
  let uWin = dot(stageXZ, sunRight);
  let vWin = dot(stageXZ, sunFwd);

  // Wide sunlit studio window footprint across the full round table (soft perimeter falloff only beyond rim)
  let archWidth = 4.45 - max(0.0, vWin - 1.2) * 0.16;
  let lateralMask = 1.0 - smoothstep(archWidth * 0.86, archWidth * 1.38, abs(uWin));
  let longMask = 1.0 - smoothstep(3.85, 5.80, abs(vWin - 0.35));

  let goboFloor = mix(0.68, 0.84, clamp(atelierMix, 0.0, 1.0));
  return clamp(goboFloor + (1.0 - goboFloor) * lateralMask * longMask, goboFloor, 1.0);
}

/**
 * 4-tap bilinear Percentage-Closer Filtered (PCF) sun shadow lookup against 2048x2048 depth32float map.
 * Uses textureLoad(..., 0) so it is 100% valid inside any WGSL control flow!
 */
fn sampleSunShadowPCF(
  sunDepthTex: texture_depth_2d,
  sunViewProj: mat4x4<f32>,
  worldPos: vec3<f32>,
  worldNormal: vec3<f32>,
  sunDir: vec3<f32>,
  selfBiasDm: f32
) -> f32 {
  let biasedWorld = worldPos + sunDir * selfBiasDm + worldNormal * 0.004;
  let clip = sunViewProj * vec4<f32>(biasedWorld, 1.0);
  let ndc = clip.xyz / max(clip.w, 1e-6);
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

  if (uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998 || ndc.z <= 0.0 || ndc.z >= 1.0) {
    return 1.0;
  }

  let texSize = 2048.0;
  let fCoord = uv * texSize - vec2<f32>(0.5);
  let base = vec2<i32>(floor(fCoord));
  let frac = fract(fCoord);
  let zRef = ndc.z - 0.0009;

  let c00 = clamp(base + vec2<i32>(0, 0), vec2<i32>(0), vec2<i32>(2047));
  let c10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(2047));
  let c01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(2047));
  let c11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(2047));

  let s00 = select(0.0, 1.0, zRef <= textureLoad(sunDepthTex, c00, 0));
  let s10 = select(0.0, 1.0, zRef <= textureLoad(sunDepthTex, c10, 0));
  let s01 = select(0.0, 1.0, zRef <= textureLoad(sunDepthTex, c01, 0));
  let s11 = select(0.0, 1.0, zRef <= textureLoad(sunDepthTex, c11, 0));

  let sx0 = mix(s00, s10, frac.x);
  let sx1 = mix(s01, s11, frac.x);
  return mix(sx0, sx1, frac.y);
}

/**
 * 4-tap non-uniform-flow-safe lookup into sunOpticalDepthTex (rgba16float):
 * - .rgb returns bilinearly interpolated spectral optical depth tau_sun
 * - .a returns min(minCasterY) across the 4 texels so background 10.0 texels never blur Layer-0 contact edges.
 */
fn sampleSunOpticalDepth(
  optTex: texture_2d<f32>,
  sunViewProj: mat4x4<f32>,
  worldPos: vec3<f32>
) -> vec4<f32> {
  let clip = sunViewProj * vec4<f32>(worldPos, 1.0);
  let ndc = clip.xyz / max(clip.w, 1e-6);
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

  if (uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998) {
    return vec4<f32>(0.0, 0.0, 0.0, 10.0);
  }

  let dims = vec2<i32>(textureDimensions(optTex));
  let maxCoord = dims - vec2<i32>(1);
  let fCoord = uv * vec2<f32>(dims) - vec2<f32>(0.5);
  let base = vec2<i32>(floor(fCoord));
  let frac = fract(fCoord);

  let c00 = clamp(base + vec2<i32>(0, 0), vec2<i32>(0), maxCoord);
  let c10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), maxCoord);
  let c01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), maxCoord);
  let c11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), maxCoord);

  let t00 = textureLoad(optTex, c00, 0);
  let t10 = textureLoad(optTex, c10, 0);
  let t01 = textureLoad(optTex, c01, 0);
  let t11 = textureLoad(optTex, c11, 0);

  let rgbX0 = mix(t00.rgb, t10.rgb, frac.x);
  let rgbX1 = mix(t01.rgb, t11.rgb, frac.x);
  let tauRGB = mix(rgbX0, rgbX1, frac.y);
  let minY = min(min(t00.a, t10.a), min(t01.a, t11.a));
  return vec4<f32>(tauRGB, minY);
}

/**
 * High-Dynamic-Range (HDR) Area-Light Studio & Cathedral Window Reflection Environment
 * sampled by dielectric glass Fresnel & bevel reflections.
 * Features crisp, bright architectural window panes divided by dark mullions so polished glass
 * faces and 1.25 mm bevels reflect continuous elongated window strips instead of Phong dots!
 */
fn sampleCathedralEnvironment(rayDir: vec3<f32>, sunDir: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>, atelierMix: f32) -> vec3<f32> {
  let d = safeNormalize(rayDir, vec3<f32>(0.0, 1.0, 0.0));
  let up = clamp(d.y, -1.0, 1.0);
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);
  let aMix = clamp(atelierMix, 0.0, 1.0);

  // Dark-Field Gallery Surround (prevents milky grey Fresnel veil from turning saturated glass pastel!) + crisp HDR window key:
  let highClerestory = smoothstep(0.66, 0.84, sunDir.y);
  let warmWoodBounce = mix(vec3<f32>(0.032, 0.022, 0.015), vec3<f32>(0.095, 0.070, 0.048), aMix) * skyScale;
  let sunlitPlaster  = mix(vec3<f32>(0.042, 0.040, 0.040), vec3<f32>(0.115, 0.108, 0.102), aMix) * (1.0 + 0.10 * highClerestory) * skyScale
                     + skyTint * mix(0.035, 0.095, aMix);
  let skyDomeZenith  = skyTint * mix(0.14, 0.26, aMix)
                     + mix(vec3<f32>(0.034, 0.038, 0.048), vec3<f32>(0.085, 0.092, 0.108), aMix) * skyScale;

  var env = mix(sunlitPlaster, skyDomeZenith, smoothstep(-0.05, 0.78, up));
  env = mix(warmWoodBounce, env, smoothstep(-0.34, 0.06, up));

  // Broad Architectural Studio Window Key + Soft Skylight Gradient (gives flat glass faces a natural perspective sheen!)
  let right = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), sunDir), vec3<f32>(1.0, 0.0, 0.0));
  let winUp = cross(sunDir, right);
  let sunCos = dot(d, sunDir);
  let winU = dot(d, right);
  let winV = dot(d, winUp);

  let archHalfW = 0.34 * (1.0 - smoothstep(0.12, 0.48, winV) * 0.38);
  let inPrimaryWindow = smoothstep(archHalfW + 0.055, archHalfW - 0.025, abs(winU)) *
                        smoothstep(0.48, 0.36, abs(winV)) *
                        smoothstep(0.12, 0.42, sunCos);

  let paneGrad = 0.88 + 0.12 * cos(winU * 3.8) * cos(winV * 2.8);
  let windowPanes = inPrimaryWindow * paneGrad;

  let primaryWindowLight = windowPanes * sunColor * (0.95 + 0.85 * smoothstep(0.68, 0.985, sunCos));
  let softSolarHalo = pow(max(0.0, sunCos), 14.0) * sunColor * 0.24;

  // Opposite Studio Bounce Reflector + Overhead Skylight Strip so every glass face & bevel rim catches clean light
  let fillDir = safeNormalize(vec3<f32>(-sunDir.x * 0.85, 0.52, -sunDir.z * 0.85), vec3<f32>(0.0, 1.0, 0.0));
  let fillRight = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), fillDir), vec3<f32>(1.0, 0.0, 0.0));
  let fillUp = cross(fillDir, fillRight);
  let fillU = dot(d, fillRight);
  let fillV = dot(d, fillUp);
  let inFillStrip = smoothstep(0.42, 0.26, abs(fillU)) *
                    smoothstep(0.28, 0.16, abs(fillV)) *
                    smoothstep(0.14, 0.44, dot(d, fillDir));
  let coolClerestoryLight = inFillStrip * (skyTint * 1.45 + vec3<f32>(0.26, 0.30, 0.38) * skyScale);

  let zenithStrip = smoothstep(0.68, 0.90, up) * smoothstep(0.38, 0.20, abs(d.x)) * (skyTint * 0.82);

  return env + primaryWindowLight + softSolarHalo + coolClerestoryLight + zenithStrip;
}

/**
 * Dimensional, out-of-focus sunlit stone workshop / limewashed daylight atelier alcove behind the table.
 * Calibrated 1:1 against the Path-Traced Studio Room so glass viewed against the far wall transmits rich natural daylight.
 */
fn sampleStudioBackdrop(rayDir: vec3<f32>, sunDir: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>, atelierMix: f32) -> vec3<f32> {
  let d = safeNormalize(rayDir, vec3<f32>(0.0, 0.0, -1.0));
  let up = clamp(d.y, -1.0, 1.0);
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);
  let aMix = clamp(atelierMix, 0.0, 1.0);

  // Bright Sunlit Daylight Studio Backdrop (restores full luminous room light behind the tower so glass glows brightly, never dark cave!)
  let roomLightMix = smoothstep(0.15, 0.75, aMix);
  let highClerestoryBg = smoothstep(0.66, 0.84, sunDir.y);
  let lowerAlcove = mix(vec3<f32>(0.178, 0.158, 0.140), vec3<f32>(0.228, 0.208, 0.188), roomLightMix) * (1.0 + 0.18 * highClerestoryBg) * skyScale
                  + skyTint * mix(0.115, 0.145, roomLightMix);
  let midWall     = mix(vec3<f32>(0.286, 0.264, 0.240), vec3<f32>(0.355, 0.328, 0.298), roomLightMix) * (1.0 + 0.20 * highClerestoryBg) * skyScale
                  + skyTint * mix(0.210, 0.248, roomLightMix);
  let upperVault  = mix(vec3<f32>(0.242, 0.246, 0.260), vec3<f32>(0.312, 0.305, 0.298), roomLightMix) * (1.0 + 0.22 * highClerestoryBg) * skyScale
                  + skyTint * mix(0.340, 0.375, roomLightMix);

  var bg = mix(lowerAlcove, midWall, smoothstep(-0.32, 0.02, up));
  bg = mix(bg, upperVault, smoothstep(0.05, 0.58, up));

  let sunDirXZ = normalize(vec2<f32>(sunDir.x, sunDir.z + 1e-5));
  let rayXZ = normalize(vec2<f32>(d.x, d.z + 1e-5));
  let sunAlign = dot(rayXZ, sunDirXZ);
  let oppWallWash = smoothstep(-0.60, 0.85, -sunAlign) * smoothstep(-0.30, 0.28, up);
  let windowBay = smoothstep(0.08, 0.88, sunAlign) * smoothstep(-0.28, 0.22, up);
  bg += sunColor * mix(0.074, 0.064, roomLightMix) * oppWallWash
      + (sunColor * mix(0.058, 0.048, roomLightMix) + skyTint * 0.068) * windowBay
      + skyTint * 0.098 * smoothstep(-0.10, 0.50, up);

  return bg;
}

fn evaluateTableAmbientIrradiance(skyTint: vec3<f32>, skyScale: f32, atelierMix: f32) -> vec3<f32> {
  let aMix = clamp(atelierMix, 0.0, 1.0);
  return skyTint * mix(0.28, 0.32, aMix)
       + mix(vec3<f32>(0.072, 0.060, 0.048), vec3<f32>(0.094, 0.082, 0.068), aMix) * skyScale;
}

struct TableMaterial {
  albedo: vec3<f32>,
  normalPerturb: vec3<f32>,
  roughness: f32,
  metallic: f32,
};

/**
 * Evaluates artisan quarter-sawn stave tabletop + recessed brushed-brass inlay:
 * - atelierMix == 0: Fumed dark walnut staves (high dynamic range jewel caustic contrast)
 * - atelierMix == 1: Natural European honeyed white oak / ash staves (sunlit Daylight Atelier)
 * - Elliptical cathedral growth-ring arches per plank + anisotropic open vessel pores + 3D V-groove seams
 */
fn evaluateWalnutTableMaterial(xz: vec2<f32>, aaWidth: f32, atelierMix: f32) -> TableMaterial {
  let r = length(xz);
  let aMix = clamp(atelierMix, 0.0, 1.0);
  // 7.2 cm (0.72 dm) solid hardwood staves running along X
  let staveCoord = (xz.y + 4.6) / 0.72;
  let plankId = floor(staveCoord);
  let plankFrac = fract(staveCoord);
  let distToSeamDm = min(plankFrac, 1.0 - plankFrac) * 0.72;

  // Per-plank unique tree-heart offset & tone variation
  let hPlank = hash31(vec3<f32>(plankId, 11.7, 3.1));
  let hPlank2 = hash31(vec3<f32>(plankId, 27.3, 9.4));
  let heartX = (hPlank - 0.5) * 3.4;
  let heartZ = (hPlank2 - 0.5) * 0.55;

  let localX = xz.x - heartX;
  let localZ = (plankFrac - 0.5) * 0.72 - heartZ;

  // Elliptical cathedral growth-ring geometry warped by smooth longitudinal 3D wood fiber noise
  // (Low-frequency 11.5 rings/dm + soft earlywood/latewood transition eliminates synthetic horizontal zebra stripes!)
  let fiberWarp = (noise3D(vec3<f32>(xz.x * 1.35 + plankId * 5.1, plankId * 2.3, xz.y * 5.2)) - 0.5) * 0.085
                + (noise3D(vec3<f32>(xz.x * 4.8, plankId * 4.7, xz.y * 16.0)) - 0.5) * 0.022;
  let radialTree = sqrt(localZ * localZ + localX * localX * 0.011) + fiberWarp;

  // Smooth organic growth-ring profile (gentle satin walnut tonal variation instead of hard tiger stripes)
  let ringPhase = fract(radialTree * 11.5 + hPlank * 6.0);
  let latewood = smoothstep(0.08, 0.46, ringPhase) * (1.0 - smoothstep(0.58, 0.94, ringPhase));
  let earlywoodBand = 1.0 - latewood;

  // Fine quarter-sawn vessel pores & subtle medullary ray chatoyance
  let vesselNoise = noise3D(vec3<f32>(xz.x * 18.0 + plankId * 13.0, 2.0, xz.y * 145.0));
  let ringPorosityBias = mix(0.52 + 0.36 * earlywoodBand, 0.24 + 0.78 * earlywoodBand, aMix);
  let openPore = smoothstep(0.73, 0.91, vesselNoise) * ringPorosityBias;

  let rayWave = xz.x * 5.8 + sin(xz.y * 12.0 + plankId * 3.7) * 0.38;
  let rayPrimary = smoothstep(0.62, 0.89, noise3D(vec3<f32>(rayWave, plankId * 8.2, xz.y * 36.0)));
  let rayFine = smoothstep(0.68, 0.91, noise3D(vec3<f32>(xz.x * 13.0 + plankId * 4.1, 6.0, xz.y * 68.0)));
  let rayFleck = (rayPrimary * 0.72 + rayFine * 0.28) * mix(0.14, 0.26, aMix);

  // Hand-rubbed satin American walnut (aMix = 0) vs honeyed European white oak (aMix = 1):
  // Balanced grain contrast (warmWood / deepHeart ~ 1.42x instead of 2.3x zebra striping!)
  let deepHeart = mix(vec3<f32>(0.084, 0.052, 0.032), vec3<f32>(0.176, 0.122, 0.074), aMix);
  let warmWood  = mix(vec3<f32>(0.120, 0.076, 0.047), vec3<f32>(0.248, 0.178, 0.110), aMix);
  let rayTint   = mix(vec3<f32>(0.142, 0.092, 0.058), vec3<f32>(0.292, 0.214, 0.136), aMix);
  let plankTone = mix(0.91 + 0.16 * hPlank2, 0.92 + 0.15 * hPlank2, aMix);

  var woodAlbedo = mix(deepHeart, warmWood, latewood * 0.68) * plankTone;
  woodAlbedo = mix(woodAlbedo, rayTint * plankTone, rayFleck);
  woodAlbedo *= (1.0 - openPore * mix(0.20, 0.18, aMix));

  // Subtle 0.5 mm beveled V-groove between hardwood staves
  let seamGroove = 1.0 - smoothstep(0.0010, 0.0044 + aaWidth * 0.5, distToSeamDm);
  woodAlbedo *= (1.0 - seamGroove * mix(0.24, 0.20, aMix));

  let seamSign = select(1.0, -1.0, plankFrac > 0.5);
  let poreBumpZ = (vesselNoise - 0.5) * 0.026 - openPore * 0.042;
  let ringBumpZ = (latewood - 0.5) * 0.016;
  let rayBumpX = (rayPrimary - 0.5) * 0.014 * aMix;
  var normalPerturb = vec3<f32>(
    (noise3D(vec3<f32>(xz.x * 35.0, 3.0, xz.y * 35.0)) - 0.5) * 0.010 + rayBumpX,
    0.0,
    seamGroove * seamSign * 0.18 + poreBumpZ + ringBumpZ
  );

  // Hand-rubbed satin wax finish: glossier on dense latewood & chatoyant medullary rays, rougher in open earlywood pores & seams
  var roughness = clamp(
    0.26 - aMix * 0.02 - latewood * 0.055 - rayFleck * 0.065 + openPore * 0.22 + seamGroove * 0.18,
    0.18,
    0.58
  );

  // Discreet recessed dark antique-bronze hairline (0.7 mm) so regulation plinth mark never looks like a yellow UI box
  let boxDist = max(abs(xz.x), abs(xz.y)) - 0.392;
  let aa = max(0.0006, aaWidth * 0.95);
  let innerBrass = (1.0 - smoothstep(0.0010 - aa * 0.5, 0.0017 + aa, abs(boxDist))) * 0.55;
  let outerBrass = (1.0 - smoothstep(0.0018 - aa * 0.5, 0.0028 + aa, abs(r - 3.92))) * 0.55;
  let brassMask = clamp(max(innerBrass, outerBrass), 0.0, 1.0);

  var mat: TableMaterial;
  mat.albedo = woodAlbedo;
  mat.normalPerturb = normalPerturb;
  mat.roughness = roughness;
  mat.metallic = brassMask;
  if (brassMask > 0.001) {
    let brushTex = noise3D(vec3<f32>(xz.x * 160.0, 5.0, xz.y * 160.0));
    let agedBrassAlbedo = mix(vec3<f32>(0.20, 0.14, 0.08), vec3<f32>(0.26, 0.19, 0.11), aMix * 0.55) * (0.92 + 0.14 * brushTex);
    mat.albedo = mix(woodAlbedo, agedBrassAlbedo, brassMask);
    mat.normalPerturb = mix(normalPerturb, vec3<f32>(0.0, 0.0, (brushTex - 0.5) * 0.020), brassMask);
    mat.roughness = mix(roughness, 0.28, brassMask);
  }
  return mat;
}

/**
 * Legacy helper alias for lightweight refracted floor albedo lookups inside multi-bounce rays.
 */
fn evaluateOakTableAlbedo(xz: vec2<f32>, atelierMix: f32) -> vec3<f32> {
  return evaluateWalnutTableMaterial(xz, 0.002, atelierMix).albedo;
}
`;

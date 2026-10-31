// Shared WGSL optical & stained-glass procedural functions used across
// Sun Caustic Transmission, Intact Glass Tower Raytracing, and Voronoi Glass Shards.

export const COMMON_GLASS_WGSL = /* wgsl */ `
const BLOCK_HX: f32 = 0.375;
const BLOCK_HY: f32 = 0.075;
const BLOCK_HZ: f32 = 0.125;
const PI: f32 = 3.14159265359;

struct SceneUniforms {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
  cameraPosAndTime: vec4<f32>,
  sunDirAndBlockCount: vec4<f32>,
  sunColorAndAspect: vec4<f32>,
  skyTintAndMode: vec4<f32>,
};

struct GPUBlockData {
  localToWorld: mat4x4<f32>,
  worldToLocal: mat4x4<f32>,
  absorptionAndMotif: vec4<f32>,
  accentAndSeed: vec4<f32>,
  crackAndDamage: vec4<f32>,
  stateFlags: vec4<f32>,
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
  cameMask: f32,      // 1.0 on soldered lead/copper-foil came, 0.0 on glass pane
  accentMix: f32,     // 0.0 = primary jewel glass, 1.0 = secondary accent jewel glass
  rippleHeight: f32,  // Cathedral waterglass undulation scalar
  rippleGrad: vec3<f32>,
};

/**
 * Evaluates soldered perimeter copper-foil/lead came, Gothic interior mullion motif,
 * and hand-rolled cathedral waterglass surface ripples in local block space.
 */
fn evaluateVitrailPattern(pLocal: vec3<f32>, motifId: i32, seed: f32) -> VitrailSample {
  let ax = BLOCK_HX - abs(pLocal.x);
  let ay = BLOCK_HY - abs(pLocal.y);
  let az = BLOCK_HZ - abs(pLocal.z);

  // Distance to any of the 12 edges of the block (second smallest distance to the 3 face pairs)
  let dMin = min(ax, min(ay, az));
  let dMax = max(ax, max(ay, az));
  let dMid = ax + ay + az - dMin - dMax;

  // Perimeter copper-foil / pewter solder border (~0.8 mm wide along the 12 outer edges)
  let borderCame = 1.0 - smoothstep(0.0062, 0.0092, dMid);

  let u = pLocal.x / BLOCK_HX; // [-1, 1]
  let v = pLocal.z / BLOCK_HZ; // [-1, 1]

  var innerLineDist: f32 = 1.0;
  var accentMix: f32 = 0.0;

  // Strictly restrict interior lead cames to the actual top & bottom exterior glass faces (ay < 0.004)
  // so looking through the side of a glass block shows pure crystal-clear colored glass without dark interior bands!
  let onTopOrBottomSurface = 1.0 - smoothstep(0.0025, 0.0055, ay);

  if (motifId == 0) {
    // Motif 0: Medieval Quarry — subtle end-chevron diamond quarries
    let q = abs(abs(u) - 0.62) * 0.28 + abs(v) * 0.09;
    innerLineDist = abs(q - 0.068);
    accentMix = 1.0 - smoothstep(0.064, 0.072, q);
  } else if (motifId == 1) {
    // Motif 1: Diamond Lozenge Vitrail
    let d = abs(u) * 0.14 + abs(v) * 0.095;
    innerLineDist = abs(d - 0.082);
    accentMix = 1.0 - smoothstep(0.078, 0.086, d);
  } else if (motifId == 2) {
    // Motif 2: Gothic Triptych Mullions
    let mDist = abs(abs(pLocal.x) - 0.125);
    innerLineDist = mDist;
    accentMix = 1.0 - smoothstep(0.121, 0.129, abs(pLocal.x));
  } else {
    // Motif 3: Rose Medallion Rondel + Axial Cross Cames
    let r = length(vec2<f32>(pLocal.x * 0.72, pLocal.z));
    let ringDist = abs(r - 0.078);
    let crossDist = select(1.0, abs(pLocal.z), r > 0.078);
    innerLineDist = min(ringDist, crossDist);
    accentMix = 1.0 - smoothstep(0.074, 0.082, r);
  }

  let innerCame = (1.0 - smoothstep(0.0030, 0.0048, innerLineDist)) * onTopOrBottomSurface;
  let cameMask = max(borderCame, innerCame);

  // Subtle hand-rolled cathedral waterglass undulation (smooth gentle optical waves, never noisy streaks)
  let sp = pLocal * vec3<f32>(11.0, 14.0, 13.0) + vec3<f32>(seed * 3.7, seed * 1.9, seed * 5.3);
  let w1 = sin(sp.x * 1.3 + sin(sp.z * 1.6 + seed)) * 0.55;
  let w2 = cos(sp.z * 1.9 - sin(sp.x * 1.1 - seed * 0.7)) * 0.45;
  let ripple = (w1 + w2) * 0.5;

  let gx = cos(sp.x * 1.3 + sin(sp.z * 1.6 + seed)) * 1.3 * 11.0 * 0.25;
  let gz = -sin(sp.z * 1.9 - sin(sp.x * 1.1 - seed * 0.7)) * 1.9 * 13.0 * 0.22;

  var out: VitrailSample;
  out.cameMask = clamp(cameMask, 0.0, 1.0);
  out.accentMix = clamp(accentMix, 0.0, 1.0);
  out.rippleHeight = ripple;
  out.rippleGrad = vec3<f32>(gx, 0.0, gz) * 0.0014;
  return out;
}

/**
 * Evaluates asymmetric, sparse (< 0.25 mm) 3D brittle glass fracture planes & chordal
 * cross-cracks radiating from crackOrigin when damage > 0 (no symmetric spiderwebs or circular rings).
 */
fn evaluateInternalCracks(pLocal: vec3<f32>, crackOrigin: vec3<f32>, damage: f32, seed: f32) -> f32 {
  if (damage <= 0.01) {
    return 0.0;
  }
  let rel = pLocal - crackOrigin;
  let dist = length(rel);
  let maxRadius = 0.10 + damage * 0.46;
  if (dist > maxRadius || dist < 1e-4) {
    return 0.0;
  }

  // Dual-octave 3D noise warp gives natural brittle glass jinks rather than ruler-straight spokes
  let warpCoarse = (noise3D(pLocal * 21.0 + vec3<f32>(seed * 2.3)) - 0.5) * 0.0085;
  let warpFine = (noise3D(pLocal * 58.0 - vec3<f32>(seed * 4.1)) - 0.5) * 0.0028;
  let warp = warpCoarse + warpFine;

  let theta = atan2(rel.z + rel.y * 0.34, rel.x - rel.y * 0.24);
  let numSectors: f32 = 7.0;
  let sectorCoord = ((theta + PI) / (2.0 * PI)) * numSectors + seed * 1.73;
  let sectorId = floor(sectorCoord);
  let sectorFrac = fract(sectorCoord) - 0.5;

  let hSec = hash31(vec3<f32>(sectorId, seed * 3.17, 2.0));
  // Sparse fracture propagation: ~40% of angular sectors remain uncracked so cracks never look like a pinwheel!
  if (fract(hSec * 19.37) < 0.38) {
    let hertzOnly = (1.0 - smoothstep(0.005, 0.024, dist)) * 0.45;
    return hertzOnly * clamp(damage * 1.2, 0.0, 1.0);
  }

  let sectorReach = maxRadius * (0.35 + 0.65 * fract(hSec * 7.91));
  if (dist > sectorReach) {
    return 0.0;
  }

  let spokeOffset = (hSec - 0.5) * 0.38;
  let perpDistDm = abs((sectorFrac - spokeOffset) * (2.0 * PI / numSectors) * dist + warp);

  // Ultra-fine hairline cleavage plane (~0.20 mm physical thickness)
  var crack = 1.0 - smoothstep(0.0005, 0.0022, perpDistDm);

  // Secondary oblique bifurcating fork on high-stress sectors
  if (fract(hSec * 13.3) > 0.46 && dist > 0.05) {
    let forkSpread = (dist - 0.05) * (0.20 + 0.14 * fract(hSec * 3.7));
    let forkDistDm = abs(perpDistDm - forkSpread + warpFine * 0.8);
    crack = max(crack, (1.0 - smoothstep(0.0004, 0.0018, forkDistDm)) * 0.78);
  }

  // Planar chordal cross-fracture (straight angled cleavage plane across the sector, never a circular arc!)
  if (fract(hSec * 29.7) > 0.56 && perpDistDm < 0.022) {
    let secCenterAng = ((sectorId + 0.5 - seed * 1.73) / numSectors) * 2.0 * PI - PI;
    let chordDir = normalize(vec3<f32>(cos(secCenterAng + 0.28), 0.18, sin(secCenterAng + 0.28)));
    let projChord = dot(rel, chordDir);
    let chordStep = 0.065 + fract(hSec * 5.3) * 0.085;
    let chordDistDm = abs(projChord - chordStep + warpFine);
    crack = max(crack, (1.0 - smoothstep(0.0004, 0.0017, chordDistDm)) * 0.65);
  }

  let hertzCore = (1.0 - smoothstep(0.005, 0.025, dist)) * 0.45;
  let tipFade = 1.0 - smoothstep(sectorReach * 0.62, sectorReach, dist);
  return clamp(max(crack * tipFade, hertzCore) * clamp(damage * 1.25, 0.0, 1.0), 0.0, 1.0);
}

/**
 * Procedural sunlit Gothic Cathedral Studio sky & rose-window environment for realistic Fresnel & refractive rays.
 */
fn sampleCathedralEnvironment(rayDir: vec3<f32>, sunDir: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>) -> vec3<f32> {
  let d = normalize(rayDir);
  let up = clamp(d.y, -1.0, 1.0);

  // Warm limestone gothic vault gradient
  let floorBounce = vec3<f32>(0.11, 0.08, 0.06);
  let wallStone = vec3<f32>(0.16, 0.145, 0.135) + skyTint * 0.16;
  let vaultZenith = skyTint * 0.48 + vec3<f32>(0.08, 0.09, 0.12);

  var env = mix(wallStone, vaultZenith, smoothstep(0.0, 0.75, max(0.0, up)));
  env = mix(floorBounce, env, smoothstep(-0.35, 0.06, up));

  // High Gothic Lancet & Rose Window bright luminous panes in the direction of the sun!
  let sunCos = dot(d, sunDir);
  let sunGlow = pow(max(0.0, sunCos), 12.0) * 0.55;
  let sunCore = pow(max(0.0, sunCos), 220.0) * 4.5;

  // Subtle cathedral lancet window mullions in reflection so glass reflections have authentic architectural bars!
  let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), sunDir));
  let winUp = cross(sunDir, right);
  let winU = dot(d, right);
  let winV = dot(d, winUp);
  let inWindow = smoothstep(0.32, 0.22, abs(winU)) * smoothstep(0.42, 0.28, abs(winV)) * step(0.3, sunCos);
  let mullionBar = smoothstep(0.008, 0.015, abs(fract(winU * 7.0 + 0.5) - 0.5)) *
                   smoothstep(0.008, 0.015, abs(fract(winV * 5.0 + 0.5) - 0.5));
  let windowLight = inWindow * mullionBar * sunColor * 0.65;

  // Secondary cool clerestory fill window on opposite side for crisp rim reflections on beveled glass
  let fillDir = normalize(vec3<f32>(-sunDir.x, 0.45, -sunDir.z));
  let fillGlow = pow(max(0.0, dot(d, fillDir)), 9.0) * skyTint * 0.85;

  return env + sunColor * (sunGlow + sunCore * mullionBar) + windowLight + fillGlow;
}

/**
 * Evaluates procedural quarter-sawn dark walnut/oak tabletop + brass regulation Jenga center square inlay!
 */
fn evaluateOakTableAlbedo(xz: vec2<f32>) -> vec3<f32> {
  let r = length(xz);
  // Subtle wood planks & medullary oak grain along X
  let plankId = floor((xz.y + 4.5) * 1.45);
  let plankSeam = smoothstep(0.0, 0.014, abs(fract((xz.y + 4.5) * 1.45) - 0.5) * 2.0);
  let grainCoord = vec3<f32>(xz.x * 2.8 + plankId * 7.3, plankId * 3.1, xz.y * 26.0);
  let ring = sin(xz.y * 34.0 + noise3D(grainCoord) * 5.5 + plankId * 2.1) * 0.5 + 0.5;
  let finePore = noise3D(vec3<f32>(xz.x * 45.0, 1.0, xz.y * 140.0));

  let darkWalnut = vec3<f32>(0.105, 0.068, 0.044);
  let warmOak = vec3<f32>(0.205, 0.132, 0.082);
  var wood = mix(darkWalnut, warmOak, ring * 0.68 + finePore * 0.22);
  wood *= mix(0.76, 1.0, plankSeam);

  // Subtle brushed brass inlay framing the regulation 7.5 cm x 7.5 cm (0.75 x 0.75 dm) tower base zone
  let boxDist = max(abs(xz.x), abs(xz.y)) - 0.395;
  let brassSquare = 1.0 - smoothstep(0.004, 0.009, abs(boxDist));
  // Outer decorative compass ring on table rim
  let rimRing = 1.0 - smoothstep(0.008, 0.016, abs(r - 3.95));
  let brassMask = max(brassSquare * 0.75, rimRing * 0.55);
  let brassColor = vec3<f32>(0.62, 0.48, 0.24);

  return mix(wood, brassColor, brassMask);
}
`;

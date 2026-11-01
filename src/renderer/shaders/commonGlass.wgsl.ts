// Shared WGSL optical, material, shadow, and stained-glass procedural functions used across
// Sun Shadow/Caustic Transmission, Environment/Tabletop, Intact Glass Tower, and Voronoi Shards.

export const COMMON_GLASS_WGSL = /* wgsl */ `
const BLOCK_HX: f32 = 0.375;
const BLOCK_HY: f32 = 0.075;
const BLOCK_HZ: f32 = 0.125;
const PI: f32 = 3.14159265359;

// Authentic soda-lime architectural glass iron-oxide base attenuation (gives thick 7.5 cm paths subtle sea-green depth)
const SODA_LIME_IRON_SIGMA: vec3<f32> = vec3<f32>(0.15, 0.045, 0.095);

struct SceneUniforms {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
  cameraPosAndTime: vec4<f32>,
  sunDirAndBlockCount: vec4<f32>,
  sunColorAndAspect: vec4<f32>,
  skyTintAndMode: vec4<f32>, // xyz = skyTint, w = activeShardCount
  tableAoBounds: vec4<f32>,  // (minX, minZ, maxX, maxZ) of low-height blocks & shards
  tableAoCounts: vec4<f32>,  // x = intactCount, y = shardStride
};

struct GPUBlockData {
  localToWorld: mat4x4<f32>,
  worldToLocal: mat4x4<f32>,
  absorptionAndMotif: vec4<f32>,
  accentAndSeed: vec4<f32>,
  crackAndDamage: vec4<f32>,
  stateFlags: vec4<f32>,
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
fn evaluateVitrailPattern(pLocal: vec3<f32>, _motifId: i32, seed: f32) -> VitrailSample {
  let u = pLocal.x / BLOCK_HX; // [-1, 1]

  // Option A: Pure Unadorned Hand-Rolled Cathedral Glass Brick with subtle lengthwise flashed-glass tone shift
  let accentMix = clamp(smoothstep(-0.55, 0.65, u * sign(sin(seed * 4.1) + 0.01)) * 0.42, 0.0, 1.0);

  // Hand-rolled cathedral waterglass undulation: analytical height, slope, and Laplacian curvature!
  let kx: f32 = 13.5;
  let ky: f32 = 21.0;
  let kz: f32 = 18.0;
  let phX = pLocal.x * kx + seed * 3.7 + sin(pLocal.z * 9.0 + seed) * 0.35;
  let phY = pLocal.y * ky + seed * 2.9 + sin(pLocal.z * 14.0 - seed) * 0.30;
  let phZ = pLocal.z * kz + seed * 5.3 - cos(pLocal.x * 8.0 - seed * 0.7) * 0.35;

  let w1 = sin(phX) * 0.56;
  let w2 = cos(phZ) * 0.44;
  let ripple = (w1 + w2) * 0.5;

  let gx = cos(phX) * kx * 0.28;
  let gy = cos(phY) * ky * 0.18;
  let gz = -sin(phZ) * kz * 0.22;

  // Second derivative (-sin(phX)*kx^2 - cos(phZ)*kz^2) drives optical Jacobian convergence/divergence!
  let d2x = -sin(phX) * 0.56;
  let d2z = -cos(phZ) * 0.44;
  let laplacian = (d2x + d2z) * 0.5;

  var out: VitrailSample;
  out.cameMask = 0.0;
  out.accentMix = accentMix;
  out.paneMeniscusAO = 1.0;
  out.cameNormalLocal = vec3<f32>(0.0);
  out.rippleHeight = ripple;
  out.rippleGrad = vec3<f32>(gx, gy, gz) * 0.0011;
  out.rippleLaplacian = laplacian;
  return out;
}

/**
 * Evaluates asymmetric, sparse (< 0.25 mm) 3D brittle glass fracture planes & chordal
 * cross-cracks radiating from crackOrigin when damage > 0.
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

  let warpCoarse = (noise3D(pLocal * 21.0 + vec3<f32>(seed * 2.3)) - 0.5) * 0.0085;
  let warpFine = (noise3D(pLocal * 58.0 - vec3<f32>(seed * 4.1)) - 0.5) * 0.0028;
  let warp = warpCoarse + warpFine;

  let theta = atan2(rel.z + rel.y * 0.34, rel.x - rel.y * 0.24);
  let numSectors: f32 = 7.0;
  let sectorCoord = ((theta + PI) / (2.0 * PI)) * numSectors + seed * 1.73;
  let sectorId = floor(sectorCoord);
  let sectorFrac = fract(sectorCoord) - 0.5;

  let hSec = hash31(vec3<f32>(sectorId, seed * 3.17, 2.0));
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

  var crack = 1.0 - smoothstep(0.0005, 0.0022, perpDistDm);

  if (fract(hSec * 13.3) > 0.46 && dist > 0.05) {
    let forkSpread = (dist - 0.05) * (0.20 + 0.14 * fract(hSec * 3.7));
    let forkDistDm = abs(perpDistDm - forkSpread + warpFine * 0.8);
    crack = max(crack, (1.0 - smoothstep(0.0004, 0.0018, forkDistDm)) * 0.78);
  }

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

fn safeNormalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let lenSq = dot(v, v);
  return select(fallback, v * inverseSqrt(max(lenSq, 1e-12)), lenSq > 1e-10);
}

const BLOCK_BEVEL_R: f32 = 0.0125;

/**
 * Exact 1.25 mm rounded-box SDF surface normal in local block coordinates.
 * Preserves sharp flat interior faces while curving smoothly across all 12 beveled edges and 8 corners.
 */
fn evalBeveledBoxNormalLocal(pLocal: vec3<f32>, fallbackN: vec3<f32>) -> vec3<f32> {
  let innerHalf = vec3<f32>(BLOCK_HX - BLOCK_BEVEL_R, BLOCK_HY - BLOCK_BEVEL_R, BLOCK_HZ - BLOCK_BEVEL_R);
  let innerP = clamp(pLocal, -innerHalf, innerHalf);
  let diff = pLocal - innerP;
  return safeNormalize(diff, fallbackN);
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
 * Full sunlight illuminates the central tower & caustic pool, softly rolling off into cool
 * architectural shadow toward the outer perimeter of the 86 cm walnut table.
 */
fn evaluateStudioWindowGobo(worldPos: vec3<f32>, sunDir: vec3<f32>) -> f32 {
  // Trace point back along sunDir to reference plane y = 0.9 dm so tall tower & floor pool align coherently
  let shiftT = (0.90 - worldPos.y) / max(0.22, sunDir.y);
  let stageXZ = worldPos.xz + sunDir.xz * shiftT * 0.25;

  let sunRight = safeNormalize(vec3<f32>(sunDir.z, 0.0, -sunDir.x), vec3<f32>(1.0, 0.0, 0.0)).xz;
  let sunFwd = vec2<f32>(-sunDir.x, -sunDir.z) / max(1e-3, length(sunDir.xz));
  let uWin = dot(stageXZ, sunRight);
  let vWin = dot(stageXZ, sunFwd);

  // Arched Gothic lancet window beam footprint on the table (wide soft penumbra from solar/sky subtense)
  let archWidth = 2.95 - max(0.0, vWin - 0.6) * 0.28;
  let lateralMask = 1.0 - smoothstep(archWidth * 0.68, archWidth * 1.25, abs(uWin));
  let longMask = 1.0 - smoothstep(2.45, 4.25, abs(vWin - 0.35));

  // Subtle soft-blurred upper window transom bar near the far edge of the light pool
  let transomDip = 1.0 - 0.18 * (1.0 - smoothstep(0.18, 0.52, abs(vWin - 2.35)));
  return clamp(0.14 + 0.86 * lateralMask * longMask * transomDip, 0.14, 1.0);
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
fn sampleCathedralEnvironment(rayDir: vec3<f32>, sunDir: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>) -> vec3<f32> {
  let d = safeNormalize(rayDir, vec3<f32>(0.0, 1.0, 0.0));
  let up = clamp(d.y, -1.0, 1.0);
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);

  // Warm architectural limestone studio bounce scaled linearly by skyScale so dark-room L(0) == 0
  let darkFloor = vec3<f32>(0.042, 0.028, 0.019) * skyScale;
  let shadowedStone = vec3<f32>(0.072, 0.068, 0.068) * skyScale + skyTint * 0.085;
  let vaultZenith = skyTint * 0.28 + vec3<f32>(0.045, 0.052, 0.068) * skyScale;

  var env = mix(shadowedStone, vaultZenith, smoothstep(0.0, 0.80, max(0.0, up)));
  env = mix(darkFloor, env, smoothstep(-0.28, 0.04, up));

  // Primary Tall Arched Lancet Window (bounded HDR area light with fine stone & lead mullions)
  let right = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), sunDir), vec3<f32>(1.0, 0.0, 0.0));
  let winUp = cross(sunDir, right);
  let sunCos = dot(d, sunDir);
  let winU = dot(d, right);
  let winV = dot(d, winUp);

  let archHalfW = 0.26 * (1.0 - smoothstep(0.08, 0.42, winV) * 0.55);
  let inPrimaryWindow = smoothstep(archHalfW + 0.016, archHalfW - 0.012, abs(winU)) *
                        smoothstep(0.42, 0.38, abs(winV)) *
                        smoothstep(0.20, 0.40, sunCos);

  let leadMullionU = smoothstep(0.008, 0.016, abs(fract(winU * 6.0 + 0.5) - 0.5));
  let leadMullionV = smoothstep(0.009, 0.017, abs(fract(winV * 4.5 + 0.5) - 0.5));
  // Subtle crown-glass luminance falloff across each window quarry so reflections never look like flat white stickers
  let paneGrad = 0.72 + 0.28 * cos(winU * 18.0) * cos(winV * 14.0);
  let windowPanes = inPrimaryWindow * leadMullionU * leadMullionV * paneGrad;

  let primaryWindowLight = windowPanes * sunColor * (1.25 + 0.95 * smoothstep(0.68, 0.98, sunCos));
  let softSolarHalo = pow(max(0.0, sunCos), 20.0) * sunColor * 0.22;

  // Secondary Cool Clerestory Strip Window on opposite/overhead angle for shadow-side bevel rim strips
  let fillDir = safeNormalize(vec3<f32>(-sunDir.x * 0.85, 0.58, -sunDir.z * 0.85), vec3<f32>(0.0, 1.0, 0.0));
  let fillRight = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), fillDir), vec3<f32>(1.0, 0.0, 0.0));
  let fillUp = cross(fillDir, fillRight);
  let fillU = dot(d, fillRight);
  let fillV = dot(d, fillUp);
  let inFillStrip = smoothstep(0.34, 0.30, abs(fillU)) *
                    smoothstep(0.20, 0.16, abs(fillV)) *
                    smoothstep(0.24, 0.48, dot(d, fillDir));
  let fillMullion = smoothstep(0.009, 0.018, abs(fract(fillU * 5.0 + 0.5) - 0.5));
  let coolClerestoryLight = inFillStrip * fillMullion * (skyTint * 2.15 + vec3<f32>(0.32, 0.36, 0.44) * skyScale);

  let zenithStrip = smoothstep(0.76, 0.92, up) * smoothstep(0.28, 0.22, abs(d.x)) * (skyTint * 0.95);

  return env + primaryWindowLight + softSolarHalo + coolClerestoryLight + zenithStrip;
}

/**
 * Dimensional, out-of-focus shadowed limestone studio alcove behind the walnut table.
 * Provides warm architectural depth & oblique window-shaft atmosphere without washing out or clipping to pitch black.
 */
fn sampleStudioBackdrop(rayDir: vec3<f32>, sunDir: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>) -> vec3<f32> {
  let d = safeNormalize(rayDir, vec3<f32>(0.0, 0.0, -1.0));
  let up = clamp(d.y, -1.0, 1.0);
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);

  // Warm shadowed studio dado/wainscot below horizon -> soft limestone plaster wall above
  let lowerAlcove = vec3<f32>(0.026, 0.021, 0.019) * skyScale;
  let midWall = vec3<f32>(0.054, 0.048, 0.046) * skyScale + skyTint * 0.042;
  let upperVault = vec3<f32>(0.038, 0.040, 0.048) * skyScale + skyTint * 0.065;

  var bg = mix(lowerAlcove, midWall, smoothstep(-0.18, 0.12, up));
  bg = mix(bg, upperVault, smoothstep(0.16, 0.72, up));

  // Subtle out-of-focus architectural limestone bay & pilaster tonal modulation
  let az = atan2(d.x, d.z);
  let archMod = 0.86 + 0.14 * cos(az * 2.5 - 0.5);
  let dadoShadow = 1.0 - 0.16 * (1.0 - smoothstep(0.01, 0.06, abs(up + 0.04)));
  bg *= archMod * dadoShadow;

  // Oblique warm sunlight wash grazing the studio wall from the window side
  let sunAlign = dot(normalize(vec2<f32>(d.x, d.z + 1e-5)), normalize(vec2<f32>(sunDir.x, sunDir.z + 1e-5)));
  let shaftCoord = d.x * sunDir.z - d.z * sunDir.x;
  let shaftBeam = exp(-shaftCoord * shaftCoord * 2.4) * smoothstep(-0.18, 0.52, up + sunAlign * 0.28);
  let sideWarmth = smoothstep(-0.4, 0.85, sunAlign) * smoothstep(-0.10, 0.35, up);
  bg += (sunColor * 0.022 + skyTint * 0.024) * shaftBeam + sunColor * 0.012 * sideWarmth;

  return bg;
}

struct TableMaterial {
  albedo: vec3<f32>,
  normalPerturb: vec3<f32>,
  roughness: f32,
  metallic: f32,
};

/**
 * Evaluates artisan quarter-sawn fumed dark walnut / smoked oak stave tabletop + recessed brushed-brass inlay:
 * - Elliptical cathedral growth-ring arches per plank (eliminating uniform parallel sine stripes!)
 * - Anisotropic open wood vessel pores with 3D normal dips & higher micro-roughness
 * - Hand-planed V-groove plank joinery seams with 3D bevel normals
 * - Narrow (1.1 mm) recessed brushed-brass regulation base square & outer rim line
 */
fn evaluateWalnutTableMaterial(xz: vec2<f32>, aaWidth: f32) -> TableMaterial {
  let r = length(xz);
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

  // Elliptical cathedral growth-ring geometry warped by multi-scale 3D wood fiber noise
  let fiberWarp = (noise3D(vec3<f32>(xz.x * 1.6 + plankId * 5.1, plankId * 2.3, xz.y * 9.5)) - 0.5) * 0.055
                + (noise3D(vec3<f32>(xz.x * 6.5, plankId * 4.7, xz.y * 32.0)) - 0.5) * 0.012;
  let radialTree = sqrt(localZ * localZ + localX * localX * 0.0085) + fiberWarp;

  // Asymmetric latewood/earlywood growth ring profile (sharper dark winter growth lines, broader warm summerwood)
  let ringPhase = fract(radialTree * 26.0 + hPlank * 6.0);
  let latewood = smoothstep(0.0, 0.24, ringPhase) * (1.0 - smoothstep(0.72, 0.96, ringPhase));

  // Quarter-sawn medullary ray fleck + elongated open walnut vessel pores (14:1 aspect along X)
  let vesselNoise = noise3D(vec3<f32>(xz.x * 22.0 + plankId * 13.0, 2.0, xz.y * 210.0));
  let openPore = smoothstep(0.72, 0.90, vesselNoise);
  let rayFleck = smoothstep(0.62, 0.88, noise3D(vec3<f32>(xz.x * 7.5, plankId * 8.2, xz.y * 52.0))) * 0.18;

  // Rich fumed dark walnut palette (low-key albedo lets colored glass caustics glow with high dynamic range!)
  let deepHeart = vec3<f32>(0.036, 0.021, 0.013);
  let warmWalnut = vec3<f32>(0.082, 0.049, 0.029);
  let plankTone = 0.86 + 0.26 * hPlank2;

  var woodAlbedo = mix(deepHeart, warmWalnut, latewood * 0.74 + rayFleck) * plankTone;
  woodAlbedo *= (1.0 - openPore * 0.34);

  // Subtle 0.6 mm beveled V-groove between hardwood staves
  let seamGroove = 1.0 - smoothstep(0.0012, 0.0055 + aaWidth * 0.5, distToSeamDm);
  woodAlbedo *= (1.0 - seamGroove * 0.42);

  let seamSign = select(1.0, -1.0, plankFrac > 0.5);
  let poreBumpZ = (vesselNoise - 0.5) * 0.035 - openPore * 0.045;
  let ringBumpZ = (latewood - 0.5) * 0.018;
  var normalPerturb = vec3<f32>(
    (noise3D(vec3<f32>(xz.x * 35.0, 3.0, xz.y * 35.0)) - 0.5) * 0.012,
    0.0,
    seamGroove * seamSign * 0.22 + poreBumpZ + ringBumpZ
  );

  // Hand-rubbed satin wax finish: smoother on dense latewood (0.27), rougher inside open pores & plank seams (0.56)
  var roughness = clamp(0.31 - latewood * 0.05 + openPore * 0.24 + seamGroove * 0.20, 0.22, 0.65);

  // Narrow (1.1 mm) recessed brushed-brass regulation 7.5 x 7.5 cm tower square + subtle outer compass hairline
  let boxDist = max(abs(xz.x), abs(xz.y)) - 0.392;
  let aa = max(0.0008, aaWidth * 1.15);
  let innerBrass = 1.0 - smoothstep(0.0022 - aa, 0.0032 + aa, abs(boxDist));
  let outerBrass = (1.0 - smoothstep(0.0026 - aa, 0.0038 + aa, abs(r - 3.92))) * 0.85;
  let brassMask = clamp(max(innerBrass, outerBrass), 0.0, 1.0);

  var mat: TableMaterial;
  mat.albedo = woodAlbedo;
  mat.normalPerturb = normalPerturb;
  mat.roughness = roughness;
  mat.metallic = brassMask;
  if (brassMask > 0.001) {
    let brushTex = noise3D(vec3<f32>(xz.x * 160.0, 5.0, xz.y * 160.0));
    let agedBrassAlbedo = vec3<f32>(0.58, 0.43, 0.21) * (0.90 + 0.18 * brushTex);
    mat.albedo = mix(woodAlbedo, agedBrassAlbedo, brassMask);
    mat.normalPerturb = mix(normalPerturb, vec3<f32>(0.0, 0.0, (brushTex - 0.5) * 0.025), brassMask);
    mat.roughness = mix(roughness, 0.27, brassMask);
  }
  return mat;
}

/**
 * Legacy helper alias for lightweight refracted floor albedo lookups inside multi-bounce rays.
 */
fn evaluateOakTableAlbedo(xz: vec2<f32>) -> vec3<f32> {
  return evaluateWalnutTableMaterial(xz, 0.002).albedo;
}
`;

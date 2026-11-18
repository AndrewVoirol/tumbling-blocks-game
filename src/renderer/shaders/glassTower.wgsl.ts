import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 3: Intact Cut-Glass & Inlaid Vitrail Tower Shader (`glassTower.wgsl.ts`)
 * - Renders 5-segment 1.25 mm diamond-polished glass bevels on all 12 edges (zero painted perimeter wireframes!)
 * - Shades 3D half-round metallic filigree inlays with convex cylindrical normals & meniscus groove AO
 * - Traces dispersed Snell-refracted rays across up to 5 stacked/tumbled blocks + underlying Voronoi shards,
 *   including glass-air-glass inter-brick Fresnel seams, Beer-Lambert + soda-lime iron edge attenuation,
 *   and real 2048x2048 PCF sun self-shadowing!
 */
export const GLASS_TOWER_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uBlocks: array<GPUBlockData>;
@group(0) @binding(2) var uCausticTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;
@group(0) @binding(4) var uSunDepthTex: texture_depth_2d;
@group(0) @binding(5) var<storage, read> uShards: array<GPUShardInstance>;
@group(0) @binding(6) var uSunOpticalDepthTex: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> uBlockCenters: array<vec4<f32>>;

struct TowerVSOut {
  @invariant @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) localPos: vec3<f32>,
  @location(3) localNormal: vec3<f32>,
  @location(4) @interpolate(flat) blockIdx: u32,
};

@vertex
fn vs_tower_depth(
  @location(0) pos: vec3<f32>,
  @location(1) _normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> @invariant @builtin(position) vec4<f32> {
  let pBlk = &uBlocks[instIdx];
  let wp = (*pBlk).localToWorld * vec4<f32>(pos, 1.0);
  return uScene.viewProj * wp;
}

@fragment
fn fs_tower_depth() -> @location(0) vec4<f32> {
  return vec4<f32>(0.0);
}

@vertex
fn vs_glass_tower(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> TowerVSOut {
  let pBlk = &uBlocks[instIdx];
  let l2w = (*pBlk).localToWorld;
  let wp = l2w * vec4<f32>(pos, 1.0);
  let wn = safeNormalize((l2w * vec4<f32>(normal, 0.0)).xyz, vec3<f32>(0.0, 1.0, 0.0));

  var out: TowerVSOut;
  out.clipPos = uScene.viewProj * wp;
  out.worldPos = wp.xyz;
  out.worldNormal = wn;
  out.localPos = pos;
  out.localNormal = normal;
  out.blockIdx = instIdx;
  return out;
}

struct BoxHit {
  hit: bool,
  tNear: f32,
  tFar: f32,
  entryNormalLocal: vec3<f32>,
  exitNormalLocal: vec3<f32>,
};

/**
 * Fast AABB slab intersection without bevel normals (used to reject 95% of non-hit candidate blocks in 6 ops).
 */
fn intersectLocalSlabRaw(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let halfExt = vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
  let invD = 1.0 / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
  let t0 = (-halfExt - ro) * invD;
  let t1 = ( halfExt - ro) * invD;
  let tSmall = min(t0, t1);
  let tBig = max(t0, t1);
  let rawNear = max(tSmall.x, max(tSmall.y, tSmall.z));
  let rawFar = min(tBig.x, min(tBig.y, tBig.z));
  return vec2<f32>(rawNear, rawFar);
}

/**
 * Bevel-aware intersection against local rounded block [-BLOCK_HX..BLOCK_HX] x [-BLOCK_HY..BLOCK_HY] x [-BLOCK_HZ..BLOCK_HZ]
 * with 1.25 mm (0.0125 dm) diamond-polished edge & corner bevels.
 * Rejects empty corner V-grooves between stacked blocks and returns exact curved bevel normals on entry & exit.
 */
fn intersectLocalBlockBox(ro: vec3<f32>, rd: vec3<f32>) -> BoxHit {
  let ch = intersectChamferedPrismInterval(ro, rd);
  var res: BoxHit;
  res.hit = ch.hit;
  res.tNear = ch.tEnter;
  res.tFar = ch.tExit;
  res.entryNormalLocal = ch.enterNormalLocal;
  res.exitNormalLocal = ch.exitNormalLocal;
  return res;
}

/**
 * Evaluates sparse, tiny trapped 3D seed air bubbles suspended inside antique cathedral glass:
 * Uses exact ray-sphere impact parameter u = b / R and inverse Maxwell-Fresnel (n_glass = 1.522 -> n_air = 1.0)
 * so the outer annulus u in [0.657, 1.0] (theta_i > 41.07 deg) exhibits a physical hollow-sphere TIR bright ring!
 */
fn evaluateSeedBubbles(roLocal: vec3<f32>, rdLocal: vec3<f32>, segLen: f32, seed: f32) -> f32 {
  var glint: f32 = 0.0;
  let rayScale: f32 = 36.0;
  for (var s: i32 = 0; s < 3; s = s + 1) {
    let t = segLen * (0.22 + f32(s) * 0.28);
    let p = (roLocal + rdLocal * t) * rayScale + vec3<f32>(seed * 5.1, seed * 2.7, seed * 8.3);
    let cell = floor(p);
    let f = fract(p) - vec3<f32>(0.5);
    let h = hash31(cell);
    if (h > 0.90) {
      let offset = vec3<f32>(fract(h * 13.7) - 0.5, fract(h * 29.3) - 0.5, fract(h * 47.1) - 0.5) * 0.45;
      let toCenter = offset - f;
      let projAlong = dot(toCenter, rdLocal);
      let perpVec = toCenter - rdLocal * projAlong;
      let bImpact = length(perpVec);
      let bubbleRadius = 0.052 + fract(h * 91.7) * 0.056;
      if (bImpact < bubbleRadius && abs(projAlong) < bubbleRadius * 1.45) {
        let u = clamp(bImpact / bubbleRadius, 0.0, 0.999);
        let cosInc = sqrt(max(0.0, 1.0 - u * u));
        // Inverse dielectric Fresnel 1.522 -> 1.0: transparent core (0.043) + bright TIR ring for u >= 0.657!
        let fInv = dielectricFresnel(cosInc, GLASS_IOR_G, 1.0);
        let edgeAA = 1.0 - smoothstep(0.86, 1.0, u);
        let axialFade = 1.0 - smoothstep(bubbleRadius * 0.75, bubbleRadius * 1.45, abs(projAlong));
        glint += fInv * edgeAA * axialFade * 0.52;
      }
    }
  }
  return clamp(glint, 0.0, 0.68);
}

fn projectTableSunUV(hitPos: vec3<f32>) -> vec2<f32> {
  let sunClip = uScene.sunViewProj * vec4<f32>(hitPos, 1.0);
  let ndc = sunClip.xyz / max(sunClip.w, 1e-6);
  return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn evalTableCausticAndLightAt(hitPos: vec3<f32>, diffuseAlbedo: vec3<f32>, L: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>, skyScale: f32, atelierMix: f32) -> vec3<f32> {
  let rXZ = length(hitPos.xz);
  let gobo = evaluateStudioWindowGobo(hitPos, L, atelierMix);
  let uv = projectTableSunUV(hitPos);
  var caustic = vec3<f32>(1.0);
  if (uv.x >= 0.002 && uv.x <= 0.998 && uv.y >= 0.002 && uv.y <= 0.998) {
    let tau = textureSampleLevel(uSunOpticalDepthTex, uLinearSampler, uv, 0.0).rgb;
    let tauPeak = max(tau.r, max(tau.g, tau.b));
    let cMask = smoothstep(0.005, 0.075, tauPeak);
    let sBase = 0.38 * exp(-tau * 0.52) + 0.17 * exp(-tau * 0.18);
    let flux = textureSampleLevel(uCausticTex, uLinearSampler, uv, 0.0).rgb;
    caustic = mix(vec3<f32>(1.0), sBase, cMask) + flux;
  }
  let nDotL = max(0.0, L.y);
  let edgeVignette = 1.0 - smoothstep(3.45, 4.30, rXZ) * mix(0.32, 0.14, atelierMix);
  let baseOcc = mix(0.45, 1.0, smoothstep(0.28, 0.65, max(abs(hitPos.x), abs(hitPos.z))));
  let ambient = evaluateTableAmbientIrradiance(skyTint, skyScale, atelierMix) * baseOcc;
  let direct = sunColor * nDotL * caustic * gobo;
  return diffuseAlbedo * (direct + ambient) * edgeVignette;
}

/**
 * Evaluates refracted background scene for dispersed R, G, B rays simultaneously,
 * sharing procedural stave grain + optical-depth base when rays land within 3.5 cm
 * while sampling exact dispersed caustics (uvR, uvG, uvB) & sky!
 */
fn sampleRefractedDispersedBackground(
  posR: vec3<f32>, dirR: vec3<f32>,
  posG: vec3<f32>, dirG: vec3<f32>,
  posB: vec3<f32>, dirB: vec3<f32>
) -> vec3<f32> {
  if (uScene.cameraPosAndTime.w < -0.01) {
    return vec3<f32>(-uScene.cameraPosAndTime.w);
  }

  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);
  let atelierMix = clamp(uScene.tableAoCounts.w, 0.0, 1.0);
  let nDotL = max(0.0, L.y);

  let hitTableG = dirG.y < -1e-4 && posG.y > 0.0;
  var sharedWoodAlbedo = vec3<f32>(0.0);
  var sharedShadowBase = vec3<f32>(1.0);
  var sharedGobo: f32 = 1.0;
  var sharedAmbient = vec3<f32>(0.0);
  var sharedTableGloss = vec3<f32>(0.0);
  var sharedVignette: f32 = 1.0;
  var hitPosG = vec3<f32>(0.0);
  var validTableG = false;

  if (hitTableG) {
    hitPosG = posG + dirG * (-posG.y / dirG.y);
    let r2G = dot(hitPosG.xz, hitPosG.xz);
    if (r2G <= 4.30 * 4.30) {
      let matG = evaluateWalnutTableMaterial(hitPosG.xz, 0.0035, atelierMix);
      sharedWoodAlbedo = matG.albedo * (1.0 - matG.metallic * 0.85);
      sharedGobo = evaluateStudioWindowGobo(hitPosG, L, atelierMix);
      sharedVignette = 1.0 - smoothstep(3.65, 4.30, sqrt(r2G)) * mix(0.10, 0.05, atelierMix);
      let baseOccG = mix(0.78, 1.0, smoothstep(0.22, 0.65, max(abs(hitPosG.x), abs(hitPosG.z))));
      sharedAmbient = (evaluateTableAmbientIrradiance(skyTint, skyScale, atelierMix) + skyTint * 0.16) * baseOccG;
      let uvG = projectTableSunUV(hitPosG);
      if (uvG.x >= 0.002 && uvG.x <= 0.998 && uvG.y >= 0.002 && uvG.y <= 0.998) {
        let tau = textureSampleLevel(uSunOpticalDepthTex, uLinearSampler, uvG, 0.0).rgb;
        let tauPeak = max(tau.r, max(tau.g, tau.b));
        let cMask = smoothstep(0.005, 0.075, tauPeak);
        let specShadow = integrateSpectralTau8ToRGB(evalSwatchSpectralSigma8(tau * 0.44));
        let sBase = 0.34 * specShadow + 0.14 * exp(-tau * 0.18);
        sharedShadowBase = mix(vec3<f32>(1.0), sBase, cMask);
      }
      // Physical satin walnut Fresnel sky/studio reflection seen THROUGH refracting glass bricks!
      // Essential so blue/violet/teal glass over warm wood transmits neutral sky sheen instead of brown mud!
      let cosTab = clamp(-dirG.y, 0.0, 1.0);
      let fTab = 0.042 + 0.958 * pow(1.0 - cosTab, 5.0);
      let reflTab = vec3<f32>(dirG.x, -dirG.y, dirG.z);
      let skyGloss = sampleStudioBackdrop(reflTab, L, sunColor, skyTint, atelierMix) * fTab * (1.0 - matG.roughness * 0.52) * baseOccG;
      let sunTabSpec = evalGGXSpecular(vec3<f32>(0.0, 1.0, 0.0), -dirG, L, matG.roughness) * fTab * sunColor * sharedShadowBase * sharedGobo * 0.42;
      sharedTableGloss = skyGloss + sunTabSpec;
      sharedWoodAlbedo *= (1.0 - fTab);
      validTableG = true;
    }
  }

  var outG: f32 = 0.0;
  if (validTableG) {
    let uvG = projectTableSunUV(hitPosG);
    var fluxG: f32 = 0.0;
    if (uvG.x >= 0.002 && uvG.x <= 0.998 && uvG.y >= 0.002 && uvG.y <= 0.998) {
      fluxG = textureSampleLevel(uCausticTex, uLinearSampler, uvG, 0.0).g;
    }
    outG = (sharedWoodAlbedo.g * (sunColor.g * nDotL * (sharedShadowBase.g + fluxG) * sharedGobo + sharedAmbient.g) + sharedTableGloss.g) * sharedVignette;
  } else {
    outG = (sampleStudioBackdrop(dirG, L, sunColor, skyTint, atelierMix) + pow(max(0.0, dot(dirG, L)), 5.0) * sunColor * 0.38).g;
  }

  var outR: f32 = 0.0;
  if (dirR.y < -1e-4 && posR.y > 0.0) {
    let hitPosR = posR + dirR * (-posR.y / dirR.y);
    if (dot(hitPosR.xz, hitPosR.xz) <= 4.30 * 4.30) {
      if (validTableG && dot(hitPosR.xz - hitPosG.xz, hitPosR.xz - hitPosG.xz) <= 0.09) {
        let uvR = projectTableSunUV(hitPosR);
        var fluxR: f32 = 0.0;
        if (uvR.x >= 0.002 && uvR.x <= 0.998 && uvR.y >= 0.002 && uvR.y <= 0.998) {
          fluxR = textureSampleLevel(uCausticTex, uLinearSampler, uvR, 0.0).r;
        }
        outR = (sharedWoodAlbedo.r * (sunColor.r * nDotL * (sharedShadowBase.r + fluxR) * sharedGobo + sharedAmbient.r) + sharedTableGloss.r) * sharedVignette;
      } else {
        let matR = evaluateWalnutTableMaterial(hitPosR.xz, 0.0035, atelierMix);
        let albR = matR.albedo * (1.0 - matR.metallic * 0.85);
        outR = evalTableCausticAndLightAt(hitPosR, albR, L, sunColor, skyTint, skyScale, atelierMix).r + sharedTableGloss.r;
      }
    } else {
      outR = (sampleStudioBackdrop(dirR, L, sunColor, skyTint, atelierMix) + pow(max(0.0, dot(dirR, L)), 5.0) * sunColor * 0.38).r;
    }
  } else {
    outR = (sampleStudioBackdrop(dirR, L, sunColor, skyTint, atelierMix) + pow(max(0.0, dot(dirR, L)), 5.0) * sunColor * 0.38).r;
  }

  var outB: f32 = 0.0;
  if (dirB.y < -1e-4 && posB.y > 0.0) {
    let hitPosB = posB + dirB * (-posB.y / dirB.y);
    if (dot(hitPosB.xz, hitPosB.xz) <= 4.30 * 4.30) {
      if (validTableG && dot(hitPosB.xz - hitPosG.xz, hitPosB.xz - hitPosG.xz) <= 0.09) {
        let uvB = projectTableSunUV(hitPosB);
        var fluxB: f32 = 0.0;
        if (uvB.x >= 0.002 && uvB.x <= 0.998 && uvB.y >= 0.002 && uvB.y <= 0.998) {
          fluxB = textureSampleLevel(uCausticTex, uLinearSampler, uvB, 0.0).b;
        }
        outB = (sharedWoodAlbedo.b * (sunColor.b * nDotL * (sharedShadowBase.b + fluxB) * sharedGobo + sharedAmbient.b) + sharedTableGloss.b) * sharedVignette;
      } else {
        let matB = evaluateWalnutTableMaterial(hitPosB.xz, 0.0035, atelierMix);
        let albB = matB.albedo * (1.0 - matB.metallic * 0.85);
        outB = evalTableCausticAndLightAt(hitPosB, albB, L, sunColor, skyTint, skyScale, atelierMix).b + sharedTableGloss.b;
      }
    } else {
      outB = (sampleStudioBackdrop(dirB, L, sunColor, skyTint, atelierMix) + pow(max(0.0, dot(dirB, L)), 5.0) * sunColor * 0.38).b;
    }
  } else {
    outB = (sampleStudioBackdrop(dirB, L, sunColor, skyTint, atelierMix) + pow(max(0.0, dot(dirB, L)), 5.0) * sunColor * 0.38).b;
  }

  return vec3<f32>(outR, outG, outB);
}

struct ExitBoxHit {
  tFar: f32,
  exitNormalLocal: vec3<f32>,
};

/**
 * Fast bevel-aware exit distance & curved exit normal when ray origin ro is already inside the block.
 */
fn intersectLocalExitBox(ro: vec3<f32>, rd: vec3<f32>) -> ExitBoxHit {
  let ch = intersectChamferedPrismInterval(ro, rd);
  var out: ExitBoxHit;
  out.tFar = max(0.001, ch.tExit);
  out.exitNormalLocal = ch.exitNormalLocal;
  return out;
}

fn mulBlockW2LPoint(idx: u32, p: vec3<f32>) -> vec3<f32> {
  let pB = &uBlocks[idx];
  return (*pB).worldToLocal[0].xyz * p.x + (*pB).worldToLocal[1].xyz * p.y + (*pB).worldToLocal[2].xyz * p.z + (*pB).worldToLocal[3].xyz;
}

fn mulBlockW2LDir(idx: u32, v: vec3<f32>) -> vec3<f32> {
  let pB = &uBlocks[idx];
  return (*pB).worldToLocal[0].xyz * v.x + (*pB).worldToLocal[1].xyz * v.y + (*pB).worldToLocal[2].xyz * v.z;
}

fn mulBlockL2WPoint(idx: u32, p: vec3<f32>) -> vec3<f32> {
  let pB = &uBlocks[idx];
  return (*pB).localToWorld[0].xyz * p.x + (*pB).localToWorld[1].xyz * p.y + (*pB).localToWorld[2].xyz * p.z + (*pB).localToWorld[3].xyz;
}

fn mulBlockL2WDir(idx: u32, v: vec3<f32>) -> vec3<f32> {
  let pB = &uBlocks[idx];
  return (*pB).localToWorld[0].xyz * v.x + (*pB).localToWorld[1].xyz * v.y + (*pB).localToWorld[2].xyz * v.z;
}

/**
 * Traces a chromatic-dispersed dielectric ray through the front glass block (including 1.25 mm bevels & TIR)
 * AND continues across up to 2 secondary blocks (3 total blocks = full 3-block Jenga width) + Voronoi shards
 * using L1-cached uniform Y-interval binary search & tight AABB culling.
 */
fn traceTowerGlassRayDispersed(
  startWorldPos: vec3<f32>,
  incidentDir: vec3<f32>,
  frontWorldNormal: vec3<f32>,
  primaryIdx: u32,
  vitrailFront: VitrailSample,
  sunTransRGB: vec3<f32>
) -> vec3<f32> {
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;
  let isRefCard = uScene.cameraPosAndTime.w < -0.01;

  let pPrim = &uBlocks[primaryIdx];
  let absMotif0 = (*pPrim).absorptionAndMotif;
  let motif0 = i32(round(absMotif0.w));
  let accSeed0 = (*pPrim).accentAndSeed;
  let crack0 = (*pPrim).crackAndDamage;
  let seed0 = accSeed0.w;
  let iorR: f32 = GLASS_IOR_R;
  let iorG: f32 = GLASS_IOR_G;
  let iorB: f32 = GLASS_IOR_B;
  let atelierMix = clamp(uScene.tableAoCounts.w, 0.0, 1.0);

  // 1. Refract Green carrier ray + R/B Abbe dispersion into primary block
  var internalDirWorld = refract(incidentDir, frontWorldNormal, 1.0 / iorG);
  if (length(internalDirWorld) < 1e-4) {
    internalDirWorld = reflect(incidentDir, frontWorldNormal);
    return sampleCathedralEnvironment(internalDirWorld, L, sunColor, skyTint, atelierMix);
  }
  internalDirWorld = safeNormalize(internalDirWorld, incidentDir);

  let roLocal = mulBlockW2LPoint(primaryIdx, startWorldPos + internalDirWorld * 0.002);
  let rdLocal = mulBlockW2LDir(primaryIdx, internalDirWorld);

  let box0 = intersectLocalExitBox(roLocal, rdLocal);
  var pathLen0 = box0.tFar;
  let exitPosLocal = roLocal + rdLocal * pathLen0;
  let exitNormalWorld = mulBlockL2WDir(primaryIdx, box0.exitNormalLocal);

  let vitrailExit = evaluateVitrailPattern(exitPosLocal, motif0, seed0);
  let exit0RippleWorld = mulBlockL2WDir(primaryIdx, vitrailExit.rippleGrad) * 1.35;
  let exitPerturbedN = safeNormalize(exitNormalWorld + exit0RippleWorld, exitNormalWorld);

  let sigma0 = mix(
    absMotif0.xyz,
    accSeed0.xyz,
    0.5 * (vitrailFront.accentMix + vitrailExit.accentMix)
  ) + SODA_LIME_IRON_SIGMA;

  var worldRayPos = mulBlockL2WPoint(primaryIdx, exitPosLocal);
  // Coupled horizontal course-to-course transmission vs outer vertical wall TIR:
  // On horizontal +/-Y bearing faces (isHorizExit0 == 1.0), coupled exit refraction (iorExit = 1.055) eliminates the
  // 41.07-deg Total Internal Reflection periscope cutoff so you can ALWAYS see through horizontal bricks into the
  // supporting bricks below at every camera angle & zoom, while vertical +/-X / +/-Z walls retain full 3-bounce TIR waveguiding!
  let isHorizExit0 = step(0.65, abs(box0.exitNormalLocal.y));
  let effExitIor0 = select(iorG, select(1.055, iorG, isRefCard), isHorizExit0 > 0.5);
  let directOut0 = refract(internalDirWorld, -exitPerturbedN, effExitIor0);
  let hasDirect0 = length(directOut0) >= 1e-4;
  var worldRayDir = select(internalDirWorld, safeNormalize(directOut0, internalDirWorld), hasDirect0);
  let internalCosExit = clamp(abs(dot(internalDirWorld, exitPerturbedN)), 0.0, 1.0);
  let exactExitF0 = select(
    dielectricFresnel(internalCosExit, iorG, 1.0),
    select(clamp(dielectricFresnel(internalCosExit, 1.18, 1.0) * 0.42, 0.03, 0.24), dielectricFresnel(internalCosExit, iorG, 1.0), isRefCard),
    isHorizExit0 > 0.5
  );
  var exitTransmission0: f32 = 1.0 - exactExitF0;

  // Multi-bounce lossless Total Internal Reflection (3-step rectangular prism waveguide on vertical +/-X / +/-Z walls):
  if (!isRefCard && isHorizExit0 < 0.5 && (exactExitF0 > 0.14 || !hasDirect0)) {
    var wgPosL = exitPosLocal;
    var wgDirL = rdLocal;
    var wgExitNL = safeNormalize(mulBlockW2LDir(primaryIdx, exitPerturbedN), box0.exitNormalLocal);
    var wgPath: f32 = 0.0;
    var wgExited = false;
    var wgOutWorld = worldRayDir;
    var wgBounceTrans: f32 = 0.88;

    for (var tb: u32 = 0u; tb < 3u; tb = tb + 1u) {
      wgDirL = reflect(wgDirL, -wgExitNL);
      let bHit = intersectLocalExitBox(wgPosL + wgDirL * 0.001, wgDirL);
      wgPath += bHit.tFar;
      wgPosL = wgPosL + wgDirL * bHit.tFar;
      wgExitNL = bHit.exitNormalLocal;
      let candWorldD = mulBlockL2WDir(primaryIdx, wgDirL);
      let candWorldN = mulBlockL2WDir(primaryIdx, wgExitNL);
      let isHorizWg = step(0.65, abs(wgExitNL.y));
      let tryOut = refract(candWorldD, -candWorldN, select(iorG, 1.055, isHorizWg > 0.5));
      if (length(tryOut) >= 1e-4) {
        let cosEx = clamp(abs(dot(candWorldD, candWorldN)), 0.0, 1.0);
        wgBounceTrans = max(0.74, 1.0 - dielectricFresnel(cosEx, select(iorG, 1.18, isHorizWg > 0.5), 1.0));
        wgOutWorld = safeNormalize(tryOut, candWorldD);
        wgExited = true;
        break;
      }
    }
    if (!wgExited) {
      let candWorldD = mulBlockL2WDir(primaryIdx, wgDirL);
      let candWorldN = mulBlockL2WDir(primaryIdx, wgExitNL);
      wgOutWorld = safeNormalize(reflect(candWorldD, -candWorldN), candWorldD);
    }
    exitTransmission0 = (1.0 - exactExitF0) + exactExitF0 * wgBounceTrans;

    // Never mix worldRayPos halfway across two distinct faces (which teleports rays past supporting blocks!);
    // snap ray trajectory cleanly only on true TIR (!hasDirect0), using exact energy weighting below theta_c.
    if (!hasDirect0) {
      pathLen0 += wgPath * 0.72;
      worldRayPos = mulBlockL2WPoint(primaryIdx, wgPosL);
      worldRayDir = wgOutWorld;
    }
  }
  var rayCoupledHoriz: f32 = select(0.0, 1.0, !isRefCard && isHorizExit0 > 0.5 && hasDirect0);
  let internalMirrorContrib = vec3<f32>(0.0);
  worldRayDir = safeNormalize(worldRayDir, incidentDir);

  let inR0 = safeNormalize(refract(incidentDir, frontWorldNormal, 1.0 / iorR), internalDirWorld);
  let inB0 = safeNormalize(refract(incidentDir, frontWorldNormal, 1.0 / iorB), internalDirWorld);
  var dirR = refract(inR0, -exitPerturbedN, iorR);
  dirR = select(worldRayDir, safeNormalize(dirR, worldRayDir), length(dirR) >= 1e-4 && hasDirect0);
  var dirB = refract(inB0, -exitPerturbedN, iorB);
  dirB = select(worldRayDir, safeNormalize(dirB, worldRayDir), length(dirB) >= 1e-4 && hasDirect0);
  var posR = worldRayPos + (inR0 - internalDirWorld) * box0.tFar;
  var posB = worldRayPos + (inB0 - internalDirWorld) * box0.tFar;

  let throughput0 = exp(-sigma0 * pathLen0);
  let specSigma0 = evalSwatchSpectralSigma8(sigma0);
  var specTau: SpectralTau8;
  specTau.lo = specSigma0.lo * pathLen0;
  specTau.hi = specSigma0.hi * pathLen0;

  let transmittedSun = sunColor * sunTransRGB;
  let lightEnergy = transmittedSun * 0.42 + skyTint * 0.20;
  let scatterGate = select(1.0, 0.0, isRefCard);

  // Subtle internal prism reflection sparkle restricted to steep bevel/TIR angles (> 45 deg, zero flat-face glare!)
  var prismExitSparkle = vec3<f32>(0.0);
  if (!isRefCard && exactExitF0 > 0.09) {
    let intReflWorld = reflect(internalDirWorld, -exitPerturbedN);
    let roomSpark0 = sampleCathedralEnvironment(intReflWorld, L, transmittedSun * 0.75, skyTint, atelierMix);
    prismExitSparkle = roomSpark0 * integrateSpectralTau8ToRGB(specTau) * smoothstep(0.09, 0.65, exactExitF0) * 0.24;
  }

  let bubbleSparkle = select(evaluateSeedBubbles(roLocal, rdLocal, box0.tFar, seed0), 0.0, isRefCard);
  var crackMask: f32 = 0.0;
  var crackInternalRGB = vec3<f32>(0.0);
  if (crack0.w > 0.01) {
    let midLocal = roLocal + rdLocal * (box0.tFar * 0.38);
    let farLocal = roLocal + rdLocal * (box0.tFar * 0.74);
    let cSample0 = evaluateInternalCrackField(roLocal, crack0.xyz, crack0.w, seed0);
    let cSample1 = evaluateInternalCrackField(midLocal, crack0.xyz, crack0.w, seed0);
    let cSample2 = evaluateInternalCrackField(farLocal, crack0.xyz, crack0.w, seed0);
    var bestCrack = cSample0;
    if (cSample1.mask > bestCrack.mask) {
      bestCrack = cSample1;
    }
    if (cSample2.mask > bestCrack.mask) {
      bestCrack = cSample2;
    }
    crackMask = bestCrack.mask;
    if (crackMask > 0.005) {
      // True block-local cleavage plane normal rotated to world space (flashes silver-mirror TIR as block wobbles!)
      let crackPlaneN = safeNormalize(mulBlockL2WDir(primaryIdx, bestCrack.normalLocal), frontWorldNormal);
      let cosCrackInc = clamp(abs(dot(internalDirWorld, crackPlaneN)), 0.0, 1.0);
      let fCrackWall = dielectricFresnel(cosCrackInc, iorG, 1.0);

      // Thin-air-film Newton/Airy two-beam interference (n_glass -> n_air = 1.0 -> n_glass) below critical angle
      let sin2Air = (iorG * iorG) * max(0.0, 1.0 - cosCrackInc * cosCrackInc);
      let cosAir = sqrt(max(0.03, 1.0 - min(1.0, sin2Air)));
      let airPhase = (2.0 * PI * bestCrack.gapNm * cosAir) * vec3<f32>(1.0 / 635.0, 1.0 / 546.0, 1.0 / 455.0);
      let sinAir = sin(airPhase);
      let newtonFringes = sinAir * sinAir;

      let crackReflDir = reflect(internalDirWorld, crackPlaneN);
      let silverMirrorEnv = sampleCathedralEnvironment(crackReflDir, L, sunColor, skyTint, atelierMix);
      let H = safeNormalize(L - internalDirWorld, crackPlaneN);
      let crackSunGlint = pow(clamp(abs(dot(crackPlaneN, H)), 0.0, 1.0), 18.0);
      let filmColor = mix(newtonFringes * min(1.0, 4.0 * fCrackWall), vec3<f32>(1.0), smoothstep(0.65, 0.98, fCrackWall));

      crackInternalRGB = crackMask * (
        silverMirrorEnv * filmColor * (0.16 + 0.58 * fCrackWall) +
        lightEnergy * (0.06 + 0.85 * crackSunGlint) * mix(vec3<f32>(1.0), newtonFringes, 0.38)
      );
    }
  }
  let crackAtten0 = 1.0 - crackMask * 0.45;
  let cameExitAtten0 = (1.0 - vitrailExit.cameMask * 0.85) * vitrailFront.paneMeniscusAO * vitrailExit.paneMeniscusAO;
  var interfaceScalar: f32 = exitTransmission0 * mix(0.76, 1.0, smoothstep(0.02, 0.30, internalCosExit)) * crackAtten0 * cameExitAtten0;
  var throughput = select(integrateSpectralTau8ToRGB(specTau) * interfaceScalar, throughput0 * interfaceScalar, isRefCard);

  var internalScatter = (
    bubbleSparkle * 0.24 * lightEnergy * throughput +
    crackInternalRGB * mix(vec3<f32>(1.0), throughput0, 0.65) +
    prismExitSparkle +
    internalMirrorContrib * throughput0
  ) * scatterGate;

  // 2. Continue across up to 10 secondary blocks (11 courses total!) via global intact-AABB clip + Y-sorted uniform binary search
  let blockCount = u32(uScene.sunDirAndBlockCount.w);
  let innerHalf = vec3<f32>(BLOCK_HX - BLOCK_BEVEL_R, BLOCK_HY - BLOCK_BEVEL_R, BLOCK_HZ - BLOCK_BEVEL_R);
  let globalMinAndHalfY = uBlockCenters[96];
  let globalMax = uBlockCenters[97].xyz;
  let maxHalfY = max(0.085, globalMinAndHalfY.w) + 0.025;
  var lastHitIdx = primaryIdx;

  for (var bounce: u32 = 0u; bounce < 10u; bounce = bounce + 1u) {
    if (blockCount <= 1u || max(throughput.r, max(throughput.g, throughput.b)) < 0.010) {
      break;
    }
    if (worldRayPos.y <= 0.002 && worldRayDir.y <= 0.0) {
      break;
    }

    // Clip secondary ray against exact global intact-block world AABB before any loop!
    let invGlobalD = 1.0 / select(worldRayDir, vec3<f32>(1e-6), abs(worldRayDir) < vec3<f32>(1e-6));
    let gt0 = (globalMinAndHalfY.xyz - worldRayPos) * invGlobalD;
    let gt1 = (globalMax - worldRayPos) * invGlobalD;
    let gtNear = max(-0.025, max(min(gt0.x, gt1.x), max(min(gt0.y, gt1.y), min(gt0.z, gt1.z))));
    let gtFar = min(max(gt0.x, gt1.x), min(max(gt0.y, gt1.y), max(gt0.z, gt1.z)));
    if (gtFar <= max(0.0, gtNear)) {
      break;
    }

    let yEnter = worldRayPos.y + worldRayDir.y * max(0.0, gtNear);
    let yExit0 = worldRayPos.y + worldRayDir.y * gtFar;
    let rayMinY = min(worldRayPos.y, min(yEnter, yExit0)) - maxHalfY;
    var rayMaxY = max(worldRayPos.y, max(yEnter, yExit0)) + maxHalfY;

    // 7-step L1-cached uniform binary search for first block with center.y >= rayMinY
    var lo: u32 = 0u;
    var hi: u32 = blockCount;
    for (var bs: u32 = 0u; bs < 7u; bs = bs + 1u) {
      let mid = (lo + hi) >> 1u;
      if (mid < hi && uBlockCenters[mid].y < rayMinY) {
        lo = mid + 1u;
      } else {
        hi = mid;
      }
    }

    var nearestT: f32 = gtFar + 0.01;
    var nearestIdx: u32 = 999u;

    for (var j: u32 = lo; j < blockCount; j = j + 1u) {
      let bc = uBlockCenters[j];
      if (bc.y > rayMaxY) {
        break;
      }
      if (j == lastHitIdx) {
        continue;
      }
      let candCenter = bc.xyz;
      let toCand = candCenter - worldRayPos;
      let projT = dot(toCand, worldRayDir);
      if (projT < -0.48 || projT > nearestT + 0.48) {
        continue;
      }
      let tClamped = clamp(projT, 0.0, nearestT);
      let diff = (worldRayPos + worldRayDir * tClamped) - candCenter;
      let ty0 = (bc.y - bc.w - worldRayPos.y) * invGlobalD.y;
      let ty1 = (bc.y + bc.w - worldRayPos.y) * invGlobalD.y;
      if (dot(diff, diff) > 0.245 || max(ty0, ty1) < -0.045 || min(ty0, ty1) > nearestT) {
        continue;
      }

      let cRo = mulBlockW2LPoint(j, worldRayPos);
      let cRd = mulBlockW2LDir(j, worldRayDir);
      let slab = intersectLocalSlabRaw(cRo, cRd);
      if (slab.y >= max(slab.x, 0.002) && max(0.0, slab.x) < nearestT && slab.x > -0.040) {
        let tEnterCand = max(0.0, slab.x);
        let pMid = cRo + cRd * (0.5 * (tEnterCand + slab.y));
        let qMid = max(abs(pMid) - innerHalf, vec3<f32>(0.0));
        if (dot(qMid, qMid) <= (BLOCK_BEVEL_R + 0.0022) * (BLOCK_BEVEL_R + 0.0022)) {
          nearestT = tEnterCand;
          nearestIdx = j;
          let yHit = worldRayPos.y + worldRayDir.y * nearestT;
          rayMaxY = min(rayMaxY, max(worldRayPos.y, max(yEnter, yHit)) + maxHalfY);
        }
      }
    }

    if (nearestIdx == 999u) {
      break;
    }

    let pHit = &uBlocks[nearestIdx];
    let hitAbsMotif = (*pHit).absorptionAndMotif;
    let hitMotif = i32(round(hitAbsMotif.w));
    let hitAccSeed = (*pHit).accentAndSeed;
    let hitCrack = (*pHit).crackAndDamage;
    let hitSeed = hitAccSeed.w;

    let cRo = mulBlockW2LPoint(nearestIdx, worldRayPos);
    let cRd = mulBlockW2LDir(nearestIdx, worldRayDir);
    let nearestBox = intersectLocalBlockBox(cRo, cRd);
    if (!nearestBox.hit) {
      lastHitIdx = nearestIdx;
      continue;
    }
    nearestT = max(0.0, nearestBox.tNear);
    let localEntryPos = cRo + cRd * nearestT;
    let vHit = evaluateVitrailPattern(localEntryPos, hitMotif, hitSeed);

    let entryNWorld = mulBlockL2WDir(nearestIdx, nearestBox.entryNormalLocal);
    let cosEntry = clamp(abs(dot(worldRayDir, entryNWorld)), 0.0, 1.0);
    let seamFresnel = dielectricFresnel(cosEntry, 1.0, iorG);
    let isTightContact = 1.0 - smoothstep(0.001, 0.022, nearestT);
    let seamReflDir = reflect(worldRayDir, entryNWorld);
    let polariscope = uBlockCenters[97].w;
    let polariscopeMod = 1.0 - 0.75 * polariscope;
    let qEntryBevel = max(abs(localEntryPos) - innerHalf, vec3<f32>(0.0));
    let entryBevelMask = smoothstep(0.0012, 0.0115, length(qEntryBevel));

    // Smooth bevel-tip grazing feather so lengthwise 1.25 mm V-grooves taper chord smoothly to 0 instead of 1-ULP dotted lines
    let pMidCheck = cRo + cRd * (0.5 * (nearestT + max(nearestT + 0.002, nearestBox.tFar)));
    let qMidCheck = max(abs(pMidCheck) - innerHalf, vec3<f32>(0.0));
    let bevelLipCoverage = select(1.0, 1.0 - smoothstep(BLOCK_BEVEL_R - 0.0016, BLOCK_BEVEL_R + 0.0020, length(qMidCheck)), !isRefCard);

    // Exact Snell parallel-slab invariance (n_exit * eta_entry == 1.000): prevents double-bending shallow rays down into dark lower courses!
    let horizCoupleIn = rayCoupledHoriz * step(0.65, abs(nearestBox.entryNormalLocal.y));
    let effEntryEtaG = mix(1.0 / iorG, 1.0 / 1.055, horizCoupleIn);
    let effEntryEtaR = mix(1.0 / iorR, 1.0 / 1.053, horizCoupleIn);
    let effEntryEtaB = mix(1.0 / iorB, 1.0 / 1.058, horizCoupleIn);

    var insideSecondary = refract(worldRayDir, entryNWorld, effEntryEtaG);
    if (length(insideSecondary) < 1e-4) {
      insideSecondary = reflect(worldRayDir, entryNWorld);
    }
    insideSecondary = safeNormalize(insideSecondary, worldRayDir);

    let insideLocal = mulBlockW2LDir(nearestIdx, insideSecondary);
    let insideBox = intersectLocalExitBox(localEntryPos + insideLocal * 0.001, insideLocal);
    var segLen = clamp(insideBox.tFar, 0.006, 0.76);
    var localExitPos = localEntryPos + insideLocal * segLen;
    let vExit2 = evaluateVitrailPattern(localExitPos, hitMotif, hitSeed);
    let exit2RippleWorld = mulBlockL2WDir(nearestIdx, vExit2.rippleGrad) * 1.10;

    let avgAccent = 0.5 * (vHit.accentMix + vExit2.accentMix);
    let sigmaHit = mix(hitAbsMotif.xyz, hitAccSeed.xyz, avgAccent) + SODA_LIME_IRON_SIGMA;

    var exitN2World = safeNormalize(
      mulBlockL2WDir(nearestIdx, insideBox.exitNormalLocal) + exit2RippleWorld,
      entryNWorld
    );
    let isHorizExit2 = step(0.65, abs(insideBox.exitNormalLocal.y));
    let effExitIor2 = select(iorG, select(1.055, iorG, isRefCard), isHorizExit2 > 0.5);
    let cosExit2 = clamp(abs(dot(insideSecondary, exitN2World)), 0.0, 1.0);
    let exactExitF2 = select(
      dielectricFresnel(cosExit2, iorG, 1.0),
      select(clamp(dielectricFresnel(cosExit2, 1.18, 1.0) * 0.38, 0.02, 0.22), dielectricFresnel(cosExit2, iorG, 1.0), isRefCard),
      isHorizExit2 > 0.5
    );
    var exitTrans2: f32 = 1.0 - exactExitF2;
    let directOut2 = refract(insideSecondary, -exitN2World, effExitIor2);
    let hasDirect2 = length(directOut2) >= 1e-4;
    var exitedDir = select(insideSecondary, safeNormalize(directOut2, insideSecondary), hasDirect2);

    // 3-step lossless internal prism waveguide on vertical +/-X / +/-Z walls of secondary blocks
    if (!isRefCard && isHorizExit2 < 0.5 && (exactExitF2 > 0.14 || !hasDirect2)) {
      var wgPos2 = localExitPos;
      var wgDir2 = insideLocal;
      var wgNL2 = safeNormalize(mulBlockW2LDir(nearestIdx, exitN2World), insideBox.exitNormalLocal);
      var wgSpan2: f32 = 0.0;
      var wgOut2 = exitedDir;
      var wgExWN2 = exitN2World;
      var wgTrans2: f32 = 0.86;
      var wgDone2 = false;
      for (var tb2: u32 = 0u; tb2 < 3u; tb2 = tb2 + 1u) {
        wgDir2 = reflect(wgDir2, -wgNL2);
        let tirBox = intersectLocalExitBox(wgPos2 + wgDir2 * 0.001, wgDir2);
        let stp = clamp(tirBox.tFar, 0.005, 0.55);
        wgSpan2 += stp;
        wgPos2 = wgPos2 + wgDir2 * stp;
        wgNL2 = tirBox.exitNormalLocal;
        let twD = mulBlockL2WDir(nearestIdx, wgDir2);
        let twN = mulBlockL2WDir(nearestIdx, wgNL2);
        let isHorizWg2 = step(0.65, abs(wgNL2.y));
        let tryO2 = refract(twD, -twN, select(iorG, 1.055, isHorizWg2 > 0.5));
        if (length(tryO2) >= 1e-4) {
          let cEx2 = clamp(abs(dot(twD, twN)), 0.0, 1.0);
          wgTrans2 = max(0.72, 1.0 - dielectricFresnel(cEx2, select(iorG, 1.18, isHorizWg2 > 0.5), 1.0));
          wgOut2 = safeNormalize(tryO2, twD);
          wgExWN2 = twN;
          wgDone2 = true;
          break;
        }
      }
      if (!wgDone2) {
        let twD = mulBlockL2WDir(nearestIdx, wgDir2);
        let twN = mulBlockL2WDir(nearestIdx, wgNL2);
        wgOut2 = safeNormalize(reflect(twD, -twN), twD);
        wgExWN2 = twN;
      }
      exitTrans2 = (1.0 - exactExitF2) + exactExitF2 * wgTrans2;

      if (!hasDirect2) {
        segLen += wgSpan2 * 0.70;
        localExitPos = wgPos2;
        exitN2World = wgExWN2;
        exitedDir = wgOut2;
      }
    }

    // Physical sunlit inter-block interface & filigree radiance (NO floor uCausticTex self-projection, NO neon bulk fog!)
    let midLocalHit = 0.5 * (localEntryPos + localExitPos);
    let midWorldP = mulBlockL2WPoint(nearestIdx, midLocalHit);
    let hitOpt = sampleSunOpticalDepth(uSunOpticalDepthTex, uScene.sunViewProj, midWorldP);
    let hitTauSelf = sigmaHit * 0.16 + vec3<f32>(0.088);
    let hitTauUp = max(vec3<f32>(0.0), hitOpt.rgb - hitTauSelf);
    let hitGobo = evaluateStudioWindowGobo(midWorldP, L, atelierMix);
    let hitSunTrans = exp(-hitTauUp * 0.65) * hitGobo * polariscopeMod;
    let hitSolarIrrad = sunColor * hitSunTrans;

    var seamEnv = skyTint * 0.10 + hitSolarIrrad * pow(max(0.0, dot(seamReflDir, L)), 8.0) * 0.65;
    if (bounce == 0u && isRefCard) {
      seamEnv = sampleCathedralEnvironment(seamReflDir, L, sunColor, skyTint, atelierMix);
    } else if (bounce == 0u) {
      // Smooth mullion-free interior window/stone reflection so 1 px internal cross-course bevels never stair-step!
      let upSeam = clamp(seamReflDir.y, -1.0, 1.0);
      let sunCosSeam = max(0.0, dot(seamReflDir, L));
      seamEnv = skyTint * mix(0.14, 0.28, smoothstep(-0.2, 0.7, upSeam))
              + hitSolarIrrad * (0.12 * smoothstep(0.15, 0.65, sunCosSeam) + 0.68 * pow(sunCosSeam, 9.0));
    }
    let seamWeight = select(
      0.24 + 0.24 * isTightContact,
      (0.04 + 0.34 * entryBevelMask + 0.05 * isTightContact) * bevelLipCoverage,
      !isRefCard
    );
    internalScatter += seamEnv * throughput * seamFresnel * seamWeight * scatterGate;

    let effSegLen = segLen * bevelLipCoverage;
    let segBeer = exp(-sigmaHit * effSegLen);
    let swatchRich = exp(-sigmaHit * max(effSegLen * 1.15, 0.16));
    let minCh = min(swatchRich.r, min(swatchRich.g, swatchRich.b));
    let pureJewelPass = max(vec3<f32>(0.0), swatchRich - vec3<f32>(minCh * 0.86));
    let qExitBevel = max(abs(localExitPos) - innerHalf, vec3<f32>(0.0));
    let exitBevelMask = smoothstep(0.0010, 0.0120, length(qExitBevel));
    let bevelSunSpark = swatchRich * hitSolarIrrad * exitBevelMask * pow(max(0.0, dot(reflect(insideSecondary, -exitN2World), L)), 10.0) * 0.38;
    // Clear optical glass: zero milky front-lit fog, preserving true backlit forward glow + internal bevel sparkle
    let hitForwardGlint = pureJewelPass * hitSolarIrrad * (0.008 + 0.24 * pow(max(0.0, dot(worldRayDir, L)), 3.8)) * (1.0 - exp(-effSegLen * 3.0)) * 0.36;
    let hitBodyGlow = (hitForwardGlint + bevelSunSpark) * bevelLipCoverage;

    let maxCame2 = max(vHit.cameMask, vExit2.cameMask) * bevelLipCoverage;
    var hitCameEmit = vec3<f32>(0.0);
    if (maxCame2 > 0.005) {
      let hitMetalF0 = select(vec3<f32>(0.36, 0.37, 0.39), vec3<f32>(0.70, 0.43, 0.25), hitMotif == 3);
      let cameNLocal = select(vExit2.cameNormalLocal, vHit.cameNormalLocal, vHit.cameMask >= vExit2.cameMask);
      let hitCameWorldN = safeNormalize(entryNWorld + mulBlockL2WDir(nearestIdx, cameNLocal), entryNWorld);
      let cameSpecHit = pow(max(0.0, dot(reflect(worldRayDir, hitCameWorldN), L)), 14.0) * 0.58;
      hitCameEmit = (hitMetalF0 * (skyTint * 0.14 + hitSolarIrrad * (0.16 + 0.28 * max(0.0, dot(hitCameWorldN, L)))) + hitSolarIrrad * hitMetalF0 * cameSpecHit) * maxCame2 * 0.80;
    }

    var crackAtten: f32 = 1.0;
    if (hitCrack.w > 0.01) {
      let cS2a = evaluateInternalCrackField(localEntryPos, hitCrack.xyz, hitCrack.w, hitSeed);
      let cS2b = evaluateInternalCrackField(midLocalHit, hitCrack.xyz, hitCrack.w, hitSeed);
      var bestC2 = cS2a;
      if (cS2b.mask > bestC2.mask) {
        bestC2 = cS2b;
      }
      crackAtten = 1.0 - bestC2.mask * 0.42 * bevelLipCoverage;
      if (bestC2.mask > 0.01) {
        let cN2World = safeNormalize(mulBlockL2WDir(nearestIdx, bestC2.normalLocal), entryNWorld);
        let fC2 = dielectricFresnel(clamp(abs(dot(insideSecondary, cN2World)), 0.0, 1.0), iorG, 1.0);
        internalScatter += bestC2.mask * (0.10 + 0.34 * fC2) * (hitSolarIrrad * 0.48 + skyTint * 0.16) * throughput * crackAtten * bevelLipCoverage * scatterGate;
      }
    }

    let effSeamF = seamFresnel * bevelLipCoverage;
    let coupledEntryTrans = mix(1.0 - effSeamF, 1.0 - effSeamF * 0.38, isTightContact * scatterGate);
    let coupledExitTrans = mix(1.0, mix(exitTrans2, max(exitTrans2, 0.90), isTightContact * scatterGate), bevelLipCoverage);
    internalScatter += throughput * coupledEntryTrans * (hitBodyGlow * (1.0 - maxCame2 * 0.80) + hitCameEmit) * scatterGate;

    // Prevent surface putty meniscus AO from compounding multiplicatively across 10 stacked courses
    let rawMeniscus2 = vHit.paneMeniscusAO * vExit2.paneMeniscusAO;
    let meniscus2 = select(rawMeniscus2, mix(1.0, rawMeniscus2, 0.34 * bevelLipCoverage), bounce > 0u && !isRefCard);
    let cameAtten2 = (1.0 - maxCame2 * 0.82) * meniscus2;
    let stepInterface = coupledEntryTrans * coupledExitTrans * crackAtten * cameAtten2;

    if (!isRefCard && exactExitF2 > 0.12) {
      let intRefl2 = reflect(insideSecondary, -exitN2World);
      let roomSpark2 = sampleCathedralEnvironment(intRefl2, L, hitSolarIrrad * 0.75, skyTint, atelierMix);
      internalScatter += throughput * coupledEntryTrans * segBeer * roomSpark2 * smoothstep(0.12, 0.70, exactExitF2) * 0.18 * bevelLipCoverage * scatterGate;
    }

    let hitSpec8 = evalSwatchSpectralSigma8(sigmaHit);
    specTau.lo += hitSpec8.lo * effSegLen;
    specTau.hi += hitSpec8.hi * effSegLen;
    interfaceScalar *= stepInterface;
    throughput = select(integrateSpectralTau8ToRGB(specTau) * interfaceScalar, throughput * segBeer * stepInterface, isRefCard);

    let prevPosG = worldRayPos;
    let prevDirG = worldRayDir;
    worldRayDir = safeNormalize(exitedDir, worldRayDir);
    worldRayPos = mulBlockL2WPoint(nearestIdx, localExitPos) + worldRayDir * 0.0025;
    let stepVecG = worldRayPos - prevPosG;

    var inR = refract(dirR, entryNWorld, effEntryEtaR);
    inR = select(safeNormalize(inR, insideSecondary), insideSecondary, length(inR) < 1e-4);
    let effExitIorR2 = select(iorR, select(1.053, iorR, isRefCard), isHorizExit2 > 0.5);
    var outR = refract(inR, -exitN2World, effExitIorR2);
    outR = select(safeNormalize(outR, worldRayDir), worldRayDir, length(outR) < 1e-4);
    posR += (dirR - prevDirG) * nearestT + stepVecG + (inR - insideSecondary) * segLen;
    dirR = outR;

    var inB = refract(dirB, entryNWorld, effEntryEtaB);
    inB = select(safeNormalize(inB, insideSecondary), insideSecondary, length(inB) < 1e-4);
    let effExitIorB2 = select(iorB, select(1.058, iorB, isRefCard), isHorizExit2 > 0.5);
    var outB = refract(inB, -exitN2World, effExitIorB2);
    outB = select(safeNormalize(outB, worldRayDir), worldRayDir, length(outB) < 1e-4);
    posB += (dirB - prevDirG) * nearestT + stepVecG + (inB - insideSecondary) * segLen;
    dirB = outB;

    rayCoupledHoriz = select(0.0, 1.0, !isRefCard && isHorizExit2 > 0.5 && hasDirect2);
    lastHitIdx = nearestIdx;
  }

  // 3. Attenuate through fractured Voronoi shards behind intact blocks (front-to-back via storage pointers)
  let shardCount = u32(uScene.skyTintAndMode.w);
  let maxShardSteps = min(shardCount, 14u);
  if (maxShardSteps > 0u && max(throughput.r, max(throughput.g, throughput.b)) > 0.03) {
    var shardHits: u32 = 0u;
    for (var step: u32 = 0u; step < maxShardSteps; step = step + 1u) {
      let s = shardCount - 1u - step;
      let pSh = &uShards[s];
      let sCenter = (*pSh).localToWorld[3].xyz;
      let toS = sCenter - worldRayPos;
      let pT = dot(toS, worldRayDir);
      if (pT > 0.0 && pT < 0.85) {
        let cPt = worldRayPos + worldRayDir * pT - sCenter;
        if (dot(cPt, cPt) < 0.045) {
          let r0 = (*pSh).localToWorld[0].xyz;
          let r1 = (*pSh).localToWorld[1].xyz;
          let r2 = (*pSh).localToWorld[2].xyz;
          let roS = vec3<f32>(dot(r0, worldRayPos - sCenter), dot(r1, worldRayPos - sCenter), dot(r2, worldRayPos - sCenter));
          let rdS = vec3<f32>(dot(r0, worldRayDir), dot(r1, worldRayDir), dot(r2, worldRayDir));
          let pCount = min(MAX_SHARD_PLANES, u32(round((*pSh).originAndPlaneCount.w)));
          var tEnter: f32 = 0.0;
          var tLeave: f32 = 0.45;
          for (var k: u32 = 0u; k < pCount; k = k + 1u) {
            let pl = (*pSh).planes[k];
            let den = dot(pl.xyz, rdS);
            let num = abs(pl.w) - dot(pl.xyz, roS);
            if (den > 1e-4) {
              tLeave = min(tLeave, num / den);
            } else if (den < -1e-4) {
              tEnter = max(tEnter, num / den);
            }
          }
          if (tLeave > tEnter && tLeave > 0.002) {
            let sLen = clamp(tLeave - max(0.0, tEnter), 0.01, 0.30);
            let sSigma = (*pSh).absorption + SODA_LIME_IRON_SIGMA;
            let shSpec8 = evalSwatchSpectralSigma8(sSigma);
            specTau.lo += shSpec8.lo * sLen;
            specTau.hi += shSpec8.hi * sLen;
            interfaceScalar *= 0.916;
            throughput = select(integrateSpectralTau8ToRGB(specTau) * interfaceScalar, throughput * exp(-sSigma * sLen) * 0.916, isRefCard);
            shardHits = shardHits + 1u;
            if (shardHits >= 2u) {
              break;
            }
          }
        }
      }
    }
  }

  let bgDispersed = sampleRefractedDispersedBackground(posR, dirR, worldRayPos, worldRayDir, posB, dirB);
  let transmittedDispersed = bgDispersed * throughput;

  // Clear cut-crystal primary pane: zero milky front-lit haze so contrasting rear blocks mix spectrally with sharp silhouettes!
  let primaryRich = sqrt(throughput0) * throughput0;
  let primMin = min(primaryRich.r, min(primaryRich.g, primaryRich.b));
  let primPureChroma = max(vec3<f32>(0.0), primaryRich - vec3<f32>(primMin * 0.84));
  let forwardAlign = pow(max(0.0, dot(incidentDir, L)), 4.0);
  let glassSunGlow = primPureChroma * transmittedSun * (0.006 + 0.24 * forwardAlign) * scatterGate;

  return transmittedDispersed + glassSunGlow + internalScatter;
}

@fragment
fn fs_glass_tower(input: TowerVSOut) -> @location(0) vec4<f32> {
  // Unconditional derivative & binding keep-alives at top of fragment shader!
  let _keepDepth = f32(textureDimensions(uSunDepthTex).x) * 0.0;
  let _keepShards = f32(arrayLength(&uShards)) * 0.0;
  let _keepOpt = f32(textureDimensions(uSunOpticalDepthTex).x) * 0.0;
  let _keepCaustic = f32(textureDimensions(uCausticTex).x) * 0.0;
  let _keepCenters = uBlockCenters[0].x * 0.0;

  let pBlk = &uBlocks[input.blockIdx];
  let motifId = i32(round((*pBlk).absorptionAndMotif.w));
  let seed = (*pBlk).accentAndSeed.w;

  let V = safeNormalize(uScene.cameraPosAndTime.xyz - input.worldPos, vec3<f32>(0.0, 0.0, 1.0));
  let incidentDir = -V;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));

  let vitrail = evaluateVitrailPattern(input.localPos, motifId, seed);

  // Crisp flat facet normals from split-normal 45-degree diamond-ground prism mesh (zero soap-bar rounding!)
  let smoothLocalN = safeNormalize(input.localNormal, vec3<f32>(0.0, 1.0, 0.0));
  let macroWorldN = safeNormalize(((*pBlk).localToWorld * vec4<f32>(smoothLocalN, 0.0)).xyz, input.worldNormal);

  // Upstream-only colored solar transmittance (replaces opaque binary self-shadow blackout!)
  let atelierMix = clamp(uScene.tableAoCounts.w, 0.0, 1.0);
  let rawSunVis = sampleSunShadowPCF(uSunDepthTex, uScene.sunViewProj, input.worldPos, macroWorldN, L, 0.022);
  let windowGobo = evaluateStudioWindowGobo(input.worldPos, L, atelierMix);
  let sunDirLocal = ((*pBlk).worldToLocal * vec4<f32>(-L, 0.0)).xyz;
  let selfChord = clamp(intersectLocalSlabRaw(input.localPos + sunDirLocal * 0.001, sunDirLocal).y, 0.015, 0.78);
  let sigmaSelf = mix((*pBlk).absorptionAndMotif.xyz, (*pBlk).accentAndSeed.xyz, vitrail.accentMix) + SODA_LIME_IRON_SIGMA;
  let tauSelf = sigmaSelf * selfChord * 0.78 + vec3<f32>(0.088);
  let optSample = sampleSunOpticalDepth(uSunOpticalDepthTex, uScene.sunViewProj, input.worldPos);
  let tauUpstream = max(vec3<f32>(0.0), optSample.rgb - tauSelf) * (1.0 - rawSunVis);

  // Zero y = 0 floor-caustic self-projection onto elevated tower blocks!
  // Only Layer-0 undersides resting directly on the tabletop (worldPos.y < 0.16 dm) receive soft wood-caustic bounce.
  let tableBounceGate = (1.0 - smoothstep(0.02, 0.16, input.worldPos.y)) * clamp(-macroWorldN.y, 0.0, 1.0);
  var causticBounce = vec3<f32>(0.0);
  if (tableBounceGate > 0.005) {
    let sunUV = projectTableSunUV(vec3<f32>(input.worldPos.x, 0.0, input.worldPos.z));
    if (sunUV.x >= 0.002 && sunUV.x <= 0.998 && sunUV.y >= 0.002 && sunUV.y <= 0.998) {
      causticBounce = textureSampleLevel(uCausticTex, uLinearSampler, sunUV, 0.0).rgb * tableBounceGate * 0.18;
    }
  }

  let polariscope = uBlockCenters[97].w;
  let polariscopeMod = 1.0 - 0.75 * polariscope;
  let rawSunTrans = mix(exp(-tauUpstream * 0.80), vec3<f32>(1.0), rawSunVis) * windowGobo;
  let sunTransRGB = (rawSunTrans + causticBounce * windowGobo) * polariscopeMod;
  let directSunVis = rawSunVis * windowGobo * polariscopeMod;

  // Smooth C1 0.68 mm fire-polished bevel mask (1.0 on flat faces, 0.0 across the rounded 0.68 mm edge fillet)
  let maxAxisN = max(abs(smoothLocalN.x), max(abs(smoothLocalN.y), abs(smoothLocalN.z)));
  let flatFaceMask = smoothstep(0.94, 0.995, maxAxisN);
  let ax = BLOCK_HX - abs(input.localPos.x);
  let ay = BLOCK_HY - abs(input.localPos.y);
  let az = BLOCK_HZ - abs(input.localPos.z);
  let dMin = min(ax, min(ay, az));
  let dMax = max(ax, max(ay, az));
  let dMid = ax + ay + az - dMin - dMax;

  let rippleWorld = ((*pBlk).localToWorld * vec4<f32>(vitrail.rippleGrad * flatFaceMask, 0.0)).xyz;
  let glassNormal = safeNormalize(macroWorldN + rippleWorld, macroWorldN);

  let refractedRGB = traceTowerGlassRayDispersed(input.worldPos, incidentDir, glassNormal, input.blockIdx, vitrail, sunTransRGB);

  // Exact Maxwell-Fresnel dielectric reflectance + outer reflections evaluated AFTER multi-bounce raytracer
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;
  let cosTheta = clamp(dot(glassNormal, V), 0.0, 1.0);
  let fresnel = dielectricFresnel(cosTheta, 1.0, GLASS_IOR_G);

  let reflDir = reflect(incidentDir, glassNormal);
  let reflSunColor = sunColor * mix(0.20 + 0.80 * sunTransRGB, vec3<f32>(1.0), directSunVis);
  var envReflection = sampleCathedralEnvironment(reflDir, L, reflSunColor, skyTint, atelierMix);
  // Soft-compress reflection luminance on flat primary faces so grazing top/front panes never white-out (#FFFFFF)
  // over the blocks below, while keeping full crisp HDR specular hairlines along the 0.68 mm polished bevel rim!
  let flatGlareCompress = envReflection / (vec3<f32>(1.0) + envReflection * 0.62);
  envReflection = mix(envReflection, flatGlareCompress, flatFaceMask);

  // Occlude overhead studio skylight inside buried horizontal course-to-course V-groove chamfers:
  // Only the top-most uncovered course (y near uBlockCenters[97].y) sees unobstructed zenith sky; interior
  // horizontal chamfers sit directly against the neighboring glass course, eliminating stark white fluorescent seam bars!
  let towerTopY = uBlockCenters[97].y;
  let isHorizBevel = (1.0 - flatFaceMask) * smoothstep(0.35, 0.68, abs(macroWorldN.y));
  let isCoveredCourse = 1.0 - smoothstep(towerTopY - 0.14, towerTopY - 0.02, input.worldPos.y);
  let seamSkyOcc = isHorizBevel * isCoveredCourse * smoothstep(-0.05, 0.65, abs(reflDir.y));
  let bodySwatchTint = exp(-sigmaSelf * 0.14);
  envReflection *= mix(vec3<f32>(1.0), mix(vec3<f32>(0.18), bodySwatchTint * 0.42, 0.65), seamSkyOcc * (1.0 - directSunVis * 0.65));

  let glassRoughness = mix(0.032, 0.055, flatFaceMask);
  let sunMicroSpec = min(
    evalGGXSpecular(glassNormal, V, L, glassRoughness) * fresnel,
    1.15
  ) * sunColor * mix(sunTransRGB * 0.35, vec3<f32>(1.0), directSunVis) * (1.0 - seamSkyOcc * 0.72) * 0.65;

  // Preserve exact dielectric Fresnel up to 70 deg incidence (fresnel <= 0.32), and body-tint extreme grazing reflections
  let untintedF = min(fresnel, 0.32);
  let grazingExcess = max(0.0, fresnel - 0.32);
  let reflContrib = envReflection * (vec3<f32>(untintedF) + bodySwatchTint * (grazingExcess * 0.68));

  var finalRGB = refractedRGB * (1.0 - fresnel) + reflContrib + sunMicroSpec + vec3<f32>(_keepDepth + _keepShards + _keepOpt + _keepCaustic + _keepCenters);

  // Recessed 60/40 tin-lead / copper-foil came conductor shading + patinated flux seam darkening when motifId > 0
  if (vitrail.cameMask > 0.001) {
    let cameWorldN = safeNormalize(macroWorldN + ((*pBlk).localToWorld * vec4<f32>(vitrail.cameNormalLocal, 0.0)).xyz, glassNormal);
    let metalF0 = select(vec3<f32>(0.34, 0.35, 0.37), vec3<f32>(0.68, 0.41, 0.24), motifId == 3);
    let metalRough = select(0.28, 0.23, motifId == 3);
    let cosCame = clamp(dot(cameWorldN, V), 0.0, 1.0);
    let fMetal = metalF0 + (vec3<f32>(1.0) - metalF0) * pow(1.0 - cosCame, 5.0);
    let cameSpec = evalGGXSpecular(cameWorldN, V, L, metalRough) * fMetal * sunColor * mix(sunTransRGB * 0.45, vec3<f32>(1.0), directSunVis) * 0.95;
    let cameEnv = sampleCathedralEnvironment(reflect(incidentDir, cameWorldN), L, reflSunColor, skyTint, atelierMix) * fMetal * 0.40;
    let cameLit = metalF0 * (skyTint * 0.08 + sunColor * sunTransRGB * max(0.0, dot(cameWorldN, L)) * 0.18) + cameSpec + cameEnv;
    finalRGB = mix(finalRGB, cameLit, vitrail.cameMask);
  }

  let flags = (*pBlk).stateFlags;
  let hover = flags.x;
  let grabbed = flags.y;
  let isLegal = flags.z;
  let resistance = flags.w;
  let pe = clamp((*pBlk).photoelastic, vec4<f32>(0.0), vec4<f32>(1.0));
  let blkCrack = (*pBlk).crackAndDamage;
  var surfCrackSample: InternalCrackSample;
  surfCrackSample.mask = 0.0;
  surfCrackSample.normalLocal = vec3<f32>(0.0, 1.0, 0.0);
  surfCrackSample.gapNm = 0.0;
  surfCrackSample.tipStress = 0.0;
  if (blkCrack.w > 0.01) {
    surfCrackSample = evaluateInternalCrackField(input.localPos, blkCrack.xyz, blkCrack.w, seed);
  }

  // Spatially-Resolved Brewster Stress-Optic Birefringence & Cross-Polarized Polariscope Lens:
  // Integrates MuJoCo's local 3-zone contact manifold (pe.xyz at local X = -2.5 cm, 0, +2.5 cm),
  // Euler-Bernoulli flexural bending (|uY|) & Jourawski parabolic shear (1 - uY^2) (pe.w), AND arrested
  // K_I crazing crack-tip residual stress across 3 quadrature samples along the internal refracted ray chord!
  let incLocal = safeNormalize(((*pBlk).worldToLocal * vec4<f32>(incidentDir, 0.0)).xyz, vec3<f32>(0.0, 0.0, 1.0));
  let rayChord = clamp(intersectLocalSlabRaw(input.localPos + incLocal * 0.001, incLocal).y, 0.06, 0.72);
  var chordStress = 0.0;
  for (var si = 0; si < 3; si = si + 1) {
    let tFrac = 0.18 + f32(si) * 0.32;
    let pS = input.localPos + incLocal * (rayChord * tFrac);
    let uX = clamp(pS.x / BLOCK_HX, -1.0, 1.0);
    let uY = clamp(pS.y / BLOCK_HY, -1.0, 1.0);
    let uY2 = uY * uY;
    // Flamant / Saint-Venant contact column spread: tighter near bearing faces (|uY| -> 1), diffusing toward mid-depth
    let colHalfWidth = 0.25 + 0.11 * (1.0 - uY2);
    let dNeg = (uX + 0.667) / colHalfWidth;
    let dMidCol = uX / colHalfWidth;
    let dPos = (uX - 0.667) / colHalfWidth;
    let wNeg = exp(-(dNeg * dNeg));
    let wMid = exp(-(dMidCol * dMidCol));
    let wPos = exp(-(dPos * dPos));
    let bearingCol = (pe.x * wNeg + pe.y * wMid + pe.z * wPos) * (0.76 + 0.28 * uY2);
    // Euler-Bernoulli outer-fiber longitudinal stress + neutral-axis transverse shear
    let bendSpan = pe.w * (abs(uY) * 1.12 + (1.0 - uY2) * 0.34) * (1.0 - 0.26 * uX * uX);
    chordStress = chordStress + (bearingCol + bendSpan);
  }
  let localPhotoStress = clamp(chordStress * (1.0 / 3.0) + surfCrackSample.tipStress * 0.52, 0.0, 1.35);
  let retardNm = localPhotoStress * (250.0 + 950.0 * (rayChord / 0.28));
  let phaseRGB = retardNm * vec3<f32>(3.14159265 / 635.0, 3.14159265 / 546.0, 3.14159265 / 455.0);
  let sinP = sin(phaseRGB);
  let isochromaticRGB = sinP * sinP;

  if (polariscope > 0.5) {
    // Recover passband multi-bounce refracted scene radiance so unloaded prisms (stressGate ~ 0) render as
    // clear smoked neutral-density crystal (~44% refracted background + subtle swatch hue + Brewster bevel speculars)
    // rather than dark charcoal, while keeping stressed dark-field extinction (0.12) crisp so spectral
    // sin^2(pi*delta/lambda) force-chain fringes pop with bright optical contrast!
    let passbandRad = max(refractedRGB.r, max(refractedRGB.g, refractedRGB.b));
    let smokedRefr = mix(vec3<f32>(passbandRad) * vec3<f32>(0.88, 0.93, 1.0), refractedRGB, 0.36);
    let dBrewster = (cosTheta - 0.549) / 0.30;
    let brewsterTower = exp(-dBrewster * dBrewster);
    let stressGate = smoothstep(0.035, 0.42, localPhotoStress);
    let baseTransWeight = mix(0.44, 0.12, stressGate);
    let darkFieldRGB = (smokedRefr * baseTransWeight * (1.0 - fresnel) + reflContrib * (0.62 + 0.26 * brewsterTower) + sunMicroSpec * 0.60) * (1.0 - vitrail.cameMask * 0.80);
    let smokedLuma = dot(smokedRefr, vec3<f32>(0.2126, 0.7152, 0.0722));
    let stressLum = max(0.32, smokedLuma * 1.75 + 0.22);
    finalRGB = darkFieldRGB + isochromaticRGB * stressLum * stressGate * (1.0 - fresnel * 0.50) * (1.0 - vitrail.cameMask * 0.86)
             + vec3<f32>(_keepDepth + _keepShards + _keepOpt + _keepCenters);
  } else if (localPhotoStress > 0.46 && grabbed < 0.5) {
    let subtleBiref = smoothstep(0.46, 0.98, localPhotoStress) * 0.055;
    finalRGB = mix(finalRGB, finalRGB * (0.94 + 0.22 * isochromaticRGB), subtleBiref);
  }

  // Surface-breaking hairline cleavage cracks: true block-local cleavage normal + thin-air-film TIR silver glint on tilt!
  if (surfCrackSample.mask > 0.01) {
    let surfCrack = surfCrackSample.mask;
    let crackWorldN = safeNormalize(mulBlockL2WDir(input.blockIdx, surfCrackSample.normalLocal), glassNormal);
    let crackBevelN = safeNormalize(glassNormal + crackWorldN * 0.68, glassNormal);
    let crackSpec = evalGGXSpecular(crackBevelN, V, L, 0.052);
    let cosViewCleavage = clamp(abs(dot(V, crackWorldN)), 0.0, 1.0);
    let fCleavage = dielectricFresnel(cosViewCleavage, GLASS_IOR_G, 1.0);
    let silverGlint = sampleCathedralEnvironment(reflect(incidentDir, crackBevelN), L, reflSunColor, skyTint, atelierMix) * fCleavage * 0.24;
    finalRGB = finalRGB * (1.0 - surfCrack * 0.26)
             + surfCrack * (sunColor * sunTransRGB * crackSpec * 0.48 + silverGlint);
  }

  // Gap B: 58% Body (|x| <= 0.2175 dm) vs Outer 15.75 mm End-Tip (|x| > 0.2175 dm) 2-Zone Sandbox Grip Cues
  // Strictly revealed on hover/grab in Sandbox (hover > 1.5 || grabbed > 1.5), keeping resting & competitive blocks untouched!
  let dBoundary = abs(abs(input.localPos.x) - 0.2175);
  let boundaryCameLine = 1.0 - smoothstep(0.0016, 0.0030, dBoundary);
  let brassCameAnchor = vec3<f32>(0.96, 0.82, 0.45);
  let jewelPivotTrailer = vec3<f32>(0.72, 0.88, 1.0);
  let isEndZone = abs(input.localPos.x) > 0.2175;
  let isBodyZone = !isEndZone;
  if (hover > 1.5 || grabbed > 1.5) {
    let boundaryCameLit = brassCameAnchor * (skyTint * 0.12 + sunColor * sunTransRGB * 0.26);
    finalRGB = mix(finalRGB, boundaryCameLit, boundaryCameLine * 0.62);
  }

  // Subtle Tactile UI Rim Feedback strictly along beveled glass edges with Sandbox 2-zone anchor differentiation
  let bevelRim = (1.0 - smoothstep(0.002, 0.016, dMid)) * (0.35 + 0.65 * pow(max(0.0, 1.0 - cosTheta), 2.5));
  if (grabbed > 0.5) {
    let looseCol = vec3<f32>(0.92, 0.88, 0.76);
    let snugCol = vec3<f32>(1.0, 0.68, 0.18);
    let pinnedCol = vec3<f32>(1.0, 0.16, 0.16);
    let resCol = select(
      mix(looseCol, snugCol, clamp(resistance * 2.0, 0.0, 1.0)),
      mix(snugCol, pinnedCol, clamp((resistance - 0.5) * 2.0, 0.0, 1.0)),
      resistance > 0.5
    );
    if (grabbed > 1.5) {
      let isGrabEnd = grabbed > 2.5;
      let zoneHue = select(brassCameAnchor, jewelPivotTrailer, isGrabEnd);
      let isGrabActiveZone = select(isBodyZone, isEndZone, isGrabEnd);
      let zoneRimBoost = select(0.28, 0.68, isGrabActiveZone);
      finalRGB += mix(resCol, zoneHue, 0.48) * bevelRim * zoneRimBoost;
      if (vitrail.cameMask > 0.001) {
        finalRGB += zoneHue * vitrail.cameMask * select(0.05, 0.22, isGrabActiveZone);
      }
    } else {
      finalRGB += resCol * bevelRim * 0.52;
    }
  } else if (hover > 0.5) {
    if (isLegal > 0.5) {
      if (hover > 1.5) {
        let isHoverEnd = hover > 2.5;
        if (isHoverEnd) {
          let hoverEndRim = select(
            vec3<f32>(0.80, 0.80, 0.74) * 0.16,
            jewelPivotTrailer * 0.54,
            isEndZone
          );
          finalRGB += hoverEndRim * bevelRim;
          if (vitrail.cameMask > 0.001 && isEndZone) {
            finalRGB += jewelPivotTrailer * vitrail.cameMask * 0.16;
          }
        } else {
          let hoverBodyRim = select(
            brassCameAnchor * 0.50,
            vec3<f32>(0.84, 0.80, 0.68) * 0.16,
            isEndZone
          );
          finalRGB += hoverBodyRim * bevelRim;
          if (vitrail.cameMask > 0.001 && isBodyZone) {
            finalRGB += brassCameAnchor * vitrail.cameMask * 0.16;
          }
        }
      } else {
        finalRGB += vec3<f32>(0.98, 0.88, 0.65) * bevelRim * 0.44;
      }
    } else {
      finalRGB = mix(finalRGB, finalRGB * vec3<f32>(0.72, 0.68, 0.75), 0.35);
    }
  }

  let camDist = length(uScene.cameraPosAndTime.xyz - input.worldPos);
  return vec4<f32>(finalRGB, camDist);
}
`;

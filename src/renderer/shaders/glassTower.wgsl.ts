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
  let halfExt = vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
  let innerHalf = vec3<f32>(BLOCK_HX - BLOCK_BEVEL_R, BLOCK_HY - BLOCK_BEVEL_R, BLOCK_HZ - BLOCK_BEVEL_R);
  let invD = 1.0 / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
  let t0 = (-halfExt - ro) * invD;
  let t1 = ( halfExt - ro) * invD;
  let tSmall = min(t0, t1);
  let tBig = max(t0, t1);

  let rawNear = max(tSmall.x, max(tSmall.y, tSmall.z));
  let rawFar = min(tBig.x, min(tBig.y, tBig.z));

  var res: BoxHit;
  if (rawFar < max(rawNear, 0.0)) {
    res.hit = false;
    res.tNear = rawNear;
    res.tFar = 0.001;
    res.entryNormalLocal = vec3<f32>(0.0, 1.0, 0.0);
    res.exitNormalLocal = vec3<f32>(0.0, 1.0, 0.0);
    return res;
  }

  let tEnter = max(0.0, rawNear);
  let tFar0 = max(0.001, rawFar);
  let pMid = ro + rd * (0.5 * (tEnter + tFar0));
  let qMid = max(abs(pMid) - innerHalf, vec3<f32>(0.0));
  if (dot(qMid, qMid) > (BLOCK_BEVEL_R + 0.0008) * (BLOCK_BEVEL_R + 0.0008)) {
    res.hit = false;
    res.tNear = rawNear;
    res.tFar = tFar0;
    res.entryNormalLocal = vec3<f32>(0.0, 1.0, 0.0);
    res.exitNormalLocal = vec3<f32>(0.0, 1.0, 0.0);
    return res;
  }

  var sharpEntryN = vec3<f32>(0.0, 1.0, 0.0);
  if (tSmall.x >= tSmall.y && tSmall.x >= tSmall.z) {
    sharpEntryN = vec3<f32>(-sign(rd.x), 0.0, 0.0);
  } else if (tSmall.y >= tSmall.x && tSmall.y >= tSmall.z) {
    sharpEntryN = vec3<f32>(0.0, -sign(rd.y), 0.0);
  } else {
    sharpEntryN = vec3<f32>(0.0, 0.0, -sign(rd.z));
  }

  var sharpExitN = vec3<f32>(0.0, 1.0, 0.0);
  if (tBig.x <= tBig.y && tBig.x <= tBig.z) {
    sharpExitN = vec3<f32>(sign(rd.x), 0.0, 0.0);
  } else if (tBig.y <= tBig.x && tBig.y <= tBig.z) {
    sharpExitN = vec3<f32>(0.0, sign(rd.y), 0.0);
  } else {
    sharpExitN = vec3<f32>(0.0, 0.0, sign(rd.z));
  }

  var tNear = rawNear;
  if (rawNear > 0.0) {
    let qEn = max(abs(ro + rd * tEnter) - innerHalf, vec3<f32>(0.0));
    let exEn = max(0.0, length(qEn) - BLOCK_BEVEL_R);
    tNear = min(tFar0 - 0.001, rawNear + exEn * 1.22);
  }
  let qEx = max(abs(ro + rd * tFar0) - innerHalf, vec3<f32>(0.0));
  let exEx = max(0.0, length(qEx) - BLOCK_BEVEL_R);
  let tFar = max(max(0.0, tNear) + 0.001, tFar0 - exEx * 1.22);

  res.hit = true;
  res.tNear = tNear;
  res.tFar = tFar;
  res.entryNormalLocal = evalBeveledBoxNormalLocal(ro + rd * max(0.0, tNear), sharpEntryN);
  res.exitNormalLocal = evalBeveledBoxNormalLocal(ro + rd * tFar, sharpExitN);
  return res;
}

/**
 * Evaluates sparse, tiny trapped 3D seed air bubbles suspended inside antique cathedral glass.
 */
fn evaluateSeedBubbles(roLocal: vec3<f32>, rdLocal: vec3<f32>, segLen: f32, seed: f32) -> f32 {
  var glint: f32 = 0.0;
  for (var s: i32 = 0; s < 3; s = s + 1) {
    let t = segLen * (0.22 + f32(s) * 0.28);
    let p = (roLocal + rdLocal * t) * 36.0 + vec3<f32>(seed * 5.1, seed * 2.7, seed * 8.3);
    let cell = floor(p);
    let f = fract(p) - vec3<f32>(0.5);
    let h = hash31(cell);
    if (h > 0.90) {
      let offset = vec3<f32>(fract(h * 13.7) - 0.5, fract(h * 29.3) - 0.5, fract(h * 47.1) - 0.5) * 0.45;
      let d = length(f - offset);
      let bubbleRadius = 0.048 + fract(h * 91.7) * 0.052;
      let rim = smoothstep(bubbleRadius, bubbleRadius * 0.45, d) * smoothstep(0.0, bubbleRadius * 0.85, d);
      glint += rim * 0.38;
    }
  }
  return clamp(glint, 0.0, 0.55);
}

fn projectTableSunUV(hitPos: vec3<f32>) -> vec2<f32> {
  let sunClip = uScene.sunViewProj * vec4<f32>(hitPos, 1.0);
  let ndc = sunClip.xyz / max(sunClip.w, 1e-6);
  return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn evalTableCausticAndLightAt(hitPos: vec3<f32>, diffuseAlbedo: vec3<f32>, L: vec3<f32>, sunColor: vec3<f32>, skyTint: vec3<f32>, skyScale: f32) -> vec3<f32> {
  let rXZ = length(hitPos.xz);
  let gobo = evaluateStudioWindowGobo(hitPos, L);
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
  let edgeVignette = 1.0 - smoothstep(3.45, 4.30, rXZ) * 0.32;
  let baseOcc = mix(0.45, 1.0, smoothstep(0.28, 0.65, max(abs(hitPos.x), abs(hitPos.z))));
  let ambient = (skyTint * 0.11 + vec3<f32>(0.024, 0.019, 0.016) * skyScale) * baseOcc;
  let direct = sunColor * nDotL * caustic * gobo;
  return diffuseAlbedo * (direct + ambient) * edgeVignette;
}

/**
 * Evaluates refracted background scene for dispersed R, G, B rays simultaneously,
 * sharing procedural walnut grain + optical-depth base when rays land within 3.5 cm
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
  let nDotL = max(0.0, L.y);

  let hitTableG = dirG.y < -1e-4 && posG.y > 0.0;
  var sharedWoodAlbedo = vec3<f32>(0.0);
  var sharedShadowBase = vec3<f32>(1.0);
  var sharedGobo: f32 = 1.0;
  var sharedAmbient = vec3<f32>(0.0);
  var sharedVignette: f32 = 1.0;
  var hitPosG = vec3<f32>(0.0);
  var validTableG = false;

  if (hitTableG) {
    hitPosG = posG + dirG * (-posG.y / dirG.y);
    let r2G = dot(hitPosG.xz, hitPosG.xz);
    if (r2G <= 4.30 * 4.30) {
      let matG = evaluateWalnutTableMaterial(hitPosG.xz, 0.0025);
      sharedWoodAlbedo = matG.albedo * (1.0 - matG.metallic * 0.85);
      sharedGobo = evaluateStudioWindowGobo(hitPosG, L);
      sharedVignette = 1.0 - smoothstep(3.45, 4.30, sqrt(r2G)) * 0.32;
      let baseOccG = mix(0.45, 1.0, smoothstep(0.28, 0.65, max(abs(hitPosG.x), abs(hitPosG.z))));
      sharedAmbient = (skyTint * 0.11 + vec3<f32>(0.024, 0.019, 0.016) * skyScale) * baseOccG;
      let uvG = projectTableSunUV(hitPosG);
      if (uvG.x >= 0.002 && uvG.x <= 0.998 && uvG.y >= 0.002 && uvG.y <= 0.998) {
        let tau = textureSampleLevel(uSunOpticalDepthTex, uLinearSampler, uvG, 0.0).rgb;
        let tauPeak = max(tau.r, max(tau.g, tau.b));
        let cMask = smoothstep(0.005, 0.075, tauPeak);
        let sBase = 0.38 * exp(-tau * 0.52) + 0.17 * exp(-tau * 0.18);
        sharedShadowBase = mix(vec3<f32>(1.0), sBase, cMask);
      }
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
    outG = sharedWoodAlbedo.g * (sunColor.g * nDotL * (sharedShadowBase.g + fluxG) * sharedGobo + sharedAmbient.g) * sharedVignette;
  } else {
    outG = (sampleStudioBackdrop(dirG, L, sunColor, skyTint) + pow(max(0.0, dot(dirG, L)), 5.0) * sunColor * 0.38).g;
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
        outR = sharedWoodAlbedo.r * (sunColor.r * nDotL * (sharedShadowBase.r + fluxR) * sharedGobo + sharedAmbient.r) * sharedVignette;
      } else {
        let matR = evaluateWalnutTableMaterial(hitPosR.xz, 0.0025);
        let albR = matR.albedo * (1.0 - matR.metallic * 0.85);
        outR = evalTableCausticAndLightAt(hitPosR, albR, L, sunColor, skyTint, skyScale).r;
      }
    } else {
      outR = (sampleStudioBackdrop(dirR, L, sunColor, skyTint) + pow(max(0.0, dot(dirR, L)), 5.0) * sunColor * 0.38).r;
    }
  } else {
    outR = (sampleStudioBackdrop(dirR, L, sunColor, skyTint) + pow(max(0.0, dot(dirR, L)), 5.0) * sunColor * 0.38).r;
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
        outB = sharedWoodAlbedo.b * (sunColor.b * nDotL * (sharedShadowBase.b + fluxB) * sharedGobo + sharedAmbient.b) * sharedVignette;
      } else {
        let matB = evaluateWalnutTableMaterial(hitPosB.xz, 0.0025);
        let albB = matB.albedo * (1.0 - matB.metallic * 0.85);
        outB = evalTableCausticAndLightAt(hitPosB, albB, L, sunColor, skyTint, skyScale).b;
      }
    } else {
      outB = (sampleStudioBackdrop(dirB, L, sunColor, skyTint) + pow(max(0.0, dot(dirB, L)), 5.0) * sunColor * 0.38).b;
    }
  } else {
    outB = (sampleStudioBackdrop(dirB, L, sunColor, skyTint) + pow(max(0.0, dot(dirB, L)), 5.0) * sunColor * 0.38).b;
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
  let halfExt = vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
  let innerHalf = vec3<f32>(BLOCK_HX - BLOCK_BEVEL_R, BLOCK_HY - BLOCK_BEVEL_R, BLOCK_HZ - BLOCK_BEVEL_R);
  let invD = 1.0 / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
  let tBig = max((-halfExt - ro) * invD, (halfExt - ro) * invD);
  let tFar0 = max(0.001, min(tBig.x, min(tBig.y, tBig.z)));

  var sharpExitN = vec3<f32>(0.0, 0.0, sign(rd.z));
  if (tBig.x <= tBig.y && tBig.x <= tBig.z) {
    sharpExitN = vec3<f32>(sign(rd.x), 0.0, 0.0);
  } else if (tBig.y <= tBig.x && tBig.y <= tBig.z) {
    sharpExitN = vec3<f32>(0.0, sign(rd.y), 0.0);
  }

  let qEx = max(abs(ro + rd * tFar0) - innerHalf, vec3<f32>(0.0));
  let exEx = max(0.0, length(qEx) - BLOCK_BEVEL_R);
  let tFar = max(0.001, tFar0 - exEx * 1.22);

  var out: ExitBoxHit;
  out.tFar = tFar;
  out.exitNormalLocal = evalBeveledBoxNormalLocal(ro + rd * tFar, sharpExitN);
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
  let accSeed0 = (*pPrim).accentAndSeed;
  let crack0 = (*pPrim).crackAndDamage;
  let seed0 = accSeed0.w;
  let iorR: f32 = 1.514;
  let iorG: f32 = 1.522;
  let iorB: f32 = 1.533;
  let f0: f32 = 0.0428;

  // 1. Refract Green carrier ray + R/B Abbe dispersion into primary block
  var internalDirWorld = refract(incidentDir, frontWorldNormal, 1.0 / iorG);
  if (length(internalDirWorld) < 1e-4) {
    internalDirWorld = reflect(incidentDir, frontWorldNormal);
    return sampleCathedralEnvironment(internalDirWorld, L, sunColor, skyTint);
  }
  internalDirWorld = safeNormalize(internalDirWorld, incidentDir);

  let roLocal = mulBlockW2LPoint(primaryIdx, startWorldPos + internalDirWorld * 0.002);
  let rdLocal = mulBlockW2LDir(primaryIdx, internalDirWorld);

  let box0 = intersectLocalExitBox(roLocal, rdLocal);
  var pathLen0 = box0.tFar;
  let exitPosLocal = roLocal + rdLocal * pathLen0;
  let exitNormalWorld = mulBlockL2WDir(primaryIdx, box0.exitNormalLocal);

  let vitrailExit = evaluateVitrailPattern(exitPosLocal, 0, seed0);
  let exit0RippleWorld = mulBlockL2WDir(primaryIdx, vitrailExit.rippleGrad) * 1.35;
  let exitPerturbedN = safeNormalize(exitNormalWorld + exit0RippleWorld, exitNormalWorld);

  var worldRayPos = mulBlockL2WPoint(primaryIdx, exitPosLocal);
  var worldRayDir = refract(internalDirWorld, -exitPerturbedN, iorG);
  var internalMirrorContrib = vec3<f32>(0.0);

  let internalCosExit = clamp(abs(dot(internalDirWorld, exitPerturbedN)), 0.0, 1.0);
  let internalReflF = f0 + (1.0 - f0) * pow(1.0 - internalCosExit, 5.0);

  if (length(worldRayDir) < 1e-4) {
    let bouncedLocalDir = reflect(rdLocal, -box0.exitNormalLocal);
    let boxBounce = intersectLocalExitBox(exitPosLocal + bouncedLocalDir * 0.001, bouncedLocalDir);
    pathLen0 += boxBounce.tFar * 0.90;
    let finalLocalPos = exitPosLocal + bouncedLocalDir * boxBounce.tFar;
    let finalExitNWorld = mulBlockL2WDir(primaryIdx, boxBounce.exitNormalLocal);
    let bouncedWorldDir = mulBlockL2WDir(primaryIdx, bouncedLocalDir);

    worldRayPos = mulBlockL2WPoint(primaryIdx, finalLocalPos);
    worldRayDir = refract(bouncedWorldDir, -finalExitNWorld, iorG);
    if (length(worldRayDir) < 1e-4) {
      worldRayDir = reflect(bouncedWorldDir, -finalExitNWorld);
    }
    internalMirrorContrib = sampleCathedralEnvironment(bouncedWorldDir, L, sunColor, skyTint) * 0.14;
  } else {
    let partialBounce = reflect(internalDirWorld, -exitPerturbedN);
    internalMirrorContrib = sampleCathedralEnvironment(partialBounce, L, sunColor, skyTint) * internalReflF * 0.22;
  }
  worldRayDir = safeNormalize(worldRayDir, incidentDir);

  let inR0 = safeNormalize(refract(incidentDir, frontWorldNormal, 1.0 / iorR), internalDirWorld);
  let inB0 = safeNormalize(refract(incidentDir, frontWorldNormal, 1.0 / iorB), internalDirWorld);
  var dirR = refract(inR0, -exitPerturbedN, iorR);
  dirR = select(safeNormalize(dirR, worldRayDir), worldRayDir, length(dirR) < 1e-4);
  var dirB = refract(inB0, -exitPerturbedN, iorB);
  dirB = select(safeNormalize(dirB, worldRayDir), worldRayDir, length(dirB) < 1e-4);
  var posR = worldRayPos + (inR0 - internalDirWorld) * pathLen0;
  var posB = worldRayPos + (inB0 - internalDirWorld) * pathLen0;

  let sigma0 = mix(
    absMotif0.xyz,
    accSeed0.xyz,
    0.5 * (vitrailFront.accentMix + vitrailExit.accentMix)
  ) + SODA_LIME_IRON_SIGMA;
  let throughput0 = exp(-sigma0 * pathLen0);

  let bubbleSparkle = select(evaluateSeedBubbles(roLocal, rdLocal, box0.tFar, seed0), 0.0, isRefCard);
  var crackMask: f32 = 0.0;
  var crackSpecAlign: f32 = 0.0;
  if (crack0.w > 0.01) {
    let midLocal = roLocal + rdLocal * (box0.tFar * 0.38);
    let farLocal = roLocal + rdLocal * (box0.tFar * 0.74);
    crackMask = max(
      evaluateInternalCracks(roLocal, crack0.xyz, crack0.w, seed0),
      max(
        evaluateInternalCracks(midLocal, crack0.xyz, crack0.w, seed0),
        evaluateInternalCracks(farLocal, crack0.xyz, crack0.w, seed0)
      )
    );
    let H = safeNormalize(L - incidentDir, frontWorldNormal);
    let crackRelLocal = safeNormalize(roLocal - crack0.xyz, vec3<f32>(1.0, 0.0, 0.0));
    let crackPlaneN = mulBlockL2WDir(primaryIdx, crackRelLocal);
    crackSpecAlign = pow(clamp(abs(dot(crackPlaneN, H)), 0.0, 1.0), 14.0);
  }
  let crackAtten0 = 1.0 - crackMask * 0.45;
  var throughput = throughput0 * (1.0 - internalReflF) * mix(0.76, 1.0, smoothstep(0.02, 0.30, internalCosExit)) * crackAtten0;

  let transmittedSun = sunColor * sunTransRGB;
  let lightEnergy = transmittedSun * 0.42 + skyTint * 0.20;
  let scatterGate = select(1.0, 0.0, isRefCard);
  var internalScatter = (
    bubbleSparkle * 0.22 * lightEnergy * throughput +
    crackMask * (0.04 + 0.85 * crackSpecAlign) * lightEnergy * mix(vec3<f32>(1.0), throughput, 0.65) +
    internalMirrorContrib * throughput
  ) * scatterGate;

  // 2. Continue across up to 2 secondary blocks via global intact-AABB clip + Y-sorted uniform binary search
  let blockCount = u32(uScene.sunDirAndBlockCount.w);
  let innerHalf = vec3<f32>(BLOCK_HX - BLOCK_BEVEL_R, BLOCK_HY - BLOCK_BEVEL_R, BLOCK_HZ - BLOCK_BEVEL_R);
  let bevelLimitSq: f32 = (BLOCK_BEVEL_R + 0.0008) * (BLOCK_BEVEL_R + 0.0008);
  let globalMinAndHalfY = uBlockCenters[96];
  let globalMax = uBlockCenters[97].xyz;
  let maxHalfY = max(0.085, globalMinAndHalfY.w) + 0.025;
  var lastHitIdx = primaryIdx;

  for (var bounce: u32 = 0u; bounce < 2u; bounce = bounce + 1u) {
    if (blockCount <= 1u || max(throughput.r, max(throughput.g, throughput.b)) < 0.025) {
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
      if (projT < -0.405 || projT > nearestT + 0.405) {
        continue;
      }
      let tClamped = clamp(projT, 0.0, nearestT);
      let diff = (worldRayPos + worldRayDir * tClamped) - candCenter;
      let ty0 = (bc.y - bc.w - worldRayPos.y) * invGlobalD.y;
      let ty1 = (bc.y + bc.w - worldRayPos.y) * invGlobalD.y;
      if (dot(diff, diff) > 0.165 || max(ty0, ty1) < -0.025 || min(ty0, ty1) > nearestT) {
        continue;
      }

      let cRo = mulBlockW2LPoint(j, worldRayPos);
      let cRd = mulBlockW2LDir(j, worldRayDir);
      let slab = intersectLocalSlabRaw(cRo, cRd);
      if (slab.y >= max(slab.x, 0.002) && slab.x < nearestT && slab.x > -0.025) {
        let tEnterCand = max(0.0, slab.x);
        let pMid = cRo + cRd * (0.5 * (tEnterCand + slab.y));
        let qMid = max(abs(pMid) - innerHalf, vec3<f32>(0.0));
        if (dot(qMid, qMid) <= bevelLimitSq) {
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
    let vHit = evaluateVitrailPattern(localEntryPos, 0, hitSeed);

    let entryNWorld = mulBlockL2WDir(nearestIdx, nearestBox.entryNormalLocal);
    let cosEntry = clamp(abs(dot(worldRayDir, entryNWorld)), 0.0, 1.0);
    let seamFresnel = f0 + (1.0 - f0) * pow(1.0 - cosEntry, 5.0);
    let isTightContact = 1.0 - smoothstep(0.001, 0.022, nearestT);
    let seamReflDir = reflect(worldRayDir, entryNWorld);
    var seamEnv = skyTint * 0.22 + sunColor * pow(max(0.0, dot(seamReflDir, L)), 6.0) * 0.55;
    if (bounce == 0u) {
      seamEnv = sampleCathedralEnvironment(seamReflDir, L, sunColor, skyTint);
    }
    internalScatter += seamEnv * throughput * seamFresnel * (0.24 + 0.24 * isTightContact) * scatterGate;

    var insideSecondary = refract(worldRayDir, entryNWorld, 1.0 / iorG);
    if (length(insideSecondary) < 1e-4) {
      insideSecondary = reflect(worldRayDir, entryNWorld);
    }
    insideSecondary = safeNormalize(insideSecondary, worldRayDir);

    let insideLocal = mulBlockW2LDir(nearestIdx, insideSecondary);
    let insideBox = intersectLocalExitBox(localEntryPos + insideLocal * 0.001, insideLocal);
    var segLen = clamp(insideBox.tFar, 0.008, 0.76);
    var localExitPos = localEntryPos + insideLocal * segLen;
    let vExit2 = evaluateVitrailPattern(localExitPos, 0, hitSeed);
    let exit2RippleWorld = mulBlockL2WDir(nearestIdx, vExit2.rippleGrad) * 1.15;

    var exitN2World = safeNormalize(
      mulBlockL2WDir(nearestIdx, insideBox.exitNormalLocal) + exit2RippleWorld,
      entryNWorld
    );
    var exitedDir = refract(insideSecondary, -exitN2World, iorG);
    if (length(exitedDir) < 1e-4) {
      let tirLocalDir = reflect(insideLocal, -insideBox.exitNormalLocal);
      let tirBox = intersectLocalExitBox(localExitPos + tirLocalDir * 0.001, tirLocalDir);
      segLen += clamp(tirBox.tFar, 0.005, 0.55) * 0.90;
      localExitPos = localExitPos + tirLocalDir * tirBox.tFar;
      let tirWorldDir = mulBlockL2WDir(nearestIdx, tirLocalDir);
      exitN2World = mulBlockL2WDir(nearestIdx, tirBox.exitNormalLocal);
      exitedDir = refract(tirWorldDir, -exitN2World, iorG);
      if (length(exitedDir) < 1e-4) {
        exitedDir = reflect(tirWorldDir, -exitN2World);
      }
      internalScatter += sampleCathedralEnvironment(tirWorldDir, L, sunColor, skyTint) * throughput * 0.12 * scatterGate;
    }

    let cosExit2 = clamp(abs(dot(insideSecondary, exitN2World)), 0.0, 1.0);
    let exitFresnel2 = f0 + (1.0 - f0) * pow(1.0 - cosExit2, 5.0);
    let avgAccent = 0.5 * (vHit.accentMix + vExit2.accentMix);
    let sigmaHit = mix(hitAbsMotif.xyz, hitAccSeed.xyz, avgAccent) + SODA_LIME_IRON_SIGMA;

    var crackAtten: f32 = 1.0;
    if (hitCrack.w > 0.01) {
      let midLocal2 = 0.5 * (localEntryPos + localExitPos);
      let cMask2 = max(
        evaluateInternalCracks(localEntryPos, hitCrack.xyz, hitCrack.w, hitSeed),
        evaluateInternalCracks(midLocal2, hitCrack.xyz, hitCrack.w, hitSeed)
      );
      crackAtten = 1.0 - cMask2 * 0.42;
      internalScatter += cMask2 * 0.12 * lightEnergy * throughput * crackAtten * scatterGate;
    }

    throughput *= exp(-sigmaHit * segLen) * (1.0 - seamFresnel) * (1.0 - exitFresnel2) * crackAtten;

    let prevPosG = worldRayPos;
    let prevDirG = worldRayDir;
    worldRayDir = safeNormalize(exitedDir, worldRayDir);
    worldRayPos = mulBlockL2WPoint(nearestIdx, localExitPos) + worldRayDir * 0.0025;
    let stepVecG = worldRayPos - prevPosG;

    var inR = refract(dirR, entryNWorld, 1.0 / iorR);
    inR = select(safeNormalize(inR, insideSecondary), insideSecondary, length(inR) < 1e-4);
    var outR = refract(inR, -exitN2World, iorR);
    outR = select(safeNormalize(outR, worldRayDir), worldRayDir, length(outR) < 1e-4);
    posR += (dirR - prevDirG) * nearestT + stepVecG + (inR - insideSecondary) * segLen;
    dirR = outR;

    var inB = refract(dirB, entryNWorld, 1.0 / iorB);
    inB = select(safeNormalize(inB, insideSecondary), insideSecondary, length(inB) < 1e-4);
    var outB = refract(inB, -exitN2World, iorB);
    outB = select(safeNormalize(outB, worldRayDir), worldRayDir, length(outB) < 1e-4);
    posB += (dirB - prevDirG) * nearestT + stepVecG + (inB - insideSecondary) * segLen;
    dirB = outB;

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
            throughput *= exp(-sSigma * sLen) * 0.916;
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

  // Colored upstream solar transmittance forward-scatter through hand-rolled waterglass (strictly 0 when sunColor == 0)
  let forwardAlign = 0.14 + 0.36 * pow(max(0.0, dot(incidentDir, L)), 3.0);
  let glassSunGlow = sqrt(throughput0) * throughput0 * transmittedSun * forwardAlign * 0.34 * scatterGate;

  return transmittedDispersed + glassSunGlow + internalScatter;
}

@fragment
fn fs_glass_tower(input: TowerVSOut) -> @location(0) vec4<f32> {
  // Unconditional derivative & binding keep-alives at top of fragment shader!
  let _keepDepth = f32(textureDimensions(uSunDepthTex).x) * 0.0;
  let _keepShards = f32(arrayLength(&uShards)) * 0.0;
  let _keepOpt = f32(textureDimensions(uSunOpticalDepthTex).x) * 0.0;
  let _keepCenters = uBlockCenters[0].x * 0.0;

  let pBlk = &uBlocks[input.blockIdx];
  let seed = (*pBlk).accentAndSeed.w;

  let V = safeNormalize(uScene.cameraPosAndTime.xyz - input.worldPos, vec3<f32>(0.0, 0.0, 1.0));
  let incidentDir = -V;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));

  let vitrail = evaluateVitrailPattern(input.localPos, 0, seed);

  // Analytical smooth 1.25 mm rounded-box normal across 1-segment bevels & corners
  let smoothLocalN = evalBeveledBoxNormalLocal(input.localPos, input.localNormal);
  let macroWorldN = safeNormalize(((*pBlk).localToWorld * vec4<f32>(smoothLocalN, 0.0)).xyz, input.worldNormal);

  // Upstream-only colored solar transmittance (replaces opaque binary self-shadow blackout!)
  let rawSunVis = sampleSunShadowPCF(uSunDepthTex, uScene.sunViewProj, input.worldPos, macroWorldN, L, 0.022);
  let windowGobo = evaluateStudioWindowGobo(input.worldPos, L);
  let sunDirLocal = ((*pBlk).worldToLocal * vec4<f32>(-L, 0.0)).xyz;
  let selfChord = clamp(intersectLocalSlabRaw(input.localPos + sunDirLocal * 0.001, sunDirLocal).y, 0.015, 0.38);
  let sigmaSelf = mix((*pBlk).absorptionAndMotif.xyz, (*pBlk).accentAndSeed.xyz, vitrail.accentMix) + SODA_LIME_IRON_SIGMA;
  let tauSelf = sigmaSelf * selfChord * 0.78 + vec3<f32>(0.088);
  let optSample = sampleSunOpticalDepth(uSunOpticalDepthTex, uScene.sunViewProj, input.worldPos);
  let tauUpstream = max(vec3<f32>(0.0), optSample.rgb - tauSelf) * (1.0 - rawSunVis);
  let sunTransRGB = mix(exp(-tauUpstream * 0.80), vec3<f32>(1.0), rawSunVis) * windowGobo;
  let directSunVis = rawSunVis * windowGobo;

  // Attenuate waterglass ripples smoothly onto the diamond-polished 1.25 mm outer bevels so bevels reflect sleek straight window strips!
  let ax = BLOCK_HX - abs(input.localPos.x);
  let ay = BLOCK_HY - abs(input.localPos.y);
  let az = BLOCK_HZ - abs(input.localPos.z);
  let dMin = min(ax, min(ay, az));
  let dMax = max(ax, max(ay, az));
  let dMid = ax + ay + az - dMin - dMax;
  let flatFaceMask = smoothstep(0.011, 0.020, dMid);

  let rippleWorld = ((*pBlk).localToWorld * vec4<f32>(vitrail.rippleGrad * flatFaceMask, 0.0)).xyz;
  let glassNormal = safeNormalize(macroWorldN + rippleWorld * 1.45, macroWorldN);

  let refractedRGB = traceTowerGlassRayDispersed(input.worldPos, incidentDir, glassNormal, input.blockIdx, vitrail, sunTransRGB);

  // Exact Schlick Dielectric Fresnel + outer reflections evaluated AFTER multi-bounce raytracer to minimize register pressure
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;
  let cosTheta = clamp(dot(glassNormal, V), 0.0, 1.0);
  let f0: f32 = 0.0428;
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);

  let reflDir = reflect(incidentDir, glassNormal);
  let reflSunColor = sunColor * mix(0.20 + 0.80 * sunTransRGB, vec3<f32>(1.0), directSunVis);
  let envReflection = sampleCathedralEnvironment(reflDir, L, reflSunColor, skyTint);
  let glassRoughness = mix(0.032, 0.055, flatFaceMask);
  let sunMicroSpec = min(
    evalGGXSpecular(glassNormal, V, L, glassRoughness) * fresnel,
    1.15
  ) * sunColor * mix(sunTransRGB * 0.45, vec3<f32>(1.0), directSunVis) * 0.65;

  // Preserve exact Schlick Fresnel up to 70 deg incidence (fresnel <= 0.32), and body-tint extreme grazing reflections so tumbled end-caps never white-out
  let untintedF = min(fresnel, 0.32);
  let grazingExcess = max(0.0, fresnel - 0.32);
  let bodySwatchTint = exp(-sigmaSelf * 0.11);
  let reflContrib = envReflection * (vec3<f32>(untintedF) + bodySwatchTint * (grazingExcess * 0.72));
  var finalRGB = refractedRGB * (1.0 - fresnel) + reflContrib + sunMicroSpec + vec3<f32>(_keepDepth + _keepShards + _keepOpt + _keepCenters);

  let flags = (*pBlk).stateFlags;
  let hover = flags.x;
  let grabbed = flags.y;
  let isLegal = flags.z;
  let resistance = flags.w;

  // Grazing Fresnel glint along surface-breaking hairline cleavage cracks ONLY when aligned with light
  let blkCrack = (*pBlk).crackAndDamage;
  if (blkCrack.w > 0.01) {
    let surfCrack = evaluateInternalCracks(input.localPos, blkCrack.xyz, blkCrack.w, seed);
    if (surfCrack > 0.01) {
      let crackN = safeNormalize(glassNormal + vec3<f32>(0.32, 0.12, -0.26), glassNormal);
      let crackSpec = evalGGXSpecular(crackN, V, L, 0.06);
      finalRGB = finalRGB * (1.0 - surfCrack * 0.28) + surfCrack * sunColor * sunTransRGB * crackSpec * 0.45;
    }
  }

  // Subtle Tactile UI Rim Feedback strictly along beveled glass edges
  let bevelRim = (1.0 - smoothstep(0.002, 0.016, dMid)) * (0.35 + 0.65 * pow(1.0 - cosTheta, 2.5));
  if (grabbed > 0.5) {
    let looseCol = vec3<f32>(0.92, 0.88, 0.76);
    let snugCol = vec3<f32>(1.0, 0.68, 0.18);
    let pinnedCol = vec3<f32>(1.0, 0.16, 0.16);
    let resCol = select(
      mix(looseCol, snugCol, clamp(resistance * 2.0, 0.0, 1.0)),
      mix(snugCol, pinnedCol, clamp((resistance - 0.5) * 2.0, 0.0, 1.0)),
      resistance > 0.5
    );
    finalRGB += resCol * bevelRim * 0.52;
  } else if (hover > 0.5) {
    if (isLegal > 0.5) {
      finalRGB += vec3<f32>(0.98, 0.88, 0.65) * bevelRim * 0.44;
    } else {
      finalRGB = mix(finalRGB, finalRGB * vec3<f32>(0.72, 0.68, 0.75), 0.35);
    }
  }

  let camDist = length(uScene.cameraPosAndTime.xyz - input.worldPos);
  return vec4<f32>(finalRGB, camDist);
}
`;

import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 4: Solid 3D Raytraced Voronoi Glass Shards & Grounded Faceted Glass Cullet Chips.
 * - Traces refracted rays through each shard's exact 3D convex polyhedron half-spaces (`planes[0..11]`)
 * - Uses two-layer depth-sorted backdrop snapshots so foreground shards refract underlying shards!
 * - Applies 2048x2048 PCF sun shadow occlusion, soda-lime iron-oxide thickness absorption,
 *   bounded GGX crystal facet speculars, and 3D half-round metallic filigree relief.
 */
export const GLASS_SHARD_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uShards: array<GPUShardInstance>;
@group(0) @binding(2) var uBackdropTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;
@group(0) @binding(5) var uSunDepthTex: texture_depth_2d;
@group(0) @binding(6) var uSunOpticalDepthTex: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> uShardVerts: array<f32>;
@group(0) @binding(8) var<storage, read> uBlockCenters: array<vec4<f32>>;
@group(0) @binding(9) var uCausticTex: texture_2d<f32>;

struct ShardVSOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) parentLocalPos: vec3<f32>,
  @location(3) centeredPos: vec3<f32>,
  @location(4) localNormal: vec3<f32>,
  @location(5) isOuter: f32,
  @location(6) @interpolate(flat) shardIdx: u32,
};

@vertex
fn vs_glass_shard(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) shardIdx: u32
) -> ShardVSOut {
  let pShV = &uShards[shardIdx];
  let packed = (*pShV).packedMesh;
  let firstVert = (packed >> 8u) & 0x03ffffu;
  let vertCount = max(3u, packed & 255u);
  let localVid = select(vid, 0u, vid >= vertCount);
  let off = (firstVert + localVid) * 10u;

  let centeredPos = vec3<f32>(uShardVerts[off], uShardVerts[off + 1u], uShardVerts[off + 2u]);
  let normal = vec3<f32>(uShardVerts[off + 3u], uShardVerts[off + 4u], uShardVerts[off + 5u]);
  let parentLocalPos = vec3<f32>(uShardVerts[off + 6u], uShardVerts[off + 7u], uShardVerts[off + 8u]);
  let isOuter = uShardVerts[off + 9u];

  let l2wV = (*pShV).localToWorld;
  let wp = l2wV * vec4<f32>(centeredPos, 1.0);
  let wn = safeNormalize((l2wV * vec4<f32>(normal, 0.0)).xyz, vec3<f32>(0.0, 1.0, 0.0));

  var out: ShardVSOut;
  out.clipPos = uScene.viewProj * wp;
  out.worldPos = wp.xyz;
  out.worldNormal = wn;
  out.parentLocalPos = parentLocalPos;
  out.centeredPos = centeredPos;
  out.localNormal = normal;
  out.isOuter = isOuter;
  out.shardIdx = shardIdx;
  return out;
}

struct PolyExitHit {
  t: f32,
  normalLocal: vec3<f32>,
  isOuter: bool,
};

fn intersectConvexShardExit(
  roCentered: vec3<f32>,
  rdLocal: vec3<f32>,
  shardIdx: u32,
  planeCount: u32
) -> PolyExitHit {
  var bestT: f32 = 0.45;
  var bestN = rdLocal;
  var bestOuter = false;
  let pSh = &uShards[shardIdx];

  for (var k: u32 = 0u; k < planeCount; k = k + 1u) {
    let pl = (*pSh).planes[k];
    let denom = dot(pl.xyz, rdLocal);
    if (denom > 1e-4) {
      let t = (abs(pl.w) - dot(pl.xyz, roCentered)) / denom;
      if (t >= 0.0 && t < bestT) {
        bestT = t;
        bestN = pl.xyz;
        bestOuter = pl.w < 0.0;
      }
    }
  }

  var res: PolyExitHit;
  res.t = clamp(bestT, 0.002, 0.45);
  res.normalLocal = bestN;
  res.isOuter = bestOuter;
  return res;
}

@fragment
fn fs_glass_shard(input: ShardVSOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4<f32> {
  // Unconditional derivative & binding keep-alive evaluation at top of fragment shader!
  let dParent = fwidth(input.parentLocalPos);
  let aaFilter = max(dParent.x, max(dParent.y, dParent.z));
  let _keepDepth = f32(textureDimensions(uSunDepthTex).x) * 0.0;
  let _keepOpt = f32(textureDimensions(uSunOpticalDepthTex).x) * 0.0;
  let _keepCaustic = f32(textureDimensions(uCausticTex).x) * 0.0;
  let polariscope = uBlockCenters[97].w;
  let _keepCenters = uBlockCenters[0].x * 0.0;

  let dims = vec2<f32>(textureDimensions(uBackdropTex));
  let screenUV = input.clipPos.xy / max(dims, vec2<f32>(1.0));

  let pSh = &uShards[input.shardIdx];
  let shL2W = (*pSh).localToWorld;
  let shAbs = (*pSh).absorption;
  let shAcc = (*pSh).accentAndSeed;
  let shOrig = (*pSh).originAndPlaneCount;
  let motifId = i32(((*pSh).packedMesh >> 28u) & 3u);
  let seed = shAcc.w;
  let impactOrigin = shOrig.xyz;
  let planeCount = min(MAX_SHARD_PLANES, u32(round(shOrig.w)));

  let V = safeNormalize(uScene.cameraPosAndTime.xyz - input.worldPos, vec3<f32>(0.0, 0.0, 1.0));
  let incidentDir = -V;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);

  let baseWN = safeNormalize(input.worldNormal, vec3<f32>(0.0, 1.0, 0.0));
  let baseLN = safeNormalize(input.localNormal, vec3<f32>(0.0, 1.0, 0.0));
  let geomN = select(-baseWN, baseWN, isFront);
  let localN = select(-baseLN, baseLN, isFront);

  let vitrail = evaluateVitrailPattern(input.parentLocalPos, motifId, seed);

  // Colored upstream solar transmittance through overlying shards & glass blocks!
  let atelierMix = clamp(uScene.tableAoCounts.w, 0.0, 1.0);
  let rawSunVis = sampleSunShadowPCF(uSunDepthTex, uScene.sunViewProj, input.worldPos, geomN, L, 0.018);
  let windowGobo = evaluateStudioWindowGobo(input.worldPos, L, atelierMix);
  let optSample = sampleSunOpticalDepth(uSunOpticalDepthTex, uScene.sunViewProj, input.worldPos);
  let sigmaSelf = mix(shAbs, shAcc.xyz, vitrail.accentMix) + SODA_LIME_IRON_SIGMA;
  let tauSelfApprox = sigmaSelf * 0.09 + vec3<f32>(0.088);
  let tauUpstream = max(vec3<f32>(0.0), optSample.rgb - tauSelfApprox) * (1.0 - rawSunVis);

  let shardTableGate = 1.0 - smoothstep(0.02, 0.18, input.worldPos.y);
  let sunClip = uScene.sunViewProj * vec4<f32>(input.worldPos.x, 0.0, input.worldPos.z, 1.0);
  let sunNdc = sunClip.xyz / max(sunClip.w, 1e-6);
  let sunUV = vec2<f32>(sunNdc.x * 0.5 + 0.5, 0.5 - sunNdc.y * 0.5);
  var causticFlux = vec3<f32>(0.0);
  if (shardTableGate > 0.005 && sunUV.x >= 0.002 && sunUV.x <= 0.998 && sunUV.y >= 0.002 && sunUV.y <= 0.998) {
    causticFlux = textureSampleLevel(uCausticTex, uLinearSampler, sunUV, 0.0).rgb * shardTableGate;
  }

  let polariscopeMod = 1.0 - 0.75 * polariscope;
  let rawSunTrans = mix(exp(-tauUpstream * 0.80), vec3<f32>(1.0), rawSunVis) * windowGobo;
  let sunTransRGB = (rawSunTrans + causticFlux * 0.28 * windowGobo) * polariscopeMod;
  let directSunVis = rawSunVis * windowGobo * polariscopeMod;

  // Measure exact 3D distance & ridge normal along this facet to nearest snapped dihedral glass edge
  var minCleavageEdgeDist: f32 = 1.0;
  var closestEdgePlaneN = localN;
  for (var k: u32 = 0u; k < planeCount; k = k + 1u) {
    let pl = (*pSh).planes[k];
    let cosDih = dot(pl.xyz, localN);
    if (cosDih < 0.965) {
      let sinDih = sqrt(max(0.04, 1.0 - cosDih * cosDih));
      let dEdge = max(0.0, abs(pl.w) - dot(pl.xyz, input.centeredPos)) / sinDih;
      if (dEdge < minCleavageEdgeDist) {
        minCleavageEdgeDist = dEdge;
        closestEdgePlaneN = pl.xyz;
      }
    }
  }

  // Crisp brittle glass cleavage facets (~1-2 deg mirror-smooth shell relief + subtle Wallner & edge hackle)
  // so Voronoi shards flash sharp diamond-like reflections & refractions instead of warped fishbowl lenses!
  let planeHash = hash31(floor(localN * 14.0 + vec3<f32>(0.5)) + vec3<f32>(seed * 3.7));
  let planeSeed = seed + planeHash * 6.28318;
  let dishRef = safeNormalize(vec3<f32>(0.37 + 0.42 * planeHash, 0.86, -0.35 - 0.38 * planeHash), vec3<f32>(0.0, 1.0, 0.0));
  let dishAxisA = safeNormalize(cross(localN, dishRef), vec3<f32>(1.0, 0.0, 0.0));
  let dishAxisB = cross(localN, dishAxisA);
  let cU = dot(input.centeredPos, dishAxisA);
  let cV = dot(input.centeredPos, dishAxisB);
  let shellSign = select(-1.0, 1.0, fract(planeHash * 7.13 + seed * 1.9) > 0.5);
  let conchoidalDish = (
    dishAxisA * (cU * 0.42 + cV * 0.11 + sin(cV * 34.0 + planeSeed * 2.1) * 0.008) +
    dishAxisB * (cV * 0.36 - cU * 0.11 - cos(cU * 29.0 - planeSeed * 1.7) * 0.008)
  ) * shellSign;

  let impactRel = input.parentLocalPos - impactOrigin;
  let rWallner = length(impactRel);
  let wallnerZone = smoothstep(0.014, 0.11, rWallner);
  let wallnerArc = sin(rWallner * 145.0 - planeSeed * 5.0) * 0.009 * wallnerZone;
  let hackleAng = atan2(impactRel.z + impactRel.y * 0.35, impactRel.x - impactRel.y * 0.2);
  let edgeProximity = 1.0 - smoothstep(0.0, 0.022, minCleavageEdgeDist);
  let twistHackle = sin(hackleAng * 26.0 + planeSeed * 4.3) * 0.008 * edgeProximity;

  let cleavagePerturbLocal = conchoidalDish
                           + safeNormalize(impactRel, localN) * wallnerArc
                           + vec3<f32>(-impactRel.z, impactRel.x * 0.3, impactRel.x) * twistHackle * 1.2;
  let cleavagePerturbWorld = (shL2W * vec4<f32>(cleavagePerturbLocal, 0.0)).xyz;
  // Crisp outer face with subtle fracture-rim chip roll
  let outerChipRoll = safeNormalize(closestEdgePlaneN - localN * dot(closestEdgePlaneN, localN), dishAxisA)
                    * (1.0 - smoothstep(0.0, 0.018, minCleavageEdgeDist)) * 0.045;
  let outerPerturbLocal = vitrail.rippleGrad * 0.85 + outerChipRoll + conchoidalDish * 0.15;
  let outerPerturbWorld = (shL2W * vec4<f32>(outerPerturbLocal, 0.0)).xyz;

  let N = safeNormalize(geomN + select(cleavagePerturbWorld, outerPerturbWorld, input.isOuter > 0.5), geomN);

  let r0 = shL2W[0].xyz;
  let r1 = shL2W[1].xyz;
  let r2 = shL2W[2].xyz;
  let ior: f32 = GLASS_IOR_G;

  // 1. Refract ray into front facet of solid 3D Voronoi shard
  var internalWorld = refract(incidentDir, N, 1.0 / ior);
  if (length(internalWorld) < 1e-4) {
    internalWorld = reflect(incidentDir, N);
  }
  internalWorld = safeNormalize(internalWorld, incidentDir);
  let rdLocal = safeNormalize(vec3<f32>(dot(r0, internalWorld), dot(r1, internalWorld), dot(r2, internalWorld)), localN);

  // 2. Intersect exact convex polyhedron exit facet
  let exitHit = intersectConvexShardExit(
    input.centeredPos + rdLocal * 0.0006,
    rdLocal,
    input.shardIdx,
    planeCount
  );

  var totalGlassPath = exitHit.t;
  var exitCentered = input.centeredPos + rdLocal * exitHit.t;
  var exitParentPos = input.parentLocalPos + (exitCentered - input.centeredPos);
  let exitVitrail = evaluateVitrailPattern(exitParentPos, motifId, seed);

  var exitNWorld = safeNormalize((shL2W * vec4<f32>(exitHit.normalLocal, 0.0)).xyz, -N);
  if (exitHit.isOuter) {
    exitNWorld = safeNormalize(exitNWorld + (shL2W * vec4<f32>(exitVitrail.rippleGrad, 0.0)).xyz * 0.75, exitNWorld);
  }

  // 3. Refract out of back facet — un-clamped dielectric Fresnel + Total Internal Reflection crystal diamond fire!
  let reflSunColor = sunColor * mix(0.24 + 0.76 * sunTransRGB, vec3<f32>(1.0), directSunVis);

  let internalExitCos = clamp(abs(dot(internalWorld, exitNWorld)), 0.0, 1.0);
  let internalExitF = dielectricFresnel(internalExitCos, ior, 1.0);
  var outWorldDir = refract(internalWorld, -exitNWorld, ior);
  var tirGlow = vec3<f32>(0.0);
  var tirWeight: f32 = 0.0;
  var dualInterfaceTrans: f32 = 0.960 * (1.0 - internalExitF);

  if (length(outWorldDir) < 1e-4) {
    let bouncedLocal = reflect(rdLocal, -exitHit.normalLocal);
    let bounceHit = intersectConvexShardExit(
      exitCentered + bouncedLocal * 0.0006,
      bouncedLocal,
      input.shardIdx,
      planeCount
    );
    totalGlassPath += bounceHit.t * 0.72;
    exitCentered = exitCentered + bouncedLocal * bounceHit.t;
    exitParentPos = input.parentLocalPos + (exitCentered - input.centeredPos);

    let bouncedWorld = safeNormalize((shL2W * vec4<f32>(bouncedLocal, 0.0)).xyz, internalWorld);
    let finalExitNWorld = safeNormalize((shL2W * vec4<f32>(bounceHit.normalLocal, 0.0)).xyz, exitNWorld);
    let cosBounceExit = clamp(abs(dot(bouncedWorld, finalExitNWorld)), 0.0, 1.0);
    dualInterfaceTrans = 0.960 * max(0.55, 1.0 - dielectricFresnel(cosBounceExit, ior, 1.0));
    outWorldDir = refract(bouncedWorld, -finalExitNWorld, ior);
    if (length(outWorldDir) < 1e-4) {
      outWorldDir = reflect(bouncedWorld, -finalExitNWorld);
    }
    let envBounce = min(sampleCathedralEnvironment(bouncedWorld, L, reflSunColor, skyTint, atelierMix), vec3<f32>(4.2));
    tirGlow = envBounce * (vec3<f32>(0.46 * skyScale) + 0.96 * sunTransRGB);
    tirWeight = 0.56;
  } else if (internalExitF > 0.04) {
    // Sub-critical steep internal Fresnel reflection approaching theta_c = 41.07 deg
    let partialBounceWorld = reflect(internalWorld, -exitNWorld);
    let envPartial = min(sampleCathedralEnvironment(partialBounceWorld, L, reflSunColor, skyTint, atelierMix), vec3<f32>(4.2));
    tirGlow = envPartial * (vec3<f32>(0.42 * skyScale) + 0.90 * sunTransRGB);
    tirWeight = clamp(internalExitF * 0.78, 0.0, 0.65);
  }
  outWorldDir = safeNormalize(outWorldDir, incidentDir);

  // 4. Trace exit ray through up to 2 sibling Voronoi shards in uShards (true multi-shard crystal facet sparkle!)
  var rayPosW = (shL2W * vec4<f32>(exitCentered, 1.0)).xyz + outWorldDir * 0.0012;
  var rayDirW = outWorldDir;
  let shardCount = u32(uScene.skyTintAndMode.w);
  let sibStride = max(1u, shardCount / 28u);
  var sibExtraTauLo = vec4<f32>(0.0);
  var sibExtraTauHi = vec4<f32>(0.0);
  var sibFacetSparkle = vec3<f32>(0.0);
  var sibAtten: f32 = 1.0;
  var sibHits: u32 = 0u;
  for (var s: u32 = 0u; s < shardCount; s = s + sibStride) {
    if (s == input.shardIdx) { continue; }
    let pSib = &uShards[s];
    let sCenter = (*pSib).localToWorld[3].xyz;
    let relC = sCenter - rayPosW;
    let tProj = dot(relC, rayDirW);
    if (tProj > -0.18 && tProj < 0.42) {
      let perpSq = dot(relC, relC) - tProj * tProj;
      if (perpSq < 0.036) {
        let sr0 = (*pSib).localToWorld[0].xyz;
        let sr1 = (*pSib).localToWorld[1].xyz;
        let sr2 = (*pSib).localToWorld[2].xyz;
        let relO = rayPosW - sCenter;
        let roS = vec3<f32>(dot(sr0, relO), dot(sr1, relO), dot(sr2, relO));
        let rdS = vec3<f32>(dot(sr0, rayDirW), dot(sr1, rayDirW), dot(sr2, rayDirW));
        let pCnt = min(MAX_SHARD_PLANES, u32(round((*pSib).originAndPlaneCount.w)));
        var tEn: f32 = 0.0005;
        var tEx: f32 = 0.42;
        var nEnL = vec3<f32>(0.0, 1.0, 0.0);
        var nExL = vec3<f32>(0.0, -1.0, 0.0);
        var okS: bool = pCnt > 0u;
        for (var k: u32 = 0u; k < pCnt; k = k + 1u) {
          let pl = (*pSib).planes[k];
          let num = abs(pl.w) - dot(pl.xyz, roS);
          let den = dot(pl.xyz, rdS);
          if (abs(den) < 1e-6) {
            if (num < 0.0) { okS = false; break; }
          } else {
            let t = num / den;
            if (den > 0.0) {
              if (t < tEx) { tEx = t; nExL = pl.xyz; }
            } else if (t > tEn) {
              tEn = t;
              nEnL = pl.xyz;
            }
            if (tEn > tEx) { okS = false; break; }
          }
        }
        if (okS && tEx > tEn) {
          let sLen = clamp(tEx - tEn, 0.008, 0.24);
          let sibEnW = safeNormalize(((*pSib).localToWorld * vec4<f32>(nEnL, 0.0)).xyz, -rayDirW);
          let sibExW = safeNormalize(((*pSib).localToWorld * vec4<f32>(nExL, 0.0)).xyz, rayDirW);
          let fSibEn = dielectricFresnel(clamp(abs(dot(rayDirW, sibEnW)), 0.0, 1.0), 1.0, ior);
          let sibRefl = reflect(rayDirW, sibEnW);
          let sibEnv = min(sampleCathedralEnvironment(sibRefl, L, reflSunColor, skyTint, atelierMix), vec3<f32>(4.5));
          let sibSpecG = evalGGXSpecular(sibEnW, -rayDirW, L, 0.026) * sunColor * sunTransRGB * 0.85;
          sibFacetSparkle += (sibEnv * (0.28 + 0.72 * fSibEn) * 0.48 + sibSpecG) * sibAtten;

          let sibSig = (*pSib).absorption + SODA_LIME_IRON_SIGMA;
          let sibSp8 = evalSwatchSpectralSigma8(sibSig);
          sibExtraTauLo += sibSp8.lo * sLen;
          sibExtraTauHi += sibSp8.hi * sLen;
          sibAtten *= (1.0 - fSibEn * 0.55) * 0.94;

          var inSib = refract(rayDirW, sibEnW, 1.0 / ior);
          inSib = select(safeNormalize(inSib, rayDirW), reflect(rayDirW, sibEnW), length(inSib) < 1e-4);
          var outSib = refract(inSib, -sibExW, ior);
          if (length(outSib) < 1e-4) {
            let tirSib = reflect(inSib, -sibExW);
            let tirSibEnv = min(sampleCathedralEnvironment(tirSib, L, reflSunColor, skyTint, atelierMix), vec3<f32>(4.5));
            sibFacetSparkle += tirSibEnv * (vec3<f32>(0.35 * skyScale) + 0.85 * sunTransRGB) * 0.42 * sibAtten;
            outSib = tirSib;
          }
          rayPosW = rayPosW + rayDirW * tEx + safeNormalize(outSib, rayDirW) * 0.0012;
          rayDirW = safeNormalize(outSib, rayDirW);
          sibHits = sibHits + 1u;
          if (sibHits >= 2u) { break; }
        }
      }
    }
  }

  let exitWorldPos = rayPosW;
  var projDist: f32 = 0.11;
  if (rayDirW.y < -0.06 && exitWorldPos.y > 0.0) {
    projDist = clamp(-exitWorldPos.y / rayDirW.y, 0.006, 0.36);
  }

  let offsetClip = uScene.viewProj * vec4<f32>(exitWorldPos + rayDirW * projDist, 1.0);
  let refractedNDC = offsetClip.xy / max(offsetClip.w, 1e-5);
  let baseRefractUV = vec2<f32>(refractedNDC.x * 0.5 + 0.5, 0.5 - refractedNDC.y * 0.5);
  let deltaUV = clamp(baseRefractUV - screenUV, vec2<f32>(-0.085), vec2<f32>(0.085));

  let dispBoost = 1.0 + tirWeight * 0.55;
  let dispFracR = (GLASS_IOR_G - GLASS_IOR_R) / (GLASS_IOR_G - 1.0) * 5.2 * dispBoost;
  let dispFracB = (GLASS_IOR_B - GLASS_IOR_G) / (GLASS_IOR_G - 1.0) * 4.8 * dispBoost;
  let uvR = clamp(screenUV + deltaUV * (1.0 - dispFracR), vec2<f32>(0.002), vec2<f32>(0.998));
  let uvG = clamp(screenUV + deltaUV, vec2<f32>(0.002), vec2<f32>(0.998));
  let uvB = clamp(screenUV + deltaUV * (1.0 + dispFracB), vec2<f32>(0.002), vec2<f32>(0.998));

  let camDist = length(uScene.cameraPosAndTime.xyz - input.worldPos);
  let centerBackdrop = textureSampleLevel(uBackdropTex, uLinearSampler, clamp(screenUV, vec2<f32>(0.002), vec2<f32>(0.998)), 0.0);
  let tapR = textureSampleLevel(uBackdropTex, uLinearSampler, uvR, 0.0);
  let tapG = textureSampleLevel(uBackdropTex, uLinearSampler, uvG, 0.0);
  let tapB = textureSampleLevel(uBackdropTex, uLinearSampler, uvB, 0.0);
  let bgR = select(tapR.r, centerBackdrop.r, tapR.a > 0.0 && tapR.a < camDist - 0.035);
  let bgG = select(tapG.g, centerBackdrop.g, tapG.a > 0.0 && tapG.a < camDist - 0.035);
  let bgB = select(tapB.b, centerBackdrop.b, tapB.a > 0.0 && tapB.a < camDist - 0.035);
  let bgRGB = vec3<f32>(bgR, bgG, bgB);

  // 5. 8-Band Spectral Transmittance + Sibling-Shard Crystal Sparkle
  let sigma = mix(
    shAbs,
    shAcc.xyz,
    0.5 * (vitrail.accentMix + exitVitrail.accentMix)
  ) + SODA_LIME_IRON_SIGMA;
  let shSpec8 = evalSwatchSpectralSigma8(sigma);
  var shSpecTau: SpectralTau8;
  shSpecTau.lo = shSpec8.lo * (totalGlassPath * 0.92) + sibExtraTauLo * 0.85;
  shSpecTau.hi = shSpec8.hi * (totalGlassPath * 0.92) + sibExtraTauHi * 0.85;
  let transmittance = integrateSpectralTau8ToRGB(shSpecTau) * max(0.62, dualInterfaceTrans) * sibAtten;

  let edgeRefractionDepth = mix(0.84, 1.0, smoothstep(0.04, 0.36, internalExitCos));

  let cosTheta = clamp(dot(N, V), 0.0, 1.0);
  let fresnel = dielectricFresnel(cosTheta, 1.0, ior);

  let facetRough = select(0.014, 0.020, input.isOuter > 0.5);
  let sunSpec = evalGGXSpecular(N, V, L, facetRough) * fresnel * sunColor * mix(sunTransRGB * 0.50, vec3<f32>(1.0), directSunVis) * 0.95;
  let envRefl = sampleCathedralEnvironment(reflect(incidentDir, N), L, reflSunColor, skyTint, atelierMix);
  let facetEnv = envRefl / (vec3<f32>(1.0) + envRefl * 0.18);

  // Physical dihedral glass ridge optics: crisp refractive rim + sunlit crystal ridge flash
  let sharpEdgeBand = 1.0 - smoothstep(0.0005, 0.0030 + aaFilter * 1.0, minCleavageEdgeDist);
  let ridgeLocalN = safeNormalize(localN + closestEdgePlaneN, localN);
  let ridgeWorldN = safeNormalize((shL2W * vec4<f32>(ridgeLocalN, 0.0)).xyz, N);
  let ridgeGGX = evalGGXSpecular(ridgeWorldN, V, L, 0.052) * sunColor * mix(sunTransRGB * 0.55, vec3<f32>(1.0), directSunVis) * 0.58;
  let ridgeFresnel = dielectricFresnel(clamp(abs(dot(ridgeWorldN, V)), 0.0, 1.0), 1.0, ior);
  let ridgeSkyGlint = sampleCathedralEnvironment(reflect(incidentDir, ridgeWorldN), L, reflSunColor, skyTint, atelierMix) * ridgeFresnel * 0.28;
  let edgeSpecGlint = sharpEdgeBand * (ridgeGGX + ridgeSkyGlint);

  let dihedralEdgeDarken = mix(0.85, 1.0, smoothstep(0.0, 0.0035, minCleavageEdgeDist));
  let meniscusAO = mix(1.0, vitrail.paneMeniscusAO, input.isOuter * 0.55);
  let pathScatter = 1.0 - exp(-totalGlassPath * 3.4);
  var swatchTau: SpectralTau8;
  swatchTau.lo = shSpec8.lo * max(totalGlassPath * 1.15, 0.14);
  swatchTau.hi = shSpec8.hi * max(totalGlassPath * 1.15, 0.14);
  let bodySwatchTint = integrateSpectralTau8ToRGB(swatchTau);
  let forwardLobe = pow(max(0.0, dot(incidentDir, L)), 5.5);
  let facetSunSpark = bodySwatchTint * sunColor * sunTransRGB * pow(max(0.0, dot(reflect(internalWorld, -exitNWorld), L)), 12.0) * 0.38;
  let transmittedSunGlow = (bodySwatchTint * sunColor * sunTransRGB * (0.04 * forwardLobe) * pathScatter) + facetSunSpark;
  let interiorRadiant = (mix(bgRGB * edgeRefractionDepth * dihedralEdgeDarken * meniscusAO, tirGlow, tirWeight) + sibFacetSparkle) * transmittance + transmittedSunGlow;

  let maxUntintedF = select(0.11, 0.14, input.isOuter > 0.5);
  let untintedF = min(fresnel, maxUntintedF);
  let tintedGrazingF = (fresnel - untintedF) * 0.86;
  let grazingTint = mix(vec3<f32>(0.10), bodySwatchTint, 0.92);
  let reflTerm = facetEnv * untintedF + envRefl / (vec3<f32>(1.0) + envRefl * 0.24) * (tintedGrazingF * grazingTint);
  let transmitWeight = 1.0 - untintedF - tintedGrazingF * 0.40;
  let rawGlint = sunSpec + edgeSpecGlint;
  let boundedGlint = rawGlint / (vec3<f32>(1.0) + rawGlint * 0.30);
  let surfaceCaustic = causticFlux * sunColor * 0.04 * clamp(-N.y, 0.0, 1.0) * windowGobo * polariscopeMod;
  var shardRGB = interiorRadiant * transmitWeight + reflTerm + boundedGlint + surfaceCaustic + vec3<f32>(_keepDepth + _keepOpt + _keepCenters + _keepCaustic);

  // Optional recessed 60/40 tin-lead / copper-foil came conductor shading on outer motif facets
  let cameWeight = vitrail.cameMask * input.isOuter;
  if (cameWeight > 0.005) {
    let cameN = safeNormalize(geomN + (shL2W * vec4<f32>(vitrail.cameNormalLocal, 0.0)).xyz, N);
    let metalF0 = select(vec3<f32>(0.34, 0.35, 0.37), vec3<f32>(0.68, 0.41, 0.24), motifId == 3);
    let metalRough = select(0.29, 0.24, motifId == 3);
    let cosCame = clamp(dot(cameN, V), 0.0, 1.0);
    let fMetal = metalF0 + (vec3<f32>(1.0) - metalF0) * pow(1.0 - cosCame, 5.0);
    let cameSpec = evalGGXSpecular(cameN, V, L, metalRough) * fMetal * sunColor * sunTransRGB * 0.95;
    let cameEnv = sampleCathedralEnvironment(reflect(incidentDir, cameN), L, reflSunColor, skyTint, atelierMix) * fMetal * 0.38;
    let cameLit = metalF0 * (skyTint * 0.08 + sunColor * sunTransRGB * max(0.0, dot(cameN, L)) * 0.18) + cameSpec + cameEnv;
    shardRGB = mix(shardRGB, cameLit, cameWeight);
  }

  // Cross-Polarized Dark-Field Polariscope Mode:
  // Unstressed Voronoi shards transmit refracted scene light as smoked neutral-density crystal (~30% crystal refraction
  // + s-polarized Brewster bevel/facet speculars, NEVER opaque matte-black silhouettes!), while snapped cleavage edges,
  // impact-origin Hertzian/Wallner shock wakes, and hackle ridges exhibit frozen sin^2(pi*delta/lambda) fringes!
  if (polariscope > 0.5) {
    let cleavageEdgeStrain = exp(-minCleavageEdgeDist / 0.014) * (1.0 - 0.50 * input.isOuter);
    let impactWakeStrain = exp(-rWallner / 0.052) * (0.46 + 0.54 * abs(sin(rWallner * 88.0 - planeSeed * 3.0)));
    let hackleStrain = abs(twistHackle) * 2.6;
    let residualStress = clamp(cleavageEdgeStrain * 0.70 + impactWakeStrain * 0.52 + hackleStrain, 0.0, 1.0);

    let shardRetardNm = residualStress * (125.0 + 545.0 * clamp(totalGlassPath / 0.15, 0.22, 1.45));
    let shardPhaseRGB = shardRetardNm * vec3<f32>(PI / 635.0, PI / 546.0, PI / 455.0);
    let sinSP = sin(shardPhaseRGB);
    let shardIsochromatic = sinSP * sinSP;

    // Smoked neutral-density crystal refraction: preserves refracted oak grain & sunlit caustics through unstressed shards
    let bgLuma = dot(bgRGB, vec3<f32>(0.2126, 0.7152, 0.0722));
    let smokedCrystalBg = mix(
      vec3<f32>(bgLuma) * vec3<f32>(0.86, 0.91, 0.98),
      bgRGB * mix(vec3<f32>(1.0), transmittance, 0.42),
      0.26
    );
    let chordExtinction = exp(-totalGlassPath * 1.25);
    // Brewster s-polarized oblique rim glint (peaks near theta_B = 56.7 deg -> cosTheta ~ 0.55)
    let dBrewster = (cosTheta - 0.549) / 0.28;
    let brewsterBell = exp(-dBrewster * dBrewster);
    let darkFieldBase = smokedCrystalBg * (0.42 * chordExtinction) * edgeRefractionDepth * dihedralEdgeDarken * meniscusAO * transmitWeight
                      + reflTerm * (0.58 + 0.28 * brewsterBell)
                      + sunSpec * 0.56 + edgeSpecGlint * 0.62;
    let fringeEnergy = max(0.24, bgLuma * 1.55 + dot(transmittedSunGlow, vec3<f32>(0.2126, 0.7152, 0.0722)) * 1.25 + 0.16 * skyScale);
    let fringeGate = smoothstep(0.11, 0.46, residualStress);
    shardRGB = (darkFieldBase + shardIsochromatic * fringeEnergy * fringeGate * (1.0 - fresnel * 0.42)) * (1.0 - cameWeight * 0.84)
             + vec3<f32>(_keepDepth + _keepOpt + _keepCenters + _keepCaustic);
  }

  return vec4<f32>(shardRGB, camDist);
}

// Grounded Faceted Glass Cullet Chips (1-3 mm crushed glass grains on the walnut table)
struct GPUSplinter {
  posAndScale: vec4<f32>,
  colorAndRot: vec4<f32>,
};

@group(0) @binding(4) var<storage, read> uSplinters: array<GPUSplinter>;

struct SplinterOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) facetNormal: vec3<f32>,
  @location(3) tint: vec3<f32>,
};

@vertex
fn vs_splinter(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32
) -> SplinterOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
  );
  let sp = uSplinters[iid];
  let q = quad[vid];
  let ang = sp.colorAndRot.w;
  let ca = cos(ang);
  let sa = sin(ang);
  let rq = vec2<f32>(q.x * ca - q.y * sa, q.x * sa + q.y * ca);

  let pos = sp.posAndScale.xyz;
  let scale = sp.posAndScale.w;

  let airborne = smoothstep(0.012, 0.08, pos.y);
  let flatAxisA = vec3<f32>(1.0, 0.0, 0.0);
  let flatAxisB = vec3<f32>(0.0, 0.0, 1.0);

  let toCam = safeNormalize(uScene.cameraPosAndTime.xyz - pos, vec3<f32>(0.0, 0.0, 1.0));
  let billRight = safeNormalize(cross(vec3<f32>(0.0, 1.0, 0.0), toCam), vec3<f32>(1.0, 0.0, 0.0));
  let billUp = cross(toCam, billRight);

  let axisA = safeNormalize(mix(flatAxisA, billRight, airborne * 0.75), flatAxisA);
  let axisB = safeNormalize(mix(flatAxisB, billUp, airborne * 0.75), flatAxisB);
  let wp = pos + (axisA * rq.x * 1.45 + axisB * rq.y * 0.72) * scale;

  let tiltX = sin(ang * 2.3 + f32(iid) * 1.7) * 0.42;
  let tiltZ = cos(ang * 1.9 - f32(iid) * 2.3) * 0.42;
  let fnWorld = normalize(vec3<f32>(tiltX, 1.0, tiltZ));

  var out: SplinterOut;
  out.clipPos = uScene.viewProj * vec4<f32>(wp, 1.0);
  out.uv = q;
  out.worldPos = wp;
  out.facetNormal = fnWorld;
  out.tint = sp.colorAndRot.xyz;
  return out;
}

@fragment
fn fs_splinter(input: SplinterOut) -> @location(0) vec4<f32> {
  let _keepDepth = f32(textureDimensions(uSunDepthTex).x) * 0.0;
  let _keepOpt = f32(textureDimensions(uSunOpticalDepthTex).x) * 0.0;
  let polariscope = uBlockCenters[97].w;
  let _keepCenters = uBlockCenters[0].x * 0.0;
  let poly = abs(input.uv.x * 0.78 + input.uv.y * 0.18) + abs(input.uv.y * 1.15 - input.uv.x * 0.12);
  if (poly > 1.0) {
    discard;
  }

  let V = safeNormalize(uScene.cameraPosAndTime.xyz - input.worldPos, vec3<f32>(0.0, 0.0, 1.0));
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let N = safeNormalize(input.facetNormal, vec3<f32>(0.0, 1.0, 0.0));
  let rawSunVis = sampleSunShadowPCF(uSunDepthTex, uScene.sunViewProj, input.worldPos, N, L, 0.015);
  let optSample = sampleSunOpticalDepth(uSunOpticalDepthTex, uScene.sunViewProj, input.worldPos);
  let sunTransRGB = mix(exp(-optSample.rgb * 0.65), vec3<f32>(1.0), rawSunVis) * evaluateStudioWindowGobo(input.worldPos, L, clamp(uScene.tableAoCounts.w, 0.0, 1.0));

  let cosTheta = clamp(abs(dot(N, V)), 0.0, 1.0);
  let fresnel = dielectricFresnel(cosTheta, 1.0, GLASS_IOR_G);
  let glint = evalGGXSpecular(N, V, L, 0.038) * 0.65;
  let rim = smoothstep(0.72, 0.98, poly);

  let bodyCol = input.tint * (skyTint * 0.045 + sunColor * 0.035 * sunTransRGB) + skyTint * fresnel * 0.14 + sunColor * sunTransRGB * glint;
  var edgeBright = bodyCol + (sunColor * 0.07 * sunTransRGB + skyTint * 0.06) * rim + vec3<f32>(_keepDepth + _keepOpt + _keepCenters);
  if (polariscope > 0.5) {
    let chipPhase = (65.0 + 190.0 * rim) * vec3<f32>(PI / 635.0, PI / 546.0, PI / 455.0);
    let chipIso = sin(chipPhase) * sin(chipPhase);
    edgeBright = bodyCol * 0.10 + chipIso * (0.04 + 0.16 * rim) + sunColor * sunTransRGB * glint * 0.24 + vec3<f32>(_keepDepth + _keepOpt + _keepCenters);
  }
  let alpha = clamp(0.08 + fresnel * 0.28 + rim * 0.10 + min(0.46, glint), 0.0, 0.72);

  return vec4<f32>(edgeBright * alpha, alpha);
}
`;

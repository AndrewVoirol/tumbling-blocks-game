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
  let sunTransRGB = mix(exp(-tauUpstream * 0.80), vec3<f32>(1.0), rawSunVis) * windowGobo;

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

  // Per-facet 3D conchoidal glass fracture shell curvature (~10-15 deg across shard) + radial Wallner arcs
  // + marginal twist hackle so Voronoi cleavage facets sweep reflections organically instead of flat white cutouts!
  let planeHash = hash31(floor(localN * 14.0 + vec3<f32>(0.5)) + vec3<f32>(seed * 3.7));
  let planeSeed = seed + planeHash * 6.28318;
  let dishRef = safeNormalize(vec3<f32>(0.37 + 0.42 * planeHash, 0.86, -0.35 - 0.38 * planeHash), vec3<f32>(0.0, 1.0, 0.0));
  let dishAxisA = safeNormalize(cross(localN, dishRef), vec3<f32>(1.0, 0.0, 0.0));
  let dishAxisB = cross(localN, dishAxisA);
  let cU = dot(input.centeredPos, dishAxisA);
  let cV = dot(input.centeredPos, dishAxisB);
  let shellSign = select(-1.0, 1.0, fract(planeHash * 7.13 + seed * 1.9) > 0.5);
  let conchoidalDish = (
    dishAxisA * (cU * 4.65 + cV * 1.15 + sin(cV * 34.0 + planeSeed * 2.1) * 0.042) +
    dishAxisB * (cV * 3.85 - cU * 1.15 - cos(cU * 29.0 - planeSeed * 1.7) * 0.042)
  ) * shellSign;

  let impactRel = input.parentLocalPos - impactOrigin;
  let rWallner = length(impactRel);
  let wallnerZone = smoothstep(0.014, 0.11, rWallner);
  let wallnerArc = sin(rWallner * 145.0 - planeSeed * 5.0) * 0.032 * wallnerZone;
  let hackleAng = atan2(impactRel.z + impactRel.y * 0.35, impactRel.x - impactRel.y * 0.2);
  let edgeProximity = 1.0 - smoothstep(0.0, 0.028, minCleavageEdgeDist);
  let twistHackle = sin(hackleAng * 26.0 + planeSeed * 4.3) * 0.022 * edgeProximity;

  let cleavagePerturbLocal = conchoidalDish
                           + safeNormalize(impactRel, localN) * wallnerArc
                           + vec3<f32>(-impactRel.z, impactRel.x * 0.3, impactRel.x) * twistHackle * 2.6;
  let cleavagePerturbWorld = (shL2W * vec4<f32>(cleavagePerturbLocal, 0.0)).xyz;
  // Chipped fracture-rim roll-off + cast waterglass curvature on outer shard faces
  let outerChipRoll = safeNormalize(closestEdgePlaneN - localN * dot(closestEdgePlaneN, localN), dishAxisA)
                    * (1.0 - smoothstep(0.0, 0.032, minCleavageEdgeDist)) * 0.11;
  let outerPerturbLocal = vitrail.rippleGrad * 1.95 + outerChipRoll + conchoidalDish * 0.28;
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
    exitNWorld = safeNormalize(exitNWorld + (shL2W * vec4<f32>(exitVitrail.rippleGrad, 0.0)).xyz * 1.25, exitNWorld);
  }

  // 3. Refract out of back facet — with exact dielectric Fresnel transition into Total Internal Reflection (TIR)!
  let directSunVis = rawSunVis * windowGobo;
  let reflSunColor = sunColor * mix(0.14 + 0.86 * sunTransRGB, vec3<f32>(1.0), directSunVis);

  let internalExitCos = clamp(abs(dot(internalWorld, exitNWorld)), 0.0, 1.0);
  let internalExitF = dielectricFresnel(internalExitCos, ior, 1.0);
  var outWorldDir = refract(internalWorld, -exitNWorld, ior);
  var tirGlow = vec3<f32>(0.0);
  var tirWeight: f32 = 0.0;
  var dualInterfaceTrans: f32 = 0.9572 * (1.0 - internalExitF);

  if (length(outWorldDir) < 1e-4) {
    let bouncedLocal = reflect(rdLocal, -exitHit.normalLocal);
    let bounceHit = intersectConvexShardExit(
      exitCentered + bouncedLocal * 0.0006,
      bouncedLocal,
      input.shardIdx,
      planeCount
    );
    totalGlassPath += bounceHit.t * 0.88;
    exitCentered = exitCentered + bouncedLocal * bounceHit.t;
    exitParentPos = input.parentLocalPos + (exitCentered - input.centeredPos);

    let bouncedWorld = safeNormalize((shL2W * vec4<f32>(bouncedLocal, 0.0)).xyz, internalWorld);
    let finalExitNWorld = safeNormalize((shL2W * vec4<f32>(bounceHit.normalLocal, 0.0)).xyz, exitNWorld);
    let cosBounceExit = clamp(abs(dot(bouncedWorld, finalExitNWorld)), 0.0, 1.0);
    dualInterfaceTrans = 0.9572 * max(0.22, 1.0 - dielectricFresnel(cosBounceExit, ior, 1.0));
    outWorldDir = refract(bouncedWorld, -finalExitNWorld, ior);
    if (length(outWorldDir) < 1e-4) {
      outWorldDir = reflect(bouncedWorld, -finalExitNWorld);
    }
    tirGlow = min(sampleCathedralEnvironment(bouncedWorld, L, reflSunColor, skyTint, atelierMix), vec3<f32>(1.05))
            * (vec3<f32>(0.24 * skyScale) + 0.58 * sunTransRGB);
    tirWeight = 0.14;
  } else if (internalExitF > 0.06) {
    // Sub-critical steep internal Fresnel reflection approaching theta_c = 41.07 deg
    let partialBounceWorld = reflect(internalWorld, -exitNWorld);
    tirGlow = min(sampleCathedralEnvironment(partialBounceWorld, L, reflSunColor, skyTint, atelierMix), vec3<f32>(1.05))
            * (vec3<f32>(0.22 * skyScale) + 0.52 * sunTransRGB);
    tirWeight = internalExitF * 0.22;
  }
  outWorldDir = safeNormalize(outWorldDir, incidentDir);

  // 4. Project doubly-refracted exit ray onto depth-sorted backdrop snapshot with Cauchy Abbe dispersion (V_d ~ 58)
  let exitWorldPos = (shL2W * vec4<f32>(exitCentered, 1.0)).xyz;
  var projDist: f32 = 0.10;
  if (outWorldDir.y < -0.06 && exitWorldPos.y > 0.0) {
    projDist = clamp(-exitWorldPos.y / outWorldDir.y, 0.006, 0.34);
  }

  let offsetClip = uScene.viewProj * vec4<f32>(exitWorldPos + outWorldDir * projDist, 1.0);
  let refractedNDC = offsetClip.xy / max(offsetClip.w, 1e-5);
  let baseRefractUV = vec2<f32>(refractedNDC.x * 0.5 + 0.5, 0.5 - refractedNDC.y * 0.5);
  let deltaUV = clamp(baseRefractUV - screenUV, vec2<f32>(-0.075), vec2<f32>(0.075));

  let dispBoost = 1.0 + tirWeight * 0.65;
  let dispFracR = (GLASS_IOR_G - GLASS_IOR_R) / (GLASS_IOR_G - 1.0) * 5.8 * dispBoost;
  let dispFracB = (GLASS_IOR_B - GLASS_IOR_G) / (GLASS_IOR_G - 1.0) * 5.2 * dispBoost;
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

  // 5. Exact Beer-Lambert + Soda-Lime Iron-Oxide volumetric absorption + Maxwell-Fresnel transmission
  let sigma = mix(
    shAbs,
    shAcc.xyz,
    0.5 * (vitrail.accentMix + exitVitrail.accentMix)
  ) + SODA_LIME_IRON_SIGMA;
  let transmittance = exp(-sigma * totalGlassPath * 1.32) * max(0.35, dualInterfaceTrans);

  let edgeRefractionDepth = mix(0.58, 1.0, smoothstep(0.05, 0.48, internalExitCos));

  let cosTheta = clamp(dot(N, V), 0.0, 1.0);
  let fresnel = dielectricFresnel(cosTheta, 1.0, ior);

  let facetRough = select(0.036, 0.052, input.isOuter > 0.5);
  let sunSpec = evalGGXSpecular(N, V, L, facetRough) * fresnel * sunColor * sunTransRGB * 0.72;
  let envRefl = sampleCathedralEnvironment(reflect(incidentDir, N), L, reflSunColor, skyTint, atelierMix);
  let facetEnv = envRefl / (vec3<f32>(1.0) + envRefl * 0.58);

  // Physical dihedral glass ridge optics: dark refractive rim + colored-sun-gated crystal ridge flash
  let sharpEdgeBand = 1.0 - smoothstep(0.0004, 0.0024 + aaFilter * 1.0, minCleavageEdgeDist);
  let ridgeLocalN = safeNormalize(localN + closestEdgePlaneN, localN);
  let ridgeWorldN = safeNormalize((shL2W * vec4<f32>(ridgeLocalN, 0.0)).xyz, N);
  let ridgeGGX = evalGGXSpecular(ridgeWorldN, V, L, 0.075) * sunColor * sunTransRGB * 0.34;
  let ridgeFresnel = dielectricFresnel(clamp(abs(dot(ridgeWorldN, V)), 0.0, 1.0), 1.0, ior);
  let ridgeSkyGlint = sampleCathedralEnvironment(reflect(incidentDir, ridgeWorldN), L, reflSunColor, skyTint, atelierMix) * ridgeFresnel * 0.18;
  let edgeSpecGlint = sharpEdgeBand * (ridgeGGX + ridgeSkyGlint);

  let dihedralEdgeDarken = mix(0.68, 1.0, smoothstep(0.0, 0.0040, minCleavageEdgeDist));
  let meniscusAO = mix(1.0, vitrail.paneMeniscusAO, input.isOuter);
  let pathScatter = 1.0 - exp(-totalGlassPath * 3.6);
  let bodySwatchTint = exp(-sigma * max(totalGlassPath * 1.45, 0.18));
  let forwardLobe = pow(max(0.0, dot(incidentDir, L)), 3.5);
  let transmittedSunGlow = bodySwatchTint * sunColor * sunTransRGB * pathScatter * (0.018 + 0.15 * forwardLobe);
  let interiorRadiant = mix(bgRGB * edgeRefractionDepth * dihedralEdgeDarken * meniscusAO, tirGlow, tirWeight) * transmittance + transmittedSunGlow;

  let maxUntintedF = select(0.062, 0.088, input.isOuter > 0.5);
  let untintedF = min(fresnel, maxUntintedF);
  let tintedGrazingF = (fresnel - untintedF) * 0.82;
  let grazingTint = mix(vec3<f32>(0.06), bodySwatchTint, 0.92);
  let reflTerm = facetEnv * untintedF + envRefl / (vec3<f32>(1.0) + envRefl * 0.32) * (tintedGrazingF * grazingTint);
  let transmitWeight = 1.0 - untintedF - tintedGrazingF * 0.45;
  let rawGlint = sunSpec + edgeSpecGlint;
  let boundedGlint = rawGlint / (vec3<f32>(1.0) + rawGlint * 0.45);
  var shardRGB = interiorRadiant * transmitWeight + reflTerm + boundedGlint + vec3<f32>(_keepDepth + _keepOpt + _keepCenters);

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
  // Unstressed Voronoi shard interiors extinguish darkly (meanLuma < 0.08), while snapped cleavage edges,
  // impact-origin Hertzian/Wallner shock wakes, and oblique Brewster rims exhibit frozen sin^2(pi*delta/lambda) fringes!
  if (polariscope > 0.5) {
    let cleavageEdgeStrain = exp(-minCleavageEdgeDist / 0.0125) * (1.0 - 0.54 * input.isOuter);
    let impactWakeStrain = exp(-rWallner / 0.048) * (0.46 + 0.54 * abs(sin(rWallner * 88.0 - planeSeed * 3.0)));
    let hackleStrain = abs(twistHackle) * 2.6;
    let residualStress = clamp(cleavageEdgeStrain * 0.66 + impactWakeStrain * 0.50 + hackleStrain, 0.0, 1.0);

    let shardRetardNm = residualStress * (115.0 + 545.0 * clamp(totalGlassPath / 0.15, 0.22, 1.45));
    let shardPhaseRGB = shardRetardNm * vec3<f32>(PI / 635.0, PI / 546.0, PI / 455.0);
    let sinSP = sin(shardPhaseRGB);
    let shardIsochromatic = sinSP * sinSP;

    // Brewster s-polarized oblique rim glint (peaks near theta_B = 56.7 deg -> cosTheta ~ 0.55)
    let brewsterBell = exp(-pow((cosTheta - 0.549) / 0.28, 2.0));
    let darkFieldBase = bgRGB * transmittance * 0.042 * meniscusAO
                      + reflTerm * (0.06 + 0.11 * brewsterBell)
                      + sunSpec * 0.15 + edgeSpecGlint * 0.05;
    let fringeEnergy = max(0.16, dot(bgRGB * transmittance + transmittedSunGlow, vec3<f32>(0.2126, 0.7152, 0.0722)) * 1.40 + 0.13 * skyScale);
    let fringeGate = smoothstep(0.14, 0.50, residualStress);
    shardRGB = (darkFieldBase + shardIsochromatic * fringeEnergy * fringeGate * (1.0 - fresnel * 0.45)) * (1.0 - cameWeight * 0.88)
             + vec3<f32>(_keepDepth + _keepOpt + _keepCenters);
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

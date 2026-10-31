import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 4: Solid 3D Raytraced Voronoi Glass Shards & Grounded Faceted Glass Cullet Chips.
 * - Traces refracted rays through each shard's exact 3D convex polyhedron half-spaces (`planes[0..11]`)
 * - Computes exact per-pixel Beer-Lambert optical path length, backface copper-foil cames,
 *   and true internal Total Internal Reflection (TIR) inside acute glass wedges!
 * - Renders sub-millimeter dihedral cleavage edge catchlights + curved Wallner fracture arcs
 * - Grounded faceted glass cullet grains (zero floating ambient dust particles).
 */
export const GLASS_SHARD_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

const MAX_SHARD_PLANES: u32 = 12u;

struct GPUShardInstance {
  localToWorld: mat4x4<f32>,
  absorptionAndMotif: vec4<f32>,
  accentAndSeed: vec4<f32>,
  originAndPlaneCount: vec4<f32>,
  planes: array<vec4<f32>, 12>,
};

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uShards: array<GPUShardInstance>;
@group(0) @binding(2) var uBackdropTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;

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
  @location(0) centeredPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) parentLocalPos: vec3<f32>,
  @location(3) isOuter: f32,
  @builtin(instance_index) shardIdx: u32
) -> ShardVSOut {
  let sh = uShards[shardIdx];
  let wp = sh.localToWorld * vec4<f32>(centeredPos, 1.0);
  let wn = normalize((sh.localToWorld * vec4<f32>(normal, 0.0)).xyz);

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

  for (var k: u32 = 0u; k < planeCount; k = k + 1u) {
    let pl = uShards[shardIdx].planes[k];
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
  // Unconditional derivative evaluation at top of fragment shader for strict WGSL uniform control flow!
  let dParent = fwidth(input.parentLocalPos);
  let aaFilter = max(dParent.x, max(dParent.y, dParent.z));

  let dims = vec2<f32>(textureDimensions(uBackdropTex));
  let screenUV = input.clipPos.xy / max(dims, vec2<f32>(1.0));

  let sh = uShards[input.shardIdx];
  let motifId = i32(round(sh.absorptionAndMotif.w));
  let seed = sh.accentAndSeed.w;
  let impactOrigin = sh.originAndPlaneCount.xyz;
  let planeCount = min(MAX_SHARD_PLANES, u32(round(sh.originAndPlaneCount.w)));

  let V = normalize(uScene.cameraPosAndTime.xyz - input.worldPos);
  let incidentDir = -V;
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let vitrail = evaluateVitrailPattern(input.parentLocalPos, motifId, seed);
  let geomN = select(-normalize(input.worldNormal), normalize(input.worldNormal), isFront);
  let localN = select(-normalize(input.localNormal), normalize(input.localNormal), isFront);

  // Measure exact 3D distance & ridge normal along this facet to the nearest snapped dihedral glass edge!
  var minCleavageEdgeDist: f32 = 1.0;
  var closestEdgePlaneN = localN;
  for (var k: u32 = 0u; k < planeCount; k = k + 1u) {
    let pl = sh.planes[k];
    let cosDih = dot(pl.xyz, localN);
    if (cosDih < 0.965) {
      let sinDih = sqrt(max(0.04, 1.0 - cosDih * cosDih));
      let dEdge = max(0.0, abs(pl.w) - dot(pl.xyz, input.centeredPos)) / sinDih;
      let bothOuter = (input.isOuter > 0.5) && (pl.w < 0.0);
      if (!bothOuter && dEdge < minCleavageEdgeDist) {
        minCleavageEdgeDist = dEdge;
        closestEdgePlaneN = pl.xyz;
      }
    }
  }

  // Fracture surface micro-topography:
  // - Original outer faces keep hand-rolled cathedral waterglass ripples
  // - Newly cleaved internal faces exhibit a glassy mirror zone near impactOrigin + curved Wallner arcs & edge hackle!
  let impactRel = input.parentLocalPos - impactOrigin;
  let rWallner = length(impactRel);
  let wallnerZone = smoothstep(0.025, 0.14, rWallner);
  let wallnerArc = sin(rWallner * 145.0 - seed * 6.0) * 0.0020 * wallnerZone;
  let hackleAng = atan2(impactRel.z + impactRel.y * 0.35, impactRel.x - impactRel.y * 0.2);
  let edgeProximity = 1.0 - smoothstep(0.0, 0.022, minCleavageEdgeDist);
  let twistHackle = sin(hackleAng * 28.0 + seed * 4.3) * 0.0026 * edgeProximity;

  let cleavagePerturbLocal = normalize(impactRel + vec3<f32>(1e-4, 2e-4, -1e-4)) * wallnerArc
                           + vec3<f32>(-impactRel.z, impactRel.x * 0.3, impactRel.x) * twistHackle * 3.5;
  let cleavagePerturbWorld = (sh.localToWorld * vec4<f32>(cleavagePerturbLocal, 0.0)).xyz;
  let rippleWorld = (sh.localToWorld * vec4<f32>(vitrail.rippleGrad, 0.0)).xyz;

  let N = normalize(geomN + select(cleavagePerturbWorld, rippleWorld * 1.45, input.isOuter > 0.5));

  // Orthonormal rotation rows of sh.localToWorld for fast world->local ray transform
  let r0 = sh.localToWorld[0].xyz;
  let r1 = sh.localToWorld[1].xyz;
  let r2 = sh.localToWorld[2].xyz;
  let ior: f32 = 1.522;

  // 1. Refract ray into front facet of solid 3D Voronoi shard
  var internalWorld = refract(incidentDir, N, 1.0 / ior);
  if (length(internalWorld) < 1e-4) {
    internalWorld = reflect(incidentDir, N);
  }
  internalWorld = normalize(internalWorld);
  let rdLocal = normalize(vec3<f32>(dot(r0, internalWorld), dot(r1, internalWorld), dot(r2, internalWorld)));

  // 2. Intersect exact convex polyhedron exit facet!
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

  var exitNWorld = normalize((sh.localToWorld * vec4<f32>(exitHit.normalLocal, 0.0)).xyz);
  if (exitHit.isOuter) {
    exitNWorld = normalize(exitNWorld + (sh.localToWorld * vec4<f32>(exitVitrail.rippleGrad, 0.0)).xyz * 1.25);
  }

  // 3. Refract out of back facet — or undergo true Total Internal Reflection (TIR) inside acute glass wedges!
  var outWorldDir = refract(internalWorld, -exitNWorld, ior);
  var tirGlow = vec3<f32>(0.0);
  var tirWeight: f32 = 0.0;

  if (length(outWorldDir) < 1e-4) {
    let bouncedLocal = reflect(rdLocal, -exitHit.normalLocal);
    let bounceHit = intersectConvexShardExit(
      exitCentered + bouncedLocal * 0.0006,
      bouncedLocal,
      input.shardIdx,
      planeCount
    );
    totalGlassPath += bounceHit.t * 0.85;
    exitCentered = exitCentered + bouncedLocal * bounceHit.t;
    exitParentPos = input.parentLocalPos + (exitCentered - input.centeredPos);

    let bouncedWorld = normalize((sh.localToWorld * vec4<f32>(bouncedLocal, 0.0)).xyz);
    let finalExitNWorld = normalize((sh.localToWorld * vec4<f32>(bounceHit.normalLocal, 0.0)).xyz);
    outWorldDir = refract(bouncedWorld, -finalExitNWorld, ior);
    if (length(outWorldDir) < 1e-4) {
      outWorldDir = reflect(bouncedWorld, -finalExitNWorld);
    }
    // Acute wedge TIR acts like a jewel prism mirror reflecting the cathedral lancet windows!
    tirGlow = sampleCathedralEnvironment(bouncedWorld, L, sunColor, skyTint);
    tirWeight = 0.52;
  }
  outWorldDir = normalize(outWorldDir);

  // 4. Project doubly-refracted exit ray from exact 3D exit point onto backdrop (table / tower) with RGB Abbe dispersion!
  let exitWorldPos = (sh.localToWorld * vec4<f32>(exitCentered, 1.0)).xyz;
  var projDist: f32 = 0.11;
  if (outWorldDir.y < -0.06 && exitWorldPos.y > 0.0) {
    projDist = clamp(-exitWorldPos.y / outWorldDir.y, 0.006, 0.36);
  }

  let offsetClip = uScene.viewProj * vec4<f32>(exitWorldPos + outWorldDir * projDist, 1.0);
  let refractedNDC = offsetClip.xy / max(offsetClip.w, 1e-5);
  let baseRefractUV = vec2<f32>(refractedNDC.x * 0.5 + 0.5, 0.5 - refractedNDC.y * 0.5);
  let deltaUV = clamp(baseRefractUV - screenUV, vec2<f32>(-0.09), vec2<f32>(0.09));

  let dispBoost = 1.0 + tirWeight * 0.65;
  let uvR = clamp(screenUV + deltaUV * (1.0 - 0.11 * dispBoost), vec2<f32>(0.002), vec2<f32>(0.998));
  let uvG = clamp(screenUV + deltaUV, vec2<f32>(0.002), vec2<f32>(0.998));
  let uvB = clamp(screenUV + deltaUV * (1.0 + 0.13 * dispBoost), vec2<f32>(0.002), vec2<f32>(0.998));

  let bgR = textureSampleLevel(uBackdropTex, uLinearSampler, uvR, 0.0).r;
  let bgG = textureSampleLevel(uBackdropTex, uLinearSampler, uvG, 0.0).g;
  let bgB = textureSampleLevel(uBackdropTex, uLinearSampler, uvB, 0.0).b;
  let bgRGB = vec3<f32>(bgR, bgG, bgB);

  // 5. Exact Beer-Lambert volumetric absorption across true 3D shard chord + backface came visibility!
  let sigma = mix(
    sh.absorptionAndMotif.xyz,
    sh.accentAndSeed.xyz,
    0.5 * (vitrail.accentMix + exitVitrail.accentMix)
  );
  // Thin razor splinters (totalGlassPath ~ 0.008 dm) are water-clear; thick prism cores (~0.18 dm) are rich jewel glass!
  var transmittance = exp(-sigma * totalGlassPath * 1.05);
  if (exitHit.isOuter && exitVitrail.cameMask > 0.04) {
    transmittance *= (1.0 - exitVitrail.cameMask * 0.72);
  }
  // Subtle internal Fresnel loss at steep internal exit angles gives solid glass shards realistic dark refractive edges
  let internalExitCos = clamp(abs(dot(internalWorld, exitNWorld)), 0.0, 1.0);
  let edgeRefractionDepth = mix(0.62, 1.0, smoothstep(0.05, 0.48, internalExitCos));

  let cosTheta = clamp(dot(N, V), 0.0, 1.0);
  let f0: f32 = 0.0426;
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);

  let H = normalize(L + V);
  // Fresh internal glass fracture facets have razor-crisp diamond specular glints!
  let specExp = select(310.0, 175.0, input.isOuter > 0.5);
  let sunSpec = pow(max(0.0, dot(N, H)), specExp) * sunColor * select(3.6, 2.5, input.isOuter > 0.5);
  let envRefl = sampleCathedralEnvironment(reflect(incidentDir, N), L, sunColor, skyTint);

  // Physical dihedral glass ridge optics:
  // Unlit snapped glass edges show crisp dark refractive depth (never uniform white wireframe outlines!),
  // while ridges whose 3D bisecting bevel normal aligns with the sun/window flash like real cut crystal!
  let sharpEdgeBand = 1.0 - smoothstep(0.0004, 0.0028 + aaFilter * 1.1, minCleavageEdgeDist);
  let ridgeLocalN = normalize(localN + closestEdgePlaneN + vec3<f32>(1e-5));
  let ridgeWorldN = normalize((sh.localToWorld * vec4<f32>(ridgeLocalN, 0.0)).xyz);
  let ridgeSunSpec = pow(max(0.0, dot(ridgeWorldN, H)), 42.0) * sunColor * 2.4;
  let ridgeFresnel = pow(1.0 - clamp(abs(dot(ridgeWorldN, V)), 0.0, 1.0), 4.0);
  let ridgeSkyGlint = sampleCathedralEnvironment(reflect(incidentDir, ridgeWorldN), L, sunColor, skyTint) * ridgeFresnel * 0.34;
  let edgeSpecGlint = sharpEdgeBand * (ridgeSunSpec + ridgeSkyGlint);

  let dihedralEdgeDarken = mix(0.76, 1.0, smoothstep(0.0, 0.0034, minCleavageEdgeDist));
  let interiorRadiant = mix(bgRGB * edgeRefractionDepth * dihedralEdgeDarken, tirGlow, tirWeight) * transmittance;
  var shardRGB = mix(interiorRadiant, envRefl, fresnel) + sunSpec + edgeSpecGlint;

  // Original exterior faces preserve their soldered patinated copper-foil / pewter came border!
  let outerCame = smoothstep(0.15 - aaFilter * 2.0, 0.85 + aaFilter * 2.0, vitrail.cameMask) * input.isOuter;
  if (outerCame > 0.02) {
    let nDotL = max(0.0, dot(geomN, L));
    let solderCol = vec3<f32>(0.13, 0.105, 0.095) * (sunColor * nDotL * 0.55 + skyTint * 0.35 + vec3<f32>(0.06))
                  + vec3<f32>(0.44, 0.30, 0.21) * pow(max(0.0, dot(geomN, H)), 28.0) * sunColor * 0.45;
    shardRGB = mix(shardRGB, solderCol, outerCame);
  }

  return vec4<f32>(shardRGB, 1.0);
}

// Grounded Faceted Glass Cullet Chips (1-3 mm crushed glass grains on the oak table, zero floating dust!)
struct GPUSplinter {
  posAndScale: vec4<f32>,
  colorAndRot: vec4<f32>, // rgb = glass transmittance tint, w = rot angle
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

  // Resting chips lie flat on the oak table (XZ plane) with a random tilted cleavage facet normal;
  // airborne tumbling chips orient across their spin axis.
  let airborne = smoothstep(0.012, 0.08, pos.y);
  let flatAxisA = vec3<f32>(1.0, 0.0, 0.0);
  let flatAxisB = vec3<f32>(0.0, 0.0, 1.0);

  let toCam = normalize(uScene.cameraPosAndTime.xyz - pos);
  let billRight = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), toCam));
  let billUp = cross(toCam, billRight);

  let axisA = normalize(mix(flatAxisA, billRight, airborne * 0.75));
  let axisB = normalize(mix(flatAxisB, billUp, airborne * 0.75));
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
  // Asymmetric angular glass chip silhouette (sharp quadrilateral splinter)
  let poly = abs(input.uv.x * 0.78 + input.uv.y * 0.18) + abs(input.uv.y * 1.15 - input.uv.x * 0.12);
  if (poly > 1.0) {
    discard;
  }

  let V = normalize(uScene.cameraPosAndTime.xyz - input.worldPos);
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let H = normalize(L + V);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let N = normalize(input.facetNormal);
  let cosTheta = clamp(abs(dot(N, V)), 0.0, 1.0);
  let fresnel = 0.06 + 0.82 * pow(1.0 - cosTheta, 4.0);
  // Pinpoint crystalline specular glint only when chip facet aligns with sun/eye half-vector!
  let glint = pow(max(0.0, dot(N, H)), 90.0) * 2.6;
  let rim = smoothstep(0.68, 0.98, poly);

  let bodyCol = input.tint * (skyTint * 0.45 + sunColor * 0.16) + skyTint * fresnel * 0.35 + sunColor * glint;
  let edgeBright = bodyCol + (sunColor * 0.18 + skyTint * 0.22) * rim;
  let alpha = clamp(0.38 + fresnel * 0.42 + rim * 0.18 + min(0.4, glint), 0.0, 0.92);

  // Premultiplied alpha output
  return vec4<f32>(edgeBright * alpha, alpha);
}
`;

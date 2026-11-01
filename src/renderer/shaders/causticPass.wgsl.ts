import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 1a: 2048x2048 Sun Shadow Depth Map (`vs_sun_depth_block`, `vs_sun_depth_shard` -> depth32float)
 * Pass 1b: 2048x2048 Colored Light-Space Optical Depth + Min-Caster-Height Atlas (`rgba16float`, RGB add + Alpha min)
 *          Accumulates spectral optical thickness tau_sun = sigma * chord - ln((1-F)^2) in .rgb and minCasterY in .a.
 * Pass 1c: 2048x2048 Additive Vertex-Refracted Snell Caustic Flux Map (`rgba16float`, additive src + dst)
 *          Projects tessellated waterglass & 1.25 mm beveled glass blocks + Voronoi shards onto y = 0,
 *          weighted by exact dpdx/dpdy 2D area-compression Jacobians so bright caustic ribbons form while conserving flux.
 */
export const CAUSTIC_PASS_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uBlocks: array<GPUBlockData>;
@group(0) @binding(2) var<storage, read> uShards: array<GPUShardInstance>;
@group(0) @binding(3) var uSunOpticalDepthTex: texture_2d<f32>;
@group(0) @binding(4) var uSunDepthTex: texture_depth_2d;
@group(0) @binding(5) var<storage, read> uShardVerts: array<f32>;

fn loadShardVertOffset(shardIdx: u32, vid: u32) -> u32 {
  let packed = uShards[shardIdx].packedMesh;
  let firstVert = packed >> 8u;
  let vertCount = max(3u, packed & 255u);
  let localVid = select(vid, 0u, vid >= vertCount);
  return (firstVert + localVid) * 10u;
}

fn computeSunChordThroughBlock(roLocal: vec3<f32>, rdLocal: vec3<f32>) -> f32 {
  let halfExt = vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
  let invD = 1.0 / select(rdLocal, vec3<f32>(1e-6), abs(rdLocal) < vec3<f32>(1e-6));
  let t0 = (-halfExt - roLocal) * invD;
  let t1 = ( halfExt - roLocal) * invD;
  let tBig = max(t0, t1);
  let tFar = min(tBig.x, min(tBig.y, tBig.z));
  return clamp(tFar, 0.015, 0.78);
}

// ============================================================================
// Pass 1a: Depth-Only Sun Shadow Casters (2048x2048 depth32float)
// ============================================================================

@vertex
fn vs_sun_depth_block(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> @builtin(position) vec4<f32> {
  let blk = uBlocks[instIdx];
  let worldPos = blk.localToWorld * vec4<f32>(pos, 1.0);
  let _keepNormal = normal.x * 0.0;
  return uScene.sunViewProj * worldPos + vec4<f32>(_keepNormal, 0.0, 0.0, 0.0);
}

@vertex
fn vs_sun_depth_shard(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) shardIdx: u32
) -> @builtin(position) vec4<f32> {
  let off = loadShardVertOffset(shardIdx, vid);
  let centeredPos = vec3<f32>(uShardVerts[off], uShardVerts[off + 1u], uShardVerts[off + 2u]);
  let sh = uShards[shardIdx];
  let worldPos = sh.localToWorld * vec4<f32>(centeredPos, 1.0);
  return uScene.sunViewProj * worldPos;
}

// ============================================================================
// Pass 1b: Spectral Optical Depth + Min-Caster-Height Atlas (RGB add, Alpha min)
// ============================================================================

struct BlockOptDepthOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) localPos: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) worldNormal: vec3<f32>,
  @location(3) @interpolate(flat) blockIdx: u32,
};

@vertex
fn vs_optical_depth_block(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> BlockOptDepthOut {
  let blk = uBlocks[instIdx];
  let wp = blk.localToWorld * vec4<f32>(pos, 1.0);
  let wn = safeNormalize((blk.localToWorld * vec4<f32>(normal, 0.0)).xyz, vec3<f32>(0.0, 1.0, 0.0));

  var out: BlockOptDepthOut;
  out.clipPos = uScene.sunViewProj * wp;
  out.localPos = pos;
  out.worldPos = wp.xyz;
  out.worldNormal = wn;
  out.blockIdx = instIdx;
  return out;
}

@fragment
fn fs_optical_depth_block(input: BlockOptDepthOut) -> @location(0) vec4<f32> {
  let blk = uBlocks[input.blockIdx];
  let motifId: i32 = 0;
  let seed = blk.accentAndSeed.w;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));

  let sunDirLocal = safeNormalize((blk.worldToLocal * vec4<f32>(-L, 0.0)).xyz, vec3<f32>(0.0, -1.0, 0.0));
  let chordDm = computeSunChordThroughBlock(input.localPos + sunDirLocal * 0.001, sunDirLocal);
  let exitLocalPos = input.localPos + sunDirLocal * chordDm;
  let exitWorldPos = (blk.localToWorld * vec4<f32>(exitLocalPos, 1.0)).xyz;

  let vitrailEntry = evaluateVitrailPattern(input.localPos, motifId, seed);
  let vitrailExit = evaluateVitrailPattern(exitLocalPos, motifId, seed);
  let sigma = mix(
    blk.absorptionAndMotif.xyz,
    blk.accentAndSeed.xyz,
    0.5 * (vitrailEntry.accentMix + vitrailExit.accentMix)
  ) + SODA_LIME_IRON_SIGMA;

  let cosSun = clamp(dot(safeNormalize(input.worldNormal, L), L), 0.0, 1.0);
  let fEntry = 0.0426 + 0.30 * pow(1.0 - cosSun, 5.0);
  let crackShadow = evaluateInternalCracks(input.localPos, blk.crackAndDamage.xyz, blk.crackAndDamage.w, seed);
  let fresnelTau = -log(max(0.18, (1.0 - fEntry) * 0.9574 * (1.0 - crackShadow * 0.38)));

  let tauRGB = sigma * min(chordDm, 0.38) * 0.78 + vec3<f32>(fresnelTau);
  let casterExitY = clamp(min(input.worldPos.y, exitWorldPos.y), 0.0, 9.5);
  return vec4<f32>(tauRGB, casterExitY);
}

struct ShardOptDepthOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) parentLocalPos: vec3<f32>,
  @location(1) centeredPos: vec3<f32>,
  @location(2) worldNormal: vec3<f32>,
  @location(3) worldY: f32,
  @location(4) @interpolate(flat) shardIdx: u32,
};

@vertex
fn vs_optical_depth_shard(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) shardIdx: u32
) -> ShardOptDepthOut {
  let off = loadShardVertOffset(shardIdx, vid);
  let centeredPos = vec3<f32>(uShardVerts[off], uShardVerts[off + 1u], uShardVerts[off + 2u]);
  let normal = vec3<f32>(uShardVerts[off + 3u], uShardVerts[off + 4u], uShardVerts[off + 5u]);
  let parentLocalPos = vec3<f32>(uShardVerts[off + 6u], uShardVerts[off + 7u], uShardVerts[off + 8u]);
  let sh = uShards[shardIdx];
  let worldPos = sh.localToWorld * vec4<f32>(centeredPos, 1.0);
  let wn = safeNormalize((sh.localToWorld * vec4<f32>(normal, 0.0)).xyz, vec3<f32>(0.0, 1.0, 0.0));

  var out: ShardOptDepthOut;
  out.clipPos = uScene.sunViewProj * worldPos;
  out.parentLocalPos = parentLocalPos;
  out.centeredPos = centeredPos;
  out.worldNormal = wn;
  out.worldY = worldPos.y;
  out.shardIdx = shardIdx;
  return out;
}

@fragment
fn fs_optical_depth_shard(input: ShardOptDepthOut) -> @location(0) vec4<f32> {
  let sh = uShards[input.shardIdx];
  let motifId: i32 = 0;
  let seed = sh.accentAndSeed.w;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));

  let r0 = sh.localToWorld[0].xyz;
  let r1 = sh.localToWorld[1].xyz;
  let r2 = sh.localToWorld[2].xyz;
  let sunDirLocal = safeNormalize(vec3<f32>(dot(r0, -L), dot(r1, -L), dot(r2, -L)), vec3<f32>(0.0, -1.0, 0.0));

  let planeCount = min(MAX_SHARD_PLANES, u32(round(sh.originAndPlaneCount.w)));
  var tExit: f32 = 0.09;
  for (var k: u32 = 0u; k < planeCount; k = k + 1u) {
    let pl = sh.planes[k];
    let denom = dot(pl.xyz, sunDirLocal);
    if (denom > 1e-4) {
      let t = (abs(pl.w) - dot(pl.xyz, input.centeredPos)) / denom;
      if (t > 0.0 && t < tExit) {
        tExit = t;
      }
    }
  }
  let chordDm = clamp(tExit, 0.010, 0.36);
  let vitrail = evaluateVitrailPattern(input.parentLocalPos, motifId, seed);
  let sigma = mix(sh.absorption, sh.accentAndSeed.xyz, vitrail.accentMix) + SODA_LIME_IRON_SIGMA;

  let cosSun = clamp(dot(safeNormalize(input.worldNormal, L), L), 0.0, 1.0);
  let fEntry = 0.0426 + 0.34 * pow(1.0 - cosSun, 5.0);
  let fresnelTau = -log(max(0.20, (1.0 - fEntry) * 0.9574));

  let tauRGB = sigma * chordDm * 0.92 + vec3<f32>(fresnelTau);
  let exitWorldY = clamp(input.worldY - L.y * chordDm, 0.0, 9.5);
  return vec4<f32>(tauRGB, exitWorldY);
}

// ============================================================================
// Pass 1c: Additive Vertex-Refracted Snell Caustic Flux (src + dst)
// ============================================================================

struct BlockCausticFluxOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) origTableXZ: vec2<f32>,
  @location(1) deflTableXZ: vec2<f32>,
  @location(2) entryWorldPos: vec3<f32>,
  @location(3) shadowTestWorldPos: vec3<f32>,
  @location(4) worldNormal: vec3<f32>,
  @location(5) tauSelf: vec3<f32>,
  @location(6) faceTint: vec3<f32>,
};

@vertex
fn vs_block_caustic_flux(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> BlockCausticFluxOut {
  let blk = uBlocks[instIdx];
  let motifId: i32 = 0;
  let seed = blk.accentAndSeed.w;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));

  let wp = (blk.localToWorld * vec4<f32>(pos, 1.0)).xyz;
  let macroWN = safeNormalize((blk.localToWorld * vec4<f32>(normal, 0.0)).xyz, vec3<f32>(0.0, 1.0, 0.0));
  let isHorizCap = step(0.5, abs(normal.y));

  // Inset horizontal cap shadow-test positions by 2.0 mm (> 1.25 mm block bevel) so covered interior caps
  // never peek out under upper block bevels in uSunDepthTex, while uncovered Layer-17 caps stay 100% sunlit.
  let insetLocalPos = vec3<f32>(
    clamp(pos.x, -(0.375 - 0.020 * isHorizCap), 0.375 - 0.020 * isHorizCap),
    pos.y,
    clamp(pos.z, -(0.125 - 0.020 * isHorizCap), 0.125 - 0.020 * isHorizCap)
  );
  let shadowTestWP = (blk.localToWorld * vec4<f32>(insetLocalPos, 1.0)).xyz;

  let sunDirLocal = safeNormalize((blk.worldToLocal * vec4<f32>(-L, 0.0)).xyz, vec3<f32>(0.0, -1.0, 0.0));
  let chordDm = computeSunChordThroughBlock(pos + sunDirLocal * 0.001, sunDirLocal);
  let exitLocalPos = pos + sunDirLocal * chordDm;

  let vitrailEntry = evaluateVitrailPattern(pos, motifId, seed);
  let vitrailExit = evaluateVitrailPattern(exitLocalPos, motifId, seed);

  let sunSlopeY = max(0.20, L.y);
  let origT = max(0.0, wp.y) / sunSlopeY;
  let origXZ = wp.xz - L.xz * origT;

  // C-infinity world-space cathedral waterglass & flank focal refraction:
  // Depends purely on continuous world position wp (and origXZ(wp)), guaranteeing 0.000 mm continuity
  // across +X / +/-Z / +Y seams, Even/Odd cross-rows, and all 18 horizontal layer boundaries.
  let contactHeightGate = smoothstep(0.05, 0.28, wp.y);
  let u1 = origXZ.x * 5.2 + origXZ.y * 3.8 + wp.y * 1.3;
  let u2 = origXZ.y * 5.5 - origXZ.x * 3.5 + wp.y * 1.1;
  let waterglassDelta = vec2<f32>(
    sin(u1) * 0.046 + cos(u2) * 0.026,
    cos(u2 * 0.92 + 0.7) * 0.046 - sin(u1 * 0.88 - 0.4) * 0.026
  );
  let perpSun3 = safeNormalize(vec3<f32>(-L.z, 0.0, L.x), vec3<f32>(1.0, 0.0, 0.0));
  let perpSunXZ = perpSun3.xz;
  let uTrans = dot(wp.xz, perpSunXZ);
  let flankPrismDelta = perpSunXZ * sin(clamp(uTrans / 0.46, -1.0, 1.0) * 1.5707963) * 0.052;

  let rawDelta = (waterglassDelta + flankPrismDelta) * contactHeightGate;
  let deltaLen = length(rawDelta);
  let deflXZ = origXZ + rawDelta / (1.0 + deltaLen / 0.22);

  let sigma = mix(
    blk.absorptionAndMotif.xyz,
    blk.accentAndSeed.xyz,
    0.5 * (vitrailEntry.accentMix + vitrailExit.accentMix)
  ) + SODA_LIME_IRON_SIGMA;
  let cosSun = dot(macroWN, L);
  let facingGate = step(0.005, cosSun);
  let cosClamped = clamp(cosSun, 0.0, 1.0);
  let fEntry = 0.0426 + 0.30 * pow(1.0 - cosClamped, 5.0);
  let fresnelTau = -log(max(0.18, (1.0 - fEntry) * 0.9574));
  let tauSelf = sigma * min(chordDm, 0.38) * 0.78 + vec3<f32>(fresnelTau);

  let crackShadow = evaluateInternalCracks(pos, blk.crackAndDamage.xyz, blk.crackAndDamage.w, seed);
  let faceTint = (1.0 - fEntry) * 0.9574 * (1.0 - crackShadow * 0.40) * facingGate * exp(-sigma * 0.04);

  let projectedWorld = vec4<f32>(deflXZ.x, 0.0, deflXZ.y, 1.0);

  var out: BlockCausticFluxOut;
  out.clipPos = uScene.sunViewProj * projectedWorld;
  out.origTableXZ = origXZ;
  out.deflTableXZ = deflXZ;
  out.entryWorldPos = wp;
  out.shadowTestWorldPos = shadowTestWP;
  out.worldNormal = macroWN;
  out.tauSelf = tauSelf;
  out.faceTint = faceTint;
  return out;
}

@fragment
fn fs_block_caustic_flux(input: BlockCausticFluxOut) -> @location(0) vec4<f32> {
  // Unconditional screen-space derivatives & binding keep-alives at top of fragment shader!
  let dOrigX = dpdx(input.origTableXZ);
  let dOrigY = dpdy(input.origTableXZ);
  let dDeflX = dpdx(input.deflTableXZ);
  let dDeflY = dpdy(input.deflTableXZ);
  let _keepOpt = f32(textureDimensions(uSunOpticalDepthTex).x) * 0.0;
  let _keepDepth = f32(textureDimensions(uSunDepthTex).x) * 0.0;
  if (max(input.faceTint.r, max(input.faceTint.g, input.faceTint.b)) < 1e-4) {
    discard;
  }

  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));
  let rawSunVis = sampleSunShadowPCF(uSunDepthTex, uScene.sunViewProj, input.shadowTestWorldPos, input.worldNormal, L, 0.012);
  // Single-entry radiative transfer: each solar ray enters the sunlit outer shell (rawSunVis > 0) ONCE,
  // preventing 3x vs 1x additive overcount on alternating Jenga cross-rows and 17 covered +Y caps.
  let sunShellVis = smoothstep(0.25, 0.75, rawSunVis);
  if (sunShellVis < 1e-3) {
    discard;
  }

  let origArea = abs(dOrigX.x * dOrigY.y - dOrigX.y * dOrigY.x);
  let deflArea = abs(dDeflX.x * dDeflY.y - dDeflX.y * dDeflY.x);
  let invJacobian = clamp(origArea / max(deflArea, 0.22 * origArea + 1e-7), 0.0, 3.4);

  let optSample = sampleSunOpticalDepth(uSunOpticalDepthTex, uScene.sunViewProj, input.entryWorldPos);
  let totalRayTau = max(input.tauSelf, optSample.rgb);
  let rayBeer = exp(-totalRayTau * 0.46);

  let fluxRGB = input.faceTint * rayBeer * invJacobian * sunShellVis * 0.62 + vec3<f32>(_keepOpt + _keepDepth);
  return vec4<f32>(fluxRGB, 0.0);
}

struct ShardCausticFluxOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) origTableXZ: vec2<f32>,
  @location(1) deflTableXZ: vec2<f32>,
  @location(2) beerPass: vec3<f32>,
};

@vertex
fn vs_shard_caustic(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) shardIdx: u32
) -> ShardCausticFluxOut {
  let off = loadShardVertOffset(shardIdx, vid);
  let centeredPos = vec3<f32>(uShardVerts[off], uShardVerts[off + 1u], uShardVerts[off + 2u]);
  let normal = vec3<f32>(uShardVerts[off + 3u], uShardVerts[off + 4u], uShardVerts[off + 5u]);
  let parentLocalPos = vec3<f32>(uShardVerts[off + 6u], uShardVerts[off + 7u], uShardVerts[off + 8u]);
  let isOuter = uShardVerts[off + 9u];
  let sh = uShards[shardIdx];
  let motifId: i32 = 0;
  let seed = sh.accentAndSeed.w;
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));

  let wp = (sh.localToWorld * vec4<f32>(centeredPos, 1.0)).xyz;
  let wn = safeNormalize((sh.localToWorld * vec4<f32>(normal, 0.0)).xyz, vec3<f32>(0.0, 1.0, 0.0));

  let vitrail = evaluateVitrailPattern(parentLocalPos, motifId, seed);
  let rippleWorld = (sh.localToWorld * vec4<f32>(vitrail.rippleGrad * 18.0 * isOuter, 0.0)).xyz;
  let surfN = safeNormalize(wn + rippleWorld, wn);

  var refrDir = refract(-L, surfN, 1.0 / 1.522);
  if (length(refrDir) < 1e-4) {
    refrDir = -L;
  } else {
    refrDir = safeNormalize(refrDir, -L);
  }

  let origT = max(0.0, wp.y) / max(0.20, L.y);
  let origXZ = wp.xz - L.xz * origT;
  let deflT = max(0.0, wp.y) / max(0.20, -refrDir.y);
  let rawDeflXZ = wp.xz + refrDir.xz * deflT;
  let deflXZ = origXZ + clamp(rawDeflXZ - origXZ, vec2<f32>(-0.18), vec2<f32>(0.18));

  let sigma = mix(sh.absorption, sh.accentAndSeed.xyz, vitrail.accentMix) + SODA_LIME_IRON_SIGMA;
  let cosSun = clamp(dot(wn, L), 0.0, 1.0);
  let fEntry = 0.0426 + 0.34 * pow(1.0 - cosSun, 5.0);
  let beerPass = exp(-sigma * 0.08) * (1.0 - fEntry) * 0.9574 * smoothstep(0.02, 0.16, cosSun);

  var out: ShardCausticFluxOut;
  out.clipPos = uScene.sunViewProj * vec4<f32>(deflXZ.x, 0.0, deflXZ.y, 1.0);
  out.origTableXZ = origXZ;
  out.deflTableXZ = deflXZ;
  out.beerPass = beerPass;
  return out;
}

@fragment
fn fs_shard_caustic(input: ShardCausticFluxOut) -> @location(0) vec4<f32> {
  let dOrigX = dpdx(input.origTableXZ);
  let dOrigY = dpdy(input.origTableXZ);
  let dDeflX = dpdx(input.deflTableXZ);
  let dDeflY = dpdy(input.deflTableXZ);
  if (max(input.beerPass.r, max(input.beerPass.g, input.beerPass.b)) < 1e-4) {
    discard;
  }

  let origArea = abs(dOrigX.x * dOrigY.y - dOrigX.y * dOrigY.x);
  let deflArea = abs(dDeflX.x * dDeflY.y - dDeflX.y * dDeflY.x);
  let invJacobian = clamp(origArea / max(deflArea, 0.16 * origArea + 1e-8), 0.0, 4.5);

  return vec4<f32>(input.beerPass * invJacobian * 0.24, 0.0);
}
`;

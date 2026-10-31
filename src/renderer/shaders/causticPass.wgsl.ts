import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 1: Colored Stained-Glass Sun Transmission & Waterglass Caustic Map (2048x2048 rgba16float).
 * Uses multiplicative optical blending (dst = src * dst) so overlapping stained-glass blocks
 * and 3D Voronoi shards physically multiply Beer-Lambert spectral transmittance while
 * grazing cleavage facets deflect sunlight into crisp dark refractive borders around bright caustic cores.
 */
export const CAUSTIC_PASS_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uBlocks: array<GPUBlockData>;

struct BlockCausticOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) localPos: vec3<f32>,
  @location(1) @interpolate(flat) blockIdx: u32,
};

@vertex
fn vs_block_caustic(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> BlockCausticOut {
  let blk = uBlocks[instIdx];
  let worldPos = blk.localToWorld * vec4<f32>(pos, 1.0);
  // Reference normal so WGSL vertex layout matches our shared 6-float vertex buffer
  let _n = normal.x * 0.0;

  var out: BlockCausticOut;
  out.clipPos = uScene.sunViewProj * worldPos + vec4<f32>(_n, 0.0, 0.0, 0.0);
  out.localPos = pos;
  out.blockIdx = instIdx;
  return out;
}

@fragment
fn fs_block_caustic(input: BlockCausticOut) -> @location(0) vec4<f32> {
  let blk = uBlocks[input.blockIdx];
  let motifId = i32(round(blk.absorptionAndMotif.w));
  let seed = blk.accentAndSeed.w;

  let vitrail = evaluateVitrailPattern(input.localPos, motifId, seed);
  let sigma = mix(blk.absorptionAndMotif.xyz, blk.accentAndSeed.xyz, vitrail.accentMix);

  // Luminous cathedral stained-glass sun transmittance (single-pass back-face culled per block)
  let rawBeer = exp(-sigma * 0.115);
  let glassTransmittance = clamp(rawBeer * 1.14 + vec3<f32>(0.05), vec3<f32>(0.18), vec3<f32>(1.18));

  // Waterglass ripple caustic focusing: bright golden-jewel caustic ridges (> 1.0 in rgba16float!)
  let wavePeak = pow(clamp(vitrail.rippleHeight * 0.5 + 0.5, 0.0, 1.0), 2.2);
  let causticFocus = 0.84 + wavePeak * 0.58;
  let crackShadow = evaluateInternalCracks(
    input.localPos,
    blk.crackAndDamage.xyz,
    blk.crackAndDamage.w,
    seed
  );

  var trans = glassTransmittance * causticFocus * (1.0 - crackShadow * 0.34);
  // Soldered copper-foil / lead came lines cast delicate architectural mullion lines
  trans = mix(trans, trans * 0.52, vitrail.cameMask * 0.78);

  return vec4<f32>(trans, 1.0);
}

// Shard caustic sub-pipeline (exact 76-float / 304-byte GPUShardInstance layout)
const MAX_SHARD_PLANES: u32 = 12u;

struct GPUShardInstance {
  localToWorld: mat4x4<f32>,
  absorptionAndMotif: vec4<f32>,
  accentAndSeed: vec4<f32>,
  originAndPlaneCount: vec4<f32>,
  planes: array<vec4<f32>, 12>,
};

@group(0) @binding(2) var<storage, read> uShards: array<GPUShardInstance>;

struct ShardCausticOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) parentLocalPos: vec3<f32>,
  @location(1) centeredPos: vec3<f32>,
  @location(2) worldNormal: vec3<f32>,
  @location(3) worldY: f32,
  @location(4) isOuter: f32,
  @location(5) @interpolate(flat) shardIdx: u32,
};

@vertex
fn vs_shard_caustic(
  @location(0) centeredPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) parentLocalPos: vec3<f32>,
  @location(3) isOuter: f32,
  @builtin(instance_index) shardIdx: u32
) -> ShardCausticOut {
  let sh = uShards[shardIdx];
  let worldPos = sh.localToWorld * vec4<f32>(centeredPos, 1.0);
  let wn = normalize((sh.localToWorld * vec4<f32>(normal, 0.0)).xyz);

  var out: ShardCausticOut;
  out.clipPos = uScene.sunViewProj * worldPos;
  out.parentLocalPos = parentLocalPos;
  out.centeredPos = centeredPos;
  out.worldNormal = wn;
  out.worldY = worldPos.y;
  out.isOuter = isOuter;
  out.shardIdx = shardIdx;
  return out;
}

@fragment
fn fs_shard_caustic(input: ShardCausticOut) -> @location(0) vec4<f32> {
  let sh = uShards[input.shardIdx];
  let motifId = i32(round(sh.absorptionAndMotif.w));
  let seed = sh.accentAndSeed.w;
  let L = normalize(uScene.sunDirAndBlockCount.xyz);

  // Compute exact 3D sun-ray chord length through this convex Voronoi shard!
  let r0 = sh.localToWorld[0].xyz;
  let r1 = sh.localToWorld[1].xyz;
  let r2 = sh.localToWorld[2].xyz;
  let sunDirLocal = normalize(vec3<f32>(dot(r0, -L), dot(r1, -L), dot(r2, -L)));

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
  let chordDm = clamp(tExit, 0.012, 0.35);

  let vitrail = evaluateVitrailPattern(input.parentLocalPos, motifId, seed);
  let sigma = mix(sh.absorptionAndMotif.xyz, sh.accentAndSeed.xyz, vitrail.accentMix);
  let glassTrans = exp(-sigma * chordDm * 0.92);

  // Grazing cleavage faces reflect/refract sunlight away -> dark refractive rim framing luminous caustic center!
  let cosSun = clamp(abs(dot(normalize(input.worldNormal), L)), 0.0, 1.0);
  let fresnelPass = mix(0.30, 1.0, smoothstep(0.06, 0.55, cosSun));
  let focus = clamp(0.92 + vitrail.rippleHeight * 0.42 + (1.0 - smoothstep(0.02, 0.14, chordDm)) * 0.18, 0.72, 1.45);

  // Grounding contact shadow where shard rests directly on the oak tabletop
  let tableContact = mix(0.82, 1.0, smoothstep(0.008, 0.055, input.worldY));

  var trans = glassTrans * focus * fresnelPass * tableContact;
  let outerCame = vitrail.cameMask * input.isOuter;
  trans = mix(trans, vec3<f32>(0.20, 0.18, 0.17), outerCame * 0.88);
  return vec4<f32>(trans, 1.0);
}
`;

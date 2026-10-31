import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 4: Refractive 3D Voronoi Shattered Glass Shards & Glittering Micro-Splinters.
 * Samples `uBackdropTex` (which already contains the caustic-lit oak table AND the intact glass tower!)
 * with screen-space chromatic dispersion, Beer-Lambert tinting, outer copper-foil came preservation,
 * and razor-sharp glossy internal conchoidal fracture cleavage highlights.
 */
export const GLASS_SHARD_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

struct GPUShardInstance {
  localToWorld: mat4x4<f32>,
  absorptionAndMotif: vec4<f32>,
  accentAndSeed: vec4<f32>,
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
  @location(3) isOuter: f32,
  @location(4) @interpolate(flat) shardIdx: u32,
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
  out.isOuter = isOuter;
  out.shardIdx = shardIdx;
  return out;
}

@fragment
fn fs_glass_shard(input: ShardVSOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(uBackdropTex));
  let screenUV = input.clipPos.xy / max(dims, vec2<f32>(1.0));

  let sh = uShards[input.shardIdx];
  let motifId = i32(round(sh.absorptionAndMotif.w));
  let seed = sh.accentAndSeed.w;

  let V = normalize(uScene.cameraPosAndTime.xyz - input.worldPos);
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let vitrail = evaluateVitrailPattern(input.parentLocalPos, motifId, seed);
  let geomN = select(-normalize(input.worldNormal), normalize(input.worldNormal), isFront);

  // Exterior faces retain cathedral waterglass ripples; fresh internal cleavage faces have razor-sharp conchoidal ripples!
  let conchoidalWave = sin(dot(input.parentLocalPos, vec3<f32>(48.0, 62.0, 54.0)) + seed * 9.0) * 0.0022;
  let rippleWorld = (sh.localToWorld * vec4<f32>(vitrail.rippleGrad, 0.0)).xyz;
  let N = normalize(geomN + select(vec3<f32>(conchoidalWave), rippleWorld * 1.4, input.isOuter > 0.5));

  // Screen-space dielectric refraction with RGB chromatic prism dispersion!
  let refractedWorld = refract(-V, N, 1.0 / 1.522);
  let offsetClip = (uScene.viewProj * vec4<f32>(input.worldPos + refractedWorld * 0.14, 1.0));
  let refractedNDC = offsetClip.xy / max(offsetClip.w, 1e-5);
  let baseRefractUV = vec2<f32>(refractedNDC.x * 0.5 + 0.5, 0.5 - refractedNDC.y * 0.5);
  let deltaUV = clamp(baseRefractUV - screenUV, vec2<f32>(-0.08), vec2<f32>(0.08));

  let uvR = clamp(screenUV + deltaUV * 0.90, vec2<f32>(0.002), vec2<f32>(0.998));
  let uvG = clamp(screenUV + deltaUV * 1.00, vec2<f32>(0.002), vec2<f32>(0.998));
  let uvB = clamp(screenUV + deltaUV * 1.12, vec2<f32>(0.002), vec2<f32>(0.998));

  let bgR = textureSampleLevel(uBackdropTex, uLinearSampler, uvR, 0.0).r;
  let bgG = textureSampleLevel(uBackdropTex, uLinearSampler, uvG, 0.0).g;
  let bgB = textureSampleLevel(uBackdropTex, uLinearSampler, uvB, 0.0).b;
  let bgRGB = vec3<f32>(bgR, bgG, bgB);

  let cosTheta = clamp(dot(N, V), 0.0, 1.0);
  let fresnel = 0.045 + 0.955 * pow(1.0 - cosTheta, 5.0);

  // Beer-Lambert glass absorption across shard thickness (~0.09 dm average)
  let sigma = mix(sh.absorptionAndMotif.xyz, sh.accentAndSeed.xyz, vitrail.accentMix);
  let pathLen = 0.075 + (1.0 - cosTheta) * 0.085;
  let transmittance = exp(-sigma * pathLen);

  let H = normalize(L + V);
  // Fresh internal glass fracture facets have ultra-crisp diamond-like specular glints!
  let specExp = select(260.0, 160.0, input.isOuter > 0.5);
  let sunSpec = pow(max(0.0, dot(N, H)), specExp) * sunColor * select(3.4, 2.4, input.isOuter > 0.5);
  let envRefl = sampleCathedralEnvironment(reflect(-V, N), L, sunColor, skyTint);

  // Bright refractive edge gleam on sharp cleavage facets
  let cleavageRim = pow(1.0 - cosTheta, 3.0) * (1.0 - input.isOuter) * 0.32 * (sunColor * 0.4 + skyTint);

  var shardRGB = mix(bgRGB * transmittance, envRefl, fresnel) + sunSpec + cleavageRim;

  // Original exterior faces preserve their soldered copper-foil / pewter came border!
  let outerCame = vitrail.cameMask * input.isOuter;
  if (outerCame > 0.05) {
    let nDotL = max(0.0, dot(geomN, L));
    let solderCol = vec3<f32>(0.14, 0.11, 0.10) * (sunColor * nDotL * 0.55 + skyTint * 0.35 + vec3<f32>(0.06))
                  + vec3<f32>(0.42, 0.29, 0.20) * pow(max(0.0, dot(geomN, H)), 26.0) * sunColor * 0.4;
    shardRGB = mix(shardRGB, solderCol, outerCame);
  }

  return vec4<f32>(shardRGB, 1.0);
}

// Glittering Micro-Splinter & Sunlit Cathedral Dust Mote Instanced Pipeline
struct GPUSplinter {
  posAndScale: vec4<f32>,
  colorAndRot: vec4<f32>, // rgb = glass jewel tint, w = rot angle (or < 0 for soft cathedral dust mote!)
};

@group(0) @binding(4) var<storage, read> uSplinters: array<GPUSplinter>;

struct SplinterOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) isDust: f32,
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
  let isDust = select(0.0, 1.0, sp.colorAndRot.w < -50.0);
  let ang = sp.colorAndRot.w;
  let ca = cos(ang);
  let sa = sin(ang);
  let rq = vec2<f32>(q.x * ca - q.y * sa, q.x * sa + q.y * ca);

  // Camera-facing billboard offset
  let toCam = normalize(uScene.cameraPosAndTime.xyz - sp.posAndScale.xyz);
  let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), toCam));
  let up = cross(toCam, right);

  let scale = sp.posAndScale.w;
  let wp = sp.posAndScale.xyz + (right * rq.x * 1.4 + up * rq.y * 0.65) * scale;

  var out: SplinterOut;
  out.clipPos = uScene.viewProj * vec4<f32>(wp, 1.0);
  out.uv = q;
  out.color = sp.colorAndRot.xyz;
  out.isDust = isDust;
  return out;
}

@fragment
fn fs_splinter(input: SplinterOut) -> @location(0) vec4<f32> {
  if (input.isDust > 0.5) {
    let r = length(input.uv);
    let soft = pow(max(0.0, 1.0 - r), 2.5) * 0.32;
    return vec4<f32>(input.color * soft, soft);
  }
  // Sharp diamond glass splinter glint!
  let diamond = abs(input.uv.x) * 0.65 + abs(input.uv.y) * 1.25;
  if (diamond > 1.0) {
    discard;
  }
  let core = pow(max(0.0, 1.0 - diamond), 1.6);
  let sparkle = input.color * (0.8 + core * 2.8);
  return vec4<f32>(sparkle * core, core * 0.9);
}
`;

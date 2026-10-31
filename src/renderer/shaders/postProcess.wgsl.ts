/**
 * Pass 5: HDR Optical Bloom + ACES Filmic Tone Mapping + sRGB Conversion (`postProcess.wgsl.ts`).
 */
export const POST_PROCESS_WGSL = /* wgsl */ `
struct PostUniforms {
  texelSizeAndBloom: vec4<f32>, // xy = 1/bloomSize, z = bloomStrength, w = exposure
};

@group(0) @binding(0) var<uniform> uPost: PostUniforms;
@group(0) @binding(1) var uInputTex: texture_2d<f32>;
@group(0) @binding(2) var uBloomTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;

struct FullscreenOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FullscreenOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = pos[vid];
  var out: FullscreenOut;
  out.clipPos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

// 1. Soft HDR Bright-Pass Downsample for Bevel Glints & Concentrated Stained-Glass Caustics
@fragment
fn fs_bloom_extract(input: FullscreenOut) -> @location(0) vec4<f32> {
  let ts = uPost.texelSizeAndBloom.xy;
  let c0 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-ts.x, -ts.y), 0.0).rgb;
  let c1 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( ts.x, -ts.y), 0.0).rgb;
  let c2 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-ts.x,  ts.y), 0.0).rgb;
  let c3 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( ts.x,  ts.y), 0.0).rgb;
  let avg = (c0 + c1 + c2 + c3) * 0.25;

  // Reference uBloomTex so auto/shared layout keeps binding active if shared, or separate bind group
  let _unused = textureDimensions(uBloomTex).x;
  let lum = dot(avg, vec3<f32>(0.2126, 0.7152, 0.0722)) + f32(_unused) * 0.0;
  let knee = smoothstep(1.05, 2.6, lum);
  return vec4<f32>(min(avg * knee, vec3<f32>(12.0)), 1.0);
}

// 2. Separable 9-tap Gaussian Blur (direction passed in uPost.texelSizeAndBloom.xy)
@fragment
fn fs_bloom_blur(input: FullscreenOut) -> @location(0) vec4<f32> {
  let stepUV = uPost.texelSizeAndBloom.xy;
  let _unused = textureDimensions(uBloomTex).x;
  var acc = textureSampleLevel(uInputTex, uLinearSampler, input.uv, 0.0).rgb * 0.227027 + f32(_unused) * 0.0;

  let o1 = stepUV * 1.384615;
  let o2 = stepUV * 3.230769;

  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv + o1, 0.0).rgb * 0.316216;
  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv - o1, 0.0).rgb * 0.316216;
  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv + o2, 0.0).rgb * 0.070270;
  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv - o2, 0.0).rgb * 0.070270;

  return vec4<f32>(acc, 1.0);
}

fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// 3. Final Composite: Scene HDR + Optical Bloom -> ACES Tone Map -> sRGB
@fragment
fn fs_composite(input: FullscreenOut) -> @location(0) vec4<f32> {
  let hdr = textureSampleLevel(uInputTex, uLinearSampler, input.uv, 0.0).rgb;
  let bloom = textureSampleLevel(uBloomTex, uLinearSampler, input.uv, 0.0).rgb;

  let bloomStrength = uPost.texelSizeAndBloom.z;
  let exposure = uPost.texelSizeAndBloom.w;

  var combined = (hdr + bloom * bloomStrength) * exposure;

  // Subtle optical camera lens vignette
  let centered = input.uv - vec2<f32>(0.5);
  let vig = 1.0 - dot(centered, centered) * 0.36;
  combined *= clamp(vig, 0.72, 1.0);

  let mapped = acesFilmic(combined);
  // Exact linear-to-sRGB gamma encoding for standard bgra8unorm/rgba8unorm swapchain
  let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(srgb, 1.0);
}
`;

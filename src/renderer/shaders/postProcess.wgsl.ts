/**
 * Pass 5: Karis Anti-Firefly Multi-Mip Optical Bloom + Physical Macro Depth-of-Field +
 * Chroma-Preserving Filmic Tone Mapping + 8-bit Anti-Banding Dither (`postProcess.wgsl.ts`).
 */
export const POST_PROCESS_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265359;

struct PostUniforms {
  texelSizeAndBloom: vec4<f32>, // xy = stepUV (1/size), z = bloomStrength, w = exposure
  dofParams: vec4<f32>,         // x = focusDistDm, y = sharpDeadbandDm, z = maxCoCPx, w = aspect (w/h)
};

@group(0) @binding(0) var<uniform> uPost: PostUniforms;
@group(0) @binding(1) var uInputTex: texture_2d<f32>;
@group(0) @binding(2) var uBloomFine: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;
@group(0) @binding(4) var uBloomWide: texture_2d<f32>;

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

fn luma709(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// 1. Karis-Weighted Anti-Firefly HDR Bright-Pass Extract (suppresses isolated 1px specular cotton-dots!)
@fragment
fn fs_bloom_extract(input: FullscreenOut) -> @location(0) vec4<f32> {
  let ts = uPost.texelSizeAndBloom.xy;

  let cCenter = min(textureSampleLevel(uInputTex, uLinearSampler, input.uv, 0.0).rgb, vec3<f32>(16.0));
  let c0 = min(textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-ts.x, -ts.y), 0.0).rgb, vec3<f32>(16.0));
  let c1 = min(textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( ts.x, -ts.y), 0.0).rgb, vec3<f32>(16.0));
  let c2 = min(textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-ts.x,  ts.y), 0.0).rgb, vec3<f32>(16.0));
  let c3 = min(textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( ts.x,  ts.y), 0.0).rgb, vec3<f32>(16.0));

  let wC = 1.0 / (1.0 + luma709(cCenter));
  let w0 = 1.0 / (1.0 + luma709(c0));
  let w1 = 1.0 / (1.0 + luma709(c1));
  let w2 = 1.0 / (1.0 + luma709(c2));
  let w3 = 1.0 / (1.0 + luma709(c3));

  let karisAvg = (cCenter * wC * 2.0 + c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3)
               / max(1e-4, wC * 2.0 + w0 + w1 + w2 + w3);

  let lum = luma709(karisAvg);
  let knee = smoothstep(1.15, 2.95, lum);
  return vec4<f32>(karisAvg * knee, 1.0);
}

// 2. Clean 5-Tap Tent Downsample for Wide Halation Mip
@fragment
fn fs_bloom_downsample(input: FullscreenOut) -> @location(0) vec4<f32> {
  let ts = uPost.texelSizeAndBloom.xy;
  let c = textureSampleLevel(uInputTex, uLinearSampler, input.uv, 0.0).rgb * 0.50;
  let n0 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-ts.x, -ts.y), 0.0).rgb * 0.125;
  let n1 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( ts.x, -ts.y), 0.0).rgb * 0.125;
  let n2 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-ts.x,  ts.y), 0.0).rgb * 0.125;
  let n3 = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( ts.x,  ts.y), 0.0).rgb * 0.125;
  return vec4<f32>(c + n0 + n1 + n2 + n3, 1.0);
}

// 3. Separable 9-tap Gaussian Blur
@fragment
fn fs_bloom_blur(input: FullscreenOut) -> @location(0) vec4<f32> {
  let stepUV = uPost.texelSizeAndBloom.xy;
  var acc = textureSampleLevel(uInputTex, uLinearSampler, input.uv, 0.0).rgb * 0.227027;

  let o1 = stepUV * 1.384615;
  let o2 = stepUV * 3.230769;

  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv + o1, 0.0).rgb * 0.316216;
  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv - o1, 0.0).rgb * 0.316216;
  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv + o2, 0.0).rgb * 0.070270;
  acc += textureSampleLevel(uInputTex, uLinearSampler, input.uv - o2, 0.0).rgb * 0.070270;

  return vec4<f32>(acc, 1.0);
}

fn acesFilmicChromaSafe(xIn: vec3<f32>) -> vec3<f32> {
  // Subtle AgX-inspired pre-crosstalk prevents saturated ruby/cobalt highlights from skewing to neon hues
  let lum = luma709(xIn);
  let x = mix(xIn, vec3<f32>(lum), clamp(lum * 0.035, 0.0, 0.14));
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn interleavedGradientNoise(px: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(px, vec2<f32>(0.06711056, 0.00583715))));
}

// 4. Final Composite: Depth-Aware Macro Lens Bokeh + Dual-Scale Optical Bloom -> Filmic Tone Map -> Dithered sRGB
@fragment
fn fs_composite(input: FullscreenOut) -> @location(0) vec4<f32> {
  let centerSample = textureSampleLevel(uInputTex, uLinearSampler, input.uv, 0.0);
  var hdr = centerSample.rgb;
  let centerDist = centerSample.a;

  let focusDist = uPost.dofParams.x;
  let deadband = uPost.dofParams.y;
  let maxRadiusPx = uPost.dofParams.z;
  let aspect = max(0.2, uPost.dofParams.w);

  // Circle of Confusion (0.0 across entire glass tower & foreground shards; opens smoothly on far table rim & backdrop)
  let farCoC = smoothstep(deadband, deadband + 3.5, centerDist - focusDist);
  let nearCoC = smoothstep(deadband + 0.6, deadband + 2.2, focusDist - centerDist) * 0.45;
  let coc = clamp(max(farCoC, nearCoC), 0.0, 1.0);

  if (coc > 0.015) {
    let dims = vec2<f32>(textureDimensions(uInputTex));
    let invDims = 1.0 / max(dims, vec2<f32>(1.0));
    let radiusUV = vec2<f32>(invDims.x, invDims.y) * (coc * maxRadiusPx);
    let jitter = interleavedGradientNoise(input.clipPos.xy) * 2.0 * PI;
    let cj = cos(jitter);
    let sj = sin(jitter);

    var bokehAcc = hdr;
    var wSum: f32 = 1.0;

    // 12-tap Golden-Angle Vogel aperture disc with foreground depth-bleed rejection!
    for (var i: i32 = 0; i < 12; i = i + 1) {
      let fi = f32(i) + 0.5;
      let r = sqrt(fi / 12.0);
      let ang = fi * 2.3999632;
      let dLocal = vec2<f32>(cos(ang), sin(ang)) * r;
      let dRot = vec2<f32>(dLocal.x * cj - dLocal.y * sj, dLocal.x * sj + dLocal.y * cj);
      let tapUV = clamp(input.uv + dRot * radiusUV + vec2<f32>(0.0, aspect * 0.0), vec2<f32>(0.001), vec2<f32>(0.999));

      let tap = textureSampleLevel(uInputTex, uLinearSampler, tapUV, 0.0);
      let tapFarCoC = smoothstep(deadband * 0.85, deadband + 3.2, tap.a - focusDist);
      let tapNearCoC = smoothstep(deadband + 0.5, deadband + 2.0, focusDist - tap.a) * 0.45;
      let tapCoC = max(tapFarCoC, tapNearCoC);
      // Reject sharp foreground tower pixels from bleeding onto out-of-focus background!
      let w = clamp(tapCoC * 1.35, 0.02, 1.0);
      bokehAcc += tap.rgb * w;
      wSum += w;
    }
    hdr = bokehAcc / wSum;
  } else {
    // Sub-pixel depth-silhouette & bevel-luma tangent anti-aliasing across sharp foreground tower & shards
    let dims = vec2<f32>(textureDimensions(uInputTex));
    let invDims = 1.0 / max(dims, vec2<f32>(1.0));
    let sN = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(0.0, -invDims.y), 0.0);
    let sS = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(0.0,  invDims.y), 0.0);
    let sW = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-invDims.x, 0.0), 0.0);
    let sE = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( invDims.x, 0.0), 0.0);

    let lC = luma709(min(hdr, vec3<f32>(8.0)));
    let lN = luma709(min(sN.rgb, vec3<f32>(8.0)));
    let lS = luma709(min(sS.rgb, vec3<f32>(8.0)));
    let lW = luma709(min(sW.rgb, vec3<f32>(8.0)));
    let lE = luma709(min(sE.rgb, vec3<f32>(8.0)));

    let lMin = min(lC, min(min(lN, lS), min(lW, lE)));
    let lMax = max(lC, max(max(lN, lS), max(lW, lE)));
    let lRange = lMax - lMin;
    let dJump = max(max(abs(sN.a - sS.a), abs(sE.a - sW.a)), max(abs(sN.a - centerDist), abs(sE.a - centerDist)));

    if (dJump > 0.016 || lRange > max(0.028, lMax * 0.085)) {
      let sNW = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-invDims.x, -invDims.y), 0.0);
      let sNE = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( invDims.x, -invDims.y), 0.0);
      let sSW = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>(-invDims.x,  invDims.y), 0.0);
      let sSE = textureSampleLevel(uInputTex, uLinearSampler, input.uv + vec2<f32>( invDims.x,  invDims.y), 0.0);

      let lNW = luma709(min(sNW.rgb, vec3<f32>(8.0)));
      let lNE = luma709(min(sNE.rgb, vec3<f32>(8.0)));
      let lSW = luma709(min(sSW.rgb, vec3<f32>(8.0)));
      let lSE = luma709(min(sSE.rgb, vec3<f32>(8.0)));

      let lumaGrad = vec2<f32>(
        (lNE + 2.0 * lE + lSE) - (lNW + 2.0 * lW + lSW),
        (lSW + 2.0 * lS + lSE) - (lNW + 2.0 * lN + lNE)
      );
      let depthGrad = vec2<f32>(
        (sNE.a + 2.0 * sE.a + sSE.a) - (sNW.a + 2.0 * sW.a + sSW.a),
        (sSW.a + 2.0 * sS.a + sSE.a) - (sNW.a + 2.0 * sN.a + sNE.a)
      );
      // Align depth gradient polarity with luma gradient so bright-near over dark-far edges reinforce rather than cancel
      let depthSign = select(1.0, -1.0, dot(lumaGrad, depthGrad) < 0.0);
      let edgeNorm = lumaGrad + clamp(depthGrad * depthSign * 0.65, vec2<f32>(-1.6), vec2<f32>(1.6));
      let tang = normalize(vec2<f32>(-edgeNorm.y, edgeNorm.x) + vec2<f32>(1e-5, 0.0));

      let step1 = tang * invDims * 0.78;
      let step2 = tang * invDims * 1.85;
      let tA1 = textureSampleLevel(uInputTex, uLinearSampler, clamp(input.uv + step1, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);
      let tB1 = textureSampleLevel(uInputTex, uLinearSampler, clamp(input.uv - step1, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);
      let tA2 = textureSampleLevel(uInputTex, uLinearSampler, clamp(input.uv + step2, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);
      let tB2 = textureSampleLevel(uInputTex, uLinearSampler, clamp(input.uv - step2, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);

      let lMean = 0.25 * (lN + lS + lW + lE);
      let invSpan = 1.0 / max(lRange, 0.04);
      let wA2 = 0.42 * (1.0 - smoothstep(0.18, 0.65, abs(luma709(min(tA2.rgb, vec3<f32>(8.0))) - lMean) * invSpan));
      let wB2 = 0.42 * (1.0 - smoothstep(0.18, 0.65, abs(luma709(min(tB2.rgb, vec3<f32>(8.0))) - lMean) * invSpan));
      let spanAvg = (tA1.rgb + tB1.rgb + tA2.rgb * wA2 + tB2.rgb * wB2) / (2.0 + wA2 + wB2);
      let blendAmt = select(0.34, 0.46, dJump > 0.016);
      hdr = mix(hdr, spanAvg, blendAmt);
    }
  }

  let bloomFine = textureSampleLevel(uBloomFine, uLinearSampler, input.uv, 0.0).rgb;
  let bloomWide = textureSampleLevel(uBloomWide, uLinearSampler, input.uv, 0.0).rgb;
  let bloom = bloomFine * 0.58 + bloomWide * 0.52;

  let bloomStrength = uPost.texelSizeAndBloom.z;
  let exposure = uPost.texelSizeAndBloom.w;

  var combined = (hdr + bloom * bloomStrength) * exposure;

  // Subtle natural optical lens falloff
  let centered = input.uv - vec2<f32>(0.5);
  let vig = 1.0 - dot(centered, centered) * 0.28;
  combined *= clamp(vig, 0.78, 1.0);

  let mapped = acesFilmicChromaSafe(combined);
  let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));

  // Sub-LSB interleaved gradient dither eliminates 8-bit Mach banding across dark limestone/walnut gradients
  let dither = (interleavedGradientNoise(input.clipPos.xy + vec2<f32>(7.0, 3.0)) - 0.5) / 255.0;
  return vec4<f32>(clamp(srgb + vec3<f32>(dither), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
`;

import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 2: Low-Key Architectural Studio Backdrop + Quarter-Sawn Dark Walnut Bullnose Table
 * with Contact-Hardening Stained-Glass Caustics (PCSS), Window Gobo Light Shaft,
 * Conductor Brushed-Brass Inlays, and Exact Block + Voronoi Shard Contact Ambient Occlusion.
 */
export const ENVIRONMENT_PASS_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uBlocks: array<GPUBlockData>;
@group(0) @binding(2) var uCausticTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;
@group(0) @binding(4) var<storage, read> uShards: array<GPUShardInstance>;
@group(0) @binding(5) var uSunDepthTex: texture_depth_2d;
@group(0) @binding(6) var uSunOpticalDepthTex: texture_2d<f32>;

// 1. Full-screen Low-Key Studio Alcove Backdrop (depth = 1.0, depthWrite = false)
struct SkyOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = pos[vid];
  var out: SkyOut;
  out.clipPos = vec4<f32>(p, 0.9999, 1.0);
  out.ndc = p;
  return out;
}

@fragment
fn fs_sky(input: SkyOut) -> @location(0) vec4<f32> {
  if (uScene.cameraPosAndTime.w < -0.01) {
    return vec4<f32>(vec3<f32>(-uScene.cameraPosAndTime.w), 14.0);
  }
  let farWorld = uScene.invViewProj * vec4<f32>(input.ndc, 1.0, 1.0);
  let worldPos = farWorld.xyz / max(farWorld.w, 1e-6);
  let rayDir = safeNormalize(worldPos - uScene.cameraPosAndTime.xyz, vec3<f32>(0.0, 0.0, -1.0));
  let col = sampleStudioBackdrop(
    rayDir,
    safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0)),
    uScene.sunColorAndAspect.xyz,
    uScene.skyTintAndMode.xyz
  );
  // Alpha stores linear camera distance (14.0 dm = distant studio background for macro DoF)
  return vec4<f32>(col, 14.0);
}

// 2. Procedural 3D Round Dark Walnut Pedestal Table with Bullnose Rim, Recessed Soffit & Turned Pedestal (192 slices * 33 verts)
struct TableOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
};

const TABLE_SEGMENTS: u32 = 192u;
const TABLE_VERTS_PER_SLICE: u32 = 33u;
const TABLE_R_FLAT: f32 = 4.24;
const TABLE_R_OUTER: f32 = 4.30;
const TABLE_BULLNOSE_Y: f32 = -0.048;
const TABLE_H: f32 = 0.34;
const TABLE_R_SOFFIT: f32 = 3.78;
const TABLE_SOFFIT_Y: f32 = -0.54;
const TABLE_R_COLLAR: f32 = 1.44;
const TABLE_COLLAR_Y: f32 = -0.72;
const TABLE_R_PED_BASE: f32 = 1.18;
const TABLE_PED_BASE_Y: f32 = -4.20;

fn evalTableVertex(vid: u32) -> TableOut {
  // Each radial slice emits 11 triangles (33 vertices):
  // Tri 0 (0..2): Flat top disc (0 -> rFlat)
  // Tri 1 & 2 (3..8): Smoothly curved bullnose upper chamfer (rFlat, 0 -> rOuter, -0.048)
  // Tri 3 & 4 (9..14): Turned hardwood apron side band (rOuter, -0.048 -> rOuter*0.982, -TABLE_H)
  // Tri 5 & 6 (15..20): Recessed cove under-bevel (rOuter*0.982, -TABLE_H -> rSoffit, -0.54)
  // Tri 7 & 8 (21..26): Dark walnut sub-frame soffit (rSoffit, -0.54 -> rCollar, -0.72)
  // Tri 9 & 10 (27..32): Turned central pedestal column (rCollar, -0.72 -> rPedBase, -4.20)
  let slice = vid / TABLE_VERTS_PER_SLICE;
  let sub = vid % TABLE_VERTS_PER_SLICE;

  let a0 = (f32(slice) / f32(TABLE_SEGMENTS)) * 2.0 * PI;
  let a1 = (f32(slice + 1u) / f32(TABLE_SEGMENTS)) * 2.0 * PI;

  let dir0 = vec2<f32>(cos(a0), sin(a0));
  let dir1 = vec2<f32>(cos(a1), sin(a1));

  let nTop = vec3<f32>(0.0, 1.0, 0.0);
  let nBevelTop0 = normalize(vec3<f32>(dir0.x * 0.18, 1.0, dir0.y * 0.18));
  let nBevelTop1 = normalize(vec3<f32>(dir1.x * 0.18, 1.0, dir1.y * 0.18));
  let nRim0 = normalize(vec3<f32>(dir0.x, 0.38, dir0.y));
  let nRim1 = normalize(vec3<f32>(dir1.x, 0.38, dir1.y));
  let nApron0 = normalize(vec3<f32>(dir0.x, -0.08, dir0.y));
  let nApron1 = normalize(vec3<f32>(dir1.x, -0.08, dir1.y));
  let nCove0 = normalize(vec3<f32>(dir0.x * 0.42, -0.91, dir0.y * 0.42));
  let nCove1 = normalize(vec3<f32>(dir1.x * 0.42, -0.91, dir1.y * 0.42));
  let nSoffit0 = normalize(vec3<f32>(dir0.x * 0.08, -0.99, dir0.y * 0.08));
  let nSoffit1 = normalize(vec3<f32>(dir1.x * 0.08, -0.99, dir1.y * 0.08));
  let nPed0 = normalize(vec3<f32>(dir0.x, -0.07, dir0.y));
  let nPed1 = normalize(vec3<f32>(dir1.x, -0.07, dir1.y));

  let pCenter = vec3<f32>(0.0, 0.0, 0.0);
  let pFlat0 = vec3<f32>(dir0.x * TABLE_R_FLAT, 0.0, dir0.y * TABLE_R_FLAT);
  let pFlat1 = vec3<f32>(dir1.x * TABLE_R_FLAT, 0.0, dir1.y * TABLE_R_FLAT);
  let pMid0 = vec3<f32>(dir0.x * TABLE_R_OUTER, TABLE_BULLNOSE_Y, dir0.y * TABLE_R_OUTER);
  let pMid1 = vec3<f32>(dir1.x * TABLE_R_OUTER, TABLE_BULLNOSE_Y, dir1.y * TABLE_R_OUTER);
  let pBot0 = vec3<f32>(dir0.x * (TABLE_R_OUTER * 0.982), -TABLE_H, dir0.y * (TABLE_R_OUTER * 0.982));
  let pBot1 = vec3<f32>(dir1.x * (TABLE_R_OUTER * 0.982), -TABLE_H, dir1.y * (TABLE_R_OUTER * 0.982));
  let pSoffit0 = vec3<f32>(dir0.x * TABLE_R_SOFFIT, TABLE_SOFFIT_Y, dir0.y * TABLE_R_SOFFIT);
  let pSoffit1 = vec3<f32>(dir1.x * TABLE_R_SOFFIT, TABLE_SOFFIT_Y, dir1.y * TABLE_R_SOFFIT);
  let pCollar0 = vec3<f32>(dir0.x * TABLE_R_COLLAR, TABLE_COLLAR_Y, dir0.y * TABLE_R_COLLAR);
  let pCollar1 = vec3<f32>(dir1.x * TABLE_R_COLLAR, TABLE_COLLAR_Y, dir1.y * TABLE_R_COLLAR);
  let pPed0 = vec3<f32>(dir0.x * TABLE_R_PED_BASE, TABLE_PED_BASE_Y, dir0.y * TABLE_R_PED_BASE);
  let pPed1 = vec3<f32>(dir1.x * TABLE_R_PED_BASE, TABLE_PED_BASE_Y, dir1.y * TABLE_R_PED_BASE);

  var wp = pCenter;
  var wn = nTop;

  switch (sub) {
    case 0u: { wp = pCenter;  wn = nTop; }
    case 1u: { wp = pFlat1;   wn = nTop; }
    case 2u: { wp = pFlat0;   wn = nTop; }

    case 3u: { wp = pFlat0;   wn = nBevelTop0; }
    case 4u: { wp = pFlat1;   wn = nBevelTop1; }
    case 5u: { wp = pMid0;    wn = nRim0; }

    case 6u: { wp = pFlat1;   wn = nBevelTop1; }
    case 7u: { wp = pMid1;    wn = nRim1; }
    case 8u: { wp = pMid0;    wn = nRim0; }

    case 9u:  { wp = pMid0;   wn = nRim0; }
    case 10u: { wp = pMid1;   wn = nRim1; }
    case 11u: { wp = pBot0;   wn = nApron0; }

    case 12u: { wp = pMid1;   wn = nRim1; }
    case 13u: { wp = pBot1;   wn = nApron1; }
    case 14u: { wp = pBot0;   wn = nApron0; }

    case 15u: { wp = pBot0;    wn = nCove0; }
    case 16u: { wp = pBot1;    wn = nCove1; }
    case 17u: { wp = pSoffit0; wn = nCove0; }

    case 18u: { wp = pBot1;    wn = nCove1; }
    case 19u: { wp = pSoffit1; wn = nCove1; }
    case 20u: { wp = pSoffit0; wn = nCove0; }

    case 21u: { wp = pSoffit0; wn = nSoffit0; }
    case 22u: { wp = pSoffit1; wn = nSoffit1; }
    case 23u: { wp = pCollar0; wn = nSoffit0; }

    case 24u: { wp = pSoffit1; wn = nSoffit1; }
    case 25u: { wp = pCollar1; wn = nSoffit1; }
    case 26u: { wp = pCollar0; wn = nSoffit0; }

    case 27u: { wp = pCollar0; wn = nPed0; }
    case 28u: { wp = pCollar1; wn = nPed1; }
    case 29u: { wp = pPed0;    wn = nPed0; }

    case 30u: { wp = pCollar1; wn = nPed1; }
    case 31u: { wp = pPed1;    wn = nPed1; }
    default:  { wp = pPed0;    wn = nPed0; }
  }

  var out: TableOut;
  out.clipPos = uScene.viewProj * vec4<f32>(wp, 1.0);
  out.worldPos = wp;
  out.worldNormal = wn;
  return out;
}

@vertex
fn vs_table(@builtin(vertex_index) vid: u32) -> TableOut {
  return evalTableVertex(vid);
}

@vertex
fn vs_table_depth(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  return evalTableVertex(vid).clipPos;
}

fn evalTransmittedBaseFromTau(tauRGB: vec3<f32>) -> vec3<f32> {
  let tauPeak = max(tauRGB.r, max(tauRGB.g, tauRGB.b));
  let casterMask = smoothstep(0.005, 0.075, tauPeak);
  let shadowBase = 0.38 * exp(-tauRGB * 0.52) + 0.17 * exp(-tauRGB * 0.18);
  return mix(vec3<f32>(1.0), shadowBase, casterMask);
}

/**
 * Contact-Hardening Caustic & Shadow Sampler (PCSS):
 * Queries uSunOpticalDepthTex (.rgb = spectral tau_sun, .a = minCasterY) + additive Snell uCausticTex:
 * - Layer-0 resting blocks/shards (minCasterY ~ 0..1.4 cm): razor-sharp < 2.5 mm shadow edge & bevel arcs.
 * - Upper tower stories (Layer 12, minCasterY ~ 18 cm): > 12 mm smooth solar/window penumbra + RGB dispersion.
 */
fn sampleContactHardeningCaustics(worldPos: vec3<f32>, sunDir: vec3<f32>) -> vec3<f32> {
  let sunClip = uScene.sunViewProj * vec4<f32>(worldPos, 1.0);
  let ndc = sunClip.xyz / max(sunClip.w, 1e-6);
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

  if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998) {
    return vec3<f32>(1.0);
  }

  let optDims = vec2<i32>(textureDimensions(uSunOpticalDepthTex));
  let maxCoord = optDims - vec2<i32>(1);
  let centerPx = vec2<i32>(uv * vec2<f32>(optDims));

  // 9-tap min-height blocker search up to +/- 11 texels (+/- 10.4 mm) so Layer-0 stays < 2.5 mm even under upper-row overlap
  var outerOffsets = array<vec2<i32>, 5>(
    vec2<i32>(  0,   0),
    vec2<i32>( 11,   4),
    vec2<i32>(-11,  -4),
    vec2<i32>( -4,  11),
    vec2<i32>(  4, -11)
  );
  var minSearchY: f32 = 10.0;
  for (var s: i32 = 0; s < 5; s = s + 1) {
    let px = clamp(centerPx + outerOffsets[s], vec2<i32>(0), maxCoord);
    minSearchY = min(minSearchY, textureLoad(uSunOpticalDepthTex, px, 0).a);
  }
  if (minSearchY < 9.9) {
    var innerOffsets = array<vec2<i32>, 4>(
      vec2<i32>(  5,   2),
      vec2<i32>( -5,  -2),
      vec2<i32>( -2,   5),
      vec2<i32>(  2,  -5)
    );
    for (var s: i32 = 0; s < 4; s = s + 1) {
      let px = clamp(centerPx + innerOffsets[s], vec2<i32>(0), maxCoord);
      minSearchY = min(minSearchY, textureLoad(uSunOpticalDepthTex, px, 0).a);
    }
  }

  let texel = 1.0 / f32(optDims.x);
  let rotAngle = hash31(vec3<f32>(floor(worldPos.x * 400.0), 1.0, floor(worldPos.z * 400.0))) * 2.0 * PI;
  let ca = cos(rotAngle);
  let sa = sin(rotAngle);

  // 1. Contact-hardened un-deflected stained-glass shadow base from uSunOpticalDepthTex + 2048x2048 uSunDepthTex contact edge
  var baseFiltered = vec3<f32>(1.0);
  let effectiveHeightDm = select(0.08, clamp(minSearchY, 0.0, 4.5), minSearchY < 9.2);
  let liftedHeightDm = max(0.0, effectiveHeightDm - 0.045);
  var contactSharpWeight: f32 = 0.0;
  var sharpLit01: f32 = 1.0;
  if (minSearchY < 9.2) {
    let shadowRadiusUV = texel * (0.18 + pow(liftedHeightDm, 1.25) * 5.0);
    let centerTau = textureSampleLevel(uSunOpticalDepthTex, uLinearSampler, uv, 0.0).rgb;
    var baseAccum = evalTransmittedBaseFromTau(centerTau) * 0.20;
    var baseWeight: f32 = 0.20;

    for (var i: i32 = 0; i < 12; i = i + 1) {
      let fi = f32(i) + 0.5;
      let r = sqrt(fi / 12.0);
      let theta = fi * 2.3999632;
      let localOff = vec2<f32>(cos(theta), sin(theta)) * r;
      let rotOff = vec2<f32>(localOff.x * ca - localOff.y * sa, localOff.x * sa + localOff.y * ca) * shadowRadiusUV;
      let w = 1.0 - r * 0.25;
      let tapUV = clamp(uv + rotOff, vec2<f32>(0.002), vec2<f32>(0.998));
      let tapTau = textureSampleLevel(uSunOpticalDepthTex, uLinearSampler, tapUV, 0.0).rgb;
      baseAccum += evalTransmittedBaseFromTau(tapTau) * w;
      baseWeight += w;
    }
    baseFiltered = baseAccum / baseWeight;

    // Sub-millimeter (0.47 mm/texel) 2048x2048 depth-map contact silhouette for resting Layer-0 blocks/shards
    contactSharpWeight = 1.0 - smoothstep(0.015, 0.14, liftedHeightDm);
    if (contactSharpWeight > 0.01) {
      let depthDims = vec2<i32>(textureDimensions(uSunDepthTex));
      let depthMax = depthDims - vec2<i32>(1);
      let dPos = uv * vec2<f32>(depthDims) - vec2<f32>(0.5);
      let dBase = vec2<i32>(floor(dPos));
      let dFrac = fract(dPos);
      let refZ = ndc.z - 0.0009;
      let v00 = select(0.0, 1.0, refZ <= textureLoad(uSunDepthTex, clamp(dBase, vec2<i32>(0), depthMax), 0));
      let v10 = select(0.0, 1.0, refZ <= textureLoad(uSunDepthTex, clamp(dBase + vec2<i32>(1, 0), vec2<i32>(0), depthMax), 0));
      let v01 = select(0.0, 1.0, refZ <= textureLoad(uSunDepthTex, clamp(dBase + vec2<i32>(0, 1), vec2<i32>(0), depthMax), 0));
      let v11 = select(0.0, 1.0, refZ <= textureLoad(uSunDepthTex, clamp(dBase + vec2<i32>(1, 1), vec2<i32>(0), depthMax), 0));
      sharpLit01 = smoothstep(0.15, 0.85, mix(mix(v00, v10, dFrac.x), mix(v01, v11, dFrac.x), dFrac.y));

      var bestTau = centerTau;
      var bestPeak = max(bestTau.r, max(bestTau.g, bestTau.b));
      var nearOffs = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
      for (var n: i32 = 0; n < 4; n = n + 1) {
        let candTau = textureLoad(uSunOpticalDepthTex, clamp(centerPx + nearOffs[n], vec2<i32>(0), maxCoord), 0).rgb;
        let candPeak = max(candTau.r, max(candTau.g, candTau.b));
        if (candPeak > bestPeak) {
          bestPeak = candPeak;
          bestTau = candTau;
        }
      }
      let sharpContactShadow = mix(evalTransmittedBaseFromTau(bestTau), vec3<f32>(1.0), sharpLit01);
      baseFiltered = mix(baseFiltered, sharpContactShadow, contactSharpWeight);
    }
  }

  // Fast early-out for unobstructed sunlit tabletop pixels outside both shadow and caustic halo (~75% of table)
  let c0 = textureSampleLevel(uCausticTex, uLinearSampler, uv, 0.0).rgb;
  if (minSearchY >= 9.2) {
    let step4 = texel * 5.0;
    let cPx = textureSampleLevel(uCausticTex, uLinearSampler, clamp(uv + vec2<f32>(step4, 0.0), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).rgb;
    let cNx = textureSampleLevel(uCausticTex, uLinearSampler, clamp(uv - vec2<f32>(step4, 0.0), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).rgb;
    let cPy = textureSampleLevel(uCausticTex, uLinearSampler, clamp(uv + vec2<f32>(0.0, step4), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).rgb;
    let cNy = textureSampleLevel(uCausticTex, uLinearSampler, clamp(uv - vec2<f32>(0.0, step4), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).rgb;
    let maxProbe = max(
      max(c0.r, max(c0.g, c0.b)),
      max(
        max(max(cPx.r, cPx.g), max(cNx.r, cNx.g)),
        max(max(cPy.r, cPy.g), max(cNy.r, cNy.g))
      )
    );
    if (maxProbe < 1e-4) {
      return vec3<f32>(1.0);
    }
  }

  // 2. Additive Snell-refracted waterglass & vertical-corner focal caustic flux from uCausticTex
  //    Filter R/G/B dispersion jointly inside the Vogel disk kernel so single-texel fold spikes never split into RGB comb stripes.
  let sunClipDelta = uScene.sunViewProj * vec4<f32>(-sunDir.x, 0.0, -sunDir.z, 0.0);
  let sunProjDir = normalize(vec2<f32>(sunClipDelta.x, -sunClipDelta.y) + vec2<f32>(1e-5, 1e-5));
  let dispMag = texel * (0.04 + pow(liftedHeightDm, 1.10) * 0.66);
  let dispVec = sunProjDir * dispMag;

  let causticRadiusUV = texel * (0.18 + pow(liftedHeightDm, 1.15) * 1.14);
  let centerR = textureSampleLevel(uCausticTex, uLinearSampler, clamp(uv + dispVec, vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).r;
  let centerG = c0.g;
  let centerB = textureSampleLevel(uCausticTex, uLinearSampler, clamp(uv - dispVec, vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).b;
  var fluxAccum = vec3<f32>(centerR, centerG, centerB) * 0.22;
  var fluxWeight: f32 = 0.22;
  for (var i: i32 = 0; i < 12; i = i + 1) {
    let fi = f32(i) + 0.5;
    let r = sqrt(fi / 12.0);
    let theta = fi * 2.3999632;
    let localOff = vec2<f32>(cos(theta), sin(theta)) * r;
    let rotOff = vec2<f32>(localOff.x * ca - localOff.y * sa, localOff.x * sa + localOff.y * ca) * causticRadiusUV;
    let w = 1.0 - r * 0.32;
    let tapUV = clamp(uv + rotOff, vec2<f32>(0.002), vec2<f32>(0.998));
    let tapR = textureSampleLevel(uCausticTex, uLinearSampler, clamp(tapUV + dispVec, vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).r;
    let tapG = textureSampleLevel(uCausticTex, uLinearSampler, tapUV, 0.0).g;
    let tapB = textureSampleLevel(uCausticTex, uLinearSampler, clamp(tapUV - dispVec, vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).b;
    fluxAccum += vec3<f32>(tapR, tapG, tapB) * w;
    fluxWeight += w;
  }
  let dispersedFlux = (fluxAccum / fluxWeight) * (1.0 - sharpLit01 * contactSharpWeight);

  return baseFiltered + dispersedFlux;
}

@fragment
fn fs_table(input: TableOut) -> @location(0) vec4<f32> {
  // Unconditional screen-space derivative & binding keep-alive at top of fragment shader!
  let dXZ = fwidth(input.worldPos.xz);
  let aaWidth = max(dXZ.x, dXZ.y);
  let _keepShards = f32(arrayLength(&uShards)) * 0.0;
  let _keepDepth = f32(textureDimensions(uSunDepthTex).x) * 0.0;
  let _keepOpt = f32(textureDimensions(uSunOpticalDepthTex).x) * 0.0;

  let V = safeNormalize(uScene.cameraPosAndTime.xyz - input.worldPos, vec3<f32>(0.0, 1.0, 0.0));
  let L = safeNormalize(uScene.sunDirAndBlockCount.xyz, vec3<f32>(0.0, 1.0, 0.0));
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;
  let skyScale = clamp(dot(skyTint, vec3<f32>(0.3333333)) / 0.567, 0.0, 4.0);

  let isUnderStructure = input.worldPos.y < -0.342;
  var mat = evaluateWalnutTableMaterial(input.worldPos.xz, aaWidth);
  if (isUnderStructure) {
    let pedGrain = noise3D(vec3<f32>(input.worldPos.x * 9.0, input.worldPos.y * 22.0, input.worldPos.z * 9.0));
    mat.albedo = mix(vec3<f32>(0.028, 0.017, 0.011), vec3<f32>(0.056, 0.034, 0.021), pedGrain);
    mat.normalPerturb = vec3<f32>(0.0, (pedGrain - 0.5) * 0.02, 0.0);
    mat.roughness = 0.44;
    mat.metallic = 0.0;
  }
  let topMask = smoothstep(0.55, 0.95, input.worldNormal.y);
  let N = safeNormalize(input.worldNormal + mat.normalPerturb * mix(0.22, 1.0, topMask), vec3<f32>(0.0, 1.0, 0.0));

  var causticRGB = vec3<f32>(0.0);
  var windowGobo: f32 = 0.0;
  if (!isUnderStructure) {
    causticRGB = sampleContactHardeningCaustics(input.worldPos, L);
    windowGobo = evaluateStudioWindowGobo(input.worldPos, L);
  }

  // Dual-scale Contact Ambient Occlusion gated by tableAoBounds + Y-sorted storage pointers
  let blockCount = u32(uScene.sunDirAndBlockCount.w);
  let shardCount = u32(uScene.skyTintAndMode.w);
  let shardStride = max(1u, u32(uScene.tableAoCounts.y));
  var aoCrevice: f32 = 1.0;
  var aoBounce: f32 = 1.0;
  if (isUnderStructure) {
    let rad = length(input.worldPos.xz);
    let overhangOcc = mix(0.26, 0.68, smoothstep(1.35, 4.18, rad));
    let pedDepthFade = smoothstep(-4.0, -0.35, input.worldPos.y);
    aoBounce = overhangOcc * (0.45 + 0.55 * pedDepthFade);
  }

  let aoB = uScene.tableAoBounds;
  if (
    input.worldPos.y > -0.008 &&
    input.worldPos.x >= aoB.x &&
    input.worldPos.x <= aoB.z &&
    input.worldPos.z >= aoB.y &&
    input.worldPos.z <= aoB.w
  ) {
    // 1. Intact blocks sorted by Y ascending: break immediately once bCenter.y >= 0.52!
    for (var i: u32 = 0u; i < blockCount; i = i + 1u) {
      let pBlk = &uBlocks[i];
      let bCenter = (*pBlk).localToWorld[3].xyz;
      if (bCenter.y >= 0.52) {
        break;
      }
      let dxz = input.worldPos.xz - bCenter.xz;
      if (dot(dxz, dxz) < 0.22) {
        let w0 = (*pBlk).worldToLocal[0].xyz;
        let w1 = (*pBlk).worldToLocal[1].xyz;
        let w2 = (*pBlk).worldToLocal[2].xyz;
        let w3 = (*pBlk).worldToLocal[3].xyz;
        let pLocal = w0 * input.worldPos.x + w1 * input.worldPos.y + w2 * input.worldPos.z + w3;
        let q = abs(pLocal) - vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
        let boxDist = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        let hWeight = 1.0 - smoothstep(0.075, 0.44, bCenter.y);
        let cOcc = (1.0 - smoothstep(0.0, 0.022, boxDist)) * (1.0 - smoothstep(0.075, 0.16, bCenter.y));
        let bOcc = (1.0 - smoothstep(0.0, 0.088, boxDist)) * hWeight;
        aoCrevice *= (1.0 - cOcc * 0.78);
        aoBounce *= (1.0 - bOcc * 0.54);
        if (aoCrevice * aoBounce < 0.16) {
          break;
        }
      }
    }

    // 2. Low-lying Voronoi glass shards via compact radial contact footprint (0 plane-array DRAM reads)
    if (aoCrevice * aoBounce >= 0.16) {
      for (var s: u32 = 0u; s < shardCount; s = s + shardStride) {
        let pSh = &uShards[s];
        let sCenter = (*pSh).localToWorld[3].xyz;
        if (sCenter.y < 0.28) {
          let relW = input.worldPos - sCenter;
          let d2 = dot(relW.xz, relW.xz);
          if (d2 < 0.052) {
            let polyDist = sqrt(dot(relW, relW)) - 0.062;
            let sCrevice = (1.0 - smoothstep(-0.002, 0.018, polyDist)) * (1.0 - smoothstep(0.02, 0.12, sCenter.y));
            let sBounce = (1.0 - smoothstep(0.0, 0.060, polyDist)) * (1.0 - smoothstep(0.02, 0.28, sCenter.y));
            aoCrevice *= (1.0 - sCrevice * 0.74);
            aoBounce *= (1.0 - sBounce * 0.44);
            if (aoCrevice * aoBounce < 0.16) {
              break;
            }
          }
        }
      }
    }
  }

  let nDotL = max(0.0, dot(N, L));
  let cosV = max(0.0, dot(N, V));

  // Dielectric wood (F0 = 0.038) vs Conductor brushed brass inlay (F0 = agedBrassAlbedo)
  let f0 = mix(vec3<f32>(0.038), mat.albedo, mat.metallic);
  let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cosV, 5.0);
  let diffuseAlbedo = mat.albedo * (1.0 - mat.metallic * 0.92);

  let ggx = evalGGXSpecular(N, V, L, mat.roughness);
  let specDirect = fresnel * ggx * sunColor * causticRGB * windowGobo * mix(0.45, 1.35, mat.metallic);

  // Soft glossy environment reflection attenuated by surface roughness & contact AO
  let envRoughScale = (1.0 - mat.roughness * 0.72) * mix(0.32, 0.85, mat.metallic);
  let envRefl = sampleCathedralEnvironment(reflect(-V, N), L, sunColor, skyTint) * fresnel * envRoughScale * aoBounce * aoCrevice;

  // Concentrated HDR jewel caustics illuminate the wood fibers while colored refractive shadows ground the silhouettes!
  let directIrradiance = sunColor * nDotL * causticRGB * windowGobo;
  let ambientIrradiance = (skyTint * 0.11 + vec3<f32>(0.024, 0.019, 0.016) * skyScale) * aoBounce * aoCrevice;

  // Subsurface wood-grain scatter ONLY where focused stained-glass ribbons (> 1.02x unobstructed sun) hit the walnut
  let focalExcess = max(vec3<f32>(0.0), causticRGB - vec3<f32>(1.02)) * windowGobo;
  let woodCausticScatter = diffuseAlbedo * focalExcess * sunColor * 0.18;

  let edgeVignette = 1.0 - smoothstep(3.45, 4.30, length(input.worldPos.xz)) * 0.32;
  let finalColor = (diffuseAlbedo * (directIrradiance + ambientIrradiance) + woodCausticScatter + specDirect + envRefl) * edgeVignette
                 + vec3<f32>(_keepShards + _keepDepth + _keepOpt);

  let camDist = length(uScene.cameraPosAndTime.xyz - input.worldPos);
  return vec4<f32>(finalColor, camDist);
}
`;

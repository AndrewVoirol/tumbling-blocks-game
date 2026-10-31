import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 2: Cathedral Sky Dome + Quarter-Sawn Oak Pedestal Table with
 * Real-Time Projected Stained-Glass Colored Caustics & Chromatic Prism Fringing.
 */
export const ENVIRONMENT_PASS_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uBlocks: array<GPUBlockData>;
@group(0) @binding(2) var uCausticTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;

// 1. Full-screen Cathedral Sky Dome (depth = 1.0, depthWrite = false)
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
  let farWorld = uScene.invViewProj * vec4<f32>(input.ndc, 1.0, 1.0);
  let worldPos = farWorld.xyz / max(farWorld.w, 1e-6);
  let rayDir = normalize(worldPos - uScene.cameraPosAndTime.xyz);
  let col = sampleCathedralEnvironment(
    rayDir,
    normalize(uScene.sunDirAndBlockCount.xyz),
    uScene.sunColorAndAspect.xyz,
    uScene.skyTintAndMode.xyz
  );
  return vec4<f32>(col, 1.0);
}

// 2. Procedural 3D Round Oak Pedestal Table (procedural cylinder + beveled rim generated in VS!)
struct TableOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
};

const TABLE_SEGMENTS: u32 = 96u;
const TABLE_R: f32 = 4.30;
const TABLE_H: f32 = 0.32;

@vertex
fn vs_table(@builtin(vertex_index) vid: u32) -> TableOut {
  // Each radial slice emits 3 triangles (9 vertices):
  // Tri 0: Top disc (center -> r0 -> r1)
  // Tri 1 & 2: Outer beveled rim band (r0_top -> r0_bot -> r1_top, r1_top -> r0_bot -> r1_bot)
  let slice = vid / 9u;
  let sub = vid % 9u;

  let a0 = (f32(slice) / f32(TABLE_SEGMENTS)) * 2.0 * PI;
  let a1 = (f32(slice + 1u) / f32(TABLE_SEGMENTS)) * 2.0 * PI;

  let dir0 = vec2<f32>(cos(a0), sin(a0));
  let dir1 = vec2<f32>(cos(a1), sin(a1));

  var wp = vec3<f32>(0.0);
  var wn = vec3<f32>(0.0, 1.0, 0.0);

  if (sub == 0u) {
    wp = vec3<f32>(0.0, 0.0, 0.0);
    wn = vec3<f32>(0.0, 1.0, 0.0);
  } else if (sub == 1u) {
    wp = vec3<f32>(dir1.x * TABLE_R, 0.0, dir1.y * TABLE_R);
    wn = vec3<f32>(0.0, 1.0, 0.0);
  } else if (sub == 2u) {
    wp = vec3<f32>(dir0.x * TABLE_R, 0.0, dir0.y * TABLE_R);
    wn = vec3<f32>(0.0, 1.0, 0.0);
  } else if (sub == 3u) {
    wp = vec3<f32>(dir0.x * TABLE_R, 0.0, dir0.y * TABLE_R);
    wn = normalize(vec3<f32>(dir0.x, 0.25, dir0.y));
  } else if (sub == 4u) {
    wp = vec3<f32>(dir1.x * TABLE_R, 0.0, dir1.y * TABLE_R);
    wn = normalize(vec3<f32>(dir1.x, 0.25, dir1.y));
  } else if (sub == 5u) {
    wp = vec3<f32>(dir0.x * (TABLE_R * 0.985), -TABLE_H, dir0.y * (TABLE_R * 0.985));
    wn = vec3<f32>(dir0.x, -0.1, dir0.y);
  } else if (sub == 6u) {
    wp = vec3<f32>(dir1.x * TABLE_R, 0.0, dir1.y * TABLE_R);
    wn = normalize(vec3<f32>(dir1.x, 0.25, dir1.y));
  } else if (sub == 7u) {
    wp = vec3<f32>(dir1.x * (TABLE_R * 0.985), -TABLE_H, dir1.y * (TABLE_R * 0.985));
    wn = vec3<f32>(dir1.x, -0.1, dir1.y);
  } else {
    wp = vec3<f32>(dir0.x * (TABLE_R * 0.985), -TABLE_H, dir0.y * (TABLE_R * 0.985));
    wn = vec3<f32>(dir0.x, -0.1, dir0.y);
  }

  var out: TableOut;
  out.clipPos = uScene.viewProj * vec4<f32>(wp, 1.0);
  out.worldPos = wp;
  out.worldNormal = wn;
  return out;
}

fn sampleCausticWithDispersion(worldPos: vec3<f32>) -> vec3<f32> {
  let sunClip = uScene.sunViewProj * vec4<f32>(worldPos, 1.0);
  let ndc = sunClip.xyz / max(sunClip.w, 1e-6);
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

  if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998) {
    return vec3<f32>(1.0);
  }

  // Soft 5-tap cross filter + subtle prismatic RGB dispersion offset!
  let disp = vec2<f32>(0.0014, 0.0009);
  let texel = 1.0 / 2048.0;

  let cCenter = textureSampleLevel(uCausticTex, uLinearSampler, uv, 0.0).rgb;
  let cN = textureSampleLevel(uCausticTex, uLinearSampler, uv + vec2<f32>(0.0, texel * 1.6), 0.0).rgb;
  let cS = textureSampleLevel(uCausticTex, uLinearSampler, uv - vec2<f32>(0.0, texel * 1.6), 0.0).rgb;
  let cE = textureSampleLevel(uCausticTex, uLinearSampler, uv + vec2<f32>(texel * 1.6, 0.0), 0.0).rgb;
  let cW = textureSampleLevel(uCausticTex, uLinearSampler, uv - vec2<f32>(texel * 1.6, 0.0), 0.0).rgb;
  let smoothC = cCenter * 0.44 + (cN + cS + cE + cW) * 0.14;

  let rShift = textureSampleLevel(uCausticTex, uLinearSampler, uv + disp, 0.0).r;
  let bShift = textureSampleLevel(uCausticTex, uLinearSampler, uv - disp, 0.0).b;

  return vec3<f32>(mix(smoothC.r, rShift, 0.45), smoothC.g, mix(smoothC.b, bShift, 0.45));
}

@fragment
fn fs_table(input: TableOut) -> @location(0) vec4<f32> {
  let N = normalize(input.worldNormal);
  let V = normalize(uScene.cameraPosAndTime.xyz - input.worldPos);
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let albedo = evaluateOakTableAlbedo(input.worldPos.xz);

  // Sample stained-glass colored sun transmission & caustic veins!
  let causticRGB = sampleCausticWithDispersion(input.worldPos);

  // Compute crisp contact ambient occlusion from lowest-layer blocks resting near y = 0
  let blockCount = u32(uScene.sunDirAndBlockCount.w);
  var ao = 1.0;
  if (input.worldPos.y > -0.02 && length(input.worldPos.xz) < 2.2) {
    for (var i: u32 = 0u; i < blockCount; i = i + 1u) {
      let blk = uBlocks[i];
      let bCenterY = blk.localToWorld[3].y;
      if (bCenterY < 0.32) {
        let pLocal = (blk.worldToLocal * vec4<f32>(input.worldPos, 1.0)).xyz;
        let q = abs(pLocal) - vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
        let boxDist = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        let occ = (1.0 - smoothstep(0.0, 0.075, boxDist)) * (1.0 - smoothstep(0.07, 0.28, bCenterY));
        ao *= (1.0 - occ * 0.58);
      }
    }
  }

  let nDotL = max(0.0, dot(N, L));
  let H = normalize(L + V);
  // Satin hand-rubbed wax sheen on oak
  let spec = pow(max(0.0, dot(N, H)), 38.0) * 0.14;
  let fresnel = 0.03 + 0.12 * pow(1.0 - max(0.0, dot(N, V)), 4.0);
  let envRefl = sampleCathedralEnvironment(reflect(-V, N), L, sunColor, skyTint) * fresnel * 0.42;

  // Rich jewel-toned caustic glow where concentrated stained-glass light hits the wood grain!
  let chromaSat = max(causticRGB.r, max(causticRGB.g, causticRGB.b)) - min(causticRGB.r, min(causticRGB.g, causticRGB.b));
  let jewelCausticBoost = causticRGB * (1.0 + chromaSat * 0.85);
  let directLight = sunColor * nDotL * jewelCausticBoost;
  let ambientLight = (skyTint * 0.34 + vec3<f32>(0.09, 0.075, 0.065)) * ao;

  // Radial vignette toward outer table edge for grounded craftsman studio feel
  let edgeShade = 1.0 - smoothstep(3.2, 4.35, length(input.worldPos.xz)) * 0.28;

  // Forward-scattered stained-glass light illuminates wood grain with luminous cathedral color!
  let jewelGlowOnWood = causticRGB * chromaSat * sunColor * 0.14;
  let color = (albedo * (directLight * 1.18 + ambientLight) + jewelGlowOnWood + sunColor * spec * causticRGB + envRefl * ao) * edgeShade;
  return vec4<f32>(color, 1.0);
}
`;

import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 3: Intact Stained-Glass Tower Shader (`glassTower.wgsl.ts`)
 * - Rasterizes smooth-chamfered beveled glass prisms at exact 7.5 x 2.5 x 1.5 cm scale
 * - Shades patinated copper-foil / pewter lead came frames & Gothic tracery mullions
 * - Traces true chromatic-dispersed dielectric rays INSIDE the glass block (including internal TIR!)
 *   AND across up to 3 blocks behind it in the tower, applying Beer-Lambert spectral absorption,
 *   3D volumetric seed bubbles, internal crazing cracks, and refracted caustic tabletop sampling!
 */
export const GLASS_TOWER_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uBlocks: array<GPUBlockData>;
@group(0) @binding(2) var uCausticTex: texture_2d<f32>;
@group(0) @binding(3) var uLinearSampler: sampler;

struct TowerVSOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) localPos: vec3<f32>,
  @location(3) localNormal: vec3<f32>,
  @location(4) @interpolate(flat) blockIdx: u32,
};

@vertex
fn vs_glass_tower(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> TowerVSOut {
  let blk = uBlocks[instIdx];
  let wp = blk.localToWorld * vec4<f32>(pos, 1.0);
  // Rigid body rotation preserves orthogonal normals
  let wn = normalize((blk.localToWorld * vec4<f32>(normal, 0.0)).xyz);

  var out: TowerVSOut;
  out.clipPos = uScene.viewProj * wp;
  out.worldPos = wp.xyz;
  out.worldNormal = wn;
  out.localPos = pos;
  out.localNormal = normal;
  out.blockIdx = instIdx;
  return out;
}

struct BoxHit {
  hit: bool,
  tNear: f32,
  tFar: f32,
  exitNormalLocal: vec3<f32>,
};

/**
 * Exact slab intersection against local block box [-BLOCK_HX, BLOCK_HX] x [-BLOCK_HY, BLOCK_HY] x [-BLOCK_HZ, BLOCK_HZ].
 */
fn intersectLocalBlockBox(ro: vec3<f32>, rd: vec3<f32>) -> BoxHit {
  let halfExt = vec3<f32>(BLOCK_HX, BLOCK_HY, BLOCK_HZ);
  let invD = 1.0 / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
  let t0 = (-halfExt - ro) * invD;
  let t1 = ( halfExt - ro) * invD;
  let tSmall = min(t0, t1);
  let tBig = max(t0, t1);

  let tNear = max(tSmall.x, max(tSmall.y, tSmall.z));
  let tFar = min(tBig.x, min(tBig.y, tBig.z));

  var exitN = vec3<f32>(0.0, 1.0, 0.0);
  if (tBig.x <= tBig.y && tBig.x <= tBig.z) {
    exitN = vec3<f32>(sign(rd.x), 0.0, 0.0);
  } else if (tBig.y <= tBig.x && tBig.y <= tBig.z) {
    exitN = vec3<f32>(0.0, sign(rd.y), 0.0);
  } else {
    exitN = vec3<f32>(0.0, 0.0, sign(rd.z));
  }

  var res: BoxHit;
  res.hit = (tFar >= max(tNear, 0.0));
  res.tNear = tNear;
  res.tFar = max(0.001, tFar);
  res.exitNormalLocal = exitN;
  return res;
}

/**
 * Evaluatestiny trapped 3D seed bubbles suspended inside mouth-blown cathedral glass!
 */
fn evaluateSeedBubbles(roLocal: vec3<f32>, rdLocal: vec3<f32>, segLen: f32, seed: f32) -> f32 {
  var glint: f32 = 0.0;
  for (var s: i32 = 0; s < 3; s = s + 1) {
    let t = segLen * (0.22 + f32(s) * 0.28);
    let p = (roLocal + rdLocal * t) * 34.0 + vec3<f32>(seed * 5.1, seed * 2.7, seed * 8.3);
    let cell = floor(p);
    let f = fract(p) - vec3<f32>(0.5);
    let h = hash31(cell);
    // ~14% of cells contain a tiny suspended cathedral glass air seed bubble
    if (h > 0.86) {
      let offset = vec3<f32>(fract(h * 13.7) - 0.5, fract(h * 29.3) - 0.5, fract(h * 47.1) - 0.5) * 0.45;
      let d = length(f - offset);
      let bubbleRadius = 0.06 + fract(h * 91.7) * 0.07;
      let rim = smoothstep(bubbleRadius, bubbleRadius * 0.45, d) * smoothstep(0.0, bubbleRadius * 0.85, d);
      glint += rim * 0.55;
    }
  }
  return clamp(glint, 0.0, 0.85);
}

/**
 * Evaluates background scene (lit caustic oak tabletop at y = 0 or Gothic Cathedral environment)
 * along any 3D refracted world ray! Uses textureSampleLevel(..., 0.0) for strict WGSL uniform control flow!
 */
fn sampleRefractedWorldBackground(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  // Check intersection with oak tabletop at y = 0.0
  if (rd.y < -1e-4 && ro.y > 0.0) {
    let tTable = -ro.y / rd.y;
    let hitPos = ro + rd * tTable;
    if (length(hitPos.xz) <= 4.30) {
      let albedo = evaluateOakTableAlbedo(hitPos.xz);
      let sunClip = uScene.sunViewProj * vec4<f32>(hitPos, 1.0);
      let ndc = sunClip.xyz / max(sunClip.w, 1e-6);
      let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
      var caustic = vec3<f32>(1.0);
      if (uv.x >= 0.002 && uv.x <= 0.998 && uv.y >= 0.002 && uv.y <= 0.998) {
        caustic = textureSampleLevel(uCausticTex, uLinearSampler, uv, 0.0).rgb;
      }
      let nDotL = max(0.0, L.y);
      let edgeShade = 1.0 - smoothstep(3.2, 4.35, length(hitPos.xz)) * 0.28;
      let ambient = skyTint * 0.30 + vec3<f32>(0.08, 0.07, 0.06);
      return albedo * (sunColor * nDotL * caustic * 1.18 + ambient) * edgeShade;
    }
  }
  return sampleCathedralEnvironment(rd, L, sunColor, skyTint);
}

/**
 * Traces a chromatic-dispersed dielectric ray through the front glass block (including internal TIR)
 * AND continues through up to 2 stained-glass blocks behind it in a single unified pass!
 */
fn traceTowerGlassRayDispersed(
  startWorldPos: vec3<f32>,
  incidentDir: vec3<f32>,
  frontWorldNormal: vec3<f32>,
  primaryIdx: u32,
  vitrailFront: VitrailSample
) -> vec3<f32> {
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let blk0 = uBlocks[primaryIdx];
  let seed0 = blk0.accentAndSeed.w;
  let ior: f32 = 1.522;

  // 1. Refract into primary block
  var internalDirWorld = refract(incidentDir, frontWorldNormal, 1.0 / ior);
  if (length(internalDirWorld) < 1e-4) {
    internalDirWorld = reflect(incidentDir, frontWorldNormal);
    return sampleCathedralEnvironment(internalDirWorld, L, sunColor, skyTint);
  }
  internalDirWorld = normalize(internalDirWorld);

  let roLocal = (blk0.worldToLocal * vec4<f32>(startWorldPos + internalDirWorld * 0.002, 1.0)).xyz;
  let rdLocal = normalize((blk0.worldToLocal * vec4<f32>(internalDirWorld, 0.0)).xyz);

  let box0 = intersectLocalBlockBox(roLocal, rdLocal);
  var pathLen0 = box0.tFar;
  let exitPosLocal = roLocal + rdLocal * pathLen0;
  let exitNormalWorld = normalize((blk0.localToWorld * vec4<f32>(box0.exitNormalLocal, 0.0)).xyz);

  // Perturb exit normal with cathedral waterglass ripples on exit face
  let motif0 = i32(round(blk0.absorptionAndMotif.w));
  let vitrailExit = evaluateVitrailPattern(exitPosLocal, motif0, seed0);
  let exitPerturbedN = normalize(exitNormalWorld + (blk0.localToWorld * vec4<f32>(vitrailExit.rippleGrad, 0.0)).xyz * 1.3);

  // Refract out of primary block (check for Total Internal Reflection inside bevel/side walls!)
  var worldRayPos = (blk0.localToWorld * vec4<f32>(exitPosLocal, 1.0)).xyz;
  var worldRayDir = refract(internalDirWorld, -exitPerturbedN, ior);
  var internalTIRGlow = vec3<f32>(0.0);

  if (length(worldRayDir) < 1e-4) {
    // Authentic Total Internal Reflection (TIR) bounce inside rectangular glass prism!
    let bouncedLocalDir = reflect(rdLocal, -box0.exitNormalLocal);
    let boxBounce = intersectLocalBlockBox(exitPosLocal + bouncedLocalDir * 0.001, bouncedLocalDir);
    pathLen0 += boxBounce.tFar * 0.75;
    let finalLocalPos = exitPosLocal + bouncedLocalDir * boxBounce.tFar;
    let finalExitNWorld = normalize((blk0.localToWorld * vec4<f32>(boxBounce.exitNormalLocal, 0.0)).xyz);
    let bouncedWorldDir = normalize((blk0.localToWorld * vec4<f32>(bouncedLocalDir, 0.0)).xyz);

    worldRayPos = (blk0.localToWorld * vec4<f32>(finalLocalPos, 1.0)).xyz;
    worldRayDir = refract(bouncedWorldDir, -finalExitNWorld, ior);
    if (length(worldRayDir) < 1e-4) {
      worldRayDir = reflect(bouncedWorldDir, -finalExitNWorld);
    }
    internalTIRGlow = sampleCathedralEnvironment(bouncedWorldDir, L, sunColor, skyTint) * 0.28;
  }
  worldRayDir = normalize(worldRayDir);

  // Beer-Lambert spectral absorption across exact 3D optical path inside block 0!
  let sigma0 = mix(blk0.absorptionAndMotif.xyz, blk0.accentAndSeed.xyz, 0.5 * (vitrailFront.accentMix + vitrailExit.accentMix));
  var throughput = exp(-sigma0 * pathLen0 * 0.88);
  throughput *= (1.0 - vitrailExit.cameMask * 0.65);

  // Suspended 3D seed air bubbles & internal planar cleavage cracks inside block 0!
  let bubbleSparkle = evaluateSeedBubbles(roLocal, rdLocal, box0.tFar, seed0);
  let midLocal = roLocal + rdLocal * (box0.tFar * 0.38);
  let farLocal = roLocal + rdLocal * (box0.tFar * 0.74);
  let crackMask = max(
    evaluateInternalCracks(roLocal, blk0.crackAndDamage.xyz, blk0.crackAndDamage.w, seed0),
    max(
      evaluateInternalCracks(midLocal, blk0.crackAndDamage.xyz, blk0.crackAndDamage.w, seed0),
      evaluateInternalCracks(farLocal, blk0.crackAndDamage.xyz, blk0.crackAndDamage.w, seed0)
    )
  );
  // Air-film internal cleavage planes: subtle transmission shadow + oblique TIR silver glint (never flat white paint!)
  let crackOblique = pow(1.0 - abs(rdLocal.x * 0.6 + rdLocal.z * 0.8), 2.0);
  throughput *= (1.0 - crackMask * 0.32);
  let internalScatter =
    bubbleSparkle * 0.42 * (sunColor * 0.32 + skyTint * 0.45) * throughput +
    crackMask * (0.28 + 0.65 * crackOblique) * (sunColor * 0.28 + skyTint * 0.52) * mix(vec3<f32>(1.0), throughput, 0.55);

  // 2. Continue ray through up to 2 stained-glass blocks behind primaryIdx in the Jenga tower!
  let blockCount = u32(uScene.sunDirAndBlockCount.w);
  var lastHitIdx = primaryIdx;

  for (var bounce: u32 = 0u; bounce < 2u; bounce = bounce + 1u) {
    var nearestT: f32 = 1e9;
    var nearestIdx: u32 = 999u;
    var nearestTFar: f32 = 0.0;
    var nearestLocalHit = vec3<f32>(0.0);

    for (var j: u32 = 0u; j < blockCount; j = j + 1u) {
      if (j == lastHitIdx) {
        continue;
      }
      let candCenter = uBlocks[j].localToWorld[3].xyz;
      let toCand = candCenter - worldRayPos;
      let projT = dot(toCand, worldRayDir);
      if (projT < -0.38 || projT > nearestT + 0.38) {
        continue;
      }
      let closestPt = worldRayPos + worldRayDir * max(0.0, projT);
      // Tight OBB bounding check rejects >90% of blocks before 4x4 matrix transforms!
      if (abs(closestPt.y - candCenter.y) > 0.22 || dot(closestPt.xz - candCenter.xz, closestPt.xz - candCenter.xz) > 0.22) {
        continue;
      }

      let cand = uBlocks[j];
      let cRo = (cand.worldToLocal * vec4<f32>(worldRayPos, 1.0)).xyz;
      let cRd = (cand.worldToLocal * vec4<f32>(worldRayDir, 0.0)).xyz;
      let cBox = intersectLocalBlockBox(cRo, cRd);
      if (cBox.hit && cBox.tNear > 0.0005 && cBox.tNear < nearestT) {
        nearestT = cBox.tNear;
        nearestTFar = cBox.tFar;
        nearestIdx = j;
        nearestLocalHit = cRo + cRd * cBox.tNear;
      }
    }

    if (nearestIdx == 999u) {
      break;
    }

    let hitBlk = uBlocks[nearestIdx];
    let hitMotif = i32(round(hitBlk.absorptionAndMotif.w));
    let hitSeed = hitBlk.accentAndSeed.w;
    let vHit = evaluateVitrailPattern(nearestLocalHit, hitMotif, hitSeed);

    let segLen = max(0.01, nearestTFar - nearestT);
    let sigmaHit = mix(hitBlk.absorptionAndMotif.xyz, hitBlk.accentAndSeed.xyz, vHit.accentMix);
    throughput *= exp(-sigmaHit * segLen * 0.82);
    throughput *= mix(1.0, 0.28, vHit.cameMask);

    let bendWorld = (hitBlk.localToWorld * vec4<f32>(vHit.rippleGrad, 0.0)).xyz;
    worldRayPos = worldRayPos + worldRayDir * (nearestTFar + 0.002);
    worldRayDir = normalize(worldRayDir + bendWorld * 0.75);
    lastHitIdx = nearestIdx;
  }

  // Analytical RGB Abbe Chromatic Dispersion spread on exiting prism ray!
  let dispTangent = normalize(cross(worldRayDir, frontWorldNormal + vec3<f32>(0.001, 0.002, 0.001)));
  let dispStrength = 0.016 * (1.0 - abs(dot(incidentDir, frontWorldNormal)));
  let bgG = sampleRefractedWorldBackground(worldRayPos, worldRayDir);
  let bgR = sampleRefractedWorldBackground(worldRayPos, normalize(worldRayDir - dispTangent * dispStrength)).r;
  let bgB = sampleRefractedWorldBackground(worldRayPos, normalize(worldRayDir + dispTangent * dispStrength)).b;

  let bgDispersed = vec3<f32>(bgR, bgG.g, bgB);
  // Warm internal stained-glass forward sunlight scatter (makes backlit stained glass glow!)
  let sunBacklit = pow(max(0.0, dot(incidentDir, -L)), 3.0) * 0.22 + 0.06;
  let internalJewelGlow = throughput * sunColor * sunBacklit * 0.38;

  return bgDispersed * throughput + internalJewelGlow + internalScatter + internalTIRGlow * throughput;
}

@fragment
fn fs_glass_tower(input: TowerVSOut) -> @location(0) vec4<f32> {
  // Unconditional derivative evaluation at top of fragment shader for strict WGSL uniform control flow!
  let dLocalPos = fwidth(input.localPos);
  let aaFilter = max(dLocalPos.x, max(dLocalPos.y, dLocalPos.z));

  let blk = uBlocks[input.blockIdx];
  let motifId = i32(round(blk.absorptionAndMotif.w));
  let seed = blk.accentAndSeed.w;
  let hover = blk.stateFlags.x;
  let grabbed = blk.stateFlags.y;
  let isLegal = blk.stateFlags.z;
  let resistance = blk.stateFlags.w;

  let V = normalize(uScene.cameraPosAndTime.xyz - input.worldPos);
  let incidentDir = -V;
  let L = normalize(uScene.sunDirAndBlockCount.xyz);
  let sunColor = uScene.sunColorAndAspect.xyz;
  let skyTint = uScene.skyTintAndMode.xyz;

  let vitrail = evaluateVitrailPattern(input.localPos, motifId, seed);

  // Perturb geometric bevel normal with cathedral waterglass undulations
  let rippleWorld = (blk.localToWorld * vec4<f32>(vitrail.rippleGrad, 0.0)).xyz;
  let glassNormal = normalize(input.worldNormal + rippleWorld * 1.65);

  // Soldered Copper-Foil / Patinated Pewter Lead Came Shading
  let H = normalize(L + V);
  var cameColor = vec3<f32>(0.0);
  if (vitrail.cameMask > 0.01) {
    // Subtle rounded solder bead normal profile
    let solderRoughness = 0.34;
    let baseMetal = vec3<f32>(0.13, 0.105, 0.095); // Antique patinated copper-pewter solder
    let warmCopperGlint = vec3<f32>(0.44, 0.30, 0.21);
    let nDotL = max(0.0, dot(input.worldNormal, L));
    let solderSpec = pow(max(0.0, dot(input.worldNormal, H)), 28.0) * 0.65;
    let solderEnv = sampleCathedralEnvironment(reflect(incidentDir, input.worldNormal), L, sunColor, skyTint);
    cameColor = baseMetal * (sunColor * nDotL * 0.55 + skyTint * 0.35 + vec3<f32>(0.06))
              + warmCopperGlint * (solderSpec * sunColor * 0.45 + solderEnv * (1.0 - solderRoughness) * 0.35);
  }

  // Exact Schlick Dielectric Fresnel for soda-lime cathedral glass (n = 1.52 -> F0 = 0.0426)
  let cosTheta = clamp(dot(glassNormal, V), 0.0, 1.0);
  let f0 = 0.0426;
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);

  let reflDir = reflect(incidentDir, glassNormal);
  let envReflection = sampleCathedralEnvironment(reflDir, L, sunColor, skyTint);
  // Crisp razor-sharp sun glint on smooth beveled glass rims!
  let sunSpec = pow(max(0.0, dot(glassNormal, H)), 180.0) * sunColor * 2.6;
  // Secondary softer waterglass sheen
  let glassSheen = pow(max(0.0, dot(glassNormal, H)), 32.0) * sunColor * 0.18;

  // Single-pass multi-block tower raytrace with analytical Abbe chromatic dispersion!
  let refractedRGB = traceTowerGlassRayDispersed(input.worldPos, incidentDir, glassNormal, input.blockIdx, vitrail);

  var glassFinal = mix(refractedRGB, envReflection, fresnel) + sunSpec + glassSheen;

  // Crisp grazing-angle Fresnel glint along surface-breaking hairline cleavage cracks
  let surfCrack = evaluateInternalCracks(input.localPos, blk.crackAndDamage.xyz, blk.crackAndDamage.w, seed);
  let crackSpec = pow(max(0.0, dot(normalize(glassNormal + vec3<f32>(0.22, 0.08, -0.18)), H)), 56.0);
  glassFinal += surfCrack * (sunColor * crackSpec * 1.15 + skyTint * 0.32 * pow(1.0 - cosTheta, 2.0));

  // Blend soldered came over stained glass with smooth anti-aliased transition
  let smoothCame = smoothstep(0.15 - aaFilter * 2.0, 0.85 + aaFilter * 2.0, vitrail.cameMask);
  var finalRGB = mix(glassFinal, cameColor, smoothCame);

  // Intuitive Tactile UI Rim Feedback strictly along beveled edges & copper-foil cames so glass interior stays saturated!
  let edgeRim = max(smoothCame * 0.85, pow(1.0 - cosTheta, 4.4));
  if (grabbed > 0.5) {
    // Smoothly shifts from luminous champagne (loose) -> warm amber (snug) -> warning crimson (pinned!)
    let looseCol = vec3<f32>(0.92, 0.88, 0.76);
    let snugCol = vec3<f32>(1.0, 0.68, 0.18);
    let pinnedCol = vec3<f32>(1.0, 0.16, 0.16);
    let resCol = select(
      mix(looseCol, snugCol, clamp(resistance * 2.0, 0.0, 1.0)),
      mix(snugCol, pinnedCol, clamp((resistance - 0.5) * 2.0, 0.0, 1.0)),
      resistance > 0.5
    );
    finalRGB += resCol * edgeRim * 0.48;
  } else if (hover > 0.5) {
    if (isLegal > 0.5) {
      finalRGB += vec3<f32>(0.98, 0.88, 0.65) * edgeRim * 0.42;
    } else {
      // Top-layer locked block: subtle dim slate-rose cue so player knows top row cannot be pulled
      finalRGB = mix(finalRGB, finalRGB * vec3<f32>(0.72, 0.68, 0.75), 0.38);
    }
  }

  return vec4<f32>(finalRGB, 1.0);
}
`;

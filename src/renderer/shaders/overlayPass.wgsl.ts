import { COMMON_GLASS_WGSL } from './commonGlass.wgsl';

/**
 * Pass 4b: Intuitive Spatial Overlays (`overlayPass.wgsl.ts`)
 * 1. Luminous Stained-Glass Top-Slot Guides when holding an extracted block (`EXTRACTED` phase).
 * 2. Tactile Spring Tension Filament from block grab point to cursor anchor (`PULLING` phase).
 */
export const OVERLAY_PASS_WGSL = /* wgsl */ `
${COMMON_GLASS_WGSL}

struct GPUOverlaySlot {
  localToWorld: mat4x4<f32>,
  colorAndHover: vec4<f32>, // xyz = jewel tint, w = isHovered (1.0 or 0.0)
};

struct GPUSpringLine {
  p0: vec4<f32>,            // xyz = block grab point, w = active (1.0 or 0.0)
  p1AndRes: vec4<f32>,      // xyz = hand anchor point, w = resistance01
};

@group(0) @binding(0) var<uniform> uScene: SceneUniforms;
@group(0) @binding(1) var<storage, read> uSlots: array<GPUOverlaySlot>;
@group(0) @binding(2) var<uniform> uSpring: GPUSpringLine;

struct SlotVSOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) localPos: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) isHovered: f32,
  @location(3) scaleXYZ: vec3<f32>,
};

@vertex
fn vs_slot_guide(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) iid: u32
) -> SlotVSOut {
  let slot = uSlots[iid];
  let wp = slot.localToWorld * vec4<f32>(pos, 1.0);
  let _n = normal.x * 0.0;
  let sx = max(0.05, length(slot.localToWorld[0].xyz));
  let sy = max(0.02, length(slot.localToWorld[1].xyz));
  let sz = max(0.05, length(slot.localToWorld[2].xyz));

  var out: SlotVSOut;
  out.clipPos = uScene.viewProj * wp + vec4<f32>(_n, 0.0, 0.0, 0.0);
  out.localPos = pos;
  out.color = slot.colorAndHover.xyz;
  out.isHovered = slot.colorAndHover.w;
  out.scaleXYZ = vec3<f32>(sx, sy, sz);
  return out;
}

@fragment
fn fs_slot_guide(input: SlotVSOut) -> @location(0) vec4<f32> {
  let ax = (BLOCK_HX - abs(input.localPos.x)) * input.scaleXYZ.x;
  let ay = (BLOCK_HY - abs(input.localPos.y)) * input.scaleXYZ.y;
  let az = (BLOCK_HZ - abs(input.localPos.z)) * input.scaleXYZ.z;
  let dMin = min(ax, min(ay, az));
  let dMax = max(ax, max(ay, az));
  let edgeDist = ax + ay + az - dMin - dMax;

  let pulse = 0.85 + 0.15 * sin(uScene.cameraPosAndTime.w * 5.0);
  let edgeMask = 1.0 - smoothstep(0.0035, 0.013, edgeDist);
  let fillAlpha = select(0.035, 0.15, input.isHovered > 0.5);
  let edgeAlpha = select(0.34, 0.92, input.isHovered > 0.5) * pulse;

  let alpha = clamp(fillAlpha + edgeMask * edgeAlpha, 0.0, 0.95);
  let glow = select(0.95, 2.1, input.isHovered > 0.5);
  let rgb = input.color * glow * alpha;
  return vec4<f32>(rgb, alpha);
}

// Tactile Spring Tension Ribbon
struct LineVSOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) tAlong: f32,
  @location(1) vCross: f32,
  @location(2) resistance: f32,
};

@vertex
fn vs_spring_line(@builtin(vertex_index) vid: u32) -> LineVSOut {
  var out: LineVSOut;
  if (uSpring.p0.w < 0.5) {
    out.clipPos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.tAlong = 0.0;
    out.vCross = 0.0;
    out.resistance = 0.0;
    return out;
  }

  var quad = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0,  1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0,  1.0), vec2<f32>(0.0,  1.0)
  );
  let q = quad[vid];
  let p0 = uSpring.p0.xyz;
  let p1 = uSpring.p1AndRes.xyz;
  let seg = p1 - p0;
  let segLen = length(seg);

  if (segLen < 0.004) {
    out.clipPos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.tAlong = 0.0;
    out.vCross = 0.0;
    out.resistance = 0.0;
    return out;
  }

  let worldPt = mix(p0, p1, q.x);
  let toCam = normalize(uScene.cameraPosAndTime.xyz - worldPt);
  let side = normalize(cross(seg / segLen, toCam));
  let thickness = 0.0055 + uSpring.p1AndRes.w * 0.0045;
  let finalWorld = worldPt + side * (q.y * thickness);

  out.clipPos = uScene.viewProj * vec4<f32>(finalWorld, 1.0);
  out.tAlong = q.x;
  out.vCross = q.y;
  out.resistance = uSpring.p1AndRes.w;
  return out;
}

@fragment
fn fs_spring_line(input: LineVSOut) -> @location(0) vec4<f32> {
  let core = 1.0 - abs(input.vCross);
  let soft = pow(max(0.0, core), 1.6);

  let looseCol = vec3<f32>(0.75, 0.94, 1.0);
  let snugCol = vec3<f32>(1.0, 0.76, 0.24);
  let pinnedCol = vec3<f32>(1.0, 0.22, 0.22);
  let col = select(
    mix(looseCol, snugCol, clamp(input.resistance * 2.0, 0.0, 1.0)),
    mix(snugCol, pinnedCol, clamp((input.resistance - 0.5) * 2.0, 0.0, 1.0)),
    input.resistance > 0.5
  );

  // Subtle flowing bead animation toward cursor
  let wave = 0.75 + 0.25 * sin(input.tAlong * 22.0 - uScene.cameraPosAndTime.w * 10.0);
  let alpha = soft * wave * 0.85;
  return vec4<f32>(col * 2.2 * alpha, alpha);
}
`;

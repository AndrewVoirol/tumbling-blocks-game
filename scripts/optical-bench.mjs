#!/usr/bin/env node
/**
 * Tier-2 Deterministic WebGPU Optical Bench CLI (`scripts/optical-bench.mjs`)
 *
 * Drives a real Apple Silicon Metal-backed Chromium WebGPU context against `?bench=` scenes,
 * computes quantitative optical metrics on full-resolution linear Float32 HDR readbacks,
 * and emits a <= 2048 px (< 1.5 MB) 2x2 diagnostic contact sheet.
 *
 * Usage:
 *   node scripts/optical-bench.mjs --scene two-block-cross
 *   node scripts/optical-bench.mjs --scene backlit-tower
 *   node scripts/optical-bench.mjs --scene caustic-pool
 *   node scripts/optical-bench.mjs --scene contact-sheet
 *   node scripts/optical-bench.mjs --scene all
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
function getArg(flag, fallback) {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : fallback;
}

const targetScene = getArg('--scene', 'all');
const presetFilter = getArg('--preset', null);
const baseUrl = getArg('--url', 'http://localhost:5173');
const outSheetPath = path.resolve(
  process.cwd(),
  getArg('--out', 'screenshots/optical-bench-contact-sheet.png')
);

async function resolvePlaywright() {
  try {
    return await import('playwright');
  } catch {
    const npxRoot = path.join(os.homedir(), '.npm', '_npx');
    if (fs.existsSync(npxRoot)) {
      for (const dir of fs.readdirSync(npxRoot)) {
        const candidate = path.join(npxRoot, dir, 'node_modules', 'playwright', 'index.mjs');
        if (fs.existsSync(candidate)) {
          return await import(pathToFileURL(candidate).href);
        }
      }
    }
    throw new Error(
      'Playwright not found in local or ~/.npm/_npx cache. Run `npx playwright --version` once to cache it.'
    );
  }
}

function checkServerAlive(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve(res.statusCode !== undefined && res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(1200, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ensureDevServer(url) {
  if (await checkServerAlive(url)) {
    return null;
  }
  const viteProc = spawn('npx', ['vite', '--port', '5173', '--strictPort'], {
    cwd: process.cwd(),
    stdio: 'ignore',
  });
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 150));
    if (await checkServerAlive(url)) {
      return viteProc;
    }
  }
  viteProc.kill('SIGTERM');
  throw new Error(`Failed to reach Vite dev server at ${url}`);
}

async function runSceneTwoBlockCross(page) {
  const t0 = performance.now();
  const res = await page.evaluate(async () => {
    const eng = window.__VITRAIL_ENGINE__;
    eng.configureBenchScene('two-block-cross');
    for (let i = 0; i < 4; i++) await new Promise((r) => requestAnimationFrame(r));
    const hdr = await eng.renderer.readHdrResolveFloat32();
    const { width: w, height: h, rgba, nanInfCount } = hdr;

    const worldToPixel = (wx, wy, wz) => {
      const vp = eng.camera.viewProjMatrix;
      const cx = vp[0] * wx + vp[4] * wy + vp[8] * wz + vp[12];
      const cy = vp[1] * wx + vp[5] * wy + vp[9] * wz + vp[13];
      const cw = vp[3] * wx + vp[7] * wy + vp[11] * wz + vp[15];
      return [
        Math.round(((cx / cw) * 0.5 + 0.5) * (w - 1)),
        Math.round((0.5 - (cy / cw) * 0.5) * (h - 1)),
      ];
    };

    const sampleBoxMean = (cx, cy, halfWin) => {
      let sr = 0, sg = 0, sb = 0, n = 0;
      for (let y = Math.max(0, cy - halfWin); y <= Math.min(h - 1, cy + halfWin); y++) {
        for (let x = Math.max(0, cx - halfWin); x <= Math.min(w - 1, cx + halfWin); x++) {
          const i = (y * w + x) * 4;
          sr += rgba[i];
          sg += rgba[i + 1];
          sb += rgba[i + 2];
          n++;
        }
      }
      return [sr / n, sg / n, sb / n];
    };

    const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    const [rx, ry] = worldToPixel(-0.76, 0.80, 0.0);
    const [ax, ay] = worldToPixel(0.00, 0.80, 0.0);
    const [seamX, seamY] = worldToPixel(0.76, 0.80, 0.125);
    const [topCobX, topCobY] = worldToPixel(0.76, 0.865, 0.125);
    const [botCobX, botCobY] = worldToPixel(0.76, 0.735, 0.125);

    const rubyCobRGB = sampleBoxMean(rx, ry, 8);
    const ambCobRGB = sampleBoxMean(ax, ay, 8);

    const intTop = sampleBoxMean(topCobX, topCobY, 5);
    const intBot = sampleBoxMean(botCobX, botCobY, 5);
    const interiorLum = 0.5 * (lum(intTop) + lum(intBot));
    let maxSeamDiff = 0;
    for (let dy = -12; dy <= 12; dy++) {
      const py = Math.max(0, Math.min(h - 1, seamY + dy));
      const i0 = (py * w + Math.max(0, seamX - 2)) * 4;
      const i1 = (py * w + Math.min(w - 1, seamX + 2)) * 4;
      const rowRGB = [
        0.5 * (rgba[i0] + rgba[i1]),
        0.5 * (rgba[i0 + 1] + rgba[i1 + 1]),
        0.5 * (rgba[i0 + 2] + rgba[i1 + 2]),
      ];
      maxSeamDiff = Math.max(maxSeamDiff, Math.abs(lum(rowRGB) - interiorLum));
    }

    // Analytical (1 - F)^4 * exp(-(sigma_1 * l_1 + sigma_2 * l_2)) * L_ref
    // At block center u = 0: accentMix = smoothstep(-0.55, 0.65, 0) * 0.42 = 0.18315
    const effSigma = (base, acc) => [
      base[0] * (1 - 0.18315) + acc[0] * 0.18315 + 0.14,
      base[1] * (1 - 0.18315) + acc[1] * 0.18315 + 0.05,
      base[2] * (1 - 0.18315) + acc[2] * 0.18315 + 0.09,
    ];
    const sRuby = effSigma([0.16, 8.6, 9.4], [0.24, 4.1, 8.9]);
    const sCob = effSigma([8.8, 4.6, 0.18], [6.4, 2.2, 0.22]);
    const sAmb = effSigma([0.14, 1.85, 9.2], [0.22, 3.6, 9.6]);
    const f4 = Math.pow(1 - 0.0428, 4) * 0.80;
    const anaRubyCob = [0, 1, 2].map((c) => f4 * Math.exp(-(sRuby[c] + sCob[c]) * 0.15));
    const anaAmbCob = [0, 1, 2].map((c) => f4 * Math.exp(-(sAmb[c] + sCob[c]) * 0.15));

    const angleDeg = (a, b) => {
      const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      const na = Math.hypot(a[0], a[1], a[2]);
      const nb = Math.hypot(b[0], b[1], b[2]);
      return (Math.acos(Math.min(1, Math.max(-1, dot / (na * nb)))) * 180) / Math.PI;
    };

    const rubyAngle = angleDeg(rubyCobRGB, anaRubyCob);
    const ambAngle = angleDeg(ambCobRGB, anaAmbCob);
    const seamContrast = maxSeamDiff / Math.max(1e-6, interiorLum);

    return {
      width: w,
      height: h,
      nanInfCount,
      rubyCobRGB: rubyCobRGB.map((v) => +v.toFixed(4)),
      anaRubyCob: anaRubyCob.map((v) => +v.toFixed(4)),
      rubyCobAngleDeg: +rubyAngle.toFixed(3),
      ambCobRGB: ambCobRGB.map((v) => +v.toFixed(4)),
      anaAmbCob: anaAmbCob.map((v) => +v.toFixed(4)),
      ambCobAngleDeg: +ambAngle.toFixed(3),
      cobaltBevelSeamContrast: +seamContrast.toFixed(4),
      pass:
        nanInfCount === 0 &&
        rubyAngle < 5.0 &&
        ambAngle < 5.0 &&
        seamContrast > 0.08,
    };
  });
  return { ...res, elapsedMs: Math.round(performance.now() - t0) };
}

async function runSceneBacklitTower(page, presets) {
  const t0 = performance.now();
  const res = await page.evaluate(async (presetList) => {
    const eng = window.__VITRAIL_ENGINE__;
    const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const worldToPixel = (wx, wy, wz, w, h) => {
      const vp = eng.camera.viewProjMatrix;
      const cx = vp[0] * wx + vp[4] * wy + vp[8] * wz + vp[12];
      const cy = vp[1] * wx + vp[5] * wy + vp[9] * wz + vp[13];
      const cw = vp[3] * wx + vp[7] * wy + vp[11] * wz + vp[15];
      return [
        Math.round(((cx / cw) * 0.5 + 0.5) * (w - 1)),
        Math.round((0.5 - (cy / cw) * 0.5) * (h - 1)),
      ];
    };

    const byPreset = {};
    let totalNanInf = 0;
    let allPresetsPass = true;

    for (const presetId of presetList) {
      eng.configureBenchScene('backlit-tower', presetId);
      for (let i = 0; i < 4; i++) await new Promise((r) => requestAnimationFrame(r));
      const hdr = await eng.renderer.readHdrResolveFloat32();
      const { width: w, height: h, rgba, nanInfCount } = hdr;
      totalNanInf += nanInfCount;

      const [x0, yTop] = worldToPixel(-0.22, 2.05, 0.0, w, h);
      const [x1, yBot] = worldToPixel(0.22, 0.50, 0.0, w, h);
      const pxMinX = Math.max(0, Math.min(x0, x1));
      const pxMaxX = Math.min(w - 1, Math.max(x0, x1));
      const pxMinY = Math.max(0, Math.min(yTop, yBot));
      const pxMaxY = Math.min(h - 1, Math.max(yTop, yBot));

      let sumL = 0, sumR = 0, sumG = 0, sumB = 0, count = 0;
      for (let py = pxMinY; py <= pxMaxY; py += 2) {
        for (let px = pxMinX; px <= pxMaxX; px += 2) {
          const idx = (py * w + px) * 4;
          if (rgba[idx + 3] < 6.0) {
            sumR += rgba[idx];
            sumG += rgba[idx + 1];
            sumB += rgba[idx + 2];
            sumL += lum(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
            count++;
          }
        }
      }
      const meanLuminance = sumL / Math.max(1, count);
      if (meanLuminance <= 0.045) allPresetsPass = false;
      byPreset[presetId] = {
        meanLuminance: +meanLuminance.toFixed(4),
        meanRGB: [
          +(sumR / Math.max(1, count)).toFixed(4),
          +(sumG / Math.max(1, count)).toFixed(4),
          +(sumB / Math.max(1, count)).toFixed(4),
        ],
        nanInfCount,
      };
    }

    // Dark-room linearity control: sunColor = [0,0,0], skyTint scaled [1.0, 0.5, 0.25, 0.0]
    const scales = [1.0, 0.5, 0.25, 0.0];
    const darkLums = [];
    for (const s of scales) {
      eng.configureBenchScene('backlit-tower', presetList[0], s);
      for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
      const hdr = await eng.renderer.readHdrResolveFloat32();
      const { width: w, height: h, rgba } = hdr;
      const [x0, yTop] = worldToPixel(-0.22, 2.05, 0.0, w, h);
      const [x1, yBot] = worldToPixel(0.22, 0.50, 0.0, w, h);
      const pxMinX = Math.max(0, Math.min(x0, x1));
      const pxMaxX = Math.min(w - 1, Math.max(x0, x1));
      const pxMinY = Math.max(0, Math.min(yTop, yBot));
      const pxMaxY = Math.min(h - 1, Math.max(yTop, yBot));
      let sumL = 0, count = 0;
      for (let py = pxMinY; py <= pxMaxY; py += 2) {
        for (let px = pxMinX; px <= pxMaxX; px += 2) {
          const idx = (py * w + px) * 4;
          if (rgba[idx + 3] < 6.0) {
            sumL += lum(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
            count++;
          }
        }
      }
      darkLums.push(+(sumL / Math.max(1, count)).toFixed(6));
    }

    // Linear regression R^2 across scales vs darkLums
    const meanX = scales.reduce((a, b) => a + b, 0) / scales.length;
    const meanY = darkLums.reduce((a, b) => a + b, 0) / darkLums.length;
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < scales.length; i++) {
      const dx = scales[i] - meanX;
      const dy = darkLums[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const rSquared = denX > 0 && denY > 0 ? (num * num) / (denX * denY) : 0;
    const zeroEmissionLuminance = darkLums[darkLums.length - 1];

    return {
      byPreset,
      darkroomScales: scales,
      darkroomLuminances: darkLums,
      darkroomLinearityR2: +rSquared.toFixed(6),
      zeroLightLuminance: zeroEmissionLuminance,
      pass:
        totalNanInf === 0 &&
        allPresetsPass &&
        rSquared > 0.999 &&
        zeroEmissionLuminance < 1e-5,
    };
  }, presets);
  return { ...res, elapsedMs: Math.round(performance.now() - t0) };
}

async function runSceneCausticPool(page, presets) {
  const t0 = performance.now();
  const res = await page.evaluate(async (presetList) => {
    const eng = window.__VITRAIL_ENGINE__;
    const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const worldToPixel = (wx, wy, wz, w, h) => {
      const vp = eng.camera.viewProjMatrix;
      const cx = vp[0] * wx + vp[4] * wy + vp[8] * wz + vp[12];
      const cy = vp[1] * wx + vp[5] * wy + vp[9] * wz + vp[13];
      const cw = vp[3] * wx + vp[7] * wy + vp[11] * wz + vp[15];
      return [
        Math.round(((cx / cw) * 0.5 + 0.5) * (w - 1)),
        Math.round((0.5 - (cy / cw) * 0.5) * (h - 1)),
      ];
    };

    const byPreset = {};
    let allPass = true;

    for (const presetId of presetList) {
      eng.configureBenchScene('caustic-pool', presetId);
      const preset = eng.renderer.lightingPreset;

      // 1. Tower sunlit tabletop
      eng.renderer.benchHideBlocks = false;
      eng.renderer.lightingOverride = null;
      for (let i = 0; i < 4; i++) await new Promise((r) => requestAnimationFrame(r));
      const hdrTower = await eng.renderer.readHdrResolveFloat32();

      // 2. Tower ambient-only tabletop (cancels exact AO + walnut grain!)
      eng.renderer.benchHideBlocks = false;
      eng.renderer.lightingOverride = { sunColor: [0, 0, 0] };
      for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
      const hdrDarkTower = await eng.renderer.readHdrResolveFloat32();

      // 3. Unobstructed sunlit tabletop
      eng.renderer.benchHideBlocks = true;
      eng.renderer.lightingOverride = null;
      for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
      const hdrEmpty = await eng.renderer.readHdrResolveFloat32();

      // 4. Unobstructed ambient-only tabletop
      eng.renderer.benchHideBlocks = true;
      eng.renderer.lightingOverride = { sunColor: [0, 0, 0] };
      for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
      const hdrDarkEmpty = await eng.renderer.readHdrResolveFloat32();

      eng.renderer.benchHideBlocks = false;
      eng.renderer.lightingOverride = null;

      const w = hdrTower.width, h = hdrTower.height;
      const az = (preset.sunAzimuthDeg * Math.PI) / 180;
      const el = (preset.sunElevationDeg * Math.PI) / 180;
      const lx = Math.cos(el) * Math.sin(az);
      const ly = Math.sin(el);
      const lz = Math.cos(el) * Math.cos(az);
      const shLen = Math.hypot(lx, lz);
      const dirX = -lx / shLen, dirZ = -lz / shLen;
      const perpX = -dirZ, perpZ = dirX;
      const maxShadowDist = 2.70 * (shLen / ly);

      const sampleSolarRatioAt = (wx, wz) => {
        if (Math.hypot(wx, wz) > 4.18) return null;
        const [px, py] = worldToPixel(wx, 0.0, wz, w, h);
        if (px < 3 || px >= w - 3 || py < 3 || py >= h - 3) return null;
        const idx = (py * w + px) * 4;
        if (hdrTower.rgba[idx + 3] < hdrEmpty.rgba[idx + 3] - 0.06) return null;
        const lTow = lum(hdrTower.rgba[idx], hdrTower.rgba[idx + 1], hdrTower.rgba[idx + 2]);
        const lDTow = lum(hdrDarkTower.rgba[idx], hdrDarkTower.rgba[idx + 1], hdrDarkTower.rgba[idx + 2]);
        const lEmp = lum(hdrEmpty.rgba[idx], hdrEmpty.rgba[idx + 1], hdrEmpty.rgba[idx + 2]);
        const lDEmp = lum(hdrDarkEmpty.rgba[idx], hdrDarkEmpty.rgba[idx + 1], hdrDarkEmpty.rgba[idx + 2]);
        const solarEmp = lEmp - lDEmp;
        if (solarEmp < 0.008) return null;
        const solarTow = Math.max(0.0, lTow - lDTow);
        return { ratio: solarTow / solarEmp, solarTow, solarEmp };
      };

      const interiorRatios = [];
      let peakRibbonRatio = 0;
      let sumFootprintSolar = 0;
      let sumUnobstructedSolar = 0;

      for (let s = 0.45; s <= maxShadowDist + 0.35; s += 0.025) {
        for (let v = -0.55; v <= 0.55; v += 0.025) {
          const wx = dirX * s + perpX * v;
          const wz = dirZ * s + perpZ * v;
          if (Math.hypot(wx, wz) > 3.85) continue;
          const samp = sampleSolarRatioAt(wx, wz);
          if (!samp) continue;
          sumFootprintSolar += samp.solarTow;
          sumUnobstructedSolar += samp.solarEmp;
          if (samp.ratio > peakRibbonRatio) peakRibbonRatio = samp.ratio;
          if (s >= 0.65 && s <= maxShadowDist * 0.85 && Math.abs(v) <= 0.26) {
            interiorRatios.push(samp.ratio);
          }
        }
      }

      interiorRatios.sort((a, b) => a - b);
      const minInterior = interiorRatios[0] ?? 0;
      const medianInterior = interiorRatios[Math.floor(interiorRatios.length * 0.5)] ?? 1;
      const minOverMedian = minInterior / Math.max(1e-5, medianInterior);
      const fluxConservationRatio = sumFootprintSolar / Math.max(1e-5, sumUnobstructedSolar);

      // Contact hardening:
      // 1) Layer-0 base contact edge at z = sign(lz)*0.358 (+0.358 dm, where caster height h <= 2.0 mm!)
      const sx = -Math.sign(lx);
      const zContact = Math.sign(lz) * 0.358;
      const prof0 = [];
      for (let u = 0.360; u <= 0.435; u += 0.001) {
        const samp = sampleSolarRatioAt(sx * u, zContact);
        if (samp) prof0.push({ uMm: u * 100, r: samp.ratio });
      }
      let rMin0 = 1.0;
      for (const p of prof0) {
        if (p.uMm <= 37.8) rMin0 = Math.min(rMin0, p.r);
      }
      const lo0 = rMin0 + 0.20 * (1.0 - rMin0);
      const hi0 = rMin0 + 0.80 * (1.0 - rMin0);
      let uLo0 = null, uHi0 = null;
      for (const p of prof0) {
        if (p.uMm >= 36.8 && uLo0 === null && p.r >= lo0) uLo0 = p.uMm;
        if (uLo0 !== null && uHi0 === null && p.r >= hi0) {
          uHi0 = p.uMm;
          break;
        }
      }
      const layer0WidthMm = uLo0 !== null && uHi0 !== null ? +(uHi0 - uLo0).toFixed(2) : 99;

      // 2) Layer-12 penumbra width at h = 1.85 dm: x_edge = 0.375 + (|lx|/ly)*1.85, z_mid = (-lz/ly)*1.85
      // Average 3 adjacent z-slices (+/- 4 mm) to suppress narrow local waterglass caustic ripples while measuring true PCSS penumbra!
      const uEdge12 = 0.375 + (Math.abs(lx) / ly) * 1.85;
      const zMid12 = (-lz / ly) * 1.85;
      const prof12 = [];
      for (let u = uEdge12 - 0.38; u <= uEdge12 + 0.35; u += 0.002) {
        let sumR = 0, cnt = 0;
        for (const dz of [-0.04, 0.0, 0.04]) {
          const samp = sampleSolarRatioAt(sx * u, zMid12 + dz);
          if (samp) {
            sumR += Math.min(samp.ratio, 1.02);
            cnt++;
          }
        }
        if (cnt > 0) prof12.push({ uMm: u * 100, r: sumR / cnt });
      }
      let rMin12 = 1.0;
      let idxMin12 = 0;
      for (let i = 0; i < prof12.length; i++) {
        if (prof12[i].uMm < uEdge12 * 100 && prof12[i].r < rMin12) {
          rMin12 = prof12[i].r;
          idxMin12 = i;
        }
      }
      const lo12 = rMin12 + 0.20 * (1.0 - rMin12);
      const hi12 = rMin12 + 0.80 * (1.0 - rMin12);
      let uLo12 = null, uHi12 = null;
      for (let i = idxMin12; i < prof12.length; i++) {
        const p = prof12[i];
        if (uLo12 === null && p.r >= lo12) uLo12 = p.uMm;
        if (uLo12 !== null && uHi12 === null && p.r >= hi12) {
          uHi12 = p.uMm;
          break;
        }
      }
      const layer12WidthMm = uLo12 !== null && uHi12 !== null ? +(uHi12 - uLo12).toFixed(2) : 0;

      // High-frequency periodic comb/barcode grating check along centerline (v=0) and flanks (v=+/-0.26), step = 4 mm:
      const evalProfileCombRatio = (vOff) => {
        const prof = [];
        for (let s = 0.70; s <= Math.min(3.40, maxShadowDist * 0.82); s += 0.04) {
          const samp = sampleSolarRatioAt(dirX * s + perpX * vOff, dirZ * s + perpZ * vOff);
          if (samp) prof.push(samp.ratio);
        }
        let sumSq = 0;
        let cnt = 0;
        for (let i = 1; i + 1 < prof.length; i++) {
          const d2 = prof[i + 1] - 2.0 * prof[i] + prof[i - 1];
          sumSq += d2 * d2;
          cnt++;
        }
        return cnt > 0 ? Math.sqrt(sumSq / cnt) / Math.max(1e-5, medianInterior) : 0;
      };
      const combRippleRatio = evalProfileCombRatio(0.0);
      const flankCombRippleRatio = Math.max(evalProfileCombRatio(-0.26), evalProfileCombRatio(0.26));

      const presetPass =
        hdrTower.nanInfCount === 0 &&
        minOverMedian > 0.35 &&
        peakRibbonRatio >= 1.35 &&
        fluxConservationRatio <= 0.94 &&
        combRippleRatio < 0.10 &&
        flankCombRippleRatio < 0.10 &&
        layer0WidthMm < 2.5 &&
        layer12WidthMm > 12.0;
      if (!presetPass) allPass = false;

      byPreset[presetId] = {
        nanInfCount: hdrTower.nanInfCount,
        minOverMedian: +minOverMedian.toFixed(4),
        peakRibbonRatio: +peakRibbonRatio.toFixed(4),
        fluxConservationRatio: +fluxConservationRatio.toFixed(4),
        combRippleRatio: +combRippleRatio.toFixed(4),
        flankCombRippleRatio: +flankCombRippleRatio.toFixed(4),
        layer0ContactWidthMm: layer0WidthMm,
        layer12PenumbraWidthMm: layer12WidthMm,
        pass: presetPass,
      };
    }

    return { byPreset, pass: allPass };
  }, presets);
  return { ...res, elapsedMs: Math.round(performance.now() - t0) };
}

async function captureContactSheet(page, outputPath) {
  const t0 = performance.now();
  const shots = [
    { scene: 'morning-front', preset: 'lancet-morning', label: '1. Morning Lancet Front (motifId=0 Waterglass)' },
    { scene: 'backlit-tower', preset: 'golden-vespers', label: '2. Backlit Tower (Colored Solar Transmittance)' },
    { scene: 'caustic-pool', preset: 'lancet-morning', label: '3. Caustic Pool (Zero Plaid + Jacobian Ribbons)' },
    { scene: 'collapse-300', preset: 'lancet-morning', label: '4. 300+ Shard Collapse (3-Slice Depth Peeling)' },
  ];

  const tileBuffers = [];
  let activeShardCount = 0;
  for (const shot of shots) {
    const info = await page.evaluate(async ({ scene, preset }) => {
      const eng = window.__VITRAIL_ENGINE__;
      eng.configureBenchScene(scene, preset);
      for (let i = 0; i < 5; i++) await new Promise((r) => requestAnimationFrame(r));
      return { shards: eng.physics.shards.length, intact: eng.physics.blocks.filter((b) => !b.shattered).length };
    }, shot);
    if (shot.scene === 'collapse-300') activeShardCount = info.shards;
    const buf = await page.screenshot({ type: 'png' });
    tileBuffers.push({ label: shot.label, base64: buf.toString('base64') });
  }

  // Stitch 2x2 contact sheet inside Chromium 2D canvas at 1056x660 (<= 2048 px max edge, < 1.5 MB lossless PNG)
  const sheetBase64 = await page.evaluate(async (tiles) => {
    const W = 1056;
    const H = 660;
    const tw = W / 2;
    const th = H / 2;
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0b0a0e';
    ctx.fillRect(0, 0, W, H);

    for (let i = 0; i < tiles.length; i++) {
      const img = new Image();
      img.src = 'data:image/png;base64,' + tiles[i].base64;
      await new Promise((r) => {
        img.onload = r;
      });
      const x = (i % 2) * tw;
      const y = Math.floor(i / 2) * th;
      ctx.drawImage(img, x, y, tw, th);

      // Subtle caption badge
      ctx.fillStyle = 'rgba(10, 10, 14, 0.76)';
      ctx.fillRect(x + 12, y + th - 36, 430, 26);
      ctx.fillStyle = '#f3ece0';
      ctx.font = '600 13px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(tiles[i].label, x + 20, y + th - 18);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tw, 0);
    ctx.lineTo(tw, H);
    ctx.moveTo(0, th);
    ctx.lineTo(W, th);
    ctx.stroke();

    const pngUrl = c.toDataURL('image/png');
    return { data: pngUrl.split(',')[1], width: W, height: H, format: 'png' };
  }, tileBuffers);

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const rawBytes = Buffer.from(sheetBase64.data, 'base64');
  fs.writeFileSync(outputPath, rawBytes);

  return {
    outputPath,
    width: sheetBase64.width,
    height: sheetBase64.height,
    sizeBytes: rawBytes.byteLength,
    collapseActiveShards: activeShardCount,
    pass:
      sheetBase64.width <= 2048 &&
      sheetBase64.height <= 2048 &&
      rawBytes.byteLength < 1500000 &&
      activeShardCount >= 300,
    elapsedMs: Math.round(performance.now() - t0),
  };
}

async function main() {
  const startAll = performance.now();
  const viteProc = await ensureDevServer(baseUrl);
  const { chromium } = await resolvePlaywright();

  let browser;
  try {
    browser = await chromium.launch({
      headless: false,
      args: [
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--enable-dawn-features=allow_unsafe_apis',
      ],
    });
    const context = await browser.newContext({
      viewport: { width: 960, height: 600 },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();
    page.on('pageerror', (err) => console.error('[bench:pageerror]', err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error' || msg.type() === 'warning') {
        const txt = msg.text();
        if (txt.includes('WGSL') || txt.includes('WebGPU') || txt.includes('Validation') || msg.type() === 'error') {
          console.error(`[bench:${msg.type()}]`, txt);
        }
      }
    });
    await page.goto(`${baseUrl}/?bench=morning-front`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => Boolean(window.__VITRAIL_ENGINE__?.renderer), null, {
      timeout: 15000,
    });

    const presets = presetFilter ? [presetFilter] : ['lancet-morning', 'golden-vespers'];
    const summary = { targetScene, results: {} };
    let overallPass = true;

    if (targetScene === 'two-block-cross' || targetScene === 'all') {
      summary.results['two-block-cross'] = await runSceneTwoBlockCross(page);
      overallPass = overallPass && summary.results['two-block-cross'].pass;
    }
    if (targetScene === 'backlit-tower' || targetScene === 'all') {
      summary.results['backlit-tower'] = await runSceneBacklitTower(page, presets);
      overallPass = overallPass && summary.results['backlit-tower'].pass;
    }
    if (targetScene === 'caustic-pool' || targetScene === 'all') {
      summary.results['caustic-pool'] = await runSceneCausticPool(page, presets);
      overallPass = overallPass && summary.results['caustic-pool'].pass;
    }
    if (targetScene === 'collapse-300') {
      const info = await page.evaluate(async () => {
        const eng = window.__VITRAIL_ENGINE__;
        eng.configureBenchScene('collapse-300', 'lancet-morning');
        for (let i = 0; i < 4; i++) await new Promise((r) => requestAnimationFrame(r));
        const hdr = await eng.renderer.readHdrResolveFloat32();
        return {
          activeShards: eng.physics.shards.length,
          nanInfCount: hdr.nanInfCount,
          pass: eng.physics.shards.length >= 300 && hdr.nanInfCount === 0,
        };
      });
      summary.results['collapse-300'] = info;
      overallPass = overallPass && info.pass;
    }
    if (targetScene === 'contact-sheet' || targetScene === 'all') {
      summary.results['contact-sheet'] = await captureContactSheet(page, outSheetPath);
      overallPass = overallPass && summary.results['contact-sheet'].pass;
    }

async function runTier3Telemetry(page) {
  const t0 = performance.now();
  // Unlock WebAudio context via real browser click on canvas
  await page.mouse.click(480, 300);
  await page.waitForTimeout(80);

  const res = await page.evaluate(async () => {
    const eng = window.__VITRAIL_ENGINE__;
    const dev = eng.renderer.device;
    const stats = (arr) => {
      if (arr.length === 0) return { mean: 0, p50: 0, p95: 0, max: 0 };
      const s = [...arr].sort((a, b) => a - b);
      const mean = s.reduce((a, b) => a + b, 0) / s.length;
      return {
        mean: +mean.toFixed(3),
        p50: +s[Math.floor(s.length * 0.5)].toFixed(3),
        p95: +s[Math.min(s.length - 1, Math.floor(s.length * 0.95))].toFixed(3),
        max: +s[s.length - 1].toFixed(3),
      };
    };

    // Query actual WebGPU adapter hardware/software identity
    const adapter = await navigator.gpu.requestAdapter();
    const aInfo = adapter?.info ?? {};
    const gpuAdapterInfo = {
      vendor: aInfo.vendor || 'unknown',
      architecture: aInfo.architecture || 'unknown',
      device: aInfo.device || 'unknown',
      description: aInfo.description || 'unknown',
      isFallbackAdapter: Boolean(adapter?.isFallbackAdapter),
    };

    // Measure 60 live rAF frame deltas first while engine rAF loop is running
    eng.resetGame('party', 2);
    eng.audio.setMuted(false);
    eng.audio.unlock();
    eng.benchPaused = false;
    await new Promise((r) => requestAnimationFrame(r));
    const rafDeltas = [];
    let prevRaf = performance.now();
    for (let i = 0; i < 60; i++) {
      const now = await new Promise((r) => requestAnimationFrame(r));
      rafDeltas.push(now - prevRaf);
      prevRaf = now;
    }

    // Pause background rAF so our 120-frame synchronous CPU + Metal queue.onSubmittedWorkDone() harness
    // measures exact per-frame hardware turnaround without double-submitting frames.
    eng.running = false;
    await dev.queue.onSubmittedWorkDone();

    // Instrument createCommandEncoder to count copyTextureToTexture & secondary shard MSAA passes
    let copyTexCalls = 0;
    let secondaryShardPasses = 0;
    const origCreateEncoder = dev.createCommandEncoder.bind(dev);
    dev.createCommandEncoder = (desc) => {
      const enc = origCreateEncoder(desc);
      const origCopy = enc.copyTextureToTexture.bind(enc);
      enc.copyTextureToTexture = (...args) => {
        copyTexCalls++;
        return origCopy(...args);
      };
      const origBeginPass = enc.beginRenderPass.bind(enc);
      enc.beginRenderPass = (rpDesc) => {
        if (rpDesc?.label && String(rpDesc.label).includes('Shard')) {
          secondaryShardPasses++;
        } else if (
          rpDesc?.colorAttachments?.[0]?.loadOp === 'load' &&
          rpDesc?.depthStencilAttachment?.depthLoadOp === 'load'
        ) {
          secondaryShardPasses++;
        }
        return origBeginPass(rpDesc);
      };
      return enc;
    };

    let simTime = 1.0;

    // Measure 10 onscreen frames first to record any macOS occluded-window CAMetalLayer wait,
    // then bind an identical-format offscreen composite target during the 120-frame GPU timestamp loops
    // so CAMetalLayer.nextDrawable occlusion throttling does not distort hardware GPU pass timestamps.
    eng.physics.rebuildTower('party', 2);
    for (let i = 0; i < 40; i++) eng.physics.step(1 / 60);
    const intactAsleepSwapchainWaitMs = [];
    for (let i = 0; i < 10; i++) {
      const f0 = performance.now();
      eng.renderer.render(eng.physics, eng.camera, (simTime += 1 / 60));
      await dev.queue.onSubmittedWorkDone();
      intactAsleepSwapchainWaitMs.push(performance.now() - f0);
    }

    const benchCompositeTex = dev.createTexture({
      size: [eng.canvas.width, eng.canvas.height],
      format: eng.renderer.canvasFormat || navigator.gpu.getPreferredCanvasFormat(),
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const origGetCurrentTexture = eng.renderer.context.getCurrentTexture.bind(eng.renderer.context);
    eng.renderer.context.getCurrentTexture = () => benchCompositeTex;

    // -------------------------------------------------------------------------
    // Regime 1: Intact 18L Tower Asleep (0 shards, 120 frames @ 2x Retina)
    // -------------------------------------------------------------------------
    for (let i = 0; i < 5; i++) {
      eng.renderer.render(eng.physics, eng.camera, (simTime += 1 / 60));
      await dev.queue.onSubmittedWorkDone();
    }
    copyTexCalls = 0;
    secondaryShardPasses = 0;
    const intactAsleepPhysMs = [];
    const intactAsleepGpuHwMs = [];
    const intactAsleepCpuPlusHwGpuMs = [];
    for (let i = 0; i < 120; i++) {
      const p0 = performance.now();
      eng.physics.step(1 / 60);
      const pMs = performance.now() - p0;
      intactAsleepPhysMs.push(pMs);
      const r0 = performance.now();
      eng.renderer.render(eng.physics, eng.camera, (simTime += 1 / 60));
      const encMs = performance.now() - r0;
      await dev.queue.onSubmittedWorkDone();
      await Promise.resolve();
      const hwGpu = eng.renderer.lastGpuFrameMs;
      intactAsleepGpuHwMs.push(hwGpu);
      intactAsleepCpuPlusHwGpuMs.push(pMs + encMs + hwGpu);
    }
    const zeroShardCopyCount = copyTexCalls;
    const zeroShardSecondaryPasses = secondaryShardPasses;

    // -------------------------------------------------------------------------
    // Regime 2: Intact 18L Active Axial Pull & Extracted (120 frames @ 2x Retina)
    // -------------------------------------------------------------------------
    eng.physics.rebuildTower('party', 2);
    for (let i = 0; i < 40; i++) eng.physics.step(1 / 60);
    let targetBlk = eng.physics.blocks.find(
      (b) => b.assignedLayer >= 4 && b.assignedLayer <= 12 && b.assignedLayer % 2 === 0 && b.isLooseSlot && b.assignedSlot === 1
    );
    if (!targetBlk) {
      targetBlk = eng.physics.blocks.find((b) => b.assignedLayer === 6 && b.assignedSlot === 1);
    }
    const isOddLayer = targetBlk.assignedLayer % 2 === 1;
    const localGrab = [0.36, 0, 0];
    const startWorld = [
      targetBlk.position[0] + (isOddLayer ? 0 : 0.36),
      targetBlk.position[1],
      targetBlk.position[2] + (isOddLayer ? -0.36 : 0),
    ];
    eng.physics.startGrabBlock(targetBlk.id, localGrab, startWorld);

    const pullingPhysMs = [];
    const extractedPhysMs = [];
    const activePullCpuPlusHwGpuMs = [];
    for (let i = 0; i < 120; i++) {
      const pullDist = Math.min(1.15, ((i + 1) / 48) * 1.15);
      if (eng.physics.turnPhase === 'PULLING') {
        eng.physics.updateGrabTarget([
          startWorld[0] + (isOddLayer ? 0 : pullDist),
          startWorld[1],
          startWorld[2] + (isOddLayer ? -pullDist : 0),
        ]);
      } else if (eng.physics.turnPhase === 'EXTRACTED') {
        eng.physics.updateCarriedHover([0.0, 4.5, 2.0], [0.0, -0.78, -0.62]);
      }
      const p0 = performance.now();
      eng.physics.step(1 / 60);
      const pMs = performance.now() - p0;
      if (eng.physics.turnPhase === 'PULLING') {
        pullingPhysMs.push(pMs);
      } else {
        extractedPhysMs.push(pMs);
      }
      const r0 = performance.now();
      eng.renderer.render(eng.physics, eng.camera, (simTime += 1 / 60));
      const encMs = performance.now() - r0;
      await dev.queue.onSubmittedWorkDone();
      await Promise.resolve();
      activePullCpuPlusHwGpuMs.push(pMs + encMs + eng.renderer.lastGpuFrameMs);
    }
    const extractedReached = eng.physics.turnPhase === 'EXTRACTED';

    // -------------------------------------------------------------------------
    // Regime 3: Peak Collapse (300-400 active shards, 120 peak frames, WebAudio live)
    // -------------------------------------------------------------------------
    eng.audio.setMuted(false);
    eng.audio.unlock();
    eng.physics.rebuildTower('sandbox', 2);
    for (let i = 0; i < 20; i++) eng.physics.step(1 / 60);

    let frameVoronoiMs = 0;
    let maxSingleFrameVoronoiMs = 0;
    const origShatter = eng.physics.shatterBlock.bind(eng.physics);
    eng.physics.shatterBlock = (...args) => {
      const v0 = performance.now();
      const out = origShatter(...args);
      frameVoronoiMs += performance.now() - v0;
      return out;
    };

    eng.physics.triggerSandboxCollapseImpulse();
    const collapsePeakWindowHwGpuMs = [];
    const collapsePeakWindowCpuPlusHwGpuMs = [];
    const collapsePeakWindowSwapchainMs = [];
    const collapsePhysMs = [];
    let peakShards = 0;
    let peakVoices = 0;

    // Step 160 frames (topple + shatter + 300-400+ shard pile); capture the 120f peak shard window
    for (let i = 0; i < 160; i++) {
      frameVoronoiMs = 0;
      const f0 = performance.now();
      const p0 = performance.now();
      eng.physics.step(1 / 60);
      const pMs = performance.now() - p0;
      collapsePhysMs.push(pMs);
      if (frameVoronoiMs > maxSingleFrameVoronoiMs) {
        maxSingleFrameVoronoiMs = frameVoronoiMs;
      }
      const r0 = performance.now();
      eng.renderer.render(eng.physics, eng.camera, (simTime += 1 / 60));
      const encMs = performance.now() - r0;
      await dev.queue.onSubmittedWorkDone();
      await Promise.resolve();
      const swapMs = performance.now() - f0;
      const hwGpu = eng.renderer.lastGpuFrameMs;
      const sc = eng.physics.shards.length;
      if (sc > peakShards) peakShards = sc;
      if (i >= 40) {
        collapsePeakWindowHwGpuMs.push(hwGpu);
        collapsePeakWindowCpuPlusHwGpuMs.push(pMs + encMs + hwGpu);
        collapsePeakWindowSwapchainMs.push(swapMs);
      }
      const vc = eng.audio.getActiveVoiceCount();
      if (vc > peakVoices) peakVoices = vc;
    }

    eng.renderer.context.getCurrentTexture = origGetCurrentTexture;
    eng.renderer.render(eng.physics, eng.camera, (simTime += 1 / 60));
    await dev.queue.onSubmittedWorkDone();
    benchCompositeTex.destroy();

    eng.physics.shatterBlock = origShatter;
    dev.createCommandEncoder = origCreateEncoder;

    // Wait 700 ms for modal decay cleanup timers to disconnect WebAudio nodes
    await new Promise((r) => setTimeout(r, 700));
    const postDecayConnectedNodes = eng.audio.getConnectedTransientNodes();

    eng.running = true;
    eng.lastFrameMs = performance.now();
    requestAnimationFrame(eng.tick);

    const asleepStats = stats(intactAsleepCpuPlusHwGpuMs);
    const asleepPhysStats = stats(intactAsleepPhysMs);
    const pullPhysStats = stats(pullingPhysMs);
    const extPhysStats = stats(extractedPhysMs);
    const activeTotalStats = stats(activePullCpuPlusHwGpuMs);
    const collapsePeakStats = stats(collapsePeakWindowCpuPlusHwGpuMs);

    return {
      gpuAdapterInfo,
      supportsHardwareTimestamps: eng.renderer.supportsTimestamps,
      framebufferSize: [eng.canvas.width, eng.canvas.height],
      audioContextState: eng.audio.ctx?.state ?? 'unknown',
      liveRafDeltaMs: stats(rafDeltas),
      intact18LAsleep: {
        frames: 120,
        snapshotTextureCopies: zeroShardCopyCount,
        secondaryShardPasses: zeroShardSecondaryPasses,
        physicsStepMs: asleepPhysStats,
        metalGpuTimestampMs: stats(intactAsleepGpuHwMs),
        cpuPlusMetalGpuFrameMs: asleepStats,
        occludedSwapchainWaitMs: stats(intactAsleepSwapchainWaitMs),
      },
      intact18LActivePull: {
        frames: 120,
        extractedReached,
        pullingPhysicsStepMs: pullPhysStats,
        extractedPhysicsStepMs: extPhysStats,
        cpuPlusMetalGpuFrameMs: activeTotalStats,
      },
      peakCollapse: {
        peakActiveShards: peakShards,
        framesInPeakWindow: collapsePeakWindowCpuPlusHwGpuMs.length,
        maxSingleFrameVoronoiMs: +maxSingleFrameVoronoiMs.toFixed(3),
        physicsStepMs: stats(collapsePhysMs),
        metalGpuTimestampMs: stats(collapsePeakWindowHwGpuMs),
        cpuPlusMetalGpuFrameMs: collapsePeakStats,
        occludedSwapchainWaitMs: stats(collapsePeakWindowSwapchainMs),
        peakConcurrentAudioVoices: peakVoices,
        postDecayConnectedNodes,
      },
      pass:
        eng.renderer.supportsTimestamps &&
        zeroShardCopyCount === 0 &&
        zeroShardSecondaryPasses === 0 &&
        asleepPhysStats.mean < 0.05 &&
        asleepStats.mean < 6.5 &&
        extractedReached &&
        pullPhysStats.mean < 1.5 &&
        peakShards >= 300 &&
        collapsePeakStats.mean < 11.5 &&
        maxSingleFrameVoronoiMs <= 18.0 &&
        peakVoices <= 14 &&
        postDecayConnectedNodes === 0,
    };
  });

  return { ...res, elapsedMs: Math.round(performance.now() - t0) };
}

    if (targetScene === 'tier3-telemetry') {
      summary.results['tier3-telemetry'] = await runTier3Telemetry(page);
      overallPass = overallPass && summary.results['tier3-telemetry'].pass;
    }

    summary.overallPass = overallPass;
    summary.totalElapsedMs = Math.round(performance.now() - startAll);
    console.log(JSON.stringify(summary, null, 2));
    process.exitCode = overallPass ? 0 : 1;
  } finally {
    if (browser) await browser.close();
    if (viteProc) viteProc.kill('SIGTERM');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

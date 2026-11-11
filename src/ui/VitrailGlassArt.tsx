import React from 'react';

/**
 * Shared SVG definitions for mouth-blown waterglass ream striae, trapped air-seed bubbles,
 * and oblique window-plane specular sheen.
 */
export const VitrailSvgDefs: React.FC = () => (
  <svg
    width="0"
    height="0"
    style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }}
    aria-hidden="true"
  >
    <defs>
      {/* Subtle mouth-blown antique glass ream striae + micro-seed air bubbles */}
      <pattern
        id="vitrail-seed-pattern"
        width="68"
        height="68"
        patternUnits="userSpaceOnUse"
        patternTransform="rotate(-20)"
      >
        <path
          d="M-8,14 Q26,10 76,16"
          fill="none"
          stroke="rgba(255,253,245,0.11)"
          strokeWidth="1.6"
        />
        <path
          d="M-8,40 Q34,45 76,38"
          fill="none"
          stroke="rgba(255,253,245,0.07)"
          strokeWidth="1.1"
        />
        <path
          d="M-8,58 Q28,54 76,60"
          fill="none"
          stroke="rgba(0,0,0,0.10)"
          strokeWidth="1.3"
        />
        <circle
          cx="18"
          cy="22"
          r="1.1"
          fill="none"
          stroke="rgba(255,255,255,0.28)"
          strokeWidth="0.6"
        />
        <circle cx="51" cy="13" r="0.75" fill="rgba(255,255,255,0.22)" />
        <circle
          cx="39"
          cy="52"
          r="1.25"
          fill="none"
          stroke="rgba(255,255,255,0.24)"
          strokeWidth="0.6"
        />
        <circle cx="12" cy="55" r="0.65" fill="rgba(255,255,255,0.2)" />
      </pattern>

      {/* Diagonal window-sheet reflection gradient */}
      <linearGradient id="vitrail-sheet-sheen" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#ffffff" stopOpacity="0.28" />
        <stop offset="28%" stopColor="#fff9eb" stopOpacity="0.06" />
        <stop offset="52%" stopColor="#ffffff" stopOpacity="0.14" />
        <stop offset="78%" stopColor="#000000" stopOpacity="0.06" />
        <stop offset="100%" stopColor="#000000" stopOpacity="0.22" />
      </linearGradient>
    </defs>
  </svg>
);

/**
 * Pointed Lancet Arch Crown Mosaic (Top apex of the left architectural stained-glass window).
 * Shifts internal sky transmittance smoothly across Morning, Vespers, and Noon presets.
 */
export const LancetArchApexSvg: React.FC<{ presetId: string }> = ({ presetId }) => {
  const p =
    presetId === 'golden-vespers'
      ? {
          c1: '#9a3412',
          c2: '#d97706',
          c3: '#be185d',
          c4: '#7e22ce',
          c5: '#f59e0b',
          opal: '#fde68a',
          jewel: '#fffbeb',
        }
      : presetId === 'daylight-atelier'
      ? {
          c1: '#0284c7',
          c2: '#38bdf8',
          c3: '#f59e0b',
          c4: '#d97706',
          c5: '#fde047',
          opal: '#fffbeb',
          jewel: '#ffffff',
        }
      : presetId === 'high-clerestory'
      ? {
          c1: '#0369a1',
          c2: '#0d9488',
          c3: '#38bdf8',
          c4: '#10b981',
          c5: '#bae6fd',
          opal: '#f0fdfa',
          jewel: '#ffffff',
        }
      : {
          c1: '#1d4ed8',
          c2: '#3b82f6',
          c3: '#0284c7',
          c4: '#6d28d9',
          c5: '#f59e0b',
          opal: '#fef3c7',
          jewel: '#fde047',
        };

  return (
    <svg className="lancet-apex-svg" viewBox="0 0 244 62" aria-hidden="true">
      {/* Outer pointed Gothic/modernist lancet arch lead housing */}
      <path
        d="M 1,62 L 1,50 C 1,24 54,6 122,1 C 190,6 243,24 243,50 L 243,62 Z"
        fill="#100e14"
        stroke="rgba(232,216,186,0.38)"
        strokeWidth="1"
      />

      {/* Left outer spandrel shards with putty-meniscus edge depth */}
      <path
        d="M 3.5,60 L 3.5,50 C 3.5,33 26,18 62,10.2 L 74,35 L 36,60 Z"
        fill={p.c1}
        stroke="rgba(8,6,12,0.48)"
        strokeWidth="2.4"
      />
      <polygon
        points="38.5,60 75.5,36.5 96,60"
        fill={p.c3}
        stroke="rgba(8,6,12,0.45)"
        strokeWidth="2.2"
      />

      {/* Right outer spandrel shards with putty-meniscus edge depth */}
      <path
        d="M 240.5,60 L 240.5,50 C 240.5,33 218,18 182,10.2 L 170,35 L 208,60 Z"
        fill={p.c4}
        stroke="rgba(8,6,12,0.48)"
        strokeWidth="2.4"
      />
      <polygon
        points="205.5,60 168.5,36.5 148,60"
        fill={p.c2}
        stroke="rgba(8,6,12,0.45)"
        strokeWidth="2.2"
      />

      {/* Upper crown flanking shards */}
      <path
        d="M 65,9.5 C 84,5.2 103,3.2 120.8,2.2 L 120.8,21 L 76.5,33.5 Z"
        fill={p.c2}
        stroke="rgba(8,6,12,0.46)"
        strokeWidth="2.4"
      />
      <path
        d="M 179,9.5 C 160,5.2 141,3.2 123.2,2.2 L 123.2,21 L 167.5,33.5 Z"
        fill={p.c5}
        stroke="rgba(8,6,12,0.46)"
        strokeWidth="2.4"
      />

      {/* Central Rose Lozenge & Inlaid Opal Medallion */}
      <polygon points="122,22.5 166,35 145.5,60 98.5,60 78,35" fill={p.opal} fillOpacity="0.90" />
      <polygon
        points="122,26 146,42 122,58 98,42"
        fill={p.c5}
        stroke="#100e14"
        strokeWidth="2"
      />
      <polygon
        points="122,33.5 132.5,42 122,50.5 111.5,42"
        fill={p.jewel}
        stroke="#100e14"
        strokeWidth="1.6"
      />

      {/* Crisp internal bevel highlights on glass shards */}
      <path
        d="M 5,50 C 5,34 27,19.5 61,11.2"
        fill="none"
        stroke="rgba(255,255,255,0.42)"
        strokeWidth="0.9"
      />
      <path
        d="M 66,10.2 C 85,6 103,4.2 120,3.8"
        fill="none"
        stroke="rgba(255,255,255,0.48)"
        strokeWidth="0.9"
      />

      {/* Continuous mouth-blown waterglass texture + sheet reflection */}
      <path
        d="M 2.5,61 L 2.5,50 C 2.5,23 56,4 122,2 C 188,4 241.5,23 241.5,50 L 241.5,61 Z"
        fill="url(#vitrail-seed-pattern)"
      />
      <path
        d="M 2.5,61 L 2.5,50 C 2.5,23 56,4 122,2 C 188,4 241.5,23 241.5,50 L 241.5,61 Z"
        fill="url(#vitrail-sheet-sheen)"
      />
    </svg>
  );
};

/**
 * Full-bleed stained-glass sky mosaic background for each of the 3 Sun Lighting Panes
 * (Morning Lancet, Golden Vespers, Clerestory Noon).
 */
export const ClerestoryPaneMosaicSvg: React.FC<{ presetId: string; active: boolean }> = ({
  presetId,
  active,
}) => {
  const p =
    presetId === 'golden-vespers'
      ? {
          topL: '#9a3412',
          topR: '#d97706',
          midL: '#be185d',
          midR: '#6d28d9',
          sun: '#fde68a',
          band: active ? 'rgba(42, 18, 12, 0.78)' : 'rgba(28, 22, 24, 0.74)',
        }
      : presetId === 'daylight-atelier'
      ? {
          topL: '#38bdf8',
          topR: '#fde68a',
          midL: '#bae6fd',
          midR: '#f59e0b',
          sun: '#ffffff',
          band: active ? 'rgba(44, 34, 18, 0.80)' : 'rgba(28, 25, 20, 0.75)',
        }
      : presetId === 'high-clerestory'
      ? {
          topL: '#0284c7',
          topR: '#2dd4bf',
          midL: '#0d9488',
          midR: '#38bdf8',
          sun: '#ffffff',
          band: active ? 'rgba(10, 38, 44, 0.78)' : 'rgba(20, 28, 32, 0.74)',
        }
      : {
          topL: '#1d4ed8',
          topR: '#60a5fa',
          midL: '#2563eb',
          midR: '#f59e0b',
          sun: '#fef08a',
          band: active ? 'rgba(14, 28, 62, 0.78)' : 'rgba(20, 24, 34, 0.74)',
        };

  const sunCx =
    presetId === 'lancet-morning'
      ? 24
      : presetId === 'daylight-atelier'
      ? 30
      : presetId === 'golden-vespers'
      ? 54
      : 39;
  const sunCy = presetId === 'high-clerestory' ? 13 : presetId === 'daylight-atelier' ? 15 : 18;

  return (
    <svg
      className="full-pane-mosaic-svg"
      viewBox="0 0 78 58"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <g opacity={active ? 1 : 0.62}>
        <polygon points="0,0 44,0 31,22 0,19" fill={p.topL} />
        <polygon points="46,0 78,0 78,21 33,22" fill={p.topR} />
        <polygon points="0,21 32,24 38,39 0,39" fill={p.midL} />
        <polygon points="34,24 78,23 78,39 40,39" fill={p.midR} />
      </g>
      {/* Lead cames between sky shards */}
      <line x1="45" y1="0" x2="32" y2="23" stroke="#100e14" strokeWidth="1.8" />
      <line x1="0" y1="20" x2="78" y2="22" stroke="#100e14" strokeWidth="1.8" />
      <line x1="32" y1="23" x2="39" y2="39" stroke="#100e14" strokeWidth="1.8" />
      {/* Inlaid Sun Rondel */}
      <circle
        cx={sunCx}
        cy={sunCy}
        r={active ? 6.2 : 5.2}
        fill={p.sun}
        fillOpacity={active ? 0.98 : 0.78}
        stroke="#100e14"
        strokeWidth="1.8"
      />
      {/* Lower inlaid smoked-glass label cartouche */}
      <rect x="0" y="39" width="78" height="19" fill={p.band} />
      <line x1="0" y1="39" x2="78" y2="39" stroke="#100e14" strokeWidth="1.8" />
      {/* Waterglass & bevel highlights */}
      <rect x="0" y="0" width="78" height="58" fill="url(#vitrail-seed-pattern)" />
      <rect x="0" y="0" width="78" height="58" fill="url(#vitrail-sheet-sheen)" />
      <rect
        x="1"
        y="1"
        width="76"
        height="56"
        fill="none"
        stroke={active ? 'rgba(255,248,220,0.62)' : 'rgba(255,255,255,0.18)'}
        strokeWidth="1"
      />
    </svg>
  );
};

/**
 * Wright-inspired vertical stained-glass border strip for the VITRAIL nameplate course.
 */
export const NameplateMarginInlaySvg: React.FC<{ side: 'left' | 'right' }> = ({ side }) => (
  <svg
    className="margin-inlay-svg"
    viewBox="0 0 22 46"
    preserveAspectRatio="none"
    aria-hidden="true"
  >
    <rect x="0" y="0" width="22" height="46" fill="#100e14" />
    {side === 'left' ? (
      <>
        <rect x="1" y="1" width="9" height="13" fill="#2563eb" />
        <rect x="12" y="1" width="9" height="21" fill="#ebe3d2" fillOpacity="0.82" />
        <rect x="1" y="16" width="9" height="13" fill="#f59e0b" />
        <rect x="12" y="24" width="9" height="21" fill="#0d9488" />
        <rect x="1" y="31" width="9" height="14" fill="#be123c" />
      </>
    ) : (
      <>
        <rect x="1" y="1" width="9" height="21" fill="#ebe3d2" fillOpacity="0.82" />
        <rect x="12" y="1" width="9" height="13" fill="#d97706" />
        <rect x="1" y="24" width="9" height="21" fill="#6d28d9" />
        <rect x="12" y="16" width="9" height="13" fill="#10b981" />
        <rect x="12" y="31" width="9" height="14" fill="#2563eb" />
      </>
    )}
    <rect x="0" y="0" width="22" height="46" fill="url(#vitrail-sheet-sheen)" />
  </svg>
);

/**
 * Slender Wright-style stained-glass right-margin jewel strip for each Game Mode course.
 */
export const ModeRightJewelSvg: React.FC<{
  mode: 'solo' | 'party' | 'online' | 'sandbox';
  active: boolean;
}> = ({ mode, active }) => {
  const c =
    mode === 'solo'
      ? ['#1d4ed8', '#fef08a', '#0284c7']
      : mode === 'party'
      ? ['#047857', '#fbbf24', '#0d9488']
      : mode === 'online'
      ? ['#0d9488', '#fde047', '#0284c7']
      : ['#6d28d9', '#f59e0b', '#be185d'];

  return (
    <svg className="mode-right-jewel-svg" viewBox="0 0 11 38" aria-hidden="true">
      <rect x="0" y="0" width="11" height="38" fill="#100e14" />
      <g opacity={active ? 1 : 0.62}>
        <rect x="1" y="1" width="9" height="11" fill={c[0]} />
        <rect x="1" y="14" width="9" height="10" fill={c[1]} fillOpacity="0.88" />
        <rect x="1" y="26" width="9" height="11" fill={c[2]} />
      </g>
      <rect x="0" y="0" width="11" height="38" fill="url(#vitrail-sheet-sheen)" />
    </svg>
  );
};

/**
 * Full-bleed inlaid stained-glass mosaic medallion on the left end of each Game Mode course.
 */
export const ModeMosaicMedallionSvg: React.FC<{
  mode: 'solo' | 'party' | 'online' | 'sandbox';
  active: boolean;
}> = ({ mode, active }) => {
  if (mode === 'solo') {
    return (
      <svg className="mode-medallion-svg" viewBox="0 0 46 38" aria-hidden="true">
        <rect x="0" y="0" width="46" height="38" fill="#100e14" />
        <g opacity={active ? 1 : 0.65}>
          <polygon points="1,1 16,1 10,19 1,19" fill="#1e3a8a" />
          <polygon points="30,1 45,1 45,19 36,19" fill="#0284c7" />
          <polygon points="1,20.5 10,20.5 16,37 1,37" fill="#0369a1" />
          <polygon points="36,20.5 45,20.5 45,37 30,37" fill="#1d4ed8" />
          {/* Central 3-course stacked crystal tower inlay */}
          <rect x="14" y="25" width="18" height="6" fill="#60a5fa" />
          <rect x="16" y="16.5" width="14" height="6" fill="#bae6fd" />
          <polygon points="23,5.5 29.5,14 16.5,14" fill="#fef08a" />
        </g>
        <rect x="0" y="0" width="46" height="38" fill="url(#vitrail-sheet-sheen)" />
      </svg>
    );
  }
  if (mode === 'party') {
    return (
      <svg className="mode-medallion-svg" viewBox="0 0 46 38" aria-hidden="true">
        <rect x="0" y="0" width="46" height="38" fill="#100e14" />
        <g opacity={active ? 1 : 0.65}>
          <polygon points="1,1 22,1 12,18 1,18" fill="#065f46" />
          <polygon points="24,1 45,1 45,18 34,18" fill="#b45309" />
          <polygon points="1,20 12,20 22,37 1,37" fill="#0f766e" />
          <polygon points="34,20 45,20 45,37 24,37" fill="#047857" />
          {/* Interlocking twin player lozenges */}
          <polygon
            points="18.5,8 26.5,19 18.5,30 10.5,19"
            fill="#34d399"
            stroke="#100e14"
            strokeWidth="1.5"
          />
          <polygon
            points="27.5,8 35.5,19 27.5,30 19.5,19"
            fill="#fbbf24"
            fillOpacity="0.88"
            stroke="#100e14"
            strokeWidth="1.5"
          />
        </g>
        <rect x="0" y="0" width="46" height="38" fill="url(#vitrail-sheet-sheen)" />
      </svg>
    );
  }
  if (mode === 'online') {
    return (
      <svg className="mode-medallion-svg" viewBox="0 0 46 38" aria-hidden="true">
        <rect x="0" y="0" width="46" height="38" fill="#100e14" />
        <g opacity={active ? 1 : 0.65}>
          <polygon points="1,1 22,1 14,18 1,18" fill="#0f766e" />
          <polygon points="24,1 45,1 45,18 32,18" fill="#0369a1" />
          <polygon points="1,20 14,20 22,37 1,37" fill="#115e59" />
          <polygon points="32,20 45,20 45,37 24,37" fill="#1d4ed8" />
          {/* Linked globe meridian + twin distant prism nodes */}
          <circle
            cx="23"
            cy="19"
            r="10.5"
            fill="#0d9488"
            fillOpacity="0.55"
            stroke="#100e14"
            strokeWidth="1.5"
          />
          <path
            d="M 13 19 Q 23 11 33 19 Q 23 27 13 19 Z"
            fill="#67e8f9"
            fillOpacity="0.85"
            stroke="#100e14"
            strokeWidth="1.2"
          />
          <polygon points="14,15 18,19 14,23 10,19" fill="#fde047" stroke="#100e14" strokeWidth="1.2" />
          <polygon points="32,15 36,19 32,23 28,19" fill="#38bdf8" stroke="#100e14" strokeWidth="1.2" />
        </g>
        <rect x="0" y="0" width="46" height="38" fill="url(#vitrail-sheet-sheen)" />
      </svg>
    );
  }
  return (
    <svg className="mode-medallion-svg" viewBox="0 0 46 38" aria-hidden="true">
      <rect x="0" y="0" width="46" height="38" fill="#100e14" />
      <g opacity={active ? 1 : 0.65}>
        {/* Voronoi stained-glass tessellation */}
        <polygon points="1,1 21,1 16,15 1,13" fill="#581c87" />
        <polygon points="23,1 45,1 45,16 18,15" fill="#9333ea" />
        <polygon points="1,15 16,17 13,37 1,37" fill="#be185d" />
        <polygon points="18,17 33,22 29,37 15,37" fill="#f59e0b" />
        <polygon points="35,18 45,18 45,37 31,37" fill="#6d28d9" />
        <polygon
          points="23,11 29,17 23,23 17,17"
          fill="#fef3c7"
          stroke="#100e14"
          strokeWidth="1.5"
        />
      </g>
      <rect x="0" y="0" width="46" height="38" fill="url(#vitrail-sheet-sheen)" />
    </svg>
  );
};

/**
 * Glass-in-Glass Sill Quarry Inlay Background (Outer colored flash-glass border strips
 * soldered around a clear/frosted central etched crystal field).
 */
export const SillGlassInGlassBgSvg: React.FC<{
  variant: 'chime' | 'muted' | 'rebuild' | 'guide';
}> = ({ variant }) => {
  const c =
    variant === 'chime'
      ? { a: '#0d9488', b: '#0284c7', core: 'rgba(26, 56, 62, 0.78)' }
      : variant === 'rebuild'
      ? { a: '#d97706', b: '#2563eb', core: 'rgba(54, 44, 32, 0.78)' }
      : variant === 'guide'
      ? { a: '#6d28d9', b: '#d97706', core: 'rgba(42, 36, 48, 0.78)' }
      : { a: '#52525b', b: '#3f3f46', core: 'rgba(32, 30, 34, 0.80)' };

  return (
    <svg
      className="full-pane-mosaic-svg"
      viewBox="0 0 78 52"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {/* 4 beveled outer flash-glass border strips ("glass in glass") */}
      <polygon points="0,0 78,0 71,6 7,6" fill={c.a} fillOpacity="0.82" />
      <polygon points="78,0 78,52 71,46 71,6" fill={c.b} fillOpacity="0.78" />
      <polygon points="78,52 0,52 7,46 71,46" fill={c.a} fillOpacity="0.72" />
      <polygon points="0,52 0,0 7,6 7,46" fill={c.b} fillOpacity="0.82" />
      {/* Inner etched crystal field */}
      <rect
        x="7"
        y="6"
        width="64"
        height="40"
        fill={c.core}
        stroke="#100e14"
        strokeWidth="1.8"
      />
      {/* Corner lead miters */}
      <line x1="0" y1="0" x2="7" y2="6" stroke="#100e14" strokeWidth="1.6" />
      <line x1="78" y1="0" x2="71" y2="6" stroke="#100e14" strokeWidth="1.6" />
      <line x1="78" y1="52" x2="71" y2="46" stroke="#100e14" strokeWidth="1.6" />
      <line x1="0" y1="52" x2="7" y2="46" stroke="#100e14" strokeWidth="1.6" />
      <rect x="0" y="0" width="78" height="52" fill="url(#vitrail-seed-pattern)" />
      <rect x="0" y="0" width="78" height="52" fill="url(#vitrail-sheet-sheen)" />
    </svg>
  );
};

/**
 * Custom Etched Glyphs for Sill Quarries (Acoustic Crystal, Stacked Courses, Rose Guide, Voronoi Smash).
 */
export const AcousticCrystalSvg: React.FC<{ muted: boolean }> = ({ muted }) => (
  <svg className="sill-etch-svg" viewBox="0 0 22 22" aria-hidden="true">
    <polygon
      points="11,2.5 15,11 11,19.5 7,11"
      fill="currentColor"
      fillOpacity={muted ? 0.15 : 0.42}
      stroke="currentColor"
      strokeWidth="1.2"
    />
    {!muted ? (
      <>
        <path
          d="M4.8,7 Q2.4,11 4.8,15"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.15"
        />
        <path
          d="M17.2,7 Q19.6,11 17.2,15"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.15"
        />
        <circle cx="11" cy="11" r="1.4" fill="#fef08a" />
      </>
    ) : (
      <line x1="4" y1="18" x2="18" y2="4" stroke="#fca5a5" strokeWidth="1.6" />
    )}
  </svg>
);

export const RebuildTowerSvg: React.FC = () => (
  <svg className="sill-etch-svg" viewBox="0 0 22 22" aria-hidden="true">
    <rect
      x="3.5"
      y="14.8"
      width="15"
      height="3.4"
      fill="currentColor"
      fillOpacity="0.32"
      stroke="currentColor"
      strokeWidth="1.1"
    />
    <rect
      x="4.8"
      y="9.8"
      width="3.6"
      height="3.6"
      fill="#60a5fa"
      stroke="currentColor"
      strokeWidth="1"
    />
    <rect
      x="9.2"
      y="9.8"
      width="3.6"
      height="3.6"
      fill="#fbbf24"
      stroke="currentColor"
      strokeWidth="1"
    />
    <rect
      x="13.6"
      y="9.8"
      width="3.6"
      height="3.6"
      fill="#34d399"
      stroke="currentColor"
      strokeWidth="1"
    />
    <rect
      x="4.5"
      y="4.8"
      width="13"
      height="3.4"
      fill="currentColor"
      fillOpacity="0.52"
      stroke="currentColor"
      strokeWidth="1.1"
    />
  </svg>
);

export const GuideQuatrefoilSvg: React.FC = () => (
  <svg className="sill-etch-svg" viewBox="0 0 22 22" aria-hidden="true">
    <polygon
      points="11,2.2 19.8,11 11,19.8 2.2,11"
      fill="currentColor"
      fillOpacity="0.18"
      stroke="currentColor"
      strokeWidth="1.15"
    />
    <circle cx="11" cy="11" r="4.6" fill="none" stroke="currentColor" strokeWidth="1.05" />
    <line x1="11" y1="4.2" x2="11" y2="17.8" stroke="currentColor" strokeWidth="0.85" />
    <line x1="4.2" y1="11" x2="17.8" y2="11" stroke="currentColor" strokeWidth="0.85" />
    <circle cx="11" cy="11" r="1.5" fill="#fef08a" />
  </svg>
);

export const SmashVoronoiSvg: React.FC = () => (
  <svg className="sill-etch-svg" viewBox="0 0 22 22" aria-hidden="true">
    <rect
      x="2.5"
      y="2.5"
      width="17"
      height="17"
      fill="#ef4444"
      fillOpacity="0.45"
      stroke="currentColor"
      strokeWidth="1.15"
    />
    <line x1="10.5" y1="11" x2="2.5" y2="5" stroke="currentColor" strokeWidth="1.1" />
    <line x1="10.5" y1="11" x2="14.5" y2="2.5" stroke="currentColor" strokeWidth="1.1" />
    <line x1="10.5" y1="11" x2="19.5" y2="10.5" stroke="currentColor" strokeWidth="1.1" />
    <line x1="10.5" y1="11" x2="17.5" y2="19.5" stroke="currentColor" strokeWidth="1.1" />
    <line x1="10.5" y1="11" x2="6.5" y2="19.5" stroke="currentColor" strokeWidth="1.1" />
    <polygon points="10.5,8.2 13.3,11 10.5,13.8 7.7,11" fill="#fef08a" />
  </svg>
);

/**
 * Architectural Stepped Stained-Glass Apron / Pendant Finial beneath the Left Lancet Window sill.
 */
export const LancetApronFinialSvg: React.FC = () => (
  <svg className="lancet-apron-svg" viewBox="0 0 244 24" aria-hidden="true">
    <polygon
      points="0,0 244,0 218,12 152,12 122,23 92,12 26,12"
      fill="#100e14"
      stroke="rgba(232,216,186,0.34)"
      strokeWidth="0.9"
    />
    <polygon points="4,1.5 88,1.5 90,10.5 27,10.5" fill="#1e3a8a" fillOpacity="0.88" />
    <polygon points="90,1.5 154,1.5 151,10.5 93,10.5" fill="#d97706" fillOpacity="0.88" />
    <polygon points="156,1.5 240,1.5 217,10.5 154,10.5" fill="#0f766e" fillOpacity="0.88" />
    <polygon points="96,12 148,12 122,21.2" fill="#f59e0b" />
    <polygon points="114,12 130,12 122,18" fill="#fef3c7" />
    <polygon
      points="0,0 244,0 218,12 152,12 122,23 92,12 26,12"
      fill="url(#vitrail-sheet-sheen)"
    />
  </svg>
);

/**
 * Wheel-Cut Acid-Etched Optical Crystal Plumb Compass Rose (Right side instrument).
 */
export const EtchedCrystalCompassSvg: React.FC = () => (
  <svg className="etched-compass-svg" viewBox="0 0 64 64" aria-hidden="true">
    {/* Outer diamond-etched crystal hairline */}
    <circle
      cx="32"
      cy="32"
      r="29"
      fill="rgba(255,252,245,0.035)"
      stroke="rgba(255,250,240,0.34)"
      strokeWidth="0.85"
    />
    <circle
      cx="32"
      cy="32"
      r="24.5"
      fill="none"
      stroke="rgba(255,250,240,0.18)"
      strokeWidth="0.65"
      strokeDasharray="1.5 2.5"
    />
    {/* 4 Inlaid Stained-Glass Cardinal Jewels set into the crystal */}
    <polygon points="32,2.5 35,8.5 29,8.5" fill="#3b82f6" stroke="#100e14" strokeWidth="0.8" />
    <polygon points="61.5,32 55.5,35 55.5,29" fill="#f59e0b" stroke="#100e14" strokeWidth="0.8" />
    <polygon points="32,61.5 35,55.5 29,55.5" fill="#10b981" stroke="#100e14" strokeWidth="0.8" />
    <polygon points="2.5,32 8.5,35 8.5,29" fill="#a855f7" stroke="#100e14" strokeWidth="0.8" />
    {/* Wheel-etched silica crosshairs & equilibrium lozenge */}
    <line
      x1="10"
      y1="32"
      x2="54"
      y2="32"
      stroke="rgba(255,250,240,0.22)"
      strokeWidth="0.75"
    />
    <line
      x1="32"
      y1="10"
      x2="32"
      y2="54"
      stroke="rgba(255,250,240,0.22)"
      strokeWidth="0.75"
    />
    <polygon
      points="32,19 45,32 32,45 19,32"
      fill="none"
      stroke="rgba(255,250,240,0.32)"
      strokeWidth="0.8"
    />
  </svg>
);

/**
 * Inlaid Multi-Shard Stained-Glass End-Jewel for the Bottom Cut-Glass Prism Rule.
 */
export const PrismEndInlaySvg: React.FC<{
  tone: 'amber' | 'emerald' | 'cobalt' | 'amethyst' | 'ruby';
}> = ({ tone }) => {
  const c =
    tone === 'emerald'
      ? ['#065f46', '#10b981', '#6ee7b7']
      : tone === 'cobalt'
      ? ['#1e3a8a', '#3b82f6', '#93c5fd']
      : tone === 'amethyst'
      ? ['#4c1d95', '#9333ea', '#e9d5ff']
      : tone === 'ruby'
      ? ['#7f1d1d', '#ef4444', '#fecaca']
      : ['#78350f', '#f59e0b', '#fef08a'];

  return (
    <svg
      className="prism-end-inlay-svg"
      viewBox="0 0 18 46"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <rect x="0" y="0" width="18" height="46" fill="#100e14" />
      <polygon points="1,1 17,1 9,15 1,11" fill={c[0]} />
      <polygon points="17,2 17,32 9,23 9,15" fill={c[1]} />
      <polygon points="1,13 9,16 9,30 1,34" fill={c[1]} fillOpacity="0.8" />
      <polygon points="1,36 9,31 17,34 17,45 1,45" fill={c[0]} />
      <polygon
        points="9,16 14,23 9,30 4,23"
        fill={c[2]}
        stroke="#100e14"
        strokeWidth="1.2"
      />
      <rect x="0" y="0" width="18" height="46" fill="url(#vitrail-sheet-sheen)" />
    </svg>
  );
};

export const EtchedCloseSvg: React.FC = () => (
  <svg className="sill-etch-svg" viewBox="0 0 20 20" aria-hidden="true">
    <polygon
      points="10,2 18,10 10,18 2,10"
      fill="none"
      stroke="currentColor"
      strokeOpacity="0.38"
      strokeWidth="0.9"
    />
    <line x1="6.6" y1="6.6" x2="13.4" y2="13.4" stroke="currentColor" strokeWidth="1.35" />
    <line x1="13.4" y1="6.6" x2="6.6" y2="13.4" stroke="currentColor" strokeWidth="1.35" />
  </svg>
);

export const PlacementGlyphSvg: React.FC<{
  kind: 'ccw' | 'cw' | 'turn90' | 'align' | 'place';
}> = ({ kind }) => {
  if (kind === 'ccw') {
    return (
      <svg className="place-glyph-svg" viewBox="0 0 18 18" aria-hidden="true">
        <path
          d="M13.5,12.5 A5.5,5.5 0 1,0 5.2,6.8"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.3"
        />
        <polygon points="3.5,8.2 5.1,4.2 8.2,7.2" fill="currentColor" />
      </svg>
    );
  }
  if (kind === 'cw') {
    return (
      <svg className="place-glyph-svg" viewBox="0 0 18 18" aria-hidden="true">
        <path
          d="M4.5,12.5 A5.5,5.5 0 1,1 12.8,6.8"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.3"
        />
        <polygon points="14.5,8.2 12.9,4.2 9.8,7.2" fill="currentColor" />
      </svg>
    );
  }
  if (kind === 'turn90') {
    return (
      <svg className="place-glyph-svg" viewBox="0 0 18 18" aria-hidden="true">
        <rect
          x="2.5"
          y="7"
          width="13"
          height="4"
          fill="currentColor"
          fillOpacity="0.28"
          stroke="currentColor"
          strokeWidth="1.05"
        />
        <rect
          x="7"
          y="2.5"
          width="4"
          height="13"
          fill="currentColor"
          fillOpacity="0.72"
          stroke="currentColor"
          strokeWidth="1.05"
        />
      </svg>
    );
  }
  if (kind === 'align') {
    return (
      <svg className="place-glyph-svg" viewBox="0 0 18 18" aria-hidden="true">
        <polygon points="9,2 16,9 9,16 2,9" fill="none" stroke="currentColor" strokeWidth="1.15" />
        <circle cx="9" cy="9" r="2" fill="currentColor" />
      </svg>
    );
  }
  return (
    <svg className="place-glyph-svg" viewBox="0 0 18 18" aria-hidden="true">
      <rect
        x="3"
        y="3.2"
        width="12"
        height="4.2"
        fill="currentColor"
        fillOpacity="0.82"
        stroke="currentColor"
        strokeWidth="1.05"
      />
      <line x1="9" y1="8.4" x2="9" y2="12.6" stroke="currentColor" strokeWidth="1.25" />
      <polygon points="9,14.8 6.4,11.6 11.6,11.6" fill="currentColor" />
    </svg>
  );
};

export const RoseHallmarkSvg: React.FC = () => (
  <svg className="inlay-svg hallmark-svg" viewBox="0 0 36 36" aria-hidden="true">
    <polygon
      points="18,1.5 34.5,18 18,34.5 1.5,18"
      fill="#100e14"
      stroke="rgba(235,220,192,0.55)"
      strokeWidth="1"
    />
    <polygon points="18,3.2 25.5,10.8 18,16.6 10.5,10.8" fill="#e12d39" />
    <polygon points="32.8,18 25.2,25.5 19.4,18 25.2,10.5" fill="#f59e0b" />
    <polygon points="18,32.8 10.5,25.2 18,19.4 25.5,25.2" fill="#2563eb" />
    <polygon points="3.2,18 10.8,10.5 16.6,18 10.8,25.5" fill="#10b981" />
    <polygon
      points="18,12.5 23.5,18 18,23.5 12.5,18"
      fill="#faf6ee"
      stroke="#100e14"
      strokeWidth="1.4"
    />
  </svg>
);

export const CodexStepMedallionSvg: React.FC<{ step: 1 | 2 | 3 }> = ({ step }) => {
  if (step === 1) {
    return (
      <svg className="codex-medallion-svg" viewBox="0 0 52 52" aria-hidden="true">
        <polygon
          points="26,2 50,26 26,50 2,26"
          fill="#172554"
          stroke="rgba(235,220,192,0.48)"
          strokeWidth="1.2"
        />
        <rect x="14" y="21.5" width="24" height="9" fill="#3b82f6" stroke="#eff6ff" strokeWidth="1.1" />
        <path d="M10,16 Q26,8 42,16" fill="none" stroke="#93c5fd" strokeWidth="1.15" />
        <path d="M14,36 Q26,42 38,36" fill="none" stroke="#93c5fd" strokeWidth="1.15" />
        <circle cx="26" cy="26" r="2" fill="#fde68a" />
      </svg>
    );
  }
  if (step === 2) {
    return (
      <svg className="codex-medallion-svg" viewBox="0 0 52 52" aria-hidden="true">
        <polygon
          points="26,2 50,26 26,50 2,26"
          fill="#451a03"
          stroke="rgba(235,220,192,0.48)"
          strokeWidth="1.2"
        />
        <rect x="9" y="21.5" width="21" height="9" fill="#f59e0b" stroke="#fef3c7" strokeWidth="1.1" />
        <line
          x1="30"
          y1="26"
          x2="44"
          y2="26"
          stroke="#fde047"
          strokeWidth="1.6"
          strokeDasharray="2.5 2"
        />
        <polygon points="45,26 39.5,22.5 39.5,29.5" fill="#fef08a" />
      </svg>
    );
  }
  return (
    <svg className="codex-medallion-svg" viewBox="0 0 52 52" aria-hidden="true">
      <polygon
        points="26,2 50,26 26,50 2,26"
        fill="#064e3b"
        stroke="rgba(235,220,192,0.48)"
        strokeWidth="1.2"
      />
      <rect x="13" y="28" width="26" height="7" fill="#0d9488" stroke="#ccfbf1" strokeWidth="1.05" />
      <rect
        x="15"
        y="17"
        width="22"
        height="7"
        transform="rotate(-8 26 20.5)"
        fill="#10b981"
        stroke="#ecfdf5"
        strokeWidth="1.1"
      />
      <polygon points="26,9 29.5,13.5 22.5,13.5" fill="#fde68a" />
    </svg>
  );
};

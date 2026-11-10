import React, { useCallback, useEffect, useRef, useState } from 'react';
import { GameEngine } from './engine/GameEngine';
import {
  INITIAL_LAYERS,
  LIGHTING_PRESETS,
  STAINED_GLASS_PALETTE,
} from './physics/constants';
import { GameMode, TowerTelemetry } from './physics/PhysicsWorld';
import {
  AcousticCrystalSvg,
  ClerestoryPaneMosaicSvg,
  CodexStepMedallionSvg,
  EtchedCloseSvg,
  EtchedCrystalCompassSvg,
  GuideQuatrefoilSvg,
  LancetApronFinialSvg,
  LancetArchApexSvg,
  ModeMosaicMedallionSvg,
  ModeRightJewelSvg,
  NameplateMarginInlaySvg,
  PlacementGlyphSvg,
  PrismEndInlaySvg,
  RebuildTowerSvg,
  RoseHallmarkSvg,
  SillGlassInGlassBgSvg,
  SmashVoronoiSvg,
  VitrailSvgDefs,
} from './ui/VitrailGlassArt';

const PLAYER_COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#10b981'];

const LIGHT_LABELS: Record<string, string> = {
  'lancet-morning': 'Morning',
  'golden-vespers': 'Vespers',
  'high-clerestory': 'Noon',
};

export type QuarryGlassTone =
  | 'opal'
  | 'crystal'
  | 'cobalt'
  | 'emerald'
  | 'amethyst'
  | 'amber'
  | 'ruby'
  | 'teal'
  | 'grisaille-cobalt'
  | 'grisaille-emerald'
  | 'grisaille-amethyst';

type UiSoundTone = 'default' | 'tab' | 'primary' | 'danger' | 'step';

const INITIAL_TELEMETRY: TowerTelemetry = {
  turnPhase: 'SELECT_BLOCK',
  gameMode: 'solo',
  currentPlayer: 0,
  playerCount: 2,
  blocksStacked: 0,
  currentLayers: INITIAL_LAYERS,
  peakLayers: INITIAL_LAYERS,
  liveHeightCm: 27.0,
  peakHeightCm: 27.0,
  leanAngleDeg: 0.0,
  wobbleSpeedDegPerSec: 0.0,
  centerOfMassXZ: [0, 0],
  shatteredCount: 0,
  crazedCount: 0,
  intactCount: 54,
  activeShardCount: 0,
  grabResistance01: 0,
  hoveredBlockId: null,
  activeBlockId: null,
  hoveredSlotIndex: null,
  carriedYawDeg: 0,
  lockedMinRow: INITIAL_LAYERS,
  settleProgress01: 0,
  collapseReason: null,
};

interface QuarryButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: QuarryGlassTone;
  active?: boolean;
  soundTone?: UiSoundTone;
  inlayArt?: React.ReactNode;
  onAudioDown?: (tone: UiSoundTone) => void;
  onAudioUp?: (tone: UiSoundTone) => void;
}

/**
 * Interactive Stained-Glass / Beveled-Crystal Pane:
 * Seated inside a stationary lead came or crystal bezel; physically recesses inward
 * on pointerdown and rebounds via an underdamped spring + refractive caustic sweep.
 */
const QuarryButton: React.FC<QuarryButtonProps> = ({
  tone = 'crystal',
  active = false,
  soundTone = 'default',
  inlayArt,
  onAudioDown,
  onAudioUp,
  className = '',
  children,
  onPointerDown,
  onPointerUp,
  onPointerLeave,
  onPointerCancel,
  onPointerMove,
  onClick,
  ...rest
}) => {
  const [pressing, setPressing] = useState(false);
  const [rebounding, setRebounding] = useState(false);
  const reboundTimerRef = useRef<number | null>(null);
  const btnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    return () => {
      if (reboundTimerRef.current !== null) {
        window.clearTimeout(reboundTimerRef.current);
      }
    };
  }, []);

  const triggerSpringRebound = useCallback(() => {
    if (reboundTimerRef.current !== null) {
      window.clearTimeout(reboundTimerRef.current);
    }
    setRebounding(false);
    requestAnimationFrame(() => {
      setRebounding(true);
      reboundTimerRef.current = window.setTimeout(() => {
        setRebounding(false);
        reboundTimerRef.current = null;
      }, 300);
    });
  }, []);

  const handlePtrMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const el = btnRef.current;
    if (el) {
      const r = el.getBoundingClientRect();
      const gx = ((e.clientX - r.left) / Math.max(1, r.width)) * 100;
      const gy = ((e.clientY - r.top) / Math.max(1, r.height)) * 100;
      el.style.setProperty('--qx', `${gx.toFixed(1)}%`);
      el.style.setProperty('--qy', `${gy.toFixed(1)}%`);
    }
    onPointerMove?.(e);
  };

  const handlePtrDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (e.button === 0 && !rest.disabled) {
      setPressing(true);
      setRebounding(false);
      onAudioDown?.(soundTone);
    }
    onPointerDown?.(e);
  };

  const handlePtrUp = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (pressing) {
      setPressing(false);
      triggerSpringRebound();
      onAudioUp?.(soundTone);
    }
    onPointerUp?.(e);
  };

  const handlePtrLeave = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (pressing) {
      setPressing(false);
    }
    onPointerLeave?.(e);
  };

  const handlePtrCancel = (e: React.PointerEvent<HTMLButtonElement>) => {
    setPressing(false);
    onPointerCancel?.(e);
  };

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (e.detail === 0 && !rest.disabled) {
      onAudioDown?.(soundTone);
      triggerSpringRebound();
      onAudioUp?.(soundTone);
    }
    onClick?.(e);
  };

  return (
    <button
      ref={btnRef}
      type="button"
      data-pressing={pressing ? 'true' : undefined}
      data-rebounding={rebounding ? 'true' : undefined}
      className={`quarry-btn quarry-tone-${tone} ${active ? 'is-lit' : ''} ${className}`}
      onPointerMove={handlePtrMove}
      onPointerDown={handlePtrDown}
      onPointerUp={handlePtrUp}
      onPointerLeave={handlePtrLeave}
      onPointerCancel={handlePtrCancel}
      onClick={handleClick}
      {...rest}
    >
      <span className="quarry-glass-pane">
        {inlayArt}
        <span className="quarry-caustic-glint" aria-hidden="true" />
        <span className="quarry-etch-content">{children}</span>
      </span>
    </button>
  );
};

export const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<GameEngine | null>(null);

  const [telemetry, setTelemetry] = useState<TowerTelemetry>(INITIAL_TELEMETRY);
  const [muted, setMuted] = useState<boolean>(false);
  const [lightPreset, setLightPreset] = useState<string>(LIGHTING_PRESETS[0].id);
  const [showHelp, setShowHelp] = useState<boolean>(false);
  const [gpuError, setGpuError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    const engine = new GameEngine(canvas, (nextTelemetry) => {
      if (!cancelled) {
        setTelemetry(nextTelemetry);
      }
    });
    engineRef.current = engine;

    engine.init().catch((err: unknown) => {
      if (!cancelled) {
        setGpuError(err instanceof Error ? err.message : String(err));
      }
    });

    return () => {
      cancelled = true;
      engine.dispose();
      if (engineRef.current === engine) {
        engineRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    engineRef.current?.setModalOpen(showHelp);
    if (!showHelp) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        engineRef.current?.audio.playUiKeyUp('default');
        setShowHelp(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [showHelp]);

  const playUiDown = useCallback((tone: UiSoundTone) => {
    engineRef.current?.audio.playUiKeyDown(tone);
  }, []);

  const playUiUp = useCallback((tone: UiSoundTone) => {
    engineRef.current?.audio.playUiKeyUp(tone);
  }, []);

  const handleModeChange = (mode: GameMode) => {
    engineRef.current?.resetGame(mode, telemetry.playerCount);
  };

  const handleSetPlayerCount = (count: number) => {
    if (count !== telemetry.playerCount) {
      engineRef.current?.resetGame('party', count);
    }
  };

  const handleToggleMute = () => {
    const eng = engineRef.current;
    if (!eng) return;
    const next = !muted;
    eng.audio.setMuted(next);
    setMuted(next);
    if (!next) {
      eng.audio.unlock();
      eng.audio.playUiKeyUp('primary');
    }
  };

  const handleSelectLighting = (id: string) => {
    setLightPreset(id);
    engineRef.current?.setLightingPreset(id);
  };

  const handleRebuild = () => {
    engineRef.current?.resetGame(telemetry.gameMode, telemetry.playerCount);
  };

  // Plumb compass jewel coordinates
  const comXClamp = Math.max(-15, Math.min(15, telemetry.centerOfMassXZ[0] * 7.2));
  const comZClamp = Math.max(-15, Math.min(15, telemetry.centerOfMassXZ[1] * 7.2));
  const leanSeverity =
    telemetry.turnPhase === 'COLLAPSED' ||
    telemetry.leanAngleDeg > 3.2 ||
    telemetry.wobbleSpeedDegPerSec > 12
      ? 'danger'
      : telemetry.leanAngleDeg > 1.3 || telemetry.wobbleSpeedDegPerSec > 4
      ? 'warn'
      : 'calm';

  const plumbColor =
    leanSeverity === 'danger'
      ? '#f87171'
      : leanSeverity === 'warn'
      ? '#fbbf24'
      : '#34d399';

  // Hovered block info
  const hoveredBlock =
    telemetry.hoveredBlockId !== null && engineRef.current?.physics
      ? engineRef.current.physics.blocks[telemetry.hoveredBlockId]
      : null;
  const hoveredSwatch = hoveredBlock ? STAINED_GLASS_PALETTE[hoveredBlock.swatchId] : null;

  const getHoveredStatusText = (): string => {
    if (!hoveredBlock) return '';
    if (telemetry.gameMode === 'sandbox') {
      return hoveredBlock.damage > 0.05 ? 'Crazed · Free' : 'Unlocked';
    }
    if (hoveredBlock.isLegalTarget) {
      return hoveredBlock.damage > 0.05 ? 'Crazed · Legal' : 'Legal to Pull';
    }
    const rowNum = hoveredBlock.assignedLayer + 1;
    if (
      telemetry.lockedMinRow !== null &&
      telemetry.lockedMinRow < telemetry.currentLayers &&
      rowNum === telemetry.lockedMinRow
    ) {
      return 'Support Row Locked';
    }
    return 'Top Row Locked';
  };

  const lockedRowsLabel =
    telemetry.lockedMinRow === null
      ? 'All rows unlocked'
      : telemetry.lockedMinRow < telemetry.currentLayers
      ? `Rows ${telemetry.lockedMinRow}–${telemetry.currentLayers} locked`
      : `Row ${telemetry.lockedMinRow} locked`;

  const resistanceLabel =
    telemetry.grabResistance01 < 0.34
      ? 'Loose · Free Glide'
      : telemetry.grabResistance01 < 0.7
      ? 'Snug · Bearing Load'
      : 'Pinned · Column Load';

  const resistanceColor =
    telemetry.grabResistance01 < 0.34
      ? '#7dd3fc'
      : telemetry.grabResistance01 < 0.7
      ? '#fcd34d'
      : '#fca5a5';

  const partyTurnLabel = (() => {
    if (telemetry.turnPhase !== 'COLLAPSED') {
      return `Player ${telemetry.currentPlayer + 1}'s Turn`;
    }
    if (telemetry.blocksStacked === 0) {
      return `Player ${telemetry.currentPlayer + 1} Toppled`;
    }
    const winnerNum =
      ((telemetry.currentPlayer + telemetry.playerCount - 1) % telemetry.playerCount) + 1;
    return `P${winnerNum} Wins · P${telemetry.currentPlayer + 1} Toppled`;
  })();

  const partyAccentIndex =
    telemetry.turnPhase === 'COLLAPSED' && telemetry.blocksStacked > 0
      ? (telemetry.currentPlayer + telemetry.playerCount - 1) % telemetry.playerCount
      : telemetry.currentPlayer;

  const activeTensionSegments = Math.max(1, Math.round(telemetry.grabResistance01 * 12));
  const activeSettleSegments = Math.round(telemetry.settleProgress01 * 10);

  const isBench =
    typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('bench');

  return (
    <div className="app-shell" data-lighting={lightPreset}>
      <VitrailSvgDefs />
      <canvas ref={canvasRef} className="webgpu-canvas" />

      {gpuError && (
        <div className="codex-backdrop">
          <div className="codex-window">
            <div className="codex-pane-inner">
              <h2 className="etched-heading">WebGPU Required</h2>
              <p className="etched-prose">
                {gpuError} Open Vitrail in a WebGPU-enabled browser (Chrome 113+, Edge 113+, or
                Safari 18+).
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="hud-layer" style={{ display: isBench ? 'none' : undefined }}>
        {/* Subtle oblique clerestory sunbeam cast from the left stained-glass lancet into the room */}
        <div className="lancet-sunbeam-veil" aria-hidden="true" />

        <div className="hud-upper-space">
          {/* ================================================================
           * LEFT JAMB: THE ARCHITECTURAL STAINED-GLASS LANCET WINDOW
           * One unified, pointed-arch leaded stained-glass window composition
           * ================================================================ */}
          <nav
            className="vitrail-lancet-window"
            aria-label="Vitrail Stained-Glass Lancet Window"
          >
            {/* 1. Pointed Lancet Rose & Sky Apex */}
            <div className="lancet-arch-crown">
              <LancetArchApexSvg presetId={lightPreset} />
            </div>

            {/* 2. Leaded Window Body */}
            <div className="lancet-sash-body">
              {/* Clerestory Sky Panes: Click sky glass directly to shift studio sunlight */}
              <div
                className="lancet-course sky-course"
                role="radiogroup"
                aria-label="Cathedral Window Sunlight"
              >
                {LIGHTING_PRESETS.map((p) => {
                  const isSelected = lightPreset === p.id;
                  return (
                    <QuarryButton
                      key={p.id}
                      role="radio"
                      aria-checked={isSelected}
                      tone="crystal"
                      active={isSelected}
                      soundTone="tab"
                      onAudioDown={playUiDown}
                      onAudioUp={playUiUp}
                      inlayArt={<ClerestoryPaneMosaicSvg presetId={p.id} active={isSelected} />}
                      className="sky-lancet-pane"
                      onClick={() => handleSelectLighting(p.id)}
                      title={`Window Light: ${p.name}`}
                    >
                      <span className="sky-pane-label">{LIGHT_LABELS[p.id] ?? p.name}</span>
                    </QuarryButton>
                  );
                })}
              </div>

              {/* Frosted Milk-Opal Cartouche + Wright Margin-Light Inlays */}
              <div className="lancet-course cartouche-course">
                <div className="margin-inlay-cell" aria-hidden="true">
                  <NameplateMarginInlaySvg side="left" />
                </div>
                <div className="opal-title-plaque">
                  <span className="vitrail-wordmark">VITRAIL</span>
                  <span className="vitrail-spec">7.5 × 2.5 × 1.5 cm · 71.1 g</span>
                </div>
                <div className="margin-inlay-cell" aria-hidden="true">
                  <NameplateMarginInlaySvg side="right" />
                </div>
              </div>

              {/* 3 Horizontal Inlaid Stained-Glass Mode Courses */}
              <div className="lancet-mode-stack" role="tablist" aria-label="Game Mode">
                <QuarryButton
                  role="tab"
                  aria-selected={telemetry.gameMode === 'solo'}
                  tone={telemetry.gameMode === 'solo' ? 'cobalt' : 'grisaille-cobalt'}
                  active={telemetry.gameMode === 'solo'}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={() => handleModeChange('solo')}
                >
                  <ModeMosaicMedallionSvg
                    mode="solo"
                    active={telemetry.gameMode === 'solo'}
                  />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Solo</span>
                    <span className="mode-course-meta">Classic Stack</span>
                  </span>
                  <ModeRightJewelSvg mode="solo" active={telemetry.gameMode === 'solo'} />
                </QuarryButton>

                <QuarryButton
                  role="tab"
                  aria-selected={telemetry.gameMode === 'party'}
                  tone={telemetry.gameMode === 'party' ? 'emerald' : 'grisaille-emerald'}
                  active={telemetry.gameMode === 'party'}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={() => handleModeChange('party')}
                >
                  <ModeMosaicMedallionSvg
                    mode="party"
                    active={telemetry.gameMode === 'party'}
                  />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Pass &amp; Play</span>
                    <span className="mode-course-meta">2–4 Players</span>
                  </span>
                  <ModeRightJewelSvg mode="party" active={telemetry.gameMode === 'party'} />
                </QuarryButton>

                {/* Pass & Play Inlaid Sub-Course */}
                {telemetry.gameMode === 'party' && (
                  <div className="lancet-course party-subcourse">
                    <div className="party-turn-pane">
                      <span
                        className="player-cabochon"
                        style={{
                          backgroundColor: PLAYER_COLORS[partyAccentIndex % PLAYER_COLORS.length],
                          boxShadow: `0 0 8px ${PLAYER_COLORS[partyAccentIndex % PLAYER_COLORS.length]}`,
                        }}
                      />
                      <span className="party-turn-copy">{partyTurnLabel}</span>
                    </div>
                    <div className="party-count-strip" role="group" aria-label="Player Count">
                      {[2, 3, 4].map((cnt) => (
                        <QuarryButton
                          key={cnt}
                          tone={telemetry.playerCount === cnt ? 'amber' : 'crystal'}
                          active={telemetry.playerCount === cnt}
                          soundTone="step"
                          onAudioDown={playUiDown}
                          onAudioUp={playUiUp}
                          className="player-num-pane"
                          onClick={() => handleSetPlayerCount(cnt)}
                          title={`${cnt} Players`}
                        >
                          <span>{cnt}P</span>
                        </QuarryButton>
                      ))}
                    </div>
                  </div>
                )}

                <QuarryButton
                  role="tab"
                  aria-selected={telemetry.gameMode === 'sandbox'}
                  tone={telemetry.gameMode === 'sandbox' ? 'amethyst' : 'grisaille-amethyst'}
                  active={telemetry.gameMode === 'sandbox'}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={() => handleModeChange('sandbox')}
                >
                  <ModeMosaicMedallionSvg
                    mode="sandbox"
                    active={telemetry.gameMode === 'sandbox'}
                  />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Sandbox</span>
                    <span className="mode-course-meta">Free Physics</span>
                  </span>
                  <ModeRightJewelSvg mode="sandbox" active={telemetry.gameMode === 'sandbox'} />
                </QuarryButton>

                {/* Sandbox Fracture Sub-Course */}
                {telemetry.gameMode === 'sandbox' && (
                  <div className="lancet-course sandbox-subcourse">
                    <QuarryButton
                      tone="ruby"
                      active
                      soundTone="danger"
                      onAudioDown={playUiDown}
                      onAudioUp={playUiUp}
                      className="smash-course-btn"
                      onClick={() => engineRef.current?.triggerCollapseDemo()}
                      title="Buckle lower courses to trigger 120Hz Voronoi shatter"
                    >
                      <SmashVoronoiSvg />
                      <span>Buckle &amp; Shatter Tower</span>
                    </QuarryButton>
                  </div>
                )}
              </div>

              {/* Lower Sill: 3 Inlaid Glass-in-Glass Panes */}
              <div className="lancet-course sill-course">
                <QuarryButton
                  tone="crystal"
                  active={!muted}
                  soundTone="default"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  inlayArt={<SillGlassInGlassBgSvg variant={muted ? 'muted' : 'chime'} />}
                  className="sill-pane-btn"
                  onClick={handleToggleMute}
                  title={muted ? 'Unmute Modal Glass Resonance' : 'Mute Modal Glass Resonance'}
                  aria-label={muted ? 'Unmute Modal Glass Audio' : 'Mute Modal Glass Audio'}
                >
                  <AcousticCrystalSvg muted={muted} />
                  <span className="sill-pane-label">{muted ? 'Muted' : 'Chime'}</span>
                </QuarryButton>

                <QuarryButton
                  tone="crystal"
                  soundTone="primary"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  inlayArt={<SillGlassInGlassBgSvg variant="rebuild" />}
                  className="sill-pane-btn"
                  onClick={handleRebuild}
                  title="Rebuild 54-Prism Stained-Glass Tower"
                >
                  <RebuildTowerSvg />
                  <span className="sill-pane-label">Rebuild</span>
                </QuarryButton>

                <QuarryButton
                  tone="crystal"
                  active={showHelp}
                  soundTone="default"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  inlayArt={<SillGlassInGlassBgSvg variant="guide" />}
                  className="sill-pane-btn"
                  onClick={() => setShowHelp(true)}
                  title="Rules, Resonance & Controls"
                  aria-label="How to Play & Controls"
                >
                  <GuideQuatrefoilSvg />
                  <span className="sill-pane-label">Guide</span>
                </QuarryButton>
              </div>
            </div>

            {/* 3. Architectural Stepped Stained-Glass Apron Finial */}
            <LancetApronFinialSvg />
          </nav>

          {/* ================================================================
           * RIGHT SPACE: WHEEL-ETCHED OPTICAL CRYSTAL & INLAID JEWELS
           * Frameless transparent beveled crystal with acid-etched hairlines
           * ================================================================ */}
          <aside
            className={`crystal-optical-plate severity-${leanSeverity}`}
            aria-label="Etched Crystal Tower Instrument"
          >
            {/* Wheel-etched corner registration marks */}
            <span className="crystal-reg reg-tl" aria-hidden="true" />
            <span className="crystal-reg reg-tr" aria-hidden="true" />
            <span className="crystal-reg reg-bl" aria-hidden="true" />
            <span className="crystal-reg reg-br" aria-hidden="true" />

            {/* Etched Plumb Rose */}
            <div className="etched-plumb-section" title="Live Center-of-Mass Plumb">
              <div className="etched-compass-wrap">
                <EtchedCrystalCompassSvg />
                <span
                  className="inlaid-plumb-jewel"
                  style={{
                    left: `calc(50% + ${comXClamp}px)`,
                    top: `calc(50% + ${comZClamp}px)`,
                    backgroundColor: plumbColor,
                    color: plumbColor,
                  }}
                />
              </div>
              <div className="etched-plumb-copy">
                <span className="etched-plumb-title">Plumb</span>
                <span className="etched-plumb-deg" style={{ color: plumbColor }}>
                  {telemetry.leanAngleDeg.toFixed(1)}°
                </span>
                <span className="etched-plumb-status">
                  {telemetry.turnPhase === 'COLLAPSED'
                    ? 'Toppled'
                    : telemetry.wobbleSpeedDegPerSec > 3.0
                    ? `${telemetry.wobbleSpeedDegPerSec.toFixed(0)}°/s sway`
                    : telemetry.leanAngleDeg > 1.8
                    ? 'Leaning'
                    : 'Equilibrium'}
                </span>
              </div>
            </div>

            {/* Acid-Etched Glass Telemetry Rules with Inlaid Glass Cabochons */}
            <div className="etched-metrics-list">
              <div className="etched-metric-row">
                <span className="inlaid-gem gem-cobalt" aria-hidden="true" />
                <div className="etched-metric-main">
                  <span className="etched-m-label">Elevation</span>
                  <span className="etched-m-sub">
                    {telemetry.currentLayers === 0
                      ? 'Toppled'
                      : `${telemetry.currentLayers} ${
                          telemetry.currentLayers === 1 ? 'course' : 'courses'
                        }`}
                  </span>
                </div>
                <span className="etched-m-val">{telemetry.liveHeightCm.toFixed(1)} cm</span>
              </div>

              <div className="etched-metric-row">
                <span className="inlaid-gem gem-emerald" aria-hidden="true" />
                <div className="etched-metric-main">
                  <span className="etched-m-label">Stacked</span>
                  <span className="etched-m-sub">Peak {telemetry.peakHeightCm.toFixed(1)} cm</span>
                </div>
                <span className="etched-m-val">{telemetry.blocksStacked}</span>
              </div>

              <div className="etched-metric-row">
                <span
                  className={`inlaid-gem ${
                    telemetry.shatteredCount > 0
                      ? 'gem-ruby'
                      : telemetry.crazedCount > 0
                      ? 'gem-amber'
                      : 'gem-opal'
                  }`}
                  aria-hidden="true"
                />
                <div className="etched-metric-main">
                  <span className="etched-m-label">
                    {telemetry.shatteredCount > 0 ? 'Fractured' : 'Prisms'}
                  </span>
                  <span className="etched-m-sub">
                    {telemetry.shatteredCount > 0
                      ? `${telemetry.activeShardCount} shards`
                      : telemetry.crazedCount > 0
                      ? `${telemetry.crazedCount} crazed`
                      : 'Intact'}
                  </span>
                </div>
                <span className="etched-m-val">
                  {telemetry.shatteredCount > 0 ? telemetry.shatteredCount : telemetry.intactCount}
                </span>
              </div>
            </div>
          </aside>
        </div>

        {/* ==================================================================
         * BOTTOM HORIZON: 3D BEVEL-MITERED CUT-GLASS PRISM BAR
         * Modeled directly after a beveled soda-lime glass prism from the tower
         * ================================================================== */}
        <div className="hud-bottom-row">
          {telemetry.turnPhase === 'COLLAPSED' ? (
            <div className="collapse-crystal-plaque">
              <div className="monument-spandrel-bar" aria-hidden="true">
                <span className="m-shard s-ruby" />
                <span className="m-shard s-amber" />
                <span className="m-shard s-rose" />
                <span className="m-shard s-cobalt" />
                <span className="m-shard s-amethyst" />
                <span className="m-shard s-amber" />
                <span className="m-shard s-ruby" />
              </div>
              <div className="monument-body">
                <div className="monument-title">
                  {telemetry.shatteredCount > 0 ? 'Tower Shattered' : 'Tower Toppled'}
                </div>
                <p className="monument-prose">
                  {telemetry.shatteredCount > 0
                    ? 'The stained-glass stack gave way across the walnut table.'
                    : 'The stained-glass stack lost equilibrium and tumbled.'}{' '}
                  Reached <strong>{telemetry.peakHeightCm.toFixed(1)} cm</strong> (
                  {telemetry.peakLayers} {telemetry.peakLayers === 1 ? 'course' : 'courses'}) with{' '}
                  <strong>{telemetry.blocksStacked}</strong>{' '}
                  {telemetry.blocksStacked === 1 ? 'prism' : 'prisms'} stacked
                  {telemetry.shatteredCount > 0 ? (
                    <>
                      {' '}
                      and <strong>{telemetry.shatteredCount}</strong> fractured into{' '}
                      <strong>{telemetry.activeShardCount}</strong> Voronoi shards.
                    </>
                  ) : (
                    '.'
                  )}
                </p>
                <QuarryButton
                  tone="amber"
                  active
                  soundTone="primary"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="keystone-cta"
                  onClick={handleRebuild}
                >
                  <RebuildTowerSvg />
                  <span>Rebuild Stained-Glass Tower</span>
                </QuarryButton>
              </div>
            </div>
          ) : telemetry.turnPhase === 'EXTRACTED' ? (
            <div className="cut-glass-prism-bar is-interactive-prism">
              <PrismEndInlaySvg tone="emerald" />
              <div className="prism-bevel-chamber">
                <div className="prism-phase-tag tag-emerald">Stack</div>
                <div className="prism-copy-group">
                  <span className="prism-primary-line">
                    Position across top course &amp; click to release
                  </span>
                  <span className="prism-secondary-line">
                    Wheel or Q / E angles · R turns 90° · 0 aligns · Drag outside to orbit
                  </span>
                </div>

                <div className="prism-inlaid-keys">
                  <QuarryButton
                    tone="crystal"
                    soundTone="step"
                    onAudioDown={playUiDown}
                    onAudioUp={playUiUp}
                    className="prism-key-btn"
                    onClick={() => engineRef.current?.handleRotateCarried(-Math.PI / 24)}
                    title="Angle −7.5° (Q)"
                  >
                    <PlacementGlyphSvg kind="ccw" />
                    <span>Q</span>
                  </QuarryButton>

                  <QuarryButton
                    tone="cobalt"
                    active
                    soundTone="step"
                    onAudioDown={playUiDown}
                    onAudioUp={playUiUp}
                    className="prism-key-btn"
                    onClick={() => engineRef.current?.handleRotateCarried(Math.PI * 0.5)}
                    title="Turn 90° (R)"
                  >
                    <PlacementGlyphSvg kind="turn90" />
                    <span>90° R</span>
                  </QuarryButton>

                  <QuarryButton
                    tone="crystal"
                    soundTone="step"
                    onAudioDown={playUiDown}
                    onAudioUp={playUiUp}
                    className="prism-key-btn"
                    onClick={() => engineRef.current?.handleRotateCarried(Math.PI / 24)}
                    title="Angle +7.5° (E)"
                  >
                    <span>E</span>
                    <PlacementGlyphSvg kind="cw" />
                  </QuarryButton>

                  {telemetry.carriedYawDeg !== 0 && (
                    <QuarryButton
                      tone="teal"
                      active
                      soundTone="step"
                      onAudioDown={playUiDown}
                      onAudioUp={playUiUp}
                      className="prism-key-btn"
                      onClick={() => engineRef.current?.handleResetCarriedRotation()}
                      title="Align square with course (0)"
                    >
                      <PlacementGlyphSvg kind="align" />
                      <span>
                        {telemetry.carriedYawDeg > 0
                          ? `+${telemetry.carriedYawDeg}°`
                          : `${telemetry.carriedYawDeg}°`}
                      </span>
                    </QuarryButton>
                  )}

                  <QuarryButton
                    tone="amber"
                    active
                    soundTone="primary"
                    onAudioDown={playUiDown}
                    onAudioUp={playUiUp}
                    className="prism-release-btn"
                    onClick={() => engineRef.current?.handleReleaseOntoTop()}
                    title="Release prism onto top course (Space / Enter)"
                  >
                    <PlacementGlyphSvg kind="place" />
                    <span>Place Prism</span>
                  </QuarryButton>
                </div>
              </div>
              <PrismEndInlaySvg tone="amber" />
            </div>
          ) : telemetry.turnPhase === 'PULLING' ? (
            <div className="cut-glass-prism-bar">
              <PrismEndInlaySvg tone="cobalt" />
              <div className="prism-bevel-chamber">
                <div className="prism-phase-tag tag-amber">Drawing</div>

                <div className="leaded-tension-strip" aria-label="Pulling Resistance">
                  {Array.from({ length: 12 }).map((_, idx) => {
                    const lit = idx < activeTensionSegments;
                    const segClass = idx < 4 ? 'seg-sky' : idx < 8 ? 'seg-amber' : 'seg-ruby';
                    return (
                      <span
                        key={idx}
                        className={`tension-quarry ${segClass} ${lit ? 'is-lit' : ''}`}
                      />
                    );
                  })}
                </div>

                <div className="prism-copy-group">
                  <span className="prism-primary-line" style={{ color: resistanceColor }}>
                    {resistanceLabel}
                  </span>
                  <span className="prism-secondary-line">
                    {telemetry.gameMode === 'sandbox'
                      ? 'Spring tether active · Release anytime or pull clear to drop onto table'
                      : 'Draw smoothly along channel · Release anytime to choose a looser prism'}
                  </span>
                </div>
              </div>
              <PrismEndInlaySvg tone="ruby" />
            </div>
          ) : telemetry.turnPhase === 'SETTLING' ? (
            <div className="cut-glass-prism-bar">
              <PrismEndInlaySvg tone="emerald" />
              <div className="prism-bevel-chamber">
                <div className="prism-phase-tag tag-emerald">Settling</div>
                <div className="prism-copy-group">
                  <span className="prism-primary-line">Awaiting equilibrium…</span>
                </div>
                <div className="leaded-tension-strip" aria-hidden="true">
                  {Array.from({ length: 10 }).map((_, idx) => (
                    <span
                      key={idx}
                      className={`tension-quarry seg-emerald ${
                        idx < activeSettleSegments ? 'is-lit' : ''
                      }`}
                    />
                  ))}
                </div>
              </div>
              <PrismEndInlaySvg tone="emerald" />
            </div>
          ) : (
            <div className="cut-glass-prism-bar">
              <PrismEndInlaySvg
                tone={telemetry.gameMode === 'sandbox' ? 'amethyst' : 'cobalt'}
              />
              <div className="prism-bevel-chamber">
                <div
                  className={`prism-phase-tag ${
                    telemetry.gameMode === 'sandbox' ? 'tag-amethyst' : 'tag-cobalt'
                  }`}
                >
                  {telemetry.gameMode === 'sandbox' ? 'Sandbox' : 'Extract'}
                </div>

                <div className="prism-copy-group">
                  <span className="prism-primary-line">
                    {telemetry.gameMode === 'sandbox'
                      ? 'Tap any prism to test resonance · Drag freely to pull or shatter'
                      : 'Tap any legal prism to hear looseness · Drag horizontally to draw out'}
                  </span>
                  <span className="prism-secondary-line">
                    {hoveredBlock && hoveredSwatch ? (
                      <span className="inlaid-swatch-readout">
                        <span
                          className="inlaid-prism-chip"
                          style={{ backgroundColor: hoveredSwatch.uiHex }}
                        />
                        <span>
                          {hoveredSwatch.name} · Course {hoveredBlock.assignedLayer + 1} ·{' '}
                          <strong
                            className={
                              hoveredBlock.isLegalTarget || telemetry.gameMode === 'sandbox'
                                ? 'status-legal'
                                : 'status-locked'
                            }
                          >
                            {getHoveredStatusText()}
                          </strong>
                        </span>
                      </span>
                    ) : (
                      `Drag background (or Right-Click) to orbit 360° · Scroll to zoom · ${lockedRowsLabel}`
                    )}
                  </span>
                </div>
              </div>
              <PrismEndInlaySvg
                tone={telemetry.gameMode === 'sandbox' ? 'amethyst' : 'amber'}
              />
            </div>
          )}
        </div>
      </div>

      {/* ====================================================================
       * VITRAIL GUIDE MODAL (Inlaid Glass Medallions + Etched Crystal Sheet)
       * ==================================================================== */}
      {showHelp && (
        <div className="codex-backdrop" onClick={() => setShowHelp(false)}>
          <div
            className="codex-window"
            role="dialog"
            aria-modal="true"
            aria-labelledby="vitrail-guide-heading"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="monument-spandrel-bar" aria-hidden="true">
              <span className="m-shard s-cobalt" />
              <span className="m-shard s-amber" />
              <span className="m-shard s-emerald" />
              <span className="m-shard s-ruby" />
              <span className="m-shard s-amethyst" />
              <span className="m-shard s-teal" />
              <span className="m-shard s-amber" />
            </div>

            <div className="codex-header-course">
              <div className="codex-title-quarry">
                <RoseHallmarkSvg />
                <div>
                  <h2 id="vitrail-guide-heading" className="etched-heading">
                    Vitrail Studio Guide
                  </h2>
                  <span className="etched-subtitle">
                    Acoustic Tap-Testing · Spring Extraction · Voronoi Glass Fracture
                  </span>
                </div>
              </div>
              <QuarryButton
                tone="crystal"
                soundTone="default"
                onAudioDown={playUiDown}
                onAudioUp={playUiUp}
                className="codex-close-quarry"
                onClick={() => setShowHelp(false)}
                aria-label="Close Guide"
              >
                <EtchedCloseSvg />
              </QuarryButton>
            </div>

            <div className="codex-courses">
              <div className="codex-step-row">
                <div className="codex-diagram-quarry">
                  <CodexStepMedallionSvg step={1} />
                </div>
                <div className="codex-text-quarry">
                  <h4>1. Scout Loose Glass by Acoustic Resonance</h4>
                  <p>
                    Each 3-prism course has micro-tolerance variance: 1 prism sits free while 2
                    bear the column load. <strong>Click</strong> any prism below the highest
                    completed course without locking your turn — free glass rings clearly at{' '}
                    <strong>2.3 kHz</strong>, while pinned glass ticks dryly at{' '}
                    <strong>3.8 kHz</strong>.
                  </p>
                </div>
              </div>

              <div className="codex-step-row">
                <div className="codex-diagram-quarry">
                  <CodexStepMedallionSvg step={2} />
                </div>
                <div className="codex-text-quarry">
                  <h4>2. Draw Smoothly Along the Channel</h4>
                  <p>
                    <strong>Drag horizontally</strong> along a prism’s length to slide it via a
                    spring tether. Watch the 12-pane stained-glass tension strip and the{' '}
                    <strong>Plumb Compass</strong> — if a piece binds red, release before clearing
                    the stack to choose a safer prism.
                  </p>
                </div>
              </div>

              <div className="codex-step-row">
                <div className="codex-diagram-quarry">
                  <CodexStepMedallionSvg step={3} />
                </div>
                <div className="codex-text-quarry">
                  <h4>3. Free-Hand Stack &amp; Guard Against Fracture</h4>
                  <p>
                    Position the extracted prism anywhere across the top course (prisms abut flush
                    bevel-to-bevel). Angle with the <strong>wheel</strong> or <strong>Q / E</strong>
                    , turn <strong>90°</strong> with <strong>R</strong> (<strong>0</strong> aligns
                    square), and <strong>click</strong> to release gently. Hard drops craze internal
                    spiderwebs or shatter into 3D Voronoi shards.
                  </p>
                </div>
              </div>
            </div>

            <div className="codex-footer-course">
              <div className="codex-hint-quarry">
                <span>Right-Click or background drag orbits 360° · Scroll zooms</span>
              </div>
              <QuarryButton
                tone="amber"
                active
                soundTone="primary"
                onAudioDown={playUiDown}
                onAudioUp={playUiUp}
                className="codex-enter-quarry"
                onClick={() => setShowHelp(false)}
              >
                <span>Return to Table</span>
              </QuarryButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

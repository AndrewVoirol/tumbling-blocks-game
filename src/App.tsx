import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  HelpCircle,
  Hammer,
  RotateCcw,
  Sparkles,
  Sun,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';
import { GameEngine } from './engine/GameEngine';
import {
  INITIAL_LAYERS,
  LIGHTING_PRESETS,
  STAINED_GLASS_PALETTE,
} from './physics/constants';
import { GameMode, TowerTelemetry } from './physics/PhysicsWorld';

const PLAYER_COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#10b981'];

const INITIAL_TELEMETRY: TowerTelemetry = {
  turnPhase: 'SELECT_BLOCK',
  gameMode: 'solo',
  currentPlayer: 0,
  playerCount: 2,
  blocksStacked: 0,
  currentLayers: INITIAL_LAYERS,
  peakHeightCm: 27.0,
  leanAngleDeg: 0.0,
  wobbleSpeedDegPerSec: 0.0,
  centerOfMassXZ: [0, 0],
  shatteredCount: 0,
  intactCount: 54,
  activeShardCount: 0,
  grabResistance01: 0,
  hoveredBlockId: null,
  activeBlockId: null,
  hoveredSlotIndex: null,
  settleProgress01: 0,
  collapseReason: null,
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

  const handleModeChange = (mode: GameMode) => {
    engineRef.current?.resetGame(mode, telemetry.playerCount);
  };

  const handlePlayerCountDelta = (delta: number) => {
    const next = Math.max(2, Math.min(4, telemetry.playerCount + delta));
    engineRef.current?.resetGame('party', next);
  };

  const handleToggleMute = () => {
    const eng = engineRef.current;
    if (!eng) return;
    const next = !muted;
    eng.audio.setMuted(next);
    setMuted(next);
  };

  const handleLightingChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setLightPreset(id);
    engineRef.current?.setLightingPreset(id);
  };

  const handleResetView = () => {
    engineRef.current?.camera.resetView(telemetry.peakHeightCm);
  };

  const handleRebuild = () => {
    engineRef.current?.resetGame(telemetry.gameMode, telemetry.playerCount);
  };

  // Plumb-bob gauge coordinates
  const comXClamp = Math.max(-16, Math.min(16, telemetry.centerOfMassXZ[0] * 7.5));
  const comZClamp = Math.max(-16, Math.min(16, telemetry.centerOfMassXZ[1] * 7.5));
  const leanSeverity =
    telemetry.leanAngleDeg > 3.2 || telemetry.wobbleSpeedDegPerSec > 12
      ? 'danger'
      : telemetry.leanAngleDeg > 1.3 || telemetry.wobbleSpeedDegPerSec > 4
      ? 'warn'
      : 'calm';

  const plumbColor =
    leanSeverity === 'danger'
      ? '#ef4444'
      : leanSeverity === 'warn'
      ? '#f59e0b'
      : '#10b981';

  // Hovered block info for immediate clarity
  const hoveredBlock =
    telemetry.hoveredBlockId !== null && engineRef.current?.physics
      ? engineRef.current.physics.blocks[telemetry.hoveredBlockId]
      : null;
  const hoveredSwatch = hoveredBlock ? STAINED_GLASS_PALETTE[hoveredBlock.swatchId] : null;

  const resistanceLabel =
    telemetry.grabResistance01 < 0.34
      ? 'Loose — Slides Smoothly'
      : telemetry.grabResistance01 < 0.7
      ? 'Snug — Watch Tower Lean'
      : 'Pinned — Load-Bearing!';

  const resistanceColor =
    telemetry.grabResistance01 < 0.34
      ? '#38bdf8'
      : telemetry.grabResistance01 < 0.7
      ? '#f59e0b'
      : '#ef4444';

  return (
    <div className="app-shell">
      <canvas ref={canvasRef} className="webgpu-canvas" />

      {gpuError && (
        <div className="webgpu-error-banner">
          <div className="glass-panel help-card">
            <h2 className="help-title">WebGPU Required</h2>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              {gpuError} Please open this game in a WebGPU-enabled browser (Chrome 113+, Edge 113+,
              or Safari 18+ with WebGPU enabled).
            </p>
          </div>
        </div>
      )}

      <div className="hud-layer">
        {/* Top Row */}
        <div className="hud-top-row">
          {/* Top-Left Brand, Mode & Score Panel */}
          <div className="glass-panel brand-panel">
            <div className="brand-header">
              <span className="brand-title">VITRAIL</span>
              <span className="scale-badge">7.5 × 2.5 × 1.5 cm · 1:1</span>
            </div>

            <div className="mode-tabs" role="tablist" aria-label="Game Mode">
              <button
                className={`mode-tab ${telemetry.gameMode === 'solo' ? 'active' : ''}`}
                onClick={() => handleModeChange('solo')}
              >
                Solo Tower
              </button>
              <button
                className={`mode-tab ${telemetry.gameMode === 'party' ? 'active' : ''}`}
                onClick={() => handleModeChange('party')}
              >
                Pass & Play
              </button>
              <button
                className={`mode-tab ${telemetry.gameMode === 'sandbox' ? 'active' : ''}`}
                onClick={() => handleModeChange('sandbox')}
              >
                Sandbox
              </button>
            </div>

            {telemetry.gameMode === 'party' && (
              <div className="turn-banner">
                <div className="player-indicator">
                  <span
                    className="player-dot"
                    style={{
                      color: PLAYER_COLORS[telemetry.currentPlayer % PLAYER_COLORS.length],
                      backgroundColor: PLAYER_COLORS[telemetry.currentPlayer % PLAYER_COLORS.length],
                    }}
                  />
                  <span>
                    {telemetry.turnPhase === 'COLLAPSED'
                      ? `Player ${((telemetry.currentPlayer + telemetry.playerCount - 1) % telemetry.playerCount) + 1} Wins!`
                      : `Player ${telemetry.currentPlayer + 1}'s Turn`}
                  </span>
                </div>
                <div className="player-stepper">
                  <span>{telemetry.playerCount}P</span>
                  <button
                    className="mini-btn"
                    onClick={() => handlePlayerCountDelta(-1)}
                    title="Fewer players"
                  >
                    −
                  </button>
                  <button
                    className="mini-btn"
                    onClick={() => handlePlayerCountDelta(1)}
                    title="More players"
                  >
                    +
                  </button>
                </div>
              </div>
            )}

            <div className="stats-grid">
              <div className="stat-cell">
                <span className="stat-label">Height</span>
                <span className="stat-value">{(telemetry.currentLayers * 1.5).toFixed(1)} cm</span>
                <span className="stat-sub">{telemetry.currentLayers} Rows</span>
              </div>
              <div className="stat-cell">
                <span className="stat-label">Stacked</span>
                <span className="stat-value">{telemetry.blocksStacked}</span>
                <span className="stat-sub">Peak {telemetry.peakHeightCm.toFixed(1)} cm</span>
              </div>
              <div className="stat-cell">
                <span className="stat-label">Shattered</span>
                <span
                  className="stat-value"
                  style={{ color: telemetry.shatteredCount > 0 ? '#fca5a5' : undefined }}
                >
                  {telemetry.shatteredCount}
                </span>
                <span className="stat-sub">{telemetry.activeShardCount} Shards</span>
              </div>
            </div>
          </div>

          {/* Top-Right Toolbar & Spirit-Level Plumb Gauge */}
          <div className="right-controls-stack">
            <div className="glass-panel toolbar-panel">
              <Sun size={15} style={{ color: '#fde68a', marginLeft: 4 }} />
              <select
                id="lighting-preset-select"
                name="lightingPreset"
                className="light-select"
                value={lightPreset}
                onChange={handleLightingChange}
                aria-label="Cathedral Window Lighting"
              >
                {LIGHTING_PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>

              {telemetry.gameMode === 'sandbox' && (
                <button
                  className="icon-btn danger-accent"
                  onClick={() => engineRef.current?.triggerCollapseDemo()}
                  title="Knock lower tower to watch stained-glass collapse & shatter"
                >
                  <Hammer size={15} />
                  <span>Smash Tower</span>
                </button>
              )}

              <button className="icon-btn" onClick={handleResetView} title="Center Camera">
                <Camera size={15} />
              </button>

              <button
                className="icon-btn"
                onClick={handleToggleMute}
                title={muted ? 'Unmute Glass Audio' : 'Mute Glass Audio'}
              >
                {muted ? <VolumeX size={15} /> : <Volume2 size={15} />}
              </button>

              <button className="icon-btn" onClick={handleRebuild} title="Rebuild New Tower">
                <RotateCcw size={15} />
                <span>Rebuild</span>
              </button>

              <button
                className="icon-btn"
                onClick={() => setShowHelp(true)}
                title="How to Play & Controls"
              >
                <HelpCircle size={15} />
              </button>
            </div>

            {/* Live Spirit-Level Plumb Gauge */}
            <div className="glass-panel plumb-panel" title="Live Tower Plumb & Wobble Gauge">
              <div className="plumb-dial">
                <div className="plumb-ring-inner" />
                <div
                  className="plumb-dot"
                  style={{
                    left: `calc(50% + ${comXClamp}px)`,
                    top: `calc(50% + ${comZClamp}px)`,
                    color: plumbColor,
                    backgroundColor: plumbColor,
                  }}
                />
              </div>
              <div className="plumb-readout">
                <span className="plumb-title">Tower Plumb</span>
                <span className="plumb-Numbers" style={{ color: plumbColor }}>
                  {telemetry.leanAngleDeg.toFixed(1)}° Tilt
                </span>
                <span className="plumb-status" style={{ color: 'var(--text-muted)' }}>
                  {telemetry.turnPhase === 'COLLAPSED'
                    ? 'Tower Toppled'
                    : telemetry.wobbleSpeedDegPerSec > 3.0
                    ? `Wobbling (${telemetry.wobbleSpeedDegPerSec.toFixed(0)}°/s)`
                    : telemetry.leanAngleDeg > 1.8
                    ? 'Off-Center Lean'
                    : 'Steady Equilibrium'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Row: Dynamic Context Coach OR Collapse Summary */}
        <div className="hud-bottom-row">
          {telemetry.turnPhase === 'COLLAPSED' ? (
            <div className="glass-panel collapse-card">
              <div className="collapse-title">TOWER SHATTERED</div>
              <p className="collapse-desc">
                {telemetry.collapseReason} Reached{' '}
                <strong>{telemetry.peakHeightCm.toFixed(1)} cm</strong> ({telemetry.currentLayers}{' '}
                rows) with <strong>{telemetry.blocksStacked}</strong> stained-glass blocks stacked
                and <strong>{telemetry.shatteredCount}</strong> fractured into{' '}
                <strong>{telemetry.activeShardCount}</strong> refractive shards.
              </p>
              <button className="primary-action-btn" onClick={handleRebuild}>
                <RotateCcw size={15} />
                <span>Rebuild Stained-Glass Tower</span>
              </button>
            </div>
          ) : telemetry.turnPhase === 'EXTRACTED' ? (
            <div className="glass-panel coach-pill">
              <div className="coach-main-line">
                <span className="step-chip place">Step 2 / 2 · Stack on Top</span>
                <span className="coach-instruction">
                  Aim at one of the glowing slots on top and click to gently release
                </span>
                <button
                  className="primary-action-btn"
                  onClick={() => {
                    engineRef.current?.physics.releaseExtractedBlockOntoTop();
                  }}
                >
                  <Sparkles size={14} />
                  <span>Drop on Slot</span>
                </button>
              </div>
              <div className="coach-subline">
                <span>Drag outside the tower to orbit & check which side needs counter-balance</span>
              </div>
            </div>
          ) : telemetry.turnPhase === 'PULLING' ? (
            <div className="glass-panel coach-pill">
              <div className="coach-main-line">
                <span className="step-chip">Sliding Block</span>
                <div className="resistance-bar-wrap">
                  <div className="resistance-track">
                    <div
                      className="resistance-fill"
                      style={{
                        width: `${Math.max(8, telemetry.grabResistance01 * 100)}%`,
                        backgroundColor: resistanceColor,
                      }}
                    />
                  </div>
                </div>
                <span className="coach-instruction" style={{ color: resistanceColor }}>
                  {resistanceLabel}
                </span>
              </div>
              <div className="coach-subline">
                <span>
                  Slide horizontally until the block clears the tower · Release anytime to try a
                  looser block
                </span>
              </div>
            </div>
          ) : telemetry.turnPhase === 'SETTLING' ? (
            <div className="glass-panel coach-pill">
              <div className="coach-main-line">
                <span className="step-chip place">Hands Off</span>
                <span className="coach-instruction">Waiting for tower wobble to settle…</span>
                <div className="resistance-bar-wrap" style={{ width: 120 }}>
                  <div className="resistance-track">
                    <div
                      className="resistance-fill"
                      style={{
                        width: `${telemetry.settleProgress01 * 100}%`,
                        backgroundColor: '#10b981',
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-panel coach-pill">
              <div className="coach-main-line">
                <span className="step-chip">
                  {telemetry.gameMode === 'sandbox' ? 'Free Sandbox' : 'Step 1 / 2 · Extract'}
                </span>
                <span className="coach-instruction">
                  Click any block to tap-test looseness · Drag horizontally to slide it out
                </span>
              </div>
              <div className="coach-subline">
                {hoveredBlock && hoveredSwatch ? (
                  <span className="swatch-tag">
                    <span
                      className="swatch-dot"
                      style={{ backgroundColor: hoveredSwatch.uiHex }}
                    />
                    <span>
                      {hoveredSwatch.name} · Row {hoveredBlock.assignedLayer + 1} ·{' '}
                      {hoveredBlock.isLegalTarget ? 'Ready to Pull' : 'Top Row Locked'}
                    </span>
                  </span>
                ) : (
                  <span>
                    Drag background (or Right-Click) to orbit 360° · Scroll to zoom · Top row is
                    locked
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Unobtrusive Quick Help Modal */}
      {showHelp && (
        <div className="help-modal-backdrop" onClick={() => setShowHelp(false)}>
          <div className="glass-panel help-card" onClick={(e) => e.stopPropagation()}>
            <div className="help-header">
              <h2 className="help-title">How to Play Vitrail</h2>
              <button className="icon-btn" onClick={() => setShowHelp(false)}>
                <X size={15} />
              </button>
            </div>

            <div className="help-steps">
              <div className="help-step">
                <div className="help-num">1</div>
                <div className="help-step-body">
                  <h4>Tap-Test to Find Loose Glass Blocks</h4>
                  <p>
                    Just like real Jenga, micro-tolerances make 1 or 2 blocks per row carry the
                    tower’s weight while another sits loose. <strong>Click</strong> a block below
                    the top row to poke it and see if it glides freely.
                  </p>
                </div>
              </div>

              <div className="help-step">
                <div className="help-num">2</div>
                <div className="help-step-body">
                  <h4>Slide Gently & Watch the Resistance Gauge</h4>
                  <p>
                    <strong>Click and drag</strong> a block horizontally along its channel. If the
                    tension filament glows red (<em>Pinned</em>), let go gently and pick a looser
                    piece before the tower wobbles over!
                  </p>
                </div>
              </div>

              <div className="help-step">
                <div className="help-num">3</div>
                <div className="help-step-body">
                  <h4>Stack on Top & Counter-Balance the Lean</h4>
                  <p>
                    Once pulled clear, move over the glowing slots on top of the tower and{' '}
                    <strong>click</strong> (or press <em>Drop on Slot</em>) to stack it at 90°. Use
                    the <strong>Tower Plumb</strong> gauge to counter-balance leans!
                  </p>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button className="primary-action-btn" onClick={() => setShowHelp(false)}>
                Start Playing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

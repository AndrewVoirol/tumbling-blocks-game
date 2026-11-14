import React, { useCallback, useEffect, useRef, useState } from 'react';
import './cathedralPlinth.css';
import { INITIAL_LAYERS, LIGHTING_PRESETS, STAINED_GLASS_PALETTE } from '../physics/constants';
import { isValidRoomCode, normalizeRoomCode } from '../multiplayer';
import type { GameMode } from '../physics/PhysicsWorld';
import type { AiDifficulty, VitrailHudTelemetry } from '../engine/GameEngine';
import type { QuarryButtonProps, QuarryGlassTone, UiSoundTone } from '../App';
import {
  AcousticCrystalSvg,
  ClerestoryPaneMosaicSvg,
  EtchedCloseSvg,
  EtchedCrystalCompassSvg,
  GuideQuatrefoilSvg,
  LancetArchApexSvg,
  ModeMosaicMedallionSvg,
  ModeRightJewelSvg,
  NameplateMarginInlaySvg,
  PlacementGlyphSvg,
  RebuildTowerSvg,
  SmashVoronoiSvg,
} from './VitrailGlassArt';

export type TouchInspectStatus =
  | 'loose'
  | 'snug'
  | 'pinned'
  | 'locked_top'
  | 'locked_support'
  | 'sandbox';

export interface TouchBlockInspectInfo {
  seq: number;
  blockId: number;
  swatchId: number;
  assignedLayer: number;
  damage: number;
  status: TouchInspectStatus;
}

export type PlinthDetent = 'closed' | 'half' | 'full';

export interface CathedralPlinthHudProps {
  telemetry: VitrailHudTelemetry;
  muted: boolean;
  lightPreset: string;
  showHelp: boolean;
  onlineDrawerOpen: boolean;
  joinCodeDraft: string;
  copiedRoomLink: boolean;
  onToggleMute: () => void;
  onSelectLightPreset: (presetId: string) => void;
  onTogglePolariscope: () => void;
  onSelectMode: (mode: GameMode) => void;
  onSetPlayerCount: (count: number) => void;
  onToggleVsAi: () => void;
  onSelectAiDifficulty: (difficulty: AiDifficulty) => void;
  onToggleOnlineDrawer: () => void;
  onJoinCodeDraftChange: (code: string) => void;
  onHostOnline: () => void;
  onJoinOnline: () => void;
  onLeaveOnline: () => void;
  onCopyInviteLink: () => void;
  onRequestOnlineRematch: () => void;
  onRebuild: () => void;
  onTriggerSmash: () => void;
  onOpenHelp: () => void;
  onRotateCarried: (deltaYawRad: number) => void;
  onResetCarriedRotation: () => void;
  onReleaseOntoTop: () => void;
  onOverlayOpenChange: (open: boolean) => void;
  onBottomInsetChange: (fraction01: number) => void;
  isExternalModalOpen?: boolean;
  keystoneButtonRef: React.RefObject<HTMLButtonElement | null>;
  playUiDown: (tone: UiSoundTone) => void;
  playUiUp: (tone: UiSoundTone) => void;
  QuarryButton: React.ForwardRefExoticComponent<
    QuarryButtonProps & React.RefAttributes<HTMLButtonElement>
  >;
}

const SUPPRESSED_INSPECT_PHASES = new Set([
  'PULLING',
  'SETTLING',
  'EXTRACTED',
  'COLLAPSED',
]);

function getLightPresetGemHex(presetId: string): string {
  switch (presetId) {
    case 'noon-clerestory':
      return '#7ec8ff';
    case 'vesper-rose':
      return '#ff8866';
    case 'candleglow-vesper':
      return '#ffbe58';
    case 'moonlight-lancet':
      return '#98b4ff';
    default:
      return '#a8c8ff';
  }
}

function getTouchInspectStatusLabel(status: TouchInspectStatus): string {
  switch (status) {
    case 'loose':
      return 'Free Ring · Loose';
    case 'snug':
      return 'Snug · Bearing';
    case 'pinned':
      return 'Pinned · Column';
    case 'locked_top':
      return 'Top Row Locked';
    case 'locked_support':
      return 'Support Row Locked';
    case 'sandbox':
      return 'Unlocked · Free';
  }
}

function getCompactStatusSummary(p: CathedralPlinthHudProps): {
  modeShort: string;
  modeTone: QuarryGlassTone;
  artMode: 'online' | 'party' | 'sandbox' | 'solo';
  jewelColor: string;
  courseLabel: string;
  isOnlineSession: boolean;
  isOnlineTabSelected: boolean;
  isLocalPartySelected: boolean;
} {
  const t = p.telemetry;
  const isOnlineSession = Boolean(
    t.mpRoomCode || t.mpRole || (t.mpConnectionState && t.mpConnectionState !== 'idle')
  );
  const isOnlineTabSelected = isOnlineSession || p.onlineDrawerOpen;
  const isLocalPartySelected =
    !isOnlineTabSelected && t.gameMode === 'party' && !t.aiOpponentEnabled;

  let modeShort = 'Solo';
  let modeTone: QuarryGlassTone = 'cobalt';
  let artMode: 'online' | 'party' | 'sandbox' | 'solo' = 'solo';
  let jewelColor = '#6fb2ff';

  if (isOnlineSession) {
    modeShort = t.mpRole === 'guest' ? 'Online · P2' : 'Online · P1';
    modeTone = 'cobalt';
    artMode = 'online';
    jewelColor = t.mpPeerConnected ? '#68e2a8' : '#78b4ff';
  } else if (p.onlineDrawerOpen) {
    modeShort = 'Online Room';
    modeTone = 'cobalt';
    artMode = 'online';
    jewelColor = '#78b4ff';
  } else if (t.gameMode === 'sandbox') {
    modeShort = 'Sandbox';
    modeTone = 'ruby';
    artMode = 'sandbox';
    jewelColor = '#ff7285';
  } else if (t.aiOpponentEnabled) {
    modeShort = t.aiDifficulty === 'apprentice' ? 'Apprentice AI' : 'Glazier AI';
    modeTone = 'amethyst';
    artMode = 'party';
    jewelColor = '#c496ff';
  } else if (isLocalPartySelected) {
    modeShort = `${t.playerCount}P Pass & Play`;
    modeTone = 'amber';
    artMode = 'party';
    jewelColor = '#ffbd66';
  }

  const courseLabel =
    t.currentLayers === 0
      ? 'Toppled'
      : `Course ${t.currentLayers} · ${t.intactCount}/54`;

  return {
    modeShort,
    modeTone,
    artMode,
    jewelColor,
    courseLabel,
    isOnlineSession,
    isOnlineTabSelected,
    isLocalPartySelected,
  };
}

function getPlumbViewModel(telemetry: VitrailHudTelemetry): {
  isToppled: boolean;
  comXClamp: number;
  comZClamp: number;
  plumbColor: string;
} {
  const isToppled = telemetry.turnPhase === 'COLLAPSED' || telemetry.currentLayers === 0;
  const [comX, comZ] = telemetry.centerOfMassXZ ?? [0, 0];
  const clampPx = (v: number) => Math.max(-18, Math.min(18, v * 3.8));
  let plumbColor = '#6be6a8';
  if (isToppled) {
    plumbColor = '#ff6d80';
  } else if (telemetry.leanAngleDeg > 2.4 || telemetry.wobbleSpeedDegPerSec > 4.5) {
    plumbColor = '#ffbe5c';
  }
  return {
    isToppled,
    comXClamp: clampPx(comX),
    comZClamp: clampPx(comZ),
    plumbColor,
  };
}

function isInteractiveCanAct(telemetry: VitrailHudTelemetry): boolean {
  if (telemetry.aiOpponentEnabled && telemetry.currentPlayer === 1) return false;
  if (telemetry.mpRole === 'host') {
    return (
      telemetry.currentPlayer === 0 &&
      telemetry.turnPhase !== 'COLLAPSED' &&
      telemetry.turnPhase !== 'SETTLING'
    );
  }
  if (telemetry.mpRole === 'guest') {
    return (
      telemetry.currentPlayer === 1 &&
      telemetry.turnPhase !== 'COLLAPSED' &&
      telemetry.turnPhase !== 'SETTLING'
    );
  }
  return true;
}

export const CathedralTransomCrest: React.FC<{
  telemetry: VitrailHudTelemetry;
  isOnlineSession: boolean;
  mpPingMs: number | null;
}> = ({ telemetry, isOnlineSession, mpPingMs }) => {
  const isToppled = telemetry.turnPhase === 'COLLAPSED' || telemetry.currentLayers === 0;
  let badgeText = `P${telemetry.currentPlayer + 1}`;
  let gemHex = '#72b6ff';

  if (isToppled) {
    badgeText = 'Toppled';
    gemHex = '#ff6d80';
  } else if (telemetry.gameMode === 'sandbox') {
    badgeText = 'Sandbox';
    gemHex = '#ff8a66';
  } else if (telemetry.aiOpponentEnabled) {
    badgeText = telemetry.currentPlayer === 1 ? 'Glazier AI' : 'You (P1)';
    gemHex = telemetry.currentPlayer === 1 ? '#c598ff' : '#72b6ff';
  } else if (isOnlineSession) {
    const ping = mpPingMs !== null ? ` ${mpPingMs}ms` : '';
    badgeText = `${telemetry.mpRole === 'guest' ? 'P2' : 'P1'}${ping}`;
    gemHex = telemetry.mpPeerConnected ? '#6be6a8' : '#72b6ff';
  } else if (telemetry.gameMode === 'solo') {
    badgeText = `${telemetry.blocksStacked} Stacked`;
    gemHex = '#6be6a8';
  }

  return (
    <div className="plinth-transom" role="status" aria-live="polite" aria-label="Tower status">
      <div className="plinth-transom-frame">
        <span className="plinth-transom-brand">VITRAIL</span>
        <span className="plinth-transom-sep" aria-hidden="true" />
        <span className="plinth-transom-stat">
          {isToppled
            ? `${telemetry.blocksStacked} Stacked`
            : `${telemetry.currentLayers}L · ${telemetry.liveHeightCm.toFixed(1)}cm`}
        </span>
        <span className="plinth-transom-sep" aria-hidden="true" />
        <span className="plinth-transom-badge">
          <span
            className="plinth-transom-gem"
            aria-hidden="true"
            style={{ backgroundColor: gemHex, boxShadow: `0 0 6px ${gemHex}` }}
          />
          <span>{badgeText}</span>
        </span>
      </div>
    </div>
  );
};

export const CrystalTelemetryContent: React.FC<{
  p: CathedralPlinthHudProps;
  showPolariscopeButton?: boolean;
}> = ({ p, showPolariscopeButton = true }) => {
  const { telemetry, QuarryButton } = p;
  const plumb = getPlumbViewModel(telemetry);

  return (
    <div className="crystal-telemetry-panel">
      <div className="etched-plumb-card">
        <div className="etched-plumb-dial" aria-hidden="true">
          <EtchedCrystalCompassSvg />
          <span
            className="inlaid-plumb-jewel"
            style={{
              left: `calc(50% + ${plumb.comXClamp}px)`,
              top: `calc(50% + ${plumb.comZClamp}px)`,
              backgroundColor: plumb.plumbColor,
              color: plumb.plumbColor,
            }}
          />
        </div>
        <div className="etched-plumb-copy">
          <span className="etched-plumb-title">Plumb</span>
          <span className="etched-plumb-deg" style={{ color: plumb.plumbColor }}>
            {plumb.isToppled ? '—' : `${telemetry.leanAngleDeg.toFixed(1)}°`}
          </span>
          <span className="etched-plumb-status">
            {plumb.isToppled
              ? 'Toppled'
              : telemetry.wobbleSpeedDegPerSec > 3.0
              ? `${telemetry.wobbleSpeedDegPerSec.toFixed(0)}°/s sway`
              : telemetry.leanAngleDeg > 1.8
              ? 'Leaning'
              : 'Equilibrium'}
          </span>
        </div>
      </div>

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

      {showPolariscopeButton && (
        <div className="polariscope-lens-row">
          <QuarryButton
            tone={telemetry.polariscopeMode ? 'teal' : 'crystal'}
            active={Boolean(telemetry.polariscopeMode)}
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            className="polariscope-lens-btn"
            onClick={p.onTogglePolariscope}
            aria-pressed={Boolean(telemetry.polariscopeMode)}
            aria-label="Toggle Polariscope Stress-Optic Lens"
          >
            <span
              className={`inlaid-gem ${telemetry.polariscopeMode ? 'gem-teal' : 'gem-opal'}`}
              aria-hidden="true"
            />
            <span>Polariscope</span>
            <span className="polariscope-state-tag">
              {telemetry.polariscopeMode ? 'Isochromatic' : 'Natural'}
            </span>
          </QuarryButton>
        </div>
      )}
    </div>
  );
};

export const StudioControlsContent: React.FC<{
  p: CathedralPlinthHudProps;
  onOpenGuideFromSheet?: () => void;
}> = ({ p, onOpenGuideFromSheet }) => {
  const { telemetry, QuarryButton } = p;
  const summary = getCompactStatusSummary(p);
  const normalizedJoinCode = normalizeRoomCode(p.joinCodeDraft);
  const canJoin = isValidRoomCode(normalizedJoinCode);

  return (
    <>
      <section className="atelier-section" aria-label="Play Modes">
        <div className="atelier-section-title">Play Modes</div>
        <div className="atelier-btn-grid">
          <QuarryButton
            tone="cobalt"
            active={
              !summary.isOnlineTabSelected &&
              telemetry.gameMode === 'solo' &&
              !telemetry.aiOpponentEnabled
            }
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            className="mode-course-btn"
            onClick={() => p.onSelectMode('solo')}
          >
            Solo
          </QuarryButton>

          <QuarryButton
            tone="amber"
            active={summary.isLocalPartySelected}
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            className="mode-course-btn"
            onClick={() => p.onSelectMode('party')}
          >
            Pass &amp; Play
          </QuarryButton>

          <QuarryButton
            tone="amethyst"
            active={!summary.isOnlineTabSelected && Boolean(telemetry.aiOpponentEnabled)}
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            className="mode-course-btn"
            onClick={p.onToggleVsAi}
          >
            Glazier AI
          </QuarryButton>

          <QuarryButton
            tone="cobalt"
            active={summary.isOnlineTabSelected}
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            className="mode-course-btn"
            onClick={p.onToggleOnlineDrawer}
            aria-expanded={summary.isOnlineTabSelected}
          >
            Online Room
          </QuarryButton>

          <QuarryButton
            tone="ruby"
            active={!summary.isOnlineTabSelected && telemetry.gameMode === 'sandbox'}
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            className="mode-course-btn"
            onClick={() => p.onSelectMode('sandbox')}
          >
            Sandbox
          </QuarryButton>
        </div>

        {summary.isLocalPartySelected && (
          <div className="atelier-row-actions">
            {[2, 3, 4].map((count) => (
              <QuarryButton
                key={count}
                tone="amber"
                active={telemetry.playerCount === count}
                soundTone="tab"
                onAudioDown={p.playUiDown}
                onAudioUp={p.playUiUp}
                onClick={() => p.onSetPlayerCount(count)}
              >
                {count} Players
              </QuarryButton>
            ))}
          </div>
        )}

        {!summary.isOnlineTabSelected && Boolean(telemetry.aiOpponentEnabled) && (
          <div className="atelier-row-actions">
            <QuarryButton
              tone="amethyst"
              active={telemetry.aiDifficulty === 'apprentice'}
              soundTone="tab"
              onAudioDown={p.playUiDown}
              onAudioUp={p.playUiUp}
              onClick={() => p.onSelectAiDifficulty('apprentice')}
            >
              Apprentice
            </QuarryButton>
            <QuarryButton
              tone="amethyst"
              active={telemetry.aiDifficulty === 'master'}
              soundTone="tab"
              onAudioDown={p.playUiDown}
              onAudioUp={p.playUiUp}
              onClick={() => p.onSelectAiDifficulty('master')}
            >
              Master AI
            </QuarryButton>
          </div>
        )}

        {!summary.isOnlineTabSelected && telemetry.gameMode === 'sandbox' && (
          <div className="atelier-row-actions">
            <QuarryButton
              tone="ruby"
              soundTone="danger"
              onAudioDown={p.playUiDown}
              onAudioUp={p.playUiUp}
              onClick={p.onTriggerSmash}
            >
              <SmashVoronoiSvg />
              <span>Shatter Tower</span>
            </QuarryButton>
            <QuarryButton
              tone="amber"
              soundTone="primary"
              onAudioDown={p.playUiDown}
              onAudioUp={p.playUiUp}
              onClick={p.onRebuild}
            >
              <RebuildTowerSvg />
              <span>Rebuild</span>
            </QuarryButton>
          </div>
        )}
      </section>

      {summary.isOnlineTabSelected && (
        <section className="atelier-section" aria-label="Online Multiplayer">
          <div className="atelier-section-title">Online Room</div>
          <div className="online-room-box">
            {summary.isOnlineSession ? (
              <>
                <div className="online-code-banner">
                  <span className="online-status-copy">
                    {telemetry.mpRole === 'guest' ? 'Guest (P2)' : 'Host (P1)'}
                  </span>
                  <span className="online-code-value">{telemetry.mpRoomCode ?? '-----'}</span>
                </div>
                <div className="online-status-copy">
                  {telemetry.mpPeerConnected
                    ? `Peer connected${
                        telemetry.mpPingMs !== null && telemetry.mpPingMs !== undefined
                          ? ` · ${telemetry.mpPingMs} ms`
                          : ''
                      }`
                    : telemetry.mpConnectionState === 'reconnecting'
                    ? 'Reconnecting peer link…'
                    : 'Waiting for opponent to join room code…'}
                </div>
                {telemetry.mpError && (
                  <div className="online-status-copy" role="alert">
                    {telemetry.mpError}
                  </div>
                )}
                <div className="atelier-row-actions">
                  <QuarryButton
                    tone="cobalt"
                    soundTone="tab"
                    onAudioDown={p.playUiDown}
                    onAudioUp={p.playUiUp}
                    onClick={p.onCopyInviteLink}
                  >
                    {p.copiedRoomLink ? 'Copied' : 'Invite Link'}
                  </QuarryButton>
                  {telemetry.turnPhase === 'COLLAPSED' && (
                    <QuarryButton
                      tone="amber"
                      soundTone="primary"
                      onAudioDown={p.playUiDown}
                      onAudioUp={p.playUiUp}
                      onClick={p.onRequestOnlineRematch}
                    >
                      Rematch
                    </QuarryButton>
                  )}
                  <QuarryButton
                    tone="ruby"
                    soundTone="danger"
                    onAudioDown={p.playUiDown}
                    onAudioUp={p.playUiUp}
                    onClick={p.onLeaveOnline}
                  >
                    Leave Room
                  </QuarryButton>
                </div>
              </>
            ) : (
              <>
                <div className="atelier-row-actions">
                  <QuarryButton
                    tone="cobalt"
                    soundTone="primary"
                    onAudioDown={p.playUiDown}
                    onAudioUp={p.playUiUp}
                    className="online-host-btn"
                    onClick={p.onHostOnline}
                  >
                    Host Room
                  </QuarryButton>
                </div>
                <div className="online-join-row">
                  <input
                    id="online-room-code-input"
                    type="text"
                    maxLength={8}
                    value={p.joinCodeDraft}
                    placeholder="ROOM CODE"
                    aria-label="Enter 5-character online room code"
                    autoCapitalize="characters"
                    autoCorrect="off"
                    spellCheck={false}
                    onChange={(e) => p.onJoinCodeDraftChange(e.target.value.toUpperCase())}
                    onFocus={(e) => {
                      e.currentTarget.scrollIntoView({ block: 'center', behavior: 'smooth' });
                    }}
                  />
                  <QuarryButton
                    tone="emerald"
                    soundTone="primary"
                    onAudioDown={p.playUiDown}
                    onAudioUp={p.playUiUp}
                    className="online-join-btn"
                    disabled={!canJoin}
                    onClick={p.onJoinOnline}
                  >
                    Join
                  </QuarryButton>
                </div>
                {telemetry.mpError && (
                  <div className="online-status-copy" role="alert">
                    {telemetry.mpError}
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      )}

      <section className="atelier-section" aria-label="Lighting and Optics">
        <div className="atelier-section-title">Cathedral Light &amp; Optics</div>
        <div className="atelier-btn-grid">
          {LIGHTING_PRESETS.map((preset) => {
            const gemHex = getLightPresetGemHex(preset.id);
            return (
              <QuarryButton
                key={preset.id}
                tone="crystal"
                active={p.lightPreset === preset.id}
                soundTone="tab"
                onAudioDown={p.playUiDown}
                onAudioUp={p.playUiUp}
                onClick={() => p.onSelectLightPreset(preset.id)}
              >
                <span
                  className="inlaid-gem"
                  aria-hidden="true"
                  style={{ backgroundColor: gemHex, boxShadow: `0 0 6px ${gemHex}` }}
                />
                <span>{preset.name}</span>
              </QuarryButton>
            );
          })}
        </div>
        <div className="atelier-row-actions">
          <QuarryButton
            tone={telemetry.polariscopeMode ? 'teal' : 'crystal'}
            active={Boolean(telemetry.polariscopeMode)}
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            onClick={p.onTogglePolariscope}
            aria-pressed={Boolean(telemetry.polariscopeMode)}
          >
            <span>{telemetry.polariscopeMode ? 'Polariscope On' : 'Polariscope'}</span>
          </QuarryButton>
          <QuarryButton
            tone={p.muted ? 'ruby' : 'crystal'}
            active={p.muted}
            soundTone="default"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            onClick={p.onToggleMute}
            aria-pressed={p.muted}
          >
            <AcousticCrystalSvg muted={p.muted} />
            <span>{p.muted ? 'Sound Muted' : 'Glass Sound'}</span>
          </QuarryButton>
          <QuarryButton
            tone="amber"
            soundTone="primary"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            onClick={p.onRebuild}
            aria-label="Rebuild 54-Prism Stained-Glass Tower"
          >
            <RebuildTowerSvg />
            <span>Rebuild {INITIAL_LAYERS}L</span>
          </QuarryButton>
          <QuarryButton
            tone="opal"
            soundTone="tab"
            onAudioDown={p.playUiDown}
            onAudioUp={p.playUiUp}
            onClick={() => {
              if (onOpenGuideFromSheet) {
                onOpenGuideFromSheet();
              } else {
                p.onOpenHelp();
              }
            }}
          >
            <GuideQuatrefoilSvg />
            <span>Vitrail Guide</span>
          </QuarryButton>
        </div>
      </section>
    </>
  );
};

export const ContextualBottomGameplayBar: React.FC<{
  p: CathedralPlinthHudProps;
  detent: PlinthDetent;
  onModeButtonClick: () => void;
  onKeystoneClick: () => void;
}> = ({ p, detent, onModeButtonClick, onKeystoneClick }) => {
  const { telemetry, QuarryButton, keystoneButtonRef } = p;
  const summary = getCompactStatusSummary(p);
  const canAct = isInteractiveCanAct(telemetry);

  if (telemetry.turnPhase === 'EXTRACTED' && canAct) {
    return (
      <div className="plinth-place-keys" role="group" aria-label="Prism Placement Controls">
        <QuarryButton
          tone="crystal"
          soundTone="step"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-place-btn"
          onClick={() => p.onRotateCarried(-Math.PI / 24)}
          aria-label="Rotate prism counter-clockwise"
        >
          <span className="place-glyph-svg" aria-hidden="true">
            <PlacementGlyphSvg kind="ccw" />
          </span>
          <span>Q</span>
        </QuarryButton>

        <QuarryButton
          tone="crystal"
          soundTone="step"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-place-turn90"
          onClick={() => p.onRotateCarried(Math.PI * 0.5)}
          aria-label="Rotate prism 90 degrees"
        >
          <span className="place-glyph-svg" aria-hidden="true">
            <PlacementGlyphSvg kind="turn90" />
          </span>
          <span>90°</span>
        </QuarryButton>

        <QuarryButton
          tone="emerald"
          active
          soundTone="primary"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-place-submit prism-release-btn"
          onClick={p.onReleaseOntoTop}
          aria-label="Place Prism onto top course"
        >
          <span className="place-glyph-svg" aria-hidden="true">
            <PlacementGlyphSvg kind="place" />
          </span>
          <span className="plinth-place-label-full">Place Prism</span>
          <span className="plinth-place-label-compact" aria-hidden="true">
            Place
          </span>
        </QuarryButton>

        <QuarryButton
          tone="crystal"
          soundTone="step"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-place-btn"
          onClick={() => p.onRotateCarried(Math.PI / 24)}
          aria-label="Rotate prism clockwise"
        >
          <span>E</span>
          <span className="place-glyph-svg" aria-hidden="true">
            <PlacementGlyphSvg kind="cw" />
          </span>
        </QuarryButton>

        <QuarryButton
          tone="opal"
          soundTone="step"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-place-btn"
          onClick={p.onResetCarriedRotation}
          aria-label="Align prism to course axis"
        >
          <span className="place-glyph-svg" aria-hidden="true">
            <PlacementGlyphSvg kind="align" />
          </span>
          <span>0°</span>
        </QuarryButton>
      </div>
    );
  }

  if (telemetry.turnPhase === 'COLLAPSED') {
    return (
      <div className="plinth-sill-row">
        <div className="plinth-status-quarry" role="status">
          <span className="plinth-status-quarry-text">
            {telemetry.shatteredCount > 0
              ? `Collapsed · ${telemetry.shatteredCount} shattered`
              : `Toppled · ${telemetry.blocksStacked} stacked`}
          </span>
        </div>
        <QuarryButton
          tone="amber"
          active
          soundTone="primary"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-sill-action keystone-cta"
          onClick={summary.isOnlineSession ? p.onRequestOnlineRematch : p.onRebuild}
          aria-label={
            summary.isOnlineSession
              ? 'Request online rematch'
              : 'Rebuild 54-Prism Stained-Glass Tower'
          }
        >
          <RebuildTowerSvg />
          <span>{summary.isOnlineSession ? 'Rematch' : 'Rebuild'}</span>
        </QuarryButton>
        <QuarryButton
          ref={keystoneButtonRef}
          tone="opal"
          active={detent === 'full'}
          soundTone="tab"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-sill-keystone"
          onClick={onKeystoneClick}
          aria-label="Open Studio Controls and Rules"
          aria-expanded={detent === 'full'}
        >
          <ClerestoryPaneMosaicSvg presetId={p.lightPreset} active={detent === 'full'} />
        </QuarryButton>
      </div>
    );
  }

  if (
    telemetry.turnPhase === 'PULLING' ||
    telemetry.turnPhase === 'SETTLING' ||
    !canAct
  ) {
    let statusCopy = 'Settling equilibrium…';
    if (telemetry.aiThinking) {
      statusCopy = telemetry.aiActionLabel || 'Glazier AI inspecting ring…';
    } else if (telemetry.mpRole && !canAct) {
      statusCopy = telemetry.mpPeerConnected ? 'Opponent turn…' : 'Awaiting peer…';
    } else if (telemetry.turnPhase === 'PULLING') {
      statusCopy =
        telemetry.grabResistance01 > 0.72
          ? 'Heavy column bind · ease back'
          : telemetry.grabResistance01 > 0.35
          ? 'Snug slide · steady draw'
          : 'Free prism sliding…';
    }

    return (
      <div className="plinth-sill-row">
        <div className="plinth-status-quarry" role="status">
          <span className="plinth-status-quarry-text">{statusCopy}</span>
        </div>
        <QuarryButton
          ref={keystoneButtonRef}
          tone="opal"
          active={detent === 'full'}
          soundTone="tab"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-sill-keystone"
          onClick={onKeystoneClick}
          aria-label="Open Studio Controls and Rules"
          aria-expanded={detent === 'full'}
        >
          <ClerestoryPaneMosaicSvg presetId={p.lightPreset} active={detent === 'full'} />
        </QuarryButton>
      </div>
    );
  }

  return (
    <div className="plinth-sill-row">
      <QuarryButton
        tone={summary.modeTone}
        active={detent === 'half'}
        soundTone="tab"
        onAudioDown={p.playUiDown}
        onAudioUp={p.playUiUp}
        className="plinth-sill-mode"
        onClick={onModeButtonClick}
        aria-label={`Studio tray (${summary.modeShort})`}
        aria-expanded={detent !== 'closed'}
      >
        <ModeMosaicMedallionSvg mode={summary.artMode} active={detent !== 'closed'} />
        <span className="plinth-mode-text-wrap">
          <span className="plinth-mode-name">{summary.modeShort}</span>
          <span className="mode-course-meta">{summary.courseLabel}</span>
        </span>
        <ModeRightJewelSvg mode={summary.artMode} active={detent !== 'closed'} />
      </QuarryButton>

      {telemetry.gameMode === 'sandbox' && (
        <QuarryButton
          tone="ruby"
          soundTone="danger"
          onAudioDown={p.playUiDown}
          onAudioUp={p.playUiUp}
          className="plinth-sill-action"
          onClick={p.onTriggerSmash}
          aria-label="Shatter tower in sandbox mode"
        >
          <SmashVoronoiSvg />
        </QuarryButton>
      )}

      <QuarryButton
        tone="amber"
        soundTone="primary"
        onAudioDown={p.playUiDown}
        onAudioUp={p.playUiUp}
        className="plinth-sill-action"
        onClick={p.onRebuild}
        aria-label="Rebuild 54-Prism Stained-Glass Tower"
      >
        <RebuildTowerSvg />
      </QuarryButton>

      <QuarryButton
        ref={keystoneButtonRef}
        tone="opal"
        active={detent === 'full'}
        soundTone="tab"
        onAudioDown={p.playUiDown}
        onAudioUp={p.playUiUp}
        className="plinth-sill-keystone"
        onClick={onKeystoneClick}
        aria-label="Open Studio Controls and Rules"
        aria-expanded={detent === 'full'}
      >
        <ClerestoryPaneMosaicSvg presetId={p.lightPreset} active={detent === 'full'} />
      </QuarryButton>
    </div>
  );
};

export const CathedralPlinthHud: React.FC<CathedralPlinthHudProps> = (p) => {
  const { telemetry, QuarryButton } = p;
  const summary = getCompactStatusSummary(p);

  const [detent, setDetent] = useState<PlinthDetent>('closed');
  const [halfTab, setHalfTab] = useState<'studio' | 'crystal'>('studio');
  const [dismissedMpError, setDismissedMpError] = useState<string | null>(null);
  const [visibleInspect, setVisibleInspect] = useState<TouchBlockInspectInfo | null>(null);

  const halfTrayRef = useRef<HTMLDivElement | null>(null);
  const halfCloseBtnRef = useRef<HTMLButtonElement | null>(null);
  const fullDialogRef = useRef<HTMLDivElement | null>(null);
  const fullCloseBtnRef = useRef<HTMLButtonElement | null>(null);

  const prevDetentRef = useRef<PlinthDetent>('closed');
  const prevTurnPhaseRef = useRef(telemetry.turnPhase);
  const wasFullOpenRef = useRef<boolean>(false);
  const pendingOnlineFocusRef = useRef<boolean>(false);

  const swipeStartRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const didSwipeRef = useRef<boolean>(false);

  const lastInspectKeyRef = useRef<string | null>(null);
  const inspectTimerRef = useRef<number | null>(null);

  useEffect(() => {
    p.onOverlayOpenChange(detent === 'full');
    return () => p.onOverlayOpenChange(false);
  }, [detent, p.onOverlayOpenChange]);

  // E1: Live Clerestory Tray Camera Framing via ResizeObserver on .plinth-half-tray.
  useEffect(() => {
    if (detent !== 'half') {
      p.onBottomInsetChange(0);
      return;
    }

    const trayEl = halfTrayRef.current;
    if (!trayEl) {
      p.onBottomInsetChange(0);
      return;
    }

    const measure = () => {
      const trayRect = trayEl.getBoundingClientRect();
      const appShell = trayEl.closest('.app-shell') as HTMLElement | null;
      const appShellHeight = appShell ? appShell.getBoundingClientRect().height : window.innerHeight;
      const fraction = Math.min(
        0.40,
        Math.max(0, trayRect.height / Math.max(1, appShellHeight))
      );
      p.onBottomInsetChange(fraction);
    };

    measure();
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => measure());
      ro.observe(trayEl);
    }
    window.addEventListener('resize', measure);

    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener('resize', measure);
      p.onBottomInsetChange(0);
    };
  }, [detent, halfTab, summary.isOnlineSession, p.onBottomInsetChange]);

  // E2: Automatically tuck Detent 1 to closed on PULLING / EXTRACTED / COLLAPSED transitions.
  useEffect(() => {
    const prev = prevTurnPhaseRef.current;
    const next = telemetry.turnPhase;
    prevTurnPhaseRef.current = next;
    if (
      prev !== next &&
      (next === 'PULLING' || next === 'EXTRACTED' || next === 'COLLAPSED')
    ) {
      setDetent((cur) => (cur === 'half' ? 'closed' : cur));
    }
  }, [telemetry.turnPhase]);

  // E3: 2.2s auto-clearing touch tap-inspection strip.
  const rawTouchInspect =
    (telemetry as unknown as { touchInspect?: TouchBlockInspectInfo | null }).touchInspect ?? null;

  useEffect(() => {
    if (!rawTouchInspect || SUPPRESSED_INSPECT_PHASES.has(telemetry.turnPhase)) {
      if (inspectTimerRef.current !== null) {
        window.clearTimeout(inspectTimerRef.current);
        inspectTimerRef.current = null;
      }
      setVisibleInspect(null);
      if (!rawTouchInspect) {
        lastInspectKeyRef.current = null;
      }
      return;
    }

    const inspectKey = `${rawTouchInspect.seq}:${rawTouchInspect.blockId}:${rawTouchInspect.status}`;
    if (lastInspectKeyRef.current === inspectKey) return;

    lastInspectKeyRef.current = inspectKey;
    setVisibleInspect(rawTouchInspect);
    if (inspectTimerRef.current !== null) {
      window.clearTimeout(inspectTimerRef.current);
    }
    inspectTimerRef.current = window.setTimeout(() => {
      inspectTimerRef.current = null;
      setVisibleInspect(null);
    }, 2200);
  }, [rawTouchInspect, telemetry.turnPhase]);

  useEffect(() => {
    return () => {
      if (inspectTimerRef.current !== null) {
        window.clearTimeout(inspectTimerRef.current);
        inspectTimerRef.current = null;
      }
    };
  }, []);

  // E4: Reset error banner dismissal whenever mpError changes.
  useEffect(() => {
    setDismissedMpError(null);
  }, [telemetry.mpError]);

  // E6: Focus management, Escape cascade, and focus restoration.
  useEffect(() => {
    didSwipeRef.current = false;
    const prev = prevDetentRef.current;
    prevDetentRef.current = detent;

    if (detent === 'full') {
      wasFullOpenRef.current = true;
      if (pendingOnlineFocusRef.current) {
        pendingOnlineFocusRef.current = false;
        const roomInput = fullDialogRef.current?.querySelector<HTMLInputElement>(
          '#online-room-code-input'
        );
        if (roomInput) {
          roomInput.focus();
          return;
        }
      }
      fullCloseBtnRef.current?.focus();
      return;
    }

    if (prev === 'full' && detent === 'half') {
      halfCloseBtnRef.current?.focus();
      return;
    }

    if (detent === 'closed' && (prev === 'full' || wasFullOpenRef.current)) {
      wasFullOpenRef.current = false;
      if (!p.showHelp && !p.isExternalModalOpen) {
        p.keystoneButtonRef.current?.focus();
      }
    }
  }, [detent, p.showHelp, p.isExternalModalOpen, p.keystoneButtonRef]);

  useEffect(() => {
    if (detent === 'closed' || p.showHelp || p.isExternalModalOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        p.playUiUp('default');
        setDetent((cur) => (cur === 'full' ? 'half' : 'closed'));
        return;
      }

      if (e.key === 'Tab' && detent === 'full' && fullDialogRef.current) {
        const candidates = Array.from(
          fullDialogRef.current.querySelectorAll<HTMLElement>(
            'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => {
          const style = window.getComputedStyle(el);
          return style.display !== 'none' && style.visibility !== 'hidden';
        });

        if (candidates.length === 0) return;
        const first = candidates[0];
        const last = candidates[candidates.length - 1];
        const active = document.activeElement as HTMLElement | null;

        if (e.shiftKey) {
          if (!active || active === first || !fullDialogRef.current.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else if (!active || active === last || !fullDialogRef.current.contains(active)) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [detent, p.showHelp, p.isExternalModalOpen, p.playUiUp]);

  // E5: Swipe gesture tracking with strict vertical angle disambiguation.
  const handleSwipePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest('input, textarea, select')) return;
    const scrollHost = target?.closest('.sheet-half-body') as HTMLElement | null;
    if (scrollHost && scrollHost.scrollTop > 1) return;
    swipeStartRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY };
  };

  const handleSwipePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const start = swipeStartRef.current;
    if (!start || start.id !== e.pointerId) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    if (Math.abs(dy) < 28) return;
    if (Math.abs(dy) <= Math.abs(dx) * 1.15) return;

    swipeStartRef.current = null;
    didSwipeRef.current = true;
    requestAnimationFrame(() => {
      didSwipeRef.current = false;
    });

    if (dy < 0 && detent === 'closed') {
      p.playUiUp('tab');
      setDetent('half');
    } else if (dy > 0 && detent === 'half') {
      p.playUiUp('default');
      setDetent('closed');
    }
  };

  const handleSwipePointerEnd = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (swipeStartRef.current?.id === e.pointerId) {
      swipeStartRef.current = null;
    }
    if (didSwipeRef.current) {
      requestAnimationFrame(() => {
        didSwipeRef.current = false;
      });
    }
  }, []);

  const handleModeButtonClick = () => {
    if (didSwipeRef.current) return;
    if (
      summary.isOnlineSession ||
      summary.isOnlineTabSelected ||
      summary.isLocalPartySelected
    ) {
      setDetent('full');
      return;
    }
    setDetent((cur) => (cur === 'half' ? 'closed' : 'half'));
  };

  const handleKeystoneClick = () => {
    if (didSwipeRef.current) return;
    setDetent((cur) => (cur === 'full' ? 'closed' : 'full'));
  };

  const handleOpenRoomFromErrorBanner = () => {
    setDismissedMpError(telemetry.mpError ?? '__dismissed__');
    if (!p.onlineDrawerOpen && !summary.isOnlineSession) {
      p.onToggleOnlineDrawer();
    }
    pendingOnlineFocusRef.current = true;
    p.playUiUp('tab');
    setDetent('full');
  };

  const handleOpenGuideFromSheet = () => {
    wasFullOpenRef.current = false;
    setDetent('closed');
    p.onOpenHelp();
  };

  const activeMpError =
    telemetry.mpError && telemetry.mpError !== dismissedMpError ? telemetry.mpError : null;

  const inspectSuppressed = SUPPRESSED_INSPECT_PHASES.has(telemetry.turnPhase);
  const showInspectStrip = Boolean(visibleInspect && !inspectSuppressed && detent !== 'full');
  const inspectSwatchHex = visibleInspect
    ? STAINED_GLASS_PALETTE[visibleInspect.swatchId]?.uiHex ?? '#7ea6ff'
    : '#7ea6ff';

  return (
    <div
      className="mobile-hud-root"
      data-detent={detent}
      aria-hidden={p.isExternalModalOpen ? true : undefined}
    >
      <CathedralTransomCrest
        telemetry={telemetry}
        isOnlineSession={summary.isOnlineSession}
        mpPingMs={telemetry.mpPingMs ?? null}
      />

      <div
        className="plinth-sill-assembly"
        aria-hidden={detent === 'full' ? true : undefined}
      >
        {activeMpError && (
          <div className="online-error-banner plinth-mp-banner" role="alert">
            <span className="plinth-mp-banner-copy">{activeMpError}</span>
            <div className="plinth-mp-banner-actions">
              <button
                type="button"
                className="plinth-banner-btn"
                onClick={handleOpenRoomFromErrorBanner}
              >
                Room
              </button>
              <button
                type="button"
                className="plinth-banner-btn"
                aria-label="Dismiss multiplayer error"
                onClick={() =>
                  setDismissedMpError(telemetry.mpError ?? '__dismissed__')
                }
              >
                ×
              </button>
            </div>
          </div>
        )}

        {showInspectStrip && visibleInspect && (
          <div className="plinth-inspect-strip" role="status" aria-live="polite">
            <div className="plinth-inspect-main">
              <span
                className="plinth-inspect-swatch"
                aria-hidden="true"
                style={{ backgroundColor: inspectSwatchHex }}
              />
              <span className="plinth-inspect-course">
                Course {visibleInspect.assignedLayer + 1}
              </span>
              <span className="plinth-inspect-status">
                {getTouchInspectStatusLabel(visibleInspect.status)}
              </span>
            </div>
            {visibleInspect.damage > 0 && (
              <span className="plinth-inspect-crazed">Crazed</span>
            )}
          </div>
        )}

        <div
          className="plinth-frame"
          onPointerDown={handleSwipePointerDown}
          onPointerMove={handleSwipePointerMove}
          onPointerUp={handleSwipePointerEnd}
          onPointerCancel={handleSwipePointerEnd}
        >
          <div className="plinth-swipe-cap" aria-hidden="true">
            <span className="plinth-arch-apex">
              <LancetArchApexSvg presetId={p.lightPreset} />
            </span>
            <span className="plinth-detent-bar" />
          </div>

          {detent === 'half' && (
            <div
              ref={halfTrayRef}
              className="plinth-half-tray"
              role="region"
              aria-label="Studio Tray"
            >
              <div className="plinth-half-header">
                <button
                  ref={halfCloseBtnRef}
                  type="button"
                  className="quarry-btn quarry-btn-crystal plinth-half-icon-btn"
                  aria-label="Collapse studio tray"
                  onClick={() => {
                    p.playUiUp('default');
                    setDetent('closed');
                  }}
                >
                  <span className="quarry-glass-pane">
                    <span className="quarry-etch-content">
                      <EtchedCloseSvg />
                    </span>
                  </span>
                </button>

                <div className="plinth-half-tabs">
                  <QuarryButton
                    tone="cobalt"
                    active={halfTab === 'studio'}
                    soundTone="tab"
                    onAudioDown={p.playUiDown}
                    onAudioUp={p.playUiUp}
                    className="plinth-half-tab-btn"
                    onClick={() => setHalfTab('studio')}
                  >
                    Studio
                  </QuarryButton>
                  <QuarryButton
                    tone="teal"
                    active={halfTab === 'crystal'}
                    soundTone="tab"
                    onAudioDown={p.playUiDown}
                    onAudioUp={p.playUiUp}
                    className="plinth-half-tab-btn"
                    onClick={() => setHalfTab('crystal')}
                  >
                    Crystal
                  </QuarryButton>
                </div>

                <QuarryButton
                  tone="opal"
                  soundTone="tab"
                  onAudioDown={p.playUiDown}
                  onAudioUp={p.playUiUp}
                  className="plinth-half-icon-btn"
                  onClick={() => setDetent('full')}
                  aria-label="Expand full studio sheet"
                >
                  <GuideQuatrefoilSvg />
                </QuarryButton>
              </div>

              {summary.isOnlineSession && (
                <div className="plinth-half-online-bar">
                  <span className="plinth-half-online-meta">
                    {telemetry.mpRole === 'guest' ? 'P2 Guest' : 'P1 Host'} · Room{' '}
                    {telemetry.mpRoomCode ?? '-----'}
                  </span>
                  <div className="plinth-half-online-actions">
                    <QuarryButton
                      tone="cobalt"
                      soundTone="tab"
                      onAudioDown={p.playUiDown}
                      onAudioUp={p.playUiUp}
                      className="plinth-half-online-btn"
                      onClick={p.onCopyInviteLink}
                    >
                      {p.copiedRoomLink ? 'Copied' : 'Invite'}
                    </QuarryButton>
                    <QuarryButton
                      tone="ruby"
                      soundTone="danger"
                      onAudioDown={p.playUiDown}
                      onAudioUp={p.playUiUp}
                      className="plinth-half-online-btn"
                      onClick={p.onLeaveOnline}
                    >
                      Exit
                    </QuarryButton>
                  </div>
                </div>
              )}

              <div className="sheet-half-body">
                {halfTab === 'studio' ? (
                  <>
                    <div className="half-section-label">Mode &amp; Opponent</div>
                    <div className="half-mode-grid">
                      <QuarryButton
                        tone="cobalt"
                        active={
                          !summary.isOnlineTabSelected &&
                          telemetry.gameMode === 'solo' &&
                          !telemetry.aiOpponentEnabled
                        }
                        soundTone="tab"
                        onAudioDown={p.playUiDown}
                        onAudioUp={p.playUiUp}
                        className="half-mode-btn"
                        onClick={() => p.onSelectMode('solo')}
                      >
                        Solo
                      </QuarryButton>
                      <QuarryButton
                        tone="amber"
                        active={summary.isLocalPartySelected}
                        soundTone="tab"
                        onAudioDown={p.playUiDown}
                        onAudioUp={p.playUiUp}
                        className="half-mode-btn"
                        onClick={() => {
                          p.onSelectMode('party');
                          setDetent('full');
                        }}
                      >
                        Pass &amp; Play
                      </QuarryButton>
                      <QuarryButton
                        tone="amethyst"
                        active={
                          !summary.isOnlineTabSelected && Boolean(telemetry.aiOpponentEnabled)
                        }
                        soundTone="tab"
                        onAudioDown={p.playUiDown}
                        onAudioUp={p.playUiUp}
                        className="half-mode-btn"
                        onClick={p.onToggleVsAi}
                      >
                        Glazier AI
                      </QuarryButton>
                      <QuarryButton
                        tone="ruby"
                        active={!summary.isOnlineTabSelected && telemetry.gameMode === 'sandbox'}
                        soundTone="tab"
                        onAudioDown={p.playUiDown}
                        onAudioUp={p.playUiUp}
                        className="half-mode-btn"
                        onClick={() => p.onSelectMode('sandbox')}
                      >
                        Sandbox
                      </QuarryButton>
                    </div>

                    <div className="half-section-label">Cathedral Window Light</div>
                    <div className="half-light-grid">
                      {LIGHTING_PRESETS.map((preset) => {
                        const gemHex = getLightPresetGemHex(preset.id);
                        return (
                          <QuarryButton
                            key={preset.id}
                            tone="crystal"
                            active={p.lightPreset === preset.id}
                            soundTone="tab"
                            onAudioDown={p.playUiDown}
                            onAudioUp={p.playUiUp}
                            className="half-light-btn"
                            onClick={() => p.onSelectLightPreset(preset.id)}
                          >
                            <span
                              className="inlaid-gem"
                              aria-hidden="true"
                              style={{ backgroundColor: gemHex, boxShadow: `0 0 6px ${gemHex}` }}
                            />
                            <span>{preset.name}</span>
                          </QuarryButton>
                        );
                      })}
                    </div>

                    <div className="half-utility-row">
                      <QuarryButton
                        tone={telemetry.polariscopeMode ? 'teal' : 'crystal'}
                        active={Boolean(telemetry.polariscopeMode)}
                        soundTone="tab"
                        onAudioDown={p.playUiDown}
                        onAudioUp={p.playUiUp}
                        onClick={p.onTogglePolariscope}
                        aria-pressed={Boolean(telemetry.polariscopeMode)}
                      >
                        {telemetry.polariscopeMode ? 'Polariscope On' : 'Polariscope'}
                      </QuarryButton>
                      <QuarryButton
                        tone={p.muted ? 'ruby' : 'crystal'}
                        active={p.muted}
                        soundTone="default"
                        onAudioDown={p.playUiDown}
                        onAudioUp={p.playUiUp}
                        onClick={p.onToggleMute}
                        aria-pressed={p.muted}
                      >
                        <AcousticCrystalSvg muted={p.muted} />
                        <span>{p.muted ? 'Muted' : 'Sound'}</span>
                      </QuarryButton>
                    </div>
                  </>
                ) : (
                  <CrystalTelemetryContent p={p} showPolariscopeButton />
                )}
              </div>
            </div>
          )}

          <div className="plinth-sill-body">
            <ContextualBottomGameplayBar
              p={p}
              detent={detent}
              onModeButtonClick={handleModeButtonClick}
              onKeystoneClick={handleKeystoneClick}
            />
          </div>
        </div>
      </div>

      {detent === 'full' && (
        <div
          className="plinth-full-overlay"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              p.playUiUp('default');
              setDetent('closed');
            }
          }}
        >
          <div
            ref={fullDialogRef}
            className="plinth-full-modal atelier-sheet-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="plinth-full-heading"
          >
            <div className="plinth-full-cartouche">
              <div className="plinth-full-title-group">
                <div className="margin-inlay-cell" aria-hidden="true">
                  <NameplateMarginInlaySvg side="left" />
                </div>
                <div className="plinth-opal-plaque">
                  <h2 id="plinth-full-heading" className="vitrail-wordmark">
                    VITRAIL
                  </h2>
                  <span className="vitrail-spec">Soda-Lime Crystal · Quarter-Sawn Oak</span>
                </div>
              </div>

              <div className="plinth-full-header-actions">
                <QuarryButton
                  tone="crystal"
                  soundTone="tab"
                  onAudioDown={p.playUiDown}
                  onAudioUp={p.playUiUp}
                  className="plinth-full-hdr-btn"
                  onClick={() => setDetent('half')}
                  aria-label="Dock to studio tray"
                >
                  <span>Tray</span>
                </QuarryButton>
                <button
                  ref={fullCloseBtnRef}
                  type="button"
                  className="quarry-btn quarry-btn-crystal plinth-full-hdr-btn"
                  aria-label="Close studio sheet"
                  onClick={() => {
                    p.playUiUp('default');
                    setDetent('closed');
                  }}
                >
                  <span className="quarry-glass-pane">
                    <span className="quarry-etch-content">
                      <EtchedCloseSvg />
                    </span>
                  </span>
                </button>
              </div>
            </div>

            <div className="atelier-sheet-scroll">
              <StudioControlsContent
                p={p}
                onOpenGuideFromSheet={handleOpenGuideFromSheet}
              />
              <section className="atelier-section" aria-label="Crystal Plumb Telemetry">
                <div className="atelier-section-title">Crystal &amp; Plumb</div>
                <CrystalTelemetryContent p={p} showPolariscopeButton={false} />
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

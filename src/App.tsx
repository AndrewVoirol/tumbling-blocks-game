import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AiDifficulty, GameEngine, VitrailHudTelemetry } from './engine/GameEngine';
import {
  INITIAL_LAYERS,
  LIGHTING_PRESETS,
  STAINED_GLASS_PALETTE,
} from './physics/constants';
import { GameMode } from './physics/PhysicsWorld';
import { isValidRoomCode, normalizeRoomCode } from './multiplayer';
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
  'daylight-atelier': 'Atelier',
  'high-clerestory': 'Noon',
  'golden-vespers': 'Vespers',
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
  | 'grisaille-amethyst'
  | 'grisaille-teal';

export type UiSoundTone = 'default' | 'tab' | 'primary' | 'danger' | 'step';

const FOCUSABLE_MODAL_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

const INITIAL_TELEMETRY: VitrailHudTelemetry = {
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
  aiDifficulty: 'master',
};

export interface QuarryButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
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
 * on pointerdown or keyboard Enter/Space and rebounds via an underdamped spring + refractive caustic sweep.
 */
export const QuarryButton = React.forwardRef<HTMLButtonElement, QuarryButtonProps>(
  (
    {
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
      onKeyDown,
      onKeyUp,
      onBlur,
      onClick,
      disabled,
      ...rest
    },
    forwardedRef
  ) => {
    const [pressing, setPressing] = useState(false);
    const [rebounding, setRebounding] = useState(false);
    const reboundTimerRef = useRef<number | null>(null);
    const btnRef = useRef<HTMLButtonElement | null>(null);
    const keyboardPressActiveRef = useRef<boolean>(false);

    const setMergedRef = useCallback(
      (el: HTMLButtonElement | null) => {
        btnRef.current = el;
        if (typeof forwardedRef === 'function') {
          forwardedRef(el);
        } else if (forwardedRef) {
          forwardedRef.current = el;
        }
      },
      [forwardedRef]
    );

    useEffect(() => {
      return () => {
        if (reboundTimerRef.current !== null && typeof window !== 'undefined') {
          window.clearTimeout(reboundTimerRef.current);
        }
      };
    }, []);

    const triggerSpringRebound = useCallback(() => {
      if (typeof window === 'undefined') return;
      if (reboundTimerRef.current !== null) {
        window.clearTimeout(reboundTimerRef.current);
      }
      setRebounding(false);
      const scheduleFrame =
        typeof requestAnimationFrame === 'function'
          ? requestAnimationFrame
          : (cb: FrameRequestCallback) => window.setTimeout(() => cb(Date.now()), 0);
      scheduleFrame(() => {
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
      if ((e.button === 0 || e.pointerType === 'touch') && !disabled) {
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
      keyboardPressActiveRef.current = false;
      onPointerCancel?.(e);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>) => {
      if ((e.key === 'Enter' || e.key === ' ') && !e.repeat && !disabled) {
        keyboardPressActiveRef.current = true;
        setPressing(true);
        setRebounding(false);
        onAudioDown?.(soundTone);
      }
      onKeyDown?.(e);
    };

    const handleKeyUp = (e: React.KeyboardEvent<HTMLButtonElement>) => {
      if ((e.key === 'Enter' || e.key === ' ') && keyboardPressActiveRef.current) {
        keyboardPressActiveRef.current = false;
        setPressing(false);
        triggerSpringRebound();
        onAudioUp?.(soundTone);
      }
      onKeyUp?.(e);
    };

    const handleBlur = (e: React.FocusEvent<HTMLButtonElement>) => {
      if (pressing || keyboardPressActiveRef.current) {
        setPressing(false);
        keyboardPressActiveRef.current = false;
      }
      onBlur?.(e);
    };

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (e.detail === 0 && !disabled && !keyboardPressActiveRef.current) {
        onAudioDown?.(soundTone);
        triggerSpringRebound();
        onAudioUp?.(soundTone);
      }
      onClick?.(e);
    };

    return (
      <button
        ref={setMergedRef}
        type="button"
        disabled={disabled}
        aria-disabled={disabled ? true : undefined}
        data-pressing={pressing ? 'true' : undefined}
        data-rebounding={rebounding ? 'true' : undefined}
        className={`quarry-btn quarry-tone-${tone} ${active ? 'is-lit' : ''} ${className}`.trim()}
        onPointerMove={handlePtrMove}
        onPointerDown={handlePtrDown}
        onPointerUp={handlePtrUp}
        onPointerLeave={handlePtrLeave}
        onPointerCancel={handlePtrCancel}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        onBlur={handleBlur}
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
  }
);
QuarryButton.displayName = 'QuarryButton';

export const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<GameEngine | null>(null);
  const guideBtnRef = useRef<HTMLButtonElement | null>(null);
  const codexWindowRef = useRef<HTMLDivElement | null>(null);
  const codexCloseBtnRef = useRef<HTMLButtonElement | null>(null);
  const wasHelpOpenRef = useRef<boolean>(false);

  const [telemetry, setTelemetry] = useState<VitrailHudTelemetry>(INITIAL_TELEMETRY);
  const [muted, setMuted] = useState<boolean>(false);
  const [lightPreset, setLightPreset] = useState<string>(LIGHTING_PRESETS[0].id);
  const [showHelp, setShowHelp] = useState<boolean>(false);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [onlineDrawerOpen, setOnlineDrawerOpen] = useState<boolean>(() =>
    typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('room')
  );
  const [joinCodeDraft, setJoinCodeDraft] = useState<string>('');
  const [copiedRoomLink, setCopiedRoomLink] = useState<boolean>(false);
  const copyFeedbackTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      const viewportMeta = document.querySelector('meta[name="viewport"]');
      if (viewportMeta) {
        const content = viewportMeta.getAttribute('content') ?? '';
        if (!content.includes('viewport-fit=')) {
          viewportMeta.setAttribute(
            'content',
            content ? `${content}, viewport-fit=cover` : 'width=device-width, initial-scale=1.0, viewport-fit=cover'
          );
        }
      }
    }

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
      if (copyFeedbackTimerRef.current !== null) {
        window.clearTimeout(copyFeedbackTimerRef.current);
      }
      engine.dispose();
      if (engineRef.current === engine) {
        engineRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    engineRef.current?.setModalOpen(showHelp);
    if (!showHelp) {
      if (wasHelpOpenRef.current) {
        wasHelpOpenRef.current = false;
        guideBtnRef.current?.focus();
      }
      return;
    }

    wasHelpOpenRef.current = true;
    codexCloseBtnRef.current?.focus();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        engineRef.current?.audio.playUiKeyUp('default');
        setShowHelp(false);
        return;
      }
      if (e.key === 'Tab') {
        const dialogEl = codexWindowRef.current;
        if (!dialogEl) return;
        const focusables = Array.from(
          dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_MODAL_SELECTOR)
        );
        if (focusables.length === 0) {
          e.preventDefault();
          return;
        }
        const firstEl = focusables[0];
        const lastEl = focusables[focusables.length - 1];
        const activeEl = typeof document !== 'undefined' ? document.activeElement : null;
        if (e.shiftKey) {
          if (activeEl === firstEl || !dialogEl.contains(activeEl)) {
            e.preventDefault();
            lastEl.focus();
          }
        } else if (activeEl === lastEl || !dialogEl.contains(activeEl)) {
          e.preventDefault();
          firstEl.focus();
        }
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
    setOnlineDrawerOpen(false);
    if (telemetry.mpRole) {
      engineRef.current?.leaveOnlineRoom(mode, telemetry.playerCount);
      return;
    }
    engineRef.current?.resetGame(mode, telemetry.playerCount);
  };

  const handleSelectOnlineTab = () => {
    setOnlineDrawerOpen(true);
  };

  const handleHostOnlineRoom = () => {
    setCopiedRoomLink(false);
    engineRef.current?.startOnlineHost();
  };

  const handleJoinOnlineRoom = () => {
    const clean = normalizeRoomCode(joinCodeDraft);
    if (!isValidRoomCode(clean)) return;
    setCopiedRoomLink(false);
    if (engineRef.current?.joinOnlineRoom(clean)) {
      setJoinCodeDraft('');
    }
  };

  const handleLeaveOnlineRoom = () => {
    setCopiedRoomLink(false);
    setOnlineDrawerOpen(false);
    engineRef.current?.leaveOnlineRoom('solo', 2);
  };

  const handleCopyInviteLink = () => {
    const url = engineRef.current?.getOnlineInviteUrl() ?? telemetry.mpRoomCode ?? '';
    if (!url) return;
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(url);
    }
    setCopiedRoomLink(true);
    if (copyFeedbackTimerRef.current !== null) {
      window.clearTimeout(copyFeedbackTimerRef.current);
    }
    copyFeedbackTimerRef.current = window.setTimeout(() => {
      setCopiedRoomLink(false);
      copyFeedbackTimerRef.current = null;
    }, 1800);
  };

  const handleSetPlayerCount = (count: number) => {
    setOnlineDrawerOpen(false);
    engineRef.current?.setAiOpponentEnabled(false);
    if (count !== telemetry.playerCount || telemetry.aiOpponentEnabled || telemetry.mpRole) {
      engineRef.current?.resetGame('party', count);
    }
  };

  const handleToggleVsAi = () => {
    setOnlineDrawerOpen(false);
    const next = !telemetry.aiOpponentEnabled;
    engineRef.current?.setAiOpponentEnabled(next, telemetry.aiDifficulty ?? 'master');
  };

  const handleSelectAiDifficulty = (difficulty: AiDifficulty) => {
    setOnlineDrawerOpen(false);
    if (!telemetry.aiOpponentEnabled) {
      engineRef.current?.setAiOpponentEnabled(true, difficulty);
    } else {
      engineRef.current?.setAiDifficulty(difficulty);
    }
  };

  const handleTogglePolariscope = () => {
    const next = !telemetry.polariscopeMode;
    engineRef.current?.setPolariscopeMode(next);
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
    if (telemetry.mpRole) {
      engineRef.current?.requestOnlineRematch();
      return;
    }
    engineRef.current?.resetGame(telemetry.gameMode, telemetry.playerCount);
  };

  // Plumb compass jewel coordinates
  const comXClamp = Math.max(-15, Math.min(15, telemetry.centerOfMassXZ[0] * 7.2));
  const comZClamp = Math.max(-15, Math.min(15, telemetry.centerOfMassXZ[1] * 7.2));
  const isToppled =
    telemetry.turnPhase === 'COLLAPSED' ||
    (telemetry.currentLayers === 0 && telemetry.shatteredCount > 0);
  const leanSeverity =
    isToppled ||
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

  const activeAiDifficulty: AiDifficulty = telemetry.aiDifficulty ?? 'master';
  const aiDisplayName =
    activeAiDifficulty === 'apprentice' ? 'Apprentice AI' : 'Master Glazier';

  const partyTurnLabel = (() => {
    const vsAi = Boolean(telemetry.aiOpponentEnabled);
    const nameOf = (idx: number) =>
      vsAi ? (idx === 0 ? 'You' : aiDisplayName) : `Player ${idx + 1}`;
    const shortOf = (idx: number) =>
      vsAi
        ? idx === 0
          ? 'You'
          : activeAiDifficulty === 'apprentice'
          ? 'Apprentice'
          : 'Master'
        : `P${idx + 1}`;
    if (telemetry.turnPhase !== 'COLLAPSED') {
      if (vsAi) {
        return telemetry.currentPlayer === 0
          ? 'Your Turn'
          : telemetry.aiActionLabel ||
            (telemetry.aiThinking
              ? `${aiDisplayName} Reading Stack…`
              : `${aiDisplayName}'s Turn`);
      }
      return `Player ${telemetry.currentPlayer + 1}'s Turn`;
    }
    if (telemetry.blocksStacked === 0) {
      return `${nameOf(telemetry.currentPlayer)} Toppled`;
    }
    const winnerIdx =
      (telemetry.currentPlayer + telemetry.playerCount - 1) % telemetry.playerCount;
    return `${shortOf(winnerIdx)} Wins · ${shortOf(telemetry.currentPlayer)} Toppled`;
  })();

  const partyAccentIndex =
    telemetry.turnPhase === 'COLLAPSED' && telemetry.blocksStacked > 0
      ? (telemetry.currentPlayer + telemetry.playerCount - 1) % telemetry.playerCount
      : telemetry.currentPlayer;

  const activeTensionSegments = Math.max(1, Math.round(telemetry.grabResistance01 * 12));
  const activeSettleSegments = Math.round(telemetry.settleProgress01 * 10);

  const isOnlineSession = Boolean(telemetry.mpRole);
  const isOnlineTabSelected = isOnlineSession || onlineDrawerOpen;
  const localOnlinePlayerIdx = telemetry.mpRole === 'guest' ? 1 : 0;
  const isLocalOnlineTurn =
    !isOnlineSession ||
    (Boolean(telemetry.mpPeerConnected || telemetry.mpRole === 'host') &&
      telemetry.currentPlayer === localOnlinePlayerIdx);

  const onlineStatusLabel = (() => {
    if (!isOnlineSession) return 'Host or Join Room';
    if (telemetry.mpConnectionState === 'error') return 'Room Error';
    if (telemetry.mpConnectionState === 'reconnecting') return 'Reconnecting…';
    if (telemetry.mpConnectionState === 'connecting') return 'Joining Room…';
    if (telemetry.mpConnectionState === 'waiting_peer' && !telemetry.mpPeerConnected) {
      return 'Waiting for P2…';
    }
    if (telemetry.turnPhase === 'COLLAPSED') {
      if (telemetry.blocksStacked === 0) {
        return telemetry.currentPlayer === localOnlinePlayerIdx
          ? 'You Toppled'
          : 'Opponent Toppled';
      }
      const winnerIdx = (telemetry.currentPlayer + 1) % 2;
      return winnerIdx === localOnlinePlayerIdx ? 'You Win!' : 'Opponent Wins';
    }
    return telemetry.currentPlayer === localOnlinePlayerIdx ? 'Your Turn' : "Opponent's Turn";
  })();

  const onlineTransportTag =
    telemetry.mpTransport === 'webrtc-p2p'
      ? 'P2P'
      : telemetry.mpTransport === 'ws-relay'
      ? 'Relay'
      : null;

  const onlinePingLabel =
    telemetry.mpPingMs !== null && telemetry.mpPingMs !== undefined
      ? `${telemetry.mpPingMs} ms${onlineTransportTag ? ` · ${onlineTransportTag}` : ''}`
      : onlineTransportTag ?? '';

  const isLocalSoloSelected = !isOnlineTabSelected && telemetry.gameMode === 'solo';
  const isLocalPartySelected = !isOnlineTabSelected && telemetry.gameMode === 'party';
  const isLocalSandboxSelected = !isOnlineTabSelected && telemetry.gameMode === 'sandbox';

  const isBench =
    typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('bench');
  const isModalOverlayOpen = showHelp || Boolean(gpuError);

  return (
    <div className="app-shell" data-lighting={lightPreset}>
      <VitrailSvgDefs />
      <canvas
        ref={canvasRef}
        className="webgpu-canvas"
        aria-label="Vitrail 3D Stained-Glass Tower Canvas"
        aria-hidden={isModalOverlayOpen ? true : undefined}
      />

      {gpuError && (
        <div className="codex-backdrop">
          <div
            className="codex-window"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="vitrail-gpu-error-heading"
          >
            <div className="codex-pane-inner">
              <h2 id="vitrail-gpu-error-heading" className="etched-heading">
                WebGPU Required
              </h2>
              <p className="etched-prose">
                {gpuError} Open Vitrail in a WebGPU-enabled browser (Chrome 113+, Edge 113+, or
                Safari 18+).
              </p>
            </div>
          </div>
        </div>
      )}

      <div
        className="hud-layer"
        aria-hidden={isModalOverlayOpen ? true : undefined}
        style={{ display: isBench ? 'none' : undefined }}
      >
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

              {/* 4 Horizontal Inlaid Stained-Glass Mode Courses */}
              <div className="lancet-mode-stack" role="tablist" aria-label="Game Mode">
                <QuarryButton
                  role="tab"
                  aria-selected={isLocalSoloSelected}
                  tone={isLocalSoloSelected ? 'cobalt' : 'grisaille-cobalt'}
                  active={isLocalSoloSelected}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={() => handleModeChange('solo')}
                >
                  <ModeMosaicMedallionSvg
                    mode="solo"
                    active={isLocalSoloSelected}
                  />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Solo</span>
                    <span className="mode-course-meta">Classic Stack</span>
                  </span>
                  <ModeRightJewelSvg mode="solo" active={isLocalSoloSelected} />
                </QuarryButton>

                <QuarryButton
                  role="tab"
                  aria-selected={isLocalPartySelected}
                  tone={isLocalPartySelected ? 'emerald' : 'grisaille-emerald'}
                  active={isLocalPartySelected}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={() => handleModeChange('party')}
                >
                  <ModeMosaicMedallionSvg
                    mode="party"
                    active={isLocalPartySelected}
                  />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Pass &amp; Play</span>
                    <span className="mode-course-meta">2–4 Players</span>
                  </span>
                  <ModeRightJewelSvg mode="party" active={isLocalPartySelected} />
                </QuarryButton>

                {/* Pass & Play Inlaid Sub-Course */}
                {isLocalPartySelected && (
                  <div className="lancet-course party-subcourse">
                    <div className="party-turn-pane">
                      <span
                        className="player-cabochon"
                        aria-hidden="true"
                        style={{
                          backgroundColor: PLAYER_COLORS[partyAccentIndex % PLAYER_COLORS.length],
                          boxShadow: `0 0 8px ${PLAYER_COLORS[partyAccentIndex % PLAYER_COLORS.length]}`,
                        }}
                      />
                      <span className="party-turn-copy">{partyTurnLabel}</span>
                    </div>
                    <div className="party-count-strip" role="group" aria-label="Player Count">
                      <QuarryButton
                        tone={telemetry.aiOpponentEnabled ? 'cobalt' : 'crystal'}
                        active={Boolean(telemetry.aiOpponentEnabled)}
                        aria-pressed={Boolean(telemetry.aiOpponentEnabled)}
                        aria-label="Play against Glazier AI"
                        soundTone="step"
                        onAudioDown={playUiDown}
                        onAudioUp={playUiUp}
                        className="player-num-pane vs-ai-pane"
                        onClick={handleToggleVsAi}
                        title="Play vs. Counterfactual Glazier AI"
                      >
                        <span>AI</span>
                      </QuarryButton>
                      {[2, 3, 4].map((cnt) => {
                        const isCountActive =
                          !telemetry.aiOpponentEnabled && telemetry.playerCount === cnt;
                        return (
                          <QuarryButton
                            key={cnt}
                            tone={isCountActive ? 'amber' : 'crystal'}
                            active={isCountActive}
                            aria-pressed={isCountActive}
                            aria-label={`${cnt} Players`}
                            soundTone="step"
                            onAudioDown={playUiDown}
                            onAudioUp={playUiUp}
                            className="player-num-pane"
                            onClick={() => handleSetPlayerCount(cnt)}
                            title={`${cnt} Players`}
                          >
                            <span>{cnt}P</span>
                          </QuarryButton>
                        );
                      })}
                    </div>
                    {Boolean(telemetry.aiOpponentEnabled) && (
                      <div
                        className="ai-difficulty-strip"
                        role="radiogroup"
                        aria-label="AI Difficulty"
                      >
                        <QuarryButton
                          role="radio"
                          aria-checked={activeAiDifficulty === 'apprentice'}
                          tone={activeAiDifficulty === 'apprentice' ? 'teal' : 'crystal'}
                          active={activeAiDifficulty === 'apprentice'}
                          soundTone="step"
                          onAudioDown={playUiDown}
                          onAudioUp={playUiUp}
                          className="ai-difficulty-pane"
                          onClick={() => handleSelectAiDifficulty('apprentice')}
                          title="Apprentice AI: Conservative extractions & center-counterbalance stacks"
                        >
                          <span>Apprentice</span>
                        </QuarryButton>
                        <QuarryButton
                          role="radio"
                          aria-checked={activeAiDifficulty === 'master'}
                          tone={activeAiDifficulty === 'master' ? 'cobalt' : 'crystal'}
                          active={activeAiDifficulty === 'master'}
                          soundTone="step"
                          onAudioDown={playUiDown}
                          onAudioUp={playUiUp}
                          className="ai-difficulty-pane"
                          onClick={() => handleSelectAiDifficulty('master')}
                          title="Master Glazier (2-Ply): Continuous clearance-azimuth pulls & adversarial eccentric stacks"
                        >
                          <span>Master Glazier</span>
                        </QuarryButton>
                      </div>
                    )}
                  </div>
                )}

                <QuarryButton
                  role="tab"
                  aria-selected={isOnlineTabSelected}
                  tone={isOnlineTabSelected ? 'teal' : 'grisaille-teal'}
                  active={isOnlineTabSelected}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={handleSelectOnlineTab}
                >
                  <ModeMosaicMedallionSvg mode="online" active={isOnlineTabSelected} />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Online</span>
                    <span className="mode-course-meta">
                      {isOnlineSession && telemetry.mpRoomCode
                        ? `Room ${telemetry.mpRoomCode}`
                        : 'Internet 2P'}
                    </span>
                  </span>
                  <ModeRightJewelSvg mode="online" active={isOnlineTabSelected} />
                </QuarryButton>

                {/* Online 2P Internet Room Sub-Course */}
                {isOnlineTabSelected && (
                  <div className="lancet-course online-subcourse">
                    {!isOnlineSession ? (
                      <div className="online-lobby-row">
                        <QuarryButton
                          tone="teal"
                          active
                          soundTone="primary"
                          onAudioDown={playUiDown}
                          onAudioUp={playUiUp}
                          className="online-host-btn"
                          onClick={handleHostOnlineRoom}
                          aria-label="Host Online Room"
                          title="Create a 5-character Internet Room Code"
                        >
                          <span>Host</span>
                        </QuarryButton>
                        <input
                          type="text"
                          value={joinCodeDraft}
                          onChange={(e) => setJoinCodeDraft(normalizeRoomCode(e.target.value))}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              handleJoinOnlineRoom();
                            }
                          }}
                          placeholder="CODE…"
                          maxLength={8}
                          autoComplete="off"
                          autoCapitalize="characters"
                          spellCheck={false}
                          aria-label="Enter 5-character Room Code"
                          className="online-code-input"
                        />
                        <QuarryButton
                          tone="amber"
                          active={isValidRoomCode(joinCodeDraft)}
                          disabled={!isValidRoomCode(joinCodeDraft)}
                          soundTone="primary"
                          onAudioDown={playUiDown}
                          onAudioUp={playUiUp}
                          className="online-join-btn"
                          onClick={handleJoinOnlineRoom}
                          aria-label="Join Online Room"
                          title="Join Room Code"
                        >
                          <span>Join</span>
                        </QuarryButton>
                      </div>
                    ) : (
                      <>
                        <div className="online-room-bar">
                          <div className="online-code-plaque" title="Active Room Code">
                            <span className="online-code-caption">
                              {telemetry.mpRole === 'host' ? 'P1 Host' : 'P2 Guest'}
                            </span>
                            <span className="online-code-value">{telemetry.mpRoomCode}</span>
                          </div>
                          <QuarryButton
                            tone={copiedRoomLink ? 'emerald' : 'amber'}
                            active
                            soundTone="step"
                            onAudioDown={playUiDown}
                            onAudioUp={playUiUp}
                            className="online-action-btn"
                            onClick={handleCopyInviteLink}
                            aria-label={copiedRoomLink ? 'Room Invite Link Copied' : 'Copy Room Invite Link'}
                            title="Copy shareable ?room= invite link"
                          >
                            <span>{copiedRoomLink ? 'Copied' : 'Invite'}</span>
                          </QuarryButton>
                          <QuarryButton
                            tone="crystal"
                            soundTone="default"
                            onAudioDown={playUiDown}
                            onAudioUp={playUiUp}
                            className="online-action-btn online-leave-btn"
                            onClick={handleLeaveOnlineRoom}
                            aria-label="Leave Online Room"
                            title="Leave Online Room"
                          >
                            <span>Exit</span>
                          </QuarryButton>
                        </div>

                        <div className="online-status-row">
                          <div className="online-status-pane">
                            <span className="online-status-left">
                              <span
                                className="player-cabochon"
                                aria-hidden="true"
                                style={{
                                  backgroundColor: telemetry.mpPeerConnected
                                    ? PLAYER_COLORS[telemetry.currentPlayer % 2]
                                    : '#fbbf24',
                                  boxShadow: `0 0 8px ${
                                    telemetry.mpPeerConnected
                                      ? PLAYER_COLORS[telemetry.currentPlayer % 2]
                                      : '#fbbf24'
                                  }`,
                                }}
                              />
                              <span className="party-turn-copy">{onlineStatusLabel}</span>
                            </span>
                            {onlinePingLabel && (
                              <span className="online-ping-pill">{onlinePingLabel}</span>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                    {telemetry.mpError && (
                      <div className="online-error-banner" role="alert">
                        {telemetry.mpError}
                      </div>
                    )}
                  </div>
                )}

                <QuarryButton
                  role="tab"
                  aria-selected={isLocalSandboxSelected}
                  tone={isLocalSandboxSelected ? 'amethyst' : 'grisaille-amethyst'}
                  active={isLocalSandboxSelected}
                  soundTone="tab"
                  onAudioDown={playUiDown}
                  onAudioUp={playUiUp}
                  className="mode-course-btn"
                  onClick={() => handleModeChange('sandbox')}
                >
                  <ModeMosaicMedallionSvg
                    mode="sandbox"
                    active={isLocalSandboxSelected}
                  />
                  <span className="mode-course-copy">
                    <span className="mode-course-name">Sandbox</span>
                    <span className="mode-course-meta">Free Physics</span>
                  </span>
                  <ModeRightJewelSvg mode="sandbox" active={isLocalSandboxSelected} />
                </QuarryButton>

                {/* Sandbox Fracture Sub-Course */}
                {isLocalSandboxSelected && (
                  <div className="lancet-course sandbox-subcourse">
                    <QuarryButton
                      tone="ruby"
                      active
                      soundTone="danger"
                      onAudioDown={playUiDown}
                      onAudioUp={playUiUp}
                      className="smash-course-btn"
                      onClick={() => engineRef.current?.triggerCollapseDemo()}
                      aria-label="Buckle and Shatter Tower"
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
                  aria-pressed={!muted}
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
                  aria-label="Rebuild 54-Prism Stained-Glass Tower"
                  title="Rebuild 54-Prism Stained-Glass Tower"
                >
                  <RebuildTowerSvg />
                  <span className="sill-pane-label">Rebuild</span>
                </QuarryButton>

                <QuarryButton
                  ref={guideBtnRef}
                  tone="crystal"
                  active={showHelp}
                  aria-haspopup="dialog"
                  aria-expanded={showHelp}
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
                  {isToppled ? '—' : `${telemetry.leanAngleDeg.toFixed(1)}°`}
                </span>
                <span className="etched-plumb-status">
                  {isToppled
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

            {/* Cross-Polarized Photoelastic Force-Chain Polariscope Toggle */}
            <div className="polariscope-lens-row">
              <QuarryButton
                tone={telemetry.polariscopeMode ? 'teal' : 'crystal'}
                active={Boolean(telemetry.polariscopeMode)}
                soundTone="tab"
                onAudioDown={playUiDown}
                onAudioUp={playUiUp}
                className="polariscope-lens-btn"
                onClick={handleTogglePolariscope}
                aria-pressed={Boolean(telemetry.polariscopeMode)}
                aria-label="Toggle Polariscope Stress-Optic Lens"
                title="Toggle Cross-Polarized Photoelastic Force-Chain Lens"
              >
                <span
                  className={`inlaid-gem ${
                    telemetry.polariscopeMode ? 'gem-teal' : 'gem-opal'
                  }`}
                  aria-hidden="true"
                />
                <span>Polariscope</span>
                <span className="polariscope-state-tag">
                  {telemetry.polariscopeMode ? 'Isochromatic' : 'Natural'}
                </span>
              </QuarryButton>
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
                  {isOnlineSession
                    ? `${onlineStatusLabel} · ${
                        telemetry.shatteredCount > 0 ? 'Tower Shattered' : 'Tower Toppled'
                      }`
                    : telemetry.shatteredCount > 0
                    ? 'Tower Shattered'
                    : 'Tower Toppled'}
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
                  <span>
                    {isOnlineSession
                      ? 'Rematch Stained-Glass Tower'
                      : 'Rebuild Stained-Glass Tower'}
                  </span>
                </QuarryButton>
              </div>
            </div>
          ) : telemetry.turnPhase === 'EXTRACTED' ? (
            <div
              className={`cut-glass-prism-bar ${
                isLocalOnlineTurn ? 'is-interactive-prism' : ''
              }`}
            >
              <PrismEndInlaySvg tone="emerald" />
              <div className="prism-bevel-chamber">
                <div className="prism-phase-tag tag-emerald">
                  {isOnlineSession && !isLocalOnlineTurn ? 'Opponent' : 'Stack'}
                </div>
                <div className="prism-copy-group">
                  <span className="prism-primary-line">
                    {isOnlineSession && !isLocalOnlineTurn
                      ? 'Opponent positioning extracted prism across top course…'
                      : 'Drag or tap top course to position · Tap or Place Prism to release'}
                  </span>
                  <span className="prism-secondary-line">
                    {isOnlineSession && !isLocalOnlineTurn
                      ? '1-finger drag orbits 360° · 2-finger pinch/pan or scroll zooms'
                      : 'Q / 90° R / E angles · 0 aligns · Drag outside to orbit · 2-finger pinch/pan'}
                  </span>
                </div>

                {isLocalOnlineTurn && (
                  <div className="prism-inlaid-keys" role="toolbar" aria-label="Prism Placement Controls">
                    <QuarryButton
                      tone="crystal"
                      soundTone="step"
                      onAudioDown={playUiDown}
                      onAudioUp={playUiUp}
                      className="prism-key-btn"
                      onClick={() => engineRef.current?.handleRotateCarried(-Math.PI / 24)}
                      aria-label="Rotate prism counter-clockwise 7.5 degrees (Q)"
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
                      aria-label="Rotate prism 90 degrees (R)"
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
                      aria-label="Rotate prism clockwise 7.5 degrees (E)"
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
                        aria-label="Align prism square with course (0)"
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
                      aria-label="Place Prism onto top course"
                      title="Release prism onto top course (Space / Enter)"
                    >
                      <PlacementGlyphSvg kind="place" />
                      <span>Place Prism</span>
                    </QuarryButton>
                  </div>
                )}
              </div>
              <PrismEndInlaySvg tone="amber" />
            </div>
          ) : telemetry.turnPhase === 'PULLING' ? (
            <div className="cut-glass-prism-bar">
              <PrismEndInlaySvg tone="cobalt" />
              <div className="prism-bevel-chamber">
                <div className="prism-phase-tag tag-amber">
                  {isOnlineSession && !isLocalOnlineTurn ? 'Opponent' : 'Drawing'}
                </div>

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
                    {isOnlineSession && !isLocalOnlineTurn
                      ? `Opponent Drawing · ${resistanceLabel}`
                      : resistanceLabel}
                  </span>
                  <span className="prism-secondary-line">
                    {isOnlineSession && !isLocalOnlineTurn
                      ? 'Drag anywhere to orbit 360° & watch live spring tension'
                      : telemetry.gameMode === 'sandbox'
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
                  {isOnlineSession
                    ? isLocalOnlineTurn
                      ? 'Your Turn'
                      : 'Spectate'
                    : telemetry.gameMode === 'sandbox'
                    ? 'Sandbox'
                    : 'Extract'}
                </div>

                <div className="prism-copy-group">
                  <span className="prism-primary-line">
                    {isOnlineSession && !telemetry.mpPeerConnected
                      ? `Share Room Code ${telemetry.mpRoomCode ?? ''} or Invite Link to connect Player 2`
                      : isOnlineSession && !isLocalOnlineTurn
                      ? 'Opponent scouting stained-glass stack · Drag anywhere to orbit 360°'
                      : telemetry.gameMode === 'sandbox'
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
                      `1-finger background orbits · 2-finger pinch/pan · ${lockedRowsLabel}`
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
            ref={codexWindowRef}
            className="codex-window"
            role="dialog"
            aria-modal="true"
            aria-labelledby="vitrail-guide-heading"
            aria-describedby="vitrail-guide-subtitle"
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
                  <span id="vitrail-guide-subtitle" className="etched-subtitle">
                    Acoustic Tap-Testing · Spring Extraction · Voronoi Glass Fracture
                  </span>
                </div>
              </div>
              <QuarryButton
                ref={codexCloseBtnRef}
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
                    bear the column load. <strong>Tap or click</strong> any prism below the highest
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
                    Drag or tap across the top course to position the extracted prism (prisms abut
                    flush bevel-to-bevel). Angle with <strong>Q / 90° R / E</strong> or the{' '}
                    <strong>wheel</strong>, then tap or press <strong>Place Prism</strong> to
                    release gently. Hard drops craze internal spiderwebs or shatter into 3D Voronoi
                    shards.
                  </p>
                </div>
              </div>
            </div>

            <div className="codex-footer-course">
              <div className="codex-hint-quarry">
                <span>1-finger background orbits 360° · 2-finger pinch zooms &amp; pans</span>
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

import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

export type MujocoThreadingMode = 'mt' | 'single';

interface LoadedMujocoRuntime {
  mj: MainModule;
  mode: MujocoThreadingMode;
  threadCount: number;
}

let runtimePromise: Promise<LoadedMujocoRuntime> | null = null;
let loadedRuntime: LoadedMujocoRuntime | null = null;

function canUseSharedArrayBuffer(): boolean {
  if (typeof SharedArrayBuffer === 'undefined') return false;
  // In Node.js (Vitest / CLI calibration scripts), SharedArrayBuffer is always enabled without window.crossOriginIsolated.
  const maybeProcess = (globalThis as { process?: { versions?: { node?: string } } }).process;
  const isNode = typeof maybeProcess?.versions?.node === 'string';
  if (isNode) return true;
  if (typeof globalThis !== 'undefined' && 'crossOriginIsolated' in globalThis) {
    return Boolean((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated);
  }
  return false;
}

function getOptimalThreadCount(): number {
  const hw =
    typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
      ? navigator.hardwareConcurrency
      : 6;
  if (hw <= 2) return 1;
  // 4 WASM pthreads hits the sweet spot on Apple Silicon iPad Pro / Mac without pthread barrier contention
  return Math.max(2, Math.min(4, hw - 2));
}

/**
 * Initializes Google DeepMind's `@mujoco/mujoco@4.0.0` WASM engine.
 * Automatically loads the multi-threaded `/mt` pthread build when `SharedArrayBuffer` /
 * cross-origin isolation is active (browser + iPad Pro + Node), with seamless fallback
 * to the single-threaded build if headers are absent.
 */
export async function ensureMujocoInitialized(
  preferMultiThreaded: boolean = true
): Promise<LoadedMujocoRuntime> {
  if (loadedRuntime) return loadedRuntime;
  if (runtimePromise) return runtimePromise;

  runtimePromise = (async () => {
    const threadCount = getOptimalThreadCount();
    const useMt = preferMultiThreaded && canUseSharedArrayBuffer() && threadCount > 1;
    if (useMt) {
      try {
        const mod = await import('@mujoco/mujoco/mt');
        const mj = await mod.default();
        loadedRuntime = { mj, mode: 'mt', threadCount };
        return loadedRuntime;
      } catch {
        // Fall through cleanly to single-threaded WASM if pthread spawn is restricted
      }
    }

    const mod = await import('@mujoco/mujoco');
    const mj = await mod.default();
    loadedRuntime = { mj, mode: 'single', threadCount: 1 };
    return loadedRuntime;
  })();

  return runtimePromise;
}

export function getMujocoRuntimeSync(): LoadedMujocoRuntime {
  if (!loadedRuntime) {
    throw new Error(
      'MuJoCo WASM runtime is not initialized yet. Call await ensurePhysicsInitialized() before constructing PhysicsWorld.'
    );
  }
  return loadedRuntime;
}

/**
 * Attaches MuJoCo 4.0's C++ island threadpool (`mju_threadpool`) when running the `/mt` build.
 */
export function attachMujocoThreadPoolIfAvailable(data: MjData): number {
  if (!loadedRuntime || loadedRuntime.mode !== 'mt') return 1;
  const mjAny = loadedRuntime.mj as unknown as {
    mju_threadpool?: (d: MjData, nthreads: number) => void;
  };
  if (typeof mjAny.mju_threadpool === 'function' && loadedRuntime.threadCount > 1) {
    mjAny.mju_threadpool(data, loadedRuntime.threadCount);
    return loadedRuntime.threadCount;
  }
  return 1;
}

/**
 * Safe Embind handle disposer: deletes top-level heap objects (`MjData`, `MjModel`) without double-freeing sub-views.
 */
export function disposeMujocoHandles(data: MjData | null, model: MjModel | null): void {
  if (data) {
    try {
      data.delete();
    } catch {
      // Ignore already-deleted handle
    }
  }
  if (model) {
    try {
      model.delete();
    } catch {
      // Ignore already-deleted handle
    }
  }
}

import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config.ts';

// Each physics suite loads `@mujoco/mujoco/mt` (`4` WASM pthreads via `mju_threadpool`).
// Capping Vitest concurrency to `maxWorkers: 2` (`<= 8` active WASM pthreads) prevents
// 24-thread CPU oversubscription from contaminating wall-clock `stepMsP95 < 16.0 ms` (`C4`) gates.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      maxWorkers: 2,
    },
  })
);

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@mujoco/mujoco'],
  },
  worker: {
    format: 'es',
  },
  server: {
    port: 5173,
    strictPort: false,
    headers: crossOriginIsolationHeaders,
  },
  preview: {
    port: 4173,
    headers: crossOriginIsolationHeaders,
  },
  build: {
    target: 'es2022',
  },
});

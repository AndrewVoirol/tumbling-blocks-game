import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { attachVitrailRoomRelay } from './workers/vitrail-rooms/nodeRelay.mjs';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

function vitrailMultiplayerRelayPlugin(): Plugin {
  return {
    name: 'vitrail-multiplayer-relay',
    configureServer(server) {
      if (process.env.VITEST) return;
      if (server.httpServer) {
        attachVitrailRoomRelay(server.httpServer, '/__vitrail_mp');
      }
    },
    configurePreviewServer(server) {
      if (process.env.VITEST) return;
      if (server.httpServer) {
        attachVitrailRoomRelay(server.httpServer, '/__vitrail_mp');
      }
    },
  };
}

export { attachVitrailRoomRelay };

export default defineConfig({
  plugins: [react(), vitrailMultiplayerRelayPlugin()],
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

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  envDir: '..', // read VITE_* variables from the repo-root .env
  server: {
    port: 5173,
    strictPort: true, // the backend's default CORS_ORIGINS expects exactly 5173
    proxy: {
      // 127.0.0.1, not "localhost": on Windows "localhost" can resolve to IPv6 only.
      '/api': { target: 'http://127.0.0.1:8000' },
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{js,jsx}'],
    setupFiles: ['./src/test/setup.js'],
  },
});
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: proxy /api to the FastAPI backend so the app fetches same-origin (no CORS).
// Prod: `vite build` emits ./dist, which the backend mounts at /.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});

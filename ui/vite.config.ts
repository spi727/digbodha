import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api is served by ui/server.py (python ui/server.py) when you want live runs.
// Without it, the UI falls back to the bundled KITTI 0007 demo in public/demo.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
});

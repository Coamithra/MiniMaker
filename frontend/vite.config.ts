import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/generate-miniature": {
        target: "http://localhost:8000",
        timeout: 600000,
      },
      "/generate-images": {
        target: "http://localhost:8000",
        timeout: 600000,
      },
      "/generate-back-view": {
        target: "http://localhost:8000",
        timeout: 300000,
      },
      "/generate-model": {
        target: "http://localhost:8000",
        timeout: 600000,
      },
      "/generate-3d": {
        target: "http://localhost:8000",
        timeout: 300000,
      },
      "/generate": {
        target: "http://localhost:8000",
        timeout: 120000,
      },
    },
  },
});

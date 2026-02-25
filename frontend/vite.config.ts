import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
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

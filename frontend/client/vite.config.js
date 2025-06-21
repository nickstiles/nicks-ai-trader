import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      // proxy /api/* to your Node proxy on 3001
      "/api": {
        target: "http://frontend_node:3001",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
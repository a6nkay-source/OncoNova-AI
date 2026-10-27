import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The API runs on 8010 by default; override with ONCONOVA_API_URL.
const api = process.env.ONCONOVA_API_URL ?? "http://127.0.0.1:8010";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: api, changeOrigin: false } } },
  preview: { proxy: { "/api": { target: api, changeOrigin: false } } },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});

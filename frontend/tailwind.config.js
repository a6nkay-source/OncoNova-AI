/** @type {import('tailwindcss').Config} */
export default {
  content: { relative: true, files: ["./index.html", "./src/**/*.{ts,tsx}"] },
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#070b13",
          925: "#0a101b",
          900: "#0d1421",
          850: "#111a2a",
          800: "#162033",
          750: "#1b2740",
          700: "#22304a",
          600: "#2e3f5e",
          500: "#4a5d80",
          400: "#7488aa",
          300: "#9fb0cc",
          200: "#c9d4e6",
          100: "#e4ebf5",
        },
        accent: { DEFAULT: "#4c9bff", soft: "#2a5fa8", glow: "#7fb8ff" },
        teal: { DEFAULT: "#2fd0b8", soft: "#1b7f73" },
        ev: {
          measured: "#2fd0b8",
          model: "#a48bff",
          fromModel: "#62a8ff",
          uncertain: "#f4b740",
          review: "#ff7a8a",
          simulated: "#e58cff",
          unavailable: "#8b9ab4",
        },
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};

import { fileURLToPath } from "node:url";

// Resolve the Tailwind config relative to this file so the dev server works from any working directory.
export default {
  plugins: {
    tailwindcss: { config: fileURLToPath(new URL("./tailwind.config.js", import.meta.url)) },
    autoprefixer: {},
  },
};

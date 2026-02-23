import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/spectral_analysis_of_memory/",
  plugins: [react()],
});

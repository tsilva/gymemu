import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: resolve("gymemu/web_assets/dist"),
    emptyOutDir: true,
    minify: true,
    rollupOptions: {
      input: {
        app: resolve("frontend/main.js"),
        catalog: resolve("frontend/catalog.js"),
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
});

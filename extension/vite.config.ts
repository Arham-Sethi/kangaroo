import { defineConfig } from "vite";
import { resolve } from "path";

/**
 * Vite config for the Kangaroo Shift browser extension.
 *
 * IMPORTANT: MV3 content scripts are loaded as CLASSIC scripts (not modules).
 * They cannot use ES `import` syntax. However, the background service worker
 * and popup CAN use ES modules.
 *
 * Solution: Build everything as IIFE format. This inlines all shared code
 * into each entry point, making every file 100% self-contained. The ~2KB
 * observer duplication across 3 content scripts is negligible.
 *
 * For a production build, we'd use two separate Vite passes (IIFE for
 * content scripts, ESM for background/popup), but for this codebase size
 * IIFE everywhere works perfectly and avoids build complexity.
 *
 * Bundles produced:
 *   - background.js       (service worker, IIFE-wrapped)
 *   - content/chatgpt.js  (self-contained IIFE)
 *   - content/claude.js   (self-contained IIFE)
 *   - content/gemini.js   (self-contained IIFE)
 *   - popup.js            (self-contained IIFE)
 */
export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        background: resolve(__dirname, "src/background/service-worker.ts"),
        "content/chatgpt": resolve(__dirname, "src/content/chatgpt.ts"),
        "content/claude": resolve(__dirname, "src/content/claude.ts"),
        "content/gemini": resolve(__dirname, "src/content/gemini.ts"),
        popup: resolve(__dirname, "src/popup/popup.ts"),
      },
      output: {
        entryFileNames: "[name].js",
        // IIFE format wraps everything in a self-executing function.
        // No imports, no exports, fully self-contained.
        format: "iife",
      },
    },
    target: "chrome110",
    minify: false,
    sourcemap: true,
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
});

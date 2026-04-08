/**
 * Multi-pass build script for the Kangaroo Shift browser extension.
 *
 * MV3 content scripts are loaded as classic scripts — they cannot use
 * ES module `import` syntax. Background service workers and popup scripts
 * CAN use ES modules.
 *
 * This script runs two Vite builds:
 *   Pass 1: Content scripts → IIFE format (self-contained, no imports)
 *   Pass 2: Background + popup → ES module format (can use imports)
 *
 * Both passes output to the same dist/ directory. Pass 1 clears it;
 * Pass 2 appends without clearing.
 */

import { build } from "vite";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const sharedConfig = {
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  logLevel: "info",
};

async function main() {
  const startTime = Date.now();

  // ── Pass 1: Content scripts (IIFE, self-contained) ──────────────────────
  console.log("\n📦 Pass 1/2: Building content scripts (IIFE)...\n");

  for (const script of ["chatgpt", "claude", "gemini"]) {
    await build({
      ...sharedConfig,
      build: {
        outDir: "dist",
        emptyOutDir: script === "chatgpt", // Only first pass clears dist/
        rollupOptions: {
          input: resolve(__dirname, `src/content/${script}.ts`),
          output: {
            entryFileNames: `content/${script}.js`,
            format: "iife",
          },
        },
        target: "chrome110",
        minify: false,
        sourcemap: true,
      },
    });
  }

  // ── Pass 2: Background + popup (IIFE, self-contained) ────────────────────
  console.log("\n📦 Pass 2/2: Building background + popup (IIFE)...\n");

  for (const entry of ["background", "popup"]) {
    const input =
      entry === "background"
        ? resolve(__dirname, "src/background/service-worker.ts")
        : resolve(__dirname, "src/popup/popup.ts");

    await build({
      ...sharedConfig,
      build: {
        outDir: "dist",
        emptyOutDir: false,
        rollupOptions: {
          input,
          output: {
            entryFileNames: `${entry}.js`,
            format: "iife",
          },
        },
        target: "chrome110",
        minify: false,
        sourcemap: true,
      },
    });
  }

  const elapsed = Date.now() - startTime;
  console.log(`\n✅ Extension built in ${elapsed}ms\n`);
}

main().catch((err) => {
  console.error("Build failed:", err);
  process.exit(1);
});

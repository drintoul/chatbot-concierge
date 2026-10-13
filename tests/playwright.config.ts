import { defineConfig } from "@playwright/test";

/**
 * Runs inside the `e2e` compose service (mcr playwright image).
 *   docker compose --profile test run --rm e2e
 * Works on the host too if playwright is installed there.
 */
export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.ts",
  timeout: 60_000,
  retries: 0,
  reporter: [["list"]],
});

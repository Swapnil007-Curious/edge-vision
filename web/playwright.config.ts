import { defineConfig, devices } from "@playwright/test";

// Tests run against `vite preview` of the production build; BASE_PATH must match the build.
const base = process.env.BASE_PATH ?? "/";
const port = 4173;
const origin = `http://localhost:${port}`;

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 180_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: `${origin}${base}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run preview -- --port ${port}`,
    url: `${origin}${base}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});

import { test, expect } from "@playwright/test";

test("all routes render and the live backend connects", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Beyond accuracy/ })).toBeVisible();
  await expect(page.getByText("Available checkpoints")).toBeVisible();
  await page.screenshot({ path: "../artifacts/reports/dashboard-desktop.png", fullPage: true });
  for (const [path, title] of [["/predict", "A closer look at a leaf"], ["/experiments", "An audit trail for every run"], ["/model-comparison", "Compare performance. Consider the cost."], ["/calibration", "When should a model answer?"], ["/methodology", "A protocol built around reliability"]]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: title, exact: true })).toBeVisible();
    await expect(page.locator("main").getByRole("alert")).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test("missing results show the runnable empty state", async ({ page }) => {
  await page.route("**/api/results/comparison", route => route.fulfill({ json: { results: [], datasets: [], warnings: [], next_command: "python -m research.scripts.run_experiments" } }));
  await page.goto("/model-comparison");
  await expect(page.getByText("Your next experiment starts here")).toBeVisible();
  await expect(page.getByText(/python -m research.scripts.run_experiments/)).toBeVisible();
});

test("offline backend shows an actionable error", async ({ page }) => {
  await page.route("**/api/results/comparison", route => route.abort());
  await page.goto("/model-comparison");
  await expect(page.locator("main").getByRole("alert")).toContainText("Cannot reach the research backend");
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});

test("mobile navigation and layout fit the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Beyond accuracy/ })).toBeVisible();
  await expect(page.getByText("Available checkpoints")).toBeVisible();
  await expect(page.getByRole("status")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: "../artifacts/reports/dashboard-mobile.png", fullPage: true });
  await page.getByRole("link", { name: "Image prediction", exact: true }).click();
  await expect(page.getByRole("heading", { name: "A closer look at a leaf" })).toBeVisible();
});

test("upload reaches checkpoint inference and displays uncertainty", async ({ page }) => {
  await page.goto("/predict");
  await expect(page.locator("#model")).toBeEnabled();
  await page.locator('input[type="file"]').setInputFiles({ name: "test.png", mimeType: "image/png", buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=", "base64") });
  await page.getByRole("button", { name: "Analyze image" }).click();
  await expect(page.getByRole("heading", { name: "Top class probabilities" })).toBeVisible({ timeout: 60000 });
  await expect(page.getByText("Calibrated confidence", { exact: true })).toBeVisible();
  await page.screenshot({ path: "../artifacts/reports/prediction.png", fullPage: true });
});

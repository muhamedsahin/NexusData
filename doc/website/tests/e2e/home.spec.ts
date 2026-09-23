import { expect, test } from "@playwright/test";

test.describe("homepage phase 1-2", () => {
  test("renders NexusData heading after skip", async ({ page }) => {
    await page.addInitScript(() => {
      sessionStorage.setItem("nexusdata-loader-seen", "1");
    });
    await page.goto("/en");
    await expect(
      page.getByRole("heading", { name: /MatrixData|NexusData/i, level: 1 }),
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      page.locator("#scene-hero").getByText(/CMakeLists\.txt|FetchContent/i),
    ).toBeVisible();
  });

  test("skip button dismisses loader", async ({ page }) => {
    await page.addInitScript(() => {
      sessionStorage.removeItem("nexusdata-loader-seen");
    });
    await page.goto("/en");
    const skip = page.getByRole("button", { name: /Skip|Atla/i });
    await expect(skip).toBeVisible();
    await skip.click();
    await expect(
      page.getByRole("heading", { name: /MatrixData|NexusData/i, level: 1 }),
    ).toBeVisible({ timeout: 15_000 });
  });

  test("story sections are present after skip", async ({ page }) => {
    await page.addInitScript(() => {
      sessionStorage.setItem("nexusdata-loader-seen", "1");
    });
    await page.goto("/en");
    await expect(
      page.getByRole("heading", { name: /MatrixData|NexusData/i, level: 1 }),
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      page.getByRole("heading", { name: "Read every format" }),
    ).toBeVisible();
    await page.locator("#scene-shuffle").scrollIntoViewIfNeeded();
    await expect(
      page.getByRole("heading", { name: /Dataset, shuffle/i }),
    ).toBeVisible();
    await page.locator("#scene-batch").scrollIntoViewIfNeeded();
    await page.locator("#scene-batch input[type='range']").fill("6");
    await expect(
      page.locator("#scene-batch").getByText(/\d+ batches/),
    ).toBeVisible();
  });

  test("docs route does not load three.js", async ({ page }) => {
    const threeRequests: string[] = [];
    page.on("request", (req) => {
      if (req.url().includes("three") || req.url().includes("@react-three")) {
        threeRequests.push(req.url());
      }
    });
    await page.goto("/en/docs");
    await expect(
      page.getByRole("heading", { name: /Documentation|Dokümantasyon/i }),
    ).toBeVisible();
    expect(threeRequests.length).toBe(0);
  });
});

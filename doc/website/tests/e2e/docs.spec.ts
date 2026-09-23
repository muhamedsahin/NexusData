import { expect, test } from "@playwright/test";

test.describe("docs engine", () => {
  test("renders introduction page with sidebar", async ({ page }) => {
    await page.goto("/en/docs/getting-started/introduction");
    await expect(
      page.getByRole("heading", { name: "Introduction", level: 1 }),
    ).toBeVisible();
    await expect(page.getByRole("navigation", { name: /sidebar/i })).toBeVisible();
    await expect(page.getByText(/standalone/i).first()).toBeVisible();
  });

  test("turkish docs page is reachable", async ({ page }) => {
    await page.goto("/tr/docs/getting-started/introduction");
    await expect(page.getByRole("heading", { name: "Giriş", level: 1 })).toBeVisible();
  });

  test("search API returns hits", async ({ request }) => {
    const res = await request.get("/api/docs-search?locale=en&q=NDArray");
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.results.length).toBeGreaterThan(0);
  });

  test("docs route still does not load three.js", async ({ page }) => {
    const threeRequests: string[] = [];
    page.on("request", (req) => {
      if (req.url().includes("three") || req.url().includes("@react-three")) {
        threeRequests.push(req.url());
      }
    });
    await page.goto("/en/docs");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    expect(threeRequests.length).toBe(0);
  });

  test("benchmarks page shows illustrative suite without three.js", async ({
    page,
  }) => {
    const threeRequests: string[] = [];
    page.on("request", (req) => {
      if (req.url().includes("three") || req.url().includes("@react-three")) {
        threeRequests.push(req.url());
      }
    });
    await page.goto("/en/docs/performance/benchmarks");
    await expect(
      page.getByRole("heading", { name: "Benchmarks", level: 1 }),
    ).toBeVisible();
    await expect(page.getByText(/Ryzen 7 5800H/i).first()).toBeVisible();
    expect(threeRequests.length).toBe(0);
  });

  test("dataloader simulator is interactive on docs", async ({ page }) => {
    await page.goto("/en/docs/concepts/dataloader");
    await expect(
      page.getByRole("heading", { name: "DataLoader simulator" }),
    ).toBeVisible();
    await page.getByLabel("Batch size").fill("3");
    await expect(page.getByText(/\[0\]/)).toBeVisible();
  });
});

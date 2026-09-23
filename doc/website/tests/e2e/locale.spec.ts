import { expect, test } from "@playwright/test";

test.describe("locale negotiation", () => {
  test("TR country header redirects / to /tr", async ({ request }) => {
    const response = await request.get("/", {
      maxRedirects: 0,
      headers: {
        "x-vercel-ip-country": "TR",
      },
    });
    expect(response.status()).toBe(307);
    expect(response.headers().location).toMatch(/\/tr\/?$/);
  });

  test("DE country header redirects / to /en", async ({ request }) => {
    const response = await request.get("/", {
      maxRedirects: 0,
      headers: {
        "x-vercel-ip-country": "DE",
      },
    });
    expect(response.status()).toBe(307);
    expect(response.headers().location).toMatch(/\/en\/?$/);
  });

  test("cookie en wins over TR country header", async ({ request }) => {
    const response = await request.get("/", {
      maxRedirects: 0,
      headers: {
        cookie: "NEXT_LOCALE=en",
        "x-vercel-ip-country": "TR",
      },
    });
    expect(response.status()).toBe(307);
    expect(response.headers().location).toMatch(/\/en\/?$/);
  });

  test("no country + Accept-Language tr redirects to /tr", async ({
    request,
  }) => {
    const response = await request.get("/", {
      maxRedirects: 0,
      headers: {
        "accept-language": "tr-TR,tr;q=0.9,en;q=0.8",
      },
    });
    expect(response.status()).toBe(307);
    expect(response.headers().location).toMatch(/\/tr\/?$/);
  });

  test("locale switcher writes NEXT_LOCALE cookie", async ({ page }) => {
    await page.addInitScript(() => {
      sessionStorage.setItem("nexusdata-loader-seen", "1");
    });
    await page.goto("/en");
    await expect(
      page.getByRole("heading", { name: /NexusData/i, level: 1 }),
    ).toBeVisible({ timeout: 15_000 });
    await page
      .getByRole("button", { name: /Switch to Türkçe|Türkçe diline/i })
      .click();
    await expect(page).toHaveURL(/\/tr(\/|$)/, { timeout: 15_000 });
    const cookies = await page.context().cookies();
    const localeCookie = cookies.find((c) => c.name === "NEXT_LOCALE");
    expect(localeCookie?.value).toBe("tr");
  });

  test("direct locale URLs are reachable", async ({ page }) => {
    await page.addInitScript(() => {
      sessionStorage.setItem("nexusdata-loader-seen", "1");
    });
    await page.goto("/en");
    await expect(
      page.getByRole("heading", { name: /NexusData/i, level: 1 }),
    ).toBeVisible({ timeout: 15_000 });
    await page.goto("/tr");
    await expect(
      page.getByRole("heading", { name: /NexusData/i, level: 1 }),
    ).toBeVisible({ timeout: 15_000 });
  });

  test("Vary header is present on locale redirect", async ({ request }) => {
    const response = await request.get("/", {
      maxRedirects: 0,
      headers: { "x-vercel-ip-country": "TR" },
    });
    const vary = response.headers().vary ?? "";
    expect(vary.toLowerCase()).toContain("accept-language");
    expect(vary.toLowerCase()).toContain("cookie");
  });
});

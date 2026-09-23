import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.describe("accessibility", () => {
  test("docs introduction has no critical/serious axe violations", async ({
    page,
  }) => {
    await page.goto("/en/docs/getting-started/introduction");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();

    const blocking = results.violations.filter((v) =>
      ["critical", "serious"].includes(v.impact ?? ""),
    );
    expect(
      blocking,
      blocking.map((v) => `${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });

  test("skip link is keyboard-reachable on docs", async ({ page }) => {
    await page.goto("/en/docs");
    await page.keyboard.press("Tab");
    const skip = page.locator("a.skip-link");
    await expect(skip).toBeFocused();
  });
});

import { chromium } from "@playwright/test";

(async () => {
  const browser = await chromium.launch({
    args: [
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
      "--enable-webgl",
      "--enable-webgl2",
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    colorScheme: "dark",
    extraHTTPHeaders: {
      cookie: "NEXT_THEME=dark; NEXT_LOCALE=en",
    },
  });
  const page = await context.newPage();

  page.on("console", (msg) => console.log("[BROWSER LOG]", msg.text()));
  page.on("pageerror", (err) => console.log("[BROWSER ERROR]", err.message));

  await page.goto("http://127.0.0.1:3001/en");
  await page.waitForTimeout(1500);

  console.log("Locating Skip button...");
  const skip = page.getByRole("button", { name: /Skip|Atla/i });
  console.log("Is skip visible?", await skip.isVisible());
  await skip.click();
  console.log("Clicked skip!");

  await page.waitForTimeout(1500);
  await page.screenshot({ path: "shots/test_after_skip.png" });
  await browser.close();
})();


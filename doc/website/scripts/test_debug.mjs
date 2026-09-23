import { chromium } from "@playwright/test";

(async () => {
  const browser = await chromium.launch({
    headless: true,
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
  });
  const page = await context.newPage();

  page.on("console", (msg) => console.log("[CONSOLE]", msg.type(), msg.text()));
  page.on("pageerror", (err) => console.log("[PAGE ERROR]", err));

  await page.goto("http://127.0.0.1:3001/en");
  await page.waitForTimeout(2000);

  const state = await page.evaluate(() => {
    return {
      buttons: Array.from(document.querySelectorAll("button")).map((b) => ({
        text: b.textContent,
        classes: b.className,
      })),
      sessionStorage: { ...sessionStorage },
      hasCanvas: !!document.querySelector("canvas"),
      bodyHtml: document.body.innerHTML.slice(0, 500),
    };
  });

  console.log("Page state:", JSON.stringify(state, null, 2));
  await browser.close();
})();


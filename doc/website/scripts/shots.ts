import { chromium } from "@playwright/test";
import * as fs from "node:fs";
import * as path from "node:path";

interface ShotOptions {
  round: string;
  baseUrl: string;
  waitMs: number;
}

const DEFAULT_OPTIONS: ShotOptions = {
  round: "round-0",
  baseUrl: process.env.PLAYWRIGHT_BASE_URL || "http://localhost:3001",
  waitMs: 1500,
};

function parseArgs(): ShotOptions {
  const options = { ...DEFAULT_OPTIONS };
  for (const arg of process.argv.slice(2)) {
    if (arg.startsWith("--round=")) {
      options.round = arg.split("=")[1];
    } else if (arg.startsWith("--url=")) {
      options.baseUrl = arg.split("=")[1];
    } else if (arg.startsWith("--wait=")) {
      options.waitMs = parseInt(arg.split("=")[1], 10) || DEFAULT_OPTIONS.waitMs;
    }
  }
  return options;
}

const PROGRESS_POINTS = [
  0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.0,
];

async function capture() {
  const { round, baseUrl, waitMs } = parseArgs();
  const outputDir = path.resolve(process.cwd(), "shots", round);
  fs.mkdirSync(outputDir, { recursive: true });

  console.log(`[shots] Starting visual capture for ${round}`);
  console.log(`[shots] Output directory: ${outputDir}`);
  console.log(`[shots] Target URL: ${baseUrl}`);

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

  try {
    // 1. Desktop Viewport (1440x900)
    console.log(`[shots] Capturing Desktop (1440x900)...`);
    const desktopContext = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
      colorScheme: "dark",
      extraHTTPHeaders: {
        cookie: "NEXT_THEME=dark; NEXT_LOCALE=en",
      },
    });

    // Capture loader active on desktop
    const loaderPage = await desktopContext.newPage();
    await loaderPage.goto(`${baseUrl}/en`, { waitUntil: "domcontentloaded" });
    await loaderPage.waitForTimeout(600);
    const loaderPath = path.join(outputDir, "desktop_loader.png");
    await loaderPage.screenshot({ path: loaderPath });
    console.log(`[shots] Saved: ${path.basename(loaderPath)}`);
    await loaderPage.close();

    // Standard story scenes with loader skipped
    const desktopPage = await desktopContext.newPage();
    await desktopPage.addInitScript(() => {
      sessionStorage.setItem("nexusdata-loader-seen", "1");
    });
    await desktopPage.goto(`${baseUrl}/en`);

    // Ensure loader is dismissed and hero heading is visible
    try {
      const skipBtn = desktopPage.getByRole("button", { name: /Skip|Atla/i });
      if (await skipBtn.isVisible({ timeout: 1500 })) {
        await skipBtn.click();
      }
    } catch {
      // already skipped or not visible
    }

    await desktopPage
      .getByRole("heading", { level: 1 })
      .waitFor({ state: "visible", timeout: 20_000 });
    await desktopPage.waitForTimeout(1000);

    for (const p of PROGRESS_POINTS) {
      const pLabel = Math.round(p * 100)
        .toString()
        .padStart(3, "0");
      const filename = `desktop_p${pLabel}.png`;
      const filePath = path.join(outputDir, filename);

      await desktopPage.evaluate((progress) => {
        const win = window as any;
        if (typeof win.__setScrollProgress === "function") {
          win.__setScrollProgress(progress);
        } else {
          const max = Math.max(
            1,
            document.documentElement.scrollHeight - window.innerHeight,
          );
          window.scrollTo(0, progress * max);
        }
      }, p);

      await desktopPage.waitForTimeout(waitMs);
      await desktopPage.screenshot({ path: filePath });
      console.log(`[shots] Saved: ${filename} (p=${p.toFixed(2)})`);
    }
    await desktopPage.close();
    await desktopContext.close();

    // 2. Mobile Viewport (390x844)
    console.log(`[shots] Capturing Mobile (390x844)...`);
    const mobileContext = await browser.newContext({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 1,
      isMobile: true,
      hasTouch: true,
      colorScheme: "dark",
      extraHTTPHeaders: {
        cookie: "NEXT_THEME=dark; NEXT_LOCALE=en",
      },
    });

    const mobilePage = await mobileContext.newPage();
    await mobilePage.addInitScript(() => {
      sessionStorage.setItem("nexusdata-loader-seen", "1");
    });
    await mobilePage.goto(`${baseUrl}/en`);

    try {
      const skipBtn = mobilePage.getByRole("button", { name: /Skip|Atla/i });
      if (await skipBtn.isVisible({ timeout: 1500 })) {
        await skipBtn.click();
      }
    } catch {
      // already skipped or not visible
    }

    await mobilePage
      .getByRole("heading", { level: 1 })
      .waitFor({ state: "visible", timeout: 20_000 });
    await mobilePage.waitForTimeout(1000);

    for (const p of PROGRESS_POINTS) {
      const pLabel = Math.round(p * 100)
        .toString()
        .padStart(3, "0");
      const filename = `mobile_p${pLabel}.png`;
      const filePath = path.join(outputDir, filename);

      await mobilePage.evaluate((progress) => {
        const win = window as any;
        if (typeof win.__setScrollProgress === "function") {
          win.__setScrollProgress(progress);
        } else {
          const max = Math.max(
            1,
            document.documentElement.scrollHeight - window.innerHeight,
          );
          window.scrollTo(0, progress * max);
        }
      }, p);

      await mobilePage.waitForTimeout(waitMs);
      await mobilePage.screenshot({ path: filePath });
      console.log(`[shots] Saved: ${filename} (p=${p.toFixed(2)})`);
    }
    await mobilePage.close();
    await mobileContext.close();

    console.log(`[shots] All screenshots captured successfully in ${outputDir}`);
  } finally {
    await browser.close();
  }
}

capture().catch((err) => {
  console.error("[shots] Error capturing screenshots:", err);
  process.exit(1);
});

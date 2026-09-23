import type { NextRequest } from "next/server";
import { siteConfig, type SiteLocale } from "@/config/site.config";

const BOT_UA =
  /bot|crawl|spider|slurp|bingpreview|facebookexternalhit|embedly|quora|pinterest|redditbot|applebot|duckduckbot|yandex|baidu|semrush|ahrefs|mj12bot|gptbot|claudebot|bytespider|amazonbot/i;

export function isSearchBot(userAgent: string | null): boolean {
  if (!userAgent) return false;
  return BOT_UA.test(userAgent);
}

export function getCountryFromHeaders(
  request: NextRequest,
  options: { allowDevOverride?: boolean } = {},
): string | null {
  const allowDev =
    options.allowDevOverride ?? process.env.NODE_ENV === "development";

  if (allowDev) {
    const forced =
      request.nextUrl.searchParams.get("__country") ??
      process.env.DEV_FORCE_COUNTRY;
    if (forced && /^[A-Za-z]{2}$/.test(forced)) {
      return forced.toUpperCase();
    }
  }

  const headers = [
    "x-vercel-ip-country",
    "cf-ipcountry",
    "x-country-code",
  ] as const;

  for (const name of headers) {
    const value = request.headers.get(name);
    if (value && value !== "XX" && value !== "T1" && /^[A-Za-z]{2}$/.test(value)) {
      return value.toUpperCase();
    }
  }

  return null;
}

export function localeFromAcceptLanguage(
  header: string | null,
): SiteLocale {
  if (!header) return siteConfig.defaultLocale;

  const candidates = header
    .split(",")
    .map((part) => {
      const [tag, ...params] = part.trim().split(";");
      const qParam = params.find((p) => p.trim().startsWith("q="));
      const q = qParam ? Number.parseFloat(qParam.trim().slice(2)) : 1;
      return { tag: tag.toLowerCase(), q: Number.isFinite(q) ? q : 0 };
    })
    .sort((a, b) => b.q - a.q);

  for (const { tag } of candidates) {
    if (tag === "tr" || tag.startsWith("tr-")) return "tr";
    if (tag === "en" || tag.startsWith("en-")) return "en";
  }

  return siteConfig.defaultLocale;
}

export function localeFromCookie(request: NextRequest): SiteLocale | null {
  const value = request.cookies.get(siteConfig.localeCookie)?.value;
  if (value === "en" || value === "tr") return value;
  return null;
}

/**
 * Decision order for paths without an explicit locale prefix:
 * 1. NEXT_LOCALE cookie (user choice always wins over country)
 * 2. Country header TR → tr, any other country → en (skipped for bots)
 * 3. Accept-Language
 * 4. Default en
 */
export function resolveLocale(request: NextRequest): SiteLocale {
  const cookieLocale = localeFromCookie(request);
  if (cookieLocale) return cookieLocale;

  const ua = request.headers.get("user-agent");
  const isBot = isSearchBot(ua);

  if (!isBot) {
    const country = getCountryFromHeaders(request);
    if (country) {
      return country === "TR" ? "tr" : "en";
    }
  }

  return localeFromAcceptLanguage(request.headers.get("accept-language"));
}

export function stripLocalePrefix(pathname: string): string {
  const match = pathname.match(/^\/(en|tr)(?=\/|$)/);
  if (!match) return pathname === "/" ? "/" : pathname;
  const rest = pathname.slice(match[0].length);
  return rest.length === 0 ? "/" : rest;
}

export function hasLocalePrefix(pathname: string): boolean {
  return /^\/(en|tr)(?=\/|$)/.test(pathname);
}

import createMiddleware from "next-intl/middleware";
import { type NextRequest, NextResponse } from "next/server";
import { routing } from "./i18n/routing";
import { hasLocalePrefix, resolveLocale } from "./lib/geo";

const intlMiddleware = createMiddleware(routing);

function withVary(response: NextResponse): NextResponse {
  const existing = response.headers.get("Vary");
  const required = [
    "Cookie",
    "Accept-Language",
    "X-Vercel-IP-Country",
    "CF-IPCountry",
    "X-Country-Code",
    "User-Agent",
  ];
  const merged = new Set(
    (existing ? existing.split(",") : [])
      .map((v) => v.trim())
      .filter(Boolean)
      .concat(required),
  );
  response.headers.set("Vary", Array.from(merged).join(", "));
  return response;
}

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (!hasLocalePrefix(pathname)) {
    const locale = resolveLocale(request);
    const url = request.nextUrl.clone();
    const suffix = pathname === "/" ? "" : pathname;
    url.pathname = `/${locale}${suffix}`;
    return withVary(NextResponse.redirect(url, 307));
  }

  return withVary(intlMiddleware(request));
}

export const config = {
  matcher: ["/((?!api|trpc|_next|_vercel|.*\\..*).*)"],
};

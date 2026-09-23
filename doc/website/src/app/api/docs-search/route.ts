import { NextResponse } from "next/server";
import { searchDocs } from "@/lib/docs/search";
import type { SiteLocale } from "@/config/site.config";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q") ?? "";
  const localeParam = searchParams.get("locale") ?? "en";
  const locale: SiteLocale = localeParam === "tr" ? "tr" : "en";
  const results = await searchDocs(locale, q, 8);
  return NextResponse.json({ results });
}

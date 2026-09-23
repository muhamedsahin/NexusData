import { describe, expect, it } from "vitest";
import {
  hasLocalePrefix,
  isSearchBot,
  localeFromAcceptLanguage,
  stripLocalePrefix,
} from "@/lib/geo";

describe("localeFromAcceptLanguage", () => {
  it("prefers Turkish when q is highest", () => {
    expect(localeFromAcceptLanguage("tr-TR,tr;q=0.9,en;q=0.8")).toBe("tr");
  });

  it("falls back to English", () => {
    expect(localeFromAcceptLanguage("de-DE,de;q=0.9,en;q=0.8")).toBe("en");
  });

  it("defaults when header missing", () => {
    expect(localeFromAcceptLanguage(null)).toBe("en");
  });
});

describe("path helpers", () => {
  it("detects locale prefixes", () => {
    expect(hasLocalePrefix("/en")).toBe(true);
    expect(hasLocalePrefix("/tr/docs")).toBe(true);
    expect(hasLocalePrefix("/docs")).toBe(false);
  });

  it("strips locale prefixes", () => {
    expect(stripLocalePrefix("/en")).toBe("/");
    expect(stripLocalePrefix("/tr/docs")).toBe("/docs");
    expect(stripLocalePrefix("/docs")).toBe("/docs");
  });
});

describe("isSearchBot", () => {
  it("detects common crawlers", () => {
    expect(isSearchBot("Mozilla/5.0 (compatible; Googlebot/2.1)")).toBe(true);
    expect(isSearchBot("Mozilla/5.0 (Windows NT 10.0) Chrome/120")).toBe(false);
  });
});

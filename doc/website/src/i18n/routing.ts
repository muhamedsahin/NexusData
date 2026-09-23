import { defineRouting } from "next-intl/routing";
import { siteConfig } from "@/config/site.config";

export const routing = defineRouting({
  locales: [...siteConfig.locales],
  defaultLocale: siteConfig.defaultLocale,
  localePrefix: "always",
  localeDetection: false,
});

export type AppLocale = (typeof routing.locales)[number];

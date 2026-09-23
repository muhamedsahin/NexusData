import type { MetadataRoute } from "next";
import { siteConfig } from "@/config/site.config";
import { getDocSlugs } from "@/lib/docs/source";

export default function sitemap(): MetadataRoute.Sitemap {
  const base = siteConfig.url.replace(/\/$/, "");
  const entries: MetadataRoute.Sitemap = [];

  for (const locale of siteConfig.locales) {
    entries.push({
      url: `${base}/${locale}`,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 1,
    });
    entries.push({
      url: `${base}/${locale}/privacy`,
      lastModified: new Date(),
      changeFrequency: "yearly",
      priority: 0.3,
    });
    entries.push({
      url: `${base}/${locale}/docs`,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 0.9,
    });
    for (const slug of getDocSlugs(locale)) {
      if (slug.length === 0) continue;
      entries.push({
        url: `${base}/${locale}/docs/${slug.join("/")}`,
        lastModified: new Date(),
        changeFrequency: "weekly",
        priority: 0.8,
      });
    }
  }

  return entries;
}

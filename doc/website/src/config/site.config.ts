/**
 * Site-wide configuration. Placeholder fields stay empty until real data exists.
 * Empty placeholders hide their UI sections — never invent people, URLs, or licenses.
 */

export type FeatureStatus = "stable" | "beta" | "in-development" | "planned";

export const PLACEHOLDER = {
  githubUrl: "PLACEHOLDER_GITHUB_URL",
  authorName: "PLACEHOLDER_AUTHOR_NAME",
  authorEmail: "PLACEHOLDER_AUTHOR_EMAIL",
  license: "PLACEHOLDER_LICENSE",
  twitterUrl: "PLACEHOLDER_TWITTER_URL",
  discordUrl: "PLACEHOLDER_DISCORD_URL",
} as const;

export function isPlaceholder(value: string | undefined | null): boolean {
  if (!value) return true;
  return value.startsWith("PLACEHOLDER_") || value.trim() === "";
}

export function resolvePublicLink(value: string | undefined | null): string | null {
  if (!value || isPlaceholder(value)) return null;
  return value;
}

export const siteConfig = {
  name: "MatrixData",
  shortName: "MatrixData",
  tagline: {
    en: "The pipeline that turns raw data into tensors",
    tr: "Ham veriyi tensöre taşıyan boru hattı",
  },
  description: {
    en: "High-performance C++20 data loading library for AI — Source → Dataset → Sampler → DataLoader → Batch → NDArray.",
    tr: "AI için yüksek performanslı C++20 veri yükleme kütüphanesi — Source → Dataset → Sampler → DataLoader → Batch → NDArray.",
  },
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3001",
  defaultLocale: "en" as const,
  locales: ["en", "tr"] as const,
  localeCookie: "NEXT_LOCALE",
  themeCookie: "NEXT_THEME",
  version: "0.1.0",
  libraryVersion: "1.0.0",
  status: "in-development" as FeatureStatus,
  license: "Apache-2.0",
  githubUrl: PLACEHOLDER.githubUrl,
  author: {
    name: PLACEHOLDER.authorName,
    email: PLACEHOLDER.authorEmail,
  },
  social: {
    twitter: PLACEHOLDER.twitterUrl,
    discord: PLACEHOLDER.discordUrl,
  },
  docsEditBase: null as string | null,
} as const;

export type SiteLocale = (typeof siteConfig.locales)[number];

export const ecosystemNodes = [
  {
    id: "matrixflash-pro",
    name: "MatrixFlash Pro",
    role: { en: "Computation", tr: "Hesaplama" },
    status: "planned" as FeatureStatus,
  },
  {
    id: "matrixdata",
    name: "MatrixData",
    role: { en: "Data I/O", tr: "Veri I/O" },
    status: "in-development" as FeatureStatus,
  },
  {
    id: "nexusloss",
    name: "NexusLoss",
    role: { en: "Loss functions", tr: "Kayıp fonksiyonları" },
    status: "planned" as FeatureStatus,
  },
  {
    id: "nexusmodel",
    name: "NexusModel",
    role: { en: "Model layers", tr: "Model katmanları" },
    status: "planned" as FeatureStatus,
  },
  {
    id: "nexusoptim",
    name: "NexusOptim",
    role: { en: "Optimizers", tr: "Optimizer'lar" },
    status: "planned" as FeatureStatus,
  },
  {
    id: "nexustrain",
    name: "NexusTrain",
    role: { en: "Training loop", tr: "Eğitim döngüsü" },
    status: "planned" as FeatureStatus,
  },
  {
    id: "ai-engine",
    name: "AI Engine",
    role: { en: "Orchestration", tr: "Orkestrasyon" },
    status: "planned" as FeatureStatus,
  },
] as const;

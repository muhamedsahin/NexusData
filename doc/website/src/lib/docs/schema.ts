import { z } from "zod";

export const featureStatusSchema = z.enum([
  "stable",
  "beta",
  "in-development",
  "planned",
]);

export const docsFrontmatterSchema = z.object({
  title: z.string().min(1),
  description: z.string().min(1),
  status: featureStatusSchema.default("in-development"),
  since: z.string().optional(),
  order: z.number().int().nonnegative().default(100),
  locale: z.enum(["en", "tr"]),
  section: z.string().min(1),
  draft: z.boolean().optional().default(false),
});

export type DocsFrontmatter = z.infer<typeof docsFrontmatterSchema>;

export const DOCS_VERSIONS = ["latest", "v0.1"] as const;
export type DocsVersion = (typeof DOCS_VERSIONS)[number];
export const DEFAULT_DOCS_VERSION: DocsVersion = "latest";

/** Section order for sidebar grouping. */
export const DOCS_SECTION_ORDER = [
  "getting-started",
  "concepts",
  "sources",
  "preparation",
  "performance",
  "integration",
  "guides",
  "examples",
  "reference",
  "about",
] as const;

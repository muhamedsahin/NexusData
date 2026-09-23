import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import readingTime from "reading-time";
import {
  DOCS_SECTION_ORDER,
  docsFrontmatterSchema,
  type DocsFrontmatter,
} from "@/lib/docs/schema";
import type { SiteLocale } from "@/config/site.config";

const CONTENT_ROOT = path.join(process.cwd(), "content", "docs");

export type DocsDoc = {
  slug: string[];
  urlPath: string;
  locale: SiteLocale;
  frontmatter: DocsFrontmatter;
  content: string;
  readingMinutes: number;
  lastModified: string;
};

export type DocsNavItem = {
  title: string;
  href: string;
  status: DocsFrontmatter["status"];
  order: number;
  slug: string[];
};

export type DocsNavGroup = {
  section: string;
  items: DocsNavItem[];
};

function assertLocale(locale: string): asserts locale is SiteLocale {
  if (locale !== "en" && locale !== "tr") {
    throw new Error(`Unsupported docs locale: ${locale}`);
  }
}

function walkMdxFiles(dir: string, base: string[] = []): string[][] {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const out: string[][] = [];
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walkMdxFiles(full, [...base, entry.name]));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".mdx")) {
      const name = entry.name.replace(/\.mdx$/, "");
      if (name === "index") out.push(base);
      else out.push([...base, name]);
    }
  }
  return out;
}

function filePathForSlug(locale: SiteLocale, slug: string[]): string {
  if (slug.length === 0) {
    return path.join(CONTENT_ROOT, locale, "index.mdx");
  }
  const asFile = path.join(CONTENT_ROOT, locale, ...slug) + ".mdx";
  if (fs.existsSync(asFile)) return asFile;
  const asIndex = path.join(CONTENT_ROOT, locale, ...slug, "index.mdx");
  return asIndex;
}

export function getDocSlugs(locale: SiteLocale): string[][] {
  assertLocale(locale);
  const root = path.join(CONTENT_ROOT, locale);
  return walkMdxFiles(root).sort((a, b) => a.join("/").localeCompare(b.join("/")));
}

export function getDocBySlug(
  locale: SiteLocale,
  slug: string[] = [],
): DocsDoc | null {
  assertLocale(locale);
  const filePath = filePathForSlug(locale, slug);
  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, "utf8");
  const { data, content } = matter(raw);
  const parsed = docsFrontmatterSchema.safeParse({ ...data, locale });
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("; ");
    throw new Error(
      `Invalid frontmatter in ${path.relative(process.cwd(), filePath)}: ${issues}`,
    );
  }
  if (parsed.data.draft) return null;

  const stats = fs.statSync(filePath);
  const urlPath =
    slug.length === 0 ? `/${locale}/docs` : `/${locale}/docs/${slug.join("/")}`;

  return {
    slug,
    urlPath,
    locale,
    frontmatter: parsed.data,
    content,
    readingMinutes: Math.max(1, Math.ceil(readingTime(content).minutes)),
    lastModified: stats.mtime.toISOString(),
  };
}

export function getAllDocs(locale: SiteLocale): DocsDoc[] {
  return getDocSlugs(locale)
    .map((slug) => getDocBySlug(locale, slug))
    .filter((d): d is DocsDoc => d !== null)
    .sort((a, b) => a.frontmatter.order - b.frontmatter.order);
}

export function getDocsNavigation(locale: SiteLocale): DocsNavGroup[] {
  const docs = getAllDocs(locale);
  const groups = new Map<string, DocsNavItem[]>();

  for (const doc of docs) {
    const section = doc.frontmatter.section;
    const items = groups.get(section) ?? [];
    items.push({
      title: doc.frontmatter.title,
      href: doc.urlPath.replace(`/${locale}`, "") || "/docs",
      status: doc.frontmatter.status,
      order: doc.frontmatter.order,
      slug: doc.slug,
    });
    groups.set(section, items);
  }

  const orderedSections = [
    ...DOCS_SECTION_ORDER.filter((s) => groups.has(s)),
    ...[...groups.keys()].filter(
      (s) => !(DOCS_SECTION_ORDER as readonly string[]).includes(s),
    ),
  ];

  return orderedSections.map((section) => ({
    section,
    items: (groups.get(section) ?? []).sort((a, b) => a.order - b.order),
  }));
}

export function getAdjacentDocs(
  locale: SiteLocale,
  slug: string[],
): { prev: DocsNavItem | null; next: DocsNavItem | null } {
  const flat = getDocsNavigation(locale).flatMap((g) => g.items);
  const key = slug.join("/");
  const index = flat.findIndex((item) => item.slug.join("/") === key);
  if (index < 0) return { prev: null, next: null };
  return {
    prev: flat[index - 1] ?? null,
    next: flat[index + 1] ?? null,
  };
}

export function findMissingTranslations(): {
  locale: SiteLocale;
  slug: string;
}[] {
  const en = new Set(getDocSlugs("en").map((s) => s.join("/")));
  const tr = new Set(getDocSlugs("tr").map((s) => s.join("/")));
  const missing: { locale: SiteLocale; slug: string }[] = [];
  for (const slug of en) {
    if (!tr.has(slug)) missing.push({ locale: "tr", slug });
  }
  for (const slug of tr) {
    if (!en.has(slug)) missing.push({ locale: "en", slug });
  }
  return missing;
}

/** Build-time validation: throws if any MDX frontmatter is invalid. */
export function validateAllDocsFrontmatter(): void {
  for (const locale of ["en", "tr"] as const) {
    for (const slug of getDocSlugs(locale)) {
      getDocBySlug(locale, slug);
    }
  }
}

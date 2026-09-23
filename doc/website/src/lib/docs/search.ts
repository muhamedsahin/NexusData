import { create, insertMultiple, search } from "@orama/orama";
import { getAllDocs } from "@/lib/docs/source";
import type { SiteLocale } from "@/config/site.config";

export type DocsSearchDoc = {
  id: string;
  title: string;
  description: string;
  content: string;
  href: string;
  section: string;
  locale: string;
};

const schema = {
  id: "string",
  title: "string",
  description: "string",
  content: "string",
  href: "string",
  section: "string",
  locale: "string",
} as const;

// Orama's inferred DB type is awkward across versions — keep opaque.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const cache = new Map<SiteLocale, any>();

export async function getSearchIndex(locale: SiteLocale) {
  const hit = cache.get(locale);
  if (hit) return hit;

  const db = create({ schema });

  const docs = getAllDocs(locale).map((doc) => ({
    id: doc.urlPath,
    title: doc.frontmatter.title,
    description: doc.frontmatter.description,
    content: doc.content.slice(0, 8000),
    href: doc.urlPath.replace(`/${locale}`, "") || "/docs",
    section: doc.frontmatter.section,
    locale,
  }));

  if (docs.length > 0) {
    await insertMultiple(db, docs);
  }
  cache.set(locale, db);
  return db;
}

export async function searchDocs(locale: SiteLocale, query: string, limit = 8) {
  if (!query.trim()) return [] as DocsSearchDoc[];
  const db = await getSearchIndex(locale);
  const result = await search(db, {
    term: query,
    limit,
    properties: ["title", "description", "content"],
    boost: { title: 3, description: 1.5, content: 1 },
  });
  return result.hits.map((hit) => hit.document as DocsSearchDoc);
}

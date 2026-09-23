import { describe, expect, it } from "vitest";
import { docsFrontmatterSchema } from "@/lib/docs/schema";
import { extractToc } from "@/lib/docs/toc";
import { findMissingTranslations, getAllDocs } from "@/lib/docs/source";

describe("docsFrontmatterSchema", () => {
  it("accepts valid frontmatter", () => {
    const parsed = docsFrontmatterSchema.parse({
      title: "Introduction",
      description: "About NexusData",
      status: "in-development",
      order: 10,
      locale: "en",
      section: "getting-started",
    });
    expect(parsed.title).toBe("Introduction");
    expect(parsed.draft).toBe(false);
  });

  it("rejects missing title", () => {
    const result = docsFrontmatterSchema.safeParse({
      description: "x",
      locale: "en",
      section: "getting-started",
    });
    expect(result.success).toBe(false);
  });
});

describe("extractToc", () => {
  it("collects h2-h4 headings", () => {
    const toc = extractToc("## Hello\n\n### World\n\n#### Deep\n");
    expect(toc.map((t) => t.id)).toEqual(["hello", "world", "deep"]);
  });
});

describe("docs content", () => {
  it("loads english docs without frontmatter errors", () => {
    const docs = getAllDocs("en");
    expect(docs.length).toBeGreaterThan(3);
  });

  it("reports no missing translations for starter set", () => {
    expect(findMissingTranslations()).toEqual([]);
  });
});

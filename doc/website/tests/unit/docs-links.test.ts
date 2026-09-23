import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const CONTENT = path.join(process.cwd(), "content", "docs");

function collectMdx(dir: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...collectMdx(full));
    else if (entry.name.endsWith(".mdx")) out.push(full);
  }
  return out;
}

function extractInternalDocLinks(markdown: string): string[] {
  const links: string[] = [];
  const re = /\[([^\]]*)\]\((\/docs\/[^)#\s]+)\)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(markdown))) {
    links.push(m[2]!);
  }
  return links;
}

function hrefToFile(locale: string, href: string): string | null {
  // /docs → index.mdx ; /docs/foo/bar → foo/bar.mdx
  const rest = href.replace(/^\/docs\/?/, "");
  if (!rest) return path.join(CONTENT, locale, "index.mdx");
  const candidate = path.join(CONTENT, locale, `${rest}.mdx`);
  if (fs.existsSync(candidate)) return candidate;
  const indexCandidate = path.join(CONTENT, locale, rest, "index.mdx");
  if (fs.existsSync(indexCandidate)) return indexCandidate;
  return null;
}

describe("docs internal links", () => {
  it("resolves /docs/... links to existing MDX for each locale", () => {
    const broken: string[] = [];
    for (const locale of ["en", "tr"]) {
      const files = collectMdx(path.join(CONTENT, locale));
      for (const file of files) {
        const text = fs.readFileSync(file, "utf8");
        for (const href of extractInternalDocLinks(text)) {
          // Prefer same-locale file; EN fallback for TR is allowed by app, but we still warn
          const local = hrefToFile(locale, href);
          const en = hrefToFile("en", href);
          if (!local && !en) {
            broken.push(`${locale}:${path.relative(CONTENT, file)} → ${href}`);
          }
        }
      }
    }
    expect(broken, broken.join("\n")).toEqual([]);
  });
});

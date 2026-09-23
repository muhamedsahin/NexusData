export type TocItem = {
  id: string;
  title: string;
  depth: number;
};

const HEADING_RE = /^(#{2,4})\s+(.+)$/gm;

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^\w\u00C0-\u024f\s-]/g, "")
    .replace(/\s+/g, "-");
}

/** Extract TOC from MDX markdown source (h2–h4). */
export function extractToc(markdown: string): TocItem[] {
  const items: TocItem[] = [];
  let match: RegExpExecArray | null;
  const re = new RegExp(HEADING_RE);
  while ((match = re.exec(markdown)) !== null) {
    const depth = match[1]!.length;
    const title = match[2]!.replace(/[*_`#]/g, "").trim();
    if (!title) continue;
    items.push({ id: slugify(title), title, depth });
  }
  return items;
}

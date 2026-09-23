"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/cn";
import type { TocItem } from "@/lib/docs/toc";

type DocsTocProps = {
  items: TocItem[];
};

export function DocsToc({ items }: DocsTocProps) {
  const t = useTranslations("docs");
  const [active, setActive] = useState<string | null>(items[0]?.id ?? null);

  useEffect(() => {
    if (items.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]?.target.id) setActive(visible[0].target.id);
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: [0, 0.25, 0.5, 1] },
    );
    for (const item of items) {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [items]);

  if (items.length === 0) return null;

  return (
    <nav aria-label={t("toc")} className="space-y-2 text-sm">
      <p className="font-medium text-[color:var(--fg)]">{t("onThisPage")}</p>
      <ul className="space-y-1 border-l border-[color:var(--border)]">
        {items.map((item) => (
          <li key={item.id} style={{ paddingLeft: (item.depth - 2) * 12 }}>
            <a
              href={`#${item.id}`}
              className={cn(
                "block border-l-2 border-transparent py-1 pl-3 text-[color:var(--fg-muted)] hover:text-[color:var(--fg)]",
                active === item.id &&
                  "border-[color:var(--accent)] text-[color:var(--accent)]",
              )}
            >
              {item.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

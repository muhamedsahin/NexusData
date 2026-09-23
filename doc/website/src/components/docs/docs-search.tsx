"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Search } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { Kbd } from "@/components/ui/kbd";
import { cn } from "@/lib/cn";
import type { DocsSearchDoc } from "@/lib/docs/search";
import type { SiteLocale } from "@/config/site.config";

type DocsSearchProps = {
  className?: string;
};

export function DocsSearch({ className }: DocsSearchProps) {
  const t = useTranslations("docs");
  const locale = useLocale() as SiteLocale;
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<DocsSearchDoc[]>([]);
  const [pending, startTransition] = useTransition();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(true);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!open) return;
    const q = query.trim();
    if (!q) {
      setResults([]);
      return;
    }
    const controller = new AbortController();
    startTransition(async () => {
      const res = await fetch(
        `/api/docs-search?locale=${locale}&q=${encodeURIComponent(q)}`,
        { signal: controller.signal },
      );
      if (!res.ok) return;
      const data = (await res.json()) as { results: DocsSearchDoc[] };
      setResults(data.results);
    });
    return () => controller.abort();
  }, [query, open, locale]);

  const emptyLabel = useMemo(() => {
    if (!query.trim()) return t("searchHint");
    if (pending) return t("searching");
    return t("noResults");
  }, [query, pending, t]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          "inline-flex h-9 w-full items-center gap-2 rounded-lg border border-[color:var(--border)] bg-[color:var(--surface)] px-3 text-sm text-[color:var(--fg-muted)] hover:border-[color:var(--border-strong)]",
          className,
        )}
      >
        <Search className="h-3.5 w-3.5" aria-hidden />
        <span className="flex-1 text-left">{t("searchPlaceholder")}</span>
        <span className="hidden items-center gap-1 sm:inline-flex">
          <Kbd>⌘</Kbd>
          <Kbd>K</Kbd>
        </span>
      </button>

      {open ? (
        <div
          className="fixed inset-0 z-[70] flex items-start justify-center bg-black/50 p-4 pt-[12vh]"
          role="dialog"
          aria-modal="true"
          aria-label={t("searchPlaceholder")}
          onClick={() => setOpen(false)}
        >
          <div
            className="w-full max-w-xl overflow-hidden rounded-2xl border border-[color:var(--border)] bg-[color:var(--bg-elevated)] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 border-b border-[color:var(--border)] px-3">
              <Search className="h-4 w-4 text-[color:var(--fg-muted)]" aria-hidden />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t("searchPlaceholder")}
                className="h-12 w-full bg-transparent text-sm text-[color:var(--fg)] outline-none"
              />
            </div>
            <ul className="max-h-80 overflow-y-auto p-2">
              {results.length === 0 ? (
                <li className="px-3 py-6 text-center text-sm text-[color:var(--fg-muted)]">
                  {emptyLabel}
                </li>
              ) : (
                results.map((hit) => (
                  <li key={hit.id}>
                    <Link
                      href={hit.href}
                      onClick={() => setOpen(false)}
                      className="block rounded-lg px-3 py-2 hover:bg-[color:var(--surface)]"
                    >
                      <p className="font-medium text-[color:var(--fg)]">{hit.title}</p>
                      <p className="line-clamp-2 text-xs text-[color:var(--fg-muted)]">
                        {hit.description}
                      </p>
                    </Link>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>
      ) : null}
    </>
  );
}

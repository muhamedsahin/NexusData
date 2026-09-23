"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { ChevronDown } from "lucide-react";
import { Link, usePathname } from "@/i18n/navigation";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/cn";
import type { DocsNavGroup } from "@/lib/docs/source";

type DocsSidebarProps = {
  groups: DocsNavGroup[];
};

export function DocsSidebar({ groups }: DocsSidebarProps) {
  const t = useTranslations("docs");
  const st = useTranslations("status");
  const pathname = usePathname();
  const [open, setOpen] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(groups.map((g) => [g.section, true])),
  );

  const normalized = useMemo(() => pathname.replace(/\/$/, "") || "/docs", [pathname]);

  return (
    <nav aria-label={t("sidebar")} className="space-y-3 text-sm">
      {groups.map((group) => {
        const isOpen = open[group.section] ?? true;
        return (
          <div key={group.section}>
            <button
              type="button"
              className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left font-medium text-[color:var(--fg)] hover:bg-[color:var(--surface)]"
              onClick={() =>
                setOpen((prev) => ({
                  ...prev,
                  [group.section]: !isOpen,
                }))
              }
              aria-expanded={isOpen}
            >
              <span>{t(`sections.${group.section}`)}</span>
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  isOpen ? "rotate-0" : "-rotate-90",
                )}
                aria-hidden
              />
            </button>
            {isOpen ? (
              <ul className="mt-1 space-y-0.5 border-l border-[color:var(--border)] pl-2">
                {group.items.map((item) => {
                  const active =
                    normalized === item.href ||
                    normalized === item.href.replace(/\/$/, "");
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        className={cn(
                          "flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-[color:var(--fg-muted)] hover:bg-[color:var(--surface)] hover:text-[color:var(--fg)]",
                          active &&
                            "bg-[color:var(--accent-soft)] text-[color:var(--accent)]",
                        )}
                      >
                        <span className="truncate">{item.title}</span>
                        <StatusBadge
                          status={item.status}
                          label={st(item.status)}
                          className="scale-90"
                        />
                      </Link>
                    </li>
                  );
                })}
              </ul>
            ) : null}
          </div>
        );
      })}
    </nav>
  );
}

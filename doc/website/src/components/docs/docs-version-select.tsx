"use client";

import { useTranslations } from "next-intl";
import {
  DEFAULT_DOCS_VERSION,
  DOCS_VERSIONS,
  type DocsVersion,
} from "@/lib/docs/schema";
import { cn } from "@/lib/cn";

type DocsVersionSelectProps = {
  className?: string;
};

/** Version selector scaffold — content is currently shared across versions. */
export function DocsVersionSelect({ className }: DocsVersionSelectProps) {
  const t = useTranslations("docs");
  const current: DocsVersion = DEFAULT_DOCS_VERSION;

  return (
    <label
      className={cn(
        "inline-flex items-center gap-2 text-xs text-[color:var(--fg-muted)]",
        className,
      )}
    >
      <span className="sr-only">{t("version")}</span>
      <select
        className="h-9 rounded-lg border border-[color:var(--border)] bg-[color:var(--surface)] px-2 text-sm text-[color:var(--fg)]"
        value={current}
        onChange={() => {
          /* Structure ready; multi-version content lands later */
        }}
        aria-label={t("version")}
      >
        {DOCS_VERSIONS.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </label>
  );
}

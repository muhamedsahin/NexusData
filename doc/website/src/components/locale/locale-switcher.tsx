"use client";

import { useLocale, useTranslations } from "next-intl";
import { usePathname, useRouter } from "@/i18n/navigation";
import { siteConfig, type SiteLocale } from "@/config/site.config";
import { cn } from "@/lib/cn";

function setLocaleCookie(locale: SiteLocale) {
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${siteConfig.localeCookie}=${locale}; Path=/; Max-Age=${maxAge}; SameSite=Lax`;
}

type LocaleSwitcherProps = {
  className?: string;
};

export function LocaleSwitcher({ className }: LocaleSwitcherProps) {
  const t = useTranslations("locale");
  const locale = useLocale() as SiteLocale;
  const pathname = usePathname();
  const router = useRouter();

  function switchTo(next: SiteLocale) {
    if (next === locale) return;
    setLocaleCookie(next);
    router.replace(pathname, { locale: next });
  }

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-lg border border-[color:var(--border)] bg-[color:var(--surface)] p-1",
        className,
      )}
      role="group"
      aria-label={t("label")}
    >
      {siteConfig.locales.map((code) => {
        const active = code === locale;
        return (
          <button
            key={code}
            type="button"
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-medium uppercase tracking-wide transition-colors duration-200 ease-[cubic-bezier(0.22,1,0.36,1)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]",
              active
                ? "bg-[color:var(--accent)] text-[color:var(--accent-fg)]"
                : "text-[color:var(--fg-muted)] hover:text-[color:var(--fg)]",
            )}
            aria-pressed={active}
            aria-label={t("switchTo", { locale: t(code) })}
            onClick={() => switchTo(code)}
          >
            {code}
          </button>
        );
      })}
    </div>
  );
}

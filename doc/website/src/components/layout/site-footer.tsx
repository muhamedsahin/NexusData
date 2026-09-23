import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { siteConfig } from "@/config/site.config";

export function SiteFooter() {
  const t = useTranslations("footer");

  return (
    <footer className="mt-auto border-t border-[color:var(--border)] bg-[color-mix(in_oklab,var(--bg)_88%,transparent)]">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-5 py-6 text-sm text-[color:var(--fg-muted)] sm:flex-row sm:items-center sm:justify-between">
        <p className="font-mono text-xs">
          <span className="font-semibold text-[color:var(--fg)]">{siteConfig.name}</span>{" "}
          <span className="opacity-60">{siteConfig.libraryVersion}</span>
          <span className="mx-2 opacity-40">·</span>
          {t("rights")}
        </p>
        <div className="flex flex-wrap items-center gap-5 text-xs">
          <span className="opacity-70">{t("license", { license: siteConfig.license })}</span>
          <Link
            href="/privacy"
            className="transition-colors duration-200 hover:text-[color:var(--fg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]"
          >
            {t("privacy")}
          </Link>
        </div>
      </div>
    </footer>
  );
}

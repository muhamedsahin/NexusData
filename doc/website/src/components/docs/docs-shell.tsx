import { getTranslations } from "next-intl/server";
import { Link } from "@/i18n/navigation";
import { DocsSidebar } from "@/components/docs/docs-sidebar";
import { DocsSearch } from "@/components/docs/docs-search";
import { DocsVersionSelect } from "@/components/docs/docs-version-select";
import { ThemeToggle } from "@/components/theme/theme-toggle";
import { LocaleSwitcher } from "@/components/locale/locale-switcher";
import { getDocsNavigation } from "@/lib/docs/source";
import type { SiteLocale } from "@/config/site.config";

type DocsShellProps = {
  locale: SiteLocale;
  children: React.ReactNode;
  toc?: React.ReactNode;
};

export async function DocsShell({ locale, children, toc }: DocsShellProps) {
  const t = await getTranslations({ locale, namespace: "docs" });
  const groups = getDocsNavigation(locale);

  return (
    <div className="mx-auto grid max-w-7xl gap-8 px-4 py-8 lg:grid-cols-[240px_minmax(0,1fr)_220px]">
      <aside className="hidden lg:block">
        <div className="sticky top-20 space-y-4">
          <DocsSearch />
          <DocsVersionSelect />
          <DocsSidebar groups={groups} />
        </div>
      </aside>

      <div className="min-w-0">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3 lg:hidden">
          <DocsSearch className="max-w-sm" />
          <div className="flex items-center gap-2">
            <DocsVersionSelect />
            <LocaleSwitcher />
            <ThemeToggle />
          </div>
        </div>
        <p className="mb-4 text-xs text-[color:var(--fg-muted)] lg:hidden">
          <Link href="/docs" className="text-[color:var(--accent)]">
            {t("docsHome")}
          </Link>
        </p>
        {children}
      </div>

      <aside className="hidden xl:block">
        <div className="sticky top-20">{toc}</div>
      </aside>
    </div>
  );
}

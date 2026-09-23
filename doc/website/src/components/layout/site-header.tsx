"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { LocaleSwitcher } from "@/components/locale/locale-switcher";
import { ThemeToggle } from "@/components/theme/theme-toggle";
import { resolvePublicLink, siteConfig } from "@/config/site.config";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/cn";

export function SiteHeader() {
  const t      = useTranslations();
  const github = resolvePublicLink(siteConfig.githubUrl);
  const [open, setOpen]         = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const links = (
    <>
      <Link
        href="/"
        className="text-[color:var(--fg-muted)] transition-colors duration-200 hover:text-[color:var(--fg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]"
        onClick={() => setOpen(false)}
      >
        {t("nav.home")}
      </Link>
      <Link
        href="/docs"
        className="text-[color:var(--fg-muted)] transition-colors duration-200 hover:text-[color:var(--fg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]"
        onClick={() => setOpen(false)}
      >
        {t("nav.docs")}
      </Link>
      <Link
        href="/docs/performance/benchmarks"
        className="text-[color:var(--fg-muted)] transition-colors duration-200 hover:text-[color:var(--fg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]"
        onClick={() => setOpen(false)}
      >
        {t("nav.performance")}
      </Link>
      {github ? (
        <a
          href={github}
          className="text-[color:var(--fg-muted)] transition-colors duration-200 hover:text-[color:var(--fg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]"
          rel="noopener noreferrer"
          target="_blank"
          onClick={() => setOpen(false)}
        >
          {t("nav.github")}
        </a>
      ) : null}
    </>
  );

  return (
    <header
      className={cn(
        "sticky top-0 z-40 transition-all duration-500",
        scrolled
          ? "border-b border-[color:var(--border)] bg-[color-mix(in_oklab,var(--bg)_86%,transparent)] backdrop-blur-xl"
          : "border-b border-transparent bg-transparent",
      )}
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-5 sm:px-8">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className={cn(
              "font-[family-name:var(--font-display)] text-[0.95rem] font-semibold tracking-[-0.03em] text-[color:var(--fg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]",
            )}
          >
            {siteConfig.name}
          </Link>
          <div className="hidden sm:inline-flex">
            <StatusBadge
              status={siteConfig.status}
              label={t(`status.${siteConfig.status}`)}
            />
          </div>
        </div>

        {/* Desktop nav */}
        <nav
          className="hidden items-center gap-6 text-sm md:flex"
          aria-label={t("nav.primaryNav")}
        >
          {links}
        </nav>

        {/* Right controls */}
        <div className="flex items-center gap-2">
          <LocaleSwitcher />
          <ThemeToggle />
          {/* Mobile hamburger */}
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[color:var(--border)] text-[color:var(--fg-muted)] transition-colors duration-200 hover:text-[color:var(--fg)] md:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label={open ? t("nav.closeMenu") : t("nav.openMenu")}
            onClick={() => setOpen((v) => !v)}
          >
            <span aria-hidden className="text-lg leading-none">
              {open ? "×" : "☰"}
            </span>
          </button>
        </div>
      </div>

      {/* Mobile nav */}
      <nav
        id="mobile-nav"
        className={cn(
          "overflow-hidden transition-all duration-300 md:hidden",
          open ? "max-h-60 border-t border-[color:var(--border)]" : "max-h-0",
        )}
        aria-label={t("nav.primaryNav")}
      >
        <div className="flex flex-col gap-4 bg-[color-mix(in_oklab,var(--bg)_94%,transparent)] px-5 py-4 text-sm backdrop-blur-xl">
          {links}
        </div>
      </nav>

    </header>
  );
}

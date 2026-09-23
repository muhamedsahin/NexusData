"use client";

import { useMemo, useState, useSyncExternalStore } from "react";
import { useLocale, useTranslations } from "next-intl";
import { usePathname, useRouter } from "@/i18n/navigation";
import { siteConfig, type SiteLocale } from "@/config/site.config";
import { Button } from "@/components/ui/button";

const DISMISS_KEY = "nexusdata-locale-hint-dismissed";

function preferredBrowserLocale(): SiteLocale | null {
  if (typeof navigator === "undefined") return null;
  const languages = navigator.languages?.length
    ? navigator.languages
    : [navigator.language];
  for (const lang of languages) {
    const lower = lang.toLowerCase();
    if (lower === "tr" || lower.startsWith("tr-")) return "tr";
    if (lower === "en" || lower.startsWith("en-")) return "en";
  }
  return null;
}

function setLocaleCookie(locale: SiteLocale) {
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${siteConfig.localeCookie}=${locale}; Path=/; Max-Age=${maxAge}; SameSite=Lax`;
}

function subscribeDismiss(onChange: () => void) {
  void onChange;
  return () => undefined;
}

function getDismissedSnapshot() {
  try {
    return sessionStorage.getItem(DISMISS_KEY) === "1";
  } catch {
    return true;
  }
}

function getServerDismissed() {
  return true;
}

function subscribeLanguages(onChange: () => void) {
  window.addEventListener("languagechange", onChange);
  return () => window.removeEventListener("languagechange", onChange);
}

function getPreferredSnapshot() {
  return preferredBrowserLocale();
}

function getServerPreferred(): SiteLocale | null {
  return null;
}

export function LanguageHintBanner() {
  const t = useTranslations("banner");
  const locale = useLocale() as SiteLocale;
  const pathname = usePathname();
  const router = useRouter();
  const [dismissedLocal, setDismissedLocal] = useState(false);
  const dismissedStored = useSyncExternalStore(
    subscribeDismiss,
    getDismissedSnapshot,
    getServerDismissed,
  );
  const preferred = useSyncExternalStore(
    subscribeLanguages,
    getPreferredSnapshot,
    getServerPreferred,
  );

  const suggested = useMemo(() => {
    if (dismissedLocal || dismissedStored) return null;
    if (!preferred || preferred === locale) return null;
    return preferred;
  }, [dismissedLocal, dismissedStored, preferred, locale]);

  if (!suggested) return null;

  function dismiss() {
    try {
      sessionStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* ignore */
    }
    setDismissedLocal(true);
  }

  function switchLocale() {
    const next = suggested;
    if (!next) return;
    setLocaleCookie(next);
    dismiss();
    router.replace(pathname, { locale: next });
  }

  const message = suggested === "tr" ? t("suggestTr") : t("suggestEn");

  return (
    <div
      className="border-b border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-2 text-sm text-[color:var(--fg)]"
      role="region"
      aria-label={message}
    >
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3">
        <p>{message}</p>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={switchLocale}>
            {t("switch")}
          </Button>
          <Button size="sm" variant="ghost" onClick={dismiss}>
            {t("dismiss")}
          </Button>
        </div>
      </div>
    </div>
  );
}

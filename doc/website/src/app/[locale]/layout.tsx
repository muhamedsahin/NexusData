import type { Metadata } from "next";
import { Geist, Geist_Mono, Sora } from "next/font/google";
import { notFound } from "next/navigation";
import { NextIntlClientProvider, hasLocale } from "next-intl";
import { getMessages, getTranslations, setRequestLocale } from "next-intl/server";
import { routing } from "@/i18n/routing";
import { ThemeProvider } from "@/components/theme/theme-provider";
import { SiteHeader } from "@/components/layout/site-header";
import { SiteFooter } from "@/components/layout/site-footer";
import { LanguageHintBanner } from "@/components/locale/language-hint-banner";
import { AnalyticsLoader } from "@/components/analytics/analytics-loader";
import { siteConfig } from "@/config/site.config";
import "../globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin", "latin-ext"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin", "latin-ext"],
  display: "swap",
});

const sora = Sora({
  variable: "--font-sora",
  subsets: ["latin", "latin-ext"],
  display: "swap",
});

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "meta" });
  const languages = Object.fromEntries(
    routing.locales.map((l) => [l, `/${l}`]),
  ) as Record<string, string>;

  return {
    metadataBase: new URL(siteConfig.url),
    title: {
      default: `${t("siteName")} — ${t("tagline")}`,
      template: `%s · ${t("siteName")}`,
    },
    description: t("description"),
    alternates: {
      canonical: `/${locale}`,
      languages: {
        ...languages,
        "x-default": `/${routing.defaultLocale}`,
      },
    },
    openGraph: {
      title: t("siteName"),
      description: t("description"),
      locale: locale === "tr" ? "tr_TR" : "en_US",
      type: "website",
      siteName: siteConfig.name,
    },
  };
}

type LocaleLayoutProps = {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
};

export default async function LocaleLayout({
  children,
  params,
}: LocaleLayoutProps) {
  const { locale } = await params;
  if (!hasLocale(routing.locales, locale)) {
    notFound();
  }

  setRequestLocale(locale);
  const messages = await getMessages();
  const t = await getTranslations("nav");

  return (
    <html
      lang={locale}
      className={`${geistSans.variable} ${geistMono.variable} ${sora.variable} h-full antialiased`}
      suppressHydrationWarning
      data-theme="dark"
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var m=document.cookie.match(/(?:^|; )${siteConfig.themeCookie}=([^;]*)/);var t=m?decodeURIComponent(m[1]):"system";var d=t==="dark"||(t!=="light"&&window.matchMedia("(prefers-color-scheme: dark)").matches);var r=document.documentElement;r.dataset.theme=d?"dark":"light";r.style.colorScheme=d?"dark":"light";}catch(e){}})();`,
          }}
        />
      </head>
      <body className="flex min-h-full flex-col">
        <NextIntlClientProvider messages={messages}>
          <ThemeProvider>
            <a href="#main-content" className="skip-link">
              {t("skipToContent")}
            </a>
            <LanguageHintBanner />
            <SiteHeader />
            <main id="main-content" className="flex-1">
              {children}
            </main>
            <SiteFooter />
            <AnalyticsLoader />
          </ThemeProvider>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}

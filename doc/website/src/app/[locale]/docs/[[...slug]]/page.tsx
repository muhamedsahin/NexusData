import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { Link } from "@/i18n/navigation";
import { DocsShell } from "@/components/docs/docs-shell";
import { DocsToc } from "@/components/docs/docs-toc";
import { DocsFeedbackWidget } from "@/components/docs/docs-feedback";
import { MdxContent } from "@/components/mdx/mdx-content";
import { StatusBadge } from "@/components/ui/status-badge";
import { Callout } from "@/components/ui/callout";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/cn";
import {
  resolvePublicLink,
  siteConfig,
  type SiteLocale,
} from "@/config/site.config";
import {
  getAdjacentDocs,
  getDocBySlug,
  getDocSlugs,
  validateAllDocsFrontmatter,
} from "@/lib/docs/source";
import { extractToc } from "@/lib/docs/toc";

type DocsPageProps = {
  params: Promise<{ locale: string; slug?: string[] }>;
};

export function generateStaticParams() {
  validateAllDocsFrontmatter();
  const params: { locale: string; slug: string[] }[] = [];
  for (const locale of siteConfig.locales) {
    params.push({ locale, slug: [] });
    for (const slug of getDocSlugs(locale)) {
      params.push({ locale, slug });
    }
  }
  return params;
}

export async function generateMetadata({
  params,
}: DocsPageProps): Promise<Metadata> {
  const { locale, slug = [] } = await params;
  const doc =
    getDocBySlug(locale as SiteLocale, slug) ??
    (locale === "tr" ? getDocBySlug("en", slug) : null);
  if (!doc) return { title: "Docs" };

  const path = slug.length ? `docs/${slug.join("/")}` : "docs";
  const languages = Object.fromEntries(
    siteConfig.locales.map((l) => [l, `/${l}/${path}`]),
  );

  return {
    title: doc.frontmatter.title,
    description: doc.frontmatter.description,
    alternates: {
      canonical: `/${locale}/${path}`,
      languages: {
        ...languages,
        "x-default": `/en/${path}`,
      },
    },
    openGraph: {
      title: doc.frontmatter.title,
      description: doc.frontmatter.description,
      type: "article",
      locale: locale === "tr" ? "tr_TR" : "en_US",
    },
  };
}

export default async function DocsPage({ params }: DocsPageProps) {
  const { locale: localeParam, slug = [] } = await params;
  const locale = localeParam as SiteLocale;
  setRequestLocale(locale);

  let doc = getDocBySlug(locale, slug);
  let fallbackEn = false;
  if (!doc && locale === "tr") {
    doc = getDocBySlug("en", slug);
    fallbackEn = Boolean(doc);
  }
  if (!doc) notFound();

  const t = await getTranslations({ locale, namespace: "docs" });
  const st = await getTranslations({ locale, namespace: "status" });
  const toc = extractToc(doc.content);
  const { prev, next } = getAdjacentDocs(
    fallbackEn ? "en" : locale,
    slug,
  );
  const github = resolvePublicLink(siteConfig.githubUrl);
  const editUrl =
    github && siteConfig.docsEditBase
      ? `${siteConfig.docsEditBase}/${doc.locale}/${slug.join("/") || "index"}.mdx`
      : null;

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: doc.frontmatter.title,
    description: doc.frontmatter.description,
    inLanguage: locale,
    dateModified: doc.lastModified,
    author: {
      "@type": "Organization",
      name: siteConfig.name,
    },
  };

  return (
    <DocsShell locale={locale} toc={<DocsToc items={toc} />}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <article className="docs-article min-w-0">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <StatusBadge
            status={doc.frontmatter.status}
            label={st(doc.frontmatter.status)}
          />
          {doc.frontmatter.since ? (
            <span className="rounded-md border border-[color:var(--border)] px-2 py-0.5 text-xs text-[color:var(--fg-muted)]">
              since {doc.frontmatter.since}
            </span>
          ) : null}
          <span className="text-xs text-[color:var(--fg-muted)]">
            {t("readingTime", { minutes: doc.readingMinutes })}
          </span>
        </div>

        <h1 className="font-[family-name:var(--font-display)] text-4xl font-semibold tracking-tight text-[color:var(--fg)]">
          {doc.frontmatter.title}
        </h1>
        <p className="mt-3 text-lg text-[color:var(--fg-muted)]">
          {doc.frontmatter.description}
        </p>

        {fallbackEn ? (
          <Callout tone="warning" className="mt-6" title={t("untranslatedTitle")}>
            {t("untranslatedBody")}
          </Callout>
        ) : null}

        <div className="docs-prose mt-8">
          <MdxContent source={doc.content} />
        </div>

        <footer className="mt-12 space-y-6 border-t border-[color:var(--border)] pt-6">
          <p className="text-xs text-[color:var(--fg-muted)]">
            {t("lastUpdated", {
              date: new Intl.DateTimeFormat(locale, {
                dateStyle: "medium",
              }).format(new Date(doc.lastModified)),
            })}
          </p>
          {editUrl ? (
            <a
              href={editUrl}
              className="text-sm text-[color:var(--accent)] hover:underline"
              rel="noopener noreferrer"
              target="_blank"
            >
              {t("editOnGithub")}
            </a>
          ) : null}

          <DocsFeedbackWidget />

          <div className="flex flex-wrap justify-between gap-3">
            {prev ? (
              <Link
                href={prev.href}
                className={cn(buttonVariants({ variant: "secondary" }))}
              >
                ← {prev.title}
              </Link>
            ) : (
              <span />
            )}
            {next ? (
              <Link
                href={next.href}
                className={cn(buttonVariants({ variant: "secondary" }))}
              >
                {next.title} →
              </Link>
            ) : null}
          </div>
        </footer>
      </article>
    </DocsShell>
  );
}

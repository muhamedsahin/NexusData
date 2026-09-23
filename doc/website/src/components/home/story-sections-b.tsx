"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { StatusBadge } from "@/components/ui/status-badge";
import { Callout } from "@/components/ui/callout";
import { buttonVariants } from "@/components/ui/button";
import { InstallSnippet } from "@/components/home/install-snippet";
import { StorySection } from "@/components/home/story-sections-a";
import {
  ecosystemNodes,
  resolvePublicLink,
  siteConfig,
  type FeatureStatus,
} from "@/config/site.config";
import { cn } from "@/lib/cn";
import previewBenchmarks from "../../../data/benchmarks/measured-2026-09-23.json";

const FEATURES: { id: string; status: FeatureStatus; href: string }[] = [
  { id: "determinism",   status: "in-development", href: "/docs" },
  { id: "zerocopy",      status: "in-development", href: "/docs" },
  { id: "streaming",     status: "planned",         href: "/docs" },
  { id: "preprocessing", status: "planned",         href: "/docs" },
  { id: "split",         status: "in-development", href: "/docs" },
  { id: "cache",         status: "planned",         href: "/docs" },
  { id: "formats",       status: "in-development", href: "/docs" },
  { id: "gpu",           status: "planned",         href: "/docs" },
];

const CODE_TABS = [
  { id: "csv",   status: "stable" as const },
  { id: "image", status: "stable" as const },
  { id: "gpu",   status: "beta" as const },
  { id: "flash", status: "planned" as const },
];

/* ── Shared GlassCard (mirrors story-sections-a) ── */
function GlassCard({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("glow-card rounded-2xl p-6 sm:p-8", className)}>
      {children}
    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-[family-name:var(--font-display)] text-3xl font-semibold tracking-[-0.03em] text-[color:var(--fg)]">
      {children}
    </h2>
  );
}

/* ── 3D Tilt Feature Card ── */
function TiltCard({
  title,
  body,
  status,
  href,
  statusLabel,
}: {
  title: string;
  body: string;
  status: FeatureStatus;
  href: string;
  statusLabel: string;
}) {
  const [style, setStyle] = useState<React.CSSProperties>({});

  return (
    <Link
      href={href}
      className="block rounded-2xl border border-[color:var(--border)] bg-[color-mix(in_oklab,var(--bg-elevated)_88%,transparent)] p-5 backdrop-blur-md transition-[border-color,transform] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] hover:border-[color-mix(in_oklab,var(--accent)_45%,var(--border))]"
      style={style}
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        setStyle({
          transform: `perspective(900px) rotateY(${(x - 0.5) * 4}deg) rotateX(${(0.5 - y) * 4}deg)`,
          background: `radial-gradient(
            520px circle at ${x * 100}% ${y * 100}%,
            color-mix(in oklab, var(--accent) 12%, transparent),
            transparent 55%
          ), color-mix(in oklab, var(--bg-elevated) 88%, transparent)`,
        });
      }}
      onMouseLeave={() => setStyle({})}
    >
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="font-semibold text-[color:var(--fg)]">{title}</h3>
        <StatusBadge status={status} label={statusLabel} />
      </div>
      <p className="text-sm leading-relaxed text-[color:var(--fg-muted)]">{body}</p>
    </Link>
  );
}

/* ── Ecosystem ── */
export function EcosystemSection({ locale }: { locale: string }) {
  const t  = useTranslations("story.ecosystem");
  const st = useTranslations("status");

  return (
    <StorySection id="ecosystem" index="05">
      <GlassCard className="max-w-3xl">
        <SectionHeading>{t("title")}</SectionHeading>
        <p className="mt-3 text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>
        <ul className="mt-7 grid gap-3 sm:grid-cols-2">
          {ecosystemNodes.map((node) => (
            <li
              key={node.id}
              className={cn(
                "rounded-xl border p-4 transition-all duration-300",
                node.id === "nexusdata"
                  ? "border-[rgba(0,245,220,0.35)] bg-[rgba(0,245,220,0.07)] shadow-[inset_0_0_30px_rgba(0,245,220,0.05)]"
                  : "border-white/8 bg-white/[0.03] opacity-75 hover:opacity-90 hover:border-white/15",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <p className={cn(
                  "font-semibold",
                  node.id === "nexusdata" ? "text-[color:var(--accent)]" : "text-[color:var(--fg)]"
                )}>
                  {node.name}
                </p>
                <StatusBadge status={node.status} label={st(node.status)} />
              </div>
              <p className="mt-1.5 text-sm text-[color:var(--fg-muted)] leading-relaxed">
                {node.role[locale as "en" | "tr"] ?? node.role.en}
              </p>
            </li>
          ))}
        </ul>
      </GlassCard>
    </StorySection>
  );
}

/* ── Features ── */
export function FeaturesSection() {
  const t  = useTranslations("story.features");
  const st = useTranslations("status");

  return (
    <StorySection id="features" index="06">
      <div>
        <SectionHeading>{t("title")}</SectionHeading>
        <p className="mt-3 max-w-2xl text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f) => (
            <TiltCard
              key={f.id}
              title={t(`items.${f.id}.title`)}
              body={t(`items.${f.id}.body`)}
              status={f.status}
              href={f.href}
              statusLabel={st(f.status)}
            />
          ))}
        </div>
      </div>
    </StorySection>
  );
}

/* ── Performance ── */
export function PerformanceSection({ locale }: { locale: string }) {
  const t    = useTranslations("story.performance");
  const st   = useTranslations("status");
  const bench = previewBenchmarks[0];
  const illustrative = bench?.illustrative !== false;

  const bars = useMemo(() => {
    if (!bench) return [];
    const max = Math.max(...bench.results.map((r) => r.value), 1);
    return bench.results.map((r) => ({
      ...r,
      pct: (r.value / max) * 100,
    }));
  }, [bench]);

  return (
    <StorySection id="performance" index="07">
      <GlassCard className="max-w-2xl">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <SectionHeading>{t("title")}</SectionHeading>
          {illustrative ? (
            <StatusBadge status="illustrative" label={st("illustrative")} />
          ) : null}
        </div>
        <p className="text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>
        {illustrative ? (
          <Callout tone="warning" className="mt-4" title={st("notMeasured")}>
            {locale === "tr" ? bench?.methodology.tr : bench?.methodology.en}
          </Callout>
        ) : null}

        {!illustrative ? (
        <div className="mt-7 space-y-4">
          {bars.map((bar, i) => (
            <div key={bar.name}>
              <div className="mb-1.5 flex justify-between text-xs text-[color:var(--fg-muted)]">
                <span className="font-medium">{bar.name}</span>
                <span className="font-mono text-[color:var(--accent)]">
                  {bar.value} {bench?.unit}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${bar.pct}%`,
                    background: "var(--accent)",
                    transitionDelay: `${i * 80}ms`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
        ) : null}
        <Link
          href="/docs/performance/benchmarks"
          className={cn(buttonVariants({ variant: "secondary" }), "mt-7 inline-flex hover:border-[rgba(0,245,220,0.3)]")}
        >
          {t("cta")}
        </Link>
      </GlassCard>
    </StorySection>
  );
}

/* ── Code ── */
export function CodeSection() {
  const t      = useTranslations("story.code");
  const st     = useTranslations("status");
  const [tab, setTab] = useState("csv");
  const active = CODE_TABS.find((c) => c.id === tab) ?? CODE_TABS[0]!;

  return (
    <StorySection id="code" index="08">
      <GlassCard className="max-w-3xl">
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <SectionHeading>{t("title")}</SectionHeading>
          <StatusBadge status={active.status} label={st(active.status)} />
        </div>

        {/* Tab bar */}
        <div className="mb-4 flex flex-wrap gap-1 rounded-xl border border-white/8 bg-[rgba(255,255,255,0.03)] p-1" role="tablist">
          {CODE_TABS.map((c) => (
            <button
              key={c.id}
              type="button"
              role="tab"
              aria-selected={tab === c.id}
              className={cn(
                "rounded-lg px-4 py-1.5 text-xs font-medium transition-all duration-200",
                tab === c.id
                  ? "bg-[rgba(0,245,220,0.12)] text-[color:var(--accent)] shadow-[0_0_12px_rgba(0,245,220,0.2)]"
                  : "text-[color:var(--fg-muted)] hover:text-[color:var(--fg)] hover:bg-white/[0.04]",
              )}
              onClick={() => setTab(c.id)}
            >
              {t(`tabs.${c.id}`)}
            </button>
          ))}
        </div>

        {/* Code block */}
        <pre className="overflow-x-auto rounded-xl border border-[rgba(0,245,220,0.12)] bg-[#030814] p-5 font-mono text-xs leading-relaxed text-[color:var(--fg-muted)] sm:text-sm shadow-[inset_0_0_30px_rgba(0,245,220,0.04)]">
          <code>{t.raw(`snippets.${tab}`)}</code>
        </pre>
        <p className="mt-3 text-xs text-[color:var(--fg-muted)]/70">{t("hoverHint")}</p>
      </GlassCard>
    </StorySection>
  );
}

/* ── CTA ── */
export function CtaSection({ github }: { github: string | null }) {
  const t  = useTranslations("story.cta");
  const th = useTranslations("home");
  const link = github ?? resolvePublicLink(siteConfig.githubUrl);

  return (
    <StorySection id="cta" index="09" className="pb-32">
      <GlassCard className="max-w-2xl">
        <SectionHeading>{t("title")}</SectionHeading>
        <p className="mt-3 text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            href="/docs"
            className={buttonVariants({ variant: "primary", size: "lg" })}
          >
            {th("ctaStart")}
          </Link>
          {link ? (
            <a
              href={link}
              className={cn(
                buttonVariants({ variant: "outline", size: "lg" }),
                "border-white/10 text-[color:var(--fg-muted)] hover:text-white hover:border-white/20",
              )}
              rel="noopener noreferrer"
              target="_blank"
            >
              {th("ctaGithub")}
            </a>
          ) : null}
        </div>
        <InstallSnippet className="mt-7" />
      </GlassCard>
    </StorySection>
  );
}

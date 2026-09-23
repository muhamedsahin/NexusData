"use client";

import { useTranslations } from "next-intl";
import { StatusBadge } from "@/components/ui/status-badge";
import { useStoryStore } from "@/lib/stores/story-store";
import { cn } from "@/lib/cn";
import type { FeatureStatus } from "@/config/site.config";

const FORMATS: {
  id: string;
  status: FeatureStatus;
}[] = [
  { id: "csv",     status: "stable" },
  { id: "json",    status: "stable" },
  { id: "image",   status: "stable" },
  { id: "parquet", status: "beta" },
  { id: "sqlite",  status: "beta" },
  { id: "npy",     status: "beta" },
];

type StorySectionProps = {
  id: string;
  index?: string;
  children: React.ReactNode;
  className?: string;
};

export function StorySection({ id, index, children, className }: StorySectionProps) {
  return (
    <section
      id={`scene-${id}`}
      data-scene={id}
      className={cn(
        "relative z-10 flex min-h-[100svh] items-center px-5 py-28 sm:px-8 lg:px-12",
        className,
      )}
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(to right, rgba(9,10,12,0.92) 0%, rgba(9,10,12,0.72) 34%, rgba(9,10,12,0.28) 52%, transparent 70%)",
        }}
        aria-hidden
      />
      <div className="pointer-events-auto relative mx-auto w-full max-w-7xl">
        {index ? (
          <p className="mb-4 font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--fg-muted)]">
            {index}
          </p>
        ) : null}
        {children}
      </div>
    </section>
  );
}

/* ── Shared inner card with glassmorphism style ── */
function GlassCard({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "glow-card rounded-2xl p-6 sm:p-8",
        className,
      )}
    >
      {children}
    </div>
  );
}

/* ── Section heading style ── */
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-[family-name:var(--font-display)] text-3xl font-semibold tracking-[-0.03em] text-[color:var(--fg)]">
      {children}
    </h2>
  );
}

export function SourcesSection() {
  const t  = useTranslations("story.sources");
  const st = useTranslations("status");

  return (
    <StorySection id="sources" index="01">
      <GlassCard className="max-w-xl">
        <SectionHeading>{t("title")}</SectionHeading>
        <p className="mt-3 text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>
        <ul className="mt-7 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {FORMATS.map((f) => (
            <li
              key={f.id}
              className={cn(
                "group rounded-xl border p-3 transition-all duration-300",
                f.status === "in-development"
                  ? "border-[color:var(--border-strong)] bg-[color:var(--accent-soft)] hover:border-[color:var(--accent)]"
                  : "border-white/8 bg-white/[0.03] hover:border-white/15 hover:bg-white/[0.05]",
              )}
              title={t(`formats.${f.id}.hint`)}
            >
              <p className="font-mono text-sm font-medium text-[color:var(--fg)] group-hover:text-[color:var(--accent)] transition-colors duration-300">
                {t(`formats.${f.id}.label`)}
              </p>
              <div className="mt-2">
                <StatusBadge status={f.status} label={st(f.status)} />
              </div>
            </li>
          ))}
        </ul>
      </GlassCard>
    </StorySection>
  );
}

export function ShuffleSection() {
  const t        = useTranslations("story.shuffle");
  const st       = useTranslations("status");
  const seed     = useStoryStore((s) => s.seed);
  const indices  = useStoryStore((s) => s.indices);
  const setSeed  = useStoryStore((s) => s.setSeed);

  return (
    <StorySection id="shuffle" index="02">
      <GlassCard className="max-w-xl">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <SectionHeading>{t("title")}</SectionHeading>
          <StatusBadge status="illustrative" label={st("illustrative")} />
        </div>
        <p className="text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>

        <label className="mt-7 block text-sm text-[color:var(--fg-muted)]">
          {t("seed")} —{" "}
          <span className="font-mono font-semibold text-[color:var(--accent)]">{seed}</span>
          <input
            type="range"
            min={0}
            max={255}
            value={seed % 256}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="mt-3 w-full accent-[color:var(--accent)] cursor-pointer"
          />
        </label>

        <p className="mt-5 font-mono text-xs leading-relaxed text-[color:var(--fg-muted)] bg-[rgba(0,245,220,0.04)] rounded-lg p-3 border border-[rgba(0,245,220,0.12)]">
          [{indices.slice(0, 16).join(", ")} …]
        </p>
        <p className="mt-3 text-xs text-[color:var(--fg-muted)]/70">{t("note")}</p>
      </GlassCard>
    </StorySection>
  );
}

export function BatchSection() {
  const t           = useTranslations("story.batch");
  const batchSize   = useStoryStore((s) => s.batchSize);
  const dropLast    = useStoryStore((s) => s.dropLast);
  const plan        = useStoryStore((s) => s.batchPlan);
  const setBatchSize = useStoryStore((s) => s.setBatchSize);
  const setDropLast  = useStoryStore((s) => s.setDropLast);

  return (
    <StorySection id="batch" index="03">
      <GlassCard className="max-w-xl">
        <SectionHeading>{t("title")}</SectionHeading>
        <p className="mt-3 text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>

        <label className="mt-7 block text-sm text-[color:var(--fg-muted)]">
          batch_size —{" "}
          <span className="font-mono font-semibold text-[color:var(--accent)]">
            {batchSize}
          </span>
          <input
            type="range"
            min={2}
            max={16}
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            className="mt-3 w-full accent-[color:var(--accent)] cursor-pointer"
          />
        </label>

        <label className="mt-5 flex items-center gap-3 text-sm text-[color:var(--fg)] cursor-pointer">
          <input
            type="checkbox"
            checked={dropLast}
            onChange={(e) => setDropLast(e.target.checked)}
            className="accent-[color:var(--accent)] h-4 w-4 rounded cursor-pointer"
          />
          drop_last
        </label>

        <p className="mt-5 font-mono text-sm font-medium text-[color:var(--accent)] bg-[rgba(0,245,220,0.06)] rounded-lg p-3 border border-[rgba(0,245,220,0.15)]">
          {plan.shapeLabel} · {plan.batches.length} batches
          {plan.dropped.length > 0 ? ` · dropped ${plan.dropped.length}` : ""}
        </p>
      </GlassCard>
    </StorySection>
  );
}

export function GpuSection() {
  const t = useTranslations("story.gpu");

  return (
    <StorySection id="gpu" index="04">
      <GlassCard className="max-w-xl">
        <SectionHeading>{t("title")}</SectionHeading>
        <p className="mt-3 text-[color:var(--fg-muted)] leading-relaxed">{t("body")}</p>
        <p className="mt-4 text-sm font-medium text-[#fbbf24]">
          {t("policy")}
        </p>
        <div className="mt-6 rounded-xl border border-dashed border-[color:var(--border-strong)] p-5 text-sm text-[color:var(--fg-muted)]">
          {t("chartEmpty")}
        </div>
      </GlassCard>
    </StorySection>
  );
}

"use client";

import { useRef, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { HomeCanvas } from "@/components/three/home-canvas";
import { PointerBridge } from "@/components/three/pointer-bridge";
import { LoadingOverlay } from "@/components/loader/loading-overlay";
import { LoaderDirector } from "@/components/loader/loader-director";
import { HomeLenis } from "@/components/layout/home-lenis";
import { InstallSnippet } from "@/components/home/install-snippet";
import { LatticeTooltip } from "@/components/home/lattice-tooltip";
import { CustomCursor } from "@/components/home/custom-cursor";
import { StatusBadge } from "@/components/ui/status-badge";
import { Callout } from "@/components/ui/callout";
import { buttonVariants } from "@/components/ui/button";
import {
  BatchSection,
  GpuSection,
  ShuffleSection,
  SourcesSection,
} from "@/components/home/story-sections-a";
import {
  CodeSection,
  CtaSection,
  EcosystemSection,
  FeaturesSection,
  PerformanceSection,
} from "@/components/home/story-sections-b";
import { cn } from "@/lib/cn";
import { useLoaderStore } from "@/lib/stores/loader-store";

gsap.registerPlugin(ScrollTrigger);

type HomeExperienceProps = {
  locale: string;
  github: string | null;
};

export function HomeExperience({ locale, github }: HomeExperienceProps) {
  const t = useTranslations();
  const shellRef = useRef<HTMLDivElement>(null);
  const heroTextRef = useRef<HTMLDivElement>(null);
  const phase = useLoaderStore((s) => s.phase);
  const revealed =
    phase === "ready" || phase === "skipped" || phase === "exploding";

  /* ── Scroll-driven parallax for hero text ── */
  useEffect(() => {
    if (!heroTextRef.current) return;

    const ctx = gsap.context(() => {
      gsap.to(heroTextRef.current, {
        y: -48,
        opacity: 0,
        ease: "none",
        scrollTrigger: {
          trigger: "#scene-hero",
          start: "top top",
          end: "40% top",
          scrub: 0.6,
        },
      });
    });

    return () => ctx.revert();
  }, []);

  return (
    <div ref={shellRef} className="home-stage relative w-full overflow-x-clip">
      {/* Global keyframes for the ambient glow orbs below. Safe to move into globals.css. */}
      <style>{`
        @keyframes floatGlowA {
          0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
          50% { transform: translate3d(3%, -4%, 0) scale(1.08); }
        }
        @keyframes floatGlowB {
          0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
          50% { transform: translate3d(-4%, 3%, 0) scale(1.05); }
        }
      `}</style>

      <HomeLenis />
      <div
        className="pointer-events-none fixed inset-x-0 top-0 z-30 h-28 bg-gradient-to-b from-[#090a0c] via-[#090a0c]/75 to-transparent"
        aria-hidden
      />
      <LoaderDirector />
      <PointerBridge targetRef={shellRef} />
      <CustomCursor />
      <LatticeTooltip />
      <LoadingOverlay />

      {/* 3D WebGL Background Layer — full screen, always behind */}
      <div className="pointer-events-auto fixed inset-0 z-0">
        <HomeCanvas className="h-full w-full" />
      </div>

      {/* ── Hero Section ─────────────────────────────────────────────────── */}
      <section
        id="scene-hero"
        data-scene="hero"
        className="pointer-events-none relative z-10 flex min-h-[100svh] flex-col justify-end overflow-hidden px-5 pb-16 pt-24 sm:px-8 lg:justify-center lg:px-12 lg:pb-20 lg:pt-0"
      >
        {/* Ambient glow orbs — this is what makes the hero feel alive instead of flat */}
        <div
          className="pointer-events-none absolute -left-24 top-1/4 z-0 h-[420px] w-[420px] rounded-full opacity-40 blur-[110px]"
          style={{
            background: "radial-gradient(circle, #2DE2D0 0%, transparent 70%)",
            animation: "floatGlowA 14s ease-in-out infinite",
          }}
          aria-hidden
        />
        <div
          className="pointer-events-none absolute right-0 top-1/3 z-0 h-[480px] w-[480px] rounded-full opacity-30 blur-[130px]"
          style={{
            background: "radial-gradient(circle, #5B4BDB 0%, transparent 70%)",
            animation: "floatGlowB 18s ease-in-out infinite",
          }}
          aria-hidden
        />

        <div
          className="pointer-events-none absolute inset-0 z-0 hidden lg:block"
          style={{
            background:
              "linear-gradient(to right, #090a0c 0%, rgba(9,10,12,0.94) 26%, rgba(9,10,12,0.55) 40%, transparent 56%)",
          }}
          aria-hidden
        />
        <div
          className="pointer-events-none absolute inset-0 z-0 lg:hidden"
          style={{
            background:
              "linear-gradient(to top, #090a0c 0%, #090a0c 46%, rgba(9,10,12,0.92) 58%, transparent 76%)",
          }}
          aria-hidden
        />

        <div className="min-h-[46svh] shrink-0 lg:hidden" aria-hidden />

        <div ref={heroTextRef} className="relative z-10 mx-auto w-full max-w-7xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 lg:items-center gap-8">
            <div
              className={cn(
                "pointer-events-auto flex w-full max-w-xl flex-col lg:col-span-6 xl:col-span-5 transition-opacity duration-700 ease-[cubic-bezier(0.22,1,0.36,1)]",
                revealed ? "opacity-100" : "opacity-0",
              )}
            >
              <div className="flex flex-col gap-5">
                <StatusBadge
                  className="w-fit"
                  status="design-preview"
                  label={t("status.designPreview")}
                />
                <h1 className="font-[family-name:var(--font-display)] text-[clamp(3.25rem,6.4vw,5.5rem)] font-semibold leading-[0.94] tracking-[-0.04em] text-[color:var(--fg)]">
                  {t("home.title")}
                </h1>
                <p className="max-w-md text-lg leading-relaxed text-[color:var(--fg-muted)]">
                  {t("home.subtitle")}
                </p>
                <p className="max-w-lg font-mono text-xs tracking-[0.04em] text-[color:var(--fg-muted)]">
                  {t("home.pipeline")}
                </p>
                <p className="max-w-lg text-sm leading-relaxed text-[color:var(--fg)]">
                  {t("home.lede")}
                </p>
                <ul className="max-w-lg space-y-1.5 text-sm leading-snug text-[color:var(--fg-muted)]">
                  <li>{t("home.useTrain")}</li>
                  <li>{t("home.useTabular")}</li>
                  <li>{t("home.useMedia")}</li>
                </ul>
              </div>

              <div className="mt-8 flex flex-col gap-5">
                <div className="flex flex-wrap items-center gap-3">
                  <Link
                    href="/docs"
                    className={buttonVariants({ variant: "primary", size: "lg" })}
                  >
                    {t("home.ctaStart")}
                  </Link>
                  <Link
                    href="/docs/examples"
                    className={buttonVariants({ variant: "secondary", size: "lg" })}
                  >
                    {t("home.ctaExamples")}
                  </Link>
                  {github ? (
                    <a
                      href={github}
                      className={buttonVariants({ variant: "outline", size: "lg" })}
                      rel="noopener noreferrer"
                      target="_blank"
                    >
                      {t("home.ctaGithub")}
                    </a>
                  ) : null}
                </div>
                <InstallSnippet />
                <div className="flex items-center gap-3 pt-1">
                  <span className="h-px w-8 bg-gradient-to-r from-[#2DE2D0] to-transparent" aria-hidden />
                  <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--fg-muted)]">
                    {t("home.scrollHint")}
                  </p>
                </div>
              </div>
            </div>

            {/* Right column — transparent for 3D lattice */}
            <div className="hidden lg:col-span-6 xl:col-span-7 lg:block pointer-events-none h-full min-h-[500px]" />
          </div>
        </div>
      </section>

      {/* Divider with real gradient glow instead of a flat line */}
      <div
        className="relative z-10 mx-auto max-w-7xl px-8"
      >
        <div
          className="h-px w-full bg-gradient-to-r from-transparent via-[#2DE2D0]/70 to-transparent"
          style={{ boxShadow: "0 0 24px 1px rgba(45,226,208,0.35)" }}
        />
      </div>

      {/* ── Story Sections ───────────────────────────────────────────────── */}
      <SourcesSection />
      <ShuffleSection />
      <BatchSection />
      <GpuSection />
      <EcosystemSection locale={locale} />
      <FeaturesSection />
      <PerformanceSection locale={locale} />
      <CodeSection />
      <CtaSection github={github} />

      <div className="relative z-10 mx-auto max-w-6xl px-4 pb-16">
        <Callout tone="note" title={t("home.statusNote")}>
          {t("home.privacyNote")}
        </Callout>
      </div>
    </div>
  );
}
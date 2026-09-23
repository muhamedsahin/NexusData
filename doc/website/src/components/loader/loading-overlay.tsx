"use client";

import { useEffect, useMemo } from "react";
import { useTranslations } from "next-intl";
import { useLoaderStore, type LoaderLogId } from "@/lib/stores/loader-store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cn";

const LOG_ORDER: LoaderLogId[] = ["mmap", "shaders", "rng", "warmup", "done"];

type LoadingOverlayProps = {
  className?: string;
};

export function LoadingOverlay({ className }: LoadingOverlayProps) {
  const t = useTranslations("loader");
  const phase = useLoaderStore((s) => s.phase);
  const progress = useLoaderStore((s) => s.progress);
  const logs = useLoaderStore((s) => s.logs);
  const reduced = useLoaderStore((s) => s.reducedMotion);
  const webgl = useLoaderStore((s) => s.webgl);

  const visible = phase !== "ready" && phase !== "skipped";
  const showSimple = reduced || webgl === false;

  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key.toLowerCase() === "s") {
        useLoaderStore.getState().skip();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible]);

  const logLines = useMemo(() => {
    return LOG_ORDER.filter((id) => logs.includes(id)).map((id) => t(`logs.${id}`));
  }, [logs, t]);

  if (!visible) return null;

  return (
    <div
      className={cn(
        "fixed inset-0 z-50 flex flex-col bg-[color:var(--bg)] transition-transform duration-700 ease-[cubic-bezier(0.22,1,0.36,1)]",
        phase === "exploding" ? "-translate-y-full pointer-events-none" : "translate-y-0",
        className,
      )}
      role="dialog"
      aria-modal="true"
      aria-label={t("aria")}
    >
        <div className="absolute right-4 top-4 z-10">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={() => useLoaderStore.getState().skip()}
          >
            {t("skip")}
          </Button>
        </div>

        <div className="mx-auto flex h-full w-full max-w-3xl flex-col items-center justify-center gap-8 px-6">
          {showSimple ? (
            <div className="w-full max-w-md">
              <p className="mb-3 text-center text-sm text-[color:var(--fg-muted)]">
                {t("loading")}
              </p>
              <div
                className="h-2 overflow-hidden rounded-full bg-[color:var(--surface)]"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress}
              >
                <div
                  className="h-full rounded-full bg-[color:var(--accent)] transition-[width] duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <p
                className="font-[family-name:var(--font-display)] text-7xl font-semibold tabular-nums tracking-tight text-[color:var(--fg)] sm:text-8xl"
                aria-live="polite"
                aria-atomic="true"
              >
                {Math.round(progress)}
                <span className="text-4xl text-[color:var(--accent)]">%</span>
              </p>
              <div
                className="w-full max-w-lg rounded-xl border border-[color:var(--border)] bg-[color:var(--bg-elevated)]/80 p-4 font-mono text-xs text-[color:var(--fg-muted)] shadow-lg backdrop-blur-sm"
                aria-live="polite"
              >
                {logLines.length === 0 ? (
                  <p>{t("waiting")}</p>
                ) : (
                  <ul className="space-y-1">
                    {logLines.map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
          <p className="text-center text-xs text-[color:var(--fg-muted)]">
            {t("hint")}
          </p>
        </div>
    </div>
  );
}

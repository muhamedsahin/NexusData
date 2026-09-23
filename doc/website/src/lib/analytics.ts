/**
 * Optional analytics adapter — no-op unless NEXT_PUBLIC_ANALYTICS_SRC is set.
 * Prefer cookieless providers (Plausible / Umami). Never invent tracking.
 */

export type AnalyticsProvider = "plausible" | "umami" | "none";

export function getAnalyticsConfig(): {
  provider: AnalyticsProvider;
  src: string | null;
} {
  const src = process.env.NEXT_PUBLIC_ANALYTICS_SRC?.trim() || null;
  const raw = (process.env.NEXT_PUBLIC_ANALYTICS_PROVIDER ?? "none").toLowerCase();
  const provider: AnalyticsProvider =
    raw === "plausible" || raw === "umami" ? raw : "none";
  if (!src || provider === "none") {
    return { provider: "none", src: null };
  }
  return { provider, src };
}

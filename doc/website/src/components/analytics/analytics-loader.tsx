"use client";

import Script from "next/script";

/**
 * Optional cookieless analytics. Renders nothing unless both env vars are set.
 * Set NEXT_PUBLIC_ANALYTICS_PROVIDER=plausible|umami and NEXT_PUBLIC_ANALYTICS_SRC.
 */
export function AnalyticsLoader() {
  const provider = (process.env.NEXT_PUBLIC_ANALYTICS_PROVIDER ?? "none").toLowerCase();
  const src = process.env.NEXT_PUBLIC_ANALYTICS_SRC?.trim();
  if (!src || (provider !== "plausible" && provider !== "umami")) {
    return null;
  }

  if (provider === "plausible") {
    const domain = process.env.NEXT_PUBLIC_SITE_URL
      ? new URL(process.env.NEXT_PUBLIC_SITE_URL).hostname
      : undefined;
    return (
      <Script
        defer
        data-domain={domain}
        src={src}
        strategy="afterInteractive"
      />
    );
  }

  return <Script defer src={src} strategy="afterInteractive" />;
}

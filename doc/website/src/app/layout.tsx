import type { ReactNode } from "react";

type RootLayoutProps = {
  children: ReactNode;
};

/**
 * Locale-aware html/body live in `app/[locale]/layout.tsx`.
 * Root layout only forwards children (next-intl App Router pattern).
 */
export default function RootLayout({ children }: RootLayoutProps) {
  return children;
}

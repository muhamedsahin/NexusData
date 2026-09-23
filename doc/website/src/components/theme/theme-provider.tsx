"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { siteConfig } from "@/config/site.config";

export type ThemeMode = "light" | "dark" | "system";

type ThemeContextValue = {
  theme: ThemeMode;
  resolved: "light" | "dark";
  setTheme: (theme: ThemeMode) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

function readCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function writeCookie(name: string, value: string) {
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${name}=${encodeURIComponent(value)}; Path=/; Max-Age=${maxAge}; SameSite=Lax`;
}

function parseTheme(value: string | null): ThemeMode {
  if (value === "light" || value === "dark" || value === "system") return value;
  return "system";
}

function subscribeSystemTheme(onChange: () => void) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}

function getSystemIsDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function getServerSystemIsDark() {
  return true;
}

function subscribeCookieTheme(onChange: () => void) {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

function getCookieThemeSnapshot(): ThemeMode {
  return parseTheme(readCookie(siteConfig.themeCookie));
}

function getServerCookieTheme(): ThemeMode {
  return "system";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const cookieTheme = useSyncExternalStore(
    subscribeCookieTheme,
    getCookieThemeSnapshot,
    getServerCookieTheme,
  );
  const [themeOverride, setThemeOverride] = useState<ThemeMode | null>(null);
  const theme = themeOverride ?? cookieTheme;
  const systemIsDark = useSyncExternalStore(
    subscribeSystemTheme,
    getSystemIsDark,
    getServerSystemIsDark,
  );
  const resolved: "light" | "dark" =
    theme === "light" ? "light" : theme === "dark" ? "dark" : systemIsDark ? "dark" : "light";

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = resolved;
    root.style.colorScheme = resolved;
  }, [resolved]);

  const setTheme = useCallback((next: ThemeMode) => {
    writeCookie(siteConfig.themeCookie, next);
    setThemeOverride(next);
  }, []);

  const value = useMemo(
    () => ({ theme, resolved, setTheme }),
    [theme, resolved, setTheme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return ctx;
}

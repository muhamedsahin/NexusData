"use client";

import { useTranslations } from "next-intl";
import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemeMode } from "@/components/theme/theme-provider";
import { cn } from "@/lib/cn";

const OPTIONS: { value: ThemeMode; icon: typeof Sun }[] = [
  { value: "light", icon: Sun },
  { value: "dark", icon: Moon },
  { value: "system", icon: Monitor },
];

type ThemeToggleProps = {
  className?: string;
};

export function ThemeToggle({ className }: ThemeToggleProps) {
  const t = useTranslations("theme");
  const { theme, setTheme } = useTheme();

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-lg border border-[color:var(--border)] bg-[color:var(--surface)] p-1",
        className,
      )}
      role="group"
      aria-label={t("label")}
    >
      {OPTIONS.map(({ value, icon: Icon }) => {
        const active = theme === value;
        return (
          <button
            key={value}
            type="button"
            className={cn(
              "rounded-md p-1.5 transition-colors duration-200 ease-[cubic-bezier(0.22,1,0.36,1)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]",
              active
                ? "bg-[color:var(--accent)] text-[color:var(--accent-fg)]"
                : "text-[color:var(--fg-muted)] hover:text-[color:var(--fg)]",
            )}
            aria-pressed={active}
            aria-label={t(value)}
            onClick={() => setTheme(value)}
          >
            <Icon className="h-3.5 w-3.5" aria-hidden />
          </button>
        );
      })}
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { Check, ChevronDown, ChevronUp, Copy, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CMAKE_SNIPPET } from "@/lib/webgl";
import { cn } from "@/lib/cn";

type InstallSnippetProps = {
  className?: string;
};

const VCPKG_SNIPPET = `vcpkg install matrixdata
# CMakeLists.txt:
find_package(matrixdata CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE matrixdata::matrixdata)`;

const CONAN_SNIPPET = `[requires]
matrixdata/1.0.0

[generators]
CMakeDeps
CMakeToolchain`;

export function InstallSnippet({ className }: InstallSnippetProps) {
  const t = useTranslations("home");
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [tab, setTab] = useState<"cmake" | "vcpkg" | "conan">("cmake");

  useEffect(() => {
    if (!copied) return;
    const id = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(id);
  }, [copied]);

  const activeSnippet =
    tab === "cmake"
      ? CMAKE_SNIPPET
      : tab === "vcpkg"
        ? VCPKG_SNIPPET
        : CONAN_SNIPPET;

  const quickPillText = "FetchContent_Declare(matrixdata v1.0.0)";

  async function onCopy(textToCopy: string) {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div
      className={cn(
        "relative w-full max-w-xl transition-all duration-300",
        className,
      )}
    >
      {/* Sleek Pill Header */}
      <div
        className={cn(
          "group flex items-center justify-between gap-2 rounded-xl border border-[color:var(--border)] bg-[color-mix(in_oklab,var(--bg-elevated)_90%,transparent)] px-3 py-2 backdrop-blur-md transition-colors duration-200 hover:border-[color:var(--accent)]",
          expanded && "rounded-b-none border-b-white/5",
        )}
      >
        <div
          role="button"
          tabIndex={0}
          onClick={() => setExpanded(!expanded)}
          onKeyDown={(e) => e.key === "Enter" && setExpanded(!expanded)}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 overflow-hidden text-left"
          title="Click to view full installation options"
        >
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
            <Terminal className="h-3.5 w-3.5" aria-hidden />
          </span>
          <span className="truncate font-mono text-xs text-[color:var(--fg)] sm:text-sm">
            <span className="text-[color:var(--fg-muted)]">$ </span>
            {quickPillText}
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={(e) => {
              e.stopPropagation();
              onCopy(activeSnippet);
            }}
            className="h-7 px-2 font-mono text-xs text-[#E8F1FF]/70 hover:bg-[#2DE2D0]/10 hover:text-[#2DE2D0]"
            title={copied ? t("copied") : t("copy")}
          >
            {copied ? (
              <span className="flex items-center gap-1 text-[#2DE2D0]">
                <Check className="h-3 w-3" aria-hidden />
                <span className="hidden sm:inline">{t("copied")}</span>
              </span>
            ) : (
              <span className="flex items-center gap-1">
                <Copy className="h-3 w-3" aria-hidden />
                <span className="hidden sm:inline">{t("copy")}</span>
              </span>
            )}
          </Button>

          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="flex h-7 w-7 items-center justify-center rounded-md text-[#E8F1FF]/50 transition-colors hover:bg-white/5 hover:text-white"
            aria-label={expanded ? "Collapse snippet" : "Expand snippet"}
          >
            {expanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded Multi-Tab Code Drawer */}
      {expanded ? (
        <div className="overflow-hidden rounded-b-xl border-x border-b border-white/10 bg-[#070b14]/95 shadow-[0_12px_32px_rgba(0,0,0,0.6)] backdrop-blur-xl">
          <div className="flex items-center justify-between border-b border-white/5 px-3 py-1.5 bg-black/30">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setTab("cmake")}
                className={cn(
                  "px-2.5 py-1 font-mono text-xs rounded transition-colors",
                  tab === "cmake"
                    ? "bg-[color:var(--accent-soft)] font-semibold text-[color:var(--accent)]"
                    : "text-[#E8F1FF]/60 hover:text-white",
                )}
              >
                CMake
              </button>
              <button
                type="button"
                onClick={() => setTab("vcpkg")}
                className={cn(
                  "px-2.5 py-1 font-mono text-xs rounded transition-colors",
                  tab === "vcpkg"
                    ? "bg-[color:var(--accent-soft)] font-semibold text-[color:var(--accent)]"
                    : "text-[#E8F1FF]/60 hover:text-white",
                )}
              >
                vcpkg
              </button>
              <button
                type="button"
                onClick={() => setTab("conan")}
                className={cn(
                  "px-2.5 py-1 font-mono text-xs rounded transition-colors",
                  tab === "conan"
                    ? "bg-[color:var(--accent-soft)] font-semibold text-[color:var(--accent)]"
                    : "text-[#E8F1FF]/60 hover:text-white",
                )}
              >
                Conan
              </button>
            </div>
            <span className="font-mono text-[10px] text-[#E8F1FF]/40 uppercase tracking-wider">
              C++20 / Header-Only
            </span>
          </div>

          <pre className="max-h-48 overflow-x-auto p-3 font-mono text-[0.72rem] leading-relaxed text-[#94a3b8] sm:text-xs">
            <code>{activeSnippet}</code>
          </pre>
        </div>
      ) : null}
    </div>
  );
}

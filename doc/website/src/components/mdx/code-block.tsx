"use client";

import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui/button";

type CodeBlockProps = {
  html: string;
  code: string;
  filename?: string;
  language?: string;
  badge?: string;
  className?: string;
};

export function CodeBlock({
  html,
  code,
  filename,
  language,
  badge,
  className,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const id = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(id);
  }, [copied]);

  return (
    <div
      className={cn(
        "group my-4 overflow-hidden rounded-xl border border-[color:var(--border)] bg-[#0d1117]",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          {filename ? (
            <span className="truncate font-mono text-xs text-white/70">
              {filename}
            </span>
          ) : (
            <span className="font-mono text-xs uppercase text-white/50">
              {language ?? "code"}
            </span>
          )}
          {badge ? (
            <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[0.65rem] text-violet-200">
              {badge}
            </span>
          ) : null}
        </div>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="text-white/70 hover:text-white"
          onClick={async () => {
            await navigator.clipboard.writeText(code);
            setCopied(true);
          }}
        >
          {copied ? (
            <Check className="h-3.5 w-3.5" aria-hidden />
          ) : (
            <Copy className="h-3.5 w-3.5" aria-hidden />
          )}
          <span className="sr-only">Copy</span>
        </Button>
      </div>
      <div
        className="overflow-x-auto p-4 text-sm [&_pre]:m-0 [&_pre]:bg-transparent! [&_code]:font-mono"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

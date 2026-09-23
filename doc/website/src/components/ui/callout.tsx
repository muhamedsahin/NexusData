import { cn } from "@/lib/cn";
import type { HTMLAttributes, ReactNode } from "react";

type CalloutTone = "note" | "tip" | "warning" | "danger";

const toneStyles: Record<CalloutTone, string> = {
  note: "border-[color:var(--callout-note-border)] bg-[color:var(--callout-note-bg)]",
  tip: "border-[color:var(--callout-tip-border)] bg-[color:var(--callout-tip-bg)]",
  warning:
    "border-[color:var(--callout-warn-border)] bg-[color:var(--callout-warn-bg)]",
  danger:
    "border-[color:var(--callout-danger-border)] bg-[color:var(--callout-danger-bg)]",
};

// Left accent bar + glow colors per tone, layered on top of the existing palette
const accentBar: Record<CalloutTone, string> = {
  note: "from-[#2DE2D0] to-[#22c5f0]",
  tip: "from-[#2DE2D0] to-[#5B4BDB]",
  warning: "from-[#f0b429] to-[#f0742f]",
  danger: "from-[#ef4444] to-[#b91c1c]",
};

const glow: Record<CalloutTone, string> = {
  note: "shadow-[0_8px_28px_-12px_rgba(45,226,208,0.35)]",
  tip: "shadow-[0_8px_28px_-12px_rgba(91,75,219,0.4)]",
  warning: "shadow-[0_8px_28px_-12px_rgba(240,180,41,0.35)]",
  danger: "shadow-[0_8px_28px_-12px_rgba(239,68,68,0.35)]",
};

type CalloutProps = HTMLAttributes<HTMLElement> & {
  tone?: CalloutTone;
  title?: string;
  children: ReactNode;
};

export function Callout({
  tone = "note",
  title,
  children,
  className,
  ...props
}: CalloutProps) {
  return (
    <aside
      className={cn(
        "relative overflow-hidden rounded-xl border px-4 py-3 pl-5 text-sm leading-relaxed text-[color:var(--fg)] backdrop-blur-md",
        toneStyles[tone],
        glow[tone],
        className,
      )}
      {...props}
    >
      <span
        aria-hidden
        className={cn(
          "absolute inset-y-0 left-0 w-1 bg-gradient-to-b",
          accentBar[tone],
        )}
      />
      {title ? (
        <p className="mb-1 font-semibold tracking-wide text-[color:var(--fg)]">
          {title}
        </p>
      ) : null}
      <div className="text-[color:var(--fg-muted)]">{children}</div>
    </aside>
  );
}
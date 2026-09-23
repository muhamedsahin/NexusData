import { cn } from "@/lib/cn";
import type { HTMLAttributes } from "react";

type KbdProps = HTMLAttributes<HTMLElement>;

export function Kbd({ className, children, ...props }: KbdProps) {
  return (
    <kbd
      className={cn(
        "inline-flex min-w-[1.5rem] items-center justify-center rounded-md border border-[color:var(--border)] bg-[color:var(--surface)] px-1.5 py-0.5 font-mono text-[0.7rem] text-[color:var(--fg-muted)] shadow-[inset_0_-1px_0_color-mix(in_oklab,var(--fg)_12%,transparent)]",
        className,
      )}
      {...props}
    >
      {children}
    </kbd>
  );
}

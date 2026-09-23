import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

const buttonVariants = cva(
  "relative inline-flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-[transform,background-color,box-shadow,color,filter] duration-200 ease-[cubic-bezier(0.22,1,0.36,1)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--bg)] disabled:pointer-events-none disabled:opacity-50 active:scale-[0.97]",
  {
    variants: {
      variant: {
        // Gradient teal -> violet, real glow, subtle lift on hover instead of a flat brightness bump
        primary:
          "bg-[linear-gradient(135deg,#2DE2D0_0%,#22c5f0_45%,#5B4BDB_100%)] text-[#04060B] font-semibold shadow-[0_0_0_1px_rgba(45,226,208,0.4),0_8px_24px_-8px_rgba(45,226,208,0.55)] hover:-translate-y-0.5 hover:shadow-[0_0_0_1px_rgba(45,226,208,0.6),0_12px_32px_-6px_rgba(91,75,219,0.65)] hover:brightness-105",
        secondary:
          "bg-[color:var(--surface)] text-[color:var(--fg)] border border-[color:var(--border)] shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset] hover:-translate-y-0.5 hover:border-[#2DE2D0]/50 hover:bg-[color:var(--surface-2)] hover:shadow-[0_8px_20px_-10px_rgba(45,226,208,0.35)]",
        ghost:
          "bg-transparent text-[color:var(--fg-muted)] hover:text-[color:var(--fg)] hover:bg-[color:var(--surface)]",
        outline:
          "border border-[color:var(--border-strong)] bg-transparent text-[color:var(--fg)] hover:-translate-y-0.5 hover:border-[#2DE2D0] hover:text-[#2DE2D0] hover:shadow-[0_8px_20px_-10px_rgba(45,226,208,0.4)]",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-10 px-4",
        lg: "h-12 px-6 text-base",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  },
);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
    children: ReactNode;
  };

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { className, variant, size, type = "button", children, ...props },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      >
        {children}
      </button>
    );
  },
);

export { buttonVariants };
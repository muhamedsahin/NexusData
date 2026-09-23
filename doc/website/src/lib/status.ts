import { cva, type VariantProps } from "class-variance-authority";

export const statusBadgeVariants = cva(
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium tracking-wide",
  {
    variants: {
      status: {
        stable:
          "border-[color:var(--status-stable-border)] bg-[color:var(--status-stable-bg)] text-[color:var(--status-stable)]",
        beta: "border-[color:var(--status-beta-border)] bg-[color:var(--status-beta-bg)] text-[color:var(--status-beta)]",
        "in-development":
          "border-[color:var(--status-dev-border)] bg-[color:var(--status-dev-bg)] text-[color:var(--status-dev)]",
        planned:
          "border-[color:var(--status-planned-border)] bg-[color:var(--status-planned-bg)] text-[color:var(--status-planned)]",
        illustrative:
          "border-[color:var(--status-warn-border)] bg-[color:var(--status-warn-bg)] text-[color:var(--status-warn)]",
        "design-preview":
          "border-[color:var(--status-preview-border)] bg-[color:var(--status-preview-bg)] text-[color:var(--status-preview)]",
      },
    },
    defaultVariants: {
      status: "planned",
    },
  },
);

export type StatusBadgeVariant = NonNullable<
  VariantProps<typeof statusBadgeVariants>["status"]
>;

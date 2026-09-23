import { cn } from "@/lib/cn";
import {
  statusBadgeVariants,
  type StatusBadgeVariant,
} from "@/lib/status";

const LABEL_KEYS: Record<StatusBadgeVariant, string> = {
  stable: "stable",
  beta: "beta",
  "in-development": "in-development",
  planned: "planned",
  illustrative: "illustrative",
  "design-preview": "designPreview",
};

type StatusBadgeProps = {
  status: StatusBadgeVariant;
  label: string;
  className?: string;
};

export function StatusBadge({ status, label, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(statusBadgeVariants({ status }), className)}
      data-status={status}
      data-status-key={LABEL_KEYS[status]}
    >
      {label}
    </span>
  );
}

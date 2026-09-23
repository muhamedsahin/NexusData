"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";

const STORAGE_KEY = "nexusdata-docs-feedback";

export function DocsFeedbackWidget() {
  const t = useTranslations("docs");
  const [sent, setSent] = useState(false);

  function vote(value: "yes" | "no") {
    try {
      const existing = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as unknown[];
      existing.push({
        value,
        path: window.location.pathname,
        at: new Date().toISOString(),
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(existing.slice(-100)));
    } catch {
      /* ignore */
    }
    setSent(true);
  }

  if (sent) {
    return (
      <p className="text-sm text-[color:var(--fg-muted)]">{t("feedbackThanks")}</p>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--fg-muted)]">
      <span>{t("feedbackAsk")}</span>
      <Button type="button" size="sm" variant="secondary" onClick={() => vote("yes")}>
        {t("feedbackYes")}
      </Button>
      <Button type="button" size="sm" variant="ghost" onClick={() => vote("no")}>
        {t("feedbackNo")}
      </Button>
    </div>
  );
}

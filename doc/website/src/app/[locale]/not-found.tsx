import { getTranslations } from "next-intl/server";
import { Link } from "@/i18n/navigation";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/cn";

type NotFoundProps = {
  params?: Promise<{ locale?: string }>;
};

export default async function LocaleNotFound({ params }: NotFoundProps) {
  const resolved = params ? await params : undefined;
  const locale = resolved?.locale === "tr" ? "tr" : "en";
  const t = await getTranslations({ locale, namespace: "notFound" });

  return (
    <div className="mx-auto flex max-w-lg flex-col items-start gap-4 px-4 py-24">
      <p className="font-mono text-sm text-[color:var(--accent)]">404</p>
      <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold">
        {t("title")}
      </h1>
      <p className="text-[color:var(--fg-muted)]">{t("body")}</p>
      <Link href="/" className={cn(buttonVariants({ variant: "primary" }))}>
        {t("home")}
      </Link>
    </div>
  );
}

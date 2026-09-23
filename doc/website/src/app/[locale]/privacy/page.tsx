import { getTranslations, setRequestLocale } from "next-intl/server";
import { Callout } from "@/components/ui/callout";

type PrivacyPageProps = {
  params: Promise<{ locale: string }>;
};

export default async function PrivacyPage({ params }: PrivacyPageProps) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations();
  const isTr = locale === "tr";

  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="font-[family-name:var(--font-display)] text-3xl font-semibold tracking-tight">
        {isTr ? "Gizlilik notu" : "Privacy note"}
      </h1>
      <div className="mt-6 space-y-4 text-[color:var(--fg-muted)]">
        <p>
          {isTr
            ? "Ülke/IP bilgisi (ör. x-vercel-ip-country, cf-ipcountry) yalnızca anlık dil seçimi için okunur. Saklanmaz, loglanmaz ve analitiğe gönderilmez."
            : "Country/IP headers (e.g. x-vercel-ip-country, cf-ipcountry) are read only for one-shot language selection. They are never stored, logged, or sent to analytics."}
        </p>
        <Callout tone="note">{t("home.privacyNote")}</Callout>
        <p>
          {isTr
            ? "İşlevsel çerezler: NEXT_LOCALE (dil tercihi), NEXT_THEME (tema). Analitik varsayılan olarak kapalıdır."
            : "Functional cookies only: NEXT_LOCALE (language preference), NEXT_THEME (theme). Analytics is off by default."}
        </p>
      </div>
    </div>
  );
}

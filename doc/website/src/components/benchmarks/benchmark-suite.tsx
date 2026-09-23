import { getLocale } from "next-intl/server";
import { loadBenchmarkSuite } from "@/lib/benchmarks/load";
import { BenchmarkCard } from "@/components/benchmarks/benchmark-card";
import { Callout } from "@/components/ui/callout";
import type { SiteLocale } from "@/config/site.config";

/** Server component: loads Zod-validated benchmark JSON for the docs page. */
export async function BenchmarkSuite() {
  const locale = ((await getLocale()) === "tr" ? "tr" : "en") as SiteLocale;
  const all = loadBenchmarkSuite();

  if (all.length === 0) {
    return (
      <Callout tone="note" title={locale === "tr" ? "Veri yok" : "No data"}>
        {locale === "tr"
          ? "data/benchmarks/ altında JSON yok. Ölçüm ekleyene kadar grafik gösterilemez."
          : "No JSON under data/benchmarks/. Charts appear after you add measured files."}
      </Callout>
    );
  }

  const allIllustrative = all.every((e) => e.illustrative);

  return (
    <div className="my-6">
      {allIllustrative ? (
        <Callout
          tone="warning"
          className="mb-6"
          title={locale === "tr" ? "Tüm seriler örnek" : "All series are illustrative"}
        >
          {locale === "tr"
            ? "Bu sayfadaki rakamlar henüz ölçülmedi. Kendi bench_suite çıktınızı Zod şemasına uygun JSON olarak ekleyin."
            : "Numbers on this page are not measured yet. Add your own bench_suite JSON matching the Zod schema."}
        </Callout>
      ) : null}
      {all.map((entry) => (
        <BenchmarkCard key={entry.id} entry={entry} locale={locale} />
      ))}
    </div>
  );
}

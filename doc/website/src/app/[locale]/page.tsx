import { getTranslations, setRequestLocale } from "next-intl/server";
import { HomeExperience } from "@/components/home/home-experience";
import { resolvePublicLink, siteConfig } from "@/config/site.config";

type HomePageProps = {
  params: Promise<{ locale: string }>;
};

export default async function HomePage({ params }: HomePageProps) {
  const { locale } = await params;
  setRequestLocale(locale);
  await getTranslations();
  const github = resolvePublicLink(siteConfig.githubUrl);

  return <HomeExperience locale={locale} github={github} />;
}

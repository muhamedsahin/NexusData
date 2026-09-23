import { ImageResponse } from "next/og";
import { siteConfig } from "@/config/site.config";

export const runtime = "edge";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const title = searchParams.get("title") ?? siteConfig.name;
  const locale = searchParams.get("locale") === "tr" ? "tr" : "en";

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "#05070d",
          padding: 64,
          color: "#e8eefc",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ fontSize: 28, color: "#2dd4bf" }}>{siteConfig.name}</div>
        <div style={{ fontSize: 56, fontWeight: 700, lineHeight: 1.15, maxWidth: 900 }}>
          {title}
        </div>
        <div style={{ fontSize: 22, color: "#9aa6bf" }}>
          {locale === "tr" ? "Dokümantasyon" : "Documentation"} · {locale.toUpperCase()}
        </div>
      </div>
    ),
    { width: 1200, height: 630 },
  );
}

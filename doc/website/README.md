# NexusData documentation website

Bilingual (EN/TR) marketing + docs site for the **NexusData** C++20 data-loading library.

## Phase 7 (current)

Polish: security headers/CSP, mobile nav, optional analytics hook, CI, CONTRIBUTING/DEPLOY, axe + internal-link checks. Lighthouse scores are **not** claimed until recorded in `MEASUREMENTS.md`.

## Stack decisions

| Choice | Why |
| --- | --- |
| **Next.js 16 App Router** + **next-intl 4** | First-party App Router i18n, `localePrefix: "always"`, typed navigation |
| **R3F 9 + three 0.186 + drei + postprocessing** | Homepage-only via `next/dynamic` (`ssr: false`); docs must not import three |
| **Lenis** | Smooth scroll writes `useScrollStore` (single progress source for Phase 2) |
| **GSAP** | Loader morph cube→grid only |
| **zustand** | Loader / quality / pointer / scroll stores |
| **Sora** (display) + **Geist** (body/mono) via `next/font` | Characterful headings, readable UI, Latin + Latin-ext for Turkish glyphs |
| **Tailwind CSS 4** + CSS variables | Design tokens without a heavy component kit |
| **Custom MDX** + Orama + Shiki | Docs engine without fumadocs-ui |
| **Playwright** (+ axe) | Locale, three isolation, a11y smoke |
| **Vitest** | Geo, RNG, benchmarks schema, internal docs links |

## Commands

```bash
npm install
npm run dev
npm run build
npm run start
npm run lint
npm run typecheck
npm run test:unit
npm run build && npx playwright install chromium && npm run test:e2e
```

See also: [CONTRIBUTING.md](./CONTRIBUTING.md), [DEPLOY.md](./DEPLOY.md), [MEASUREMENTS.md](./MEASUREMENTS.md), `.env.example`.

### Dev country override

Only when `NODE_ENV=development`:

- Query: `http://localhost:3001/?__country=TR`
- Or env: `DEV_FORCE_COUNTRY=TR`

## Locale decision order (`/`)

1. Explicit `/en` or `/tr` prefix → keep
2. `NEXT_LOCALE` cookie → user choice wins
3. Country header (`x-vercel-ip-country` / `cf-ipcountry` / `x-country-code`): `TR` → `tr`, else → `en` (skipped for search bots)
4. `Accept-Language`
5. Default `en`

Redirects use **307** + `Vary` headers.

## Placeholders

Edit `src/config/site.config.ts`. Values starting with `PLACEHOLDER_` hide GitHub/author/social UI. Library license is **Apache-2.0**.

## Privacy

Country/IP headers are used only for one-shot language selection. Not stored, logged, or sent to analytics. Optional Plausible/Umami is **off** unless env vars are set. See `/[locale]/privacy`.

## Security headers

`next.config.ts` sets `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and a baseline CSP (inline scripts allowed for theme bootstrap + Next hydration).

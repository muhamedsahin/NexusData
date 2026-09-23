# Deploy guide — NexusData docs site

## Build artifact

```bash
cd doc/website
npm ci
npm run build
npm run start   # Node server on PORT (default 3000)
```

Or export static-friendly hosting via your platform’s Next.js adapter (Vercel, Node, Docker).

## Environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_SITE_URL` | Recommended in prod | Canonical URL for sitemap / OG / metadataBase |
| `DEV_FORCE_COUNTRY` | Dev only | Force geo locale (`TR` → `tr`) |
| `NEXT_PUBLIC_ANALYTICS_PROVIDER` | Optional | `plausible` \| `umami` — default off |
| `NEXT_PUBLIC_ANALYTICS_SRC` | Optional | Script URL for the provider |

See `.env.example`.

## Platform notes

### Vercel

- Root directory: `doc/website`
- Framework preset: Next.js
- Country headers (`x-vercel-ip-country`) drive first-visit locale when no cookie

### Docker (sketch)

```dockerfile
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_PUBLIC_SITE_URL=https://example.com
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/content ./content
COPY --from=build /app/data ./data
COPY --from=build /app/messages ./messages
EXPOSE 3001
CMD ["npm", "run", "start"]
```

Adjust copy set if your Next standalone output is enabled later.

## Post-deploy checks

1. `/en` and `/tr` resolve; `/` 307 → locale
2. `/en/docs` loads without `three` network requests
3. `/en/docs/performance/benchmarks` shows Illustrative badges until real JSON exists
4. Security headers present (`X-Content-Type-Options`, `CSP`, `X-Frame-Options`)

## Measurements

Do not publish Lighthouse scores without running Lighthouse yourself. See `MEASUREMENTS.md`.

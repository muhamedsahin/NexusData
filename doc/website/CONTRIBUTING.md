# Contributing to the NexusData docs site

## Prerequisites

- Node.js 22+
- npm

## Setup

```bash
cd doc/website
cp .env.example .env.local   # optional
npm install
npm run dev
```

Open `http://localhost:3001/en` (or `/tr`).

## Workflows

| Task | Command |
| --- | --- |
| Lint | `npm run lint` |
| Types | `npm run typecheck` |
| Unit | `npm run test:unit` |
| E2E | `npm run test:e2e` (builds first) |
| Format | `npm run format` |

## Content

- Docs live in `content/docs/{en,tr}/**/*.mdx`
- Frontmatter is Zod-validated (`src/lib/docs/schema.ts`)
- Internal links use `/docs/...` paths
- Keep EN and TR in sync for the same slug when possible

### Honesty rules

- Do **not** invent benchmark numbers — only `data/benchmarks/*.json`
- Set `illustrative: true` until measurements include hardware + date + version
- C++ snippets must match public headers or carry a Design preview badge
- Leave `PLACEHOLDER_*` site.config fields empty rather than inventing people/URLs

## Benchmarks

1. Run library `bench_suite` on your machine
2. Add Zod-valid JSON under `data/benchmarks/`
3. Set `illustrative: false` only for real runs

## Interactive widgets

Registered in `src/components/mdx/mdx-components.tsx`. Docs routes must not import `three` / R3F.

## Pull requests

- Keep PRs focused
- CI runs lint, typecheck, unit, build, Playwright (`/.github/workflows/website.yml`)

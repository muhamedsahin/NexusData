# Measurements (honesty log)

Do **not** invent Lighthouse, FPS, or bundle claims. Fill this file after running tools locally or in CI.

## How to measure

```bash
cd doc/website
npm run build && npm run start
# separate terminal:
npx lighthouse http://127.0.0.1:3001/en/docs --only-categories=performance,accessibility,seo --quiet --chrome-flags="--headless"
npx lighthouse http://127.0.0.1:3001/en --only-categories=performance,accessibility,seo --quiet --chrome-flags="--headless"
```

Bundle: inspect `.next` build output / Next route sizes after `npm run build`. Docs routes must not ship `three` (enforced by Playwright).

## Recorded runs

| Date | URL | Perf | A11y | SEO | Notes |
| --- | --- | --- | --- | --- | --- |
| _(none yet)_ | | | | | Run Lighthouse and paste scores here |

## Targets (aspirational — not claimed)

| Surface | Perf | A11y | SEO |
| --- | --- | --- | --- |
| Docs | ≥ 95 | ≥ 95 | ≥ 95 |
| Home (3D) | ≥ 80 | ≥ 95 | ≥ 95 |

#!/usr/bin/env node
console.log(`
Lighthouse is not run automatically (honesty: no invented scores).

1. npm run build && npm run start
2. npx lighthouse http://127.0.0.1:3001/en/docs --only-categories=performance,accessibility,seo
3. Record results in MEASUREMENTS.md

See MEASUREMENTS.md for targets and the results table.
`);

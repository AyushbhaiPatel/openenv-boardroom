# OpenBoardroom

A responsive reminder-app frontend inspired by the public [NudgeBell landing page](https://nudgebell.app/), branded as OpenBoardroom. The website now lives at the repository root; the previous Python/OpenEnv Boardroom application has been removed.

## Run locally

```sh
npm install
npm run dev
```

Open the local address printed by Vite. `npm run build` produces a static site in `dist/`, and `npm run preview` serves that production build.

## Included

- The full responsive landing page, including pricing, testimonials from the original inspiration, escalation chain, and FAQs.
- Interactive channel previews, monthly/yearly billing, command copying, and accessible FAQ accordions.
- A clearly labeled local reminder demo opened by the sign-in, get-started, and plan buttons. It stores reminders only in this browser and supports acknowledgment and deletion.
- Locally stored fonts, styles, and icons; rendering doesn't depend on the reference site's servers.

This is a frontend demo, not an operational reminder service. There is no authentication, billing, agent connection, or notification delivery. The example agent command requires your own backend before it can work. Documentation, legal, blog, and testimonial links still point to the original reference and are not OpenBoardroom policies or endorsements.

The reference site's assets remain the property of their respective owners. Obtain permission or replace them before publishing this as your own product.

## Verification

```sh
npx playwright install chromium
npm test
npm run build
```

Tests cover desktop and mobile layouts and interactions. They can also use an existing browser via `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`; on macOS, installed Brave and Google Chrome browsers are detected automatically.

## Docker

```sh
docker build -t openboardroom .
docker run --rm -p 8080:80 openboardroom
```

The container serves the static production build using Nginx. CI checks frontend tests, the production build, and the Docker image.

import { test, expect } from "@playwright/test";

test("renders the full page with local assets and no horizontal overflow", async ({
  page,
}) => {
  const errors = [];
  const failures = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("requestfailed", (request) => failures.push(request.url()));
  const externalRequests = [];
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:5173"))
      externalRequests.push(request.url());
  });
  await page.goto("/");
  await expect(page).toHaveTitle("OpenBoardroom — Escalating reminders");
  await expect(page.locator("header > div > a")).toHaveText("OpenBoardroom");
  await expect(page.locator("h1")).toContainText("Stops when you acknowledge.");
  await expect(page.locator("main > section")).toHaveCount(10);
  await expect(page.locator("#faq")).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
  expect(failures).toEqual([]);
  expect(externalRequests).toEqual([]);
  if (process.env.SCREENSHOT_DIR) {
    await page.screenshot({
      path: `${process.env.SCREENSHOT_DIR}/${test.info().project.name}.png`,
      fullPage: true,
    });
  }
});

test("switches billing periods and uses the selected plan in the demo", async ({
  page,
}) => {
  await page.goto("/");
  const pricing = page.locator("#pricing");
  await pricing.getByRole("button", { name: "Monthly", exact: true }).click();
  await expect(pricing.locator(".items-baseline span").nth(0)).toHaveText("$5");
  await expect(pricing.locator(".items-baseline span").nth(2)).toHaveText(
    "$15",
  );
  await expect(pricing.locator(".items-baseline span").nth(4)).toHaveText(
    "$49",
  );
  await pricing.getByRole("link", { name: "Choose plan" }).nth(1).click();
  await expect(page.locator("#selected-plan")).toHaveText(
    "Pro · Monthly — demo only, no purchase",
  );
  await page.keyboard.press("Escape");
  await pricing.getByRole("switch").click();
  await expect(pricing.locator(".items-baseline span").nth(0)).toHaveText("$4");
  await expect(pricing.getByText("$144/yr billed annually")).toBeVisible();
});

test("plan dialog uses opaque themed surfaces and stays open when its padding is clicked", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .locator("#pricing")
    .getByRole("link", { name: "Choose plan" })
    .nth(1)
    .click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveCSS("background-color", "rgb(249, 246, 241)");
  await expect(dialog).toHaveCSS("color", "rgb(22, 16, 10)");
  await expect(page.locator("#reminder-title")).toHaveCSS(
    "background-color",
    "rgb(249, 246, 241)",
  );
  await expect(page.locator(".demo-submit")).toHaveCSS(
    "background-color",
    "rgb(22, 16, 10)",
  );
  await expect(page.locator(".demo-submit")).toHaveCSS(
    "color",
    "rgb(249, 246, 241)",
  );
  await dialog.click({ position: { x: 8, y: 8 } });
  await expect(dialog).toBeVisible();
  await page.getByRole("button", { name: "Close reminder demo" }).click();
  await expect(dialog).not.toBeVisible();
});

test("opens and closes accessible FAQ answers", async ({ page }) => {
  await page.goto("/");
  const first = page.getByRole("button", {
    name: "Do I need to download an app?",
  });
  await first.click();
  await expect(first).toHaveAttribute("aria-expanded", "true");
  await expect(
    page.getByText("No. OpenBoardroom is a web app.", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Can I set recurring reminders?" })
    .click();
  await expect(first).toHaveAttribute("aria-expanded", "false");
  await expect(
    page.getByText("Yes. Daily, weekly, monthly, and yearly recurrence", {
      exact: false,
    }),
  ).toBeVisible();
});

test("channel previews and reminder creation, persistence, acknowledgment and deletion work", async ({
  page,
}, testInfo) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Preview WhatsApp" }).click();
  await expect(
    page.getByRole("button", { name: "Preview WhatsApp" }),
  ).toHaveAttribute("aria-pressed", "true");
  if (testInfo.project.name === "desktop") {
    await expect(
      page.getByText("Mom’s birthday tomorrow. Tap to acknowledge.", {
        exact: true,
      }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Preview Phone call" }).click();
    await page
      .getByRole("button", { name: "Acknowledge reminder", exact: true })
      .click();
    await expect(
      page.getByText("Acknowledged. No further nudges.", { exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Preview WhatsApp" }).click();
  }
  await page
    .locator("header")
    .getByRole("link", { name: "Get started →" })
    .click();
  await expect(page.locator("#reminder-channel")).toHaveValue("WhatsApp");
  await page
    .getByLabel("What do you want to remember?")
    .fill("Renew domain <script>alert(1)</script>");
  await page.getByLabel("When", { exact: true }).fill("2027-01-01T09:00");
  await page.getByRole("button", { name: "Create demo reminder →" }).click();
  await expect(page.locator(".reminder-card h3")).toHaveText(
    "Renew domain <script>alert(1)</script>",
  );
  await page.reload();
  await page.locator("header").getByRole("link", { name: "Sign in" }).click();
  await expect(page.locator(".reminder-card")).toHaveCount(1);
  await page.getByRole("button", { name: "Acknowledge", exact: true }).click();
  await expect(page.locator(".reminder-card p")).toContainText("Acknowledged");
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.locator(".reminder-card")).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
});

test("copies the agent command", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await page.getByRole("button", { name: "Copy command" }).click();
  await expect(page.locator("#toast")).toHaveText(
    "Command copied to clipboard",
  );
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(
    "claude mcp add --transport http openboardroom",
  );
});

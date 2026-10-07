const toast = document.querySelector("#toast");
let toastTimer;

document.querySelectorAll("[data-phone-date]").forEach((date) => {
  date.textContent = new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  }).format(new Date());
});

function notify(message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, 3500);
}

document
  .querySelectorAll('[data-slot="accordion-trigger"]')
  .forEach((button) => {
    button.addEventListener("click", () => {
      const expanded = button.getAttribute("aria-expanded") !== "true";
      document
        .querySelectorAll('[data-slot="accordion-trigger"]')
        .forEach((trigger) => {
          const open = trigger === button && expanded;
          trigger.setAttribute("aria-expanded", String(open));
          trigger.dataset.state = open ? "open" : "closed";
          const content = document.getElementById(
            trigger.getAttribute("aria-controls"),
          );
          content.hidden = !open;
          content.dataset.state = trigger.dataset.state;
        });
    });
  });

document
  .querySelector('[aria-label="Copy command"]')
  .addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(
        document.querySelector("#agents pre").textContent.trim(),
      );
      notify("Command copied to clipboard");
    } catch {
      notify(
        "Clipboard access is unavailable. Select the command to copy it manually.",
      );
    }
  });

const pricing = document.querySelector("#pricing");
const billingButtons = [...pricing.querySelectorAll("button")];
const planCards = [...pricing.querySelectorAll("h3")].map(
  (heading) => heading.parentElement,
);
const monthlyPrices = [5, 15, 49];
const yearlyPrices = [4, 12, 39];
const yearlyTotals = [48, 144, 468];
let yearly = true;

function setBilling(value) {
  yearly = value;
  billingButtons[0].setAttribute("aria-pressed", String(!yearly));
  billingButtons[2].setAttribute("aria-pressed", String(yearly));
  billingButtons[0].classList.toggle("text-nb-ink", !yearly);
  billingButtons[0].classList.toggle("text-nb-ink-mute", yearly);
  billingButtons[2].classList.toggle("text-nb-ink", yearly);
  billingButtons[2].classList.toggle("text-nb-ink-mute", !yearly);
  billingButtons[1].setAttribute("aria-checked", String(yearly));
  billingButtons[1].firstElementChild.style.transform = `translateX(${yearly ? 28 : 4}px)`;
  planCards.forEach((card, index) => {
    card.querySelector(".items-baseline span").textContent =
      `$${yearly ? yearlyPrices[index] : monthlyPrices[index]}`;
    const caption = card.querySelector(".items-baseline + p");
    caption.textContent = yearly
      ? `$${yearlyTotals[index]}/yr billed annually`
      : "Billed monthly";
  });
}

billingButtons[1].setAttribute("role", "switch");
billingButtons[0].addEventListener("click", () => setBilling(false));
billingButtons[1].addEventListener("click", () => setBilling(!yearly));
billingButtons[2].addEventListener("click", () => setBilling(true));
setBilling(true);

const hero = document.querySelector("main > section");
const channelGrid = hero.querySelector(".grid.grid-cols-2");
const channels = ["Phone call", "WhatsApp", "SMS", "Email"];
const channelColors = ["#dca45e", "#36996b", "#5ca358", "#3987c4"];
const channelMessages = [
  "Incoming reminder call",
  "Mom’s birthday tomorrow. Tap to acknowledge.",
  "Mom’s birthday tomorrow. Reply OK to acknowledge.",
  "Mom’s birthday tomorrow. Don’t forget to call.",
];
const phone = hero.querySelector(".hidden.lg\\:flex > div > div");
const phoneContent = phone.children[2];
phoneContent.setAttribute("aria-live", "polite");
let selectedChannel = 0;

const channelButtons = [...channelGrid.children].map((card, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = card.className + " channel-button";
  button.innerHTML = card.innerHTML;
  button.setAttribute("aria-label", `Preview ${channels[index]}`);
  button.setAttribute("aria-pressed", String(index === 0));
  card.replaceWith(button);
  button.addEventListener("click", () => selectChannel(index));
  return button;
});

function selectChannel(index) {
  selectedChannel = index;
  channelButtons.forEach((button, position) => {
    const active = position === index;
    button.setAttribute("aria-pressed", String(active));
    button.className = `p-3 border flex flex-col gap-2 channel-button ${active ? "bg-nb-ink border-nb-ink" : "bg-nb-paper border-nb-rule"}`;
    button.querySelector("svg").style.color = active
      ? "var(--color-nb-amber)"
      : "var(--color-nb-ink)";
    button.querySelectorAll("p")[0].style.color = active
      ? "var(--color-nb-paper)"
      : "var(--color-nb-ink)";
    button.querySelectorAll("p")[1].style.color = active
      ? "#ffffff99"
      : "var(--color-nb-ink-soft)";
  });
  phoneContent.replaceChildren();
  const time = document.createElement("div");
  time.className = "preview-time";
  time.textContent = "9:12";
  const notification = document.createElement("div");
  notification.className = `preview-notification ${index === 0 ? "preview-call" : ""}`;
  const icon = channelButtons[index].querySelector("svg").cloneNode(true);
  icon.style.color = channelColors[index];
  const app = document.createElement("p");
  app.className = "preview-app";
  app.textContent = channels[index];
  const title = document.createElement("h3");
  title.textContent = "OpenBoardroom";
  const body = document.createElement("p");
  body.textContent = channelMessages[index];
  notification.append(icon, app, title, body);
  if (index === 0) {
    const answer = document.createElement("button");
    answer.className = "preview-answer";
    answer.textContent = "Acknowledge reminder";
    answer.addEventListener("click", () => {
      body.textContent = "Acknowledged. No further nudges.";
      answer.hidden = true;
    });
    notification.append(answer);
  }
  phoneContent.append(time, notification);
}

const dialog = document.querySelector("#reminder-dialog");
const form = document.querySelector("#reminder-form");
const reminderList = document.querySelector("#reminders");
const storageKey = "openboardroom-reminders";
let reminders = [];
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
  if (Array.isArray(saved)) {
    reminders = saved.filter(
      (item) =>
        item &&
        typeof item.id === "string" &&
        typeof item.title === "string" &&
        typeof item.channel === "string" &&
        Number.isFinite(Date.parse(item.date)),
    );
  }
} catch {
  reminders = [];
}

function saveReminders() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(reminders));
  } catch {
    notify(
      "Browser storage is unavailable. This demo will last only for this visit.",
    );
  }
  renderReminders();
}

function renderReminders() {
  reminderList.replaceChildren();
  for (const reminder of reminders) {
    const card = document.createElement("article");
    card.className = "reminder-card";
    const heading = document.createElement("h3");
    heading.textContent = reminder.title;
    const details = document.createElement("p");
    details.textContent = `${reminder.channel} · ${new Date(reminder.date).toLocaleString()} · ${reminder.acknowledged ? "Acknowledged" : "Demo only"}`;
    const actions = document.createElement("div");
    if (!reminder.acknowledged) {
      const acknowledge = document.createElement("button");
      acknowledge.type = "button";
      acknowledge.textContent = "Acknowledge";
      acknowledge.addEventListener("click", () => {
        reminder.acknowledged = true;
        saveReminders();
      });
      actions.append(acknowledge);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => {
      reminders = reminders.filter((item) => item.id !== reminder.id);
      saveReminders();
    });
    actions.append(remove);
    card.append(heading, details, actions);
    reminderList.append(card);
  }
}

document.querySelectorAll('a[href^="/sign-in"]').forEach((link) => {
  link.setAttribute("href", "#demo");
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const planName = link.closest("#pricing")
      ? link.parentElement.querySelector("h3").textContent
      : null;
    const selected = document.querySelector("#selected-plan");
    selected.hidden = !planName;
    selected.textContent = planName
      ? `${planName} · ${yearly ? "Yearly" : "Monthly"} — demo only, no purchase`
      : "";
    document.querySelector("#reminder-channel").value =
      channels[selectedChannel];
    const tomorrow = new Date(Date.now() + 86400000);
    tomorrow.setMinutes(tomorrow.getMinutes() - tomorrow.getTimezoneOffset());
    document.querySelector("#reminder-date").value = tomorrow
      .toISOString()
      .slice(0, 16);
    renderReminders();
    dialog.showModal();
  });
});

document
  .querySelector("#close-demo")
  .addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (
    event.clientX < bounds.left ||
    event.clientX > bounds.right ||
    event.clientY < bounds.top ||
    event.clientY > bounds.bottom
  )
    dialog.close();
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const data = new FormData(form);
  const title = data.get("title").trim();
  if (!title) {
    notify("Add a reminder title.");
    return;
  }
  reminders.unshift({
    id: crypto.randomUUID(),
    title,
    date: data.get("date"),
    channel: data.get("channel"),
    acknowledged: false,
  });
  saveReminders();
  form.reset();
  notify("Demo reminder saved in this browser");
});

document
  .querySelectorAll('a[href="https://nudgebell.app/pricing#faq"]')
  .forEach((link) => {
    link.href = "#faq";
  });
document
  .querySelectorAll('a[href="https://nudgebell.app/pricing"]')
  .forEach((link) => {
    link.href = "#pricing";
  });
document.querySelectorAll('a[href^="https://"]').forEach((link) => {
  link.target = "_blank";
  link.rel = "noopener noreferrer";
});

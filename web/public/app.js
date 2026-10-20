const views = {
  intake: document.getElementById("intake"),
  chat: document.getElementById("chat"),
  done: document.getElementById("done"),
};
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message");
const formError = document.getElementById("form-error");

let sessionId = null;
let dialog = null;

function show(name) {
  for (const [key, view] of Object.entries(views)) view.hidden = key !== name;
}

function addMessage(role, text) {
  const item = document.createElement("li");
  item.className = `message ${role}`;
  item.textContent = text;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong.");
  return data;
}

function finish() {
  setTimeout(() => show("done"), 1500);
}

document.getElementById("profile-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const age = Number(form.age.value);
  formError.hidden = true;

  let problem = null;
  if (!Number.isInteger(age) || age < 16 || age > 120) problem = "Please enter a valid age (16 or older).";
  else if (!form.gender.value) problem = "Please select your gender.";
  else if (!form.consent.checked) problem = "Please give your consent to take part.";
  if (problem) {
    formError.textContent = problem;
    formError.hidden = false;
    return;
  }

  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const data = await post("/api/session", { age, gender: form.gender.value, consent: true });
    sessionId = data.session_id;
    dialog = data.dialog;
    show("chat");
    addMessage("system", data.system);
    messageInput.focus();
  } catch (error) {
    formError.textContent = error.message;
    formError.hidden = false;
  } finally {
    button.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  addMessage("user", text);
  const fields = chatForm.querySelectorAll("input, button");
  fields.forEach((field) => (field.disabled = true));
  const pending = addMessage("system pending", "…");

  try {
    const data = await post("/api/chat", { session_id: sessionId, message: text, dialog });
    dialog = data.dialog;
    pending.className = "message system";
    pending.textContent = data.system;
    if (data.done) return finish();
  } catch (error) {
    pending.className = "message error";
    pending.textContent = error.message;
  }
  fields.forEach((field) => (field.disabled = false));
  messageInput.focus();
});

document.getElementById("finish").addEventListener("click", () => show("done"));

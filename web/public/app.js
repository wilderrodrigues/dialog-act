// The study runs in the browser: the server is stateless, so the progress is kept here (and in sessionStorage,
// so a page reload does not lose a participant) and each system's results are sent once its questionnaire is done.
const STORAGE_KEY = "dialog-act-study";
const STEPS = ["loading", "researcher", "consent", "info", "instructions", "chat", "rate", "thanks"];
const views = Object.fromEntries(STEPS.map((step) => [step, document.getElementById(step)]));
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message");

let study = null;
let state = null;

function freshState() {
  return { step: "researcher", researcher: null, person: null, participant: null, plan: null,
           system: 0, conversation: 0, trials: [], current: null };
}

function save() {
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* private mode: keep going */ }
}

function restore() {
  try { return JSON.parse(sessionStorage.getItem(STORAGE_KEY)); } catch { return null; }
}

function now() {
  return new Date().toISOString();
}

async function request(url, body) {
  const response = await fetch(url, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(" ") : data.detail;
    throw new Error(detail || "Something went wrong. Please try again.");
  }
  return data;
}

function showError(form, message) {
  const error = form.querySelector(".error");
  error.textContent = message || "";
  error.hidden = !message;
}

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "className") node.className = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

function system() {
  return state.plan[state.system];
}

function goal() {
  return system().goals[state.conversation];
}

function progress() {
  return `${system().label} · conversation ${state.conversation + 1} of ${system().goals.length}`;
}

function go(step) {
  state.step = step;
  save();
  render();
}

function render() {
  for (const [step, view] of Object.entries(views)) view.hidden = step !== state.step;
  const renderers = { researcher: renderResearcher, chat: renderChat, rate: renderRate };
  renderers[state.step]?.();
  window.scrollTo(0, 0);
}

// Researcher ------------------------------------------------------------------------------------------------

const TEST = "test";

function planPerson() {
  return state.person === TEST ? study.researchers[state.researcher][0] : state.person;
}

function personStatus(person) {
  if (person === TEST) return "";
  const done = study.completed[person] || [];
  if (done.length >= 2) return "completed";
  if (done.length === 1) return "half done";
  return "";
}

function renderResearcher() {
  const buttons = document.getElementById("researcher-buttons");
  buttons.replaceChildren(...Object.keys(study.researchers).map((name) => {
    const button = element("button", { type: "button", className: name === state.researcher ? "choice selected" : "choice" }, name);
    button.addEventListener("click", () => {
      state.researcher = name;
      state.person = null;
      go("researcher");
    });
    return button;
  }));

  const picker = document.getElementById("person-picker");
  picker.hidden = !state.researcher;
  if (!state.researcher) return;

  // Only the persons themselves: which mode a person gets first is in the PDF, not on this screen.
  const persons = study.researchers[state.researcher];
  const personButtons = [TEST, ...persons].map((person) => {
    const status = personStatus(person);
    const className = ["choice", "person", person === state.person ? "selected" : "", person === TEST ? "test" : ""]
      .filter(Boolean).join(" ");
    const button = element("button", { type: "button", className },
      element("span", { className: "person-name" }, person === TEST ? "Test" : `Person ${person}`),
      ...(status ? [element("span", { className: "person-status" }, status)] : []));
    button.addEventListener("click", () => {
      state.person = person;
      go("researcher");
    });
    return button;
  });
  document.getElementById("person-buttons").replaceChildren(...personButtons);

  const storageDir = document.getElementById("storage-dir");
  storageDir.textContent = study.storage_dir ? `Results are saved in ${study.storage_dir}` : "";
  storageDir.hidden = !study.storage_dir;

  const warning = document.getElementById("storage-warning");
  warning.textContent = study.storage_error ? `Could not check which persons are done: ${study.storage_error}` : "";
  warning.hidden = !study.storage_error;

  const start = document.getElementById("start-participant");
  start.disabled = !state.person;
  start.textContent = state.person === TEST ? "Start test run"
    : state.person ? `Hand over to participant (person ${state.person})` : "Hand over to participant";
}

document.getElementById("start-participant").addEventListener("click", () => {
  const status = personStatus(state.person);
  if (status && !confirm(`Person ${state.person} is already ${status}. Run this person again?`)) return;
  go("consent");
});

// Consent and participant info --------------------------------------------------------------------------------

document.getElementById("consent-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!event.target.consent.checked) return showError(event.target, "Please give your consent to take part.");
  showError(event.target, null);
  go("info");
});

function buildAiUsage() {
  const scale = document.getElementById("ai-usage");
  scale.replaceChildren(...study.ai_usage.map((label, index) =>
    element("label", { className: "option" },
      element("input", { type: "radio", name: "ai_usage", value: String(index + 1), required: "" }),
      element("span", { className: "number" }, String(index + 1)),
      element("span", { className: "label" }, label))));
}

document.getElementById("info-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const age = Number(form.age.value);
  const aiUsage = Number(form.ai_usage.value);

  let problem = null;
  if (!Number.isInteger(age) || age < 16 || age > 120) problem = "Please enter a valid age (16 or older).";
  else if (!form.gender.value) problem = "Please select your gender.";
  else if (!aiUsage) problem = "Please say how often you use AI.";
  if (problem) return showError(form, problem);
  showError(form, null);

  const participant = { researcher: state.researcher, person: planPerson(), test: state.person === TEST, consent: true,
                        age, gender: form.gender.value, ai_usage: aiUsage };
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const data = await request("/api/participant", participant);
    Object.assign(state, { participant: { ...participant, participant_id: data.participant_id }, plan: data.plan,
                           system: 0, conversation: 0, trials: [], current: null });
    go("instructions");
  } catch (error) {
    showError(form, error.message);
  } finally {
    button.disabled = false;
  }
});

document.getElementById("instructions-next").addEventListener("click", () => go("chat"));

// Back: possible until the first conversation starts (nothing is stored before that), e.g. after picking the
// wrong person. Going back to the researcher keeps the chosen researcher and person, so a click fixes it.
for (const button of document.querySelectorAll("[data-back]")) {
  button.addEventListener("click", () => go(button.dataset.back));
}

// Chat --------------------------------------------------------------------------------------------------------

// The info button next to the chat shows the instructions the participant read before the first conversation.
document.getElementById("info-dialog-steps").append(document.querySelector("#instructions .steps").cloneNode(true));
document.getElementById("chat-info").addEventListener("click", () => document.getElementById("info-dialog").showModal());

// A conversation starts as soon as the chat opens (there is no separate goal screen); on a failure the participant
// can retry from the chat itself.
let starting = false;

async function startConversation() {
  if (starting) return;
  starting = true;
  chatForm.hidden = true;
  document.getElementById("chat-done").hidden = true;
  messages.replaceChildren();
  const pending = addMessage("system pending", "…");
  try {
    const data = await request("/api/dialog", { mode: system().mode });
    state.current = { started_at: now(), dialog: data.dialog, done: false,
                      transcript: [{ system: data.system, state: data.dialog.state, time: now() }] };
    save();
    renderChat();
    messageInput.focus();
  } catch (error) {
    pending.remove();
    const retry = element("button", { type: "button" }, "Try again");
    retry.addEventListener("click", startConversation);
    messages.append(element("li", { className: "message error" }, `${error.message} `, retry));
  } finally {
    starting = false;
  }
}

function addMessage(role, text) {
  const item = element("li", { className: `message ${role}` }, text);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

function setChatDone(done) {
  chatForm.hidden = done;
  document.getElementById("chat-done").hidden = !done;
}

function renderChat() {
  document.getElementById("chat-progress").textContent = progress();
  document.getElementById("chat-goal").textContent = goal().text;
  if (!state.current) return void startConversation();
  messages.replaceChildren();
  for (const turn of state.current.transcript) {
    if (turn.user) addMessage("user", turn.user);
    addMessage("system", turn.system);
  }
  setChatDone(state.current.done);
}

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
    const data = await request("/api/chat", { mode: system().mode, message: text, dialog: state.current.dialog });
    pending.className = "message system";
    pending.textContent = data.system;
    state.current.dialog = data.dialog;
    state.current.transcript.push({ user: text, act: data.act, state: data.dialog.state,
                                    preferences: data.dialog.preferences, restaurant: data.dialog.restaurant,
                                    system: data.system, time: now() });
    if (data.done) {
      state.current.done = true;
      state.current.ended_at = now();
      setChatDone(true);
    }
    save();
  } catch (error) {
    pending.className = "message error";
    pending.textContent = error.message;
  }
  fields.forEach((field) => (field.disabled = false));
  if (!state.current.done) messageInput.focus();
});

document.getElementById("chat-next").addEventListener("click", () => {
  const { current } = state;
  const lastRecommendation = current.transcript.findLast((turn) => turn.restaurant != null);
  // The conversation stays in the browser; only the counts are sent and stored.
  const userTurns = current.transcript.filter((turn) => turn.user !== undefined);
  state.trials.push({ goal_slot: goal().slot, goal_value: goal().value, started_at: current.started_at,
                      ended_at: current.ended_at, restaurant: lastRecommendation?.restaurant ?? null,
                      nb_prompts: userTurns.length,
                      nb_resets: userTurns.filter((turn) => turn.act === "restart").length });
  state.current = null;
  if (state.conversation + 1 < system().goals.length) {
    state.conversation += 1;
    go("chat");
  } else {
    go("rate");
  }
});

// Questionnaire -----------------------------------------------------------------------------------------------

function scale(question, labels, left, right) {
  const options = [1, 2, 3, 4, 5].map((score) =>
    element("label", { className: "option", title: labels ? labels[score - 1] : String(score) },
      element("input", { type: "radio", name: `q${question.id}`, value: String(score), required: "" }),
      element("span", { className: "number" }, String(score)),
      ...(labels ? [element("span", { className: "label" }, labels[score - 1])] : [])));
  const row = element("div", { className: left ? "scale differential" : "scale" }, ...options);
  if (left) {
    row.prepend(element("span", { className: "pole" }, left));
    row.append(element("span", { className: "pole" }, right));
  }
  return row;
}

function renderRate() {
  document.getElementById("rate-progress").textContent = `${system().label} · questionnaire`;
  document.getElementById("rate-title").textContent = `What did you think of ${system().label}?`;
  document.querySelectorAll(".system-name").forEach((node) => (node.textContent = system().label));
  const form = document.getElementById("rate-form");
  form.reset();
  showError(form, null);

  const partA = study.questionnaire.filter((question) => question.part === "A");
  const partB = study.questionnaire.filter((question) => question.part === "B");
  document.getElementById("part-a").replaceChildren(...partA.map((question, index) =>
    element("fieldset", { className: "question" },
      element("legend", {}, `${index + 1}. ${question.text}`),
      scale(question, study.likert))));
  document.getElementById("part-b").replaceChildren(...partB.map((question) =>
    element("fieldset", { className: "question" },
      element("legend", { className: "visually-hidden" }, `${question.left} – ${question.right}`),
      scale(question, null, question.left, question.right))));
}

document.getElementById("rate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const answers = {};
  for (const question of study.questionnaire) {
    const value = form[`q${question.id}`].value;
    if (!value) return showError(form, "Please answer every question.");
    answers[question.id] = Number(value);
  }
  showError(form, null);

  const button = form.querySelector("button");
  button.disabled = true;
  try {
    await request("/api/results", { ...state.participant, order: system().order, trials: state.trials, answers });
    state.trials = [];
    if (state.system + 1 < state.plan.length) {
      state.system += 1;
      state.conversation = 0;
      go("chat");
    } else {
      const note = document.getElementById("test-note");
      note.textContent = state.participant.test
        ? `Test run saved in ${study.storage_dir}/test-${state.researcher.toLowerCase()}.json` : "";
      note.hidden = !state.participant.test;
      go("thanks");
    }
  } catch (error) {
    showError(form, `${error.message} Your answers are kept; please try again or call the researcher.`);
  } finally {
    button.disabled = false;
  }
});

// Start -------------------------------------------------------------------------------------------------------

document.getElementById("next-participant").addEventListener("click", async () => {
  study = await request("/api/study").catch(() => study);
  state = freshState();
  go("researcher");
});

async function init() {
  try {
    study = await request("/api/study");
  } catch (error) {
    views.loading.querySelector("p").textContent = `The study could not be loaded: ${error.message}`;
    return;
  }
  buildAiUsage();
  const saved = restore();
  if (saved?.step === "task") saved.step = "chat";  // the old goal screen; the chat now starts the conversation
  state = saved && STEPS.includes(saved.step) && saved.step !== "loading" ? saved : freshState();
  render();
}

init();

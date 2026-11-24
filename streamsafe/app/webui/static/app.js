// StreamSafe window logic. Talks only to the local app (127.0.0.1) with the session token.
"use strict";

const W = {
  ar: {
    nav_live: "البث", nav_setup: "الإعداد", nav_settings: "الإعدادات", nav_check: "الفحص",
    viewers_see: "هذا اللي يشوفه جمهورك", trial: "جرّب بدون بث", start: "ابدأ الحماية والبث", stop: "أوقف",
    starting: "يجهّز…", panic: "غطِّ الكل", panic_off: "ارجع للعرض", panic_on_title: "البث مغطّى بالكامل",
    sens: "قوة التغطية", live_change: "تتغير وأنت تبث", s_light: "خفيفة", s_mid: "متوازنة", s_high: "عالية",
    s_light_t: "تغطي الواضح فقط. أقل تغطية بالغلط، وتناسب اللي يتجنب المشاهد بنفسه.",
    s_mid_t: "الوضع المقترح: يغطي الملابس الكاشفة ويترك العادي.",
    s_high_t: "تغطي أي شي فيه شك. أمان أكثر، وتغطية بالغلط أكثر.",
    everyone: "لبّس الكل", everyone_sub: "يغطي أي شخص حتى الرجال (للإعلانات واللوحات)",
    full: "غطِّ الشاشة كاملة", full_sub: "بدل تغطية الشخص، تنحجب الشاشة كلها لحظة الخطر",
    areas: "مناطق الشاشة", camera_sub: "حدد مكان كاميرتك عشان ما نغطيك أبدًا، أو جزء يتغطى دائمًا (خريطة، كود).",
    camera_edit: "مكان الكاميرا", camera_done: "تم", camera_add: "أضف منطقة", zone: "كاميرتي",
    cover_edit: "غطِّ جزء دائمًا", cover_zone: "مغطى دائمًا",
    edit_cam: "اسحب المربع الأخضر فوق كاميرتك، وكبّره من الزاوية.", edit_cover: "اسحب المربع الوردي فوق الجزء اللي تبيه يتغطى دائمًا.",
    boxes_hint: "اضغط على أي تغطية في الصورة عشان تشيلها دقيقة (لو غطّت شخص بالغلط). ترجع لحالها.",
    release: "شيل التغطية دقيقة", restore_in: "ترجع بعد {n} ث — اضغط ترجعها الحين",
    connection: "الاتصال", obs: "OBS", idle_screen: "اضغط «جرّب بدون بث» عشان تشوف الحماية قبل ما تبث، أو ابدأ البث مباشرة.",
    wait_screen: "ننتظر الصورة من OBS… افتح OBS واضغط Start Streaming.",
    back: "رجوع", next: "التالي", finish: "جاهز، روح للبث", cancel: "إلغاء",
    st1: "الجهاز", st2: "المنصة", st3: "OBS", st4: "جاهز",
    s1_h: "نجهّز جهازك", s1_p: "نحمّل عقل الحماية مرة وحدة (تقريبًا 300MB) ونفحص الجهاز.",
    s1_ok: "جهازك جاهز", download: "حمّل الآن", nvidia: "سرّع بكرت NVIDIA", nvidia_no: "لا شكرًا",
    s2_h: "وين تبث؟", s2_p: "اختر منصتك والصق مفتاح البث. المفتاح يبقى في جهازك فقط.",
    key: "مفتاح البث (Stream Key)", server: "رابط السيرفر", test: "اختبر الاتصال",
    tiktok_note: "تيك توك يحتاج رابط السيرفر والمفتاح من صفحة LIVE Producer.",
    s3_h: "نربط OBS", s3_p: "نضبط OBS يرسل لنا بدل المنصة. ما نغيّر شي ثاني، ونقدر نرجعه زي ما كان.",
    obs_do: "اضبط OBS تلقائيًا", obs_ok: "OBS مضبوط", obs_missing: "ما لقينا OBS. ثبّته من obsproject.com ثم ارجع.",
    s4_h: "كل شي جاهز", s4_p: "من الحين: افتح StreamSafe، اضغط ابدأ، وبعدها Start Streaming في OBS.",
    platform_main: "المنصة الأساسية", platform_second: "منصة ثانية (اختياري)",
    second_sub: "نبث نفس الصورة المحمية لمنصة ثانية بنفس الشكل اللي ضبطته في OBS.",
    none: "بدون", save: "احفظ", saved: "انحفظ",
    panic_key: "زر الطوارئ", panic_sub: "زر من الكيبورد يغطي البث كامل فورًا حتى وأنت داخل اللعبة.",
    more: "خيارات", follow_obs: "اتبع إعدادات OBS", follow_obs_sub: "نفس الدقة والجودة اللي في OBS",
    remember: "تذكّر المفتاح", remember_sub: "يُحفظ مشفّر في جهازك", obs_restore: "رجّع OBS زي ما كان",
    plan: "الاشتراك", activate: "فعّل", support: "ادعمنا", pro: "Pro مفعّل", free: "النسخة المجانية",
    updates: "التحديثات", update_now: "حدّث الآن", up_to_date: "عندك آخر نسخة", update_ready: "نسخة جديدة جاهزة",
    recheck: "افحص مرة ثانية", open_report: "افتح آخر تقرير", check_ok: "كل شي سليم", check_bad: "فيه شي يحتاج انتباه",
    leave_live: "البث شغّال. إذا سكّرت النافذة يوقف البث.", days_left: "باقي {n} يوم على النسخة التجريبية",
  },
  en: {
    nav_live: "Live", nav_setup: "Setup", nav_settings: "Settings", nav_check: "Check",
    viewers_see: "What your viewers see", trial: "Try without streaming", start: "Start protection & stream", stop: "Stop",
    starting: "Starting…", panic: "Cover everything", panic_off: "Back to stream", panic_on_title: "Stream fully covered",
    sens: "Cover strength", live_change: "Changes while live", s_light: "Light", s_mid: "Balanced", s_high: "High",
    s_light_t: "Covers only the obvious. Fewest wrong covers; for streamers who avoid scenes themselves.",
    s_mid_t: "Recommended: covers revealing outfits, leaves normal ones.",
    s_high_t: "Covers anything doubtful. Safer, with more wrong covers.",
    everyone: "Dress everyone", everyone_sub: "Covers every person, men too (ads, billboards)",
    full: "Cover the whole screen", full_sub: "Instead of covering the person, the whole screen hides at the risky moment",
    areas: "Screen areas", camera_sub: "Mark your webcam so we never cover you, or a part that is always covered (a map, a code).",
    camera_edit: "My webcam", camera_done: "Done", camera_add: "Add area", zone: "My webcam",
    cover_edit: "Always cover a part", cover_zone: "Always covered",
    edit_cam: "Drag the green box over your webcam; resize from the corner.", edit_cover: "Drag the pink box over the part to always cover.",
    boxes_hint: "Click any cover in the picture to lift it for a minute (a wrong cover). It comes back by itself.",
    release: "Lift this cover for a minute", restore_in: "Back in {n} s — click to cover now",
    connection: "Connection", obs: "OBS", idle_screen: "Press “Try without streaming” to see protection before going live, or start streaming.",
    wait_screen: "Waiting for OBS… open OBS and press Start Streaming.",
    back: "Back", next: "Next", finish: "Ready, go live", cancel: "Cancel",
    st1: "Device", st2: "Platform", st3: "OBS", st4: "Ready",
    s1_h: "Preparing your PC", s1_p: "We download the protection brain once (about 300 MB) and check your PC.",
    s1_ok: "Your PC is ready", download: "Download now", nvidia: "Speed up with NVIDIA", nvidia_no: "No thanks",
    s2_h: "Where do you stream?", s2_p: "Pick your platform and paste your stream key. It stays on this PC.",
    key: "Stream key", server: "Server URL", test: "Test connection",
    tiktok_note: "TikTok needs the server URL and key from LIVE Producer.",
    s3_h: "Connect OBS", s3_p: "OBS sends to us instead of the platform. Nothing else changes, and you can undo it.",
    obs_do: "Set up OBS for me", obs_ok: "OBS is set", obs_missing: "OBS not found. Install it from obsproject.com and come back.",
    s4_h: "All set", s4_p: "From now on: open StreamSafe, press Start, then Start Streaming in OBS.",
    platform_main: "Main platform", platform_second: "Second platform (optional)",
    second_sub: "Sends the same protected picture to a second platform, in the shape you set in OBS.",
    none: "None", save: "Save", saved: "Saved",
    panic_key: "Panic key", panic_sub: "A keyboard key that covers the whole stream at once, even in game.",
    more: "Options", follow_obs: "Follow OBS settings", follow_obs_sub: "Same resolution and quality as OBS",
    remember: "Remember key", remember_sub: "Stored encrypted on this PC", obs_restore: "Restore OBS as it was",
    plan: "Plan", activate: "Activate", support: "Support us", pro: "Pro active", free: "Free version",
    updates: "Updates", update_now: "Update now", up_to_date: "You have the latest version", update_ready: "A new version is ready",
    recheck: "Check again", open_report: "Open last report", check_ok: "Everything looks good", check_bad: "Something needs attention",
    leave_live: "You are live. Closing the window stops the stream.", days_left: "{n} days left in this beta",
  },
};

const SENS = [["s_light", 1.3], ["s_mid", 1.0], ["s_high", 0.75]];
let S = null; // last state
let lang = "ar";
let view = "live";
let step = 0;
let editing = null; // null | "cam" | "cover"
const rects = { cam: [], cover: [] };
const SAVE = { cam: (r) => act("zones", { zones: r }), cover: (r) => act("covers", { covers: r }) };
let previewAt = 0;
const $ = (id) => document.getElementById(id);
const w = (k, v = {}) => (W[lang][k] || k).replace(/\{(\w+)\}/g, (_, n) => v[n] ?? "");

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  opt.headers = { ...(opt.headers || {}), "X-Token": window.TOKEN };
  const r = await fetch("/api/" + path, opt);
  return r.headers.get("Content-Type")?.includes("json") ? r.json() : r;
}
const act = async (name, body = {}) => { const out = await api(name, body); await tick(); return out; };

// --- words ------------------------------------------------------------------------------------------------
function paintWords() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-w]").forEach((el) => { el.textContent = w(el.dataset.w); });
  $("lang").textContent = lang === "ar" ? "English" : "عربي";
}

// --- navigation -------------------------------------------------------------------------------------------
function go(name) {
  view = name;
  document.querySelectorAll(".nav").forEach((b) => b.classList.toggle("on", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== "view-" + name; });
  if (name === "setup") paintSetup();
  if (name === "settings") paintSettings(true);
}

// --- live view --------------------------------------------------------------------------------------------
function paintLive() {
  const s = S, st = s.status;
  const live = s.running;
  $("ring").classList.toggle("on", live && st.look === "ok");
  const mb = $("main-btn");
  mb.textContent = live ? w("stop") : w("start");
  mb.classList.toggle("stop", live);
  mb.disabled = !live && (!!s.job || !!s.setup.models);
  $("trial-btn").hidden = live;
  mb.parentElement.classList.toggle("live", live);
  $("trial-btn").disabled = !!s.job || !!s.setup.models;
  const pb = $("panic-btn");
  pb.textContent = (s.panic ? w("panic_off") : w("panic")) + (s.settings.panic_button ? `  (${s.settings.panic_button})` : "");
  pb.classList.toggle("on", s.panic);
  $("panic-veil").hidden = !s.panic;
  $("panic-hint").textContent = s.settings.panic_button ? `${s.settings.panic_button}` : "";
  $("stage-meta").textContent = live ? st.hint : "";
  // picture
  const img = $("preview");
  if (live && s.preview_at && s.preview_at !== previewAt) {
    previewAt = s.preview_at;
    const next = new Image();
    next.onload = () => { img.src = next.src; img.hidden = false; };
    next.src = `/api/preview?t=${window.TOKEN}&v=${previewAt}`;
  }
  if (!live) { img.hidden = true; previewAt = 0; }
  $("screen-empty").hidden = live && !img.hidden;
  $("screen-empty-text").textContent = live ? w("wait_screen") : w("idle_screen");
  // message
  const m = $("msg");
  m.hidden = !s.message.text;
  m.textContent = s.message.text;
  m.className = "msg " + (s.message.kind || "");
  // sensitivity
  const k = s.settings.sens;
  const pick = SENS.reduce((a, b) => (Math.abs(b[1] - k) < Math.abs(a[1] - k) ? b : a));
  const seg = $("sens");
  if (!seg.childElementCount) {
    SENS.forEach(([key, val]) => {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "radio"); b.dataset.k = val; b.dataset.w = key;
      b.onclick = () => act("sens", { k: val });
      seg.append(b);
    });
    paintWords();
  }
  seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(+b.dataset.k === pick[1])));
  $("sens-text").textContent = w(pick[0] + "_t");
  $("sw-everyone").checked = s.settings.everyone;
  $("sw-full").checked = s.settings.cover_mode === "screen";
  // health
  const h = $("health");
  h.replaceChildren();
  const row = (name, text, look) => { const d = document.createElement("div"); d.innerHTML = "<span></span><b></b>"; d.firstChild.textContent = name; d.lastChild.textContent = text; d.lastChild.className = look; h.append(d); };
  if (s.link) {
    row(w("obs"), s.link.obs.text, s.link.obs.look);
    s.link.out.forEach((o) => row(o.name, o.text, o.look));
  } else {
    row(w("obs"), S.setup.obs_set ? "✓" : "—", S.setup.obs_set ? "ok" : "idle");
    row(s.platforms.find((p) => p.id === s.settings.platform)?.name || "—", s.setup.key_ok ? "✓" : "—", s.setup.key_ok ? "ok" : "idle");
  }
  if (!editing) {
    rects.cam = s.settings.zones.map((z) => [...z]);
    rects.cover = s.settings.covers.map((z) => [...z]);
    paintZones();
  }
  paintLiveBoxes(s);

}

// --- covers on the live picture: click to lift one for a minute, click again to bring it back ------------
function paintLiveBoxes(s) {
  const box = $("live-boxes");
  box.replaceChildren();
  const show = s.running && !editing && !s.panic;
  $("boxes-hint").hidden = !show || !(s.boxes.length || s.allow.length);
  if (!show) return;
  const add = (z, cls, text, onclick) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "lbox " + cls; place(b, z);
    const t = document.createElement("span"); t.textContent = text; b.append(t);
    b.onclick = onclick; box.append(b);
  };
  s.allow.forEach((a, i) => add(a, "allowed", w("restore_in", { n: a[4] }), () => act("restore", { i })));
  s.boxes.forEach((z) => add(z, "", w("release"), () => act("release", { box: z })));
}

// --- screen areas: webcam (never covered), always-covered parts, the TikTok 9:16 frame ---------------------------
function paintZones() {
  const box = $("zones");
  box.classList.toggle("edit", !!editing);
  box.replaceChildren();
  for (const kind of ["cover", "cam"]) {
    rects[kind].forEach((z, i) => {
      const el = document.createElement("div");
      el.className = `zone ${kind}` + (editing === kind ? " active" : "");
      place(el, z);
      el.textContent = w({ cam: "zone", cover: "cover_zone" }[kind]);
      const h = document.createElement("i"); h.className = "h";
      el.append(h);
      const x = document.createElement("button"); x.className = "x"; x.type = "button"; x.textContent = "×"; x.setAttribute("aria-label", "remove");
      x.onclick = (e) => { e.stopPropagation(); rects[kind].splice(i, 1); paintZones(); };
      el.append(x);
      el.onpointerdown = (e) => editing === kind && drag(e, kind, i, e.target === h ? "size" : "move");
      box.append(el);
    });
  }
}
function place(el, z) { Object.assign(el.style, { left: z[0] * 100 + "%", top: z[1] * 100 + "%", width: (z[2] - z[0]) * 100 + "%", height: (z[3] - z[1]) * 100 + "%" }); }
function drag(e, kind, i, mode) {
  e.preventDefault();
  const r = $("screen").getBoundingClientRect();
  const z0 = [...rects[kind][i]], x0 = e.clientX, y0 = e.clientY;
  const clamp = (v) => Math.min(1, Math.max(0, v));
  const el = e.currentTarget; el.setPointerCapture(e.pointerId);
  el.onpointermove = (m) => {
    const dx = (m.clientX - x0) / r.width, dy = (m.clientY - y0) / r.height;
    const z = rects[kind][i];
    const ww = z0[2] - z0[0], hh = z0[3] - z0[1];
    if (mode === "move") {
      z[0] = Math.min(1 - ww, Math.max(0, z0[0] + dx)); z[1] = Math.min(1 - hh, Math.max(0, z0[1] + dy));
      z[2] = z[0] + ww; z[3] = z[1] + hh;

    } else { // the handle sits on the bottom corner at the inline-start side
      if (document.documentElement.dir === "rtl") z[2] = clamp(Math.max(z0[0] + 0.04, z0[2] + dx)); else z[0] = clamp(Math.min(z0[2] - 0.04, z0[0] + dx));
      z[3] = clamp(Math.max(z0[1] + 0.04, z0[3] + dy));
    }
    place(el, z);
  };
  el.onpointerup = () => { el.onpointermove = null; };
}
function editAreas(kind) {
  if (editing) SAVE[editing](rects[editing]);
  editing = kind;
  if (kind && !rects[kind].length) rects[kind].push(kind === "cam" ? [0.74, 0.62, 0.98, 0.96] : [0.02, 0.62, 0.22, 0.96]);
  $("area-btns").hidden = !!kind;
  $("area-edit").hidden = !kind;
  $("areas-text").textContent = kind ? w("edit_" + kind) : w("camera_sub");
  paintZones();
}

// --- setup wizard -----------------------------------------------------------------------------------------
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k.startsWith("on") ? (e[k] = v) : k === "text" ? (e.textContent = v) : e.setAttribute(k, v);
  e.append(...kids);
  return e;
}
function stepDone(i) {
  const s = S.setup;
  return [!s.models, s.key_ok, s.obs_set, s.key_ok && s.obs_set && !s.models][i];
}
function paintSetup() {
  if (!S) return;
  const steps = $("steps");
  steps.replaceChildren(...["st1", "st2", "st3", "st4"].map((k, i) => el("li", { class: i === step ? "now" : stepDone(i) ? "done" : "" }, el("i", { text: stepDone(i) && i !== step ? "✓" : String(i + 1) }), w(k))));
  const b = $("step-body");
  b.replaceChildren();
  const s = S.setup, set = S.settings;
  if (step === 0) {
    b.append(el("h1", { text: w("s1_h") }), el("p", { text: w("s1_p") }));
    if (s.models) b.append(el("p", { class: "note", text: s.models }), el("button", { class: "primary", type: "button", text: w("download"), onclick: () => act("models") }));
    else b.append(el("div", { class: "big-ok", text: "✓  " + w("s1_ok") }));
    if (s.nvidia) b.append(el("div", { class: "row-btns" }, el("button", { class: "secondary", type: "button", text: w("nvidia"), onclick: () => act("nvidia") }), el("button", { class: "ghost", type: "button", text: w("nvidia_no"), onclick: () => act("nvidia_no") })));
  } else if (step === 1) {
    b.append(el("h1", { text: w("s2_h") }), el("p", { text: w("s2_p") }), platformForm(false));
  } else if (step === 2) {
    b.append(el("h1", { text: w("s3_h") }), el("p", { text: w("s3_p") }));
    if (!s.obs_found) b.append(el("p", { class: "note", text: w("obs_missing") }));
    else if (s.obs_set) b.append(el("div", { class: "big-ok", text: "✓  " + w("obs_ok") }));
    else b.append(el("button", { class: "primary", type: "button", text: w("obs_do"), onclick: () => act("obs_setup") }));
    if (S.message.text) b.append(el("p", { class: "msg " + S.message.kind, text: S.message.text }));
  } else {
    b.append(el("h1", { text: w("s4_h") }), el("p", { text: w("s4_p") }));
  }
  $("step-back").style.visibility = step ? "visible" : "hidden";
  $("step-next").textContent = step === 3 ? w("finish") : w("next");
}

// redraw the wizard only when something it shows changed (never while the streamer types)
let setupSig = "";
function paintSetupLight() {
  const sig = JSON.stringify([S.setup, S.settings.platform, S.message.text, S.job?.name || ""]);
  if (sig !== setupSig) { setupSig = sig; paintSetup(); }
}

function platformForm(second) {
  const set = S.settings;
  const cur = second ? set.platform2 : set.platform;
  const box = el("div");
  const grid = el("div", { class: "plats", role: "radiogroup" });
  const list = second ? [{ id: "", name: w("none") }, ...S.platforms] : S.platforms;
  list.forEach((p) => grid.append(el("button", { class: "plat", type: "button", role: "radio", "aria-checked": String(p.id === cur), text: p.name, onclick: () => act(second ? "platform2" : "platform", second ? { platform2: p.id } : { platform: p.id }).then(() => (second ? paintSettings(true) : paintSetup())) })));
  box.append(grid);
  if (second && !cur) return box;
  if (cur === "tiktok") box.append(el("p", { class: "note", text: w("tiktok_note") }));
  const custom = cur === "tiktok" || cur === "custom" || cur === "other";
  const server = el("input", { class: "field", dir: "ltr", type: "text", value: (second ? set.rtmp_url2 : set.rtmp_url) || "", placeholder: "rtmps://" });
  const key = el("input", { class: "field", dir: "ltr", type: "password", autocomplete: "off", placeholder: (second ? set.has_key2 : set.has_key) ? "••••••••••" : "" });
  if (custom) box.append(el("label", { class: "lab", text: w("server") }), server);
  box.append(el("label", { class: "lab", text: w("key") }), key);
  const save = () => {
    const body = second ? { platform2: cur, rtmp_url2: server.value } : { platform: cur, rtmp_url: server.value };
    if (key.value) body[second ? "key2" : "key"] = key.value;
    key.value = "";
    return act(second ? "platform2" : "platform", body);
  };
  const result = el("span", { class: "result " + (S.test.kind || ""), text: S.test.text });
  const row = el("div", { class: "row-btns" }, el("button", { class: "secondary", type: "button", text: w("save"), onclick: () => save().then(() => (second ? paintSettings(true) : paintSetup())) }));
  if (!second) row.append(el("button", { class: "ghost", type: "button", text: w("test"), onclick: () => save().then(() => act("test")) }), result);
  box.append(row);
  return box;
}

// --- settings / check -------------------------------------------------------------------------------------
function paintSettings(full) {
  if (!S) return;
  const s = S.settings;
  if (full) {
    $("set-plat1").replaceChildren(platformForm(false));
    $("set-plat2").replaceChildren(platformForm(true));
    const pk = $("panic-key");
    pk.replaceChildren(el("option", { value: "", text: w("none") }), ...S.panic_keys.map((k) => el("option", { value: k, text: k })));
    pk.value = s.panic_button;
  }
  $("sw-follow").checked = s.follow_obs;
  $("sw-remember").checked = s.remember;
  $("plan-text").textContent = S.plan.pro ? `${w("pro")} ${S.plan.until ? "· " + S.plan.until : ""}` : w("free") + (S.days_left != null && S.days_left >= 0 ? " — " + w("days_left", { n: S.days_left }) : "");
  $("update-text").textContent = S.update ? `${w("update_ready")} ${S.update.version || ""}` : w("up_to_date");
  $("update-btn").hidden = !S.update;
}
function paintCheck() {
  const c = S.check;
  $("verdict").textContent = c.verdict === "ok" ? w("check_ok") : w("check_bad");
  $("checks").replaceChildren(...c.items.map((i) => el("li", { class: i.status }, el("b", { text: i.name }), el("small", { text: [i.detail, i.fix].filter(Boolean).join(" — ") }))));
}

// --- loop ---------------------------------------------------------------------------------------------------
async function tick() {
  try { S = await api("state"); } catch { return; }
  if (S.lang !== lang) { lang = S.lang; paintWords(); if (view === "setup") paintSetup(); if (view === "settings") paintSettings(true); }
  $("version").textContent = S.version;
  const chip = $("status-chip");
  chip.textContent = S.status.label;
  chip.className = "chip " + S.status.look;
  const job = $("job");
  job.hidden = !S.job;
  if (S.job) {
    $("job-text").textContent = S.job.text;
    const bar = $("job-bar");
    bar.classList.toggle("spin", S.job.pct < 0);
    bar.style.width = S.job.pct < 0 ? "30%" : Math.round(S.job.pct * 100) + "%";
  }
  const banner = $("banner");
  banner.hidden = !S.setup.models || view === "setup";
  banner.textContent = S.setup.models;
  paintLive();
  if (view === "settings") paintSettings(false);
  if (view === "check") paintCheck();
  document.querySelectorAll(".result").forEach((r) => { r.textContent = S.test.text; r.className = "result " + (S.test.kind || ""); });
  if (view === "setup" && !document.activeElement?.matches("input")) paintSetupLight();
}
let timer = null;
function loop() { clearTimeout(timer); tick().finally(() => { timer = setTimeout(loop, document.hidden ? 3000 : S?.running ? 700 : 1500); }); }

// --- wiring ---------------------------------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll(".nav").forEach((b) => { b.onclick = () => go(b.dataset.view); });
  $("lang").onclick = () => act("lang");
  $("main-btn").onclick = () => act(S?.running ? "stop" : "start", {});
  $("trial-btn").onclick = () => act("start", { preview: true });
  $("panic-btn").onclick = () => act("panic", { on: !S?.panic });
  $("sw-everyone").onchange = (e) => act("everyone", { on: e.target.checked });
  $("sw-full").onchange = (e) => act("cover_mode", { mode: e.target.checked ? "screen" : "people" });
  document.querySelectorAll(".area-btn").forEach((b) => { b.onclick = () => editAreas(b.dataset.kind); });
  $("zone-done").onclick = () => editAreas(null);
  $("zone-add").onclick = () => { rects[editing].push([0.4, 0.4, 0.6, 0.6]); paintZones(); };
  $("banner").onclick = () => { step = 0; go("setup"); };
  $("job-cancel").onclick = () => act("cancel");
  $("step-back").onclick = () => { step = Math.max(0, step - 1); paintSetup(); };
  $("step-next").onclick = () => { if (step === 3) go("live"); else { step += 1; paintSetup(); } };
  $("panic-key").onchange = (e) => act("panic_key", { key: e.target.value });
  $("sw-follow").onchange = (e) => act("toggle", { name: "follow_obs", on: e.target.checked });
  $("sw-remember").onchange = (e) => act("toggle", { name: "remember_key", on: e.target.checked });
  $("obs-restore").onclick = () => act("obs_restore");
  $("pass-btn").onclick = () => { const v = $("pass").value; $("pass").value = ""; act("pass", { key: v }); };
  $("support").onclick = () => act("open", { what: "support" });
  $("update-btn").onclick = () => act("update");
  $("recheck").onclick = () => act("recheck");
  $("open-report").onclick = () => act("open", { what: "report" });
  window.addEventListener("beforeunload", (e) => { if (S?.running) { e.preventDefault(); e.returnValue = w("leave_live"); } });
  document.addEventListener("visibilitychange", loop);
  paintWords();
  await tick();
  // first run: send the streamer to the guided setup
  if (S?.tab) { go({ home: "live", check: "check", settings: "settings", setup: "setup" }[S.tab] || "live"); }
  else if (S && (S.setup.models || !S.setup.key_ok || !S.setup.obs_set)) { step = S.setup.models ? 0 : !S.setup.key_ok ? 1 : 2; go("setup"); }
  loop();
});

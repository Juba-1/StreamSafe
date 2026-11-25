// StreamSafe window logic. Talks only to the local app (127.0.0.1) with the session token.
"use strict";

const W = {
  ar: {
    nav_live: "البث", nav_setup: "الإعداد", nav_pictures: "تغطية دائمة",
    pics_title: "صور تتغطى دائمًا", pics_sub: "لكل لعبة: أي صورة تطلع بنفس المكان (الخريطة، الجوال، قائمة) تتغطى أول ما تظهر.",
    pic_new: "أضف صورة", pic_game: "اللعبة", pic_name: "اسم الصورة (مثل: الخريطة)", pic_full: "غطِّ الشاشة كاملة لما تطلع",
    pic_full_sub: "بدل ما نغطي مكانها بس", pic_save: "احفظ", pic_retake: "صورة جديدة",
    pic_step: "افتح الشي في اللعبة (مثل الخريطة)، وبعدها اسحب مربع على الجزء اللي يميّزه وما يتغير.",
    pic_need: "عشان نصوّر الشاشة لازم الحماية تكون شغالة: اضغط «جرّب بدون بث»، افتح الشي في اللعبة، وارجع هنا.",
    pics_empty: "ما أضفت صور للحين. اضغط «أضف صورة» وأنت فاتح الشي اللي تبيه يتغطى في اللعبة.",
    pic_on_all: "شغّل الكل", pic_off_all: "طفّ الكل", pic_area: "مكانها فقط", pic_screen: "الشاشة كاملة", pic_del: "احذف", nav_settings: "الإعدادات", nav_check: "الفحص",
    viewers_see: "هذا اللي يشوفه جمهورك", trial: "جرّب بدون بث", start: "ابدأ الحماية والبث", stop: "أوقف", stopping: "يوقف…",
    starting: "يجهّز…", held: "الحماية", held_ready: "✓ جاهزة، البث يبدأ فوراً", held_loading: "تتجهز في الخلفية…", panic: "غطِّ الكل", panic_off: "ارجع للعرض", panic_on_title: "البث مغطّى بالكامل",
    sens: "قوة التغطية", live_change: "تتغير وأنت تبث", s_light: "خفيفة", s_mid: "متوازنة", s_high: "عالية",
    s_light_t: "تغطي الواضح جداً فقط. أقل تغطية بالغلط، وتناسب اللي يتجنب المشاهد بنفسه.",
    s_mid_t: "الوضع المقترح: يغطي الملابس الكاشفة ويترك العادي والرجال.",
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
    brand: "شعارك على التغطية", brand_pick: "اختر صورة", brand_clear: "رجّع شعارنا",
    brand_pro: "مع برو: شعارك أو شعار الراعي يطلع على الملابس والشاشة الكاملة بدل شعار StreamSafe.",
    brand_free: "هذي ميزة برو. بالنسخة المجانية يطلع شعار StreamSafe. لو انتهى الاشتراك يرجع شعارنا تلقائيًا.",
    updates: "التحديثات", update_now: "حدّث الآن", up_to_date: "عندك آخر نسخة", update_ready: "نسخة جديدة جاهزة",
    recheck: "افحص مرة ثانية", open_report: "افتح آخر تقرير", check_ok: "كل شي سليم", check_bad: "فيه شي يحتاج انتباه",
    leave_live: "البث شغّال. إذا سكّرت النافذة يوقف البث.",
    no_answer: "التطبيق ما رد على الضغطة.", offline: "النافذة فقدت الاتصال بالتطبيق… نحاول مرة ثانية. لو طوّل سكّر StreamSafe وافتحه.",
    report_title: "تقرير آخر جلسة", report_open: "افتح التقرير", report_copy: "انسخ", report_hide: "إخفاء", copied: "انسخ ✓", days_left: "باقي {n} يوم على النسخة التجريبية",
  },
  en: {
    nav_live: "Live", nav_setup: "Setup", nav_pictures: "Always cover",
    pics_title: "Pictures always covered", pics_sub: "Per game: anything that shows at the same place (the map, the phone, a menu) is covered as soon as it appears.",
    pic_new: "Add picture", pic_game: "Game", pic_name: "Picture name (e.g. the map)", pic_full: "Cover the whole screen when it shows",
    pic_full_sub: "Instead of only its place", pic_save: "Save", pic_retake: "New snapshot",
    pic_step: "Open it in the game (e.g. the map), then drag a box over the part that identifies it and does not change.",
    pic_need: "To take a snapshot, protection must be running: press “Try without streaming”, open it in the game, and come back here.",
    pics_empty: "No pictures yet. Press “Add picture” while the thing to cover is open in the game.",
    pic_on_all: "All on", pic_off_all: "All off", pic_area: "Its place only", pic_screen: "Whole screen", pic_del: "Delete", nav_settings: "Settings", nav_check: "Check",
    viewers_see: "What your viewers see", trial: "Try without streaming", start: "Start protection & stream", stop: "Stop", stopping: "Stopping…",
    starting: "Starting…", held: "Protection", held_ready: "✓ Ready, the stream starts at once", held_loading: "Getting ready in the background…", panic: "Cover everything", panic_off: "Back to stream", panic_on_title: "Stream fully covered",
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
    brand: "Your logo on the covers", brand_pick: "Choose image", brand_clear: "Use our logo",
    brand_pro: "With Pro: your or your sponsor's logo shows on the clothing and the full-screen cover instead of StreamSafe's.",
    brand_free: "A Pro feature. The free version shows the StreamSafe logo, and it comes back by itself when Pro ends.",
    updates: "Updates", update_now: "Update now", up_to_date: "You have the latest version", update_ready: "A new version is ready",
    recheck: "Check again", open_report: "Open last report", check_ok: "Everything looks good", check_bad: "Something needs attention",
    leave_live: "You are live. Closing the window stops the stream.",
    no_answer: "The app did not answer that press.", offline: "The window lost the app… retrying. If it lasts, close StreamSafe and open it again.",
    report_title: "Last session report", report_open: "Open report", report_copy: "Copy", report_hide: "Hide", copied: "Copied ✓", days_left: "{n} days left in this beta",
  },
};

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
  opt.signal = AbortSignal.timeout(path === "state" ? 8000 : 30000);
  const r = await fetch("/api/" + path, opt);
  if (!r.ok && r.status !== 404) {
    const why = await r.json().catch(() => ({}));
    throw new Error(why.error || "HTTP " + r.status);
  }
  return r.headers.get("Content-Type")?.includes("json") ? r.json() : r;
}
// every press: the button shows it is working until the app answers; any failure is shown, never swallowed
async function act(name, body = {}) {
  const btn = document.activeElement?.tagName === "BUTTON" ? document.activeElement : null;
  btn?.classList.add("busy");
  try {
    const out = await api(name, body);
    if (out && out.ok === false && out.why && !out.shown) toast(out.why, "bad");
    await tick();
    return out || {};
  } catch (e) {
    toast(w("no_answer") + "\n" + (e?.message || e), "bad");
    return { ok: false };
  } finally {
    btn?.classList.remove("busy");
  }
}

function toast(text, kind = "info") {
  if (!text) return;
  const t = document.createElement("div");
  t.className = "toast " + kind;
  const x = document.createElement("button"); x.type = "button"; x.textContent = "×"; x.onclick = () => t.remove();
  const span = document.createElement("span"); span.textContent = text;
  t.append(span, x);
  $("toasts").append(t);
  while ($("toasts").childElementCount > 3) $("toasts").firstChild.remove();
  setTimeout(() => t.remove(), kind === "bad" ? 15000 : 7000);  // an error stays longer, never for ever
}

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
  if (name === "pictures") { $("pic-need").hidden = true; paintPictures(true); }
}

// --- live view --------------------------------------------------------------------------------------------
function paintLive() {
  const s = S, st = s.status;
  const live = s.running;
  $("ring").classList.toggle("on", live && st.look === "ok");
  const mb = $("main-btn");
  mb.textContent = s.stopping ? w("stopping") : live ? w("stop") : w("start");
  mb.classList.toggle("stop", live);
  mb.disabled = !!s.stopping || (!live && (!!s.job || !!s.setup.models));
  paintReport(s);
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
  const SENS = s.settings.sens_levels;  // [[word key, k], ...] from streamsafe/sensitivity.py
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
    if (s.held) row(w("held"), w(s.held === "ready" ? "held_ready" : "held_loading"), s.held === "ready" ? "ok" : "warn");
  }
  if (!editing) {
    rects.cam = s.settings.zones.map((z) => [...z]);
    rects.cover = s.settings.covers.map((z) => [...z]);
    paintZones();
  }
  paintLiveBoxes(s);

}

// --- the last session report, right under the picture after a stop --------------------------------------
function paintReport(s) {
  const r = s.report, box = $("report");
  const show = !s.running && r && r.new && reportHidden !== r.name;
  box.hidden = !show;
  if (!show) return;
  $("report-at").textContent = new Date(r.at * 1000).toLocaleTimeString(lang === "ar" ? "ar" : "en");
  if ($("report-text").textContent !== r.text) $("report-text").textContent = r.text;
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
      el.onpointerdown = (e) => editing === kind && e.target !== x && drag(e, kind, i, e.target === h ? "size" : "move");
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

// the platform cards live in two places (setup wizard and settings): redraw whichever is on screen
function redrawForms() {
  if (view === "settings") paintSettings(true);
  if (view === "setup") paintSetup();
}

function platformForm(second) {
  const set = S.settings;
  const cur = second ? set.platform2 : set.platform;
  const box = el("div");
  const grid = el("div", { class: "plats", role: "radiogroup" });
  const list = second ? [{ id: "", name: w("none") }, ...S.platforms] : S.platforms;
  list.forEach((p) => grid.append(el("button", { class: "plat", type: "button", role: "radio", "aria-checked": String(p.id === cur), text: p.name, onclick: () => act(second ? "platform2" : "platform", second ? { platform2: p.id } : { platform: p.id }).then(redrawForms) })));
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
  const row = el("div", { class: "row-btns" }, el("button", { class: "secondary", type: "button", text: w("save"), onclick: () => save().then((out) => { redrawForms(); if (out.ok !== false) toast(w("saved"), "ok"); }) }));
  if (!second) row.append(el("button", { class: "ghost", type: "button", text: w("test"), onclick: () => save().then(() => act("test")) }), result);
  box.append(row);
  return box;
}

// --- pictures always covered -------------------------------------------------------------------------------
let brandV = 0;
let picRect = null;
let picSig = "";
const tok = () => window.TOKEN;
function paintPictures(force) {
  if (!S) return;
  const sig = JSON.stringify([S.pictures, lang]);
  $("pic-need").hidden = S.running || $("pic-add").hidden === false ? true : $("pic-need").hidden;
  if (sig === picSig && !force) return;
  picSig = sig;
  const games = [...new Set(S.pictures.map((p) => p.game))];
  $("pic-games").replaceChildren(...games.map((g) => el("option", { value: g })));
  const list = $("pic-list");
  if (!S.pictures.length) { list.replaceChildren(el("div", { class: "card empty", text: w("pics_empty") })); return; }
  list.replaceChildren(...games.map((g) => {
    const items = S.pictures.filter((p) => p.game === g);
    const allOn = items.every((p) => p.on);
    const head = el("div", { class: "game-head" }, el("h2", { text: g }),
      el("button", { class: "ghost small", type: "button", text: allOn ? w("pic_off_all") : w("pic_on_all"), onclick: () => act("pic_set", { game: g, on: !allOn }) }));
    const grid = el("div", { class: "pic-grid" }, ...items.map((p) => {
      const sw = el("input", { type: "checkbox", "aria-label": p.name });
      sw.checked = p.on;
      sw.onchange = () => act("pic_set", { id: p.id, on: sw.checked });
      const thumb = el("div", { class: "thumb" });
      thumb.style.backgroundImage = `url("/api/picture?id=${p.id}&t=${tok()}")`;
      return el("div", { class: "card pic" + (p.on ? "" : " off") }, thumb,
        el("div", { class: "pic-row" }, el("span", {}, el("b", { text: p.name }), el("br"), el("small", { text: p.full ? w("pic_screen") : w("pic_area") })), sw),
        el("div", { class: "pic-row" }, el("span"), el("button", { class: "del", type: "button", text: w("pic_del"), onclick: () => act("pic_del", { id: p.id }) })));
    }));
    return el("section", { class: "game" }, head, grid);
  }));
}
async function picStart() {
  $("pic-err").hidden = true;
  if (!S?.running) { $("pic-need").hidden = false; $("pic-add").hidden = true; return; }
  const r = await api("pic_freeze", {});
  if (!r.ok) { $("pic-need").hidden = false; return; }
  $("pic-need").hidden = true;
  $("pic-add").hidden = false;
  $("pic-step").textContent = w("pic_step");
  $("pic-frozen").src = `/api/picture?id=_frozen&t=${tok()}&v=${Date.now()}`;
  picRect = null; $("pic-sel").hidden = true;
  if (!$("pic-game").value && S.pictures.length) $("pic-game").value = S.pictures[S.pictures.length - 1].game;
}
function picDraw() {
  const shot = document.querySelector(".pic-shot");
  shot.onpointerdown = (e) => {
    const r = shot.getBoundingClientRect();
    const fx = (x) => Math.min(1, Math.max(0, (x - r.left) / r.width)), fy = (y) => Math.min(1, Math.max(0, (y - r.top) / r.height));
    const x0 = fx(e.clientX), y0 = fy(e.clientY);
    shot.setPointerCapture(e.pointerId);
    shot.onpointermove = (m) => {
      const x1 = fx(m.clientX), y1 = fy(m.clientY);
      picRect = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
      $("pic-sel").hidden = false; place($("pic-sel"), picRect);
    };
    shot.onpointerup = () => { shot.onpointermove = null; };
  };
}
async function picSave() {
  const err = $("pic-err");
  const out = await api("pic_add", { game: $("pic-game").value, name: $("pic-name").value, rect: picRect || [], full: $("pic-full").checked });
  if (!out.ok) { err.textContent = out.why; err.hidden = false; return; }
  $("pic-add").hidden = true; $("pic-name").value = ""; $("pic-full").checked = false;
  await tick(); paintPictures(true);
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
  $("plan-text").textContent = S.plan.text || "";
  $("update-text").textContent = S.update ? `${w("update_ready")} ${S.update.version || ""}` : w("up_to_date");
  $("update-btn").hidden = !S.update;
  $("brand-text").textContent = S.brand.pro ? w("brand_pro") : w("brand_free");
  $("brand-pick").disabled = !S.brand.pro;
  $("brand-clear").hidden = !S.brand.set;
  const bi = $("brand-img"), want = S.brand.set ? `url("/api/brand?t=${window.TOKEN}&v=${brandV}")` : "";
  if (bi.style.backgroundImage !== want) bi.style.backgroundImage = want;
}
function paintCheck() {
  const c = S.check;
  $("verdict").textContent = c.verdict === "ok" ? w("check_ok") : w("check_bad");
  $("checks").replaceChildren(...c.items.map((i) => el("li", { class: i.status }, el("b", { text: i.name }), el("small", { text: [i.detail, i.fix].filter(Boolean).join(" — ") }))));
}

// --- loop ---------------------------------------------------------------------------------------------------
async function tick() {
  try {
    S = await api("state");
    fails = 0;
    $("offline").hidden = true;
  } catch (e) {
    fails += 1;
    if (fails >= 3) { $("offline").hidden = false; $("offline").textContent = w("offline") + "\n\n" + (e?.message || ""); }
    return;
  }
  if (S.message.id !== lastMsg) { if (lastMsg !== null && S.message.text) toast(S.message.text, S.message.kind); lastMsg = S.message.id; }
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
  if (view === "pictures") paintPictures(false);
  document.querySelectorAll(".result").forEach((r) => { r.textContent = S.test.text; r.className = "result " + (S.test.kind || ""); });
  if (view === "setup" && !document.activeElement?.matches("input")) paintSetupLight();
}
let timer = null;
let fails = 0;
let lastMsg = null;
let reportHidden = "";
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
  $("report-open").onclick = () => act("open", { what: "report" });
  $("report-copy").onclick = async () => { try { await navigator.clipboard.writeText(S.report.text); toast(w("copied"), "ok"); } catch (e) { toast(String(e), "bad"); } };
  $("report-hide").onclick = () => { reportHidden = S.report?.name || ""; paintReport(S); };
  $("brand-pick").onclick = () => $("brand-file").click();
  $("brand-file").onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f) return;
    const r = await fetch("/api/brand_upload", { method: "POST", headers: { "X-Token": window.TOKEN }, body: f }).then((x) => x.json());
    $("brand-err").hidden = r.ok; $("brand-err").textContent = r.why || "";
    brandV = Date.now(); await tick(); paintSettings(false);
  };
  $("brand-clear").onclick = () => act("brand_clear").then(() => { brandV = Date.now(); paintSettings(false); });
  $("pic-new").onclick = picStart;
  $("pic-retake").onclick = picStart;
  $("pic-cancel").onclick = () => { $("pic-add").hidden = true; };
  $("pic-save").onclick = picSave;
  $("pic-trial").onclick = () => act("start", { preview: true }).then(() => setTimeout(picStart, 3500));
  picDraw();
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
  $("pass-btn").onclick = async () => {
    const v = $("pass").value.trim();
    const btn = $("pass-btn"); btn.classList.add("busy");
    const out = await api("pass", { key: v }).catch((e) => ({ ok: false, why: String(e) }));
    if (!out.ok) { btn.classList.remove("busy"); toast(out.why || w("no_answer"), "bad"); return; }
    // the check runs in the background (it asks the internet for the date): wait for its answer here
    for (let i = 0; i < 40; i += 1) { await new Promise((r) => setTimeout(r, 300)); await tick(); if (!S.job) break; }
    btn.classList.remove("busy");
    const res = $("plan-result");
    res.hidden = false; res.textContent = S.message.text; res.className = "msg " + S.message.kind;
    if (S.message.kind === "ok") $("pass").value = "";
  };
  $("support").onclick = () => act("open", { what: "support" });
  $("update-btn").onclick = () => act("update");
  $("recheck").onclick = () => act("recheck");
  $("open-report").onclick = () => act("open", { what: "report" });
  window.addEventListener("beforeunload", (e) => { if (S?.running) { e.preventDefault(); e.returnValue = w("leave_live"); } });
  document.addEventListener("visibilitychange", loop);
  paintWords();
  await tick();
  // first run: send the streamer to the guided setup
  if (S?.tab) { go({ home: "live", check: "check", settings: "settings", setup: "setup", pictures: "pictures" }[S.tab] || "live"); }
  else if (S && (S.setup.models || !S.setup.key_ok || !S.setup.obs_set)) { step = S.setup.models ? 0 : !S.setup.key_ok ? 1 : 2; go("setup"); }
  loop();
});

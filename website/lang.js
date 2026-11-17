(function () {
  var h = document.documentElement;
  function get() {
    var q = new URLSearchParams(location.search).get("lang");
    if (q === "ar" || q === "en") return q;
    try { var s = localStorage.getItem("lang"); if (s === "ar" || s === "en") return s; } catch (e) {}
    return (navigator.language || "").toLowerCase().indexOf("ar") === 0 ? "ar" : "en";
  }
  function set(l) {
    h.lang = l;
    try { localStorage.setItem("lang", l); } catch (e) {}
  }
  set(get());
  document.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest(".lang-btn")) set(h.lang === "ar" ? "en" : "ar");
  });
})();
// hero video: honour reduced motion and give a pause button (motion longer than 5 s needs one)
document.addEventListener("DOMContentLoaded", function () {
  var v = document.getElementById("hero-video"), b = document.querySelector(".pp");
  if (!v || !b) return;
  function show(paused) {
    b.setAttribute("aria-pressed", paused ? "true" : "false");
    b.querySelector('[data-l="en"]').textContent = paused ? "Play" : "Pause";
    b.querySelector('[data-l="ar"]').textContent = paused ? "تشغيل" : "إيقاف";
  }
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { v.removeAttribute("autoplay"); v.pause(); show(true); }
  b.addEventListener("click", function () { if (v.paused) { v.play(); show(false); } else { v.pause(); show(true); } });
});

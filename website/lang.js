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

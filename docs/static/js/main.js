document.addEventListener("DOMContentLoaded", function () {
  // Tabs: <div class="seg" data-tabs="scopeId"><button data-panel="panelId">...</button></div>
  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    var scope = document.getElementById(group.dataset.tabs);
    var buttons = group.querySelectorAll("button[data-panel]");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        buttons.forEach(function (b) { b.classList.toggle("is-active", b === btn); });
        scope.querySelectorAll(".panel").forEach(function (p) {
          p.classList.toggle("is-active", p.id === btn.dataset.panel);
        });
      });
    });
  });

  // Toggles: <div class="seg" data-toggle="targetId"><button data-compact="1">...</button></div>
  document.querySelectorAll("[data-toggle]").forEach(function (group) {
    var target = document.getElementById(group.dataset.toggle);
    var buttons = group.querySelectorAll("button");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        buttons.forEach(function (b) { b.classList.toggle("is-active", b === btn); });
        target.classList.toggle("compact", btn.dataset.compact === "1");
      });
    });
  });

  // Copy BibTeX
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = document.getElementById(btn.dataset.copy).innerText;
      var done = function () {
        var old = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
        setTimeout(function () { btn.innerHTML = old; }, 1600);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } finally { document.body.removeChild(ta); }
      }
    });
  });

  // Math
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }
});

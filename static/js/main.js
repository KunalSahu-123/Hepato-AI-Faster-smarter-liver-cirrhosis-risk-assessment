// ---------- Theme toggle (Dark / Light) ----------
(function initTheme() {
  const saved = localStorage.getItem("hepatoai-theme") || "light";
  document.documentElement.setAttribute("data-theme", saved);
  // The icon lives in the navbar, which isn't parsed yet when this runs.
  document.addEventListener("DOMContentLoaded", () => {
    const icon = document.getElementById("theme-icon");
    if (icon) icon.className = saved === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
  });
})();

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const next = current === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("hepatoai-theme", next);
  const icon = document.getElementById("theme-icon");
  if (icon) icon.className = next === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
}

// ---------- Loading overlay ----------
function showLoading(message) {
  const overlay = document.getElementById("loading-overlay");
  if (!overlay) return;
  if (message) {
    const label = document.getElementById("loading-message");
    if (label) label.textContent = message;
  }
  overlay.classList.add("show");
  overlay.setAttribute("aria-hidden", "false");
}
function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (!overlay) return;
  overlay.classList.remove("show");
  overlay.setAttribute("aria-hidden", "true");
}

// A browser restoring a cached page would otherwise show the overlay forever.
window.addEventListener("pageshow", hideLoading);

// ---------- CSRF ----------
function csrfToken() {
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.getAttribute("content") : "";
}

/** fetch() wrapper that attaches the CSRF header Flask-WTF expects. */
function postForm(url, formData) {
  return fetch(url, {
    method: "POST",
    body: formData,
    headers: { "X-CSRFToken": csrfToken() },
    credentials: "same-origin",
  });
}

// ---------- Toast helper ----------
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const id = "toast-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7);
  const bg = { success: "success", danger: "danger", warning: "warning", info: "primary" }[type] || "primary";

  const wrapper = document.createElement("div");
  wrapper.id = id;
  wrapper.className = `toast align-items-center text-bg-${bg} border-0`;
  wrapper.setAttribute("role", "alert");

  const inner = document.createElement("div");
  inner.className = "d-flex";

  // textContent, not innerHTML: message may echo server text or a filename.
  const body = document.createElement("div");
  body.className = "toast-body";
  body.textContent = message;

  const close = document.createElement("button");
  close.type = "button";
  close.className = "btn-close btn-close-white me-2 m-auto";
  close.setAttribute("data-bs-dismiss", "toast");
  close.setAttribute("aria-label", "Close");

  inner.append(body, close);
  wrapper.append(inner);
  container.append(wrapper);

  const toast = new bootstrap.Toast(wrapper, { delay: 5000 });
  toast.show();
  wrapper.addEventListener("hidden.bs.toast", () => wrapper.remove());
}

// ---------- Bootstrap-style client validation ----------
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".needs-validation").forEach(function (form) {
      form.addEventListener("submit", function (event) {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
          form.classList.add("was-validated");
          const firstInvalid = form.querySelector(":invalid");
          if (firstInvalid) {
            firstInvalid.focus();
            firstInvalid.scrollIntoView({ block: "center", behavior: "smooth" });
          }
          return;
        }
        form.classList.add("was-validated");
        if (form.dataset.showLoading !== "false") {
          showLoading(form.dataset.loadingMessage);
        }
      }, false);
    });

    // Confirm-before-submit for destructive actions.
    document.querySelectorAll("form[data-confirm]").forEach(function (form) {
      form.addEventListener("submit", function (event) {
        if (!window.confirm(form.dataset.confirm)) {
          event.preventDefault();
          event.stopPropagation();
        }
      });
    });
  });
})();

// ---------- Auto-dismiss flash toasts on load ----------
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".auto-toast").forEach((el) => {
    new bootstrap.Toast(el, { delay: 5000 }).show();
  });
});

// ---------- Scroll-triggered reveal (Intersection Observer) ----------
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const reveals = document.querySelectorAll(".scroll-reveal");
    if (!reveals.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );

    reveals.forEach((el) => observer.observe(el));

    // Fallback: reveal any element already in the viewport on load
    setTimeout(() => {
      reveals.forEach((el) => {
        if (!el.classList.contains("revealed")) {
          const rect = el.getBoundingClientRect();
          if (rect.top < window.innerHeight && rect.bottom > 0) {
            el.classList.add("revealed");
          }
        }
      });
    }, 300);
  });
})();

// ---------- Animated number counters ----------
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const counters = document.querySelectorAll("[data-count-to]");
    if (!counters.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;
          observer.unobserve(el);

          const end = parseFloat(el.dataset.countTo);
          const prefix = el.dataset.countPrefix || "";
          const suffix = el.dataset.countSuffix || "";
          const decimals = (el.dataset.countDecimals || "0") | 0;
          const duration = 1600;
          const start = performance.now();

          function tick(now) {
            const t = Math.min((now - start) / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3);
            const val = (end * ease).toFixed(decimals);
            el.textContent = prefix + val + suffix;
            if (t < 1) requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
        });
      },
      { threshold: 0.3 }
    );

    counters.forEach((el) => observer.observe(el));
  });
})();

// ---------- Scroll progress bar ----------
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const bar = document.getElementById("scroll-progress");
    if (!bar) return;
    window.addEventListener("scroll", () => {
      const docH = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = docH > 0 ? (window.scrollY / docH) * 100 + "%" : "0%";
    }, { passive: true });
  });
})();

// ---------- Navbar solid on scroll ----------
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const nav = document.querySelector(".navbar-glass");
    if (!nav) return;
    window.addEventListener("scroll", () => {
      nav.classList.toggle("scrolled", window.scrollY > 50);
    }, { passive: true });
  });
})();

// ---------- 3D liver tilt on mouse move ----------
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const wrap = document.querySelector(".liver-3d-wrap");
    if (!wrap) return;
    const liver = wrap.querySelector(".liver-3d");
    if (!liver) return;

    wrap.addEventListener("mousemove", (e) => {
      const rect = wrap.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      liver.style.transform =
        "rotateY(" + (x * 25) + "deg) rotateX(" + (-y * 20) + "deg)";
    });
    wrap.addEventListener("mouseleave", () => {
      liver.style.transform = "";
    });
  });
})();

// ---------- Parallax on scroll for hero elements ----------
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    const hero = document.querySelector(".hero-section");
    if (!hero) return;
    const orbs = hero.querySelectorAll(".hero-orb");
    const liverWrap = hero.querySelector(".liver-3d-wrap");

    window.addEventListener("scroll", () => {
      const y = window.scrollY;
      if (y > window.innerHeight) return;
      orbs.forEach((orb, i) => {
        const speed = 0.03 + i * 0.015;
        orb.style.transform = "translateY(" + (y * speed) + "px)";
      });
      if (liverWrap) {
        liverWrap.style.transform = "translateY(" + (y * 0.06) + "px)";
      }
    }, { passive: true });
  });
})();

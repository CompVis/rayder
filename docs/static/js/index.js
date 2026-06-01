/* ── Related Works Dropdown ───────────────────────── */

function setMoreWorksOpen(isOpen) {
  const dropdown = document.getElementById("moreWorksDropdown");
  const button = document.querySelector(".more-works-btn");
  if (!dropdown || !button) return;
  dropdown.classList.toggle("show", isOpen);
  button.classList.toggle("active", isOpen);
  button.setAttribute("aria-expanded", String(isOpen));
  dropdown.setAttribute("aria-hidden", String(!isOpen));
}

function toggleMoreWorks() {
  const dropdown = document.getElementById("moreWorksDropdown");
  if (!dropdown) return;
  setMoreWorksOpen(!dropdown.classList.contains("show"));
}


/* ── Copy BibTeX ──────────────────────────────────── */

function copyBibTeX() {
  const source = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const label = document.querySelector(".copy-text");
  if (!source || !button || !label) return;

  const flash = () => {
    button.classList.add("copied");
    label.textContent = "Copied!";
    setTimeout(() => {
      button.classList.remove("copied");
      label.textContent = "Copy";
    }, 2000);
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(source.textContent).then(flash);
    return;
  }

  const ta = document.createElement("textarea");
  ta.value = source.textContent;
  ta.setAttribute("readonly", "");
  ta.style.cssText = "position:absolute;left:-9999px";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  document.body.removeChild(ta);
  flash();
}


/* ── Scroll to Top ────────────────────────────────── */

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}


/* ── Expandable Sections ──────────────────────────── */

function toggleExpand(btn) {
  const content = btn.parentElement.querySelector(".expandable-content");
  if (!content) return;
  const isOpen = content.classList.contains("show");
  content.classList.toggle("show");
  btn.setAttribute("aria-expanded", String(!isOpen));
  const label = btn.querySelector("span");
  if (label) {
    const showText = label.dataset.show || "Show details";
    const hideText = label.dataset.hide || "Hide details";
    label.textContent = isOpen ? showText : hideText;
  }
}


/* ── Full Table Toggle ───────────────────────────── */

function toggleFullTable(btn) {
  var table = document.getElementById("openset-table");
  if (!table) return;
  var isExpanded = table.classList.toggle("show-all");
  btn.setAttribute("aria-pressed", String(isExpanded));
  var label = btn.querySelector("span");
  if (label) label.textContent = isExpanded ? "Fewer columns" : "Show all columns";
}


/* ── Event Listeners ──────────────────────────────── */

document.addEventListener("click", (e) => {
  const container = document.querySelector(".more-works-container");
  if (!container || container.contains(e.target)) return;
  setMoreWorksOpen(false);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setMoreWorksOpen(false);
});

window.addEventListener("scroll", () => {
  const btn = document.querySelector(".scroll-to-top");
  if (!btn) return;
  btn.classList.toggle("visible", window.scrollY > 400);
}, { passive: true });


/* ── Video Carousel ──────────────────────────────── */

var carouselStates = {};

function getCarousel(carouselRef) {
  if (carouselRef && carouselRef.nodeType === 1) return carouselRef;
  if (typeof carouselRef === "string") return document.getElementById(carouselRef);
  return document.getElementById("videoCarousel");
}

function getCarouselState(carousel) {
  var id = carousel.id || "carousel-" + Array.prototype.indexOf.call(document.querySelectorAll(".video-carousel"), carousel);
  if (!carouselStates[id]) {
    carouselStates[id] = {
      current: 0,
      timer: null,
      cleanup: null
    };
  }
  return carouselStates[id];
}

function syncVideoPair(slide) {
  var videos = slide.querySelectorAll("video");
  if (videos.length !== 2) return;
  var ready = 0;
  var check = function () {
    if (++ready < 2) return;
    var d0 = videos[0].duration;
    var d1 = videos[1].duration;
    if (!isFinite(d0) || !isFinite(d1)) return;

    var longer  = d0 >= d1 ? videos[0] : videos[1];
    var shorter = d0 <  d1 ? videos[0] : videos[1];
    videos.forEach(function (v) { v.playbackRate = 1; });

    longer.loop  = false;
    shorter.loop = false;

    var restarting = false;
    var restartPair = function () {
      if (restarting) return;
      restarting = true;
      longer.currentTime  = 0;
      shorter.currentTime = 0;
      longer.play();
      shorter.play();
      setTimeout(function () { restarting = false; }, 100);
    };

    longer.addEventListener("ended", restartPair);
    shorter.addEventListener("ended", restartPair);
    restartPair();
  };
  videos.forEach(function (v) {
    if (v.readyState >= 1) check();
    else v.addEventListener("loadedmetadata", check, { once: true });
  });
}

document.querySelectorAll(".video-carousel .carousel-slide").forEach(syncVideoPair);

function carouselFinish(carouselRef) {
  var carousel = getCarousel(carouselRef);
  if (!carousel) return;
  var state = getCarouselState(carousel);
  if (state.cleanup) {
    state.cleanup();
    state.cleanup = null;
  }
}

function carouselGoTo(idx, carouselRef) {
  var carousel = getCarousel(carouselRef);
  if (!carousel) return;
  var state = getCarouselState(carousel);
  var slides = carousel.querySelectorAll(".carousel-slide");
  var dots = carousel.querySelectorAll(".carousel-dot");
  if (!slides.length) return;

  var newIdx = ((idx % slides.length) + slides.length) % slides.length;
  if (newIdx === state.current) return;

  carouselFinish(carousel);

  var forward = newIdx > state.current;
  if (state.current === slides.length - 1 && newIdx === 0) forward = true;
  if (state.current === 0 && newIdx === slides.length - 1) forward = false;

  var oldSlide = slides[state.current];
  var newSlide = slides[newIdx];
  var animClasses = ["exit-left", "exit-right", "enter-left", "enter-right"];

  oldSlide.classList.remove("active");
  oldSlide.classList.add(forward ? "exit-left" : "exit-right");
  newSlide.classList.add("active", forward ? "enter-right" : "enter-left");

  state.cleanup = function () {
    animClasses.forEach(function (c) {
      oldSlide.classList.remove(c);
      newSlide.classList.remove(c);
    });
    state.cleanup = null;
  };
  newSlide.addEventListener("animationend", state.cleanup, { once: true });

  if (dots[state.current]) dots[state.current].classList.remove("active");
  if (dots[newIdx]) dots[newIdx].classList.add("active");

  state.current = newIdx;

  newSlide.querySelectorAll("video").forEach(function (v) {
    v.currentTime = 0;
    v.play();
  });

  resetCarouselTimer(carousel);
}

function carouselNext(carouselRef) {
  var carousel = getCarousel(carouselRef);
  if (!carousel) return;
  var state = getCarouselState(carousel);
  carouselGoTo(state.current + 1, carousel);
}

function carouselPrev(carouselRef) {
  var carousel = getCarousel(carouselRef);
  if (!carousel) return;
  var state = getCarouselState(carousel);
  carouselGoTo(state.current - 1, carousel);
}

function resetCarouselTimer(carouselRef) {
  var carousel = getCarousel(carouselRef);
  if (!carousel) return;
  var state = getCarouselState(carousel);
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(function () { carouselNext(carousel); }, 7000);
}

document.querySelectorAll(".video-carousel").forEach(resetCarouselTimer);

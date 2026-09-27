function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

window.addEventListener("scroll", function () {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (!scrollButton) return;

  if (window.pageYOffset > 300) {
    scrollButton.classList.add("visible");
  } else {
    scrollButton.classList.remove("visible");
  }
});

// Scale each complete table into its available width, with captions left readable.
function initializeFittedTables() {
  const viewports = [...document.querySelectorAll(".table-viewport")];
  let frame;

  function fitTable(viewport) {
    const availableWidth = viewport.getBoundingClientRect().width;
    if (!availableWidth) return; // A table inside a closed details element.

    const table = viewport.querySelector("table");
    table.style.transform = "none";
    const naturalWidth = Math.max(table.scrollWidth, table.getBoundingClientRect().width);
    const scale = Math.min(1, availableWidth / naturalWidth);
    table.style.transform = `scale(${scale})`;
    viewport.style.height = `${Math.ceil(table.getBoundingClientRect().height)}px`;
  }

  function scheduleFit() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => viewports.forEach(fitTable));
  }

  if ("ResizeObserver" in window) {
    const widths = new WeakMap();
    const observer = new ResizeObserver(entries => {
      for (const { target, contentRect } of entries) {
        if (widths.get(target) !== contentRect.width) {
          widths.set(target, contentRect.width);
          fitTable(target);
        }
      }
    });
    viewports.forEach(viewport => observer.observe(viewport));
  }

  window.addEventListener("resize", scheduleFit);
  document.querySelectorAll("details").forEach(details => {
    details.addEventListener("toggle", scheduleFit);
  });
  if (document.fonts) document.fonts.ready.then(scheduleFit);
  scheduleFit();
}

initializeFittedTables();

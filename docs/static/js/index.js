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

const header = document.querySelector('.sm-home .sm-header');
const brand = header?.querySelector('.sm-brand');
const titleEnd = document.querySelector('.sm-hero .english');
if (header && brand && titleEnd) {
  let queued = false;
  function update() {
    queued = false;
    const visible = titleEnd.getBoundingClientRect().bottom <= header.getBoundingClientRect().bottom;
    document.body.classList.toggle('sm-brand-scrolled', visible);
    brand.setAttribute('aria-hidden', String(!visible));
    brand.tabIndex = visible ? 0 : -1;
  }
  function schedule() {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }
  addEventListener('scroll', schedule, {passive: true});
  addEventListener('resize', schedule);
  addEventListener('pageshow', schedule);
  new ResizeObserver(schedule).observe(titleEnd);
  update();
}

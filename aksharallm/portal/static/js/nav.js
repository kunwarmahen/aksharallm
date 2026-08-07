/* The view menu: a left drawer that is closed until you ask for it.
 *
 * This module owns opening and closing only. *Which* view is on screen is still the
 * router's business, and the buttons inside the drawer are the same `.tab` elements the
 * router has always marked — so nothing here has to know the list of views, and adding a
 * view is still an entry in index.html plus a registerTab call in its own module.
 *
 * An overlay that covers the page has to be honest about it: while it is open the rest of
 * the document is `inert`, so Tab cannot walk behind the scrim and a screen reader does not
 * read out a page the reader cannot reach. While it is closed the drawer itself is inert,
 * which is what keeps fourteen off-canvas buttons out of the tab order. */

import { $, $$ } from './core.js';

/* Everything the drawer covers. Marked inert while it is open — the drawer and its scrim
 * are deliberately not in this list. */
const BEHIND = () => [$('.topbar'), $('.footer'), ...$$('.view')].filter(Boolean);

let opener = null;   /* what to give focus back to on close */

export function isNavOpen() {
  return document.body.classList.contains('nav-open');
}

export function openNav() {
  if (isNavOpen()) return;
  opener = document.activeElement;
  document.body.classList.add('nav-open');
  $('#nav-open').setAttribute('aria-expanded', 'true');
  $('#nav').inert = false;
  for (const el of BEHIND()) el.inert = true;
  /* Open with the current view under the cursor: it is both the answer to "where am I" and
   * the most likely thing to be pressed. Fourteen entries can outgrow a short window, so it
   * is scrolled into the drawer's own scroll box as well. */
  const cur = $('.tab.on') || $('.tab');
  if (cur) {
    /* The class above is what makes the drawer visible, and focus() on something still
     * computing as `visibility: hidden` is a no-op. Reading a layout property forces the
     * recalc to happen now rather than at the next frame. */
    void $('#nav').offsetWidth;
    cur.scrollIntoView({ block: 'nearest' });
    cur.focus({ preventScroll: true });
  }
}

export function closeNav() {
  if (!isNavOpen()) return;
  document.body.classList.remove('nav-open');
  $('#nav-open').setAttribute('aria-expanded', 'false');
  for (const el of BEHIND()) el.inert = false;
  /* Focus has to leave the drawer before the drawer becomes inert, or the browser drops it
   * on <body> and the next Tab starts from the top of the page. */
  if (opener && document.contains(opener)) opener.focus({ preventScroll: true });
  else $('#nav-open').focus({ preventScroll: true });
  opener = null;
  $('#nav').inert = true;
}

export function toggleNav() {
  if (isNavOpen()) closeNav();
  else openNav();
}

export function wireNav() {
  $('#nav').inert = true;             /* closed at boot; the CSS agrees */
  $('#nav-open').addEventListener('click', toggleNav);
  $('#nav-close').addEventListener('click', closeNav);
  $('#nav-scrim').addEventListener('click', closeNav);
  /* Escape closes — but not while a dialog is up, which has its own Escape. */
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isNavOpen() && !document.querySelector('dialog[open]')) {
      e.preventDefault();
      closeNav();
    }
  });
}

/* Which tab is on screen, and the hash/localStorage plumbing that survives a reload.
 *
 * The router knows the list of views but nothing about what any of them do: each tab module
 * calls registerTab() with what to run when it is opened and when it is left. Adding a tab
 * is a registerTab call in that tab's module — there is nothing to edit here. */

import { $, $$ } from './core.js';
import { state } from './state.js';

export const VIEWS = ['dashboard', 'play', 'code', 'quant', 'lora', 'evals', 'synth', 'docs'];

/* view -> { open, leave }. A tab registers itself when its module is first imported, so the
 * router never has to name one. `open` runs every time the tab is shown — the tab modules
 * decide for themselves whether that means loading anything. `leave` runs when the reader
 * goes elsewhere, and is where a tab stops its own polling. */
const TABS = {};

export function registerTab(view, { open, leave } = {}) {
  TABS[view] = { open, leave };
}

export function showView(view) {
  state.view = view;
  for (const v of VIEWS) {
    $(`#view-${v}`).hidden = view !== v;
    /* Not every view has a footer of its own — the docs tab reads fine without one. */
    const foot = $(`.foot-${v}`);
    if (foot) foot.hidden = view !== v;
  }
  /* The run picker and the phase badge belong to the dashboard, not to the chrome. */
  $('#run-field').hidden = view !== 'dashboard';
  $('#phase').hidden = view !== 'dashboard';

  for (const tab of $$('.tab')) {
    const on = tab.dataset.view === view;
    tab.classList.toggle('on', on);
    if (on) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
    // On a phone the strip is narrower than its six tabs and scrolls sideways, so the
    // current view can sit off-screen — including on load, from a #hash or localStorage.
    if (on) tab.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  localStorage.setItem('aksharallm-view', view);
  /* The tab in the address bar, so a view can be linked to and reloaded into. Written
   * without a history entry: the back button should leave the portal, not walk back
   * through every tab you glanced at. */
  if (location.hash.slice(1) !== view) {
    history.replaceState(null, '', `#${view}`);
  }
  /* Leave before open, so a tab that polls has stopped before the next one starts. */
  for (const [v, tab] of Object.entries(TABS)) {
    if (v !== view && tab.leave) tab.leave();
  }
  if (TABS[view] && TABS[view].open) TABS[view].open();
}

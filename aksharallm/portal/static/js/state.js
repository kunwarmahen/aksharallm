/* The one piece of state more than one tab cares about: which run is selected, and what
 * the last poll returned for it. The dashboard writes most of it, the router writes `view`,
 * and the playground reads `run` to know which checkpoint it should be talking about. */

export const state = {
  run: null,
  status: null,
  schedule: null,
  gpu: null,
  gpuWindow: '3600',
  cost: null,        // the energy ledger's totals, priced
  log: null,
  logFile: null,     // null = whichever file was written most recently
  timer: null,
  charts: {},        // last spec per chart, so a resize can redraw without a fetch
  zoom: {},          // per chart: the dragged-out { x0, x1 } window, or absent for all of it
  busy: false,
  view: 'dashboard', // 'dashboard' | 'code'
  budget: null,      // {unit:'after'|'in', value} for the next launch; null = the whole budget
};

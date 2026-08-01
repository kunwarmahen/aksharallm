/* Wiring and boot: bind every control to its handler, then open on the run that most wants
 * looking at. The only module with side effects at import time — importing it starts the
 * portal, and importing the tab modules is what registers them with the router. */

import { $, $$, api, flash, fmt, live, post } from './core.js';
import { state } from './state.js';
import { VIEWS, showView } from './router.js';
import { UNIT_MINUTES, UNIT_MORE_STEPS, act, boundPicker, drawCharts, fmtMins, fmtWhen, refresh, renderRuns, renderSessionBudget, schedule, secPerStep, selectRun, wireGpu, wireSchedule } from './dashboard.js';
import { wireCode } from './code.js';
import { wireQuantTab } from './quantize.js';
import { wireLoraTab } from './lora.js';
import { wireEvalTab, drawTrend } from './evals.js';
import { wireSynthTab } from './synth.js';
import { wireLearnTab } from './learn.js';
import { wirePlay } from './play.js';
import './docs.js';

function wire() {
  wireGpu();
  wireSchedule();
  wireCode();
  wirePlay();
  wireQuantTab();
  wireLoraTab();
  wireEvalTab();
  wireSynthTab();
  wireLearnTab();
  boundPicker.wire();
  for (const tab of $$('.tab')) {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  }
  $('#run-select').addEventListener('change', (e) => selectRun(e.target.value));

  $('#btn-start').addEventListener('click', () => {
    const s = state.status || {};
    /* Starting a finished run means archiving it and beginning again from step 0 — a
     * different thing from resuming, so it is confirmed and says what happens to the old
     * one. Nothing is deleted here; the archive keeps every checkpoint and every log. */
    if (s.can_restart) {
      const best = s.last && s.last.best_val != null ? `, best val ${fmt.num(s.last.best_val, 4)}` : '';
      if (!confirm(`'${state.run}' has trained its whole budget `
        + `(${fmt.int(s.max_steps)} steps${best}).\n\n`
        + `Start a NEW run from step 0?\n\n`
        + `The finished one is archived under a timestamped name — every checkpoint and log `
        + `kept, still readable from the run picker. Nothing is deleted.`)) return;
    }
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/start`, {
      stop_after: state.budget && state.budget.unit === 'after' ? state.budget.value : null,
      stop_after_s: state.budget && state.budget.unit === 'in' ? state.budget.value : null,
      skip_smoke: $('#skip-smoke').checked,
      fresh: !!s.can_restart,
    }), s.can_restart ? 'Archiving, then launching.' : 'Launching.');
  });

  /* Delete: the only control here that destroys anything. The dialog names what goes, what
   * stays and how big it is, and the API is sent the run's name back as the confirmation —
   * so a request that never went through this dialog cannot delete anything either. */
  $('#btn-delete').addEventListener('click', async () => {
    const s = state.status || {};
    const run = state.run;
    const lines = [
      `Delete '${run}' permanently?`,
      '',
      `  checkpoints/${run}/  and  logs/${run}/`,
      `  ${fmt.bytes(s.size_bytes)}${s.step == null ? '' : `, ${fmt.int(s.step + 1)} steps trained`}`
        + (s.last && s.last.best_val != null ? `, best val ${fmt.num(s.last.best_val, 4)}` : ''),
      '',
      s.archived
        ? 'This is an archive of a finished run. Its history goes with it.'
        : s.has_config
          ? `configs/${run}.yaml is kept, so the run can be started again from scratch.`
          : 'There is no config for this run, so nothing of it remains afterwards.',
      '',
      'This cannot be undone.',
    ];
    if (!confirm(lines.join('\n'))) return;
    try {
      const res = await post(`/api/run/${encodeURIComponent(run)}/delete`, { confirm: run });
      /* The selected run no longer exists; fall back to whatever is left. `selectRun` clears
       * the flash, so the message goes up after the switch, not before it. */
      const { runs } = await api('/api/runs');
      renderRuns(runs);
      if (runs.length) selectRun((runs.find((r) => r.run !== run) || runs[0]).run);
      flash(res.note || `Deleted ${run}.`, 'ok');
    } catch (err) {
      flash(err.message, 'error');
    }
  });

  /* The session budget uses the same picker as the stop, with one extra unit for "no
   * bound at all" — which is the default, and has to stay one click away. */
  $('#btn-budget').addEventListener('click', async () => {
    const s = state.status || {};
    const sps = secPerStep(s);
    const from = (s.step == null ? 0 : s.step + 1);
    const chosen = await boundPicker.open({
      title: 'Budget for this session',
      sub: `'${state.run}' resumes at step ${fmt.int(from)}. Whatever you pick here, the `
        + 'trainer saves and exits when it lands, and the next start carries on from there.',
      okLabel: 'Set budget',
      unit: (state.budget && state.budget.unit) || 'none',
      units: [
        { ...UNIT_MORE_STEPS, label: 'Steps', noun: 'steps this session',
          value: (state.budget && state.budget.unit === 'after') ? state.budget.value : 500,
          preview: (v) => `finishes step ${fmt.int(from + v - 1)}`
            + (sps ? ` · about ${fmt.dur(v * sps)} at the last measured ${sps.toFixed(2)}s/step` : '')
            + (s.max_steps && from + v - 1 >= s.max_steps - 1 ? ' — that is the whole budget' : '') },
        { ...UNIT_MINUTES, label: 'Time', noun: 'minutes this session',
          value: (state.budget && state.budget.unit === 'in') ? state.budget.value : 30 * 60,
          preview: (v) => `trains for ${fmtMins(Math.round(v / 60))} from the first step`
            + (sps ? ` · roughly ${fmt.int(v / sps)} steps at ${sps.toFixed(2)}s/step` : '')
            + ' · pre-flight and compilation are not counted' },
        { id: 'none', label: 'Whole budget', none: true,
          preview: () => 'runs to the config\'s own max_steps, or until you stop it' },
      ],
    });
    if (chosen === null) return;
    state.budget = chosen.unit === 'none' ? null : chosen;
    renderSessionBudget();
  });

  // Post-training panel: one delegated handler for all SFT/DPO/GRPO start/stop buttons.
  $('#pipeline-stages').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action]');
    if (!btn || btn.disabled) return;
    const { base, stage, action } = btn.dataset;
    if (action === 'stop' && !confirm(`Stop '${base} · ${stage}'?`)) return;
    act(() => post(`/api/pipeline/${encodeURIComponent(base)}/${stage}/${action}`, {}),
      action === 'start' ? `Starting ${stage.toUpperCase()}.` : 'Stop requested.');
  });

  $('#btn-stop').addEventListener('click', () => {
    const s = state.status || {};
    const msg = s.phase === 'launching'
      ? `Abort the launch of '${state.run}'?\n\nIt is in pre-flight `
        + `(${(s.launcher && s.launcher.stage) || '?'}); nothing has trained yet, so nothing `
        + 'is lost. You would press Start again to relaunch.'
      : `Stop '${state.run}' after the current step?\n\n`
        + `It saves ckpt_last.pt at step ~${fmt.int(s.step)} and exits; starting again `
        + 'resumes there with no loss spike.';
    if (!confirm(msg)) return;
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/stop`, { mode: 'now' }),
      s.phase === 'launching' ? 'Aborting.' : 'Stop requested.');
  });

  $('#btn-stop-at').addEventListener('click', async () => {
    const s = state.status || {};
    const cur = s.step || 0;
    const sps = secPerStep(s);
    const every = (s.config && s.config.ckpt_every) || null;
    /* How many periodic checkpoints land before the stop — the honest cost of stopping
     * later rather than sooner, and the number that decides whether it is worth waiting. */
    const ckpts = (steps) => (every ? ` · ${Math.floor(((cur % every) + steps) / every)} `
      + `checkpoint${Math.floor(((cur % every) + steps) / every) === 1 ? '' : 's'} on the way` : '');
    const chosen = await boundPicker.open({
      title: `Stop '${state.run}' at…`,
      sub: `It is at step ${fmt.int(cur)}. It finishes the step it lands on, saves `
        + 'ckpt_last.pt there and exits — starting again resumes from it with no loss spike.',
      unit: 'in',
      units: [
        { ...UNIT_MINUTES,
          preview: (v) => `stops about ${fmtWhen(v)}`
            + (sps ? ` · around step ${fmt.int(cur + v / sps)}${ckpts(v / sps)}` : '') },
        { ...UNIT_MORE_STEPS,
          preview: (v) => `finishes step ${fmt.int(cur + v)}`
            + (sps ? ` · about ${fmt.dur(v * sps)}, so around ${fmtWhen(v * sps)}` : '')
            + ckpts(v) },
      ],
    });
    if (!chosen) return;
    const body = chosen.unit === 'in'
      ? { mode: 'in', seconds: chosen.value }
      : { mode: 'after', steps: chosen.value };
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/stop`, body), 'Queued.');
  });

  $('#btn-cancel-stop').addEventListener('click', () => {
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/stop`, { mode: 'cancel' }),
      'Cancelled.');
  });

  $('#log-select').addEventListener('change', (e) => {
    state.logFile = e.target.value || null;
    schedule(0);
  });

  /* Each chart card carries its own chart/table switch — the table view is the WCAG-clean
   * twin, not an afterthought. */
  for (const btn of $$('.view-toggle')) {
    btn.addEventListener('click', () => {
      const key = btn.dataset.target;
      const chart = $(`.chart[data-chart="${key}"]`);
      const tbl = $(`.chart-table[data-table="${key}"]`);
      const toTable = !chart.hidden;
      chart.hidden = toTable;
      tbl.hidden = !toTable;
      btn.textContent = toTable ? 'chart' : 'table';
      drawCharts();
    });
  }

  /* Theme: follow the OS by default, with an explicit override that wins both ways. */
  const themes = ['auto', 'light', 'dark'];
  const applyTheme = (t) => {
    document.documentElement.dataset.theme = t;
    $('#theme-label').textContent = t;
    localStorage.setItem('aksharallm-theme', t);
  };
  applyTheme(localStorage.getItem('aksharallm-theme') || 'auto');
  $('#theme').addEventListener('click', () => {
    const now = document.documentElement.dataset.theme || 'auto';
    applyTheme(themes[(themes.indexOf(now) + 1) % themes.length]);
    drawCharts();
  });

  /* Charts are sized from their container, so a resize needs a redraw, not a refetch. The
   * Eval tab's trend chart is drawn by the same rule and from the same cached data. */
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { drawCharts(); drawTrend(); }, 150);
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) schedule(0);
  });
}

/* The top bar is sticky, and how tall it is depends on how many rows it wrapped into —
 * one on a desktop, two on a phone. Anything that sticks below it has to clear it, so the
 * measured height is published as a custom property instead of being guessed at in CSS. */
function trackTopbarHeight() {
  const bar = $('.topbar');
  const publish = () => document.documentElement.style
    .setProperty('--topbar-h', `${Math.round(bar.getBoundingClientRect().height)}px`);
  publish();
  if (window.ResizeObserver) new ResizeObserver(publish).observe(bar);
  else window.addEventListener('resize', publish);
}

async function boot() {
  wire();
  trackTopbarHeight();
  /* A #hash wins over the remembered tab: an explicit link should land where it says. */
  const asked = location.hash.slice(1);
  showView(VIEWS.includes(asked) ? asked
    : (localStorage.getItem('aksharallm-view') || 'dashboard'));
  window.addEventListener('hashchange', () => {
    const want = location.hash.slice(1);
    if (VIEWS.includes(want) && want !== state.view) showView(want);
  });
  live('connecting…');
  try {
    const { runs, root } = await api('/api/runs');
    $('#foot-root').textContent = root;
    if (!runs.length) {
      live('no runs found', 'err');
      flash('No runs found under this repo: expected configs/*.yaml or checkpoints/<run>/.',
        'error');
      return;
    }
    /* Open on whatever is actually happening; otherwise the furthest-along run. */
    const busy = runs.find((r) => r.phase !== 'idle');
    const best = busy || runs.slice().sort((a, b) => (b.updated || -1) - (a.updated || -1))[0];
    state.run = best.run;
    renderRuns(runs);
    await refresh();
  } catch (err) {
    live(err.message, 'err');
    flash(`Cannot reach the portal API: ${err.message}`, 'error');
  }
}

boot();

/* The dashboard: the render passes that turn one /api/status poll into the phase badge,
 * progress, tiles, charts, sessions, pipeline, config, log, GPU panel and schedule — plus the
 * poll loop that drives them and the bound picker that says how far the next launch should go. */

import { $, $$, api, escHtml, flash, fmt, live, post } from './core.js';
import { state } from './state.js';
import { chartTable, lineChart, table } from './charts.js';
import { renderMarkdown } from './markdown.js';
import { registerTab } from './router.js';

/** What a queued stop says on the badge: a step, or a clock time with how long is left. */
export function stopLabel(stop) {
  if (!stop || stop.now) return '';
  if (stop.deadline != null) {
    const left = stop.deadline - Date.now() / 1000;
    return ` · stops ${fmtWhen(Math.max(0, left))}`
      + (left > 0 ? ` (${fmt.dur(left)} left)` : '');
  }
  return ` · stop queued at ${fmt.int(stop.target)}`;
}

function renderPhase(s) {
  const badge = $('#phase');
  const queued = stopLabel(s.stop);
  const stage = s.launcher && s.launcher.stage ? ` · ${s.launcher.stage}` : '';
  const label = {
    training: 'training',
    launching: 'pre-flight',
    stopping: 'stopping',
    idle: 'idle',
  }[s.phase] || s.phase;
  badge.textContent = label + stage + (s.pid ? ` · pid ${s.pid}` : '')
    + (s.launcher && !s.pid ? ` · pid ${s.launcher.pid}` : '') + queued;
  badge.className = `badge badge-${s.phase}`;
}

function renderControls(s) {
  /* Three states for one button: start, resume, or set the finished run aside and begin
   * again. A finished experiment is not a dead end — running it a second time is what you
   * do with an experiment — so the button changes verb rather than switching off. */
  $('#btn-start').disabled = !(s.can_start || s.can_restart) || state.busy;
  $('#btn-start').title = s.start_hint || 'runs scripts/phase2.sh: pre-flight, data check, '
    + 'smoke test, then the real run (resumes from ckpt_last.pt)';
  $('#btn-stop').disabled = !s.can_stop || state.busy;
  /* A bounded stop needs a step (or a clock) to count from, which a pre-flight has neither
   * of yet — there is no trainer to read the file. */
  $('#btn-stop-at').disabled = !s.can_bound || state.busy;
  $('#btn-stop').textContent = s.phase === 'launching' ? 'Abort launch' : 'Stop now';
  $('#btn-cancel-stop').disabled = !s.stop || state.busy;
  $('#btn-start').textContent = s.can_restart ? 'Start fresh…'
    : s.finished ? 'Budget spent'
    : s.step == null ? 'Start run' : `Resume from ${fmt.int(s.step + 1)}`;
  $('#btn-budget').disabled = !(s.can_start || s.can_restart) || state.busy;

  const del = $('#btn-delete');
  del.disabled = !s.can_delete || state.busy;
  del.title = s.can_delete
    ? `Remove checkpoints/${s.run}/ and logs/${s.run}/ (${fmt.bytes(s.size_bytes)}). `
      + (s.archived ? 'This is an archive; nothing else refers to it.'
        : `configs/${s.run}.yaml is kept, so the run can be started again from scratch.`)
    : 'a live run cannot be deleted — stop it first';
  renderSessionBudget();
}

/** The Start button's companion: what the next session is bounded by, in its own words.
 *
 * Not `renderBudget` — that name is taken by the Finetune tab's memory-budget table, and a
 * second declaration of it silently replaces the first for the whole file. */
export function renderSessionBudget() {
  const b = state.budget;
  $('#budget-label').textContent = !b ? 'to budget'
    : b.unit === 'in' ? fmtMins(Math.round(b.value / 60))
    : `${fmt.int(b.value)} steps`;
  $('#btn-budget').classList.toggle('is-set', !!b);
}

function renderProgress(s) {
  const last = s.last || {};
  const pct = s.progress == null ? null : Math.min(1, s.progress);
  $('#hero-step').textContent = s.step == null ? 'no steps logged'
    : `step ${fmt.int(s.step)}${s.max_steps ? ` / ${fmt.int(s.max_steps)}` : ''}`;
  $('#hero-sub').textContent = s.phase === 'launching'
    ? `pre-flight (${(s.launcher && s.launcher.stage) || '?'}) — tests, data check and a `
      + '50-step smoke test run before training starts'
    : s.finished
    ? `finished — all ${fmt.int(s.max_steps)} steps trained; raise train.max_steps to go further`
    : s.step == null
    ? (s.can_start ? 'ready to start' : 'nothing logged for this run yet')
    : [
      pct == null ? null : `${fmt.pct(pct)} of the budget`,
      s.tokens_seen ? `${fmt.compact(s.tokens_seen)} tokens seen` : null,
      last.step_time ? `last step logged ${fmt.ago(last.step_time)}` : null,
      s.phase === 'idle' ? `resumes at step ${fmt.int(s.step + 1)}` : null,
    ].filter(Boolean).join(' · ');

  /* ETA is only meaningful while stepping; a stale one from last night is a lie. */
  $('#eta').textContent = s.phase === 'idle' || last.eta_s == null ? '–' : fmt.dur(last.eta_s);
  $('#meter-fill').style.width = `${(pct || 0) * 100}%`;

  /* The finish line on the meter. A step target is exact; a deadline has to be projected
   * through the last measured s/step, so it is drawn dashed and says "about" — the mark
   * would otherwise claim a precision the clock cannot give it. */
  const mark = $('#meter-stop');
  const sps = secPerStep(s);
  const at = !s.stop || s.stop.now ? null
    : s.stop.target != null ? s.stop.target
    : (sps && s.step != null && s.stop.deadline != null)
      ? s.step + Math.max(0, s.stop.deadline - Date.now() / 1000) / sps
      : null;
  if (at != null && s.max_steps) {
    mark.hidden = false;
    mark.classList.toggle('is-estimate', s.stop.target == null);
    mark.style.left = `calc(${Math.min(100, (at / s.max_steps) * 100)}% - 1px)`;
    mark.title = s.stop.target != null
      ? `queued stop at step ${fmt.int(at)}`
      : `queued stop at ${fmtWhen(s.stop.deadline - Date.now() / 1000)} — about step ${fmt.int(at)}`;
  } else {
    mark.hidden = true;
  }

  $('#meter-left').textContent = s.max_steps
    ? `${fmt.int(s.step ?? 0)} of ${fmt.int(s.max_steps)} steps`
    : `${fmt.int(s.step ?? 0)} steps`;
  $('#meter-right').textContent = s.max_steps && s.tokens_per_step
    ? `budget ${fmt.compact(s.max_steps * s.tokens_per_step)} tokens`
    : '';
}

function renderTiles(s) {
  const l = s.last || {};
  const set = (id, value, note) => {
    $(`#t-${id}`).textContent = value;
    $(`#t-${id}-note`).textContent = note;
  };
  set('ema', fmt.num(l.ema, 3), l.loss == null ? '–'
    : `raw ${fmt.num(l.loss, 3)} · ppl ${fmt.num(Math.exp(Math.min(l.ema ?? 20, 20)), 1)}`);
  set('val', fmt.num(l.best_val, 4), l.val_step == null ? 'no eval yet'
    : `latest ${fmt.num(l.val_loss, 4)} at step ${fmt.int(l.val_step)}`);
  set('tok', l.tok_per_sec == null ? '–' : `${(l.tok_per_sec / 1000).toFixed(1)}k/s`,
    l.s_per_step == null ? '–' : `${fmt.num(l.s_per_step, 2)}s per step`);
  set('mfu', fmt.pct(l.mfu, 1), 'of the GPU’s peak bf16');
  set('tokens', fmt.compact(s.tokens_seen),
    s.tokens_per_step ? `${fmt.compact(s.tokens_per_step)} per step` : '–');
  set('up', s.uptime_s == null ? '–' : fmt.dur(s.uptime_s),
    `${(s.sessions || []).length} session${(s.sessions || []).length === 1 ? '' : 's'} logged`);
}

function renderCharts(s) {
  const ser = s.series || {};
  const step = ser.step || [];
  const clip = s.config && s.config.grad_clip;

  state.charts = {
    ...state.charts,
    loss: {
      label: 'training and validation loss by step',
      yFmt: (v) => v.toFixed(2),
      series: [
        { name: 'loss (per log step)', color: '--ink-muted', x: step, y: ser.loss || [], faint: true, fmt: (v) => v.toFixed(4) },
        { name: 'loss (ema)', color: '--series-1', x: step, y: ser.ema || [], label: true, fmt: (v) => v.toFixed(4) },
        { name: 'validation loss', color: '--series-2', x: ser.val_step || [], y: ser.val_loss || [], dots: true, fmt: (v) => v.toFixed(4) },
      ],
    },
    tok: {
      label: 'throughput in thousands of tokens per second',
      yFmt: (v) => (v / 1000).toFixed(0) + 'k',
      series: [{ name: 'tokens/sec', color: '--series-1', x: step, y: ser.tok_per_sec || [], label: true, fmt: (v) => (v / 1000).toFixed(1) + 'k' }],
      zeroFloor: true,
    },
    gnorm: {
      label: 'gradient norm by step',
      yFmt: (v) => v.toFixed(2),
      series: [{ name: 'grad norm', color: '--series-1', x: step, y: ser.grad_norm || [], label: true, fmt: (v) => v.toFixed(3) }],
      rules: clip ? [{ y: clip, label: `clip ${clip}` }] : [],
      zeroFloor: true,
    },
    lr: {
      label: 'learning rate by step',
      yFmt: (v) => v.toExponential(1),
      series: [{ name: 'learning rate', color: '--series-1', x: step, y: ser.lr || [], label: true, fmt: fmt.exp }],
      zeroFloor: true,
    },
  };

  /* Mixture of experts only. A dense run has no `moe_shares` and the card stays hidden —
   * rather than showing an empty chart that reads as a broken one. */
  const nExperts = ser.moe_experts || 0;
  const card = $('#card-moe');
  if (card) card.hidden = !nExperts;
  if (nExperts) {
    const mstep = ser.moe_step || [];
    const even = 1 / nExperts;
    const last = (ser.moe_balance || []).filter((v) => v != null).slice(-1)[0];
    const dead = (ser.moe_dead || []).filter((v) => v != null).slice(-1)[0] || 0;
    $('#moe-note').textContent =
      `${nExperts} experts · an equal share is ${(100 * even).toFixed(1)}% (the rule)`
      + (last == null ? '' : ` · balance ${last.toFixed(2)}`)
      + (dead ? ` · ${dead} expert${dead === 1 ? '' : 's'} receiving almost nothing` : '');
    state.charts.moe = {
      label: 'share of routed tokens per expert, by step',
      yFmt: (v) => `${(100 * v).toFixed(0)}%`,
      /* One line per expert, all the same colour family: the question is never "which
       * expert is which", it is "are they together or is one running away". */
      series: (ser.moe_shares || []).map((y, i) => ({
        name: `expert ${i}`,
        color: `--series-${(i % 5) + 1}`,
        x: mstep, y,
        fmt: (v) => `${(100 * v).toFixed(1)}%`,
      })),
      rules: [{ y: even, label: 'even' }],
      zeroFloor: true,
    };
  } else {
    delete state.charts.moe;
  }
  drawCharts();
}

export function drawCharts() {
  for (const key of Object.keys(state.charts)) drawChart(key);
}

/** One chart. The zoom window lives in state, keyed by chart, so the five-second poll
 * redraws into the window the reader chose instead of snapping back to the whole run. */
function drawChart(key) {
  const spec = state.charts[key];
  const host = $(`.chart[data-chart="${key}"]`);
  const tableHost = $(`.chart-table[data-table="${key}"]`);
  if (!spec || !host || host.dataset.dragging) return;
  /* The table twin always lists every reading — it is the accessible path to the data,
   * and a zoom is a way of looking, not a filter. */
  if (host.hidden) { chartTable(tableHost, spec); return; }
  lineChart(host, {
    ...spec,
    zoom: state.zoom[key],
    onZoom: (win) => {
      if (win) state.zoom[key] = win; else delete state.zoom[key];
      drawChart(key);
    },
  });
}

function renderSessions(s) {
  const rows = (s.sessions || []).slice().reverse().map((x) => [
    `#${x.index}`,
    x.started || '?',
    x.first_step == null ? '–' : `${fmt.int(x.first_step)} → ${fmt.int(x.last_step)}`,
    x.ema_first == null ? '–' : `${fmt.num(x.ema_first, 3)} → ${fmt.num(x.ema_last, 3)}`,
    x.best_val == null ? '–' : fmt.num(x.best_val, 4),
    x.tok_per_sec == null ? '–' : `${(x.tok_per_sec / 1000).toFixed(1)}k`,
    fmt.dur(x.wall_s),
    x.ended || (x.open && s.pid && x.index === s.sessions.length ? 'running now'
      : x.unmarked ? 'before session markers' : 'no end record (killed or crashed)'),
  ]);
  const host = $('#sessions');
  $('#sessions-count').textContent = rows.length
    ? `${rows.length} session${rows.length === 1 ? '' : 's'} · newest first`
    : '';
  // The poll re-renders this table; keep where the reader had scrolled to.
  const { scrollTop, scrollLeft } = host;
  host.textContent = '';
  host.appendChild(rows.length
    ? table(['#', 'started', 'steps', 'loss (ema)', 'best val', 'tok/s', 'wall', 'ended'], rows,
      { currentRow: s.pid ? 0 : -1 })
    : Object.assign(document.createElement('div'),
      { className: 'chart-empty', textContent: 'No sessions logged yet.' }));
  host.scrollTop = scrollTop;
  host.scrollLeft = scrollLeft;
}

/** The base run for a stage run: 'small-code-sft' -> 'small-code'. */
function baseOf(run) {
  return (run || '').replace(/-(sft|dpo|grpo)$/, '');
}

/** The post-training panel: SFT -> DPO / GRPO, each gated on its prerequisite checkpoint.
 * Buttons post to /api/pipeline/<base>/<stage>/<action>, which shells out to stage.sh.  A
 * blocked stage's Start is disabled with the reason as its tooltip. */
function renderPipeline(p) {
  const host = $('#pipeline-stages');
  if (!host) return;
  if (!p || !p.stages) { host.innerHTML = ''; return; }
  host.innerHTML = p.stages.map((s) => {
    const m = s.metric || {};
    const val = m.value == null ? '' : (m.key === 'reward'
      ? `reward ${fmt.num(m.value, 3)}` : `val ${fmt.num(m.value, 4)}`);
    const sub = s.step == null ? s.blurb
      : `step ${fmt.int(s.step)}${val ? ` · ${val}` : ''}`;
    const startAttrs = s.can_start ? '' : `disabled title="${escHtml(s.reason || '')}"`;
    return `
      <div class="stage stage-${s.phase}">
        <div class="stage-head">
          <span class="stage-name">${s.stage.toUpperCase()}</span>
          <span class="badge badge-pipe-${s.phase}">${s.phase}</span>
        </div>
        <div class="stage-sub">${escHtml(sub)}</div>
        <div class="stage-actions">
          <button data-base="${escHtml(p.base)}" data-stage="${s.stage}" data-action="start" ${startAttrs}>${s.done ? 'Re-run' : 'Start'}</button>
          <button data-base="${escHtml(p.base)}" data-stage="${s.stage}" data-action="stop" ${s.can_stop ? '' : 'disabled'}>Stop</button>
        </div>
      </div>`;
  }).join('');
}

function renderConfig(s) {
  const c = s.config || {};
  const dl = $('#config');
  dl.textContent = '';
  const add = (k, v) => {
    if (v == null || v === '') return;
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    if (v instanceof Node) dd.appendChild(v); else dd.textContent = v;
    dl.append(dt, dd);
  };
  const code = (t) => Object.assign(document.createElement('code'), { textContent: t });
  add('config', c.path ? code(c.path) : '(no YAML for this run)');
  if (c.error) add('problem', c.error);
  add('architecture', c.arch);
  add('vocab', c.vocab_size == null ? null : fmt.int(c.vocab_size));
  add('batch', c.batch && `${c.batch} = ${fmt.int(c.tokens_per_step)} tokens/step`);
  add('budget', c.max_steps == null ? null
    : `${fmt.int(c.max_steps)} steps = ${fmt.compact(c.max_steps * (c.tokens_per_step || 0))} tokens`);
  add('optimiser', c.lr == null ? null
    : `lr ${fmt.exp(c.lr)} ${c.schedule} · grad clip ${c.grad_clip}`);
  add('cadence', c.eval_every == null ? null
    : `eval every ${fmt.int(c.eval_every)} · checkpoint every ${fmt.int(c.ckpt_every)} steps`);
  add('data', (c.sources || []).filter(Boolean).length ? code((c.sources || []).join('  ')) : null);
  /* Which launcher this run actually uses — there are two now, and naming the wrong one
   * sends someone to a script that does not know how to build their data. */
  add('launch', s.can_start || s.pid || s.finished
    ? code(s.run.startsWith('tiny') ? `scripts/experiment.sh ${s.run}`
      : `scripts/phase2.sh   (run ${s.run})`) : null);

  const rows = (s.checkpoints || []).map((k) => [
    k.name, fmt.bytes(k.size), fmt.clock(k.mtime), fmt.ago(k.mtime)]);
  const host = $('#checkpoints');
  host.textContent = '';
  host.appendChild(rows.length
    ? table(['checkpoint', 'size', 'written', ''], rows)
    : Object.assign(document.createElement('div'),
      { className: 'chart-empty', textContent: 'No checkpoints written yet.' }));
}

function renderLog(log) {
  const sel = $('#log-select');
  const files = log.files || [];
  const wanted = state.logFile;
  if (sel.dataset.run !== state.run || sel.options.length !== files.length + 1) {
    sel.textContent = '';
    sel.appendChild(Object.assign(document.createElement('option'),
      { value: '', textContent: 'newest (auto)' }));
    for (const f of files) {
      sel.appendChild(Object.assign(document.createElement('option'),
        { value: f.name, textContent: `${f.name}  (${fmt.bytes(f.size)})` }));
    }
    sel.dataset.run = state.run;
    sel.value = wanted || '';
  }
  const pre = $('#log');
  const pinned = $('#log-follow').checked;
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 24;
  pre.textContent = (log.lines || []).join('\n') || '(nothing logged yet)';
  if (pinned || atBottom) pre.scrollTop = pre.scrollHeight;
  $('#log-note').textContent = log.file
    ? `${log.file} · ${fmt.bytes(log.size)}${log.truncated ? ' · showing the tail' : ''}`
    : 'no log files for this run yet';
}

export function renderRuns(runs) {
  const sel = $('#run-select');
  /* Archives are labelled, because a picker with `tiny-moe` and `tiny-moe.20260801-105843`
   * side by side and no explanation is a puzzle. */
  const labels = runs.map((r) => r.run
    + (r.phase !== 'idle' ? ` — ${r.phase}`
      : r.archived ? ' — archived' : r.finished ? ' — finished' : ''));
  /* Rebuild only when something actually changed: replacing the options while the select
   * is open would close it under the pointer every poll. */
  const sig = labels.join('|') + '@' + state.run;
  if (sel.dataset.sig === sig) return;
  sel.textContent = '';
  runs.forEach((r, i) => {
    sel.appendChild(Object.assign(document.createElement('option'),
      { value: r.run, textContent: labels[i] }));
  });
  sel.dataset.sig = sig;
  sel.value = state.run;
}

/* ---------------------------------------------------------------- gpu ----------------- */

/** Clock-time x axis: the GPU charts are wall-clock, not step number. */
const timeFmt = (t) => new Date(t * 1000).toLocaleTimeString(undefined,
  { hour: '2-digit', minute: '2-digit' });

const GPU_CHARTS = ['gutil', 'gmem', 'gtemp', 'gpower'];

function renderGpu(gpu) {
  state.gpu = gpu;
  const tiles = ['util', 'mem', 'temp', 'power'];

  if (!gpu.available) {
    $('#gpu-status').textContent = gpu.reason || 'no GPU';
    for (const t of tiles) {
      $(`#g-${t}`).textContent = '–';
      $(`#g-${t}-note`).textContent = '';
    }
    $('#gpu-summary').textContent = '';
    for (const key of GPU_CHARTS) delete state.charts[key];
    for (const key of GPU_CHARTS) {
      const host = $(`.chart[data-chart="${key}"]`);
      if (host) {
        host.textContent = '';
        const div = document.createElement('div');
        div.className = 'chart-empty';
        div.textContent = gpu.reason || 'No GPU telemetry.';
        host.appendChild(div);
      }
    }
    return;
  }

  const dev = (gpu.devices || [])[gpu.index] || (gpu.devices || [])[0] || {};
  const cur = gpu.current || {};
  const memTotal = dev.mem_total || null;
  const limit = dev.power_limit || null;

  $('#gpu-status').textContent = [
    dev.name,
    gpu.sampling ? `sampling every ${gpu.interval_s}s` : 'NOT SAMPLING — no history is being recorded',
    gpu.current_run ? `${gpu.current_run} is training`
      : gpu.current_job ? `${gpu.current_job} job is running` : 'no run training',
    gpu.samples ? `${fmt.int(gpu.samples)} samples in view` : 'no samples yet',
  ].filter(Boolean).join(' · ');

  const set = (id, value, note) => {
    $(`#g-${id}`).textContent = value;
    $(`#g-${id}-note`).textContent = note;
  };
  set('util', cur.util == null ? '–' : `${Math.round(cur.util)}%`,
    gpu.current_age_s == null ? '–' : `${fmt.dur(gpu.current_age_s)} ago`);
  set('mem', cur.mem_used == null ? '–' : `${(cur.mem_used / 1024).toFixed(1)} GB`,
    memTotal ? `of ${(memTotal / 1024).toFixed(0)} GB` : '–');
  set('temp', cur.temp == null ? '–' : `${Math.round(cur.temp)}°C`,
    (gpu.summary.training && gpu.summary.training.temp_max)
      ? `peak ${Math.round(gpu.summary.training.temp_max)}°C while training` : 'idle');
  set('power', cur.power == null ? '–' : `${Math.round(cur.power)} W`,
    limit ? `of ${Math.round(limit)} W limit` : '–');

  /* The comparison the panel exists for, as numbers rather than eyeballed off the chart. */
  const rows = [];
  for (const [key, label] of [['training', 'while training'], ['idle', 'idle']]) {
    const s = gpu.summary[key];
    if (!s) continue;
    rows.push([
      label,
      fmt.dur(s.seconds),
      s.util == null ? '–' : `${s.util.toFixed(0)}%`,
      s.mem_used == null ? '–' : `${(s.mem_used / 1024).toFixed(1)} GB`,
      s.temp == null ? '–' : `${s.temp.toFixed(0)}°C`,
      s.temp_max == null ? '–' : `${s.temp_max.toFixed(0)}°C`,
      s.power == null ? '–' : `${s.power.toFixed(0)} W`,
    ]);
  }
  const sumHost = $('#gpu-summary');
  sumHost.textContent = '';
  if (rows.length) {
    sumHost.appendChild(table(
      ['', 'time in window', 'avg util', 'avg memory', 'avg temp', 'peak temp', 'avg power'],
      rows));
  } else {
    const div = document.createElement('div');
    div.className = 'chart-empty';
    div.textContent = 'No samples in this window yet — the sampler writes one every '
      + `${gpu.interval_s}s.`;
    sumHost.appendChild(div);
  }

  const t = gpu.series.time || [];
  const spans = (gpu.spans || []).map((s) => ({ ...s, label: `${s.run} training` }));
  const common = { xFmt: timeFmt, spans, spanLabel: 'a run was training', zeroFloor: true };
  Object.assign(state.charts, {
    gutil: {
      ...common, label: 'GPU utilisation over time', yFmt: (v) => `${v.toFixed(0)}%`,
      yMin: 0,
      series: [{ name: 'utilisation', color: '--series-1', x: t, y: gpu.series.util || [], label: true, fmt: (v) => `${v.toFixed(0)}%` }],
    },
    gmem: {
      ...common, label: 'GPU memory used over time', yFmt: (v) => `${(v / 1024).toFixed(0)}G`,
      rules: memTotal ? [{ y: memTotal, label: `${(memTotal / 1024).toFixed(0)} GB total` }] : [],
      series: [{ name: 'memory used', color: '--series-1', x: t, y: gpu.series.mem_used || [], label: true, fmt: (v) => `${(v / 1024).toFixed(1)}G` }],
    },
    gtemp: {
      ...common, label: 'GPU temperature over time', yFmt: (v) => `${v.toFixed(0)}°`,
      series: [{ name: 'temperature', color: '--series-1', x: t, y: gpu.series.temp || [], label: true, fmt: (v) => `${v.toFixed(0)}°C` }],
    },
    gpower: {
      ...common, label: 'GPU power draw over time', yFmt: (v) => `${v.toFixed(0)}W`,
      rules: limit ? [{ y: limit, label: `${Math.round(limit)} W limit` }] : [],
      series: [{ name: 'power', color: '--series-1', x: t, y: gpu.series.power || [], label: true, fmt: (v) => `${v.toFixed(0)}W` }],
    },
  });
  drawCharts();
}

/* ---------------------------------------------------------------- cost ---------------- */

/** Watt-hours, in the unit a person would say out loud. */
const wh = (v) => (v == null ? '–'
  : v >= 1000 ? `${(v / 1000).toFixed(2)} kWh` : `${v.toFixed(v < 10 ? 1 : 0)} Wh`);

/** Money, or a dash. Never 0.00 for "no rate set" — that reads as free. */
function money(c, v) {
  if (v == null) return '–';
  const d = Math.abs(v) >= 10 ? 2 : Math.abs(v) >= 0.1 ? 2 : 4;
  return `${c.currency || ''}${v.toFixed(d)}`;
}

/** The headline for a tile: the price when there is one, the energy when there isn't. */
const headline = (c, b) => (b.money != null ? money(c, b.money) : wh(b.wh));

function renderCost(c, gpu) {
  if (!c) return;
  state.cost = c;
  const rate = [
    c.per_kwh != null ? `${money(c, c.per_kwh)}/kWh` : null,
    c.per_hour != null ? `${money(c, c.per_hour)}/hour` : null,
  ].filter(Boolean).join(' + ');

  $('#cost-status').textContent = c.note || c.hint
    || `${rate} · measured from ${fmt.int((c.total || {}).buckets)} ledger buckets`
       + (c.since ? ` since ${new Date(c.since * 1000).toLocaleDateString()}` : '');

  const tile = (id, bundle, note) => {
    $(`#c-${id}`).textContent = bundle ? headline(c, bundle) : '–';
    $(`#c-${id}-note`).textContent = bundle ? note(bundle) : '–';
  };
  const withEnergy = (b) => `${wh(b.wh)} · ${fmt.dur(b.seconds)}`;
  tile('total', c.total, withEnergy);
  tile('today', c.today, withEnergy);
  tile('week', c.week, withEnergy);

  /* The one number that is not from the ledger: it scopes to whatever window the GPU
   * charts above are drawing, so the two panels always agree about the same hour. */
  const e = gpu && gpu.energy;
  $('#c-window').textContent = e ? (e.price ? headline(c, { ...e, money: e.price.money })
    : wh(e.wh)) : '–';
  $('#c-window-note').textContent = e
    ? `${wh(e.wh)}${e.uncovered_s > 60 ? ` · ${fmt.dur(e.uncovered_s)} not sampled` : ''}`
    : '–';

  /* Per run: the answer to "which of these spent the money". */
  const KIND = { training: 'run', job: 'job', idle: 'idle' };
  const rows = (c.runs || []).map((r) => [
    r.label || 'idle — nothing running',
    KIND[r.kind] || r.kind,
    fmt.dur(r.seconds),
    wh(r.wh),
    money(c, r.money),
    r.coverage == null ? '–' : fmt.pct(r.coverage, 0),
    r.estimated_money != null ? money(c, r.estimated_money)
      : r.estimated_wh != null ? wh(r.estimated_wh) : '–',
    r.tokens == null ? '–' : fmt.compact(r.tokens),
    r.per_mtoken == null ? (r.wh_per_mtoken == null ? '–' : wh(r.wh_per_mtoken))
      : money(c, r.per_mtoken),
  ]);
  const runsHost = $('#cost-runs');
  runsHost.textContent = '';
  if (rows.length) {
    runsHost.appendChild(table(
      ['', 'what', 'measured time', 'energy', 'measured cost', 'of run’s life',
        'whole run (est.)', 'tokens', 'per 1M tokens'], rows));
  } else {
    const div = document.createElement('div');
    div.className = 'chart-empty';
    div.textContent = 'Nothing measured yet — the ledger fills as the sampler runs.';
    runsHost.appendChild(div);
  }

  const dayHost = $('#cost-daily');
  dayHost.textContent = '';
  if ((c.daily || []).length) {
    dayHost.appendChild(table(['day', 'measured time', 'energy', 'cost'],
      c.daily.slice().reverse().map((d) => [
        d.day, fmt.dur(d.seconds), wh(d.wh), money(c, d.money)])));
  }

  /* Serving is billed per MILLION COMPLETION tokens, and idle energy is reported beside
   * the rate rather than folded into it — a server that was mostly not serving is a
   * different problem from one whose tokens are expensive. */
  const s = c.serving || {};
  const hasServing = !!s.requests;
  $('#cost-serving').hidden = !hasServing;
  if (hasServing) {
    const rate = s.money_per_million_completion != null
      ? money(c, s.money_per_million_completion)
      : wh(s.wh_per_million_completion);
    $('#cost-serving-table').textContent = '';
    $('#cost-serving-table').appendChild(table(
      ['requests', 'completion tokens', 'prompt tokens', 'generating', 'per 1M completion',
       'idle share'],
      [[fmt.int(s.requests), fmt.int(s.completion_tokens), fmt.int(s.prompt_tokens),
        fmt.dur(s.busy_seconds), rate,
        s.idle_share == null ? '–' : `${Math.round(s.idle_share * 100)}%`]]));
    $('#cost-serving-note').textContent = s.caveat || '';
  }

  $('#cost-basis').textContent = `Measures the ${c.basis}. `
    + 'Coverage is how much of a run the sampler actually saw — the portal only records '
    + 'while it is up, so “whole run” and “per 1M tokens” scale the measured part up on the '
    + 'assumption that the unwatched hours drew power at the same rate. '
    + (c.configured ? '' : 'Set a rate in configs/portal.yaml to price any of it.');
}

export function wireGpu() {
  for (const btn of $$('.gpu-window button')) {
    btn.addEventListener('click', () => {
      state.gpuWindow = btn.dataset.window;
      /* The window buttons are the coarse zoom; a drag inside the old one would fight it. */
      for (const key of GPU_CHARTS) delete state.zoom[key];
      localStorage.setItem('aksharallm-gpu-window', state.gpuWindow);
      markGpuWindow();
      schedule(0);
    });
  }
  state.gpuWindow = localStorage.getItem('aksharallm-gpu-window') || '3600';
  markGpuWindow();
}

function markGpuWindow() {
  for (const btn of $$('.gpu-window button')) {
    const on = btn.dataset.window === state.gpuWindow;
    btn.className = on ? '' : 'ghost';
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
}

/* ---------------------------------------------------------------- schedule ------------ */

const DAY_LETTERS = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function selectedDays() {
  return $$('#sched-days .day')
    .map((b, i) => (b.getAttribute('aria-pressed') === 'true' ? i : -1))
    .filter((i) => i >= 0);
}

function describeDays(days) {
  if (days.length === 7) return 'daily';
  if (String(days) === '0,1,2,3,4') return 'mon–fri';
  if (String(days) === '5,6') return 'sat, sun';
  return days.map((d) => DAY_NAMES[d].toLowerCase()).join(', ');
}

function buildDayPicker() {
  const host = $('#sched-days');
  host.textContent = '';
  DAY_LETTERS.forEach((letter, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'day';
    b.textContent = letter;
    b.title = DAY_NAMES[i];
    b.setAttribute('aria-label', DAY_NAMES[i]);
    b.setAttribute('aria-pressed', 'true');
    b.addEventListener('click', () => {
      b.setAttribute('aria-pressed', b.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    });
    host.appendChild(b);
  });
  const presets = document.createElement('span');
  presets.className = 'day-presets';
  for (const [label, days] of [['daily', [0, 1, 2, 3, 4, 5, 6]],
    ['mon–fri', [0, 1, 2, 3, 4]], ['sat/sun', [5, 6]]]) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ghost';
    b.textContent = label;
    b.addEventListener('click', () => {
      $$('#sched-days .day').forEach((d, i) => {
        d.setAttribute('aria-pressed', days.includes(i) ? 'true' : 'false');
      });
    });
    presets.appendChild(b);
  }
  host.appendChild(presets);
}

function renderSchedule(sched) {
  state.schedule = sched;

  const arm = $('#sched-arm');
  arm.textContent = sched.enabled ? 'Armed' : 'Paused';
  arm.className = sched.enabled ? '' : 'ghost';
  arm.title = sched.enabled ? 'nothing scheduled will fire if you pause this'
    : 'rules are kept but nothing fires';

  /* Rules mean nothing without something watching the clock — say so plainly. */
  $('#sched-status').textContent = sched.running
    ? `clock running${sched.in_portal ? ' in this portal' : ` as pid ${sched.holder}`} · `
      + `${sched.rules.length} rule${sched.rules.length === 1 ? '' : 's'} · times are this machine’s local time`
    : 'NOTHING IS WATCHING THE CLOCK — rules will not fire. Run scripts/portal.sh or '
      + 'scripts/schedule.sh daemon.';

  const sel = $('#sched-run');
  const startable = sched.startable || [];
  if (sel.dataset.sig !== String(startable)) {
    sel.textContent = '';
    for (const r of startable) {
      sel.appendChild(Object.assign(document.createElement('option'),
        { value: r, textContent: r }));
    }
    sel.dataset.sig = String(startable);
    if (startable.includes(state.run)) sel.value = state.run;
  }

  const rows = sched.rules.map((r) => {
    const actions = document.createElement('div');
    actions.className = 'row-actions';
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'ghost';
    toggle.textContent = r.enabled ? 'pause' : 'resume';
    toggle.addEventListener('click', () => act(
      () => post('/api/schedule/toggle', { id: r.id, enabled: !r.enabled }), 'Updated.'));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'ghost';
    del.textContent = 'remove';
    del.addEventListener('click', () => {
      if (!confirm(`Remove this rule?\n\n${r.describe}`)) return;
      act(() => post('/api/schedule/remove', { id: r.id }), 'Removed.');
    });
    actions.append(toggle, del);
    return {
      enabled: r.enabled,
      cells: [
        r.run,
        r.action + (r.stop_after ? ` · ${fmt.int(r.stop_after)} steps` : ''),
        r.at,
        describeDays(r.days),
        r.enabled ? (r.next_fire ? `${new Date(r.next_fire).toLocaleString(undefined,
          { weekday: 'short', hour: '2-digit', minute: '2-digit' })}`
          + (r.next_fire_in_s != null ? ` · in ${fmt.dur(r.next_fire_in_s)}` : '') : '—')
          : 'paused',
        r.last_result || '—',
        actions,
      ],
    };
  });

  const host = $('#sched-rules');
  host.textContent = '';
  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'chart-empty';
    empty.textContent = 'Nothing scheduled. Add a window above — for example 22:00 to 06:30, '
      + 'mon–fri, and the GPU trains overnight and hands itself back in the morning.';
    host.appendChild(empty);
  } else {
    const t = table(['run', 'action', 'at', 'days', 'next', 'last result', ''],
      rows.map((r) => r.cells));
    [...t.tBodies[0].rows].forEach((tr, i) => {
      if (!rows[i].enabled) tr.className = 'rule-paused';
    });
    host.appendChild(t);
  }

  const log = $('#sched-log');
  const events = sched.events || [];
  log.textContent = events.length ? events.join('\n') : '(the scheduler has not done anything yet)';
  if (!$('.sched-events').dataset.touched) log.scrollTop = log.scrollHeight;
}

export function wireSchedule() {
  buildDayPicker();

  const mode = $('#sched-mode');
  const syncMode = () => {
    const m = mode.value;
    $('#sched-to-field').hidden = m !== 'window';
    $('#sched-steps-field').hidden = m === 'stop';
    $('#sched-smoke-field').hidden = m === 'stop';
    $('#sched-from-label').textContent = m === 'stop' ? 'stop at' : 'start at';
  };
  mode.addEventListener('change', syncMode);
  syncMode();

  $('#sched-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const days = selectedDays();
    if (!days.length) { flash('Pick at least one day.', 'error'); return; }
    const run = $('#sched-run').value;
    const steps = $('#sched-steps').value.trim();
    const common = {
      run, days,
      stop_after: steps ? Number(steps) : null,
      skip_smoke: !$('#sched-smoke').checked,
    };
    const m = mode.value;
    const body = m === 'window'
      ? { ...common, start_at: $('#sched-from').value, stop_at: $('#sched-to').value }
      : { ...common, action: m, at: $('#sched-from').value };
    act(() => post(`/api/schedule/${m === 'window' ? 'window' : 'rule'}`, body), 'Scheduled.');
  });

  $('#sched-arm').addEventListener('click', () => act(
    () => post('/api/schedule/enable', { enabled: !(state.schedule || {}).enabled }),
    'Schedule updated.'));

  /* Don't yank the activity log back to the bottom while it is being read. */
  $('.sched-events').addEventListener('toggle', (e) => {
    e.target.dataset.touched = e.target.open ? '1' : '';
  });
}

/* ---------------------------------------------------------------- poll loop ----------- */

function pollInterval(phase) {
  /* Fast while something is happening, lazy when nothing is: a step takes ~9s, so 2s is
   * already faster than the data changes, and an idle run needs no attention at all. */
  return phase === 'idle' ? 10000 : 2500;
}

export async function refresh() {
  if (!state.run) return;
  try {
    const q = state.logFile ? `?lines=400&file=${encodeURIComponent(state.logFile)}` : '?lines=400';
    const [status, log, runs, sched, gpu, pipeline, cost] = await Promise.all([
      api(`/api/run/${encodeURIComponent(state.run)}`),
      api(`/api/run/${encodeURIComponent(state.run)}/log${q}`),
      api('/api/runs'),
      api('/api/schedule'),
      api(`/api/gpu?window=${encodeURIComponent(state.gpuWindow)}`),
      // never let a pipeline hiccup break the dashboard
      api(`/api/pipeline/${encodeURIComponent(baseOf(state.run))}`).catch(() => null),
      api('/api/cost').catch(() => null),
    ]);
    state.status = status;
    state.log = log;
    document.body.classList.remove('stale');
    renderRuns(runs.runs);
    $('#foot-root').textContent = runs.root;
    renderPhase(status);
    renderControls(status);
    renderProgress(status);
    renderTiles(status);
    renderCharts(status);
    renderSessions(status);
    renderConfig(status);
    renderLog(log);
    renderGpu(gpu);
    renderCost(cost, gpu);
    renderSchedule(sched);
    renderPipeline(pipeline);
    live(`updated ${new Date().toLocaleTimeString()}`, 'on');
  } catch (err) {
    document.body.classList.add('stale');
    live(`no answer from the portal — ${err.message}`, 'err');
  } finally {
    schedule();
  }
}

export function schedule(delay) {
  clearTimeout(state.timer);
  if (document.hidden) return;  // a background tab does not need to poll
  // nor a dashboard nobody is looking at
  if (['code', 'docs', 'quant', 'lora'].includes(state.view)) return;
  const ms = delay != null ? delay : pollInterval(state.status ? state.status.phase : 'idle');
  state.timer = setTimeout(refresh, ms);
}

/** Run an action, then poll hard for a few seconds so the phase badge reacts immediately. */
export async function act(fn, okPrefix) {
  state.busy = true;
  if (state.status) renderControls(state.status);
  try {
    const res = await fn();
    flash(`${okPrefix} ${res.note || ''}`.trim(), 'ok');
  } catch (err) {
    flash(err.message, 'error');
  } finally {
    state.busy = false;
    schedule(400);
  }
}

/* ================================================================= bound picker ========
 * "How long should this go on for?" — asked at launch (how much of the budget does this
 * session get?) and at every stop (how much longer?), for training runs, fine-tunes and
 * QAT. One dialog answers all of them, because they differ only in which units make sense.
 *
 * Three things make it usable where a prompt() was not:
 *
 *   presets   the answer is nearly always one of six numbers. Those are one click.
 *   the dial  logarithmic, so a single throw spans a minute to twelve hours (or ten steps
 *             to a hundred thousand) without the low end being a pixel wide. Linear would
 *             put every useful short stop in the first 2% of the track.
 *   the line  underneath, in words: what step it lands on, what time it finishes, how many
 *             checkpoints that is. The number you picked is not the number you care about.
 *
 * It resolves to {unit, value} in the unit's own base (steps, or seconds), or the string
 * 'now', or null if dismissed. The caller decides what those mean to its endpoint.
 */
export const boundPicker = (() => {
  const dlg = () => $('#bound');
  let spec = null;      // the descriptor passed to open()
  let unit = null;      // the unit descriptor currently selected
  let value = 0;        // in the unit's own base
  let resolve = null;

  /* Slider position <-> value, logarithmically. */
  const toPos = (v) => Math.round(1000 * Math.log(v / unit.min) / Math.log(unit.max / unit.min));
  const toValue = (p) => unit.min * Math.pow(unit.max / unit.min, p / 1000);

  /** Round to something a person would have typed: coarser as the number grows. */
  function snap(v) {
    let g = 1;
    for (const [below, gran] of unit.snap) { if (v < below) { g = gran; break; } g = gran; }
    return Math.min(unit.max, Math.max(unit.min, Math.round(v / g) * g));
  }

  function setValue(v, from) {
    value = unit.none ? null : snap(v);
    if (from !== 'field') $('#bound-value').value = unit.none ? '' : value / (unit.step || 1);
    if (from !== 'dial') $('#bound-dial').value = unit.none ? 0 : toPos(value);
    for (const chip of $$('#bound-chips .chip')) {
      chip.classList.toggle('is-on', Number(chip.dataset.value) === value);
    }
    $('#bound-preview').textContent = unit.preview ? unit.preview(value) : '';
  }

  function selectUnit(id) {
    unit = spec.units.find((u) => u.id === id) || spec.units[0];
    for (const btn of $$('#bound-units .seg-btn')) {
      const on = btn.dataset.unit === unit.id;
      btn.classList.toggle('is-on', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    /* A unit with nothing to choose (the whole budget) keeps only its preview line. */
    const hide = !!unit.none;
    for (const sel of ['#bound-chips', '#bound-dial', '.dial-ends', '.bound-exact']) {
      $(sel).hidden = hide;
    }
    $('#bound-chips').innerHTML = hide ? '' : unit.chips.map((v) =>
      `<button type="button" class="chip" data-value="${v}">${escHtml(unit.fmt(v))}</button>`).join('');
    $('#bound-unit-label').textContent = unit.noun || '';
    $('#bound-min').textContent = hide ? '' : unit.fmt(unit.min);
    $('#bound-max').textContent = hide ? '' : unit.fmt(unit.max);
    /* A unit with no scale to pick from still has to refresh the preview line, so it is
     * `setValue(null)` rather than an early return — the sentence is the whole dialog. */
    setValue(hide ? null
      : unit.value != null ? unit.value : unit.chips[Math.min(2, unit.chips.length - 1)]);
  }

  function close(result) {
    dlg().close();
    const r = resolve; resolve = null;
    if (r) r(result);
  }

  function open(next) {
    spec = next;
    $('#bound-title').textContent = spec.title;
    $('#bound-sub').textContent = spec.sub || '';
    $('#bound-sub').hidden = !spec.sub;
    $('#bound-ok').textContent = spec.okLabel || 'Queue stop';
    $('#bound-now').hidden = !spec.showNow;
    $('#bound-now').textContent = spec.nowLabel || 'Stop now';
    $('#bound-units').innerHTML = spec.units.map((u) =>
      `<button type="button" class="seg-btn" data-unit="${u.id}" aria-pressed="false">`
      + `${escHtml(u.label)}</button>`).join('');
    selectUnit(spec.unit || spec.units[0].id);
    dlg().showModal();
    $(unit.none ? '#bound-ok' : '#bound-value').focus();
    return new Promise((r) => { resolve = r; });
  }

  /* Wired once, from wire(), like every other panel. */
  function wireBound() {
    $('#bound-units').addEventListener('click', (e) => {
      const btn = e.target.closest('.seg-btn');
      if (btn) selectUnit(btn.dataset.unit);
    });
    $('#bound-chips').addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (chip) setValue(Number(chip.dataset.value));
    });
    $('#bound-dial').addEventListener('input', (e) => setValue(toValue(Number(e.target.value)), 'dial'));
    $('#bound-value').addEventListener('input', (e) => {
      const n = Number(e.target.value);
      if (n > 0) setValue(n * (unit.step || 1), 'field');
    });
    $('#bound-value').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); close({ unit: unit.id, value }); }
    });
    $('#bound-ok').addEventListener('click', () => close({ unit: unit.id, value }));
    $('#bound-now').addEventListener('click', () => close('now'));
    $('#bound-cancel').addEventListener('click', () => close(null));
    /* Escape and the backdrop both dismiss; `cancel` fires for Escape only. */
    dlg().addEventListener('cancel', () => close(null));
    dlg().addEventListener('click', (e) => { if (e.target === dlg()) close(null); });
  }

  return { open, wire: wireBound };
})();

/** Minutes as a person says them: 45m, 1h, 2h30m. */
export function fmtMins(mins) {
  if (mins < 60) return `${mins} min`;
  const h = Math.floor(mins / 60), m = mins % 60;
  return m ? `${h}h${String(m).padStart(2, '0')}` : `${h} hr${h > 1 ? 's' : ''}`;
}

/** The seconds-per-step to reason with, or null when nothing has been logged yet. */
export function secPerStep(s) {
  const v = s && s.last && s.last.s_per_step;
  return v && v > 0 ? v : null;
}

/** {step, total} from the last matching line of a job's log tail, or null.
 *
 * The fine-tune and QAT panels have no step field in their status — they stream a log. It
 * is the log the person is already reading, so reading the same number out of it keeps the
 * dialog and the panel telling one story. */
export function progressFromLog(lines, re) {
  for (let i = (lines || []).length - 1; i >= 0; i--) {
    const m = re.exec(lines[i]);
    if (m) return { step: Number(m[1]), total: Number(m[2]) };
  }
  return null;
}

/** "about 13:07" for a number of seconds from now. */
export function fmtWhen(seconds) {
  return new Date(Date.now() + seconds * 1000)
    .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/* The unit descriptors. Chips are the answers people actually give; the dial covers the
 * rest. `step` is the divisor between what the field shows and what the API is sent —
 * minutes on screen, seconds on the wire. */
export const UNIT_MINUTES = {
  id: 'in', label: 'Time', noun: 'minutes', step: 60,
  min: 60, max: 12 * 3600, value: 30 * 60,
  chips: [60, 300, 600, 1800, 3600, 2 * 3600, 4 * 3600],
  snap: [[600, 60], [3600, 300], [14400, 900], [Infinity, 1800]],
  fmt: (v) => fmtMins(Math.round(v / 60)),
};

export const UNIT_MORE_STEPS = {
  id: 'after', label: 'Steps', noun: 'more steps',
  min: 10, max: 100000, value: 500,
  chips: [10, 50, 100, 500, 1000, 5000, 10000],
  snap: [[100, 10], [1000, 50], [10000, 100], [Infinity, 500]],
  fmt: (v) => fmt.int(v),
};

/* ---- the run report ------------------------------------------------------------------
 * Built on demand, never on the poll: one build re-reads the entire training log, the energy
 * ledger and every benchmark result this run ever produced. That is a fine thing to do when
 * a person asks for it and a terrible thing to do every three seconds.
 *
 * The panel shows the *live* report rather than whatever is in checkpoints/<run>/report.md,
 * because it is usually opened mid-run and a snapshot from Tuesday's exit would be the most
 * confidently wrong thing on the page. Saving is a separate button, and it writes the same
 * file a finished trainer leaves behind. */
export async function buildReport(save = false) {
  const run = state.run;
  if (!run) return;
  const body = $('#report-body');
  const note = $('#report-note');
  note.textContent = save ? 'saving…' : 'building…';
  try {
    const path = `/api/run/${encodeURIComponent(run)}/report`;
    const res = save ? await post(path, {}) : await api(path);
    body.innerHTML = renderMarkdown(res.markdown || '');
    body.dataset.run = run;
    body.scrollTop = 0;
    $('#btn-report-save').disabled = false;
    note.textContent = (res.saved ? `saved to ${res.saved} · ` : '')
      + `built ${new Date().toLocaleTimeString()}`;
    if (save) flash(`Report written to ${res.saved}`, 'ok');
  } catch (err) {
    note.textContent = '';
    body.innerHTML = `<p class="docs-hint">could not build the report — ${escHtml(err.message)}</p>`;
  }
}

/* ---- the HTTP server ---------------------------------------------------------------------
 * A separate process with its own lifetime, so this panel holds none of its state: it posts to
 * scripts/serve.sh through the API and then reads the server's own /health. That is why a
 * server started in a terminal appears here, and why stopping the portal never stops it. */
export async function refreshServe() {
  let d;
  try {
    d = await api('/api/serve');
  } catch { return; }
  state.serve = d;
  const h = d.health || {};
  const kv = h.kv_blocks || {};
  $('#serve-status').textContent = d.phase === 'running'
    ? `${d.url} · ${h.model || ''} on the ${(h.device || '?').toUpperCase()}`
    : d.phase === 'starting' ? 'starting — loading the checkpoint…' : 'not running';
  $('#btn-serve-start').disabled = d.running || state.busy;
  $('#btn-serve-stop').disabled = !d.running || state.busy;

  const tiles = !h.ok ? '' : [
    ['in flight', `${fmt.int(h.running)}`, `${fmt.int(h.waiting)} waiting for a slot`],
    ['max batch', fmt.int(h.max_batch), 'sequences per pass over the weights'],
    ['kv pool', `${fmt.pct((kv.used || 0) / (kv.total || 1), 0)}`,
      `${fmt.int(kv.used)} of ${fmt.int(kv.total)} blocks · ${fmt.bytes(kv.bytes)}`],
    ['served', fmt.int((h.stats || {}).tokens),
      `${fmt.num((h.stats || {}).tokens_per_step, 2)} tokens per model pass`],
    ...(h.speculate ? [['drafting', fmt.pct((h.stats || {}).accept_rate, 0),
      `${fmt.int((h.stats || {}).drafted)} guessed ${h.speculate} at a time, accepted`]] : []),
  ].map(([label, value, note]) => `<div class="tile"><div class="tile-label">${label}</div>`
    + `<div class="tile-value">${value}</div><div class="tile-note">${escHtml(note)}</div></div>`).join('');
  $('#serve-tiles').innerHTML = tiles;
  if (d.running) $('#serve-hint').innerHTML = `Try it: <code>${escHtml(d.hint)}</code>`;
  const log = $('#serve-log');
  log.hidden = !(d.log || []).length;
  log.textContent = (d.log || []).slice(-20).join('\n');
}

export function wireServe() {
  $('#btn-serve-start').addEventListener('click', () => act(() => post('/api/serve/start', {
    checkpoint: $('#serve-ckpt').value || undefined,
    port: Number($('#serve-port').value) || undefined,
    max_batch: Number($('#serve-batch').value) || undefined,
    speculate: Number($('#serve-spec').value) || undefined,
  }), 'Server starting'));
  $('#btn-serve-stop').addEventListener('click', () => act(() => post('/api/serve/stop', {}),
    'Server stopped'));
  /* The checkpoint list is the Playground's — one place that knows what exists. */
  api('/api/infer').then((d) => {
    $('#serve-ckpt').innerHTML = (d.checkpoints || [])
      .map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)}</option>`).join('');
    if (d.default) $('#serve-ckpt').value = d.default;
  }).catch(() => {});
  refreshServe();
  setInterval(refreshServe, 5000);
}

export function wireReport() {
  $('#btn-report').addEventListener('click', () => buildReport(false));
  $('#btn-report-save').addEventListener('click', () => buildReport(true));
}

/** Another run's report must not sit under this run's heading, so it is cleared rather than
 *  left to be replaced on the next click. */
function clearReport() {
  const body = $('#report-body');
  if (!body || !body.dataset.run) return;
  body.dataset.run = '';
  body.innerHTML = '<p class="docs-hint">The trainers write this file when they exit. '
    + 'Build it here to read one for a run that is still going.</p>';
  $('#report-note').textContent = '';
  $('#btn-report-save').disabled = true;
}

export function selectRun(run) {
  state.run = run;
  state.logFile = null;
  state.status = null;
  state.zoom = {};        /* another run's step range means nothing here */
  flash('');
  clearReport();
  $('#log-select').dataset.run = '';
  schedule(0);
}

registerTab('dashboard', { open: () => { schedule(0); drawCharts(); } });

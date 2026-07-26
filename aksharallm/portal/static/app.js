/* aksharallm portal — the whole client, in one file, with no dependencies.
 *
 * Three parts:
 *   1. formatting + fetch helpers
 *   2. a small SVG line-chart engine (axes, hairline grid, crosshair, tooltip, table twin)
 *   3. render passes driven by a poll loop
 *
 * The charts are hand-written for the same reason the transformer is: a plotting library
 * would be one more thing this project doesn't explain. Colour is never the only encoding —
 * every series is in the legend, every value is reachable from the table view. */

'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ---------------------------------------------------------------- formatting ---------- */

const fmt = {
  int: (n) => (n == null || Number.isNaN(n) ? '–' : Math.round(n).toLocaleString()),
  num: (n, d = 3) => (n == null || Number.isNaN(n) ? '–' : Number(n).toFixed(d)),
  pct: (n, d = 1) => (n == null || Number.isNaN(n) ? '–' : (n * 100).toFixed(d) + '%'),
  exp: (n) => (n == null || Number.isNaN(n) ? '–' : Number(n).toExponential(2)),
  /** 1.23B / 45.6M — the same compact form the trainer prints. */
  compact: (n) => {
    if (n == null || Number.isNaN(n)) return '–';
    const units = ['', 'K', 'M', 'B', 'T'];
    let v = n, i = 0;
    while (Math.abs(v) >= 1000 && i < units.length - 1) { v /= 1000; i++; }
    return `${v.toFixed(v >= 100 || i === 0 ? 0 : 2)}${units[i]}`;
  },
  /** Mirrors runlog.fmt_dur so the page and the log agree: 45.2s / 12m30s / 6h05m / 3d04h. */
  dur: (s) => {
    if (s == null || Number.isNaN(s)) return '–';
    if (s < 0) return '-' + fmt.dur(-s);
    if (s < 60) return `${s.toFixed(1)}s`;
    let m = Math.floor(s / 60), sec = Math.floor(s % 60);
    let h = Math.floor(m / 60); m %= 60;
    const d = Math.floor(h / 24); h %= 24;
    if (d) return `${d}d${String(h).padStart(2, '0')}h`;
    if (h) return `${h}h${String(m).padStart(2, '0')}m`;
    return `${m}m${String(sec).padStart(2, '0')}s`;
  },
  ago: (ts) => (ts == null ? '–' : fmt.dur(Date.now() / 1000 - ts) + ' ago'),
  clock: (ts) => (ts == null ? '–' : new Date(ts * 1000).toLocaleString()),
  bytes: (n) => (n == null ? '–' : (n / 1e9 >= 1 ? (n / 1e9).toFixed(2) + ' GB'
    : (n / 1e6).toFixed(1) + ' MB')),
};

/* ---------------------------------------------------------------- api ----------------- */

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', 'X-Portal': '1', ...(options.headers || {}) },
  });
  let data = {};
  try { data = await res.json(); } catch { /* a 500 with no body */ }
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body || {}) });

/* ---------------------------------------------------------------- chart engine -------- */

const SVG_NS = 'http://www.w3.org/2000/svg';
const el = (name, attrs = {}, parent = null) => {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v != null) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
};

/** Tick values at "nice" round intervals covering [min, max]. */
function niceTicks(min, max, count = 5) {
  if (!(isFinite(min) && isFinite(max))) return [];
  if (min === max) return [min];
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

/**
 * Draw a line chart into `host`.
 *
 * spec = {
 *   series: [{ name, color, x:[], y:[], faint?, dots?, fmt? }],
 *   yFmt, xFmt, height, rules: [{ y, label }], legend: bool
 * }
 *
 * Marks are thin, the grid is a hairline one shade off the surface, and a crosshair layer
 * is always present — an SVG chart in a browser is interactive by default.
 */
function lineChart(host, spec) {
  const series = spec.series
    .map((s) => ({
      ...s,
      pts: s.x.map((x, i) => [x, s.y[i]]).filter(([x, y]) => isFinite(x) && isFinite(y) && y != null),
    }))
    .filter((s) => s.pts.length);

  host.textContent = '';
  if (!series.length) {
    const empty = document.createElement('div');
    empty.className = 'chart-empty';
    empty.textContent = spec.empty || 'No readings yet — this chart fills in as the run logs steps.';
    host.appendChild(empty);
    return;
  }

  /* A legend is always present for two or more series, so identity is never colour-alone.
   * One series needs none: the title names it. */
  if (series.length > 1 && spec.legend !== false) {
    const legend = document.createElement('div');
    legend.className = 'legend';
    for (const s of series) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const sw = document.createElement('span');
      sw.className = 'legend-swatch' + (s.dots ? ' dot' : '');
      sw.style.background = `var(${s.color})`;
      item.append(sw, document.createTextNode(s.name));
      legend.appendChild(item);
    }
    host.appendChild(legend);
  }

  const width = Math.max(320, host.clientWidth || 640);
  const height = spec.height || 210;
  const pad = { t: 10, r: 16, b: 26, l: 52 };
  const pw = width - pad.l - pad.r;
  const ph = height - pad.t - pad.b;

  const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
  const ys = series.flatMap((s) => s.pts.map((p) => p[1]))
    .concat((spec.rules || []).map((r) => r.y));
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (x1 === x0) x1 = x0 + 1;
  const yPad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.1 || 1;
  y0 -= yPad; y1 += yPad;
  if (spec.yMin != null) y0 = Math.min(y0, spec.yMin);
  if (spec.zeroFloor && y0 < 0) y0 = 0;

  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * pw;
  const sy = (y) => pad.t + ph - ((y - y0) / (y1 - y0)) * ph;

  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`, width, height,
    role: 'img', 'aria-label': spec.label || 'chart',
  }, host);

  /* grid + axes: solid hairlines, never dashed, one shade off the surface */
  const yTicks = niceTicks(y0, y1, 5);
  for (const t of yTicks) {
    el('line', { class: 'grid-line', x1: pad.l, x2: pad.l + pw, y1: sy(t), y2: sy(t) }, svg);
    el('text', { class: 'axis-label', x: pad.l - 8, y: sy(t) + 4, 'text-anchor': 'end' }, svg)
      .textContent = (spec.yFmt || fmt.num)(t);
  }
  el('line', { class: 'axis-line', x1: pad.l, x2: pad.l + pw, y1: pad.t + ph, y2: pad.t + ph }, svg);

  for (const t of niceTicks(x0, x1, Math.max(3, Math.floor(pw / 90)))) {
    if (t < x0 || t > x1) continue;
    el('text', {
      class: 'axis-label', x: sx(t), y: height - 8, 'text-anchor': 'middle',
    }, svg).textContent = (spec.xFmt || fmt.compact)(t);
  }

  /* a threshold rule is the one legitimate dashed line here */
  for (const r of spec.rules || []) {
    if (r.y < y0 || r.y > y1) continue;
    el('line', { class: 'rule-line', x1: pad.l, x2: pad.l + pw, y1: sy(r.y), y2: sy(r.y) }, svg);
    /* Anchored left, because the right-hand end is where the newest-value label lives. */
    el('text', { class: 'rule-label', x: pad.l + 4, y: sy(r.y) - 5 }, svg).textContent = r.label;
  }

  for (const s of series) {
    const d = s.pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(' ');
    const path = el('path', { class: 'series-line' + (s.faint ? ' faint' : ''), d }, svg);
    path.style.stroke = `var(${s.color})`;
    /* Sparse series (validation loss lands every 1000 steps) get markers: a 3-point line
     * is hard to see, and the markers carry a second, non-colour cue. */
    if (s.dots && s.pts.length <= 80) {
      for (const [x, y] of s.pts) {
        const dot = el('circle', { class: 'series-dot', cx: sx(x), cy: sy(y), r: 4.5 }, svg);
        dot.style.fill = `var(${s.color})`;
      }
    }
    /* Direct-label the newest point of *one* series — selectively, never every point, and
     * never two at once: on a converged run the ema and the validation loss land on the
     * same pixel and the two labels overprint each other. The rest is in the tooltip. */
    if (s.label === true) {
      const [lx, ly] = s.pts[s.pts.length - 1];
      const tx = el('text', {
        class: 'axis-label', x: Math.min(sx(lx) + 6, pad.l + pw), y: sy(ly) - 7,
        'text-anchor': sx(lx) > pad.l + pw - 46 ? 'end' : 'start',
      }, svg);
      tx.textContent = (s.fmt || spec.yFmt || fmt.num)(ly);
      tx.style.fill = `var(${s.color})`;
    }
  }

  /* ---- crosshair + tooltip ---- */
  const primary = series.reduce((a, b) => (b.pts.length > a.pts.length ? b : a));
  const cross = el('line', { class: 'crosshair', y1: pad.t, y2: pad.t + ph, x1: 0, x2: 0 }, svg);
  cross.style.display = 'none';
  const marks = series.map((s) => {
    const c = el('circle', { class: 'series-dot', r: 4.5, cx: 0, cy: 0 }, svg);
    c.style.fill = `var(${s.color})`;
    c.style.display = 'none';
    return c;
  });
  const tip = $('#tooltip');
  /* The hit area is the whole plot, so there is nothing to land on precisely. */
  const hit = el('rect', { class: 'hit', x: pad.l, y: pad.t, width: pw, height: ph }, svg);

  const show = (evt) => {
    const box = svg.getBoundingClientRect();
    const xPix = ((evt.clientX - box.left) / box.width) * width;
    const xVal = x0 + ((xPix - pad.l) / pw) * (x1 - x0);
    let bi = 0;
    for (let i = 1; i < primary.pts.length; i++) {
      if (Math.abs(primary.pts[i][0] - xVal) < Math.abs(primary.pts[bi][0] - xVal)) bi = i;
    }
    const step = primary.pts[bi][0];
    cross.setAttribute('x1', sx(step));
    cross.setAttribute('x2', sx(step));
    cross.style.display = '';

    let html = `<div class="tt-head">step ${fmt.int(step)}</div>`;
    series.forEach((s, i) => {
      /* Nearest reading at or before the crosshair — series are sampled at different rates. */
      let hitPt = null;
      for (const p of s.pts) {
        if (p[0] <= step + 1e-9) hitPt = p; else break;
      }
      if (!hitPt) { marks[i].style.display = 'none'; return; }
      marks[i].setAttribute('cx', sx(hitPt[0]));
      marks[i].setAttribute('cy', sy(hitPt[1]));
      marks[i].style.display = '';
      html += `<div class="tt-row"><span class="tt-key">`
        + `<span class="tt-swatch" style="background:var(${s.color})"></span>${s.name}</span>`
        + `<span>${(s.fmt || spec.yFmt || fmt.num)(hitPt[1])}</span></div>`;
    });
    tip.innerHTML = html;
    tip.hidden = false;
    const tb = tip.getBoundingClientRect();
    const left = evt.clientX + 14 + tb.width > window.innerWidth
      ? evt.clientX - tb.width - 14 : evt.clientX + 14;
    tip.style.left = `${Math.max(6, left)}px`;
    tip.style.top = `${Math.max(6, Math.min(evt.clientY - tb.height / 2, window.innerHeight - tb.height - 6))}px`;
  };
  const hide = () => {
    cross.style.display = 'none';
    marks.forEach((m) => { m.style.display = 'none'; });
    tip.hidden = true;
  };
  hit.addEventListener('pointermove', show);
  hit.addEventListener('pointerleave', hide);
  svg.addEventListener('pointerleave', hide);
}

/** The table twin of a chart: every plotted value, reachable without hovering anything. */
function chartTable(host, spec) {
  const steps = [...new Set(spec.series.flatMap((s) => s.x))].sort((a, b) => b - a);
  const rows = steps.map((step) => [
    fmt.int(step),
    ...spec.series.map((s) => {
      const i = s.x.indexOf(step);
      return i < 0 || s.y[i] == null ? '' : (s.fmt || spec.yFmt || fmt.num)(s.y[i]);
    }),
  ]);
  host.textContent = '';
  host.appendChild(rows.length
    ? table(['step', ...spec.series.map((s) => s.name)], rows)
    : Object.assign(document.createElement('div'),
      { className: 'chart-empty', textContent: 'Nothing logged yet.' }));
}

function table(head, rows, opts = {}) {
  const t = document.createElement('table');
  t.className = 'data';
  const thead = t.createTHead().insertRow();
  for (const h of head) {
    const th = document.createElement('th');
    th.textContent = h;
    thead.appendChild(th);
  }
  const body = t.createTBody();
  rows.forEach((r, ri) => {
    const tr = body.insertRow();
    if (opts.currentRow === ri) tr.className = 'current';
    for (const c of r) {
      const td = tr.insertCell();
      if (c instanceof Node) td.appendChild(c); else td.textContent = c;
    }
  });
  return t;
}

/* ---------------------------------------------------------------- state --------------- */

const state = {
  run: null,
  status: null,
  log: null,
  logFile: null,     // null = whichever file was written most recently
  timer: null,
  charts: {},        // last spec per chart, so a resize can redraw without a fetch
  busy: false,
};

function flash(msg, kind = '') {
  const box = $('#flash');
  box.textContent = msg;
  box.className = 'flash ' + kind;
  box.hidden = !msg;
}

function live(text, kind) {
  $('#live-label').textContent = text;
  $('#live').className = 'live ' + (kind || '');
}

/* ---------------------------------------------------------------- render -------------- */

function renderPhase(s) {
  const badge = $('#phase');
  const queued = s.stop && !s.stop.now ? ` · stop queued at ${fmt.int(s.stop.target)}` : '';
  const label = {
    training: 'training',
    launching: 'pre-flight',
    stopping: 'stopping',
    idle: 'idle',
  }[s.phase] || s.phase;
  badge.textContent = label + (s.pid ? ` · pid ${s.pid}` : '') + queued;
  badge.className = `badge badge-${s.phase}`;
}

function renderControls(s) {
  $('#btn-start').disabled = !s.can_start || state.busy;
  $('#btn-start').title = s.start_hint || 'runs scripts/phase2.sh: pre-flight, data check, '
    + 'smoke test, then the real run (resumes from ckpt_last.pt)';
  for (const id of ['#btn-stop', '#btn-stop-after', '#btn-stop-at']) {
    $(id).disabled = !s.can_stop || state.busy;
  }
  $('#btn-cancel-stop').disabled = !s.stop || state.busy;
  $('#btn-start').textContent = s.step == null ? 'Start run' : `Resume from ${fmt.int(s.step + 1)}`;
}

function renderProgress(s) {
  const last = s.last || {};
  const pct = s.progress == null ? null : Math.min(1, s.progress);
  $('#hero-step').textContent = s.step == null ? 'no steps logged'
    : `step ${fmt.int(s.step)}${s.max_steps ? ` / ${fmt.int(s.max_steps)}` : ''}`;
  $('#hero-sub').textContent = s.step == null
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

  const mark = $('#meter-stop');
  if (s.stop && !s.stop.now && s.max_steps) {
    mark.hidden = false;
    mark.style.left = `calc(${Math.min(100, (s.stop.target / s.max_steps) * 100)}% - 1px)`;
    mark.title = `queued stop at step ${fmt.int(s.stop.target)}`;
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
  drawCharts();
}

function drawCharts() {
  for (const [key, spec] of Object.entries(state.charts)) {
    const host = $(`.chart[data-chart="${key}"]`);
    const tableHost = $(`.chart-table[data-table="${key}"]`);
    if (!host) continue;
    if (host.hidden) chartTable(tableHost, spec); else lineChart(host, spec);
  }
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
  host.textContent = '';
  host.appendChild(rows.length
    ? table(['#', 'started', 'steps', 'loss (ema)', 'best val', 'tok/s', 'wall', 'ended'], rows,
      { currentRow: s.pid ? 0 : -1 })
    : Object.assign(document.createElement('div'),
      { className: 'chart-empty', textContent: 'No sessions logged yet.' }));
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
  add('launch', s.can_start || s.pid ? code(`scripts/phase2.sh   (run ${s.run})`) : null);

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

function renderRuns(runs) {
  const sel = $('#run-select');
  const labels = runs.map((r) => `${r.run}${r.phase !== 'idle' ? ` — ${r.phase}` : ''}`);
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

/* ---------------------------------------------------------------- poll loop ----------- */

function pollInterval(phase) {
  /* Fast while something is happening, lazy when nothing is: a step takes ~9s, so 2s is
   * already faster than the data changes, and an idle run needs no attention at all. */
  return phase === 'idle' ? 10000 : 2500;
}

async function refresh() {
  if (!state.run) return;
  try {
    const q = state.logFile ? `?lines=400&file=${encodeURIComponent(state.logFile)}` : '?lines=400';
    const [status, log, runs] = await Promise.all([
      api(`/api/run/${encodeURIComponent(state.run)}`),
      api(`/api/run/${encodeURIComponent(state.run)}/log${q}`),
      api('/api/runs'),
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
    live(`updated ${new Date().toLocaleTimeString()}`, 'on');
  } catch (err) {
    document.body.classList.add('stale');
    live(`no answer from the portal — ${err.message}`, 'err');
  } finally {
    schedule();
  }
}

function schedule(delay) {
  clearTimeout(state.timer);
  if (document.hidden) return;  // a background tab does not need to poll
  const ms = delay != null ? delay : pollInterval(state.status ? state.status.phase : 'idle');
  state.timer = setTimeout(refresh, ms);
}

/** Run an action, then poll hard for a few seconds so the phase badge reacts immediately. */
async function act(fn, okPrefix) {
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

/* ---------------------------------------------------------------- wiring -------------- */

function selectRun(run) {
  state.run = run;
  state.logFile = null;
  state.status = null;
  flash('');
  $('#log-select').dataset.run = '';
  schedule(0);
}

function wire() {
  $('#run-select').addEventListener('change', (e) => selectRun(e.target.value));

  $('#btn-start').addEventListener('click', () => {
    const after = $('#stop-after').value.trim();
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/start`, {
      stop_after: after ? Number(after) : null,
      skip_smoke: $('#skip-smoke').checked,
    }), 'Launching.');
  });

  $('#btn-stop').addEventListener('click', () => {
    const at = state.status && state.status.step;
    if (!confirm(`Stop '${state.run}' after the current step?\n\n`
      + `It saves ckpt_last.pt at step ~${fmt.int(at)} and exits; starting again resumes `
      + 'there with no loss spike.')) return;
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/stop`, { mode: 'now' }),
      'Stop requested.');
  });

  $('#btn-stop-after').addEventListener('click', () => {
    const n = prompt('Train how many more steps, then save and exit?', '500');
    if (!n) return;
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/stop`,
      { mode: 'after', steps: Number(n) }), 'Queued.');
  });

  $('#btn-stop-at').addEventListener('click', () => {
    const cur = (state.status && state.status.step) || 0;
    const n = prompt('Finish which step, then save and exit?', String(cur + 1000));
    if (!n) return;
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/stop`,
      { mode: 'at', steps: Number(n) }), 'Queued.');
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

  /* Charts are sized from their container, so a resize needs a redraw, not a refetch. */
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawCharts, 150);
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) schedule(0);
  });
}

async function boot() {
  wire();
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

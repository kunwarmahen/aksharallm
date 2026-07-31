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

/** Points inside [a, b], plus the one on each side, so a clipped line still reaches the
 * edges of the plot instead of stopping short.  Empty when the window holds no data. */
function clipPts(pts, a, b) {
  const first = pts.findIndex(([x]) => x >= a);
  if (first < 0 || pts[first][0] > b) return [];
  let last = pts.length - 1;
  while (pts[last][0] > b) last--;
  return pts.slice(Math.max(0, first - 1), last + 2);
}

let clipUid = 0;

/**
 * Draw a line chart into `host`.
 *
 * spec = {
 *   series: [{ name, color, x:[], y:[], faint?, dots?, fmt? }],
 *   yFmt, xFmt, height, rules: [{ y, label }], legend: bool,
 *   zoom: { x0, x1 } | null,   // the visible x-window, in data units
 *   onZoom: (win|null) => {}   // drag committed a new window, or a reset
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
  if ((series.length > 1 || (spec.spans || []).length) && spec.legend !== false) {
    const legend = document.createElement('div');
    legend.className = 'legend';
    if ((spec.spans || []).length) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const sw = document.createElement('span');
      sw.className = 'legend-swatch band';
      item.append(sw, document.createTextNode(spec.spanLabel || 'training'));
      legend.appendChild(item);
    }
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

  /* The x-window: the whole run, or whatever the reader dragged out.  A zoom that lands on
   * a gap in the data (or on a run that has since been switched out) is ignored, so the
   * chart can never come back blank. */
  const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let view = series;
  if (spec.zoom && spec.zoom.x1 > spec.zoom.x0) {
    const a = Math.max(x0, spec.zoom.x0);
    const b = Math.min(x1, spec.zoom.x1);
    const win = series.map((s) => ({ ...s, pts: clipPts(s.pts, a, b) })).filter((s) => s.pts.length);
    if (b > a && win.length) { x0 = a; x1 = b; view = win; }
  }
  const zoomed = view !== series;
  if (x1 === x0) x1 = x0 + 1;

  /* y refits to what is inside the window — that is the whole point of zooming a loss
   * curve.  The kept neighbours are excluded, or one off-screen spike would set the scale;
   * so is a threshold rule, which would otherwise pin the axis to the clip value. */
  const ys = view.flatMap((s) => s.pts.filter(([x]) => x >= x0 && x <= x1).map((p) => p[1]))
    .concat(zoomed ? [] : (spec.rules || []).map((r) => r.y));
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
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

  /* Shaded periods behind everything: "the GPU was training here". A neutral wash, not a
   * categorical hue — it is context for the series, and must not compete with it. */
  for (const sp of spec.spans || []) {
    const a = Math.max(sx(sp.start), pad.l);
    const b = Math.min(sx(sp.end), pad.l + pw);
    if (b <= a) continue;
    const band = el('rect', {
      class: 'span-band', x: a, y: pad.t, width: Math.max(b - a, 1.5), height: ph,
    }, svg);
    el('title', {}, band).textContent = sp.label || 'training';
  }

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

  /* Marks live in a clipped group: zoomed in, the kept neighbours sit outside the plot and
   * their segments must be trimmed at the frame, not drawn over the axis. */
  const clipId = `plot-clip-${++clipUid}`;
  el('rect', { x: pad.l, y: pad.t, width: pw, height: ph },
    el('clipPath', { id: clipId }, el('defs', {}, svg)));
  const plot = el('g', { 'clip-path': `url(#${clipId})` }, svg);

  for (const s of view) {
    const d = s.pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(' ');
    const path = el('path', { class: 'series-line' + (s.faint ? ' faint' : ''), d }, plot);
    path.style.stroke = `var(${s.color})`;
    /* Sparse series (validation loss lands every 1000 steps) get markers: a 3-point line
     * is hard to see, and the markers carry a second, non-colour cue. */
    if (s.dots && s.pts.length <= 80) {
      for (const [x, y] of s.pts) {
        const dot = el('circle', { class: 'series-dot', cx: sx(x), cy: sy(y), r: 4.5 }, plot);
        dot.style.fill = `var(${s.color})`;
      }
    }
    /* Direct-label the newest point of *one* series — selectively, never every point, and
     * never two at once: on a converged run the ema and the validation loss land on the
     * same pixel and the two labels overprint each other. The rest is in the tooltip. */
    if (s.label === true) {
      /* The newest point *in the window*, not the neighbour kept beyond its right edge. */
      const [lx, ly] = s.pts.filter(([x]) => x <= x1).pop() || s.pts[s.pts.length - 1];
      const tx = el('text', {
        class: 'axis-label', x: Math.min(sx(lx) + 6, pad.l + pw), y: sy(ly) - 7,
        'text-anchor': sx(lx) > pad.l + pw - 46 ? 'end' : 'start',
      }, svg);
      tx.textContent = (s.fmt || spec.yFmt || fmt.num)(ly);
      tx.style.fill = `var(${s.color})`;
    }
  }

  /* ---- crosshair + tooltip ---- */
  const primary = view.reduce((a, b) => (b.pts.length > a.pts.length ? b : a));
  /* The drag-to-zoom selection, under the crosshair so the readout stays legible. */
  const sel = el('rect', { class: 'zoom-band', x: 0, y: pad.t, width: 0, height: ph }, svg);
  sel.style.display = 'none';
  const cross = el('line', { class: 'crosshair', y1: pad.t, y2: pad.t + ph, x1: 0, x2: 0 }, svg);
  cross.style.display = 'none';
  const marks = view.map((s) => {
    const c = el('circle', { class: 'series-dot', r: 4.5, cx: 0, cy: 0 }, svg);
    c.style.fill = `var(${s.color})`;
    c.style.display = 'none';
    return c;
  });
  const tip = $('#tooltip');
  /* The hit area is the whole plot, so there is nothing to land on precisely. */
  const hit = el('rect', { class: 'hit', x: pad.l, y: pad.t, width: pw, height: ph }, svg);

  /* Pointer x in viewBox pixels and in data units — the chart is drawn at a fixed viewBox
   * width and scaled by CSS, so the two differ by the element's own scale factor. */
  const xAt = (evt) => {
    const box = svg.getBoundingClientRect();
    const px = ((evt.clientX - box.left) / box.width) * width;
    return { px, val: x0 + ((px - pad.l) / pw) * (x1 - x0) };
  };

  const show = (evt) => {
    const xVal = xAt(evt).val;
    let bi = 0;
    for (let i = 1; i < primary.pts.length; i++) {
      if (Math.abs(primary.pts[i][0] - xVal) < Math.abs(primary.pts[bi][0] - xVal)) bi = i;
    }
    const step = primary.pts[bi][0];
    cross.setAttribute('x1', sx(step));
    cross.setAttribute('x2', sx(step));
    cross.style.display = '';

    let html = `<div class="tt-head">step ${fmt.int(step)}</div>`;
    view.forEach((s, i) => {
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
  hit.addEventListener('pointerleave', hide);
  svg.addEventListener('pointerleave', hide);

  /* ---- drag across the plot to zoom into that x-window ----
   * The y-axis refits to the selection on redraw, so this is a real zoom, not a crop.
   * Double-click anywhere in the chart goes back to the whole run. */
  if (!spec.onZoom) {
    hit.addEventListener('pointermove', show);
    return;
  }
  let from = null;
  /* The five-second poll redraws this chart from scratch. Mid-drag that would throw the
   * selection away under the reader's finger, so the flag holds the redraw off. */
  const clearDrag = () => { from = null; sel.style.display = 'none'; delete host.dataset.dragging; };

  hit.addEventListener('pointerdown', (evt) => {
    if (evt.button) return;                       /* left button only */
    from = xAt(evt);
    host.dataset.dragging = '1';
    hit.setPointerCapture(evt.pointerId);
    hide();
  });
  hit.addEventListener('pointermove', (evt) => {
    if (!from) { show(evt); return; }
    const now = xAt(evt);
    const a = Math.max(Math.min(from.px, now.px), pad.l);
    const b = Math.min(Math.max(from.px, now.px), pad.l + pw);
    sel.setAttribute('x', a);
    sel.setAttribute('width', Math.max(b - a, 0));
    sel.style.display = '';
  });
  hit.addEventListener('pointerup', (evt) => {
    if (!from) return;
    const now = xAt(evt);
    const start = from;
    clearDrag();
    /* Under a few pixels this was a click, not a drag — zooming there would be a trap. */
    if (Math.abs(now.px - start.px) < 6) return;
    spec.onZoom({ x0: Math.min(start.val, now.val), x1: Math.max(start.val, now.val) });
  });
  hit.addEventListener('pointercancel', clearDrag);
  /* Capture lost some other way (a tab switch mid-drag) must not leave redraws blocked. */
  hit.addEventListener('lostpointercapture', clearDrag);
  svg.addEventListener('dblclick', () => spec.onZoom(null));

  /* The corner control: a hint on hover, or the way back out once zoomed. */
  const tools = document.createElement('div');
  tools.className = 'chart-tools';
  if (zoomed) {
    const xFmt = spec.xFmt || fmt.compact;
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'ghost zoom-reset';
    reset.textContent = `${xFmt(x0)}–${xFmt(x1)} ✕`;
    reset.title = 'back to the whole run (or double-click the chart)';
    reset.addEventListener('click', () => spec.onZoom(null));
    tools.appendChild(reset);
  } else {
    const hint = document.createElement('span');
    hint.className = 'zoom-hint';
    hint.textContent = 'drag to zoom';
    tools.appendChild(hint);
  }
  host.appendChild(tools);
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
  schedule: null,
  gpu: null,
  gpuWindow: '3600',
  log: null,
  logFile: null,     // null = whichever file was written most recently
  timer: null,
  charts: {},        // last spec per chart, so a resize can redraw without a fetch
  zoom: {},          // per chart: the dragged-out { x0, x1 } window, or absent for all of it
  busy: false,
  view: 'dashboard', // 'dashboard' | 'code'
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
  $('#btn-start').disabled = !s.can_start || state.busy;
  $('#btn-start').title = s.start_hint || 'runs scripts/phase2.sh: pre-flight, data check, '
    + 'smoke test, then the real run (resumes from ckpt_last.pt)';
  $('#btn-stop').disabled = !s.can_stop || state.busy;
  /* A bounded stop needs a step to count from, which a pre-flight doesn't have yet. */
  for (const id of ['#btn-stop-after', '#btn-stop-at']) {
    $(id).disabled = !s.can_bound || state.busy;
  }
  $('#btn-stop').textContent = s.phase === 'launching' ? 'Abort launch' : 'Stop now';
  $('#btn-cancel-stop').disabled = !s.stop || state.busy;
  $('#btn-start').textContent = s.step == null ? 'Start run' : `Resume from ${fmt.int(s.step + 1)}`;
}

function renderProgress(s) {
  const last = s.last || {};
  const pct = s.progress == null ? null : Math.min(1, s.progress);
  $('#hero-step').textContent = s.step == null ? 'no steps logged'
    : `step ${fmt.int(s.step)}${s.max_steps ? ` / ${fmt.int(s.max_steps)}` : ''}`;
  $('#hero-sub').textContent = s.phase === 'launching'
    ? `pre-flight (${(s.launcher && s.launcher.stage) || '?'}) — tests, data check and a `
      + '50-step smoke test run before training starts'
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
  drawCharts();
}

function drawCharts() {
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

const escHtml = (s) => String(s == null ? '' : s).replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

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
    gpu.current_run ? `${gpu.current_run} is training` : 'no run training',
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

function wireGpu() {
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

function wireSchedule() {
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

/* ---------------------------------------------------------------- code tab ------------ */

/* The second half of the portal: read the project, and ask a model on this machine what a
 * selection is doing. Everything here is inert until the tab is opened for the first time —
 * a dashboard left up overnight should not be listing files or loading a model.
 *
 * Three small engines: a filterable file list, a numbered code pane you select in, and a
 * streaming reader that turns server-sent events into markdown as they arrive. */

const code = {
  files: [],
  dirs: [],
  root: '',
  dir: '',           // the folder being browsed, '' = where the portal is running
  path: null,
  text: '',
  lines: [],
  lang: '',
  doc: null,
  start: null,       // 1-based, inclusive
  end: null,
  snippet: null,     // the exact characters, when the selection is not whole lines
  model: localStorage.getItem('aksharallm-explain-model') || '',
  messages: [],      // the follow-up conversation for the current selection
  answer: '',
  abort: null,
  loaded: false,
  anchor: null,      // gutter click-and-drag origin
};

/* --- markdown ------------------------------------------------------------------------ */

const escapeHtml = (s) => s.replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Just enough markdown for an explanation: fences, headings, lists, bold, inline code.
 *  Hand-written for the same reason as the charts — and because the input is a stream, so
 *  it is re-rendered from scratch on every chunk and has to stay cheap. */
function renderMarkdown(src) {
  const out = [];
  let list = null;      // 'ul' | 'ol' | null
  let para = [];

  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const closePara = () => {
    if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; }
  };
  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    // [text](url) -> a link. The Docs tab rewires .md links to load in the reader and
    // sends http(s) links to a new tab (see loadDoc); elsewhere they're plain anchors.
    .replace(/(^|[^!])\[([^\]]+)\]\(([^)]+)\)/g, '$1<a href="$3">$2</a>');

  const parts = src.split(/```/);
  parts.forEach((block, i) => {
    if (i % 2) {                                   /* inside a fence */
      closePara(); closeList();
      const nl = block.indexOf('\n');
      const lang = (nl >= 0 ? block.slice(0, nl) : '').trim().toLowerCase().replace(/[^a-z0-9-]/g, '');
      const body = nl >= 0 ? block.slice(nl + 1) : block;
      // Keep the fence language as a class, so the Docs tab can find ```mermaid blocks and
      // render them as diagrams. Harmless to every other caller (just an extra class).
      const cls = lang ? ` language-${lang}` : '';
      out.push(`<pre class="md-code${cls}"><code>${escapeHtml(body.replace(/\n$/, ''))}</code></pre>`);
      return;
    }
    const lines = block.split('\n');
    const isSep = (s) => /^[\s|:-]+$/.test(s) && s.includes('-') && s.includes('|');
    const cells = (s) => s.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li].replace(/\s+$/, '');
      if (!line.trim()) { closePara(); closeList(); continue; }
      // GFM table: a header row of `| … |`, then a `|---|---|` separator, then body rows.
      if (line.includes('|') && li + 1 < lines.length && isSep(lines[li + 1])) {
        closePara(); closeList();
        const head = cells(line);
        const rows = [];
        li += 2;
        while (li < lines.length && lines[li].trim() && lines[li].includes('|')) {
          rows.push(cells(lines[li])); li++;
        }
        li--;  // the for-loop increment will step past the last consumed row
        out.push('<table><thead><tr>'
          + head.map((c) => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>'
          + rows.map((r) => '<tr>'
            + head.map((_, k) => `<td>${inline(r[k] || '')}</td>`).join('') + '</tr>').join('')
          + '</tbody></table>');
        continue;
      }
      // Blockquote: one or more consecutive `>` lines become one <blockquote>.
      if (/^>\s?/.test(line)) {
        closePara(); closeList();
        const buf = [];
        while (li < lines.length && /^>\s?/.test(lines[li])) {
          buf.push(lines[li].replace(/^>\s?/, '')); li++;
        }
        li--;  // the for-loop increment steps past the last quoted line
        out.push(`<blockquote>${inline(buf.join(' '))}</blockquote>`);
        continue;
      }
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        closePara(); closeList();
        const lvl = Math.min(6, h[1].length + 2);
        out.push(`<h${lvl}>${inline(h[2])}</h${lvl}>`);
        continue;
      }
      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ul || ol) {
        closePara();
        const want = ul ? 'ul' : 'ol';
        if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
        out.push(`<li>${inline((ul || ol)[1])}</li>`);
        continue;
      }
      closeList();
      para.push(line.trim());
    }
  });
  closePara(); closeList();
  return out.join('\n');
}

/* --- syntax highlighting -------------------------------------------------------------- */

const KEYWORDS = {
  python: /\b(def|class|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|import|from|as|with|try|except|finally|raise|yield|lambda|global|nonlocal|assert|pass|break|continue|async|await|del)\b/g,
  javascript: /\b(const|let|var|function|return|if|else|for|while|of|in|new|class|extends|try|catch|finally|throw|typeof|instanceof|async|await|null|undefined|true|false|this|export|import|from|delete)\b/g,
  bash: /\b(if|then|elif|else|fi|for|while|do|done|case|esac|function|return|local|export|readonly|source|exit|set|trap)\b/g,
  yaml: null, toml: null, markdown: null, css: null, html: null, json: null,
};

/** Highlight a whole file at once, because the interesting cases span lines: a Python
 *  docstring is not a string on any single line of it. Returns one HTML string per line. */
function highlight(text, lang) {
  const lines = text.split('\n');
  const kw = KEYWORDS[lang];
  const comment = lang === 'python' || lang === 'bash' || lang === 'yaml' || lang === 'toml'
    ? '#' : (lang === 'javascript' || lang === 'css' ? '//' : null);
  let triple = null;   // the open """ or ''' delimiter, or null

  return lines.map((line) => {
    if (!lang) return escapeHtml(line);

    /* Multi-line strings first: while one is open, the whole line is string. */
    if (triple) {
      const close = line.indexOf(triple);
      if (close < 0) return `<span class="s">${escapeHtml(line)}</span>`;
      const head = line.slice(0, close + 3);
      triple = null;
      return `<span class="s">${escapeHtml(head)}</span>` + hl(line.slice(close + 3));
    }
    const open = lang === 'python' ? line.match(/("""|''')/) : null;
    if (open) {
      const at = open.index;
      const rest = line.slice(at + 3);
      const closeAt = rest.indexOf(open[1]);
      if (closeAt < 0) {
        triple = open[1];
        return hl(line.slice(0, at)) + `<span class="s">${escapeHtml(line.slice(at))}</span>`;
      }
      return hl(line.slice(0, at))
        + `<span class="s">${escapeHtml(line.slice(at, at + 3 + closeAt + 3))}</span>`
        + hl(line.slice(at + 3 + closeAt + 3));
    }
    return hl(line);

    function hl(src) {
      if (!src) return '';
      /* Comments swallow the rest of the line, so find the first one that is not inside a
       * quoted string — the cheap way is to walk the line once. */
      if (comment) {
        let q = null;
        for (let i = 0; i < src.length; i++) {
          const c = src[i];
          if (q) { if (c === q && src[i - 1] !== '\\') q = null; continue; }
          if (c === '"' || c === "'") { q = c; continue; }
          if (src.startsWith(comment, i)) {
            return hl(src.slice(0, i)) + `<span class="c">${escapeHtml(src.slice(i))}</span>`;
          }
        }
      }
      let html = escapeHtml(src)
        .replace(/(&quot;[^&]*?&quot;|&#39;[^&]*?&#39;|'[^']*'|"[^"]*")/g, '<span class="s">$1</span>');
      if (kw) html = html.replace(kw, '<span class="k">$1</span>');
      html = html.replace(/\b(\d+\.?\d*(e-?\d+)?)\b/g, '<span class="n">$1</span>');
      return html;
    }
  });
}

/* --- file list ------------------------------------------------------------------------ */

/** The immediate children of `dir` — the folders you can go down into from here. */
function subdirs(dir) {
  const prefix = dir ? dir + '/' : '';
  const depth = dir ? dir.split('/').length : 0;
  return code.dirs
    .filter((d) => d.startsWith(prefix) && d.split('/').length === depth + 1)
    .map((d) => ({ path: d, name: d.split('/').pop() }));
}

function goto(dir) {
  code.dir = dir;
  localStorage.setItem('aksharallm-code-dir', dir);
  renderTree();
  $('#file-tree').scrollTop = 0;
}

function renderTree() {
  const host = $('#file-tree');
  const crumbs = $('#file-crumbs');
  const q = $('#file-filter').value.trim().toLowerCase();
  host.textContent = '';
  crumbs.textContent = '';

  /* Breadcrumbs are the way back up: every ancestor is one click, including the root. */
  const rootName = code.root.split('/').filter(Boolean).pop() || '/';
  const trail = [{ path: '', name: rootName }];
  if (code.dir) {
    code.dir.split('/').forEach((part, i, all) => {
      trail.push({ path: all.slice(0, i + 1).join('/'), name: part });
    });
  }
  trail.forEach((step, i) => {
    if (i) crumbs.append(Object.assign(document.createElement('span'),
      { className: 'crumb-sep', textContent: '/' }));
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'crumb' + (i === trail.length - 1 ? ' on' : '');
    btn.dataset.dir = step.path;
    btn.textContent = step.name;
    crumbs.appendChild(btn);
  });

  /* Filtering searches the whole tree, not just this folder — otherwise you would have to
   * know where a file is before you could find it. */
  if (q) {
    const hits = code.files.filter((f) => f.path.toLowerCase().includes(q));
    $('#file-count').textContent = `${hits.length} of ${code.files.length} files match`;
    let dir = null;
    for (const f of hits.slice(0, 400)) {
      if (f.dir !== dir) {
        dir = f.dir;
        host.append(Object.assign(document.createElement('div'),
          { className: 'file-dir', textContent: dir || rootName }));
      }
      host.appendChild(fileButton(f));
    }
    if (!hits.length) {
      host.append(Object.assign(document.createElement('div'),
        { className: 'file-dir', textContent: 'nothing matches' }));
    }
    return;
  }

  const folders = subdirs(code.dir);
  const here = code.files.filter((f) => f.dir === code.dir);
  $('#file-count').textContent =
    `${folders.length} folder${folders.length === 1 ? '' : 's'} · ${here.length} file`
    + `${here.length === 1 ? '' : 's'}`;

  if (code.dir) {                       /* the way up, one level */
    const up = document.createElement('button');
    up.type = 'button';
    up.className = 'folder up';
    up.dataset.dir = code.dir.split('/').slice(0, -1).join('/');
    up.textContent = '../';
    host.appendChild(up);
  }
  for (const d of folders) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'folder';
    btn.dataset.dir = d.path;
    btn.textContent = d.name + '/';
    host.appendChild(btn);
  }
  for (const f of here) host.appendChild(fileButton(f));
  if (!folders.length && !here.length) {
    host.append(Object.assign(document.createElement('div'),
      { className: 'file-dir', textContent: 'nothing readable here' }));
  }
}

function fileButton(f) {
  const btn = document.createElement('button');
  btn.className = 'file' + (f.path === code.path ? ' on' : '');
  btn.type = 'button';
  btn.dataset.path = f.path;
  btn.title = `${f.path} · ${fmt.bytes(f.size)}`;
  btn.textContent = f.name;
  return btn;
}

/* --- the code pane -------------------------------------------------------------------- */

async function openFile(path) {
  try {
    const data = await api(`/api/source/file?path=${encodeURIComponent(path)}`);
    code.path = data.path;
    code.text = data.text;
    code.lang = data.lang;
    code.doc = data.doc;
    code.lines = highlight(data.text, data.lang);
    code.start = code.end = null;
    code.snippet = null;
    code.messages = [];
    code.answer = '';
    $('#explain-out').textContent = '';
    $('#followup-form').hidden = true;
    $('#btn-explain-clear').hidden = true;
    explainStatus('');
    $('#code-path').textContent = data.path;
    $('#code-note').textContent =
      `${fmt.int(data.lines)} lines · ${fmt.bytes(data.size)}`
      + (data.doc ? ` · deep dive: ${data.doc}` : '');
    renderCode();
    /* Follow the file into its folder, so opening a filter hit leaves you somewhere you
     * can keep browsing rather than back at the root. */
    const dir = data.path.includes('/') ? data.path.replace(/\/[^/]*$/, '') : '';
    if (!$('#file-filter').value.trim()) code.dir = dir;
    renderTree();
    localStorage.setItem('aksharallm-code-path', data.path);
    localStorage.setItem('aksharallm-code-dir', code.dir);
    $('#code-view').scrollTop = 0;
  } catch (err) {
    $('#code-note').textContent = err.message;
  }
}

function renderCode() {
  const host = $('#code-view');
  host.textContent = '';
  const frag = document.createDocumentFragment();
  code.lines.forEach((html, i) => {
    const row = document.createElement('div');
    row.className = 'cl';
    row.dataset.line = String(i + 1);
    const num = document.createElement('span');
    num.className = 'ln';
    num.textContent = String(i + 1);
    const src = document.createElement('code');
    src.className = 'lc';
    src.innerHTML = html || '​';
    row.append(num, src);
    frag.appendChild(row);
  });
  host.appendChild(frag);
  markSelection();
}

function setSelection(start, end, snippet) {
  code.start = Math.min(start, end);
  code.end = Math.max(start, end);
  code.snippet = snippet || null;
  code.messages = [];          /* a new selection is a new conversation */
  markSelection();
}

function markSelection() {
  for (const row of $$('#code-view .cl')) {
    const n = Number(row.dataset.line);
    row.classList.toggle('sel', code.start != null && n >= code.start && n <= code.end);
  }
  const label = $('#explain-sel');
  const btn = $('#btn-explain');
  if (code.start == null) {
    label.textContent = code.path ? 'nothing selected' : 'no file open';
    btn.disabled = true;
    return;
  }
  const n = code.end - code.start + 1;
  label.textContent = code.start === code.end
    ? `line ${code.start}` + (code.snippet ? ' (part of it)' : '')
    : `lines ${code.start}–${code.end} · ${n} lines`;
  btn.disabled = false;
}

/** Turn whatever the browser thinks is selected into a line range. Dragging across the
 *  code, double-clicking a word and shift-clicking all end up here. */
function selectionToLines() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return;
  const pane = $('#code-view');
  const row = (node) => {
    const el = node && (node.nodeType === 1 ? node : node.parentElement);
    return el && pane.contains(el) ? el.closest('.cl') : null;
  };
  const a = row(sel.anchorNode);
  const b = row(sel.focusNode);
  if (!a || !b) return;
  const text = sel.toString();
  const from = Number(a.dataset.line), to = Number(b.dataset.line);
  const whole = text.trim() === slice(Math.min(from, to), Math.max(from, to)).trim();
  setSelection(from, to, whole ? null : text);
}

const slice = (from, to) => code.text.split('\n').slice(from - 1, to).join('\n');

/* --- asking --------------------------------------------------------------------------- */

function explainStatus(text, kind = '') {
  const box = $('#explain-status');
  box.textContent = text;
  box.className = 'explain-status ' + kind;
}

async function loadModels() {
  const sel = $('#explain-model');
  try {
    const info = await api('/api/explain/models');
    sel.textContent = '';
    const names = info.models.map((m) => m.name);
    if (!names.includes(info.model)) names.unshift(info.model);
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    code.model = names.includes(code.model) ? code.model : info.model;
    sel.value = code.model;
    if (!info.available) {
      explainStatus(info.error, 'err');
    } else if (!info.models.some((m) => m.name === info.model)) {
      explainStatus(`${info.model} is configured but not pulled — `
        + `run: ollama pull ${info.model}`, 'err');
    }
    /* The explainer and the trainer share one card, and the trainer got there first. */
    const warn = $('#explain-warn');
    if (info.training && info.training.length && !info.on_cpu) {
      warn.innerHTML = `<strong>${escapeHtml(info.training.join(', '))}</strong> is training `
        + 'on this GPU. Asking here loads the model onto the same card and can run it out '
        + 'of memory. To read while training, set <code>num_gpu: 0</code> under '
        + '<code>explain:</code> in <code>configs/portal.yaml</code> — slower, but it stays '
        + 'off the GPU entirely.';
      warn.hidden = false;
    } else if (info.on_cpu) {
      warn.textContent = 'The explainer is pinned to the CPU (num_gpu: 0), so it cannot '
        + 'disturb a training run — but a 12B model can take minutes to produce its first '
        + 'word this way. A smaller model is the better trade while a run is going.';
      warn.hidden = false;
    } else {
      warn.hidden = true;
    }
  } catch (err) {
    explainStatus(err.message, 'err');
  }
}

/** A reasoning model's scratchpad, folded away. Empty for a model that does not think. */
const thinkingBlock = (text) => (text
  ? `<details class="md-think"><summary>reasoning (${fmt.int(text.length)} chars)</summary>`
    + `<pre>${escapeHtml(text)}</pre></details>`
  : '');

/** Stream one answer. `question` null means the default ask; `follow` keeps the thread. */
async function explain(question, follow = false) {
  if (code.start == null || code.abort) return;
  const controller = new AbortController();
  code.abort = controller;
  $('#btn-explain').disabled = true;
  $('#btn-explain-stop').hidden = false;
  const started = Date.now();
  explainStatus(`${code.model} is reading ${code.path}…`, 'busy');

  const history = follow ? code.messages.slice() : [];
  if (follow) history.push({ role: 'user', content: question });
  else code.messages = [];

  let answer = '';
  let thinking = '';
  const out = $('#explain-out');
  if (!follow) out.textContent = '';
  const head = follow
    ? `<h4 class="md-ask">${escapeHtml(question)}</h4>`
    : '';
  const prefix = out.innerHTML + head;

  try {
    const res = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Portal': '1' },
      signal: controller.signal,
      body: JSON.stringify({
        path: code.path,
        start: code.start,
        end: code.end,
        snippet: code.snippet,
        model: code.model,
        question: follow ? question : (question || null),
        history,
      }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).error || msg; } catch { /* no body */ }
      throw new Error(msg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let stop = false;
    while (!stop) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.replace(/^data: /, '').trim();
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        if (evt.error) throw new Error(evt.error);
        /* A reasoning model talks to itself first. Show that it is working, and keep the
         * transcript foldable underneath — but never mix it into the answer. */
        if (evt.thinking) {
          thinking += evt.thinking;
          explainStatus(`${code.model} is thinking… ${fmt.int(thinking.length)} chars`,
            'busy');
        }
        if (evt.delta) {
          answer += evt.delta;
          out.innerHTML = prefix + thinkingBlock(thinking) + renderMarkdown(answer);
          out.scrollTop = out.scrollHeight;
        }
        if (evt.done) stop = true;
      }
    }
    if (follow) code.messages.push({ role: 'user', content: question });
    code.messages.push({ role: 'assistant', content: answer });
    out.innerHTML = prefix + thinkingBlock(thinking) + renderMarkdown(answer);
    if (!answer.trim()) {
      /* A reasoning model that spent its whole budget thinking. Say so, rather than
       * leaving an empty panel that looks like the portal broke. */
      explainStatus(thinking
        ? `${code.model} thought but never answered — it ran out of budget. Raise `
          + 'num_predict, or set think: false, in configs/portal.yaml.'
        : `${code.model} returned nothing.`, 'err');
    } else {
      explainStatus(`${code.model} · ${fmt.dur((Date.now() - started) / 1000)}`
        + (code.doc ? ` · the human-written version is ${code.doc}` : ''));
    }
    $('#followup-form').hidden = false;
    $('#btn-explain-clear').hidden = false;
  } catch (err) {
    if (err.name === 'AbortError') {
      explainStatus(`stopped after ${fmt.dur((Date.now() - started) / 1000)}`);
      if (answer) code.messages.push({ role: 'assistant', content: answer });
    } else {
      explainStatus(err.message, 'err');
    }
  } finally {
    code.abort = null;
    $('#btn-explain-stop').hidden = true;
    $('#btn-explain').disabled = code.start == null;
  }
}

function wireCode() {
  $('#file-filter').addEventListener('input', renderTree);

  $('#file-tree').addEventListener('click', (e) => {
    const folder = e.target.closest('.folder');
    if (folder) return goto(folder.dataset.dir);
    const btn = e.target.closest('.file');
    if (btn) openFile(btn.dataset.path);
  });

  $('#file-crumbs').addEventListener('click', (e) => {
    const crumb = e.target.closest('.crumb');
    if (crumb) { $('#file-filter').value = ''; goto(crumb.dataset.dir); }
  });

  const pane = $('#code-view');

  /* Click a line number for that line; drag or shift-click down the gutter for a block.
   * The gutter is a separate gesture from selecting text, so a reader can grab a 40-line
   * function without the browser scrolling away under a text selection. */
  pane.addEventListener('mousedown', (e) => {
    const num = e.target.closest('.ln');
    if (!num) return;
    e.preventDefault();
    const line = Number(num.parentElement.dataset.line);
    if (e.shiftKey && code.start != null) setSelection(code.start, line);
    else { code.anchor = line; setSelection(line, line); }
  });
  pane.addEventListener('mouseover', (e) => {
    if (code.anchor == null || !(e.buttons & 1)) return;
    const row = e.target.closest('.cl');
    if (row) setSelection(code.anchor, Number(row.dataset.line));
  });
  window.addEventListener('mouseup', () => { code.anchor = null; });

  /* Selecting the text itself works too, including part of a line. */
  pane.addEventListener('mouseup', () => setTimeout(selectionToLines, 0));
  pane.addEventListener('dblclick', () => setTimeout(selectionToLines, 0));

  $('#explain-model').addEventListener('change', (e) => {
    code.model = e.target.value;
    localStorage.setItem('aksharallm-explain-model', code.model);
  });

  for (const btn of $$('#lenses .ghost')) {
    btn.addEventListener('click', () => {
      for (const other of $$('#lenses .ghost')) other.classList.toggle('on', other === btn);
      if (code.start != null) explain(btn.dataset.ask || null);
    });
  }

  $('#btn-explain').addEventListener('click', () => {
    const lens = $('#lenses .on');
    explain((lens && lens.dataset.ask) || null);
  });
  $('#btn-explain-stop').addEventListener('click', () => code.abort && code.abort.abort());
  $('#btn-explain-clear').addEventListener('click', () => {
    code.messages = [];
    $('#explain-out').textContent = '';
    $('#followup-form').hidden = true;
    $('#btn-explain-clear').hidden = true;
    explainStatus('');
  });

  $('#followup-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('#followup').value.trim();
    if (!q || code.abort) return;
    $('#followup').value = '';
    explain(q, true);
  });
}

/** First time the tab is opened: fetch the file list and the model list, not before. */
async function openCodeTab() {
  if (code.loaded) return;
  code.loaded = true;
  try {
    const { files, dirs, root } = await api('/api/source');
    code.files = files;
    code.dirs = dirs;
    code.root = root;
    const lastDir = localStorage.getItem('aksharallm-code-dir') || '';
    code.dir = (lastDir && dirs.includes(lastDir)) ? lastDir : '';
    renderTree();
    await loadModels();
    const last = localStorage.getItem('aksharallm-code-path');
    const first = files.find((f) => f.path === last)
      || files.find((f) => f.path === 'aksharallm/model/transformer.py')
      || files.find((f) => f.path.startsWith('aksharallm/'));
    if (first) openFile(first.path);
  } catch (err) {
    code.loaded = false;
    $('#file-count').textContent = err.message;
  }
}

/* --- docs tab: read the same docs/*.md in the portal, diagrams and all ---------------
 * Content comes from /api/source/file (SourceTree already serves .md); the ordered list
 * from /api/docs. Mermaid is vendored locally and loaded LAZILY — only the first time this
 * tab is opened — so a dashboard left up overnight never pays for a 3 MB diagram library. */
const docState = { list: [], path: null, loaded: false };

let mermaidReady = null;
function ensureMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  if (!mermaidReady) {
    mermaidReady = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = '/static/mermaid.min.js';
      s.onload = () => resolve(window.mermaid);
      s.onerror = () => reject(new Error('could not load the diagram library'));
      document.head.appendChild(s);
    });
  }
  return mermaidReady;
}

async function renderDocDiagrams(container) {
  const pres = [...container.querySelectorAll('pre.language-mermaid')];
  if (!pres.length) return;
  let mermaid;
  try { mermaid = await ensureMermaid(); } catch { return; }  // keep the source if it won't load
  mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });
  const nodes = pres.map((pre) => {
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = pre.textContent;   // the raw diagram source, unescaped
    pre.replaceWith(div);
    return div;
  });
  try { await mermaid.run({ nodes }); } catch { /* a bad diagram just shows its source */ }
}

/** Resolve a doc-relative link ('02-tokenizer.md' from within 'docs/01-data.md') to a
 *  repo-relative path ('docs/02-tokenizer.md'), handling ./ and ../ and a #fragment. */
function resolveDocPath(base, href) {
  const target = href.split('#')[0];
  const parts = base.includes('/') ? base.slice(0, base.lastIndexOf('/')).split('/') : [];
  for (const seg of target.split('/')) {
    if (seg === '..') parts.pop();
    else if (seg && seg !== '.') parts.push(seg);
  }
  return parts.join('/');
}

async function loadDoc(path) {
  docState.path = path;
  for (const li of $$('#docs-list li')) li.classList.toggle('on', li.dataset.path === path);
  const reader = $('#docs-reader');
  reader.innerHTML = '<p class="docs-hint">loading…</p>';
  try {
    const res = await api(`/api/source/file?path=${encodeURIComponent(path)}`);
    reader.innerHTML = renderMarkdown(res.text || '');
    // Rewire links: another doc (.md) loads in this reader; anything external opens in a
    // new tab so a click never navigates the portal away from itself.
    for (const a of reader.querySelectorAll('a[href]')) {
      const href = a.getAttribute('href');
      if (/^https?:/i.test(href)) { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
      else if (/^#/.test(href)) { /* in-page anchor: harmless, leave it */ }
      else if (/\.md(#|$)/i.test(href)) { a.dataset.doc = resolveDocPath(path, href); a.setAttribute('href', '#'); }
      // any other repo-relative link is a source file -> open it in the Code tab, never
      // let it navigate the portal to a dead URL like /aksharallm/data/prepare.py
      else { a.dataset.src = resolveDocPath(path, href); a.setAttribute('href', '#'); }
    }
    reader.scrollTop = 0;
    await renderDocDiagrams(reader);
  } catch (err) {
    reader.innerHTML = `<p class="docs-hint">could not open ${escapeHtml(path)} — ${escapeHtml(err.message)}</p>`;
  }
}

async function openDocsTab() {
  if (docState.loaded) return;
  docState.loaded = true;
  try {
    const res = await api('/api/docs');
    docState.list = res.docs || [];
    $('#docs-list').innerHTML = docState.list.map((d) =>
      `<li data-path="${escapeHtml(d.path)}"><button type="button">${escapeHtml(d.title)}</button></li>`).join('');
    $('#docs-list').addEventListener('click', (e) => {
      const li = e.target.closest('li[data-path]');
      if (li) loadDoc(li.dataset.path);
    });
    // In-reader links: another doc loads in place; a source file opens in the Code tab.
    $('#docs-reader').addEventListener('click', (e) => {
      const doc = e.target.closest('a[data-doc]');
      if (doc) { e.preventDefault(); loadDoc(doc.dataset.doc); return; }
      const src = e.target.closest('a[data-src]');
      if (src) {
        e.preventDefault();
        const p = src.dataset.src;
        localStorage.setItem('aksharallm-code-path', p);  // so lazy init opens THIS file
        showView('code');
        openFile(p);
      }
    });
    const first = docState.list.find((d) => d.path.includes('00-')) || docState.list[0];
    if (first) loadDoc(first.path);
  } catch (err) {
    $('#docs-reader').innerHTML = `<p class="docs-hint">could not list docs — ${escapeHtml(err.message)}</p>`;
  }
}


/* ---------------------------------------------------------------- quantize tab -------
 * Make a trained model smaller and see what it cost. Every button here POSTs to
 * /api/quant/start, which shells out to `python -m aksharallm.quant` — the panel never
 * quantizes anything itself, so a job started here and one typed into a terminal produce
 * the same files and either can stop the other.
 *
 * The results table always shows the bf16 baseline row when there is one, because a
 * perplexity with nothing beside it is not a measurement.
 */

const quant = { timer: null, ckpts: [], status: null, loaded: false };

async function openQuantTab() {
  if (!quant.loaded) {
    quant.loaded = true;
    try {
      const data = await api('/api/quant/checkpoints');
      quant.ckpts = data.checkpoints || [];
      renderQuantCheckpoints();
    } catch (err) {
      $('#q-ckpt-note').textContent = `could not list checkpoints — ${err.message}`;
    }
  }
  pollQuant();
}

function renderQuantCheckpoints() {
  const sel = $('#q-ckpt');
  const usable = quant.ckpts.filter((c) => c.can_quantize);
  sel.innerHTML = usable.map((c) => {
    const bits = [c.step == null ? null : `step ${fmt.int(c.step)}`,
                  c.params == null ? null : `${fmt.compact(c.params)} params`,
                  fmt.bytes(c.size)].filter(Boolean).join(' · ');
    return `<option value="${escHtml(c.id)}">${escHtml(c.rel)} — ${escHtml(bits)}</option>`;
  }).join('');
  const already = quant.ckpts.filter((c) => c.quantized).length;
  $('#q-ckpt-note').textContent = usable.length
    ? (already ? `${already} quantized checkpoint${already > 1 ? 's' : ''} already exist and are not offered as sources — quantizing one again compounds the error.` : '')
    : 'No float checkpoints found. Train something first.';
  $('#q-run').disabled = !usable.length;
  $('#q-compare').disabled = !usable.length;
}

function renderQuantForm(st) {
  const methodSel = $('#q-method');
  if (methodSel.options.length === 0 && st.methods) {
    methodSel.innerHTML = st.methods.map((m) =>
      `<option value="${m.id}">${m.id.toUpperCase()}</option>`).join('');
  }
  const groupSel = $('#q-group');
  if (groupSel.options.length === 0 && st.groups) {
    groupSel.innerHTML = st.groups.map((g) =>
      `<option value="${g.value}"${g.value === 64 ? ' selected' : ''}>${escHtml(String(g.value === -1 ? 'per-channel' : g.value))}</option>`).join('');
  }
  const method = (st.methods || []).find((m) => m.id === methodSel.value);
  $('#q-method-note').textContent = method ? method.blurb : '';
  $('#q-steps-field').hidden = methodSel.value !== 'qat';

  const group = (st.groups || []).find((g) => String(g.value) === groupSel.value);
  $('#q-group-note').textContent = group ? group.label : '';

  /* int8 is symmetric-and-free; the group size barely matters there. Say so rather than
   * letting someone spend a run discovering it. */
  if ($('#q-bits').value === '8') {
    $('#q-group-note').textContent += '  ·  at 8 bits this makes almost no difference — int8 is free either way.';
  }

  const dev = st.device || {};
  $('#q-device-note').textContent = dev.reason || '';
  $('#q-device-note').classList.toggle('warn', (dev.training || []).length > 0);
}

function quantSpec(compare) {
  const spec = {
    checkpoint: $('#q-ckpt').value,
    method: $('#q-method').value || 'rtn',
    bits: Number($('#q-bits').value),
    group: Number($('#q-group').value),
    device: $('#q-device').value || null,
    bench: $('#q-bench').checked,
    save: $('#q-save').checked,
    compare: !!compare,
  };
  if (spec.method === 'qat') spec.qat_steps = Number($('#q-steps').value || 800);
  return spec;
}

function renderQuantStatus(st) {
  quant.status = st;
  renderQuantForm(st);

  const cur = st.current;
  const running = st.running;
  $('#q-stop').hidden = !running;
  $('#q-run').disabled = running || !quant.ckpts.some((c) => c.can_quantize);
  $('#q-compare').disabled = $('#q-run').disabled;

  if (cur) {
    const state = running ? 'running' : cur.state;
    const started = cur.started ? new Date(cur.started * 1000).toLocaleTimeString() : '';
    $('#q-state').textContent =
      `${state} — ${cur.method} on ${cur.checkpoint} (${cur.device}), started ${started}`;
    $('#q-cmd').innerHTML = `<code>python -m ${escHtml(cur.cmd || '')}</code>`;
  } else {
    $('#q-state').textContent = 'nothing running';
  }

  const log = $('#q-log');
  if (st.log && st.log.length) {
    const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
    log.textContent = st.log.join('\n');
    if (stick) log.scrollTop = log.scrollHeight;
  }
  renderQuantResults(st.results || []);
}

function renderQuantResults(rows) {
  const host = $('#q-results');
  if (!rows.length) {
    host.innerHTML = '<p class="q-empty">No measurements yet. A single scheme in isolation '
      + 'says very little — <strong>Compare all</strong> runs every method against the bf16 '
      + 'baseline on the same evaluation batches.</p>';
    return;
  }
  host.innerHTML = rows.map((r) => {
    const when = new Date(r.when * 1000).toLocaleString();
    const base = (r.bench || []).find((b) => /bf16/.test(b.label));
    const body = (r.bench || []).length ? `
      <table class="q-table">
        <thead><tr><th>scheme</th><th>size</th><th>ratio</th><th>perplexity</th>
          <th>vs bf16</th><th>tok/s</th></tr></thead>
        <tbody>${(r.bench || []).map((b) => {
          const ratio = base && b.nbytes ? (base.nbytes / b.nbytes) : null;
          const d = base && base.perplexity && b.perplexity ? b.perplexity - base.perplexity : null;
          const isBase = base && b.label === base.label;
          const dCell = isBase ? '<span class="q-dim">baseline</span>'
            : (d == null ? '–' : `<span class="${d > 0.2 ? 'q-bad' : d > 0.05 ? 'q-warn' : 'q-good'}">${d >= 0 ? '+' : ''}${fmt.num(d, 3)}</span>`);
          return `<tr${isBase ? ' class="q-baseline"' : ''}>
            <td><code>${escHtml(b.label)}</code></td>
            <td>${fmt.bytes(b.nbytes)}</td>
            <td>${ratio ? fmt.num(ratio, 2) + '×' : '–'}</td>
            <td>${b.perplexity == null ? '–' : fmt.num(b.perplexity, 3)}</td>
            <td>${dCell}</td>
            <td>${b.tok_s == null ? '–' : fmt.num(b.tok_s, 1)}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>` : quantReportOnly(r);
    const extra = [];
    if (r.awq) extra.push(`AWQ scaled ${r.awq.n_sites} sites, mean alpha ${fmt.num(r.awq.mean_alpha, 2)}`);
    if (r.qat) extra.push(`QAT ${r.qat.steps} steps, loss ${fmt.num(r.qat.loss_start, 4)} → ${fmt.num(r.qat.loss_end, 4)}`);
    if (r.out) extra.push(`wrote <code>${escHtml(r.out)}</code>`);
    return `
      <details class="q-run" open>
        <summary><span class="q-run-name">${escHtml(r.checkpoint || r.job)}</span>
          <span class="q-run-when">${escHtml(when)} · ${escHtml(r.device || '')}</span></summary>
        ${body}
        ${extra.length ? `<div class="q-extra">${extra.join(' · ')}</div>` : ''}
      </details>`;
  }).join('');
}

function quantReportOnly(r) {
  const t = r.totals;
  if (!t) return '<p class="q-empty">no measurements recorded for this job</p>';
  return `<table class="q-table"><tbody>
    <tr><td>quantized weights</td><td>${fmt.bytes(t.linear_float_bytes)} → ${fmt.bytes(t.linear_quant_bytes)}</td><td>${fmt.num(t.linear_ratio, 2)}×</td></tr>
    <tr><td>whole model</td><td>${fmt.bytes(t.model_float_bytes)} → ${fmt.bytes(t.model_quant_bytes)}</td><td>${fmt.num(t.model_ratio, 2)}×</td></tr>
    </tbody></table>
    <div class="q-extra">The whole-model ratio is the honest one — the embedding table is
    never quantized, so 4-bit is never 4× overall.</div>`;
}

async function pollQuant() {
  clearTimeout(quant.timer);
  if (state.view !== 'quant' || document.hidden) return;
  try {
    const st = await api('/api/quant?lines=250');
    renderQuantStatus(st);
    quant.timer = setTimeout(pollQuant, st.running ? 2000 : 8000);
  } catch (err) {
    $('#q-state').textContent = `no answer — ${err.message}`;
    quant.timer = setTimeout(pollQuant, 8000);
  }
}

async function startQuant(compare) {
  try {
    $('#q-run').disabled = $('#q-compare').disabled = true;
    const res = await post('/api/quant/start', quantSpec(compare));
    flash(`Quantizing: ${res.method} on ${res.checkpoint} (${res.device}). ${res.device_reason || ''}`, 'ok');
    $('#q-log').textContent = 'starting…';
    pollQuant();
  } catch (err) {
    flash(err.message, 'error');
    $('#q-run').disabled = $('#q-compare').disabled = false;
  }
}

function wireQuantTab() {
  $('#q-run').addEventListener('click', () => startQuant(false));
  $('#q-compare').addEventListener('click', () => startQuant(true));
  $('#q-stop').addEventListener('click', async () => {
    try { await post('/api/quant/stop', {}); flash('Stopped the quantization job.', 'ok'); }
    catch (err) { flash(err.message, 'error'); }
    pollQuant();
  });
  for (const id of ['#q-method', '#q-group', '#q-bits']) {
    $(id).addEventListener('change', () => { if (quant.status) renderQuantForm(quant.status); });
  }
}

/* ---------------------------------------------------------------- finetune tab -------
 * Teach the model something new without training the model.
 *
 * This tab leads with the memory budget rather than with the Run button, deliberately.
 * LoRA is the first thing in this project whose point is a *cost*, not a loss curve, and
 * the budget table makes the whole argument visible before anything is spent: what full
 * fine-tuning would need, what LoRA needs, what QLoRA needs, on the checkpoint actually
 * selected. Reading it is the lesson; running a job is optional.
 *
 * Every button POSTs to /api/lora/start, which shells out to
 * `python -m aksharallm.train.sft` — the panel never trains anything itself.
 */

const lora = { timer: null, ckpts: [], status: null, loaded: false, budget: null,
               budgetFor: null };

async function openLoraTab() {
  if (!lora.loaded) {
    lora.loaded = true;
    try {
      const data = await api('/api/lora/checkpoints');
      lora.ckpts = data.checkpoints || [];
      renderLoraCheckpoints();
    } catch (err) {
      $('#l-ckpt-note').textContent = `could not list checkpoints — ${err.message}`;
    }
  }
  pollLora();
  refreshBudget();
}

function renderLoraCheckpoints() {
  const sel = $('#l-ckpt');
  const usable = lora.ckpts.filter((c) => !c.error);
  sel.innerHTML = usable.map((c) => {
    const bits = [c.step == null ? null : `step ${fmt.int(c.step)}`,
                  c.params == null ? null : `${fmt.compact(c.params)} params`,
                  c.quantized ? 'already 4-bit' : null].filter(Boolean).join(' · ');
    return `<option value="${escHtml(c.id)}">${escHtml(c.rel)} — ${escHtml(bits)}</option>`;
  }).join('');
  const missing = usable.filter((c) => !c.tokenizer_ok).length;
  $('#l-ckpt-note').textContent = usable.length
    ? (missing ? `${missing} checkpoint(s) have no tokenizer on disk and cannot be fine-tuned safely.` : '')
    : 'No checkpoints found. Train a base model first.';
  $('#l-run').disabled = !usable.length;
}

function renderLoraForm(st) {
  if ($('#l-why').textContent === '') $('#l-why').textContent = st.why || '';

  const methodSel = $('#l-method');
  if (methodSel.options.length === 0 && st.methods) {
    methodSel.innerHTML = st.methods.map((m) =>
      `<option value="${m.id}"${m.id === 'qlora' ? ' selected' : ''}>${escHtml(m.label)}</option>`).join('');
  }
  const rankSel = $('#l-rank');
  if (rankSel.options.length === 0 && st.ranks) {
    rankSel.innerHTML = st.ranks.map((r) =>
      `<option value="${r.value}"${r.value === 8 ? ' selected' : ''}>${r.value}</option>`).join('');
  }
  const targetSel = $('#l-targets');
  if (targetSel.options.length === 0 && st.targets) {
    targetSel.innerHTML = st.targets.map((t) =>
      `<option value="${t.id}"${t.id === 'all-linear' ? ' selected' : ''}>${escHtml(t.id)}</option>`).join('');
  }
  const dataSel = $('#l-data');
  if (dataSel.options.length === 0) {
    const sets = st.datasets || [];
    dataSel.innerHTML = sets.map((d) =>
      `<option value="${escHtml(d.id)}">${escHtml(d.name)}</option>`).join('');
    $('#l-data-note').innerHTML = sets.length
      ? escHtml(`${sets.length} prepared dataset(s) under data/`)
      : `No SFT data prepared yet. Make some in a terminal:<br><code>${escHtml(st.data_hint || '')}</code>`;
    $('#l-run').disabled = $('#l-run').disabled || !sets.length;
  }
  const set = (st.datasets || []).find((d) => d.id === dataSel.value);
  if (set) {
    $('#l-data-note').textContent =
      `${fmt.int(set.blocks)} blocks of ${fmt.int(set.seq_len)} tokens · ${fmt.bytes(set.bytes)}`;
  }

  const method = (st.methods || []).find((m) => m.id === methodSel.value);
  $('#l-method-note').textContent = method ? method.blurb : '';
  /* Rank and target layers are meaningless for a full fine-tune; hiding them would make
   * the form jump, so they are disabled and explained instead. */
  const isFull = methodSel.value === 'full';
  rankSel.disabled = isFull;
  targetSel.disabled = isFull;
  $('#l-rank-note').textContent = isFull
    ? 'not used — a full fine-tune trains every weight, so there is no rank to choose.'
    : ((st.ranks || []).find((r) => String(r.value) === rankSel.value) || {}).label || '';
  $('#l-targets-note').textContent = isFull
    ? 'not used — every layer trains.'
    : (((st.targets || []).find((t) => t.id === targetSel.value) || {}).blurb || '');

  const dev = st.device || {};
  $('#l-device-note').textContent = dev.reason || '';
  $('#l-device-note').classList.toggle('warn', (dev.training || []).length > 0);
}

function loraSpec() {
  return {
    checkpoint: $('#l-ckpt').value,
    data_dir: $('#l-data').value,
    method: $('#l-method').value || 'qlora',
    r: Number($('#l-rank').value || 8),
    targets: $('#l-targets').value || 'all-linear',
    epochs: Number($('#l-epochs').value || 2),
    device: $('#l-device').value || null,
  };
}

/* ---- the budget table ---------------------------------------------------------------
 * Fetched per checkpoint and cached, because building it instantiates the model a few
 * times server-side. Re-fetched when the checkpoint or the target preset changes, since
 * those are the two things that move the numbers.
 */
async function refreshBudget() {
  const ckpt = $('#l-ckpt').value;
  const targets = $('#l-targets').value || 'all-linear';
  if (!ckpt) return;
  const key = `${ckpt}|${targets}`;
  if (lora.budgetFor === key) return;
  $('#l-budget').innerHTML = '<div class="q-empty">measuring…</div>';
  try {
    const b = await api(`/api/lora/budget?checkpoint=${encodeURIComponent(ckpt)}&targets=${encodeURIComponent(targets)}`);
    lora.budget = b;
    lora.budgetFor = key;
    renderBudget(b);
  } catch (err) {
    $('#l-budget').innerHTML = `<div class="q-empty">could not measure — ${escHtml(err.message)}</div>`;
  }
}

function renderBudget(b) {
  const rows = b.rows || [];
  const worst = Math.max(...rows.map((r) => r.total_bytes || 0), 1);
  $('#l-headline').textContent = b.headline || '';
  $('#l-budget').innerHTML = `
    <table class="q-table">
      <thead><tr><th>strategy</th><th>trainable</th><th>weights</th><th>grads</th>
        <th>optimiser</th><th>total</th><th></th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr class="${r.strategy === 'full' ? 'q-baseline' : ''}">
          <td>${escHtml(r.label)}</td>
          <td>${fmt.compact(r.trainable_params)}</td>
          <td>${fmt.bytes(r.weight_bytes)}</td>
          <td>${fmt.bytes(r.grad_bytes)}</td>
          <td>${fmt.bytes(r.optimizer_bytes)}</td>
          <td><strong>${fmt.bytes(r.total_bytes)}</strong></td>
          <td class="bar-cell"><span class="bar" style="width:${Math.max(2, 100 * (r.total_bytes || 0) / worst)}%"></span></td>
        </tr>`).join('')}
      </tbody>
    </table>
    <div class="q-extra">${escHtml(b.note || '')}</div>`;
}

function renderLoraAdapters(list) {
  const host = $('#l-adapters');
  if (!list || !list.length) {
    host.innerHTML = '<div class="q-empty">None yet. A finished job writes one beside its base checkpoint.</div>';
    return;
  }
  host.innerHTML = `
    <table class="q-table">
      <thead><tr><th>adapter</th><th>rank</th><th>layers</th><th>size</th><th>teaches</th>
        <th>val loss</th></tr></thead>
      <tbody>${list.map((a) => `
        <tr>
          <td>${escHtml(a.rel)}</td>
          <td>${a.r == null ? '–' : `r=${a.r}`}</td>
          <td>${escHtml(a.targets || '–')}</td>
          <td>${fmt.bytes(a.size)}</td>
          <td>${escHtml(a.stage || '–')}</td>
          <td>${a.val_loss == null ? '–' : fmt.num(a.val_loss, 4)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
    <div class="q-extra">Pick one in the Playground's adapter box to hear the difference
      against the same base model.</div>`;
}

function renderLoraStatus(st) {
  lora.status = st;
  renderLoraForm(st);

  const cur = st.current;
  const running = st.running;
  $('#l-stop').hidden = !running;
  $('#l-run').disabled = running || !lora.ckpts.some((c) => !c.error)
                         || !(st.datasets || []).length;

  if (cur) {
    const label = running ? 'running' : cur.state;
    const started = cur.started ? new Date(cur.started * 1000).toLocaleTimeString() : '';
    const what = cur.method === 'full' ? 'full fine-tune'
                                       : `${cur.method} r=${cur.r} on ${cur.targets}`;
    $('#l-state').textContent =
      `${label} — ${what}, ${cur.checkpoint} (${cur.device}), started ${started}`;
    $('#l-cmd').innerHTML = `<code>python -m ${escHtml(cur.cmd || '')}</code>`;
  } else {
    $('#l-state').textContent = 'nothing running';
  }

  const log = $('#l-log');
  if (st.log && st.log.length) {
    const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
    log.textContent = st.log.join('\n');
    if (stick) log.scrollTop = log.scrollHeight;
  }
  renderLoraAdapters(st.adapters || []);
}

async function pollLora() {
  clearTimeout(lora.timer);
  if (state.view !== 'lora' || document.hidden) return;
  try {
    const st = await api('/api/lora?lines=250');
    renderLoraStatus(st);
    lora.timer = setTimeout(pollLora, st.running ? 2000 : 8000);
  } catch (err) {
    $('#l-state').textContent = err.message;
    lora.timer = setTimeout(pollLora, 8000);
  }
}

async function startLora() {
  try {
    $('#l-run').disabled = true;
    const res = await post('/api/lora/start', loraSpec());
    flash(`Fine-tuning: ${res.method} on ${res.checkpoint} (${res.device}). ${res.device_reason || ''}`, 'ok');
    $('#l-log').textContent = 'starting…';
    pollLora();
  } catch (err) {
    flash(err.message, 'error');
    $('#l-run').disabled = false;
  }
}

function wireLoraTab() {
  $('#l-run').addEventListener('click', startLora);
  $('#l-stop').addEventListener('click', async () => {
    try { await post('/api/lora/stop', {}); flash('Stopped the fine-tuning job.', 'ok'); }
    catch (err) { flash(err.message, 'error'); }
    pollLora();
  });
  for (const id of ['#l-method', '#l-rank', '#l-data']) {
    $(id).addEventListener('change', () => { if (lora.status) renderLoraForm(lora.status); });
  }
  for (const id of ['#l-ckpt', '#l-targets']) {
    $(id).addEventListener('change', () => {
      if (lora.status) renderLoraForm(lora.status);
      refreshBudget();
    });
  }
}

function showView(view) {
  state.view = view;
  $('#view-dashboard').hidden = view !== 'dashboard';
  $('#view-code').hidden = view !== 'code';
  $('#view-play').hidden = view !== 'play';
  $('#view-docs').hidden = view !== 'docs';
  $('#view-quant').hidden = view !== 'quant';
  $('#view-lora').hidden = view !== 'lora';
  $('#run-field').hidden = view !== 'dashboard';
  $('#phase').hidden = view !== 'dashboard';
  $('.foot-dashboard').hidden = view !== 'dashboard';
  $('.foot-code').hidden = view !== 'code';
  $('.foot-play').hidden = view !== 'play';
  $('.foot-quant').hidden = view !== 'quant';
  $('.foot-lora').hidden = view !== 'lora';
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
  if (view !== 'play') clearTimeout(play.statusTimer);
  if (view !== 'quant') clearTimeout(quant.timer);
  if (view !== 'lora') clearTimeout(lora.timer);
  if (view === 'code') openCodeTab();
  else if (view === 'play') openPlayTab();
  else if (view === 'docs') openDocsTab();
  else if (view === 'quant') openQuantTab();
  else if (view === 'lora') openLoraTab();
  else { schedule(0); drawCharts(); }   /* charts sized while hidden measure as zero */
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
    const [status, log, runs, sched, gpu, pipeline] = await Promise.all([
      api(`/api/run/${encodeURIComponent(state.run)}`),
      api(`/api/run/${encodeURIComponent(state.run)}/log${q}`),
      api('/api/runs'),
      api('/api/schedule'),
      api(`/api/gpu?window=${encodeURIComponent(state.gpuWindow)}`),
      // never let a pipeline hiccup break the dashboard
      api(`/api/pipeline/${encodeURIComponent(baseOf(state.run))}`).catch(() => null),
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

function schedule(delay) {
  clearTimeout(state.timer);
  if (document.hidden) return;  // a background tab does not need to poll
  // nor a dashboard nobody is looking at
  if (['code', 'docs', 'quant', 'lora'].includes(state.view)) return;
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


/* ================================================================= playground ==========
 * Talking to the model this portal is training.
 *
 * The Code tab streams from an Ollama model; this streams from one of our own checkpoints.
 * Both are server-sent events with the same frame shape, so the reading loop below is
 * deliberately the twin of `explain()` — two extra event kinds and nothing else new:
 *   start : the checkpoint's provenance (step, val loss, tokens seen) and where it ran
 *   test  : the sandbox verdict, for a graded Python task
 */

const play = {
  loaded: false,
  data: null,          /* the /api/infer overview */
  ckpt: null,
  mode: 'complete',
  messages: [],        /* chat turns, client-side; the server is stateless */
  abort: null,
  statusTimer: null,
};

const MODE_TITLES = {
  complete: 'Completion',
  chat: 'Chat',
  code: 'Python',
};

const MODE_PLACEHOLDERS = {
  complete: 'Type a prompt and the model continues it…',
  chat: 'Say something to the model…',
  code: 'def is_palindrome(s):\n    """Return True if s reads the same backwards."""',
};

/* "CUDA" is what torch calls it; "GPU" is what the person reading owns. */
const devName = (d) => (String(d).toLowerCase() === 'cuda' ? 'GPU' : 'CPU');

function playStatus(text, kind = '') {
  const box = $('#play-status');
  box.textContent = text;
  box.className = 'panel-note ' + kind;
}

const ckptLabel = (c) => {
  const step = c.step == null ? '?' : fmt.int(c.step);
  const val = c.best_val == null ? '' : ` · val ${c.best_val.toFixed(4)}`;
  return `${c.rel} — step ${step}${c.max_steps ? '/' + fmt.int(c.max_steps) : ''}${val}`;
};

/** The card under the picker: what this checkpoint *is*, so the output can be judged. */
function renderCkptMeta() {
  const c = play.ckpt;
  const box = $('#play-meta');
  if (!c) { box.innerHTML = ''; return; }
  const rows = [
    ['stage', c.stage_label],
    ['step', c.step == null ? '?' : fmt.int(c.step) + (c.max_steps ? ` of ${fmt.int(c.max_steps)}` : '')],
    ['val loss', c.best_val == null ? '?' : c.best_val.toFixed(4)],
    ['train loss', c.train_loss == null ? '?' : c.train_loss.toFixed(4)],
    ['tokens seen', c.tokens_seen == null ? '?' : fmt.compact(c.tokens_seen)],
    ['parameters', c.params == null ? '?' : fmt.compact(c.params)],
    ['context', c.max_seq_len == null ? '?' : fmt.int(c.max_seq_len) + ' tokens'],
  ];
  box.innerHTML = rows.map(([k, v]) =>
    `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd></div>`).join('')
    + `<p class="hint">${escapeHtml(c.stage_note || '')}</p>`
    + (c.error ? `<p class="err">${escapeHtml(c.error)}</p>` : '');
}

/** Where it will run, and why. Shown before you press Generate, not after. */
function renderPlan(status) {
  const plan = (status && status.plan) || {};
  const warn = $('#play-warn');
  if (plan.reason) {
    warn.innerHTML =
      `<strong>${plan.device === 'cuda' ? 'GPU' : 'CPU'}</strong> — ${escapeHtml(plan.reason)}`;
    warn.hidden = false;
    warn.classList.toggle('warn-slow', !!plan.slow);
  } else {
    warn.hidden = true;
  }
  const res = $('#play-resident');
  if (status && status.loaded) {
    const secs = status.unload_in_s == null ? null : Math.round(status.unload_in_s);
    res.textContent = `${status.loaded.rel} is resident on the ${devName(status.device)}`
      + (status.load_s ? `, loaded in ${status.load_s.toFixed(1)}s` : '')
      + (secs != null ? ` — unloads after ${fmt.dur(secs)} idle.` : '.');
  } else {
    res.textContent = 'Nothing is loaded; the first generation reads the checkpoint off disk '
      + '(a few seconds) and then it stays warm.';
  }
}

/** Which modes this checkpoint can honestly do. A base model cannot chat, and saying so
 *  up front is better than letting someone conclude the model is broken. */
function renderModes() {
  let allowed = (play.ckpt && play.ckpt.modes) || ['complete'];
  const ad = (play.data.adapters || []).find((a) => a.rel === play.adapter);
  if (ad && ['sft', 'dpo', 'code'].includes(ad.stage)) {
    allowed = ['complete', 'chat', 'code'];
  }
  for (const btn of $$('#play-modes .ghost')) {
    const ok = allowed.includes(btn.dataset.mode);
    btn.disabled = !ok;
    btn.title = ok ? '' : 'this checkpoint has not been trained for that yet';
    btn.classList.toggle('on', btn.dataset.mode === play.mode);
  }
  if (!allowed.includes(play.mode)) setMode(allowed[0]);
  const note = $('#play-modenote');
  if (play.mode === 'chat') {
    note.textContent = 'Multi-turn, using the ChatML template the SFT trainer teaches.';
  } else if (play.mode === 'code') {
    const sb = (play.data && play.data.sandbox) || {};
    note.textContent = sb.enabled && sb.available
      ? `Pick a task and the generated function is executed against its asserts `
        + `(${sb.timeout_s}s, ${sb.memory_mb} MB, in a throwaway directory).`
      : `Code is generated and shown but not executed (${sb.note || 'run_tests: false'}).`;
  } else {
    note.textContent = 'Raw continuation — what a base model does, and all it can do.';
  }
}

/** The fixed prompts. Clicking one fills the box; it is also what gets recorded under a
 *  probe id, which is what makes the comparison across steps possible. */
function renderPresets() {
  const box = $('#play-presets');
  const d = play.data;
  if (!d) { box.innerHTML = ''; return; }
  let items;
  if (play.mode === 'code') {
    items = d.tasks.map((t) => ({
      key: t.id, label: t.title, hint: `${t.difficulty} · executed`, task: t.id,
    }));
  } else {
    const pool = play.mode === 'chat' ? d.chat : d.probes;
    items = pool.map((p) => ({ key: p.id, label: p.id, hint: p.expect, probe: p.id,
      prompt: p.prompt }));
  }
  box.innerHTML = `<div class="panel-head"><h3>${play.mode === 'code'
    ? 'Graded tasks' : 'Fixed probes'}</h3></div>`
    + items.map((it) => `<button class="ghost preset" type="button" data-key="${escapeHtml(it.key)}"`
      + ` title="${escapeHtml(it.hint || '')}">${escapeHtml(it.label)}</button>`).join('');
  box._items = items;
}

function setMode(mode) {
  play.mode = mode;
  $('#play-title').textContent = MODE_TITLES[mode] || mode;
  $('#play-prompt').placeholder = MODE_PLACEHOLDERS[mode] || '';
  $('#play-system-field').hidden = mode !== 'chat';
  $('#play-suite').textContent = mode === 'code'
    ? 'Run all tasks' : 'Run all probes';
  renderModes();
  renderPresets();
}

/* ---- the transcript ---- */

const testBadge = (t) => {
  if (!t) return '';
  const cls = t.ok ? 'ok' : (t.status === 'fail' ? 'bad' : 'warn');
  return `<div class="play-test ${cls}"><strong>${escapeHtml(t.status.toUpperCase())}</strong> `
    + `${escapeHtml(t.detail || '')}`
    + (t.program ? `<details><summary>what was executed</summary>`
      + `<pre>${escapeHtml(t.program)}</pre></details>` : '')
    + (t.stderr && !t.ok ? `<details><summary>stderr</summary>`
      + `<pre>${escapeHtml(t.stderr)}</pre></details>` : '')
    + '</div>';
};

/** One exchange. `meta` is the provenance line — the thing that makes an old answer
 *  meaningful a month later. */
function addTurn({ role, text, meta, cls }) {
  const div = document.createElement('div');
  div.className = `turn turn-${role} ${cls || ''}`;
  div.innerHTML = (meta ? `<div class="turn-meta">${meta}</div>` : '')
    + `<div class="turn-body"></div>`;
  div.querySelector('.turn-body').textContent = text || '';
  $('#play-transcript').appendChild(div);
  $('#play-transcript').scrollTop = $('#play-transcript').scrollHeight;
  return div;
}

const provLine = (c, dev) => {
  const step = c.step == null ? '?' : fmt.int(c.step);
  const val = c.best_val == null ? '' : ` · val ${c.best_val.toFixed(4)}`;
  return `${escapeHtml(c.rel)} · step ${step}${val} · ${devName(dev)}`;
};

/* ---- generating ---- */

function sampling() {
  const num = (id, fallback) => {
    const v = $(id).value.trim();
    return v === '' ? fallback : Number(v);
  };
  const d = (play.data && play.data.status && play.data.status.config
    && play.data.status.config.sampling) || {};
  const out = {
    max_new_tokens: num('#k-max', d.max_new_tokens),
    temperature: num('#k-temp', d.temperature),
    top_k: num('#k-topk', d.top_k),
    top_p: num('#k-topp', d.top_p),
    repetition_penalty: num('#k-rep', d.repetition_penalty),
  };
  const seed = $('#k-seed').value.trim();
  if (seed !== '') out.seed = Number(seed);
  return out;
}

/** Stream one generation into the transcript. Returns the final stats, or null. */
async function generate({ prompt, probe, task, quiet }) {
  if (play.abort) return null;
  if (!play.ckpt) { playStatus('no checkpoint selected', 'err'); return null; }

  const controller = new AbortController();
  play.abort = controller;
  $('#play-go').disabled = true;
  $('#play-stop').hidden = false;
  playStatus('loading the model…', 'busy');

  const shown = task
    ? (play.data.tasks.find((t) => t.id === task) || {}).prompt || task
    : prompt;
  if (!quiet) addTurn({ role: 'user', text: shown, cls: play.mode === 'code' ? 'mono' : '' });
  const answer = addTurn({ role: 'model', text: '', cls: play.mode === 'code' ? 'mono' : '' });
  const body = answer.querySelector('.turn-body');
  let text = '';
  let stats = null;
  const started = Date.now();

  try {
    const res = await fetch('/api/infer/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Portal': '1' },
      signal: controller.signal,
      body: JSON.stringify({
        checkpoint: play.ckpt.rel,
        adapter: play.adapter || null,
        mode: play.mode,
        prompt: prompt || '',
        task: task || null,
        probe: probe || null,
        messages: play.mode === 'chat' ? play.messages : [],
        system: play.mode === 'chat' ? $('#play-system').value : null,
        sampling: sampling(),
      }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).error || msg; } catch { /* no body */ }
      throw new Error(msg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let stop = false;
    while (!stop) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.replace(/^data: /, '').trim();
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        if (evt.error) throw new Error(evt.error);
        if (evt.start) {
          answer.querySelector('.turn-meta')
            || answer.insertAdjacentHTML('afterbegin', '<div class="turn-meta"></div>');
          answer.querySelector('.turn-meta').innerHTML =
            provLine(evt.start.checkpoint, evt.start.device);
          playStatus(`generating on the ${devName(evt.start.device)}…`, 'busy');
          if (evt.start.truncated_tokens) {
            playStatus(`generating — ${evt.start.truncated_tokens} prompt tokens dropped `
              + 'from the front to fit the context window', 'busy');
          }
        }
        if (evt.delta) {
          text += evt.delta;
          body.textContent = text;
          $('#play-transcript').scrollTop = $('#play-transcript').scrollHeight;
        }
        if (evt.test) answer.insertAdjacentHTML('beforeend', testBadge(evt.test));
        if (evt.done) { stats = evt.done; stop = true; }
      }
    }

    if (!text.trim()) {
      body.innerHTML = '<em class="hint">the model produced nothing at all — with a very '
        + 'low top-p or a temperature of 0 on an undertrained model this does happen.</em>';
    }
    if (stats) {
      const rate = stats.tok_per_s ? `${stats.tok_per_s.toFixed(1)} tok/s` : '';
      playStatus(`${stats.tokens} tokens · ${rate} · `
        + `${stats.finish === 'stop' ? 'stopped on its own' : 'hit the token limit'}`);
    } else {
      playStatus(`done in ${fmt.dur((Date.now() - started) / 1000)}`);
    }
    return stats;
  } catch (err) {
    if (err.name === 'AbortError') {
      playStatus('stopped');
    } else {
      body.innerHTML = `<span class="err">${escapeHtml(err.message)}</span>`;
      playStatus(err.message, 'err');
    }
    return null;
  } finally {
    play.abort = null;
    $('#play-stop').hidden = true;
    $('#play-go').disabled = false;
    pollPlayStatus(0);
    loadHistory();
  }
}

async function submitPrompt() {
  const box = $('#play-prompt');
  const prompt = box.value;
  if (!prompt.trim()) return;
  if (play.mode === 'chat') {
    const stats = await generate({ prompt });
    play.messages.push({ role: 'user', content: prompt });
    play.messages.push({ role: 'assistant', content: (stats && stats.text) || '' });
    if (play.messages.length > 20) play.messages = play.messages.slice(-20);
  } else {
    await generate({ prompt });
  }
  box.value = '';
}

/** The whole fixed suite, one after another. This is the button that produces a row in
 *  every probe's comparison at this step. */
async function runSuite() {
  const d = play.data;
  if (!d) return;
  const items = play.mode === 'code'
    ? d.tasks.map((t) => ({ task: t.id }))
    : (play.mode === 'chat' ? d.chat : d.probes).map((p) => ({ probe: p.id, prompt: p.prompt }));
  $('#play-suite').disabled = true;
  let passed = 0;
  try {
    for (let i = 0; i < items.length; i += 1) {
      playStatus(`suite ${i + 1} of ${items.length}…`, 'busy');
      const stats = await generate(items[i]);
      if (stats && stats.test && stats.test.ok) passed += 1;
      if (!play.ckpt) break;
    }
    playStatus(play.mode === 'code'
      ? `${passed} of ${items.length} tasks passed`
      : `${items.length} probes recorded — compare them on the right`);
  } finally {
    $('#play-suite').disabled = false;
  }
}

/* ---- history ---- */

function renderHistory(rows, stats) {
  const box = $('#play-hist');
  $('#play-hist-note').textContent = stats
    ? `${fmt.int(stats.count)} kept in ${stats.path}` : '';
  if (!rows.length) {
    box.innerHTML = '<p class="hint">Nothing recorded yet. Every generation is appended '
      + 'with the step and loss of the model that produced it, so a prompt run now and '
      + 'again at step 30,000 can be compared directly.</p>';
    return;
  }
  box.innerHTML = rows.map((r) => {
    const val = r.best_val == null ? '?' : r.best_val.toFixed(4);
    const badge = r.test
      ? `<span class="pill ${r.test.ok ? 'ok' : 'bad'}">${escapeHtml(r.test.status)}</span>` : '';
    return `<article class="hist-row">
      <header>${escapeHtml(r.run || '?')}/${escapeHtml(r.checkpoint || '?')} ·
        step ${r.step == null ? '?' : fmt.int(r.step)} · val ${val} ${badge}</header>
      <div class="hist-when">${escapeHtml(r.iso || '')} · ${escapeHtml(r.mode || '')}
        ${r.probe ? '· probe ' + escapeHtml(r.probe) : ''}
        ${r.task ? '· task ' + escapeHtml(r.task) : ''}</div>
      <p class="hist-prompt">${escapeHtml((r.prompt || '').slice(0, 140))}</p>
      <p class="hist-out">${escapeHtml((r.output || '').slice(0, 300))}</p>
    </article>`;
  }).join('');
}

async function loadHistory() {
  try {
    const probe = $('#play-compare').value;
    if (probe) {
      const cmp = await api(`/api/infer/compare?probe=${encodeURIComponent(probe)}`);
      const box = $('#play-hist');
      $('#play-hist-note').textContent = `${cmp.count} generation(s), oldest step first`;
      box.innerHTML = cmp.rows.map((r) => {
        const val = r.best_val == null ? '?' : r.best_val.toFixed(4);
        return `<article class="hist-row cmp">
          <header>step ${r.step == null ? '?' : fmt.int(r.step)} · val ${val}
            · ${escapeHtml(r.run || '')}</header>
          <p class="hist-out">${escapeHtml((r.output || '').slice(0, 400))}</p>
        </article>`;
      }).join('') || '<p class="hint">no rows for that probe yet</p>';
      return;
    }
    const h = await api('/api/infer/history?limit=30');
    renderHistory(h.rows, h.stats);
    const sel = $('#play-compare');
    const cur = sel.value;
    sel.innerHTML = '<option value="">recent generations</option>'
      + (h.probes_seen || []).map((p) =>
        `<option value="${escapeHtml(p.probe)}">${escapeHtml(p.probe)} — ${p.count} run(s), `
        + `steps ${p.first_step == null ? '?' : fmt.int(p.first_step)}–`
        + `${p.last_step == null ? '?' : fmt.int(p.last_step)}</option>`).join('');
    sel.value = cur;
  } catch (err) {
    $('#play-hist-note').textContent = err.message;
  }
}

/* ---- status polling ---- */

function pollPlayStatus(delay = 4000) {
  clearTimeout(play.statusTimer);
  play.statusTimer = setTimeout(async () => {
    if (state.view !== 'play' || document.hidden) return;
    try {
      renderPlan(await api('/api/infer/status'));
    } catch { /* the dashboard's own banner covers a dead portal */ }
    pollPlayStatus();
  }, delay);
}

/* ---- setup ---- */

function selectCkpt(rel) {
  play.ckpt = (play.data.checkpoints || []).find((c) => c.rel === rel) || null;
  if (play.ckpt) localStorage.setItem('aksharallm-play-ckpt', rel);
  renderAdapters();
  renderCkptMeta();
  renderModes();
  renderPresets();
}

/* Only adapters trained against *this* checkpoint's architecture are offered. Applying an
 * adapter to the wrong base does not error, it silently degrades the model — so the filter
 * is here as well as in the engine's strict check. */
function renderAdapters() {
  const all = play.data.adapters || [];
  /* Architecture must match: an adapter is a delta on specific shapes, and on a different
   * model it is either a shape error or — worse — a silent degradation. `arch` is built by
   * the same helper on both sides, so the comparison is exact. Adapters whose base did not
   * record an arch are offered anyway and left to the engine's strict check. */
  const arch = (play.ckpt && play.ckpt.arch) || null;
  const usable = all.filter((a) => !a.error && (!a.arch || !arch || a.arch === arch));
  const hidden = all.length - usable.length;
  const sel = $('#play-adapter');
  sel.innerHTML = '<option value="">none — the base model</option>'
    + usable.map((a) => `<option value="${escapeHtml(a.rel)}">`
        + `${escapeHtml(a.rel)} — r=${a.r} ${escapeHtml(a.targets || '')} (${escapeHtml(a.stage || '')})`
        + '</option>').join('');
  $('#play-adapter-field').hidden = !usable.length;
  if (!usable.some((a) => a.rel === play.adapter)) play.adapter = '';
  sel.value = play.adapter || '';
  if (hidden) {
    sel.insertAdjacentHTML('beforeend',
      `<option value="" disabled>${hidden} adapter(s) hidden — trained on a different architecture</option>`);
  }
}

function selectAdapter(rel) {
  play.adapter = rel || '';
  /* An SFT adapter makes chat legitimate on a base checkpoint — that is the whole point of
   * having them — so the mode buttons have to be recomputed when one is attached. */
  renderModes();
  renderCkptMeta();
}

async function openPlayTab() {
  pollPlayStatus(0);
  if (play.loaded) { loadHistory(); return; }
  play.loaded = true;
  try {
    const d = await api('/api/infer');
    play.data = d;
    $('#play-count').textContent = d.checkpoints.length
      ? `${d.checkpoints.length} checkpoint(s)` : 'none yet';

    const sel = $('#play-ckpt');
    sel.innerHTML = d.checkpoints.map((c) =>
      `<option value="${escapeHtml(c.rel)}"${c.error ? ' disabled' : ''}>`
      + `${escapeHtml(ckptLabel(c))}</option>`).join('');
    if (!d.checkpoints.length) {
      $('#play-meta').innerHTML = '<p class="hint">No checkpoints under <code>checkpoints/'
        + '</code> yet. Start a run from the Dashboard; the first one lands after '
        + '<code>ckpt_every</code> steps and you can talk to it while the run continues.</p>';
      $('#play-go').disabled = true;
      $('#play-suite').disabled = true;
      return;
    }
    const want = localStorage.getItem('aksharallm-play-ckpt');
    const pick = d.checkpoints.some((c) => c.rel === want) ? want : d.default;
    sel.value = pick;
    selectCkpt(pick);

    const s = (d.status && d.status.config && d.status.config.sampling) || {};
    $('#k-max').value = s.max_new_tokens;
    $('#k-temp').value = s.temperature;
    $('#k-topk').value = s.top_k;
    $('#k-topp').value = s.top_p;
    $('#k-rep').value = s.repetition_penalty;
    $('#play-system').value = (d.status && d.status.config && d.status.config.system) || '';

    setMode(play.mode);
    renderPlan(d.status);
    loadHistory();
  } catch (err) {
    play.loaded = false;
    $('#play-count').textContent = err.message;
  }
}

function wirePlay() {
  $('#play-ckpt').addEventListener('change', (e) => selectCkpt(e.target.value));
  $('#play-adapter').addEventListener('change', (e) => selectAdapter(e.target.value));

  for (const btn of $$('#play-modes .ghost')) {
    btn.addEventListener('click', () => { if (!btn.disabled) setMode(btn.dataset.mode); });
  }

  $('#play-presets').addEventListener('click', (e) => {
    const btn = e.target.closest('.preset');
    if (!btn) return;
    const items = $('#play-presets')._items || [];
    const it = items.find((x) => x.key === btn.dataset.key);
    if (!it) return;
    if (it.task) { generate({ task: it.task }); return; }
    $('#play-prompt').value = it.prompt;
    generate({ prompt: it.prompt, probe: it.probe });
  });

  $('#play-form').addEventListener('submit', (e) => { e.preventDefault(); submitPrompt(); });

  /* Enter sends, Shift-Enter makes a newline — a code prompt is multi-line by nature. */
  $('#play-prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitPrompt(); }
  });

  $('#play-stop').addEventListener('click', () => { if (play.abort) play.abort.abort(); });

  $('#play-clear').addEventListener('click', () => {
    $('#play-transcript').textContent = '';
    play.messages = [];
    playStatus('');
  });

  $('#play-suite').addEventListener('click', runSuite);
  $('#play-compare').addEventListener('change', loadHistory);

  $('#play-unload').addEventListener('click', async () => {
    try {
      const res = await post('/api/infer/unload', {});
      playStatus(res.note);
      pollPlayStatus(0);
    } catch (err) { playStatus(err.message, 'err'); }
  });
}


/* ---------------------------------------------------------------- wiring -------------- */

function selectRun(run) {
  state.run = run;
  state.logFile = null;
  state.status = null;
  state.zoom = {};        /* another run's step range means nothing here */
  flash('');
  $('#log-select').dataset.run = '';
  schedule(0);
}

function wire() {
  wireGpu();
  wireSchedule();
  wireCode();
  wirePlay();
  wireQuantTab();
  wireLoraTab();
  for (const tab of $$('.tab')) {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  }
  $('#run-select').addEventListener('change', (e) => selectRun(e.target.value));

  $('#btn-start').addEventListener('click', () => {
    const after = $('#stop-after').value.trim();
    act(() => post(`/api/run/${encodeURIComponent(state.run)}/start`, {
      stop_after: after ? Number(after) : null,
      skip_smoke: $('#skip-smoke').checked,
    }), 'Launching.');
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

const VIEWS = ['dashboard', 'play', 'code', 'quant', 'lora', 'docs'];

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

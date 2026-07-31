/* A small SVG line-chart engine: axes, hairline grid, crosshair, tooltip, drag-to-zoom, and
 * the table twin that makes every plotted value reachable without hovering anything.
 *
 * Hand-written for the same reason the transformer is: a plotting library would be one more
 * thing this project doesn't explain. Colour is never the only encoding — every series is in
 * the legend, every value is reachable from the table view. */

import { $, fmt } from './core.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
export const el = (name, attrs = {}, parent = null) => {
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
export function lineChart(host, spec) {
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
export function chartTable(host, spec) {
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

export function table(head, rows, opts = {}) {
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

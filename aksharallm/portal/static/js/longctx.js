/* The Context tab: how far the model reads, what would extend it, and whether it worked.
 *
 * Two halves with deliberately different costs, and the UI is shaped to say so:
 *
 *   - "What would extending it do?" is a GET against pure arithmetic. Nothing is loaded,
 *     nothing is written, so it re-runs on every change of the dropdowns. A reader should be
 *     able to understand the whole trade-off before spending anything.
 *   - "Measure it" starts a detached job, one at a time, exactly like the Quantize tab. The
 *     device is chosen by the server (CPU while a run is training) and the reason is shown
 *     rather than left implicit.
 *
 * The charts are hand-drawn SVG like everywhere else in this portal, and every one of them
 * has a table twin — the position curve is the whole result of this tab and it has to be
 * readable without colour.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';

const lc = { loaded: false, ckpt: null, timer: null, results: [], open: null };

function status(text, kind = '') {
  const el = $('#lc-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

/* ---- the free half ---------------------------------------------------------------------- */

function renderWindow(cur) {
  if (!cur) { $('#lc-window').innerHTML = ''; return; }
  const ratio = cur.trained_window ? cur.addressable / cur.trained_window : 1;
  const bar = Math.min(100, 100 / Math.max(ratio, 1));
  $('#lc-window').innerHTML = `
    <div class="lc-bar" role="img" aria-label="trained ${cur.trained_window} of ${cur.addressable} addressable tokens">
      <div class="lc-bar-trained" style="width:${bar}%"></div>
    </div>
    <dl class="lc-facts">
      <div><dt>trained on</dt><dd>${fmt.int(cur.trained_window)} tokens</dd></div>
      <div><dt>can address</dt><dd>${fmt.int(cur.addressable)} tokens</dd></div>
      <div><dt>scaling</dt><dd>${escHtml(cur.scaling && cur.scaling.type || 'none')}${
        cur.scaling && cur.scaling.factor > 1 ? ` ×${cur.scaling.factor}` : ''}</dd></div>
      <div><dt>sliding window</dt><dd>${cur.window ? `${fmt.int(cur.window)} tokens${
        cur.sinks ? ` + ${cur.sinks} sinks` : ''}` : 'none — full attention'}</dd></div>
    </dl>
    <p class="hint">${cur.extended
      ? 'This checkpoint has already been extended. Anything below compares against the window '
        + 'its <em>weights</em> were trained on, not the one it can address.'
      : 'Everything past the trained window is where the cliff lives. Try a method below.'}</p>`;
}

async function loadPlan() {
  if (!lc.ckpt) return;
  const method = $('#lc-method').value;
  const factor = $('#lc-factor').value;
  try {
    const p = await api(`/api/longctx/plan?checkpoint=${encodeURIComponent(lc.ckpt)}`
      + `&method=${method}&factor=${factor}`);
    const rows = p.changes.map((c) => `<li>${escHtml(c)}</li>`).join('');
    $('#lc-plan').innerHTML = `
      <ul class="lc-changes">${rows}</ul>
      <p class="lc-weights"><strong>No weights change.</strong> RoPE has no parameters — the
        rotation angles are computed from the position, so this is a config edit. The file
        <code>extend</code> writes has byte-for-byte the same tensors in it.</p>`;
    $('#lc-advice').textContent = p.advice;
  } catch (err) {
    $('#lc-plan').innerHTML = `<p class="err">${escHtml(err.message)}</p>`;
    $('#lc-advice').textContent = '';
  }
}

/* ---- the measured half ------------------------------------------------------------------- */

async function start(kind) {
  const body = { kind, checkpoint: lc.ckpt };
  /* The kernel benchmark measures the card, not a checkpoint — no model, no options, and
   * no CPU fallback, which the server enforces and explains. */
  if (kind === 'flash') {
    try {
      const res = await post('/api/longctx/start', body);
      flash(`Started — ${res.why_device}. First run compiles the kernels; give it a minute.`, 'ok');
      poll(1500);
    } catch (err) { flash(err.message, 'error'); }
    return;
  }
  if (kind === 'needle') {
    body.trials = Number($('#lc-trials').value) || 3;
    body.lengths = [512, 1024, 2048];
  } else {
    body.length = Number($('#lc-len').value);
    body.windows = Number($('#lc-windows').value) || 8;
    body.bucket = Math.max(64, Math.round(Number($('#lc-len').value) / 8));
    if (kind === 'sweep') {
      body.factor = Number($('#lc-factor').value);
      body.methods = ['none', 'linear', 'ntk', 'yarn'];
    } else {
      const m = $('#lc-method').value;
      if (m !== 'none') { body.factor = Number($('#lc-factor').value); }
    }
  }
  try {
    const res = await post('/api/longctx/start', body);
    flash(`Started — ${res.why_device}.`, 'ok');
    poll(600);
  } catch (err) { flash(err.message, 'error'); }
}

async function poll(delay = 2000) {
  clearTimeout(lc.timer);
  try {
    const res = await api(`/api/longctx?checkpoint=${encodeURIComponent(lc.ckpt || '')}`);
    const job = res.job || {};
    $('#lc-log').hidden = !job.log;
    if (job.log) $('#lc-log').textContent = job.log;
    $('#lc-stop').hidden = !job.running;
    $('#lc-job-note').textContent = job.running
      ? `running (pid ${job.pid})${res.training ? ` — on the CPU, ${res.training} has the card` : ''}`
      : (res.training ? `${res.training} is training, so a measurement would run on the CPU` : '');
    renderResults(res.results || []);
    if (job.running) lc.timer = setTimeout(() => poll(), delay);
  } catch (err) { /* the tab may have been left; nothing to say */ }
}

/* ---- results ------------------------------------------------------------------------------ */

function renderResults(rows) {
  lc.results = rows;
  if (!rows.length) {
    $('#lc-results').innerHTML = '<p class="hint">Nothing measured yet.</p>';
    return;
  }
  const body = rows.map((r) => {
    let summary = '';
    if (r.kind === 'sweep') {
      const best = r.methods.slice().sort((a, b) => a.loss - b.loss)[0];
      summary = `${r.methods.length} methods at ${fmt.int(r.seq_len)} — best `
        + `<strong>${escHtml(best.method)}</strong> ${fmt.num(best.loss, 3)}`;
    } else if (r.kind === 'needle') {
      summary = `${fmt.num(r.accuracy * 100, 1)}% retrieved (chance ${fmt.num(r.chance * 100, 0)}%)`;
    } else {
      summary = `loss ${fmt.num(r.loss, 3)}`;
    }
    return `<tr data-name="${escHtml(r.name)}"><th>${escHtml(r.kind)}</th>`
      + `<td>${escHtml(r.checkpoint || '')}</td><td>${summary}</td>`
      + `<td class="lc-when">${fmt.ago(r.when)}</td></tr>`;
  }).join('');
  $('#lc-results').innerHTML =
    `<table class="lc-table"><thead><tr><th>what</th><th>model</th><th>result</th><th></th>`
    + `</tr></thead><tbody>${body}</tbody></table>`;
  for (const tr of $('#lc-results').querySelectorAll('tr[data-name]')) {
    tr.addEventListener('click', () => openResult(tr.dataset.name));
  }
}

async function openResult(name) {
  try {
    const blob = await api(`/api/longctx/result?name=${encodeURIComponent(name)}`);
    lc.open = blob;
    if (blob.rows) renderSweep(blob);
    else if (blob.grid) renderNeedle(blob);
    else renderCurve(blob, blob.seq_len);
  } catch (err) { flash(err.message, 'error'); }
}

/* The position curve. One line per method, a dashed rule where the trained window ends —
 * which is the only annotation that matters, because the whole point is what happens after
 * it. Losses, not perplexities: a collapsed bucket's perplexity is in the millions and would
 * flatten every other line onto the axis. */
/* The palette is deliberately two categorical hues (see css/base.css), and a sweep has four
 * lines. Rather than invent colours the design system does not have, `none` takes the
 * *status* colour it has earned — it is the failure case, not a peer — and the three working
 * methods are separated by the two series hues plus a dash. Distinguishable without colour,
 * which is the point of the table underneath either way. */
const STROKES = {
  none:    { stroke: 'var(--critical)', dash: '' },
  linear:  { stroke: 'var(--series-2)', dash: '' },
  ntk:     { stroke: 'var(--series-1)', dash: '' },
  yarn:    { stroke: 'var(--series-1)', dash: '6 3' },
  dynamic: { stroke: 'var(--series-2)', dash: '6 3' },
  loss:    { stroke: 'var(--series-1)', dash: '' },
};

function curveSvg(series, trained, seqLen) {
  const W = 720, H = 260;
  const pad = { l: 46, r: 12, t: 16, b: 34 };
  const all = series.flatMap((s) => s.points.map((p) => p.loss));
  const lo = Math.min(...all) * 0.95, hi = Math.max(...all) * 1.05;
  const x = (pos) => pad.l + (pos / seqLen) * (W - pad.l - pad.r);
  const y = (v) => H - pad.b - ((v - lo) / (hi - lo || 1)) * (H - pad.b - pad.t);
  const pen = (label) => STROKES[label] || STROKES.loss;

  const lines = series.map((s) => {
    const d = s.points.map((p, j) =>
      `${j ? 'L' : 'M'}${x(p.start).toFixed(1)},${y(p.loss).toFixed(1)}`).join('');
    const { stroke, dash } = pen(s.label);
    return `<path class="series-line" d="${d}" stroke="${stroke}"`
      + `${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
  }).join('');

  const legend = series.map((s) => {
    const { stroke, dash } = pen(s.label);
    return `<span class="legend-item"><span class="legend-swatch" style="background:${stroke}`
      + `${dash ? ';opacity:.65' : ''}"></span>${escHtml(s.label)}</span>`;
  }).join('');

  /* The only annotation worth drawing: everything interesting happens to the right of it. */
  const edge = trained && trained < seqLen
    ? `<line class="rule-line" x1="${x(trained)}" y1="${pad.t}" x2="${x(trained)}" `
      + `y2="${H - pad.b}"/><text class="rule-label" x="${x(trained) + 5}" `
      + `y="${pad.t + 10}">trained window ends</text>`
    : '';
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) =>
    `<text class="axis-label" x="${x(f * seqLen)}" y="${H - 12}" text-anchor="middle">`
    + `${fmt.int(f * seqLen)}</text>`).join('');
  const yticks = [lo, (lo + hi) / 2, hi].map((v) =>
    `<line class="grid-line" x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}"/>`
    + `<text class="axis-label" x="${pad.l - 8}" y="${y(v) + 4}" text-anchor="end">`
    + `${fmt.num(v, 1)}</text>`).join('');

  return `<div class="legend">${legend}</div>`
    + `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" `
    + `aria-label="cross-entropy loss by token position; the table below has the numbers">`
    + `<g>${yticks}${ticks}${edge}${lines}</g></svg></div>`;
}

function renderCurve(blob, seqLen) {
  const series = [{ label: 'loss', points: blob.buckets }];
  $('#lc-detail').innerHTML = curveSvg(series, blob.trained, seqLen || blob.seq_len)
    + table([{ label: 'loss', points: blob.buckets }]);
}

function renderSweep(blob) {
  const series = blob.rows.map((r) => ({ label: r.method, points: r.curve.buckets }));
  const rows = blob.rows.map((r) => {
    const bs = r.curve.buckets;
    const ins = bs.filter((b) => b.end < blob.trained).map((b) => b.loss);
    const out = bs.filter((b) => b.start >= blob.trained).map((b) => b.loss);
    const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
    return `<tr><th>${escHtml(r.method)}</th><td>${fmt.num(r.curve.loss, 3)}</td>`
      + `<td>${fmt.num(mean(ins), 3)}</td><td>${fmt.num(mean(out), 3)}</td>`
      + `<td>${r.cliff ? fmt.int(r.cliff.position) : '—'}</td></tr>`;
  }).join('');
  $('#lc-detail').innerHTML = curveSvg(series, blob.trained, blob.seq_len)
    + `<table class="lc-table"><thead><tr><th>method</th><th>overall</th>`
    + `<th>in-window</th><th>past it</th><th>cliff at</th></tr></thead>`
    + `<tbody>${rows}</tbody></table>`
    + `<p class="hint">Read the <em>in-window</em> column as carefully as the others: a method
       that buys range by damaging short contexts has not helped. A cliff of “—” is the
       result you want.</p>`;
}

function table(series) {
  const rows = series[0].points.map((p, i) =>
    `<tr><th>${fmt.int(p.start)}–${fmt.int(p.end)}</th>`
    + series.map((s) => `<td>${fmt.num(s.points[i].loss, 3)}</td>`).join('') + '</tr>').join('');
  return `<table class="lc-table"><thead><tr><th>positions</th>`
    + series.map((s) => `<th>${escHtml(s.label)}</th>`).join('')
    + `</tr></thead><tbody>${rows}</tbody></table>`;
}

/* The needle grid. Cells are shaded against the chance line rather than against zero — a
 * grid of 25%s from a four-way choice is a grid of nothing, and colouring it as "a quarter
 * of the way there" would be the single most misleading thing this tab could draw. */
function renderNeedle(blob) {
  const head = blob.lengths.map((n) => `<th>${fmt.int(n)}</th>`).join('');
  const rows = blob.grid.map((row) => {
    const cells = row.map((c) => {
      if (c.accuracy == null) return '<td>—</td>';
      const over = (c.accuracy - blob.chance) / (1 - blob.chance);
      const shade = Math.max(0, Math.min(1, over));
      return `<td class="lc-cell" style="--w:${shade.toFixed(2)}">`
        + `${fmt.num(c.accuracy * 100, 0)}%</td>`;
    }).join('');
    return `<tr><th>${fmt.num(row[0].depth * 100, 0)}%</th>${cells}</tr>`;
  }).join('');
  const acc = blob.accuracy, se = blob.stderr;
  const chancey = se && acc - 2 * se <= blob.chance;
  $('#lc-detail').innerHTML =
    `<table class="lc-table lc-grid"><thead><tr><th>depth</th>${head}</tr></thead>`
    + `<tbody>${rows}</tbody></table>`
    + `<p class="${chancey ? 'err' : 'hint'}">Overall ${fmt.num(acc * 100, 1)}%`
    + (se ? ` ± ${fmt.num(se * 100, 1)}%` : '')
    + ` against a chance line of ${fmt.num(blob.chance * 100, 0)}%. `
    + (chancey
      ? 'That is <strong>not distinguishable from guessing</strong> — which is a real result, '
        + 'not a broken test. Scaling makes distant positions <em>legible</em>; being able to '
        + '<em>use</em> them is a capability, and capabilities come from training.'
      : 'Shading is against the chance line, not against zero.')
    + '</p>';
}

/* ---- wiring ------------------------------------------------------------------------------- */

async function openLongCtxTab() {
  if (!lc.loaded) {
    const res = await api('/api/longctx');
    const sel = $('#lc-ckpt');
    sel.innerHTML = (res.checkpoints || [])
      .map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)}</option>`).join('');
    lc.ckpt = sel.value;
    $('#lc-len').innerHTML = (res.lengths || [])
      .map((n) => `<option value="${n}"${n === 2048 ? ' selected' : ''}>${n}</option>`).join('');

    sel.addEventListener('change', async () => {
      lc.ckpt = sel.value;
      await refreshCurrent();
      await loadPlan();
    });
    for (const id of ['#lc-method', '#lc-factor']) {
      $(id).addEventListener('change', loadPlan);
    }
    $('#lc-run-curve').addEventListener('click', () => start('curve'));
    $('#lc-run-sweep').addEventListener('click', () => start('sweep'));
    $('#lc-run-needle').addEventListener('click', () => start('needle'));
    $('#lc-run-flash').addEventListener('click', () => start('flash'));
    $('#lc-stop').addEventListener('click', async () => {
      await post('/api/longctx/stop', {});
      flash('Stop requested.', 'ok');
      poll(600);
    });
    lc.loaded = true;
  }
  await refreshCurrent();
  await loadPlan();
  poll();
}

async function refreshCurrent() {
  if (!lc.ckpt) { status('no checkpoints found', 'err'); return; }
  try {
    const res = await api(`/api/longctx?checkpoint=${encodeURIComponent(lc.ckpt)}`);
    renderWindow(res.current);
    renderResults(res.results || []);
    status(res.training ? `${res.training} is training — measurements will run on the CPU` : '');
  } catch (err) { status(err.message, 'err'); }
}

registerTab('longctx', { open: openLongCtxTab, leave: () => clearTimeout(lc.timer) });

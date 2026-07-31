/* The Eval tab: is the model actually any good?
 *
 * Every button POSTs to /api/eval/*, which shells out to `python -m aksharallm.eval` — the
 * panel never evaluates anything itself, so a job started here and one typed into a
 * terminal write the same file into logs/eval/ and appear in the same table.
 *
 * The tab leads with the **trend**, not with the Run button, for the same reason the
 * Finetune tab leads with the memory budget: one benchmark score is close to meaningless.
 * 25.4% on MMLU only says something next to the chance line at 25%, the standard error,
 * and what the same suite scored ten thousand steps ago.
 */
import { $, $$, api, escHtml, flash, fmt, post } from './core.js';
import { state } from './state.js';
import { lineChart, table } from './charts.js';
import { registerTab } from './router.js';

const ev = {
  timer: null, ckpts: [], adapters: [], status: null, loaded: false,
  suites: [], selected: new Set(), trend: null, trendSuite: null,
};

/* Perplexity goes down; everything else goes up. Getting this backwards would paint an
 * improving model red, which is the one thing a trend chart must never do. */
const LOWER_IS_BETTER = new Set(['perplexity']);

async function openEvalTab() {
  if (!ev.loaded) {
    ev.loaded = true;
    try {
      const data = await api('/api/eval/checkpoints');
      ev.ckpts = data.checkpoints || [];
      ev.adapters = data.adapters || [];
      renderCheckpoints();
    } catch (err) {
      $('#ev-ckpt-note').textContent = `could not list checkpoints — ${err.message}`;
    }
  }
  pollEval();
}

function renderCheckpoints() {
  const sel = $('#ev-ckpt');
  const usable = ev.ckpts.filter((c) => !c.error);
  sel.innerHTML = usable.map((c) => {
    const bits = [c.step == null ? null : `step ${fmt.int(c.step)}`,
      c.params == null ? null : `${fmt.compact(c.params)} params`,
      c.stage].filter(Boolean).join(' · ');
    return `<option value="${escHtml(c.id)}">${escHtml(c.rel)} — ${escHtml(bits)}</option>`;
  }).join('');
  $('#ev-ckpt-note').textContent = usable.length
    ? 'A quantized checkpoint can be evaluated too — that is how you find out what int4 cost on a benchmark rather than on perplexity.'
    : 'No checkpoints found. Train something first.';
  $('#ev-run').disabled = !usable.length;

  const ad = $('#ev-adapter');
  ad.innerHTML = '<option value="">none — the base checkpoint alone</option>'
    + ev.adapters.filter((a) => !a.error).map((a) =>
      `<option value="${escHtml(a.rel)}">${escHtml(a.rel)}</option>`).join('');
}

function renderSuites(st) {
  const host = $('#ev-suite-list');
  if (host.dataset.built) return;
  host.dataset.built = '1';
  ev.suites = st.suites || [];
  ev.selected = new Set((st.groups && st.groups.default) || []);
  host.innerHTML = ev.suites.map((s) => `
    <label class="ev-suite">
      <input type="checkbox" value="${escHtml(s.name)}"${ev.selected.has(s.name) ? ' checked' : ''}>
      <span class="ev-suite-name">${escHtml(s.name)}<em>${escHtml(s.kind)}</em></span>
      <span class="ev-suite-blurb">${escHtml(s.blurb)}</span>
      <span class="ev-suite-expect">${escHtml(s.expect)}</span>
    </label>`).join('');
  host.addEventListener('change', (e) => {
    if (e.target.type !== 'checkbox') return;
    if (e.target.checked) ev.selected.add(e.target.value);
    else ev.selected.delete(e.target.value);
    renderRunNote();
  });

  const trendSel = $('#ev-trend-suite');
  trendSel.innerHTML = ev.suites.map((s) =>
    `<option value="${escHtml(s.name)}">${escHtml(s.name)}</option>`).join('');
  /* Open on a suite that has actually been measured. Defaulting to the first name in the
   * registry lands on MMLU, which on a fresh checkout has no results — so the tab's
   * headline chart would be empty on the one visit where a first impression is formed. */
  const measured = ev.suites.find((s) => (st.results || []).some((r) => r.scores[s.name]));
  ev.trendSuite = ev.trendSuite || (measured || ev.suites[0] || {}).name;
  trendSel.value = ev.trendSuite;
  loadTrend();
}

function applyGroup(name) {
  const st = ev.status || {};
  const want = (st.groups && st.groups[name]) || [];
  ev.selected = new Set(want);
  for (const box of $$('#ev-suite-list input[type=checkbox]')) box.checked = ev.selected.has(box.value);
  renderRunNote();
}

function renderRunNote() {
  const chosen = [...ev.selected];
  const st = ev.status || {};
  const missing = new Set((st.datasets || []).filter((d) => !d.cached).map((d) => d.name));
  const blocked = chosen.filter((name) => {
    const s = (ev.suites || []).find((x) => x.name === name);
    return s && ((s.source && missing.has(s.source)) || (s.shot_source && missing.has(s.shot_source)));
  });
  $('#ev-run').disabled = !chosen.length || !!blocked.length || (st.running === true);
  $('#ev-cmd').textContent = blocked.length
    ? `${blocked.join(', ')} needs data that is not downloaded yet.`
    : (chosen.length ? `python -m aksharallm.eval run <ckpt> --suite ${chosen.join(',')}` : 'pick at least one suite');
}

function renderDatasets(st) {
  const rows = st.datasets || [];
  const missing = rows.filter((d) => !d.cached);
  const bytes = rows.reduce((a, d) => a + (d.bytes || 0), 0);
  $('#ev-data-note').textContent = missing.length
    ? `${missing.length} of ${rows.length} not downloaded`
    : `all ${rows.length} cached · ${fmt.bytes(bytes)}`;
  $('#ev-datasets').innerHTML = rows.map((d) => `
    <div class="ev-ds${d.cached ? ' on' : ''}">
      <span class="ev-ds-name">${escHtml(d.name)}</span>
      <span class="ev-ds-meta">${d.cached ? `${fmt.int(d.rows)} rows · ${fmt.bytes(d.bytes)}` : 'not downloaded'}</span>
    </div>`).join('');
  $('#ev-fetch').disabled = !missing.length || st.running;
  $('#ev-fetch').textContent = missing.length
    ? `Download ${missing.length} missing` : 'All data downloaded';
}

function renderStatus(st) {
  ev.status = st;
  renderSuites(st);
  renderDatasets(st);
  renderRunNote();

  const dev = st.device || {};
  $('#ev-device-note').textContent = dev.reason || '';
  $('#ev-device-note').classList.toggle('warn', (dev.training || []).length > 0);

  const cur = st.current;
  $('#ev-stop').hidden = !st.running;
  if (cur) {
    const label = st.running ? 'running' : cur.state;
    const started = cur.started ? new Date(cur.started * 1000).toLocaleTimeString() : '';
    $('#ev-state').textContent = cur.kind === 'fetch'
      ? `${label} — downloading ${(cur.datasets || []).length} datasets, started ${started}`
      : `${label} — ${(cur.suites || []).join(', ')} on ${cur.checkpoint} (${cur.device}), started ${started}`;
  } else {
    $('#ev-state').textContent = 'nothing running';
  }

  const prog = st.progress;
  $('#ev-progress').hidden = !(st.running && prog);
  if (st.running && prog) {
    $('#ev-bar-fill').style.width = `${prog.pct}%`;
    $('#ev-progress-label').textContent = `${prog.label} — ${fmt.int(prog.done)} / ${fmt.int(prog.total)}`;
  }

  const log = $('#ev-log');
  if (st.log && st.log.length) {
    const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
    log.textContent = st.log.join('\n');
    if (stick) log.scrollTop = log.scrollHeight;
  }
  renderResults(st.results || []);
}

/** The score cell: the number, and enough context that it cannot be misread on its own. */
function scoreCell(name, entry) {
  if (!entry || entry.score == null) return '<td class="ev-dim">–</td>';
  if (entry.kind === 'ppl') return `<td>${fmt.num(entry.score, 3)}</td>`;
  const pct = entry.score * 100;
  const base = entry.baseline == null ? null : entry.baseline * 100;
  const err = entry.stderr == null ? null : entry.stderr * 100;
  /* Above chance by more than one standard error is the only claim worth colouring. Two
   * points over the baseline with a three-point error bar is not a result. */
  const cls = base == null ? '' : (pct - (err || 0) > base ? ' ev-good'
    : pct + (err || 0) < base ? ' ev-bad' : ' ev-chance');
  return `<td class="${cls.trim()}" title="${err == null ? '' : `± ${err.toFixed(1)}`}${base == null ? '' : `, chance ${base.toFixed(0)}%`}">${pct.toFixed(1)}%</td>`;
}

function renderResults(rows) {
  const host = $('#ev-results');
  if (!rows.length) {
    host.innerHTML = '<p class="ev-empty">No evaluations yet. Pick a checkpoint and a few '
      + 'suites — <strong>fast</strong> takes a couple of minutes and gives you the first '
      + 'row of a table that only becomes useful once it has two.</p>';
    return;
  }
  const names = (ev.suites || []).map((s) => s.name).filter((n) => rows.some((r) => r.scores[n]));
  host.innerHTML = `
    <table class="ev-table">
      <thead><tr><th>when</th><th>step</th><th>val</th>
        ${names.map((n) => `<th>${escHtml(n)}</th>`).join('')}
        <th>checkpoint</th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr>
          <td class="ev-dim">${escHtml(new Date(r.when * 1000).toLocaleString())}</td>
          <td>${r.step == null ? '–' : fmt.int(r.step)}</td>
          <td>${r.best_val == null ? '–' : fmt.num(r.best_val, 4)}</td>
          ${names.map((n) => scoreCell(n, r.scores[n])).join('')}
          <td class="ev-ckpt-cell"><code>${escHtml(r.checkpoint)}</code>${r.adapter ? ` + <code>${escHtml(r.adapter)}</code>` : ''}</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

/* ---------------------------------------------------------------- the trend ------------ */

async function loadTrend() {
  if (!ev.trendSuite) return;
  try {
    ev.trend = await api(`/api/eval/compare?suite=${encodeURIComponent(ev.trendSuite)}`);
  } catch (err) {
    $('#ev-trend-note').textContent = err.message;
    return;
  }
  drawTrend();
}

export function drawTrend() {
  const data = ev.trend;
  const host = $('#ev-trend-chart');
  if (!data) return;
  const pts = (data.points || []).filter((p) => p.step != null);
  const isPpl = data.kind === 'ppl';
  const scale = isPpl ? 1 : 100;

  $('#ev-trend-note').textContent = data.expect || '';

  /* One point is not a trend, and drawing it as one is worse than drawing nothing: a
   * single-x chart gives every tick the same value and reads like a flat line, which is
   * exactly the wrong conclusion. Say what is missing instead. */
  if (pts.length < 2) {
    host.textContent = '';
    const note = document.createElement('div');
    note.className = 'chart-empty';
    note.textContent = pts.length
      ? `One measurement of ${data.suite}, at step ${fmt.int(pts[0].step)}`
        + `${isPpl ? ` (${fmt.num(pts[0].score, 3)})` : ` (${fmt.pct(pts[0].score, 1)})`}. `
        + 'A trend needs two — evaluate another checkpoint.'
      : `No ${data.suite} results yet — run it on two checkpoints and this fills in.`;
    host.appendChild(note);
    renderTrendTable(data, pts, isPpl, scale);
    return;
  }

  lineChart(host, {
    label: `${data.suite} by training step`,
    height: 200,
    yFmt: (v) => (isPpl ? v.toFixed(2) : `${v.toFixed(0)}%`),
    /* The chance line is the whole point of this chart. Without it a flat 25% reads as a
     * model that has learnt something and stopped, rather than one that is still guessing. */
    rules: data.baseline == null ? [] : [{ y: data.baseline * scale, label: 'chance' }],
    empty: `No ${data.suite} results yet — run it on two checkpoints and this fills in.`,
    series: [{
      name: data.suite, color: '--series-1', dots: true, label: true,
      x: pts.map((p) => p.step), y: pts.map((p) => p.score * scale),
      fmt: (v) => (isPpl ? v.toFixed(3) : `${v.toFixed(1)}%`),
    }],
  });

  renderTrendTable(data, pts, isPpl, scale);
}

/** Every plotted value as a table — the WCAG-clean twin of the chart, and the only view
 *  when there is a single point. */
function renderTrendTable(data, pts, isPpl, scale) {
  /* Newest first, but each row's Δ is against the evaluation *before* it in training
   * order — which is the row underneath. That is the number being read: "since last
   * time", not "since the top of the table". */
  const rows = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    const older = pts[i - 1];
    rows.push([
      fmt.int(p.step),
      p.best_val == null ? '–' : fmt.num(p.best_val, 4),
      isPpl ? fmt.num(p.score, 3) : fmt.pct(p.score, 1),
      older ? deltaCell((p.score - older.score) * scale, isPpl, data.suite) : '',
      String(p.n || ''),
      p.checkpoint + (p.adapter ? ` + ${p.adapter}` : ''),
    ]);
  }
  const fresh = table(['step', 'val', data.suite, 'Δ', 'n', 'checkpoint'], rows);
  fresh.id = 'ev-trend-table';
  fresh.className = 'data ev-table';
  $('#ev-trend-table').replaceWith(fresh);
}

/** A signed change, coloured by whether it is an improvement for *this* suite. */
function deltaCell(delta, isPpl, suite) {
  const span = document.createElement('span');
  span.textContent = `${delta >= 0 ? '+' : ''}${isPpl ? delta.toFixed(3) : delta.toFixed(1)}`;
  const better = LOWER_IS_BETTER.has(suite) ? delta < 0 : delta > 0;
  span.className = Math.abs(delta) < 1e-9 ? 'ev-dim' : better ? 'ev-good' : 'ev-bad';
  return span;
}

/* ---------------------------------------------------------------- polling -------------- */

async function pollEval() {
  clearTimeout(ev.timer);
  if (state.view !== 'evals' || document.hidden) return;
  try {
    const st = await api('/api/eval?lines=250');
    const wasRunning = ev.status && ev.status.running;
    renderStatus(st);
    /* A finished job means a new row and a new point — refresh the trend once, on the
     * transition, rather than refetching it every two seconds forever. */
    if (wasRunning && !st.running) loadTrend();
    ev.timer = setTimeout(pollEval, st.running ? 2000 : 10000);
  } catch (err) {
    $('#ev-state').textContent = `no answer — ${err.message}`;
    ev.timer = setTimeout(pollEval, 10000);
  }
}

async function startEval() {
  const limit = $('#ev-limit').value;
  try {
    $('#ev-run').disabled = true;
    const res = await post('/api/eval/start', {
      checkpoint: $('#ev-ckpt').value,
      adapter: $('#ev-adapter').value || null,
      suites: [...ev.selected].join(','),
      limit: limit === '' ? null : Number(limit),
      device: $('#ev-device').value || null,
      label: $('#ev-label').value || 'eval',
    });
    flash(`Evaluating ${res.checkpoint} on ${(res.suites || []).join(', ')} (${res.device}). ${res.device_reason || ''}`, 'ok');
    $('#ev-log').textContent = 'starting…';
    pollEval();
  } catch (err) {
    flash(err.message, 'error');
    renderRunNote();
  }
}

export function wireEvalTab() {
  $('#ev-run').addEventListener('click', startEval);
  $('#ev-stop').addEventListener('click', async () => {
    if (!confirm('Stop this evaluation?\n\nA result is written at the end, so a stopped '
      + 'job records nothing. That is deliberate: half a benchmark looks like a whole one '
      + 'in a table.')) return;
    try {
      const res = await post('/api/eval/stop', {});
      flash(res.note || 'Stopped.', 'ok');
    } catch (err) { flash(err.message, 'error'); }
    pollEval();
  });
  $('#ev-fetch').addEventListener('click', async () => {
    const missing = ((ev.status || {}).datasets || []).filter((d) => !d.cached).map((d) => d.name);
    try {
      await post('/api/eval/fetch', { datasets: missing });
      flash(`Downloading ${missing.join(', ')}. Fetched once, then read from disk forever.`, 'ok');
      pollEval();
    } catch (err) { flash(err.message, 'error'); }
  });
  for (const chip of $$('.ev-groups .chip')) {
    chip.addEventListener('click', () => applyGroup(chip.dataset.group));
  }
  $('#ev-trend-suite').addEventListener('change', (e) => {
    ev.trendSuite = e.target.value;
    loadTrend();
  });
}

registerTab('evals', { open: openEvalTab, leave: () => clearTimeout(ev.timer) });

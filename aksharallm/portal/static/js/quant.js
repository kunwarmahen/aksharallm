import { $, api, escHtml, flash, fmt, post } from './core.js';
import { state } from './state.js';
import { table } from './charts.js';
import { registerTab } from './router.js';
import { UNIT_MINUTES, UNIT_MORE_STEPS, boundPicker, fmtWhen, progressFromLog, stopLabel } from './dashboard.js';

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
  $('#q-stop').textContent = !st.can_bound ? 'Stop'
    : st.stop && !st.stop.now ? 'Change stop…' : 'Stop…';
  $('#q-run').disabled = running || !quant.ckpts.some((c) => c.can_quantize);
  $('#q-compare').disabled = $('#q-run').disabled;

  if (cur) {
    const state = running ? 'running' : cur.state;
    const started = cur.started ? new Date(cur.started * 1000).toLocaleTimeString() : '';
    $('#q-state').textContent =
      `${state} — ${cur.method} on ${cur.checkpoint} (${cur.device}), started ${started}`
      + stopLabel(st.stop);
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

export function wireQuantTab() {
  $('#q-run').addEventListener('click', () => startQuant(false));
  $('#q-compare').addEventListener('click', () => startQuant(true));
  $('#q-stop').addEventListener('click', async () => {
    const st = quant.status || {};
    /* Only QAT has steps to stop at — the other methods are one pass over the weights, so
     * the button stays the blunt instrument it was, and says what that costs. */
    if (!st.can_bound) {
      if (!confirm('Stop this job now?\n\nRTN, GPTQ and AWQ are a single pass over the '
        + 'weights with no useful halfway point, so this kills it and writes nothing.')) return;
      try { await post('/api/quant/stop', { mode: 'now' }); flash('Stopped the quantization job.', 'ok'); }
      catch (err) { flash(err.message, 'error'); }
      pollQuant();
      return;
    }
    const at = progressFromLog(st.log, /qat step (\d+)\/(\d+)/);
    const chosen = await boundPicker.open({
      title: 'Stop this QAT run at…',
      sub: at ? `It is at step ${fmt.int(at.step)} of ${fmt.int(at.total)}. Stopping early `
        + 'still exports and measures the model — QAT only nudges a trained checkpoint, so a '
        + 'partly-recovered one is usable.'
        : 'Stopping early still exports and measures the model.',
      unit: 'in', showNow: true, nowLabel: 'Stop now (writes nothing)',
      units: [
        { ...UNIT_MINUTES, preview: (v) => `stops about ${fmtWhen(v)}, then exports` },
        { ...UNIT_MORE_STEPS, id: 'at', noun: 'stop at step',
          min: 10, max: at ? at.total : 5000, value: at ? at.step + 100 : 200,
          preview: (v) => (at && v <= at.step ? 'already past that step — it will stop at once'
            : `stops at QAT step ${fmt.int(v)}${at ? ` of ${fmt.int(at.total)}` : ''}, then exports`) },
      ],
    });
    if (!chosen) return;
    const body = chosen === 'now' ? { mode: 'now' }
      : chosen.unit === 'in' ? { mode: 'in', seconds: chosen.value }
      : { mode: 'at', steps: chosen.value };
    try {
      const res = await post('/api/quant/stop', body);
      flash(res.note || 'Stop requested.', 'ok');
    } catch (err) { flash(err.message, 'error'); }
    pollQuant();
  });
  for (const id of ['#q-method', '#q-group', '#q-bits']) {
    $(id).addEventListener('change', () => { if (quant.status) renderQuantForm(quant.status); });
  }
}

registerTab('quant', { open: openQuantTab, leave: () => clearTimeout(quant.timer) });

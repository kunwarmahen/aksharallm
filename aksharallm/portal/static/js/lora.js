import { $, api, escHtml, flash, fmt, post } from './core.js';
import { state } from './state.js';
import { table } from './charts.js';
import { registerTab } from './router.js';
import { UNIT_MINUTES, UNIT_MORE_STEPS, boundPicker, fmtWhen, progressFromLog, stopLabel } from './dashboard.js';

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
  $('#l-stop').textContent = st.stop && !st.stop.now ? 'Change stop…' : 'Stop…';
  $('#l-run').disabled = running || !lora.ckpts.some((c) => !c.error)
                         || !(st.datasets || []).length;

  if (cur) {
    const label = running ? 'running' : cur.state;
    const started = cur.started ? new Date(cur.started * 1000).toLocaleTimeString() : '';
    const what = cur.method === 'full' ? 'full fine-tune'
                                       : `${cur.method} r=${cur.r} on ${cur.targets}`;
    $('#l-state').textContent =
      `${label} — ${what}, ${cur.checkpoint} (${cur.device}), started ${started}`
      + stopLabel(st.stop);
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

export function wireLoraTab() {
  $('#l-run').addEventListener('click', startLora);
  $('#l-stop').addEventListener('click', async () => {
    const st = lora.status || {};
    const at = progressFromLog(st.log, /step\s+(\d+)\/(\d+)/);
    const chosen = await boundPicker.open({
      title: 'Stop this fine-tune at…',
      sub: (at ? `It is at step ${fmt.int(at.step)} of ${fmt.int(at.total)}. ` : '')
        + 'However it stops, it evaluates once more and writes sft_last and sft_best — a '
        + 'stopped fine-tune still leaves a usable adapter.',
      unit: 'in', showNow: true,
      units: [
        { ...UNIT_MINUTES, max: 6 * 3600, chips: [60, 300, 600, 1800, 3600, 2 * 3600],
          preview: (v) => `stops about ${fmtWhen(v)}, then evaluates and saves` },
        { ...UNIT_MORE_STEPS, id: 'at', noun: 'stop at step',
          min: 10, max: at ? at.total : 10000, value: at ? at.step + 200 : 500,
          preview: (v) => (at && v <= at.step ? 'already past that step — it will stop at once'
            : `stops at step ${fmt.int(v)}${at ? ` of ${fmt.int(at.total)}` : ''}, then saves`) },
      ],
    });
    if (!chosen) return;
    const body = chosen === 'now' ? { mode: 'now' }
      : chosen.unit === 'in' ? { mode: 'in', seconds: chosen.value }
      : { mode: 'at', steps: chosen.value };
    try {
      const res = await post('/api/lora/stop', body);
      flash(res.note || 'Stop requested.', 'ok');
    } catch (err) { flash(err.message, 'error'); }
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

registerTab('lora', { open: openLoraTab, leave: () => clearTimeout(lora.timer) });

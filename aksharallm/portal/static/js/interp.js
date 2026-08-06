/* The Interp tab: four ways of looking inside the model that is sitting in memory anyway.
 *
 * Everything here runs *inline* on the Playground's resident model — no job, no log to tail —
 * because a logit lens is one forward pass and a patch grid is a few hundred. That also means
 * this tab inherits the device policy: if a run is training, the model is on the CPU and the
 * status line says so, so opening this page can never cost somebody a run.
 *
 * The four views share one prompt and one checkpoint on purpose. The whole value of having
 * them together is asking the same question twice: the lens says *when* the answer appeared,
 * patching says *what carried it*, and a story that only one of them supports is not a story.
 */
import { $, $$, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';

const interp = { loaded: false, view: 'lens', ckpt: null, attn: null, layers: 0 };

function status(text, kind = '') {
  const el = $('#interp-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

function showInterpView(name) {
  interp.view = name;
  for (const b of $$('.interp-tabs button')) b.classList.toggle('on', b.dataset.view === name);
  for (const v of $$('.interp-view')) v.hidden = v.id !== `ipanel-${name}`;
  if (name === 'attn' && !interp.attn) loadAttention();
  if (name === 'sae') loadFeatures();
}

async function loadCheckpoints() {
  const res = await api('/api/interp');
  const sel = $('#interp-ckpt');
  sel.innerHTML = (res.checkpoints || [])
    .map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)}</option>`).join('');
  if (res.loaded) sel.value = res.loaded.rel || sel.value;
  interp.ckpt = sel.value;
}

/* ---- logit lens ---------------------------------------------------------------------- */

function renderLens(d) {
  $('#lens-story').innerHTML = d.settled_label
    ? `It answers <strong>${escHtml(d.answer_text)}</strong>, and it settles on that at `
      + `<strong>${escHtml(d.settled_label)}</strong> of ${d.layers - 1} — changing its mind `
      + `${d.flips} time${d.flips === 1 ? '' : 's'} on the way.`
    : `It answers <strong>${escHtml(d.answer_text || '?')}</strong>, and never settles: the top `
      + `token is still moving in the last layer.`;

  const rows = d.rows.map((r) => {
    const cells = r.top.map((t, i) =>
      `<span class="lens-tok${i === 0 ? ' lens-top' : ''}">${escHtml(t.text)}`
      + `<em>${fmt.num(t.prob, 2)}</em></span>`).join('');
    return `<tr><th>${escHtml(r.label)}</th><td class="lens-entropy">${fmt.num(r.entropy, 2)}</td>`
      + `<td>${cells}</td></tr>`;
  }).join('');
  $('#lens-table').innerHTML =
    `<table class="lens-table"><thead><tr><th>after</th><th>entropy</th>`
    + `<th>what it would have said</th></tr></thead><tbody>${rows}</tbody></table>`;

  /* How much each block moved the stream, and whether it argued for the final answer. A bar
   * per block: the eye finds "the work happens in blocks 18-21" instantly and would not find
   * it in the table above. */
  const contrib = d.contributions || [];
  const max = Math.max(...contrib.map((c) => Math.abs(c.answer_delta)), 0.001);
  $('#lens-chart').innerHTML = contrib.map((c) => {
    const w = (Math.abs(c.answer_delta) / max) * 100;
    const cls = c.answer_delta >= 0 ? 'up' : 'down';
    return `<div class="contrib-row" title="block ${c.layer}: ${c.answer_delta.toFixed(3)} on the answer's logit, ||delta|| ${c.norm_delta.toFixed(1)}">`
      + `<span class="contrib-label">${c.layer}</span>`
      + `<span class="contrib-bar"><i class="${cls}" style="width:${w}%"></i></span></div>`;
  }).join('');
}

async function runLens() {
  const prompt = $('#interp-prompt').value.trim();
  if (!prompt) return;
  status('running the model once per layer…');
  try {
    const d = await post('/api/interp/lens', { checkpoint: interp.ckpt, prompt });
    renderLens(d);
    status(`${d.rows.length} layers · ${d.checkpoint} on the ${d.device.toUpperCase()}`);
    interp.attn = null;                 /* the prompt changed; the maps are stale */
    if (interp.view === 'attn') loadAttention();
  } catch (err) {
    status(err.message, 'err');
  }
}

/* ---- attention ------------------------------------------------------------------------ */

function renderAttention(d) {
  interp.layers = d.layers;
  $('#attn-layer').max = String(d.layers - 1);
  const head = $('#attn-head');
  if (head.options.length !== d.heads) {
    head.innerHTML = Array.from({ length: d.heads },
      (_, i) => `<option value="${i}">head ${i}</option>`).join('');
  }
  $('#attn-heads').innerHTML =
    `<table class="lens-table"><thead><tr><th>head</th><th>looks back</th><th>self</th>`
    + `<th>the last token attended to</th></tr></thead><tbody>`
    + d.summary.map((h) => `<tr><th>${h.head}</th><td>${fmt.num(h.distance, 2)}</td>`
      + `<td>${fmt.num(h.self_weight, 2)}</td><td>`
      + h.attends_to.map((a) => `<span class="lens-tok">${escHtml(a.token)}`
        + `<em>${fmt.num(a.weight, 2)}</em></span>`).join('') + '</td></tr>').join('')
    + '</tbody></table>';

  if (!d.matrix) { $('#attn-map').innerHTML = ''; return; }
  /* The map itself. One cell per (query, key); opacity is the weight, so the diagonal band of
   * a local head and the vertical stripe of a head that fixates on one token are both visible
   * without reading a number. */
  const head0 = `<tr><th></th>${d.tokens.map((t) =>
    `<th class="attn-key">${escHtml(t)}</th>`).join('')}</tr>`;
  const body = d.matrix.map((row, i) =>
    `<tr><th class="attn-query">${escHtml(d.tokens[i])}</th>`
    + row.map((v, j) => j > i ? '<td class="attn-void"></td>'
      : `<td class="attn-cell" style="--w:${v.toFixed(3)}" title="${d.tokens[i]} → ${d.tokens[j]}: ${v.toFixed(3)}"></td>`).join('')
    + '</tr>').join('');
  $('#attn-map').innerHTML = `<table class="attn-table">${head0}${body}</table>`;
}

async function loadAttention() {
  const prompt = $('#interp-prompt').value.trim();
  if (!prompt) return;
  status('recomputing attention…');
  try {
    const d = await post('/api/interp/attn', {
      checkpoint: interp.ckpt, prompt,
      layer: Number($('#attn-layer').value || 0),
      head: Number($('#attn-head').value || 0),
    });
    interp.attn = d;
    renderAttention(d);
    status(`layer ${d.layer} of ${d.layers - 1} · ${d.heads} heads`);
  } catch (err) {
    status(err.message, 'err');
  }
}

/* ---- patching --------------------------------------------------------------------------- */

function renderPatch(d) {
  $('#patch-story').textContent = d.summary;
  const head = `<tr><th>block</th>${d.tokens.map((t) =>
    `<th class="attn-key">${escHtml(t)}</th>`).join('')}</tr>`;
  const body = d.grid.map((row, li) =>
    `<tr><th>${li}</th>` + row.map((v) => {
      const clamped = Math.max(0, Math.min(1, v));
      return `<td class="patch-cell" style="--w:${clamped.toFixed(3)}" `
        + `title="block ${li}, position ${d.positions[row.indexOf(v)] ?? ''}: `
        + `${(v * 100).toFixed(0)}% restored">${v > 0.15 ? (v * 100).toFixed(0) : ''}</td>`;
    }).join('') + '</tr>').join('');
  $('#patch-grid').innerHTML = `<table class="attn-table patch-table">${head}${body}</table>`;
}

async function runPatch() {
  status('patching every layer at every position — a few hundred forward passes…');
  try {
    const d = await post('/api/interp/patch', {
      checkpoint: interp.ckpt,
      clean: $('#patch-clean').value, corrupt: $('#patch-corrupt').value,
      answer: $('#patch-answer').value, other: $('#patch-other').value,
    });
    renderPatch(d);
    status(`clean ${fmt.num(d.clean_diff, 2)} · corrupted ${fmt.num(d.corrupt_diff, 2)} logit diff`);
  } catch (err) {
    $('#patch-story').textContent = '';
    $('#patch-grid').innerHTML = '';
    status(err.message, 'err');
  }
}

/* ---- dictionary features ------------------------------------------------------------------ */

async function loadFeatures() {
  const layer = Number($('#sae-layer').value || 12);
  try {
    const d = await api(`/api/interp/features?checkpoint=${encodeURIComponent(interp.ckpt)}&layer=${layer}`);
    if (!d.trained) {
      $('#sae-body').innerHTML = `<p class="hint">${escHtml(d.hint)}</p>`;
      return;
    }
    const r = d.report || {};
    const last = (d.history || [])[d.history.length - 1] || {};
    $('#sae-body').innerHTML = `
      <div class="tiles">
        <div class="tile"><div class="tile-value">${fmt.pct(last.explained, 1)}</div>
          <div class="tile-label">variance explained</div>
          <div class="tile-note">how faithful the dictionary is</div></div>
        <div class="tile"><div class="tile-value">${fmt.num(last.l0, 1)}</div>
          <div class="tile-label">features per token</div>
          <div class="tile-note">L0 — how sparse it actually is</div></div>
        <div class="tile"><div class="tile-value">${fmt.pct(1 - (r.dead_fraction || 0), 0)}</div>
          <div class="tile-label">alive</div>
          <div class="tile-note">${fmt.int(r.dead || 0)} of ${fmt.int(r.n_features || 0)} never fired</div></div>
        <div class="tile"><div class="tile-value">${fmt.int(d.config && d.config.n_features)}</div>
          <div class="tile-label">dictionary size</div>
          <div class="tile-note">${fmt.int(d.config && d.config.d_model)} dims, alpha ${d.config && d.config.alpha}</div></div>
      </div>
      <table class="lens-table"><thead><tr><th>feature</th><th>fires on</th>
        <th>mean strength</th><th></th></tr></thead><tbody>`
      + (r.features || []).map((f) => `<tr><th>${f.id}</th>`
        + `<td>${fmt.pct(f.rate, 2)} of tokens</td><td>${fmt.num(f.mean_activation, 2)}</td>`
        + `<td><code>python -m aksharallm.interp features ${escHtml((interp.ckpt || '').split('/')[0])} --layer ${d.layer} --feature ${f.id}</code></td></tr>`).join('')
      + '</tbody></table>';
  } catch (err) {
    $('#sae-body').innerHTML = `<p class="hint">${escHtml(err.message)}</p>`;
  }
}

/* ---- wiring ------------------------------------------------------------------------------- */

async function openInterpTab() {
  if (interp.loaded) return;
  interp.loaded = true;
  try {
    await loadCheckpoints();
    $('#interp-ckpt').addEventListener('change', (e) => { interp.ckpt = e.target.value; });
    $('#interp-run').addEventListener('click', runLens);
    $('#interp-prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') runLens(); });
    $('#patch-run').addEventListener('click', runPatch);
    $('#attn-layer').addEventListener('change', loadAttention);
    $('#attn-head').addEventListener('change', loadAttention);
    $('#sae-layer').addEventListener('change', loadFeatures);
    for (const b of $$('.interp-tabs button')) {
      b.addEventListener('click', () => showInterpView(b.dataset.view));
    }
    runLens();
  } catch (err) {
    status(err.message, 'err');
  }
}

registerTab('interp', { open: openInterpTab });

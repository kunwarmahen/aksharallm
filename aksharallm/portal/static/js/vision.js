/* The Vision tab: pictures, captions, and three booleans per caption.
 *
 * The reason this tab exists rather than a chart is that a caption is only interesting
 * beside the image it describes. Everything else in this portal is a number you read.
 *
 * Images arrive as `data:` URIs for the same reason the Audio tab's clips do: a rendered
 * image is derived from a corpus and an index and has no stable identity, so a URL for it
 * would be a cache-invalidation problem in exchange for nothing.
 */
import { $, api, escHtml, fmt, post } from './core.js';
import { registerTab } from './router.js';

const vs = { ckpt: null, corpus: null, busy: false };

function status(text, kind = '') {
  const el = $('#vs-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

function card(row, scored) {
  const marks = scored
    ? `<span class="vs-marks">${['count', 'colour', 'shape']
        .map((k) => `<span class="vs-mark${row.marks[k] ? ' on' : ''}">${k}</span>`).join('')}</span>`
    : '';
  const said = scored ? `<span class="vs-said">${escHtml(row.said)}</span>` : '';
  const cls = scored ? (row.all_three ? ' vs-right' : ' vs-wrong') : '';
  return `<div class="vs-card${cls}">
    <img src="${row.png}" alt="${escHtml(row.truth)}">
    ${said}
    <span class="vs-truth">${escHtml(row.truth)}</span>
    ${marks}
  </div>`;
}

async function loadOverview() {
  const res = await api('/api/vision');
  const cks = res.checkpoints || [];
  const corpora = res.corpora || [];

  $('#vs-empty').hidden = cks.length > 0;
  $('#vs-ckpt').innerHTML = cks.map((c) =>
    `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — step ${fmt.int(c.step)} · `
    + `${c.patches} patches → ${c.image_tokens} tokens</option>`).join('');
  $('#vs-corpus').innerHTML = corpora.map((c) => {
    const held = c.missing_pairs.length
      ? ` · never shown: ${c.missing_pairs.slice(0, 2).join(', ')}` : '';
    return `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — ${fmt.int(c.images)} images `
      + `@ ${c.size}px${escHtml(held)}</option>`;
  }).join('');

  if (cks.length) { vs.ckpt = vs.ckpt || cks[0].rel; $('#vs-ckpt').value = vs.ckpt; }
  if (corpora.length) { vs.corpus = vs.corpus || corpora[0].rel; $('#vs-corpus').value = vs.corpus; }
  $('#vs-run').disabled = !(vs.ckpt && vs.corpus);
  $('#vs-samples').disabled = !vs.corpus;
  status(corpora.length ? `on the ${res.device} — ${res.device_reason}`
    : 'no corpus yet — render one with `python -m aksharallm.vision corpus`');
}

async function loadRuns() {
  const res = await api('/api/vision/runs');
  const rows = res.runs || [];
  $('#vs-runs').innerHTML = rows.length
    ? '<thead><tr><th>run</th><th>state</th><th>step</th><th>all three</th>'
      + '<th>held-out combination</th></tr></thead><tbody>'
      + rows.map((r) => {
        const s = r.score || {};
        const h = s.holdout || {};
        const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
        return `<tr><td>${escHtml(r.name)}</td>`
          + `<td>${r.training ? '<b>training</b>' : 'idle'}</td>`
          + `<td>${r.step == null ? '—' : fmt.int(r.step)}</td>`
          + `<td>${pct(s.all_three)}</td><td>${pct(h.all_three)}</td></tr>`;
      }).join('') + '</tbody>'
    : '<tbody><tr><td>no vision runs yet</td></tr></tbody>';
}

async function showSamples() {
  if (!vs.corpus) return;
  $('#vs-samples').disabled = true;
  try {
    const res = await post('/api/vision/samples',
      { corpus: vs.corpus, n: Number($('#vs-n').value) || 12 });
    if (res.error) { status(res.error, 'warn'); return; }
    $('#vs-score-panel').hidden = true;
    $('#vs-grid-title').textContent = `The corpus — ${res.n} of ${fmt.int(res.total)} held-out images`;
    $('#vs-grid').innerHTML = res.rows.map((r) => card(r, false)).join('');
    status(`${res.n} images from ${res.corpus} (${res.split})`);
  } finally {
    $('#vs-samples').disabled = false;
  }
}

async function runCaption() {
  if (vs.busy || !vs.ckpt || !vs.corpus) return;
  vs.busy = true;
  $('#vs-run').disabled = true;
  status('captioning…');
  try {
    const res = await post('/api/vision/caption', {
      checkpoint: vs.ckpt, corpus: vs.corpus, n: Number($('#vs-n').value) || 12,
    });
    if (res.error) { status(res.error, 'warn'); return; }

    const s = res.score;
    const pct = (v) => `${Math.round(v * 100)}%`;
    $('#vs-score-panel').hidden = false;
    $('#vs-score').innerHTML =
      '<thead><tr><th>images</th><th>count</th><th>colour</th><th>shape</th>'
      + '<th>all three</th></tr></thead>'
      + `<tbody><tr><td>${s.n}</td><td>${pct(s.count)}</td><td>${pct(s.colour)}</td>`
      + `<td>${pct(s.shape)}</td><td><b>${pct(s.all_three)}</b></td></tr></tbody>`;
    $('#vs-note').textContent = res.note;
    $('#vs-grid-title').textContent = 'What it said';
    $('#vs-grid').innerHTML = res.rows.map((r) => card(r, true)).join('');
    $('#vs-params').textContent =
      `${fmt.int(res.params.trainable)} trainable parameters against `
      + `${fmt.int(res.params.language_model)} frozen · an image costs `
      + `${res.image_tokens} positions of the model's context`;
    status(`${s.n} images on the ${res.device} — ${res.device_reason}`);
  } catch (e) {
    status(`no answer from the portal — ${e.message}`, 'warn');
  } finally {
    vs.busy = false;
    $('#vs-run').disabled = false;
  }
}

registerTab('vision', {
  async open() {
    $('#vs-ckpt').onchange = (e) => { vs.ckpt = e.target.value; };
    $('#vs-corpus').onchange = (e) => { vs.corpus = e.target.value; };
    $('#vs-run').onclick = runCaption;
    $('#vs-samples').onclick = showSamples;
    await loadOverview();
    await loadRuns();
  },
  /* Nothing polls here; the hook stays declared so adding a poll later cannot forget it. */
  leave() {},
});

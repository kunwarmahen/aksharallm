/* The Audio tab: the bitrate ladder, the collapse detector, and the integers themselves.
 *
 * The whole tab exists for one panel. Every other visualisation in this portal asks you to
 * interpret a chart; the ladder asks you to listen, and what it makes audible is the same
 * trade quantization makes silently in the weights.
 *
 * Audio arrives as `data:` URIs rather than from a route, because a reconstruction is
 * derived from a checkpoint and a clip and has no stable identity — a URL for it would be a
 * cache-invalidation problem in exchange for nothing.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';

const au = { loaded: false, ckpt: null, corpus: null, busy: false };

function status(text, kind = '') {
  const el = $('#au-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

/* ---- a spectrogram, as one canvas --------------------------------------------------- */

/* Painted rather than built from elements: an 80x160 grid is 12,800 nodes as divs, which is
 * what makes a tab feel slow, and it is invisible until the corpus gets bigger. */
function paintSpectrogram(canvas, spec) {
  if (!spec || !spec.rows || !spec.rows.length) return;
  const h = spec.rows.length;
  const w = spec.rows[0].length;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const v = spec.rows[y][x] / 100;
      const i = (y * w + x) * 4;
      /* A single-hue ramp, dark to bright. Two hues would read as two quantities. */
      img.data[i] = Math.round(20 + v * 235);
      img.data[i + 1] = Math.round(16 + v * 190);
      img.data[i + 2] = Math.round(48 + v * 120);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* ---- discovery ---------------------------------------------------------------------- */

async function loadOverview() {
  const res = await api('/api/audio');
  const cks = res.checkpoints || [];
  const corpora = res.corpora || [];

  $('#au-empty').hidden = cks.length > 0 && corpora.length > 0;
  $('#au-ckpt').innerHTML = cks.map((c) =>
    `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — step ${fmt.int(c.step)} · `
    + `${c.n_codebooks}x${c.codebook_size} · ${c.kbps} kbps</option>`).join('');
  $('#au-corpus').innerHTML = corpora.map((c) =>
    `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — ${fmt.int(c.clips)} clips, `
    + `${c.hours} h @ ${fmt.int(c.sample_rate)} Hz</option>`).join('');

  if (cks.length) { au.ckpt = au.ckpt || cks[0].rel; $('#au-ckpt').value = au.ckpt; }
  if (corpora.length) { au.corpus = au.corpus || corpora[0].rel; $('#au-corpus').value = au.corpus; }

  for (const id of ['#au-run', '#au-usage', '#au-tokens']) {
    $(id).disabled = !(au.ckpt && au.corpus);
  }
  status(cks.length && corpora.length
    ? `on the ${res.device} — ${res.device_reason}`
    : 'nothing to load yet');
  return res;
}

async function loadRuns() {
  const res = await api('/api/audio/runs');
  const rows = res.runs || [];
  $('#au-runs').innerHTML = rows.length
    ? `<thead><tr><th>run</th><th>kind</th><th>state</th><th>step</th><th>samples</th></tr></thead>`
      + `<tbody>${rows.map((r) => `<tr><td>${escHtml(r.name)}</td><td>${escHtml(r.kind)}</td>`
        + `<td>${r.training ? '<b>training</b>' : 'idle'}</td>`
        + `<td>${r.step == null ? '—' : fmt.int(r.step)}</td>`
        + `<td>${r.samples ? `<code>${escHtml(r.samples)}</code>` : '—'}</td></tr>`).join('')}</tbody>`
    : '<tbody><tr><td>no audio runs yet</td></tr></tbody>';
}

/* ---- the ladder --------------------------------------------------------------------- */

async function runLadder() {
  if (au.busy || !au.ckpt || !au.corpus) return;
  au.busy = true;
  $('#au-run').disabled = true;
  status('reconstructing at every bitrate…');
  try {
    const res = await post('/api/audio/ladder', {
      checkpoint: au.ckpt, corpus: au.corpus,
      index: Number($('#au-index').value) || 0,
      seconds: Number($('#au-seconds').value) || 4,
    });
    if (res.error) { status(res.error, 'warn'); return; }

    $('#au-ladder').innerHTML = res.rows.map((r) => `
      <div class="au-rung${r.codebooks == null ? ' au-original' : ''}">
        <h4>${escHtml(r.label)}</h4>
        <canvas class="au-spec" data-i="${escHtml(String(r.codebooks ?? 'orig'))}"></canvas>
        <audio controls preload="none" src="${r.audio}"></audio>
        <span class="au-num"><span>bitrate</span><b>${r.kbps} kbps</b></span>
        ${r.compression ? `<span class="au-num"><span>vs PCM</span><b>${r.compression}×</b></span>` : ''}
        ${r.convergence != null ? `<span class="au-num"><span>convergence</span><b>${r.convergence}</b></span>` : ''}
        ${r.mcd_db != null ? `<span class="au-num"><span>MCD</span><b>${r.mcd_db} dB</b></span>` : ''}
      </div>`).join('');

    /* Paint after the markup lands, so every canvas exists. */
    const canvases = $('#au-ladder').querySelectorAll('canvas');
    res.rows.forEach((r, i) => paintSpectrogram(canvases[i], r.spectrogram));

    $('#au-caveat').textContent = res.caveat || '';
    status(`clip ${res.index + 1} of ${res.n_clips}, ${res.seconds}s, on the ${res.device}`);
  } catch (e) {
    status(`no answer from the portal — ${e.message}`, 'warn');
  } finally {
    au.busy = false;
    $('#au-run').disabled = false;
  }
}

/* ---- usage -------------------------------------------------------------------------- */

async function runUsage() {
  if (!au.ckpt || !au.corpus) return;
  $('#au-usage').disabled = true;
  try {
    const res = await post('/api/audio/usage', { checkpoint: au.ckpt, corpus: au.corpus });
    if (res.error) { flash(res.error, 'warn'); return; }
    $('#au-usage-table').innerHTML =
      '<thead><tr><th>codebook</th><th>entries used</th><th>perplexity</th><th>effective size</th></tr></thead>'
      + `<tbody>${res.rows.map((r) => {
        const pct = Math.round(r.usage * 100);
        return `<tr><td>${r.codebook}</td><td>${fmt.int(r.used)} / ${fmt.int(r.size)}</td>`
          + `<td>${r.perplexity}</td>`
          + `<td><span class="au-bar${pct < 5 ? ' au-bar-low' : ''}" style="width:${Math.max(pct, 1)}%"></span> ${pct}%</td></tr>`;
      }).join('')}</tbody>`;
    $('#au-usage-note').textContent = `${res.note} (measured on ${res.clips} held-out clips)`;
  } finally {
    $('#au-usage').disabled = false;
  }
}

/* ---- the integers ------------------------------------------------------------------- */

async function runTokens() {
  if (!au.ckpt || !au.corpus) return;
  $('#au-tokens').disabled = true;
  try {
    const res = await post('/api/audio/tokens', {
      checkpoint: au.ckpt, corpus: au.corpus,
      index: Number($('#au-index').value) || 0,
    });
    if (res.error) { flash(res.error, 'warn'); return; }
    $('#au-tokens-summary').textContent =
      `${res.seconds}s → ${fmt.int(res.n_frames)} frames × ${res.n_codebooks} codebooks `
      + `= ${fmt.int(res.positions)} integers at ${res.frames_per_second} frames/s. `
      + `A language model over this pays ${fmt.int(res.n_frames)} positions.`;
    $('#au-tokens-table').innerHTML = `<tbody>${res.codes.map((row, k) =>
      `<tr><th>book ${k}</th>${row.map((v) => `<td>${v}</td>`).join('')}`
      + `${res.truncated ? '<td>…</td>' : ''}</tr>`).join('')}</tbody>`;
  } finally {
    $('#au-tokens').disabled = false;
  }
}

/* ---- lifecycle ---------------------------------------------------------------------- */

registerTab('audio', {
  async open() {
    $('#au-ckpt').onchange = (e) => { au.ckpt = e.target.value; };
    $('#au-corpus').onchange = (e) => { au.corpus = e.target.value; };
    $('#au-run').onclick = runLadder;
    $('#au-usage').onclick = runUsage;
    $('#au-tokens').onclick = runTokens;
    await loadOverview();
    await loadRuns();
    au.loaded = true;
  },
  /* Nothing polls here, so there is nothing to tear down — but the hook stays declared so
   * that adding a poll later cannot forget it. That omission is how a tab keeps working
   * behind whatever the reader opened next. */
  leave() {},
});

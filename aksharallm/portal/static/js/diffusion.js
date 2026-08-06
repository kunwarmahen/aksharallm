/* The Diffusion tab: the second paradigm, which is far easier to watch than to read about.
 *
 * The whole tab is built around one claim — that the *order* a diffusion model decides
 * positions in is the interesting thing, and it is invisible in the finished text. So the
 * server returns every intermediate state of a generation and this module animates them.
 * Everything else here (the corruption preview, infilling, the measurement panel) exists to
 * make that one animation mean something.
 *
 * Runs inline on the Playground's resident model, like the Interp tab, so it inherits the
 * device policy: while a run is training the model is on the CPU and the status line says so.
 */
import { $, $$, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';

const diff = { loaded: false, ckpt: null, trace: null, at: 0, timer: null, prefixLen: 0 };

function status(text, kind = '') {
  const el = $('#diff-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

/* ---- checkpoints ---------------------------------------------------------------------- */

async function loadCheckpoints() {
  const res = await api(`/api/diffusion${diff.ckpt ? `?checkpoint=${encodeURIComponent(diff.ckpt)}` : ''}`);
  const sel = $('#diff-ckpt');
  const cks = res.checkpoints || [];
  /* Non-diffusion checkpoints are listed and disabled rather than hidden: a picker that
   * silently drops every checkpoint on a machine that has not trained one looks broken. */
  sel.innerHTML = cks.map((c) => `<option value="${escHtml(c.rel)}"${c.diffusion ? '' : ' disabled'}>`
    + `${escHtml(c.rel)}${c.diffusion ? '' : ' — autoregressive'}</option>`).join('');
  const first = cks.find((c) => c.diffusion);
  $('#diff-empty').hidden = !!first;
  if (first) {
    diff.ckpt = (cks.find((c) => c.rel === diff.ckpt && c.diffusion) || first).rel;
    sel.value = diff.ckpt;
  } else {
    diff.ckpt = null;
  }
  const cur = res.current;
  $('#diff-current').textContent = cur && !cur.error
    ? `[MASK] = id ${cur.mask_token_id} of ${fmt.int(cur.vocab_size)} · context ${fmt.int(cur.max_seq_len)} · on the ${cur.device}`
    : '';
  for (const id of ['#diff-run', '#diff-infill-run', '#diff-corrupt-run', '#diff-elbo', '#diff-byt']) {
    $(id).disabled = !diff.ckpt;
  }
  return res;
}

/* ---- 1. the forward process --------------------------------------------------------- */

async function runCorrupt() {
  if (!diff.ckpt) return;
  const t = Number($('#diff-t').value) / 100;
  $('#diff-t-label').textContent = `${Math.round(t * 100)}%`;
  try {
    const d = await post('/api/diffusion/corrupt', {
      checkpoint: diff.ckpt, text: $('#diff-corrupt-text').value,
      t, seed: Math.floor(Math.random() * 1e6),
    });
    $('#diff-corrupt-out').innerHTML = d.tokens.map((tok, i) => (d.masked[i]
      ? `<span class="cell masked" title="${escHtml(tok)} — the model has to put this back">▁</span>`
      : `<span class="cell">${escHtml(tok)}</span>`)).join('');
    $('#diff-weight').innerHTML =
      `${d.n_masked} of ${d.n_tokens} tokens hidden (asked for ${fmt.pct(d.t, 0)}, got `
      + `${fmt.pct(d.realised, 0)} — it is a coin per position, not a quota). Each one's `
      + `cross-entropy is multiplied by <strong>1/t = ${d.weight}</strong>, so this sequence `
      + `counts the same as one masked at 90%.`;
  } catch (err) {
    status(err.message, 'err');
  }
}

/* ---- 2. generation, and the animation ------------------------------------------------ */

function drawTrace(i) {
  if (!diff.trace) return;
  const steps = diff.trace.steps;
  diff.at = Math.max(0, Math.min(i, steps.length - 1));
  const st = steps[diff.at];
  const fresh = new Map();
  st.committed.forEach((p, k) => fresh.set(p, st.confidence ? st.confidence[k] : 1));

  $('#diff-trace').innerHTML = st.cells.map((text, pos) => {
    if (pos < diff.trace.prefix_len) {
      return `<span class="cell given">${escHtml(text || '')}</span>`;
    }
    if (text == null) return '<span class="cell masked">▁</span>';
    if (fresh.has(pos)) {
      /* Opacity is the confidence: a wall of pale cells on the first step is a model
       * guessing, and the same wall going dark by step three is one that has found its
       * skeleton. No number could show that as fast. */
      const w = Math.max(0.25, Math.min(fresh.get(pos), 1));
      return `<span class="cell fresh" style="--w:${w}" title="committed at step ${st.step}, `
        + `confidence ${fmt.num(fresh.get(pos), 2)}">${escHtml(text)}</span>`;
    }
    return `<span class="cell done">${escHtml(text)}</span>`;
  }).join('');

  $('#diff-scrub').value = String(diff.at);
  $('#diff-step-label').textContent = st.step === 0
    ? 'step 0 — everything blank'
    : `step ${st.step} of ${steps.length - 1} · ${st.committed.length} committed · ${st.remaining} left`;
}

function stopPlay() {
  if (diff.timer) clearInterval(diff.timer);
  diff.timer = null;
  $('#diff-play').textContent = '▶ play';
}

function play() {
  if (diff.timer) { stopPlay(); return; }
  if (!diff.trace) return;
  if (diff.at >= diff.trace.steps.length - 1) drawTrace(0);
  $('#diff-play').textContent = '❚❚ pause';
  diff.timer = setInterval(() => {
    if (diff.at >= diff.trace.steps.length - 1) { stopPlay(); return; }
    drawTrace(diff.at + 1);
  }, 420);
}

async function runGenerate() {
  if (!diff.ckpt) return;
  stopPlay();
  status('denoising…');
  $('#diff-run').disabled = true;
  try {
    const d = await post('/api/diffusion/generate', {
      checkpoint: diff.ckpt,
      prompt: $('#diff-prompt').value,
      length: Number($('#diff-length').value),
      steps: Number($('#diff-steps').value),
      temperature: Number($('#diff-temp').value),
      remask: $('#diff-remask').value,
    });
    diff.trace = d;
    $('#diff-player').hidden = false;
    $('#diff-scrub').max = String(d.steps.length - 1);
    drawTrace(0);
    $('#diff-cost').innerHTML =
      `<strong>${d.passes} forward passes</strong> for ${fmt.int($('#diff-length').value)} `
      + `tokens — ${d.tokens_per_pass} tokens per pass, where an autoregressive model needs `
      + `exactly one pass per token. ${fmt.dur(d.elapsed_s)} on the ${d.device}. `
      + `<em>But</em> each pass here is over the whole sequence with no cache, so this is a `
      + `trade, not a free win.`;
    status(`generated on the ${d.device}`);
    play();
  } catch (err) {
    status(err.message, 'err');
    flash(err.message, 'error');
  } finally {
    $('#diff-run').disabled = false;
  }
}

/* ---- 3. infilling --------------------------------------------------------------------- */

async function runInfill() {
  if (!diff.ckpt) return;
  status('filling the gap…');
  try {
    const d = await post('/api/diffusion/infill', {
      checkpoint: diff.ckpt,
      prefix: $('#diff-prefix').value,
      suffix: $('#diff-suffix').value,
      length: Number($('#diff-mid-len').value),
    });
    $('#diff-infill-out').innerHTML = `${escHtml(d.prefix)} `
      + `<mark>${escHtml(d.middle)}</mark> ${escHtml(d.suffix)}`;
    status(`filled on the ${d.device}`);
  } catch (err) {
    status(err.message, 'err');
  }
}

/* ---- 4. measurement ------------------------------------------------------------------- */

async function measure(kind) {
  if (!diff.ckpt) return;
  status(kind === 'elbo' ? 'measuring the bound…' : 'measuring across mask rates…');
  $('#diff-measure').innerHTML = '<p class="hint">running…</p>';
  try {
    const d = await post('/api/diffusion/measure', { checkpoint: diff.ckpt, kind });
    if (d.kind === 'elbo') {
      $('#diff-measure').innerHTML = `<div class="tiles">
        <div class="tile"><div class="tile-value">${fmt.num(d.nelbo, 4)}</div>
          <div class="tile-label">NELBO, nats/token</div>
          <div class="tile-note">an upper bound on the true NLL</div></div>
        <div class="tile"><div class="tile-value">&le; ${fmt.num(d.ppl_upper_bound, 2)}</div>
          <div class="tile-label">perplexity bound</div>
          <div class="tile-note">not an AR perplexity — note the &le;</div></div>
        <div class="tile"><div class="tile-value">${fmt.num(d.ce_masked, 3)}</div>
          <div class="tile-label">ce on masked positions</div>
          <div class="tile-note">unweighted; the number with a meaning</div></div>
      </div>
      <p class="hint">A quick reading over ${d.batches} batches. For a number to write down:
        <code>python -m aksharallm.diffusion ${escHtml(diff.ckpt)} elbo --batches 20</code></p>`;
      return;
    }
    const rows = d.rows || [];
    const max = Math.max(...rows.map((r) => r.ce_masked), 0.001);
    $('#diff-measure').innerHTML = '<div class="byt">' + rows.map((r) =>
      `<div class="byt-row"><span class="byt-label">${fmt.pct(r.t, 0)}</span>`
      + `<span class="byt-bar"><i style="width:${(r.ce_masked / max) * 100}%"></i></span>`
      + `<span class="byt-val">${fmt.num(r.ce_masked, 3)}</span></div>`).join('') + '</div>';
  } catch (err) {
    $('#diff-measure').innerHTML = `<p class="hint">${escHtml(err.message)}</p>`;
    status(err.message, 'err');
  }
}

/* ---- wiring ---------------------------------------------------------------------------- */

async function openDiffTab() {
  if (diff.loaded) return;
  diff.loaded = true;
  try {
    await loadCheckpoints();
    $('#diff-ckpt').addEventListener('change', async (e) => {
      diff.ckpt = e.target.value;
      await loadCheckpoints();
    });
    $('#diff-t').addEventListener('input', runCorrupt);
    $('#diff-corrupt-text').addEventListener('change', runCorrupt);
    $('#diff-corrupt-run').addEventListener('click', runCorrupt);
    $('#diff-run').addEventListener('click', runGenerate);
    $('#diff-play').addEventListener('click', play);
    $('#diff-scrub').addEventListener('input', (e) => { stopPlay(); drawTrace(Number(e.target.value)); });
    $('#diff-infill-run').addEventListener('click', runInfill);
    $('#diff-elbo').addEventListener('click', () => measure('elbo'));
    $('#diff-byt').addEventListener('click', () => measure('by-t'));
    if (diff.ckpt) runCorrupt();
  } catch (err) {
    status(err.message, 'err');
  }
}

registerTab('diff', { open: openDiffTab, leave: stopPlay });

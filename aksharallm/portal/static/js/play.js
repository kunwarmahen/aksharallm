import { $, $$, api, escHtml, fmt, post } from './core.js';
import { state } from './state.js';
import { registerTab } from './router.js';

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
    `<div><dt>${escHtml(k)}</dt><dd>${escHtml(String(v))}</dd></div>`).join('')
    + `<p class="hint">${escHtml(c.stage_note || '')}</p>`
    + (c.error ? `<p class="err">${escHtml(c.error)}</p>` : '');
}

/** Where it will run, and why. Shown before you press Generate, not after. */
function renderPlan(status) {
  const plan = (status && status.plan) || {};
  const warn = $('#play-warn');
  if (plan.reason) {
    warn.innerHTML =
      `<strong>${plan.device === 'cuda' ? 'GPU' : 'CPU'}</strong> — ${escHtml(plan.reason)}`;
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
    + items.map((it) => `<button class="ghost preset" type="button" data-key="${escHtml(it.key)}"`
      + ` title="${escHtml(it.hint || '')}">${escHtml(it.label)}</button>`).join('');
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
  return `<div class="play-test ${cls}"><strong>${escHtml(t.status.toUpperCase())}</strong> `
    + `${escHtml(t.detail || '')}`
    + (t.program ? `<details><summary>what was executed</summary>`
      + `<pre>${escHtml(t.program)}</pre></details>` : '')
    + (t.stderr && !t.ok ? `<details><summary>stderr</summary>`
      + `<pre>${escHtml(t.stderr)}</pre></details>` : '')
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
  return `${escHtml(c.rel)} · step ${step}${val} · ${devName(dev)}`;
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
      body.innerHTML = `<span class="err">${escHtml(err.message)}</span>`;
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
      ? `<span class="pill ${r.test.ok ? 'ok' : 'bad'}">${escHtml(r.test.status)}</span>` : '';
    return `<article class="hist-row">
      <header>${escHtml(r.run || '?')}/${escHtml(r.checkpoint || '?')} ·
        step ${r.step == null ? '?' : fmt.int(r.step)} · val ${val} ${badge}</header>
      <div class="hist-when">${escHtml(r.iso || '')} · ${escHtml(r.mode || '')}
        ${r.probe ? '· probe ' + escHtml(r.probe) : ''}
        ${r.task ? '· task ' + escHtml(r.task) : ''}</div>
      <p class="hist-prompt">${escHtml((r.prompt || '').slice(0, 140))}</p>
      <p class="hist-out">${escHtml((r.output || '').slice(0, 300))}</p>
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
            · ${escHtml(r.run || '')}</header>
          <p class="hist-out">${escHtml((r.output || '').slice(0, 400))}</p>
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
        `<option value="${escHtml(p.probe)}">${escHtml(p.probe)} — ${p.count} run(s), `
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
    + usable.map((a) => `<option value="${escHtml(a.rel)}">`
        + `${escHtml(a.rel)} — r=${a.r} ${escHtml(a.targets || '')} (${escHtml(a.stage || '')})`
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
      `<option value="${escHtml(c.rel)}"${c.error ? ' disabled' : ''}>`
      + `${escHtml(ckptLabel(c))}</option>`).join('');
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

export function wirePlay() {
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

registerTab('play', { open: openPlayTab, leave: () => clearTimeout(play.statusTimer) });

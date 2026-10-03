/* The Dictation tab: hold a button, speak, read what our own recogniser heard.
 *
 * Capture is plain Web Audio rather than MediaRecorder. MediaRecorder hands back Opus in a
 * WebM container, which the server would need ffmpeg to decode; a ScriptProcessor hands back
 * float samples, which need nothing. The samples go up as 16-bit PCM at the context's own
 * rate (usually 48 kHz) and the server resamples with the repo's own resampler — so the one
 * conversion is in our code and the response says it happened.
 *
 * A dropped file goes through `decodeAudioData`, which means any format the browser can
 * play — WAV, MP3, M4A, FLAC — arrives as the same float samples.
 */
import { $, api, escHtml, fmt, post } from './core.js';
import { registerTab } from './router.js';

const dc = { ckpt: null, busy: false, rec: null, maxSeconds: 30, maxDictate: 120, last: null };

function status(text, kind = '') {
  const el = $('#dc-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

/* Float32 samples -> base64 little-endian int16. Chunked, because String.fromCharCode over
 * a few megabytes in one call overflows the argument limit. */
function toPcm16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const bytes = new Uint8Array(out.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

async function send(samples, rate, label) {
  if (!dc.ckpt) { status('pick a recogniser first', 'warn'); return; }
  dc.busy = true;
  status(`transcribing ${(samples.length / rate).toFixed(1)} s of ${label}…`);
  try {
    const res = await post('/api/dictate/transcribe', {
      checkpoint: dc.ckpt, pcm: toPcm16(samples), sample_rate: rate,
      decoder: $('#dc-decoder').value,
    });
    $('#dc-out').hidden = false;
    $('#dc-text').textContent = res.text || '(nothing — it heard no words)';
    $('#dc-text').classList.toggle('dim', !res.text);
    // Greedy beside beam, so what the language model changed is visible, not asserted.
    const diff = res.greedy !== res.text ? ` · greedy alone wrote: “${res.greedy}”` : '';
    const conv = res.resampled ? `resampled ${fmt.int(res.rate_in)} → ${fmt.int(res.rate_model)} Hz` : `${fmt.int(res.rate_model)} Hz`;
    const quiet = res.rms_dbfs < -45 ? ' — that was very quiet; move closer or speak up' : '';
    $('#dc-meta').textContent = `${res.seconds} s · ${conv} · level ${res.rms_dbfs} dBFS rms, `
      + `peak ${res.peak_dbfs}${quiet} · ${res.ms} ms on the ${res.device} (${res.realtime}x real time) · `
      + `step ${fmt.int(res.step)} · ${res.decoder}${diff}`;
    status('');
  } catch (e) {
    status(e.message, 'warn');
  } finally {
    dc.busy = false;
  }
}

/* One capture path for both buttons. `kind` is 'lab' (hold-to-talk, raw recogniser) or
 * 'dictate' (click to start, click to finish, the cleaned pipeline) — they differ only in the
 * button, the limit and where the samples go. */
const KINDS = {
  lab: { btn: '#dc-rec', idle: '● Hold to talk', live: '■ Listening — release to send',
         status: (t, k) => status(t, k), limit: () => dc.maxSeconds,
         send: (all, rate) => send(all, rate, 'your voice') },
  myvoice: { btn: '#mv-rec', idle: '● Record this sentence', live: '■ Recording — press when done',
             // The sentence's own limit, plus a little, so an over-long take stops itself
             // and is refused by the server with the reason, rather than running to 30 s.
             status: (t, k) => mvStatus(t, k), limit: () => (mv.prompts[mv.i]?.max_seconds || 12) + 1,
             send: (all, rate) => saveMyVoice(all, rate) },
  dictate: { btn: '#dc-go', idle: '● Start dictating', live: '■ Listening — press to finish',
             status: (t, k) => goStatus(t, k), limit: () => dc.maxDictate,
             send: (all, rate) => sendDictation(all, rate, 'your voice') },
};

async function startRecording(kind = 'lab') {
  const K = KINDS[kind];
  if (dc.rec || dc.busy) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    K.status('this browser will not open the microphone here (it needs localhost or https) — use a file', 'warn');
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      // Off, all three: the recogniser should hear the room as it is, and echo cancellation
      // and noise suppression are a second model's opinion about what you said.
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (e) {
    K.status(`no microphone: ${e.message}`, 'warn');
    return;
  }
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  proc.onaudioprocess = (ev) => { chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
  src.connect(proc);
  proc.connect(ctx.destination);
  const t0 = performance.now();
  dc.rec = { ctx, stream, proc, chunks, t0, kind };
  if (kind === 'dictate') startLive(dc.rec);
  $(K.btn).classList.add('dc-live');
  $(K.btn).textContent = K.live;
  K.status('listening…');
  dc.rec.timer = setTimeout(() => stopRecording(), K.limit() * 1000);
}

async function stopRecording() {
  const r = dc.rec;
  if (!r) return;
  const K = KINDS[r.kind];
  dc.rec = null;
  clearTimeout(r.timer);
  clearInterval(r.live);
  r.proc.disconnect();
  r.stream.getTracks().forEach((t) => t.stop());
  const rate = r.ctx.sampleRate;
  await r.ctx.close();
  $(K.btn).classList.remove('dc-live');
  $(K.btn).textContent = K.idle;
  const n = r.chunks.reduce((a, c) => a + c.length, 0);
  const all = new Float32Array(n);
  let at = 0;
  for (const c of r.chunks) { all.set(c, at); at += c.length; }
  if (n / rate < 0.3) { K.status('too short — say something first', 'warn'); return; }
  await K.send(all, rate);
}

/* Any audio file -> mono float samples at its own rate, via the browser's decoder. */
async function decodeFile(file, limit) {
  const ctx = new AudioContext();
  try {
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    // Downmix to mono by averaging channels — the same thing `audio/io.to_mono` does.
    const mono = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c += 1) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < ch.length; i += 1) mono[i] += ch[i] / buf.numberOfChannels;
    }
    return { samples: mono.subarray(0, Math.floor(limit * buf.sampleRate)), rate: buf.sampleRate,
             cut: buf.duration > limit, duration: buf.duration };
  } finally {
    await ctx.close();
  }
}

async function fromFile(file) {
  if (!file) return;
  try {
    const d = await decodeFile(file, dc.maxSeconds);
    if (d.cut) status(`${d.duration.toFixed(0)} s is past the ${dc.maxSeconds} s this panel takes — sending the first ${dc.maxSeconds}`, 'warn');
    await send(d.samples, d.rate, file.name);
  } catch (e) {
    status(`could not decode ${file.name}: ${e.message}`, 'warn');
  }
}

/* ---- the live preview (dictate/stream.py) --------------------------------------------
 * While you talk, the audio so far is re-read twice a second; words two readings agree on
 * are shown solid and never change until you finish, the rest grey. Only NEW samples are
 * sent each tick; the server keeps the session's audio. The final text still comes from the
 * full pipeline when you press again. */
function startLive(r) {
  r.session = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  r.sent = 0;
  r.inflight = false;
  $('#dc-live').hidden = false;
  $('#dc-live').innerHTML = '<span class="dim">…</span>';
  r.live = setInterval(async () => {
    if (r.inflight || r.sent >= r.chunks.length) return;
    r.inflight = true;
    const fresh = r.chunks.slice(r.sent);
    r.sent = r.chunks.length;
    const n = fresh.reduce((a, c) => a + c.length, 0);
    const buf = new Float32Array(n);
    let at = 0;
    for (const c of fresh) { buf.set(c, at); at += c.length; }
    try {
      const res = await post('/api/dictate/stream', { session: r.session, pcm: toPcm16(buf), sample_rate: r.ctx.sampleRate });
      if (dc.rec === r) {
        $('#dc-live').innerHTML = `${escHtml(res.stable)} <span class="dim">${escHtml(res.tentative)}</span>`;
      }
    } catch { /* a missed preview tick is not worth a message; the final pass is what counts */ }
    r.inflight = false;
  }, 600);
}

/* ---- dictation: the cleaned pipeline, the same one the desktop hotkey runs ------------- */

function goStatus(text, kind = '') {
  const el = $('#dc-go-status');
  el.textContent = text;
  el.className = `panel-note${kind ? ` ${kind}` : ''}`;
}

const STEP_WORDS = {
  fillers: (s) => `removed “${s.removed}”`,
  'scratch that': (s) => `scratch that — removed “${s.removed || '(nothing)'}”`,
  replacement: (s) => `you taught it: “${s.heard}” → “${s.wrote}”`,
  'new line': () => 'new line', 'new paragraph': () => 'new paragraph',
};

function showResult(r) {
  dc.last = r;
  $('#dc-live').hidden = true;
  $('#dc-result').hidden = false;
  $('#dc-final').value = r.text || '';
  $('#dc-final').placeholder = r.text ? '' : (r.note || 'nothing heard');
  $('#dc-teach').disabled = true;
  $('#dc-teach-note').textContent = '';
  const steps = (r.steps || []).map((s) => (STEP_WORDS[s.step] || ((x) => x.step))(s));
  $('#dc-heard').innerHTML = r.heard
    ? `the recogniser heard: <i>${escHtml(r.heard)}</i>`
      + (steps.length ? ` · ${steps.map(escHtml).join(' · ')}` : '')
    : escHtml(r.note || '');
  $('#dc-result-meta').textContent = `${r.seconds} s · ${fmt.int(r.ms)} ms (${r.realtime}x real time)`
    + (r.decoder ? ` · ${r.decoder}` : '') + (r.dictionary ? ` with ${r.dictionary} words of yours` : '')
    + (r.punctuator ? ` · punctuation: ${r.punctuator === 'tagger' ? 'the tagger' : 'rules only (no tagger trained yet)'}` : '');
}

async function sendDictation(samples, rate, label) {
  dc.busy = true;
  goStatus(`writing ${(samples.length / rate).toFixed(1)} s of ${label}… (the first time loads the models)`);
  try {
    const r = await post('/api/dictate/dictate', { pcm: toPcm16(samples), sample_rate: rate });
    showResult(r);
    goStatus('');
    await loadPersonal();
  } catch (e) {
    goStatus(e.message, 'warn');
  } finally {
    dc.busy = false;
  }
}

async function dictateFile(file) {
  if (!file) return;
  try {
    const d = await decodeFile(file, dc.maxDictate);
    if (d.cut) goStatus(`sending the first ${dc.maxDictate} s`, 'warn');
    await sendDictation(d.samples, d.rate, file.name);
  } catch (e) {
    goStatus(`could not decode ${file.name}: ${e.message}`, 'warn');
  }
}

/* Cleanup alone, on typed words: the same `clean` the pipeline runs after the ear. */
async function cleanTyped() {
  const text = $('#dc-clean-in').value.trim();
  if (!text) return;
  try {
    const r = await post('/api/dictate/clean', { text });
    const out = $('#dc-clean-out');
    out.hidden = false;
    out.textContent = r.text || '(nothing left — every word was a filler or a command)';
    const steps = (r.steps || []).map((x) => (STEP_WORDS[x.step] || ((y) => y.step))(x));
    $('#dc-clean-steps').textContent = `punctuation: ${r.punctuator === 'tagger' ? 'the tagger' : 'rules only (no tagger trained yet)'}`
      + (steps.length ? ` · ${steps.join(' · ')}` : '');
  } catch (e) {
    $('#dc-clean-steps').textContent = e.message;
  }
}

async function teach() {
  const r = dc.last;
  const corrected = $('#dc-final').value;
  if (!r || !r.text || corrected === r.text) return;
  try {
    const res = await post('/api/dictate/correct', { shown: r.text, corrected, heard: r.heard });
    const L = res.learned;
    const parts = [];
    if (L.words.length) parts.push(`dictionary: ${L.words.join(', ')}`);
    if (L.spellings.length) parts.push(`spelling: ${L.spellings.join(', ')}`);
    for (const x of L.replacements) {
      parts.push(`“${x.from}” → “${x.to}” ${x.active ? '(now automatic)' : `(seen ${x.count}×; automatic after 2)`}`);
    }
    for (const x of L.ignored) parts.push(`not learned: “${x.from}” → “${x.to}” — ${x.why}`);
    $('#dc-teach-note').textContent = parts.length ? `learned — ${parts.join(' · ')}` : 'nothing to learn from that edit';
    dc.last = { ...r, text: corrected };
    $('#dc-teach').disabled = true;
    await loadPersonal();
  } catch (e) {
    $('#dc-teach-note').textContent = e.message;
  }
}

function renderHistory(rows) {
  $('#dc-history').innerHTML = rows.length
    ? rows.map((r, i) => `<div class="dc-hist"><span class="dim">${escHtml(r.time.slice(5, 16))} · ${r.seconds}s</span> `
      + `${r.text ? escHtml(r.text) : `<span class="dim">(${escHtml(r.note || 'nothing')})</span>`} `
      + (r.text ? `<button type="button" class="dc-fix" data-i="${i}">fix</button>` : '') + '</div>').join('')
    : '<p class="dim">nothing dictated yet</p>';
  $('#dc-history').querySelectorAll('.dc-fix').forEach((b) => {
    b.onclick = () => {
      showResult(rows[Number(b.dataset.i)]);
      $('#dc-final').focus();
    };
  });
}

async function loadPersonal() {
  try {
    const p = await api('/api/dictate/personal');
    const s = p.summary;
    $('#dc-learned-sum').textContent = `${s.words} word${s.words === 1 ? '' : 's'} in your dictionary, `
      + `${s.spellings} spelling${s.spellings === 1 ? '' : 's'}, ${s.active_replacements} automatic `
      + `replacement${s.active_replacements === 1 ? '' : 's'} (${s.replacements - s.active_replacements} waiting for a second correction) · kept in ${p.dir}/`;
    $('#dc-words').innerHTML = p.words.map((w) => `<span class="dc-chip" title="${escHtml(w.source || '')}, ${w.count || 0}×">`
      + `${escHtml(w.spelling || w.word)}<button type="button" data-w="${escHtml(w.word)}" aria-label="remove ${escHtml(w.word)}">×</button></span>`).join('');
    $('#dc-words').querySelectorAll('button').forEach((b) => {
      b.onclick = async () => {
        try { await post('/api/dictate/personal', { action: 'remove', word: b.dataset.w }); } catch (e) { goStatus(e.message, 'warn'); }
        await loadPersonal();
      };
    });
    $('#dc-repl').innerHTML = p.replacements.length
      ? '<thead><tr><th>heard</th><th>write</th><th>corrections</th><th></th></tr></thead><tbody>'
        + p.replacements.map((r) => `<tr><td>${escHtml(r.from)}</td><td>${escHtml(r.to)}</td><td>${r.count}</td>`
          + `<td>${r.active ? '<span class="ok">automatic</span>' : '<span class="dim">needs 2</span>'}</td></tr>`).join('') + '</tbody>'
      : '';
    $('#dc-corrections').innerHTML = p.corrections.length
      ? p.corrections.map((c) => `<div class="dc-hist"><span class="dim">${escHtml(c.time.slice(5, 16))}</span> `
        + `<s>${escHtml(c.shown)}</s> → ${escHtml(c.corrected)}</div>`).join('')
      : '<p class="dim">no corrections yet</p>';
    renderHistory(p.history || []);
  } catch (e) {
    $('#dc-learned-sum').textContent = e.message;
  }
}

/* ---- your voice as a test set ------------------------------------------------------- */

const mv = { prompts: [], i: 0 };

function mvStatus(text, kind = '') {
  const el = $('#mv-status');
  el.textContent = text;
  el.className = kind === 'warn' ? 'bad' : 'dim';
}

function renderMyVoice(st, keepIndex = false) {
  mv.prompts = st.prompts;
  if (!keepIndex) {
    const first = st.prompts.findIndex((p) => !p.recorded);
    mv.i = first < 0 ? 0 : first;
  }
  $('#mv-progress').innerHTML = `<b>${st.recorded}/${st.total}</b> sentences recorded`
    + ` (${st.seconds} s)` + (st.recorded === st.total ? ' — all done; score it below' : '');
  const p = mv.prompts[mv.i];
  if (p) {
    $('#mv-id').textContent = `${mv.i + 1} of ${mv.prompts.length}${p.recorded ? ` · recorded (${p.seconds} s) — record again to replace it` : ''}`;
    $('#mv-text').textContent = p.text;
  }
  $('#mv-del').disabled = !(p && p.recorded);
  $('#mv-clear').disabled = !st.recorded;
  $('#mv-prev').disabled = mv.i === 0;
  $('#mv-next').disabled = mv.i >= mv.prompts.length - 1;
  $('#mv-score').disabled = !st.recorded;
  const r = (st.results || [])[0];
  $('#mv-score-note').textContent = r
    ? `your voice: WER ${pct(r.wer)} (${escHtml(r.decoder || 'greedy')}, ${r.utts} sentences, ${r.time})`
    : (st.recorded ? `${st.recorded} sentences ready to score` : 'record a few sentences first');
}

async function loadMyVoice(keepIndex = false) {
  try { renderMyVoice(await api('/api/dictate/myvoice'), keepIndex); } catch (e) { mvStatus(e.message, 'warn'); }
}

async function saveMyVoice(samples, rate) {
  const p = mv.prompts[mv.i];
  if (!p) return;
  mvStatus('saving…');
  try {
    const st = await post('/api/dictate/myvoice', { prompt: p.id, pcm: toPcm16(samples), sample_rate: rate });
    mvStatus(`saved ${(samples.length / rate).toFixed(1)} s`);
    if (mv.i < st.prompts.length - 1) mv.i += 1;
    renderMyVoice(st, true);
  } catch (e) {
    mvStatus(e.message, 'warn');
  }
}

/* ---- how noise hurts it --------------------------------------------------------------- */

function renderStream(st) {
  $('#dc-stream-run').disabled = !dc.ckpt;
  $('#dc-stream').innerHTML = st
    ? `a word settles <b>${st.latency_s.median.toFixed(2)} s</b> after it is spoken (median; 90% within `
      + `${st.latency_s.p90.toFixed(2)} s) · <b>${(st.revision_rate * 100).toFixed(1)}%</b> of settled words are `
      + `re-spelt by the final pass · <b>${Math.round(st.tick_ms.mean)} ms</b> per re-reading on the ${escHtml(st.device)} `
      + `<span class="dim">(${st.utts} sentences, ${st.run} step ${fmt.int(st.step)}, ${st.time})</span>`
    : '<span class="dim">not measured yet</span>';
}

function renderRobust(res) {
  const r = res.robust;
  const n = res.noise || { have: [], total: 6 };
  $('#dc-noise-fetch').disabled = n.have.length === n.total;
  $('#dc-noise-fetch').textContent = n.have.length === n.total ? 'Noise downloaded ✓' : 'Download the noise (0.65 GB)';
  $('#dc-robust-run').disabled = !n.have.length || !dc.ckpt;
  if (!r) {
    $('#dc-robust').innerHTML = '<tbody><tr><td class="dim">no noise test yet</td></tr></tbody>';
    $('#dc-robust-note').textContent = '';
    return;
  }
  // The result's own order (quiet room first). Object.keys would sort these numeric keys
  // ascending and flip the table relative to the terminal's.
  const snrs = (r.snrs || []).map((x) => String(x)).filter((k) => k in r.mean_by_snr);
  const cell = (v) => {
    // Shade by how much worse than clean: the eye should go straight to the bad corner.
    const rel = Math.min(1, Math.max(0, (v - r.clean) / 0.5));
    return `<td style="background: color-mix(in srgb, var(--critical) ${Math.round(rel * 45)}%, transparent)">${pct(v)}</td>`;
  };
  $('#dc-robust').innerHTML = `<thead><tr><th>place</th>${snrs.map((s) => `<th>${s} dB</th>`).join('')}</tr></thead><tbody>`
    + Object.entries(r.table).map(([env, row]) => `<tr><td>${escHtml(r.environments[env] || env)}</td>`
      + snrs.map((s) => cell(row[s])).join('') + '</tr>').join('')
    + `<tr><td><b>mean</b></td>${snrs.map((s) => cell(r.mean_by_snr[s])).join('')}</tr></tbody>`;
  $('#dc-robust-note').textContent = `clean ${pct(r.clean)} · ${r.run} step ${fmt.int(r.step)}, ${r.utts} sentences of ${r.corpus.split('/').pop()}, ${r.decoder} · ${r.time}`;
}

/* ---- the desktop: daemon, shortcut, tools ------------------------------------------- */

function renderDesktop(d) {
  const st = d.running ? `running (pid ${d.pid}) — ${d.state === 'loading' ? 'loading the models…' : d.state}` : 'not running';
  $('#dc-daemon').innerHTML = `<b class="${d.running ? 'ok' : ''}">${escHtml(st)}</b>`
    + (d.last && d.last.text ? `<br><span class="dim">last: “${escHtml(d.last.text.slice(0, 120))}” (${escHtml(d.last.delivered || '')})</span>` : '');
  $('#dc-daemon-start').disabled = d.running;
  $('#dc-daemon-stop').disabled = !d.running;
  $('#dc-shortcut').innerHTML = d.shortcut
    ? `<b class="ok">${escHtml(d.shortcut.binding)}</b> starts and finishes a dictation`
    : (d.session && d.session !== 'x11'
      ? `<b class="bad">this is a ${escHtml(d.session)} session</b> — typing into other apps needs X11`
      : 'not installed');
  $('#dc-sc-remove').disabled = !d.shortcut;
  $('#dc-tools').innerHTML = Object.entries(d.tools).map(([name, t]) => `<div>${t.found ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>'} `
    + `<code>${escHtml(name)}</code> <span class="dim">— ${escHtml(t.for)}</span>`
    + (t.found ? '' : `<pre class="au-cmd">${escHtml(t.install)}</pre>`) + '</div>').join('');
}

async function loadDesktop() {
  try { renderDesktop(await api('/api/dictate/desktop')); } catch (e) { $('#dc-daemon').textContent = e.message; }
}

async function desktop(action) {
  try {
    renderDesktop(await post('/api/dictate/desktop', { action, binding: $('#dc-binding').value }));
  } catch (e) {
    $('#dc-daemon').textContent = e.message;
  }
  // A starting daemon loads for a few seconds; look again so "loading" turns into "idle".
  if (action === 'start') setTimeout(loadDesktop, 4000);
}

async function runSilence() {
  if (!dc.ckpt) return;
  const cell = $('#dc-check-1');
  cell.textContent = 'running…';
  try {
    const r = await post('/api/dictate/silence', { checkpoint: dc.ckpt });
    cell.innerHTML = r.chars === 0
      ? '<b class="ok">0 characters ✓</b>'
      : `<b class="bad">${r.chars} characters ✗</b>`;
    const box = $('#dc-silence-out');
    box.hidden = false;
    box.innerHTML = Object.entries(r.outputs).map(([name, text]) =>
      `<div><span class="dim">${escHtml(name)}</span> → ${text.trim()
        ? `<b class="bad">“${escHtml(text)}”</b>` : '<span class="ok">nothing</span>'}</div>`).join('');
  } catch (e) {
    cell.textContent = e.message;
  }
}

function pct(v) { return v == null ? '—' : `${(v * 100).toFixed(1)}%`; }

/* The day-two table, from the latest `asr daytwo` result (or, before one exists, from the
 * evaluations: names and speakers are in those too). */
function renderChecks(res) {
  const d = res.daytwo;
  const rows = res.results || [];
  if (d) {
    const sil = d.silence;
    $('#dc-check-1').innerHTML = (sil.chars === 0 ? '<b class="ok">0 characters ✓</b>' : `<b class="bad">${sil.chars} characters ✗</b>`)
      + ' <button id="dc-silence" type="button">Run it again</button>';
    $('#dc-silence').onclick = runSilence;
    const sp = d.speakers;
    $('#dc-check-2').innerHTML = `median ${pct(sp.median)}, <b>worst ${pct(sp.worst)}</b> <span class="dim">(${sp.speakers} speakers)</span>`;
    const n = d.names;
    $('#dc-check-4').innerHTML = `${Math.round(n.without.recall * 100)}% → <b>${Math.round(n.with_dictionary.recall * 100)}%</b> `
      + `of ${n.words} unseen words <span class="dim">(${n.with_dictionary.false_alarms} written where not said)</span>`;
    const c = d.corrections;
    $('#dc-check-5').innerHTML = c.curve.length
      ? `${c.curve.map((x) => `#${x.occurrence} <b>${Math.round(x.recall * 100)}%</b>`).join(' → ')} `
        + `<span class="dim">(${c.targets} words; ${c.false_insertions} false insertions in ${c.false_insertion_utts} utterances)</span>`
      : '<span class="dim">no word repeats often enough in this corpus</span>';
    const cl = d.cleanup;
    $('#dc-check-6').innerHTML = (cl.invented === 0 && cl.dropped === 0 ? '<b class="ok">0 ✓</b>' : `<b class="bad">${cl.invented} invented, ${cl.dropped} dropped ✗</b>`)
      + ` <span class="dim">over ${fmt.int(cl.words)} words (${escHtml(cl.punctuator)})</span>`;
    $('#dc-daytwo-note').textContent = `${d.run} step ${fmt.int(d.step)} on ${d.corpus}, ${d.time}`;
    renderCurve(c);
  } else {
    const r = rows.find((x) => x.names && x.dictionary_size) || rows.find((x) => x.names);
    if (r) {
      $('#dc-check-4').innerHTML = `${Math.round(r.names.recall * 100)}% of ${fmt.int(r.names.said)} unseen words `
        + `<span class="dim">(${escHtml(r.decoder_desc || '')})</span>`;
    }
    const latest = rows.find((x) => x.speakers > 1);
    if (latest) {
      $('#dc-check-2').innerHTML = `median ${pct(latest.median_speaker)}, worst ${pct(latest.worst[0]?.wer)} `
        + `<span class="dim">(${escHtml(latest.run)} on ${escHtml(latest.corpus)})</span>`;
    }
    $('#dc-daytwo-note').textContent = 'no day-two run yet — Run all the checks (about a minute)';
  }
  const pe = res.punct;
  if (pe) {
    const t = pe.tagger;
    const b = pe.rules_only;
    $('#dc-check-7').innerHTML = ['comma', 'period', 'question', 'capital'].map((k) =>
      `${{ comma: ',', period: '.', question: '?', capital: 'Aa' }[k]} <b>${Math.round(t[k].f1 * 100)}</b>`
      + `<span class="dim">/${Math.round(b[k].f1 * 100)}</span>`).join(' · ')
      + ` <span class="dim">F1, tagger/rules, step ${fmt.int(pe.step)}</span>`;
  } else {
    $('#dc-check-7').innerHTML = res.pipeline?.punctuator
      ? '<span class="dim">not scored yet — Score the punctuation</span>'
      : '<span class="dim">no tagger trained yet: <code>scripts/experiment.sh punct</code></span>';
  }
  $('#dc-punct-eval').disabled = !res.pipeline?.punctuator;
}

/* Check 5 as a picture: recall on the 1st, 2nd, 3rd… time a word is said, each after the
 * previous ones were corrected. A system that learns rises; one that only says so is flat. */
function renderCurve(c) {
  const box = $('#dc-curve');
  if (!c.curve.length) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = '<div class="dim">how often an unseen word comes out right, by how many times it has been corrected before</div>'
    + '<div class="dc-bars">' + c.curve.map((x) => `<div class="dc-bar"><div class="dc-bar-fill" style="height:${Math.max(2, x.recall * 100)}%"></div>`
      + `<b>${Math.round(x.recall * 100)}%</b><span class="dim">${x.occurrence === 1 ? '1st, never corrected' : `${x.occurrence}${['', 'st', 'nd', 'rd'][x.occurrence] || 'th'}`}</span></div>`).join('') + '</div>';
}

function renderResults(rows) {
  $('#dc-results').innerHTML = rows.length
    ? '<thead><tr><th>run</th><th>step</th><th>corpus</th><th>WER</th><th>CER</th>'
      + '<th>speakers</th><th>median speaker</th><th>worst speaker</th><th>silence</th><th>decoder</th></tr></thead><tbody>'
      + rows.map((r) => {
        const w = r.worst[0];
        return `<tr><td>${escHtml(r.run)}</td><td>${fmt.int(r.step)}</td><td>${escHtml(r.corpus)}</td>`
          + `<td><b>${pct(r.wer)}</b> <span class="dim">± ${pct(r.wer_pm)}</span></td><td>${pct(r.cer)}</td>`
          + `<td>${r.speakers}</td><td>${pct(r.median_speaker)}</td>`
          + `<td>${w ? `${pct(w.wer)} <span class="dim">(${escHtml(w.speaker)})</span>` : '—'}</td>`
          + `<td>${r.silence_chars == null ? '—' : (r.silence_chars === 0 ? '<span class="ok">0</span>' : `<b class="bad">${r.silence_chars}</b>`)}</td>`
          + `<td class="dim">${escHtml(r.decoder || 'greedy')}</td></tr>`;
      }).join('') + '</tbody>'
    : '<tbody><tr><td>no evaluations yet — <code>python -m aksharallm.asr eval &lt;run&gt; --corpus data/asr/test-clean</code></td></tr></tbody>';
}

function renderRuns(rows) {
  $('#dc-runs').innerHTML = rows.length
    ? '<thead><tr><th>run</th><th>state</th><th>step</th><th>val WER</th><th>silence</th></tr></thead><tbody>'
      + rows.map((r) => `<tr><td>${escHtml(r.name)}</td><td>${r.training ? '<b>training</b>' : 'idle'}</td>`
        + `<td>${r.step == null ? '—' : fmt.int(r.step)}</td><td>${pct(r.val_wer)}</td>`
        + `<td>${r.silence_chars == null ? '—' : r.silence_chars}</td></tr>`).join('') + '</tbody>'
    : '<tbody><tr><td>no recogniser configs</td></tr></tbody>';
}

async function load() {
  const res = await api('/api/dictate');
  const cks = res.checkpoints || [];
  dc.maxSeconds = res.max_seconds || 30;
  $('#dc-empty').hidden = cks.length > 0;
  $('#dc-ckpt').innerHTML = cks.map((c) =>
    `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — step ${fmt.int(c.step)}`
    + `${c.best_wer == null ? '' : ` · val WER ${pct(c.best_wer)}`}</option>`).join('');
  if (cks.length) {
    if (!cks.some((c) => c.rel === dc.ckpt)) dc.ckpt = cks[0].rel;
    $('#dc-ckpt').value = dc.ckpt;
  }
  $('#dc-rec').disabled = !cks.length;
  const beam = $('#dc-decoder').querySelector('option[value="beam"]');
  beam.disabled = !res.lm;
  if (!res.lm) $('#dc-decoder').value = 'greedy';
  $('#dc-lm-note').textContent = !res.lm
    ? 'beam + word LM is off: build the LM first — python -m aksharallm.asr lm build'
    : ((t) => (t
      ? `beam uses alpha ${t.alpha}, beta ${t.beta}, unk ${t.unk_penalty} — `
        + `chosen for ${t.run} on dev-clean (${t.file}), where it took WER from `
        + `${pct(t.greedy_wer)} to ${pct(t.wer)}`
      : `beam uses untuned defaults for this recogniser — tune it: python -m aksharallm.asr tune <run>`))(
      (res.tuned || {})[(dc.ckpt || '').split('/').slice(-2, -1)[0]]);
  $('#dc-silence').disabled = !cks.length;
  renderResults(res.results || []);
  renderChecks(res);
  renderRobust(res);
  renderStream(res.stream);
  renderRuns(res.runs || []);
  dc.maxDictate = res.max_dictate_seconds || 120;
  const pi = res.pipeline || {};
  $('#dc-go').disabled = !pi.recognizer;
  $('#dc-pipe').textContent = pi.recognizer
    ? `uses ${pi.recognizer} · ${pi.lm ? 'beam search + word LM' : 'greedy (no word LM built)'} · `
      + `${pi.punctuator ? `punctuation by the tagger (${pi.punctuator})` : `punctuation by rules only — train the tagger: scripts/experiment.sh punct`}`
      + ` · up to ${dc.maxDictate} s · settings: configs/portal.yaml → dictate:`
    : `no recogniser at ${pi.recognizer_name} — see the lab below`;
  if (!dc.busy) status(cks.length ? `runs on the ${res.device} — ${res.device_reason}` : '');
}

/* ---- jobs: the CLI, from the browser ------------------------------------------------- */

let jobTimer = null;

function renderJobs(j) {
  const sel = $('#dc-split');
  const keep = sel.value;
  sel.innerHTML = j.splits.map((s) => `<option value="${escHtml(s.split)}">${escHtml(s.split)} — `
    + `${s.gb} GB${s.packed ? ' · packed' : s.downloaded ? ' · downloaded' : ''}</option>`).join('');
  if (keep) sel.value = keep;
  const cur = j.splits.find((s) => s.split === sel.value) || j.splits[0];
  $('#dc-split-note').textContent = cur
    ? (cur.packed ? `${cur.split} is ready to train or evaluate on.`
      : cur.downloaded ? `${cur.split} is downloaded; pack it next.` : `${cur.split} is not downloaded.`)
    : '';
  $('#dc-pack').disabled = !(cur && cur.downloaded);
  const lm = j.lm;
  $('#dc-lmfetch').disabled = j.lm_text;
  $('#dc-lmfetch').textContent = j.lm_text ? 'LM text downloaded ✓' : 'Download its text (1.5 GB)';
  $('#dc-lm').disabled = !j.lm_text;
  $('#dc-lm-info').textContent = lm
    ? `built ${lm.built}: ${fmt.int(lm.vocab)} words of vocabulary from ${fmt.compact(lm.words)} words of text`
      + (lm.perplexity?.['dev-clean'] ? `, perplexity ${lm.perplexity['dev-clean'].perplexity.toFixed(0)} on dev-clean` : '')
      + (lm.overlap?.['test-clean'] ? `, ${(lm.overlap['test-clean'].rate * 100).toFixed(2)}% of test sentences verbatim in its text` : '')
    : 'not built yet';
  const ec = $('#dc-eval-corpus');
  const keepC = ec.value;
  ec.innerHTML = j.corpora.map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — ${c.hours} h</option>`).join('');
  if (keepC) ec.value = keepC; else if (j.corpora.some((c) => c.rel.endsWith('test-clean'))) ec.value = 'data/asr/test-clean';
  $('#dc-eval-decoder').querySelector('option[value="beam"]').disabled = !lm;

  const box = $('#dc-jobbox');
  const c = j.current;
  box.hidden = !c;
  if (c) {
    const st = j.running ? 'running' : c.state;
    $('#dc-job-state').textContent = st;
    $('#dc-job-state').className = st === 'done' ? 'ok' : (st === 'failed' || st === 'lost') ? 'bad' : '';
    $('#dc-job-label').textContent = ` ${c.label} · started ${fmt.ago(c.started)}`
      + (c.rc != null && c.rc !== 0 ? ` · exit code ${c.rc}` : '');
    $('#dc-job-cmd').textContent = c.command;
    $('#dc-job-log').textContent = (j.log || []).join('\n');
    $('#dc-job-stop').hidden = !j.running;
  }
  for (const id of ['#dc-fetch', '#dc-pack', '#dc-lmfetch', '#dc-lm', '#dc-tune', '#dc-eval', '#dc-daytwo', '#dc-punct-eval', '#dc-noise-fetch', '#dc-robust-run', '#mv-score', '#dc-stream-run']) {
    if (j.running) $(id).disabled = true;
  }
  if (!j.running) {
    $('#dc-fetch').disabled = !!(cur && cur.downloaded);
    $('#dc-tune').disabled = !lm || !dc.ckpt;
    $('#dc-eval').disabled = !dc.ckpt;
    $('#dc-daytwo').disabled = !lm || !dc.ckpt;
  }
}

async function pollJobs(again = true) {
  clearTimeout(jobTimer);
  try {
    const j = await api('/api/dictate/jobs');
    const wasRunning = dc.jobRunning;
    dc.jobRunning = j.running;
    renderJobs(j);
    // A job that just finished may have written a result; refresh the tables once.
    if (wasRunning && !j.running) { await load(); await loadMyVoice(true); }
    if (again) jobTimer = setTimeout(pollJobs, j.running ? 2000 : 10000);
  } catch (e) {
    status(`jobs: ${e.message}`, 'warn');
    if (again) jobTimer = setTimeout(pollJobs, 10000);
  }
}

async function startJob(spec) {
  try {
    await post('/api/dictate/job', spec);
    await pollJobs();
  } catch (e) {
    status(e.message, 'warn');
  }
}

function wireJobs() {
  $('#dc-split').onchange = () => pollJobs(false);
  $('#dc-fetch').onclick = () => startJob({ kind: 'fetch', split: $('#dc-split').value });
  $('#dc-pack').onclick = () => startJob({ kind: 'pack', split: $('#dc-split').value });
  $('#dc-lmfetch').onclick = () => startJob({ kind: 'lm_fetch' });
  $('#dc-lm').onclick = () => startJob({ kind: 'lm', every: Number($('#dc-every').value), overlap: $('#dc-overlap').checked });
  $('#dc-tune').onclick = () => startJob({ kind: 'tune', checkpoint: dc.ckpt, limit: Number($('#dc-tune-n').value) || 800 });
  $('#dc-eval').onclick = () => startJob({
    kind: 'eval', checkpoint: dc.ckpt, corpus: $('#dc-eval-corpus').value,
    decoder: $('#dc-eval-decoder').value,
    personal: $('#dc-eval-dict').checked,
  });
  $('#dc-daytwo').onclick = () => startJob({ kind: 'daytwo', checkpoint: dc.ckpt, corpus: 'data/asr/test-clean' });
  $('#dc-punct-eval').onclick = () => startJob({ kind: 'punct_eval', passages: 300 });
  $('#dc-noise-fetch').onclick = () => startJob({ kind: 'noise_fetch' });
  $('#dc-stream-run').onclick = () => startJob({ kind: 'stream_eval', checkpoint: dc.ckpt, limit: 200 });
  $('#dc-robust-run').onclick = () => startJob({ kind: 'robust', checkpoint: dc.ckpt, corpus: 'data/asr/test-clean', limit: 400 });
  $('#mv-score').onclick = () => startJob({ kind: 'eval', checkpoint: dc.ckpt, corpus: 'data/asr/my-voice', decoder: 'beam', personal: true });
  $('#dc-job-stop').onclick = async () => {
    try { await post('/api/dictate/stop', {}); } catch (e) { status(e.message, 'warn'); }
    await pollJobs();
  };
}

registerTab('dictate', {
  async open() {
    // The tuned weights shown depend on which recogniser is picked: redraw on change.
    $('#dc-ckpt').onchange = (e) => { dc.ckpt = e.target.value; load(); };
    const btn = $('#dc-rec');
    // Hold-to-talk with the mouse, a finger, or the space bar.
    btn.onpointerdown = (e) => { e.preventDefault(); startRecording('lab'); };
    btn.onpointerup = stopRecording;
    btn.onpointerleave = () => { if (dc.rec) stopRecording(); };
    btn.onkeydown = (e) => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); startRecording('lab'); } };
    btn.onkeyup = (e) => { if (e.key === ' ' || e.key === 'Enter') stopRecording(); };
    $('#dc-file').onchange = (e) => fromFile(e.target.files[0]);
    $('#dc-silence').onclick = runSilence;
    // Dictation: click to start, click to finish -- the same toggle as the desktop shortcut.
    $('#dc-go').onclick = () => (dc.rec ? stopRecording() : startRecording('dictate'));
    $('#dc-go-file').onchange = (e) => dictateFile(e.target.files[0]);
    $('#dc-final').oninput = () => {
      $('#dc-teach').disabled = !dc.last || !dc.last.text || $('#dc-final').value === dc.last.text;
    };
    $('#dc-teach').onclick = teach;
    $('#dc-clean').onclick = cleanTyped;
    $('#dc-clean-in').onkeydown = (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) cleanTyped(); };
    $('#dc-copy').onclick = async () => {
      try { await navigator.clipboard.writeText($('#dc-final').value); goStatus('copied'); } catch { $('#dc-final').select(); goStatus('select-all is done; press Ctrl+C', 'warn'); }
    };
    $('#dc-add').onclick = async () => {
      const w = $('#dc-add-word').value.trim();
      if (!w) return;
      try { await post('/api/dictate/personal', { action: 'add', word: w }); $('#dc-add-word').value = ''; } catch (e) { goStatus(e.message, 'warn'); }
      await loadPersonal();
    };
    $('#mv-rec').onclick = () => (dc.rec ? stopRecording() : startRecording('myvoice'));
    $('#mv-del').onclick = async () => {
      const p = mv.prompts[mv.i];
      if (!p) return;
      try { renderMyVoice(await post('/api/dictate/myvoice-delete', { prompt: p.id }), true); mvStatus('deleted'); } catch (e) { mvStatus(e.message, 'warn'); }
    };
    $('#mv-clear').onclick = async () => {
      if (!window.confirm('Delete every recording of your voice? This cannot be undone.')) return;
      try { renderMyVoice(await post('/api/dictate/myvoice-delete', { all: true })); mvStatus('all deleted'); } catch (e) { mvStatus(e.message, 'warn'); }
    };
    $('#mv-prev').onclick = () => { mv.i = Math.max(0, mv.i - 1); loadMyVoice(true); };
    $('#mv-next').onclick = () => { mv.i = Math.min(mv.prompts.length - 1, mv.i + 1); loadMyVoice(true); };
    $('#dc-daemon-start').onclick = () => desktop('start');
    $('#dc-daemon-stop').onclick = () => desktop('stop');
    $('#dc-sc-install').onclick = () => desktop('install');
    $('#dc-sc-remove').onclick = () => desktop('uninstall');
    wireJobs();
    await load();
    await Promise.all([loadPersonal(), loadDesktop(), loadMyVoice()]);
    await pollJobs();
  },
  /* Stop a recording the moment the tab is left, so the microphone is never held open, and
   * stop polling the job runner — a job keeps running; only this page stops asking. */
  leave() { if (dc.rec) stopRecording(); clearTimeout(jobTimer); },
});

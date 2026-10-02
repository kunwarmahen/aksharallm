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

const dc = { ckpt: null, busy: false, rec: null, maxSeconds: 30 };

/* The dictionary is personal and typed by hand, so it is kept in this browser between visits
 * (a convenience; nothing depends on it surviving). */
function loadDict() { try { return localStorage.getItem('dc-dict') || ''; } catch { return ''; } }
function saveDict(v) { try { localStorage.setItem('dc-dict', v); } catch { /* private mode */ } }

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
      decoder: $('#dc-decoder').value, dictionary: $('#dc-dict').value,
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

async function startRecording() {
  if (dc.rec || dc.busy) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    status('this browser will not open the microphone here (it needs localhost or https) — use a file', 'warn');
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
    status(`no microphone: ${e.message}`, 'warn');
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
  dc.rec = { ctx, stream, proc, chunks, t0 };
  $('#dc-rec').classList.add('dc-live');
  $('#dc-rec').textContent = '■ Listening — release to send';
  status('listening…');
  dc.rec.timer = setTimeout(stopRecording, dc.maxSeconds * 1000);
}

async function stopRecording() {
  const r = dc.rec;
  if (!r) return;
  dc.rec = null;
  clearTimeout(r.timer);
  r.proc.disconnect();
  r.stream.getTracks().forEach((t) => t.stop());
  const rate = r.ctx.sampleRate;
  await r.ctx.close();
  $('#dc-rec').classList.remove('dc-live');
  $('#dc-rec').textContent = '● Hold to talk';
  const n = r.chunks.reduce((a, c) => a + c.length, 0);
  const all = new Float32Array(n);
  let at = 0;
  for (const c of r.chunks) { all.set(c, at); at += c.length; }
  if (n / rate < 0.3) { status('too short — hold the button while you speak', 'warn'); return; }
  await send(all, rate, 'your voice');
}

async function fromFile(file) {
  if (!file) return;
  const ctx = new AudioContext();
  try {
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    // Downmix to mono by averaging channels — the same thing `audio/io.to_mono` does.
    const mono = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c += 1) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < ch.length; i += 1) mono[i] += ch[i] / buf.numberOfChannels;
    }
    if (buf.duration > dc.maxSeconds) {
      status(`${buf.duration.toFixed(0)} s is past the ${dc.maxSeconds} s this tab takes — sending the first ${dc.maxSeconds}`, 'warn');
    }
    await send(mono.subarray(0, Math.floor(dc.maxSeconds * buf.sampleRate)), buf.sampleRate, file.name);
  } catch (e) {
    status(`could not decode ${file.name}: ${e.message}`, 'warn');
  } finally {
    await ctx.close();
  }
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

function renderNames(rows) {
  // Prefer the run that used a dictionary: check 4 is "with and without", and the with is
  // the claim. Fall back to any beam result.
  const r = rows.find((x) => x.names && x.dictionary_size) || rows.find((x) => x.names);
  if (!r) return;
  const n = r.names;
  $('#dc-check-4').innerHTML = `${Math.round(n.recall * 100)}% of ${fmt.int(n.said)} unseen words `
    + `<span class="dim">(${escHtml(r.decoder_desc || '')}; ${n.false_alarms} written where not said)</span>`;
}

function renderResults(rows) {
  renderNames(rows);
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
  const latest = rows.find((r) => r.speakers > 1);
  if (latest) {
    $('#dc-check-2').innerHTML = `median ${pct(latest.median_speaker)}, worst ${pct(latest.worst[0]?.wer)} `
      + `<span class="dim">(${escHtml(latest.run)} on ${escHtml(latest.corpus)})</span>`;
  }
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
    : (res.tuned
      ? `beam uses alpha ${res.tuned.alpha}, beta ${res.tuned.beta}, unk ${res.tuned.unk_penalty} — `
        + `chosen on dev-clean (${res.tuned.file}), where it took WER from `
        + `${pct(res.tuned.greedy_wer)} to ${pct(res.tuned.wer)}`
      : 'beam uses untuned defaults — run python -m aksharallm.asr tune <run> to choose them on dev-clean');
  $('#dc-silence').disabled = !cks.length;
  renderResults(res.results || []);
  renderRuns(res.runs || []);
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
  for (const id of ['#dc-fetch', '#dc-pack', '#dc-lmfetch', '#dc-lm', '#dc-tune', '#dc-eval']) {
    if (j.running) $(id).disabled = true;
  }
  if (!j.running) {
    $('#dc-fetch').disabled = !!(cur && cur.downloaded);
    $('#dc-tune').disabled = !lm || !dc.ckpt;
    $('#dc-eval').disabled = !dc.ckpt;
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
    if (wasRunning && !j.running) await load();
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
    dictionary: $('#dc-eval-dict').checked ? $('#dc-dict').value : '',
  });
  $('#dc-job-stop').onclick = async () => {
    try { await post('/api/dictate/stop', {}); } catch (e) { status(e.message, 'warn'); }
    await pollJobs();
  };
}

registerTab('dictate', {
  async open() {
    $('#dc-ckpt').onchange = (e) => { dc.ckpt = e.target.value; };
    const btn = $('#dc-rec');
    // Hold-to-talk with the mouse, a finger, or the space bar.
    btn.onpointerdown = (e) => { e.preventDefault(); startRecording(); };
    btn.onpointerup = stopRecording;
    btn.onpointerleave = () => { if (dc.rec) stopRecording(); };
    btn.onkeydown = (e) => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); startRecording(); } };
    btn.onkeyup = (e) => { if (e.key === ' ' || e.key === 'Enter') stopRecording(); };
    $('#dc-file').onchange = (e) => fromFile(e.target.files[0]);
    $('#dc-dict').value = loadDict();
    $('#dc-dict').oninput = (e) => saveDict(e.target.value);
    $('#dc-silence').onclick = runSilence;
    wireJobs();
    await load();
    await pollJobs();
  },
  /* Stop a recording the moment the tab is left, so the microphone is never held open, and
   * stop polling the job runner — a job keeps running; only this page stops asking. */
  leave() { if (dc.rec) stopRecording(); clearTimeout(jobTimer); },
});

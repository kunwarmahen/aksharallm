/* The Audio tab's TTS panel: every saved TTS report, two of them compared sentence by
 * sentence, the clips to listen to, and Speak / Run-a-report as CLI jobs (portal/tts.py).
 *
 * A comparison counts SENTENCES (fewer errors, closer voice) and gives a sign test, never a
 * difference of two WERs: 40 sentences make totals noisy, and the counts are what the docs
 * quote.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { jobBox, pairedLine } from './clijob.js';

const tt = { reports: [], ckpts: [], job: null, sentences: [] };

const pct = (v) => (v == null ? '–' : fmt.pct(v));

function reportsTable() {
  $('#tts-reports').innerHTML = tt.reports.length
    ? '<thead><tr><th>report</th><th>speakers</th><th>voice given as</th><th>n</th>'
      + '<th>original WER</th><th>codec WER</th><th>TTS WER</th><th>voice</th><th>closer to own voice</th></tr></thead><tbody>'
      + tt.reports.map((r) => {
        const v = r.voice || {};
        const voice = v.tts_own == null ? '–'
          : `${fmt.num(v.tts_own, 3)} <span class="dim">(codec ${fmt.num(v.codec_own, 2)}, same speaker ${fmt.num(v.same_speaker, 2)})</span>`;
        return `<tr><td>${escHtml(r.name.replace(/^tts-|\.json$/g, ''))}</td><td>${escHtml(r.speakers)}</td>`
          + `<td>${escHtml(r.voicing)}</td><td>${r.n ?? '–'}</td><td>${pct(r.wer.original)}</td>`
          + `<td>${pct(r.wer.codec)}</td><td><b>${pct(r.wer.tts)}</b></td><td>${voice}</td>`
          + `<td>${v.pairs ? `${v.own_closer}/${v.pairs}` : '–'}</td></tr>`;
      }).join('') + '</tbody>'
    : '<tbody><tr><td>no TTS reports yet — Run a TTS report below</td></tr></tbody>';
}

async function compare() {
  const a = $('#tts-a').value; const b = $('#tts-b').value;
  if (!a || !b || a === b) { $('#tts-paired').textContent = 'pick two different reports'; return; }
  try {
    const c = await post('/api/tts/compare', { a, b });
    let html = `${c.n} sentences, b against a. Word error rate ${fmt.pct(c.wer.a)} → ${fmt.pct(c.wer.b)}.<br>`
      + pairedLine(c.wer.b_better, c.wer.b_worse, c.wer.p, 'fewer word errors');
    if (c.voice) {
      html += `<br>Voice cosine ${fmt.num(c.voice.a, 3)} → ${fmt.num(c.voice.b, 3)} (codec ceiling ${fmt.num(c.voice.ceiling, 3)}).<br>`
        + pairedLine(c.voice.b_closer, c.voice.b_further, c.voice.p, 'closer to the speaker');
    }
    $('#tts-paired').innerHTML = html;
    $('#tts-cmp-cmd').hidden = false;
    $('#tts-cmp-cmd').textContent = c.command;
  } catch (e) {
    $('#tts-paired').textContent = e.message;
    $('#tts-cmp-cmd').hidden = true;
  }
}

async function listen(name) {
  if (!name) { $('#tts-sentences').innerHTML = ''; return; }
  const r = await api(`/api/tts/report?name=${encodeURIComponent(name)}`);
  tt.sentences = r.sentences;
  $('#tts-sentences').innerHTML = '<thead><tr><th>#</th><th>the sentence</th><th>the recogniser heard (TTS)</th>'
    + '<th>speaker</th><th>voice</th><th></th></tr></thead><tbody>'
    + r.sentences.map((s) => `<tr><td>${s.index}</td><td>${escHtml(s.ref)}</td><td>${escHtml(s.heard)}</td>`
      + `<td>${escHtml(s.speaker ?? '')}</td><td>${s.voice == null ? '' : fmt.num(s.voice, 2)}</td>`
      + `<td class="tts-clips" data-i="${s.index}">${r.wavs ? `<button type="button" data-i="${s.index}">listen</button>` : 'no WAVs'}</td></tr>`).join('')
    + '</tbody>';
  $('#tts-sentences').querySelectorAll('button[data-i]').forEach((btn) => {
    btn.onclick = async () => {
      const i = Number(btn.dataset.i);
      const c = await post('/api/tts/clip', { name, index: i });
      btn.parentElement.innerHTML = ['original', 'codec', 'tts'].filter((k) => c[k])
        .map((k) => `<span>${k}<br><audio controls preload="none" src="${c[k]}"></audio></span>`).join('');
    };
  });
  voices();
}

function voices() {
  const ck = tt.ckpts.find((c) => c.rel === $('#tts-ckpt').value);
  const sel = $('#tts-voice');
  if (!ck || !ck.multi) {
    sel.innerHTML = '<option value="">its single voice</option>';
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  const rep = $('#tts-report').value;
  sel.innerHTML = '<option value="spk:LJ">LJ (embedding)</option>'
    + tt.sentences.map((s) => `<option value="clone:${s.index}">clone sentence ${s.index}${s.speaker ? ` (speaker ${escHtml(s.speaker)})` : ''} of the report above</option>`).join('');
  sel.dataset.report = rep;
}

function spokenList(rows) {
  $('#tts-spoken').innerHTML = rows.map((r) => `<div><button type="button" data-n="${escHtml(r.name)}">play</button> `
    + `<b>${escHtml(r.text || r.name)}</b> <span class="dim">— ${escHtml(r.voice || '')}, ${escHtml(r.checkpoint || '')}, ${fmt.ago(r.when)}</span></div>`).join('');
  $('#tts-spoken').querySelectorAll('button[data-n]').forEach((btn) => {
    btn.onclick = async () => {
      const c = await post('/api/tts/spoken', { name: btn.dataset.n });
      btn.outerHTML = `<audio controls autoplay src="${c.wav}"></audio>`;
    };
  });
}

async function load() {
  const res = await api('/api/tts');
  tt.reports = res.reports || [];
  tt.ckpts = res.checkpoints || [];
  reportsTable();
  const opts = tt.reports.map((r) => `<option value="${escHtml(r.name)}">${escHtml(r.name)}</option>`).join('');
  const keep = { a: $('#tts-a').value, b: $('#tts-b').value, r: $('#tts-report').value };
  $('#tts-a').innerHTML = opts; $('#tts-b').innerHTML = opts; $('#tts-report').innerHTML = opts;
  // Default pair: the newest cloned report against its own control.
  const cloned = tt.reports.find((r) => r.control);
  if (keep.a) { $('#tts-a').value = keep.a; $('#tts-b').value = keep.b; }
  else if (cloned) { $('#tts-a').value = cloned.control; $('#tts-b').value = cloned.name; }
  if (keep.r) $('#tts-report').value = keep.r;
  $('#tts-ckpt').innerHTML = tt.ckpts.map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — step ${fmt.int(c.step)}${c.multi ? ', many voices' : ''}</option>`).join('');
  $('#tts-codes').innerHTML = '<option value="">its own held-out sentences (seen speakers)</option>'
    + (res.codes || []).map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)}</option>`).join('');
  spokenList(res.spoken || []);
  $('#tts-speak').disabled = !tt.ckpts.length;
  $('#tts-run-report').disabled = !tt.ckpts.length;
  tt.job.render(res.jobs);
  if (res.jobs && res.jobs.running) tt.job.poll();
  await listen($('#tts-report').value);
}

function speak() {
  const v = $('#tts-voice').value;
  const spec = { kind: 'speak', checkpoint: $('#tts-ckpt').value, text: $('#tts-text').value };
  if (v.startsWith('spk:')) spec.speaker = v.slice(4);
  if (v.startsWith('clone:')) spec.clone = { report: $('#tts-voice').dataset.report, index: Number(v.slice(6)) };
  tt.job.start(spec);
}

export async function openTts() {
  if (!tt.job) tt.job = jobBox($('#tts-job'), 'tts', { onDone: () => load() });
  $('#tts-compare').onclick = compare;
  $('#tts-report').onchange = (e) => listen(e.target.value).catch((err) => flash(err.message, 'error'));
  $('#tts-ckpt').onchange = voices;
  $('#tts-speak').onclick = speak;
  $('#tts-run-report').onclick = () => tt.job.start({
    kind: 'tts-report', checkpoint: $('#tts-ckpt').value, codes: $('#tts-codes').value,
    voicing: $('#tts-voicing').value, n: Number($('#tts-n').value) || 40,
  });
  await load();
  if (tt.reports.some((r) => r.control)) await compare();
}

export function leaveTts() { if (tt.job) tt.job.stop(); }

/* The Pictures tab: the picture tokenizer's reports (PSNR and the reader's verdict at each
 * codebook count), its gallery, a paired comparison of two reports, and its two CLI jobs
 * (docs/26). G1: draw from a caption (inline, CPU), G1 reports with their per-caption
 * gallery (held-out captions first), and a paired comparison of two G1 reports.
 *
 * The reader's row on the originals comes first in the table: it is the positive control,
 * and a report where it is not 100% means every number under it is wrong.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';
import { jobBox } from './clijob.js';

const pc = { reports: [], genReports: [], job: null };

// The words G1 knows (pictures/codes.py WORDS): three selects make a caption it can read.
const COUNTS = ['one', 'two', 'three', 'four'];
const COLOURS = ['red', 'green', 'blue', 'yellow', 'purple'];
const SHAPES = ['circle', 'square', 'triangle'];

function caption() {
  const n = $('#pc-w-count').value;
  const shape = $('#pc-w-shape').value + (n === 'one' ? '' : 's');
  return `${n} ${$('#pc-w-colour').value} ${shape}`;
}

function pct(x) { return x == null ? '–' : fmt.pct(x, 1); }

async function showReport(name) {
  if (!name) {
    $('#pc-ladder').innerHTML = '<tbody><tr><td>no report yet — train <code>imgtok-shapes</code>, then Make a report below</td></tr></tbody>';
    $('#pc-usage').textContent = '';
    $('#pc-gallery').innerHTML = '';
    return;
  }
  let r;
  try { r = await api(`/api/pictures/report?name=${encodeURIComponent(name)}`); } catch (e) {
    flash(`pictures: ${e.message}`, 'error'); return;
  }
  const o = r.originals || {};
  const held = {};
  ((r.holdout || {}).ladder || []).forEach((h) => { held[h.books] = h.all; });
  $('#pc-ladder').innerHTML = '<thead><tr><th>codebooks</th><th>numbers / picture</th><th>bits</th><th>PSNR</th>'
    + '<th>count</th><th>colour</th><th>shape</th><th>all three</th><th>held-out: all three</th></tr></thead><tbody>'
    + `<tr class="dim"><td>the originals</td><td>–</td><td>${fmt.int(64 * 64 * 24)}</td><td>–</td>`
    + `<td>${pct(o.count)}</td><td>${pct(o.colour)}</td><td>${pct(o.shape)}</td><td><b>${pct(o.all)}</b></td><td>–</td></tr>`
    + r.ladder.map((x) => `<tr><td>${x.books}</td><td>${fmt.int(x.books * r.grid * r.grid)}</td><td>${fmt.int(x.bits)}</td>`
      + `<td>${fmt.num(x.psnr, 1)} dB</td><td>${pct(x.count)}</td><td>${pct(x.colour)}</td><td>${pct(x.shape)}</td>`
      + `<td><b>${pct(x.all)}</b></td><td>${pct(held[x.books])}</td></tr>`).join('')
    + '</tbody>';
  $('#pc-usage').innerHTML = `${fmt.int(r.n)} val pictures, ${r.grid}×${r.grid} grid. Codebook use `
    + '(distinct entries used / effective size; a codebook of 512 with an effective size of 20 '
    + 'is a 20-entry codebook, which the loss curve never shows): '
    + (r.usage || []).map((u) => `#${u.book} <b>${fmt.int(u.used)}</b> / ${fmt.num(u.perplexity, 0)}`).join(' · ');
  $('#pc-gallery').innerHTML = (r.gallery_png || []).map((src, i) =>
    `<img src="${src}" alt="picture ${i + 1}: the original, then its reconstructions" loading="lazy">`).join('')
    || '<p class="panel-note">no gallery in this report</p>';
}

function options(sel, keep) {
  sel.innerHTML = pc.reports.map((r) => `<option value="${escHtml(r.name)}">${escHtml(r.name)} — `
    + `${r.grid}×${r.grid}, all ${pct(r.all)}</option>`).join('');
  if (keep && pc.reports.some((r) => r.name === keep)) sel.value = keep;
}

async function draw() {
  const btn = $('#pc-draw');
  btn.disabled = true;
  $('#pc-draw-note').textContent = `drawing “${caption()}” on the CPU…`;
  try {
    const r = await post('/api/pictures/draw', {
      checkpoint: $('#pc-gen').value, caption: caption(),
      n: Number($('#pc-draw-n').value) || 8, guidance: Number($('#pc-guidance').value),
      seed: Number($('#pc-seed').value) || 0,
    });
    $('#pc-draw-note').innerHTML = `“${escHtml(r.caption)}”: the reader agrees with <b>${r.agree} of `
      + `${r.pictures.length}</b> pictures (${escHtml(r.kind || 'G1')}, step ${fmt.int(r.step)})`
      + (r.held_out ? ' — <b>a held-out combination</b>: no training picture shows it' : '');
    $('#pc-drawn').innerHTML = r.pictures.map((p, i) => `<figure class="${p.ok ? 'ok' : 'bad'}">`
      + `<img src="${p.png}" alt="drawing ${i + 1} of ${escHtml(r.caption)}">`
      + `<figcaption>${p.ok ? '✓' : '✗'} ${p.read.count} ${escHtml(String(p.read.colour))} `
      + `${escHtml(String(p.read.shape))}</figcaption></figure>`).join('');
    $('#pc-draw-cmd').hidden = false;
    $('#pc-draw-cmd').textContent = r.command;
  } catch (e) {
    $('#pc-draw-note').textContent = '';
    flash(e.message, 'error');
  } finally { btn.disabled = !$('#pc-gen').value; }
}

async function showGenReport(name) {
  if (!name) {
    $('#pc-gen-table').innerHTML = '<tbody><tr><td>no G1 report yet — train <code>imggen-shapes</code>, then Make a G1 report below</td></tr></tbody>';
    $('#pc-gen-note').textContent = '';
    $('#pc-gen-gallery').innerHTML = '';
    return;
  }
  let r;
  try { r = await api(`/api/pictures/report?name=${encodeURIComponent(name)}`); } catch (e) {
    flash(`pictures: ${e.message}`, 'error'); return;
  }
  const row = (label, g) => (g && g.n ? `<tr><td>${label}</td><td>${fmt.int(g.n)}</td><td>${pct(g.count)}</td>`
    + `<td>${pct(g.colour)}</td><td>${pct(g.shape)}</td><td><b>${pct(g.all)}</b></td><td>${pct(g.copies)}</td></tr>` : '');
  $('#pc-gen-table').innerHTML = '<thead><tr><th>captions</th><th>pictures</th><th>count</th><th>colour</th>'
    + '<th>shape</th><th>all three</th><th>copies</th></tr></thead><tbody>'
    + row('seen in training (56)', r.seen) + row('held out: purple triangles (4)', r.held_out) + '</tbody>';
  const held = (r.per_caption || []).filter((p) => p.held_out);
  $('#pc-gen-note').innerHTML = `${r.k} pictures per caption, guidance ${fmt.num(r.guidance, 1)}, step `
    + `${fmt.int(r.step)}. Variety <b>${pct(r.variety)}</b>. Held-out captions: `
    + held.map((p) => `${escHtml(p.caption)} <b>${pct(p.all)}</b>`).join(' · ');
  // held-out captions first: they are the pictures the experiment exists for
  const order = (r.per_caption || []).map((p, i) => ({ p, src: (r.gallery_png || [])[i] }))
    .filter((x) => x.src).sort((a, b) => Number(b.p.held_out) - Number(a.p.held_out));
  $('#pc-gen-gallery').innerHTML = order.map(({ p, src }) => `<figure><img src="${src}" loading="lazy" `
    + `alt="${escHtml(p.caption)}"><figcaption>${escHtml(p.caption)}${p.held_out ? ' (held out)' : ''} — `
    + `${pct(p.all)} right, ${p.distinct} distinct</figcaption></figure>`).join('');
}

async function genCompare() {
  try {
    const res = await post('/api/pictures/compare', { a: $('#pc-ga').value, b: $('#pc-gb').value });
    $('#pc-gcmp').innerHTML = '<thead><tr><th>captions</th><th>n</th><th>a: all three</th><th>b: all three</th>'
      + '<th>b right, a wrong</th><th>a right, b wrong</th><th>p</th></tr></thead><tbody>'
      + res.rows.map((r) => `<tr><td>${escHtml(r.set)}</td><td>${fmt.int(r.n)}</td><td>${pct(r.a)}</td>`
        + `<td>${pct(r.b)}</td><td>${r.b_only}</td><td>${r.a_only}</td>`
        + `<td>${r.p == null ? '–' : fmt.num(r.p, 3)}</td></tr>`).join('') + '</tbody>';
    $('#pc-gcmp-cmd').hidden = false;
    $('#pc-gcmp-cmd').textContent = res.command;
  } catch (e) { flash(e.message, 'error'); }
}

function genOptions(sel, keep) {
  sel.innerHTML = pc.genReports.map((r) => `<option value="${escHtml(r.name)}">${escHtml(r.name)} — `
    + `seen ${pct(r.seen_all)}, held out ${pct(r.held_out_all)}</option>`).join('');
  if (keep && pc.genReports.some((r) => r.name === keep)) sel.value = keep;
}

async function load() {
  const res = await api('/api/pictures');
  pc.reports = res.reports || [];
  const cur = $('#pc-report').value;
  options($('#pc-report'), cur);
  options($('#pc-a'), $('#pc-a').value);
  options($('#pc-b'), $('#pc-b').value || (pc.reports[1] || {}).name);
  await showReport($('#pc-report').value || null);
  const cks = res.checkpoints || [];
  $('#pc-ckpt').innerHTML = cks.map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — `
    + `${c.grid}×${c.grid} × ${c.n_codebooks}, step ${fmt.int(c.step)}</option>`).join('');
  $('#pc-run').disabled = !cks.length;
  $('#pc-compare').disabled = pc.reports.length < 2;
  const c = res.corpus || {};
  $('#pc-corpus').innerHTML = c.ready
    ? `${escHtml(c.dir)}: ${fmt.int(c.pictures)} pictures of 1–4 shapes, 5 colours, 3 shapes, `
      + 'each drawn from a description we chose' + (c.holdout ? ', plus the held-out purple triangles' : '')
    : 'the shapes corpus is not made yet: <code>python -m aksharallm.vision corpus</code> (seconds)';
  pc.genReports = res.gen_reports || [];
  genOptions($('#pc-gen-report'), $('#pc-gen-report').value);
  genOptions($('#pc-ga'), $('#pc-ga').value);
  genOptions($('#pc-gb'), $('#pc-gb').value || (pc.genReports[1] || {}).name);
  await showGenReport($('#pc-gen-report').value || null);
  $('#pc-gcompare').disabled = pc.genReports.length < 2;
  const gens = res.generators || [];
  const genOpts = gens.map((g) => `<option value="${escHtml(g.rel)}">${escHtml(g.kind || 'G1')} · `
    + `${escHtml(g.rel)} — step ${fmt.int(g.step)}</option>`).join('');
  for (const id of ['#pc-gen', '#pc-gen-job']) {
    const keep = $(id).value;
    $(id).innerHTML = genOpts;
    if (keep && gens.some((g) => g.rel === keep)) $(id).value = keep;
  }
  $('#pc-draw').disabled = !gens.length;
  $('#pc-gen-run').disabled = !gens.length;
  $('#pc-codes').disabled = !cks.length;
  if (!gens.length) {
    $('#pc-draw-note').innerHTML = 'no G1 model yet: encode the corpus below, then train '
      + '<code>imggen-shapes</code> with the dashboard’s Start';
  }
  pc.job.render(res.jobs);
  if (res.jobs && res.jobs.running) pc.job.poll();
}

async function compare() {
  try {
    const res = await post('/api/pictures/compare', { a: $('#pc-a').value, b: $('#pc-b').value });
    $('#pc-cmp').innerHTML = '<thead><tr><th>pictures</th><th>n</th><th>a: all three</th><th>b: all three</th>'
      + '<th>b right, a wrong</th><th>a right, b wrong</th><th>p</th></tr></thead><tbody>'
      + res.rows.map((r) => `<tr><td>${escHtml(r.set)}</td><td>${fmt.int(r.n)}</td><td>${pct(r.a)}</td>`
        + `<td>${pct(r.b)}</td><td>${r.b_only}</td><td>${r.a_only}</td>`
        + `<td>${r.p == null ? '–' : fmt.num(r.p, 3)}</td></tr>`).join('') + '</tbody>';
    $('#pc-cmp-cmd').hidden = false;
    $('#pc-cmp-cmd').textContent = res.command;
  } catch (e) { flash(e.message, 'error'); }
}

registerTab('pictures', {
  async open() {
    if (!pc.job) pc.job = jobBox($('#pc-job'), 'pictures', { onDone: () => load() });
    $('#pc-report').onchange = (e) => showReport(e.target.value);
    $('#pc-compare').onclick = compare;
    $('#pc-run').onclick = () => pc.job.start({
      kind: 'report', checkpoint: $('#pc-ckpt').value, n: Number($('#pc-n').value) || 256,
    });
    $('#pc-read').onclick = () => pc.job.start({ kind: 'read' });
    $('#pc-codes').onclick = () => pc.job.start({ kind: 'codes', checkpoint: $('#pc-ckpt').value, books: 2 });
    $('#pc-gen-run').onclick = () => pc.job.start({
      kind: 'gen-report', checkpoint: $('#pc-gen-job').value, k: Number($('#pc-k').value) || 8,
      guidance: Number($('#pc-gen-g').value),
    });
    if (!$('#pc-w-count').options.length) {
      const fill = (id, xs, v) => { $(id).innerHTML = xs.map((x) => `<option>${x}</option>`).join(''); $(id).value = v; };
      fill('#pc-w-count', COUNTS, 'two');
      fill('#pc-w-colour', COLOURS, 'red');
      fill('#pc-w-shape', SHAPES, 'circle');
    }
    $('#pc-draw').onclick = draw;
    $('#pc-gen-report').onchange = (e) => showGenReport(e.target.value);
    $('#pc-gcompare').onclick = genCompare;
    await load().catch((e) => flash(`pictures: ${e.message}`, 'error'));
  },
  leave() { if (pc.job) pc.job.stop(); },
});

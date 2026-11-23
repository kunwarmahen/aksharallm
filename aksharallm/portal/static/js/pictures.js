/* The Pictures tab: the picture tokenizer's reports (PSNR and the reader's verdict at each
 * codebook count), its gallery, a paired comparison of two reports, and its two CLI jobs
 * (docs/26). G1: draw from a caption (inline, CPU), G1 reports with their per-caption
 * gallery (held-out captions first), and a paired comparison of two G1 reports. G4b: our own
 * contrastive model beside the reader (clip reports, and the own_clip block of a G1 report),
 * and the scenes corpus, whose captions also say size and place.
 *
 * The reader's row on the originals comes first in the table: it is the positive control,
 * and a report where it is not 100% means every number under it is wrong.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';
import { jobBox } from './clijob.js';

const pc = { reports: [], genReports: [], clipReports: [], job: null };

// The words G1 knows (pictures/codes.py WORDS): three selects make a caption it can read.
const COUNTS = ['one', 'two', 'three', 'four'];
const COLOURS = ['red', 'green', 'blue', 'yellow', 'purple'];
const SHAPES = ['circle', 'square', 'triangle'];

function genKind() {
  const g = (pc.gens || []).find((x) => x.rel === $('#pc-gen').value);
  return g ? g.kind : 'G1';
}

function syncDrawInputs() {
  const free = genKind() === 'G4';
  $('#pc-free-row').hidden = !free;
  for (const id of ['#pc-w-count', '#pc-w-colour', '#pc-w-shape']) $(id).closest('label').hidden = free;
}

function caption() {
  if (genKind() === 'G4') return $('#pc-free').value.trim();
  const n = $('#pc-w-count').value;
  const shape = $('#pc-w-shape').value + (n === 'one' ? '' : 's');
  return `${n} ${$('#pc-w-colour').value} ${shape}`;
}

function pct(x) { return x == null ? '–' : fmt.pct(x, 1); }

// What the reader saw, in the corpus's own words (size and place only when it read a scene).
function said(r, scene) {
  if (!r || !r.count) return 'nothing';
  const where = { left: 'on the left', middle: 'in the middle', right: 'on the right' }[r.where] || r.where;
  return `${r.count} ${scene && r.size ? `${r.size} ` : ''}${r.colour} ${r.shape}`
    + (scene && r.where ? ` ${where}` : '');
}

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
    $('#pc-draw-note').innerHTML = (r.agree == null
      ? `“${escHtml(r.caption)}”: no count, colour and shape found in the sentence, so the `
        + 'pictures are shown with what the reader sees in them, not judged'
      : `“${escHtml(r.caption)}”: the reader agrees with <b>${r.agree} of ${r.pictures.length}</b> pictures`)
      + ` (${escHtml(r.kind || 'G1')}, step ${fmt.int(r.step)})`
      + (r.held_out ? ' — <b>a held-out combination</b>: no training picture shows it' : '');
    $('#pc-drawn').innerHTML = r.pictures.map((p, i) => `<figure class="${p.ok == null ? '' : (p.ok ? 'ok' : 'bad')}">`
      + `<img src="${p.png}" alt="drawing ${i + 1} of ${escHtml(r.caption)}">`
      + `<figcaption>${p.ok == null ? '' : (p.ok ? '✓' : '✗')} `
      + `${escHtml(said(p.read, r.judged_as && 'where' in r.judged_as))}</figcaption></figure>`).join('');
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
  const attrs = r.attributes || ['count', 'colour', 'shape'];
  const row = (label, g) => (g && g.n ? `<tr><td>${label}</td><td>${fmt.int(g.n)}</td>`
    + attrs.map((a) => `<td>${pct(g[a])}</td>`).join('')
    + `<td><b>${pct(g.all)}</b></td><td>${pct(g.copies)}</td></tr>` : '');
  const caps = r.per_caption || [];
  const nSeen = caps.filter((p) => !p.held_out).length;
  const heldNames = [...new Set(caps.filter((p) => p.held_out).map((p) => p.held_out_as || 'purple triangles'))];
  $('#pc-gen-table').innerHTML = '<thead><tr><th>captions</th><th>pictures</th>'
    + attrs.map((a) => `<th>${a}</th>`).join('') + '<th>all</th><th>copies</th></tr></thead><tbody>'
    + row(`seen in training (${nSeen})`, r.seen)
    + row(`held out: ${escHtml(heldNames.join(', '))} (${caps.length - nSeen})`, r.held_out)
    + Object.entries(r.held_out_by || {}).map(([name, g]) => row(`&nbsp;&nbsp;${escHtml(name)}`, g)).join('')
    + '</tbody>';
  const held = caps.filter((p) => p.held_out);
  const oc = r.own_clip;
  const ocLine = (label, g) => (g && g.n ? `${label}: top-1 <b>${pct(g.top1)}</b>, agrees with the reader on `
    + `${pct(g.agreement.agree)} (kappa ${g.agreement.kappa == null ? '–' : fmt.num(g.agreement.kappa, 2)}; `
    + `model only ${g.agreement.model_only}, reader only ${g.agreement.reader_only})` : '');
  const closed = oc && oc.closed ? ` Without the none-fits floor (${fmt.num(oc.floor, 3)}): top-1 ${pct(oc.closed.top1)}, `
    + `agrees ${pct(oc.closed.agreement.agree)}; ${oc.closed.none_fits} pictures judged “none fits”.` : '';
  $('#pc-gen-note').innerHTML = `${r.k} pictures per caption, guidance ${fmt.num(r.guidance, 1)}, step `
    + `${fmt.int(r.step)}${r.phrasing ? `, phrasing <code>${escHtml(r.phrasing)}</code>` : ''}. `
    + `Variety <b>${pct(r.variety)}</b>. `
    + (held.length <= 8 ? 'Held-out captions: '
      + held.map((p) => `${escHtml(p.caption)} <b>${pct(p.all)}</b>`).join(' · ') : '')
    + (oc ? `<br>Our contrastive model (<code>${escHtml(String(oc.checkpoint).replace(/^.*?(?=checkpoints\/)/, ''))}</code>), the asked caption the closest of `
      + `${fmt.int(oc.candidates)}: ${ocLine('seen', oc.seen)}; ${ocLine('held out', oc.held_out)}.${closed}` : '');
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
    + `seen ${pct(r.seen_all)}, held out ${pct(r.held_out_all)}`
    + (r.own_clip == null ? '' : `, our model ${pct(r.own_clip)}`) + '</option>').join('');
  if (keep && pc.genReports.some((r) => r.name === keep)) sel.value = keep;
}

function showClipReport(name) {
  const r = pc.clipReports.find((x) => x.name === name);
  if (!r) {
    $('#pc-clip-table').innerHTML = '<tbody><tr><td>no report yet — train <code>clip-shapes</code> '
      + '(dashboard Start), then Make a contrastive report below</td></tr></tbody>';
    return;
  }
  $('#pc-clip-table').innerHTML = '<thead><tr><th>pictures</th><th>n</th><th>reader</th><th>our model</th>'
    + '<th>closed set</th><th>held-out phrasing</th><th>agree</th><th>kappa</th><th>model\'s misses were wrong about</th></tr></thead><tbody>'
    + r.sets.map((x) => `<tr><td>${escHtml(x.pictures)} (${escHtml(x.split)})</td><td>${fmt.int(x.n)}</td>`
      + `<td>${pct(x.reader)}</td><td><b>${pct(x.top1)}</b></td><td>${pct(x.closed_top1 == null ? x.top1 : x.closed_top1)}</td><td>${pct(x.show_me)}</td>`
      + `<td>${pct(x.agreement.agree)}</td><td>${x.agreement.kappa == null ? '–' : fmt.num(x.agreement.kappa, 2)}</td>`
      + `<td>${escHtml(Object.entries(x.misses_by_attribute || {}).map(([k, v]) => `${k} ${v}`).join(', ') || '–')}</td></tr>`).join('')
    + '</tbody>';
  $('#pc-clip-note').innerHTML = `step ${fmt.int(r.step)}`
    + (r.through ? `; every picture passed through <code>${escHtml(r.through)}</code> first (damaged pictures)` : '; the originals')
    + ((r.sets[0] || {}).floor == null ? '; no none-fits floor (closed set only)'
      : `; none-fits floor ${(r.sets || []).map((x) => `${escHtml(x.kind)} ${fmt.num(x.floor, 3)}`).filter((v, i, a) => a.indexOf(v) === i).join(', ')}`);
}

async function showCocoReport(name) {
  if (!name) {
    $('#pc-coco-table').innerHTML = '<tbody><tr><td>no photo report yet — train <code>flow-coco</code>, then Make a photo report below</td></tr></tbody>';
    $('#pc-coco-note').textContent = '';
    $('#pc-coco-gallery').innerHTML = '';
    return;
  }
  let r;
  try { r = await api(`/api/pictures/report?name=${encodeURIComponent(name)}`); } catch (e) {
    flash(`pictures: ${e.message}`, 'error'); return;
  }
  const row = (label, x) => `<tr><td>${label}</td><td>${fmt.num(x.match, 3)}</td><td>${fmt.num(x.mismatch, 3)}</td>`
    + `<td>${pct(x.top1)}</td><td><b>${pct(x.top5)}</b></td><td>${pct(x.top10)}</td></tr>`;
  $('#pc-coco-table').innerHTML = '<thead><tr><th></th><th>match</th><th>another photo’s caption</th>'
    + '<th>top-1</th><th>top-5</th><th>top-10</th></tr></thead><tbody>'
    + row('real photos', r.real) + row('<b>drawn</b>', r.drawn)
    + `<tr class="dim"><td>chance</td><td>–</td><td>–</td><td>${pct(r.chance.top1)}</td><td>${pct(r.chance.top5)}</td><td>${pct(r.chance.top10)}</td></tr></tbody>`;
  $('#pc-coco-note').innerHTML = `${fmt.int(r.n)} captions, guidance ${fmt.num(r.guidance, 1)}, step ${fmt.int(r.step)}, `
    + `judged by <code>${escHtml(r.clip || '')}</code>. Distance to the real photos in our features: drawings `
    + `<b>${fmt.num(r.distance.drawn_vs_real, 4)}</b>, two halves of the real photos ${fmt.num(r.distance.real_half_vs_half, 4)}.`;
  $('#pc-coco-gallery').innerHTML = (r.gallery_png || []).map((src) => `<figure><img src="${src}" loading="lazy" `
    + 'alt="real photos above, drawings of their captions below"><figcaption>real photos above, drawings of the same captions below: '
    + (r.captions || []).slice(0, 16).map((c, i) => `${i + 1}. ${escHtml(c)}`).join(' · ') + '</figcaption></figure>').join('');
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
  pc.gens = gens;
  const genOpts = gens.map((g) => `<option value="${escHtml(g.rel)}">${escHtml(g.kind || 'G1')} · `
    + `${escHtml(g.rel)} — step ${fmt.int(g.step)}</option>`).join('');
  for (const id of ['#pc-gen', '#pc-gen-job']) {
    const keep = $(id).value;
    $(id).innerHTML = genOpts;
    if (keep && gens.some((g) => g.rel === keep)) $(id).value = keep;
  }
  $('#pc-draw').disabled = !gens.length;
  syncDrawInputs();
  $('#pc-gen-run').disabled = !gens.length;
  $('#pc-codes').disabled = !cks.length;
  if (!gens.length) {
    $('#pc-draw-note').innerHTML = 'no G1 model yet: encode the corpus below, then train '
      + '<code>imggen-shapes</code> with the dashboard’s Start';
  }
  pc.clipReports = res.clip_reports || [];
  const keepClip = $('#pc-clip-report').value;
  $('#pc-clip-report').innerHTML = pc.clipReports.map((r) => `<option value="${escHtml(r.name)}">`
    + `${escHtml(r.name)}</option>`).join('');
  if (keepClip && pc.clipReports.some((r) => r.name === keepClip)) $('#pc-clip-report').value = keepClip;
  showClipReport($('#pc-clip-report').value);
  const clips = res.clips || [];
  const clipOpts = clips.map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — step ${fmt.int(c.step)}</option>`).join('');
  const keepC = $('#pc-clip-ckpt').value;
  $('#pc-clip-ckpt').innerHTML = clipOpts;
  if (keepC && clips.some((c) => c.rel === keepC)) $('#pc-clip-ckpt').value = keepC;
  const keepG = $('#pc-gen-clip').value;
  $('#pc-gen-clip').innerHTML = '<option value="">reader only</option>' + clipOpts;
  if (keepG && clips.some((c) => c.rel === keepG)) $('#pc-gen-clip').value = keepG;
  const keepT = $('#pc-clip-through').value;
  $('#pc-clip-through').innerHTML = '<option value="">the originals</option>'
    + (res.coders || []).map((c) => `<option value="${escHtml(c)}">through ${escHtml(c)}</option>`).join('');
  if (keepT) $('#pc-clip-through').value = keepT;
  $('#pc-clip-run').disabled = !clips.length;
  $('#pc-clip-cal').disabled = !clips.length;
  $('#pc-scenes-note').textContent = c.scenes
    ? 'data/vision/scenes64k is rendered (64,000 scenes, plus 128 held-out pictures).'
    : 'the scenes corpus is not made yet: Render the scenes corpus (seconds).';
  const cocoReports = res.coco_reports || [];
  const keepR = $('#pc-coco-report').value;
  $('#pc-coco-report').innerHTML = cocoReports.map((r) => `<option value="${escHtml(r.name)}">${escHtml(r.name)} — `
    + `top-5 ${pct(r.top5)} (real ${pct(r.real_top5)})</option>`).join('');
  if (keepR && cocoReports.some((r) => r.name === keepR)) $('#pc-coco-report').value = keepR;
  await showCocoReport($('#pc-coco-report').value || null);
  const cocoGens = gens.filter((g) => g.corpus_kind === 'coco');
  $('#pc-coco-gen').innerHTML = cocoGens.map((g) => `<option value="${escHtml(g.rel)}">${escHtml(g.rel)} — step ${fmt.int(g.step)}</option>`).join('');
  $('#pc-coco-clip').innerHTML = clipOpts;
  const cc = clips.find((x) => x.rel.includes('clip-coco'));
  if (cc) $('#pc-coco-clip').value = cc.rel;
  $('#pc-coco-run').disabled = !cocoGens.length || !clips.length;
  const cd = res.coco || {};
  $('#pc-coco-data').textContent = cd.train && cd.val
    ? 'data/vision/coco64 and coco64-val are packed.'
    : 'COCO is not packed yet: python -m aksharallm.pictures coco (after downloading the three archives into data/coco/raw/).';
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
      guidance: Number($('#pc-gen-g').value), clip: $('#pc-gen-clip').value,
    });
    $('#pc-clip-report').onchange = (e) => showClipReport(e.target.value);
    $('#pc-clip-run').onclick = () => pc.job.start({
      kind: 'clip-report', checkpoint: $('#pc-clip-ckpt').value, through: $('#pc-clip-through').value,
    });
    $('#pc-scenes').onclick = () => pc.job.start({ kind: 'scenes' });
    $('#pc-coco-report').onchange = (e) => showCocoReport(e.target.value);
    $('#pc-coco-run').onclick = () => pc.job.start({
      kind: 'coco-report', checkpoint: $('#pc-coco-gen').value, clip: $('#pc-coco-clip').value,
      n: Number($('#pc-coco-n').value) || 200, guidance: Number($('#pc-coco-g').value),
    });
    $('#pc-clip-cal').onclick = () => pc.job.start({ kind: 'clip-calibrate', checkpoint: $('#pc-clip-ckpt').value });
    if (!$('#pc-w-count').options.length) {
      const fill = (id, xs, v) => { $(id).innerHTML = xs.map((x) => `<option>${x}</option>`).join(''); $(id).value = v; };
      fill('#pc-w-count', COUNTS, 'two');
      fill('#pc-w-colour', COLOURS, 'red');
      fill('#pc-w-shape', SHAPES, 'circle');
    }
    $('#pc-draw').onclick = draw;
    $('#pc-gen').onchange = syncDrawInputs;
    $('#pc-gen-report').onchange = (e) => showGenReport(e.target.value);
    $('#pc-gcompare').onclick = genCompare;
    await load().catch((e) => flash(`pictures: ${e.message}`, 'error'));
  },
  leave() { if (pc.job) pc.job.stop(); },
});

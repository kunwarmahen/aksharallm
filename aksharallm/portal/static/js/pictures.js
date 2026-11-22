/* The Pictures tab: the picture tokenizer's reports (PSNR and the reader's verdict at each
 * codebook count), its gallery, a paired comparison of two reports, and its two CLI jobs
 * (docs/26).
 *
 * The reader's row on the originals comes first in the table: it is the positive control,
 * and a report where it is not 100% means every number under it is wrong.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';
import { jobBox } from './clijob.js';

const pc = { reports: [], job: null };

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
    await load().catch((e) => flash(`pictures: ${e.message}`, 'error'));
  },
  leave() { if (pc.job) pc.job.stop(); },
});

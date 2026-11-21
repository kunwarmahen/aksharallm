/* The World tab: the toy world model's scores beside the copy baseline, a game it plays
 * open loop next to the true one, and its two CLI jobs (docs/25).
 *
 * Every score is shown against what copying the last frame would get, because most cells of
 * a frame never change and a model that only copies looks about a third right overall.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';
import { jobBox } from './clijob.js';

const wd = { results: [], ckpt: null, job: null, busy: false };
const CELL = { '#': 'c-wall', '.': '', '@': 'c-agent', '+': 'c-agent', $: 'c-box', o: 'c-goal', '*': 'c-done' };

function frame(cells, against) {
  return `<div class="wd-frame">${cells.map((c, i) => {
    const wrong = against && against[i] !== c ? ' wd-wrong' : '';
    return `<span class="wd-cell ${CELL[c] || ''}${wrong}">${c === '.' || c === '#' ? '' : escHtml(c)}</span>`;
  }).join('')}</div>`;
}

function showResult(name) {
  const r = wd.results.find((x) => x.name === name);
  if (!r) {
    $('#wd-outcomes').innerHTML = '<tbody><tr><td>no evaluation yet — train <code>world-sokoban</code>, then Evaluate below</td></tr></tbody>';
    $('#wd-rollout').innerHTML = '';
    return;
  }
  const nf = r.next_frame;
  $('#wd-outcomes').innerHTML = '<thead><tr><th>outcome</th><th>n</th><th>exact frame</th>'
    + '<th>cells right</th><th>copying would score</th></tr></thead><tbody>'
    + Object.entries(nf.by_outcome).map(([w, o]) => `<tr><td>${escHtml(w)}</td><td>${fmt.int(o.n)}</td>`
      + `<td><b>${fmt.pct(o.exact)}</b></td><td>${fmt.pct(o.cells, 2)}</td><td>${fmt.pct(o.copy_baseline, 0)}</td></tr>`).join('')
    + `<tr><td><b>all</b></td><td>${fmt.int(nf.n)}</td><td><b>${fmt.pct(nf.exact)}</b> ± ${fmt.pct(nf.stderr)}</td>`
    + `<td>${fmt.pct(nf.cells, 2)}</td><td>${fmt.pct(nf.copy_baseline)}</td></tr></tbody>`;
  const ro = r.rollout;
  $('#wd-rollout').innerHTML = '<span class="panel-note">rollout, exact frame by step (blue) and impossible frames (orange):</span>'
    + ro.exact_by_step.map((x, k) => `<div class="wd-bar" title="step ${k + 1}: ${fmt.pct(x)} exact, ${fmt.pct(ro.impossible_by_step[k])} impossible">`
      + `<span style="height:${Math.max(2, x * 60)}px"></span>`
      + `<span class="wd-imp" style="height:${ro.impossible_by_step[k] * 60}px"></span>${k + 1}</div>`).join('');
}

async function load() {
  const res = await api('/api/world');
  wd.results = res.results || [];
  const cks = res.checkpoints || [];
  $('#wd-result').innerHTML = wd.results.map((r) =>
    `<option value="${escHtml(r.name)}">${escHtml(r.name)} — ${fmt.pct(r.next_frame.exact)} exact</option>`).join('');
  showResult(wd.results.length ? wd.results[0].name : null);
  $('#wd-ckpt').innerHTML = cks.map((c) =>
    `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — step ${fmt.int(c.step)}</option>`).join('');
  wd.ckpt = cks.length ? (cks.some((c) => c.rel === wd.ckpt) ? wd.ckpt : cks[0].rel) : null;
  if (wd.ckpt) $('#wd-ckpt').value = wd.ckpt;
  $('#wd-play').disabled = !wd.ckpt;
  $('#wd-eval').disabled = !wd.ckpt;
  const d = res.data;
  $('#wd-data').textContent = d
    ? `data/world: ${fmt.compact(d.tokens.train)} training tokens from ${fmt.int(d.episodes.train)} games of ${d.steps} steps · `
      + `moves: ${Object.entries(d.outcomes.train).map(([k, v]) => `${k} ${fmt.int(v)}`).join(', ')}`
    : 'no data yet — Make the data below (about a minute on the CPU)';
  $('#wd-status').textContent = `on the ${res.device} — ${res.device_reason}`;
  wd.job.render(res.jobs);
  if (res.jobs && res.jobs.running) wd.job.poll();
}

async function play() {
  if (wd.busy || !wd.ckpt) return;
  wd.busy = true;
  $('#wd-play').disabled = true;
  $('#wd-status').textContent = 'playing…';
  try {
    const res = await post('/api/world/play', {
      checkpoint: wd.ckpt, horizon: Number($('#wd-horizon').value) || 6, seed: Number($('#wd-seed').value) || 0,
    });
    $('#wd-cmd').hidden = false;
    $('#wd-cmd').textContent = res.command;
    const bad = res.steps.filter((s) => s.wrong).length;
    $('#wd-status').textContent = `${res.steps.length - 1} steps on the ${res.device}: `
      + (bad ? `${bad} predicted frame(s) wrong` : 'every predicted frame exactly right');
    $('#wd-steps').innerHTML = res.steps.map((s) => `<div class="wd-step${s.wrong ? ' wd-bad' : ''}">
        <h4>${s.step ? `step ${s.step}: ${escHtml(s.action)} → ${escHtml(s.outcome)}` : 'the starting level'}
          ${s.step ? (s.wrong ? ` · ${s.wrong} cells wrong` : ' · exact') : ''}${s.impossible ? ' · impossible frame' : ''}</h4>
        <div class="wd-pair"><div><small>true</small>${frame(s.true)}</div>
        ${s.step ? `<div><small>model</small>${frame(s.model, s.true)}</div>` : ''}</div></div>`).join('');
  } catch (e) {
    $('#wd-status').textContent = e.message;
  } finally {
    wd.busy = false;
    $('#wd-play').disabled = false;
  }
}

registerTab('world', {
  async open() {
    if (!wd.job) wd.job = jobBox($('#wd-job'), 'world', { onDone: () => load() });
    $('#wd-result').onchange = (e) => showResult(e.target.value);
    $('#wd-ckpt').onchange = (e) => { wd.ckpt = e.target.value; };
    $('#wd-play').onclick = play;
    $('#wd-prepare').onclick = () => wd.job.start({ kind: 'prepare', episodes: Number($('#wd-episodes').value) || 60000 });
    $('#wd-eval').onclick = () => wd.job.start({ kind: 'eval', checkpoint: wd.ckpt });
    await load().catch((e) => flash(`world: ${e.message}`, 'error'));
  },
  leave() { if (wd.job) wd.job.stop(); },
});

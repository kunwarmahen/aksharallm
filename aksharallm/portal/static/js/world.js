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
const CELL = { '#': 'c-wall', '.': '', '@': 'c-agent', '+': 'c-agent', $: 'c-box', o: 'c-goal', '*': 'c-done', k: 'c-key', D: 'c-door' };

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
  $('#wd-doors').innerHTML = '';
  if (r.game === 'keys') { showKeys(r); return; }
  $('#wd-outcomes').innerHTML = '<thead><tr><th>outcome</th><th>n</th><th>exact frame</th>'
    + '<th>cells right</th><th>copying would score</th></tr></thead><tbody>'
    + Object.entries(nf.by_outcome).map(([w, o]) => `<tr><td>${escHtml(w)}</td><td>${fmt.int(o.n)}</td>`
      + `<td><b>${fmt.pct(o.exact)}</b></td><td>${fmt.pct(o.cells, 2)}</td><td>${fmt.pct(o.copy_baseline, 0)}</td></tr>`).join('')
    + `<tr><td><b>all</b></td><td>${fmt.int(nf.n)}</td><td><b>${fmt.pct(nf.exact)}</b> ± ${fmt.pct(nf.stderr)}</td>`
    + `<td>${fmt.pct(nf.cells, 2)}</td><td>${fmt.pct(nf.copy_baseline)}</td></tr></tbody>`;
  showRollout(r.rollout);
}

/* The keys world has no exact answer: a move slips 10% of the time and a door depends on a
 * key the frame never shows. So the table is the most likely frame (with the copy column),
 * KL from the true distribution, and the probability given to the most likely frame, model
 * beside truth; and a second table asks, at every door, how sure the model is that it opens. */
function showKeys(r) {
  const nf = r.next_frame;
  const cal = r.calibration || {};
  $('#wd-outcomes').innerHTML = '<thead><tr><th>the move</th><th>n</th><th>most likely frame</th>'
    + '<th>copying would score</th><th>KL (nats)</th><th>P(most likely) — true</th><th>— model</th></tr></thead><tbody>'
    + Object.entries(nf.by_outcome).map(([w, o]) => `<tr><td>${escHtml(w)}</td><td>${fmt.int(o.n)}</td>`
      + `<td><b>${fmt.pct(o.exact)}</b></td><td>${fmt.pct(o.copy_baseline, 0)}</td><td>${fmt.num(o.kl, 3)}</td>`
      + `<td>${fmt.num(o.p_mode_true, 3)}</td><td>${fmt.num(o.p_mode_model, 3)}</td></tr>`).join('')
    + `<tr><td><b>all</b></td><td>${fmt.int(nf.n)}</td><td><b>${fmt.pct(nf.exact)}</b></td>`
    + `<td>${fmt.pct(nf.copy_baseline)}</td><td><b>${fmt.num(cal.kl, 3)}</b></td><td colspan="2">`
    + `${fmt.pct(cal.off, 2)} of its probability on frames the game cannot produce</td></tr></tbody>`;
  const doors = cal.doors || {};
  $('#wd-doors').innerHTML = '<thead><tr><th>at a door, the agent…</th><th>n</th><th>model: P(it opens)</th>'
    + '<th>truth</th><th>right</th></tr></thead><tbody>'
    + Object.entries(doors).map(([w, d]) => `<tr><td>${escHtml(w)}</td><td>${fmt.int(d.n)}</td>`
      + `<td><b>${fmt.num(d.belief, 3)}</b></td><td>${d.truth}</td><td>${fmt.pct(d.right)}</td></tr>`).join('')
    + (cal.base_rate == null ? '' : `<tr><td colspan="5" class="panel-note">doors with no pickup in view: `
      + `${fmt.num(cal.base_rate, 3)} held, the model's mean belief `
      + (cal.unseen_belief == null ? '(not recorded in this file)' : fmt.num(cal.unseen_belief, 3))
      + ' — calibrated when the two match. It is not a ceiling on the middle row: the frame can '
      + 'still say a key was taken (no key left on the map). '
      + `Episodes from the <b>${escHtml(r.policy || 'aware')}</b> agent`
      + (r.trained_policy && r.trained_policy !== r.policy ? ` (the model trained on the ${escHtml(r.trained_policy)} one)` : '')
      + '.</td></tr>')
    + '</tbody>';
  showRollout(r.rollout);
}

function showRollout(ro) {
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
    `<option value="${escHtml(r.name)}">${escHtml(r.name)} — ${r.game === 'keys'
      ? `KL ${fmt.num(r.calibration && r.calibration.kl, 3)}` : `${fmt.pct(r.next_frame.exact)} exact`}</option>`).join('');
  showResult(wd.results.length ? wd.results[0].name : null);
  $('#wd-ckpt').innerHTML = cks.map((c) =>
    `<option value="${escHtml(c.rel)}">${escHtml(c.rel)} — ${escHtml(c.game)}, step ${fmt.int(c.step)}</option>`).join('');
  wd.ckpt = cks.length ? (cks.some((c) => c.rel === wd.ckpt) ? wd.ckpt : cks[0].rel) : null;
  if (wd.ckpt) $('#wd-ckpt').value = wd.ckpt;
  $('#wd-play').disabled = !wd.ckpt;
  $('#wd-eval').disabled = !wd.ckpt;
  const ds = res.data || [];
  $('#wd-data').innerHTML = ds.length
    ? ds.map((d) => `<span>${escHtml(d.dir)} (${escHtml(d.game)}): ${fmt.compact(d.tokens.train)} training tokens from `
      + `${fmt.int(d.episodes.train)} games of ${d.steps} steps · moves: `
      + `${Object.entries(d.outcomes.train).map(([k, v]) => `${escHtml(k)} ${fmt.int(v)}`).join(', ')}</span>`).join('<br>')
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
    $('#wd-prepare').onclick = () => wd.job.start({
      kind: 'prepare', game: $('#wd-game').value, episodes: Number($('#wd-episodes').value) || 60000,
    });
    $('#wd-eval').onclick = () => wd.job.start({ kind: 'eval', checkpoint: wd.ckpt, policy: $('#wd-on').value });
    await load().catch((e) => flash(`world: ${e.message}`, 'error'));
  },
  leave() { if (wd.job) wd.job.stop(); },
});

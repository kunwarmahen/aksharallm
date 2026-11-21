/* One detached CLI job, drawn and polled — the browser half of portal/jobs.py (CliJobs).
 *
 * Shared by the World and Tools tabs and the Audio tab's TTS panel. Every job box shows the
 * exact `python -m ...` command it ran, so the browser never does anything a terminal could
 * not, and its state comes from the job's EXIT CODE (done / failed / lost / stopped), never
 * from "an output file appeared" (gotcha 20).
 *
 * The box repolls straight after a start (gotcha 21(c): a button that changes nothing on
 * screen for ten seconds reads as a button that does nothing), then every 3 s while running,
 * and calls `onDone` once when a job it saw running finishes, so the tab can reload results.
 */
import { api, escHtml, flash, post } from './core.js';

export function jobBox(container, base, { onDone } = {}) {
  container.innerHTML = `<div class="cj-box" hidden>
      <div class="row"><b class="cj-state"></b><span class="cj-label"></span>
        <button class="cj-stop" type="button">Stop</button></div>
      <pre class="au-cmd cj-cmd"></pre>
      <pre class="cj-log"></pre>
    </div>`;
  const box = container.querySelector('.cj-box');
  let timer = null;
  let sawRunning = false;

  function render(st) {
    const cur = st && st.current;
    box.hidden = !cur;
    if (!cur) return;
    box.querySelector('.cj-state').textContent = st.running ? 'running' : cur.state;
    box.querySelector('.cj-state').className = `cj-state cj-${st.running ? 'running' : cur.state}`;
    box.querySelector('.cj-label').textContent = ` ${cur.label || ''}`;
    box.querySelector('.cj-cmd').textContent = cur.command || '';
    box.querySelector('.cj-log').textContent = (st.log || []).join('\n');
    box.querySelector('.cj-stop').hidden = !st.running;
    if (st.running) sawRunning = true;
    else if (sawRunning) { sawRunning = false; if (onDone) onDone(cur); }
  }

  async function poll() {
    clearTimeout(timer);
    try {
      const st = await api(`/api/${base}`);
      render(st.jobs);
      if (st.jobs && st.jobs.running) timer = setTimeout(poll, 3000);
    } catch (e) { /* the next action repolls; one failed read is not worth a flash */ }
  }

  async function start(spec) {
    try {
      const res = await post(`/api/${base}/job`, spec);
      flash(`started: ${res.command}`);
      sawRunning = true;
      setTimeout(poll, 300);
    } catch (e) {
      flash(e.message, 'error');
    }
  }

  box.querySelector('.cj-stop').onclick = async () => {
    try { await post(`/api/${base}/stop`, {}); } catch (e) { flash(e.message, 'error'); }
    setTimeout(poll, 300);
  };

  return { render, poll, start, stop() { clearTimeout(timer); } };
}

/* A one-line verdict on a paired comparison: the counts, then the p-value, never a delta. */
export function pairedLine(better, worse, p, what) {
  const sig = p < 0.05 ? ' <b>(a real difference)</b>' : ' (not distinguishable from noise)';
  return `${escHtml(what)}: better on <b>${better}</b>, worse on <b>${worse}</b>, `
    + `sign test p ${p < 0.001 ? p.toExponential(1) : p.toFixed(3)}${sig}`;
}

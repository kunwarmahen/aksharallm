/* The shared kernel: DOM lookup, number formatting, the fetch wrapper, and the two status
 * surfaces (the flash banner and the live badge). Every other module imports from here;
 * this one imports nothing. */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ---------------------------------------------------------------- formatting ---------- */

export const fmt = {
  int: (n) => (n == null || Number.isNaN(n) ? '–' : Math.round(n).toLocaleString()),
  num: (n, d = 3) => (n == null || Number.isNaN(n) ? '–' : Number(n).toFixed(d)),
  pct: (n, d = 1) => (n == null || Number.isNaN(n) ? '–' : (n * 100).toFixed(d) + '%'),
  exp: (n) => (n == null || Number.isNaN(n) ? '–' : Number(n).toExponential(2)),
  /** 1.23B / 45.6M — the same compact form the trainer prints. */
  compact: (n) => {
    if (n == null || Number.isNaN(n)) return '–';
    const units = ['', 'K', 'M', 'B', 'T'];
    let v = n, i = 0;
    while (Math.abs(v) >= 1000 && i < units.length - 1) { v /= 1000; i++; }
    return `${v.toFixed(v >= 100 || i === 0 ? 0 : 2)}${units[i]}`;
  },
  /** Mirrors runlog.fmt_dur so the page and the log agree: 45.2s / 12m30s / 6h05m / 3d04h. */
  dur: (s) => {
    if (s == null || Number.isNaN(s)) return '–';
    if (s < 0) return '-' + fmt.dur(-s);
    if (s < 60) return `${s.toFixed(1)}s`;
    let m = Math.floor(s / 60), sec = Math.floor(s % 60);
    let h = Math.floor(m / 60); m %= 60;
    const d = Math.floor(h / 24); h %= 24;
    if (d) return `${d}d${String(h).padStart(2, '0')}h`;
    if (h) return `${h}h${String(m).padStart(2, '0')}m`;
    return `${m}m${String(sec).padStart(2, '0')}s`;
  },
  ago: (ts) => (ts == null ? '–' : fmt.dur(Date.now() / 1000 - ts) + ' ago'),
  clock: (ts) => (ts == null ? '–' : new Date(ts * 1000).toLocaleString()),
  bytes: (n) => (n == null ? '–' : (n / 1e9 >= 1 ? (n / 1e9).toFixed(2) + ' GB'
    : (n / 1e6).toFixed(1) + ' MB')),
};

/* ---------------------------------------------------------------- api ----------------- */

export async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', 'X-Portal': '1', ...(options.headers || {}) },
  });
  let data = {};
  try { data = await res.json(); } catch { /* a 500 with no body */ }
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body || {}) });

export function flash(msg, kind = '') {
  const box = $('#flash');
  box.textContent = msg;
  box.className = 'flash ' + kind;
  box.hidden = !msg;
}

export function live(text, kind) {
  $('#live-label').textContent = text;
  $('#live').className = 'live ' + (kind || '');
}

export const escHtml = (s) => String(s == null ? '' : s).replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

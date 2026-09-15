// Session cookies are sent automatically — no manual auth header injection needed.
const getJSON = (u) => fetch(u).then(r => r.json());
const postJSON = (u, body) => fetch(u, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});
const putJSON = (u, body) => fetch(u, {
  method: 'PUT', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});
const $ = (id) => document.getElementById(id);
const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstChild; };
const screen = () => $('screen');

function normalizeEndpointUrl(raw) {
  let u = (raw || '').trim();
  if (!u) return u;
  if (!/^https?:\/\//i.test(u)) u = 'http://' + u;
  if (!/\/v1\/?$/.test(u)) u = u.replace(/\/$/, '') + '/v1';
  return u;
}

// Placeholder hint for a secret field: masks whether a value already exists
// in credentials.json without ever surfacing it (the store only reports
// booleans). Fields never carry the real value — leaving it blank keeps it.
function credPlaceholder(alreadySet, fallback) {
  return alreadySet ? '*** already set — enter a new value to replace' : fallback;
}

function showAlert(detail) {
  const d = document.createElement('div');
  d.className = 'alert-detail';
  d.textContent = detail || 'Please choose a different setup below.';
  $('alert').innerHTML =
    `<div class="alert-banner"><span class="alert-icon" aria-hidden="true">⚠</span>`
    + `<div><strong>This configuration doesn't work on your system.</strong>`
    + `${d.outerHTML}</div></div>`;
}
function clearAlert() { const a = $('alert'); if (a) a.innerHTML = ''; }

// Render the blocking preflight errors as a bulleted list in the alert pane.
// Each error is {check: 'disk'|'network'|'gpu', message: '…'}. One banner
// header + one bullet per failed check, so the user can fix them all in
// one pass instead of round-tripping per failure.
function showPreflightErrors(errors) {
  if (!errors || !errors.length) return;
  const labels = { disk: 'Disk space', network: 'Network', gpu: 'GPU' };
  const items = errors.map(e => {
    const label = labels[e.check] || e.check;
    return `<li><strong>${label}:</strong> ${e.message}</li>`;
  }).join('');
  $('alert').innerHTML =
    `<div class="alert-banner"><span class="alert-icon" aria-hidden="true">⚠</span>`
    + `<div><strong>Can't start the model download yet.</strong>`
    + `<ul class="alert-list">${items}</ul></div></div>`;
}
function wakewordModelOptions(schema) {
  const seen = new Set();
  return (schema.wakeword_presets || []).flatMap(p => p.model_options || []).filter(option => {
    if (!option.path || seen.has(option.path)) return false;
    seen.add(option.path);
    return true;
  });
}

function wakewordModelDatalist(schema, id) {
  const options = wakewordModelOptions(schema).map(option =>
    `<option value="${option.path.replace(/"/g, '&quot;')}">${option.label}</option>`).join('');
  return `<datalist id="${id}">${options}</datalist>`;
}

const WIZARD_STEPS = ['Brain', 'Set up', 'Download', 'Starting up'];

function renderSteps(activeIdx, mode) {
  const box = $('steps');
  if (mode === 'settings') { box.innerHTML = ''; return; }
  box.innerHTML = '';
  WIZARD_STEPS.forEach((s, i) => {
    const cls = i === activeIdx ? 'active' : (i < activeIdx ? 'done' : '');
    box.appendChild(el(`<span class="s ${cls}">${i + 1}. ${s}</span>`));
  });
}
function cpuVariantNotice(schema) {
  if (!schema || schema.variant !== 'cpu') return null;
  return el(`<div class="variant-notice">
    <h3>Running on the CPU image</h3>
    <p>Audio (speech recognition + text-to-speech) runs fully local. The language model is either regex-only (simple commands) or off-box via an OpenAI-compatible server you already run.</p>
    <p>For the full stack (voice cloning + 9B SLM on one NVIDIA box), stop this container and run a new one with the <code>:latest</code> tag and <code>--gpus all</code> — see the <strong>GPU</strong> block in the README. The wizard is the same on both images; only the model download differs.</p>
  </div>`);
}
function setBranding(remote) {
  const img = document.getElementById('brand-logo');
  if (img) img.src = remote ? '/logo.png?remote=1' : '/logo.png';
}

export { getJSON, postJSON, putJSON, $, el, screen, normalizeEndpointUrl, credPlaceholder, showAlert, clearAlert, showPreflightErrors, wakewordModelDatalist, renderSteps, cpuVariantNotice, setBranding };

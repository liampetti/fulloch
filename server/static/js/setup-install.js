import {
  getJSON, postJSON, putJSON, $, el, screen, showAlert, clearAlert,
  showPreflightErrors, renderSteps,
} from './setup-common.js';
import { createLifetime } from './browser-lifetime.js';

export function createInstallProgress({ state, chosenModels, boot, stepFinish }) {
  const lifetime = createLifetime();
  let pollGeneration = 0;
  const stopPolling = () => { pollGeneration++; lifetime.clearTimers(); };
  const sel = state.sel;
// ---- install ---------------------------------------------------------------
async function doInstall() {
  const btn = $('get-started');
  if (btn) btn.disabled = true;
  clearAlert();
  // Blocking preflight before the model download starts: disk space,
  // network reach to the model hub, and (for GPU tiers) an NVIDIA GPU
  // being visible. Each failed check becomes one bullet in the error
  // pane so the user knows exactly which to fix.
  // docs/ease-of-use-tasks.md.
  const models = chosenModels();
  const pre = await postJSON('/setup/preflight-download', { models });
  const preBody = await pre.json().catch(() => ({ ok: true, errors: [] }));
  if (!pre.ok) {
    showPreflightErrors(preBody.errors || []);
    if (btn) btn.disabled = false;
    return;
  }
  await postJSON('/setup/models', { models });
  const updates = {
    'general.wakeword': sel.wakeword,
    'general.wakeword_pattern': sel.wakeword_pattern || '',
    'general.voice_clone': sel.voice_clone || '',
  };
  if (sel.ha.url || sel.clear_integrations) updates['home_assistant.url'] = sel.ha.url;
  if (sel.search_url || sel.clear_integrations) updates['search.searxng_url'] = sel.search_url;
  await putJSON('/config', { updates });
  // Tokens go to credentials.json, not config.yml.
  if (sel.ha.token) await postJSON('/setup/credential', { key: 'ha_token', value: sel.ha.token });
  if (sel.openai && sel.openai.api_key) await postJSON('/setup/credential', { key: 'llm_api_key', value: sel.openai.api_key });
  const r = await postJSON('/setup/install');
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    alert('Could not start setup: ' + (e.detail || r.status));
    if (btn) btn.disabled = false;
    return;
  }
  state.cameViaWizard = true;
  showProgress();
}

// ---- download progress ----------------------------------------------------
function showProgress() {
  stopPolling();
  state.curStep = 2; renderSteps(2);
  clearAlert();
  screen().innerHTML = `<div class="card"><h2>Downloading models</h2>
    <p class="lead">Pulling model weights. You can leave this page open.</p>
    <div id="assets"></div><div id="dl-error"></div>
    <div class="actions"><span></span><button class="danger" id="cancel-startup">Stop and return to setup</button></div></div>`;
  $('cancel-startup').addEventListener('click', cancelStartup);
  pollProgress();
}

async function pollProgress(generation = pollGeneration) {
  if (lifetime.destroyed || generation !== pollGeneration) return;
  let snap;
  try { snap = await getJSON('/setup/progress'); } catch (e) {
    if (generation === pollGeneration) lifetime.setTimeout(() => pollProgress(generation), 1500);
    return;
  }
  if (lifetime.destroyed || generation !== pollGeneration) return;
  const box = $('assets');
  box.innerHTML = '';
  (snap.assets || []).forEach(a => {
    const pct = a.pct != null ? a.pct : (a.status === 'done' ? 100 : (a.status === 'downloading' ? null : 0));
    const barStyle = pct != null ? `width:${pct}%` : 'width:40%;animation:pulse 1.2s ease-in-out infinite';
    let sizeLabel = '';
    if (a.bytes_total && a.status === 'downloading') {
      const done = (a.bytes_done / 1e9).toFixed(2);
      const total = (a.bytes_total / 1e9).toFixed(2);
      sizeLabel = ` <span style="font-size:.75rem;opacity:.7">${done}/${total} GB</span>`;
    } else if (a.size_gb && a.status !== 'done') {
      sizeLabel = ` <span style="font-size:.75rem;opacity:.7">${a.size_gb} GB</span>`;
    }
    box.appendChild(el(`<div class="asset"><div style="min-width:9rem">${a.label}${sizeLabel}</div>
      <div class="bar"><i style="${barStyle}"></i></div>
      <div class="st">${a.status}</div></div>`));
  });
  if (snap.state === 'error') {
    const tokenHelp = snap.needs_hf_token ? `
      <div class="hf-token-help">
        <strong>This model needs Hugging Face access.</strong>
        <p>Accept the model's access terms on Hugging Face, then create a read token and paste it here. It is stored only in <code>data/credentials.json</code>.</p>
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">Create a Hugging Face read token</a>
        <label for="hf-token">Hugging Face token</label>
        <input type="password" id="hf-token" autocomplete="off" placeholder="hf_...">
        <button class="primary" onclick="saveHfTokenAndRetry()">Save token and retry</button>
      </div>` : '';
    $('dl-error').innerHTML = `<div class="banner error"><div>${snap.error || 'Download failed.'}</div>${tokenHelp}
      <button style="margin-top:.75rem" onclick="retryDownload()">Retry download</button></div>`;
    return;
  }
  if (snap.state === 'done') { return showLoading(); }
  lifetime.setTimeout(() => pollProgress(generation), 1200);
}

async function retryDownload() {
  stopPolling();
  const generation = pollGeneration;
  $('dl-error').innerHTML = '';
  await postJSON('/setup/retry-download', {});
  pollProgress(generation);
}

async function saveHfTokenAndRetry() {
  const token = ($('hf-token').value || '').trim();
  if (!token) { $('hf-token').focus(); return; }
  const r = await postJSON('/setup/credential', { key: 'hf_token', value: token });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    alert('Could not save the Hugging Face token: ' + (e.detail || r.status));
    return;
  }
  await retryDownload();
}

// ---- loading (models loading into memory) ---------------------------------
let loadSeenSeq = 0;

function showLoading() {
  stopPolling();
  state.curStep = 3;
  loadSeenSeq = 0;
  clearAlert();
  $('subtitle').textContent = state.cameViaWizard ? 'First-run setup' : 'Starting up';
  renderSteps(3, state.cameViaWizard ? undefined : 'settings');
  screen().innerHTML = `<div class="card"><h2>Starting up</h2>
    <p class="lead">Loading models and warming up prompts — the assistant will be ready shortly.</p>
    <div id="term" class="term" aria-live="polite"></div>
    <div class="term-foot"><span class="spinner sm"></span><span class="muted" id="load-detail"></span></div>
    <div class="actions"><span></span><button class="danger" id="cancel-startup">Stop and return to setup</button></div></div>`;
  $('cancel-startup').addEventListener('click', cancelStartup);
  pollLoading();
}

async function cancelStartup() {
  const btn = $('cancel-startup');
  if (btn) { btn.disabled = true; btn.textContent = 'Stopping…'; }
  const r = await postJSON('/setup/cancel-startup', {});
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    if (btn) { btn.disabled = false; btn.textContent = 'Stop and return to setup'; }
    showAlert(e.detail || 'Could not stop startup.');
    return;
  }
  if (btn) btn.textContent = 'Returning to setup…';
  stopPolling();
  await waitForRestart(state.status && state.status.server_instance_id);
  if (!lifetime.destroyed) location.replace(`/?setup=${Date.now()}`);
}

async function waitForRestart(previousInstanceId) {
  // The old process is about to exit. Do not navigate until its replacement is
  // serving requests, otherwise the wizard's later voice/schema fetches race
  // the restart and leave a partially rendered page.
  while (!lifetime.destroyed) {
    try {
      const status = await fetch('/status', { cache: 'no-store' }).then(r => r.json());
      if (status.server_instance_id && status.server_instance_id !== previousInstanceId) return;
    } catch (e) { /* The server is between processes; retry shortly. */ }
    await new Promise(resolve => {
      const done = () => {
        lifetime.signal.removeEventListener('abort', done);
        resolve();
      };
      lifetime.setTimeout(done, 150);
      lifetime.signal.addEventListener('abort', done, { once: true });
    });
  }
}

function streamLogLines(log) {
  const term = $('term');
  if (!term || !Array.isArray(log)) return;
  const fresh = log.filter(it => it.seq > loadSeenSeq);
  fresh.forEach((it, i) => {
    const div = document.createElement('div');
    const lvl = it.level === 'WARNING' ? ' warn'
      : (it.level === 'ERROR' || it.level === 'CRITICAL') ? ' error' : '';
    div.className = 'term-line' + lvl;
    div.style.animationDelay = (i * 90) + 'ms';
    div.textContent = it.text;
    term.appendChild(div);
    loadSeenSeq = Math.max(loadSeenSeq, it.seq);
  });
  while (term.children.length > 40) term.removeChild(term.firstChild);
}

async function pollLoading(generation = pollGeneration) {
  if (lifetime.destroyed || generation !== pollGeneration) return;
  let s;
  try { s = await getJSON('/status'); } catch (e) {
    if (generation === pollGeneration) lifetime.setTimeout(() => pollLoading(generation), 1200);
    return;
  }
  if (lifetime.destroyed || generation !== pollGeneration) return;
  streamLogLines(s.log);
  $('load-detail') && ($('load-detail').textContent = s.detail || '');
  if (s.phase === 'READY') { return state.cameViaWizard ? stepFinish() : (location.href = '/'); }
  if (s.phase === 'ERROR') { return boot(); }
  lifetime.setTimeout(() => pollLoading(generation), 900);
}
async function doRestart(ev) {
  const btn = ev && ev.target;
  if (btn) { btn.disabled = true; btn.textContent = 'Restarting…'; }
  try { await postJSON('/restart'); } catch (e) { /* server may drop the connection */ }
  let tries = 0;
  const poll = () => lifetime.setTimeout(async () => {
    try { await getJSON('/status'); if (!lifetime.destroyed) location.reload(); }
    catch (e) {
      if (++tries < 90) poll();
      else if (btn) { btn.disabled = false; btn.textContent = 'Restart now'; }
    }
  }, 2000);
  lifetime.setTimeout(poll, 3000);
}

// Enabling/regenerating the HTTPS cert flips uvicorn's single listener from
// HTTP to HTTPS on the same port — there's no HTTP port left afterwards to
// poll or to 30x from. So instead of polling same-origin http://, just send
// the browser straight to https:// once the restart has had time to land.
async function doRestartToHttps(ev) {
  const btn = ev && ev.target;
  if (btn) { btn.disabled = true; btn.textContent = 'Restarting…'; }
  try { await postJSON('/restart'); } catch (e) { /* server may drop the connection */ }
  const httpsUrl = `https://${location.hostname}${location.port ? ':' + location.port : ''}/`;
  lifetime.setTimeout(() => { location.href = httpsUrl; }, 4000);
}

  return { doInstall, showProgress, showLoading, retryDownload, saveHfTokenAndRetry,
    doRestart, doRestartToHttps, stopPolling,
    destroy() { stopPolling(); lifetime.destroy(); } };
}

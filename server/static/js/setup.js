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

// Kokoro built-in English voices (static; mirrors core/tts_onnx KOKORO_VOICES).
// af_heart is the recommended default.
const KOKORO_RECOMMENDED = 'af_heart';
const KOKORO_VOICES = [
  'af_heart','af_alloy','af_aoede','af_bella','af_jessica','af_kore',
  'af_nicole','af_nova','af_river','af_sarah','af_sky',
  'am_adam','am_echo','am_eric','am_fenrir','am_liam','am_michael',
  'am_onyx','am_puck','am_santa',
  'bf_alice','bf_emma','bf_isabella','bf_lily',
  'bm_daniel','bm_fable','bm_george','bm_lewis',
];
const kokoroOption = (v, selected) =>
  `<option value="${v}"${v === selected ? ' selected' : ''}>` +
  `${v === KOKORO_RECOMMENDED ? v + ' — recommended' : v}</option>`;

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

let SCHEMA = null, PREFLIGHT = null, STATUS = null;
const sel = {
  tier: 'cpu_local', models: null,
  wakeword: 'hey atticus', wakeword_pattern: '', voice_clone: '',
  openai: { base_url: '', model: '', api_key: '' },
  ha: { url: '', token: '' },
  search_url: '',
};
let tierChosen = false;

const WIZARD_STEPS = ['Brain', 'Set up', 'Download', 'Starting up'];
let curStep = 0;
let cameViaWizard = false;

function renderSteps(activeIdx, mode) {
  const box = $('steps');
  if (mode === 'settings') { box.innerHTML = ''; return; }
  box.innerHTML = '';
  WIZARD_STEPS.forEach((s, i) => {
    const cls = i === activeIdx ? 'active' : (i < activeIdx ? 'done' : '');
    box.appendChild(el(`<span class="s ${cls}">${i + 1}. ${s}</span>`));
  });
}

// ---- entry ----------------------------------------------------------------
async function boot() {
  fetch('/setup/timezone', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tz: Intl.DateTimeFormat().resolvedOptions().timeZone }) }).catch(() => {});
  let status;
  try { status = await getJSON('/status'); }
  catch (e) { screen().innerHTML = `<div class="card">Waiting for server…</div>`; setTimeout(boot, 1500); return; }
  STATUS = status;

  if (status.phase === 'READY') { return openSettings(); }
  if (status.phase === 'DOWNLOADING') { return showProgress(); }
  if (status.phase === 'LOADING') { return showLoading(); }

  $('subtitle').textContent = 'First-run setup';
  SCHEMA = await getJSON('/setup/schema');
  try { PREFLIGHT = await getJSON('/setup/preflight'); } catch (e) { PREFLIGHT = null; }
  if (status.phase === 'ERROR') {
    showAlert(status.detail || 'The current configuration could not start.');
  } else {
    clearAlert();
  }
  // Seed wakeword from any existing config
  const wakeF = SCHEMA.fields.find(f => f.path === 'general.wakeword');
  if (wakeF && wakeF.value) sel.wakeword = wakeF.value;
  const wakePatF = SCHEMA.fields.find(f => f.path === 'general.wakeword_pattern');
  if (wakePatF && wakePatF.value) sel.wakeword_pattern = wakePatF.value;
  // Seed voice from any existing config
  const voiceF = SCHEMA.fields.find(f => f.path === 'general.voice_clone');
  if (voiceF && voiceF.value) sel.voice_clone = voiceF.value;
  // Seed integration fields from existing config
  const haUrl = SCHEMA.fields.find(f => f.path === 'home_assistant.url');
  if (haUrl && haUrl.value) sel.ha.url = haUrl.value;
  const searchUrl = SCHEMA.fields.find(f => f.path === 'search.searxng_url');
  if (searchUrl && searchUrl.value) sel.search_url = searchUrl.value;
  // Seed model tier + backends from any existing `models:` block. Match by
  // backend only (ignoring extra keys like a custom model path) — an exact
  // preset match lets the pretty tier card highlight; anything else falls to
  // 'custom' so sel.models (the real config) is what gets submitted, not a
  // preset's defaults.
  if (SCHEMA.models) {
    sel.models = JSON.parse(JSON.stringify(SCHEMA.models));
    const matched = (SCHEMA.tier_presets || []).find(t =>
      ['asr', 'tts', 'llm'].every(d => (t.models[d] || {}).backend === (sel.models[d] || {}).backend));
    sel.tier = matched ? matched.id : 'custom';
    tierChosen = true;
    if (sel.models.llm && (sel.models.llm.backend === 'openai' || sel.models.llm.backend === 'external')) {
      sel.openai.base_url = sel.models.llm.base_url || '';
      sel.openai.model = sel.models.llm.model || '';
    }
  }
  stepBrain();
}

// ---- step 1: brain --------------------------------------------------------
function fitFor(tierId) {
  if (!PREFLIGHT) return null;
  return (PREFLIGHT.tier_fit || []).find(t => t.id === tierId);
}

function recommendedTierId() {
  const fullFit = fitFor('full');
  const gpu = PREFLIGHT && PREFLIGHT.gpu && PREFLIGHT.gpu.available;
  const fullOffered = (SCHEMA.tier_presets || []).some(t => t.id === 'full' && t.offerable !== false);
  if (gpu && fullOffered && fullFit && fullFit.badge === 'ok') return 'full';
  const staticRec = (SCHEMA.tier_presets || []).find(t => t.recommended);
  return staticRec ? staticRec.id : 'cpu_local';
}

function cpuVariantNotice() {
  if (!SCHEMA || SCHEMA.variant !== 'cpu') return null;
  return el(`<div class="variant-notice">
    <h3>Running on the CPU image</h3>
    <p>Audio (speech recognition + text-to-speech) runs fully local. The language model is either regex-only (simple commands) or off-box via an OpenAI-compatible server you already run.</p>
    <p>For the full stack (voice cloning + 9B SLM on one NVIDIA box), stop this container and run a new one with the <code>:latest</code> tag and <code>--gpus all</code> — see the <strong>GPU</strong> block in the README. The wizard is the same on both images; only the model download differs.</p>
  </div>`);
}

// TLS banner: shown on the first wizard step when the dashboard is
// serving over HTTPS. Pre-empts the browser's self-signed-cert warning
// and gives the user a copyable URL (the user might be reaching the
// dashboard from a phone or another device on the LAN, where the URL
// bar isn't obvious). Dismissable — we remember the dismiss in
// localStorage so returning users don't see it again.
const TLS_BANNER_DISMISS_KEY = 'fulloch.tls_banner_dismissed_v1';
function tlsBanner() {
  if (!STATUS || !STATUS.dashboard_url) return null;
  if (localStorage.getItem(TLS_BANNER_DISMISS_KEY)) return null;
  const url = STATUS.dashboard_url;
  const banner = el(`<div class="banner tls-info" role="note">
    <div class="tls-info-head">
      <strong>Open the dashboard at this URL</strong>
      <button class="banner-dismiss" id="tls-banner-dismiss" type="button" aria-label="Dismiss">✕</button>
    </div>
    <p>Your browser will warn about a self-signed certificate — click through; this is expected for a private LAN install.</p>
    <div class="tls-url-row">
      <code class="tls-url" id="tls-url"></code>
      <button id="tls-url-copy" type="button">Copy</button>
    </div>
  </div>`);
  banner.querySelector('#tls-url').textContent = url;
  banner.querySelector('#tls-banner-dismiss').addEventListener('click', () => {
    try { localStorage.setItem(TLS_BANNER_DISMISS_KEY, '1'); } catch (e) { /* private mode */ }
    banner.remove();
  });
  const copyBtn = banner.querySelector('#tls-url-copy');
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(url);
      const orig = copyBtn.textContent;
      copyBtn.textContent = 'Copied';
      setTimeout(() => { copyBtn.textContent = orig; }, 1500);
    } catch (e) {
      // Clipboard API blocked (insecure context, permissions). Fall back
      // to selecting the text so the user can ⌘C / Ctrl-C it.
      const range = document.createRange();
      range.selectNode(banner.querySelector('#tls-url'));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
  return banner;
}

// Plain-English labels for the wizard (no model sizes in main view)
const TIER_META = {
  'cpu_local':  { icon: '⚡', label: 'Simple commands',  blurb: 'Pattern-matching for smart home, timers, and quick questions. No download, no AI model — instant.' },
  'full':       { icon: '🧠', label: 'Full conversation', blurb: 'A local AI handles anything you ask. Reasons through problems, fully private and offline. Requires a GPU.' },
  'cpu_server': { icon: '🌐', label: 'Remote AI',         blurb: 'Uses an AI server you already run (Ollama, LM Studio, OpenAI). Full conversation without the local download.' },
};

function stepBrain() {
  curStep = 0; renderSteps(0);
  const tiers = SCHEMA.tier_presets.filter(t => t.offerable !== false);
  const recId = recommendedTierId();
  if (!tierChosen) sel.tier = recId;
  const keyInCreds = SCHEMA && SCHEMA.credentials && SCHEMA.credentials.llm_api_key;
  const keyPlaceholder = credPlaceholder(keyInCreds, '(blank for local servers)');

  const c = el(`<div class="card">
    <h2>How should Fulloch think?</h2>
    <p class="lead">Pick a mode. You can change this anytime in settings.</p>
    <div id="tier-list"></div>
    <div id="tier-warn"></div>
    <div id="openai-form" style="display:none;margin-top:0.75rem">
      <label>Server address</label>
      <input type="text" id="oai-url" placeholder="http://localhost:11434/v1">
      <label>Model <span class="muted" style="font-weight:400;font-size:0.8rem">(optional)</span></label>
      <input type="text" id="oai-model" placeholder="leave blank for single-model servers">
      <label>API key <span class="muted" style="font-weight:400;font-size:0.8rem">(optional)</span></label>
      <input type="text" id="oai-key" placeholder="${keyPlaceholder}">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.5rem">
        <button id="oai-test">Test connection</button>
        <span id="oai-status" class="muted"></span>
      </div>
    </div>
    <details class="advanced" id="adv-backends" style="margin-top:0.75rem">
      <summary>Advanced: pick specific backends</summary>
      <div id="backend-cfg"></div>
    </details>
    <div class="actions"><span></span><button class="primary next" id="next1">Next</button></div>
  </div>`);
  screen().innerHTML = '';
  const banner = tlsBanner();
  if (banner) {
    // Render the TLS banner above the card so it's the first thing the
    // user sees on a fresh HTTPS install. The dismiss is per-browser
    // (localStorage) — once acknowledged, the banner is gone for good.
    screen().appendChild(banner);
  }
  screen().appendChild(c);

  const list = $('tier-list');
  tiers.forEach(t => {
    const meta = TIER_META[t.id] || { icon: '⚙', label: t.label, blurb: t.blurb };
    const rec = t.id === recId ? '<span class="badge rec">recommended</span>' : '';
    const node = el(`<div class="opt ${sel.tier === t.id ? 'sel' : ''}" data-tier="${t.id}">
      <div class="row"><span class="name">${meta.icon} ${meta.label} ${rec}</span></div>
      <div class="blurb">${meta.blurb}</div></div>`);
    node.addEventListener('click', () => {
      sel.tier = t.id; sel.models = null; tierChosen = true;
      document.querySelectorAll('#tier-list .opt').forEach(o => o.classList.toggle('sel', o.dataset.tier === t.id));
      syncOpenaiForm();
      updateTierWarn(t.id);
    });
    list.appendChild(node);
  });

  renderBackendCfg();
  if (sel.tier === 'custom') $('adv-backends').setAttribute('open', '');
  $('oai-url').value = sel.openai.base_url;
  $('oai-model').value = sel.openai.model;
  $('oai-key').value = sel.openai.api_key;
  ['oai-url', 'oai-model', 'oai-key'].forEach(id => $(id).addEventListener('input', () => {
    sel.openai.base_url = $('oai-url').value.trim();
    sel.openai.model = $('oai-model').value.trim();
    sel.openai.api_key = $('oai-key').value.trim();
  }));
  $('oai-url').addEventListener('blur', () => {
    const n = normalizeEndpointUrl($('oai-url').value);
    if (n !== $('oai-url').value) { $('oai-url').value = n; sel.openai.base_url = n; }
  });
  $('oai-test').addEventListener('click', async () => {
    const n = normalizeEndpointUrl($('oai-url').value);
    if (n !== $('oai-url').value) { $('oai-url').value = n; sel.openai.base_url = n; }
    const s = $('oai-status'); s.textContent = 'Testing…';
    const r = await postJSON('/setup/test-llm', {
      base_url: sel.openai.base_url || 'http://localhost:11434/v1',
      model: sel.openai.model, api_key: sel.openai.api_key,
    });
    const j = await r.json();
    s.textContent = j.ok ? '✓ reachable' : ('✗ ' + (j.error || 'failed'));
    s.style.color = j.ok ? 'var(--primary)' : 'var(--error)';
  });

  syncOpenaiForm();
  updateTierWarn(sel.tier);
  $('next1').addEventListener('click', stepSetup);
}

function updateTierWarn(tierId) {
  const warn = $('tier-warn');
  if (!warn) return;
  const fit = fitFor(tierId);
  if (!fit || fit.badge !== 'warn') { warn.innerHTML = ''; return; }
  const r = fit.reason || '';
  let msg;
  if (r.includes('RAM') || r.includes('memory')) {
    const need = fit.ram_gb ? `${fit.ram_gb}GB` : 'more';
    msg = `Not enough memory allocated to Docker for this option. `
        + `In <b>Docker Desktop → Settings → Resources → Memory</b> set it to at least <b>${need}</b>, `
        + `then restart the container.`;
  } else if (r.includes('disk')) {
    msg = `Not enough free disk space. Free up some space and try again.`;
  } else if (r.includes('GPU') || r.includes('VRAM')) {
    msg = `This option needs a GPU that isn't available or doesn't have enough VRAM.`;
  } else {
    msg = r;
  }
  warn.innerHTML = `<div class="banner warn" style="margin-top:0.75rem">${msg}</div>`;
}

function renderBackendCfg() {
  const box = $('backend-cfg');
  const b = SCHEMA.backends;
  const curBackend = (domain) => (sel.models && sel.models[domain] && sel.models[domain].backend) || '';
  const mk = (domain, label) => {
    const cur = curBackend(domain);
    if (domain === 'llm') {
      const mode = cur === 'openai' || cur === 'external' ? 'external' : 'local';
      return `<div><label>${label}</label><select id="be-llm">
        <option value="local"${mode === 'local' ? ' selected' : ''}>Local</option>
        <option value="external"${mode === 'external' ? ' selected' : ''}>External</option>
      </select></div>`;
    }
    const opts = b[domain].filter(o => o.offerable).map(o => {
      const exp = o.experimental ? ' [experimental]' : '';
      return `<option value="${o.backend}"${o.backend === cur ? ' selected' : ''}>${o.display_name}${exp}</option>`;
    }).join('');
    return `<div><label>${label}</label><select id="be-${domain}">${opts}</select></div>`;
  };
  box.innerHTML = `<p class="help">Overrides the mode selected above.</p>
    <div class="grid2">${mk('asr','Speech-to-text')}${mk('tts','Text-to-speech')}${mk('llm','Language model')}</div>`;
  const syncCustom = () => {
    sel.tier = 'custom';
    const llmMode = $('be-llm').value;
    sel.models = { asr: { backend: $('be-asr').value }, tts: { backend: $('be-tts').value },
                   llm: llmMode === 'local'
                     ? { backend: 'local', local_model: 'qwen' }
                     : { backend: 'external' } };
    document.querySelectorAll('#tier-list .opt').forEach(o => o.classList.remove('sel'));
    syncOpenaiForm();
  };
  ['asr','tts','llm'].forEach(d => $(`be-${d}`).addEventListener('change', syncCustom));
  const adv = $('adv-backends');
  if (adv) adv.addEventListener('toggle', () => { if (adv.open) syncCustom(); });
}

function chosenModels() {
  let m;
  if (sel.tier === 'custom' && sel.models) m = JSON.parse(JSON.stringify(sel.models));
  else { const t = SCHEMA.tier_presets.find(x => x.id === sel.tier); m = t ? JSON.parse(JSON.stringify(t.models)) : null; }
  if (m && m.llm && (m.llm.backend === 'openai' || m.llm.backend === 'external')) {
    if (sel.openai.base_url) m.llm.base_url = normalizeEndpointUrl(sel.openai.base_url);
    if (sel.openai.model) m.llm.model = sel.openai.model;
    // api_key goes to credentials.json, not models config
  }
  return m;
}
function ttsBackend() { const m = chosenModels(); return m && m.tts ? m.tts.backend : 'qwen'; }

function syncOpenaiForm() {
  const m = chosenModels();
  const isOai = !!(m && m.llm && (m.llm.backend === 'openai' || m.llm.backend === 'external'));
  const form = $('openai-form');
  if (form) form.style.display = isOai ? '' : 'none';
  setBranding(isOai);
}

function setBranding(remote) {
  const img = document.getElementById('brand-logo');
  if (img) img.src = remote ? '/logo.png?remote=1' : '/logo.png';
}

// ---- step 2: set up (wakeword + voice) -------------------------------------
// HA + SearXNG used to live here as collapsed <details> cards. They were
// moved to a dedicated sub-step (`stepConnect`) so the "skip by default"
// treatment is the page's primary affordance instead of a footnote — the
// step's own Skip button (and an explicit "this is optional" lead) make
// the choice obvious. Task 4 of docs/ease-of-use-tasks.md.
async function stepSetup() {
  curStep = 1; renderSteps(1);
  const isKokoro = ttsBackend() === 'kokoro-onnx';
  const def = KOKORO_VOICES.includes(sel.voice_clone) ? sel.voice_clone : KOKORO_RECOMMENDED;
  const kokoroOpts = KOKORO_VOICES.map(v => kokoroOption(v, def)).join('');
  const presets = SCHEMA.wakeword_presets;
  const customWake = presets.some(p => p.wakeword === sel.wakeword) ? '' : sel.wakeword;

  const c = el(`<div class="card">
    <h2>Set up your assistant</h2>
    <p class="lead">Everything has a sensible default — pick a name and voice, then click <strong>Get started</strong>.</p>

    <div class="section-title">What should you call it?</div>
    <div id="wake-list"></div>
    <details class="advanced"${customWake ? ' open' : ''}><summary>Custom name</summary>
      <input type="text" id="wake-custom" placeholder="e.g. hey jarvis" value="${customWake.replace(/"/g, '&quot;')}" style="margin-top:0.5rem">
      <div class="help">A tolerant pattern is built automatically.</div>
    </details>

    <div class="section-title">Voice</div>
    ${isKokoro
      ? `<div class="voice-row"><select id="voice-sel">${kokoroOpts}</select></div>`
      : `<div class="voice-row"><select id="voice-sel"></select></div>
         <button id="gen-voice" style="margin-top:0.5rem">+ Generate new voice clone</button>
         <div id="gen-panel"></div>`
    }

    <div class="actions">
      <button id="back2" class="back">Back</button>
      <button class="primary" id="get-started">Get started</button>
    </div>
  </div>`);
  screen().innerHTML = ''; screen().appendChild(c);

  // --- wakeword ---
  const wakeList = $('wake-list');
  presets.forEach(p => {
    const rec = p.recommended ? '<span class="badge rec">recommended</span>' : '';
    const node = el(`<div class="opt ${sel.wakeword === p.wakeword ? 'sel' : ''}" data-wake="${p.wakeword}" data-pattern="${p.pattern.replace(/"/g, '&quot;')}">
      <div class="row"><span class="name">${p.label} ${rec}</span></div></div>`);
    node.addEventListener('click', () => {
      sel.wakeword = p.wakeword; sel.wakeword_pattern = p.pattern;
      $('wake-custom').value = '';
      document.querySelectorAll('#wake-list .opt').forEach(o => o.classList.toggle('sel', o.dataset.wake === p.wakeword));
    });
    wakeList.appendChild(node);
  });
  $('wake-custom').addEventListener('input', (e) => {
    const v = e.target.value.trim();
    if (v) { sel.wakeword = v; sel.wakeword_pattern = ''; document.querySelectorAll('#wake-list .opt').forEach(o => o.classList.remove('sel')); }
  });

  // --- voice ---
  _voiceStop();
  if (isKokoro) {
    $('voice-sel').value = def; sel.voice_clone = def;
    $('voice-sel').addEventListener('change', e => sel.voice_clone = e.target.value);
    $('voice-sel').parentElement.appendChild(makeVoicePreview($('voice-sel')));
  } else {
    await refreshVoiceList();
    $('voice-sel').parentElement.appendChild(makeVoicePreview($('voice-sel')));
    $('gen-voice').addEventListener('click', showGenPanel);
  }

  $('back2').addEventListener('click', stepBrain);
  // "Get started" goes to the new "Connect (optional)" sub-step where HA
  // and SearXNG are configured. Both cards default-collapsed, and the
  // sub-step's own Skip button drops the user straight into Obsidian
  // (which itself has a Skip). Task 4 of docs/ease-of-use-tasks.md.
  $('get-started').addEventListener('click', stepConnect);
}

// ---- step 2.4: connect HA + SearXNG (optional) ----------------------------
// Same sub-step shape as Obsidian (step 2.5): one screen, one Skip button,
// every card collapsed by default. The "skip by default" treatment matches
// the pattern Obsidian already uses — the user only expands a card if
// they want to configure that integration right now.
function stepConnect() {
  const haUrl = (sel.ha.url || '').replace(/"/g, '&quot;');
  const haToken = (sel.ha.token || '').replace(/"/g, '&quot;');
  const haTokenSet = SCHEMA.credentials && SCHEMA.credentials.ha_token;
  const haTokenPh = credPlaceholder(haTokenSet, 'create one in HA → Profile → Security');
  const searchUrl = (sel.search_url || '').replace(/"/g, '&quot;');

  const c = el(`<div class="card">
    <h2>Connect (optional)</h2>
    <p class="lead">Skip this — your assistant works fine without any of these. Expand a card only if you want to set it up now; you can add them later from the settings console.</p>

    <details id="ha-section" class="connect-section"${haUrl ? ' open' : ''}>
      <summary>
        <span class="csname">🏠 Home Assistant</span>
        <span class="cstatus" id="ha-cstatus"></span>
      </summary>
      <div class="cbody">
        <p class="help">Control lights, media, climate, and more by voice.</p>
        <label>URL</label>
        <input type="text" id="ha-url" placeholder="http://homeassistant.local:8123" value="${haUrl}">
        <label>Long-lived access token</label>
        <input type="text" id="ha-token" placeholder="${haTokenPh}" value="${haToken}">
        <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.6rem">
          <button id="ha-test">Test</button>
          <span id="ha-status" class="muted"></span>
        </div>
      </div>
    </details>

    <details id="search-section" class="connect-section"${searchUrl ? ' open' : ''}>
      <summary>
        <span class="csname">🔍 Web search</span>
        <span class="cstatus" id="search-cstatus"></span>
      </summary>
      <div class="cbody">
        <p class="help">Live web answers, summarised into a short spoken reply.</p>
        <label>SearXNG URL</label>
        <input type="text" id="search-url" placeholder="http://localhost:8080" value="${searchUrl}">
        <p class="help" style="margin-top:0.4rem">To run a local SearXNG: <code style="font-family:monospace;font-size:0.78rem">docker run -d --name searxng -p 8080:8080 -e SEARXNG_SECRET=change-me searxng/searxng</code>, then enter <code style="font-family:monospace;font-size:0.78rem">http://localhost:8080</code> above.</p>
      </div>
    </details>

    <div class="actions">
      <button id="connect-back" class="back">Back</button>
      <span style="display:flex;gap:0.5rem">
        <button id="connect-skip">Skip</button>
        <button class="primary" id="connect-next">Next</button>
      </span>
    </div>
  </div>`);
  screen().innerHTML = ''; screen().appendChild(c);

  // --- HA ---
  $('ha-url').addEventListener('input', () => sel.ha.url = $('ha-url').value.trim());
  $('ha-token').addEventListener('input', () => sel.ha.token = $('ha-token').value.trim());
  $('ha-test').addEventListener('click', async () => {
    const s = $('ha-status'); s.textContent = 'Testing…'; s.style.color = '';
    try {
      const r = await postJSON('/setup/test-ha', { url: $('ha-url').value.trim(), token: $('ha-token').value.trim() });
      const j = await r.json();
      if (j.ok) {
        s.textContent = '✓ connected'; s.style.color = 'var(--primary)';
        $('ha-cstatus').textContent = 'connected'; $('ha-cstatus').style.color = 'var(--primary)';
      } else {
        s.textContent = '✗ ' + (j.error || 'unreachable'); s.style.color = 'var(--error)';
        $('ha-cstatus').textContent = 'not connected'; $('ha-cstatus').style.color = 'var(--error)';
      }
    } catch { s.textContent = '✗ unreachable'; s.style.color = 'var(--error)'; }
  });

  // --- Search ---
  $('search-url').addEventListener('input', () => sel.search_url = $('search-url').value.trim());

  $('connect-back').addEventListener('click', stepSetup);
  $('connect-skip').addEventListener('click', stepObsidian);
  $('connect-next').addEventListener('click', stepObsidian);
}

// ---- step 2.5: connect Obsidian (optional) ---------------------------------
async function stepObsidian() {
  const c = el(`<div class="card">
    <h2>Connect Obsidian (optional)</h2>
    <p class="lead">If you use Obsidian to manage your notes, Fulloch can read and write directly to your vault. Skip this to keep Fulloch's default notes folder.</p>

    <label for="obsidian-vault-path">Obsidian vault path</label>
    <input type="text" id="obsidian-vault-path" placeholder="/home/you/Documents/MyVault">
    <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.6rem">
      <button id="obsidian-detect" type="button">Auto-detect</button>
      <span id="obsidian-status" class="muted"></span>
    </div>
    <p class="help" id="obsidian-hint" style="margin-top:0.4rem">Auto-detect scans <code>~/Documents</code>, <code>~/Obsidian</code>, and <code>~/.config/obsidian</code> for vaults. You can also type the path manually.</p>

    <div class="actions">
      <button id="obsidian-back" class="back">Back</button>
      <span style="display:flex;gap:0.5rem">
        <button id="obsidian-skip">Skip</button>
        <button class="primary" id="obsidian-save">Save and continue</button>
      </span>
    </div>
  </div>`);
  screen().innerHTML = ''; screen().appendChild(c);

  $('obsidian-back').addEventListener('click', stepConnect);
  $('obsidian-skip').addEventListener('click', doInstall);
  $('obsidian-detect').addEventListener('click', async () => {
    const status = $('obsidian-status'); status.textContent = 'Scanning…'; status.style.color = '';
    const hint = $('obsidian-hint');
    try {
      const r = await postJSON('/api/setup/detect-obsidian-vaults', {});
      const j = await r.json();
      const candidates = (j && j.candidates) || [];
      if (candidates.length > 0) {
        $('obsidian-vault-path').value = candidates[0].path;
        status.textContent = `✓ found ${candidates.length}`;
        status.style.color = 'var(--primary)';
        if (candidates.length === 1) {
          hint.textContent = `Found 1 vault: ${candidates[0].name}.`;
        } else {
          hint.innerHTML = `Found ${candidates.length} candidate(s). First: <code>${candidates[0].path}</code> — edit if you'd like a different one.`;
        }
      } else {
        status.textContent = 'no vaults found';
        status.style.color = 'var(--text-muted)';
        hint.textContent = 'No vault auto-detected. Enter the path to your vault (the folder that contains a .obsidian/ subfolder) manually.';
      }
    } catch (e) {
      status.textContent = '✗ scan failed'; status.style.color = 'var(--error)';
    }
  });
  $('obsidian-save').addEventListener('click', async () => {
    const path = ($('obsidian-vault-path').value || '').trim();
    const hint = $('obsidian-hint');
    if (!path) {
      hint.textContent = 'Enter a vault path or click Skip.';
      return;
    }
    const status = $('obsidian-status'); status.textContent = 'Saving…'; status.style.color = '';
    const r = await postJSON('/api/setup/obsidian-vault', { path });
    if (r.ok) {
      doInstall();
    } else {
      const err = await r.json().catch(() => ({}));
      status.textContent = '✗ ' + (err.detail || 'invalid vault path');
      status.style.color = 'var(--error)';
      hint.textContent = err.detail || "That path isn't a vault. The folder must contain a .obsidian/ subfolder.";
    }
  });
}

// --- voice preview button bound to a <select> ---
let _voiceAudio = null, _voiceBtn = null;
const _voiceIcon = (paused = false) => paused
  ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3v14H7zm7 0h3v14h-3z"></path></svg>'
  : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10-6.5z"></path></svg>';
function _voiceStop() {
  if (_voiceAudio) { _voiceAudio.pause(); _voiceAudio = null; }
  if (_voiceBtn) { _voiceBtn.innerHTML = _voiceIcon(); _voiceBtn = null; }
}
function makeVoicePreview(selectEl) {
  const btn = el(`<button type="button" class="voice-play" title="Preview voice" aria-label="Preview voice">${_voiceIcon()}</button>`);
  btn.addEventListener('click', () => {
    const togglingOff = _voiceBtn === btn && _voiceAudio && !_voiceAudio.paused;
    _voiceStop();
    if (togglingOff || !selectEl.value) return;
    const url = '/voice/sample?name=' + encodeURIComponent(selectEl.value);
    const a = new Audio(url);
    _voiceAudio = a; _voiceBtn = btn; btn.innerHTML = _voiceIcon(true);
    const reset = () => { if (_voiceBtn === btn) _voiceStop(); };
    a.onended = reset; a.onerror = reset;
    a.play().catch(reset);
  });
  selectEl.addEventListener('change', _voiceStop);
  return btn;
}

async function refreshVoiceList(selectName) {
  const { voices } = await getJSON('/setup/voices');
  const s = $('voice-sel');
  s.innerHTML = (voices.length ? voices : ['atticus']).map(v => `<option value="${v}">${v}</option>`).join('');
  const pick = selectName || (voices.includes(sel.voice_clone) ? sel.voice_clone : voices[0]) || 'atticus';
  s.value = pick; sel.voice_clone = pick;
  s.onchange = e => sel.voice_clone = e.target.value;
}

function showGenPanel() {
  const p = $('gen-panel');
  p.innerHTML = `<div class="opt" style="cursor:default;margin-top:0.75rem">
    <label>Describe the voice</label>
    <textarea id="gv-instruct" placeholder="A warm, friendly Australian woman in her 30s, relaxed pace."></textarea>
    <label>Phrase to speak (optional)</label>
    <input type="text" id="gv-phrase" placeholder="(uses a default sentence if blank)">
    <div class="actions"><button id="gv-generate" class="primary">Generate preview</button><span id="gv-status" class="muted"></span></div>
    <div id="gv-audio"></div>
    <div id="gv-save" style="display:none">
      <label>Save as</label><input type="text" id="gv-name" placeholder="my-voice">
      <div class="actions"><span></span><button id="gv-save-btn" class="primary">Save voice</button></div>
    </div></div>`;
  $('gv-generate').addEventListener('click', generateVoice);
  $('gv-save-btn').addEventListener('click', saveVoice);
}

async function generateVoice() {
  const instruct = $('gv-instruct').value.trim();
  if (!instruct) { $('gv-status').textContent = 'Enter a description.'; return; }
  $('gv-status').textContent = 'Generating… (this can take a while)';
  $('gv-generate').disabled = true;
  try {
    const r = await postJSON('/setup/voice', { instruct, phrase: $('gv-phrase').value.trim() });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.status); }
    const blob = await r.blob();
    $('gv-audio').innerHTML = '';
    const audio = el('<audio controls autoplay></audio>');
    audio.src = URL.createObjectURL(blob);
    $('gv-audio').appendChild(audio);
    $('gv-save').style.display = '';
    $('gv-status').textContent = 'Preview ready.';
  } catch (e) {
    $('gv-status').textContent = 'Generation failed: ' + e.message;
  } finally { $('gv-generate').disabled = false; }
}

async function saveVoice() {
  const name = $('gv-name').value.trim();
  if (!name) return;
  const r = await postJSON('/setup/voice/save', { name });
  if (!r.ok) { const e = await r.json().catch(() => ({})); alert('Save failed: ' + (e.detail || r.status)); return; }
  const { saved } = await r.json();
  $('gen-panel').innerHTML = '';
  await refreshVoiceList(saved);
}

// ---- install ---------------------------------------------------------------
async function doInstall() {
  const btn = $('get-started');
  if (btn) btn.disabled = true;
  clearAlert();
  // Blocking preflight before the model download starts: disk space,
  // network reach to the model hub, and (for GPU tiers) an NVIDIA GPU
  // being visible. Each failed check becomes one bullet in the error
  // pane so the user knows exactly which to fix. Task 3 of
  // docs/ease-of-use-tasks.md.
  const pre = await postJSON('/setup/preflight-download');
  const preBody = await pre.json().catch(() => ({ ok: true, errors: [] }));
  if (!pre.ok) {
    showPreflightErrors(preBody.errors || []);
    if (btn) btn.disabled = false;
    return;
  }
  const models = chosenModels();
  await postJSON('/setup/models', { models });
  const updates = {
    'general.wakeword': sel.wakeword,
    'general.wakeword_pattern': sel.wakeword_pattern || '',
    'general.voice_clone': sel.voice_clone || '',
  };
  if (sel.ha.url) updates['home_assistant.url'] = sel.ha.url;
  if (sel.search_url) updates['search.searxng_url'] = sel.search_url;
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
  cameViaWizard = true;
  showProgress();
}

// ---- download progress ----------------------------------------------------
function showProgress() {
  curStep = 2; renderSteps(2);
  clearAlert();
  screen().innerHTML = `<div class="card"><h2>Downloading models</h2>
    <p class="lead">Pulling model weights. You can leave this page open.</p>
    <div id="assets"></div><div id="dl-error"></div></div>`;
  pollProgress();
}

async function pollProgress() {
  let snap;
  try { snap = await getJSON('/setup/progress'); } catch (e) { setTimeout(pollProgress, 1500); return; }
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
    $('dl-error').innerHTML = `<div class="banner error">${snap.error || 'Download failed.'}
      <button style="margin-left:1rem" onclick="retryDownload()">Retry download</button></div>`;
    return;
  }
  if (snap.state === 'done') { return showLoading(); }
  setTimeout(pollProgress, 1200);
}

async function retryDownload() {
  $('dl-error').innerHTML = '';
  await postJSON('/setup/retry-download', {});
  pollProgress();
}

// ---- loading (models loading into memory) ---------------------------------
let loadSeenSeq = 0;

function showLoading() {
  curStep = 3;
  loadSeenSeq = 0;
  clearAlert();
  $('subtitle').textContent = cameViaWizard ? 'First-run setup' : 'Starting up';
  renderSteps(3, cameViaWizard ? undefined : 'settings');
  screen().innerHTML = `<div class="card"><h2>Starting up</h2>
    <p class="lead">Loading models and warming up prompts — the assistant will be ready shortly.</p>
    <div id="term" class="term" aria-live="polite"></div>
    <div class="term-foot"><span class="spinner sm"></span><span class="muted" id="load-detail"></span></div></div>`;
  pollLoading();
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

async function pollLoading() {
  let s;
  try { s = await getJSON('/status'); } catch (e) { setTimeout(pollLoading, 1200); return; }
  streamLogLines(s.log);
  $('load-detail') && ($('load-detail').textContent = s.detail || '');
  if (s.phase === 'READY') { return cameViaWizard ? stepFinish() : (location.href = '/'); }
  if (s.phase === 'ERROR') { return boot(); }
  setTimeout(pollLoading, 900);
}

// ---- token step -----------------------------------------------------------
function stepFinish() {
  renderSteps(3, 'settings');
  const c = el(`<div class="card">
    <h2>Almost done</h2>
    <p class="lead">Give the assistant a name to call you, and optionally set a password to protect the dashboard when it's reachable on your network.</p>

    <label for="finish-name">Your name <span style="font-weight:400;font-size:.8rem;color:var(--text-muted)">(optional)</span></label>
    <input type="text" id="finish-name" placeholder="e.g. Alex" autocomplete="name">

    <label for="finish-pw" style="margin-top:1.1rem">Dashboard password <span style="font-weight:400;font-size:.8rem;color:var(--text-muted)">(optional — leave blank for local-only)</span></label>
    <input type="password" id="finish-pw" placeholder="Choose a password" autocomplete="new-password">
    <input type="password" id="finish-pw2" placeholder="Confirm password" autocomplete="new-password" style="margin-top:.4rem">
    <p style="font-size:.8rem;color:var(--text-muted);margin:.5rem 0 0">If set, you'll log in with this password from any device on your network.</p>

    <div id="finish-err" class="banner error" style="display:none;margin-top:.75rem"></div>
    <div class="actions" style="justify-content:flex-start;gap:.75rem;margin-top:1.25rem">
      <button class="primary" id="finish-btn">Go to dashboard →</button>
    </div>
  </div>`);
  screen().innerHTML = '';
  screen().appendChild(c);

  $('finish-btn').addEventListener('click', async () => {
    const name  = ($('finish-name').value || '').trim();
    const pw    = ($('finish-pw').value  || '').trim();
    const pw2   = ($('finish-pw2').value || '').trim();
    const errEl = $('finish-err');
    errEl.style.display = 'none';

    if (pw && pw !== pw2) {
      errEl.textContent = 'Passwords do not match.';
      errEl.style.display = '';
      return;
    }
    if (pw && pw.length < 8) {
      errEl.textContent = 'Password must be at least 8 characters.';
      errEl.style.display = '';
      return;
    }

    $('finish-btn').disabled = true;
    const r = await postJSON('/setup/password', { name: name || null, password: pw || null });
    if (!r.ok) {
      errEl.textContent = 'Could not save — try again.';
      errEl.style.display = '';
      $('finish-btn').disabled = false;
      return;
    }
    // A password was set → must log in; otherwise go straight to the dashboard.
    location.href = pw ? '/login' : '/';
  });
}

// ---- settings console (post-setup) ----------------------------------------
async function openSettings() {
  renderSteps(0, 'settings');
  $('subtitle').textContent = 'Settings';
  SCHEMA = await getJSON('/setup/schema');
  screen().innerHTML = '';

  const bar = el(`<div class="topbar">
    <button class="back" id="settings-back">Back to dashboard</button>
    <span class="spacer"></span><h2>Settings</h2></div>`);
  screen().appendChild(bar);
  $('settings-back').addEventListener('click', () => location.href = '/');

  // CPU-image banner
  {
    const vn = cpuVariantNotice();
    if (vn) screen().appendChild(vn);
  }

  // Integrations card (quick access to HA/Notes/Search with test buttons)
  screen().appendChild(integrationsCard());
  wireIntegrationsCard();

  // Models card (speech + language backends)
  screen().appendChild(modelsCard());
  wireModelsCard();

  // Security card (password change + obsidian token)
  screen().appendChild(securityCard());
  wireSecurityCard();

  // Full config console
  const byGroup = {};
  SCHEMA.fields.forEach(f => { (byGroup[f.group] = byGroup[f.group] || []).push(f); });
  const c = el(`<div class="card"><h2>Configuration</h2>
    <p class="lead">Every <code>config.yml</code> value is editable here. Hover the <span class="info">i</span> on any field for help. Restart-flagged changes need a restart.</p>
    <div id="cfg-form"></div>
    <div id="save-note"></div>
    <div class="actions"><button class="back" id="cfg-back">Back to dashboard</button><button class="primary" id="save-cfg">Save changes</button></div>
    </div>`);
  screen().appendChild(c);
  $('cfg-back').addEventListener('click', () => location.href = '/');
  const form = $('cfg-form');
  SCHEMA.groups.forEach(g => {
    if (!byGroup[g]) return;
    form.appendChild(el(`<div class="group-title">${g}</div>`));
    byGroup[g].forEach(f => { const n = fieldRow(f); if (n) form.appendChild(n); });
  });
  $('save-cfg').addEventListener('click', saveSettings);
  CFG_INITIAL = {};
  document.querySelectorAll('#cfg-form [data-path]').forEach(node => {
    CFG_INITIAL[node.dataset.path] = node.dataset.ghost === '1' ? '' : node.value;
  });
  populateVoiceField();

  const dz = el(`<div class="card">
    <h2>Re-run setup</h2>
    <p class="lead">Start the setup wizard again — re-pick tier, models, wakeword and voice. Your settings, credentials, Obsidian link, voice clones and entity denylist are backed up first; downloaded models and the HTTPS cert are kept. Takes effect after a restart.</p>
    <div id="reset-note"></div>
    <div class="actions"><span></span><button class="danger" id="reset-setup">Re-run setup wizard…</button></div>
    <div id="backup-list-wrap"></div>
    </div>`);
  screen().appendChild(dz);
  $('reset-setup').addEventListener('click', resetSetup);
  loadBackupList();
}

// Security card — password change + Obsidian linkage + HTTPS cert
function securityCard() {
  const certField = SCHEMA.fields.find(f => f.section === 'general' && f.name === 'dashboard_ssl_certfile');
  const certEnabled = !!(certField && certField.value);
  return el(`<div class="card"><h2>Security &amp; Access</h2>
    <p class="lead">Manage the dashboard password, Obsidian plugin token, and HTTPS certificate.</p>

    <details class="connect-section" id="sec-pw-section">
      <summary><span class="csname">🔑 Dashboard password</span></summary>
      <div class="cbody">
        <label>New password</label><input type="password" id="sec-pw" placeholder="min 8 characters" autocomplete="new-password">
        <label style="margin-top:.4rem">Confirm password</label><input type="password" id="sec-pw2" placeholder="repeat password" autocomplete="new-password" style="margin-top:.4rem">
        <div id="sec-pw-err" class="banner error" style="display:none;margin-top:.5rem"></div>
        <button id="sec-pw-save" style="margin-top:.75rem">Update password</button>
        <span id="sec-pw-status" class="muted" style="margin-left:.75rem"></span>
      </div>
    </details>

    <details class="connect-section" id="sec-obs-section" style="margin-top:.75rem">
      <summary><span class="csname">📓 Obsidian</span><span class="cstatus" id="sec-obs-cstatus">loading…</span></summary>
      <div class="cbody">
        <p class="help" style="margin:0 0 .65rem">Connect Fulloch to an Obsidian vault for voice read/write. The plugin reports your vault and any open note. Cloud sync (Remotely Save, etc.) is unchanged — Fulloch only sees the local vault.</p>

        <div class="obs-status-row" style="margin-bottom: .75rem">
          <span class="obs-pill" id="sec-obs-pill">—</span>
          <span class="obs-status-detail" id="sec-obs-status-detail"></span>
        </div>

        <div class="group-title" style="margin-top: 0">Vault</div>
        <label for="sec-obs-vault-path">Obsidian vault path</label>
        <div class="obs-switch" style="margin-top:.35rem">
          <input type="text" id="sec-obs-vault-path" placeholder="/home/you/Documents/MyVault">
        </div>
        <div style="display:flex;align-items:center;gap:.5rem;margin-top:.5rem;flex-wrap:wrap">
          <button id="sec-obs-detect" type="button">Auto-detect</button>
          <button class="primary" id="sec-obs-save-vault" type="button">Save</button>
          <span id="sec-obs-vault-status" class="muted"></span>
        </div>
        <p class="help" id="sec-obs-vault-hint" style="margin-top:.4rem">Path to the folder that contains your <code>.obsidian/</code> subfolder. Auto-detect scans <code>~/Documents</code>, <code>~/Obsidian</code>, and <code>~/.config/obsidian</code>.</p>

        <div class="group-title">Plugin</div>
        <p class="help" style="margin:0 0 .5rem">The plugin ships in the repo. Until it's in the Obsidian community store, download the zip and extract it into <code>&lt;vault&gt;/.obsidian/plugins/fulloch/</code>, then enable it in <strong>Settings → Community plugins</strong>.</p>
        <a class="obs-btn" href="/api/obsidian/plugin.zip" download>Download plugin.zip</a>

        <div class="group-title">Auth token</div>
        <div class="obs-token" style="margin-top:.35rem">
          <code id="sec-obs-token">—</code>
          <button class="obs-btn ghost" id="sec-obs-copy-token" type="button">Copy</button>
          <button class="obs-btn danger" id="sec-obs-regen-token" type="button">Regenerate</button>
        </div>
        <p class="help" style="margin-top:.4rem">Paste this into the Fulloch plugin settings. Rotating the token drops the plugin connection within 10 seconds.</p>
        <span id="sec-obs-status" class="muted" style="display:block;margin-top:.35rem"></span>
      </div>
    </details>

    <details class="connect-section" id="sec-cert-section" style="margin-top:.75rem">
      <summary><span class="csname">🔒 HTTPS certificate</span><span class="cstatus" id="sec-cert-cstatus">${certEnabled ? 'enabled' : ''}</span></summary>
      <div class="cbody">
        <p style="font-size:.8rem;color:var(--text-muted);margin:0 0 .6rem">${certEnabled
          ? 'Self-signed certificate used for LAN HTTPS (needed for mic access from phones and other devices). Regenerate it if your LAN IP changed and the old certificate no longer covers it. Every device that trusted the old one will see the browser warning again. Takes effect after a restart.'
          : 'HTTPS isn\'t enabled for this install. Browsers refuse microphone access on plain HTTP for anything but localhost, so phones and other LAN devices can\'t use the mic without it. Generating a self-signed certificate fixes that — every browser shows a one-time "not private" warning on first visit, which is expected. Takes effect after a restart.'}</p>
        <button id="sec-cert-regen">${certEnabled ? 'Regenerate certificate…' : 'Enable HTTPS…'}</button>
        <span id="sec-cert-status" class="muted" style="margin-left:.75rem"></span>
      </div>
    </details>
    <div id="security-note"></div>
  </div>`);
}

function wireSecurityCard() {
  $('sec-pw-save').addEventListener('click', async () => {
    const pw = ($('sec-pw').value || '').trim();
    const pw2 = ($('sec-pw2').value || '').trim();
    const errEl = $('sec-pw-err');
    const status = $('sec-pw-status');
    errEl.style.display = 'none';
    if (!pw) { errEl.textContent = 'Enter a new password.'; errEl.style.display = ''; return; }
    if (pw !== pw2) { errEl.textContent = 'Passwords do not match.'; errEl.style.display = ''; return; }
    if (pw.length < 8) { errEl.textContent = 'Password must be at least 8 characters.'; errEl.style.display = ''; return; }
    $('sec-pw-save').disabled = true;
    const r = await postJSON('/setup/password', { password: pw });
    $('sec-pw-save').disabled = false;
    if (r.ok) {
      status.textContent = '✓ password updated';
      status.style.color = 'var(--primary)';
      $('sec-pw').value = ''; $('sec-pw2').value = '';
    } else {
      errEl.textContent = 'Could not save — try again.'; errEl.style.display = '';
    }
  });
  // Obsidian linkage: vault path + auto-detect + save, plugin download,
  // auth token copy/regenerate. Mirrors the dashboard Obsidian tab so a user
  // who already has a manual config.yml can wire up Obsidian from here.
  $('sec-obs-detect').addEventListener('click', async () => {
    const status = $('sec-obs-vault-status');
    status.textContent = 'Scanning…'; status.style.color = '';
    const hint = $('sec-obs-vault-hint');
    try {
      const r = await postJSON('/api/setup/detect-obsidian-vaults', {});
      const j = await r.json();
      const candidates = (j && j.candidates) || [];
      if (candidates.length > 0) {
        $('sec-obs-vault-path').value = candidates[0].path;
        status.textContent = `✓ found ${candidates.length}`;
        status.style.color = 'var(--primary)';
        if (candidates.length === 1) {
          hint.textContent = `Found 1 vault: ${candidates[0].name}.`;
        } else {
          hint.innerHTML = `Found ${candidates.length} candidate(s). First: <code>${candidates[0].path}</code> — edit if you'd like a different one.`;
        }
      } else {
        status.textContent = 'no vaults found';
        status.style.color = 'var(--text-muted)';
        hint.textContent = 'No vault auto-detected. Enter the path to your vault (the folder that contains a .obsidian/ subfolder) manually.';
      }
    } catch (e) {
      status.textContent = '✗ scan failed'; status.style.color = 'var(--error)';
    }
  });
  $('sec-obs-save-vault').addEventListener('click', async () => {
    const path = ($('sec-obs-vault-path').value || '').trim();
    const status = $('sec-obs-vault-status');
    const hint = $('sec-obs-vault-hint');
    if (!path) { status.textContent = 'Enter a path or click Auto-detect.'; status.style.color = 'var(--error)'; return; }
    status.textContent = 'Saving…'; status.style.color = '';
    const r = await postJSON('/api/setup/obsidian-vault', { path });
    if (r.ok) {
      status.textContent = '✓ saved';
      status.style.color = 'var(--primary)';
      hint.textContent = `Vault set to ${path}. Voice notes will be written here.`;
      // Refresh status pill + detail
      await loadSecObsStatus();
    } else {
      const err = await r.json().catch(() => ({}));
      status.textContent = '✗ ' + (err.detail || 'invalid vault path');
      status.style.color = 'var(--error)';
      hint.textContent = err.detail || "That path isn't a vault. The folder must contain a .obsidian/ subfolder.";
    }
  });
  $('sec-obs-copy-token').addEventListener('click', () => {
    const t = $('sec-obs-token').textContent || '';
    if (t && t !== '—') navigator.clipboard.writeText(t).catch(() => {});
  });
  $('sec-obs-regen-token').addEventListener('click', async () => {
    if (!confirm('Regenerate the Obsidian auth token?\n\nThe plugin will disconnect within ~10 seconds. You\'ll need to paste the new token into the Fulloch plugin settings in Obsidian.')) return;
    $('sec-obs-regen-token').disabled = true;
    const r = await postJSON('/api/obsidian/regenerate-token', {});
    $('sec-obs-regen-token').disabled = false;
    const status = $('sec-obs-status');
    if (r.ok && r.body && r.body.token) {
      $('sec-obs-token').textContent = r.body.token;
      status.textContent = '✓ token rotated';
      status.style.color = 'var(--primary)';
    } else {
      status.textContent = '✗ regenerate failed';
      status.style.color = 'var(--error)';
    }
  });

  // Populate the Obsidian section's status pill, vault path field, and token
  // on initial render. Wrapped in a helper so the Save button can refresh.
  async function loadSecObsStatus() {
    const status = await getJSON('/api/obsidian/status').catch(() => null);
    const token = await postJSON('/api/obsidian/show-token', {}).then(r => r.json()).catch(() => null);
    const pill = $('sec-obs-pill');
    const detail = $('sec-obs-status-detail');
    const cstatus = $('sec-obs-cstatus');
    pill.className = 'obs-pill';
    if (token && token.token) {
      $('sec-obs-token').textContent = token.token;
    }
    if (status && status.last_error) {
      pill.textContent = 'Error';
      pill.classList.add('error');
      detail.textContent = status.vault_path || '';
      cstatus.textContent = 'error';
      cstatus.style.color = 'var(--error)';
    } else if (status && status.connected) {
      pill.textContent = 'Connected';
      pill.classList.add('connected');
      detail.textContent = status.vault_path || '';
      cstatus.textContent = 'connected';
      cstatus.style.color = 'var(--primary)';
    } else if (status && status.vault_path) {
      pill.textContent = 'Disconnected';
      pill.classList.add('disconnected');
      detail.textContent = status.vault_path;
      cstatus.textContent = 'vault set, plugin offline';
      cstatus.style.color = 'var(--text-muted)';
    } else {
      pill.textContent = 'Not configured';
      pill.classList.add('idle');
      detail.textContent = '';
      cstatus.textContent = 'not configured';
      cstatus.style.color = 'var(--text-muted)';
    }
    if (status && status.vault_path && !$('sec-obs-vault-path').value) {
      $('sec-obs-vault-path').value = status.vault_path;
    }
  }
  loadSecObsStatus();
  const certBtn = $('sec-cert-regen');
  if (certBtn) {
    const certField = SCHEMA.fields.find(f => f.section === 'general' && f.name === 'dashboard_ssl_certfile');
    const certEnabled = !!(certField && certField.value);
    const confirmMsg = certEnabled
      ? 'Regenerate the HTTPS certificate?\n\nThis overwrites the current certificate file. Any device that already trusted it (phones, other machines) will see the browser\'s "not private" warning again on its next visit. Fulloch must restart to apply.'
      : 'Enable HTTPS with a self-signed certificate?\n\nEvery browser will show a one-time "not private" warning on first visit — expected for a private LAN certificate. Fulloch must restart to apply.';
    certBtn.addEventListener('click', async () => {
      if (!confirm(confirmMsg)) return;
      const status = $('sec-cert-status');
      certBtn.disabled = true;
      const r = await postJSON('/setup/regen-cert');
      certBtn.disabled = false;
      if (r.ok) {
        status.innerHTML = (certEnabled ? '✓ regenerated' : '✓ enabled') + ' — <b>restart required</b> <button id="do-restart-cert" class="primary" style="margin-left:0.5rem;padding:0.2rem 0.6rem">Restart now</button>';
        status.style.color = 'var(--primary)';
        $('do-restart-cert').addEventListener('click', doRestartToHttps);
      } else {
        status.textContent = '✗ ' + (certEnabled ? 'regenerate' : 'enable') + ' failed'; status.style.color = 'var(--error)';
      }
    });
  }
}

// Quick integrations card shown at the top of settings
function integrationsCard() {
  const haUrl = String(((SCHEMA.fields.find(f => f.path === 'home_assistant.url') || {}).value) || '').replace(/"/g, '&quot;');
  const haTokenSet = SCHEMA.credentials && SCHEMA.credentials.ha_token;
  const haTokenPh = credPlaceholder(haTokenSet, 'create one in HA → Profile → Security');
  const notesPath = String(((SCHEMA.fields.find(f => f.path === 'notes.path') || {}).value) || '').replace(/"/g, '&quot;');
  const searchUrl = String(((SCHEMA.fields.find(f => f.path === 'search.searxng_url') || {}).value) || '').replace(/"/g, '&quot;');
  return el(`<div class="card"><h2>Integrations</h2>
    <p class="lead">Connect Fulloch to your other tools. Tokens take effect immediately; URL or path changes need a restart.</p>

    <details class="connect-section" id="ii-ha-section">
      <summary><span class="csname">🏠 Home Assistant</span><span class="cstatus" id="ii-ha-cstatus">${haUrl ? 'configured' : ''}</span></summary>
      <div class="cbody">
        <label>URL</label><input type="text" id="ii-ha-url" placeholder="http://homeassistant.local:8123" value="${haUrl}">
        <label>Long-lived access token</label><input type="text" id="ii-ha-token" placeholder="${haTokenPh}">
        <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.6rem">
          <button id="ii-ha-test">Test</button><span id="ii-ha-status" class="muted"></span>
        </div>
      </div>
    </details>

    <details class="connect-section" id="ii-obs-section">
      <summary><span class="csname">📓 Obsidian notes</span><span class="cstatus" id="ii-obs-cstatus">${notesPath ? 'configured' : ''}</span></summary>
      <div class="cbody">
        <label>Path to your vault</label><input type="text" id="ii-notes-path" placeholder="/home/you/Documents/MyVault" value="${notesPath}">
        <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.6rem">
          <button id="ii-obs-test">Test path</button><span id="ii-obs-status" class="muted"></span>
        </div>
        <p class="help" style="margin-top:0.4rem">For two-way navigation, install the <strong>Fulloch</strong> plugin from the Obsidian community plugin store.</p>
      </div>
    </details>

    <details class="connect-section" id="ii-search-section">
      <summary><span class="csname">🔍 Web search</span><span class="cstatus" id="ii-search-cstatus">${searchUrl ? 'configured' : ''}</span></summary>
      <div class="cbody">
        <label>SearXNG URL</label><input type="text" id="ii-search-url" placeholder="http://localhost:8080 (or blank for bundled container)" value="${searchUrl}">
      </div>
    </details>

    <div id="integrations-note"></div>
    <div class="actions"><span></span><button class="primary" id="save-integrations">Save integrations</button></div>
  </div>`);
}

function wireIntegrationsCard() {
  // Snapshot config values at card-open time — used to diff on save so we only
  // send fields that actually changed (and only show restart if they did).
  const _field = (path) => String(((SCHEMA.fields.find(f => f.path === path) || {}).value) || '');
  const savedHaUrl    = _field('home_assistant.url');
  const savedNotes    = _field('notes.path') || './data/notes';
  const savedSearchUrl = _field('search.searxng_url');

  $('ii-ha-test').addEventListener('click', async () => {
    const s = $('ii-ha-status'); s.textContent = 'Testing…'; s.style.color = '';
    try {
      const r = await postJSON('/setup/test-ha', {
        url: $('ii-ha-url').value.trim(), token: $('ii-ha-token').value.trim(),
      });
      const j = await r.json();
      s.textContent = j.ok ? '✓ connected' : ('✗ ' + (j.error || 'unreachable'));
      s.style.color = j.ok ? 'var(--primary)' : 'var(--error)';
      if (j.ok) { $('ii-ha-cstatus').textContent = 'connected'; $('ii-ha-cstatus').style.color = 'var(--primary)'; }
    } catch { s.textContent = '✗ unreachable'; s.style.color = 'var(--error)'; }
  });
  $('ii-obs-test').addEventListener('click', async () => {
    const s = $('ii-obs-status'); s.textContent = 'Checking…'; s.style.color = '';
    try {
      const r = await postJSON('/setup/test-path', { path: $('ii-notes-path').value.trim() });
      const j = await r.json();
      s.textContent = j.ok ? '✓ found' : '✗ path not found';
      s.style.color = j.ok ? 'var(--primary)' : 'var(--error)';
      if (j.ok) { $('ii-obs-cstatus').textContent = 'path ok'; $('ii-obs-cstatus').style.color = 'var(--primary)'; }
    } catch { s.textContent = '✗ error'; s.style.color = 'var(--error)'; }
  });
  $('save-integrations').addEventListener('click', async () => {
    const haUrl     = $('ii-ha-url').value.trim();
    const haToken   = $('ii-ha-token').value.trim();
    const notesPath = $('ii-notes-path').value.trim() || './data/notes';
    const searchUrl = $('ii-search-url').value.trim();
    const note = $('integrations-note');

    // HA token is live — save to credentials.json (no restart needed).
    if (haToken) {
      const rc = await postJSON('/setup/credential', { key: 'ha_token', value: haToken });
      if (!rc.ok) {
        const e = await rc.json().catch(() => ({}));
        note.innerHTML = `<div class="banner error">Could not save token: ${JSON.stringify(e.detail || rc.status)}</div>`;
        return;
      }
    }

    // Only send config fields that actually changed — restart only if they did.
    const updates = {};
    if (haUrl     !== savedHaUrl)     updates['home_assistant.url'] = haUrl;
    if (notesPath !== savedNotes)     updates['notes.path'] = notesPath;
    if (searchUrl !== savedSearchUrl) updates['search.searxng_url'] = searchUrl;

    if (Object.keys(updates).length === 0) {
      note.innerHTML = `<div class="banner" style="border:1px solid var(--primary);color:var(--primary)">Saved — changes took effect immediately.</div>`;
      return;
    }
    const r = await putJSON('/config', { updates });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      note.innerHTML = `<div class="banner error">Save failed: ${JSON.stringify(e.detail || r.status)}</div>`;
      return;
    }
    const { restart_required } = await r.json();
    if (restart_required) {
      note.innerHTML = `<div class="banner warn">Saved. Restart Fulloch for changes to take effect.
        <button id="do-restart-int" class="primary" style="margin-left:0.5rem;padding:0.3rem 0.8rem">Restart now</button></div>`;
      $('do-restart-int').addEventListener('click', doRestart);
    } else {
      note.innerHTML = `<div class="banner" style="border:1px solid var(--primary);color:var(--primary)">Saved — changes took effect immediately.</div>`;
    }
  });
}

async function resetSetup() {
  // Strong confirmation — list what gets wiped, what gets backed up, and
  // require the user to type a short phrase to proceed. Single `confirm()`
  // dialogs are too easy to click through for a destructive action.
  const ok = confirm(
    'Re-run the setup wizard?\n\n' +
    'A backup of your settings, credentials, Obsidian link, voice clones and\n' +
    'entity denylist will be created automatically. You can restore from the\n' +
    'backup list below after restarting.\n\n' +
    'Fulloch will restart into the wizard.'
  );
  if (!ok) return;
  const phrase = prompt('Type RE-RUN to confirm:');
  if ((phrase || '').trim().toUpperCase() !== 'RE-RUN') {
    const note = $('reset-note');
    if (note) note.innerHTML = `<div class="banner" style="border:1px solid var(--text-muted);color:var(--text-muted)">Cancelled — nothing was changed.</div>`;
    return;
  }
  const note = $('reset-note');
  const r = await postJSON('/setup/reset');
  if (!r.ok) {
    note.innerHTML = `<div class="banner error">Reset failed: ${r.status}</div>`;
    return;
  }
  const d = await r.json();
  note.innerHTML = `<div class="banner warn">Setup reset armed${d.backup ? ` — backup saved as <code>backups/${d.backup}</code>` : ''}. <b>Restart Fulloch</b> to run the wizard.
    <button id="do-restart-reset" class="primary" style="margin-left:0.5rem;padding:0.3rem 0.8rem">Restart now</button></div>`;
  $('do-restart-reset').addEventListener('click', doRestart);
  loadBackupList();
}

async function loadBackupList() {
  const wrap = $('backup-list-wrap');
  if (!wrap) return;
  let data;
  try { data = await getJSON('/setup/backups'); } catch { return; }
  const backups = (data && data.backups) || [];
  if (backups.length === 0) {
    wrap.innerHTML = `<p class="help" style="margin-top:1rem">No backups yet. Backups appear here after the first time you re-run setup.</p>`;
    return;
  }
  const fmtSize = (n) => {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  };
  wrap.innerHTML = `<div class="section-title" style="margin-top:1.25rem">Backups</div>
    <p class="help" style="margin:0 0 .5rem">Each backup is a timestamped snapshot of your settings. Restoring overwrites the current files — Fulloch will need a restart to pick up the change.</p>
    <div class="backup-list">${backups.map(b => `
      <div class="backup-row">
        <div class="backup-meta">
          <code>${b.name}</code>
          <span class="muted">${b.created_at || ''}${b.size_bytes ? ' · ' + fmtSize(b.size_bytes) : ''}</span>
          <span class="muted backup-files">${(b.files || []).join(', ')}</span>
        </div>
        <button class="obs-btn ghost" data-restore="${b.name}" type="button">Restore</button>
      </div>
    `).join('')}</div>`;
  wrap.querySelectorAll('[data-restore]').forEach(btn => {
    btn.addEventListener('click', () => restoreBackup(btn.getAttribute('data-restore')));
  });
}

async function restoreBackup(name) {
  if (!confirm(`Restore from backup "${name}"?\n\nThis will overwrite the current config, credentials, Obsidian link, voice clones and entity denylist with the backed-up versions. You'll need to restart Fulloch afterwards.`)) return;
  const r = await postJSON('/setup/backups/restore', { name });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    alert('Restore failed: ' + (err.detail || r.status));
    return;
  }
  const d = await r.json();
  alert(`Restored ${d.restored.length} entries from "${name}". Restart Fulloch to apply.`);
  loadBackupList();
}

async function populateVoiceField() {
  const sel = document.querySelector('#cfg-form select[data-voices="qwen"]');
  if (!sel) return;
  const cur = sel.value;
  try {
    const { voices } = await getJSON('/setup/voices');
    const list = (voices && voices.length) ? voices.slice() : [];
    if (cur && !list.includes(cur)) list.unshift(cur);
    if (list.length) {
      sel.innerHTML = list.map(v =>
        `<option value="${v}"${v === cur ? ' selected' : ''}>${v}</option>`).join('');
    }
  } catch (e) { /* keep the current value as the sole option */ }
}

function currentBackend(domain) {
  const m = SCHEMA.models;
  if (m && m[domain] && m[domain].backend) return m[domain].backend;
  const off = (SCHEMA.backends[domain] || []).filter(o => o.offerable);
  return off.length ? off[0].backend : '';
}

const DEFAULT_LLAMA_FILE = 'Qwen3.5-9B-UD-Q4_K_XL.gguf';
let CFG_INITIAL = {};

function modelsCard() {
  const b = SCHEMA.backends;
  const llm = (SCHEMA.models && SCHEMA.models.llm) || {};
  const llmMode = llm.backend === 'openai' || llm.backend === 'external' ? 'external' : 'local';
  const llamaCustom = (llm.local_model === 'custom' || llm.backend === 'llama') && llm.model &&
    !String(llm.model).endsWith(DEFAULT_LLAMA_FILE);
  const llmChoice = llamaCustom ? 'custom' : (llm.local_model || (llm.backend === 'gemma' ? 'gemma' : 'qwen'));
  const llamaPath = llamaCustom ? String(llm.model) : '';
  const llamaCtx = llmMode === 'local' && llm.n_context ? llm.n_context : '';
  const asrPath = String(((SCHEMA.models && SCHEMA.models.asr) || {}).model || '');
  const ttsPath = String(((SCHEMA.models && SCHEMA.models.tts) || {}).model || '');
  const fieldValue = (path, fallback = '') => {
    const field = SCHEMA.fields.find(f => f.path === path);
    return field && field.value != null ? field.value : fallback;
  };
  const higgsPersonality = String(fieldValue('general.higgs_personality', 'balanced'));
  const higgsCustom = String(fieldValue('general.higgs_personality_custom', ''));
  const optsFor = (domain) => {
    const cur = currentBackend(domain);
    return b[domain].filter(o => o.offerable).map(o =>
      `<option value="${o.backend}"${o.backend === cur ? ' selected' : ''}>${o.display_name}</option>`
    ).join('');
  };
  return el(`<div class="card"><h2>Models</h2>
    <p class="lead">Speech + language backends. <b>Model changes take effect after a restart.</b></p>
    <div class="grid2 model-speech-grid">
      <div><label>Speech-to-text</label><select id="sm-asr">${optsFor('asr')}</select></div>
      <div><label>Text-to-speech</label><select id="sm-tts">${optsFor('tts')}</select>
        <div id="sm-higgs-personality" style="display:none;margin-top:0.75rem">
          <label>Higgs personality</label>
          <select id="sm-higgs-personality-sel">
            <option value="balanced"${higgsPersonality === 'balanced' ? ' selected' : ''}>Balanced</option>
            <option value="playful"${higgsPersonality === 'playful' ? ' selected' : ''}>Playful</option>
            <option value="calm"${higgsPersonality === 'calm' ? ' selected' : ''}>Calm</option>
            <option value="wry"${higgsPersonality === 'wry' ? ' selected' : ''}>Wry</option>
            <option value="custom"${higgsPersonality === 'custom' ? ' selected' : ''}>Custom</option>
          </select>
          <div id="sm-higgs-custom" style="display:none;margin-top:0.5rem">
            <label>Custom delivery guidance</label>
            <input type="text" id="sm-higgs-custom-text" placeholder="e.g. Warm and reassuring; use pauses sparingly." value="${higgsCustom.replace(/"/g,'&quot;')}">
          </div>
        </div>
      </div>
    </div>
    <details class="advanced" id="sm-asr-custom"${asrPath ? ' open' : ''}>
      <summary>Speech-to-text: use a model I already have</summary>
      <label>Path to the ASR model folder</label>
      <input type="text" id="sm-asr-model" placeholder="/abs/path/asr-model-dir  (or ./data/models/x)" value="${asrPath.replace(/"/g,'&quot;')}">
      <div class="help">Load an ASR model already on disk from here instead of the default. Blank = default.</div>
    </details>
    <details class="advanced" id="sm-tts-custom"${ttsPath ? ' open' : ''}>
      <summary>Text-to-speech: use a model I already have</summary>
      <label>Path to the TTS model folder</label>
      <input type="text" id="sm-tts-model" placeholder="/abs/path/tts-model-dir  (or ./data/models/x)" value="${ttsPath.replace(/"/g,'&quot;')}">
      <div class="help">Load a TTS model already on disk from here instead of the default. Blank = default.</div>
    </details>
    <label>Language model</label>
    <select id="sm-llama-sel">
      <option value="local"${llmMode === 'local' ? ' selected' : ''}>Local</option>
      <option value="external"${llmMode === 'external' ? ' selected' : ''}>External</option>
    </select>
    <div id="sm-openai" style="display:none;margin-top:0.75rem">
      <label>Base URL</label><input type="text" id="sm-oai-url" placeholder="http://localhost:8888/v1" value="${(llmMode === 'external' ? llm.base_url || '' : '').replace(/"/g,'&quot;')}">
      <label>Model (optional)</label><input type="text" id="sm-oai-model" placeholder="blank for single-model servers; e.g. gpt-4o-mini for OpenAI" value="${(llmMode === 'external' ? llm.model || '' : '').replace(/"/g,'&quot;')}">
      <label>API key (optional)</label><input type="text" id="sm-oai-key" placeholder="${credPlaceholder(SCHEMA.credentials && SCHEMA.credentials.llm_api_key, '(blank for local servers; saved to credentials.json)')}">
      <div class="actions" style="justify-content:flex-start;gap:0.75rem">
        <button id="sm-oai-test">Test connection</button><span id="sm-oai-status" class="muted"></span></div>
      <div id="sm-oai-models" style="display:none;margin-top:0.5rem">
        <label>Available models</label>
        <div class="actions" style="justify-content:flex-start;gap:0.5rem">
          <select id="sm-oai-model-list" style="flex:1"></select>
          <button id="sm-oai-apply">Apply live</button><span id="sm-oai-apply-status" class="muted"></span>
        </div>
        <div class="help">Switches the running model instantly — no restart — and saves it. Picking one also fills the box above.</div>
      </div>
      <div class="help">An unreachable endpoint drops to limited regex-only commands at runtime.</div>
    </div>
    <div id="sm-llama" style="display:none;margin-top:0.75rem">
      <label>Local model</label>
      <select id="sm-local-model">
        <option value="qwen"${llmChoice === 'qwen' ? ' selected' : ''}>Qwen3.5 9B MTP (recommended)</option>
        <option value="gemma"${llmChoice === 'gemma' ? ' selected' : ''}>Gemma 4 12B QAT</option>
        <option value="custom"${llmChoice === 'custom' ? ' selected' : ''}>Custom GGUF file</option>
      </select>
      <div id="sm-llama-custom" style="display:none;margin-top:0.5rem">
        <label>Path to .gguf file</label>
        <input type="text" id="sm-llama-model" placeholder="/abs/path/model.gguf  (or ./data/models/x.gguf)" value="${llamaPath.replace(/"/g,'&quot;')}">
        <div class="help">Use a .gguf you already have to skip re-downloading. In Docker, a path outside ./data must be mounted into the container.</div>
      </div>
      <label>Context size (tokens)</label>
      <input type="number" id="sm-llama-ctx" placeholder="12288" value="${llamaCtx}">
      <div class="help">Larger context uses more VRAM — 16384 may OOM on a 16GB card.</div>
    </div>
    <div id="models-note"></div>
    <div class="actions"><span></span><button class="primary" id="save-models">Save models</button></div>
    </div>`);
}

function wireModelsCard() {
  const toggleHiggsPersonality = () => {
    const higgs = $('sm-tts').value === 'higgs-gguf';
    $('sm-higgs-personality').style.display = higgs ? '' : 'none';
    $('sm-higgs-custom').style.display = higgs && $('sm-higgs-personality-sel').value === 'custom' ? '' : 'none';
  };
  $('sm-tts').addEventListener('change', toggleHiggsPersonality);
  $('sm-higgs-personality-sel').addEventListener('change', toggleHiggsPersonality);
  toggleHiggsPersonality();
  const toggleLlmModel = () => {
    const mode = $('sm-llama-sel').value;
    const isOpenai = mode === 'external';
    const isLocal = mode === 'local';
    $('sm-openai').style.display = isOpenai ? '' : 'none';
    $('sm-llama').style.display = isLocal ? '' : 'none';
    $('sm-llama-custom').style.display = isLocal && $('sm-local-model').value === 'custom' ? '' : 'none';
    setBranding(isOpenai);
  };
  $('sm-llama-sel').addEventListener('change', toggleLlmModel);
  $('sm-local-model').addEventListener('change', toggleLlmModel);
  toggleLlmModel();
  $('sm-oai-url').addEventListener('blur', () => {
    const n = normalizeEndpointUrl($('sm-oai-url').value);
    if (n !== $('sm-oai-url').value) $('sm-oai-url').value = n;
  });
  $('sm-oai-test').addEventListener('click', async () => {
    const n = normalizeEndpointUrl($('sm-oai-url').value);
    if (n !== $('sm-oai-url').value) $('sm-oai-url').value = n;
    const s = $('sm-oai-status'); s.textContent = 'Testing…';
    const r = await postJSON('/setup/test-llm', {
      base_url: $('sm-oai-url').value.trim(), model: $('sm-oai-model').value.trim(),
      api_key: $('sm-oai-key').value.trim(),
    });
    const j = await r.json();
    s.textContent = j.ok ? '✓ reachable' : ('✗ ' + (j.error || 'failed'));
    s.style.color = j.ok ? 'var(--primary)' : 'var(--error)';
    if (j.ok) fetchOaiModels();
  });
  $('sm-oai-model-list').addEventListener('change', () => {
    $('sm-oai-model').value = $('sm-oai-model-list').value;
  });
  $('sm-oai-apply').addEventListener('click', async () => {
    const model = $('sm-oai-model-list').value;
    if (!model) return;
    const s = $('sm-oai-apply-status'); s.textContent = 'Switching…'; s.style.color = '';
    const r = await postJSON('/llm/model', { model });
    const j = await r.json().catch(() => ({}));
    if (j.ok) {
      $('sm-oai-model').value = model;
      s.textContent = '✓ now using ' + model + (j.persist_error ? ' (not saved)' : '');
      s.style.color = 'var(--primary)';
    } else {
      s.textContent = '✗ ' + (j.error || 'failed');
      s.style.color = 'var(--error)';
    }
  });
  if ($('sm-llama-sel').value === 'external' && $('sm-oai-url').value.trim()) fetchOaiModels();
  $('save-models').addEventListener('click', saveModels);
}

async function fetchOaiModels() {
  const url = $('sm-oai-url').value.trim();
  if (!url) return;
  const box = $('sm-oai-models');
  const r = await postJSON('/setup/list-llm-models', {
    base_url: url, api_key: $('sm-oai-key').value.trim(),
  });
  const j = await r.json().catch(() => ({}));
  const models = (j && j.models) || [];
  if (!j.ok || !models.length) { box.style.display = 'none'; return; }
  const cur = $('sm-oai-model').value.trim();
  $('sm-oai-model-list').innerHTML = models.map(m => {
    const e = m.replace(/"/g, '&quot;');
    return `<option value="${e}"${m === cur ? ' selected' : ''}>${e}</option>`;
  }).join('');
  box.style.display = '';
}

async function saveModels() {
  const llmMode = $('sm-llama-sel').value;
  const llm = { backend: llmMode };
  if (llmMode === 'external') {
    const model = $('sm-oai-model').value.trim();
    if (model) llm.model = model;
    llm.base_url = normalizeEndpointUrl($('sm-oai-url').value);
    if ($('sm-oai-url').value.trim() && llm.base_url !== $('sm-oai-url').value.trim()) $('sm-oai-url').value = llm.base_url;
    if (!llm.base_url) {
      $('models-note').innerHTML = `<div class="banner error">OpenAI needs a base URL.</div>`;
      return;
    }
  } else {
    const localModel = $('sm-local-model').value;
    llm.local_model = localModel;
    if (localModel === 'custom') {
      const m = $('sm-llama-model').value.trim();
      if (!m) {
        $('models-note').innerHTML = `<div class="banner error">Enter the path to your .gguf file, or pick Default.</div>`;
        return;
      }
      if (!m.toLowerCase().endsWith('.gguf')) {
        $('models-note').innerHTML = `<div class="banner error">That doesn't look like a .gguf model file.</div>`;
        return;
      }
      llm.model = m;
    }
    const ctx = $('sm-llama-ctx').value.trim();
    if (ctx) llm.n_context = parseInt(ctx, 10);
  }
  const models = { asr: { backend: $('sm-asr').value }, tts: { backend: $('sm-tts').value }, llm };
  const asrModel = $('sm-asr-model').value.trim();
  const ttsModel = $('sm-tts-model').value.trim();
  if (asrModel) models.asr.model = asrModel;
  if (ttsModel) models.tts.model = ttsModel;
  // API key goes to credentials.json, not config.yml.
  const llmKey = llmMode === 'external' ? $('sm-oai-key').value.trim() : '';
  if (llmKey) await postJSON('/setup/credential', { key: 'llm_api_key', value: llmKey });
  const r = await postJSON('/setup/models', { models });
  const note = $('models-note');
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    note.innerHTML = `<div class="banner error">Save failed: ${JSON.stringify(e.detail || r.status)}</div>`;
    return;
  }
  if (models.tts.backend === 'higgs-gguf') {
    const personality = $('sm-higgs-personality-sel').value;
    const custom = $('sm-higgs-custom-text').value.trim();
    if (personality === 'custom' && !custom) {
      note.innerHTML = `<div class="banner error">Enter custom Higgs delivery guidance, or choose a built-in personality.</div>`;
      return;
    }
    const config = await putJSON('/config', {
      updates: {
        'general.higgs_personality': personality,
        'general.higgs_personality_custom': personality === 'custom' ? custom : '',
      },
    });
    if (!config.ok) {
      const e = await config.json().catch(() => ({}));
      note.innerHTML = `<div class="banner error">Saved model choice, but Higgs settings failed: ${JSON.stringify(e.detail || config.status)}</div>`;
      return;
    }
  }
  SCHEMA.models = models;
  note.innerHTML = `<div class="banner warn">Saved to config.yml. <b>Restart Fulloch</b> for the model change to take effect. If you switched to a backend whose model isn't downloaded yet, the restart re-opens the setup wizard to fetch it.
    <button id="do-restart-models" class="primary" style="margin-left:0.5rem;padding:0.3rem 0.8rem">Restart now</button></div>`;
  $('do-restart-models').addEventListener('click', doRestart);
}

function autoWakePattern(wakeword) {
  const wk = String(wakeword || '').trim();
  if (!wk) return '';
  const preset = (SCHEMA.wakeword_presets || []).find(
    p => p.wakeword.toLowerCase() === wk.toLowerCase());
  if (preset) return preset.pattern;
  const tolerant = w => Array.from(w).map(
    c => (c === 's' || c === 'z') ? '[sz]' : c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('');
  const tokens = wk.split(/\s+/).filter(Boolean).map(tolerant);
  return '\\b' + tokens.join('\\W+') + '\\b';
}

function fieldRow(f) {
  if (f.path === 'general.higgs_personality' || f.path === 'general.higgs_personality_custom') return null;
  if (f.path === 'general.tts_speed' && currentBackend('tts') !== 'kokoro-onnx') return null;
  const id = 'cf-' + f.path.replace(/\W/g, '_');
  let needsRestart = f.apply === 'restart';
  if (f.path === 'general.voice_clone' && currentBackend('tts') === 'kokoro-onnx') needsRestart = false;
  const restart = needsRestart ? '<span class="restart">restart</span>' : '';
  const defRaw = (f.default === null || f.default === undefined) ? ''
    : (Array.isArray(f.default) ? f.default.join(', ') : String(f.default));
  const titleAttr = ((f.help || '') + (defRaw !== '' ? ` (default: ${defRaw})` : '')).replace(/"/g, '&quot;');
  const defPh = defRaw !== '' ? ` placeholder="${defRaw.replace(/"/g, '&quot;')}"` : '';
  let input;
  const val = f.value === null || f.value === undefined ? '' : f.value;
  if (f.path === 'general.wakeword_pattern') {
    const wkInput = document.getElementById('cf-general_wakeword');
    const wkF = (SCHEMA.fields || []).find(x => x.path === 'general.wakeword');
    const wkVal = wkInput ? wkInput.value : ((wkF && wkF.value) || '');
    const override = String(val);
    const node = el(`<div class="field"><div class="lbl"><label style="margin:0">${f.name}</label>
      <span class="info" title="${titleAttr}">i</span> ${restart}</div>
      <input type="text" id="${id}" data-path="${f.path}" data-type="str"
        value="${override.replace(/"/g,'&quot;')}" placeholder="${autoWakePattern(wkVal).replace(/"/g,'&quot;')}"></div>`);
    const inp = node.querySelector('input');
    inp.dataset.ghost = override === '' ? '1' : '0';
    inp.addEventListener('input', () => { inp.dataset.ghost = inp.value === '' ? '1' : '0'; });
    if (wkInput) wkInput.addEventListener('input', () => {
      inp.placeholder = autoWakePattern(wkInput.value);
    });
    return node;
  }
  if (f.path === 'general.voice_clone') {
    const cur = String(val);
    let opts;
    if (currentBackend('tts') === 'kokoro-onnx') {
      const list = KOKORO_VOICES.includes(cur) ? KOKORO_VOICES : [cur, ...KOKORO_VOICES].filter(Boolean);
      opts = list.map(v => kokoroOption(v, cur)).join('');
    } else {
      opts = `<option value="${cur.replace(/"/g,'&quot;')}" selected>${cur || '(default)'}</option>`;
    }
    const node = el(`<div class="field"><div class="lbl"><label style="margin:0">Voice model</label>
      <span class="info" title="${titleAttr}">i</span> ${restart}</div>
      <div class="voice-row"><select id="${id}" data-path="${f.path}" data-type="str"${currentBackend('tts') === 'kokoro-onnx' ? '' : ' data-voices="qwen"'}>${opts}</select></div></div>`);
    node.querySelector('.voice-row').appendChild(makeVoicePreview(node.querySelector('select')));
    return node;
  }
  if (f.type === 'bool') {
    input = `<select id="${id}" data-path="${f.path}" data-type="bool">
      <option value="true"${val === true ? ' selected' : ''}>true</option>
      <option value="false"${val === false ? ' selected' : ''}>false</option></select>`;
  } else if (f.type === 'enum') {
    input = `<select id="${id}" data-path="${f.path}" data-type="enum">${
      f.choices.map(ch => `<option value="${ch}"${ch === val ? ' selected' : ''}>${ch}</option>`).join('')}</select>`;
  } else if (f.type === 'list') {
    const txt = Array.isArray(val) ? val.join(', ') : val;
    const ph = defRaw !== '' ? defRaw.replace(/"/g, '&quot;') : 'comma-separated';
    input = `<input type="text" id="${id}" data-path="${f.path}" data-type="list" value="${String(txt).replace(/"/g,'&quot;')}" placeholder="${ph}">`;
  } else {
    const t = (f.type === 'int' || f.type === 'float') ? 'number' : 'text';
    input = `<input type="${t}" id="${id}" data-path="${f.path}" data-type="${f.type}" value="${String(val).replace(/"/g,'&quot;')}"${defPh}>`;
  }
  return el(`<div class="field"><div class="lbl"><label style="margin:0">${f.name}</label>
    <span class="info" title="${titleAttr}">i</span> ${restart}</div>${input}</div>`);
}

async function saveSettings() {
  const updates = {};
  document.querySelectorAll('#cfg-form [data-path]').forEach(node => {
    const v = node.dataset.ghost === '1' ? '' : node.value;
    if (!(node.dataset.path in CFG_INITIAL) || v !== CFG_INITIAL[node.dataset.path]) {
      updates[node.dataset.path] = v;
    }
  });
  const note = $('save-note');
  if (Object.keys(updates).length === 0) {
    note.innerHTML = `<div class="banner" style="border:1px solid var(--primary);color:var(--primary)">No changes to save.</div>`;
    return;
  }
  const r = await putJSON('/config', { updates });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    note.innerHTML = `<div class="banner error">Save failed: ${JSON.stringify(e.detail || r.status)}</div>`;
    return;
  }
  const { restart_required } = await r.json();
  Object.assign(CFG_INITIAL, updates);
  if (restart_required) {
    note.innerHTML = `<div class="banner warn">Saved. Some changes need a restart to take effect.
      <button id="do-restart" class="primary" style="margin-left:0.5rem;padding:0.3rem 0.8rem">Restart now</button></div>`;
    $('do-restart').addEventListener('click', doRestart);
  } else {
    note.innerHTML = `<div class="banner" style="border:1px solid var(--primary);color:var(--primary)">Saved and applied live — no restart needed.</div>`;
  }
}

async function doRestart(ev) {
  const btn = ev && ev.target;
  if (btn) { btn.disabled = true; btn.textContent = 'Restarting…'; }
  try { await postJSON('/restart'); } catch (e) { /* server may drop the connection */ }
  let tries = 0;
  const poll = () => setTimeout(async () => {
    try { await getJSON('/status'); location.reload(); }
    catch (e) {
      if (++tries < 90) poll();
      else if (btn) { btn.disabled = false; btn.textContent = 'Restart now'; }
    }
  }, 2000);
  setTimeout(poll, 3000);
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
  setTimeout(() => { location.href = httpsUrl; }, 4000);
}

boot();

import {
  postJSON, $, el, screen, normalizeEndpointUrl, credPlaceholder,
  wakewordModelDatalist, renderSteps, setBranding,
} from './setup-common.js';

export function createWizard({ state, voice, doInstall }) {
  const sel = state.sel;
  const { KOKORO_RECOMMENDED, KOKORO_VOICES, kokoroOption, stop: _voiceStop,
    makeVoicePreview, refreshVoiceList, showGenPanel } = voice;
// ---- step 1: brain --------------------------------------------------------
function fitFor(tierId) {
  if (!state.preflight) return null;
  return (state.preflight.tier_fit || []).find(t => t.id === tierId);
}

function recommendedTierId() {
  const fullFit = fitFor('full');
  const gpu = state.preflight && state.preflight.gpu && state.preflight.gpu.available;
  const fullOffered = (state.schema.tier_presets || []).some(t => t.id === 'full' && t.offerable !== false);
  if (gpu && fullOffered && fullFit && fullFit.badge === 'ok') return 'full';
  const staticRec = (state.schema.tier_presets || []).find(t => t.recommended);
  return staticRec ? staticRec.id : 'cpu_local';
}

// TLS banner: shown on the first wizard step when the dashboard is
// serving over HTTPS. Pre-empts the browser's self-signed-cert warning
// and gives the user a copyable URL (the user might be reaching the
// dashboard from a phone or another device on the LAN, where the URL
// bar isn't obvious). Dismissable — we remember the dismiss in
// localStorage so returning users don't see it again.
const TLS_BANNER_DISMISS_KEY = 'fulloch.tls_banner_dismissed_v1';
function tlsBanner() {
  if (!state.status || !state.status.dashboard_url) return null;
  if (localStorage.getItem(TLS_BANNER_DISMISS_KEY)) return null;
  const url = state.status.dashboard_url;
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
  'cpu_local':  { icon: '⚡', label: 'Simple commands',  blurb: 'Pattern-matching for smart home, timers, and quick questions. ASR and TTS run locally on the CPU; no language model or cloud service.' },
  'full':       { icon: '🧠', label: 'Full conversation', blurb: 'A local AI handles anything you ask. ASR, TTS, and the language model run on your GPU, fully private and offline.' },
  'cpu_server': { icon: '🌐', label: 'Remote AI',         blurb: 'Uses an AI server you already run (Ollama, LM Studio, OpenAI). ASR and TTS run locally on the CPU.' },
};

function stepBrain() {
  voice.stop();
  state.curStep = 0; renderSteps(0);
  const tiers = state.schema.tier_presets.filter(t => t.offerable !== false);
  const recId = recommendedTierId();
  if (!state.tierChosen) sel.tier = recId;
  const keyInCreds = state.schema && state.schema.credentials && state.schema.credentials.llm_api_key;
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
      sel.tier = t.id;
      // Keep the advanced controls aligned with the selected preset, rather
      // than leaving them on the previous custom/default backend values.
      sel.models = JSON.parse(JSON.stringify(t.models));
      state.tierChosen = true;
      document.querySelectorAll('#tier-list .opt').forEach(o => o.classList.toggle('sel', o.dataset.tier === t.id));
      renderBackendCfg();
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
  const b = state.schema.backends;
  const preset = (state.schema.tier_presets || []).find(t => t.id === sel.tier);
  const displayedModels = sel.models || (preset && preset.models);
  const curBackend = (domain) => (displayedModels && displayedModels[domain] && displayedModels[domain].backend) || '';
  const mk = (domain, label) => {
    const cur = curBackend(domain);
    if (domain === 'llm') {
      const mode = cur === 'none' ? 'none' : (cur === 'openai' || cur === 'external' ? 'external' : 'local');
      return `<div><label>${label}</label><select id="be-llm">
        <option value="local"${mode === 'local' ? ' selected' : ''}>Local</option>
        <option value="external"${mode === 'external' ? ' selected' : ''}>External</option>
        <option value="none"${mode === 'none' ? ' selected' : ''}>Regex-only commands</option>
      </select></div>`;
    }
    const opts = b[domain].filter(o => o.offerable).map(o => {
      const exp = o.experimental ? ' [experimental]' : '';
      return `<option value="${o.backend}"${o.backend === cur ? ' selected' : ''}>${o.display_name}${exp}</option>`;
    }).join('');
    return `<div><label>${label}</label><select id="be-${domain}">${opts}</select></div>`;
  };
  const currentLlm = (displayedModels && displayedModels.llm) || {};
  box.innerHTML = `<p class="help">Overrides the mode selected above.</p>
    <div class="grid2">${mk('asr','ASR')}${mk('tts','TTS')}${mk('llm','Language model')}</div>
    <div id="wizard-llm-accelerators" class="help" style="margin-top:0.7rem">
      <label><input type="checkbox" id="wizard-mtp"${currentLlm.mtp ? ' checked' : ''}> Enable experimental MTP speculative decoding</label>
      <label><input type="checkbox" id="wizard-flash-attn"${currentLlm.flash_attn ? ' checked' : ''}> Enable experimental llama.cpp Flash Attention</label>
      <div>Both are off by default. Enable only after stability-testing this GPU.</div>
    </div>`;
  const syncCustom = () => {
    sel.tier = 'custom';
    const llmMode = $('be-llm').value;
    sel.models = { asr: { backend: $('be-asr').value }, tts: { backend: $('be-tts').value },
                    llm: llmMode === 'local'
                      ? { backend: 'local', local_model: 'qwen' }
                      : llmMode === 'external' ? { backend: 'external' } : { backend: 'none' } };
    document.querySelectorAll('#tier-list .opt').forEach(o => o.classList.remove('sel'));
    syncOpenaiForm();
  };
  ['asr','tts','llm'].forEach(d => $(`be-${d}`).addEventListener('change', syncCustom));
  const syncAccelerators = () => {
    const local = $('be-llm').value === 'local';
    $('wizard-llm-accelerators').style.display = local ? '' : 'none';
  };
  $('be-llm').addEventListener('change', syncAccelerators);
  syncAccelerators();
}

function chosenModels() {
  let m;
  if (sel.tier === 'custom' && sel.models) m = JSON.parse(JSON.stringify(sel.models));
  else { const t = state.schema.tier_presets.find(x => x.id === sel.tier); m = t ? JSON.parse(JSON.stringify(t.models)) : null; }
  if (m && m.llm && (m.llm.backend === 'openai' || m.llm.backend === 'external')) {
    if (sel.openai.base_url) m.llm.base_url = normalizeEndpointUrl(sel.openai.base_url);
    if (sel.openai.model) m.llm.model = sel.openai.model;
    // api_key goes to credentials.json, not models config
  }
  if (m && m.llm && (m.llm.backend === 'local' || m.llm.backend === 'llama')) {
    m.llm.mtp = !!($('wizard-mtp') && $('wizard-mtp').checked);
    m.llm.flash_attn = !!($('wizard-flash-attn') && $('wizard-flash-attn').checked);
  }
  if (m) {
    if (sel.wakeword_model) {
      m.wakeword = { backend: 'openwakeword', model: sel.wakeword_model, threshold: 0.7, smoothing_frames: 1, cooldown_ms: 1500 };
    } else {
      delete m.wakeword;
    }
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

// ---- step 2: set up (wakeword + voice) -------------------------------------
async function stepSetup() {
  voice.stop();
  state.curStep = 1; renderSteps(1);
  const isKokoro = ttsBackend() === 'kokoro-onnx';
  const def = KOKORO_VOICES.includes(sel.voice_clone) ? sel.voice_clone : KOKORO_RECOMMENDED;
  const kokoroOpts = KOKORO_VOICES.map(v => kokoroOption(v, def)).join('');
  const presets = state.schema.wakeword_presets;
  const customWake = presets.some(p => p.wakeword === sel.wakeword) ? '' : sel.wakeword;

  const c = el(`<div class="card">
    <h2>Set up your assistant</h2>
    <p class="lead">Everything has a sensible default — pick a name and voice, then click <strong>Get started</strong>.</p>

    <div class="section-title">What should you call it?</div>
    <div id="wake-list"></div>
    <details class="advanced"${customWake ? ' open' : ''}><summary>Custom name</summary>
      <input type="text" id="wake-custom" placeholder="e.g. hey jarvis" value="${customWake.replace(/"/g, '&quot;')}" style="margin-top:0.5rem">
      <label style="margin-top:0.5rem">Wakeword model path <span class="muted" style="font-weight:400;font-size:0.8rem">(optional)</span></label>
       <input type="text" id="wake-model" list="wake-model-options" placeholder="/path/to/wakeword.onnx" value="${sel.wakeword_model.replace(/"/g, '&quot;')}">
       ${wakewordModelDatalist(state.schema, 'wake-model-options')}
       <div class="help">Choose a bundled Hey Atticus version or enter a compatible ONNX path. Leave blank to use ASR wakeword detection, which transcribes all voice activity.</div>
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
    const node = el(`<div class="opt ${sel.wakeword === p.wakeword ? 'sel' : ''}" data-wake="${p.wakeword}" data-pattern="${p.pattern.replace(/"/g, '&quot;')}" data-model="${p.model.replace(/"/g, '&quot;')}">
      <div class="row"><span class="name">${p.label} ${rec}</span></div></div>`);
    node.addEventListener('click', () => {
      sel.wakeword = p.wakeword; sel.wakeword_pattern = p.pattern; sel.wakeword_model = p.model;
      $('wake-custom').value = '';
      $('wake-model').value = p.model;
      document.querySelectorAll('#wake-list .opt').forEach(o => o.classList.toggle('sel', o.dataset.wake === p.wakeword));
    });
    wakeList.appendChild(node);
  });
  $('wake-custom').addEventListener('input', (e) => {
    const v = e.target.value.trim();
    if (v) { sel.wakeword = v; sel.wakeword_pattern = ''; sel.wakeword_model = ''; $('wake-model').value = ''; document.querySelectorAll('#wake-list .opt').forEach(o => o.classList.remove('sel')); }
  });
  $('wake-model').addEventListener('input', e => sel.wakeword_model = e.target.value.trim());

  // --- voice ---
  _voiceStop();
  if (isKokoro) {
    $('voice-sel').value = def; sel.voice_clone = def;
    $('voice-sel').addEventListener('change', e => sel.voice_clone = e.target.value);
    $('voice-sel').parentElement.appendChild(makeVoicePreview($('voice-sel')));
  } else {
    await refreshVoiceList();
    if (!c.isConnected) return;
    $('voice-sel').parentElement.appendChild(makeVoicePreview($('voice-sel')));
    $('gen-voice').addEventListener('click', showGenPanel);
  }

  $('back2').addEventListener('click', stepBrain);
  // Optional integrations are configured in a separate step.
  $('get-started').addEventListener('click', stepConnect);
}

// ---- step 2.4: connect HA + SearXNG (optional) ----------------------------
function stepConnect() {
  voice.stop();
  const haUrl = (sel.ha.url || '').replace(/"/g, '&quot;');
  const haToken = (sel.ha.token || '').replace(/"/g, '&quot;');
  const haTokenSet = state.schema.credentials && state.schema.credentials.ha_token;
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
        <input type="text" id="ha-url" placeholder="http://192.168.1.50:8123" value="${haUrl}">
        <p class="help">This runs inside Docker: use Home Assistant's LAN IP and port, not <code>localhost</code>. For example, <code>http://192.168.1.50:8123</code>.</p>
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

  if (state.schema.models) {
    const clear = el(`<label class="help" style="display:block;margin-top:0.75rem"><input type="checkbox" id="clear-integrations"> Remove existing Home Assistant and Search settings</label>`);
    c.querySelector('.actions').before(clear);
    $('clear-integrations').addEventListener('change', e => sel.clear_integrations = e.target.checked);
  }

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
  voice.stop();
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
      return false;
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
// ---- token step -----------------------------------------------------------
function stepFinish() {
  voice.stop();
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
      return false;
    }
    if (pw && pw.length < 8) {
      errEl.textContent = 'Password must be at least 8 characters.';
      errEl.style.display = '';
      return false;
    }

    $('finish-btn').disabled = true;
    const r = await postJSON('/setup/password', { name: name || null, password: pw || null });
    if (!r.ok) {
      errEl.textContent = 'Could not save — try again.';
      errEl.style.display = '';
      $('finish-btn').disabled = false;
      return false;
    }
    // A password was set → must log in; otherwise go straight to the dashboard.
    location.href = pw ? '/login' : '/';
  });
}

  return { stepBrain, stepFinish, chosenModels };
}

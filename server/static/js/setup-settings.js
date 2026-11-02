import { getJSON, postJSON, putJSON, $, el, screen, normalizeEndpointUrl, credPlaceholder,
  renderSteps, setBranding } from './setup-common.js';
import { clone, escapeHtml as esc, offeredTiers, matchingTier, selectedWakeword,
  applyWakewordPreset, wakewordOptions, wakewordWarning, info, gatedNotice } from './setup-models.js';
import { integrationCard, wireIntegrationCard } from './setup-ha-integration.js';

// Deliberately allowlisted UI. The complete config schema remains available to
// config.yml users; editing these controls must not reset their hidden options.
export function createSettings({ state, voice, doRestart, doRestartToHttps }) {
  let models, initialModels, initialFields, initialWakeword, customAiChanged = false;
  let restartPending = false;
  let voiceGeneration = 0;
  let selectedTier = null, presetBaseline = null;
  const field = (path, fallback = '') => {
    const f = state.schema.fields.find(f => f.path === path);
    return f?.value ?? f?.default ?? fallback;
  };
  const icons = { General: '⚙️', Models: '🤖', Dashboard: '🖥️', Wakeword: '👂', Voice: '🗣️', 'Home Assistant': '🏠', 'Web Search': '🔍' };
  const section = (name, help, body) => el(`<div class="settings-group">
    <details class="connect-section settings-card" data-section="${esc(name)}">
    <summary><span class="settings-icon" aria-hidden="true">${icons[name] || '•'}</span><span class="csname">${esc(name)}</span></summary><div class="cbody"><p class="settings-section-intro">${esc(help)}</p>${body}</div></details></div>`);
  const input = (path, label, help, type = 'text') => `<div class="field">
    <label for="cf-${path.replaceAll('.', '_')}">${esc(label)} ${info(help)}</label>
    <input id="cf-${path.replaceAll('.', '_')}" data-path="${path}" type="${type}" value="${esc(field(path))}"></div>`;
  const backend = domain => {
    const b = models[domain]?.backend || state.schema.default_backends?.[domain];
    if (domain !== 'llm') return b;
    const localModel = models.llm?.local_model || 'qwen';
    return b === 'local' ? ({ qwen: 'llama', custom: 'custom' }[localModel] || localModel)
      : b === 'external' ? 'openai' : b;
  };
  const optsFor = domain => (state.schema.backends[domain] || []).filter(b => b.offerable !== false
    && !(state.schema.variant === 'cpu' && b.gpu_only));
  const options = (items, current) => items.map(item =>
    `<option value="${esc(item.backend)}"${item.backend === current ? ' selected' : ''}>${esc(item.display_name)}</option>`).join('');
  const preservedOption = '<option value="__configured__">Configured in config.yml (unchanged)</option>';

  async function openSettings() {
    voice.stop(); renderSteps(0, 'settings'); $('subtitle').textContent = 'Settings';
    state.schema = await getJSON('/setup/schema');
    if (state.schema.models && !Object.keys(state.schema.models).length) state.schema.models = null;
    models = clone(state.schema.models || Object.fromEntries(Object.entries(state.schema.default_backends || {})
      .map(([domain, backend]) => [domain, { backend }])));
    initialModels = JSON.stringify(models);
    customAiChanged = false;
    restartPending = false;
    const wake = selectedWakeword(state.schema, models, field('general.wakeword'), field('general.wakeword_pattern'));
    initialWakeword = wake?.id || '';
    screen().innerHTML = '';
    screen().appendChild(el('<div class="topbar"><button class="back" id="settings-back">Go back</button><h2>Settings</h2></div>'));
    $('settings-back').addEventListener('click', () => location.href = '/');
    const config = el('<div id="settings-config"></div>'); screen().appendChild(config);
    const tiers = offeredTiers(state.schema), matched = matchingTier(state.schema, models);
    selectedTier = matched?.id || null;
    presetBaseline = null;
    config.appendChild(el(`<section class="preset-panel" aria-labelledby="preset-heading">
      <div class="preset-heading"><div><span class="preset-kicker">YOUR SETUP</span><h3 id="preset-heading">Choose a preset</h3></div>${info('Applying a preset replaces speech and AI models, while keeping your wakeword and other settings. A faded selection means you have customised settings since choosing it.')}</div>
      <p class="preset-intro">Pick the stack that suits your hardware. Fine-tune it below.</p>
      <div class="preset-pills" role="group" aria-label="Hardware presets">${tiers.map(t => `<button type="button" class="preset-pill" data-tier="${esc(t.id)}" aria-pressed="false"><span class="preset-pill-name">${esc(t.label.split(' — ')[0])}</span><span class="preset-pill-detail">${t.vram_class_gb ? `${t.vram_class_gb} GB` : 'No GPU'}</span></button>`).join('')}</div>
      <p class="preset-status" id="preset-description" aria-live="polite"></p><div id="preset-access"></div></section>`));
    config.appendChild(section('General', 'Notes storage and conversation history. Other tuning remains in config.yml.',
      input('notes.path', 'Notes directory', 'A directory on the Fulloch host/container for Markdown notes. It does not need to be an Obsidian vault.') + `
      <label>Delete conversation history ${info('Clears the dashboard and active conversation context, not saved notes or reports.')}</label>
      <button class="danger" id="delete-conversation">Delete conversation history</button>`));
    const asr = optsFor('asr'), tts = optsFor('tts');
    const llms = optsFor('llm').filter(b => b.backend !== 'openai');
    const aiCustom = ['custom', 'openai'].includes(backend('llm'));
    const aiValue = aiCustom ? 'custom' : backend('llm');
    config.appendChild(section('Models', 'Speech and AI models available in this image. Model changes require a restart; downloaded files are reused.', `
      <label for="sm-asr">Speech-to-text ${info('Transcribes your voice. Moonshine Base/Tiny run on CPU. The CPU image never offers GPU-only backends.')}</label>
      <select id="sm-asr">${!asr.some(b => b.backend === backend('asr')) ? preservedOption : ''}${options(asr, backend('asr'))}</select>
      <label for="sm-ai">AI ${info('Laya handles local semantic commands; local language models also generate answers. Custom accepts an OpenAI-compatible API address, or a GGUF path on the GPU image.')}</label>
      <select id="sm-ai">${!aiCustom && !llms.some(b => b.backend === aiValue) ? preservedOption : ''}${options(llms, aiValue)}<option value="custom"${aiCustom ? ' selected' : ''}>Custom</option></select>
      <div id="sm-ai-custom"${aiCustom ? '' : ' hidden'}>
        <label for="sm-ai-address">Model path or API address ${info('For a remote/local HTTP server enter its OpenAI-compatible URL (for example http://server:8888/v1). A custom local .gguf file requires the GPU image. Paths are on the server, not your browser device.')}</label>
        <input id="sm-ai-address" value="${esc(backend('llm') === 'openai' ? models.llm?.base_url : aiCustom ? models.llm?.model : '')}" placeholder="http://server:8888/v1 or /path/model.gguf">
        <div id="sm-api-fields">
          <label for="sm-api-model">API model (optional) ${info('Leave blank for a single-model server, or enter the model identifier required by your provider.')}</label>
          <input id="sm-api-model" value="${esc(backend('llm') === 'openai' ? models.llm?.model : '')}">
          <label for="sm-api-key">API key (optional) ${info('Stored separately in credentials.json. Leaving blank preserves the saved key.')}</label>
          <input id="sm-api-key" type="password" autocomplete="off" placeholder="${esc(credPlaceholder(state.schema.credentials?.llm_api_key, 'Optional'))}">
        </div>
      </div>
      <label for="sm-tts">Text-to-speech ${info('Kokoro has built-in voices. Qwen and Pocket support reference-audio voice cloning.')}</label>
      <select id="sm-tts">${!tts.some(b => b.backend === backend('tts')) ? preservedOption : ''}${options(tts, backend('tts'))}</select><div id="model-access"></div>`));
    config.appendChild(section('Dashboard', 'Network access, HTTPS, appearance and access protection.',
      input('general.dashboard_port', 'Port', 'Port used by the dashboard. Changing it requires reconnecting at the new address.', 'number') +
      input('general.dashboard_host', 'Host', '127.0.0.1 is local only; 0.0.0.0 listens on your network. Set a password before exposing it.') + `
      <label>HTTPS certificate ${info('Browsers require HTTPS for microphone access over the LAN. Generate a self-signed certificate, or load your own certificate and private key paths.')}</label>
      <button id="sec-cert-regen">Generate certificate</button>
      ${input('general.dashboard_ssl_certfile', 'Load certificate', 'Server path to a PEM certificate. Both certificate and private key are required.')}
      ${input('general.dashboard_ssl_keyfile', 'Private key', 'Server path to the matching PEM private key. Never share it.')}
      <label for="dashboard-theme">Colour scheme ${info('Auto follows your browser or operating system preference.')}</label>
      <select id="dashboard-theme" data-path="general.dashboard_theme">${['auto', 'light', 'dark'].map(t => `<option${field('general.dashboard_theme') === t ? ' selected' : ''}>${t}</option>`).join('')}</select>
      <label for="sec-pw">Dashboard password ${info('At least eight characters. Blank keeps the current password. Password changes are saved separately.')}</label>
      <input id="sec-pw" type="password" autocomplete="new-password" placeholder="New password">
      <label for="sec-pw2">Confirm password ${info('Repeat the new password exactly.')}</label>
      <input id="sec-pw2" type="password" autocomplete="new-password">
      <button id="sec-pw-save">Update password</button>
      <label>Update ${info('Container applications cannot replace their running image safely. Check releases, pull the matching CPU/GPU image and recreate the container with the same data volume.')}</label>
      <button id="dashboard-update">Update instructions</button><p id="update-instructions" class="help" hidden>See <a href="https://github.com/liampetti/local-assistant/releases" target="_blank" rel="noopener noreferrer">Fulloch releases</a>. Pull the new image and recreate your container with the same data volume and settings. Your data is kept.</p>
      <p id="dashboard-action-note" class="help" role="status"></p>`));
    config.appendChild(section('Wakeword', 'Choose a bundled acoustic detector. Custom names, paths, regex and tuning are config-file only.', `
      <label for="wakeword-preset">Wakeword ${info('Hey Atticus is the default. Alexa and Hey Jarvis are local openWakeWord models, not connections to commercial voice services.')}</label>
      <select id="wakeword-preset">${wakewordOptions(state.schema, initialWakeword, !initialWakeword)}</select>
      <p id="wakeword-warning" class="help warning"></p><p id="wakeword-license" class="help"></p>`));
    config.appendChild(section('Voice', 'Choose a voice and play its sample. Custom reference audio appears only for cloning-capable TTS.', `
      <label for="voice-sel">Voice ${info('Preview plays the reference sample. Synthesized output depends on the selected TTS model.')}</label>
      <div class="voice-row"><select id="voice-sel"></select></div>
      <div id="custom-voice" hidden><label for="voice-reference">Custom WAV path ${info('Use a WAV file on the server with a matching .txt transcript alongside it. Only use voices you have permission to clone. For bundled samples choose the dropdown instead.')}</label>
        <input id="voice-reference" placeholder="/path/reference.wav"></div>`));
    config.appendChild(section('Home Assistant', 'Optional household device, music and calendar integration.',
      input('home_assistant.url', 'URL', 'Home Assistant LAN address. Inside Docker, localhost points to the container.') + `
      <label for="ha-access-token">Home Assistant access token ${info('A Home Assistant long-lived access token lets Fulloch control Home Assistant. It is different from the Fulloch integration token below. Blank keeps the existing secret.')}</label>
      <input id="ha-access-token" type="password" autocomplete="off" placeholder="${esc(credPlaceholder(state.schema.credentials?.ha_token, 'Optional'))}">` +
      input('home_assistant.spotify_entity', 'Music entity', 'The Home Assistant media_player entity used for music/Spotify.') +
      input('home_assistant.calendar', 'Calendar entity', 'The Home Assistant calendar name or entity used for reminders.') + integrationCard()));
    config.appendChild(section('Web Search', 'Optional current information through SearXNG.',
      input('search.searxng_url', 'URL', 'A SearXNG-compatible JSON search endpoint. A general search-provider homepage is not compatible.')));
    screen().appendChild(el('<div class="card settings-save"><p id="save-note" role="status"></p><div class="actions"><button id="cfg-back" class="back">Go back</button><button id="save-cfg" class="primary">Save changes</button></div></div>'));
    $('cfg-back').addEventListener('click', () => location.href = '/');
    wireIntegrationCard();
    initialFields = {};
    document.querySelectorAll('[data-path]').forEach(n => initialFields[n.dataset.path] = n.value);
    document.querySelectorAll('.preset-pill').forEach(button => button.addEventListener('click', async () => {
      const tier = tiers.find(t => t.id === button.dataset.tier);
      selectedTier = tier.id;
      for (const domain of ['asr', 'tts', 'llm']) models[domain] = clone(tier.models[domain]);
      $('sm-asr').value = backend('asr'); $('sm-tts').value = backend('tts'); $('sm-ai').value = backend('llm');
      customAiChanged = false; syncAi(); refreshWakeword(); refreshAccess();
      await refreshVoices(true);
      presetBaseline = settingsSnapshot(); renderPreset();
    }));
    ['asr', 'tts'].forEach(domain => $(`sm-${domain}`).addEventListener('change', e => {
      if (e.target.value === '__configured__') return;
      models[domain] = { backend: e.target.value };
      refreshWakeword(); refreshAccess();
      if (domain === 'tts') refreshVoices(true);
    }));
    $('sm-ai').addEventListener('change', () => {
      const selected = $('sm-ai').value;
      if (selected === 'custom') customAiChanged = true;
      else if (selected !== '__configured__') {
        models.llm = ['llama', 'gemma', 'ornith', 'qwen27b'].includes(selected)
          ? { backend: 'local', local_model: selected === 'llama' ? 'qwen' : selected }
          : { backend: selected };
        customAiChanged = false;
      }
      syncAi(); refreshAccess();
    });
    ['sm-ai-address', 'sm-api-model', 'sm-api-key'].forEach(id => $(id).addEventListener('input', () => { customAiChanged = true; syncAi(); }));
    $('wakeword-preset').addEventListener('change', refreshWakeword);
    $('voice-sel').addEventListener('change', () => {
      $('custom-voice').hidden = $('voice-sel').value !== '__custom__';
      document.querySelector('.voice-play').disabled = $('voice-sel').value === '__custom__';
    });
    $('voice-sel').parentElement.appendChild(voice.makeVoicePreview($('voice-sel')));
    $('save-cfg').addEventListener('click', saveAllSettings);
    $('delete-conversation').addEventListener('click', async () => {
      if (!confirm('Delete conversation history? Saved notes and reports will be kept.')) return;
      await action('/reset', {}, 'Conversation history deleted.');
    });
    $('sec-pw-save').addEventListener('click', async () => {
      const password = $('sec-pw').value;
      if (password.length < 8 || password !== $('sec-pw2').value) {
        $('dashboard-action-note').textContent = 'Passwords must match and contain at least eight characters.'; return;
      }
      if (await action('/setup/password', { password }, 'Password updated.')) { $('sec-pw').value = ''; $('sec-pw2').value = ''; }
    });
    $('sec-cert-regen').addEventListener('click', async () => {
      if (!confirm('Generate a new self-signed certificate? Existing certificate trust may need updating.')) return;
      if (await action('/setup/regen-cert', {}, 'Certificate generated. Restart required.')) {
        const cfg = await getJSON('/setup/schema');
        for (const path of ['general.dashboard_ssl_certfile', 'general.dashboard_ssl_keyfile']) {
          const value = cfg.fields.find(f => f.path === path)?.value || '';
          document.querySelector(`[data-path="${path}"]`).value = value; initialFields[path] = value;
        }
        const button = el('<button>Restart with HTTPS</button>');
        button.addEventListener('click', doRestartToHttps); $('dashboard-action-note').appendChild(button);
      }
    });
    $('dashboard-update').addEventListener('click', () => $('update-instructions').hidden = !$('update-instructions').hidden);
    $('dashboard-theme').addEventListener('change', applyTheme);
    config.addEventListener('input', renderPreset);
    config.addEventListener('change', renderPreset);
    syncAi(); refreshWakeword(); refreshAccess(); await refreshVoices(false);
    presetBaseline = settingsSnapshot(); renderPreset();
    if (new URLSearchParams(location.search).get('section') === 'notes') {
      document.querySelector('[data-section="General"]').open = true;
    }
  }

  function settingsSnapshot() {
    return JSON.stringify({ models, values: Array.from(document.querySelectorAll('#settings-config input, #settings-config select'))
      .map(node => [node.id || node.dataset.path, node.value]) });
  }

  function renderPreset() {
    const tier = offeredTiers(state.schema).find(t => t.id === selectedTier);
    const custom = tier && presetBaseline !== null && settingsSnapshot() !== presetBaseline;
    document.querySelectorAll('.preset-pill').forEach(button => {
      const selected = button.dataset.tier === selectedTier;
      button.classList.toggle('selected', selected);
      button.classList.toggle('customised', selected && !!custom);
      button.setAttribute('aria-pressed', String(selected));
    });
    $('preset-description').textContent = tier
      ? `${custom ? 'Customised from ' : 'Selected: '}${tier.label}. ${tier.blurb}`
      : 'Custom configuration — choose a preset to start from a ready-made stack.';
  }

  function refreshAccess() {
    $('model-access').innerHTML = gatedNotice(state.schema, models);
    const tier = offeredTiers(state.schema).find(t => t.id === selectedTier);
    $('preset-access').innerHTML = tier ? gatedNotice(state.schema, tier.models) : '';
  }

  function syncAi() {
    const custom = $('sm-ai').value === 'custom';
    $('sm-ai-custom').hidden = !custom;
    $('sm-api-fields').hidden = /\.gguf$/i.test($('sm-ai-address').value.trim());
    setBranding(custom && !$('sm-api-fields').hidden);
  }

  function refreshWakeword() {
    const p = state.schema.wakeword_presets.find(p => p.id === $('wakeword-preset').value);
    const preview = applyWakewordPreset(clone(models), p);
    $('wakeword-warning').textContent = wakewordWarning(state.schema, preview);
    $('wakeword-license').textContent = p?.license ? `Model license: ${p.license}.` : '';
  }

  async function refreshVoices(changedTts) {
    voice.stop();
    const generation = ++voiceGeneration;
    $('save-cfg').disabled = true;
    const kokoro = backend('tts') === 'kokoro-onnx';
    const spec = optsFor('tts').find(b => b.backend === backend('tts'));
    const cloning = spec?.voice_cloning === true;
    $('custom-voice').hidden = true;
    const current = changedTts ? (kokoro ? voice.KOKORO_RECOMMENDED : 'atticus') : field('general.voice_clone', 'atticus');
    let voices = kokoro ? voice.KOKORO_VOICES : [];
    if (!kokoro) {
      try { voices = (await getJSON('/setup/voices')).voices || []; } catch { /* keep current */ }
    }
    if (generation !== voiceGeneration || !$('voice-sel')) return;
    const custom = cloning && /\.wav$/i.test(current);
    if (current && !custom && !voices.includes(current)) voices = [current, ...voices];
    $('voice-sel').innerHTML = voices.map(v => `<option value="${esc(v)}"${v === current ? ' selected' : ''}>${esc(v)}</option>`).join('')
      + (cloning ? `<option value="__custom__"${custom ? ' selected' : ''}>Custom reference</option>` : '');
    $('voice-reference').value = custom ? current : '';
    $('custom-voice').hidden = !custom;
    document.querySelector('.voice-play').disabled = custom;
    $('save-cfg').disabled = false;
  }

  function applyTheme() {
    const theme = $('dashboard-theme').value;
    document.documentElement.classList.toggle('dark', theme === 'dark'
      || (theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches));
  }

  async function action(url, body, message) {
    try {
      const r = await postJSON(url, body);
      if (!r.ok) throw new Error((await r.json()).detail || 'Request failed');
      $('dashboard-action-note').textContent = message;
      return true;
    } catch (e) { $('dashboard-action-note').textContent = String(e.message); return false; }
  }

  async function saveAllSettings() {
    const note = $('save-note'), button = $('save-cfg');
    button.disabled = true;
    try {
      const next = clone(models);
      if ($('sm-ai').value === 'custom' && customAiChanged) {
        const address = $('sm-ai-address').value.trim();
        if (/\.gguf$/i.test(address)) {
          if (state.schema.variant === 'cpu') throw new Error('Custom local GGUF models require the GPU image. Enter an API address instead.');
          const previous = next.llm?.backend === 'local' && next.llm.local_model === 'custom' ? next.llm : {};
          next.llm = { ...previous, backend: 'local', local_model: 'custom', model: address };
        } else {
          if (!address) throw new Error('Enter a GGUF path or an OpenAI-compatible API address.');
          if (address.startsWith('/')) throw new Error('A local model path must end in .gguf. Otherwise enter an HTTP(S) API address.');
          const baseUrl = normalizeEndpointUrl(address);
          const parsedUrl = new URL(baseUrl);
          if (!['http:', 'https:'].includes(parsedUrl.protocol)) throw new Error('Enter an HTTP(S) API address.');
          const previous = ['external', 'openai'].includes(next.llm?.backend) ? next.llm : {};
          next.llm = { ...previous, backend: previous.backend || 'external', base_url: baseUrl };
          if ($('sm-api-model').value.trim()) next.llm.model = $('sm-api-model').value.trim();
          else delete next.llm.model;
        }
      }
      const wake = state.schema.wakeword_presets.find(p => p.id === $('wakeword-preset').value);
      applyWakewordPreset(next, wake);
      const updates = {};
      document.querySelectorAll('[data-path]').forEach(n => {
        if (n.value !== initialFields[n.dataset.path]) updates[n.dataset.path] = n.value;
      });
      if (wake && wake.id !== initialWakeword) {
        updates['general.wakeword'] = wake.wakeword;
        updates['general.wakeword_pattern'] = wake.pattern;
      }
      const selectedVoice = $('voice-sel').value;
      const selectedReference = $('voice-reference').value.trim();
      if (selectedVoice === '__custom__' && !/\.wav$/i.test(selectedReference)) throw new Error('Enter the custom WAV reference path; a matching .txt transcript must exist alongside it.');
      const voiceValue = selectedVoice === '__custom__' ? selectedReference : selectedVoice;
      if (voiceValue && voiceValue !== field('general.voice_clone')) updates['general.voice_clone'] = voiceValue;
      let restart = JSON.stringify(next) !== initialModels;
      if (restart) {
        const r = await postJSON('/setup/models', { models: next });
        if (!r.ok) throw new Error(JSON.stringify((await r.json()).detail || 'Could not save models'));
        models = next; initialModels = JSON.stringify(next);
        restartPending = true;
      }
      if (Object.keys(updates).length) {
        const r = await putJSON('/config', { updates });
        if (!r.ok) throw new Error(JSON.stringify((await r.json()).detail || 'Could not save settings'));
        restart = (await r.json()).restart_required || restart;
        restartPending ||= restart;
        Object.assign(initialFields, updates);
        for (const f of state.schema.fields) if (f.path in updates) f.value = updates[f.path];
      }
      for (const [key, value] of [['ha_token', $('ha-access-token').value.trim()], ['llm_api_key', $('sm-api-key').value.trim()]]) {
        if (!value) continue;
        const r = await postJSON('/setup/credential', { key, value });
        if (!r.ok) throw new Error('Could not save credential: ' + key);
      }
      $('ha-access-token').value = ''; $('sm-api-key').value = '';
      initialWakeword = $('wakeword-preset').value; customAiChanged = false;
      note.textContent = restartPending ? 'Saved. Restart Fulloch to apply these changes.' : 'Saved.';
      if (restartPending) {
        const restartButton = el('<button class="primary">Restart now</button>');
        restartButton.addEventListener('click', doRestart); note.appendChild(restartButton);
      }
    } catch (e) { note.textContent = 'Save failed: ' + e.message; }
    finally { button.disabled = false; }
  }

  return { openSettings };
}

import {
  getJSON, postJSON, putJSON, $, el, screen, normalizeEndpointUrl, credPlaceholder,
  wakewordModelDatalist, renderSteps, cpuVariantNotice, setBranding,
} from './setup-common.js';

export function createSettings({ state, voice, doRestart, doRestartToHttps }) {
  const { KOKORO_VOICES, kokoroOption,
    makeVoicePreview, populateVoiceField } = voice;
// ---- settings console (post-setup) ----------------------------------------
async function openSettings() {
  voice.stop();
  renderSteps(0, 'settings');
  $('subtitle').textContent = 'Settings';
  state.schema = await getJSON('/setup/schema');
  screen().innerHTML = '';

  const bar = el(`<div class="topbar">
    <button class="back" id="settings-back">Back to dashboard</button>
    <span class="spacer"></span><h2>Settings</h2></div>`);
  screen().appendChild(bar);
  $('settings-back').addEventListener('click', () => location.href = '/');

  // CPU-image banner
  {
    const vn = cpuVariantNotice(state.schema);
    if (vn) screen().appendChild(vn);
  }

  // Every setting belongs to one user-facing domain. This intentionally differs
  // from the storage schema, whose groups are optimized for config.yml.
  const config = el(`<div id="settings-config"><p class="lead settings-lead">Expand a section to configure it. Every change is saved to <code>config.yml</code>.</p></div>`);
  const categories = [
    { name: 'Dashboard', icon: '▣', fields: f => f.group === 'Dashboard' },
    { name: 'Voice', icon: '◉', fields: f => f.group === 'Voice' || f.group === 'Endpointing' || ['general.wakeword', 'general.wakeword_pattern', 'general.asr_language', 'general.asr_context_hint', 'general.asr_context_terms'].includes(f.path) },
    { name: 'Notes', icon: '✎', fields: f => f.group === 'Notes' || f.group === 'Obsidian' },
    { name: 'Home Assistant', icon: '⌂', fields: f => f.group === 'Home Assistant' },
    { name: 'Search', icon: '⌕', fields: f => f.group === 'Search' },
    { name: 'Advanced', icon: '⚙', fields: f => !['Dashboard', 'Voice', 'Endpointing', 'Notes', 'Obsidian', 'Home Assistant', 'Search'].includes(f.group) && !['general.wakeword', 'general.wakeword_pattern', 'general.asr_language', 'general.asr_context_hint', 'general.asr_context_terms'].includes(f.path) },
  ];
  const categoryCards = {};
  categories.forEach(category => {
    const sourceFields = state.schema.fields.filter(category.fields);
    const fields = sourceFields.map(fieldRow).filter(Boolean);
    if (!fields.length) return;
    const configured = sourceFields.some(f => f.set && String(f.value) !== String(f.default));
    const status = configured ? 'customised' : 'defaults';
    const card = el(`<details class="connect-section settings-card"><summary><span class="csname"><span class="settings-icon">${category.icon}</span>${category.name}</span><span class="cstatus">${status}</span></summary><div class="cbody"><div class="settings-common"></div></div></details>`);
    fields.forEach(node => card.querySelector('.settings-common').appendChild(node));
    config.appendChild(card);
    categoryCards[category.name] = card;
  });
  screen().appendChild(config);
  const saveCard = el(`<div class="card settings-save">
    <div id="save-note"></div>
    <div class="actions"><button class="back" id="cfg-back">Back to dashboard</button><button class="primary" id="save-cfg">Save changes</button></div>
    </div>`);
  screen().appendChild(saveCard);
  $('cfg-back').addEventListener('click', () => location.href = '/');
  const wakewordPreset = $('wakeword-preset');
  if (wakewordPreset) wakewordPreset.addEventListener('change', () => {
    const customFields = $('wakeword-custom-fields');
    const custom = wakewordPreset.value === 'custom';
    customFields.hidden = !custom;
    if (!custom) {
      const preset = (state.schema.wakeword_presets || []).find(p => p.wakeword === wakewordPreset.value);
      if (preset) {
        $('cf-general_wakeword').value = preset.wakeword;
        $('cf-general_wakeword_pattern').value = preset.pattern;
        $('sm-wakeword-model').value = preset.model || '';
      }
    }
  });
  $('save-cfg').addEventListener('click', saveAllSettings);
  CFG_INITIAL = {};
  document.querySelectorAll('#settings-config [data-path]').forEach(node => {
    CFG_INITIAL[node.dataset.path] = node.dataset.ghost === '1' ? '' : node.value;
  });
  populateVoiceField();

  // Move action-oriented controls into their matching domain rather than
  // presenting duplicate top-level cards.
  const appendCardBody = (card, target) => {
    const body = card.querySelector('.cbody');
    while (body.firstChild) target.appendChild(body.firstChild);
  };
  appendCardBody(dashboardPreferencesCard(), categoryCards.Dashboard.querySelector('.cbody'));
  appendCardBody(homeAssistantAccessCard(), categoryCards['Home Assistant'].querySelector('.cbody'));
  const models = modelsCard();
  appendCardBody(models, categoryCards.Voice.querySelector('.cbody'));
  const agentCard = el(`<details class="connect-section settings-card"><summary><span class="csname"><span class="settings-icon">◫</span>AI Agent</span><span class="cstatus">configured</span></summary><div class="cbody"><p class="help">Personality and the language model that plans and answers requests.</p></div></details>`);
  const agentBody = agentCard.querySelector('.cbody');
  const speechGrid = categoryCards.Voice.querySelector('.model-speech-grid');
  agentBody.appendChild(speechGrid.children[2]);
  const llmSelect = categoryCards.Voice.querySelector('#sm-llama-sel');
  agentBody.appendChild(llmSelect.previousElementSibling);
  agentBody.appendChild(llmSelect);
  agentBody.appendChild(categoryCards.Voice.querySelector('#sm-openai'));
  agentBody.appendChild(categoryCards.Voice.querySelector('#sm-llama'));
  agentBody.appendChild(categoryCards.Voice.querySelector('#models-note'));
  agentBody.appendChild(categoryCards.Voice.querySelector('#save-models').parentElement);
  categoryCards.Voice.after(agentCard);
  const security = securityCard();
  categoryCards.Dashboard.querySelector('.cbody').appendChild(security.querySelector('#sec-pw-section'));
  categoryCards.Dashboard.querySelector('.cbody').appendChild(security.querySelector('#sec-cert-section'));
  categoryCards.Notes.querySelector('.cbody').appendChild(security.querySelector('#sec-obs-section'));
  wireDashboardPreferencesCard();
  wireHomeAssistantAccessCard();
  wireModelsCard();
  MODEL_INITIAL = modelSettingsSignature();
  wireSecurityCard();

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

  if (new URLSearchParams(location.search).get('section') === 'notes') {
    categoryCards.Notes.open = true;
    requestAnimationFrame(() => {
      $('sec-obs-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
}

function homeAssistantAccessCard() {
  const tokenSet = state.schema.credentials && state.schema.credentials.ha_token;
  return el(`<details class="connect-section settings-card"><summary><span class="csname"><span class="settings-icon">⌂</span>Home Assistant access</span><span class="cstatus">${tokenSet ? 'configured' : 'token needed'}</span></summary><div class="cbody">
    <p class="help">Set the long-lived access token used with the Home Assistant URL above.</p>
    <label for="ha-access-token">Long-lived access token</label>
    <input type="text" id="ha-access-token" placeholder="${credPlaceholder(tokenSet, 'create one in HA → Profile → Security')}">
    <div id="ha-access-note"></div>
    <div class="actions"><span></span><button class="primary" id="save-ha-access">Save token</button></div>
  </div></details>`);
}

function wireHomeAssistantAccessCard() {
  $('save-ha-access').addEventListener('click', async () => {
    const token = $('ha-access-token').value.trim();
    const note = $('ha-access-note');
    if (!token) { note.innerHTML = '<div class="banner error">Enter a token to save it.</div>'; return; }
    const r = await postJSON('/setup/credential', { key: 'ha_token', value: token });
    if (r.ok) {
      $('ha-access-token').value = '';
      note.innerHTML = '<div class="banner saved">Token saved.</div>';
    } else {
      note.innerHTML = '<div class="banner error">Could not save the token.</div>';
    }
  });
}

function dashboardPreferencesCard() {
  const field = path => (state.schema.fields || []).find(f => f.path === path) || {};
  const theme = field('general.dashboard_theme').value || 'auto';
  const showDetails = !!field('general.dashboard_show_turn_details').value;
  const status = theme === 'auto' && !showDetails ? 'defaults' : 'customised';
  return el(`<details class="connect-section settings-card dashboard-preferences"><summary><span class="csname"><span class="settings-icon">▣</span>Dashboard</span><span class="cstatus">${status}</span></summary><div class="cbody">
    <p class="help">Choose how this browser displays the dashboard.</p>
    <fieldset>
      <legend>Colour scheme</legend>
      <label><input type="radio" name="dashboard-theme" value="auto"${theme === 'auto' ? ' checked' : ''}> Auto</label>
      <label><input type="radio" name="dashboard-theme" value="dark"${theme === 'dark' ? ' checked' : ''}> Dark</label>
      <label><input type="radio" name="dashboard-theme" value="light"${theme === 'light' ? ' checked' : ''}> Light</label>
    </fieldset>
    <label class="checkbox-row"><input id="dashboard-show-details" type="checkbox"${showDetails ? ' checked' : ''}> Always show turn details</label>
    <p class="help">Turn details include the agent loop and inference statistics.</p>
    <div id="dashboard-pref-note"></div>
    <div class="actions"><span></span><button class="primary" id="save-dashboard-preferences">Save dashboard preferences</button></div>
    <div class="dashboard-actions">
      <button class="danger" id="delete-conversation" type="button">Delete conversation</button>
      <button class="danger" id="dashboard-logout" type="button" hidden>Log out</button>
    </div>
  </div></details>`);
}

function wireDashboardPreferencesCard() {
  const applyTheme = theme => {
    const dark = theme === 'dark' || (theme === 'auto' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', dark);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = getComputedStyle(document.documentElement).backgroundColor;
  };
  $('save-dashboard-preferences').addEventListener('click', async () => {
    const theme = document.querySelector('input[name="dashboard-theme"]:checked').value;
    const showDetails = $('dashboard-show-details').checked;
    const note = $('dashboard-pref-note');
    const r = await putJSON('/config', { updates: {
      'general.dashboard_theme': theme,
      'general.dashboard_show_turn_details': showDetails,
    } });
    if (!r.ok) {
      note.innerHTML = '<div class="banner error">Could not save dashboard preferences.</div>';
      return;
    }
    applyTheme(theme);
    note.innerHTML = '<div class="banner saved">Saved. The dashboard will use these preferences on your next visit.</div>';
  });
  document.querySelectorAll('input[name="dashboard-theme"]').forEach(input => {
    input.addEventListener('change', () => applyTheme(input.value));
  });
  $('delete-conversation').addEventListener('click', async () => {
    if (!confirm('Delete this conversation? This cannot be undone.')) return;
    const r = await fetch('/reset', { method: 'POST' });
    if (r.ok) $('dashboard-pref-note').innerHTML = '<div class="banner saved">Conversation deleted.</div>';
  });
  getJSON('/status').then(status => {
    $('dashboard-logout').hidden = !status.auth_enabled;
  }).catch(() => {});
  $('dashboard-logout').addEventListener('click', async () => {
    await fetch('/auth/logout', { method: 'POST' });
    location.href = '/login';
  });
}

// Security card — password change + Obsidian linkage + HTTPS cert
function securityCard() {
  const certField = state.schema.fields.find(f => f.section === 'general' && f.name === 'dashboard_ssl_certfile');
  const certEnabled = !!(certField && certField.value);
  return el(`<div class="card"><h2>Security &amp; Access</h2>
    <p class="lead">Manage the dashboard password, Obsidian plugin token, and HTTPS certificate.</p>

    <section class="connect-section exposed-section" id="sec-pw-section">
      <div class="connect-heading"><span class="csname">🔑 Dashboard password</span></div>
      <div class="cbody">
        <label>New password</label><input type="password" id="sec-pw" placeholder="min 8 characters" autocomplete="new-password">
        <label style="margin-top:.4rem">Confirm password</label><input type="password" id="sec-pw2" placeholder="repeat password" autocomplete="new-password" style="margin-top:.4rem">
        <div id="sec-pw-err" class="banner error" style="display:none;margin-top:.5rem"></div>
        <button id="sec-pw-save" style="margin-top:.75rem">Update password</button>
        <span id="sec-pw-status" class="muted" style="margin-left:.75rem"></span>
      </div>
    </section>

    <section class="connect-section exposed-section" id="sec-obs-section" style="margin-top:.75rem">
      <div class="connect-heading"><span class="csname">📓 Obsidian</span><span class="cstatus" id="sec-obs-cstatus">loading…</span></div>
      <div class="cbody">
        <p class="help" style="margin:0 0 .65rem">Configure the Obsidian plugin and choose a vault as Fulloch's notes location. Fulloch keeps writing Markdown notes and <code>fulloch_facts.md</code> there even when Obsidian is closed.</p>

        <div class="obs-status-row" style="margin-bottom: .75rem">
          <span class="obs-pill" id="sec-obs-pill">—</span>
          <span class="obs-status-detail" id="sec-obs-status-detail"></span>
        </div>

        <div class="group-title" style="margin-top: 0">Vault</div>
        <label for="sec-obs-vault-path">Use this Obsidian vault for notes</label>
        <div class="obs-switch" style="margin-top:.35rem">
          <input type="text" id="sec-obs-vault-path" placeholder="/home/you/Documents/MyVault">
        </div>
        <div style="display:flex;align-items:center;gap:.5rem;margin-top:.5rem;flex-wrap:wrap">
          <button id="sec-obs-detect" type="button">Auto-detect</button>
          <button class="primary" id="sec-obs-save-vault" type="button">Save</button>
          <span id="sec-obs-vault-status" class="muted"></span>
        </div>
        <p class="help" id="sec-obs-vault-hint" style="margin-top:.4rem">This saves <code>notes.path</code> in config.yml. Path must contain a <code>.obsidian/</code> subfolder. Auto-detect scans <code>~/Documents</code>, <code>~/Obsidian</code>, and <code>~/.config/obsidian</code>.</p>

         <div class="group-title">Plugin</div>
         <p class="help" style="margin:0 0 .5rem">Download the plugin archive and extract it into <code>&lt;vault&gt;/.obsidian/plugins/fulloch/</code>, then enable it in <strong>Settings → Community plugins</strong>.</p>
         <a class="obs-btn" href="/api/obsidian/plugin.zip" download>Download plugin.zip</a>
         <p class="help" style="margin:.7rem 0 0"><strong>HTTPS trust for Obsidian desktop:</strong> the plugin connects over secure WebSockets and cannot bypass a self-signed certificate warning. Create a private CA on the Fulloch host with <code>python scripts/create_local_ca.py --force --ip &lt;Fulloch-LAN-IP&gt;</code>, then on Linux install only <code>data/certs/fulloch-home-ca.crt</code> with <code>sudo cp data/certs/fulloch-home-ca.crt /usr/local/share/ca-certificates/ &amp;&amp; sudo update-ca-certificates</code>. Restart Fulloch and Obsidian afterwards. Use the HTTPS host/IP in plugin settings; never install or share <code>fulloch-home-ca.key</code>.</p>

         <div class="group-title">Auth token</div>
        <div class="obs-token" style="margin-top:.35rem">
          <code id="sec-obs-token">—</code>
          <button class="obs-btn ghost" id="sec-obs-copy-token" type="button">Copy</button>
          <button class="obs-btn danger" id="sec-obs-regen-token" type="button">Regenerate</button>
        </div>
        <p class="help" style="margin-top:.4rem">Paste this into the Fulloch plugin settings. Rotating the token drops the plugin connection within 10 seconds.</p>
        <span id="sec-obs-status" class="muted" style="display:block;margin-top:.35rem"></span>
      </div>
    </section>

    <section class="connect-section exposed-section" id="sec-cert-section" style="margin-top:.75rem">
      <div class="connect-heading"><span class="csname">🔒 HTTPS certificate</span><span class="cstatus" id="sec-cert-cstatus">${certEnabled ? 'enabled' : ''}</span></div>
      <div class="cbody">
        <p style="font-size:.8rem;color:var(--text-muted);margin:0 0 .6rem">${certEnabled
          ? 'Self-signed certificate used for LAN HTTPS (needed for mic access from phones and other devices). Obsidian desktop also needs the certificate trusted by its operating system; use the private-CA instructions in the Obsidian section above. Regenerate it if your LAN IP changed and the old certificate no longer covers it. Every device that trusted the old one will see the browser warning again. Takes effect after a restart.'
          : 'HTTPS isn\'t enabled for this install. Browsers refuse microphone access on plain HTTP for anything but localhost, so phones and other LAN devices can\'t use the mic without it. Generating a self-signed certificate fixes that — every browser shows a one-time "not private" warning on first visit, which is expected. Takes effect after a restart.'}</p>
        <button id="sec-cert-regen">${certEnabled ? 'Regenerate certificate…' : 'Enable HTTPS…'}</button>
        <span id="sec-cert-status" class="muted" style="margin-left:.75rem"></span>
      </div>
    </section>
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
    const certField = state.schema.fields.find(f => f.section === 'general' && f.name === 'dashboard_ssl_certfile');
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
  const haUrl = String(((state.schema.fields.find(f => f.path === 'home_assistant.url') || {}).value) || '').replace(/"/g, '&quot;');
  const haTokenSet = state.schema.credentials && state.schema.credentials.ha_token;
  const haTokenPh = credPlaceholder(haTokenSet, 'create one in HA → Profile → Security');
  const notesPath = String(((state.schema.fields.find(f => f.path === 'notes.path') || {}).value) || '').replace(/"/g, '&quot;');
  const searchUrl = String(((state.schema.fields.find(f => f.path === 'search.searxng_url') || {}).value) || '').replace(/"/g, '&quot;');
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
  const _field = (path) => String(((state.schema.fields.find(f => f.path === path) || {}).value) || '');
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
        return false;
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
    return false;
  }
  const note = $('reset-note');
  const r = await postJSON('/setup/reset');
  if (!r.ok) {
    note.innerHTML = `<div class="banner error">Reset failed: ${r.status}</div>`;
    return false;
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
    return false;
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

function currentBackend(domain) {
  const m = state.schema.models;
  if (m && m[domain] && m[domain].backend) return m[domain].backend;
  const off = (state.schema.backends[domain] || []).filter(o => o.offerable);
  return off.length ? off[0].backend : '';
}

const DEFAULT_LLAMA_FILE = 'Qwen3.5-9B-UD-Q4_K_XL.gguf';
let CFG_INITIAL = {};
let MODEL_INITIAL = '';

function modelSettingsSignature() {
  return [
    'sm-asr', 'sm-asr-model', 'sm-tts', 'sm-tts-model', 'sm-wakeword-model', 'sm-wakeword-threshold',
    'sm-wakeword-smoothing-frames', 'sm-wakeword-cooldown-ms', 'sm-personality-sel',
    'sm-personality-custom-text', 'sm-llama-sel', 'sm-local-model', 'sm-llama-model',
    'sm-llama-ctx', 'sm-mtp', 'sm-flash-attn', 'sm-oai-url', 'sm-oai-model', 'sm-oai-key',
  ].map(id => {
    const node = $(id) || {};
    return node.type === 'checkbox' ? String(node.checked) : (node.value || '');
  }).join('\u0000');
}

function modelsCard() {
  const b = state.schema.backends;
  const llm = (state.schema.models && state.schema.models.llm) || {};
  const llmMode = llm.backend === 'none' ? 'none' : (llm.backend === 'openai' || llm.backend === 'external' ? 'external' : 'local');
  const llamaCustom = (llm.local_model === 'custom' || llm.backend === 'llama') && llm.model &&
    !String(llm.model).endsWith(DEFAULT_LLAMA_FILE);
  const llmChoice = llamaCustom ? 'custom' : (llm.local_model || (llm.backend === 'gemma' ? 'gemma' : (llm.backend === 'ornith' ? 'ornith' : 'qwen')));
  const llamaPath = llamaCustom ? String(llm.model) : '';
  const llamaCtx = llmMode === 'local' && llm.n_context ? llm.n_context : '';
  const asrPath = String(((state.schema.models && state.schema.models.asr) || {}).model || '');
  const ttsPath = String(((state.schema.models && state.schema.models.tts) || {}).model || '');
  const fieldValue = (path, fallback = '') => {
    const field = state.schema.fields.find(f => f.path === path);
    return field && field.value != null ? field.value : fallback;
  };
  const personality = String(fieldValue('general.personality', 'balanced'));
  const personalityCustom = String(fieldValue('general.personality_custom', ''));
  const optsFor = (domain, customPath) => {
    const cur = currentBackend(domain);
    return b[domain].filter(o => o.offerable).map(o =>
      `<option value="${o.backend}"${!customPath && o.backend === cur ? ' selected' : ''}>${o.display_name}</option>`
    ).join('') + `<option value="custom"${customPath ? ' selected' : ''}>Custom model path</option>`;
  };
  const modelStatus = (state.schema.models && Object.keys(state.schema.models).length) ? 'configured' : 'defaults';
  return el(`<details class="connect-section settings-card"><summary><span class="csname"><span class="settings-icon">◫</span>Models</span><span class="cstatus">${modelStatus}</span></summary><div class="cbody">
    <p class="help">ASR, TTS, and language-model backends. Model changes take effect after a restart.</p>
    <div class="grid2 model-speech-grid">
      <div><label>ASR</label><select id="sm-asr">${optsFor('asr', asrPath)}</select>
        <div id="sm-asr-custom" style="display:${asrPath ? '' : 'none'};margin-top:0.5rem"><label>Path to the ASR model</label><input type="text" id="sm-asr-model" placeholder="/abs/path/asr-model-dir  (or ./data/models/x)" value="${asrPath.replace(/"/g,'&quot;')}"><div class="help">Use an ASR model already on disk.</div></div></div>
      <div><label>TTS</label><select id="sm-tts">${optsFor('tts', ttsPath)}</select>
        <div id="sm-tts-custom" style="display:${ttsPath ? '' : 'none'};margin-top:0.5rem"><label>Path to the TTS model</label><input type="text" id="sm-tts-model" placeholder="/abs/path/tts-model-dir  (or ./data/models/x)" value="${ttsPath.replace(/"/g,'&quot;')}"><div class="help">Use a TTS model already on disk.</div></div></div>
      <div><label>Personality</label>
        <select id="sm-personality-sel">
          <option value="balanced"${personality === 'balanced' ? ' selected' : ''}>Balanced</option>
          <option value="playful"${personality === 'playful' ? ' selected' : ''}>Playful</option>
          <option value="calm"${personality === 'calm' ? ' selected' : ''}>Calm</option>
          <option value="wry"${personality === 'wry' ? ' selected' : ''}>Wry</option>
          <option value="custom"${personality === 'custom' ? ' selected' : ''}>Custom</option>
        </select>
        <div id="sm-personality-custom" style="display:none;margin-top:0.5rem">
          <label>Custom personality</label>
          <input type="text" id="sm-personality-custom-text" placeholder="e.g. Warm and reassuring; use pauses sparingly." value="${personalityCustom.replace(/"/g,'&quot;')}">
        </div>
      </div>
    </div>
    <label>Language model</label>
    <select id="sm-llama-sel">
      <option value="local"${llmMode === 'local' ? ' selected' : ''}>Local</option>
      <option value="external"${llmMode === 'external' ? ' selected' : ''}>External</option>
      <option value="none"${llmMode === 'none' ? ' selected' : ''}>Regex-only commands</option>
    </select>
    <div id="sm-openai" style="display:none;margin-top:0.75rem">
      <label>Base URL</label><input type="text" id="sm-oai-url" placeholder="http://localhost:8888/v1" value="${(llmMode === 'external' ? llm.base_url || '' : '').replace(/"/g,'&quot;')}">
      <label>Model (optional)</label><input type="text" id="sm-oai-model" placeholder="blank for single-model servers; e.g. gpt-4o-mini for OpenAI" value="${(llmMode === 'external' ? llm.model || '' : '').replace(/"/g,'&quot;')}">
      <label>API key (optional)</label><input type="text" id="sm-oai-key" placeholder="${credPlaceholder(state.schema.credentials && state.schema.credentials.llm_api_key, '(blank for local servers; saved to credentials.json)')}">
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
        <option value="ornith"${llmChoice === 'ornith' ? ' selected' : ''}>Ornith 1.5 9B Q4</option>
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
      <div class="help" style="margin-top:0.7rem">
        <label><input type="checkbox" id="sm-mtp"${llm.mtp ? ' checked' : ''}> Enable experimental MTP speculative decoding</label>
        <label><input type="checkbox" id="sm-flash-attn"${llm.flash_attn ? ' checked' : ''}> Enable experimental llama.cpp Flash Attention</label>
        <div>Both are off by default. Enable only after stability-testing this GPU.</div>
      </div>
    </div>
    <div id="models-note"></div>
    <div class="actions"><span></span><button class="primary" id="save-models">Save models</button></div>
    </div></details>`);
}

function wireModelsCard() {
  const togglePersonalityCustom = () => {
    $('sm-personality-custom').style.display = $('sm-personality-sel').value === 'custom' ? '' : 'none';
  };
  $('sm-personality-sel').addEventListener('change', togglePersonalityCustom);
  togglePersonalityCustom();
  const toggleCustomModel = (domain) => {
    $(`sm-${domain}-custom`).style.display = $(`sm-${domain}`).value === 'custom' ? '' : 'none';
  };
  ['asr', 'tts'].forEach(domain => {
    $(`sm-${domain}`).addEventListener('change', () => {
      if ($(`sm-${domain}`).value !== 'custom') $(`sm-${domain}-model`).value = '';
      toggleCustomModel(domain);
    });
    toggleCustomModel(domain);
  });
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
      return false;
    }
  } else if (llmMode === 'local') {
    const localModel = $('sm-local-model').value;
    llm.local_model = localModel;
    if (localModel === 'custom') {
      const m = $('sm-llama-model').value.trim();
      if (!m) {
        $('models-note').innerHTML = `<div class="banner error">Enter the path to your .gguf file, or pick Default.</div>`;
        return false;
      }
      if (!m.toLowerCase().endsWith('.gguf')) {
        $('models-note').innerHTML = `<div class="banner error">That doesn't look like a .gguf model file.</div>`;
        return false;
      }
      llm.model = m;
    }
    const ctx = $('sm-llama-ctx').value.trim();
    if (ctx) llm.n_context = parseInt(ctx, 10);
    llm.mtp = $('sm-mtp').checked;
    llm.flash_attn = $('sm-flash-attn').checked;
  }
  const models = {
    asr: { backend: $('sm-asr').value === 'custom' ? currentBackend('asr') : $('sm-asr').value },
    tts: { backend: $('sm-tts').value === 'custom' ? currentBackend('tts') : $('sm-tts').value },
    llm,
  };
  const asrModel = $('sm-asr').value === 'custom' ? $('sm-asr-model').value.trim() : '';
  const ttsModel = $('sm-tts').value === 'custom' ? $('sm-tts-model').value.trim() : '';
  if (asrModel) models.asr.model = asrModel;
  if (ttsModel) models.tts.model = ttsModel;
  const wakewordModel = $('sm-wakeword-model').value.trim();
  if (wakewordModel) {
    models.wakeword = {
      backend: 'openwakeword',
      model: wakewordModel,
      threshold: Number($('sm-wakeword-threshold').value),
      smoothing_frames: Number($('sm-wakeword-smoothing-frames').value),
      cooldown_ms: Number($('sm-wakeword-cooldown-ms').value),
    };
  }
  // API key goes to credentials.json, not config.yml.
  const llmKey = llmMode === 'external' ? $('sm-oai-key').value.trim() : '';
  if (llmKey) await postJSON('/setup/credential', { key: 'llm_api_key', value: llmKey });
  const r = await postJSON('/setup/models', { models });
  const note = $('models-note');
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    note.innerHTML = `<div class="banner error">Save failed: ${JSON.stringify(e.detail || r.status)}</div>`;
    return false;
  }
  const personality = $('sm-personality-sel').value;
  const custom = $('sm-personality-custom-text').value.trim();
  if (personality === 'custom' && !custom) {
    note.innerHTML = `<div class="banner error">Enter a custom personality, or choose a built-in personality.</div>`;
    return false;
  }
  const config = await putJSON('/config', {
    updates: {
      'general.personality': personality,
      'general.personality_custom': personality === 'custom' ? custom : '',
    },
  });
  if (!config.ok) {
    const e = await config.json().catch(() => ({}));
    note.innerHTML = `<div class="banner error">Saved model choice, but personality settings failed: ${JSON.stringify(e.detail || config.status)}</div>`;
    return false;
  }
  state.schema.models = models;
  note.innerHTML = `<div class="banner warn">Saved to config.yml. <b>Restart Fulloch</b> for the model change to take effect. If you switched to a backend whose model isn't downloaded yet, the restart re-opens the setup wizard to fetch it.
    <button id="do-restart-models" class="primary" style="margin-left:0.5rem;padding:0.3rem 0.8rem">Restart now</button></div>`;
  $('do-restart-models').addEventListener('click', doRestart);
  return true;
}

function autoWakePattern(wakeword) {
  const wk = String(wakeword || '').trim();
  if (!wk) return '';
  const preset = (state.schema.wakeword_presets || []).find(
    p => p.wakeword.toLowerCase() === wk.toLowerCase());
  if (preset) return preset.pattern;
  const tolerant = w => Array.from(w).map(
    c => (c === 's' || c === 'z') ? '[sz]' : c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('');
  const tokens = wk.split(/\s+/).filter(Boolean).map(tolerant);
  return '\\b' + tokens.join('\\W+') + '\\b';
}

function fieldRow(f) {
  if (['general.personality', 'general.personality_custom', 'general.dashboard_theme',
    'general.dashboard_show_turn_details', 'general.wakeword_pattern', 'notes.path',
    'general.dashboard_ssl_certfile', 'general.dashboard_ssl_keyfile'].includes(f.path)) return null;
  if (f.path === 'general.tts_speed' && currentBackend('tts') !== 'kokoro-onnx') return null;
  const id = 'cf-' + f.path.replace(/\W/g, '_');
  let needsRestart = f.apply === 'restart';
  if (f.path === 'general.voice_clone' && ['kokoro-onnx', 'pocket-tts-onnx', 'pocket-tts-gguf', 'pocket-tts-pytorch', 'audio8'].includes(currentBackend('tts'))) needsRestart = false;
  const restart = needsRestart ? '<span class="restart">restart</span>' : '';
  const defRaw = (f.default === null || f.default === undefined) ? ''
    : (Array.isArray(f.default) ? f.default.join(', ') : String(f.default));
  const titleAttr = ((f.help || '') + (defRaw !== '' ? ` (default: ${defRaw})` : '')).replace(/"/g, '&quot;');
  const defPh = defRaw !== '' ? ` placeholder="${defRaw.replace(/"/g, '&quot;')}"` : '';
  let input;
  const val = f.value === null || f.value === undefined ? '' : f.value;
  if (f.path === 'general.wakeword') {
    const patternField = (state.schema.fields || []).find(x => x.path === 'general.wakeword_pattern') || {};
    const pattern = String(patternField.value || '');
    const presets = state.schema.wakeword_presets || [];
    const preset = presets.find(p => p.wakeword === val && p.pattern === pattern);
    const options = presets.map(p => `<option value="${p.wakeword.replace(/"/g, '&quot;')}"${preset === p ? ' selected' : ''}>${p.label}</option>`).join('');
    const custom = !preset;
    const wakeword = (state.schema.models && state.schema.models.wakeword) || {};
    const wakewordModel = String(wakeword.model || '');
    const wakewordThreshold = wakeword.threshold ?? 0.7;
    const wakewordSmoothingFrames = wakeword.smoothing_frames ?? 1;
    const wakewordCooldownMs = wakeword.cooldown_ms ?? 1500;
    return el(`<div class="field"><div class="lbl"><label style="margin:0">Wakeword</label>
      <span class="info" title="Choose Hey Atticus or enter a custom wakeword. An optional ONNX model prevents ASR from transcribing every voice activity segment.">i</span> ${restart}</div>
      <select id="wakeword-preset"><option value="custom"${custom ? ' selected' : ''}>Custom</option>${options}</select>
      <div id="wakeword-custom-fields"${custom ? '' : ' hidden'}>
        <label for="cf-general_wakeword">Custom wakeword</label>
        <input type="text" id="cf-general_wakeword" data-path="general.wakeword" data-type="str" value="${String(val).replace(/"/g, '&quot;')}">
        <label for="cf-general_wakeword_pattern">Wakeword regex (optional)</label>
        <input type="text" id="cf-general_wakeword_pattern" data-path="general.wakeword_pattern" data-type="str" value="${pattern.replace(/"/g, '&quot;')}" placeholder="${autoWakePattern(val).replace(/"/g, '&quot;')}">
      </div>
       <label for="sm-wakeword-model">Wakeword model path (optional)</label>
       <input type="text" id="sm-wakeword-model" list="sm-wakeword-model-options" placeholder="/path/to/wakeword.onnx" value="${wakewordModel.replace(/"/g, '&quot;')}">
       ${wakewordModelDatalist(state.schema, 'sm-wakeword-model-options')}
       <div class="help">Choose a bundled Hey Atticus version or enter a compatible ONNX path. Leave blank for ASR-only wakeword detection, which transcribes all voice activity.</div>
       <div class="grid2" style="margin-top:0.5rem">
         <div><label for="sm-wakeword-threshold">Detection threshold</label>
           <input type="number" id="sm-wakeword-threshold" min="0" max="1" step="0.01" value="${wakewordThreshold}">
           <div class="help">0 to 1. Raise it to reduce false activations.</div></div>
         <div><label for="sm-wakeword-smoothing-frames">Smoothing frames</label>
           <input type="number" id="sm-wakeword-smoothing-frames" min="1" max="100" step="1" value="${wakewordSmoothingFrames}">
           <div class="help">Consecutive detections required.</div></div>
       </div>
       <label for="sm-wakeword-cooldown-ms">Cooldown (ms)</label>
       <input type="number" id="sm-wakeword-cooldown-ms" min="0" max="60000" step="100" value="${wakewordCooldownMs}">
       <div class="help">Wait time after activation before detecting another wakeword.</div>
       </div>`);
  }
  if (f.path === 'general.wakeword_pattern') {
    const wkInput = document.getElementById('cf-general_wakeword');
    const wkF = (state.schema.fields || []).find(x => x.path === 'general.wakeword');
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
  } else if (f.type === 'dict') {
    const txt = JSON.stringify(val || {}, null, 2).replace(/</g, '&lt;');
    input = `<textarea id="${id}" data-path="${f.path}" data-type="dict" placeholder='{"/host/path": "/container/path"}'>${txt}</textarea>`;
  } else {
    const t = (f.type === 'int' || f.type === 'float') ? 'number' : 'text';
    input = `<input type="${t}" id="${id}" data-path="${f.path}" data-type="${f.type}" value="${String(val).replace(/"/g,'&quot;')}"${defPh}>`;
  }
  return el(`<div class="field"><div class="lbl"><label style="margin:0">${f.name}</label>
    <span class="info" title="${titleAttr}">i</span> ${restart}</div>${input}</div>`);
}

async function saveSettings(silentIfEmpty = false) {
  const updates = {};
  document.querySelectorAll('#settings-config [data-path]').forEach(node => {
    const v = node.dataset.ghost === '1' ? '' : node.value;
    if (!(node.dataset.path in CFG_INITIAL) || v !== CFG_INITIAL[node.dataset.path]) {
      updates[node.dataset.path] = v;
    }
  });
  const note = $('save-note');
  if (Object.keys(updates).length === 0) {
    if (!silentIfEmpty) note.innerHTML = `<div class="banner" style="border:1px solid var(--primary);color:var(--primary)">No changes to save.</div>`;
    return false;
  }
  const r = await putJSON('/config', { updates });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    note.innerHTML = `<div class="banner error">Save failed: ${JSON.stringify(e.detail || r.status)}</div>`;
    return false;
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
  return true;
}

async function saveAllSettings() {
  const modelsChanged = modelSettingsSignature() !== MODEL_INITIAL;
  const settingsSaved = await saveSettings(modelsChanged);
  if (modelsChanged) {
    const modelsSaved = await saveModels();
    if (modelsSaved) {
      MODEL_INITIAL = modelSettingsSignature();
      if (!settingsSaved) {
        $('save-note').innerHTML = `<div class="banner warn">Model settings saved. <b>Restart Fulloch</b> for the change to take effect.
          <button id="do-restart-global-models" class="primary" style="margin-left:0.5rem;padding:0.3rem 0.8rem">Restart now</button></div>`;
        $('do-restart-global-models').addEventListener('click', doRestart);
      }
    }
  }
}

  return { openSettings };
}

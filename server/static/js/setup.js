import { getJSON, $, screen, showAlert, clearAlert } from './setup-common.js';
import { createLifetime } from './browser-lifetime.js';
import { createVoiceManager } from './setup-voices.js';
import { createWizard } from './setup-wizard.js';
import { createInstallProgress } from './setup-install.js';
import { createSettings } from './setup-settings.js';

const lifetime = createLifetime();
const state = {
  schema: null, preflight: null, status: null,
  tierChosen: false, curStep: 0, cameViaWizard: false,
  sel: {
    tier: 'cpu_local', models: null,
    wakeword: 'hey atticus', wakeword_pattern: '', wakeword_model: '', voice_clone: '',
    openai: { base_url: '', model: '', api_key: '' },
    ha: { url: '', token: '' }, search_url: '', clear_integrations: false,
  },
};
const sel = state.sel;
const voice = createVoiceManager({ state });
const wizard = createWizard({ state, voice, doInstall: () => install.doInstall() });
const install = createInstallProgress({
  state, chosenModels: wizard.chosenModels, boot: () => boot(), stepFinish: wizard.stepFinish,
});
const settings = createSettings({
  state, voice, doRestart: install.doRestart, doRestartToHttps: install.doRestartToHttps,
});
const { stepBrain } = wizard;
const { showProgress, showLoading } = install;
const { openSettings } = settings;
// These two callbacks are the public contract of the download error HTML.
window.retryDownload = install.retryDownload;
window.saveHfTokenAndRetry = install.saveHfTokenAndRetry;
// ---- entry ----------------------------------------------------------------
async function boot() {
  install.stopPolling();
  voice.stop();
  let status;
  try { status = await getJSON('/status'); }
  catch (e) { screen().innerHTML = `<div class="card">Waiting for server…</div>`; lifetime.setTimeout(boot, 1500); return; }
  if (lifetime.destroyed) return;
  state.status = status;

  // Only initial setup may seed an unset household timezone from this browser.
  if (status.phase !== 'READY') {
    fetch('/setup/timezone', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tz: Intl.DateTimeFormat().resolvedOptions().timeZone }) }).catch(() => {});
  }

  if (status.phase === 'READY') { return openSettings(); }
  if (status.phase === 'DOWNLOADING') { return showProgress(); }
  if (status.phase === 'LOADING') { return showLoading(); }

  $('subtitle').textContent = 'First-run setup';
  state.schema = await getJSON('/setup/schema');
  try { state.preflight = await getJSON('/setup/preflight'); } catch (e) { state.preflight = null; }
  if (lifetime.destroyed) return;
  if (status.phase === 'ERROR') {
    showAlert(status.detail || 'The current configuration could not start.');
  } else {
    clearAlert();
  }
  // Seed wakeword from any existing config
  const wakeF = state.schema.fields.find(f => f.path === 'general.wakeword');
  if (wakeF && wakeF.value) sel.wakeword = wakeF.value;
  const wakePatF = state.schema.fields.find(f => f.path === 'general.wakeword_pattern');
  if (wakePatF && wakePatF.value) sel.wakeword_pattern = wakePatF.value;
  // Seed voice from any existing config
  const voiceF = state.schema.fields.find(f => f.path === 'general.voice_clone');
  if (voiceF && voiceF.value) sel.voice_clone = voiceF.value;
  // Seed integration fields from existing config
  const haUrl = state.schema.fields.find(f => f.path === 'home_assistant.url');
  if (haUrl && haUrl.value) sel.ha.url = haUrl.value;
  const searchUrl = state.schema.fields.find(f => f.path === 'search.searxng_url');
  if (searchUrl && searchUrl.value) sel.search_url = searchUrl.value;
  // Only an exact preset match gets a tier card. A backend-only match would
  // discard custom paths, context sizes, or remote endpoint options on save.
  if (state.schema.models) {
    sel.models = JSON.parse(JSON.stringify(state.schema.models));
    const wakewordModel = sel.models.wakeword || {};
    if (wakewordModel.backend === 'openwakeword') sel.wakeword_model = wakewordModel.model || '';
    const matched = (state.schema.tier_presets || []).find(t =>
      JSON.stringify(t.models) === JSON.stringify(sel.models));
    sel.tier = matched ? matched.id : 'custom';
    state.tierChosen = true;
    if (sel.models.llm && (sel.models.llm.backend === 'openai' || sel.models.llm.backend === 'external')) {
      sel.openai.base_url = sel.models.llm.base_url || '';
      sel.openai.model = sel.models.llm.model || '';
    }
  }
  const defaultWakePreset = (state.schema.wakeword_presets || []).find(p => p.recommended);
  if (!sel.wakeword_model && !(state.schema.models && state.schema.models.wakeword)
      && defaultWakePreset && sel.wakeword === defaultWakePreset.wakeword) {
    sel.wakeword_pattern = defaultWakePreset.pattern;
    sel.wakeword_model = defaultWakePreset.model;
  }
  stepBrain();
}

boot();
window.addEventListener('pagehide', () => {
  lifetime.destroy();
  install.destroy();
  voice.destroy();
}, { once: true });
window.addEventListener('pageshow', event => { if (event.persisted) location.reload(); });

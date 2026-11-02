import { $, el, screen, renderSteps, credPlaceholder } from './setup-common.js';
import { clone, escapeHtml as esc, offeredTiers, recommendedTierId, applyWakewordPreset,
  wakewordOptions, wakewordWarning, info, gatedNotice } from './setup-models.js';
import { integrationCard, wireIntegrationCard } from './setup-ha-integration.js';

export function createWizard({ state, voice, doInstall }) {
  const sel = state.sel;
  const preset = () => (state.schema.tier_presets || []).find(t => t.id === sel.tier);

  function chosenModels() {
    const models = clone(sel.models || preset()?.models || {});
    const wake = (state.schema.wakeword_presets || []).find(p => p.id === sel.wakeword_id);
    return applyWakewordPreset(models, wake);
  }

  function renderWarning() {
    const fit = (state.preflight?.tier_fit || []).find(t => t.id === sel.tier);
    $('tier-warn').textContent = fit?.badge === 'warn' ? fit.reason : '';
    $('wakeword-warning').textContent = wakewordWarning(state.schema, chosenModels());
    $('tier-access').innerHTML = gatedNotice(state.schema, chosenModels());
    const wake = state.schema.wakeword_presets.find(p => p.id === sel.wakeword_id);
    $('wakeword-license').textContent = wake?.license ? `Model license: ${wake.license}. Runs locally; no connection to Amazon or other voice services.` : '';
  }

  function stepBrain() {
    voice.stop();
    state.curStep = 0; renderSteps(0);
    const tiers = offeredTiers(state.schema);
    const recommended = recommendedTierId(state.schema, state.preflight);
    if (!state.tierChosen) {
      sel.tier = recommended;
      sel.models = clone(preset().models);
    }
    const wakewords = state.schema.wakeword_presets || [];
    if (!sel.wakeword_id && !state.schema.models) {
      sel.wakeword_id = (wakewords.find(p => p.recommended && p.model) || wakewords.find(p => p.model))?.id || '';
    }
    const existing = sel.tier === 'custom' ? '<option value="custom">Configured in config.yml (unchanged)</option>' : '';
    screen().innerHTML = '';
    screen().appendChild(el(`<div class="card">
      <h2>Choose a preset</h2>
      <p class="lead">The best available preset for this system is selected. Continue with these defaults, or choose another option.</p>
      <label for="tier-select">Preset ${info('GPU labels are hardware capacity classes, not exact memory usage. Lite needs no GPU. Model files download once; inference is local.')}</label>
      <select id="tier-select">${existing}${tiers.map(t => `<option value="${t.id}"${t.id === sel.tier ? ' selected' : ''}>${esc(t.label)}${t.id === recommended ? ' — recommended' : ''}</option>`).join('')}</select>
      <p class="help" id="tier-description"></p><p class="help warning" id="tier-warn"></p><div id="tier-access"></div>
      <label for="wizard-wakeword">Wakeword ${info('Choose a bundled acoustic wakeword model. Custom names, model paths, regex and detection tuning are available only in config.yml.')}</label>
      <select id="wizard-wakeword">${wakewordOptions(state.schema, sel.wakeword_id, !sel.wakeword_id)}</select>
      <p class="help warning" id="wakeword-warning"></p><p class="help" id="wakeword-license"></p>
      <div class="actions"><button id="wizard-back" class="back">Go back</button><button id="next1" class="primary">Continue</button></div>
    </div>`));
    const describe = () => { $('tier-description').textContent = preset()?.blurb || 'Your existing model configuration will be preserved.'; renderWarning(); };
    $('tier-select').addEventListener('change', e => {
      sel.tier = e.target.value;
      const wakeword = sel.models?.wakeword;
      sel.models = clone(preset().models);
      if (wakeword) sel.models.wakeword = clone(wakeword);
      state.tierChosen = true;
      describe();
    });
    $('wizard-wakeword').addEventListener('change', e => { sel.wakeword_id = e.target.value; renderWarning(); });
    $('wizard-back').addEventListener('click', () => history.back());
    $('next1').addEventListener('click', stepHomeAssistant);
    describe();
  }

  function stepHomeAssistant() {
    voice.stop(); state.curStep = 1; renderSteps(1);
    screen().innerHTML = '';
    screen().appendChild(el(`<div class="card">
      <h2>Home Assistant ${info('Optional. Connect your Home Assistant server to control household devices. Leave blank to keep existing settings or continue without it.')}</h2>
      <label for="ha-url">URL ${info('Use the server’s LAN address, such as http://192.168.1.50:8123. Inside Docker, localhost refers to the container itself.')}</label>
      <input id="ha-url" type="url" placeholder="http://192.168.1.50:8123" value="${esc(sel.ha.url)}">
      <label for="ha-token">Home Assistant access token ${info('Create a long-lived access token under your Home Assistant profile’s Security tab. Fulloch uses this to control Home Assistant; it is not the Fulloch integration token below.')}</label>
      <input id="ha-token" type="password" autocomplete="off" placeholder="${esc(credPlaceholder(state.schema.credentials?.ha_token, 'Optional'))}" value="${esc(sel.ha.token)}">
      ${integrationCard()}
      <div class="actions"><button id="ha-back" class="back">Go back</button><button id="ha-next" class="primary">Continue</button></div>
    </div>`));
    $('ha-url').addEventListener('input', e => sel.ha.url = e.target.value.trim());
    $('ha-token').addEventListener('input', e => sel.ha.token = e.target.value.trim());
    wireIntegrationCard();
    $('ha-back').addEventListener('click', stepBrain);
    $('ha-next').addEventListener('click', stepSearch);
  }

  function stepSearch() {
    voice.stop(); state.curStep = 2; renderSteps(2);
    screen().innerHTML = '';
    screen().appendChild(el(`<div class="card">
      <h2>Web search ${info('Optional. A SearXNG endpoint provides current web results. You can configure it later in Settings; it is not required for local commands.')}</h2>
      <label for="search-url">URL ${info('Enter your SearXNG search URL. It must return the SearXNG JSON search format; arbitrary search-provider websites are not compatible.')}</label>
      <input id="search-url" type="url" placeholder="http://your-searxng:8080/search" value="${esc(sel.search_url)}">
      <div class="actions"><button id="search-back" class="back">Go back</button><button id="get-started" class="primary">Continue</button></div>
    </div>`));
    $('search-url').addEventListener('input', e => sel.search_url = e.target.value.trim());
    $('search-back').addEventListener('click', stepHomeAssistant);
    $('get-started').addEventListener('click', doInstall);
  }

  return { stepBrain, chosenModels };
}

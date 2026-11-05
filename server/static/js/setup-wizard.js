import { $, el, screen, renderSteps, credPlaceholder, postJSON } from './setup-common.js';
import { clone, escapeHtml as esc, offeredTiers, recommendedTierId, applyWakewordPreset,
  wakewordOptions, wakewordWarning, info, gatedNotice } from './setup-models.js';
import { integrationCard, wireIntegrationCard } from './setup-ha-integration.js';
import { createLanguageSelect } from './language-select.js';

export function createWizard({ state, voice, doInstall }) {
  const sel = state.sel;
  const preset = () => (state.schema.tier_presets || []).find(t => t.id === sel.tier);
  let preparationGeneration = 0;
  const intentToTts = { en: 'english', fr: 'french', de: 'german', it: 'italian', pt: 'portuguese', es: 'spanish' };

  function refreshTtsLanguage() {
    const backend = sel.models?.tts?.backend || state.schema.default_backends?.tts;
    const supported = state.schema.tts_model_languages?.[backend] || [];
    const container = $('wizard-tts-language-field');
    const select = $('wizard-tts-language');
    container.hidden = !supported.length;
    if (!supported.length) return;
    if (!sel.tts_language_manual || !supported.includes(sel.tts_language)) {
      sel.tts_language = sel.intent_languages
        .map(code => intentToTts[code.toLowerCase().split('-')[0]])
        .find(language => language && language !== 'english' && supported.includes(language))
        || (supported.includes('english') ? 'english' : supported[0]);
    }
    select.innerHTML = supported.map(language => `<option value="${esc(language)}"${language === sel.tts_language ? ' selected' : ''}>${esc(language.charAt(0).toUpperCase() + language.slice(1))}</option>`).join('');
  }

  function showLanguagePreparation() {
    const node = $('intent-preparation');
    if (node) node.textContent = state.intentPreparationStatus || '';
  }

  function prepareLanguages() {
    const generation = ++preparationGeneration;
    const languages = [...sel.intent_languages];
    if (!languages.some(code => code !== 'en')) {
      state.intentPreparationStatus = '';
      state.intentPreparation = Promise.resolve(true);
      return;
    }
    state.intentPreparationStatus = 'Preparing selected Home Assistant languages in the background…';
    state.intentPreparation = postJSON('/setup/intent-languages', { languages }).then(response => {
      if (!response.ok) throw new Error('Download failed');
      if (generation === preparationGeneration) state.intentPreparationStatus = 'Home Assistant languages are ready.';
      return true;
    }).catch(() => {
      if (generation === preparationGeneration) state.intentPreparationStatus = 'Could not prepare languages. Go back and continue to retry.';
      return false;
    }).finally(() => { if (generation === preparationGeneration) showLanguagePreparation(); });
  }

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
      <label>Home Assistant intent languages ${info('Choose the languages used to match Home Assistant commands. Speech recognition and spoken responses depend on the selected speech models. Additional language tables are downloaded once.')}</label>
      <div id="wizard-intent-languages"></div>
      <div id="wizard-tts-language-field" hidden><label for="wizard-tts-language">TTS model language ${info('Select the voice model weights. Defaults to the first supported non-English intent language. French uses a larger, slower model. This does not translate responses.')}</label>
      <select id="wizard-tts-language"></select></div>
      <div class="actions"><button id="wizard-back" class="back">Go back</button><button id="next1" class="primary">Continue</button></div>
    </div>`));
    const languages = createLanguageSelect({ options: state.schema.intent_languages, value: sel.intent_languages,
      onChange: value => { sel.intent_languages = value; refreshTtsLanguage(); } });
    $('wizard-intent-languages').appendChild(languages.element);
    $('wizard-tts-language').addEventListener('change', e => { sel.tts_language = e.target.value; sel.tts_language_manual = true; });
    const describe = () => { $('tier-description').textContent = preset()?.blurb || 'Your existing model configuration will be preserved.'; renderWarning(); refreshTtsLanguage(); };
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
    $('next1').addEventListener('click', () => { prepareLanguages(); stepHomeAssistant(); });
    describe();
  }

  function stepHomeAssistant() {
    voice.stop(); state.curStep = 1; renderSteps(1);
    screen().innerHTML = '';
    screen().appendChild(el(`<div class="card">
      <h2>Home Assistant ${info('Optional. Connect your Home Assistant server to control household devices. Leave blank to keep existing settings or continue without it.')}</h2>
      <p id="intent-preparation" class="help" role="status"></p>
      <label for="ha-url">URL ${info('Use the server’s LAN address, such as http://192.168.1.50:8123. Inside Docker, localhost refers to the container itself.')}</label>
      <input id="ha-url" type="url" placeholder="http://192.168.1.50:8123" value="${esc(sel.ha.url)}">
      <label for="ha-token">Home Assistant access token ${info('Create a long-lived access token under your Home Assistant profile’s Security tab. Fulloch uses this to control Home Assistant; it is not the Fulloch integration token below.')}</label>
      <input id="ha-token" type="password" autocomplete="off" placeholder="${esc(credPlaceholder(state.schema.credentials?.ha_token, 'Optional'))}" value="${esc(sel.ha.token)}">
      ${integrationCard()}
      <div class="actions"><button id="ha-back" class="back">Go back</button><button id="ha-next" class="primary">Continue</button></div>
    </div>`));
    showLanguagePreparation();
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
      <p id="intent-preparation" class="help" role="status"></p>
      <label for="search-url">URL ${info('Enter your SearXNG search URL. It must return the SearXNG JSON search format; arbitrary search-provider websites are not compatible.')}</label>
      <input id="search-url" type="url" placeholder="http://your-searxng:8080/search" value="${esc(sel.search_url)}">
      <div class="actions"><button id="search-back" class="back">Go back</button><button id="get-started" class="primary">Continue</button></div>
    </div>`));
    showLanguagePreparation();
    $('search-url').addEventListener('input', e => sel.search_url = e.target.value.trim());
    $('search-back').addEventListener('click', stepHomeAssistant);
    $('get-started').addEventListener('click', doInstall);
  }

  return { stepBrain, chosenModels };
}

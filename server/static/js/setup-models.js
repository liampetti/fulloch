// Shared preset selection and wakeword policy for wizard and settings.
export const clone = value => JSON.parse(JSON.stringify(value));
export const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function offeredTiers(schema) {
  return (schema.tier_presets || []).filter(t => t.offerable !== false
    && (schema.variant !== 'cpu' || !t.vram_class_gb));
}

export function recommendedTierId(schema, preflight) {
  if (schema.variant !== 'cpu' && preflight?.gpu?.available) {
    const fits = preflight.tier_fit || [];
    const tier = offeredTiers(schema).find(t => t.vram_class_gb > 0
      && fits.some(f => f.id === t.id && f.badge === 'ok'));
    if (tier) return tier.id;
  }
  return 'lite';
}

export function matchingTier(schema, models) {
  if (!models) return null;
  // Ignore the independent wakeword choice, but do not call a stack with
  // custom model paths/context/decoding options an exact preset.
  return offeredTiers(schema).find(t => ['asr', 'tts', 'llm'].every(domain => {
    const current = models[domain] || {};
    const preset = t.models[domain];
    return Object.keys(current).every(key => current[key] === preset[key])
      && Object.keys(preset).every(key => current[key] === preset[key]
        || (preset[key] === false && current[key] === undefined));
  }));
}

export function selectedWakeword(schema, models, phrase, pattern) {
  return (schema.wakeword_presets || []).find(p => p.wakeword === phrase
    && (!pattern || p.pattern === pattern)
    && models?.wakeword?.backend === 'openwakeword' && models.wakeword.model === p.model);
}

export function applyWakewordPreset(models, preset) {
  if (!preset) return models; // Preserve config-file-only custom activation.
  if (models.wakeword?.backend !== 'openwakeword' || models.wakeword.model !== preset.model) {
    models.wakeword = { backend: 'openwakeword', model: preset.model,
      threshold: preset.id === 'hey_atticus' ? 0.7 : 0.5, smoothing_frames: 1, cooldown_ms: 1500 };
  }
  return models;
}

export function wakewordWarning(schema, models) {
  const asr = (schema.backends?.asr || []).find(b => b.backend === models?.asr?.backend);
  const wake = models?.wakeword;
  const verifies = wake?.verify_asr ?? (asr?.verify_asr_wakeword !== false);
  return wake?.backend === 'openwakeword' && !verifies
    ? 'This ASR model is not gating wakeword detection. The acoustic model detects the wakeword on its own; false activations may be more likely.' : '';
}

export function wakewordOptions(schema, selectedId, preserve = false) {
  const existing = preserve ? '<option value="">Configured in config.yml (unchanged)</option>' : '';
  return existing + (schema.wakeword_presets || []).filter(p => p.model).map(p =>
    `<option value="${escapeHtml(p.id)}"${p.id === selectedId ? ' selected' : ''}>${escapeHtml(p.label)}</option>`).join('');
}

export function info(text) {
  return `<details class="info-popover"><summary aria-label="More information">i</summary><p>${escapeHtml(text)}</p></details>`;
}

export function gatedModels(schema, models) {
  return ['asr', 'tts', 'llm'].flatMap(domain => {
    const block = models?.[domain] || {};
    const id = domain === 'llm' && block.backend === 'local'
      ? ({ qwen: 'llama', custom: 'llama' }[block.local_model] || block.local_model || 'llama')
      : block.backend === 'external' ? 'openai' : block.backend;
    const spec = (schema.backends?.[domain] || []).find(b => b.backend === id);
    // An explicit filesystem path bypasses the Hugging Face asset download;
    // an explicit repo ID (including the default) does not.
    const localPath = typeof block.model === 'string' && /^(?:\/|\.\.?\/)/.test(block.model);
    return spec?.gated_repo && !localPath ? [spec] : [];
  });
}

export function gatedNotice(schema, models) {
  const specs = gatedModels(schema, models);
  if (!specs.length) return '';
  return specs.map(spec => `<div class="access-notice" role="status"><strong>Hugging Face access required</strong>
    <p>${escapeHtml(spec.display_name)} uses gated weights. Before downloading, accept the terms at
    <a href="https://huggingface.co/${encodeURI(spec.gated_repo)}" target="_blank" rel="noopener noreferrer">${escapeHtml(spec.gated_repo)}</a>
    using your Hugging Face account. If prompted during download, provide a read token for that account. Already-downloaded weights can be reused.</p></div>`).join('');
}

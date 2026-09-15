import { getJSON, postJSON, $, el } from './setup-common.js';

export function createVoiceManager({ state }) {
  const sel = state.sel;
  let generation = 0;
  let generatedAudio = null;
  let generatedUrl = null;
  const clearGenerated = () => {
    generatedAudio?.pause();
    generatedAudio = null;
    if (generatedUrl) URL.revokeObjectURL(generatedUrl);
    generatedUrl = null;
  };
  const stop = () => { generation++; _voiceStop(); clearGenerated(); };
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
  const current = generation;
  const { voices } = await getJSON('/setup/voices');
  const s = $('voice-sel');
  if (!s || current !== generation) return;
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
  const current = generation;
  const instruct = $('gv-instruct').value.trim();
  if (!instruct) { $('gv-status').textContent = 'Enter a description.'; return; }
  $('gv-status').textContent = 'Generating… (this can take a while)';
  $('gv-generate').disabled = true;
  try {
    const r = await postJSON('/setup/voice', { instruct, phrase: $('gv-phrase').value.trim() });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.status); }
    const blob = await r.blob();
    if (current !== generation) return;
    clearGenerated();
    $('gv-audio').innerHTML = '';
    const audio = el('<audio controls autoplay></audio>');
    generatedAudio = audio;
    audio.src = generatedUrl = URL.createObjectURL(blob);
    $('gv-audio').appendChild(audio);
    $('gv-save').style.display = '';
    $('gv-status').textContent = 'Preview ready.';
  } catch (e) {
    if (current === generation) $('gv-status').textContent = 'Generation failed: ' + e.message;
  } finally { if (current === generation && $('gv-generate')) $('gv-generate').disabled = false; }
}

async function saveVoice() {
  const current = generation;
  const name = $('gv-name').value.trim();
  if (!name) return;
  const r = await postJSON('/setup/voice/save', { name });
  if (!r.ok) { const e = await r.json().catch(() => ({})); alert('Save failed: ' + (e.detail || r.status)); return; }
  const { saved } = await r.json();
  if (current !== generation) return;
  clearGenerated();
  $('gen-panel').innerHTML = '';
  await refreshVoiceList(saved);
}
async function populateVoiceField() {
  const sel = document.querySelector('#settings-config select[data-voices="qwen"]');
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

  return { KOKORO_RECOMMENDED, KOKORO_VOICES, kokoroOption, stop,
    makeVoicePreview, refreshVoiceList, showGenPanel, populateVoiceField, destroy: stop };
}

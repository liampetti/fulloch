import { createLifetime } from './browser-lifetime.js';

export function createBrowserSatellite({ clearEmpty, scrollEnd }) {
  // ---- Browser satellite (push-to-talk via WebSocket) ---------------------
  // satellite-btn:    toggle the mic+speaker link.
  // conversation-mode-toggle: enable/disable exclusive full-duplex Conversation
  //                           mode; disabling it also disconnects Voice mode.
  // Protocol (binary = Float32 PCM; text = JSON control):
  //   browser → server:  Float32 chunks at 16 kHz mono
  //   browser → server:  {"type":"conversation_mode.set","enabled":<bool>}
  //   server  → browser: {"type":"session","satellite_id":<str>} — sent once,
  //                      right after connect; lets this tab tell (via
  //                      /status's active_owner_id) whether a busy turn is
  //                      its own or another satellite's
  //   server  → browser: {"type":"tts_start","sr":<int>}
  //                      <binary Float32 chunks>
  //   browser → server:  {"type":"tts_credit","seconds":<float>}
  //                      {"type":"tts_end"} — all chunks sent; browser
  //                      playback may continue until `satPlayAt`
  //                      {"type":"tts_cancel"}  — barge-in: stop already-
  //                      scheduled playback immediately

  const lifetime = createLifetime();
  const chat = document.getElementById('chat');
  const satBtn = document.getElementById('satellite-btn');
  const conversationModeBtn = document.getElementById('conversation-mode-toggle');
  let satWs = null;
  let satConnecting = false;
  let connectionGeneration = 0;
  let resolveConnection = null;
  let satPendingConversationMode = null;
  let satAudioCtx = null;
  let satMicStream = null;
  let satWorkletNode = null;
  let satPlayAt = 0;        // AudioContext scheduled-end time for TTS chunks
  let satTtsSr = 24000;     // sample rate announced by server in tts_start
  let satTtsActive = false; // false after cancel/end, so stale PCM is ignored
  let satScheduledSources = [];  // AudioBufferSourceNodes pending/playing, so tts_cancel can stop them
  let satPendingPcm = [];
  let satPendingSamples = 0;
  const SAT_PLAYBACK_BATCH_SECONDS = 0.24;
  let satPlaybackGeneration = 0;
  let satMicMuted = false;  // true during TTS playback — stops mic data to prevent echo
  let satMicResumeTimer = null;  // setTimeout handle for delayed unmute after satPlayAt
  let satHalfDuplex = true; // false when barge-in or Conversation mode keeps the mic live
  // Conversation mode only exists while Voice mode is connected. Do not revive
  // a stale preference after a page reload.
  let conversationModeOverride = '0';
  let conversationMode = false;
  let satPlaybackOnly = false; // replay speaker connection; no microphone stream
  let mySatelliteId = null; // this tab's own id, from the "session" frame
  let satHeartbeatTimer = null;
  const SAT_HEARTBEAT_MS = 15000;

  // Browser-satellite area picker (6b) — a one-time chat bubble asking which
  // HA zone this device sits in, so bare "turn off the lights" can default
  // to the right room. Thin/native satellite clients configure this via
  // YAML instead; this picker only exists for the browser path.
  const SAT_AREA_KEY = 'sat_ha_area';           // chosen area id, '' = none/skipped
  const SAT_AREA_DECIDED_KEY = 'sat_ha_area_decided'; // '1' once the user has picked or skipped
  let satHaArea = localStorage.getItem(SAT_AREA_KEY) || '';
  let satAreaName = '';   // display name for the pill, resolved once areas load
  const satAreaPill = document.getElementById('sat-area-pill');

  const SAT_WORKLET = `
class ResampleTo16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ratio = sampleRate / 16000;
    this._buf = [];
    this._target = Math.round(16000 * 0.2);
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const outLen = Math.floor(ch.length / this._ratio);
    for (let i = 0; i < outLen; i++) {
      const p = i * this._ratio;
      const lo = p | 0, hi = Math.min(lo + 1, ch.length - 1);
      this._buf.push(ch[lo] + (ch[hi] - ch[lo]) * (p - lo));
    }
    while (this._buf.length >= this._target) {
      const f32 = new Float32Array(this._buf.splice(0, this._target));
      this.port.postMessage(f32.buffer, [f32.buffer]);
    }
    return true;
  }
}
registerProcessor('fulloch-resample', ResampleTo16k);
`;

  const syncSatBtn = () => {
    if (!satWs) {
      satBtn.classList.remove('active', 'always-on');
      satBtn.setAttribute('aria-label', 'Voice mode — click to connect the mic');
    } else if (satPlaybackOnly) {
      satBtn.classList.remove('active', 'always-on');
      satBtn.setAttribute('aria-label', 'Speaker connected for replay; click to disconnect');
    } else if (conversationMode) {
      satBtn.classList.remove('active');
      satBtn.classList.add('always-on');
      satBtn.setAttribute('aria-label', 'Voice mode — Conversation mode active');
    } else {
      satBtn.classList.remove('always-on');
      satBtn.classList.add('active');
      satBtn.setAttribute('aria-label', 'Voice mode — listening for the wakeword');
    }
    conversationModeBtn.classList.toggle('active', conversationMode);
    conversationModeBtn.setAttribute('aria-pressed', conversationMode ? 'true' : 'false');
  };

  const setConversationModePreference = (enabled) => {
    conversationMode = enabled;
    localStorage.setItem('conversation_mode', enabled ? '1' : '0');
    conversationModeOverride = enabled ? '1' : '0';
  };

  const satSchedulePendingPcm = () => {
    if (!satAudioCtx || !satPendingSamples) return;
    const buf = satAudioCtx.createBuffer(1, satPendingSamples, satTtsSr);
    const samples = buf.getChannelData(0);
    let offset = 0;
    for (const pcm of satPendingPcm) {
      samples.set(pcm, offset);
      offset += pcm.length;
    }
    satPendingPcm = [];
    satPendingSamples = 0;
    const src = satAudioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(satAudioCtx.destination);
    const now = satAudioCtx.currentTime;
    const start = Math.max(satPlayAt, now + 0.02);
    src.onended = () => {
      const i = satScheduledSources.indexOf(src);
      if (i !== -1) satScheduledSources.splice(i, 1);
    };
    satScheduledSources.push(src);
    src.start(start);
    satPlayAt = start + buf.duration;
  };

  const satScheduleChunk = (f32) => {
    satPendingPcm.push(f32);
    satPendingSamples += f32.length;
    if (satPendingSamples >= Math.ceil(satTtsSr * SAT_PLAYBACK_BATCH_SECONDS)) {
      satSchedulePendingPcm();
    }
  };

  const satSetAudioProcessing = (enabled) => {
    if (!satMicStream) return;
    for (const track of satMicStream.getAudioTracks()) {
      track.applyConstraints({ echoCancellation: enabled, noiseSuppression: enabled }).catch(() => {});
    }
  };

  // Barge-in: stop everything already scheduled/playing and reset playback
  // timing so the next turn's audio starts fresh instead of queuing behind
  // the cut-off reply.
  const satCancelPlayback = () => {
    satPlaybackGeneration += 1;
    for (const src of satScheduledSources) {
      try { src.stop(); } catch (_) { /* already ended */ }
    }
    satScheduledSources = [];
    satPendingPcm = [];
    satPendingSamples = 0;
    satTtsActive = false;
    satPlayAt = satAudioCtx ? satAudioCtx.currentTime : 0;
  };

  const satMuteMicForPlayback = () => {
    if (satMicResumeTimer !== null) {
      lifetime.clearTimeout(satMicResumeTimer);
      satMicResumeTimer = null;
    }
    if (satHalfDuplex) satMicMuted = true;
  };

  const satUnmuteMicAfterPlayback = () => {
    if (!satHalfDuplex) {
      satMicMuted = false;
      return;
    }
    // Wait until all scheduled TTS audio has actually finished playing
    // (satPlayAt is the AudioContext time of the last chunk's end), then
    // unmute. tts_end only means the server finished sending chunks — the
    // browser may still be playing them for several more seconds.
    const delayMs = satAudioCtx ? Math.max(0, satPlayAt - satAudioCtx.currentTime) * 1000 + 100 : 0;
    satMicResumeTimer = lifetime.setTimeout(() => {
      satMicResumeTimer = null;
      satMicMuted = false;
    }, delayMs);
  };

  const satDisconnect = (clearConversationMode = true) => {
    connectionGeneration++;
    satConnecting = false;
    resolveConnection?.(false);
    resolveConnection = null;
    satCancelPlayback();
    if (satHeartbeatTimer !== null) { lifetime.clearInterval(satHeartbeatTimer); satHeartbeatTimer = null; }
    if (satWorkletNode) {
      satWorkletNode.port.onmessage = null;
      satWorkletNode.port.close();
      try { satWorkletNode.disconnect(); } catch(_) {}
      satWorkletNode = null;
    }
    if (satMicStream) { satMicStream.getTracks().forEach(t => t.stop()); satMicStream = null; }
    if (satWs) { try { satWs.close(); } catch(_) {} satWs = null; }
    if (satAudioCtx) { try { satAudioCtx.close(); } catch(_) {} satAudioCtx = null; }
    satPlayAt = 0;
    satScheduledSources = [];
    satPendingPcm = [];
    satPendingSamples = 0;
    satTtsActive = false;
    satMicMuted = false;
    if (satMicResumeTimer !== null) { lifetime.clearTimeout(satMicResumeTimer); satMicResumeTimer = null; }
    satHalfDuplex = true;
    satPlaybackOnly = false;
    mySatelliteId = null;
    satPendingConversationMode = null;
    if (clearConversationMode) setConversationModePreference(false);
    syncSatBtn();
  };

  const satConnect = async (playbackOnly = false) => {
    if (lifetime.destroyed) return false;
    // getUserMedia is asynchronous, so two quick Voice/Conversation taps used
    // to create competing streams and sockets. Keep one activation path.
    if (satConnecting) return false;
    if (satWs) {
      if (!playbackOnly && satPlaybackOnly) {
        // Replay leaves a speaker-only socket open. Upgrade it directly when
        // Voice mode is requested instead of making the user click twice.
        satDisconnect();
        return satConnect();
      }
      if (playbackOnly && satWs.readyState === WebSocket.OPEN) return true;
      if (playbackOnly && satWs.readyState === WebSocket.CONNECTING) {
        return new Promise(resolve => {
          const finish = result => {
            lifetime.clearInterval(wait);
            lifetime.signal.removeEventListener('abort', aborted);
            resolve(result);
          };
          const aborted = () => finish(false);
          const wait = lifetime.setInterval(() => {
            if (!satWs || satWs.readyState === WebSocket.CLOSED) {
              finish(false);
            } else if (satWs.readyState === WebSocket.OPEN) {
              finish(true);
            }
          }, 25);
          lifetime.signal.addEventListener('abort', aborted, { once: true });
        });
      }
      satDisconnect();
      return false;
    }
    satConnecting = true;
    const generation = ++connectionGeneration;
    const current = () => !lifetime.destroyed && generation === connectionGeneration;
    satPlaybackOnly = playbackOnly;
    try {
    if (!playbackOnly) {
      // Safari only reliably unlocks an AudioContext when resume() is called
      // synchronously from the button gesture. Do this before awaiting the mic
      // permission prompt, otherwise iOS can leave audio suspended or routed to
      // its call/earpiece path until a later interaction.
      satAudioCtx = new AudioContext({ latencyHint: 'interactive' });
      const resumeAudio = satAudioCtx.resume().catch(() => {});
      try {
        // Pin the mic stream to mono and 16 kHz. Conversation mode enables the
        // browser's echo/noise processing; normal mode keeps the raw mic path.
        // Defaults (audio: true) work on macOS/Windows but on Linux/Chrome,
        // PulseAudio/PipeWire's webrtc-audio-processing is rougher than Core
        // Audio and introduces audible dropouts / AGC pumping. We run Silero
        // VAD and a noise baseline server-side, so the browser's AGC/EC/NS is
        // actively harmful here. Pinning sampleRate to 16 kHz also makes the
        // worklet's resample ratio exactly 1.0 — the loop becomes a copy and
        // we stop doing linear-interp on every quantum.
        //
        // No navigator.audioSession manipulation: previous versions let Safari
        // manage the audio session automatically, which correctly routes TTS
        // to the loudspeaker. Explicitly setting 'play-and-record' overrides
        // Safari's DefaultToSpeaker option and routes to the earpiece.
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: conversationMode,
            noiseSuppression: conversationMode,
            autoGainControl: false,
          },
          video: false,
        });
        if (!current()) { stream.getTracks().forEach(track => track.stop()); return false; }
        satMicStream = stream;
      } catch (e) {
        if (!current()) return false;
        console.error('Satellite: mic access denied', e);
        alert('Microphone access denied — check browser permissions.');
        satDisconnect();
        return false;
      }
      await resumeAudio;
      if (!current()) return false;
    } else {
      satAudioCtx = new AudioContext({ latencyHint: 'interactive' });
      // Replay is also user-initiated; unlock playback before opening the socket.
      await satAudioCtx.resume().catch(() => {});
      if (!current()) return false;
    }
    satPlayAt = 0;

    if (!playbackOnly) {
      // Load AudioWorklet for resampling mic to 16 kHz.
      const blob = new Blob([SAT_WORKLET], { type: 'application/javascript' });
      const blobUrl = URL.createObjectURL(blob);
      try {
        await satAudioCtx.audioWorklet.addModule(blobUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }
      if (!current()) return false;
      const src = satAudioCtx.createMediaStreamSource(satMicStream);
      satWorkletNode = new AudioWorkletNode(satAudioCtx, 'fulloch-resample');
      src.connect(satWorkletNode);
    }

    // Open WebSocket
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const conversation = conversationModeOverride === null ? '' : `conversation=${conversationMode ? '1' : '0'}`;
    const area = satHaArea ? `&area=${encodeURIComponent(satHaArea)}` : '';
    // area_name carries the human-readable room name (already resolved client-side
    // from /ha/areas) so the server can show it as a location pill on this
    // satellite's turns — the server only knows the HA area_id otherwise, and
    // resolving it back to a display name isn't worth a second HA round-trip.
    const areaName = satHaArea && satAreaName ? `&area_name=${encodeURIComponent(satAreaName)}` : '';
    const url = `${proto}://${location.host}/ws/satellite?${conversation}${area}${areaName}`;
    const ws = new WebSocket(url);
    satWs = ws;
    ws.binaryType = 'arraybuffer';
    let resolveConnected;
    const connected = new Promise(resolve => { resolveConnected = resolveConnection = resolve; });

    ws.onopen = () => {
      if (!current() || satWs !== ws) { ws.close(); resolveConnected(false); return; }
      // Stream resampled mic chunks to server (muted during TTS playback for
      // half-duplex — prevents the assistant hearing its own reply as input).
      if (satWorkletNode) {
        satWorkletNode.port.onmessage = (e) => {
          if (satWs && satWs.readyState === WebSocket.OPEN && !satMicMuted) {
            satWs.send(e.data);
          }
        };
      }
      satHeartbeatTimer = lifetime.setInterval(() => {
        if (satWs === ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'satellite.heartbeat' }));
        }
      }, SAT_HEARTBEAT_MS);
      syncSatBtn();
      if (satPendingConversationMode !== null) {
        ws.send(JSON.stringify({ type: 'conversation_mode.set', enabled: satPendingConversationMode }));
        satPendingConversationMode = null;
      }
      resolveConnected(true);
    };

    ws.onmessage = (e) => {
      if (!current() || satWs !== ws) return;
      if (typeof e.data === 'string') {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'session') {
            mySatelliteId = msg.satellite_id || null;
            satHalfDuplex = msg.half_duplex !== false;
            setConversationModePreference(!!msg.conversation_mode);
            satSetAudioProcessing(conversationMode);
            syncSatBtn();
          } else if (msg.type === 'conversation_mode.result') {
            if (msg.enabled) {
              setConversationModePreference(true);
            } else if (!msg.message) {
              setConversationModePreference(false);
            } else {
              alert(msg.message);
            }
            satHalfDuplex = msg.half_duplex !== false;
            satSetAudioProcessing(conversationMode);
            syncSatBtn();
          } else if (msg.type === 'error') {
            alert(msg.message || 'Voice connection unavailable.');
          } else if (msg.type === 'tts_start') {
            satTtsSr = msg.sr || 24000;
            satTtsActive = true;
            satPlayAt = satAudioCtx ? Math.max(satPlayAt, satAudioCtx.currentTime) : 0;
            satPlaybackGeneration += 1;
            satPendingPcm = [];
            satPendingSamples = 0;
            satMuteMicForPlayback();
          } else if (msg.type === 'tts_end' || msg.type === 'tts_cancel') {
            if (msg.type === 'tts_cancel') {
              satCancelPlayback();
            } else {
              satSchedulePendingPcm();
              satTtsActive = false;
            }
            satUnmuteMicAfterPlayback();
          }
        } catch(_) {}
      } else {
        // Binary Float32 PCM audio chunk from TTS
        if (satTtsActive) satScheduleChunk(new Float32Array(e.data));
      }
    };

    ws.onerror = (e) => { console.error('Satellite WS error', e); resolveConnected(false); };
    ws.onclose = () => {
      resolveConnected(false);
      // A just-closed replay socket must not tear down its replacement when
      // the user immediately enables Voice mode.
      if (satWs === ws) satDisconnect();
    };
    syncSatBtn();
    return await connected;
    } catch (error) {
      if (current()) {
        console.warn('Satellite connection failed', error);
        satDisconnect();
      }
      return false;
    } finally {
      if (current()) { satConnecting = false; resolveConnection = null; }
    }
  };

  const toggleConversationMode = async () => {
    if (conversationMode) {
      // Conversation mode is always paired with Voice mode, so turning either
      // one off closes the microphone connection and both controls reset.
      satDisconnect();
      return;
    }
    const enabled = true;
    if (satWs && satWs.readyState === WebSocket.OPEN) {
      satWs.send(JSON.stringify({ type: 'conversation_mode.set', enabled }));
      return;
    }
    setConversationModePreference(enabled);
    syncSatBtn();
    if (satWs && satWs.readyState === WebSocket.CONNECTING) {
      satPendingConversationMode = enabled;
      return;
    }
    // Conversation mode is a voice mode, not a dormant preference. Starting
    // it from a cold page must request the microphone in this same gesture.
    await satConnect();
  };

  const syncSatAreaPill = () => {
    if (satHaArea && satAreaName) {
      satAreaPill.textContent = `📍 ${satAreaName}`;
      satAreaPill.classList.add('shown');
    } else if (satHaArea) {
      // Area chosen but its display name hasn't resolved yet (e.g. page just
      // loaded, /ha/areas hasn't been fetched) — fall back to the raw id
      // rather than showing nothing.
      satAreaPill.textContent = `📍 ${satHaArea}`;
      satAreaPill.classList.add('shown');
    } else {
      satAreaPill.classList.remove('shown');
    }
  };

  const chooseArea = (id, name, wrapEl) => {
    satHaArea = id;
    satAreaName = name;
    localStorage.setItem(SAT_AREA_KEY, id);
    localStorage.setItem(SAT_AREA_DECIDED_KEY, '1');
    if (wrapEl) wrapEl.remove();
    syncSatAreaPill();
    // A live connection was opened under the old (or no) area — reconnect so
    // the new choice takes effect immediately instead of on next connect.
    if (satWs) { satDisconnect(false); satConnect(); }
  };

  const renderAreaPicker = (areas) => {
    clearEmpty();
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = "Which room is this device in? That way a bare "
      + '"turn off the lights" knows which room you mean.';
    const row = document.createElement('div');
    row.className = 'area-picker-buttons';
    for (const a of areas) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'area-picker-btn';
      btn.textContent = a.name;
      btn.addEventListener('click', () => chooseArea(a.id, a.name, wrap));
      row.appendChild(btn);
    }
    const skip = document.createElement('button');
    skip.type = 'button';
    skip.className = 'area-picker-btn skip';
    skip.textContent = 'Skip';
    skip.addEventListener('click', () => chooseArea('', '', wrap));
    row.appendChild(skip);
    wrap.append(bubble, row);
    chat.appendChild(wrap);
    scrollEnd();
  };

  let satAreasCache = null;
  const fetchHaAreas = async () => {
    if (satAreasCache) return satAreasCache;
    try {
      const r = await fetch('/ha/areas');
      const body = await r.json();
      satAreasCache = body.available ? (body.areas || []) : [];
    } catch (_) {
      satAreasCache = [];
    }
    // Resolve the pill's display name now that areas are known, in case a
    // choice was already persisted from a previous visit.
    if (satHaArea) {
      const match = satAreasCache.find((a) => a.id === satHaArea);
      if (match) { satAreaName = match.name; syncSatAreaPill(); }
    }
    return satAreasCache;
  };

  const maybeShowAreaPicker = async () => {
    if (localStorage.getItem(SAT_AREA_DECIDED_KEY) === '1') { await fetchHaAreas(); return; }
    const areas = await fetchHaAreas();
    if (!lifetime.destroyed && areas.length) renderAreaPicker(areas);
  };

  lifetime.listen(satAreaPill, 'click', async () => {
    const areas = await fetchHaAreas();
    if (!lifetime.destroyed && areas.length) renderAreaPicker(areas);
  });

  lifetime.listen(satBtn, 'click', () => {
    if (satWs || satConnecting) {
      satDisconnect();
      return;
    }
    // Voice mode alone must never restore a previous Conversation mode.
    setConversationModePreference(false);
    satConnect();
  });
  lifetime.listen(conversationModeBtn, 'click', toggleConversationMode);

  syncSatBtn();

  syncSatAreaPill();

  return { connect: satConnect, disconnect: satDisconnect, showAreaPicker: maybeShowAreaPicker, get id() { return mySatelliteId; }, destroy() { lifetime.destroy(); satDisconnect(); } };
}

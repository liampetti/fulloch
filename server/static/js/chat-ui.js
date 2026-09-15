import { escapeHtml } from './browser-utils.js';
import { createLifetime } from './browser-lifetime.js';
import { createArtifactCards } from './artifact-cards.js';

export function createChatUI({ connectSpeaker, onThinking }) {
  const lifetime = createLifetime();
  let stream = null;
  const cards = createArtifactCards({ submitCommand });
  const { renderArtifacts } = cards;
  const chat = document.getElementById('chat');
  const empty = document.getElementById('empty');
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  // When on, every turn's trace group + stats panel start expanded.
  let alwaysDetail = !!window.FULLOCH_DASHBOARD_PREFS.show_turn_details;

  let typingEl = null;
  let waiting = false;       // this page's text turn is in flight
  let voiceBusy = false;     // a voice/other turn is working, from /status
  let lastTs = 0;
  let startupMessage = null;

  // The send button doubles as a stop button while the agent is working.
  const SEND_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.4 20.4 22 12 3.4 3.6 3 10l13 2-13 2z"/></svg>';
  const STOP_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5"/></svg>';
  const syncButton = () => {
    const stop = waiting || voiceBusy;
    sendBtn.classList.toggle('stop', stop);
    sendBtn.disabled = false;
    sendBtn.setAttribute('aria-label', stop ? 'Stop' : 'Send');
    sendBtn.title = stop ? 'Stop' : 'Send';
    sendBtn.innerHTML = stop ? STOP_SVG : SEND_SVG;
  };
  const doStop = async () => {
    try { await fetch('/stop', { method: 'POST' }); }
    catch (e) { console.warn('stop failed', e); }
    // Optimistic: snap any in-flight typewriter to full, drop the typing
    // indicator, and revert the button. The `stopped` SSE event and /status
    // poll confirm the wind-down.
    finishTyping();
    removeTyping();
    waiting = false;
    voiceBusy = false;
    syncButton();
    input.focus();
  };
  let typingTimer = null;   // live typewriter tick handle
  let finishActive = null;  // finalises the in-flight typewriter, if any
  const statsMsgs = new Map();  // assistant ts -> {stats, panel, btn}
  // Per-turn trace group. Reset on every user event so the next batch of
  // agent events (plan/step/observation) gets gathered under its own
  // collapsible <details> block placed right after the user bubble.
  let activeTraceGroup = null;
  let activeTraceSummary = null;
  let activeTraceCount = 0;
  let pendingArtifacts = [];

  // Reverse TTS-friendly word forms back to a natural display.
  // Mirrors the inverse of `tools/time_tools.py:get_current_time` and
  // `core/datetime_utils.py:tts_friendly_event_summary`. Anything that
  // doesn't match falls through unchanged.
  const WORD_NUM = {
    zero:0, one:1, two:2, three:3, four:4, five:5, six:6, seven:7,
    eight:8, nine:9, ten:10, eleven:11, twelve:12, thirteen:13,
    fourteen:14, fifteen:15, sixteen:16, seventeen:17, eighteen:18,
    nineteen:19, twenty:20, thirty:30, forty:40, fifty:50,
  };
  const ALL_NUM = { ...WORD_NUM };
  for (const [t, tv] of Object.entries(WORD_NUM)) {
    if (tv >= 20 && tv % 10 === 0) {
      for (const [o, ov] of Object.entries(WORD_NUM)) {
        if (ov >= 1 && ov <= 9) ALL_NUM[`${t}-${o}`] = tv + ov;
      }
    }
  }
  const NUM_PAT = Object.keys(ALL_NUM).sort((a,b) => b.length - a.length).join('|');
  const DAY_ORD = {
    first:1, second:2, third:3, fourth:4, fifth:5, sixth:6, seventh:7,
    eighth:8, ninth:9, tenth:10, eleventh:11, twelfth:12, thirteenth:13,
    fourteenth:14, fifteenth:15, sixteenth:16, seventeenth:17,
    eighteenth:18, nineteenth:19, twentieth:20, 'twenty-first':21,
    'twenty-second':22, 'twenty-third':23, 'twenty-fourth':24,
    'twenty-fifth':25, 'twenty-sixth':26, 'twenty-seventh':27,
    'twenty-eighth':28, 'twenty-ninth':29, thirtieth:30, 'thirty-first':31,
  };
  const ORD_PAT = Object.keys(DAY_ORD).sort((a,b) => b.length - a.length).join('|');
  const CENTURY = 'eighteen|nineteen|twenty';

  const naturalize = (text) => {
    if (typeof text !== 'string' || !text) return text;
    let s = text;
    s = s.replace(/\b([ap]) m\b/g, (_, c) => c.toUpperCase() + 'M');
    s = s.replace(new RegExp(`\\b(${CENTURY}) hundred\\b`, 'g'),
      (_, c) => String(ALL_NUM[c] * 100));
    s = s.replace(new RegExp(`\\b(${CENTURY}) oh (${NUM_PAT})\\b`, 'g'),
      (m, c, r) => {
        const rv = ALL_NUM[r];
        return (rv >= 1 && rv <= 9) ? `${ALL_NUM[c]}0${rv}` : m;
      });
    s = s.replace(new RegExp(`\\b(${CENTURY}) (${NUM_PAT})\\b`, 'g'),
      (m, c, r) => {
        const rv = ALL_NUM[r];
        return (rv >= 10 && rv <= 99) ? `${ALL_NUM[c]}${rv}` : m;
      });
    s = s.replace(new RegExp(`\\b(${NUM_PAT}) oh (${NUM_PAT}) (AM|PM)\\b`, 'g'),
      (m, h, mw, ap) => {
        const hv = ALL_NUM[h], mv = ALL_NUM[mw];
        return (hv >= 1 && hv <= 12 && mv >= 1 && mv <= 9)
          ? `${hv}:0${mv} ${ap}` : m;
      });
    s = s.replace(new RegExp(`\\b(${NUM_PAT}) (${NUM_PAT}) (AM|PM)\\b`, 'g'),
      (m, h, mw, ap) => {
        const hv = ALL_NUM[h], mv = ALL_NUM[mw];
        return (hv >= 1 && hv <= 12 && mv >= 10 && mv <= 59)
          ? `${hv}:${String(mv).padStart(2,'0')} ${ap}` : m;
      });
    s = s.replace(new RegExp(`\\b(${NUM_PAT}) (AM|PM)\\b`, 'g'),
      (m, h, ap) => {
        const hv = ALL_NUM[h];
        return (hv >= 1 && hv <= 12) ? `${hv} ${ap}` : m;
      });
    s = s.replace(/\b(\d{1,2}) (\d{2}) (AM|PM)\b/g,
      (_, h, m, ap) => `${h}:${m} ${ap}`);
    s = s.replace(new RegExp(`\\b(${NUM_PAT}) o'clock\\b`, 'g'),
      (m, h) => {
        const hv = ALL_NUM[h];
        return (hv >= 1 && hv <= 12) ? `${hv} o'clock` : m;
      });
    s = s.replace(new RegExp(`\\b(${ORD_PAT})\\b`, 'g'),
      (m) => String(DAY_ORD[m]));
    return s;
  };

  const fmtTime = (ts) => {
    if (!Number.isFinite(ts)) return '';
    const d = new Date(ts * 1000);
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  };

  const clearEmpty = () => document.getElementById('empty')?.remove();

  const resetUI = () => {
    finishTyping();
    cards.clear();
    // Remove all message/trace/typing elements, restore the empty state
    [...chat.children].forEach(el => {
      if (el !== empty) el.remove();
    });
    if (!chat.contains(empty)) {
      // Re-create the empty state node if it was removed
      const div = document.createElement('div');
      div.className = 'empty';
      div.id = 'empty';
      div.innerHTML = `
        <div class="logo-lg"><img src="/logo.png" alt=""></div>
        <p class="wake-hint" id="wake-hint" hidden></p>
        <p>Use <b>Conversation</b> for hands-free chat without a wakeword. If rooms are available, choose this device's room when prompted so home commands target the right place.</p>`;
      chat.prepend(div);
      loadWakeHint();
    }
    lastTs = 0;
    startupMessage = null;
    statsMsgs.clear();
    resetTraceGroup();
    pendingArtifacts = [];
    removeTyping();
    waiting = false;
    voiceBusy = false;
    syncButton();
  };

  // Show the Voice-mode instruction in the empty state, with the configured
  // wakeword so the onboarding remains accurate after a settings change.
  const loadWakeHint = async () => {
    const hint = document.getElementById('wake-hint');
    if (!hint) return;
    try {
      const r = await fetch('/config');
      const { wakeword } = await r.json();
      if (!wakeword) return;
      hint.innerHTML = `Click <b>Voice</b>, then say <b>“${escapeHtml(wakeword)}”</b> to use the wakeword.`;
      hint.hidden = false;
    } catch (e) {
      console.warn('config load failed', e);
    }
  };

  const removeTyping = () => {
    typingEl?.remove();
    typingEl = null;
  };

  const showTyping = () => {
    removeTyping();
    typingEl = document.createElement('div');
    typingEl.className = 'typing';
    typingEl.innerHTML = '<span></span><span></span><span></span>';
    chat.appendChild(typingEl);
    scrollEnd();
  };

  const renderTrace = (ev) => {
    const wrap = document.createElement('div');
    const isReplan = ev.kind === 'plan' && ev.replan;
    wrap.className = `trace ${ev.kind}${isReplan ? ' replan' : ''}`;
    if (isReplan) {
      // Make it explicit the agent re-decided: the prior plan was superseded
      // (e.g. actions bundled after a web search are dropped and re-decided
      // from the findings), not a failure of the previous plan.
      wrap.title = 'Agent re-decided from new observations — the previous plan was superseded, not failed';
    }
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = isReplan ? 'replan' : ev.kind;
    wrap.appendChild(label);

    const body = document.createElement('span');
    const p = ev.payload || {};
    if (ev.kind === 'plan') {
      if (Array.isArray(p.actions)) {
        const names = p.actions.map(a => {
          const args = Array.isArray(a.args) ? a.args : [];
          return `${a.intent}(${args.map(JSON.stringify).join(', ')})`;
        });
        body.textContent = names.length
          ? names.join(' → ')
          : '(empty actions — falling through to stall)';
      } else if (typeof p.reply === 'string') {
        const truncated = p.reply.length > 200 ? p.reply.slice(0, 200) + '…' : p.reply;
        body.textContent = `reply: ${truncated}`;
      } else {
        body.textContent = JSON.stringify(p);
      }
    } else if (ev.kind === 'step') {
      const args = Array.isArray(p.args) ? p.args : [];
      body.textContent = `${p.intent}(${args.map(JSON.stringify).join(', ')})`;
    } else if (ev.kind === 'observation') {
      const r = naturalize(p.result || '');
      body.textContent = `${p.intent} → ${r.length > 180 ? r.slice(0, 180) + '…' : r}`;
    } else {
      body.textContent = JSON.stringify(p);
    }
    wrap.appendChild(body);
    return wrap;
  };

  const resetTraceGroup = () => {
    activeTraceGroup = null;
    activeTraceSummary = null;
    activeTraceCount = 0;
  };

  const ensureTraceGroup = () => {
    if (activeTraceGroup) return activeTraceGroup;
    activeTraceGroup = document.createElement('details');
    activeTraceGroup.className = 'trace-group';
    activeTraceSummary = document.createElement('summary');
    activeTraceSummary.textContent = 'trace';
    activeTraceGroup.appendChild(activeTraceSummary);
    if (alwaysDetail) activeTraceGroup.open = true;
    chat.appendChild(activeTraceGroup);
    return activeTraceGroup;
  };

  const updateTraceSummary = () => {
    if (!activeTraceSummary) return;
    activeTraceSummary.textContent = activeTraceCount === 1
      ? 'trace · 1 event'
      : `trace · ${activeTraceCount} events`;
  };

  // Finalise any in-flight typewriter immediately (show the full text).
  const finishTyping = () => { if (finishActive) finishActive(); };

  // Reveal voice responses progressively to roughly track speech (~165 wpm).
  // Text-chat responses are already complete and render immediately.
  const typeInto = (bubble, text) => {
    finishTyping();
    const words = text.trim().split(/\s+/).length || 1;
    const speechMs = (words / 165) * 60000;            // est. spoken duration
    const perChar = Math.min(60, Math.max(12, speechMs / text.length));
    let i = 0;
    const step = () => {
      i = Math.min(text.length, i + 1);
      bubble.textContent = text.slice(0, i);
      scrollEnd();
      if (i < text.length) {
        typingTimer = lifetime.setTimeout(step, perChar);
      } else {
        typingTimer = null;
        finishActive = null;
      }
    };
    finishActive = () => {
      if (typingTimer) { lifetime.clearTimeout(typingTimer); typingTimer = null; }
      bubble.textContent = text;
      finishActive = null;
    };
    step();
  };

  const fmtSecs = (s) => (s == null ? '—' : `${s.toFixed(2)}s`);

  // Build the monospace inference-stats block, adaptively (rows for stages
  // that didn't run this turn are omitted by the backend payload).
  const formatStats = (s) => {
    const L = ['Inference Stats', '-'.repeat(50)];
    L.push(`Total Response Time : ${fmtSecs(s.total)}`);
    if (s.stt)
      L.push(`↳ Audio Input (ASR) : ${fmtSecs(s.stt.seconds)} (${s.stt.model})`);
    if (s.retrieval) {
      const c = s.retrieval.chunks != null ? ` / ${s.retrieval.chunks} chunks` : '';
      L.push(`↳ Context Retrieval : ${fmtSecs(s.retrieval.seconds)} (${s.retrieval.model}${c})`);
    }
    if (s.llm) {
      const m = s.llm;
      L.push(`↳ LLM Generation    : ${fmtSecs(m.seconds)} (${m.model})`);
      L.push(`  ├─ TTFT           : ${fmtSecs(m.ttft)}`);
      L.push(`  ├─ Tokens / sec   : ${m.tps != null ? m.tps.toFixed(1) + ' t/s' : '—'}`);
      L.push(`  ├─ Token Count    : ${m.prompt_tokens} prompt | ${m.output_tokens} output`);
      L.push(`  └─ Agent Loop     : ${m.calls} call${m.calls === 1 ? '' : 's'} | ${m.tools} tool${m.tools === 1 ? '' : 's'}`);
    }
    if (s.tts)
      L.push(`↳ Audio Output(TTS) : ${fmtSecs(s.tts.seconds)} (${s.tts.model})`);
    L.push('-'.repeat(50));
    if (s.vram)
      L.push(`VRAM Usage          : ${s.vram.used.toFixed(1)} GB / ${s.vram.total.toFixed(1)} GB`);
    if (s.ram)
      L.push(`RAM Usage           : ${s.ram.used.toFixed(1)} GB / ${s.ram.total.toFixed(1)} GB`);
    return L.join('\n');
  };

  // Tiny muted button (bottom-right of the meta row) + collapsible detail panel.
  const attachStats = (wrap, ev) => {
    const meta = wrap.querySelector('.meta');
    const btn = document.createElement('button');
    btn.className = 'stats-btn';
    btn.type = 'button';
    const panel = document.createElement('div');
    panel.className = alwaysDetail ? 'stats-panel' : 'stats-panel hidden';
    const entry = { stats: ev.stats, panel, btn };
    entry.refresh = () => {
      btn.textContent = fmtSecs(entry.stats.total);
      panel.textContent = formatStats(entry.stats);
    };
    statsMsgs.set(ev.ts, entry);
    entry.refresh();
    btn.addEventListener('click', () => {
      panel.classList.toggle('hidden');
      scrollEnd();
    });
    meta.appendChild(btn);
    wrap.appendChild(panel);
  };

  const appendMessage = (ev, live = false) => {
    // Stats patch (e.g. TTS time, known only after playback): merge into the
    // already-rendered assistant message keyed by ref_ts. Handled before the
    // dedup guard since it carries its own later ts.
    if (ev.role === 'reset') { resetUI(); return; }
    if (ev.role === 'thinking') {
      onThinking(ev);
      return;
    }
    if (ev.role === 'stopped') {
      // A turn was stopped (dashboard button or voice). Snap any in-flight
      // typewriter to full, drop the typing indicator, and revert the button —
      // no answer bubble follows.
      finishTyping();
      removeTyping();
      waiting = false;
      voiceBusy = false;
      syncButton();
      return;
    }
    if (ev.role === 'stats') {
      const entry = statsMsgs.get(ev.ref_ts);
      if (entry && ev.patch) {
        Object.assign(entry.stats, ev.patch);
        entry.refresh();
      }
      return;
    }
    const isStartup = ev.role === 'assistant' && ev.source === 'startup';
    // The startup greeting is a permanent introduction, not a chronological
    // turn: history and SSE can arrive in either order, so never let its older
    // timestamp suppress it and keep its single bubble first.
    if (isStartup) {
      if (startupMessage) return;
    } else {
      if (ev.ts <= lastTs && lastTs > 0) return; // duplicate (history replay)
      lastTs = ev.ts;
    }
    finishTyping();  // a new event ends any prior turn's animation
    clearEmpty();

    if (ev.role === 'agent') {
      if (ev.kind === 'observation' && ev.payload?.artifact) pendingArtifacts.push(ev.payload.artifact);
      ensureTraceGroup();
      activeTraceGroup.appendChild(renderTrace(ev));
      activeTraceCount++;
      updateTraceSummary();
      scrollEnd();
      return;
    }

    if (ev.role === 'user') { resetTraceGroup(); pendingArtifacts = []; }
    if (ev.role === 'assistant') removeTyping();

    const wrap = document.createElement('div');
    wrap.className = `msg ${ev.role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const text = ev.role === 'assistant'
      ? naturalize(ev.content).replace(/<\|[^|>]+\|>/g, '').replace(/\s{2,}/g, ' ').trim()
      : ev.content;
    const animate = ev.role === 'assistant' && ev.source === 'voice' && live && !!text;
    bubble.textContent = animate ? '' : text;

    const meta = document.createElement('div');
    meta.className = 'meta';
    const tag = document.createElement('span');
    const source = typeof ev.source === 'string' && ev.source ? ev.source : 'system';
    tag.className = `tag ${source}`;
    tag.textContent = source;
    meta.append(tag);
    if (ev.role === 'assistant' && ev.source === 'voice' && ev.tts_backend === 'higgs-gguf') {
      const credit = 'This audio was created with Boson AI\'s Higgs Audio — https://www.boson.ai/higgs-audio';
      const disclosure = document.createElement('span');
      disclosure.className = 'higgs-credit';
      disclosure.textContent = credit;
      meta.append(disclosure);
    }
    // Which satellite/room this turn came from — only worth a second pill
    // when there's something more specific to say than the source tag above
    // already does. A labelled satellite ("kitchen") or a chosen HA room
    // (server-side fallback to SatelliteSession.ha_area_name) is new
    // information; an unlabelled, no-room satellite isn't (voice/text
    // already says as much).
    if (ev.satellite_label) {
      const loc = document.createElement('span');
      loc.className = 'tag location';
      loc.textContent = ev.satellite_label;
      meta.append(loc);
    }
    const time = document.createElement('span');
    time.textContent = fmtTime(ev.ts);
    meta.append(time);

    if (ev.role === 'assistant' && text) {
      const play = document.createElement('button');
      play.className = 'msg-play';
      play.type = 'button';
      play.setAttribute('aria-label', 'Read this response aloud');
      play.title = 'Read aloud';
      const playIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      playIcon.setAttribute('viewBox', '0 0 24 24');
      playIcon.setAttribute('aria-hidden', 'true');
      const playPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      playPath.setAttribute('d', 'M8 5.5v13l10-6.5z');
      playIcon.append(playPath);
      play.append(playIcon);
      play.addEventListener('click', async () => {
        play.disabled = true;
        try {
          if (!await connectSpeaker()) {
            throw new Error('speaker connection failed');
          }
          await fetch('/replay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
          });
        } catch (err) {
          console.warn('replay failed', err);
        } finally {
          play.disabled = false;
        }
      });
      meta.append(play);
    }

    const artifacts = ev.role === 'assistant'
      ? renderArtifacts([...(ev.artifact ? [ev.artifact] : []), ...pendingArtifacts]) : [];
    if (artifacts.length) wrap.classList.add('has-artifacts');
    wrap.append(...artifacts, bubble, meta);
    if (ev.role === 'assistant') pendingArtifacts = [];
    if (isStartup) {
      chat.prepend(wrap);
      startupMessage = wrap;
    } else {
      chat.appendChild(wrap);
    }
    // The submit handler starts the thinking indicator before the SSE user
    // event arrives; move it after that user bubble rather than above it.
    if (ev.role === 'user' && typingEl) chat.appendChild(typingEl);
    if (ev.role === 'assistant' && ev.stats) attachStats(wrap, ev);
    if (animate) typeInto(bubble, text);
    scrollEnd();

    if (ev.role === 'assistant') {
      // Text turns end here; voice turns stay "busy" through TTS (cleared by
      // the /status poll once playback finishes) so the stop button can still
      // cut off speech.
      if (waiting) { waiting = false; input.focus(); }
      if (ev.source === 'text') voiceBusy = false;
      syncButton();
    }
  };

  const scrollEnd = () => {
    requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
  };

  const setLive = (on) => {
    statusDot.classList.toggle('live', !!on);
    statusText.textContent = on ? 'live' : 'reconnecting';
  };

  const autosize = () => {
    input.style.height = 'auto';
    // scrollHeight excludes the textarea's 1px border, but box-sizing:
    // border-box means style.height includes it — without the +2 the box
    // comes up 2px short of the content on every resize (even one line),
    // which was tripping the overflow scrollbar prematurely.
    input.style.height = Math.min(input.scrollHeight + 2, 140) + 'px';
  };

  const loadHistory = async () => {
    try {
      const r = await fetch('/history');
      const items = await r.json();
      if (lifetime.destroyed) return;
      if (!items.length) return;
      clearEmpty();
      for (const ev of items) appendMessage(ev);
    } catch (e) {
      console.warn('history failed', e);
    }
  };

  const startStream = () => {
    if (lifetime.destroyed) return;
    stream?.close();
    const es = stream = new EventSource('/stream');
    es.onopen = () => setLive(true);
    es.onmessage = (e) => {
      if (!e.data) return;
      try { appendMessage(JSON.parse(e.data), true); } catch (err) { console.warn(err); }
    };
    es.onerror = () => {
      setLive(false);
      es.close();
      lifetime.setTimeout(startStream, 2000);
    };
  };

  lifetime.listen(input, 'input', autosize);
  lifetime.listen(input, 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  lifetime.listen(form, 'submit', async (e) => {
    e.preventDefault();
    // While working, the button is a stop button (and Enter stops too).
    if (waiting || voiceBusy) { doStop(); return; }
    const text = input.value.trim();
    if (!text) return;
    waiting = true;
    syncButton();
    input.value = '';
    autosize();
    showTyping();
    try {
      await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
    } catch (err) {
      console.error(err);
      removeTyping();
      waiting = false;
      syncButton();
      input.focus();
    }
  });
  // Reflect the configured preference onto every rendered turn.
  const applyDetailPref = () => {
    document.querySelectorAll('.trace-group').forEach(g => { g.open = alwaysDetail; });
    document.querySelectorAll('.stats-panel').forEach(p =>
      p.classList.toggle('hidden', !alwaysDetail));
  };
  applyDetailPref();

  async function submitCommand(text, button) {
    if (waiting || voiceBusy || lifetime.destroyed) return;
    button.disabled = true;
    waiting = true;
    syncButton();
    showTyping();
    try {
      await fetch('/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
    } catch (err) {
      console.error(err);
      removeTyping();
      waiting = false;
      syncButton();
    } finally { button.disabled = false; }
  }
  lifetime.listen(document, 'visibilitychange', finishTyping);
  const start = async () => {
    syncButton();
    await loadWakeHint();
    if (lifetime.destroyed) return;
    await loadHistory();
    if (!lifetime.destroyed) startStream();
  };
  const destroy = () => {
    lifetime.destroy();
    stream?.close();
    stream = null;
    finishTyping();
    removeTyping();
    cards.destroy();
    statsMsgs.clear();
  };
  const setVoiceBusy = busy => {
    if (busy !== voiceBusy) { voiceBusy = busy; syncButton(); }
  };

  return { start, destroy, setVoiceBusy, clearEmpty, scrollEnd };
}

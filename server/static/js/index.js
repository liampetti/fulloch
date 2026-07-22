(() => {
  // Session cookie auth — the browser sends the cookie automatically on every
  // fetch and WebSocket upgrade. On 401 (session expired after server restart)
  // redirect to /login so the user can re-authenticate.
  fetch('/setup/timezone', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tz: Intl.DateTimeFormat().resolvedOptions().timeZone }) }).catch(() => {});
  const streamUrl = (path) => path;
  const _origFetch = window.fetch.bind(window);
  window.fetch = (url, opts = {}) => _origFetch(url, opts).then((r) => {
    if (r.status === 401) location.href = '/login';
    return r;
  });

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

  // Busy-status banner: another satellite (or the dashboard's own text chat)
  // is mid-turn. Only shown when it's genuinely *another* satellite — this
  // tab's own turn already has its own "thinking/speaking" indicator via
  // voiceBusy, and showing both would be redundant and confusing.
  const busyBanner = document.getElementById('busy-banner');
  const busyBannerText = document.getElementById('busy-banner-text');
  const setSatelliteBusy = (ownerId, ownerLabel) => {
    const showBusy = !!ownerId && ownerId !== mySatelliteId;
    if (busyBanner) busyBanner.hidden = !showBusy;
    if (showBusy && busyBannerText) {
      busyBannerText.textContent = ownerLabel
        ? `Busy — talking to ${ownerLabel}`
        : 'Busy — talking to another room';
    }
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
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  };

  const clearEmpty = () => empty?.remove();

  const resetUI = () => {
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
    statsMsgs.clear();
    resetTraceGroup();
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

  // Reveal `text` into `bubble` progressively to mimic a live stream, paced to
  // roughly track speech (~165 wpm). Used only for live turns, never history
  // replay. A new message finalises any in-flight animation via finishTyping().
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
        typingTimer = setTimeout(step, perChar);
      } else {
        typingTimer = null;
        finishActive = null;
      }
    };
    finishActive = () => {
      if (typingTimer) { clearTimeout(typingTimer); typingTimer = null; }
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
    if (ev.ts <= lastTs && lastTs > 0) return; // duplicate (history replay)
    lastTs = ev.ts;
    finishTyping();  // a new event ends any prior turn's animation
    clearEmpty();

    if (ev.role === 'agent') {
      ensureTraceGroup();
      activeTraceGroup.appendChild(renderTrace(ev));
      activeTraceCount++;
      updateTraceSummary();
      scrollEnd();
      return;
    }

    if (ev.role === 'user') resetTraceGroup();
    if (ev.role === 'assistant') removeTyping();

    const wrap = document.createElement('div');
    wrap.className = `msg ${ev.role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const text = ev.role === 'assistant'
      ? naturalize(ev.content).replace(/<\|[^|>]+\|>/g, '').replace(/\s{2,}/g, ' ').trim()
      : ev.content;
    const animate = ev.role === 'assistant' && live && !!text;
    bubble.textContent = animate ? '' : text;

    const meta = document.createElement('div');
    meta.className = 'meta';
    const tag = document.createElement('span');
    tag.className = `tag ${ev.source}`;
    tag.textContent = ev.source;
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
          if (!await satConnect(true)) {
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

    wrap.append(bubble, meta);
    chat.appendChild(wrap);
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
      if (!items.length) return;
      clearEmpty();
      for (const ev of items) appendMessage(ev);
    } catch (e) {
      console.warn('history failed', e);
    }
  };

  const startStream = () => {
    const es = new EventSource(streamUrl('/stream'));
    es.onopen = () => setLive(true);
    es.onmessage = (e) => {
      if (!e.data) return;
      try { appendMessage(JSON.parse(e.data), true); } catch (err) { console.warn(err); }
    };
    es.onerror = () => {
      setLive(false);
      es.close();
      setTimeout(startStream, 2000);
    };
  };

  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener('submit', async (e) => {
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

  // ---- Tabs + facts viewer ----
  const tabs = document.querySelectorAll('.tab');
  const views = {
    chat: document.getElementById('chat'),
    facts: document.getElementById('facts'),
    entities: document.getElementById('entities'),
    obsidian: document.getElementById('obsidian'),
  };
  const chatFooter = document.getElementById('chat-footer');
  const factsList = document.getElementById('facts-list');
  const factsForm = document.getElementById('facts-form');
  const factInput = document.getElementById('fact-input');
  const addFactBtn = document.getElementById('add-fact');
  const factsPath = document.getElementById('facts-path');

  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  const renderFacts = (facts) => {
    factsList.innerHTML = '';
    if (!facts.length) {
      const empty = document.createElement('div');
      empty.className = 'facts-empty';
      empty.textContent = 'No facts saved yet. Add one above, or say "remember that …" to Fulloch.';
      factsList.appendChild(empty);
      return;
    }
    for (const f of facts) factsList.appendChild(renderFactRow(f));
  };

  const renderFactRow = (f) => {
    const row = document.createElement('div');
    row.className = 'fact-row';
    row.dataset.index = f.index;
    row.innerHTML = `
      <div class="fact-body">
        <div class="fact-date">${escapeHtml(f.date)}</div>
        <div class="fact-text"></div>
      </div>
      <div class="fact-actions">
        <button class="fact-btn edit" type="button" aria-label="Edit" title="Edit">
          <svg viewBox="0 0 24 24"><path d="M3 17.25V21h3.75l11.06-11.06-3.75-3.75L3 17.25zm17.71-10.04a1 1 0 0 0 0-1.41l-2.5-2.5a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>
        </button>
        <button class="fact-btn danger delete" type="button" aria-label="Delete" title="Delete">
          <svg viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
        </button>
      </div>`;
    row.querySelector('.fact-text').textContent = f.text;
    row.querySelector('.edit').addEventListener('click', () => startEdit(row, f));
    row.querySelector('.delete').addEventListener('click', () => deleteFact(f));
    return row;
  };

  const startEdit = (row, f) => {
    const body = row.querySelector('.fact-body');
    body.innerHTML = `
      <div class="fact-date">${escapeHtml(f.date)}</div>
      <textarea class="fact-edit"></textarea>`;
    const ta = body.querySelector('.fact-edit');
    ta.value = f.text;
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);

    const actions = row.querySelector('.fact-actions');
    actions.innerHTML = `
      <button class="fact-btn save" type="button" aria-label="Save" title="Save">
        <svg viewBox="0 0 24 24"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>
      </button>
      <button class="fact-btn cancel" type="button" aria-label="Cancel" title="Cancel">
        <svg viewBox="0 0 24 24"><path d="M19 6.4 17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z"/></svg>
      </button>`;
    actions.querySelector('.save').addEventListener('click', () => saveEdit(f.index, ta.value));
    actions.querySelector('.cancel').addEventListener('click', loadFacts);
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) saveEdit(f.index, ta.value);
      else if (e.key === 'Escape') loadFacts();
    });
  };

  const loadFacts = async () => {
    try {
      const r = await fetch('/facts');
      const data = await r.json();
      renderFacts(data.facts || []);
    } catch (e) {
      console.warn('facts load failed', e);
      factsList.innerHTML = '<div class="facts-empty">Couldn\'t load facts.</div>';
    }
  };

  const saveEdit = async (idx, text) => {
    text = text.trim();
    if (!text) return;
    try {
      const r = await fetch(`/facts/${idx}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      renderFacts(data.facts || []);
    } catch (e) {
      console.error('save failed', e);
      loadFacts();
    }
  };

  const deleteFact = async (f) => {
    if (!confirm(`Forget this fact?\n\n${f.text}`)) return;
    try {
      const r = await fetch(`/facts/${f.index}`, { method: 'DELETE' });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      renderFacts(data.facts || []);
    } catch (e) {
      console.error('delete failed', e);
      loadFacts();
    }
  };

  factsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = factInput.value.trim();
    if (!text) return;
    addFactBtn.disabled = true;
    try {
      const r = await fetch('/facts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      factInput.value = '';
      renderFacts(data.facts || []);
    } catch (e) {
      console.error('add failed', e);
    } finally {
      addFactBtn.disabled = false;
      factInput.focus();
    }
  });

  // ---- Entities viewer ----
  const entitiesList = document.getElementById('entities-list');
  const entitiesHint = document.getElementById('entities-hint');
  const entitySearch = document.getElementById('entity-search');
  let allEntities = [];

  const renderEntities = () => {
    const q = entitySearch.value.trim().toLowerCase();
    const items = q
      ? allEntities.filter((e) =>
          e.name.toLowerCase().includes(q) ||
          e.entity_id.toLowerCase().includes(q))
      : allEntities;
    entitiesList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('div');
      empty.className = 'entities-empty';
      empty.textContent = allEntities.length
        ? 'No entities match your search.'
        : 'No Home Assistant entities found.';
      entitiesList.appendChild(empty);
      return;
    }
    let domain = null;
    for (const e of items) {
      if (e.domain !== domain) {
        domain = e.domain;
        const head = document.createElement('div');
        head.className = 'entity-domain-head';
        head.textContent = domain;
        entitiesList.appendChild(head);
      }
      entitiesList.appendChild(renderEntityRow(e));
    }
  };

  const renderEntityRow = (e) => {
    const row = document.createElement('div');
    row.className = 'entity-row' + (e.denied ? ' denied' : '');
    const body = document.createElement('div');
    body.className = 'entity-body';
    const name = document.createElement('div');
    name.className = 'entity-name';
    name.textContent = e.name;
    const id = document.createElement('div');
    id.className = 'entity-id';
    id.textContent = e.entity_id;
    body.append(name, id);

    const label = document.createElement('label');
    label.className = 'switch';
    label.title = 'Voice control';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !e.denied;  // checked = allowed for voice
    const slider = document.createElement('span');
    slider.className = 'slider';
    label.append(input, slider);
    input.addEventListener('change', () => setEntityDenied(e, !input.checked, input));

    row.append(body, label);
    return row;
  };

  const setEntityDenied = async (e, denied, input) => {
    input.disabled = true;
    try {
      const r = await fetch('/entities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_id: e.entity_id, denied }),
      });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      allEntities = data.entities || [];
      renderEntities();
    } catch (err) {
      console.error('entity toggle failed', err);
      input.checked = !denied;  // revert on failure
    } finally {
      input.disabled = false;
    }
  };

  const loadEntities = async () => {
    try {
      const r = await fetch('/entities');
      const data = await r.json();
      if (!data.available) {
        allEntities = [];
        entitiesList.innerHTML =
          '<div class="entities-empty">Home Assistant isn\'t configured.</div>';
        entitiesHint.hidden = true;
        return;
      }
      entitiesHint.hidden = false;
      allEntities = data.entities || [];
      renderEntities();
    } catch (e) {
      console.warn('entities load failed', e);
      entitiesList.innerHTML =
        '<div class="entities-empty">Couldn\'t load entities.</div>';
    }
  };

  entitySearch.addEventListener('input', renderEntities);

  // ---- Obsidian viewer ----
  const obsPill = document.getElementById('obs-pill');
  const obsStatusDetail = document.getElementById('obs-status-detail');
  const obsError = document.getElementById('obs-error');
  const obsProgress = document.getElementById('obs-progress');
  const obsProgressBar = document.getElementById('obs-progress-bar');
  const obsProgressText = document.getElementById('obs-progress-text');
  const obsActions = document.getElementById('obs-actions');
  const obsidianHint = document.getElementById('obsidian-hint');
  const obsEditAlert = document.getElementById('obs-edit-alert');
  let obsState = null;

  const escapeObs = (s) => escapeHtml(s || '');

  const obsJson = async (url, opts) => {
    const r = await fetch(url, opts);
    const t = await r.text();
    let body = null;
    try { body = t ? JSON.parse(t) : null; } catch (_) { body = t; }
    return { ok: r.ok, status: r.status, body };
  };

  const renderObsidian = (state) => {
    obsState = state || {};
    const err = state.last_error;
    const connected = !!state.connected;
    obsEditAlert.hidden = !(connected && obsState.allow_edit_delete);
    const vaultPath = state.vault_path;
    if (factsPath) factsPath.textContent = `${state.notes_path || './data/notes'}/fulloch_facts.md`;
    const lastConn = state.last_connected_at;
    obsPill.className = 'obs-pill';
    obsError.hidden = true;
    obsProgress.hidden = true;

    if (err) {
      obsPill.textContent = 'Error';
      obsPill.classList.add('error');
      const msg = err === 'not_a_vault' ? "Fulloch says that path isn't a vault (no .obsidian/ folder)."
        : err === 'unreadable' ? "Fulloch can't read that path."
        : err === 'missing' ? 'Fulloch says the vault path is missing.'
        : 'Connection error: ' + escapeObs(err);
      obsError.textContent = msg;
      obsError.hidden = false;
      obsStatusDetail.innerHTML = vaultPath ? `Last vault: <code>${escapeObs(vaultPath)}</code>` : '';
      renderObsActions(connected);
    } else if (connected) {
      obsPill.textContent = 'Connected';
      obsPill.classList.add('connected');
      obsStatusDetail.innerHTML = vaultPath ? `<code>${escapeObs(vaultPath)}</code>` : '';
      if (state.indexing_progress != null) {
        const pct = Math.max(0, Math.min(1, state.indexing_progress));
        obsProgress.hidden = false;
        obsProgressBar.style.width = (pct * 100).toFixed(0) + '%';
        obsProgressText.textContent = `Indexing ${(pct * 100).toFixed(0)}%`;
      }
      renderObsActions(connected);
    } else if (vaultPath) {
      obsPill.textContent = 'Disconnected';
      obsPill.classList.add('disconnected');
      const last = lastConn ? new Date(lastConn * 1000).toLocaleString() : '';
      obsStatusDetail.innerHTML = `Last vault: <code>${escapeObs(vaultPath)}</code>` +
        (last ? ` <span class="obs-help">— last seen ${escapeObs(last)}</span>` : '');
      renderObsActions(connected);
    } else {
      obsPill.textContent = 'Not configured';
      obsPill.classList.add('idle');
      obsStatusDetail.textContent = 'Fulloch hasn\'t connected to an Obsidian vault yet.';
      renderObsActions(connected);
    }
  };

  const renderObsActions = (connected) => {
    obsActions.innerHTML = '';
    const settingsBtn = document.createElement('button');
    settingsBtn.className = 'obs-btn';
    settingsBtn.type = 'button';
    settingsBtn.textContent = 'Configure notes & Obsidian';
    settingsBtn.addEventListener('click', () => { location.href = '/setup'; });
    obsActions.appendChild(settingsBtn);
    if (connected && obsState.allow_edit_delete) {
      const indicator = document.createElement('span');
      indicator.className = 'obs-edit-mode';
      indicator.innerHTML = '<i></i>Edit/delete mode active';
      obsActions.appendChild(indicator);
    }
    obsidianHint.innerHTML = connected
      ? 'The plugin supplies the active note and selected text. With edit/delete enabled in Settings, Fulloch can insert at the cursor, replace selected text, rename, and delete the active note.'
      : 'Fulloch still creates, appends, reads, and searches Markdown notes directly in the notes location above. Connect the plugin in Settings for active-note and selected-text context.';
  };

  const loadObsidian = async () => {
    try {
      const status = await obsJson('/api/obsidian/status');
      if (status.ok) renderObsidian(status.body);
    } catch (e) {
      console.warn('obsidian status load failed', e);
    }
  };

  if (false) { // Legacy setup controls now live exclusively in Settings.
  const ensureObsToken = async () => {
    try {
      const r = await obsJson('/api/obsidian/show-token', { method: 'POST' });
      if (r.ok && r.body && r.body.token) {
        obsTokenValue = r.body.token;
        obsTokenEl.textContent = r.body.token;
        obsTokenRow.hidden = false;
      }
    } catch (e) {
      // Token is a nice-to-have for the panel; failure here shouldn't break the tab.
    }
  };

  const maybePromptMigration = async () => {
    if (localStorage.getItem('obsidian-migrated-shown') === '1') return;
    const dismissed = await obsJson('/api/obsidian/migration-candidate');
    if (!dismissed.ok || !dismissed.body || !dismissed.body.has_legacy_notes) return;
    showMigrationModal(dismissed.body.legacy_count || 0);
  };

  // OS detection for install-command suggestions. Cheap UA sniff — only used
  // to suggest the right shell snippet, so an inaccurate guess is harmless.
  const detectOs = () => {
    const ua = (navigator.userAgent || '').toLowerCase();
    const platform = (navigator.platform || '').toLowerCase();
    if (/win/.test(platform) || /windows/.test(ua)) return 'windows';
    if (/mac/.test(platform) || /mac os/.test(ua)) return 'mac';
    return 'linux';
  };

  const quoteShell = (p) => {
    const os = detectOs();
    if (os === 'windows') return `"\${USERPROFILE}\\Downloads\\fulloch.zip"`;
    return `'${p.replace(/'/g, "'\\''")}'`;
  };

  const buildInstallCommand = (vaultPath) => {
    const os = detectOs();
    const target = `${vaultPath}/.obsidian/plugins/fulloch`.replace(/\/+$/, '');
    if (os === 'windows') {
      return `Expand-Archive -Path "$env:USERPROFILE\\Downloads\\fulloch.zip" -DestinationPath "${target}" -Force`;
    }
    return `unzip -o ~/Downloads/fulloch.zip -d ${quoteShell(target)}`;
  };

  // Pre-submission: zip is the only install path. Once the plugin is accepted
  // into the community store, replace the "Install via zip" section with an
  // `obsidian://show-plugin?id=fulloch` button (one-line change). The
  // `community_store_available` flag is the single switch to flip.
  const obsidianCommunityStoreAvailable = false;

  const openObsidianSetup = async () => {
    await ensureObsToken();
    const token = obsTokenValue || '—';
    const vault = (obsState && obsState.vault_path) || (obsState && obsState.vault_resolved_path) || '';
    const target = vault ? `${vault}/.obsidian/plugins/fulloch`.replace(/\/+$/, '') : '';
    const vaultUri = vault ? `obsidian://open?path=${encodeURIComponent(vault)}` : 'obsidian://';
    const installCmd = vault ? buildInstallCommand(vault) : '';

    const primaryBlock = obsidianCommunityStoreAvailable
      ? `<div class="obs-step">
          <span class="num">1</span>
          <div class="body">
            <p>Open Obsidian and install <strong>Fulloch</strong> from Community plugins:</p>
            <div class="obs-modal-actions">
              <a class="obs-btn" href="obsidian://show-plugin?id=fulloch">Open in Obsidian</a>
            </div>
            <p class="obs-help">Pops Obsidian open on the plugin page. Click <strong>Install</strong>, then <strong>Enable</strong>.</p>
          </div>
        </div>`
      : `<div class="obs-step">
          <span class="num">1</span>
          <div class="body">
            <p>Download the plugin, then move the two files into this folder:</p>
            <div class="obs-target">
              <code id="obs-modal-target">${escapeObs(target || '/path/to/your/vault/.obsidian/plugins/fulloch')}</code>
              <button class="obs-btn ghost" id="obs-modal-copy-target" type="button" ${target ? '' : 'disabled'}>Copy path</button>
            </div>
            <div class="obs-modal-actions">
              <a class="obs-btn" href="/api/obsidian/plugin.zip" download>Download plugin.zip</a>
              <a class="obs-btn ghost" id="obs-modal-open-vault" href="${vaultUri}">Open vault in Obsidian</a>
            </div>
            <details class="obs-terminal">
              <summary>Or paste this in a terminal</summary>
              <pre id="obs-modal-cmd">${escapeObs(installCmd || '# set your vault path first')}</pre>
              <div class="obs-modal-actions"><button class="obs-btn ghost" id="obs-modal-copy-cmd" type="button">Copy command</button></div>
            </details>
            <p class="obs-help">Extract the zip — you'll get <code>manifest.json</code> and <code>main.js</code>. They go <em>directly</em> into the folder above (not into a sub-folder).</p>
          </div>
        </div>`;

    const tokenStepNumber = obsidianCommunityStoreAvailable ? '2' : '2';

    obsOpenModal('Connect Obsidian', `
      ${primaryBlock}
      <div class="obs-step">
        <span class="num">${tokenStepNumber}</span>
        <div class="body">
          <p>In Obsidian: <strong>Settings → Community plugins</strong> → enable <strong>Fulloch</strong>.</p>
        </div>
      </div>
      <div class="obs-step">
        <span class="num">${obsidianCommunityStoreAvailable ? '3' : '3'}</span>
        <div class="body">
          <p>Paste this token in the plugin settings:</p>
          <div class="token-row">
            <pre id="obs-modal-token">${escapeObs(token)}</pre>
            <button class="obs-btn ghost" id="obs-modal-copy" type="button">Copy</button>
          </div>
          <p class="obs-help">The plugin connects automatically once the token matches and your vault is detected.</p>
        </div>
      </div>
    `);
    document.getElementById('obs-modal-copy').addEventListener('click', () => {
      navigator.clipboard.writeText(token).catch(() => {});
    });
    const tgtBtn = document.getElementById('obs-modal-copy-target');
    if (tgtBtn) tgtBtn.addEventListener('click', () => {
      if (target) navigator.clipboard.writeText(target).catch(() => {});
    });
    const cmdBtn = document.getElementById('obs-modal-copy-cmd');
    if (cmdBtn) cmdBtn.addEventListener('click', () => {
      if (installCmd) navigator.clipboard.writeText(installCmd).catch(() => {});
    });
  };

  const showMigrationModal = (legacyCount) => {
    obsOpenModal('Migrate existing notes?', `
      <p>You have <strong>${legacyCount}</strong> note${legacyCount === 1 ? '' : 's'} in Fulloch's default folder
         (<code>./data/notes</code>) that aren't in your Obsidian vault yet.</p>
      <p>Copy them into <code>Inbox/fulloch-import/</code> in your vault? The originals are left in place.</p>
      <div class="obs-modal-actions">
        <button class="obs-btn ghost" data-mig="dismiss" type="button">Don't ask again</button>
        <button class="obs-btn ghost" data-mig="skip" type="button">Skip</button>
        <button class="obs-btn" data-mig="copy" type="button">Copy</button>
      </div>
    `);
    obsModalBody.querySelectorAll('[data-mig]').forEach((b) => {
      b.addEventListener('click', async () => {
        const action = b.getAttribute('data-mig');
        b.disabled = true;
        await obsJson('/api/obsidian/migration-decision', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action }),
        });
        localStorage.setItem('obsidian-migrated-shown', '1');
        obsCloseModal();
        loadObsidian();
      });
    });
  };

  const obsCopyBtn = { addEventListener() {} };
  const obsRegenBtn = { addEventListener() {} };
  const obsSwitchBtn = { addEventListener() {} };
  obsCopyBtn.addEventListener('click', () => {
    if (obsTokenValue) navigator.clipboard.writeText(obsTokenValue).catch(() => {});
  });
  obsRegenBtn.addEventListener('click', async () => {
    if (!confirm('Regenerate the Obsidian auth token?\n\nThe plugin will disconnect within ~10 seconds. You\'ll need to paste the new token into the plugin settings in Obsidian.')) return;
    obsRegenBtn.disabled = true;
    const r = await obsJson('/api/obsidian/regenerate-token', { method: 'POST' });
    obsRegenBtn.disabled = false;
    if (r.ok && r.body && r.body.token) {
      obsTokenValue = r.body.token;
      obsTokenEl.textContent = r.body.token;
    }
  });
  obsSwitchBtn.addEventListener('click', async () => {
    const p = (obsSwitchPath.value || '').trim();
    if (!p) { obsSwitchStatus.textContent = 'Enter a vault path.'; return; }
    obsSwitchBtn.disabled = true;
    obsSwitchStatus.textContent = 'Switching…';
    const r = await obsJson('/api/obsidian/switch-vault', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: p }),
    });
    obsSwitchBtn.disabled = false;
    if (r.ok) {
      obsSwitchStatus.textContent = '✓ switched — voice notes now go to ' + (r.body && r.body.vault_path || p);
      obsSwitchPath.value = '';
      localStorage.removeItem('obsidian-migrated-shown');
      obsMigrationChecked = false;
      loadObsidian(true);
    } else {
      const detail = (r.body && r.body.detail) || 'invalid path';
      obsSwitchStatus.textContent = '✗ ' + detail;
    }
  });

  }

  const setTab = (name) => {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    for (const [k, el] of Object.entries(views)) el.hidden = k !== name;
    chatFooter.hidden = name !== 'chat';
    if (name === 'facts') loadFacts();
    if (name === 'entities') loadEntities();
    if (name === 'obsidian') loadObsidian();
  };

  tabs.forEach((t) => t.addEventListener('click', () => setTab(t.dataset.tab)));

  // Reflect the configured preference onto every rendered turn.
  const applyDetailPref = () => {
    document.querySelectorAll('.trace-group').forEach(g => { g.open = alwaysDetail; });
    document.querySelectorAll('.stats-panel').forEach(p =>
      p.classList.toggle('hidden', !alwaysDetail));
  };
  applyDetailPref();

  // Keep Auto in sync with the browser or OS while this dashboard is open.
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  const darkMql = window.matchMedia('(prefers-color-scheme: dark)');
  const syncThemeMeta = () => {
    if (themeMeta) {
      themeMeta.setAttribute('content',
        getComputedStyle(document.documentElement).backgroundColor);
    }
  };
  syncThemeMeta();
  darkMql.addEventListener('change', (e) => {
    if (window.FULLOCH_DASHBOARD_PREFS.theme === 'auto') {
      document.documentElement.classList.toggle('dark', e.matches);
      syncThemeMeta();
    }
  });

  // Background tabs throttle/pause setTimeout, freezing the typewriter
  // mid-reveal — so a minimised window reopened after TTS already finished
  // would resume slow-crawling a stale answer. Snap any in-flight reveal to
  // full on every visibility change (no-op when nothing is animating).
  document.addEventListener('visibilitychange', () => { finishTyping(); });

  // Poll the agent's state so the stop button appears for voice turns (and any
  // work this page didn't start), and clears once playback finishes. Text turns
  // this page started are tracked locally via `waiting` for snappier feedback.
  const pollStatus = async () => {
    try {
      const r = await fetch('/status');
      if (!r.ok) return;
      const s = await r.json();
      // The dashboard is only usable once the assistant is READY (models loaded
      // + greeting done). If the server is pre-READY — e.g. it restarted under
      // an already-open tab — bounce to '/', which serves the loading screen
      // until READY. Normal load never trips this (index.html is only served
      // when READY).
      if (s.phase && s.phase !== 'READY') { location.href = '/'; return; }
      const busy = s.state !== 'idle';
      if (busy !== voiceBusy) { voiceBusy = busy; syncButton(); }
      applyBranding(!!s.remote_llm);
      setLlmUnreachable(!!s.llm_unreachable);
      setSatelliteBusy(s.active_owner_id, s.active_owner_label);
    } catch (e) { /* transient; next tick retries */ }
  };

  // Remote-LLM mode: when the LLM runs off-device (remote OpenAI endpoint) the
  // assistant isn't fully local — swap the character/favicon (server-side at
  // /logo.png) and update the tagline, but keep the Fulloch name throughout.
  let brandRemote = null;
  const applyBranding = (remote) => {
    if (remote === brandRemote) return;
    brandRemote = remote;
    const sub = document.getElementById('brand-sub');
    const logo = document.getElementById('brand-logo');
    if (remote) {
      sub.textContent = 'language model is off-device';
      if (logo) logo.title = "Fulloch's gone travelling — the language model is running on a remote server, so this isn't fully local.";
    } else {
      sub.textContent = 'fully-local home assistant';
      if (logo) logo.title = '';
    }
  };
  // Remote-LLM-unreachable surface: a full banner the user can minimise to a
  // small warning chip beside the tabs (choice persisted in localStorage) and
  // re-expand by clicking the chip. Hidden entirely while the LLM is reachable.
  const LLM_MIN_KEY = 'fulloch.llmAlertMinimised';
  let llmMinimised = localStorage.getItem(LLM_MIN_KEY) === '1';
  let llmUnreachable = false;
  const renderLlmAlert = () => {
    const banner = document.getElementById('llm-banner');
    const chip = document.getElementById('llm-alert');
    if (banner) banner.hidden = !(llmUnreachable && !llmMinimised);
    if (chip) chip.hidden = !(llmUnreachable && llmMinimised);
  };
  const setLlmUnreachable = (v) => {
    if (v === llmUnreachable) return;
    llmUnreachable = v;
    renderLlmAlert();
  };
  document.getElementById('llm-banner-min')?.addEventListener('click', () => {
    llmMinimised = true;
    localStorage.setItem(LLM_MIN_KEY, '1');
    renderLlmAlert();
  });
  document.getElementById('llm-alert')?.addEventListener('click', () => {
    llmMinimised = false;
    localStorage.removeItem(LLM_MIN_KEY);
    renderLlmAlert();
  });

  // 5s polling: the UI elements driven by /status (mic button state,
  // branding, busy indicator, LLM-unreachable warning) all change on the
  // order of seconds, not milliseconds. 1Hz was bloating the server log
  // and burning battery on mobile devices; 5s is the slowest rate that
  // still feels "live" to a human watching the dashboard. Polling also
  // pauses when the tab is hidden (see `pollStatus` below) so a backgrounded
  // dashboard doesn't keep firing requests.
  const STATUS_POLL_MS = 5000;
  let statusTimer = null;
  const startStatusPolling = () => {
    if (statusTimer !== null) return;
    pollStatus();
    statusTimer = setInterval(pollStatus, STATUS_POLL_MS);
  };
  const stopStatusPolling = () => {
    if (statusTimer === null) return;
    clearInterval(statusTimer);
    statusTimer = null;
  };
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") startStatusPolling();
    else stopStatusPolling();
  });
  if (document.visibilityState === "visible") startStatusPolling();

  // Poll globally so the destructive-edit indicator stays accurate beside the
  // Obsidian tab even while the user is chatting.
  loadObsidian();
  setInterval(loadObsidian, 5000);

  // ---- Browser satellite (push-to-talk via WebSocket) ---------------------
  // satellite-btn:    connect / disconnect the mic+speaker link.
  // conversation-mode-toggle: toggle exclusive full-duplex Conversation mode.
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

  const satBtn = document.getElementById('satellite-btn');
  const conversationModeBtn = document.getElementById('conversation-mode-toggle');
  let satWs = null;
  let satConnecting = false;
  let satPendingConversationMode = null;
  let satAudioCtx = null;
  let satMicStream = null;
  let satWorkletNode = null;
  let satPlayAt = 0;        // AudioContext scheduled-end time for TTS chunks
  let satTtsSr = 24000;     // sample rate announced by server in tts_start
  let satScheduledSources = [];  // AudioBufferSourceNodes pending/playing, so tts_cancel can stop them
  let satPlaybackGeneration = 0;
  let satMicMuted = false;  // true during TTS playback — stops mic data to prevent echo
  let satMicResumeTimer = null;  // setTimeout handle for delayed unmute after satPlayAt
  let satHalfDuplex = true; // false when barge-in or Conversation mode keeps the mic live
  let conversationModeOverride = localStorage.getItem('conversation_mode');
  let conversationMode = conversationModeOverride === '1';
  let satPlaybackOnly = false; // replay speaker connection; no microphone stream
  let mySatelliteId = null; // this tab's own id, from the "session" frame

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

  const satScheduleChunk = (f32) => {
    if (!satAudioCtx) return;
    const buf = satAudioCtx.createBuffer(1, f32.length, satTtsSr);
    buf.getChannelData(0).set(f32);
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
    const generation = satPlaybackGeneration;
    const creditDelay = Math.max(0, (start + buf.duration - satAudioCtx.currentTime - 0.15) * 1000);
    setTimeout(() => {
      if (generation === satPlaybackGeneration && satWs?.readyState === WebSocket.OPEN) {
        satWs.send(JSON.stringify({ type: 'tts_credit', seconds: buf.duration }));
      }
    }, creditDelay);
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
    satPlayAt = satAudioCtx ? satAudioCtx.currentTime : 0;
  };

  const satMuteMicForPlayback = () => {
    if (satMicResumeTimer !== null) {
      clearTimeout(satMicResumeTimer);
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
    satMicResumeTimer = setTimeout(() => {
      satMicResumeTimer = null;
      satMicMuted = false;
    }, delayMs);
  };

  const satDisconnect = () => {
    if (satWorkletNode) { try { satWorkletNode.disconnect(); } catch(_) {} satWorkletNode = null; }
    if (satMicStream) { satMicStream.getTracks().forEach(t => t.stop()); satMicStream = null; }
    if (satWs) { try { satWs.close(); } catch(_) {} satWs = null; }
    if (satAudioCtx) { try { satAudioCtx.close(); } catch(_) {} satAudioCtx = null; }
    satPlayAt = 0;
    satScheduledSources = [];
    satMicMuted = false;
    if (satMicResumeTimer !== null) { clearTimeout(satMicResumeTimer); satMicResumeTimer = null; }
    satHalfDuplex = true;
    satPlaybackOnly = false;
    mySatelliteId = null;
    satPendingConversationMode = null;
    syncSatBtn();
  };

  const satConnect = async (playbackOnly = false) => {
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
          const wait = setInterval(() => {
            if (!satWs || satWs.readyState === WebSocket.CLOSED) {
              clearInterval(wait);
              resolve(false);
            } else if (satWs.readyState === WebSocket.OPEN) {
              clearInterval(wait);
              resolve(true);
            }
          }, 25);
        });
      }
      satDisconnect();
      return false;
    }
    satConnecting = true;
    satPlaybackOnly = playbackOnly;
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
        satMicStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: conversationMode,
            noiseSuppression: conversationMode,
            autoGainControl: false,
          },
          video: false,
        });
      } catch (e) {
        console.error('Satellite: mic access denied', e);
        alert('Microphone access denied — check browser permissions.');
        try { await satAudioCtx.close(); } catch (_) {}
        satAudioCtx = null;
        satConnecting = false;
        return false;
      }
      await resumeAudio;
    } else {
      satAudioCtx = new AudioContext({ latencyHint: 'interactive' });
      // Replay is also user-initiated; unlock playback before opening the socket.
      await satAudioCtx.resume().catch(() => {});
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
    const connected = new Promise(resolve => { resolveConnected = resolve; });

    ws.onopen = () => {
      // Stream resampled mic chunks to server (muted during TTS playback for
      // half-duplex — prevents the assistant hearing its own reply as input).
      if (satWorkletNode) {
        satWorkletNode.port.onmessage = (e) => {
          if (satWs && satWs.readyState === WebSocket.OPEN && !satMicMuted) {
            satWs.send(e.data);
          }
        };
      }
      syncSatBtn();
      if (satPendingConversationMode !== null) {
        ws.send(JSON.stringify({ type: 'conversation_mode.set', enabled: satPendingConversationMode }));
        satPendingConversationMode = null;
      }
      resolveConnected(true);
    };

    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'session') {
            mySatelliteId = msg.satellite_id || null;
            satHalfDuplex = msg.half_duplex !== false;
            conversationMode = !!msg.conversation_mode;
            satSetAudioProcessing(conversationMode);
            syncSatBtn();
          } else if (msg.type === 'conversation_mode.result') {
            if (msg.enabled) {
              conversationMode = true;
              localStorage.setItem('conversation_mode', '1');
              conversationModeOverride = '1';
            } else if (!msg.message) {
              conversationMode = false;
              localStorage.setItem('conversation_mode', '0');
              conversationModeOverride = '0';
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
            satPlayAt = satAudioCtx ? Math.max(satPlayAt, satAudioCtx.currentTime) : 0;
            satPlaybackGeneration += 1;
            satWs?.send(JSON.stringify({ type: 'tts_credit', seconds: 0.5 }));
            satMuteMicForPlayback();
          } else if (msg.type === 'tts_end' || msg.type === 'tts_cancel') {
            if (msg.type === 'tts_cancel') satCancelPlayback();
            satUnmuteMicAfterPlayback();
          }
        } catch(_) {}
      } else {
        // Binary Float32 PCM audio chunk from TTS
        satScheduleChunk(new Float32Array(e.data));
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
    try {
      return await connected;
    } finally {
      satConnecting = false;
    }
  };

  const toggleConversationMode = async () => {
    const enabled = !conversationMode;
    if (satWs && satWs.readyState === WebSocket.OPEN) {
      satWs.send(JSON.stringify({ type: 'conversation_mode.set', enabled }));
      return;
    }
    conversationMode = enabled;
    localStorage.setItem('conversation_mode', conversationMode ? '1' : '0');
    conversationModeOverride = conversationMode ? '1' : '0';
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
    if (satWs) { satDisconnect(); satConnect(); }
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
    if (areas.length) renderAreaPicker(areas);
  };

  satAreaPill.addEventListener('click', async () => {
    const areas = await fetchHaAreas();
    if (areas.length) renderAreaPicker(areas);
  });

  satBtn.addEventListener('click', () => { satConnect(); });
  conversationModeBtn.addEventListener('click', toggleConversationMode);

  syncSatBtn();

  syncButton();
  syncSatAreaPill();
  // Show the initial Voice/Conversation guidance before the startup greeting
  // replaces the empty state. On a later reset, resetUI restores it instead.
  loadWakeHint().finally(() => loadHistory()).then(() => {
    startStream();
    maybeShowAreaPicker();
  });
  pollStatus();
})();

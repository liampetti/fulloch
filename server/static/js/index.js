(() => {
  // Session cookie auth — the browser sends the cookie automatically on every
  // fetch and WebSocket upgrade. On 401 (session expired after server restart)
  // redirect to /login so the user can re-authenticate.
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

  // Busy-status banner: another satellite (or the dashboard's own text chat)
  // is mid-turn. Only shown when it's genuinely *another* satellite — this
  // tab's own turn already has its own "thinking/speaking" indicator via
  // voiceBusy, and showing both would be redundant and confusing.
  const busyBanner = document.getElementById('busy-banner');
  const busyBannerText = document.getElementById('busy-banner-text');
  const thinkingJobCard = document.getElementById('thinking-job-card');
  const thinkingJobTask = document.getElementById('thinking-job-task');
  const thinkingJobStage = document.getElementById('thinking-job-stage');
  const thinkingJobCancel = document.getElementById('thinking-job-cancel');
  let thinkingJob = null;
  const renderThinkingJob = (job) => {
    thinkingJob = job || null;
    if (!thinkingJobCard) return;
    thinkingJobCard.hidden = !thinkingJob;
    if (!thinkingJob) return;
    if (thinkingJobTask) thinkingJobTask.textContent = thinkingJob.task || 'Deliberate work';
    if (thinkingJobStage) thinkingJobStage.textContent = thinkingJob.stage || thinkingJob.status || 'Working';
    if (thinkingJobCancel) thinkingJobCancel.hidden = ['READY', 'FAILED', 'CANCELLED'].includes(thinkingJob.status);
  };
  thinkingJobCancel?.addEventListener('click', async () => {
    if (!thinkingJob?.id) return;
    thinkingJobCancel.disabled = true;
    try {
      await fetch(`/thinking/${encodeURIComponent(thinkingJob.id)}/cancel`, { method: 'POST' });
      renderThinkingJob(null);
    } finally {
      thinkingJobCancel.disabled = false;
    }
  });
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

  const weatherIcon = (condition, compact = false) => {
    const normalized = String(condition || '').toLowerCase();
    const cls = compact ? 'weather-icon compact' : 'weather-icon';
    if (normalized.includes('partly')) {
      return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><g class="weather-sun"><circle cx="17" cy="17" r="6"/><path d="M17 5v3m0 18v3M5 17h3m18 0h3M8.5 8.5l2.2 2.2m12.6 12.6 2.2 2.2m0-17-2.2 2.2"/></g><path class="weather-cloud" d="M13 34h22a7 7 0 0 0 .5-14A10.5 10.5 0 0 0 16 18a8 8 0 0 0-3 16Z"/></svg>`;
    }
    if (normalized.includes('fog')) {
      return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><path class="weather-cloud" d="M13 28h22a7 7 0 0 0 0-14 11 11 0 0 0-21-1A7.5 7.5 0 0 0 13 28Z"/><path d="M10 33h28m-24 5h20"/></svg>`;
    }
    if (normalized.includes('snow') || normalized.includes('hail')) {
      return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><path class="weather-cloud" d="M13 28h22a7 7 0 0 0 0-14 11 11 0 0 0-21-1A7.5 7.5 0 0 0 13 28Z"/><path d="m17 34 4 4m0-4-4 4m10-4 4 4m0-4-4 4"/></svg>`;
    }
    if (normalized.includes('storm') || normalized.includes('thunder')) {
      return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><path class="weather-cloud" d="M13 28h22a7 7 0 0 0 0-14 11 11 0 0 0-21-1A7.5 7.5 0 0 0 13 28Z"/><path class="weather-bolt" d="m26 29-6 9h5l-2 6 7-10h-5l3-5Z"/></svg>`;
    }
    if (normalized.includes('rain') || normalized.includes('shower') || normalized.includes('sleet')) {
      return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><path class="weather-cloud" d="M13 28h22a7 7 0 0 0 0-14 11 11 0 0 0-21-1A7.5 7.5 0 0 0 13 28Z"/><path d="M18 34v5m6-5v5m6-5v5"/></svg>`;
    }
    if (normalized.includes('cloud')) {
      return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><path class="weather-cloud" d="M11 31h25a8 8 0 0 0 .5-16A11.5 11.5 0 0 0 14 17a8 8 0 0 0-3 14Z"/></svg>`;
    }
    return `<svg class="${cls}" viewBox="0 0 48 48" aria-hidden="true"><g class="weather-sun"><circle cx="24" cy="24" r="8"/><path d="M24 5v6m0 26v6M5 24h6m26 0h6m-32.5-13.5 4.2 4.2m18.6 18.6 4.2 4.2m0-28-4.2 4.2M14.7 33.3l-4.2 4.2"/></g></svg>`;
  };

  const renderWeatherCard = (data) => {
    if (!data || !Array.isArray(data.forecast) || !data.current) return null;
    const card = document.createElement('section');
    card.className = 'artifact weather-card';
    const temperature = Number.isFinite(data.current.temperature) ? `${data.current.temperature}°` : '';
    card.innerHTML = `<div class="weather-card-head"><div class="weather-title">${weatherIcon(data.current.condition)}<span>${escapeHtml(data.title || 'Weather')}</span></div><strong>${escapeHtml(temperature)}</strong></div><div class="weather-current">${escapeHtml(data.current.condition || 'Unknown')}</div>`;
    const days = document.createElement('div');
    days.className = 'weather-days';
    for (const day of data.forecast) {
      const item = document.createElement('div');
      item.className = 'weather-day';
      const range = Number.isFinite(day.low) && Number.isFinite(day.high) ? `${day.low}° / ${day.high}°` : Number.isFinite(day.high) ? `${day.high}°` : '-';
      const rain = Number.isFinite(day.precipitation_probability) ? `${day.precipitation_probability}% rain` : '';
      item.innerHTML = `<div class="weather-day-top">${weatherIcon(day.condition, true)}<strong>${escapeHtml(day.label || '')}</strong></div><span>${escapeHtml(day.condition || 'Unknown')}</span><b>${escapeHtml(range)}</b>${rain ? `<small>${escapeHtml(rain)}</small>` : ''}`;
      days.append(item);
    }
    card.append(days);
    return card;
  };

  const entityIcon = (domain, state) => {
    const cls = 'entity-status-icon';
    if (domain === 'light') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18h6m-5 3h4M8.2 14.2A7 7 0 1 1 15.8 14.2c-.9.8-1.5 1.8-1.6 2.8H9.8c-.1-1-.7-2-1.6-2.8Z"/></svg>`;
    if (domain === 'lock') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 4v2"/></svg>`;
    if (domain === 'cover') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h14v16H5zM5 9h14M5 14h14M5 19h14"/></svg>`;
    if (domain === 'climate') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v18m-5-15 10 12m0-12L7 18M3 12h18"/></svg>`;
    if (domain === 'fan') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="2"/><path d="M11 10C6 10 5 5 8 4c3-1 5 2 4 6m2 3c0 5 5 6 6 3 1-3-2-5-6-4m-3 2c-3 4 0 8 3 6 3-2 1-5-3-6"/></svg>`;
    if (domain === 'media_player') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="5" width="16" height="14" rx="2"/><path d="m10 9 5 3-5 3z"/></svg>`;
    if (domain === 'switch' || domain === 'vacuum') return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4v6m8-6v6M7 10h10v5a4 4 0 0 1-8 0v-5m3 9h.01"/></svg>`;
    return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/></svg>`;
  };

  const appendEntityStatus = (parent, entity) => {
    if (!entity || typeof entity.title !== 'string') return;
    const row = document.createElement('div');
    row.className = 'entity-status-row';
    const heading = document.createElement('div');
    heading.className = 'entity-status-heading';
    heading.innerHTML = entityIcon(entity.domain, entity.state);
    const name = document.createElement('strong');
    name.textContent = entity.title;
    const state = document.createElement('span');
    state.className = `entity-state ${String(entity.state || '').toLowerCase()}`;
    state.textContent = entity.state || 'Unknown';
    heading.append(name, state);
    row.append(heading);
    if (Array.isArray(entity.details) && entity.details.length) {
      const details = document.createElement('div');
      details.className = 'entity-status-details';
      for (const detail of entity.details.slice(0, 8)) {
        if (typeof detail?.label !== 'string' || typeof detail?.value !== 'string') continue;
        const item = document.createElement('span');
        item.textContent = `${detail.label}: ${detail.value}`;
        if (detail.label === 'Colour' && /^#[0-9a-f]{6}$/i.test(detail.value)) {
          item.style.setProperty('--entity-colour', detail.value);
          item.classList.add('has-colour');
        }
        details.append(item);
      }
      row.append(details);
    }
    parent.append(row);
  };

  const renderEntityStatusCard = (data) => {
    if (!data || data.type !== 'entity_status') return null;
    const entities = Array.isArray(data.entities) ? data.entities : [data];
    if (!entities.length || entities.length > 12) return null;
    const card = document.createElement('section');
    card.className = 'artifact entity-status-card';
    const title = document.createElement('div');
    title.className = 'entity-status-card-title';
    title.textContent = data.entities ? `${data.title || 'Area'} status` : 'Entity status';
    card.append(title);
    for (const entity of entities) appendEntityStatus(card, entity);
    return card;
  };

  const renderTemperatureHistoryCard = (data) => {
    const points = Array.isArray(data?.points) ? data.points.filter(point => Number.isFinite(point?.value)).slice(0, 40) : [];
    if (!points.length || !Number.isFinite(data?.current) || !Number.isFinite(data?.min) || !Number.isFinite(data?.max)) return null;
    const card = document.createElement('section');
    card.className = 'artifact temperature-history-card';
    const heading = document.createElement('div');
    heading.className = 'temperature-history-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 5a2 2 0 1 1 4 0v8.2a4 4 0 1 1-4 0Z"/><path d="M12 8v8"/></svg>';
    const title = document.createElement('span');
    title.textContent = data.title || 'Temperature history';
    const current = document.createElement('strong');
    current.textContent = `${data.current}${data.unit || '°'}`;
    heading.append(title, current);
    card.append(heading);

    const chart = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    chart.classList.add('temperature-history-chart');
    chart.setAttribute('viewBox', '0 0 300 86');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', `Temperature ranged from ${data.min}${data.unit || '°'} to ${data.max}${data.unit || '°'}`);
    for (const y of [18, 44, 70]) {
      const guide = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      guide.setAttribute('x1', '8');
      guide.setAttribute('x2', '292');
      guide.setAttribute('y1', String(y));
      guide.setAttribute('y2', String(y));
      guide.classList.add('temperature-history-guide');
      chart.append(guide);
    }
    const range = data.max - data.min || 1;
    const coordinates = points.map((point, index) => {
      const x = points.length === 1 ? 150 : index / (points.length - 1) * 284 + 8;
      const y = 70 - (point.value - data.min) / range * 52;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', coordinates);
    chart.append(line);
    const last = coordinates.split(' ').at(-1).split(',');
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', last[0]);
    dot.setAttribute('cy', last[1]);
    dot.setAttribute('r', '3.5');
    chart.append(dot);
    card.append(chart);
    const stats = document.createElement('div');
    stats.className = 'temperature-history-stats';
    for (const [label, value] of [['Low', data.min], ['High', data.max], ['Target', data.target]]) {
      if (!Number.isFinite(value)) continue;
      const item = document.createElement('span');
      item.textContent = `${label} ${value}${data.unit || '°'}`;
      stats.append(item);
    }
    card.append(stats);
    return card;
  };

  const renderLightHistoryCard = (data) => {
    const points = Array.isArray(data?.points) ? data.points.filter(point => typeof point?.on === 'boolean').slice(0, 40) : [];
    if (!points.length) return null;
    const card = document.createElement('section');
    card.className = 'artifact light-history-card';
    const heading = document.createElement('div');
    heading.className = 'light-history-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18h6m-5 3h4M8.2 14.2A7 7 0 1 1 15.8 14.2c-.9.8-1.5 1.8-1.6 2.8H9.8c-.1-1-.7-2-1.6-2.8Z"/></svg>';
    const title = document.createElement('span');
    title.textContent = data.title || 'Light history';
    const state = document.createElement('strong');
    state.className = data.state === 'on' ? 'on' : '';
    state.textContent = data.state || 'Unknown';
    heading.append(title, state);
    card.append(heading);
    const chart = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    chart.classList.add('light-history-chart');
    chart.setAttribute('viewBox', '0 0 300 76');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', `${data.title || 'Light'} state and brightness history`);
    const x = (index) => (points.length === 1 ? 150 : index / (points.length - 1) * 284 + 8);
    const stateCoordinates = points.map((point, index) => `${x(index).toFixed(1)},${point.on ? 21 : 62}`).join(' ');
    const stateLine = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    stateLine.classList.add('light-history-state-line');
    stateLine.setAttribute('points', stateCoordinates);
    chart.append(stateLine);
    const brightnessCoordinates = points.filter(point => Number.isFinite(point.brightness)).map((point, index) => {
      const sourceIndex = points.indexOf(point);
      return `${x(sourceIndex).toFixed(1)},${(62 - point.brightness * .41).toFixed(1)}`;
    }).join(' ');
    if (brightnessCoordinates) {
      const brightnessLine = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      brightnessLine.classList.add('light-history-brightness-line');
      brightnessLine.setAttribute('points', brightnessCoordinates);
      chart.append(brightnessLine);
    }
    card.append(chart);
    const legend = document.createElement('div');
    legend.className = 'light-history-legend';
    const status = document.createElement('span');
    status.textContent = 'On/off';
    legend.append(status);
    if (Number.isFinite(data.brightness)) {
      const brightness = document.createElement('span');
      brightness.textContent = `Brightness ${data.brightness}%`;
      legend.append(brightness);
    }
    card.append(legend);
    return card;
  };

  const mediaControlIcon = (action) => {
    if (action === 'previous') return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 6v12m10-12-8 6 8 6z"/></svg>';
    if (action === 'skip') return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7 6 8 6-8 6zm10 0v12"/></svg>';
    if (action === 'resume') return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 10 7-10 7z"/></svg>';
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6v12m8-12v12"/></svg>';
  };

  const renderMediaCard = (data) => {
    if (!data || data.type !== 'media' || typeof data.title !== 'string' || typeof data.player !== 'string') return null;
    const card = document.createElement('section');
    card.className = 'artifact media-card';
    const artwork = document.createElement('div');
    artwork.className = 'media-artwork';
    artwork.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18V7l10-2v11M9 18a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm10-2a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"/></svg>';
    const fallback = artwork.firstChild;
    if (typeof data.artwork_url === 'string' && /^\/media-artwork\/media_player\.[A-Za-z0-9_]+$/.test(data.artwork_url)) {
      const image = document.createElement('img');
      image.src = data.artwork_url;
      image.alt = '';
      image.addEventListener('error', () => image.remove());
      image.addEventListener('load', () => fallback.remove());
      artwork.append(image);
    }
    const copy = document.createElement('div');
    copy.className = 'media-copy';
    const title = document.createElement('strong');
    title.textContent = data.title;
    const artist = document.createElement('span');
    artist.textContent = data.artist || data.player;
    const player = document.createElement('small');
    player.textContent = data.player;
    copy.append(title, artist, player);
    card.append(artwork, copy);
    if (Number.isFinite(data.volume)) {
      const volume = document.createElement('div');
      volume.className = 'media-volume';
      volume.setAttribute('aria-label', `Volume ${data.volume}%`);
      const level = document.createElement('span');
      level.style.width = `${Math.max(0, Math.min(100, data.volume))}%`;
      volume.append(level);
      card.append(volume);
    }
    const controls = document.createElement('div');
    controls.className = 'media-controls';
    const primary = data.state === 'playing' ? 'pause' : 'resume';
    for (const action of ['previous', primary, 'skip']) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = action === primary ? 'primary' : '';
      button.setAttribute('aria-label', `${action} ${data.player}`);
      button.title = action;
      button.innerHTML = mediaControlIcon(action);
      button.addEventListener('click', async () => {
        if (waiting || voiceBusy) return;
        button.disabled = true;
        waiting = true;
        syncButton();
        showTyping();
        try {
          await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: `${action} ${data.player}` }),
          });
        } catch (err) {
          console.error(err);
          removeTyping();
          waiting = false;
          syncButton();
        } finally {
          button.disabled = false;
        }
      });
      controls.append(button);
    }
    card.append(controls);
    return card;
  };

  const renderCalendarCard = (data) => {
    if (!data || data.type !== 'calendar' || !Array.isArray(data.events) || !data.events.length || data.events.length > 12) return null;
    const card = document.createElement('section');
    card.className = 'artifact calendar-card';
    const heading = document.createElement('div');
    heading.className = 'calendar-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4m8-4v4M4 10h16"/></svg>';
    const title = document.createElement('strong');
    title.textContent = data.title || 'Agenda';
    const count = document.createElement('span');
    count.textContent = `${data.events.length} event${data.events.length === 1 ? '' : 's'}`;
    heading.append(title, count);
    card.append(heading);
    const timeline = document.createElement('div');
    timeline.className = 'calendar-timeline';
    for (const event of data.events) {
      if (typeof event?.title !== 'string' || typeof event?.when !== 'string') continue;
      const row = document.createElement('div');
      row.className = event.all_day ? 'calendar-event all-day' : 'calendar-event';
      const when = document.createElement('span');
      when.textContent = event.when;
      const eventTitle = document.createElement('strong');
      eventTitle.textContent = event.title;
      row.append(when, eventTitle);
      timeline.append(row);
    }
    card.append(timeline);
    return card;
  };

  const timerRemaining = (endsAt) => Math.max(0, Math.round(endsAt - Date.now() / 1000));
  const timerLabel = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor(seconds % 3600 / 60);
    const secs = seconds % 60;
    return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}` : `${minutes}:${String(secs).padStart(2, '0')}`;
  };

  const renderTimersCard = (data) => {
    const timers = Array.isArray(data?.timers) ? data.timers.filter(timer => typeof timer?.id === 'string' && Number.isFinite(timer.ends_at) && Number.isFinite(timer.duration)).slice(0, 32) : [];
    if (!timers.length) return null;
    const card = document.createElement('section');
    card.className = 'artifact timers-card';
    const heading = document.createElement('div');
    heading.className = 'timers-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="13" r="7"/><path d="M12 13V9m0 4 3 2M9 3h6m-3 0v3"/></svg>';
    const title = document.createElement('strong');
    title.textContent = 'Active timers';
    heading.append(title);
    card.append(heading);
    for (const timer of timers) {
      const row = document.createElement('div');
      row.className = 'timer-row';
      const ring = document.createElement('div');
      ring.className = 'timer-ring';
      const countdown = document.createElement('span');
      ring.append(countdown);
      const copy = document.createElement('div');
      copy.className = 'timer-copy';
      const label = document.createElement('strong');
      label.textContent = typeof timer.label === 'string' ? timer.label : 'Timer';
      const detail = document.createElement('span');
      copy.append(label, detail);
      const controls = document.createElement('div');
      controls.className = 'timer-controls';
      for (const [action, text] of [['extend', '+1m'], ['cancel', 'Cancel']]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = text;
        button.addEventListener('click', async () => {
          if (waiting || voiceBusy) return;
          button.disabled = true;
          waiting = true;
          syncButton();
          showTyping();
          const request = action === 'extend' ? `extend timer ${timer.id} by 1 minute` : `cancel timer ${timer.id}`;
          try {
            await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: request }) });
          } catch (err) {
            console.error(err);
            removeTyping();
            waiting = false;
            syncButton();
          } finally {
            button.disabled = false;
          }
        });
        controls.append(button);
      }
      let interval;
      const update = () => {
        const remaining = timerRemaining(timer.ends_at);
        const progress = Math.max(0, Math.min(100, remaining / Math.max(1, timer.duration) * 100));
        countdown.textContent = timerLabel(remaining);
        ring.style.setProperty('--timer-progress', `${progress}%`);
        detail.textContent = remaining ? `${Math.ceil(remaining / 60)} min remaining` : 'Finished';
        if (!remaining && interval) window.clearInterval(interval);
      };
      update();
      interval = window.setInterval(update, 1000);
      row.append(ring, copy, controls);
      card.append(row);
    }
    return card;
  };

  const overviewIcon = (kind) => {
    if (kind === 'lights') return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18h6m-5 3h4M8.2 14.2A7 7 0 1 1 15.8 14.2c-.9.8-1.5 1.8-1.6 2.8H9.8c-.1-1-.7-2-1.6-2.8Z"/></svg>';
    if (kind === 'openings') return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h10v16H5zM15 6h4v14h-4M10 12h.01"/></svg>';
    if (kind === 'locks') return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>';
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v18m-5-15 10 12m0-12L7 18M3 12h18"/></svg>';
  };

  const renderHomeOverviewCard = (data) => {
    if (!data || data.type !== 'home_overview' || !Array.isArray(data.groups) || data.groups.length > 4) return null;
    const card = document.createElement('section');
    card.className = 'artifact home-overview-card';
    const heading = document.createElement('div');
    heading.className = 'home-overview-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 11 9-7 9 7v9H3zM9 20v-6h6v6"/></svg>';
    const title = document.createElement('strong');
    title.textContent = 'Home overview';
    heading.append(title);
    card.append(heading);
    if (!data.groups.length) {
      const settled = document.createElement('div');
      settled.className = 'home-settled';
      settled.textContent = 'Everything looks settled';
      card.append(settled);
      return card;
    }
    const groups = document.createElement('div');
    groups.className = 'home-overview-groups';
    for (const group of data.groups) {
      if (typeof group?.label !== 'string' || typeof group?.kind !== 'string' || !Number.isFinite(group?.count) || !Array.isArray(group.entities)) continue;
      const item = document.createElement('div');
      item.className = `home-overview-group ${group.kind}`;
      item.innerHTML = overviewIcon(group.kind);
      const count = document.createElement('strong');
      count.textContent = group.count;
      const label = document.createElement('span');
      label.textContent = group.label;
      const names = document.createElement('small');
      names.textContent = group.entities.filter(name => typeof name === 'string').slice(0, 8).join(', ');
      item.append(count, label, names);
      groups.append(item);
    }
    card.append(groups);
    return card;
  };

  const energyIcon = (kind) => {
    if (kind === 'solar') return '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3"/><path d="M12 1v2m0 10v2M5 8h2m10 0h2M7.1 3.1l1.4 1.4m7 7 1.4 1.4m0-9.8-1.4 1.4m-7 7-1.4 1.4M5 20h14l-2-7H7zM9 16h6"/></svg>';
    if (kind === 'battery') return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="7" width="15" height="10" rx="2"/><path d="M21 10v4M8 10v4h5"/></svg>';
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m13 2-7 11h5l-1 9 8-12h-5z"/></svg>';
  };

  const renderEnergyCard = (data) => {
    const metrics = Array.isArray(data?.metrics) ? data.metrics.filter(metric => typeof metric?.kind === 'string' && typeof metric?.label === 'string' && Number.isFinite(metric?.value)).slice(0, 3) : [];
    if (!metrics.length) return null;
    const card = document.createElement('section');
    card.className = 'artifact energy-card';
    const heading = document.createElement('div');
    heading.className = 'energy-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m13 2-7 11h5l-1 9 8-12h-5z"/></svg>';
    const title = document.createElement('strong');
    title.textContent = 'Energy';
    heading.append(title);
    card.append(heading);
    const readings = document.createElement('div');
    readings.className = 'energy-readings';
    for (const metric of metrics) {
      const item = document.createElement('div');
      item.className = `energy-reading ${metric.kind}`;
      item.innerHTML = energyIcon(metric.kind);
      const value = document.createElement('strong');
      value.textContent = `${metric.value}${metric.unit ? ` ${metric.unit}` : ''}`;
      const label = document.createElement('span');
      label.textContent = metric.label;
      item.append(value, label);
      readings.append(item);
    }
    card.append(readings);
    const points = Array.isArray(data.history) ? data.history.filter(point => Number.isFinite(point?.value)).slice(0, 36) : [];
    if (points.length > 1) {
      const low = Math.min(...points.map(point => point.value));
      const high = Math.max(...points.map(point => point.value));
      const range = high - low || 1;
      const chart = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      chart.classList.add('energy-chart');
      chart.setAttribute('viewBox', '0 0 300 48');
      chart.setAttribute('role', 'img');
      chart.setAttribute('aria-label', `Consumption ranged from ${low} to ${high} ${data.history_unit || ''}`.trim());
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      line.setAttribute('points', points.map((point, index) => `${(index / (points.length - 1) * 284 + 8).toFixed(1)},${(40 - (point.value - low) / range * 32).toFixed(1)}`).join(' '));
      chart.append(line);
      card.append(chart);
    }
    return card;
  };

  const securityIcon = (kind) => {
    if (kind === 'openings') return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h10v16H5zM15 6h4v14h-4M10 12h.01"/></svg>';
    if (kind === 'locks') return '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>';
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9h16v10H4zM8 9l2-4h4l2 4M9 14h.01"/></svg>';
  };

  const renderSecurityCard = (data) => {
    if (!data || data.type !== 'security' || !['secure', 'attention'].includes(data.status) || !Array.isArray(data.groups) || data.groups.length > 4) return null;
    const card = document.createElement('section');
    card.className = `artifact security-card ${data.status}`;
    const heading = document.createElement('div');
    heading.className = 'security-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 4 6v5c0 5 3.4 8.7 8 10 4.6-1.3 8-5 8-10V6zM8.5 12l2.2 2.2 4.8-4.8"/></svg>';
    const title = document.createElement('strong');
    title.textContent = data.status === 'secure' ? 'Home secure' : 'Security attention';
    heading.append(title);
    card.append(heading);
    if (!data.groups.length) {
      const detail = document.createElement('p');
      detail.textContent = 'No open entries or unlocked locks';
      card.append(detail);
      return card;
    }
    const groups = document.createElement('div');
    groups.className = 'security-groups';
    for (const group of data.groups) {
      if (typeof group?.kind !== 'string' || typeof group?.label !== 'string' || !Number.isFinite(group?.count) || !Array.isArray(group.entities)) continue;
      const item = document.createElement('div');
      item.className = `security-group ${group.kind}`;
      item.innerHTML = securityIcon(group.kind);
      const copy = document.createElement('div');
      const label = document.createElement('strong');
      label.textContent = `${group.count} ${group.label}`;
      const names = document.createElement('span');
      names.textContent = group.entities.filter(name => typeof name === 'string').slice(0, 8).join(', ');
      copy.append(label, names);
      item.append(copy);
      groups.append(item);
    }
    card.append(groups);
    return card;
  };

  const renderNoteCard = (data) => {
    if (!data || data.type !== 'note' || typeof data.title !== 'string' || typeof data.excerpt !== 'string') return null;
    const card = document.createElement('section');
    card.className = 'artifact note-card';
    const heading = document.createElement('div');
    heading.className = 'note-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h11l3 3v15H5zM16 3v4h4M8 12h8m-8 4h8"/></svg>';
    const title = document.createElement('strong');
    title.textContent = data.title;
    heading.append(title);
    const excerpt = document.createElement('p');
    excerpt.textContent = data.excerpt;
    card.append(heading, excerpt);
    if (data.truncated) {
      const more = document.createElement('small');
      more.textContent = 'Note continues';
      card.append(more);
    }
    return card;
  };

  const renderNotesSearchCard = (data) => {
    if (!data || data.type !== 'notes_search' || typeof data.query !== 'string' || !Array.isArray(data.matches) || !data.matches.length || data.matches.length > 5) return null;
    const card = document.createElement('section');
    card.className = 'artifact notes-search-card';
    const heading = document.createElement('div');
    heading.className = 'note-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="m15 15 5 5"/></svg>';
    const title = document.createElement('strong');
    title.textContent = `Notes about ${data.query}`;
    heading.append(title);
    card.append(heading);
    for (const match of data.matches) {
      if (typeof match?.title !== 'string' || typeof match?.excerpt !== 'string') continue;
      const row = document.createElement('div');
      row.className = 'notes-search-match';
      const matchTitle = document.createElement('strong');
      matchTitle.textContent = match.title;
      const excerpt = document.createElement('span');
      excerpt.textContent = match.excerpt;
      row.append(matchTitle, excerpt);
      card.append(row);
    }
    return card;
  };

  const renderTodosCard = (data) => {
    if (!data || data.type !== 'todos' || !Array.isArray(data.items) || !data.items.length || data.items.length > 20) return null;
    const card = document.createElement('section');
    card.className = 'artifact todos-card';
    const heading = document.createElement('div');
    heading.className = 'todos-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="4" width="14" height="17" rx="2"/><path d="m8 10 1.5 1.5L12 8.5m1 2h3m-8 5 1.5 1.5L12 14.5m1 2h3"/></svg>';
    const title = document.createElement('strong');
    title.textContent = 'To do';
    heading.append(title);
    card.append(heading);
    for (const item of data.items) {
      if (typeof item !== 'string') continue;
      const row = document.createElement('div');
      row.className = 'todo-item';
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-label', `Complete ${item}`);
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="m8.5 12 2.3 2.3 4.7-4.7"/></svg>';
      const label = document.createElement('span');
      label.textContent = item;
      button.addEventListener('click', async () => {
        if (waiting || voiceBusy) return;
        button.disabled = true;
        waiting = true;
        syncButton();
        showTyping();
        try {
          await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: `complete ${item}` }) });
        } catch (err) {
          console.error(err);
          removeTyping();
          waiting = false;
          syncButton();
        } finally {
          button.disabled = false;
        }
      });
      row.append(button, label);
      card.append(row);
    }
    return card;
  };

  const renderWebResearchCard = (data) => {
    if (!data || data.type !== 'web_research' || typeof data.query !== 'string' || !Array.isArray(data.sources) || !data.sources.length || data.sources.length > 3) return null;
    const card = document.createElement('section');
    card.className = 'artifact web-research-card';
    const heading = document.createElement('div');
    heading.className = 'web-research-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c2 2.2 3 4.9 3 8s-1 5.8-3 8c-2-2.2-3-4.9-3-8s1-5.8 3-8"/></svg>';
    const title = document.createElement('strong');
    title.textContent = `Research: ${data.query}`;
    heading.append(title);
    card.append(heading);
    for (const source of data.sources) {
      if (typeof source?.url !== 'string' || typeof source?.host !== 'string' || typeof source?.evidence !== 'string') continue;
      let url;
      try { url = new URL(source.url); } catch { continue; }
      if (!['http:', 'https:'].includes(url.protocol)) continue;
      const item = document.createElement('details');
      item.className = 'web-source';
      const summary = document.createElement('summary');
      const link = document.createElement('a');
      link.href = url.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = source.host;
      link.addEventListener('click', event => event.stopPropagation());
      summary.append(link);
      const evidence = document.createElement('p');
      evidence.textContent = source.evidence;
      item.append(summary, evidence);
      card.append(item);
    }
    return card;
  };

  const renderGeneratedReportCard = (data) => {
    if (!data || data.type !== 'generated_report' || typeof data.title !== 'string' || typeof data.summary !== 'string' || !Number.isFinite(data.created_at) || typeof data.report_url !== 'string' || !/^\/reports\/fulloch-reports\/\d{4}-\d{2}-\d{2}-[0-9a-f]{8}$/.test(data.report_url)) return null;
    const card = document.createElement('section');
    card.className = 'artifact generated-report-card';
    const heading = document.createElement('div');
    heading.className = 'generated-report-heading';
    heading.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h11l3 3v15H5zM16 3v4h4M8 12h8m-8 4h6"/></svg>';
    const title = document.createElement('strong');
    title.textContent = data.title;
    heading.append(title);
    card.append(heading);
    const created = document.createElement('time');
    created.dateTime = new Date(data.created_at * 1000).toISOString();
    created.textContent = `Created ${new Date(data.created_at * 1000).toLocaleString()}`;
    const summary = document.createElement('p');
    summary.textContent = data.summary;
    const download = document.createElement('a');
    download.href = data.report_url;
    download.target = '_blank';
    download.rel = 'noopener';
    download.textContent = 'Open full report';
    card.append(created, summary, download);
    return card;
  };

  const flightDuration = (minutes) => Number.isFinite(minutes)
    ? `${Math.floor(minutes / 60)}h ${minutes % 60}m`
    : 'Duration unavailable';

  const renderFlightPlanCard = (data) => {
    const offer = data?.offer;
    if (!offer || !data.route) return null;
    const card = document.createElement('section');
    card.className = 'artifact flight-plan-card';
    const price = typeof offer.price === 'number'
      ? `${data.currency || ''} ${offer.price.toLocaleString()}`.trim()
      : String(offer.price || 'Price unavailable');
    const route = `${data.route.origin || '?'} to ${data.route.destination || '?'}`;
    const airlines = Array.isArray(offer.airlines) && offer.airlines.length
      ? offer.airlines.join(', ') : 'Carrier unavailable';
    const stops = Number.isFinite(offer.stops) ? (offer.stops ? `${offer.stops} stop${offer.stops === 1 ? '' : 's'}` : 'Nonstop') : 'Stops unavailable';
    const details = Array.isArray(offer.segments) ? offer.segments : [];
    card.innerHTML = `<div class="flight-plan-kicker"><span>Recommended flight</span><span>Prices can change</span></div><div class="flight-plan-route"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 13.5h6l3.5 6 1.5-.5-1.5-5.5H20a1.5 1.5 0 0 0 0-3h-7.5L14 5l-1.5-.5L9 10.5H3z"/></svg><strong>${escapeHtml(route)}</strong><b>${escapeHtml(price)}</b></div><div class="flight-plan-meta"><span>${escapeHtml(airlines)}</span><span>${escapeHtml(flightDuration(offer.duration_minutes))}</span><span>${escapeHtml(stops)}</span></div>`;
    if (details.length) {
      const detail = document.createElement('details');
      detail.className = 'flight-plan-legs';
      const summary = document.createElement('summary');
      summary.textContent = `Flight details (${details.length} leg${details.length === 1 ? '' : 's'})`;
      detail.append(summary);
      for (const leg of details) {
        const row = document.createElement('div');
        row.textContent = `${leg.departure?.id || '?'} ${leg.departure?.time || ''} -> ${leg.arrival?.id || '?'} ${leg.arrival?.time || ''} | ${leg.airline || 'Carrier unavailable'}`;
        detail.append(row);
      }
      card.append(detail);
    }
    if (typeof data.report_url === 'string' && data.report_url.startsWith('/reports/')) {
      const report = document.createElement('a');
      report.className = 'flight-report-link';
      report.href = data.report_url;
      report.target = '_blank';
      report.rel = 'noopener';
      report.textContent = 'Open full report';
      card.append(report);
    }
    return card;
  };

  const renderArtifacts = (artifacts) => artifacts.map(artifact => {
    if (artifact?.type === 'weather') return renderWeatherCard(artifact);
    if (artifact?.type === 'entity_status') return renderEntityStatusCard(artifact);
    if (artifact?.type === 'temperature_history') return renderTemperatureHistoryCard(artifact);
    if (artifact?.type === 'light_history') return renderLightHistoryCard(artifact);
    if (artifact?.type === 'media') return renderMediaCard(artifact);
    if (artifact?.type === 'calendar') return renderCalendarCard(artifact);
    if (artifact?.type === 'timers') return renderTimersCard(artifact);
    if (artifact?.type === 'home_overview') return renderHomeOverviewCard(artifact);
    if (artifact?.type === 'energy') return renderEnergyCard(artifact);
    if (artifact?.type === 'security') return renderSecurityCard(artifact);
    if (artifact?.type === 'note') return renderNoteCard(artifact);
    if (artifact?.type === 'notes_search') return renderNotesSearchCard(artifact);
    if (artifact?.type === 'todos') return renderTodosCard(artifact);
    if (artifact?.type === 'web_research') return renderWebResearchCard(artifact);
    if (artifact?.type === 'generated_report') return renderGeneratedReportCard(artifact);
    if (artifact?.type === 'flight_plan') return renderFlightPlanCard(artifact);
    return null;
  }).filter(Boolean);

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
    if (ev.role === 'thinking') {
      renderThinkingJob({ ...thinkingJob, ...ev });
      pollStatus();
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
    const animate = ev.role === 'assistant' && live && !!text;
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
    satellites: document.getElementById('satellites'),
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
  const obsidianPluginInfo = document.getElementById('obsidian-plugin-info');
  const obsidianPluginInfoDismiss = document.getElementById('obsidian-plugin-info-dismiss');
  const obsPathWarning = document.getElementById('obs-path-warning');
  const OBSIDIAN_PLUGIN_INFO_DISMISS_KEY = 'fulloch.obsidian_plugin_info_dismissed_v1';
  let obsState = null;

  if (localStorage.getItem(OBSIDIAN_PLUGIN_INFO_DISMISS_KEY) === '1') {
    obsidianPluginInfo.hidden = true;
  }
  obsidianPluginInfoDismiss.addEventListener('click', () => {
    obsidianPluginInfo.hidden = true;
    try { localStorage.setItem(OBSIDIAN_PLUGIN_INFO_DISMISS_KEY, '1'); } catch (_) { /* private mode */ }
  });

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
    const pathNavigationMismatch = !!state.path_navigation_mismatch;
    obsPathWarning.hidden = !pathNavigationMismatch;
    if (pathNavigationMismatch) {
      obsPathWarning.textContent = 'Docker path translation detected: live context and editing work, but Fulloch cannot automatically open notes it writes in Obsidian because the host and container vault paths differ.';
    }
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
    settingsBtn.addEventListener('click', () => { location.href = '/setup?section=notes'; });
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

  // ---- Satellites workspace ----
  const satellitesStage = document.getElementById('satellites-stage');

  const satelliteIcon = (satellite) => {
    const name = `${satellite.label || ''} ${satellite.area || ''}`.toLowerCase();
    const rooms = [
      [/bed(room)?|nursery|guest/, '🛏️'],
      [/living|lounge|family|media|theat(er|re)/, '🛋️'],
      [/kitchen|pantry|scullery/, '🍳'],
      [/dining|breakfast/, '🍽️'],
      [/office|study|desk|workshop/, '💻'],
      [/bath(room)?|ensuite|toilet|powder/, '🛁'],
      [/garage|carport/, '🚗'],
      [/garden|yard|patio|deck|outdoor|porch|veranda/, '🌿'],
      [/laundry|utility/, '🧺'],
      [/hall|entry|foyer|corridor/, '🚪'],
      [/gym|fitness/, '🏋️'],
    ];
    const match = rooms.find(([pattern]) => pattern.test(name));
    return match ? match[1] : satellite.transport === 'native' ? '📡' : '🎙️';
  };

  const satelliteButton = (label, className, action) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = className;
    button.textContent = label;
    button.addEventListener('click', action);
    return button;
  };

  const setSatelliteSetting = async (satellite, setting, enabled) => {
    const route = setting === 'conversation'
      ? `/satellites/${encodeURIComponent(satellite.id)}/conversation-mode`
      : `/satellites/${encodeURIComponent(satellite.id)}/mute`;
    if (setting === 'conversation' && enabled && !confirm(
      'Start conversation mode here?\n\nOther connected voice satellites will be disconnected.'
    )) return;
    try {
      const r = await fetch(route, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      const body = await r.json();
      if (!r.ok || !body.ok) throw new Error(body.message || 'Satellite unavailable');
      loadSatellites();
    } catch (error) {
      console.warn('satellite setting failed', error);
      loadSatellites('Could not update that satellite.');
    }
  };

  const stopSatellite = async (satellite) => {
    try {
      await fetch('/stop', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ satellite_id: satellite.id }),
      });
    } finally {
      loadSatellites();
    }
  };

  const renderSatellite = (satellite, index) => {
    const card = document.createElement('article');
    card.className = `satellite-card ${satellite.transport} ${satellite.conversation_owner ? 'conversation-owner' : ''}`;
    card.style.setProperty('--sat-delay', `${index * 55}ms`);
    const beacon = document.createElement('div');
    beacon.className = 'satellite-beacon';
    beacon.textContent = satelliteIcon(satellite);
    const head = document.createElement('div');
    head.className = 'satellite-card-head';
    const title = document.createElement('div');
    title.className = 'satellite-title';
    const name = document.createElement('h2');
    name.textContent = satellite.label;
    const meta = document.createElement('p');
    meta.textContent = satellite.transport === 'native'
      ? (satellite.area || 'Native room satellite')
      : (satellite.area || 'Dashboard voice satellite');
    title.append(name, meta);
    const state = document.createElement('span');
    state.className = `satellite-state ${satellite.state}`;
    state.textContent = satellite.muted ? 'Listening paused' : satellite.conversation_owner ? 'In conversation' : satellite.state;
    head.append(beacon, title, state);

    const orbit = document.createElement('div');
    orbit.className = 'satellite-orbit';
    const mode = document.createElement('span');
    mode.textContent = satellite.conversation_owner ? 'Wakeword-free room' : 'Wakeword gated';
    const type = document.createElement('span');
    type.textContent = satellite.transport === 'native' ? 'Native satellite' : 'Browser satellite';
    orbit.append(mode, type);

    const controls = document.createElement('div');
    controls.className = 'satellite-controls';
    controls.append(
      satelliteButton(
        satellite.conversation_mode ? 'End conversation' : 'Start conversation',
        `satellite-action ${satellite.conversation_mode ? 'active' : ''}`,
        () => setSatelliteSetting(satellite, 'conversation', !satellite.conversation_mode),
      ),
      satelliteButton(
        satellite.muted ? 'Resume listening' : 'Pause listening',
        'satellite-action quiet',
        () => setSatelliteSetting(satellite, 'mute', !satellite.muted),
      ),
    );
    if (satellite.state !== 'idle') {
      controls.append(satelliteButton('Stop', 'satellite-action stop', () => stopSatellite(satellite)));
    }

    const details = document.createElement('details');
    details.className = 'satellite-details';
    const summary = document.createElement('summary');
    summary.textContent = 'Satellite details';
    const detailGrid = document.createElement('div');
    detailGrid.className = 'satellite-detail-grid';
    const detail = (label, value) => {
      const cell = document.createElement('div');
      const key = document.createElement('span');
      key.textContent = label;
      const val = document.createElement('code');
      val.textContent = value;
      cell.append(key, val);
      return cell;
    };
    detailGrid.append(
      detail('Transport', satellite.transport),
      detail('Endpointing', satellite.server_vad ? 'Server VAD' : 'Device VAD'),
      detail('Session', satellite.id.slice(0, 12)),
    );
    if (satellite.device_id) detailGrid.append(detail('Device', satellite.device_id));
    details.append(summary, detailGrid);
    card.append(head, orbit, controls, details);
    return card;
  };

  const renderSatellites = (state, error = '') => {
    satellitesStage.innerHTML = '';
    const satellites = state.satellites || [];
    const masthead = document.createElement('section');
    masthead.className = 'satellite-masthead';
    const eyebrow = document.createElement('p');
    eyebrow.textContent = satellites.length ? `${satellites.length} room${satellites.length === 1 ? '' : 's'} online` : 'No rooms online';
    const heading = document.createElement('h1');
    heading.textContent = satellites.length ? 'Your home, in listening range.' : 'The house is quiet.';
    const copy = document.createElement('p');
    copy.className = 'satellite-masthead-copy';
    copy.textContent = satellites.length
      ? 'Conversation mode belongs to one room at a time. Pause listening without disconnecting a room.'
      : 'Open Voice mode in the dashboard or connect a native satellite to bring a room online.';
    const refresh = satelliteButton('Refresh rooms', 'satellite-refresh', () => loadSatellites());
    masthead.append(eyebrow, heading, copy, refresh);
    satellitesStage.append(masthead);
    if (error) {
      const message = document.createElement('p');
      message.className = 'satellite-error';
      message.textContent = error;
      satellitesStage.append(message);
    }
    const map = document.createElement('div');
    map.className = 'satellite-map';
    if (satellites.length) satellites.forEach((satellite, index) => map.append(renderSatellite(satellite, index)));
    else {
      const empty = document.createElement('div');
      empty.className = 'satellite-empty';
      empty.textContent = 'No connected satellites yet.';
      map.append(empty);
    }
    satellitesStage.append(map);
  };

  const loadSatellites = async (error = '') => {
    try {
      const r = await fetch('/satellites');
      if (!r.ok) throw new Error(`status ${r.status}`);
      renderSatellites(await r.json(), error);
    } catch (e) {
      console.warn('satellites load failed', e);
      renderSatellites({ satellites: [] }, error || 'Could not reach the satellite service.');
    }
  };

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
    if (name === 'satellites') loadSatellites();
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
      // + greeting done). A restart can return it to setup mode while this page
      // remains open, so force a fresh document request for the wizard.
      if (s.phase === 'NEEDS_SETUP') { location.reload(); return; }
      if (s.phase && s.phase !== 'READY') { location.href = '/'; return; }
      const busy = s.state !== 'idle';
      if (busy !== voiceBusy) { voiceBusy = busy; syncButton(); }
      applyBranding(!!s.remote_llm);
      setLlmUnreachable(!!s.llm_unreachable);
      setSatelliteBusy(s.active_owner_id, s.active_owner_label);
      renderThinkingJob(s.thinking_job);
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

  const satDisconnect = (clearConversationMode = true) => {
    if (satHeartbeatTimer !== null) { clearInterval(satHeartbeatTimer); satHeartbeatTimer = null; }
    if (satWorkletNode) { try { satWorkletNode.disconnect(); } catch(_) {} satWorkletNode = null; }
    if (satMicStream) { satMicStream.getTracks().forEach(t => t.stop()); satMicStream = null; }
    if (satWs) { try { satWs.close(); } catch(_) {} satWs = null; }
    if (satAudioCtx) { try { satAudioCtx.close(); } catch(_) {} satAudioCtx = null; }
    satPlayAt = 0;
    satScheduledSources = [];
    satPendingPcm = [];
    satPendingSamples = 0;
    satTtsActive = false;
    satMicMuted = false;
    if (satMicResumeTimer !== null) { clearTimeout(satMicResumeTimer); satMicResumeTimer = null; }
    satHalfDuplex = true;
    satPlaybackOnly = false;
    mySatelliteId = null;
    satPendingConversationMode = null;
    if (clearConversationMode) setConversationModePreference(false);
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
      satHeartbeatTimer = setInterval(() => {
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
    try {
      return await connected;
    } finally {
      satConnecting = false;
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
    if (areas.length) renderAreaPicker(areas);
  };

  satAreaPill.addEventListener('click', async () => {
    const areas = await fetchHaAreas();
    if (areas.length) renderAreaPicker(areas);
  });

  satBtn.addEventListener('click', () => {
    if (satWs || satConnecting) {
      satDisconnect();
      return;
    }
    // Voice mode alone must never restore a previous Conversation mode.
    setConversationModePreference(false);
    satConnect();
  });
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

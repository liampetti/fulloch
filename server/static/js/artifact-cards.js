import { escapeHtml } from './browser-utils.js';
import { createLifetime } from './browser-lifetime.js';

export function createArtifactCards({ submitCommand }) {
  const lifetime = createLifetime();
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
        await submitCommand(`${action} ${data.player}`, button);
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
          const request = action === 'extend' ? `extend timer ${timer.id} by 1 minute` : `cancel timer ${timer.id}`;
          await submitCommand(request, button);
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
        if (!remaining && interval) lifetime.clearInterval(interval);
      };
      update();
      interval = lifetime.setInterval(update, 1000);
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
        await submitCommand(`complete ${item}`, button);
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

  const validReportUrl = value => typeof value === 'string' && /^\/reports\/fulloch-reports\/\d{4}-\d{2}-\d{2}-[0-9a-f]{8}$/.test(value);
  const appendReportLink = (card, url) => {
    if (!validReportUrl(url)) return;
    const link = document.createElement('a');
    link.className = 'report-card-link';
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = 'Open full report';
    card.append(link);
  };
  const appendSourceLink = (card, url) => {
    let source;
    try { source = new URL(url); } catch { return; }
    if (!['http:', 'https:'].includes(source.protocol)) return;
    const link = document.createElement('a');
    link.className = 'report-card-source';
    link.href = source.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Market source';
    card.append(link);
  };
  const appendSparkline = (card, points) => {
    const values = Array.isArray(points) ? points.filter(point => Number.isFinite(point?.value)).slice(0, 48) : [];
    if (values.length < 2) return;
    const low = Math.min(...values.map(point => point.value));
    const high = Math.max(...values.map(point => point.value));
    const range = high - low || 1;
    const chart = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    chart.classList.add('finance-sparkline');
    chart.setAttribute('viewBox', '0 0 300 62');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', `Price range ${low} to ${high}`);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', values.map((point, index) => {
      const x = index / (values.length - 1) * 288 + 6;
      const y = 54 - (point.value - low) / range * 46;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' '));
    chart.append(line);
    card.append(chart);
  };
  const appendFinanceQuote = (card, quote, chart = true) => {
    if (!quote || typeof quote.name !== 'string' || typeof quote.price !== 'string') return;
    const quoteLine = document.createElement('div');
    quoteLine.className = 'finance-quote-line';
    const identity = document.createElement('span');
    identity.textContent = [quote.name, quote.symbol].filter(value => typeof value === 'string' && value).join(' · ');
    const price = document.createElement('strong');
    price.textContent = [quote.price, quote.currency, quote.change, quote.change_percent].filter(value => typeof value === 'string' && value).join(' ');
    quoteLine.append(identity, price);
    card.append(quoteLine);
    if (chart) appendSparkline(card, quote.points);
    const freshness = document.createElement('small');
    freshness.textContent = [quote.exchange, quote.quoted_at && `Quoted ${quote.quoted_at}`, quote.retrieved_at && `Retrieved ${quote.retrieved_at}`, quote.cache_mode].filter(Boolean).join(' · ');
    if (freshness.textContent) card.append(freshness);
    appendSourceLink(card, quote.source_url);
  };
  const renderCurrencyConversionCard = (data) => {
    const rate = data?.exchange_rate;
    if (data?.type !== 'finance_exchange_rate' || !rate || typeof rate.base_currency !== 'string' || typeof rate.quote_currency !== 'string' || !Number.isFinite(rate.rate)) return null;
    const card = document.createElement('section');
    card.className = 'artifact finance-card currency-conversion-card';
    card.innerHTML = '<div class="finance-card-heading"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7h11m0 0-3-3m3 3-3 3M17 17H6m0 0 3 3m-3-3 3-3"/></svg><strong>Currency Conversion</strong></div>';
    const rateLine = document.createElement('div');
    rateLine.className = 'finance-rate';
    rateLine.textContent = `1 ${rate.base_currency} = ${rate.rate} ${rate.quote_currency}`;
    card.append(rateLine);
    if (Number.isFinite(rate.amount) && Number.isFinite(rate.converted_amount)) {
      const conversion = document.createElement('p');
      conversion.textContent = `${rate.amount} ${rate.base_currency} = approximately ${rate.converted_amount} ${rate.quote_currency}`;
      card.append(conversion);
    }
    const freshness = document.createElement('small');
    freshness.textContent = [rate.quoted_at && `Quoted ${rate.quoted_at}`, rate.retrieved_at && `Retrieved ${rate.retrieved_at}`, rate.cache_mode].filter(Boolean).join(' · ');
    if (freshness.textContent) card.append(freshness);
    appendSourceLink(card, rate.source_url);
    return card;
  };
  const renderStockQuoteCard = (data) => {
    if (data?.type !== 'finance_quote' || !data.quote) return null;
    const card = document.createElement('section');
    card.className = 'artifact finance-card stock-quote-card';
    card.innerHTML = '<div class="finance-card-heading"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 17 9 12l3 3 7-8M14 7h5v5"/></svg><strong>Stock Quote</strong></div>';
    appendFinanceQuote(card, data.quote);
    return card;
  };
  const renderDomainReportCard = (data, type, icon) => {
    if (!data || data.type !== type || typeof data.title !== 'string' || typeof data.summary !== 'string' || !Number.isFinite(data.created_at)) return null;
    const card = document.createElement('section');
    card.className = `artifact domain-report-card ${type}`;
    const heading = document.createElement('div');
    heading.className = 'domain-report-heading';
    heading.innerHTML = icon;
    const title = document.createElement('strong');
    title.textContent = data.title;
    heading.append(title);
    card.append(heading);
    const summary = document.createElement('p');
    summary.textContent = data.summary;
    card.append(summary);
    const source = data.data;
    if (type === 'finance_report') {
       if (source?.type === 'finance_quote') appendFinanceQuote(card, source.quote);
       if (source?.type === 'finance_summary' && Array.isArray(source.quotes)) {
         for (const quote of source.quotes.slice(0, 4)) appendFinanceQuote(card, quote);
       }
      if (source?.type === 'finance_exchange_rate') {
        const rate = source.exchange_rate;
        if (rate && typeof rate.base_currency === 'string' && typeof rate.quote_currency === 'string' && Number.isFinite(rate.rate)) {
          const item = document.createElement('div');
          item.className = 'domain-report-row';
          item.textContent = `1 ${rate.base_currency} = ${rate.rate} ${rate.quote_currency}${Number.isFinite(rate.amount) && Number.isFinite(rate.converted_amount) ? ` · ${rate.amount} = ${rate.converted_amount}` : ''}`;
          card.append(item);
          const freshness = document.createElement('small');
          freshness.textContent = [rate.quoted_at && `Quoted ${rate.quoted_at}`, rate.retrieved_at && `Retrieved ${rate.retrieved_at}`, rate.cache_mode].filter(Boolean).join(' · ');
          if (freshness.textContent) card.append(freshness);
          appendSourceLink(card, rate.source_url);
        }
      }
       const rows = source?.type === 'finance_watchlist'
         ? source.quotes
         : source?.type === 'finance_market'
           ? source.markets
           : source?.type === 'finance_summary' ? source.markets : [];
      if (Array.isArray(rows)) for (const row of rows.slice(0, 4)) {
        if (typeof row?.name !== 'string') continue;
        const item = document.createElement('div');
        item.className = 'domain-report-row';
        item.textContent = [row.name, row.price, row.currency, row.change, row.change_percent].filter(value => typeof value === 'string' && value).join(' · ');
        card.append(item);
      }
       const retrievedAt = source?.type === 'finance_watchlist'
         ? source.quotes?.[0]?.retrieved_at
         : source?.type === 'finance_market' ? source.retrieved_at : source?.quotes?.[0]?.retrieved_at;
      if (typeof retrievedAt === 'string') {
        const freshness = document.createElement('small');
        freshness.textContent = `Retrieved ${retrievedAt}`;
        card.append(freshness);
      }
    } else if (type === 'research_report') {
      const papers = source?.type === 'paper_search' ? source.papers : source?.type === 'paper_detail' ? [source.paper] : [];
      if (Array.isArray(papers)) for (const paper of papers.slice(0, 3)) {
        if (typeof paper?.title !== 'string') continue;
        const item = document.createElement('div');
        item.className = 'domain-report-row';
        item.textContent = [paper.title, paper.year && String(paper.year), paper.source].filter(Boolean).join(' · ');
        card.append(item);
      }
      if (source?.type === 'web_research' && Array.isArray(source.sources)) {
        const item = document.createElement('div');
        item.className = 'domain-report-row';
        item.textContent = `${source.sources.slice(0, 3).filter(source => typeof source?.url === 'string').length} web sources retrieved`;
        card.append(item);
      }
    } else if (type === 'travel_report') {
      const route = source?.route;
      if (route?.origin || route?.destination) {
        const item = document.createElement('div');
        item.className = 'domain-report-row';
        item.textContent = `${route.origin || '?'} to ${route.destination || '?'}`;
        card.append(item);
      } else if (Array.isArray(source?.legs)) {
        const item = document.createElement('div');
        item.className = 'domain-report-row';
        item.textContent = `${source.legs.length} itinerary leg${source.legs.length === 1 ? '' : 's'} retrieved`;
        card.append(item);
      }
    }
    appendReportLink(card, data.report_url);
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
    if (artifact?.type === 'finance_exchange_rate') return renderCurrencyConversionCard(artifact);
    if (artifact?.type === 'finance_quote') return renderStockQuoteCard(artifact);
    if (artifact?.type === 'research_report') return renderDomainReportCard(artifact, 'research_report', '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h12a2 2 0 0 1 2 2v14l-4-2-4 2-4-2-2 1Z"/><path d="M8 9h8m-8 4h6"/></svg>');
    if (artifact?.type === 'travel_report') return renderDomainReportCard(artifact, 'travel_report', '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 13.5h6l3.5 6 1.5-.5-1.5-5.5H20a1.5 1.5 0 0 0 0-3h-7.5L14 5l-1.5-.5L9 10.5H3z"/></svg>');
    if (artifact?.type === 'finance_report') return renderDomainReportCard(artifact, 'finance_report', '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 17 9 12l3 3 7-8M14 7h5v5"/></svg>');
    if (artifact?.type === 'generated_report') return renderGeneratedReportCard(artifact);
    if (artifact?.type === 'flight_plan') return renderFlightPlanCard(artifact);
    return null;
  }).filter(Boolean);

  return { renderArtifacts, clear: () => lifetime.clearTimers(), destroy: () => lifetime.destroy() };
}

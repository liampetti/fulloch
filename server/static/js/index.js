import { escapeHtml } from './browser-utils.js';
import { createLifetime } from './browser-lifetime.js';
import { createChatUI } from './chat-ui.js';
import { createBrowserSatellite } from './browser-satellite.js';
import { createBrandLogo } from './brand-logo.js';

const lifetime = createLifetime();
const brandLogo = createBrandLogo(document.getElementById('brand-logo'));
const chatUI = createChatUI({
  connectSpeaker: () => satellite.connect(true),
  getChatContext: () => satellite.chatContext,
  onThinking: ev => {
    const id = ev.job_id || ev.id;
    renderThinkingJob({ ...(thinkingJob?.id === id ? thinkingJob : {}), ...ev, id });
    pollStatus();
  },
});
const satellite = createBrowserSatellite({
  clearEmpty: chatUI.clearEmpty, scrollEnd: chatUI.scrollEnd,
  onPlaybackChange: brandLogo.setTalking,
});
  // Session cookie auth — the browser sends the cookie automatically on every
  // fetch and WebSocket upgrade. On 401 (session expired after server restart)
  // redirect to /login so the user can re-authenticate.
  const _origFetch = window.fetch.bind(window);
  window.fetch = (url, opts = {}) => _origFetch(url, opts).then((r) => {
    if (r.status === 401) location.href = '/login';
    return r;
  });
  // Busy-status banner: another satellite or browser session is mid-turn.
  // Typed chat shares the server's ``dashboard-text`` pseudo-session, while
  // this tab only has a satellite ID after Voice mode connects. Treat this
  // tab's in-flight typed request as local too; otherwise its own request is
  // incorrectly presented as work happening in another room.
  const busyBanner = document.getElementById('busy-banner');
  const busyBannerText = document.getElementById('busy-banner-text');
  const thinkingJobCard = document.getElementById('thinking-job-card');
  const thinkingJobTask = document.getElementById('thinking-job-task');
  const thinkingJobStage = document.getElementById('thinking-job-stage');
  const thinkingJobOutcome = document.getElementById('thinking-job-outcome');
  const thinkingJobReport = document.getElementById('thinking-job-report');
  const thinkingJobCancel = document.getElementById('thinking-job-cancel');
  const thinkingJobDetails = document.getElementById('thinking-job-details');
  const thinkingJobDiagnostics = document.getElementById('thinking-job-diagnostics');
  let thinkingJob = null;
  let thinkingReceivedAt = 0;
  let thinkingOutcomeTimer = null;
  const terminalThinkingJob = job => ['READY', 'FAILED', 'CANCELLED'].includes(job?.status);
  const renderThinkingOutcome = () => {
    if (!thinkingJob || !thinkingJobOutcome) return;
    const terminal = terminalThinkingJob(thinkingJob);
    const rawDuration = thinkingJob.job_duration_seconds ?? thinkingJob.elapsed_seconds;
    const duration = rawDuration == null ? NaN : Number(rawDuration);
    const extra = thinkingJob.status === 'RUNNING' ? (Date.now() - thinkingReceivedAt) / 1000 : 0;
    const seconds = Math.floor(Math.max(0, duration + extra));
    const durationText = Number.isFinite(duration)
      ? `${terminal ? 'Duration' : 'Elapsed'}: ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}` : '';
    const progress = thinkingJob.progress || {};
    const processed = Number(progress.processed_unique_documents) || 0;
    const fetched = Number(progress.fetched_documents) || 0;
    const saved = Number(progress.saved_sections) || 0;
    const sources = processed || fetched;
    const sourceText = sources ? `${sources} source${sources === 1 ? '' : 's'} ${processed ? 'reviewed' : 'collected'}` : '';
    const sectionText = saved ? `${saved} section${saved === 1 ? '' : 's'} saved` : '';
    thinkingJobOutcome.textContent = [durationText, sourceText, sectionText].filter(Boolean).join(' · ');
    thinkingJobOutcome.hidden = !thinkingJobOutcome.textContent;
  };
  const renderThinkingJob = (job) => {
    const changedJob = thinkingJob?.id !== job?.id;
    thinkingJob = job || null;
    thinkingReceivedAt = Date.now();
    if (thinkingJob?.status === 'RUNNING' && thinkingOutcomeTimer === null) {
      thinkingOutcomeTimer = lifetime.setInterval(renderThinkingOutcome, 1000);
    } else if (thinkingJob?.status !== 'RUNNING' && thinkingOutcomeTimer !== null) {
      lifetime.clearInterval(thinkingOutcomeTimer);
      thinkingOutcomeTimer = null;
    }
    if (!thinkingJobCard) return;
    thinkingJobCard.hidden = !thinkingJob;
    if (!thinkingJob) return;
    thinkingJobCard.dataset.status = thinkingJob.status || '';
    if (changedJob && thinkingJobDetails) thinkingJobDetails.open = false;
    if (thinkingJobTask) thinkingJobTask.textContent = thinkingJob.objective || thinkingJob.task || 'Deliberate work';
    const statusLabels = { QUEUED: 'Queued', PAUSED: 'Paused', NEEDS_INPUT: 'Waiting for your answer',
      FAILED: 'Couldn’t complete the report', CANCELLED: 'Cancelled' };
    if (thinkingJobStage) thinkingJobStage.textContent = statusLabels[thinkingJob.status]
      || thinkingJob.stage || (thinkingJob.status === 'READY' ? 'Report ready' : 'Working');
    renderThinkingOutcome();
    if (thinkingJobDiagnostics) {
      const terminal = terminalThinkingJob(thinkingJob);
      const duration = Number(thinkingJob.elapsed_seconds);
      const durationText = Number.isFinite(duration)
        ? `${terminal ? 'Research duration' : 'Elapsed'}: ${duration.toFixed(1)}s` : '';
      const reason = terminal && !['completed', 'complete', 'ready'].includes(thinkingJob.terminal_reason)
        ? String(thinkingJob.terminal_reason || '') : '';
      const progress = thinkingJob.progress && typeof thinkingJob.progress === 'object'
        ? thinkingJob.progress : {};
      const count = (key) => Number.isFinite(Number(progress[key])) ? Number(progress[key]) : 0;
      const progressText = [
        `${count('processed_unique_documents')} processed · ${count('fetched_documents')} fetched · ${count('discovered_documents')} discovered`,
        `${count('cited_documents')} cited`,
        `${count('saved_sections')} sections saved`,
        `${Array.isArray(progress.unresolved_obligations) ? progress.unresolved_obligations.length : 0} open questions`,
      ].join(' · ');
      const ack = thinkingJob.acknowledgement_seconds == null ? NaN : Number(thinkingJob.acknowledgement_seconds);
      const jobDuration = Number(thinkingJob.job_duration_seconds);
      const lifecycleText = Number.isFinite(ack)
        ? `Acknowledged in ${ack.toFixed(3)}s · job ${Number.isFinite(jobDuration) ? jobDuration.toFixed(1) : '0.0'}s` : durationText;
      const requests = Array.isArray(thinkingJob.request_metrics) ? thinkingJob.request_metrics : [];
      const latest = requests.length ? requests[requests.length - 1] : null;
      const requestText = latest
        ? `Request ${latest.input_tokens} in / ${latest.output_reservation} reserved out${latest.measured_prefill_seconds != null || latest.measured_decode_seconds != null
          ? ` · prefill ${latest.measured_prefill_seconds ?? 'n/a'}s / decode ${latest.measured_decode_seconds ?? 'n/a'}s`
          : ' · prefill/decode timing unavailable'}`
        : '';
      const error = thinkingJob.error || reason;
      thinkingJobDiagnostics.textContent = [thinkingJob.task, lifecycleText, progressText, requestText, error]
        .filter(Boolean).join('\n');
    }
    if (thinkingJobReport) {
      const link = String(thinkingJob.partial_artifact_link || '');
      const valid = link.startsWith('/reports/') && !link.startsWith('//');
      thinkingJobReport.hidden = !valid;
      if (valid) thinkingJobReport.href = link;
      else thinkingJobReport.removeAttribute('href');
    }
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
    const ownTextTurn = ownerId === 'dashboard-text' && chatUI.isTextTurnActive();
    const showBusy = !!ownerId && ownerId !== satellite.id && !ownTextTurn;
    if (busyBanner) busyBanner.hidden = !showBusy;
    if (showBusy && busyBannerText) {
      busyBannerText.textContent = ownerId === 'dashboard-text'
        ? 'Busy — another browser is chatting'
        : ownerLabel
        ? `Busy — talking to ${ownerLabel}`
        : 'Busy — talking to another room';
    }
  };
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

  // ---- Documents workspace + optional Obsidian integration ----
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
  const documentsList = document.getElementById('documents-list');
  const documentsRefresh = document.getElementById('documents-refresh');
  const documentsSort = document.getElementById('documents-sort');
  const DOCUMENTS_PAGE_SIZE = 25;
  let loadedDocuments = [];
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

  const documentUrl = (path) => '/documents/' + path.split('/').map(encodeURIComponent).join('/');

  const makeDocumentLink = (file) => {
    const link = document.createElement('a');
    link.className = 'document-file';
    link.href = documentUrl(file.path);
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = file.title || file.path.split('/').at(-1).replace(/\.md$/i, '');
    link.title = file.path;
    return link;
  };

  const renderDocuments = (documents) => {
    documentsList.innerHTML = '';
    if (!documents.length) {
      const empty = document.createElement('p');
      empty.className = 'documents-empty';
      empty.textContent = 'No Markdown documents are available in this folder yet.';
      documentsList.appendChild(empty);
      return;
    }
    const root = { folders: new Map(), files: [] };
    for (const file of documents) {
      if (!file || typeof file.path !== 'string' || !file.path.endsWith('.md')) continue;
      const parts = file.path.split('/');
      const filename = parts.pop();
      if (!filename || parts.some(part => !part || part === '.' || part === '..')) continue;
      let current = root;
      for (const folder of parts) {
        if (!current.folders.has(folder)) current.folders.set(folder, { folders: new Map(), files: [] });
        current = current.folders.get(folder);
      }
      current.files.push(file);
    }
    const prepareTree = (node) => {
      node.entries = [
        ...[...node.folders].map(([name, child]) => {
          prepareTree(child);
          return { name, child, modified: child.modified };
        }),
        ...node.files.map(file => ({ name: file.path.split('/').at(-1), file, modified: file.modified_at || 0 })),
      ];
      node.modified = node.entries.reduce((latest, entry) => Math.max(latest, entry.modified), 0);
      node.entries.sort((a, b) => {
        const byName = a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
        return documentsSort.value === 'recent' ? b.modified - a.modified || byName : byName;
      });
    };
    prepareTree(root);
    const appendTree = (container, node, page = 0) => {
      container.replaceChildren();
      const start = page * DOCUMENTS_PAGE_SIZE;
      for (const entry of node.entries.slice(start, start + DOCUMENTS_PAGE_SIZE)) {
        if (entry.file) {
          container.appendChild(makeDocumentLink(entry.file));
          continue;
        }
        const details = document.createElement('details');
        details.className = 'documents-folder';
        const summary = document.createElement('summary');
        summary.textContent = entry.name;
        const children = document.createElement('div');
        children.className = 'documents-folder-children';
        // Populate only when opened, keeping large collapsed trees lightweight.
        details.addEventListener('toggle', () => {
          if (details.open && !children.childElementCount) appendTree(children, entry.child);
        });
        details.append(summary, children);
        container.appendChild(details);
      }
      if (node.entries.length > DOCUMENTS_PAGE_SIZE) {
        const pager = document.createElement('nav');
        pager.className = 'documents-pagination';
        pager.setAttribute('aria-label', 'Document pages');
        const pages = Math.ceil(node.entries.length / DOCUMENTS_PAGE_SIZE);
        const previous = document.createElement('button');
        previous.type = 'button';
        previous.textContent = 'Previous';
        previous.disabled = page === 0;
        const next = document.createElement('button');
        next.type = 'button';
        next.textContent = 'Next';
        next.disabled = page + 1 === pages;
        const status = document.createElement('span');
        status.textContent = `Page ${page + 1} of ${pages} · ${node.entries.length} entries`;
        const changePage = (newPage, direction) => {
          appendTree(container, node, newPage);
          const buttons = container.lastElementChild.querySelectorAll('button');
          const preferred = buttons[direction];
          (preferred.disabled ? buttons[1 - direction] : preferred).focus();
        };
        previous.addEventListener('click', () => changePage(page - 1, 0));
        next.addEventListener('click', () => changePage(page + 1, 1));
        pager.append(previous, status, next);
        container.appendChild(pager);
      }
    };
    appendTree(documentsList, root);
  };

  const loadDocuments = async () => {
    documentsList.innerHTML = '<p class="documents-empty">Loading documents…</p>';
    try {
      const r = await fetch('/api/documents');
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      loadedDocuments = Array.isArray(data.documents) ? data.documents : [];
      renderDocuments(loadedDocuments);
    } catch (e) {
      console.warn('documents load failed', e);
      documentsList.innerHTML = '<p class="documents-empty">Couldn\'t load documents.</p>';
    }
  };
  documentsRefresh.addEventListener('click', loadDocuments);
  documentsSort.addEventListener('change', () => renderDocuments(loadedDocuments));

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
      obsStatusDetail.innerHTML = `Document folder: <code>${escapeObs(state.notes_path || './data/notes')}</code>`;
      renderObsActions(connected);
    } else if (connected) {
      obsPill.textContent = 'Obsidian connected';
      obsPill.classList.add('connected');
      obsStatusDetail.innerHTML = `Document folder: <code>${escapeObs(state.notes_path || './data/notes')}</code>`;
      if (state.indexing_progress != null) {
        const pct = Math.max(0, Math.min(1, state.indexing_progress));
        obsProgress.hidden = false;
        obsProgressBar.style.width = (pct * 100).toFixed(0) + '%';
        obsProgressText.textContent = `Indexing ${(pct * 100).toFixed(0)}%`;
      }
      renderObsActions(connected);
    } else if (vaultPath) {
      obsPill.textContent = 'Obsidian disconnected';
      obsPill.classList.add('disconnected');
      const last = lastConn ? new Date(lastConn * 1000).toLocaleString() : '';
      obsStatusDetail.innerHTML = `Document folder: <code>${escapeObs(state.notes_path || vaultPath)}</code>` +
        (last ? ` <span class="obs-help">— last seen ${escapeObs(last)}</span>` : '');
      renderObsActions(connected);
    } else {
      obsPill.textContent = 'Folder ready';
      obsPill.classList.add('idle');
      obsStatusDetail.innerHTML = `Document folder: <code>${escapeObs(state.notes_path || './data/notes')}</code>`;
      renderObsActions(connected);
    }
  };

  const renderObsActions = (connected) => {
    obsActions.innerHTML = '';
    const settingsBtn = document.createElement('button');
    settingsBtn.className = 'obs-btn';
    settingsBtn.type = 'button';
    settingsBtn.textContent = 'Configure document folder';
    settingsBtn.addEventListener('click', () => { location.href = '/setup?section=notes'; });
    obsActions.appendChild(settingsBtn);
    if (connected && obsState.allow_edit_delete) {
      const indicator = document.createElement('span');
      indicator.className = 'obs-edit-mode';
      indicator.innerHTML = '<i></i>Edit/delete mode active';
      obsActions.appendChild(indicator);
    }
    obsidianHint.innerHTML = connected
      ? 'Obsidian is connected. Its plugin supplies the active note and selected text; with edit/delete enabled in Settings, Fulloch can edit the active note.'
      : 'Fulloch creates, appends, reads, and searches Markdown documents directly in the folder above. Connect Obsidian in Settings only if you want active-note and selected-text context.';
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
    if (name === 'obsidian') { loadObsidian(); loadDocuments(); }
    if (name === 'satellites') loadSatellites();
  };

  tabs.forEach((t) => t.addEventListener('click', () => setTab(t.dataset.tab)));
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
  lifetime.listen(darkMql, 'change', (e) => {
    if (window.FULLOCH_DASHBOARD_PREFS.theme === 'auto') {
      document.documentElement.classList.toggle('dark', e.matches);
      syncThemeMeta();
    }
  });
  // Poll the agent's state so the stop button appears for voice turns (and any
  // work this page didn't start), and clears once playback finishes. Text turns
  // this page started are tracked locally via `waiting` for snappier feedback.
  let statusPollSequence = 0;
  const pollStatus = async () => {
    const sequence = ++statusPollSequence;
    try {
      const r = await fetch('/status');
      if (!r.ok) return;
      const s = await r.json();
      if (sequence !== statusPollSequence || lifetime.destroyed) return;
      // The dashboard is only usable once the assistant is READY (models loaded
      // + greeting done). A restart can return it to setup mode while this page
      // remains open, so force a fresh document request for the wizard.
      if (s.phase === 'NEEDS_SETUP') { location.reload(); return; }
      if (s.phase && s.phase !== 'READY') { location.href = '/'; return; }
      const busy = s.state !== 'idle';
      chatUI.setVoiceBusy(busy);
      applyBranding(!!s.remote_llm);
      setLlmUnreachable(!!s.llm_unreachable);
      setSatelliteBusy(s.active_owner_id, s.active_owner_label);
      if (s.thinking_job) renderThinkingJob(s.thinking_job);
      else if (thinkingJob?.id && (!terminalThinkingJob(thinkingJob)
        || (thinkingJob.status === 'READY' && !thinkingJob.partial_artifact_link))) {
        // Active jobs disappear from /status when they finish. Fetch the last
        // job directly so missed SSE events cannot leave a stale running card.
        const id = thinkingJob.id;
        const result = await fetch(`/thinking/${encodeURIComponent(id)}`);
        const job = result.ok ? await result.json() : null;
        if (sequence !== statusPollSequence || lifetime.destroyed || thinkingJob?.id !== id) return;
        if (job) renderThinkingJob(job);
        else if (result.status === 404) renderThinkingJob(null);
      }
    } catch (e) { /* transient; next tick retries */ }
  };

  // Remote-LLM mode: when the LLM runs off-device (remote OpenAI endpoint) the
  // assistant isn't fully local — update the tagline. Static characters and
  // favicons still use /logo.png; the header uses Fulloch's animated sprite.
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
    statusTimer = lifetime.setInterval(pollStatus, STATUS_POLL_MS);
  };
  const stopStatusPolling = () => {
    if (statusTimer === null) return;
    lifetime.clearInterval(statusTimer);
    statusTimer = null;
  };
  lifetime.listen(document, "visibilitychange", () => {
    if (document.visibilityState === "visible") startStatusPolling();
    else stopStatusPolling();
  });
  if (document.visibilityState === "visible") startStatusPolling();

  // Poll globally so the destructive-edit indicator stays accurate beside the
  // Obsidian tab even while the user is chatting.
  loadObsidian();
  lifetime.setInterval(loadObsidian, 5000);

chatUI.start().then(() => {
  if (!lifetime.destroyed) satellite.showAreaPicker();
});
window.addEventListener('pagehide', () => {
  lifetime.destroy();
  brandLogo.destroy();
  chatUI.destroy();
  satellite.destroy();
}, { once: true });
// A cached document has already disposed its audio and event streams.
window.addEventListener('pageshow', event => { if (event.persisted) location.reload(); });
